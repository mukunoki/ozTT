// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: planning of one run.
//
// Decides, without touching the device and without allocating any device buffer,
//   - how the partial-product GEMMs are issued and accumulated, and the K block length
//     that keeps them error-free (IssuePlan, plan_issue);
//   - how the output is covered with panels and cut into row bands so that the buffers
//     fit the DRAM (PanelPlan, plan_panels);
//   - the constants of the reconstruction (recon_nsmall, recon_scale).
// A plan depends on its arguments only: the configuration RunCfg, the free DRAM and the
// size of the core grid. A configuration that cannot be executed is reported with
// std::runtime_error.
//
// Names used throughout ozaki1/:
//   p        number of slices                 L      last diagonal that is used, p-1
//   nslice   the slices that are used, p = L+1, and also the number of diagonals (ndiag)
//   Mt, Nt   output tiles, M/32 and N/32      Kto    K tiles of one slice, K/32
//   Kc, nkb  K block length [elements] and number of K blocks K/Kc
//   Kbt      K tiles of one K block, Kc/32
//   Ktm      K tiles after slice concatenation, (L+1)*Kto
//   bw       kernel block width: K tiles the GEMM kernel multiplies at a time (divides Kbt)
//   band     row band of the output; the buffer of partial sums holds nslice planes of
//            one band, band_tiles tiles apart

#include "ozaki1/format.hpp"
#include "ozaki1/gemm.hpp"
#include "ozaki1/recon.hpp"
#include "common/bits.hpp"
#include "common/dram_margin.hpp"
#include "common/kernels/f64_tf.h"
#include <fmt/format.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// Problem size and number of slices of one run (filled by run() in src/ozaki1.cpp). The
// values fixed by the format (radix, digit bound, fidelity, normalization) are the
// constants OZ1_* of ozaki1/format.hpp.
struct RunCfg {
    uint32_t M = 0, N = 0, K = 0, Mt = 0, Nt = 0;  // dimensions [elements]; Mt = M/32, Nt = N/32
    int p = 0;                                     // number of slices

    // Last diagonal that is used: only the pairs with i+j <= L = p-1 are used.
    int L() const { return p - 1; }
    // Number of slices, which is also the number of diagonals.
    uint32_t nslice() const { return (uint32_t)p; }
};

// ===== DRAM estimate =====
// Sizes [bytes] of the buffers still to be allocated, for comparison with the DRAM that
// is free before any allocation. Not counted are the partial sums per diagonal (their
// size is what the row bands are chosen for) and the small buffers (out-of-range flags,
// the one-tile limb buffer), which the margin OZ_DRAM_MARGIN_BYTES
// (common/dram_margin.hpp) absorbs.
struct Oz1DramNeed {
    uint64_t operands = 0;  // FP64 operands A and B (8 bytes per element)
    uint64_t slices = 0;    // slice-concatenated A and B (L+1 slices in the GEMM input format)
    uint64_t scale = 0;     // exponent deltas (Int32 tiles) and row/column maxima (BF16 tiles)
    uint64_t output = 0;    // FP64 output
    uint64_t limbs = 0;     // three limb planes of the reconstruction (only if Horner runs per K block;
                            // one band would suffice, the whole output is assumed)
    uint64_t total() const { return operands + slices + scale + output + limbs; }
};
static Oz1DramNeed oz1_dram_need(const RunCfg& c, bool recon_limbs) {
    const uint64_t f32t = tt::tile_size(tt::DataFormat::Float32);
    const uint64_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    const uint64_t in_t = tt::tile_size(OZ_FMT.in);
    const uint64_t f64t = OZ_F64_TILE_BYTES;
    const uint64_t Mt = c.Mt, Nt = c.Nt, Kto = c.K / 32, Ktm = (uint64_t)c.nslice() * Kto;
    Oz1DramNeed d;
    d.operands = f64t * Mt * Kto + f64t * Kto * Nt;
    d.slices = in_t * Mt * Ktm + in_t * Nt * Ktm;
    d.scale = f32t * (Mt + Nt) + bf16t * (Mt + Nt);
    d.output = f64t * Mt * Nt;
    d.limbs = recon_limbs ? 3 * f32t * Mt * Nt : 0u;
    return d;
}

// ===== Issue and accumulation of the partial products =====
// The slice pairs (i, j) of one diagonal d = i + j have the same weight, and the partial
// sum P_d of the diagonal is the sum of their products. The GEMMs are issued in one of
// three ways:
//   concatenated  One GEMM per diagonal over its d+1 pairs concatenated along K. One
//                 accumulation then covers up to (L+1) K_c terms, so K is cut into K
//                 blocks until (L+1) K_c satisfies the exactness condition.
//   pair loop     One GEMM per diagonal that loops over the pairs and folds its FP32
//                 accumulator into an Int32 plane in L1 at every pair boundary. The
//                 exactness condition covers one pair only (K_eff = K_c), the number of
//                 GEMMs stays the number of diagonals, and the partial sums between
//                 pairs never pass through DRAM. Requires that one pair fits a single K
//                 block (with several K blocks, pair and block boundaries would
//                 interleave).
//   pairwise      One GEMM per pair (K_eff = K_c): ndiag(ndiag+1)/2 GEMMs per K block.
// The contributions of the pairs and K blocks of a diagonal are accumulated either
//   in Int32      by the pair loop in L1, or by the GEMM writer, which reads the Int32
//                 partial sum S_d from DRAM, adds and writes it back (fuse). In both
//                 cases the buffer of partial sums is the Int32 accumulation itself, and
//                 the reconstruction runs once at the end (deferred reconstruction); or
//   as limbs      if the Int32 accumulation is not available (Int32 partial products
//                 of INT8, or a sum that could overflow Int32): the issue is
//                 concatenated, and with several K blocks the reconstruction runs
//                 Horner per K block and accumulates the triple-float limbs.
// Horner's rule is linear in P_d, so both give the same value mathematically. In finite
// precision the per-block Horner rounds the triple-float once per K block, so the last
// bit can differ.
struct IssuePlan {
    bool pairloop = false;     // pair loop
    bool pairwise = false;     // one GEMM per pair
    bool fuse = false;         // the GEMM writer accumulates in Int32 in DRAM
    bool defer = false;        // deferred reconstruction: Int32 accumulation, one Horner at the end
    uint32_t Kc = 0, nkb = 0;  // K block length [elements] and number of K blocks

    // True if one accumulation of the GEMM covers a single pair (K_eff = K_c). Only the
    // concatenated issue carries the factor L+1.
    bool per_pair() const { return pairwise || pairloop; }

    // True if the reconstruction keeps three limb planes in DRAM: Horner runs per K
    // block (several K blocks, not deferred) and passes the limbs on to the next block.
    bool recon_limbs() const { return nkb > 1 && !defer; }

    // True for the configuration with a single K block that uses neither the pair loop
    // nor the pairwise issue: per panel, one concatenated GEMM per diagonal, then one
    // reconstruction per band. Every other configuration is run as a sweep (Oz1Sweep in
    // ozaki1/stages.hpp). In practice this is the INT8 configuration: INT8 has no Int32
    // accumulation inside the GEMM and admits a single K block only (issue_feasible),
    // whereas for the FP formats a K that needs no blocking always satisfies the
    // condition of the Int32 accumulation as well, so the pair loop is taken.
    bool single_kblock() const { return nkb == 1 && !pairwise && !pairloop; }

    // GEMMs issued per panel and K block: the number of pairs for the pairwise issue,
    // the number of diagonals otherwise.
    uint32_t gemm_issues(uint32_t ndiag) const { return pairwise ? ndiag * (ndiag + 1) / 2 : ndiag; }
};

// The two candidate K blockings (concatenated: cat, one pair per accumulation: pw) and
// the conditions of the Int32 accumulation.
struct IssueCandidates {
    uint32_t kc_cat = 0, kc_pw = 0;    // error-free K block length (0: none exists)
    uint32_t nkb_cat = 0, nkb_pw = 0;  // number of K blocks
    uint32_t n_cat = 0, n_pw = 0;      // GEMMs per panel of the concatenated / pairwise issue
    bool defer_ok = false;             // the Int32 accumulation cannot overflow (oz_defer_fits)
    bool fuse_ok = false;              // Int32 accumulation inside the GEMM is available
};

static IssueCandidates issue_candidates(const RunCfg& c) {
    IssueCandidates ic;
    const uint32_t K = c.K;
    const int L = c.L();
    const uint32_t ndiag = c.nslice();

    // K block lengths. With one pair per accumulation the factor L+1 does not enter the
    // exactness condition. Either can be 0 (no error-free block length), so it is tested
    // before dividing.
    ic.kc_cat = ozaki_block_k(K, L);
    ic.kc_pw = ozaki_block_k(K, 0);
    ic.nkb_cat = ic.kc_cat ? K / ic.kc_cat : 0u;
    ic.nkb_pw = ic.kc_pw ? K / ic.kc_pw : 0u;

    // Number of GEMMs, which is also the number of plane updates of the accumulation.
    // Without a block length the concatenated count is taken as infinite: with 0, the
    // comparison n_pw < n_cat would always fail and a usable pairwise issue be passed over.
    ic.n_cat = ic.kc_cat ? ndiag * ic.nkb_cat : UINT32_MAX;
    ic.n_pw = ic.kc_pw ? ndiag * (ndiag + 1) / 2 * ic.nkb_pw : 0u;

    // Whatever the issue, the Int32 accumulation of a diagonal sums all its pairs and K
    // blocks, at most (L+1) K dmax^2 in magnitude. Without this test a configuration
    // whose individual GEMMs are exact could still overflow Int32.
    ic.defer_ok = oz_defer_fits(K, L);

    // Accumulation in Int32 inside the GEMM (pair loop or fused writer) requires FP32
    // partial products: the INT8 path does not use it.
    ic.fuse_ok = !OZ1_INT32_PARTIAL && ic.defer_ok;
    return ic;
}

// Chooses the issue. The pair loop is taken whenever one pair fits a single K block.
// Otherwise the pairwise issue is taken if it needs fewer GEMMs than the concatenated
// one: it lengthens the K of each GEMM but raises their number per K block from ndiag to
// ndiag(ndiag+1)/2, which pays only if the K blocking of the concatenated issue raises
// its number even further. Both need the Int32 accumulation inside the GEMM.
static void choose_issue(IssuePlan& ip, const IssueCandidates& ic) {
    ip.pairloop = ic.fuse_ok && ic.kc_pw && ic.nkb_pw == 1u;
    ip.pairwise = !ip.pairloop && ic.fuse_ok && ic.kc_pw && ic.n_pw < ic.n_cat;
}

// Throws std::runtime_error if the chosen issue has no K block length or if the
// reconstruction cannot be carried out.
static void issue_feasible(const RunCfg& c, const IssuePlan& ip) {
    // Only the concatenated issue can end up without a K block length: the pair loop and
    // the pairwise issue are chosen only if kc_pw != 0.
    if (ip.Kc == 0)
        throw std::runtime_error(fmt::format(
            "Ozaki scheme I: the partial products for k={} with {} slices cannot be accumulated without error "
            "in this format (no K block length satisfies the exactness condition). "
            "Use fewer slices (Options::slices) or a smaller k.",
            c.K, c.p));

    // The reconstruction kernel of the INT8 path (recon_tf_i32in.cpp) has no accumulation
    // over K blocks: with several blocks each would silently overwrite the result of the
    // previous one. (With digits +-127 and b_acc = 31, K_c is at most 2^17/(L+1).)
    if (OZ1_INT32_PARTIAL && ip.nkb > 1)
        throw std::runtime_error(
            fmt::format("Ozaki scheme I with Format::INT8 does not support K blocking, but k={} with {} slices needs "
                        "{} K blocks of {} columns. Use a smaller k, fewer slices (Options::slices) or another format.",
                        c.K, c.p, ip.nkb, ip.Kc));

    // FP32 exponent budget of the Horner evaluation (oz_horner_fits). Independent of the
    // exactness of the accumulation; reached with many slices.
    if (!oz_horner_fits(c.K, c.L()))
        throw std::runtime_error(
            fmt::format("Ozaki scheme I: the reconstruction would overflow FP32 for k={} with {} slices (radix {}): "
                        "log2(k) + slices*log2(radix) + 12 must be below 128. "
                        "Use fewer slices (Options::slices) or a smaller k.",
                        c.K, c.p, OZ1_RADIX));
}

// Plans the issue (concatenated, pair loop or pairwise), the accumulation (pair loop,
// fuse or per-block Horner) and the K block length K_c that satisfies the exactness
// condition. Throws std::runtime_error if the configuration cannot be executed.
static IssuePlan plan_issue(const RunCfg& c) {
    IssuePlan ip;
    const IssueCandidates ic = issue_candidates(c);
    choose_issue(ip, ic);
    ip.Kc = ip.per_pair() ? ic.kc_pw : ic.kc_cat;
    ip.nkb = ip.Kc ? c.K / ip.Kc : 0u;

    // Without the pair loop, the GEMM writer accumulates in Int32 whenever a diagonal
    // receives more than one GEMM result: with the pairwise issue, or with several K
    // blocks of the concatenated issue. A single concatenated K block needs no
    // accumulation.
    ip.fuse = !ip.pairloop && ic.fuse_ok && (ip.pairwise || ip.nkb > 1);

    // Deferred reconstruction whenever the GEMMs accumulate in Int32 (the pairwise issue
    // implies fuse). Otherwise Horner runs per K block.
    ip.defer = ip.pairloop || ip.fuse;
    issue_feasible(c, ip);

    // Internal consistency: the block length from ozaki_block_k satisfies the exactness
    // condition by construction. With per_pair one accumulation covers a single pair, so
    // the check uses L = 0.
    TT_FATAL(ozaki_check_exact(ip.Kc, ip.per_pair() ? 0 : c.L()),
             "plan_issue: K block length {} violates the exactness condition (digit bound {}, L={}, per pair {})",
             ip.Kc, OZ1_DMAX, c.L(), ip.per_pair());
    return ip;
}

// ===== Output panels and row bands =====
// A panel is the part of the output that one GEMM program computes; a row band is a
// range of tile rows whose partial sums are held in DRAM at the same time.
struct Panel {
    uint32_t row0, col0;
    bool tail;
};  // top-left tile; tail: uses the remainder-row configuration gR
struct Band {
    uint32_t row0, rows, p0, p1;
};  // first tile row, tile rows, panel indices [p0, p1)

struct PanelPlan {
    uint32_t pdiag_tile = 0;             // tile size of the partial sums [bytes]
    uint32_t Ktm = 0, Kbt = 0;           // K tiles after concatenation; K tiles of one K block
    OptConfig gU, gR;                    // configuration of the regular panels / of the remainder panel row
    uint32_t nbr = 0, nbc = 0;           // panel rows and columns
    std::vector<Panel> panels;           // row-major; built and issued in this order
    uint32_t nband = 0, band_tiles = 0;  // bands; tiles of one band as allocated (distance between planes)
    std::vector<Band> bands;
    const OptConfig& cfg_of(const Panel& pn) const { return pn.tail ? gR : gU; }
    uint32_t panel_rows() const { return gU.cr * gU.pm; }  // height of a regular panel [tiles]
};

// First half of plan_panels: chooses the GEMM shape (core grid, tiles per core, kernel
// block width) and covers the output with panels. gd is the size of the core grid and
// l1_budget the L1 budget of the shape search [bytes]. Throws std::runtime_error if no
// shape is found.
static void plan_panel_shapes(PanelPlan& pp, const RunCfg& c, const IssuePlan& ip, const tt::tt_metal::CoreCoord& gd,
                              uint32_t l1_budget) {
    const uint32_t Mt = c.Mt, Nt = c.Nt;

    // The buffer of partial sums holds the signed partial sum of every diagonal: plane d
    // starts at tile d*band_tiles. With fuse or the pair loop it is the Int32
    // accumulation itself (the GEMM reads, adds and writes back).
    pp.pdiag_tile = tt::tile_size((ip.fuse || ip.pairloop) ? tt::DataFormat::Int32 : tt::DataFormat::Float32);

    // The fused and pair-loop modes add CBs to the GEMM, which the L1 estimate of the
    // shape search takes into account.
    const GemmMode gmode{ip.fuse, ip.pairloop};

    // One program serves all diagonals. Diagonal d has d+1 pairs, so its length along K
    // differs, but nothing is zero-padded: the concatenated operands hold nslice*K
    // columns (Ktm tiles), and gemm_run passes the number of kernel blocks of each
    // diagonal as a runtime argument. All diagonals thus share even the compile-time arguments
    // and the kernels are JIT-built once.
    // nslice is also the number of slices per K block in the layout of the concatenated
    // operands: the slicing stage, the buffer length Ktm and the GEMM all use
    // RunCfg::nslice().
    pp.Ktm = c.nslice() * (c.K / 32);
    pp.Kbt = ip.Kc / 32;
    const uint32_t Kbt = pp.Kbt;

    // Shape of the regular panels: the output is covered with nbr x nbc panels of
    // (cr*pm) x (cc*pn) tiles, so that any M and N use the whole core grid.
    const OptConfig gU = choose_cfg_1b(Mt, Nt, Kbt, gd.y, gd.x, gmode, l1_budget);
    if (!gU.ok)
        throw std::runtime_error(fmt::format(
            "Ozaki scheme I: no GEMM configuration covers the {}x{} tile output on the {}x{} core grid within the "
            "L1 budget. Change the matrix dimensions m or n.",
            Mt, Nt, (uint32_t)gd.y, (uint32_t)gd.x));

    // Rows may leave a remainder, which a last panel row with a smaller core grid gR
    // covers. Columns always divide (choose_cfg_1b returns no other shape).
    const uint32_t ph = gU.cr * gU.pm, pw = gU.cc * gU.pn;  // height and width of a regular panel
    const uint32_t nbr_full = Mt / ph, nbc = Nt / pw;
    const uint32_t rem_rows = Mt - nbr_full * ph;
    OptConfig gR = gU;
    if (rem_rows) {
        // (cr, pm) are chosen anew for the remainder to use the most cores. This always
        // succeeds: choose_cfg_1b admits a remainder only if (Mt/pm) % cr >= 2, so
        // rem_rows = r*pm with 2 <= r < cr <= gy, and cr' = r with pm' = pm, the same
        // pn, subblock and bw is a candidate of choose_remainder_cfg.
        OptConfig gc = choose_remainder_cfg(rem_rows, gU, gd.y, Kbt, gmode, l1_budget);
        TT_FATAL(gc.ok, "plan_panels: no configuration for the remainder of {} tile rows (pm={})", rem_rows, gU.pm);
        gR = gc;
    }
    const uint32_t nbr = nbr_full + (rem_rows ? 1u : 0u);
    pp.gU = gU;
    pp.gR = gR;
    pp.nbr = nbr;
    pp.nbc = nbc;

    pp.panels.reserve((size_t)nbr * nbc);
    for (uint32_t br = 0; br < nbr; br++)
        for (uint32_t bc = 0; bc < nbc; bc++) pp.panels.push_back({br * ph, bc * pw, rem_rows && br == nbr_full});
}

// Second half of plan_panels: cutting the output into row bands.
// For the whole output the planes of partial sums (one per diagonal) can exhaust the
// DRAM. The output is therefore cut into bands of tile rows; planes are allocated for
// one band only and "GEMMs of the band, then reconstruction of the band" is repeated.
// The amount of computation is unchanged. Band boundaries coincide with panel-row
// boundaries, so that no panel spans two bands, and bands are cut along rows only: the
// number of output columns Nt is unchanged and only the write position of a panel
// becomes relative to its band.
// bpr is the number of panel rows per band: the largest for which the planes of one
// band fit avail, the DRAM that remains free of dram_free after the buffers still to be
// allocated (oz1_dram_need) and the margin. fits tells whether even that band fits
// (need <= avail); if not, the allocation would fail.
struct BandRows {
    uint32_t bpr = 1;              // panel rows per band
    bool fits = false;             // the planes of one band fit avail
    uint64_t need = 0, avail = 0;  // both sides of that comparison [bytes]
};
static BandRows plan_band_panel_rows(const PanelPlan& pp, const RunCfg& c, const IssuePlan& ip, size_t dram_free) {
    const uint32_t Mt = c.Mt, Nt = c.Nt, nbr = pp.nbr, ph = pp.panel_rows();

    // Partial sums of one tile row: one plane per diagonal.
    const size_t row_bytes = (size_t)pp.pdiag_tile * c.nslice() * Nt;
    const Oz1DramNeed need = oz1_dram_need(c, ip.recon_limbs());
    const uint64_t later = need.total() + OZ_DRAM_MARGIN_BYTES;
    const uint64_t avail = ((uint64_t)dram_free > later) ? (uint64_t)dram_free - later : 0u;
    auto band_bytes = [&](uint32_t bpr) { return (uint64_t)row_bytes * std::min(Mt, bpr * ph); };
    BandRows r;
    r.bpr = nbr;
    while (r.bpr > 1 && band_bytes(r.bpr) > avail) r.bpr--;
    r.need = band_bytes(r.bpr);
    r.avail = avail;
    r.fits = r.need <= avail;
    return r;
}

// Plans the GEMM shape, the panels and the row bands. gd is the size of the core grid
// and df the DRAM that is free before any allocation; the shape search uses the L1
// budget oz_l1_budget() (common/gemm_host.hpp). The total free space decides the
// number of row bands; each large buffer must in addition fit the largest contiguous
// free block (a necessary condition only: how earlier allocations split that block is
// not modeled). Throws std::runtime_error if the configuration does not fit.
static PanelPlan plan_panels(const RunCfg& c, const IssuePlan& ip, const tt::tt_metal::CoreCoord& gd,
                             const DramFree& df) {
    PanelPlan pp;
    const size_t dram_free = df.total;
    plan_panel_shapes(pp, c, ip, gd, oz_l1_budget());
    const uint32_t Mt = c.Mt, Nt = c.Nt, nbr = pp.nbr, nbc = pp.nbc, ph = pp.panel_rows();
    const BandRows brs = plan_band_panel_rows(pp, c, ip, dram_free);
    if (!brs.fits)  // not even a band of one panel row fits
        throw std::runtime_error(fmt::format(
            "Ozaki scheme I: not enough device DRAM. The partial sums of one panel row need {:.2f} GB, but only "
            "{:.2f} GB remain free after the operands, slices and output ({:.1f} GB kept in reserve). "
            "Reduce m, n, k or the number of slices (Options::slices).",
            brs.need / 1e9, brs.avail / 1e9, OZ_DRAM_MARGIN_BYTES / 1e9));
    const uint32_t bpr = brs.bpr;
    pp.nband = (nbr + bpr - 1) / bpr;
    const uint32_t band_rows = std::min(Mt, bpr * ph);  // tile rows of the allocated planes
    pp.band_tiles = band_rows * Nt;                     // distance between consecutive planes [tiles]
    for (uint32_t b0 = 0; b0 < nbr; b0 += bpr) {
        const uint32_t b1 = std::min(nbr, b0 + bpr);
        pp.bands.push_back({b0 * ph, std::min(Mt, b1 * ph) - b0 * ph, b0 * nbc, b1 * nbc});
    }
    // Each large buffer must fit the largest contiguous free block. A buffer is placed
    // at the same address in every DRAM bank, so the sizes over all banks are compared.
    const Oz1DramNeed nd = oz1_dram_need(c, ip.recon_limbs());
    const uint64_t f64t = OZ_F64_TILE_BYTES, in_t = tt::tile_size(OZ_FMT.in);
    const uint64_t Kto = c.K / 32, Ktm = (uint64_t)c.nslice() * Kto;
    const std::pair<const char*, uint64_t> big[] = {
        {"The FP64 operand A", f64t * Mt * Kto},
        {"The FP64 operand B", f64t * Kto * Nt},
        {"The slices of A", in_t * Mt * Ktm},
        {"The slices of B", in_t * Nt * Ktm},
        {"The FP64 output", nd.output},
        {"One row band of partial sums", (uint64_t)pp.pdiag_tile * c.nslice() * pp.band_tiles}};
    for (const auto& [what, bytes] : big)
        if (bytes > df.largest_block)
            throw std::runtime_error(fmt::format(
                "Ozaki scheme I: not enough contiguous device DRAM. {} needs {:.2f} GB, but the largest free block "
                "is {:.2f} GB ({:.2f} GB free in total). Reduce m, n, k or the number of slices (Options::slices).",
                what, bytes / 1e9, df.largest_block / 1e9, df.total / 1e9));
    return pp;
}

// ===== Constants of the reconstruction =====
// Number of leading diagonals whose Int32 accumulation S_d of the deferred
// reconstruction is an exact FP32 integer. Diagonal d has d+1 pairs, and each of their K
// products is bounded by dmax^2, so after all K blocks
//     |S_d| <= (d+1) * K * dmax^2      (K: the full, unblocked length).
// If the right-hand side is at most 2^24, the reconstruction kernel reads S_d with a
// single typecast instead of splitting it into two FP32 values. The bound grows with d,
// so the diagonals that qualify are the first floor(2^24 / (K dmax^2)), at most ndiag.
static uint32_t recon_nsmall(uint32_t K, uint32_t ndiag) {
    const uint64_t dm = OZ1_DMAX;
    const uint64_t per = (uint64_t)K * dm * dm;  // bound of one pair, K*dmax^2
    if (per == 0) return 0;
    return (uint32_t)std::min<uint64_t>((1ull << 24) / per, ndiag);
}

// The radix and the final scale radix^-(L+2) in the form the reconstruction kernels take
// them (host arithmetic only).
static ReconScale recon_scale(const RunCfg& c) {
    const int radix = (int)OZ1_RADIX, L = c.L();
    const uint32_t K = c.K;
    ReconScale rs;

    // Used by the deferred reconstruction only; recon_build passes 0 to the kernel in
    // the other configurations.
    rs.nsmall = recon_nsmall(K, c.nslice());

    rs.radixbits = f32_bits((float)radix);

    // radix^-(L+2) is in general not a power of two (radix 14 of BFP4, 254 of INT8). For
    // all formats alike it is passed as a 64-bit significand W and an exponent qw, with
    // which the writer multiplies in integer arithmetic (W is exactly 2^63 for a
    // power-of-two radix). Multiplying by an FP64 constant instead would add a relative
    // error of 2^-53 to every element.
    // Error of W: the long double (64-bit significand) is divided by the radix L+2 times,
    // each division rounding by at most 2^-64 relative, about (L+2) * 2^-64 in total;
    // the split into W and qw copies the significand and adds nothing. For a
    // power-of-two radix the divisions are exact.
    {
        long double w = 1.0L;
        for (int i = 0; i < L + 2; i++) w /= (long double)radix;
        const OzScale64 W = oz_split_scale_ld(w);
        rs.Whi = W.hi;
        rs.Wlo = W.lo;
        rs.qw = W.q;
    }

    // The same scale as a double, for the soft-float variant of the writer. The library
    // never selects that variant; the two runtime arguments are still filled.
    {
        const uint64_t u = f64_bits(std::pow((double)radix, -(double)(L + 2)));
        rs.Wdlo = (uint32_t)u;
        rs.Wdhi = (uint32_t)(u >> 32);
    }
    return rs;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
