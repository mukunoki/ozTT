// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: configuration search and issue of the partial-product GEMMs.
//
// The GEMM stage multiplies the slices of A and B: the slice pairs (i, j) of one diagonal
// d = i + j share the weight radix^-(d+2), so their products are summed into one partial
// sum P_d per diagonal, which the reconstruction stage then combines. This header provides
//   - the L1 estimate of a GEMM program and the search for the kernel block width
//     (oz1_gemm_l1_bytes, oz1_max_bw),
//   - the search for the shape of an output panel (choose_cfg_1b, choose_remainder_cfg),
//   - the construction of the program of one panel (gemm_build) and the issue of one
//     diagonal with it (gemm_run).
// The program itself (multicast data movement and the compute kernel) is shared with
// scheme II: common/mcast_gemm.hpp.
//
// Program reuse: a diagonal differs from the next only in its length along K (the number
// of concatenated slice pairs) and in its output position. Both are runtime arguments, so
// one compiled program per panel serves every diagonal and gemm_run only replaces a few
// runtime arguments. Kernels, CBs and JIT builds are thus paid once, not per diagonal.
//
// Two block sizes along K are to be told apart. A K block is the unit of the exactness
// condition: K_c elements, Kbt tiles (ozaki1/plan.hpp). A kernel block is what the GEMM
// kernel reads and multiplies at a time: bw tiles of K. bw divides Kbt.

#include "ozaki1/format.hpp"
#include "common/mcast_gemm.hpp"
#include <array>
#include <utility>

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// How the GEMM accumulates. The L1 estimate of the shape search depends on the set of
// CBs, so the mode is passed to the search explicitly.
struct GemmMode {
    bool fuse = false;      // the writer adds the result to the Int32 partial sums already in DRAM
    bool pairloop = false;  // one program loops over the slice pairs of a diagonal and folds the
                            // FP32 accumulator into Int32 at every pair boundary (excludes fuse)
};

// CB and compute-kernel options of the GEMM (input of spill_gemm_spec). gemm_build and the
// L1 estimate oz1_gemm_l1_bytes both use this function, so the estimate cannot disagree
// with the CBs that are actually created.
inline SpillOpts oz1_spill_opts(GemmMode mode) {
    SpillOpts so;
    so.in_fmt = OZ_FMT.in;
    so.acc_fmt = OZ_FMT.out;
    so.fidelity = OZ1_FID;
    so.fuse = mode.fuse;
    so.pairloop = mode.pairloop;
    return so;
}

// Total bytes of the CBs that a GEMM with pm x pn output tiles per core, subblock sh x sw
// and kernel block width bw creates on one core (spill_gemm_l1_bytes, which counts the
// CBs of spill_gemm_spec as gemm_build creates them): the double-buffered input CBs in
// the input format, the pm*pn output tiles, and the extra CBs of the fused and pair-loop
// modes. The estimate is in bytes of the actual tile sizes because the input tile ranges
// from 320 bytes (BFP2) to 4096 bytes (TF32); a fixed budget in tiles would leave L1
// unused for the narrow formats. L1 use does not depend on the length of K.
inline uint64_t oz1_gemm_l1_bytes(uint32_t pm, uint32_t pn, uint32_t sh, uint32_t sw, uint32_t bw, GemmMode mode) {
    OptConfig g;
    g.pm = pm;
    g.pn = pn;
    g.sh = sh;
    g.sw = sw;
    g.bw = bw;
    return spill_gemm_l1_bytes(g, oz1_spill_opts(mode));
}

// Largest kernel block width bw [tiles] among the divisors of Kbt whose CBs fit the L1
// budget l1_budget [bytes]; 0 if none fits. bw must divide Kbt (the K tiles of one slice
// pair in one K block): otherwise a kernel block would straddle a slice or K block
// boundary and break the pairing of slice q of A with slice d-q of B.
inline uint32_t oz1_max_bw(uint32_t pm, uint32_t pn, uint32_t sh, uint32_t sw, uint32_t Kbt, GemmMode mode,
                           uint32_t l1_budget) {
    for (uint32_t cand : divisors_desc(Kbt))
        if (oz1_gemm_l1_bytes(pm, pn, sh, sw, cand, mode) <= (uint64_t)l1_budget) return cand;
    return 0;
}

// ===== Shape of an output panel =====
// The output tiles per core (pm, pn) are limited by L1 (at most 8 x 8). The capacity of
// the destination register (4 tiles with 32-bit accumulation) limits only the subblock
// (sh, sw): the compute kernel works through the subblocks within a K block and spills
// the partial sums. One program covers a panel of (cr*pm) x (cc*pn) output tiles on a cr x cc core
// grid. The host splits the Mt x Nt output into panels and runs K once per panel (the
// inputs are read again; no partial sums are carried between panels), so that sizes the
// core grid does not divide still use all cores. With gy x gx the size of the core grid:
//   columns  must divide evenly: some pn in {1, 2, 4, 8} dividing Nt must leave Nt/pn
//            with a divisor in 2..gx;
//   rows     may leave a remainder panel row: some pm dividing Mt and some cr with
//            2 <= cr <= min(gy, Mt/pm) must satisfy (Mt/pm) % cr != 1 (only a remainder
//            of a single core row cannot be built).
// All arguments of the searches are in tiles:
//   Kbt  K tiles of one K block (K/32 without blocking); bw divides it (oz1_max_bw);
//   l1_budget  L1 budget of the CBs in bytes.
// The subblock is the largest entry of the table SB_CHOICES (area <= 4) that divides (pm, pn)
// (pick_subblock in common/gemm_host.hpp).

// Configuration of the remainder panel row of rem tile rows. (cc, pn) of the regular
// panels gU are kept and only (cr, pm) are chosen anew so as to use the most cores; the
// subblock and bw are chosen again for the new pm. rem is r*pm of the regular panel with
// 2 <= r < cr <= gy (see choose_cfg_1b), so the four candidates of pm suffice.
static OptConfig choose_remainder_cfg(uint32_t rem, const OptConfig& gU, uint32_t gy, uint32_t Kbt, GemmMode mode,
                                      uint32_t l1_budget) {
    static constexpr std::array<uint32_t, 4> PM = {{8, 4, 2, 1}};
    OptConfig best;
    uint32_t bcore = 0;
    for (uint32_t cr = std::min(gy, rem); cr >= 2; --cr) {
        if (rem % cr) continue;
        const uint32_t pm = rem / cr;
        if (std::find(PM.begin(), PM.end(), pm) == PM.end()) continue;
        uint32_t sh = 0, sw = 0;
        if (!pick_subblock(pm, gU.pn, sh, sw)) continue;
        const uint32_t bw = oz1_max_bw(pm, gU.pn, sh, sw, Kbt, mode, l1_budget);
        if (!bw) continue;
        if (cr * gU.cc > bcore) {
            bcore = cr * gU.cc;
            best = {true, cr, gU.cc, pm, gU.pn, sh, sw, bw};
        }
    }
    return best;
}

// Configuration of the regular panels for an Mt x Nt output on a gy x gx core grid.
// Priorities: the number of cores, then the panel area; remaining ties go to the first
// candidate in the enumeration order. ok stays false if no shape satisfies the conditions
// above or fits the L1 budget.
static OptConfig choose_cfg_1b(uint32_t Mt, uint32_t Nt, uint32_t Kbt, uint32_t gy, uint32_t gx, GemmMode mode,
                               uint32_t l1_budget) {
    // Candidate (pm, pn), largest area first. They may exceed the destination register
    // because the kernel spills; each gets its largest subblock with sh | pm, sw | pn and
    // sh*sw <= 4.
    static constexpr std::array<std::pair<uint32_t, uint32_t>, 16> PC_ALL = {{{8, 8},
                                                                              {8, 4},
                                                                              {4, 8},
                                                                              {8, 2},
                                                                              {2, 8},
                                                                              {4, 4},
                                                                              {8, 1},
                                                                              {1, 8},
                                                                              {4, 2},
                                                                              {2, 4},
                                                                              {4, 1},
                                                                              {1, 4},
                                                                              {2, 2},
                                                                              {2, 1},
                                                                              {1, 2},
                                                                              {1, 1}}};
    OptConfig best;
    uint32_t bcore = 0, bcov = 0;
    for (auto& a : PC_ALL) {
        uint32_t pm = a.first, pn = a.second;
        if (Mt % pm || Nt % pn) continue;

        uint32_t sh = 0, sw = 0;
        if (!pick_subblock(pm, pn, sh, sw)) continue;

        // bw depends on (pm, pn) and the subblock only, not on the core grid.
        const uint32_t bw = oz1_max_bw(pm, pn, sh, sw, Kbt, mode, l1_budget);
        if (!bw) continue;

        // The core grid needs at least 2 rows and 2 columns: the two cores at the ends
        // of each row and column are the senders of the multicast.
        for (uint32_t cr = std::min(gy, Mt / pm); cr >= 2; --cr) {
            // Rows may leave a remainder, which the caller covers with a smaller core
            // grid (choose_remainder_cfg). The remainder needs 2 core rows as well, so
            // a remainder of exactly one core row is rejected. Columns must divide:
            // remainders in both directions would need four kinds of panels.
            if ((Mt / pm) % cr == 1) continue;
            for (uint32_t cc = std::min(gx, Nt / pn); cc >= 2; --cc) {
                if ((Nt / pn) % cc) continue;
                const uint32_t cores = cr * cc, cov = cr * pm * cc * pn;

                // The core count comes first because, with remainder rows allowed, a
                // larger panel that uses only part of the cores is also a candidate.
                if (cores > bcore || (cores == bcore && cov > bcov)) {
                    bcore = cores;
                    bcov = cov;
                    best = {true, cr, cc, pm, pn, sh, sw, bw};
                }
            }
        }
    }
    return best;
}

// One GEMM program (one output panel) and the state gemm_run needs to retarget it.
struct GemmCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};

    // A and B are the complete slice-concatenated operands, shared by all panels (held
    // here only to keep them alive); C is the buffer of the partial sums per diagonal.
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Abuf, Bbuf, Cbuf;
    uint32_t pm = 0, pn = 0, sh = 0, sw = 0, Nt = 0;
    uint32_t dWblock = 0;  // output offset of this panel within its row band [tiles]

    // Distance between the planes of consecutive diagonals in C [tiles]: the tiles of one row band.
    uint32_t planeStride = 0;

    // bw: kernel block width [tiles]; nslice: slices per K block, L+1 (the step the
    // slicing stage wrote with); Kbt: K tiles of one K block (K/32 without blocking).
    uint32_t bw = 0, nslice = 0, Kbt = 0;
    bool fuse = false;      // GemmMode::fuse
    bool pairloop = false;  // GemmMode::pairloop

    // Kernels and build-time reader arguments per core (common/mcast_gemm.hpp).
    McastGemm mg;
};

// The part of the GEMM program description that does not depend on the panel.
struct GemmBuildSpec {
    uint32_t M = 0, N = 0, K = 0;  // matrix dimensions [elements]; K is that of one slice
    uint32_t nslice = 0;           // slices L+1; the concatenated operand has nslice*K columns
    uint32_t Kbt = 0;              // K tiles of one K block (divides K/32)
    uint32_t plane_stride = 0;     // distance between the planes of consecutive diagonals [tiles]
    GemmMode mode;
};

// Builds the GEMM program of one output panel. (row0, col0) is the top-left tile of the
// panel and band_row0 the first tile row of its row band; the read offset of A and the
// write base of the output are fixed here. Abuf and Bbuf are the slice-concatenated
// operands, allocated and sliced once and shared by all panels. Per diagonal, gemm_run
// then replaces the read window along K, the number of kernel blocks and the output
// plane.
static GemmCtx gemm_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const OptConfig& g,
                          const GemmBuildSpec& b, const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig,
                          const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Abuf,
                          const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Bbuf, uint32_t row0,
                          uint32_t col0, uint32_t band_row0) {
    namespace dist = tt::tt_metal::distributed;
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    dist::MeshWorkload workload;
    auto device_range = dist::MeshCoordinateRange(md->shape());
    tt::tt_metal::Program program{};

    const uint32_t Mt = b.M / TILE_HEIGHT, Kt = b.nslice * (b.K / TILE_WIDTH), Nt = b.N / TILE_WIDTH;
    const uint32_t bw = g.bw;
    TT_FATAL(b.plane_stride > 0, "gemm_build: the plane stride of the diagonals is 0");

    // bw must divide the K tiles of one slice pair: otherwise a kernel block straddles
    // a slice boundary and the per-diagonal retargeting in gemm_run drops a fraction.
    TT_FATAL(b.Kbt > 0 && bw > 0 && b.Kbt % bw == 0 && Kt % bw == 0,
             "gemm_build: bw={} does not divide the {} K tiles of one pair and the concatenated Kt={}", bw, b.Kbt, Kt);
    const uint32_t bpp = b.Kbt / bw;  // kernel blocks per slice pair (pair boundary of the pair loop)

    // CBs, compute kernel and compile-time arguments are shared with scheme II
    // (spill_gemm_spec). spill_gemm_spec also checks that fuse and pairloop exclude each
    // other and are used with FP32 partial products only.
    McastGemmSpec spec = spill_gemm_spec(g, Kt, oz1_spill_opts(b.mode));

    // The CBs created must fit the L1 budget. The shape search estimated them with the
    // same formula, so a failure means the two were changed inconsistently.
    mcast_check_l1(g, spec, "gemm_build");

    // The number of kernel blocks is a runtime argument of the compute kernel that
    // gemm_run replaces per diagonal; the maximum is set here.
    spill_check_run(Kt / bw, b.mode.pairloop, bpp);
    spec.compute_rt = spill_compute_rt(Kt / bw, 0u, bpp);

    // Offsets of this panel. The panel is given by its top-left tile, so panels need not
    // have equal sizes (remainder panel row).
    spec.dA = Kt * row0;  // first tile of A
    spec.dB = col0;       // first tile of B

    // The buffer of partial sums holds one row band only, so the output position is
    // relative to the band (the read offset of A stays absolute).
    const uint32_t dWblock = col0 + (row0 - band_row0) * Nt;
    spec.dW = dWblock;
    McastGemm mg = build_mcast_gemm(program, md, g, Mt, Nt, Kt, Abuf, Bbuf, Pbig, spec);

    workload.add_program(device_range, std::move(program));
    GemmCtx c;
    c.workload = std::move(workload);
    c.range = device_range;
    c.Abuf = Abuf;
    c.Bbuf = Bbuf;
    c.Cbuf = Pbig;
    c.pm = g.pm;
    c.pn = g.pn;
    c.sh = g.sh;
    c.sw = g.sw;
    c.Nt = Nt;
    c.dWblock = dWblock;
    c.planeStride = b.plane_stride;
    c.bw = bw;
    c.Kbt = b.Kbt;
    c.fuse = b.mode.fuse;
    c.pairloop = b.mode.pairloop;
    c.nslice = b.nslice;
    c.mg = std::move(mg);
    return c;
}

// Issues the GEMM of diagonal d and K block blk with the program of one panel. The
// slice-concatenated operands are block-major (ozaki1/kernels/slice_writer.cpp): the
// slices of K block blk occupy a contiguous range of K, slice after slice. Only runtime
// arguments are replaced (names: common/kernels/mcast_args.h and gemm_compute_args.h):
//   reader, start of A   += (blk*nslice + a0) * Kbt                       [K tiles]
//   reader, start of B   += (blk*nslice + (nslice-1-d) + a0) * Kbt * Nt   [tiles]
//   reader, kernel blocks        nb = n * Kbt / bw
//   compute kernel               nb, acc and the kernel blocks per pair
//   writer, first output tile    dWblock + d * planeStride (plane d of the partial sums), and acc
// pair < 0: the d+1 pairs of the diagonal are concatenated along K and run as one GEMM
//   (n = d+1, a0 = 0): slices 0..d of A against slices d..0 of B.
// pair >= 0: only the pair of slice `pair` of A and slice d-pair of B is run (n = 1,
//   a0 = pair). K_eff is then K_c instead of (d+1) K_c.
// The result goes to plane d of the partial-sum buffer. acc matters only in the fused mode: 0 stores the result (first write to the plane), 1 adds it to the
// Int32 values already there.
// The workload is enqueued without waiting.
static void gemm_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, GemmCtx& c, uint32_t d,
                     uint32_t blk = 0, int pair = -1, uint32_t acc = 0) {
    auto& cq = md->mesh_command_queue();
    tt::tt_metal::Program& prog = c.workload.get_programs().at(c.range);
    const uint32_t plane = c.planeStride;
    const uint32_t n = (pair < 0) ? (d + 1) : 1u;          // number of concatenated pairs
    const uint32_t a0 = (pair < 0) ? 0u : (uint32_t)pair;  // first slice of A

    // d and pair must lie inside the concatenated window (slices 0..nslice-1); otherwise
    // nslice-1-d wraps around and the reader leaves the buffer.
    TT_FATAL(d < c.nslice && (pair < 0 || (uint32_t)pair <= d),
             "gemm_run: diagonal d={} / pair {} is outside the {} concatenated slices", d, pair, c.nslice);
    uint32_t nb = n * c.Kbt / c.bw;
    uint32_t aoff = (blk * c.nslice + a0) * c.Kbt;
    uint32_t boff = (blk * c.nslice + (c.nslice - 1 - d) + a0) * c.Kbt * c.Nt;
    const uint32_t bpp = c.Kbt / c.bw;  // kernel blocks per pair (gemm_build checked that bw divides Kbt)
    spill_check_run(nb, c.pairloop, bpp);
    for (const auto& e : c.mg.cores) {
        const uint32_t cx = (uint32_t)e.core.x, cy = (uint32_t)e.core.y;
        auto wa = mcast_writer_args((uint32_t)c.Cbuf->address(), c.Nt, c.pm, c.pn, c.sh, c.sw, cx, cy,
                                    c.dWblock + d * plane, c.fuse ? acc : 0u);
        c.mg.retarget_k(prog, e, aoff, boff, nb, wa);
        tt::tt_metal::SetRuntimeArgs(prog, c.mg.ck, e.core, spill_compute_rt(nb, c.fuse ? acc : 0u, bpp));
    }
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, c.workload, false);
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
