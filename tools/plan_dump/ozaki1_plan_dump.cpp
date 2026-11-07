// SPDX-License-Identifier: Apache-2.0
// Host-only regression harness for the scheme I planner (no device is opened).
// Prints every decision of the host planner for a fixed list of cases.  compare.sh
// builds it once against a baseline revision and once against the working tree,
// for every format, and diffs the outputs.  Not part of the library build.
//   ozaki1_plan_dump [PART]   0 = all (default), 1 = fixed p, 2 = automatic p, 3 = large shapes
#include "ozaki1/stages.hpp"
#include "ozaki1/auto_precision.hpp"
#include <cinttypes>
#include <cstdlib>
#include <random>
#include <string>
#include <exception>

namespace ttgemm::detail {
std::string kernel_path(std::string_view) { return {}; }
}  // namespace ttgemm::detail
using namespace ttgemm::detail::TTGEMM_BACKEND_NS;

struct Case {
    const char* tag;
    uint32_t M, N, K;
    int p;  // 0: automatic
    int input;
};

static void make_input(int kind, uint32_t M, uint32_t N, uint32_t K, std::vector<double>& A, std::vector<double>& B) {
    A.assign((size_t)M * K, 0.0);
    B.assign((size_t)K * N, 0.0);
    std::mt19937_64 rng(1234 + kind);
    std::uniform_real_distribution<double> u(-1, 1), e(0, 1);
    for (auto& x : A) x = u(rng);
    for (auto& x : B) x = u(rng);
    if (kind == 2 || kind == 3) {  // wide dynamic range inside rows/columns
        const double phi = kind == 2 ? 8.0 : 24.0;
        for (auto& x : A) x *= std::exp2(-phi * 4 * e(rng));
        for (auto& x : B) x *= std::exp2(-phi * 4 * e(rng));
    }
    if (kind == 4) {
        std::fill(A.begin(), A.end(), 1.0);
        std::fill(B.begin(), B.end(), 1.0);
    }
    if (kind == 5) std::fill(A.begin(), A.end(), 0.0);
    if (kind == 6) {
        for (size_t i = 0; i < A.size(); i++) A[i] = (i % K) % 2 ? -1 : 1;
        std::fill(B.begin(), B.end(), 1.0);
    }
    if (kind == 7) {
        for (auto& x : A) x *= 0x1p-120;
        for (auto& x : B) x *= 0x1p100;
    }
    if (kind == 8) {  // exact powers of two as row/column maxima
        for (auto& x : A) x = 0.5;
        for (auto& x : B) x = 2.0;
    }
}

static void print_cfg(const char* nm, const OptConfig& g) {
    printf("   %s: ok=%d cr=%u cc=%u pm=%u pn=%u sh=%u sw=%u bw=%u\n", nm, (int)g.ok, g.cr, g.cc, g.pm, g.pn, g.sh,
           g.sw, g.bw);
}

// L1 byte counts of the candidates of the shape search for one K block length (tiles).
static void dump_l1(uint32_t Kbt) {
    static constexpr std::pair<uint32_t, uint32_t> PC[] = {{8, 8}, {8, 4}, {4, 8}, {8, 2}, {2, 8}, {4, 4}, {8, 1}, {1, 8},
                                                          {4, 2}, {2, 4}, {4, 1}, {1, 4}, {2, 2}, {2, 1}, {1, 2}, {1, 1}};
    const GemmMode modes[] = {GemmMode{false, false}, GemmMode{true, false}, GemmMode{false, true}};
    for (int mi = 0; mi < 3; mi++) {
        const GemmMode& m = modes[mi];
        if (OZ1_INT32_PARTIAL && mi > 0) continue;  // spill_gemm_spec rejects fuse/pairloop with Int32
        printf("  l1 mode=%d:", mi);
        for (const auto& pc : PC) {
            uint32_t sh = 0, sw = 0;
            if (!pick_subblock(pc.first, pc.second, sh, sw)) continue;
            for (uint32_t bw : divisors_desc(Kbt)) {
                const uint64_t l1 = oz1_gemm_l1_bytes(pc.first, pc.second, sh, sw, bw, m);
                // The estimate of the search must equal the size of the CBs actually created.
                OptConfig g;
                g.pm = pc.first;
                g.pn = pc.second;
                g.sh = sh;
                g.sw = sw;
                g.bw = bw;
                g.cr = g.cc = 2;
                g.ok = true;
                const uint64_t l1s = mcast_gemm_l1_bytes(g, spill_gemm_spec(g, bw, oz1_spill_opts(m)));
                if (l1 != l1s) {
                    printf("\nL1 MISMATCH pm=%u pn=%u bw=%u: %" PRIu64 " vs %" PRIu64 "\n", pc.first, pc.second, bw, l1,
                           l1s);
                    abort();
                }
                printf(" %u,%u,%u:%" PRIu64, pc.first, pc.second, bw, l1);
            }
            printf(" max=%u", oz1_max_bw(pc.first, pc.second, sh, sw, Kbt, m, oz_l1_budget()));
        }
        printf("\n");
    }
}

// The exactness check over a grid of K and L (the planner only exercises the chosen K_c).
static void dump_exact() {
    printf("EXACT");
    for (uint32_t K : {32u, 64u, 96u, 160u, 1024u, 4096u, 8192u, 16384u, 16416u, 32768u, 65536u, 131072u, 1048576u,
                       16777216u, 33554432u, 67108864u})
        for (int L : {0, 1, 2, 3, 7, 15, 31, 63}) {
            const bool acc = ozaki_check_exact(K, L), mul = true;
            printf(" %u/%d:%d%d", K, L, (int)mul, (int)acc);
        }
    printf("\n");
}

static void run_case(const Case& cs) {
    printf("CASE %s M=%u N=%u K=%u p=%d input=%d\n", cs.tag, cs.M, cs.N, cs.K, cs.p, cs.input);
    RunCfg c;
    c.M = cs.M;
    c.N = cs.N;
    c.K = cs.K;
    c.Mt = cs.M / 32;
    c.Nt = cs.N / 32;
    if (cs.p) {
        c.p = cs.p;
    } else {
        std::vector<double> A, B;
        make_input(cs.input, cs.M, cs.N, cs.K, A, B);
        c.p = oz_fp64_p(A, B, c.M, c.N, c.K);
    }
    printf("  cfg: radix=%d dmax=%u fid=%d nshift=%d bf_ceil=%d tau_above=%d ceil_mode=%u p=%d L=%d nslice=%u\n",
           (int)OZ1_RADIX, OZ1_DMAX, (int)OZ1_FID, OZ1_NSHIFT, (int)OZ1_BF_CEIL, (int)OZ1_TAU_ABOVE, OZ1_CEIL_MODE, c.p,
           c.L(), c.nslice());
    {
        const IssueCandidates ic = issue_candidates(c);
        printf("  cand: kc_cat=%u kc_pw=%u nkb_cat=%u nkb_pw=%u n_cat=%u n_pw=%u defer_ok=%d fuse_ok=%d\n", ic.kc_cat,
               ic.kc_pw, ic.nkb_cat, ic.nkb_pw, ic.n_cat, ic.n_pw, (int)ic.defer_ok, (int)ic.fuse_ok);
    }
    IssuePlan ip;
    try {
        ip = plan_issue(c);
    } catch (const std::exception& e) {
        printf("  ISSUE FAIL: %s\n", e.what());
        return;
    }
    printf("  issue: pairloop=%d pairwise=%d fuse=%d defer=%d Kc=%u nkb=%u per_pair=%d limbs=%d single=%d issues=%u\n",
           (int)ip.pairloop, (int)ip.pairwise, (int)ip.fuse, (int)ip.defer, ip.Kc, ip.nkb, (int)ip.per_pair(),
           (int)ip.recon_limbs(), (int)ip.single_kblock(), ip.gemm_issues(c.nslice()));
    if (ip.Kc / 32 <= 4096) dump_l1(ip.Kc / 32);
    {
        const ReconScale rs = recon_scale(c);
        printf("  recon_scale: radixbits=%08x W=%08x:%08x qw=%d Wd=%08x:%08x nsmall=%u\n", rs.radixbits, rs.Whi, rs.Wlo,
               rs.qw, rs.Wdhi, rs.Wdlo, rs.nsmall);
        const Oz1DramNeed nd = oz1_dram_need(c, ip.recon_limbs());
        printf("  need: operands=%" PRIu64 " slices=%" PRIu64 " scale=%" PRIu64 " output=%" PRIu64 " limbs=%" PRIu64
               " total=%" PRIu64 "\n",
               nd.operands, nd.slices, nd.scale, nd.output, nd.limbs, nd.total());
    }
    const tt::tt_metal::CoreCoord grids[] = {{11, 10}, {13, 10}, {8, 8}};
    const DramFree dfs[] = {{34000000000ull, 34000000000ull}, {30000000000ull, 30000000000ull},
                            {20000000000ull, 20000000000ull}, {12000000000ull, 12000000000ull},
                            {6000000000ull, 6000000000ull},   {2000000000ull, 2000000000ull},
                            {30000000000ull, 3000000000ull},  {1200000000ull, 1200000000ull}};
    for (const auto& gd : grids) {
        for (const DramFree& df : dfs) {
            printf("  grid %zux%zu dram %.1f/%.1f GB\n", (size_t)gd.x, (size_t)gd.y, df.total / 1e9,
                   df.largest_block / 1e9);
            PanelPlan pp;
            try {
                pp = plan_panels(c, ip, gd, df);
            } catch (const std::exception& e) {
                printf("   PANEL FAIL: %s\n", e.what());
                continue;
            }
            printf("   pp: pdiag_tile=%u nslice=%u Ktm=%u Kbt=%u nbr=%u nbc=%u nband=%u band_tiles=%u panel_rows=%u\n",
                   pp.pdiag_tile, c.nslice(), pp.Ktm, pp.Kbt, pp.nbr, pp.nbc, pp.nband, pp.band_tiles, pp.panel_rows());
            print_cfg("gU", pp.gU);
            print_cfg("gR", pp.gR);
            printf("   panels %zu:", pp.panels.size());
            for (const Panel& pn : pp.panels) printf(" (%u,%u,%d)", pn.row0, pn.col0, (int)pn.tail);
            printf("\n   bands %zu:", pp.bands.size());
            for (const Band& b : pp.bands) printf(" (%u,%u,%u,%u)", b.row0, b.rows, b.p0, b.p1);
            printf("\n");
            {
                const BandRows br = plan_band_panel_rows(pp, c, ip, df.total);
                printf("   bandrows: bpr=%u fits=%d need=%" PRIu64 " avail=%" PRIu64 "\n", br.bpr, (int)br.fits, br.need,
                       br.avail);
            }
            // The per-diagonal retargeting of gemm_run, derived from the plan (first and last
            // panel, all diagonals of K block 0 and of the last K block).
            for (const Panel& pn : {pp.panels.front(), pp.panels.back()}) {
                const OptConfig& g = pp.cfg_of(pn);
                const uint32_t Kt = pp.Ktm;
                printf("   panel(%u,%u): dA=%u dB=%u bw=%u bpp=%u nbmax=%u", pn.row0, pn.col0, Kt * pn.row0, pn.col0,
                       g.bw, pp.Kbt / g.bw, Kt / g.bw);
                for (uint32_t blk : {0u, ip.nkb - 1})
                    for (uint32_t d = 0; d < c.nslice(); d++) {
                        const uint32_t n = ip.pairwise ? 1u : d + 1, a0 = 0;
                        printf(" [%u,%u:%u,%u,%u]", blk, d, n * pp.Kbt / g.bw, (blk * c.nslice() + a0) * pp.Kbt,
                               (blk * c.nslice() + (c.nslice() - 1 - d) + a0) * pp.Kbt * c.Nt);
                    }
                printf("\n");
            }
        }
    }
}

int main(int argc, char** argv) {
    const int part = argc > 1 ? atoi(argv[1]) : 0;  // 0 = all, 1 = fixed p, 2 = automatic p, 3 = large shapes
    std::vector<Case> cs;
    if (part == 0 || part == 1) dump_exact();
    auto valid = [](uint32_t M, uint32_t N, uint32_t K) {
        const uint64_t lim = 0xFFFFFFFFull / 4096u;
        const uint64_t Mt = M / 32, Nt = N / 32, Kt = K / 32;
        return Mt * Kt <= lim && Kt * Nt <= lim && Mt * Nt <= lim;
    };
    if (part == 0 || part == 1) {
        for (uint32_t K : {32u, 64u, 96u, 128u, 1024u, 3520u, 7040u, 14080u, 15136u, 16352u, 16384u, 16416u, 65536u,
                           131072u, 262144u, 1048576u})
            for (int p : {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64})
                for (uint32_t M : {64u, 128u}) cs.push_back({"fixed-K", M, 64, K, p, 0});
        for (uint32_t M : {64u, 96u, 128u, 1024u, 3520u, 7040u, 14080u, 15136u})
            for (uint32_t N : {64u, 96u, 128u, 1024u, 3520u, 7040u, 14080u, 15136u})
                for (int p : {2, 4, 8, 24})
                    if (valid(M, N, 1024)) cs.push_back({"fixed-MN", M, N, 1024, p, 0});
    }
    if (part == 0 || part == 2) {
        for (uint32_t K : {32u, 64u, 128u, 1024u, 4096u, 8192u}) cs.push_back({"auto-uniform", 64, 64, K, 0, 1});
        for (uint32_t K : {64u, 1024u, 4096u}) {
            cs.push_back({"auto-wide8", 64, 64, K, 0, 2});
            cs.push_back({"auto-wide24", 64, 64, K, 0, 3});
            cs.push_back({"auto-ones", 64, 64, K, 0, 4});
            cs.push_back({"auto-alt", 64, 64, K, 0, 6});
            cs.push_back({"auto-scaled", 64, 64, K, 0, 7});
            cs.push_back({"auto-pow2", 64, 64, K, 0, 8});
        }
        cs.push_back({"auto-zeroA", 64, 64, 64, 0, 5});
        cs.push_back({"auto-rect", 1024, 128, 512, 0, 1});
        cs.push_back({"auto-rect", 128, 1024, 512, 0, 3});
        cs.push_back({"auto-cube", 1024, 1024, 1024, 0, 1});
    }
    if (part == 0 || part == 3) {
        for (uint32_t n : {1024u, 3520u, 7040u, 14080u, 15136u})
            for (int p : {2, 3, 4, 6, 8, 12, 16, 24, 32})
                if (valid(n, n, n)) cs.push_back({"fixed-cube", n, n, n, p, 0});
        for (uint32_t K : {3520u, 7040u, 14080u, 15136u, 65536u, 262144u, 1048576u})
            for (uint32_t M : {64u, 1024u, 3520u, 14080u, 15136u})
                for (int p : {2, 4, 8, 16, 24})
                    if (valid(M, 64, K)) {
                        cs.push_back({"fixed-rectA", M, 64, K, p, 0});
                        cs.push_back({"fixed-rectB", 64, M, K, p, 0});
                    }
    }
    for (const Case& c : cs) {
        run_case(c);
        fflush(stdout);
    }
    return 0;
}
