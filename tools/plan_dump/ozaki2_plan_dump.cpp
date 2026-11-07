// SPDX-License-Identifier: Apache-2.0
// Host-only regression harness for the scheme II planner (no device is opened).
// Prints everything the host planner decides for a fixed list of cases.  compare.sh
// builds it once against a baseline revision and once against the working tree,
// for every format, and diffs the outputs.  Not part of the library build.
//   ozaki2_plan_dump [PART]   0 = all (default), 1 = automatic precision, 2 = fixed precision,
//                             3 = long cubes, 4 = default width and limited moduli
#include "ozaki2/stages.hpp"
#include <cinttypes>
#include <cstring>
#include <random>
#include <string>

namespace ozTTgemm::detail {
std::string kernel_path(std::string_view) { return {}; }
}  // namespace ozTTgemm::detail
using namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS;

static void dump_moduli(const Ozaki2Moduli& mo) {
    printf("  mo: s=%d b=%d nprod=%d nplane=%d nout=%d maxg=%d fold=%d KMl=%La\n", mo.s, mo.b, mo.nprod, mo.nplane,
           mo.nout, mo.maxg, (int)mo.fold_first, mo.KMl);
    for (int i = 0; i < mo.s; i++) {
        printf("   m=%d q=%d mode=%d poff=%d maxg=%d yi=%d rinv=%a c=%a,%a,%a,%a m16=%a\n", mo.m[i], mo.q[i], mo.mode[i],
               mo.poff[i], mo.maxg_m[i], mo.yi[i], mo.rinv[i], mo.c0[i], mo.c1[i], mo.c2[i], mo.c3[i], mo.m16[i]);
        printf("    rp0");
        for (float v : mo.rp0[i]) printf(" %a", v);
        printf(" rp1");
        for (float v : mo.rp1[i]) printf(" %a", v);
        printf(" rp2");
        for (float v : mo.rp2[i]) printf(" %a", v);
        printf("\n");
        if (OZ2_DIGIT) {
            Oz2ProductGroup gr[OZ2_MAXPROD];
            const uint32_t np = oz2_product_groups(mo.m[i], gr, OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[i]);
            printf("    groups %u:", np);
            for (uint32_t j = 0; j < np; j++)
                printf(" (%d,%d,%d,%a,%a)", gr[j].a0, gr[j].b0, gr[j].g, gr[j].coeff_lo, gr[j].coeff_hi);
            printf(" ndig=%d\n", oz2_bal_digits(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd()));
        }
    }
}

struct Case {
    const char* tag;
    uint32_t M, N, K, smax;
    int fixed_b;  // 0: the width is chosen from the input, as the library does
    int passes;   // 0: automatic, 1: one pass, 2: two passes
    int input;    // 0: no input (fixed width only)
};

static void make_input(int kind, uint32_t M, uint32_t N, uint32_t K, std::vector<double>& A, std::vector<double>& B) {
    A.clear();
    B.clear();
    if (kind == 0) return;
    A.resize((size_t)M * K);
    B.resize((size_t)K * N);
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
}

static void run_case(const Case& c) {
    printf("CASE %s M=%u N=%u K=%u smax=%u fixed_b=%d passes=%d input=%d\n", c.tag, c.M, c.N, c.K, c.smax, c.fixed_b,
           c.passes, c.input);
    std::vector<double> A, B;
    make_input(c.input, c.M, c.N, c.K, A, B);
    Oz2Plan r;
    r.M = c.M;
    r.N = c.N;
    r.K = c.K;
    Oz2Knobs kn;
    kn.max_moduli = c.smax;
    if (c.fixed_b > 0) kn.b = (uint32_t)c.fixed_b;
    kn.passes = c.passes == 1 ? ozTTgemm::Reconstruction::OnePass
                              : (c.passes == 2 ? ozTTgemm::Reconstruction::TwoPass : ozTTgemm::Reconstruction::Automatic);
    bool ok;
    try {
        choose_moduli_and_width(r, kn, A, B);
        ok = true;
    } catch (const std::runtime_error& e) {
        ok = false;
        fprintf(stderr, "[%s K=%u smax=%u b=%d] %s\n", c.tag, c.K, c.smax, c.fixed_b, e.what());
    }
    if (!ok) {
        printf("  MODULI FAIL\n");
        return;
    }
    printf("  plan: b=%u ND=%u S=%u S_req=%u two_pass=%d shift=%d Mt=%u Nt=%u Kt=%u ntot=%u,%u,%u cplan=%u,%u\n", r.b,
           r.ND, r.S, r.S_req, (int)r.two_pass, oz2_shift(r), r.Mt, r.Nt, r.Kt, r.ntotA, r.ntotB, r.ntotC, r.cplan.chunk,
           r.cplan.planes);
    dump_moduli(r.mo);
    if (OZ2_DIGIT && r.cplan.planes) printf("  compose depth=%u\n", compose_cb_depth(r.mo, r.cplan.chunk));
    {
        check_moduli_layout(r.mo);
        const Ozaki2Moduli& g = r.mo;
        printf("  slice group: s=%d nplane=%d maxg=%d poff", g.s, g.nplane, g.maxg);
        for (int v : g.poff) printf(" %d", v);
        printf(" planes=%u\n", (uint32_t)g.nplane);
        printf("  slice c0");
        for (int i = 0; i < g.s; i++)
            printf(" %d,%d,%d,%a,%a,%a,%a,%a", g.m[i], g.q[i], g.mode[i], g.rinv[i], g.c0[i], g.c1[i], g.c2[i], g.c3[i]);
        printf("\n");
    }
    const tt::tt_metal::CoreCoord gd{11, 10};
    bool pok;
    try {
        plan_panels(r, gd);
        pok = true;
    } catch (const std::runtime_error& e) {
        pok = false;
        fprintf(stderr, "[%s M=%u N=%u K=%u] %s\n", c.tag, c.M, c.N, c.K, e.what());
    }
    if (!pok) {
        printf("  PANEL FAIL\n");
        return;
    }
    printf("  panels: pMt=%u pNt=%u nbr=%u nbc=%u g=%d,%u,%u,%u,%u,%u,%u,%u,%u\n", r.pMt, r.pNt, r.nbr, r.nbc,
           (int)r.g.ok, r.g.cr, r.g.cc, r.g.pm, r.g.pn, r.g.sh, r.g.sw, r.g.bw, r.Kt / r.g.bw);
    const DramFree dfs[] = {{34000000000ull, 34000000000ull}, {30000000000ull, 30000000000ull},
                            {20000000000ull, 20000000000ull}, {12000000000ull, 12000000000ull},
                            {6000000000ull, 6000000000ull},   {2000000000ull, 2000000000ull},
                            {30000000000ull, 3000000000ull},  {1200000000ull, 1200000000ull}};
    for (const DramFree& df : dfs) {
        bool fits;
        try {
            plan_bands(r, df);
            fits = true;
        } catch (const std::runtime_error& e) {
            fits = false;
            if (df.total == 2000000000ull)
                fprintf(stderr, "[%s M=%u N=%u K=%u df=%.1f/%.1f GB] %s\n", c.tag, c.M, c.N, c.K, df.total / 1e9,
                        df.largest_block / 1e9, e.what());
        }
        if (!fits) {
            printf("  bands[%.0f/%.0f]: FAIL\n", df.total / 1e9, df.largest_block / 1e9);
            continue;
        }
        printf("  bands[%.0f/%.0f]: bpr=%u nband=%u bMt=%u ntotAb=%u ntotCb=%u\n", df.total / 1e9,
               df.largest_block / 1e9, r.bpr, r.nband, r.bMt(), r.ntotAb(), r.ntotCb());
        const ReconPlan rp = make_recon_plan(r.mo, r.two_pass, r.ntotCb());
        printf("   recon: passes=%zu expected=%u limb_tiles=%u a128_tiles=%u\n", rp.pass.size(), r.two_pass ? 2u : 1u,
               OZ2_LIMB_TILES, rp.a128_tiles);
        for (const ReconPass& p : rp.pass) {
            const Oz2ReconCT ct = oz2_recon_ct(r.mo, p.rp.data(), p.nrp, OZ2_DIGIT, p.limb3);
            uint64_t h = 1469598103934665603ull;
            for (float v : p.rp) {
                uint32_t w;
                memcpy(&w, &v, 4);
                h = (h ^ w) * 1099511628211ull;
            }
            // The role and what the stages derive from it (FINAL, out0 selection).
            const int role = (int)p.role, fin = p.role != recon_args::ROLE_TWO_FIRST,
                      pack = p.role == recon_args::ROLE_TWO_FIRST, other = p.role == recon_args::ROLE_TWO_FINAL;
            const int split = 1;  // the assembly split is unconditional
            printf("    pass nrp=%u asm=%d other=%d pack=%d split=%d limb3=%d role=%d rphash=%016" PRIx64
                   " ct=%u,%u,%u,%u,%u,%u\n",
                   p.nrp, fin, other, pack, split, (int)p.limb3, role, h, ct.S, ct.NLIMB, ct.IN_RESIDUE, ct.NRP,
                   ct.NSMALL_FRAG, ct.RAWMASK);
        }
    }
    const ReconScale sc = make_recon_scale(r.mo, 2 * oz2_shift(r));
    printf("  scale: %08x %08x %d %d\n", sc.Whi, sc.Wlo, sc.qw, sc.sh2);
}

int main(int argc, char** argv) {
    const int part = argc > 1 ? atoi(argv[1]) : 0;
    std::vector<Case> cs;
    if (part == 0 || part == 1) {
        for (uint32_t K : {32u, 64u, 128u, 512u, 1024u, 2048u, 4096u, 8192u, 14080u, 16352u, 16384u, 16416u, 32768u,
                           65536u, 65568u, 131072u, 262144u})
            cs.push_back({"auto-uniform", 64, 64, K, 0, 0, 0, 1});
        for (uint32_t K : {64u, 1024u, 4096u}) {
            cs.push_back({"auto-wide8", 64, 64, K, 0, 0, 0, 2});
            cs.push_back({"auto-wide24", 64, 64, K, 0, 0, 0, 3});
            cs.push_back({"auto-ones", 64, 64, K, 0, 0, 0, 4});
            cs.push_back({"auto-alt", 64, 64, K, 0, 0, 0, 6});
            cs.push_back({"auto-scaled", 64, 64, K, 0, 0, 0, 7});
        }
        cs.push_back({"auto-zeroA", 64, 64, 64, 0, 0, 0, 5});
        cs.push_back({"auto-one", 128, 192, 1024, 0, 0, 1, 1});
        cs.push_back({"auto-two", 128, 192, 1024, 0, 0, 2, 1});
        cs.push_back({"auto-wide-one", 128, 192, 1024, 0, 0, 1, 3});
        cs.push_back({"auto-wide-two", 128, 192, 1024, 0, 0, 2, 3});
        cs.push_back({"auto-smax4", 64, 64, 1024, 4, 0, 0, 1});
        cs.push_back({"auto-rect", 5056, 64, 32, 0, 0, 0, 1});
        cs.push_back({"auto-cube", 1024, 1024, 1024, 0, 0, 0, 1});
        cs.push_back({"auto-cube", 3520, 3520, 3520, 0, 0, 0, 1});
    }
    if (part == 0 || part == 2) {
        for (int b : {1, 2, 10, 23, 24, 30, 53, 54, 55, 60, 61, 70, 80, 81, 100})
            for (int pass : {1, 2}) cs.push_back({"fixed-b", 64, 64, 1024, 0, b, pass, 0});
        for (uint32_t s : {1u, 2u, 3u, 4u, 8u, 10u, 11u, 12u, 13u, 14u, 16u, 20u, 24u})
            cs.push_back({"fixed-smax", 64, 64, 1024, s, 54, 1, 0});
        for (uint32_t K : {32u, 64u, 96u, 512u, 2048u, 8192u, 16352u, 16384u, 16416u, 20000u, 32768u, 45024u, 65504u,
                           65536u, 65568u, 133120u, 133152u, 262144u, 1048576u, 16777216u, 1073741824u})
            for (uint32_t s : {0u, 24u}) cs.push_back({"fixed-K", 64, 64, K, s, 54, 1, 0});
        cs.push_back({"fixed-auto-pass", 64, 64, 1024, 0, 54, 0, 1});
        cs.push_back({"fixed-auto-pass-wide", 64, 64, 1024, 0, 54, 0, 3});
        cs.push_back({"fixed-rect", 5056, 64, 32, 0, 54, 1, 0});
        cs.push_back({"fixed-rect", 64, 5056, 32, 0, 54, 1, 0});
        cs.push_back({"fixed-rect", 14080, 64, 14080, 0, 54, 1, 0});
        cs.push_back({"fixed-rect", 64, 14080, 14080, 0, 54, 1, 0});
        cs.push_back({"fixed-rect", 14080, 14080, 64, 0, 54, 2, 0});
    }
    if (part == 0 || part == 3) {
        for (uint32_t n : {64u, 1024u, 3520u, 7040u, 14080u, 15136u})
            for (uint32_t s : {0u, 24u}) cs.push_back({"fixed-cube", n, n, n, s, 54, 1, 0});
        cs.push_back({"fixed-cube-two", 14080, 14080, 14080, 24, 54, 2, 0});
        cs.push_back({"fixed-cube-b70", 7040, 7040, 7040, 24, 70, 2, 0});
    }
    if (part == 0 || part == 4) {
        // The width fixed at the default cap, with and without a limit on the number of moduli.
        for (uint32_t K : {64u, 1024u, 14080u, 262144u, 1048576u, 2097152u, 4194304u})
            for (uint32_t s : {0u, 2u, 4u, 6u, 8u, 12u, 24u}) cs.push_back({"default-b", 64, 64, K, s, 54, 1, 0});
        // Automatic width with a limit on the number of moduli (K small enough for the input).
        for (uint32_t K : {64u, 1024u, 14080u})
            for (uint32_t s : {2u, 4u, 6u, 8u, 12u, 24u}) cs.push_back({"auto-smax", 64, 64, K, s, 0, 0, 1});
    }
    for (const Case& c : cs) {
        run_case(c);
        fflush(stdout);
    }
    return 0;
}
