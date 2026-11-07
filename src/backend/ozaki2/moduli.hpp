// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: selection of the moduli and of the integer width b (host side).
//
// From the inner dimension K and a limit on the number of moduli, choose pairwise coprime
// moduli m_i and the width b (bits kept of each scaled operand; default cap 54) that their
// CRT window covers. Two conditions must hold:
//   (i)  the raw sums of the matrix products are exact: |C_i| = K*r_max^2 for INT8,
//        digit products K*d^2 for the digit formats;
//   (ii) the product M of the moduli exceeds 2*max|C'| (the CRT window is wide enough).
// The per-modulus constants of the slicing and reconstruction kernels are computed here
// as well; the kernels receive them as runtime arguments (the number of moduli, the number
// of digits of A' and the number of reciprocal fragments are compile-time arguments).
// split.hpp holds the path table and the per-modulus splitting; awe.hpp selects the set
// with the stage-time model for the formats that use it; ozaki2/format.hpp provides the
// path OZ2_PARAM of the format this translation unit is compiled for. All are included
// here.
#include <algorithm>
#include <array>
#include <iterator>
#include <cstdint>
#include <quadmath.h>
#include <cmath>
#include <numeric>
#include <vector>
#include <string>
#include "ozaki2/format.hpp"
#include "ozaki2/split.hpp"
#include "ozaki2/awe.hpp"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Decomposition of the reciprocals 1/m_i used by the reconstruction (see oz2_crt_constants).
//   OZ2_RGRAN     common granularity 2^-OZ2_RGRAN of the first pass of the two-pass
//                 reconstruction. The integer part is folded per modulus, so the
//                 accumulation needs a little more than 1+RGRAN bits and fits the
//                 triple-float.
//   OZ2_NRP1/2    number of 12-bit fragments of the first / second pass.
//   OZ2_NRP0      number of fragments of the plain decomposition of the one-pass reconstruction.
// Every count is at most OZ2_MAXRP, the length of the fixed arrays in recon_crt.cpp.
#define OZ2_RGRAN 64
#define OZ2_NRP1 5
#define OZ2_NRP2 3

#define OZ2_NRP0 5
static_assert(OZ2_NRP0 <= OZ2_MAXRP && OZ2_NRP1 <= OZ2_MAXRP && OZ2_NRP2 <= OZ2_MAXRP,
              "the number of reciprocal fragments exceeds OZ2_MAXRP, the fixed array length in recon_crt.cpp");

// The window condition |C'| < M/2 is strict: with |C'| = M/2 exactly, +M/2 and -M/2 have
// the same residues for every modulus and the CRT cannot tell them apart. The comparison
// in log2 therefore carries this width. It is far larger than the rounding of the log2
// sums (about 1e-13) and far smaller than the step between usable products of moduli, so
// only the case of exact equality falls inside it.
#define OZ2_WIN_EPS 1e-9

// Largest excess over the window [bits] up to which the reselection of the last two moduli
// compares candidates with equal product counts by "fewer outputs and digit planes"
// instead of by excess (oz2_reselect_tail). One bit of excess costs about one bit of the
// maximum element-wise relative error.
#define OZ2_TAIL_EXCESS_TOL 0.25

// acc_bits of the moduli selection: the width within which the reconstruction reads the
// INT8 GEMM output C_i exactly. C_i is a sign-magnitude int32 decoded as two planes (upper
// 15 and lower 16 bits), so 31 bits. The digit formats check K*d^2 <= 2^24 instead.
#define OZ2_ACC_BITS 31

// Accuracy target [bits]: the significand width of FP64.
inline constexpr int OZ2_TARGET_BITS = 53;

// Default cap on the width b: the target plus one bit, because the scaled row/column
// maximum lies in (0.5, 1].
inline constexpr int OZ2_B_DEFAULT = 54;
static_assert(OZ2_B_DEFAULT == OZ2_TARGET_BITS + 1,
              "OZ2_B_DEFAULT is the accuracy target plus the one bit of the normalization");

// Result of the moduli selection: the set, the width, the layout of the digit planes and
// the per-modulus constants for the kernels.
struct Ozaki2Moduli {
    bool ok = false;  // a usable set was found
    int s = 0;        // number of moduli
    int nprod = 0;    // matrix products (digit products) in total: per modulus 3 for two
                      // digits (4 for WIDE), 1 for one digit, n^2 for balanced n digits
                      // (n(n+1)/2 if m divides q^n), 1 for INT8

    int nplane = 0;  // digit planes in total (per modulus: Karatsuba 3, balanced n digits
                     // n, one digit 1, otherwise 2; INT8: 1 residue plane)

    int maxg = 1;  // largest number of digit products concatenated along K into one GEMM
                   // (maximum of maxg_m); decides the layout of the slice planes

    std::vector<int> maxg_m;  // concatenation bound per modulus (1 for one-digit moduli)

    int nout = 0;             // GEMMs after concatenation (nprod products merged into nout outputs)
    bool fold_first = false;  // the composition reduces the digit products modulo m first

    bool acc_ok = true;  // the accumulation conditions hold; if not, ok is false as well and
                         // the cause is K, not the number of moduli

    // Every accumulation condition that failed, in the order of the checks (empty when all
    // hold). why is
    //   "int8"     INT8: the GEMM output bound K*floor(m/2)^2 (val) of modulus m exceeds 2^31-1
    //   "digit"    the digit-product bound K*d^2 (val) exceeds the exact FP32 integers, 2^24
    //   "one"      the GEMM output bound K*R^2 plus the modulus (val; R is d) of the one-digit
    //              modulus m exceeds 2^24
    //   "compose"  some modulus cannot be reduced first, and the sum of the 2^12-split
    //              composition (val) of modulus m reaches 2^22
    struct AccFail {
        std::string why;
        int m = 0, d = 0;
        double val = 0.0;
    };
    std::vector<AccFail> acc_fails;

    // Cap on the candidate moduli (INT8: the smaller of the path cap and the largest m with
    // K*floor(m/2)^2 <= 2^acc_bits - 1; other paths: the path cap), and the number of
    // candidates obtained before surplus moduli were dropped. The error messages use them
    // to tell "all candidates are in use" from "the limit on the number of moduli is too
    // small". For a set chosen by the stage-time model (model_set) ncand is the size of
    // that set, not a count of candidates.
    int m_cap = 0, ncand = 0;
    bool model_set = false;
    int b = 23;          // integer width: bits kept of each scaled operand
    std::vector<int> m;  // the moduli (pairwise coprime)

    std::vector<int> q;  // radix of the digits (sqrt(m) for squares, 16 or 8 for Karatsuba,
                         // near sqrt(m) for WIDE, 2d+1 for balanced digits, 1 for one digit,
                         // 0 for INT8)

    std::vector<int> mode;    // splitting method (OZ2_SPLIT_*)
    std::vector<int> poff;    // first digit plane of modulus i (running total)
    std::vector<float> rinv;  // f32(1/m_i)
    std::vector<int> yi;      // (M/m_i)^{-1} mod m_i, the CRT coefficient

    // 1/m_i split into 12-bit fragments that stay exact when multiplied by k_i (construction
    // and reasons in oz2_crt_constants). rp1 and rp2 belong to the first and second pass
    // of the two-pass reconstruction, rp0 to the one-pass reconstruction.
    //   rp1  fragments of R_i, 1/m_i rounded to the common granularity 2^-OZ2_RGRAN (the
    //        accumulation is exact)
    //   rp2  fragments of the remainder 1/m_i - R_i (either sign)
    //   rp0  plain decomposition without a common granularity; the absolute accuracy of
    //        the fraction saturates near 2^-67
    std::vector<std::array<float, OZ2_NRP1>> rp1;
    std::vector<std::array<float, OZ2_NRP2>> rp2;
    std::vector<std::array<float, OZ2_NRP0>> rp0;
    std::vector<float> c0, c1, c2, c3;  // 2^20, 2^40, 2^60, 2^80 mod m_i: weights of the digits of A' (c2 with
                                        // 4 or more digits, c3 with 5)
    std::vector<float> m16;             // 2^16 mod m_i, for decoding the int32 C_i as two planes (INT8)
    long double KMl = 0.0L;             // M * 2^-2b, the final scale (64-bit significand). The device splits
                                        // it into a 64-bit significand and an exponent and multiplies in
                                        // integers; a double constant would add a relative 2^-53 everywhere.
};

static inline float o2_f32(double x) { return (float)x; }

// Inverse of a modulo mod (a > 0, gcd(a, mod) = 1; extended Euclid).
static inline long o2_inv(long a, long mod) {
    long t = 0, nt = 1, r = mod, nr = a % mod;
    while (nr != 0) {
        const long q = r / nr;
        long tmp = t - q * nt;
        t = nt;
        nt = tmp;
        tmp = r - q * nr;
        r = nr;
        nr = tmp;
    }
    if (t < 0) t += mod;
    return t;
}

// --- Candidate tables -----------------------------------------------------------------------
// INT8: pairwise coprime integers in descending order: 255, 254 = 2*127 and the entries of
// the GEMMul8 table down to 173, then the primes 167, 163, 157, 151, OZ2_MAXS = 24 entries
// in all (the last four serve inputs that need b above 80). 256 of the GEMMul8 table is
// not usable: residues are carried as sign-magnitude UInt8, and the symmetric
// representative -128 does not fit 7 bits; 2 and 127 occur in no other entry and
// log2(254) is practically 8. 169 = 13^2 is left out: it is not coprime to 247 = 13*19.
// When K is so large that K*floor(m/2)^2 <= 2^31-1 lowers the modulus cap below 255, the
// selection appends coprime integers below 151 in descending order (oz2_pick_candidates).
static constexpr int OZ2_TABLE_I8[] = {255, 254, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217,
                                       211, 199, 197, 193, 191, 181, 179, 173, 167, 163, 157, 151};
static constexpr int OZ2_TABLE_I8_MIN = 151;  // smallest entry; the extension starts below it

// FP8 with dense digits: the FP8 table of GEMMul8 (descending, pairwise coprime, from
// 1089). The first six entries are squares (33^2, 32^2, 31^2, 29^2, 25^2, 23^2): there are
// two ways to carry a modulus with three products, one for squares and one for
// non-squares (split.hpp).
static constexpr int OZ2_TABLE_F8[] = {1089, 1024, 961, 841, 625, 529, 511, 509, 503, 499,
                                       491,  487,  481, 479, 467, 463, 461, 457, 449, 443};

// FP8 with sparse digits (sparse_max = 32): pairwise coprime squares q^2 in descending
// order; q > 33 must be odd (the digit set S_32 contains a complete residue system modulo
// q for q <= 33 and for odd q <= 49). 33^2, 45^2 and 35^2 are left out because they share
// a factor with 39^2, 25^2 or 49^2. The primes after the squares (Karatsuba) serve widths
// above the default.
static constexpr int OZ2_TABLE_F8S[] = {2401, 2209, 1849, 1681, 1521, 1369, 1024, 961, 841, 625, 529, 361,
                                        289,  121,  509,  503,  499,  491,  487,  479, 467, 463, 461, 457};

// Digit bound 32 (TF32; BF16, FP16 and BFP8 only when the stage-time model returns no
// set): squares q^2 with pairwise coprime q <= 64. A square needs 3 products and its digit
// bound is ceil(q/2) <= 32. The reselection of the last two moduli considers every usable
// integer up to the cap, so Karatsuba and WIDE moduli may still enter.
static constexpr int OZ2_TABLE_B8[] = {4096, 3969, 3721, 3481, 3025, 2809, 2209, 1849,
                                       1681, 1369, 961,  841,  529,  361,  289,  169};

// Digit bound 7 (BFP4, only when the stage-time model returns no set): squares q^2 with
// q <= 2d+1 first, then primes in descending order, carried with two balanced digits
// (4 products); radix-16 Karatsuba would need digits up to 15.
static constexpr int OZ2_TABLE_B4[] = {225, 196, 169, 121, 223, 211, 199, 197, 193, 191, 181,
                                       179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127};

// Digit bound 1 (BFP2, only when the stage-time model returns no set): moduli up to
// 3^5 = 243 carried with balanced ternary digits.
static constexpr int OZ2_TABLE_B2[] = {243, 242, 241, 239, 235, 233, 229, 227, 223, 221,
                                       217, 211, 199, 197, 193, 191, 181, 179, 173, 167};

// INT8: whether the GEMM output bound K*floor(m/2)^2 fits the magnitude of a sign-magnitude
// int32, 2^acc_bits - 1 (integer arithmetic; the residues are symmetric, |r| <= floor(m/2)).
static inline bool oz2_int8_fits(uint32_t K, int m, int acc_bits) {
    const unsigned __int128 r = (unsigned)(m / 2);
    return (unsigned __int128)K * r * r <= (((unsigned __int128)1 << acc_bits) - 1);
}

// Largest |r| with K*r^2 <= 2^acc_bits - 1.
static inline long oz2_int8_rmax(uint32_t K, int acc_bits) {
    const unsigned __int128 lim = ((unsigned __int128)1 << acc_bits) - 1;
    long r = (long)std::floor(std::sqrt((double)lim / (double)K));
    while (r > 0 && (unsigned __int128)K * r * r > lim) --r;
    while ((unsigned __int128)K * (r + 1) * (r + 1) <= lim) ++r;
    return r;
}

// --- Steps of the selection (ozaki2_choose_moduli calls each once, in this order) ----------
// Conditions of the path, fixed during the selection (smax and mod_cap after the fallback
// of FP8 to dense digits for large K).
struct Oz2SelRoute {
    uint32_t K;
    int mod_cap, acc_bits, digit_max, smax;
    Oz2Ndig ndig;

    // With digit splitting only moduli whose digits fit digit_max are usable.
    bool usable(int mm) const { return oz2_usable(mm, digit_max, ndig, smax); }
    // Matrix products of one modulus.
    int nprod(int mm) const { return digit_max ? oz2_digit_prods(mm, digit_max, ndig) : 1; }
};

// Candidate table of the path and the cap m_cap on the moduli. With digit splitting the
// GEMMs see digits (|digit| <= digit_max), not residues, so the sums are bounded by
// K*digit_max^2 regardless of the modulus; the moduli need no cap below mod_cap, and
// oz2_check_accumulation checks the sums instead.
struct Oz2Table {
    const int* t;
    int n;
};
static inline Oz2Table oz2_candidate_table(const Oz2SelRoute& R, int& m_cap) {
    auto table = [](const auto& t) { return Oz2Table{t, (int)std::size(t)}; };
    m_cap = R.mod_cap;
    // balanced n digits (BFP2)
    if (R.ndig.n > 2) return table(OZ2_TABLE_B2);
    // sparse digits (FP8)
    if (R.smax) return table(OZ2_TABLE_F8S);
    // digit bound above 16
    if (R.digit_max > 16) return table(OZ2_TABLE_B8);
    // digit bound below 16 (BFP4)
    if (R.digit_max && R.digit_max < 16) return table(OZ2_TABLE_B4);
    if (R.digit_max) return table(OZ2_TABLE_F8);
    // INT8 multiplies the residues themselves, so only moduli whose |C_i| = K*rmax^2 is
    // read back exactly are usable. The limit is 2^acc_bits - 1, not 2^acc_bits: the
    // magnitude of a sign-magnitude int32 has 31 bits. rmax is computed in integers.
    m_cap = (int)std::min<long>(R.mod_cap, 2 * oz2_int8_rmax(R.K, R.acc_bits) + 1);
    return table(OZ2_TABLE_I8);
}

// Put the initial candidates into o.m and return log2 of their product in logM. A set
// chosen by the stage-time model (awe) is used as it is, without dropping or reselecting
// the tail (return value true). Otherwise the first num_moduli usable entries of the table
// are taken, plus the extension of the INT8 table.
static inline bool oz2_pick_candidates(Ozaki2Moduli& o, double& logM, const Oz2SelRoute& R, Oz2Table tb, int m_cap,
                                       int num_moduli, int b_cap, double Wlog2, int awe, uint32_t Mdim, uint32_t Ndim) {
    bool fixed_set = false;
    logM = 0.0;
    if (awe && R.digit_max) {
        const double need_awe = 2.0 * std::min(b_cap, OZ2_MAXDIGIT * OZ2_DIGIT_BITS) + 1.0 + Wlog2 + OZ2_WIN_EPS;
        const std::vector<int> ms = oz2_awe_choose(R.K, awe, m_cap, R.digit_max, R.ndig, R.smax, need_awe,
                                                   OZ2_TAIL_EXCESS_TOL, std::min(num_moduli, OZ2_MAXS), Mdim, Ndim);
        if (!ms.empty()) {
            for (int mm : ms) {
                o.m.push_back(mm);
                logM += std::log2((double)mm);
            }
            fixed_set = true;
        }
    }
    if (!fixed_set)
        for (int t = 0; t < tb.n; ++t) {
            int mm = tb.t[t];
            if ((int)o.m.size() >= num_moduli) break;
            if (mm > m_cap || !R.usable(mm)) continue;
            o.m.push_back(mm);
            logM += std::log2((double)mm);
        }
    // Extension of the INT8 table: when K is large and the table entries up to m_cap are
    // fewer than requested, append integers below 151 in descending order that are coprime
    // to everything taken so far. If the window is then exceeded, oz2_trim_tail drops
    // moduli from the end (the extension side).
    if (!fixed_set && !R.digit_max) {
        const int want = std::min(num_moduli, OZ2_MAXS);
        for (int mm = std::min(m_cap, OZ2_TABLE_I8_MIN - 1); mm >= 3 && (int)o.m.size() < want; --mm) {
            bool cop = true;
            for (int x : o.m)
                if (std::gcd(x, mm) != 1) {
                    cop = false;
                    break;
                }
            if (!cop) continue;
            o.m.push_back(mm);
            logM += std::log2((double)mm);
        }
    }
    o.m_cap = m_cap;
    o.ncand = (int)o.m.size();
    o.model_set = fixed_set;
    o.s = (int)o.m.size();
    return fixed_set;
}

// Width b from the CRT window. The scaled operands satisfy |a~|, |b~| <= 1, so
//     |C'| <= K * 2^(2b) < M/2
// is sufficient; hence b = floor((log2 M - 1 - log2 K)/2), and the fraction C'/M then uses
// almost the whole range up to 1/2. Leaving window unused and overflowing the window both
// lose accuracy. The window condition is strict (OZ2_WIN_EPS), so the bound on b is taken
// stricter by EPS as well; with exact equality b drops by one instead of failing.
// Two caps apply: OZ2_MAXDIGIT*OZ2_DIGIT_BITS = 100, the width that A' with 5 radix-2^20
// digits can carry, and the caller's b_cap (default 54). 54 is a design decision
// ("FP64-equivalent"), not lossless: for inputs whose exponents spread within a row the
// captured information keeps growing with b, which is what the automatic precision
// (auto_precision.hpp) exploits by passing a larger b_cap.
static inline int oz2_width_from_window(double logM, double Wlog2, int b_cap) {
    const double room = logM - 1.0 - Wlog2 - OZ2_WIN_EPS;
    const int b_win = (int)std::floor(room / 2.0);
    return std::min(std::min(b_win, OZ2_MAXDIGIT * OZ2_DIGIT_BITS), b_cap);
}

// When b is capped, the product of the requested number of moduli exceeds the window. The
// surplus does not add accuracy (it only makes the fraction C'/M smaller), so moduli are
// dropped from the end while the window stays covered; each modulus costs matrix products.
static inline void oz2_trim_tail(std::vector<int>& m, double& logM, double need_log2) {
    while (m.size() > 2) {
        double l2 = std::log2((double)m.back());
        if (logM - l2 < need_log2) break;
        m.pop_back();
        logM -= l2;
    }
}

// Reselect the last two moduli: replace them by one modulus or a coprime pair that covers
// the window with as little excess as possible. The table has steps of several bits per
// modulus, and the excess would be lost from the accuracy of the fraction, because the
// reconstruction solves C'/M to a fixed absolute accuracy.
// Order of preference (lexicographic, see `better`): fewer matrix products in the tail;
// excess within OZ2_TAIL_EXCESS_TOL; fewer GEMM outputs and digit planes; smaller excess.
// The product count comes first because with digit splitting a modulus costs 3 or 4
// products, and minimizing the excess alone can pick 4-product moduli. For INT8 every
// modulus is one product, so a single modulus that suffices wins.
static inline void oz2_reselect_tail(std::vector<int>& m, double& logM, double need_log2, const Oz2SelRoute& R,
                                     int m_cap) {
    if (!(m.size() >= 2 && logM > need_log2)) return;
    const uint32_t K = R.K;
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    std::vector<int> head(m.begin(), m.end() - 2);
    double base = 0.0;
    for (int x : head) base += std::log2((double)x);
    auto cop_head = [&](int mm) {
        for (int x : head)
            if (std::gcd(mm, x) != 1) return false;
        return true;
    };
    double best = logM;
    int bp = m[m.size() - 2], bq = m.back();
    // Candidates with equal product counts and an excess within the tolerance are compared
    // by the number of GEMM outputs after concatenation (oz2_nout), then by digit planes:
    // fewer outputs mean fewer GEMM launches and composition inputs, fewer planes less
    // slice output. The excess is a loss of accuracy, hence the tolerance. The
    // concatenation bound is estimated with the modulus cap m_cap (the bound of the final
    // set, which uses its largest modulus, is at least this).
    auto maxg_est_of = [&](int mm) { return digit_max ? oz2_maxg_for(mm, K, m_cap, digit_max, ndig, smax) : 1; };
    auto layout_cost_raw = [&](int mm) {
        if (!digit_max) return 0;
        return 1000 * (int)oz2_nout(mm, digit_max, ndig, maxg_est_of(mm)) + oz2_digit_planes(mm, digit_max, ndig);
    };
    // The per-candidate values are looked up many times in the pair loop; tabulate them
    // for the usable moduli in 3..m_cap. Every modulus looked up is one of those: the
    // current tail comes from the candidate table (usable entries up to m_cap, and for
    // INT8 the extension down to 3), and the loops below test usable_c first.
    std::vector<char> use_c(m_cap + 1, 0);
    std::vector<int> np_c(m_cap + 1, 0), lay_c(m_cap + 1, 0);
    for (int mm = 3; mm <= m_cap; ++mm)
        if (R.usable(mm)) {
            use_c[mm] = 1;
            np_c[mm] = R.nprod(mm);
            lay_c[mm] = layout_cost_raw(mm);
        }
    auto layout_cost = [&](int mm) { return lay_c[mm]; };
    auto usable_c = [&](int mm) { return mm >= 3 && mm <= m_cap && use_c[mm]; };
    auto nprod_c = [&](int mm) { return np_c[mm]; };
    auto over_tol = [&](double l) { return (l - need_log2) > OZ2_TAIL_EXCESS_TOL ? 1 : 0; };
    int bcost = nprod_c(bp) + nprod_c(bq);         // matrix products of the current tail
    int blay = layout_cost(bp) + layout_cost(bq);  // outputs and digit planes of the current tail
    // Replace when (products, excess over tolerance, outputs and planes, logM) is smaller.
    auto better = [&](int c, int lay, double l) {
        if (c != bcost) return c < bcost;
        if (over_tol(l) != over_tol(best)) return over_tol(l) < over_tol(best);
        return lay < blay || (lay == blay && l < best);
    };
    // If the kept head contains an even modulus, only odd candidates can be coprime to it,
    // so the search steps by 2. With an all-odd (short) head even candidates are tried too.
    // The search starts at 3 because small moduli have fine steps in log2 and approach the
    // required amount closely.
    bool head_even = false;
    for (int x : head) head_even = head_even || (x % 2 == 0);
    const int pstep = head_even ? 2 : 1;
    for (int p2 = 3; p2 <= m_cap; p2 += pstep) {
        if (!usable_c(p2) || !cop_head(p2)) continue;
        double lp = base + std::log2((double)p2);
        if (lp >= need_log2 && better(nprod_c(p2), layout_cost(p2), lp)) {
            best = lp;
            bp = p2;
            bq = 0;
            bcost = nprod_c(p2);
            blay = layout_cost(p2);
        }
        // (p, q) and (q, p) are the same pair, so q runs above p2 only.
        for (int q = p2 + pstep; q <= m_cap; q += pstep) {
            if (!usable_c(q) || std::gcd(p2, q) != 1 || !cop_head(q)) continue;
            double lq = lp + std::log2((double)q);
            const int c = nprod_c(p2) + nprod_c(q), lay = layout_cost(p2) + layout_cost(q);
            if (lq >= need_log2 && better(c, lay, lq)) {
                best = lq;
                bp = p2;
                bq = q;
                bcost = c;
                blay = lay;
            }
        }
    }
    m = head;
    m.push_back(bp);
    if (bq) m.push_back(bq);
    logM = best;
}

// Record a failed accumulation condition (clears ok and acc_ok).
static inline void oz2_acc_fail(Ozaki2Moduli& o, const char* why, int mm, int dd, double v) {
    o.acc_fails.push_back({why, mm, dd, v});
    o.ok = o.acc_ok = false;
}

// Accumulation conditions (those of the composition stage are in oz2_plan_concat). Every
// failed condition is recorded in acc_fails so that the error message can name the cause.
//   INT8: for each modulus the GEMM output bound K*floor(m/2)^2 must fit the magnitude
//     2^acc_bits - 1 of a sign-magnitude int32. This repeats the condition behind m_cap in
//     integer arithmetic for every selected modulus, whatever step selected it.
//   Digit formats: the GEMM accumulates digit products in FP32, so |sum| <= K*d^2 must
//     stay within the exact integers, 2^24 (2^24 itself is representable). d is the path
//     bound digit_max, or, on the paths with sparse digits or one-digit moduli, the largest
//     per-plane bound (oz2_plane_bound) of the selected set.
//   One-digit moduli bypass the composition: the reconstruction reduces the GEMM output P
//     (|P| <= K*R^2, R = oz2_resid_bound) modulo m with one ModRound. The residue class is
//     preserved only if the rounded quotient times the modulus is exact in FP32, which
//     needs |P| + m <= 2^24 (the same margin as the reduce-first composition).
static inline void oz2_check_accumulation(Ozaki2Moduli& o, const Oz2SelRoute& R) {
    const uint32_t K = R.K;
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    if (!digit_max)
        for (int mm : o.m)
            if (!oz2_int8_fits(K, mm, R.acc_bits))
                oz2_acc_fail(o, "int8", mm, 0, (double)K * (double)(mm / 2) * (double)(mm / 2));
    auto pbound = [&](int mm, int plane) { return oz2_plane_bound(mm, plane, digit_max, ndig, smax); };
    int deff = digit_max;
    if (smax || ndig.one) {
        deff = 0;
        for (int mm : o.m)
            for (int pl = 0; pl < oz2_digit_planes(mm, digit_max, ndig); ++pl) deff = std::max(deff, pbound(mm, pl));
    }
    if (digit_max && (double)K * deff * deff > std::ldexp(1.0, 24))
        oz2_acc_fail(o, "digit", 0, deff, (double)K * deff * deff);
    if (digit_max && ndig.one)
        for (int mm : o.m)
            if (oz2_split_mode(mm, digit_max, ndig, smax) == OZ2_SPLIT_ONE) {
                const double Rb = (double)oz2_resid_bound(mm);
                if ((double)K * Rb * Rb + mm > std::ldexp(1.0, 24))
                    oz2_acc_fail(o, "one", mm, (int)Rb, (double)K * Rb * Rb + mm);
            }
}

// Concatenation bounds (maxg_m per modulus, maxg overall) and the method and conditions of
// the composition stage (kernels/compose.cpp). The stage has two methods; in both the sum
// that enters the last ModRound must stay below 2^22, the range in which ModRound returns
// the correct symmetric residue. The coefficients differ widely between moduli, so the
// sums are evaluated with the actual coefficients.
//   fold_first  reduce each GEMM output P_j modulo m first, then apply the coefficient
//               c_j. This needs P_j plus the modulus within the exact FP32 integers, 2^24
//               (P_j < 2^24 alone is not enough: the rounded quotient times the modulus
//               reaches |P_j| + m/2, and if that is rounded the residue class changes).
//               After the reduction |t_j| <= m/2+2, so the sum is (sum |c_j|)(m/2+2).
//   2^12 split  used only when the above fails: P_j is split into a high part h_j and a
//               low part l_j with radix 2^12 (OZ2_COMPOSE_SPLIT) and the coefficients
//               A_j = c_j*2^12 mod m, B_j = c_j mod m are applied. h_j is rounded to
//               nearest, so |h_j| <= floor(Pmax/2^12 + 1/2) with Pmax the bound of one
//               output (oz2_group_bound), and |l_j| <= 2^11.
// Concatenation (oz2_product_groups in split.hpp): g digit products are merged while
// g*K*d^2 plus the modulus stays within 2^24. If the merged outputs break the conditions
// of the composition, the check is repeated without concatenation (maxg = 1). On the
// paths with sparse digits or one-digit moduli the bound is per modulus (oz2_maxg_for with
// the per-plane digit bounds); one-digit moduli are never concatenated.
static inline void oz2_plan_concat(Ozaki2Moduli& o, const Oz2SelRoute& R) {
    const uint32_t K = R.K;
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    int mmax = 0;
    for (int mm : o.m) mmax = std::max(mmax, mm);
    o.maxg_m.assign(o.s, 1);
    if (digit_max)
        for (int i = 0; i < o.s; ++i) {
            // A one-digit modulus has a single product and nothing to concatenate; keep 1
            // (a larger value would make it decide maxg and switch the slice planes to
            // the concatenated layout for nothing).
            if (oz2_split_mode(o.m[i], digit_max, ndig, smax) == OZ2_SPLIT_ONE) continue;

            o.maxg_m[i] = oz2_maxg_for(o.m[i], K, mmax, digit_max, ndig, smax);
        }
    auto sync_maxg = [&]() {
        o.maxg = 1;
        for (int g : o.maxg_m) o.maxg = std::max(o.maxg, g);
    };
    sync_maxg();
    if (!digit_max) return;
    auto group_bound = [&](int mm, const Oz2ProductGroup& g) {
        return oz2_group_bound(mm, g, (double)K, digit_max, ndig, smax);
    };
    // Sum bound of modulus i: (sum |c|)(m/2+2) with fold_first, the bound of the 2^12 split
    // otherwise. One term per GEMM output after concatenation.
    auto compose_sum = [&](int i, bool fold) {
        const int mm = o.m[i];
        Oz2ProductGroup gr[OZ2_MAXPROD];
        const uint32_t ng = oz2_product_groups(mm, gr, digit_max, ndig, o.maxg_m[i]);
        double sum = 0;
        // The coefficients are in [0, m) or symmetric representatives; count magnitudes.
        for (uint32_t j = 0; j < ng; ++j)
            sum += fold ? std::fabs((double)gr[j].coeff_lo) * ((double)mm / 2.0 + 2.0)
                        : std::floor(group_bound(mm, gr[j]) / (double)OZ2_COMPOSE_SPLIT + 0.5) *
                                  std::fabs((double)gr[j].coeff_hi) +
                              (double)(OZ2_COMPOSE_SPLIT / 2) * std::fabs((double)gr[j].coeff_lo);
        return sum;
    };
    // Largest bound of one output of modulus i.
    auto gbound_of = [&](int i) {
        Oz2ProductGroup gr[OZ2_MAXPROD];
        const uint32_t ng = oz2_product_groups(o.m[i], gr, digit_max, ndig, o.maxg_m[i]);
        double gmax = 0;
        for (uint32_t j = 0; j < ng; ++j) gmax = std::max(gmax, group_bound(o.m[i], gr[j]));
        return gmax;
    };
    // Check with concatenation; if that fails, drop the concatenation (all maxg_m = 1) and
    // check once more. The second attempt has maxg = 1 and cannot retry.
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool can_fold = true, grouped_ok = true;
        for (int i = 0; i < o.s; ++i) {
            const int mm = o.m[i];
            if (oz2_nout(mm, digit_max, ndig, o.maxg_m[i]) == 0) grouped_ok = false;
            if (gbound_of(i) + mm > std::ldexp(1.0, 24) || compose_sum(i, true) >= std::ldexp(1.0, 22))
                can_fold = false;
        }
        if (o.maxg > 1 && (!can_fold || !grouped_ok)) {
            o.maxg_m.assign(o.s, 1);
            sync_maxg();
            continue;
        }
        o.fold_first = can_fold;
        if (!can_fold)
            for (int i = 0; i < o.s; ++i)
                if (const double cs = compose_sum(i, false); cs >= std::ldexp(1.0, 22))
                    oz2_acc_fail(o, "compose", o.m[i], 0, cs);
        break;
    }
}

// Per modulus: radix and method of the digit splitting and the first digit plane; in
// total: matrix products, digit planes and GEMM outputs after concatenation. The method
// records sparse digits (SQS) separately: the slicer needs it, while products and planes
// equal those of SQ.
static inline void oz2_layout(Ozaki2Moduli& o, const Oz2SelRoute& R) {
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    o.nprod = 0;
    o.nplane = 0;
    o.nout = 0;
    for (int i = 0; i < o.s; ++i) {
        const int mm = o.m[i];
        int qq = digit_max ? oz2_digit_base(mm, digit_max, ndig) : 0;
        int md = digit_max ? oz2_split_mode(mm, digit_max, ndig, smax) : OZ2_SPLIT_SQ;
        o.q.push_back(qq);
        o.mode.push_back(md);
        o.poff.push_back(o.nplane);
        o.nplane += digit_max ? oz2_digit_planes(mm, digit_max, ndig) : 1;
        o.nprod += R.nprod(mm);
        o.nout += digit_max ? (int)oz2_nout(mm, digit_max, ndig, o.maxg_m[i]) : 1;
    }
}

// Per-modulus constants for the CRT and for the slicing and reconstruction kernels, at
// width o.b.
static inline void oz2_crt_constants(Ozaki2Moduli& o) {
    const int b = o.b;
    // CRT coefficient y_i = (prod_{j != i} m_j)^{-1} mod m_i. The product is reduced
    // modulo m_i at every step, so no multi-precision integer is needed.
    for (int i = 0; i < o.s; ++i) {
        long P = 1, mi = o.m[i];
        for (int j = 0; j < o.s; ++j)
            if (j != i) P = (P * (o.m[j] % mi)) % mi;
        o.yi.push_back((int)o2_inv(P, mi));
    }
    // KM = M * 2^-2b in long double: with b = 54 the product M is about 2^125 and, with the
    // excess, overflows even a signed __int128.
    long double Mld = 1.0L;
    for (int mm : o.m) Mld *= (long double)mm;
    o.KMl = Mld * std::pow(2.0L, -2.0 * b);

    // Split 1/m_i into 12-bit fragments that stay exact when multiplied by k_i.
    // |k_i| <= oz2_crt_k_bound(m_i) = floor(m_i/2) + 1 and the largest modulus cap is 4096,
    // so |k_i| <= 2049 (12 bits); with 12-bit fragments the product has 24 bits, exactly the
    // significand of FP32. There is no margin: a modulus cap above 4096 or wider fragments
    // break this.
    // Two decompositions exist (the two passes of recon_crt.cpp):
    //   rp1  fragments of R_i, 1/m_i rounded to the common absolute granularity 2^-OZ2_RGRAN.
    //        R_i is a multiple of the granularity and k_i is an integer, so k_i*R_i and
    //        the sum are multiples as well, and the accumulation is exact as long as the
    //        sum fits the 72 bits of the triple-float.
    //   rp2  fragments of the remainder 1/m_i - R_i (R_i is rounded to nearest, so either
    //        sign; magnitude at most 2^-RGRAN). The values are small, so accumulating
    //        them in the same triple-float has a negligible absolute error.
    // The remainders are kept in __float128: with long double (64-bit significand) the
    // remainder is known to 2^-64 only when the third fragment is made.
    const float SPL = 4097.0f;
    auto top12 = [&](__float128 v) {  // the top 12 bits of v, as FP32
        const float p = (float)(double)v;
        if (p == 0.0f) return 0.0f;
        const float c = o2_f32((double)SPL * (double)p);
        return o2_f32((double)c - (double)o2_f32((double)c - (double)p));
    };
    for (int mm : o.m) {
        const __float128 r = (__float128)1 / (__float128)mm;
        const __float128 gran = powq((__float128)2, -(__float128)OZ2_RGRAN);
        const __float128 Rg = roundq(r / gran) * gran;  // R_i
        std::array<float, OZ2_NRP1> p1{};
        std::array<float, OZ2_NRP2> p2{};
        // Every fragment of rp1 has 12 bits: k_i times a fragment must be exact, or the first
        // pass loses the "multiple of the common granularity" property.
        __float128 v = Rg;
        for (int t = 0; t < OZ2_NRP1; ++t) {
            p1[t] = top12(v);
            v -= (__float128)p1[t];
        }
        const __float128 sum1 = Rg - v;  // what the fragments actually carry
        // rp2 is built from r - sum(rp1), not from r - R: if the fragments of rp1 did not
        // exhaust R, the rest moves to the second pass and the two passes still add up to
        // r. sum(rp1) is R without its low part, hence still a multiple of the granularity.
        // The last fragment of rp2 and of rp0 is a full FP32 value (24 bits) to gain width;
        // multiplying it by k rounds, but at that position the loss is negligible.
        v = r - sum1;
        for (int t = 0; t < OZ2_NRP2; ++t) {
            p2[t] = (t + 1 < OZ2_NRP2) ? top12(v) : (float)(double)v;
            v -= (__float128)p2[t];
        }
        std::array<float, OZ2_NRP0> p0{};  // plain decomposition
        v = r;
        for (int t = 0; t < OZ2_NRP0; ++t) {
            p0[t] = (t + 1 < OZ2_NRP0) ? top12(v) : (float)(double)v;
            v -= (__float128)p0[t];
        }
        o.rp1.push_back(p1);
        o.rp2.push_back(p2);
        o.rp0.push_back(p0);
        o.rinv.push_back((float)(double)r);  // approximation of 1/m for the residue computation
    }
    // Constants of the slicer: A' is made on the device as 3 to 5 radix-2^20 digits
    // (kernels/digits.h), which needs 2^20 mod m and 2^40 mod m (and 2^60, 2^80 mod m with
    // 4 or 5 digits). The weights are kept in [0, m).
    auto pow2mod = [](int e, long mod) {  // 2^e mod mod (e >= 0)
        long r = 1 % mod, base = 2 % mod;
        for (int i = 0; i < e; ++i) r = (r * base) % mod;
        return (float)r;
    };
    for (int mm : o.m) {
        o.c0.push_back(pow2mod(OZ2_DIGIT_BITS, mm));
        o.c1.push_back(pow2mod(2 * OZ2_DIGIT_BITS, mm));
        o.c2.push_back(pow2mod(3 * OZ2_DIGIT_BITS, mm));
        o.c3.push_back(pow2mod(4 * OZ2_DIGIT_BITS, mm));
        o.m16.push_back(pow2mod(16, mm));
    }
}

// Select the moduli and the width on the path of this format (OZ2_PARAM: the largest
// modulus mod_cap, the digit bound, the shape of the digits, the sparse digit bound and
// the row of the stage-time model; OZ2_ACC_BITS: the width within which |C_i| = K*rmax^2
// is read back exactly, INT8 only).
//   M, N        output dimensions; only the DRAM estimate of the stage-time model uses them
//   K           inner dimension
//   num_moduli  limit on the number of moduli (each modulus costs matrix products)
//   b_cap       cap on the width b (default 54; the automatic precision passes the value
//               it derived from the input). The structural cap is 100.
// With a model row (OZ2_PARAM.awe nonzero) the set is the one the model search finds for
// the window of b = b_cap (awe.hpp: a heuristic with cutoffs, so not necessarily the set
// that minimizes the model); if the search returns no set, the table selection is used.
// The CRT window is the worst case: |C'| <= K*2^(2b) for |a~|, |b~| <= 1, i.e. log2 K
// bits on top of 2b+1.
static inline Ozaki2Moduli ozaki2_choose_moduli(uint32_t M, uint32_t N, uint32_t K, int num_moduli, int b_cap) {
    const int digit_max = OZ2_PARAM.digit_bound;
    const Oz2Ndig ndig = OZ2_PARAM.nd();
    int mod_cap = OZ2_PARAM.mod_cap, smax = OZ2_PARAM.sparse_max;
    const double Wlog2 = std::log2((double)K);
    // Sparse digits (FP8, |digit| <= 32) are used only while the digit product K*smax^2
    // plus the modulus stays within the exact FP32 integers, 2^24 (K <= 16381 for
    // smax = 32). A larger K uses dense digits (|digit| <= digit_max) with the table
    // OZ2_TABLE_F8 and the modulus cap (2*digit_max+1)^2 = 1089.
    if (smax && digit_max && (double)K * smax * smax + mod_cap > std::ldexp(1.0, 24)) {
        smax = 0;
        mod_cap = std::min(mod_cap, (2 * digit_max + 1) * (2 * digit_max + 1));
    }
    Ozaki2Moduli o;
    const Oz2SelRoute R{K, mod_cap, OZ2_ACC_BITS, digit_max, smax, ndig};
    int m_cap = 0;
    const Oz2Table tb = oz2_candidate_table(R, m_cap);
    double logM = 0.0;
    const bool fixed_set = oz2_pick_candidates(o, logM, R, tb, m_cap, num_moduli, b_cap, Wlog2, OZ2_PARAM.awe, M, N);
    o.b = oz2_width_from_window(logM, Wlog2, b_cap);
    const int b = o.b;
    // No safety margin is added to the required amount: 2b+1+log2(K) is exactly what the
    // worst case |C'| <= K*2^2b needs, so the CRT result is unique once it is met. An
    // excess is not "safe" but a loss of accuracy: the reconstruction solves the fraction
    // C'/M to a fixed absolute accuracy, and a larger M makes the fraction smaller.
    // OZ2_WIN_EPS only expresses the strict inequality in the log2 comparison.
    const double need_log2 = 2.0 * b + 1.0 + Wlog2 + OZ2_WIN_EPS;
    // A set chosen by the model is neither trimmed nor reselected.
    if (!fixed_set) {
        oz2_trim_tail(o.m, logM, need_log2);
        oz2_reselect_tail(o.m, logM, need_log2, R, m_cap);
    }
    o.s = (int)o.m.size();
    // Even all candidates may fall short of need_log2 (paths with a small modulus cap, or
    // a large K). Going on would give a plausible but wrong result without any error, so
    // it is detected here. OZ2_MAXS is the array length of the kernels.
    o.ok = (logM >= need_log2) && o.s > 0 && o.s <= OZ2_MAXS && o.s <= num_moduli && b > 0;
    // The CRT needs pairwise coprime moduli (otherwise y_i does not exist). The table and
    // the reselection ensure it; a violation would silently give wrong results, so it is
    // checked exhaustively (s <= 24).
    for (int i = 0; i < o.s; ++i)
        for (int j = i + 1; j < o.s; ++j)
            if (std::gcd(o.m[i], o.m[j]) != 1) o.ok = false;
    // The accumulation conditions are evaluated even for an unusable set: the error
    // message reads them.
    oz2_check_accumulation(o, R);
    oz2_plan_concat(o, R);
    oz2_layout(o, R);
    oz2_crt_constants(o);
    return o;
}

// Number of limbs of the reconstruction: 3 (triple-float, FP64-equivalent) or 2. One pass
// uses 3 when b >= 24. With two passes the writer of the first pass always reads 3 limbs
// to add them to the 128-bit fixed point, so both passes use 3 regardless of b.
static inline bool oz2_recon_limb3(const Ozaki2Moduli& mo, bool two_pass) { return two_pass || mo.b >= 24; }

// Compile-time argument RAWMASK of recon_crt.cpp (C_RAWMASK in recon_args.h): bit i is set
// for a one-digit modulus, whose residue plane holds the unreduced GEMM output; the kernel
// reduces those planes modulo m first.
static inline uint32_t oz2_recon_rawmask(const Ozaki2Moduli& mo) {
    uint32_t raw = 0;
    for (int i = 0; i < mo.s; i++)
        if (mo.mode[i] == OZ2_SPLIT_ONE) raw |= 1u << i;
    return raw;
}

// Bound floor(m/2) + 1 on |k_i| = |ModRound(r_i*y_i)| in recon_crt.cpp. r_i is a composed
// residue (|r_i| <= m/2) or a one-digit residue reduced by the reconstruction
// (|r_i| <= floor(m/2)+2, m <= 65), and 0 <= y_i < m, so |r_i*y_i| < 2^23 for m <= 4096.
// That can exceed 2^22, the range in which ModRound guarantees the symmetric residue; the
// overshoot is below 1, i.e. k_i exceeds floor(m/2) by 1 for some odd moduli.
static inline long oz2_crt_k_bound(int m) { return m / 2 + 1; }

// --- Number of reciprocal fragments added with tfAddSmall -------------------------------------
// The accumulation in recon_crt.cpp adds the last NSMALL_FRAG fragments with tfAddSmall,
// which skips the first TwoSum of the full triple-float addition. Its error bound assumes
// |v| <= eps for the added value; this function chooses eps and returns the count.
// eps is relative to the accumulator of the pass: the largest term
// vmax = max_{t,i} |k_i * RP[t][i]|, capped at 1 because the accumulator holds a fraction
// (|a0| <= 1). An absolute threshold would fail in the second pass of the two-pass
// reconstruction, whose fragments are what the first pass left (about 2^-60): every fragment
// would count as small and the top limb would never be used. With 7 bits of margin over
// ulp = 2^-23, eps = vmax * 2^-30.
// rp holds the fragments as [i*nrp + t] (modulus i, fragment t); m are the moduli.
static inline uint32_t oz2_choose_small_tail(const float* rp, uint32_t nrp, const int* m, uint32_t s) {
    if (nrp > 8) return 0;  // unexpected; 0 fragments is the safe side
    float worst[8] = {0};   // largest term per fragment
    float vmax = 0.f;       // largest term of the pass
    for (uint32_t t = 0; t < nrp; t++) {
        for (uint32_t i = 0; i < s; i++) {
            // term bound: |RP[t][i]| * max |k_i|
            const float v = std::fabs(rp[(size_t)i * nrp + t]) * (float)oz2_crt_k_bound(m[i]);
            if (v > worst[t]) worst[t] = v;
        }
        if (worst[t] > vmax) vmax = worst[t];
    }
    const float scale = vmax < 1.f ? vmax : 1.f;
    const float eps = (float)std::ldexp((double)scale, -30);
    uint32_t n = 0;  // trailing fragments added with tfAddSmall
    for (uint32_t t = nrp; t-- > 0;) {
        if (worst[t] <= eps)
            n++;
        else
            break;
    }
    return n;
}

// Compile-time arguments of the reconstruction compute kernel (recon_crt.cpp) that follow
// from the moduli set and the pass: C_S, C_NLIMB, C_IN_RESIDUE, C_NRP, C_NSMALL_FRAG,
// C_RAWMASK of recon_args.h. rp are the reciprocal fragments of the pass ([i*nrp + t]),
// limb3 comes from oz2_recon_limb3.
struct Oz2ReconCT {
    uint32_t S, NLIMB, IN_RESIDUE, NRP, NSMALL_FRAG, RAWMASK;
};
static inline Oz2ReconCT oz2_recon_ct(const Ozaki2Moduli& mo, const float* rp, uint32_t nrp, bool in_residue,
                                      bool limb3) {
    return {(uint32_t)mo.s,
            limb3 ? 3u : 2u,
            in_residue ? 1u : 0u,
            nrp,
            oz2_choose_small_tail(rp, nrp, mo.m.data(), (uint32_t)mo.s),
            oz2_recon_rawmask(mo)};
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
