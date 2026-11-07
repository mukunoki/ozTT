// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: the per-format path table and the way one modulus is carried
// (host side).
//
// A residue modulo m does not fit the exact integer range of most GEMM formats, so the
// slicing stage splits it into small balanced digits, the GEMM stage multiplies digit
// planes, and the composition stage adds the digit partial products with weights and
// reduces modulo m. This header provides
//   - the path of each format (OZ2_ROUTES, oz2_param): modulus cap, digit bound, ...;
//   - the splitting method of a modulus (OZ2_SPLIT_*), its digit bounds, digit planes and
//     digit products (oz2_product_terms);
//   - the grouping of digit products of equal weight into one GEMM by concatenation along
//     K (oz2_product_groups) and the bound on how many may be concatenated (oz2_maxg_for);
//   - the plane count of the composition stage and the DRAM estimate of the stage-time
//     model (oz2_compose_plan, oz2_dram_one_group).
// moduli.hpp (moduli selection, CRT constants) and awe.hpp (stage-time model) include it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "common/format.hpp"
#include "common/dram_margin.hpp"
#include "ozaki2/kernels/digit_limits.h"

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// --- Paths: how each format takes its moduli ---------------------------------------

// Shape of the digits of a residue (Oz2Route::nd()); the per-modulus functions take it.
//   n    number of digits (2 or 5); also the compile-time argument NRDIG of the slicer.
//   one  carry a small modulus whose residues all fit one digit (|r| <= d; m <= 65 for
//        d = 32) as a single digit plane (OZ2_SPLIT_ONE). Such moduli sit next to moduli
//        whose digit bound is larger, so the concatenation bound and the accumulation
//        conditions are evaluated per digit plane (oz2_plane_bound) instead of with the
//        path-wide digit bound.
struct Oz2Ndig {
    int n = 2;
    bool one = false;
    constexpr Oz2Ndig(int n_ = 2, bool one_ = false) : n(n_), one(one_) {}
};
// The scheme II path of one format.
struct Oz2Route {
    int mod_cap;             // largest modulus
    int digit_bound;         // bound d on |digit| actually used (0: no digit splitting, the INT8 path)
    int nrdig;               // digits of a residue: 2, or 5 (balanced ternary of BFP2; per modulus the
                             // smallest count up to 5)
    uint32_t def_s;          // limit on the number of moduli when the evaluator fixes the width b and
                             // the caller gives no max_moduli (choose_moduli_and_width in plan.hpp)
    int sparse_max = 0;      // bound of the sparse digits (0: unused). Up to digit_bound every integer
                             // may be a digit; beyond it, up to sparse_max, only the integers the format
                             // represents exactly (FP8/e4m3: the even integers in 16..32). FP8 only.
    int awe = 0;             // 0: take the moduli from the table and reselect the last two;
                             // otherwise the row (OZ2_AWE_*) of the stage-time model that selects the set
    bool one_digit = false;  // mix in one-digit moduli (Oz2Ndig::one)
    constexpr Oz2Ndig nd() const { return Oz2Ndig(nrdig, one_digit); }
};

// Row numbers of the stage-time model (rows of OZ2_AWE_COEF in awe.hpp, 1-based). New
// numbers go before OZ2_AWE_END: the coefficient table is declared with OZ2_AWE_COUNT rows.
enum {
    OZ2_AWE_NONE = 0,
    OZ2_AWE_BFP4 = 1,
    OZ2_AWE_BFP2 = 2,
    OZ2_AWE_BF16 = 3,
    OZ2_AWE_FP16 = 4,
    OZ2_AWE_BFP8 = 5,
    OZ2_AWE_END
};
inline constexpr int OZ2_AWE_COUNT = OZ2_AWE_END - 1;
// INT8: residues are carried as they are (sign-magnitude UInt8), so the moduli are at most 255.
inline constexpr Oz2Route OZ2_ROUTE_INT8 = {255, 0, 2, 16u};

// FP8 (e4m3): digits in S_32 = {0..16} and the even integers {18..32} (and negatives) reach
// moduli up to 49^2 = 2401. With digits limited to [-16,16] the cap is 33^2 = 1089; a large
// K that cannot use the sparse digits falls back to that (ozaki2_choose_moduli).
inline constexpr Oz2Route OZ2_ROUTE_FP8 = {2401, 16, 2, 14u, 32};

// Digit bound 32 with the table selection (TF32).
inline constexpr Oz2Route OZ2_ROUTE_D32 = {4096, 32, 2, 11u};

// Digit bound 32 with the stage-time model (BF16, FP16, BFP8): squares, Karatsuba and wide
// moduli mixed with one-digit moduli (m <= 65); the concatenation bound is per modulus.
inline constexpr Oz2Route OZ2_ROUTE_BF16A = {4096, 32, 2, 24u, 0, OZ2_AWE_BF16, true};
inline constexpr Oz2Route OZ2_ROUTE_FP16A = {4096, 32, 2, 24u, 0, OZ2_AWE_FP16, true};
inline constexpr Oz2Route OZ2_ROUTE_BFP8A = {4096, 32, 2, 24u, 0, OZ2_AWE_BFP8, true};

// BFP4: squares (up to 225), radix-8 Karatsuba (up to 113), two balanced digits (up to 225)
// and three balanced digits (up to 15^3 = 3375), chosen per modulus; the set is selected
// with the stage-time model.
inline constexpr Oz2Route OZ2_ROUTE_BFP4 = {3375, 7, 2, 24u, 0, OZ2_AWE_BFP4};

// BFP2: balanced ternary with n digits (n = 1..5, moduli up to 3^n), the smallest n per
// modulus; the set is selected with the stage-time model.
inline constexpr Oz2Route OZ2_ROUTE_BFP2 = {243, 1, 5, 24u, 0, OZ2_AWE_BFP2};

// Format -> path. The digit bound is not the exact integer range of the format alone: the
// accumulator limits it as well (one GEMM adds K digit products, which must stay exact).
struct Oz2RouteRow {
    OzFmt fmt;
    Oz2Route r;
};
inline constexpr Oz2RouteRow OZ2_ROUTES[] = {
    // No digit splitting. With 8-bit moduli the same window needs 16 moduli.
    {OzFmt::INT8, OZ2_ROUTE_INT8},
    // Dense digits are bounded by the format (16); the sparse digit set of e4m3 reaches the
    // squares of odd q <= 49 (method (E) below). When K*32^2 plus the modulus exceeds 2^24
    // (K > 16381) the path returns to dense digits and moduli up to 1089.
    {OzFmt::FP8, OZ2_ROUTE_FP8},
    // Stored as FP32 and multiplied as TF32. Same digit bound 32 as BF16, but the moduli
    // come from the table (no one-digit moduli, no stage-time model). 4 bytes per element,
    // so large problems are processed in output row bands (plan_bands in plan.hpp).
    {OzFmt::FP32, OZ2_ROUTE_D32},
    // BF16 represents integers up to 256 exactly, so the accumulator sets the digit bound
    // (d <= 32 for K = 14080), as for BFP8; only the carried format differs.
    {OzFmt::BF16, OZ2_ROUTE_BF16A},
    // BFP8 represents +-127 exactly; the accumulator sets the digit bound: one GEMM adds K
    // digit products, so 2*log2(d) + ceil(log2 K) <= 24, i.e. d <= 32 for K = 14080. With
    // d = 32 two balanced digits reach moduli up to 4 d^2 = 4096. The composition reduces
    // the digit products modulo m first (fold_first, oz2_plan_concat in moduli.hpp).
    {OzFmt::BFP8, OZ2_ROUTE_BFP8A},
    // BFP4 represents integers up to 7 exactly, so here the format sets the digit bound;
    // the methods per modulus are (A), (B) with radix 8, (C) and (D) with three digits
    // (OZ2_ROUTE_BFP4 above).
    {OzFmt::BFP4, OZ2_ROUTE_BFP4},
    // BFP2 represents only +-1, so two digits reach (2d+1)^2 = 9 and the window is out of
    // reach; every modulus is carried with balanced ternary digits, method (D)
    // (OZ2_ROUTE_BFP2 above).
    {OzFmt::BFP2, OZ2_ROUTE_BFP2},
    // The splitting rules and the digit bound of BF16; the set is selected with its own
    // row of the stage-time model, so it can differ from the BF16 set. The digits are
    // written as binary16 bit patterns.
    {OzFmt::FP16, OZ2_ROUTE_FP16A},
};

// Path of a format (an all-zero path for a format missing from the table; excluded below).
constexpr Oz2Route oz2_param(OzFmt f) {
    for (const Oz2RouteRow& x : OZ2_ROUTES)
        if (x.fmt == f) return x.r;
    return {};
}

// Every format appears exactly once.
constexpr bool oz2_routes_cover() {
    for (OzFmt f : OZ_FMTS) {
        int n = 0;
        for (const Oz2RouteRow& x : OZ2_ROUTES) n += (x.fmt == f);
        if (n != 1) return false;
    }
    return sizeof(OZ2_ROUTES) / sizeof(OZ2_ROUTES[0]) == sizeof(OZ_FMTS) / sizeof(OZ_FMTS[0]);
}
static_assert(oz2_routes_cover(), "a format is missing from OZ2_ROUTES or appears twice");

// --- Splitting methods (A)-(F) ---------------------------------------------------------
// A residue r is split into digits and r_a*r_b is assembled from GEMMs of digit planes.
// The digits stay in the set the path allows: |digit| <= d for the dense sets, plus the
// even integers up to 2d for the sparse set of FP8 (E). The text uses FP8 (d = 16); BF16,
// FP16, TF32 and BFP8 have d = 32 (BF16, FP16, BFP8 also use (F)), BFP4 has d = 7 ((B) with
// radix 8, (D) with 3 digits), BFP2 has d = 1 and uses (D).
//
// (A) OZ2_SPLIT_SQ    square modulus m = q^2: balanced digits by rounding to nearest;
//       2 planes, 3 products.
//         h = round(r/q),  l = r - q*h      (|l| <= q/2, |h| <= round((m/2)/q))
//       q^2 = 0 (mod m), so the high*high term vanishes:
//         r_a*r_b = q*(h_a l_b + l_a h_b) + l_a l_b   (mod m)
// (B) OZ2_SPLIT_KAR   non-square modulus: Karatsuba with radix B (16 for paths with
//       d >= 16, d+1 = 8 for BFP4; oz2_kar_base); 3 planes, 3 products. For B = 16:
//         h = sign(r)*ceil(|r|/16),  l = r - 16*h,  s = h + l
//       Rounding up makes h and l have opposite signs, so |h|, |l|, |s| all stay within
//       16 (rounding to nearest lets |h+l| reach 24, outside the exact integers of e4m3).
//         r_a*r_b = 256*HH + 16*(SS - HH - LL) + LL
//       The digits fit when ceil((m/2)/16) <= 16, i.e. m <= 513.
// (C) OZ2_SPLIT_WIDE  non-square modulus above 513: balanced digits by rounding to nearest
//       with q ~ sqrt(m); 2 planes, 4 products (high*high does not vanish):
//         r_a*r_b = q^2*HH + q*(HL+LH) + LL   (mod m)
//       Appears through the reselection of the last two moduli and the stage-time model;
//       without it the reselection candidates shrink to m <= 513 and one more modulus is
//       needed.
// (D) OZ2_SPLIT_BAL   balanced n digits (n = 1..5; the slicer receives n per modulus):
//       r = sum_j x_j q^j with q = 2d+1 and |x_j| <= d; n planes and n^2 products
//       (n(n+1)/2 when m divides q^n, because the terms with i+j >= n vanish). The
//       composition is D = sum_{i,j} q^(i+j) P_ij. BFP2 carries every modulus this way
//       with the smallest n (oz2_bal_ndig); BFP4 carries 225 < m <= 3375 with 3 digits.
// (E) OZ2_SPLIT_SQS   square modulus m = q^2 with sparse digits (odd q, 33 < q <= |S|;
//       q <= 49 for the set S_32 of FP8); 2 planes, 3 products, products and composition
//       as in (A). Only the digits differ: they are taken from
//       S = {|x| <= 16} + {even x : 16 < |x| <= 32}, the integers e4m3 represents exactly
//       (covering the residues with a non-contiguous set of exactly representable
//       integers follows arXiv:2608.06812). |S| = 49, and for odd q <= 49 S contains a complete residue system modulo q (the
//       odd residues 17..q-18 are covered by negated even values).
//         h0 = round(r/q),  t = r - q*h0                       (as (A); |t| <= (q-1)/2)
//         if t is odd and |t| >= 17:   l = t - sgn(t)*q, h1 = h0 + sgn(t)   (r = l + q*h1)
//         otherwise                    l = t, h1 = h0
//         if h1 is odd and |h1| >= 17: h = h1 - sgn(h1)*q   (q*h = q*h1 mod q^2)
//         otherwise                    h = h1
//       An odd |t| >= 17 becomes the even value q - |t| <= q - 17 <= 32, likewise h, so
//       l, h are in S and r = l + q*h (mod q^2). The per-modulus digit bounds are counted
//       by oz2_sqs_bounds.
// (F) OZ2_SPLIT_ONE   one digit (paths with Oz2Ndig::one): a small modulus whose residues
//       all fit the digit bound (|r| <= d) is one digit plane and one product. Its
//       composition would be a single term with coefficient 1, so the GEMM output goes
//       straight to the residue plane and the reconstruction reduces it modulo m (RAWMASK).
enum {
    OZ2_SPLIT_SQ = 0,
    OZ2_SPLIT_KAR = 1,
    OZ2_SPLIT_WIDE = 2,
    OZ2_SPLIT_BAL = 3,
    OZ2_SPLIT_SQS = 4,
    OZ2_SPLIT_ONE = 5
};

// Digit-making mode of the slicer kernel (Oz2XMode in digit_limits.h) for a splitting
// method. (A) and (C) are the same procedure for the slicer. An unknown method returns
// OZ2_XMODE_COUNT so that the caller stops instead of falling into a default branch.
constexpr Oz2XMode oz2_xmode(int split_mode) {
    switch (split_mode) {
    case OZ2_SPLIT_SQ:
    case OZ2_SPLIT_WIDE: return OZ2_XMODE_RND;
    case OZ2_SPLIT_KAR: return OZ2_XMODE_KAR;
    case OZ2_SPLIT_SQS: return OZ2_XMODE_SQS;
    case OZ2_SPLIT_BAL: return OZ2_XMODE_BAL;
    case OZ2_SPLIT_ONE: return OZ2_XMODE_ONE;
    }
    return OZ2_XMODE_COUNT;
}
static_assert(oz2_xmode(OZ2_SPLIT_SQ) == OZ2_XMODE_RND && oz2_xmode(OZ2_SPLIT_WIDE) == OZ2_XMODE_RND &&
                  oz2_xmode(OZ2_SPLIT_KAR) == OZ2_XMODE_KAR && oz2_xmode(OZ2_SPLIT_SQS) == OZ2_XMODE_SQS &&
                  oz2_xmode(OZ2_SPLIT_BAL) == OZ2_XMODE_BAL && oz2_xmode(OZ2_SPLIT_ONE) == OZ2_XMODE_ONE,
              "mapping from the splitting method of a modulus to the digit-making mode of the slicer");

// --- Sparse digits ((E), e4m3 of FP8) ----------------------------------------------------
// Whether e4m3 represents the integer v exactly (at most 4 significant bits).
static inline bool oz2_e4m3_int(long v) {
    if (v < 0) v = -v;
    if (v == 0) return true;
    int e = 0;
    for (long x = v; x; x >>= 1) ++e;
    return e <= 4 || (v % (1L << (e - 4))) == 0;
}

// Digit set S_smax = { x exactly representable in e4m3 : |x| <= smax }.
static inline bool oz2_in_sparse(long x, int smax) { return (x < 0 ? -x : x) <= smax && oz2_e4m3_int(x); }

// Whether S_smax contains a complete residue system modulo q (q <= 2*smax+1 is necessary).
static inline bool oz2_sparse_covers_count(int q, int smax) {
    std::vector<char> hit(q, 0);
    int n = 0;
    for (int x = -smax; x <= smax; ++x)
        if (oz2_in_sparse(x, smax)) {
            const int r = ((x % q) + q) % q;
            if (!hit[r]) {
                hit[r] = 1;
                ++n;
            }
        }
    return n == q;
}
// The moduli selection asks this many times; for smax, q <= 128 the answers come from a
// table built once (on first use, thread-safe).
static inline bool oz2_sparse_covers(int q, int smax) {
    if (q <= 0 || smax <= 0) return false;
    if (q > 2 * smax + 1) return false;
    if (smax <= 128 && q <= 128) {
        static const std::vector<char> table = [] {  // [smax*129 + q]
            std::vector<char> t((size_t)129 * 129, 0);
            for (int s = 1; s <= 128; ++s)
                for (int qq = 1; qq <= std::min(128, 2 * s + 1); ++qq)
                    t[(size_t)s * 129 + qq] = oz2_sparse_covers_count(qq, s);
            return t;
        }();
        return table[(size_t)smax * 129 + q] != 0;
    }
    return oz2_sparse_covers_count(q, smax);
}

// The digits of (E) in integer arithmetic. r is a residue modulo q^2 (|r| <= floor(q^2/2)+1
// is assumed), q is odd.
static inline void oz2_sqs_split(long r, int q, long& l, long& h) {
    // h0 = round(r/q); q is odd, so r/q is never a half-integer and there are no ties.
    long h0 = (2 * r + (r >= 0 ? q : -q)) / (2 * q);
    long t = r - (long)q * h0;
    const long st = t > 0 ? 1 : (t < 0 ? -1 : 0);
    if ((t & 1) && (t >= 17 || t <= -17)) {
        t -= st * q;
        h0 += st;
    }
    const long sh = h0 > 0 ? 1 : (h0 < 0 ? -1 : 0);
    if ((h0 & 1) && (h0 >= 17 || h0 <= -17)) h0 -= sh * q;
    l = t;
    h = h0;
}

// Digit bounds of (E): the largest |l| and |h| over all |r| <= floor(m/2)+1 (the last
// ModRound of the slicer may exceed the symmetric range by 1 for large moduli).
static inline void oz2_sqs_bounds_count(int q, int& dl, int& dh) {
    const long m = (long)q * q, R = m / 2 + 1;
    dl = dh = 0;
    for (long r = -R; r <= R; ++r) {
        long l, h;
        oz2_sqs_split(r, q, l, h);
        dl = std::max<int>(dl, (int)(l < 0 ? -l : l));
        dh = std::max<int>(dh, (int)(h < 0 ? -h : h));
    }
}
// The moduli reselection asks for the same q many times; for q <= 64 the bounds come from
// a table built once (on first use, thread-safe).
static inline void oz2_sqs_bounds(int q, int& dl, int& dh) {
    if (q > 0 && q <= 64) {
        static const std::vector<std::pair<int, int>> table = [] {
            std::vector<std::pair<int, int>> t(65, {0, 0});
            for (int qq = 1; qq <= 64; ++qq) oz2_sqs_bounds_count(qq, t[qq].first, t[qq].second);
            return t;
        }();
        dl = table[q].first;
        dh = table[q].second;
        return;
    }
    oz2_sqs_bounds_count(q, dl, dh);
}

// --- Digit bounds per method ---------------------------------------------------------------
// Largest modulus that ndig balanced digits reach: q^ndig with q = 2*dmax+1.
static inline long oz2_bal_cap(int dmax, int ndig) {
    long q = 2L * dmax + 1, c = 1;
    for (int i = 0; i < ndig; ++i) c *= q;
    return c;
}

// Analytic bound on |r| returned by the last ModRound of the slicer (oz2_residue in
// kernels/digits.h): |r| <= m/2 + |S|*2^-23, where S is the input of that ModRound. With
// 3 digits |S| <= D0 + floor(m/2)(c0+c1); with 4 digits the sum is reduced once before the
// d3 term, so |S| <= m/2+2 + floor(m/2)*c2 (c_j = 2^(20(j+1)) mod m); 5 digits have the same
// last input as 4. D0 = 2^19 + 2^16 + 1 bounds the lowest digit d0: each digit-making step
// rounds only the high limb of a triple-float T, and the part below it is at most 2^-4
// (both the high limb and the rest are below 2^20), so the next step sees
// |T'| <= 2^20 (1/2 + 2^-4) = 2^19 + 2^16 and its rounded digit is at most 2^19 + 2^16 + 1.
// Used where a digit bound is counted from this value (radix-B Karatsuba, the digit count
// of balanced n digits, one-digit moduli, the per-plane bounds of oz2_plane_bound);
// oz2_digit_bound_rnd and oz2_digit_bound_kar count with floor(m/2). The analysis is for
// radix-2^20 digits.
static_assert(OZ2_DIGIT_BITS == 20, "D0 and the residue bound of oz2_resid_bound are derived for radix-2^20 digits");
static inline long oz2_resid_bound(int m) {
    long c0 = 1 % m, c1, c2;
    for (int i = 0; i < OZ2_DIGIT_BITS; ++i) c0 = (c0 * 2) % m;
    c1 = (c0 * c0) % m;
    c2 = (c1 * c0) % m;
    const double u = std::ldexp(1.0, -23) * (1.0 + std::ldexp(1.0, -23));
    const double D0 = 524288.0 + 65536.0 + 1.0;
    const double S3 = D0 + (double)(m / 2) * (double)(c0 + c1);
    const double S4 = m / 2.0 + 2.0 + (double)(m / 2) * (double)c2;
    const double e = std::max(S3, S4) * u;
    return std::min<long>((long)std::floor(m / 2.0 + e), m / 2 + 1);
}

// Number of balanced digits of a modulus: the smallest n in 1..nmax with m <= q^n and
// |r| <= (q^n-1)/2; nmax+1 if none.
static inline int oz2_bal_ndig(int m, int dmax, int nmax) {
    const long R = oz2_resid_bound(m);
    for (int n = 1; n <= nmax; ++n) {
        const long c = oz2_bal_cap(dmax, n);
        if (m <= c && 2 * R + 1 <= c) return n;
    }
    return nmax + 1;
}

// Digit bound of methods (A)/(C) with radix q.
static inline int oz2_digit_bound_rnd(int m, int q) {
    int rs = m / 2;            // bound on the symmetric residue
    int h = (rs + q / 2) / q;  // bound on round(rs/q)
    return std::max(h, q / 2);
}
// Digit bound of method (B) with radix 16.
static inline int oz2_digit_bound_kar(int m) {
    int rs = m / 2;
    int h = (rs + 15) / 16;  // |h| <= ceil((m/2)/16)
    // |l| <= 15 for every m (h is rounded up, so l has the opposite sign and stays below
    // 16), and |s| = |h+l| <= max(|h|,|l|) for the same reason. For small m |l| exceeds
    // |h|, hence the max.
    return std::max(h, 15);
}

// Radix-B Karatsuba (rounding up): |h| <= ceil(R/B), |l| <= B-1, |s| <= max of the two,
// with R = oz2_resid_bound. Paths with digits narrower than 15 (BFP4, d = 7) use
// B = d+1 = 8 (m <= 2d(d+1)+1 = 113); the others use the radix-16 bound above.
static inline int oz2_kar_base(int dmax) { return (dmax && dmax < 15) ? dmax + 1 : 16; }
static inline int oz2_digit_bound_kar_b(int m, int B) {
    const long R = oz2_resid_bound(m);
    return std::max<int>((int)((R + B - 1) / B), B - 1);
}
// floor(sqrt(m))
static inline int oz2_isqrt(int m) {
    int r = (int)std::sqrt((double)m);
    while ((long)(r + 1) * (r + 1) <= m) ++r;
    while ((long)r * r > m) --r;
    return r;
}
// sqrt(m) if m is a square, otherwise 0.
static inline int oz2_is_square(int m) {
    const int r = oz2_isqrt(m);
    return ((long)r * r == m) ? r : 0;
}

// Radix of the balanced digits by rounding to nearest: the q near sqrt(m) that minimizes
// the digit bound.
static inline int oz2_digit_base_rnd(int m) {
    const int r = oz2_isqrt(m);
    int best = r + 1, bb = oz2_digit_bound_rnd(m, r + 1);
    for (int qq = r; qq <= r + 3; ++qq) {
        int v = oz2_digit_bound_rnd(m, qq);
        if (v < bb) {
            bb = v;
            best = qq;
        }
    }
    return best;
}

// Splitting method of modulus m. dmax is the digit bound of the path. With a nonzero smax
// (sparse digit bound) a square of odd q that dense digits cannot carry becomes (E);
// its products, planes and composition equal those of (A), so the functions that count
// them (oz2_product_terms, ...) do not take smax and treat it as (A). Only oz2_usable,
// oz2_plane_bound and the mode passed to the slicer distinguish (E).
// Radix-16 Karatsuba has digits up to 15, so paths with narrower digits (BFP4, d = 7)
// cannot use it; there a square whose digits do not fit is handled like a non-square:
// radix d+1 Karatsuba (m <= 113), then two balanced digits (up to 225), then three
// balanced digits (radix 2d+1 = 15, m <= 3375).
static inline int oz2_split_mode(int m, int dmax = 0, Oz2Ndig ndig = {}, int smax = 0) {
    if (ndig.n > 2) return OZ2_SPLIT_BAL;
    if (ndig.one && dmax && oz2_resid_bound(m) <= dmax) return OZ2_SPLIT_ONE;
    const bool narrow = dmax && dmax < 15;
    if (const int q = oz2_is_square(m)) {
        if (smax && dmax && oz2_digit_bound_rnd(m, q) > dmax && (q & 1) && oz2_sparse_covers(q, smax))
            return OZ2_SPLIT_SQS;
        if (!narrow || oz2_digit_bound_rnd(m, q) <= dmax) return OZ2_SPLIT_SQ;
    }
    if (narrow) {
        if (oz2_digit_bound_kar_b(m, oz2_kar_base(dmax)) <= dmax) return OZ2_SPLIT_KAR;
        if (oz2_digit_bound_rnd(m, oz2_digit_base_rnd(m)) <= dmax) return OZ2_SPLIT_WIDE;
        if (oz2_bal_ndig(m, dmax, 3) <= 3) return OZ2_SPLIT_BAL;
        return OZ2_SPLIT_WIDE;  // not carriable (oz2_usable rejects it)
    }
    const int kcap = dmax ? std::min(16, dmax) : 16;
    if (oz2_digit_bound_kar(m) <= kcap) return OZ2_SPLIT_KAR;
    return OZ2_SPLIT_WIDE;
}

// Number of balanced digits of a modulus carried by (D) (per modulus for BFP2, 3 on the
// two-digit paths); 0 for the other methods.
static inline int oz2_bal_digits(int m, int dmax, Oz2Ndig ndig) {
    if (ndig.n > 2) return oz2_bal_ndig(m, dmax, ndig.n);
    return oz2_split_mode(m, dmax, ndig) == OZ2_SPLIT_BAL ? 3 : 0;
}
// Radix of the digits of modulus m (1 for a one-digit modulus; 0 marks the INT8 path).
static inline int oz2_digit_base(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    if (ndig.n > 2) return 2 * dmax + 1;
    const int mode = oz2_split_mode(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) return 1;
    if (mode == OZ2_SPLIT_BAL) return 2 * dmax + 1;
    if (mode == OZ2_SPLIT_SQ) return oz2_is_square(m);
    return (mode == OZ2_SPLIT_KAR) ? oz2_kar_base(dmax) : oz2_digit_base_rnd(m);
}
// Bound on the dense digits of modulus m. A balanced-digit modulus that needs more digits
// than the path allows returns dmax+1 so that oz2_usable rejects it.
static inline int oz2_digit_bound(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    if (ndig.n > 2) return (oz2_bal_ndig(m, dmax, ndig.n) <= ndig.n) ? dmax : dmax + 1;
    int mode = oz2_split_mode(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) return (int)oz2_resid_bound(m);
    if (mode == OZ2_SPLIT_BAL) return (oz2_bal_ndig(m, dmax, 3) <= 3) ? dmax : dmax + 1;
    if (mode == OZ2_SPLIT_KAR)
        return (dmax && dmax < 15) ? oz2_digit_bound_kar_b(m, oz2_kar_base(dmax)) : oz2_digit_bound_kar(m);
    return oz2_digit_bound_rnd(m, oz2_digit_base(m, dmax, ndig));
}

// Whether the path can carry modulus m: its dense digits fit dmax, or it is carried with
// sparse digits (E). Always true without digit splitting (dmax = 0; the caller checks the
// modulus cap).
static inline bool oz2_usable(int m, int dmax, Oz2Ndig ndig, int smax) {
    if (!dmax) return true;
    if (oz2_split_mode(m, dmax, ndig, smax) == OZ2_SPLIT_SQS) return true;
    return oz2_digit_bound(m, dmax, ndig) <= dmax;
}

// Bound on |digit| in digit plane `plane` of modulus m (0 = l, 1 = h, 2 = s). It feeds the
// bounds on sums of digit products (concatenation bound, composition and accumulation
// conditions). Paths without sparse digits and without one-digit moduli return the
// path-wide dmax. The sparse path (FP8) is evaluated per modulus: the counted maxima of l
// and h for (E), oz2_digit_bound (<= 16) otherwise.
static inline int oz2_plane_bound(int m, int plane, int dmax, Oz2Ndig ndig, int smax) {
    // Paths that mix in one-digit moduli are evaluated per digit plane, with the residue
    // bound oz2_resid_bound: R for one digit; q/2 (low) and round(R/q) (high) for squares
    // and wide moduli; 15, ceil(R/16) and their maximum for Karatsuba.
    if (ndig.one && !smax) {
        const int mode = oz2_split_mode(m, dmax, ndig), R = (int)oz2_resid_bound(m);
        if (mode == OZ2_SPLIT_ONE) return R;
        if (mode == OZ2_SPLIT_KAR) {
            const int h = (R + 15) / 16;
            return plane == 0 ? 15 : (plane == 1 ? h : std::max(h, 15));
        }
        const int q = oz2_digit_base(m, dmax, ndig);
        return plane == 0 ? q / 2 : (R + q / 2) / q;
    }
    if (!smax) return dmax;
    if (oz2_split_mode(m, dmax, ndig, smax) == OZ2_SPLIT_SQS) {
        int dl, dh;
        oz2_sqs_bounds(oz2_is_square(m), dl, dh);
        return plane == 0 ? dl : dh;
    }
    return oz2_digit_bound(m, dmax, ndig);
}

// Number of digit planes that carry modulus m.
static inline int oz2_digit_planes(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    if (const int n = oz2_bal_digits(m, dmax, ndig)) return n;
    const int mode = oz2_split_mode(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) return 1;
    return mode == OZ2_SPLIT_KAR ? 3 : 2;
}

// --- Digit products and composition coefficients ----------------------------------------
// Product j multiplies digit plane a_plane of A by digit plane b_plane of B (0 = l, 1 = h,
// 2 = s = h+l; relative to the first plane poff[] of the modulus); the composition
// multiplies it by coeff_lo and adds. coeff_hi is the same coefficient times 2^12
// (OZ2_COMPOSE_SPLIT), for the composition that takes the residue in two steps.
//   SQ/WIDE          (l,l), (h,l), (l,h)  (+ (h,h) for WIDE)
//   KAR              (l,l), (h,h), (s,s)
//   balanced n digits all (i,j); the terms with i+j >= n vanish when m divides q^n
// The planes and the coefficients come from this single enumeration; oz2_digit_prods and
// oz2_product_groups derive from it.
struct Oz2ProductTerm {
    int a_plane = 0, b_plane = 0;
    float coeff_lo = 0, coeff_hi = 0;  // both reduced modulo m
};

// Radix of the 2^12 split of the composition stage: the host coefficient
// coeff_hi = c*2^12 mod m and the kernel's hi = round(P/2^12) use the same value.
inline constexpr long OZ2_COMPOSE_SPLIT = 1L << OZ2_COMPOSE_SPLIT_BITS;

static inline uint32_t oz2_product_terms(int m, Oz2ProductTerm t[OZ2_MAXPROD], int dmax = 0, Oz2Ndig ndig = {}) {
    auto rm = [&](long v) {
        long r = v % m;
        return r < 0 ? r + m : r;
    };
    uint32_t k = 0;
    // Symmetric representative in (-m/2, m/2]: used for the coefficients of balanced n
    // digits and of Karatsuba with digits narrower than 15, to keep the sum of the
    // reduce-first composition, sum |c_j| (m/2+2), below 2^22. The other paths keep the
    // coefficients in [0, m).
    auto sm = [&](long v) {
        long r = rm(v);
        return r > m / 2 ? r - m : r;
    };
    if (const int n = oz2_bal_digits(m, dmax, ndig)) {  // balanced n digits: coefficients q^(i+j) mod m
        const long q = 2L * dmax + 1;
        const bool drop = (oz2_bal_cap(dmax, n) % m == 0);
        // A modulus that needs more digits than the path allows (oz2_bal_ndig returned
        // nmax+1; oz2_usable rejects it, so it is never selected) would write past t.
        const int nterm = drop ? n * (n + 1) / 2 : n * n;
        if (nterm > OZ2_MAXPROD)
            throw std::runtime_error("internal error: modulus " + std::to_string(m) + " with " + std::to_string(n) +
                                     " balanced digits needs " + std::to_string(nterm) +
                                     " digit products, more than the limit " + std::to_string(OZ2_MAXPROD));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (drop && i + j >= n) continue;
                long c = 1;
                for (int e = 0; e < i + j; ++e) c = rm(c * q);
                t[k++] = {i, j, (float)sm(c), (float)sm(c * OZ2_COMPOSE_SPLIT)};
            }
        return k;
    }
    const int mode = oz2_split_mode(m, dmax, ndig), q = oz2_digit_base(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) {  // one digit: (r,r) with coefficient 1
        t[k++] = {0, 0, 1.f, (float)rm(OZ2_COMPOSE_SPLIT)};
        return k;
    }
    if (mode == OZ2_SPLIT_KAR) {  // coefficients 1-B, B^2-B, B (-15, 240, 16 for radix 16)
        const long B = q;
        if (dmax && dmax < 15) {  // digits narrower than 15 (BFP4, radix 8): symmetric representatives
            const long c[3] = {sm(1 - B), sm(B * B - B), sm(B)};
            for (int j = 0; j < 3; ++j) t[k++] = {j, j, (float)c[j], (float)sm(c[j] * OZ2_COMPOSE_SPLIT)};
            return k;
        }
        const long c[3] = {rm(1 - B), rm(B * B - B), rm(B)};
        for (int j = 0; j < 3; ++j) t[k++] = {j, j, (float)c[j], (float)rm(c[j] * OZ2_COMPOSE_SPLIT)};
        return k;
    }
    // SQ/WIDE: coefficients 1, q, q [, q^2]
    const long c[4] = {rm(1), rm(q), rm(q), rm((long)q * q)};
    const int pa[4] = {0, 1, 0, 1}, pb[4] = {0, 0, 1, 1};
    const uint32_t n = (mode == OZ2_SPLIT_WIDE) ? 4u : 3u;
    for (uint32_t j = 0; j < n; ++j) t[k++] = {pa[j], pb[j], (float)c[j], (float)rm(c[j] * OZ2_COMPOSE_SPLIT)};
    return k;
}

// Number of digit products of modulus m: the length of the enumeration above (derived from
// it so that the count and the enumeration cannot diverge).
static inline int oz2_digit_prods(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    Oz2ProductTerm t[OZ2_MAXPROD];
    return (int)oz2_product_terms(m, t, dmax, ndig);
}

// --- Concatenating digit products of equal weight along K into one GEMM ------------------
// The composition is D = sum_j c_j P_j (mod m) and c_j depends only on the weight i+j of
// the digit product (q^(i+j) for squares, wide moduli and balanced n digits). Products of
// equal weight may be added before the composition, and their sum
//     sum_{i+j=k} A_i B_j = [A_{i0} | A_{i0+1} | ...] * [B_{k-i0} ; B_{k-i0-1} ; ...]
// is one product of the digit planes of A placed side by side along K and the digit
// planes of B stacked along K in reverse order. The MAC count of the GEMM stage is the
// same; the number of outputs (and of planes the composition reads) drops.
// The slice writer lays the planes out accordingly (kernels/modslice_writer.cpp): within
// a modulus, A is concatenated along K in digit order (tile of row r, digit plane p,
// column c at r*(P*Kt) + p*Kt + c) and B is stacked in reverse (digit j at position
// np-1-j). Then the A digits i0..i0+g-1 of a group are contiguous and so are the B
// positions. The three Karatsuba products (weights 0, 2, 4) have distinct coefficients and
// are never grouped.
// Exactness: one output is the sum of g digit products, |sum| <= g*K*d^2, and it must stay
// an exact FP32 integer even after the modulus is added (so that the composition can
// reduce it first): g <= maxg, see oz2_maxg_for.
struct Oz2ProductGroup {
    int a0 = 0, b0 = 0, g = 1;  // digit planes a0..a0+g-1 of A (from poff of the modulus), positions b0..b0+g-1 of B
    float coeff_lo = 0, coeff_hi = 0;  // coefficient of the weight and its 2^12 multiple (both reduced modulo m)
};
// Groups of modulus m with at most maxg products per group; returns their number (0 if
// products of equal weight had different coefficients, which does not happen).
static inline uint32_t oz2_product_groups(int m, Oz2ProductGroup gr[OZ2_MAXPROD], int dmax, Oz2Ndig ndig, int maxg) {
    Oz2ProductTerm t[OZ2_MAXPROD];
    const uint32_t n = oz2_product_terms(m, t, dmax, ndig);
    const int np = oz2_digit_planes(m, dmax, ndig);
    uint32_t k = 0;
    // Per weight key = a+b: sort by a, then cut into runs of at most maxg consecutive a.
    for (int key = 0; key <= 2 * (np - 1); ++key) {
        int as[OZ2_MAXPROD];
        float lo = 0, hi = 0;
        int na = 0;
        for (uint32_t j = 0; j < n; ++j)
            if (t[j].a_plane + t[j].b_plane == key) {
                if (na && (t[j].coeff_lo != lo || t[j].coeff_hi != hi)) return 0;
                as[na++] = t[j].a_plane;
                lo = t[j].coeff_lo;
                hi = t[j].coeff_hi;
            }
        std::sort(as, as + na);
        for (int i = 0; i < na;) {
            int g = 1;
            while (i + g < na && g < maxg && as[i + g] == as[i] + g) ++g;
            gr[k].a0 = as[i];
            gr[k].b0 = np - 1 - (key - as[i]);
            gr[k].g = g;
            gr[k].coeff_lo = lo;
            gr[k].coeff_hi = hi;
            ++k;
            i += g;
        }
    }
    return k;
}
// Number of GEMM outputs of modulus m after concatenation.
static inline uint32_t oz2_nout(int m, int dmax, Oz2Ndig ndig, int maxg) {
    Oz2ProductGroup gr[OZ2_MAXPROD];
    return oz2_product_groups(m, gr, dmax, ndig, maxg);
}

// Bound on |output| of one group (the sum of g digit products): sum_t K*d(a0+t)*d(b), with
// the per-plane digit bounds (g*K*dmax^2 on paths where every plane has bound dmax). The B
// position b0+t holds digit np-1-(b0+t) because B is stacked in reverse.
static inline double oz2_group_bound(int m, const Oz2ProductGroup& g, double K, int dmax, Oz2Ndig ndig, int smax) {
    const int np = oz2_digit_planes(m, dmax, ndig);
    double sb = 0;
    for (int t = 0; t < g.g; ++t)
        sb += K * oz2_plane_bound(m, g.a0 + t, dmax, ndig, smax) *
              oz2_plane_bound(m, np - 1 - (g.b0 + t), dmax, ndig, smax);
    return sb;
}

// Concatenation bound: the largest g >= 1 such that g*K*d_l*d_h plus the modulus `cap`
// stays within the exact FP32 integers (2^24), which leaves room for the composition to
// reduce modulo m first. d_l, d_h are the digit bounds of the low and high plane (with
// two digits only the weight-1 products l*h are grouped). The caller chooses `cap`: the
// modulus cap while candidates are still being compared, the largest selected modulus for
// the final set. One-digit moduli are not concatenated; the caller uses 1 for them.
static inline int oz2_maxg_for(int m, uint32_t K, int cap, int dmax, Oz2Ndig ndig, int smax) {
    const double pb = (double)oz2_plane_bound(m, 0, dmax, ndig, smax) *
                      oz2_plane_bound(m, std::min(1, oz2_digit_planes(m, dmax, ndig) - 1), dmax, ndig, smax);
    return std::max(1, (int)std::floor((std::ldexp(1.0, 24) - cap) / ((double)K * pb)));
}

// --- Planes of the composition stage and the DRAM estimate of the stage-time model ---------
// The composition stage reads the GEMM outputs of one modulus at a time, so the number of
// product planes is the largest output count mx of a modulus (one-digit moduli bypass the
// stage and do not count; no composed modulus means no planes). The minimum
// OZ2_COMPOSE_MIN keeps the plane count and the circular-buffer depth of the stage the
// same when concatenation leaves 2 outputs. When mx exceeds OZ2_COMPOSE_NOSPLIT_MAX
// (balanced 5 digits without concatenation: 25 outputs per modulus), the outputs are
// processed in chunks of OZ2_COMPOSE_CHUNK and planes for the partial sums are added.
inline constexpr uint32_t OZ2_COMPOSE_MIN = 3;
inline constexpr uint32_t OZ2_COMPOSE_NOSPLIT_MAX = 12;
inline constexpr uint32_t OZ2_COMPOSE_CHUNK = 5;  // a common divisor of 15 and 25
struct Oz2ComposePlan {
    uint32_t chunk = 0;   // outputs composed at once (0: nothing is composed)
    uint32_t planes = 0;  // planes to allocate (chunk product planes + partial sums)
};
// mx: largest output count of a composed modulus.
constexpr Oz2ComposePlan oz2_compose_plan(bool composes, uint32_t mx) {
    Oz2ComposePlan p;
    if (!composes) return p;
    const uint32_t max_nout = mx > OZ2_COMPOSE_MIN ? mx : OZ2_COMPOSE_MIN;
    p.chunk = max_nout > OZ2_COMPOSE_NOSPLIT_MAX ? OZ2_COMPOSE_CHUNK : max_nout;
    p.planes = p.chunk < max_nout ? p.chunk + (max_nout + p.chunk - 1) / p.chunk : max_nout;
    return p;
}

// DRAM [bytes] still to be allocated once the FP64 operands are on the device, as assumed
// by the stage-time model (awe.hpp): P is one digit plane of A plus one of B, out one
// output plane, F the FP64 operands, and Q = (composition planes + S residue planes) * out
// the product buffers. The model assumes the FP64 operands are released after slicing
// before Q is allocated: nplane*P + max(0, Q - F) + margin. All values are integers below
// 2^53, so double arithmetic is exact. The run-time band planning (plan_bands in plan.hpp)
// counts differently: it keeps the operands resident.
inline constexpr double OZ2_DRAM_MARGIN = (double)OZ_DRAM_MARGIN_BYTES;
constexpr double oz2_dram_one_group(double nplane, double P, double compose_planes, double S, double out, double F) {
    const double Q = (compose_planes + S) * out;
    return nplane * P + (Q > F ? Q - F : 0.0) + OZ2_DRAM_MARGIN;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
