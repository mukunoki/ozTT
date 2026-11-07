// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: format-dependent constants and the exactness conditions.
//
// Host-side decisions only; nothing here touches the device. This header provides the
// scheme I parameters of the format (OZ1_PARAM), the constants derived from them (the
// digit bound, the math fidelity and the normalization settings; all fixed per format),
// and the conditions under which the partial products are error-free: the K block length
// (ozaki_block_k), the check of a chosen configuration (ozaki_check_exact), the Int32
// bound of the deferred reconstruction (oz_defer_fits) and the FP32 exponent budget of
// the reconstruction (oz_horner_fits). ozaki1/plan.hpp plans the partial-product GEMMs
// with them. Every other header of ozaki1/ includes this one.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cmath>

#include "common/gemm_host.hpp"

#include "common/format.hpp"

#include "ozaki1/param.hpp"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {
// Scheme I parameters of the format this translation unit is compiled for.
inline constexpr Oz1Param OZ1_PARAM = oz1_param(OZ_FMT_ID);

// True if the partial products are Int32 (INT8 path): the reconstruction then splits every
// Int32 partial sum into two exact FP32 values. Derived from the output format of the GEMM.
inline constexpr bool OZ1_INT32_PARTIAL = (OZ_FMT.out == tt::DataFormat::Int32);

// ===== Constants of the format =====
// Radix (weight of a digit) and digit bound (rule and rationale: oz1_digit_max in
// ozaki1/param.hpp). In every format of the table the digit bound is half the radix.
inline constexpr uint32_t OZ1_RADIX = OZ1_PARAM.radix;
inline constexpr uint32_t OZ1_DMAX = oz1_digit_max(OZ1_RADIX, OZ_FMT.exact_int_max);
static_assert(OZ1_DMAX == OZ1_RADIX / 2, "the digit bound of this format is not half the radix");

// Math fidelity of the matrix engine for the digit bound. One rule serves all formats
// and both schemes (oz_fidelity_rule in common/format.hpp). The LoFi multiplier (one
// pass) reads 5 bits of SrcA, so integers with at most 5 significant bits (0..31 and 32)
// are multiplied exactly: a digit bound of at most 32 gives exact products at LoFi, and
// since the radix is twice the digit bound, radix 64 still runs at LoFi. INT8 needs
// HiFi4 for any radix: the matrix engine reads INT8 operands through a different mask
// than FP operands, and the upper partition of SrcA then holds only the magnitude bits
// 7..5, so narrower digits do not help. Every format of the table satisfies one of the
// two, so the products of two digits are exact (the condition "mul" of the Ozaki
// scheme); the condition on the accumulation depends on K and is checked per run below.
inline constexpr tt::tt_metal::MathFidelity OZ1_FID = oz_fidelity_rule(OZ1_DMAX);
static_assert(OZ_FMT_ID == OzFmt::INT8 ? OZ1_FID == tt::tt_metal::MathFidelity::HiFi4
                                       : (OZ1_DMAX <= 32 && OZ1_FID == tt::tt_metal::MathFidelity::LoFi),
              "the digit bound and the fidelity of this format do not give exact digit products");

// Normalization. The balanced expansion with radix beta and digit bound dmax represents
// |x| <= span = dmax/(beta-1). For beta = 2*dmax that is about 1/2, so every row of A
// and column of B is normalized to at most 1/2 (nshift = -1); for radix 2 (BFP2) the
// span is 1 and so is the target (nshift = 0).
// The device derives the scale from the BF16 form of the largest magnitude. When that is
// truncated, it can be up to 2^-7 (relative) too small and the normalized values can
// exceed the target 2^nshift by as much. Formats whose span exceeds the target by at
// least 1/128 can represent the excess. The others (INT8 with 127/253 and BFP2 with 1)
// round the maximum up instead (bf_ceil), which guarantees |x| <= 2^nshift.
// If the target itself is not a single digit (beta * 2^nshift > dmax; BFP2 only), an
// element equal to the target, i.e. a row or column maximum that is an exact power of
// two, would expand into digits 1 without end. The scale is then the power of two
// strictly above the maximum (tau_above): such an element becomes 1/2, one exact digit,
// and the scales of all other rows and columns are unchanged.
inline constexpr double OZ1_SPAN = double(OZ1_DMAX) / (OZ1_RADIX - 1);
inline constexpr int32_t OZ1_NSHIFT = OZ1_SPAN >= 1.0 ? 0 : -1;    // normalization target 2^nshift
inline constexpr double OZ1_TARGET = OZ1_NSHIFT == 0 ? 1.0 : 0.5;  // the target itself
inline constexpr bool OZ1_BF_CEIL = OZ1_SPAN < OZ1_TARGET * (1.0 + 1.0 / 128.0);
inline constexpr bool OZ1_TAU_ABOVE = double(OZ1_RADIX) * OZ1_TARGET > double(OZ1_DMAX);
static_assert(OZ1_NSHIFT == (OZ_FMT_ID == OzFmt::BFP2 ? 0 : -1), "unexpected normalization target");
static_assert(OZ1_BF_CEIL == (OZ_FMT_ID == OzFmt::INT8 || OZ_FMT_ID == OzFmt::BFP2), "unexpected bf_ceil");
static_assert(OZ1_TAU_ABOVE == (OZ_FMT_ID == OzFmt::BFP2), "unexpected tau_above");

// Rounding selector of the scaling stage (CT_CEIL of scale_f64_reader.cpp): 0 truncate,
// 1 round up, 2 round up and step exact powers of two up.
inline constexpr uint32_t OZ1_CEIL_MODE = OZ1_TAU_ABOVE ? 2u : OZ1_BF_CEIL ? 1u : 0u;

// ===== Exactness condition of the accumulation =====
// Every product satisfies |a^(i) b^(j)| <= dmax^2 (dmax: digit bound), and one GEMM
// accumulates K_eff of them. Let 2w = ceil(log2 dmax^2) (ozaki_prod_bits; 10 bits for
// digits +-32) and let b_acc be the width of the integers that the accumulator holds
// without rounding. If
//     2w + ceil(log2 K_eff) <= b_acc
// the accumulation is exact even when all terms are maximal and of the same sign.
// A GEMM that concatenates the L+1 slice pairs of a diagonal along K has
// K_eff = (L+1) K_c, where K_c is the K block length. The pair loop, which folds the
// accumulator at every pair boundary, and the pairwise issue have K_eff = K_c
// (plan_issue in ozaki1/plan.hpp).
// b_acc is 24 for the FP32 accumulation of the seven FP formats (FP32 represents the
// integers up to 2^24 exactly) and 31 for the Int32 accumulation of INT8 (the magnitude
// part of a sign-magnitude int32).
// With random inputs the partial sums cancel and a violated condition goes unnoticed, but
// inputs without cancellation, such as nonnegative matrices, do lose accuracy. The
// condition is therefore enforced when a configuration is planned.
static constexpr uint32_t OZAKI_ACC_BITS = OZ1_INT32_PARTIAL ? 31u : 24u;

// Number of bits ceil(log2(dmax^2)) of a digit product |d_i d_j| <= dmax^2: the term 2w
// of the exactness condition. It is 2s for dmax = 2^s as well as for dmax = 2^s - 1, so a
// power-of-two digit bound (32 for radix 64) does not cost an extra bit.
static inline uint32_t ozaki_prod_bits(uint32_t dmax) {
    const uint64_t r2 = (uint64_t)dmax * dmax;
    uint32_t b = 0;
    while ((1ull << b) < r2) ++b;
    return b;
}

// Whether k_eff = (L+1) K digit products accumulate exactly with the digit bound of the
// format: dmax^2 * 2^ceil(log2 k_eff) <= 2^b_acc, and for INT8 also dmax^2 * k_eff <=
// 2^31 - 1 (a sign-magnitude Int32 cannot hold 2^31, whereas FP32 represents 2^24
// exactly). When k_eff is not a power of two the first condition can fail although
// dmax^2 * k_eff <= 2^b_acc holds; it stays on the safe side. K is the K block length
// and L+1 the number of slice pairs that one accumulation covers (L = 0 if the
// accumulator is folded after every pair).
static inline bool ozaki_check_exact(uint32_t K, int L) {
    const uint64_t k_eff = (uint64_t)(L + 1) * K;
    uint32_t bits = 0;
    while ((1ull << bits) < k_eff) ++bits;  // ceil(log2 k_eff)
    if (OZAKI_ACC_BITS < bits) return false;
    const uint64_t budget = 1ull << (OZAKI_ACC_BITS - bits);  // upper bound of dmax^2
    const uint64_t d2 = (uint64_t)OZ1_DMAX * OZ1_DMAX;
    if (d2 > budget) return false;
    return !OZ1_INT32_PARTIAL || d2 * k_eff <= (uint64_t)INT32_MAX;
}

// K block length K_c that satisfies the exactness condition (the inverse problem of
// ozaki_check_exact): the largest K_c with 2w + ceil(log2((L+1) K_c)) <= b_acc that
// divides K and is a multiple of the tile width. Equal block lengths let every K block
// reuse the same program. Returns K if the whole K satisfies the condition (no blocking)
// and 0 if no such block length exists. L+1 is the number of slice pairs that one
// accumulation covers.
static inline uint32_t ozaki_block_k(uint32_t K, int L) {
    const uint32_t pb = ozaki_prod_bits(OZ1_DMAX);  // 2w: bits of a digit product
    if (OZAKI_ACC_BITS < pb + 1) return 0;          // this digit bound does not fit the accumulator
    uint64_t keff_max = 1ull << (OZAKI_ACC_BITS - pb);

    // An INT8 partial product is a sign-magnitude Int32 and cannot hold 2^31. If d^2 is a
    // power of two, the budget above admits exactly 2^31, so it is tightened with the
    // integer bound (for digits +-127, INT32_MAX / d^2 is the larger of the two).
    if (OZ1_INT32_PARTIAL) {
        const uint64_t d = OZ1_DMAX;
        keff_max = std::min<uint64_t>(keff_max, (uint64_t)INT32_MAX / (d * d));
    }
    const uint64_t kc_max = keff_max / (uint64_t)(L + 1);
    if (kc_max >= (uint64_t)K) return K;  // no blocking needed
    uint32_t best = 0;
    for (uint32_t c = tt::constants::TILE_WIDTH; c <= kc_max; c += tt::constants::TILE_WIDTH)
        if (K % c == 0) best = c;
    return best;
}

// Exponent budget of the Horner evaluation in the reconstruction (triple-float;
// acc = P_0, then acc = acc * radix + P_d for d = 1..L). The multiplication by a general
// radix splits the leading limb by Dekker's method, which multiplies it by 4097 (about
// 2^12) and overflows FP32 (limit 2^128) once the accumulated value exceeds 2^116. The
// value that is split last is the one before the final multiplication,
//     H_{L-1} = sum_{d<L} P_d radix^(L-1-d),   |P_d| <= (d+1) K dmax^2,
// and |H_{L-1}| < K dmax^2 radix^(L+1) / (radix-1)^2 <= K radix^(L+1) because
// dmax <= radix - 1. Hence the condition
//     log2(K) + (L+1) log2(radix) + 12 < 128.
// A power-of-two radix is multiplied without the split, but the same test is applied to
// every radix (safe side). The test is independent of the exactness of the accumulation
// (oz_defer_fits): a configuration can pass that one and still overflow here.
static inline bool oz_horner_fits(uint32_t K, int L) {
    const double v = std::log2((double)K) + (double)(L + 1) * std::log2((double)OZ1_RADIX);
    return v + 12.0 < 128.0;
}

// Whether the sum over all K blocks fits one Int32 word (deferred reconstruction). The sum
// of the nkb blocks of a diagonal is bounded by nkb * (L+1) * K_c * d^2 = (L+1) * K * d^2,
// where d is the digit bound and K the full, unblocked length, and the largest positive
// Int32 is 2^31 - 1. The comparison is done on integers: the bit-count form
// 2w + ceil(log2((L+1) K)) <= 31 would admit exactly 2^31 in the equality case.
static inline bool oz_defer_fits(uint32_t K, int L) {
    const uint64_t k_eff = (uint64_t)(L + 1) * K;
    const uint64_t dsq = (uint64_t)OZ1_DMAX * OZ1_DMAX;

    // k_eff * dsq <= INT32_MAX, evaluated without overflowing the product.
    return dsq != 0 && k_eff <= (uint64_t)INT32_MAX / dsq;
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
