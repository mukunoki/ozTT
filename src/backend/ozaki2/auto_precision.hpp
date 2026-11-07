// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: error models that choose the integer width b and the number of
// reconstruction passes from the input (host only; no device access). Called by the
// planner (choose_moduli_and_width in plan.hpp) before any device stage.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "common/norm_estimate.hpp"
#include "ozaki2/moduli.hpp"
namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// Statistics of the operands after the power-of-two scaling: tau is the power-of-two
// scale of a row of A or a column of B (its maximum rounded up to a power of two).
struct Oz2OperandStats {
    double TA = 0, TB = 0;  // sum of tau^2 over the rows of A / the columns of B
    double nA = 0, nB = 0;  // ||A||_F^2, ||B||_F^2
};

// One pass over A (M x K, row-major) and B (K x N). rgA and rgB are power-of-two factors
// applied to all elements so that the sums of squares neither overflow nor underflow.
static inline Oz2OperandStats measure_operand_stats(const std::vector<double>& A, const std::vector<double>& B,
                                                    uint32_t M, uint32_t N, uint32_t K, double rgA, double rgB) {
    Oz2OperandStats st;
    for (uint32_t r = 0; r < M; r++) {
        double mx = 0, s2 = 0;
        for (uint32_t c = 0; c < K; c++) {
            const double v = A[(size_t)r * K + c] * rgA;
            mx = std::max(mx, std::fabs(v));
            s2 += v * v;
        }
        // A zero row has no scale (the device uses 1), but its quantization error is
        // exactly 0, so it enters the error model with 0; counting 1 would inflate kappa2
        // for matrices with many zero rows.
        if (mx > 0) {
            const double t = std::exp2(std::ceil(std::log2(mx)));
            st.TA += t * t;
        }
        st.nA += s2;
    }
    for (uint32_t c = 0; c < N; c++) {
        double mx = 0, s2 = 0;
        for (uint32_t r = 0; r < K; r++) {
            const double v = B[(size_t)r * N + c] * rgB;
            mx = std::max(mx, std::fabs(v));
            s2 += v * v;
        }
        if (mx > 0) {
            const double t = std::exp2(std::ceil(std::log2(mx)));
            st.TB += t * t;
        }
        st.nB += s2;
    }
    return st;
}

// Result of the automatic precision selection.
struct Oz2PrecisionPolicy {
    int b_cap = OZ2_B_DEFAULT;  // cap on the width b
    // Input statistics (oz2_measure_stats); also used for the choice of the number of
    // reconstruction passes (oz2_recon1_log2).
    Oz2OperandStats st;
    double nC = 0;  // estimate of ||C||_F^2 (estimate_product_norm, including its safety factor)
};

// Measure the input statistics. kappa2 and the pass estimate do not depend on the absolute
// scale of A and B, so the operands are first normalized by powers of two (exact); the
// plain sums of squares would overflow for |a| ~ 1e150.
static inline void oz2_measure_stats(const std::vector<double>& A, const std::vector<double>& B, uint32_t M, uint32_t N,
                                     uint32_t K, Oz2PrecisionPolicy& p) {
    const double rgA = 1.0 / oz_pow2_scale(A), rgB = 1.0 / oz_pow2_scale(B);
    p.st = measure_operand_stats(A, B, M, N, K, rgA, rgB);
    p.nC = estimate_product_norm(A, B, M, N, K, rgA, rgB);
}

// Automatic precision: choose the cap on the width b from the input.
//
// Scheme II carries each element as a b-bit fixed-point number relative to the scale tau
// of its row (A) or column (B):
//   A' = round(a/tau_A * 2^b)        (quantization error |delta| <= 1/2)
// The step of row i is tau_A(i)*2^-b and of column j tau_B(j)*2^-b. Modelling delta as
// independent with mean 0 and variance 1/12, the expected squared norm of the error is
//   E||E||_F^2 ~ (2^-2b/12) * [ (sum_i tau_A(i)^2) ||B||_F^2 + (sum_j tau_B(j)^2) ||A||_F^2 ]
// (the second-order term delta_A*delta_B is dropped). For a target relative error 2^-t,
//   b = t + ceil(log2 kappa2),   kappa2 = ||E||_F * 2^b / ||C||_F,
// with t = OZ2_TARGET_BITS. This is a model of the expectation, not a guarantee: delta
// can deviate from rounding to nearest (the slicer guarantees |A'-X| <= 1, not 1/2), and
// the safety factor of estimate_product_norm absorbs such deviations.
//
// The CRT window that the moduli must cover stays the worst case |C'| <= K*2^(2b)
// (ozaki2_choose_moduli); only b is chosen here. The cap is never below OZ2_TARGET_BITS.
// The planner requests all OZ2_MAXS moduli with this cap (choose_moduli_and_width).
static inline Oz2PrecisionPolicy choose_precision_policy(const std::vector<double>& A, const std::vector<double>& B,
                                                         uint32_t M, uint32_t N, uint32_t K) {
    Oz2PrecisionPolicy out;
    oz2_measure_stats(A, B, M, N, K, out);
    const Oz2OperandStats& st = out.st;
    const double nC = out.nC;
    // Zero cases. A zero operand gives nC = 0 and a zero numerator, hence k2 = 0 and, with
    // the log floored at 0, b_cap = OZ2_TARGET_BITS. Nonzero operands whose product
    // estimate is 0 (or underflows) give a huge k2, capped at 1e18, so b_cap becomes the
    // structural cap through the min below.
    const double k2 = std::min(std::sqrt((st.TA * st.nB + st.TB * st.nA) / std::max(12.0 * nC, 1e-300)), 1e18);
    out.b_cap = std::min(OZ2_MAXDIGIT * OZ2_DIGIT_BITS, OZ2_TARGET_BITS + (int)std::ceil(std::log2(std::max(k2, 1.0))));
    return out;
}

// Choice of the number of reconstruction passes.
// One pass solves the fraction C'/M to a fixed absolute accuracy (the lowest limb of the
// triple-float in recon_crt.cpp), and an error e of the fraction appears in the output
// multiplied by M 2^-2b tau_A(i) tau_B(j). With eps1 the effective size of e, the expected
// relative Frobenius error is
//     eps1 * M 2^-2b * sqrt(sum_i tau_A(i)^2 * sum_j tau_B(j)^2) / ||C||_F.
// M is about 2^(2b) W, so the estimate hardly depends on b; it grows with the exponent
// range of the input (||C||_F small compared with the products of the row and column
// maxima). Two passes split 1/m_i into a part on a common granularity and a remainder and
// do not have this floor, at the cost of running the reconstruction twice. One pass is
// chosen only when the estimate is at least OZ2_RECON1_MARGIN bits below the target 2^-53.
// OZ2_RECON1_EPS_LOG2 is log2 of eps1, calibrated on the device.
#define OZ2_RECON1_EPS_LOG2 (-70.0)
#define OZ2_RECON1_MARGIN 2.0

// One pass is used when the estimate (log2) is at most this.
constexpr double OZ2_RECON1_THR_LOG2 = -(OZ2_TARGET_BITS + OZ2_RECON1_MARGIN);

// log2 of the expected relative Frobenius error of the one-pass reconstruction; p must
// hold the input statistics (oz2_measure_stats). -infinity, i.e. one pass, when the
// product estimate or a sum of tau^2 is zero. This includes nonzero operands whose
// product-norm estimate is 0 or has underflowed, a case that the width selection above
// treats as difficult (structural cap on b); the pass selection does not.
static inline double oz2_recon1_log2(const Ozaki2Moduli& mo, int b, const Oz2PrecisionPolicy& p) {
    if (!(p.nC > 0) || !(p.st.TA > 0) || !(p.st.TB > 0)) return -INFINITY;
    double l2m = 0;
    for (int m : mo.m) l2m += std::log2((double)m);
    return OZ2_RECON1_EPS_LOG2 + l2m - 2.0 * b + 0.5 * (std::log2(p.st.TA) + std::log2(p.st.TB) - std::log2(p.nC));
}
}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
