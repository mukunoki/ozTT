// SPDX-License-Identifier: Apache-2.0
#pragma once
// Estimate of ||C||_F for the automatic precision selection of both Ozaki schemes
// (ozaki1/auto_precision.hpp, ozaki2/auto_precision.hpp).
//
// Both schemes use it as the denominator of their model of the expected error. It runs on the host only,
// before the device pipeline (scaling -> slicing -> partial-product GEMMs -> reconstruction) starts.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "common/bits.hpp"  // checks the width of long double, in which the sums below are formed

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Power of two by which a matrix is divided before sums of squares are formed (exact, so ratios are not
// affected): the exponent of the largest magnitude. The exponent is clamped to the range of the normal
// numbers, so the factor itself never becomes 0 or inf, even if the input has only subnormal values.
inline double oz_pow2_scale(const std::vector<double>& v) {
    double mx = 0;
    for (double x : v) mx = std::max(mx, std::fabs(x));
    if (!(mx > 0) || !std::isfinite(mx)) return 1.0;
    int e;
    std::frexp(mx, &e);
    e = std::min(std::max(e, -1000), 1000);
    return std::ldexp(1.0, e);
}

// Estimates ||C||_F^2 for C = (rgA*A)(rgB*B) with a Hutchinson sketch (average over L vectors). A is
// M x K and B is K x N, row-major; rgA and rgB are scale factors applied to the elements (the callers
// pass the reciprocals of oz_pow2_scale). The result is a deliberately low estimate: the largest sample
// is dropped and the rest is divided by a safety factor.
//
// ||C||_F must not be measured by sampling elements: for heavy-tailed inputs the sum of squares is
// dominated by a few elements and the variance of the sample diverges. For a random vector g with mean 0
// and variance 1, E||A(B g)||^2 = ||AB||_F^2 holds exactly and one vector costs O(MK+KN). Every element
// is seen at once, so the sketch is robust against tails.
// Being unbiased is not enough, however: overestimating ||C||_F^2 errs toward too little precision. The
// relative variance of the average over L vectors is 2/L * tr((C^T C)^2)/||C||_F^4, at worst (rank 1)
// 2/L, so with L=8 the relative standard deviation reaches 0.5. Two measures are taken:
//   (1) The largest of the L samples is dropped before averaging. The seed is fixed, so an input whose
//       right singular vector is aligned with one particular g_l makes that one sample jump; dropping
//       it removes the effect.
//   (2) The remaining average is divided by 1+2*sqrt(2/L) (a factor 1/2 for L=8).
// This is a rule of thumb for practical inputs, not a guaranteed lower bound. For inputs with a high
// effective rank it is more than needed, but erring toward more precision is the cheaper side.
inline double estimate_product_norm(const std::vector<double>& A, const std::vector<double>& B, uint32_t M, uint32_t N,
                                    uint32_t K, double rgA, double rgB) {
    const uint32_t L = 8;
    std::mt19937 rg(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> g(N), y(K);
    double cs2 = 0, cmax = 0;
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t j = 0; j < N; j++) g[j] = nd(rg);
        for (uint32_t k = 0; k < K; k++) {
            long double s = 0;
            for (uint32_t j = 0; j < N; j++) s += (long double)(B[(size_t)k * N + j] * rgB) * g[j];
            y[k] = (double)s;
        }
        double q = 0;
        for (uint32_t i = 0; i < M; i++) {
            long double s = 0;
            for (uint32_t k = 0; k < K; k++) s += (long double)(A[(size_t)i * K + k] * rgA) * y[k];
            q += (double)s * (double)s;
        }
        cs2 += q;
        cmax = std::max(cmax, q);
    }
    return (cs2 - cmax) / (L - 1) / (1.0 + 2.0 * std::sqrt(2.0 / (double)L));
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
