// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: selection of the number of slices p from the input.
//
// Host only; the device is not touched. The driver calls oz_fp64_p when
// Options::slices is zero. From a model of the truncation error it searches a p for which
// the relative error ||E||_F / ||C||_F is expected to be at most 2^-53. This is an
// estimate, not a guarantee, and because estimate (2) below need not be monotonic in p,
// the p found is not necessarily the smallest that satisfies the model.
//
// The model normalizes and slices by the rules of the device.
//   Normalization: the largest magnitude of each row of A / column of B is converted to
//     BF16 in the direction of the format (truncated, rounded up, or rounded up with
//     exact powers of two stepped up; CT_CEIL of scale_f64_reader.cpp), e is the ceiling
//     of its log2 (expdelta.cpp), and x = a * 2^(nshift-e). oz1_dev_exp computes e with
//     the functions the kernels use. The target 2^nshift is 1 for BFP2 and 1/2 for the
//     other formats; with truncation x can exceed the target by less than 2^-7 relative.
//   Slicing: u = beta*t, d = clamp(round(u), +-dmax), t = u - d (slice_tf_num.cpp,
//     slice_tf_sm8.cpp). The rounding is to nearest even and, as on the device, decided
//     on u rounded to FP32 (the device rounds the leading limb of a triple-float); this
//     matters only near ties. The residual t is carried in two doubles (OzDigits), which
//     follows the expansion exactly down to p digits also for the radices 14 and 254.
// Write x = sum_{i<p} D_i + R_p with D_i = d_i beta^-(i+1) and R_p = t_p beta^-p (y
// likewise). The device keeps the pairs with i+j <= p-1 only, so the truncation error of
// output element (m, n) is
//   E_mn = s_m s_n sum_k [ sum_{i<p} D_i(x_mk) R_{p-i}(y_kn) + R_p(x_mk) y_kn ],
// where s = 2^(e-nshift) is the divisor of the normalization. It contains all pairs with
// i+j >= p and the products with the residuals that entered no slice.
// The relative error is estimated in two ways, and p must satisfy both:
//   (1) Sum of squares assuming independent signs (all elements; oz1_fp64_indep). With
//       the signs of the terms along k and across pairs taken as independent,
//         E||E||_F^2 ~= sum_k [ sum_{i+j>=p, i,j<p} SA_i[k] SB_j[k] + RA_p[k] NB[k] + NA[k] RB_p[k] ],
//         SA_i[k] = sum_m (s_m D_i(x_mk))^2,  RA_p[k] = sum_m (s_m R_p(x_mk))^2,
//         NA[k] = sum_m a_mk^2   (likewise for B).
//       The denominator ||C||_F^2 is a Hutchinson estimate (common/norm_estimate.hpp)
//       biased low, so that p errs on the large side. For inputs whose errors do not
//       align in sign, such as random matrices, this estimate decides.
//   (2) Simulation of the device result on sampled rows and columns, without the
//       independence assumption (oz1_fp64_sampled). For each sampled pair, C_mn is
//       computed in double-double, E_mn is subtracted and the result is rounded to FP64
//       as the device does; the squared differences from C_mn are summed with weights.
//       Rows and columns are sampled separately: the OZ1_FP64_TOP of largest norm, plus
//       OZ1_FP64_UNI drawn uniformly from the rest and weighted by (remaining count) /
//       (drawn count) (a stratified estimate; exact if everything is sampled).
//       For inputs whose errors align in sign (constant matrices, fixed sign patterns)
//       the error grows in proportion to K and to the number of pairs, and (1) is far too
//       optimistic; moreover the FP64 rounding errors of equal elements align too, so a
//       truncation error that pushes them across a rounding midpoint shifts every
//       element by one ulp. (2) captures both effects. It sees sign alignment only in
//       the sampled rows and columns and can be optimistic if the alignment is confined
//       to a few rows or columns outside the sample.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <random>
#include <vector>
#include "ozaki1/format.hpp"
#include "common/norm_estimate.hpp"
#include "common/bits.hpp"
#include "common/kernels/scale_bits.h"
namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Rows and columns sampled for estimate (2): the TOP of largest norm and UNI drawn
// uniformly from the rest (for rows and for columns separately).
constexpr uint32_t OZ1_FP64_TOP = 32, OZ1_FP64_UNI = 32;

// Fixed seeds of the uniform draws (rows of A, columns of B), so that the same input
// always selects the same p.
constexpr uint32_t OZ1_FP64_SEED_ROWS = 20261021u, OZ1_FP64_SEED_COLS = 20261022u;

// Exponent e that the device uses to normalize a row or column with largest magnitude mx
// (x = a * 2^(nshift-e)). ceil_mode is the rounding selector of the scaling stage
// (0 truncate, 1 round up, 2 round up and step exact powers of two up); the BF16
// conversion and the exponent come from the functions the kernels use (scale_bits.h).
// in_range is false for mx = 0 (an all-zero row or column, which adds no error) and for
// values the device rejects (above the largest finite BF16, nonzero below 2^-126, Inf,
// NaN).
static int oz1_dev_exp(double mx, uint32_t ceil_mode, bool& in_range) {
    const uint64_t u = f64_bits(mx);
    const uint32_t h = (uint32_t)(u >> 32) & 0x7FFFFFFFu, l = (uint32_t)u;
    const uint16_t b = ceil_mode == 2   ? oz_absf64_to_bf16<2>(h, l)
                       : ceil_mode == 1 ? oz_absf64_to_bf16<1>(h, l)
                                        : oz_absf64_to_bf16<0>(h, l);
    uint32_t bad = 0;
    const int32_t e = oz_bf16_scale_exp(b, bad);
    in_range = b != 0 && bad == 0u;
    return (int)e;
}

// Radix-beta expansion of one element, advanced digit by digit by the rule of the device
// slicer. The residual t is held as two doubles th + tl; the product beta*th is split
// exactly into two words with an FMA.
struct OzDigits {
    double th = 0.0, tl = 0.0;

    // Error-free sum: s + e = a + b exactly.
    static void two_sum(double a, double b, double& s, double& e) {
        s = a + b;
        const double bb = s - a;
        e = (a - (s - bb)) + (b - bb);
    }

    // One step u = beta*t, d = clamp(round(u), +-dmax), t = u - d; returns the digit d.
    // uh - d is exact: it is about 1/2 or less in magnitude, or, when the clamp acts,
    // d <= uh <= 2d. nearbyint rounds in the current rounding mode of the host, which
    // must be round-to-nearest (the default) to match the device.
    double step(double beta, double dmax) {
        const double ph = beta * th, pe = std::fma(beta, th, -ph);
        double uh, ul;
        two_sum(ph, pe + beta * tl, uh, ul);
        const double d = std::min(dmax, std::max(-dmax, std::nearbyint((double)(float)uh)));
        two_sum(uh - d, ul, th, tl);
        return d;
    }
    double value() const { return th + tl; }
};

// Parameters of the expansion and the search bound.
struct Oz1Fp64Cfg {
    uint32_t M = 0, N = 0, K = 0;
    double beta = 0.0, dmax = 0.0;  // radix and digit bound
    int32_t nshift = 0;             // normalization target 2^nshift
    uint32_t ceil_mode = 0;         // rounding selector of the scaling stage (OZ1_CEIL_MODE)
    int pmax = 0;                   // largest p that is searched
};

// Normalization of the rows of A and columns of B: the exponents e (oz1_dev_exp) and
// the divisors 2^(e-nshift) multiplied by rgA / rgB (sA, sB). A divisor of 0 marks a
// zero or out-of-range row or column, which is not counted. nrA and nrB are the squared
// norms used to pick the sample of estimate (2). All sums of squares are taken after
// multiplying by the powers of two rgA and rgB so that they cannot overflow; ratios are
// unaffected.
struct Oz1Fp64Norm {
    double rgA = 1.0, rgB = 1.0;
    std::vector<int> eA, eB;
    std::vector<double> sA, sB, nrA, nrB;
};
static Oz1Fp64Norm oz1_fp64_norm(const std::vector<double>& A, const std::vector<double>& Bm, const Oz1Fp64Cfg& g) {
    const uint32_t M = g.M, N = g.N, K = g.K;
    const int32_t nshift = g.nshift;
    const uint32_t ceil_mode = g.ceil_mode;
    Oz1Fp64Norm nm;
    nm.rgA = 1.0 / oz_pow2_scale(A);
    nm.rgB = 1.0 / oz_pow2_scale(Bm);
    const double rgA = nm.rgA, rgB = nm.rgB;
    nm.eA.assign(M, 0);
    nm.eB.assign(N, 0);
    nm.sA.assign(M, 0.0);
    nm.sB.assign(N, 0.0);
    nm.nrA.assign(M, 0.0);
    nm.nrB.assign(N, 0.0);
    std::vector<int>& eA = nm.eA;
    std::vector<int>& eB = nm.eB;
    std::vector<double>& sA = nm.sA;
    std::vector<double>& sB = nm.sB;
    std::vector<double>& nrA = nm.nrA;
    std::vector<double>& nrB = nm.nrB;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t m = 0; m < (int64_t)M; m++) {
        double mx = 0, nr = 0;
        for (uint32_t k = 0; k < K; k++) {
            const double v = A[(size_t)m * K + k];
            mx = std::max(mx, std::fabs(v));
            nr += (v * rgA) * (v * rgA);
        }
        bool ok = false;
        const int e = oz1_dev_exp(mx, ceil_mode, ok);
        if (mx > 0 && ok) {
            eA[m] = e;
            sA[m] = std::ldexp(rgA, e - nshift);
        }
        nrA[m] = nr;
    }
    {
        // Columns of B in blocks of CB with k as the outer loop (B is stored row-major).
        constexpr uint32_t CB = 64;
        const uint32_t nblk = (N + CB - 1) / CB;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t bk = 0; bk < (int64_t)nblk; bk++) {
            const uint32_t n0 = (uint32_t)bk * CB, n1 = std::min(N, n0 + CB);
            double mx[CB] = {0}, nr[CB] = {0};
            for (uint32_t k = 0; k < K; k++)
                for (uint32_t n = n0; n < n1; n++) {
                    const double v = Bm[(size_t)k * N + n];
                    mx[n - n0] = std::max(mx[n - n0], std::fabs(v));
                    nr[n - n0] += (v * rgB) * (v * rgB);
                }
            for (uint32_t n = n0; n < n1; n++) {
                bool ok = false;
                const int e = oz1_dev_exp(mx[n - n0], ceil_mode, ok);
                if (mx[n - n0] > 0 && ok) {
                    eB[n] = e;
                    sB[n] = std::ldexp(rgB, e - nshift);
                }
                nrB[n] = nr[n - n0];
            }
        }
    }
    return nm;
}

// Estimate (1): returns e2r[q], the expected ||E||_F^2 for q slices (q = 2..pmax), in
// the scaled units of Oz1Fp64Norm.
static std::vector<double> oz1_fp64_indep(const std::vector<double>& A, const std::vector<double>& Bm,
                                          const Oz1Fp64Cfg& g, const Oz1Fp64Norm& nm) {
    const uint32_t M = g.M, N = g.N, K = g.K;
    const int pmax = g.pmax;
    const size_t P1 = (size_t)pmax + 1;
    const double beta = g.beta, dmax = g.dmax, rgA = nm.rgA, rgB = nm.rgB;
    const int32_t nshift = g.nshift;

    // S[i*K+k]: contribution of slice i; R[q*K+k]: contribution of the residual after q slices.
    std::vector<double> SA(P1 * K, 0.0), RA(P1 * K, 0.0), SB(P1 * K, 0.0), RB(P1 * K, 0.0), NA(K, 0.0), NB(K, 0.0);
    // x: normalized value, s: its divisor (scaled).
    auto cut = [&](double x, double s, double* S, double* R) {
        OzDigits t;
        t.th = x;
        double w = 1.0;
        for (int i = 0; i < pmax; i++) {
            const double d = t.step(beta, dmax);
            w /= beta;
            S[(size_t)i * K] += (s * d * w) * (s * d * w);
            const double r = s * t.value() * w;
            R[(size_t)(i + 1) * K] += r * r;
            if (t.th == 0) break;  // all further digits and residuals are zero
        }
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t k = 0; k < (int64_t)K; k++) {
        double na = 0, nb = 0;
        for (uint32_t m = 0; m < M; m++) {
            const double v = A[(size_t)m * K + k];
            na += (v * rgA) * (v * rgA);
            if (v != 0 && nm.sA[m] > 0) cut(std::ldexp(v, nshift - nm.eA[m]), nm.sA[m], &SA[k], &RA[k]);
        }
        for (uint32_t n = 0; n < N; n++) {
            const double v = Bm[(size_t)k * N + n];
            nb += (v * rgB) * (v * rgB);
            if (v != 0 && nm.sB[n] > 0) cut(std::ldexp(v, nshift - nm.eB[n]), nm.sB[n], &SB[k], &RB[k]);
        }
        NA[k] = na;
        NB[k] = nb;
    }

    // G[i][j] = sum_k SA_i[k] SB_j[k]: the truncated pairs.
    std::vector<double> G((size_t)pmax * pmax, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < pmax; i++)
        for (int j = 0; j < pmax; j++) {
            double gij = 0;
            for (uint32_t k = 0; k < K; k++) gij += SA[(size_t)i * K + k] * SB[(size_t)j * K + k];
            G[(size_t)i * pmax + j] = gij;
        }
    std::vector<double> e2r(P1, 0.0);
    for (int q = 2; q <= pmax; q++) {
        double e2 = 0;
        for (int i = 0; i < q; i++)
            for (int j = std::max(0, q - i); j < q; j++) e2 += G[(size_t)i * pmax + j];
        for (uint32_t k = 0; k < K; k++) e2 += RA[(size_t)q * K + k] * NB[k] + NA[k] * RB[(size_t)q * K + k];
        e2r[q] = e2;
    }
    return e2r;
}

// Estimate (2): the sampled rows and columns with their exact products; e2(q) evaluates
// the model for q slices.
struct Oz1Fp64Sampled {
    const Oz1Fp64Cfg* g = nullptr;
    const Oz1Fp64Norm* nm = nullptr;
    std::vector<uint32_t> iA, iB;  // sampled rows of A / columns of B
    std::vector<double> wA, wB;    // their weights
    size_t na = 0, nb = 0;
    std::vector<double> XA, YB;  // normalized values of the sampled rows / columns
    std::vector<double> Ch, Cl;  // C_mn of the sampled pairs as two words (scaled)
    double nC2 = 0.0;            // denominator of estimate (2)

    // Squared relative error ||fl(C-E) - C||_F^2 / ||C||_F^2 for q slices: the weighted
    // sum over the sampled pairs divided by nC2. k is split into 64 fixed ranges that
    // are summed separately and combined in order, so the value does not depend on the
    // number of threads. Without any sampled pair (all rows or all columns zero or out
    // of range) the empty sum 0 is returned.
    double e2(int q) const {
        if (na == 0 || nb == 0) return 0.0;
        const uint32_t K = g->K;
        const double beta = g->beta, dmax = g->dmax;
        constexpr uint32_t NCH = 64;
        std::vector<double> part((size_t)NCH * na * nb, 0.0);
        // The work areas are allocated inside the parallel region; an exception there
        // would end the process, so it is kept and rethrown after the region.
        std::exception_ptr failure;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int64_t c = 0; c < (int64_t)NCH; c++) {
            try {
                const uint32_t k0 = (uint32_t)((uint64_t)K * c / NCH), k1 = (uint32_t)((uint64_t)K * (c + 1) / NCH);
                std::vector<double> D(na * q), RX(na), RY(nb * (q + 1)), Y(nb);
                double* acc = &part[(size_t)c * na * nb];
                for (uint32_t k = k0; k < k1; k++) {
                    for (size_t a = 0; a < na; a++) {  // D_i(x) for i < q, and R_q(x)
                        OzDigits t;
                        t.th = XA[a * K + k];
                        double w = 1.0;
                        for (int i = 0; i < q; i++) {
                            const double d = t.step(beta, dmax);
                            w /= beta;
                            D[a * q + i] = d * w;
                        }
                        RX[a] = t.value() * w;
                    }
                    for (size_t b = 0; b < nb; b++) {  // R_c(y) for c = 1..q, and y
                        OzDigits t;
                        t.th = Y[b] = YB[b * K + k];
                        double w = 1.0;
                        for (int c2 = 1; c2 <= q; c2++) {
                            (void)t.step(beta, dmax);
                            w /= beta;
                            RY[b * (q + 1) + c2] = t.value() * w;
                        }
                    }
                    for (size_t a = 0; a < na; a++)
                        for (size_t b = 0; b < nb; b++) {
                            double e = RX[a] * Y[b];
                            for (int i = 0; i < q; i++) e += D[a * q + i] * RY[b * (q + 1) + (q - i)];
                            acc[a * nb + b] += e;
                        }
                }
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(oz1_fp64_e2)
#endif
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
        double e2 = 0;
        for (size_t a = 0; a < na; a++)
            for (size_t b = 0; b < nb; b++) {
                double E = 0;
                for (uint32_t c = 0; c < NCH; c++) E += part[((size_t)c * na + a) * nb + b];

                // The device returns C - E rounded to FP64. Its 64-bit assembly window
                // (assemble_tf_f64 in common/kernels/f64_tf.h keeps 60 bits below the
                // leading bit and truncates the lower limbs toward zero) can move the
                // value before rounding by about 2^-60 relative, so for a value within
                // that distance (2^-59 here) of a rounding midpoint the direction is
                // undetermined; the side farther from C is then assumed.
                const size_t ab = a * nb + b;
                double h, r;
                OzDigits::two_sum(Ch[ab], Cl[ab] - nm->sA[iA[a]] * nm->sB[iB[b]] * E, h, r);
                double out = h;
                if (h != 0) {
                    const double h2 = std::nextafter(h, r >= 0 ? INFINITY : -INFINITY);  // neighbor on the side of r
                    const double gap = 0.5 * std::fabs(h2 - h) - std::fabs(r);           // distance to the midpoint
                    if (gap <= std::ldexp(1.0, std::ilogb(h) - 59)) {
                        const double d1 = (h - Ch[ab]) - Cl[ab], d2 = (h2 - Ch[ab]) - Cl[ab];
                        if (std::fabs(d2) > std::fabs(d1)) out = h2;
                    }
                }
                const double dv = (out - Ch[ab]) - Cl[ab];
                e2 += wA[a] * wB[b] * dv * dv;
            }
        return nC2 > 0 ? e2 / nC2 : (e2 > 0 ? INFINITY : 0.0);
    }
};

// Picks the sample of estimate (2) and computes its exact products. Per operand: the
// OZ1_FP64_TOP rows (columns) of largest norm with weight 1, and OZ1_FP64_UNI drawn
// uniformly from the rest with weight (remaining count) / (drawn count). Zero and
// out-of-range rows and columns contribute no error and are excluded. nC is the
// denominator of estimate (1), the Hutchinson estimate of ||C||_F^2.
static Oz1Fp64Sampled oz1_fp64_sampled(const std::vector<double>& A, const std::vector<double>& Bm, const Oz1Fp64Cfg& g,
                                       const Oz1Fp64Norm& nm, double nC) {
    const uint32_t M = g.M, N = g.N, K = g.K;
    const int32_t nshift = g.nshift;
    Oz1Fp64Sampled sm;
    sm.g = &g;
    sm.nm = &nm;
    auto pick = [&](const std::vector<double>& nr, const std::vector<double>& s, uint32_t L, uint32_t seed,
                    std::vector<uint32_t>& idx, std::vector<double>& wt) {
        std::vector<uint32_t> ord;
        for (uint32_t i = 0; i < L; i++)
            if (s[i] > 0) ord.push_back(i);
        const size_t nt = std::min<size_t>(OZ1_FP64_TOP, ord.size());
        std::partial_sort(ord.begin(), ord.begin() + nt, ord.end(),
                          [&](uint32_t a, uint32_t b) { return nr[a] > nr[b] || (nr[a] == nr[b] && a < b); });
        for (size_t i = 0; i < nt; i++) {
            idx.push_back(ord[i]);
            wt.push_back(1.0);
        }
        std::vector<uint32_t> rest(ord.begin() + nt, ord.end());
        const size_t nu = std::min<size_t>(OZ1_FP64_UNI, rest.size());
        std::mt19937 rg(seed);
        for (size_t i = 0; i < nu; i++) {
            std::swap(rest[i], rest[i + rg() % (rest.size() - i)]);
            idx.push_back(rest[i]);
            wt.push_back((double)rest.size() / (double)nu);
        }
    };
    pick(nm.nrA, nm.sA, M, OZ1_FP64_SEED_ROWS, sm.iA, sm.wA);
    pick(nm.nrB, nm.sB, N, OZ1_FP64_SEED_COLS, sm.iB, sm.wB);
    const size_t na = sm.iA.size(), nb = sm.iB.size();
    sm.na = na;
    sm.nb = nb;
    sm.XA.assign(na * K, 0.0);
    sm.YB.assign(nb * K, 0.0);
    for (size_t a = 0; a < na; a++)
        for (uint32_t k = 0; k < K; k++)
            sm.XA[a * K + k] = std::ldexp(A[(size_t)sm.iA[a] * K + k], nshift - nm.eA[sm.iA[a]]);
    for (size_t b = 0; b < nb; b++)
        for (uint32_t k = 0; k < K; k++)
            sm.YB[b * K + k] = std::ldexp(Bm[(size_t)k * N + sm.iB[b]], nshift - nm.eB[sm.iB[b]]);

    // C_mn of the sampled pairs by a compensated dot product, as two words Ch + Cl.
    sm.Ch.assign(na * nb, 0.0);
    sm.Cl.assign(na * nb, 0.0);
    const double rgA = nm.rgA, rgB = nm.rgB;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ab = 0; ab < (int64_t)(na * nb); ab++) {
        const size_t a = (size_t)ab / nb, b = (size_t)ab % nb;
        double sh = 0, sl = 0;
        for (uint32_t k = 0; k < K; k++) {
            const double u = A[(size_t)sm.iA[a] * K + k] * rgA, v = Bm[(size_t)k * N + sm.iB[b]] * rgB;
            const double pr = u * v, pe = std::fma(u, v, -pr);
            double t, te;
            OzDigits::two_sum(sh, pr, t, te);
            sh = t;
            sl += pe + te;
        }
        OzDigits::two_sum(sh, sl, sm.Ch[ab], sm.Cl[ab]);
    }

    // Denominator of (2): the larger of ||C||_F^2 estimated from the sample with the
    // same weights and the denominator of (1). The former is exact for a constant
    // matrix. For inputs with a wide exponent range ||C||_F^2 is dominated by a few
    // large elements; if the rows and columns picked by norm miss those pairs, the
    // sampled sum comes out far too small and (2) would be far too pessimistic, so the
    // latter bounds it from below.
    double nCs = 0;
    for (size_t a = 0; a < na; a++)
        for (size_t b = 0; b < nb; b++) nCs += sm.wA[a] * sm.wB[b] * sm.Ch[a * nb + b] * sm.Ch[a * nb + b];
    sm.nC2 = std::max(nC, nCs);
    return sm;
}

// Searches q such that max(r1[q], e2(q)) <= tgt, where r1[q] is the squared relative
// error of estimate (1) and e2(q) that of estimate (2). The search starts at the
// smallest q for which (1) alone meets the target; if (2) does not, the range is doubled
// upward and then bisected. (2) need not be monotonic in q, so the result is a q whose
// predecessor in the bisection fails, not necessarily the smallest q that passes.
// Returns pmax if the target is not reached. e2sv caches the evaluated values of (2)
// (negative: not evaluated).
static int oz1_fp64_search(const std::vector<double>& r1, std::vector<double>& e2sv, const Oz1Fp64Sampled& sm, int pmax,
                           double tgt) {
    auto f2 = [&](int q) {
        if (e2sv[q] < 0) e2sv[q] = sm.e2(q);
        return std::max(r1[q], e2sv[q]);
    };
    int q0 = pmax;
    for (int q = 2; q <= pmax; q++)
        if (r1[q] <= tgt) {
            q0 = q;
            break;
        }
    int p = q0;
    if (f2(q0) > tgt) {
        int lo = q0, hi = -1;
        for (int step = 1; lo < pmax; step *= 2) {
            const int q = std::min(pmax, lo + step);
            if (f2(q) <= tgt) {
                hi = q;
                break;
            }
            lo = q;
        }
        if (hi < 0)
            p = pmax;  // the target is not reached within the search bound
        else {
            while (hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                if (f2(mid) <= tgt)
                    hi = mid;
                else
                    lo = mid;
            }
            p = hi;
        }
    }
    return p;
}

// Selects the number of slices p for A (M x K) and B (K x N), both row-major, with the
// radix, the digit bound and the normalization of the format (OZ1_* in ozaki1/format.hpp).
static int oz_fp64_p(const std::vector<double>& A, const std::vector<double>& Bm, uint32_t M, uint32_t N, uint32_t K) {
    Oz1Fp64Cfg g;
    g.M = M;
    g.N = N;
    g.K = K;
    g.beta = (double)OZ1_RADIX;
    g.dmax = (double)OZ1_DMAX;
    g.nshift = OZ1_NSHIFT;
    g.ceil_mode = OZ1_CEIL_MODE;

    // Search bound: 160 bits below the normalization position. Only inputs with an
    // extreme exponent range need deeper slicing; pmax is returned if the target is not
    // reached by then.
    g.pmax = (int)std::ceil(160.0 / std::log2(g.beta)) + 1;
    const int pmax = g.pmax;
    const size_t P1 = (size_t)pmax + 1;
    const Oz1Fp64Norm nm = oz1_fp64_norm(A, Bm, g);
    const std::vector<double> e2r = oz1_fp64_indep(A, Bm, g, nm);
    const double nC = estimate_product_norm(A, Bm, M, N, K, nm.rgA, nm.rgB);  // low-biased estimate of ||C||_F^2
    const Oz1Fp64Sampled sm = oz1_fp64_sampled(A, Bm, g, nm, nC);
    const double tgt = std::ldexp(1.0, -106);              // (2^-53)^2
    std::vector<double> r1(P1, INFINITY), e2sv(P1, -1.0);  // r1[q]: squared relative error of (1)
    for (int q = 2; q <= pmax; q++) r1[q] = nC > 0 ? e2r[q] / nC : (e2r[q] > 0 ? INFINITY : 0.0);
    return oz1_fp64_search(r1, e2sv, sm, pmax, tgt);
}
}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
