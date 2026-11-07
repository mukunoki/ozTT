// SPDX-License-Identifier: Apache-2.0
#pragma once
// Input order, Dot2 reference and accuracy metric copied from the paper's
// ozgen1 implementation. This helper has no experimental environment readers.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace paper_eval {
inline void inputs(std::vector<double>& a, std::vector<double>& b, unsigned n, double phi) {
    std::mt19937 rng(7);
    a.resize(std::size_t(n) * n);
    b.resize(a.size());
    if (phi < 0) {
        std::uniform_real_distribution<double> u(-1, 1);
        for (auto& x : a) x = u(rng);
        for (auto& x : b) x = u(rng);
    } else {
        std::uniform_real_distribution<double> u(0, 1);
        std::normal_distribution<double> normal(0, 1);
        auto value = [&] {
            const double v = u(rng);
            const double z = normal(rng);
            return (v - 0.5) * std::exp(phi * z);
        };
        for (auto& x : a) x = value();
        for (auto& x : b) x = value();
    }
}
// Relative Frobenius error and effective bits.
struct Acc {
    double rel_fro, ebits;
};
// FNV-1a-style hash of the output bit patterns, bytes in ascending order. The initial
// value is the research implementation's (not the FNV offset basis) and is kept so
// that archived hashes stay comparable.
inline std::uint64_t output_hash(const std::vector<double>& C) {
    uint64_t h = 1469598103934665603ull;
    for (size_t z = 0; z < C.size(); z++) {
        uint64_t u;
        std::memcpy(&u, &C[z], 8);
        for (int b = 0; b < 64; b += 8) {
            h ^= (u >> b) & 0xFFu;
            h *= 1099511628211ull;
        }
    }
    return h;
}
inline void reference_gemm(const std::vector<double>& A, const std::vector<double>& B, uint32_t M, uint32_t K,
                           uint32_t N, std::vector<double>& Chi, std::vector<double>& Clo) {
    Chi.assign((size_t)M * N, 0.0);
    Clo.assign((size_t)M * N, 0.0);

    const double* __restrict__ Ap = A.data();
    const double* __restrict__ Bp = B.data();
    double* __restrict__ Cp = Chi.data();
    double* __restrict__ Lp = Clo.data();
    constexpr uint32_t IB = 8, JB = 512;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ib = 0; ib < (int64_t)M; ib += IB) {
        const uint32_t i1 = (uint32_t)std::min<int64_t>(ib + IB, (int64_t)M);
        for (uint64_t j0 = 0; j0 < N; j0 += JB) {
            const uint32_t j1 = (uint32_t)std::min<uint64_t>(j0 + JB, N);
            for (uint32_t k = 0; k < K; ++k) {
                const double* __restrict__ Br = Bp + (size_t)k * N;
                for (uint32_t i = (uint32_t)ib; i < i1; ++i) {
                    const double a = Ap[(size_t)i * K + k];
                    double* __restrict__ chi = Cp + (size_t)i * N;
                    double* __restrict__ clo = Lp + (size_t)i * N;
                    for (uint32_t j = (uint32_t)j0; j < j1; ++j) {
                        const double b = Br[j];
                        const double p = a * b, e = std::fma(a, b, -p);
                        const double c0 = chi[j];
                        const double s = c0 + p, bb = s - c0;
                        clo[j] += (c0 - (s - bb)) + (p - bb) + e;
                        chi[j] = s;
                    }
                }
            }
        }
    }
}
inline std::pair<double, double> accuracy_reference(double hi, double lo) {
    const double s = hi + lo, bb = s - hi;
    return {s, (hi - (s - bb)) + (lo - bb)};
}
inline Acc accuracy(const std::vector<double>& C, const std::vector<double>& Chi, const std::vector<double>& Clo) {
    if (C.empty() || C.size() != Chi.size() || C.size() != Clo.size())
        throw std::runtime_error("accuracy: empty or mismatched arrays (C " + std::to_string(C.size()) + ", Chi " +
                                 std::to_string(Chi.size()) + ", Clo " + std::to_string(Clo.size()) + " elements)");
    double num = 0, den = 0, mx = 0;
    for (size_t z = 0; z < C.size(); z++) {
        const auto [hi, lo] = accuracy_reference(Chi[z], Clo[z]);
        mx = std::max(mx, std::max(std::fabs(hi), std::fabs((C[z] - hi) - lo)));
    }

    int sce = (mx > 0 && std::isfinite(mx)) ? -std::ilogb(mx) : 0;
    const int sce1 = std::min(sce, 1023);
    const double sc1 = std::ldexp(1.0, sce1), sc2 = std::ldexp(1.0, sce - sce1);
    for (size_t z = 0; z < C.size(); z++) {
        const auto [hi, lo] = accuracy_reference(Chi[z], Clo[z]);
        const double g = hi;
        const double dd = (C[z] - hi) - lo;
        const double ds = (dd * sc1) * sc2, gs = (g * sc1) * sc2;
        num += ds * ds;
        den += gs * gs;
    }

    double rel;
    if (!std::isfinite(num) || !std::isfinite(den))
        rel = std::numeric_limits<double>::infinity();
    else if (den > 0)
        rel = std::sqrt(num) / std::sqrt(den);
    else
        rel = (num > 0) ? std::numeric_limits<double>::infinity() : 0.0;
    const double ebits = std::isfinite(rel) ? (rel > 0 ? -std::log2(rel) : 99.0) : -1.0;
    return {rel, ebits};
}
}  // namespace paper_eval
