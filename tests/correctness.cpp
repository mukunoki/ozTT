// SPDX-License-Identifier: Apache-2.0
//
// Correctness checks: API input validation (no device needed, --host-only) and small
// device GEMMs compared with a long-double CPU reference for every scheme and format.
#include <ozTTgemm/ozTTgemm.hpp>
#include "evaluation_hooks.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
// The call must be rejected with std::invalid_argument.
template <class F>
void invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}
// The call must be rejected with std::runtime_error (a configuration that cannot be executed).
template <class F>
void unsupported(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("unsupported configuration was reported as invalid input");
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("unsupported configuration was accepted");
}
// Bit-identical contents (operator== on doubles would equate +0 and -0).
bool same_bits(const std::vector<double>& x, const std::vector<double>& y) {
    return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0);
}
// Invalid arguments are rejected before the device is opened and leave C untouched.
void host_validation(ozTTgemm::Context& context, ozTTgemm::Format format) {
    std::vector<double> a(4, 1), b(4, 1), c(4, 123);
    ozTTgemm::Options opt;
    opt.format = format;
    invalid([&] { context.gemm(a, b, c, 0, 2, 2, opt); });
    invalid([&] { context.gemm(a, b, c, 2, 3, 2, opt); });
    invalid([&] {
        auto o = opt;
        o.scheme = static_cast<ozTTgemm::Scheme>(100);
        context.gemm(a, b, c, 2, 2, 2, o);
    });
    invalid([&] {
        auto o = opt;
        o.format = static_cast<ozTTgemm::Format>(100);
        context.gemm(a, b, c, 2, 2, 2, o);
    });
    invalid([&] {
        auto o = opt;
        o.slices = 10;
        context.gemm(a, b, c, 2, 2, 2, o);
    });
    invalid([&] {
        auto o = opt;
        o.scheme = ozTTgemm::Scheme::I;
        o.reconstruction = ozTTgemm::Reconstruction::TwoPass;
        context.gemm(a, b, c, 2, 2, 2, o);
    });
    a[0] = std::numeric_limits<double>::quiet_NaN();
    invalid([&] { context.gemm(a, b, c, 2, 2, 2, opt); });
    a[0] = 1;
    b[0] = std::numeric_limits<double>::infinity();
    invalid([&] { context.gemm(a, b, c, 2, 2, 2, opt); });
    b[0] = 1;
    a.assign(4, 0x1p-130);
    invalid([&] { context.gemm(a, b, c, 2, 2, 2, opt); });
    require(std::all_of(c.begin(), c.end(), [](double x) { return x == 123; }), "invalid input modified output");
}
// The evaluator hooks (repeated, timed execution and the whole-pass replay) must give
// exactly the bits of an ordinary call, and their checks must pass.
void evaluation_hooks(ozTTgemm::Context& context, ozTTgemm::Format format) {
    namespace ev = ozTTgemm::detail::evaluation;
    std::mt19937_64 rng(731);
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<double> a(64 * 64), b(a.size()), expected(a.size()), measured(a.size());
    for (auto& x : a) x = uniform(rng);
    for (auto& x : b) x = uniform(rng);
    for (auto scheme : {ozTTgemm::Scheme::I, ozTTgemm::Scheme::II}) {
        ozTTgemm::Options o;
        o.scheme = scheme;
        o.format = format;
        context.gemm(a, b, expected, 64, 64, 64, o);
        ev::Settings settings;
        settings.repeats = 2;
        std::array<unsigned, ev::Count> intervals{};
        settings.interval = [&](ev::Stage stage, double begin, double end) {
            require(end >= begin, "invalid timing interval");
            ++intervals[stage];
        };
        ozTTgemm::Info info;
        {
            ev::Scope scope(settings);
            info = context.gemm(a, b, measured, 64, 64, 64, o);
            require(settings.steps.empty(), "whole-pass records were not released after the replay");
        }
        require(ev::active == nullptr, "evaluation settings leaked into normal GEMM");
        require(same_bits(expected, measured), "profiling/repetition changed the result bits");
        require(settings.seconds[ev::Slice] > 0 && settings.seconds[ev::Gemm] > 0 && settings.seconds[ev::Recon] > 0,
                "a required phase was not timed");
        require(settings.whole_checked && settings.whole_seconds > 0,
                "the whole-pass replay was not checked and timed");
        if (scheme == ozTTgemm::Scheme::II)
            require(intervals[ev::Recon] == info.reconstruction_passes, "reconstruction timing included another phase");
    }
    {  // phase timing alone, without the whole-pass replay
        ev::Settings phases;
        phases.repeats = 1;
        phases.whole = false;
        ozTTgemm::Options o;
        o.format = format;
        o.scheme = ozTTgemm::Scheme::II;
        {
            ev::Scope scope(phases);
            context.gemm(a, b, measured, 64, 64, 64, o);
        }
        require(same_bits(expected, measured) && std::isnan(phases.whole_seconds),
                "phase-only profiling changed the result");
    }
    {  // a Settings reused by a second Scope starts from reset results (scheme I has no scale stage)
        ev::Settings reused;
        reused.repeats = 1;
        ozTTgemm::Options o;
        o.format = format;
        o.scheme = ozTTgemm::Scheme::II;
        {
            ev::Scope scope(reused);
            context.gemm(a, b, measured, 64, 64, 64, o);
        }
        require(reused.seconds[ev::Scale] > 0, "the scale phase of scheme II was not timed");
        o.scheme = ozTTgemm::Scheme::I;
        {
            ev::Scope scope(reused);
            context.gemm(a, b, measured, 64, 64, 64, o);
        }
        require(reused.seconds[ev::Scale] == 0 && reused.seconds[ev::Compose] == 0,
                "results of an earlier Scope were kept");
    }
    {  // a Scope cannot be opened with zero repetitions (the hooks divide by repeats)
        ev::Settings zero;
        zero.repeats = 0;
        invalid([&] { ev::Scope scope(zero); });
        require(ev::active == nullptr, "a rejected evaluation scope was activated");
    }
    ev::Settings fixed;
    fixed.repeats = 1;
    fixed.fixed_bits = 54;
    ozTTgemm::Options o;
    o.format = format;
    o.reconstruction = ozTTgemm::Reconstruction::OnePass;
    {
        ev::Scope scope(fixed);
        const auto info = context.gemm(a, b, measured, 64, 64, 64, o);
        require(info.integer_bits == 54 && info.reconstruction_passes == 1, "paper fixed precision was not applied");
    }
}
// Input patterns of check(). All but the last three start from uniform values in [-1,1).
enum Pattern {
    Uniform,           // A and B uniform
    ZeroA,             // A is zero
    Nonnegative,       // |A|, |B|
    Scaled,            // rows of A and columns of B scaled by powers of two (2^-15..2^15)
    Cancellation,      // rows of A alternate +1/-1, B is one: exact cancellation
    ZeroRowAndColumn,  // first row of A and first column of B are zero
    SmallestMaximum,   // A = 2^-126, the lower end of the normalization range
    LargestMaximum,    // A = the largest finite BF16 value, the upper end
};
constexpr const char* pattern_name[] = {"uniform",      "zero-A",          "nonnegative",      "scaled",
                                        "cancellation", "zero-row-column", "smallest-maximum", "largest-maximum"};
// One GEMM against a long-double reference; the relative Frobenius error must be
// within 8 ulp of FP64.
void check(ozTTgemm::Context& context, ozTTgemm::Options opt, std::size_t m, std::size_t n, std::size_t k,
           Pattern mode) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::vector<double> a(m * k), b(k * n), c(m * n, 123);
    for (auto& x : a) x = mode == ZeroA ? 0 : uniform(rng);
    for (auto& x : b) x = uniform(rng);
    if (mode == Nonnegative) {
        for (auto& x : a) x = std::abs(x);
        for (auto& x : b) x = std::abs(x);
    }
    if (mode == Scaled) {
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < k; ++j) a[i * k + j] = std::ldexp(a[i * k + j], int(i % 31) - 15);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < n; ++j) b[i * n + j] = std::ldexp(b[i * n + j], int(j % 27) - 13);
    }
    if (mode == ZeroRowAndColumn) {
        for (std::size_t j = 0; j < k; ++j) a[j] = 0;
        for (std::size_t i = 0; i < k; ++i) b[i * n] = 0;
    }
    if (mode == SmallestMaximum) {
        std::fill(a.begin(), a.end(), 0x1p-126);
        std::fill(b.begin(), b.end(), 1);
    }
    if (mode == LargestMaximum) {
        std::fill(a.begin(), a.end(), 0x1.fep127);
        std::fill(b.begin(), b.end(), 1);
    }
    if (mode == Cancellation) {
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < k; ++j) a[i * k + j] = j % 2 ? -1 : 1;
        std::fill(b.begin(), b.end(), 1);
    }
    const auto info = context.gemm(a, b, c, m, n, k, opt);
    long double error = 0, norm = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            long double ref = 0;
            for (std::size_t l = 0; l < k; ++l) ref += static_cast<long double>(a[i * k + l]) * b[l * n + j];
            require(std::isfinite(c[i * n + j]), "nonfinite output");
            const long double delta = c[i * n + j] - ref;
            error += delta * delta;
            norm += ref * ref;
        }
    const double relative = norm > 0 ? double(std::sqrt(error / norm)) : double(std::sqrt(error));
    std::cout << (opt.scheme == ozTTgemm::Scheme::I ? "I " : "II ") << ozTTgemm::format_name(opt.format) << ' ' << m
              << 'x' << n << 'x' << k << ' ' << pattern_name[mode] << " error=" << relative << " slices=" << info.slices
              << " moduli=" << info.moduli << " passes=" << info.reconstruction_passes << '\n';
    require(relative <= 8 * std::numeric_limits<double>::epsilon(), "result differs from long-double reference");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const ozTTgemm::Format formats[] = {ozTTgemm::Format::BF16, ozTTgemm::Format::FP16, ozTTgemm::Format::TF32,
                                            ozTTgemm::Format::BFP8, ozTTgemm::Format::FP8,  ozTTgemm::Format::INT8,
                                            ozTTgemm::Format::BFP4, ozTTgemm::Format::BFP2};
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--host-only"))
            throw std::invalid_argument("usage: ozTTgemm_test [--host-only]");
        ozTTgemm::Context context;
        auto first = ozTTgemm::Format::BF16;
        for (auto f : formats)
            if (ozTTgemm::is_available(f)) {
                first = f;
                break;
            }
        host_validation(context, first);
        if (argc == 2 && std::string(argv[1]) == "--host-only") {
            std::cout << "input validation passed\n";
            return 0;
        }
        for (auto f : formats)
            if (ozTTgemm::is_available(f))
                for (auto scheme : {ozTTgemm::Scheme::I, ozTTgemm::Scheme::II}) {
                    ozTTgemm::Options o;
                    o.scheme = scheme;
                    o.format = f;
                    check(context, o, 64, 64, 64, Uniform);
                    check(context, o, 3, 5, 7, Nonnegative);
                }
        ozTTgemm::Options o;
        o.format = first;
        for (auto scheme : {ozTTgemm::Scheme::I, ozTTgemm::Scheme::II}) {
            o.scheme = scheme;
            check(context, o, 65, 67, 97, Scaled);
            check(context, o, 9, 11, 17, ZeroA);
            check(context, o, 3, 5, 64, Cancellation);
            check(context, o, 3, 5, 7, ZeroRowAndColumn);
            check(context, o, 1, 1, 1, SmallestMaximum);
            check(context, o, 1, 1, 1, LargestMaximum);
        }
        o.scheme = ozTTgemm::Scheme::I;
        const unsigned fixed_slices[] = {10, 10, 10, 10, 12, 8, 16, 59};
        o.slices = fixed_slices[static_cast<unsigned>(first)];
        check(context, o, 3, 5, 7, Uniform);
        o = {};
        o.format = first;
        std::vector<double> overlap{1, 2, 3, 4};
        const std::vector<double> identity{1, 0, 0, 1};
        context.gemm(overlap, identity, overlap, 2, 2, 2, o);
        require(overlap == std::vector<double>{1, 2, 3, 4}, "overlapping input/output failed");
        o.scheme = ozTTgemm::Scheme::II;
        check(context, o, 5056, 64, 32, Uniform);  // requires a panel fallback from the largest divisor
        o.reconstruction = ozTTgemm::Reconstruction::TwoPass;
        check(context, o, 64, 64, 64, Uniform);
        // Two moduli cannot cover an FP64-equivalent width: rejected, and the output is untouched.
        o = {};
        o.format = first;
        o.max_moduli = 2;
        {
            std::vector<double> a(64 * 64, 1), b(64 * 64, 1), c(64 * 64, 123);
            unsupported([&] { context.gemm(a, b, c, 64, 64, 64, o); });
            require(std::all_of(c.begin(), c.end(), [](double x) { return x == 123; }),
                    "rejected configuration modified output");
        }
        if (ozTTgemm::is_available(ozTTgemm::Format::BF16)) {
            o = {};
            o.scheme = ozTTgemm::Scheme::I;
            o.format = ozTTgemm::Format::BF16;
            check(context, o, 64, 64, 32768, Nonnegative);  // crosses the exact-accumulator K-block limit
        }
        evaluation_hooks(context, first);
        std::cout << "all correctness checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << '\n';
        return 1;
    }
}
