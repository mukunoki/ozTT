// SPDX-License-Identifier: Apache-2.0
//
// Public interface: argument validation, zero padding, kernel lookup, device lifetime.
// The calculation itself is done by the backend selected in dispatch.cpp.
#include "backend_interface.hpp"
#include "evaluation_hooks.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace ttgemm {
namespace {
// Serializes device use by all contexts of the process.
std::mutex runtime_mutex;
thread_local const std::string* active_kernel_root = nullptr;

// Element count a*b, rejecting zero dimensions and sizes whose byte count overflows.
std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0 || a > std::numeric_limits<std::size_t>::max() / sizeof(double) / b)
        throw std::invalid_argument("matrix dimensions are zero or overflow the address space");
    return a * b;
}
// Rounds a dimension up to a multiple of quantum; device indices are 32-bit.
std::uint32_t padded(std::size_t value, std::size_t quantum) {
    constexpr std::size_t limit = std::numeric_limits<std::int32_t>::max();
    if (value == 0 || value > limit - quantum)
        throw std::invalid_argument("matrix dimension exceeds the supported device index range");
    return static_cast<std::uint32_t>((value + quantum - 1) / quantum * quantum);
}
// The device normalizes each row of A and each column of B by its largest magnitude,
// held as BF16, so every nonzero maximum must be a normal BF16 value.
void check_values(std::span<const double> a, std::span<const double> b, std::size_t m, std::size_t n, std::size_t k) {
    constexpr double maximum = 0x1.fep127;  // Largest finite BF16, used by device normalization.
    constexpr double minimum = 0x1p-126;
    const auto check_max = [&](double x) {
        if (x != 0 && (x < minimum || x > maximum))
            throw std::invalid_argument(
                "each nonzero row maximum of A and column maximum of B must be in [2^-126, max_BF16]");
    };
    for (std::size_t i = 0; i < m; ++i) {
        double mx = 0;
        for (std::size_t j = 0; j < k; ++j) {
            const double x = a[i * k + j];
            if (!std::isfinite(x)) throw std::invalid_argument("A contains NaN or infinity");
            mx = std::max(mx, std::abs(x));
        }
        check_max(mx);
    }
    std::vector<double> maxima(n, 0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = b[i * n + j];
            if (!std::isfinite(x)) throw std::invalid_argument("B contains NaN or infinity");
            maxima[j] = std::max(maxima[j], std::abs(x));
        }
    for (double x : maxima) check_max(x);
}
// Kernel root: the requested directory, else the source tree the library was built
// from, else the install prefix. Each candidate must contain the three kernel sets.
std::string resolve_kernels(const std::string& requested) {
    namespace fs = std::filesystem;
    const auto valid = [](const fs::path& path) {
        return fs::is_regular_file(path / "common/kernels/scale_f64_reader.cpp") &&
               fs::is_regular_file(path / "ozaki1/kernels/recon_f64_writer.cpp") &&
               fs::is_regular_file(path / "ozaki2/kernels/recon_writer.cpp");
    };
    if (!requested.empty()) {
        const auto path = fs::absolute(requested);
        if (!valid(path)) throw std::invalid_argument("kernel_directory does not contain the ttgemm compute kernels");
        return path.lexically_normal().string();
    }
    for (const fs::path& path : {fs::path(TTGEMM_SOURCE_KERNEL_DIR), fs::path(TTGEMM_INSTALL_KERNEL_DIR)})
        if (valid(path)) return fs::absolute(path).lexically_normal().string();
    throw std::runtime_error("ttgemm kernels were not found; set DeviceOptions::kernel_directory");
}
// Enum values are checked too: a caller can pass any integer through a cast.
void check_options(const Options& o) {
    if (o.scheme != Scheme::I && o.scheme != Scheme::II) throw std::invalid_argument("unknown Ozaki scheme");
    if (!is_available(o.format)) throw std::invalid_argument("format is unknown or was not built");
    if (o.reconstruction != Reconstruction::Automatic && o.reconstruction != Reconstruction::OnePass &&
        o.reconstruction != Reconstruction::TwoPass)
        throw std::invalid_argument("unknown reconstruction mode");
    if (o.slices > 256) throw std::invalid_argument("slices must be zero (automatic) or at most 256");
    if (o.max_moduli > 24) throw std::invalid_argument("max_moduli exceeds the supported bound of 24");
    if (o.scheme == Scheme::I && (o.max_moduli != 0 || o.reconstruction != Reconstruction::Automatic))
        throw std::invalid_argument("max_moduli and reconstruction apply only to Ozaki II");
    if (o.scheme == Scheme::II && o.slices != 0) throw std::invalid_argument("slices applies only to Ozaki I");
}
// Padded dimensions and the output size of a call whose arguments passed validation.
struct Checked {
    std::uint32_t mp, np, kp;
    std::size_t c_size;
};
// Every check of a call, before anything is allocated: options, dimensions, span sizes
// (the output span, if there is one already, must hold m*n elements), the padded
// dimensions, every tile index and product-group offset of which must fit in the
// 32-bit device ABI, and finally the input values.
Checked validate(const Options& options, std::span<const double> a, std::span<const double> b,
                 std::optional<std::size_t> c_size, std::size_t m, std::size_t n, std::size_t k) {
    check_options(options);
    if (a.size() != checked_product(m, k) || b.size() != checked_product(k, n))
        throw std::invalid_argument("matrix span sizes do not match m, n, k");
    const std::size_t mn = checked_product(m, n);
    if (c_size && *c_size != mn) throw std::invalid_argument("matrix span sizes do not match m, n, k");
    const Checked v{padded(m, 64), padded(n, 64), padded(k, 32), mn};
    for (auto count : {std::uint64_t(v.mp / 32) * (v.kp / 32), std::uint64_t(v.kp / 32) * (v.np / 32),
                       std::uint64_t(v.mp / 32) * (v.np / 32)})
        if (count > std::numeric_limits<std::uint32_t>::max() / 4096u)
            throw std::invalid_argument("matrix is too large for the device's 32-bit offsets");
    check_values(a, b, m, n, k);
    return v;
}
}  // namespace

struct Context::Impl {
    explicit Impl(DeviceOptions o) : options(std::move(o)), kernels(resolve_kernels(options.kernel_directory)) {
        if (options.device_id < 0) throw std::invalid_argument("device_id must be nonnegative");
    }
    // The GEMM of validated arguments: padding, device, backend, cropping.
    Info execute(std::span<const double> a, std::span<const double> b, std::span<double> c, std::size_t m,
                 std::size_t n, std::size_t k, const Options& options, const Checked& v);
    DeviceOptions options;
    std::string kernels;
    detail::Device device;
};
Context::Context(DeviceOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Context::~Context() {
    std::lock_guard lock(runtime_mutex);
    if (impl_->device) {
        // Destructors cannot propagate a runtime shutdown failure; the device is closed
        // even when the queue cannot be drained.
        try {
            tt::tt_metal::distributed::Finish(impl_->device->mesh_command_queue());
        } catch (...) {}
        try {
            impl_->device->close();
        } catch (...) {}
    }
}
Info Context::Impl::execute(std::span<const double> a, std::span<const double> b, std::span<double> c, std::size_t m,
                            std::size_t n, std::size_t k, const Options& options, const Checked& v) {
    const auto mp = v.mp, np = v.np, kp = v.kp;
    // Copy into zero-padded matrices before taking the lock; C may alias A or B.
    std::vector<double> ap(checked_product(mp, kp), 0), bp(checked_product(kp, np), 0);
    for (std::size_t i = 0; i < m; ++i) std::copy_n(a.data() + i * k, k, ap.data() + i * kp);
    for (std::size_t i = 0; i < k; ++i) std::copy_n(b.data() + i * n, n, bp.data() + i * np);
    std::lock_guard lock(runtime_mutex);
    detail::KernelScope scope(kernels);
    if (!device) {
        auto opened = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(this->options.device_id);
        if (opened->arch() != tt::ARCH::BLACKHOLE) {
            opened->close();
            throw std::runtime_error("ttgemm supports Blackhole devices only");
        }
        device = std::move(opened);
    }
    // The evaluator's records of device steps live from here to the end of the call's
    // whole pass; a call that failed earlier inside the same Scope leaves none behind.
    if (detail::evaluation::active) detail::evaluation::active->steps.clear();
    const detail::Problem problem{mp, np, kp, ap, bp, options};
    detail::Output output = detail::backend_for(options.format, options.scheme)(device, problem);
    if (output.values.size() != std::size_t(mp) * np)
        throw std::runtime_error("backend returned an invalid output shape");
    for (std::size_t i = 0; i < m; ++i) std::copy_n(output.values.data() + i * np, n, c.data() + i * n);
    output.info.padded_m = mp;
    output.info.padded_n = np;
    output.info.padded_k = kp;
    return output.info;
}
Info Context::gemm(std::span<const double> a, std::span<const double> b, std::span<double> c, std::size_t m,
                   std::size_t n, std::size_t k, const Options& options) {
    const Checked v = validate(options, a, b, c.size(), m, n, k);
    return impl_->execute(a, b, c, m, n, k, options, v);
}
std::vector<double> Context::multiply(std::span<const double> a, std::span<const double> b, std::size_t m,
                                      std::size_t n, std::size_t k, const Options& options) {
    // Every check precedes the allocation of the output.
    const Checked v = validate(options, a, b, std::nullopt, m, n, k);
    std::vector<double> c(v.c_size);
    impl_->execute(a, b, c, m, n, k, options, v);
    return c;
}

namespace detail {
KernelScope::KernelScope(const std::string& root) : previous_(active_kernel_root) { active_kernel_root = &root; }
KernelScope::~KernelScope() { active_kernel_root = previous_; }
std::string kernel_path(std::string_view relative) {
    if (!active_kernel_root) throw std::logic_error("kernel path requested outside a ttgemm call");
    return *active_kernel_root + "/" + std::string(relative);
}
}  // namespace detail
}  // namespace ttgemm
