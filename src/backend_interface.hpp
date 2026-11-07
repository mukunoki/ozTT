// SPDX-License-Identifier: Apache-2.0
//
// Interface between the public Context (context.cpp) and the backends.
//
// A backend is one (scheme, format) combination. ozaki1.cpp and ozaki2.cpp are each
// compiled once per format into the namespace ttgemm::detail::TTGEMM_BACKEND_NS
// (ozaki1_bf16, ozaki2_bfp8, ...), and dispatch.cpp maps (format, scheme) to the
// backend's run() function.
#pragma once
#include <ttgemm/ttgemm.hpp>
#include <tt-metalium/distributed.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace ttgemm::detail {
using Device = std::shared_ptr<tt::tt_metal::distributed::MeshDevice>;

// One GEMM after validation and zero padding: A is m x k and B is k x n, row-major,
// with m and n multiples of 64 and k a multiple of 32.
struct Problem {
    std::uint32_t m, n, k;
    const std::vector<double>& a;
    const std::vector<double>& b;
    const Options& options;
};

// The padded m x n result (row-major) and what the backend selected.
struct Output {
    std::vector<double> values;
    Info info;
};

using Backend = Output (*)(Device&, const Problem&);

// Throws std::invalid_argument if the format was left out of the build.
Backend backend_for(Format format, Scheme scheme);

// Absolute path of a kernel source, given relative to the kernel root
// (for example "ozaki2/kernels/recon_writer.cpp"). Valid only during a call.
std::string kernel_path(std::string_view relative);

// Sets the kernel root for the duration of one call. Paths passed to Metalium are
// absolute, so its run-time kernel compilation is independent of the working directory.
class KernelScope {
public:
    explicit KernelScope(const std::string& root);
    ~KernelScope();
    KernelScope(const KernelScope&) = delete;
    KernelScope& operator=(const KernelScope&) = delete;

private:
    const std::string* previous_;
};
}  // namespace ttgemm::detail
