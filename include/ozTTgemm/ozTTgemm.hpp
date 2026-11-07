// SPDX-License-Identifier: Apache-2.0
//
// ozTTgemm: FP64 matrix multiplication on Tenstorrent Blackhole with the Ozaki schemes.
//
// The device multiplies low-precision matrices exactly; the library splits the FP64
// operands into such matrices (slices for scheme I, residues for scheme II), runs the
// partial products on the device and reconstructs the FP64 result there as well.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#define OZTTGEMM_VERSION_MAJOR 0
#define OZTTGEMM_VERSION_MINOR 1
#define OZTTGEMM_VERSION_PATCH 0

namespace ozTTgemm {

// Ozaki scheme I splits each operand into slices and multiplies slice pairs;
// scheme II multiplies residues modulo several small moduli and applies the CRT.
enum class Scheme { I, II };

// Number format of the device matrix multiplications. BFP8, BFP4 and BFP2 are
// TT-Metalium's block floating-point formats; TF32 is its 32-bit float path.
enum class Format { BF16, FP16, TF32, BFP8, FP8, INT8, BFP4, BFP2 };

// Number of reconstruction passes of scheme II. Automatic selects one pass when an
// error estimate from the input statistics allows it and two otherwise (README.md).
enum class Reconstruction { Automatic, OnePass, TwoPass };

// The options of the scheme not selected must keep their defaults; otherwise the
// call is rejected.
struct Options {
    Scheme scheme = Scheme::II;
    Format format = Format::BFP8;
    // Scheme I only. Zero selects the number of slices from the input; a positive
    // value (at most 256) fixes it.
    std::uint32_t slices = 0;
    // Scheme II only. Zero leaves the modulus selection to the library; a positive
    // value (at most 24) limits the number of CRT moduli.
    std::uint32_t max_moduli = 0;
    // Scheme II only.
    Reconstruction reconstruction = Reconstruction::Automatic;
};

struct DeviceOptions {
    // TT-Metalium device ID.
    int device_id = 0;
    // Directory containing common/kernels, ozaki1/kernels and ozaki2/kernels.
    // Empty selects the build-time source directory or the installed data directory.
    // The directory is resolved when the Context is constructed.
    std::string kernel_directory;
};

// What a call actually used.
struct Info {
    Scheme scheme;
    Format format;
    std::uint32_t slices = 0;                 // scheme I: number of slices
    std::uint32_t moduli = 0;                 // scheme II: number of CRT moduli
    std::uint32_t integer_bits = 0;           // scheme II: bits kept of each scaled operand
    std::uint32_t reconstruction_passes = 0;  // scheme II: 1 or 2
    // Dimensions after zero padding (m and n to multiples of 64, k to a multiple of 32).
    std::size_t padded_m = 0, padded_n = 0, padded_k = 0;
};

// False for formats left out of the build (the FORMATS setting of the Makefile).
bool is_available(Format format) noexcept;
// "BF16", "FP16", "TF32", "BFP8", "FP8", "INT8", "BFP4" or "BFP2".
const char* format_name(Format format) noexcept;

// Owns a single Blackhole device, opened lazily on the first valid GEMM.
// Calls on contexts in one process are serialized. Keep the Context alive to
// reuse the device; intermediate buffers are released after each call.
class Context {
public:
    // Resolves the kernel directory at once (std::invalid_argument if the requested
    // directory lacks the kernels or device_id is negative, std::runtime_error if no
    // default location has the kernels); the device is opened by the first valid GEMM.
    explicit Context(DeviceOptions options = {});
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    // C = A*B for contiguous row-major FP64 matrices A[m,k], B[k,n], C[m,n].
    // Dimensions must be positive and the spans must have exactly m*k, k*n and m*n
    // elements. Internal zero padding is removed on return. All input is consumed
    // before C is written, so input and output may overlap.
    // Inputs must be finite, and every nonzero row maximum of A and column maximum
    // of B must lie in [2^-126, largest finite BF16] (the device normalizes by BF16
    // maxima). Dimensions whose tile counts exceed the device's 32-bit offsets are
    // rejected.
    // Numerical limits (README.md has the details): subnormal elements of an
    // admissible row or column count as zero (a row or column whose maximum is
    // subnormal is rejected by the range check above);
    // an element far below the maximum of its row of A or column of B keeps fewer
    // bits or is dropped, and no check detects this; a result that would be subnormal
    // is zero; cancellation is not protected, and automatic precision is a model
    // estimate, not a correctly-rounded guarantee.
    // Throws std::invalid_argument for invalid input (before the device is opened
    // and before C is written), std::runtime_error for configurations that cannot
    // be executed and for failures of the TT-Metalium runtime, and std::bad_alloc
    // if host memory runs out. The library itself never terminates the process or
    // writes to stdout or stderr (TT-Metalium may log according to its own settings).
    Info gemm(std::span<const double> a, std::span<const double> b, std::span<double> c, std::size_t m, std::size_t n,
              std::size_t k, const Options& options = {});

    // Same as gemm(), returning C as a new vector.
    std::vector<double> multiply(std::span<const double> a, std::span<const double> b, std::size_t m, std::size_t n,
                                 std::size_t k, const Options& options = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ozTTgemm
