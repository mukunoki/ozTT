// SPDX-License-Identifier: Apache-2.0
#pragma once
// Small host-side bit conversions: a float / double as its bit pattern, and the final scale of the
// reconstruction as a 64-bit significand. Independent of the device API.
//
// They are used where the host passes floating-point values to kernels as (uint32_t) compile-time or
// runtime arguments.

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// The host arithmetic in long double (the scale below, the sums of common/norm_estimate.hpp) assumes
// the x86 extended format with a 64-bit significand.
static_assert(std::numeric_limits<long double>::digits == 64,
              "ozTTgemm requires a long double with a 64-bit significand (x86 extended precision)");

inline uint32_t f32_bits(float f) { return std::bit_cast<uint32_t>(f); }
inline uint64_t f64_bits(double x) { return std::bit_cast<uint64_t>(x); }

// Splits a positive long double w into a 64-bit significand W = (hi:lo), whose most significant bit is
// set, and an exponent q, so that w = W * 2^q. The final scale of the reconstruction is not a power of
// two in general (radix^-(L+2) in Ozaki scheme I, the constant KM in Ozaki scheme II); it is passed in
// this form so that the kernel can multiply by it as an integer (multiplying once by an FP64 constant
// would put a relative error of 2^-53 uniformly on every element; pack_f64 in
// common/kernels/f64_tf.h). The long double has a 64-bit significand (checked above), so this split adds
// no error.
struct OzScale64 {
    uint32_t hi, lo;
    int32_t q;
};
inline OzScale64 oz_split_scale_ld(long double w) {
    int e;
    const long double f = frexpl(w, &e);
    const unsigned long long W = (unsigned long long)ldexpl(f, 64);
    return {(uint32_t)(W >> 32), (uint32_t)W, e - 64};
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
