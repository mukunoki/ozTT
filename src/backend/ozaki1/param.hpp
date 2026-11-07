// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: per-format radix and slice encoding, and the rule for the digit bound.
//
// ozaki1/format.hpp instantiates the table for the format of the translation unit
// (OZ1_PARAM, OZ1_DMAX).

#include <cstdint>
#include "common/format.hpp"

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// Per-format parameters of scheme I. The slicing and reconstruction algorithms are the
// same for every format; what differs per format is this table, OZ_FMT
// (common/format.hpp) and the kernels that the host selects from them: the
// sign-magnitude slicer of INT8 (signmag_slices below) and the reconstruction kernels
// for Int32 partial sums (recon_build_prog in ozaki1/recon.hpp).
// The digit bound d and the radix beta (the weight of a digit) are different quantities,
// related by beta = 2d (oz1_digit_max below).
//   format  GEMM input  partial products  radix  digits
//   BF16    Float16_b   Float32           64     +-32
//   FP16    Float16     Float32           64     +-32   (slices written as binary16 bit patterns, UInt16)
//   TF32    Float32     Float32           64     +-32   (OzFmt::FP32; the matrix engine reads it as TF32)
//   BFP8    Bfp8_b      Float32           64     +-32
//   FP8     Fp8_e4m3    Float32           32     +-16
//   BFP4    Bfp4_b      Float32           14     +-7
//   BFP2    Bfp2_b      Float32           2      +-1
//   INT8    Int8        Int32             254    +-127  (sign-magnitude byte slices)
// The math fidelity is not in the table: it follows from the digit bound (OZ1_FID in
// ozaki1/format.hpp), which gives LoFi for every format except INT8 (HiFi4).
// exact_int_max of a format (common/format.hpp) is the largest |digit| that the format
// represents exactly. It bounds the digit, not the radix; a digit beyond it would silently
// turn into a different number.
struct Oz1Param {
    uint32_t radix;  // radix (weight of a digit)

    // True if the slices are written as sign-magnitude UInt8 (INT8 path only; slicer
    // slice_tf_sm8.cpp). The other formats write the digit value itself in the GEMM input
    // format (slice_tf_num.cpp). The reconstruction kernel is chosen by the format of the
    // partial sums, not by this flag (recon_build_prog in ozaki1/recon.hpp).
    bool signmag_slices;
};
constexpr Oz1Param oz1_param(OzFmt f) {
    switch (f) {
    // Digit weight 254, digits +-127. The partial products are Int32 and the
    // reconstruction decodes them into two exact FP32 values (recon_tf_i32in.cpp).
    case OzFmt::INT8: return {254, true};

    // Fp8_e4m3 represents the integers in [-16, 16] exactly: digits +-16, radix 32.
    case OzFmt::FP8: return {32, false};

    // Radix 64, digits +-32, |digit product| <= 2^10. Digits up to 32 fit the LoFi
    // multiplier (5 bits of SrcA), and the exactness condition 2w + ceil(log2 K) <= 24
    // holds with w = 5 up to K = 2^14.
    case OzFmt::BF16: return {64, false};

    // IEEE binary16. Radix, slice count and partial products are those of BF16; only the
    // transport format differs. binary16 represents the integers up to 2048 exactly, but
    // the digit bound is set by the multiplier (32 at LoFi) and the accumulator width.
    case OzFmt::FP16: return {64, false};

    // Stored as FP32 and passed to the matrix engine as TF32. Digit bound, radix and slice
    // count are those of BF16, at 4 bytes per element.
    case OzFmt::FP32: return {64, false};

    // Radix, slice count and accumulator of BF16, with the slices carried as block
    // floating point 8. The slices are integers, so they stay exact although 16 elements
    // share one exponent: the quantization step is a power of two and the integers
    // 0..127 are exact in every block. At 1.0625 instead of 2 bytes per element, the data
    // moved through DRAM and the NoC and the input CBs in L1 are nearly halved.
    case OzFmt::BFP8: return {64, false};

    // Block floating point 4 (0.5625 bytes per element). The 3-bit significand limits the
    // digits to +-7, hence the radix 14 and a larger number of slices.
    case OzFmt::BFP4: return {14, false};

    // Block floating point 2 (0.3125 bytes per element), the narrowest input format of
    // the hardware. Under a shared exponent only the integers in [-1, 1] are exact, so
    // the expansion is a signed binary one (radix 2, digits +-1) and one slice extracts
    // one bit.
    case OzFmt::BFP2: return {2, false};
    }
    return {};
}

// Digit bound for a radix: min(ceil(radix/2), exact_int_max of the format).
// The slicers round to nearest and produce signed digits,
//   t *= radix;  d = clamp(round(t), +-bound);  t -= d
// (slice_tf_num.cpp, slice_tf_sm8.cpp), i.e. a balanced expansion whose digits reach
// ceil(radix/2). The bound is not lowered to (radix-1)/2: the representable range
// bound/(radix-1) would then equal the normalization target 1/2 with no margin, and the
// attained accuracy drops for large K. Conversely, a format that represents the integers
// up to D exactly admits the radix 2D: the sign-magnitude byte of the INT8 path has 7
// magnitude bits, so its digits are +-127 and its radix is 254. The exactness conditions
// and the math fidelity depend on the digit values that the multiplier actually receives,
// so they are stated in terms of this bound, not of the radix.
constexpr uint32_t oz1_digit_max(uint32_t radix, uint32_t exact_int_max) {
    const uint32_t half = (radix + 1) / 2;
    return half < exact_int_max ? half : exact_int_max;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
