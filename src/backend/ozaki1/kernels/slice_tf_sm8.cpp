// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: radix-254 slicer on the device (INT8 path; outputs the digits as sign-magnitude UInt8)
// ---------------------------------------------------------------------------
// Role: decomposes each element of an operand into nslice (L+1) small-integer slices in radix 254 (digits
//       +-127) and outputs them as the "sign-magnitude" bytes that the INT8 matrix multiplication expects.
//
// The radix and the upper bound of a digit are different quantities
//   The magnitude part of a sign-magnitude byte has only 7 bits, so a digit fits in d = +-127, and the
//   weight (radix) is taken from it as beta = 2d = 254. The upper bound of the representable range is
//   R = d/(beta-1) = 127/253, which exceeds the target T = 1/2 of the normalization by 0.4%. This margin is
//   below 1/128, so the host rounds the conversion of the maximum to BF16 upward and guarantees
//   |x/tau| <= T (OZ1_BF_CEIL in ozaki1/format.hpp). With that, the residual stays within |t| <= R, and the
//   truncation error after p steps falls to at most R*beta^-p.
//   The earlier implementation ozIMMU takes 2^7 as the weight and keeps the digits within 7 bits
//   (weight = digit + 1).
//
// Why the residual is held in triple-float
//   Every time a slice is extracted, the residual is updated as t = t - d. Held in a single FP32, this
//   residual saturates at about 24 bits, and in two FP32 values (double-single) it stops at about 48 bits.
//   So as not to lose any of the 53 significand bits of the FP64 input, the input value and the residual
//   are held in a triple of FP32 values (triple-float, working precision of about 72 bits).
//
// The body (radix multiplication, rounding, clamp, update and carrying of the residual) is in
// slice_tf_body.h, which is shared with the slicer of the 7 formats. This file only determines the encoding
// of the digit d (SLICE_SIGNMAG8 = true). The output c_16 is NSLICE tiles of INT8 sign-magnitude slices.
//
// Sign-magnitude encoding
//   A slice value d is output as the UInt8 byte g(d) = (d<0) ? (128-d) : d, i.e. the sign in bit 7 and
//   |d| in the lower 7 bits (SignMag8 in sfpu_tf_proc.h):
//     d=+5   -> g=5    -> 0x05 (sign 0, magnitude 5)
//     d=-5   -> g=133  -> 0x85 (sign 1, magnitude 5)
//     d=-127 -> g=255  -> 0xFF (sign 1, magnitude 127)
// The host also passes OUT_HALF, but it has no effect on the sign-magnitude output (slice_tf_body.h).

constexpr bool SLICE_SIGNMAG8 = true;    // outputs the digits as sign-magnitude UInt8 (slice_tf_body.h)
#include "slice_tf_body.h"                // kernel_main
