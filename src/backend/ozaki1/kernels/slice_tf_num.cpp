// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: radix slicer on the device (common to the 7 formats other than INT8; outputs the digit
// value as is)
// ---------------------------------------------------------------------------
// Role: decomposes each element of an operand into nslice (L+1) small-integer slices in radix `radix` and
//       outputs the signed integers as values of the input format of the GEMM (BF16, FP16, TF32 (stored
//       as Float32 tiles), BFP8, FP8, BFP4, BFP2).
//       Unlike the INT8 path (slice_tf_sm8.cpp), these formats represent the integers of the digit range
//       exactly, so no encoding into sign-magnitude is needed. The packer converts directly from the FP32
//       in DST to the output format. The exception is FP16: packing from the FP32 DST to Float16 corrupts
//       the values, so the binary16 bit pattern is built as an integer and written as UInt16 (OUT_HALF;
//       HalfBits in sfpu_tf.h).
//
// Why the residual is held in triple-float
//   A residual held in a single FP32 saturates at about 24 bits, and in two FP32 values at about 48 bits.
//   So as not to lose any of the 53 significand bits of the FP64 input, the value and the residual are held
//   in a triple of FP32 values (triple-float, working precision of about 72 bits).
//
// The body (radix multiplication, rounding, clamp, update and carrying of the residual) is in
// slice_tf_body.h, which is shared with the INT8 slicer. This file only determines the output form of the
// digit d (SLICE_SIGNMAG8 = false). The output c_16 is NSLICE tiles of signed integer slices.
//
// The radix (weight of a digit) and the upper bound of a digit are different quantities, and by default
// weight = 2*digit (BF16 and BFP8 have weight 64 / digit +-32, FP8 32 / +-16, BFP4 14 / +-7, BFP2
// 2 / +-1). The clamp uses the upper bound of the digit.
// The weight is kept at 2*digit in order to leave a margin by which the range R = digit/(radix-1) exceeds
// the target T of the normalization (for the formats with T = 1/2: 1.6% for digit 32, 3.2% for digit 16,
// 7.7% for digit 7. BFP2 has R = T = 1; the host rounds the scale up and, in addition, takes a power of
// two strictly greater than the maximum, because for an element that becomes exactly T = 1 the expansion
// would stay at digit 1 and residual 1 and never end; OZ1_BF_CEIL and OZ1_TAU_ABOVE in ozaki1/format.hpp).
// With weight 2*digit+1 the margin would be 0 (R = T); the margin absorbs the effect of rounding only
// the top limb of the triple (see "Invariant of the residual" in slice_tf_body.h).
// If the compile-time argument OUT_HALF (SC_OUT_HALF in args.h) is 1, the slices are output as binary16 bit
// patterns (FP16 path only).
// DST: HalfBits uses term, tmp0 and tmp1 as work registers (the triple of t, acc_hi, acc_mid, acc_lo, is
// saved as the residual after the packing, so it is not touched. The names are in namespace dst of
// sfpu_tf_proc.h)

constexpr bool SLICE_SIGNMAG8 = false;   // digit value as is (binary16 bits for FP16 only; slice_tf_body.h)
#include "slice_tf_body.h"                // kernel_main
