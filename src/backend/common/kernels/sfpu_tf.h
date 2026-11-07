// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// SFPU tile-operation wrappers and triple-float arithmetic (shared by the compute kernels that use the
// SFPU; the GEMM-stage kernel bmm_int32_spill.cpp does not include it)
// ---------------------------------------------------------------------------
// Holds the tile-operation wrappers shared by the compute kernels of Ozaki schemes I and II (Add, MulS,
// Rnd, ...) and the entry points of procedures such as TwoSum, tfMulR, tfAddS and ModRound. A kernel
// only has to include this file after the API headers.
// The bodies of the procedures and the arguments for their correctness (lemmas, guaranteed ranges) are
// in sfpu_tf_proc.h. The bodies are templates on the wrapper type that do not depend on the device API,
// so that the very same code can be instantiated with other hooks (the checks described in
// docs/sfpu_lemmas.md used a host-side model of the SFPU operations, which is not part of this
// distribution; a separate copy of the procedures could pass such a check while the kernel runs an
// older sequence).
//
// Constraints of this hardware
//   There is no FMA (fusedMultiplyAdd of IEEE 754: computes a*b+c with infinite precision and rounds
//   only once). The SFPU has the multiply-add instruction SFPMAD, and the tile-operation API exposes
//   it in the form of addcmul (a + v*b*c), but it keeps only the upper 27 bits (the 24 bits of FP32
//   + 3 bits) of the 48-bit product of two FP32 values before rounding, so it is not an operation that
//   rounds only once (measured: the sweep a=b=1+2^-j, c=-(1+2^-(j-1)) saturates at 2^-26, and scaling
//   the operands by 2^+-10 leaves the saturation point at the same position relative to the exponent
//   of the product). Therefore a*b - fl(a*b) cannot be obtained with one instruction, and the
//   error-free product (TwoProd) uses the Dekker splitting version instead of the FMA version.
//   When the multiplication and the addition are written as separate tile operations, the product is
//   correctly rounded to FP32, so with the way this header is written these 3 bits do not get mixed in.
//   Only the copy between DST registers is done by a dedicated API (copy_dest_values) with one
//   instruction pair.
//
//   In addition, TT-Metalium JIT-builds the kernels with -ffast-math by default, so when an error-free
//   transformation is written as a C float expression on the RISC-V, the expression of the residual is
//   simplified algebraically to 0 and accuracy is lost without any error being reported. All procedures
//   here are written with the SFPU tile intrinsics and are therefore not affected.
//
// The register conventions (the DST triple, the numbers of the work registers, the names in dst) are at
// the top of sfpu_tf_proc.h.

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/right_shift.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
// Bodies of the procedures and the DST register names dst (independent of the device API, so it can
// be included outside a kernel as well)
#include "sfpu_tf_proc.h"

// --- Conversion between scalars and bit patterns ----------------------------
// Runtime arguments and compile-time arguments both arrive as uint32, so the bit pattern is turned
// back into a float as is.
ALWI uint32_t fb(float s)   { union { float f; uint32_t u; } c; c.f = s; return c.u; }
ALWI float    fbf(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }

// --- Basic tile-wise operations ---------------------------------------------
ALWI void Add(uint32_t a, uint32_t b, uint32_t o) { add_binary_tile_init(); add_binary_tile(a, b, o); }
ALWI void Sub(uint32_t a, uint32_t b, uint32_t o) { sub_binary_tile_init(); sub_binary_tile(a, b, o); }
ALWI void Mul(uint32_t a, uint32_t b, uint32_t o) { mul_binary_tile_init(); mul_binary_tile(a, b, o); }
// d = s. A copy between DST registers goes to a dedicated API that needs only one SFPLOAD/SFPSTORE
// pair (ckernel_sfpu_copy_dest_values.h). On Blackhole copy_dest_values is a pure register move, one
// operation fewer than copying with the two operations "make 0, then add". Cpy appears twice in tfMulR
// and twice in tfAddS, so one Horner step with tfMulR (tfMulR and tfAddS; the path whose radix is not
// a power of two) takes 59 operations instead of 63 (when the radix is a power of two, the radix
// multiplication is tfMulP2 and one step takes 34 operations).
// s and d must be different registers.
ALWI void Cpy(uint32_t s, uint32_t d) {
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(s, d);
}
// Multiplication by a scalar and addition of a scalar. Both a version that takes the bit pattern (for
// compile-time constants) and a version that takes a float (for runtime arguments) are provided.
ALWI void MulS(uint32_t d, uint32_t sb) { binop_with_scalar_tile_init(); mul_unary_tile(d, sb); }
ALWI void MulS(uint32_t d, float s)     { MulS(d, fb(s)); }
ALWI void AddS(uint32_t d, uint32_t sb) { binop_with_scalar_tile_init(); add_unary_tile(d, sb); }
ALWI void AddS(uint32_t d, float s)     { AddS(d, fb(s)); }
// d = constant. Does not depend on the previous contents of d. Making 0 by self-subtraction
// (Sub(d,d,o)) does not give 0 in a situation where d is undefined (it can be NaN/Inf).
ALWI void Set(uint32_t d, float s) { fill_tile_init(); fill_tile(d, s); }
ALWI void Rnd(uint32_t d) { rounding_op_tile_init(); round_tile(d, 0); }
ALWI void Ltz(uint32_t d) { ltz_tile_init(); ltz_tile(d); }   // d = (d<0) ? 1 : 0

// --- Procedures (triple-float and residues modulo m) -------------------------
// The bodies and the arguments for their correctness are in sfpu_tf_proc.h (templates that take the
// wrapper type as an argument, so that the same code can be instantiated with other hooks).
// These are the entry points that call them with the device wrapper SfpuDev; the kernels call them by
// these names.
struct SfpuDev {
    // Bit pattern of the constant of the Dekker (Veltkamp) splitting, 2^12+1 = 4097.0f
    static constexpr uint32_t SPL = 0x45800800u;
    ALWI void Add(uint32_t a, uint32_t b, uint32_t o) { ::Add(a, b, o); }
    ALWI void Sub(uint32_t a, uint32_t b, uint32_t o) { ::Sub(a, b, o); }
    ALWI void Mul(uint32_t a, uint32_t b, uint32_t o) { ::Mul(a, b, o); }
    ALWI void Cpy(uint32_t s, uint32_t d) { ::Cpy(s, d); }
    ALWI void MulS(uint32_t d, uint32_t sb) { ::MulS(d, sb); }
    ALWI void MulS(uint32_t d, float s) { ::MulS(d, s); }
    ALWI void AddS(uint32_t d, uint32_t sb) { ::AddS(d, sb); }
    ALWI void AddS(uint32_t d, float s) { ::AddS(d, s); }
    ALWI void Rnd(uint32_t d) { ::Rnd(d); }
    ALWI void Ltz(uint32_t d) { ::Ltz(d); }
    // The two steps that another wrapper may replace with variants for comparison. On the device both
    // are exactly the procedures of sfpu_tf_proc.h.
    ALWI void FinalFts(uint32_t a, uint32_t b, uint32_t s, uint32_t t, uint32_t w1, uint32_t /*w2*/) {
        sfpu_proc::FastTwoSum(*this, a, b, s, t, w1);
    }
    ALWI void SmallTwoSum() { sfpu_proc::SmallTwoSum(*this); }
};
ALWI void TwoSum(uint32_t x, uint32_t y, uint32_t s, uint32_t e, uint32_t a, uint32_t b) {
    SfpuDev w; sfpu_proc::TwoSum(w, x, y, s, e, a, b);
}
// Multiplies the triple dst::acc_hi/acc_mid/acc_lo by the radix rdx (in place; rdx is a bit pattern;
// dst::term, tmp0..2 and keep are used as work registers and clobbered). A general radix uses tfMulR
// with the Dekker splitting, a power of two uses tfMulP2 (sfpu_tf_proc.h).
ALWI void tfMulR(uint32_t rdx) { SfpuDev w; sfpu_proc::tfMulR(w, rdx); }
ALWI void tfMulP2(uint32_t rdx) { SfpuDev w; sfpu_proc::tfMulP2(w, rdx); }
// Dispatch of the radix multiplication (at compile time). When the significand field of the radix RDX
// (the bit pattern of a float) is 0, i.e. the radix is a power of two, tfMulP2 is used, otherwise
// tfMulR. In the value range that occurs in actual use the result is the same either way (the condition
// for agreement at tfMulP2 in sfpu_tf_proc.h), so for a power-of-two radix the identity work of the
// Dekker splitting disappears and 44 operations become 19.
// Among the default radices of Ozaki scheme I, BF16, FP16, TF32 and BFP8 (64), FP8 (32) and BFP2 (2)
// are powers of two, while INT8 (254) and BFP4 (14) stay with tfMulR. The radix arrives as a
// compile-time argument, so the correct side is selected at each JIT build of the kernel even when the
// radix is changed. Both use dst::term, tmp0..2 and keep as work registers.
template <uint32_t RDX>
ALWI void tfMulRadix() {
    if constexpr (sfpu_proc::radix_is_pow2(RDX)) tfMulP2(RDX);
    else                                         tfMulR(RDX);
}
// Adds v (dst::term) to the triple dst::acc_hi/acc_mid/acc_lo (a pair when NL=2) (in place; work
// registers dst::tmp0..2; keep is not touched)
template <uint32_t NL>
ALWI void tfAddS() { SfpuDev w; sfpu_proc::tfAddS<NL>(w); }
// Addition of a small term v (dst::term) (skips the first step of tfAddS; the precondition |v| <= eps
// is described in sfpu_tf_proc.h)
template <uint32_t NL>
ALWI void tfAddSmall() { SfpuDev w; sfpu_proc::tfAddSmall<NL>(w); }
// d -= round(d/m)*m (o and d may be the same; s is a work register different from d; the contract, i.e.
// the preconditions on d, m and rinv and the guaranteed range, is at ModRound in sfpu_tf_proc.h)
ALWI void ModRound(uint32_t d, uint32_t o, uint32_t s, float m, float rinv) {
    SfpuDev w; sfpu_proc::ModRound(w, d, o, s, m, rinv);
}
// Alias for the in-place update (so that the intent of the caller is easier to read)
ALWI void ModRound(uint32_t d, uint32_t s, float m, float rinv) { ModRound(d, d, s, m, rinv); }
// Sign-magnitude 8-bit encoding (in place; w1 and w2 are work registers; sfpu_tf_proc.h)
ALWI void SignMag8(uint32_t d, uint32_t w1, uint32_t w2) { SfpuDev w; sfpu_proc::SignMag8(w, d, w1, w2); }

// --- FP16 path: integer value to the bit pattern of binary16 ------------------
// Why it is needed
//   The slicer of the FP16 path uses DST in FP32 (to hold the residual of the triple-float). However,
//   packing from an FP32 DST into a Float16 CB corrupts the values (they become different values in
//   which only the ordering is preserved). Writing from a 16-bit DST would be correct, but the slicer
//   cannot be built that way. Therefore the bit pattern of binary16 is created in DST as "the integer
//   that has that value" and written to the CB as UInt16. The GEMM side reads the same bytes as
//   Float16.
//
// What it does
//   For an integer d, it creates in reg d the integer value equal to the binary16 bit pattern
//   sign<<15 | (15+e)<<10 | 10-bit significand. Example: d = -3 = -1.1b x 2^1 gives
//   0x8000 | 16<<10 | 0x200 = 0xC200.
//
// How it is done
//   The FP32 bit pattern of a = |d| is (127+e)<<23 | 23-bit significand. Shifting it right by 13 bits
//   gives (127+e)<<10 | upper 10 bits of the significand, and only replacing the exponent bias 127 by
//   15 (subtracting (127-15)<<10 = 114688) turns it into the magnitude part of binary16.
//   right_shift_tile treats the 32-bit word of DST not as a value but as an integer bit pattern, so
//   this operation of "shifting the bit pattern of a float" can be written directly. Since a >= 0 the
//   sign bit is 0, and whether the shift is arithmetic or logical does not matter for the result.
//   The shifted result is an integer bit pattern, so it is turned back into a value with the
//   Int32 -> Float32 typecast, and then the bias is subtracted and the sign bit 32768 is added in float
//   arithmetic. The intermediate values are below 2^18 and are exactly representable in FP32. Only for
//   a = 0 the FP32 bit pattern is 0 and the formula for the exponent does not hold, so at the end the
//   result is multiplied by [a>0] to make it 0 (d = -0 also becomes 0x0000).
//   After this the caller applies the Float32 -> UInt16 typecast and packs into a UInt16 CB
//   (ozaki1/kernels/slice_tf_num.cpp, ozaki2/kernels/modslice_digit.cpp).
//
// Contract
//   d is an integer value with |d| <= 2048. Every integer in this range fits in an 11-bit significand
//   (10 bits + the implicit 1), and the shift drops no low bit. Above 2048, an integer that does not
//   fit in 11 bits, such as 2049, becomes a different value (those that fit, such as 2050 or 4096, are
//   correct). The digits of a slice are at most +-32, so this is well inside. d and w1..w3 are mutually
//   distinct DST registers, and w1..w3 are work registers (their values after the call are not
//   guaranteed). About 18 operations.
ALWI void HalfBits(uint32_t d, uint32_t w1, uint32_t w2, uint32_t w3) {
    Cpy(d, w1); Ltz(w1);                                               // w1 = [d<0] (sign)
    Mul(d, w1, w2); MulS(w2, -2.f); Add(d, w2, w2);                    // w2 = a = d - 2d[d<0] = |d|
    Cpy(w2, w3); AddS(w3, -0.5f); Ltz(w3); MulS(w3, -1.f); AddS(w3, 1.f);   // w3 = 1 - [a<0.5] = [a>0]
    Cpy(w2, d);
    // d = bits(a) >> 13 = (127+e)<<10 | significand (an integer bit pattern)
    right_shift_tile_init(); right_shift_tile(d, 13);
    typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
    // the integer value of that bit pattern becomes a float value
    typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(d);
    // exponent bias 127 -> 15: (15+e)<<10 | significand
    AddS(d, -114688.f);
    MulS(w1, 32768.f); Add(d, w1, d);                                  // add the sign bit (bit 15)
    Mul(d, w3, d);                                                     // 0x0000 when a = 0
}
