// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Ozaki scheme I: body (kernel_main) shared by the two slicers (slice_tf_num.cpp, slice_tf_sm8.cpp)
// ---------------------------------------------------------------------------
// The two differ only in the part that encodes the extracted digit d into the output form and in the index
// of the DST register that is packed. The difference is selected with the constexpr bool SLICE_SIGNMAG8 of
// the including file (true in slice_tf_sm8.cpp), and receiving the input, the radix multiplication, the
// rounding, the clamp, the update of the residual, the output and the carrying of the residual are kept in
// this one place (if only one of the two were changed, the procedure would silently disagree depending on
// the format). The including file defines SLICE_SIGNMAG8 and then includes this header (this header
// defines kernel_main).
// When the encoding was passed as a function type (a template argument), the register allocation changed
// even for the same computation and the machine code differed from the one before the split into two
// kernels, so the body remains a single function and selects with if constexpr (the same machine code as
// before the split).
// The handover of the input and the carrying of the residual are in slice_tf_core.h, which is also shared
// with the digit generation of Ozaki scheme II.
//
// Processing of each element
//   t = (hi,mid,lo)           (the reader has folded the power of two of the normalization into the FP64
//                              exponent)
//   for i = 0..NSLICE-1: u = radix*t ; d = clamp(round(u0)) ; encode d and output it ; t = u - d
//   The radix multiplication is tfMulP2 (19 operations) when the radix is a power of two (64 of BF16, FP16,
//   TF32 and BFP8, 32 of FP8, 2 of BFP2) and tfMulR (44 operations) otherwise (14 of BFP4, 254 of
//   INT8); in the value range of actual runs the result is the same (see the condition for agreement at
//   tfMulP2 in sfpu_tf_proc.h).
//   round looks only at the top word (t0) of the triple-float.
//
// Invariant of the residual: both the rounding and the subtraction act only on the top limb t0 of the
// triple, and the two lower limbs are left untouched. d is an integer made from t0, and t0-d is exact while
// |t0| < 2^24, so t as the value of the triple decreases by exactly d.
// The invariant is therefore |t0| <= R on the top limb (R = d_max/(beta-1). By default beta = 2*d_max, so
// R = d_max/(2*d_max-1), i.e. slightly above 1/2): if |t0| <= R, then
// |beta*t0 - clamp(round(beta*t0))| <= max(1/2, beta*R - d_max) = R, and repeating the step never leaves
// the range (the clamp acts only when beta*|t0| exceeds d_max+1/2, or in the edge case where it is exactly
// equal and is rounded outward, and even then the residual stays within R).
// After p steps the error falls to at most R*beta^{-p}. This is an estimate over the reals for t0 alone.
// At the time of the rounding (right after the radix multiplication has normalized the triple with TwoSum),
// the two lower limbs are within about an ulp of t0, so they do not break this estimate beyond the extent
// of rounding errors. Right after t0 -= d, t0 has been cancelled and the two lower limbs may look
// relatively large, but the radix multiplication of the next step normalizes again (both tfMulR and tfMulP2
// end with a normalization by a TwoSum chain).
// The host is responsible for |t0| <= R at entry (R is the upper bound of the representable range,
// d/(beta-1). T is the target of the normalization, a power of two).
// For the normalized value x, the formats that truncate (BF16 and others) have |x| < T(1+2^-7) <= R, and the
// configurations in which R hardly exceeds T (INT8, BFP2) round the scale up to get |x| <= T (OZ1_BF_CEIL
// in ozaki1/format.hpp. BFP2 raises the scale one step even when the maximum is exactly a power of two, to
// get |x| < T; OZ1_TAU_ABOVE in the same file). t0 is the top limb obtained by rounding x to FP32, so it can
// be equal to the upper bound of x (t0 = 1 = T for x = 1-2^-53 in BFP2), but |t0| <= R holds in every case.
//
// Compile-time arguments (order: SlicerCT in args.h): number of slices NSLICE (L+1), bit pattern of radix as
//   a float, whether the reader is the soft-float version, bit pattern of the upper bound of a digit as a
//   float, its negative, OUT_HALF (effective only when the digit value is output as is)
// Runtime arguments: ntiles (SlicerRT in args.h)
// DST (8 tiles with fp32_dest_acc. slice_build on the host sets dst_full_sync_en, so the 8 tiles can be used
//   as one section; with half sync they would be divided into 4 and 4). The names are in namespace dst of
//   sfpu_tf_proc.h: triple of t = acc_hi, acc_mid, acc_lo (updated by the radix multiplication tfMulRadix),
//   extracted digit d = keep, work registers of the encoding = term, tmp0, tmp1. tfMulRadix uses term,
//   tmp0..2 and keep as work registers and clobbers them, so d is made after the radix multiplication

#include <cstdint>
#include "../../common/kernels/slice_tf_core.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "args.h"

constexpr auto cb_out = tt::CBIndex(oz1_args::CB_SLICE_OUT);

constexpr uint32_t NSLICE  = get_compile_time_arg_val(oz1_args::SC_NSLICE);  // number of slices (L+1)
constexpr uint32_t RADIX_B = get_compile_time_arg_val(oz1_args::SC_RADIX);   // bit pattern of radix as a float
// 1: the reader is the soft-float version and passes the triple as is (for measurement, as a comparison)
constexpr uint32_t SF_IN   = get_compile_time_arg_val(oz1_args::SC_SF_IN);
constexpr uint32_t DMAX_B  = get_compile_time_arg_val(oz1_args::SC_DMAX);    // upper bound of a digit (+)
constexpr uint32_t NDMAX_B = get_compile_time_arg_val(oz1_args::SC_NDMAX);   // upper bound of a digit (-)
// 1: the slices are output as IEEE binary16 bit patterns (UInt16) (FP16 path; the sign-magnitude slicer
// does not read it)
constexpr uint32_t OUT_HALF = get_compile_time_arg_val(oz1_args::SC_OUT_HALF);
constexpr uint32_t DF_F32   = (uint32_t)DataFormat::Float32;
constexpr uint32_t DF_U16   = (uint32_t)DataFormat::UInt16;
constexpr uint32_t DF_U8    = (uint32_t)DataFormat::UInt8;
// DST register that holds the output digit: term for sign-magnitude, keep (d itself) when the digit value
// is output as is
constexpr uint32_t OUT_REG  = SLICE_SIGNMAG8 ? dst::term : dst::keep;

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(oz1_args::SR_NTILES);
    using namespace dst;

    for (uint32_t ti = 0; ti < ntiles; ti++) {
        tf_wait_input();
        for (uint32_t i = 0; i < NSLICE; i++) {
            tile_regs_acquire();
            init_sfpu(cb_tf_hi, cb_out);
            if (i == 0) {                   // t = triple built from FP64 (normalization included)
                if constexpr (SF_IN) tf_load_plain(); else tf_load_f64();
            }
            else        tf_pop_residual();
            tfMulRadix<RADIX_B>();                   // t *= radix (tfMulP2 for a power of two; sfpu_tf.h)
            Cpy(acc_hi, keep);                       // slice value d = top limb of t
            rounding_op_tile_init(); round_tile(keep, 0);
            clamp_tile_init();       clamp_tile(keep, NDMAX_B, DMAX_B);   // d = clamp(round(t0), +-DMAX)
            // t0 -= d (d in keep is kept for the output; see "Invariant of the residual" above)
            Sub(acc_hi, keep, acc_hi);
            if constexpr (SLICE_SIGNMAG8) {
                // INT8 path: encode d into sign-magnitude and place it in term. Only term, tmp0 and tmp1
                // are used, so the residual that has just been updated (acc_hi, acc_mid, acc_lo) is not
                // clobbered.
                Cpy(keep, term); SignMag8(term, tmp0, tmp1);   // term = g(d) (sfpu_tf.h)
                typecast_tile_init<DF_F32, DF_U8>();
                typecast_tile<DF_F32, DF_U8>(term);      // term = (uint8) g(d)
            } else if constexpr (OUT_HALF) {
                // FP16 path: turn d into the integer value equal to its binary16 bit pattern (HalfBits;
                // work registers term, tmp0, tmp1) and typecast it to UInt16. The packer writes it as is
                // into the UInt16 CB, and the GEMM reads those bytes as Float16.
                HalfBits(keep, term, tmp0, tmp1);
                typecast_tile_init<DF_F32, DF_U16>();
                typecast_tile<DF_F32, DF_U16>(keep);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(OUT_REG, cb_out);              // output slice i (converted to the format of cb_out)
            cb_push_back(cb_out, 1);
            if (i + 1 < NSLICE) tf_push_residual();   // save the residual for the next slice
            tile_regs_release();
        }
        tf_pop_input();
    }
}
