// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Ozaki scheme II: the stage that makes the ND radix-2^20 digits of A' from the operand (device side)
// ---------------------------------------------------------------------------
// Role: the FP64 operand x before normalization (it arrives as an FP32 triple-float, i.e. a three-term
//       expansion) is converted to the integer
//           A' ~ X,  X = x / 2^e * 2^b          (|A'| <= 2^b)
//       and its ND radix-2^20 digits d_{ND-1}, ..., d0 are made and passed to the modular slicer that
//       follows. ND is a compile-time argument: 3 for b <= 60, 4 for b up to 80, and 5 above that.
//       The difference between A' and the ideal value X is at most 1, as described under "Conditions
//       that the digits satisfy" below.
//       The stage is shared by the INT8 path (modslice_int8.cpp) and the 7 digit-splitting paths
//       (modslice_digit.cpp).
//
// Procedure
//   The normalization enters as an addition to the exponent, not as a multiplication. The per-row (A) /
//   per-column (B) value
//       d = (b - (ND-1)G) - e     (G = bit width of a digit, 20. 2^e = the maximum of the row / column
//                                   rounded up to a power of 2)
//   is expanded into Int32 tiles by expdelta.cpp, and slice_f64_reader.cpp adds it as an integer to the
//   exponent field of the FP64 value and then forms the triple-float (tf_load_f64 in slice_tf_core.h).
//   When the value reaches the compute side it is therefore already
//       T = x / 2^e * 2^(b-(ND-1)G) = X / 2^((ND-1)G)
//   and what remains has the same form as the slicer of Ozaki scheme I (ozaki1/kernels/slice_tf_num.cpp.
//   The body is slice_tf_body.h): "round and subtract to get the residual" is done ND times, with
//   "multiply by the radix 2^G" inserted ND-1 times in between (the first multiplication by the radix is
//   folded into d above):
//       d2 = round(T) ; T -= d2 ; T *= 2^G
//       d1 = round(T) ; T -= d1 ; T *= 2^G
//       d0 = round(T)
//   Receiving the input, the normalization and the carrying of the residual are common to the three
//   slicers and are therefore in slice_tf_core.h.
//
// Conditions that the digits satisfy
//   The radix is a power of 2, so the multiplication by the radix (tfMulP2) is exact, and no error
//   enters when the residual is carried into the next digit.
//   Only the high limb of the triple-float is rounded, so at a boundary where the middle and lower limbs
//   decide the rounding, A' can be a neighbor of the integer nearest to X. What can be guaranteed is
//   therefore |A' - X| <= 1, not rounding to nearest. What the CRT window requires is |A'| <= 2^b, and
//   this follows from |x/2^e| <= 1. Keeping the window and not affecting the accuracy are different
//   things: the difference between A' and X enters the output as it is, as the integer-conversion
//   error (the bound is 1, not the 1/2 of rounding to nearest. For b=54 and a row whose maximum is 1,
//   x = 2^-55 + 2^-80 gives X = 1/2 + 2^-26 and A' = 0).
//   The digits are in balanced (signed) representation: the top digit satisfies |T| <= 2^(b-(ND-1)G)
//   (2^20 even for b = 80 or 100, which fill the digits completely), and the others are at most
//   2^19 + 2^16 + 1 (only the high limb is rounded, so the residual is 1/2 plus the contribution of
//   the middle and lower limbs (at most 2^-4). This bound is D0 of oz2_resid_bound in split.hpp).
//   All of them are exactly representable in FP32 and are within the range 2^22 that ModRound of the
//   following stage requires. The composition A' = sum_j d_j*2^(jG) is linear with respect to the
//   modulus, so the signs of the digits pass through as they are.
//
// Buffers
//   Input    the 4 buffers of slice_tf_core.h (3 with the raw limb data, and the scale s)
//   Scratch  the 2 residual buffers of slice_tf_core.h
//   Output   c_26 = the high digits in the order they are made (d_{ND-1} down to d1)  c_27 = digit d0

#include <cstdint>
#include "../../common/kernels/slice_tf_core.h"
#include "args.h"   // argument layouts and CB numbers (and the limits of digit_limits.h)

// output: the high digits d_{ND-1}..d1 (in the order they are made)
constexpr auto cb_dhi = tt::CBIndex(oz2_args::CB_DIGIT_HI);
constexpr auto cb_dlo = tt::CBIndex(oz2_args::CB_DIGIT_LO);   // output: digit d0

// Compile-time argument SC_RDX = the digit radix 2^G as a float bit pattern
// (SC_S is used by the including kernel for the number of moduli S)
constexpr uint32_t OZ2_RDX_B  = get_compile_time_arg_val(oz2_args::SC_RDX);
// Number of digits (compile-time argument SC_NDIGIT). N digits carry up to N*G bits. The host decides it
// from b and passes it: 3 digits for the default b<=54, 4 digits only when b is raised above 60 for an
// input with a wide dynamic range, and 5 digits only when b is raised above 80
// (choose_moduli_and_width in plan.hpp). The depth of the output CB must also match the number of
// digits (oz2_make_digits below pushes one tile at a time, but the ND-1 tiles of cb_dhi are waited
// for as one set, read by index, and popped).
constexpr uint32_t OZ2_NDIGIT = get_compile_time_arg_val(oz2_args::SC_NDIGIT);
static_assert(OZ2_NDIGIT >= 3 && OZ2_NDIGIT <= (uint32_t)OZ2_MAXDIGIT, "A' has 3 to 5 digits (oz2_residue)");

// Consumes one input tile and pushes the ND digit tiles to cb_dhi (ND-1 tiles) and cb_dlo (1 tile).
// On return, cb_wait_front has been done for all of them, so oz2_digits_copy can be called directly.
ALWI void oz2_make_digits() {
    tf_wait_input();
    for (uint32_t i = 0; i < OZ2_NDIGIT; i++) {
        tile_regs_acquire();
        init_sfpu(cb_tf_hi, cb_dhi);
        if (i == 0) tf_load_f64();                    // T = triple-float built from FP64 (normalized)
        else      { tf_pop_residual(); tfMulP2(OZ2_RDX_B); }   // the radix is 2^20 (a power of 2)
        Cpy(0, 7); Rnd(7);          // digit d = round(T)
        // residual T -= d, acting on the top limb only. Exact: d is an integer within 1/2 of T and
        // |T| <= 2^20 < 2^24, so T - d is a multiple of ulp(T) not larger than |T| (the same argument
        // as "Invariant of the residual" in ozaki1/kernels/slice_tf_body.h); the result has |T| <= 1/2
        Sub(0, 7, 0);
        tile_regs_commit();
        tile_regs_wait();
        // Destination of the digit. The high digits and d0 use separate CBs because the following
        // stage reads the high digits and d0 separately by index; this is unrelated to the reason why
        // the residual is split into 2+1.
        const auto cb_d = (i + 1 < OZ2_NDIGIT) ? cb_dhi : cb_dlo;
        cb_reserve_back(cb_d, 1);
        pack_reconfig_data_format(cb_d);
        pack_tile(7, cb_d);
        cb_push_back(cb_d, 1);
        if (i + 1 < OZ2_NDIGIT) tf_push_residual();   // save the residual for the next digit
        tile_regs_release();
    }
    tf_pop_input();
    cb_wait_front(cb_dhi, OZ2_NDIGIT - 1);
    cb_wait_front(cb_dlo, 1);
}

// Loads the ND digits into DST (d0 -> r0, d1 -> r1, d2 -> r2, d3 -> r3, d4 -> r4). The contents of DST
// are not kept across acquire/release, so the digits must be reloaded for each modulus.
// cb_dhi holds the ND-1 tiles in the order they are made (from the top digit), so the indices are read
// in reverse order.
ALWI void oz2_digits_copy(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4) {
    copy_tile_to_dst_init_short(cb_dlo); copy_tile(cb_dlo, 0, r0);   // d0 (the digit made last)
    copy_tile_to_dst_init_short(cb_dhi);
    copy_tile(cb_dhi, OZ2_NDIGIT - 2, r1);                           // d1
    copy_tile(cb_dhi, OZ2_NDIGIT - 3, r2);                           // d2
    if constexpr (OZ2_NDIGIT >= 4) copy_tile(cb_dhi, OZ2_NDIGIT - 4, r3);   // d3
    if constexpr (OZ2_NDIGIT >= 5) copy_tile(cb_dhi, OZ2_NDIGIT - 5, r4);   // d4
}
// Pops the digits of one input tile once they have been used
ALWI void oz2_digits_pop() { cb_pop_front(cb_dhi, OZ2_NDIGIT - 1); cb_pop_front(cb_dlo, 1); }

// Makes a representative r == A' (mod m) of the residue class of A' modulo m from the digits and puts
// it in reg2 (scratch: reg0..7). The bound is |r| <= floor(m/2)+1 (see below): the symmetric residue
// for the moduli of the INT8 path, possibly 1 beyond it for large moduli of the digit-splitting paths.
//   A' mod m = ( ... + (d2 mod m)*c1 + (d1 mod m)*c0 + d0 ) mod m
// Only the digits that carry a weight are reduced first. d0 has weight 1, so it may be added without
// being reduced.
// Magnitudes: |d0| <= D0 = 2^19 + 2^16 + 1 (2^20 times the rounded residual. Only the high limb is
// rounded, so it exceeds 2^19 by the contribution of the middle and lower limbs. oz2_resid_bound in
// split.hpp), a reduced digit is a symmetric residue, |t_j| <= floor(m/2) (its input is a digit below
// 2^22, inside the general guarantee of ModRound), and c[j] is the value in [0, m) returned by pow2mod
// of the host (not symmetric), so the input of the last ModRound satisfies
//     |S| <= D0 + floor(m/2)*(c[0]+c[1])
// For the moduli of the INT8 path (m <= 255) this stays below 2^22, so r is the symmetric residue,
// |r| <= floor(m/2). On the BF16, BFP8, FP16 and TF32 paths, whose moduli extend up to 4096, it exceeds
// 2^22 and reaches 1.55e7 (2^23.9) for m=3913. That is outside the general guarantee |S| < 2^22 of
// ModRound in sfpu_tf_proc.h, but the residue class is kept correct by the following two points.
//   - The partial sums, t_j*c[j], and the product of the quotient and the modulus round(S/m)*m are all
//     integers of at most 2^24, so they are exact in FP32 (the residue class is determined here).
//   - The deviation of fl(S*rinv) is at most |S|*2^-23/m, so even if round selects the neighboring
//     integer, |r| <= m/2 + |S|*2^-23 < m/2 + 2 still holds.
// With 4 digits there are moduli for which adding up to the d3 term exceeds 2^24 (odd moduli such as
// m=3605, which the reselection of the trailing moduli can pick), so the sum is reduced once before the
// d3 term is added. 3 digits (b <= 60, the default) do not take this branch.
// With 5 digits (b > 80), the d4 term (coefficient c3) is added to the reduced value, the sum is reduced
// once more, and the d3 term is added last. The input of the last ModRound then has the same form
// "reduced value + t3*c2" as with 4 digits, so the bound on |r| (oz2_resid_bound of the host) is the
// same as with 4 digits. If the d4 term were added later, c3 would enter the bound, which would grow by
// 1 for some moduli.
// What is guaranteed is therefore "the residue class is correct and |r| <= floor(m/2)+1", not the
// symmetric residue itself. What the following stages rely on is |r| <= 127 on the INT8 path (7 bits
// of sign-magnitude; m <= 255 gives the symmetric residue, see above) and, on the digit-splitting
// paths, that the digits fit within the bound (digit_bound in the path table OZ2_ROUTES of split.hpp).
// The excess of 1 occurs only for the large moduli where |S| exceeds 2^22 (paths such as BF16), and
// even with that excess the digits fit within 32.
// The residue class and the digit bounds have been verified for every modulus up to the bound of each
// path and for 3 to 5 digits.
// The signed representative is returned without wrapping it into [0, m) because the bound on the
// digit magnitudes is halved when the digit-splitting path starts its digit splitting from here; it
// also suits the sign-magnitude encoding of the INT8 path.
//   rinv = 1/m, c[j] = 2^((j+1)*G) mod m (computed by the host. c[0]=2^20, c[1]=2^40, c[2]=2^60, c[3]=2^80)
ALWI void oz2_residue(float m, float rinv, const float* c) {
    // reg2 = d0 (weight 1), reg7/reg1/reg0/reg3 = d1/d2/d3/d4 (weighted). reg4 is scratch for ModRound
    oz2_digits_copy(2, 7, 1, 0, 3);
    ModRound(7, 4, m, rinv); MulS(7, c[0]); Add(2, 7, 2);
    ModRound(1, 4, m, rinv); MulS(1, c[1]); Add(2, 1, 2);
    if constexpr (OZ2_NDIGIT >= 4) {                        // without reducing first, 2^24 is exceeded
        ModRound(2, 4, m, rinv);
        if constexpr (OZ2_NDIGIT >= 5) {                    // add the d4 term, reduce again (see above)
            ModRound(3, 4, m, rinv); MulS(3, c[3]); Add(2, 3, 2);
            ModRound(2, 4, m, rinv);
        }
        ModRound(0, 4, m, rinv); MulS(0, c[2]); Add(2, 0, 2);
    }
    ModRound(2, 4, m, rinv);                                // representative with |r| <= floor(m/2)+1
}
