// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Ozaki scheme II (FP8): digit fix-up of method (E) SQS with sparse e4m3 digits (device side, SFPI)
// ---------------------------------------------------------------------------
// Role: the balanced digits of a square modulus m = q^2 (q odd, 33 < q <= 49), made by rounding to
//       nearest as in method (A),
//           h0 = round(r/q),  t = r - q*h0          (|t| <= (q-1)/2 <= 24, |h0| <= (q+1)/2 <= 25)
//       are moved into the set of integers that e4m3 represents exactly,
//       S = {|x| <= 16} U {even x : 16 < |x| <= 32}.
//         if t is odd and |t| >= 17:    l = t - sgn(t)*q,  h1 = h0 + sgn(t)   (r = l + q*h1 still holds)
//         if h1 is odd and |h1| >= 17:  h = h1 - sgn(h1)*q                    (q*h == q*h1 because q^2 == 0)
//       q is odd, so an odd t (h1) minus q is even, and its magnitude is q - |t| <= q - 17 <= 32.
//       That l, h are in S and r == l + q*h (mod q^2) has been proved and also checked exhaustively
//       over all moduli and all residues, both for the host copy oz2_sqs_split (split.hpp) and for the
//       same FP32 procedure as here.
//       This applies to e4m3 the idea of "covering the residues with a non-contiguous set of exactly
//       representable integers" of arXiv 2608.06812 (Hayashi, Mukunoki, Hoshino, Katagiri).
//
// Why this is written in SFPI
//   Written with the tile operations of this library (Cpy, MulS, Rnd, etc. of sfpu_tf.h. One operation
//   is one round trip of reading and writing a tile), the same fix-up takes 26 operations for the
//   parity and sign tests (the slicing stage per modulus would grow from 25 to 51 operations). With
//   SFPI, conditional additions and subtractions within a single read and write are enough. The SFPI
//   code is written in the same way as
//   programming_examples/custom_sfpi_* of tt-metal.
//
// Numerical handling (everything is exact)
//   l and h are FP32 values holding integers with |.| <= 26. |x| + 2^23 is an integer below 2^24, so it
//   is exactly representable in FP32, and the least significant bit of its bit pattern is the parity
//   of |x| (the units place of the significand). The comparison |x| > 16.5 and the additions and
//   subtractions of q and 1 are also exact, since they operate on integer values.
//
// Registers: reg dl (l) and reg dh (h) of DST are rewritten in place. No other register is read.
// Preconditions: fp32_dest_acc_en (DST is FP32). The init of some SFPU binary operation must have been
//       done before the call (Oz2SqsFix calls add_binary_tile_init).
// The compute API used here (ALWI, MATH, add_binary_tile_init, _llk_math_eltwise_binary_sfpu_params_,
// VectorMode and SFPI) comes from the header of the SFPU binary operations (the including file
// modslice_digit.cpp has already included the same header through digits.h).
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_MATH
// One face (32 SFPI lanes x 8 rows). _llk_math_eltwise_binary_sfpu_params_ calls this 4 times, once per
// face. The third argument (the output index) is not used (l and h are fixed in place).
inline void oz2_sqs_fix_face(const uint32_t dl, const uint32_t dh, const uint32_t /*unused*/, const float q) {
    constexpr uint32_t T = 32;   // one DST tile is 32 SFPI rows (as in ckernel_sfpu_lerp.h)
    const sfpi::vFloat vq = q;
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat l = sfpi::dst_reg[dl * T];
        sfpi::vFloat h = sfpi::dst_reg[dh * T];
        // Fix l: if it is odd and |l| >= 17, then l -= sgn(l)*q, h += sgn(l)
        sfpi::vFloat al = sfpi::abs(l);
        sfpi::vInt pl = sfpi::reinterpret<sfpi::vInt>(al + 8388608.0f) & 1;
        v_if (al > 16.5f && pl != 0) {
            v_if (l < 0.0f) { l = l + vq; h = h - 1.0f; }
            v_else          { l = l - vq; h = h + 1.0f; }
            v_endif;
        }
        v_endif;
        // Fix h: if it is odd and |h| >= 17, then h -= sgn(h)*q (q^2 == 0, so the residue class is unchanged)
        sfpi::vFloat ah = sfpi::abs(h);
        sfpi::vInt ph = sfpi::reinterpret<sfpi::vInt>(ah + 8388608.0f) & 1;
        v_if (ah > 16.5f && ph != 0) {
            v_if (h < 0.0f) { h = h + vq; }
            v_else          { h = h - vq; }
            v_endif;
        }
        v_endif;
        sfpi::dst_reg[dl * T] = l;
        sfpi::dst_reg[dh * T] = h;
        sfpi::dst_reg++;
    }
}
#endif

// Fixes l (reg dl) and h (reg dh) into the digits of method (E). q is the square root of the modulus (odd).
ALWI void Oz2SqsFix(uint32_t dl, uint32_t dh, float q) {
    // Set up the SFPU binary operation (how the address advances). The addition itself is not performed
    add_binary_tile_init();
    MATH(_llk_math_eltwise_binary_sfpu_params_(oz2_sqs_fix_face, dl, dh, dh, VectorMode::RC, q));
}
