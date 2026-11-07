// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: the "modular slicer" on the device (INT8 path, FP64-equivalent)
// ---------------------------------------------------------------------------
// Role: for each element x of the operand, take the integer A' ~ x / 2^e * 2^b (|A' - x/2^e*2^b| <= 1,
//       not necessarily the nearest integer. See "Conditions that the digits satisfy" in digits.h),
//       compute its residues (A' mod m_i) for the S moduli m_i, and output them as bytes that can be
//       passed to the INT8 matrix product. This is the slicing of Ozaki scheme II. b is decided by the
//       host from the CRT window and the dynamic range of the input (54 by default), and 2^e is the
//       normalization scale of each row / column.
//
// Why the value is handled "split into digits"
//   A' is an integer of up to 100 bits, so it does not fit in FP32 (which represents integers exactly
//   only up to 2^24). A' is therefore split into ND radix-2^20 digits d0..d_{ND-1} (balanced
//   representation, signed) (each digit is at most 2^20, so it fits in FP32 exactly. ND follows from b
//   and is 3 to 5).
//   This kernel composes the residue as
//       A' mod m = ( sum_{j>0} (d_j mod m)*c_j + d0 ) mod m
//   where c_j = 2^(20j) mod m (computed by the host). d0 has weight 1, so it is added without being
//   reduced (for the magnitude condition see oz2_residue in digits.h).
//   The digits are reduced modulo m first and multiplied by the weights afterwards, so the intermediate
//   products are below m^2 (255^2 = 65025), and everything fits in FP32 exactly.
//
//   The point is that "the rounded integer A', not the operand x, is split into digits".
//   If x itself were split into FP32 multiple-precision limbs, each limb would be an integer only for
//   elements with |x| near 1, and the method would break down for general inputs.
//
//   The stage that makes the digits themselves (normalization, integer conversion, digit splitting) is
//   in digits.h and is shared with the digit-splitting path (modslice_digit.cpp).
//
// Arguments and buffers
//   Compile-time arguments: 0 = S (number of moduli), 1 = float bit pattern of the digit radix 2^20,
//     2 = number ND of radix-2^20 digits of A' (3 to 5. digits.h). Arguments 3 and 4 are for the
//     digit-splitting path and are not read by this kernel (the host passes the same layout to both
//     paths)
//   Runtime arguments: 0 = ntiles (number of tiles to process)
//   Input   the inputs of the digit-making stage themselves (see digits.h and slice_tf_core.h)
//           b and the per-row / per-column scale 2^e have already been added into the FP64 exponent
//           as the exponent delta by the reader (slice_f64_reader.cpp), so no scale tile is received;
//           c_2 holds the scale s of the low limb of the triple-float (tf_load_f64 in slice_tf_core.h)
//   Output  c_16            : the S residues (symmetric residues, encoded as sign-magnitude uint8 for
//                             the INT8 matrix product)
#include "digits.h"
#include "api/compute/eltwise_unary/typecast.h"

using namespace oz2_args;
constexpr auto cb_out = tt::CBIndex(CB_PLANE_OUT);  // output: the S residues

constexpr uint32_t S = get_compile_time_arg_val(SC_S);   // number of moduli (the host decides it from K)
// Upper bound on the number of moduli (for the static allocation of the arrays. digit_limits.h)
constexpr uint32_t MAXS = OZ2_MAXS;
static_assert(S <= MAXS, "the number of moduli exceeds the size of the fixed arrays");
// The per-modulus MOD[i], the reciprocal RINV[i] and the digit weights C0/C1 (/C2/C3) are all passed as
// runtime arguments (float bit patterns). This lets the host alone replace both the precision b and
// the set of moduli. The layout is sl_* in args.h: ntiles, MOD, RINV,
// C0, C1 (C2 = 2^60 mod m only with 4 or more digits (b > 60), C3 = 2^80 mod m only with 5 digits (b > 80))
constexpr uint32_t DF_F32 = (uint32_t)DataFormat::Float32;
constexpr uint32_t DF_U8  = (uint32_t)DataFormat::UInt8;

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(sl_ntiles());
    float MOD[MAXS], RINV[MAXS], C[MAXS][OZ2_NDIGIT - 1];
    // Read the modulus-related constants from the runtime arguments (all computed by the host)
    constexpr uint32_t A_MOD = sl_mod(S, 0), A_RINV = sl_rinv(S, 0);
    for (uint32_t i = 0; i < S; i++) { MOD[i] = fbf(get_arg_val<uint32_t>(A_MOD + i));
                                       RINV[i] = fbf(get_arg_val<uint32_t>(A_RINV + i)); }
    // C[i][j] = 2^((j+1)*20) mod m_i (digit weights). 4 digits use up to the third, 5 digits up to the
    // fourth.
    constexpr uint32_t A_W = sl_weight(S, 0, 0);   // first digit weight. Modulus i of digit j: A_W + j*S + i
    for (uint32_t i = 0; i < S; i++)
        for (uint32_t j = 0; j + 1 < OZ2_NDIGIT; j++)
            C[i][j] = fbf(get_arg_val<uint32_t>(A_W + j * S + i));
    for (uint32_t ti = 0; ti < ntiles; ti++) {
        // Make the digits of A' from one operand tile (normalization, integer conversion, digit splitting)
        oz2_make_digits();
        for (uint32_t i = 0; i < S; i++) {   // one output tile per modulus
            tile_regs_acquire();
            init_sfpu(cb_dlo, cb_out);
            // reg2 = A' mod m (the symmetric residue, |r| <= floor(m/2) <= 127: for m <= 255 the sum
            // in oz2_residue stays below 2^22; digits.h). Up to here the processing is shared with
            // the digit-splitting path.
            oz2_residue(MOD[i], RINV[i], C[i]);
            SignMag8(2, 3, 4);                   // encode as sign-magnitude (sfpu_tf.h)
            typecast_tile_init<DF_F32, DF_U8>(); typecast_tile<DF_F32, DF_U8>(2);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(2, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        oz2_digits_pop();
    }
}
