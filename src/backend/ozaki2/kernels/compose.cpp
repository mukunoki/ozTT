// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II (the 7 digit-splitting paths): the stage that composes the digit products and takes
// the residue modulo m (device side)
// ---------------------------------------------------------------------------
// Role: from the np GEMM outputs of one modulus (np is the count after the digit products of the same
//       place value have been concatenated along K: 2 to 4 for 2-digit splitting, n to n^2 for balanced
//       n digits. In a configuration that cuts them into chunks, it is the outputs of one chunk, or the
//       partial sums of the chunks), make the residue r == D (mod m) that the reconstruction needs.
//       A one-digit modulus (AWE path with digit bound 32) does not go through this stage (the
//       reconstruction reduces modulo m).
//       What is computed here is the sum of the outputs j (each the sum of the g digit products with
//       the same place value a+b. gr[j] of oz2_product_groups in split.hpp), each multiplied by the
//       coefficient of its place value:
//           D == sum_j gr[j].coeff_lo * P_j   (mod m)
//       Written with the digit products before the concatenation (in the order of oz2_product_terms.
//       XY is the product of the digit matrix X of A and the digit matrix Y of B, and L, H, S are the
//       low, high and sum digit matrices), this takes the following form for each digit-splitting
//       method (q is the digit radix, B the Karatsuba radix. OZ2_SPLIT_* in split.hpp).
//           SQ, SQS : D == q*(HL + LH) + LL                   (no HH term because q^2 == 0)
//           KAR     : D == (1-B)*LL + (B^2-B)*HH + B*SS       (-15, 240, 16 for B=16. For B=8 of BFP4,
//                                                              -7, 56, 8)
//           WIDE    : D == q^2*HH + q*(HL + LH) + LL
//           BAL     : D == sum_{i,j} q^(i+j) P_ij
//       Digit products of the same place value also have the same coefficient, so one concatenated
//       output (HL and LH of SQ and WIDE, the P_ij with the same i+j of BAL) only has to be multiplied
//       once by the coefficient of that place value.
//
// Why D itself does not have to be formed
//   With the digit bound d, |D| reaches K*d^2*(q+1)^2 for WIDE (4.2e9 for d=16 and K=14080 of FP8) and
//   exceeds the exact integers 2^24 of FP32. However, what is needed is not D itself but only D mod m,
//   and the composition is linear with respect to the modulus, so the digit products can be reduced
//   modulo m before they are multiplied by the coefficients.
//
// There are two composition methods, and the host selects one with the runtime argument CP_FOLD_FIRST
// (fold_first in moduli.hpp).
//   fold_first=1: the digit products are first reduced modulo m with ModRound, and the raw coefficients
//                 are applied afterwards. If |P_j|+m <= 2^24, the product of the quotient and the
//                 modulus (a little more than |P_j|+m/2) is exact, so the residue class is kept; the
//                 reduced value t_j is a representative of that class, not necessarily the symmetric
//                 residue (|P_j| may exceed the 2^22 of the general guarantee of ModRound), with
//                 |t_j| <= m/2+2, so the total sum is only (sum|c_j|)(m/2+2).
//                 The host selects this method when the bound of one output (the sum of the g
//                 concatenated products. oz2_group_bound) plus m is at most 2^24 and this total sum,
//                 with the actual coefficients, fits within 2^22. It takes fewer operations because no
//                 split is needed. For K=14080 all 7 digit-splitting paths use this method (moduli with
//                 digit bound 32 are possible up to K < 16384. The path that selects the moduli with a
//                 model of the stage times selects moduli with bound 31 or less when K is large. See
//                 the accumulation condition in moduli.hpp).
//   fold_first=0: "Reducing the coefficients together with the digit products" below. It also handles
//                 the case where a digit product plus the modulus exceeds 2^24, but each digit product
//                 has to be split into a high and a low part.
//                 Both methods assume that the digit products themselves are exact integers
//                 (the host rejects K*d^2 > 2^24. Low bits lost in the FP32 accumulation of the GEMM
//                 do not come back by splitting at 2^12 here).
//
// Reducing the coefficients together with the digit products
//   A digit product is split at 2^12 (the number of bits of the radix is OZ2_COMPOSE_SPLIT_BITS, shared
//   with the host. digit_limits.h) into a high part hi and a low part lo, and instead of reducing them
//   modulo m at that point, constants that already include the coefficient c_j are applied directly:
//       c_j * P_j == hi * (c_j*2^12 mod m) + lo * (c_j mod m)   (mod m)
//   The host passes A_j = (c_j*2^12) mod m and B_j = c_j mod m (coeff_hi and coeff_lo of
//   oz2_product_groups in split.hpp). This removes the residue computation per digit product, and a
//   single reduction modulo m at the end is enough.
//
// Condition on the total sum
//   In this method the digit products are not concatenated (the host sets maxg back to 1), so an output
//   is a single digit product with |P_j| <= Pmax (Pmax is the bound of the output. With the uniform
//   digit bound d of a path it is K*d^2, which is the exactness condition of this path itself. For the
//   path that mixes in one-digit moduli and for the sparse-digit path it is the product of the bounds
//   of the individual digit matrices, oz2_group_bound). hi is obtained by rounding to nearest (Rnd),
//   not by truncation, so |hi| <= floor(Pmax/2^12 + 1/2) (K/16 rounded, for d=16. For K=64 and d=7,
//   P=3136 gives hi=1 although Pmax/2^12 = 0.77), and |lo| <= 2^11. The total sum is at most
//   sum_j (|hi|*A_j + |lo|*B_j), and the final ModRound is applied to it, so the limit to be respected
//   is not the exact integers 2^24 of FP32 but 2^22, up to which ModRound returns the correct symmetric
//   residue. The coefficients differ greatly from modulus to modulus, so the modulus selection in
//   moduli.hpp (compose_sum) checks this with the bound on the rounded |hi| above and the actual A_j
//   and B_j (FP8 stays at 2^21.2 at most even for K=16384. The paths with d=32 exceed the limit at
//   K=16384, so the modulus selection returns failure for a K that would fall into this method). With
//   either method the sum that enters the final ModRound is below 2^22, so the output is the symmetric
//   residue [-m/2, m/2] (for an odd modulus the ends are not reached, so it is effectively
//   (-m/2, m/2). For an even modulus both +m/2 and -m/2 can occur because the quotient is rounded to
//   nearest even. ModRound in sfpu_tf_proc.h), and the reconstruction accepts either sign
//   (recon_crt.cpp).
//
// DST registers (8 with fp32_dest_acc)
//   0 = digit product just read   4,5 = scratch   6 = total sum   the others are unused
// The digit products are loaded into DST one at a time and added into the total sum. The balanced
// 5-digit path has np = 25, so placing all of them side by side in DST is not possible.
// Precondition: the host sets fp32_dest_acc_en and c_0 is received with UnpackToDestFp32.
//
// Arguments and buffers
//   Runtime arguments (the layout is ComposeRT in args.h): ntiles, np (number of GEMM outputs), m, 1/m,
//              fold_first, followed by MAXPROD values each of B_j (= c_j mod m. 0 beyond np) and
//              A_j (= c_j*2^12 mod m)
//   Input   c_0  : the np GEMM outputs (in the order of the outputs j0.. of oz2_product_groups on the
//                  host. Exact integers in FP32. Pushed by the plane reader plane_reader.cpp)
//   Output  c_16 : the residue r (FP32. Symmetric residue)
#include "../../common/kernels/sfpu_tf.h"
#include "args.h"     // argument layouts and CB numbers (the limit OZ2_MAXPROD is in digit_limits.h)

constexpr auto cb_p   = tt::CBIndex(oz2_args::CB_PLANE_IN);    // input (pushed by plane_reader.cpp)
constexpr auto cb_out = tt::CBIndex(oz2_args::CB_PLANE_OUT);   // output (written by modslice_writer.cpp)
// Radix of the 2^12 split and its reciprocal (both are powers of 2, so they are exact in float)
constexpr float SPLIT     = (float)(1u << OZ2_COMPOSE_SPLIT_BITS);
constexpr float SPLIT_INV = 1.0f / SPLIT;

void kernel_main() {
    using namespace oz2_args;
    const uint32_t ntiles = get_arg_val<uint32_t>(CP_NTILES);
    const uint32_t np     = get_arg_val<uint32_t>(CP_NP);
    const float m    = fbf(get_arg_val<uint32_t>(CP_MOD));
    const float rinv = fbf(get_arg_val<uint32_t>(CP_RINV));
    // Composition method. 1 reduces the digit products modulo m first (see the loop below and the
    // header comment). B[] then holds the raw coefficients.
    const uint32_t fold_first = get_arg_val<uint32_t>(CP_FOLD_FIRST);
    constexpr uint32_t MAXPROD = OZ2_MAXPROD;   // limit shared with the host (digit_limits.h)
    float B[MAXPROD], A[MAXPROD];
    // The coefficients are passed in MAXPROD slots (the number of runtime arguments is the same for
    // every modulus. The layout is cp_coeff_* in args.h)
    for (uint32_t j = 0; j < np; j++) { B[j] = fbf(get_arg_val<uint32_t>(cp_coeff_lo(j)));
                                        A[j] = fbf(get_arg_val<uint32_t>(cp_coeff_hi(j))); }

    for (uint32_t ti = 0; ti < ntiles; ti++) {
        cb_wait_front(cb_p, np);
        tile_regs_acquire();
        init_sfpu(cb_p, cb_out);
        copy_tile_to_dst_init_short(cb_p);
        // Load the digit products one at a time into reg0 and add them into the total sum reg6 (the
        // conditions of the two methods are explained in the header comment).
        for (uint32_t j = 0; j < np; j++) {
            copy_tile(cb_p, j, 0);
            if (fold_first) {
                ModRound(0, 4, m, rinv);                       // t_j = P_j mod m (|t_j| <= m/2+2)
                MulS(0, B[j]);                                 // c_j * t_j (B[] holds the raw coefficients)
                if (j == 0) Cpy(0, 6); else Add(6, 0, 6);
            } else {
                Cpy(0, 4); MulS(4, SPLIT_INV); Rnd(4);         // hi = round(P_j / 2^12)
                Cpy(4, 5); MulS(5, SPLIT); Sub(0, 5, 5);       // lo = P_j - hi*2^12
                MulS(4, A[j]); MulS(5, B[j]); Add(4, 5, 4);    // c_j*P_j == hi*A_j + lo*B_j
                if (j == 0) Cpy(4, 6); else Add(6, 4, 6);
            }
        }
        ModRound(6, 4, m, rinv);   // reduce the total sum modulo m (symmetric residue)
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(6, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_p, np);
    }
}
