// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: digit-splitting modular slicer (shared by the 7 formats other than INT8)
// ---------------------------------------------------------------------------
// The implementation that corresponds to the FP8 path of GEMMul8 is also used for BF16, FP16, TF32
// (stored as Float32 tiles), BFP8, BFP4 and BFP2. It implements the splitting of the residues into
// digits in Ozaki scheme II and the digit bounds. The explanation below takes FP8 as the example; the
// only differences between the formats are the digit bound (FP8 16 (32 for sparse digits); BF16, FP16,
// TF32 and BFP8 32; BFP4 7; BFP2 1) and the corresponding choice of moduli on the host side.
//
// Why the residue is split into digits
//   FP8_E4M3 represents consecutive integers exactly only in the range [-16,16], so a residue
//   r = A' mod m (m up to 2401) does not fit in one FP8 value. The signed residue r (a representative
//   with |r| <= floor(m/2)+1; oz2_residue in digits.h) is therefore split into digits, and the product
//   is assembled from the matrix products of the digits. There are six methods (A)-(F), depending on
//   the modulus (OZ2_SPLIT_* in split.hpp, listed there in the same order). (A) and (B) need only 3
//   matrix products, and every modulus of the GEMMul8 modulus table falls under one of the two. (C)
//   appears when the trailing moduli are reselected to fill the remainder of the window, or when the
//   path that selects the moduli with a model of the stage times (awe.hpp) picks a non-square modulus
//   above 513. (D) is used only by BFP2 and BFP4, (E) only by FP8, and (F) only by the AWE path
//   with digit bound 32 (BF16, FP16, BFP8).
//
//   (A) Square modulus m = q^2: balanced digits by rounding to nearest
//         h = round(r/q),  l = r - q*h        (|l| <= q/2, |h| <= round((m/2)/q))
//       Since q^2 == 0 (mod m), the high x high term vanishes and 3 matrix products HL, LH, LL remain:
//         r_a*r_b == q*(h_a l_b + l_a h_b) + l_a l_b   (mod m)
//       The output is the 2 planes l, h.
//
//   (B) Non-square modulus: Karatsuba with radix B (B is a per-modulus runtime argument. The paths whose
//       digit bound d is 15 or more (d=16 of FP8, d=32 of BF16, FP16, TF32 and BFP8) use B = 16, and
//       BFP4 (d=7) uses B = d+1 = 8. See oz2_kar_base in split.hpp.)
//         h = sign(r)*ceil(|r|/B),  l = r - B*h,  s = h + l
//       Because of the rounding up, h and l always have opposite signs, so |h|, |l| and |s| all stay
//       within max(ceil((m/2)/B), B-1) (within 16 for B = 16. With rounding to nearest, |h+l| grows up
//       to 24 and leaves the exact integers of e4m3).
//       The matrix products are the 3 products HH, LL, SS, and the middle term is obtained by subtraction:
//         r_a*r_b = B^2*HH + B*(SS - HH - LL) + LL
//       The output is the 3 planes l, h, s. The digits fit up to m <= 513 for B = 16 and up to
//       m <= 113 for B = 8.
//
//   (C) Non-square modulus whose digits do not fit with Karatsuba: balanced digits by rounding to nearest
//       (q ~ sqrt(m). OZ2_SPLIT_WIDE on the host).
//       On the paths with digit bound 16 or 32 this applies to the non-square moduli above 513
//       (Karatsuba with radix 16 keeps the digits within 16 only up to m <= 513. The symmetric residue
//       satisfies |r| <= floor(m/2), so |h| <= 16 even for m = 513). For BFP4 it applies to the
//       non-square moduli 113 < m <= 225 that do not fit Karatsuba with radix 8 (for BFP4, the squares
//       up to 225 = 15^2 are all (A) and those from 256 on are (D). Non-squares above 225 also become
//       the balanced 3 digits of (D)).
//       The high x high term does not vanish, so there are 4 matrix products HH, HL, LH, LL:
//         r_a*r_b == q^2*HH + q*(HL+LH) + LL   (mod m)
//       The digits are made in the same way as in (A), so this kernel does not need to distinguish (C)
//       from (A) (only the composition formula on the host side differs). The output is the 2 planes l, h.
//
//   (D) Balanced n digits: the balanced representation r = sum_j x_j q^j (|x_j| <= d) with radix
//       q = 2d+1 is peeled off one digit at a time toward the high end:
//         x_j = cur - q*round(cur/q),  cur <- round(cur/q)
//       Since q is odd, round has no ties (cur/q is never a half-integer) and the representation is
//       unique. The output is the n planes x_0..x_{n-1}. The composition coefficients are q^(i+j) and
//       are held by the host.
//       The number of digits n (1..5) is the per-modulus runtime argument NDIG[i]. BFP2 (d=1, q=3)
//       uses, for each modulus, the smallest n with m <= 3^n (5 digits would take 25 products, whereas
//       a modulus up to 81 takes 4 digits and 16 products, and one up to 27 takes 3 digits and 9
//       products). BFP4 (d=7, q=15) carries with 3 digits the moduli 225 < m <= 3375 that the 2-digit
//       methods cannot carry.
//       The top digit is the cur left after the peeling (reg2); the lower digits are placed in
//       reg 0,1,3,5.
//
//   (E) Square modulus with sparse digits (FP8 only. m = q^2, q odd, 33 < q <= 49): the digits are made
//       by rounding to nearest as in (A) and are then moved into the integers that e4m3 represents
//       exactly, S = {|x| <= 16} U {even x : 16 < |x| <= 32} (ozaki2/kernels/sparse_fix.h. SFPI). A low
//       digit that is odd with |t| >= 17 is made even by subtracting q, and 1 is carried into the high
//       digit; the high digit is fixed in the same way (q^2 == 0, so the high digit may be moved
//       modulo q). The products and the composition are the same as in (A).
//       The output is the 2 planes l, h (|l|,|h| <= 32).
//
//   (F) One digit (AWE path with digit bound 32. OZ2_SPLIT_ONE on the host): for a small modulus
//       (m <= 65) whose whole residue fits within the digit bound, the residue itself is output as a
//       single digit matrix. There is 1 matrix product, and the composition stage is not used (the
//       reconstruction reduces the GEMM output modulo m. recon_crt.cpp).
//
//   The rounding up of (B) is made with round: for an integer u >= 0,
//       ceil(u/q) = round((u + q/2 - 0.5)/q)
//   (the argument on the right-hand side is (u+q-1)/q - 1/2 + 1/(2q), and since round(x) = floor(x + 1/2)
//   the right-hand side becomes floor((u+q-1)/q + 1/(2q)). (u+q-1)/q moves in steps of 1/q, so adding
//   1/(2q) does not cross an integer, and the right-hand side equals floor((u+q-1)/q) = ceil(u/q). The
//   argument is never a half-integer, so no rounding tie occurs either).
//   round is an odd function, so the sign can be factored out: h = round((r + sign(r)*(q/2-0.5))/q).
//
// The output per modulus is 2 planes ((A)(C)(E)), 3 planes ((B)), n planes ((D)) or 1 plane ((F)),
// nplane planes in total. The order matches poff[] of the host.
//
// Arguments and buffers
//   Compile-time arguments: 0 = S (number of moduli), 1 = float bit pattern of the radix 2^20 that
//     splits A', 2 = number of radix-2^20 digits of A', 3 = NRDIG (upper bound on the number of digits
//     a residue is split into. 2 or 5. A balanced-n-digit modulus is carried with NDIG[i] digits even
//     on the 2-digit path),
//     4 = OUT_HALF (1 outputs the digits as binary16 bit patterns. Only the FP16 path sets 1)
//   Runtime arguments: 0 = ntiles (for the rest, see the comment above kernel_main below)
//   Input   the inputs of the digit-making stage themselves (see digits.h and slice_tf_core.h)
//   Output  c_16          : nplane digit matrices (in the GEMM input format. UInt16 on the FP16 path)
// The stage that makes the radix-2^20 digits of A' is shared with the INT8 path and is in digits.h.
#include "digits.h"
#include "sparse_fix.h"
#include "api/compute/eltwise_unary/typecast.h"

using namespace oz2_args;
constexpr auto cb_out = tt::CBIndex(CB_PLANE_OUT);

constexpr uint32_t S = get_compile_time_arg_val(SC_S);
// SC_NRDIG = upper bound on the number of digits a residue is split into. 2 selects (A)(B)(C)(E)(F) above
// (BFP4 additionally uses the 3 digits of (D) at run time), 5 selects (D), balanced n digits (n is at
// most 5 and is set per modulus).
constexpr uint32_t NRDIG = get_compile_time_arg_val(SC_NRDIG);
// 1 outputs the digits as IEEE binary16 bit patterns (UInt16) (FP16 path. HalfBits in sfpu_tf.h).
// This works around the values being corrupted when packing directly from the FP32 DST to Float16.
constexpr uint32_t OUT_HALF = get_compile_time_arg_val(SC_OUT_HALF);
constexpr uint32_t DF_F32   = (uint32_t)DataFormat::Float32;
constexpr uint32_t DF_U16   = (uint32_t)DataFormat::UInt16;
// The balanced-n-digit branch (method (D), below) does not convert the digits to binary16 (it continues
// without passing through HalfBits). The FP16 path (digit bound 32) does not use balanced n digits
// (oz2_split_mode on the host does not assign BAL to the paths with digit bound 15 or more). XMODE=3 can
// still occur at run time with NRDIG=2 (the balanced 3 digits of BFP4), so the static_assert below
// only forbids the combination with the 5-digit configuration; that the FP16 path has no BAL modulus is
// checked by the host (build_slice in stages.hpp).
static_assert(!OUT_HALF || NRDIG <= 2, "the binary16 output is used only on the 2-digit splitting path");
constexpr uint32_t MAXS = OZ2_MAXS;   // digit_limits.h
static_assert(S <= MAXS, "the number of moduli exceeds the size of the fixed arrays");
// Numbers of the digit-making modes (Oz2XMode in digit_limits.h). When a number is added, add the branch
// below and then update this assertion (otherwise a modulus with the new number silently falls into the
// default branch (2 digits by rounding to nearest)).
static_assert(OZ2_XMODE_COUNT == 5, "a digit-making mode was added: add the branch in kernel_main");
// The upper bound on the number of digits is 2 ((A)(B)(C)(E)(F); BFP4 additionally the 3 digits of (D))
// or 5 ((D). BFP2).
static_assert(NRDIG == 2 || NRDIG == 5, "NRDIG must be 2 ((A)(B)(C)(E)(F), 3-digit (D)) or 5 ((D), BFP2)");
// Runtime arguments (S values each, in the order of the moduli. The layout is sl_* in args.h):
//   ntiles, MOD, RINV = 1/m, the constants 2^(20k) mod m (k = 1..NDIGIT-1), which are the weights of the
//   radix-2^20 digits of A' reduced modulo m, followed by Q (radix of the digit splitting), QINV = 1/q,
//   XMODE (how the digits are made: Oz2XMode in digit_limits.h. RND = 2 digits by rounding to nearest
//   (square, WIDE), KAR = Karatsuba with rounding up (radix q), SQS = (E) square with sparse digits,
//   BAL = (D) balanced n digits, ONE = (F) one digit), NDIG (number of digits n of (D). 0 otherwise)

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(sl_ntiles());
    float MOD[MAXS], RINV[MAXS], C[MAXS][OZ2_NDIGIT - 1], Q[MAXS], QINV[MAXS], XMODE[MAXS];
    constexpr uint32_t A_MOD = sl_mod(S, 0), A_RINV = sl_rinv(S, 0);
    for (uint32_t i = 0; i < S; i++) { MOD[i] = fbf(get_arg_val<uint32_t>(A_MOD + i));
                                       RINV[i] = fbf(get_arg_val<uint32_t>(A_RINV + i)); }
    constexpr uint32_t A_W = sl_weight(S, 0, 0);   // first digit weight. Modulus i of digit j: A_W + j*S + i
    for (uint32_t i = 0; i < S; i++)
        for (uint32_t j = 0; j + 1 < OZ2_NDIGIT; j++)
            C[i][j] = fbf(get_arg_val<uint32_t>(A_W + j * S + i));
    constexpr uint32_t A_Q = sl_q(S, OZ2_NDIGIT, 0), A_QINV = sl_qinv(S, OZ2_NDIGIT, 0),
                       A_XMODE = sl_xmode(S, OZ2_NDIGIT, 0), A_NDIG = sl_ndig(S, OZ2_NDIGIT, 0);
    for (uint32_t i = 0; i < S; i++) { Q[i] = fbf(get_arg_val<uint32_t>(A_Q + i));
                                       QINV[i] = fbf(get_arg_val<uint32_t>(A_QINV + i)); }
    uint32_t NDIG[MAXS];
    for (uint32_t i = 0; i < S; i++) XMODE[i] = fbf(get_arg_val<uint32_t>(A_XMODE + i));
    for (uint32_t i = 0; i < S; i++) NDIG[i] = get_arg_val<uint32_t>(A_NDIG + i);
    for (uint32_t ti = 0; ti < ntiles; ti++) {
        // Make the digits of A' from one operand tile (normalization, integer conversion, digit splitting)
        oz2_make_digits();
        for (uint32_t i = 0; i < S; i++) {
            tile_regs_acquire();
            init_sfpu(cb_dlo, cb_out);
            const float q = Q[i], qinv = QINV[i];
            // reg2 = A' mod m (a signed representative, |r| <= floor(m/2)+1; digits.h). Up to here the
            // processing is shared with the INT8 path.
            oz2_residue(MOD[i], RINV[i], C[i]);
            // Splitting into digits. The point is to split the signed residue as it is, without
            // bringing it back to [0,m); this halves the bound on the digit magnitudes, so that the
            // digits fit in the exact integers of FP8.
            const bool kar = XMODE[i] == (float)OZ2_XMODE_KAR, sqs = XMODE[i] == (float)OZ2_XMODE_SQS,
                       bal = XMODE[i] == (float)OZ2_XMODE_BAL, one = XMODE[i] == (float)OZ2_XMODE_ONE;
            if (bal) {
                // Method (D): balanced n digits (n = NDIG[i], 1..5). All n planes are made within one
                // acquire (the contents of DST do not survive once it is released).
                // reg2 is cur (the residue at first), reg6 the quotient, reg7 scratch. As for the digit
                // locations OUT, the lower n-1 digits use registers that do not overlap these, and the
                // top digit is the cur (reg2) that remains at the end (for n=1 the residue itself is
                // the digit).
                constexpr uint32_t OUT[4] = {0, 1, 3, 5};
                const uint32_t nd = NDIG[i];
                for (uint32_t j = 0; j + 1 < nd; j++) {
                    Cpy(2, 6); MulS(6, qinv); Rnd(6);        // h = round(cur/q)
                    Cpy(6, 7); MulS(7, q); Sub(2, 7, OUT[j]);// x_j = cur - q*h
                    Cpy(6, 2);                               // cur <- h
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                for (uint32_t j = 0; j < nd; j++) {
                    cb_reserve_back(cb_out, 1); pack_tile(j + 1 < nd ? OUT[j] : 2u, cb_out); cb_push_back(cb_out, 1);
                }
                tile_regs_release();
                continue;                                    // skips the 2-digit processing below
            }
            if (one) {
                // Method (F): one digit. The residue itself (|r| <= 32. The moduli m <= 65 of the AWE
                // path with digit bound 32) becomes a single digit matrix.
                Cpy(2,5);
            } else if (!kar) {
                // Method (A)/(C): balanced digits by rounding to nearest. h = round(r/q), l = r - q*h.
                Cpy(2,6); MulS(6, qinv); Rnd(6);                              // hi -> reg6
                Cpy(6,4); MulS(4, q); Sub(2,4,5);                             // lo -> reg5
                // Method (E): move the digits into the exact integers of e4m3 (|x| <= 16, or even and
                // at most 32)
                if (sqs) Oz2SqsFix(5, 6, q);
            } else {
                // Method (B): Karatsuba digits by rounding up. h = sign(r)*ceil(|r|/q),
                // l = r - q*h, s = h + l. The form with the sign factored out,
                //   h = round((r + sign(r)*(q/2-0.5))/q)
                // is computed as it is (sign(r) is made from Ltz).
                Cpy(2,4); Ltz(4); MulS(4,-2.f); AddS(4,1.f);                  // sgn = +-1 -> reg4
                MulS(4, q*0.5f - 0.5f);                                       // sgn*(q/2-0.5)
                Add(2,4,6); MulS(6, qinv); Rnd(6);                            // hi -> reg6
                Cpy(6,4); MulS(4, q); Sub(2,4,5);                             // lo -> reg5
                Add(6,5,7);                                                   // s = hi+lo -> reg7
            }
            if constexpr (OUT_HALF) {
                // FP16 path: the digits (reg5 = lo, reg6 = hi, and reg7 = s for Karatsuba) are turned
                // into integer values equal to their binary16 bit patterns and are then typecast to
                // UInt16. reg0..4 are no longer used at this point, so reg0,1,3 are passed to HalfBits
                // as scratch. The digits are at most +-32, inside the HalfBits contract |d| <= 2048.
                HalfBits(5, 0, 1, 3);
                if (!one) HalfBits(6, 0, 1, 3);
                if (kar) HalfBits(7, 0, 1, 3);
                typecast_tile_init<DF_F32, DF_U16>();
                typecast_tile<DF_F32, DF_U16>(5);
                if (!one) typecast_tile<DF_F32, DF_U16>(6);
                if (kar) typecast_tile<DF_F32, DF_U16>(7);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            // Reserve and push one tile at a time. Moduli with 2 planes and with 3 planes are mixed, so
            // reserving the planes of a modulus together would not divide the ring length of the CB
            // (16), and a write that crosses the boundary would be corrupted (the same reason why
            // recon_crt takes a depth that is a multiple of the number of limbs).
            auto emit = [&](uint32_t r) {
                cb_reserve_back(cb_out, 1); pack_tile(r, cb_out); cb_push_back(cb_out, 1);
            };
            emit(5);                 // lo digit (the residue itself in the one-digit method)
            if (!one) emit(6);       // hi digit
            if (kar) emit(7);        // sum digit (third matrix of Karatsuba)
            tile_regs_release();
        }
        oz2_digits_pop();
    }
}
