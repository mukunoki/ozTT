// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Procedures for triple-float and for residues modulo m (shared through sfpu_tf.h by the compute kernels
// that use the SFPU; the GEMM-stage kernel bmm_int32_spill.cpp does not include it. The procedures call
// only the names of the SFPU tile-operation wrappers and do not depend on the device API)
// ---------------------------------------------------------------------------
// The procedures take the type W of the tile-operation wrapper as a template argument, so that the
// same procedures can be instantiated with other hooks. The kernels instantiate them with the device
// wrapper SfpuDev of sfpu_tf.h (called by the free functions of the same names in sfpu_tf.h). The
// checks described in docs/sfpu_lemmas.md instantiated the very same procedures with a host-side model
// of a single FP32 operation (not part of this distribution); because no copy of the procedures
// exists, such a check only has to confirm that the model of one wrapper operation agrees with the
// hardware.
// What W provides (the device uses SfpuDev of sfpu_tf.h; another instantiation its own wrapper type):
//   Add, Sub, Mul (a, b, o), Cpy (s, d), MulS, AddS (d and a scalar given as a float value or as the bit
//   pattern of a float), Rnd, Ltz,
//   FinalFts (a, b, s, t, w1, w2: the final normalization of tfMulR and tfMulP2. On the device it is
//     exactly FastTwoSum below. Another wrapper may switch it to TwoSum for comparison (which also
//     uses w2 as a work register) and instrument the precondition),
//   SmallTwoSum (the second step of tfAddSmall. On the device it is exactly SmallTwoSum below. Another
//     wrapper may switch it to FastTwoSum for comparison),
//   SPL (the constant 4097 of the Dekker splitting, in the form passed to MulS: a bit pattern on the
//     device, a float in a wrapper that works on float values)
//
// Register conventions (the names are in namespace dst below)
//   A triple-float value is placed in the DST triple acc_hi, acc_mid, acc_lo (high, middle, low).
//   tfMulR and tfMulP2 (and tfMulRadix, which dispatches between the two) use term, tmp0..2 and keep as
//   work registers; tfAddS uses tmp0..2 and term, which holds v (tfAddS does not clobber keep).
#include <cstdint>
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))   // in kernels the tt-metal API headers define the same
#endif

// --- Roles of the DST registers ---------------------------------------------
// The triple-float arithmetic of this header is written on the assumption that the eight DST registers
// are used as follows. Writing the numbers directly would not show which registers may be clobbered and
// which can hold a value across a call.
//   acc_hi/mid/lo  the accumulator triple. Updated by tfMulR, tfAddS and tfAddSmall
//   term           the value v to add. Set by the caller. tfAddS<3> also uses it as a work register and
//                  clobbers it (tfAddS<2> and tfAddSmall only read it)
//   tmp0..2        work registers. Every function may clobber them freely
//   keep           tfAddS and tfAddSmall do not touch it, so the caller can use it for a value that is
//                  held throughout the computation of a term (k_i in recon_crt.cpp). However, tfMulR and
//                  tfMulP2 use term and keep as work registers as well and clobber them, so a value
//                  cannot be kept across them
namespace dst {
constexpr uint32_t acc_hi = 0, acc_mid = 1, acc_lo = 2;
constexpr uint32_t term = 3;
constexpr uint32_t tmp0 = 4, tmp1 = 5, tmp2 = 6;
constexpr uint32_t keep = 7;
}

namespace sfpu_proc {

// --- Error-free transformations ---------------------------------------------
// TwoSum: s = fl(x+y), e = (x+y) - s (recovers the rounding error exactly; Knuth's 6-operation
// version). It is correct for any ordering of the magnitudes of x and y, so it is used wherever the
// ordering cannot be proved; FastTwoSum below (3 operations) is used only where its precondition is
// established (Lemmas P and R, docs/sfpu_lemmas.md). s, e, a, b are mutually distinct registers.
template <class W>
ALWI void TwoSum(W& w, uint32_t x, uint32_t y, uint32_t s, uint32_t e, uint32_t a, uint32_t b) {
    w.Add(x, y, s); w.Sub(s, x, b); w.Sub(s, b, a); w.Sub(x, a, a); w.Sub(y, b, b); w.Add(a, b, e);
}
// FastTwoSum (Dekker): (s, t) = (fl(a+b), b - (fl(a+b) - a)). w1 is a work register. s + t = a + b is
// exact when "a = 0, or b = 0, or the exponent of a >= the exponent of b" (the final normalization of
// tfMulR and tfMulP2; the precondition is given by Lemmas P and R below).
// The procedures call it through the member FinalFts of W (so that another wrapper can replace it with
// TwoSum for comparison and count violations of the precondition. SfpuDev::FinalFts of the device calls
// it as is)
template <class W>
ALWI void FastTwoSum(W& w, uint32_t a, uint32_t b, uint32_t s, uint32_t t, uint32_t w1) {
    w.Add(a, b, s); w.Sub(s, a, w1); w.Sub(b, w1, t);
}

// --- triple-float arithmetic ------------------------------------------------
// [Precondition of the final FastTwoSum of tfMulR and tfMulP2] (contract)
// Both normalize at the end with the three operations (a, b) -> (s, t) = (fl(a+b), b - (fl(a+b) - a))
// (tfMulR with (a,b) = (e_c, G), tfMulP2 with (e6, e4); the names are given in the notes at the bodies
// below). This returns s + t = a + b exactly when "a = 0, or b = 0, or the exponent of a >= the exponent
// of b" (Dekker); otherwise about half an ulp of b is lost. The condition fails when the input triple
// overlaps (multiplying (1, 2^50, 2^26) by the radix 2 gives (2^51, 2^27, 0) and loses the trailing +2
// of the true value).
// Guarantee: when the triple (x0,x1,x2) satisfies H_c(u) (x0 is an integer multiple of a power of two u,
// |x1| + |x2| <= c u) and, for the radix beta,
//   tfMulP2 (beta is a power of two): beta c < 2^23 (Lemma P),
//   tfMulR: beta c (1+2^-22) < 2^8 (Lemma R; for H_4 of the Horner recurrence < 2^10 passes as well)
// then the precondition holds and the output satisfies H_{c1} (c1 = 1 + 2^-24) again. The calls in
// actual use (the slicer entry H_1(u) of Ozaki schemes I and II, and H_4(ulp(s)) and H_{2^11}(1) of the
// Horner recurrence of Ozaki scheme I) are inside this condition.
// This has been checked by building these procedures on a model of single SFPU operations (0 violations
// of the precondition, 0 mismatches), and the model has been confirmed to agree bit for bit with the
// hardware. The proofs of Lemmas P and R and of the conditions H of these calls are in
// docs/sfpu_lemmas.md (Part A).
//
// tfMulR multiplies the triple acc_hi, acc_mid, acc_lo by the radix rdx (in place). rdx is in the form
// passed to MulS of W (the bit pattern of a float on the device).
// The default radices are at most 254 (8 bits), so the product with a half produced by the Dekker
// splitting (12 bits) fits in 12+8 <= 24 bits and the product of each limb is error-free. This takes
// fewer steps than a general triple-word multiplication.
// term, tmp0..2 and keep are used as work registers (term and keep are clobbered here as well). On the
// way acc_hi and acc_mid are used as work registers too, and the note on each line shows the contents
// at that point with the names used in the proof of the lemma.
// Names used in the proof: P1 + F1 = beta x1, P0 + F0 = beta x0 (Dekker products; both exact),
// G1 = fl(F1 + fl(beta x2)), (s_a, e_a) = TwoSum(P1, G1), (s_b, e_b) = TwoSum(F0, s_a),
// (r0, e_c) = TwoSum(P0, s_b), G = fl(e_a + e_b), and finally FastTwoSum(e_c, G) -> (r1, r2).
// The only approximations are the three places fl(beta x2), G1 and G.
template <class W, class R>
ALWI void tfMulR(W& w, R rdx) {
    using namespace dst;
    // lowest limb: fl(beta x2), the first of the three approximations
    w.MulS(acc_lo, rdx);
    // tmp1 = upper half of the middle limb
    w.Cpy(acc_mid, tmp1); w.MulS(tmp1, W::SPL); w.Sub(tmp1, acc_mid, tmp2); w.Sub(tmp1, tmp2, tmp1);
    w.Sub(acc_mid, tmp1, tmp2);                                     // tmp2 = lower half of the middle limb
    w.MulS(tmp1, rdx); w.MulS(tmp2, rdx);                           // both error-free
    // product into term (P1), residual into tmp0 (F1)
    w.Add(tmp1, tmp2, term); w.Sub(term, tmp1, keep); w.Sub(tmp2, keep, tmp0);
    // tmp1 = upper half of the high limb
    w.Cpy(acc_hi, tmp1); w.MulS(tmp1, W::SPL); w.Sub(tmp1, acc_hi, tmp2); w.Sub(tmp1, tmp2, tmp1);
    w.Sub(acc_hi, tmp1, tmp2);                                      // tmp2 = lower half of the high limb
    w.MulS(tmp1, rdx); w.MulS(tmp2, rdx);
    // product into acc_mid (P0), residual into acc_hi (F0)
    w.Add(tmp1, tmp2, acc_mid); w.Sub(acc_mid, tmp1, keep); w.Sub(tmp2, keep, acc_hi);
    w.Add(tmp0, acc_lo, tmp0);                                      // tmp0 = G1
    // Normalize the four components into a triple: (s_a, e_a) = (tmp1, tmp2)
    TwoSum(w, term, tmp0, tmp1, tmp2, keep, acc_lo);
    TwoSum(w, acc_hi, tmp1, term, tmp0, keep, acc_lo);              // (s_b, e_b) = (term, tmp0)
    TwoSum(w, acc_mid, term, acc_hi, tmp1, keep, acc_lo);           // (r0, e_c) = (acc_hi, tmp1)
    w.Add(tmp2, tmp0, tmp2);                                        // tmp2 = G
    // FastTwoSum(e_c, G) -> (r1, r2) (the precondition is Lemma R)
    w.FinalFts(tmp1, tmp2, acc_mid, acc_lo, keep, term);
}

// Multiplies the triple acc_hi, acc_mid, acc_lo by a power of two rdx (in place; the form of rdx is the
// same as for tfMulR).
// Multiplying by a power of two only shifts the exponent of each limb, so the Dekker splitting is not
// needed and the parts of tfMulR above that recover the residuals all become identities. What remains
// is only the normalizing TwoSum chain, and it cannot be dropped: after a digit has been subtracted the
// high limb has become small and the middle limb can be relatively large, so unless the triple is
// brought back to non-overlapping form after the multiplication, the round that extracts the next digit
// looks at the high limb only and goes wrong (term, tmp0..2 and keep are used as work registers, as in
// tfMulR).
// [Condition for bit-for-bit agreement with tfMulR] Every limb is either 0 or such that the intermediate
// values of tfMulR (4097 times the limb, rdx times a half split by Dekker) stay in the range of
// normalized FP32 numbers, roughly 2^-100 <~ |limb| and |high| < 2^116. Then all residuals of the
// splitting in tfMulR are 0 and the remaining normalizing TwoSum chain proceeds in the same order with
// the same values. Outside this condition the two differ: both on the upper side (4097 times the limb
// overflows 2^128) and on the lower side (exponent of the limb in -149..-100; truncation of subnormals)
// in around 40 % of the elements (found by comparing the DST triples directly).
// The value range in actual use is inside on both sides. The triple of the slicers (slice_tf_* of Ozaki
// scheme I, digits.h of Ozaki scheme II) satisfies at the entry after normalization |x| <= 1 for Ozaki
// scheme I and |x| <= 2^20 for Ozaki scheme II (the same as the entry of the induction on H). Elements
// below 2^-100 (elements smaller than the row maximum by a factor of 2^100 or more) can occur, but the
// digits of such an element are all 0 with either way of writing. The limbs of the Horner recurrence of
// the reconstruction (recon_tf_*) are integers or 0, and oz_horner_fits (ozaki1/format.hpp) keeps their
// magnitude well below 2^116.
// Names used in the proof of the lemma: y_i = beta x_i, (s3, e4) = TwoSum(y1, y2),
// (s5, e6) = TwoSum(y0, s3), r0 = s5, and finally FastTwoSum(e6, e4) -> (r1, r2).
template <class W, class R>
ALWI void tfMulP2(W& w, R rdx) {
    using namespace dst;
    w.MulS(acc_hi, rdx); w.MulS(acc_mid, rdx); w.MulS(acc_lo, rdx);   // all error-free
    TwoSum(w, acc_mid, acc_lo, term, tmp0, tmp2, keep);   // term = middle+low (s3), tmp0 = its error (e4)
    TwoSum(w, acc_hi, term, tmp1, tmp2, acc_mid, acc_lo); // tmp1 = high (s5), tmp2 = its error (e6)
    w.Cpy(tmp1, acc_hi);
    // Split the two errors into the middle and low limbs (FastTwoSum; the precondition is Lemma P)
    w.FinalFts(tmp2, tmp0, acc_mid, acc_lo, keep, term);
}

// Whether the significand field of the radix (the bit pattern of a float) is 0, i.e. whether the radix
// is a power of two. The dispatch of the radix multiplication (tfMulRadix of sfpu_tf.h, at compile
// time) selects tfMulP2 or tfMulR with this test (in the value range of actual use the result is the
// same either way; see the condition for agreement at tfMulP2 above)
constexpr bool radix_is_pow2(uint32_t rdx_bits) { return (rdx_bits & 0x7FFFFFu) == 0u; }

// Adds the scalar v (term) to the triple acc_hi, acc_mid, acc_lo (in place). The residuals of TwoSum
// are propagated in turn to the lower limbs, so the upper two limbs stay error-free even though each
// individual FP32 operation rounds to 24 bits. Only the lowest limb has nowhere to pass its residual
// on to, and that is where the approximation is (Add(acc_lo, tmp2, acc_lo) below). Work registers
// tmp0..2. keep is not touched. v is destroyed.
//   NL=3 is triple-float, NL=2 is a pair of FP32 (used only by the two-limb reconstruction of scheme II
//   for a small width b < 24; oz2_recon_limb3 in ozaki2/moduli.hpp).
template <uint32_t NL, class W>
ALWI void tfAddS(W& w) {
    using namespace dst;
    // TwoSum(a0, v): sum into acc_hi, residual into tmp1
    w.Add(acc_hi, term, tmp0); w.Sub(tmp0, acc_hi, tmp1); w.Sub(tmp0, tmp1, tmp2);
    w.Sub(acc_hi, tmp2, tmp2); w.Sub(term, tmp1, tmp1); w.Add(tmp2, tmp1, tmp1); w.Cpy(tmp0, acc_hi);
    if constexpr (NL >= 3) {
        // TwoSum(a1, residual): sum into acc_mid, and the residual of that into tmp2
        w.Add(acc_mid, tmp1, tmp0); w.Sub(tmp0, acc_mid, tmp2); w.Sub(tmp0, tmp2, term);
        w.Sub(acc_mid, term, term); w.Sub(tmp1, tmp2, tmp2); w.Add(term, tmp2, tmp2); w.Cpy(tmp0, acc_mid);
        w.Add(acc_lo, tmp2, acc_lo);                       // the lowest limb is approximated here
    } else {
        w.Add(acc_mid, tmp1, acc_mid);
    }
}

// Addition of a small term (used by the trailing fragments of the CRT in scheme II)
// Skips the first step TwoSum(a0, v) of tfAddS and feeds v in at the step of a1.
// 15 operations -> 8 operations.
//
// No value is lost. The second step is Knuth's TwoSum, which is correct for any ordering of the
// magnitudes, so a1 + v is kept exactly together with its residual, and the approximation enters only
// once, in the addition into the lowest limb Add(2, 6, 2) (i.e. Add(acc_lo, tmp2, acc_lo)), as in
// tfAddS. Therefore no assumption from below on the magnitude of a0 is needed (it holds however small
// a0 becomes through cancellation).
//
// The form of the error is the same as for tfAddS as well. In both, the error produced by one call is
// only the rounding of the lowest limb |fl(a2+e) - (a2+e)| <= 2^-24 |a2+e|; what differs is the amount
// that enters a1. In tfAddS it is the residual of the first step (at most ulp/2 of a0 after the
// addition), in tfAddSmall it is v itself. When the caller guarantees |v| <= eps, the amount that
// enters per call is at most max(ulp(a0)/2, eps) in both cases.
// a1 keeps the amounts that entered in the past, so the bound after n calls cannot be expressed with
// the ulp of the current a0 and becomes
//   |a1| <= n * max(max_t ulp(a0^(t))/2, eps) * (1 + 2^-24)^n
// (a0^(t) is a0 after the t-th addition, and (1 + 2^-24)^n is the rounding of the updates of a1).
// As long as eps is chosen sufficiently smaller than the bound on max_t ulp(a0^(t))/2, this bound is
// the same as for tfAddS.
// In scheme II the integer part is removed for each modulus, so |a0| < 2 at every step (k_i of ModRound
// is within floor(m/2)+1 ((a') in the note of ModRound below; oz2_crt_k_bound in ozaki2/moduli.hpp;
// exceeding the symmetric residue by 1 does not change this bound)) and max_t ulp(a0^(t))/2 <= 2^-24;
// eps is set to vmax * 2^-30 with vmax the largest term, capped at 1 (oz2_choose_small_tail in
// ozaki2/moduli.hpp), so eps <= 2^-30 is 6 bits below this bound on ulp(a0)/2 (7 bits below the bound
// ulp(a0) <= 2^-23, which is how oz2_choose_small_tail counts its margin). The margin is against the
// maximum of ulp(a0^(t))/2 over the whole accumulation, the quantity in the bound above, and not
// against the ulp of the current a0, which can be far smaller after cancellation; the bound does not
// need it to be.
// This bound is loose. In measurements the worst absolute error of frac agrees with the case that uses
// tfAddS only.
//
// [FastTwoSum is not used for the second step] The three operations of Dekker would require
// |a1| >= |v|, which cannot be stated without a lower bound on a0 (a step whose precondition cannot be
// proved is not kept in use by relying on a property of some other place). The error of frac does not
// show up in the end-to-end attained accuracy, so it has to be examined directly on frac.
template <uint32_t NL, class W>
ALWI void tfAddSmall(W& w) {
    using namespace dst;
    if constexpr (NL >= 3) {
        w.SmallTwoSum();                                   // TwoSum(a1, v): SmallTwoSum below, via W
        w.Add(acc_lo, tmp2, acc_lo);                       // the lowest limb is approximated here
    } else {
        w.Add(acc_mid, term, acc_mid);
    }
}

// The second step TwoSum(a1, v) of tfAddSmall: sum into acc_mid, residual into tmp2 (work registers
// tmp0 and tmp1; keep is not clobbered).
// tfAddSmall calls it through the member SmallTwoSum of W (so that another wrapper can replace it with
// FastTwoSum for comparison, as listed at the top. SfpuDev::SmallTwoSum of the device calls it as is).
// Why TwoSum and not FastTwoSum is used here: docs/sfpu_lemmas.md (Part B).
template <class W>
ALWI void SmallTwoSum(W& w) {
    using namespace dst;
    w.Add(acc_mid, term, tmp0); w.Sub(tmp0, acc_mid, tmp2); w.Sub(tmp0, tmp2, tmp1);
    w.Sub(acc_mid, tmp1, tmp1); w.Sub(term, tmp2, tmp2); w.Add(tmp1, tmp2, tmp2); w.Cpy(tmp0, acc_mid);
}

// --- Residue modulo m -------------------------------------------------------
// d -= round(d/m)*m. Preconditions: d is an integer that FP32 represents exactly (|d| <= 2^24; at every
// call site (a)-(e) below d is a digit, a residue, a product of a residue and an integer coefficient,
// the integer output of the GEMM, or an exact sum of such terms), m is an integer modulus of the sets in
// ozaki2/moduli.hpp (2 <= m <= 4096; the float m is exact), and rinv is the FP32 nearest to 1/m (the
// host rounds 1/m, computed in quadruple precision, to FP32: o.rinv in ozaki2/moduli.hpp; for every
// integer m up to 4096 that is the correctly rounded reciprocal, |rinv - 1/m| <= 2^-24/m). The
// guarantee covers only the range |d| < 2^22, where the result lies in the symmetric residue range
// [-m/2, m/2] modulo m (the calls that leave this range are collected in [ ] below).
// Beyond it the result breaks down in two steps: first round selects the neighboring integer (reason
// below), and then, when round(d/m)*m exceeds 2^24, the product itself is rounded in FP32 and even the
// residue class changes (d=2^24, m=29 returns -8, while the correct symmetric residue is -9).
// With an odd modulus the ends are not reached, so the range is effectively (-m/2, m/2), but with an
// even modulus (254) d/m can be exactly a half-integer, and both +m/2 and -m/2 appear depending on the
// direction in which Rnd resolves a tie. Both are in the same residue class and have the absolute value
// m/2, so the later stages (7-bit sign-magnitude, another reduction) are not affected.
// Why 2^22 comes out from the rounding side: fl(d*rinv) deviates from d/m by a relative 2^-23 (2^-24
// from rinv and 2^-24 from the rounding of the product), while the distance between d/m and a
// half-integer is at least 1/(2m) because d and m are integers (2d - (2k+1)m is an integer; it is 0
// only in the tie case of an even modulus above, where both neighbors are correct), so as long as
// |d|/m * 2^-23 < 1/(2m), i.e. |d| < 2^22, round always selects the correct integer.
// [This general bound is not enough for the BF16 / BFP8 / FP16 / TF32 paths, whose moduli go up to 4096]
// Four of the five calls (a)-(e) leave 2^22: (a) r_i*y_i in recon_crt.cpp (up to m^2/2 = 2^23),
// (c) oz2_residue of the slicer (ozaki2/kernels/digits.h; the sum goes up to 2^23.9), (d) the ModRound
// that reduces the digit product P_j modulo m with fold_first=1 in the composition stage (compose.cpp),
// and (e) the ModRound in recon_crt.cpp that reduces the output P of the GEMM of a one-digit modulus
// with RAWMASK (for (d) and (e) the modulus selection of the host imposes |P| + m <= 2^24;
// ozaki2/moduli.hpp).
// (b) (chi mod m)*(2^16 mod m) + (clo mod m) in recon_crt.cpp is the two-plane decoding of
// sign-magnitude int32, and only the INT8 path (m <= 255) goes through it, so it stays below 2^15.
// For the residue class to be preserved, |d| < 2^24 alone is not enough: the product round(d/m)*m also
// has to be at most 2^24 so that it is exactly representable in FP32 (for d=16777215, m=29 the product
// 16777225 is rounded to 16777224 and -9 is returned; the correct value is -10). round selects the
// neighboring integer only when d/m is within |d|*2^-23/m of a half-integer, and then the product is
// only m/2 + |d|*2^-23 away from d. (a) and (c) have |d| <= 2^23.9 with ample room up to 2^24 (about
// 1.1e6), and (d) and (e) have |d| + m <= 2^24, so in all of them |product| <= 2^24 holds. Therefore
// the residue class is preserved. For the excess over the symmetric residue, the bound only says that
// it is below |d|*2^-23 (below 2 for (d) and (e) with |d| + m <= 2^24, hence up to floor(m/2)+2), so
// it is counted call by call as follows.
// An exhaustive check enumerates (a) over the default moduli sets and judges it with
// |k| <= floor(m/2)+1 (with the coefficients of the default sets the excess is 0); as (a') it
// enumerates r*y for all moduli the paths can select and all y coprime to m and judges with the same
// |k| <= floor(m/2)+1 (there are 8170 products that exceed floor(m/2) by exactly 1; for example
// m=3475, r=1737, y=3474 gives k=1738. The bound floor(m/2)+1 is the range that the decision on the
// small terms of the fragments of the reciprocal counts on (oz2_crt_k_bound in ozaki2/moduli.hpp)).
// (b) is enumerated over the moduli of the INT8 path. (c) is examined for every modulus from 2 up to
// the upper limit over all paths and for 3 to 5 digits, by enumeration near the half-integer boundaries
// and with the upper limit of the digits, and the maximum of |r| is also confirmed to be at most the
// bound oz2_resid_bound of the host analysis (excess of at most 1). (e) is enumerated for all one-digit
// moduli (2 or more) near the boundaries and at both ends of |d| <= 2^24 - m and judged with
// |r| <= floor(m/2)+2 (the smaller the modulus, the larger the excess, up to 2). This is about 2.38
// billion cases in total. (d) is not enumerated; as the bound says, the excess reaches 2 even with a
// large modulus (m=2005, d=-16425964 returns -1004 = -(floor(m/2)+2)), so the modulus selection of the
// host (compose_sum in ozaki2/moduli.hpp) and compose.cpp count the reduced value with |t| <= m/2+2.
// When the way the moduli or the coefficients are chosen is changed, this check has to be redone.
// o and d may be the same (in-place update). s is a work register and has to differ from d.
template <class W>
ALWI void ModRound(W& w, uint32_t d, uint32_t o, uint32_t s, float m, float rinv) {
    w.Cpy(d, s); w.MulS(s, rinv); w.Rnd(s); w.MulS(s, m); w.Sub(d, s, o);
}
// The four calls (a), (c), (d), (e) above handle the range up to just below 2^24 with ModRound as it
// is. Only fold_first=0 of the composition stage, where |P| + m can exceed 2^24, first splits P into an
// upper and a lower part at 2^12, applies "the constants obtained by reducing the coefficients modulo
// m" directly to each part, and reduces after that (the procedure is written in
// ozaki2/kernels/compose.cpp).

// Sign-magnitude 8-bit encoding g(d) = d + [d<0]*(128 - 2d) in [0,255] (in place).
// It exists because the matrix-multiplication kernel expects sign-magnitude uint8, and the slicers of
// schemes I and II share it (the actual byte representation is determined by the DataFormat of cb_out).
// w1 and w2 are work registers.
template <class W>
ALWI void SignMag8(W& w, uint32_t d, uint32_t w1, uint32_t w2) {
    w.Cpy(d, w1); w.Ltz(w1);                                         // w1 = [d<0]
    w.Cpy(d, w2); w.MulS(w2, -2.f); w.AddS(w2, 128.f); w.Mul(w2, w1, w2);
    w.Add(d, w2, d);
}

}  // namespace sfpu_proc
