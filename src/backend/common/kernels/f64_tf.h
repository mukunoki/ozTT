// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Bit-level procedures that convert between FP64 and triple-float (shared by Ozaki schemes I and II)
// ---------------------------------------------------------------------------
// Used by the data-movement kernels (readers and writers). The point is that no floating-point
// arithmetic is used at all: the RISC-V cores of the device have no FP64 unit (only part of the
// single-precision F extension and no double-precision D extension; BabyRISCV/InstructionSet.md of the
// Blackhole ISA documentation), so code written with double becomes soft-float and is much slower.
//
//   split_f64          entry: the two words of an FP64 -> four bit patterns of a triple-float
//   assemble_tf_f64    exit (scheme I): three limbs -> FP64. 64-bit accumulator
//   assemble_frac_f64  exit (scheme II): three or six limbs -> FP64. Folds frac back exactly in 128-bit
//                      fixed point before assembling (values in (-1/2,1/2) are unchanged and an excess
//                      is brought back by +-1. The image is [-1/2,1/2]; the handling of the endpoints
//                      is described in the note at assemble_frac_f64 / a128_frac_to_f64)
//
// There are two exits because the width to collect differs between the schemes:
//   scheme I   carries the result itself, so 64 bits suffice
//   scheme II  frac has a magnitude around 2^-8 and has to keep an absolute precision of 2^-99,
//              which needs a width of more than 90 bits
// The stage after collecting, which multiplies by the final scale and rounds (pack_f64), is shared.
//
// The header does not depend on the device API (only <cstdint>), so the host side can include it as is.
//
// What is guaranteed, and what is not
//   - split is exact: hi + mid + lo equals x * 2^d exactly (within the scope of application below).
//   - The output of assemble_* is within 1 ulp of the true value rounded to FP64, as long as the limbs
//     fit in the window. The direction of a tie (exactly half an ulp) is not specified: ties are sent
//     away from zero.
//     When the limbs do not fit in the window, the error is not bounded by 1 ulp. In scheme I the bottom
//     of the window is 60 bits below the leading bit of the limb with the largest exponent (37 bits
//     below the least significant bit of its significand), so when the upper two limbs cancel heavily
//     the bits below the bottom of the window are lost (with a reference of 1, bits down to 2^-60
//     remain and 2^-61 is lost)
//     (hi=1, mid=-1+2^-24, lo=2^-61 returns 2^-24, while the true value is 2^-24+2^-61).
//     Scheme II is similar: within one limb, what is below 2^-124 is lost (2^-120+2^-143, for example).
//     In both cases the amount lost is at most "the magnitude of that limb". Seen element by element,
//     the relative error with respect to the value after cancellation can become arbitrarily large.
//     Guaranteeing 1 ulp per element would require holding the accumulation at full width (192 bits
//     for scheme II).
//     The norm can be said to be unaffected only when the output as a whole does not become small
//     through cancellation. This is the case for the evaluated inputs, where the target is accuracy in
//     the norm. It does not necessarily hold when the scheme is applied to a problem whose output
//     itself cancels heavily.
//     Exact round-to-nearest-even is not implemented. What is needed here is the prescribed digit
//     range, the error-free condition of the integer GEMM, and the window condition of the CRT, and
//     the direction of a rounding tie changes none of these conditions. ModRound produces ties with an
//     even modulus, but both directions give the same residue class; the +-1/2 of the frac folding
//     does not occur because the window is strict. The direction of a tie changes the answer only in
//     the lowest bit of the FP64 returned to the user, and no later stage consumes it. The goal is an
//     accuracy equivalent to a native FP64 DGEMM, not the reproduction of correct rounding.
//   - The final scale W is computed by the host in long double (64-bit significand), so that the
//     multiplication by it does not put the relative 2^-53 of an FP64 constant on every element. Its
//     accumulated rounding is, in relative terms, at most about (p+1)*2^-64 for radix^-(p+1) of scheme I
//     (p slices; the radix is divided p+1 times; exact when the radix is a power of two; recon_scale in
//     ozaki1/plan.hpp) and about (s-1)*2^-64 for KM = prod m_i * 2^-2b of scheme II (the moduli are
//     multiplied s-1 times; KMl in ozaki2/moduli.hpp), i.e. several bits below 2^-53 for the
//     configurations of the library. W is not the original constant itself.
//
// Scope of application (to be satisfied by the caller)
//   - The input is limited to finite, normalized FP64 numbers. Subnormals and 0 are flushed to 0, and
//     Inf/NaN are not checked here. For the operand as a whole, the scale-computation stage has a
//     narrower restriction: that stage detects and stops on elements of 2^128 or more, rows/columns
//     whose largest nonzero absolute value is below 2^-126, and Inf/NaN elements ("Accepted magnitude
//     range" in common/kernels/scale_f64_reader.cpp).
//   - split has been verified to be exact in the range where the normalized value |x*2^d| is 2^-72 or
//     more (verified for values lowered by 0 to 72 binades). Below that, the first to lose digits is
//     the low limb lo = c*2^elsb (split_f64 below; elsb is the exponent of the least significant bit
//     of the significand, and the value is about 2^(elsb+52)): if the SFPU flushes FP32 subnormals to
//     0, lo is lost when c*2^elsb falls below 2^-126 (for c=1, elsb <= -127, i.e. a normalized value
//     below 2^-74) (if it does not flush, lo stays exact as a subnormal).
//     For elsb <= -134, where s (and mid = b*s) becomes 0, i.e. below 2^-81, both mid and lo vanish
//     ("exponent of s" below). Below 2^-126 hi becomes 0 as well. Whether the SFPU of the actual
//     hardware flushes subnormals has not been verified, so the guarantee is for 2^-72 and above.
//     All binade counts are relative to a normalized value of 1; to read them as a distance from the
//     maximum, subtract the position the maximum is moved to:
//     scheme I moves the row/column maximum to around 1/2 ((1/4, (1+2^-7)/2) for the truncating
//     formats, [1/2, 1) for BFP2), so 2^-72 is about 70 to 72 binades below the maximum. In scheme II
//     the position depends on b and on the number of digits ND of A' (exponent shift
//     sh_i = b - 20(ND-1), oz2_shift in ozaki2/plan.hpp; the maximum falls in (2^(sh_i-1), 2^sh_i]),
//     and the distance from the maximum is about 72 + sh_i binades (for example, with b=1 and ND=3 the
//     maximum falls around 2^-39, so elements about 33 binades below it are already outside; with the
//     default b=54 and ND=3 it is about 86).
//     This range is wider than the width the later stages retain: the slices of scheme I hold down to
//     (number of slices) x (bits per slice) below the row/column maximum (59 to 64 bits for the
//     default p), and the integer conversion of scheme II holds down to |A'| <= 2^b (default b=54);
//     elements and bits below that are not retained with the default settings and do not enter the
//     result.
//     Whether the elements that are not retained may be ignored is decided by the layout of the input
//     and cannot be stated in general: even an element that is small compared with the maximum of its
//     row contributes much to the product when the corresponding element of the other operand is large
//     (with K=64, every row A=(1, 2^-127, 0, ...) and every column B=(0, 2^127, 0, ...)^T, both maxima
//     are inside the range of the scale-computation stage and the true value is 1 in every element,
//     but the second element of A is not retained and the result is 0). This is a limitation of the
//     accuracy of the method that aligns the digits with a scale per row/column; it is separate from
//     the range check of the scale-computation stage in the item above (elements of 2^128 or more,
//     rows/columns whose largest nonzero absolute value is below 2^-126), and no stage detects it.
//   - When the output would be an FP64 subnormal (|C| < 2^-1022), it is flushed to +-0. Only when the
//     input is all zero does the result carry no sign and +0 is returned.
//   - The exponent arguments (d of split, g of assemble_*) are correct only in the range that the
//     ordinary normalization passes. elsb = E - 1075 + d and qw + g + ... are computed in int32, so
//     passing wildly out-of-range values causes signed overflow. Likewise the FP32 of a limb must be
//     finite (ee != 255), and the final scale W must be a normalized significand with its top bit set
//     (W >= 2^63). The result is not guaranteed on violation (even W = 0 does not give zero and goes
//     through the same path as a nonzero product). All of these are conditions the caller has to
//     satisfy and are not checked here (on the device this path is taken once per element, so no
//     branches are placed).

#include <cstdint>

// Size [bytes] of one FP64 tile (1024 elements x 8 B). Used both by the work-area CB of the readers and
// writers (created by the host) and by the range the kernel walks by address inside it (writing 8192
// directly on both sides would allow only one of them to change).
constexpr uint32_t OZ_F64_TILE_BYTES = 1024u * 8u;

// ===========================================================================
// Entry: the two words of an FP64 -> four bit patterns of a triple-float
// ===========================================================================
// The lower two limbs cannot be assembled without searching for the leading bit position (CLZ), so they
// are output with a "magic number" added, and only the subtraction of a constant and the multiplication
// are left to the SFPU (tf_load_f64 in slice_tf_core.h):
//   hi    FP32 as is. a * 2^(elsb+30) (elsb is the exponent of the least significant bit of the
//         significand; split_f64 below)
//   braw  2^23 + 2^22 + b   (the SFPU subtracts 12582912.0f)
//   craw  2^23 + c          (the SFPU subtracts  8388608.0f)
//   s     +-2^(elsb+7)      (the SFPU multiplies the two above by it; lo additionally by 2^-7)
//
// The exponent of s is placed at the position of mid, 2^(elsb+7), instead of 2^elsb so that s drops to
// 0 only for elsb <= -134, i.e. 82 binades below a normalized value of 1 (for scheme I, roughly below
// the row/column maximum), with a margin from the verified scope of application (72 binades). With
// 2^elsb that point would be 75 binades below, right next to the scope, and s alone would drop first,
// taking both mid and lo with it. The 7 is a margin, not an alignment: mid itself stays a normalized
// FP32 much further down (a mid with the largest significand, |b| ~ 2^22, down to elsb = -155). As a
// side effect mid needs no multiplication by 128 on the SFPU (only lo is scaled by 2^-7).
//
// d is the exponent shift of the normalization (the value is multiplied by 2^d). Being a power of two
// it is error-free and only has to be added as an integer to the exponent field of the FP64. There is
// no need to multiply by an FP32 reciprocal tile, and the constraint that the scale has to fit in the
// exponent range of FP32 disappears as well.
inline void split_f64(uint32_t w1, uint32_t w0, int32_t d,
                      uint32_t& hi, uint32_t& braw, uint32_t& craw, uint32_t& s) {
    const uint32_t sgn = w1 & 0x80000000u;
    const int32_t  E   = (int32_t)((w1 >> 20) & 0x7FFu);
    if (E == 0) {                       // zero and subnormals are flushed to 0
        hi = 0u; braw = 0x4B000000u; craw = 0x4B000000u; s = 0u; return;
    }
    // Split the significand S = 2^52 + (w1&0xFFFFF)*2^32 + w0 (53 bits) into 30/23/7 bits.
    uint32_t a = 0x400000u | ((w1 & 0x000FFFFFu) << 2) | (w0 >> 30);
    const uint32_t rem = w0 & 0x3FFFFFFFu;
    int32_t r;
    if (rem >= 0x20000000u) { a += 1u; r = (int32_t)rem - 0x40000000; }   // to nearest
    else                    {          r = (int32_t)rem; }
    const int32_t  b = r >> 7;                       // arithmetic shift, [-2^22, 2^22-1]
    const uint32_t c = (uint32_t)r & 0x7Fu;          // [0,127]
    const int32_t  elsb = E - 1075 + d;              // exponent of the significand LSB (not a slice count)
    int32_t eh; uint32_t mh;
    if (a == 0x800000u) { eh = elsb + 180; mh = 0u; }   // rounding up reached 2^23
    else                { eh = elsb + 179; mh = (a << 1) & 0x7FFFFFu; }
    const int32_t es = elsb + 134;                   // see "exponent of s" above
    hi = (eh > 0 && eh < 255) ? (sgn | ((uint32_t)eh << 23) | mh) : 0u;
    s  = (es > 0 && es < 255) ? (sgn | ((uint32_t)es << 23)) : 0u;
    braw = 0x4B000000u | (((uint32_t)(b + 0x400000)) & 0x7FFFFFu);
    craw = 0x4B000000u | c;
}
// Constant that the SFPU multiplies onto the s above (brings lo back to the 2^elsb position; mid uses
// s as is)
constexpr float OZ_TF_LO_SCALE = 0.0078125f;   // 2^-7 (a power of two, so exact in FP32)

// ===========================================================================
// Common part of the exits
// ===========================================================================
// Decomposes an FP32 into (signed significand, exponent). Subnormals are flushed to 0.
inline void dec32(uint32_t u, int32_t& e, int32_t& sig) {
    const int32_t ee = (int32_t)((u >> 23) & 0xFFu);
    if (ee == 0) { e = 0; sig = 0; return; }
    e = ee;
    const int32_t s = (int32_t)(0x800000u | (u & 0x7FFFFFu));
    sig = (u & 0x80000000u) ? -s : s;
}

// A is a significand normalized to [2^63, 2^64), and W = (Whi:Wlo) is the 64-bit significand of the
// final scale (the part that is not a power of two: radix^-(L+2) for scheme I, KM for scheme II).
// The value is +-A * W * 2^total, and it is rounded to FP64 (see "What is guaranteed" above).
// A and W are both in [2^63,2^64), so the product lies in [2^126,2^128) and the leading bit of the
// upper word hi can only be at position 62 or 63. The number of bits to drop can therefore be fixed to
// one of two choices, 10 or 11, which removes the expansion of variable 64-bit shifts that is expensive
// on RV32.
inline uint64_t pack_f64(uint64_t A, uint32_t Whi, uint32_t Wlo,
                         int32_t total, uint64_t sgn) {
    // A * W: four 32-bit multiplications (RV32 has no 64-bit multiplication)
    const uint32_t a1 = (uint32_t)(A >> 32), a0 = (uint32_t)A;
    const uint64_t p00 = (uint64_t)a0 * Wlo, p01 = (uint64_t)a0 * Whi;
    const uint64_t p10 = (uint64_t)a1 * Wlo, p11 = (uint64_t)a1 * Whi;
    const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    const uint64_t hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    const int32_t n = (hi >> 63) ? 63 : 62;
    int32_t exp2 = n + 64 + total;
    // The lower 64 bits that are dropped are not examined. Ties are sent away from zero (see "What is
    // guaranteed" above), so whether to round up is decided by the single guard bit (the top bit of the
    // dropped part).
    uint64_t mant; bool up;
    if (n == 63) { mant = hi >> 11; up = (hi & 0x400ull) != 0; }
    else         { mant = hi >> 10; up = (hi & 0x200ull) != 0; }
    if (up) {
        mant += 1ull;
        if (mant >> 53) { mant >>= 1; exp2 += 1; }
    }
    const int32_t ebias = exp2 + 1023;
    if (ebias <= 0)    return sgn;                            // subnormals are flushed to +-0
    if (ebias >= 2047) return sgn | 0x7FF0000000000000ull;    // overflow -> +-inf
    return sgn | ((uint64_t)ebias << 52) | (mant & 0x000FFFFFFFFFFFFFull);
}

// ===========================================================================
// Exit (scheme I): the three limbs of a triple-float -> FP64
// ===========================================================================
// Returns sig * 2^sh. Returns 0 when the term falls completely below the window (the caller looks at
// this 0 to identify a limb that has a value but did not contribute). The left shift is done on the
// unsigned (two's complement) bit pattern and converted back to int64_t, which yields sig * 2^sh
// modulo 2^64 for either sign (a left shift of a negative signed integer would be undefined).
// Truncation is toward 0 regardless of the sign: an arithmetic right shift (toward -inf) would round up
// the absolute value of negative terms only, so the behavior would change with the sign.
inline int64_t acc64_term(int32_t sig, int32_t sh) {
    if (!sig) return 0;
    if (sh >= 0)   return (int64_t)((uint64_t)(int64_t)sig << sh);
    if (sh <= -32) return 0;
    const uint32_t n = (uint32_t)(-sh);
    const int32_t  m = (sig < 0) ? -sig : sig;     // |sig| < 2^24, so the negation is safe
    const int64_t  q = (int64_t)(m >> n);
    return (sig < 0) ? -q : q;
}

// The largest exponent of the three limbs is taken as the base, and the limbs are collected shifted 37
// bits to the left of it.
// 37 is chosen to prevent overflow: the output of the reconstruction is not necessarily a canonical
// triple (|mid| <= ulp(hi)/2), and the middle limb can come to the same position as the high limb. With
// a shift of 39 the sum of the three limbs can exceed int64 and flip the sign (observed on real data).
// With 37 every term is below 2^61, so adding three of them is safe, and 8 bits of margin still remain
// below the rounding position.
//
// The bottom of the window is 60 bits below the leading bit of the base limb, and the bits below it are
// lost (sh<=-32 in acc64_term is not the exponent difference of the original limbs but the shift amount
// inside this window).
// When the upper two limbs almost cancel, the difference between the returned value and the true value
// is "the magnitude of the lost bits", which can greatly exceed 1 ulp with respect to the value after
// cancellation (an example is given in "What is guaranteed" above). With respect to the original high
// limb it is less than a factor 2^-58 (less than one unit of the window bottom for each of the three
// limbs), so the norm is unaffected as long as the output as a whole does not become small through
// cancellation. That condition is as written in "What is guaranteed, and what is not" at the top, and
// it holds for the evaluated inputs.
// Only the case where everything is lost and acc becomes 0 is picked up by the re-collection below.
inline uint64_t assemble_tf_f64(uint32_t hb, uint32_t mb, uint32_t lb, int32_t g,
                                uint32_t Whi, uint32_t Wlo, int32_t qw) {
    int32_t eh, em, el, gh, gm, gl;               // g* are the signed significands
    dec32(hb, eh, gh); dec32(mb, em, gm); dec32(lb, el, gl);
    if (gh == 0 && gm == 0 && gl == 0) return 0ull;
    int32_t ebase = eh;
    if (em > ebase) ebase = em;
    if (el > ebase) ebase = el;
    const int64_t ch = acc64_term(gh, eh - ebase + 37);
    const int64_t cm = acc64_term(gm, em - ebase + 37);
    const int64_t cl = acc64_term(gl, el - ebase + 37);
    int64_t acc = ch + cm + cl;
    // The limbs that stayed in the window cancel to 0, and there is a limb that fell below the window.
    // Returning 0 as is would give 0 although the true value is exactly the dropped limb
    // (example: hi=1, mid=-1, lo=2^-70). The contribution of the cancelled side is 0, so the limbs that
    // stayed in the window are cleared, and the base is taken again and the collection is redone with
    // the rest only.
    if (acc == 0) {
        if (ch) { eh = 0; gh = 0; }              // the limbs that contributed cancel completely
        if (cm) { em = 0; gm = 0; }
        if (cl) { el = 0; gl = 0; }
        if (gh == 0 && gm == 0 && gl == 0) return 0ull;   // truly 0
        ebase = eh; if (em > ebase) ebase = em; if (el > ebase) ebase = el;
        acc = acc64_term(gh, eh - ebase + 37)
            + acc64_term(gm, em - ebase + 37)
            + acc64_term(gl, el - ebase + 37);
    }
    if (acc == 0) return 0ull;
    const uint64_t sgn = (acc < 0) ? 0x8000000000000000ull : 0ull;
    uint64_t A = (acc < 0) ? (uint64_t)(-acc) : (uint64_t)acc;
    // Normalize to 64 bits. The RISC-V of the device has the Zbb clz, so the number of leading zeros is
    // counted in one instruction instead of a loop that shifts one bit at a time (the result is
    // bit-identical). acc != 0 was checked just above, so the undefined domain of clz is not entered.
    const int32_t shA = __builtin_clzll(A); A <<= shA;
    return pack_f64(A, Whi, Wlo, qw + g + (ebase - 187) - shA, sgn);
}

// ===========================================================================
// Exit (scheme II): three limbs (one pass) or six limbs (two passes) of frac -> FP64
// ===========================================================================
// The limbs are collected into the 128-bit fixed-point number acc = frac * 2^124 (four words, w[0] is
// the least significant). For a limb with exponent e the shift amount is e-26; a limb at the upper end
// 2^-1 does not exceed 2^124, and a limb at the lower end 2^-97 does not lose a single bit.
struct A128 { uint32_t w[4]; };
// Size [bytes] of one tile of 128-bit fixed-point numbers (1024 elements x 16 B). It is the DRAM page
// that the first pass of the two-pass configuration of Ozaki scheme II writes and the final pass reads,
// and the L1 work area (ozaki2/kernels/recon_writer.cpp, recon_reader.cpp, and the host code that runs
// the reconstruction passes).
constexpr uint32_t OZ_A128_TILE_BYTES = 1024u * (uint32_t)sizeof(A128);
static_assert(sizeof(A128) == 16, "A128 is a 128-bit fixed-point number of four words");
inline void a128_add(A128& a, int32_t k, uint32_t v) {
    for (; k < 4 && v; ++k) { const uint32_t o = a.w[k]; a.w[k] = o + v; v = (a.w[k] < o); }
}
inline void a128_sub(A128& a, int32_t k, uint32_t v) {
    for (; k < 4 && v; ++k) { const uint32_t o = a.w[k]; a.w[k] = o - v; v = (o < v); }
}
inline void a128_term(A128& a, int32_t sig, int32_t sh) {   // acc += sig * 2^sh
    if (!sig || sh <= -24) return;               // the whole significand is below the fixed-point LSB
    const bool     neg = sig < 0;
    const uint64_t m   = (uint64_t)(neg ? -(int64_t)sig : (int64_t)sig);
    // For sh < 0 the part that sticks out below the LSB is dropped and the rest is added (w stays 0;
    // m < 2^24, so the upper word is 0 and its addition/subtraction does nothing).
    const uint64_t v = (sh < 0) ? (m >> (uint32_t)(-sh)) : (m << (sh & 31));
    const int32_t  w = (sh < 0) ? 0 : (sh >> 5);
    if (!v) return;
    if (neg) { a128_sub(a, w, (uint32_t)v); a128_sub(a, w + 1, (uint32_t)(v >> 32)); }
    else     { a128_add(a, w, (uint32_t)v); a128_add(a, w + 1, (uint32_t)(v >> 32)); }
}

// Adds nl limbs to the 128-bit fixed-point number a (acc = value * 2^124). Each term drops what is below
// the LSB, so the result is the same regardless of the order of the additions and of adding the limbs
// in several portions (in the two-pass configuration the first pass adds its own three limbs to the
// fixed-point number first and the second pass adds the remaining three; this is bit-identical to
// adding all six at once).
inline void a128_add_limbs(A128& a, const uint32_t* limb, uint32_t nl) {
    for (uint32_t i = 0; i < nl; i++) {
        int32_t e, sig; dec32(limb[i], e, sig);
        a128_term(a, sig, e - 26);
    }
}

inline uint64_t a128_frac_to_f64(A128 a, int32_t g, uint32_t Whi, uint32_t Wlo, int32_t qw);

inline uint64_t assemble_frac_f64(const uint32_t* limb, uint32_t nl,
                                  int32_t g, uint32_t Whi, uint32_t Wlo, int32_t qw) {
    A128 a{{0, 0, 0, 0}};
    a128_add_limbs(a, limb, nl);
    return a128_frac_to_f64(a, g, Whi, Wlo, qw);
}

// Folds the collected fixed-point number a (frac * 2^124) and turns it into an FP64 multiplied by
// KM * 2^g
inline uint64_t a128_frac_to_f64(A128 a, int32_t g, uint32_t Whi, uint32_t Wlo, int32_t qw) {
    // By contract frac lies in the open interval (-1/2,1/2), and it is folded back into it exactly,
    // still in fixed point. The fold is always done here, in the one-pass and in the two-pass
    // configuration:
    //   - The compute kernel (ozaki2/kernels/recon_crt.cpp) only does a0 -= round(a0) for each modulus,
    //     so the value of the triple can stick out of [-1/2,1/2] by a1 + a2. Folding this in the FP32
    //     of the kernel (deciding by fl(a0+a1) and applying +-1 to a0) would miss an excess finer than
    //     about 2^-25 (when a1 = -2^-30 comes on top of a0 = exactly -1/2, the output stays off by 1
    //     from the true frac 1/2 - 2^-30, and C differs by M*2^-2b*tau_A*tau_B), and a0 - 1 for a0 in
    //     (1/4,1/2) is not exact in FP32 either (frac is rounded by 2^-25). The margin guaranteed by the
    //     window condition (OZ2_WIN_EPS in ozaki2/moduli.hpp) is only 1/2 - |frac| >= about 2^-31.4, so
    //     both can happen for elements at the edge of the window.
    //     The 128-bit fixed point (LSB 2^-124) is far finer than the margin, and adding or subtracting
    //     1 is exact as well.
    //   - In the two-pass configuration the sum of the two passes is folded together.
    // The sum of the input limbs lies in (-3/2, 3/2) (the value of each pass sticks out of [-1/2,1/2]
    // by at most about 2^-18), so adding or subtracting 1 once is enough. The rule is the same as
    // f -= roundq(f). roundq sends ties away from zero, so exactly +-1/2 are swapped (+1/2 -> -1/2,
    // -1/2 -> +1/2), but the window of the CRT is strict (|C'| < M/2), so frac can in principle never
    // be exactly +-1/2, and the handling of the endpoints only serves the agreement with a reference
    // computation that uses the same rule. Values inside (-1/2,1/2) are not changed.
    // 1.0 corresponds to 2^124, i.e. bit 28 of w[3].
    // This path is taken once per element, so ordinary elements (|value| < 1/2 and the upper word not
    // exactly -2^27) pass straight through with one unsigned comparison and one equality comparison,
    // and the decision to fold is made only on the rare side (top + 2^27 seen as unsigned is 2^28 or
    // more <=> top is outside [-2^27, 2^27). For top = -2^27 the value is at least -1/2 and below
    // -1/2 + 2^-28; only when the lower words are all 0 is it treated as exactly -1/2 and 1 is added).
    {
        const uint32_t top = a.w[3];
        if (top + 0x08000000u >= 0x10000000u || top == 0xF8000000u) {
            if ((int32_t)top >= 0x08000000) a128_sub(a, 3, 0x10000000u);
            else if ((int32_t)top < -0x08000000 || (a.w[0] | a.w[1] | a.w[2]) == 0u)
                a128_add(a, 3, 0x10000000u);
        }
    }
    const bool neg = (a.w[3] & 0x80000000u) != 0;
    if (neg) { for (int32_t i = 0; i < 4; i++) a.w[i] = ~a.w[i]; a128_add(a, 0, 1u); }
    if (!(a.w[0] | a.w[1] | a.w[2] | a.w[3])) return 0ull;
    // Number of leading zero bits. The scan over the words takes at most three iterations and is left
    // as it is; inside the word clz does the counting (the all-zero test above has been passed, so
    // a.w[3-wsh] != 0 is guaranteed).
    int32_t wsh = 0; while (a.w[3 - wsh] == 0) wsh++;
    const int32_t lz = 32 * wsh + __builtin_clz(a.w[3 - wsh]);
    uint64_t A;
    if (lz <= 64) {                                   // extract the upper 64 bits
        // k <= 2. If off != 0, 64-lz is not a multiple of 32, so k <= 1 and a.w[k+2] is in range.
        const int32_t k = (64 - lz) >> 5, off = (64 - lz) & 31;
        A = (((uint64_t)a.w[k + 1] << 32) | a.w[k]) >> off;
        if (off) A |= (uint64_t)a.w[k + 2] << (64 - off);
    } else {
        A = (((uint64_t)a.w[1] << 32) | a.w[0]) << (lz - 64);
    }
    // value = +-A * 2^(-lz-60)
    return pack_f64(A, Whi, Wlo, qw + g - lz - 60,
                    neg ? 0x8000000000000000ull : 0ull);
}
