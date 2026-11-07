// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: CRT reconstruction on the device (triple-float, FP64-equivalent)
// ---------------------------------------------------------------------------
// Role: receives the S per-modulus residues (c_0; for the layout see IN_RESIDUE below) and computes
//       the "fractional form" of the Chinese remainder theorem
//           frac = ( sum_i k_i / m_i ) mod 1,   k_i = (C_i mod m_i) * y_i mod m_i
//       as an FP32 triple-float (three-term expansion, about 68 effective bits), which is the output.
//       The final C = frac * KM * tau_A * tau_B (KM = M*2^-2b) is multiplied on the device by the
//       following recon_writer.cpp (the host only reorders).
//
// Why the fractional form
//   The plain CRT recovery x = sum c_i e_i becomes a huge integer close to M, which brings back the
//   multiple-precision integers that were to be avoided in the first place. In the fractional form,
//   with both sides divided by M, every intermediate value is at most about S.
//
// Why triple-float is needed
//   frac = C'/M is a small number of magnitude below 1, and its absolute precision directly
//   determines the attainable precision. With b=54, M is about 2^123 and the true product |C'| is
//   just below M/2 in the worst case, so frac is used up to 1/2, but a typical element is smaller in
//   magnitude than that. A pair of FP32 (double-single) has an absolute precision of only about
//   2^-48, which does not reach 53 bits.
//   Adding a third limb brings the absolute precision to about 2^-68, which reaches 2^-55 in relative
//   terms, i.e. FP64-equivalent.
//   The GPU implementation (GEMMul8) computes this part in FP64, but this hardware has no FP64 and
//   emulates it with multi-word FP32. This is the main reason why scheme II is bound by the non-GEMM
//   part on this hardware.
//
// Use of the DST registers (8 with fp32_dest_acc enabled)
//   0,1,2 = triple-float accumulator (a0,a1,a2)
//   3     = the value v to add
//   4,5,6 = scratch
//   7     = k_i (held throughout the computation of a term)
// How the term k_i*(1/m_i) is formed
//   The host decomposes 1/m_i beforehand into a sum of NRP FP32 fragments. 1/m itself has no finite
//   binary expansion, so this is an approximation. The fragments are 12-bit Dekker halves, so their
//   products with |k_i| <= floor(m/2) + 1 <= 2049 all fit in 24 bits and are exact in FP32. (k_i has
//   12 bits: the upper limit of the moduli is 4096, on the digit-32 paths (BF16, FP16, TF32, BFP8).
//   k_i is a representative of the residue class of r_i*y_i, not necessarily the symmetric residue:
//   the input of its ModRound can exceed 2^22, so |k_i| can exceed floor(m/2) by 1 (the bound
//   oz2_crt_k_bound in ozaki2/moduli.hpp, which the host also uses to count the small fragments in
//   oz2_choose_small_tail).) The term therefore splits
//   into "NRP exact FP32 scalars", which are added into the triple-float one at a time. Each addition
//   (tfAddS / tfAddSmall in sfpu_tf_proc.h) keeps the upper two limbs error-free and rounds only the
//   lowest limb, so the accumulation as a whole is not exact; that rounding is the error analyzed at
//   tfAddSmall and at the per-modulus reduction in kernel_main below.
//   NOTE: 12 + 12 = 24 is exactly the width of the FP32 significand; there is no slack. Raising the
//   upper limit of the moduli above 4096, or widening the fragments beyond 12 bits, puts a rounding
//   into this product and breaks the assumption. If either is changed, review OZ2_RGRAN and the
//   decomposition in ozaki2/moduli.hpp as well.
//   The exceptions are the last fragment of the one-pass configuration (rp0) and of the second pass
//   (rp2): only there, all 24 bits are used to gain width (the product itself is rounded, but that
//   position is already small enough).
#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "recon_args.h"
#include "digit_limits.h"
// Per-modulus procedures (k_i, adding the fragments, dropping the integer part; templates on the
// tile-operation wrapper, instantiated here with the device wrapper SfpuDev)
#include "recon_crt_proc.h"
using namespace recon_args;
#include "api/compute/eltwise_binary.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"

constexpr auto cb_in   = tt::CBIndex(CB_IN);     // input: C_i
constexpr auto cb_mask = tt::CBIndex(CB_MASK);   // input: the 2 masks that split the sign-magnitude value
constexpr auto cb_out  = tt::CBIndex(CB_LIMB);   // output: the limbs of frac

constexpr uint32_t S = get_compile_time_arg_val(C_S);     // number of moduli (the host derives it from K)
// Number of limbs of the reconstruction: 2 = FP32 class, 3 = FP64-equivalent
constexpr uint32_t NLIMB = get_compile_time_arg_val(C_NLIMB);
// The input has two layouts depending on the path. IN_RESIDUE=0 receives C_i as one tile of
// sign-magnitude int32 written by the GEMM stage, and taking the modulus is also done here (INT8
// path). IN_RESIDUE=1 receives one FP32 tile (digit-split paths; neither the conversion nor the
// high/low split is needed). Its content is normally the symmetric residue that the composition
// stage (compose.cpp) reduced modulo m, but a modulus whose RAWMASK bit is set (a one-digit modulus)
// carries the GEMM output, which does not pass through the composition stage, and it is reduced here
// (RAWMASK below).
constexpr uint32_t IN_RESIDUE = get_compile_time_arg_val(C_IN_RESIDUE);
// Number of reciprocal fragments. The default reconstruction finishes in one pass (NRP=OZ2_NRP0).
// Only when the host chooses two passes (with the automatic FP64-equivalent precision, only when the
// estimated error of one pass is large; oz2_recon1_log2 in ozaki2/auto_precision.hpp) is the work
// split into two passes, running this same kernel twice with only the constants changed.
//   Pass 1 (NRP=OZ2_NRP1): fragments of R_i, which is 1/m_i rounded to the common granularity 2^-64.
//     k_i is an integer, so k_i*R_i and their sum are integer multiples of the same granularity, and
//     the accumulation is exact as long as it fits in the triple-float.
//   Pass 2 (NRP=OZ2_NRP2): fragments of the remainder of the rounding (either sign). The value itself
//     is at most 2^-64, so the absolute error is negligible even with the same triple-float.
// The split into two is needed because DST holds only 8 FP32 registers (confirmed by measurement),
// so a four-term accumulator (which needs 9) does not fit. Two triple-floats reach the same
// precision.
// The two accumulations must not be mixed: adding a term of pass 2 into the (large) triple-float of
// pass 1 incurs an error of ULP(a2) per addition and loses the exactness. They are accumulated
// separately and combined at the end (the combination is done by recon_writer.cpp with the helpers
// of common/kernels/f64_tf.h, and
// the multiplication by KM is done in the same place. In the one-pass configuration, by default
// recon_reader.cpp takes over the first jsplit elements (512 by default). The two-pass configuration
// also shares the work in both passes; the first pass adds its own 3 limbs up to 128-bit fixed point
// and leaves that).
constexpr uint32_t NRP = get_compile_time_arg_val(C_NRP);
// How many of the trailing fragments are added with tfAddSmall (the cheap addition that skips the
// first stage). The host counts them from the magnitudes of the fragments (for the condition and the
// error analysis see tfAddSmall in sfpu_tf_proc.h).
constexpr uint32_t NSMALL_FRAG = get_compile_time_arg_val(C_NSMALL_FRAG);
// CPROBE: diagnostic selector of the kernel (the library always passes 0 = Normal). A nonzero value
// removes work and breaks the answer (the CrtMode table below: 1 keeps only the input load, 2 removes
// that as well).
constexpr uint32_t CPROBE = get_compile_time_arg_val(C_PROBE);
// RAWMASK: if the bit of modulus i is set, the residue plane holds a GEMM output P that did not pass
// through the composition stage (a one-digit modulus of the digit-32 AWE path; a single digit
// product, so |P| <= K*R^2; the moduli selection of the host (ozaki2_choose_moduli) imposes
// |P| + m <= 2^24). It is turned into a representative of its residue class here with one ModRound: P
// and the product of the quotient and the modulus are integers of at most 2^24, so they are exact in
// FP32 and the residue class is preserved, and the result satisfies |r| <= m/2 + |P|*2^-23 <= m/2 + 2
// (m <= 65; confirmed by an exhaustive check). The following r*y_i ((m/2+2)(m-1) < 2^22) is inside the
// general guarantee of ModRound, so k_i is the symmetric residue for these moduli.
// The other paths pass 0 (the branch disappears at compile time). On the host side see
// recon_make_kernels in ozaki2/stages.hpp.
constexpr uint32_t RAWMASK = get_compile_time_arg_val(C_RAWMASK);
constexpr uint32_t MAXS = OZ2_MAXS;     // upper limits shared with the host (digit_limits.h)
constexpr uint32_t MAXRP = OZ2_MAXRP;
static_assert(S <= MAXS, "the number of moduli exceeds the size of the fixed arrays");
static_assert(NRP <= MAXRP, "the number of reciprocal fragments exceeds the size of the fixed arrays");
constexpr uint32_t I32 = (uint32_t)DataFormat::Int32, F32 = (uint32_t)DataFormat::Float32;
// All per-modulus constants are passed as runtime arguments (bit patterns of float). The layout is
// defined only by crt_* in ozaki2/kernels/recon_args.h; the host (recon_set_args in
// ozaki2/stages.hpp) and load_crt_constants below build the indices with the same functions.
// The final multiplication by KM and the multiplication that undoes the normalization scales are
// done by recon_writer.cpp.

// o = v mod m (ModRound of sfpu_tf_proc.h: the symmetric residue for |v| < 2^22, a representative of
// the residue class with an excess below 1 for larger |v|). This kernel takes the modulus of the
// following three kinds of values.
//   the upper 15 bits and the lower 16 bits of |C_i| (both below 2^16; INT8 path only)
//   |C_i| mod m with the sign restored (below m^2/2 + m/2; also INT8 path only (m <= 255), so below
//   2^15)
//   r_i * y_i (all paths; below m^2/2 for a composed or INT8 residue |r_i| <= m/2, and below
//   (m/2+2)(m-1) < 2^12 for a RAWMASK modulus, m <= 65)
// The last one, r_i * y_i, exceeds the |d| < 2^22 that ModRound guarantees in general on the digit-32
// paths (BF16, FP16, TF32, BFP8), where the moduli extend up to 4096 (m^2/2 = 2^23). Since
// |d| < 2^23, the product of the quotient and the modulus is at most 2^24 and the residue class is
// preserved, and the excess over the symmetric residue is |d|*2^-23 < 1 (see the note on ModRound in
// sfpu_tf_proc.h). For all selectable moduli up to the upper limit of the path and all coefficients
// y (if the set is changed, y_i can become any value in 1..m_i-1 that is coprime to m_i),
// |k| <= floor(m/2)+1 (oz2_crt_k_bound in ozaki2/moduli.hpp) has been confirmed by an exhaustive
// check (for odd moduli there are combinations that exceed floor(m/2) by 1). This is the bound that
// keeps the products with the 12-bit fragments exact (see the top of this file).
// The decoding of the two planes (the first two above) is not executed on the digit-split paths (the
// residue is received in FP32; IN_RESIDUE=1).
// The form of ModRound that takes an output register is used; it writes to another register and
// keeps the input.
// The representative is not wrapped into [0,m); the signed value is used as is. The CRT looks only
// at the fractional part of the sum of k_i/m_i, so shifting k_i by an integer multiple of m_i shifts
// the sum by almost an integer, which vanishes when the fractional part is extracted.
// "Almost" because 1/m_i is approximated by r_i, a sum of 5 FP32 values, so the shift is not 1 but
// 1 + (m_i*r_i - 1). This residual is at most 2^-64.5 for every modulus of this implementation and
// at most 2^-60.7 summed over the S moduli; it is of the same order as the amount that each term
// contains anyway because r_i is not 1/m_i, so the attainable precision does not change without the
// wrap.
// --- Diagnostic modes (CPROBE; the library always uses Normal) ---------------
//   Mode               Input load   CRT coefficient   Fragment multiply   Accumulation
//   Normal                yes             yes                yes               yes
//   SkipCrt               yes             no                 no                no
//   TransportOnly         no              no                 no                no
// "Input load" means one copy_tile for the residue input and the whole sign-magnitude decoding for
// the int32 input. The processing below looks only at the flags DO_LOAD, DO_CRT and SEED_ACC.
enum class CrtMode : uint32_t {
    Normal = 0, SkipCrt = 1, TransportOnly = 2,
};
constexpr CrtMode MODE = (CrtMode)CPROBE;
static_assert(CPROBE <= 2, "CPROBE must be 0..2");
constexpr bool DO_LOAD   = (MODE != CrtMode::TransportOnly);
constexpr bool DO_CRT    = (MODE != CrtMode::SkipCrt && MODE != CrtMode::TransportOnly);
// The modes that skip work seed the accumulator with a nonzero constant so that the FP64 assembly of
// the writer (which skips a zero triple-float) still runs.
constexpr bool SEED_ACC  = (MODE != CrtMode::Normal);

// --- Per-modulus constants --------------------------------------------------
struct CrtConstants {
    float mod[MAXS], rinv[MAXS], yi[MAXS];   // m_i, 1/m_i, y_i = (M/m_i)^{-1} mod m_i
    float rp[MAXRP][MAXS];                   // 12-bit fragments of 1/m_i
    float m16[MAXS];                         // 2^16 mod m_i (used only by the sign-magnitude decoding)
    // Fragment t of modulus i (read by accumulate in recon_crt_proc.h)
    ALWI float frag(uint32_t t, uint32_t i) const { return rp[t][i]; }
};

ALWI void load_crt_constants(CrtConstants& c) {
    for (uint32_t i = 0; i < S; i++) {
        c.mod[i]  = fbf(get_arg_val<uint32_t>(crt_mod(S, i)));
        c.rinv[i] = fbf(get_arg_val<uint32_t>(crt_rinv(S, i)));
        c.yi[i]   = fbf(get_arg_val<uint32_t>(crt_yi(S, i)));
        for (uint32_t t = 0; t < NRP; t++)
            c.rp[t][i] = fbf(get_arg_val<uint32_t>(crt_frag(S, t, i)));
        if constexpr (!IN_RESIDUE)
            c.m16[i] = fbf(get_arg_val<uint32_t>(crt_m16(S, NRP, i)));
    }
}

// --- Build r_i = C_i mod m from one input tile and place it in dst::tmp0 ------
// Residue input: the symmetric residue reduced by the composition stage is used as is (|r_i| <= m/2,
// which is 2048 even at the upper limit 4096 of the moduli; it is exact in FP32. The final ModRound
// of compose.cpp receives a sum below 2^22, which the modulus selection of the host guarantees with
// compose_sum in ozaki2/moduli.hpp, so there the symmetric residue holds). For the GEMM output of a
// one-digit modulus, the caller looks at RAWMASK and applies ModRound.
ALWI void load_residue_fp32() {
    cb_wait_front(cb_in, 1);
    if constexpr (DO_LOAD) { copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, dst::tmp0); }
    cb_pop_front(cb_in, 1);
}

// int32 input: the sign-magnitude value written by the GEMM stage is split here. The magnitude has
// 31 bits and can exceed 2^24, the limit of exact integers in FP32, so it is split into the upper
// 15 bits (chi) and the lower 16 bits (clo) before the modulus is taken:
// |C_i| = chi*2^16 + clo, |C_i| mod m = (chi mod m)*(2^16 mod m) + (clo mod m).
// chi*2^16 is a 15-bit value multiplied by 2^16 and is therefore exact in FP32; clo < 2^16 is exact
// as well. The sign is multiplied back at the end.
ALWI void decode_signmag_residue(float m, float rinv, float m16) {
    using namespace dst;
    cb_wait_front(cb_in, 1);
    if constexpr (DO_LOAD) {
        copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, tmp2);
        typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(tmp2);
        sign_tile_init(); sign_tile(tmp2);                 // sgn = +1/-1/0 (kept in tmp2)
        copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, term);
        copy_tile_to_dst_init_short(cb_mask); copy_tile(cb_mask, 0, tmp1);
        binary_bitwise_tile_init();
        bitwise_and_binary_tile<DataFormat::Int32>(term, tmp1, term);   // raw & 0x7FFF0000
        typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(term);
        MulS(term, 1.52587890625e-05f);                    // /2^16 -> chi (exact: a power of two)
        ModRound(term, keep, tmp1, m, rinv);
        MulS(keep, m16);                                   // (chi mod m) * (2^16 mod m)
        copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, term);
        copy_tile_to_dst_init_short(cb_mask); copy_tile(cb_mask, 1, tmp1);
        binary_bitwise_tile_init();
        bitwise_and_binary_tile<DataFormat::Int32>(term, tmp1, term);   // raw & 0x0000FFFF
        typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(term);  // clo
        ModRound(term, tmp0, tmp1, m, rinv);
        Add(tmp0, keep, tmp0);      // exact: both terms are below m^2/2+m/2
        Mul(tmp0, tmp2, tmp0);      // multiply the sign back (|C_i| mod m with the sign attached)
        ModRound(tmp0, tmp1, m, rinv);                     // r_i = C_i mod m into tmp0
    }
    cb_pop_front(cb_in, 1);
}

// --- Initialization of the triple-float -------------------------------------
// The values are placed by writing constants, independent of the previous contents of DST (the
// register that holds r_i is not loaded in the diagnostic mode TransportOnly).
ALWI void clear_accumulator() {
    using namespace dst;
    if constexpr (SEED_ACC) { Set(acc_hi, 0.3f); Set(acc_mid, 1e-8f); Set(acc_lo, 1e-16f); }
    else                    { Set(acc_hi, 0.f);  Set(acc_mid, 0.f);   Set(acc_lo, 0.f); }
}


// --- The fold of the fraction is done on the writer side --------------------
// a0 -= round(a0) is applied after the accumulation of every modulus (kernel_main below), so after
// the last modulus a0 lies in [-1/2, 1/2] and the value of the triple-float leaves [-1/2,1/2] only by
// a1 + a2 (about 2^-18 at most). Folding this back (values in (-1/2,1/2) are unchanged; the excess is
// returned by +-1) is done by a128_frac_to_f64 on the writer side (recon_writer.cpp and
// recon_reader.cpp; defined in common/kernels/f64_tf.h; with one pass it is called from
// assemble_frac_f64, and in the final pass of two passes it is called directly after the own 3 limbs
// are added to the fixed-point value of the first pass), which adds or subtracts exactly 1 while the
// value is still in 128-bit fixed point.
// Folding here in FP32 (a0 -= 1 if t = fl(a0+a1) exceeds 1/2, a0 += 1 if it falls below -1/2) causes
// two kinds of errors (found with an FP32 simulation):
//   (1) Miss: when a0 = exactly -1/2 (which arises when a0 = 1.5 is rounded to 2 by the
//       round-to-nearest-even of Rnd) carries a1 = -2^-30, t = -1/2 and no correction is applied, so
//       the value leaves off by 1 from the true frac (1/2 - 2^-30). Unless the writer folds it, C
//       becomes an answer that differs by M*2^-2b*tau_A*tau_B.
//   (2) Rounding: when a0 is in (1/4, 1/2) and a1 pushes t above 1/2, a0 - 1 is not exact in FP32
//       (the ulp 2^-25 of a0 does not fit in the ulp 2^-24 of the result). frac is rounded to 2^-25
//       and the relative error of C becomes about 2^-24. This happens in both the one-pass and the
//       two-pass configuration.
// Both occur only for elements where 1/2 - |frac| is smaller than the magnitude of a1 (2^-25 to
// 2^-18). The margin guaranteed by the window condition (OZ2_WIN_EPS in ozaki2/moduli.hpp) is about
// 2^-31.4, so elements at the edge of the window (for example a constant input, whose elements all
// equal the row and column maxima) can reach it. The smallest margin of the default moduli sets is
// 1.5e-7 bit for BF16 (K=12096; 1/2 - |frac| >= about 2^-24; examined for K = 32..16384 in steps of
// 32).
// With random inputs frac is far from 1/2 and neither occurs.

ALWI void pack_fraction_limbs() {
    cb_reserve_back(cb_out, NLIMB);
    pack_reconfig_data_format(cb_out);
    pack_tile(dst::acc_hi,  cb_out);                 // high part of frac
    pack_tile(dst::acc_mid, cb_out);                 // middle part of frac (low part with 2 limbs)
    if constexpr (NLIMB >= 3) pack_tile(dst::acc_lo, cb_out);   // low part of frac
    cb_push_back(cb_out, NLIMB);
}

void kernel_main() {
    using namespace dst;
    SfpuDev w;   // device wrapper passed to the per-modulus procedures (recon_crt_proc.h)
    const uint32_t ntiles = get_arg_val<uint32_t>(crt_ntiles());
    CrtConstants c; load_crt_constants(c);
    // The 2 split masks are constants and stay resident in the CB (the reader sends them only once
    // before the loop)
    if constexpr (!IN_RESIDUE) cb_wait_front(cb_mask, 2);
    for (uint32_t ti = 0; ti < ntiles; ti++) {
        tile_regs_acquire();
        // The SFPU configuration is lost at every tile_regs cycle, so it is set for every tile
        // (hoisting it out of the loop breaks the answer).
        init_sfpu(cb_in, cb_out);
        for (uint32_t i = 0; i < S; i++) {           // add one term per modulus
            const float m = c.mod[i], rinv = c.rinv[i];
            if constexpr (IN_RESIDUE) load_residue_fp32();
            else                      decode_signmag_residue(m, rinv, c.m16[i]);
            if (i == 0) clear_accumulator();
            if constexpr (!DO_CRT) continue;
            // k_i into keep (a representative with |k_i| <= floor(m/2)+1; for a RAWMASK modulus, the
            // GEMM output of the one-digit modulus, which does not pass through the composition
            // stage, is first reduced modulo m)
            crt_proc::term_k(w, (RAWMASK & (1u << i)) != 0u, m, rinv, c.yi[i]);
            // The term k_i*(1/m_i) is added fragment by fragment. The fragments are NRP FP32 scalars
            // that stay exact when multiplied by k_i (except the last fragment of rp0/rp2, whose
            // product may round), so each one is added as a single FP32 value; the addition itself
            // rounds only the lowest limb of the triple-float. The trailing NSMALL_FRAG fragments
            // skip the first stage and are added cheaply (for the condition and the error see
            // tfAddSmall in sfpu_tf_proc.h)
            crt_proc::accumulate<NLIMB>(w, NRP, NSMALL_FRAG, c, i);
            // The integer part is dropped at every modulus to keep a0 at O(1). Taking mod 1 only
            // once at the end would let the sum grow up to S, and every addition in between drops an
            // error of one ULP of the lowest limb of the triple-float (proportional to a0). Reducing
            // step by step makes that ULP S times smaller. The absolute error of frac was measured
            // directly with a host model of these procedures (docs/sfpu_lemmas.md, Part B); the
            // per-modulus reduction lowers it by a few bits.
            crt_proc::drop_integer(w);
        }
        // The fold (image [-1/2,1/2]) is done by the writer in fixed point (see "The fold of the
        // fraction" above)
        tile_regs_commit();
        tile_regs_wait();
        pack_fraction_limbs();
        tile_regs_release();
    }
}
