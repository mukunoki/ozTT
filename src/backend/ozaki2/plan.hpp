// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: planning of one calculation (host side).
//
// Decides everything the stages need, at two points of the run (src/ozaki2.cpp):
// before anything is allocated
//   choose_moduli_and_width  moduli set, integer width b, number of reconstruction
//                            passes, planes of the composition stage
//   plan_panels              output panels and the core/block configuration of the GEMMs
// and once the FP64 operands are on the device
//   plan_bands               output row bands, from the DRAM that is then free
//   make_recon_plan          the reconstruction passes and their buffers
// Nothing here touches the device: the free DRAM and the core grid are passed in. Buffer
// sizes and kernel arguments are derived from the plan (stages.hpp, src/ozaki2.cpp), which
// reduces the risk that what is allocated and what the kernels are told disagree.
// Pipeline: scaling -> slicing (residues or digit planes per modulus) -> GEMMs
// (-> composition) -> CRT reconstruction to FP64.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <fmt/format.h>
#include <ozTTgemm/ozTTgemm.hpp>
#include "common/bits.hpp"
#include "ozaki2/format.hpp"
#include "ozaki2/auto_precision.hpp"
#include "common/mcast_gemm.hpp"
#include "common/kernels/f64_tf.h"
#include "ozaki2/kernels/recon_args.h"

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// Settings of one calculation, filled by the driver (run in src/ozaki2.cpp).
struct Oz2Knobs {
    // Requested number of reconstruction passes (Automatic: estimate from the input).
    ozTTgemm::Reconstruction passes = ozTTgemm::Reconstruction::Automatic;
    // Width b fixed by the caller (the evaluator only); otherwise b is chosen from the
    // input (auto_precision.hpp).
    std::optional<uint32_t> b;
    uint32_t max_moduli = 0;  // explicit limit of the caller on the number of moduli (0: none; at most OZ2_MAXS)
};

// The plan. The run state of the driver (Oz2Run) derives from it and adds the device
// resources.
struct Oz2Plan {
    // Dimensions in elements (already padded: M and N to multiples of 64, K of 32).
    uint32_t M = 0, N = 0, K = 0;
    uint32_t S_req = 0;  // limit on the number of moduli (choose_moduli_and_width)

    // Moduli and width (choose_moduli_and_width).
    bool two_pass = false;  // two-pass reconstruction
    Ozaki2Moduli mo;
    uint32_t b = 0, ND = 3, S = 0;  // width, number of radix-2^20 digits that carry A', number of moduli
    uint32_t Mt = 0, Nt = 0, Kt = 0, ntotA = 0, ntotB = 0, ntotC = 0;  // tiles per dimension and per matrix

    // Planes and chunking of the composition stage (make_compose_plan; all zero for INT8).
    Oz2ComposePlan cplan;

    // GEMM stage (plan_panels): configuration of one panel, panel size in tiles, panel counts.
    OptConfig g;
    uint32_t pMt = 1, pNt = 1, nbr = 1, nbc = 1;
    // Output row bands (plan_bands): one band is bpr panel rows (bMt tile rows), nband = nbr / bpr.
    // The A slice planes, product planes/residues and reconstruction buffers hold one band;
    // per band the A rows are sliced, multiplied for every modulus, composed and
    // reconstructed. The B slice planes are made once. One band is the plain flow.
    uint32_t nband = 1, bpr = 1;
    uint32_t bMt() const { return bpr * pMt; }      // tile rows of one band
    uint32_t ntotAb() const { return bMt() * Kt; }  // tiles of one A slice plane of a band
    uint32_t ntotCb() const { return bMt() * Nt; }  // tiles of one output plane of a band
};

// log2 of the first scale of the slicer: b - (ND-1)*20. The slicer multiplies by
// 2^(b-(ND-1)*20-e) (the exponent deltas carry it) so that the integer part of the scaled
// value is the highest digit of A'; it then takes the remaining ND-1 digits by multiplying
// the residual by 2^20 each time (kernels/digits.h). The reconstruction multiplies back
// twice this shift (once for A, once for B).
static int32_t oz2_shift(const Oz2Plan& r) { return (int32_t)r.b - (int32_t)(r.ND - 1) * (int32_t)OZ2_DIGIT_BITS; }

// ===== Planes of the composition stage (digit formats) =====
// How many product planes are kept and how many are composed at once. Counted are the
// GEMM outputs per modulus after concatenation (oz2_product_groups). One-digit moduli
// (OZ2_SPLIT_ONE) bypass the composition, their GEMM output goes straight to the residue
// plane; if no modulus is composed, there are no product planes and no composition stage
// (all zero). The rule itself is oz2_compose_plan in split.hpp, shared with the DRAM
// estimate of the stage-time model.
static Oz2ComposePlan make_compose_plan(const Ozaki2Moduli& mo) {
    if constexpr (!OZ2_DIGIT) return {};
    uint32_t mx = 0;
    bool any = false;
    for (int i = 0; i < mo.s; i++) {
        if (mo.mode[i] == OZ2_SPLIT_ONE) continue;
        any = true;
        mx = std::max(mx, oz2_nout(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[i]));
    }
    return oz2_compose_plan(any, mx);
}

// Smallest depth of the input circular buffer of the composition stage, and the largest
// depth for which a single composition program is built. Two-digit formats have groups of
// 3 and 4 planes, hence at least 12 tiles. When the common multiple exceeds the maximum
// (balanced 3, 4 and 5 digits mixed: 5, 7 and 9 outputs, common multiple 315), one program
// is built per number of planes issued at once (compose_for in src/ozaki2.cpp).
constexpr uint32_t OZ2_COMPOSE_CB_MIN = 12;
constexpr uint32_t OZ2_COMPOSE_CB_MAX = 64;

// Depth [tiles] of the input circular buffer of the composition stage. It must be a common
// multiple of every number of planes that flows through per launch: a group that wraps
// around the ring boundary is corrupted. The lengths actually issued are, per modulus,
//   without chunking  the number of outputs np
//   with chunking     the chunk CH, the last chunk np % CH (if nonzero) and the number of
//                     partial sums ceil(np/CH) of the final pass
static inline uint32_t compose_cb_depth(const Ozaki2Moduli& mo, uint32_t CH) {
    uint32_t np_lcm = 1;
    auto take_lcm = [&np_lcm](uint32_t v) {
        if (v) np_lcm = (uint32_t)std::lcm(np_lcm, v);
    };
    for (int i = 0; i < mo.s; i++) {
        const uint32_t v = oz2_nout(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[i]);
        if (CH < v) {
            take_lcm(CH);
            take_lcm(v % CH);
            take_lcm((v + CH - 1) / CH);
        } else
            take_lcm(v);
    }
    // Raise the depth to the minimum in multiples of np_lcm, so that it stays a common multiple.
    if (np_lcm < OZ2_COMPOSE_CB_MIN) np_lcm *= (OZ2_COMPOSE_CB_MIN + np_lcm - 1) / np_lcm;
    return np_lcm;
}

// ===== Moduli and width (host only) =====

// Arguments of the moduli selection; only the number of moduli varies between calls (the
// error messages reselect with other limits to find out what would help). M and N feed the
// DRAM estimate of the stage-time model.
struct Oz2ModuliRequest {
    uint32_t M, N, K;
    int b_cap;
    Ozaki2Moduli choose(uint32_t S) const { return ozaki2_choose_moduli(M, N, K, (int)S, b_cap); }
};

// Why the product of the moduli is too small for the width `target`, and what the caller
// can change: the modulus cap of INT8, exhausted candidates, or a limit on the number of
// moduli that is too small (found by actually reselecting with larger limits).
static std::string oz2_short_cause(const Oz2Plan& r, const Ozaki2Moduli& mo, const Oz2ModuliRequest& rq, int target) {
    const bool int8 = OZ2_PARAM.digit_bound == 0;
    std::string s;
    if (int8 && mo.m_cap < OZ2_PARAM.mod_cap)
        s = fmt::format(
            "For this K the INT8 moduli are limited to {} or less, because the GEMM output bound "
            "K*floor(m/2)^2 must not exceed 2^{}-1. ",
            mo.m_cap, OZ2_ACC_BITS);
    const uint32_t want = std::min<uint32_t>(r.S_req, OZ2_MAXS);
    // For a set chosen by the stage-time model ncand is the size of that set, not a count
    // of candidates, so only the table selection can report exhausted candidates.
    if (!mo.model_set && (uint32_t)mo.ncand < want)
        return s + fmt::format(
                       "Every candidate modulus is in use ({}), so a larger Options::max_moduli does not help; "
                       "use a smaller K or a different format.",
                       mo.ncand);
    if (want >= OZ2_MAXS)
        return s +
               fmt::format("The maximum of {} moduli is not enough; use a smaller K or a different format.", OZ2_MAXS);
    for (uint32_t n = r.S_req + 1; n <= (uint32_t)OZ2_MAXS; n++) {
        const Ozaki2Moduli t = rq.choose(n);
        if (t.ok && t.b >= target)
            return s + fmt::format("The limit of {} moduli is too small; set Options::max_moduli to {} or more.",
                                   r.S_req, n);
    }
    const Ozaki2Moduli t = rq.choose(OZ2_MAXS);
    return s + fmt::format(
                   "The limit of {} moduli is too small, and the maximum of {} moduli still gives only {} bits; "
                   "use a smaller K or a different format.",
                   r.S_req, OZ2_MAXS, t.ok ? t.b : 0);
}

// No moduli set satisfies the accumulation conditions (mo.acc_ok is false): throw with the
// first condition that failed (mo.acc_fails, see moduli.hpp). These limits come from K and
// the exact integer range of the accumulation, not from the number of moduli.
[[noreturn]] static inline void oz2_report_accumulation_failure(const Oz2Plan& r, const Ozaki2Moduli& mo) {
    std::string cause = "an exactness condition of the accumulation does not hold";
    if (!mo.acc_fails.empty()) {
        const Ozaki2Moduli::AccFail& f = mo.acc_fails.front();
        if (f.why == "int8")
            cause = fmt::format(
                "the GEMM output bound K*floor(m/2)^2 = {:.0f} of modulus {} exceeds the sign-magnitude "
                "int32 range 2^{}-1",
                f.val, f.m, OZ2_ACC_BITS);
        else if (f.why == "digit")
            cause = fmt::format(
                "the digit-product bound K*d^2 = {:.0f} (digit bound d={}) exceeds 2^24, the limit of "
                "exact FP32 accumulation",
                f.val, f.d);
        else if (f.why == "one")
            cause = fmt::format(
                "the GEMM output bound K*R^2 + m = {:.0f} of the one-digit modulus {} (residue bound R={}) "
                "exceeds 2^24, so the reconstruction cannot reduce it modulo m exactly",
                f.val, f.m, f.d);
        else
            cause = fmt::format(
                "the digit products of modulus {} cannot be composed exactly (the bound {:.0f} of the "
                "composition sum reaches 2^22)",
                f.m, f.val);
    }
    throw std::runtime_error(
        fmt::format("Ozaki II: the padded inner dimension K={} is too large for this format: {}. "
                    "Use a smaller K or a different format; Options::max_moduli does not change this limit.",
                    r.K, cause));
}

// No usable moduli set (mo.ok is false): throw with the cause and the remedy.
[[noreturn]] static inline void oz2_report_moduli_failure(const Oz2Plan& r, const Ozaki2Moduli& mo,
                                                          const Oz2ModuliRequest& rq) {
    if (!mo.acc_ok) oz2_report_accumulation_failure(r, mo);
    throw std::runtime_error(
        fmt::format("Ozaki II: the available moduli do not cover the CRT window for the padded inner "
                    "dimension K={}. {}",
                    r.K, oz2_short_cause(r, mo, rq, 1)));
}

// The width that the window allows can be smaller than requested: the CRT window is set by
// the product of the moduli, and when even all candidates do not reach 2b+1+log2(K) bits,
// b stops there. The relative error is proportional to 2^-b, so every missing bit is a bit
// of accuracy. A configuration whose b stays below FP64-equivalent (OZ2_TARGET_BITS) is
// rejected here, before any device work, unless the caller fixed the width itself
// (b_fixed: the evaluator's fixed_bits). The cap chosen from the input is never below
// OZ2_TARGET_BITS (choose_precision_policy), so the target is the floor.
static inline void oz2_check_width(const Oz2Plan& r, const Ozaki2Moduli& mo, const Oz2ModuliRequest& rq, bool b_fixed) {
    if (!b_fixed && mo.b < OZ2_TARGET_BITS)
        throw std::runtime_error(
            fmt::format("Ozaki II: for the padded inner dimension K={} the {} selected moduli cover a width "
                        "of only {} bits, below the {} bits of FP64-equivalent accuracy. {}",
                        r.K, mo.s, mo.b, OZ2_TARGET_BITS, oz2_short_cause(r, mo, rq, OZ2_TARGET_BITS)));
}

// Number of reconstruction passes: as requested, or (Automatic) one pass when the
// estimated error of the one-pass reconstruction is small enough (auto_precision.hpp; the
// estimate needs the input statistics in pol).
static inline void oz2_choose_passes(Oz2Plan& r, const Oz2PrecisionPolicy& pol, ozTTgemm::Reconstruction pq) {
    if (pq != ozTTgemm::Reconstruction::Automatic) {
        r.two_pass = pq == ozTTgemm::Reconstruction::TwoPass;
        return;
    }
    r.two_pass = !(oz2_recon1_log2(r.mo, (int)r.b, pol) <= OZ2_RECON1_THR_LOG2);
}

// Choose the moduli set and the width b, the number of reconstruction passes, the number
// of digits of A' and the planes of the composition stage. Host only; nothing is
// allocated. r.M, r.N and r.K must be set. A and B (row-major, padded) are read only when
// the width or the number of passes is chosen from the input.
// Throws std::runtime_error when the configuration cannot be executed exactly.
static inline void choose_moduli_and_width(Oz2Plan& r, const Oz2Knobs& k, const std::vector<double>& A,
                                           const std::vector<double>& B) {
    const uint32_t M = r.M, N = r.N, K = r.K;
    // The structural cap on b is OZ2_MAXDIGIT*OZ2_DIGIT_BITS = 100, the width that A' with
    // 5 radix-2^20 digits carries. The slicer uses 4 digits only for b > 60 and 5 only for
    // b > 80 (3 digits at the default b = 54).
    int b_cap = OZ2_B_DEFAULT;
    // A width fixed by the caller is passed to the moduli selection as the cap; otherwise
    // the cap is chosen from the input.
    const bool b_fixed = k.b.has_value();
    Oz2PrecisionPolicy pol;
    if (b_fixed) {
        b_cap = (int)*k.b;
    } else {
        pol = choose_precision_policy(A, B, M, N, K);
        b_cap = pol.b_cap;
    }
    // Limit on the number of moduli: the caller's, if given. Otherwise all OZ2_MAXS moduli
    // when the width is chosen from the input: a larger b needs a wider window
    // (2b+1+log2 K bits), and the moduli selection drops every modulus the window does not
    // need, so it settles at the smallest number that covers b_cap (surplus moduli would
    // widen the window and lower the accuracy). With a fixed width (the evaluator) the
    // default of the path applies.
    r.S_req = k.max_moduli ? k.max_moduli : (b_fixed ? OZ2_PARAM.def_s : OZ2_MAXS);
    // The estimate of the number of passes needs the input statistics; with a fixed width
    // they have not been measured yet.
    const ozTTgemm::Reconstruction pq = k.passes;
    if (b_fixed && pq == ozTTgemm::Reconstruction::Automatic) oz2_measure_stats(A, B, M, N, K, pol);

    // The CRT window is the worst case |C'| <= K*2^(2b) (ozaki2_choose_moduli).
    const Oz2ModuliRequest rq{M, N, K, b_cap};
    Ozaki2Moduli& mo = r.mo;
    mo = rq.choose(r.S_req);
    if (!mo.ok) oz2_report_moduli_failure(r, mo, rq);
    oz2_check_width(r, mo, rq, b_fixed);
    const uint32_t b = (uint32_t)mo.b;
    r.b = b;
    oz2_choose_passes(r, pol, pq);
    // Digits of A': 3 digits carry 60 bits and 4 carry 80, so more are used only above that.
    r.ND = (b > 4 * OZ2_DIGIT_BITS) ? 5u : ((b > 3 * OZ2_DIGIT_BITS) ? 4u : 3u);
    r.S = mo.s;
    r.Mt = M / 32;
    r.Nt = N / 32;
    r.Kt = K / 32;
    r.ntotA = r.Mt * r.Kt;
    r.ntotB = r.Kt * r.Nt;
    r.ntotC = r.Mt * r.Nt;
    r.cplan = make_compose_plan(mo);
}

// ===== Plan of the reconstruction stage =====
// The number of passes, the constants of each pass and what each pass writes are fixed
// here, before the buffers are allocated; the allocation sizes and the kernel arguments
// both follow from this plan.
struct ReconPass {
    std::vector<float> rp;  // reciprocal fragments, [i*nrp + t] (modulus i, fragment t)
    uint32_t nrp = 0;
    // Role of the pass (compile-time argument CT_ROLE, ReconRole in recon_args.h):
    // ROLE_SINGLE and ROLE_TWO_FINAL assemble FP64 and write the output (FINAL of the
    // kernels); ROLE_TWO_FIRST adds its limbs to a 128-bit fixed point and writes that,
    // which ROLE_TWO_FINAL reads and adds.
    recon_args::ReconRole role = recon_args::ROLE_SINGLE;
    bool limb3 = false;  // three limbs (FP64-equivalent, or any pass of the two-pass reconstruction)
};

struct ReconPlan {
    std::vector<ReconPass> pass;
    uint32_t a128_tiles = 1;  // pages (one per tile, OZ2_A128_PAGE bytes) of the 128-bit fixed-point buffer
};

// Bytes of one tile of 128-bit fixed-point values (1024 elements of 16 bytes).
constexpr uint32_t OZ2_A128_PAGE = OZ_A128_TILE_BYTES;

// Tiles of each of the three limb buffers. Every pass either assembles FP64 or packs the
// 128-bit fixed point, so the limb buffers are never written; one tile each is allocated
// only because the writer kernel needs buffers to build its accessors.
constexpr uint32_t OZ2_LIMB_TILES = 1;

// Pages of the 128-bit fixed-point buffer for a band of ntot output tiles: the buffer is
// written by the first pass of two; otherwise it is a one-tile placeholder as well.
static inline uint32_t oz2_a128_tiles(bool two_pass, uint32_t ntot) { return two_pass ? ntot : 1u; }

// ntot: tiles of one output plane (of one band).
static ReconPlan make_recon_plan(const Ozaki2Moduli& mo, bool two_pass, uint32_t ntot) {
    const uint32_t S = (uint32_t)mo.s;
    auto pack = [&](auto& src, uint32_t np) {
        std::vector<float> c((size_t)S * np);
        for (uint32_t i = 0; i < S; i++)
            for (uint32_t t = 0; t < np; t++) c[(size_t)i * np + t] = src[i][t];
        return c;
    };

    // The initializers list all four fields {rp, nrp, role, limb3}; one pass, or two.
    ReconPlan p;
    if (!two_pass) {
        p.pass.push_back({pack(mo.rp0, OZ2_NRP0), OZ2_NRP0, recon_args::ROLE_SINGLE, oz2_recon_limb3(mo, false)});
    } else {
        // The first pass adds its three limbs to a 128-bit fixed point kept in DRAM
        // (16 bytes per element); the second pass reads it, adds its own three limbs and
        // assembles FP64. This is bitwise the same as adding all six limbs at once, and it
        // moves half of the assembly work into the first pass. Both passes use three limbs
        // (the writer of the first pass always reads three; oz2_recon_limb3).
        p.pass.push_back({pack(mo.rp1, OZ2_NRP1), OZ2_NRP1, recon_args::ROLE_TWO_FIRST, oz2_recon_limb3(mo, true)});
        p.pass.push_back({pack(mo.rp2, OZ2_NRP2), OZ2_NRP2, recon_args::ROLE_TWO_FINAL, oz2_recon_limb3(mo, true)});
    }

    p.a128_tiles = oz2_a128_tiles(two_pass, ntot);
    return p;
}

// Final scale of the reconstruction (the same in every pass): KM = M*2^-2b as a 64-bit
// significand W (Whi upper, Wlo lower 32 bits) and an exponent qw, multiplied in integers
// (64x64 -> 128 bits), because a double constant would add a relative 2^-53 to every
// element; and sh2 = 2*oz2_shift, the exponent that undoes the first scale of the slicer
// for A and B.
struct ReconScale {
    uint32_t Whi = 0, Wlo = 0;
    int32_t qw = 0, sh2 = 0;
};
static ReconScale make_recon_scale(const Ozaki2Moduli& mo, int32_t sh2) {
    const OzScale64 w = oz_split_scale_ld(mo.KMl);
    return {w.hi, w.lo, w.q, sh2};
}

// Margin added to the estimate of the simultaneous allocation (1 GiB: the small buffers
// and what the estimate does not count).
constexpr uint64_t OZ2_WHOLE_DRAM_MARGIN = 1ull << 30;

// Output row bands. Called after plan_panels and after the FP64 operands are on the
// device; df is the free DRAM at that point. The number of bands is decided from the
// simultaneous allocation of one GEMM call in which nothing is released between stages
// (the set the evaluator replays in its whole-pass timing; an ordinary call uses the same
// bands and merely releases earlier):
//   A slice planes (one band) + B slice planes (whole) + product planes and residues
//   (cplan.planes + S planes of one band) + the FP64 output (whole) + limb and 128-bit
//   buffers of the reconstruction (OZ2_LIMB_TILES, oz2_a128_tiles) + OZ2_WHOLE_DRAM_MARGIN
// Bands divide the panel rows evenly (bpr divides nbr); the smallest number of bands that
// fits is chosen. Every buffer is placed at the same address in all DRAM banks, so besides
// the total the largest single buffer must fit the largest free block; that is a necessary
// condition only (earlier allocations may split the block further).
// Throws std::runtime_error when even one panel row per band does not fit.
static inline void plan_bands(Oz2Plan& r, const DramFree& df) {
    const Ozaki2Moduli& mo = r.mo;
    static_assert(
        OZ2_PARAM.awe == 0 || OZ2_AWE_SLICE_TILE[OZ2_PARAM.awe - 1] == tt::tile_size(OZ_FMT.slice_out),
        "the slice tile size of the stage-time model (OZ2_AWE_SLICE_TILE in awe.hpp) differs from that of the format");
    const uint64_t slt = tt::tile_size(OZ_FMT.slice_out), outt = tt::tile_size(OZ_FMT.out);
    const uint64_t f32t = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t cpl = r.cplan.planes;  // product planes of the composition stage (0 for INT8)
    const uint64_t o64B = (uint64_t)r.ntotC * OZ_F64_TILE_BYTES;
    struct Need {
        uint64_t a, b, q, limb, total, maxbuf;
    };  // bytes
    auto need = [&](uint32_t bpr) {
        // Slice planes of A and B: all moduli are sliced at once (the digit planes, or one
        // residue plane per modulus for INT8; mo.nplane counts both).
        const uint32_t mp = (uint32_t)mo.nplane;
        const uint64_t ntCb = (uint64_t)bpr * r.pMt * r.Nt;
        const uint64_t a128 = oz2_a128_tiles(r.two_pass, (uint32_t)ntCb);
        Need w;
        w.a = (uint64_t)mp * bpr * r.pMt * r.Kt * slt;
        w.b = (uint64_t)mp * r.ntotB * slt;
        w.q = (uint64_t)(cpl + r.S) * ntCb * outt;
        w.limb = (uint64_t)OZ2_LIMB_TILES * 3 * f32t + a128 * OZ2_A128_PAGE;
        w.total = w.a + w.b + w.q + o64B + w.limb + OZ2_WHOLE_DRAM_MARGIN;
        w.maxbuf = std::max({w.a, w.b, (uint64_t)cpl * ntCb * outt, (uint64_t)r.S * ntCb * outt, o64B,
                             (uint64_t)OZ2_LIMB_TILES * f32t, a128 * OZ2_A128_PAGE});
        return w;
    };
    auto fits = [&](const Need& w) { return w.total <= df.total && w.maxbuf <= df.largest_block; };
    // Largest divisor of the panel rows that fits (i.e. the fewest bands).
    uint32_t bpr = r.nbr;
    for (uint32_t k = r.nbr; k >= 1; k--)
        if (r.nbr % k == 0) {
            bpr = k;
            if (fits(need(k))) break;
        }
    const Need w = need(bpr);
    r.bpr = bpr;
    r.nband = r.nbr / bpr;
    if (!fits(w))
        throw std::runtime_error(
            fmt::format("Ozaki II: the buffers do not fit in device DRAM even with one panel row per output "
                        "band (needs {:.1f} GB, largest buffer {:.1f} GB; free {:.1f} GB, largest free "
                        "block {:.1f} GB). Use smaller matrix dimensions or a format with smaller elements.",
                        w.total / 1e9, w.maxbuf / 1e9, df.total / 1e9, df.largest_block / 1e9));
}

// ===== GEMM stage: output panels and the configuration of one GEMM =====
// The output is split into panels so that the output block of each core fits in L1. With
// panel sides of at most 120 tiles a core of the 11x10 grid holds at most 12x11 = 132
// output tiles. One panel is the plain GEMM of the whole matrix. grid is the compute grid
// of the device. Throws std::runtime_error when no configuration exists.
static inline void plan_panels(Oz2Plan& r, tt::tt_metal::CoreCoord grid) {
    // Try the largest panels first; fall back to smaller divisors if a largest
    // divisor cannot be mapped to the multicast grid (e.g. 158 tiles -> 79).
    // The L1 budget (oz_l1_budget) is counted in bytes with the tile sizes of the GEMM
    // input and output formats.
    const auto choice = choose_panels(r.Mt, r.Nt, 120, [&](uint32_t rows, uint32_t cols) {
        return choose_cfg(rows, cols, r.Kt, grid.y, grid.x, tt::tile_size(OZ_FMT.in), tt::tile_size(OZ_FMT.out),
                          oz_l1_budget());
    });
    if (!choice.g.ok)
        throw std::runtime_error(
            fmt::format("Ozaki II: no GEMM configuration fits the padded dimensions M={}, N={}, K={} on the "
                        "{}x{} core grid (rows x columns) within the L1 memory of a core.",
                        r.M, r.N, r.K, grid.y, grid.x));
    r.pMt = choice.pMt;
    r.pNt = choice.pNt;
    r.g = choice.g;
    r.nbr = r.Mt / r.pMt;
    r.nbc = r.Nt / r.pNt;
}

// Panels are addressed by tile offsets on the device. The row panels of A are contiguous,
// so the origin is br*pMt*Kt; for a column panel of B the row stride stays the whole Nt
// and the origin moves by bc*pNt (the GEMM reader takes the row strides as arguments).
// The A slice planes hold one band, so the plane stride is ntotAb and br is the in-band row panel.
static inline uint32_t panel_off_A(const Oz2Plan& r, uint32_t plane, uint32_t br) {
    return plane * r.ntotAb() + br * r.pMt * r.Kt;
}
static inline uint32_t panel_off_B(const Oz2Plan& r, uint32_t plane, uint32_t bc) {
    return plane * r.ntotB + bc * r.pNt;
}

// With concatenation the A slice of the digit formats is one matrix of Mt x (PA*Kt) tiles:
// the PA digit planes side by side along K (LAYOUT_A_CAT of kernels/modslice_writer.cpp).
// Offset of the first column of digit plane `plane` in row panel br. B is the digit planes
// stacked along K, (PA*Kt) x Nt tiles, so the row of position `plane` is given by
// panel_off_B.
static inline uint32_t panel_off_Acat(const Oz2Plan& r, uint32_t PA, uint32_t plane, uint32_t br) {
    return br * r.pMt * (PA * r.Kt) + plane * r.Kt;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
