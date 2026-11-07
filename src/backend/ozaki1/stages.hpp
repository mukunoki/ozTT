// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: resources of the device stages and the order in which their programs
// are issued.
//
// The pipeline is scaling -> slicing -> partial-product GEMMs (with accumulation) ->
// reconstruction. The functions here only enqueue programs; they neither wait for the
// device nor time anything. The driver (src/ozaki1.cpp) issues the stages through them
// and adds the synchronization, and the evaluator's uninterrupted whole pass issues the
// same steps through the same functions.
// Every enqueued program is reported to the callable `note` with its kind
// (evaluation::Step), which the evaluator uses to count the steps of the whole pass. The
// normal execution passes oz1_no_note.

#include "ozaki1/format.hpp"
#include "common/scale.hpp"
#include "ozaki1/gemm.hpp"
#include "ozaki1/recon.hpp"
#include "ozaki1/slice.hpp"
#include "ozaki1/plan.hpp"
#include "evaluation_hooks.hpp"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Device buffers of the input side.
struct InputBufs {
    //   rAf64, rBf64  the FP64 operands as uploaded (8 bytes per element, tile order). The
    //                 scaling and slicing stages read them directly; the conversion to
    //                 triple-float is done on the device.
    //   maxAb, maxBb  largest magnitude of each row of A / column of B (BF16)
    //   dltAb, dltBb  exponent deltas (Int32), broadcast per row / column. The slicing
    //                 stage adds them to the FP64 exponents (normalization) and the
    //                 reconstruction removes them again.
    //   flgA, flgB    out-of-range flags of the scaling stage, one tile per core
    //   shA, shB      slice-concatenated operands, the inputs of the GEMMs
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> rAf64, rBf64, maxAb, maxBb, dltAb, dltBb;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> flgA, flgB, shA, shB;
};

// Programs of the scaling and slicing stages. The exponent-delta contexts are kept until
// the result has been read, because the out-of-range flags are checked last.
struct FillStage {
    ExpDeltaCtx edA, edB;     // exponent deltas of A and B
    ScaleCtx scA, scB;        // row / column maxima of A and B
    SliceCtx slc;             // slicing program (shared by A and B)
    SliceRun sliceA, sliceB;  // its two runs
};

// Issue of the scaling (maxima), exponent-delta and slicing programs, two each (A and B).
// The order is maxima A, B -> exponent deltas A, B -> slices A, B. Programs on one
// command queue execute in the order of issue, so the dependencies hold without waiting
// in between. OZ1_FILL_PROGRAMS is the total number of programs issued by the three
// functions.
constexpr uint64_t OZ1_FILL_PROGRAMS = 6;
template <class Note>
static void fill_issue_scale(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs, Note&& note) {
    scale_run(md, fs.scA);
    note(evaluation::Head);
    scale_run(md, fs.scB);
    note(evaluation::Head);
}
template <class Note>
static void fill_issue_expdelta(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs,
                                Note&& note) {
    expdelta_run(md, fs.edA);
    note(evaluation::Head);
    expdelta_run(md, fs.edB);
    note(evaluation::Head);
}

template <class Note>
static void fill_issue_slice(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs, Note&& note) {
    slice_run(md, fs.slc, fs.sliceA);
    note(evaluation::Head);
    slice_run(md, fs.slc, fs.sliceB);
    note(evaluation::Head);
}

// Configuration and resources that the GEMM and reconstruction stages refer to (not owned).
struct StageCtx {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md;
    const RunCfg& c;
    const IssuePlan& ip;
    const PanelPlan& pp;
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig;  // partial sums per diagonal (one row band)
    const InputBufs& in;
    const ReconScale& rs;
};

// Reconstruction programs of the row bands. Bands have at most two heights (the regular
// one and that of the last, shorter band), so two reconstructions are built; they share
// the limb planes and the FP64 output. rcn.out holds the final result.
struct BandRecon {
    ReconCtx rcn, rcnT;  // regular height / height of the last band
    uint32_t rcn_h = 0;  // band height of rcn [tile rows]
    bool rcn_built = false, rcnT_built = false;
};

// Returns the reconstruction for bands of `rows` tile rows, building it on first use.
// src is the buffer of partial sums.
// How it is built follows from the plan: the accumulating programs exist unless the
// configuration is IssuePlan::single_kblock(), and defer is IssuePlan::defer; full-size
// limb planes are allocated only for the per-block Horner (IssuePlan::recon_limbs()).
// The first call builds the reconstruction of the last band's height as well (rcnT
// shares the limb planes and the output with rcn), so that every program and buffer of
// the stage exists before the first band is reconstructed.
static ReconCtx& band_recon(const StageCtx& s, BandRecon& br, uint32_t rows,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& src) {
    const uint32_t Mt = s.c.Mt, Nt = s.c.Nt, ndiag = s.c.nslice();
    const bool with_acc = !s.ip.single_kblock(), defer_in = s.ip.defer;
    if (!br.rcn_built) {
        br.rcn = recon_build(s.md, src, rows, Nt, ndiag, s.rs, with_acc, defer_in, s.in.dltAb, s.in.dltBb,
                             s.pp.band_tiles, Mt * Nt, ReconShared{});
        br.rcn_built = true;
        br.rcn_h = rows;
    }
    const uint32_t tail_h = s.pp.bands.back().rows;
    if (!br.rcnT_built && tail_h != br.rcn_h) {
        br.rcnT = recon_build(s.md, src, tail_h, Nt, ndiag, s.rs, with_acc, defer_in, s.in.dltAb, s.in.dltBb,
                              s.pp.band_tiles, Mt * Nt, ReconShared{br.rcn.out, br.rcn.hi, br.rcn.mid, br.rcn.lo});
        br.rcnT_built = true;
    }
    if (rows == br.rcn_h) return br.rcn;

    // plan_panels produces bands of the first and of the last height only.
    TT_FATAL(rows == tail_h, "band_recon: unexpected band height {} (expected {} or {})", rows, br.rcn_h, tail_h);
    return br.rcnT;
}

// Returns the reconstruction of band bd (band_recon) with its output positioned at the
// first tile row of the band.
static ReconCtx& band_recon_at(const StageCtx& s, BandRecon& br, const Band& bd,
                               const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& src) {
    ReconCtx& rr = band_recon(s, br, bd.rows, src);
    if (s.pp.nband > 1) recon_set_out_off(rr, bd.row0 * s.c.Nt);
    return rr;
}

// Builds the GEMM program of panel pi, which belongs to band bd.
static GemmCtx build_panel(const StageCtx& s, uint32_t pi, const Band& bd) {
    const Panel& pn = s.pp.panels[pi];
    GemmBuildSpec b;
    b.M = s.c.M;
    b.N = s.c.N;
    b.K = s.c.K;
    b.nslice = s.c.nslice();
    b.Kbt = s.pp.Kbt;
    b.plane_stride = s.pp.band_tiles;
    b.mode = GemmMode{s.ip.fuse, s.ip.pairloop};
    return gemm_build(s.md, s.pp.cfg_of(pn), b, s.Pbig, s.in.shA, s.in.shB, pn.row0, pn.col0, bd.row0);
}

// ===== Single K block without pair loop or pairwise issue (IssuePlan::single_kblock) =====
// All diagonals of one panel: diagonal d runs its d+1 pairs concatenated along K as one
// GEMM and writes plane d.
template <class Note>
static void issue_panel_diagonals(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, GemmCtx& ctx, int L,
                                  Note&& note) {
    for (int d = 0; d <= L; d++) {
        gemm_run(md, ctx, (uint32_t)d, (uint32_t)d);
        note(evaluation::GemmStep);
    }
}

// ===== Sweep: every other configuration =====
// Per band, the sweep issues rounds (b, a) for each K block b and a = 0..inner(): the
// GEMMs of the round for all panels of the band (gemms), followed by the reconstruction
// of K block b if Horner runs per K block (post). With deferred reconstruction a single
// reconstruction ends the band (final). A round issues
//   pairwise   for every diagonal d >= a, its pair number a (slice a of A with slice
//              d-a of B); inner() = L;
//   otherwise  all diagonals of K block b (pair loop, fused concatenated issue,
//              per-block Horner); inner() = 0.
// With fuse, a GEMM stores its result when its plane is written for the first time and
// adds to the plane afterwards. The first round of a band, (b, a) = (0, 0), writes every
// plane 0..L, so only that round stores.
struct Oz1Sweep {
    const StageCtx& s;
    std::vector<GemmCtx>& ctxs;  // GEMM programs of all panels, indexed by panel

    uint32_t inner() const { return s.ip.pairwise ? (uint32_t)s.c.L() : 0u; }

    // Reconstruction of band bd, reading the buffer of partial sums.
    ReconCtx& recon(BandRecon& br, const Band& bd) const { return band_recon_at(s, br, bd, s.Pbig); }

    // GEMMs of round (b, a) for all panels of band bd. first: this is the first round
    // of the band.
    template <class Note>
    void gemms(const Band& bd, uint32_t b, uint32_t a, bool first, Note&& note) const {
        const bool pairwise = s.ip.pairwise;
        const int L = s.c.L();
        for (uint32_t pi = bd.p0; pi < bd.p1; pi++) {
            auto& ctx = ctxs[pi];
            for (int d = (pairwise ? (int)a : 0); d <= L; d++) {
                gemm_run(s.md, ctx, (uint32_t)d, (uint32_t)d, b, pairwise ? (int)a : -1, first ? 0u : 1u);
                note(evaluation::GemmStep);
            }
        }
    }

    // Issues what follows the GEMMs of K block b and returns whether anything was
    // issued: nothing if the GEMMs accumulate themselves (fuse, pair loop), otherwise
    // the Horner of the K block (initialize, accumulate, or accumulate and assemble).
    template <class Note>
    bool post(ReconCtx& rr, uint32_t b, Note&& note) const {
        if (s.ip.fuse || s.ip.pairloop) return false;
        recon_exec(s.md, rr, b == 0 ? 0u : (b + 1 == s.ip.nkb ? 2u : 1u));
        note(evaluation::ReconStep);
        return true;
    }

    // The single reconstruction of the deferred case: Horner over the Int32 accumulation
    // of all K blocks, and FP64 assembly.
    template <class Note>
    void final(ReconCtx& rr, Note&& note) const {
        recon_exec(s.md, rr, 0u);
        note(evaluation::ReconStep);
    }
};

// The `note` of the normal execution: counts nothing.
inline constexpr auto oz1_no_note = [](evaluation::Step) {};

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
