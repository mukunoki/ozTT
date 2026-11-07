// SPDX-License-Identifier: Apache-2.0
// Ozaki scheme II backend driver: one FP64 GEMM on the device. Compiled once per format
// into namespace ttgemm::detail::ozaki2_<format>.
//
// Pipeline of run(): plan (moduli, width b, passes: backend/ozaki2/plan.hpp) -> upload
// the FP64 operands -> scale (row/column maxima, exponent deltas) -> slice (residue or
// digit planes per modulus) -> per output band: partial-product GEMMs, composition of the
// digit partial products (digit formats), CRT reconstruction to FP64 -> read back.
// The stage programs are built by backend/ozaki2/stages.hpp; this file allocates the
// buffers, enqueues the programs and feeds the evaluator hooks (ev::), which time the
// phases and record every device step for the whole-pass replay.
#include "backend_interface.hpp"
#include "evaluation_hooks.hpp"
#include "ozaki2/stages.hpp"
#include "common/scale.hpp"
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
namespace ttgemm::detail::TTGEMM_BACKEND_NS {
namespace dist = tt::tt_metal::distributed;
namespace ev = ::ttgemm::detail::evaluation;
// One calculation. When DRAM is short the output is processed in row bands
// (plan_bands): B is sliced once for all moduli, and per band the A rows are sliced,
// multiplied for every modulus, composed and reconstructed into the band's rows of
// the FP64 output. The band buffers are allocated in the first band and reused.
// Buffers normally released between stages are kept while the evaluator records the
// whole pass (may_release); the plan sizes the bands for that resident set.
struct Oz2Run : Oz2Plan {
    Device md;
    tt::tt_metal::CoreCoord gd;  // compute grid of the device
    Oz2Knobs kn;
    // FP64 operands (tile order), row/column maxima (BF16), exponent deltas (Int32) and
    // the out-of-range flags of the scaling stage (one tile per core).
    std::shared_ptr<dist::MeshBuffer> rAf64, rBf64, maxAb, maxBb, dltAb, dltBb, flgA, flgB;
    std::optional<ScaleCtx> scCA, scCB;   // programs that reduce the row/column maxima
    std::optional<ExpDeltaCtx> edA, edB;  // programs that turn the maxima into exponent deltas
    std::optional<Oz2Slice> slA, slB;
    std::shared_ptr<dist::MeshBuffer> slAsh;        // A slice planes of one band, reused by every band
    std::shared_ptr<dist::MeshBuffer> crt_in, o64;  // S residue planes of one band (reconstruction input); FP64 output
    std::shared_ptr<dist::MeshBuffer> rlimb[3], ra128;  // reconstruction buffers of one band
    uint32_t band = 0;                                  // current band
    uint32_t br0() const { return band * bpr; }         // first panel row of the band
    uint32_t aOff() const { return br0() * pMt * Kt; }  // first FP64 tile of the band in A
    uint32_t cOff() const { return br0() * pMt * Nt; }  // first tile of the band in the output
    bool last_band() const { return band + 1 == nband; }
    bool may_release() const { return !ev::recording_whole(); }
#ifdef OZ2_DIGITK
    std::shared_ptr<dist::MeshBuffer> Dbuf;  // product planes of one band (cplan.planes)
    std::optional<Oz2Compose> cmp;           // the composition program, when one depth serves all launches
    std::map<uint32_t, std::unique_ptr<Oz2Compose>> cmp_by_jn;  // otherwise one program per number of planes per launch
    bool cmp_split = false;
#endif
};
// Build, run and record one partial-product GEMM (panel of m rows, inner length k
// elements; offsets da, db, dw in tiles; stride: row stride of A in tiles, 0 = k/32).
static void run_partial_gemm(Oz2Run& r, std::shared_ptr<dist::MeshBuffer> a, std::shared_ptr<dist::MeshBuffer> b,
                             uint32_t da, uint32_t db, uint32_t m, uint32_t k, std::shared_ptr<dist::MeshBuffer> c,
                             uint32_t nt, uint32_t dw, uint32_t stride = 0) {
    auto workload =
        std::make_shared<dist::MeshWorkload>(gemm_build_partial(r.md, r.g, a, b, da, db, m, k, c, nt, dw, stride));
    dist::MeshCommandQueue* q = &r.md->mesh_command_queue();
    ev::execute(
        ev::Gemm, [q, workload] { dist::EnqueueMeshWorkload(*q, *workload, false); }, [q] { dist::Finish(*q); });
    ev::record(ev::GemmStep, [q, workload] { dist::EnqueueMeshWorkload(*q, *workload, false); });
}
#ifdef OZ2_DIGITK
// Run and record one launch of the composition stage for modulus s: input planes
// [in0,in0+jn) with the coefficients of the GEMM outputs j0.. (or 1 with unit), output
// plane `plane` of the buffer at `address` (the residue buffer, or the product buffer when
// a partial sum is stored).
static void compose_run(Oz2Run& r, Oz2Compose& c, uint32_t s, uint32_t j0, uint32_t jn, bool unit, uint32_t in0,
                        uint32_t address, uint32_t plane) {
    if (address != c.out_addr && !(address == c.in_addr && c.out_accepts_in))
        throw std::runtime_error("incompatible composition output buffer");
    compose_set_args(c, r.mo, s, j0, jn, unit, in0, address, plane);
    ev::execute(
        ev::Compose, [&] { dist::EnqueueMeshWorkload(r.md->mesh_command_queue(), c.workload, false); },
        [&] { dist::Finish(r.md->mesh_command_queue()); });
    // The composition workload is reused with different arguments, so the recorded
    // step sets the arguments again before issuing (host work inside the whole pass).
    dist::MeshCommandQueue* q = &r.md->mesh_command_queue();
    Oz2Compose* cp = &c;
    const Ozaki2Moduli* mop = &r.mo;
    ev::record(ev::ComposeStep, [q, cp, mop, s, j0, jn, unit, in0, address, plane] {
        compose_set_args(*cp, *mop, s, j0, jn, unit, in0, address, plane);
        dist::EnqueueMeshWorkload(*q, cp->workload, false);
    });
}
#endif
// Allocate the operand-side buffers and send A and B to the device as FP64 in tile order
// (8 bytes per element, unnormalized). The host only reorders into tiles; the conversion
// to triple-float and the scaling happen on the device.
static void upload_operands(Oz2Run& r, const std::vector<double>& A, const std::vector<double>& B) {
    auto& md = r.md;
    auto& cq = md->mesh_command_queue();
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32), bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    auto mkb = [&](size_t nt, uint32_t ps) { return mesh_dram_buffer(md, nt, ps); };
    // Allocation order. The formats that select their moduli with the stage-time model
    // allocate the small long-lived buffers (maxima, exponent deltas, flags) before the
    // FP64 operands and take the slice planes from the top of DRAM (build_slice). With
    // the FP64 operands first, the hole they leave when released after slicing would be
    // enclosed by the small buffers and the slice planes, and the residue planes (S planes)
    // might not find a contiguous block. The other formats allocate the FP64 operands
    // first and the slice planes bottom-up.
    const uint32_t nslot = r.gd.x * r.gd.y;
    auto mk_small = [&]() {
        r.maxAb = mkb(r.Mt, bf16t);
        r.maxBb = mkb(r.Nt, bf16t);
        r.dltAb = mkb(r.Mt, f32t);
        r.dltBb = mkb(r.Nt, f32t);
        // Out-of-range flags: one tile per core, zeroed before the operands are sent.
        r.flgA = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
        r.flgB = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
    };
    if (OZ2_PARAM.awe) mk_small();

    r.rAf64 = mkb(r.ntotA, OZ_F64_TILE_BYTES);
    r.rBf64 = mkb(r.ntotB, OZ_F64_TILE_BYTES);
    if (!OZ2_PARAM.awe) mk_small();
    {
        std::vector<uint32_t> z((size_t)nslot * tt::tile_size(tt::DataFormat::Int32) / 4, 0);
        dist::EnqueueWriteMeshBuffer(cq, r.flgA, z, false);
        dist::EnqueueWriteMeshBuffer(cq, r.flgB, z, false);
        dist::Finish(cq);
    }
    upload_operand_f64(md, A, r.M, r.K, r.rAf64);
    upload_operand_f64(md, B, r.K, r.N, r.rBf64);
}

// Stage 0 (scale, exponent deltas) and stage 1 (slicing). With several bands only B is
// sliced here; A is sliced per band by slice_band. The recorded head step issues the
// same programs without synchronization.
// Scaling: the maximum magnitude of every row of A and column of B is reduced on the
// device (the BF16 conversion of |x| rounds up: a maximum below the true one would let a
// normalized value exceed 1 and break |A'| <= 2^b) and rounded up to a power of two 2^e;
// the exponent deltas then carry 2^(oz2_shift - e), so that the slicer only has to take
// the digits. The operand buffers and the maxima are released after slicing; the exponent
// deltas stay, because the reconstruction adds them back.
static void scale_and_slice(Oz2Run& r) {
    auto& md = r.md;
    auto& q = md->mesh_command_queue();
    r.scCA.emplace(scale_build(md, 0u, r.Kt, r.rAf64, r.maxAb, r.Mt, 0u, true));
    r.scCB.emplace(scale_build(md, 1u, r.Kt, r.rBf64, r.maxBb, r.Nt, r.Nt, true));
    r.edA.emplace(expdelta_build(md, r.maxAb, r.dltAb, r.flgA, r.Mt, true, oz2_shift(r)));
    r.edB.emplace(expdelta_build(md, r.maxBb, r.dltBb, r.flgB, r.Nt, false, oz2_shift(r)));
    ev::execute(ev::Scale, [&] {
        scale_run(md, *r.scCA);
        scale_run(md, *r.scCB);
        if (ev::active) dist::Finish(q);
        expdelta_run(md, *r.edA);
        expdelta_run(md, *r.edB);
        dist::Finish(q);
    });
    check_moduli_layout(r.mo);  // the slicers and the GEMM stage read the plane layout of the set
    r.slB.emplace(build_slice(md, r.mo, r.ND, r.K, r.N, r.rBf64, r.dltBb, 1u));
    const bool banded = r.nband > 1;
    if (!banded) r.slA.emplace(build_slice(md, r.mo, r.ND, r.M, r.K, r.rAf64, r.dltAb, 0u));
    ev::execute(
        ev::Slice,
        [&] {
            if (!banded) dist::EnqueueMeshWorkload(q, r.slA->workload, false);
            dist::EnqueueMeshWorkload(q, r.slB->workload, false);
        },
        [&] { dist::Finish(q); });
    Oz2Run* rp = &r;
    ev::record(ev::Head, [rp, banded] {
        Oz2Run& r = *rp;
        auto& q = r.md->mesh_command_queue();
        scale_run(r.md, *r.scCA);
        scale_run(r.md, *r.scCB);
        expdelta_run(r.md, *r.edA);
        expdelta_run(r.md, *r.edB);
        if (!banded) dist::EnqueueMeshWorkload(q, r.slA->workload, false);
        dist::EnqueueMeshWorkload(q, r.slB->workload, false);
    });
    if (r.may_release()) {
        if (!banded) r.rAf64.reset();
        r.rBf64.reset();
        r.maxAb.reset();
        r.maxBb.reset();
    }
}
// Slice the A rows of band r.band into the shared band buffer (nband > 1 only). The
// FP64 reader starts at the band's first tile; the writer uses in-band tile numbers.
static void slice_band(Oz2Run& r) {
    auto& md = r.md;
    auto& q = md->mesh_command_queue();
    if (!r.slAsh)  // all slice planes of one band (mo.nplane; for INT8 one residue plane per modulus)
        r.slAsh =
            mesh_dram_buffer(md, (size_t)r.mo.nplane * r.ntotAb(), tt::tile_size(OZ_FMT.slice_out), OZ2_PARAM.awe != 0);
    r.slA.reset();
    r.slA.emplace(build_slice(md, r.mo, r.ND, r.bMt() * 32, r.K, r.rAf64, r.dltAb, 0u, r.slAsh, r.aOff()));
    ev::execute(ev::Slice, [&] { dist::EnqueueMeshWorkload(q, r.slA->workload, false); }, [&] { dist::Finish(q); });
    // The recorded head step issues the band's slicer again, so the program moves into the
    // step; otherwise it is released here (slA keeps the output buffer and the layout).
    if (ev::recording_whole()) {
        auto workload = std::make_shared<dist::MeshWorkload>(std::move(r.slA->workload));
        dist::MeshCommandQueue* qp = &q;
        ev::record(ev::Head, [qp, workload] { dist::EnqueueMeshWorkload(*qp, *workload, false); });
    }
    r.slA->workload = dist::MeshWorkload{};
    if (r.last_band() && r.may_release()) r.rAf64.reset();
}
#ifdef OZ2_DIGITK
// The composition program for launches of jn planes. With cmp_split one program per jn is
// built on first use, with the depth jn*ceil(OZ2_COMPOSE_CB_MIN/jn).
static Oz2Compose& compose_for(Oz2Run& r, uint32_t jn) {
    if (!r.cmp_split) return *r.cmp;
    auto it = r.cmp_by_jn.find(jn);
    if (it == r.cmp_by_jn.end()) {
        const uint32_t depth = jn * ((OZ2_COMPOSE_CB_MIN + jn - 1) / jn);
        it = r.cmp_by_jn
                 .emplace(jn, std::make_unique<Oz2Compose>(build_compose(r.md, r.ntotCb(), r.Dbuf, r.crt_in, depth)))
                 .first;
    }
    return *it->second;
}
// Issue the GEMMs of the outputs j0..j0+jn-1 (one chunk) of modulus s for every output
// panel of the band (panel row br, panel column bc, output j). Output j is the sum of the
// products of the A digit planes a0..a0+g-1 with the B positions b0..b0+g-1, so its inner
// length is g*K (oz2_product_groups in split.hpp; g = 1 without concatenation). The layout
// of the slice planes follows the overall maxg, so a modulus that is not concatenated is
// read with g = 1 on the concatenated layout as well. PA: number of digit planes; po:
// first digit plane of modulus s. A one-digit modulus (raw) writes its GEMM output
// directly to residue plane s (the reconstruction reduces it modulo m); the others write
// product plane j.
static void issue_digit_chunk(Oz2Run& r, uint32_t s, uint32_t PA, uint32_t po, const Oz2ProductGroup* gr, uint32_t j0,
                              uint32_t jn, bool raw) {
    const Ozaki2Moduli& mo = r.mo;
    auto& sA = r.slA->out;
    auto& sB = r.slB->out;
    // Output planes and A slice planes hold one band; br is the row panel within the band.
    const uint32_t K = r.K, Nt = r.Nt, ntotC = r.ntotCb();
    for (uint32_t br = 0; br < r.bpr; br++) {
        for (uint32_t bc = 0; bc < r.nbc; bc++) {
            uint32_t Mp = r.pMt * 32, dW = br * r.pMt * Nt + bc * r.pNt;
            for (uint32_t j = 0; j < jn; j++) {
                const Oz2ProductGroup& q = gr[j0 + j];
                auto& Cb = raw ? r.crt_in : r.Dbuf;
                const uint32_t cw = (raw ? s : j) * ntotC + dW;
                if (mo.maxg > 1)  // concatenated layout (LAYOUT_A_CAT / LAYOUT_B_PERM of the slice writer)
                    run_partial_gemm(r, sA, sB, panel_off_Acat(r, PA, (uint32_t)(po + q.a0), br),
                                     panel_off_B(r, (uint32_t)(po + q.b0), bc), Mp, (uint32_t)q.g * K, Cb, Nt, cw,
                                     PA * r.Kt);
                else {  // plane-by-plane layout: position b0 of B is digit npl-1-b0
                    const uint32_t npl = (uint32_t)oz2_digit_planes(mo.m[s], OZ2_PARAM.digit_bound, OZ2_PARAM.nd());
                    run_partial_gemm(r, sA, sB, panel_off_A(r, (uint32_t)(po + q.a0), br),
                                     panel_off_B(r, (uint32_t)(po + (npl - 1 - q.b0)), bc), Mp, K, Cb, Nt, cw);
                }
            }
        }
    }
}
// GEMMs and composition of modulus s. The np outputs are processed in chunks of CH: the
// GEMMs of a chunk are issued and the composition reduces them into residue plane s.
// Chunking happens only when a modulus has more than OZ2_COMPOSE_NOSPLIT_MAX outputs; then
// each chunk leaves a partial sum in product plane CH+nch, and after the last chunk the
// partial sums are added with coefficient 1 and reduced. Without chunking CH >= np and one
// launch finishes the modulus. A one-digit modulus is neither chunked nor composed.
static void run_digit_modulus(Oz2Run& r, uint32_t s, uint32_t PA, uint32_t po) {
    const Ozaki2Moduli& mo = r.mo;
    const uint32_t CH = r.cplan.chunk;
    Oz2ProductGroup gr[OZ2_MAXPROD];
    // The concatenation bound is per modulus (maxg_m).
    const uint32_t np = oz2_product_groups(mo.m[s], gr, OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[s]);
    const bool raw = mo.mode[s] == OZ2_SPLIT_ONE;
    TT_FATAL(!raw || np == 1, "run_digit_modulus: one-digit modulus {} has {} outputs", mo.m[s], np);
    // When no modulus is composed CH is 0, which must not be used as the step.
    const uint32_t CHs = raw ? np : CH;
    uint32_t nch = 0;
    for (uint32_t j0 = 0; j0 < np; j0 += CHs, nch++) {
        const uint32_t jn = std::min(CHs, np - j0);
        issue_digit_chunk(r, s, PA, po, gr, j0, jn, raw);
        if (raw) continue;  // not composed: residue plane s already holds the GEMM output
        // Without chunking write residue plane s directly; with chunking store the partial
        // sum in product plane CH+nch.
        const bool single_chunk = (np <= CH);
        compose_run(r, compose_for(r, jn), s, j0, jn, false, 0u,
                    single_chunk ? (uint32_t)r.crt_in->address() : (uint32_t)r.Dbuf->address(),
                    single_chunk ? s : (CH + nch));
    }
    if (!raw && np > CH)  // add the nch partial sums (planes CH..CH+nch-1) with coefficient 1 and reduce
        compose_run(r, compose_for(r, nch), s, 0u, nch, true, CH, (uint32_t)r.crt_in->address(), s);
}
// Stages 2 and 2.5 of the digit formats for the current band: the GEMMs of the digit
// products and the composition per modulus. Allocated in the first band, in this order:
// Dbuf (cplan.planes product planes), crt_in (S residue planes), then the composition
// program. One FP32 plane per modulus suffices for the residues: the moduli selection has
// verified that every value stays within the exact FP32 integers. The exactness
// conditions (digit products within 2^24, composition sums within 2^22) are those checked
// in moduli.hpp.
static void run_digit_gemms(Oz2Run& r) {
    auto& md = r.md;
    const Ozaki2Moduli& mo = r.mo;
    const uint32_t S = r.S, ntotC = r.ntotCb();  // product planes and residues hold one band
    const uint32_t oz2_out_tile = tt::tile_size(OZ_FMT.out);
    const uint32_t nD = r.cplan.planes, CH = r.cplan.chunk;
    if (r.band == 0) {  // buffers and the composition stage are built once and reused by every band
        // No composed modulus (all one-digit): neither product planes nor a composition program.
        if (nD) r.Dbuf = mesh_dram_buffer(md, (size_t)nD * ntotC, oz2_out_tile);
        r.crt_in = mesh_dram_buffer(md, (size_t)S * ntotC, oz2_out_tile);
        if (nD) {
            const uint32_t depth = compose_cb_depth(mo, CH);
            // One program per launch size only when the common multiple exceeds the maximum depth.
            r.cmp_split = depth > OZ2_COMPOSE_CB_MAX;
            if (!r.cmp_split) r.cmp.emplace(build_compose(md, ntotC, r.Dbuf, r.crt_in, depth));
        }
    }
    // Number of digit planes (sets the width PA*Kt of the concatenated A).
    const uint32_t PA = (uint32_t)mo.nplane;
    // The layout the slicers wrote must be the one read here: plane count, concatenation
    // bound, and layout (A_CAT / B_PERM with concatenation, PLANES without).
    TT_FATAL(r.slA->planes == PA && r.slB->planes == PA,
             "run_digit_gemms: the slices have {} (A) and {} (B) planes, the GEMMs read {}", r.slA->planes,
             r.slB->planes, PA);
    TT_FATAL(r.slA->maxg == mo.maxg && r.slB->maxg == mo.maxg,
             "run_digit_gemms: maxg {}/{} of the slices differs from {} of the GEMMs", r.slA->maxg, r.slB->maxg,
             mo.maxg);
    TT_FATAL(r.slA->layout == (mo.maxg > 1 ? oz2_args::LAYOUT_A_CAT : oz2_args::LAYOUT_PLANES) &&
                 r.slB->layout == (mo.maxg > 1 ? oz2_args::LAYOUT_B_PERM : oz2_args::LAYOUT_PLANES),
             "run_digit_gemms: slice layouts {} (A) and {} (B) do not match maxg={}", r.slA->layout, r.slB->layout,
             mo.maxg);
    for (uint32_t s = 0; s < S; s++) run_digit_modulus(r, s, PA, (uint32_t)mo.poff[s]);
    // The product planes are done (the residues are in crt_in); the next band reuses them.
    if (r.last_band() && r.may_release()) r.Dbuf.reset();
}

#else
// Stage 2 of INT8 for the current band: one GEMM per modulus and output panel. C_i
// (sign-magnitude int32, one plane per modulus) is written straight into plane s of crt_in
// and stays on the device; the reconstruction kernel decodes it.
static void run_int8_gemms(Oz2Run& r) {
    auto& md = r.md;
    const uint32_t S = r.S, K = r.K, Nt = r.Nt, ntotC = r.ntotCb();  // C_i planes hold one band
    if (r.band == 0) r.crt_in = mesh_dram_buffer(md, (size_t)S * ntotC, tt::tile_size(OZ_FMT.out));
    auto& sA = r.slA->out;
    auto& sB = r.slB->out;
    for (uint32_t s = 0; s < S; s++) {
        for (uint32_t br = 0; br < r.bpr; br++) {
            for (uint32_t bc = 0; bc < r.nbc; bc++) {
                uint32_t dW = (uint32_t)s * ntotC + br * r.pMt * Nt + bc * r.pNt;
                run_partial_gemm(r, sA, sB, panel_off_A(r, s, br), panel_off_B(r, s, bc), r.pMt * 32, K, r.crt_in, Nt,
                                 dW);
            }
        }
    }
}

#endif
// Stage 2 (and 2.5) for the current band.
static void run_gemm_stage(Oz2Run& r) {
#ifdef OZ2_DIGITK
    run_digit_gemms(r);
#else
    run_int8_gemms(r);
#endif
    // Slice planes are reused by the next band; release after the last band only.
    if (r.last_band() && r.may_release()) {
        r.slA->out.reset();
        r.slB->out.reset();
        r.slAsh.reset();
    }
}

// Stage 3: reconstruct the current band's rows into o64 (the whole output). The limb
// buffers and the 128-bit fixed-point buffer are allocated in the first band and reused
// (sizes from make_recon_plan). The input never passes through the host; the device also
// applies the final scale and adds the row and column exponents back.
static void reconstruct(Oz2Run& r) {
    if (!r.o64) r.o64 = mesh_dram_buffer(r.md, r.ntotC, OZ_F64_TILE_BYTES);
    const auto plan = make_recon_plan(r.mo, r.two_pass, r.ntotCb());
    const auto scale = make_recon_scale(r.mo, 2 * oz2_shift(r));
    if (!r.ra128) {
        for (auto& limb : r.rlimb)
            limb = mesh_dram_buffer(r.md, OZ2_LIMB_TILES, tt::tile_size(tt::DataFormat::Float32));
        r.ra128 = mesh_dram_buffer(r.md, plan.a128_tiles, OZ2_A128_PAGE);
    }
    ReconBufs buffers;
    buffers.in = r.crt_in;
    buffers.o64 = r.o64;
    buffers.dltA = r.dltAb;
    buffers.dltB = r.dltBb;
    for (int i = 0; i < 3; ++i) buffers.limb[i] = r.rlimb[i];
    buffers.a128 = r.ra128;
    buffers.out_off = r.cOff();
    dist::MeshCommandQueue* q = &r.md->mesh_command_queue();
    for (const auto& pass : plan.pass) {
        auto workload =
            std::make_shared<dist::MeshWorkload>(recon_build_pass(r.md, pass, buffers, r.bMt() * 32, r.N, r.mo, scale));
        ev::execute(
            ev::Recon, [q, workload] { dist::EnqueueMeshWorkload(*q, *workload, false); }, [q] { dist::Finish(*q); });
        ev::record(ev::ReconStep, [q, workload] { dist::EnqueueMeshWorkload(*q, *workload, false); });
    }
}
// Expected step counts of one whole pass, derived from the configuration.
static ev::Counts whole_expected(const Oz2Run& r) {
    const Ozaki2Moduli& mo = r.mo;
    ev::Counts exp;
    exp[ev::Head] = r.nband > 1 ? 1u + r.nband : 1u;
    exp[ev::GemmStep] = (unsigned long long)mo.nout * r.nbr * r.nbc;
    const uint32_t chunk = r.cplan.chunk;
    for (int s = 0; s < mo.s; s++) {
        if (mo.mode[s] == OZ2_SPLIT_ONE || chunk == 0) continue;
        const uint32_t np = oz2_nout(mo.m[s], OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[s]);
        exp[ev::ComposeStep] += (np + chunk - 1) / chunk + (np > chunk ? 1u : 0u);
    }
    exp[ev::ComposeStep] *= r.nband;
    exp[ev::ReconStep] = (r.two_pass ? 2ull : 1ull) * r.nband;  // passes of make_recon_plan
    return exp;
}
// Check and time the whole pass (evaluator only): the intermediate and output buffers
// are filled with the sentinel, the recorded steps are replayed once and compared with
// the phase-timed output, then the pass is warmed and timed.
static void whole_pipeline(Oz2Run& r) {
    if (!ev::recording_whole()) return;
    auto& cq = r.md->mesh_command_queue();
    std::vector<std::shared_ptr<dist::MeshBuffer>> bufs{r.maxAb,    r.maxBb,    r.dltAb,    r.dltBb,
                                                        r.slA->out, r.slB->out, r.o64,      r.crt_in,
                                                        r.rlimb[0], r.rlimb[1], r.rlimb[2], r.ra128};
#ifdef OZ2_DIGITK
    if (r.Dbuf) bufs.push_back(r.Dbuf);  // absent when no modulus is composed
#endif
    for (const auto& b : bufs)
        if (!b) throw std::runtime_error("a buffer of the whole pass was released");
    ev::whole(
        [&] {
            for (const auto& b : bufs) fill_mesh_buffer_u32(r.md, b, ev::SENTINEL);
        },
        [&](std::vector<uint32_t>& out) {
            out.assign((size_t)r.ntotC * (OZ_F64_TILE_BYTES / 4), 0u);
            dist::EnqueueReadMeshBuffer(cq, out, r.o64, true);
        },
        [&] { dist::Finish(cq); }, [&](auto&& note) { ev::replay(note); }, whole_expected(r));
}
// The backend entry: C = A*B for the padded problem p (m and n multiples of 64, k of 32).
// Throws std::runtime_error when the configuration cannot be executed (the planner
// functions carry the cause in the message).
Output run(Device& md, const Problem& p) {
    Oz2Run r;
    r.M = p.m;
    r.N = p.n;
    r.K = p.k;
    r.kn.max_moduli = p.options.max_moduli;
    r.kn.passes = p.options.reconstruction;
    // The width b is chosen from the input unless the evaluator fixes it.
    if (ev::active && ev::active->fixed_bits) r.kn.b = ev::active->fixed_bits;
    choose_moduli_and_width(r, r.kn, p.a, p.b);
    r.md = md;
    r.gd = md->compute_with_storage_grid_size();
    plan_panels(r, r.gd);
    if (ev::active) {
        ev::active->partial_products = r.mo.nprod;
        ev::active->bandwidth = r.g.bw;
        ev::active->cores = r.gd.x * r.gd.y;
    }
    upload_operands(r, p.a, p.b);
    // The bands are planned from the DRAM that is free once the FP64 operands are allocated.
    plan_bands(r, dram_free(md));
    scale_and_slice(r);
    for (r.band = 0; r.band < r.nband; r.band++) {
        if (r.nband > 1) slice_band(r);
        run_gemm_stage(r);
        reconstruct(r);
    }
    whole_pipeline(r);
    // The device assembled FP64; the host only reorders the tiles into row-major order.
    std::vector<uint32_t> tiles;
    dist::EnqueueReadMeshBuffer(md->mesh_command_queue(), tiles, r.o64, true);
    // Flags raised by the scaling stage for row/column maxima outside the supported range.
    expdelta_check(md, *r.edA, "rows of A");
    expdelta_check(md, *r.edB, "columns of B");
    Info info{Scheme::II, TTGEMM_PUBLIC_FORMAT};
    info.moduli = r.S;
    info.integer_bits = r.b;
    info.reconstruction_passes = r.two_pass ? 2 : 1;
    return {untilize_f64(tiles, r.M, r.N), info};
}
}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
