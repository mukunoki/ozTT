// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I backend: driver of one GEMM call.
//
// Compiled once per format into the namespace ttgemm::detail::ozaki1_<format>
// (TTGEMM_BACKEND_NS; the format is selected by an OZ_FMT_* macro). run() carries out the
// whole pipeline on the device: scaling (row/column maxima and exponent deltas), slicing,
// the partial-product GEMMs per diagonal with their accumulation, and the reconstruction
// to FP64. The host plans the run, selects the number of slices from the operand values
// when none is requested (auto_precision.hpp), uploads the FP64 operands in tile order
// and reorders the FP64 result; the transfer helpers do no arithmetic on the data. The
// stages are in backend/ozaki1/ (plan.hpp plans the run, stages.hpp issues the programs).
//
// The calls into evaluation:: (ev::execute, ev::phase, ev::sweep, ev::whole) execute
// their argument once in normal operation. Under the evaluator they repeat and time the
// stages, and whole_pipeline() replays every device step of the call without
// synchronization between the stages.
#include "backend_interface.hpp"
#include "evaluation_hooks.hpp"
#include "ozaki1/stages.hpp"
#include "ozaki1/auto_precision.hpp"
#include <stdexcept>
namespace ttgemm::detail::TTGEMM_BACKEND_NS {
namespace dist = tt::tt_metal::distributed;
namespace ev = ::ttgemm::detail::evaluation;
// Allocates the buffer of partial sums: one plane per diagonal, each of one row band.
// It is allocated before the input buffers; the order of the allocations determines the
// DRAM layout and is kept as is.
static std::shared_ptr<dist::MeshBuffer> alloc_pbig(std::shared_ptr<dist::MeshDevice>& md, const RunCfg& c,
                                                    const PanelPlan& pp) {
    dist::ReplicatedBufferConfig bP{.size = (size_t)pp.pdiag_tile * (size_t)c.nslice() * pp.band_tiles};
    dist::DeviceLocalBufferConfig dP{.page_size = pp.pdiag_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    return dist::MeshBuffer::create(bP, dP, md.get());
}
// Allocates the input-side buffers in the order dltAb, dltBb, rAf64, rBf64, maxAb, maxBb,
// flgA, flgB, clears the flags, uploads the padded row-major FP64 operands A (M x K) and
// Bm (K x N), and then allocates the slice-concatenated operands shA, shB. The order must
// not be changed (DRAM layout).
static InputBufs setup_inputs(std::shared_ptr<dist::MeshDevice>& md, const RunCfg& c, const PanelPlan& pp,
                              const tt::tt_metal::CoreCoord& gd, const std::vector<double>& A,
                              const std::vector<double>& Bm) {
    using tt::tt_metal::BufferType;
    InputBufs in;
    auto& cq = md->mesh_command_queue();
    const uint32_t Mt = c.Mt, Nt = c.Nt, Kto = c.K / 32;
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32);
    {
        dist::DeviceLocalBufferConfig dF{.page_size = f32t, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bIa{.size = (size_t)f32t * Mt};
        dist::ReplicatedBufferConfig bIb{.size = (size_t)f32t * Nt};
        in.dltAb = dist::MeshBuffer::create(bIa, dF, md.get());
        in.dltBb = dist::MeshBuffer::create(bIb, dF, md.get());

        // FP64 operands: 8 bytes per element, one page of OZ_F64_TILE_BYTES per tile.
        dist::DeviceLocalBufferConfig dD{.page_size = OZ_F64_TILE_BYTES, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bAd{.size = (size_t)OZ_F64_TILE_BYTES * Mt * Kto};
        dist::ReplicatedBufferConfig bBd{.size = (size_t)OZ_F64_TILE_BYTES * Kto * Nt};
        in.rAf64 = dist::MeshBuffer::create(bAd, dD, md.get());
        in.rBf64 = dist::MeshBuffer::create(bBd, dD, md.get());
        uint32_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
        dist::DeviceLocalBufferConfig dB{.page_size = bf16t, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bMa{.size = (size_t)bf16t * Mt};
        dist::ReplicatedBufferConfig bMb{.size = (size_t)bf16t * Nt};
        in.maxAb = dist::MeshBuffer::create(bMa, dB, md.get());
        in.maxBb = dist::MeshBuffer::create(bMb, dB, md.get());
    }

    // Out-of-range flags, one tile per core. They are cleared before the operands are
    // uploaded: between the upload of A and B and the read-back of C the host does not
    // communicate with the device.
    const uint32_t nslot = gd.x * gd.y;
    in.flgA = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
    in.flgB = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
    {
        std::vector<uint32_t> z((size_t)nslot * tt::tile_size(tt::DataFormat::Int32) / 4, 0);
        dist::EnqueueWriteMeshBuffer(cq, in.flgA, z, false);
        dist::EnqueueWriteMeshBuffer(cq, in.flgB, z, false);
        dist::Finish(cq);
    }
    // The operands are uploaded as FP64; the host only reorders them into tiles.
    upload_operand_f64(md, A, c.M, c.K, in.rAf64);
    upload_operand_f64(md, Bm, c.K, c.N, in.rBf64);

    // shA and shB hold the slices of the whole operands; each panel selects its part
    // through the read offsets of its GEMM program, so they are built once and shared.
    {
        uint32_t in_tile = tt::tile_size(OZ_FMT.in);
        dist::DeviceLocalBufferConfig dIn{.page_size = in_tile, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bAc{.size = (size_t)in_tile * Mt * pp.Ktm};
        dist::ReplicatedBufferConfig bBc{.size = (size_t)in_tile * Nt * pp.Ktm};
        in.shA = dist::MeshBuffer::create(bAc, dIn, md.get());
        in.shB = dist::MeshBuffer::create(bBc, dIn, md.get());
    }
    return in;
}

// Builds and executes the scaling and slicing stages (shared by all panels, K blocks and
// diagonals). The scaling is closed on the device: the row/column maxima (scale), then
// the exponent deltas derived from them (expdelta); the normalization itself is fused
// into the slicer. A has Mt*kt input tiles with per-row deltas, B has kt*Nt with
// per-column deltas; both are written block-major with K blocks of pp.Kbt tiles, so one
// slicing run provides the concatenated slices of every K block.
static FillStage fill(Device& md, const RunCfg& c, const PanelPlan& pp, InputBufs& in) {
    FillStage fs;
    const uint32_t kt = c.K / 32;
    fs.edA = expdelta_build(md, in.maxAb, in.dltAb, in.flgA, c.Mt, true, OZ1_NSHIFT);
    fs.edB = expdelta_build(md, in.maxBb, in.dltBb, in.flgB, c.Nt, false, OZ1_NSHIFT);
    fs.scA = scale_build(md, 0u, kt, in.rAf64, in.maxAb, c.Mt, 0u, OZ1_BF_CEIL, OZ1_TAU_ABOVE);
    fs.scB = scale_build(md, 1u, kt, in.rBf64, in.maxBb, c.Nt, c.Nt, OZ1_BF_CEIL, OZ1_TAU_ABOVE);
    fs.slc = slice_build(md, c.nslice(), in.rAf64, in.dltAb, in.shA);
    fs.sliceA = SliceRun{in.rAf64, in.dltAb, in.shA, c.Mt * kt, kt, 0u, 0u, pp.Kbt};
    fs.sliceB = SliceRun{in.rBf64, in.dltBb, in.shB, kt * c.Nt, kt, c.Nt, 1u, pp.Kbt};
    ev::execute(ev::Slice, [&] {
        fill_issue_scale(md, fs, oz1_no_note);
        if (ev::active) dist::Finish(md->mesh_command_queue());
        fill_issue_expdelta(md, fs, oz1_no_note);
        if (ev::active) dist::Finish(md->mesh_command_queue());
        fill_issue_slice(md, fs, oz1_no_note);
        dist::Finish(md->mesh_command_queue());
    });
    return fs;
}
// Evaluator only: check and time the whole pass (scale, exponent deltas, slicing, then
// per band all GEMMs with accumulation/reconstruction), issued without synchronization
// between stages in the same order as the phase timing. The single-K-block path builds
// every panel's GEMM program again outside the timing (the phase timing built them one
// panel at a time); the sweep path reuses its contexts.
static std::vector<GemmCtx> all_panels(const StageCtx& s) {
    std::vector<GemmCtx> all;
    if (!ev::recording_whole() || !s.ip.single_kblock()) return all;
    all.reserve(s.pp.panels.size());
    for (const auto& band : s.pp.bands)
        for (uint32_t i = band.p0; i < band.p1; ++i) all.push_back(build_panel(s, i, band));
    return all;
}
static void whole_pipeline(const StageCtx& s, FillStage& fs, BandRecon& br, const Oz1Sweep& sw,
                           std::vector<GemmCtx> all) {
    if (!ev::recording_whole()) return;
    auto& md = s.md;
    auto& queue = md->mesh_command_queue();
    const bool single = s.ip.single_kblock();
    const uint32_t nkb = s.ip.nkb, inner = sw.inner();
    auto run_once = [&](auto&& note) {
        fill_issue_scale(md, fs, note);
        fill_issue_expdelta(md, fs, note);
        fill_issue_slice(md, fs, note);
        for (const auto& band : s.pp.bands) {
            if (single) {
                for (uint32_t i = band.p0; i < band.p1; ++i) issue_panel_diagonals(md, all[i], s.c.L(), note);
                recon_exec(md, band_recon_at(s, br, band, s.Pbig), 0u);
                note(ev::ReconStep);
            } else {
                auto& recon = sw.recon(br, band);
                bool first = true;
                for (uint32_t block = 0; block < nkb; ++block)
                    for (uint32_t diagonal = 0; diagonal <= inner; ++diagonal) {
                        sw.gemms(band, block, diagonal, first, note);
                        (void)sw.post(recon, block, note);
                        first = false;
                    }
                if (s.ip.defer) sw.final(recon, note);
            }
        }
    };
    // Expected step counts: the six scaling/slicing programs; per panel one GEMM per
    // diagonal (per K block; one per pair for the pairwise issue); per band one
    // reconstruction, or one per K block if Horner runs per K block.
    const unsigned long long nd = s.c.nslice(), nbd = s.pp.bands.size(), panels = s.pp.panels.size();
    ev::Counts expected;
    expected[ev::Head] = OZ1_FILL_PROGRAMS;
    if (single) {
        expected[ev::GemmStep] = panels * nd;
        expected[ev::ReconStep] = nbd;
    } else {
        expected[ev::GemmStep] = panels * nkb * s.ip.gemm_issues(s.c.nslice());
        expected[ev::ReconStep] = nbd * (s.ip.defer ? 1u : nkb);
    }
    // Every buffer the pass writes is filled with the sentinel first. The limb planes
    // are written only when Horner runs per K block.
    std::vector<std::shared_ptr<dist::MeshBuffer>> bufs{s.in.maxAb, s.in.maxBb, s.in.dltAb, s.in.dltBb,
                                                        s.in.shA,   s.in.shB,   s.Pbig,     br.rcn.out};
    if (br.rcn.limbs)
        for (const auto& b : {br.rcn.hi, br.rcn.mid, br.rcn.lo}) bufs.push_back(b);
    ev::whole(
        [&] {
            for (const auto& b : bufs) fill_mesh_buffer_u32(md, b, ev::SENTINEL);
        },
        [&](std::vector<uint32_t>& out) { recon_read_f64(md, br.rcn, out); }, [&] { dist::Finish(queue); }, run_once,
        expected);
}
// Scheme I GEMM of the padded problem p on device md.
Output run(Device& md, const Problem& p) {
    RunCfg c;
    c.M = p.m;
    c.N = p.n;
    c.K = p.k;
    c.Mt = p.m / 32;
    c.Nt = p.n / 32;
    // Number of slices: as requested, or selected from the input with the normalization
    // rules of the device (the radix, digit bound and normalization are the constants
    // OZ1_* of ozaki1/format.hpp).
    c.p = p.options.slices ? int(p.options.slices) : oz_fp64_p(p.a, p.b, c.M, c.N, c.K);
    const auto grid = md->compute_with_storage_grid_size();
    // Planning (host only); throws std::runtime_error if the configuration cannot be executed.
    const IssuePlan ip = plan_issue(c);
    const PanelPlan pp = plan_panels(c, ip, grid, dram_free(md));
    if (ev::active) {
        ev::active->partial_products = c.p * (c.p + 1) / 2;
        ev::active->bandwidth = pp.gU.bw;
        ev::active->cores = grid.x * grid.y;
    }
    const ReconScale rs = recon_scale(c);
    const auto partial = alloc_pbig(md, c, pp);
    InputBufs in = setup_inputs(md, c, pp, grid, p.a, p.b);
    FillStage fs = fill(md, c, pp, in);
    BandRecon br;
    std::vector<GemmCtx> contexts;
    const StageCtx stage{md, c, ip, pp, partial, in, rs};
    auto& queue = md->mesh_command_queue();
    Oz1Sweep sweep{stage, contexts};
    if (ip.single_kblock()) {
        // Per band: the GEMMs of all its panels, then its reconstruction. The program of
        // a panel is built, issued for all diagonals and released, so that the same
        // program is issued repeatedly. The diagonals need no synchronization between
        // them: the runtime arguments are fixed when a workload is enqueued.
        for (const auto& band : pp.bands) {
            for (uint32_t i = band.p0; i < band.p1; ++i) {
                auto context = build_panel(stage, i, band);
                ev::execute(ev::Gemm, [&] {
                    issue_panel_diagonals(md, context, c.L(), oz1_no_note);
                    dist::Finish(queue);
                });
            }
            auto& recon = band_recon_at(stage, br, band, partial);
            ev::execute(ev::Recon, [&] { recon_exec(md, recon, 0u); }, [&] { dist::Finish(queue); });
        }
    } else {
        // Sweep: the loop over K blocks is outside the loop over panels (the
        // reconstruction processes a whole band at once, so the GEMMs of a K block must
        // have been issued for all panels of the band before it), so the programs of
        // all panels are kept. The reconstruction, with its buffers, is built after the
        // GEMM programs (allocation order).
        contexts.reserve(pp.panels.size());
        for (const auto& band : pp.bands)
            for (uint32_t i = band.p0; i < band.p1; ++i) contexts.push_back(build_panel(stage, i, band));
        band_recon(stage, br, pp.bands.front().rows, partial);
        ev::sweep([&] {
            for (const auto& band : pp.bands) {
                auto& recon = sweep.recon(br, band);
                bool first = true;
                for (uint32_t block = 0; block < ip.nkb; ++block)
                    for (uint32_t diagonal = 0; diagonal <= sweep.inner(); ++diagonal) {
                        ev::phase(ev::Gemm, [&] {
                            sweep.gemms(band, block, diagonal, first, oz1_no_note);
                            dist::Finish(queue);
                        });
                        if (!(ip.fuse || ip.pairloop))
                            ev::phase(ev::Recon, [&] {
                                if (sweep.post(recon, block, oz1_no_note)) dist::Finish(queue);
                            });
                        first = false;
                    }
                if (ip.defer)
                    ev::phase(ev::Recon, [&] {
                        sweep.final(recon, oz1_no_note);
                        dist::Finish(queue);
                    });
            }
        });
    }
    whole_pipeline(stage, fs, br, sweep, all_panels(stage));
    std::vector<uint32_t> tiles;
    recon_read_f64(md, br.rcn, tiles);
    // src/context.cpp rejected out-of-range input on the host before the device ran. The
    // scaling stage flags row/column maxima outside the range of the device normalization
    // all the same; the flags are read only now, after the result, as a defensive check.
    expdelta_check(md, fs.edA, "rows of A");
    expdelta_check(md, fs.edB, "columns of B");
    Info info{Scheme::I, TTGEMM_PUBLIC_FORMAT};
    info.slices = c.p;
    return {untilize_f64(tiles, c.M, c.N), info};
}
}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
