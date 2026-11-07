// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: reconstruction stage (triple-float Horner evaluation and FP64 assembly
// on the device).
//
// The buffer of partial sums holds one plane per diagonal: plane d starts at tile
// d*pstride, where pstride is the number of tiles of one row band. The reconstruction
// folds the planes by Horner's rule into S = sum_d radix^(L-d) P_d as a triple-float
// (hi, mid, lo), and the writer assembles
//     C = S * radix^-(L+2) * 2^-(dA+dB)
// into FP64, where dA and dB are the exponent deltas of the row and the column from the
// scaling stage. The host only reorders the result.
// This header provides the constants passed to the kernels (ReconScale), the programs
// (ReconProg, ReconCtx, recon_build_prog, recon_build) and their use (recon_exec,
// recon_set_out_off, recon_read_f64).
//
// With K blocking and no deferred reconstruction, three programs of the same shape are
// built:
//   pr[0]  no accumulation; initializes the result (K block 0)
//   pr[1]  reads the current (hi, mid, lo), adds, and writes back (middle K blocks)
//   pr[2]  adds and assembles FP64 (last K block)
// Horner's rule is linear in P_d, so the total sum_b sum_d radix^(L-d) P_d^(b) may be
// folded per K block b and then summed. Otherwise only pr[0] is built and it assembles
// FP64 directly.
// The in-place accumulation is safe: the tile ranges of the cores are disjoint, and for
// one tile the CBs serialize reader -> compute -> writer, so a tile has been read
// completely before it is overwritten.
//
// There are two sets of kernels (ozaki1/kernels/), selected by the format of the partial
// sums:
//   recon_reader_f32in.cpp, recon_tf_f32in.cpp  FP32 partial sums, added one at a time
//     (reconstruction that is not deferred);
//   recon_reader_i32in.cpp, recon_tf_i32in.cpp  Int32 partial sums, each split into two
//     exact FP32 values: the INT8 path (sign-magnitude) and the Int32 accumulation of
//     the deferred reconstruction (two's complement).

#include "ozaki1/format.hpp"
#include "ozaki1/kernels/args.h"
#include "backend_interface.hpp"
#include "common/kernels/f64_tf.h"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Radix and final scale in the form the reconstruction kernels take them (computed on
// the host by recon_scale in ozaki1/plan.hpp).
struct ReconScale {
    uint32_t radixbits = 0;       // radix as the bit pattern of a float
    uint32_t Whi = 0, Wlo = 0;    // 64-bit significand of radix^-(L+2)
    int32_t qw = 0;               // its exponent
    uint32_t Wdlo = 0, Wdhi = 0;  // radix^-(L+2) as a double (read only by the soft-float variant of the writer)
    uint32_t nsmall = 0;          // leading diagonals that one typecast moves to FP32 exactly (recon_nsmall)
};

// One reconstruction program.
struct ReconProg {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::KernelHandle wk = 0;  // writer
    bool ok = false;

    // Writer runtime arguments per core as built, kept so that recon_set_out_off can
    // replace the output offset of a row band.
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<std::vector<uint32_t>> wa;
};

// The reconstruction of row bands of one height.
struct ReconCtx {
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};

    ReconProg pr[3];  // 0 = initialize, 1 = accumulate (middle K block), 2 = accumulate and assemble FP64

    // Limb planes and the FP64 result. The limbs are read and written only when Horner
    // runs per K block (limbs == true); hi, mid and lo are then separate planes of one
    // row band. Otherwise the three share a single one-tile buffer that exists only to
    // form the TensorAccessorArgs of the kernels and is never accessed.
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> hi, mid, lo, out;
    bool limbs = false;
    uint32_t ntiles = 0;      // tiles folded by one execution (one row band)
    uint32_t ntiles_out = 0;  // tiles of the whole output buffer (for the read-back)
};

// Sets the output offset [tiles] of a row band in all programs of rc. Not needed when
// the output is a single band.
static void recon_set_out_off(ReconCtx& rc, uint32_t off) {
    for (auto& pr : rc.pr) {
        if (!pr.ok) continue;
        tt::tt_metal::Program& prog = pr.workload.get_programs().at(rc.range);
        for (size_t i = 0; i < pr.cores.size(); i++) {
            auto a = pr.wa[i];
            a[oz1_args::RW_OUT_OFF] = off;
            tt::tt_metal::SetRuntimeArgs(prog, pr.wk, pr.cores[i], a);
        }
    }
}

// Buffers one reconstruction program reads and writes.
struct ReconBufs {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> P;            // partial sums per diagonal
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> hi, mid, lo;  // limb planes (also input if acc)
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out;          // FP64 output
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltA, dltB;   // exponent deltas of rows / columns
};

// What distinguishes the programs of a ReconCtx.
struct ReconVariant {
    uint32_t acc = 0;         // 1: read the limbs of the previous K blocks and add them (FP32 input only)
    uint32_t emit_limbs = 0;  // 1: write the three limbs (every K block but the last, when Horner runs per block)
    uint32_t emit_f64 = 0;    // 1: assemble FP64 (last K block)
    bool defer = false;       // input is the Int32 accumulation of the deferred reconstruction (two's complement)
    uint32_t nsmall = 0;      // with defer: leading diagonals read with a single typecast
};

// Builds one reconstruction program. Mt x Nt is the output folded by one execution (one
// row band) and pstride the distance between the planes of consecutive diagonals, both
// in tiles.
static ReconProg recon_build_prog(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                  const tt::tt_metal::distributed::MeshCoordinateRange& device_range,
                                  const ReconBufs& bf, uint32_t Mt, uint32_t Nt, uint32_t ndiag, const ReconScale& rs,
                                  const ReconVariant& v, uint32_t pstride) {
    namespace tm = tt::tt_metal;
    using tt::DataFormat;
    tm::distributed::MeshWorkload workload;
    tm::Program program{};

    const uint32_t ntiles = Mt * Nt;
    const uint32_t f32_tile = tt::tile_size(DataFormat::Float32);

    // With defer the input is the Int32 accumulation, which is read like the INT8 partial
    // sums: every Int32 is split into two exact FP32 values.
    const bool i32_in = OZ1_INT32_PARTIAL || v.defer;

    // The representation of the Int32 depends on its origin: sign-magnitude if packed by
    // the matrix engine (INT8 partial sums), two's complement if produced by a typecast
    // from FP32 (deferred accumulation). The mask of the reader and the sign handling of
    // the compute kernel must agree, so one flag selects both.
    const uint32_t signmag = (i32_in && !v.defer) ? 1u : 0u;

    const GridSpan gs = full_grid_span(md, ntiles);
    const tm::CoreRange all_cores = gs.range();

    // CB indices: ozaki1/kernels/args.h. The order of creation determines the L1 layout
    // and must not be changed.
    namespace oa = oz1_args;
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    make_cb(program, all_cores, cbi(oa::CB_RECON_IN), i32_in ? DataFormat::Int32 : OZ_FMT.out, ndiag * 2, f32_tile);
    if (i32_in)
        make_cb(program, all_cores, cbi(oa::CB_MASK), DataFormat::Int32, 2,
                f32_tile);  // mask tiles that split an Int32
    make_cb(program, all_cores, cbi(oa::CB_LIMB_HI), DataFormat::Float32, 2, f32_tile);
    make_cb(program, all_cores, cbi(oa::CB_LIMB_LO), DataFormat::Float32, 2, f32_tile);
    make_cb(program, all_cores, cbi(oa::CB_LIMB_MID), DataFormat::Float32, 2, f32_tile);
    make_cb(program, all_cores, cbi(oa::CB_F64), DataFormat::Float32, 1,
            OZ_F64_TILE_BYTES);                                                      // writer work: FP64 output tile
    make_cb(program, all_cores, cbi(oa::CB_DLT_A), DataFormat::Int32, 1, f32_tile);  // writer work: row deltas
    make_cb(program, all_cores, cbi(oa::CB_DLT_B), DataFormat::Int32, 1, f32_tile);  // writer work: column deltas

    if (v.acc) {  // limbs accumulated up to the previous K block
        make_cb(program, all_cores, cbi(oa::CB_ACC), DataFormat::Float32, 6, f32_tile);
    }
    std::vector<uint32_t> rargs_ct;

    // The Int32 reader takes SIGNMAG first. It builds the mask tiles in L1 itself, so no
    // TensorAccessorArgs are passed for them.
    if (i32_in) {
        rargs_ct.resize(oa::RI_CT_COUNT);
        rargs_ct[oa::RI_CT_SIGNMAG] = signmag;
    }
    tm::TensorAccessorArgs(*bf.P).append_to(rargs_ct);
    if (!i32_in) {
        // The accumulated limbs are in the order hi, mid, lo for the reader (hi, lo, mid
        // for the writer; see the top of ozaki1/kernels/args.h).
        tm::TensorAccessorArgs(*(v.acc ? bf.hi : bf.P)).append_to(rargs_ct);
        tm::TensorAccessorArgs(*(v.acc ? bf.mid : bf.P)).append_to(rargs_ct);
        tm::TensorAccessorArgs(*(v.acc ? bf.lo : bf.P)).append_to(rargs_ct);
    }

    // TensorAccessorArgs of the writer in the order of WriterTA (hi, lo, mid, out, dltA, dltB).
    std::vector<uint32_t> wargs_ct;
    {
        std::shared_ptr<tm::distributed::MeshBuffer> ta[oa::TA_COUNT];
        ta[oa::TA_HI] = bf.hi;
        ta[oa::TA_LO] = bf.lo;
        ta[oa::TA_MID] = bf.mid;
        ta[oa::TA_OUT] = bf.out;
        ta[oa::TA_DLT_A] = bf.dltA;
        ta[oa::TA_DLT_B] = bf.dltB;
        for (const auto& b : ta) tm::TensorAccessorArgs(*b).append_to(wargs_ct);
    }

    auto rk = tm::CreateKernel(program,
                               i32_in ? ::ttgemm::detail::kernel_path("ozaki1/kernels/recon_reader_i32in.cpp")
                                      : ::ttgemm::detail::kernel_path("ozaki1/kernels/recon_reader_f32in.cpp"),
                               all_cores,
                               tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                                                      .noc = tm::NOC::RISCV_1_default,
                                                      .compile_args = rargs_ct});
    auto wk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki1/kernels/recon_f64_writer.cpp"), all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                               .noc = tm::NOC::RISCV_0_default,
                               // The trailing WriterCT arguments: whether to write the limbs and whether
                               // to assemble FP64 (the assembly is costly, so with several K blocks only
                               // the last one does it), and two diagnostic selectors of the kernel.
                               .compile_args = [&] {
                                   auto a = wargs_ct;
                                   const size_t o = a.size();
                                   a.resize(o + oa::RW_CT_COUNT);
                                   a[o + oa::RW_CT_EMIT_LIMBS] = v.emit_limbs;
                                   a[o + oa::RW_CT_EMIT_F64] = v.emit_f64;
                                   a[o + oa::RW_CT_SF] = 0u;     // soft-float assembly; always 0 in the library
                                   a[o + oa::RW_CT_PROBE] = 0u;  // diagnostic selector; always 0 in the library
                                   static_assert(oa::RW_CT_COUNT == 4, "a field was added to WriterCT: fill it here");
                                   return a;
                               }()});

    // The input CB holds signed FP32 partial sums (or Int32), and the accumulation CB
    // the triple-float of the previous K blocks: both must be unpacked to the destination
    // register as 32-bit values, without the default truncation.
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[oa::CB_RECON_IN] = tm::UnpackToDestMode::UnpackToDestFp32;
    u2d[oa::CB_ACC] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (i32_in) u2d[oa::CB_MASK] = tm::UnpackToDestMode::UnpackToDestFp32;
    std::vector<uint32_t> cargs_ct(oa::RC_COUNT);
    cargs_ct[oa::RC_NDIAG] = ndiag;
    cargs_ct[oa::RC_IN_FMT] = (uint32_t)(i32_in ? DataFormat::Int32 : OZ_FMT.out);
    cargs_ct[oa::RC_F32] = (uint32_t)DataFormat::Float32;
    cargs_ct[oa::RC_RADIX] = rs.radixbits;
    cargs_ct[oa::RC_ACC] = v.acc;
    cargs_ct[oa::RC_SIGNMAG] = signmag;
    cargs_ct[oa::RC_NSINGLE_CAST] = signmag ? 0u : std::min(v.nsmall, ndiag);
    static_assert(oa::RC_COUNT == 7, "a field was added to ReconCT: fill it here");

    // dst_full_sync_en is required: the multiplication by the radix and the decoding use
    // 8 destination registers (0..7). With the default half synchronization the FP32
    // registers form two banks of 4, and the computation of the next tile could overwrite
    // registers 4..7 while the packer still reads the previous tile.
    auto ck = tm::CreateKernel(
        program,
        i32_in ? ::ttgemm::detail::kernel_path("ozaki1/kernels/recon_tf_i32in.cpp")
               : ::ttgemm::detail::kernel_path("ozaki1/kernels/recon_tf_f32in.cpp"),
        all_cores,
        tm::ComputeConfig{
            .fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = cargs_ct});

    std::vector<tm::CoreCoord> wcores;
    std::vector<std::vector<uint32_t>> wargs_rt;
    for (size_t c = 0; c < gs.cores.size(); ++c) {
        const tm::CoreCoord core = gs.cores[c];
        const uint32_t nt = gs.count(c), start = gs.begin(c);

        // The Int32 reader reads the arguments up to RR_SLOT_STRIDE only: it has no
        // accumulation input (the deferred reconstruction runs once at the end, and the
        // INT8 path has a single K block).
        std::vector<uint32_t> ra(i32_in ? oa::RR_I32_COUNT : oa::RR_COUNT);
        ra[oa::RR_SRC_ADDR] = (uint32_t)bf.P->address();
        ra[oa::RR_NUM] = nt;
        ra[oa::RR_START] = start;
        ra[oa::RR_NDIAG] = ndiag;
        ra[oa::RR_SLOT_STRIDE] = pstride;
        if (!i32_in) {
            ra[oa::RR_ACC_HI] = v.acc ? (uint32_t)bf.hi->address() : 0u;
            ra[oa::RR_ACC_MID] = v.acc ? (uint32_t)bf.mid->address() : 0u;
            ra[oa::RR_ACC_LO] = v.acc ? (uint32_t)bf.lo->address() : 0u;
        }
        tm::SetRuntimeArgs(program, rk, core, ra);
        std::vector<uint32_t> ca(oa::RCR_COUNT);
        ca[oa::RCR_NTILES] = nt;
        tm::SetRuntimeArgs(program, ck, core, ca);
        std::vector<uint32_t> wa(oa::RW_COUNT);
        wa[oa::RW_HI_ADDR] = (uint32_t)bf.hi->address();
        wa[oa::RW_NUM] = nt;
        wa[oa::RW_START] = start;
        wa[oa::RW_LO_ADDR] = (uint32_t)bf.lo->address();
        wa[oa::RW_MID_ADDR] = (uint32_t)bf.mid->address();
        wa[oa::RW_OUT_ADDR] = (uint32_t)bf.out->address();
        wa[oa::RW_DLT_A_ADDR] = (uint32_t)bf.dltA->address();
        wa[oa::RW_DLT_B_ADDR] = (uint32_t)bf.dltB->address();
        wa[oa::RW_NT] = Nt;
        wa[oa::RW_WHI] = rs.Whi;
        wa[oa::RW_WLO] = rs.Wlo;
        wa[oa::RW_QW] = (uint32_t)rs.qw;
        wa[oa::RW_WDLO] = rs.Wdlo;
        wa[oa::RW_WDHI] = rs.Wdhi;
        wa[oa::RW_OUT_OFF] = 0u;  // output offset of a row band (recon_set_out_off replaces it)
        static_assert(oa::RW_COUNT == 15 && oa::RR_COUNT == 8 && oa::RCR_COUNT == 1,
                      "a field was added to WriterRT, ReaderRT or ReconRT: fill it here");
        tm::SetRuntimeArgs(program, wk, core, wa);
        wargs_rt.push_back(std::move(wa));
        wcores.push_back(core);
    }
    workload.add_program(device_range, std::move(program));
    ReconProg pr;
    pr.workload = std::move(workload);
    pr.wk = wk;
    pr.ok = true;
    pr.cores = std::move(wcores);
    pr.wa = std::move(wargs_rt);
    return pr;
}

// Buffers that the reconstruction of a second band height (the last, shorter band)
// shares with the first one. Empty for the first, which allocates them.
struct ReconShared {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out, hi, mid, lo;
};

// Allocates the output buffers and builds the reconstruction programs (nothing is
// executed; the kernels are JIT-built at the first recon_exec).
//   Pbig        partial sums per diagonal (input)
//   Mt, Nt      output folded by one execution (one row band) [tiles]
//   with_acc    the GEMMs are issued as a sweep (IssuePlan::single_kblock() is false)
//   defer       deferred reconstruction: the input is the Int32 accumulation over all K blocks
//   pstride     distance between the planes of consecutive diagonals [tiles]
//   ntiles_out  tiles of the whole FP64 output (the output buffer is shared by all bands)
// The accumulating programs pr[1] and pr[2] and full-size limb planes exist only if
// with_acc && !defer (ReconCtx::limbs).
static ReconCtx recon_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig, uint32_t Mt,
                            uint32_t Nt, uint32_t ndiag, const ReconScale& rs, bool with_acc, bool defer,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& dltA,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& dltB, uint32_t pstride,
                            uint32_t ntiles_out, const ReconShared& share) {
    namespace dist = tt::tt_metal::distributed;
    ReconCtx rc;
    rc.range = dist::MeshCoordinateRange(md->shape());
    rc.ntiles = Mt * Nt;
    rc.ntiles_out = ntiles_out;
    const uint32_t f32_tile = tt::tile_size(tt::DataFormat::Float32);

    // The limb planes are read and written only when Horner runs per K block (pr[0]
    // writes them, pr[1] and pr[2] accumulate). In the other configurations no kernel
    // accesses them, so a single one-tile buffer is created for the TensorAccessorArgs of
    // the writer and shared by the three limbs, which saves three full planes of DRAM.
    rc.limbs = with_acc && !defer;
    dist::ReplicatedBufferConfig bO{.size = (size_t)f32_tile * (rc.limbs ? std::max(pstride, rc.ntiles) : 1u)};
    dist::DeviceLocalBufferConfig dO{.page_size = f32_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    if (rc.limbs) {
        rc.hi = share.hi ? share.hi : dist::MeshBuffer::create(bO, dO, md.get());
        rc.lo = share.lo ? share.lo : dist::MeshBuffer::create(bO, dO, md.get());
        rc.mid = share.mid ? share.mid : dist::MeshBuffer::create(bO, dO, md.get());
    } else {
        rc.hi = share.hi ? share.hi : dist::MeshBuffer::create(bO, dO, md.get());
        rc.mid = share.mid ? share.mid : rc.hi;
        rc.lo = share.lo ? share.lo : rc.hi;
    }

    // FP64 result, 8 bytes per element in tile order. With row bands the first
    // reconstruction allocates the whole output and passes it to the second.
    dist::ReplicatedBufferConfig bF{.size = (size_t)OZ_F64_TILE_BYTES * rc.ntiles_out};
    dist::DeviceLocalBufferConfig dFo{.page_size = OZ_F64_TILE_BYTES, .buffer_type = tt::tt_metal::BufferType::DRAM};
    rc.out = share.out ? share.out : dist::MeshBuffer::create(bF, dFo, md.get());

    // With a single K block, or with defer (one run over the Int32 accumulation at the
    // end), pr[0] assembles FP64 directly. Otherwise the K blocks before the last leave
    // limbs and the last one assembles FP64.
    const ReconBufs bf{Pbig, rc.hi, rc.mid, rc.lo, rc.out, dltA, dltB};
    ReconVariant v0;
    v0.emit_limbs = rc.limbs ? 1u : 0u;
    v0.emit_f64 = rc.limbs ? 0u : 1u;
    v0.defer = defer;
    v0.nsmall = defer ? rs.nsmall : 0u;
    rc.pr[0] = recon_build_prog(md, rc.range, bf, Mt, Nt, ndiag, rs, v0, pstride);
    if (rc.limbs) {
        ReconVariant v1;
        v1.acc = 1u;
        v1.emit_limbs = 1u;  // middle K blocks
        ReconVariant v2;
        v2.acc = 1u;
        v2.emit_f64 = 1u;  // last K block
        rc.pr[1] = recon_build_prog(md, rc.range, bf, Mt, Nt, ndiag, rs, v1, pstride);
        rc.pr[2] = recon_build_prog(md, rc.range, bf, Mt, Nt, ndiag, rs, v2, pstride);
    }
    return rc;
}

// Enqueues the reconstruction of one K block: which = 0 initialize, 1 accumulate (middle
// K block), 2 accumulate and assemble FP64 (last K block). Does not wait.
static void recon_exec(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ReconCtx& rc, uint32_t which) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), rc.pr[which].workload, false);
}

// Reads the FP64 result back to the host, in tile order (two uint32 words per element).
// Assembly and scaling were done on the device; the host only has to reorder.
static void recon_read_f64(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ReconCtx& rc,
                           std::vector<uint32_t>& Ot) {
    auto& cq = md->mesh_command_queue();
    Ot.resize((size_t)rc.ntiles_out * (OZ_F64_TILE_BYTES / 4u));
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, Ot, rc.out, true);
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
