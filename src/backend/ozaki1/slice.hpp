// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme I: slicing stage (the operands are split into slices on the device).
//
// The host uploads the FP64 operands as they are, in tile order. On the device, the
// reader (common/kernels/slice_f64_reader.cpp) converts each FP64 tile to triple-float
// and applies the power-of-two normalization of the scaling stage (per row of A, per
// column of B) by adding the integer exponent deltas, in the same pass. The slicer
// (ozaki1/kernels/slice_tf_num.cpp, or slice_tf_sm8.cpp for INT8) extracts nslice signed
// integer slices from each tile, and the writer (ozaki1/kernels/slice_writer.cpp) stores
// them directly in the slice-concatenated buffer that the partial-product GEMMs read. No
// conversion, normalization pass or intermediate buffer is left on the host.
// This header provides SliceCtx / slice_build (the program) and SliceRun / slice_set_args
// / slice_run (one execution over an operand).

#include "ozaki1/format.hpp"
#include "ozaki1/kernels/args.h"
#include "backend_interface.hpp"
#include "common/bits.hpp"
#include "common/slice_f64_host.hpp"
#include "common/kernels/f64_tf.h"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// The slicing program; built once and run for A and for B.
struct SliceCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;  // reader, writer, slicer (compute)
    std::vector<tt::tt_metal::CoreCoord> cores;         // row-major, the order of full_grid_span

    // Number of slices the slicer was compiled for; slice_set_args passes the same count
    // to the writer (the writer would otherwise wait for slices that never arrive).
    uint32_t nslice = 0;
};

// Builds the slicing program on the whole compute grid: reader -> slicer -> writer, with
// nslice slices per input tile. The slicer expands with the radix OZ1_RADIX and clamps the
// digits to +-OZ1_DMAX, the digit bound the planner used for the exactness conditions.
//   refF64, refDlt, refDst  buffers with the layout of the FP64 operand, the exponent
//                deltas and the slice-concatenated output. They only provide the
//                compile-time TensorAccessorArgs; the buffers of a run are runtime
//                arguments (SliceRun).
static SliceCtx slice_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, uint32_t nslice,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& refF64,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& refDlt,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& refDst) {
    namespace tm = tt::tt_metal;
    using tt::DataFormat;
    tm::distributed::MeshWorkload workload;
    auto device_range = tm::distributed::MeshCoordinateRange(md->shape());
    tm::Program program{};
    const uint32_t f32_tile = tt::tile_size(DataFormat::Float32);

    // Only the core order and range are used here; the tiles per core are set for each
    // run from the number of input tiles (slice_run).
    const GridSpan gs = full_grid_span(md, 0u);
    const tm::CoreRange all_cores = gs.range();

    // CB indices: common/kernels/slice_f64_args.h (reader output, residual, reader work
    // areas, mailbox) and ozaki1/kernels/args.h (slice output). The order of creation
    // determines the L1 layout and must not be changed.
    namespace sa = slice_f64_args;
    namespace oa = oz1_args;
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    make_cb(program, all_cores, cbi(sa::CB_HI), DataFormat::Float32, 4, f32_tile);     // input: high limb
    make_cb(program, all_cores, cbi(sa::CB_CRAW), DataFormat::Float32, 4, f32_tile);   // input: low limb
    make_cb(program, all_cores, cbi(sa::CB_SCALE), DataFormat::Float32, 4, f32_tile);  // input: scale of the low limb
    make_cb(program, all_cores, cbi(sa::CB_F64), DataFormat::Float32, 1, OZ_F64_TILE_BYTES);  // reader work: FP64 tile
    make_cb(program, all_cores, cbi(sa::CB_DLT), DataFormat::Int32, 1, f32_tile);     // reader work: exponent deltas
    make_cb(program, all_cores, cbi(sa::CB_BRAW), DataFormat::Float32, 8, f32_tile);  // input: middle limb

    // A slice is a signed integer within the digit bound. Most formats write it in the
    // GEMM input format; INT8 writes sign-magnitude UInt8 and FP16 writes binary16 bit
    // patterns as UInt16, which the GEMM reads as Int8 and Float16. The written and the
    // read format must have the same tile size, or the buffer in DRAM would not match.
    TT_FATAL(tt::tile_size(OZ_FMT.slice_out) == tt::tile_size(OZ_FMT.in),
             "slice_build: the slice output format and the GEMM input format differ in tile size");
    make_cb(program, all_cores, cbi(oa::CB_SLICE_OUT), OZ_FMT.slice_out, 8, (uint32_t)tt::tile_size(OZ_FMT.slice_out));
    make_cb(program, all_cores, cbi(sa::CB_RES01), DataFormat::Float32, 8, f32_tile);  // work: residual limbs 0 and 1
    make_cb(program, all_cores, cbi(sa::CB_RES2), DataFormat::Float32, 4, f32_tile);   // work: residual limb 2

    // Mailbox through which the reader hands the conversion of the second half of each
    // FP64 tile to the writer of the same core (common/kernels/slice_f64_split.h).
    make_cb(program, all_cores, cbi(sa::CB_MBOX), DataFormat::UInt32, 1, sa::MBOX_PAGE);

    // Reader: bit-operation conversion, second half delegated to the writer.
    const std::vector<uint32_t> rargs_ct = f64_reader_ct(*refF64, *refDlt, true);
    std::vector<uint32_t> wargs_ct;
    tm::TensorAccessorArgs(*refDst).append_to(wargs_ct);
    {
        const size_t o = wargs_ct.size();
        wargs_ct.resize(o + oa::SW_CT_COUNT);
        wargs_ct[o + oa::SW_CT_SPLIT] = 1u;
    }

    // The reader is the same kernel for every format: the data formats of the CBs are set
    // by the host, so the kernel has no format-dependent path.
    auto rk = tm::CreateKernel(program, ::ttgemm::detail::kernel_path("common/kernels/slice_f64_reader.cpp"), all_cores,
                               tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                                                      .noc = tm::NOC::RISCV_1_default,
                                                      .compile_args = rargs_ct});
    auto wk = tm::CreateKernel(program, ::ttgemm::detail::kernel_path("ozaki1/kernels/slice_writer.cpp"), all_cores,
                               tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                                                      .noc = tm::NOC::RISCV_0_default,
                                                      .compile_args = wargs_ct});

    // The kernels receive the radix and the digit bound as FP32 bit patterns. The digit
    // bound is passed separately from the radix (INT8: radix 254, digits +-127).
    const uint32_t rb = f32_bits((float)OZ1_RADIX);
    const uint32_t db = f32_bits((float)OZ1_DMAX), ndb = f32_bits(-(float)OZ1_DMAX);
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    for (uint32_t idx : {sa::CB_HI, sa::CB_CRAW, sa::CB_SCALE, sa::CB_RES01, sa::CB_BRAW, sa::CB_RES2})
        u2d[idx] = tm::UnpackToDestMode::UnpackToDestFp32;
    std::vector<uint32_t> cargs_ct(oa::SC_COUNT);
    cargs_ct[oa::SC_NSLICE] = nslice;
    cargs_ct[oa::SC_RADIX] = rb;
    cargs_ct[oa::SC_SF_IN] = 0u;  // soft-float reader (diagnostic selector of the kernel); always 0 in the library
    cargs_ct[oa::SC_DMAX] = db;
    cargs_ct[oa::SC_NDMAX] = ndb;
    cargs_ct[oa::SC_OUT_HALF] = (uint32_t)OZ_FMT.out_half;
    static_assert(oa::SC_COUNT == 6, "a field was added to SlicerCT: fill it here");

    // Two slicers by digit encoding: sign-magnitude UInt8 for INT8, the digit value itself
    // for the other formats (Oz1Param::signmag_slices). dst_full_sync_en makes the 8
    // destination registers one bank; the triple-float residual and the work values use
    // all 8 at once.
    auto ck = tm::CreateKernel(
        program,
        OZ1_PARAM.signmag_slices ? ::ttgemm::detail::kernel_path("ozaki1/kernels/slice_tf_sm8.cpp")
                                 : ::ttgemm::detail::kernel_path("ozaki1/kernels/slice_tf_num.cpp"),
        all_cores,
        tm::ComputeConfig{
            .fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = cargs_ct});
    SliceCtx s;
    for (const tm::CoreCoord& core : gs.cores) {
        tm::SetRuntimeArgs(program, rk, core, std::vector<uint32_t>(sa::R_COUNT, 0u));
        tm::SetRuntimeArgs(program, ck, core, std::vector<uint32_t>(oa::SR_COUNT, 0u));
        tm::SetRuntimeArgs(program, wk, core, std::vector<uint32_t>(oa::SW_COUNT, 0u));
        s.cores.push_back(core);
    }
    workload.add_program(device_range, std::move(program));
    s.workload = std::move(workload);
    s.range = device_range;
    s.rk = rk;
    s.wk = wk;
    s.ck = ck;
    s.nslice = nslice;
    return s;
}

// Input, output and layout of one slicing run (one operand).
//   mode 0 = A (M x K):  s1 = Kto, the K tiles of one slice; the exponent deltas are per
//                        row (delta index = tile index / Kto).
//   mode 1 = B (K x N):  s1 = Kto, s2 = Nt; the exponent deltas are per column (delta
//                        index = tile index % Nt).
// Kbt is the number of K tiles of one K block (divides Kto; Kto without blocking). The
// writer uses it to lay the output out block-major, so that one run produces the
// concatenated slices of every K block (ozaki1/kernels/slice_writer.cpp).
struct SliceRun {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> f64, dlt,
        dst;              // FP64 operand, exponent deltas, sliced output
    uint32_t ntiles = 0;  // number of input tiles
    uint32_t s1 = 0, s2 = 0, mode = 0, Kbt = 0;
};

// Sets the runtime arguments of reader, slicer and writer for run j, with cnt[c] input
// tiles assigned to core s.cores[c] in order (layouts: ozaki1/kernels/args.h and
// common/kernels/slice_f64_args.h). Does not enqueue.
static inline void slice_set_args(SliceCtx& s, const SliceRun& j, const std::vector<uint32_t>& cnt) {
    TT_FATAL(cnt.size() == s.cores.size(), "slice_set_args: {} tile counts for {} cores", cnt.size(), s.cores.size());
    TT_FATAL(j.Kbt > 0 && j.s1 % j.Kbt == 0, "slice_set_args: the K block of {} tiles does not divide Kto={}", j.Kbt,
             j.s1);
    tt::tt_metal::Program& prog = s.workload.get_programs().at(s.range);
    const uint32_t dlt_stride = (j.mode == 0) ? j.s1 : j.s2;
    namespace oa = oz1_args;
    uint32_t start = 0;
    for (size_t c = 0; c < s.cores.size(); ++c) {
        const uint32_t nt = cnt[c];
        tt::tt_metal::SetRuntimeArgs(
            prog, s.rk, s.cores[c],
            f64_reader_rt((uint32_t)j.f64->address(), (uint32_t)j.dlt->address(), nt, start, dlt_stride, j.mode));
        std::vector<uint32_t> ca(oa::SR_COUNT);
        ca[oa::SR_NTILES] = nt;
        tt::tt_metal::SetRuntimeArgs(prog, s.ck, s.cores[c], ca);
        std::vector<uint32_t> wa(oa::SW_COUNT);
        wa[oa::SW_DST_ADDR] = (uint32_t)j.dst->address();
        wa[oa::SW_NUM] = nt;
        wa[oa::SW_START] = start;
        wa[oa::SW_NSLICE] = s.nslice;
        wa[oa::SW_S1] = j.s1;
        wa[oa::SW_S2] = j.s2;
        wa[oa::SW_MODE] = j.mode;
        wa[oa::SW_KBT] = j.Kbt;
        static_assert(oa::SW_COUNT == 8 && oa::SR_COUNT == 1,
                      "a field was added to SliceWriterRT or SlicerRT: fill it here");
        tt::tt_metal::SetRuntimeArgs(prog, s.wk, s.cores[c], wa);
        start += nt;
    }
}

// Enqueues one slicing run over the j.ntiles input tiles, distributed evenly over the
// cores from the first tile on (full_grid_span, as in the other stages). Does not wait.
static inline void slice_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, SliceCtx& s,
                             const SliceRun& j) {
    slice_set_args(s, j, full_grid_span(md, j.ntiles).counts());
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), s.workload, false);
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
