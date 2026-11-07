// SPDX-License-Identifier: Apache-2.0
#pragma once
// Transfer of the operands and the scaling stage, the first stage of the pipeline of both Ozaki schemes
// (scaling -> slicing -> partial-product GEMMs -> accumulation / composition -> reconstruction).
//
// Before slicing, both schemes find the largest magnitude of every row of A and every column of B,
// convert it to BF16 in the direction that the path requires (rounded down, rounded up, or moved one
// power of two up; CEIL of scale_f64_reader.cpp), round that BF16 value up to a power of two 2^e, and
// derive the exponent shift of the normalization from e. On the paths that round up, 2^e is at least the
// maximum; on the paths that round down, a maximum just above a power of two 2^k (by less than a
// relative 2^-7) gives 2^k, which is below the maximum (the excess is covered by the margin that those
// paths keep in the bound of the slices). Scaling by a power of two makes the normalization in the
// slicer an exponent shift only, which is exact.
//
// The stage runs entirely on the device: common/kernels/scale_f64_reader.cpp (one instance on each of
// the two data-movement RISC-V processors; no compute kernel) finds the maxima directly on the FP64 bit
// patterns and writes them as BF16, and common/kernels/expdelta.cpp turns the maxima into "the value to
// add to the exponent" (expdelta_build / expdelta_run). Between the upload of A and B and the readback
// of C the host does not communicate with the device, so this last step is not left to the host either.
// Input outside the range of the normalization is rejected on the host by the public entry point
// (src/context.cpp) before the device runs; the device records such input in flags all the same, which
// expdelta_check examines after C has been read.
//
// The operands are transferred as FP64 (tilize_f64 / upload_operand_f64). The conversion from FP64 to
// triple-float is done on the device (common/kernels/slice_f64_reader.cpp), and the normalization is
// fused into the same pass as an integer added to the FP64 exponent. The transfer helpers only reorder
// elements and do no arithmetic on them.

#include "common/gemm_host.hpp"
#include "backend_interface.hpp"
#include "common/kernels/scale_args.h"
#include "common/kernels/f64_tf.h"
#include "common/kernels/tile_nf.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/format.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// ===== Transfer of the operands as FP64 =====
// Reorders a row-major R x C matrix (R and C multiples of 32) into tile order: tiles row-major, and
// within a tile the element order nf(r, c) of common/kernels/tile_nf.h. One element is 8 bytes, so a
// page of 1024 elements is 8192 bytes (2048 words); the low word comes first (the memory image of a
// little-endian double).
inline std::vector<uint32_t> tilize_f64(const std::vector<double>& s, uint32_t R, uint32_t C) {
    const uint32_t Rt = R / 32, Ct = C / 32;
    std::vector<uint32_t> o((size_t)Rt * Ct * 2048);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t tr = 0; tr < (int64_t)Rt; tr++)
        for (uint32_t tc = 0; tc < Ct; tc++) {
            uint32_t* t = &o[((size_t)tr * Ct + tc) * 2048];
            for (uint32_t r = 0; r < 32; r++)
                for (uint32_t c = 0; c < 32; c++) {
                    uint64_t u;
                    const double v = s[(size_t)(tr * 32 + r) * C + (tc * 32 + c)];
                    std::memcpy(&u, &v, 8);
                    const uint32_t off = nf(r, c);
                    t[2 * off] = (uint32_t)u;
                    t[2 * off + 1] = (uint32_t)(u >> 32);
                }
        }
    return o;
}

// Inverse of tilize_f64: tile-ordered FP64 (two words per element, low word first) back to a row-major
// R x C matrix. Used on the result that the device has assembled as FP64.
inline std::vector<double> untilize_f64(const std::vector<uint32_t>& t, uint32_t R, uint32_t C) {
    const uint32_t Rt = R / 32, Ct = C / 32;
    std::vector<double> o((size_t)R * C, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t tr = 0; tr < (int64_t)Rt; tr++)
        for (uint32_t tc = 0; tc < Ct; tc++) {
            const uint32_t* s = &t[((size_t)tr * Ct + tc) * 2048];
            for (uint32_t r = 0; r < 32; r++)
                for (uint32_t c = 0; c < 32; c++) {
                    const uint32_t off = nf(r, c);
                    const uint64_t u = (uint64_t)s[2 * off] | ((uint64_t)s[2 * off + 1] << 32);
                    std::memcpy(&o[(size_t)(tr * 32 + r) * C + (tc * 32 + c)], &u, 8);
                }
        }
    return o;
}

// Writes an FP64 operand to one DRAM buffer and waits for completion. The device kernels
// (scale_f64_reader.cpp, slice_f64_reader.cpp) read it from there directly.
inline void upload_operand_f64(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const std::vector<double>& X,
                               uint32_t R, uint32_t C, std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buf) {
    auto& cq = md->mesh_command_queue();
    auto t = tilize_f64(X, R, C);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf, t, false);
    tt::tt_metal::distributed::Finish(cq);
}

// ===== Exponent deltas (on the device) =====
// Program that turns the tiles of maxima into the tiles of "values to add to the exponent" (see
// common/kernels/expdelta.cpp).
struct ExpDeltaCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> flag;  // out-of-range flags
    uint32_t nslot = 0;                                           // number of flag slots (= number of cores)
};

// Builds the program for ntiles tiles of maxima in maxbuf (BF16), writing the deltas to dltbuf (Int32).
// rowwise selects the row direction (A) or the column direction (B); shift is the exponent shift of the
// normalization of the path. The program is placed on the whole grid (the grid of full_grid_span, which
// also distributes the tiles). Every core has a flag slot (one Int32 tile) of its own, so that the result
// does not depend on how overlapping NoC writes behave; flagbuf must have one tile per core of the grid.
inline ExpDeltaCtx expdelta_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> maxbuf,
                                  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltbuf,
                                  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> flagbuf, uint32_t ntiles,
                                  bool rowwise, int32_t shift) {
    namespace tm = tt::tt_metal;
    // With ntiles = 0 the distribution of the tiles would be meaningless.
    TT_FATAL(ntiles > 0, "expdelta_build: zero output tiles (empty matrices are not accepted)");
    const uint32_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t i32t = tt::tile_size(tt::DataFormat::Int32);
    // The cores are enumerated row-major, and the flag slots are numbered in the same order.
    const GridSpan gs = full_grid_span(md, ntiles);
    const tm::CoreRange all_cores = gs.range();
    ExpDeltaCtx ec;
    ec.flag = flagbuf;
    ec.nslot = (uint32_t)gs.cores.size();
    TT_FATAL(flagbuf->size() >= (uint64_t)ec.nslot * i32t,
             "expdelta_build: the flag buffer of {} B is too small for the slots of {} cores", flagbuf->size(),
             ec.nslot);
    tm::distributed::MeshWorkload workload;
    tm::Program program{};
    namespace ea = expdelta_args;
    make_cb(program, all_cores, (tt::CBIndex)ea::CB_MAX, tt::DataFormat::Float16_b, 1, bf16t);
    make_cb(program, all_cores, (tt::CBIndex)ea::CB_OUT, tt::DataFormat::Int32, 1, i32t);
    std::vector<uint32_t> ct(ea::CT_COUNT);
    ct[ea::CT_ROWWISE] = rowwise ? 1u : 0u;
    tm::TensorAccessorArgs(*maxbuf).append_to(ct);
    tm::TensorAccessorArgs(*dltbuf).append_to(ct);
    tm::TensorAccessorArgs(*flagbuf).append_to(ct);
    auto k = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("common/kernels/expdelta.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = ct});
    for (size_t c = 0; c < gs.cores.size(); ++c) {
        std::vector<uint32_t> rt(ea::R_COUNT);
        rt[ea::R_MAX_ADDR] = (uint32_t)maxbuf->address();
        rt[ea::R_OUT_ADDR] = (uint32_t)dltbuf->address();
        rt[ea::R_FLAG_ADDR] = (uint32_t)flagbuf->address();
        rt[ea::R_NUM] = gs.count(c);
        rt[ea::R_START] = gs.begin(c);
        rt[ea::R_SHIFT] = (uint32_t)shift;
        rt[ea::R_SLOT] = (uint32_t)c;  // the flag slot dedicated to this core
        static_assert(ea::R_COUNT == 7, "fill the new RT entry of expdelta_args here");
        tm::SetRuntimeArgs(program, k, gs.cores[c], rt);
    }
    workload.add_program(tm::distributed::MeshCoordinateRange(md->shape()), std::move(program));
    ec.workload = std::move(workload);
    return ec;
}

// Issues the program once (nonblocking).
inline void expdelta_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ExpDeltaCtx& ec) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), ec.workload, false);
}

// Reads the flag word (the OR over all slots; blocking). Bit 0: a maximum exceeds the BF16 range; bit 1:
// a nonzero maximum is below 2^-126; bit 2: an element is Inf or NaN (expdelta.cpp).
inline uint32_t expdelta_flags(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const ExpDeltaCtx& ec) {
    const uint32_t wpt = tt::tile_size(tt::DataFormat::Int32) / 4;
    std::vector<uint32_t> f(wpt * ec.nslot);
    auto fb = ec.flag;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(md->mesh_command_queue(), f, fb, true);
    uint32_t bits = 0;
    for (uint32_t i = 0; i < ec.nslot; i++) bits |= f[(size_t)i * wpt];
    return bits;
}

// Call after C has been read. Throws std::runtime_error with the reasons if the flags report input
// outside the range of the device normalization. `what` names the operand side in the message
// ("rows of A", "columns of B"). The flags mean: the largest magnitude of a row (column) is above the BF16
// range (bit 0) or nonzero and below 2^-126 (bit 1), or an element is Inf / NaN (bit 2). The public entry
// point (src/context.cpp) already rejects such input on the host, so for library calls this is a defensive
// check of the device-side normalization that never fires.
static inline void expdelta_check(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const ExpDeltaCtx& ec,
                                  const char* what) {
    const uint32_t bits = expdelta_flags(md, ec);
    if (bits == 0u) return;
    std::string msg = fmt::format("Input outside the range of the device normalization ({}):", what);
    if (bits & 1u)
        msg +=
            " a largest magnitude exceeds the BF16 range (it must be at most the largest finite BF16 value, "
            "2^128 - 2^120); divide the whole matrix by a power of two.";
    if (bits & 2u)
        msg +=
            " a nonzero largest magnitude is below 2^-126 and cannot be represented as a normal BF16 value; "
            "multiply the whole matrix by a power of two.";
    if (bits & 4u) msg += " an element is Inf or NaN; only finite FP64 values are accepted.";
    throw std::runtime_error(msg);
}

// ===== Row / column maxima (on the device) =====
// Nothing in this stage changes between issues (addresses, direction and tile distribution are all fixed
// when the program is built), so the runtime arguments are set at build time and scale_run only issues
// the program. The context does not hold the buffers: holding a shared_ptr would keep them alive after
// the caller releases them, so their lifetime stays with the caller. The only DRAM of this stage is the
// input and the maxima (the reader takes the maximum on the FP64 bit patterns; no copy of |x| is made).
struct ScaleCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
};

// Builds the program that writes nout tiles of maxima (BF16) to refOut. Each output tile reduces Kt
// input tiles of the FP64 operand refIn; the input tile index is o*Kt+kt in mode 0 (A, along the rows)
// and kt*stride+o in mode 1 (B, along the columns). scale_f64_reader.cpp finds the maximum of |x| on the
// FP64 bit patterns and converts only the 32 results of a tile to BF16.
// bf_ceil rounds the BF16 conversion up instead of down. above_pow2 (requires bf_ceil) additionally
// moves the BF16 value one step up when the maximum is exactly a power of two, so that the scale is a
// power of two strictly larger than the maximum (CEIL = 2 of scale_f64_reader.cpp).
// The program is placed on the whole grid (the grid of full_grid_span, which also distributes the tiles).
inline ScaleCtx scale_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, uint32_t mode, uint32_t Kt,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> refIn,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> refOut, uint32_t nout,
                            uint32_t stride, bool bf_ceil = false, bool above_pow2 = false) {
    namespace tm = tt::tt_metal;
    // Empty matrices are not accepted: with Kt = 0 the kernel would prefetch tile 0 and finish without
    // waiting for the read, and with nout = 0 the assignment of the cores would be meaningless.
    TT_FATAL(nout > 0 && Kt > 0, "scale_build: {} output tiles or {} K tiles is zero (empty matrices are not accepted)",
             nout, Kt);
    TT_FATAL(!above_pow2 || bf_ceil, "scale_build: above_pow2 requires bf_ceil (rounding up)");
    tm::distributed::MeshWorkload workload;
    auto device_range = tm::distributed::MeshCoordinateRange(md->shape());
    tm::Program program{};
    uint32_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    // The tiles are distributed by full_grid_span, and the CBs and kernels are placed on the same grid.
    // The kernels compute global tile indices from start_o, so the TensorAccessor base stays the start of
    // the whole buffer.
    const GridSpan gs = full_grid_span(md, nout);
    const tm::CoreRange all_cores = gs.range();
    // The same kernel (scale_f64_reader.cpp) is placed on both data-movement RISC-V processors, and each
    // takes half of the output tiles of the core. Each has its own work areas: two FP64 tiles (8192 B
    // each; the next one is read while one is examined) and one BF16 result tile (expdelta.cpp reads it
    // as BF16; the scale only needs the exponent of the maximum, for which BF16 is enough).
    namespace sc = scale_args;
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_F64_R1, tt::DataFormat::Float32, 2,
            OZ_F64_TILE_BYTES);  // FP64 tiles of RISCV_1
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_F64_R0, tt::DataFormat::Float32, 2,
            OZ_F64_TILE_BYTES);  // FP64 tiles of RISCV_0
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_OUT_R1, tt::DataFormat::Float16_b, 1, bf16t);  // result of RISCV_1
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_OUT_R0, tt::DataFormat::Float16_b, 1, bf16t);  // result of RISCV_0
    auto ct_for = [&](uint32_t cbf, uint32_t cbo) {
        std::vector<uint32_t> a;
        tm::TensorAccessorArgs(*refIn).append_to(a);
        tm::TensorAccessorArgs(*refOut).append_to(a);
        const size_t o = a.size();
        a.resize(o + sc::CT_COUNT);
        a[o + sc::CT_CEIL] = above_pow2 ? 2u : bf_ceil ? 1u : 0u;  // CEIL of scale_f64_reader.cpp
        a[o + sc::CT_MODE] = mode;
        a[o + sc::CT_CB_F64] = cbf;
        a[o + sc::CT_CB_OUT] = cbo;
        static_assert(sc::CT_COUNT == 4, "fill the new CT entry of scale_args here");
        return a;
    };
    auto rk = tm::CreateKernel(program, ::ttgemm::detail::kernel_path("common/kernels/scale_f64_reader.cpp"), all_cores,
                               tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                                                      .noc = tm::NOC::RISCV_1_default,
                                                      .compile_args = ct_for(sc::CB_F64_R1, sc::CB_OUT_R1)});
    auto wk = tm::CreateKernel(program, ::ttgemm::detail::kernel_path("common/kernels/scale_f64_reader.cpp"), all_cores,
                               tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                                                      .noc = tm::NOC::RISCV_0_default,
                                                      .compile_args = ct_for(sc::CB_F64_R0, sc::CB_OUT_R0)});
    const uint32_t in_addr = (uint32_t)refIn->address(), out_addr = (uint32_t)refOut->address();
    for (size_t c = 0; c < gs.cores.size(); ++c) {
        const uint32_t nt = gs.count(c), start = gs.begin(c);
        // RISCV_1 takes the first half (the larger one if nt is odd) and RISCV_0 the rest (with a single
        // tile, only RISCV_1 works).
        const uint32_t n1 = (nt + 1) / 2, n0 = nt - n1;
        auto rt = [&](uint32_t n, uint32_t s0) {
            std::vector<uint32_t> a(sc::R_COUNT);
            a[sc::R_IN_ADDR] = in_addr;
            a[sc::R_OUT_ADDR] = out_addr;
            a[sc::R_NOUT] = n;
            a[sc::R_KT] = Kt;
            a[sc::R_STRIDE] = stride;
            a[sc::R_START_O] = s0;
            static_assert(sc::R_COUNT == 6, "fill the new RT entry of scale_args here");
            return a;
        };
        tm::SetRuntimeArgs(program, rk, gs.cores[c], rt(n1, start));
        tm::SetRuntimeArgs(program, wk, gs.cores[c], rt(n0, start + n1));
    }
    ScaleCtx s;
    workload.add_program(device_range, std::move(program));
    s.workload = std::move(workload);
    return s;
}

// Issues the program once (nonblocking; the runtime arguments were set at build time).
inline void scale_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ScaleCtx& s) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), s.workload, false);
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
