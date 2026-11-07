// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: construction of the device programs of the stages (circular buffers,
// kernels, runtime arguments). Nothing here enqueues, synchronizes or times a program; the
// driver (src/ozaki2.cpp) does that.
//   build_slice           slicing: FP64 operand -> residue planes (INT8) or digit planes
//   gemm_build_partial    one partial-product GEMM for one output panel
//   build_compose, compose_set_args
//                         composition: digit partial products of one modulus -> residue
//   recon_build_pass      one pass of the CRT reconstruction -> FP64
// The order in which circular buffers are created fixes the L1 layout, and the order in
// which buffers, kernels and semaphores are created fixes addresses and ids, so these
// orders must not change.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include "ozaki2/plan.hpp"
#include "common/mcast_gemm.hpp"
#include "common/slice_f64_host.hpp"
#include "common/bits.hpp"
#include "backend_interface.hpp"
#include "common/kernels/f64_tf.h"
#include "common/kernels/slice_f64_args.h"
#include "ozaki2/kernels/args.h"
#include "ozaki2/kernels/recon_args.h"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// ===== Stage 1: modular slicing =====
// An operand of R x C elements is decomposed into planes of Rt*Ct tiles: one residue
// plane per modulus (INT8) or the digit planes of every modulus (digit formats). The
// kernels normalize, convert to an integer A', split A' into 3 to 5 radix-2^20 digits
// (kernels/digits.h) and combine the residues of the digits (modslice_int8.cpp /
// modslice_digit.cpp). The input is the unnormalized FP64 operand: the reader
// (common/kernels/slice_f64_reader.cpp) adds the per-row (A) or per-column (B) exponent
// delta to the FP64 exponent in integers and hands over a triple-float, so |A'| <= 2^b
// holds whatever the magnitude of the input.
// The runtime arguments are fixed at construction; the same workload can be enqueued
// again. The planes stay on the device (out) and the GEMM stage reads panels by offset.
struct Oz2Slice {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out;
    // Record of the layout, checked by the GEMM stage (run_digit_gemms): number of output
    // planes, concatenation bound maxg, and the layout of the slice writer (Layout in
    // kernels/args.h: LAYOUT_PLANES plane by plane, LAYOUT_A_CAT A concatenated along K,
    // LAYOUT_B_PERM B in reverse order within a modulus).
    uint32_t planes = 0, layout = oz2_args::LAYOUT_PLANES;
    int maxg = 0;
};

// The slicer takes the per-modulus constants it needs (modulus, reciprocal, digit
// weights, radix and method of the digit splitting), the plane layout (poff, nplane) and
// the concatenation bound maxg from the moduli set. Check the layout before the stages
// are built: the per-modulus arrays have one entry per modulus; poff starts at plane 0,
// grows by the digit planes of each modulus and ends at nplane (for INT8 one residue
// plane per modulus, so nplane = s).
static void check_moduli_layout(const Ozaki2Moduli& mo) {
    TT_FATAL(mo.s > 0, "check_moduli_layout: the moduli set is empty");
    auto len = [&](const auto& v) {
        TT_FATAL(v.size() == (size_t)mo.s, "check_moduli_layout: a per-modulus array has length {}, not {}", v.size(),
                 mo.s);
    };
    len(mo.m);
    len(mo.q);
    len(mo.mode);
    len(mo.poff);
    len(mo.rinv);
    len(mo.c0);
    len(mo.c1);
    len(mo.c2);
    len(mo.c3);
    TT_FATAL(mo.poff[0] == 0, "check_moduli_layout: the first digit plane is not plane 0");
    for (uint32_t i = 0; i < (uint32_t)mo.s; i++) {
        const int npl = OZ2_DIGIT ? oz2_digit_planes(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd()) : 1;
        const int next = (i + 1 < (uint32_t)mo.s) ? mo.poff[i + 1] : mo.nplane;
        TT_FATAL(next - mo.poff[i] == npl, "check_moduli_layout: modulus {} has {} digit planes (expected {})", mo.m[i],
                 next - mo.poff[i], npl);
    }
}

// Circular buffers of the slicing stage. CB_HI, CB_BRAW, CB_CRAW hold the triple-float of
// the operand (high, middle, low; the lower two as raw bits), CB_SCALE the scale of the
// low limb, CB_RES01/CB_RES2 the remainders while digits are made, CB_DIGIT_HI/CB_DIGIT_LO
// the digits, CB_PLANE_OUT the output. The indices come from
// common/kernels/slice_f64_args.h (shared FP64 reader) and kernels/args.h.
static void slice_make_cbs(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores, uint32_t ND,
                           uint32_t slice_tile) {
    namespace sa = slice_f64_args;
    namespace oa = oz2_args;
    const uint32_t f32 = tt::tile_size(tt::DataFormat::Float32);
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    auto cb_f32 = [&](uint32_t idx, uint32_t depth) {
        make_cb(program, all_cores, cbi(idx), tt::DataFormat::Float32, depth, f32);
    };
    cb_f32(sa::CB_HI, 4);
    cb_f32(sa::CB_CRAW, 4);
    cb_f32(sa::CB_SCALE, 4);
    cb_f32(sa::CB_BRAW, 4);
    cb_f32(sa::CB_RES01, 8);
    cb_f32(sa::CB_RES2, 4);
    // The upper digits are pushed one tile at a time, but the ND-1 tiles of one input tile
    // are waited for, indexed and popped as a group, so the depth must be a multiple of
    // ND-1 (a group that wraps around the ring boundary is corrupted).
    cb_f32(oa::CB_DIGIT_HI, 2 * (ND - 1));
    cb_f32(oa::CB_DIGIT_LO, 2);
    // Work area of the FP64 reader: one FP64 tile and the exponent deltas (Int32).
    make_cb(program, all_cores, cbi(sa::CB_F64), tt::DataFormat::Float32, 1, OZ_F64_TILE_BYTES);
    make_cb(program, all_cores, cbi(sa::CB_DLT), tt::DataFormat::Int32, 1, f32);
    make_cb(program, all_cores, cbi(oa::CB_PLANE_OUT), OZ_FMT.slice_out, 16, slice_tile);
}

// The three kernels of the slicing stage (reader, writer, slicer) and the writer layout.
struct Oz2SliceKernels {
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    uint32_t layout = oz2_args::LAYOUT_PLANES;
    std::vector<uint32_t> perm;  // LAYOUT_B_PERM: digit plane i is written to plane perm[i]
};
// mode: 0 = A (scales per row), 1 = B (scales per column). Sout: number of output planes.
static Oz2SliceKernels slice_make_kernels(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                                          const Ozaki2Moduli& mo, uint32_t ND, uint32_t Sout,
                                          const tt::tt_metal::distributed::MeshBuffer& f64B,
                                          const tt::tt_metal::distributed::MeshBuffer& dltB,
                                          const tt::tt_metal::distributed::MeshBuffer& outB, uint32_t mode) {
    namespace tm = tt::tt_metal;
    namespace sa = slice_f64_args;
    namespace oa = oz2_args;
    const uint32_t S = mo.s;
    Oz2SliceKernels k;
    // The slicer kernel depends on whether residues are split into digits.
    const std::string SLICE_KERNEL = OZ2_DIGIT ? ::ttgemm::detail::kernel_path("ozaki2/kernels/modslice_digit.cpp")
                                               : ::ttgemm::detail::kernel_path("ozaki2/kernels/modslice_int8.cpp");
    // The reader is shared with scheme I: it reads the FP64 operand and splits it into a
    // triple-float with integer bit operations (the normalization is folded in as the
    // exponent delta).
    const std::vector<uint32_t> ra = f64_reader_ct(f64B, dltB, false);
    // Layout (Layout in kernels/args.h). To concatenate digit products of equal weight
    // along K into one GEMM (oz2_product_groups in split.hpp), the digit planes of A are
    // placed side by side along K (LAYOUT_A_CAT) and those of B are stacked in reverse
    // order within each modulus (LAYOUT_B_PERM). INT8, and digit formats with nothing to
    // concatenate (maxg = 1), keep one plane after the other (LAYOUT_PLANES).
    k.layout = (OZ2_DIGIT && mo.maxg > 1) ? ((mode == 0) ? oa::LAYOUT_A_CAT : oa::LAYOUT_B_PERM) : oa::LAYOUT_PLANES;
    std::vector<uint32_t> wa;
    tm::TensorAccessorArgs(outB).append_to(wa);
    {
        const size_t o = wa.size();
        wa.resize(o + oa::SW_CT_COUNT);
        wa[o + oa::SW_CT_LAYOUT] = k.layout;
    }
    // Permutation of B: digit plane j of modulus s (plane poff[s]+j) goes to poff[s]+(np-1-j).
    k.perm.resize(Sout);
    for (uint32_t i = 0; i < Sout; i++) k.perm[i] = i;
    if constexpr (OZ2_DIGIT)
        for (uint32_t i = 0; i < S; i++) {
            const uint32_t p0 = (uint32_t)mo.poff[i], p1 = (i + 1 < S) ? (uint32_t)mo.poff[i + 1] : Sout;
            for (uint32_t j = p0; j < p1; j++) k.perm[j] = p1 - 1 - (j - p0);
        }
    k.rk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("common/kernels/slice_f64_reader.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_1, .noc = tm::NOC::RISCV_1_default, .compile_args = ra});
    k.wk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/modslice_writer.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = wa});
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    for (uint32_t idx : {sa::CB_HI, sa::CB_CRAW, sa::CB_SCALE, sa::CB_BRAW, sa::CB_RES01, sa::CB_RES2, oa::CB_DIGIT_HI,
                         oa::CB_DIGIT_LO})
        u2d[idx] = tm::UnpackToDestMode::UnpackToDestFp32;
    // dst_full_sync_en is required: the radix multiplication that makes the digits uses all
    // 8 DST registers. Compile-time arguments: number of moduli S, radix, number of digits
    // ND of A', digits of a residue, whether the output is written as binary16.
    std::vector<uint32_t> cct(oa::SC_COUNT);
    cct[oa::SC_S] = S;
    cct[oa::SC_RDX] = f32_bits(std::ldexp(1.0f, OZ2_DIGIT_BITS));
    cct[oa::SC_NDIGIT] = ND;
    cct[oa::SC_NRDIG] = (uint32_t)OZ2_PARAM.nrdig;
    cct[oa::SC_OUT_HALF] = (uint32_t)OZ_FMT.out_half;
    static_assert(oa::SC_COUNT == 5, "a field was added to SlicerCT: fill it here");
    k.ck = tm::CreateKernel(
        program, SLICE_KERNEL, all_cores,
        tm::ComputeConfig{
            .fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = cct});
    return k;
}

// Runtime arguments of the slicing stage, per core. The per-modulus constants (modulus,
// reciprocal, digit weights, radix and method of the digit splitting) are runtime
// arguments (layout: sl_* in kernels/args.h), so another moduli set with the same counts
// does not recompile the kernels. ntot: tiles of one output plane; Ct: tiles per row of
// the operand.
// in_off: first FP64 tile of the input (the band's first row of A); the reader and the
// exponent deltas use beg+in_off, the writer uses the in-band tile number beg.
static void slice_set_args(tt::tt_metal::Program& program, const GridSpan& gs, const Oz2SliceKernels& k,
                           const Ozaki2Moduli& mo, uint32_t ND, uint32_t Sout, uint32_t ntot, uint32_t Ct,
                           const tt::tt_metal::distributed::MeshBuffer& f64B,
                           const tt::tt_metal::distributed::MeshBuffer& dltB,
                           const tt::tt_metal::distributed::MeshBuffer& outB, uint32_t mode, uint32_t in_off) {
    namespace tm = tt::tt_metal;
    namespace oa = oz2_args;
    const uint32_t S = mo.s;
    for (size_t c = 0; c < gs.cores.size(); c++) {
        uint32_t nt = gs.count(c), beg = gs.begin(c);
        std::vector<uint32_t> ca(oa::sl_count(S, ND, OZ2_DIGIT), 0u);
        ca[oa::sl_ntiles()] = nt;
        for (uint32_t i = 0; i < S; i++) {
            ca[oa::sl_mod(S, i)] = f32_bits((float)mo.m[i]);
            ca[oa::sl_rinv(S, i)] = f32_bits(mo.rinv[i]);
            // Weight 2^(20(j+1)) mod m of digit d_{j+1} of A' (j = 0..ND-2).
            const float wt[OZ2_MAXDIGIT - 1] = {mo.c0[i], mo.c1[i], mo.c2[i], mo.c3[i]};
            for (uint32_t j = 0; j + 1 < ND; j++) ca[oa::sl_weight(S, j, i)] = f32_bits(wt[j]);
            if constexpr (OZ2_DIGIT) {
                // Radix of the digit splitting and the digit-making mode (Oz2XMode in
                // kernels/digit_limits.h, passed as a float value), followed by the number
                // of balanced digits (0 unless the modulus uses balanced n digits).
                ca[oa::sl_q(S, ND, i)] = f32_bits((float)mo.q[i]);
                ca[oa::sl_qinv(S, ND, i)] = f32_bits(1.0f / (float)mo.q[i]);
                const Oz2XMode xm = oz2_xmode(mo.mode[i]);
                TT_FATAL(xm < OZ2_XMODE_COUNT,
                         "build_slice: modulus {} has splitting method {}, which the slicer does not know", mo.m[i],
                         mo.mode[i]);
                ca[oa::sl_xmode(S, ND, i)] = f32_bits((float)xm);
                ca[oa::sl_ndig(S, ND, i)] = (uint32_t)oz2_bal_digits(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd());
            }
        }
        // Exponent deltas: per row for A (mode 0: tile number divided by Ct), per column
        // for B (mode 1: tile number modulo Ct).
        tm::SetRuntimeArgs(
            program, k.rk, gs.cores[c],
            f64_reader_rt((uint32_t)f64B.address(), (uint32_t)dltB.address(), nt, beg + in_off, Ct, mode));
        tm::SetRuntimeArgs(program, k.ck, gs.cores[c], ca);
        std::vector<uint32_t> wr(oa::sw_count(k.layout, Sout));
        wr[oa::SW_DST_ADDR] = (uint32_t)outB.address();
        wr[oa::SW_NUM] = nt;
        wr[oa::SW_START] = beg;
        wr[oa::SW_PLANES] = Sout;
        wr[oa::SW_NTOT] = ntot;
        if (k.layout == oa::LAYOUT_A_CAT) wr[oa::SW_A_KTILES] = Ct;
        if (k.layout == oa::LAYOUT_B_PERM)
            for (uint32_t i = 0; i < Sout; i++) wr[oa::sw_perm(i)] = k.perm[i];
        tm::SetRuntimeArgs(program, k.wk, gs.cores[c], wr);
    }
}

// Build the slicing stage for an operand of R x C elements. ND: number of radix-2^20
// digits of A' (3 to 5, from b). mode: 0 = A, 1 = B. f64B: the FP64 operand; dltB: its
// exponent deltas. Allocates the output planes (out) unless `shared` is given, in which
// case the planes are written there (the A slicers of all bands reuse one buffer); it must
// have the slice tile size and room for the planes. in_off: first FP64 tile of the input.
// The workload is not enqueued.
static Oz2Slice build_slice(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const Ozaki2Moduli& mo,
                            uint32_t ND, uint32_t R, uint32_t C,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> f64B,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltB, uint32_t mode,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> shared = nullptr,
                            uint32_t in_off = 0) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const uint32_t S = mo.s;
    const uint32_t Rt = R / 32, Ct = C / 32, ntot = Rt * Ct;
    // Tile size of the format that carries the slices (it differs per format, so the DRAM
    // buffer and the circular buffer are sized from the format).
    const uint32_t slice_tile = tt::tile_size(OZ_FMT.slice_out);
    // The written format (slice_out) and the format the GEMM reads (in) differ for INT8
    // (UInt8 / Int8) and FP16 (UInt16 / Float16). The same bytes are reinterpreted, so the
    // tile sizes must agree.
    if (slice_tile != tt::tile_size(OZ_FMT.in))
        throw std::runtime_error("internal error: the slice format and the GEMM input format differ in tile size");
    // Number of output planes: the digit planes counted by the moduli selection (nplane,
    // poff; for INT8 one residue plane per modulus, check_moduli_layout).
    const uint32_t Sout = (uint32_t)mo.nplane;
    // The FP16 slicer converts digits to binary16 bit patterns, except in the branch for
    // balanced n digits. The formats with digit bound 32 never use balanced n digits
    // (oz2_split_mode); if that rule changed, FP32 values would be written into the UInt16
    // buffer unnoticed, so it is checked here.
    if (OZ2_DIGIT && OZ_FMT.out_half)
        for (uint32_t i = 0; i < S; i++)
            TT_FATAL(mo.mode[i] != OZ2_SPLIT_BAL,
                     "build_slice: modulus {} uses balanced n digits in a format written as binary16", mo.m[i]);
    // The formats that select their moduli with the stage-time model allocate the planes
    // from the top of DRAM (see upload_operands in src/ozaki2.cpp).
    auto outB = shared ? shared : mesh_dram_buffer(md, (size_t)Sout * ntot, slice_tile, OZ2_PARAM.awe != 0);
    TT_FATAL(outB->page_size() == slice_tile && outB->num_pages() >= (uint64_t)Sout * ntot,
             "build_slice: the given buffer (page {} B, {} pages) cannot hold {} planes (tile {} B, {} tiles)",
             outB->page_size(), outB->num_pages(), Sout, slice_tile, (uint64_t)Sout * ntot);
    // Slicing is element-wise work per tile: the shape of the core grid hardly matters,
    // but the time is nearly proportional to the number of cores, so the whole compute
    // grid is used and the tiles are spread evenly.
    auto gs = full_grid_span(md, ntot);
    tm::CoreRange all_cores = gs.range();
    dist::MeshWorkload workload;
    auto range = dist::MeshCoordinateRange(md->shape());
    tm::Program program{};
    slice_make_cbs(program, all_cores, ND, slice_tile);
    const Oz2SliceKernels k = slice_make_kernels(program, all_cores, mo, ND, Sout, *f64B, *dltB, *outB, mode);
    slice_set_args(program, gs, k, mo, ND, Sout, ntot, Ct, *f64B, *dltB, *outB, mode, in_off);
    workload.add_program(range, std::move(program));
    Oz2Slice s;
    s.workload = std::move(workload);
    s.out = outB;
    s.planes = Sout;
    s.layout = k.layout;
    s.maxg = mo.maxg;
    return s;
}

// ===== Stage 2: one partial-product GEMM for one output panel =====
// Computes one output panel of M x N elements (M = panel rows; N is implied by the
// configuration g) over the whole inner dimension K in one launch, and writes it directly
// into the persistent buffer Cbuf. All eight formats use this function; what is carried
// (residue planes or digit planes) differs, the launch does not.
//   dA, dB  tile offsets of the panel in the A and B slice buffers
//   Ntg     tiles per row of Cbuf (the whole output width); dW: tile offset of the panel
//           origin in Cbuf (plane and panel position)
//   KsA     row stride of A in tiles (0: K/32). With concatenation the A buffer is the
//           digit planes side by side along K, and K is the length g*Kt of the g
//           concatenated digit products.
// The panels bound the output block per core, and therefore the L1 use. The output stays
// on the device, so the composition and reconstruction stages read it directly. The
// workload is not enqueued.
static tt::tt_metal::distributed::MeshWorkload gemm_build_partial(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const OptConfig& g,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Abuf,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Bbuf, uint32_t dA, uint32_t dB, uint32_t M, uint32_t K,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Cbuf, uint32_t Ntg, uint32_t dW, uint32_t KsA) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    dist::MeshWorkload workload;
    auto device_range = dist::MeshCoordinateRange(md->shape());
    tm::Program program{};
    const uint32_t Mt = M / tt::constants::TILE_HEIGHT, Kt = K / tt::constants::TILE_WIDTH;
    const uint32_t KtS = KsA ? KsA : Kt;  // row stride of A; the program is built for this width
    TT_FATAL(KtS >= Kt, "gemm_build_partial: the row stride {} of A is shorter than K {}", KtS, Kt);
    const uint32_t num_blocks = Kt / g.bw;
    // The program is assembled by the code shared with scheme I (spill_gemm_spec and
    // build_mcast_gemm in common/mcast_gemm.hpp). The digit formats accumulate in FP32
    // and their output is read as Float32 holding exact integers; INT8 accumulates in
    // sign-magnitude int32 (OZ_FMT.out). The reader reads A as Mt x Kt tiles and B as
    // Kt x Ntg; the writer uses the row stride Ntg and starts at dW. The partial sums
    // along K are accumulated in L1 and read back once after the last K block.
    SpillOpts so;
    so.in_fmt = OZ_FMT.in;
    so.acc_fmt = OZ_FMT.out;
    so.fidelity = oz2_fidelity();
    McastGemmSpec spec = spill_gemm_spec(g, Kt, so);
    spill_check_run(num_blocks, false, 0u);
    spec.compute_rt = spill_compute_rt(num_blocks);  // the compute kernel receives the number of K blocks at run time
    spec.dA = dA;
    spec.dB = dB;
    spec.dW = dW;
    McastGemm mg = build_mcast_gemm(program, md, g, Mt, Ntg, KtS, Abuf, Bbuf, Cbuf, spec);
    // The number of K blocks of the readers (RA_NUM_BLOCKS) and of the compute kernel
    // (CR_NUM_BLOCKS) must agree, or the side with more blocks waits forever. The ranges
    // read must lie inside the A and B buffers (A: Mt rows of stride KtS, Kt tiles along
    // K; B: Kt rows of stride Ntg).
    {
        namespace ma = mcast_args;
        namespace ga = gemm_args;
        TT_FATAL(num_blocks * g.bw == Kt && spec.compute_rt.size() > ga::CR_NUM_BLOCKS &&
                     spec.compute_rt[ga::CR_NUM_BLOCKS] == num_blocks,
                 "gemm_build_partial: {} K blocks (bw={}, Kt={}) do not match the {} of the compute kernel", num_blocks,
                 g.bw, Kt, spec.compute_rt.size() > ga::CR_NUM_BLOCKS ? spec.compute_rt[ga::CR_NUM_BLOCKS] : 0u);
        const uint64_t ta = Abuf->size() / tt::tile_size(OZ_FMT.in), tb = Bbuf->size() / tt::tile_size(OZ_FMT.in);
        TT_FATAL((uint64_t)dA + (uint64_t)(Mt - 1) * KtS + Kt <= ta,
                 "gemm_build_partial: the range read from A is outside its buffer ({} tiles)", ta);
        TT_FATAL(((uint64_t)dB / Ntg + Kt) * Ntg <= tb,
                 "gemm_build_partial: rows {}..{} read from B are outside its buffer ({} rows)", dB / Ntg,
                 dB / Ntg + Kt, tb / Ntg);
        for (const auto& e : mg.cores)
            TT_FATAL(e.ra[ma::RA_NUM_BLOCKS] == KtS / g.bw,
                     "gemm_build_partial: the reader was built with {} blocks (expected {})", e.ra[ma::RA_NUM_BLOCKS],
                     KtS / g.bw);
    }
    if (KtS != Kt) {
        // The program was built for the row stride KtS; along K only the first Kt tiles
        // (num_blocks K blocks) are read (McastGemm::retarget_k). The writer arguments are
        // those of the construction. The replaced reader arguments go to the A side and
        // the B side, so B is supplied for num_blocks blocks as well.
        for (const auto& e : mg.cores) {
            auto wa = mcast_writer_args((uint32_t)Cbuf->address(), Ntg, g.pm, g.pn, g.sh, g.sw, (uint32_t)e.core.x,
                                        (uint32_t)e.core.y, dW, 0u);
            mg.retarget_k(program, e, 0u, 0u, num_blocks, wa);
        }
    }
    workload.add_program(device_range, std::move(program));
    return workload;
}

#ifdef OZ2_DIGITK
// ===== Stage 2.5: composition of the digit partial products (digit formats) =====
// From the GEMM outputs of one modulus (after concatenation) the residue r = D (mod m) is
// formed and written to the plane of the modulus; one-digit moduli do not pass here. The
// composition is linear modulo m, so reducing the digit products first keeps everything
// in FP32 even where |D| would exceed 2^24 (kernels/compose.cpp).
// It is a stage of its own, not fused into the reconstruction, so that the product planes
// are reused for every modulus: one workload is built and enqueued again per modulus with
// other runtime arguments.
struct Oz2Compose {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    // The input and output buffers are fixed at construction: the accessors (compile-time
    // arguments) are made from them. With chunking the driver passes the address of the
    // product buffer as the output (to store partial sums); that is valid only when both
    // buffers give identical accessor arguments, page size and buffer type, which is
    // checked at construction and kept in out_accepts_in. Only addresses are kept, so that
    // this object does not keep the buffers alive.
    uint32_t in_addr = 0, out_addr = 0;
    bool out_accepts_in = false;  // the writer may be given the address of the input buffer
    GridSpan grid;
};
// ntot: tiles of one plane. Dbuf: product planes (input). outB: residue planes (output).
// np_lcm: depth of the input circular buffer (compose_cb_depth in plan.hpp).
static Oz2Compose build_compose(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, uint32_t ntot,
                                std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Dbuf,
                                std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> outB, uint32_t np_lcm) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    namespace oa = oz2_args;
    const uint32_t f32 = tt::tile_size(tt::DataFormat::Float32);
    auto gs = full_grid_span(md, ntot);
    tm::CoreRange all_cores = gs.range();
    dist::MeshWorkload workload;
    auto range = dist::MeshCoordinateRange(md->shape());
    tm::Program program{};
    // CB_PLANE_IN receives the GEMM outputs of one modulus per output tile. Its depth must
    // be a common multiple of the group sizes, or a group that wraps around the ring
    // boundary is corrupted.
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_PLANE_IN, tt::DataFormat::Float32, np_lcm, f32);
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_PLANE_OUT, tt::DataFormat::Float32, 4, f32);
    std::vector<uint32_t> ra, wa;
    // Plane j is at tile j*ntot+g; the plane reader reads the planes of one output tile.
    tm::TensorAccessorArgs(*Dbuf).append_to(ra);
    tm::TensorAccessorArgs(*outB).append_to(wa);
    // The writer stores residue planes one after the other (LAYOUT_PLANES).
    {
        const size_t o = wa.size();
        wa.resize(o + oa::SW_CT_COUNT);
        wa[o + oa::SW_CT_LAYOUT] = oa::LAYOUT_PLANES;
    }
    auto rk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/plane_reader.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_1, .noc = tm::NOC::RISCV_1_default, .compile_args = ra});
    auto wk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/modslice_writer.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = wa});
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[oa::CB_PLANE_IN] = tm::UnpackToDestMode::UnpackToDestFp32;
    // dst_full_sync_en is required: the composition uses DST registers 0..6.
    auto ck = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/compose.cpp"), all_cores,
        tm::ComputeConfig{
            .fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = {}});
    Oz2Compose c;
    c.rk = rk;
    c.wk = wk;
    c.ck = ck;
    c.range = range;
    c.in_addr = (uint32_t)Dbuf->address();
    c.out_addr = (uint32_t)outB->address();
    {
        std::vector<uint32_t> ai, ao;
        tm::TensorAccessorArgs(*Dbuf).append_to(ai);
        tm::TensorAccessorArgs(*outB).append_to(ao);
        c.out_accepts_in = ai == ao && Dbuf->page_size() == outB->page_size() &&
                           Dbuf->device_local_config().buffer_type == outB->device_local_config().buffer_type;
    }
    c.grid = gs;
    // compose_set_args fills the runtime arguments per modulus; here only their shape is set.
    for (auto& core : c.grid.cores) {
        tm::SetRuntimeArgs(program, rk, core, std::vector<uint32_t>(oa::PR_COUNT, 0u));
        tm::SetRuntimeArgs(program, ck, core, std::vector<uint32_t>(oa::CP_COUNT, 0u));
        tm::SetRuntimeArgs(program, wk, core, std::vector<uint32_t>(oa::sw_count(oa::LAYOUT_PLANES, 1u), 0u));
    }
    workload.add_program(range, std::move(program));
    c.workload = std::move(workload);
    return c;
}

// Set the runtime arguments of the composition stage for modulus s (nothing is enqueued):
// compose the input planes [in0, in0+jn) and write plane out_plane of the buffer at
// out_addr. With `unit` every coefficient is 1 (adding partial sums); otherwise the
// coefficients of the GEMM outputs j0..j0+jn-1 are used (one per output, oz2_product_groups
// in split.hpp). The high coefficient of `unit` is 2^12 mod m (OZ2_COMPOSE_SPLIT), which is
// coefficient 1 for the 2^12-split method.
static void compose_set_args(Oz2Compose& c, const Ozaki2Moduli& mo, uint32_t s, uint32_t j0, uint32_t jn, bool unit,
                             uint32_t in0, uint32_t out_addr, uint32_t out_plane) {
    namespace tm = tt::tt_metal;
    namespace oa = oz2_args;
    tm::Program& prog = c.workload.get_programs().at(c.range);
    Oz2ProductGroup gr[OZ2_MAXPROD];
    oz2_product_groups(mo.m[s], gr, OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[s]);
    float cb[OZ2_MAXPROD], ch[OZ2_MAXPROD];
    for (uint32_t j = 0; j < jn; j++) {
        cb[j] = unit ? 1.f : gr[j0 + j].coeff_lo;
        ch[j] = unit ? (float)(OZ2_COMPOSE_SPLIT % mo.m[s]) : gr[j0 + j].coeff_hi;
    }
    for (size_t i = 0; i < c.grid.cores.size(); i++) {
        auto& core = c.grid.cores[i];
        uint32_t nt = c.grid.count(i), beg = c.grid.begin(i);
        // Plane reader: the input planes [in0, in0+jn) per output tile (PlaneReaderRT in kernels/args.h).
        std::vector<uint32_t> ra(oa::PR_COUNT, 0u);
        ra[oa::PR_IN_ADDR] = c.in_addr;
        ra[oa::PR_NUM] = nt;
        ra[oa::PR_START] = in0 * c.grid.ntiles + beg;
        ra[oa::PR_NTOT] = c.grid.ntiles;
        ra[oa::PR_NPLANES] = jn;
        tm::SetRuntimeArgs(prog, c.rk, core, ra);
        // The number of runtime arguments of a kernel is fixed, so the coefficients always
        // occupy OZ2_MAXPROD slots and the rest is zero (ComposeRT in kernels/args.h).
        // Whether the products are reduced first is fold_first of the moduli selection.
        std::vector<uint32_t> ca(oa::CP_COUNT, 0u);
        ca[oa::CP_NTILES] = nt;
        ca[oa::CP_NP] = jn;
        ca[oa::CP_MOD] = f32_bits((float)mo.m[s]);
        ca[oa::CP_RINV] = f32_bits(mo.rinv[s]);
        ca[oa::CP_FOLD_FIRST] = mo.fold_first ? 1u : 0u;
        for (uint32_t j = 0; j < jn; j++) {
            ca[oa::cp_coeff_lo(j)] = f32_bits(cb[j]);
            ca[oa::cp_coeff_hi(j)] = f32_bits(ch[j]);
        }
        tm::SetRuntimeArgs(prog, c.ck, core, ca);
        // Output: one plane (LAYOUT_PLANES); the start position carries the plane.
        std::vector<uint32_t> wr(oa::sw_count(oa::LAYOUT_PLANES, 1u));
        wr[oa::SW_DST_ADDR] = out_addr;
        wr[oa::SW_NUM] = nt;
        wr[oa::SW_START] = out_plane * c.grid.ntiles + beg;
        wr[oa::SW_PLANES] = 1u;
        wr[oa::SW_NTOT] = c.grid.ntiles;
        tm::SetRuntimeArgs(prog, c.wk, core, wr);
    }
}
#endif

// ===== Stage 3: CRT reconstruction (one pass) =====
// Buffers a reconstruction pass reads and writes.
struct ReconBufs {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
        in;  // S input planes: residues in FP32, or the sign-magnitude int32 C_i (INT8)
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
        limb[3];  // limb planes; never written by the library's passes (one-tile placeholders)
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
        a128;  // 128-bit fixed point: written by the first of two passes, read by the second
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> o64;  // the FP64 output C (tile order)
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltA,
        dltB;  // exponent deltas of the rows and columns (tau_A*tau_B is added back to the exponent)
    // First output tile of the current band (o64 and the exponent deltas are indexed by
    // tile + out_off; in, limb and a128 hold one band and use in-band numbers).
    uint32_t out_off = 0;
};

// Element index in a tile at which the per-element assembly (FP64, or the 128-bit fixed
// point in the first of two passes) is divided between the two data-movement cores: the
// reader handles the first OZ2_JSPLIT of the 1024 elements, the writer the rest. Every
// pass shares the assembly this way (the writer of each pass does per-element integer
// work: FP64 assembly, or the conversion to the fixed point).
constexpr uint32_t OZ2_JSPLIT = 512;

// Append the compile-time arguments that follow the TensorAccessorArgs of the reader and
// the writer (ReaderWriterCT in recon_args.h). final_or_share means "this pass assembles
// FP64" for the writer and "the reader shares the assembly" for the reader (always 1);
// role is the role of the pass (ReconRole).
static void append_rw_ct(std::vector<uint32_t>& v, uint32_t final_or_share, recon_args::ReconRole role) {
    namespace rg = recon_args;
    const size_t o = v.size();
    v.resize(o + rg::CT_COUNT);
    v[o + rg::CT_FINAL_OR_SHARE] = final_or_share;
    v[o + rg::CT_ROLE] = role;
    v[o + rg::CT_PROBE] = 0u;  // diagnostic selector of the kernels; always 0 in the library
    static_assert(rg::CT_COUNT == 3, "a field was added to ReaderWriterCT: fill it here");
}

// Shape of one reconstruction pass: the values that follow from the pass, the buffers and
// the moduli set, shared by recon_make_cbs, recon_make_kernels and recon_set_args.
struct ReconPassShape {
    uint32_t S = 0, Mt = 0, Nt = 0, ntot = 0;  // moduli; tile rows, tile columns and tiles of the processed band
    uint32_t FINAL = 0;                        // this pass assembles FP64 (every role but ROLE_TWO_FIRST)
    recon_args::ReconRole role = recon_args::ROLE_SINGLE;
    const float* rp = nullptr;
    uint32_t NRP = 0;  // reciprocal fragments, [i*nrp + t]
    Oz2ReconCT ct{};   // compile-time arguments of the compute kernel that follow from the moduli set and the pass
    // Destinations of the three limbs (high, middle, low). In the first of two passes
    // number 0 is the 128-bit fixed point.
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out0, out1, out2;
};
// M, N: elements of the processed band (rows) and of the output width.
static ReconPassShape recon_pass_shape(const ReconPass& pass, const ReconBufs& bf, uint32_t M, uint32_t N,
                                       const Ozaki2Moduli& mo) {
    ReconPassShape p;
    p.S = mo.s;
    p.Mt = M / 32;
    p.Nt = N / 32;
    p.ntot = p.Mt * p.Nt;
    p.rp = pass.rp.data();
    p.NRP = pass.nrp;
    p.role = pass.role;
    p.FINAL = p.role != recon_args::ROLE_TWO_FIRST ? 1u : 0u;
    // The first of two passes writes the 128-bit fixed point in place of limb 0.
    p.out0 = p.role == recon_args::ROLE_TWO_FIRST ? bf.a128 : bf.limb[0];
    p.out1 = bf.limb[1];
    p.out2 = bf.limb[2];
    // Number of limbs (three for FP64-equivalent and for every pass of the two-pass
    // reconstruction, see oz2_recon_limb3), number of fragments added with tfAddSmall, and
    // the one-digit moduli to be reduced first (oz2_recon_ct in moduli.hpp).
    p.ct = oz2_recon_ct(mo, p.rp, p.NRP, OZ2_DIGIT, pass.limb3);
    return p;
}

// Circular buffers of the reconstruction stage (indices in kernels/recon_args.h).
static void recon_make_cbs(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                           const ReconPassShape& p) {
    namespace rg = recon_args;
    const uint32_t i32 = tt::tile_size(tt::DataFormat::Int32), f32 = tt::tile_size(tt::DataFormat::Float32);
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    // The residues of the digit formats arrive as FP32, so the buffer has that format (no
    // conversion); INT8 reads the sign-magnitude int32 of the GEMM stage.
    const tt::DataFormat in_fmt = OZ2_DIGIT ? tt::DataFormat::Float32 : tt::DataFormat::Int32;
    // The reader reserves the S planes of one output tile together and reads them behind
    // one barrier, so the depth must be a multiple of S (a group that wraps around the
    // ring boundary is corrupted); 2S doubles the buffering.
    make_cb(program, all_cores, cbi(rg::CB_IN), in_fmt, 2 * p.S, f32);
    if (!OZ2_DIGIT) {  // two mask tiles (the depth is a multiple of 2)
        make_cb(program, all_cores, cbi(rg::CB_MASK), tt::DataFormat::Int32, 4, i32);
    }
    make_cb(program, all_cores, cbi(rg::CB_LIMB), tt::DataFormat::Float32, (p.ct.NLIMB * 4),
            f32);   // a multiple of NLIMB
    if (p.FINAL) {  // work area in which the writer assembles FP64
        make_cb(program, all_cores, cbi(rg::CB_F64), tt::DataFormat::Float32, 1, OZ_F64_TILE_BYTES);
        make_cb(program, all_cores, cbi(rg::CB_DLT_A), tt::DataFormat::Int32, 1, f32);
        make_cb(program, all_cores, cbi(rg::CB_DLT_B), tt::DataFormat::Int32, 1, f32);
    }
    if (p.role !=
        rg::ROLE_SINGLE) {  // one tile of 128-bit fixed point: written by the first pass, read by the final pass
        make_cb(program, all_cores, cbi(rg::CB_A128), tt::DataFormat::Float32, 1, OZ2_A128_PAGE);
    }
    // The work areas and the delta tiles are in the L1 of the same core, so the reader
    // sees those of the writer. One word of the handoff of the shared assembly, the
    // address of the limbs. It is a circular buffer only so that both data-movement
    // kernels obtain the same L1 address; nothing is pushed or popped. The two tags are
    // semaphores (they return to 0 at every launch).
    make_cb(program, all_cores, cbi(rg::CB_SYNC), tt::DataFormat::Int32, 1, rg::SYNC_PAGE);
}

// The three kernels of a reconstruction pass (reader, writer, compute) and the semaphores
// of the handoff. Creation order: reader, writer, semaphores, compute.
struct ReconPassKernels {
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    uint32_t sem_pub = 0, sem_done = 0;
};
static ReconPassKernels recon_make_kernels(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                                           const ReconPassShape& p, const ReconBufs& bf) {
    namespace tm = tt::tt_metal;
    namespace rg = recon_args;
    ReconPassKernels k;

    std::vector<uint32_t> ra, wa;
    tm::TensorAccessorArgs(*bf.in).append_to(ra);
    // CT_FINAL_OR_SHARE of the reader: it shares the assembly.
    append_rw_ct(ra, 1u, p.role);
    tm::TensorAccessorArgs(*p.out0).append_to(wa);
    tm::TensorAccessorArgs(*p.out1).append_to(wa);
    tm::TensorAccessorArgs(*p.out2).append_to(wa);
    append_rw_ct(wa, p.FINAL, p.role);
    tm::TensorAccessorArgs(*bf.o64).append_to(wa);
    tm::TensorAccessorArgs(*bf.dltA).append_to(wa);
    tm::TensorAccessorArgs(*bf.dltB).append_to(wa);
    tm::TensorAccessorArgs(*bf.a128).append_to(wa);  // read by the final pass of two
    // The reader fetches the S planes of each output tile, recon_crt solves the CRT in
    // fraction form, and the writer stores FP64 (final pass) or the 128-bit fixed point
    // (first of two passes).
    k.rk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/recon_reader.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_1, .noc = tm::NOC::RISCV_1_default, .compile_args = ra});
    k.wk = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/recon_writer.cpp"), all_cores,
        tm::DataMovementConfig{
            .processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = wa});
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[rg::CB_IN] = tm::UnpackToDestMode::UnpackToDestFp32;
    // Tags of the handoff. Semaphores return to their initial value at every program
    // launch, so enqueuing the same workload again does not see old tags.
    k.sem_pub = tm::CreateSemaphore(program, all_cores, 0);
    k.sem_done = tm::CreateSemaphore(program, all_cores, 0);

    // dst_full_sync_en is required: the triple-float and the work values use all 8 DST
    // registers. With the default half synchronization the packer could still be reading
    // a tile while the next one overwrites registers 4..7.
    k.ck = tm::CreateKernel(
        program, ::ttgemm::detail::kernel_path("ozaki2/kernels/recon_crt.cpp"), all_cores,
        tm::ComputeConfig{
            .fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = [&] {
                std::vector<uint32_t> v(rg::C_COUNT, 0u);
                v[rg::C_S] = p.ct.S;
                v[rg::C_NLIMB] = p.ct.NLIMB;
                v[rg::C_IN_RESIDUE] = p.ct.IN_RESIDUE;
                v[rg::C_NRP] = p.ct.NRP;
                v[rg::C_NSMALL_FRAG] = p.ct.NSMALL_FRAG;
                v[rg::C_PROBE] = 0u;  // diagnostic selector of the kernel; always 0 in the library
                v[rg::C_RAWMASK] = p.ct.RAWMASK;
                return v;
            }()});
    return k;
}

// Runtime arguments of a reconstruction pass, per core. The per-modulus constants
// (modulus, reciprocal, CRT coefficient, reciprocal fragments) are runtime arguments (crt_*
// in kernels/recon_args.h); the counts are compile-time arguments.
static void recon_set_args(tt::tt_metal::Program& program, const GridSpan& gs, const ReconPassShape& p,
                           const ReconPassKernels& k, const ReconBufs& bf, const Ozaki2Moduli& mo,
                           const ReconScale& sc) {
    namespace tm = tt::tt_metal;
    const uint32_t S = p.S, NRP = p.NRP, NLIMB = p.ct.NLIMB, ntot = p.ntot, Nt = p.Nt;
    const float* rp = p.rp;
    for (size_t c = 0; c < gs.cores.size(); c++) {
        uint32_t nt = gs.count(c), beg = gs.begin(c);
        // The indices come from the helpers the kernel uses, so the layouts cannot diverge.
        std::vector<uint32_t> ca(recon_args::crt_count(S, NRP, OZ2_DIGIT), 0u);
        ca[recon_args::crt_ntiles()] = nt;
        for (uint32_t i = 0; i < S; i++) {
            ca[recon_args::crt_mod(S, i)] = f32_bits((float)mo.m[i]);
            ca[recon_args::crt_rinv(S, i)] = f32_bits(mo.rinv[i]);
            ca[recon_args::crt_yi(S, i)] = f32_bits((float)mo.yi[i]);
            // reciprocal fragments: NRP per modulus
            for (uint32_t t = 0; t < NRP; t++) ca[recon_args::crt_frag(S, t, i)] = f32_bits(rp[(size_t)i * NRP + t]);
            if (!OZ2_DIGIT)  // 2^16 mod m is needed only to decode the sign-magnitude int32 (INT8)
                ca[recon_args::crt_m16(S, NRP, i)] = f32_bits(mo.m16[i]);
        }
        // Filled by name (layout in kernels/recon_args.h), not by position.
        using namespace recon_args;
        std::vector<uint32_t> ra_rt(R_COUNT, 0u), wa_rt(W_COUNT, 0u);
        ra_rt[R_IN_ADDR] = (uint32_t)bf.in->address();
        ra_rt[R_NUM] = nt;
        ra_rt[R_START] = beg;
        ra_rt[R_NTOT] = ntot;
        ra_rt[R_S] = S;
        ra_rt[R_MASK] = OZ2_DIGIT ? 0u : 1u;
        ra_rt[R_NLIMB] = NLIMB;
        ra_rt[R_WHI] = sc.Whi;
        ra_rt[R_WLO] = sc.Wlo;
        ra_rt[R_QW] = (uint32_t)sc.qw;
        ra_rt[R_SH2] = (uint32_t)sc.sh2;
        ra_rt[R_JSPLIT] = OZ2_JSPLIT;
        ra_rt[R_SEM_PUB] = k.sem_pub;
        ra_rt[R_SEM_DONE] = k.sem_done;

        wa_rt[W_OUT0] = (uint32_t)p.out0->address();
        wa_rt[W_NUM] = nt;
        wa_rt[W_START] = beg;
        wa_rt[W_OUT1] = (uint32_t)p.out1->address();
        wa_rt[W_OUT2] = (uint32_t)p.out2->address();
        wa_rt[W_NLIMB] = NLIMB;
        wa_rt[W_O64] = (uint32_t)bf.o64->address();
        wa_rt[W_DLT_A] = (uint32_t)bf.dltA->address();
        wa_rt[W_DLT_B] = (uint32_t)bf.dltB->address();
        wa_rt[W_NT] = Nt;
        wa_rt[W_WHI] = sc.Whi;
        wa_rt[W_WLO] = sc.Wlo;
        wa_rt[W_QW] = (uint32_t)sc.qw;
        wa_rt[W_SH2] = (uint32_t)sc.sh2;
        wa_rt[W_A128] = (uint32_t)bf.a128->address();
        wa_rt[W_JLO] = OZ2_JSPLIT;
        wa_rt[W_SEM_PUB] = k.sem_pub;
        wa_rt[W_SEM_DONE] = k.sem_done;
        wa_rt[W_OUT_OFF] = bf.out_off;

        tm::SetRuntimeArgs(program, k.rk, gs.cores[c], ra_rt);
        tm::SetRuntimeArgs(program, k.ck, gs.cores[c], ca);
        tm::SetRuntimeArgs(program, k.wk, gs.cores[c], wa_rt);
    }
}

// Build one reconstruction pass. bf.in holds the S input planes of the processed band:
// the residues in FP32 for the digit formats, the sign-magnitude int32 written by the GEMM
// stage for INT8 (the kernel decodes it). M: rows (elements) of the processed band; N:
// output columns. A pass that assembles FP64 writes o64; the first of two passes adds its
// three limbs to the 128-bit fixed point a128. The workload is not enqueued.
static tt::tt_metal::distributed::MeshWorkload recon_build_pass(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const ReconPass& pass, const ReconBufs& bf, uint32_t M,
    uint32_t N, const Ozaki2Moduli& mo, const ReconScale& sc) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const ReconPassShape p = recon_pass_shape(pass, bf, M, N, mo);
    // Tiles are spread over the whole compute grid, as in the slicing stage.
    auto gs = full_grid_span(md, p.ntot);
    tm::CoreRange all_cores = gs.range();
    dist::MeshWorkload workload;
    auto range = dist::MeshCoordinateRange(md->shape());
    tm::Program program{};
    recon_make_cbs(program, all_cores, p);
    const ReconPassKernels k = recon_make_kernels(program, all_cores, p, bf);
    recon_set_args(program, gs, p, k, bf, mo, sc);
    workload.add_program(range, std::move(program));
    return workload;
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
