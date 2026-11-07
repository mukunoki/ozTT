// SPDX-License-Identifier: Apache-2.0
#pragma once
// Builder of the multicast matmul program: the partial-product GEMM stage of both Ozaki schemes
// (gemm_build in ozaki1/gemm.hpp, the GEMM stage in ozaki2/stages.hpp). It sits between slicing and
// accumulation / composition / reconstruction, and uses the helpers of common/gemm_host.hpp
// (mcast_reader_args, mcast_writer_args, make_cb, OptConfig).
//
// A is multicast to the cores of the same row and B to the cores of the same column, so each operand
// tile is read from DRAM once per row / column instead of once per core. One data-movement kernel,
// common/kernels/mcast_split_dm.cpp, is specialized by defines; A and B are placed on different RISC-V
// processors so that they move concurrently:
//   A (RISCV_1, NOC 1): the leftmost column reads the first half of each block from DRAM and the
//                       rightmost column the second half, and each sends its half to all other cores of
//                       its row; the columns in between only receive.
//   B (RISCV_0, NOC 0): the top row sends the first half and the bottom row the second half along the
//                       column; the rows in between only receive. After the K loop this kernel also
//                       writes the output (it doubles as the writer).
//
// The runtime arguments of the data-movement kernel are the 33 reader entries (mcast_reader_args),
// followed by 7 entries for the second-half sender and, on the B side only, by the writer arguments. The
// indices are the single table common/kernels/mcast_args.h, shared by the host and the kernel.
// McastGemm::set_args does the concatenation and stops if a count disagrees with the table.
//
// The compute kernel is common/kernels/bmm_int32_spill.cpp; spill_gemm_spec assembles its CBs, its
// compile-time arguments and the UnpackToDestFp32 settings in one place for both schemes.
//
// What differs between the uses (CB formats, standard or fused writer, shifts of the first tile of
// A/B/C, runtime arguments of the compute kernel) is passed in a McastGemmSpec. TT-Metalium objects are
// created in this order: CBs -> up to 6 data-movement kernels (first-half sender, second-half sender and
// receiver for A and for B; a receiver exists only if the row / column has at least 3 cores or a block is
// a single tile, a second-half sender only if a block has at least 2 tiles) -> compute kernel -> 6
// semaphores -> per-core runtime arguments. build_mcast_gemm calls the parts below (mcast_build_check,
// mcast_make_cbs, mcast_make_kernels, mcast_make_sems, mcast_set_core_args) in this order. Kernel ids,
// semaphore ids and the L1 layout of the CBs follow from the order of creation, so it must not change.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/gemm_host.hpp"
#include "backend_interface.hpp"
#include "common/kernels/mcast_args.h"
#include "common/kernels/gemm_compute_args.h"

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// One additional CB (capacity ntiles pages of `page` bytes), passed to make_cb.
struct McastCB {
    tt::CBIndex idx;
    tt::DataFormat fmt;
    uint32_t ntiles, page;
};

// Everything that distinguishes one multicast matmul program from another.
struct McastGemmSpec {
    // CB_IN0 / CB_IN1: the inputs A and B (double buffered: pm*bw*2 and pn*bw*2 tiles of in_tile bytes).
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b;
    uint32_t in_tile = 0;
    // CB_OUT (c16_fmt): the final output; CB_PARTIAL (c24_fmt): the partial sums carried across K blocks.
    // Both have pm*pn pages of out_tile bytes and share one CB (the same L1 region).
    tt::DataFormat c16_fmt = tt::DataFormat::Float16_b, c24_fmt = tt::DataFormat::Float16_b;
    uint32_t out_tile = 0;
    std::vector<McastCB> extra_cbs;  // created after CB_PARTIAL, in this order
    // Writer: false selects the standard one, true the fused writer (WRITER=2 of mcast_split_dm.cpp),
    // which prefetches the existing S into CB_S before the result is added to it (scheme I).
    bool writer_fused = false;
    std::string compute;               // path of the compute kernel
    tt::tt_metal::ComputeConfig ccfg;  // includes its compile-time arguments
    // If not empty, the runtime arguments given to the compute kernel of every core. A fused writer needs
    // them up to CR_ACC (whether the existing S is added), and build_mcast_gemm takes W_ACC of the writer
    // from here too, so that the compute kernel and the writer agree (if they differed, the compute
    // kernel would wait for S in CB_S while the writer never prefetches it, and the program would hang).
    std::vector<uint32_t> compute_rt;
    uint32_t dA = 0, dB = 0, dW = 0;  // shifts of the first tile of A / B / C, in tiles
};

// Handles of a built program, kept so that its runtime arguments can be replaced and the program reused.
struct McastGemm {
    // The compute kernel (callers replace its runtime arguments per issue). The data-movement kernels are
    // held per core below; the semaphores are used only inside build_mcast_gemm.
    tt::tt_metal::KernelHandle ck = 0;
    // Per core: its kernels and the reader arguments it was built with (the base for retarget_k). The
    // order is cy outer, cx inner, the order in which the runtime arguments were set. ka is the kernel of
    // the A side, kb that of the B side; xa / xb are the 7 second-half sender entries (X_* of
    // mcast_args.h) of the A / B side.
    struct Core {
        tt::tt_metal::CoreCoord core;
        tt::tt_metal::KernelHandle ka, kb;
        std::vector<uint32_t> ra;
        std::vector<uint32_t> xa, xb;
    };
    std::vector<Core> cores;

    // Sets the reader arguments ra (RA_COUNT entries) and the writer arguments wa (W_COUNT entries; the
    // standard writer does not read W_ACC) of one core. The counts are fixed by the table in
    // kernels/mcast_args.h; with one entry too many or too few the kernel would silently read its
    // neighbor, so a mismatch stops here.
    void set_args(tt::tt_metal::Program& prog, const Core& c, const std::vector<uint32_t>& ra,
                  const std::vector<uint32_t>& wa) const {
        namespace ma = mcast_args;
        TT_FATAL(ra.size() == ma::RA_COUNT, "McastGemm::set_args: {} reader arguments (expected {})", ra.size(),
                 (uint32_t)ma::RA_COUNT);
        TT_FATAL(c.xa.size() == ma::X_COUNT && c.xb.size() == ma::X_COUNT,
                 "McastGemm::set_args: {}/{} second-half sender arguments (expected {})", c.xa.size(), c.xb.size(),
                 ma::X_COUNT);
        TT_FATAL(wa.size() == ma::W_COUNT, "McastGemm::set_args: {} writer arguments (expected {})", wa.size(),
                 (uint32_t)ma::W_COUNT);
        std::vector<uint32_t> a = ra;
        a.insert(a.end(), c.xa.begin(), c.xa.end());
        tt::tt_metal::SetRuntimeArgs(prog, c.ka, c.core, a);
        std::vector<uint32_t> b = ra;
        b.insert(b.end(), c.xb.begin(), c.xb.end());
        b.insert(b.end(), wa.begin(), wa.end());
        tt::tt_metal::SetRuntimeArgs(prog, c.kb, c.core, b);
    }

    // Re-points the program along K: relative to the reader arguments of the build, the first tiles of A
    // and B are shifted by aoff and boff, the number of K blocks becomes nb, and these are set on core c
    // together with the writer arguments wa (both the A-side and the B-side kernel receive them). The
    // runtime arguments of the compute kernel, which include the number of K blocks, are set by the
    // caller.
    void retarget_k(tt::tt_metal::Program& prog, const Core& c, uint32_t aoff, uint32_t boff, uint32_t nb,
                    const std::vector<uint32_t>& wa) const {
        namespace ma = mcast_args;
        std::vector<uint32_t> ra = c.ra;
        ra[ma::RA_A_START] += aoff;
        ra[ma::RA_B_START] += boff;
        ra[ma::RA_NUM_BLOCKS] = nb;
        set_args(prog, c, ra, wa);
    }
};

// Preconditions on the per-core shape, checked before the compute-kernel arguments are made and at the
// start of build_mcast_gemm. Configurations from the searches (choose_cfg and those of scheme I) satisfy
// them, but mcast_compute_args divides by pm/sh, pn/sw and Kt/bw, so the check has to come first. A
// violation would not stop the device but silently corrupt the result:
//   sh | pm, sw | pn  the reader pushes pm*bw tiles and the compute kernel pops sh*bw*(pm/sh); if they
//                     do not divide, the push / pop counts of the input differ and the ring of
//                     CB_PARTIAL does not return to its start
//   bw | Kt           Kt/bw truncates, so the tail of K would silently be dropped
static inline void mcast_shape_check(const OptConfig& g, uint32_t Kt, const char* who) {
    TT_FATAL(g.pm > 0 && g.pn > 0 && g.sh > 0 && g.sw > 0 && g.bw > 0,
             "{}: the shape has a zero (pm={} pn={} sh={} sw={} bw={})", who, g.pm, g.pn, g.sh, g.sw, g.bw);
    TT_FATAL(g.pm % g.sh == 0 && g.pn % g.sw == 0, "{}: subblock {}x{} does not divide the per-core output {}x{}", who,
             g.sh, g.sw, g.pm, g.pn);
    TT_FATAL(Kt >= g.bw && Kt % g.bw == 0, "{}: K block width bw={} does not divide Kt={}", who, g.bw, Kt);
}

// Shape parameters of the compute kernel: the first CA_STD_COUNT compile-time arguments, in the layout
// ComputeCT of common/kernels/gemm_compute_args.h (the same as the standard TT-Metalium matmul kernel).
// The caller appends the rest (CA_FUSE onward; spill_gemm_spec). The batch count CA_BATCH is 1 because
// the data-movement kernel has no batches.
static inline std::vector<uint32_t> mcast_compute_args(const OptConfig& g, uint32_t Kt) {
    mcast_shape_check(g, Kt, "mcast_compute_args");  // before the divisions below
    namespace ga = gemm_args;
    const uint32_t pm = g.pm, pn = g.pn, sh = g.sh, sw = g.sw, bw = g.bw;
    const uint32_t in0_ns = pm / sh, in1_ns = pn / sw;
    std::vector<uint32_t> a(ga::CA_STD_COUNT);
    a[ga::CA_IN0_BLOCK_W] = bw;
    a[ga::CA_IN0_NUM_SUBBLOCKS] = in0_ns;
    a[ga::CA_IN0_BLOCK_NUM_TILES] = sh * bw * in0_ns;
    a[ga::CA_IN0_SUBBLOCK_NUM_TILES] = sh * bw;
    a[ga::CA_IN1_NUM_SUBBLOCKS] = in1_ns;
    a[ga::CA_IN1_BLOCK_NUM_TILES] = sw * bw * in1_ns;
    a[ga::CA_IN1_PER_CORE_W] = sw * in1_ns;
    a[ga::CA_NUM_BLOCKS] = Kt / bw;
    a[ga::CA_OUT_SUBBLOCK_H] = sh;
    a[ga::CA_OUT_SUBBLOCK_W] = sw;
    a[ga::CA_OUT_SUBBLOCK_NUM_TILES] = sh * sw;
    a[ga::CA_BATCH] = 1u;
    static_assert(ga::CA_STD_COUNT == 12, "fill the new standard ComputeCT entry here");
    return a;
}

// Total bytes per core of the CBs that build_mcast_gemm creates for this spec: mcast_cb_bytes of
// common/gemm_host.hpp plus extra_cbs. It counts exactly what mcast_make_cbs creates, so a configuration
// search can test whether the real CBs of a candidate fit in the budget (the fused and pair-loop CBs
// come from spill_gemm_spec through extra_cbs).
static inline uint64_t mcast_gemm_l1_bytes(const OptConfig& g, const McastGemmSpec& s) {
    uint64_t extra = 0;
    for (const auto& e : s.extra_cbs) extra += (uint64_t)e.ntiles * e.page;
    return mcast_cb_bytes(g.pm, g.pn, g.bw, s.in_tile, s.out_tile, extra);
}

// Confirms, before any CB is allocated, that the CBs to be created fit in the L1 budget oz_l1_budget(): a
// safety net for configurations whose search did not test mcast_gemm_l1_bytes. `who` names the caller in
// the message.
static inline void mcast_check_l1(const OptConfig& g, const McastGemmSpec& s, const char* who) {
    const uint32_t budget = oz_l1_budget();
    const uint64_t l1 = mcast_gemm_l1_bytes(g, s);
    TT_FATAL(l1 <= (uint64_t)budget,
             "{}: the CBs of the configuration {}x{} cores, {}x{} per core, bw={} take {} KiB, more than the L1 "
             "budget of {} KiB (counted with the output CBs and the additional CBs)",
             who, g.cr, g.cc, g.pm, g.pn, g.bw, l1 / 1024, budget / 1024);
}

// Number of tiles that the destination register of the compute kernel holds: 8 with half sync and 16 with
// full sync for 16-bit tiles, half of that for 32-bit tiles (fp32_dest_acc_en). A subblock of sh*sw tiles
// beyond this would overwrite another section of the register (build_mcast_gemm stops).
static inline uint32_t mcast_dst_tiles(const tt::tt_metal::ComputeConfig& c) {
    return (c.dst_full_sync_en ? 16u : 8u) / (c.fp32_dest_acc_en ? 2u : 1u);
}

// ===== Spec of the shared matmul kernel common/kernels/bmm_int32_spill.cpp =====
// Both schemes build the same CB layout, UnpackToDestFp32 settings and compile-time arguments here.
//   in_fmt    format of the slices that the GEMM reads
//   acc_fmt   format of the partial products (destination register and CB_PARTIAL): Int32 on the INT8
//             path, Float32 otherwise
//   fidelity  MathFidelity of the multiplication (oz_fidelity_rule in common/format.hpp)
//   fuse      on the output path, read the existing S (Int32) from DRAM and add to it (deferred
//             reconstruction of scheme I). Adds CB_S (prefetched S, 2 subblocks) and CB_TMP (FP32
//             intermediate, 1 subblock) and selects the fused writer
//   pairloop  at each pair boundary, fold the FP32 accumulator and add it to the Int32 plane CB_SL in L1
//             (scheme I). Adds CB_TMP and CB_SL (a ring of pm*pn tiles + 1 subblock). Exclusive with fuse
// With fuse or pairloop the output CB_OUT is Int32 (two's complement).
// The runtime arguments of the compute kernel (number of K blocks nb, acc of the fused path, bpp of the
// pair loop; ComputeRT in gemm_compute_args.h) are put into compute_rt by the caller with
// spill_compute_rt, after spill_check_run has checked the ones that change per issue.
struct SpillOpts {
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat acc_fmt = tt::DataFormat::Float32;
    tt::tt_metal::MathFidelity fidelity = tt::tt_metal::MathFidelity::LoFi;
    bool fuse = false, pairloop = false;
};

// The additional CBs of the fused and pair-loop paths, in the order in which they are created (the L1
// layout follows the order of creation). The tiles of one subblock are sh*sw. spill_gemm_spec creates
// them and spill_gemm_l1_bytes counts them, so the estimate of a configuration search cannot disagree
// with the CBs that are actually created.
static inline std::vector<McastCB> spill_extra_cbs(const OptConfig& g, const SpillOpts& o) {
    namespace ga = gemm_args;
    const uint32_t out_tile = tt::tile_size(o.acc_fmt);
    const uint32_t snt = g.sh * g.sw;  // tiles of one subblock
    std::vector<McastCB> cbs;
    if (o.fuse)  // CB_S: the existing S (Int32) that the writer prefetches
        cbs.push_back({(tt::CBIndex)ga::CB_S, tt::DataFormat::Int32, snt * 2, out_tile});
    if (o.fuse || o.pairloop)  // CB_TMP: where the matmul result is parked, still in FP32
        cbs.push_back({(tt::CBIndex)ga::CB_TMP, tt::DataFormat::Float32, snt, out_tile});
    if (o.pairloop)  // CB_SL: Int32 accumulation plane across pairs. The ring is one subblock larger than
        // the plane; otherwise the reserve for writing would overlap the region still being read.
        cbs.push_back({(tt::CBIndex)ga::CB_SL, tt::DataFormat::Int32, g.pm * g.pn + snt, out_tile});
    return cbs;
}

// Total bytes per core of the CBs that a program built from spill_gemm_spec(g, Kt, o) creates:
// mcast_cb_bytes of common/gemm_host.hpp plus the additional CBs above. Independent of Kt, so a
// configuration search can test a candidate without building the spec (mcast_gemm_l1_bytes counts a
// spec that has been built and gives the same number).
static inline uint64_t spill_gemm_l1_bytes(const OptConfig& g, const SpillOpts& o) {
    uint64_t extra = 0;
    for (const auto& e : spill_extra_cbs(g, o)) extra += (uint64_t)e.ntiles * e.page;
    return mcast_cb_bytes(g.pm, g.pn, g.bw, tt::tile_size(o.in_fmt), tt::tile_size(o.acc_fmt), extra);
}

static inline McastGemmSpec spill_gemm_spec(const OptConfig& g, uint32_t Kt, const SpillOpts& o) {
    namespace tm = tt::tt_metal;
    // Fusing (adding to the prefetched S from DRAM) and the pair loop (adding to the plane in L1) are
    // exclusive: with two targets to add to, the destination register (four 32-bit tiles) would not be
    // enough. The compute kernel assumes this and works two tiles at a time.
    TT_FATAL(!(o.fuse && o.pairloop), "spill_gemm_spec: fuse and pairloop cannot both be set");
    // The second pass of the fused / pair-loop path treats the matmul result as FP32 and converts it with
    // typecast<Float32, Int32>. The destination register of the INT8 path holds Int32 (sign-magnitude),
    // which that conversion would corrupt; the INT8 path of scheme I uses neither.
    TT_FATAL(!((o.fuse || o.pairloop) && o.acc_fmt != tt::DataFormat::Float32),
             "spill_gemm_spec: fuse and pairloop require FP32 partial products (partial-product format {})",
             (int)o.acc_fmt);
    TT_FATAL(o.acc_fmt == tt::DataFormat::Float32 || o.acc_fmt == tt::DataFormat::Int32,
             "spill_gemm_spec: the partial-product format must be Float32 or Int32 ({})", (int)o.acc_fmt);
    McastGemmSpec s;
    s.in_fmt = o.in_fmt;
    s.in_tile = tt::tile_size(o.in_fmt);
    // The output of the fused / pair-loop path is Int32 (already added, in two's complement, to the
    // existing S or to CB_SL). CB_PARTIAL holds the partial sums across K blocks and keeps the format of
    // the partial products.
    s.c16_fmt = (o.fuse || o.pairloop) ? tt::DataFormat::Int32 : o.acc_fmt;
    s.c24_fmt = o.acc_fmt;
    s.out_tile = tt::tile_size(o.acc_fmt);
    s.extra_cbs = spill_extra_cbs(g, o);
    namespace ga = gemm_args;
    // FUSE of the compute kernel and the fused writer always go together (a mismatch hangs on CB_S).
    s.writer_fused = o.fuse;
    // Compile-time arguments of the compute kernel: the standard shape (first CA_STD_COUNT entries), then
    // FUSE, the two formats of the typecast (Float32 -> Int32), PAIRLOOP and BLOCKMM (ComputeCT in
    // gemm_compute_args.h).
    std::vector<uint32_t> cargs = mcast_compute_args(g, Kt);
    cargs.resize(ga::CA_COUNT);
    cargs[ga::CA_FUSE] = o.fuse ? 1u : 0u;
    cargs[ga::CA_F32] = (uint32_t)tt::DataFormat::Float32;
    cargs[ga::CA_I32] = (uint32_t)tt::DataFormat::Int32;
    cargs[ga::CA_PAIRLOOP] = o.pairloop ? 1u : 0u;
    cargs[ga::CA_BLOCKMM] = 1u;  // matmul_block (0 would select matmul_tiles, one tile at a time)
    static_assert(ga::CA_COUNT == 17, "fill the new ComputeCT entry here");
    // CB_PARTIAL (partial sums across K blocks) and CB_TMP (FP32 intermediate) must be expanded into the
    // destination register without truncation (UnpackToDestFp32; without it they would be shortened to
    // TF32 and the partial sums would no longer be exact). The same mode is set for the Int32 CBs (CB_S,
    // CB_SL, and CB_PARTIAL of the INT8 path).
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[ga::CB_PARTIAL] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (o.fuse) u2d[ga::CB_S] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (o.fuse || o.pairloop) u2d[ga::CB_TMP] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (o.pairloop) u2d[ga::CB_SL] = tm::UnpackToDestMode::UnpackToDestFp32;
    s.compute = ::ozTTgemm::detail::kernel_path("common/kernels/bmm_int32_spill.cpp");
    s.ccfg = tm::ComputeConfig{
        .math_fidelity = o.fidelity, .fp32_dest_acc_en = true, .unpack_to_dest_mode = u2d, .compile_args = cargs};
    return s;
}

// ===== Choice of the output panel =====
// The output of Mt x Nt tiles is split into panels of pMt x pNt tiles, each computed by one matmul program.
// The candidates are all pairs (dm, dn) of divisors of Mt and Nt that are at most cap, tried in
// descending order of the area dm*dn (for equal areas, larger dm first, then larger dn); cfg_for(dm, dn)
// is called for each, and the first pair for which it returns ok is taken. The first candidate is the
// pair of the largest divisors not exceeding cap. Looking only at that pair would fail whenever it cannot
// be split over the core grid although another pair can (for example Mt = 158 = 2 x 79: 79 is not
// divisible within a grid of 10 rows, but dm = 2 works). The divisor 1 is not a candidate because the
// core grid needs at least 2 rows and 2 columns (choose_cfg).
// If nothing is found, the result has g.ok == false and pMt = pNt = 0.
struct PanelChoice {
    uint32_t pMt = 0, pNt = 0;
    OptConfig g;
};
template <typename CfgFor>
static inline PanelChoice choose_panels(uint32_t Mt, uint32_t Nt, uint32_t cap, CfgFor&& cfg_for) {
    std::vector<std::pair<uint32_t, uint32_t>> cand;
    for (uint32_t dm = std::min(Mt, cap); dm >= 2; --dm) {
        if (Mt % dm) continue;
        for (uint32_t dn = std::min(Nt, cap); dn >= 2; --dn)
            if (Nt % dn == 0) cand.push_back({dm, dn});
    }
    // Order by descending area, keeping the generation order (larger dm, then larger dn) among equals.
    std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
        return (uint64_t)a.first * a.second > (uint64_t)b.first * b.second;
    });
    for (const auto& c : cand) {
        const OptConfig g = cfg_for(c.first, c.second);
        if (g.ok) return {c.first, c.second, g};
    }
    return {};
}

// Checks the runtime arguments of the compute kernel that change per issue. nb is the number of K blocks
// of this program and bpp the number of K blocks per pair of the pair loop. With bpp = 0, or nb not a
// multiple of bpp, the last block would not be a pair boundary and the sum in CB_SL would never reach the
// output: a wrong result without any stop.
static inline void spill_check_run(uint32_t nb, bool pairloop, uint32_t bpp) {
    TT_FATAL(nb > 0, "bmm_int32_spill: the number of K blocks is 0");
    if (pairloop)
        TT_FATAL(
            bpp > 0 && nb % bpp == 0,
            "bmm_int32_spill: the K blocks per pair of the pair loop, bpp={}, do not divide the number of K blocks {}",
            bpp, nb);
}

// Runtime arguments of bmm_int32_spill.cpp (ComputeRT in gemm_compute_args.h). A configuration with
// neither fuse nor pair loop passes only the number of K blocks (the kernel reads CR_ACC only with FUSE
// and CR_BPP only with PAIRLOOP).
static inline std::vector<uint32_t> spill_compute_rt(uint32_t nb) {
    std::vector<uint32_t> v(gemm_args::CR_NUM_BLOCKS + 1);
    v[gemm_args::CR_NUM_BLOCKS] = nb;
    return v;
}
static inline std::vector<uint32_t> spill_compute_rt(uint32_t nb, uint32_t acc, uint32_t bpp) {
    namespace ga = gemm_args;
    std::vector<uint32_t> v(ga::CR_COUNT);
    v[ga::CR_NUM_BLOCKS] = nb;
    v[ga::CR_ACC] = acc;
    v[ga::CR_BPP] = bpp;
    static_assert(ga::CR_COUNT == 3, "fill the new ComputeRT entry here");
    return v;
}

// ---- The parts of build_mcast_gemm (called in the order given at the top of this file) ----
// Checks before building: the core grid is at least 2 x 2 (the multicast needs a second core in each row
// and column); the shape preconditions (mcast_shape_check); core grid x per-core output fits in the
// matrix; a subblock of sh*sw tiles fits in the destination register; the CB pages (in_tile, out_tile)
// equal the tile sizes of the formats (otherwise the page count and the tile size that the kernel reads
// disagree; CB_OUT and CB_PARTIAL share one region with pages of out_tile); and a fused writer has CB_S
// to prefetch S into and CR_ACC on the compute side (without them the compute kernel waits on CB_S
// forever).
static inline void mcast_build_check(const OptConfig& g, uint32_t Mt, uint32_t Nt, uint32_t Kt,
                                     const McastGemmSpec& s) {
    TT_FATAL(g.cr >= 2 && g.cc >= 2, "build_mcast_gemm: the core grid {}x{} must be at least 2 in both directions",
             g.cr, g.cc);
    mcast_shape_check(g, Kt, "build_mcast_gemm");
    TT_FATAL(g.cr * g.pm <= Mt && g.cc * g.pn <= Nt,
             "build_mcast_gemm: core grid {}x{} x per-core output {}x{} exceeds the matrix of {}x{} tiles", g.cr, g.cc,
             g.pm, g.pn, Mt, Nt);
    const uint32_t dst_cap = mcast_dst_tiles(s.ccfg);
    TT_FATAL(g.sh * g.sw <= dst_cap,
             "build_mcast_gemm: subblock {}x{} does not fit in the {} tiles of the destination register", g.sh, g.sw,
             dst_cap);
    TT_FATAL(tt::tile_size(s.in_fmt) == s.in_tile,
             "build_mcast_gemm: in_tile={} differs from the tile size {} of the input format", s.in_tile,
             tt::tile_size(s.in_fmt));
    TT_FATAL(tt::tile_size(s.c16_fmt) == s.out_tile && tt::tile_size(s.c24_fmt) == s.out_tile,
             "build_mcast_gemm: out_tile={} differs from the tile sizes {} / {} of the formats of CB_OUT / CB_PARTIAL",
             s.out_tile, tt::tile_size(s.c16_fmt), tt::tile_size(s.c24_fmt));
    if (s.writer_fused) {
        bool has_c2 = false;
        for (const auto& e : s.extra_cbs) has_c2 = has_c2 || e.idx == (tt::CBIndex)gemm_args::CB_S;
        TT_FATAL(has_c2, "build_mcast_gemm: the fused writer has no CB_S to prefetch S into");
        TT_FATAL(
            s.compute_rt.size() > gemm_args::CR_ACC,
            "build_mcast_gemm: the compute-kernel runtime arguments of a fused-writer configuration lack CR_ACC ({} entries; use spill_compute_rt(nb, acc, bpp))",
            s.compute_rt.size());
    }
}

// CBs, counted exactly as in mcast_gemm_l1_bytes: CB_IN0 and CB_IN1 (A and B, double buffered), CB_OUT
// and CB_PARTIAL in one shared region, then the additional CBs in the order of extra_cbs.

static inline void mcast_make_cbs(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                                  const OptConfig& g, const McastGemmSpec& s) {
    namespace tm = tt::tt_metal;
    namespace ga = gemm_args;
    const McastIoTiles io = mcast_io_cb_tiles(g.pm, g.pn, g.bw);
    const tt::CBIndex cb_in0 = (tt::CBIndex)ga::CB_IN0, cb_in1 = (tt::CBIndex)ga::CB_IN1,
                      cb_out = (tt::CBIndex)ga::CB_OUT, cb_part = (tt::CBIndex)ga::CB_PARTIAL;
    make_cb(program, all_cores, cb_in0, s.in_fmt, io.c0, s.in_tile);
    make_cb(program, all_cores, cb_in1, s.in_fmt, io.c1, s.in_tile);
    tm::CreateCircularBuffer(
        program, all_cores,
        tm::CircularBufferConfig(g.pm * g.pn * s.out_tile, {{cb_out, s.c16_fmt}, {cb_part, s.c24_fmt}})
            .set_page_size(cb_out, s.out_tile)
            .set_page_size(cb_part, s.out_tile));
    for (const auto& e : s.extra_cbs) make_cb(program, all_cores, e.idx, e.fmt, e.ntiles, e.page);
}

// Handles of the data-movement kernels (kernels/mcast_split_dm.cpp). a_* is the A side (RISCV_1), b_* the
// B side (RISCV_0, which is also the writer). *_s0 sends the first half (A: leftmost column, B: top row),
// *_s1 the second half (rightmost column, bottom row), *_r only receives (the cores in between; left 0
// and unused if not created). a2 / b2: whether a second half is sent (a block has at least 2 tiles).
struct McastDmKernels {
    tt::tt_metal::KernelHandle a_s0 = 0, a_s1 = 0, a_r = 0, b_s0 = 0, b_s1 = 0, b_r = 0;
    bool a2 = false, b2 = false;
    // A-side kernel of a core in column cx (grid width cc); B-side kernel of a core in row cy (height cr).
    tt::tt_metal::KernelHandle a_of(uint32_t cx, uint32_t cc) const {
        return (cx == 0) ? a_s0 : (a2 && cx == cc - 1) ? a_s1 : a_r;
    }
    tt::tt_metal::KernelHandle b_of(uint32_t cy, uint32_t cr) const {
        return (cy == 0) ? b_s0 : (b2 && cy == cr - 1) ? b_s1 : b_r;
    }
};

// Creates the data-movement kernels (up to 6, in this order) and the compute kernel. The compile-time
// arguments are the TensorAccessorArgs of A, B and C, concatenated in this order; the role of each kernel
// is passed as defines, together with the kernel's diagnostic selectors (always 0 in the library). The
// handle of the compute kernel is returned in ck.
static inline McastDmKernels mcast_make_kernels(tt::tt_metal::Program& program,
                                                const tt::tt_metal::CoreRange& all_cores, const OptConfig& g,
                                                const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Abuf,
                                                const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Bbuf,
                                                const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Cbuf,
                                                const McastGemmSpec& s, tt::tt_metal::KernelHandle& ck) {
    namespace tm = tt::tt_metal;
    using tm::CoreRange;
    std::vector<uint32_t> ct;
    tm::TensorAccessorArgs(*Abuf).append_to(ct);
    tm::TensorAccessorArgs(*Bbuf).append_to(ct);
    tm::TensorAccessorArgs(*Cbuf).append_to(ct);
    const std::string dm_kernel = ::ozTTgemm::detail::kernel_path("common/kernels/mcast_split_dm.cpp");
    const std::string W = s.writer_fused ? "2" : "1";
    // operand: 0 = A, 1 = B. role: 1 = first-half sender, 2 = receiver, 3 = second-half sender. Only the
    // B side writes the output (WRITER 1 = standard, 2 = fused).
    auto mk = [&](const CoreRange& rng, int operand, int role) {
        const bool isB = operand == 1;
        const std::map<std::string, std::string> defs = {{"OPERAND", std::to_string(operand)},
                                                         {"ROLE", std::to_string(role)},
                                                         {"WRITER", isB ? W : std::string("0")},
                                                         {"NODRAM", "0"},
                                                         {"NOMCAST", "0"},
                                                         {"NOWRITE", "0"}};
        return tm::CreateKernel(program, dm_kernel, rng,
                                tm::DataMovementConfig{.processor = isB ? tm::DataMovementProcessor::RISCV_0
                                                                        : tm::DataMovementProcessor::RISCV_1,
                                                       .noc = isB ? tm::NOC::NOC_0 : tm::NOC::NOC_1,
                                                       .compile_args = ct,
                                                       .defines = defs});
    };
    // An operand whose block is a single tile (pm*bw == 1, ...) has an empty second half, so no
    // second-half sender is placed and the first-half sender distributes everything (the opposite end
    // gets a receiver kernel; the same condition as in mcast_split_dm.cpp).
    McastDmKernels k;
    k.a2 = g.pm * g.bw >= 2;
    k.b2 = g.pn * g.bw >= 2;
    const std::size_t xl = g.cc - 1, yl = g.cr - 1;             // rightmost column, bottom row
    k.a_s0 = mk(CoreRange({0, 0}, {0, yl}), 0, 1);              // A: leftmost column sends the first half
    if (k.a2) k.a_s1 = mk(CoreRange({xl, 0}, {xl, yl}), 0, 3);  //    rightmost column sends the second half
    if (g.cc > 2 || !k.a2) k.a_r = mk(CoreRange({1, 0}, {k.a2 ? xl - 1 : xl, yl}), 0, 2);
    k.b_s0 = mk(CoreRange({0, 0}, {xl, 0}), 1, 1);              // B: top row sends the first half
    if (k.b2) k.b_s1 = mk(CoreRange({0, yl}, {xl, yl}), 1, 3);  //    bottom row sends the second half
    if (g.cr > 2 || !k.b2) k.b_r = mk(CoreRange({0, 1}, {xl, k.b2 ? yl - 1 : yl}), 1, 2);
    ck = tm::CreateKernel(program, s.compute, all_cores, s.ccfg);
    return k;
}

// Semaphores that synchronize the multicasts. For A and for B: the sender's wait until the receivers have
// room (free), the flag of the first half (flag) and the flag of the second half (flag2). The ids follow
// from the order of creation, so it must not change.
struct McastSemIds {
    uint32_t A_free, A_flag, B_free, B_flag, A_flag2, B_flag2;
};
static inline McastSemIds mcast_make_sems(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores) {
    namespace tm = tt::tt_metal;
    McastSemIds m{};
    m.A_free = tm::CreateSemaphore(program, all_cores, INVALID);
    m.A_flag = tm::CreateSemaphore(program, all_cores, INVALID);
    m.B_free = tm::CreateSemaphore(program, all_cores, INVALID);
    m.B_flag = tm::CreateSemaphore(program, all_cores, INVALID);
    m.A_flag2 = tm::CreateSemaphore(program, all_cores, INVALID);
    m.B_flag2 = tm::CreateSemaphore(program, all_cores, INVALID);
    return m;
}

// Per-core runtime arguments: the first tiles of A / B that the core handles and its multicast peers
// (cy outer, cx inner). acc of a fused writer is the same value as CR_ACC of the compute kernel (see
// McastGemmSpec::compute_rt).
static inline void mcast_set_core_args(tt::tt_metal::Program& program,
                                       std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, McastGemm& r,
                                       const OptConfig& g, uint32_t Nt, uint32_t Kt, uint32_t Aaddr, uint32_t Baddr,
                                       uint32_t Caddr, const McastDmKernels& k, const McastSemIds& sem,
                                       const McastGemmSpec& s) {
    namespace tm = tt::tt_metal;
    using tm::CoreCoord;
    const uint32_t pm = g.pm, pn = g.pn, sh = g.sh, sw = g.sw, bw = g.bw, cr = g.cr, cc = g.cc;
    const uint32_t wacc = s.writer_fused ? s.compute_rt[gemm_args::CR_ACC] : 0u;
    for (uint32_t cy = 0; cy < cr; cy++)
        for (uint32_t cx = 0; cx < cc; cx++) {
            CoreCoord core = {(std::size_t)cx, (std::size_t)cy};
            // Peers: left end, its right neighbor and right end of the row; top end, the one below it and
            // bottom end of the column.
            auto Lp = md->worker_core_from_logical_core(CoreCoord{0, (std::size_t)cy});
            auto Lp1 = md->worker_core_from_logical_core(CoreCoord{1, (std::size_t)cy});
            auto Rp = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cc - 1, (std::size_t)cy});
            auto Tp = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, 0});
            auto Tp1 = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, 1});
            auto Bp = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, (std::size_t)cr - 1});
            // The first half goes to the columns 1..cc-1 of the row (the rows 1..cr-1 of the column).
            auto ra = mcast_reader_args(Aaddr, Baddr, Nt, Kt, pm, pn, bw, cr, cc, cx, cy, {Lp, Lp1, Rp, Tp, Tp1, Bp},
                                        {sem.A_free, sem.A_flag, sem.B_free, sem.B_flag}, s.dA, s.dB);
            auto wa = mcast_writer_args(Caddr, Nt, pm, pn, sh, sw, cx, cy, s.dW, wacc);
            // Second-half sender entries (X_* of kernels/mcast_args.h): the second-half sender core (A:
            // right end of the row, B: bottom end of the column), its destination rectangle (A: columns
            // 0..cc-2, B: rows 0..cr-2; the hi corner first) and the flag of the second half.
            auto A2 = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cc - 2, (std::size_t)cy});
            auto B2 = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, (std::size_t)cr - 2});

            auto second = [](const CoreCoord& s1, const CoreCoord& hi, const CoreCoord& lo, uint32_t fsem) {
                namespace ma = mcast_args;
                std::vector<uint32_t> x(ma::X_COUNT, 0xFFFFFFFFu);
                x[ma::X_S1_X - ma::X_S1_X] = (uint32_t)s1.x;
                x[ma::X_S1_Y - ma::X_S1_X] = (uint32_t)s1.y;
                x[ma::X_HI_X - ma::X_S1_X] = (uint32_t)hi.x;
                x[ma::X_HI_Y - ma::X_S1_X] = (uint32_t)hi.y;
                x[ma::X_LO_X - ma::X_S1_X] = (uint32_t)lo.x;
                x[ma::X_LO_Y - ma::X_S1_X] = (uint32_t)lo.y;
                x[ma::X_FLAG_SEM - ma::X_S1_X] = fsem;
                mcast_args_filled(x, "build_mcast_gemm (second-half sender)");
                return x;
            };
            McastGemm::Core c{core,
                              k.a_of(cx, cc),
                              k.b_of(cy, cr),
                              std::move(ra),
                              second(Rp, A2, Lp, sem.A_flag2),
                              second(Bp, B2, Tp, sem.B_flag2)};
            r.set_args(program, c, c.ra, wa);
            if (!s.compute_rt.empty()) tm::SetRuntimeArgs(program, r.ck, core, s.compute_rt);
            r.cores.push_back(std::move(c));
        }
}

// Adds the CBs, kernels, semaphores and per-core runtime arguments of one multicast matmul to `program`
// (calling the parts above in the order given at the top of this file). Mt, Nt, Kt are the tile counts of
// the matrices that this program addresses (they become the strides of the reader and the writer). The
// buffers of A, B and C are created by the caller (their addresses and TensorAccessorArgs are used).
// Everything that is built follows from the arguments and the spec.
static inline McastGemm build_mcast_gemm(tt::tt_metal::Program& program,
                                         std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const OptConfig& g,
                                         uint32_t Mt, uint32_t Nt, uint32_t Kt,
                                         const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Abuf,
                                         const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Bbuf,
                                         const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Cbuf,
                                         const McastGemmSpec& s) {
    mcast_build_check(g, Mt, Nt, Kt, s);
    const tt::tt_metal::CoreRange all_cores({0, 0}, {(std::size_t)g.cc - 1, (std::size_t)g.cr - 1});
    mcast_make_cbs(program, all_cores, g, s);
    McastGemm r;
    const McastDmKernels k = mcast_make_kernels(program, all_cores, g, Abuf, Bbuf, Cbuf, s, r.ck);
    const McastSemIds sem = mcast_make_sems(program, all_cores);
    mcast_set_core_args(program, md, r, g, Nt, Kt, (uint32_t)Abuf->address(), (uint32_t)Bbuf->address(),
                        (uint32_t)Cbuf->address(), k, sem, s);
    return r;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
