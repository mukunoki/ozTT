// SPDX-License-Identifier: Apache-2.0
#pragma once
// Host-side helpers shared by every stage of both Ozaki schemes.
//
// Provided here: the distribution of tiles over the compute grid (GridSpan), the creation of circular
// buffers (CBs) and DRAM buffers (make_cb, mesh_dram_buffer), the free DRAM (dram_free), the L1 budget
// (oz_l1_budget) and the size of the CBs of the GEMM stage (mcast_cb_bytes), the search for the per-core
// matmul configuration (choose_cfg), and the runtime arguments of the multicast data-movement kernel
// (mcast_reader_args, mcast_writer_args). The multicast matmul program itself is built in
// common/mcast_gemm.hpp, which includes this header.
//
// Unless stated otherwise, sizes are in tiles (32x32 elements); Mt, Nt, Kt are M/32, N/32, K/32.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>
#include <array>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "common/kernels/mcast_args.h"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Distributes ntiles tiles evenly over the whole compute grid. Every stage other than the GEMM uses all
// cores and hands out the tiles in order, so the enumeration of the cores, the core range and the share
// of each core are defined once, here. Core i handles the tiles [begin(i), begin(i)+count(i)); `per` is
// the share of a core, rounded up, so trailing cores may receive fewer tiles or none.
struct GridSpan {
    std::vector<tt::tt_metal::CoreCoord> cores;  // row-major over the ncx x ncy grid
    uint32_t ntiles = 0, per = 0, ncx = 0, ncy = 0;
    uint32_t begin(size_t i) const { return std::min(per * (uint32_t)i, ntiles); }
    uint32_t count(size_t i) const { return std::min(per, ntiles - begin(i)); }

    // Number of tiles of every core, in the order of `cores`.
    std::vector<uint32_t> counts() const {
        std::vector<uint32_t> c(cores.size());
        for (size_t i = 0; i < c.size(); ++i) c[i] = count(i);
        return c;
    }
    tt::tt_metal::CoreRange range() const {
        return tt::tt_metal::CoreRange({0, 0}, {(std::size_t)ncx - 1, (std::size_t)ncy - 1});
    }
};
static inline GridSpan full_grid_span(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, uint32_t ntiles) {
    auto gd = md->compute_with_storage_grid_size();
    uint32_t ncx = (uint32_t)gd.x, ncy = (uint32_t)gd.y;
    if (!ncx) ncx = 1;
    if (!ncy) ncy = 1;
    GridSpan g;
    g.ncx = ncx;
    g.ncy = ncy;
    for (uint32_t y = 0; y < ncy; y++)
        for (uint32_t x = 0; x < ncx; x++) g.cores.push_back({(std::size_t)x, (std::size_t)y});
    uint32_t ncores = std::min<uint32_t>((uint32_t)g.cores.size(), ntiles ? ntiles : 1);
    g.ntiles = ntiles;
    g.per = (ntiles + ncores - 1) / ncores;
    return g;
}

// Creates one CB. Written out by hand, the same index has to be given to both CircularBufferConfig and
// set_page_size, and changing only one of them breaks silently. The capacity is ntiles pages; a page is
// one tile of fmt by default and `page` bytes if given (some work areas treat a whole tile of another
// size as one page).
static inline void make_cb(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& cores, tt::CBIndex idx,
                           tt::DataFormat fmt, uint32_t ntiles, uint32_t page = 0) {
    const uint32_t ps = page ? page : (uint32_t)tt::tile_size(fmt);
    tt::tt_metal::CreateCircularBuffer(
        program, cores, tt::tt_metal::CircularBufferConfig(ntiles * ps, {{idx, fmt}}).set_page_size(idx, ps));
}

// Creates one DRAM buffer of ntiles pages of page_size bytes (one tile = one page). With top_down the
// buffer is taken from the top end of DRAM (Ozaki scheme II places the slices of some paths there);
// otherwise the direction is left to the allocator (bottom up).
static inline std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> mesh_dram_buffer(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, size_t ntiles, uint32_t page_size,
    bool top_down = false) {
    tt::tt_metal::distributed::ReplicatedBufferConfig bc{.size = (size_t)page_size * ntiles};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dc{.page_size = page_size,
                                                          .buffer_type = tt::tt_metal::BufferType::DRAM};
    if (top_down) dc.bottom_up = false;
    return tt::tt_metal::distributed::MeshBuffer::create(bc, dc, md.get());
}

// Free DRAM in bytes. get_statistics returns per-bank values, which are multiplied by the number of banks
// to give the total over all banks (a buffer is spread evenly over the banks, at the same address in
// each). largest_block is the largest contiguous free block, also over all banks: a necessary condition
// for one buffer to fit. This only queries; nothing is allocated. Both schemes use it to decide the number
// of output row bands before allocating (plan_panels in ozaki1/plan.hpp, plan_bands in ozaki2/plan.hpp).
struct DramFree {
    uint64_t total = 0, largest_block = 0;
};
static inline DramFree dram_free(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md) {
    using tt::tt_metal::BufferType;
    const auto st = md->allocator()->get_statistics(BufferType::DRAM);
    const uint64_t nbank = md->allocator()->get_num_banks(BufferType::DRAM);
    return {(uint64_t)st.total_free_bytes * nbank, (uint64_t)st.largest_free_block_bytes * nbank};
}

// Fills a whole buffer with the 32-bit word `word` and waits for completion (the evaluation hooks write a
// sentinel this way; never part of a timed section). A buffer can exceed 10 GB, so no host array of the
// full size is made: sections of about 64 MiB, a multiple of the page size, are written from one array
// (the write copies the host array into the command queue before returning, so the same array can be
// passed again).
static inline void fill_mesh_buffer_u32(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& buf,
                                        uint32_t word) {
    namespace dist = tt::tt_metal::distributed;
    if (!buf) return;
    auto& cq = md->mesh_command_queue();
    const uint64_t total = buf->size(), page = buf->page_size();
    const uint64_t chunk = std::max<uint64_t>(page, ((64ull << 20) / page) * page);
    const std::vector<uint32_t> src((size_t)((chunk + 3) / 4), word);
    const dist::MeshCoordinateRange all(md->shape());
    for (uint64_t off = 0; off < total; off += chunk) {
        const uint64_t n = std::min<uint64_t>(chunk, total - off);  // the tail is a multiple of page too
        cq.enqueue_write_shard_to_sub_grid(*buf, src.data(), all, false, tt::tt_metal::BufferRegion(off, n));
    }
    dist::Finish(cq);
}

// Per-core configuration of one multicast matmul program:
//   ok      a configuration was found
//   cr, cc  core grid used, rows x columns (both at least 2)
//   pm, pn  output tiles per core, rows x columns (the program computes cr*pm x cc*pn tiles)
//   sh, sw  output subblock in tiles (what the compute kernel holds in the destination register at once);
//           sh divides pm and sw divides pn
//   bw      width of one K block in tiles (in0_block_w): the kernel multiplies bw tiles of K per block and
//           carries the partial sums across blocks (the number of K blocks, Kt / bw, is a runtime argument
//           of the program)
struct OptConfig {
    bool ok = false;
    uint32_t cr = 0, cc = 0, pm = 0, pn = 0, sh = 0, sw = 0, bw = 0;
};

// The divisors of n in descending order.
inline std::vector<uint32_t> divisors_desc(uint32_t n) {
    std::vector<uint32_t> d;
    for (uint32_t i = n; i >= 1; --i)
        if (n % i == 0) d.push_back(i);
    return d;
}

// Candidate subblocks (h, w), largest area first. The destination register limits the area: with
// fp32_dest_acc_en, which every GEMM of this library uses for exact accumulation, it holds 4 tiles.
// clang-format off
static constexpr std::array<std::pair<uint32_t, uint32_t>, 8> SB_CHOICES = {{
    {2, 2}, {4, 1}, {1, 4},  // area 4
    {3, 1}, {1, 3},          // area 3
    {2, 1}, {1, 2}, {1, 1},  // area 1-2
}};
// clang-format on

// Picks the first entry of SB_CHOICES that divides (pm, pn) (the table is ordered by area, so this is
// the largest one in the table). Shapes that are not in the table (3x3, 5x2, ...) are never chosen even
// if they would qualify. Returns false if no entry fits.
inline bool pick_subblock(uint32_t pm, uint32_t pn, uint32_t& sh, uint32_t& sw) {
    for (const auto& s : SB_CHOICES) {
        if (pm % s.first == 0 && pn % s.second == 0) {
            sh = s.first;
            sw = s.second;
            return true;
        }
    }
    return false;
}

// L1 budget in bytes that the configuration searches of both schemes give to the CBs of one core. L1 is
// 1536 KiB per core; the reserved area and the kernel-side buffers need somewhat more than 64 KiB, and
// this value leaves about another 64 KiB of margin below that limit. Exceeding L1 fails explicitly when
// the program is created, so a budget that is too large cannot corrupt results silently.
inline uint32_t oz_l1_budget() { return 1408u * 1024u; }

// Number of tiles of the input CBs that build_mcast_gemm (common/mcast_gemm.hpp) creates: c0 for A and
// c1 for B, double buffered, i.e. twice one block (pm x bw and bw x pn tiles).
struct McastIoTiles {
    uint32_t c0, c1;
};
static inline McastIoTiles mcast_io_cb_tiles(uint32_t pm, uint32_t pn, uint32_t bw) {
    return {pm * bw * 2u, pn * bw * 2u};
}

// Total bytes per core of the CBs that build_mcast_gemm creates: the input CBs of A and B (pages of
// in_tile bytes; mcast_io_cb_tiles), the output CB and the partial-sum CB, which share one L1 region of
// pm*pn pages of out_tile bytes, and `extra` bytes of additional CBs. The estimate of the configuration
// search (cfg_fits) and the count of the CBs actually created (mcast_gemm_l1_bytes in
// common/mcast_gemm.hpp) both use this one formula.
static inline uint64_t mcast_cb_bytes(uint32_t pm, uint32_t pn, uint32_t bw, uint32_t in_tile, uint32_t out_tile,
                                      uint64_t extra = 0) {
    const McastIoTiles io = mcast_io_cb_tiles(pm, pn, bw);
    return ((uint64_t)io.c0 + io.c1) * in_tile + (uint64_t)pm * pn * out_tile + extra;
}

// Whether the CBs of a configuration fit in the L1 budget [bytes], assuming no additional CBs. Estimating
// in bytes of the actual tile sizes matters for the narrow formats, which would otherwise leave L1 unused.
inline bool cfg_fits(uint32_t pm, uint32_t pn, uint32_t bw, uint32_t in_tile, uint32_t out_tile, uint32_t budget) {
    return mcast_cb_bytes(pm, pn, bw, in_tile, out_tile) <= (uint64_t)budget;
}

// Chooses the configuration of one matmul program for Mt x Nt output tiles and Kt tiles along K on a
// gy x gx core grid. The priorities are (a) as many cores as possible, then (b) the largest K block width
// bw, i.e. the fewest round trips of the partial sums through L1. Remaining ties go to the first in the
// enumeration order (larger cr, then larger cc): cr and cc only take divisors of Mt and Nt, so the output
// area per core pm*pn = Mt*Nt/(cr*cc) is the same whenever the number of cores is.
//   in_tile, out_tile  bytes per tile of the input format and of the partial products
//   budget     L1 budget in bytes (see oz_l1_budget)
//
// Supported shapes: Mt must have a divisor in 2..gy and Nt one in 2..gx (the multicast needs cr, cc >= 2).
// There is no condition on Kt, because bw = 1 is always a candidate. If the condition does not hold (for
// example Nt = 13 on a grid 11 wide: a prime larger than the grid) no configuration is found and ok stays
// false; the caller must then stop or try another shape. A wrong result is never produced.
inline OptConfig choose_cfg(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t gy, uint32_t gx, uint32_t in_tile,
                            uint32_t out_tile, uint32_t budget) {
    OptConfig best;
    uint32_t bc = 0, bbw = 0;

    for (uint32_t cr = std::min(gy, Mt); cr >= 2; --cr) {
        if (Mt % cr) continue;
        uint32_t pm = Mt / cr;
        for (uint32_t cc = std::min(gx, Nt); cc >= 2; --cc) {
            if (Nt % cc) continue;
            uint32_t pn = Nt / cc;
            uint32_t sh = 0, sw = 0;
            if (!pick_subblock(pm, pn, sh, sw)) continue;
            uint32_t bw = 0;

            // The largest bw that fits in L1.
            for (uint32_t cand : divisors_desc(Kt)) {
                if (!cfg_fits(pm, pn, cand, in_tile, out_tile, budget)) continue;
                bw = cand;
                break;
            }
            if (!bw) continue;
            const uint32_t cores = cr * cc;
            if (cores > bc || (cores == bc && bw > bbw)) {
                bc = cores;
                bbw = bw;
                best = {true, cr, cc, pm, pn, sh, sw, bw};
            }
        }
    }
    return best;
}

// ===== Runtime arguments of the multicast matmul data-movement kernel =====
// The kernel common/kernels/mcast_split_dm.cpp reads these. The layout is that of the reader and writer of
// the TT-Metalium matmul example without the batch arguments (the GEMM stage has no batches). The indices
// are the table common/kernels/mcast_args.h (RA_*, OP_*, MC_*, W_*), which the kernel reads by the same
// names. A shift by one element would still compile and only break silently at run time, so
// McastGemm::set_args checks the counts against the table.
// A multicast rectangle is given by its two corners (hi, lo); hi is the corner with the larger
// coordinates (A: R and L1, the columns 1..cc-1 of the row; B: B and T1, the rows 1..cr-1 of the column).
//
// dA / dB / dW shift the first tile that is read from A / B and written to C, in tiles (a program that
// works on a part of a larger buffer: K blocks, concatenated slices, panels).

// Peer cores of the multicast (device coordinates from worker_core_from_logical_core): left end (L), its
// right neighbor (L1) and right end (R) of the row of the core; top end (T), the one below it (T1) and
// bottom end (B) of its column.
struct McastPeers {
    tt::tt_metal::CoreCoord L, L1, R, T, T1, B;
};
// Semaphores of A (s0*) and B (s1*): sender side (waits until the receivers have room) and receiver flag.
struct McastSems {
    uint32_t s0s, s0r, s1s, s1r;
};

// The argument arrays are filled by the names of the table, so reordering the table cannot swap values;
// an entry that was forgotten keeps the sentinel 0xFFFFFFFF and is caught here.
static inline void mcast_args_filled(const std::vector<uint32_t>& a, const char* who) {
    for (size_t i = 0; i < a.size(); ++i)
        TT_FATAL(a[i] != 0xFFFFFFFFu, "{}: argument {} was not filled (mismatch with the table in mcast_args.h)", who,
                 i);
}
// Reader arguments (RA_COUNT entries) of core (cx, cy) of a cr x cc grid: where its block of A (row cy)
// and of B (column cx) starts and how to step through K, and the two multicasts.
static inline std::vector<uint32_t> mcast_reader_args(uint32_t Aaddr, uint32_t Baddr, uint32_t Nt, uint32_t Kt,
                                                      uint32_t pm, uint32_t pn, uint32_t bw, uint32_t cr, uint32_t cc,
                                                      uint32_t cx, uint32_t cy, const McastPeers& p, const McastSems& s,
                                                      uint32_t dA = 0, uint32_t dB = 0) {
    namespace ma = mcast_args;
    using tt::tt_metal::CoreCoord;
    TT_FATAL(bw > 0, "mcast_reader_args: bw=0");
    std::vector<uint32_t> a(ma::RA_COUNT, 0xFFFFFFFFu);

    // The 7 operand entries. A: Mt x Kt tiles, row-major; the block of this core starts at row pm*cy and
    // advances bw tiles along K per block. B: Kt x Nt tiles; the block starts at column pn*cx and advances
    // bw rows (bw*Nt tiles) per block.
    auto operand = [&](uint32_t base, uint32_t addr, uint32_t start, uint32_t sw_, uint32_t sh_, uint32_t next,
                       uint32_t bw_, uint32_t bt) {
        a[base + ma::OP_ADDR] = addr;
        a[base + ma::OP_START] = start;
        a[base + ma::OP_STRIDE_W] = sw_;
        a[base + ma::OP_STRIDE_H] = sh_;
        a[base + ma::OP_NEXT_BLOCK] = next;
        a[base + ma::OP_BLOCK_W] = bw_;
        a[base + ma::OP_BLOCK_TILES] = bt;
    };
    operand(ma::RA_A, Aaddr, Kt * pm * cy + dA, 1u, Kt, bw, bw, bw * pm);
    operand(ma::RA_B, Baddr, pn * cx + dB, 1u, Nt, bw * Nt, pn, pn * bw);
    a[ma::RA_NUM_BLOCKS] = Kt / bw;

    // The 9 multicast entries: hi and lo corner of the rectangle, number of destinations, the first-half
    // sender core, and the two semaphores.
    auto mcast = [&](uint32_t base, const CoreCoord& hi, const CoreCoord& lo, uint32_t ndest, const CoreCoord& snd,
                     uint32_t ssem, uint32_t fsem) {
        a[base + ma::MC_HI_X] = (uint32_t)hi.x;
        a[base + ma::MC_HI_Y] = (uint32_t)hi.y;
        a[base + ma::MC_LO_X] = (uint32_t)lo.x;
        a[base + ma::MC_LO_Y] = (uint32_t)lo.y;
        a[base + ma::MC_NUM_DESTS] = ndest;
        a[base + ma::MC_SENDER_X] = (uint32_t)snd.x;
        a[base + ma::MC_SENDER_Y] = (uint32_t)snd.y;
        a[base + ma::MC_SENDER_SEM] = ssem;
        a[base + ma::MC_FLAG_SEM] = fsem;
    };
    mcast(ma::RA_MC_A, p.R, p.L1, cc - 1, p.L, s.s0s, s.s0r);
    mcast(ma::RA_MC_B, p.B, p.T1, cr - 1, p.T, s.s1s, s.s1r);
    mcast_args_filled(a, "mcast_reader_args");
    return a;
}

// Writer arguments (W_COUNT entries) of core (cx, cy): its pm x pn output tiles start at tile
// cx*pn + cy*pm*Nt + dW of C (Nt tiles per row) and are written subblock by subblock (sh x sw).
// W_ACC is read only by the fused writer (WRITER=2 of mcast_split_dm.cpp); the standard writer reads the
// first 11 entries, and every writer is passed all 12 (McastGemm::set_args checks the count). The row
// stride of the output is determined by Nt alone, so the number of rows is not needed.
static inline std::vector<uint32_t> mcast_writer_args(uint32_t Caddr, uint32_t Nt, uint32_t pm, uint32_t pn,
                                                      uint32_t sh, uint32_t sw, uint32_t cx, uint32_t cy,
                                                      uint32_t dW = 0, uint32_t acc = 0) {
    namespace ma = mcast_args;
    TT_FATAL(sh > 0 && sw > 0, "mcast_writer_args: subblock {}x{} has a zero side", sh, sw);
    std::vector<uint32_t> w(ma::W_COUNT, 0xFFFFFFFFu);
    w[ma::W_ADDR] = Caddr;
    w[ma::W_START] = cx * pn + cy * pm * Nt + dW;
    w[ma::W_STRIDE_W] = 1u;
    w[ma::W_STRIDE_H] = Nt;
    w[ma::W_NEXT_SB_W] = sw;
    w[ma::W_NEXT_SB_H] = sh * Nt;
    w[ma::W_SB_W] = sw;
    w[ma::W_SB_H] = sh;
    w[ma::W_SB_TILES] = sw * sh;
    w[ma::W_NSB_W] = pn / sw;
    w[ma::W_NSB_H] = pm / sh;
    w[ma::W_ACC] = acc;
    mcast_args_filled(w, "mcast_writer_args");
    return w;
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
