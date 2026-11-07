// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernel of the multicast matmul (one RISC per operand; each operand is distributed from
// the two cores at both ends)
// ---------------------------------------------------------------------------
// In the four multicast readers in matmul_common of tt-metal
// (reader_bmm_tile_layout_in0_{sender,receiver}_in1_{sender,receiver}.cpp), a single RISC reads a block
// of A and finishes distributing it before it reads a block of B and distributes that.
// With that scheme the sender core spends a large part of every block period waiting for DRAM reads to
// complete, and the period is lengthened because the reads of A and B are serialized.
//
// Two kernels are built from this single source:
//   OPERAND=0 : handles only A (c_0). Placed on RISCV_1 and uses NOC 1.
//   OPERAND=1 : handles only B (c_1) and writes the output after the K loop (WRITER).
//               Placed on RISCV_0 and uses NOC 0.
// This way the DRAM reads and the multicasts of A and B proceed in parallel on two RISCs.
// The output is written only after all blocks, so even though the writer shares the RISC of B, it does
// not overlap in time with the supply of B (batches are not handled: the batch arguments and loops of the
// tt-metal readers and writer have been removed).
//
// The tiles of one block are divided into a first half and a second half, and the two cores at both ends
// of the row (column) each read one half from DRAM and distribute it to all cores other than themselves.
// The amount of DRAM reads and multicast per sender core is halved, and the two cores proceed in parallel.
// A core at an end is a sender and at the same time a receiver of the half that the opposite end
// distributes. A receiver waits for two flags (first half and second half).
//   ROLE=1 : sender of the first half (A: left end of the row, B: top end of the column)
//   ROLE=3 : sender of the second half (A: right end of the row, B: bottom end of the column)
//   ROLE=2 : receive-only core (the cores in between; present only when the row/column has 3 or more cores)
// When a block is a single tile, the second half has 0 tiles. In that case the host does not place a
// second-half sender (the opposite end is also ROLE=2) and the first-half sender distributes everything.
// The handshake of the second-half flag is not performed either (half2).
//   WRITER=0 : does not write (A side)
//   WRITER=1 : standard writer (same write order as writer_bmm_tile_layout.cpp of tt-metal; its arguments
//              are the 11 that remain after removing the 2 batch entries. W_* in mcast_args.h)
//   WRITER=2 : fused writer. Before writing, it prefetches the existing S at the same position into c_2
//              (in the deferred reconstruction of Ozaki scheme I, this is the addend to which FUSE of the
//              compute kernel bmm_int32_spill.cpp adds in Int32).
//   NODRAM, NOMCAST, NOWRITE : diagnostic probes (they corrupt the result; see the defines below). The
//              default 0 is the production configuration; the library always passes 0
//              (mcast_make_kernels in common/mcast_gemm.hpp).
//
// The runtime arguments are the 33 reader entries (0..32: the layout of the tt-metal readers without the
// 4 batch entries and the 2 block heights; mcast_reader_args), followed by the 7 entries for the
// second-half sender (33..39: the coordinates of the second-half sender core, its destination rectangle,
// and the semaphore of the second-half flag) and, when WRITER is set, by the writer arguments from index
// 40 (mcast_writer_args). The indices are read through the table in mcast_args.h, which is shared with the
// host (RA_*, X_*, W_*).
// The reader layout was copied from the tt-metal readers, but nothing depends on that order: patching
// (gemm_run of Ozaki scheme I and so on) goes through the names of the table in mcast_args.h (RA_A_START,
// RA_B_START, RA_NUM_BLOCKS).
// The compile-time arguments are the TensorAccessorArgs of A, B and C concatenated in this order.
//
// The host passes the multicast rectangle as its two corners (hi, lo) (hi is the side with the larger
// coordinates; it is called start in the argument names of the tt-metal readers). NOC 0 runs in the
// direction of increasing coordinates, so the destination is built in the order (lo, hi); NOC 1 runs the
// opposite way, so it is built in the order (hi, lo) (MCAST_ADDR).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "mcast_args.h"
// CB indices of the GEMM stage (shared with the compute kernel and with common/mcast_gemm.hpp on the host)
#include "gemm_compute_args.h"
#include "l1_order.h"   // l1_writes_landed (after the stores to the flags and semaphores below)

#ifndef OPERAND
#error "OPERAND (0=A, 1=B) must be defined"
#endif
#ifndef ROLE
#error "ROLE (1=first-half sender, 2=receiver, 3=second-half sender) must be defined"
#endif
#ifndef WRITER
#define WRITER 0
#endif
// NODRAM=1: diagnostic. The senders read from DRAM only for the first 2 blocks and afterwards distribute
// the stale data left in the CB as is (the result is corrupted). It shows the time of the GEMM stage
// without the DRAM reads.
#ifndef NODRAM
#define NODRAM 0
#endif
// NOMCAST=1: diagnostic. A/B are not supplied (no DRAM reads, multicast or synchronization); only the
// reserve/push of the input CB is performed (the compute kernel runs on whatever values remain in the CB,
// so the result is corrupted). Writing the output (the writes to DRAM in write_output below and the wait
// for their completion, and with the fused writer WRITER=2 also the prefetch of the existing S) remains,
// so what can be measured is the time of "compute pipeline + output transfer", not the time of the
// computation alone (the name means "no multicast of A/B"; it does not mean that the NoC is not used at all).
#ifndef NOMCAST
#define NOMCAST 0
#endif
// NOWRITE=1: diagnostic. The output is not written to DRAM (and with the fused writer the existing S is
// not prefetched either); c_16 is only received and discarded (the output is not written, so the result
// is corrupted). It shows the time of the GEMM stage without writing the output.
#ifndef NOWRITE
#define NOWRITE 0
#endif

// Builds the multicast destination in the order that matches the direction of the NOC from the
// destination rectangle (hi corner, lo corner)
#define MCAST_ADDR(hx, hy, lx, ly, a) \
    (noc_index == 0 ? get_noc_multicast_addr(lx, ly, hx, hy, a) : get_noc_multicast_addr(hx, hy, lx, ly, a))

// The relations between the sections are checked by the static_asserts in mcast_args.h.
namespace ma = mcast_args;

#if WRITER
// Writes the output. The writer arguments start at argument W0.
template <uint32_t FUSED, typename Acc>
static inline void write_output(const Acc& s, uint32_t start_tile) {
    constexpr uint32_t w0 = ma::W0;
    const uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(w0 + ma::W_STRIDE_W);
    const uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(w0 + ma::W_STRIDE_H);
    const uint32_t next_sb_w = get_arg_val<uint32_t>(w0 + ma::W_NEXT_SB_W);
    const uint32_t next_sb_h = get_arg_val<uint32_t>(w0 + ma::W_NEXT_SB_H);
    const uint32_t out_subblock_w = get_arg_val<uint32_t>(w0 + ma::W_SB_W);
    const uint32_t out_subblock_h = get_arg_val<uint32_t>(w0 + ma::W_SB_H);
    const uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(w0 + ma::W_SB_TILES);
    const uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(w0 + ma::W_NSB_W);
    const uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(w0 + ma::W_NSB_H);
    // Only the fused writer reads acc at W_ACC (the host passes W_COUNT entries (mcast_args.h), which
    // include acc, to every writer, and McastGemm::set_args checks that count. The standard writer does
    // not read W_ACC).
    const uint32_t acc = FUSED ? get_arg_val<uint32_t>(w0 + ma::W_ACC) : 0u;
    constexpr uint32_t cb_out = gemm_args::CB_OUT;
    constexpr uint32_t cb_s = gemm_args::CB_S;   // fused writer: where the existing S is prefetched to
    const uint32_t tsz = get_tile_size(cb_out);

    uint32_t sbh_start = start_tile;
    for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
        uint32_t sbw_start = sbh_start;
        for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
            if (FUSED && acc && !NOWRITE) {
                // Read the existing S first (the compute kernel adds it in Int32 and outputs to c_16)
                cb_reserve_back(cb_s, out_subblock_tile_count);
                uint32_t l1w = get_write_ptr(cb_s);
                uint32_t row = sbw_start;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t id = row;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_read_page(id, s, l1w);
                        l1w += tsz;
                        id += out_tensor_stride_w;
                    }
                    row += out_tensor_stride_h;
                }
                noc_async_read_barrier();
                cb_push_back(cb_s, out_subblock_tile_count);
            }
            // Hand over the CB of S, which the compute kernel waits for, without contents
            if (FUSED && acc && NOWRITE) {
                cb_reserve_back(cb_s, out_subblock_tile_count);
                cb_push_back(cb_s, out_subblock_tile_count);
            }
            cb_wait_front(cb_out, out_subblock_tile_count);
            if (!NOWRITE) {
                uint32_t l1r = get_read_ptr(cb_out);
                uint32_t row = sbw_start;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t id = row;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_write_page(id, s, l1r);
                        l1r += tsz;
                        id += out_tensor_stride_w;
                    }
                    row += out_tensor_stride_h;
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, out_subblock_tile_count);
            sbw_start += next_sb_w;
        }
        sbh_start += next_sb_h;
    }
}
#endif

void kernel_main() {
    // A uses arguments 0..6 and B uses 7..13, in the same layout (address, start tile, strides, block shape)
    constexpr uint32_t T = OPERAND == 0 ? ma::RA_A : ma::RA_B;
    const uint32_t tensor_addr = get_arg_val<uint32_t>(T + ma::OP_ADDR);
    const uint32_t start_tile = get_arg_val<uint32_t>(T + ma::OP_START);
    const uint32_t stride_w = get_arg_val<uint32_t>(T + ma::OP_STRIDE_W);
    const uint32_t stride_h = get_arg_val<uint32_t>(T + ma::OP_STRIDE_H);
    const uint32_t next_block_stride = get_arg_val<uint32_t>(T + ma::OP_NEXT_BLOCK);
    const uint32_t block_w = get_arg_val<uint32_t>(T + ma::OP_BLOCK_W);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(T + ma::OP_BLOCK_TILES);
    const uint32_t num_blocks = get_arg_val<uint32_t>(ma::RA_NUM_BLOCKS);
    // Multicast arguments: A uses 15..23 and B uses 24..32. The rectangle is the destination of the first
    // half (all cores other than ROLE=1)
    constexpr uint32_t Mc = OPERAND == 0 ? ma::RA_MC_A : ma::RA_MC_B;
    const uint32_t mc_hx = get_arg_val<uint32_t>(Mc + ma::MC_HI_X);
    const uint32_t mc_hy = get_arg_val<uint32_t>(Mc + ma::MC_HI_Y);
    const uint32_t mc_lx = get_arg_val<uint32_t>(Mc + ma::MC_LO_X);
    const uint32_t mc_ly = get_arg_val<uint32_t>(Mc + ma::MC_LO_Y);
    const uint32_t mc_num_dests = get_arg_val<uint32_t>(Mc + ma::MC_NUM_DESTS);
    const uint32_t sender_x = get_arg_val<uint32_t>(Mc + ma::MC_SENDER_X);
    const uint32_t sender_y = get_arg_val<uint32_t>(Mc + ma::MC_SENDER_Y);
    const uint32_t sender_sem_addr = get_semaphore(get_arg_val<uint32_t>(Mc + ma::MC_SENDER_SEM));
    const uint32_t receiver_sem_addr = get_semaphore(get_arg_val<uint32_t>(Mc + ma::MC_FLAG_SEM));
    // The second-half sender core (the opposite end of the row/column), its destination rectangle
    // (hi corner, lo corner), and the second-half flag
    const uint32_t s1_x = get_arg_val<uint32_t>(ma::X_S1_X), s1_y = get_arg_val<uint32_t>(ma::X_S1_Y);
    const uint32_t r1_hx = get_arg_val<uint32_t>(ma::X_HI_X), r1_hy = get_arg_val<uint32_t>(ma::X_HI_Y);
    const uint32_t r1_lx = get_arg_val<uint32_t>(ma::X_LO_X), r1_ly = get_arg_val<uint32_t>(ma::X_LO_Y);
    const uint32_t flag1_addr = get_semaphore(get_arg_val<uint32_t>(ma::X_FLAG_SEM));

    constexpr uint32_t cb = OPERAND == 0 ? gemm_args::CB_IN0 : gemm_args::CB_IN1;
    constexpr bool IS_S0 = ROLE == 1, IS_S1 = ROLE == 3;
    // The first-half flag is receiver_sem and the second-half flag is flag1. A sender core uses sender_sem
    // for its own wait until the receivers have room
    volatile tt_l1_ptr uint32_t* rsem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    volatile tt_l1_ptr uint32_t* flag1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag1_addr);
    volatile tt_l1_ptr uint32_t* ssem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
#if OPERAND == 0
    const auto src = TensorAccessor(a_args, tensor_addr);
#else
    const auto src = TensorAccessor(b_args, tensor_addr);
#endif
#if WRITER
    const uint32_t out_addr = get_arg_val<uint32_t>(ma::W0 + ma::W_ADDR);
    const uint32_t out_start = get_arg_val<uint32_t>(ma::W0 + ma::W_START);
    const auto dst = TensorAccessor(c_args, out_addr);
#endif

    const uint32_t tile_bytes = get_tile_size(cb);
    const uint32_t n0 = (block_num_tiles + 1) / 2;   // number of tiles in the first half (handled by ROLE=1)
    const uint64_t s0_ready_noc = get_noc_addr(sender_x, sender_y, sender_sem_addr);
    const uint64_t s1_ready_noc = get_noc_addr(s1_x, s1_y, sender_sem_addr);
    // Source of the flag that a sender core distributes (VALID is placed in its own L1). The multicast
    // below reads this location through the NoC, but the order between a store to L1 and a NoC command
    // issued after it is not guaranteed (l1_order.h), so the kernel proceeds only after the store has
    // landed in L1
    if constexpr (IS_S0) { *rsem = VALID; l1_writes_landed(receiver_sem_addr); }
    if constexpr (IS_S1) { *flag1 = VALID; l1_writes_landed(flag1_addr); }
    // What a sender core is in charge of: the destination rectangle, the flag to distribute, and the
    // tile range [t_beg, t_end)
    const uint32_t my_hx = IS_S1 ? r1_hx : mc_hx, my_hy = IS_S1 ? r1_hy : mc_hy;
    const uint32_t my_lx = IS_S1 ? r1_lx : mc_lx, my_ly = IS_S1 ? r1_ly : mc_ly;
    const uint32_t my_flag = IS_S1 ? flag1_addr : receiver_sem_addr;
    const uint32_t t_beg = IS_S1 ? n0 : 0u, t_end = IS_S1 ? block_num_tiles : n0;
    // Whether a second half exists (it does not for a one-tile block; then there is no second-half sender
    // core and the flag handshake is skipped as well. This is the same condition by which build_mcast_gemm
    // on the host decides whether to place a second-half sender core)
    const bool half2 = block_num_tiles > n0;

    uint32_t cur = start_tile;
    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb, block_num_tiles);
#if !NOMCAST
        const uint32_t slot = get_write_ptr(cb);
        // Clear the flag on the receiving side, then tell the sender core that writes to this core that
        // there is room. INVALID is a store to L1 and the notification is a NoC command, and their order
        // is not guaranteed (l1_order.h). If the notification left first and INVALID landed after the
        // VALID of the sender core, the wait below would never end, so the notification is sent only
        // after INVALID has landed in L1 (a few tens of cycles per block)
        if constexpr (!IS_S0) {
            noc_semaphore_set(rsem, INVALID); l1_writes_landed(receiver_sem_addr); noc_semaphore_inc(s0_ready_noc, 1); }
        if (!IS_S1 && half2) {
            noc_semaphore_set(flag1, INVALID); l1_writes_landed(flag1_addr); noc_semaphore_inc(s1_ready_noc, 1); }
        if (IS_S0 || (IS_S1 && half2)) {
            // Read the tiles this core is in charge of (the first or the second half). Tile t is at
            // (h, w) = (t / block_w, t % block_w) within the block and is also placed t-th in the CB.
            uint32_t h = t_beg / block_w, w = t_beg % block_w;
            uint32_t l1 = slot + t_beg * tile_bytes;
            for (uint32_t t = t_beg; t < t_end; t++) {
                if (!NODRAM || block < 2) noc_async_read_page(cur + h * stride_h + w * stride_w, src, l1);
                l1 += tile_bytes;
                if (++w == block_w) { w = 0; h++; }
            }
            noc_async_read_barrier();
            // Wait until all cores other than this one have reported room, then distribute
            noc_semaphore_wait(ssem, mc_num_dests);
            // Distribute only after the store that resets the semaphore to 0 has landed in L1 (the order
            // is not guaranteed here either; if the notification for the next block arrived from a
            // destination before the store landed, the store would erase it with 0. l1_order.h)
            noc_semaphore_set(ssem, 0);
            l1_writes_landed(sender_sem_addr);
            const uint32_t a = slot + t_beg * tile_bytes;
            noc_async_write_multicast(a, MCAST_ADDR(my_hx, my_hy, my_lx, my_ly, a),
                                      (t_end - t_beg) * tile_bytes, mc_num_dests);
#ifdef ARCH_BLACKHOLE
            // On Blackhole the commands go into separate FIFOs and their order is not guaranteed, so the
            // data is sent out before the flag (same as the tt-metal readers)
            noc_async_writes_flushed();
#endif
            noc_semaphore_set_multicast(my_flag, MCAST_ADDR(my_hx, my_hy, my_lx, my_ly, my_flag), mc_num_dests);
        }
        cur += next_block_stride;
        if constexpr (!IS_S0) noc_semaphore_wait(rsem, VALID);
        if (!IS_S1 && half2) noc_semaphore_wait(flag1, VALID);
#endif
        cb_push_back(cb, block_num_tiles);
    }
#if WRITER
    write_output<WRITER == 2 ? 1u : 0u>(dst, out_start);
#endif
#if ROLE == 1 || ROLE == 3
    // Wait until the multicasts that were sent have left completely (so that L1 is not reused after the
    // kernel ends)
    noc_async_write_barrier();
#endif
}
