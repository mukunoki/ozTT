// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: data-movement kernel that writes the slices in the "diagonal-concatenated" layout
// ---------------------------------------------------------------------------
// Drains c_16 (NS slice tiles for each input tile; NS is the number of slices nslice = L+1) and
// redistributes the tiles into the concatenated buffer for the GEMM. With this layout, when one GEMM is
// later issued for each diagonal d, the K direction can be read contiguously (a device that reduces the
// intermediate reads from O(p^2) to O(L)).
//
// The layout is "K block first". K is divided into C blocks of length Kbt (in tiles), and for each block
// one set of concatenated slices [slice0|...|slice_{NS-1}] is laid out:
//
//   [block 0: slice0|slice1|...|slice_{NS-1}][block 1: ...] ...
//
// In this way the GEMM of block b and diagonal d can read a contiguous K range, so the contiguity of the
// reads is kept even with K blocking (the exactness condition then applies to the block length instead
// of K).
// When Kbt == Kto (C=1), the formula coincides with the plain slice concatenation
// [slice0|...|slice_{NS-1}] that is not divided into blocks.
//
// Destination for input tile g and slice s (0..NS-1):
//   mode 0 (A, M x Kto): Acat is tiled as M x (NS*Kto). The K position kt = g%Kto of tile g corresponds
//     to the position kk = kt%Kbt within block b = kt/Kbt.
//     dst = (g/Kto)*(NS*Kto) + b*(NS*Kbt) + s*Kbt + kk
//   mode 1 (B, Kto x N): Bcat is tiled as (NS*Kto) x N, and within each block the r-th position in the
//     K direction holds slice (NS-1-r) (reverse order).
//     row = b*(NS*Kbt) + (NS-1-s)*Kbt + kk,  dst = row*Nt + (g%Nt)
//   The B side is laid out in reverse order because matching the s-th block of A with the (NS-1-s)-th
//   block of B makes i+j constant, i.e. gives the same diagonal.
//
// Runtime arguments (order: SliceWriterRT in args.h): dst_addr, num_in_tiles, start_id, NS, s1, s2, mode, Kbt
//   A: s1 = Kto (in tiles)       B: s1 = Kto, s2 = Nt (in tiles)
//   Kbt = number of K tiles in one block (must divide Kto)

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/slice_f64_split.h"
#include "../../common/kernels/l1_order.h"
#include "args.h"
using namespace oz1_args;

// Compile-time argument (after the TensorAccessor): SPLIT. If 1, the job that the reader
// (slice_f64_reader.cpp) puts in the mailbox c_7 (splitting the second half, 512 elements, of an input tile
// into triple-float) is taken over in between the writes (slice_f64_split.h). The kernel does not block
// waiting on the mailbox or on c_16: if it blocked on either one, the reader (waiting for the second half),
// the slicer (waiting for space in c_16) and this kernel would wait for one another and stop.

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(SW_DST_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(SW_NUM);    // number of input tiles
    uint32_t start_id  = get_arg_val<uint32_t>(SW_START);
    uint32_t nslice    = get_arg_val<uint32_t>(SW_NSLICE);   // number of slices (L+1)
    uint32_t s1        = get_arg_val<uint32_t>(SW_S1);     // Kto
    uint32_t s2        = get_arg_val<uint32_t>(SW_S2);     // Nt for B (unused for A)
    uint32_t mode      = get_arg_val<uint32_t>(SW_MODE);
    uint32_t Kbt       = get_arg_val<uint32_t>(SW_KBT);    // number of K tiles in one block

    constexpr uint32_t cb_o = CB_SLICE_OUT;
    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(d_args, dst_addr);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + SW_CT_SPLIT);

    uint32_t end_id = start_id + num_tiles;
    uint32_t jobs = 0;                               // second halves taken over (one per input tile)
    for (uint32_t g = start_id; g < end_id; ++g) {
        for (uint32_t s = 0; s < nslice; ++s) {
            if constexpr (SPLIT) {
                // Until one slice tile arrives, wait while working off the jobs in the mailbox
                while (!cb_pages_available_at_front(cb_o, 1)) {
                    if (jobs < num_tiles && cb_pages_available_at_front(SLICE_MBOX_CB, 1)) {
                        // Before reading the job written by the reader and the FP64 read by the NoC
                        invalidate_l1_cache();
                        volatile tt_l1_ptr SliceJob* job =
                            (volatile tt_l1_ptr SliceJob*)get_read_ptr(SLICE_MBOX_CB);
                        const uint32_t src = job->src, dl = job->dl, oh = job->oh, om = job->om,
                                       ol = job->ol, os = job->os;
                        const uint32_t os_last = os + 4u * 1023u;   // last address written (element 1023 of s)
                        // No-op in normal operation (probe 1 returns the mailbox here; slice_f64_split.h)
                        mbox_writer_before_split();
                        slice_split_range(src, dl, oh, om, ol, os, SLICE_SPLIT_J, 1024);
                        // Return the mailbox only after the split values have landed in L1 (a fence alone
                        // does not mean that they have landed; l1_order.h). The earlier fence also waits
                        // for the reads of the FP64, the exponent deltas and the job to complete, so the
                        // reader may overwrite them after the return.
                        l1_writes_landed(os_last);
                        mbox_writer_return();
                        ++jobs;
                    }
                }
            } else {
                cb_wait_front(cb_o, 1);
            }
            uint32_t dst;
            if (mode == 0) {          // A: within block b, slice s goes to the s-th position
                uint32_t mt = g / s1, kt = g % s1;
                uint32_t b = kt / Kbt, kk = kt - b * Kbt;
                dst = mt * nslice * s1 + b * nslice * Kbt + s * Kbt + kk;
            } else {                  // B: within block b, slice (nslice-1-r) is placed at the r-th position
                uint32_t kt = g / s2, nt = g % s2;
                uint32_t b = kt / Kbt, kk = kt - b * Kbt;
                uint32_t row = b * nslice * Kbt + (nslice - 1 - s) * Kbt + kk;
                dst = row * s2 + nt;
            }
            noc_async_write_tile(dst, d, get_read_ptr(cb_o));
            noc_async_write_barrier();
            cb_pop_front(cb_o, 1);
        }
    }
}
