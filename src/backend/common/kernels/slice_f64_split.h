// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Building blocks for splitting an FP64 tile into the four triple-float tiles, half and half, on the two
// data-movement RISC-Vs
// ---------------------------------------------------------------------------
// When a single RISC-V does the splitting, the time of the slicing stage of scheme I is determined by the
// FP64 reader (slice_f64_reader.cpp) splitting the elements one by one into triple-float with integer
// operations (split_f64; about 45 instructions and 72 cycles per element). The RISC-V of the writer of the
// same core (slice_writer.cpp) only writes out the slices and is otherwise idle, so the second half of
// each tile (512 elements) is split by the writer.
//
// Hand-off (CB c_7 is used as a one-page mailbox)
//   Reader: after the tile has been read and the four output CBs have been reserved, writes "the L1
//           addresses of the FP64 tile and the exponent-delta tile, and the write positions of the four
//           output tiles" into the mailbox page and pushes it. It splits the first half, waits until the
//           mailbox becomes free (the writer pops it), and then pushes the four output tiles.
//   Writer: peeks at the mailbox between writes of slices (without waiting); if a job has arrived, splits
//           the second half and pops.
// A CB is used as the mailbox because its hand-off counters (tiles_received / acked) start from 0 at
// every launch of the program. A flag kept in an L1 variable would still hold the value of the previous
// run and be misread.
// The writer must not block waiting for the mailbox: the slicer waits for the writer's pop and the reader
// waits for the writer's second half, so if the writer blocked on either the mailbox or c_16, all three
// would wait for one another and stop.
// The writer therefore only peeks with cb_pages_available_at_front (slice_writer.cpp).
//
// Ordering guarantees (both directions)
//   The data-movement RISC-V of Blackhole has a data cache (L0) for L1 loads. Before reading L1 that was
//   written by the other RISC-V or by the NoC, the cache is discarded with invalidate_l1_cache() (fence)
//   after the notification from the other side (the CB counter) has been seen, and only then is L1 read
//   (the writer does so before reading the job; before the reader reads the FP64 tile and the exponent
//   deltas written by the NoC, noc_async_read_barrier discards the cache).
//   When values written to L1 by this RISC-V are handed to the other side, a fence alone is not enough:
//   a fence only empties the store queue and does not guarantee that the writes have landed in L1, and
//   the update of the CB counter, which lives in a different memory region, can be processed first.
//   Therefore the last address written is read back right before the notification (l1_writes_landed in
//   l1_order.h). Reader -> writer: the 6 words of the job (os, written last, is read back before the
//   push); writer -> reader: the 512 elements of the second half (element 1023 of s is read back before
//   the pop); reader -> slicer: the first half (element 511 of s is read back before the four output
//   tiles are pushed). The fence before the read-back also waits for preceding loads to complete, so by
//   the time the writer pops, its reads of the FP64 tile, the exponent deltas and the job are finished
//   and the reader may overwrite them for the next tile.

#include "api/dataflow/dataflow_api.h"   // tt_l1_ptr, get_write_ptr, cb_push_back, cb_pop_front
#include "f64_tf.h"
#include "l1_order.h"
#include "slice_f64_args.h"   // CB index and page size of the mailbox (shared with the host)

// SLICE_SPLIT_PROBE: diagnostic macro of a hand-off test (not part of this distribution); nothing in the
// library defines it, so it is 0. A nonzero value makes a notification arrive before its data (1: the
// writer returns the mailbox before splitting the second half; 2: the reader notifies before writing the
// job) and breaks the result. Only the mailbox procedures below (mbox_*) look at it.
#ifndef SLICE_SPLIT_PROBE
#define SLICE_SPLIT_PROBE 0
#endif
constexpr uint32_t SLICE_MBOX_CB = slice_f64_args::CB_MBOX;   // mailbox
constexpr uint32_t SLICE_SPLIT_J = 512;        // the reader splits [0, 512), the writer [512, 1024)

// Contents of the mailbox page (same size as the page created by the host, slice_f64_args::MBOX_PAGE)
struct SliceJob {
    uint32_t src, dl, oh, om, ol, os;          // L1 addresses
    uint32_t pad[2];
};
static_assert(sizeof(SliceJob) == slice_f64_args::MBOX_PAGE, "mailbox job size differs from the host's page");

// Splits the elements [j0, j1). The reader without SPLIT (slice_f64_reader.cpp) also calls this with
// [0, 1024) (there is a single procedure).
// The pointers are left volatile (removing volatile and reshaping the loop turns out slower).
inline void slice_split_range(uint32_t src, uint32_t dlp, uint32_t ohp, uint32_t omp,
                              uint32_t olp, uint32_t osp, uint32_t j0, uint32_t j1) {
    volatile tt_l1_ptr uint32_t* s  = (volatile tt_l1_ptr uint32_t*)src;
    volatile tt_l1_ptr int32_t*  dl = (volatile tt_l1_ptr int32_t*)dlp;
    volatile tt_l1_ptr uint32_t* oh = (volatile tt_l1_ptr uint32_t*)ohp;
    volatile tt_l1_ptr uint32_t* om = (volatile tt_l1_ptr uint32_t*)omp;
    volatile tt_l1_ptr uint32_t* ol = (volatile tt_l1_ptr uint32_t*)olp;
    volatile tt_l1_ptr uint32_t* os = (volatile tt_l1_ptr uint32_t*)osp;
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t w0 = s[2 * j];        // little endian: low word first
        const uint32_t w1 = s[2 * j + 1];
        uint32_t hi, braw, craw, sc;
        split_f64(w1, w0, dl[j], hi, braw, craw, sc);
        oh[j] = hi; om[j] = braw; ol[j] = craw; os[j] = sc;
    }
}

// ===== Mailbox procedures (reader and writer; only this part knows about the probes) =====
// In normal operation (SLICE_SPLIT_PROBE=0) the notification and the return happen at their normal
// positions, and the probe parts do nothing.
// Writes the job contents (the addresses of the FP64 tile and the exponent deltas, and the write
// positions of the four output CBs cb_h, cb_m, cb_l, cb_s) into the mailbox page job and makes sure they
// have landed in L1 (a fence alone does not mean they have landed; l1_order.h). The output addresses are
// taken from the CBs right before they are written (the same order of reads and writes as in the reader
// before this became a function, and the same machine code).
inline void mbox_write_job(volatile tt_l1_ptr SliceJob* job, uint32_t src, uint32_t dl, uint32_t cb_h, uint32_t cb_m,
                           uint32_t cb_l, uint32_t cb_s) {
    job->src = src; job->dl = dl;
    job->oh = get_write_ptr(cb_h); job->om = get_write_ptr(cb_m);
    job->ol = get_write_ptr(cb_l); job->os = get_write_ptr(cb_s);
    l1_writes_landed((uint32_t)&job->os);
}
// Reader: whether to fill the mailbox with valid addresses before the first job (probe 2 only; so that
// the writer reads valid addresses even though it is notified before the contents are written)
constexpr bool MBOX_PREFILL = SLICE_SPLIT_PROBE == 2;
// Reader: after reserving the mailbox and before writing the job (probe 2 notifies here, then waits for a
// while and writes; the result becomes wrong)
inline void mbox_reader_before_write() {
    if constexpr (SLICE_SPLIT_PROBE == 2) {
        cb_push_back(SLICE_MBOX_CB, 1);
        for (uint32_t k = 0; k < 4000; ++k) asm volatile("nop");
    }
}
// Reader: notifies after the job has landed in L1 (probe 2 has already notified before writing)
inline void mbox_reader_notify() {
    if constexpr (SLICE_SPLIT_PROBE != 2) cb_push_back(SLICE_MBOX_CB, 1);
}
// Writer: before splitting the second half (probe 1 returns the mailbox here; the result becomes wrong)
inline void mbox_writer_before_split() {
    if constexpr (SLICE_SPLIT_PROBE == 1) cb_pop_front(SLICE_MBOX_CB, 1);
}
// Writer: returns the mailbox after the split second half has landed in L1 (probe 1 has already returned
// it before splitting)
inline void mbox_writer_return() {
    if constexpr (SLICE_SPLIT_PROBE != 1) cb_pop_front(SLICE_MBOX_CB, 1);
}
