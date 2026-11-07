// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Before a data-movement RISC-V tells another party about values it wrote to L1, wait until "the writes
// have landed in L1"
// ---------------------------------------------------------------------------
// On the baby RISC-V of Blackhole a store is placed in the store queue, and the write request flows to
// L1 or to another memory region later. A fence waits only until the store queue is empty (the requests
// have been issued); it does not guarantee that the requests have been processed (written to L1). The
// counters of a CB (tiles_received / acked are stream registers) and the NoC command registers are
// memory regions separate from L1, so with
//     write to L1 -> fence -> cb_push_back (or issuing noc_async_write)
// the update of the counter (the NoC read) can be processed before the write to L1
// (tt-isa-documentation BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md, sections "Fences",
// "Enforcing stronger ordering" and "Cross-core signalling using mailboxes").
// cb_push_back and noc_async_write of tt-metal do not give this guarantee either (cb_push_back in
// dataflow_api.h only adds to the counter register and does nothing about the ordering with respect to
// values the RISC-V wrote to L1).
//
// The reliable method the documentation gives is "read back the written L1 on the same RISC-V and issue
// the next request only after the result of that read has been obtained". l1_writes_landed(a) does this
// with the following three instructions:
//   fence   empties the store queue (all write requests to L1 up to this point are issued). It also
//           discards the L0 data cache, so the next lw is sure to go to L1
//   lw      reads a (an L1 address; the caller passes the address it wrote last). The requests of one
//           RISC-V to L1 go through the same L1 access port in order, and the port processes the
//           requests in order, so this read is processed after all the writes issued before it have
//           been processed
//   fence   keeps the following instructions from advancing to EX1 until the result of this read is
//           settled (a fence does not enter the LSU until the values of the preceding loads are
//           settled, and the following instructions cannot enter EX1 until the fence retires)
// On exit, all values this RISC-V has written to L1 so far are in L1, and the counter updates and NoC
// requests that follow are processed after that. The first fence also waits for the completion of the
// preceding loads, so all values this RISC-V has read from L1 so far are settled as well (the other
// party may overwrite that region).
// The reading side needs a different procedure: after seeing the notification of the other party
// (counter, semaphore), it discards L0 with invalidate_l1_cache() (a fence) and then reads L1.
//
// One call costs a few tens of cycles, so at a frequency of once per tile (1024 elements are split or
// assembled over several thousand cycles) it does not show up in the time.
#include <cstdint>

inline __attribute__((always_inline)) void l1_writes_landed(uint32_t l1_addr) {
    uint32_t v;
    asm volatile("fence\n\t"
                 "lw %0, 0(%1)\n\t"
                 "fence"
                 : "=r"(v) : "r"(l1_addr) : "memory");
    (void)v;
}
