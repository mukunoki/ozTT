// SPDX-License-Identifier: Apache-2.0
//
// Layout of the runtime arguments of the data-movement kernel of the multicast matmul (the host and the
// kernel read the same single table)
// ---------------------------------------------------------------------------
// mcast_split_dm.cpp reads the arguments with get_arg_val<uint32_t>(i), and on the host
// mcast_reader_args / mcast_writer_args (common/gemm_host.hpp) and McastGemm::set_args /
// build_mcast_gemm (common/mcast_gemm.hpp) fill them with the same indices. The two sides correspond by
// index only, so adding one entry on one side shifts everything after it, and because every entry has the
// same type uint32_t, a different value is read silently (the same situation as described at the top of
// ozaki2/kernels/recon_args.h). The host fills the arrays by indexing them with the names of this table
// (mcast_reader_args / mcast_writer_args in common/gemm_host.hpp, the second-half sender entries in
// common/mcast_gemm.hpp), and an entry that was forgotten keeps its sentinel value and stops the run.
// The relations between the sections (aliases, contiguity, counts) are checked by the static_asserts below.
// To change the layout, edit only this file.
//
// The whole list is the concatenation of three sections in this order.
//   0..32   Reader: the layout of the four readers in matmul_common of tt-metal, without the four trailing
//           batch entries (Mt*Kt, Kt*Nt, the batch count, the reuse of B; the GEMM stage does not handle
//           batches) and without the block heights of A and B (in0_block_h, in1_block_h; the kernel does
//           not read them), which leaves 33 entries (RA_*). The order is a remnant of the layout of the
//           tt-metal readers, but nothing depends on that order (patching also goes through the names of
//           the table)
//   33..39  Second-half sender: 7 entries (X_*) with which the core at the opposite end of the row/column
//           distributes the second half of each block
//   40..    Writer (only the kernel on the B side): the layout of writer_bmm_tile_layout.cpp of tt-metal
//           without the two batch entries (Mt*Nt, the batch count), which leaves 11 entries, plus acc,
//           which the fused writer (WRITER=2) reads: 12 entries (W_*; offsets are relative to W0)
//
// The multicast rectangle is passed as its two corners (hi, lo). hi is the corner with the larger
// coordinates and lo the corner with the smaller ones; in the argument names of the tt-metal readers,
// start corresponds to hi and end to lo (the order used when the sender is on NOC 0).
// The kernel reorders them according to the direction of the NOC (MCAST_ADDR in mcast_split_dm.cpp).
//
// This header does not depend on the device API (only <cstdint>).
#pragma once
#include <cstdint>

namespace mcast_args {

// --- Reader (33 entries; A is 0..6 and B is 7..13, in the same layout) -------
enum Reader : uint32_t {
    RA_A = 0,          // first of the 7 operand entries of A (address, start tile, strides, block shape)
    RA_A_START = 1,    // start tile of A (patched by gemm_run of Ozaki scheme I)
    RA_B = 7,          // first of the 7 operand entries of B
    RA_B_START = 8,    // start tile of B (same)
    RA_NUM_BLOCKS = 14,  // number of K blocks (same)
    RA_MC_A = 15,      // 9 multicast entries of A (hi_x, hi_y, lo_x, lo_y, number of destinations,
                       //                           sender core x, y, the sender's semaphore for waiting
                       //                           until the receivers have room, first-half flag)
    RA_MC_B = 24,      // 9 multicast entries of B (same layout as A)
    RA_COUNT = 33
};
// Positions within the 7 operand entries (offsets from RA_A / RA_B). The kernel reads the block width
// OP_BLOCK_W (for A the K block width, for B the number of column tiles per core). The block height of the
// tt-metal readers is not passed because the kernel does not read it (the number of tiles per block is
// passed as OP_BLOCK_TILES).
enum Operand : uint32_t {
    OP_ADDR = 0, OP_START = 1, OP_STRIDE_W = 2, OP_STRIDE_H = 3, OP_NEXT_BLOCK = 4,
    OP_BLOCK_W = 5, OP_BLOCK_TILES = 6, OP_COUNT = 7
};
// Positions within the 9 multicast entries (offsets from RA_MC_A / RA_MC_B)
enum Mcast : uint32_t {
    MC_HI_X = 0, MC_HI_Y = 1, MC_LO_X = 2, MC_LO_Y = 3, MC_NUM_DESTS = 4,
    MC_SENDER_X = 5, MC_SENDER_Y = 6, MC_SENDER_SEM = 7, MC_FLAG_SEM = 8, MC_COUNT = 9
};

// --- Second-half sender (7 entries) ------------------------------------------
enum Second : uint32_t {
    // Coordinates of the second-half sender core (A: right end of the row, B: bottom end of the column).
    X_S1_X = RA_COUNT,
    X_S1_Y,
    X_HI_X,              // hi corner of its destination rectangle (A: column cc-2, B: row cr-2)
    X_HI_Y,
    X_LO_X,              // lo corner of the same rectangle (A: column 0, B: row 0)
    X_LO_Y,
    X_FLAG_SEM,          // semaphore of the second-half flag
    X_END
};
constexpr uint32_t X_COUNT = X_END - X_S1_X;   // 7

// --- Writer (only the kernel on the B side; offsets from W0) -----------------
constexpr uint32_t W0 = X_END;                 // 40
enum Writer : uint32_t {
    W_ADDR = 0,          // address of the output
    W_START = 1,         // start tile (patched by gemm_run of Ozaki scheme I)
    W_STRIDE_W = 2, W_STRIDE_H = 3,
    W_NEXT_SB_W = 4, W_NEXT_SB_H = 5,
    W_SB_W = 6, W_SB_H = 7, W_SB_TILES = 8,
    W_NSB_W = 9, W_NSB_H = 10,
    // Read only by the fused writer: whether to read the existing S and add to it (the standard writer
    // reads only the 11 entries up to index 10).
    W_ACC = 11,
    W_COUNT = 12         // count including acc (the host passes 12 entries to every writer)
};

// Checks of the relations between the sections (reordering and miscounting are stopped at compile time).
// Each section has its own enumeration type, so the values are converted to uint32_t with u() before they
// are compared (a comparison between different enumeration types is rejected by -Werror=enum-compare).
constexpr uint32_t u(uint32_t v) { return v; }
static_assert(u(RA_A) == 0 && u(RA_B) == u(RA_A) + u(OP_COUNT),
              "the operands of A and B follow from index 0 with 7 entries each");
static_assert(u(RA_A_START) == u(RA_A) + u(OP_START) && u(RA_B_START) == u(RA_B) + u(OP_START),
              "RA_A_START and RA_B_START are aliases of OP_START of the operands");
static_assert(u(RA_NUM_BLOCKS) == u(RA_B) + u(OP_COUNT),
              "the number of blocks comes right after the operand of B");
static_assert(u(RA_MC_A) == u(RA_NUM_BLOCKS) + 1 && u(RA_MC_B) == u(RA_MC_A) + u(MC_COUNT),
              "the multicast entries come in the order A, B with 9 entries each");
static_assert(u(RA_COUNT) == u(RA_MC_B) + u(MC_COUNT),
              "the reader section ends with the multicast entries of B");
static_assert(u(OP_COUNT) == 7 && u(OP_BLOCK_TILES) + 1 == u(OP_COUNT),
              "an operand has 7 entries and ends with the number of tiles per block");
static_assert(u(RA_COUNT) == 33, "the reader section has 33 entries");
static_assert(u(X_S1_X) == u(RA_COUNT) && u(X_S1_Y) == u(X_S1_X) + 1 && u(X_HI_X) == u(X_S1_X) + 2 &&
              u(X_HI_Y) == u(X_S1_X) + 3 && u(X_LO_X) == u(X_S1_X) + 4 && u(X_LO_Y) == u(X_S1_X) + 5 &&
              u(X_FLAG_SEM) == u(X_S1_X) + 6 && X_COUNT == 7,
              "the 7 second-half sender entries come right after the reader section");
static_assert(W0 == u(RA_COUNT) + X_COUNT,
              "the writer arguments come after the reader section and the second-half sender entries");
static_assert(u(W_ADDR) == 0 && u(W_NSB_H) == 10 && u(W_ACC) == u(W_NSB_H) + 1 &&
              u(W_COUNT) == u(W_ACC) + 1, "the writer has 12 entries: the 11 standard ones plus acc");

}  // namespace mcast_args
