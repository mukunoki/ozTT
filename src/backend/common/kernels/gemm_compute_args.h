// SPDX-License-Identifier: Apache-2.0
//
// CB indices of the GEMM stage and the argument layout of the compute kernel (the host and the kernels
// read the same single table)
// ---------------------------------------------------------------------------
// Compile-time and runtime arguments of the shared matmul kernel bmm_int32_spill.cpp (both Ozaki schemes),
// and the indices of the CBs shared by the data-movement kernel mcast_split_dm.cpp and the compute kernel.
// On the host, common/mcast_gemm.hpp (mcast_compute_args, spill_gemm_spec, spill_compute_rt,
// build_mcast_gemm) fills the arguments by the names of this table. The table of the runtime arguments of
// the data-movement kernel is mcast_args.h.
// If the two sides were matched by position alone, a reordering on one side could not be detected (the
// data-movement side is guarded by the table in mcast_args.h and its static_asserts).
//
// Compile-time arguments 0-11 have the same order as the standard tt-metal matmul kernel
// (bmm_large_block_zm.cpp). The same mcast_compute_args can be passed to the standard kernel as is, so
// the order must not be changed (static_assert below).
// bmm_int32_spill.cpp does not read argument 7 (CA_NUM_BLOCKS); it receives the number of K blocks as the
// runtime argument CR_NUM_BLOCKS (the length of K in the Ozaki GEMM stage changes with the diagonal, and
// this lets a compiled program be reused by replacing its runtime arguments). Argument 7 is still filled
// in order to share the layout, and a different value of the unread argument 7 results in a separate
// entry of the JIT cache.
//
// This header does not depend on the device API (only <cstdint>).
#pragma once
#include <cstdint>

namespace gemm_args {

// --- CB indices ----------------------------------------------------------------
constexpr uint32_t CB_IN0     = 0;    // input A (double buffered)
constexpr uint32_t CB_IN1     = 1;    // input B (double buffered)
constexpr uint32_t CB_OUT     = 16;   // final output (read by the writer)
// Partial sums across K blocks (L1 accumulation of the packer; may share its region with the output).
constexpr uint32_t CB_PARTIAL = 24;
constexpr uint32_t CB_S       = 2;    // fused (FUSE): existing S_d (Int32; prefetched by the fused writer)
// Fused / pair loop: where the matmul result is saved temporarily, still in FP32.
constexpr uint32_t CB_TMP     = 3;
constexpr uint32_t CB_SL      = 5;    // pair loop (PAIRLOOP): Int32 accumulation plane across pairs

// --- Compile-time arguments of the compute kernel -------------------------------------
enum ComputeCT : uint32_t {
    CA_IN0_BLOCK_W = 0,          // width bw of one K block [tiles]
    CA_IN0_NUM_SUBBLOCKS,        // number of subblocks of A, pm/sh
    CA_IN0_BLOCK_NUM_TILES,      // number of tiles in one block of A
    CA_IN0_SUBBLOCK_NUM_TILES,   // number of tiles in one subblock of A
    CA_IN1_NUM_SUBBLOCKS,        // number of subblocks of B, pn/sw
    CA_IN1_BLOCK_NUM_TILES,      // number of tiles in one block of B
    CA_IN1_PER_CORE_W,           // output width per core, pn
    // Number of K blocks (the standard kernel reads it; bmm_int32_spill.cpp does not).
    CA_NUM_BLOCKS,
    CA_OUT_SUBBLOCK_H,           // height sh of the output subblock
    CA_OUT_SUBBLOCK_W,           // width sw of the output subblock
    CA_OUT_SUBBLOCK_NUM_TILES,   // number of tiles sh*sw of the output subblock
    // Batch count. Always 1 because the data-movement kernel does not handle batches (kept to share the
    // layout with the standard kernel).
    CA_BATCH,
    CA_STD_COUNT,                // number of arguments read by the standard kernel
    // The remaining arguments are read only by bmm_int32_spill.cpp.
    CA_FUSE = CA_STD_COUNT,      // add the output of the final block to the existing S in Int32
    CA_F32,                      // data format for the typecast (Float32)
    CA_I32,                      // data format for the typecast (Int32)
    CA_PAIRLOOP,                 // fold the FP32 accumulator at each pair boundary and add it to CB_SL
    CA_BLOCKMM,                  // use matmul_block (0 selects matmul_tiles)
    CA_COUNT
};
// Arguments 0-11 have the same order as in the standard kernel, which reads them by position
// (get_compile_time_arg_val(0..11) in bmm_large_block_zm.cpp). The enumerators are numbered implicitly,
// so if two lines in the middle were swapped, the kernels of this library and the host would follow by
// name and nobody would notice. The position of every field is pinned here so that the correspondence
// with the standard kernel cannot break (do not change them).
static_assert(CA_IN0_BLOCK_W == 0 && CA_IN0_NUM_SUBBLOCKS == 1 && CA_IN0_BLOCK_NUM_TILES == 2 &&
              CA_IN0_SUBBLOCK_NUM_TILES == 3 && CA_IN1_NUM_SUBBLOCKS == 4 && CA_IN1_BLOCK_NUM_TILES == 5 &&
              CA_IN1_PER_CORE_W == 6 && CA_NUM_BLOCKS == 7 && CA_OUT_SUBBLOCK_H == 8 && CA_OUT_SUBBLOCK_W == 9 &&
              CA_OUT_SUBBLOCK_NUM_TILES == 10 && CA_BATCH == 11 && CA_STD_COUNT == 12,
              "arguments 0-11 have the same order as the standard tt-metal matmul kernel (do not change)");
static_assert(CA_FUSE == 12 && CA_F32 == 13 && CA_I32 == 14 && CA_PAIRLOOP == 15 && CA_BLOCKMM == 16,
              "positions of arguments 12-16, which only bmm_int32_spill.cpp reads");
static_assert(CA_COUNT == 17, "bmm_int32_spill.cpp has 17 compile-time arguments");

// --- Runtime arguments of the compute kernel bmm_int32_spill.cpp ---------------------------
enum ComputeRT : uint32_t {
    CR_NUM_BLOCKS = 0,   // number of K blocks of this program
    // Fused: 0 for the first write to that plane (initialize without adding), 1 afterwards (read only
    // when FUSE is set).
    CR_ACC,
    // Pair loop: number of K blocks per pair (at least 1 and a divisor of the number of K blocks; read
    // only when PAIRLOOP is set).
    CR_BPP,
    CR_COUNT
};
static_assert(CR_NUM_BLOCKS == 0 && CR_ACC == 1 && CR_BPP == 2 && CR_COUNT == 3,
              "positions of the runtime arguments of bmm_int32_spill.cpp");

}  // namespace gemm_args
