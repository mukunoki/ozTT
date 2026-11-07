// SPDX-License-Identifier: Apache-2.0
//
// CB indices and argument layout of the FP64 reader (slice_f64_reader.cpp) and of the slicer input (shared
// by Ozaki schemes I and II; the host and the kernels look at this one table)
// ---------------------------------------------------------------------------
// The reader splits an FP64 tile into the four tiles of the triple-float form and pushes them to CBs, and
// the slicer (slice_tf_*.cpp in Ozaki scheme I, digits.h in Ozaki scheme II; the input side and the
// carrying of the residual are in slice_tf_core.h) reads them. If the hosts of the two schemes
// (slice_build / slice_run in ozaki1/slice.hpp, build_slice in ozaki2/stages.hpp) and the kernels held the
// CB indices and the argument layout as bare numbers, moving only one side would make the other silently
// read a different CB or a different value (the same reason as given at the top of
// ozaki2/kernels/recon_args.h). Therefore the table is placed here and both sides use it by name.
// The functions with which the host fills the reader arguments are in common/slice_f64_host.hpp. The order
// in which the CBs are created determines the L1 layout and differs between the two schemes, so the
// creation itself is not shared (only the indices are taken from here).
//
// This header does not depend on the device API (only <cstdint>).
#pragma once
#include <cstdint>

namespace slice_f64_args {

// --- CB indices ----------------------------------------------------------------
// The four tiles written by the reader and read by the slicer (the middle and low ones are raw bit
// patterns in magic-number form; tf_load_f64 in slice_tf_core.h)
constexpr uint32_t CB_HI    = 0;    // high limb hi
constexpr uint32_t CB_CRAW  = 1;    // raw bit pattern craw of the low limb (= 2^23 + c)
// Scale of the low limb, s = +-2^(elsb+7) (elsb is the exponent of the low limb in split_f64 of f64_tf.h;
// see slice_tf_core.h)
constexpr uint32_t CB_SCALE = 2;
constexpr uint32_t CB_BRAW  = 4;    // raw bit pattern braw of the middle limb (= 2^23 + 2^22 + b)
// Work areas of the reader (not pushed/popped; the write position is fixed and reused)
constexpr uint32_t CB_F64   = 5;    // one FP64 tile (OZ_F64_TILE_BYTES in f64_tf.h)
constexpr uint32_t CB_DLT   = 6;    // exponent-delta tile (Int32)
// Mailbox (only with SPLIT=1; the reader passes the job of splitting the second half of a tile to the
// writer; slice_f64_split.h)
constexpr uint32_t CB_MBOX  = 7;
// Size of one page [B]. Equal to SliceJob in slice_f64_split.h (static_assert in that file)
constexpr uint32_t MBOX_PAGE = 32;
// CBs in which the slicer carries the three terms of the residual (the reason for the 2 + 1 split is in
// slice_tf_core.h)
constexpr uint32_t CB_RES01 = 24;   // residual limbs 0 and 1
constexpr uint32_t CB_RES2  = 25;   // residual limb 2

// --- Compile-time arguments of the reader (after the TensorAccessorArgs (FP64, then exponent delta)) ---
enum ReaderCT : uint32_t {
    // 1 selects the soft-float version (diagnostic selector; not used by the library, always 0)
    CT_SF = 0,
    // 1 lets the writer of the same core split the second half, 512 elements (slice_f64_split.h; Ozaki
    // scheme I only)
    CT_SPLIT,
    CT_COUNT
};

// --- Runtime arguments of the reader -------------------------------------------------
enum ReaderRT : uint32_t {
    R_F64_ADDR = 0,  // start address of the FP64 operand
    R_DLT_ADDR,      // start address of the exponent deltas (per-row / per-column broadcast tiles)
    R_NUM,           // number of input tiles handled by this core
    R_START,         // index of the first input tile handled by this core
    // Exponent-delta lookup: mode 0 (A) divides the tile index by this value, mode 1 (B) takes the
    // remainder of that division
    R_DLT_STRIDE,
    R_MODE,          // 0 = A (per row), 1 = B (per column)
    R_COUNT
};

}  // namespace slice_f64_args
