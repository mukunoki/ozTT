// SPDX-License-Identifier: Apache-2.0
//
// Argument layout and CB indices of the scaling stage (shared by Ozaki schemes I and II; the host and the
// kernels look at this one table)
// ---------------------------------------------------------------------------
// Arguments of scale_f64_reader.cpp, which finds the maxima (host side: scale_build in common/scale.hpp),
// and of expdelta.cpp, which builds the exponent-delta tiles (host side: expdelta_build in the same header).
// If the two sides were matched by bare numbers, adding one argument on one side would silently shift
// everything after it (the same reason as given at the top of ozaki2/kernels/recon_args.h), so the host
// fills the arguments using the names of this table as indices and the kernels read them by name.
//
// This header does not depend on the device API (only <cstdint>).
#pragma once
#include <cstdint>

// --- Reader of the maxima (scale_f64_reader.cpp) -----------------------------------------
// The same kernel is placed on both data-movement RISC-Vs, and each takes half of the output tiles
// assigned to the core. Each RISC-V receives its own work-area CBs through compile-time arguments
namespace scale_args {

// CB indices (separate for RISCV_1 and RISCV_0)
constexpr uint32_t CB_F64_R1 = 3;    // RISCV_1: two FP64 tiles (the next is read while one is examined)
constexpr uint32_t CB_F64_R0 = 4;    // RISCV_0: the same
constexpr uint32_t CB_OUT_R1 = 16;   // RISCV_1: one BF16 result tile
constexpr uint32_t CB_OUT_R0 = 17;   // RISCV_0: the same

// Compile-time arguments (after the TensorAccessorArgs (input, then output))
enum ReaderCT : uint32_t {
    CT_CEIL = 0,     // BF16 conversion of the maximum: 0 round down, 1 round up, 2 round up and move a
                     // power of two one step up
    CT_MODE,         // 0 = rows of A, 1 = columns of B
    CT_CB_F64,       // work-area CB (two FP64 tiles)
    CT_CB_OUT,       // CB of the BF16 result tile
    CT_COUNT
};
// Runtime arguments (the input tile index is og*Kt+kt in MODE 0 and kt*stride+og in MODE 1; og = start_o + o)
enum ReaderRT : uint32_t {
    R_IN_ADDR = 0,   // start address of the FP64 operand
    R_OUT_ADDR,      // start address of the maxima (BF16 tiles)
    R_NOUT,          // number of output tiles handled by this kernel
    R_KT,            // number of input tiles reduced into one output tile (at least 1)
    R_STRIDE,        // stride of the input tiles in MODE 1
    R_START_O,       // index of the first output tile handled
    R_COUNT
};

}  // namespace scale_args

// --- Exponent delta (expdelta.cpp) ------------------------------------------------------
namespace expdelta_args {

constexpr uint32_t CB_MAX = 0;   // tile of maxima (BF16)
constexpr uint32_t CB_OUT = 1;   // output tile (Int32)

// Compile-time arguments (followed by the TensorAccessorArgs (maxima, output, flags, in this order))
enum CT : uint32_t {
    CT_ROWWISE = 0,  // 1 for the row direction (A), 0 for the column direction (B)
    CT_COUNT
};
// Runtime arguments
enum RT : uint32_t {
    R_MAX_ADDR = 0,  // start address of the maxima (BF16 tiles)
    R_OUT_ADDR,      // start address of the exponent-delta output
    R_FLAG_ADDR,     // start address of the out-of-range flags
    R_NUM,           // number of tiles handled by this core
    R_START,         // index of the first tile handled by this core
    R_SHIFT,         // exponent shift of the normalization (an int32 carried as is in a uint32)
    R_SLOT,          // flag slot dedicated to this core
    R_COUNT
};

}  // namespace expdelta_args
