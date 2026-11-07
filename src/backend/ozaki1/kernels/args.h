// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: argument layouts and CB indices of the kernels of the slicing stage and the
// reconstruction stage (the host and the kernels look at one and the same table)
// ---------------------------------------------------------------------------
// The host (slice_build and slice_set_args in slice.hpp, recon_build_prog and recon_set_out_off in recon.hpp)
// passes sequences of uint32_t to SetRuntimeArgs and compile_args, and the kernels
// receive them with get_arg_val<uint32_t>(i) and get_compile_time_arg_val(i). If the two sides are matched by
// number alone, adding one entry on one side silently shifts everything after it (the same reason as at the
// top of ozaki2/kernels/recon_args.h). The table is therefore placed here: the host fills the arguments by
// indexing with the names of the table, and the kernels read them by name. The table of the slicer input
// (the FP64 reader) is in common/kernels/slice_f64_args.h, which both schemes share, and the table of the
// scaling stage is in common/kernels/scale_args.h.
//
// Mind the order of the limbs of the triple: the CBs that the compute kernel of the reconstruction produces
// and the writer reads, and the arguments of the writer (addresses of the limb planes in DRAM,
// TensorAccessorArgs), are in the order hi, lo, mid (the middle limb last). This differs from the order in
// which the reader reads the accumulation up to the previous block (hi, mid, lo). They are easy to mix up,
// so all of them are referred to by the names defined here.
//
// This header does not depend on the device API (only <cstdint>).
#pragma once
#include <cstdint>

namespace oz1_args {

// ===== Slicing stage =====
// --- Slicer (slice_tf_num.cpp, slice_tf_sm8.cpp; the shared body is slice_tf_body.h) ------------------------
constexpr uint32_t CB_SLICE_OUT = 16;   // slice output (read by the writer slice_writer.cpp)
enum SlicerCT : uint32_t {
    // Number of slices nslice produced from one input tile (the number of slices p = L+1; RunCfg::nslice() in plan.hpp)
    SC_NSLICE = 0,
    SC_RADIX,        // radix (weight of a digit) as the bit pattern of a float
    SC_SF_IN,        // 1: the reader is the soft-float version and passes the triple as is (for measurement)
    SC_DMAX,         // upper bound of a digit as the bit pattern of a float
    SC_NDMAX,        // its negative
    // 1: the slices are output as binary16 bit patterns (UInt16) (FP16 path; no effect on the
    // sign-magnitude output of slice_tf_sm8.cpp)
    SC_OUT_HALF,
    SC_COUNT
};
enum SlicerRT : uint32_t { SR_NTILES = 0, SR_COUNT };

// --- Slice writer (slice_writer.cpp) ----------------------------------------------------
// Compile-time arguments (after the TensorAccessorArgs of the output)
enum SliceWriterCT : uint32_t {
    // 1: takes over the second-half job that the reader puts in the mailbox (common/kernels/slice_f64_split.h)
    SW_CT_SPLIT = 0,
    SW_CT_COUNT
};
// Runtime arguments (the formula of the destination is at the top of slice_writer.cpp)
enum SliceWriterRT : uint32_t {
    SW_DST_ADDR = 0,   // start address of the concatenated slice buffer
    SW_NUM,            // number of input tiles handled by this core
    SW_START,          // index of the first of those tiles
    // Number of slices nslice (must equal SC_NSLICE of the compute kernel; slice_set_args checks it)
    SW_NSLICE,
    SW_S1,             // number of tiles in the K direction, Kto (for both A and B)
    SW_S2,             // number of tiles in the column direction of B, Nt (unused for A)
    SW_MODE,           // 0 = A, 1 = B
    SW_KBT,            // number of K tiles in one K block (must divide Kto)
    SW_COUNT
};

// ===== Reconstruction stage =====
// --- CB indices ----------------------------------------------------------------------
constexpr uint32_t CB_RECON_IN = 0;   // partial sums P_d of the diagonals (ndiag tiles at a time)
// Two mask tiles that split an Int32 into a pair (Int32 input only; the reader builds them in L1)
constexpr uint32_t CB_MASK     = 1;
constexpr uint32_t CB_ACC      = 2;   // triple accumulated up to the previous block (ACC=1; order hi, mid, lo)
// Output triple of the compute kernel (read by the writer). The indices are in the order hi, lo, mid
constexpr uint32_t CB_LIMB_HI  = 16;
constexpr uint32_t CB_LIMB_LO  = 17;
constexpr uint32_t CB_LIMB_MID = 18;
constexpr uint32_t CB_F64      = 19;  // writer work area: one FP64 output tile (OZ_F64_TILE_BYTES in f64_tf.h)
constexpr uint32_t CB_DLT_A    = 20;  // writer work area: exponent deltas of the rows
constexpr uint32_t CB_DLT_B    = 21;  // writer work area: exponent deltas of the columns

// --- Reader (recon_reader_f32in.cpp, recon_reader_i32in.cpp) --------------------------------
// Compile-time arguments. The f32in version has only the TensorAccessorArgs (P, then the accumulation in the
// order hi, mid, lo). The i32in version has RI_CT_SIGNMAG first, followed by the TensorAccessorArgs of P (it
// does not read the accumulation)
enum ReaderI32CT : uint32_t {
    // 1: sign-magnitude (P_d of the INT8 path); 0: two's complement (the accumulation S_d of the deferred
    // reconstruction)
    RI_CT_SIGNMAG = 0,
    RI_CT_COUNT
};
// Runtime arguments. The i32in version uses the 5 words up to RR_SLOT_STRIDE (it has no accumulation input)
enum ReaderRT : uint32_t {
    RR_SRC_ADDR = 0,     // start address of the partial sums Pbig
    RR_NUM,              // number of output tiles handled by this core
    RR_START,            // index of the first of those tiles
    RR_NDIAG,            // number of diagonals (= the number of slices p)
    // Spacing of the planes of the diagonals [tiles] (default Mt*Nt; with row bands, the allocated width of
    // a band)
    RR_SLOT_STRIDE,
    RR_I32_COUNT,        // count for the i32in version
    // Start addresses of the accumulated triple (f32in version only; 0 means no accumulation)
    RR_ACC_HI = RR_I32_COUNT,
    RR_ACC_MID,
    RR_ACC_LO,
    RR_COUNT             // count for the f32in version
};

// --- Compute kernel (recon_tf_f32in.cpp, recon_tf_i32in.cpp) ---------------------------------
enum ReconCT : uint32_t {
    RC_NDIAG = 0,    // number of diagonals (= the number of slices p)
    RC_IN_FMT,       // format value of the partial products (Int32 for the i32in version)
    RC_F32,          // format value of Float32
    RC_RADIX,        // radix as the bit pattern of a float
    // 1: reads the accumulation up to the previous block from CB_ACC and adds it (only the f32in version
    // reads it)
    RC_ACC,
    RC_SIGNMAG,      // 1: sign-magnitude (only the i32in version reads it)
    // Number of leading diagonals that are read with a single typecast (only the i32in version reads it)
    RC_NSINGLE_CAST,
    RC_COUNT
};
enum ReconRT : uint32_t { RCR_NTILES = 0, RCR_COUNT };

// --- Writer (recon_f64_writer.cpp) --------------------------------------------------------
// Compile-time arguments: six TensorAccessorArgs in the order of WriterTA below, followed by WriterCT
enum WriterTA : uint32_t { TA_HI = 0, TA_LO, TA_MID, TA_OUT, TA_DLT_A, TA_DLT_B, TA_COUNT };
enum WriterCT : uint32_t {
    RW_CT_EMIT_LIMBS = 0,   // writes out the three limbs (with K blocking, for every block but the last)
    RW_CT_EMIT_F64,         // assembles FP64 (last block only)
    RW_CT_SF,               // measurement mode that assembles with C double expressions (soft-float)
    RW_CT_PROBE,            // [diagnostic] selector 0--3 (not used by the library; always 0)
    RW_CT_COUNT
};
enum WriterRT : uint32_t {
    RW_HI_ADDR = 0,   // start address of the limb plane hi
    RW_NUM,           // number of output tiles handled by this core
    RW_START,         // index of the first of those tiles (local index within the band)
    RW_LO_ADDR,       // limb plane lo
    RW_MID_ADDR,      // limb plane mid
    RW_OUT_ADDR,      // FP64 output
    RW_DLT_A_ADDR,    // exponent deltas of the rows
    RW_DLT_B_ADDR,    // exponent deltas of the columns
    RW_NT,            // number of tiles in the column direction of the output
    RW_WHI,           // upper 32 bits of the significand of the final scale W
    RW_WLO,           // lower 32 bits of the same
    RW_QW,            // exponent of W (an int32 carried as is in a uint32)
    RW_WDLO,          // low word of wscale (bit pattern of a double) of the soft-float version
    RW_WDHI,          // high word of the same
    // Tile offset on the output side of a row band (0 in the configuration that is not cut into bands;
    // recon_set_out_off replaces it)
    RW_OUT_OFF,
    RW_COUNT
};

}  // namespace oz1_args
