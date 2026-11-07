// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: argument layouts of the reconstruction stage (the host and the kernels look at
// one and the same table)
// ---------------------------------------------------------------------------
// The host passes sequences of uint32_t to SetRuntimeArgs and compile_args, and the kernels receive
// them with get_arg_val<uint32_t>(i) and get_compile_time_arg_val(i). The two sides correspond only
// by index, so adding one entry on one side shifts everything after it. Even when shifted the type is
// the same uint32_t, so with luck the kernel stops with "Index out of range", and without luck it
// silently reads another value.
//
// Therefore the index tables are placed here and both sides refer to them. The host fills arrays of
// *_COUNT entries indexed by the names of these tables, and the kernels read by the enum names, so
// changing the layout only requires an edit here.
// The compile-time arguments of the reader and the writer (ReaderWriterCT below) follow the
// TensorAccessorArgs, so the host fills them by name with append_rw_ct in ozaki2/stages.hpp. The CB
// indices are also placed here, and the host and the kernels look them up by the same names.
// The tables of the slice stage and the composition stage are in args.h, and the upper limits shared
// by both sides are in digit_limits.h.
//
// This header does not depend on the device API (only <cstdint>).
#pragma once
#include <cstdint>

namespace recon_args {

// --- CB indices (created by recon_make_cbs on the host (ozaki2/stages.hpp); used by the reader, the
//     compute kernel and the writer) ---------------------------------------------------------------
constexpr uint32_t CB_IN     = 0;    // per-modulus input (residue or GEMM output); S tiles per output tile
constexpr uint32_t CB_MASK   = 1;    // the 2 masks that split the sign-magnitude value (INT8 path only;
                                     // the reader builds them in L1)
constexpr uint32_t CB_LIMB   = 16;   // limbs of frac (the compute kernel writes them, the writer consumes
                                     // them, the reader only looks at the address)
constexpr uint32_t CB_F64    = 19;   // writer scratch: one FP64 tile (OZ_F64_TILE_BYTES; the reader looks
                                     // at the same L1)
constexpr uint32_t CB_DLT_A  = 20;   // writer scratch: row exponent deltas
constexpr uint32_t CB_DLT_B  = 21;   // writer scratch: column exponent deltas
constexpr uint32_t CB_A128   = 22;   // writer scratch: one tile of 128-bit fixed point
                                     // (OZ_A128_TILE_BYTES; two-pass configuration only)
constexpr uint32_t CB_SYNC   = 26;   // handoff between the reader and the writer (one word, the address
                                     // of the limbs; only when the assembly is shared)
constexpr uint32_t SYNC_PAGE = 64;   // size of one page of CB_SYNC [bytes]

// --- Runtime arguments of the reader (recon_reader.cpp) -----------------------
enum Reader : uint32_t {
    R_IN_ADDR = 0,   // start address of the input (per-modulus planes, or residues)
    R_NUM,           // number of output tiles handled by this core
    R_START,         // index of the first of those tiles
    R_NTOT,          // total number of output tiles (also the spacing between the planes)
    R_S,             // number of moduli
    R_MASK,          // 1: build the 2 masks that split the sign-magnitude value in L1 and send them to
                     // c_1 (INT8 path)
    R_NLIMB,         // triple-float or two limbs
    R_WHI,           // upper 32 bits of the final scale W
    R_WLO,           // lower 32 bits of W
    R_QW,            // exponent of W (an int32 carried as is in a uint32)
    R_SH2,           // exponent to multiply back, 2*(b-(ND-1)*G)
    R_JSPLIT,        // split point of the shared FP64 assembly (in the first pass of the two-pass
                     // configuration, of the conversion to 128-bit fixed point); the reader takes the
                     // first jsplit elements
    R_SEM_PUB,       // semaphore of the tag with which the writer announces where the limbs are
    R_SEM_DONE,      // semaphore of the tag with which the reader announces that the first part is done
    R_COUNT
};

// --- Runtime arguments of the writer (recon_writer.cpp) -----------------------
enum Writer : uint32_t {
    W_OUT0 = 0,      // destination of a pass that is not FINAL. The first pass of the two-pass
                     // configuration writes 128-bit fixed point (16 KB pages); the final pass of the
                     // diagnostic mode that skips the FP64 assembly (not used by the library) writes
                     // limb plane 0 (planes 1 and 2 go to W_OUT1 and W_OUT2)
    W_NUM,
    W_START,
    W_OUT1,
    W_OUT2,
    W_NLIMB,
    W_O64,           // start address of the FP64 output
    W_DLT_A,         // row exponent deltas
    W_DLT_B,         // column exponent deltas
    W_NT,            // number of tiles of the output in the column direction (used to derive the row
                     // and the column from the tile index)
    W_WHI,
    W_WLO,
    W_QW,
    W_SH2,
    W_A128,          // 128-bit fixed point of the first pass, read by the final pass of the two-pass
                     // configuration (16 KB pages)
    W_JLO,           // number of leading elements handed to the reader (0 when the assembly is not shared)
    W_SEM_PUB,
    W_SEM_DONE,
    W_OUT_OFF,       // first tile of the band when the output is split into row bands (added to the
                     // tile index of o64 and of the row and column exponent deltas; 0 with one band)
    W_COUNT
};

// --- Role of a pass (the values of CT_ROLE below) -----------------------------
// The numbers are not the order of the passes: the one-pass configuration is ROLE_SINGLE; in the
// two-pass configuration the first pass is ROLE_TWO_FIRST (adds its own 3 limbs into 128-bit fixed
// point and writes it) and the final pass is ROLE_TWO_FINAL (reads the fixed point of the first pass,
// adds its own 3 limbs and assembles FP64).
// The composition stage does not use the reader of the reconstruction (plane_reader.cpp)
enum ReconRole : uint32_t { ROLE_SINGLE = 0, ROLE_TWO_FINAL = 1, ROLE_TWO_FIRST = 2 };

// --- Compile-time arguments of the reader and the writer (after the TensorAccessorArgs) ---
// Both take the same three in the same order. The number of TensorAccessorArgs differs per side, so
// each side counts from its own next_compile_time_args_offset().
// The meaning of CT_FINAL_OR_SHARE differs per side, so the kernels receive it under different
// names: the writer (FINAL) is passed "whether this pass is the final pass that assembles FP64", and
// the reader (SHARE) is passed "whether it takes the first part of the assembly (in the first pass
// of the two-pass configuration, of the conversion to 128-bit fixed point)". The host passes
// SHARE = 1 in every pass of every configuration (recon_make_kernels in ozaki2/stages.hpp) and
// FINAL = 1 in every pass but the first of two (ReconPassShape::FINAL in the same file).
enum ReaderWriterCT : uint32_t {
    // Writer: whether this is the final pass that assembles FP64. Reader: whether it takes the first
    // part of the assembly (see the note above)
    CT_FINAL_OR_SHARE = 0,
    CT_ROLE,                 // role of the pass (ReconRole)
    CT_PROBE,        // diagnostic (not used by the library; always 0): 1 removes assemble_frac_f64
    CT_COUNT
};

// --- Compile-time arguments of the compute kernel (recon_crt.cpp) -------------
enum CrtCT : uint32_t {
    C_S = 0,         // number of moduli
    C_NLIMB,         // triple-float or two limbs
    C_IN_RESIDUE,    // whether the input is a reduced residue or the raw int32 of the GEMM stage
    C_NRP,           // number of reciprocal fragments
    C_NSMALL_FRAG,   // how many of the trailing reciprocal fragments are added with tfAddSmall (the
                     // host derives it from the fragment magnitudes; oz2_choose_small_tail in
                     // ozaki2/moduli.hpp)
    C_PROBE,         // diagnostic (not used by the library; always 0): CPROBE 0..2
    C_RAWMASK,       // if the bit of modulus i is set, its input is a GEMM output that did not pass
                     // through the composition stage (one-digit modulus; reduced modulo m before use)
    C_COUNT
};

// --- Runtime arguments of the compute kernel ----------------------------------
// The number of tiles comes first, followed by the per-modulus constants, S entries each.
//   [0]                     number of tiles handled
//   [1 + 0*S + i]           m_i
//   [1 + 1*S + i]           1/m_i
//   [1 + 2*S + i]           y_i = (M/m_i)^{-1} mod m_i
//   [1 + (3+t)*S + i]       reciprocal fragment t (t = 0..NRP-1)
//   [1 + (3+NRP)*S + i]     2^16 mod m_i (not passed for the residue input)
// Writing the indices directly would let the packing of the host and the reading of the kernel drift
// apart separately.
inline constexpr uint32_t crt_ntiles()                     { return 0u; }
inline constexpr uint32_t crt_mod(uint32_t S, uint32_t i)  { return 1u + 0u * S + i; }
inline constexpr uint32_t crt_rinv(uint32_t S, uint32_t i) { return 1u + 1u * S + i; }
inline constexpr uint32_t crt_yi(uint32_t S, uint32_t i)   { return 1u + 2u * S + i; }
inline constexpr uint32_t crt_frag(uint32_t S, uint32_t t, uint32_t i) {
    return 1u + (3u + t) * S + i;
}
inline constexpr uint32_t crt_m16(uint32_t S, uint32_t nrp, uint32_t i) {
    return 1u + (3u + nrp) * S + i;
}
inline constexpr uint32_t crt_count(uint32_t S, uint32_t nrp, bool in_residue) {
    return 1u + (3u + nrp + (in_residue ? 0u : 1u)) * S;
}

}  // namespace recon_args
