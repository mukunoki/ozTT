// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: argument layouts of the slicing stage and the composition stage (the host and the
// kernels see the same single table)
// ---------------------------------------------------------------------------
// If the two sides were matched by number only, adding one entry on one side would silently shift
// everything after it (the same reason as at the top of recon_args.h).
// The host (slice_set_args, build_compose and compose_set_args in stages.hpp) fills the arguments by
// giving the index with the names of this table, and the kernels (modslice_digit.cpp, modslice_int8.cpp,
// modslice_writer.cpp, plane_reader.cpp, compose.cpp) read them by name.
// The table of the reconstruction stage is in recon_args.h; the limits shared by both sides (OZ2_MAXS,
// OZ2_MAXPROD) and the numbers of the digit-making modes are in digit_limits.h.
//
// This header does not depend on the device API (<cstdint> only).
#pragma once
#include <cstdint>
#include "digit_limits.h"

namespace oz2_args {

// --- CB numbers ----------------------------------------------------------------
// The 4 inputs and the residuals of the slicing stage are in the table common to both schemes
// (common/kernels/slice_f64_args.h); the ones here belong to Ozaki scheme II only.
// Input of the composition stage: the GEMM outputs that the plane reader plane_reader.cpp pushes
// (np tiles per output tile)
constexpr uint32_t CB_PLANE_IN  = 0;
// Planes that the writer modslice_writer.cpp writes to DRAM (residues and digit matrices of the slicers,
// residues of the composition stage)
constexpr uint32_t CB_PLANE_OUT = 16;
constexpr uint32_t CB_DIGIT_HI  = 26;   // high digits d_{ND-1}..d1 of A' (in the order they are made. digits.h)
constexpr uint32_t CB_DIGIT_LO  = 27;   // digit d0 of A'

// --- Compile-time arguments of the slicers (modslice_digit.cpp, modslice_int8.cpp) ------------
enum SlicerCT : uint32_t {
    SC_S = 0,        // number of moduli
    SC_RDX,          // float bit pattern of the radix 2^OZ2_DIGIT_BITS that splits A'
    SC_NDIGIT,       // number of radix-2^20 digits of A' (3 to OZ2_MAXDIGIT)
    SC_NRDIG,        // upper bound on the number of digits of a residue (2 or 5. modslice_digit.cpp only)
    SC_OUT_HALF,     // 1 outputs the digits as binary16 bit patterns (FP16 path. modslice_digit.cpp only)
    SC_COUNT
};

// --- Runtime arguments of the slicers ---------------------------------------------
// The first argument is the number of tiles, followed by the per-modulus constants, S values each (ND is
// the number of digits of A').
//   [0]                     number of tiles assigned to the core
//   [1 + 0*S + i]           m_i
//   [1 + 1*S + i]           1/m_i
//   [1 + (2+j)*S + i]       2^(20(j+1)) mod m_i (j = 0..ND-2. Weights of the digits of A' reduced modulo m_i)
// The digit-splitting path (modslice_digit.cpp) additionally has
//   [1 + (ND+1)*S + i]      radix q_i of the digit splitting
//   [1 + (ND+2)*S + i]      1/q_i
//   [1 + (ND+3)*S + i]      digit-making mode (float bit pattern of the Oz2XMode number)
//   [1 + (ND+4)*S + i]      number of digits n of balanced n digits (uint32. 0 for the other methods)
// Writing the numbers directly would let the way the host fills the arguments and the way the kernel
// reads them drift apart independently. The kernels compute the position of modulus 0 (i = 0) as a
// constexpr and then add the runtime i (calling these functions with the runtime i changes the register
// allocation of the generated code even though the values are the same).
inline constexpr uint32_t sl_ntiles()                                   { return 0u; }
inline constexpr uint32_t sl_mod(uint32_t S, uint32_t i)                { return 1u + 0u * S + i; }
inline constexpr uint32_t sl_rinv(uint32_t S, uint32_t i)               { return 1u + 1u * S + i; }
inline constexpr uint32_t sl_weight(uint32_t S, uint32_t j, uint32_t i) { return 1u + (2u + j) * S + i; }
inline constexpr uint32_t sl_q(uint32_t S, uint32_t nd, uint32_t i)     { return 1u + (nd + 1u) * S + i; }
inline constexpr uint32_t sl_qinv(uint32_t S, uint32_t nd, uint32_t i)  { return 1u + (nd + 2u) * S + i; }
inline constexpr uint32_t sl_xmode(uint32_t S, uint32_t nd, uint32_t i) { return 1u + (nd + 3u) * S + i; }
inline constexpr uint32_t sl_ndig(uint32_t S, uint32_t nd, uint32_t i)  { return 1u + (nd + 4u) * S + i; }
inline constexpr uint32_t sl_count(uint32_t S, uint32_t nd, bool digitk) {
    return 1u + (nd + 1u + (digitk ? 4u : 0u)) * S;
}

// --- Slice writer (modslice_writer.cpp. Also used by the composition stage) --------------------
// Compile-time arguments (after the TensorAccessorArgs of the output)
enum WriterCT : uint32_t {
    SW_CT_LAYOUT = 0,   // placement (Layout below)
    SW_CT_COUNT
};
enum Layout : uint32_t {
    // Matrix i is placed at tile i*ntot + g (slices of the INT8 path, and the composition stage)
    LAYOUT_PLANES = 0,
    // A of the digit-splitting path: the digit matrices of the moduli are concatenated along K (the 1
    // word SW_A_KTILES that follows is the number of tiles of A along K)
    LAYOUT_A_CAT = 1,
    // B of the digit-splitting path: digit matrix i is placed at tile perm[i]*ntot + g (the planes words
    // that follow are perm)
    LAYOUT_B_PERM = 2,
};
// Runtime arguments. The first SW_FIXED words are independent of the placement; what follows depends on
// the placement (1 word for LAYOUT_A_CAT, planes words for LAYOUT_B_PERM)
enum WriterRT : uint32_t {
    SW_DST_ADDR = 0,    // start address of the output
    SW_NUM,             // number of input tiles assigned to this core
    SW_START,           // their first tile number (the composition stage puts the output plane here)
    // Number of output tiles per input tile (planes: the number of moduli on the INT8 path, nplane of the
    // host on the digit-splitting paths, 1 for the composition stage)
    SW_PLANES,
    SW_NTOT,            // tiles per matrix (spacing of the planes)
    SW_FIXED
};
// LAYOUT_A_CAT: number of tiles Ct of A along K
inline constexpr uint32_t SW_A_KTILES = SW_FIXED;
// LAYOUT_B_PERM: destination plane of digit matrix s
inline constexpr uint32_t sw_perm(uint32_t s) { return SW_FIXED + s; }
// Number of runtime arguments; the parameter S is the number of planes (SW_PLANES)
inline constexpr uint32_t sw_count(uint32_t layout, uint32_t S) {
    return SW_FIXED + (layout == LAYOUT_A_CAT ? 1u : layout == LAYOUT_B_PERM ? S : 0u);
}

// --- Runtime arguments of the plane reader of the composition stage (plane_reader.cpp) ---------
enum PlaneReaderRT : uint32_t {
    PR_IN_ADDR = 0,     // start address of the input (planes of digit products)
    PR_NUM,             // number of output tiles assigned to this core
    PR_START,           // their first tile number (the first plane to read, in0, is put here as in0*ntot)
    PR_NTOT,            // tiles per plane (spacing of the planes)
    PR_NPLANES,         // number of planes np read per output tile
    PR_COUNT
};

// --- Runtime arguments of the compute kernel of the composition stage (compose.cpp) ------------
// The number of runtime arguments must be constant for a kernel, so OZ2_MAXPROD coefficient slots are
// always reserved and the unused ones are filled with 0 (because the same workload is enqueued again for
// each modulus with only the arguments replaced).
enum ComposeRT : uint32_t {
    CP_NTILES = 0,      // number of tiles assigned to the core
    // Number np of outputs to add up (GEMM outputs, each merging the digit products of the same place
    // value) (<= OZ2_MAXPROD)
    CP_NP,
    CP_MOD,             // float bit pattern of the modulus m
    CP_RINV,            // float bit pattern of 1/m
    // 1 selects the method that reduces the digit products modulo m first (fold_first in moduli.hpp)
    CP_FOLD_FIRST,
    CP_FIXED
};
inline constexpr uint32_t cp_coeff_lo(uint32_t j) { return CP_FIXED + j; }                       // B_j = c_j mod m
inline constexpr uint32_t cp_coeff_hi(uint32_t j) { return CP_FIXED + (uint32_t)OZ2_MAXPROD + j; }  // A_j = c_j*2^12 mod m
inline constexpr uint32_t CP_COUNT = CP_FIXED + 2u * (uint32_t)OZ2_MAXPROD;

}  // namespace oz2_args
