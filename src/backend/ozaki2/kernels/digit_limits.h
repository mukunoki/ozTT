// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: limits and constants shared by the host and the kernels (both sides see one value)
// ---------------------------------------------------------------------------
// Values that silently give wrong results when only one side is changed are defined here, once. For
// example, if only the host increased the coefficient slots OZ2_MAXPROD of the composition stage, the
// kernel (compose.cpp) would read A_j from the old position, get a B-side coefficient or 0, and the
// residue would silently be wrong. If only the host added a slicing-method number (Oz2XMode), the slicer
// (modslice_digit.cpp) would silently fall into the default branch.
// The argument layouts are in recon_args.h and args.h.
//
// This header does not depend on the device API (<cstdint> only). The host includes it through split.hpp.
#pragma once
#include <cstdint>

// Upper bound on the number of moduli. It is the length of the per-modulus fixed arrays of the kernels
// (MAXS in modslice_digit.cpp, modslice_int8.cpp and recon_crt.cpp), and the modulus selection of the
// host (moduli.hpp) never selects a set larger than this
inline constexpr int OZ2_MAXS = 24;
// Upper bound on the number of matrix products per modulus. 2-digit splitting takes 3 to 4, but balanced
// 5 digits (BFP2) take 25. It is the size of the coefficient slots of the composition stage
// (compose.cpp), and it also determines the number of runtime arguments (CP_COUNT in args.h)
inline constexpr int OZ2_MAXPROD = 32;
// Radix (as a number of bits) with which A' is split into 3 to OZ2_MAXDIGIT radix-2^20 digits. The
// slicer takes the residue of each digit first and multiplies by the weight afterwards (oz2_residue in
// digits.h), so a digit may be as large as 2^22, which ModRound handles exactly. With 20 bits the
// default b=54 is carried with 3 digits (up to 60 bits), and 4 or 5 digits are used only when b is
// raised above 60 or 80 (ND in choose_moduli_and_width of plan.hpp).
inline constexpr int OZ2_DIGIT_BITS = 20;
// Upper bound on the number of digits. A' is carried with N radix-2^20 digits, so b <= N*20. 3 digits
// are enough for the default b<=54; 4 digits are used only when b is raised above 60 for an input with
// a wide dynamic range, and 5 digits only when b is raised above 80 (digits.h).
// 5 digits are needed for inputs in which an element smaller than 2^-80 times the row maximum is
// multiplied by a large element of the other matrix and determines one element of the output (such
// inputs need b above 80). This is bounded by the FP64 to triple-float conversion, not only by b: the
// conversion (split_f64 in common/kernels/f64_tf.h) is verified exact only for elements down to about
// 72 + sh binades below the row / column maximum (sh = b - 20*(ND-1), at most 20; about 92 binades at
// best), and below that it drops the lower limbs, so a wider b cannot recover elements further down.
inline constexpr int OZ2_MAXDIGIT = 5;
// Upper bound on the number of pieces of the reciprocals in the reconstruction (length of the fixed
// arrays in recon_crt.cpp). The piece counts OZ2_NRP0, NRP1 and NRP2 of the host (moduli.hpp) are at
// most this
inline constexpr int OZ2_MAXRP = 5;
// Number of bits of the radix of the 2^12 split in the composition stage (the method fold_first=0.
// hi = round(P/2^12), lo = P - hi*2^12 in compose.cpp). The host passes the coefficients as
// A_j = c_j*2^12 mod m and B_j = c_j mod m (OZ2_COMPOSE_SPLIT in split.hpp and coeff_hi of
// oz2_product_terms), and the modulus selection counts the bound on the composition sum with this radix
// (oz2_plan_concat in moduli.hpp). Changing only one side makes the residue silently wrong
inline constexpr int OZ2_COMPOSE_SPLIT_BITS = 12;

// Numbers of the digit-making modes of the slicer (modslice_digit.cpp) (the per-modulus runtime argument
// XMODE. It is passed as a float bit pattern and compared as a float).
// WARNING: keep the float comparison. A version that passed XMODE as uint32 and compared integers hung
// the TRISC thread of the slicer on hardware (the board had to be reset); the cause was not identified
// (only the instruction sequence and the timing of the math and pack threads differ), and the float
// comparison has not shown the hang.
// The host derives the number with oz2_xmode from the splitting method of the modulus (OZ2_SPLIT_* in
// split.hpp). Methods (A) and (C) are the same procedure as seen from the slicer (only the composition
// formula on the host side differs), so they share one number.
enum Oz2XMode : uint32_t {
    OZ2_XMODE_RND = 0,   // 2 balanced digits by rounding to nearest (square (A), WIDE (C))
    OZ2_XMODE_KAR = 1,   // 3 Karatsuba digits by rounding up ((B). Radix q)
    OZ2_XMODE_SQS = 2,   // square with sparse digits ((E). FP8 only)
    OZ2_XMODE_BAL = 3,   // balanced n digits ((D). The number of digits n is the NDIG that follows)
    OZ2_XMODE_ONE = 4,   // one digit ((F). The moduli m <= 65 of the AWE path with digit bound 32)
    // When a number is added, also update the branches and the static_assert in modslice_digit.cpp
    OZ2_XMODE_COUNT
};
