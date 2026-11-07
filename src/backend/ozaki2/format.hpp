// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: the values of the format this translation unit is compiled for (host
// side). The format is selected by one macro -DOZ_FMT_<format> (common/format.hpp provides
// OZ_FMT and OZ_FMT_ID); the scheme II specific values are the path OZ2_PARAM from the
// table in split.hpp. moduli.hpp, plan.hpp and stages.hpp include this header.
//
// INT8 carries each residue as it is (sign-magnitude UInt8), so its moduli are at most
// 255. The other seven formats split the symmetric residue into digits
// (kernels/modslice_digit.cpp): they are not bound by the exact integer range of the
// format and reach moduli of about (2d+1)^2 for a digit bound d with two digits, and
// (2d+1)^n with n balanced digits (three for BFP4, up to five for BFP2); the splitting
// methods are in split.hpp.

#include <algorithm>
#include <cstdint>
#include "common/format.hpp"
#include "ozaki2/split.hpp"

namespace ttgemm::detail::TTGEMM_BACKEND_NS {

// Path of the selected format.
using Oz2Param = Oz2Route;
inline constexpr Oz2Param OZ2_PARAM = oz2_param(OZ_FMT_ID);

// Digit splitting switches whole functions and struct members, so it is also needed at
// preprocessing time (OZ2_DIGITK). It is derived from the format macro and checked against
// the digit bound of the table.
#if !defined(OZ_FMT_INT8)
#define OZ2_DIGITK 1
#endif
#ifdef OZ2_DIGITK
static_assert(OZ2_PARAM.digit_bound != 0, "a digit-splitting format needs a digit bound");
#else
static_assert(OZ2_PARAM.digit_bound == 0, "the INT8 format does not split digits");
#endif

// For choices of values and branches (if constexpr, conditional expressions). OZ2_DIGITK
// is used only for the types, functions and members that exist only with digit splitting.
inline constexpr bool OZ2_DIGIT = OZ2_PARAM.digit_bound != 0;

// MathFidelity of the GEMM stage, by the rule shared with scheme I (oz_fidelity_rule in
// common/format.hpp) applied to the largest digit: HiFi4 for INT8, LoFi for the digit
// formats, whose digits are at most 32 (integers up to 31 have at most 5 significant bits
// and 32 is a power of two, so all fit the upper 5 bits that LoFi multiplies).
inline tt::tt_metal::MathFidelity oz2_fidelity() {
    // With sparse digits (FP8, sparse_max = 32) those are the largest digits.
    return oz_fidelity_rule((uint32_t)std::max(OZ2_PARAM.digit_bound, OZ2_PARAM.sparse_max));
}

}  // namespace ttgemm::detail::TTGEMM_BACKEND_NS
