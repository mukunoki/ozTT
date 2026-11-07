// SPDX-License-Identifier: Apache-2.0
#pragma once
// Margin added to the DRAM estimates of both Ozaki schemes. Independent of the device API.
//
// The DRAM estimates of both schemes add it to what is still to be allocated: Ozaki scheme I when it cuts
// the output into row bands from the free DRAM (plan_panels in ozaki1/plan.hpp), Ozaki scheme II in the
// model with which the automatic choice of the moduli decides whether a set of moduli fits without row
// bands (oz2_dram_one_group in ozaki2/split.hpp, used by ozaki2/awe.hpp). It covers the small buffers
// that the estimates leave out (the out-of-range flags, one-tile buffers, ...) and the rounding of the
// allocator. Both schemes use this single value so that their estimates cannot diverge.

#include <cstdint>

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

inline constexpr uint64_t OZ_DRAM_MARGIN_BYTES = 512ull << 20;  // 0.5 GiB

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
