// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Bit-level procedures of the normalization in the scaling stage (shared by Ozaki schemes I and II; no
// dependency on the device API)
// ---------------------------------------------------------------------------
// The maximum absolute value (FP64) of each row / column is converted to BF16 (scale_f64_reader.cpp), and
// the normalization exponent e = ceil(log2(.)) and the out-of-range flags are derived from that BF16
// (expdelta.cpp). The host model of Ozaki scheme I that selects the slice count p from the input
// (oz1_dev_exp in ozaki1/auto_precision.hpp) must reproduce the normalization chosen by the device bit for
// bit, so the host calls the very functions used by the kernels (with a copy on the host, a fix made only
// in the kernel would silently change how p is selected).
// Depends only on <cstdint>.

#include <cstdint>

// FP64 |x| (high word h with the sign bit cleared, low word l) -> BF16 bit pattern. CEIL is the rounding
// direction:
//   0 round down; 1 round up (one up when any dropped significand bit is nonzero); 2 round up and, in
//   addition, one up when the value is exactly a power of two (the upper 7 significand bits and the
//   dropped bits are all 0) (only the significand becomes 1 and the exponent does not change, so even
//   2^127 does not become inf; used by BFP2 in Ozaki scheme I, see OZ1_TAU_ABOVE in ozaki1/format.hpp).
//   A carry propagates naturally into the exponent.
// Out-of-range cases are returned as markers that do not appear as the BF16 of an |x| (all have exponent
// 255; they are distinguished by the sign and the significand; see "Accepted magnitude range" in
// scale_f64_reader.cpp):
//   0xFFC0  there is an Inf / NaN element (FP64 exponent 2047)
//   0xFF80  the nonzero maximum is below 2^-126 (not representable as a normalized BF16 number; FP64
//           subnormals included)
//   0x7F80  beyond the BF16 range (2^128 or more; with CEIL set, rounding up produces it as soon as the
//           maximum exceeds the largest finite BF16 value)
// 0 stays 0 (an all-zero row / column).
template <uint32_t CEIL>
inline uint16_t oz_absf64_to_bf16(uint32_t h, uint32_t l) {
    const int32_t E  = (int32_t)((h >> 20) & 0x7FFu);
    if (E == 0x7FF) return 0xFFC0u;                    // Inf / NaN
    const int32_t e8 = E - 896;                        // to the FP32 exponent
    if (e8 <= 0)   return (h | l) ? 0xFF80u : 0u;     // below 2^-126 (nonzero), and 0
    if (e8 >= 255) return 0x7F80u;                     // above the upper limit -> inf
    uint32_t v = ((uint32_t)e8 << 7) | ((h >> 13) & 0x7Fu);
    if constexpr (CEIL == 1) { if ((h & 0x1FFFu) || l) v += 1u; }
    if constexpr (CEIL == 2) { if ((h & 0x1FFFu) || l || !(v & 0x7Fu)) v += 1u; }
    return (uint16_t)v;
}

// Returns the normalization exponent e = ceil(log2(u)) from the BF16 u of the maximum, and ORs a flag into
// bad when u is out of range (bad is not touched when u is in range).
// BF16 has 1 sign bit, 8 exponent bits and 7 significand bits, and its value is (-1)^s * 1.m * 2^(E-127)
// (s=0 because the maximum is an |x|), so
//     ceil(log2(u)) = (E-127) + (m != 0 ? 1 : 0).
// E=0 (0 and subnormals) is treated as u<=0 and gives e=0 (an all-zero row / column; no flag is raised).
// E=255 is one of the markers above; it gives e=0 and a flag is added: 1 for the upper-end inf (0x7F80),
// 2 for the lower-end marker (0xFF80), 4 for the Inf / NaN marker (0xFFC0).
// Adding the flag only in the out-of-range branch and forcing the function inline keep the machine code
// identical to what expdelta.cpp had before it used this function.
__attribute__((always_inline)) inline int32_t oz_bf16_scale_exp(uint16_t u, uint32_t& bad) {
    const uint32_t E = (u >> 7) & 0xFFu, m = u & 0x7Fu;
    int32_t e;
    if (E == 0u)            e = 0;
    else if (E == 0xFFu) { e = 0; bad |= !(u & 0x8000u) ? 1u : m ? 4u : 2u; }
    else                    e = (int32_t)E - 127 + (m ? 1 : 0);
    return e;
}
