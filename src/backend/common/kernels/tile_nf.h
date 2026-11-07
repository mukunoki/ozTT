// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// (row, column) -> running index inside a tile (32x32). A tile is stored as four 16x16 faces, and the
// index follows that layout. Used both when a data-movement kernel reads or writes the elements of a
// tile directly (scale_f64_reader.cpp, expdelta.cpp) and when the host reorders between tile order and
// row-major order (common/gemm_host.hpp includes it).
#include <cstdint>

inline uint32_t nf(uint32_t r, uint32_t c) {
    return ((c < 16) ? 0u : 256u) + ((r < 16) ? 0u : 512u) + (r % 16) * 16u + (c % 16);
}
