// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: reads the n planes of output tile g (tiles s*ntot + g, s = 0..n-1) into a CB from a
// buffer in which the planes are laid out one after another
// ---------------------------------------------------------------------------
// Used by the reader of the reconstruction (recon_reader.cpp. The S per-modulus inputs) and by the reader
// of the composition stage (plane_reader.cpp. The np GEMM outputs). The tiles are carried as raw bits,
// so this does not depend on the format. The n tiles are reserved together and read with a single
// barrier, so the depth of the CB must be a multiple of n (if it is not, a set that crosses the boundary
// of the ring is silently corrupted. The host decides the depth).
// Only data-movement kernels include this header.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"   // cb_reserve_back, get_write_ptr, noc_async_read_tile, etc.

template <class TA>
inline __attribute__((always_inline)) void oz2_fetch_planes(uint32_t cb, const TA& in, uint32_t g, uint32_t n, uint32_t ntot,
                                                            uint32_t tile_bytes) {
    cb_reserve_back(cb, n);
    uint32_t l1 = get_write_ptr(cb);
    for (uint32_t s = 0; s < n; ++s) { noc_async_read_tile(s * ntot + g, in, l1); l1 += tile_bytes; }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}
