// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Ozaki scheme I: read step shared by the two readers of the reconstruction (recon_reader_f32in.cpp,
// recon_reader_i32in.cpp)
// ---------------------------------------------------------------------------
// Reads the partial sums of the diagonals of output tile i into c_0, ndiag tiles in the order
// d = 0..ndiag-1. The d-th one is tile d*slot_stride + i (slot_stride is the spacing of the planes of the
// diagonals: Mt*Nt by default and, when the output is cut into row bands, the allocated width of a band;
// pstride of recon_build_prog in recon.hpp). The tiles are read in forward order because the reconstruction
// evaluates Horner's rule ((((P_0)*radix + P_1)*radix + ...)) from the first term. The space for the ndiag
// tiles is reserved at once and the reads complete with a single barrier.

#include "api/dataflow/dataflow_api.h"
#include "args.h"

template <class Accessor>
FORCE_INLINE void recon_read_diags(const Accessor& s, uint32_t i, uint32_t ndiag, uint32_t slot_stride, uint32_t tb) {
    constexpr uint32_t cb_in = oz1_args::CB_RECON_IN;
    cb_reserve_back(cb_in, ndiag);
    uint32_t l1 = get_write_ptr(cb_in);
    for (uint32_t d = 0; d < ndiag; ++d) {
        noc_async_read_tile(d * slot_stride + i, s, l1);
        l1 += tb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, ndiag);
}
