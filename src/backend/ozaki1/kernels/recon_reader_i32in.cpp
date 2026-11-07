// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: data-movement kernel of the reconstruction that reads Int32 partial sums (paired with
// recon_tf_i32in.cpp)
// ---------------------------------------------------------------------------
// For each output tile i, reads the Int32 values of the diagonals into c_0, ndiag tiles in the order
// d = 0..L (recon_read.h, shared with recon_reader_f32in.cpp, which reads FP32 partial products).
// It also pushes the two mask tiles used for the split into c_1. They are constants, so they are not placed
// in DRAM but built once in the L1 of this core (they are not sent from the host either, because by
// specification there is no communication between the transfer of A and B and the return of C). The masks
// are needed to decode an Int32 into two exact FP32 values (see decode in recon_tf_i32in.cpp).
//
// The upper mask depends on the representation of the Int32. The compile-time argument SIGNMAG selects it:
//   SIGNMAG=0  two's complement (produced by typecast<Float32,Int32>; the accumulation S_d of the deferred
//              reconstruction)
//              -> 0xFFFFFF00. The sign bit is extracted together with the value, so the sign need not be
//                 restored
//   SIGNMAG=1  sign-magnitude (an Int32 packed by the matrix engine; P_d of the INT8 path)
//              -> 0x7FFFFF00. The compute kernel multiplies the sign back
// Mixing them up corrupts negative values (the result fails with a relative error on the order of 1e4), so
// the host must pass the same SIGNMAG to this kernel and to the compute kernel (recon_build_prog in
// recon.hpp).
//
// Runtime arguments (order: the first 5 words of ReaderRT in args.h): input buffer, number of tiles handled
//             by this core, first tile, ndiag, tile spacing of the planes (Mt*Nt by default; with row bands,
//             the allocated width of a band)
#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/l1_order.h"
#include "recon_read.h"
#include "args.h"
using namespace oz1_args;

constexpr uint32_t SIGNMAG = get_compile_time_arg_val(RI_CT_SIGNMAG);
constexpr uint32_t MHI = SIGNMAG ? 0x7FFFFF00u : 0xFFFFFF00u;
constexpr uint32_t MLO = 0x000000FFu;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(RR_SRC_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(RR_NUM);
    uint32_t start_id = get_arg_val<uint32_t>(RR_START);
    uint32_t ndiag = get_arg_val<uint32_t>(RR_NDIAG);
    uint32_t slot_stride = get_arg_val<uint32_t>(RR_SLOT_STRIDE);
    // No accumulation input (RR_ACC_* of the FP32-input path) is received: this path is the final stage of
    // the deferred reconstruction and runs only once, after the accumulation in the K direction has been
    // done in Int32 inside the GEMM.

    constexpr uint32_t cb_mask = CB_MASK;
    constexpr auto s_args = TensorAccessorArgs<RI_CT_COUNT>();
    const auto s = TensorAccessor(s_args, src_addr);
    const uint32_t tb = get_tile_size(CB_RECON_IN);

    // The two mask tiles for the split are built once in the L1 of this core. They are constants, so they
    // need not be placed in DRAM or sent from the host (by specification there is no communication between
    // the transfer of A and B and the return of C). Building them once also avoids reading two tiles
    // from DRAM for every output tile.
    {
        cb_reserve_back(cb_mask, 2);
        volatile tt_l1_ptr uint32_t* mp = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_mask);
        for (uint32_t z = 0; z < 1024; ++z) { mp[z] = MHI; mp[1024 + z] = MLO; }
        // Hand the masks to the compute side only after they have landed in L1 (l1_order.h)
        l1_writes_landed((uint32_t)&mp[2047]);
        cb_push_back(cb_mask, 2);
    }
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i)
        recon_read_diags(s, i, ndiag, slot_stride, tb);   // ndiag tiles in diagonal order (d = 0..L)
}
