// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: data-movement kernel that reads the input of the reconstruction (the non-deferred path
// that reads FP32 partial products; paired with recon_tf_f32in.cpp)
// ---------------------------------------------------------------------------
// For each output tile i, streams the partial sums P_d (FP32) of the diagonals into c_0, ndiag tiles in the
// order d = 0..L (recon_read.h, shared with recon_reader_i32in.cpp, which reads Int32 partial sums). In the
// configuration that runs the Horner evaluation for every K block, the triple accumulated up to the previous
// block (hi, mid, lo) is also pushed into c_2.

// The order of the runtime arguments is ReaderRT in args.h (the compile-time arguments are the
// TensorAccessorArgs in the order P, then the accumulation hi, mid, lo).

#include "api/dataflow/dataflow_api.h"
#include "recon_read.h"
#include "args.h"
using namespace oz1_args;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(RR_SRC_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(RR_NUM);
    uint32_t start_id = get_arg_val<uint32_t>(RR_START);
    uint32_t ndiag = get_arg_val<uint32_t>(RR_NDIAG);
    uint32_t slot_stride = get_arg_val<uint32_t>(RR_SLOT_STRIDE);
    uint32_t acc_hi = get_arg_val<uint32_t>(RR_ACC_HI);   // 0 means no accumulation
    uint32_t acc_mid = get_arg_val<uint32_t>(RR_ACC_MID);
    uint32_t acc_lo = get_arg_val<uint32_t>(RR_ACC_LO);

    constexpr uint32_t cb_acc = CB_ACC;
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, src_addr);
    constexpr auto h_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto ah = TensorAccessor(h_args, acc_hi);
    constexpr auto m_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto am = TensorAccessor(m_args, acc_mid);
    constexpr auto l_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto al = TensorAccessor(l_args, acc_lo);
    const uint32_t tb = get_tile_size(CB_RECON_IN);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        recon_read_diags(s, i, ndiag, slot_stride, tb);   // ndiag tiles in diagonal order (d = 0..L)
        if (acc_hi) {   // accumulation up to the previous block (order hi, mid, lo)
            cb_reserve_back(cb_acc, 3);
            uint32_t la = get_write_ptr(cb_acc);
            noc_async_read_tile(i, ah, la);
            noc_async_read_tile(i, am, la + tb);
            noc_async_read_tile(i, al, la + 2 * tb);
            noc_async_read_barrier();
            cb_push_back(cb_acc, 3);
        }
    }
}
