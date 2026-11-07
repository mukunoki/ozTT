// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：再構成への入力を読むデータ移動カーネル（FP32 の部分積を読む非遅延の経路．recon_tf_f32in.cpp と対）
// ---------------------------------------------------------------------------
// 出力タイル i ごとに，対角ごとの部分和 P_d（FP32）を d = 0..L の順に ndiag 枚，c_0 へ流し込む
// （recon_read.h．Int32 の部分和を読む recon_reader_i32in.cpp と共有）．K ブロックごとに Horner を回す構成では，
// 前ブロックまでの累積の 3 本組（hi・mid・lo）も c_2 へ積む．

// 実行時引数の並びは oz1_args.h の ReaderRT（コンパイル時引数は TensorAccessorArgs が P・累積の hi・mid・lo の順）．

#include "api/dataflow/dataflow_api.h"
#include "recon_read.h"
#include "oz1_args.h"
using namespace oz1_args;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(RR_SRC_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(RR_NUM);
    uint32_t start_id = get_arg_val<uint32_t>(RR_START);
    uint32_t ndiag = get_arg_val<uint32_t>(RR_NDIAG);
    uint32_t slot_stride = get_arg_val<uint32_t>(RR_SLOT_STRIDE);
    uint32_t acc_hi = get_arg_val<uint32_t>(RR_ACC_HI);   // 0 なら累積なし
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
        recon_read_diags(s, i, ndiag, slot_stride, tb);   // 対角の順（d = 0..L）に ndiag 枚
        if (acc_hi) {   // 前ブロックまでの累積（hi,mid,lo の順）
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
