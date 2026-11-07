// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：対角ごとの部分和を Int32 で K 方向へ累積する（読み側）
// ---------------------------------------------------------------------------
// Pbig（FP32）と Sbig（Int32）はタイル並びが同一（ndiag 面 x Mt*Nt タイル）なので，
// 対角の構造を意識せず線形に流してよい．ACC=0（最初の K ブロック）では Sbig を
// 読まない．
// コンパイル時引数（並びは oz1_args.h の AccumReaderCT）: ACC（1 なら Sbig も読む），続いて TensorAccessorArgs（P・S）
// 実行時引数（同 AccumReaderRT）: Pbig, 担当タイル数, P 側の先頭タイル, Sbig, S 側の先頭タイル
//
// P と S で先頭を分けてあるのは，Pbig を 1 面のスクラッチにする構成があるため
// （対角ごとに「GEMM -> スクラッチ -> S_d へ累積」を回す）．両者が同じ並びなら
// 同じ値を渡せばよい．
#include "api/dataflow/dataflow_api.h"
#include "oz1_args.h"
using namespace oz1_args;

constexpr uint32_t ACC = get_compile_time_arg_val(AR_CT_ACC);

void kernel_main() {
    uint32_t p_addr    = get_arg_val<uint32_t>(AR_P_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(AR_NUM);
    uint32_t p_start   = get_arg_val<uint32_t>(AR_P_START);
    uint32_t s_addr    = get_arg_val<uint32_t>(AR_S_ADDR);
    uint32_t s_start   = get_arg_val<uint32_t>(AR_S_START);

    constexpr uint32_t cb_p = CB_ACCUM_P;   // P_d（FP32）
    constexpr uint32_t cb_s = CB_ACCUM_S;   // 累積 S_d（Int32）
    constexpr auto p_args = TensorAccessorArgs<AR_CT_COUNT>();
    const auto ap = TensorAccessor(p_args, p_addr);
    constexpr auto s_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto as = TensorAccessor(s_args, s_addr);

    for (uint32_t n = 0; n < num_tiles; ++n) {
        cb_reserve_back(cb_p, 1);
        noc_async_read_tile(p_start + n, ap, get_write_ptr(cb_p));
        if constexpr (ACC) {
            cb_reserve_back(cb_s, 1);
            noc_async_read_tile(s_start + n, as, get_write_ptr(cb_s));
        }
        noc_async_read_barrier();
        cb_push_back(cb_p, 1);
        if constexpr (ACC) cb_push_back(cb_s, 1);
    }
}
