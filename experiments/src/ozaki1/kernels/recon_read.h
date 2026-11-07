// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 尾崎スキーム I：再構成のリーダ 2 本（recon_reader_f32in.cpp・recon_reader_i32in.cpp）が共有する読み込み
// ---------------------------------------------------------------------------
// 出力タイル i の対角ごとの部分和を d = 0..ndiag-1 の順に ndiag 枚 c_0 へ読み込む．d 番目はタイル d*slot_stride + i
// （slot_stride は対角の面の間隔．既定は Mt*Nt で，行バンドに切るときはバンドの確保幅．ozaki1_recon.hpp の
// recon_build_prog の pstride）．順方向に読むのは，再構成側が Horner 法（(((P_0)*radix + P_1)*radix + ...)）で
// 先頭から畳むためである．ndiag 枚ぶんをまとめて確保し，1 回のバリアで読み終える．

#include "api/dataflow/dataflow_api.h"
#include "oz1_args.h"

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
