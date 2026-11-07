// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：平面を並べたバッファから，出力タイル g の平面 n 枚（タイル s*ntot + g，s = 0..n-1）を CB へ読む
// ---------------------------------------------------------------------------
// 再構成のリーダ（recon_reader.cpp．法ごとの入力 S 枚）と合成段のリーダ（oz2_plane_reader.cpp．GEMM の出力 np 枚）が
// 使う．タイルはビット列のまま運ぶので形式に依らない．n 枚をまとめて確保して 1 回のバリアで読み終えるので，CB の深さは
// n の倍数でなければならない（割り切れないとリングの境界をまたいだ組が黙って壊れる．深さはホストが決める）．
// データ移動カーネルだけが取り込む．
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"   // cb_reserve_back・get_write_ptr・noc_async_read_tile など

template <class TA>
inline __attribute__((always_inline)) void oz2_fetch_planes(uint32_t cb, const TA& in, uint32_t g, uint32_t n, uint32_t ntot,
                                                            uint32_t tile_bytes) {
    cb_reserve_back(cb, n);
    uint32_t l1 = get_write_ptr(cb);
    for (uint32_t s = 0; s < n; ++s) { noc_async_read_tile(s * ntot + g, in, l1); l1 += tile_bytes; }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}
