// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II（桁分割の 7 経路）：合成段の入力を読むデータ移動カーネル
// ---------------------------------------------------------------------------
// 出力タイル g ごとに，GEMM の出力（桁積の平面）np 枚のタイル s*ntot+g を s = 0..np-1 の順で CB_PLANE_IN へ読む
// （oz2_plane_fetch.h．np は法 1 本ぶんの出力か，塊に切る構成では塊 1 つぶん・部分和の枚数）．合成段の計算カーネル
// oz2_compose.cpp が受け取る．
//
// コンパイル時引数：入力の TensorAccessorArgs．実行時引数（並びは oz2_args.h の PlaneReaderRT）：in_addr, num_tiles（この
// コアの担当分）, start_id（入力の先頭の平面を載せた先頭のタイル番号）, ntot（平面の間隔）, np（1 回に読む平面の枚数）
#include "api/dataflow/dataflow_api.h"
#include "oz2_args.h"
#include "oz2_plane_fetch.h"
using namespace oz2_args;

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(PR_IN_ADDR);
    const uint32_t num     = get_arg_val<uint32_t>(PR_NUM);
    const uint32_t start   = get_arg_val<uint32_t>(PR_START);
    const uint32_t ntot    = get_arg_val<uint32_t>(PR_NTOT);
    const uint32_t np      = get_arg_val<uint32_t>(PR_NPLANES);
    constexpr auto ia = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(ia, in_addr);
    const uint32_t tb = get_tile_size(CB_PLANE_IN);
    for (uint32_t g = start; g < start + num; ++g) oz2_fetch_planes(CB_PLANE_IN, in, g, np, ntot, tb);
}
