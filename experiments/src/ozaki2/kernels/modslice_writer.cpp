// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：モジュラースライスを DRAM へ書き出すデータ移動カーネル
// ---------------------------------------------------------------------------
// c_16（入力タイル 1 枚につき S 枚の出力タイル）を吸い出し，S 個の「独立した
// 行列」として連続に並べる．行列 i はタイル番号
// [i*ntiles_total, (i+1)*ntiles_total) を占める．
// S の中身は経路による：INT8 経路は法の本数ぶんの剰余行列，桁分割経路は
// 法ごとに 1〜5 枚（1 桁の法は 1 枚，平衡 n 桁は n 枚）の桁行列で，並びの先頭はホスト側の poff[] が持つ．
// 桁分割経路の合成段（ozaki2/kernels/oz2_compose.cpp）も同じライタを使う．あちらは
// 出力が法 1 本ぶんの剰余 1 枚なので planes=1 で呼び，平面の位置は start_id に載せる．
//
// 置き方はコンパイル時引数 LAYOUT（TensorAccessor の後．oz2_args.h の Layout）で選ぶ：
//   LAYOUT_PLANES：行列 i をタイル i*ntot + g に置く（INT8 経路のスライスと合成段）
//   LAYOUT_A_CAT ：桁分割経路の A．法の桁行列を K 方向に連結する．タイル g = r*Ct + c（行 r・K 位置 c）の
//                  桁行列 i を r*(S*Ct) + i*Ct + c に置く（S*Ct 列の 1 枚の行列）
//   LAYOUT_B_PERM：桁分割経路の B．桁行列 i をタイル perm[i]*ntot + g に置く（法の中で逆順．K 方向に積むので
//                  (S*Kt)×Nt の 1 枚の行列になる）
// 1・2 は同じ位の桁積を K 方向に連結して 1 本の GEMM にまとめるための置き方である（尾崎 I の
// スライス連結ライタと同じ考え方．ozaki2_split.h の oz2_product_groups）．
//
// 実行時引数（並びは oz2_args.h の WriterRT）: dst_addr, num_tiles（このコアの担当分）, start_id,
//   planes（入力タイル 1 枚あたりの出力の平面の枚数．上の S）, ntiles_total,
//   LAYOUT_A_CAT ならさらに Ct，LAYOUT_B_PERM ならさらに perm[0..planes-1]

#include "api/dataflow/dataflow_api.h"
#include "oz2_args.h"
using namespace oz2_args;

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(SW_DST_ADDR);
    uint32_t num_tiles  = get_arg_val<uint32_t>(SW_NUM);
    uint32_t start_id   = get_arg_val<uint32_t>(SW_START);
    uint32_t planes     = get_arg_val<uint32_t>(SW_PLANES);   // 入力タイル 1 枚あたりの出力の平面の枚数
    uint32_t ntot       = get_arg_val<uint32_t>(SW_NTOT);   // 行列 1 枚のタイル数（平面の間隔）

    constexpr uint32_t cb_o = CB_PLANE_OUT;
    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(d_args, dst_addr);
    constexpr uint32_t LAYOUT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + SW_CT_LAYOUT);
    const uint32_t Ct = (LAYOUT == LAYOUT_A_CAT) ? get_arg_val<uint32_t>(SW_A_KTILES) : 0u;

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t g = start_id; g < end_id; ++g) {
        for (uint32_t s = 0; s < planes; ++s) {
            cb_wait_front(cb_o, 1);
            uint32_t dst;
            if constexpr (LAYOUT == LAYOUT_A_CAT) {          // A：桁行列を K 方向に連結
                const uint32_t r = g / Ct, c = g - r * Ct;
                dst = r * (planes * Ct) + s * Ct + c;
            } else if constexpr (LAYOUT == LAYOUT_B_PERM) {  // B：桁行列を並べ替えて K 方向に積む
                dst = get_arg_val<uint32_t>(sw_perm(s)) * ntot + g;
            } else {
                dst = s * ntot + g;                 // 法 s の剰余行列の，タイル g
            }
            noc_async_write_tile(dst, d, get_read_ptr(cb_o));
            noc_async_write_barrier();
            cb_pop_front(cb_o, 1);
        }
    }
}
