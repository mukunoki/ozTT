// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：スライスを「対角連結」レイアウトで書き出すデータ移動カーネル
// ---------------------------------------------------------------------------
// c_16（入力タイル 1 枚につき NS 枚のスライスタイル．NS はスライスの本数 nslice = L+1）を吸い出し，GEMM 用の
// 連結バッファへ撒き直す．こうしておくと，あとで対角 d ごとに 1 本の GEMM を
// 発行するとき K 方向が連続に読めるようになる（中間読み出しを
// O(p^2) から O(L) に減らす工夫．論文の ssec:bw（データ移動の削減）の GEMM の起動回数の段落）．
//
// レイアウトは「K ブロック優先」である．K を長さ Kbt（タイル数）の C 個のブロックに
// 分け，ブロックごとにスライス連結の一式 [slice0|…|slice_{NS-1}] を並べる：
//
//   [ブロック 0: slice0|slice1|…|slice_{NS-1}][ブロック 1: …] …
//
// こうするとブロック b・対角 d の GEMM が連続した K 範囲を読めるので，
// K ブロック化しても読み出しの連続性が保たれる（無誤差条件が K ではなく
// ブロック長にかかるようになる．docs/kblock_plan.md）．
// Kbt == Kto（C=1）のときは，ブロックに分けない単純なスライス連結 [slice0|…|slice_{NS-1}] と式が一致する．
//
// 入力タイル g，スライス s（0..NS-1）に対する書き先：
//   mode 0（A，M×Kto）: Acat は M×(NS*Kto) にタイル化される．タイル g の K 位置
//     kt = g%Kto はブロック b = kt/Kbt のブロック内位置 kk = kt%Kbt にあたる．
//     dst = (g/Kto)*(NS*Kto) + b*(NS*Kbt) + s*Kbt + kk
//   mode 1（B，Kto×N）: Bcat は (NS*Kto)×N にタイル化され，各ブロックの中で
//     K 方向の r 番目にスライス (NS-1-r) が入る（逆順）．
//     row = b*(NS*Kbt) + (NS-1-s)*Kbt + kk,  dst = row*Nt + (g%Nt)
//   B 側を逆順に並べるのは，A の s ブロック目と B の (NS-1-s) ブロック目を
//   突き合わせると i+j が一定，すなわち同じ対角になるからである．
//
// 実行時引数（並びは oz1_args.h の SliceWriterRT）: dst_addr, num_in_tiles, start_id, NS, s1, s2, mode, Kbt
//   A: s1 = Kto（タイル数）      B: s1 = Kto, s2 = Nt（タイル数）
//   Kbt = 1 ブロックの K タイル数（Kto を割り切ること）

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/slice_f64_split.h"
#include "../../common/kernels/l1_order.h"
#include "oz1_args.h"
using namespace oz1_args;

// コンパイル時引数（TensorAccessor の後）：SPLIT．1 ならリーダ（slice_f64_reader.cpp）が郵便受け c_7 に
// 置く仕事（入力タイルの後半 512 要素を triple-float へ割る）を，書き出しの合間に引き受ける
// （slice_f64_split.h）．郵便受けも c_16 も待って止まらない：どちらか一方で止まると，
// リーダ（後半を待つ）とスライサ（c_16 の空きを待つ）と互いを待って止まるため．

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(SW_DST_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(SW_NUM);    // 入力タイルの枚数
    uint32_t start_id  = get_arg_val<uint32_t>(SW_START);
    uint32_t nslice    = get_arg_val<uint32_t>(SW_NSLICE);   // スライスの本数（L+1）
    uint32_t s1        = get_arg_val<uint32_t>(SW_S1);     // Kto
    uint32_t s2        = get_arg_val<uint32_t>(SW_S2);     // B は Nt（A では使わない）
    uint32_t mode      = get_arg_val<uint32_t>(SW_MODE);
    uint32_t Kbt       = get_arg_val<uint32_t>(SW_KBT);    // 1 ブロックの K タイル数

    constexpr uint32_t cb_o = CB_SLICE_OUT;
    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(d_args, dst_addr);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + SW_CT_SPLIT);

    uint32_t end_id = start_id + num_tiles;
    uint32_t jobs = 0;                               // 引き受けた後半の数（入力タイルの数だけある）
    for (uint32_t g = start_id; g < end_id; ++g) {
        for (uint32_t s = 0; s < nslice; ++s) {
            if constexpr (SPLIT) {
                // スライスが 1 枚来るまで，郵便受けの仕事を片付けながら待つ
                while (!cb_pages_available_at_front(cb_o, 1)) {
                    if (jobs < num_tiles && cb_pages_available_at_front(SLICE_MBOX_CB, 1)) {
                        invalidate_l1_cache();       // リーダが書いた仕事と NoC が読んだ FP64 を読む前に
                        volatile tt_l1_ptr SliceJob* job =
                            (volatile tt_l1_ptr SliceJob*)get_read_ptr(SLICE_MBOX_CB);
                        const uint32_t src = job->src, dl = job->dl, oh = job->oh, om = job->om,
                                       ol = job->ol, os = job->os;
                        const uint32_t os_last = os + 4u * 1023u;   // 最後に書く番地（s の 1023 番）
                        mbox_writer_before_split();   // 本番は何もしない（探針 1 はここで返す．slice_f64_split.h）
                        slice_split_range(src, dl, oh, om, ol, os, SLICE_SPLIT_J, 1024);
                        // 割った値が L1 に届いてから返す（fence だけでは届いたことにならない．
                        // l1_order.h）．前の fence は FP64・指数差分・仕事の読みの完了も待つので，
                        // 返した後にリーダがそれらを上書きしてよい．
                        l1_writes_landed(os_last);
                        mbox_writer_return();
                        ++jobs;
                    }
                }
            } else {
                cb_wait_front(cb_o, 1);
            }
            uint32_t dst;
            if (mode == 0) {          // A：ブロック b の中で，スライス s を s 番目へ
                uint32_t mt = g / s1, kt = g % s1;
                uint32_t b = kt / Kbt, kk = kt - b * Kbt;
                dst = mt * nslice * s1 + b * nslice * Kbt + s * Kbt + kk;
            } else {                  // B：ブロック b の中で，r 番目にスライス (nslice-1-r) を置く
                uint32_t kt = g / s2, nt = g % s2;
                uint32_t b = kt / Kbt, kk = kt - b * Kbt;
                uint32_t row = b * nslice * Kbt + (nslice - 1 - s) * Kbt + kk;
                dst = row * s2 + nt;
            }
            noc_async_write_tile(dst, d, get_read_ptr(cb_o));
            noc_async_write_barrier();
            cb_pop_front(cb_o, 1);
        }
    }
}
