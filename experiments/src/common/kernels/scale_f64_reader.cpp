// SPDX-License-Identifier: Apache-2.0
//
// スケール算出：行／列ごとの最大絶対値を FP64 のビット列のまま RISC-V で求めるデータ移動カーネル
// ---------------------------------------------------------------------------
// 役割：DRAM にある FP64 のオペランドを読み，出力タイル 1 枚（A なら 32 行，B なら 32 列）
// ごとに K 方向の Kt 枚を通して
//     sA[m] = max_k |A[m,k]|   （MODE 0，行ごと）
//     sB[n] = max_k |B[k,n]|   （MODE 1，列ごと）
// を求め，その BF16 表現をタイルにして DRAM へ書く．expdelta.cpp がそれを指数へ足す差分にする．
// 同じカーネルを 1 つのコアの 2 つのデータ移動 RISC-V（RISCV_0・RISCV_1）に置き，コアの担当する
// 出力タイルを半分ずつ受け持たせる（ホストの scale_build．作業領域の CB は別々）．
//
// FP64 のまま最大値を取り，BF16 にするのは出力の 32 個だけで，計算カーネルもライタも使わない．要素ごとに |x| の BF16 を
// 作って行列エンジンの MAX リダクションで畳む方式では，BF16 化をデータ移動の RISC-V が 1 要素ずつ volatile の読み書きで
// 行うので段の時間がその処理で決まる（両方式の時間は docs/nongemm_opt_results.md の §4）．
//
// 要素ごとに BF16 にしてから最大値を取る方式との関係
//   FP64 の |x| の最大値を取ってから BF16 化 f（切り捨て，または下の CEIL の切り上げ）を 1 回だけ当てる．
//   |x| の大小は，符号ビットを落とした 63 bit を上位語・下位語の順に符号なし整数として比べればよい（有限の FP64 の
//   非負値はビット列の順と値の順が一致する．NaN は inf より大きく並ぶが，どちらも f で同じ印 0xFFC0 になる）．下位語まで
//   比べるのは，CEIL の判定が落とす仮数ビット（下位語を含む）を見るからである．
//   f は非零で 2^-126 未満の最大値を範囲外の印 0xFF80 に写す（下の「受け付ける大きさの範囲」）．0xFF80 は BF16 の
//   値としても uint16 としても順序を保たないので f は単調ではなく，BF16 で比べてから f を当てると印が大小比較に
//   入ってしまう．FP64 のまま最大値を取ってから f を 1 回当てるので，0xFF80 が他の要素との比較に入ることはない
//   （src/checks/scale_check.cpp が参照と比べる）．要素ごとに BF16 にしてから最大値を取る方式は，下端を 0 に潰す f
//   （2^-126 未満は 0）なら |x| について単調非減少なので max_k f(|x_k|) = f(max_k |x_k|) となり成り立つ．
//   その方式とこの実装がビット単位で一致するのは，最大値が 2^-126 以上の有限値か 0 の行（列）である（2^-126 未満の
//   非零の最大値の行（列）では，そちらは 0 を，この実装は 0xFF80 を出す．Inf・NaN の行（列）では，そちらは inf か
//   NaN の BF16 を，この実装は 0xFFC0 を出す）．
//
// 切り上げ（CEIL）
//   コンパイル時引数 CEIL を立てると，落とすビットが残っていれば BF16 を 1 つ上へ上げる．
//   scheme II は正規化後の値が 1 を超えると |A'| <= 2^b を破って CRT の窓から出るので，
//   最大値は必ず真値以上でなければならない．scheme I は展開の上限 R = d/(β-1) が目標を
//   1/128 以上上回る形式なら切り捨てでよく，余裕の薄い INT8（R = 127/253）と BFP2（R = 1）
//   だけが切り上げる（ozaki1_main.cpp の bf_ceil）．切り上げが働くのは BF16 の正規化域に
//   入る最大値だけである．
//   CEIL=2 は 1 に加えて，最大値がちょうど 2 のべき（仮数がすべて 0）のときも 1 つ上へ上げる．BF16 そのものは
//   最大値以上で，真に大きくなるとは限らない（1.5 は 1.5 のまま）．最大値より真に大きくなるのは，そこから
//   expdelta.cpp が e = ceil(log2(.)) として作るスケール 2^e のほうで，2^e は最大値より真に大きい最小の 2 のべき
//   2^(floor(log2 max)+1) になる（CEIL=1 とスケールが違うのは最大値がちょうど 2 のべきの行・列だけで，1 段上がる）．
//   目標 2^nshift の値そのものを桁 1 個で表せない BFP2_b（R = 目標 = 1）が使う：最大値の要素がちょうど目標に
//   なると，その展開は毎段 桁 1・残差 1 のまま終わらない（ozaki1_main.cpp の tau_above）．最大値が 2 のべきの
//   すぐ下（正規化後 1-δ）の行・列も先頭に約 log2(1/δ) 個の桁 1 が並ぶが，並びは有限で終わるので，ここ（スケール）
//   では扱わずスライス数 p で扱う（同じ注記．BFP2_b の既定の p=59 は scripts/diagnose/select_slice_count.sh の規則 (ii) で選んだ）．
//
// 受け付ける大きさの範囲（この経路が課す制限）
//   上端 2^128 以上の要素があると最大値が inf になる（CEIL を立てているときは，
//   BF16 の最大有限値 2^128 - 2^120 を超えた時点で切り上げが inf を作る）．
//   expdelta.cpp が指数 255 を見て旗を立て，C を読み終えたあとに
//   expdelta_check が止めるので，誤答にはならない（使う側はオペランド全体を
//   2 のべきで割ってから渡す）．FP64 の上限は 2^1024 なので，ここは f64_tf.h が言う
//   「有限かつ正規化数の FP64」より狭い．
//   Inf・NaN（FP64 の指数 2047）は，上位語の符号を落とした比較でどの有限値よりも大きく並ぶので，その行・列の
//   最大値として残る．これは 2 のべきで割っても直らないので，上端の inf とは別の印 0xFFC0（符号 1・指数 255・
//   仮数 0x40．下端の印 0xFF80 とは仮数で見分ける）を書き，expdelta.cpp が別の旗（ビット 2）を立てる．
//   下端：非零の最大値が 2^-126 を下回ると，BF16 の正規化数では表せない．ここを 0 にすると，
//   行（列）のすべてがそこまで小さいとき最大値が 0 になり，expdelta.cpp は全零行と区別できずスケールを
//   2^0 とし，スライスがすべて 0 になってその行は黙って零として出る（A=2^-127, B=2^127, K=64 の
//   全要素は真値 64 なのに 0 になる．小さい入力でも積への寄与が小さいとは限らない）．
//   そこで上端と同じく範囲外として旗を立てて止める：最大値が非零（上位語・下位語のどちらかが非零．FP64 の
//   非正規化数も含む）で 2^-126 未満なら，BF16 の |x| には現れない印 0xFF80（符号 1・指数 255）を書き，
//   expdelta.cpp が符号ビットで上端の inf（0x7F80）と見分けて別の旗を立てる．C を読み終えたあと
//   expdelta_check（common/ozaki_scale.hpp）が理由を示して止める（使う側はオペランド全体を 2 のべきで
//   掛けてから渡す）．行の一部だけが小さい要素は最大値を決めないので，この制限には触れない．
//   全零の行（列）は 0 のままで，旗は立たない．
//
// 読み込み
//   FP64 タイル（OZ_F64_TILE_BYTES）の作業領域を 2 枚持ち，タイル kt を調べているあいだに kt+1 を読む．
//
// 出力タイルの形：expdelta.cpp が読むのは，A なら各行の 0 列目 nf(k,0)，B なら 0 行目の
// 各列 nf(0,k) だけである（行列エンジンのリダクションが結果を置く位置と同じ）．
// ほかの要素は 0 にしておく．
//
// コンパイル時引数（並びは scale_args.h の ReaderCT）: 入力・出力の TensorAccessor の引数，CEIL（0 切り捨て，1 切り上げ，
//   2 切り上げと 2 のべきの 1 段上げ），MODE（0 = A の行，1 = B の列），作業領域の CB 2 つ（FP64 2 枚，結果の BF16 1 枚）
// 実行時引数（同 ReaderRT）: in_addr, out_addr, nout, Kt, stride, start_o（Kt >= 1．ホストの scale_build が
//   空の行列を止める．Kt=0 だとタイル 0 を先読みしたまま読みの完了を待たずに終わってしまう）
//   入力タイルの番号は MODE 0 が og*Kt+kt，MODE 1 が kt*stride+og（og = start_o + o）．
//   出力タイルの番号は og．

#include "api/dataflow/dataflow_api.h"
#include "l1_order.h"
#include "tile_nf.h"   // (行, 列) → タイル内の通し番号 nf
#include "f64_tf.h"    // OZ_F64_TILE_BYTES
#include "scale_args.h"
#include "scale_bits.h"   // oz_absf64_to_bf16（ホストのモデルと共有）

// 16 要素（間隔 st 語）の |x| の最大値で (mh, ml) を更新する．|x| の上位語（符号ビットを落としたもの）の
// 最大値 hm を先に取り（maxu 1 命令ずつ．分岐が無いので読みが先へ進められる），hm が今の最大値に
// 届いたときだけ，上位語が hm に等しい要素の下位語を見る（ほとんどの行は最初の比較で終わる）．
// 結果は要素ごとに (上位語, 下位語) を辞書順で比べて最大を取ったものと同じである．
inline void upd16(const uint32_t* e, uint32_t st, uint32_t& mh, uint32_t& ml) {
    uint32_t hm = 0;
#pragma GCC unroll 16
    for (uint32_t k = 0; k < 16; ++k) {                // little endian：下位語が先，上位語が後
        const uint32_t h = e[st * k + 1] & 0x7FFFFFFFu;
        hm = (h > hm) ? h : hm;
    }
    if (hm < mh) return;
    uint32_t lm = (hm == mh) ? ml : 0u;
    for (uint32_t k = 0; k < 16; ++k)
        if ((e[st * k + 1] & 0x7FFFFFFFu) == hm) { const uint32_t l = e[st * k]; lm = (l > lm) ? l : lm; }
    mh = hm; ml = lm;
}

// FP64 の |x|（上位語 h・下位語 l）→ BF16 は scale_bits.h の oz_absf64_to_bf16（下端の印 0xFF80 と Inf・NaN の印 0xFFC0 を
// 除けば，要素ごとに BF16 にする方式と同じ式．ホストの OZ_FP64 のモデルも同じ関数を使う）．

void kernel_main() {
    using namespace scale_args;
    const uint32_t in_addr  = get_arg_val<uint32_t>(R_IN_ADDR);
    const uint32_t out_addr = get_arg_val<uint32_t>(R_OUT_ADDR);
    const uint32_t nout     = get_arg_val<uint32_t>(R_NOUT);
    const uint32_t Kt       = get_arg_val<uint32_t>(R_KT);
    const uint32_t stride   = get_arg_val<uint32_t>(R_STRIDE);
    const uint32_t start_o  = get_arg_val<uint32_t>(R_START_O);

    constexpr auto i_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(i_args, in_addr);
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(o_args, out_addr);
    constexpr uint32_t CT0  = o_args.next_compile_time_args_offset();
    constexpr uint32_t CEIL = get_compile_time_arg_val(CT0 + CT_CEIL);
    constexpr uint32_t MODE = get_compile_time_arg_val(CT0 + CT_MODE);
    // 同じカーネルを 2 つの RISC-V に置くので，作業領域の CB はそれぞれ別のものを受け取る
    constexpr uint32_t cb_f = get_compile_time_arg_val(CT0 + CT_CB_F64);   // FP64 タイル 2 枚（OZ_F64_TILE_BYTES ずつ）
    constexpr uint32_t cb_o = get_compile_time_arg_val(CT0 + CT_CB_OUT);   // 結果の BF16 タイル 1 枚
    const uint32_t op = get_write_ptr(cb_o);

    // 作業領域 2 枚（push/pop せず使い回す）
    const uint32_t fb0 = get_write_ptr(cb_f);
    const uint32_t fbuf[2] = {fb0, fb0 + OZ_F64_TILE_BYTES};

    uint32_t mh[32], ml[32];
    for (uint32_t o = 0; o < nout; o++) {
        const uint32_t og = start_o + o;
        for (uint32_t k = 0; k < 32; ++k) { mh[k] = 0u; ml[k] = 0u; }
        auto tile_id = [&](uint32_t kt) { return (MODE == 0) ? (og * Kt + kt) : (kt * stride + og); };
        noc_async_read_tile(tile_id(0), in, fbuf[0]);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_barrier();                        // タイル kt が届いた
            if (kt + 1 < Kt) noc_async_read_tile(tile_id(kt + 1), in, fbuf[(kt + 1) & 1u]);
            // 読みの完了はコンパイラから見えないので，ここより前へ L1 の読みを動かさせない
            asm volatile("" ::: "memory");
            const uint32_t* src = reinterpret_cast<const uint32_t*>(fbuf[kt & 1u]);
            for (uint32_t f = 0; f < 4; ++f) {               // 16x16 の面ごと
                const uint32_t* fp = src + 512u * f;         // 面 1 つは 256 要素 = 512 語
                if constexpr (MODE == 0) {                   // 行 (f/2)*16 + rr の 16 要素
                    const uint32_t rb = (f >> 1) * 16u;
                    for (uint32_t rr = 0; rr < 16; ++rr) {
                        upd16(fp + 32u * rr, 2u, mh[rb + rr], ml[rb + rr]);   // 面の 1 行（連続 16 要素）
                    }
                } else {                                     // 列 (f%2)*16 + cc の 16 要素
                    const uint32_t cb = (f & 1u) * 16u;
                    for (uint32_t cc = 0; cc < 16; ++cc) {
                        upd16(fp + 2u * cc, 32u, mh[cb + cc], ml[cb + cc]);   // 面の 1 列（間隔 16 要素）
                    }
                }
            }
            // 次の反復がこの作業領域へ読み始める前に，上の L1 の読みを終えておく
            asm volatile("" ::: "memory");
        }
        volatile tt_l1_ptr uint32_t* z = (volatile tt_l1_ptr uint32_t*)op;
        for (uint32_t i = 0; i < 512; ++i) z[i] = 0u;        // BF16 1024 要素を 0 に
        volatile tt_l1_ptr uint16_t* dst = (volatile tt_l1_ptr uint16_t*)op;
        for (uint32_t k = 0; k < 32; ++k)
            dst[(MODE == 0) ? nf(k, 0) : nf(0, k)] = oz_absf64_to_bf16<CEIL>(mh[k], ml[k]);
        // 書いた値が L1 に届いてから NoC に読ませる（l1_order.h．最後に書いた語を読み戻す）
        l1_writes_landed((op + 2u * ((MODE == 0) ? nf(31, 0) : nf(0, 31))) & ~3u);
        noc_async_write_tile(og, out, op);
        noc_async_write_barrier();                           // 次の出力で作業タイルを書き換える前に
    }
}
