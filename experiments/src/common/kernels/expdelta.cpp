// SPDX-License-Identifier: Apache-2.0
//
// スケールの仕上げ：最大値タイルから「指数へ足す差分」タイルを作る
// ---------------------------------------------------------------------------
// 役割：行ごと／列ごとのリダクションが出した最大値 mx（BF16）から
//         e = ceil(log2(mx))   ただし mx <= 0 のときは e = 0
//         d = shift - e
//       を求め，タイル全体へブロードキャストした Int32 タイルを書く．
//       スライサはこの d を FP64 の指数フィールドへ整数で足す．
//       mx は scale_f64_reader.cpp が FP64 の最大値を BF16 にした値で，丸めの向き（切り捨て・切り上げ・2 のべきの
//       1 段上げ．同ファイルの CEIL）はそちらで決まる．ここは mx の BF16 を見るだけである．
//
// 本実装は A・B を転送してから C が戻るまでホストと通信しないという仕様なので，
// この仕上げ（最大値から ceil(log2(.)) を取り，タイルへ展開する）もホストへ
// 読み戻さずにデバイス上で行う．
//
// log2 は要らない：BF16 の指数と仮数から ceil(log2(mx)) を整数で求める（導き方と 0・非正規化数の扱いは scale_bits.h の
// oz_bf16_scale_exp．ホストの OZ_FP64 のモデルも同じ関数を使う）．
// E=255 は範囲外の印で，後段が黙って誤答を返さないよう旗を立てる．符号ビットと仮数で 3 通りを見分ける：
// 0x7F80（inf）は最大値が BF16 の範囲（約 2^128）を超えた（旗のビット 0），0xFF80 は非零の最大値が
// 2^-126 を下回った（旗のビット 1），0xFFC0 は行・列に Inf・NaN の要素がある（旗のビット 2）
// （scale_bits.h の oz_absf64_to_bf16．|x| の BF16 は符号が 0 なので，後の 2 つは通常の値とは重ならない）．
// 旗の語はビットの OR で，ホストの expdelta_check が理由を分けて表示する．
//
// 対象は M+N 個の値しかないので SFPU は使わず，RISC-V 上で直に作る．
//
// コンパイル時引数・実行時引数・CB 番号の並びは scale_args.h の expdelta_args（行方向か，最大値・出力・旗のバッファ，
// 担当タイル数・先頭タイル・shift・旗スロット（コア番号））
#include "api/dataflow/dataflow_api.h"
#include "l1_order.h"
#include "tile_nf.h"   // (行, 列) → タイル内の通し番号 nf
#include "scale_args.h"
#include "scale_bits.h"   // oz_bf16_scale_exp（ホストのモデルと共有）
using namespace expdelta_args;

constexpr uint32_t ROWWISE = get_compile_time_arg_val(CT_ROWWISE);

void kernel_main() {
    const uint32_t max_addr = get_arg_val<uint32_t>(R_MAX_ADDR);
    const uint32_t out_addr = get_arg_val<uint32_t>(R_OUT_ADDR);
    const uint32_t flg_addr = get_arg_val<uint32_t>(R_FLAG_ADDR);
    const uint32_t num      = get_arg_val<uint32_t>(R_NUM);
    const uint32_t start    = get_arg_val<uint32_t>(R_START);
    const int32_t  shift    = (int32_t)get_arg_val<uint32_t>(R_SHIFT);
    const uint32_t slot     = get_arg_val<uint32_t>(R_SLOT);   // このコア専用の旗スロット

    constexpr uint32_t cb_in = CB_MAX;    // 最大値タイル（BF16, 2048 B）
    constexpr uint32_t cb_ot = CB_OUT;    // 出力タイル（Int32, 4096 B）
    constexpr auto m_args = TensorAccessorArgs<CT_COUNT>();
    const auto am = TensorAccessor(m_args, max_addr);
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto ao = TensorAccessor(o_args, out_addr);
    constexpr auto f_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto af = TensorAccessor(f_args, flg_addr);

    const uint32_t ip = get_write_ptr(cb_in), op = get_write_ptr(cb_ot);
    volatile tt_l1_ptr uint16_t* mx = (volatile tt_l1_ptr uint16_t*)ip;
    volatile tt_l1_ptr int32_t*  dt = (volatile tt_l1_ptr int32_t*)op;
    uint32_t bad = 0;

    for (uint32_t t = start; t < start + num; ++t) {
        noc_async_read_tile(t, am, ip);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 32; ++k) {
            // 範囲外なら旗（上端 1・下端 2・Inf/NaN 4）を bad に足し，e = 0
            const int32_t e = oz_bf16_scale_exp(mx[ROWWISE ? nf(k, 0) : nf(0, k)], bad);
            const int32_t d = shift - e;
            for (uint32_t j = 0; j < 32; ++j)
                dt[ROWWISE ? nf(k, j) : nf(j, k)] = d;
        }
        l1_writes_landed(op + 4u * nf(31, 31));   // 最後に書いた要素．L1 に届いてから NoC に読ませる（l1_order.h）
        noc_async_write_tile(t, ao, op);
        noc_async_write_barrier();
    }
    // 旗はコアごとに専用のタイルへ立てる．ホストは C を読んだあとで全スロットを
    // 見る（窓の外）．同じタイルを複数コアが書かないので，重複 NoC 書き込みの
    // 振る舞いに依存しない．
    if (bad) {
        dt[0] = (int32_t)bad;
        l1_writes_landed(op);
        noc_async_write_tile(slot, af, op);
        noc_async_write_barrier();
    }
}
