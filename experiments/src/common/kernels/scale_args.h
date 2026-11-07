// SPDX-License-Identifier: Apache-2.0
//
// スケール段の引数の並びと CB 番号（尾崎スキーム I・II 共通．ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// 最大値を求める scale_f64_reader.cpp（ホストは ozaki_scale.hpp の scale_build）と，指数差分のタイルを作る
// expdelta.cpp（同 expdelta_build）の引数．番号だけで対応づけると，片側に 1 つ足したときに以降が黙ってずれる
// （ozaki2/kernels/recon_args.h の冒頭と同じ理由）ので，ホストはこの表の名前で添字を指定して詰め，カーネルは名前で読む．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

// --- 最大値のリーダ（scale_f64_reader.cpp）----------------------------------------------
// 同じカーネルを 2 つのデータ移動 RISC-V に置き，コアの担当する出力タイルを半分ずつ受け持たせる．作業領域の CB は
// RISC-V ごとに別のものをコンパイル時引数で受け取る
namespace scale_args {

// CB 番号（RISCV_1 と RISCV_0 で別々）
constexpr uint32_t CB_F64_R1 = 3;    // RISCV_1 の FP64 タイル 2 枚（1 枚を調べるあいだに次を読む）
constexpr uint32_t CB_F64_R0 = 4;    // RISCV_0 の同じもの
constexpr uint32_t CB_OUT_R1 = 16;   // RISCV_1 の結果の BF16 タイル 1 枚
constexpr uint32_t CB_OUT_R0 = 17;   // RISCV_0 の同じもの

// コンパイル時引数（TensorAccessorArgs（入力・出力の順）の後ろ）
enum ReaderCT : uint32_t {
    CT_CEIL = 0,     // 最大値の BF16 化：0 切り捨て，1 切り上げ，2 切り上げと 2 のべきの 1 段上げ
    CT_MODE,         // 0 = A の行，1 = B の列
    CT_CB_F64,       // 作業領域の CB（FP64 2 枚）
    CT_CB_OUT,       // 結果の BF16 タイルの CB
    CT_COUNT
};
// 実行時引数（入力タイルの番号は MODE 0 が og*Kt+kt，MODE 1 が kt*stride+og．og = start_o + o）
enum ReaderRT : uint32_t {
    R_IN_ADDR = 0,   // FP64 のオペランドの先頭アドレス
    R_OUT_ADDR,      // 最大値（BF16 タイル）の先頭アドレス
    R_NOUT,          // このカーネルが受け持つ出力タイルの枚数
    R_KT,            // 出力 1 枚あたりに畳む入力タイルの枚数（1 以上）
    R_STRIDE,        // MODE 1 の入力タイルの刻み
    R_START_O,       // 受け持つ出力タイルの先頭の番号
    R_COUNT
};

}  // namespace scale_args

// --- 指数差分（expdelta.cpp）---------------------------------------------------------
namespace expdelta_args {

constexpr uint32_t CB_MAX = 0;   // 最大値タイル（BF16）
constexpr uint32_t CB_OUT = 1;   // 出力タイル（Int32）

// コンパイル時引数（この後ろに TensorAccessorArgs（最大値・出力・旗の順）が続く）
enum CT : uint32_t {
    CT_ROWWISE = 0,  // 行方向（A）なら 1，列方向（B）なら 0
    CT_COUNT
};
// 実行時引数
enum RT : uint32_t {
    R_MAX_ADDR = 0,  // 最大値（BF16 タイル）の先頭アドレス
    R_OUT_ADDR,      // 指数差分の出力の先頭アドレス
    R_FLAG_ADDR,     // 範囲外の旗の先頭アドレス
    R_NUM,           // このコアが担当するタイルの枚数
    R_START,         // 同，先頭のタイル番号
    R_SHIFT,         // 正規化の指数のずらし（int32 をそのまま uint32 で運ぶ）
    R_SLOT,          // このコア専用の旗のスロット
    R_COUNT
};

}  // namespace expdelta_args
