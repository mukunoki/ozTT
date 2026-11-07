// SPDX-License-Identifier: Apache-2.0
//
// FP64 のリーダ（slice_f64_reader.cpp）とスライサの入力の CB 番号・引数の並び（尾崎スキーム I・II 共通．
// ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// リーダは FP64 のタイルを triple-float の 4 枚へ割って CB へ積み，スライサ（尾崎 I の slice_tf_*.cpp，尾崎 II の
// oz2_digits.h．入口と残差の持ち回りは slice_tf_core.h）がそれを読む．CB 番号・引数の並びを両スキームのホスト
// （ozaki1_slice.hpp の slice_build・slice_run，ozaki2_main.cpp の build_slice）とカーネルが番号で持つと，片側だけを
// 動かしたときに黙って別の CB・別の値を読む（recon_args.h の冒頭と同じ理由）．そこでここに表を置き，両側が名前で使う．
// ホストがリーダの引数を詰める関数は common/slice_f64_host.hpp．CB を作る順は L1 の配置を決め，両スキームで違うので，
// 作る処理そのものは共通にしない（番号だけをここから引く）．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

namespace slice_f64_args {

// --- CB 番号 -------------------------------------------------------------------
// リーダが書き，スライサが読む 4 枚（中位・下位はマジックナンバ形式の生のビット列．slice_tf_core.h の tf_load_f64）
constexpr uint32_t CB_HI    = 0;    // 上位 limb hi
constexpr uint32_t CB_CRAW  = 1;    // 下位 limb の生のビット列 craw（= 2^23 + c）
constexpr uint32_t CB_SCALE = 2;    // 下位 limb のスケール s = ±2^(elsb+7)（elsb は f64_tf.h の split_f64 の下位 limb の指数．slice_tf_core.h）
constexpr uint32_t CB_BRAW  = 4;    // 中位 limb の生のビット列 braw（= 2^23 + 2^22 + b）
// リーダの作業領域（push/pop せず書き込み位置を固定して使い回す）
constexpr uint32_t CB_F64   = 5;    // FP64 タイル 1 枚（f64_tf.h の OZ_F64_TILE_BYTES）
constexpr uint32_t CB_DLT   = 6;    // 指数差分タイル（Int32）
// 郵便受け（SPLIT=1 のときだけ．リーダがタイルの後半を割る仕事をライタへ渡す．slice_f64_split.h）
constexpr uint32_t CB_MBOX  = 7;
constexpr uint32_t MBOX_PAGE = 32;  // 1 ページの大きさ［B］．slice_f64_split.h の SliceJob と同じ（同ファイルの static_assert）
// スライサが残差の 3 本組を持ち回る CB（2 + 1 に分ける理由は slice_tf_core.h）
constexpr uint32_t CB_RES01 = 24;   // 残差の limb 0,1
constexpr uint32_t CB_RES2  = 25;   // 残差の limb 2

// --- リーダのコンパイル時引数（TensorAccessorArgs（FP64・指数差分の順）の後ろ）----------------
enum ReaderCT : uint32_t {
    CT_SF = 0,       // 1 なら soft-float 版（測定用．ホスト側 OZ_SOFTFLOAT=1）
    CT_SPLIT,        // 1 なら後半 512 要素を同じコアのライタに割らせる（slice_f64_split.h．尾崎 I だけ）
    CT_COUNT
};

// --- リーダの実行時引数 ----------------------------------------------------------
enum ReaderRT : uint32_t {
    R_F64_ADDR = 0,  // FP64 のオペランドの先頭アドレス
    R_DLT_ADDR,      // 指数差分（行ごと／列ごとのブロードキャストタイル）の先頭アドレス
    R_NUM,           // このコアが担当する入力タイルの枚数
    R_START,         // 同，先頭のタイル番号
    R_DLT_STRIDE,    // 指数差分の引き当て：mode 0（A）はタイル番号をこれで割り，mode 1（B）はこれで割った余り
    R_MODE,          // 0 = A（行ごと），1 = B（列ごと）
    R_COUNT
};

}  // namespace slice_f64_args
