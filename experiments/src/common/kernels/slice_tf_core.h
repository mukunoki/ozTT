// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// triple-float スライサの共通部分（scheme I・II の 3 つのスライサが使う）
// ---------------------------------------------------------------------------
// どのスライサも「リーダが FP64 から割った 4 枚を受けて triple-float を組み立て，
// 基数倍して丸め，残差を引く」を繰り返す．違うのは基数の掛け方と，切り出した値の
// 符号化・出力先だけなので，入力の受け渡しと残差の持ち回りをここにまとめる．
//   scheme I  INT8 以外の 7 形式 : ozaki1/kernels/slice_tf_num.cpp（本体は slice_tf_body.h）
//   scheme I  INT8               : ozaki1/kernels/slice_tf_sm8.cpp（同）
//   scheme II          : ozaki2/kernels/oz2_digits.h
//
// CB の割り当ては slice_f64_reader.cpp が書き込む先と対になっている（番号は両側とホストが
// slice_f64_args.h の 1 つの表から引く）．
//   c_0 hi   c_4 braw   c_1 craw   c_2 s（scale．下の tf_load_f64 を参照）
//   c_24 残差の limb 0,1   c_25 残差の limb 2
//
// DST の使い方（名前は sfpu_tf_proc.h の namespace dst）：組み立てた triple-float と残差は 3 本組 acc_hi・acc_mid・acc_lo，
// 下位 limb のスケール s は term に一時に置く．
//
// 残差を 2 つの CB に分ける理由：カーネル内で CB を往復させるとき，一度に 3 タイル
// （FP32 かつ dst_full_sync）を渡すと本ハードウェアで非決定的にデッドロックする．
// 2 + 1 に分ければ起きない．3 つのスライサすべてがこの制約を共有するので，
// 往復はここの 2 つの関数だけが行う．

#include "sfpu_tf.h"
#include "f64_tf.h"
#include "slice_f64_args.h"
#include "api/compute/pack.h"

constexpr auto cb_tf_hi    = tt::CBIndex(slice_f64_args::CB_HI);
constexpr auto cb_tf_lo    = tt::CBIndex(slice_f64_args::CB_CRAW);
constexpr auto cb_tf_scale = tt::CBIndex(slice_f64_args::CB_SCALE);  // s = ±2^(elsb+7)（elsb は f64_tf.h の split_f64 の仮数の最下位の指数）
constexpr auto cb_tf_mid   = tt::CBIndex(slice_f64_args::CB_BRAW);
constexpr auto cb_tf_res   = tt::CBIndex(slice_f64_args::CB_RES01);
constexpr auto cb_tf_res2  = tt::CBIndex(slice_f64_args::CB_RES2);

// 入力タイル 1 枚ぶん（hi・braw・craw と s の 4 枚）が揃うのを待つ／使い終えて捨てる
ALWI void tf_wait_input() {
    cb_wait_front(cb_tf_hi, 1);
    cb_wait_front(cb_tf_lo, 1);
    cb_wait_front(cb_tf_mid, 1);
    cb_wait_front(cb_tf_scale, 1);
}
ALWI void tf_pop_input() {
    cb_pop_front(cb_tf_hi, 1);
    cb_pop_front(cb_tf_lo, 1);
    cb_pop_front(cb_tf_mid, 1);
    cb_pop_front(cb_tf_scale, 1);
}

// リーダ（slice_f64_reader.cpp）が FP64 を割って 4 枚のビット列で渡してくるので，
// ここで triple-float に組み立てる（両スキーム共通）：
//     hi  = c_0
//     mid = (c_4 - 12582912.0f) * s          （c_4 = 2^23 + 2^22 + b）
//     lo  = (c_1 -  8388608.0f) * s * 2^-7   （c_1 = 2^23 + c，s = c_2 = ±2^(elsb+7)．elsb は split_f64 の指数）
// 下位 2 本をマジックナンバ形式で受けるのは，リーダ側で先頭ビット位置を探さずに
// 済ませるためである（手続きの正しさは tools/tfconv.cpp が検査している）．
// 正規化（2 のべき）はリーダが指数へ畳んであるので inv の乗算は要らない．
ALWI void tf_load_f64() {
    using namespace dst;
    copy_tile_to_dst_init_short(cb_tf_scale); copy_tile(cb_tf_scale, 0, term);   // s
    copy_tile_to_dst_init_short(cb_tf_mid); copy_tile(cb_tf_mid, 0, acc_mid);   // braw
    copy_tile_to_dst_init_short(cb_tf_lo);  copy_tile(cb_tf_lo, 0, acc_lo);     // craw
    copy_tile_to_dst_init_short(cb_tf_hi);  copy_tile(cb_tf_hi, 0, acc_hi);     // hi
    AddS(acc_mid, -12582912.0f); Mul(acc_mid, term, acc_mid);                    // mid
    AddS(acc_lo,  -8388608.0f); Mul(acc_lo, term, acc_lo); MulS(acc_lo, OZ_TF_LO_SCALE);   // lo
}

// soft-float 版のリーダは 3 本組をそのまま（正規化済みで）寄こすので，写すだけでよい．
ALWI void tf_load_plain() {
    copy_tile_to_dst_init_short(cb_tf_hi);  copy_tile(cb_tf_hi, 0, dst::acc_hi);
    copy_tile_to_dst_init_short(cb_tf_mid); copy_tile(cb_tf_mid, 0, dst::acc_mid);
    copy_tile_to_dst_init_short(cb_tf_lo);  copy_tile(cb_tf_lo, 0, dst::acc_lo);
}

// 残差の持ち回り．DST の内容は acquire/release をまたいで保持されないので，
// 反復のたびに CB へ退避して読み戻す．
ALWI void tf_pop_residual() {
    cb_wait_front(cb_tf_res, 2);
    copy_tile_to_dst_init_short(cb_tf_res);
    copy_tile(cb_tf_res, 0, dst::acc_hi);
    copy_tile(cb_tf_res, 1, dst::acc_mid);
    cb_pop_front(cb_tf_res, 2);
    cb_wait_front(cb_tf_res2, 1);
    copy_tile_to_dst_init_short(cb_tf_res2);
    copy_tile(cb_tf_res2, 0, dst::acc_lo);
    cb_pop_front(cb_tf_res2, 1);
}
ALWI void tf_push_residual() {
    cb_reserve_back(cb_tf_res, 2);
    pack_reconfig_data_format(cb_tf_res);
    pack_tile(dst::acc_hi, cb_tf_res);
    pack_tile(dst::acc_mid, cb_tf_res);
    cb_push_back(cb_tf_res, 2);
    cb_reserve_back(cb_tf_res2, 1);
    pack_reconfig_data_format(cb_tf_res2);
    pack_tile(dst::acc_lo, cb_tf_res2);
    cb_push_back(cb_tf_res2, 1);
}
