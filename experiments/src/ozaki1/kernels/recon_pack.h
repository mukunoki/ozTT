// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 尾崎スキーム I：再構成の計算カーネル 2 本（recon_tf_f32in.cpp・recon_tf_i32in.cpp）が共有する出口
// ---------------------------------------------------------------------------
// Horner で畳んだアキュムレータの 3 本組（DST の dst::acc_hi・acc_mid・acc_lo）を，ライタ recon_f64_writer.cpp が
// 読む 3 つの CB へ出す．CB の番号は hi・lo・mid の順（oz1_args.h の冒頭）で DST の並びと違うので，取り違えないよう
// 出し方をここの 1 か所に置く（片方だけ直すと，入力の形式によって limb の並びが黙って食い違うため）．

#include "api/compute/pack.h"
#include "../../common/kernels/sfpu_tf_proc.h"   // DST の名前（namespace dst）
#include "oz1_args.h"

ALWI void recon_pack_triple() {
    constexpr auto cb_hi  = tt::CBIndex(oz1_args::CB_LIMB_HI);
    constexpr auto cb_lo  = tt::CBIndex(oz1_args::CB_LIMB_LO);
    constexpr auto cb_mid = tt::CBIndex(oz1_args::CB_LIMB_MID);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_hi, 1);
    cb_reserve_back(cb_mid, 1);
    cb_reserve_back(cb_lo, 1);
    pack_reconfig_data_format(cb_hi);  pack_tile(dst::acc_hi, cb_hi);
    pack_reconfig_data_format(cb_mid); pack_tile(dst::acc_mid, cb_mid);
    pack_reconfig_data_format(cb_lo);  pack_tile(dst::acc_lo, cb_lo);
    cb_push_back(cb_hi, 1);
    cb_push_back(cb_mid, 1);
    cb_push_back(cb_lo, 1);
    tile_regs_release();
}
