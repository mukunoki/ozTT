// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：再構成の出口の 1 要素の組み立て（データ移動カーネルの RISC-V の整数演算）
// ---------------------------------------------------------------------------
// 1 タイル 1024 要素のうち前半をリーダ（recon_reader.cpp）が，後半をライタ（recon_writer.cpp）が組み立てる
// （分担の理由は recon_reader.cpp の冒頭）．両者が同じ要素の手続きをこの 1 つから呼ぶ：片方だけを直すと同じ C の中で
// 前半と後半の丸めが黙って食い違い，検証の 53 bit の判定では捕まらない大きさでもありうる．
// L1 のポインタの持ち方（ライタは volatile で読む）と受け渡しの順序（セマフォ・公開）は各カーネルに残し，ここはポインタの
// 型をテンプレートの引数に取る．組み立ての手続き（a128_add_limbs・a128_frac_to_f64・assemble_frac_f64）は
// common/kernels/f64_tf.h（tools/tfconv.cpp が検査する）．
#pragma once
#include <cstdint>
#include "../../common/kernels/f64_tf.h"
#include "recon_args.h"

namespace recon_elem {

// 2 パス構成の第 1 パス（ROLE_TWO_FIRST）：要素 j の limb 3 本（2 パス構成は全パスで 3 本．ozaki2_moduli.h の
// oz2_recon_limb3）を 128 bit 固定小数へ足して作業領域 qa の 4 語（w[0] が最下位）へ置く．PROBE は組み立てを外す調査用
// （答えは壊れる．出力は 0）
template <uint32_t PROBE, class LP, class QP>
inline __attribute__((always_inline)) void pack_a128(LP l0, LP l1, LP l2, QP qa, uint32_t j) {
    const uint32_t lb[3] = {l0[j], l1[j], l2[j]};
    A128 a{{0, 0, 0, 0}};
    if constexpr (PROBE == 0) a128_add_limbs(a, lb, 3u);
    else                      { (void)lb; }
    qa[4 * j] = a.w[0]; qa[4 * j + 1] = a.w[1]; qa[4 * j + 2] = a.w[2]; qa[4 * j + 3] = a.w[3];
}

// FP64 を組み立てる最終パス：要素 j の limb（nlimb 本．2 本組なら 3 本目は 0）から FP64 を組み立てて po の 2 語
// （下位語が先）へ置く．ROLE が ROLE_TWO_FINAL なら，第 1 パスの 128 bit 固定小数（qa の 4 語）に自分の 3 本を足して畳む．
// 差分タイル da・db には sh-e が入っているので，掛け戻す指数は 2sh-(dA+dB)（sh2 は 2sh）．PROBE は出力に定数を詰める
template <uint32_t ROLE, uint32_t PROBE, class LP, class DP, class QP, class OP>
inline __attribute__((always_inline)) void assemble_f64(LP l0, LP l1, LP l2, uint32_t nlimb, DP da, DP db, QP qa, OP po,
                                                         uint32_t j, int32_t sh2, uint32_t Whi, uint32_t Wlo, int32_t qw) {
    uint32_t lb[3];
    lb[0] = l0[j]; lb[1] = l1[j]; lb[2] = (nlimb >= 3) ? l2[j] : 0u;
    const int32_t gg = sh2 - (da[j] + db[j]);
    uint64_t u = 0;
    if constexpr (PROBE == 0) {
        if constexpr (ROLE == recon_args::ROLE_TWO_FINAL) {
            A128 a{{qa[4 * j], qa[4 * j + 1], qa[4 * j + 2], qa[4 * j + 3]}};
            a128_add_limbs(a, lb, 3u);
            u = a128_frac_to_f64(a, gg, Whi, Wlo, qw);
        } else {
            u = assemble_frac_f64(lb, 3u, gg, Whi, Wlo, qw);
        }
    } else { (void)lb; (void)gg; }
    po[2 * j]     = (uint32_t)u;
    po[2 * j + 1] = (uint32_t)(u >> 32);
}

}  // namespace recon_elem
