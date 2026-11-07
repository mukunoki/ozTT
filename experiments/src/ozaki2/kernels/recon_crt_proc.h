// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：再構成（recon_crt.cpp）の法ごとの手続き（デバイス API に依存しない）
// ---------------------------------------------------------------------------
// recon_crt.cpp の kernel_main は法ごとに「r_i を tmp0 へ読む → k_i = ModRound(r_i*y_i) を keep へ → k_i に 1/m_i の
// 断片を掛けて 3 本組へ足す → 整数部を落とす」を回す．読み込みの後の 3 つをここに置く．sfpu_tf_proc.h と同じく
// タイル演算のラッパの型 W をテンプレートの引数に取り，カーネルはデバイスのラッパ SfpuDev で，ホストの模擬
// （tools/oz_sim.h の crt_term_k・crt_accumulate．tools の crt_fold_check・twosum_probe・aligned_accum が使う）は
// src/checks/sfpu_tf_emu.hpp の Dst で同じものを組む（写しを持つと，カーネルの列を変えても検査が古い列のまま通る）．
// 読み込み（剰余入力の copy_tile，INT8 経路の符号-絶対値の復号）は CB と型変換のタイル演算を使うのでカーネルに残し，
// 模擬は oz_sim.h の crt_load が 1 演算ずつ写す．定数（RAWMASK・NSMALL_FRAG・limb の本数）の決め方はホスト側の
// ozaki2_moduli.h の oz2_recon_ct を，ドライバと模擬が共有する．
#pragma once
#include <cstdint>
#include "../../common/kernels/sfpu_tf_proc.h"

namespace crt_proc {

// tmp0 の r_i から k_i = ModRound(r_i*y_i)（対称剰余．はみ出しは recon_crt.cpp の ModRound の注記）を keep へ置く．
// raw（1 桁の法．recon_args.h の C_RAWMASK）なら r_i は法へ畳む前の GEMM の出力なので，先に法へ畳む
template <class W>
ALWI void term_k(W& w, bool raw, float m, float rinv, const float& yi) {
    using namespace dst;
    if (raw) sfpu_proc::ModRound(w, tmp0, tmp0, tmp1, m, rinv);
    w.MulS(tmp0, yi);                                    // r_i * y_i（|r_i*y_i| < m^2/2）
    sfpu_proc::ModRound(w, tmp0, keep, tmp1, m, rinv);
}

// keep の k_i に1/m_iの断片c.frag(t,i)を掛ける．rp0/rp2の末尾断片は積の丸めを許し，
// ほかの断片の積は無誤差．多語累積全体には最下位語の丸めが残る．3本組（NLIMB=2なら
// 2 本組）へ 1 本ずつ足す．末尾の nsmall 本は第 1 段を飛ばして安く足す（条件と誤差は sfpu_tf_proc.h の tfAddSmall）
template <uint32_t NLIMB, class W, class C>
ALWI void accumulate(W& w, uint32_t nrp, uint32_t nsmall, const C& c, uint32_t i) {
    using namespace dst;
    for (uint32_t t = 0; t < nrp; t++) {
        w.Cpy(keep, term); w.MulS(term, c.frag(t, i));
        if (t >= nrp - nsmall) {
            sfpu_proc::tfAddSmall<NLIMB>(w);
        } else {
            sfpu_proc::tfAddS<NLIMB>(w);
        }
    }
}

// 法ごとに a0 の整数部を落とす（a0 -= round(a0)．理由は recon_crt.cpp の kernel_main）
template <class W>
ALWI void drop_integer(W& w) {
    using namespace dst;
    w.Cpy(acc_hi, term); w.Rnd(term); w.Sub(acc_hi, term, acc_hi);
}

}  // namespace crt_proc
