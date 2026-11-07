// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 尾崎スキーム II（FP8）：疎な e4m3 桁の方式 (E) SQS の桁の手直し（デバイス側，SFPI）
// ---------------------------------------------------------------------------
// 役割：方式 (A) と同じ最近接丸めで作った平方数 m = q^2（q は奇数で 33 < q <= 49）の平衡桁
//           h0 = round(r/q),  t = r - q*h0          （|t| <= (q-1)/2 <= 24，|h0| <= (q+1)/2 <= 25）
//       を，e4m3 が無誤差に表せる整数の集合 S = {|x| <= 16} ∪ {偶数 x : 16 < |x| <= 32} へ移す．
//         t が奇数で |t| >= 17 なら  l = t - sgn(t)*q,  h1 = h0 + sgn(t)   （r = l + q*h1 のまま）
//         h1 が奇数で |h1| >= 17 なら h = h1 - sgn(h1)*q                   （q^2 ≡ 0 なので q*h ≡ q*h1）
//       q は奇数なので，奇数の t（h1）から q を引いた値は偶数で，絶対値は q - |t| <= q - 17 <= 32．
//       l, h ∈ S と r ≡ l + q*h (mod q^2) の証明と全法・全剰余の総当たりは ozaki2_split.h の (E) と
//       tools/modround_check.cpp（ホストの写し oz2_sqs_split と，ここと同じ FP32 の手順の両方で見る）．
//       arXiv 2608.06812（Hayashi–Mukunoki–Hoshino–Katagiri）の「無誤差に表せる非連続な整数集合で剰余を覆う」
//       考え方を e4m3 に当てはめたもの（docs/fp8_sparse_digits.md）．
//
// なぜ SFPI で書くか
//   同じ手直しを本リポジトリのタイル演算（sfpu_tf.h の Cpy・MulS・Rnd など．1 演算がタイル 1 枚の
//   読み書き 1 往復）で書くと，偶奇と符号の判定に 26 演算かかる（法 1 本あたりのスライス段が 25 演算から
//   51 演算になり，14080^3 で法 6 本ぶん約 +90 ms の見積り）．SFPI なら 1 回の読み書きのあいだに
//   条件つきの加減算で済む．SFPI の書き方は tt-metal の programming_examples/custom_sfpi_* と同じ．
//
// 数値の扱い（どれも無誤差）
//   l, h は |.| <= 26 の整数値の FP32．|x| + 2^23 は整数で 2^24 未満なので FP32 で無誤差に表せ，
//   そのビット列の最下位ビットが |x| の偶奇になる（仮数の 1 の位）．比較 |x| > 16.5 と q・1 の加減算も
//   整数値どうしで無誤差．
//
// レジスタ：DST の reg dl（l）と reg dh（h）をその場で書き換える．他のレジスタは読まない．
// 前提：fp32_dest_acc_en（DST が FP32）．呼ぶ前に何かの SFPU の二項演算の init が済んでいること
//       （Oz2SqsFix が add_binary_tile_init を呼ぶ）．
// 使う計算 API（ALWI・MATH・add_binary_tile_init・_llk_math_eltwise_binary_sfpu_params_・VectorMode と SFPI）は SFPU の二項演算の
// ヘッダから取り込む（取り込む側の modslice_digit.cpp は oz2_digits.h を通して先に同じヘッダを取り込んでいる）．
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_MATH
// 面 1 枚（SFPI の 32 レーン × 8 行）ぶん．_llk_math_eltwise_binary_sfpu_params_ が面ごとに 4 回呼ぶ．
// 引数の 3 つめ（出力の番号）は使わない（l と h をその場で直す）．
inline void oz2_sqs_fix_face(const uint32_t dl, const uint32_t dh, const uint32_t /*unused*/, const float q) {
    constexpr uint32_t T = 32;   // DST のタイル 1 枚は SFPI の行で 32 行（ckernel_sfpu_lerp.h と同じ）
    const sfpi::vFloat vq = q;
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat l = sfpi::dst_reg[dl * T];
        sfpi::vFloat h = sfpi::dst_reg[dh * T];
        // l の手直し：奇数で |l| >= 17 なら l -= sgn(l)*q, h += sgn(l)
        sfpi::vFloat al = sfpi::abs(l);
        sfpi::vInt pl = sfpi::reinterpret<sfpi::vInt>(al + 8388608.0f) & 1;
        v_if (al > 16.5f && pl != 0) {
            v_if (l < 0.0f) { l = l + vq; h = h - 1.0f; }
            v_else          { l = l - vq; h = h + 1.0f; }
            v_endif;
        }
        v_endif;
        // h の手直し：奇数で |h| >= 17 なら h -= sgn(h)*q（q^2 ≡ 0 なので剰余類は変わらない）
        sfpi::vFloat ah = sfpi::abs(h);
        sfpi::vInt ph = sfpi::reinterpret<sfpi::vInt>(ah + 8388608.0f) & 1;
        v_if (ah > 16.5f && ph != 0) {
            v_if (h < 0.0f) { h = h + vq; }
            v_else          { h = h - vq; }
            v_endif;
        }
        v_endif;
        sfpi::dst_reg[dl * T] = l;
        sfpi::dst_reg[dh * T] = h;
        sfpi::dst_reg++;
    }
}
#endif

// l（reg dl）と h（reg dh）を方式 (E) の桁へ直す．q は法の平方根（奇数）．
ALWI void Oz2SqsFix(uint32_t dl, uint32_t dh, float q) {
    add_binary_tile_init();   // SFPU の二項演算の設定（アドレスの進め方）．足し算そのものはしない
    MATH(_llk_math_eltwise_binary_sfpu_params_(oz2_sqs_fix_face, dl, dh, dh, VectorMode::RC, q));
}
