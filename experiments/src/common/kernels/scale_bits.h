// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// スケール段の正規化のビット手続き（尾崎スキーム I・II 共通．デバイス API に依存しない）
// ---------------------------------------------------------------------------
// 行・列ごとの最大絶対値（FP64）を BF16 にし（scale_f64_reader.cpp），その BF16 から正規化の指数 e = ceil(log2(.)) と
// 範囲外の旗を作る（expdelta.cpp）．スキーム I の OZ_FP64 のモデル（ozaki1/ozaki1_autop.hpp の oz1_dev_exp）は，デバイスが
// 選ぶ正規化をビット単位で写す必要があるので，カーネルと同じこの関数をホストでも呼ぶ（写しを持つと，カーネルだけ
// 直したときに p の選び方が黙ってずれる）．検査の src/checks/scale_check.cpp は独立な参照として自前の式を持つ．
// <cstdint> だけに依存する．

#include <cstdint>

// FP64 の |x|（上位語 h は符号ビットを落としたもの，下位語 l）→ BF16 のビット列．CEIL は丸めの向き：
//   0 切り捨て，1 切り上げ（落とす仮数ビットが残っていれば 1 つ上へ），2 切り上げに加えて，ちょうど 2 のべき
//   （仮数の上位 7 bit も落とすビットも 0）のときも 1 つ上へ（仮数が 1 になるだけで指数は変わらないので，2^127 でも
//   inf にはならない．scheme I の BFP2_b．ozaki1_main.cpp の tau_above）．繰り上がりは指数へ自然に伝わる．
// 範囲外は BF16 の |x| には現れない印で返す（どれも指数 255．符号と仮数で見分ける．scale_f64_reader.cpp の
// 「受け付ける大きさの範囲」）：
//   0xFFC0  Inf・NaN の要素がある（FP64 の指数 2047）
//   0xFF80  非零の最大値が 2^-126 を下回る（BF16 の正規化数で表せない．FP64 の非正規化数も含む）
//   0x7F80  BF16 の範囲を超える（2^128 以上．CEIL を立てると BF16 の最大有限値を超えた時点で切り上げが作る）
// 0 は 0 のまま（全零の行・列）．
template <uint32_t CEIL>
inline uint16_t oz_absf64_to_bf16(uint32_t h, uint32_t l) {
    const int32_t E  = (int32_t)((h >> 20) & 0x7FFu);
    if (E == 0x7FF) return 0xFFC0u;                    // Inf・NaN
    const int32_t e8 = E - 896;                        // FP32 の指数へ
    if (e8 <= 0)   return (h | l) ? 0xFF80u : 0u;     // 2^-126 未満（非零）と 0
    if (e8 >= 255) return 0x7F80u;                     // 上限超え -> inf
    uint32_t v = ((uint32_t)e8 << 7) | ((h >> 13) & 0x7Fu);
    if constexpr (CEIL == 1) { if ((h & 0x1FFFu) || l) v += 1u; }
    if constexpr (CEIL == 2) { if ((h & 0x1FFFu) || l || !(v & 0x7Fu)) v += 1u; }
    return (uint16_t)v;
}

// 最大値の BF16 u から正規化の指数 e = ceil(log2(u)) を返し，範囲外なら旗を bad に足す（範囲内なら bad に触らない）．
// BF16 は符号 1・指数 8・仮数 7 で値が (-1)^s * 1.m * 2^(E-127) なので（最大値は |x| なので s=0），
//     ceil(log2(u)) = (E-127) + (m != 0 ? 1 : 0)
// である．E=0（0・非正規化数）は u<=0 とみなして e=0 とする（全零の行・列．旗は立てない）．E=255 は上の印で，
// e=0 とし旗を足す：1 は上端の inf（0x7F80），2 は下端の印（0xFF80），4 は Inf・NaN の印（0xFFC0）．
// 旗を範囲外の枝でだけ足す形と常に展開する指定は，expdelta.cpp がこの関数を使う前の機械語と同じになるようにしてある．
__attribute__((always_inline)) inline int32_t oz_bf16_scale_exp(uint16_t u, uint32_t& bad) {
    const uint32_t E = (u >> 7) & 0xFFu, m = u & 0x7Fu;
    int32_t e;
    if (E == 0u)            e = 0;
    else if (E == 0xFFu) { e = 0; bad |= !(u & 0x8000u) ? 1u : m ? 4u : 2u; }
    else                    e = (int32_t)E - 127 + (m ? 1 : 0);
    return e;
}
