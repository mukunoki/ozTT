// SPDX-License-Identifier: Apache-2.0
#pragma once
// 両尾崎スキームで共通の形式の選択と MathFidelity の決め方
// ---------------------------------------------------------------------------
// ターゲットはマクロ 1 つ -DOZ_FMT_<形式>（INT8 / FP8 / BF16 / FP16 / FP32 / BFP8 / BFP4 / BFP2）
// で形式を選ぶ．scheme I・II とも同じマクロで，Makefile の DEF_ozaki1_* と DEF_ozaki2_* が
// 1 つずつ渡す．形式に依存する値はすべて表から引き，値を書くのは表の 1 か所だけである（表の分け方は
// oz_format_host.h の冒頭．形式そのものの性質 oz_format_host.h・oz_format_table.hpp，scheme I の
// ozaki1/ozaki1_param.h，scheme II の ozaki2/ozaki2_split.h の OZ2_ROUTES）．ここは選んだ形式 OZ_FMT_ID と
// その性質 OZ_FMT を置く．前処理の段階で要る分岐（ozaki2_main.cpp の #ifdef OZ2_DIGITK）だけがマクロとして残り，
// それも OZ_FMT_* から導く．形式を 1 つに決めない探針・検査は oz_format_table.hpp を取り込む．
#include "common/oz_format_table.hpp"
#include "common/oz_fidelity.hpp"   // OZ_FIDELITY の上書き（oz_fidelity_env）・表示名（形式に依らない部分）
#include <cstdint>

#if (defined(OZ_FMT_INT8) + defined(OZ_FMT_FP8) + defined(OZ_FMT_BF16) + defined(OZ_FMT_FP16) + \
     defined(OZ_FMT_FP32) + defined(OZ_FMT_BFP8) + defined(OZ_FMT_BFP4) + defined(OZ_FMT_BFP2)) != 1
#  error "OZ_FMT_* をちょうど 1 つ指定すること（OZ_FMT_BF16 など．Makefile の DEF_ozaki{1,2}_*）"
#endif

#if   defined(OZ_FMT_INT8)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::INT8;
#elif defined(OZ_FMT_FP8)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::FP8;
#elif defined(OZ_FMT_BF16)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BF16;
#elif defined(OZ_FMT_FP16)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::FP16;
#elif defined(OZ_FMT_FP32)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::FP32;
#elif defined(OZ_FMT_BFP8)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BFP8;
#elif defined(OZ_FMT_BFP4)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BFP4;
#elif defined(OZ_FMT_BFP2)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BFP2;
#endif

inline constexpr OzFormat OZ_FMT = oz_format(OZ_FMT_ID);

// MathFidelity の決め方（両スキーム共通）
// ---------------------------------------------------------------------------
// 規則は「乗算器が実際に受け取る桁の上限」だけで決まる：
//   INT8               HiFi4．INT8 経路は乗算器が SrcA の上位区画に bit 7--5 の 3 bit しか
//                      割り当てないので，桁を小さくしても LoFi では A_lo が残る
//                      （docs/int8_fidelity_mechanism.md）．FP 経路の判定を流用してはならない．
//   それ以外で桁 <= 32  LoFi．LoFi の乗算器は SrcA の 5 bit を読むので，有効ビットが 5 以下の
//                      整数（0..31 と 32）までが無誤差に掛かる（docs/fidelity_radix_sweep.md）．
//   それ以外           HiFi4（HiFi2 で足りる場合の扱いは ozaki1_format.hpp の ozaki_check_exact）．
// 桁の上限は scheme I では基数から（ozaki_digit_max），scheme II では表の digit_bound から来る．
// 既定の構成では INT8 の 2 経路だけが HiFi4，残る 14 経路が LoFi になる．
// 規則の本体は形式を引数に取る oz_fidelity_rule_for（oz_format_table.hpp．形式を 1 つに決めない検査 gemm_exact_check も
// 同じものを呼ぶ）で，ここは選んだ形式 OZ_FMT_ID を渡す．
inline tt::tt_metal::MathFidelity oz_fidelity_rule(uint32_t digit_bound) { return oz_fidelity_rule_for(OZ_FMT_ID, digit_bound); }

// OZ_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4 は上の規則を上書きする（oz_fidelity.hpp の oz_fidelity_env．表示名 oz_fidelity_name も
// 同じヘッダ）．両スキームが使う MathFidelity はこれ（上書きがあればそれ，無ければ桁の上限 digit_bound からの規則）
inline tt::tt_metal::MathFidelity oz_fidelity_for(uint32_t digit_bound) {
    return oz_fidelity_env().value_or(oz_fidelity_rule(digit_bound));
}
