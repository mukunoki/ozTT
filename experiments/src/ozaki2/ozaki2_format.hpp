// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎スキーム II：形式ごとの値（経路 OZ2_PARAM）・桁分割の有無・MathFidelity（ホスト側の判定だけで，デバイスを触らない）
// ---------------------------------------------------------------------------
// スライス／GEMM のオペランド形式は 8 通りで，scheme I と同じマクロ 1 つ -DOZ_FMT_<形式> で選ぶ
// （Makefile の DEF_ozaki2_*）．形式そのものの性質（OZ_FMT）と選択は oz_format.hpp，scheme II に
// 固有の値は経路の表（ozaki2_split.h の OZ2_ROUTES．ここでは OZ2_PARAM）にある．scheme I の ozaki1_format.hpp に当たる．
// ozaki2 のほかのヘッダ（計画 ozaki2_plan.hpp・段の構築 ozaki2_stages.hpp）はどれもこれを取り込む．
// ヘッダの中では名前空間を開かない（tt-metal の名前は tt:: で修飾する）．
//
// INT8 は剰余をそのまま符号-絶対値の UInt8 で運ぶので法は 255 まで．
//
// 残る 7 形式（FP8 / BF16 / FP16 / FP32 / BFP8 / BFP4 / BFP2）は桁分割経路で，
// 対称剰余を複数の桁に分けて運ぶ（ozaki2/kernels/modslice_digit.cpp．GEMMul8 の FP8 経路に対応）．
// 形式が無誤差に表せる整数の上限（FP8 なら 16）に縛られず，桁の上限 d に対して (2d+1)^2 程度
// までの法が使える．形式ごとに違うのは桁の上限 digit_bound と，そこから決まる法の上限・
// 本数だけである．BF16・FP16・FP32・BFP8_b は桁の上限が形式ではなく累積幅で決まる（K=14080
// で d <= 32）．BF16・FP16・BFP8_b は形式別の時間モデルで法を選び，評価した3サイズでは同じ法になる．
// FP32（TF32）だけは表の選び方（ozaki2_moduli.h の awe=0）で別の法になる．
// 2 桁での分割の方式は法によって 3 通りある（ozaki2_split.h の OZ2_SPLIT_* を参照）．
//   SQ   平方数 m=q^2 : h=round(r/q), l=r-q*h の 2 枚．q^2 ≡ 0 (mod m) で上位×上位が
//                       消え，積 3 本で D = q*(HL+LH) + LL（法テーブル先頭の 6 本）．
//   KAR  非平方数     : 基数 16 の Karatsuba．h=sign(r)*ceil(|r|/16), l=r-16*h, s=h+l の
//                       3 枚で，積 3 本で D = 256*HH + 16*(SS-HH-LL) + LL．
//   WIDE 513 超の非平方数 : 平衡桁 2 枚に戻し，積 4 本で D = q^2*HH + q*(HL+LH) + LL．
//   SQS  FP8 だけ．奇数 q <= 49 の平方数 : SQ と同じ桁を e4m3 が無誤差に表せる疎な整数
//                       （|x| <= 16 か 32 以下の偶数）へ手直しした 2 枚．積と合成は SQ と同じ
//                       （ozaki2_split.h の (E)，ozaki2/kernels/oz2_sparse_fix.h，docs/fp8_sparse_digits.md）．
//
// 剰余を単一の FP8 値で運ぶ「素朴な FP8」経路は存在しない：それには法が 33 以下で
// なければならず，窓を覆うには本数が足りない（法 1 本 5 bit で 24 本でも 120 bit）．
// 桁 32 の AWE 経路（BF16・FP16・BFP8_b）は m <= 65 の小さい法だけを 1 桁（剰余そのもの）で運び，
// 大きい法と混ぜて使う（ozaki2_split.h の (F)）．
//
// 法選択へ渡す acc_bits は OZ2_ACC_BITS（ozaki2_moduli.h．INT8 経路の C_i を再構成が無誤差に
// 読める 31 bit）．GEMM 段の INT8 の部分和はその範囲でパッカの Int32 の L1 累積を通る
// （2^31 の手前まで無誤差であることは scripts/diagnose/gemm_exact.sh の int8 の項目で確かめる）．
// 幅bには既定値・利用者の指定・入力からの自動選択があり，法の積で実現可能なCRTの窓にも制限される．
#include <algorithm>
#include <cstdint>
#include "common/oz_format.hpp"
#include "ozaki2/ozaki2_moduli.h"

// 形式ごとの値（経路）．表そのもの（Oz2Route と，形式から経路への対応 OZ2_ROUTES・oz2_param．値の理由も）は
// ozaki2_split.h にあり，ホスト単体の検査・模擬（tools/）と共有する．
using Oz2Param = Oz2Route;
inline constexpr Oz2Param OZ2_PARAM = oz2_param(OZ_FMT_ID);

// 桁分割の有無は関数や構造体の成員ごと出し分けるので，前処理の段階でも要る．
// 選択マクロから導き，表の digit_bound と食い違わないことを確かめる．
#if !defined(OZ_FMT_INT8)
#define OZ2_DIGITK 1
#endif
#ifdef OZ2_DIGITK
static_assert(OZ2_PARAM.digit_bound != 0, "桁分割経路には桁の上限が要る");
#else
static_assert(OZ2_PARAM.digit_bound == 0, "INT8 経路は桁分割しない");
#endif
// 値や分岐の選択はこちらで書く（if constexpr・三項演算子）．前処理の OZ2_DIGITK は，桁分割の経路だけにある型・関数・欄
// （合成段 Oz2Compose・連結の検査・Oz2Run の合成段の欄）と，それを呼ぶ所だけに使う
inline constexpr bool OZ2_DIGIT = OZ2_PARAM.digit_bound != 0;

// GEMM の MathFidelity．規則は scheme I と共通の oz_fidelity_rule（oz_format.hpp）で，
// 桁の上限 digit_bound から決める：INT8 は HiFi4（乗算器が SrcA の上位区画に bit 7--5 の
// 3 bit しか割り当てない．docs/int8_fidelity_mechanism.md），桁分割の 7 経路は桁の上限が
// 32 以下（FP8 16，BF16・FP16・FP32・BFP8_b 32，BFP4_b 7，BFP2_b 1）なので LoFi になる．
// 31 以下の整数は有効桁が 5 bit 以下，32 は 2 のべきなので，どれも LoFi が SrcA に割り当てる
// 上位 5 bit に収まる（docs/fidelity_radix_sweep.md）．桁の上限を 32 より上げれば規則が
// 自動で HiFi4 へ切り替える．
// 実行時の上書き OZ_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4 も scheme I と共通（oz_fidelity_env．値は入口の oz_drv::check_env が
// 検査済み）．環境を読むので，main の冒頭で 1 度だけ呼んで Oz2Knobs::fid に置く（ozaki2_plan.hpp の oz2_knobs_from_env）．
inline tt::tt_metal::MathFidelity oz2_fidelity() {
    if (const auto f = oz_fidelity_env()) return *f;
    // 疎な桁（FP8 の sparse_max=32）はそちらが桁の最大（32 以下なので LoFi のまま）
    return oz_fidelity_rule((uint32_t)std::max(OZ2_PARAM.digit_bound, OZ2_PARAM.sparse_max));
}
