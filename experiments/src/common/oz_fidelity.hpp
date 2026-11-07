// SPDX-License-Identifier: Apache-2.0
#pragma once
// 行列エンジンの MathFidelity：上書きのつまみ OZ_FIDELITY の読み方・表示名・パス数・理論ピーク（形式に依らない部分）
// ---------------------------------------------------------------------------
// 尾崎の両スキームと基準実装（gemm_bf16・gemm_bf16ext・gemm_int8）が同じ解析と表を使う（別々に持つと，一方だけを直した
// ときに同じ設定名で違う測定条件（ピーク換算や表示）になりうる）．既定の決め方だけが違う：尾崎の経路は形式の桁の上限から
// 決め（oz_format.hpp の oz_fidelity_rule），基準実装は HiFi4（baseline_fidelity）．
//   OZ_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4 で上書きする（それ以外の値は黙って既定に戻さず exit(2)．読み方はつまみの表）．
//   ピークは LoFi 1 パスの 4096 演算/サイクル/コア（積和を 2 演算と数える）をパス数で割ったもの．
#include <cstring>
#include <iterator>
#include <optional>
#include <string_view>
#include <tt-metalium/base_types.hpp>
#include "common/oz_knobs.hpp"   // OZ_FIDELITY の読み方（つまみの表．tt-metal に依存しない）

// パス数の順（LoFi・HiFi2・HiFi3・HiFi4）．つまみの表（oz_knobs.hpp）の OZ_FIDELITY の choices と，探針が位置引数で受ける
// 番号 1..4（fp8_probe）も同じ並び
inline constexpr tt::tt_metal::MathFidelity kOzFidelities[] = {
    tt::tt_metal::MathFidelity::LoFi, tt::tt_metal::MathFidelity::HiFi2, tt::tt_metal::MathFidelity::HiFi3,
    tt::tt_metal::MathFidelity::HiFi4};
// 表示名の並び（oz_fidelity_name と同じ）．つまみの表の OZ_FIDELITY の choices が同じ名前を同じ並びで持つことを
// コンパイル時に確かめる（どちらかだけを並べ替えると，OZ_FIDELITY=HiFi2 で別のパス数を選ぶ）
inline constexpr std::string_view kOzFidelityNames[] = {"LoFi", "HiFi2", "HiFi3", "HiFi4"};
constexpr bool oz_fidelity_choices_ok() {
    const std::string_view c = oz_knobs::TABLE[OZK("OZ_FIDELITY")].choices;
    size_t p = 0;
    for (size_t i = 0; i < std::size(kOzFidelityNames); ++i) {
        const std::string_view n = kOzFidelityNames[i];
        if (c.substr(p, n.size()) != n) return false;
        p += n.size();
        if (i + 1 < std::size(kOzFidelityNames) && (p >= c.size() || c[p++] != '|')) return false;
    }
    return p == c.size() && std::size(kOzFidelityNames) == std::size(kOzFidelities);
}
static_assert(oz_fidelity_choices_ok(), "oz_knobs.hpp の OZ_FIDELITY の choices と kOzFidelities・kOzFidelityNames の並びが合わない");

// OZ_FIDELITY の値（未設定なら nullopt）．調査用で，桁が乗算器の幅を超えるところまでパス数を下げれば積が丸まり，答えは
// 無誤差でなくなる（論文の fidelity の掃引）．ホスト側で ComputeConfig に渡すだけなので再ビルドは要らない．
// ドライバは入口の oz_knobs::preflight で値を先に検査する
inline std::optional<tt::tt_metal::MathFidelity> oz_fidelity_env() {
    const int i = oz_knobs::choice(OZK("OZ_FIDELITY"));
    if (i < 0) return std::nullopt;
    return kOzFidelities[i];
}

// 表示名（CONFIG 行の fidelity=）
inline const char* oz_fidelity_name(tt::tt_metal::MathFidelity mf) {
    using tt::tt_metal::MathFidelity;
    switch (mf) {
        case MathFidelity::LoFi:  return kOzFidelityNames[0].data();
        case MathFidelity::HiFi2: return kOzFidelityNames[1].data();
        case MathFidelity::HiFi3: return kOzFidelityNames[2].data();
        default:                  return kOzFidelityNames[3].data();
    }
}

// パス数（スループットと理論ピークはこれに反比例する）
inline double oz_fidelity_passes(tt::tt_metal::MathFidelity mf) {
    using tt::tt_metal::MathFidelity;
    switch (mf) {
        case MathFidelity::LoFi:  return 1.0;
        case MathFidelity::HiFi2: return 2.0;
        case MathFidelity::HiFi3: return 3.0;
        default:                  return 4.0;
    }
}

// 探針の位置引数の読み方（つまみの表を通さない）：表示名（oz_fidelity_name の逆）か，パス数 1..4 から引く．知らない値は
// nullopt（呼び出し側が使い方を出して止める）
inline std::optional<tt::tt_metal::MathFidelity> oz_fidelity_by_name(const char* name) {
    for (const auto mf : kOzFidelities)
        if (std::strcmp(name, oz_fidelity_name(mf)) == 0) return mf;
    return std::nullopt;
}
inline std::optional<tt::tt_metal::MathFidelity> oz_fidelity_by_passes(long passes) {
    if (passes < 1 || passes > (long)std::size(kOzFidelities)) return std::nullopt;
    return kOzFidelities[passes - 1];
}

// LoFi 1 パスの理論ピーク［演算/サイクル/コア］．積和を 2 演算と数えた値 2*8*16*16 で，積和の数にすると
// 2048 積和/サイクル/コアである．呼び出し側は演算数（2*M*N*K）と比べるので，さらに 2 倍しない．
constexpr double kPeakOpsPerCyclePerCoreLoFi = 4096.0;
// チップの理論ピーク［TOP/s］：LoFi 1 パスのピークをパス数で割り，クロック clk_hz［Hz］とコア数 ncores を掛ける．
// 尾崎 I（ozaki1_main.cpp の device_info）と基準実装 3 本の表示が同じ式で求める（掛ける順もここで 1 つに決まる）．
// 行列エンジンの理論上限で，実効スループットがパス数に反比例する保証ではない
inline double oz_peak_tops(tt::tt_metal::MathFidelity mf, double clk_hz, uint32_t ncores) {
    return kPeakOpsPerCyclePerCoreLoFi / oz_fidelity_passes(mf) * clk_hz * ncores / 1e12;
}

// 基準実装の MathFidelity：既定は HiFi4（BF16 の積も INT8 の積も，下位の区画どうしの積 x_lo*y_lo を累積するのは HiFi4
// だけなので，積の全ビットを得るには 4 パスが要る．INT8 は値を絶対値の下位ビットに置くので，値を狭めても lo の区画に
// ビットが残る．論文の表 tab:field）．OZ_FIDELITY で上書きできる
inline tt::tt_metal::MathFidelity baseline_fidelity() {
    return oz_fidelity_env().value_or(tt::tt_metal::MathFidelity::HiFi4);
}
