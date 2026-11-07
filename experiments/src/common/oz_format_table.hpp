// SPDX-License-Identifier: Apache-2.0
#pragma once
// 形式の表（デバイスのデータ形式まで．ドライバの形式の選択 -DOZ_FMT_* に依らない）
// ---------------------------------------------------------------------------
// 形式そのものの性質のうちホスト単体で読めるもの（表示名・無誤差に表せる整数の上限など）は oz_format_host.h の
// oz_fmt_host にあり，ここはそれに tt-metal のデータ形式を足した表 oz_format(OzFmt) を置く．ドライバは
// common/oz_format.hpp（形式の選択 OZ_FMT_ID・OZ_FMT）を通して，形式を 1 つに決めない探針・検査は
// このヘッダを直接取り込む．
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/base_types.hpp>
#include <cstdint>
#include "common/oz_format_host.h"

struct OzFormat {
    const char*    name;           // 表示名（CONFIG・RESULT 行の fmt）
    tt::DataFormat in;             // GEMM が読むスライスの形式
    tt::DataFormat slice_out;      // スライサが書く CB の形式（in とタイル長が同じであること．oz_format_consistent）
    tt::DataFormat out;            // GEMM の累算・部分積の形式
    bool           out_half;       // スライサが binary16 のビット列を UInt16 で書く（FP16）
    uint32_t       exact_int_max;  // 形式が無誤差に表せる整数 |a| の上限（桁の上限ではない）
};

// データ形式の理由は oz_fmt_host の各行の注記（INT8 は符号-絶対値の UInt8 を Int8 として読み，FP16 は binary16 の
// ビット列の UInt16 を Float16 として読み，FP32 は行列エンジンに TF32 として渡る）．
constexpr OzFormat oz_format(OzFmt f) {
    using DF = tt::DataFormat;
    const OzFmtHost h = oz_fmt_host(f);
    switch (f) {
    case OzFmt::INT8: return {h.name, DF::Int8,      DF::UInt8,     DF::Int32,   h.out_half, h.exact_int_max};
    case OzFmt::FP8:  return {h.name, DF::Fp8_e4m3,  DF::Fp8_e4m3,  DF::Float32, h.out_half, h.exact_int_max};
    case OzFmt::BF16: return {h.name, DF::Float16_b, DF::Float16_b, DF::Float32, h.out_half, h.exact_int_max};
    case OzFmt::FP16: return {h.name, DF::Float16,   DF::UInt16,    DF::Float32, h.out_half, h.exact_int_max};
    case OzFmt::FP32: return {h.name, DF::Float32,   DF::Float32,   DF::Float32, h.out_half, h.exact_int_max};
    case OzFmt::BFP8: return {h.name, DF::Bfp8_b,    DF::Bfp8_b,    DF::Float32, h.out_half, h.exact_int_max};
    case OzFmt::BFP4: return {h.name, DF::Bfp4_b,    DF::Bfp4_b,    DF::Float32, h.out_half, h.exact_int_max};
    case OzFmt::BFP2: return {h.name, DF::Bfp2_b,    DF::Bfp2_b,    DF::Float32, h.out_half, h.exact_int_max};
    }
    return {};
}

// ホスト単体の表（部分積が Int32 か・binary16 で書くか・タイルのバイト数）と，ここのデータ形式が食い違わないこと．
// ホスト単体の道具は oz_fmt_host だけを見るので，片方だけ直すと道具が本番と違う形式を模擬する
constexpr bool oz_format_consistent() {
    for (OzFmt f : OZ_FMTS) {
        const OzFormat d = oz_format(f);
        const OzFmtHost h = oz_fmt_host(f);
        if ((d.out == tt::DataFormat::Int32) != h.acc_int32) return false;
        if ((d.slice_out == tt::DataFormat::UInt16) != h.out_half) return false;
        if (tt::tile_size(d.in) != h.tile_bytes || tt::tile_size(d.slice_out) != h.tile_bytes) return false;
    }
    return true;
}
static_assert(oz_format_consistent(),
              "oz_format_host.h の acc_int32・out_half・tile_bytes と oz_format の out・slice_out・in が合わない");

// MathFidelity の決め方の規則（両スキーム共通．説明は oz_format.hpp の oz_fidelity_rule）を，形式 f を引数にして求める．
// ドライバは選んだ形式で oz_format.hpp の oz_fidelity_rule を，形式を 1 つに決めない検査（gemm_exact_check）はこれを直接呼ぶ
constexpr tt::tt_metal::MathFidelity oz_fidelity_rule_for(OzFmt f, uint32_t digit_bound) {
    if (f == OzFmt::INT8) return tt::tt_metal::MathFidelity::HiFi4;
    return (digit_bound <= 32) ? tt::tt_metal::MathFidelity::LoFi : tt::tt_metal::MathFidelity::HiFi4;
}
