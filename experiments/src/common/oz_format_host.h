// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 両尾崎スキームで共通の形式の列挙と，形式そのもののうちデバイス API に依存しない性質
// ---------------------------------------------------------------------------
// 形式に依存する値は表から引き，値を書くのは表の 1 か所だけにする．表は次の 4 つに分かれる：
//   OzFmtHost（このヘッダ）                      形式そのものの性質のうちホスト単体で読めるもの（表示名・
//                                                無誤差に表せる整数の上限・部分積が Int32 か・binary16 で書くか・
//                                                タイル 1 枚のバイト数）
//   OzFormat（common/oz_format_table.hpp）       GEMM が読む・スライサが書く・累算するデバイスのデータ形式
//   Oz1Param（ozaki1/ozaki1_param.h）            scheme I の既定の基数とカーネル接尾辞
//   OZ2_ROUTES（ozaki2/ozaki2_split.h）          scheme II の経路（法の上限・桁の上限・桁数・既定の法の本数）
// このヘッダと ozaki1_param.h・ozaki2_split.h は tt-metal を取り込まない（<cstdint> と <cstring> だけ）ので，
// ホスト単体の検査・模擬（tools/）も本番と同じ表を取り込む．ドライバの形式の選択（-DOZ_FMT_* と OZ_FMT_ID）は
// common/oz_format.hpp にある．
//
// 「形式が無誤差に表せる整数の上限」（exact_int_max）と「実際に使う桁の上限」は別の量である．後者は乗算器の区画
// （LoFi で 32）と累算器の幅にも縛られるので，前者より小さいことがある（BF16 は 256 対 32）．scheme I の桁の上限は
// 基数から実行時に決まり（ozaki1_format.hpp の ozaki_digit_max．既定の規則は ozaki1_param.h の oz1_digit_max），
// scheme II のそれは経路の表の digit_bound である．
#include <cstdint>
#include <cstring>

enum class OzFmt { INT8, FP8, BF16, FP16, FP32, BFP8, BFP4, BFP2 };
// 全形式（列挙の順）．形式の表を総当たりで見る検査と，名前から形式を引く oz_fmt_by_name が使う
inline constexpr OzFmt OZ_FMTS[] = {OzFmt::INT8, OzFmt::FP8,  OzFmt::BF16, OzFmt::FP16,
                                    OzFmt::FP32, OzFmt::BFP8, OzFmt::BFP4, OzFmt::BFP2};

// 形式そのものの性質（スライサが書き，GEMM が読み，GEMM が累算する形式はスキームによらない）
struct OzFmtHost {
    const char* name;           // 表示名（CONFIG・RESULT 行の fmt）
    bool        out_half;       // スライサが binary16 のビット列を UInt16 で書く（FP16）
    bool        acc_int32;      // GEMM の部分積が Int32（INT8．ほかは Float32）
    uint32_t    exact_int_max;  // 形式が無誤差に表せる整数 |a| の上限（桁の上限ではない）
    // 32x32 のタイル 1 枚のバイト数．スライサが書く形式と GEMM が読む形式で同じ（INT8 の UInt8 と Int8，FP16 の UInt16 と
    // Float16）．BFP は 16 要素ごとの共有指数 64 B を含む．tt-metal の tile_size と同じことを oz_format_table.hpp が
    // 確かめ，ホスト単体の見積り（ozaki2_awe.h の OZ2_AWE_SLICE_TILE）はここから引く
    uint32_t    tile_bytes;
};
constexpr OzFmtHost oz_fmt_host(OzFmt f) {
    switch (f) {
    // 符号-絶対値バイト（to_sign_mag）で運び，部分積は無誤差な Int32．上限は絶対値部の 7 bit
    case OzFmt::INT8: return {"int8", false, true,  127,  1024};
    // e4m3 が無誤差に表せる整数の上限（fp8_probe で実測）
    case OzFmt::FP8:  return {"fp8",  false, false, 16,   1024};
    // 仮数 8 bit
    case OzFmt::BF16: return {"bf16", false, false, 256,  2048};
    // IEEE binary16．仮数 11 bit（暗黙の 1 を含む）．
    // FP32 の DST から Float16 へ直接パックすると値が壊れる（docs/exponent_format_constraint.md）．
    // そこでスライサは binary16 のビット列を整数値として作り（sfpu_tf.h の HalfBits），UInt16 の
    // CB で書き出す．GEMM は同じバイト列を Float16 として読む（タイル長はどちらも 2048 B）．
    // 2048 は格納の上限である．連続整数を無誤差に掛けられる上限は，LoFi では乗算器の区画幅により32，
    // HiFi2以上では8項の指数合わせにより63となる．SrcAの仮数bit0の欠落はこれとは別の制約
    // （docs/fidelity_radix_sweep.md）．
    case OzFmt::FP16: return {"fp16", true,  false, 2048, 2048};
    // FP32 で格納し，行列エンジンには TF32 として渡る（論文での名前は TF32）．TF32 は L1 の CB
    // 形式としては存在せず，FP32 の CB を fp32_dest_acc_en の下でアンパッカが仮数 10 bit へ縮めて
    // 展開する（jit_build/genfiles.cpp の compute_data_formats）．本実装の GEMM カーネルはすべて
    // fp32_dest_acc_en=true なので乗算は TF32 で行われ，上限も TF32 の仮数 11 bit で数える．
    // Float32 は指数部の族の検査を免除されるので，FP16 のような細工は要らない．
    case OzFmt::FP32: return {"fp32", false, false, 2048, 4096};
    // block-float 8．16 要素が指数を共有しても整数 0..127 はどの面でも厳密（仮数 7 bit）．
    // 1 要素 1.0625 B（1024 B の仮数 + 64 B の指数 / タイル）
    case OzFmt::BFP8: return {"bfp8", false, false, 127,  1088};
    // block-float 4．仮数 3 bit（符号は別）なので共有指数のもとで厳密な連続整数は [-7,7]．
    // 1 タイル 576 B で 0.5625 B/要素
    case OzFmt::BFP4: return {"bfp4", false, false, 7,    576};
    // block-float 2．仮数 1 bit（符号は別）なので厳密な整数は [-1,1] だけ．1 タイル 320 B で
    // 0.3125 B/要素と，本ハードウェアが入力に取れる最も狭い形式
    case OzFmt::BFP2: return {"bfp2", false, false, 1,    320};
    }
    return {};
}

// 名前（OzFmtHost::name．CONFIG・RESULT 行の fmt と同じ）から形式を引く．知らない名前なら false（out は変えない）
inline bool oz_fmt_by_name(const char* name, OzFmt& out) {
    for (OzFmt f : OZ_FMTS)
        if (std::strcmp(name, oz_fmt_host(f).name) == 0) { out = f; return true; }
    return false;
}
