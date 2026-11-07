// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// ホスト側の小さなビット変換（float・double とそのビット列，IEEE binary16，FP32 の仮数の丸め，最終スケールの 64 bit 仮数）．
// デバイス API に依存しない
// ---------------------------------------------------------------------------
// カーネルへ float を実行時引数・コンパイル時引数として渡すとき（ドライバ・探針・検査）と，ホスト単体の模擬・検査
// （tools/．SFPU の模擬 checks/sfpu_tf_emu.hpp）が同じ 1 つの関数を使う（書き写すと，丸め方や 0・非正規化数の扱いが
// 写しごとに食い違いうる）．カーネルの側の同じ変換は common/kernels/sfpu_tf.h の fb・fbf（デバイスのコンパイラで組む）．
#include <bit>
#include <cmath>
#include <cstdint>

inline uint32_t f32_bits(float f) { return std::bit_cast<uint32_t>(f); }
inline float    bits_f32(uint32_t u) { return std::bit_cast<float>(u); }
inline uint64_t f64_bits(double x) { return std::bit_cast<uint64_t>(x); }
inline double   bits_f64(uint64_t u) { return std::bit_cast<double>(u); }

// 正の long double w を 64 bit の仮数 W = (hi:lo)（最上位ビットが立つ）と指数 q に分ける（w = W·2^q）．2 のべきとは
// 限らない再構成の最終スケール（尾崎 I の radix^-(L+2)，尾崎 II の KM）を，カーネルが整数で掛けられる形で渡すのに使う
// （FP64 の定数として 1 回掛けると相対 2^-53 の誤差が全要素に一様に乗る．common/kernels/f64_tf.h の pack_f64）．
// x86 の long double は仮数が 64 bit なので，この写しは誤差を足さない
struct OzScale64 { uint32_t hi, lo; int32_t q; };
inline OzScale64 oz_split_scale_ld(long double w) {
    int e; const long double f = frexpl(w, &e);
    const unsigned long long W = (unsigned long long)ldexpl(f, 64);
    return {(uint32_t)(W >> 32), (uint32_t)W, e - 64};
}

// FP32 -> IEEE binary16 のビット列（最近接偶数丸め．非正規化数と Inf/NaN も扱う．範囲を超える値は Inf へ飽和）
inline uint16_t half_from_f32(float f) {
    const uint32_t b = f32_bits(f);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const int e = (int)((b >> 23) & 0xFF);
    const uint32_t man = b & 0x7FFFFFu;
    if (e == 0xFF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));   // Inf / NaN
    const int he = e - 127 + 15;                                             // half の指数（バイアス込み）
    if (he >= 0x1F) return (uint16_t)(sign | 0x7C00u);                      // 飽和 -> Inf
    if (he >= 1) {                                                           // 正規化数：仮数 23 -> 10 bit
        uint32_t r = man >> 13;
        const uint32_t rem = man & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;             // 最近接偶数
        return (uint16_t)(sign | (((uint32_t)he << 10) + r));               // r の桁上がりは指数へ繰り上がる
    }
    // 非正規化数（half の最小正規化数 2^-14 未満）．暗黙の 1 を含めた仮数を右へずらす
    if (he < -10) return (uint16_t)sign;                                     // 2^-25 未満は 0
    const uint32_t m = man | 0x800000u;
    const int sh = 14 - he;                                                  // 13 + (1-he)
    uint32_t r = m >> sh;
    const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1u))) r++;
    return (uint16_t)(sign | r);
}
// FP32 の値を精度 prec bit（暗黙の 1 を含む仮数の桁数．binary32 は 24，TF32 は 11，BF16 は 8．1 以上）へ最近接偶数で丸める．
// ビット列の整数加算で丸める：落とす下位 drop bit に「半分 - 1」と，残す最下位 bit（同点のとき偶数へ寄せるため）を足してから
// 切り捨てる．仮数が桁あふれすれば指数へ繰り上がり，値として正しい（2 のべきになる．最大の有限値の上は Inf）．非有限の値と
// prec >= 24 はそのまま返す
inline float f32_round_to_prec(float x, int prec) {
    if (!std::isfinite(x) || prec >= 24) return x;
    const int drop = 24 - prec;
    uint32_t u = f32_bits(x);
    u += ((1u << (drop - 1)) - 1u) + ((u >> drop) & 1u);
    u &= ~((1u << drop) - 1u);
    return bits_f32(u);
}
// IEEE binary16 のビット列 -> FP32（厳密）
inline float f32_from_half(uint16_t h) {
    const uint32_t sgn = (uint32_t)(h & 0x8000u) << 16;
    const int e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x3FFu;
    if (e == 0x1F) return bits_f32(sgn | 0x7F800000u | (m << 13));               // Inf / NaN
    if (e != 0)    return bits_f32(sgn | (uint32_t)((e - 15 + 127) << 23) | (m << 13));   // 正規化数
    if (m == 0)    return bits_f32(sgn);                                          // 0
    return bits_f32(f32_bits(std::ldexp((float)m, -24)) | sgn);                    // 非正規化数：m * 2^-24
}
