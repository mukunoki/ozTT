// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// SFPU のタイル演算ラッパと triple-float 演算（SFPU を使う計算カーネルが共有する．GEMM 段の bmm_int32_spill.cpp・
// bmm_l1acc.cpp は取り込まない）
// ---------------------------------------------------------------------------
// 尾崎スキーム I／II の計算カーネルが共有するタイル演算のラッパ（Add・MulS・Rnd など）と，TwoSum・tfMulR・tfAddS・
// ModRound などの手続きの入口を置く．カーネル側は API ヘッダの後にこのファイルを include すればよい．
// 手続きの本体と正しさの議論（補題・保証する範囲）は sfpu_tf_proc.h にある．本体をデバイス API に依存しない
// テンプレートにしてあるのは，ホストの模擬（src/checks/sfpu_tf_emu.hpp）が同じものを組むためである：写しを持つと，
// 片方だけを直したときに模擬が古い手順のまま検査を通しうる．
//
// 本ハードウェアの制約
//   FMA（IEEE 754 の fusedMultiplyAdd．a*b+c を無限精度で計算して 1 度だけ丸める）は
//   無い．SFPU には積和命令 SFPMAD があり，タイル演算 API にも addcmul（a + v*b*c）の
//   形で出ているが，2 つの FP32 の積 48 bit のうち上位 27 bit（FP32 の 24 bit + 3 bit）
//   しか保たずに丸めるので，1 度だけ丸める演算にはなっていない（fma_probe で実測．
//   a=b=1+2^-j, c=-(1+2^-(j-1)) の掃引が 2^-26 で飽和し，オペランドを 2^±10 倍しても
//   飽和点は積の指数に対して同じ位置にある）．したがって a*b - fl(a*b) を 1 命令で
//   取り出せず，誤差なしの積（TwoProd）は FMA 版ではなく Dekker 分割版を使う．
//   乗算と加算を別のタイル演算として書けば積は FP32 へ正しく丸められるので，
//   本ヘッダの書き方ならこの 3 bit は混ざらない．
//   DST 間のコピーだけは専用 API（copy_dest_values）が 1 命令対で行う．
//
//   また TT-Metalium はカーネルを既定で -ffast-math 付きで JIT ビルドするため，
//   無誤差変換を RISC-V 上の C の float 式で書くと残差の式が代数的に 0 へ簡約され，
//   誤りを出さないまま精度が落ちる．ここの手続きはすべて SFPU のタイル組込み関数
//   で書いてあるので影響を受けない．
//
// レジスタの約束（DST の 3 本組と作業用の番号，名前 dst）は sfpu_tf_proc.h の冒頭．

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/right_shift.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "sfpu_tf_proc.h"   // 手続きの本体と DST レジスタの名前 dst（デバイス API に依存しない．ホストの模擬も取り込む）

// --- スカラとビット列の相互変換 ---------------------------------------------
// 実行時引数もコンパイル時引数も uint32 で届くので，ビット列のまま float に戻す．
ALWI uint32_t fb(float s)   { union { float f; uint32_t u; } c; c.f = s; return c.u; }
ALWI float    fbf(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }

// --- タイル単位の基本演算 ---------------------------------------------------
ALWI void Add(uint32_t a, uint32_t b, uint32_t o) { add_binary_tile_init(); add_binary_tile(a, b, o); }
ALWI void Sub(uint32_t a, uint32_t b, uint32_t o) { sub_binary_tile_init(); sub_binary_tile(a, b, o); }
ALWI void Mul(uint32_t a, uint32_t b, uint32_t o) { mul_binary_tile_init(); mul_binary_tile(a, b, o); }
// d = s．DST 間のコピーは SFPLOAD/SFPSTORE 1 対で済む専用 API に落ちる
// （ckernel_sfpu_copy_dest_values.h）．copy_dest_values は Blackhole では純粋な
// レジスタ移動で，「0 を作ってから足す」2 演算で写すより 1 演算少ない．Cpy は
// tfMulR に 2 回・tfAddS に 2 回現れるので，tfMulR の Horner 1 段（tfMulR と tfAddS．基数が 2 のべきでない経路）は
// 63 演算ではなく 59 演算で済む（基数が 2 のべきなら基数倍は tfMulP2 で，1 段 34 演算）．
// s と d は別のレジスタでなければならない．
ALWI void Cpy(uint32_t s, uint32_t d) {
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(s, d);
}
// スカラ倍・スカラ加算．ビット列で渡す版（コンパイル時定数向け）と
// float で渡す版（実行時引数向け）の両方を用意する．
ALWI void MulS(uint32_t d, uint32_t sb) { binop_with_scalar_tile_init(); mul_unary_tile(d, sb); }
ALWI void MulS(uint32_t d, float s)     { MulS(d, fb(s)); }
ALWI void AddS(uint32_t d, uint32_t sb) { binop_with_scalar_tile_init(); add_unary_tile(d, sb); }
ALWI void AddS(uint32_t d, float s)     { AddS(d, fb(s)); }
// d = 定数．直前の d の中身に依らない．自己減算（Sub(d,d,o)）で 0 を作る手は，
// d が未定義（NaN/Inf でありうる）の場面では 0 にならない．
ALWI void Set(uint32_t d, float s) { fill_tile_init(); fill_tile(d, s); }
ALWI void Rnd(uint32_t d) { rounding_op_tile_init(); round_tile(d, 0); }
ALWI void Ltz(uint32_t d) { ltz_tile_init(); ltz_tile(d); }   // d = (d<0) ? 1 : 0

// --- 手続き（triple-float と法の剰余）----------------------------------------
// 本体と正しさの議論は sfpu_tf_proc.h（ラッパの型を引数に取るテンプレートで，ホストの模擬 sfpu_tf_emu.hpp も同じものを組む）．
// ここはデバイスのラッパ SfpuDev で呼ぶ入口で，カーネルはこの名前で呼ぶ．
struct SfpuDev {
    static constexpr uint32_t SPL = 0x45800800u;   // Dekker（Veltkamp）分割の定数 2^12+1 = 4097.0f のビット列
    ALWI void Add(uint32_t a, uint32_t b, uint32_t o) { ::Add(a, b, o); }
    ALWI void Sub(uint32_t a, uint32_t b, uint32_t o) { ::Sub(a, b, o); }
    ALWI void Mul(uint32_t a, uint32_t b, uint32_t o) { ::Mul(a, b, o); }
    ALWI void Cpy(uint32_t s, uint32_t d) { ::Cpy(s, d); }
    ALWI void MulS(uint32_t d, uint32_t sb) { ::MulS(d, sb); }
    ALWI void MulS(uint32_t d, float s) { ::MulS(d, s); }
    ALWI void AddS(uint32_t d, uint32_t sb) { ::AddS(d, sb); }
    ALWI void AddS(uint32_t d, float s) { ::AddS(d, s); }
    ALWI void Rnd(uint32_t d) { ::Rnd(d); }
    ALWI void Ltz(uint32_t d) { ::Ltz(d); }
    // 模擬が比較用の版へ差し替える 2 段．本番はどちらも sfpu_tf_proc.h の手続きそのもの
    ALWI void FinalFts(uint32_t a, uint32_t b, uint32_t s, uint32_t t, uint32_t w1, uint32_t /*w2*/) {
        sfpu_proc::FastTwoSum(*this, a, b, s, t, w1);
    }
    ALWI void SmallTwoSum() { sfpu_proc::SmallTwoSum(*this); }
};
ALWI void TwoSum(uint32_t x, uint32_t y, uint32_t s, uint32_t e, uint32_t a, uint32_t b) {
    SfpuDev w; sfpu_proc::TwoSum(w, x, y, s, e, a, b);
}
// 3 本組 dst::acc_hi・acc_mid・acc_lo を基数 rdx 倍する（その場更新．rdx はビット列．作業に dst::term・tmp0..2・keep を
// 使って壊す）．一般の基数は Dekker 分割の tfMulR，2 のべきは tfMulP2（sfpu_tf_proc.h）
ALWI void tfMulR(uint32_t rdx) { SfpuDev w; sfpu_proc::tfMulR(w, rdx); }
ALWI void tfMulP2(uint32_t rdx) { SfpuDev w; sfpu_proc::tfMulP2(w, rdx); }
// 基数倍の振り分け（コンパイル時）．基数 RDX（float のビット列）の仮数部が 0，すなわち
// 2 のべきなら tfMulP2，そうでなければ tfMulR を使う．本番の値域では結果はどちらでも同じなので（sfpu_tf_proc.h の
// tfMulP2 の一致の条件），2 のべきの基数では Dekker 分割の恒等作業が消えて 44 演算が 19 演算になる．
// 尾崎 I の既定の基数では BF16・FP16・TF32・BFP8_b（64），FP8（32），BFP2_b（2）が 2 のべきで，
// INT8（254）と BFP4_b（14）は tfMulR のまま．基数はコンパイル時引数で届くので，OZ_RADIX で
// 変えてもカーネルの JIT ビルドごとに正しい側が選ばれる．作業に使うレジスタは両者とも dst::term・tmp0..2・keep．
template <uint32_t RDX>
ALWI void tfMulRadix() {
    if constexpr (sfpu_proc::radix_is_pow2(RDX)) tfMulP2(RDX);
    else                                         tfMulR(RDX);
}
// 3 本組 dst::acc_hi・acc_mid・acc_lo（NL=2 なら 2 本組）に v（dst::term）を足す（その場更新．作業 dst::tmp0..2．keep は触らない）
template <uint32_t NL>
ALWI void tfAddS() { SfpuDev w; sfpu_proc::tfAddS<NL>(w); }
// 小さい項 v（dst::term）の足し込み（tfAddS の第 1 段を飛ばす．|v| <= eps の前提は sfpu_tf_proc.h）
template <uint32_t NL>
ALWI void tfAddSmall() { SfpuDev w; sfpu_proc::tfAddSmall<NL>(w); }
// d -= round(d/m)*m（o と d は同じでもよい．s は作業用で d とは別．保証する範囲は sfpu_tf_proc.h）
ALWI void ModRound(uint32_t d, uint32_t o, uint32_t s, float m, float rinv) {
    SfpuDev w; sfpu_proc::ModRound(w, d, o, s, m, rinv);
}
// その場更新の別名（呼び出し側の意図が読みやすいように）
ALWI void ModRound(uint32_t d, uint32_t s, float m, float rinv) { ModRound(d, d, s, m, rinv); }
// 符号-絶対値の 8 bit 符号化（その場更新．w1, w2 は作業用．sfpu_tf_proc.h）
ALWI void SignMag8(uint32_t d, uint32_t w1, uint32_t w2) { SfpuDev w; sfpu_proc::SignMag8(w, d, w1, w2); }

// --- FP16 経路：整数値を binary16 のビット列へ ------------------------------------
// なぜ要るか
//   FP16 経路のスライサは DST を FP32 で使う（triple-float の残差を持つため）．ところが
//   FP32 の DST から Float16 の CB へパックすると値が壊れる（大小関係だけが残った別の値に
//   なる．docs/exponent_format_constraint.md 追記 4）．16 bit の DST からなら正しく書けるが，
//   それではスライサが組めない．そこで binary16 のビット列を「その値を持つ整数」として
//   DST に作り，UInt16 のまま CB へ書く．GEMM 側は同じバイト列を Float16 として読む．
//
// 何をするか
//   整数 d に対し，binary16 のビット列 sign<<15 | (15+e)<<10 | 仮数 10 bit に等しい
//   整数値を reg d に作る．例：d = -3 = -1.1b × 2^1 なら 0x8000 | 16<<10 | 0x200 = 0xC200．
//
// どうやるか
//   a = |d| の FP32 のビット列は (127+e)<<23 | 仮数 23 bit．これを 13 bit 右へずらすと
//   (127+e)<<10 | 仮数の上位 10 bit になり，指数の下駄を 127 から 15 へ付け替える
//   （(127-15)<<10 = 114688 を引く）だけで binary16 の絶対値部になる．
//   right_shift_tile は DST の 32 bit 語を値ではなく整数のビット列として扱うので，この
//   「float のビット列をずらす」操作がそのまま書ける．a >= 0 なので符号ビットは 0 で，
//   算術シフトか論理シフトかは結果に関係しない．
//   ずらした結果は整数のビット列なので，Int32 -> Float32 の typecast で値に戻してから
//   float 演算で下駄を引き，符号ビット 32768 を足す．途中の値は 2^18 未満で FP32 に厳密に
//   載る．a = 0 だけは FP32 のビット列が 0 で指数の式が成り立たないので，最後に
//   [a>0] を掛けて 0 にする（d = -0 も 0x0000 になる）．
//   呼び出し側はこのあと Float32 -> UInt16 の typecast をかけてから UInt16 の CB へ
//   パックする（ozaki1/kernels/slice_tf_num.cpp，ozaki2/kernels/modslice_digit.cpp）．
//
// 契約
//   d は整数値で |d| <= 2048．この範囲の整数はすべて仮数 11 bit（10 bit + 暗黙の 1）に
//   収まり，ずらしても下位ビットが落ちない．2048 を超えると 2049 のように 11 bit に
//   収まらない整数が別の値になる（2050 や 4096 のように収まるものは正しい）．スライスの桁は高々 ±32 なので
//   十分に内側である．d と w1..w3 は互いに異なる DST レジスタで，w1..w3 は作業用
//   （呼び出し後の値は保証しない）．約 18 演算．
ALWI void HalfBits(uint32_t d, uint32_t w1, uint32_t w2, uint32_t w3) {
    Cpy(d, w1); Ltz(w1);                                               // w1 = [d<0]（符号）
    Mul(d, w1, w2); MulS(w2, -2.f); Add(d, w2, w2);                    // w2 = a = d - 2d[d<0] = |d|
    Cpy(w2, w3); AddS(w3, -0.5f); Ltz(w3); MulS(w3, -1.f); AddS(w3, 1.f);   // w3 = 1 - [a<0.5] = [a>0]
    Cpy(w2, d);
    right_shift_tile_init(); right_shift_tile(d, 13);                  // d = bits(a) >> 13 = (127+e)<<10 | 仮数（整数のビット列）
    typecast_tile_init<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>();
    typecast_tile<(uint32_t)DataFormat::Int32, (uint32_t)DataFormat::Float32>(d);   // そのビット列の整数値を float の値へ
    AddS(d, -114688.f);                                                // 指数の下駄 127 -> 15：(15+e)<<10 | 仮数
    MulS(w1, 32768.f); Add(d, w1, d);                                  // 符号ビット（bit 15）を足す
    Mul(d, w3, d);                                                     // a = 0 なら 0x0000
}
