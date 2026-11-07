// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：ホストとカーネルが共有する上限と定数（両側が同じ 1 つの値を見る）
// ---------------------------------------------------------------------------
// 片側だけ直すと黙って誤る値をここに 1 つだけ置く．例えば合成段の係数の枠 OZ2_MAXPROD をホストだけ増やすと，
// カーネル（oz2_compose.cpp）は A_j を古い位置から読み，B 側の係数か 0 を読んで剰余が黙って誤る．スライス方式の番号
// （Oz2XMode）をホストだけ足すと，スライサ（modslice_digit.cpp）は既定の枝へ黙って落ちる．
// 引数の並びは recon_args.h・oz2_args.h にある．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．ホストは ozaki2_split.h を通して取り込む．
#pragma once
#include <cstdint>

// 法の本数の上限．カーネルの法ごとの固定配列の長さ（modslice_digit.cpp・modslice_int8.cpp・recon_crt.cpp の MAXS）で，
// ホストの法選択（ozaki2_moduli.h）もこれを超える集合を選ばない
inline constexpr int OZ2_MAXS = 24;
// 法 1 本あたりの行列積の上限．2 桁分割では 3〜4 本だが，平衡 5 桁（BFP2_b）は 25 本になる．
// 合成段（oz2_compose.cpp）の係数の枠の大きさで，実行時引数の個数もこれで決まる（oz2_args.h の CP_COUNT）
inline constexpr int OZ2_MAXPROD = 32;
// A' を 2^20 進 3〜OZ2_MAXDIGIT 桁に分ける基数（のビット数）．スライサは桁ごとに先に剰余を取ってから重みを掛けるので
// （oz2_digits.h の oz2_residue），桁は ModRound が無誤差に扱える 2^22 まで許される．20 bit に
// 採ると既定の b=54 は 3 桁（60 bit まで）で運べ，b を 60・80 より上げるときだけ 4・5 桁にする（ozaki2_plan.hpp の
// choose_moduli_and_width の ND）．
inline constexpr int OZ2_DIGIT_BITS = 20;
// 桁数の上限．A' は 2^20 進 N 桁で運ぶので b <= N*20．既定の b<=54 なら 3 桁で足り，
// 動的域の広い入力で b を 60 より上げるときだけ 4 桁，80 より上げるときだけ 5 桁を使う（oz2_digits.h）．
// 5 桁が要るのは，行の最大値の 2^-80 より小さい要素が他方の行列の大きい要素と掛かって
// 出力の 1 要素を決める入力である（4096^3・phi=16 で b=83 が要る．b<=80 では 50.0 bit で止まる）．
inline constexpr int OZ2_MAXDIGIT = 5;
// 再構成の逆数の断片の本数の上限（recon_crt.cpp の固定配列の長さ）．ホストの断片の本数 OZ2_NRP0・NRP1・NRP2
// （ozaki2_moduli.h）はこれ以下
inline constexpr int OZ2_MAXRP = 5;
// 合成段の 2^12 分割の基数のビット数（fold_first=0 の方式．oz2_compose.cpp の hi = round(P/2^12)・lo = P - hi*2^12）．
// ホストは係数を A_j = c_j*2^12 mod m・B_j = c_j mod m として渡し（ozaki2_split.h の OZ2_COMPOSE_SPLIT と oz2_product_terms の
// coeff_hi），法の選択はこの基数で合成の総和の上界を数える（ozaki2_moduli.h の oz2_plan_concat）．片側だけ変えると剰余が黙って誤る
inline constexpr int OZ2_COMPOSE_SPLIT_BITS = 12;

// スライサ（modslice_digit.cpp）の桁の作り方の番号（法ごとの実行時引数 XMODE．float のビット列で渡り，float で比べる）．
// uint32 で渡して整数で比べる版は，II-BFP2・II-BFP4 の法の 3 組の組分け（OZ2_MOD_GROUPS=3，512^3．selftest.sh の
// 「II-BFP2 groups」）でスライサが止まり，ボードのリセットが要る（2026-10-23 に 3 回再現．float の版は同じ木で通る）．
// 原因は突き止めていない：値の意味は同じで，変わるのは計算カーネルの数学・パックのスレッドの命令列と時間だけである
// （TRISC は浮動小数点の演算器を持たないので float の比較は soft-float の呼び出しになる．展開のスレッドの命令列は同じ．
// スタックは局所メモリに収まる）．それまで float のまま比べる．
// ホストは法の分け方（ozaki2_split.h の OZ2_SPLIT_*）から oz2_xmode で引く．方式 (A)(C) はスライサから見ると同じ手続き
// （違うのはホスト側の合成式だけ）なので同じ番号になる．
enum Oz2XMode : uint32_t {
    OZ2_XMODE_RND = 0,   // 最近接丸めの平衡桁 2 枚（平方数 (A)・WIDE (C)）
    OZ2_XMODE_KAR = 1,   // Karatsuba の切り上げ桁 3 枚（(B)．基数 q）
    OZ2_XMODE_SQS = 2,   // 疎な桁の平方数（(E)．FP8 だけ）
    OZ2_XMODE_BAL = 3,   // 平衡 n 桁（(D)．桁数 n は続く NDIG）
    OZ2_XMODE_ONE = 4,   // 1 桁（(F)．桁 32 の AWE 経路の m <= 65 の法）
    OZ2_XMODE_COUNT      // 番号を足したら modslice_digit.cpp の枝と static_assert も直す
};
