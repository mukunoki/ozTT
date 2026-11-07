// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// triple-float と法の剰余の手続き（SFPU を使う計算カーネルが sfpu_tf.h を通して共有する．GEMM 段の bmm_int32_spill.cpp・
// bmm_l1acc.cpp は取り込まない．SFPU のタイル演算のラッパの名前だけを呼び，デバイス API に依存しない）
// ---------------------------------------------------------------------------
// 手続きはタイル演算のラッパの型 W をテンプレートの引数に取る．カーネルは sfpu_tf.h のデバイスのラッパ SfpuDev で組み
// （sfpu_tf.h の同名の自由関数が呼ぶ），ホストの模擬（src/checks/sfpu_tf_emu.hpp の Dst．FP32 の 1 演算の模型）は同じ手続きを
// そのまま組む．手続きの写しを持たないので，模擬と実機の照合（src/checks/tfmul_probe.cpp）は「ラッパの 1 演算の模型が実機と
// 同じか」を見ればよい．
// W が持つもの（デバイスは sfpu_tf.h の SfpuDev，模擬は sfpu_tf_emu.hpp の Dst）：
//   Add・Sub・Mul（a, b, o），Cpy（s, d），MulS・AddS（d と，float の値か float のビット列のスカラ），Rnd，Ltz，
//   FinalFts（a, b, s, t, w1, w2：tfMulR・tfMulP2 の最後の正規化．デバイスは下の FastTwoSum そのもの．
//     模擬は比較用の TwoSum への切り替え（w2 も作業に使う）と前提の計装をここに持つ），
//   SmallTwoSum（tfAddSmall の第 2 段．デバイスは下の SmallTwoSum そのもの．模擬は比較用の FastTwoSum への切り替えを持つ），
//   SPL（Dekker 分割の定数 4097．MulS に渡す形：デバイスはビット列，模擬は float）
//
// レジスタの約束（名前は下の namespace dst）
//   triple-float の値は DST の 3 本組 acc_hi・acc_mid・acc_lo に置く（上位・中位・下位）．
//   tfMulR・tfMulP2（と両者を振り分ける tfMulRadix）は作業に term・tmp0..2・keep を，tfAddS は tmp0..2 と
//   v の入る term を使う（tfAddS は keep を壊さない）．
#include <cstdint>
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))   // カーネルでは tt-metal の API ヘッダが同じものを定義する
#endif

// --- DST レジスタの役割 -----------------------------------------------------
// このヘッダの triple-float 演算は，DST の 8 本をこう使う前提で書いてある．
// 番号を直に書くと，どれが壊れてよくてどれを跨いで保持できるかが読めない．
//   acc_hi/mid/lo  3 本組のアキュムレータ．tfMulR・tfAddS・tfAddSmall が更新する
//   term           足し込む値 v．呼び出し側が入れる．tfAddS<3> は作業にも使って壊す
//                  （tfAddS<2> と tfAddSmall は読むだけ）
//   tmp0..2        作業用．各関数が自由に壊してよい
//   keep           tfAddS・tfAddSmall は触らないので，呼び出し側が項の計算中ずっと保持する
//                  値に使える（recon_crt.cpp の k_i）．ただし tfMulR と tfMulP2 は term・keep も
//                  作業に使って壊すので，それらをまたいで値を残すことはできない
namespace dst {
constexpr uint32_t acc_hi = 0, acc_mid = 1, acc_lo = 2;
constexpr uint32_t term = 3;
constexpr uint32_t tmp0 = 4, tmp1 = 5, tmp2 = 6;
constexpr uint32_t keep = 7;
}

namespace sfpu_proc {

// --- 無誤差変換 -------------------------------------------------------------
// TwoSum：s = fl(x+y)，e = (x+y) - s（丸め誤差を無誤差に回収する．Knuth）．
// 融合積和が使えないので加減算だけの 6 演算版を使う．s,e,a,b は互いに異なるレジスタ．
template <class W>
ALWI void TwoSum(W& w, uint32_t x, uint32_t y, uint32_t s, uint32_t e, uint32_t a, uint32_t b) {
    w.Add(x, y, s); w.Sub(s, x, b); w.Sub(s, b, a); w.Sub(x, a, a); w.Sub(y, b, b); w.Add(a, b, e);
}
// FastTwoSum（Dekker）：(s, t) = (fl(a+b), b - (fl(a+b) - a))．w1 は作業．s + t = a + b が厳密なのは
// 「a = 0 か b = 0 か，a の指数 >= b の指数」のとき（tfMulR・tfMulP2 の最後の正規化．前提は付録 A の補題 P・R）．
// 手続きはこれを W のメンバ FinalFts を通して呼ぶ（模擬が比較用の TwoSum へ差し替え，前提を数えるため．
// デバイスの SfpuDev::FinalFts はこれをそのまま呼ぶ）
template <class W>
ALWI void FastTwoSum(W& w, uint32_t a, uint32_t b, uint32_t s, uint32_t t, uint32_t w1) {
    w.Add(a, b, s); w.Sub(s, a, w1); w.Sub(b, w1, t);
}

// --- triple-float 演算 ------------------------------------------------------
// 【tfMulR・tfMulP2 の最後の FastTwoSum の前提】（契約．証明と検査の内訳はdocs/accuracy.md「SFPU の補題と累積誤差の検査」の付録 A）
// 両者は最後に (a, b) -> (s, t) = (fl(a+b), b - (fl(a+b) - a)) の 3 演算で正規化する（tfMulR は (a,b) = (e_c, G)，
// tfMulP2 は (e6, e4)．名前は下の本体の注記）．これが s + t = a + b を厳密に返すのは「a = 0 か b = 0 か，a の指数 >=
// b の指数」のときで（Dekker），外れると b の ulp の半分ほどを落とす．入力の 3 本組が重なっていると外れる（(1, 2^50, 2^26) を
// 基数 2 倍すると (2^51, 2^27, 0) になり真値の末尾 +2 を失う．tools/tfmul_check.cpp の [0]）．
// 保証：3 本組 (x0,x1,x2) が H_c(u)（x0 は 2 のべき u の整数倍，|x1| + |x2| <= c u）を満たし，基数 β について
//   tfMulP2（β は 2 のべき）は β c < 2^23（補題 P），tfMulR は β c (1+2^-22) < 2^8（補題 R．Horner の H_4 では < 2^10 でも通る）
// なら前提が成り立ち，出力も H_{c1}（c1 = 1 + 2^-24）を満たす．本番の呼び出し（尾崎 I・II のスライサの入口 H_1(u)，
// 尾崎 I の Horner の H_4(ulp(s))・H_{2^11}(1)）はこの条件の内側にある（付録 A）．
// 検査の入口：tools/tfmul_check.cpp の [1]〜[6]（この手続きを SFPU の 1 演算の模型の上で組み，前提違反 0・不一致 0）と，
// 模擬が実機とビット単位で一致することを確かめる src/checks/tfmul_probe.cpp（どちらも scripts/selftest.sh が毎回まわす）．
// 3 本組 acc_hi・acc_mid・acc_lo を基数 rdx 倍する（その場更新）．rdx は W の MulS に渡す形（デバイスは float のビット列，模擬は float）．
// 既定の基数は最大でも 254（8 bit）なので，Dekker 分割した半分（12 bit）との積は
// 12+8 <= 24 bit に収まり，各 limb の積が無誤差になる．一般の triple-word 乗算より
// 手数が短い．
// 作業に term・tmp0..2・keep を使う（term と keep もここでは壊す）．途中では acc_hi・acc_mid も作業に使い，
// 各行の注記がその時点の中身を付録 A の証明の名前で示す．
// 付録 A の証明の名前：P1 + F1 = β x1，P0 + F0 = β x0（Dekker の積．どちらも厳密），G1 = fl(F1 + fl(β x2))，
// (s_a, e_a) = TwoSum(P1, G1)，(s_b, e_b) = TwoSum(F0, s_a)，(r0, e_c) = TwoSum(P0, s_b)，G = fl(e_a + e_b)，
// 最後に FastTwoSum(e_c, G) -> (r1, r2)．近似は fl(β x2)・G1・G の 3 か所だけ．
template <class W, class R>
ALWI void tfMulR(W& w, R rdx) {
    using namespace dst;
    w.MulS(acc_lo, rdx);                                            // 最下位はここだけ誤差を捨てる
    w.Cpy(acc_mid, tmp1); w.MulS(tmp1, W::SPL); w.Sub(tmp1, acc_mid, tmp2); w.Sub(tmp1, tmp2, tmp1);   // tmp1 = 中位の上半分
    w.Sub(acc_mid, tmp1, tmp2);                                     // tmp2 = 中位の下半分
    w.MulS(tmp1, rdx); w.MulS(tmp2, rdx);                           // どちらも無誤差
    w.Add(tmp1, tmp2, term); w.Sub(term, tmp1, keep); w.Sub(tmp2, keep, tmp0);   // 積を term（P1），残差を tmp0（F1）へ
    w.Cpy(acc_hi, tmp1); w.MulS(tmp1, W::SPL); w.Sub(tmp1, acc_hi, tmp2); w.Sub(tmp1, tmp2, tmp1);     // tmp1 = 上位の上半分
    w.Sub(acc_hi, tmp1, tmp2);                                      // tmp2 = 上位の下半分
    w.MulS(tmp1, rdx); w.MulS(tmp2, rdx);
    w.Add(tmp1, tmp2, acc_mid); w.Sub(acc_mid, tmp1, keep); w.Sub(tmp2, keep, acc_hi);   // 積を acc_mid（P0），残差を acc_hi（F0）へ
    w.Add(tmp0, acc_lo, tmp0);                                      // tmp0 = G1
    TwoSum(w, term, tmp0, tmp1, tmp2, keep, acc_lo);                // 4 成分を 3 本組へ正規化する：(s_a, e_a) = (tmp1, tmp2)
    TwoSum(w, acc_hi, tmp1, term, tmp0, keep, acc_lo);              // (s_b, e_b) = (term, tmp0)
    TwoSum(w, acc_mid, term, acc_hi, tmp1, keep, acc_lo);           // (r0, e_c) = (acc_hi, tmp1)
    w.Add(tmp2, tmp0, tmp2);                                        // tmp2 = G
    w.FinalFts(tmp1, tmp2, acc_mid, acc_lo, keep, term);            // FastTwoSum(e_c, G) -> (r1, r2)（前提は付録 A の補題 R）
}

// 3 本組 acc_hi・acc_mid・acc_lo を 2 のべき rdx 倍する（その場更新．rdx の形は tfMulR と同じ）．
// 2 のべき倍は limb ごとに指数をずらすだけなので Dekker 分割が要らず，上の tfMulR で
// 残差を回収する部分がすべて恒等になる．残るのは正規化の TwoSum 鎖だけで，これは
// 落とせない：桁を引いたあとの上位 limb は小さくなっていて，中位 limb が相対的に
// 大きくなりうるため，掛けたあとに組を非重複へ戻しておかないと，次の桁を取り出す
// round が上位 limb だけを見て誤る（作業に term・tmp0..2・keep を使う．tfMulR と同じ）．
// 【tfMulR とビット単位で一致する条件】どの limb も 0 か，tfMulR の途中の値（limb の 4097 倍，Dekker で分けた半分の
// rdx 倍）が FP32 の正規化数の範囲に収まること，おおむね 2^-100 <~ |limb| かつ |上位| < 2^116 である．そのとき tfMulR の
// 分割の残差はすべて 0 になり，残る正規化の TwoSum 鎖が同じ順・同じ値で進む．外れると食い違う：上側（4097 倍が 2^128 を
// 溢れる）でも，下側（limb の指数が -149..-100．非正規化数の切り捨て）でも要素の 4 割前後（src/checks/tfmul_probe.cpp で DST の
// 3 本組を直接比べた．docs/nongemm_opt_results.md の §1）．
// 本番の値域は両側の内側にある．スライサ（尾崎 I の slice_tf_*．尾崎 II の oz2_digits.h）の 3 本組は正規化後の入口で
// 尾崎 I が |x| <= 1，尾崎 II が |x| <= 2^20（付録 A の H の帰納の入口と同じ）で，2^-100 を下回る要素（行の最大値より 2^100
// 以上小さい要素）はありうるが，その要素の桁はどちらの書き方でも
// 全部 0 になる．再構成の Horner（recon_tf_*）の limb は整数か 0 で，大きさは oz_horner_fits（ozaki1_format.hpp）が
// 2^116 より十分下に抑える（BF16 の 14080^3 で 2^74 程度）．
// 付録 A の証明の名前：y_i = β x_i，(s3, e4) = TwoSum(y1, y2)，(s5, e6) = TwoSum(y0, s3)，r0 = s5，
// 最後に FastTwoSum(e6, e4) -> (r1, r2)．
template <class W, class R>
ALWI void tfMulP2(W& w, R rdx) {
    using namespace dst;
    w.MulS(acc_hi, rdx); w.MulS(acc_mid, rdx); w.MulS(acc_lo, rdx);   // どれも無誤差
    TwoSum(w, acc_mid, acc_lo, term, tmp0, tmp2, keep);   // term = 中位+下位（s3），tmp0 = その誤差（e4）
    TwoSum(w, acc_hi, term, tmp1, tmp2, acc_mid, acc_lo); // tmp1 = 上位（s5），tmp2 = その誤差（e6）
    w.Cpy(tmp1, acc_hi);
    w.FinalFts(tmp2, tmp0, acc_mid, acc_lo, keep, term);  // 誤差 2 つを中位と下位へ分ける（FastTwoSum．前提は付録 A の補題 P）
}

// 基数（float のビット列）の仮数部が 0，すなわち 2 のべきか．基数倍の振り分け（sfpu_tf.h の tfMulRadix がコンパイル時に，
// 模擬が実行時に）はこの判定で tfMulP2 か tfMulR を選ぶ（本番の値域では結果はどちらでも同じ．上の tfMulP2 の一致の条件）
constexpr bool radix_is_pow2(uint32_t rdx_bits) { return (rdx_bits & 0x7FFFFFu) == 0u; }

// 3 本組 acc_hi・acc_mid・acc_lo にスカラ v（term）を足す（その場更新）．TwoSum の残差を順に
// 下位 limb へ伝播させるので，上位 2 本は個々の FP32 演算が 24 bit に丸めても
// 無誤差でいられる．最下位の 1 本だけは残差を受け取る先が無く，そこは近似である
// （下の Add(acc_lo, tmp2, acc_lo)）．作業 tmp0..2．keep は触らない．v は破壊される．
//   NL=3 が triple-float，NL=2 は FP32 2 本組（FP32 級のときだけ使う）．
template <uint32_t NL, class W>
ALWI void tfAddS(W& w) {
    using namespace dst;
    // TwoSum(a0, v)：和を acc_hi に，残差を tmp1 に
    w.Add(acc_hi, term, tmp0); w.Sub(tmp0, acc_hi, tmp1); w.Sub(tmp0, tmp1, tmp2);
    w.Sub(acc_hi, tmp2, tmp2); w.Sub(term, tmp1, tmp1); w.Add(tmp2, tmp1, tmp1); w.Cpy(tmp0, acc_hi);
    if constexpr (NL >= 3) {
        // TwoSum(a1, 残差)：和を acc_mid に，さらにその残差を tmp2 に
        w.Add(acc_mid, tmp1, tmp0); w.Sub(tmp0, acc_mid, tmp2); w.Sub(tmp0, tmp2, term);
        w.Sub(acc_mid, term, term); w.Sub(tmp1, tmp2, tmp2); w.Add(term, tmp2, tmp2); w.Cpy(tmp0, acc_mid);
        w.Add(acc_lo, tmp2, acc_lo);                       // 最下位はここで近似
    } else {
        w.Add(acc_mid, tmp1, acc_mid);
    }
}

// 小さい項の足し込み（scheme II の CRT の後ろの断片が使う）
// tfAddS の第 1 段 TwoSum(a0, v) を飛ばし，v を a1 の段から入れる．
// 15 演算 -> 8 演算．
//
// 値は落ちない．第 2 段は任意の大小関係で正しい Knuth の TwoSum なので，
// a1 + v は残差ごと厳密に保たれ，近似が入るのは tfAddS と同じく最下位への
// 足し込み Add(2, 6, 2) の 1 回だけである．したがって a0 の大きさについて
// 下からの仮定は要らない（相殺で a0 がいくら小さくなっても成り立つ）．
//
// 誤差の形も tfAddS と同じである．どちらも 1 回の呼び出しで生じる誤差は
// 最下位の丸め |fl(a2+e) - (a2+e)| <= 2^-24 |a2+e| だけで，違うのは a1 へ入る量である．
// tfAddS では第 1 段の残差（足し込み後の a0 の ulp/2 以下）が，tfAddSmall では v そのものが入る．
// 呼び出し側が |v| <= eps を保証すれば，1 回に入る量はどちらも max(ulp(a0)/2, eps) 以下になる．
// a1 は過去に入った量を保つので，n 回の後の上界はいまの a0 の ulp では抑えられず，
//   |a1| <= n * max(max_t ulp(a0^(t))/2, eps) * (1 + 2^-24)^n
// （a0^(t) は t 回目の足し込み後の a0，(1 + 2^-24)^n は a1 の更新の丸め）になる．
// eps を ulp(a0)/2 の大きさより十分小さく採るかぎり，この上界は tfAddS と同じである．
// scheme II では法ごとに整数部を落とすので |a0| < 2（ModRound の k_i は floor(m/2)+1 以内（下の ModRound の注記の
// (a')．ozaki2_moduli.h の oz2_crt_k_bound．対称剰余を 1 はみ出してもこの上界は変わらない））で ulp(a0)/2 <= 2^-24，eps は 7 bit の余裕を取って（最大項に対する相対で）2^-30 とする．
// この上界は緩い．実測では frac の最悪絶対誤差が tfAddS だけの場合と一致する
// （tools/twosum_probe.cpp．scripts/selftest.sh が毎回まわす）．
//
// 【第 2 段に FastTwoSum を使わない】Dekker の 3 演算にすると |a1| >= |v| が要るが，a0 の下からの評価が無いと言えない
// （前提を証明できないものを，別の場所の性質に頼って使い続けることはしない）．比べた frac の最悪絶対誤差と，frac の誤差が
// 端から端までの到達精度に出ない（したがって tools/twosum_probe.cpp で直接見る）理由は，docs/accuracy.md「SFPU の補題と累積誤差の検査」の付録 B．
template <uint32_t NL, class W>
ALWI void tfAddSmall(W& w) {
    using namespace dst;
    if constexpr (NL >= 3) {
        w.SmallTwoSum();                                   // TwoSum(a1, v)：下の SmallTwoSum（W のメンバを通す）
        w.Add(acc_lo, tmp2, acc_lo);                       // 最下位はここで近似
    } else {
        w.Add(acc_mid, term, acc_mid);
    }
}

// tfAddSmall の第 2 段 TwoSum(a1, v)：和を acc_mid に，残差を tmp2 に（作業 tmp0・tmp1．keep は壊さない）．
// tfAddSmall はこれを W のメンバ SmallTwoSum を通して呼ぶ（模擬が上の表の比較用の FastTwoSum へ差し替えるため．
// デバイスの SfpuDev::SmallTwoSum はこれをそのまま呼ぶ）
template <class W>
ALWI void SmallTwoSum(W& w) {
    using namespace dst;
    w.Add(acc_mid, term, tmp0); w.Sub(tmp0, acc_mid, tmp2); w.Sub(tmp0, tmp2, tmp1);
    w.Sub(acc_mid, tmp1, tmp1); w.Sub(term, tmp2, tmp2); w.Add(tmp1, tmp2, tmp2); w.Cpy(tmp0, acc_mid);
}

// --- 法での剰余 -------------------------------------------------------------
// d -= round(d/m)*m．保証するのは |d| < 2^22 の範囲だけで，そこでは結果が法 m の
// 対称剰余 [-m/2, m/2] に収まる（この範囲を出る呼び出しは下の【】にまとめた）．
// それを超えると二段階で崩れる：まず round が隣の整数を選び（下の理由），さらに
// round(d/m)*m が 2^24 を超えると積そのものが FP32 で丸まって，剰余類まで変わる
// （d=2^24, m=29 は -8 を返すが，正しい対称剰余は -9）．
// 奇数の法では端が届かないので実質 (-m/2, m/2) だが，偶数の法（254）では d/m が
// ちょうど半整数になり得て，Rnd の同点の向き次第で +m/2 と -m/2 の両方が出る．
// どちらも同じ剰余類で絶対値は m/2 なので後段（符号-絶対値 7 bit，再度の剰余）に影響しない．
// 2^22 が丸めの側から出てくる理由：fl(d*rinv) は d/m から相対 2^-23 ずれる一方，
// d/m と半整数の距離は 1/(2m) 以上あるので，|d|/m * 2^-23 < 1/(2m) すなわち |d| < 2^22 の
// あいだは round が必ず正しい整数を選ぶ．
// 【この一般の上界は，法が 4096 まである BF16 / BFP8 / FP16 / TF32 経路では足りない】
// 2^22 を出る呼び出しは 4 つある：recon_crt.cpp の (a) r_i*y_i（m^2/2 = 2^23 まで），
// (c) スライサの oz2_residue（ozaki2/kernels/oz2_digits.h．和が 2^23.9 まで），合成段（oz2_compose.cpp）の
// fold_first で桁積 P_j を法へ畳む ModRound と，recon_crt.cpp の (e) RAWMASK で 1 桁の法の GEMM の出力 P を畳む
// ModRound（どちらも |P| + m <= 2^24 をホストの法選択が課す．ozaki2_moduli.h）．recon_crt.cpp の
// (b) (chi mod m)*(2^16 mod m) + (clo mod m) は符号-絶対値 int32 の 2 平面復号で，INT8 経路
// （m <= 255）だけが通るので 2^15 未満に収まる．
// 剰余類が保たれるには |d| < 2^24 だけでは
// 足りず，積 round(d/m)*m も 2^24 以下で FP32 に厳密に載る必要がある（d=16777215, m=29 は
// 積 16777225 が 16777224 へ丸まり -9 を返す．正しくは -10）．round が隣の整数を選ぶのは d/m が
// 半整数から |d|*2^-23/m 以内のときだけで，そのとき積は d から m/2 + |d|*2^-23 しか離れない．(a)(c) は
// |d| <= 2^23.9 で 2^24 までの余白（約 1.1e6）が十分あり，合成と RAWMASK は |d| + m <= 2^24 なので，どれも
// |積| <= 2^24 に収まる．したがって剰余類は保たれる．対称剰余からのはみ出しは，上界では |d|*2^-23 未満
// （|d| + m <= 2^24 の合成と RAWMASK では 2 未満なので floor(m/2)+2 まで）としか言えないので，呼び出しごとに
// 次のとおり数える：
// tools/modround_check.cpp が (a) を既定の法集合で総当たりして |k| <= floor(m/2)+1 で判定し（既定の集合の係数では
// はみ出しは 0），(a') で経路が選びうる全法と m と互いに素な全 y について r·y を総当たりして同じ |k| <= floor(m/2)+1 で
// 判定する（floor(m/2) を 1 だけ超える積が 8170 通りある．例えば m=3475・r=1737・y=3474 は k=1738．上界 floor(m/2)+1 は
// 逆数の断片の小さい項の判定が数える範囲（ozaki2_moduli.h の oz2_crt_k_bound））．(b) を INT8 経路の
// 法で総当たりし，(c) を全経路の上限以下の法 2 以上のすべての法・3〜5 桁について半整数の境界近傍の総当たりと桁の上限で
// 調べて，|r| の最大がホストの解析の上界 oz2_resid_bound（はみ出し高々 1）以下であることも見る．(e) は 1 桁の法の
// 全部（法 2 以上）について |d| <= 2^24 - m の境界近傍と両端を総当たりし，|r| <= floor(m/2)+2 で見る（小さい法ほど
// 2 まではみ出す）．計約 23.8 億通り（scripts/selftest.sh が毎回まわす）．合成段の fold_first は総当たりしておらず，
// 上界のとおり大きい法でも 2 はみ出す（m=2005・d=-16425964 は -1004 = -(floor(m/2)+2) を返す）ので，
// ホストの法選択（ozaki2_moduli.h の compose_sum）と oz2_compose.cpp は畳んだ値を |t| <= m/2+2 で数える．
// 法や係数の選び方を変えたらこの検査を通し直すこと．s は作業用．
// o と d は同じでもよい（その場更新）．s は作業用で d とは別レジスタ．
template <class W>
ALWI void ModRound(W& w, uint32_t d, uint32_t o, uint32_t s, float m, float rinv) {
    w.Cpy(d, s); w.MulS(s, rinv); w.Rnd(s); w.MulS(s, m); w.Sub(d, s, o);
}
// 上の【】の 4 つは 2^24 の手前までの範囲を ModRound のまま扱う．|P| + m が 2^24 を超えうる合成段の
// fold_first=0 だけは，先に P を 2^12 の上位と下位へ分け，それぞれに「係数を法で畳んだ定数」を直接当てて
// から畳む（手続きは ozaki2/kernels/oz2_compose.cpp に書いてある）．

// 符号-絶対値の 8 bit 符号化 g(d) = d + [d<0]*(128 - 2d) ∈ [0,255]（その場更新）．
// 行列積カーネルが符号-絶対値の uint8 を期待するためで，scheme I・II のスライサが
// 共通で使う（実際のバイト表現は cb_out の DataFormat が決める）．w1, w2 は作業用．
template <class W>
ALWI void SignMag8(W& w, uint32_t d, uint32_t w1, uint32_t w2) {
    w.Cpy(d, w1); w.Ltz(w1);                                         // w1 = [d<0]
    w.Cpy(d, w2); w.MulS(w2, -2.f); w.AddS(w2, 128.f); w.Mul(w2, w1, w2);
    w.Add(d, w2, d);
}

}  // namespace sfpu_proc

// ===========================================================================
// 証明と比較実験：docs/accuracy.md「SFPU の補題と累積誤差の検査」（付録 A・B）．
