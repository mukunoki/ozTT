// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 尾崎スキーム II：オペランドから A' の 2^20 進 ND 桁を作る段（デバイス側）
// ---------------------------------------------------------------------------
// 役割：正規化前の FP64 オペランド x（FP32 の 3 本組で届く）を整数化した
//           A' ≈ X,  X = x / 2^e * 2^b          （|A'| <= 2^b）
//       の 2^20 進 ND 桁 d_{ND-1}, ..., d0 を作り，後段のモジュラースライサへ渡す．
//       ND はコンパイル時引数で，b <= 60 なら 3，80 以下なら 4，それを超えると 5 である．
//       A' と理想値 X の差は下の「桁が満たす条件」のとおり 1 以下である．
//       INT8 経路（modslice_int8.cpp）と桁分割の 7 経路（modslice_digit.cpp）が
//       共用する．
//
// 手続き
//   正規化は乗算ではなく指数への加算で入る．行ごと（A）／列ごと（B）の
//       d = (b - (ND-1)G) - e     （G = 桁のビット幅 20，2^e = 行／列の最大値の
//                                   2 のべきへの切り上げ）
//   を expdelta.cpp が Int32 タイルへ展開し，slice_f64_reader.cpp が FP64 の指数
//   フィールドへ整数で足したうえで 3 本組にする（slice_tf_core.h の tf_load_f64）．
//   したがって計算側に届く時点で
//       T = x / 2^e * 2^(b-(ND-1)G) = X / 2^((ND-1)G)
//   になっており，あとは scheme I のスライサ（ozaki1/kernels/slice_tf_num.cpp．本体は slice_tf_body.h）と
//   同じ形で「丸めて残差を引く」を ND 回，その間に「基数 2^G 倍」を ND-1 回
//   はさめばよい（初回の基数倍は上の d に畳んである）：
//       d2 = round(T) ; T -= d2 ; T *= 2^G
//       d1 = round(T) ; T -= d1 ; T *= 2^G
//       d0 = round(T)
//   入力の受け取り・正規化・残差の持ち回りは 3 つのスライサで共通なので
//   slice_tf_core.h にある．
//
// 桁が満たす条件
//   基数が 2 のべきなので基数倍（tfMulP2）は無誤差であり，残差が桁に落ちる誤差は入らない．
//   丸めるのは 3 本組の上位 limb だけなので，中位以下が丸めを左右する境目では A' が X の
//   最近接整数の隣になりうる．したがって保証できるのは |A' - X| <= 1 であって
//   最近接性ではない．CRT の窓が要求するのは |A'| <= 2^b のほうで，これは
//   |x/2^e| <= 1 から従う．窓を保つことと精度に影響しないことは別で，A' と X の差は整数化の
//   誤差としてそのまま出力に入る（上界は最近接丸めの 1/2 ではなく 1．b=54・行の最大値 1 の行の
//   x = 2^-55 + 2^-80 は X = 1/2 + 2^-26 で A' = 0．ozaki2_concat_check.hpp の「独立な整数参照」の注記）．
//   桁は平衡表現（符号つき）である：最上位は |T| <= 2^(b-(ND-1)G)（b が桁数いっぱいの
//   80 や 100 でも 2^20），それ以外は 2^19 + 2^16 + 1 以下である（丸めるのは上位 limb だけなので，残差は
//   1/2 に中位以下の limb の分（2^-4 以下）を足したものになる．導き方は ozaki2_split.h の oz2_resid_bound の D0）．
//   どれも FP32 で無誤差に表せ，後段の ModRound が要求する 2^22 の範囲にも収まる．合成
//   A' = Σ_j d_j*2^(jG) は法について線形なので，桁の符号はそのまま通る．
//
// バッファ
//   入力  slice_tf_core.h の 4 つ（limb の生データ 3 つとスケール s）
//   作業  slice_tf_core.h の残差 2 つ
//   出力  c_26 = 上位の桁を作った順（d_{ND-1} から d1 まで）  c_27 = 桁 d0

#include <cstdint>
#include "../../common/kernels/slice_tf_core.h"
#include "oz2_args.h"   // 引数の並びと CB 番号（oz2_limits.h の上限も）

constexpr auto cb_dhi = tt::CBIndex(oz2_args::CB_DIGIT_HI);   // 出力：上位の桁 d_{ND-1}..d1（作った順）
constexpr auto cb_dlo = tt::CBIndex(oz2_args::CB_DIGIT_LO);   // 出力：桁 d0

// コンパイル時引数 SC_RDX = 桁の基数 2^G を float のビット列にしたもの
// （SC_S は取り込む側のカーネルが法の本数 S に使う）
constexpr uint32_t OZ2_RDX_B  = get_compile_time_arg_val(oz2_args::SC_RDX);
// 桁数（コンパイル時引数 SC_NDIGIT）．N 桁で N*G bit まで運べる．ホストが b から決めて渡す：
// 既定の b<=54 なら 3 桁，動的域の広い入力で b を 60 より上げるときだけ 4 桁，80 より上げるときだけ
// 5 桁になる（ozaki2_plan.hpp の choose_moduli_and_width）．出力 CB の深さも桁数に合わせる必要がある（下の oz2_make_digits は 1 枚ずつ
// 押すが，cb_dhi の ND-1 枚を一組として待ち，添字で読み，捨てる）．
constexpr uint32_t OZ2_NDIGIT = get_compile_time_arg_val(oz2_args::SC_NDIGIT);
static_assert(OZ2_NDIGIT >= 3 && OZ2_NDIGIT <= (uint32_t)OZ2_MAXDIGIT, "A' の桁数は 3〜5（oz2_residue が扱う範囲）");

// 入力タイル 1 枚を消費し，桁 ND 枚を cb_dhi（ND-1 枚）と cb_dlo（1 枚）へ押し出す．
// 戻った時点で全枚 cb_wait_front 済みなので，そのまま oz2_digits_copy できる．
ALWI void oz2_make_digits() {
    tf_wait_input();
    for (uint32_t i = 0; i < OZ2_NDIGIT; i++) {
        tile_regs_acquire();
        init_sfpu(cb_tf_hi, cb_dhi);
        if (i == 0) tf_load_f64();                    // T = FP64 から組んだ 3 本組（正規化込み）
        else      { tf_pop_residual(); tfMulP2(OZ2_RDX_B); }   // 基数は 2^20（2 のべき）
        Cpy(0, 7); Rnd(7);          // 桁 d = round(T)
        Sub(0, 7, 0);               // 残差 T -= d（|T| <= 1/2 なので無誤差）
        tile_regs_commit();
        tile_regs_wait();
        // 桁の出力先．上位の桁と d0 で CB を分けるのは，後段が上位の桁と d0 を
        // 別々に添字で読むためで，残差を 2+1 に分ける理由とは別である．
        const auto cb_d = (i + 1 < OZ2_NDIGIT) ? cb_dhi : cb_dlo;
        cb_reserve_back(cb_d, 1);
        pack_reconfig_data_format(cb_d);
        pack_tile(7, cb_d);
        cb_push_back(cb_d, 1);
        if (i + 1 < OZ2_NDIGIT) tf_push_residual();   // 次の桁のために残差を退避
        tile_regs_release();
    }
    tf_pop_input();
    cb_wait_front(cb_dhi, OZ2_NDIGIT - 1);
    cb_wait_front(cb_dlo, 1);
}

// 作った ND 桁を DST へ読み込む（d0 -> r0，d1 -> r1，d2 -> r2，d3 -> r3，d4 -> r4）．DST の内容は
// acquire/release をまたいで保持されないので，法ごとに読み直す必要がある．
// cb_dhi には作った順（上位から）に ND-1 枚入っているので，添字は逆順に読む．
ALWI void oz2_digits_copy(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4) {
    copy_tile_to_dst_init_short(cb_dlo); copy_tile(cb_dlo, 0, r0);   // d0（最後に作った桁）
    copy_tile_to_dst_init_short(cb_dhi);
    copy_tile(cb_dhi, OZ2_NDIGIT - 2, r1);                           // d1
    copy_tile(cb_dhi, OZ2_NDIGIT - 3, r2);                           // d2
    if constexpr (OZ2_NDIGIT >= 4) copy_tile(cb_dhi, OZ2_NDIGIT - 4, r3);   // d3
    if constexpr (OZ2_NDIGIT >= 5) copy_tile(cb_dhi, OZ2_NDIGIT - 5, r4);   // d4
}
// 入力タイル 1 枚ぶんの桁を使い終えたら捨てる
ALWI void oz2_digits_pop() { cb_pop_front(cb_dhi, OZ2_NDIGIT - 1); cb_pop_front(cb_dlo, 1); }

// 桁から法 m の対称剰余 r ≡ A' (mod m) を作り，reg2 に置く（作業用 reg0..7）．
//   A' mod m = ( ... + (d2 mod m)*c1 + (d1 mod m)*c0 + d0 ) mod m
// 重みの掛かる桁だけを先に畳む．d0 は重みが 1 なので畳まずに足してよい．
// 大きさ：|d0| <= D0 = 2^19 + 2^16 + 1（丸めた残差の 2^20 倍．上位 limb だけを丸めるので 2^19 を中位以下の limb の
// 分だけ超える．ozaki2_split.h の oz2_resid_bound），畳んだ桁は |t_j| <= floor(m/2)，
// c[j] はホストの pow2mod が返す [0, m) の値（対称ではない）なので，最後の ModRound の入力は
//     |S| <= D0 + floor(m/2)*(c[0]+c[1])
// である．法が 4096 まで伸びる BF16・BFP8・FP16・FP32 の経路では 2^22 を超え，m=3913 で
// 1.55e7（2^23.9）に達する．sfpu_tf_proc.h の ModRound の一般の保証 |S| < 2^22 の外だが，
// 次の 2 点で剰余類は正しく保たれる．
//   ・部分和・t_j*c[j]・商と法の積 round(S/m)*m はどれも整数で 2^24 以下なので，
//     FP32 で無誤差である（剰余類はここで決まる）．
//   ・fl(S*rinv) のずれは |S|*2^-23/m 以下なので，round が隣の整数を選んでも
//     |r| <= m/2 + |S|*2^-23 < m/2 + 2 にとどまる．
// 4 桁のときは d3 の項まで足すと 2^24 を超える法がある（m=3605 などの奇数．末尾の
// 選び直しで引きうる）ので，d3 の項を足す前に 1 度畳む．3 桁（b <= 60，既定）は通らない．
// 5 桁（b > 80）では，畳んだ値に d4 の項（係数 c3）を足してもう 1 度畳み，最後に d3 の項を足す．
// 最後の ModRound の入力は 4 桁と同じ「畳んだ値 + t3*c2」の形になるので，|r| の上界（ホストの
// oz2_resid_bound）は 4 桁と変わらない．d4 の項を後に回すと上界に c3 が入り，法によっては 1 増える．
// 保証するのは「剰余類が正しく，|r| <= floor(m/2)+1」であって，対称剰余 (-m/2, m/2] そのもの
// ではない．後段が頼るのは，INT8 経路では |r| <= 127（符号-絶対値 7 bit），桁分割経路では
// 桁が上限（経路の表 ozaki2_split.h の OZ2_ROUTES の digit_bound）に収まることである．1 の超過が出るのは |S| が 2^22 を超える
// 大きい法（BF16 などの経路）だけで，その超過を入れても桁は 32 に収まる．
// tools/modround_check.cpp の (c) が，全経路の上限以下のすべての法・3〜5 桁について，
// 剰余類と桁の上限を確かめている（scripts/selftest.sh が毎回まわす）．
// 対称剰余のまま返すのは，桁分割経路がここから桁分割を始めるとき桁の絶対値の上限が
// 半分になるからで，INT8 経路の符号-絶対値への符号化にとっても都合がよい．
//   rinv = 1/m，c[j] = 2^((j+1)*G) mod m（ホストが算出．c[0]=2^20, c[1]=2^40, c[2]=2^60, c[3]=2^80）
ALWI void oz2_residue(float m, float rinv, const float* c) {
    // reg2 = d0（重み 1），reg7/reg1/reg0/reg3 = d1/d2/d3/d4（重みつき）．reg4 は ModRound の作業用
    oz2_digits_copy(2, 7, 1, 0, 3);
    ModRound(7, 4, m, rinv); MulS(7, c[0]); Add(2, 7, 2);
    ModRound(1, 4, m, rinv); MulS(1, c[1]); Add(2, 1, 2);
    if constexpr (OZ2_NDIGIT >= 4) {                        // 先に畳まないと 2^24 を超える
        ModRound(2, 4, m, rinv);
        if constexpr (OZ2_NDIGIT >= 5) {                    // d4 の項を足してもう 1 度畳む（上の説明）
            ModRound(3, 4, m, rinv); MulS(3, c[3]); Add(2, 3, 2);
            ModRound(2, 4, m, rinv);
        }
        ModRound(0, 4, m, rinv); MulS(0, c[2]); Add(2, 0, 2);
    }
    ModRound(2, 4, m, rinv);                                // |r| <= floor(m/2)+1 へ
}
