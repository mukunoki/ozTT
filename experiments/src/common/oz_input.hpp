// SPDX-License-Identifier: Apache-2.0
#pragma once
// 入力 A・B の生成（host_eval.hpp の傘の下．尾崎の 2 つのドライバ・基準実装 gemm_bf16・gemm_bf16ext と CPU の DGEMM の参照
// tools/dgemmcheck が同じ関数で作る）
// ---------------------------------------------------------------------------
// オペランドは FP64 そのものである．FP64 GEMM のエミュレーションを評価する以上，入力も FP64 でなければ意味がない．
// 同じ seed なら，どのドライバもビット単位で同一の A, B を見る．入力・参照解・精度の指標を共有するので，尾崎の両スキームと
// 基準実装を同じ条件で比べられる（ドライバごとに違うのはデバイスのプログラムだけ）．
//
// 環境変数（値は host_eval.hpp の check_env がデバイスを開く前にも検査する）
//   OZ_PHI=φ      指数部レンジを φ で振る標準的な入力（下の gen_operands を参照）
//   OZ_WIDE=R     オペランドの動的域を約 R bit まで広げてから使う
//   OZ_CONST=v    全要素を v にする（v > 0．1 は全要素 1.0．分解の手計算と突き合わせるときや，
//                 遅延再構成の S_d を上界に届かせるとき．scripts/tests/check_nsmall.sh）
//   OZ_CONST_SIGN=col|half|halfeps  OZ_CONST のとき B の符号の付け方（下の gen_operands）
//   OZ_POS=lo     全要素を [lo,1] の正の乱数にする．符号の相殺が起きないので
//                 無誤差条件を破っているかどうかがそのまま精度に出る
//                 （docs/kblock_plan.md の再現手順で使う）
//   OZ_ROWSCALE=R 行ごと・列ごとに約 R bit の範囲でスケールをばらつかせる
//   OZ_EXPDIFF=d  各内積を「1.0 と 2^-d の 2 項だけ」にする（累算経路の縮約幅を測る
//                 マイクロベンチ．2 項目の位置は OZ_EXPK=k で選ぶ．既定 1）
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "oz_env.hpp"   // つまみの読み方（同じディレクトリから相対で取り込む理由は host_eval.hpp の冒頭）

namespace oz_eval {

// 入力生成・判定のつまみ（OZ_PHI など）は oz_env.hpp の読み方で読む．atoi／atof のように打ち間違いを 0 へ倒すと，
// 別の実験条件（φ=0 など）を黙って測ってしまうので，文字列全体が [lo,hi] の有限な数として読めなければ止める．
// 範囲はスクリプトが使う値（φ<=30，WIDE 40，ROWSCALE 20 など）より十分広く取り，生成が破綻する値（OZ_POS の lo>=1 など）
// だけを弾く．削除したつまみ・GEMM 段の探針・ドライバが読まないつまみの拒否は oz_knobs.hpp にある（つまみの表から作る．
// ここがその表を取り込まない理由は host_eval.hpp の冒頭）．

// 入力生成のつまみの値（読むのは input_env だけで，範囲もここにしか書かない．check_env がデバイスを開く前にも読んで検査し，
// gen_operands・input_desc は使う所で読み直す）
struct InputEnv {
    bool expdiff = false; int d = 0; int k2 = 1;      // OZ_EXPDIFF=d，OZ_EXPK=k2
    bool phi = false;     double phiv = 0.0;          // OZ_PHI
    bool cst = false;     double c = 1.0;             // OZ_CONST
    std::string sign;                                 // OZ_CONST_SIGN（空 = 付けない）
    bool pos = false;     double lo = 0.0;            // OZ_POS
    bool wide = false;    double R = 0.0;             // OZ_WIDE
    bool rows = false;    double Rs = 0.0;            // OZ_ROWSCALE
};
inline InputEnv input_env() {
    InputEnv e;
    // d は 2^-d が FP64 の正規数に収まる範囲に限る（-d の符号付きあふれも防ぐ）．
    if (const char* v = oz_env::get("OZ_EXPDIFF")) { e.expdiff = true; e.d = (int)oz_env::parse_i64("OZ_EXPDIFF", v, -1000, 1000); }
    if (const char* v = oz_env::get("OZ_EXPK")) e.k2 = (int)oz_env::parse_i64("OZ_EXPK", v, 0, 0x7FFFFFFF);   // K との関係は gen_operands
    if (const char* v = oz_env::get("OZ_PHI")) { e.phi = true; e.phiv = oz_env::parse_f64("OZ_PHI", v, 0.0, 100.0); }
    if (const char* v = oz_env::get("OZ_CONST")) { e.cst = true; e.c = oz_env::parse_f64("OZ_CONST", v, 1e-300, 1e300); }
    static const char* const kSigns[] = {"col", "half", "halfeps"};
    if (const int i = oz_env::choice("OZ_CONST_SIGN", {kSigns[0], kSigns[1], kSigns[2]}, -1); i >= 0) e.sign = kSigns[i];
    // lo<1 でないと分布が組めない
    if (const char* v = oz_env::get("OZ_POS")) { e.pos = true; e.lo = oz_env::parse_f64("OZ_POS", v, 0.0, 0.999); }
    if (const char* v = oz_env::get("OZ_WIDE")) { e.wide = true; e.R = oz_env::parse_f64("OZ_WIDE", v, 0.0, 1000.0); }
    if (const char* v = oz_env::get("OZ_ROWSCALE")) { e.rows = true; e.Rs = oz_env::parse_f64("OZ_ROWSCALE", v, 0.0, 500.0); }
    return e;
}

// OZ_EXPDIFF の第 2 項の位置 OZ_EXPK と K の関係：1 以上 K 未満（K >= 2）．k2 はそのまま添字に使うので，外れるとホストの
// ヒープを壊す．k2=0 は大きい項を上書きして 2 項にならない．OZ_EXPDIFF が無ければ見ない．外れていれば理由を出して
// exit(2)（gen_operands と，デバイスを開く前の host_eval.hpp の check_env が同じこれを呼ぶ）
inline void check_expk(const InputEnv& e, uint32_t K) {
    if (e.expdiff && (K < 2 || e.k2 < 1 || (uint32_t)e.k2 >= K)) {
        std::fprintf(stderr, "ERROR: OZ_EXPK は 1 以上 K 未満でなければならない（OZ_EXPK=%d K=%u）\n", e.k2, K);
        std::exit(2);
    }
}

// A（M×K）と B（K×N）を行優先・FP64 で生成する．
//
// 既定は [-1,1] の一様乱数である．指数部レンジはほぼ広がらないので，尾崎スキームの
// スライス数が最も少なくて済む「易しい」入力にあたる．
//
// OZ_PHI=φ：尾崎スキームの評価で標準的に使われる
//     a_ij = (rand - 0.5) * exp(φ * randn)      rand ~ U(0,1), randn ~ N(0,1)
// で生成する（Mukunoki らの ISC 2020 論文，Uchino らの GEMMul8 と同じ形）．φ が
// 指数部レンジを決める：正規分布に従うのは乗じる因子 exp(φ randn) の自然対数（標準偏差 φ）で，要素の絶対値の
// 対数は log|rand - 0.5| + φ randn である（φ=0 でも一様乱数のぶんだけ散らばる）．因子の log2 は標準偏差
// φ/ln2 ≈ 1.44φ bit で散らばり，±1σ の幅で 2φ/ln2 ≈ 2.9φ bit になる（行列の中の最大と最小の比は要素の数にも依り，
// これより広い）．φ=0 は一様乱数（既定とほぼ同じ）．
//
// スライス分解はデータ依存で，行内の最大値から大きく下回る要素ほど多くのスライスを
// 要するため，FP64 相当の精度に必要なスライス数 p は φ とともに増える．固定の p で
// 走らせれば逆に到達精度が落ちる．この依存を測るための入力である．
//
// 環境変数の優先順位（複数立てたとき．使わないものも値の検査は input_env が行う）：
//   1. OZ_EXPDIFF  立っていればそれだけで決まり，以下はすべて使わない
//   2. OZ_PHI      同上（OZ_EXPDIFF が無いとき）
//   3. OZ_CONST    全要素 OZ_CONST の値（1 なら 1.0．B の符号は OZ_CONST_SIGN）で確定し，
//                  OZ_POS・OZ_WIDE・OZ_ROWSCALE は無視される
//   4. OZ_POS      一様乱数を [lo,1] の正の乱数に置き換える
//   5. OZ_WIDE → OZ_ROWSCALE  この順に掛けて重ねる（既定の一様乱数にも OZ_POS にも掛かる）
// 乱数の seed の既定値（どのドライバもこれで生成する．CONFIG 行の seed 欄にも出す）
constexpr uint32_t kDefaultSeed = 7;
// gen_operands の版の名前（尾崎の CONFIG 行の gen= 欄）．どの作り方でも，同じ環境と seed から作る A・B のビット列を変えたら
// 名前を変える（照合 scripts/rules/probe_checks.py の PHI_GEN は，φ の入力 a=(u-1/2)exp(φz) をこの関数の φ の枝で作る版の
// 名前の集合で，φ の掃引のログの gen= がそこに入ることを要る．φ の枝を変えずに名前を変えたら PHI_GEN にも足す）．
// 乱数は std::mt19937 と libstdc++ の分布の実装に依る（tools/dgemmcheck は g++ 12.3 で作る．lib/peer_logs.py の DGEMM_CXX）．
// 乱数を引く順はこの関数の文の順で決め，1 つの式の中で同じ rng を 2 度進めない（被演算子の評価の順は C++ で決まらない）
constexpr const char* kGenVersion = "ozgen1";
inline void gen_operands(std::vector<double>& A, std::vector<double>& B,
                         uint32_t M, uint32_t K, uint32_t N, uint32_t seed = kDefaultSeed) {
    const InputEnv e = input_env();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    A.resize((size_t)M * K);
    B.resize((size_t)K * N);
    // OZ_EXPDIFF=d：各出力要素の内積を 1.0 + 2^-d の 2 項だけにする．
    // 第 2 項を置く k を OZ_EXPK で選べる（既定 1）：行列エンジンは 1 命令で内積長 16 を
    // K 方向に 8 項ずつ区切った 2 組として処理し，組ごとに積を最大の指数に揃えて丸める（論文の
    // 8 項の指数合わせ）．k=1〜7 なら 2 項は同じ組に入り，k≥8 なら別の組（k≥16 なら別の命令）に入る．
    // 同じ和を「組の中」と「組の間」で計算させて分けられるので，丸めが組の中の揃えで
    // 起きているのか後段の FP32 累算器で起きているのかを切り分けられる．
    // 2^-d が BF16（と FP32）の正規数になるのは -127 <= d <= 126 で，その外も d <= 133 までは BF16 の非正規数として，
    // d <= 149 までは FP32 の非正規数として厳密に表せる（float にして 0 になるのは d >= 150，inf になるのは d <= -128）．
    // input_env はその外の d も受け付ける（FP64 の範囲で検査する）．基準の BF16 GEMM（gemm_bf16・gemm_bf16ext）は
    // 入力を float（と BF16）にしてデバイスへ渡すので，d >= 127 の小さい項は非正規数で渡り，デバイスが非正規数を
    // 0 に潰すかどうか（確かめていない）でも消えうる．行列エンジンの指数合わせの測定には d <= 126 を使う（論文の
    // 測定は d <= 24 程度）．
    if (e.expdiff) {
        const int d = e.d, k2 = e.k2;
        check_expk(e, K);   // k2 はそのまま添字に使う（check_expk の注記）
        std::fill(A.begin(), A.end(), 0.0);
        std::fill(B.begin(), B.end(), 0.0);
        for (uint32_t i = 0; i < M; i++) {
            A[(size_t)i * K + 0]  = 1.0;              // 大きい項
            A[(size_t)i * K + k2] = std::exp2(-d);    // 小さい項（2 のべき．d <= 133 なら BF16 で厳密．上の注）
        }
        for (uint32_t j = 0; j < N; j++) {
            B[(size_t)0 * N + j]  = 1.0;
            B[(size_t)k2 * N + j] = 1.0;
        }
        return;   // 厳密な内積は 1 + 2^-d
    }
    if (e.phi) {
        const double phi = e.phiv;
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::normal_distribution<double> nrm(0.0, 1.0);
        // 1 要素ごとに一様乱数 u を先に，正規乱数 z を後に引く（同じ rng を進める 2 つの呼び出しを 1 つの式の 2 つの
        // 被演算子に書くと，評価の順は C++ では決まらず，コンパイラの版や最適化で入力が黙って変わりうる．この順は
        // 以前の 1 式の書き方を g++ 12.3 が評価していた順で，入力のビット列は前の版と同じ）
        auto phi_elem = [&] {
            const double u = u01(rng);
            const double z = nrm(rng);
            return (u - 0.5) * std::exp(phi * z);
        };
        for (auto& v : A) v = phi_elem();
        for (auto& v : B) v = phi_elem();
        return;
    }
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // OZ_CONST=1：A,B を全要素 1.0 にする（調査用）．分解が手計算で追える入力で，
    // 対角ごとの最悪ケース (ℓ+1)*K*d^2 を試す入力ではない：scheme I の既定では 1 は 1/2 へ
    // 正規化され，スライスは先頭桁だけ（BF16 なら (32,0,0,…)）になるので，
    // 部分和は P_0 = K*32^2，ℓ>0 では 0 である．
    // 符号の相殺は起きないので，先頭の対角が累算の予算を超えるかは直接調べられる．
    // 値を指定できる：OZ_CONST=v（v > 0．1 なら上の全要素 1.0）．例えば 0.4921875 = 63/128 は BF16 の既定
    // （基数 64，1/2 へ正規化）で桁が (32, -32, 0, …) になり，遅延再構成の S_0 = K*32^2 と
    // S_1 = -2*K*32^2 が対角ごとの上界 (d+1)*K*d_max^2 にちょうど届く（ozaki1/ozaki1_plan.hpp の recon_nsmall の
    // 境目を試す入力．docs/nongemm_opt_results.md の §3）．
    // OZ_CONST_SIGN で B の符号を変えられる：col は奇数列を負に（行と列で符号の違う出力が混じる），
    // half は k >= K/2 の行を負に（K が偶数なら C は厳密に 0．途中の Int32 の和は K/2 ぶんまで振れる），
    // halfeps は half に加えて最後の行だけ大きさを (1 - 2^-30) 倍にする（途中の和や各項に比べて桁違いに小さい出力．
    // その行の値は FP64 で丸めた RN(v(1 - 2^-30)) なので，K が偶数なら C = v (v - RN(v(1 - 2^-30)))．v の仮数が 23 bit
    // 以下なら丸めは起きず C = v^2 2^-30．一般の v では違う：v = 1 + 2^-52 なら括弧の中がちょうど 2^-30 で C = v 2^-30）．
    if (e.cst) {
        const double c = e.c;
        for (auto& v : A) v = c;
        for (auto& v : B) v = c;
        if (!e.sign.empty()) {
            const std::string& m = e.sign;
            for (uint32_t k = 0; k < K; ++k)
                for (uint32_t j = 0; j < N; ++j) {
                    double& b = B[(size_t)k * N + j];
                    if (m == "col" && (j & 1)) b = -b;
                    if ((m == "half" || m == "halfeps") && k >= K / 2) b = -b;
                    if (m == "halfeps" && k + 1 == K) b *= 1.0 - std::ldexp(1.0, -30);
                }
        }
        return;
    }
    // OZ_POS=lo：全要素を [lo,1] の正の一様乱数にする（調査用）．符号の相殺が
    // 起きないので対角の部分和が最悪ケースに近づき，かつ値がばらつくので
    // 「全要素同一」のような退化した並びにもならない．
    if (e.pos) {
        std::uniform_real_distribution<double> u(e.lo, 1.0);
        for (auto& v : A) v = u(rng);
        for (auto& v : B) v = u(rng);
    }
    // OZ_WIDE=R：各要素に 2^{-R*u}（u は [0,1] の一様乱数）を掛け，行内／列内の
    // 動的域を約 R bit まで広げる．必要なスライス数がオペランドの動的域にどう
    // 依存するかを調べるためのもの（スライス分解はデータ依存で，行ごとの最大値から
    // 大きく下回る要素ほど多くのスライスを要する）．
    if (e.wide) {
        const double R = e.R;
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        for (auto& v : A) v *= std::exp2(-R * u01(rng));
        for (auto& v : B) v *= std::exp2(-R * u01(rng));
    }
    // OZ_ROWSCALE=R：A の各行と B の各列に 2^{[-R,R] の一様乱数} を掛け，行ごと・
    // 列ごとの大きさを揃わなくする．これは正規化（行ごと／列ごとのスケール算出）が
    // 正しく働いているかを見るためのもので，スケールが正しければ到達精度は変わらない．
    // 逆にスケールを持たない実装（|x| <= 1 を前提にした実装）はここで破綻する．
    // 2 のべきではなく実数倍にしてあるので，入力そのものは特別な形にならない．
    if (e.rows) {
        const double R = e.Rs;
        std::uniform_real_distribution<double> ur(-R, R);
        for (uint32_t m = 0; m < M; m++) {
            double f = std::exp2(ur(rng));
            for (uint32_t k = 0; k < K; k++) A[(size_t)m * K + k] *= f;
        }
        for (uint32_t n = 0; n < N; n++) {
            double f = std::exp2(ur(rng));
            for (uint32_t k = 0; k < K; k++) B[(size_t)k * N + n] *= f;
        }
    }
}

}  // namespace oz_eval
