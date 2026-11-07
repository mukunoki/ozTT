// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎スキーム II：幅 b と再構成のパス数を入力から選ぶ誤差のモデル（OZ2_FP64・OZ2_PASSES=auto）
// ---------------------------------------------------------------------------
// scheme I の ozaki1_autop.hpp（スライス数 p を入力から選ぶ OZ_FP64）に当たる．ホストだけで完結し，デバイスにも
// tt-metal にも触らない（計時区間の外）．計画（ozaki2_plan.hpp の choose_moduli_and_width）が呼ぶ．
// 依存：標準ヘッダ，oz_normest.hpp（||C||_F の見積り），ozaki2_moduli.h（法の集合 Ozaki2Moduli と幅の定数）．どれも tt-metal に
// 依存しないので，ホスト単体の道具（tools/）からも取り込める（今このモデルを単体で検査する道具は無い）．
// 環境変数は読まない（OZ2_NOWIN などは呼び出し側が引数で渡す）．
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "common/oz_normest.hpp"   // oz_pow2_scale・estimate_product_norm（scheme I の自動決定と共有）
#include "ozaki2/ozaki2_moduli.h"

// --- 入力の統計（OZ2_FP64 と OZ2_PASSES=auto が使う実測）-----------------------
// どちらもオペランドを走査するだけで，デバイスも乱数以外の状態も要らない．
// 「決める」「印字する」（下の choose_precision_policy）から切り離してある．

// 行（A）／列（B）ごとの 2 のべきスケール tau の二乗和，オペランドの Frobenius ノルム，
// 正規化後の行／列 2 ノルムの最大（CRT の窓に使う）．
struct Oz2OperandStats {
    double TA = 0, TB = 0;   // sum tau^2
    double nA = 0, nB = 0;   // ||A||_F^2, ||B||_F^2（2 のべきで正規化した後）
    double wA = 0, wB = 0;   // max_i ||a~_i||_2, max_j ||b~_j||_2
};

static inline Oz2OperandStats measure_operand_stats(const std::vector<double>& A,
                                                    const std::vector<double>& B,
                                                    uint32_t M, uint32_t N, uint32_t K,
                                                    double rgA, double rgB) {
    Oz2OperandStats st;
    for (uint32_t r = 0; r < M; r++) {
        double mx = 0, s2 = 0;
        for (uint32_t c = 0; c < K; c++) {
            const double v = A[(size_t)r * K + c] * rgA;
            mx = std::max(mx, std::fabs(v)); s2 += v * v;
        }
        // 零行はスケールが未定義（デバイス側は 1 を使う）だが，量子化誤差も厳密に 0 な
        // ので誤差モデルには 0 で入れる．1 で数えると零行の多い行列で kappa2 が膨らむ．
        if (mx > 0) {
            const double t = std::exp2(std::ceil(std::log2(mx)));
            st.TA += t * t; st.wA = std::max(st.wA, std::sqrt(s2) / t);
        }
        st.nA += s2;
    }
    for (uint32_t c = 0; c < N; c++) {
        double mx = 0, s2 = 0;
        for (uint32_t r = 0; r < K; r++) {
            const double v = B[(size_t)r * N + c] * rgB;
            mx = std::max(mx, std::fabs(v)); s2 += v * v;
        }
        if (mx > 0) {
            const double t = std::exp2(std::ceil(std::log2(mx)));
            st.TB += t * t; st.wB = std::max(st.wB, std::sqrt(s2) / t);
        }
        st.nB += s2;
    }
    return st;
}

// ||C||_F^2 の見積り estimate_product_norm と，二乗和を組む前の 2 のべき oz_pow2_scale は，
// scheme I の自動決定（OZ_FP64）と共有するので common/oz_normest.hpp に置いた．

// 自動決定の結果．
struct Oz2PrecisionPolicy {
    int      b_cap  = OZ2_B_DEFAULT;   // 幅 b の上限（ozaki2_moduli.h）
    // CRT の窓の実測．測っていなければ w_measured=false で，最悪ケース log2 K を使う．
    // 「測ったか」を w の符号で表さない：実測の log2 W は負になりうる（ozaki2_moduli.h）．
    bool     w_measured = false;
    double   wA = 0, wB = 0;  // max_i ||a~_i||_2, max_j ||b~_j||_2
    uint32_t s_req  = 0;      // 法の本数の要求
    // 入力の統計（oz2_measure_stats）．再構成のパス数の見積り（oz2_recon1_log2）にも使う
    bool     stats = false;
    Oz2OperandStats st;
    double   nC = 0;          // ||C||_F^2 の見積り（estimate_product_norm．安全率込み）
};

// 入力の統計を測る（choose_precision_policy と，OZ2_FP64 なしで OZ2_PASSES=auto のときに呼ぶ）．
// kappa2 もパス数の見積りも A,B の絶対スケールに依らない量なので，先に 2 のべき（無誤差）で正規化
// してから測る（そのまま二乗和を組むと |a| ~ 1e150 で桁溢れする）．
static inline void oz2_measure_stats(const std::vector<double>& A, const std::vector<double>& B,
                                     uint32_t M, uint32_t N, uint32_t K, Oz2PrecisionPolicy& p) {
    const double rgA = 1.0 / oz_pow2_scale(A), rgB = 1.0 / oz_pow2_scale(B);
    p.st = measure_operand_stats(A, B, M, N, K, rgA, rgB);
    p.nC = estimate_product_norm(A, B, M, N, K, rgA, rgB);
    p.stats = true;
}

// CRT の窓が覆うべき上界の log2．幅 b に依る．
// スライサが保証するのは |A' - X| <= 1（X = 2^b a~．oz2_digits.h）で，|A'| <= 2^b|a~| + 1 である．
// 行ベクトルの 2 ノルムでは ||A'_i|| <= 2^b ||a~_i|| + sqrt(K) なので，Cauchy--Schwarz から
//     |C'_ij| <= (2^b wA + sqrt K)(2^b wB + sqrt K)
//             = 2^(2b) W (1 + sqrt K/(2^b wA))(1 + sqrt K/(2^b wB)),   W = wA*wB
// となる．零でない行・列は最大要素が (1/2, 1] に正規化されるので wA, wB >= 1/2 で，
// 係数は (1 + 2^(1-b) sqrt K)^2 以下である．b=54 なら 1 + 10^-14 程度だが，OZ2_B で
// b を 10 まで下げると K=14080 で 1.5 倍に達する．
// 下限 1.001 はノルムを倍精度で組む丸めの吸収で，b >= 19（K <= 14080）では係数がこれを
// 下回るので，既定の b では係数は下限の 1.001 そのものになる．
// デバイス側の tau は BF16 の切り上げでホストの tau 以上なので，デバイスの a~ は
// ここで測ったものより小さく，上界の向きは保たれる．
static inline double oz2_window_log2(double wA, double wB, uint32_t K, int b) {
    const double r = std::sqrt((double)K);
    const double f = (1.0 + r / std::ldexp(wA, b)) * (1.0 + r / std::ldexp(wB, b));
    return std::log2(wA * wB * std::max(f, 1.001));
}

// --- OZ2_FP64 の自動決定：幅 b を入力から決める ---------------------------------
//
// scheme II は要素を行／列の最大値 tau に対する b bit の固定小数
//   A' = round(a/tau_A * 2^b)   （量子化誤差 |delta| <= 1/2 が一様に乗る）
// として運ぶ．行 i の刻みは tau_A(i)*2^-b，列 j の刻みは tau_B(j)*2^-b なので，delta を
// 独立で平均 0・分散 1/12 と見るモデルのもとで誤差行列の二乗ノルムの期待値は
//   E||E||_F^2 ~= (2^-2b/12) * [ (sum_i tau_A(i)^2) ||B||_F^2 + (sum_j tau_B(j)^2) ||A||_F^2 ]
// と書ける（delta_A*delta_B の 2 次項は 2^-2b だけ小さいので落としてある．等号でないのは
// このため）．厳密演算での模擬とは 1.5 倍以内で一致する．ここから狙う相対誤差 2^-t に対し
//   b = t + ceil(log2 kappa2),   kappa2 = ||E||_F * 2^b / ||C||_F
// と採る．t=OZ2_TARGET_BITS（53）とすると一様乱数では kappa2 = 0.99（estimate_product_norm の安全率込み）
// となり b=53 に落ち着く．
// scheme I の自動決定（ozaki1/ozaki1_autop.hpp の oz_fp64_p）も同じ考え方の期待値のモデルで，そちらは量子化雑音の
// 代わりに，実際に切ったスライスから対角の打ち切りと残差の二乗和を求める．
// これは保証ではなく期待値のモデルである：delta が最近接丸めからずれる場合（スライサの
// 保証は |A'-X| <= 1 であって 1/2 ではない）もあれば，入力が 2^-b の格子にちょうど載って
// 誤差が厳密に 0 になる場合もある．estimate_product_norm の安全率はこのずれも併せて吸収する．
//
// b_forced は OZ2_B が明示されたかどうか．明示されていれば b_cap は動かさない．nowin は OZ2_NOWIN（窓を入力から
// 見積もらず最悪ケースのままにする．論文の φ の値の測定）．
static inline Oz2PrecisionPolicy choose_precision_policy(const std::vector<double>& A,
                                                         const std::vector<double>& B,
                                                         uint32_t M, uint32_t N, uint32_t K,
                                                         bool b_forced, bool nowin, Oz2PrecisionPolicy in) {
    Oz2PrecisionPolicy out = in;
    // kappa2 は A,B の絶対スケールに依らない（A->alpha*A, B->beta*B で分子・分母とも
    // alpha^2*beta^2 倍になる）．一方，二乗和は素直に組むと桁溢れする：|a| ~ 1e150 は
    // FP64 の範囲内でも nA*TB は 1e600 になって inf になり，kappa2 が NaN になる．
    // そこで先に 2 のべき（無誤差）で正規化してから測る．逆数も 2 のべきで無誤差なので，
    // 内側ループは除算ではなく乗算にする．
    oz2_measure_stats(A, B, M, N, K, out);
    const Oz2OperandStats& st = out.st;
    // CRT の窓：最悪ケース上界 |C'| <= K*2^(2b) の代わりに，Cauchy--Schwarz による
    //     max_ij sum_k |a~_ik||b~_kj| <= (max_i ||a~_i||_2)(max_j ||b~_j||_2)
    // を使う．行の最大値と列の最大値が別の k に居る入力ほど K を大きく下回るので，
    // 窓が縮んで M が小さくなり，再構成が解く frac = C'/M が大きくなる．
    // 整数化の誤差（|A'-X| <= 1）ぶんの交差項は b に依るので，係数は b が決まってから
    // oz2_window_log2 で掛ける（呼び出し側）．ここでは wA, wB だけを返す．
    // O(MK+KN) で済み，追加の行列積は要らない．
    if (st.wA > 0 && st.wB > 0 && !nowin) {
        out.w_measured = true; out.wA = st.wA; out.wB = st.wB;
    }
    const double nC = out.nC;
    // 零行列なら nC=0 で k2 が inf になり，(int)ceil(log2(inf)) が未定義になる．
    // 有限に留める（この上限に当たれば b_cap は下の min でどのみち OZ2_MAXDIGIT*OZ2_DIGIT_BITS = 100 になる）．
    const double k2 = std::min(std::sqrt((st.TA * st.nB + st.TB * st.nA) /
                                         std::max(12.0 * nC, 1e-300)), 1e18);
    if (!b_forced)
        out.b_cap = std::min(OZ2_MAXDIGIT * OZ2_DIGIT_BITS,
                             OZ2_TARGET_BITS + (int)std::ceil(std::log2(std::max(k2, 1.0))));
    // 窓の上界の係数は b に依る（oz2_window_log2）ので，「何 bit 小さいか」は b_cap を決めた後の幅で見積もる
    // （b >= 19 では係数が下限に張り付くので既定では前と同じ値．OZ2_B で b を小さくした調査で食い違わないように）．
    // 照合（verify/check_paper_numbers.py の load_phi_nowin）はこの行の有無で窓を入力から見積もったかを見る（CONFIG の win_meas）
    if (out.w_measured)
        printf("OZ2_WIN: 窓の上界を実測 W=%.4g（最悪ケース K=%u に対し %.1f bit 小さい）\n",
               st.wA * st.wB, K,
               std::log2((double)K) - oz2_window_log2(st.wA, st.wB, K, out.b_cap));
    // b を上げるには CRT の窓も広げねばならず，窓は法の本数で決まる（2b+1+log2 K bit
    // 必要）．そこで自動時は表の全長を要求し，本数の決定は法選択に任せる：あちらは
    // 「窓を満たしたまま落とせる法は落とす」ので，b_cap を満たす最小の本数へ自然に
    // 落ち着く．余分な法は窓を過剰にして frac を小さくし，かえって精度を落とすので
    // （4096^3・phi=2 で b=60・s=18 は b=59・s=17 より 2 bit 悪い），この最小化は必須である．
    out.s_req = OZ2_MAXS;
    printf("OZ2_FP64: kappa2=%.3g (||E||_F*2^b/||C||_F) -> b<=%d（上限 %d），法は最大 %d 本から選ぶ\n",
           k2, out.b_cap, OZ2_MAXDIGIT * OZ2_DIGIT_BITS, out.s_req);
    return out;
}

// --- 再構成のパス数（OZ2_PASSES=auto）の見込み ------------------------------------
// 1 パスの再構成は frac = C'/M を固定の絶対精度で解く（recon_crt.cpp の 3 本組の最下位 limb）ので，
// frac の誤差 e は出力では M 2^-2b tau_A(i) tau_B(j) 倍になる．e の実効値を eps1 と置くと，
// 相対 Frobenius 誤差の見込みは
//     eps1 * M 2^-2b * sqrt(sum_i tau_A(i)^2 * sum_j tau_B(j)^2) / ||C||_F
// である．窓を覆う M は 2^(2b) W（W は窓の上界）程度なので，見込みは b にほとんど依らず，
// 入力の指数部の範囲が広いほど（||C||_F が行・列の最大値の積に比べて小さいほど）大きい．
// 2 パスは 1/m_i を共通の刻み 2^-64 へ丸めた部分と残りに分けて別々に積むのでこの床が無いが，
// DST に両方の累積を同時に置けないので再構成を 2 回流す（recon_crt.cpp）．そこで見込みが目標 2^-53 より
// OZ2_RECON1_MARGIN bit 以上小さいときだけ 1 パスにする．eps1 は実機で較正した値（docs/accuracy.md の
// 「再構成のパス数の自動選択」）．照合（scripts/lib/srcrev.py の recon1_eps_log2）は CONFIG の recon1_eps= の無い古いログの
// ために，測った rev のソースからこの #define を名前と書き方で読むので，書き方を保つ．
#define OZ2_RECON1_EPS_LOG2 (-70.0)
#define OZ2_RECON1_MARGIN   2.0
// 1 パスにする条件（見込みの log2 がこれ以下なら 1 パス．OZ2_PASSES 行と CONFIG の recon1_thr）
constexpr double OZ2_RECON1_THR_LOG2 = -(OZ2_TARGET_BITS + OZ2_RECON1_MARGIN);

// 1 パスの復元誤差の見込み（相対 Frobenius 誤差の log2）．統計が無ければ NaN，積が零なら -inf．
static inline double oz2_recon1_log2(const Ozaki2Moduli& mo, int b, const Oz2PrecisionPolicy& p) {
    if (!p.stats) return NAN;
    if (!(p.nC > 0) || !(p.st.TA > 0) || !(p.st.TB > 0)) return -INFINITY;
    double l2m = 0;
    for (int m : mo.m) l2m += std::log2((double)m);
    return OZ2_RECON1_EPS_LOG2 + l2m - 2.0 * b
           + 0.5 * (std::log2(p.st.TA) + std::log2(p.st.TB) - std::log2(p.nC));
}
