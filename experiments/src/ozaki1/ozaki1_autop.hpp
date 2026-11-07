#pragma once
// 尾崎スキーム I：スライス数 p を入力から選ぶ誤差のモデル（OZ_FP64）
// ---------------------------------------------------------------------------
// ホストだけで完結し，デバイスには触らない（計時区間の外）．ドライバの prepare_operands（ozaki1_main.cpp）が呼ぶ．
// 依存：<cmath> などの標準ヘッダ，fmt（表示），oz_normest.hpp（||C||_F の見積り），oz_bits.h，
// デバイスの正規化と共有するビット手続き common/kernels/scale_bits.h．
//
// OZ_FP64【入力から p を選ぶ経路】：誤差のモデルから，相対誤差 ||E||_F/||C||_F が
// 2^-53 以下になると見込まれる p を探す（scheme II の OZ2_FP64 と同じ考え方で，保証ではない．下の (2) が p について
// 単調とは限らないので，見つかるのは条件を満たす最小の p とは限らない．探し方は oz1_fp64_search）．
// 表の測定は p を固定し（値の正本は scripts/routes.tsv），この経路は指数部レンジ φ を広げた入力での
// 選び直しに用いる：4096^3 の fig:phi の自動側（scripts/measure/exponent_sweep.sh）と，論文の φ の段落の
// 14080^3 の Ozaki-I-BFP8 の到達精度と性能（scripts/measure/exponent_adaptive.sh）．
//
// 正規化と切り方はデバイスと同じ規則で行う．
//   正規化：行（A）／列（B）の最大絶対値を経路の向きで BF16 にし（切り捨て・切り上げ（bf_ceil）・2 のべきの 1 段上げ
//     （tau_above）．scale_f64_reader.cpp の CEIL），e = ceil(log2(その BF16))（expdelta.cpp）として
//     x = a·2^(nshift-e) と置く（oz1_dev_exp がカーネルと同じ関数で求める）．目標 2^nshift は BFP2_b が 1，ほかが 1/2 で
//     （parse_run_config），切り捨ての形式では x が目標を相対 2^-7 未満だけ超えうる．
//   切り方：u = beta t，d = clamp(round(u), ±桁の上限)，t = u - d（slice_tf_num.cpp・slice_tf_sm8.cpp）．
//     round は最近接偶数で，デバイスと同じく u を FP32 へ丸めた値で決める（triple-float の上位 1 語を丸めるため．
//     同点の近くでだけ違いが出る）．残差 t は double 2 語で持つ（OzDigits．基数 14・254 でも p 段まで厳密に追える）．
//     BFP2_b で行・列の最大値が 2 のべきのすぐ下（正規化後 1-δ）だと，桁の上限 1 で切られて先頭に約 log2(1/δ) 個の
//     桁 1 が並ぶ（parse_run_config の tau_above の注記）が，それもデバイスと同じに再現する．
// x = sum_{i<p} D_i + R_p（D_i = d_i beta^-(i+1)，R_p = t_p beta^-p．y も同様）と書くと，デバイスが残すのは対角
// i+j <= p-1 の対だけなので，出力 (m,n) の打ち切りの誤差は
//   E_mn = s_m s_n sum_k [ sum_{i<p} D_i(x_mk) R_{p-i}(y_kn) + R_p(x_mk) y_kn ]
// である（s は正規化に用いた除数 2^(e-nshift)．i+j >= p の対と，スライスに入らなかった残差の積をすべて含む）．
// 相対誤差を次の 2 通りに見積もり，どちらも 2^-53 以下になる p を探す（大きいほうで判定する）．
//   (1) 符号を独立とみなした打ち切りの誤差の二乗和（全要素を使う．oz1_fp64_indep）．k 方向の項と対ごとの項の符号を
//       独立とみなすと
//         E||E||_F^2 ~= sum_k [ sum_{i+j>=p, i,j<p} SA_i[k] SB_j[k] + RA_p[k] NB[k] + NA[k] RB_p[k] ]
//         SA_i[k] = sum_m (s_m D_i(x_mk))^2，RA_p[k] = sum_m (s_m R_p(x_mk))^2，NA[k] = sum_m a_mk^2（B 側も同様）
//       分母 ||C||_F^2 は scheme II と同じ Hutchinson の見積り（common/oz_normest.hpp．下側へ寄せてあるので p は多めに振れる）．
//       一様乱数・φ の乱数のように誤差の符号がそろわない入力では，ほぼこれで決まる．
//   (2) 行と列を抜き出し，その組についてデバイスが返す値を模擬する（符号の独立を仮定しない．oz1_fp64_sampled）．
//       抜き出した組の C_mn を Dot2 で求め，上の式の E_mn を差し引いて FP64 へ丸めた値（同点は 0 から遠い側．中点から
//       組み立ての窓の幅 2^-59 以内なら C_mn から遠い側）と C_mn の差を，重みを付けて足す．分母は同じ重みで足した
//       C_mn^2 と (1) の分母の大きいほう．
//       抜き出すのは，行・列それぞれノルムの大きい順に OZ1_FP64_TOP 本と，残りから一様に OZ1_FP64_UNI 本で，
//       一様の分には（残りの本数）/（抜き出した本数）の重みを掛ける（層別の推定．全部を抜き出せば厳密）．
//       定数や符号の型の決まった行列のように誤差の符号がそろう入力では，誤差は k について K に比例して，対について
//       対の数に比例して足されるので，(1) は楽観側に大きく外れる．さらに全要素が同じ値なので FP64 への丸めの誤差も
//       全要素でそろい（要素ごとに最大で半 ulp），打ち切りの誤差がそれを中点の向こうへ押すと全要素が 1 ulp ずれる．
//       (2) はその両方を拾う．
//       分母を (1) の分母で下から抑えるのは，指数部レンジの広い入力では ||C||_F^2 が少数の大きい要素に支配され，
//       ノルムで選んだ行・列の組がそれを外すと抜き出しの和が桁違いに小さく出て，(2) が悲観側へ大きく外れるためである．
//       計算量は q 1 つあたり O(K q (行数 + 列数 + 行数 x 列数))（抜き出した数）で，(1) で選んだ p から上へ探す．
//       (2) は抜き出した行・列でしか符号のそろいを見ないので，そろう部分が抜き出しから外れた少数の行・列に限られる
//       入力では楽観側に外れうる（保証ではない．Accuracy・RESULT 行の実測で見る）．
// 抜き出しの本数・(2) の分母の取り方を決めた実測，実測の到達精度との比較，以前の規則（Abdelfattah らの誤差限界に
// 倣ったもの）との違いは docs/accuracy.md の「scheme I の自動決定（OZ_FP64）のモデル」（rev と条件つき）．

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <fmt/core.h>
#include "common/oz_normest.hpp"       // estimate_product_norm・oz_pow2_scale
#include "common/oz_bits.h"            // f64_bits
#include "common/kernels/scale_bits.h" // oz_absf64_to_bf16・oz_bf16_scale_exp（デバイスの正規化と同じ関数）

// (2) で抜き出す行・列の本数：ノルムの大きい順に TOP 本と，残りから一様に UNI 本（行・列それぞれ）
constexpr uint32_t OZ1_FP64_TOP = 32, OZ1_FP64_UNI = 32;
// (2) の一様な抜き出しの乱数の種（固定．同じ入力なら同じ p を選ぶ）．A の行と B の列で別の種
constexpr uint32_t OZ1_FP64_SEED_ROWS = 20261021u, OZ1_FP64_SEED_COLS = 20261022u;

// 行・列の最大絶対値 mx（FP64）から，デバイスが正規化に用いる指数 e（x = a·2^(nshift-e)）を求める．
// ceil_mode は scale_f64_reader.cpp の CEIL（0 切り捨て・1 切り上げ・2 切り上げと 2 のべきの 1 段上げ）で，
// BF16 化と指数はカーネルと同じ関数（scale_bits.h）で求める．範囲外（BF16 の最大有限値を超える・
// 非零で 2^-126 未満・Inf・NaN．デバイスは C を読んだ後に止まる）と 0 は in_range=false で返す．
static int oz1_dev_exp(double mx, uint32_t ceil_mode, bool& in_range) {
    const uint64_t u = f64_bits(mx);
    const uint32_t h = (uint32_t)(u >> 32) & 0x7FFFFFFFu, l = (uint32_t)u;
    const uint16_t b = ceil_mode == 2 ? oz_absf64_to_bf16<2>(h, l)
                     : ceil_mode == 1 ? oz_absf64_to_bf16<1>(h, l) : oz_absf64_to_bf16<0>(h, l);
    uint32_t bad = 0;
    const int32_t e = oz_bf16_scale_exp(b, bad);
    in_range = b != 0 && bad == 0u;   // 0 は全零の行・列（誤差に数えない）
    return (int)e;
}

// 1 要素の基数 beta の展開．デバイスのスライサと同じ規則で 1 段ずつ進める（上の「切り方」）．
// 残差 t を double 2 語 th + tl で持つ（基数 beta の積 beta*th は FMA で厳密に 2 語へ分ける）．
struct OzDigits {
    double th = 0.0, tl = 0.0;
    // 補償和の TwoSum．参照解の Dot2（common/oz_reference.hpp の reference_gemm）と手法は同じだが，ここは
    // (2) の C_mn と残差の足す順（sl += pe + te）を持つので共通化しない（どちらかへ寄せると，参照解か選ぶ p の
    // ビットが変わる）
    static void two_sum(double a, double b, double& s, double& e) {
        s = a + b; const double bb = s - a; e = (a - (s - bb)) + (b - bb);
    }
    // u = beta t，d = clamp(round(u), ±dmax)，t = u - d として桁 d を返す．uh - d は無誤差
    // （|uh - d| <= 1/2 程度で d と同符号の近い値か，clamp なら d <= uh <= beta*R <= 2d）．
    double step(double beta, double dmax) {
        const double ph = beta * th, pe = std::fma(beta, th, -ph);
        double uh, ul; two_sum(ph, pe + beta * tl, uh, ul);
        const double d = std::min(dmax, std::max(-dmax, std::nearbyint((double)(float)uh)));
        two_sum(uh - d, ul, th, tl);
        return d;
    }
    double value() const { return th + tl; }
};

// 展開の基数・桁の上限・正規化の目標と，探す p の上限
struct Oz1Fp64Cfg {
    uint32_t M = 0, N = 0, K = 0;
    double beta = 0.0, dmax = 0.0;   // 基数と桁の上限
    int32_t nshift = 0;              // 正規化の目標 2^nshift
    uint32_t ceil_mode = 0;          // BF16 化の向き（scale_build へ渡す CEIL と同じ値）
    int pmax = 0;                    // 探す p の上限
};

// 行・列ごとの正規化の指数 e（oz1_dev_exp）と，除数 2^(e-nshift) を二乗和の尺度（rgA・rgB を掛けた側）へ
// 寄せた sA・sB．0 は零の行・列と範囲外の行・列で，誤差に数えない．nrA・nrB は (2) の抜き出しに使うノルムの二乗．
// 二乗和が桁溢れしないよう 2 のべき rgA・rgB を掛けてから測る（比は変わらない．scheme II と同じ）．
struct Oz1Fp64Norm {
    double rgA = 1.0, rgB = 1.0;
    std::vector<int> eA, eB;
    std::vector<double> sA, sB, nrA, nrB;
};
static Oz1Fp64Norm oz1_fp64_norm(const std::vector<double>& A, const std::vector<double>& Bm, const Oz1Fp64Cfg& g) {
    const uint32_t M = g.M, N = g.N, K = g.K;
    const int32_t nshift = g.nshift;
    const uint32_t ceil_mode = g.ceil_mode;
    Oz1Fp64Norm nm;
    nm.rgA = 1.0 / oz_pow2_scale(A); nm.rgB = 1.0 / oz_pow2_scale(Bm);
    const double rgA = nm.rgA, rgB = nm.rgB;
    nm.eA.assign(M, 0); nm.eB.assign(N, 0);
    nm.sA.assign(M, 0.0); nm.sB.assign(N, 0.0); nm.nrA.assign(M, 0.0); nm.nrB.assign(N, 0.0);
    std::vector<int>& eA = nm.eA; std::vector<int>& eB = nm.eB;
    std::vector<double>& sA = nm.sA; std::vector<double>& sB = nm.sB;
    std::vector<double>& nrA = nm.nrA; std::vector<double>& nrB = nm.nrB;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t m = 0; m < (int64_t)M; m++) {
        double mx = 0, nr = 0;
        for (uint32_t k = 0; k < K; k++) {
            const double v = A[(size_t)m*K+k];
            mx = std::max(mx, std::fabs(v)); nr += (v*rgA) * (v*rgA);
        }
        bool ok = false; const int e = oz1_dev_exp(mx, ceil_mode, ok);
        if (mx > 0 && ok) { eA[m] = e; sA[m] = std::ldexp(rgA, e - nshift); }
        nrA[m] = nr;
    }
    {
        constexpr uint32_t CB = 64;   // 列は k を外側に回す（B は行優先）
        const uint32_t nblk = (N + CB - 1) / CB;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t bk = 0; bk < (int64_t)nblk; bk++) {
            const uint32_t n0 = (uint32_t)bk * CB, n1 = std::min(N, n0 + CB);
            double mx[CB] = {0}, nr[CB] = {0};
            for (uint32_t k = 0; k < K; k++)
                for (uint32_t n = n0; n < n1; n++) {
                    const double v = Bm[(size_t)k*N+n];
                    mx[n-n0] = std::max(mx[n-n0], std::fabs(v)); nr[n-n0] += (v*rgB) * (v*rgB);
                }
            for (uint32_t n = n0; n < n1; n++) {
                bool ok = false; const int e = oz1_dev_exp(mx[n-n0], ceil_mode, ok);
                if (mx[n-n0] > 0 && ok) { eB[n] = e; sB[n] = std::ldexp(rgB, e - nshift); }
                nrB[n] = nr[n-n0];
            }
        }
    }
    return nm;
}

// (1) 符号を独立とみなした二乗和．返すのは e2r[q]（q = 2..pmax の ||E||_F^2．二乗和の尺度）．
static std::vector<double> oz1_fp64_indep(const std::vector<double>& A, const std::vector<double>& Bm,
                                          const Oz1Fp64Cfg& g, const Oz1Fp64Norm& nm) {
    const uint32_t M = g.M, N = g.N, K = g.K;
    const int pmax = g.pmax;
    const size_t P1 = (size_t)pmax + 1;
    const double beta = g.beta, dmax = g.dmax, rgA = nm.rgA, rgB = nm.rgB;
    const int32_t nshift = g.nshift;
    // S[i*K+k]：第 i スライスの寄与，R[p*K+k]：p 本で切ったときの残差の寄与
    std::vector<double> SA(P1*K, 0.0), RA(P1*K, 0.0), SB(P1*K, 0.0), RB(P1*K, 0.0), NA(K, 0.0), NB(K, 0.0);
    auto cut = [&](double x, double s, double* S, double* R) {   // x：正規化した値，s：除数（二乗和の尺度）
        OzDigits t; t.th = x;
        double w = 1.0;
        for (int i = 0; i < pmax; i++) {
            const double d = t.step(beta, dmax);
            w /= beta;
            S[(size_t)i*K] += (s*d*w) * (s*d*w);
            const double r = s * t.value() * w;
            R[(size_t)(i+1)*K] += r * r;
            if (t.th == 0) break;   // 残りのスライスと残差はすべて 0
        }
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t k = 0; k < (int64_t)K; k++) {
        double na = 0, nb = 0;
        for (uint32_t m = 0; m < M; m++) {
            const double v = A[(size_t)m*K+k]; na += (v*rgA) * (v*rgA);
            if (v != 0 && nm.sA[m] > 0) cut(std::ldexp(v, nshift - nm.eA[m]), nm.sA[m], &SA[k], &RA[k]);
        }
        for (uint32_t n = 0; n < N; n++) {
            const double v = Bm[(size_t)k*N+n]; nb += (v*rgB) * (v*rgB);
            if (v != 0 && nm.sB[n] > 0) cut(std::ldexp(v, nshift - nm.eB[n]), nm.sB[n], &SB[k], &RB[k]);
        }
        NA[k] = na; NB[k] = nb;
    }
    // G[i][j] = sum_k SA_i[k] SB_j[k]（対角の打ち切りの項）
    std::vector<double> G((size_t)pmax * pmax, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < pmax; i++)
        for (int j = 0; j < pmax; j++) {
            double gij = 0; for (uint32_t k = 0; k < K; k++) gij += SA[(size_t)i*K+k] * SB[(size_t)j*K+k];
            G[(size_t)i*pmax+j] = gij;
        }
    std::vector<double> e2r(P1, 0.0);   // e2r[q]：(1) の ||E||_F^2（q = 2..pmax）
    for (int q = 2; q <= pmax; q++) {
        double e2 = 0;
        for (int i = 0; i < q; i++) for (int j = std::max(0, q - i); j < q; j++) e2 += G[(size_t)i*pmax+j];
        for (uint32_t k = 0; k < K; k++) e2 += RA[(size_t)q*K+k] * NB[k] + NA[k] * RB[(size_t)q*K+k];
        e2r[q] = e2;
    }
    return e2r;
}

// (2) 抜き出した行と列の組の E_mn．構築で組を抜き出して C_mn を求め，e2(q) が q 本で切ったときの相対誤差の二乗を返す．
struct Oz1Fp64Sampled {
    const Oz1Fp64Cfg* g = nullptr;
    const Oz1Fp64Norm* nm = nullptr;
    std::vector<uint32_t> iA, iB;       // 抜き出した行・列
    std::vector<double> wA, wB;         // その重み
    size_t na = 0, nb = 0;
    std::vector<double> XA, YB;         // 抜き出した行・列の正規化した値
    std::vector<double> Ch, Cl;         // 抜き出した組の C_mn（二乗和の尺度．2 語）
    double nC2 = 0.0;                   // (2) の分母

    // q 本で切ったときの (2) の相対誤差の二乗 ||fl(C-E) - C||_F^2 / ||C||_F^2（抜き出した組の重み付きの和を nC2 で割る）．
    // k を固定の 64 区間に分けて区間ごとに足し，区間の順に合わせる（スレッド数によらず同じ値になる）．
    // 組が無い（全行か全列が零・範囲外）ときは空和の 0
    double e2(int q) const {
        if (na == 0 || nb == 0) return 0.0;
        const uint32_t K = g->K;
        const double beta = g->beta, dmax = g->dmax;
        constexpr uint32_t NCH = 64;
        std::vector<double> part((size_t)NCH * na * nb, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int64_t c = 0; c < (int64_t)NCH; c++) {
            const uint32_t k0 = (uint32_t)((uint64_t)K * c / NCH), k1 = (uint32_t)((uint64_t)K * (c + 1) / NCH);
            std::vector<double> D(na * q), RX(na), RY(nb * (q + 1)), Y(nb);
            double* acc = &part[(size_t)c * na * nb];
            for (uint32_t k = k0; k < k1; k++) {
                for (size_t a = 0; a < na; a++) {   // D_i(x)（i < q）と R_q(x)
                    OzDigits t; t.th = XA[a*K+k];
                    double w = 1.0;
                    for (int i = 0; i < q; i++) { const double d = t.step(beta, dmax); w /= beta; D[a*q+i] = d * w; }
                    RX[a] = t.value() * w;
                }
                for (size_t b = 0; b < nb; b++) {   // R_c(y)（c = 1..q）と y
                    OzDigits t; t.th = Y[b] = YB[b*K+k];
                    double w = 1.0;
                    for (int c2 = 1; c2 <= q; c2++) { (void)t.step(beta, dmax); w /= beta; RY[b*(q+1)+c2] = t.value() * w; }
                }
                for (size_t a = 0; a < na; a++)
                    for (size_t b = 0; b < nb; b++) {
                        double e = RX[a] * Y[b];
                        for (int i = 0; i < q; i++) e += D[a*q+i] * RY[b*(q+1)+(q-i)];
                        acc[a*nb+b] += e;
                    }
            }
        }
        double e2 = 0;
        for (size_t a = 0; a < na; a++)
            for (size_t b = 0; b < nb; b++) {
                double E = 0; for (uint32_t c = 0; c < NCH; c++) E += part[((size_t)c * na + a) * nb + b];
                // デバイスは C - E を FP64 へ丸めて返す．丸める前の値は，組み立ての 64 bit の窓（f64_tf.h の assemble_tf_f64．
                // 最上位の limb の先頭から 60 bit 下までを残し，下位の limb を limb ごとに 0 の方向へ切る）で相対 2^-60 ほど
                // 動きうるので，中点からその幅（ここでは 2^-59）以内にある値はどちらへ丸まるか決められない．そのときは
                // C から遠い側を採る（定数の行列は全要素が同じ値なので，中点の近くに来ると全要素が 1 ulp ずれる）
                const size_t ab = a*nb+b;
                double h, r; OzDigits::two_sum(Ch[ab], Cl[ab] - nm->sA[iA[a]] * nm->sB[iB[b]] * E, h, r);
                double out = h;
                if (h != 0) {
                    const double h2 = std::nextafter(h, r >= 0 ? INFINITY : -INFINITY);   // r の側の隣
                    const double gap = 0.5 * std::fabs(h2 - h) - std::fabs(r);           // 中点までの距離
                    if (gap <= std::ldexp(1.0, std::ilogb(h) - 59)) {
                        const double d1 = (h - Ch[ab]) - Cl[ab], d2 = (h2 - Ch[ab]) - Cl[ab];
                        if (std::fabs(d2) > std::fabs(d1)) out = h2;
                    }
                }
                const double dv = (out - Ch[ab]) - Cl[ab];
                e2 += wA[a] * wB[b] * dv * dv;
            }
        return nC2 > 0 ? e2 / nC2 : (e2 > 0 ? INFINITY : 0.0);
    }
};

// (2) の組を抜き出す：ノルムの大きい順に OZ1_FP64_TOP 本（重み 1）と，残りから一様に OZ1_FP64_UNI 本（重み 残り/本数）．
// 零・範囲外の行・列は誤差 0 なので候補から除く．nC は (1) の分母（全要素の Hutchinson の見積り．下側へ寄せてある）．
static Oz1Fp64Sampled oz1_fp64_sampled(const std::vector<double>& A, const std::vector<double>& Bm,
                                       const Oz1Fp64Cfg& g, const Oz1Fp64Norm& nm, double nC) {
    const uint32_t M = g.M, N = g.N, K = g.K;
    const int32_t nshift = g.nshift;
    Oz1Fp64Sampled sm;
    sm.g = &g; sm.nm = &nm;
    auto pick = [&](const std::vector<double>& nr, const std::vector<double>& s, uint32_t L, uint32_t seed,
                    std::vector<uint32_t>& idx, std::vector<double>& wt) {
        std::vector<uint32_t> ord;
        for (uint32_t i = 0; i < L; i++) if (s[i] > 0) ord.push_back(i);
        const size_t nt = std::min<size_t>(OZ1_FP64_TOP, ord.size());
        std::partial_sort(ord.begin(), ord.begin() + nt, ord.end(),
                          [&](uint32_t a, uint32_t b) { return nr[a] > nr[b] || (nr[a] == nr[b] && a < b); });
        for (size_t i = 0; i < nt; i++) { idx.push_back(ord[i]); wt.push_back(1.0); }
        std::vector<uint32_t> rest(ord.begin() + nt, ord.end());
        const size_t nu = std::min<size_t>(OZ1_FP64_UNI, rest.size());
        std::mt19937 rg(seed);
        for (size_t i = 0; i < nu; i++) {
            std::swap(rest[i], rest[i + rg() % (rest.size() - i)]);
            idx.push_back(rest[i]); wt.push_back((double)rest.size() / (double)nu);
        }
    };
    pick(nm.nrA, nm.sA, M, OZ1_FP64_SEED_ROWS, sm.iA, sm.wA);
    pick(nm.nrB, nm.sB, N, OZ1_FP64_SEED_COLS, sm.iB, sm.wB);
    const size_t na = sm.iA.size(), nb = sm.iB.size();
    sm.na = na; sm.nb = nb;
    sm.XA.assign(na * K, 0.0); sm.YB.assign(nb * K, 0.0);
    for (size_t a = 0; a < na; a++)
        for (uint32_t k = 0; k < K; k++) sm.XA[a*K+k] = std::ldexp(A[(size_t)sm.iA[a]*K+k], nshift - nm.eA[sm.iA[a]]);
    for (size_t b = 0; b < nb; b++)
        for (uint32_t k = 0; k < K; k++) sm.YB[b*K+k] = std::ldexp(Bm[(size_t)k*N+sm.iB[b]], nshift - nm.eB[sm.iB[b]]);
    // 抜き出した組の C_mn（二乗和の尺度．Dot2 で 2 語 Ch + Cl に求める）．e2 はこれと打ち切りの誤差から，デバイスが返す
    // FP64 の値 fl(C_mn - E_mn) を作り，その C_mn との差で見る
    sm.Ch.assign(na * nb, 0.0); sm.Cl.assign(na * nb, 0.0);
    const double rgA = nm.rgA, rgB = nm.rgB;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ab = 0; ab < (int64_t)(na * nb); ab++) {
        const size_t a = (size_t)ab / nb, b = (size_t)ab % nb;
        double sh = 0, sl = 0;
        for (uint32_t k = 0; k < K; k++) {
            const double u = A[(size_t)sm.iA[a]*K+k] * rgA, v = Bm[(size_t)k*N+sm.iB[b]] * rgB;
            const double pr = u * v, pe = std::fma(u, v, -pr);
            double t, te; OzDigits::two_sum(sh, pr, t, te);
            sh = t; sl += pe + te;
        }
        OzDigits::two_sum(sh, sl, sm.Ch[ab], sm.Cl[ab]);
    }
    // (2) の分母：抜き出しから同じ重みで推定した ||C||_F^2 と，(1) の分母の大きいほう．定数の行列では前者が厳密な値に
    // なる．指数部レンジの広い入力では ||C||_F^2 が少数の大きい要素に支配され，行・列をノルムで選んでもその要素の組
    // （同じ k で大きい a_mk と b_kn）を外すことがあり，前者が桁違いに小さく出て (2) が悲観側へ大きく外れるので，
    // そのときは後者を使う．
    double nCs = 0;
    for (size_t a = 0; a < na; a++)
        for (size_t b = 0; b < nb; b++) nCs += sm.wA[a] * sm.wB[b] * sm.Ch[a*nb+b] * sm.Ch[a*nb+b];
    sm.nC2 = std::max(nC, nCs);
    return sm;
}

// p の選択：(1) の相対誤差の二乗 r1[q] と (2) の e2(q) の大きいほうが (2^-53)^2 以下になる q を探す．
// (1) だけで目標に届く最小の q から始め，(2) が外れていれば上へ倍々に広げてから二分で詰める（(2) は q について
// 単調とは限らないので，見つかるのは直前の q が外れる q で，条件を満たす最小の q とは限らない）．
// 上限 pmax まで届かなければ pmax を返す（表示に出す）．e2sv は (2) を評価した q の値を持つ（未評価は負）．
static int oz1_fp64_search(const std::vector<double>& r1, std::vector<double>& e2sv, const Oz1Fp64Sampled& sm,
                           int pmax, double tgt) {
    auto f2 = [&](int q) { if (e2sv[q] < 0) e2sv[q] = sm.e2(q); return std::max(r1[q], e2sv[q]); };
    int q0 = pmax;
    for (int q = 2; q <= pmax; q++) if (r1[q] <= tgt) { q0 = q; break; }
    int p = q0;
    if (f2(q0) > tgt) {
        int lo = q0, hi = -1;
        for (int step = 1; lo < pmax; step *= 2) {
            const int q = std::min(pmax, lo + step);
            if (f2(q) <= tgt) { hi = q; break; }
            lo = q;
        }
        if (hi < 0) p = pmax;   // 上限まで届かない（下の表示に出す）
        else {
            while (hi - lo > 1) { const int mid = (lo + hi) / 2; if (f2(mid) <= tgt) hi = mid; else lo = mid; }
            p = hi;
        }
    }
    (void)f2(p);
    return p;
}

// 入力 A（M×K）・B（K×N）から p を選び，OZ_FP64: の行を表示して返す．radix は基数，dmax は桁の上限，nshift は
// 正規化の目標の指数，ceil_mode は scale_build へ渡す CEIL（RunCfg の tau_above ? 2 : bf_ceil ? 1 : 0）．
static int oz_fp64_p(const std::vector<double>& A, const std::vector<double>& Bm,
                     uint32_t M, uint32_t N, uint32_t K, int radix, uint32_t dmax, int32_t nshift, uint32_t ceil_mode) {
    Oz1Fp64Cfg g;
    g.M = M; g.N = N; g.K = K;
    g.beta = (double)radix; g.dmax = (double)dmax; g.nshift = nshift; g.ceil_mode = ceil_mode;
    // 探す p の上限：正規化の位置から 160 bit 下まで．これより深く切るのは指数部レンジが極端な入力だけで，
    // そこまでで目標に届かなければ上限を返す（表示に出す）．
    g.pmax = (int)std::ceil(160.0 / std::log2(g.beta)) + 1;
    const int pmax = g.pmax;
    const size_t P1 = (size_t)pmax + 1;
    const Oz1Fp64Norm nm = oz1_fp64_norm(A, Bm, g);
    const std::vector<double> e2r = oz1_fp64_indep(A, Bm, g, nm);
    const double nC = estimate_product_norm(A, Bm, M, N, K, nm.rgA, nm.rgB);   // ||C||_F^2 の下側の見積り（(1) の分母）
    const Oz1Fp64Sampled sm = oz1_fp64_sampled(A, Bm, g, nm, nC);
    const double tgt = std::ldexp(1.0, -106);                            // (2^-53)^2
    std::vector<double> r1(P1, INFINITY), e2sv(P1, -1.0);               // r1[q]：(1) の相対誤差の二乗
    for (int q = 2; q <= pmax; q++) r1[q] = nC > 0 ? e2r[q] / nC : (e2r[q] > 0 ? INFINITY : 0.0);
    const int p = oz1_fp64_search(r1, e2sv, sm, pmax, tgt);
    const double r2p = std::max(r1[p], e2sv[p]);
    auto bits = [](double r2) { return r2 > 0 ? -0.5 * std::log2(r2) : INFINITY; };
    // measure/exponent_sweep.sh・study/phi_sweep.sh が "auto p=<数>" を，figures/update_fig_phi.py などが行頭の "OZ_FP64:" を読むので，
    // その部分の書き方は変えない（"auto p=" は 1 行に 1 回だけ出す）．選んだ p は CONFIG 行の auto_p= にも出す（照合は
    // そちらを先に読む）．到達精度はこの行ではなく Accuracy・RESULT 行の実測で見る．
    // sampled の値は FP64 への丸めを含むので，誤差の小さい入力では実測の到達精度（一様乱数で約 54.2 bit）に近い値になる．
    fmt::print("OZ_FP64: radix={} expected-error model: ~{:.1f} bit at p={} (independent-sign truncation ~{:.1f} bit, "
               "sampled {}x{} rows/cols incl. FP64 rounding ~{:.1f} bit; target 53{}) -> auto p={} "
               "(model estimate, no accuracy guarantee -- see the measured Accuracy line)\n",
               radix, bits(r2p), p, bits(r1[p]), sm.na, sm.nb, bits(e2sv[p]),
               (p == pmax && r2p > tgt) ? "; search limit reached" : "", p);
    return p;
}
