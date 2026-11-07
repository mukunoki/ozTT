// SPDX-License-Identifier: Apache-2.0
#pragma once
// 自動決定（尾崎スキーム I の OZ_FP64・II の OZ2_FP64）が共通に使う ||C||_F の見積り
// ---------------------------------------------------------------------------
// どちらのスキームも，誤差の期待値のモデルの分母 ||C||_F をここで見積もる（ホストだけで完結し，
// デバイスは触らない．計時区間の外）．scheme II の ozaki2_main.cpp から移した．
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// 二乗和を組む前に行列を割っておくための 2 のべき（無誤差なので比には影響しない）．
// 最大絶対値の指数をそのまま採る．指数は正規化数の範囲へ丸め込むので，入力が
// 非正規化数だけでも係数自体が 0 や inf になることはない．
inline double oz_pow2_scale(const std::vector<double>& v){
    double mx=0; for(double x:v) mx=std::max(mx,std::fabs(x));
    if(!(mx>0)||!std::isfinite(mx)) return 1.0;
    int e; std::frexp(mx,&e);
    e=std::min(std::max(e,-1000),1000);
    return std::ldexp(1.0,e);
}

// ||C||_F^2 を Hutchinson のスケッチで測る（L 本の平均）．返すのは下側の見積りで，
// 最大を 1 本捨て，さらに安全率で割ってある．
//
// kappa2 の分母 ||C||_F を要素の抜き取りで測ってはいけない．裾の重い入力では二乗和が
// 少数の要素に支配され，抜き取りの分散が発散する（phi=8 では 10 桁外し，必要な b を
// 8 bit 過小に見積もる）．代わりに Hutchinson のスケッチを使う：平均 0・分散 1 の
// 乱数ベクトル g に対して E||A(B g)||^2 = ||AB||_F^2 が厳密に成り立ち，1 本あたり
// O(MK+KN) で済む．全要素を一度に見るので裾に強く，L=8 本で phi=0..8 のいずれでも
// 真の ||C||_F^2 の数 % 以内に入る．
// ただし無偏なだけでは足りない：||C||_F^2 を過大に見積もると b が不足する側へ
// 外れる．L 本平均の相対分散は 2/L * tr((C^T C)^2)/||C||_F^4 で，最悪（rank 1）で
// 2/L なので，L=8 では相対標準偏差が 0.5 に達する．そこで 2 つ手を打つ：
//   (1) 最大の 1 本を捨てて平均する．seed が固定なので，C の右特異ベクトルが特定の
//       g_l に揃う入力を作れば 1 本だけ N/L 倍に跳ねるが，それはこれで落ちる．
//   (2) 残りの平均をさらに 1+2*sqrt(2/L) で割る（L=8 なら 1/2）．
// kappa2 が sqrt(2) 倍になるので b は 0.5 bit ぶん安全側に振れる．
// これは実用入力に対する経験則であって下界の保証ではない．有効ランクの高い入力では
// 過剰だが，b が 1 上がっても法が 1 本増えるだけで済むので，不足側より安い．
inline double estimate_product_norm(const std::vector<double>& A,
                                    const std::vector<double>& B,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    double rgA, double rgB) {
    const uint32_t L = 8;
    std::mt19937 rg(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> g(N), y(K);
    double cs2 = 0, cmax = 0;
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t j = 0; j < N; j++) g[j] = nd(rg);
        for (uint32_t k = 0; k < K; k++) {
            long double s = 0;
            for (uint32_t j = 0; j < N; j++) s += (long double)(B[(size_t)k * N + j] * rgB) * g[j];
            y[k] = (double)s;
        }
        double q = 0;
        for (uint32_t i = 0; i < M; i++) {
            long double s = 0;
            for (uint32_t k = 0; k < K; k++) s += (long double)(A[(size_t)i * K + k] * rgA) * y[k];
            q += (double)s * (double)s;
        }
        cs2 += q; cmax = std::max(cmax, q);
    }
    return (cs2 - cmax) / (L - 1) / (1.0 + 2.0 * std::sqrt(2.0 / (double)L));
}
