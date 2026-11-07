// SPDX-License-Identifier: Apache-2.0
#pragma once
// 参照解（host_eval.hpp の傘の下）：FP64 の結果を評価する Dot2 の参照 reference_gemm（尾崎の 2 つのドライバ・基準実装
// gemm_bf16・gemm_bf16ext と CPU の DGEMM の参照 tools/dgemmcheck で共通）と，整数の行列積の int64 の参照 int_reference_gemm
// （gemm_int8・gemm_exact_check）
// ---------------------------------------------------------------------------
// 環境変数
//   OZ_NOVERIFY   参照解の計算を丸ごと省く（性能測定のとき．精度・一致の判定もしない）
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "oz_env.hpp"   // つまみの読み方（相対で取り込む理由は host_eval.hpp の冒頭）

namespace oz_eval {

// 参照解を求めるか（OZ_NOVERIFY が立っていれば求めない）
inline bool verify_enabled() { return !oz_env::set("OZ_NOVERIFY"); }

// 参照解 C = A*B を，非評価和 Chi + Clo の形で返す．
//
// 誤差なし変換による補償付き総和（Dot2）：積は TwoProd で無誤差に分解し，和は
// TwoSum で上位と残差に分けて，残差を Clo に足し込む．結果はペア (Chi, Clo) の
// 非評価和として返す．
//
// この参照の精度は，__float128 の参照和と突き合わせると phi=0--16・K=1024/14080 で
// 相対誤差 2^-85 以下である（tools/dot2check.cpp．measure_logs/probe_dot2.log で最悪 2^-85.2）．判定したい 53 bit に対して
// 32 bit の余裕がある．素の FP64 DGEMM だと自身の丸め（誤差が
// sqrt(K)*u で増える）より細かい差を解像できず，本当に 53 bit 出ていても
// 過小に報告してしまうので，参照にはこちらを使う．
// 尾崎 I の OZ_FP64 のモデル（ozaki1/ozaki1_autop.hpp の OzDigits）も同じ TwoProd と TwoSum の補償和を持つが，足す順が
// 違う（ここは (c0-(s-bb))+(p-bb)+e，モデルは sl += pe + te）．共通化しないのは意図どおりで，どちらかへ寄せると参照解か
// モデルが選ぶ p のビットが変わる．
inline void reference_gemm(const std::vector<double>& A, const std::vector<double>& B,
                           uint32_t M, uint32_t K, uint32_t N,
                           std::vector<double>& Chi, std::vector<double>& Clo) {
    Chi.assign((size_t)M * N, 0.0);
    Clo.assign((size_t)M * N, 0.0);
    // ブロック化と OpenMP 並列化．素の 3 重ループでは 14080^3 に 1 時間以上かかり，
    // 大きいサイズだけ参照を DGEMM に落とすことになるので，全サイズを同じ
    // Dot2 で検証できるように最適化してある．
    //
    // 加算順序は素の 3 重ループと同じ：どの (i,j) についても k は 0..K-1 の昇順のままなので，
    // 結果は素の 3 重ループとビット単位で一致し，スレッド数にも依存しない．
    //   ib で行を分ける -> スレッドごとに Chi/Clo の行が排他になり競合しない
    //   j0 で列を分ける -> 累算器 IB*JB 個が L2 に収まる
    //   k の内側で i を回す -> B の 1 行が IB 回再利用され，B の読み出しが 1/IB になる
    const double* __restrict__ Ap = A.data();
    const double* __restrict__ Bp = B.data();
    double* __restrict__ Cp = Chi.data();
    double* __restrict__ Lp = Clo.data();
    constexpr uint32_t IB = 8, JB = 512;
    const auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ib = 0; ib < (int64_t)M; ib += IB) {
        // 行・列ブロックの端は 64 bit で数える（uint32_t の ib + IB は M > 2^32 - IB で 0 へ周回して最後の行ブロックを
        // 黙って飛ばし，j0 + JB は N が 2^32 - JB を超えると周回して止まらない）
        const uint32_t i1 = (uint32_t)std::min<int64_t>(ib + IB, (int64_t)M);
        for (uint64_t j0 = 0; j0 < N; j0 += JB) {
            const uint32_t j1 = (uint32_t)std::min<uint64_t>(j0 + JB, N);
            for (uint32_t k = 0; k < K; ++k) {
                const double* __restrict__ Br = Bp + (size_t)k * N;
                for (uint32_t i = (uint32_t)ib; i < i1; ++i) {
                    const double a = Ap[(size_t)i * K + k];
                    double* __restrict__ chi = Cp + (size_t)i * N;
                    double* __restrict__ clo = Lp + (size_t)i * N;
                    for (uint32_t j = (uint32_t)j0; j < j1; ++j) {
                        const double b = Br[j];
                        const double p = a * b, e = std::fma(a, b, -p);   // TwoProd
                        const double c0 = chi[j];
                        const double s = c0 + p, bb = s - c0;             // TwoSum
                        clo[j] += (c0 - (s - bb)) + (p - bb) + e;
                        chi[j] = s;
                    }
                }
            }
        }
    }
    std::printf("Reference: Dot2 (compensated, %d thread)  %.1f s\n",
#ifdef _OPENMP
                omp_get_max_threads(),
#else
                1,
#endif
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count());
}

// 整数の行列積の int64 の参照 C = A*B（A は M×K，B は K×N，行優先．要素は符号つきの整数型）．gemm_int8 の一致の判定と
// gemm_exact_check の期待値に使う．足すのは K 方向の範囲 [k0, k1) だけで（gemm_exact_check の付け替え），整数の和なので
// 足す順によらず値は同じ（無誤差）．B を転置して内側の k を連続にし，行と列を OpenMP で分ける（K=13 万の gemm_exact_check
// を実用の時間で回すため）．
// st を渡せば途中の和の大きさも数える（区間の長さ seg［要素］ごとに途中の和を 0 から数え直す．gemm_exact_check の対ループは
// 1 対ごとに FP32 のアキュムレータを積み直すので，区間は 1 対．呼び出しをまたいで最大を取る）：
//   peak_seg  参照の順序（k の昇順）での区間内の途中の和の最大絶対値
//   sum_seg   区間内の Σ|a_k b_k| の最大（どの順序で足しても途中の和はこれを超えない）
//   peak_acc  区間の和を順に足していったときの途中の和の最大絶対値（対ループの c_5 の Int32 累積）
struct IntRefStats { int64_t peak_seg = 0, sum_seg = 0, peak_acc = 0; };
template <class T>
inline void int_reference_gemm(const std::vector<T>& A, const std::vector<T>& B, uint32_t M, uint32_t K, uint32_t N,
                               uint32_t k0, uint32_t k1, uint32_t seg, std::vector<int64_t>& C,
                               IntRefStats* st = nullptr) {
    static_assert(std::is_integral_v<T> && std::is_signed_v<T>, "int_reference_gemm は符号つきの整数の行列を取る");
    // 区間は 0 <= k0 <= k1 <= K（外れると A・B の外を読む）．呼び出し側の誤りなので止める
    if (k0 > k1 || k1 > K) {
        std::fprintf(stderr, "ERROR: int_reference_gemm の K の区間 [%u, %u) が 0..%u の外\n", k0, k1, K);
        std::abort();
    }
    if (seg == 0) seg = std::max<uint32_t>(1u, k1 - k0);   // 区間を切らない
    C.assign((size_t)M * N, 0);
    std::vector<T> Bt((size_t)N * K);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (uint32_t j = 0; j < N; j++)
        for (uint32_t k = 0; k < K; k++) Bt[(size_t)j * K + k] = B[(size_t)k * N + j];
    int64_t pk_all = 0, sa_all = 0, pa_all = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(max : pk_all, sa_all, pa_all)
#endif
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            const T* a = &A[(size_t)i * K];
            const T* b = &Bt[(size_t)j * K];
            int64_t s = 0, pk = 0, sa = 0, pa = 0;
            // 区間ごとに途中の和を 0 から数える．区間の端は 64 bit で数える（32 bit の kb + seg は seg が 2^32 - k0 以上だと
            // 周回し，k0 より前の項まで足す．FP64 の参照 reference_gemm の列ブロックの端と同じ扱い）
            for (uint64_t kb = k0; kb < k1; kb += seg) {
                const uint64_t ke = std::min<uint64_t>(k1, kb + seg);
                int64_t t = 0, ta = 0;
                for (uint64_t k = kb; k < ke; k++) {
                    const int64_t p = (int64_t)a[k] * b[k];
                    t += p; ta += p < 0 ? -p : p;
                    pk = std::max<int64_t>(pk, t < 0 ? -t : t);
                }
                sa = std::max(sa, ta);
                s += t;
                pa = std::max<int64_t>(pa, s < 0 ? -s : s);
            }
            C[(size_t)i * N + j] = s;
            pk_all = std::max(pk_all, pk); sa_all = std::max(sa_all, sa); pa_all = std::max(pa_all, pa);
        }
    if (st) {
        st->peak_seg = std::max(st->peak_seg, pk_all);
        st->sum_seg = std::max(st->sum_seg, sa_all);
        st->peak_acc = std::max(st->peak_acc, pa_all);
    }
}
// K 全体（区間を切らない）
template <class T>
inline void int_reference_gemm(const std::vector<T>& A, const std::vector<T>& B, uint32_t M, uint32_t K, uint32_t N,
                               std::vector<int64_t>& C) {
    int_reference_gemm(A, B, M, K, N, 0u, K, K, C);
}

}  // namespace oz_eval
