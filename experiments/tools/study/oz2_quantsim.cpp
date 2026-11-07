// oz2_quantsim.cpp -- 尾崎スキーム II の量子化（幅 b）だけの誤差を，実機と同じ入力の全要素で求める
// ---------------------------------------------------------------------------
// A・B を行ごと／列ごとに 2 のべき 2^e（最大絶対値以上の最小の 2 のべき）で正規化し，b bit の整数へ
// 最近接で丸める（std::rint．既定の丸めモードなので同点は偶数へ倒す．本番の桁作り ozaki2/kernels/oz2_digits.h の Rnd
// （tt-llk の round_tile．最近接偶数）と tools/splitsim.cpp の rintq と同じ向き）．再構成（CRT）は厳密とみなし，
// そのときの ||E||_F/||C||_F を b ごとに出す．
//     E = dA*B + Ahat*dB      （Ahat = A + dA．dA・dB は丸めの差で，FP64 で厳密に表せる）
// E は小さいので FP64 の行列積で十分な精度で求まり，||C||_F も FP64 の DGEMM で足りる（1% の精度で十分）．
// 実機の結果がこの値に一致すれば，その b では再構成の誤差は効いていない（docs/accuracy.md の φ=16）．
// tools/splitsim.cpp も同じ入力の全要素で量子化の誤差を厳密演算で求めるが，||E||_F/||C||_F だけを出す．こちらは FP64 の
// 行列積で E を作り，誤差の最も大きい要素と，その要素で誤差を生んだ項も出す．
//
// 入力は oz_input.hpp の gen_operands（seed 7．OZ_PHI などのつまみも同じ環境変数）で，ドライバと同じ行列になる．
// 乱数は libstdc++ の版で変わりうるので，tools/dgemmcheck.cpp と同じく g++-12 と OpenBLAS のイメージ（scripts/measure/cpu_dgemm_reference.sh が作る
// localhost/ttgemm-dgemmref:df-<scripts/Dockerfile.dgemmref の sha256 の先頭 12 桁>．無ければ measure/cpu_dgemm_reference.sh を 1 度流すと作る）で作る．
// フラグは src/Makefile の HOST_OPT（ホストで make -s -C src print-host-flags が出す）．リポジトリ直下で：
//   IMG=localhost/ttgemm-dgemmref:df-$(sha256sum scripts/Dockerfile.dgemmref | cut -c1-12)
//   FLAGS=$(make -s -C src print-host-flags)
//   podman run --rm -v "$PWD":/work:ro -v <出力先>:/out -w /work -e OZ_PHI=16 "$IMG" bash -c "g++-12 $FLAGS -Isrc
//     tools/study/oz2_quantsim.cpp -o /out/oz2_quantsim -lopenblas -lquadmath && /out/oz2_quantsim 4096 4096 4096 54 60 71 80 84"
// 出力（b ごと）：
//   QUANT,b=…,rel=…,bits=…,zeroA=…,zeroB=…    zeroA・zeroB は 0 に丸められた非零要素の割合
//   その前の 2 行に，|E| が最大の要素 (I,J) と，そこで誤差の最も大きい項 k（a・b の大きさと，行・列の最大値との比）
// ノルムの二乗和と，ノルムどうしの比・二乗の比は __float128 で求める（double のままだと大きい値は溢れ，小さい値は下位へ
// あふれて 0 になる．||C||_F が FP64 の最大値を超える入力でも比は有限に出る）．表示は double に直す（溢れる大きさは inf と出る）．
// ||C||_F が 0 なら相対誤差は定義できないので rel=undefined とし，絶対誤差 abs=||E||_F を出す．C か E に有限でない要素が
// あれば（入力の値域を外れた積），その旨を出して終了コード 1．
#include <cblas.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <quadmath.h>
#include <vector>
#include "common/host_eval.hpp"
#include "common/oz_knobs.hpp"   // つまみの表（読まないつまみを止める）
#include "../oz_sim.h"           // exp_ceil（2 のべきのスケールの指数．tools/splitsim.cpp と同じ関数）

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "使い方: oz2_quantsim M N K b [b ...]  （入力のつまみは環境変数．OZ_PHI=φ など）\n"); return 2; }
    // 引数は 10 進の整数として全体を読み切れたときだけ受け付ける（src/common/oz_env.hpp．atoi では "4096x" が 4096 に，"x" が
    // 0 になる．範囲の外は終了コード 2）．M・N・K は cblas の int に収まる範囲，b は 1〜200
    for (int a = 4; a < argc; a++) (void)oz_env::arg_u32(argc, argv, a, 0, 1, 200, "b");
    const uint32_t M = oz_env::arg_u32(argc, argv, 1, 0, 1, 0x7FFFFFFF, "M"), N = oz_env::arg_u32(argc, argv, 2, 0, 1, 0x7FFFFFFF, "N"),
                   K = oz_env::arg_u32(argc, argv, 3, 0, 1, 0x7FFFFFFF, "K");
    // 入力のつまみ（OZ_PHI など）は gen_operands が読んで検査する．結果の書き出し（OZ_DUMP_C）と合否の基準（OZ_MIN_BITS）は
    // 持たないので，立っていれば止める．空の OZ_DUMP_C は「書き出さない」なので止めない（つまみの表の入力の組の拒否
    // oz_knobs.hpp の reject_unread_input と同じ扱い．入力の作り方のつまみと OZ_DUMP_C は表の読むプログラムが同じなので，
    // 読むプログラムのビットではこの 2 つだけを選べず，名前を表から引いて止める）
    oz_knobs::reject_named("OZ_DUMP_C", "oz2_quantsim", "結果の書き出しを持たない");
    oz_knobs::reject_named("OZ_MIN_BITS", "oz2_quantsim", "合否の判定を持たない");
    std::vector<double> A, B;
    oz_eval::gen_operands(A, B, M, K, N);
    // 全要素が零の行・列は正規化しない（丸めの差も 0 になる）
    std::vector<int> eA(M, 0), eB(N, 0);
    for (uint32_t i = 0; i < M; i++) {
        double m = 0; for (uint32_t k = 0; k < K; k++) m = std::max(m, std::fabs(A[(size_t)i * K + k]));
        if (m > 0) eA[i] = oz_sim::exp_ceil(m);
    }
    for (uint32_t j = 0; j < N; j++) {
        double m = 0; for (uint32_t k = 0; k < K; k++) m = std::max(m, std::fabs(B[(size_t)k * N + j]));
        if (m > 0) eB[j] = oz_sim::exp_ceil(m);
    }
    std::vector<double> C((size_t)M * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C.data(), N);
    // 二乗和は __float128（指数 15 bit）で足すので，double のどの値の二乗も溢れず下位へあふれない．ノルムも __float128 のまま返す
    auto fro = [](const std::vector<double>& v, size_t& nonfinite) {
        __float128 s = 0; nonfinite = 0;
        for (double x : v) { if (!std::isfinite(x)) { nonfinite++; continue; } s += (__float128)x * (__float128)x; }
        return sqrtq(s);
    };
    size_t nfC = 0;
    const __float128 nC = fro(C, nfC);
    double mxC = 0;
    for (double v : C) if (std::isfinite(v)) mxC = std::max(mxC, std::fabs(v));
    int rc = 0;
    if (nfC) { std::printf("ERROR: C に有限でない要素が %zu 個ある（入力の値域が FP64 の積和を溢れる）\n", nfC); rc = 1; }
    if (nC > 0) std::printf("||C||_F=%.6e max|C|=%.6e (max/norm=%.6f)  %s\n", (double)nC, mxC, (double)((__float128)mxC / nC),
                            oz_eval::input_desc().c_str());
    else        std::printf("||C||_F=0（相対誤差は定義できないので絶対誤差を出す）  %s\n", oz_eval::input_desc().c_str());
    std::vector<double> dA(A.size()), dB(B.size()), Ah(A.size()), E((size_t)M * N);
    for (int a = 4; a < argc; a++) {
        const int b = (int)oz_env::arg_u32(argc, argv, a, 0, 1, 200, "b");   // 上で範囲を検査済み
        size_t zA = 0, zB = 0;
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t k = 0; k < K; k++) {
                const size_t z = (size_t)i * K + k;
                const double x = std::ldexp(A[z], b - eA[i]), r = std::rint(x);   // 2 のべき倍は無誤差，r - x も無誤差
                dA[z] = std::ldexp(r - x, eA[i] - b); Ah[z] = A[z] + dA[z];
                if (r == 0 && A[z] != 0) zA++;
            }
        for (uint32_t k = 0; k < K; k++)
            for (uint32_t j = 0; j < N; j++) {
                const size_t z = (size_t)k * N + j;
                const double x = std::ldexp(B[z], b - eB[j]), r = std::rint(x);
                dB[z] = std::ldexp(r - x, eB[j] - b);
                if (r == 0 && B[z] != 0) zB++;
            }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, dA.data(), K, B.data(), N, 0.0, E.data(), N);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, Ah.data(), K, dB.data(), N, 1.0, E.data(), N);
        size_t nfE = 0;
        const __float128 nE = fro(E, nfE);
        double mE = 0; size_t zE = 0;
        for (size_t z = 0; z < E.size(); z++)
            if (std::isfinite(E[z]) && std::fabs(E[z]) > mE) { mE = std::fabs(E[z]); zE = z; }
        if (nfE) { std::printf("ERROR: b=%d の誤差 E に有限でない要素が %zu 個ある\n", b, nfE); rc = 1; }
        if (mE > 0 && nC > 0) {
            const uint32_t I = zE / N, J = zE % N;
            const double cij = C[zE];
            const __float128 mEq = mE;
            std::printf("  max|E| at (%u,%u): |E|/||C||=2^%.2f  |E|/|C_ij|=2^%.2f  ||E||^2 に占める割合 %.3f  行の最大 2^%d  列の最大 2^%d\n",
                        I, J, (double)log2q(mEq / nC), std::log2(mE / std::fabs(cij)), (double)(mEq * mEq / (nE * nE)), eA[I], eB[J]);
            double bt = 0; uint32_t bk = 0;
            for (uint32_t k = 0; k < K; k++) {
                const double t = std::fabs(dA[(size_t)I * K + k] * B[(size_t)k * N + J]) + std::fabs(Ah[(size_t)I * K + k] * dB[(size_t)k * N + J]);
                if (t > bt) { bt = t; bk = k; }
            }
            const double av = std::fabs(A[(size_t)I * K + bk]), bv = std::fabs(B[(size_t)bk * N + J]);
            std::printf("  誤差の最も大きい項 k=%u: |a|=2^%.2f（行の最大の 2^%.2f 倍） |b|=2^%.2f（列の最大の 2^%.2f 倍） "
                        "dA*b=2^%.2f  Ahat*dB=2^%.2f（||C||_F に対する比）\n",
                        bk, std::log2(av), std::log2(av) - eA[I], std::log2(bv), std::log2(bv) - eB[J],
                        (double)log2q((__float128)std::fabs(dA[(size_t)I * K + bk] * B[(size_t)bk * N + J]) / nC),
                        (double)log2q((__float128)std::fabs(Ah[(size_t)I * K + bk] * dB[(size_t)bk * N + J]) / nC));
        }
        if (nC > 0)
            std::printf("QUANT,b=%d,rel=%.4e,bits=%.2f,zeroA=%.4f,zeroB=%.4f\n", b, (double)(nE / nC), -(double)log2q(nE / nC),
                        (double)zA / A.size(), (double)zB / B.size());
        else
            std::printf("QUANT,b=%d,rel=undefined,abs=%.4e,zeroA=%.4f,zeroB=%.4f\n", b, (double)nE,
                        (double)zA / A.size(), (double)zB / B.size());
        std::fflush(stdout);
    }
    return rc;
}
