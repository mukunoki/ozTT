// dot2check.cpp -- 参照解 Dot2 の精度を __float128 の参照和と突き合わせる
// ---------------------------------------------------------------------------
// 論文が「参照解の相対誤差は 2^-85 以下」と述べる根拠を再現するための検査．
// ドライバと同じ入力生成（oz_input.hpp の gen_operands，seed 既定 7）を使い，各出力要素で
//
//     Dot2 の非評価和 (Chi + Clo)   対   __float128 で同じ順序に足した和
//
// の相対誤差を取り，最悪値を出す．__float128 は有効仮数 113 bit である．参照側の
// 相対誤差は gamma_K * sum|a_k b_k| / |sum a_k b_k| で抑えられ，相殺があれば条件数が
// 掛かるので K だけでは決まらない．一律の保証は置かず，測った値を報告する
// （本稿の入力では，最悪の要素でも参照側の相対誤差は 3.4e-31 程度である）．
// 参照和がちょうど 0 の要素は相対誤差が定義できないので別に数え，Dot2 の和も 0 であることを見る（0 でない要素があれば
// その条件の最悪相対誤差を inf とし，DOT2-ZERO 行に件数と |Chi+Clo| の最大を出して終了コード 1．0 の要素が無ければ
// DOT2-ZERO 行は出さない）．参照和・Dot2 の和・相対誤差のどれかが有限でない要素も最悪値の外で数え（NaN は大小の比較が
// 偽になるので，最悪値に入れようとすると黙って落ちる），1 個でもあればその条件の最悪相対誤差を nan とし，DOT2-NONFINITE 行に
// 件数を出して終了コード 1（無ければ行は出さない）．
//
// src/Makefile の HOST_TOOLS で src/build/dot2check を作る（コンテナの中で make -C src dot2check．
// scripts/diagnose/probes.sh の probe_dot2 はこれを使う）．フラグは Makefile の HOST_CXXFLAGS で，-ffp-contract=off が要る
// （積和が勝手に融合されると Dot2 の誤差なし変換の残差が消え，検査そのものが変わる）：
//   ./src/build/dot2check            # 既定の掃引（phi=0,2,4,8,12,16，K=1024/14080．M=N=64 の 12 条件）
//   ./src/build/dot2check M N K phi  # 1 条件だけ（測定のモード）
// 引数は 0 個か 4 個だけを受ける（それ以外は終了コード 2．足りないときに既定の掃引へ落ちて，指定した条件を
// 測ったつもりにならないように）．φ は読み戻すと同じ double になる桁数で入力の生成へ渡し，DOT2 行にもその桁数で出す．
// 終了コード：既定の掃引は論文の条件を判定するモードで，12 条件の最悪相対誤差が有限で 2^-85 以下（kPaperBound）なら 0，
// そうでなければ 1（判定の行 DOT2-BOUND を最後に出す）．1 条件のモードは任意の入力を測るためのもので，誤差の大きさは
// 判定しない（参照和が 0 なのに Dot2 の和が 0 でない要素か，有限でない要素があるときだけ 1）．
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <quadmath.h>
#include "common/host_eval.hpp"
#include "common/oz_knobs.hpp"   // reject_unused_input（つまみの表の入力・判定・書き出しの組）
using oz_eval::gen_operands;
using oz_eval::reference_gemm;

// 論文（「測定条件」）が述べる参照解の相対誤差の上界．既定の掃引の終了コードはこれで判定する
constexpr double kPaperBound = 0x1p-85;

// 参照和が 0 の要素と有限でない要素の集計（相対誤差の外で見る）
struct ZeroTally { long long nzero = 0, nbad = 0; double maxabs = 0; long long nf_ref = 0, nf_dot2 = 0, nf_rel = 0; };
// 1 条件を測る．戻り値は最悪相対誤差（有限でない要素があれば nan，参照和が 0 なのに Dot2 の和が 0 でない要素があれば inf）．
static double check(uint32_t M, uint32_t N, uint32_t K, double phi, uint32_t seed, ZeroTally& zt) {
    const std::string ph = oz_eval::rt_double(phi);
    setenv("OZ_PHI", ph.c_str(), 1);
    std::vector<double> A, B, Chi, Clo;
    gen_operands(A, B, M, K, N, seed);
    reference_gemm(A, B, M, K, N, Chi, Clo);

    double worst = 0.0, zmax = 0.0;
    // nf_elem は有限でない要素の数（1 要素は 1 回だけ数える），nf_ref・nf_dot2・nf_rel はその理由ごとの内訳（参照和と Dot2 の和の
    // 両方が有限でない要素は両方に数える）
    long long nzero = 0, nbad = 0, nf_elem = 0, nf_ref = 0, nf_dot2 = 0, nf_rel = 0;
#pragma omp parallel for reduction(max : worst, zmax) reduction(+ : nzero, nbad, nf_elem, nf_ref, nf_dot2, nf_rel) schedule(static)
    for (long long ij = 0; ij < (long long)M * N; ij++) {
        const uint32_t i = (uint32_t)(ij / N), j = (uint32_t)(ij % N);
        // 参照和：Dot2 と同じ k 昇順で __float128 に足す
        __float128 s = 0;
        for (uint32_t k = 0; k < K; k++)
            s += (__float128)A[(size_t)i * K + k] * (__float128)B[(size_t)k * N + j];
        const __float128 got = (__float128)Chi[ij] + (__float128)Clo[ij];
        if (!finiteq(s) || !finiteq(got)) { nf_elem++; nf_ref += !finiteq(s); nf_dot2 += !finiteq(got); continue; }
        if (s == 0) {                      // 相対誤差は定義できない．Dot2 の和も 0 でなければ誤り
            nzero++;
            if (got != 0) { nbad++; zmax = std::max(zmax, (double)fabsq(got)); }
            continue;
        }
        const double r = (double)fabsq((got - s) / s);
        if (!std::isfinite(r)) { nf_elem++; nf_rel++; continue; }   // double へ移して溢れた相対誤差
        if (r > worst) worst = r;
    }
    zt = {nzero, nbad, zmax, nf_ref, nf_dot2, nf_rel};
    if (nzero)
        std::printf("DOT2-ZERO,%u,%u,%u,%s,参照和が 0 の要素 %lld 個のうち Dot2 の和が 0 でない %lld 個（|Chi+Clo| の最大 %.3e）\n",
                    M, N, K, ph.c_str(), nzero, nbad, zmax);
    const long long nf = nf_elem;
    if (nf)
        std::printf("DOT2-NONFINITE,%u,%u,%u,%s,有限でない要素 %lld 個（参照和 %lld 個・Dot2 の和 %lld 個・相対誤差 %lld 個．"
                    "参照和と Dot2 の和の両方が有限でない要素は両方に数える）\n",
                    M, N, K, ph.c_str(), nf, nf_ref, nf_dot2, nf_rel);
    return nf ? NAN : nbad ? INFINITY : worst;
}

int main(int argc, char** argv) {
    const uint32_t seed = 7;
    if (argc != 1 && argc != 5) {
        std::fprintf(stderr, "usage: %s            （既定の掃引．論文の条件 2^-85 以下を終了コードで判定する）\n"
                             "       %s M N K phi  （1 条件だけ測る）\n", argv[0], argv[0]);
        return 2;
    }
    // 入力は φ だけで作る（check が条件ごとに OZ_PHI を立て直す）．ほかの入力のつまみのうち OZ_EXPDIFF（と OZ_EXPK）は OZ_PHI より
    // 優先されて別の入力になり，残り（OZ_CONST など）は OZ_PHI があると使われない（oz_input.hpp の gen_operands の優先順位）．
    // 親の OZ_PHI は check が上書きし，合否の基準 OZ_MIN_BITS と書き出し OZ_DUMP_C は持たない．どれも指定とは別のものを
    // 測ったつもりになるので，立っていれば止める．一覧は入力のつまみを持たないドライバと同じもの（oz_knobs.hpp の
    // reject_unused_input．表につまみを足したときに一緒に止まる）
    oz_knobs::reject_unused_input("dot2check");
    if (argc == 5) {
        // 10 進の整数として全体を読み切れて範囲に入るときだけ受け付ける（src/common/oz_env.hpp．atoi では "64x" が 64 に，
        // "x" が 0 になる．負号はどこにあっても拒む）．範囲の外は終了コード 2
        const uint32_t M = oz_env::arg_u32(argc, argv, 1, 0, 1, 1u << 20, "M"), N = oz_env::arg_u32(argc, argv, 2, 0, 1, 1u << 20, "N"),
                       K = oz_env::arg_u32(argc, argv, 3, 0, 1, 1u << 30, "K");
        double phi = 0.0;
        if (!oz_env::to_f64(argv[4], phi) || phi < 0) {
            std::fprintf(stderr, "ERROR: phi=%s は 0 以上の有限の数ではない\n", argv[4]);
            return 2;
        }
        ZeroTally zt;
        const double w = check(M, N, K, phi, seed, zt);
        std::printf("DOT2,%u,%u,%u,%s,%.3e,%.1f\n", M, N, K, oz_eval::rt_double(phi).c_str(), w, -std::log2(w));
        return (zt.nbad || !std::isfinite(w)) ? 1 : 0;
    }
    // 既定の掃引．M=N は小さく取り（1 条件あたり M*N*K の __float128 積和が要る），
    // K は論文が測る両端に合わせる．
    std::printf("DOT2,M,N,K,phi,maxrel,bits\n");
    double worst = 0.0;
    long long nbad = 0;
    bool finite = true;
    for (uint32_t K : {1024u, 14080u})
        for (double phi : {0.0, 2.0, 4.0, 8.0, 12.0, 16.0}) {
            ZeroTally zt;
            const double w = check(64, 64, K, phi, seed, zt);
            nbad += zt.nbad;
            finite = finite && std::isfinite(w);
            std::printf("DOT2,64,64,%u,%s,%.3e,%.1f\n", K, oz_eval::rt_double(phi).c_str(), w, -std::log2(w));
            std::fflush(stdout);
            if (w > worst) worst = w;
        }
    std::printf("DOT2,worst,,,,%.3e,%.1f\n", worst, -std::log2(worst));
    // 論文の条件：12 条件のすべてで最悪相対誤差が有限で 2^-85 以下（参照和が 0 の要素は Dot2 の和も 0）
    const bool ok = nbad == 0 && finite && worst <= kPaperBound;
    std::printf("DOT2-BOUND,2^%.0f,%s（12 条件の最悪 2^%.1f%s）\n", std::log2(kPaperBound), ok ? "PASS" : "FAIL",
                std::log2(worst), nbad ? "．参照和が 0 の要素で Dot2 の和が 0 でないものがある" : "");
    return ok ? 0 : 1;
}
