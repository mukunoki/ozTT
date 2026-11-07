// dgemmcheck.cpp -- ネイティブの FP64 DGEMM（ホストの CPU の OpenBLAS）の到達精度を，尾崎の実装と同じ入力・参照解・指標で測る
// ---------------------------------------------------------------------------
// 尾崎の実装の到達精度（「effective ~X bit」）を，同じ入力に対するネイティブの DGEMM の到達精度と並べるための道具
// （GEMMul8 の評価と同じ比べ方）．ドライバと同じ部品だけを使う（どれも common/host_eval.hpp の傘の下から取り込む）：
//   入力    oz_input.hpp の gen_operands（seed 既定 7．OZ_PHI などのつまみも同じ環境変数）
//   参照解  oz_reference.hpp の reference_gemm（Dot2 の非評価和 Chi + Clo）
//   指標    oz_accuracy.hpp の accuracy（||C - (Chi+Clo)||_F / ||Chi+Clo||_F の -log2．print_accuracy の行も同じもの）
// C だけを cblas_dgemm（行優先・転置なし・alpha=1・beta=0）で計算する．
//
// OpenBLAS が要るので，tt-metal のイメージ（src/Makefile の HOST_TOOLS）ではなく scripts/Dockerfile.dgemmref の
// イメージで作る（scripts/measure/cpu_dgemm_reference.sh がイメージの用意・コンパイル・掃引をまとめて行う）．フラグは src/Makefile の HOST_OPT で，手で作るなら
//   g++-12 $(make -s -C src print-host-flags) -Isrc tools/dgemmcheck.cpp -o /tmp/dgemmcheck -lopenblas
// g++ の版は測定のイメージ（tt-metalium の ubuntu-22.04．g++ 12.3）に合わせる：入力の乱数は libstdc++ の
// mt19937・uniform_real_distribution・normal_distribution のヘッダの実装で決まるので，版が違うと同じ seed でも
// 別の行列になりうる．出力の ahash・bhash（A・B の FP64 のビット列のハッシュ．oz_accuracy.hpp の fnv64．FNV-1a の更新式で，
// 初期値は標準の FNV-1a と違う）で入力が同じことを確かめられる．
// host_eval.hpp は（oz_timing.hpp を通して）telemetry.hpp を取り込むが，UMD のヘッダが無ければファームウェアの表を読まない版になり，
// テレメトリはここでは使わない（計時の関数を呼ばない）ので，tt-metal 無しでコンパイルできる．
//
//   /tmp/dgemmcheck M N K [seed]      入力のつまみは環境変数（OZ_PHI=φ など）で与える
//   /tmp/dgemmcheck info              BLAS の版・設定・スレッド数の 1 行（BLASINFO,…）だけを出す
//
// 出力（1 条件ごと）：
//   Reference: Dot2 …               reference_gemm が出す行（参照の計算時間）
//   Accuracy : …(effective ~X bit)…  print_accuracy（尾崎の実装と同じ行）
//   CHECKSUM,…                       print_accuracy（chash は DGEMM の結果 C のビット列）
//   DGEMMREF,M=…,N=…,K=…,input=…,seed=…,bits=…,rel_fro=…,maxrel=…,ahash=…,bhash=…,chash=…,blas_threads=…,dgemm_s=…
//     bits は accuracy の ebits（%.2f．尾崎の RESULT 行と同じ桁），rel_fro・maxrel は読み戻すと同じ double になる桁数，
//     input・seed は input_desc（CONFIG 行の input= と同じ書き方．phi:0.5，uniform など）
//   DUMP,…                           OZ_DUMP_C を与えたときだけ．DGEMM の結果 C・参照 Chi/Clo・A の行ごと・B の列ごとの最大絶対値を
//                                    尾崎の各ドライバと同じ形式（oz_dump.hpp の dump_result）で書き出す．tools/cmp_dump.cpp で
//                                    尾崎の結果と要素ごとに比べられる（check_env が置き場所を先に見る）
// 終了コードは，到達精度が kSaneBits 以上なら 0，それ未満か相対誤差が有限でなければ 1（BLAS の結果が壊れている．
// 例えば Ubuntu 22.04 の OpenBLAS 0.3.20 は OPENBLAS_CORETYPE=Cooperlake で Ryzen 7 9700X（Zen 5）に誤った C を返し，
// 到達精度が -0.5 bit になる），OZ_DUMP_C を書けなければ 1，引数・環境変数が不正なら 2．
// 精度の判定は固定の kSaneBits で，合否の基準 OZ_MIN_BITS は使わないので，立っていれば 2 で止める（黙って受けると，
// その基準で判定したつもりになる）．止めるのはつまみの表による拒否（oz_knobs.hpp の reject_unread_input．表の入力の組の
// うち読むプログラム W_DGEMM に無いもの＝OZ_MIN_BITS だけが止まる）．検証を省くことは無い（参照解との比較がこの道具の目的）
// ので OZ_NOVERIFY は見ず，立っていても参照解を作って比べ，OZ_DUMP_C も書き出す．
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "common/host_eval.hpp"
#include "common/oz_knobs.hpp"   // reject_unread_input（つまみの表．scripts/measure/cpu_dgemm_reference.sh は src/common をまるごと渡す）

// openblas_get_config などは OpenBLAS の cblas.h だけが宣言する．cblas.h が別の CBLAS（netlib など）のものなら
// print_info でコンパイルが止まる（別の BLAS とつないだまま「OpenBLAS の DGEMM」と記録しないように）

namespace {

// これを下回る到達精度は DGEMM の丸めでは説明できない（FP64 の DGEMM は本稿の入力で 50 bit 前後）ので，BLAS の誤りとして
// 失敗にする（値は DGEMMREF 行に残す）
constexpr double kSaneBits = 40.0;

// key=value の値に使えない文字（, と空白と = など）を _ にする（provenance.sh の clean と同じ文字の集合）
std::string field_safe(const char* s) {
    std::string out;
    for (const char* p = s ? s : ""; *p; ++p) {
        const char c = *p;
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == ':' || c == '+' || c == '-';
        out += ok ? c : '_';
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? std::string("unknown") : out;
}

void print_info() {
    const int par = openblas_get_parallel();   // 0 逐次・1 pthread・2 OpenMP
    std::printf("BLASINFO,blas_config=%s,blas_core=%s,blas_threads=%d,blas_parallel=%s,omp_threads=%d\n",
                field_safe(openblas_get_config()).c_str(), field_safe(openblas_get_corename()).c_str(),
                openblas_get_num_threads(), par == 0 ? "sequential" : par == 1 ? "pthread" : par == 2 ? "openmp" : "unknown",
#ifdef _OPENMP
                omp_get_max_threads()
#else
                1
#endif
    );
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "info")) {
        print_info();
        return 0;
    }
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "usage: %s M N K [seed]   （入力は OZ_PHI などの環境変数）\n"
                             "       %s info\n", argv[0], argv[0]);
        return 2;
    }
    const uint32_t M = oz_env::arg_u32(argc, argv, 1, 0), N = oz_env::arg_u32(argc, argv, 2, 0),
                   K = oz_env::arg_u32(argc, argv, 3, 0);
    const uint32_t seed = oz_env::arg_u32(argc, argv, 4, oz_eval::kDefaultSeed);
    // cblas の次元は int なので，それを超える大きさ・0 は拒む（黙って切り詰めた行列を測らない）
    const uint32_t imax = 0x7FFFFFFFu;
    if (!M || !N || !K || M > imax || N > imax || K > imax) {
        std::fprintf(stderr, "ERROR: M,N,K は 1 以上 %u 以下（M=%u N=%u K=%u）\n", imax, M, N, K);
        return 2;
    }
    // 合否の基準 OZ_MIN_BITS（表の入力の組で W_DGEMM が読まないもの）が立っていれば止める（冒頭．終了コード 2）
    oz_knobs::reject_unread_input(oz_knobs::W_DGEMM, "dgemmcheck",
                                  ("判定は固定の " + std::to_string((int)kSaneBits) + " bit").c_str());
    // 入力のつまみ（OZ_PHI など）と OZ_DUMP_C の置き場所の検査．不正なら exit(2)．検証はいつも行うので verify=true を渡す
    // （既定の verify_enabled() のままだと，環境に残った OZ_NOVERIFY を理由に OZ_DUMP_C を誤って断る）
    oz_eval::check_env(K, true);

    std::vector<double> A, B, Chi, Clo;
    oz_eval::gen_operands(A, B, M, K, N, seed);
    oz_eval::reference_gemm(A, B, M, K, N, Chi, Clo);

    std::vector<double> C((size_t)M * N, 0.0);
    const auto t0 = std::chrono::steady_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K, 1.0, A.data(), (int)K,
                B.data(), (int)N, 0.0, C.data(), (int)N);
    const double dgemm_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const oz_eval::Acc acc = oz_eval::accuracy(C, Chi, Clo);
    oz_eval::print_accuracy(acc);
    std::printf("DGEMMREF,M=%u,N=%u,K=%u,%s,bits=%.2f,rel_fro=%s,maxrel=%s,ahash=%016llx,bhash=%016llx,"
                "chash=%016llx,blas_threads=%d,dgemm_s=%.3f\n",
                M, N, K, oz_eval::input_desc(seed, false).c_str(), acc.ebits, oz_eval::rt_double(acc.rel_fro).c_str(),
                oz_eval::rt_double(acc.maxrel).c_str(), (unsigned long long)oz_eval::fnv64(A), (unsigned long long)oz_eval::fnv64(B),
                (unsigned long long)acc.chash, openblas_get_num_threads(), dgemm_s);
    // OZ_DUMP_C（調査用．与えたときだけ書く．書けなければ失敗とする）．DGEMMREF 行の後に置き，書けなくても記録は残す
    const bool dump_ok = oz_eval::dump_result(C, Chi, Clo, A, B, M, N, K);
    if (!std::isfinite(acc.rel_fro) || !(acc.ebits >= kSaneBits)) {
        std::printf("ERROR: 到達精度 %.2f bit が %.0f bit を下回る（BLAS が誤った結果を返した）\n", acc.ebits, kSaneBits);
        std::fflush(stdout);
        return 1;
    }
    std::fflush(stdout);
    return dump_ok ? 0 : 1;
}
