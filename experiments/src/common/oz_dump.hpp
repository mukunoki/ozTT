// SPDX-License-Identifier: Apache-2.0
#pragma once
// 結果の書き出し（host_eval.hpp の傘の下．OZ_DUMP_C．調査用で答えは変えない）
// ---------------------------------------------------------------------------
// 環境変数（値と書き出し先は host_eval.hpp の check_env がデバイスを開く前に検査する）
//   OZ_DUMP_C=f   結果 C と参照解などをファイル f へ書き出す（dump_result．参照解を書くので検証を省く実行
//                 （OZ_NOVERIFY，gemm_bf16 の verify=0）とは併用できない．check_env が止める）
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>
#include <unistd.h>     // access（check_dump_path）
#include <sys/stat.h>   // stat（check_dump_path）
#include "oz_env.hpp"   // つまみの読み方（相対で取り込む理由は host_eval.hpp の冒頭）

namespace oz_eval {

// OZ_DUMP_C=<ファイル>（調査用．答えは変えない）：結果 C・参照 Chi/Clo（行優先 M*N）と，A の行ごと・
// B の列ごとの最大絶対値を書き出す．2 つの実行（実装の変更の前後など）を tools/cmp_dump.cpp が要素ごとに
// 比べる（異なる要素の数，差の ulp，参照に近づいた／遠ざかった数，K max|a| max|b| を単位にした差）．
// 形式：8 バイトの "OZDUMP1"，uint32 の M・N・K・0，double の C・Chi・Clo（各 M*N），amax（M），bmax（N）．
// 書き出せなければ理由を表示して false を返す（呼び出し側がデバイスを閉じてから失敗で終わる．
// ここで exit するとデバイスが開いたまま終わる）．書き出さない（OZ_DUMP_C が無い）なら true．
[[nodiscard]] inline bool dump_result(const std::vector<double>& C, const std::vector<double>& Chi,
                        const std::vector<double>& Clo, const std::vector<double>& A,
                        const std::vector<double>& B, uint32_t M, uint32_t K, uint32_t N) {
    const char* path = oz_env::get("OZ_DUMP_C");
    if (!path || !*path) return true;
    // 見出しの M・N・K と配列の長さが合わない書き出しは cmp_dump が読み違えるので，書かずに失敗にする
    const size_t mn = (size_t)M * N;
    if (C.size() != mn || Chi.size() != mn || Clo.size() != mn || A.size() != (size_t)M * K || B.size() != (size_t)K * N) {
        std::fprintf(stderr, "ERROR: OZ_DUMP_C=%s：配列の長さが M=%u N=%u K=%u と合わない（C %zu・Chi %zu・Clo %zu・"
                             "A %zu・B %zu 要素）\n", path, M, N, K, C.size(), Chi.size(), Clo.size(), A.size(), B.size());
        return false;
    }
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "ERROR: OZ_DUMP_C=%s を開けない\n", path); return false; }
    std::vector<double> am(M, 0.0), bm(N, 0.0);
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t k = 0; k < K; ++k) am[i] = std::max(am[i], std::fabs(A[(size_t)i * K + k]));
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t j = 0; j < N; ++j) bm[j] = std::max(bm[j], std::fabs(B[(size_t)k * N + j]));
    const char magic[8] = {'O', 'Z', 'D', 'U', 'M', 'P', '1', 0};
    const uint32_t hdr[4] = {M, N, K, 0};
    bool ok = std::fwrite(magic, 1, 8, f) == 8 && std::fwrite(hdr, 4, 4, f) == 4;
    for (const auto* v : {&C, &Chi, &Clo}) ok = ok && std::fwrite(v->data(), 8, v->size(), f) == v->size();
    ok = ok && std::fwrite(am.data(), 8, M, f) == M && std::fwrite(bm.data(), 8, N, f) == N;
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) { std::fprintf(stderr, "ERROR: OZ_DUMP_C=%s へ書けない\n", path); return false; }
    std::printf("DUMP,path=%s,M=%u,N=%u,K=%u\n", path, M, N, K);
    return true;
}

// OZ_DUMP_C の書き出し先 path（空でない）を，ファイルを作らずに見る（check_env がデバイスを開く前に呼ぶ）．見ないと，計算を
// 済ませて VERIFY,pass を出した後で開けずに Test Failed になるか，FIFO なら読み手を待って止まる．
//   既にあるなら：ディレクトリでも FIFO・デバイスなどでもない通常のファイルで，書き込める
//   無いなら：末尾が '/' でなく（ディレクトリの指定でない），置き場所のディレクトリに書き込めて中へ入れる
// どれかを外れれば理由を出して終了コード 2 で止める（書き込みと fclose の失敗は dump_result が見る）
inline void check_dump_path(const char* path) {
    const std::string p = path;
    struct stat sb;
    if (p.back() == '/' || (stat(path, &sb) == 0 && S_ISDIR(sb.st_mode)))
        oz_env::fail("OZ_DUMP_C=" + p + " はディレクトリである（書き出すファイルのパスを与える）");
    if (stat(path, &sb) == 0) {
        if (!S_ISREG(sb.st_mode))
            oz_env::fail("OZ_DUMP_C=" + p + " は通常のファイルでない（FIFO・デバイスなど．書き出すファイルのパスを与える）");
        if (access(path, W_OK) != 0) oz_env::fail("OZ_DUMP_C=" + p + " は既にあって書き込めない");
        return;
    }
    const size_t sl = p.find_last_of('/');
    const std::string dir = sl == std::string::npos ? std::string(".") : (sl == 0 ? std::string("/") : p.substr(0, sl));
    if (access(dir.c_str(), W_OK | X_OK) != 0) oz_env::fail("OZ_DUMP_C=" + p + " の置き場所 " + dir + " へ書けない");
}

}  // namespace oz_eval
