// FP64 の値を FP32 3 本組（hi・mid・lo）へ分けたとき，無誤差に運べる指数の下端を実測する
// ---------------------------------------------------------------------------
// 要素 v = 2^e*(1.m) を hi/mid/lo の 3 本へ分けると，下位 limb は大きさ 2^(e-48) 前後でビットは 2^(e-52) まで
// 伸びる．そのため limb が FP32 の表せる下端を割ると下から桁が落ちる．下端は FP32 の非正規化数の扱いで変わるので，
// 2 通りを並べて出す：
//   IEEE  ホストの FP32（段階的下位あふれ．非正規化数を保つ）．下限 2^-149 を割ると崩れる（e = -97 まで無誤差，-98 から崩れる）
//   FTZ   SFPU のように非正規化数を符号つきの 0 に潰す模擬（checks/sfpu_tf_emu.hpp の ftz）．limb が正規数の下端
//         2^-126 を割ると崩れるので，下端は IEEE より 23 高い（e = -74 まで無誤差，-75 から崩れる）
// どちらも「ホストの最近接丸めで 3 本へ分けたとき」の性質で，デバイスの 3 本組の作り方（slice_f64_reader.cpp と SFPU）
// そのものではない．common/kernels/f64_tf.h の「下端 2^-97 の limb も 1 bit も落ちない」はこれとは別の量で，
// 出口の 128 bit 固定小数へのシフトの性質（limb の指数 e に対し最下位ビットが 2^(e-23) >= 2^-124 で落ちない）である．
// 仮数は 52 bit の乱数をそのまま [1,2) の小数部にする（53 bit を 1 に足すと FP64 へ丸め直されて最下位ビットが偶数に偏る）．
// ビルドはリポジトリ直下で（コンテナの中で．フラグは src/Makefile の HOST_OPT，取り込みの根は src．どのスクリプトも作らず回さない）
//     c++ $(make -s -C src print-host-flags) -Isrc tools/study/tfrange.cpp -lquadmath -o /tmp/tfrange
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include "checks/sfpu_tf_emu.hpp"
using sfpu_emu::ftz;
int main(int argc, char** argv) {
    if (argc != 1) { std::fprintf(stderr, "usage: %s（引数は取らない）\n", argv[0]); return 2; }
    std::mt19937_64 rng(1);
    std::printf(" e   無誤差だった割合（各 200000 本のランダム仮数）\n");
    std::printf("       IEEE（非正規化数を保つ）  FTZ（非正規化数を 0 に潰す）\n");
    for (int e = -60; e >= -102; e--) {
        int ok = 0, okz = 0; const int N = 200000;
        for (int t = 0; t < N; t++) {
            const double m = 1.0 + std::ldexp((double)(rng() >> 12), -52);   // [1,2) の 52 bit（無誤差）
            const double v = std::ldexp(m, e);
            {
                const float hi = (float)v; const double rem = v - (double)hi;
                const float mid = (float)rem; const float lo = (float)(rem - (double)mid);
                if ((double)hi + (double)mid + (double)lo == v) ok++;
            }
            {
                const float hi = ftz((float)v); const double rem = v - (double)hi;
                const float mid = ftz((float)rem); const float lo = ftz((float)(rem - (double)mid));
                if ((double)hi + (double)mid + (double)lo == v) okz++;
            }
        }
        std::printf("%4d   %6.2f%%                  %6.2f%%%s\n", e, 100.0 * ok / N, 100.0 * okz / N,
                    e == -97 ? "   <- IEEE の下端" : e == -74 ? "   <- FTZ の下端" : "");
    }
    return 0;
}
