// SPDX-License-Identifier: Apache-2.0
// ホスト精度指標の回帰検査．デバイス不要：相殺したDot2，低位語，零，極小値，非有限値．
#include "common/oz_accuracy.hpp"
#include "common/oz_reference.hpp"
#include <quadmath.h>
#include <random>

int main() {
    int bad = 0, checked = 0;
    auto require = [&](bool ok, const char* name) {
        ++checked;
        if (!ok) { ++bad; std::fprintf(stderr, "FAIL: %s\n", name); }
    };
    auto acc = [](double c, double hi, double lo = 0) {
        return oz_eval::accuracy({c}, {hi}, {lo});
    };
    // 本番Dot2から実際に非正規化の参照を得る（32x32x32，厳密な積はすべて0）．
    std::vector<double> A(32 * 32, 1), B(32 * 32), hi, lo;
    const double terms[] = {0x1p107, 0x1p53, -0x1p107, -0x1p53};
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 32; ++j) B[k * 32 + j] = terms[k];
    oz_eval::reference_gemm(A, B, 32, 32, 32, hi, lo);
    require(hi[0] == -0x1p53 && lo[0] == 0x1p53, "Dot2 cancellation fixture");
    require(std::isinf(oz_eval::accuracy(std::vector<double>(1024, 1), hi, lo).rel_fro),
            "wrong output against zero reference must fail");
    require(oz_eval::accuracy(std::vector<double>(1024), hi, lo).rel_fro == 0,
            "exact cancellation must pass");
    const auto mixed = oz_eval::accuracy({1, 1}, {-0x1p53, 1}, {0x1p53, 0});
    require(mixed.rel_fro == 1 && mixed.ebits == 0, "nonzero reference norm must detect wrong element");
    require(acc(1, 1, 0x1p-54).rel_fro == 0x1p-54, "keep low reference word");
    require(acc(0, 0x1p-600).rel_fro == 1, "tiny nonzero error");
    require(acc(0x1p-1074, 0x1p-1074).rel_fro == 0, "exact subnormal");
    require(acc(0, 0x1p-1074).rel_fro == 1, "wrong subnormal");
    require(acc(0, 0).rel_fro == 0, "exact zero");
    require(std::isinf(acc(1, 0).rel_fro), "wrong zero");
    for (double v : {INFINITY, -INFINITY, NAN}) {
        require(std::isinf(acc(v, 1).rel_fro), "nonfinite candidate");
        require(std::isinf(acc(1, v).rel_fro), "nonfinite high reference");
        require(std::isinf(acc(1, 1, v).rel_fro), "nonfinite low reference");
    }
    require(std::isinf(oz_eval::accuracy({}, {}, {}).rel_fro), "empty result");
    require(std::isinf(oz_eval::accuracy({1}, {1, 2}, {0}).rel_fro), "length mismatch");
    // 二つのdoubleの和を113bitで厳密に保持できる範囲で，独立な__float128参照と照合．
    // 非正規化のペアと強い相殺を含め，指数を変えてノルムのスケーリングも通す．
    std::mt19937_64 rng(7);
    for (int i = 0; i < 10000; ++i) {
        const int e = int(rng() % 1001) - 500;
        const double h = std::ldexp(1 + double(rng() % 65536) / 65536, e);
        const double l = (i % 2) ? -h + std::ldexp(double(rng() % 1024), e - 20)
                                  : std::ldexp(double(rng() % 65536), e - 70);
        const __float128 q = (__float128)h + (__float128)l;
        const double c = std::nextafter((double)q, INFINITY);
        const __float128 want = q == 0 ? HUGE_VALQ : fabsq(((__float128)c - q) / q);
        const double got = acc(c, h, l).rel_fro;
        require(isinfq(want) ? std::isinf(got) : fabsq((__float128)got - want) <= want * 0x1p-48Q,
                "quad precision oracle");
    }
    std::printf("accuracy_check: %d checks, %d failures\n", checked, bad);
    return bad != 0;
}
