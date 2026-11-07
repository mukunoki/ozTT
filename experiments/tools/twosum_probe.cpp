// SPDX-License-Identifier: Apache-2.0
//
// scheme II の再構成（分数形 CRT）をホスト上で模擬し，tfAddSmall を入れても
// frac の精度が落ちないことを確かめる【検査用】
// ---------------------------------------------------------------------------
//     make -C src twosum_probe            （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     ./src/build/twosum_probe [K] [seed] [試行数]
//
// 何を確かめるか
//   recon_crt.cpp の足し込みは，末尾 NSMALL_FRAG 本の断片だけ tfAddS の第 1 段
//   TwoSum(a0,v) を飛ばして tfAddSmall で足す（15 演算 -> 8 演算）．
//   本数はホスト（ozaki2_main.cpp）が断片の大きさから数える．
//   ここでは同じ規則で NSMALL_FRAG を決め，NSMALL_FRAG=0（全部 tfAddS）と並べて，
//   frac の絶対誤差が悪化しないことを見る．
//   足し込みの手続き・断片・limb の本数は本番と同じもの（tools/oz_sim.h の crt_accumulate と make_recon_kernels．SFPU の
//   演算は checks/sfpu_tf_emu.hpp の模擬で，非正規化数を 0 に潰す）を使い，入口だけを変える：本番は r_i*y_i の
//   ModRound で k_i を作るが，ここは k_i を対称剰余から直接振る（frac の誤差を決めるのは k_i と断片だけなので）．
//
// 比較用の逆転スイッチ（どれも 0/1 で与える．0/1 以外は止める）
//   TWOSUM_PROBE_ABS=1   NSMALL_FRAG の判定を絶対値 2^-30 固定にした誤った規則（FAIL するのが期待どおり）
//   TWOSUM_PROBE_FAST=1  tfAddSmall の第 2 段を FastTwoSum に置き換える
//   TWOSUM_PROBE_ALL=1   全断片を tfAddSmall で足す（OZ2_ALLSMALL の模擬）
//
// 数えないもの
//   「第 1 段が恒等になる割合」や「|a1| >= |v| が成り立つ割合」は数えない．
//   これは第 2 段に Dekker の FastTwoSum を使うときの前提で，その前提は a0 の
//   下からの評価が要るので成り立たない（sfpu_tf_proc.h の tfAddSmall を参照）．
//   本番の実装は第 2 段も Knuth の TwoSum なので，その割合には意味が無い．
//   法は INT8 系列に固定せず実際に選ばれる法集合を使い，剰余は非負だけでなく
//   対称剰余 [-m/2, m/2] で振る．
//
// 2 パス構成も見る．第 2 パスの断片は第 1 パスが 60 bit 取った残りなので
// 2^-60 前後しかなく，NSMALL_FRAG の判定を絶対値で固定すると先頭まで「小さい」に
// 数えてしまう（limb 1 本ぶん落ちる）．本番の規則は最大項に対する相対なので
// そうならないことを，ここで押さえる．
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <quadmath.h>
#include "oz_sim.h"

using oz_sim::CrtKernel;
using oz_sim::Dst;

struct Tri { float a0 = 0, a1 = 0, a2 = 0; };

// NSMALL_FRAG の判定は本番と同じ実装を呼ぶ（ozaki2_moduli.h の oz2_choose_small_tail．oz_sim.h の make_crt_kernel が数える）．
// ここで書き直すと，本番だけを誤って変えてもこの検査が素通りする．
// TWOSUM_PROBE_ABS=1 を立てたときだけ，しきい値を「絶対値 2^-30 で固定」にした誤った規則
// （出力の【比較用】）に置き換える（この検査がその誤りを捕まえられることを確かめるための逆転スイッチ）．
// TWOSUM_PROBE_ALL=1 は全断片を安く足す（OZ2_ALLSMALL の模擬）
static uint32_t count_small(const CrtKernel& k, const std::vector<int>& m, bool all, bool absj) {
    if (all) return k.NRP;
    if (!absj) return k.NSMALL_FRAG;
    const Oz2SmallTail st = oz2_choose_small_tail(k.rp.data(), k.NRP, m.data(), k.S);
    const float abs_eps = (float)std::ldexp(1.0, -30);             // 誤った規則の固定しきい値
    uint32_t n = 0;
    for (size_t t = k.NRP; t-- > 0; ) { if (st.worst[t] <= abs_eps) n++; else break; }
    return n;
}

// 1 パス分の足し込み．k_i を直接与え，recon_crt.cpp と同じ手続き（oz_sim.h の crt_accumulate：断片の積を tfAddS／
// 末尾 NSMALL_FRAG 本は tfAddSmall で足し，法ごとに a0 -= round(a0)）で足す．fast は tfAddSmall の第 2 段を FastTwoSum に
// した【比較用】の版（TWOSUM_PROBE_FAST）
static Tri accumulate(const CrtKernel& k, const std::vector<int>& kv, bool fast) {
    Dst D; D.small_fast = fast;
    for (uint32_t i = 0; i < k.S; i++) {
        D.r[sfpu_emu::dst::keep] = (float)kv[i];
        oz_sim::crt_accumulate(D, k, i);
    }
    return {D.r[0], D.r[1], D.r[2]};
}

static __float128 fold(__float128 x) {            // (-1/2, 1/2] へ畳む
    x -= floorq(x); if (x > (__float128)0.5) x -= 1; return x;
}

// 構成 1 つの frac の絶対誤差の最悪［log2］．非有限の結果（limb の和か真値が inf・NaN）は最悪の比較から黙って落ちる
// （NaN との比較は偽）ので別に数え，1 件でもあれば判定を落とす．評価した件数 n が 0 なら最悪は初期値のままなので，それも落とす
struct Stat {
    double worst = -1e9;
    uint64_t n = 0, nonfinite = 0;
    void add(__float128 sum, __float128 ex) {
        if (!finiteq(sum) || !finiteq(ex)) { nonfinite++; return; }
        n++;
        const double d = log2((double)fabsq(fold(sum) - ex) + 1e-300);
        if (d > worst) worst = d;
    }
};

int main(int argc, char** argv) {
    // 位置引数は 10 進の整数として全体を読み切れたときだけ受け付ける（src/common/oz_env.hpp．strtoul/atoi では "14080x" が
    // 14080 と読まれ，K=0 が法選択の log2(0) に届いて (int)inf になる）．範囲の外は終了コード 2
    const uint32_t K = (uint32_t)oz_env::arg_i64(argc, argv, 1, 14080, 1, UINT32_MAX, "K");
    const unsigned seed = (unsigned)oz_env::arg_i64(argc, argv, 2, 11, 0, UINT32_MAX, "seed");
    const int iters = (int)oz_env::arg_i64(argc, argv, 3, 100000, 1, 1000000000, "試行数");   // 0 だと初期値どうしを比べて通る
    if (argc > 4) { fprintf(stderr, "ERROR: 引数は K seed 試行数 の 3 個まで\n"); return 2; }
    const bool fast = oz_sim::env_flag01("TWOSUM_PROBE_FAST"), absj = oz_sim::env_flag01("TWOSUM_PROBE_ABS"),
               all = oz_sim::env_flag01("TWOSUM_PROBE_ALL");

    // 経路ごとの法の取り方（本番の経路の表 ozaki2_split.h の OZ2_ROUTES と ozaki2_choose_moduli_route）．FP16（OZ2_ROUTE_FP16A）は
    // 段の時間のモデルが BF16 と別で，K によっては別の法の集合を選ぶ（K=3520・14080 は同じ）．法の取り方が BF16 と同じなら
    // BF16 の行が兼ね，違えば FP16 の行も流す（乱数は経路ごとに seed から引き直すので，FP16 の行の有無は他の行の値を変えない）
    int fail = 0;
    printf("===== 分数形 CRT の frac 絶対誤差（K=%u，seed=%u，%d 試行）=====\n", K, seed, iters);
    printf("  第 2 段 %s，NSMALL の判定 %s%s\n",
           fast ? "FastTwoSum【比較用】" : "TwoSum",
           absj ? "絶対値 2^-30 固定【比較用】" : "最大項に対する相対",
           all ? "（全断片を安く＝OZ2_ALLSMALL の模擬）" : "");
    printf("%-6s %-3s %-7s %14s %14s %8s\n", "経路", "s", "構成", "NSMALL=0", "既定の NSMALL", "判定");
    for (const Oz2RouteRow& p : OZ2_ROUTES) {
        Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, p.r);
        if (!mo.ok) { printf("  %s 法が取れない\n", p.name); fail = 1; continue; }
        if (oz_sim::fp16_same_as_bf16(p.r, K, mo)) {
            printf("%-6s %-3d （BF16 と同じ法の集合・逆数の断片なので BF16 の行が兼ねる）\n", p.name, mo.s);
            continue;
        }
        const int S = mo.s;

        // 1 パス（rp0）と 2 パス（rp1 -> rp2）の両方を見る．カーネルの定数は本番と同じ作り（limb の本数・断片）で，
        // NSMALL_FRAG だけを [0]=0（全部 tfAddS）と [1]=既定（または比較用の規則）に替える
        const oz_sim::ReconKernels base = oz_sim::make_recon_kernels(mo, oz2_in_residue(p.r));
        const uint32_t n0 = count_small(base.one, mo.m, all, absj);
        const uint32_t n1 = count_small(base.first, mo.m, all, absj);
        const uint32_t n2 = count_small(base.second, mo.m, all, absj);
        CrtKernel k1p[2] = {base.one, base.one}, k2a[2] = {base.first, base.first}, k2b[2] = {base.second, base.second};
        k1p[0].NSMALL_FRAG = k2a[0].NSMALL_FRAG = k2b[0].NSMALL_FRAG = 0;
        k1p[1].NSMALL_FRAG = n0; k2a[1].NSMALL_FRAG = n1; k2b[1].NSMALL_FRAG = n2;

        std::mt19937 rng(seed);
        Stat s1p[2], s2p[2];                       // [0]=NSMALL_FRAG 0，[1]=既定
        for (int it = 0; it < iters; ++it) {
            std::vector<int> k(S);
            for (int i = 0; i < S; i++) {          // 対称剰余 [-m/2, m/2]
                std::uniform_int_distribution<int> K(-mo.m[i] / 2, mo.m[i] / 2);
                k[i] = K(rng);
            }
            __float128 ex = 0;
            for (int i = 0; i < S; i++) ex += (__float128)k[i] / (__float128)mo.m[i];
            ex = fold(ex);

            for (int mode = 0; mode < 2; mode++) {
                // 1 パス
                Tri a = accumulate(k1p[mode], k, fast);
                s1p[mode].add((__float128)a.a0 + a.a1 + a.a2, ex);
                // 2 パス（第 1 パスの 3 本と第 2 パスの 3 本を出口で足す）
                Tri b = accumulate(k2a[mode], k, fast);
                Tri c = accumulate(k2b[mode], k, fast);
                s2p[mode].add((__float128)b.a0 + b.a1 + b.a2 + c.a0 + c.a1 + c.a2, ex);
            }
        }
        // 判定は 2 つ．NSMALL_FRAG=0 から悪化しないこと（模擬の揺れで 0.5 bit だけ緩める）と，
        // 絶対の上限を割らないこと．相対だけだと両方が一緒に悪化したときに通る．
        //   1 パスは 3 本組の最下位 limb の丸めで 2^-67 前後に頭打ちになる（docs/accuracy.md．tools/reconsim.cpp の法 16 本の
        //   最悪で 2^-67.0（measure_logs/probe_reconsim.log），法 8〜20 本では 2^-65.9〜2^-67.7．この模擬は本番の断片 rp0 と
        //   対称剰余で，既定（K=14080・seed=11・10 万試行）の経路ごとの最悪が 2^-66.6（BFP2）〜2^-67.7（TF32），selftest と
        //   probe_twosum の 2 万試行では seed=11 で 2^-67.1〜2^-68.0，seed=16 で 2^-66.8〜2^-68.0）．下限 2^-64 はこれらの
        //   最悪より 2.6 bit 以上緩い
        //   2 パスは共通粒度で累算が無誤差に閉じるのでもっと良い
        // 2 パスは第 1・第 2 パスの本数を「第 1+第 2」の形で両方出す．floor_bits は上の絶対の上限［log2］．
        // 非有限の結果があれば件数を行に足して FAIL にする（無ければ行は前と同じ）
        auto judge = [&](Stat* st, const char* tag, const std::string& nsmall, double floor_bits) {
            const uint64_t nf = st[0].nonfinite + st[1].nonfinite;
            const bool ok = nf == 0 && st[0].n > 0 && st[1].n > 0 &&
                            st[1].worst <= st[0].worst + 0.5 && st[1].worst <= floor_bits;
            if (!ok) fail = 1;
            printf("%-6s %-3d %-7s   最悪 2^%-6.1f   最悪 2^%-6.1f %8s  (NSMALL=%s，下限 2^%.0f)", p.name, mo.s, tag,
                   st[0].worst, st[1].worst, ok ? "OK" : "FAIL", nsmall.c_str(), floor_bits);
            if (nf) printf("  非有限の結果 %llu 件", (unsigned long long)nf);
            printf("\n");
        };
        judge(s1p, "1 パス", std::to_string(n0), -64.0);
        judge(s2p, "2 パス", std::to_string(n1) + "+" + std::to_string(n2), -95.0);
    }
    printf("\n%s\n", fail ? "FAIL あり（tfAddSmall が精度を落としている）"
                          : "すべて OK（tfAddSmall を入れても frac の最悪誤差は悪化しない）");
    return fail;
}
