// SPDX-License-Identifier: Apache-2.0
//
// 尾崎 II の 2 パス構成の第 1 パス（共通粒度へ丸めた逆数の断片 rp1）の累算が無誤差かを，本番の構成で確かめる【検査用】
// ---------------------------------------------------------------------------
//     make -C src aligned_accum           （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     ./src/build/aligned_accum     本番の粒度 OZ2_RGRAN で判定し，粒度を振って破綻の始まりを出す（scripts/selftest.sh が回す）
//     ./src/build/aligned_accum g   粒度 2^-g だけで同じ判定をする（g は 1--100）
// 終了コード：判定した粒度（既定は OZ2_RGRAN）で 1 件でも無誤差でなければ 1，引数の誤りは 2．
//
// 何を確かめるか（GEMMul8 の手．ozaki2_moduli.h の rp1 の注記と OZ2_RGRAN の根拠）
//   逆数 1/m_i を共通の絶対粒度 2^-g の整数倍 R_i へ最近接で丸め，R_i を 12 bit の断片 5 本に分ける．k_i が整数なので
//   k_i*断片はどれも 2^-g の整数倍で，3 本組の累算が表せる範囲に収まるかぎり第 1 パスの累算は完全に無誤差になる
//   （丸めの残り 1/m_i - R_i は第 2 パスが拾う）．無誤差が崩れると第 1 パスの丸めが frac にそのまま残る．
// どうやって確かめるか（本番と同じものを使う）
//   法の集合     本番の 8 経路（ozaki2_split.h の OZ2_ROUTES）の K = 3520・7040・14080 の既定（ozaki2_choose_moduli_route）
//   断片         ozaki2_moduli.h の rp1 と同じ作り方を粒度 g で行う（g = OZ2_RGRAN では本番の mo.rp1 とビット単位で同じことを確かめる）
//   k_i          再構成の r_i*y_i の ModRound の出力の範囲 |k_i| <= floor(m/2)+1（ozaki2_moduli.h の oz2_crt_k_bound）．
//                対称剰余の一様乱数に，全部が上界・全部が負の上界・符号を交互に振った上界を混ぜる
//   足し込み     recon_crt.cpp と同じ手続き（oz_sim.h の crt_accumulate：k_i*断片を tfAddS で，末尾 NSMALL_FRAG 本は tfAddSmall で
//                足し，法ごとに a0 -= round(a0)．NSMALL_FRAG は本番の oz2_choose_small_tail．SFPU の演算は checks/sfpu_tf_emu.hpp）
//   厳密な値     Σ k_i * (断片の和 * 2^g) を __int128 の整数で求め，2^g を法に畳んだもの（frac の第 1 パスぶん）．3 本組の和との差を
//                1 を法にして見る（0 なら無誤差）
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <quadmath.h>
#include "oz_sim.h"

typedef __int128 i128;

// ozaki2_moduli.h の oz2_crt_constants と同じ作り方で，1/m を粒度 2^-g へ最近接で丸めた R の 12 bit 断片 5 本を作る
static std::array<float, OZ2_NRP1> rp1_of(int m, int g) {
    const float SPL = 4097.0f;
    auto top12 = [&](__float128 v) {                      // v の上位 12 bit を FP32 で返す
        const float p = (float)(double)v;
        if (p == 0.0f) return 0.0f;
        const float c = (float)((double)SPL * (double)p);
        return (float)((double)c - (double)(float)((double)c - (double)p));
    };
    const __float128 r = (__float128)1 / (__float128)m;
    const __float128 gran = powq((__float128)2, -(__float128)g);
    const __float128 R = roundq(r / gran) * gran;
    std::array<float, OZ2_NRP1> p1{};
    __float128 v = R;
    for (int t = 0; t < OZ2_NRP1; ++t) { p1[t] = top12(v); v -= (__float128)p1[t]; }
    return p1;
}
// 2 つの断片の並びがビット単位で同じか（std::array<float> の == は数値の比較で，±0 を同じとみなす）
template <class A, class B>
static bool same_bits(const A& a, const B& b) {
    if (a.size() != b.size()) return false;
    for (size_t t = 0; t < a.size(); ++t)
        if (f32_bits(a[t]) != f32_bits(b[t])) return false;
    return true;
}

struct Res { long n = 0, bad = 0; double worst = 0; bool frag_ok = true; uint32_t nsmall = 0; double rho = 0; };

// 1 経路・1 つの K・粒度 g の判定
static Res run(const Ozaki2Moduli& mo, const Oz2Route& rt, int g, int trials, uint64_t seed) {
    Res res;
    const int S = mo.s;
    std::vector<std::array<float, OZ2_NRP1>> rp(S);
    std::vector<i128> nfr(S);               // 断片の和 * 2^g（整数）
    for (int i = 0; i < S; i++) {
        rp[i] = rp1_of(mo.m[i], g);
        __float128 sum = 0;
        for (float f : rp[i]) sum += (__float128)f;
        const __float128 sc = ldexpq(sum, g);
        if (sc != floorq(sc)) res.frag_ok = false;      // 断片が 2^-g の整数倍でない（作り方の前提が崩れた）
        nfr[i] = (i128)sc;
        res.rho += (double)fabsq((__float128)1 / (__float128)mo.m[i] - sum) * (double)oz2_crt_k_bound(mo.m[i]);
    }
    const oz_sim::CrtKernel k = oz_sim::make_crt_kernel(mo, rp, oz2_in_residue(rt), oz2_recon_limb3(mo, true));
    res.nsmall = k.NSMALL_FRAG;
    std::mt19937_64 rng(seed);
    std::vector<long> kv(S);
    const i128 mask = ((i128)1 << g) - 1;
    for (int t = 0; t < trials; t++) {
        const int kind = t < 3 ? t : 3;             // 0：全部 +上界，1：全部 -上界，2：交互，3：一様（たまに上界）
        for (int i = 0; i < S; i++) {
            const long b = oz2_crt_k_bound(mo.m[i]);
            if (kind == 0) kv[i] = b;
            else if (kind == 1) kv[i] = -b;
            else if (kind == 2) kv[i] = (i & 1) ? -b : b;
            else if (rng() % 16 == 0) kv[i] = (rng() & 1) ? b : -b;
            else kv[i] = (long)(rng() % (uint64_t)(2 * (mo.m[i] / 2) + 1)) - mo.m[i] / 2;
        }
        oz_sim::Dst D;
        for (int i = 0; i < S; i++) {
            D.r[sfpu_emu::dst::keep] = (float)kv[i];
            oz_sim::crt_accumulate(D, k, (uint32_t)i);
        }
        i128 N = 0;
        for (int i = 0; i < S; i++) N += (i128)kv[i] * nfr[i];
        const i128 f = N & mask;                     // 2^g を法に（0 <= f < 2^g）
        const __float128 ex = ldexpq((__float128)f, -g);
        __float128 d = (__float128)D.r[0] + (__float128)D.r[1] + (__float128)D.r[2] - ex;
        d -= roundq(d);                              // 1 を法にした差
        res.n++;
        if (d != 0) { res.bad++; res.worst = std::max(res.worst, (double)fabsq(d)); }
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc > 2) { std::fprintf(stderr, "使い方: aligned_accum [g]  （g は粒度 2^-g の 1--100．無ければ OZ2_RGRAN と掃引）\n"); return 2; }
    int gsel = 0;
    if (argc == 2) {
        long long gv = 0;   // 読み方は src/common/oz_env.hpp（10 進の整数として全体を読み切る）
        if (!oz_env::to_i64(argv[1], gv) || gv < 1 || gv > 100) {
            std::fprintf(stderr, "ERROR: g=%s は 1--100 の整数ではない\n", argv[1]); return 2; }
        gsel = (int)gv;
    }
    const int gjudge = gsel ? gsel : OZ2_RGRAN, trials = 20000;
    struct Case { const Oz2RouteRow* rt; uint32_t K; Ozaki2Moduli mo; };
    std::vector<Case> cases;
    for (const auto& r : OZ2_ROUTES)
        for (uint32_t K : {3520u, 7040u, 14080u}) {
            Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, r.r);
            if (!mo.ok) { std::printf("%s K=%u 法が取れない -> FAIL\n", r.name, K); return 1; }
            if (oz_sim::fp16_same_as_bf16(r.r, K, mo)) continue;
            cases.push_back({&r, K, mo});
        }
    std::printf("===== 第 1 パスの累算（粒度 2^-%d，本番の法の集合・対称剰余の k_i・本番の足し込み．各 %d 試行）=====\n", gjudge, trials);
    bool ok = true;
    for (const Case& c : cases) {
        const Res r = run(c.mo, c.rt->r, gjudge, trials, 11);
        // g = OZ2_RGRAN の断片は本番が作る rp1 と同じでなければならない（作り方の写しの確認）
        bool same = true;
        if (gjudge == OZ2_RGRAN)
            for (int i = 0; i < c.mo.s; i++) same = same && same_bits(rp1_of(c.mo.m[i], gjudge), c.mo.rp1[i]);
        const bool good = r.bad == 0 && r.frag_ok && same;
        ok = ok && good;
        std::printf("  %-5s K=%-6u s=%-2d NSMALL=%u  無誤差でない %ld/%ld%s  丸めの残り（第 2 パスが拾う）<= 2^%.1f%s%s  %s\n",
                    c.rt->name, c.K, c.mo.s, r.nsmall, r.bad, r.n,
                    r.bad ? (" 最悪 2^" + std::to_string((int)std::floor(std::log2(r.worst)))).c_str() : "",
                    std::log2(r.rho), r.frag_ok ? "" : "  断片が 2^-g の整数倍でない",
                    same ? "" : "  本番の rp1 と違う", good ? "OK" : "FAIL");
    }
    if (!gsel) {
        // 粒度を振って，無誤差が崩れ始める g を経路ごとに出す（OZ2_RGRAN との余裕）
        std::printf("===== 粒度の掃引（K=14080．無誤差でない試行の数／%d）=====\n", trials);
        const int gs[] = {56, 60, 62, 64, 65, 66, 67, 68, 69, 70, 72};
        std::printf("  %-5s", "g");
        for (int g : gs) std::printf(" %6d", g);
        std::printf("\n");
        for (const Case& c : cases) {
            if (c.K != 14080) continue;
            std::printf("  %-5s", c.rt->name);
            for (int g : gs) std::printf(" %6ld", run(c.mo, c.rt->r, g, trials, 11).bad);
            std::printf("\n");
        }
    }
    std::printf("ALIGNED-ACCUM g=%d %s\n", gjudge, ok ? "PASS（第 1 パスの累算は全経路・3 サイズで無誤差）" : "FAIL");
    return ok ? 0 : 1;
}
