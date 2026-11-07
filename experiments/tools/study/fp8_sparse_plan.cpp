// fp8_sparse_plan.cpp -- 尾崎 II の FP8 経路：疎な e4m3 桁で法を選び直したときの見積り（Phase A）
// ---------------------------------------------------------------------------
// リポジトリ直下で（コンテナの中で．フラグは src/Makefile の HOST_OPT．どのスクリプトも作らず回さない）
//     c++ $(make -s -C src print-host-flags) -Isrc tools/study/fp8_sparse_plan.cpp -lquadmath -o /tmp/fp8_sparse_plan
//     /tmp/fp8_sparse_plan [X_SQS] [per_mod_maxg]      既定 X_SQS=12，per_mod_maxg=1
// デバイスは使わない．docs/fp8_sparse_digits.md の Phase A の数値は，この道具を入れた 72333d3（2026-09-25）の版の出力である．
// 【記録用】その数値の出し方を残してある．法の判定と選び方は今の ozaki2_moduli.h を取り込むので，今の版を回しても記録の
// 再現にはならない（再現するなら 72333d3 の木で作って回す．tools/README.md の study の行）．モデルの係数と方式の判定を独自に
// 持つので，新しい見積りには FP8 も扱う tools/awe_plan.cpp（ホストの ozaki2_moduli.h の関数を使う）を使うこと．
//
// 何をするか
//   (1) e4m3 が無誤差に表せる整数 S_d = {x : |x| <= d，有効ビット <= 4}（d=32 なら {0..16} ∪ {18,20,..,32}
//       と負で 49 個）が法 q の完全剰余系を含む q を並べる（奇数 q <= |S_d| と q <= 33）．
//   (2) 法の集合ごとに，行列積（桁積）・GEMM の出力（同じ位の桁積を K 方向に連結した後）・桁行列の本数と，
//       段ごとの時間のモデルによる予測を出す．並べるのは
//         current      比較の基準にする稠密な桁の選び方（GEMMul8 の表．桁は [-16,16]．ozaki2_choose_moduli の smax=0）
//         chooser      ホストの選び方（ozaki2_choose_moduli_route(K, OZ2_ROUTE_FP8)．疎な桁 (E) と法ごとの maxg）
//         dense plan   稠密な桁（|d| <= 16）の方式だけで下の探索が選ぶ集合（疎な桁を使わない改善の上限の目安）
//         sparse d=..  疎な桁 S_d（d = 20..32）の平方数 (E) を加えて下の探索が選ぶ集合
//       探索は「互いに素な平方数の部分集合を総当たり＋非平方数（WIDE16・KAR16）を方式ごとに本数を振って
//       降順・互いに素の貪欲で足す＋末尾 2 本の選び直し」で，費用はモデルの予測時間（法ごとの和）．
//
// 段の時間のモデル［ms］（係数は measure_logs の実測から当てはめたもの．docs/fp8_sparse_digits.md）
//   GEMM    = gp*行列積 + go*出力             （14080：§8 の A/B で出力 42->36 本が 707.0->699.1 ms）
//   合成    = co*出力 + cs*法の本数
//   スライス = s0 + sop*演算数 + spl*桁行列    演算数 = Σ_法 (19 + 桁の作り方の演算数)．
//             19 は剰余（ModRound 3 回ほか），桁の作り方は SQ・WIDE 6，Karatsuba 12，疎な桁 SQS は X_SQS
//             （タイル演算だけで書けば 32，SFPI で 1 回に済ませれば 6 + 数演算相当．既定 12）
//   再構成・スケール = 定数（法 14 本以下では法の本数によらない：BFP8 の 11 本も FP8 の 14 本も 273.6 ms）
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>
#include "ozaki2/ozaki2_moduli.h"
#include "common/oz_env.hpp"   // 位置引数の読み方

static std::vector<int> Sd(int d) { std::vector<int> s; for (int x = -d; x <= d; ++x) if (oz2_e4m3_int(x)) s.push_back(x); return s; }

// 方式：名前・行列積・桁行列・位ごとの積の本数（連結の対象）・桁の作り方の演算数・l と h の桁の上限
struct Mode { const char* name; int nprod, nplane; std::map<int, int> keys; int xops; int dl, dh; };
static int nout_of(const Mode& c, int g) { int n = 0; for (auto& kv : c.keys) n += (kv.second + g - 1) / g; return n; }
static int maxg_of(double K, double pb, int mmax) { return std::max(1, (int)std::floor((16777216.0 - mmax) / (K * pb))); }

struct Model { double gp, go, co, cs, s0, sop, spl, rec, sc; };
// 14080：cool_oz2{fp8,bfp8,bfp4,bfp2}_14080_nongemm.log の段の時間と §8 の A/B から
static const Model M14 = {15.5, 1.3, 3.7, 1.5, 33.2, 0.5725, 0.127, 274.0, 11.4};
// 3520：acc_perf1・acc_c*（連結後）と acc_it4〜6（連結前）の FP8 の段の時間から
static const Model M35 = {0.163, 0.309, 0.223, 0.022, 2.24, 0.03578, 0.0079, 17.05, 1.30};
// 7040：14080 と 3520 の間を取った推定．Phase A の数値を再現するため，Phase B の実測
// （docs/fp8_sparse_digits.md．モデルより 1.8% 速い）では置き換えない
static const Model M70 = {1.91, 0.50, 0.87, 0.20, 8.3, 0.1431, 0.032, 68.0, 2.85};
static const Model& model_for(int K) { return K <= 3520 ? M35 : K <= 7040 ? M70 : M14; }

struct Plan { std::vector<int> m; std::vector<std::string> tag; int s = 0, nprod = 0, nplane = 0, nout = 0, ops = 0; double excess = 0, t[6] = {0}; };
static void eval(Plan& p, const Model& M) {
    const double g = M.gp * p.nprod + M.go * p.nout, c = M.co * p.nout + M.cs * p.s;
    const double sl = M.s0 + M.sop * p.ops + M.spl * p.nplane;
    p.t[0] = M.sc; p.t[1] = sl; p.t[2] = g; p.t[3] = c; p.t[4] = M.rec; p.t[5] = M.sc + sl + g + c + M.rec;
}
static int g_xsqs = 12;
static bool g_pmaxg = true;

static std::vector<Mode> modes_for(int m, int d) {
    std::vector<Mode> out;
    const int q = oz2_is_square(m);
    if (q) {
        const int b = oz2_digit_bound_rnd(m, q);
        if (b <= 16) out.push_back({"*", 3, 2, {{0, 1}, {1, 2}}, 6, b, b});
        else if (d > 16 && (q & 1) && oz2_sparse_covers(q, d)) {
            int dl, dh; oz2_sqs_bounds(q, dl, dh);
            if (dl <= d && dh <= d) out.push_back({"s", 3, 2, {{0, 1}, {1, 2}}, g_xsqs, dl, dh});
        }
    } else {
        if (oz2_digit_bound_kar(m) <= 16) out.push_back({"k", 3, 3, {{0, 1}, {2, 1}, {4, 1}}, 12, 16, 16});
        const int qq = oz2_digit_base_rnd(m), b = oz2_digit_bound_rnd(m, qq);
        if (b <= 16) out.push_back({"w", 4, 2, {{0, 1}, {1, 2}, {2, 1}}, 6, b, b});
    }
    return out;
}
static double modcost(const Mode& c, int g, const Model& M) {
    return M.gp * c.nprod + (M.go + M.co) * nout_of(c, g) + M.cs + M.sop * (19 + c.xops) + M.spl * c.nplane;
}

// d：疎な桁の上限（16 なら稠密な桁だけ）
static Plan plan(int K, int d, double tol = 0.25, int b = 54) {
    const Model& M = model_for(K);
    const double need = 2.0 * b + 1 + std::log2((double)K) + 1e-9;
    const int cap = d > 16 ? (int)(Sd(d).size() * Sd(d).size()) : 1089;
    struct Cand { int m; Mode c; double lg, cost; };
    std::vector<Cand> sq, cl[2];   // cl：0 WIDE16，1 KAR16
    auto gof = [&](const Mode& c) { return g_pmaxg ? maxg_of(K, (double)c.dl * c.dh, cap) : maxg_of(K, (double)d * d, cap); };
    for (int m = cap; m >= 3; --m)
        for (auto& c : modes_for(m, d)) {
            Cand x{m, c, std::log2((double)m), modcost(c, gof(c), M)};
            std::string n(c.name);
            if (n == "*" || n == "s") { if (m >= 100) sq.push_back(x); }
            else cl[n == "w" ? 0 : 1].push_back(x);
        }
    auto cop = [](int m, const std::vector<Cand>& v) { for (auto& x : v) if (std::gcd(m, x.m) != 1) return false; return true; };
    double bestc = 1e30; std::vector<Cand> best, head;
    auto consider = [&](const std::vector<Cand>& H) {
        double hb = 0, hc = 0; for (auto& x : H) { hb += x.lg; hc += x.cost; }
        if (hb - need > tol) return;
        for (int nw = 0; nw <= 4; ++nw) for (int nk = 0; nk <= 14; ++nk) {
            std::vector<Cand> set = H; double bits = hb, cost = hc; bool fail = false;
            const int want[2] = {nw, nk};
            for (int k = 0; k < 2 && !fail; ++k) {
                int got = 0;
                for (auto& x : cl[k]) { if (got == want[k]) break; if (cop(x.m, set)) { set.push_back(x); bits += x.lg; cost += x.cost; ++got; } }
                fail = got < want[k];
            }
            if (fail || bits < need) continue;
            if (cost < bestc - 1e-9) { bestc = cost; best = set; }
            break;
        }
    };
    std::function<void(int)> rec = [&](int i) {
        consider(head);
        for (int j = i; j < (int)sq.size(); ++j) {
            if (!cop(sq[j].m, head)) continue;
            head.push_back(sq[j]); rec(j + 1); head.pop_back();
        }
    };
    rec(0);
    Plan p;
    if (best.empty()) return p;
    // 末尾 2 本の選び直し（費用 → 超過が許容内か → 超過の順）
    std::sort(best.begin(), best.end(), [](const Cand& a, const Cand& b) { return a.cost / a.lg < b.cost / b.lg; });
    if (best.size() >= 2) {
        std::vector<Cand> base(best.begin(), best.end() - 2), all = sq;
        for (int k = 0; k < 2; ++k) all.insert(all.end(), cl[k].begin(), cl[k].end());
        double bb = 0; for (auto& x : base) bb += x.lg;
        double bex = 1e9, bcost = 1e30; int bi = -1, bj = -1;
        for (int i = 0; i < (int)all.size(); ++i) {
            if (!cop(all[i].m, base)) continue;
            for (int j = i + 1; j < (int)all.size(); ++j) {
                if (all[j].m == all[i].m || std::gcd(all[i].m, all[j].m) != 1 || !cop(all[j].m, base)) continue;
                const double ex = bb + all[i].lg + all[j].lg - need; if (ex < 0) continue;
                const double c = all[i].cost + all[j].cost;
                const bool exok = ex <= tol, bok = bex <= tol;
                if ((exok && !bok) || (exok == bok && (c < bcost - 1e-9 || (std::fabs(c - bcost) < 1e-9 && ex < bex)))) {
                    bex = ex; bcost = c; bi = i; bj = j; }
            }
        }
        if (bi >= 0) { base.push_back(all[bi]); base.push_back(all[bj]); best = base; }
    }
    std::sort(best.begin(), best.end(), [](const Cand& a, const Cand& b) { return a.m > b.m; });
    double lm = 0; int mmax = 0; for (auto& x : best) { lm += x.lg; mmax = std::max(mmax, x.m); }
    for (auto& x : best) {
        p.m.push_back(x.m); p.tag.push_back(x.c.name);
        p.nprod += x.c.nprod; p.nplane += x.c.nplane; p.ops += 19 + x.c.xops;
        p.nout += nout_of(x.c, g_pmaxg ? maxg_of(K, (double)x.c.dl * x.c.dh, mmax) : maxg_of(K, (double)d * d, mmax));
    }
    p.s = (int)p.m.size(); p.excess = lm - need;
    eval(p, M);
    return p;
}

// ozaki2_choose_moduli の結果をモデルで評価する
static Plan from_chooser(int K, const Ozaki2Moduli& o) {
    Plan p; p.s = o.s; p.nprod = o.nprod; p.nplane = o.nplane; p.nout = o.nout; p.m = o.m;
    double lm = 0; for (int mm : o.m) lm += std::log2((double)mm);
    p.excess = lm - (2.0 * o.b + 1 + std::log2((double)K));
    for (int i = 0; i < o.s; ++i) {
        const int md = o.mode[i];
        p.tag.push_back(md == OZ2_SPLIT_SQ ? "*" : md == OZ2_SPLIT_SQS ? "s" : md == OZ2_SPLIT_KAR ? "k" : "w");
        p.ops += 19 + (md == OZ2_SPLIT_KAR ? 12 : md == OZ2_SPLIT_SQS ? g_xsqs : 6);
    }
    eval(p, model_for(K));
    return p;
}

static void print(int K, const char* tag, const Plan& p, double base) {
    printf("K=%5d %-14s s=%2d 行列積=%2d 出力=%2d 桁行列=%2d 超過=%.3f 演算=%3d  予測 %7.1f ms（%+5.1f%%．slice %.1f gemm %.1f 合成 %.1f）  {",
           K, tag, p.s, p.nprod, p.nout, p.nplane, p.excess, p.ops, p.t[5], base > 0 ? 100.0 * (p.t[5] / base - 1) : 0.0,
           p.t[1], p.t[2], p.t[3]);
    for (int i = 0; i < p.s; ++i) printf("%d%s%s", p.m[i], p.tag[i].c_str(), i + 1 < p.s ? "," : "}\n");
}

int main(int argc, char** argv) {
    // 引数は [X_SQS [法ごとの maxg 0|1]] の 2 個まで．10 進の整数として全体を読み切れて範囲に入るときだけ受け付ける
    // （src/common/oz_env.hpp．範囲の外は終了コード 2）
    if (argc > 3) { fprintf(stderr, "usage: %s [X_SQS（0--1000）] [法ごとの maxg（0|1）]\n", argv[0]); return 2; }
    g_xsqs = (int)oz_env::arg_i64(argc, argv, 1, g_xsqs, 0, 1000, "X_SQS");
    g_pmaxg = oz_env::arg_i64(argc, argv, 2, g_pmaxg ? 1 : 0, 0, 1, "法ごとの maxg") != 0;
    printf("# (1) S_d が完全剰余系になる法 q（33 より大きいもの）\n");
    for (int d : {16, 20, 24, 28, 32}) {
        printf("  d=%2d |S_d|=%2zu  q =", d, Sd(d).size());
        for (int q = 34; q <= 64; ++q) if (oz2_sparse_covers(q, d)) printf(" %d", q);
        printf("\n");
    }
    printf("# (2) 法の集合と予測（X_SQS=%d，法ごとの maxg=%d．%% は current に対する予測時間の差）\n", g_xsqs, (int)g_pmaxg);
    for (int K : {3520, 7040, 14080}) {
        const Plan cur = from_chooser(K, ozaki2_choose_moduli(K, 14, 1089, OZ2_ACC_BITS, 16, 54, -1.0, 2));
        print(K, "current", cur, cur.t[5]);
        print(K, "chooser", from_chooser(K, ozaki2_choose_moduli_route(K, OZ2_ROUTE_FP8)), cur.t[5]);
        print(K, "dense plan", plan(K, 16), cur.t[5]);
        for (int d : {20, 24, 28, 32}) {
            char tag[32]; snprintf(tag, sizeof tag, "sparse d=%d", d);
            print(K, tag, plan(K, d), cur.t[5]);
        }
        fflush(stdout);
    }
    return 0;
}
