// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：法ごとの運び方を混ぜた法の集合を，段の時間のモデルが最小になるように選ぶ（ホスト側）
// ---------------------------------------------------------------------------
// 法ごとの運び方の関数（oz2_split_mode など）と経路の番号（OZ2_AWE_*）は ozaki2_split.h から取り込む．ozaki2_moduli.h の
// 法の選択（ozaki2_choose_moduli_w）がここの oz2_awe_choose を呼ぶ．
//
// 着想は Hayashi–Mukunoki–Hoshino–Katagiri, "AWE: Adaptive Weight Encoding for Exact Integer Matrix Products
// with Fewer GEMMs on FP4 Tensor Cores"（arXiv 2609.24519）：法ごとに行列積の本数が最小になる運び方を選び，
// 法の集合は CRT の窓のもとで整数計画で選ぶ．本実装では法ごとの運び方は oz2_split_mode が m から決める
// （BFP2_b は平衡三進 n 桁の n を法ごとに最小，BFP4_b は平方数・基数 8 の Karatsuba・平衡 2 桁・平衡 3 桁，
// 桁 32 の BF16・FP16・BFP8_b は平方数・基数 16 の Karatsuba・平衡 2 桁に 1 桁の法（m <= 65．Oz2Ndig::one）を混ぜる）．
// ここは法の集合だけを選ぶ．目的は行列積の本数ではなく，段の時間のモデル（tools/awe_plan.cpp の Phase A の
// モデルを段の時間の実測で較正したもの．較正に使った実測点は下の OZ2_AWE_COEF の注記．docs/awe_moduli.md の §2）の和である：
// 法を増やすと行列積は減るが，
// スライス・合成・再構成の法ごとの費用が増えるので，本数の最小と時間の最小は一致しない．
//
//   法 1 本の費用 = gp*行列積 + (go+co)*出力 + cs + sop*(19 + 桁の作り方の演算数) + spl*桁行列
//   1 桁の法      = gp*1 + go*1 + sop*19 + spl*1 + OZ2_AWE_ONE_RECON*rb（合成段を通らず，再構成の ModRound 1 回が乗る）
//   全体          = Σ 法の費用 + max(R0, ra + rb*S)（再構成．法 14 本までは平ら）
// 係数はサイズ（M=N=K を仮定）で違うので，3520・7040・14080 の値を log K で補間する（外側は端の 2 点で外挿）．
// 係数の表 OZ2_AWE_COEF は tools/awe_plan.cpp が較正して書き出したもの（ozaki2_awe_coef.inc．手で直さない）．
// Phase A のツール（tools/awe_plan.cpp）との違いが 2 つある：1 桁の法の再構成の上乗せを，ツールは折れ目の式
// max(R0, ra + rb*(S + OZ2_AWE_ONE_RECON*1 桁の法の本数)) に入れる（平らな部分では 0）が，ここは法ごとに線形に足す．
// 候補の法は，ツールが組ごとに上位 400 本に絞るのに対し，ここは全部を使う．論文の 3 サイズで選ぶ集合は同じだが，
// 両者が同じモデルの最適を返すとは限らない．
//
// 探索は「費用が同じ法の組（行列積・出力・桁行列・演算数が同じ）」ごとの本数のベクトルを費用の安い順に並べ，
// 実際に互いに素な法を窓 [need, need+tol] に収めて選べるかを割り当ての探索で確かめ，最初に見つかったものを返す．
// 【これは打ち切り付きのヒューリスティックで，モデルの上での最適は保証しない】．理由は 4 つある：
//   ・組ごとの本数の上限と bit の見積り est は「大きい順に互いに素なものを貪欲に取った和」で，到達できる bit の
//     上界ではない（1 桁の候補 2〜65 を大きい順に取ると 16 本だが，{64,49,27,25,13,11,17,...,61} は互いに素な
//     18 本である）．その本数より多い構成は列挙の前に除かれる．
//   ・列挙するベクトルの数に上限 OZ2_AWE_VEC_MAX（20 万）がある．
//   ・割り当ての探索に，ベクトルあたりの節点の上限 OZ2_AWE_NODE_MAX（40 万）と，単純な貪欲解で上界が取れないときの探索に
//     限る合計の上限 OZ2_AWE_NEW_BUDGET（1000 万）がある．合計の上限はベクトルの単位で見る（ベクトルを始める前にだけ見るので，
//     予算の直前に始めた 1 本ぶん，OZ2_AWE_NODE_MAX までは越えて探す．oz2_awe_search）．
//   ・費用の上界は許容超過を満たすとは限らない貪欲解から作る（OZ2_AWE_UB_SLACK 倍）．
// 打ち切りに当たったか・DRAM の条件を外して選び直したか・表の選び方へ戻ったかは Oz2AweInfo に返し，ozaki2_moduli.h が
// Ozaki2Moduli の awe_status・awe_caps に要約して，ドライバ（ozaki2_main.cpp の print_config）が CONFIG 行の awe・awe_caps 欄に
// 残す．組の中では大きい法ほど bit あたりが安いので，組の本数の見積りは大きい順に互いに素なものを取った和で行う．割り当ての探索は超過の最小を探す（結果は決定的）．
// ホストでの所要（計時の外）は BFP4_b・BFP2_b で 1 回数 ms，桁 32 の経路は組と割り当ての試行が多く，既定の b=54 の
// 14080 で 1 回 1 秒弱，b を 80 まで上げると数秒になる．
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>   // std::greater
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>
#include "ozaki2/ozaki2_split.h"

struct Oz2AweCoef { int K; double gp, go, co, cs, sop, spl, R0, ra, rb; };
// 係数の表（行は awe の番号（ozaki2_split.h の OZ2_AWE_*．1 始まり）の順，各行は K = 3520・7040・14080）．
// tools/awe_plan.cpp の emit-coef が，Phase A のモデルを段の時間の実測（awe_plan.cpp の MEAS．冷却条件の中央値）に合わせた係数を
// %.5g で書き出したもので，手で直さない．実測点は経路で違う：BF16 は 3520・7040・14080 の 3 サイズの実測，BFP8_b・FP16・BFP4_b・
// BFP2_b は 3520 と 14080 の実測で，7040 は両者の段ごとの倍率の相乗平均（awe_plan.cpp の factors_at）．較正の基準か実測を
// 直したら書き出し直す（scripts/selftest.sh の hostcheck が awe_plan verify-coef で，表と道具の出力が同じことを毎回確かめる）
#include "ozaki2/ozaki2_awe_coef.inc"
// 表は第 1 次元を OZ2_AWE_COUNT で宣言するので，行の数を std::size で比べても常に一致し，行が足りなければ 0 で埋まって
// コンパイルは通る（oz2_awe_coef の補間が log2(0) を通る）．そこで行ごとに，K が 3520・7040・14080 の順で係数が 0 でないことを
// 確かめる（行が多すぎれば初期化子の数でコンパイルが止まる）
constexpr bool oz2_awe_coef_rows_set() {
    constexpr int Ks[3] = {3520, 7040, 14080};
    for (int a = 0; a < OZ2_AWE_COUNT; ++a)
        for (int i = 0; i < 3; ++i) {
            const Oz2AweCoef& c = OZ2_AWE_COEF[a][i];
            if (c.K != Ks[i]) return false;
            for (double v : {c.gp, c.go, c.co, c.cs, c.sop, c.spl, c.R0, c.ra, c.rb})
                if (v == 0.0) return false;
        }
    return true;
}
static_assert(oz2_awe_coef_rows_set(), "OZ2_AWE_COEF に行の欠けた経路がある（K が 3520・7040・14080 の順でないか，係数が 0）");

static inline Oz2AweCoef oz2_awe_coef(int awe, uint32_t K) {
    if (awe < 1 || awe > OZ2_AWE_COUNT) std::abort();   // 表に行の無い経路（呼び出し側の誤り）
    const Oz2AweCoef* t = OZ2_AWE_COEF[awe - 1];
    const double x = std::log2((double)K);
    int i = 0;
    if (x > std::log2((double)t[1].K)) i = 1;
    const double x0 = std::log2((double)t[i].K), x1 = std::log2((double)t[i + 1].K), f = (x - x0) / (x1 - x0);
    auto ip = [&](double a, double b) {       // log 空間で線形（べき乗則）．0 以下は線形補間
        if (a > 0 && b > 0) return std::exp(std::log(a) + f * (std::log(b) - std::log(a)));
        return a + f * (b - a);
    };
    Oz2AweCoef c;
    c.K = (int)K;
    c.gp = ip(t[i].gp, t[i + 1].gp); c.go = ip(t[i].go, t[i + 1].go); c.co = ip(t[i].co, t[i + 1].co);
    c.cs = ip(t[i].cs, t[i + 1].cs); c.sop = ip(t[i].sop, t[i + 1].sop); c.spl = ip(t[i].spl, t[i + 1].spl);
    c.R0 = ip(t[i].R0, t[i + 1].R0); c.ra = ip(t[i].ra, t[i + 1].ra); c.rb = ip(t[i].rb, t[i + 1].rb);
    return c;
}
static inline double oz2_awe_recon(const Oz2AweCoef& c, int S) { return std::max(c.R0, c.ra + c.rb * S); }

// 桁の作り方の演算数（スライサのタイル演算の数．剰余までの 19 は別に足す）
static inline int oz2_awe_ops(int m, int dmax, Oz2Ndig ndig, int smax) {
    const int mode = oz2_split_mode(m, dmax, ndig, smax);
    if (mode == OZ2_SPLIT_BAL) return 7 * (oz2_bal_digits(m, dmax, ndig) - 1);
    if (mode == OZ2_SPLIT_ONE) return 0;
    if (mode == OZ2_SPLIT_KAR) return 12;
    if (mode == OZ2_SPLIT_SQS) return 13;
    return 6;
}

// 1 桁の法（合成段を通らない）の再構成の上乗せ：再構成の CRT に ModRound が 1 回乗る（法 1 本あたりの約 81 演算の約 6%）．
// rb（再構成の法 1 本あたりの傾き）に掛ける．tools/awe_plan.cpp も同じ値を使う
inline constexpr double OZ2_AWE_ONE_RECON = 0.062;
// DRAM の見積り（ozaki2_split.h の oz2_dram_one_group）の空き［B］．モデルだけの仮定で，14080^3 の実測の空き 31.0 GB を使う
// （実行時は plan_bands が実測の空きと A・B を残す同時確保で帯の数を決める）
inline constexpr double OZ2_AWE_DRAM_FREE = 31.0e9;
// スライス出力（桁行列）のタイル 1 枚のバイト数（awe の番号順）．経路の表 OZ2_ROUTES でその番号を持つ形式の
// oz_format_host.h の tile_bytes（tt-metal の tile_size と同じことは oz_format_table.hpp が確かめる）
constexpr uint32_t oz2_awe_slice_tile(int awe) {
    for (const Oz2RouteRow& x : OZ2_ROUTES)
        if (x.r.awe == awe) return oz_fmt_host(x.fmt).tile_bytes;
    return 0;
}
inline constexpr uint32_t OZ2_AWE_SLICE_TILE[] = {oz2_awe_slice_tile(OZ2_AWE_BFP4), oz2_awe_slice_tile(OZ2_AWE_BFP2),
                                                  oz2_awe_slice_tile(OZ2_AWE_BF16), oz2_awe_slice_tile(OZ2_AWE_FP16),
                                                  oz2_awe_slice_tile(OZ2_AWE_BFP8)};
static_assert(std::size(OZ2_AWE_SLICE_TILE) == OZ2_AWE_COUNT, "OZ2_AWE_SLICE_TILE の長さが OZ2_AWE_* の数と合わない");
constexpr bool oz2_awe_slice_tiles_set() {
    for (int a = 1; a <= OZ2_AWE_COUNT; ++a)
        if (OZ2_AWE_SLICE_TILE[a - 1] == 0 || OZ2_AWE_SLICE_TILE[a - 1] != oz2_awe_slice_tile(a)) return false;
    return true;
}
static_assert(oz2_awe_slice_tiles_set(), "OZ2_AWE_SLICE_TILE の並びが OZ2_AWE_* の番号順でないか，番号を持つ経路が無い");
// 探索の打ち切り（冒頭の「理由は 4 つある」）：列挙するベクトルの数，ベクトル 1 本の割り当ての節点，単純な貪欲解で上界が
// 取れないときの割り当ての節点の合計．組の本数の枝を窓の OZ2_AWE_OVER_NEED_BITS bit 上で止め，費用の上界は貪欲解の
// OZ2_AWE_UB_SLACK 倍に取る
inline constexpr size_t OZ2_AWE_VEC_MAX = 200000;
inline constexpr long OZ2_AWE_NODE_MAX = 400000;
inline constexpr long OZ2_AWE_NEW_BUDGET = 10000000;
inline constexpr double OZ2_AWE_OVER_NEED_BITS = 16.0;
inline constexpr double OZ2_AWE_UB_SLACK = 1.05;

// 探索の経過（CONFIG 行の awe・awe_caps 欄）．vec_cap：列挙が OZ2_AWE_VEC_MAX に達した，node_cap：あるベクトルの割り当てが
// 節点の上限 OZ2_AWE_NODE_MAX に達した，budget：合計の上限 OZ2_AWE_NEW_BUDGET に達した，dram_retry：1 組で DRAM に収まる
// 集合が無く DRAM の条件を外して選び直した，found：集合を返した（false なら呼び出し側は表の選び方へ戻る）．
struct Oz2AweInfo { bool found = false, vec_cap = false, node_cap = false, budget = false, dram_retry = false; };

// --- 探索の部品（oz2_awe_choose が下の順に呼ぶ）---------------------------------------------------------
// 候補の法（m と log2 m）と，費用が同じ法の組（行列積・出力・桁行列・演算数・1 桁の法か が同じ）．est[t] は組から大きい順に
// 互いに素なものを t 本取った bit の和（本数の見積り．冒頭の「理由は 4 つある」の 1 つめ）
struct Oz2AweCand { int m; double bits; };
struct Oz2AweClass {
    int nprod, nout, npl, ops;
    bool one;                      // 1 桁の法の組（合成段を通らないので，DRAM の見積りの合成段の平面に数えない）
    double cost;                   // 法 1 本の費用（冒頭の式）
    std::vector<Oz2AweCand> v;     // 組の法（m の降順）
    std::vector<double> est;
};

// 組を作る：使える法を組に分け，組ごとに本数の見積り est を作り，bit あたりの費用の安い順に並べる（同点は費用の安い順．
// std::sort なので，それでも同点の組どうしの順は実装の並べ方で決まる．入力の並びは m の降順で決定的）
static inline std::vector<Oz2AweClass> oz2_awe_classes(uint32_t K, const Oz2AweCoef& C, int mod_cap, int digit_max,
                                                       Oz2Ndig ndig, int smax) {
    std::vector<Oz2AweClass> ks;
    for (int m = mod_cap; m >= 2; --m) {
        if (!oz2_usable(m, digit_max, ndig, smax)) continue;
        // 連結の上限：1 桁の法を混ぜる経路（Oz2Ndig::one．桁 32 の AWE 経路）は法ごと，他の経路は一律（ozaki2_split.h の
        // oz2_maxg_for．法の上限 mod_cap で見積もるので，実際はこれ以上になる）
        const int mg = oz2_maxg_for(m, K, mod_cap, digit_max, ndig, smax);
        const int pl = oz2_digit_planes(m, digit_max, ndig);
        const int np = oz2_digit_prods(m, digit_max, ndig), no = (int)oz2_nout(m, digit_max, ndig, mg),
                  op = oz2_awe_ops(m, digit_max, ndig, smax);
        const bool one = oz2_split_mode(m, digit_max, ndig, smax) == OZ2_SPLIT_ONE;
        // 桁積 1 本（連結 1 本）の上界が FP32 の無誤差整数に載らない法は使わない．合成は先に法へ畳む方式
        // （fold_first）で総和 Σ|c_j|(m/2+2) < 2^22 が要る（ozaki2_moduli.h の oz2_plan_concat の検査と同じ式）
        {
            Oz2ProductGroup gr[OZ2_MAXPROD];
            const uint32_t ng = oz2_product_groups(m, gr, digit_max, ndig, mg);
            double csum = 0, gmax = 0;
            for (uint32_t j = 0; j < ng; ++j) {
                csum += std::fabs((double)gr[j].coeff_lo);
                gmax = std::max(gmax, oz2_group_bound(m, gr[j], (double)K, digit_max, ndig, smax));
            }
            if (!ng || gmax + mod_cap > std::ldexp(1.0, 24) || csum * (m / 2.0 + 2.0) >= std::ldexp(1.0, 22)) continue;
        }
        Oz2AweClass* k = nullptr;
        for (auto& x : ks) if (x.nprod == np && x.nout == no && x.npl == pl && x.ops == op && x.one == one) { k = &x; break; }
        if (!k) {
            // 1 桁の法は合成段を通らない（GEMM の出力を再構成が直接法へ畳む）ので合成の費用が無く，代わりに再構成の
            // CRT の演算に ModRound 1 回（法 1 本あたりの約 6%）が乗る
            const double cost = one ? C.gp * np + C.go * no + C.sop * (19 + op) + C.spl * pl + OZ2_AWE_ONE_RECON * C.rb
                                    : C.gp * np + (C.go + C.co) * no + C.cs + C.sop * (19 + op) + C.spl * pl;
            ks.push_back({np, no, pl, op, one, cost, {}, {}});
            k = &ks.back();
        }
        k->v.push_back({m, std::log2((double)m)});      // m の降順に入る
    }
    if (ks.empty()) return ks;
    for (auto& k : ks) {                                 // 大きい順に互いに素なものを取った和（本数の見積り）
        std::vector<int> got; k.est.assign(1, 0.0);
        for (auto& c : k.v) {
            bool ok = true; for (int g : got) if (std::gcd(g, c.m) != 1) { ok = false; break; }
            if (ok) { got.push_back(c.m); k.est.push_back(k.est.back() + c.bits); }
        }
    }
    std::sort(ks.begin(), ks.end(), [](const Oz2AweClass& a, const Oz2AweClass& b) {
        const double ra = a.cost / a.v[0].bits, rb = b.cost / b.v[0].bits;
        return ra != rb ? ra < rb : a.cost < b.cost; });
    return ks;
}

// 探索の状態（探索のあいだ変わらない量と，打ち切りの経過）．need は CRT の窓（log2 M の下限），tol は超過の上限，
// max_moduli は法の本数の上限，minrate[c] は組 c 以降の bit あたりの費用の最小，P・out・F は DRAM の見積りの量（dram_ok）．
// 経過 inf は列挙・割り当て・打ち切りが書く
struct Oz2AweSearch {
    const std::vector<Oz2AweClass>& ks;
    const Oz2AweCoef& C;
    double need, tol;
    int max_moduli;
    std::vector<double> minrate;
    double P, out, F;
    Oz2AweInfo& inf;
    // DRAM：スライス出力（桁行列）と GEMM の出力・剰余の平面が 1 組で収まる本数か（A・B をスライスの後で手放す前提の
    // 関数 oz2_dram_one_group と，合成段の平面の枚数の規則 oz2_compose_plan．空きだけはモデルの仮定 OZ2_AWE_DRAM_FREE．
    // 実行時の帯の数はこれとは別に ozaki2_plan.hpp の plan_bands が決める）．mxo は合成する法（1 桁の法を除く）の出力の最大で，0 なら合成する法が無い
    bool dram_ok(int S, int npl, int mxo) const {
        const Oz2ComposePlan cp = oz2_compose_plan(mxo > 0, (uint32_t)mxo);
        return oz2_dram_one_group(npl, P, cp.planes, S, out, F) <= OZ2_AWE_DRAM_FREE;
    }
};
// 組ごとの本数のベクトルと，その費用（法の費用の和と再構成）
struct Oz2AweVec { double T; std::vector<int> n; };

// 列挙：組ごとの本数のベクトルのうち，費用が上界 incumbent 未満で窓 need に届くもの（dram なら DRAM の条件も見る）を集める．
// DRAM の条件で落としたベクトルがあれば dram_cut を立てる．集める数は OZ2_AWE_VEC_MAX まで（当たれば inf.vec_cap）
struct Oz2AweEnum {
    const Oz2AweSearch& s;
    double incumbent;
    bool dram;
    bool& dram_cut;
    std::vector<Oz2AweVec>& vecs;
    std::vector<int> n;            // 今の枝の組ごとの本数
    // 組 c から先の本数を決める（S は決めた本数の和，cost は費用の和，est は bit の見積りの和，npl は桁行列の枚数の和，
    // mxo は合成する法の出力の最大）
    void rec(int c, int S, double cost, double est, int npl, int mxo) {
        const int NC = (int)s.ks.size();
        if (vecs.size() >= OZ2_AWE_VEC_MAX) { s.inf.vec_cap = true; return; }
        // ベクトルは葉（全部の組の本数を決めたところ）でだけ集める．途中で集めると，残りの組を 0 本にした同じベクトルを
        // t=0 の枝を下りて葉でもう一度集め，同じベクトルが重複して OZ2_AWE_VEC_MAX と割り当ての予算を食う（t=0 の枝の打ち切りは
        // 集めた条件の裏返しなので，重複は必ず起きる）．重複は連続して積まれたので，葉でだけ集めても異なるベクトルの集合と
        // 初めて現れる順は同じで，下の stable_sort の結果も同じになる
        if (c == NC) {
            if (est >= s.need && S > 0) {
                const double T = cost + oz2_awe_recon(s.C, S);
                if (T < incumbent) {
                    if (!dram || s.dram_ok(S, npl, mxo)) vecs.push_back({T, n});
                    else dram_cut = true;
                }
            }
            return;
        }
        const double rem = std::max(0.0, s.need - est);
        if (cost + rem * s.minrate[c] + oz2_awe_recon(s.C, S) >= incumbent) return;
        const auto& k = s.ks[c];
        const int maxn = std::min<int>((int)k.est.size() - 1, s.max_moduli - S);
        for (int t = 0; t <= maxn; ++t) {
            n[c] = t;
            const double e2 = est + k.est[t];
            if (cost + t * k.cost + oz2_awe_recon(s.C, S + t) >= incumbent) break;
            if (t > 0 && cost + t * k.cost + std::max(0.0, s.need - e2) * s.minrate[c + 1] + oz2_awe_recon(s.C, S + t) >= incumbent)
                continue;                                // 下界は t について単調でない
            rec(c + 1, S + t, cost + t * k.cost, e2, npl + t * k.npl, (t && !k.one) ? std::max(mxo, k.nout) : mxo);
            if (e2 >= s.need + OZ2_AWE_OVER_NEED_BITS) break;
        }
        n[c] = 0;
    }
};

// 割り当て：ベクトル 1 本の本数どおりに，組の候補から互いに素な法を選んで窓 [need, need+tol] に収める（超過の最小を探す）．
// 組は ord の順（候補の少ない組から）に埋め，節点は OZ2_AWE_NODE_MAX まで（超過がほぼ 0 の解が見つかっても止める）
struct Oz2AweAssign {
    const Oz2AweSearch& s;
    const std::vector<int>& ord;
    std::vector<int> cnt;                  // ord の順の本数
    std::vector<double> ubrest, lbrest;    // ord の i 番目から後ろの組で取れる bit の和の上限・下限
    std::vector<int> pick, best;
    double bestex = 1e300;
    long nodes = 0;
    void dfs(int i, int start, int left, double bits) {
        const int NC = (int)s.ks.size();
        if (++nodes > OZ2_AWE_NODE_MAX || bestex <= 1e-4) return;
        if (i == NC) {
            const double ex = bits - s.need;
            if (ex >= 0 && ex <= s.tol && ex < bestex) { bestex = ex; best = pick; }
            return;
        }
        if (left == 0) { dfs(i + 1, 0, i + 1 < NC ? cnt[i + 1] : 0, bits); return; }
        const auto& v = s.ks[ord[i]].v;
        for (int j = start; j + left <= (int)v.size(); ++j) {
            double ubc = 0; for (int t = 0; t < left; ++t) ubc += v[j + t].bits;
            if (bits + ubc + ubrest[i + 1] < s.need) return;
            if (bits + v[j].bits + lbrest[i + 1] > s.need + s.tol && left == 1) continue;
            bool ok = true; for (int x : pick) if (std::gcd(x, v[j].m) != 1) { ok = false; break; }
            if (!ok) continue;
            pick.push_back(v[j].m);
            dfs(i, j + 1, left - 1, bits + v[j].bits);
            pick.pop_back();
            if (nodes > OZ2_AWE_NODE_MAX || bestex <= 1e-4) return;
        }
    }
};
// ベクトル vec の割り当て（見つからなければ空）．使った節点の数を nodes に返す
static inline std::vector<int> oz2_awe_assign(const Oz2AweSearch& s, const std::vector<int>& ord, const Oz2AweVec& vec,
                                              long& nodes) {
    const int NC = (int)s.ks.size();
    Oz2AweAssign a{s, ord, std::vector<int>(NC), std::vector<double>(NC + 1, 0.0), std::vector<double>(NC + 1, 0.0), {}, {}};
    for (int i = 0; i < NC; ++i) a.cnt[i] = vec.n[ord[i]];
    for (int i = NC - 1; i >= 0; --i) {
        const auto& v = s.ks[ord[i]].v; const int t = a.cnt[i];
        double ubs = 0, lbs = 0;
        for (int j = 0; j < t && j < (int)v.size(); ++j) { ubs += v[j].bits; lbs += v[v.size() - 1 - j].bits; }
        a.ubrest[i] = a.ubrest[i + 1] + ubs; a.lbrest[i] = a.lbrest[i + 1] + lbs;
    }
    a.dfs(0, 0, NC ? a.cnt[0] : 0, 0.0);
    nodes = a.nodes;
    return a.best;
}

// 本数のベクトルを列挙し（上界 ub 未満の費用のものだけ．dram なら DRAM の条件も見る），費用の安い順に
// 実際の法を割り当てて，窓に収まる互いに素な集合を返す（無ければ空）．DRAM の条件で落としたベクトルがあれば
// dram_cut を立てる．budget > 0 なら割り当ての節点をベクトルをまたいで合計 budget で打ち切る（空を返す）．
// 予算はベクトルを始める前にだけ見るので，予算の直前に始めたベクトルの分（1 本につき OZ2_AWE_NODE_MAX まで）だけ
// 予算を越えて探し，その中で解が見つかれば inf.budget は立たない（結果は決定的）
static inline std::vector<int> oz2_awe_search(const Oz2AweSearch& s, double ub, bool dram, bool& dram_cut, long budget) {
    const int NC = (int)s.ks.size();
    std::vector<Oz2AweVec> vecs;
    Oz2AweEnum e{s, ub, dram, dram_cut, vecs, std::vector<int>(NC, 0)};
    e.rec(0, 0, 0.0, 0.0, 0, 0);
    std::stable_sort(vecs.begin(), vecs.end(), [](const Oz2AweVec& a, const Oz2AweVec& b) { return a.T < b.T; });
    // 割り当て：候補の少ない組から
    std::vector<int> ord(NC); for (int c = 0; c < NC; ++c) ord[c] = c;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return s.ks[a].v.size() < s.ks[b].v.size(); });
    long spent = 0;
    for (const auto& vec : vecs) {
        if (budget > 0 && spent > budget) { s.inf.budget = true; break; }
        long nodes = 0;
        std::vector<int> best = oz2_awe_assign(s, ord, vec, nodes);
        spent += nodes;
        if (nodes > OZ2_AWE_NODE_MAX) s.inf.node_cap = true;
        if (!best.empty()) { std::sort(best.begin(), best.end(), std::greater<int>()); return best; }
    }
    return {};
}

// 費用の上界：bit あたりの費用の安い順に互いに素な法を窓に届くまで取った貪欲解の OZ2_AWE_UB_SLACK 倍．
// ただし本数の上限 max_moduli を見越して，取ると残りの枠を最大の法（log2 mod_cap bit）で埋めても窓に届かなく
// なる法は飛ばす．見越しが無いと，貪欲が bit あたりの安い小さい法で枠を使い切って窓に届かず，大きい法を使えば
// 届く場合（BFP2_b の b を 58 以上，桁 32 の経路の b を 66〜70 以上に上げたときなど）にも上界が作れない．
// 飛ばす法が無ければ（飛ばさずに窓に届けば），見越しの無い単純な貪欲解と同じになる（as_before を立てる）．
// それでも届かなければ上界なし（1e300）で探す（列挙の量は OZ2_AWE_VEC_MAX が抑える）．
static inline double oz2_awe_upper_bound(const Oz2AweSearch& s, int mod_cap, bool& as_before) {
    double ub = 1e300;
    std::vector<std::pair<double, std::pair<int, double>>> all;   // (費用/bit, (m, 費用))
    for (auto& k : s.ks) for (auto& c : k.v) all.push_back({k.cost / c.bits, {c.m, k.cost}});
    std::stable_sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const double maxbits = std::log2((double)mod_cap);
    std::vector<int> got; double bits = 0, cost = 0; bool skipped = false;
    for (auto& a : all) {
        if (bits >= s.need || (int)got.size() >= s.max_moduli) break;
        bool ok = true; for (int g : got) if (std::gcd(g, a.second.first) != 1) { ok = false; break; }
        if (!ok) continue;
        const double lb = std::log2((double)a.second.first);
        if (bits + lb + (s.max_moduli - (int)got.size() - 1) * maxbits < s.need) { skipped = true; continue; }
        got.push_back(a.second.first); bits += lb; cost += a.second.second;
    }
    if (bits >= s.need) ub = (cost + oz2_awe_recon(s.C, (int)got.size())) * OZ2_AWE_UB_SLACK + 1e-9;
    as_before = bits >= s.need && !skipped;
    return ub;
}

// 返すのは法の集合（大きい順）．窓に収まる集合が見つからなければ空（呼び出し側は表の選び方へ戻る）．
//   need   CRT の窓（log2 M の下限），tol 超過の上限，max_moduli 法の本数の上限
//   M, N   出力の大きさ（DRAM の見積り．0 なら M=N=K）
//   info   探索の経過（nullptr なら返さない）
// 手順：組を作る（oz2_awe_classes）→ 費用の上界（oz2_awe_upper_bound）→ 列挙と割り当て（oz2_awe_search）→ 1 組で DRAM に
// 収まる集合が無ければ DRAM の条件を外して選び直す
static inline std::vector<int> oz2_awe_choose(uint32_t K, int awe, int mod_cap, int digit_max, Oz2Ndig ndig, int smax,
                                              double need, double tol, int max_moduli,
                                              uint32_t M = 0, uint32_t N = 0, Oz2AweInfo* info = nullptr) {
    Oz2AweInfo inf;
    struct Put { Oz2AweInfo* d; Oz2AweInfo& s; ~Put() { if (d) *d = s; } } put{info, inf};
    const Oz2AweCoef C = oz2_awe_coef(awe, K);
    const std::vector<Oz2AweClass> ks = oz2_awe_classes(K, C, mod_cap, digit_max, ndig, smax);
    if (ks.empty()) return {};
    const int NC = (int)ks.size();
    // DRAM：スライス出力（桁行列）と GEMM の出力・剰余の平面が 1 組で収まる本数に限る（Oz2AweSearch::dram_ok）．収まらないと
    // 法を組に分けて切り直すことになり，スライス段が組の数だけ伸びる（14080^3 の BFP2_b は 20 本で 30.4 GB，22 本だと 32.2 GB）．
    // 出力の本数は法の上限 mod_cap で見積もった連結の上限から数えるので，実際の集合の値以上である
    const double Md = M ? M : K, Nd = N ? N : K;
    Oz2AweSearch s{ks, C, need, tol, max_moduli, std::vector<double>(NC + 1, 1e300),
                   ((double)Md * K + (double)K * Nd) / 1024.0 * OZ2_AWE_SLICE_TILE[awe - 1], Md * Nd * 4.0,
                   ((double)Md * K + (double)K * Nd) * 8.0, inf};
    for (int c = NC - 1; c >= 0; --c) s.minrate[c] = std::min(s.minrate[c + 1], ks[c].cost / ks[c].v[0].bits);
    bool as_before = false;   // 見越しを使わずに窓に届いた（単純な貪欲解の上界で探し，探索を打ち切らない）
    const double ub = oz2_awe_upper_bound(s, mod_cap, as_before);
    // 単純な貪欲解で上界が取れないときの探索（見越しで上界を作った・上界なし・下の DRAM の条件を外した選び直し）は，
    // 割り当ての節点を合計 OZ2_AWE_NEW_BUDGET（1 秒強）で打ち切り，見つからなければ表の選び方へ戻る．
    // 桁 32 の経路は割り当ての試行が多く，打ち切らなければ数十秒かかる（打ち切って表へ戻ることが多い）．BFP2_b・BFP4_b は
    // 数万節点で見つかる．単純な貪欲解の上界で探す場合（as_before）は打ち切らない（既定の b=54 の 16352 で約 3000 万節点）．
    bool dram_cut = false;
    std::vector<int> best = oz2_awe_search(s, ub, true, dram_cut, as_before ? 0 : OZ2_AWE_NEW_BUDGET);
    // 1 組で DRAM に収まる集合が無ければ（窓に届く本数のベクトルが DRAM の条件で落ちた），DRAM の条件を外して選び直す．
    // 実行時は plan_bands が出力を行の帯に分ける．表の選び方へ戻っても表の集合も DRAM に収まらず，帯分けは同じで
    // 積の本数が多い集合になるだけなので，モデルで選び直す（当たるのは，ホストの選択（b=54・窓は最悪の上界・本数 24）を
    // 32 刻みで模擬すると，M=N=K なら BFP2_b は K >= 15136，BF16・FP16 は K >= 15456，BFP8_b は K >= 18560．K <= 16384 では
    // どれも選び直しで集合が取れる．K > 16384 の BF16・FP16・BFP8_b は選び直しが予算で打ち切られて表の選び方へ戻る K が多い
    // （K の区間は docs/awe_moduli.md の「K > 16352」）．b を上げると小さい K でも当たる：BF16 の 14080^3 は窓を最悪の上界に
    // したまま b_cap >= 68 で表へ戻る）．
    if (best.empty() && dram_cut) { inf.dram_retry = true; bool unused = false; best = oz2_awe_search(s, ub, false, unused, OZ2_AWE_NEW_BUDGET); }
    inf.found = !best.empty();
    return best;
}
