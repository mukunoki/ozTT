// SPDX-License-Identifier: Apache-2.0
#pragma once
// Ozaki scheme II: selection of the moduli set that minimizes a fitted stage-time model
// (host side). Used by the moduli selection (ozaki2_choose_moduli in moduli.hpp) for
// the formats whose path has a model row (Oz2Route::awe: BF16, FP16, BFP8, BFP4, BFP2).
//
// The set of moduli is chosen under the CRT window constraint, as in "AWE: Adaptive
// Weight Encoding for Exact Integer Matrix Products with Fewer GEMMs on FP4 Tensor Cores"
// (arXiv:2609.24519). The way a modulus is carried is fixed by oz2_split_mode (split.hpp);
// this file only chooses the set. The objective is the modelled time of all stages, not
// the number of matrix products: more moduli mean fewer products but more per-modulus
// work in slicing, composition and reconstruction.
//
//   cost of a modulus  = gp*products + (go+co)*outputs + cs + sop*(19 + digit ops) + spl*planes
//   one-digit modulus  = gp*1 + go*1 + sop*19 + spl*1 + OZ2_AWE_ONE_RECON*rb
//                        (bypasses the composition; one ModRound more in the reconstruction)
//   total              = sum of the modulus costs + max(R0, ra + rb*S)   (reconstruction)
// The coefficients depend on the problem size; the values fitted at K = 3520, 7040, 14080
// (cubic problems) are interpolated in log K (extrapolated from the two nearest rows).
//
// Search: moduli with identical cost (same products, outputs, planes, digit ops) form a
// class. Vectors of per-class counts are collected by a pruned depth-first traversal (in
// traversal order, capped at OZ2_AWE_VEC_MAX) and then sorted by cost; for each vector in
// that order an assignment search looks for pairwise coprime moduli whose product falls in
// the window [need, need+tol]; the first hit is returned. This is a heuristic with cutoffs
// and does not guarantee the optimum of the model:
//   - the per-class bit estimate est[] is the sum over the largest pairwise coprime
//     members taken greedily, not an upper bound on the reachable bits;
//   - at most OZ2_AWE_VEC_MAX vectors are collected (the cap drops the vectors the
//     traversal has not reached, whatever their cost);
//   - the assignment search is limited to OZ2_AWE_NODE_MAX nodes per vector and, when the
//     cost bound did not come from the plain greedy solution, to OZ2_AWE_NODE_BUDGET nodes
//     in total;
//   - the cost bound comes from a greedy solution (times OZ2_AWE_UB_SLACK) that need not
//     satisfy the tolerance.
// The result is deterministic. When no set is found the caller falls back to the table
// selection.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include "ozaki2/split.hpp"

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// Coefficients of the stage-time model at one K (see the formulas above): gp per matrix
// product and go per output in the GEMM stage, co per output and cs per modulus in the
// composition, sop per slicer tile operation and spl per digit plane in the slicing, and
// the reconstruction max(R0, ra + rb*S).
struct Oz2AweCoef {
    int K;
    double gp, go, co, cs, sop, spl, R0, ra, rb;
};

// OZ2_AWE_COEF[row][i]: rows in the order of the OZ2_AWE_* numbers (1-based), each with
// K = 3520, 7040, 14080. Fitted data; not to be edited by hand.
#include "ozaki2/awe_coef.inc"

// The table is declared with OZ2_AWE_COUNT rows, so a missing row would be zero-filled and
// still compile (and the interpolation would take log2(0)). Check per row that K is 3520,
// 7040, 14080 and that no coefficient is zero.
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
static_assert(oz2_awe_coef_rows_set(),
              "OZ2_AWE_COEF has an incomplete row (K not 3520, 7040, 14080 in order, or a zero coefficient)");

// Coefficients of model row `awe` at inner dimension K.
static inline Oz2AweCoef oz2_awe_coef(int awe, uint32_t K) {
    if (awe < 1 || awe > OZ2_AWE_COUNT)
        throw std::runtime_error("internal error: no stage-time model row for this format");
    const Oz2AweCoef* t = OZ2_AWE_COEF[awe - 1];
    const double x = std::log2((double)K);
    int i = 0;
    if (x > std::log2((double)t[1].K)) i = 1;
    const double x0 = std::log2((double)t[i].K), x1 = std::log2((double)t[i + 1].K), f = (x - x0) / (x1 - x0);
    auto ip = [&](double a, double b) {  // linear in log space (power law); linear for non-positive values
        if (a > 0 && b > 0) return std::exp(std::log(a) + f * (std::log(b) - std::log(a)));
        return a + f * (b - a);
    };
    Oz2AweCoef c;
    c.K = (int)K;
    c.gp = ip(t[i].gp, t[i + 1].gp);
    c.go = ip(t[i].go, t[i + 1].go);
    c.co = ip(t[i].co, t[i + 1].co);
    c.cs = ip(t[i].cs, t[i + 1].cs);
    c.sop = ip(t[i].sop, t[i + 1].sop);
    c.spl = ip(t[i].spl, t[i + 1].spl);
    c.R0 = ip(t[i].R0, t[i + 1].R0);
    c.ra = ip(t[i].ra, t[i + 1].ra);
    c.rb = ip(t[i].rb, t[i + 1].rb);
    return c;
}
// Modelled reconstruction time with S moduli (flat up to a break point, then linear).
static inline double oz2_awe_recon(const Oz2AweCoef& c, int S) { return std::max(c.R0, c.ra + c.rb * S); }

// Slicer tile operations for making the digits of modulus m (the 19 operations up to the
// residue are added separately).
static inline int oz2_awe_ops(int m, int dmax, Oz2Ndig ndig, int smax) {
    const int mode = oz2_split_mode(m, dmax, ndig, smax);
    if (mode == OZ2_SPLIT_BAL) return 7 * (oz2_bal_digits(m, dmax, ndig) - 1);
    if (mode == OZ2_SPLIT_ONE) return 0;
    if (mode == OZ2_SPLIT_KAR) return 12;
    if (mode == OZ2_SPLIT_SQS) return 13;
    return 6;
}

// Reconstruction surcharge of a one-digit modulus: one more ModRound in the CRT, about 6%
// of the per-modulus operations; multiplies rb.
inline constexpr double OZ2_AWE_ONE_RECON = 0.062;

// Free DRAM [bytes] assumed by the DRAM estimate of the model (oz2_dram_one_group). A
// model assumption only; the run-time band planning uses the measured free memory.
inline constexpr double OZ2_AWE_DRAM_FREE = 31.0e9;

// Bytes of one tile of the slice output (digit planes) per model row: the tile size of
// the format whose path has that row.
constexpr uint32_t oz2_awe_slice_tile(int awe) {
    for (const Oz2RouteRow& x : OZ2_ROUTES)
        if (x.r.awe == awe) return (uint32_t)tt::tile_size(oz_format(x.fmt).in);
    return 0;
}
inline constexpr uint32_t OZ2_AWE_SLICE_TILE[] = {oz2_awe_slice_tile(OZ2_AWE_BFP4), oz2_awe_slice_tile(OZ2_AWE_BFP2),
                                                  oz2_awe_slice_tile(OZ2_AWE_BF16), oz2_awe_slice_tile(OZ2_AWE_FP16),
                                                  oz2_awe_slice_tile(OZ2_AWE_BFP8)};
static_assert(std::size(OZ2_AWE_SLICE_TILE) == OZ2_AWE_COUNT,
              "OZ2_AWE_SLICE_TILE does not have one entry per OZ2_AWE_* number");
constexpr bool oz2_awe_slice_tiles_set() {
    for (int a = 1; a <= OZ2_AWE_COUNT; ++a)
        if (OZ2_AWE_SLICE_TILE[a - 1] == 0 || OZ2_AWE_SLICE_TILE[a - 1] != oz2_awe_slice_tile(a)) return false;
    return true;
}
static_assert(oz2_awe_slice_tiles_set(),
              "OZ2_AWE_SLICE_TILE is not in the order of the OZ2_AWE_* numbers, or a number has no format");

// Cutoffs of the search (see the header comment): enumerated vectors, assignment nodes per
// vector, assignment nodes in total when the cost bound is not the plain greedy one. The
// per-class count stops OZ2_AWE_OVER_NEED_BITS above the window, and the cost bound is
// OZ2_AWE_UB_SLACK times the greedy solution.
inline constexpr size_t OZ2_AWE_VEC_MAX = 200000;
inline constexpr long OZ2_AWE_NODE_MAX = 400000;
inline constexpr long OZ2_AWE_NODE_BUDGET = 10000000;
inline constexpr double OZ2_AWE_OVER_NEED_BITS = 16.0;
inline constexpr double OZ2_AWE_UB_SLACK = 1.05;

// --- Parts of the search (oz2_awe_choose calls them in this order) ------------------------
// A candidate modulus (m and log2 m), and a class of moduli with identical cost.
struct Oz2AweCand {
    int m;
    double bits;
};
struct Oz2AweClass {
    int nprod, nout, npl, ops;  // matrix products, GEMM outputs, digit planes, digit ops of one modulus
    bool one;                   // one-digit moduli (they bypass the composition, so they do not
                                // count for the composition planes of the DRAM estimate)
    double cost;                // modelled cost of one modulus
    std::vector<Oz2AweCand> v;  // the moduli, in descending order
    std::vector<double> est;    // est[t]: bits of the t largest pairwise coprime members taken greedily
};

// Build the classes: sort the usable moduli into classes, compute est[] per class, and
// order the classes by cost per bit (ties by cost).
static inline std::vector<Oz2AweClass> oz2_awe_classes(uint32_t K, const Oz2AweCoef& C, int mod_cap, int digit_max,
                                                       Oz2Ndig ndig, int smax) {
    std::vector<Oz2AweClass> ks;
    for (int m = mod_cap; m >= 2; --m) {
        if (!oz2_usable(m, digit_max, ndig, smax)) continue;
        // Concatenation bound, estimated with the modulus cap (the bound of the final set,
        // which uses its largest modulus, is at least this).
        const int mg = oz2_maxg_for(m, K, mod_cap, digit_max, ndig, smax);
        const int pl = oz2_digit_planes(m, digit_max, ndig);
        const int np = oz2_digit_prods(m, digit_max, ndig), no = (int)oz2_nout(m, digit_max, ndig, mg),
                  op = oz2_awe_ops(m, digit_max, ndig, smax);
        const bool one = oz2_split_mode(m, digit_max, ndig, smax) == OZ2_SPLIT_ONE;
        // Skip a modulus whose GEMM outputs are not exact FP32 integers once the modulus is
        // added, or whose reduce-first composition sum, sum |c_j| (m/2+2), reaches 2^22
        // (the same conditions as oz2_plan_concat in moduli.hpp).
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
        for (auto& x : ks)
            if (x.nprod == np && x.nout == no && x.npl == pl && x.ops == op && x.one == one) {
                k = &x;
                break;
            }
        if (!k) {
            // A one-digit modulus has no composition cost; instead the CRT of the
            // reconstruction runs one more ModRound.
            const double cost = one ? C.gp * np + C.go * no + C.sop * (19 + op) + C.spl * pl + OZ2_AWE_ONE_RECON * C.rb
                                    : C.gp * np + (C.go + C.co) * no + C.cs + C.sop * (19 + op) + C.spl * pl;
            ks.push_back({np, no, pl, op, one, cost, {}, {}});
            k = &ks.back();
        }
        k->v.push_back({m, std::log2((double)m)});
    }
    if (ks.empty()) return ks;
    for (auto& k : ks) {
        std::vector<int> got;
        k.est.assign(1, 0.0);
        for (auto& c : k.v) {
            bool ok = true;
            for (int g : got)
                if (std::gcd(g, c.m) != 1) {
                    ok = false;
                    break;
                }
            if (ok) {
                got.push_back(c.m);
                k.est.push_back(k.est.back() + c.bits);
            }
        }
    }
    std::sort(ks.begin(), ks.end(), [](const Oz2AweClass& a, const Oz2AweClass& b) {
        const double ra = a.cost / a.v[0].bits, rb = b.cost / b.v[0].bits;
        return ra != rb ? ra < rb : a.cost < b.cost;
    });
    return ks;
}

// Quantities that stay fixed during the search: need is the CRT window (lower bound on
// log2 of the product of the moduli), tol the allowed excess [bits], max_moduli the limit
// on the number of moduli, minrate[c] the smallest cost per bit among the classes c and
// later, and P, out, F the byte counts of the DRAM estimate (see oz2_dram_one_group).
struct Oz2AweSearch {
    const std::vector<Oz2AweClass>& ks;
    const Oz2AweCoef& C;
    double need, tol;
    int max_moduli;
    std::vector<double> minrate;
    double P, out, F;

    // Whether S moduli with npl digit planes fit in DRAM without bands, under the model
    // assumptions. mxo is the largest output count of a composed modulus (0: none).
    bool dram_ok(int S, int npl, int mxo) const {
        const Oz2ComposePlan cp = oz2_compose_plan(mxo > 0, (uint32_t)mxo);
        return oz2_dram_one_group(npl, P, cp.planes, S, out, F) <= OZ2_AWE_DRAM_FREE;
    }
};

// A vector of per-class counts and its modelled time.
struct Oz2AweVec {
    double T;
    std::vector<int> n;
};

// Enumeration: collect the count vectors whose time is below `incumbent` and whose bit
// estimate reaches the window (and, with `dram`, that pass dram_ok). dram_cut is set when
// a vector was dropped by the DRAM condition. At most OZ2_AWE_VEC_MAX vectors.
struct Oz2AweEnum {
    const Oz2AweSearch& s;
    double incumbent;
    bool dram;
    bool& dram_cut;
    std::vector<Oz2AweVec>& vecs;
    std::vector<int> n;  // counts of the current branch

    // Decide the counts of class c and later. S: moduli so far, cost: their cost,
    // est: their bit estimate, npl: their digit planes, mxo: largest composed output count.
    void rec(int c, int S, double cost, double est, int npl, int mxo) {
        const int NC = (int)s.ks.size();
        if (vecs.size() >= OZ2_AWE_VEC_MAX) return;
        // Vectors are collected at the leaves only (all classes decided), so that a
        // vector with trailing zero counts is not collected twice.
        if (c == NC) {
            if (est >= s.need && S > 0) {
                const double T = cost + oz2_awe_recon(s.C, S);
                if (T < incumbent) {
                    if (!dram || s.dram_ok(S, npl, mxo))
                        vecs.push_back({T, n});
                    else
                        dram_cut = true;
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
            if (t > 0 &&
                cost + t * k.cost + std::max(0.0, s.need - e2) * s.minrate[c + 1] + oz2_awe_recon(s.C, S + t) >=
                    incumbent)
                continue;  // this lower bound is not monotone in t
            rec(c + 1, S + t, cost + t * k.cost, e2, npl + t * k.npl, (t && !k.one) ? std::max(mxo, k.nout) : mxo);
            if (e2 >= s.need + OZ2_AWE_OVER_NEED_BITS) break;
        }
        n[c] = 0;
    }
};

// Assignment: pick pairwise coprime moduli from the classes according to one count vector
// so that the bits fall in [need, need+tol], minimizing the excess. Classes are filled in
// the order `ord` (fewest candidates first); the search stops after OZ2_AWE_NODE_MAX nodes
// or when the excess is practically zero.
struct Oz2AweAssign {
    const Oz2AweSearch& s;
    const std::vector<int>& ord;
    std::vector<int> cnt;                // counts in the order of ord
    std::vector<double> ubrest, lbrest;  // largest / smallest bits obtainable from the classes ord[i..]
    std::vector<int> pick, best;
    double bestex = 1e300;
    long nodes = 0;
    void dfs(int i, int start, int left, double bits) {
        const int NC = (int)s.ks.size();
        if (++nodes > OZ2_AWE_NODE_MAX || bestex <= 1e-4) return;
        if (i == NC) {
            const double ex = bits - s.need;
            if (ex >= 0 && ex <= s.tol && ex < bestex) {
                bestex = ex;
                best = pick;
            }
            return;
        }
        if (left == 0) {
            dfs(i + 1, 0, i + 1 < NC ? cnt[i + 1] : 0, bits);
            return;
        }
        const auto& v = s.ks[ord[i]].v;
        for (int j = start; j + left <= (int)v.size(); ++j) {
            double ubc = 0;
            for (int t = 0; t < left; ++t) ubc += v[j + t].bits;
            if (bits + ubc + ubrest[i + 1] < s.need) return;
            if (bits + v[j].bits + lbrest[i + 1] > s.need + s.tol && left == 1) continue;
            bool ok = true;
            for (int x : pick)
                if (std::gcd(x, v[j].m) != 1) {
                    ok = false;
                    break;
                }
            if (!ok) continue;
            pick.push_back(v[j].m);
            dfs(i, j + 1, left - 1, bits + v[j].bits);
            pick.pop_back();
            if (nodes > OZ2_AWE_NODE_MAX || bestex <= 1e-4) return;
        }
    }
};

// Assignment for the vector vec (empty if none). nodes returns the nodes visited.
static inline std::vector<int> oz2_awe_assign(const Oz2AweSearch& s, const std::vector<int>& ord, const Oz2AweVec& vec,
                                              long& nodes) {
    const int NC = (int)s.ks.size();
    Oz2AweAssign a{s,  ord, std::vector<int>(NC), std::vector<double>(NC + 1, 0.0), std::vector<double>(NC + 1, 0.0),
                   {}, {}};
    for (int i = 0; i < NC; ++i) a.cnt[i] = vec.n[ord[i]];
    for (int i = NC - 1; i >= 0; --i) {
        const auto& v = s.ks[ord[i]].v;
        const int t = a.cnt[i];
        double ubs = 0, lbs = 0;
        for (int j = 0; j < t && j < (int)v.size(); ++j) {
            ubs += v[j].bits;
            lbs += v[v.size() - 1 - j].bits;
        }
        a.ubrest[i] = a.ubrest[i + 1] + ubs;
        a.lbrest[i] = a.lbrest[i + 1] + lbs;
    }
    a.dfs(0, 0, NC ? a.cnt[0] : 0, 0.0);
    nodes = a.nodes;
    return a.best;
}

// Enumerate the count vectors with time below ub (with `dram`, only those that pass the
// DRAM condition), then try the assignment in order of time and return the first set that
// fits the window (descending; empty if none). dram_cut is set when the DRAM condition
// dropped a vector. With budget > 0 the assignment nodes are limited to budget in total;
// the budget is checked before a vector is started, so the last vector may exceed it by
// up to OZ2_AWE_NODE_MAX nodes.
static inline std::vector<int> oz2_awe_search(const Oz2AweSearch& s, double ub, bool dram, bool& dram_cut,
                                              long budget) {
    const int NC = (int)s.ks.size();
    std::vector<Oz2AweVec> vecs;
    Oz2AweEnum e{s, ub, dram, dram_cut, vecs, std::vector<int>(NC, 0)};
    e.rec(0, 0, 0.0, 0.0, 0, 0);
    std::stable_sort(vecs.begin(), vecs.end(), [](const Oz2AweVec& a, const Oz2AweVec& b) { return a.T < b.T; });
    // Assignment order: classes with the fewest candidates first.
    std::vector<int> ord(NC);
    for (int c = 0; c < NC; ++c) ord[c] = c;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return s.ks[a].v.size() < s.ks[b].v.size(); });
    long spent = 0;
    for (const auto& vec : vecs) {
        if (budget > 0 && spent > budget) break;
        long nodes = 0;
        std::vector<int> best = oz2_awe_assign(s, ord, vec, nodes);
        spent += nodes;
        if (!best.empty()) {
            std::sort(best.begin(), best.end(), std::greater<int>());
            return best;
        }
    }
    return {};
}

// Upper bound on the time: OZ2_AWE_UB_SLACK times the greedy solution that takes pairwise
// coprime moduli in order of cost per bit until the window is reached. To respect the
// limit max_moduli, a modulus is skipped when taking it would make the window unreachable
// even if all remaining slots held the largest modulus (log2 mod_cap bits each); without
// this look-ahead the greedy solution can exhaust the slots with cheap small moduli.
// `unbounded` is set when the plain greedy solution reached the window (nothing skipped):
// the search then runs without a total node budget. When the window is not reached there
// is no bound (1e300) and OZ2_AWE_VEC_MAX limits the enumeration.
static inline double oz2_awe_upper_bound(const Oz2AweSearch& s, int mod_cap, bool& unbounded) {
    double ub = 1e300;
    std::vector<std::pair<double, std::pair<int, double>>> all;  // (cost/bit, (m, cost))
    for (auto& k : s.ks)
        for (auto& c : k.v) all.push_back({k.cost / c.bits, {c.m, k.cost}});
    std::stable_sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const double maxbits = std::log2((double)mod_cap);
    std::vector<int> got;
    double bits = 0, cost = 0;
    bool skipped = false;
    for (auto& a : all) {
        if (bits >= s.need || (int)got.size() >= s.max_moduli) break;
        bool ok = true;
        for (int g : got)
            if (std::gcd(g, a.second.first) != 1) {
                ok = false;
                break;
            }
        if (!ok) continue;
        const double lb = std::log2((double)a.second.first);
        if (bits + lb + (s.max_moduli - (int)got.size() - 1) * maxbits < s.need) {
            skipped = true;
            continue;
        }
        got.push_back(a.second.first);
        bits += lb;
        cost += a.second.second;
    }
    if (bits >= s.need) ub = (cost + oz2_awe_recon(s.C, (int)got.size())) * OZ2_AWE_UB_SLACK + 1e-9;
    unbounded = bits >= s.need && !skipped;
    return ub;
}

// Select the moduli set (descending) for inner dimension K on model row `awe`. Returns an
// empty set when none fits the window; the caller then uses the table selection.
//   need        CRT window: lower bound on log2 of the product of the moduli
//   tol         allowed excess over need [bits]
//   max_moduli  limit on the number of moduli
//   M, N        output dimensions for the DRAM estimate
// Steps: classes -> cost bound -> enumeration and assignment -> if no set fits in DRAM
// without bands, search again without the DRAM condition.
static inline std::vector<int> oz2_awe_choose(uint32_t K, int awe, int mod_cap, int digit_max, Oz2Ndig ndig, int smax,
                                              double need, double tol, int max_moduli, uint32_t M, uint32_t N) {
    const Oz2AweCoef C = oz2_awe_coef(awe, K);
    const std::vector<Oz2AweClass> ks = oz2_awe_classes(K, C, mod_cap, digit_max, ndig, smax);
    if (ks.empty()) return {};
    const int NC = (int)ks.size();
    // DRAM estimate: digit planes of A and B, product planes and residues (dram_ok). The
    // output counts come from the concatenation bound estimated with the modulus cap, so
    // they are at least those of the final set.
    const double Md = M, Nd = N;
    Oz2AweSearch s{ks,
                   C,
                   need,
                   tol,
                   max_moduli,
                   std::vector<double>(NC + 1, 1e300),
                   ((double)Md * K + (double)K * Nd) / 1024.0 * OZ2_AWE_SLICE_TILE[awe - 1],
                   Md * Nd * 4.0,
                   ((double)Md * K + (double)K * Nd) * 8.0};
    for (int c = NC - 1; c >= 0; --c) s.minrate[c] = std::min(s.minrate[c + 1], ks[c].cost / ks[c].v[0].bits);
    bool unbounded = false;  // the plain greedy solution reached the window: search without a node budget
    const double ub = oz2_awe_upper_bound(s, mod_cap, unbounded);
    // Searches whose cost bound is not the plain greedy one (look-ahead bound, no bound, or
    // the retry below) are cut off after OZ2_AWE_NODE_BUDGET assignment nodes; without the
    // cutoff the paths with digit bound 32 can take tens of seconds.
    bool dram_cut = false;
    std::vector<int> best = oz2_awe_search(s, ub, true, dram_cut, unbounded ? 0 : OZ2_AWE_NODE_BUDGET);
    // The bounded search found no set and the DRAM condition dropped some vectors that
    // reached the window. That does not prove that no set fits without bands; as a
    // heuristic, search again without the condition and accept a set that the estimate
    // says needs bands (the run then processes the output in row bands, plan_bands). The
    // table selection is not tried instead: its set needs more matrix products.
    if (best.empty() && dram_cut) {
        bool unused = false;
        best = oz2_awe_search(s, ub, false, unused, OZ2_AWE_NODE_BUDGET);
    }
    return best;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
