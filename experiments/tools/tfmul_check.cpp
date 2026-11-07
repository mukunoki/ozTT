// SPDX-License-Identifier: Apache-2.0
//
// 基数倍 tfMulR・tfMulP2（common/kernels/sfpu_tf_proc.h）の最後の FastTwoSum が，本番で届く 3 本組に対して
// 前提を満たすか・誤差を出すかを，スライサと Horner をホストで模擬して調べる【検査用】
// ---------------------------------------------------------------------------
//     make -C src tfmul_check             （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     ./src/build/tfmul_check [1 構成あたりの入力数=20000] [seed=1]
// リポジトリ直下で回す：尾崎 I の経路の p・L は scripts/routes.tsv（経路の正本）から読み（tools/oz_sim.h の load_routes1．
// OZ_ROUTES_TSV で置き場所を変えられる），読めなければ止める（終了コード 2．古い p で黙って回らないように）．
// 基数と桁の上限は本番の表（ozaki1_param.h の oz1_param・oz1_default_digit_max）．読んだ構成は最初の行に出す．
// 経路の表そのものの検査（スキーム I・II の行が 8 形式 1 回ずつ，II の法の本数が本番の既定）は tools/routes_check.cpp．
//
// 何を確かめるか
//   tfMulR・tfMulP2 は最後に (a,b) -> (s,t) を FastTwoSum（s=a+b, t=b-(s-a)．3 演算）で正規化する．
//   FastTwoSum が s+t=a+b を厳密に返すのは「a の指数 >= b の指数」（a か b が 0 でもよい）のときで，
//   外れると最大で b の ulp の半分ほどを落とす．入力の 3 本組が非重複でない（中位が上位より大きい）
//   と外れうる：(1, 2^50, 2^26) を基数 2 倍すると (2^51, 2^27, 0) になり，真値の末尾 +2 を失う
//   （下の [0] で再現する）．問題は本番の 3 本組がそこに届くかである．
//   模擬は checks/sfpu_tf_emu.hpp（本番の sfpu_tf_proc.h の手続きを，SFPU の 1 演算のビット単位の模型の上で組む．
//   tfMulR・tfMulP2 の実機との一致は src/checks/tfmul_probe.cpp が見る）．各構成を 2 通り回す：A = 本番（FastTwoSum），B = 最後を Knuth の
//   TwoSum（6 演算）に置き換えたもの（模擬の Dst::final_twosum）．
//
//   [1] 尾崎 I のスライサ（slice_tf_num.cpp・slice_tf_sm8.cpp）：routes.tsv のスキーム I の経路ごとの基数・桁の上限・p
//       （同じ組の経路はまとめる．BF16・FP16・TF32・BFP8_b は基数 64・桁 32・p=10 で 1 つ）で，
//       FP64 の入力を split_f64 から切り，t *= 基数；d = clamp(round(t0))；t0 -= d を p 回．
//       厳密な残差 T_i = x β^i - Σ d_j β^(i-j)（512 bit の固定小数）と 3 本組の和の差（追跡誤差．
//       x の単位に直す）と，最後の打ち切り |T_p| β^-p を見る．入口の 3 本組が帰納の出発点 H_1(u)
//       （u = 2^(P+30)．P は x = S 2^P の指数で S は 53 bit の整数仮数．sfpu_tf_proc.h）を満たすことも全入力で数える
//       （入力には split_f64 の尾の絶対値の和が ulp(hi) を超える形も混ぜる：sfpu_tf_emu.hpp の gen_x の kind 4）．
//   [2] 尾崎 II の桁作り（ozaki2/kernels/oz2_digits.h．基数 2^20，3〜5 桁）：同上．
//   [3] 尾崎 I の再構成の Horner（recon_tf_i32in.cpp の 2 の補数入力と NSMALL，符号-絶対値入力（SIGNMAG=1．
//       INT8 経路の本番），recon_tf_f32in.cpp）：acc = acc*β + S_d を厳密な整数和と比べる．対角数は routes.tsv の
//       L + 1 で，構成は [1] と同じ組ごとに，FP 形式は遅延再構成（2 の補数の S_d．K=14080）と非遅延（FP32 の P_d．K=4096），
//       INT8 は符号-絶対値（K=14080）と，大きな S_d を 2 の補数で分けて tfMulR へ渡す本番に無い組（同じ基数・L）．符号-絶対値は
//       recon_tf_i32in.cpp の decode と同じく |S_d| を上位 |S_d| & 0x7FFFFF00・下位 |S_d| & 0xFF に分けて両方に符号を
//       掛け戻す（S_d = -257 は (-256, -1)．2 の補数の分け方 (-512, +255) とは足す列が違う）ので，2 の補数の経路とは
//       別に模擬する．基数倍の入力が sfpu_tf_proc.h の仮定 H_4(ulp(x0))（相殺しないとき）か
//       H_{2^11}(1)（相殺したとき）に入っていること（limb が整数で |x1| + |x2| <= max(4 ulp(x0), 2^11)）も数え，
//       直前の足し込みが相殺しなかった入力（sfpu_tf_proc.h の「相殺しない」）の 尾/ulp(x0) の最大を出す．
//   [4] 基数倍 1 回を，スライサの桁を引いた直後に現れうる 3 本組の族（桁を引いた後の x0 = 0, ±u, ±2u, 乱数と，
//       尾 x1 + x2 を ±u の端に寄せたもの）に直接当てる．基数は本番の基数（checks/sfpu_tf_emu.hpp の production_radices：
//       スキーム I の形式ごとの既定の基数と II の 2^20．[5] も同じ表から 2 のべきとそうでないものを取る）．
//   [5] sfpu_tf_proc.h の補題の族を直接当てる：(a) tfMulP2 に W（x0 = 0 か |x1 + x2| <= |x0|）の組．W は補題 P の仮定
//       H_c(u) とは別の条件で（u が 2 のべきでも x0 の約数でもない），「W なら前提が成り立ち，基数倍は厳密で，出力も W」
//       という命題そのものをここで確かめる．(b) tfMulR に E0 = β x0 - fl(β x0) を中位・下位がほぼ打ち消す組（前提が
//       崩れるとすればここ）．(c) tfMulR に補題 R の仮定 H_c(u)（x0 = k u，|x1| + |x2| <= c u．尾の 2 本は重なってよく，
//       大きく打ち消し合う組も入れる）の組を，β c < 2^8 の c と，Horner の H_4 に当たる β c < 2^10 の c で当て，前提違反 0・
//       FastTwoSum 版と TwoSum 版の誤差が同じ・出力の尾が c1 max(ulp(r0), ulp(β c u)) 以下（補題 R の出力 H_{c1}(u')）を見る．
//   [6] 入口 split_f64 の 3 本組そのもの：x = 2^-2 + 2^-25 + 127·2^-54 は (2^-2 + 2^-24, -2^-25, 127·2^-54) に割れ，
//       尾の絶対値の和が ulp(hi) = 2^-25 を超える（ulp(hi) を格子にした H_1 は入口で成り立たない）が，u = 2^-24 に
//       対しては H_1(u) を満たす．この例と同じ形（仮数の下位 30 bit が 2^29 + c）を指数と上位の仮数を振って作り，
//       H_1(u) を満たすこと・ulp(hi) を超える組が実際にあること・5 形式のスライサと尾崎 II の桁作りが
//       前提違反 0・不一致 0・（2 のべきの基数で）追跡誤差 0 であることを確かめる．
//
// 判定（どれかが崩れたら FAIL，終了コード 1）
//   ・[1]〜[6] のすべてで最後の FastTwoSum の前提違反が 0・不一致（s + t != a + b）が 0（補題 P・R が言うこと）．
//   ・[1][2][6] の入口の 3 本組がすべて H_1(u)（u = 2^(P+30)）．[6] では尾が ulp(hi) を超える組が 1 つ以上ある．
//   ・2 のべきの基数（tfMulP2）は [1][2][4] で追跡誤差 0（基数倍が厳密）．
//   ・tfMulR（INT8 の 254，BFP4_b の 14）は A の追跡誤差が打ち切り R β^-p の 2^-4 倍以下で，
//     かつ B（TwoSum）の追跡誤差の 2 倍 + 最下位の丸め 1 回ぶん以内．
//   ・[2] は A と B の桁が全入力で一致．[3] は A の誤差が B と同じ上界（Σ|S_d|β^(L-d) の 2^-66 倍）以内で，
//     基数倍の入力が H_4(ulp(x0))・H_{2^11}(1) の外に出ず，相殺しない入力の尾が 4 ulp(x0) 未満（符号-絶対値の経路も同じ）．
//   ・[0] の反例が実際に誤差を出すこと（計装が働いていることの確認）．
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <quadmath.h>
#include <random>
#include <vector>
#include <string>
#include "oz_sim.h"

using namespace sfpu_emu;

static bool g_verbose = false;   // TFMUL_VERBOSE=1：最後の FastTwoSum で誤差の出た組を表示する
// 模擬の DST（twosum = 比較用の B 版，st = 最後の FastTwoSum の計装の数える先）
static Dst emu(bool twosum, FtsStat* st) {
    Dst D; D.final_twosum = twosum; D.fts = st; D.fts_verbose = g_verbose;
    return D;
}

// --- 厳密な固定小数（512 bit の 2 の補数，最下位 2^-300）---------------------------------------
struct Fix {
    static constexpr int NW = 16, LSB = -300;
    uint32_t w[NW] = {};
    bool neg() const { return (w[NW - 1] >> 31) != 0; }
    bool zero() const { for (int i = 0; i < NW; i++) if (w[i]) return false; return true; }
    void add_mag(uint64_t m, int bit) {   // += m * 2^(bit + LSB)，bit >= 0
        int k = bit >> 5, sh = bit & 31;
        unsigned __int128 v = (unsigned __int128)m << sh;
        uint64_t c = 0;
        for (; k < NW && (v || c); k++) {
            const uint64_t s = (uint64_t)w[k] + (uint32_t)v + c;
            w[k] = (uint32_t)s; c = s >> 32; v >>= 32;
        }
    }
    void sub_mag(uint64_t m, int bit) {
        int k = bit >> 5, sh = bit & 31;
        unsigned __int128 v = (unsigned __int128)m << sh;
        uint64_t b = 0;
        for (; k < NW && (v || b); k++) {
            const uint64_t sub = (uint64_t)(uint32_t)v + b;
            b = ((uint64_t)w[k] < sub) ? 1 : 0;
            w[k] = (uint32_t)((uint64_t)w[k] - sub);
            v >>= 32;
        }
    }
    void add_f(float f, int sgn = 1) {
        if (f == 0.0f || !std::isfinite(f)) return;
        int e; const double m = std::frexp((double)f, &e);       // f = m * 2^e，0.5 <= |m| < 1
        const int64_t mi = (int64_t)std::ldexp(m, 24);           // 24 bit の整数仮数
        const int bit = e - 24 - LSB;
        if ((mi < 0) != (sgn < 0)) sub_mag((uint64_t)std::llabs(mi), bit); else add_mag((uint64_t)std::llabs(mi), bit);
    }
    void add_d(double f, int sgn = 1) {
        if (f == 0.0) return;
        int e; const double m = std::frexp(f, &e);
        const int64_t mi = (int64_t)std::ldexp(m, 53);
        const int bit = e - 53 - LSB;
        if ((mi < 0) != (sgn < 0)) sub_mag((uint64_t)std::llabs(mi), bit); else add_mag((uint64_t)std::llabs(mi), bit);
    }
    void add_i(int64_t v, int sgn = 1) {   // 整数 v
        if (!v) return;
        const int bit = -LSB;
        if ((v < 0) != (sgn < 0)) sub_mag((uint64_t)std::llabs(v), bit); else add_mag((uint64_t)std::llabs(v), bit);
    }
    void negate() { for (int i = 0; i < NW; i++) w[i] = ~w[i]; add_mag(1, 0); }
    void mul_small(int64_t k) {           // *= k（|k| < 2^32）
        const bool kn = k < 0; const uint64_t km = (uint64_t)(kn ? -k : k);
        const bool n = neg(); if (n) negate();
        uint64_t c = 0;
        for (int i = 0; i < NW; i++) { const unsigned __int128 p = (unsigned __int128)w[i] * km + c; w[i] = (uint32_t)p; c = (uint64_t)(p >> 32); }
        if (n != kn) negate();
    }
    Fix operator-(const Fix& o) const {
        Fix r = *this; uint64_t b = 0;
        for (int i = 0; i < NW; i++) {
            const uint64_t sub = (uint64_t)o.w[i] + b;
            b = ((uint64_t)r.w[i] < sub) ? 1 : 0;
            r.w[i] = (uint32_t)((uint64_t)r.w[i] - sub);
        }
        return r;
    }
    // |値| を __float128 で（大きさの比較用．113 bit に丸める）
    __float128 absq() const {
        Fix a = *this; if (a.neg()) a.negate();
        __float128 s = 0;
        for (int i = NW - 1; i >= 0; i--) s = s * 4294967296.0Q + (__float128)a.w[i];
        return s * powq(2.0Q, LSB);
    }
};
static Fix fix_tri(const Dst& D) { Fix f; f.add_f(D.r[0]); f.add_f(D.r[1]); f.add_f(D.r[2]); return f; }
static double lg2(__float128 v) { return v > 0 ? (double)log2q(v) : -9999.0; }

static bool g_fail = false;
static void check(bool ok, const char* what) { if (!ok) { std::printf("    NG: %s\n", what); g_fail = true; } }

// 入口の 3 本組（load_f64．split_f64 の指数シフト d = 0）が帰納の出発点 H_1(u) を満たすか：u = 2^(P+30)（x = S 2^P，
// S は 53 bit の整数仮数なので P = e - 53．e は frexp の指数）について x0 は u の倍数，|x1| + |x2| < u．
// over は尾の絶対値の和が ulp(hi) を超えた数（ulp(hi) を格子にした H_1 なら外れる組）
struct EntryH { uint64_t n = 0, out = 0, over = 0; double worst = 0; };   // worst は (|x1|+|x2|)/ulp(hi) の最大
static void entry_hcheck(double x, const Dst& D, EntryH& h) {
    if (x == 0.0) return;
    int e; std::frexp(x, &e);
    const double u = std::ldexp(1.0, e - 53 + 30);
    const double x0 = D.r[0], tail = std::fabs((double)D.r[1]) + std::fabs((double)D.r[2]);
    h.n++;
    if (!(std::fmod(x0, u) == 0.0 && tail < u)) h.out++;
    const double uh = (x0 == 0.0) ? 0.0 : std::ldexp(1.0, fexp(D.r[0]) - 24);
    if (uh > 0 && tail > uh) h.over++;
    if (uh > 0) h.worst = std::max(h.worst, tail / uh);
}

// --- [0] 前提を外した例（冒頭の (1, 2^50, 2^26) の反例）----------------------------------------
static void part0() {
    std::printf("[0] 非重複でない 3 本組（到達しない入力．計装の確認）\n");
    const float in[3] = {1.0f, 0x1p50f, 0x1p26f};
    for (int v = 0; v < 2; v++) {
        for (int which = 0; which < 2; which++) {
            FtsStat st;
            Dst D = emu(v == 1, &st); D.r[0] = in[0]; D.r[1] = in[1]; D.r[2] = in[2];
            if (which == 0) tfMulP2(D, 2.0f); else tfMulR(D, 2.0f);
            Fix ex; ex.add_f(in[0]); ex.add_f(in[1]); ex.add_f(in[2]); ex.mul_small(2);
            const Fix d = ex - fix_tri(D);
            std::printf("    %s %-7s (1,2^50,2^26)*2 -> (%a, %a, %a)  真値との差 %s%.0f  前提違反 %llu\n",
                        v ? "TwoSum    " : "FastTwoSum", which ? "tfMulR" : "tfMulP2", D.r[0], D.r[1], D.r[2],
                        d.neg() ? "-" : "+", (double)d.absq(), (unsigned long long)st.viol);
            if (v == 0) check(!d.zero() && st.inexact == 1, "反例で FastTwoSum の誤差が検出されない");
            else        check(d.zero(), "TwoSum 版が反例で厳密でない");
        }
    }
}

struct SliceCfg { std::string name; float beta; float dmax; int p; };

struct SliceRes {
    FtsStat fts;
    __float128 track = 0;   // max_i |T_i - (t0+t1+t2)_i| * β^-i（x の単位）
    __float128 trunc = 0;   // max |T_p| β^-p
    EntryH eh;              // 入口の 3 本組の H_1(u)
};

// 尾崎 I のスライサ 1 要素．digits に桁を書き，追跡誤差と最後の打ち切りを返す（proto は模擬の切り替えと計装）
static void slice1(const SliceCfg& c, const Dst& proto, double x, int* digits, __float128& track, __float128& trunc,
                   EntryH* eh = nullptr) {
    Dst D = proto; load_f64(D, x);
    if (eh) entry_hcheck(x, D, *eh);
    Fix T; T.add_d(x);
    __float128 scale = 1;
    { const Fix d0 = T - fix_tri(D); if (!d0.zero()) { std::printf("    NG: 入口の 3 本組が厳密でない x=%a\n", x); g_fail = true; } }
    for (int i = 0; i < c.p; i++) {
        const int d = slice_step(D, c.beta, c.dmax);
        digits[i] = d;
        T.mul_small((int64_t)c.beta); T.add_i(d, -1);
        scale /= (__float128)c.beta;
        const Fix df = T - fix_tri(D);
        if (!df.zero()) track = std::max<__float128>(track, df.absq() * scale);
    }
    trunc = std::max<__float128>(trunc, T.absq() * scale);
}

// xs が空でなければその入力を，空なら gen_x の 5 種類を n 個まわす
static SliceRes run_slicer(const SliceCfg& c, int n, uint64_t seed, bool twosum, std::vector<int>* digs,
                           const std::vector<double>& xs = {}) {
    SliceRes r;
    const Dst proto = emu(twosum, &r.fts);
    std::mt19937_64 g(seed);
    std::vector<int> dg(c.p);
    const double T = (double)c.dmax / ((double)c.beta - 1.0);   // 表せる上限 R（入口は |x| <= R）
    const int cnt = xs.empty() ? n : (int)xs.size();
    for (int k = 0; k < cnt; k++) {
        const double x = xs.empty() ? gen_x(g, k % 5, c.beta, c.p, T) : xs[k];
        slice1(c, proto, x, dg.data(), r.track, r.trunc, &r.eh);
        if (digs) digs->insert(digs->end(), dg.begin(), dg.end());
    }
    return r;
}

// 尾崎 I の構成（routes.tsv のスキーム I の経路を基数・桁の上限・p でまとめたもの．main が読む）
static std::vector<oz_sim::Cfg1> g_cfg1;
static std::vector<SliceCfg> slice_cfgs() {
    std::vector<SliceCfg> v;
    for (const auto& c : g_cfg1) v.push_back({c.name, (float)c.radix, (float)c.dmax, c.p});
    return v;
}

// スライサの 1 構成の結果を表示して判定する（[1] と [6] が使う）
static void slicer_verdict(const SliceCfg& c, const std::vector<double>& xs, int n, uint64_t seed) {
    {
        std::vector<int> da, db;
        const SliceRes A = run_slicer(c, n, seed, false, &da, xs), B = run_slicer(c, n, seed, true, &db, xs);
        uint64_t dd = 0; for (size_t i = 0; i < da.size(); i++) dd += da[i] != db[i];
        const bool p2 = (f32_bits(c.beta) & 0x7FFFFFu) == 0u;
        const __float128 R = (__float128)c.dmax / ((__float128)c.beta - 1), bnd = R * powq((__float128)c.beta, -c.p);
        std::printf("  %-20s %s 呼び出し %llu 前提違反 %llu 不一致 %llu（最大 |損失|/|b| = 2^%.1f）  追跡誤差 A 2^%.1f B 2^%.1f"
                    "  打ち切り 2^%.1f（R β^-p = 2^%.1f）  A/B で桁の違う要素 %llu\n",
                    c.name.c_str(), p2 ? "tfMulP2" : "tfMulR ", (unsigned long long)A.fts.calls, (unsigned long long)A.fts.viol,
                    (unsigned long long)A.fts.inexact, A.fts.inexact ? std::log2(A.fts.worst_loss_rel) : -9999.0,
                    lg2(A.track), lg2(B.track), lg2(A.trunc), lg2(bnd), (unsigned long long)dd);
        if (p2) {
            check(A.fts.viol == 0 && A.fts.inexact == 0, "2 のべきの基数で FastTwoSum の前提が崩れた");
            check(A.track == 0 && B.track == 0, "2 のべきの基数で基数倍が厳密でない");
        } else {
            check(A.fts.viol == 0 && A.fts.inexact == 0, "tfMulR で FastTwoSum の前提が崩れた（補題 R）");
            check(A.track <= bnd / 16, "tfMulR の追跡誤差が打ち切りの 2^-4 倍を超えた");
            // B の誤差の 2 倍に，最下位の丸め 1 回ぶん（|t| <= β/2 の 2^-72 倍を x の単位へ：β^-1 倍）を足した上界
            check(A.track <= 2 * B.track + powq(2.0Q, -72) * (__float128)c.beta / 2 / (__float128)c.beta,
                  "tfMulR の追跡誤差が TwoSum 版の 2 倍 + 最下位の丸め 1 回ぶんを超えた");
        }
        check(A.trunc <= bnd * (1 + 1e-6Q) + A.track, "打ち切りが R β^-p + 追跡誤差を超えた");
        std::printf("  %-20s 入口の 3 本組 %llu 個のうち H_1(u) の外 %llu 個（尾の絶対値の和が ulp(hi) を超えた組 %llu 個．"
                    "尾/ulp(hi) の最大 %.9f）\n", "", (unsigned long long)A.eh.n, (unsigned long long)A.eh.out,
                    (unsigned long long)A.eh.over, A.eh.worst);
        check(A.eh.n > 0 && A.eh.out == 0, "入口の 3 本組が H_1(u)（u = 2^(P+30)）の外に出た");
    }
}

static void part1(int n, uint64_t seed) {
    std::printf("[1] 尾崎 I のスライサ（入力 %d 個 × 5 種類を混ぜる）\n", n);
    for (const auto& c : slice_cfgs()) slicer_verdict(c, {}, n, seed);
}

// --- [2] 尾崎 II の桁作り（ozaki2/kernels/oz2_digits.h）---------------------------------------------------
static void part2(int n, uint64_t seed) {
    std::printf("[2] 尾崎 II の桁作り（基数 2^20，入力 %d 個）\n", n);
    for (int ND : {3, 4, 5}) {
        const int top = (ND == 3) ? 14 : 20;   // 最上位の |T| <= 2^(b-(ND-1)G)：b=54 なら 2^14，b=80・100 なら 2^20
        std::vector<int64_t> dg[2]; FtsStat st[2]; __float128 trk[2] = {0, 0};
        EntryH eh;
        for (int v = 0; v < 2; v++) {
            const Dst proto = emu(v == 1, &st[v]);
            std::mt19937_64 g(seed + ND);
            for (int k = 0; k < n; k++) {
                double x = gen_x(g, k % 5, 1048576.0, ND, 1.0);    // |x| <= 1
                x = std::ldexp(x, top);                              // T = A' / 2^((ND-1)G)
                Dst D = proto; load_f64(D, x);
                if (v == 0) entry_hcheck(x, D, eh);
                Fix Tx; Tx.add_d(x);
                __float128 scale = 1;
                for (int i = 0; i < ND; i++) {
                    if (i > 0) { tfMulP2(D, 1048576.0f); Tx.mul_small(1048576); scale /= 1048576.0Q; }
                    D.Cpy(0, 7); D.Rnd(7); D.Sub(0, 7, 0);
                    const int64_t d = (int64_t)D.r[7];
                    dg[v].push_back(d);
                    Tx.add_i(d, -1);
                    const Fix df = Tx - fix_tri(D);
                    // 最後の桁の後の残差は使わない（捨てる）ので，追跡は最後の桁の前まで見る
                    if (i + 1 < ND && !df.zero()) trk[v] = std::max<__float128>(trk[v], df.absq() * scale);
                }
            }
        }
        uint64_t dd = 0; for (size_t i = 0; i < dg[0].size(); i++) dd += dg[0][i] != dg[1][i];
        std::printf("  ND=%d 呼び出し %llu 前提違反 %llu 不一致 %llu  追跡誤差 A 2^%.1f B 2^%.1f  A/B で違う桁 %llu\n", ND,
                    (unsigned long long)st[0].calls, (unsigned long long)st[0].viol, (unsigned long long)st[0].inexact,
                    lg2(trk[0]), lg2(trk[1]), (unsigned long long)dd);
        check(dd == 0, "尾崎 II の桁が FastTwoSum 版と TwoSum 版で違う");
        check(st[0].viol == 0 && st[0].inexact == 0, "尾崎 II の桁作りで FastTwoSum の前提が崩れた（補題 P）");
        check(trk[0] == 0, "尾崎 II の桁作りで最後の桁の前の残差が厳密でない");
        std::printf("       入口の 3 本組 %llu 個のうち H_1(u) の外 %llu 個（尾が ulp(hi) を超えた組 %llu 個）\n",
                    (unsigned long long)eh.n, (unsigned long long)eh.out, (unsigned long long)eh.over);
        check(eh.n > 0 && eh.out == 0, "尾崎 II の入口の 3 本組が H_1(u) の外に出た");
    }
}

// --- [3] 尾崎 I の再構成の Horner（recon_tf_i32in.cpp／recon_tf_f32in.cpp）----------------------------
// path 0 = i32 2 の補数（遅延再構成の S_d．NSMALL あり），1 = fp32p，2 = i32 符号-絶対値（SIGNMAG=1．INT8 経路の P_d．NSMALL なし）
struct HornerCfg { std::string name; float beta; int dmax; int L; int K; int path; };
// Horner の基数倍の入力が sfpu_tf_proc.h の仮定に入っているか：limb がすべて整数で，尾の絶対値の和が
// 4 ulp(x0)（H_4(ulp(x0))．相殺しないとき）か 2^11（H_{2^11}(1)．相殺したとき）以下．
// nc は直前の足し込みが sfpu_tf_proc.h の意味で相殺しなかったか（足し込みの前の上位 r0，1 回目の後の s1，後の s = x0 について
// |s| >= |r0|/2 かつ |s| >= |s1|/2）で，そのときは尾が 4 ulp(x0) 未満（1 回だけ足したなら 3 ulp(x0) 未満）のはずである．
struct HornerH {
    uint64_t calls = 0, out = 0;
    double worst = 0;                 // 尾 > 2^11 の入力での 尾/ulp(x0) の最大
    uint64_t nc = 0; double worst_nc = 0;   // 相殺しない入力の数と，その 尾/ulp(x0) の最大
};
static void horner_hcheck(const Dst& D, HornerH& h, bool nc) {
    const float x0 = D.r[0], x1 = D.r[1], x2 = D.r[2];
    const double tail = std::fabs((double)x1) + std::fabs((double)x2);
    const double u0 = (x0 == 0.0f) ? 0.0 : std::ldexp(1.0, fexp(x0) - 24);
    const bool ints = x0 == std::floor(x0) && x1 == std::floor(x1) && x2 == std::floor(x2);
    h.calls++;
    if (!ints || !(tail <= std::max(4.0 * u0, 2048.0))) h.out++;
    if (tail > 2048.0 && u0 > 0) h.worst = std::max(h.worst, tail / u0);
    if (nc && u0 > 0) { h.nc++; h.worst_nc = std::max(h.worst_nc, tail / u0); }
}

static void part3(int n, uint64_t seed) {
    std::printf("[3] 尾崎 I の再構成の Horner（入力 %d 個）\n", n);
    // 構成は経路の表（routes.tsv の L と本番の基数・桁の上限）から作る（冒頭の [3]）．INT8 経路の本番は符号-絶対値
    // （SIGNMAG=1）．同じ基数の 2 の補数は本番に無い組だが，大きな S_d を 2 の補数で分けて tfMulR へ渡す族として残す
    std::vector<HornerCfg> cfgs;
    for (const auto& c : g_cfg1) {
        if (c.int32_partial) {
            cfgs.push_back({c.name + " i32 signmag", (float)c.radix, c.dmax, c.L, 14080, 2});
            cfgs.push_back({std::to_string(c.radix) + " i32 2c (外)", (float)c.radix, c.dmax, c.L, 14080, 0});
        } else {
            cfgs.push_back({c.name + " i32 (defer)", (float)c.radix, c.dmax, c.L, 14080, 0});
        }
    }
    for (const auto& c : g_cfg1)
        if (!c.int32_partial) cfgs.push_back({c.name + " fp32p", (float)c.radix, c.dmax, c.L, 4096, 1});
    for (const auto& c : cfgs) {
        const int nd = c.L + 1;
        std::vector<int64_t> bd(nd);
        for (int d = 0; d < nd; d++) {
            const long double b = (long double)(d + 1) * c.K * c.dmax * c.dmax;
            bd[d] = c.path == 1 ? (int64_t)std::min<long double>(b, 16777215.0L) : (int64_t)std::min<long double>(b, 2147483647.0L);
        }
        // 型変換 1 回で読む先頭の対角の本数は本番と同じ関数（ozaki1_param.h の oz1_recon_nsmall．ozaki1_plan.hpp の recon_nsmall）
        const int nsmall = c.path == 0 ? (int)oz1_recon_nsmall((uint32_t)c.dmax, (uint32_t)c.K, (uint32_t)nd) : 0;
        FtsStat st[2]; __float128 worst[2] = {0, 0};
        HornerH hh;
        for (int v = 0; v < 2; v++) {
            const Dst proto = emu(v == 1, &st[v]);
            std::mt19937_64 g(seed + 7 * c.L);
            std::vector<int64_t> S(nd);
            for (int k = 0; k < n; k++) {
                const int kind = k % 3;
                for (int d = 0; d < nd; d++) {
                    std::uniform_int_distribution<int64_t> U(-bd[d], bd[d]);
                    S[d] = U(g);
                    if (kind == 1 && (g() & 3)) S[d] = (g() & 1) ? bd[d] : -bd[d];          // 上界ちょうど
                    if (kind == 2 && d > 0) {                                                // 先頭の数段を打ち消す
                        __int128 acc = 0; for (int j = 0; j < d; j++) acc = acc * (int64_t)c.beta + S[j];
                        const __int128 want = -acc * (int64_t)c.beta + (int64_t)(g() % 9) - 4;
                        if (want >= -bd[d] && want <= bd[d]) S[d] = (int64_t)want;
                    }
                }
                Dst D = proto;
                Fix ex, ab;
                bool nc = true;   // 直前の足し込みが相殺しなかったか（horner_hcheck）
                for (int d = 0; d < nd; d++) {
                    if (d > 0) { if (v == 0) horner_hcheck(D, hh, nc); tfMulRadix(D, c.beta); }
                    const float r0 = (d == 0) ? 0.0f : D.r[0];   // 足し込みの前の上位（初回はアキュムレータを 0 か S_0 に置く）
                    float s1 = 0.0f;                             // 2 回足すときの 1 回目の後の上位
                    bool two = false;
                    if (c.path == 1) {
                        if (d == 0) { D.r[0] = (float)S[0]; D.Sub(0, 0, 1); D.Sub(0, 0, 2); }
                        else        { D.r[3] = (float)S[d]; tfAddS<3>(D); }
                    } else if (c.path == 2) {
                        // 符号-絶対値（recon_tf_i32in.cpp の decode の SIGNMAG=1）：sgn = sign(S)，rh = |S| & 0x7FFFFF00，
                        // rl = |S| & 0xFF をそれぞれ FP32 へ移し，両方に sgn を掛ける（±1 倍・0 倍は厳密）
                        const int64_t mag = S[d] < 0 ? -S[d] : S[d];
                        D.r[6] = S[d] > 0 ? 1.0f : (S[d] < 0 ? -1.0f : 0.0f);
                        D.r[3] = (float)(int32_t)(mag & 0x7FFFFF00); D.r[7] = (float)(int32_t)(mag & 0xFF);
                        D.Mul(3, 6, 3); D.Mul(7, 6, 7);
                        if (d == 0) { D.Sub(3, 3, 0); D.Sub(3, 3, 1); D.Sub(3, 3, 2); }
                        tfAddS<3>(D); s1 = D.r[0]; two = true; D.Cpy(7, 3); tfAddS<3>(D);
                    } else if (d < nsmall) {
                        if (d == 0) { D.r[0] = (float)S[0]; D.Sub(0, 0, 1); D.Sub(0, 0, 2); }
                        else        { D.r[3] = (float)S[d]; tfAddS<3>(D); }
                    } else {
                        const int32_t s32 = (int32_t)S[d];
                        D.r[3] = (float)(int32_t)(s32 & (int32_t)0xFFFFFF00); D.r[7] = (float)(s32 & 0xFF);
                        if (d == 0) { D.Sub(3, 3, 0); D.Sub(3, 3, 1); D.Sub(3, 3, 2); }
                        tfAddS<3>(D); s1 = D.r[0]; two = true; D.Cpy(7, 3); tfAddS<3>(D);
                    }
                    {
                        const double s = std::fabs((double)D.r[0]);
                        nc = s >= std::fabs((double)r0) / 2 && (!two || s >= std::fabs((double)s1) / 2);
                    }
                    ex.mul_small((int64_t)c.beta); ex.add_i(S[d]);
                    ab.mul_small((int64_t)c.beta); ab.add_i(std::llabs(S[d]));
                }
                const Fix df = ex - fix_tri(D);
                if (!df.zero()) worst[v] = std::max<__float128>(worst[v], df.absq() / ab.absq());
            }
        }
        std::printf("  %-18s 呼び出し %llu 前提違反 %llu 不一致 %llu（最大 |損失|/|b| = 2^%.1f）  誤差/Σ|S_d|β^(L-d)：A 2^%.1f B 2^%.1f\n",
                    c.name.c_str(), (unsigned long long)st[0].calls, (unsigned long long)st[0].viol, (unsigned long long)st[0].inexact,
                    st[0].inexact ? std::log2(st[0].worst_loss_rel) : -9999.0, lg2(worst[0]), lg2(worst[1]));
        std::printf("  %-18s 基数倍の入力 %llu 個のうち H_4(ulp(x0))・H_{2^11}(1) の外 %llu 個（尾 > 2^11 の入力の 尾/ulp(x0) の最大 %.3f）\n",
                    "", (unsigned long long)hh.calls, (unsigned long long)hh.out, hh.worst);
        std::printf("  %-18s 相殺しない入力 %llu 個の 尾/ulp(x0) の最大 %.3f（sfpu_tf_proc.h の評価は 4 未満）\n",
                    "", (unsigned long long)hh.nc, hh.worst_nc);
        check(st[0].viol == 0 && st[0].inexact == 0, "Horner で FastTwoSum の前提が崩れた（補題 P・R）");
        check(hh.calls > 0 && hh.out == 0, "Horner の基数倍の入力が補題の仮定 H_4(ulp(x0))・H_{2^11}(1) の外に出た");
        check(hh.nc > 0 && hh.worst_nc < 4.0, "相殺しない入力の尾が 4 ulp(x0) に届いた（sfpu_tf_proc.h の H_4 の評価が崩れた）");
        check(worst[0] <= powq(2.0Q, -66), "Horner の誤差が Σ|S_d|β^(L-d) の 2^-66 倍を超えた");
        check(worst[1] <= powq(2.0Q, -66), "Horner（TwoSum 版）の誤差が 2^-66 倍を超えた");
    }
}

// --- [4] 基数倍 1 回を「桁を引いた直後の形」の 3 本組に当てる -----------------------------------------
// 桁を引いた直後の 3 本組 (x0, x1, x2)：直前の上位 t0 の ulp を u として x0 = t0 - d は u の整数倍
// （|x0| <= 1/2 強），(x1, x2) は直前の正規化（FastTwoSum/TwoSum）の出力で |x2| <= ulp(x1)/2，
// かつ |x1 + x2| <= u（基数倍の出力の中位・下位は上位の ulp 以下．split_f64 も |mid+lo| <= ulp(hi)）．
// x0 を ±u, ±2u, 0 と乱数に振り，x1+x2 を ±u の近く（打ち消しの端）に寄せる．
// 本番の基数のうち 2 のべき（p2=true）かそうでないもの（小さい順．production_radices の並び）
static std::vector<float> radices(bool p2) {
    std::vector<float> v;
    for (float b : production_radices(true))
        if (((f32_bits(b) & 0x7FFFFFu) == 0u) == p2) v.push_back(b);
    return v;
}
static void part4(int n, uint64_t seed) {
    std::printf("[4] 桁を引いた直後の 3 本組への基数倍 1 回（%d 組 × 基数）\n", n);
    for (float beta : production_radices(true)) {
        FtsStat stA; __float128 worstA = 0, worstB = 0;
        const bool p2 = (f32_bits(beta) & 0x7FFFFFu) == 0u;
        for (int v = 0; v < 2; v++) {
            FtsStat st;
            const Dst proto = emu(v == 1, &st);
            std::mt19937_64 g(seed + (uint64_t)beta);
            std::uniform_real_distribution<double> U(-1.0, 1.0);
            for (int k = 0; k < n; k++) {
                const int eu = -8 - (int)(g() % 40);          // u = 2^eu
                const double u = std::ldexp(1.0, eu);
                const int jm = (int)(g() % 5);
                double x0 = (jm == 0) ? 0.0 : (jm == 1 ? u : jm == 2 ? 2 * u : std::floor(U(g) * 64) * u);
                if (g() & 1) x0 = -x0;
                double s12 = (k & 1) ? u * (1.0 - std::ldexp(std::fabs(U(g)), -(int)(g() % 30))) : u * U(g);
                if (g() & 1) s12 = -s12;
                const float x1 = (float)s12;
                float x2 = (float)(s12 - (double)x1);
                if (std::fabs((double)x1 + x2) > u) x2 = 0.0f;
                Dst D = proto; D.r[0] = (float)x0; D.r[1] = x1; D.r[2] = x2;
                Fix ex = fix_tri(D); ex.mul_small((int64_t)beta);
                // 相対の基準は |x0|+|x1|+|x2| の β 倍（打ち消して小さくなった値に対する相対は，最下位の丸め
                // （MulS(2, rdx)）だけで大きくなり，FastTwoSum の良し悪しを表さない）
                const __float128 mag = ((__float128)std::fabs(x0) + std::fabs(x1) + std::fabs(x2)) * beta;
                tfMulRadix(D, beta);
                const Fix df = ex - fix_tri(D);
                const __float128 rel = df.zero() ? 0 : df.absq() / mag;
                (v ? worstB : worstA) = std::max<__float128>(v ? worstB : worstA, rel);
            }
            if (v == 0) stA = st;
        }
        std::printf("  β=%-8g %s 前提違反 %llu 不一致 %llu（最大 |損失|/|b| = 2^%.1f）  誤差/(β Σ|x_i|) A 2^%.1f B 2^%.1f\n", beta,
                    p2 ? "tfMulP2" : "tfMulR ", (unsigned long long)stA.viol, (unsigned long long)stA.inexact,
                    stA.inexact ? std::log2(stA.worst_loss_rel) : -9999.0, lg2(worstA), lg2(worstB));
        if (p2) check(stA.viol == 0 && stA.inexact == 0 && worstA == 0, "2 のべきの基数で桁を引いた直後の形に誤差が出た");
        else    check(stA.viol == 0 && stA.inexact == 0 && worstA <= 2 * worstB + powq(2.0Q, -70),
                      "tfMulR で前提が崩れたか，誤差が TwoSum 版の 2 倍 + 2^-70 を超えた");
    }
}


// --- [5] 前提の補題を一般の族で直接確かめる ------------------------------------------------------------
// (a) tfMulP2：入力が W（x0 = 0 か |x1 + x2| <= |x0|）なら前提は必ず成り立ち，基数倍は厳密で，出力も W を満たす．
//     W は補題 P の仮定 H_c(u)（x0 は 2 のべき u の倍数で |x1| + |x2| <= c u）とは別の条件で，この命題そのものを
//     ここで確かめる（補題 P の証明の一部ではない）．x0 を任意の FP32，(x1,x2) を |x1 + x2| <= |x0| の正規化された組に振る．
// (b) tfMulR：E0 = β x0 - fl(β x0) を中位・下位の β 倍がほぼ打ち消す組（gen_r_cancel．s_b が小さくなり，
//     前提が崩れるとすればここ）でも前提違反 0・不一致 0（補題 R）．A と B（TwoSum 版）の誤差も同じになる．
static void part5(int n, uint64_t seed) {
    std::printf("[5] 前提の補題（一般の族．%d 組）\n", n);
    std::mt19937_64 g(seed + 555);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (float beta : radices(true)) {
        FtsStat st; uint64_t notexact = 0, notw = 0;
        for (int k = 0; k < n; k++) {
            const float x0 = (k % 5 == 0) ? 0.0f : (float)std::ldexp(U(g), -(int)(g() % 60));
            const double lim = (x0 == 0.0f) ? std::ldexp(1.0, -(int)(g() % 60)) : std::fabs((double)x0);
            double s12 = lim * U(g);
            if (k & 1) s12 = std::copysign(lim * (1.0 - std::ldexp(std::fabs(U(g)), -(int)(g() % 40))), U(g));
            const float x1 = (float)s12; float x2 = (float)(s12 - (double)x1);
            if (x0 != 0.0f && std::fabs((double)x1 + x2) > std::fabs((double)x0)) continue;
            Dst D = emu(false, &st); D.r[0] = x0; D.r[1] = x1; D.r[2] = x2;
            Fix ex = fix_tri(D); ex.mul_small((int64_t)beta);
            tfMulP2(D, beta);
            notexact += !(ex - fix_tri(D)).zero();
            notw += !(D.r[0] == 0.0f || std::fabs((double)D.r[1] + D.r[2]) <= std::fabs((double)D.r[0]));
        }
        std::printf("  (a) β=%-8g tfMulP2 呼び出し %llu 前提違反 %llu 不一致 %llu 基数倍が厳密でない %llu 出力が W でない %llu\n", beta,
                    (unsigned long long)st.calls, (unsigned long long)st.viol, (unsigned long long)st.inexact,
                    (unsigned long long)notexact, (unsigned long long)notw);
        check(st.viol == 0 && st.inexact == 0 && notexact == 0 && notw == 0, "tfMulP2 の命題（W => 前提・厳密・W）が崩れた");
    }
    for (float beta : radices(false)) {
        uint64_t made = 0, viol = 0, inex = 0; __float128 dmax = 0, errA = 0, errB = 0;
        for (int k = 0; k < n; k++) {
            float x[3];
            if (!gen_r_cancel(g, beta, x)) continue;
            made++;
            const double P0 = (double)(float)((double)x[0] * beta);
            __float128 e[2];
            for (int v = 0; v < 2; v++) {
                FtsStat st;
                Dst D = emu(v == 1, &st); D.r[0] = x[0]; D.r[1] = x[1]; D.r[2] = x[2];
                Fix ex = fix_tri(D); ex.mul_small((int64_t)beta);
                tfMulR(D, beta);
                e[v] = (ex - fix_tri(D)).absq() / (__float128)std::fabs(P0);
                if (v == 0) { viol += st.viol; inex += st.inexact; }
            }
            errA = std::max<__float128>(errA, e[0]); errB = std::max<__float128>(errB, e[1]);
            dmax = std::max<__float128>(dmax, e[0] > e[1] ? e[0] - e[1] : 0);
        }
        std::printf("  (b) β=%-4g tfMulR 構成 %llu 前提違反 %llu 不一致 %llu  誤差/|P0|：A 2^%.1f B 2^%.1f  (A-B)/|P0| の最大 2^%.1f\n", beta,
                    (unsigned long long)made, (unsigned long long)viol, (unsigned long long)inex,
                    lg2(errA), lg2(errB), lg2(dmax));
        check(made > (uint64_t)n / 10, "gen_r_cancel が構成を作れない");
        check(viol == 0 && inex == 0 && dmax == 0, "tfMulR の打ち消しの形で FastTwoSum の前提が崩れた（補題 R）");
    }
    // (c) 補題 R の仮定 H_c(u) そのもの．u = 2^eu，x0 = k u（k は 0，小さい整数，2^24/β 未満（F0 = 0），2^24/β 以上
    //     2^24 未満（F0 != 0）），尾は |x1| + |x2| <= c u を満たす任意の 2 本（重なってよい）で，半分は x2 ≈ -x1 の打ち消す組．
    for (float beta : radices(false)) {
        const double cmax8 = std::floor(255.0 / beta);                  // β c < 2^8 の最大の整数 c（14 は 18，254 は 1）
        const double cmax10 = std::floor(1023.0 / beta / (1 + 1e-6));   // Horner の H_4 が使う β c < 2^10（254 は 4）
        const double cl[3] = {1.0, cmax8, cmax10};
        for (int ci = 0; ci < 3; ci++) {
            const double c = cl[ci];
            if (ci > 0 && c == cl[ci - 1]) continue;   // 254 では β c < 2^8 の最大が 1 と同じ
            std::mt19937_64 g(seed + 777 + (uint64_t)beta * 131 + (uint64_t)c);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            uint64_t made = 0, viol = 0, inex = 0, tailbad = 0; __float128 dmax = 0;
            for (int k = 0; k < n; k++) {
                const int eu = -30 + (int)(g() % 40);
                const double u = std::ldexp(1.0, eu);
                const int kk = (int)(g() % 4);
                const int64_t lim0 = (int64_t)std::ceil(16777216.0 / beta);
                int64_t q = kk == 0 ? 0 : kk == 1 ? 1 + (int64_t)(g() % 8)
                          : kk == 2 ? (int64_t)(g() % (uint64_t)lim0) : lim0 + (int64_t)(g() % (uint64_t)(16777216 - lim0));
                if (g() & 1) q = -q;
                const float x0 = (float)((double)q * u);
                const double T = c * u * ((k % 5 == 0) ? 1.0 : U(g));        // 尾の絶対値の和の目標（端も入れる）
                float x1, x2;
                if (k & 1) {   // 打ち消す組：|x1| <= T/2，x2 = -x1 (1 - ε)
                    x1 = (float)(T / 2 * U(g)) * ((g() & 1) ? 1.f : -1.f);
                    x2 = (float)(-(double)x1 * (1.0 - std::ldexp(U(g), -(int)(g() % 30))));
                } else {       // 任意の 2 本（重なってよい）
                    const double a = T * U(g);
                    x1 = (float)a * ((g() & 1) ? 1.f : -1.f);
                    x2 = (float)((T - std::fabs((double)x1)) * U(g)) * ((g() & 1) ? 1.f : -1.f);
                }
                if ((__float128)std::fabs(x1) + std::fabs(x2) > (__float128)(c * u)) continue;
                made++;
                __float128 e[2]; float out[3] = {0, 0, 0};
                for (int v = 0; v < 2; v++) {
                    FtsStat st;
                    Dst D = emu(v == 1, &st); D.r[0] = x0; D.r[1] = x1; D.r[2] = x2;
                    Fix ex = fix_tri(D); ex.mul_small((int64_t)beta);
                    tfMulR(D, beta);
                    e[v] = (ex - fix_tri(D)).absq();
                    if (v == 0) { viol += st.viol; inex += st.inexact;
                                  out[0] = D.r[0]; out[1] = D.r[1]; out[2] = D.r[2]; }
                }
                dmax = std::max<__float128>(dmax, e[0] > e[1] ? e[0] - e[1] : 0);
                // 出力の尾：H_{c1}(u')，u' = ulp(r0) か ulp(s_a)（|s_a| <= β c u (1+2^-22)）
                const double sa = beta * c * u * (1 + std::ldexp(1.0, -22));
                const double usa = std::ldexp(1.0, fexp((float)sa) - 24);       // ulp(s_a) の上界
                const double ur0 = out[0] == 0.0f ? 0.0 : std::ldexp(1.0, fexp(out[0]) - 24);
                const double tail = std::fabs((double)out[1]) + std::fabs((double)out[2]);
                if (!(tail <= (1 + std::ldexp(1.0, -24)) * std::max(ur0, usa))) tailbad++;
            }
            std::printf("  (c) β=%-4g c=%-3g（β c=%g）tfMulR H_c(u) の組 %llu 前提違反 %llu 不一致 %llu 出力の尾が H_{c1}(u') の外 %llu"
                        "  (A-B) の最大 2^%.1f\n", beta, c, beta * c, (unsigned long long)made, (unsigned long long)viol,
                        (unsigned long long)inex, (unsigned long long)tailbad, lg2(dmax));
            check(made > (uint64_t)n / 2, "H_c(u) の組が作れない");
            check(viol == 0 && inex == 0 && dmax == 0 && tailbad == 0, "tfMulR の補題 R（H_c(u) => 前提・出力 H_{c1}(u')）が崩れた");
        }
    }
}

// --- [6] 入口 split_f64 の 3 本組（尾の絶対値の和が ulp(hi) を超える形）------------------------------
// 仮数 S = A 2^30 + 2^29 + c（A は上位 23 bit，1 <= c <= 127）の x = S 2^P は上位 23 bit が切り上がり，
// (hi, mid, lo) = ((A+1) 2^(P+30), -2^(P+29), c 2^P) になる．A + 1 < 2^23 なら ulp(hi) = 2^(P+29) なので
// |mid| + |lo| = ulp(hi) + c 2^P > ulp(hi)，一方 u = 2^(P+30) に対しては < u．A = 2^23 - 1 なら hi = 2^(P+53) で ulp(hi) = u．
static void part6(int n, uint64_t seed) {
    std::printf("[6] 入口 split_f64 の 3 本組（仮数の下位 30 bit が 2^29 + c の形．%d 個）\n", n);
    // 例：x = 2^-2 + 2^-25 + 127·2^-54（S = 2^52 + 2^29 + 127，P = -54）
    {
        const double x = 0x1p-2 + 0x1p-25 + 127 * 0x1p-54;
        Dst D = emu(false, nullptr); load_f64(D, x);
        const double uh = std::ldexp(1.0, fexp(D.r[0]) - 24), u = 0x1p-24;
        const double tail = std::fabs((double)D.r[1]) + std::fabs((double)D.r[2]);
        std::printf("  例 x = 2^-2 + 2^-25 + 127*2^-54 -> (%a, %a, %a)  尾 %a  ulp(hi) %a  u %a\n", D.r[0], D.r[1], D.r[2],
                    tail, uh, u);
        check(D.r[0] == 0x1p-2f + 0x1p-24f && D.r[1] == -0x1p-25f && D.r[2] == 127 * 0x1p-54f,
              "例の 3 本組が (2^-2 + 2^-24, -2^-25, 127*2^-54) でない");
        check(tail > uh, "例の尾の絶対値の和が ulp(hi) を超えない（ulp(hi) の格子との違いを示す例になっていない）");
        EntryH eh; entry_hcheck(x, D, eh);
        check(eh.out == 0, "例の 3 本組が H_1(u)（u = 2^-24）を満たさない");
    }
    // 同じ形を，符号・指数・上位の仮数（A の切り上がりで 2^23 になる端 A = 2^23 - 1 を含む）・c を振って作る
    std::mt19937_64 g(seed + 666);
    std::vector<double> xs;
    for (int k = 0; k < n; k++) {
        uint64_t A = (1ull << 22) | (g() & ((1ull << 22) - 1));
        if (k % 7 == 0) A = (1ull << 23) - 1;
        const uint64_t c = (k % 5 == 0) ? 127 : (k % 5 == 1) ? 1 : 1 + g() % 127;
        const uint64_t S = (A << 30) | (0x20000000ull + c);
        double x = std::ldexp((double)S, -53 - 1 - (int)(g() % 60));   // |x| < 2^-1（5 形式の上限 R はどれも 1/2 以上）
        if (g() & 1) x = -x;
        xs.push_back(x);
    }
    xs.push_back(0x1p-2 + 0x1p-25 + 127 * 0x1p-54);
    xs.push_back(-(0x1p-2 + 0x1p-25 + 127 * 0x1p-54));
    for (const auto& c : slice_cfgs()) slicer_verdict(c, xs, 0, seed);
    // 尾崎 II の桁作りの入口（|T| <= 2^20）にも同じ形を 2^20 倍して当てる
    {
        FtsStat st; EntryH eh; uint64_t trk = 0;
        for (double x0 : xs) {
            const double x = std::ldexp(x0, 20);
            Dst D = emu(false, &st); load_f64(D, x);
            entry_hcheck(x, D, eh);
            Fix Tx; Tx.add_d(x);
            // 5 桁ぶん回す．4 桁の桁と残差はこの先頭 4 段と同じなので，4 桁の検査も兼ねる
            for (int i = 0; i < 5; i++) {
                if (i > 0) { tfMulP2(D, 1048576.0f); Tx.mul_small(1048576); }
                D.Cpy(0, 7); D.Rnd(7); D.Sub(0, 7, 0);
                Tx.add_i((int64_t)D.r[7], -1);
                if (i + 1 < 5 && !(Tx - fix_tri(D)).zero()) trk++;
            }
        }
        std::printf("  尾崎 II ND=4,5    呼び出し %llu 前提違反 %llu 不一致 %llu  残差が厳密でない段 %llu  入口 %llu 個のうち "
                    "H_1(u) の外 %llu 個（尾が ulp(hi) を超えた組 %llu 個）\n",
                    (unsigned long long)st.calls, (unsigned long long)st.viol, (unsigned long long)st.inexact,
                    (unsigned long long)trk, (unsigned long long)eh.n, (unsigned long long)eh.out, (unsigned long long)eh.over);
        check(st.viol == 0 && st.inexact == 0 && trk == 0, "尾崎 II の桁作りで [6] の入口の形に前提違反か誤差が出た");
        check(eh.n > 0 && eh.out == 0 && eh.over > 0, "尾崎 II の入口で H_1(u) が崩れたか，尾が ulp(hi) を超える組が無い");
    }
}

int main(int argc, char** argv) {
    int n = 20000; uint64_t seed = 1;
    // 入力数は int へ縮める前に上限を見る（[4]・[5] は n*5 本を回すので INT_MAX/5 まで．超える値が小さい n に化けたり
    // n*5 が溢れたりしない）．seed は 0 以上の整数で，負号はどこにあっても拒む（strtoull は "-1" を黙って 2^64-1 にする）
    if (argc > 1) {
        long long v = 0;   // 読み方は src/common/oz_env.hpp
        if (!oz_env::to_i64(argv[1], v) || v < 4 || v > INT_MAX / 5) {
            std::fprintf(stderr, "ERROR: 入力数 %s は 4--%d の整数ではない\n", argv[1], INT_MAX / 5); return 2; }
        n = (int)v;
    }
    if (argc > 2) {
        unsigned long long v = 0;   // 負号はどこにあっても拒む（oz_env::to_u64）
        if (!oz_env::to_u64(argv[2], v)) {
            std::fprintf(stderr, "ERROR: seed %s は 0 以上の整数ではない\n", argv[2]); return 2; }
        seed = v;
    }
    if (argc > 3) { std::fprintf(stderr, "ERROR: 引数は 入力数 seed の 2 個まで\n"); return 2; }
    g_verbose = oz_sim::env_flag01("TFMUL_VERBOSE");
    g_cfg1 = oz_sim::scheme1_configs("tfmul_check");
    oz_sim::print_scheme1(g_cfg1);
    part0();
    part1(n, seed);
    part2(n, seed);
    part3(n / 4, seed);
    part4(n * 5, seed);
    part5(n * 5, seed);
    part6(n, seed);
    std::printf("TFMUL-CHECK n=%d seed=%llu %s\n", n, (unsigned long long)seed, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
