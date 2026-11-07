// awe_plan.cpp -- 尾崎 II：法ごとに桁の運び方（エンコーディング）を選び，段の時間のモデルを最小にする法の集合を探す（Phase A）
// ---------------------------------------------------------------------------
// リポジトリ直下で（コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     make -C src awe_plan              -> src/build/awe_plan
//     ./src/build/awe_plan [route ...]      route = bf16 fp16 tf32 bfp8 fp8 bfp4 bfp2（既定は全部）
//     ./src/build/awe_plan coef route ...   較正後の係数（経路ごと・3 サイズ）
//     ./src/build/awe_plan emit-coef        本番が法の集合を選ぶ係数の表 src/ozaki2/ozaki2_awe_coef.inc の中身を書き出す
//                                           （段の時間のモデルで法を選ぶ 5 経路の coef を awe の番号順に．較正の基準か実測
//                                           （MEAS）を直したら ./src/build/awe_plan emit-coef > src/ozaki2/ozaki2_awe_coef.inc）
//     ./src/build/awe_plan verify-coef      較正後の係数が本番の表 OZ2_AWE_COEF（ozaki2_awe_coef.inc）と表示の桁（%.5g）で
//                                           一致するかを確かめる（外れれば終了コード 1．scripts/selftest.sh の hostcheck）
//     ./src/build/awe_plan table route NUM [tail [m,m,...]]   表による選び方の模擬と最適化の比較（main の table）
// 無い経路の名前・余分な引数・整数でない NUM などは終了コード 2 で止める．法の運び方を再現できない計画があれば終了コード 1．
// 較正（factors_at）に使う計画が無効（法が取れない・運び方を再現できない）か，その段の時間が有限の正の数でなければ，
// 係数を作らずに終了コード 1 で止める（0 や無限大の倍率で本番の係数の表 emit-coef・照合 verify-coef を作らないように）．
// デバイスは使わない．docs/awe_moduli.md の Phase A の数値はこれの出力である．
//
// 着想は Hayashi–Mukunoki–Hoshino–Katagiri, "AWE: Adaptive Weight Encoding for Exact Integer Matrix Products
// with Fewer GEMMs on FP4 Tensor Cores"（arXiv 2609.24519）：桁（limb）の重みを自由に取り，格納する平面を limb の
// 線形結合（a0, a1, a0+a1 など，和が形式に収まるとき）にし，再構成の係数を正確な線形代数で求めて総当たりで確かめる．
// 法ごとに積の本数が最小になる運び方を選び，法の集合は CRT の窓のもとで整数計画で選ぶ．本実装に当てはめると
// 法ごとに次の運び方が候補になる（桁の集合 T は経路で決まる：桁 32 の 4 経路は [-32,32]，FP8 は疎な e4m3 の
// S_32，BFP4_b は [-7,7]，BFP2_b は {-1,0,1}）．以下の「既存」は比較の基準にする運び方の組（main の変種
// existing-modes．Phase A の出発点のカーネルが持っていたもの）を指す．本番のカーネルが持つ運び方は
// ozaki2_split.h の (A)〜(F) にある．
//   S   1 桁（SINGLE）：T が法 m の完全剰余系を含む（d=32 なら m <= 65）．積 1 本・桁行列 1 枚
//   *   平方数 q^2（SQ）／ s 疎な桁の平方数（SQS．FP8）：積 3 本・桁行列 2 枚（既存）
//   k   Karatsuba（切り上げ．既存のカーネル）を基数 B で：h = sgn(r)ceil(|r|/B)，l = r - B h，s = l + h．
//       積 3 本・桁行列 3 枚．既存の方式（基数 16 固定）を B <= d+1 へ一般化すると d=32 で m <= 2113
//   f   Karatsuba（切り下げ）h = sgn(r)floor((|r|+d)/B)，B = d+2：行の区間が隙間なく続き m <= 2d^2+4d+1
//       （d=32 で 2177）．カーネルは切り上げの定数 (B-1)/2 を法ごとの定数にするだけ
//   h   六角形の Karatsuba（HEX．自由な重み w）：r ≡ l + w h (mod m)，|l|,|h|,|l+h| <= d．六角形の点
//       3d^2+3d+1 個の像が Z_m を覆う w を総当たりで探す（d=32 で m <= 2860 は全部，3169 まで飛び飛び）．
//       積と桁行列は k と同じだが，桁の作り方（最近点の探索）はカーネルに無いので【仮定の費用】で見積もる
//   w   平衡 2 桁（WIDE）：積 4 本・桁行列 2 枚（既存）
//   b   平衡 n 桁（BAL-n，基数 2d+1）：積 n^2 本（m が q^n を割れば n(n+1)/2 本）・桁行列 n 枚．BFP2_b の既存は
//       n=5 だけで，n=1..4 を混ぜる変種は法ごとに n を受けるカーネル（ozaki2/kernels/modslice_digit.cpp）を前提にする．
//       BFP4_b は n=3（m <= 3375）
// 選ぶのは「互いに素な法の集合と法ごとの運び方」で，目的は下の段の時間のモデルの和，制約は
//   ・CRT の窓：Σ log2 m >= 2b+1+log2 K+eps（b=54．ozaki2_choose_moduli_w と同じ式），超過 <= tol（既定 0.25 bit．
//     超過はそのまま要素ごとの精度の損失なので OZ2_TAIL_EXCESS_TOL と同じ上限を置く）
//   ・法の本数 <= MAXS（既定 24．recon_crt.cpp・modslice_*.cpp の固定配列）
//   ・無誤差：GEMM の出力 1 本（連結 g 本の和）の上界 g*K*d_a*d_b + max m <= 2^24，合成の総和（先に法へ畳む方式）
//     (Σ|c_j|)(m/2+2) < 2^22（係数は対称代表元で数える），スライサの剰余 |r| の上界は oz2_residue の解析値
//   ・DRAM（14080^3）：桁行列 + GEMM の出力と剰余の平面が空き 31.0 GB に収まる（TF32 は組分けの段数を数える）
// 探索は「費用が同じ候補の組」ごとの本数のベクトルを費用の下界の安い順に列挙し，互いに素な法を窓に収めて選べるかを
// 割り当ての探索で確かめて，見つかった集合を評価する．残りのベクトルの下界が最良以上になるまで続ける（分枝限定．モデルの上で
// 最適．ただし組ごとに大きい法の上位 400 本に絞り，法ごとの運び方を 1 つに決めた候補の中で，最適を示せたときで，示せなかった
// 行（探索の打ち切りなど）には * を付ける．table の optimizer の行も同じ．詳しくは Klass の前）．ホストの src/ozaki2/ozaki2_awe.h も同じ形の探索だが，
// 組ごとの本数を貪欲の見積りで打ち切る（上界ではない）ので，モデルの上の最適とは限らない．
//
// 段の時間のモデル［ms］（係数は measure_logs の実測から当てはめたもの．docs/awe_moduli.md の §2）
//   GEMM    = gp*行列積 + go*出力
//   合成    = co*出力 + cs*法の本数
//   スライス = s0 + sop*Σ_法(19 + 桁の作り方の演算数) + spl*桁行列（FP16 は桁行列ごとに HalfBits，TF32 は 4 B の書き出しと組分け）
//   再構成  = max(R0, ra + rb*S)（法 14 本までは FP64 の組み立てに隠れて平ら，それを超えると法 1 本ごとに rb）
//   スケール = 定数
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>
#include <chrono>
#include "oz_sim.h"

// ---------------------------------------------------------------------------
// 経路．ホストの選び方の経路 hostr は本番の経路の表（ozaki2_split.h の OZ2_ROUTES）から引き，
// 桁の上限・疎な桁の上限・法の上限は hostr の Oz2Route の値をそのまま使う（ここに別の値を持つと本番と違う候補を見る）
struct Route {
    const char* name;
    Oz2Route cur;       // 段の時間の実測（MEAS）を取ったときの選び方の経路（較正の基準．plan_current）
    int balmax;         // 平衡 n 桁の n の上限（0 = 使わない）
    Oz2Route hostr;     // ホストの選び方の経路（ozaki2_choose_moduli_route．awe なら段の時間のモデルで選ぶ）
    int D;              // 稠密な桁の上限（hostr.digit_bound）
    int smax;           // 疎な桁の上限（hostr.sparse_max．FP8 は 32，他は 0）
    int cap;            // 法の上限（hostr.mod_cap）
    double bytes;       // 桁行列の 1 要素のバイト数（DRAM の見積り．段の時間のモデルで選ぶ経路は ozaki2_awe.h の
                        // OZ2_AWE_SLICE_TILE（タイル 1 枚のバイト数）から）
};
// bytes_noawe は段の時間のモデルを使わない経路（TF32 は FP32 で 4 B，FP8 は 1 B）の桁行列の 1 要素のバイト数
static Route mk_route(const char* name, double bytes_noawe, Oz2Route cur, int balmax) {
    const Oz2RouteRow* x = oz_sim::route2(name);
    if (!x) { fprintf(stderr, "ERROR: 経路 %s が本番の経路の表（ozaki2_split.h の OZ2_ROUTES）に無い\n", name); std::exit(2); }
    return {name, cur, balmax, x->r, x->r.digit_bound, x->r.sparse_max, x->r.mod_cap,
            x->r.awe ? OZ2_AWE_SLICE_TILE[x->r.awe - 1] / 1024.0 : bytes_noawe};
}
static const Route ROUTES[] = {
    mk_route("bf16", 0.0, OZ2_ROUTE_D32, 0), mk_route("fp16", 0.0, OZ2_ROUTE_D32, 0),
    mk_route("tf32", 4.0, OZ2_ROUTE_D32, 0), mk_route("bfp8", 0.0, OZ2_ROUTE_D32, 0),
    mk_route("fp8", 1.0, OZ2_ROUTE_FP8, 0),   mk_route("bfp4", 0.0, OZ2_ROUTE_BFP4, 3),
    mk_route("bfp2", 0.0, OZ2_ROUTE_BFP2, 5),
};
// 1 桁の法の合成段を再構成へ畳むときに再構成の CRT に乗る ModRound 1 回の割合は本番のモデルと同じ OZ2_AWE_ONE_RECON
// （ozaki2_awe.h）．DRAM の見積り（14080 だけ見る）は本番と同じ関数（ozaki2_split.h の oz2_dram_one_group・oz2_dram_groups と
// 合成段の平面の oz2_compose_plan）で，空きはモデルの仮定 OZ2_AWE_DRAM_FREE（14080^3 の実測の 31.0 GB）

// ---------------------------------------------------------------------------
// 段の時間の係数（サイズごと・経路ごと）．値の出所は docs/awe_moduli.md の §2．
// ここの値の出どころ（2026-10-23 に辿った結果）：
//   ・14080 の scale・s0・sop・spl・go は tools/study/fp8_sparse_plan.cpp の M14（72333d3．docs/fp8_sparse_digits.md の FP8 の
//     段の当てはめ）と同じ値．
//   ・それ以外（14080 の co・cs・R0・ra・rb，7040・3520 の全部，下の gt の行列積 1 本あたりの値，plx・grp・xsqs）は 7c06f82
//     （2026-09-25．この道具を入れたコミット）で当てはめた値で，当てはめに使ったログの組と手順は残っていない（作業の表
//     measure_logs/tmp_awe/rows.json が当時のログの段の時間を集めたもの）．再現できない記録である．
//   どの値も段ごとの較正（下の MEAS と factors_at）の前の値で，較正が段の和を MEAS に合わせるので，本番の係数（emit-coef）に
//   効くのは段の中の重み（行列積・出力・法の本数・桁行列の比）である．値を変えると emit-coef の出力が変わる（verify-coef が止める）
struct Coef {
    double scale, s0, sop, spl, gp, go, co, cs, R0, ra, rb;
    double plx;     // 桁行列 1 枚あたりのスライス段の上乗せ（FP16 の HalfBits，TF32 の 4 B 書き出し）
    double grp;     // TF32 の組分け 1 組あたりの上乗せ（14080 だけ）
    double xsqs;    // FP8 の SQS の手直しの演算数相当
};
static Coef coef_raw(const char* r, int K) {
    Coef c{};
    const bool k14 = K >= 14080, k7 = !k14 && K >= 7040;
    // 共通（BF16・BFP8 の段の実測と fp8_sparse_plan の当てはめ）
    if (k14)     { c.scale = 11.4; c.s0 = 33.2; c.sop = 0.5725; c.spl = 0.127; c.go = 1.3;  c.co = 3.6;  c.cs = 2.5;
                   c.R0 = 276.0; c.ra = 63.0; c.rb = 15.0; }
    else if (k7) { c.scale = 2.49; c.s0 = 8.3;  c.sop = 0.1431; c.spl = 0.032; c.go = 0.86; c.co = 0.92; c.cs = 0.05;
                   c.R0 = 68.2;  c.ra = 1.36; c.rb = 4.6; }
    else         { c.scale = 1.29; c.s0 = 2.24; c.sop = 0.03578; c.spl = 0.0079; c.go = 0.32; c.co = 0.23; c.cs = 0.02;
                   c.R0 = 17.04; c.ra = 0.48; c.rb = 1.15; }
    // 行列積 1 本あたり（出力の分を除く）：14080 と 3520 は実測の GEMM 段 − go*出力 を行列積の本数で割ったもの，
    // 7040 は BF16・FP8 の実測，他の経路は 14080 の値を BF16 の 7040/14080 の比（0.0897）で縮めた推定
    struct G { const char* r; double g14, g7, g35; };
    static const G gt[] = {{"bf16", 18.50, 1.66, 0.180}, {"fp16", 18.32, 1.643, 0.181}, {"tf32", 29.51, 2.647, 0.351},
                           {"bfp8", 16.02, 1.437, 0.165}, {"fp8", 15.26, 1.50, 0.140}, {"bfp4", 14.30, 1.283, 0.151},
                           {"bfp2", 14.01, 1.257, 0.141}};
    for (auto& g : gt) if (!strcmp(g.r, r)) c.gp = k14 ? g.g14 : k7 ? g.g7 : g.g35;
    // 桁行列ごとの上乗せ
    if (!strcmp(r, "fp16")) c.plx = k14 ? 8.90 : k7 ? 2.23 : 0.558;          // HalfBits（BF16 との差 / 桁行列 22 枚）
    if (!strcmp(r, "tf32")) { c.plx = k14 ? 7.37 : k7 ? 1.84 : 0.461; c.grp = k14 ? 228.0 : 0; }
    c.xsqs = 6.6;
    return c;
}
static double recon_ms(const Coef& c, int S) { return std::max(c.R0, c.ra + c.rb * S); }

// ---------------------------------------------------------------------------
// 桁の集合
static bool in_digits(const Route& R, long x) {
    if (R.smax) return oz2_in_sparse(x, R.smax);
    return std::labs(x) <= R.D;
}
// スライサの最後の ModRound が返しうる |r| の上界はホストと同じ oz2_resid_bound（ozaki2_moduli.h．3 桁と 4 桁の
// 大きいほうで，5 桁も 4 桁と同じ形になる）を使う．

// ---------------------------------------------------------------------------
// 運び方
struct Term { int a, b, key; long coef; };     // 桁行列 a × b，合成の係数（法で畳んだ対称代表元）
struct Enc {
    int m = 0; char kind = '?'; int B = 0, n = 0;   // B：基数・重み（Karatsuba の切り上げ・切り下げの定数は kar_check に渡すだけで持たない），n：桁数
    int nprod = 0, nplane = 0, ops = 0;
    std::vector<int> pb;             // 桁行列ごとの桁の絶対値の上限
    std::vector<Term> terms;
    double csum = 0;                 // 合成の係数の絶対値の和（対称代表元）
    int maxprod = 0;                 // 桁積 1 本の上界 d_a*d_b の最大
};
static long symm(long v, long m) { v %= m; if (v < 0) v += m; if (v > m / 2) v -= m; return v; }

// 位 key ごとの本数から，連結 g 本までで出力の本数
static int nout_of(const Enc& e, int g) {
    std::map<int, int> cnt; for (auto& t : e.terms) cnt[t.key]++;
    int n = 0; for (auto& kv : cnt) n += (kv.second + g - 1) / g;
    return n;
}
// 連結してよい本数：同じ位の積の上界 d_a*d_b の最大で決める（法ごと）
static int maxg_of(const Enc& e, double K, int mmax) {
    int pmax = 1; std::map<int, int> cnt; for (auto& t : e.terms) cnt[t.key]++;
    for (auto& t : e.terms) if (cnt[t.key] > 1) pmax = std::max(pmax, e.pb[t.a] * e.pb[t.b]);
    return std::max(1, (int)std::floor((16777216.0 - mmax) / (K * pmax)));
}

static void finish(Enc& e) {
    e.nprod = (int)e.terms.size(); e.nplane = (int)e.pb.size();
    e.csum = 0; e.maxprod = 0;
    std::map<int, long> kc;
    for (auto& t : e.terms) { kc[t.key] = t.coef; e.maxprod = std::max(e.maxprod, e.pb[t.a] * e.pb[t.b]); }
    for (auto& kv : kc) e.csum += std::labs(kv.second);
}

// (S) 1 桁
static bool enc_single(const Route& R, int m, Enc& e) {
    const long Rr = oz2_resid_bound(m);
    int dmax = 0;
    for (long r = -Rr; r <= Rr; r++) {
        long d = r;
        if (!in_digits(R, d)) { d = r - (r > 0 ? m : -m); if (!in_digits(R, d)) return false; }
        dmax = std::max<int>(dmax, (int)std::labs(d));
    }
    e = Enc{}; e.m = m; e.kind = 'S'; e.pb = {dmax}; e.terms = {{0, 0, 0, 1}};
    e.ops = R.smax ? 4 : 0;   // 既存の組に無い方式（カーネルでは ozaki2_split.h の (F) OZ2_SPLIT_ONE にあたる）
    finish(e); return true;
}
// (*) 平方数（稠密）／(s) 疎な桁の平方数
static bool enc_square(const Route& R, int m, Enc& e, const Coef& C) {
    const int q = oz2_is_square(m); if (!q) return false;
    const long Rr = oz2_resid_bound(m);
    int dl = 0, dh = 0; bool ok = true, sqs = false;
    for (long r = -Rr; r <= Rr && ok; r++) {
        long h = (long)std::floor((double)r / q + 0.5), l = r - (long)q * h;   // 同点はどちらでも上限は同じ
        long h2 = (long)std::ceil((double)r / q - 0.5), l2 = r - (long)q * h2;
        for (auto [ll, hh] : {std::pair<long,long>{l, h}, {l2, h2}}) {
            if (std::labs(ll) <= R.D && std::labs(hh) <= R.D) { dl = std::max<int>(dl, std::labs(ll)); dh = std::max<int>(dh, std::labs(hh)); }
            else ok = false;
        }
    }
    if (!ok && R.smax && (q & 1) && oz2_sparse_covers(q, R.smax)) {
        ok = true; sqs = true; dl = dh = 0;
        for (long r = -Rr; r <= Rr; r++) {
            long l, h; oz2_sqs_split(r, q, l, h);
            if (!oz2_in_sparse(l, R.smax) || !oz2_in_sparse(h, R.smax)) { ok = false; break; }
            dl = std::max<int>(dl, std::labs(l)); dh = std::max<int>(dh, std::labs(h));
        }
    }
    if (!ok) return false;
    e = Enc{}; e.m = m; e.kind = sqs ? 's' : '*'; e.B = q; e.pb = {dl, dh};
    e.terms = {{0, 0, 0, 1}, {0, 1, 1, symm(q, m)}, {1, 0, 1, symm(q, m)}};
    e.ops = 6 + (sqs ? (int)std::lround(C.xsqs) : 0);
    finish(e); return true;
}
// (k)(f) 整数の Karatsuba：h = sgn(r) floor((|r| + t)/B)，l = r - B h，s = l + h（t = B-1 が切り上げ＝既存）
static bool kar_check(const Route& R, int m, int B, int t, int& dl, int& dh, int& ds) {
    const long Rr = oz2_resid_bound(m);
    dl = dh = ds = 0;
    for (long r = -Rr; r <= Rr; r++) {
        const long a = std::labs(r), sg = r > 0 ? 1 : (r < 0 ? -1 : 0);
        const long h = sg * ((a + t) / B), l = r - (long)B * h, s = l + h;
        if (!in_digits(R, l) || !in_digits(R, h) || !in_digits(R, s)) return false;
        if (R.smax && (std::labs(l) > R.D || std::labs(h) > R.D || std::labs(s) > R.D)) return false;   // 疎な桁は使わない
        dl = std::max<int>(dl, std::labs(l)); dh = std::max<int>(dh, std::labs(h)); ds = std::max<int>(ds, std::labs(s));
    }
    return true;
}
// mode 0：既存（基数 16 の切り上げだけ），1：切り上げで B <= D+1，2：加えて切り下げ B = D+2
static bool enc_kar(const Route& R, int m, Enc& e, int mode) {
    const int D = R.D;
    // 既存のカーネル（切り上げ t = B-1）で B = D+1 から下へ，だめなら切り下げ B = D+2, t = D
    struct Cand { int B, t; char kind; };
    std::vector<Cand> cs;
    if (mode == 0) { if (D >= 16) cs.push_back({16, 15, 'k'}); }
    else for (int B = D + 1; B >= 2; B--) cs.push_back({B, B - 1, 'k'});
    if (mode >= 2) cs.push_back({D + 2, D, 'f'});
    for (auto c : cs) {
        // B が小さすぎて m に届かない切り上げの候補は飛ばす（B は降順なので残りの切り上げも届かないが，
        // break すると後ろの切り下げ B = D+2 を調べずに終わり，切り下げだけが運べる法（d=32 で 2114〜2177）を落とす）
        if (2L * c.B * D + 1 < m && c.kind == 'k') continue;
        int dl, dh, ds;
        if (!kar_check(R, m, c.B, c.t, dl, dh, ds)) continue;
        e = Enc{}; e.m = m; e.kind = c.kind; e.B = c.B; e.pb = {dl, dh, ds};
        // r_a r_b = (1-B) LL + (B^2-B) HH + B SS
        e.terms = {{0, 0, 0, symm(1 - c.B, m)}, {1, 1, 2, symm((long)c.B * c.B - c.B, m)}, {2, 2, 4, symm(c.B, m)}};
        e.ops = 12;
        finish(e); return true;
    }
    return false;
}
// (h) 六角形の Karatsuba（法 m での自由な重み w）．覆う w のうち合成の係数の和が最小のもの
static bool enc_hex(const Route& R, int m, Enc& e, int xhex) {
    if (R.smax) return false;
    const int D = R.D;
    if (m > 3 * D * D + 3 * D + 1) return false;
    if (m <= 2 * D * D + 4 * D + 1) return false;      // 整数の Karatsuba（k・f）で足りる範囲は見ない
    // (D, m) ごとに 1 度だけ数える（サイズ・変種をまたいで同じ結果）
    static std::map<std::pair<int,int>, int> cache;
    auto key = std::make_pair(D, m);
    if (auto it = cache.find(key); it != cache.end()) {
        if (it->second <= 0) return false;
        const int bw = it->second;
        e = Enc{}; e.m = m; e.kind = 'h'; e.B = bw; e.pb = {D, D, D};
        e.terms = {{0, 0, 0, symm(1 - bw, m)}, {1, 1, 2, symm((long)bw * bw - bw, m)}, {2, 2, 4, symm(bw, m)}};
        e.ops = xhex; finish(e); return true;   // 仮定の費用（カーネルに無い桁の作り方）
    }
    std::vector<std::pair<int,int>> H;
    for (int h = -D; h <= D; h++) for (int l = -D; l <= D; l++) if (std::abs(l + h) <= D) H.push_back({l, h});
    std::vector<int> mark(m, -1);
    long bestc = -1; int bw = 0;
    for (int w = 1; w < m; w++) {
        int cnt = 0;
        for (auto& p : H) { long v = ((long)p.first + (long)w * p.second) % m; if (v < 0) v += m;
                            if (mark[v] != w) { mark[v] = w; if (++cnt == m) break; } }
        if (cnt < m) continue;
        const long c = std::labs(symm(1 - w, m)) + std::labs(symm((long)w * w - w, m)) + std::labs(symm(w, m));
        if (bestc < 0 || c < bestc) { bestc = c; bw = w; }
    }
    cache[key] = bestc < 0 ? 0 : bw;
    if (bestc < 0) return false;
    e = Enc{}; e.m = m; e.kind = 'h'; e.B = bw; e.pb = {D, D, D};
    e.terms = {{0, 0, 0, symm(1 - bw, m)}, {1, 1, 2, symm((long)bw * bw - bw, m)}, {2, 2, 4, symm(bw, m)}};
    e.ops = xhex;   // 仮定の費用（カーネルに無い桁の作り方）
    finish(e); return true;
}
// (w) 平衡 2 桁（WIDE）
static bool enc_wide(const Route& R, int m, Enc& e) {
    if (oz2_is_square(m)) return false;
    const int q = oz2_digit_base_rnd(m);
    const long Rr = oz2_resid_bound(m);
    int dl = 0, dh = 0;
    for (long r = -Rr; r <= Rr; r++) {
        const long h1 = (long)std::floor((double)r / q + 0.5), h2 = (long)std::ceil((double)r / q - 0.5);   // 同点の両向き
        for (long h : {h1, h2}) {
            const long l = r - (long)q * h;
            if (std::labs(l) > R.D || std::labs(h) > R.D) return false;
            dl = std::max<int>(dl, std::labs(l)); dh = std::max<int>(dh, std::labs(h));
        }
    }
    e = Enc{}; e.m = m; e.kind = 'w'; e.B = q; e.pb = {dl, dh};
    e.terms = {{0, 0, 0, 1}, {0, 1, 1, symm(q, m)}, {1, 0, 1, symm(q, m)}, {1, 1, 2, symm((long)q * q, m)}};
    e.ops = 6; finish(e); return true;
}
// (b) 平衡 n 桁（基数 q = 2D+1）
static bool enc_bal(const Route& R, int m, int n, Enc& e) {
    const long q = 2L * R.D + 1; long qn = 1; for (int i = 0; i < n; i++) qn *= q;
    if (m > qn) return false;
    const long Rr = oz2_resid_bound(m);
    if (Rr > (qn - 1) / 2) return false;
    const bool drop = (qn % m == 0);
    e = Enc{}; e.m = m; e.kind = 'b'; e.B = (int)q; e.n = n; e.pb.assign(n, R.D);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        if (drop && i + j >= n) continue;
        long c = 1; for (int k = 0; k < i + j; k++) c = (c * q) % m;
        e.terms.push_back({i, j, i + j, symm(c, m)});
    }
    e.ops = 7 * (n - 1);
    finish(e); return true;
}

// ---------------------------------------------------------------------------
// 1 本の法の費用（法の集合に依らない部分）．maxg は法ごと（pmg）か経路一律（Dmax^2 で決める）か
struct Opt {
    bool single = true, kar = true, karf = true, hex = false, bal = true, wide = true, sq = true;
    bool bal5 = false;       // 平衡 n 桁は既存の n=5 だけ（BFP2_b の既存の組）
    bool pmg = false;        // 法ごとの連結の上限（FP8 は既存の組でも法ごと．他の経路はこれを立てたときだけ）
    bool fuse = false;       // 1 桁の法の合成段を再構成へ畳む（合成の出力を読まない）
    int xhex = 22;           // 六角形の桁の作り方の演算数相当（仮定）
    int topn = 400;          // 費用が同じ組の中で残す候補の数（大きい法から．表の模擬 plan_table・efficiency_table は全部を使う）
    double tol = 0.25;
    int maxs = OZ2_MAXS;        // 法の本数の上限（カーネルの固定配列．ozaki2/kernels/oz2_limits.h）
    int b = OZ2_B_DEFAULT;      // 幅 b の上限の既定（ozaki2_moduli.h）
};
// cost・nout は法の上限 cap を集合の最大の法とみなした値（連結の本数 g が最小で出力が最多．実際の集合の評価の上界），
// cost_lo・nout_lo はその法自身を集合の最大の法とみなした値（g が最大で出力が最少．集合の最大の法はその法以上なので下界）．
// 運び方 e は cost で選ぶ（法ごとに 1 つに決める．集合の最大の法で運び方を選び直すことはしない）
struct ModCost { Enc e; double cost; int nout; double bits; double cost_lo = 0; int nout_lo = 0; };
static double mod_cost(const Coef& C, const Enc& e, int nout, const Opt& o) {
    double gem = C.gp * e.nprod + C.go * nout;
    double com = C.co * nout + C.cs;
    if (o.fuse && e.kind == 'S') com = 0;     // 仮定：再構成が GEMM の出力を直接読んで畳む
    const double sli = C.sop * (19 + e.ops) + (C.spl + C.plx) * e.nplane;
    return gem + com + sli;
}
static int route_maxg(const Route& R, const Enc& e, double K, int mmax, const Opt& o) {
    if (o.pmg || R.smax) return maxg_of(e, K, mmax);
    const int D = R.D;
    return std::max(1, (int)std::floor((16777216.0 - mmax) / (K * (double)D * D)));
}

// 法ごとの候補（最安の運び方）
static std::vector<ModCost> candidates(const Route& R, const Coef& C, int K, const Opt& o, int smax_eff) {
    Route Rx = R; Rx.smax = smax_eff;
    int cap = R.cap;
    if (R.smax && !smax_eff) cap = std::min(cap, (2 * R.D + 1) * (2 * R.D + 1));
    std::vector<ModCost> out;
    for (int m = 2; m <= cap; m++) {
        std::vector<Enc> es; Enc e;
        if (o.single && enc_single(Rx, m, e)) es.push_back(e);
        if (o.sq && enc_square(Rx, m, e, C)) es.push_back(e);
        if (enc_kar(Rx, m, e, o.karf ? 2 : o.kar ? 1 : 0)) es.push_back(e);
        if (o.wide && enc_wide(Rx, m, e)) es.push_back(e);
        if (o.hex && enc_hex(Rx, m, e, o.xhex)) es.push_back(e);
        if (o.bal && R.balmax) {
            if (o.bal5) { if (enc_bal(Rx, m, 5, e)) es.push_back(e); }
            else for (int n = 1; n <= R.balmax; n++) if (enc_bal(Rx, m, n, e)) { es.push_back(e); break; }
        }
        double best = 1e30; ModCost bm{};
        for (auto& x : es) {
            // 無誤差：桁積 1 本 K*d_a*d_b + m <= 2^24
            if ((double)K * x.maxprod + cap > 16777216.0) continue;
            // 合成の総和（先に法へ畳む）
            if (x.csum * (m / 2.0 + 2.0) >= 4194304.0) continue;
            const int g = route_maxg(Rx, x, K, cap, o), no = nout_of(x, g);
            const double c = mod_cost(C, x, no, o);
            if (c < best) {
                const int nolo = nout_of(x, route_maxg(Rx, x, K, m, o));
                best = c; bm = {x, c, no, std::log2((double)m), mod_cost(C, x, nolo, o), nolo};
            }
        }
        if (best < 1e29) out.push_back(bm);
    }
    // 費用が同じ組（運び方・行列積・出力・桁行列・演算数が同じ）の中では，大きい法ほど bit あたりが安い．
    // 各組の上位 o.topn 本だけを残して探索を軽くする（末尾の窓合わせに要る細かさはこれで足りる）．
    std::map<std::string, int> seen;
    std::sort(out.begin(), out.end(), [](const ModCost& a, const ModCost& b) { return a.e.m > b.e.m; });
    std::vector<ModCost> kept;
    for (auto& x : out) {
        char k[96]; snprintf(k, sizeof k, "%c/%d/%d/%d/%d", x.e.kind, x.e.nprod, x.nout, x.e.nplane, x.e.ops);
        if (++seen[k] <= o.topn) kept.push_back(x);
    }
    return kept;
}

// ---------------------------------------------------------------------------
// 法の集合の評価（全体）
struct Plan {
    std::vector<ModCost> mods; double bits = 0, excess = 0; int S = 0, nprod = 0, nout = 0, nplane = 0, ops = 0;
    double t[6] = {0}; int groups = 1; bool dram_ok = true, ok = false;
    bool valid = true;       // 法の運び方をすべて再現できたか（false の計画は評価が意味を持たない．plan_from）
};
static double need_bits(int K, int b) { return 2.0 * b + 1 + std::log2((double)K) + OZ2_WIN_EPS; }
static int g_invalid = 0;    // 運び方を再現できなかった計画の数（main の終了コードに反映する）

// DRAM（14080 だけ）：1 組で収まるか．収まらなければ組分けの組数（組分けの費用 grp を持つ TF32），それも無ければ dram_ok=false．
// maxnout は合成段を通る法の出力の最大（0 なら合成する法が無い）
static void dram_plan(const Route& R, const Coef& C, int K, int S, int nplane, int maxnout, bool& dram_ok, int& groups) {
    dram_ok = true; groups = 1;
    if (K < 14080) return;
    const double el = 2.0 * (double)K * K, P = el * R.bytes, out = (double)K * K * 4, F = el * 8;
    const double cpl = oz2_compose_plan(maxnout > 0, (uint32_t)maxnout).planes;
    if (oz2_dram_one_group(nplane, P, cpl, S, out, F) <= OZ2_AWE_DRAM_FREE) return;
    if (C.grp > 0) {
        int n = 2;
        for (; n <= S; n++) {
            // 組の桁行列の最大（法の順に均等に切る近似）．組分けでは FP64 のオペランドを持ったまま組ごとに切るので F を引かない
            const int per = (nplane + n - 1) / n + 1;
            if (oz2_dram_groups(per, P, cpl, S, out) <= OZ2_AWE_DRAM_FREE) break;
        }
        groups = n;
    } else dram_ok = false;
}

static void eval_plan(const Route& R, const Coef& C, int K, Plan& p, const Opt& o) {
    p.S = (int)p.mods.size(); p.nprod = p.nout = p.nplane = p.ops = 0; p.bits = 0;
    int mmax = 0; for (auto& x : p.mods) mmax = std::max(mmax, x.e.m);
    double gem = 0, com = 0, sli = C.s0;
    int maxnout = 0;
    for (auto& x : p.mods) {
        const int g = route_maxg(R, x.e, K, mmax, o);
        x.nout = nout_of(x.e, g);
        p.nprod += x.e.nprod; p.nout += x.nout; p.nplane += x.e.nplane; p.ops += 19 + x.e.ops; p.bits += x.bits;
        gem += C.gp * x.e.nprod + C.go * x.nout;
        const bool composed = !(o.fuse && x.e.kind == 'S');   // 合成段を通るか（fuse の 1 桁の法は通らない）
        com += composed ? C.co * x.nout + C.cs : 0;
        if (composed) maxnout = std::max(maxnout, x.nout);
    }
    sli += C.sop * p.ops + (C.spl + C.plx) * p.nplane;
    p.excess = p.bits - need_bits(K, o.b);
    dram_plan(R, C, K, p.S, p.nplane, maxnout, p.dram_ok, p.groups);
    if (C.grp > 0) sli += C.grp * (p.groups - 1);
    // 1 桁の法の合成を再構成へ畳む（fuse）なら，その法の CRT の演算に ModRound 1 回（OZ2_AWE_ONE_RECON）が足される．法 14 本を
    // 超えて再構成が CRT で律速される側では，その分だけ傾きに乗せる（ホストの選び方 src/ozaki2/ozaki2_awe.h は，この上乗せを
    // 折れ目の式に入れず法ごとに OZ2_AWE_ONE_RECON*rb を線形に足す．違いは ozaki2_awe.h の冒頭）
    int nfused = 0; if (o.fuse) for (auto& x : p.mods) nfused += (x.e.kind == 'S');
    const double rec = std::max(C.R0, C.ra + C.rb * (p.S + OZ2_AWE_ONE_RECON * nfused));
    p.t[0] = C.scale; p.t[1] = sli; p.t[2] = gem; p.t[3] = com; p.t[4] = rec;
    p.t[5] = C.scale + sli + gem + com + rec;
    p.ok = p.excess >= 0 && p.S <= o.maxs && p.dram_ok;
}

// 法の集合（法と方式）をモデルで評価する．方式は ozaki2_moduli.h の値（OZ2_SPLIT_*），q は基数
static Plan plan_from(const Route& R, const Coef& C, int K, const Opt& o0, const std::vector<int>& ms,
                      const std::vector<int>& modes, const std::vector<int>& qs, int smax_eff, int nout_fix = -1) {
    Opt o = o0; o.pmg = false;
    Plan p;
    Route Rx = R; Rx.smax = smax_eff;
    for (size_t i = 0; i < ms.size(); i++) {
        const int m = ms[i]; Enc e{};
        const int md = modes[i];
        bool got = false;
        if (md == OZ2_SPLIT_ONE) got = enc_single(Rx, m, e);
        else if (md == OZ2_SPLIT_SQ || md == OZ2_SPLIT_SQS) got = enc_square(Rx, m, e, C);
        else if (md == OZ2_SPLIT_KAR) { const int B = qs[i]; int dl, dh, ds; got = kar_check(Rx, m, B, B - 1, dl, dh, ds);
            if (B == 16) { dl = std::max(dl, 15); dh = std::max(dh, 15); ds = std::max(ds, 15); }
            e = Enc{}; e.m = m; e.kind = 'k'; e.B = B; e.pb = {dl, dh, ds};
            e.terms = {{0,0,0,symm(1 - B,m)},{1,1,2,symm((long)B * B - B,m)},{2,2,4,symm(B,m)}}; e.ops = 12; finish(e); }
        else if (md == OZ2_SPLIT_WIDE) got = enc_wide(Rx, m, e);
        else got = enc_bal(Rx, m, qs[i], e);               // BAL は qs に桁数を入れる
        // 再現できない法を空の運び方のまま評価すると，その法の費用が 0 の計画になって予測が黙って安く出る
        if (!got || !e.m) { fprintf(stderr, "ERROR: %s K=%d 法 %d（方式 %d）の運び方を再現できない\n", R.name, K, m, md); p.valid = false; }
        p.mods.push_back(ModCost{e, 0, 0, std::log2((double)m)});
    }
    eval_plan(R, C, K, p, o);
    if (nout_fix >= 0 && p.nout != nout_fix) {      // chooser の出力本数と合わせる（FP8 の法ごとの maxg を含む）
        const double d = (nout_fix - p.nout) * (C.go + C.co);
        p.t[2] += (nout_fix - p.nout) * C.go; p.t[3] += (nout_fix - p.nout) * C.co; p.t[5] += d; p.nout = nout_fix;
    }
    return p;
}
// ozaki2_choose_moduli_route の結果（ホストの選び方）をモデルで評価する．法が取れなければ（mo.ok が偽）無効な計画を返す
// （空の法の集合を評価すると段の時間が 0 の計画になり，較正の倍率が無限大になる）
static Plan plan_route(const Route& R, const Oz2Route& rt, const Coef& C, int K, const Opt& o) {
    const Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, rt);
    if (!mo.ok) {
        fprintf(stderr, "ERROR: %s K=%d ホストの選び方（ozaki2_choose_moduli_route）で法が取れない\n", R.name, K);
        Plan p; p.valid = false;
        return p;
    }
    std::vector<int> qs = mo.q;
    for (int i = 0; i < mo.s; i++) if (mo.mode[i] == OZ2_SPLIT_BAL) qs[i] = oz2_bal_digits(mo.m[i], rt.digit_bound, rt.nd());
    Plan p = plan_from(R, C, K, o, mo.m, mo.mode, qs, mo.smax, mo.nout);
    // 1 桁の法は合成段を通らない（ozaki2_main.cpp）：合成の費用を外し，再構成に ModRound の分を乗せる
    int n1 = 0; for (int i = 0; i < mo.s; i++) n1 += mo.mode[i] == OZ2_SPLIT_ONE;
    if (n1) {
        p.t[3] -= n1 * (C.co + C.cs);
        const double rec = std::max(C.R0, C.ra + C.rb * (p.S + OZ2_AWE_ONE_RECON * n1));
        p.t[5] += (rec - p.t[4]) - n1 * (C.co + C.cs); p.t[4] = rec;
    }
    return p;
}
static Plan plan_host(const Route& R, const Coef& C, int K, const Opt& o) { return plan_route(R, R.hostr, C, K, o); }
// 段の時間の実測（MEAS）を取ったときの法の集合（較正の基準）．BFP4_b・BFP2_b は表から取って末尾を選び直す
// 選び方の集合を下に書き下してあり，BFP2_b は全部の法を平衡 5 桁で運ぶ．それ以外の経路は R.cur をホストの
// ozaki2_choose_moduli_route に渡した選び方（桁 32 の経路は表から取る OZ2_ROUTE_D32，TF32・FP8 はホストと同じ）．
static Plan plan_current(const Route& R, const Coef& C, int K, const Opt& o0) {
    if (!R.cur.awe && !R.hostr.awe) return plan_host(R, C, K, o0);
    if (!R.cur.awe) return plan_route(R, R.cur, C, K, o0);      // 桁 32 の経路：実測を取った表の選び方
    const bool b2 = R.cur.nrdig > 2;
    std::vector<int> ms;
    if (b2) {
        ms = {243, 242, 241, 239, 235, 233, 229, 227, 223, 221, 217, 211, 199, 197};
        if (K <= 3520) { ms.push_back(37); ms.push_back(73); } else if (K <= 7040) { ms.push_back(41); ms.push_back(131); }
        else { ms.push_back(71); ms.push_back(151); }
    } else {
        ms = {225, 196, 169, 121, 223, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163};
        if (K <= 3520) { ms.push_back(17); ms.push_back(19); } else if (K <= 7040) { ms.push_back(17); ms.push_back(29); }
        else { ms.push_back(23); ms.push_back(43); }
    }
    std::vector<int> modes, qs;
    for (int m : ms) {
        if (b2) { modes.push_back(OZ2_SPLIT_BAL); qs.push_back(5); }
        else if (oz2_is_square(m)) { modes.push_back(OZ2_SPLIT_SQ); qs.push_back(oz2_is_square(m)); }
        else { modes.push_back(OZ2_SPLIT_WIDE); qs.push_back(oz2_digit_base_rnd(m)); }
    }
    return plan_from(R, C, K, o0, ms, modes, qs, 0);
}

static std::string tagstr(const Plan& p) {
    std::string s = "{";
    std::vector<ModCost> ms = p.mods;
    std::sort(ms.begin(), ms.end(), [](const ModCost& a, const ModCost& b) { return a.e.m > b.e.m; });
    for (size_t i = 0; i < ms.size(); i++) {
        char b[64]; const Enc& e = ms[i].e;
        if (e.kind == 'k' || e.kind == 'f') snprintf(b, sizeof b, "%d%c%d", e.m, e.kind, e.B);
        else if (e.kind == 'h') snprintf(b, sizeof b, "%dh", e.m);
        else if (e.kind == 'b') snprintf(b, sizeof b, "%db%d", e.m, e.n);
        else snprintf(b, sizeof b, "%d%c", e.m, e.kind);
        s += b; if (i + 1 < ms.size()) s += ",";
    }
    return s + "}";
}
static void print_plan(const char* tag, const Plan& p, double base) {
    if (!p.valid) {          // 評価が意味を持たないので普通の行として出さない（終了コードは main が 1 にする）
        printf("  %-22s 無効（法の運び方を再現できない．%s）\n", tag, tagstr(p).c_str());
        g_invalid++;
        return;
    }
    printf("  %-22s S=%2d 行列積=%3d 出力=%3d 桁行列=%3d 超過=%.3f  予測 %8.2f ms（%+6.2f%%）"
           " [scale %.1f slice %.1f gemm %.1f 合成 %.1f recon %.1f%s]\n      %s\n",
           tag, p.S, p.nprod, p.nout, p.nplane, p.excess, p.t[5], base > 0 ? 100.0 * (p.t[5] / base - 1) : 0.0,
           p.t[0], p.t[1], p.t[2], p.t[3], p.t[4], p.groups > 1 ? (" 組 " + std::to_string(p.groups)).c_str() : "",
           tagstr(p).c_str());
}

// ---------------------------------------------------------------------------
// 段ごとの較正：実測を取った選び方（plan_current）をモデルで評価した段の時間を，実測（measure_logs の冷却条件の中央値）に合わせる
// 倍率を段ごとに掛ける（モデルの相対的な変化はそのままで，絶対値と段どうしの重みを経路ごとの実測に合わせる）．
// 7040^3 に実測の無い経路は 3520^3 と 14080^3 の倍率の相乗平均を使う．
// MEAS の各行はスライス・GEMM・合成・再構成の段の時間［ms］（尾崎 II のドライバの段ごとの表示の slice A+B・GEMM・compose・recon）．
// 出どころ（2026-10-23 に measure_logs/tmp_awe/rows.json（当時のログごとの段の時間を集めた作業の表）とログを照らして辿った結果．
// ログはどれも measure_logs/ の下．rev は各ログの PROVENANCE 行，dirty=1 は作業木を汚したままの A/B．出どころの行の無いログもある）：
//   bf16 7040   _ab/rev_f26_7040/B_oz2bf16_r1・r2（48e6aaf，dirty=1）と _work/f26_B/ozaki2_bf16_7040.log の 3 回の段ごとの中央値
//               （法の集合 {4096,3969,3721,3481,3025,2809,2209,1849,1681,289,1369}．以下も法の集合は plan_current のもの）
//   bf16 14080  cool_oz2bf16_14080_nongemm.log（fa49353，dirty=0），_ab/it6/B_oz2bf16_r1・r2（60b2568，dirty=1），
//               _ab/it8/A・B_oz2bf16_r1・r2（e5b861a，dirty=1）の 7 回の中央値
//   bfp8 3520   tmp_fp8sparse/cks/A・B_oz2bfp8.log と acc_c5/oz2bfp8_3520.log（どれも出どころの行なし）の 3 回の中央値
//   bfp8 14080  cool_oz2bfp8_14080_nongemm.log（fa49353），_ab/it6/B_oz2bfp8_r1・r2（60b2568），
//               _ab/rev_f18f19/A・B_oz2bfp8_r1・r2（48e6aaf，dirty=1）の 7 回の中央値
//   fp16 3520   tmp_fp8sparse/cks/A・B_oz2fp16.log と acc_c5/oz2fp16_3520.log の 3 回の中央値
//   fp16 14080  cool_oz2fp16_14080_nongemm.log（fa49353）の 1 回
//   tf32 14080  cool_oz2fp32_14080_nongemm.log（fa49353）と _ab/it6/B_oz2fp32_r1・r2（60b2568）の 3 回の中央値
//   fp8 7040    _ab/fp8s_7040/B_oz2fp8_r1・r2（5424088，dirty=0）と tmp_fp8sparse/acc・dev/B_7040.log の 4 回の中央値
//   残りの 7 行（bf16・tf32・fp8・bfp4・bfp2 の 3520，fp8・bfp4・bfp2 の 14080）は再現できない記録：段ごとには一致する回があるが
//   （bf16・tf32 の 3520 は acc_c5・_work/f26_B・_ab/rev_f26_3520，fp8 は _ab/fp8s_3520・_ab/fp8s_14080 の B（5424088），bfp4・bfp2 の
//   3520 は acc_* の回，bfp4・bfp2 の 14080 は _ab/it8/B_oz2bfp4・B_oz2bfp2 の r1・r2（e5b861a，dirty=1）），4 段を同じ回の組の
//   中央値としてそろえる組が無く，値の作り方（どの回をどう組み合わせたか）が残っていない．
// どれも 7c06f82 で入れた値で，ログの多くは dirty=1 の A/B である．値を変えると emit-coef の出力が変わる（verify-coef が止める）
struct Meas { const char* r; int K; double sl, ge, co, re; };
static const Meas MEAS[] = {
    {"bf16", 3520, 12.23, 12.98, 5.10, 17.05}, {"bf16", 7040, 48.60, 73.80, 20.23, 68.35}, {"bf16", 14080, 193.51, 653.27, 150.42, 278.85},
    {"bfp8", 3520, 12.25, 12.48, 5.15, 17.04}, {"bfp8", 14080, 193.38, 571.58, 141.85, 274.29},
    {"fp16", 3520, 24.51, 13.02, 5.18, 17.17}, {"fp16", 14080, 389.35, 647.59, 148.95, 275.92},
    {"tf32", 3520, 22.37, 18.62, 5.09, 17.07}, {"tf32", 14080, 583.73, 1016.71, 140.72, 276.12},
    {"fp8",  3520, 14.55, 12.72, 5.40, 16.97}, {"fp8", 7040, 58.67, 75.36, 23.06, 67.80}, {"fp8", 14080, 230.60, 601.01, 118.00, 276.96},
    {"bfp4", 3520, 17.72, 24.71, 10.89, 20.04}, {"bfp4", 14080, 280.78, 976.00, 198.97, 318.17},
    {"bfp2", 3520, 27.65, 99.80, 35.20, 18.89}, {"bfp2", 14080, 439.68, 5645.95, 695.98, 304.26},
};
struct Fac { double sl = 1, ge = 1, co = 1, re = 1; bool meas = false; };
static Fac factors_at(const Route& R, int K) {
    Fac f;
    for (auto& m : MEAS) if (!strcmp(m.r, R.name) && m.K == K) {
        const Coef c = coef_raw(R.name, K);
        Plan p = plan_current(R, c, K, Opt{});
        // 無効な計画や 0・非有限の段の時間で割ると，倍率が無限大・NaN になって本番の係数の表まで壊れる（冒頭の終了コード 1）
        bool bad = !p.valid;
        for (int s = 1; s <= 4; s++) bad = bad || !std::isfinite(p.t[s]) || !(p.t[s] > 0);
        if (bad) {
            fprintf(stderr, "ERROR: %s K=%d の較正の基準（plan_current）が無効か，段の時間が有限の正の数でない"
                            "（slice %g gemm %g 合成 %g recon %g）\n", R.name, K, p.t[1], p.t[2], p.t[3], p.t[4]);
            std::exit(1);
        }
        f.sl = m.sl / p.t[1]; f.ge = m.ge / p.t[2]; f.co = m.co / p.t[3]; f.re = m.re / p.t[4]; f.meas = true;
    }
    return f;
}
static Fac factors_for(const Route& R, int K) {
    Fac f = factors_at(R, K);
    if (f.meas) return f;
    const Fac a = factors_at(R, 3520), b = factors_at(R, 14080);
    f.sl = std::sqrt(a.sl * b.sl); f.ge = std::sqrt(a.ge * b.ge); f.co = std::sqrt(a.co * b.co); f.re = std::sqrt(a.re * b.re);
    return f;
}
static const Route* route_by_name(const char* r) { for (auto& x : ROUTES) if (!strcmp(x.name, r)) return &x; return nullptr; }
static Coef coef_for(const char* r, int K) {
    Coef c = coef_raw(r, K);
    const Route* R = route_by_name(r);
    if (!R) { fprintf(stderr, "ERROR: coef_for：経路 %s が ROUTES に無い\n", r); std::exit(2); }
    const Fac f = factors_for(*R, K);
    c.s0 *= f.sl; c.sop *= f.sl; c.spl *= f.sl; c.plx *= f.sl; c.grp *= f.sl;
    c.gp *= f.ge; c.go *= f.ge; c.co *= f.co; c.cs *= f.co; c.R0 *= f.re; c.ra *= f.re; c.rb *= f.re;
    return c;
}

// ---------------------------------------------------------------------------
// 組（費用が同じ候補）の本数を列挙する探索．組の費用は組の本数と集合の最大の法で決まる（最大の法で連結の本数 g が変わり，
// 出力の本数と DRAM の見積りが変わる．大きいほど g が減って出力が増える）．そこで組ごとに上界（最大の法を法の上限 cap と
// した値．Klass の cost・nout）と下界（最大の法を組の各法自身とした値の最小．cost_lo・nout_lo）を持ち，
// (1) 組ごとの本数のベクトルを，bit の上界が窓に届き，bit の下界が窓の上端を超えず，費用の下界 T_lo が基準（plan_current）を
//     下回るものだけ集めて，T_lo の安い順に並べ，
// (2) 安い順に，実際に互いに素な法を窓 [need, need+tol] に収めて選べるかを割り当ての探索で確かめ，見つかった集合を評価する
//     （eval_plan）．評価が最良を下回れば入れ替え，残りのベクトルの T_lo が最良以上になったところで止める（分枝限定．
//     T_lo は下界なので，止めた後ろのベクトルから最良より安い集合は出ない）．
// 割り当ての探索は互いに素で窓に入る集合のうち超過の最小を探すので，ベクトルの費用が集合で変わる（T_lo < 上界 T_hi）とき，
// 見つけた集合の評価がそのベクトルの最安とは限らない．評価が T_lo ちょうどでなければ，そのベクトルは最安を示せていない．
// 最適は候補を組ごとに大きい法の上位 400 本（optimize の topn）に絞り，法ごとの運び方を法の上限で見積もった費用で 1 つに
// 決めた中での話で，そう言えるのは次の 3 つが成り立つときである（成り立たなければ optimize は *to を立て，行に * を付ける）：
//   ・本数の打ち切りと bit の上界・下界，費用の下界が証明された値であること（optimize の ubpre・lbpre，組をまたぐ最小の
//     素因数の割り当て，ModCost の cost_lo）
//   ・列挙の費用の式 global_T が最後の評価 eval_plan と同じ式であること（1 桁の法の合成を再構成へ畳む fuse の上乗せを含む．
//     評価が [T_lo, T_hi] の外に出たら式が食い違っているので * を付ける）
//   ・割り当ての探索と列挙を打ち切らず，最良より T_lo の安いベクトルがどれも「割り当てが無い」か「評価が T_lo ちょうど」で
//     片付いたこと（打ち切ったベクトルと最安を示せていないベクトルの T_lo が最良を下回れば * を付ける）
// main と table が回す 3520・7040・14080 では，全経路・全変種で候補のどれも cost_lo = cost なので T_lo = T_hi で，最初に
// 割り当てが見つかったベクトルが最適になる（法ごとの連結の上限 pmg・FP8・BFP2_b などで g が最大の法で変わる運び方はあるが，
// 変わっても g が位ごとの積の本数以上あるので出力の本数は変わらない）．T_lo < T_hi になるのは，例えば BF16 の K=23232 の
// pmg の変種（候補 2 本）である．
struct Klass { std::string key; double cost; int nout, nplane; bool single; std::vector<ModCost> mods; std::vector<double> pre;
               double cost_lo; int nout_lo; };
struct CountVec { double T, Thi; std::vector<int> n; };   // T は費用の下界 T_lo，Thi は上界

// nS は 1 桁の法の本数（fuse のときだけ再構成に法 1 本あたり OZ2_AWE_ONE_RECON 本ぶんを乗せる．eval_plan と同じ式）
static double global_T(const Route& R, const Coef& C, int K, int S, int nplane, int maxnout, double sumcost, int nS, bool fuse,
                       bool& dram_ok, int& groups) {
    dram_plan(R, C, K, S, nplane, maxnout, dram_ok, groups);
    const double extra = C.grp > 0 ? C.grp * (groups - 1) : 0.0;
    const double rec = std::max(C.R0, C.ra + C.rb * (S + (fuse ? OZ2_AWE_ONE_RECON * nS : 0.0)));
    return C.scale + C.s0 + sumcost + rec + extra;
}
// m の最小の素因数（m >= 2）
static int min_prime_factor(int m) {
    for (int p = 2; (long)p * p <= m; p++) if (m % p == 0) return p;
    return m;
}

// 割り当て：組 c から n[c] 本，互いに素で bit の和が [need, need+tol] に入る法を選ぶ（超過の最小を探す）
struct Assign {
    const std::vector<Klass>& K; const std::vector<int>& n; double need, tol;
    std::vector<int> pickm; std::vector<const ModCost*> pick, best; double bestex = 1e9; long nodes = 0, limit = 3000000;
    std::vector<double> ubrest, lbrest;   // 組 c 以降の bit の上界・下界
    const std::vector<std::vector<double>>* ubp = nullptr;   // 組ごとの本数に対する bit の上界（optimize の ubpre．無ければ pre だけ）
    std::vector<std::vector<double>> low;  // 組ごとの下位 t 本の bit の和（互いに素を無視した下界）
    Assign(const std::vector<Klass>& k, const std::vector<int>& nn, double nd, double tl) : K(k), n(nn), need(nd), tol(tl) {}
    bool cop(int m) const { for (int x : pickm) if (std::gcd(m, x) != 1) return false; return true; }
    void dfs(int c, int start, int left, double bits) {
        if (++nodes > limit || bestex <= 1e-4) return;
        if (c == (int)K.size()) {
            const double ex = bits - need;
            if (ex >= 0 && ex <= tol && ex < bestex) { bestex = ex; best = pick; }
            return;
        }
        if (left == 0) { dfs(c + 1, 0, c + 1 < (int)K.size() ? n[c + 1] : 0, bits); return; }
        const auto& L = K[c].mods;
        for (int j = start; j + left <= (int)L.size(); j++) {
            // 残りの上界・下界（この組の残り left 本は j 以降の上位／下位で見積もる）
            const double ubc = K[c].pre[std::min((int)L.size(), j + left)] - K[c].pre[j];
            if (bits + ubc + ubrest[c + 1] < need) return;           // j を進めると上界はさらに下がる
            if (!cop(L[j].e.m)) continue;
            // 下界が窓の上端を超えるなら L[j] は取れない（j を進めると L[j] は小さくなるので先は調べる）
            if (bits + L[j].bits + low[c][left - 1] + lbrest[c + 1] > need + tol + 1e-12) continue;
            pickm.push_back(L[j].e.m); pick.push_back(&L[j]);
            dfs(c, j + 1, left - 1, bits + L[j].bits);
            pickm.pop_back(); pick.pop_back();
            if (nodes > limit || bestex <= 1e-4) return;
        }
    }
    void run() {
        const int C = (int)K.size();
        ubrest.assign(C + 1, 0); lbrest.assign(C + 1, 0); low.assign(C, {});
        for (int c = C - 1; c >= 0; c--) {
            const auto& L = K[c].mods;
            double ub = K[c].pre[std::min((int)L.size(), n[c])];
            if (ubp && n[c] < (int)(*ubp)[c].size()) ub = std::min(ub, (*ubp)[c][n[c]]);
            ubrest[c] = ubrest[c + 1] + ub;
            low[c].assign(1, 0);
            for (int t = 0; t < n[c] && t < (int)L.size(); t++) low[c].push_back(low[c].back() + L[L.size() - 1 - t].bits);
            lbrest[c] = lbrest[c + 1] + low[c].back();
        }
        dfs(0, 0, C ? n[0] : 0, 0);
    }
};

// ---- optimize の段階（呼ぶ順は optimize の本体．どの段階も前の段階の結果だけを読む）----
// 段階 1：候補を費用が同じ組に分け，組の中を法の降順に並べて bit の累積和 pre を作り，組を bit あたりの費用の昇順に並べる
// （上界の見積りと列挙の順）
static std::vector<Klass> make_classes(const std::vector<ModCost>& cand) {
    std::map<std::string, int> idx; std::vector<Klass> ks;
    for (auto& x : cand) {
        char k[96]; snprintf(k, sizeof k, "%c/%d/%d/%d/%d", x.e.kind, x.e.nprod, x.nout, x.e.nplane, x.e.ops);
        auto it = idx.find(k);
        if (it == idx.end()) { idx[k] = (int)ks.size();
                               ks.push_back({k, x.cost, x.nout, x.e.nplane, x.e.kind == 'S', {}, {}, x.cost_lo, x.nout_lo});
                               it = idx.find(k); }
        Klass& kk = ks[it->second];
        kk.mods.push_back(x);
        kk.cost_lo = std::min(kk.cost_lo, x.cost_lo); kk.nout_lo = std::min(kk.nout_lo, x.nout_lo);
    }
    for (auto& k : ks) {
        std::sort(k.mods.begin(), k.mods.end(), [](const ModCost& a, const ModCost& b) { return a.e.m > b.e.m; });
        k.pre.assign(k.mods.size() + 1, 0);
        for (size_t i = 0; i < k.mods.size(); i++) k.pre[i + 1] = k.pre[i] + k.mods[i].bits;
    }
    std::sort(ks.begin(), ks.end(), [](const Klass& a, const Klass& b) {
        return a.cost / a.mods[0].bits < b.cost / b.mods[0].bits; });
    return ks;
}

// 段階 2：列挙の bit の上界と下界（組から t 本を取ったとき）．
//   上界 ubpre：互いに素な法どうしは最小の素因数がすべて違うので，組から互いに素な t 本を取った bit の和は，
//     「最小の素因数ごとに組の中で最大の法の log2」を降順に並べた先頭 t 本の和を超えない．取れる本数も最小の素因数の
//     種類の数を超えない．互いに素を無視した上位 t 本の和（Klass の pre）も上界だが，小さい法の組（BFP2_b の平衡 3 桁の
//     27 以下など）で大きく外れ，届かない本数の組を大量に試すことになる．大きい順に互いに素なものを取った貪欲の和は
//     上界ではない（{6,5,4,3} の貪欲は {6,5} の 2 本で止まるが {5,4,3} の 3 本が取れる）ので，本数の打ち切りには使わない．
//   下界 lbpre：組の下位 t 本の log2 の和（互いに素を無視）．これが窓の上端 need+tol を超えたら，本数を増やしても窓に入らない．
static void count_bounds(const std::vector<Klass>& ks, std::vector<std::vector<double>>& ubpre,
                         std::vector<std::vector<double>>& lbpre) {
    ubpre.assign(ks.size(), {}); lbpre.assign(ks.size(), {});
    for (size_t c = 0; c < ks.size(); c++) {
        std::map<int, double> top;       // 最小の素因数 -> その素因数を持つ組の法の log2 の最大
        for (auto& x : ks[c].mods) { double& v = top[min_prime_factor(x.e.m)]; v = std::max(v, x.bits); }
        std::vector<double> v; for (auto& kv : top) v.push_back(kv.second);
        std::sort(v.begin(), v.end(), std::greater<double>());
        ubpre[c].assign(1, 0); for (double b : v) ubpre[c].push_back(ubpre[c].back() + b);
        lbpre[c].assign(1, 0);           // mods は法の降順なので下位は後ろから
        for (size_t i = ks[c].mods.size(); i-- > 0 && lbpre[c].size() < ubpre[c].size(); )
            lbpre[c].push_back(lbpre[c].back() + ks[c].mods[i].bits);
    }
}

// 段階 3：組をまたぐ互いに素の必要条件：選んだ法の最小の素因数はすべて違う．組 c の n[c] 本に，組の法の最小の素因数の集合 P_c から
// 互いに違う素数を割り当てられなければ（二部グラフのマッチング），そのベクトルも本数を増やしたベクトルも窓に届く集合を持たない．
// P_c が法の本数の上限 maxs 以上の組は Hall の条件を破る組の集まりに入りえないので，P_c の小さい組だけで調べれば足りる
// （BFP2_b の 2・3 の冪と積だけから成る小さい法の組どうしが同じ素数を取り合うベクトルを，割り当ての探索の前に落とす）．
struct SpfMatch {
    std::vector<std::vector<int>> pset;   // 組ごとの最小の素因数の番号（大きい組は空＝調べない）
    int nprime = 0;
    bool feasible(const std::vector<int>& n) const {
        std::vector<int> slot_c, own(nprime, -1);
        std::vector<char> vis;
        for (int c = 0; c < (int)pset.size(); c++) if (!pset[c].empty()) for (int t = 0; t < n[c]; t++) slot_c.push_back(c);
        std::function<bool(int)> aug = [&](int sl) {
            for (int q : pset[slot_c[sl]]) {
                if (vis[q]) continue;
                vis[q] = 1;
                if (own[q] < 0 || aug(own[q])) { own[q] = sl; return true; }
            }
            return false;
        };
        for (int sl = 0; sl < (int)slot_c.size(); sl++) { vis.assign(nprime, 0); if (!aug(sl)) return false; }
        return true;
    }
};
static SpfMatch make_spf(const std::vector<Klass>& ks, int maxs) {
    SpfMatch s; s.pset.assign(ks.size(), {});
    std::map<int, int> pid;
    for (size_t c = 0; c < ks.size(); c++) {
        std::map<int, int> seen;
        for (auto& x : ks[c].mods) seen[min_prime_factor(x.e.m)] = 1;
        if ((int)seen.size() >= maxs) continue;          // 大きい組（調べない）
        for (auto& kv : seen) { auto it = pid.emplace(kv.first, (int)pid.size()).first; s.pset[c].push_back(it->second); }
    }
    s.nprime = (int)pid.size();
    return s;
}

// 段階 4：本数のベクトルの列挙．基準（plan_current）より安くなりうるベクトル（費用の下界 T_lo < incumbent）を集める．
// 枝刈りは費用の下界（cost_lo）で行う（上界で刈ると，上界は基準を超えるが実際の集合は基準を下回るベクトルを落とす）．
// 本数のベクトルは葉（全部の組の本数を決めたところ）でだけ集める（途中で集めると，残りの組を 0 本にした同じベクトルを
// 葉でもう一度集めて重複する）．上界が窓に届いても先へ進むのは，上界が実際の bit を上回りうるからである
// （本数を足すのは費用の上では損だが，届かないベクトルの代わりになる）．
// rec の cost・mxo は費用と出力の本数の最大の上界，clo・mlo は下界（出力の本数は合成段を通る組だけで数える．DRAM の見積り）
struct CountEnum {
    const Route& Rx; const Coef& C; int K; const Opt& o; const std::vector<Klass>& ks;
    const std::vector<std::vector<double>>& ubpre; const std::vector<std::vector<double>>& lbpre; const SpfMatch& spf;
    double need, incumbent;
    int NC = 0;
    std::vector<double> minrate;
    std::vector<int> n;
    std::vector<CountVec> vecs;
    CountEnum(const Route& rx, const Coef& c, int k, const Opt& op, const std::vector<Klass>& kk,
              const std::vector<std::vector<double>>& ub, const std::vector<std::vector<double>>& lb, const SpfMatch& sp,
              double nd, double inc)
        : Rx(rx), C(c), K(k), o(op), ks(kk), ubpre(ub), lbpre(lb), spf(sp), need(nd), incumbent(inc) {}
    void rec(int c, int S, double cost, double clo, double ub, double lb, int npl, int mxo, int mlo, int nS) {
        if (c == NC) {
            if (ub >= need && lb <= need + o.tol) {
                bool dok, dlo; int g;
                const double Thi = global_T(Rx, C, K, S, npl, mxo, cost, nS, o.fuse, dok, g);
                const double T = global_T(Rx, C, K, S, npl, mlo, clo, nS, o.fuse, dlo, g);
                if (dlo && T < incumbent && S <= o.maxs) vecs.push_back({T, dok ? Thi : 1e30, n});
            }
            return;
        }
        // 下界：固定費 + ここまで + 残りの bit × 残りの組の最良の bit あたり + 再構成
        const double rem = std::max(0.0, need - ub);
        if (C.scale + C.s0 + clo + rem * minrate[c] + recon_ms(C, S) >= incumbent) return;
        const auto& k = ks[c];
        const int maxn = std::min<int>((int)ubpre[c].size() - 1, o.maxs - S);
        for (int t = 0; t <= maxn; t++) {
            n[c] = t;
            const double ub2 = ub + ubpre[c][t], lb2 = lb + lbpre[c][t];
            // 下界は t について単調でない（この組が残りの組より bit あたり安いと，t を増やすと下がる）ので飛ばすだけ．
            // 本数の費用だけで超えたら先も超える．bit の下界が窓の上端を超えたら先も超える．
            if (C.scale + C.s0 + clo + t * k.cost_lo + recon_ms(C, S + t) >= incumbent) break;
            if (lb2 > need + o.tol + 1e-12) break;
            if (t > 0 && !spf.pset[c].empty() && !spf.feasible(n)) break;   // 本数を増やしても割り当てられない
            if (t > 0 && C.scale + C.s0 + clo + t * k.cost_lo + std::max(0.0, need - ub2) * minrate[c + 1] + recon_ms(C, S + t) >= incumbent) continue;
            const bool composed = t && !(o.fuse && k.single);   // 合成段を通るか（eval_plan と同じ）
            rec(c + 1, S + t, cost + t * k.cost, clo + t * k.cost_lo, ub2, lb2, npl + t * k.nplane,
                composed ? std::max(mxo, k.nout) : mxo, composed ? std::max(mlo, k.nout_lo) : mlo, nS + (k.single ? t : 0));
        }
        n[c] = 0;
    }
    // 列挙して費用の下界の安い順に並べる
    std::vector<CountVec> run() {
        NC = (int)ks.size();
        minrate.assign(NC + 1, 1e30);
        for (int c = NC - 1; c >= 0; c--) minrate[c] = std::min(minrate[c + 1], ks[c].cost_lo / ks[c].mods[0].bits);
        n.assign(NC, 0);
        rec(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        std::sort(vecs.begin(), vecs.end(), [](const CountVec& a, const CountVec& b) { return a.T < b.T; });
        return vecs;
    }
};

// 段階 5：安い順に割り当てを試す（ベクトルは葉でだけ集めたので重複しない）．T（下界）が最良以上になったら止める（Klass の前の
// 説明）．割り当ての試行は 20000 本か 60 秒で打ち切り，1 本の割り当ての探索は 300 万節点で打ち切る．最適を示せなかったら *to を
// 立てる（試行の打ち切り，最良より費用の下界が安いのに片付かなかったベクトル，列挙の費用と評価の食い違い）．
static Plan try_assignments(const Route& Rx, const Coef& C, int K, const Opt& o, const std::vector<Klass>& ks,
                            const std::vector<std::vector<double>>& ubpre, const std::vector<CountVec>& vecs, const Plan& cur,
                            double need, bool dbg, bool* to) {
    const int NC = (int)ks.size();
    Plan best = cur; long tried = 0, node_cap = 0; bool trunc = false, improved = false;
    double open_lo = 1e30;   // 最安を示せていないベクトル（割り当ての探索の打ち切り・評価が下界を上回る集合）の T の最小
    // 割り当ては候補の少ない組から（大きい組は窓合わせの細かさに使う）
    std::vector<int> ord(NC); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return ks[a].mods.size() < ks[b].mods.size(); });
    std::vector<Klass> ks2; std::vector<std::vector<double>> ub2pre;
    for (int c : ord) { ks2.push_back(ks[c]); ub2pre.push_back(ubpre[c]); }
    const auto tstart = std::chrono::steady_clock::now();
    auto eps = [](double x) { return 1e-9 * std::max(1.0, std::fabs(x)); };
    for (auto& v : vecs) {
        // 最良を入れ替える前は基準（plan_current）より真に安いものだけを試す（基準と同じ費用なら基準を残す）．入れ替えた後は
        // 評価の丸めの揺れ（eps）で同じ費用の集合に入れ替わらないよう，eps だけ安いものに限る
        if (v.T >= (improved ? best.t[5] - eps(best.t[5]) : best.t[5])) break;
        if (++tried > 20000) { trunc = true; break; }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - tstart).count() > 60.0) { trunc = true; break; }
        std::vector<int> n2; for (int c : ord) n2.push_back(v.n[c]);
        Assign a(ks2, n2, need, o.tol); a.limit = 3000000; a.ubp = &ub2pre;
        a.run();
        if (dbg && tried <= 10) {
            fprintf(stderr, "  vec T=%.2f（上界 %.2f）nodes=%ld found=%d ex=%.4f n=", v.T, v.Thi, a.nodes, (int)!a.best.empty(), a.bestex);
            for (int c = 0; c < NC; c++) { if (v.n[c]) fprintf(stderr, "%s:%d ", ks[c].key.c_str(), v.n[c]); }
            fprintf(stderr, "\n");
        }
        // 見つからないまま打ち切ったベクトルは届かないと示せていない
        if (a.nodes > a.limit && a.best.empty()) { open_lo = std::min(open_lo, v.T); ++node_cap;
            if (dbg) { fprintf(stderr, "  打ち切り（割り当てが見つからない）vec T=%.2f n=", v.T);
                       for (int c = 0; c < NC; c++) if (v.n[c]) fprintf(stderr, "%s:%d ", ks[c].key.c_str(), v.n[c]);
                       fprintf(stderr, "\n"); } }
        if (a.best.empty()) continue;
        Plan p; for (auto* x : a.best) p.mods.push_back(*x);
        eval_plan(Rx, C, K, p, o);
        // 評価は [T, Thi] に入るはず．外れたら列挙と評価の式が食い違う（Klass の前の説明）
        if (p.t[5] < v.T - eps(v.T) || p.t[5] > v.Thi + eps(v.Thi)) {
            trunc = true;
            if (dbg) fprintf(stderr, "  列挙の費用 [%.6f, %.6f] と評価 %.6f が食い違う\n", v.T, v.Thi, p.t[5]);
        }
        const bool good = p.ok && p.excess <= o.tol + 1e-9;
        if (good && p.t[5] < (improved ? best.t[5] - eps(best.t[5]) : best.t[5])) { best = p; improved = true; }
        // 評価が下界ちょうどでなければ，同じベクトルの別の集合（最大の法が小さいもの）がもっと安いかもしれない
        if (!good || p.t[5] > v.T + eps(v.T)) {
            open_lo = std::min(open_lo, v.T);
            if (dbg) fprintf(stderr, "  vec T=%.6f の集合の評価 %.6f（%s）：このベクトルの最安は示せていない\n", v.T, p.t[5],
                             good ? "下界を上回る" : "条件を満たさない");
        }
    }
    if (open_lo < best.t[5] - eps(best.t[5])) trunc = true;
    if (dbg) fprintf(stderr, "  試した %ld 本（割り当ての探索の打ち切り %ld 本）%s\n", tried, node_cap,
                     trunc ? "．打ち切りか食い違いか最安を示せていないベクトルあり" : "");
    if (to) *to = trunc;
    return best;
}

// モデルの上で最適な法の集合と運び方（Klass の前の説明）．段階 1〜5 を順に回す．最適を示せなかったら *to を立てる
static Plan optimize(const Route& R, int K, const Opt& o, bool* to = nullptr) {
    const Coef C = coef_for(R.name, K);
    int smax_eff = R.smax;
    if (R.smax && (double)K * R.smax * R.smax + R.cap > 16777216.0) smax_eff = 0;
    Route Rx = R; Rx.smax = smax_eff;
    const std::vector<ModCost> cand = candidates(R, C, K, o, smax_eff);
    const double need = need_bits(K, o.b);
    Plan cur = plan_current(R, C, K, o);
    const std::vector<Klass> ks = make_classes(cand);
    std::vector<std::vector<double>> ubpre, lbpre;
    count_bounds(ks, ubpre, lbpre);
    const SpfMatch spf = make_spf(ks, o.maxs);
    const double incumbent = cur.t[5] + 1e-6;
    CountEnum en(Rx, C, K, o, ks, ubpre, lbpre, spf, need, incumbent);
    const std::vector<CountVec> vecs = en.run();
    static const bool dbg = oz_sim::env_flag01("AWE_DEBUG");   // 列挙と割り当ての経過を stderr へ（0/1）
    if (dbg) { fprintf(stderr, "classes %d vecs %zu incumbent %.2f\n", (int)ks.size(), vecs.size(), incumbent);
               for (size_t c = 0; c < ks.size(); c++) fprintf(stderr, "  class %s cost %.3f（下界 %.3f）n=%zu top %d\n", ks[c].key.c_str(),
                                                    ks[c].cost, ks[c].cost_lo, ks[c].mods.size(), ks[c].mods[0].e.m); }
    return try_assignments(Rx, C, K, o, ks, ubpre, vecs, cur, need, dbg, to);
}

// ---------------------------------------------------------------------------
// 表による選び方の模擬（ホストの ozaki2_choose_moduli_w と同じ手順：表の先頭から num 本 → 窓が余るぶん末尾を落とす →
// 末尾 2 本を選び直す）．tail = 0 はホストと同じ基準（行列積 → 超過が許容内か → 出力・桁行列 → logM），1 はモデルの費用．
static Plan plan_table(const Route& R, int K, const std::vector<int>& table, int num, const Opt& o, int tail) {
    const Coef C = coef_for(R.name, K);
    int smax_eff = R.smax;
    if (R.smax && (double)K * R.smax * R.smax + R.cap > 16777216.0) smax_eff = 0;
    Route Rx = R; Rx.smax = smax_eff;
    Opt oo = o; oo.topn = 1000000;
    std::map<int, ModCost> byM; for (auto& x : candidates(R, C, K, oo, smax_eff)) byM[x.e.m] = x;
    std::vector<int> ms; double logM = 0;
    auto cop = [&](int m, const std::vector<int>& v) { for (int x : v) if (std::gcd(x, m) != 1) return false; return true; };
    const double W = std::log2((double)K);
    // 表の先頭から num 本（ただし窓 b = o.b に届くまでは本数を超えても足す．精度を落とさない比較のため）
    for (int m : table) {
        if ((int)ms.size() >= num && logM >= 2.0 * o.b + 1 + W + OZ2_WIN_EPS) break;
        if (!byM.count(m) || !cop(m, ms)) continue;
        ms.push_back(m); logM += std::log2((double)m);
    }
    int b = std::min((int)std::floor((logM - 1 - W - OZ2_WIN_EPS) / 2), o.b);
    const double need = 2.0 * b + 1 + W + OZ2_WIN_EPS;
    while (ms.size() > 2 && logM - std::log2((double)ms.back()) >= need) { logM -= std::log2((double)ms.back()); ms.pop_back(); }
    if (ms.size() >= 2 && logM > need) {
        std::vector<int> head(ms.begin(), ms.end() - 2); double base = 0; for (int x : head) base += std::log2((double)x);
        auto key = [&](const std::vector<int>& t, double l) {
            int np = 0, lay = 0; double cost = 0;
            for (int m : t) { const auto& x = byM[m]; np += x.e.nprod; lay += 1000 * x.nout + x.e.nplane; cost += x.cost; }
            const int over = (l - need) > o.tol ? 1 : 0;
            return tail ? std::make_tuple((double)over, cost, 0.0, l) : std::make_tuple((double)np, (double)over, (double)lay, l);
        };
        std::vector<int> bt(ms.end() - 2, ms.end()); auto bk = key(bt, logM);
        std::vector<int> cands; for (auto& kv : byM) if (cop(kv.first, head)) cands.push_back(kv.first);
        for (size_t i = 0; i < cands.size(); i++) {
            const double lp = base + std::log2((double)cands[i]);
            if (lp >= need) { auto k = key({cands[i]}, lp); if (k < bk) { bk = k; bt = {cands[i]}; } }
            for (size_t j = i + 1; j < cands.size(); j++) {
                if (std::gcd(cands[i], cands[j]) != 1) continue;
                const double lq = lp + std::log2((double)cands[j]);
                if (lq < need) continue;
                auto k = key({cands[i], cands[j]}, lq); if (k < bk) { bk = k; bt = {cands[i], cands[j]}; }
            }
        }
        ms = head; for (int x : bt) ms.push_back(x);
    }
    Plan p; for (int m : ms) p.mods.push_back(byM[m]);
    Opt ob = o; ob.b = b;
    eval_plan(Rx, C, K, p, ob);
    return p;
}
// 14080 の候補を bit あたりの費用の昇順に，互いに素なものだけ並べた表
static std::vector<int> efficiency_table(const Route& R, const Opt& o, int K = 14080, int len = 40) {
    const Coef C = coef_for(R.name, K);
    Opt oo = o; oo.topn = 1000000;
    auto cs = candidates(R, C, K, oo, R.smax);
    std::sort(cs.begin(), cs.end(), [](const ModCost& a, const ModCost& b) { return a.cost / a.bits < b.cost / b.bits; });
    std::vector<int> t;
    for (auto& x : cs) { bool ok = true; for (int y : t) if (std::gcd(x.e.m, y) != 1) { ok = false; break; } if (ok) t.push_back(x.e.m); if ((int)t.size() >= len) break; }
    return t;
}

// 較正後の係数の 1 行（本番の表 OZ2_AWE_COEF の行の書式．spl は桁行列ごとの上乗せ plx を含む）
static constexpr int kCoefK[3] = {3520, 7040, 14080};
static std::string coef_row(const Route& R, int K) {
    const Coef c = coef_for(R.name, K);
    char b[256];
    snprintf(b, sizeof b, "{%5d, %.5g, %.5g, %.5g, %.5g, %.5g, %.5g, %.5g, %.5g, %.5g},", K, c.gp, c.go, c.co, c.cs, c.sop,
             c.spl + c.plx, c.R0, c.ra, c.rb);
    return b;
}
// 段の時間のモデルで法を選ぶ経路（本番の経路の表の awe がその番号の経路）
static const Route* route_by_awe(int awe) { for (auto& x : ROUTES) if (x.hostr.awe == awe) return &x; return nullptr; }

int main(int argc, char** argv) {
    std::vector<std::string> want;
    for (int i = 1; i < argc; i++) want.push_back(argv[i]);
    auto known = [](const std::string& r) {
        if (route_by_name(r.c_str())) return true;
        fprintf(stderr, "ERROR: 経路 %s は無い（bf16 fp16 tf32 bfp8 fp8 bfp4 bfp2）\n", r.c_str());
        return false;
    };
    if (!want.empty() && want[0] == "coef") {
        // 較正後の係数（経路ごと・3 サイズ．本番の表の行と同じ書式）
        if (want.size() < 2) { fprintf(stderr, "usage: %s coef route ...\n", argv[0]); return 2; }
        for (size_t i = 1; i < want.size(); i++) if (!known(want[i])) return 2;
        for (size_t i = 1; i < want.size(); i++) {
            const Route& R = *route_by_name(want[i].c_str());
            for (int K : kCoefK) printf("    %s   // %s\n", coef_row(R, K).c_str(), R.name);
        }
        return 0;
    }
    if (!want.empty() && want[0] == "emit-coef") {
        // 本番の係数の表 src/ozaki2/ozaki2_awe_coef.inc（ozaki2_awe.h が取り込む）の中身．行は awe の番号（ozaki2_split.h の
        // OZ2_AWE_*）の順で，各経路は本番の経路の表で awe がその番号の経路
        if (want.size() != 1) { fprintf(stderr, "usage: %s emit-coef\n", argv[0]); return 2; }
        std::string body;
        for (int awe = 1; awe <= OZ2_AWE_COUNT; awe++) {
            const Route* R = route_by_awe(awe);
            if (!R) { fprintf(stderr, "ERROR: awe=%d の経路が ROUTES に無い\n", awe); return 2; }
            body += std::string("    {   // ") + R->name + "\n";
            for (int K : kCoefK) body += "        " + coef_row(*R, K) + "\n";
            body += "    },\n";
        }
        printf("// SPDX-License-Identifier: Apache-2.0\n"
               "// 生成：tools/awe_plan.cpp emit-coef（手で直さない．ozaki2_awe.h の注記）．段の時間のモデルの係数の表\n"
               "// （行は awe の番号順，各行は K = 3520・7040・14080 の {K, gp, go, co, cs, sop, spl, R0, ra, rb}）\n"
               "static constexpr Oz2AweCoef OZ2_AWE_COEF[OZ2_AWE_COUNT][3] = {\n%s};\n", body.c_str());
        return 0;
    }
    if (!want.empty() && want[0] == "verify-coef") {
        // 本番の表 OZ2_AWE_COEF（ozaki2_awe_coef.inc）は emit-coef の出力．較正の基準や実測を直した後の書き出し直し忘れと手の
        // 直しを止めるため，段の時間のモデルで法を選ぶ全経路の 3 サイズで，coef と同じ値を同じ桁（%.5g）で比べる
        if (want.size() != 1) { fprintf(stderr, "usage: %s verify-coef\n", argv[0]); return 2; }
        static const char* kNames[9] = {"gp", "go", "co", "cs", "sop", "spl", "R0", "ra", "rb"};
        int nbad = 0, nrow = 0;
        // 本番の ozaki2_awe.h は表の行を位置で読み（t[0]・t[1]・t[2] の K の昇順を前提に log K で補間する），awe の番号ごとに
        // 3 行を使う．行の入れ替え・重複・欠けを値の比べだけでは見逃す（表の K で期待値を作るので）ので，並びも確かめる：
        // 各 awe の番号に ROUTES の経路が 1 本ずつあり，その 3 行の K が kCoefK と同じ並び
        for (int awe = 1; awe <= OZ2_AWE_COUNT; awe++) {
            int nr = 0;
            for (const Route& R : ROUTES) nr += R.hostr.awe == awe;
            if (nr != 1) { printf("  awe=%d の経路が ROUTES に %d 本ある（1 本のはず）  不一致\n", awe, nr); nbad++; }
            for (int k = 0; k < 3; k++)
                if (OZ2_AWE_COEF[awe - 1][k].K != kCoefK[k]) {
                    printf("  awe=%d の %d 行目の K=%d が %d でない（表の並びが崩れている）  不一致\n", awe, k, OZ2_AWE_COEF[awe - 1][k].K,
                           kCoefK[k]);
                    nbad++;
                }
        }
        for (const Route& R : ROUTES) {
            if (!R.hostr.awe) continue;
            const Oz2AweCoef* t = OZ2_AWE_COEF[R.hostr.awe - 1];
            for (int k = 0; k < 3; k++) {
                const Coef c = coef_for(R.name, t[k].K);
                const double mine[9] = {c.gp, c.go, c.co, c.cs, c.sop, c.spl + c.plx, c.R0, c.ra, c.rb};
                const double prod[9] = {t[k].gp, t[k].go, t[k].co, t[k].cs, t[k].sop, t[k].spl, t[k].R0, t[k].ra, t[k].rb};
                std::string diff;
                for (int j = 0; j < 9; j++) {
                    char a[32], b[32];
                    snprintf(a, sizeof a, "%.5g", mine[j]); snprintf(b, sizeof b, "%.5g", prod[j]);
                    if (strcmp(a, b)) diff += std::string(" ") + kNames[j] + "=" + a + "（表 " + b + "）";
                }
                nrow++;
                if (!diff.empty()) nbad++;
                printf("  %-5s K=%-5d %s%s\n", R.name, t[k].K, diff.empty() ? "一致" : "不一致：", diff.c_str());
            }
        }
        printf("AWE-COEF-CHECK %s（%d 行のうち不一致 %d 行．ozaki2_awe_coef.inc の OZ2_AWE_COEF と tools/awe_plan.cpp coef）\n",
               nbad ? "FAIL" : "PASS", nrow, nbad);
        return nbad ? 1 : 0;
    }
    if (!want.empty() && want[0] == "table") {
        // ./awe_plan table ROUTE NUM [tail [表]]：効率順の表で ozaki2_choose_moduli_w の手順を模擬して最適化の結果と比べる
        if (want.size() < 3 || want.size() > 5) {
            fprintf(stderr, "usage: %s table route NUM [tail [m,m,...]]\n", argv[0]); return 2; }
        if (!known(want[1])) return 2;
        const Route& R = *route_by_name(want[1].c_str());
        // 整数の引数は 10 進の整数として全体を読み切れて範囲に入るときだけ受け付ける（src/common/oz_env.hpp．atoi では "16x" が
        // 16 に，"x" が 0 になる）．範囲の外は終了コード 2
        const int num = (int)oz_env::parse_i64("num", want[2].c_str(), 1, OZ2_MAXS);
        const int tail = want.size() > 3 ? (int)oz_env::parse_i64("tail", want[3].c_str(), 0, 1) : 0;
        Opt o; o.pmg = true;
        auto tb = efficiency_table(R, o);
        if (want.size() > 4) {       // 表を明示（カンマ区切り）．後ろに効率順の表を足す
            std::vector<int> t;
            for (const char* q = want[4].c_str();;) {
                const char* e = q; while (*e && *e != ',') e++;
                t.push_back((int)oz_env::parse_i64("表の法", std::string(q, e).c_str(), 2, R.cap));
                if (!*e) break;
                q = e + 1;
            }
            for (int m : tb) if (std::find(t.begin(), t.end(), m) == t.end()) t.push_back(m);
            tb = t;
        }
        printf("table(%s):", R.name); for (int m : tb) printf(" %d", m); printf("\n");
        for (int K : {3520, 7040, 14080}) {
            const Coef C = coef_for(R.name, K);
            Plan cur = plan_current(R, C, K, o);
            Plan pt = plan_table(R, K, tb, num, o, tail);
            bool to = false;
            Plan po = optimize(R, K, o, &to);
            printf("[%s K=%d]\n", R.name, K);
            print_plan("current", cur, cur.t[5]); print_plan("table", pt, cur.t[5]);
            print_plan(to ? "optimizer*" : "optimizer", po, cur.t[5]);
        }
        return g_invalid ? 1 : 0;
    }
    for (const auto& r : want) if (!known(r)) return 2;
    printf("# AWE 型の法ごとの運び方と法の集合の最適化（段の時間のモデルの予測．%% は較正の基準にした選び方（段の時間を"
           "実測したときの法の集合．plan_current．行 current(model)）のモデル値に対する差）\n");
    for (const Route& R : ROUTES) {
        if (!want.empty() && std::find(want.begin(), want.end(), std::string(R.name)) == want.end()) continue;
        for (int K : {3520, 7040, 14080}) {
            const Coef C = coef_for(R.name, K);
            Opt o0;
            Plan cur = plan_current(R, C, K, o0);
            const Fac f = factors_for(R, K);
            printf("[%s K=%d]  較正倍率 slice %.3f gemm %.3f 合成 %.3f recon %.3f%s\n", R.name, K, f.sl, f.ge, f.co, f.re,
                   f.meas ? "" : "（実測なし：3520 と 14080 の相乗平均）");
            print_plan("current(model)", cur, cur.t[5]);
            if (R.hostr.awe) print_plan("host(ozaki2_awe.h)", plan_host(R, C, K, o0), cur.t[5]);
            struct V { const char* tag; Opt o; };
            std::vector<V> vs;
            const bool b2 = R.balmax == 5;   // BFP2_b：既存は平衡 5 桁だけ
            { Opt o; o.single = false; o.kar = false; o.karf = false; o.bal = b2; o.bal5 = true; o.sq = !b2; o.wide = !b2;
              vs.push_back({"existing-modes", o}); }
            { Opt o; o.kar = true; o.karf = false; o.single = false; o.bal = b2; o.bal5 = true; o.sq = !b2; o.wide = !b2;
              vs.push_back({"+KAR(B<=d+1)", o}); }
            { Opt o; o.karf = false; o.bal = b2; o.bal5 = true; vs.push_back({"+KAR+SINGLE", o}); }
            { Opt o; vs.push_back({"+KARF+BAL(all)", o}); }
            { Opt o; o.pmg = true; vs.push_back({"+per-mod maxg", o}); }
            { Opt o; o.pmg = true; o.hex = true; vs.push_back({"+HEX(hyp)", o}); }
            { Opt o; o.pmg = true; o.hex = true; o.fuse = true; vs.push_back({"+fuse-single(hyp)", o}); }
            { Opt o; o.pmg = true; o.fuse = true; o.kar = false; o.karf = false; vs.push_back({"SINGLE+pmg+fuse", o}); }
            for (auto& v : vs) {
                bool to = false;
                Plan p = optimize(R, K, v.o, &to);
                print_plan((std::string(v.tag) + (to ? "*" : "")).c_str(), p, cur.t[5]);
                fflush(stdout);
            }
        }
    }
    if (g_invalid) fprintf(stderr, "ERROR: 法の運び方を再現できない計画が %d 個ある\n", g_invalid);
    return g_invalid ? 1 : 0;
}
