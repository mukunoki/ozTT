// SPDX-License-Identifier: Apache-2.0
//
// scheme II の再構成の「小数部の折り返し」を窓の端で確かめる【検査用】
// ---------------------------------------------------------------------------
//     make -C src crt_fold_check          （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     ./src/build/crt_fold_check [試行数=400] [seed=7] [table]
//
// 何を確かめるか
//   分数形 CRT の frac = C'/M は，窓の条件 |C'| < M/2 により (-1/2, 1/2) にある．
//   ozaki2/kernels/recon_crt.cpp は frac を FP32 3 本組 (a0,a1,a2) で積み，法ごとに a0 -= round(a0) を行う．
//   3 本組の値は [-1/2,1/2] から a1 + a2 だけはみ出しうるので，最後に畳み直す必要がある（(-1/2,1/2) の値は変えず，
//   はみ出しを ±1 で戻す．像は [-1/2,1/2]）．
//   本番の方式（出力の「本番」．出口で固定小数のまま畳む）では，カーネルは畳まず（法ごとの a0 -= round(a0) だけ），
//   ライタ（common/kernels/f64_tf.h の assemble_frac_f64）が 1 パスでも 2 パスでも 128 bit の固定小数のまま厳密に畳む．
//   比較のために模擬するもう 1 つの方式（出力の「FP32 畳み」．計算カーネルが FP32 で畳む）は，recenter_fraction が FP32 の和
//   fl(a0+a1) で判定して a0 に ±1 し，1 パス構成のライタは畳まない．この方式には 2 つの誤りがある：
//     (1) 見逃し：a0 = ±1/2 ちょうどに 2^-25 以下の a1 が同じ向きに乗ると fl(a0+a1) = ±1/2 で補正が入らず，
//         値は ±1/2 を越えたまま出ていく（真の frac から ±1 ずれる＝C が ±M*2^-2b*tau ずれる）．
//     (2) 丸め：a0 が (1/4,1/2) にあって a1 が fl(a0+a1) を 1/2 の上へ押すと，a0 - 1 が FP32 で厳密でなく
//         frac が 2^-25 丸まる（C の相対誤差 2^-24 程度．2 パス構成でも起きる）．
//
// どうやって確かめるか
//   recon_crt.cpp の演算列を 1 要素ぶん FP32 でそのまま模擬する（tools/oz_sim.h の crt_element．SFPU の演算は
//   checks/sfpu_tf_emu.hpp の Dst：加減乗算は 1 回ずつ最近接偶数に丸め，非正規化数は入力も出力も 0 に潰す．Rnd は
//   tt-llk の _round_even_ と同じ「|v| + 2^23 - 2^23」，Ltz は符号ビットを見る）．定数（法・1/m の断片・y_i・NSMALL_FRAG・
//   limb の本数）は本番と同じ関数（ozaki2_moduli.h と oz_sim.h の make_recon_kernels）で作り，
//   出てきた limb を本番と同じ assemble_frac_f64 に通す．FP32 畳みの演算列（recenter_fraction と，1 パスで
//   畳まない出口）も同じ入力で流し，FP32 畳みなら外れる件数を数える（失敗の経路が実在する証拠）．
//   入力は法の集合の積 M を __int128 で持ち，C' を窓の端に置いて作る：
//     (a) 1/2 - |frac| = 2^-j（j = 20..96）とその近傍の乱数．両符号．法ごとの剰余の代表元（剰余入力は
//         |r| <= m/2+1 の全候補，1 桁の法は |P| <= K*R^2，INT8 経路は |C_i| <= K*127^2）も振る．
//     (b) 窓の内側の一様乱数（折り返しと無関係な要素で出口が壊れていないことの確認）．
//   判定：1/2 - |frac| >= 2^-58（1 パス）・2^-90（2 パス）の入力のすべてで，本番の出口の FP64 が
//   真値 C'*2^-2b から 2 ulp 以内であること（それより窓の端に近い要素は frac の絶対精度 2^-66・2^-99 では
//   どちらの側か決まらないので情報として数えるだけ）．実装が保証する範囲はもっと内側である：法選択は
//   log2 M >= 2b + 1 + log2 W + OZ2_WIN_EPS（1e-9 bit）を課すので，窓の上界を満たす C' は
//   1/2 - |frac| >= (1 - 2^-1e-9)/2 ~ 2^-31.4 を満たす．128 bit 固定小数の折り返しには十分で，
//   FP32 の和による判定（2^-25 まで見えない）には足りない．これはどの法集合にも言える下限で，法集合ごとの実際の
//   余白 e = log2 M - (2b+1+log2 K) が大きければ，届く範囲はさらに内側（1/2 - |frac| >= (1 - 2^-e)/2）になる．
//   そこで件数は 2 通りに数える：「最小の余白なら届く範囲」（1/2 - |frac| >= 2^-31.4．余白が OZ2_WIN_EPS ちょうどの
//   法集合なら届く入力．FP32 畳みの演算列が外れることを確かめる感度の条件に使う）と，「この法集合の積和の上界以内」
//   （|C'| <= K*2^2b．正規化した入力 |a'|,|b'| <= 2^b の積和の上界）．後者は届くための必要条件であって，この範囲の
//   整数がどれも実際の積和で作れるわけではない（例えば K*2^2b - 1 は作れない：積 1 本は最大で 2^2b，それより小さい積は
//   高々 2^b(2^b - 1) なので，最大の和から 1 だけ減らせない．FP64 の入力を量子化した値という制約も入れていない）．
//   したがってその件数は「実際の入力で起こりうる件数」ではなく，上界で絞った件数である．
//   第 3 引数に table を付けると，既定の法集合の窓の余白（log2 M - (2b+1+log2 K)）の最小を K = 32..16384
//   （32 刻み）で表にする（段の時間のモデルで法を選ぶ経路は 1 回 1 秒弱かかるので既定では出さない）．累算の条件は満たすのに
//   法が取れない K と，定数入力の C' = K*2^2b が窓の外になる K（法選択が log2 M >= 2b+1+log2 K を課すので起きないはず）も
//   数え，1 件でもあれば終了コード 1 にする（飛ばして表を作ると，選び方の誤りを見落とす）．累算の条件で取れない K（経路の
//   K の上限の外．本番もデバイスを開く前に止める）は理由とともに示すだけにする．
//   経路は本番の 8 つである（ozaki2_split.h の経路の表 OZ2_ROUTES）．FP16（OZ2_ROUTE_FP16A）は運ぶ形式が BF16 と違い，段の時間の
//   モデルも別なので選ぶ法の集合が違いうる．K ごとに FP16 と BF16 の法の取り方（法・方式・b・CRT の係数・逆数の断片）を
//   比べ，同じなら BF16 の行が FP16 も兼ねる（その旨を表示する）．違えば FP16 の行を足して同じ検査を流す．
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <quadmath.h>
#include "oz_sim.h"

typedef __int128 i128;
using oz_sim::CrtKernel;
using oz_sim::Dst;

// 比較のために模擬する「FP32 畳み」の計算カーネル（本番のカーネルには無い）：最後の法の後に recenter_fraction で
// fl(a0+a1) を見て a0 に ±1 する（冒頭の (1)(2)）
static void recenter_fp32(Dst& D) {
    using namespace sfpu_emu::dst;
    D.Cpy(acc_hi, term); D.Rnd(term); D.Sub(acc_hi, term, acc_hi);
    D.Cpy(acc_hi, tmp0); D.Add(tmp0, acc_mid, tmp0);
    D.Cpy(tmp0, term); D.MulS(term, -1.f); D.AddS(term, 0.5f); D.Ltz(term);
    D.Sub(acc_hi, term, acc_hi);
    D.Cpy(tmp0, term); D.AddS(term, 0.5f); D.Ltz(term); D.Add(acc_hi, term, acc_hi);
}
// 入力：剰余入力（IN_RESIDUE=1）は FP32 の値，INT8 経路（IN_RESIDUE=0）は符号-絶対値 int32 の語．
// fp32_fold：FP32 畳みの計算カーネル（最後に recenter_fp32）
static void crt_element(const CrtKernel& k, const float* resf, const uint32_t* raw, uint32_t out[3], bool fp32_fold) {
    Dst D;
    oz_sim::crt_element(D, k, resf, raw);
    if (fp32_fold) recenter_fp32(D);
    oz_sim::crt_pack(D, k, out);
}

// 2 つの FP64 の間の ulp 数（符号が違っても溢れないよう __int128 で引き，int64 の上限で頭打ちにする．
// int64 で引くと，符号の違う大きな値どうし（±M ずれた答え）で溢れて負になり，外れを見逃す）
static inline long long ulp_dist(double a, double b) {
    int64_t x, y; std::memcpy(&x, &a, 8); std::memcpy(&y, &b, 8);
    const i128 xx = (x < 0) ? (i128)INT64_MIN - (i128)x : (i128)x;
    const i128 yy = (y < 0) ? (i128)INT64_MIN - (i128)y : (i128)y;
    const i128 d = (xx > yy) ? (xx - yy) : (yy - xx);
    return (d > (i128)INT64_MAX) ? INT64_MAX : (long long)d;
}
static i128 smod(i128 x, long m) {                // 対称剰余 [-(m-1)/2 .. m/2]
    i128 r = x % m; if (r < 0) r += m;
    if (r > m / 2) r -= m;
    return r;
}
static const i128 I128_MAX = (i128)(((unsigned __int128)1 << 127) - 1);
// 法の集合の積 M を __int128 で作る．掛ける前に I128_MAX / m と比べ，溢れるなら false を返す
// （多倍長整数は持たないので，その集合は扱えない．既定の法集合は K <= 16384 で 124 bit 未満）．
static bool mod_product(const std::vector<int>& m, i128& M) {
    M = 1;
    for (int mm : m) {
        if (mm < 2 || M > I128_MAX / mm) return false;
        M *= mm;
    }
    return true;
}
static double mod_log2(const std::vector<int>& m) {
    long double l = 0; for (int mm : m) l += log2l((long double)mm);
    return (double)l;
}

using Route = Oz2RouteRow;

struct Tally {
    long n = 0;          // 判定した件数
    long outside = 0;    // 本番のカーネルの limb の和が [-1/2,1/2] の外（出口が畳まねばならない件数）
    long wrong = 0;      // 本番の出口が真値から 2 ulp を超えて外れた件数（0 でなければ FAIL）
    long ulp1 = 0;       // 1〜2 ulp の差
    long f32_miss = 0;   // FP32 畳み：recenter_fraction の後も limb の和が [-1/2,1/2] の外（見逃し (1)）
    long f32_wrong = 0;  // FP32 畳みの演算列なら 2 ulp を超えて外れた件数（(1) の ±M と (2) の 2^-24 の両方）
    long nreach = 0, f32_wrong_reach = 0;   // うち最小の余白なら届く範囲（1/2 - |frac| >= (1 - 2^-OZ2_WIN_EPS)/2 ~ 2^-31.4）
    long nset = 0, f32_wrong_set = 0;       // うちこの法集合の積和の上界以内（|C'| <= K*2^2b．届くための必要条件）
    double worst_ex = 0; // 本番のカーネルの limb の和の |V| - 1/2 の最大（はみ出しの大きさ．出口が 1 回で畳める範囲か）
    long nx = 0, wrongx = 0;                      // 窓の余白の外（情報）
    void add(const Tally& o) { n += o.n; outside += o.outside; wrong += o.wrong; ulp1 += o.ulp1;
                               f32_miss += o.f32_miss; f32_wrong += o.f32_wrong;
                               nreach += o.nreach; f32_wrong_reach += o.f32_wrong_reach;
                               nset += o.nset; f32_wrong_set += o.f32_wrong_set;
                               worst_ex = std::max(worst_ex, o.worst_ex); nx += o.nx; wrongx += o.wrongx; }
};

// 1 つの C' について，代表元を ntry 通り振って 1 パスと 2 パスを流す
struct Ctx {
    const Ozaki2Moduli* mo; Route rt; uint32_t K;
    CrtKernel k0, k1, k2;
    uint32_t Whi, Wlo; int32_t qw;
    long double KMl;
    i128 M;
    std::mt19937_64* rng;
};
// 法の集合 mo の 3 通りのカーネル（1 パス・2 パスの第 1・第 2）と出口の定数を組む（3 つのモードで共通）．
// 法の積 M は呼び出し側が mod_product で確かめてから入れる
static Ctx make_ctx(const Ozaki2Moduli& mo, const Route& rt, uint32_t K, std::mt19937_64* rng) {
    Ctx c; c.mo = &mo; c.rt = rt; c.K = K; c.rng = rng; c.M = 0;
    const oz_sim::ReconKernels ks = oz_sim::make_recon_kernels(mo, oz2_in_residue(rt.r));
    c.k0 = ks.one; c.k1 = ks.first; c.k2 = ks.second;
    const OzScale64 w = oz_split_scale_ld(mo.KMl);   // ドライバの make_recon_scale と同じ分け方（common/oz_bits.h）
    c.Whi = w.hi; c.Wlo = w.lo; c.qw = w.q; c.KMl = mo.KMl;
    return c;
}

static __float128 limb_sum(const uint32_t* l, int n) {
    __float128 v = 0;
    for (int i = 0; i < n; i++) v += (__float128)bits_f32(l[i]);
    return v;
}

// 窓の条件（法選択の OZ2_WIN_EPS）が許す 1/2 - |frac| の下限（log2 M >= need + OZ2_WIN_EPS で，C' が上界ちょうどのとき）
static const __float128 kReachMin = 0.5Q * (1.0Q - exp2q(-(__float128)OZ2_WIN_EPS));
// C' を 2 通りに分類する（冒頭の説明）．reach：最小の余白なら届く範囲，inset：この法集合の積和の上界以内（|C'| <= K*2^2b．
// 届くための必要条件で，十分ではない）．
// c.M（法の積）を入れてから呼ぶ
static void classify(const Ctx& c, i128 Cp, bool& reach, bool& inset) {
    const i128 a = Cp < 0 ? -Cp : Cp;
    reach = (0.5Q - (__float128)a / (__float128)c.M) >= kReachMin;
    const int sh = 2 * c.mo->b, kb = 32 - __builtin_clz(c.K | 1u);
    // K*2^2b < M/2 <= I128_MAX/2 なので M が __int128 に収まる集合では溢れない．念のため幅を見てから寄せる
    inset = (kb + sh <= 126) ? (a <= ((i128)c.K << sh)) : true;
}
static void run_one(Ctx& c, i128 Cp, bool legal1, bool legal2, Tally& t1, Tally& t2, int ntry) {
    const Ozaki2Moduli& mo = *c.mo;
    const int S = mo.s;
    bool reach, inset;
    classify(c, Cp, reach, inset);
    // 真値：C = C' * 2^-2b（KM = M*2^-2b，frac = C'/M）
    const double ref = (double)ldexpq((__float128)Cp, -2 * mo.b);
    const long R = c.rt.r.digit_bound ? c.rt.r.digit_bound : 127;
    std::vector<float> resf(S); std::vector<uint32_t> raw(S);
    for (int tr = 0; tr < ntry; tr++) {
        for (int i = 0; i < S; i++) {
            const long m = mo.m[i];
            const i128 r = smod(Cp, m);
            i128 x = r;
            if (!oz2_in_residue(c.rt.r) || mo.mode[i] == OZ2_SPLIT_ONE) {
                // GEMM の出力（INT8 は |C_i| <= K*127^2，1 桁の法は |P| <= K*R^2）
                const i128 lim = (i128)c.K * R * R;
                const i128 jmax = (lim - (r < 0 ? -r : r)) / m;
                if (tr > 0 && jmax > 0) {
                    const i128 j = (i128)((*c.rng)() % (uint64_t)(2 * jmax + 1)) - jmax;
                    x = r + j * m;
                }
            } else if (tr > 0) {
                // 合成段の出力は総和（< 2^22．ホストの法選択が課す）を ModRound で畳んだ値で，一般の保証の内側なので
                // 対称剰余そのものである．ここでは代表元に依らないことも見るため，対称剰余から 1 はみ出る値（|x| <= m/2+1）も
                // 振る（本番には現れない）．r_i*y_i の ModRound の出力 k_i は，対称剰余の x でも y_i によっては floor(m/2) を
                // 1 超えうる（tools/modround_check.cpp の (a')．既定の集合では起きない）
                i128 cand[3]; int nc = 0;
                for (int d = -1; d <= 1; d++) {
                    const i128 y = r + d * m;
                    if ((y < 0 ? -y : y) <= m / 2 + 1) cand[nc++] = y;
                }
                x = cand[(*c.rng)() % nc];
            }
            if (oz2_in_residue(c.rt.r)) resf[i] = (float)(long long)x;
            else {
                const long long v = (long long)x, a = v < 0 ? -v : v;
                raw[i] = (v < 0 ? 0x80000000u : 0u) | (uint32_t)a;
            }
        }
        // 1 パス
        {
            uint32_t lb[3], lo[3];
            crt_element(c.k0, resf.data(), raw.data(), lb, false);
            crt_element(c.k0, resf.data(), raw.data(), lo, true);
            const __float128 v = limb_sum(lb, 3), vo = limb_sum(lo, 3);
            const double got = bits_f64(assemble_frac_f64(lb, 3, 0, c.Whi, c.Wlo, c.qw));
            const long long u = ulp_dist(got, ref);
            // FP32 畳みの 1 パスの出口は畳まずに limb の和をそのまま KM 倍する（__float128 で足して掛ける．
            // 丸めは 1 回余分に入るが，外れたときの差は 2^-24 相対以上なので 2 ulp の判定を左右しない）
            const double f32v = (double)(vo * (__float128)c.KMl);
            if (legal1) {
                t1.n++;
                if (v > 0.5Q || v < -0.5Q) t1.outside++;
                t1.worst_ex = std::max(t1.worst_ex, (double)(fabsq(v) - 0.5Q));
                if (vo > 0.5Q || vo < -0.5Q) t1.f32_miss++;
                if (ulp_dist(f32v, ref) > 2) { t1.f32_wrong++; if (reach) t1.f32_wrong_reach++; if (inset) t1.f32_wrong_set++; }
                if (reach) t1.nreach++;
                if (inset) t1.nset++;
                if (u > 2) { t1.wrong++;
                    if (t1.wrong <= 3) printf("    1 パス NG %s K=%u V=%.20g got=%.17g ref=%.17g\n",
                                              c.rt.name, c.K, (double)v, got, ref); }
                else if (u) t1.ulp1++;
            } else { t1.nx++; if (u > 2) t1.wrongx++; }
        }
        // 2パス：本番は第1パスの3語を128bit固定小数としてDRAMに残す．ここでは6語をまとめる等価な模型も検査する．
        // FP32 畳みでも 2 パスの出口は畳む（外れるのは recenter_fraction の丸め (2) のときだけ）
        {
            uint32_t p1[3], p2[3], q1[3], q2[3];
            crt_element(c.k1, resf.data(), raw.data(), p1, false);
            crt_element(c.k2, resf.data(), raw.data(), p2, false);
            crt_element(c.k1, resf.data(), raw.data(), q1, true);
            crt_element(c.k2, resf.data(), raw.data(), q2, true);
            const uint32_t l6[6] = {p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]};
            const uint32_t o6[6] = {q2[0], q2[1], q2[2], q1[0], q1[1], q1[2]};
            const __float128 v = limb_sum(l6, 6), vo = limb_sum(q1, 3);
            const double got = bits_f64(assemble_frac_f64(l6, 6, 0, c.Whi, c.Wlo, c.qw));
            const double f32v = bits_f64(assemble_frac_f64(o6, 6, 0, c.Whi, c.Wlo, c.qw));
            // デバイスの 2 パスは，第 1 パスが自分の 3 本を固定小数へ足して残し，第 2 パスがそこへ自分の 3 本を
            // 足して畳む（recon_writer.cpp・recon_reader.cpp）．6 本をまとめて足すのとビット単位で同じことを確かめる
            {
                A128 a{{0, 0, 0, 0}};
                a128_add_limbs(a, p1, 3);
                a128_add_limbs(a, p2, 3);
                const uint64_t split = a128_frac_to_f64(a, 0, c.Whi, c.Wlo, c.qw);
                const uint64_t whole = assemble_frac_f64(l6, 6, 0, c.Whi, c.Wlo, c.qw);
                if (split != whole) { t2.wrong++;
                    if (t2.wrong <= 3) printf("    2 パス NG（第 1 パスで先に足す形が 6 本をまとめた形と違う）%s K=%u\n",
                                              c.rt.name, c.K); }
            }
            const long long u = ulp_dist(got, ref);
            if (legal2) {
                t2.n++;
                if (v > 0.5Q || v < -0.5Q) t2.outside++;
                t2.worst_ex = std::max(t2.worst_ex, (double)(fabsq(v) - 0.5Q));
                if (vo > 0.5Q || vo < -0.5Q) t2.f32_miss++;
                if (ulp_dist(f32v, ref) > 2) { t2.f32_wrong++; if (reach) t2.f32_wrong_reach++; if (inset) t2.f32_wrong_set++; }
                if (reach) t2.nreach++;
                if (inset) t2.nset++;
                if (u > 2) { t2.wrong++;
                    if (t2.wrong <= 3) printf("    2 パス NG %s K=%u got=%.17g ref=%.17g\n",
                                              c.rt.name, c.K, got, ref); }
                else if (u) t2.ulp1++;
            } else { t2.nx++; if (u > 2) t2.wrongx++; }
        }
    }
}

// 定数入力（OZ_CONST=1．全要素 1 で C' = K*2^2b）の 1 要素を，法の集合を与えて模擬する（実機の境界検査の設計用）．
//   crt_fold_check const <経路 INT8|FP8|TF32|BF16|FP16|BFP8|BFP4|BFP2> <K> <法,法,...>
// 本番と FP32 畳みの 1 パスの出口と真値を表示する．本番が 2 ulp を超えて外れたら終了コード 1．
// 法の積が __int128 に収まらない集合（例えば TF32・K=32 に OZ2_TABLE_B8 の 16 本，約 167 bit）と，
// 型の幅を超える K・法は終了コード 2 で断る．
static int const_mode(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "ERROR: const <経路> <K> <法,...>\n"); return 2; }
    const Route* rt = oz_sim::route2(argv[2]);
    if (!rt) { fprintf(stderr, "ERROR: 経路 %s は無い\n", argv[2]); return 2; }
    const Oz2Route& R = rt->r;
    // K と法は型を縮める前に範囲を見る（K は uint32_t，法は 2..経路の法の上限 mod_cap の int）
    unsigned long long K = 0;   // 負号はどこにあっても拒む（oz_env::to_u64）
    if (!oz_env::to_u64(argv[3], K) || K < 32 || K % 32 || K > UINT32_MAX) {
        fprintf(stderr, "ERROR: K=%s は 32 の倍数で 32--%u の整数ではない\n", argv[3], UINT32_MAX - 31u); return 2; }
    std::vector<int> force;
    for (const char* q = argv[4];;) {
        char* end = nullptr; errno = 0;
        const long v = (*q >= '0' && *q <= '9') ? std::strtol(q, &end, 10) : -1;   // 数字で始まらなければ -1（下で断る）
        if (end == q || errno == ERANGE || v < 2 || v > R.mod_cap || (*end && *end != ',')) {
            fprintf(stderr, "ERROR: 法 %s は 2--%d の整数のカンマ区切りではない\n", argv[4], R.mod_cap); return 2; }
        force.push_back((int)v);
        if (!*end) break;
        q = end + 1;
    }
    Ozaki2Moduli mo = ozaki2_choose_moduli_w((uint32_t)K, (int)force.size(), R.mod_cap, OZ2_ACC_BITS, R.digit_bound, OZ2_B_DEFAULT,
                                             false, 0.0, R.nd(), R.sparse_max, R.awe, &force);
    if (!mo.ok) { printf("法の集合が使えない（ok=0）\n"); return 2; }
    std::mt19937_64 rng(1);
    Ctx c = make_ctx(mo, *rt, (uint32_t)K, &rng);
    if (!mod_product(mo.m, c.M)) {
        printf("法の積 M（log2 M = %.1f）が __int128 に収まらない（この検査は扱えない）\n", mod_log2(mo.m));
        return 2;
    }
    // C' = K*2^2b．2 C' < M（窓）を満たすかを，C' を作る前に溢れない形で見る（M <= I128_MAX なので，
    // K > I128_MAX >> (2b+1) なら 2 C' > M で窓の外）
    if ((i128)K > (I128_MAX >> (2 * mo.b + 1))) {
        printf("経路 %s K=%llu b=%d：定数入力の C' = K*2^2b が窓の外（2 C' > M）\n", rt->name, K, mo.b);
        return 2;
    }
    const i128 Cp = (i128)K << (2 * mo.b);
    const double e = mod_log2(mo.m) - (2.0 * mo.b + 1.0 + std::log2((double)K));
    const __float128 dist = 0.5Q - (__float128)Cp / (__float128)c.M;
    printf("経路 %s K=%llu 法 {", rt->name, K);
    for (int i = 0; i < mo.s; i++) printf("%s%d(mode %d)", i ? "," : "", mo.m[i], mo.mode[i]);
    printf("} b=%d 窓の余白 e=%.3e bit，定数入力の 1/2 - |frac| = 2^%.2f%s\n", mo.b, e, (double)log2q(dist),
           2 * Cp < c.M ? "" : "（窓の外）");
    if (2 * Cp >= c.M) return 2;
    int rc = 0;
    for (int sgn = 0; sgn < 2; sgn++) {
        const i128 x = sgn ? -Cp : Cp;
        std::vector<float> resf(mo.s); std::vector<uint32_t> raw(mo.s);
        for (int i = 0; i < mo.s; i++) {
            const i128 r = smod(x, mo.m[i]);
            if (oz2_in_residue(rt->r)) resf[i] = (float)(long long)r;
            else { const long long v = (long long)r, a = v < 0 ? -v : v; raw[i] = (v < 0 ? 0x80000000u : 0u) | (uint32_t)a; }
        }
        uint32_t lb[3], lo[3];
        crt_element(c.k0, resf.data(), raw.data(), lb, false);
        crt_element(c.k0, resf.data(), raw.data(), lo, true);
        const double ref = (double)ldexpq((__float128)x, -2 * mo.b);
        const double got = bits_f64(assemble_frac_f64(lb, 3, 0, c.Whi, c.Wlo, c.qw));
        const double f32v = (double)(limb_sum(lo, 3) * (__float128)c.KMl);
        // C' = K*2^2b は K が 2 のべきでなければ 2 のべきではないので，K と指数を分けてそのまま出す
        printf("  C'=%s%llu*2^%d（K*2^2b）：真値 %.17g  本番 %.17g（%lld ulp）  FP32 畳み %.17g（%lld ulp．limb の和 %.12g）\n",
               sgn ? "-" : "+", K, 2 * mo.b, ref, got, ulp_dist(got, ref), f32v, ulp_dist(f32v, ref),
               (double)limb_sum(lo, 3));
        if (ulp_dist(got, ref) > 2) rc = 1;
    }
    return rc;
}

// ---- 窓の端の検査（冒頭の「どうやって確かめるか」の (a)・(b)）--------------------------------------------
// 本番の経路（ozaki2_split.h の OZ2_ROUTES．FP16 は最後で，K ごとに BF16 と同じ法の取り方なら BF16 の行で兼ねる．冒頭の説明．
// 乱数の列は前の経路から続けて引くので，FP16 を最後に置けば他の行の入力は FP16 の有無で変わらない）．rng は呼び出し側の
// 1 本を経路・K・試行の順に続けて引く．途中で止めるとき（法が取れない 1，法の積が __int128 に収まらない 2）はその終了コードを，
// 最後まで回せば 0 を返す
static int window_sweep(int iters, long long seedv, std::mt19937_64& rng, Tally& all1, Tally& all2) {
    const uint32_t Ks[] = {3520, 14080};
    printf("===== 分数形 CRT の折り返し（窓の端．試行 %d，seed %lld）=====\n", iters, seedv);
    printf("列：件数／limb の和が ±1/2 の外（出口が畳む）／|V|-1/2 の最大／FP32 畳みの見逃し (1)／FP32 畳みなら NG／本番の NG／1〜2 ulp\n");
    printf("%-5s %-6s %-3s | %7s %6s %9s %6s %6s %5s %5s | %7s %6s %9s %6s %6s %5s\n", "経路", "K", "s",
           "1P件数", "外", "はみ出し", "F32見逃", "F32NG", "NG", "1ulp", "2P件数", "外", "はみ出し", "F32見逃", "F32NG", "NG");
    for (const Route& rt : OZ2_ROUTES) for (uint32_t K : Ks) {
        if (oz_sim::fp16_same_as_bf16(rt.r, K)) { printf("%-5s %-6u     （BF16 と同じ法の集合・定数なので BF16 の行が兼ねる）\n", rt.name, K); continue; }
        Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, rt.r);
        if (!mo.ok) { printf("  %s K=%u 法が取れない\n", rt.name, K); return 1; }
        Ctx c = make_ctx(mo, rt, K, &rng);
        if (!mod_product(mo.m, c.M)) {
            printf("  %s K=%u 法の積（log2 M = %.1f）が __int128 に収まらない\n", rt.name, K, mod_log2(mo.m)); return 2; }
        const __float128 Mq = (__float128)c.M;
        Tally t1, t2;
        auto at_dist = [&](__float128 d, int sgn) {  // 1/2 - |frac| ~ d の C'
            i128 a = (i128)(Mq * ((__float128)0.5 - d));
            const i128 cmax = (c.M - 1) / 2;         // |C'| < M/2 の最大
            if (a > cmax) a = cmax;
            if (a < 0) a = 0;
            const i128 Cp = sgn ? -a : a;
            // 実際の 1/2 - |frac|（整数へ丸めたあと）で合法かを決める（届く範囲かは run_one が classify で見る）
            const __float128 dist = (__float128)0.5 - (__float128)a / Mq;
            const bool legal1 = dist >= ldexpq(1.0Q, -58), legal2 = dist >= ldexpq(1.0Q, -90);
            run_one(c, Cp, legal1, legal2, t1, t2, 4);
        };
        // (a) 1/2 - |frac| = 2^-j（両符号）と，[2^-j, 2^-j+1) の乱数
        for (int j = 20; j <= 96; j++)
            for (int sgn = 0; sgn < 2; sgn++) {
                at_dist(ldexpq(1.0Q, -j), sgn);
                for (int r = 0; r < iters / 20 + 1; r++) {
                    const double u = std::ldexp((double)(rng() >> 11), -53);   // [0,1)
                    at_dist(ldexpq((__float128)(1.0 + u), -j), sgn);
                }
            }
        // (a') 窓の端そのもの（|C'| = M/2 - 1/2 - t．1/2 - |frac| ~ 2^-124：窓の余白の外なので情報のみ）
        for (int tt = 0; tt < 16; tt++)
            for (int sgn = 0; sgn < 2; sgn++) {
                const i128 a = (c.M - 1) / 2 - tt;
                run_one(c, sgn ? -a : a, false, false, t1, t2, 2);
            }
        // (b) 一様乱数（|frac| >= 2^-8 のものだけ判定．小さい frac は絶対精度で相対が落ちるのが仕様）
        for (int r = 0; r < iters; r++) {
            // 128 bit の乱数語は符号なしのまま M で割った余り（0 <= w mod M < M <= I128_MAX）にしてから符号付きへ移す
            // （符号付きの語を絶対値にすると，最小値 -2^127 の符号反転が溢れる）
            const uint64_t hi = rng(), lo = rng();
            const unsigned __int128 w = ((unsigned __int128)hi << 64) | lo;
            i128 Cp = (i128)(w % (unsigned __int128)c.M) - c.M / 2;
            if (Cp <= -c.M / 2 || (Cp >= 0 && 2 * Cp >= c.M)) continue;
            const __float128 f = (__float128)Cp / Mq;
            if (fabsq(f) < ldexpq(1.0Q, -8)) continue;
            run_one(c, Cp, true, true, t1, t2, 1);
        }
        auto ex = [](double v) { return v > 0 ? std::log2(v) : -999.0; };
        printf("%-5s %-6u %-3d | %7ld %6ld %8s%.1f %6ld %6ld %5ld %5ld | %7ld %6ld %8s%.1f %6ld %6ld %5ld   （余白外：1P NG %ld/%ld，2P NG %ld/%ld）\n",
               rt.name, K, mo.s, t1.n, t1.outside, "2^", ex(t1.worst_ex), t1.f32_miss, t1.f32_wrong, t1.wrong, t1.ulp1,
               t2.n, t2.outside, "2^", ex(t2.worst_ex), t2.f32_miss, t2.f32_wrong, t2.wrong,
               t1.wrongx, t1.nx, t2.wrongx, t2.nx);
        all1.add(t1); all2.add(t2);
    }
    return 0;
}

// ---- 既定の法集合の窓の余白の表（第 3 引数 table）------------------------------------------------------
// 既定の法集合の窓の余白：e = log2 M - (2b + 1 + log2 K)．上界近くの C' では 1/2 - |frac| ~ e ln2 / 2．
// FP32 畳みの見逃し (1) は 1/2 - |frac| < 2^-25 程度（e < 8.6e-8 bit），丸め (2) は 1/2 - |frac| < |a1|
// （2^-25〜2^-18 程度．e < 約 1e-5 bit）の法集合に上界近くの C' が来たときに起こりうる．
// table_bad に法が取れない K と定数入力が窓の外になる K の数を足し，定数入力での本番の NG を all1.wrong に足す．
// 法の積が __int128 に収まらなければ 2 を返す（最後まで回せば 0）
static int margin_table(std::mt19937_64& rng, Tally& all1, long& table_bad) {
    printf("\n===== 既定の法集合の窓の余白 e = log2 M - (2b+1+log2 K)（K = 32..16384，32 刻み）=====\n");
    for (const Route& rt : OZ2_ROUTES) {
        double emin = 1e9; uint32_t kmin = 0; int bmin = 0;
        std::string f32ng;   // 定数入力（全要素 1．C' = K*2^2b が窓の上界ちょうど）で FP32 畳みの演算列が外れる K
        long nconst = 0, prodng = 0;
        long nsame = 0;      // FP16：BF16 と同じ法の取り方なので BF16 の行が兼ねた K の数
        long ncov = 0;       // 窓の余白を測った K の数
        // 累算の条件（acc_ok．桁積・合成の総和などが FP32 の無誤差整数に収まらない）で法が取れない K は，その経路の K の
        // 上限の外で，本番もデバイスを開く前に理由を示して止める（ozaki2_main.cpp．BF16・FP16・TF32 は K=16384 から）ので
        // 数えて示すだけにする．それ以外で法が取れない K（窓を覆えない）と，定数入力 C' = K*2^2b が窓の外になる K は起きないはず
        std::string limit, unsup, outwin;
        for (uint32_t K = 32; K <= 16384; K += 32) {
            if (oz_sim::fp16_same_as_bf16(rt.r, K)) { nsame++; continue; }
            Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, rt.r);
            if (!mo.ok && !mo.acc_ok) { limit += (limit.empty() ? "" : " ") + std::to_string(K) + "(" + mo.acc_fails.front().why + ")"; continue; }
            if (!mo.ok) { unsup += (unsup.empty() ? "" : " ") + std::to_string(K); table_bad++; continue; }
            ncov++;
            i128 M;
            if (!mod_product(mo.m, M)) {
                printf("  %s K=%u 法の積（log2 M = %.1f）が __int128 に収まらない\n", rt.name, K, mod_log2(mo.m)); return 2; }
            const double e = (double)(log2l((long double)M) - (2.0L * mo.b + 1.0L + log2l((long double)K)));
            if (e < emin) { emin = e; kmin = K; bmin = mo.b; }
            // OZ_CONST=1（全要素 1）は正規化後の要素が 2^b ちょうどになり，C' = K*2^2b（窓の上界そのもの）になる．
            // 符号は A・B の定数の符号で両方出る．代表元は対称剰余（定数入力の k_i は代表元に依らない）
            Ctx c = make_ctx(mo, rt, K, &rng); c.M = M;
            const i128 Cp = (i128)K << (2 * mo.b);
            if (2 * Cp >= M) {                       // 窓の外（法選択が許さないはずなので，起きれば選び方の誤り）
                outwin += (outwin.empty() ? "" : " ") + std::to_string(K); table_bad++; continue; }
            for (int sgn = 0; sgn < 2; sgn++) {
                Tally a1, a2;
                run_one(c, sgn ? -Cp : Cp, true, true, a1, a2, 1);
                nconst++;
                prodng += a1.wrong + a2.wrong;
                if (a1.f32_wrong || a2.f32_wrong)
                    f32ng += (f32ng.empty() ? "" : " ") + std::to_string(K) + (sgn ? "-" : "+")
                             + (a1.f32_miss ? "(1)" : "(2)") + (a2.f32_wrong ? "/2P" : "");
            }
        }
        if (nsame) printf("  %-4s  K = 32..16384 の %ld 通りで BF16 と同じ法の集合・定数（BF16 の行が兼ねる）\n", rt.name, nsame);
        if (!limit.empty()) printf("  %-4s  累算の条件で法が取れない K（経路の K の上限の外．括弧は落ちた条件）：%s\n", rt.name, limit.c_str());
        if (!unsup.empty()) printf("  %-4s  NG: 法が取れない K（累算の条件は満たすのに窓を覆えない）：%s\n", rt.name, unsup.c_str());
        if (!outwin.empty()) printf("  %-4s  NG: 定数入力 C' = K*2^2b が窓の外になる K：%s\n", rt.name, outwin.c_str());
        if (kmin == 0) continue;   // 全部の K を BF16 の行が兼ねた（または法が取れない）
        printf("  %-4s  窓の余白を測った K %ld 通り\n", rt.name, ncov);
        printf("  %-4s  最小 e = %.3e bit（K=%u，b=%d．1/2 - |frac| >= 2^%.1f）%s\n", rt.name, emin, kmin, bmin,
               std::log2(0.5 * (1.0 - std::exp2(-emin))),
               emin < 8.6e-8 ? "  <-- FP32 畳みの見逃し (1) が起こりうる余白" : "");
        printf("        定数入力 %ld 通りで本番の NG %ld，FP32 畳みなら外れた K（+/- は C' の符号，(1) 見逃し (2) 丸め，/2P は 2 パスも）：%s\n",
               nconst, prodng, f32ng.empty() ? "なし" : f32ng.c_str());
        if (prodng) all1.wrong += prodng;
    }
    return 0;
}

// ---- 合計と判定（終了コード）--------------------------------------------------------------------------
static int report(const Tally& all1, const Tally& all2, long table_bad) {
    const bool ok = all1.wrong == 0 && all2.wrong == 0 && all1.n > 0 && all2.n > 0 && table_bad == 0;
    if (table_bad) printf("NG: table で法が取れない K か窓の外になる K が %ld 通りある\n", table_bad);
    printf("\n合計：1 パス %ld 件（limb の和が ±1/2 の外 %ld 件，はみ出しの最大 2^%.1f．FP32 畳みなら見逃し %ld 件・NG %ld 件．"
           "本番の NG %ld 件），2 パス %ld 件（FP32 畳みなら NG %ld 件．本番の NG %ld 件）\n",
           all1.n, all1.outside, all1.worst_ex > 0 ? std::log2(all1.worst_ex) : -999.0, all1.f32_miss, all1.f32_wrong,
           all1.wrong, all2.n, all2.f32_wrong, all2.wrong);
    printf("うち最小の余白（OZ2_WIN_EPS）の法集合なら届く範囲（1/2 - |frac| >= 2^%.1f）：1 パス %ld 件で FP32 畳みなら NG %ld 件，"
           "2 パス %ld 件で FP32 畳みなら NG %ld 件\n",
           (double)log2q(kReachMin), all1.nreach, all1.f32_wrong_reach, all2.nreach, all2.f32_wrong_reach);
    printf("うちこの法集合の積和の上界以内（|C'| <= K*2^2b．届くための必要条件で，件数は実際の入力で起こりうる件数ではない）："
           "1 パス %ld 件で FP32 畳みなら NG %ld 件，2 パス %ld 件で FP32 畳みなら NG %ld 件\n",
           all1.nset, all1.f32_wrong_set, all2.nset, all2.f32_wrong_set);
    // FP32 畳みの演算列が実際に外れること（検査が失敗の経路を捉えていることの確認）も条件にする．見るのは最小の余白なら
    // 届く範囲（ストレスの条件）で，既定の法集合の余白は大きいので，この法集合の積和の上界以内では外れなくてよい
    const bool sens = all1.f32_wrong_reach > 0;
    if (!sens) printf("NG: FP32 畳みの演算列が最小の余白なら届く範囲で 1 件も外れない（検査が失敗の経路を捉えていない）\n");
    printf("CRT-FOLD-CHECK %s\n", (ok && sens) ? "PASS（出口の 128 bit 固定小数の畳み込みで，窓の端でも真値から 2 ulp 以内）"
                                               : "FAIL");
    return (ok && sens) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "const") == 0) return const_mode(argc, argv);
    if (argc > 4 || (argc == 4 && std::strcmp(argv[3], "table") != 0)) {
        fprintf(stderr, "ERROR: 引数は 試行数 seed [table] の 3 個まで\n"); return 2; }
    // 位置引数の読み方は src/common/oz_env.hpp（10 進の整数として全体を読み切る．範囲の外は終了コード 2）
    const int iters = (int)oz_env::arg_i64(argc, argv, 1, 400, 1, 100000000, "試行数");
    const long long seedv = oz_env::arg_i64(argc, argv, 2, 7, 0, UINT32_MAX, "seed");
    const bool table = argc == 4;
    std::mt19937_64 rng((uint64_t)seedv);
    Tally all1, all2;
    if (const int rc = window_sweep(iters, seedv, rng, all1, all2)) return rc;
    long table_bad = 0;      // 法が取れない K と，定数入力が窓の外になる K の合計（0 でなければ FAIL）
    if (table)
        if (const int rc = margin_table(rng, all1, table_bad)) return rc;
    return report(all1, all2, table_bad);
}
