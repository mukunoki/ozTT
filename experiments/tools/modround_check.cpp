// modround_check.cpp -- ModRound を実際の法・係数の全域で総当たりする
// ---------------------------------------------------------------------------
// リポジトリ直下で（コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作り，scripts/selftest.sh の hostcheck もこれを回す）
//     make -C src modround_check
//     ./src/build/modround_check [K]      （引数は K の 1 個まで．それより多ければ終了コード 2）
//
// なぜ要るか：sfpu_tf_proc.h の ModRound が保証するのは |d| < 2^22 の範囲だけである
// （fl(d*rinv) の相対誤差 2^-23 が d/m と半整数の距離 1/(2m) を超えない条件）．
// ところが次の呼び出しのうち (a) と (c) は，法を 4096 まで取る BF16 / BFP8 / FP16 / FP32
// 経路でこの範囲を出る：
//
//   (a) r_i * y_i          |.| < m^2/2 = 2^23   （recon_crt.cpp．m=4096．1 桁の法は r_i が RAWMASK の出力で
//                          |r_i| <= floor(m/2)+2 だが，m <= 65 なので |.| < 2^22 の一般の保証の内側）
//   (b) (chi mod m)*(2^16 mod m) + (clo mod m)   |.| <= m^2/2 + m/2   （同．符号-絶対値 int32 の
//       2 平面復号で，INT8 経路（m <= 255）だけが通る．|.| < 2^15 なので一般の保証の内側だが，
//       安く回せるので確かめておく．桁分割の経路は剰余を FP32 で受けるので通らない）
//   (c) スライサの最後の ModRound（ozaki2/kernels/oz2_digits.h の oz2_residue）
//         S = d0 + Σ_j t_j*c_j,  |d0| <= 2^19 強,  |t_j| <= floor(m/2),  c_j = 2^(20(j+1)) mod m ∈ [0,m)
//       m=3913 の 3 桁で |S| は 1.6e7 に達し，2^22 どころか 2^24 に迫る．
//
// 一般の上界としては足りないが，実際に現れるのは「その法のその係数」に限られる．
// (a)(b) は既定の法集合について総当たりする．(a) の r_i は実際に現れる値の全部で，合成段の出力（総和 < 2^22 の ModRound）は
// 対称剰余なので m+1 通り，1 桁の法（合成段を通らず再構成の RAWMASK が畳む）は (e) の上界 floor(m/2)+2 までの m+5 通りである．
// (b) は chi mod m と clo mod m の組で (m/2+1)^2 通りある（INT8 経路だけなので法 1 本あたり 1.6 万通り）．
// (b) を桁分割の経路へ回さないのは，実際には通らない呼び出しのためである（m=4096 なら法 1 本あたり
// 420 万通り，合計約 1 億通りになる）．
// (a') (b') は法の集合を既定に限らない．y_i = (Π_{j≠i} m_j)^{-1} mod m_i は他の法で決まるので，集合が変われば（OZ2_MODULI，
// OZ2_B・OZ2_FP64 で本数が変わる，大きい K の INT8 経路の表の延長など）m_i と互いに素な 1..m_i-1 のどれにもなりうる．
// そこで経路ごとに上限以下の使える法のすべて（(c) と同じ集合）で，(a) と同じ範囲の r_i と互いに素なすべての y について
// ModRound(r_i*y) を確かめる（(b') は INT8 経路の 255 以下の全法で (b) と同じ組を回す）．積は |d| <= R(m-1) < 2^23 なので，
// (c) と同じく半整数の境界の近傍 ±W だけを総当たりすれば漏れがない（境界から離れた d は round が正しい整数を選び，
// 結果は |k| <= floor(m/2) の正しい対称剰余になる）．境界の近傍で |k| が floor(m/2) を超える d は，|d| = |r|*y と分解できる
// （|r| <= R，1 <= y < m，gcd(y,m) = 1．実際に現れうる積）ものだけを数える．
// (a)(a') の判定の |k_i| の上界は，続く逆数の断片の小さい項の判定（ozaki2_moduli.h の oz2_choose_small_tail）が数える
// floor(m/2)+1 である（同じ関数 oz2_crt_k_bound を呼ぶ）．対称剰余 floor(m/2) では足りない：奇数の法で 1 超える積があり，例えば
// m=3475・r=1737・y=3474（{3475,3474} の CRT の係数）は fl(d*fl(1/m)) = 1736.5 が偶数の 1736 へ丸まって k=1738 になる．
// 既定の集合の y_i ではこのはみ出しは起きない（(a) の行の「はみ出し 0」）．
//
// (c) は法の集合を既定に限らない．OZ2_FP64 や OZ2_B では本数も b も変わり，末尾の
// 選び直しは上限以下の奇数をどれでも引きうるので，経路ごとに「上限以下で桁が収まる
// すべての法」（表の偶数の法も含む上位集合）を見る．法 2 も入れる（段の時間のモデルで法を選ぶ経路 ozaki2_awe.h は
// 2 まで候補にする）．桁数は 3，4，5 のすべて．
// S の全域（法 1 本あたり 2^25 通り）は回せないので，次の形で漏れなく確かめる．
//   ・部分和はすべて整数で，|S| の上界 Smax に m を足しても 2^24 以下であることを
//     解析的に確かめる（部分和・商と法の積・最後の引き算が FP32 で無誤差になる）．
//   ・fl(S*rinv) と S/m のずれは |S|*(2^-23+2^-47)/m 以下なので，半整数 (k+1/2)m から
//     W = floor(Smax*(2^-23+2^-47))+1 より離れた S では round が正しい整数を選び，
//     結果は |r| < m/2 の正しい対称剰余になる．残る境界の近傍 ±W だけを，
//     Rnd の同点を両向きに倒して総当たりする．
//   ・得た |r| の最大 R が，ホストが桁の上限・桁数を数える解析の上界 oz2_resid_bound 以下であることを見る．
//   ・得た |r| の最大 R までのすべての r について，スライサと同じ FP32 の手順で
//     桁分割し（方式 SQ/KAR/WIDE/平衡 n 桁/疎な桁の平方数 SQS/1 桁 ONE），桁が経路の上限に収まることを見る．
//     INT8 経路では符号-絶対値の 7 bit（|r| <= 127）に収まることを見る．
//     SQS（FP8 の疎な e4m3 桁．ozaki2_split.h の (E)，ozaki2/kernels/oz2_sparse_fix.h）は加えて
//       - 桁 l, h が e4m3 で無誤差に表せる整数（|x| <= 16 か 32 以下の偶数）であること，
//       - l + q*h ≡ r (mod q^2) であること（剰余類），
//       - FP32 の手順（カーネルの写し）とホストの整数の写し oz2_sqs_split が同じ桁を出すこと，
//       - 法ごとの l・h の最大がホストの桁の上限 oz2_sqs_bounds（連結の本数と累算の条件に使う）以下であり，
//         ホストが数え上げる範囲 floor(m/2)+1 が実際の |r| の最大 R を覆っていること
//     を全法・全剰余で確かめる．
//
// (d) は合成の恒等式（桁の作り方と合成の係数の整合），(e) は再構成が 1 桁の法の GEMM の出力を法へ畳む
// ModRound（recon_crt.cpp の RAWMASK）で，どちらも下のそれぞれの説明にある．
//
// 判定は (a)(b)(c)(e) とも 2 つ．結果が対称剰余の範囲に入っていること（(a)(a') は floor(m/2)+1，(c) は桁の
// 上限，(e) は floor(m/2)+2 で見る），そして剰余類が正しいこと（真値との差が m の整数倍）．
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include "oz_sim.h"
#include "testkit/probe_line.hpp"   // 機械が読む結果の行（PROBE,modround_check,...．main の末尾）

// ---- 写しと本番の手続きの突き合わせ -------------------------------------------------------
// 下の modround・modround2 は sfpu_tf_proc.h の ModRound を FP32 のまま写したもの（23.8 億通りを回すので，Dst の模擬を
// 通さない速い写しを使い，(c) などの同点の両向きも写しで作る）．写しが本番の手続きから外れないよう，同じ入力で本番の
// 手続き（checks/sfpu_tf_emu.hpp の sfpu_emu::ModRound．Dst の上の sfpu_proc::ModRound）とも計算して値を比べる：
// modround は全件，modround2 は kEmuStride 件に 1 件（呼んだ順の決まった間引き．(a')(c)(e) のどの掃引も含む）．同点は
// 本番が最近接偶数の一方なので，modround2 の r0・r1 のどちらかと一致すればよい．0 の符号は値で比べる（模擬の Rnd は
// 0 を +0 にする）．比べた件数は「合計 N 通り」に足さない（同じ入力の計算し直し）．不一致は FAIL
constexpr unsigned kEmuStride = 16;
static long long g_emu_cases = 0, g_emu_mismatch = 0;
static unsigned long long g_emu_tick = 0;
static inline float emu_modround(float d, float m, float rinv) {
    sfpu_emu::Dst D;
    D.r[0] = d;
    sfpu_emu::ModRound(D, 0, 1, m, rinv);
    return D.r[0];
}
static inline void emu_compare(float d, float m, float rinv, float r0, float r1) {
    const float e = emu_modround(d, m, rinv);
    g_emu_cases++;
    if (e != r0 && e != r1) {
        if (g_emu_mismatch < 5)
            printf("  FAIL 写しと本番の ModRound が違う m=%.0f d=%.1f：写し %.1f/%.1f，本番 %.1f\n", (double)m, (double)d,
                   (double)r0, (double)r1, (double)e);
        g_emu_mismatch++;
    }
}

// sfpu_tf_proc.h の ModRound を FP32 のまま写したもの．デバイス側は
//   Cpy(d,s); MulS(s,rinv); Rnd(s); MulS(s,m); Sub(d,s,o)
// で，ここはその 1 行ずつの対応である．Rnd の同点の向きはデバイスと合わせていないが，
// 下の判定は「対称剰余に収まるか」と「剰余類が正しいか」だけを見るので，
// 同点が両向きのどちらへ倒れても結論は変わらない（どちらも同じ剰余類で絶対値 m/2）．
// 本番の手続きとの一致は上の emu_compare で全件を見る
static inline float modround(float d, float m, float rinv) {
    volatile float s = d * rinv;          // volatile で 80 bit への昇格と融合を止める
    s = std::nearbyintf(s);
    volatile float t = s * m;
    const float r = d - t;
    emu_compare(d, m, rinv, r, r);
    return r;
}

// lim は |結果| の上限（既定は対称剰余の m/2．(a) は oz2_crt_k_bound）
static int check_one(int m, float rinv, const char* what, long double dval, int& worst, double lim = -1) {
    const float d = (float)dval;
    const float r = modround(d, (float)m, rinv);
    if (lim < 0) lim = m * 0.5;
    if (std::fabs(r) > lim) {
        printf("  FAIL %s m=%d d=%.1f -> %.1f（範囲 %.1f を超えた）\n",
               what, m, (double)d, (double)r, lim);
        return 1;
    }
    const long double diff = (long double)d - (long double)r;
    const long double q = diff / (long double)m;
    if (fabsl(q - roundl(q)) > 1e-9L) {
        printf("  FAIL %s m=%d d=%.1f -> %.1f（剰余類が違う）\n",
               what, m, (double)d, (double)r);
        return 1;
    }
    const int a = (int)std::fabs((double)r);
    if (a > worst) worst = a;
    return 0;
}

// ---- (c) スライサの oz2_residue ----------------------------------------------
// ModRound を Rnd の同点の両向きで評価する（r0 = 下へ，r1 = 上へ倒したもの．同点で
// なければ両者は等しい）．
// 本番の手続きとの一致は kEmuStride 件に 1 件で見る（冒頭の emu_compare）
static inline void modround2(float d, float m, float rinv, float& r0, float& r1) {
    volatile float s = d * rinv;
    const float lo = std::floor((float)s), hi = std::ceil((float)s);
    float a = lo, b = hi;
    if ((float)s - lo != 0.5f) a = b = std::nearbyintf(s);   // 同点でなければ最近接
    volatile float ta = a * m, tb = b * m;
    r0 = d - ta; r1 = d - tb;
    if (g_emu_tick++ % kEmuStride == 0) emu_compare(d, m, rinv, r0, r1);
}
static inline float rnd_dir(float x, int dir) {             // Rnd の同点を dir で倒す
    const float f = std::floor(x);
    if (x - f == 0.5f) return dir ? f + 1.f : f;
    return std::nearbyintf(x);
}

// |S| <= Smax の整数 S すべてについて ModRound(S) の剰余類が正しいことを確かめ，
// |r| の最大を返す（失敗なら -1）．半整数の境界の近傍だけを総当たりする（冒頭の説明）．
static long long g_ncase_c = 0;
static long sweep_modround(int m, long Smax) {
    const float mf = (float)m, rinv = (float)(1.0 / (double)m);
    if ((double)Smax + m > 16777216.0) return -1;          // 部分和と s*m が 2^24 を超える
    const long W = (long)std::floor((double)Smax * (std::ldexp(1.0, -23) + std::ldexp(1.0, -47))) + 1;
    long R = m / 2;                                        // 境界から離れた S は |r| <= floor(m/2)
    const long kmax = Smax / m + 1;
    for (long k = -kmax - 1; k <= kmax; k++) {
        const long c = (long)std::floor((k + 0.5) * m);   // 境界 (k+1/2)m の下側の整数
        for (long S = c - W; S <= c + 1 + W; S++) {
            if (S < -Smax || S > Smax) continue;
            float r0, r1;
            modround2((float)S, mf, rinv, r0, r1);
            g_ncase_c++;
            for (float r : {r0, r1}) {
                const long ri = (long)r;
                if ((float)ri != r || (S - ri) % m != 0) {
                    printf("  FAIL residue m=%d S=%ld -> %.1f（剰余類が違う）\n", m, S, (double)r);
                    return -1;
                }
                R = std::max(R, std::labs(ri));
            }
        }
    }
    return R;
}

// ---- (a') r_i*y_i と (b') の 2 平面復号を，選びうる全法・全係数で ----------------------------
// 冒頭の (a') (b') の説明．
static long long g_ncase_a2 = 0;
// |d| = |r|*y（1 <= |r| <= R，1 <= y <= m-1，gcd(y,m) = 1）と分解できるか（できれば分解の 1 つを r・y に返す）
static bool crt_product_of(long d, int m, long R, long& r, long& y) {
    const long a = std::labs(d);
    if (a == 0) { r = 0; y = 1; return true; }
    const long q0 = std::max(1L, (a + m - 2) / (m - 1));    // y <= m-1 なので |r| >= ceil(a/(m-1))
    for (long q = q0; q <= std::min(R, a); q++)
        if (a % q == 0 && std::gcd(a / q, (long)m) == 1) { r = d < 0 ? -q : q; y = a / q; return true; }
    return false;
}
static int check_ry_path(const char* name, int cap, int dmax, Oz2Ndig ndig, int smax) {
    int fail = 0, nm = 0, worst_m = 0;
    long nex = 0, worst_ex = 0, wd = 0, wr = 0, wy = 0;   // はみ出した（|k| > floor(m/2)）積の数と，はみ出しの最大の例
    long long nb = 0;
    for (int m = 2; m <= cap; m++) {
        if (dmax && !oz2_usable(m, dmax, ndig, smax)) continue;
        nm++;
        // r の範囲は (a) と同じ：1 桁の法は RAWMASK の出力で floor(m/2)+2，ほかは合成段（INT8 は (b) の復号）の対称剰余
        const bool one = dmax && oz2_split_mode(m, dmax, ndig, smax) == OZ2_SPLIT_ONE;
        const long R = one ? m / 2 + 2 : m / 2, Smax = R * (m - 1), kb = oz2_crt_k_bound(m);
        const float mf = (float)m, rinv = (float)(double)((__float128)1 / (__float128)m);   // ozaki2_moduli.h の rinv と同じ作り
        const long W = (long)std::floor((double)Smax * (std::ldexp(1.0, -23) + std::ldexp(1.0, -47))) + 1;
        const long kmax = Smax / m + 1;
        for (long k = -kmax - 1; k <= kmax && !fail; k++) {
            const long c = (long)std::floor((k + 0.5) * m);   // 境界 (k+1/2)m の下側の整数
            for (long S = c - W; S <= c + 1 + W; S++) {
                if (S < -Smax || S > Smax) continue;
                float r0, r1;
                modround2((float)S, mf, rinv, r0, r1);
                g_ncase_a2++;
                for (float rr : {r0, r1}) {
                    const long ri = (long)rr;
                    if ((float)ri != rr || (S - ri) % m != 0) {
                        printf("  FAIL r*y（全法）%s m=%d d=%ld -> %.1f（剰余類が違う）\n", name, m, S, (double)rr);
                        fail = 1; break;
                    }
                    if (std::labs(ri) <= m / 2) continue;
                    long r, y;
                    if (!crt_product_of(S, m, R, r, y)) continue;   // r*y の形で現れない d
                    nex++;
                    if (std::labs(ri) - m / 2 > worst_ex) { worst_ex = std::labs(ri) - m / 2; worst_m = m; wd = S; wr = r; wy = y; }
                    if (std::labs(ri) > kb) {
                        printf("  FAIL r*y（全法）%s m=%d r=%ld y=%ld d=%ld -> %ld（上界 floor(m/2)+1 = %ld を超えた）\n",
                               name, m, r, y, S, ri, kb);
                        fail = 1; break;
                    }
                }
            }
        }
        // (b') INT8 経路の 2 平面復号（(chi mod m)*(2^16 mod m) + (clo mod m)）．|.| < 2^15 で一般の保証の内側
        if (!dmax) {
            long v = 1 % m; for (int e = 0; e < 16; e++) v = (v * 2) % m;
            int wb = 0;
            for (int a = -m / 2; a <= m / 2; a++)
                for (int b = -m / 2; b <= m / 2; b++, nb++)
                    fail |= check_one(m, rinv, "chi+clo（全法）", (long double)a * (long double)v + (long double)b, wb);
        }
    }
    g_ncase_a2 += nb;
    if (nex) printf("  %s 法 %4d 本  r*y が floor(m/2) をはみ出す積 %ld 通り（最大 +%ld：m=%d r=%ld y=%ld d=%ld）  上界 floor(m/2)+1 以下  %s\n",
                    name, nm, nex, worst_ex, worst_m, wr, wy, wd, fail ? "FAIL" : "OK");
    else     printf("  %s 法 %4d 本  r*y が floor(m/2) をはみ出す積 0 通り%s  %s\n", name, nm,
                    nb ? "（INT8 の 2 平面復号も全法で対称剰余）" : "", fail ? "FAIL" : "OK");
    return fail;
}

// 桁の重み c[j] = 2^(20(j+1)) mod m（j = 0,1,2,3）．ホストの c0/c1/c2/c3 とは独立に作り，
// (c) の入力に使うとともに main でホストの値と突き合わせる．
static void digit_weights(int m, long c[4]) {
    long v = 1 % m;
    for (int j = 0; j < 4; j++) {
        for (int e = 0; e < OZ2_DIGIT_BITS; e++) v = (v * 2) % m;
        c[j] = v;
    }
}

// 方式 (E) の手直し（ozaki2/kernels/oz2_sparse_fix.h の oz2_sqs_fix_face を FP32 のまま写したもの）．
// 偶奇は |x| + 2^23 のビット列の最下位ビット，条件は |x| > 16.5．
static inline bool sfpi_odd(float x) {
    volatile float a = std::fabs(x) + 8388608.0f;
    uint32_t u; std::memcpy(&u, (const void*)&a, 4);
    return (u & 1u) != 0;
}
static inline void sqs_fix_fp32(float& l, float& h, float q) {
    if (std::fabs(l) > 16.5f && sfpi_odd(l)) {
        if (l < 0.0f) { l = l + q; h = h - 1.0f; } else { l = l - q; h = h + 1.0f; }
    }
    if (std::fabs(h) > 16.5f && sfpi_odd(h)) {
        if (h < 0.0f) h = h + q; else h = h - q;
    }
}
static long g_sqs_fail = 0;
static int g_sqs_worst_l = 0, g_sqs_worst_h = 0, g_nsqs = 0;

// 法 m の剰余 ri をスライサ（modslice_digit.cpp）と同じ FP32 の手順で桁に分ける（Rnd の同点は dir で倒す．下の (d) で定義）
static void digits_fp32(int m, int dmax, Oz2Ndig ndig, int smax, long ri, int dir, float out[8], int& np);
// |r| <= R のすべての r を digits_fp32 で桁に分け，桁の絶対値の最大を返す．Rnd の同点は両向きを見る．
// smax は疎な桁の上限（FP8 の 32．他は 0）．
static int split_digits_max(int m, long R, int dmax, Oz2Ndig ndig, int smax = 0) {
    const int mode = oz2_split_mode(m, dmax, ndig, smax), q = oz2_digit_base(m, dmax, ndig);
    int worst = 0;
    int hdl = 0, hdh = 0;
    if (mode == OZ2_SPLIT_SQS) {
        oz2_sqs_bounds(q, hdl, hdh); g_nsqs++;
        if (R > m / 2 + 1) {
            printf("  FAIL SQS m=%d |r| の最大 %ld がホストの桁の上限の数え上げ範囲 floor(m/2)+1 を超える\n", m, R);
            g_sqs_fail++;
        }
    }
    int wl = 0, wh = 0;
    for (long ri = -R; ri <= R; ri++) {
        for (int dir = 0; dir < 2; dir++) {
            g_ncase_c++;
            float d[8]; int np;
            digits_fp32(m, dmax, ndig, smax, ri, dir, d, np);
            for (int k = 0; k < np; k++) worst = std::max(worst, (int)std::fabs(d[k]));
            if (mode != OZ2_SPLIT_SQS) continue;
            const float l = d[0], h = d[1];
            long li, hi; oz2_sqs_split(ri, q, li, hi);
            const long rec = (long)l + (long)q * (long)h - ri;
            const bool bad = (float)li != l || (float)hi != h || !oz2_in_sparse((long)l, smax) ||
                             !oz2_in_sparse((long)h, smax) || rec % ((long)m) != 0 ||
                             std::fabs(l) > hdl || std::fabs(h) > hdh;
            if (bad) {
                if (g_sqs_fail < 5)
                    printf("  FAIL SQS m=%d r=%ld dir=%d -> l=%.0f h=%.0f（ホストの写し %ld,%ld，上限 %d,%d）\n",
                           m, ri, dir, (double)l, (double)h, li, hi, hdl, hdh);
                g_sqs_fail++;
            }
            wl = std::max(wl, (int)std::fabs(l)); wh = std::max(wh, (int)std::fabs(h));
        }
    }
    if (mode == OZ2_SPLIT_SQS) {
        g_sqs_worst_l = std::max(g_sqs_worst_l, wl); g_sqs_worst_h = std::max(g_sqs_worst_h, wh);
        // 疎な桁の上限は smax（32）で見る（呼び出し側は dmax=16 で見るので，ここで 16 以下に見せる）
        return (wl <= smax && wh <= smax) ? std::min(worst, dmax) : smax + 1;
    }
    return worst;
}

// 桁行列ごとの桁の上限（oz2_plane_bound．連結の本数 maxg と累算の条件に使う）が，|r| <= R の実際の桁の最大以上か．
// 1 桁の法を混ぜる経路（Oz2Ndig::one．桁 32 の AWE 経路）と疎な桁の経路は法・桁行列ごとの細かい上限を使うので，ここで確かめる
// （他の経路は上限 dmax そのもので，(c) の「桁の最大 <= dmax」と同じ）．
static long g_plane_fail = 0;
static void check_plane_bounds(const char* name, int m, long R, int dmax, Oz2Ndig ndig, int smax) {
    int mx[8] = {0}, np = 0;
    for (long r = -R; r <= R; r++)
        for (int dir = 0; dir < 2; dir++) {
            float d[8]; digits_fp32(m, dmax, ndig, smax, r, dir, d, np);
            for (int p = 0; p < np; p++) mx[p] = std::max(mx[p], (int)std::fabs(d[p]));
        }
    for (int p = 0; p < np; p++) {
        const int b = oz2_plane_bound(m, p, dmax, ndig, smax);
        if (mx[p] > b) {
            if (g_plane_fail < 5) printf("  FAIL %s m=%d 桁行列 %d の桁の最大 %d がホストの上限 %d を超える\n", name, m, p, mx[p], b);
            g_plane_fail++;
        }
    }
}

// (c) が法ごとに得た |r| の最大（経路に依らない）．(d) が剰余の範囲に使う
static std::vector<long> g_rall;
static long g_rb_fail = 0;
// 経路 1 本ぶん．法は「上限以下で桁が収まるすべての m」（表の法と末尾の選び直しの
// 候補の上位集合．2 から）．桁数 ND は 3〜5．
static int check_residue_path(const char* name, int cap, int dmax, Oz2Ndig ndig, int smax = 0) {
    // d0 は丸めた残差の 2^20 倍で |d0| <= 2^19 強（oz2_digits.h．中位以下の limb のぶんを含めた上界は ozaki2_moduli.h の
    // oz2_resid_bound の D0）．余裕を見て 2^20 とする．
    const long D0 = 1L << 20;
    int fail = 0, nm = 0, worst_excess = 0, worst_digit = 0;
    long worst_S = 0; int worst_m = 0;
    const long plane_fail0 = g_plane_fail, rb_fail0 = g_rb_fail, sqs_fail0 = g_sqs_fail;
    if ((int)g_rall.size() <= cap) g_rall.resize(cap + 1, -1);
    for (int m = 2; m <= cap; m++) {
        if (dmax && !oz2_usable(m, dmax, ndig, smax)) continue;       // 使えない法
        nm++;
        const long h = m / 2;
        long c[4];
        digit_weights(m, c);
        // 3 桁：桁 d1, d2 の項まで足した S3 が最後の ModRound の入力になる．
        // 4 桁：d3 の項を足す前に S3 を 1 度畳み（oz2_residue），その値 |r1| <= R3 に
        //       t3*c2 を足した S4 が最後の ModRound の入力になる．
        // 5 桁：R3 に t4*c3 を足した S4a を畳み，その値 |r2| <= R4a に t3*c2 を足した S5 が
        //       最後の ModRound の入力になる．
        const long S3 = D0 + h * (c[0] + c[1]);
        const long R3 = sweep_modround(m, S3);
        const long S4 = (R3 < 0) ? -1 : R3 + h * c[2];
        const long R4 = (S4 < 0) ? -1 : sweep_modround(m, S4);
        const long S4a = (R3 < 0) ? -1 : R3 + h * c[3];
        const long R4a = (S4a < 0) ? -1 : sweep_modround(m, S4a);
        const long S5 = (R4a < 0) ? -1 : R4a + h * c[2];
        const long R5 = (S5 < 0) ? -1 : sweep_modround(m, S5);
        for (long Sx : {S3, S4, S4a, S5}) if (Sx > worst_S) { worst_S = Sx; worst_m = m; }
        if (R3 < 0 || R4 < 0 || R4a < 0 || R5 < 0) {
            printf("  FAIL %s m=%d 最後の ModRound が剰余類を保てない（%d 桁）\n", name, m, R3 < 0 ? 3 : (R4 < 0 ? 4 : 5));
            fail = 1; continue;
        }
        // 3 桁の R3 は 4・5 桁では途中の値だが，最後の値としても現れるので桁の上限の判定に含める．
        // 5 桁の途中の R4a は ModRound の入力の条件（S5 の 2^24）だけに効き，桁には現れない
        const long Rall = std::max(std::max(R3, R4), R5);
        g_rall[m] = Rall;
        worst_excess = std::max(worst_excess, (int)(Rall - h));
        // ホストは KAR（基数 B）・平衡 n 桁の桁の上限と桁数，1 桁の法の桁行列の上限をこの解析の上界で数える
        if (Rall > oz2_resid_bound(m)) {
            if (g_rb_fail < 5) printf("  FAIL %s m=%d |r| の最大 %ld がホストの上界 oz2_resid_bound = %ld を超える\n",
                                      name, m, Rall, oz2_resid_bound(m));
            g_rb_fail++; fail = 1;
        }
        if (!dmax) {                                          // INT8：符号-絶対値 7 bit
            if (Rall > 127) { printf("  FAIL %s m=%d |r|<=%ld が 127 を超える\n", name, m, Rall); fail = 1; }
            continue;
        }
        const int dg = split_digits_max(m, Rall, dmax, ndig, smax);
        if (ndig.one || smax) check_plane_bounds(name, m, Rall, dmax, ndig, smax);
        worst_digit = std::max(worst_digit, dg);
        if (dg > dmax) {
            printf("  FAIL %s m=%d |r|<=%ld で桁が %d（上限 %d）\n", name, m, Rall, dg, dmax);
            fail = 1;
        }
    }
    if (g_plane_fail > plane_fail0 || g_rb_fail > rb_fail0 || g_sqs_fail > sqs_fail0) fail = 1;
    printf("  %s 法 %4d 本  最大 |S|=%ld (m=%d, 2^%.2f)  |r|-floor(m/2) <= %d  桁の最大 %d <= %d  %s\n",
           name, nm, worst_S, worst_m, std::log2((double)worst_S), worst_excess,
           worst_digit, dmax, fail ? "FAIL" : "OK");
    return fail;
}

// ---- (d) 合成の恒等式 ------------------------------------------------------------
// 法ごとの運び方（oz2_split_mode）で作った桁（スライサと同じ FP32 の手順．Rnd の同点は両向き）と，
// ホストの合成の列挙 oz2_product_terms（桁行列の組と係数）から，Σ_j c_j * a[pa_j] * b[pb_j] ≡ r_a * r_b (mod m) を
// 確かめる．剰余の範囲はスライサの出力が取りうる全部で，(c) が得た |r| の最大と oz2_resid_bound の大きいほう
// （floor(m/2) を 1 超えうる）．r_a はその範囲のすべて，r_b は端・0・±1・±q^j（桁ベクトルが桁行列の空間を張る値）と
// 刻み 1+m/8 の値．Rnd の同点の向きは a・b それぞれ両向きを見る（b は向きで桁が変わる値だけ両方）．
// r_a を止めると左辺は r_b の桁ベクトルについて線形（右辺も r_b = Σ w_i d_i で線形）なので，
// 桁ベクトルが張る集合で成り立てば全 r_b で成り立つ．
// 係数・桁行列の並び・桁数（平衡 n 桁の法ごとの n，基数 B の Karatsuba）の食い違いがここで出る．
static long long g_ncase_d = 0;
static void digits_fp32(int m, int dmax, Oz2Ndig ndig, int smax, long ri, int dir, float out[8], int& np) {
    const int mode = oz2_split_mode(m, dmax, ndig, smax), q = oz2_digit_base(m, dmax, ndig);
    const float qf = (float)q, qinv = 1.0f / (float)q, r = (float)ri;
    np = oz2_digit_planes(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) { out[0] = r; return; }
    if (mode == OZ2_SPLIT_BAL) {
        float cur = r;
        for (int j = 0; j + 1 < np; j++) {
            volatile float t = cur * qinv; const float h = rnd_dir(t, dir);
            volatile float hq = h * qf; out[j] = cur - hq; cur = h;
        }
        out[np - 1] = cur;
    } else if (mode == OZ2_SPLIT_KAR) {
        const float sg = (r < 0.f) ? -1.f : 1.f;
        volatile float u = sg * (qf * 0.5f - 0.5f);
        volatile float x = r + u;
        volatile float t = x * qinv; const float h = rnd_dir(t, dir);
        volatile float hq = h * qf; const float l = r - hq;
        out[0] = l; out[1] = h; out[2] = h + l;
    } else {
        volatile float t = r * qinv; const float h0 = rnd_dir(t, dir);
        volatile float hq = h0 * qf;
        float l = r - hq, h = h0;
        if (mode == OZ2_SPLIT_SQS) sqs_fix_fp32(l, h, qf);
        out[0] = l; out[1] = h;
    }
}
static int check_compose_identity(const char* name, int cap, int dmax, Oz2Ndig ndig, int smax) {
    int fail = 0, nm = 0;
    for (int m = 2; m <= cap; m++) {
        if (!dmax || !oz2_usable(m, dmax, ndig, smax)) continue;
        nm++;
        Oz2ProductTerm t[OZ2_MAXPROD];
        const uint32_t nt = oz2_product_terms(m, t, dmax, ndig);
        // スライサの剰余の範囲（対称剰余を超える分も含む．(c) を先に回すので g_rall は埋まっている）
        const long R = std::max<long>(oz2_resid_bound(m), m < (int)g_rall.size() ? g_rall[m] : m / 2);
        std::vector<long> rbs = {-R, -R + 1, -1, 0, 1, R - 1, R};
        for (long v = -R; v <= R; v += 1 + m / 8) rbs.push_back(v);
        { const long q = oz2_digit_base(m, dmax, ndig); long p = 1;
          for (int j = 0; j < 6 && p <= R; j++, p *= q) { rbs.push_back(p); rbs.push_back(-p); } }
        // r_b の桁は法ごとに 1 度だけ作る．同点の向きで桁が変わる r_b だけ両向きを残す（変わらなければ同じ検査になる）
        struct BDig { long r; int dir; float d[8]; };
        std::vector<BDig> bds;
        for (long rb : rbs) {
            BDig b0{rb, 0, {}}, b1{rb, 1, {}}; int np0, np1;
            digits_fp32(m, dmax, ndig, smax, rb, 0, b0.d, np0);
            digits_fp32(m, dmax, ndig, smax, rb, 1, b1.d, np1);
            bds.push_back(b0);
            if (std::memcmp(b0.d, b1.d, sizeof b0.d) != 0) bds.push_back(b1);
        }
        long long cf[OZ2_MAXPROD];
        for (uint32_t j = 0; j < nt; j++) cf[j] = std::llround(t[j].coeff_lo);
        for (long ra = -R; ra <= R && !fail; ra++)
            for (int da = 0; da < 2 && !fail; da++) {
                float a[8] = {}; int npa; digits_fp32(m, dmax, ndig, smax, ra, da, a, npa);
                for (const BDig& b : bds) {
                    long long acc = 0;
                    for (uint32_t j = 0; j < nt; j++)
                        acc += cf[j] * (long long)a[t[j].a_plane] * (long long)b.d[t[j].b_plane];
                    const long long want = (long long)ra * b.r, d = ((acc - want) % m + m) % m;
                    g_ncase_d++;
                    if (d != 0) {
                        printf("  FAIL 合成 %s m=%d r_a=%ld r_b=%ld（方式 %d，同点の向き %d/%d）: Σ c a b - r_a r_b ≢ 0\n", name, m,
                               ra, b.r, oz2_split_mode(m, dmax, ndig, smax), da, b.dir);
                        fail = 1; break;
                    }
                }
            }
    }
    printf("  %s 法 %4d 本の合成の恒等式  %s\n", name, nm, fail ? "FAIL" : "OK");
    return fail;
}

// ---- (e) 再構成の RAWMASK の ModRound（1 桁の法） ------------------------------------
// 1 桁の法（OZ2_SPLIT_ONE．Oz2Ndig::one の経路）は合成段を通らず，GEMM の出力 P（桁＝剰余の積の K 項の和）を
// 再構成（recon_crt.cpp の RAWMASK）が ModRound 1 回で法へ畳む．ホストの法選択（ozaki2_choose_moduli_w）は 1 桁の法に
// |P| + m <= K*R^2 + m <= 2^24 を課す（R は oz2_resid_bound）．|P| + m が 2^24 を超えると丸めた商と法の積が FP32 に
// 載らず剰余類が崩れる（m=25・S=-16777215 で 9 を返す．正しくは 10）ので，この余裕は要る．
// そこで法ごとの K の上限を仮定せず，|S| <= Smax = 2^24 - m の全域について，半整数の境界の近傍 ±W（Rnd の同点は
// 両向き）と両端 Smax - m - 1 <= |S| <= Smax を総当たりし，剰余類が正しいことと |r| <= floor(m/2)+2（recon_crt.cpp の
// RAWMASK の注記の m/2 + |P|*2^-23）を見る．境界から W より離れた S では round が正しい整数を選ぶ（(c) の説明と同じ）．
// |S| が 2^22 を超えると round が隣の整数を選びうるので，小さい法ほど |r| は floor(m/2) を大きくはみ出す（m=7 で 5）が，
// 剰余類は保たれる．
// 続く r*y_i は |r| <= m/2+2・|y_i| < m で (m/2+2)(m-1) < 2^22 なので，ModRound の一般の保証の内側である．
static long long g_ncase_e = 0;
static int check_raw_modround(const char* name, int cap, int dmax, Oz2Ndig ndig) {
    int fail = 0, nm = 0, worst_m = 0; long worst = -1;
    for (int m = 2; m <= cap; m++) {
        if (oz2_split_mode(m, dmax, ndig) != OZ2_SPLIT_ONE) continue;
        nm++;
        const long Smax = (1L << 24) - m;
        const long W = (long)std::floor((double)Smax * (std::ldexp(1.0, -23) + std::ldexp(1.0, -47))) + 1;
        const float mf = (float)m, rinv = (float)(1.0 / (double)m);
        long R = 0;
        auto one = [&](long S) {
            float r0, r1;
            modround2((float)S, mf, rinv, r0, r1);
            g_ncase_e++;
            for (float r : {r0, r1}) {
                const long ri = (long)r;
                if ((float)ri != r || (S - ri) % m != 0 || std::labs(ri) > m / 2 + 2) {
                    if (fail < 5) printf("  FAIL RAWMASK %s m=%d S=%ld -> %.1f\n", name, m, S, (double)r);
                    fail++;
                }
                R = std::max(R, std::labs(ri));
            }
        };
        const long kmax = Smax / m + 1;
        for (long k = -kmax - 1; k <= kmax; k++) {
            const long c = (long)std::floor((k + 0.5) * m);   // 境界 (k+1/2)m の下側の整数
            for (long S = c - W; S <= c + 1 + W; S++)
                if (S >= -Smax && S <= Smax) one(S);
        }
        for (long S = Smax - m - 1; S <= Smax; S++) { one(S); one(-S); }
        if (R - m / 2 > worst) { worst = R - m / 2; worst_m = m; }
    }
    printf("  %s 1 桁の法 %2d 本  |P| <= 2^24-m  |r|-floor(m/2) <= %ld (m=%d)  %s\n", name, nm, worst, worst_m,
           fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char** argv) {
    // K は 1 以上 2^32 未満の整数として全体を読み切れたときだけ受け付ける
    // （"14080junk" や 2^32+14080 が 14080 に化けず，0 が法選択の log2(0) に届かない）．
    uint32_t K = 14080u;
    if (argc > 2) { fprintf(stderr, "ERROR: 引数は K の 1 個まで\n"); return 2; }
    if (argc > 1) {
        long long v = 0;   // 読み方は src/common/oz_env.hpp
        if (!oz_env::to_i64(argv[1], v) || v < 1 || v > (long long)UINT32_MAX) {
            fprintf(stderr, "ERROR: K=%s は 1--%u の整数ではない\n", argv[1], UINT32_MAX);
            return 2;
        }
        K = (uint32_t)v;
    }
    // (a)(b)：論文が測る経路ぶんの既定の法の取り方（本番と同じ ozaki2_moduli.h の表 Oz2Route．
    // BF16・FP16・BFP8_b は段の時間のモデルで法を選び，K=3520・14080 では同じ集合になる（K によっては違う集合を選ぶ．
    // K=128・3744 など）．TF32 は段の時間のモデルを使わず表から取る選び方（OZ2_ROUTE_D32）．
    // acc は OZ2_ACC_BITS で，INT8 経路でしか使われない）
    // 経路の表は本番そのもの（ozaki2_split.h の OZ2_ROUTES）．(a')〜(e) は法の上限・桁の上限・桁数・疎な桁・1 桁の法だけで決まるので，
    // それが同じ経路は 1 行にまとめる（BF16・BFP8_b・FP16 は段の時間のモデルの係数だけが違う）
    struct Group { std::string name; Oz2Route r; };
    std::vector<Group> groups;
    for (const auto& x : OZ2_ROUTES) {
        bool merged = false;
        for (Group& g : groups)
            if (g.r.mod_cap == x.r.mod_cap && g.r.digit_bound == x.r.digit_bound && g.r.nrdig == x.r.nrdig &&
                g.r.sparse_max == x.r.sparse_max && g.r.one_digit == x.r.one_digit) { g.name += std::string("/") + x.name; merged = true; break; }
        if (!merged) groups.push_back({x.name, x.r});
    }
    for (Group& g : groups) g.name.resize(std::max<size_t>(g.name.size(), 15), ' ');
    int fail = 0;
    long long ncase = 0;
    // 機械が読む行（PROBE,modround_check,part=…．main の末尾でまとめて出す．人向けの行の並びは変えない）．
    // 照合（scripts/rules/probe_checks.py）が part=total の cases をスライドの「計 X 億通り」と比べる
    std::vector<probe::ProbeLine> pls;
    auto route_line = [&](const char* route, int s, int mmax, long long n, bool bad) {
        pls.push_back(probe::ProbeLine("modround_check").add("part", "route").add("route", "%s", route).add("s", "%d", s)
                          .add("mmax", "%d", mmax).add("cases", "%lld", n).add("result", "%s", bad ? "FAIL" : "OK"));
    };
    printf("===== ModRound の総当たり（K=%u）=====\n", K);
    for (const auto& p : OZ2_ROUTES) {
        Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, p.r);
        if (!mo.ok) { printf("  %-5s 法が取れない\n", p.name); fail = 1; route_line(p.name, 0, 0, 0, true); continue; }
        int row_fail = 0;   // この行の判定（fail は終了コードのために累積する）
        const bool int8_path = p.r.digit_bound == 0;   // (b) の 2 平面復号を通るのは INT8 経路だけ
        int worst_a = 0, worst_b = 0, ex_a = 0;   // ex_a：(a) で |k| が floor(m/2) を超えた最大
        long long n = 0;
        for (int i = 0; i < mo.s; i++) {
            const int   m    = mo.m[i];
            const float rinv = mo.rinv[i];
            const int   y    = mo.yi[i];
            const float m16  = mo.m16[i];
            // (c) が自前で作る桁の重みがホストの c0/c1/c2/c3 と一致すること
            {
                long c[4];
                digit_weights(m, c);
                const float* hc[4] = {mo.c0.data(), mo.c1.data(), mo.c2.data(), mo.c3.data()};
                for (int j = 0; j < 4; j++)
                    if (hc[j][i] != (float)c[j]) { printf("  FAIL m=%d c%d がホストと違う\n", m, j); row_fail = 1; }
            }
            // (a) r_i * y_i：r_i は合成段の出力（対称剰余）なら m+1 通り，1 桁の法（RAWMASK の出力）は floor(m/2)+2 まで
            const int ra = (mo.mode[i] == OZ2_SPLIT_ONE) ? m / 2 + 2 : m / 2;
            for (int r = -ra; r <= ra; r++, n++) {
                int wa = 0;
                row_fail |= check_one(m, rinv, "r*y ", (long double)r * (long double)y, wa, (double)oz2_crt_k_bound(m));
                worst_a = std::max(worst_a, wa); ex_a = std::max(ex_a, wa - m / 2);
            }
            // (b) (chi mod m)*(2^16 mod m) + (clo mod m)．どちらも対称剰余で，
            //     符号は最後に掛け戻すので両符号を見る（INT8 経路だけ．冒頭の説明）
            if (int8_path)
                for (int a = -m / 2; a <= m / 2; a++)
                    for (int b = -m / 2; b <= m / 2; b++, n++)
                        row_fail |= check_one(m, rinv, "chi+clo",
                                          (long double)a * (long double)m16 + (long double)b, worst_b);
        }
        ncase += n;
        const int mmax = *std::max_element(mo.m.begin(), mo.m.end());
        fail |= row_fail;
        printf("  %-5s s=%-2d max m=%-5d  %10lld 通り  %s（剰余の最大 %d <= %d．r*y のはみ出し %d）  m={",
               p.name, mo.s, mmax, n, row_fail ? "FAIL" : "OK",
               worst_a > worst_b ? worst_a : worst_b, mmax / 2, ex_a);
        for (int i = 0; i < mo.s; i++) printf("%s%d", i ? "," : "", mo.m[i]);
        printf("}\n");
        route_line(p.name, mo.s, mmax, n, row_fail != 0);
    }
    // (a')(b') r_i*y_i と INT8 の 2 平面復号を，経路が選びうる全法・全係数で（冒頭の説明．経路の並びは下の (c) と同じ）
    printf("===== r_i*y_i の ModRound（上限以下の全法・m と互いに素な全 y．INT8 は 2 平面復号も）=====\n");
    int f_ry = 0;
    for (const Group& p : groups)
        f_ry |= check_ry_path(p.name.c_str(), p.r.mod_cap, p.r.digit_bound, p.r.nd(), p.r.sparse_max);
    fail |= f_ry;
    ncase += g_ncase_a2;
    // (c) スライサの oz2_residue．法の上限と桁の上限は経路の表のとおりで，BF16・FP16・BFP8_b（1 桁の法を混ぜる）は
    // 同じ関数で法を見るので 1 行，TF32 は 1 桁の法を使わないので別の行で見る．
    printf("===== スライサの最後の ModRound（oz2_residue．桁数 3〜5，上限以下の全法）=====\n");
    // 上限・桁の上限は本番と同じ表 Oz2Route から取る（FP8 は疎な桁 sparse_max=32 で 2401 まで）
    int f_res = 0;
    for (const Group& p : groups)
        f_res |= check_residue_path(p.name.c_str(), p.r.mod_cap, p.r.digit_bound, p.r.nd(), p.r.sparse_max);
    fail |= f_res;
    // (d) 合成の恒等式（桁の作り方と合成の列挙・係数の整合．全経路の上限以下の全法）
    printf("===== 合成の恒等式（Σ c_j a_{pa_j} b_{pb_j} ≡ r_a r_b (mod m)．上限以下の全法）=====\n");
    int f_cmp = 0;
    for (const Group& p : groups)
        f_cmp |= check_compose_identity(p.name.c_str(), p.r.mod_cap, p.r.digit_bound, p.r.nd(), p.r.sparse_max);
    fail |= f_cmp;
    ncase += g_ncase_d;
    // (e) 再構成の RAWMASK の ModRound（1 桁の法を混ぜる経路だけ）
    printf("===== 再構成の RAWMASK の ModRound（1 桁の法の GEMM の出力 |P| <= 2^24 - m）=====\n");
    int f_raw = 0;
    for (const Group& p : groups)
        if (p.r.one_digit) f_raw |= check_raw_modround(p.name.c_str(), p.r.mod_cap, p.r.digit_bound, p.r.nd());
    fail |= f_raw;
    ncase += g_ncase_e;
    // 方式 (E) の総当たりの結果（FP8 の疎な桁．上の FP8 の行に含まれる）
    printf("  SQS（疎な e4m3 桁の平方数）法 %d 本  |l| <= %d  |h| <= %d  桁が S_32 に入り剰余類も正しい：%s\n",
           g_nsqs, g_sqs_worst_l, g_sqs_worst_h, g_sqs_fail ? "FAIL" : "OK");
    const bool f_sqs = g_sqs_fail || !g_nsqs;
    if (f_sqs) fail = 1;
    ncase += g_ncase_c;
    // 写しと本番の手続きの突き合わせ（冒頭の emu_compare．比べた件数は合計に足さない）．不一致の例は上に出ている
    const bool f_emu = g_emu_mismatch || !g_emu_cases;
    if (f_emu) {
        printf("  写しの ModRound と本番の手続き（sfpu_emu::ModRound）の不一致 %lld 件（比べた %lld 件）  FAIL\n",
               g_emu_mismatch, g_emu_cases);
        fail = 1;
    }
    printf("\n合計 %lld 通り．%s\n", ncase,
           fail ? "FAIL あり" : "すべて対称剰余（(a)(a') は floor(m/2)+1，(c) は桁の上限）に収まり，剰余類も正しい．");
    // 節ごとの行（cases はその節の通り数．合計に足すのは route・ry・residue・compose・rawmask．sqs は residue の内訳なので
    // 法の本数，emu は比べた件数で，どちらも合計に足さない）
    auto part_line = [&](const char* part, long long cases, bool bad) {
        pls.push_back(probe::ProbeLine("modround_check").add("part", "%s", part).add("cases", "%lld", cases)
                          .add("result", "%s", bad ? "FAIL" : "OK"));
    };
    part_line("ry", g_ncase_a2, f_ry);
    part_line("residue", g_ncase_c, f_res || f_sqs);
    part_line("compose", g_ncase_d, f_cmp);
    part_line("rawmask", g_ncase_e, f_raw);
    pls.push_back(probe::ProbeLine("modround_check").add("part", "sqs").add("nmod", "%d", g_nsqs)
                      .add("result", "%s", f_sqs ? "FAIL" : "OK"));
    pls.push_back(probe::ProbeLine("modround_check").add("part", "emu").add("cases", "%lld", g_emu_cases)
                      .add("mismatch", "%lld", g_emu_mismatch).add("result", "%s", f_emu ? "FAIL" : "OK"));
    pls.push_back(probe::ProbeLine("modround_check").add("part", "total").add("K", "%u", K).add("cases", "%lld", ncase)
                      .add("result", "%s", fail ? "FAIL" : "PASS"));
    for (const auto& l : pls) l.print();
    return fail;
}
