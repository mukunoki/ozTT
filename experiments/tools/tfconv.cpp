// FP64 <-> triple-float の変換をデバイス上で行うためのビット手続きと，その検証
// ---------------------------------------------------------------------------
// 尾崎スキーム I の入口・出口をデバイス上で行うため，次の 2 つを
// 「浮動小数点演算を使わない整数手続き」として書き，正しさをここで確かめる．
//
//   split  : FP64 の 2 語 (w1,w0) -> triple-float の 3 本
//            slice_f64_reader.cpp（RISC-V，整数のみ）がやる仕事
//   assemble: triple-float の 3 本 -> FP64 の 2 語（真値から 1 ulp 以内）
//            recon_f64_writer.cpp（scheme I）と ozaki2/kernels/recon_writer.cpp（scheme II）
//            がやる仕事．どちらも RISC-V の整数演算だけで書かれている．
//
// 検査する実装は src/common/kernels/f64_tf.h そのものである（tools/oz_sim.h を通して
// include する）．独立なのは参照計算のほうで，__float128 と有理数比較を使う．
//
// どちらも FP64 の演算を一切使わない：デバイスの RISC-V は FP64 の演算器を持たず（単精度の F 拡張も一部だけで，倍精度の D 拡張は無い．Blackhole の ISA 資料 BabyRISCV/InstructionSet.md），
// double を書くと soft-float（約 76 ns/op）になってしまうためである．
//
// split の出力は 3 本の FP32 そのものではなく，リーダが整数だけで作れる 4 つの
// ビット列である（下位 2 本の正規化には先頭ビット位置の探索が要り，それを
// 避けるために「マジックナンバ」を使う．最後の定数減算と乗算は SFPU が行う）：
//
//   hi_bits    そのまま FP32．a * 2^(P+30) を組み立てたもの（P = E-1075+d は正規化の d を含む）
//   braw_bits  2^23 + 2^22 + b  （SFPU が 12582912.0f を引いて b を得る）
//   craw_bits  2^23 + c         （SFPU が  8388608.0f を引いて c を得る）
//   s_bits     ±2^(P+7)         （SFPU が上の 2 つに掛ける）
//
// すなわちデバイス側の計算カーネルは
//     hi  = hi_bits
//     mid = (braw - 12582912.0f) * s
//     lo  = (craw -  8388608.0f) * s * 2^-7
// の 5 演算だけで triple-float を得る．正規化（2 のべき）は d に畳んであるので
// inv の乗算も要らない．
//
// 検査 4（定数倍つきの組み立て）の基数と L は尾崎 I の本番の構成（scripts/routes.tsv の L と本番の基数 ozaki1_param.h の oz1_param）
// から，検査 6（尾崎 II の出口）の KM は本番の法の選択（ozaki2_moduli.h）から取る．そのためリポジトリ直下で回す
// （routes.tsv が読めなければ終了コード 2．src/Makefile の HOST_TOOLS が src/build/tfconv を作る）：
//   make -C src tfconv && ./src/build/tfconv      （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <utility>
#include <vector>
#include "oz_sim.h"


// リーダの split_f64 と，計算カーネルが行う 5 演算（SFPU の FP32 演算．非正規化数は 0 に潰す）で 3 本組を作る．
// 5 演算はカーネル側が SFPU の組込み関数で書いているので，模擬（checks/sfpu_tf_emu.hpp の load_f64）を使う
static void limbs_of(double x, int d, float& hi, float& mid, float& lo) {
    sfpu_emu::Dst D;
    sfpu_emu::load_f64(D, x, d);
    hi = D.r[0]; mid = D.r[1]; lo = D.r[2];
}

// 定数を 64 bit 仮数 + 指数へ分解する（ホスト側が 1 度だけ行う．ドライバと同じ common/oz_bits.h の oz_split_scale_ld）
static void split_const64(long double w, uint32_t& Whi, uint32_t& Wlo, int& qw) {
    const OzScale64 s = oz_split_scale_ld(w);
    Whi = s.hi; Wlo = s.lo; qw = s.q;
}

// 参照値を 2^sh 倍する（2 のべき倍なので __float128 では無誤差）
static __float128 q_scale(__float128 r, int sh) {
    while (sh < 0) { r *= (__float128)0.5; sh++; }
    while (sh > 0) { r *= (__float128)2.0; sh--; }
    return r;
}
static double ref_scale(__float128 r, int sh) { return (double)q_scale(r, sh); }

// 2 つの FP64 の ulp 距離．bit パターンを符号付きで引くと，符号の違う値どうしで
// signed overflow（未定義動作）になるので，符号を先に見て unsigned で測る．
// 符号が違えば「1 ulp 差ではない」とみなす（±0 どうしだけは一致扱い）．
static uint64_t ulp_dist(uint64_t a, uint64_t b) {
    if (a == b) return 0;
    if ((a >> 63) != (b >> 63)) return (((a | b) << 1) == 0) ? 0 : ~0ull;
    return (a > b) ? (a - b) : (b - a);
}

// ===========================================================================
// 8 つの検査．どれも 1 つの関数で，表示の行を出して合否を返す（main がまとめて終了コードにする）．乱数は main の 1 本を
// 検査の順に続けて引く（検査 1〜6 の入力は前の検査が引いた数に依る．順を変えると入力が変わる）

// ---- 1. split：hi+mid+lo が x*2^d に厳密一致するか -----------------------
// long double（仮数 64 bit）で足しても足りないので，3 本の和は有理数として
// 比較する：それぞれ 2 のべき倍の整数なので，共通の指数へ寄せて整数で比べる．
static bool exact_sum_equals(float hi, float mid, float lo, double x, int d) {
    // x*2^d と hi+mid+lo を「2^(-1200) 単位の整数」として比較する代わりに，
    // __int128 で十分：3 本の指数差は高々 60 bit で，仮数は 24 bit ずつ．
    auto dec32 = [](float f, int& e, __int128& s) {
        const uint32_t u = f32_bits(f);
        const int ee = (int)((u >> 23) & 0xFFu);
        if (ee == 0) { e = 0; s = 0; return; }
        e = ee - 150;
        s = (__int128)(0x800000u | (u & 0x7FFFFFu));
        if (u & 0x80000000u) s = -s;
    };
    auto dec64 = [](double v, int& e, __int128& s) {
        const uint64_t u = f64_bits(v);
        const int ee = (int)((u >> 52) & 0x7FFull);
        if (ee == 0) { e = 0; s = 0; return; }
        e = ee - 1075;
        s = (__int128)((1ull << 52) | (u & 0x000FFFFFFFFFFFFFull));
        if (u >> 63) s = -s;
    };
    int e0, e1, e2, ex; __int128 s0, s1, s2, sx;
    dec32(hi, e0, s0); dec32(mid, e1, s1); dec32(lo, e2, s2); dec64(x, ex, sx);
    ex += d;
    int emin = 1 << 30;
    if (s0) emin = std::min(emin, e0);
    if (s1) emin = std::min(emin, e1);
    if (s2) emin = std::min(emin, e2);
    if (sx) emin = std::min(emin, ex);
    if (emin == (1 << 30)) return true;                 // すべて 0
    // 負の符号付き整数の左シフトは C++17 で未定義なので，絶対値で寄せて符号を戻す
    auto sc = [&](__int128 s, int e) -> __int128 {
        if (!s) return 0;
        const bool neg = s < 0;
        const unsigned __int128 m = (unsigned __int128)(neg ? -s : s) << (e - emin);
        return neg ? -(__int128)m : (__int128)m;
    };
    return sc(s0, e0) + sc(s1, e1) + sc(s2, e2) == sc(sx, ex);
}
static bool check_split(std::mt19937_64& rng) {
    long bad = 0, n = 0;
    auto one = [&](double x, int d) {
        const uint64_t u = f64_bits(x);
        float hi, mid, lo; limbs_of(x, d, hi, mid, lo);
        n++;
        // 正規化後の指数が FP32 の範囲を外れる要素は 0 に潰れる（設計どおり）．
        // 下端は「行の最大値から 126 binade 以上下の要素」で，スライスでも 0 に丸まる．
        // 上端は本番では起きない：正規化後は |x/tau| <= 1 なので eh <= 127 である．
        const int E = (int)((u >> 52) & 0x7FFull);
        if (E == 0 || E - 1075 + d + 179 <= 0 || E - 1075 + d + 180 >= 255) return;
        if (!exact_sum_equals(hi, mid, lo, x, d)) {
            if (bad < 10) std::printf("  MISMATCH x=%.17g d=%d  hi=%a mid=%a lo=%a\n", x, d, hi, mid, lo);
            bad++;
        }
    };
    std::printf("=== 1. split：hi+mid+lo == x*2^d を厳密に検査 ===\n");
    for (int t = 0; t < 400000; t++) {                      // 一様乱数の仮数・指数
        const uint64_t m = rng() & 0x000FFFFFFFFFFFFFull;
        const int e = 900 + (int)(rng() % 250);
        const uint64_t s = (rng() & 1) ? 0x8000000000000000ull : 0ull;
        // d の下げ幅は 0--72 binade（f64_tf.h の「適用範囲」で厳密と確かめた |x*2^d| >= 2^-72）．これより下では
        // 下位 limb が 2^-74 未満になって（SFPU が非正規化数を 0 に潰す場合）先に落ち，s が 0 になる 2^-81 未満では
        // 中位・下位の両方が消える（s を 2^P に置くと 66 binade で切れる．2^(P+7) に置くのはこのため）．
        one(bits_f64(s | ((uint64_t)e << 52) | m), -(e - 1023) - (int)(rng() % 73));
    }
    for (double v : {0.0, 1.0, -1.0, 0.5, -0.5, 0.75, 1e-300, 1e300, 3.0, 1.0 / 3.0,
                     std::ldexp(1.0, 52) + 1.0, std::nextafter(1.0, 2.0)})
        for (int d = -60; d <= 60; d += 3) one(v, d);
    std::printf("  検査 %ld 件，不一致 %ld 件 -> %s\n", n, bad, bad ? "FAIL" : "PASS");
    return bad == 0;
}

// ---- 2. assemble：FP64 に 1 ulp 以内で戻るか ----------------------------
// 参照は __float128 で 3 本を足して FP64 へ丸めたもの．
// 合否は「1 ulp を超える差があるか」で決める．最近接偶数は実装していない
// （f64_tf.h の「保証」を参照：尾崎スキームは丸めの同点の向きに依存しない）ので，
// 1 ulp 差の件数そのものは仕様ではない．検査 5a・6 だけは割合にも 5% の上限を
// 置いてあるが，これは仕様ではなく，将来の変更で桁違いに増えたときの検知用である．
static bool check_assemble(std::mt19937_64& rng) {
    std::printf("=== 2. assemble：3 本 -> FP64（1 ulp 以内）を検査 ===\n");
    long bad2 = 0, n2 = 0, ulp1 = 0;
    auto check_asm = [&](double x, int d, int g) {
        float hi, mid, lo; limbs_of(x, d, hi, mid, lo);
        const uint64_t got = assemble_tf_f64(f32_bits(hi), f32_bits(mid), f32_bits(lo), g,
                                                    0x80000000u, 0u, -63);   // W = 1
        const __float128 ref128 = ((__float128)hi + (__float128)mid + (__float128)lo)
                                * (__float128)std::ldexp(1.0, g);
        const double ref = (double)ref128;
        n2++;
        if (got != f64_bits(ref)) {
            const double gv = bits_f64(got);
            if (ulp_dist(got, f64_bits(ref)) == 1) { ulp1++; return; }
            if (bad2 < 10) std::printf("  MISMATCH x=%.17g d=%d g=%d  got=%.17g ref=%.17g\n",
                                       x, d, g, gv, ref);
            bad2++;
        }
    };
    for (int t = 0; t < 400000; t++) {
        const uint64_t m = rng() & 0x000FFFFFFFFFFFFFull;
        const int e = 900 + (int)(rng() % 250);
        const uint64_t s = (rng() & 1) ? 0x8000000000000000ull : 0ull;
        const double x = bits_f64(s | ((uint64_t)e << 52) | m);
        check_asm(x, -(e - 1023), (int)(rng() % 120) - 60);
    }
    // 下位ビットを 0 にしてから半 ulp を足す形も混ぜる（同点の向きは規定しないので
    // 合否には関わらないが，桁上がりが上位語へ伝わる経路を通しておく）
    for (int t = 0; t < 200000; t++) {
        uint64_t m = rng() & 0x000FFFFFFFFFFFFFull;
        m &= ~0x7Full;                                       // 下位 7 bit を落とす
        if (rng() & 1) m |= 0x40ull;                         // ちょうど半分
        const int e = 1000 + (int)(rng() % 40);
        const double x = bits_f64(((uint64_t)e << 52) | m);
        check_asm(x, -(e - 1023), 0);
    }
    std::printf("  検査 %ld 件，1 ulp 差 %ld 件，それ以上の差 %ld 件 -> %s\n",
                n2, ulp1, bad2, bad2 ? "FAIL" : "PASS");
    return bad2 == 0;
}

// ---- 3. 往復：x -> 3 本 -> x（g=-d）がビット完全か ----------------------
static bool check_roundtrip(std::mt19937_64& rng) {
    std::printf("=== 3. 往復 x -> triple-float -> x ===\n");
    long bad3 = 0, n3 = 0;
    for (int t = 0; t < 400000; t++) {
        const uint64_t m = rng() & 0x000FFFFFFFFFFFFFull;
        const int e = 900 + (int)(rng() % 250);
        const uint64_t s = (rng() & 1) ? 0x8000000000000000ull : 0ull;
        const double x = bits_f64(s | ((uint64_t)e << 52) | m);
        const int d = -(e - 1023);
        const uint64_t u = f64_bits(x);
        float hi, mid, lo; limbs_of(x, d, hi, mid, lo);
        n3++;
        if (assemble_tf_f64(f32_bits(hi), f32_bits(mid), f32_bits(lo), -d,
                                0x80000000u, 0u, -63) != u) {
            if (bad3 < 10) std::printf("  ROUNDTRIP FAIL x=%.17g (%a)\n", x, x);
            bad3++;
        }
    }
    std::printf("  検査 %ld 件，不一致 %ld 件 -> %s\n", n3, bad3, bad3 ? "FAIL" : "PASS");
    return bad3 == 0;
}

// ---- 4. assemble_scaled：定数倍つきで 1 ulp 以内か -------------------------
// 参照は __float128 で 3 本を足して定数（同じく __float128）を掛け，FP64 へ丸めたもの．
// 定数は尾崎 I の再構成が掛ける radix^-(L+2)（ozaki1/ozaki1_plan.hpp の recon_scale）で，基数と L は本番の構成
// （routes.tsv のスキーム I の経路を基数・L でまとめたもの．rl）．全体で 90 万件を構成に等分する
static bool check_scaled(std::mt19937_64& rng, const std::vector<std::pair<int, int>>& rl) {
    std::printf("=== 4. assemble_scaled：radix^-(L+2) 倍つき（尾崎 I の本番の基数と L）===\n");
    long bad4 = 0, n4 = 0, ulp4 = 0;
    for (size_t i = 0; i < rl.size(); i++) std::printf("%s基数 %d L=%d", i ? "，" : "  ", rl[i].first, rl[i].second);
    std::printf("\n");
    const int per4 = 900000 / (int)rl.size();
    for (const auto& [radix, L] : rl) {
        long double w = 1.0L;
        for (int i = 0; i < L + 2; i++) w /= (long double)radix;
        uint32_t Whi, Wlo; int qw; split_const64(w, Whi, Wlo, qw);
        const __float128 w128 = (__float128)Whi * (__float128)4294967296.0 + (__float128)Wlo;
        for (int t = 0; t < per4; t++) {
            const uint64_t m = rng() & 0x000FFFFFFFFFFFFFull;
            const int e = 1000 + (int)(rng() % 60);
            const uint64_t sb = (rng() & 1) ? 0x8000000000000000ull : 0ull;
            const double x = bits_f64(sb | ((uint64_t)e << 52) | m);
            const int d = -(e - 1023), g = (int)(rng() % 80) - 40;
            float hi, mid, lo; limbs_of(x, d, hi, mid, lo);
            const uint64_t got = assemble_tf_f64(f32_bits(hi), f32_bits(mid), f32_bits(lo),
                                                     g, Whi, Wlo, qw);
            // 参照：3 本の和 * (W * 2^qw) * 2^g を __float128 で
            __float128 r = ((__float128)hi + (__float128)mid + (__float128)lo) * w128;
            const double ref = ref_scale(r, qw + g);
            n4++;
            if (got != f64_bits(ref)) {
                if (ulp_dist(got, f64_bits(ref)) == 1) { ulp4++; continue; }
                if (bad4 < 8) std::printf("  MISMATCH radix=%d L=%d x=%.17g g=%d "
                                          "got=%.17g ref=%.17g\n",
                                          radix, L, x, g, bits_f64(got), ref);
                bad4++;
            }
        }
    }
    std::printf("  検査 %ld 件，1 ulp 差 %ld 件，それ以上の差 %ld 件 -> %s\n",
                n4, ulp4, bad4, bad4 ? "FAIL" : "PASS");
    return bad4 == 0;
}

// ---- 5. 正準でない 3 本組 ------------------------------------------------
// 再構成カーネルが出す 3 本組は，必ずしも |mid| <= ulp(hi)/2 を満たさない
// （tfAddS の最下位は近似なので，中位が正準の位置より数 binade 上に来る）．
// 集めるときの桁溢れをここで検査する：上位を 39 bit 左へ寄せると，中位が正準より
// 1 binade 上に来ただけで int64 を越えて符号が反転する（実機の 7040^3（FP8）でも
// 1 要素が誤答になる．f64_tf.h が 37 bit で寄せるのはこのため）．
//
// 5a は実際に出る範囲（中位・下位が正準の位置から高々 16 binade 上まで）で
// 1 ulp 以内を要求する．5b は指数差を無制限にとり，桁落ちが軽い組についてのみ
// 符号と相対精度を確かめる（激しい桁落ちでは下位 limb が窓の底から落ちて，
// 相殺後の値に対する誤差が 1 ulp を超えうる．f64_tf.h の「保証」を参照）．
static bool check_noncanonical(std::mt19937_64& rng) {
    std::printf("=== 5. 正準でない 3 本組 ===\n");
    long bad5 = 0, n5 = 0, ulp5 = 0, bad5b = 0, n5b = 0;
    {
        long double w = 1.0L;
        for (int i = 0; i < 12; i++) w /= 31.0L;
        uint32_t Whi, Wlo; int qw; split_const64(w, Whi, Wlo, qw);
        const __float128 w128 = (__float128)Whi * (__float128)4294967296.0 + (__float128)Wlo;
        auto mk = [&](int e) {
            const uint32_t m = (uint32_t)(rng() & 0x7FFFFFu);
            const uint32_t sb = (rng() & 1) ? 0x80000000u : 0u;
            return sb | ((uint32_t)(e > 0 ? e : 1) << 23) | m;
        };
        auto refval = [&](uint32_t hb, uint32_t mb, uint32_t lb, int g) {
            return q_scale(((__float128)bits_f32(hb) + (__float128)bits_f32(mb)
                          + (__float128)bits_f32(lb)) * w128, qw + g);
        };
        for (int t = 0; t < 500000; t++) {
            const int e0 = 60 + (int)(rng() % 140);
            const int e1 = e0 - 24 + (int)(rng() % 17) - 8;   // 正準 -24 の ±8/+8
            const int e2 = e1 - 24 + (int)(rng() % 17) - 8;
            const uint32_t hb = (rng() % 16) ? mk(e0) : 0u;
            const uint32_t mb = (rng() % 16) ? mk(e1) : 0u;
            const uint32_t lb = (rng() % 16) ? mk(e2) : 0u;
            const int g = (int)(rng() % 80) - 40;
            const uint64_t got = assemble_tf_f64(hb, mb, lb, g, Whi, Wlo, qw);
            const double ref = (double)refval(hb, mb, lb, g);
            n5++;
            if (got != f64_bits(ref)) {
                if (ulp_dist(got, f64_bits(ref)) == 1) { ulp5++; continue; }
                if (bad5 < 8) std::printf("  5a MISMATCH hi=%a mid=%a lo=%a g=%d got=%.17g ref=%.17g\n",
                                          bits_f32(hb), bits_f32(mb), bits_f32(lb), g, bits_f64(got), ref);
                bad5++;
            }
        }
        // 5b：指数差を無制限に（同じ位・逆符号まで）．桁落ちが 8 bit 未満の組だけ見る．
        for (int t = 0; t < 500000; t++) {
            const int e0 = 60 + (int)(rng() % 140);
            const uint32_t hb = mk(e0);
            const uint32_t mb = mk(e0 - (int)(rng() % 61));
            const uint32_t lb = mk(e0 - (int)(rng() % 61));
            const int g = (int)(rng() % 80) - 40;
            const __float128 r = refval(hb, mb, lb, g);
            const __float128 big = refval(hb, 0x00000000u, 0x00000000u, g);
            __float128 ar = r < 0 ? -r : r, ab = big < 0 ? -big : big;
            for (int i = 0; i < 8; i++) ab *= (__float128)0.5;
            if (!(ar > ab)) continue;                       // 桁落ちが激しい組は除く
            n5b++;
            const uint64_t got = assemble_tf_f64(hb, mb, lb, g, Whi, Wlo, qw);
            const double ref = (double)r, gv = bits_f64(got);
            const double rel = std::fabs((gv - ref) / ref);
            if (!(rel < 1e-14)) {                            // 符号反転や大崩れを捕まえる
                if (bad5b < 8) std::printf("  5b BAD hi=%a mid=%a lo=%a g=%d got=%.17g ref=%.17g rel=%.3e\n",
                                           bits_f32(hb), bits_f32(mb), bits_f32(lb), g, gv, ref, rel);
                bad5b++;
            }
        }
    }
    // 5a では 1 ulp の差が 0.2% ほど残る．37 bit 寄せの下から溢れた分を切り捨てる
    // ためで，契約どおり（1 ulp 以内）である．割合を出すのは，将来の変更でこれが
    // 桁で増えたときに気づくためだけで，仕様ではない．
    std::printf("  5a（実際に出る範囲）検査 %ld 件，1 ulp 差 %ld 件（%.2f%%），"
                "それ以上 %ld 件 -> %s\n",
                n5, ulp5, 100.0 * (double)ulp5 / (double)n5, bad5,
                (bad5 == 0 && ulp5 * 20 < n5) ? "PASS" : "FAIL");
    const bool ok5 = (bad5 == 0 && ulp5 * 20 < n5);   // 20 = 5%（回帰検知の目安）
    std::printf("  5b（指数差 無制限，桁落ち 8 bit 未満）検査 %ld 件，相対 1e-14 超 %ld 件 -> %s\n",
                n5b, bad5b, bad5b == 0 ? "PASS" : "FAIL");
    return ok5 && bad5b == 0;
}

// ---- 6. 尾崎 II の出口（frac の 3/6 本 -> FP64，KM 倍つき）-----------------
// KM は法の積 M * 2^-2b で，法の選択が log2 M を窓 2b+1+log2 K のわずかに上に取るので約 2K になる（論文の 3 サイズで
// 2^12.8〜2^14.9）．INT8 経路の 3 サイズの法の集合の値（ozaki2_moduli.h が作る KMl そのもの）を使い，全体で 40 万件を等分する．
// 法が取れないサイズがあれば，この検査だけを FAIL にして残りの検査へ進む（乱数を引くのはこの検査が最後なので，続く検査の
// 入力は変わらない）
static bool check_oz2_exit(std::mt19937_64& rng) {
    std::printf("=== 6. 尾崎 II の出口（frac -> FP64）===\n");
    long bad6 = 0, n6 = 0, ulp6 = 0;
    std::vector<long double> kms;
    for (uint32_t K : {3520u, 7040u, 14080u}) {
        const Ozaki2Moduli mo = ozaki2_choose_moduli_route(K, OZ2_ROUTE_INT8);
        if (!mo.ok) { std::printf("  INT8 K=%u の法が取れない -> FAIL\n", K); return false; }
        kms.push_back(mo.KMl);
        std::printf("  INT8 K=%u の KM = %.6Lg（2^%.2f）\n", K, mo.KMl, (double)log2l(mo.KMl));
    }
    for (const long double KM : kms) {
        uint32_t Whi, Wlo; int qw; split_const64(KM, Whi, Wlo, qw);
        const __float128 w128 = (__float128)Whi * (__float128)4294967296.0 + (__float128)Wlo;
        for (int t = 0; t < 400000 / (int)kms.size(); t++) {
            const bool two = (t & 1);
            const int nl = two ? 6 : 3;
            uint32_t lb[6] = {0, 0, 0, 0, 0, 0};
            // パス 1：(-1/2,1/2] の triple-float．パス 2：2^-49 前後の triple-float．
            auto mk3 = [&](uint32_t* d, int e0) {
                for (int i = 0; i < 3; i++) {
                    const int e = e0 - 24 * i;
                    if (e < 1 || e > 254) { d[i] = 0; continue; }
                    d[i] = ((rng() & 1) ? 0x80000000u : 0u)
                         | ((uint32_t)e << 23) | (uint32_t)(rng() & 0x7FFFFFu);
                }
            };
            mk3(lb, 126 - (int)(rng() % 12));            // 2^-1 .. 2^-12
            if (two) mk3(lb + 3, 126 - 48 - (int)(rng() % 12));
            const int g = (int)(rng() % 60) - 30;
            const uint64_t got = assemble_frac_f64(lb, (uint32_t)nl, g, Whi, Wlo, qw);
            __float128 f = 0;
            for (int i = 0; i < nl; i++) f += (__float128)bits_f32(lb[i]);
            // f -= roundq(f)（実装と同じ規則．1 パスも 2 パスも畳む．f64_tf.h の assemble_frac_f64）
            if (f >= (__float128)0.5) f -= 1;
            else if (f <= (__float128)-0.5) f += 1;
            __float128 r = f * w128;
            const double ref = ref_scale(r, qw + g);
            n6++;
            if (got != f64_bits(ref)) {
                if (ulp_dist(got, f64_bits(ref)) == 1) { ulp6++; continue; }
                if (bad6 < 8) std::printf("  MISMATCH two=%d g=%d got=%.17g ref=%.17g\n",
                                          (int)two, g, bits_f64(got), ref);
                bad6++;
            }
        }
    }
    // 5a と同じ理由で 1 ulp の差が 0.03% ほど残る．割合の閾値は 5a と同じ意味である．
    std::printf("  検査 %ld 件，1 ulp 差 %ld 件（%.3f%%），それ以上の差 %ld 件 -> %s\n",
                n6, ulp6, 100.0 * (double)ulp6 / (double)n6, bad6,
                (bad6 == 0 && ulp6 * 20 < n6) ? "PASS" : "FAIL");
    return bad6 == 0 && ulp6 * 20 < n6;
}

// ---- 7. 畳み込みの端点と窓の端（frac がちょうど ±1/2 と，±1/2 をわずかに越えた 3 本組）----
// (a) 端点：CRT の窓が狭義なので実際には起きないが，f -= roundq(f)
//     （半数は 0 から遠い側）と一致することを確かめる：+1/2 -> -1/2，-1/2 -> +1/2．1 パス（3 本）も 2 パス（6 本）も．
// (b) 窓の端：計算カーネルで FP32 の和 fl(a0+a1) を見て畳もうとすると見逃す形（tools/crt_fold_check.cpp の (1)）
//     (∓1/2, ∓2^-j, 0) は ±1/2 を越えた値で，組み立てが 1 を足し引きして (1/2 - 2^-j) 側へ戻さねばならない
//     （例えば j=30．1 パス構成でも畳まないと ±1 ずれた値を返す）．
//     逆に (±1/2, ∓2^-j, 0) は内側なので変えてはならない．
static bool check_fold_edges() {
    std::printf("=== 7. 畳み込みの端点と窓の端（frac = ±1/2 と ±(1/2 + 2^-j)）===\n");
    uint32_t Whi, Wlo; int qw; split_const64(1.0L, Whi, Wlo, qw);
    int bad7 = 0;
    for (int nl : {3, 6})
        for (int sgn = 0; sgn < 2; sgn++) {
            // 3 本組で ±1/2 ちょうどを作る（hi = ±0.5，mid = lo = 0）
            uint32_t lb[6] = {0, 0, 0, 0, 0, 0};
            lb[0] = (sgn ? 0x80000000u : 0u) | 0x3F000000u;   // ±0.5
            const uint64_t got = assemble_frac_f64(lb, (uint32_t)nl, 0, Whi, Wlo, qw);
            const double want = sgn ? 0.5 : -0.5;             // roundq と同じ向き
            std::printf("  nl=%d frac=%+.1f -> %+.1f（期待 %+.1f）%s\n", nl, sgn ? -0.5 : 0.5,
                        bits_f64(got), want, (bits_f64(got) == want) ? "" : "  <-- FAIL");
            if (bits_f64(got) != want) bad7++;
        }
    int n7b = 0;
    for (int nl : {3, 6})
        for (int j = 24; j <= 90; j++)
            for (int sgn = 0; sgn < 2; sgn++)
                for (int outside = 0; outside < 2; outside++) {
                    // hi = ±1/2．outside なら mid は hi と同じ向き（値が ±1/2 の外），そうでなければ逆向き（内側）
                    const float h = sgn ? -0.5f : 0.5f;
                    const float m = (outside ? 1.0f : -1.0f) * std::copysign(std::ldexp(1.0f, -j), h);
                    uint32_t lb[6] = {f32_bits(h), f32_bits(m), 0u, 0u, 0u, 0u};
                    if (nl == 6) { lb[3] = f32_bits(std::ldexp(m, -40)); }   // 2 パスの第 2 パスぶん（同じ向きの小さい値）
                    __float128 f = (__float128)h + (__float128)m + (nl == 6 ? (__float128)std::ldexp(m, -40) : 0);
                    if (f >= (__float128)0.5) f -= 1;
                    else if (f <= (__float128)-0.5) f += 1;
                    const double want = (double)f;               // |f| < 1/2 で 2^-j の位まで FP64 に載る（j <= 90 なら 53 bit 以内とは
                                                                 // 限らないので 1 ulp まで許す）
                    const double got = bits_f64(assemble_frac_f64(lb, (uint32_t)nl, 0, Whi, Wlo, qw));
                    n7b++;
                    if (got != want && ulp_dist(f64_bits(got), f64_bits(want)) > 1) {
                        if (bad7 < 8) std::printf("  MISMATCH nl=%d j=%d h=%+.1f outside=%d got=%.17g want=%.17g\n",
                                                  nl, j, (double)h, outside, got, want);
                        bad7++;
                    }
                }
    std::printf("  窓の端（±1/2 の内外 2^-j，j=24..90）%d 件\n", n7b);
    std::printf("  -> %s\n", bad7 ? "FAIL" : "PASS");
    return bad7 == 0;
}

// ---- 8. 窓の外へ落ちる limb（相殺と，固定小数の最下位より下）--------------
// どちらも，窓を最大の limb の位置で決めて外の limb を落とすだけの組み立てでは 0 を返してしまう経路である．
static bool check_dropped_limbs() {
    std::printf("=== 8. 窓の外へ落ちる limb ===\n");
    uint32_t Whi8, Wlo8; int qw8; split_const64(1.0L, Whi8, Wlo8, qw8);
    int bad8 = 0;
    // (a) scheme I：上位 2 本が厳密に相殺し，下位だけが残る
    for (int d = 40; d <= 100; d += 5) {
        const float lo = std::ldexp(1.0f, -d);
        const double want = (double)lo;
        const double got = bits_f64(assemble_tf_f64(f32_bits(1.0f), f32_bits(-1.0f), f32_bits(lo),
                                                0, Whi8, Wlo8, qw8));
        if (got != want) { std::printf("  MISMATCH(I) lo=2^-%d got=%g want=%g\n", d, got, want); bad8++; }
    }
    // (b) scheme II：固定小数の最下位（2^-124）の近くと，それより下にビットを持つ limb．固定小数の契約は
    //     「limb ごとに，絶対値の 2^-124 未満のビットを落として（0 へ向けて切り捨てて）足す」（f64_tf.h の a128_term）で，
    //     期待値はこの契約から整数で独立に作る：sgn(x) * floor(|x| * 2^124) * 2^-124（KM = 1 なので FP64 に厳密に載る）．
    //     |x| = (2^23 + 仮数) * 2^(-d-23) なので floor(|x| * 2^124) は (2^23 + 仮数) を 101-d だけずらした整数である．
    //     入力は 2 のべき 2^-d（d = 95..126．124 は最下位ちょうど，125・126 は丸ごと下で 0 になる）と，仮数を全部・最下位だけ・
    //     最上位と最下位・交互に立てた値（仮数の一部だけが 2^-124 をまたぐ．指数 -d で仮数の下端は 2^-(d+23)）を両符号で．
    int n8b = 0;
    for (int d = 95; d <= 126; d++)
        for (uint32_t frac : {0u, 0x7FFFFFu, 0x000001u, 0x400001u, 0x2AAAABu})
            for (int sgn = 0; sgn < 2; sgn++) {
                const uint32_t xb = (sgn ? 0x80000000u : 0u) | ((uint32_t)(127 - d) << 23) | frac;
                const float x = bits_f32(xb);
                uint32_t lb[3] = { xb, 0u, 0u };
                const uint64_t m24 = (1u << 23) | frac;
                const uint64_t kept = (d <= 101) ? (m24 << (101 - d)) : (m24 >> (d - 101));   // 2^-124 単位（2^30 未満）
                const double want = std::ldexp(sgn ? -(double)kept : (double)kept, -124);
                const double got = bits_f64(assemble_frac_f64(lb, 3, 0, Whi8, Wlo8, qw8));
                n8b++;
                if (got != want) {
                    if (bad8 < 8) std::printf("  MISMATCH(II) limb=%a got=%a want=%a\n", (double)x, got, want);
                    bad8++;
                }
            }
    std::printf("  (b) 固定小数の最下位 2^-124 の近く（limb の指数 -95..-126，仮数 5 通り，両符号）%d 件\n", n8b);
    // (c) 両方が重なる場合
    uint32_t lb2[3] = { f32_bits(0.25f), f32_bits(-0.25f), f32_bits(std::ldexp(1.0f, -102)) };
    if (bits_f64(assemble_frac_f64(lb2, 3, 0, Whi8, Wlo8, qw8)) != std::ldexp(1.0, -102)) {
        std::printf("  MISMATCH(II) 相殺＋小 limb\n"); bad8++; }
    std::printf("  -> %s\n", bad8 ? "FAIL" : "PASS");
    return bad8 == 0;
}

int main(int argc, char** argv) {
    if (argc != 1) { std::fprintf(stderr, "usage: %s（引数は取らない）\n", argv[0]); return 2; }
    // 検査 4 の (基数, L) は尾崎 I の本番の構成（routes.tsv が読めなければ何も検査せずに止める）
    std::vector<std::pair<int, int>> rl;
    for (const auto& c : oz_sim::scheme1_configs("tfconv")) {
        bool dup = false;
        for (const auto& x : rl) dup = dup || (x.first == c.radix && x.second == c.L);
        if (!dup) rl.push_back({c.radix, c.L});
    }
    std::mt19937_64 rng(12345);
    // 8 つとも最後まで流してからまとめて返す（どれかが FAIL でも後の検査の表示を出す．表示が FAIL なら終了コードも 1）
    const bool ok1 = check_split(rng);
    const bool ok2 = check_assemble(rng);
    const bool ok3 = check_roundtrip(rng);
    const bool ok4 = check_scaled(rng, rl);
    const bool ok5 = check_noncanonical(rng);
    const bool ok6 = check_oz2_exit(rng);
    const bool ok7 = check_fold_edges();
    const bool ok8 = check_dropped_limbs();
    return (ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8) ? 0 : 1;
}
