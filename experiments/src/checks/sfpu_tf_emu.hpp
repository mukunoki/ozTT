// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// sfpu_tf.h のタイル演算をホストでビット単位に模擬し，その上で本番の triple-float の手続き（sfpu_tf_proc.h）を組む
// （検査用．本番のドライバは取り込まない）
// ---------------------------------------------------------------------------
// 取り込むのは tools のホスト単体の検査・模擬（tools/oz_sim.h を通して）と src/checks/tfmul_probe.cpp（実機との突き合わせ）．
// SFPU の模擬はここに 1 つだけ置く：検査ごとに書き写すと，sfpu_tf.h を直したときに直す場所が増え，非正規化数の
// 扱いのような細部も検査ごとに食い違う．
// 手続き（TwoSum・tfMulR・tfMulP2・tfAddS・tfAddSmall・ModRound）は本番と同じ common/kernels/sfpu_tf_proc.h のテンプレートを，
// DST の 8 本を Dst::r[8] に置き換えたラッパ Dst の上で組む（写しを持たない）．ここが持つのはタイル演算 1 つの模型
// （下の「SFPU の演算の模型」）と，比較用の版への切り替え・計装だけである．模型が実機とビット単位で一致するかは
// src/checks/tfmul_probe.cpp が実機と比べる（比べる手続きは同ファイルの冒頭．tools の検査だけが使う tfAddSmall・ModRound は
// 実機と比べていないが，本番と同じ手続きなので模型が一致すれば一致する）．
//
// SFPU の演算の模型
//   タイル演算 1 つ（Add・Sub・Mul・MulS・AddS）は SFPMAD 1 本で，FP32 へ最近接偶数で丸める．
//   非正規化数は入力も出力も符号つきの 0 に潰す（Dst::ftz）．
//   0 の符号は実機に合わせる（src/checks/tfmul_probe.cpp で実機と比べた．2026-10-22）：積（Mul・MulS）は入力に 0（0 に潰した
//   非正規化数を含む）があれば +0（IEEE の符号つきの -0 にはならない）．Rnd は tt-llk の _round_even_（|v|+2^23-2^23．同点は
//   偶数）で，結果が 0 なら +0（Rnd(-0.3)・Rnd(-0.5)・Rnd(-0) も +0），0 でなければ元の符号．値は IEEE と同じで，0 の符号だけが
//   違う．本番の値域の外の積（非零どうしで下へ溢れる積の 0 の符号，NaN の符号）は実機と合わせていない．
//   Ltz は符号ビットを見る（-0 も 1 になる．実機とは比べていない）．
//
// 切り替えと計装は Dst のメンバに持つ（大域変数にすると，比較用の版へ切り替えたまま戻し忘れた検査が，
// 後の検査を比較用の版で回してしまう）．既定は本番のカーネルと同じ動作である：
//   final_twosum  【比較用】tfMulR・tfMulP2 の最後の FastTwoSum（3 演算）を Knuth の TwoSum（6 演算）に置き換える
//   small_fast    【比較用】tfAddSmall の第 2 段の TwoSum を Dekker の FastTwoSum に置き換える（sfpu_tf_proc.h が使わない理由を書く形）
//   fts           最後の FastTwoSum の前提（「a の指数 >= b の指数」か a,b の一方が 0）と誤差を数える先（nullptr なら数えない）．
//                 前提を満たせば s + t = a + b が厳密に成り立つ（Dekker）
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>
#include "common/kernels/f64_tf.h"
#include "common/kernels/sfpu_tf_proc.h"   // 本番の手続き（sfpu_proc）と DST レジスタの名前 dst
#include "common/oz_bits.h"   // f32_bits・bits_f32
#include "common/oz_format_host.h"         // OZ_FMTS（本番の形式の並び）
#include "ozaki1/ozaki1_param.h"           // スキーム I の形式ごとの既定の基数 oz1_param(f).radix
#include "ozaki2/kernels/oz2_limits.h"     // スキーム II の桁の基数 2^OZ2_DIGIT_BITS

namespace sfpu_emu {

// 本番の基数倍（tfMulRadix・tfMulR・tfMulP2）が使う基数：スキーム I の形式ごとの既定の基数（oz1_param(f).radix．OZ_FMTS の
// 形式から重複を除く）と，with_ii なら スキーム II の桁の基数 2^OZ2_DIGIT_BITS（oz2_digits.h の tfMulP2）．並びは 2 のべき
// （小さい順）の後に 2 のべきでないもの（小さい順）．基数倍を確かめる検査（tools/tfmul_check.cpp の [4][5]・
// src/checks/tfmul_probe.cpp の mul）はこれを回す（本番の表で基数を足したり変えたりしたとき，検査が古い基数のまま通らないように）
inline std::vector<float> production_radices(bool with_ii) {
    std::vector<float> p2, other;
    auto add = [&](float b) {
        std::vector<float>& v = (f32_bits(b) & 0x7FFFFFu) == 0u ? p2 : other;
        if (std::find(v.begin(), v.end(), b) == v.end()) v.push_back(b);
    };
    for (OzFmt f : OZ_FMTS) add((float)oz1_param(f).radix);
    if (with_ii) add(std::ldexp(1.0f, OZ2_DIGIT_BITS));
    std::sort(p2.begin(), p2.end());
    std::sort(other.begin(), other.end());
    p2.insert(p2.end(), other.begin(), other.end());
    return p2;
}

inline int fexp(float x) { int e; std::frexp(x, &e); return e; }   // x != 0

// タイル演算 1 つの模型（冒頭．Dst の演算はこれを呼ぶ．on は非正規化数を 0 に潰すか（Dst::ftz））
inline float ftz(float x, bool on = true) {
    if (on && x != 0.0f && std::fabs(x) < FLT_MIN) return std::copysign(0.0f, x);
    return x;
}
inline float fadd(float a, float b, bool on = true) { return ftz(ftz(a, on) + ftz(b, on), on); }
inline float fsub(float a, float b, bool on = true) { return ftz(ftz(a, on) - ftz(b, on), on); }
// 積：入力に 0 があれば +0（実機は符号つきの 0 を返さない）
inline float fmul(float a, float b, bool on = true) {
    a = ftz(a, on); b = ftz(b, on);
    if (a == 0.0f || b == 0.0f) return 0.0f;
    return ftz(a * b, on);
}
// tt-llk の _round_even_：|v| < 2^23 なら |v| + 2^23 - 2^23（最近接偶数）に元の符号を付ける．結果が 0 なら +0
inline float frnd(float v, bool on = true) {
    v = ftz(v, on);
    const float a = std::fabs(v);
    if (!(a < 8388608.0f)) return v;
    const float t = (a + 8388608.0f) - 8388608.0f;
    return (t == 0.0f) ? 0.0f : std::copysign(t, v);
}

// DST レジスタの役割（本番の sfpu_tf_proc.h の namespace dst そのもの）
namespace dst = ::dst;

// 最後の FastTwoSum の計装（呼び出し側が持ち，Dst::fts に渡す）
struct FtsStat {
    uint64_t calls = 0;       // 呼び出し回数
    uint64_t viol = 0;        // 前提（指数 a >= 指数 b，または a,b の一方が 0）を満たさない回数
    uint64_t inexact = 0;     // s + t != a + b（前提を満たさずに実際に誤差が出た回数）
    double worst_loss_rel = 0; // |(a+b)-(s+t)| / |b| の最大（inexact のときだけ）
    // 厳密な比較：a + b と s + t を __float128（仮数 113 bit）で足して比べる．FP32 の 2 数の和は指数の差が 89 以下なら
    // 113 bit に収まって厳密に比べられる．差が 89 を超える組では和が丸まり，s + t と a + b が違っても同じ値に丸まれば
    // inexact に数えない（例 a = 2^-120，b = 1，s = 1，t = 0）．そうした組の小さい方は大きい方の 2^-89 未満なので，前提
    // （指数 a >= 指数 b）を満たしていれば FastTwoSum は厳密で inexact は起きず，満たしていなければ viol に数える．
    // 見逃す差は大きい方の 2^-89 未満で，tools/tfmul_check.cpp の判定（誤差 2^-66 以下）には効かない．
    void record(float a, float b, float s, float t, bool verbose) {
        calls++;
        const bool pre = (a == 0.0f || b == 0.0f || fexp(a) >= fexp(b));
        if (!pre) viol++;
        const __float128 x = (__float128)a + (__float128)b, y = (__float128)s + (__float128)t;
        if (x != y) {
            inexact++;
            const __float128 d = x - y;
            const double rel = (double)((d < 0 ? -d : d) / ((__float128)std::fabs(b) + (b == 0.0f)));
            if (rel > worst_loss_rel) worst_loss_rel = rel;
            if (verbose) std::printf("      FTS inexact a=%a b=%a s=%a t=%a\n", a, b, s, t);
        }
    }
};

// DST の 8 本（fp32_dest_acc，dst_full_sync）
struct Dst {
    float r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool ftz = true;             // 非正規化数を入力も出力も 0 に潰す（SFPU）
    bool final_twosum = false;   // 【比較用】最後の FastTwoSum を TwoSum に（冒頭）
    bool small_fast = false;     // 【比較用】tfAddSmall の第 2 段を FastTwoSum に（冒頭）
    FtsStat* fts = nullptr;      // 最後の FastTwoSum の計装
    bool fts_verbose = false;    // 計装：誤差の出た組を表示する

    void Add(int a, int b, int o) { r[o] = fadd(r[a], r[b], ftz); }
    void Sub(int a, int b, int o) { r[o] = fsub(r[a], r[b], ftz); }
    void Mul(int a, int b, int o) { r[o] = fmul(r[a], r[b], ftz); }
    void Cpy(int s, int d) { r[d] = r[s]; }
    void MulS(int d, float s) { r[d] = fmul(r[d], s, ftz); }
    void AddS(int d, float s) { r[d] = fadd(r[d], s, ftz); }
    void Set(int d, float s) { r[d] = s; }
    void Rnd(int d) { r[d] = frnd(r[d], ftz); }
    // 符号ビットを見る（立っていれば 1．-0 も 1 になり，IEEE の比較 v < 0 とは -0 で違う．実機とは比べていない）
    void Ltz(int d) { r[d] = std::signbit(r[d]) ? 1.0f : 0.0f; }
    void Clamp(int d, float lo, float hi) { if (r[d] < lo) r[d] = lo; else if (r[d] >= hi) r[d] = hi; }
    void TwoSum(int x, int y, int s, int e, int a, int b) { sfpu_proc::TwoSum(*this, x, y, s, e, a, b); }
    static constexpr float SPL = 4097.0f;   // Dekker 分割の定数（本番の SfpuDev::SPL のビット列 0x45800800 の値）
    // 最後の正規化：(a,b) -> (s 番, t 番)．本番は FastTwoSum（3 演算），比較用に TwoSum．
    // w1, w2 は作業用（FastTwoSum は w1 だけ使う）．a, b のレジスタは壊さない．
    void FinalFts(int a, int b, int s, int t, int w1, int w2) {
        const float av = r[a], bv = r[b];
        if (final_twosum) TwoSum(a, b, s, t, w1, w2);
        else sfpu_proc::FastTwoSum(*this, a, b, s, t, w1);
        if (fts) fts->record(av, bv, r[s], r[t], fts_verbose);
    }
    // tfAddSmall の第 2 段：本番は TwoSum(a1, v)．small_fast なら FastTwoSum（s = a1 + v，e = v - (s - a1)）に
    // 置き換える【比較用．本番に無い版】
    void SmallTwoSum() {
        using namespace ::dst;
        if (!small_fast) { sfpu_proc::SmallTwoSum(*this); return; }
        Add(acc_mid, term, tmp0); Sub(tmp0, acc_mid, tmp1); Sub(term, tmp1, tmp2); Cpy(tmp0, acc_mid);
    }
};

constexpr float SPL = Dst::SPL;

// 本番の手続き（sfpu_tf_proc.h）を Dst の上で呼ぶ入口（tools の検査と src/checks/tfmul_probe.cpp が使う名前）
inline void tfMulR(Dst& D, float rdx) { sfpu_proc::tfMulR(D, rdx); }       // 作業 3..7
inline void tfMulP2(Dst& D, float rdx) { sfpu_proc::tfMulP2(D, rdx); }     // 作業 3..7
// 本番の tfMulRadix：基数の仮数部が 0（2 のべき）なら tfMulP2（本番はコンパイル時に振り分ける）
inline void tfMulRadix(Dst& D, float rdx) {
    if (sfpu_proc::radix_is_pow2(f32_bits(rdx))) tfMulP2(D, rdx); else tfMulR(D, rdx);
}

// 本番の tfAddS<NL>：3 本組（NL=3）か 2 本組（NL=2）へ term（reg3）を足す．作業 4..6，reg7 は触らない
template <int NL>
inline void tfAddS(Dst& D) { sfpu_proc::tfAddS<(uint32_t)NL>(D); }
inline void tfAddS3(Dst& D) { tfAddS<3>(D); }   // 3 本組の別名（src/checks/tfmul_probe.cpp が実機と比べる名前）

// 本番の tfAddSmall<NL>：第 1 段 TwoSum(a0, v) を飛ばし，term を a1 の段から入れる（keep は壊さない）．
// Dst::small_fast を立てると第 2 段が比較用の FastTwoSum になる（Dst::SmallTwoSum）
template <int NL>
inline void tfAddSmall(Dst& D) { sfpu_proc::tfAddSmall<(uint32_t)NL>(D); }

// 本番の ModRound：o = d - round(d*rinv)*m（s は作業用で d とは別．o と d は同じでもよい）
inline void ModRound(Dst& D, int d, int o, int s, float m, float rinv) { sfpu_proc::ModRound(D, d, o, s, m, rinv); }
inline void ModRound(Dst& D, int d, int s, float m, float rinv) { ModRound(D, d, d, s, m, rinv); }

// --- 本番の手順の 1 段（検査と実機の突き合わせが共用する）--------------------------------------------
// FP64 x を slice_f64_reader.cpp（split_f64，指数シフト d）+ slice_tf_core.h の tf_load_f64 と同じ手順で
// 3 本組 (0,1,2) にする（reg3 は s のビット列のまま残る）
inline void load_f64(Dst& D, double x, int d = 0) {
    uint64_t b; std::memcpy(&b, &x, 8);
    uint32_t hi, braw, craw, s;
    split_f64((uint32_t)(b >> 32), (uint32_t)b, d, hi, braw, craw, s);
    D.r[3] = bits_f32(s); D.r[1] = bits_f32(braw); D.r[2] = bits_f32(craw); D.r[0] = bits_f32(hi);
    D.AddS(1, -12582912.0f); D.Mul(1, 3, 1);
    D.AddS(2, -8388608.0f); D.Mul(2, 3, 2); D.MulS(2, OZ_TF_LO_SCALE);
}
// 尾崎 I のスライサの 1 段（ozaki1/kernels/slice_tf_body.h．slice_tf_num.cpp・slice_tf_sm8.cpp）：t *= β；d = clamp(round(t0))；t0 -= d
inline int slice_step(Dst& D, float beta, float dmax) {
    tfMulRadix(D, beta);
    D.Cpy(0, 7); D.Rnd(7); D.Clamp(7, -dmax, dmax);
    D.Sub(0, 7, 0);
    return (int)D.r[7];
}

// 正規化後の |x| <= T の FP64 入力．種類 kind：
//   0 一様（指数 0..-70）
//   1 基数 β の i 桁目の近くでほぼ整数（x β^i が整数から ulp(t0) の数倍．桁を引いた後の上位 limb が ±ulp だけ
//     残り，中位と打ち消し合う形を作る）
//   2 split_f64 の最近接の同点（仮数の下位 30 bit がちょうど 2^29：|mid+lo| = ulp(hi)）
//   3 仮数の中の長い 1 か 0 の並び
//   4 split_f64 の尾の絶対値の和が ulp(hi) を超える形（仮数の下位 30 bit が 2^29 + c，1 <= c <= 127：上位 23 bit が
//     切り上がり，mid = -2^(P+29)，lo = c 2^P で |mid| + |lo| = ulp(hi) + c 2^P．入口の格子 u = 2^(P+30) = 2 ulp(hi)
//     に対しては |mid| + |lo| < u．sfpu_tf_proc.h の「本番の呼び出しが H を満たすこと」）
// split_f64 が厳密なのは行／列の最大値（ここでは T）から 73 binade 以内（f64_tf.h の適用範囲）なので，
// その外へは出さない（外の要素は入口の 3 本組がすでに近似で，基数倍の検査の対象にならない）．
inline double gen_x(std::mt19937_64& g, int kind, double beta, int p, double T) {
    std::uniform_real_distribution<double> U(-1.0, 1.0), H(0.5, 1.0);
    std::uniform_int_distribution<int> Ek(0, 70);
    auto sg = [&]() { return (g() & 1) ? -1.0 : 1.0; };
    auto inrange = [&](double x) { return std::fabs(x) >= std::ldexp(T, -72) ? x : std::copysign(std::ldexp(T, -72), x); };
    switch (kind) {
    case 0: return sg() * T * std::ldexp(H(g), -Ek(g));
    case 1: {
        std::uniform_int_distribution<int> Ki(1, p < 4 ? p : 4);
        const int i = Ki(g);
        const double bi = std::pow(beta, i);
        const double N = std::floor(U(g) * bi * T);
        const double t0 = std::fmax(1.0, std::fabs(N) / std::pow(beta, i - 1)) * beta;   // 桁を切る直前の上位の大きさの目安
        const double u = std::ldexp(1.0, std::ilogb(t0) - 23);
        std::uniform_int_distribution<int> J(-6, 6);
        const double eps = u * (J(g) * 0.25 + std::ldexp(U(g), -30));
        double x = (N + eps) / bi;
        if (std::fabs(x) > T) x = std::copysign(T, x) * 0.999;
        return inrange(x);
    }
    case 2: {
        uint64_t b; double x = sg() * T * std::ldexp(H(g), -Ek(g) / 4);
        std::memcpy(&b, &x, 8);
        b = (b & ~0x3FFFFFFFull) | 0x20000000ull | ((g() & 1) ? 0x40000000ull : 0);
        std::memcpy(&x, &b, 8);
        return std::fabs(x) > T ? x * 0.5 : x;
    }
    case 4: {
        uint64_t b; double x = sg() * T * std::ldexp(H(g), -Ek(g) / 4);
        std::memcpy(&b, &x, 8);
        b = (b & ~0x3FFFFFFFull) | (0x20000000ull + 1 + g() % 127);
        std::memcpy(&x, &b, 8);
        return std::fabs(x) > T ? x * 0.5 : x;
    }
    default: {
        uint64_t b; double x = sg() * T * std::ldexp(H(g), -Ek(g) / 8);
        std::memcpy(&b, &x, 8);
        const int lo = (int)(g() % 30), len = 5 + (int)(g() % 20);   // 仮数部（52 bit）の中だけ
        const uint64_t m = ((1ull << len) - 1) << lo;
        b = (g() & 1) ? (b | m) : (b & ~m);
        std::memcpy(&x, &b, 8);
        return std::fabs(x) > T ? x * 0.5 : x;
    }
    }
}

// tfMulR の最後の FastTwoSum の前提が崩れるとすればここ，という形を作る（正規化された 3 本組の中の最悪の構成．
// sfpu_tf_proc.h の補題 R により実際には崩れない．tools/tfmul_check.cpp の [5](b) がそれを確かめる）．
// x0 の β 倍の丸め誤差 E0 = x0 β - fl(x0 β) を，中位・下位の β 倍がほぼ打ち消す：x1 = RN(-E0/β)，
// x2 = RN(残り/β) にわずかなずれを載せる．|x1 + x2| <= ulp(x0)，|x2| <= ulp(x1)/2 の正規化された組で，
// tfMulR の中では S1 = fl(P1 + F) が -E0 に近く S2 = fl(E0 + S1) が打ち消し合って小さくなる．
// 作れなかった（正規化の条件を外れた）ときは false．
inline bool gen_r_cancel(std::mt19937_64& g, float beta, float* x) {
    std::uniform_real_distribution<double> H(0.5, 1.0);
    const double x0 = (double)(float)(std::ldexp(H(g), -(int)(g() % 12) - 1) * ((g() & 1) ? -1 : 1));
    const double p = x0 * beta;                    // 24 + 8 bit なので double で厳密
    const double E0 = p - (double)(float)p;
    if (E0 == 0.0) return false;
    const float x1 = (float)(-E0 / beta);
    const double rem = -E0 - (double)x1 * beta;    // 厳密（打ち消しの残り）
    const int sh = (int)(g() % 40);
    const double tweak = (g() % 3 == 0) ? 0.0 : std::ldexp((double)((int)(g() % 7) - 3), -sh) * std::fabs(rem / beta);
    const float x2 = (float)(rem / beta + tweak);
    const double ux1 = std::ldexp(1.0, std::ilogb(x1) - 23), ux0 = std::ldexp(1.0, std::ilogb(x0) - 23);
    if (x1 == 0.0f || std::fabs((double)x2) > ux1 / 2 || std::fabs((double)x1 + x2) > ux0) return false;
    x[0] = (float)x0; x[1] = x1; x[2] = x2;
    return true;
}

}  // namespace sfpu_emu
