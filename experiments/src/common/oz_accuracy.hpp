// SPDX-License-Identifier: Apache-2.0
#pragma once
// 精度の指標と合否（host_eval.hpp の傘の下）：FP64 の結果の Dot2 の参照解に対する相対誤差・実効ビット数・結果のビット列の
// ハッシュ（尾崎の 2 つのドライバ・基準実装と CPU の DGEMM の参照で共通の accuracy）と，整数の結果の完全一致（exact_check）
// ---------------------------------------------------------------------------
// 環境変数（値は host_eval.hpp の check_env がデバイスを開く前にも検査する）
//   OZ_MIN_BITS=b 精度の合否の基準［bit］（既定 53．accuracy_ok）
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>   // std::cmp_equal（exact_check）
#include <vector>
#include "oz_env.hpp"   // つまみの読み方（相対で取り込む理由は host_eval.hpp の冒頭）

namespace oz_eval {

// ---- 整数の参照値と完全一致するかの確認（INT8 / FP8 の検査用） -------------
// 一致は整数どうしなら整数のまま比べる（double に直すと 2^53 を超える値で隣の整数どうしが同じ値に丸まり，
// 違う整数を一致と数える．符号の違う型どうしも std::cmp_equal で値として比べる）．浮動小数どうしの組（fp8_probe の
// float の出力と double の参照）は double で比べる（float と double の値はそのまま double で表せる）．整数と浮動小数を
// 混ぜた組は，整数側を double に丸めると 2^53 を超える値で違う値を一致と数えるので，コンパイル時に拒む（使う組は無い）．
// 長さが違う・空の組は比べられないので，全要素の不一致として返す（短い got で範囲外を読む・空の want で合格する穴を塞ぐ）．
// max_abs_golden は表示用で，2^53 を超えると丸まる
struct Exact { size_t mismatches; double max_abs_golden; };
template <class T, class U>
inline Exact exact_check(const std::vector<T>& got, const std::vector<U>& want) {
    static_assert(std::is_integral_v<T> == std::is_integral_v<U>,
                  "exact_check は整数どうしか浮動小数どうしだけを比べる（混ぜると整数側が double に丸まる）");
    Exact e{0, 0.0};
    if (got.size() != want.size() || want.empty()) {
        std::fprintf(stderr, "ERROR: exact_check の配列の長さが違うか空である（got %zu 要素，want %zu 要素）\n",
                     got.size(), want.size());
        e.mismatches = std::max<size_t>(1, std::max(got.size(), want.size()));
        return e;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        e.max_abs_golden = std::max(e.max_abs_golden, std::fabs((double)want[i]));
        bool same;
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) same = std::cmp_equal(got[i], want[i]);
        else same = (double)got[i] == (double)want[i];
        if (!same) ++e.mismatches;
    }
    return e;
}

// ---- 結果・入力のビット列のハッシュ（CHECKSUM の chash，tools/dgemmcheck の ahash・bhash，fp8_probe の chash）----
// FNV-1a の更新式（バイトごとに h ^= byte; h *= 1099511628211）を，配列のバイトをメモリ上の順（この計算機は little
// endian なので各語の下位のバイトから）に当てる．初期値は kHashInit = 1469598103934665603（0x14650fb0739d0383）で，
// 標準の 64 bit FNV-1a の初期値 14695981039346656037（0xcbf29ce484222325）とは違う（末尾の 1 桁が欠けた値）．過去の
// ログのハッシュと比べられるようにこの値のまま使うので，標準の FNV-1a の実装で計算し直しても一致しない．
// 下の accuracy の chash も同じ規則で求める（DGEMM の参照と共有する定義なので，本体はこの関数を呼ばずに同じ
// ループを持つ．host_eval.hpp の冒頭）
inline constexpr uint64_t kHashInit = 1469598103934665603ull;
inline uint64_t fnv64(const void* p, size_t nbytes, uint64_t h = kHashInit) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < nbytes; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
template <class T> inline uint64_t fnv64(const std::vector<T>& v) { return fnv64(v.data(), v.size() * sizeof(T)); }

// CONFIG 行の chash 欄（結果 C を行優先に並べた FP64 のビット列の fnv64．16 桁の 16 進）．検証の有無に依らず出し，
// 下の CHECKSUM の chash と同じ値になる．到達精度のログと性能のログ（OZ_NOVERIFY）の出力が同じことを照合が見る
// （scripts/lib/oz_logs.py の config_mismatch．2026-11-04 から）
inline std::string chash_hex(const std::vector<double>& C) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long)fnv64(C));
    return s;
}

// chash は結果 C（行優先）の FP64 ビット列のハッシュ（上の fnv64 と同じ規則）．実装を変えたときに
// 「誤差の値が同じ」より強い「結果がビット単位で同じ」を確かめるために出す（CHECKSUM 行は検証時だけ．検証を省く性能の
// 実行も同じ値を CONFIG の chash に出す．上の chash_hex）．
struct Acc { double rel_fro, ebits, maxrel; uint64_t chash = 0; };

// 到達精度が基準を満たすかを判定し，満たさなければ理由を出して false を返す．
// 基準は OZ_MIN_BITS で上書きできる（既定 53．FP64 の仮数幅）．上書きは 1 以上 99 以下の
// 有限な数に限る．読めない値を atof のように 0 と読むと，どんな精度でも合格にしてしまう．
// 精度の合否の基準［bit］（OZ_MIN_BITS．既定 53）
inline double min_bits() { return oz_env::f64("OZ_MIN_BITS", 53.0, 1.0, 99.0); }

// 検証を省いた実行（OZ_NOVERIFY）では判定できないので，その旨を出して true を返す．
// 呼び出し側はこの値を終了コードへ反映する：精度が出ていない実行を成功として
// 集計すると，測定スクリプトが誤りを見逃す．
inline bool accuracy_ok(const Acc& acc, bool verified) {
    if (!verified) {
        std::printf("Verify   : skipped (OZ_NOVERIFY) -- 精度は判定していない\n");
        return true;
    }
    const double need = min_bits();
    if (!std::isfinite(acc.rel_fro) || !std::isfinite(acc.ebits)) {
        std::printf("Verify   : FAILED -- 相対誤差が有限でない\n");
        return false;
    }
    if (acc.ebits < need) {
        std::printf("Verify   : FAILED -- 実効 %.2f bit が基準 %.2f bit を下回る\n",
                    acc.ebits, need);
        return false;
    }
    std::printf("Verify   : ok (実効 %.2f bit >= 基準 %.2f bit)\n", acc.ebits, need);
    return true;
}

// Dot2 が返す非評価和は，強い相殺では |lo| << |hi| とは限らない．TwoSum で和と残差へ
// 正規化してから比較する．そのまま (C-hi)-lo とすると，hi=-2^53, lo=2^53, C=1 さえ
// 誤差0になる．和を1語へ丸めるだけでも参照の低位語を失うので，残差も保持する．
inline std::pair<double, double> accuracy_reference(double hi, double lo) {
    const double s = hi + lo, bb = s - hi;
    return {s, (hi - (s - bb)) + (lo - bb)};
}

// 行優先 M*N の結果に対する ||C-ref||_F / ||ref||_F を求める．正規化した参照の
// 上位語を先に，下位語を後に引く（比較自体はFP64であり，無限精度のノルムではない）．
// maxrel は |ref| が 1e-3 * max|ref| 未満の要素を無視する．そこは桁落ちのせいで
// 要素ごとの相対誤差が意味を持たないためである．
inline Acc accuracy(const std::vector<double>& C,
                    const std::vector<double>& Chi, const std::vector<double>& Clo) {
    // 長さの食い違い・空の結果は比べられない（ループが 1 度も回らないと「誤差 0＝99 bit」で合格する）ので，
    // 精度を求めずに失敗側（rel=inf，ebits=-1）へ倒す
    if (C.empty() || C.size() != Chi.size() || C.size() != Clo.size()) {
        std::fprintf(stderr, "ERROR: accuracy の配列の長さが違うか空である（C %zu・Chi %zu・Clo %zu 要素）\n",
                     C.size(), Chi.size(), Clo.size());
        return {std::numeric_limits<double>::infinity(), -1.0, std::numeric_limits<double>::infinity(), 0};
    }
    double num = 0, den = 0, maxabs = 0, maxrel = 0, mx = 0;
    for (size_t z = 0; z < C.size(); z++) {
        const auto [hi, lo] = accuracy_reference(Chi[z], Clo[z]);
        maxabs = std::max(maxabs, std::fabs(hi));
        mx = std::max(mx, std::max(std::fabs(hi), std::fabs((C[z] - hi) - lo)));
    }
    const double tau = 1e-3 * maxabs;
    // 二乗の前に 2 のべき sc を掛けて最大値を 1 前後へ寄せる．値が小さいと dd*dd と g*g が
    // ともに 0 へ落ち，誤答が「相対誤差 0」で合格してしまう（Chi=2^-600 で C=0 など）．
    // 2 のべきの倍率は正規数の範囲では丸めを変えないので，通常の入力では結果は同じ．
    // 倍率は 2 つの有限な 2 のべき sc1*sc2 に分けて掛ける：最大値が非正規化数（2^-1074 など）だと
    // 2^-ilogb(mx) = 2^1074 が FP64 を溢れて inf になり，0*inf が NaN を作って完全一致の結果まで
    // 不合格にしてしまう．sc1 は 2^1023 で頭打ちにし，残りを sc2 に回す．通常の入力
    // （-ilogb(mx) <= 1023）では sc2 = 1 で，倍率 1 つを掛けるのと同じ値になる．上げる向きの 2 のべき倍は無誤差．
    int sce = (mx > 0 && std::isfinite(mx)) ? -std::ilogb(mx) : 0;
    const int sce1 = std::min(sce, 1023);
    const double sc1 = std::ldexp(1.0, sce1), sc2 = std::ldexp(1.0, sce - sce1);
    for (size_t z = 0; z < C.size(); z++) {
        const auto [hi, lo] = accuracy_reference(Chi[z], Clo[z]);
        const double g = hi;
        const double dd = (C[z] - hi) - lo;
        const double ds = (dd * sc1) * sc2, gs = (g * sc1) * sc2;
        num += ds * ds;
        den += gs * gs;
        if (std::fabs(g) > tau) maxrel = std::max(maxrel, std::fabs(dd) / std::fabs(g));
    }
    // 参照ノルムが 0 のときは相対誤差が定義できないので，完全一致（誤差 0）だけを合格とする（誤差が少しでも残れば
    // 相対誤差を inf にして失敗側へ倒す．絶対誤差の大きさは見ない）．
    // また num・den に非有限が混じると比較がすべて偽になり，そのままでは
    // 「誤差 0＝99 bit」と報告してしまうので，明示的に失敗側へ落とす．
    double rel;
    if (!std::isfinite(num) || !std::isfinite(den)) rel = std::numeric_limits<double>::infinity();
    else if (den > 0)                               rel = std::sqrt(num) / std::sqrt(den);
    else                                            rel = (num > 0) ? std::numeric_limits<double>::infinity() : 0.0;
    const double ebits = std::isfinite(rel) ? (rel > 0 ? -std::log2(rel) : 99.0) : -1.0;
    uint64_t h = 1469598103934665603ull;          // kHashInit（標準の FNV-1a の初期値ではない．上の fnv64 の注記）
    for (size_t z = 0; z < C.size(); z++) {
        uint64_t u;
        std::memcpy(&u, &C[z], 8);
        for (int b = 0; b < 64; b += 8) { h ^= (u >> b) & 0xFFu; h *= 1099511628211ull; }
    }
    return { rel, ebits, maxrel, h };
}

inline void print_accuracy(const Acc& a) {
    std::printf("Accuracy : ||C-ref||_F/||ref||=%.3e  (effective ~%.1f bit)  maxrel=%.3e  [ref=Dot2]\n",
                a.rel_fro, a.ebits, a.maxrel);
    // 誤差を丸めずに出す行と，結果のビット列のハッシュ（実装の変更の前後の比較用）
    std::printf("CHECKSUM,rel_fro=%.17g,maxrel=%.17g,chash=%016llx\n",
                a.rel_fro, a.maxrel, (unsigned long long)a.chash);
}

}  // namespace oz_eval
