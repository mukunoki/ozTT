// SPDX-License-Identifier: Apache-2.0
#pragma once
// 環境変数（つまみ）と位置引数の読み方（1 系統）
// ---------------------------------------------------------------------------
// 尾崎の 2 つのドライバ・基準実装・検査・探針・ホスト単体の道具（tools/）は，つまみと位置引数をここの関数で読む．
// tt-metal に依存しない（tools/ と CPU の DGEMM の参照 tools/dgemmcheck もそのまま取り込む）．どのつまみがどの読み方かは
// oz_knobs.hpp の表にある．読み方は次の 5 つ：
//   有無   set：立っていれば（空の値も）真．値は見ない
//   整数   u32・i64：10 進の整数として文字列全体を読み切り，範囲 [lo, hi] に入ること．空・末尾の余計な文字・桁あふれは
//          止める．u32 は負号を文字列のどこにあっても拒む（strtoull は先頭の空白の後の負号を受けて符号反転した値を返すので，
//          "-1" が 2^64-1 に，"\n-18446744073709551552" が 64 になる）．0/1 のつまみは範囲 [0, 1] の u32 で読む
//   実数   f64：有限な数として読み切り，[lo, hi] に入ること
//   列挙   choice：与えた名前のどれか（大文字・小文字も区別する）
//   文字列 get：値をそのまま（書式の検査は使う側）
// 読めない値は理由を 1 行（"ERROR: <名前>=<値> は …"）で出して終了コード 2 で止める．黙って既定や 0 へ倒すと，打ち間違いの
// まま別の構成を測ってしまうからである．デバイスを開いた後に初めて読むと閉じずに終わるので，ドライバは入口で 1 度読んで
// 検査しておく（oz_knobs.hpp の preflight．以降の読み取りは同じ値を返す）．
// 位置引数（arg_u32・arg_i64）も同じ規則で読む．範囲の外は "ERROR: 引数 <番号>（<名前>）"<値>" は …" で止める．
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>

namespace oz_env {

// 理由を出して終了コード 2 で止める（入口の検査の失敗．引数・つまみの誤りはどれもこの終わり方にそろえる）
[[noreturn]] inline void fail(const std::string& why) {
    std::fprintf(stderr, "ERROR: %s\n", why.c_str());
    std::exit(2);
}

// 値（未設定なら nullptr）と有無
inline const char* get(const char* name) { return std::getenv(name); }
inline bool set(const char* name) { return std::getenv(name) != nullptr; }

// 文字列 v を 10 進の符号なし整数として読み切れれば true（負号はどこにあっても拒む）
inline bool to_u64(const char* v, unsigned long long& x) {
    if (!v || std::strchr(v, '-')) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtoull(v, &end, 10);
    return end != v && *end == '\0' && errno != ERANGE;
}
// 符号つき整数
inline bool to_i64(const char* v, long long& x) {
    if (!v) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtoll(v, &end, 10);
    return end != v && *end == '\0' && errno != ERANGE;
}
// 有限な実数
inline bool to_f64(const char* v, double& x) {
    if (!v) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtod(v, &end);
    return end != v && *end == '\0' && errno != ERANGE && std::isfinite(x);
}

// 範囲の文（"<lo> 以上 <hi> 以下の整数"）
inline std::string int_range(long long lo, long long hi) {
    return std::to_string(lo) + " 以上 " + std::to_string(hi) + " 以下の整数";
}

// 環境変数 name の値 v（呼び出し側が読んだもの）を読む．読めなければ止める
inline uint32_t parse_u32(const char* name, const char* v, uint32_t lo, uint32_t hi) {
    unsigned long long x = 0;
    if (!to_u64(v, x) || x < lo || x > hi)
        fail(std::string(name) + "=" + (v ? v : "") + " は " + int_range(lo, hi) + "ではない");
    return (uint32_t)x;
}
inline long long parse_i64(const char* name, const char* v, long long lo, long long hi) {
    long long x = 0;
    if (!to_i64(v, x) || x < lo || x > hi)
        fail(std::string(name) + "=" + (v ? v : "") + " は " + int_range(lo, hi) + "ではない");
    return x;
}
inline double parse_f64(const char* name, const char* v, double lo, double hi) {
    double x = 0.0;
    if (!to_f64(v, x) || x < lo || x > hi) {
        char b[96];
        std::snprintf(b, sizeof b, " は %g 以上 %g 以下の数ではない", lo, hi);
        fail(std::string(name) + "=" + (v ? v : "") + b);
    }
    return x;
}

// 環境変数を読む（未設定なら dflt）
inline uint32_t u32(const char* name, uint32_t dflt, uint32_t lo, uint32_t hi) {
    const char* v = std::getenv(name);
    return v ? parse_u32(name, v, lo, hi) : dflt;
}
inline long long i64(const char* name, long long dflt, long long lo, long long hi) {
    const char* v = std::getenv(name);
    return v ? parse_i64(name, v, lo, hi) : dflt;
}
inline double f64(const char* name, double dflt, double lo, double hi) {
    const char* v = std::getenv(name);
    return v ? parse_f64(name, v, lo, hi) : dflt;
}
// 列挙の照合：値 v（呼び出し側が読んだもの）が names（名前を '|' で区切った並び）の何番目か．どれでもなければ止める．
// 列挙の照合と誤りの文はここ 1 つ（下の choice と，つまみの表の choices で読む oz_knobs::choice が使う）
inline int choice_in(const char* name, const char* v, std::string_view names) {
    int i = 0;
    std::string all;
    for (size_t p = 0;; ++i) {
        const size_t e = names.find('|', p);
        const std::string_view n = names.substr(p, e == std::string_view::npos ? std::string_view::npos : e - p);
        if (v && n == v) return i;
        all += (i ? "・" : "") + std::string(n);
        if (e == std::string_view::npos) break;
        p = e + 1;
    }
    fail(std::string(name) + "=" + (v ? v : "") + " は " + all + " のどれでもない");
}
// 列挙：値が names の何番目か（未設定なら dflt）．どれでもなければ止める（名前に '|' は使わない）
inline int choice(const char* name, std::initializer_list<const char*> names, int dflt) {
    const char* v = std::getenv(name);
    if (!v) return dflt;
    std::string all;
    for (const char* n : names) {
        if (!all.empty()) all += '|';
        all += n;
    }
    return choice_in(name, v, all);
}

// 位置引数 idx（argv[idx]）を [lo, hi] の 10 進の整数として読む（無ければ dflt）．what は誤りの文に出す名前（省略可）
inline uint32_t arg_u32(int argc, char** argv, int idx, uint32_t dflt, uint32_t lo = 0, uint32_t hi = UINT32_MAX,
                        const char* what = nullptr) {
    if (idx >= argc) return dflt;
    unsigned long long x = 0;
    if (!to_u64(argv[idx], x) || x < lo || x > hi)
        fail("引数 " + std::to_string(idx) + (what ? std::string("（") + what + "）" : std::string()) + "\"" + argv[idx] +
             "\" は " + int_range(lo, hi) + "ではない");
    return (uint32_t)x;
}

// 位置引数 idx を [lo, hi] の 10 進の符号つき整数として読む（無ければ dflt．誤りの文は arg_u32 と同じ形）
inline long long arg_i64(int argc, char** argv, int idx, long long dflt, long long lo, long long hi,
                         const char* what = nullptr) {
    if (idx >= argc) return dflt;
    long long x = 0;
    if (!to_i64(argv[idx], x) || x < lo || x > hi)
        fail("引数 " + std::to_string(idx) + (what ? std::string("（") + what + "）" : std::string()) + "\"" + argv[idx] +
             "\" は " + int_range(lo, hi) + "ではない");
    return x;
}

// 余分な位置引数は止める（引数の追加・位置ずれを黙って読み捨てない）．names は使い方の並び（"M N K p L warmup iters"）．
// 止めるべきなら理由を出して false
inline bool check_argc(int argc, int nmax, const char* names) {
    if (argc - 1 <= nmax) return true;
    std::fprintf(stderr, "ERROR: 引数が多すぎる（%s の %d 個まで，%d 個）\n", names, nmax, argc - 1);
    return false;
}

}  // namespace oz_env
