// SPDX-License-Identifier: Apache-2.0
#pragma once
// 機械が読む 1 行（"CONFIG,<種類>,k=v,k=v,..." の形）の組み立て器
// ---------------------------------------------------------------------------
// 欄を名前と値の組で 1 つずつ足す．書式指定子と引数を位置で対応させる printf 1 回で書くと，欄を足すときに位置が
// 1 つずれただけで別の欄の値が黙って入り，検査側（scripts/lib/oz_logs.py の fields）は key=value で読むので「名前は正しく
// 値が誤った欄」として通ってしまう．欄は足した順に並ぶ（検査側は位置に依らないが，人がログを読み比べるので並びは
// 保つ．新しい欄は末尾に足す）．
// 同じ名前の欄を 2 度足すこと，値に ','・'='・改行を含めること（検査側の読み方を壊す）は組み立ての誤りなので，
// 理由を出して abort する（測定のログに壊れた行を残さない）．tt-metal に依存しない．
// 検査側は ',' で区切って最初の '=' で割るので，値の '=' は読み方を壊さない．値に '=' を書く行（探針の PROBE 行の
// fp8_probe の mode=R=8．testkit/probe_line.hpp の ProbeLine）は，構築のときに OzLine::EqInValue を渡して許す
// （尾崎の CONFIG 行は許さない．欄の値に '=' が混じるのは組み立ての誤りとして止める）．
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

class OzLine {
public:
    // 値の '=' を許す印（上の注記）
    struct EqInValue {};
    explicit OzLine(std::string head) : s_(std::move(head)) {}
    OzLine(std::string head, EqInValue) : s_(std::move(head)), eq_ok_(true) {}

    OzLine& kv(const char* k, const std::string& v) {
        check(k, v);
        s_ += ',';
        s_ += k;
        s_ += '=';
        s_ += v;
        return *this;
    }
    OzLine& kv(const char* k, const char* v) { return kv(k, std::string(v ? v : "")); }
    template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    OzLine& kv(const char* k, T v) { return kv(k, std::to_string(v)); }
    // 真偽は 1・0 で書く（kv に bool を渡すと整数の版と紛れるので名前を分ける）
    OzLine& flag(const char* k, bool v) { return kv(k, std::string(v ? "1" : "0")); }
    // 実数は printf の書式 fmt で書く（"%.4f" など）
    OzLine& num(const char* k, const char* fmt, double v) {
        char b[64];
        std::snprintf(b, sizeof b, fmt, v);
        return kv(k, std::string(b));
    }
    // 2 つの整数を "<a>x<b>" で書く（格子・形状・パネルの枚数など．grid=10x11）
    OzLine& dims(const char* k, uint64_t a, uint64_t b) { return kv(k, std::to_string(a) + "x" + std::to_string(b)); }
    // すでに "k=v,k=v" の形に組んだ欄の並び（oz_eval::input_desc・timing_desc）を足す．1 つずつ同じ検査をする
    OzLine& raw(const std::string& fields) {
        size_t p = 0;
        while (p <= fields.size()) {
            const size_t q = std::min(fields.find(',', p), fields.size());
            const std::string f = fields.substr(p, q - p);
            const size_t e = f.find('=');
            if (e == std::string::npos || e == 0) die("欄の並び \"" + fields + "\" に k=v でない部分 \"" + f + "\" がある");
            kv(f.substr(0, e).c_str(), f.substr(e + 1));
            p = q + 1;
        }
        return *this;
    }
    const std::string& str() const { return s_; }
    // 標準出力へ 1 行（改行つき）で書く
    void print() const {
        std::fputs(s_.c_str(), stdout);
        std::fputc('\n', stdout);
    }

private:
    std::string s_;
    std::vector<std::string> keys_;
    bool eq_ok_ = false;

    [[noreturn]] void die(const std::string& why) const {
        std::fprintf(stderr, "ERROR: ログの行の組み立ての誤り（%s）：%s\n", why.c_str(), s_.c_str());
        std::abort();
    }
    void check(const std::string& k, const std::string& v) {
        if (k.empty() || k.find_first_of(",= \n") != std::string::npos) die("欄の名前 \"" + k + "\"");
        if (v.find_first_of(eq_ok_ ? ",\n" : ",=\n") != std::string::npos)
            die("欄 " + k + " の値 \"" + v + "\" に ','" + (eq_ok_ ? "" : "・'='") + "・改行がある");
        for (const auto& x : keys_)
            if (x == k) die("欄 " + k + " を 2 度足した");
        keys_.push_back(k);
    }
};
