// SPDX-License-Identifier: Apache-2.0
#pragma once
// 探針・検査と，ホストだけの道具（tools/）の機械が読む結果の行 "PROBE,<探針の名前>,<欄>=<値>,..."
// ---------------------------------------------------------------------------
// 照合は欄の名前で読み，人向けの表の文面や欄の並びに頼らない（欄を足しても古い読み手は壊れない）．照合（oz_logs.fields）は
// ',' で区切って最初の '=' で割るので，欄の名前には ',' と '=' を，値には ',' を書けない（値の '=' は書ける．fp8_probe の
// mode=R=8）．組み立ての規則（同じ名前の欄を 2 度足さない・書けない文字・誤りは abort）は尾崎の CONFIG 行と同じ
// common/oz_logline.hpp の OzLine で，値の '=' を許すこと（OzLine::EqInValue）と値を空にしないことだけを足す．
// 値の書式は printf の書式で与える（既存の欄の桁数を変えないため）．tt-metal に依存しない（tools/ からも取り込む．
// デバイスを使う探針・検査は testkit/probe_common.hpp を通して取り込む）．
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "common/oz_logline.hpp"

namespace probe {

class ProbeLine {
public:
    explicit ProbeLine(const char* probe) : l_(head(probe), OzLine::EqInValue{}) {}
    [[gnu::format(printf, 3, 4)]] ProbeLine& add(const char* key, const char* fmt, ...) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        const int n = std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        if (n < 0 || (size_t)n >= sizeof buf) die(key, "値が長すぎる");
        if (!*buf) die(key, "値が空");
        l_.kv(key, std::string(buf));
        return *this;
    }
    const std::string& str() const { return l_.str(); }
    // 1 行で出してすぐに流す（JIT のビルドの記録などほかの出力と行の途中で混ざらないように）
    void print() const {
        std::fflush(stdout);
        l_.print();
        std::fflush(stdout);
    }

private:
    [[noreturn]] static void die(const char* key, const char* why) {
        std::fprintf(stderr, "ERROR: PROBE 行の欄 %s：%s\n", key, why);
        std::abort();
    }
    static std::string head(const char* probe) {
        if (!probe || !*probe || std::strpbrk(probe, ",=\n ")) die("（探針の名前）", "空か，書けない文字（',' '=' 空白 改行）を含む");
        return std::string("PROBE,") + probe;
    }
    OzLine l_;
};

}  // namespace probe
