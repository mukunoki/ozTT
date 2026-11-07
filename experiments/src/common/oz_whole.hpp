// SPDX-License-Identifier: Apache-2.0
#pragma once
// OZ_TIME_WHOLE（論文の全体の時間）の検査（WHOLE-CHECK 行）：尾崎の 2 つのドライバが共有する部分
// ---------------------------------------------------------------------------
// OZ_TIME_WHOLE=1 は，段ごとの計時（DRAM->DRAM 行はその和で，論文の内訳に使う）の後に，全工程を段の間で同期せずに続けて流す
// 計時を足す（docs/env_vars.md の A．論文の全体の時間はこちら）．その流し直しが「何もしない」「段を抜かす」「前の段の書き込みより先に読む」ことを捕まえるため，
// 計時の前に次を行い，計時の後にも C を比べる：
//   1. 流し直しが書くはずの中間・出力の器を番兵（OZ_WHOLE_SENTINEL）で埋め，出力に番兵が載ったことを読み戻して確かめる
//      （段ごとの計時の後の器にはどれも正しい最終値が残っているので，番兵が無いと 1 本も流さなくても C が一致してしまう）
//   2. 全工程を 1 回だけ流し，C を段ごとの計時の後の C とビット単位で比べる
//   3. 流した手順の本数を，種類ごとに構成から数えた本数と比べる（記録の抜け・二重記録を捕まえる）
// 投入の数え方と番兵で埋める器は経路の構造に依るので各スキームに置く（尾崎 I は投入のたびに数え，II は投入を記録して
// 流し直す）．ここは種類・数の持ち方・番兵の手順・行の書式を持つ．
// WHOLE-CHECK 行：identical（すべて通ったときだけ 1）・filled・sentinel・post_timing・counts の後に，流した手順の数の和
// steps と期待値の和 expected，種類ごとの「流した数/期待値」（head・gemm・accum・compose・recon．その経路に無い種類は 0/0）．
// 尾崎 I は 2026-10-22 より前は steps を programs と書き，compose を持たなかった．II は accum を持たなかった．
// 照合（verify/check_paper_numbers.py の load_whole）は identical だけを読み，tests/check_probes.sh は両方の名前を読む．
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include "common/ozaki_gemm_host.hpp"   // fill_mesh_buffer_u32

// 全工程を流し直す前に中間・出力の器へ書く番兵（上の 1）．
// FP32・BF16（2 つとも）では NaN，FP64（2 語とも番兵で 0x7FC07FC07FC07FC0）では指数 0x7FC の有限の値 約 2.3e307
// （NaN ではない），符号-絶対値 int32 では 2^31 近い絶対値，BFP の面では指数 0x7F・0xC0 の塊になる．検査は NaN に
// 頼らない：o64 に番兵が載ったことを読み戻して確かめてから流し直し，C を段ごとの計時の後の C とビット単位で比べるので，
// 流し直しがその器を書かなければ出口の C が変わる（番兵のままか，読んだ番兵から作った値になる）．
constexpr uint32_t OZ_WHOLE_SENTINEL = 0x7FC07FC0u;

namespace oz_whole {

// 手順の種類．HEAD はスケール算出・指数差分・スライス（尾崎 I はその投入 6 本，II はまとめた 1 手順），GEMM は行列積，
// ACCUM は尾崎 I の別プログラムの累積，COMPOSE は尾崎 II の桁分割の合成段，RECON は再構成
enum Kind : uint8_t { HEAD = 0, GEMM, ACCUM, COMPOSE, RECON, NKIND };
inline const char* kind_name(int k) {
    static const char* const n[NKIND] = {"head", "gemm", "accum", "compose", "recon"};
    return n[k];
}

struct Counts {
    uint64_t n[NKIND] = {};
    uint64_t& operator[](int k) { return n[k]; }
    uint64_t operator[](int k) const { return n[k]; }
    uint64_t sum() const { uint64_t s = 0; for (uint64_t v : n) s += v; return s; }
    bool operator==(const Counts& o) const {
        for (int k = 0; k < NKIND; ++k) if (n[k] != o.n[k]) return false;
        return true;
    }
};

struct Check {
    std::vector<uint32_t> pre;     // 段ごとの計時の後の C（比べる相手）
    bool filled = false;           // 番兵が出力に載ったか（載らなければ検査が成り立たない）
    bool sentinel = false;         // 番兵で埋めてから 1 回流した C が pre と同じか
    // rec：記録した手順の数（尾崎 II．I は流しながら数えるので ran を写す），ran：その 1 回で流した数，exp：構成から
    // 数えた数
    Counts rec, ran, exp;
    Counts* counting = nullptr;    // 1 回流すあいだだけ &ran（計時のあいだは nullptr で数えない）
    void note(Kind k, uint64_t n = 1) { if (counting) (*counting)[k] += n; }
};

// 器 bufs を番兵で埋め，出力に番兵が載ったことを read_out で読み戻して確かめてから，run_once を 1 回流し（その中で
// note で数える），sync で待って出力を pre と比べる（計時の外）．read_out(std::vector<uint32_t>&) は出力の全体を読む
template <class ReadOut, class RunOnce, class Sync>
inline void sentinel_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                         const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>>& bufs,
                         ReadOut&& read_out, RunOnce&& run_once, Sync&& sync, Check& w) {
    for (const auto& b : bufs) fill_mesh_buffer_u32(md, b, OZ_WHOLE_SENTINEL);
    std::vector<uint32_t> got;
    read_out(got);
    w.filled = !got.empty();
    for (uint32_t v : got) if (v != OZ_WHOLE_SENTINEL) { w.filled = false; break; }
    w.ran = Counts{};
    w.counting = &w.ran;
    run_once();
    sync();
    w.counting = nullptr;
    read_out(got);
    w.sentinel = w.filled && got == w.pre;
}

// 計時の後の C を比べた結果 post も合わせて WHOLE-CHECK 行と理由を出し，すべて通ったか（identical）を返す
inline bool report(const Check& w, bool post) {
    const bool counts = w.rec == w.exp && w.ran == w.exp;
    const bool same = w.sentinel && counts && post;
    std::printf("WHOLE-CHECK,identical=%d,filled=%d,sentinel=%d,post_timing=%d,counts=%d,steps=%llu,expected=%llu",
                same ? 1 : 0, w.filled ? 1 : 0, w.sentinel ? 1 : 0, post ? 1 : 0, counts ? 1 : 0,
                (unsigned long long)w.rec.sum(), (unsigned long long)w.exp.sum());
    for (int k = 0; k < NKIND; ++k)
        std::printf(",%s=%llu/%llu", kind_name(k), (unsigned long long)w.rec[k], (unsigned long long)w.exp[k]);
    std::printf("\n");
    if (!w.filled)
        std::fprintf(stderr, "ERROR: OZ_TIME_WHOLE の検査で出力の器に番兵が載らなかった（検査が成り立たない）\n");
    else if (!w.sentinel)
        std::fprintf(stderr, "ERROR: OZ_TIME_WHOLE で器を番兵で埋めてから 1 回流した C が段ごとの計時の C と違う"
                             "（流し直しが段を抜かしたか，前の段の書き込みより先に読んだ）\n");
    if (!counts)
        std::fprintf(stderr, "ERROR: OZ_TIME_WHOLE で流した手順の本数（記録・実行）が構成から数えた本数と違う\n");
    if (!post) std::fprintf(stderr, "ERROR: OZ_TIME_WHOLE で続けて流した C が段ごとの計時の C と違う\n");
    return same;
}

}  // namespace oz_whole
