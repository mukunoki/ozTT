// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎スキーム II：OZ_TIME_WHOLE の全工程の記録・流し直し（論文の全体の時間）・検査（ozaki2_main.cpp が取り込む．scheme I の ozaki1_whole.hpp
// に当たる）
// ---------------------------------------------------------------------------
// 段ごとの計時（DRAM->DRAM 行はその和で，論文の内訳に使う）では，行列積は 1 本ずつプログラムを組んでウォームアップと
// iters 回の反復を行い，合成・再構成も段ごとに同じことをする．全工程を 1 回ずつ流した時間とは限らない．
// そこで OZ_TIME_WHOLE=1 のときは，段ごとの計時で投入したワークロードを捨てずに投入の順に
// 記録しておき（record・record_workload），最後に全部を同期せずに続けて流して計時する（time）．組み立ては計時の外である．
// 投入し直すワークロードが読むバッファを生かしておくため，ドライバは段の途中の解放（FP64 のオペランド・スライス出力・
// 桁積の平面）をしない（ozaki2_main.cpp の Oz2Run::may_release．keep は解放しないバッファの受け皿）．そのぶん全部のバッファが
// 同時に DRAM に載る必要がある．ドライバの plan_bands（ozaki2_plan.hpp）は帯・組の数をこの同時確保（A・B を残す）で決めるので，
// 通しの計時もふだんの実行と同じ構成で流せる（14080^3 の II-BF16・II-FP16・II-BFP2・II-TF32 は出力を行の帯に分ける．
// II-INT8・II-FP8・II-BFP8・II-BFP4 は帯 1 本）．帯・組に分けるときは全帯・全組のスライサが 1 本の器を使い回す
// （slice_band・slice_group）．
// ドライバは記録の器を実行の状態 Oz2Run の欄に持ち（OZ_TIME_WHOLE が立っていなければ空），段の関数へ引数で渡す．
// 手順には種類を付けて記録し（種類は scheme I と共通の oz_whole::Kind），種類ごとの本数を構成から数えた本数
// （oz2_whole_expected）と突き合わせる（記録の抜け・二重記録を捕まえる．WHOLE-CHECK 行．common/oz_whole.hpp）．
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <tt-metalium/distributed.hpp>
#include "common/host_eval.hpp"      // oz_eval::time_dispatch（oz_timing.hpp．傘から取り込む）
#include "common/oz_whole.hpp"       // 種類・数・番兵の手順・WHOLE-CHECK 行（scheme I と共通）
#include "ozaki2/ozaki2_moduli.h"    // Ozaki2Moduli・Oz2Route・oz2_nout（期待値の数え方）

class Oz2Whole {
public:
    using MeshBuffer = tt::tt_metal::distributed::MeshBuffer;
    using MeshWorkload = tt::tt_metal::distributed::MeshWorkload;
    using MeshCommandQueue = tt::tt_metal::distributed::MeshCommandQueue;
    using MeshDevice = tt::tt_metal::distributed::MeshDevice;

    // 投入するだけ（同期しない）の手順 f を種類 k で記録する（合成段は 1 つのワークロードを実行時引数だけ差し替えて
    // 使い回すので，記録するのは「詰めて投入する」ことそのもの．引数の設定はホストの仕事で，全工程の計時の中にも入る）
    void record(oz_whole::Kind k, std::function<void()> f) {
        steps_.push_back(std::move(f));
        kind_.push_back((uint8_t)k);
    }
    // 投入したワークロード wl を取っておき，同じキューへ投入し直す手順として記録する（GEMM 段・再構成段）．
    // ワークロードの寿命は手順（ラムダ）が値で持つ shared_ptr が保つ（clear で手順と一緒に手放す）
    void record_workload(oz_whole::Kind k, MeshCommandQueue& cq, MeshWorkload&& wl) {
        auto p = std::make_shared<MeshWorkload>(std::move(wl));
        MeshCommandQueue* q = &cq;
        record(k, [q, p] { tt::tt_metal::distributed::EnqueueMeshWorkload(*q, *p, false); });
    }
    // 段の途中で手放すはずのバッファを，流し直しが読む（書く）ので生かしておく
    void keep(std::shared_ptr<MeshBuffer> b) { keep_buf_.push_back(std::move(b)); }

    // 流し直しが「何もしない」「段を抜かす」「段の間で前の段の書き込みより先に読む」ことを捕まえる検査（計時の外．手順は
    // common/oz_whole.hpp の sentinel_run）．流し直す前に，流し直しが書くはずの器 bufs と keep で生かした器を全部番兵で埋め，
    // 出口に番兵が載ったことを read_out（出口の全体を読む）で確かめてから，記録した全手順を同期せずに 1 回だけ流して
    // 出口を段ごとの計時の後の値 pre とビット単位で比べる．あわせて種類ごとの本数を期待値 exp と比べ，1 回流したときに
    // 実際に呼んだ本数とも比べる
    template <class ReadOut>
    oz_whole::Check check(std::shared_ptr<MeshDevice>& md, std::vector<std::shared_ptr<MeshBuffer>> bufs, ReadOut&& read_out,
                          const std::vector<uint32_t>& pre, const oz_whole::Counts& exp) {
        auto& cq = md->mesh_command_queue();
        oz_whole::Check w;
        w.pre = pre;
        for (auto& b : keep_buf_) if (b) bufs.push_back(b);   // 桁積の平面（合成段が無い構成では空），limb 3 面，128 bit 固定小数の器
        for (auto& b : bufs) TT_FATAL(b != nullptr, "Oz2Whole::check: 番兵で埋める器が手放されている");
        oz_whole::sentinel_run(md, bufs, read_out,
            [&] { for (size_t i = 0; i < steps_.size(); i++) { steps_[i](); w.note((oz_whole::Kind)kind_[i]); } },
            [&] { tt::tt_metal::distributed::Finish(cq); }, w);
        for (uint8_t k : kind_) w.rec[k]++;
        w.exp = exp;
        return w;
    }
    // 段「whole」として温めてから iters 回の平均を取る（計時の回数は段ごとの計時と同じ．温めは oz_timing.hpp の warm_up）．
    // 組み立てはすべて済んでいる
    double time(uint32_t warmup, uint32_t iters, MeshCommandQueue& cq) {
        return oz_eval::time_dispatch("whole", warmup, iters,
            [&] { for (auto& f : steps_) f(); },
            [&] { tt::tt_metal::distributed::Finish(cq); });
    }
    // 取っておいたワークロードとバッファを手放す（デバイスを閉じる前に呼ぶ．閉じた後の静的な破棄で MeshBuffer・
    // MeshWorkload を壊すと落ちる）
    void clear() { *this = Oz2Whole{}; }

private:
    std::vector<std::function<void()>> steps_;
    std::vector<uint8_t> kind_;   // steps_ と同じ並び（oz_whole::Kind）
    std::vector<std::shared_ptr<MeshBuffer>> keep_buf_;
};

// 記録した手順の種類ごとの本数の期待値（記録の経路とは別に構成から数える）
//   HEAD     1（最大値 → 指数差分 → スライスをまとめた 1 手順．ドライバの scale_and_slice）．法を組に分ける（ngroups > 1）
//            なら最大値 → 指数差分の 1 手順と，組ごとのスライスの 1 手順ずつ（slice_group）で 1 + ngroups．出力を行の帯に
//            分ける（nband > 1）なら最大値 → 指数差分 → B のスライスの 1 手順と，帯ごとの A のスライスの 1 手順ずつ
//            （slice_band）で 1 + nband
//   GEMM     GEMM の出力の本数 nout × 出力パネルの枚数 panels（ドライバの run_partial_gemm は出力 1 本・パネル 1 枚で 1 回．
//            帯に分けても全帯のパネルの和は panels）
//   COMPOSE  桁分割経路だけ．1 桁の法を除く法ごとに ceil(np/CH)（塊ごと）+ 塊に切ったら畳み込みの 1 回
//            （np は法の出力の本数 oz2_nout，CH は合成段の chunk．合成する法が無い構成と INT8 経路は CH = 0 で 0）
//   RECON    再構成のパス数 recon_passes（ozaki2_plan.hpp の make_recon_plan．1 パスか 2 パス）．COMPOSE・RECON は帯ごとに
//            回すので nband 倍
//   ACCUM    この経路には無い（0．scheme I の別プログラムの累積）
inline oz_whole::Counts oz2_whole_expected(const Ozaki2Moduli& mo, const Oz2Route& rt, uint32_t panels, uint32_t chunk,
                                           uint32_t recon_passes, uint32_t ngroups, uint32_t nband) {
    oz_whole::Counts exp;
    exp[oz_whole::HEAD] = ngroups > 1 ? 1u + ngroups : (nband > 1 ? 1u + nband : 1u);
    exp[oz_whole::GEMM] = (uint64_t)mo.nout * panels;
    for (int s = 0; s < mo.s; s++) {
        if (mo.mode[s] == OZ2_SPLIT_ONE || chunk == 0) continue;
        const uint32_t np = oz2_nout(mo.m[s], rt.digit_bound, rt.nd(), mo.maxg_m[s]);
        exp[oz_whole::COMPOSE] += (np + chunk - 1) / chunk + (np > chunk ? 1u : 0u);
    }
    exp[oz_whole::COMPOSE] *= nband;
    exp[oz_whole::RECON] = (uint64_t)recon_passes * nband;
    return exp;
}
