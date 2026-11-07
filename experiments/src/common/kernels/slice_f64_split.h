// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// FP64 のタイルを triple-float の 4 枚へ割る処理を，2 つのデータ移動 RISC-V で半分ずつ行うための部品
// ---------------------------------------------------------------------------
// 1 つの RISC-V で割ると，スキーム I のスライス段の時間は FP64 のリーダ（slice_f64_reader.cpp）が 1 要素ずつ
// 整数演算で triple-float へ割る処理（split_f64．約 45 命令・72 サイクル／要素）で決まる
// （docs/nongemm_opt_results.md の §5）．同じコアのライタ（slice_writer.cpp）の RISC-V は
// スライスを書き出すだけで手が空いているので，1 タイルの後半 512 要素をライタに割らせる．
//
// 受け渡し（CB c_7 を 1 ページの郵便受けに使う）
//   リーダ：タイルを読み終え，出力 4 枚の CB を確保したら，郵便受けのページに
//           「FP64 タイルと指数差分タイルの L1 アドレス，出力 4 枚の書き込み位置」を書いて
//           push する．前半を割り，郵便受けが空く（ライタが pop する）のを待ってから出力 4 枚を push する．
//   ライタ：スライスの書き出しの合間に郵便受けを覗き（待たない），届いていれば後半を割って pop する．
// 郵便受けに CB を使うのは，受け渡しの数え方（tiles_received／acked）がプログラムの起動ごとに
// 0 から始まるからである．L1 の変数で旗を立てると，前の実行の値が残っていて取り違える．
// ライタは郵便受けを待って止まってはならない：スライサがライタの pop を待ち，リーダがライタの
// 後半を待つので，ライタが郵便受けか c_16 の一方で止まると 3 者が互いを待って止まる．
// そのためライタは cb_pages_available_at_front で覗くだけにする（slice_writer.cpp）．
//
// 順序の保証（両方向）
//   Blackhole のデータ移動 RISC-V は L1 の読みにデータキャッシュ（L0）を持つ．もう一方の RISC-V や
//   NoC が書いた L1 を読む前には，相手の知らせ（CB のカウンタ）を見た後に invalidate_l1_cache()
//   （fence）で捨ててから読む（ライタは仕事を読む前．リーダが NoC の書いた FP64・指数差分を読む前は
//   noc_async_read_barrier が捨てる）．
//   自分が L1 へ書いた値を相手に渡すときは，fence だけでは足りない：fence は store queue を空に
//   するだけで，書き込みが L1 に届いたことは保証せず，別のメモリ領域にある CB のカウンタの更新が
//   先に処理されうる．そこで知らせる直前に最後に書いた番地を読み戻す（l1_order.h の
//   l1_writes_landed）．リーダ→ライタは仕事の 6 語（最後に書く os を読み戻してから push），
//   ライタ→リーダは後半 512 要素（s の 1023 番を読み戻してから pop），リーダ→スライサは前半
//   （s の 511 番を読み戻してから出力 4 枚を push）．読み戻しの前の fence は先行ロードの完了も待つので，
//   ライタが pop した時点で FP64・指数差分・仕事の読みは終わっており，リーダは次のタイルでそれらを
//   上書きしてよい．検査は tools ではなく実機の slice_split_check（scripts/tests/check_slice_split.sh）．

#include "api/dataflow/dataflow_api.h"   // tt_l1_ptr・get_write_ptr・cb_push_back・cb_pop_front
#include "f64_tf.h"
#include "l1_order.h"
#include "slice_f64_args.h"   // 郵便受けの CB 番号とページの大きさ（ホストと共有）

// 検査用の探針（slice_split_check.cpp だけが define で立てる．答えが壊れる．本番は 0）．探針の振る舞いは下の
// 郵便受けの手順（mbox_*）に閉じ込め，カーネルの本体は探針を知らない（本番と同じカーネルを検査するため，別のコピーは
// 作らない）．
// どちらも「知らせがデータより先に着く」を大きな時間差でわざと起こし，検査がそれを捕まえることを示す．
//   1  ライタが後半を割る前に郵便受けを返す（ライタ→リーダ：出力 4 枚の後半と FP64 の再利用）
//   2  リーダが郵便受けへ push してからしばらく待ち，その後で仕事の中身を書く（リーダ→ライタ．
//      ライタは前のタイルの書き込み位置を読む．番地はどれも有効な CB の中に留まる）
#ifndef SLICE_SPLIT_PROBE
#define SLICE_SPLIT_PROBE 0
#endif
constexpr uint32_t SLICE_MBOX_CB = slice_f64_args::CB_MBOX;   // 郵便受け
constexpr uint32_t SLICE_SPLIT_J = 512;        // リーダが [0, 512)，ライタが [512, 1024) を割る

// 郵便受けのページの中身（ホストが作るページの大きさ slice_f64_args::MBOX_PAGE と同じ）
struct SliceJob {
    uint32_t src, dl, oh, om, ol, os;          // L1 アドレス
    uint32_t pad[2];
};
static_assert(sizeof(SliceJob) == slice_f64_args::MBOX_PAGE, "郵便受けの中身がホストの作るページの大きさと違う");

// [j0, j1) の要素を割る．SPLIT を立てないリーダ（slice_f64_reader.cpp）もこれを [0, 1024) で呼ぶ（手続きは 1 つ）．
// volatile のままにしてある（volatile を外して形を変えると，かえって遅い．docs/nongemm_opt_results.md の §5）．
inline void slice_split_range(uint32_t src, uint32_t dlp, uint32_t ohp, uint32_t omp,
                              uint32_t olp, uint32_t osp, uint32_t j0, uint32_t j1) {
    volatile tt_l1_ptr uint32_t* s  = (volatile tt_l1_ptr uint32_t*)src;
    volatile tt_l1_ptr int32_t*  dl = (volatile tt_l1_ptr int32_t*)dlp;
    volatile tt_l1_ptr uint32_t* oh = (volatile tt_l1_ptr uint32_t*)ohp;
    volatile tt_l1_ptr uint32_t* om = (volatile tt_l1_ptr uint32_t*)omp;
    volatile tt_l1_ptr uint32_t* ol = (volatile tt_l1_ptr uint32_t*)olp;
    volatile tt_l1_ptr uint32_t* os = (volatile tt_l1_ptr uint32_t*)osp;
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t w0 = s[2 * j];        // little endian：下位語が先
        const uint32_t w1 = s[2 * j + 1];
        uint32_t hi, braw, craw, sc;
        split_f64(w1, w0, dl[j], hi, braw, craw, sc);
        oh[j] = hi; om[j] = braw; ol[j] = craw; os[j] = sc;
    }
}

// ===== 郵便受けの手順（リーダとライタ．探針はここだけが知る）=====
// 本番（SLICE_SPLIT_PROBE=0）では，知らせる・返す位置は本番の位置で，探針の部分は何もしない．
// 仕事の中身（FP64 と指数差分の番地，出力 4 枚の CB cb_h・cb_m・cb_l・cb_s の書き込み先）を郵便受けのページ job に
// 書き，L1 に届いたことを確かめる（fence だけでは届いたことにならない．l1_order.h）．出力の番地は書く直前に CB から
// 引く（この関数にする前のリーダと同じ順の読み書きで，機械語も同じ）．
inline void mbox_write_job(volatile tt_l1_ptr SliceJob* job, uint32_t src, uint32_t dl, uint32_t cb_h, uint32_t cb_m,
                           uint32_t cb_l, uint32_t cb_s) {
    job->src = src; job->dl = dl;
    job->oh = get_write_ptr(cb_h); job->om = get_write_ptr(cb_m);
    job->ol = get_write_ptr(cb_l); job->os = get_write_ptr(cb_s);
    l1_writes_landed((uint32_t)&job->os);
}
// リーダ：最初の仕事の前に郵便受けを有効な番地で埋めておくか（探針 2 だけ．中身を書く前に知らせても，ライタが
// 有効な番地を読むように）
constexpr bool MBOX_PREFILL = SLICE_SPLIT_PROBE == 2;
// リーダ：郵便受けを確保した後，仕事を書く前（探針 2 はここで知らせてから，しばらく待って書く．答えが壊れる）
inline void mbox_reader_before_write() {
    if constexpr (SLICE_SPLIT_PROBE == 2) {
        cb_push_back(SLICE_MBOX_CB, 1);
        for (uint32_t k = 0; k < 4000; ++k) asm volatile("nop");
    }
}
// リーダ：仕事が L1 に届いた後に知らせる（探針 2 は書く前に知らせ済み）
inline void mbox_reader_notify() {
    if constexpr (SLICE_SPLIT_PROBE != 2) cb_push_back(SLICE_MBOX_CB, 1);
}
// ライタ：後半を割る前（探針 1 はここで郵便受けを返す．答えが壊れる）
inline void mbox_writer_before_split() {
    if constexpr (SLICE_SPLIT_PROBE == 1) cb_pop_front(SLICE_MBOX_CB, 1);
}
// ライタ：割った後半が L1 に届いた後に返す（探針 1 は割る前に返し済み）
inline void mbox_writer_return() {
    if constexpr (SLICE_SPLIT_PROBE != 1) cb_pop_front(SLICE_MBOX_CB, 1);
}
