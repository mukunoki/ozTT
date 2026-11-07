// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// データ移動 RISC-V が L1 へ書いた値を，別の相手に知らせる前に「書き込みが L1 に届いた」ところまで待つ
// ---------------------------------------------------------------------------
// Blackhole の baby RISC-V では，store は store queue に積まれ，書き込み要求は後から L1 やほかの
// メモリ領域へ流れる．fence が待つのは store queue が空になる（要求が発行される）ところまでで，
// 要求が処理された（L1 に書かれた）ことは保証しない．CB のカウンタ（tiles_received／acked は
// ストリームのレジスタ）や NoC のコマンドレジスタは L1 とは別のメモリ領域なので，
//     L1 へ書く → fence → cb_push_back（または noc_async_write の発行）
// では，カウンタの更新（NoC の読み出し）が L1 への書き込みより先に処理されうる
// （tt-isa-documentation BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md の「Fences」
// 「Enforcing stronger ordering」「Cross-core signalling using mailboxes」）．
// tt-metal の cb_push_back・noc_async_write もこの保証をしない（dataflow_api.h の cb_push_back は
// カウンタのレジスタを足すだけで，RISC-V が L1 へ書いた値との順序には触れない）．
//
// 文書が挙げる確実な方法は「書いた L1 を同じ RISC-V で読み戻し，その読みの結果を得てから次の要求を
// 出す」である．l1_writes_landed(a) はそれを次の 3 命令で行う：
//   fence   store queue を空にする（それまでの L1 への書き込み要求がすべて発行される）．L0 データ
//           キャッシュも捨てるので，次の lw は必ず L1 まで行く
//   lw      a（L1 の番地．呼び出し側は最後に書いた番地を渡す）を読む．1 つの RISC-V の L1 への要求は
//           同じ L1 アクセスポートを順に通り，ポートは要求を順序どおりに処理するので，この読みは
//           先に発行された書き込みがすべて処理された後で処理される
//   fence   この読みの結果が確定するまで後続の命令を EX1 へ進ませない（fence は先行ロードの値が
//           確定するまで LSU に入らず，後続の命令は fence が退役するまで EX1 に入れない）
// 抜けた時点で，それまでにこの RISC-V が L1 へ書いた値はすべて L1 にあり，以後のカウンタ更新・
// NoC の発行はその後に処理される．前の fence は先行ロードの完了も待つので，この RISC-V が
// それまでに L1 から読んだ値もすべて確定している（相手がその領域を上書きしてよい）．
// 読む側に要るのはこれとは別の手順で，相手の知らせ（カウンタ・セマフォ）を見た後に invalidate_l1_cache()（fence）で
// L0 を捨ててから L1 を読む．
//
// 1 回あたり数十サイクルなので，タイル 1 枚（1024 要素を数千サイクルかけて割る・組み立てる）に
// 1 回の頻度なら時間には現れない．
#include <cstdint>

inline __attribute__((always_inline)) void l1_writes_landed(uint32_t l1_addr) {
    uint32_t v;
    asm volatile("fence\n\t"
                 "lw %0, 0(%1)\n\t"
                 "fence"
                 : "=r"(v) : "r"(l1_addr) : "memory");
    (void)v;
}
