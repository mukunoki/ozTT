// SPDX-License-Identifier: Apache-2.0
//
// GEMM 段の CB 番号と計算カーネルの引数の並び（ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// 共有の行列積カーネル bmm_int32_spill.cpp（尾崎の両スキーム・gemm_int8・gemm_exact_check）と bmm_l1acc.cpp
// （gemm_bf16ext・gemm_exact_check の l1acc）のコンパイル時引数・実行時引数，データ移動カーネル mcast_split_dm.cpp と
// 計算カーネルが共有する CB の番号．ホストは common/mcast_gemm.hpp（mcast_compute_args・spill_gemm_spec・
// spill_compute_rt・build_mcast_gemm）がこの表の名前で詰める．データ移動カーネルの実行時引数の表は mcast_args.h．
// 番号だけで対応づけると，片側の並べ替えを検出できない（データ移動の側は mcast_args.h の表と static_assert で守る）．
//
// コンパイル時引数の 0〜11 番は tt-metal の標準の行列積カーネル（bmm_large_block_zm.cpp．gemm_bf16 が使う）と同じ並びで，
// gemm_bf16 は標準カーネルへ同じ mcast_compute_args を渡すので並びを変えてはならない（下の static_assert）．
// bmm_int32_spill.cpp は 7 番（CA_NUM_BLOCKS）を読まず，K ブロック数を実行時引数 CR_NUM_BLOCKS で受ける（尾崎の GEMM 段は
// 対角ごとに K の長さが変わるので，コンパイル済みのプログラムを付け替えて使い回すため）．それでも 7 番を詰めるのは並びを
// 共有するためで，読まれない 7 番の値が違えば JIT のキャッシュは別になる．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

namespace gemm_args {

// --- CB 番号 -------------------------------------------------------------------
constexpr uint32_t CB_IN0     = 0;    // A の入力（2 重緩衝）
constexpr uint32_t CB_IN1     = 1;    // B の入力（2 重緩衝）
constexpr uint32_t CB_OUT     = 16;   // 最終出力（ライタが読む）
constexpr uint32_t CB_PARTIAL = 24;   // K ブロック間の部分和（パッカの L1 累積．出力と同じ領域を共有することがある）
constexpr uint32_t CB_S       = 2;    // 融合（FUSE）：既存の S_d（Int32．融合ライタが先読みする）
constexpr uint32_t CB_TMP     = 3;    // 融合・対ループ：行列積の結果をいったん FP32 のまま逃がす先
constexpr uint32_t CB_SL      = 5;    // 対ループ（PAIRLOOP）：対をまたぐ Int32 の累積面

// --- 計算カーネルのコンパイル時引数 ---------------------------------------------------
enum ComputeCT : uint32_t {
    CA_IN0_BLOCK_W = 0,          // 1 K ブロックの幅 bw［タイル］
    CA_IN0_NUM_SUBBLOCKS,        // A のサブブロック数 pm/sh
    CA_IN0_BLOCK_NUM_TILES,      // A の 1 ブロックのタイル数
    CA_IN0_SUBBLOCK_NUM_TILES,   // A の 1 サブブロックのタイル数
    CA_IN1_NUM_SUBBLOCKS,        // B のサブブロック数 pn/sw
    CA_IN1_BLOCK_NUM_TILES,      // B の 1 ブロックのタイル数
    CA_IN1_PER_CORE_W,           // コアあたりの出力の幅 pn
    CA_NUM_BLOCKS,               // K ブロック数（bmm_l1acc.cpp と標準カーネルが読む．bmm_int32_spill.cpp は読まない）
    CA_OUT_SUBBLOCK_H,           // 出力サブブロックの高さ sh
    CA_OUT_SUBBLOCK_W,           // 同，幅 sw
    CA_OUT_SUBBLOCK_NUM_TILES,   // 同，タイル数 sh*sw
    CA_BATCH,                    // バッチ数．データ移動カーネルがバッチを扱わないので常に 1（標準カーネルと並びを共有するために残る）
    CA_STD_COUNT,                // 標準カーネル・bmm_l1acc.cpp が読む個数
    CA_FUSE = CA_STD_COUNT,      // bmm_int32_spill.cpp だけ：最終ブロックの出力を既存の S へ Int32 で足し込む
    CA_F32,                      // 同，typecast の形式（Float32）
    CA_I32,                      // 同，（Int32）
    CA_PAIRLOOP,                 // 同，対の境界で FP32 のアキュムレータを畳んで CB_SL へ足す
    CA_BLOCKMM,                  // 同，matmul_block を使う（0 なら matmul_tiles．OZ_NO_BLOCKMM）
    CA_COUNT
};
// 0〜11 番は位置で読む標準カーネル（bmm_large_block_zm.cpp の get_compile_time_arg_val(0..11)）と同じ並び．列挙子は暗黙の
// 連番なので，途中の 2 行を入れ替えても自前のカーネルとホストは名前で追随して気づかない．標準カーネルとの対応が壊れない
// よう，各欄の位置を固定する（変えない）
static_assert(CA_IN0_BLOCK_W == 0 && CA_IN0_NUM_SUBBLOCKS == 1 && CA_IN0_BLOCK_NUM_TILES == 2 &&
              CA_IN0_SUBBLOCK_NUM_TILES == 3 && CA_IN1_NUM_SUBBLOCKS == 4 && CA_IN1_BLOCK_NUM_TILES == 5 &&
              CA_IN1_PER_CORE_W == 6 && CA_NUM_BLOCKS == 7 && CA_OUT_SUBBLOCK_H == 8 && CA_OUT_SUBBLOCK_W == 9 &&
              CA_OUT_SUBBLOCK_NUM_TILES == 10 && CA_BATCH == 11 && CA_STD_COUNT == 12,
              "0〜11 番は tt-metal の標準の行列積カーネルと同じ並び（変えない）");
static_assert(CA_FUSE == 12 && CA_F32 == 13 && CA_I32 == 14 && CA_PAIRLOOP == 15 && CA_BLOCKMM == 16,
              "bmm_int32_spill.cpp だけが読む 12〜16 番の位置");
static_assert(CA_COUNT == 17, "bmm_int32_spill.cpp のコンパイル時引数は 17 個");

// --- 計算カーネル bmm_int32_spill.cpp の実行時引数 ----------------------------------------
enum ComputeRT : uint32_t {
    CR_NUM_BLOCKS = 0,   // このプログラムの K ブロック数
    CR_ACC,              // 融合のとき：その面への最初の書き込みなら 0（足さずに初期化），以降は 1（FUSE のときだけ読む）
    CR_BPP,              // 対ループのとき：1 対あたりの K ブロック数（1 以上で K ブロック数を割り切る．PAIRLOOP のときだけ読む）
    CR_COUNT
};
static_assert(CR_NUM_BLOCKS == 0 && CR_ACC == 1 && CR_BPP == 2 && CR_COUNT == 3, "bmm_int32_spill.cpp の実行時引数の位置");

}  // namespace gemm_args
