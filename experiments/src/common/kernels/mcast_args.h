// SPDX-License-Identifier: Apache-2.0
//
// マルチキャスト行列積のデータ移動カーネルの実行時引数の並び（ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// mcast_split_dm.cpp が get_arg_val<uint32_t>(i) で読み，ホストの
// mcast_reader_args・mcast_writer_args（ozaki_gemm_host.hpp）と McastGemm::set_args・
// build_mcast_gemm（mcast_gemm.hpp）が同じ番号で詰める．番号だけで対応づいているので，
// 片側に 1 つ足すと以降が全部ずれ，型が同じ uint32_t なので黙って別の値を読む
// （recon_args.h の冒頭と同じ事情）．ホストは配列をこの表の名前で添字指定して詰め
// （ozaki_gemm_host.hpp の mcast_reader_args・mcast_writer_args，mcast_gemm.hpp の後半の送信），
// 詰め忘れは番兵のまま残って止まる．区画どうしの関係（別名・連続・個数）は下の static_assert が見る．
// 並びを変えるときはここだけを直す．
//
// 全体は 3 つの区画をこの順に連結したもの．
//   0..32   リーダ：tt-metal の matmul_common のリーダ 4 種の並びから，末尾のバッチの 4 個（Mt*Kt・Kt*Nt・バッチ数・
//           B の使い回し．GEMM 段はバッチを扱わない）と，A・B のブロックの高さ（in0_block_h・in1_block_h．カーネルは
//           読まない）を除いた 33 個（RA_*）．並びは tt-metal のリーダの並びの名残だが，その並びに依存する箇所は無い
//           （付け替えも表の名前で行う）
//   33..39  後半の送信：行／列の反対の端のコアが 1 ブロックの後半を配るための 7 個（X_*）
//   40..    ライタ（B 側のカーネルだけ）：tt-metal の writer_bmm_tile_layout.cpp の並びから，バッチの 2 個（Mt*Nt・
//           バッチ数）を除いた 11 個に，融合ライタ（WRITER=2）が読む acc を足した 12 個（W_*．オフセットは W0 から）
//
// マルチキャストの矩形は (hi, lo) の 2 隅で渡す．hi は座標の大きい側の隅，lo は小さい側の隅で，
// tt-metal のリーダの引数名では start が hi，end が lo にあたる（送信側が NOC 0 のときの並び）．
// カーネルは NOC の向きに合わせて並べ替える（mcast_split_dm.cpp の MCAST_ADDR）．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

namespace mcast_args {

// --- リーダ（33 個．A は 0..6，B は 7..13 が同じ並び）------------------------
enum Reader : uint32_t {
    RA_A = 0,          // A のオペランド 7 個の先頭（アドレス・開始タイル・刻み・ブロックの形）
    RA_A_START = 1,    // A の開始タイル（scheme I の gemm_run が付け替える）
    RA_B = 7,          // B のオペランド 7 個の先頭
    RA_B_START = 8,    // B の開始タイル（同）
    RA_NUM_BLOCKS = 14,  // K ブロック数（同）
    RA_MC_A = 15,      // A のマルチキャスト 9 個（hi_x, hi_y, lo_x, lo_y, 宛先数, 送信コア x, y,
                       //                        送信側の空き待ちセマフォ, 前半のフラグ）
    RA_MC_B = 24,      // B のマルチキャスト 9 個（並びは A と同じ）
    RA_COUNT = 33
};
// オペランド 7 個の中の位置（RA_A / RA_B からのずれ）．ブロックの幅 OP_BLOCK_W はカーネルが読む（A は K ブロック幅，B は
// コアあたりの列のタイル数）．tt-metal のリーダにあるブロックの高さはカーネルが読まないので渡さない（1 ブロックのタイル数は
// OP_BLOCK_TILES で渡る）
enum Operand : uint32_t {
    OP_ADDR = 0, OP_START = 1, OP_STRIDE_W = 2, OP_STRIDE_H = 3, OP_NEXT_BLOCK = 4,
    OP_BLOCK_W = 5, OP_BLOCK_TILES = 6, OP_COUNT = 7
};
// マルチキャスト 9 個の中の位置（RA_MC_A / RA_MC_B からのずれ）
enum Mcast : uint32_t {
    MC_HI_X = 0, MC_HI_Y = 1, MC_LO_X = 2, MC_LO_Y = 3, MC_NUM_DESTS = 4,
    MC_SENDER_X = 5, MC_SENDER_Y = 6, MC_SENDER_SEM = 7, MC_FLAG_SEM = 8, MC_COUNT = 9
};

// --- 後半の送信（7 個）-------------------------------------------------------
enum Second : uint32_t {
    X_S1_X = RA_COUNT,   // 後半の送信コア（A は行の右端，B は列の下端）の座標
    X_S1_Y,
    X_HI_X,              // その送信先の矩形の hi 隅（A は列 cc-2，B は行 cr-2）
    X_HI_Y,
    X_LO_X,              // 同，lo 隅（A は列 0，B は行 0）
    X_LO_Y,
    X_FLAG_SEM,          // 後半のフラグのセマフォ
    X_END
};
constexpr uint32_t X_COUNT = X_END - X_S1_X;   // 7

// --- ライタ（B 側のカーネルだけ．W0 からのずれ）------------------------------
constexpr uint32_t W0 = X_END;                 // 40
enum Writer : uint32_t {
    W_ADDR = 0,          // 出力のアドレス
    W_START = 1,         // 開始タイル（scheme I の gemm_run が付け替える）
    W_STRIDE_W = 2, W_STRIDE_H = 3,
    W_NEXT_SB_W = 4, W_NEXT_SB_H = 5,
    W_SB_W = 6, W_SB_H = 7, W_SB_TILES = 8,
    W_NSB_W = 9, W_NSB_H = 10,
    W_ACC = 11,          // 融合ライタだけが読む：既存の S を読んで足すか（標準のライタは 10 番までの 11 個しか読まない）
    W_COUNT = 12         // acc まで含めた個数（ホストはどのライタにも 12 個渡す）
};

// 区画の関係の検査（並べ替え・足し間違いをコンパイル時に止める）．列挙型が区画ごとに別なので，
// 比べるときは u() で uint32_t に揃える（別の列挙型どうしの比較は -Werror=enum-compare で止まる）
constexpr uint32_t u(uint32_t v) { return v; }
static_assert(u(RA_A) == 0 && u(RA_B) == u(RA_A) + u(OP_COUNT), "A・B のオペランドは 0 番から 7 個ずつ続く");
static_assert(u(RA_A_START) == u(RA_A) + u(OP_START) && u(RA_B_START) == u(RA_B) + u(OP_START),
              "RA_A_START・RA_B_START はオペランドの OP_START の別名");
static_assert(u(RA_NUM_BLOCKS) == u(RA_B) + u(OP_COUNT), "ブロック数は B のオペランドの直後");
static_assert(u(RA_MC_A) == u(RA_NUM_BLOCKS) + 1 && u(RA_MC_B) == u(RA_MC_A) + u(MC_COUNT),
              "マルチキャストは A・B の順に 9 個ずつ");
static_assert(u(RA_COUNT) == u(RA_MC_B) + u(MC_COUNT), "リーダは B のマルチキャストで終わる");
static_assert(u(OP_COUNT) == 7 && u(OP_BLOCK_TILES) + 1 == u(OP_COUNT), "オペランドは 7 個で，ブロックのタイル数で終わる");
static_assert(u(RA_COUNT) == 33, "リーダは 33 個（tt-metal の matmul_common のリーダの先頭 35 個からバッチの 4 個と"
                                 "ブロックの高さ 2 個を除いたもの）");
static_assert(u(X_S1_X) == u(RA_COUNT) && u(X_S1_Y) == u(X_S1_X) + 1 && u(X_HI_X) == u(X_S1_X) + 2 &&
              u(X_HI_Y) == u(X_S1_X) + 3 && u(X_LO_X) == u(X_S1_X) + 4 && u(X_LO_Y) == u(X_S1_X) + 5 &&
              u(X_FLAG_SEM) == u(X_S1_X) + 6 && X_COUNT == 7, "後半の送信の 7 個はリーダの直後");
static_assert(W0 == u(RA_COUNT) + X_COUNT, "ライタの引数はリーダと後半の送信の後ろ");
static_assert(u(W_ADDR) == 0 && u(W_NSB_H) == 10 && u(W_ACC) == u(W_NSB_H) + 1 &&
              u(W_COUNT) == u(W_ACC) + 1, "ライタは標準の 11 個に acc を足した 12 個");

}  // namespace mcast_args
