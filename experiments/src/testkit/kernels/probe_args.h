// SPDX-License-Identifier: Apache-2.0
//
// 探針のカーネルの引数の並び（ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// 番号だけで対応づけると，片側に 1 つ足したときに以降が黙ってずれる（ozaki2/kernels/recon_args.h の冒頭と同じ理由）．
// ホストはこの表の名前で添字を指定して詰め，カーネルは名前で読む．
//
// 探針は本番のリーダ・ライタ（recon_reader.cpp・modslice_writer.cpp）を借りず，下の小さいタイル運びのカーネルを使う．
// 本番のカーネルは本番の都合で引数を足し替えるので，借りると探針が JIT で落ちるか，黙って別の値を読む．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

// --- タイル運び（probe_tile_reader.cpp・probe_tile_writer.cpp）--------------------------
// 入力・出力とも，平面 p のタイル g を DRAM のタイル p*ntot + g に置く．タイルはビット列のまま運ぶ．
namespace probe_tile_args {

// リーダの実行時引数
enum ReaderRT : uint32_t {
    R_IN_ADDR = 0,   // 入力の先頭アドレス
    R_NTILES,        // 読むタイルの組の数（g = 0..R_NTILES-1）
    R_NTOT,          // 平面の間隔［タイル］
    R_COUNT
};
// リーダのコンパイル時引数（TensorAccessorArgs の後ろ）．1 組につき平面 0..R_NPL-1 を順に，平面 p は CB R_CB0+p へ
// 1 枚ずつ積む（同じ CB を並べれば 1 つの CB に R_NPL 枚が並ぶ）．R_NPL を超える番号の CB の欄も詰める（読まない）
enum ReaderCT : uint32_t {
    R_NPL = 0,       // 1 組の平面の数（1..R_MAX_PLANES）
    R_CB0,
    R_CB1,
    R_CB2,
    R_CB3,
    R_CT_COUNT
};
inline constexpr uint32_t R_MAX_PLANES = R_CT_COUNT - R_CB0;

// ライタの実行時引数．c_16 から 1 組につき W_NOUT 枚を受け取り，k 枚目をタイル k*ntot + g に置く
enum WriterRT : uint32_t {
    W_OUT_ADDR = 0,  // 出力の先頭アドレス
    W_NTILES,        // 書くタイルの組の数
    W_NTOT,          // 平面の間隔［タイル］
    W_NOUT,          // 1 組の出力の枚数
    W_COUNT
};

}  // namespace probe_tile_args

// --- 模擬と実機の SFPU の照合（checks/kernels/tfmul_probe.cpp．ホストは checks/tfmul_probe.cpp）-------------
namespace tfmul_probe_args {

// 計算カーネルのコンパイル時引数
enum ComputeCT : uint32_t {
    C_OP = 0,        // 下の Op
    C_RDX,           // OP_MUL の基数（float のビット列）．ほかの演算では読まない
    C_COUNT
};
// 計算カーネルの実行時引数
enum ComputeRT : uint32_t {
    R_NGROUPS = 0,   // 入力の組の数（どの平面も R_NGROUPS 枚）
    R_COUNT
};
// 実機で呼ぶ本番の関数（common/kernels/sfpu_tf.h・slice_tf_core.h）．入力は 1 組 nin 枚を DST 0..nin-1 へ写す
// （OP_LOAD だけはスライサと同じ 4 つの CB から tf_load_f64 が読む）．出力は 1 組 nvar 回 × nout 枚
enum Op : uint32_t {
    OP_ARITH = 0,    // Add(0,1,2)・Sub(0,1,3)・Mul(0,1,4)（タイル演算のラッパ）．出力 2,3,4
    OP_RND,          // Rnd(0)．出力 0
    OP_MUL,          // 1 回目 tfMulRadix<RDX>，2 回目 tfMulR(RDX)．出力 0,1,2 を 2 回
    OP_ADDS3,        // tfAddS<3>（3 本組 0,1,2 に 3 を足す）．出力 0,1,2
    OP_LOAD,         // tf_load_f64（hi・braw・craw・s の 4 枚から 3 本組）．出力 0,1,2
    OP_COUNT
};
constexpr uint32_t nin(uint32_t op)  { return op == OP_ARITH ? 2u : op == OP_RND ? 1u : op == OP_MUL ? 3u : 4u; }
constexpr uint32_t nout(uint32_t op) { return op == OP_RND ? 1u : 3u; }
constexpr uint32_t nvar(uint32_t op) { return op == OP_MUL ? 2u : 1u; }
// 出力の k 枚目を取る DST の番号
constexpr uint32_t out_reg(uint32_t op, uint32_t k) { return op == OP_ARITH ? 2u + k : k; }

}  // namespace tfmul_probe_args

// --- DST の FP32 の本数の探針（probes/kernels/dst_probe.cpp．ホストは probes/dst_probe.cpp）--------------------
namespace dst_probe_args {

// 計算カーネルのコンパイル時引数
enum ComputeCT : uint32_t {
    C_NREG = 0,      // 試すレジスタの本数
    C_NT,            // 繰り返す回数（出力は 1 回につき C_NREG 枚）
    C_COUNT
};

}  // namespace dst_probe_args

// --- SFPU の積和の保持幅の探針（probes/kernels/fma_probe.cpp．ホストは probes/fma_probe.cpp）-------------------
namespace fma_probe_args {

// 計算カーネルのコンパイル時引数
enum ComputeCT : uint32_t {
    C_MODE = 0,      // 下の Mode
    C_COUNT
};
// 計算の書き方（計算カーネルの冒頭の説明）
enum Mode : uint32_t {
    MODE_ADDCMUL = 0,   // addcmul_tile（SFPMAD 1 本）で a*b + c
    MODE_MULADD,        // Mul してから Add（本実装の書き方）
    MODE_MUL,           // 積 a*b だけ（経路の確認用）
    MODE_COUNT
};
// 計算カーネルの実行時引数
enum ComputeRT : uint32_t {
    R_C = 0,         // 足す定数 c（float のビット列）
    R_COUNT
};

}  // namespace fma_probe_args

// --- タイル積の速さの探針（probes/kernels/tile_rate_probe.cpp・tile_rate_fill.cpp．ホストは probes/tile_rate_probe.cpp）----
namespace tile_rate_args {

// 計算カーネルのコンパイル時引数
enum ComputeCT : uint32_t {
    C_BW = 0,        // K 方向のブロック幅［タイル］
    C_PM,            // 出力ブロックの高さ［タイル］
    C_PN,            // 同，幅
    C_SH,            // サブブロックの高さ
    C_SW,            // 同，幅
    C_MODE,          // 下の Mode
    C_COUNT
};
// 測る命令列（計算カーネルの冒頭の説明）
enum Mode : uint32_t {
    MODE_MM = 0,     // matmul_block（アンパッカの展開＋行列エンジンの積）
    MODE_MATH,       // 行列エンジンだけ
    MODE_UNPACK,     // アンパッカだけ
    MODE_MMPACK,     // MODE_MM に出力の処理（パックと L1 累積）を足す
    MODE_COUNT
};
// 計算カーネルの実行時引数
enum ComputeRT : uint32_t {
    R_REPS = 0,      // 1 ブロックの行列積の繰り返し回数
    R_RES_ADDR,      // 結果の欄（下の Result）の L1 番地
    R_COUNT
};
// L1 の結果の欄（語の番号）
enum Result : uint32_t {
    RES_T0 = 0,      // UNPACK スレッドが入力を受け取った時点の壁時計（下位 32 bit）
    RES_T1,          // PACK スレッドが最後のサブブロックを受け取った時点の壁時計
    RES_DONE,        // PACK スレッドが終わりに 1 を書く（計時の欄が書かれた印）
    RES_WORDS = 4    // ホストが 0 で埋めてから読み戻す語数
};
// 入力を埋めるデータ移動カーネル（tile_rate_fill.cpp）のコンパイル時引数
enum FillCT : uint32_t {
    F_N0 = 0,        // c_0 のタイル数
    F_N1,            // c_1 のタイル数
    F_B0,            // c_0 のバイト数
    F_B1,            // c_1 のバイト数
    F_COUNT
};

}  // namespace tile_rate_args
