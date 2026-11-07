// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：スライス段と合成段の引数の並び（ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// 番号だけで対応づけると，片側に 1 つ足したときに以降が黙ってずれる（recon_args.h の冒頭と同じ理由）．
// ホスト（ozaki2_stages.hpp の slice_set_args・build_compose・compose_set_args）はこの表の名前で添字を指定して詰め，
// カーネル（modslice_digit.cpp・modslice_int8.cpp・modslice_writer.cpp・oz2_plane_reader.cpp・oz2_compose.cpp）は名前で読む．
// 再構成段の表は recon_args.h，両側が共有する上限（OZ2_MAXS・OZ2_MAXPROD）と桁の作り方の番号は oz2_limits.h にある．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>
#include "oz2_limits.h"

namespace oz2_args {

// --- CB 番号 -------------------------------------------------------------------
// スライス段の入力の 4 枚と残差は両スキーム共通の表（common/kernels/slice_f64_args.h），ここは尾崎 II だけのもの．
constexpr uint32_t CB_PLANE_IN  = 0;    // 合成段の入力：平面リーダ oz2_plane_reader.cpp が積む GEMM の出力（出力タイル 1 枚につき np 枚）
constexpr uint32_t CB_PLANE_OUT = 16;   // ライタ modslice_writer.cpp が DRAM へ書く平面（スライサの剰余・桁行列，合成段の剰余）
constexpr uint32_t CB_DIGIT_HI  = 26;   // A' の上位の桁 d_{ND-1}..d1（作った順．oz2_digits.h）
constexpr uint32_t CB_DIGIT_LO  = 27;   // A' の桁 d0

// --- スライサ（modslice_digit.cpp・modslice_int8.cpp）のコンパイル時引数 ---------------------
enum SlicerCT : uint32_t {
    SC_S = 0,        // 法の本数
    SC_RDX,          // A' を分ける基数 2^OZ2_DIGIT_BITS の float のビット列
    SC_NDIGIT,       // A' の 2^20 進の桁数（3〜OZ2_MAXDIGIT）
    SC_NRDIG,        // 剰余を分ける桁の数の上限（2 か 5．modslice_digit.cpp だけが読む）
    SC_OUT_HALF,     // 1 なら桁を binary16 のビット列で出す（FP16 経路．modslice_digit.cpp だけが読む）
    SC_COUNT
};

// --- スライサの実行時引数 ---------------------------------------------------------
// 先頭がタイル枚数で，そのあとに法ごとの定数が S 本ずつ並ぶ（ND は A' の桁数）．
//   [0]                     担当タイル枚数
//   [1 + 0*S + i]           m_i
//   [1 + 1*S + i]           1/m_i
//   [1 + (2+j)*S + i]       2^(20(j+1)) mod m_i（j = 0..ND-2．A' の桁の重みを法で畳んだもの）
// 桁分割の経路（modslice_digit.cpp）はさらに
//   [1 + (ND+1)*S + i]      桁分割の基数 q_i
//   [1 + (ND+2)*S + i]      1/q_i
//   [1 + (ND+3)*S + i]      桁の作り方（Oz2XMode の番号の float のビット列）
//   [1 + (ND+4)*S + i]      平衡 n 桁の桁数 n（uint32．他の方式は 0）
// 番号を直に書くと，ホストの詰め方とカーネルの読み方が別々にずれる．カーネルは法 0 の位置（i = 0）を constexpr で求めて
// から実行時の i を足す（実行時の i を渡して呼ぶと，同じ値でも生成コードのレジスタの割り当てが変わるため）．
inline constexpr uint32_t sl_ntiles()                                   { return 0u; }
inline constexpr uint32_t sl_mod(uint32_t S, uint32_t i)                { return 1u + 0u * S + i; }
inline constexpr uint32_t sl_rinv(uint32_t S, uint32_t i)               { return 1u + 1u * S + i; }
inline constexpr uint32_t sl_weight(uint32_t S, uint32_t j, uint32_t i) { return 1u + (2u + j) * S + i; }
inline constexpr uint32_t sl_q(uint32_t S, uint32_t nd, uint32_t i)     { return 1u + (nd + 1u) * S + i; }
inline constexpr uint32_t sl_qinv(uint32_t S, uint32_t nd, uint32_t i)  { return 1u + (nd + 2u) * S + i; }
inline constexpr uint32_t sl_xmode(uint32_t S, uint32_t nd, uint32_t i) { return 1u + (nd + 3u) * S + i; }
inline constexpr uint32_t sl_ndig(uint32_t S, uint32_t nd, uint32_t i)  { return 1u + (nd + 4u) * S + i; }
inline constexpr uint32_t sl_count(uint32_t S, uint32_t nd, bool digitk) {
    return 1u + (nd + 1u + (digitk ? 4u : 0u)) * S;
}

// --- スライスのライタ（modslice_writer.cpp．合成段も使う）-------------------------------
// コンパイル時引数（出力の TensorAccessorArgs の後ろ）
enum WriterCT : uint32_t {
    SW_CT_LAYOUT = 0,   // 置き方（下の Layout）
    SW_CT_COUNT
};
enum Layout : uint32_t {
    LAYOUT_PLANES = 0,   // 行列 i をタイル i*ntot + g に置く（INT8 経路のスライスと合成段）
    LAYOUT_A_CAT = 1,    // 桁分割経路の A：法の桁行列を K 方向に連結する（続く 1 語 SW_A_KTILES が A の K 方向のタイル数）
    LAYOUT_B_PERM = 2,   // 桁分割経路の B：桁行列 i をタイル perm[i]*ntot + g に置く（続く S 語が perm）
};
// 実行時引数．先頭 SW_FIXED 語は置き方によらず，続きは置き方で違う（LAYOUT_A_CAT は 1 語，LAYOUT_B_PERM は S 語）
enum WriterRT : uint32_t {
    SW_DST_ADDR = 0,    // 出力の先頭アドレス
    SW_NUM,             // このコアが担当する入力タイルの枚数
    SW_START,           // 同，先頭のタイル番号（合成段は出力の平面をここに載せる）
    SW_PLANES,          // 入力タイル 1 枚あたりの出力タイルの枚数（平面の数）
    SW_NTOT,            // 行列 1 枚のタイル数（平面の間隔）
    SW_FIXED
};
inline constexpr uint32_t SW_A_KTILES = SW_FIXED;                            // LAYOUT_A_CAT：A の K 方向のタイル数 Ct
inline constexpr uint32_t sw_perm(uint32_t s) { return SW_FIXED + s; }       // LAYOUT_B_PERM：桁行列 s の置き先の平面
inline constexpr uint32_t sw_count(uint32_t layout, uint32_t S) {
    return SW_FIXED + (layout == LAYOUT_A_CAT ? 1u : layout == LAYOUT_B_PERM ? S : 0u);
}

// --- 合成段の平面リーダ（oz2_plane_reader.cpp）の実行時引数 ------------------------------------
enum PlaneReaderRT : uint32_t {
    PR_IN_ADDR = 0,     // 入力（桁積の平面）の先頭アドレス
    PR_NUM,             // このコアが担当する出力タイルの枚数
    PR_START,           // 同，先頭のタイル番号（読む平面の先頭 in0 を in0*ntot として載せる）
    PR_NTOT,            // 平面 1 枚のタイル数（平面の間隔）
    PR_NPLANES,         // 1 出力タイルにつき読む平面の枚数 np
    PR_COUNT
};

// --- 合成段の計算カーネル（oz2_compose.cpp）の実行時引数 -----------------------------------
// 実行時引数の個数はカーネルごとに一定でなければならないので，係数は OZ2_MAXPROD 個ぶんの枠を常に確保し，
// 使わない分は 0 を詰める（法ごとに引数だけを差し替えて同じワークロードを投入し直すため）．
enum ComposeRT : uint32_t {
    CP_NTILES = 0,      // 担当タイル枚数
    CP_NP,              // 足し合わせる出力（同じ位の桁積をまとめた GEMM の出力）の本数 np（<= OZ2_MAXPROD）
    CP_MOD,             // 法 m の float のビット列
    CP_RINV,            // 1/m の float のビット列
    CP_FOLD_FIRST,      // 1 なら桁積を先に法へ畳む方式（ozaki2_moduli.h の fold_first）
    CP_FIXED
};
inline constexpr uint32_t cp_coeff_lo(uint32_t j) { return CP_FIXED + j; }                       // B_j = c_j mod m
inline constexpr uint32_t cp_coeff_hi(uint32_t j) { return CP_FIXED + (uint32_t)OZ2_MAXPROD + j; }  // A_j = c_j*2^12 mod m
inline constexpr uint32_t CP_COUNT = CP_FIXED + 2u * (uint32_t)OZ2_MAXPROD;

}  // namespace oz2_args
