// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：再構成段の引数の並び（ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// ホストは SetRuntimeArgs と compile_args へ uint32_t の列を渡し，カーネルは
// get_arg_val<uint32_t>(i) と get_compile_time_arg_val(i) で受け取る．番号だけで
// 対応づいているので，片側に 1 つ足すと以降が全部ずれる．ずれても型は同じ uint32_t
// なので，運が良ければ「Index out of range」で止まり，悪ければ黙って別の値を読む．
//
// そこでここに番号表を置き，両側がこれを参照する．ホストは *_COUNT 個の配列をこの表の名前で
// 添字指定して詰め，カーネルは enum の名前で読むので，並びを変えるときはここだけを直せばよい．
// リーダ・ライタのコンパイル時引数（下の ReaderWriterCT）は TensorAccessorArgs の後ろに続くので，ホストは
// ozaki2_stages.hpp の append_rw_ct で名前で詰める．CB 番号もここに置き，ホストとカーネルが同じ名前で引く．
// スライス段と合成段の表は oz2_args.h，両側が共有する上限は oz2_limits.h にある．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

namespace recon_args {

// --- CB 番号（ホストの recon_make_cbs（ozaki2_stages.hpp）が作り，リーダ・計算カーネル・ライタが使う）----------------
constexpr uint32_t CB_IN     = 0;    // 法ごとの入力（剰余か GEMM の出力）．1 出力タイルにつき S 枚
constexpr uint32_t CB_MASK   = 1;    // 符号-絶対値を割るマスク 2 枚（INT8 経路だけ．リーダが L1 で作る）
constexpr uint32_t CB_LIMB   = 16;   // frac の limb（計算カーネルが書き，ライタが出し入れし，リーダはアドレスだけを見る）
constexpr uint32_t CB_F64    = 19;   // ライタの作業：FP64 タイル 1 枚（OZ_F64_TILE_BYTES．リーダも同じ L1 を見る）
constexpr uint32_t CB_DLT_A  = 20;   // ライタの作業：行の指数差分
constexpr uint32_t CB_DLT_B  = 21;   // ライタの作業：列の指数差分
constexpr uint32_t CB_A128   = 22;   // ライタの作業：128 bit 固定小数 1 タイル（OZ_A128_TILE_BYTES．2 パス構成だけ）
constexpr uint32_t CB_SYNC   = 26;   // リーダとライタの受け渡し（limb のアドレス 1 語．組み立てを分担するときだけ）
constexpr uint32_t SYNC_PAGE = 64;   // CB_SYNC の 1 ページの大きさ［B］

// --- リーダ（recon_reader.cpp）の実行時引数 --------------------------------
enum Reader : uint32_t {
    R_IN_ADDR = 0,   // 入力（法ごとの平面，または剰余）の先頭アドレス
    R_NUM,           // このコアが担当する出力タイルの枚数
    R_START,         // 同，先頭のタイル番号
    R_NTOT,          // 出力タイルの総数（平面の間隔でもある）
    R_S,             // 法の本数
    R_MASK,          // 1 なら符号-絶対値を割るマスク 2 枚を L1 で作って c_1 へ流す（INT8 経路）
    R_NLIMB,         // 3 本組か 2 本組か
    R_WHI,           // 最終スケール W の上位 32 bit
    R_WLO,           // 同，下位 32 bit
    R_QW,            // W の指数（int32 をそのまま uint32 で運ぶ）
    R_SH2,           // 掛け戻す指数 2*(b-(ND-1)*G)
    R_JSPLIT,        // FP64 組み立て（2 パス構成の第 1 パスでは 128 bit 固定小数への変換）の分担の境目（リーダが前半 jsplit 要素）
    R_SEM_PUB,       // ライタが limb の置き場を知らせる番号
    R_SEM_DONE,      // リーダが前半を終えたと知らせる番号
    R_COUNT
};

// --- ライタ（recon_writer.cpp）の実行時引数 --------------------------------
enum Writer : uint32_t {
    W_OUT0 = 0,      // FINAL でないパスの書き先．2 パス構成の第 1 パスは 128 bit 固定小数（16 KB ページ），
                     // 診断用 OZ2_RECON_NOF64 の最終パスは limb 平面 0（平面 1・2 は W_OUT1・W_OUT2）
    W_NUM,
    W_START,
    W_OUT1,
    W_OUT2,
    W_NLIMB,
    W_O64,           // FP64 出力の先頭アドレス
    W_DLT_A,         // 行の指数差分
    W_DLT_B,         // 列の指数差分
    W_NT,            // 出力の列方向タイル数（タイル番号から行・列を割り出す）
    W_WHI,
    W_WLO,
    W_QW,
    W_SH2,
    W_A128,          // 2 パス構成の最終パスが読む，第 1 パスの 128 bit 固定小数（16 KB ページ）
    W_JLO,           // リーダへ渡す前半の要素数（分担しないときは 0）
    W_SEM_PUB,
    W_SEM_DONE,
    W_OUT_OFF,       // 出力を行の帯に分けるときの帯の先頭タイル（o64 と行・列の指数差分の番号に足す．帯 1 本なら 0）
    W_COUNT
};

// --- パスの役割（下の CT_ROLE の値）---------------------------------------
// 番号はパスの順ではない：1 パス構成は ROLE_SINGLE，2 パス構成は第 1 パスが ROLE_TWO_FIRST（自分の limb 3 本を 128 bit
// 固定小数へ足して書く），最終パスが ROLE_TWO_FINAL（第 1 パスの固定小数を読んで自分の 3 本を足し，FP64 を組み立てる）．
// 合成段は再構成のリーダを使わない（oz2_plane_reader.cpp）
enum ReconRole : uint32_t { ROLE_SINGLE = 0, ROLE_TWO_FINAL = 1, ROLE_TWO_FIRST = 2 };

// --- リーダ・ライタのコンパイル時引数（TensorAccessorArgs の後ろ）-----------
// どちらも同じ 3 つを同じ順で取る．TensorAccessorArgs の個数は側で違うので，
// 起点はそれぞれの next_compile_time_args_offset() から数える．
// CT_FINAL_OR_SHARE の意味は側で違うので，カーネルは別の名前で受ける：ライタ（FINAL）には「このパスが FP64 を組み立てる最終パスか」，
// リーダ（SHARE）には「組み立て（2 パス構成の第 1 パスでは 128 bit 固定小数への変換）の前半を分担するか」
// （ozaki2_plan.hpp の oz2_split_assembly．1 パス構成の最終パスと 2 パス構成の両パスで分担するとき 1 で，
// OZ2_JSPLIT=0 か 1024・診断用 OZ2_RECON_NOF64 の最終パスでは 0）を渡す．
enum ReaderWriterCT : uint32_t {
    CT_FINAL_OR_SHARE = 0,   // ライタ：FP64 を組み立てる最終パスか．リーダ：組み立ての前半を分担するか（上の注記）
    CT_ROLE,                 // パスの役割（ReconRole）
    CT_PROBE,        // 【調査用】1 なら assemble_frac_f64 を外す
    CT_COUNT
};

// --- 計算カーネル（recon_crt.cpp）のコンパイル時引数 -----------------------
enum CrtCT : uint32_t {
    C_S = 0,         // 法の本数
    C_NLIMB,         // 3 本組か 2 本組か
    C_IN_RESIDUE,    // 入力が畳んだ剰余か，GEMM 段の生の int32 か
    C_NRP,           // 逆数の断片の本数
    C_NSMALL_FRAG,   // 逆数の断片の末尾の何本を tfAddSmall で足すか（調査用のつまみ OZ2_NSMALL で変えられる）
    C_PROBE,         // 【調査用】CPROBE 0--2
    C_RAWMASK,       // 法 i のビットが立っていれば，入力は合成段を通らない GEMM の出力（1 桁の法．法へ畳んでから使う）
    C_COUNT
};

// --- 計算カーネルの実行時引数 ----------------------------------------------
// 先頭がタイル枚数で，そのあとに法ごとの定数が S 本ずつ並ぶ．
//   [0]                     担当タイル枚数
//   [1 + 0*S + i]           m_i
//   [1 + 1*S + i]           1/m_i
//   [1 + 2*S + i]           y_i = (M/m_i)^{-1} mod m_i
//   [1 + (3+t)*S + i]       逆数の断片 t（t = 0..NRP-1）
//   [1 + (3+NRP)*S + i]     2^16 mod m_i（剰余入力では渡さない）
// 番号を直に書くと，ホストの詰め方とカーネルの読み方が別々にずれる．
inline constexpr uint32_t crt_ntiles()                     { return 0u; }
inline constexpr uint32_t crt_mod(uint32_t S, uint32_t i)  { return 1u + 0u * S + i; }
inline constexpr uint32_t crt_rinv(uint32_t S, uint32_t i) { return 1u + 1u * S + i; }
inline constexpr uint32_t crt_yi(uint32_t S, uint32_t i)   { return 1u + 2u * S + i; }
inline constexpr uint32_t crt_frag(uint32_t S, uint32_t t, uint32_t i) {
    return 1u + (3u + t) * S + i;
}
inline constexpr uint32_t crt_m16(uint32_t S, uint32_t nrp, uint32_t i) {
    return 1u + (3u + nrp) * S + i;
}
inline constexpr uint32_t crt_count(uint32_t S, uint32_t nrp, bool in_residue) {
    return 1u + (3u + nrp + (in_residue ? 0u : 1u)) * S;
}

}  // namespace recon_args
