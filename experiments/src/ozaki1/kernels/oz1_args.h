// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：スライス段・再構成段・Int32 の累積のカーネルの引数の並びと CB 番号（ホストとカーネルが同じ 1 つの表を見る）
// ---------------------------------------------------------------------------
// ホスト（ozaki1_slice.hpp の slice_build・slice_set_args，ozaki1_recon.hpp の recon_build_prog・recon_set_out_off，
// ozaki1_accum.hpp の accum_*）は SetRuntimeArgs と compile_args へ uint32_t の列を渡し，カーネルは get_arg_val<uint32_t>(i) と
// get_compile_time_arg_val(i) で受け取る．番号だけで対応づけると，片側に 1 つ足したときに以降が黙ってずれる
// （ozaki2/kernels/recon_args.h の冒頭と同じ理由）．そこでここに表を置き，ホストは表の名前で添字を指定して詰め，
// カーネルは名前で読む．スライサの入力（FP64 のリーダ）の表は両スキーム共通の common/kernels/slice_f64_args.h，
// スケール段の表は common/kernels/scale_args.h にある．
//
// 3 本組の limb の並びに注意：再構成の計算カーネルが出してライタが読む CB と，ライタの引数（DRAM の limb 面の
// アドレス・TensorAccessorArgs）は hi・lo・mid の順（中位が最後）で，リーダが前ブロックまでの累積を読む並び
// （hi・mid・lo）と違う．取り違えやすいので，どれもここの名前で引く．
//
// このヘッダはデバイス API に依存しない（<cstdint> だけ）．
#pragma once
#include <cstdint>

namespace oz1_args {

// ===== スライス段 =====
// --- スライサ（slice_tf_num.cpp・slice_tf_sm8.cpp．共通の本体は slice_tf_body.h）-----------------------------------
constexpr uint32_t CB_SLICE_OUT = 16;   // スライスの出力（ライタ slice_writer.cpp が読む）
enum SlicerCT : uint32_t {
    SC_NSLICE = 0,   // 入力タイル 1 枚から出すスライスの本数 nslice（= L+1．位置引数の p ではない．ozaki1_plan.hpp の PanelPlan）
    SC_RADIX,        // 基数（桁の重み）の float のビット列
    SC_SF_IN,        // 1 ならリーダが soft-float 版で，3 本組をそのまま寄こす（測定用）
    SC_DMAX,         // 桁の上限の float のビット列
    SC_NDMAX,        // その負値
    SC_OUT_HALF,     // 1 ならスライスを binary16 のビット列（UInt16）で出す（FP16 経路．slice_tf_num.cpp だけが読む）
    SC_COUNT
};
enum SlicerRT : uint32_t { SR_NTILES = 0, SR_COUNT };

// --- スライスのライタ（slice_writer.cpp）-------------------------------------------------
// コンパイル時引数（出力の TensorAccessorArgs の後ろ）
enum SliceWriterCT : uint32_t {
    SW_CT_SPLIT = 0,   // 1 ならリーダが郵便受けに置く後半の仕事を引き受ける（common/kernels/slice_f64_split.h）
    SW_CT_COUNT
};
// 実行時引数（書き先の式は slice_writer.cpp の冒頭）
enum SliceWriterRT : uint32_t {
    SW_DST_ADDR = 0,   // スライス連結バッファの先頭アドレス
    SW_NUM,            // このコアが担当する入力タイルの枚数
    SW_START,          // 同，先頭のタイル番号
    SW_NSLICE,         // スライスの本数 nslice（計算カーネルの SC_NSLICE と同じでなければならない．slice_set_args が照合する）
    SW_S1,             // K 方向のタイル数 Kto（A・B とも）
    SW_S2,             // B の列方向のタイル数 Nt（A では使わない）
    SW_MODE,           // 0 = A，1 = B
    SW_KBT,            // 1 K ブロックの K タイル数（Kto を割り切ること）
    SW_COUNT
};

// ===== 再構成段 =====
// --- CB 番号 -------------------------------------------------------------------------
constexpr uint32_t CB_RECON_IN = 0;   // 対角ごとの部分和 P_d（ndiag 枚ずつ）
constexpr uint32_t CB_MASK     = 1;   // Int32 を 2 本組へ分けるマスク 2 枚（Int32 入力だけ．リーダが L1 で作る）
constexpr uint32_t CB_ACC      = 2;   // 前ブロックまでの累積の 3 本組（ACC=1．hi・mid・lo の順）
constexpr uint32_t CB_LIMB_HI  = 16;  // 計算カーネルの出力の 3 本組（ライタが読む）．hi・lo・mid の番号の順
constexpr uint32_t CB_LIMB_LO  = 17;
constexpr uint32_t CB_LIMB_MID = 18;
constexpr uint32_t CB_F64      = 19;  // ライタの作業：FP64 の出力タイル 1 枚（f64_tf.h の OZ_F64_TILE_BYTES）
constexpr uint32_t CB_DLT_A    = 20;  // ライタの作業：行の指数差分
constexpr uint32_t CB_DLT_B    = 21;  // ライタの作業：列の指数差分

// --- リーダ（recon_reader_f32in.cpp・recon_reader_i32in.cpp）---------------------------------
// コンパイル時引数．f32in 版は TensorAccessorArgs（P・累積の hi・mid・lo の順）だけ．i32in 版は先頭に RI_CT_SIGNMAG を
// 置き，その後ろに P の TensorAccessorArgs（累積を読まない）
enum ReaderI32CT : uint32_t {
    RI_CT_SIGNMAG = 0,   // 1 なら符号-絶対値（INT8 経路の P_d），0 なら 2 の補数（遅延再構成の累積 S_d）
    RI_CT_COUNT
};
// 実行時引数．i32in 版は RR_SLOT_STRIDE までの 5 語（累積入力を持たない）
enum ReaderRT : uint32_t {
    RR_SRC_ADDR = 0,     // 部分和 Pbig（i32in 版は Sbig）の先頭アドレス
    RR_NUM,              // このコアが担当する出力タイルの枚数
    RR_START,            // 同，先頭のタイル番号
    RR_NDIAG,            // 対角の本数 L+1
    RR_SLOT_STRIDE,      // 対角の面の間隔［タイル］（既定 Mt*Nt．行バンドではバンドの確保幅）
    RR_I32_COUNT,        // i32in 版の個数
    RR_ACC_HI = RR_I32_COUNT,   // 累積の 3 本組の先頭アドレス（f32in 版だけ．0 なら累積なし）
    RR_ACC_MID,
    RR_ACC_LO,
    RR_COUNT             // f32in 版の個数
};

// --- 計算カーネル（recon_tf_f32in.cpp・recon_tf_i32in.cpp）-----------------------------------
enum ReconCT : uint32_t {
    RC_NDIAG = 0,    // 対角の本数 L+1
    RC_IN_FMT,       // 部分積の形式値（i32in 版は Int32）
    RC_F32,          // Float32 の形式値
    RC_RADIX,        // 基数の float のビット列
    RC_ACC,          // 1 なら前ブロックまでの累積を CB_ACC から読んで足す（f32in 版だけが読む）
    RC_SIGNMAG,      // 1 なら符号-絶対値（i32in 版だけが読む）
    RC_NSINGLE_CAST, // 先頭の何本を 1 回の型変換で読むか（i32in 版だけが読む．ホストのつまみは OZ_NSMALL）
    RC_COUNT
};
enum ReconRT : uint32_t { RCR_NTILES = 0, RCR_COUNT };

// --- ライタ（recon_f64_writer.cpp）--------------------------------------------------------
// コンパイル時引数：TensorAccessorArgs が下の WriterTA の順に 6 つ並び，その後ろに WriterCT
enum WriterTA : uint32_t { TA_HI = 0, TA_LO, TA_MID, TA_OUT, TA_DLT_A, TA_DLT_B, TA_COUNT };
enum WriterCT : uint32_t {
    RW_CT_EMIT_LIMBS = 0,   // 3 本の limb を書き出す（K ブロック化のとき最終ブロック以外）
    RW_CT_EMIT_F64,         // FP64 を組み立てる（最終ブロックだけ）
    RW_CT_SF,               // 組み立てを C の double 式（soft-float）で行う測定モード
    RW_CT_PROBE,            // 【調査用】OZ_RECON_PROBE（0--3）
    RW_CT_COUNT
};
enum WriterRT : uint32_t {
    RW_HI_ADDR = 0,   // limb 面 hi の先頭アドレス
    RW_NUM,           // このコアが担当する出力タイルの枚数
    RW_START,         // 同，先頭のタイル番号（バンド内の局所番号）
    RW_LO_ADDR,       // limb 面 lo
    RW_MID_ADDR,      // limb 面 mid
    RW_OUT_ADDR,      // FP64 の出力
    RW_DLT_A_ADDR,    // 行の指数差分
    RW_DLT_B_ADDR,    // 列の指数差分
    RW_NT,            // 出力の列方向のタイル数
    RW_WHI,           // 最終スケール W の仮数の上位 32 bit
    RW_WLO,           // 同，下位 32 bit
    RW_QW,            // W の指数（int32 をそのまま uint32 で運ぶ）
    RW_WDLO,          // soft-float 版の wscale（double のビット列）の下位語
    RW_WDHI,          // 同，上位語
    RW_OUT_OFF,       // 行バンドの出力側のタイル offset（バンドに切らない構成では 0．recon_set_out_off が差し替える）
    RW_COUNT
};

// ===== Int32 の K 方向累積（遅延再構成）=====
constexpr uint32_t CB_ACCUM_P   = 0;    // 入力：P_d（FP32）
constexpr uint32_t CB_ACCUM_S   = 1;    // 入力：前ブロックまでの S_d（Int32）
constexpr uint32_t CB_ACCUM_OUT = 16;   // 出力：S_d（Int32）
// リーダ（accum_i32_reader.cpp）：コンパイル時引数は先頭に AR_CT_ACC，その後ろに TensorAccessorArgs（P・S の順）
enum AccumReaderCT : uint32_t { AR_CT_ACC = 0, AR_CT_COUNT };
enum AccumReaderRT : uint32_t {
    AR_P_ADDR = 0,   // Pbig（FP32）
    AR_NUM,          // 担当タイル数
    AR_P_START,      // P 側の先頭タイル
    AR_S_ADDR,       // Sbig（Int32）
    AR_S_START,      // S 側の先頭タイル
    AR_COUNT
};
// ライタ（accum_i32_writer.cpp）：コンパイル時引数は TensorAccessorArgs（S）だけ
enum AccumWriterRT : uint32_t { AW_S_ADDR = 0, AW_NUM, AW_START, AW_COUNT };
// 計算カーネル（accum_i32.cpp）
enum AccumCT : uint32_t { AC_ACC = 0, AC_F32, AC_I32, AC_COUNT };
enum AccumRT : uint32_t { ACR_NTILES = 0, ACR_COUNT };

}  // namespace oz1_args
