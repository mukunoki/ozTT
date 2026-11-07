#pragma once
// 尾崎スキーム I：完全オンデバイスの GEMM パイプラインの段（スケール・スライス・GEMM・再構成）
// ---------------------------------------------------------------------------
// scheme I の 8 つの形式（BF16 / FP16 / FP32(TF32) / BFP8_b / FP8 / INT8 / BFP4_b / BFP2_b）で共通の実装．
// ベンチマークの main() は src/ozaki1/ozaki1_main.cpp にある．
//
// 工程は次の 4 段で，この順に実行する．どれもデバイス上で走る（中間データをホストへ戻さない）．
//   スケール算出  行ごと（A）／列ごと（B）の最大絶対値を求め，正規化の指数差分を作る（FP64 から直接）
//   スライス分解  FP64 を triple-float に変換し，正規化しながら基数 radix の小整数スライスに分解する
//   GEMM          同じ対角 d = i+j のスライス対をまとめて 1 本の整数 GEMM で計算する
//   再構成        対角ごとの部分和を triple-float で Horner 累積して復元し，
//                 radix^-(L+2) と 2 のべきスケールを掛けて FP64 に組み立てる
//
// 本ヘッダは取りまとめで，中身は次の 5 本（とスキーム II と共通のヘッダ）に分けてある．
// 下の順（依存の順）に取り込む．
//   ozaki1_format.hpp  形式ごとの値 OZ1_PARAM（表は ozaki1_param.h．形式の選択 OZ_FMT_* と形式の性質 OZ_FMT は
//                      scheme II と共通の oz_format.hpp）・桁の上限・fidelity・無誤差条件（ozaki_check_exact /
//                      ozaki_block_k / oz_defer_fits / oz_horner_fits）・OZ_SOFTFLOAT・OZ_DBG．ホスト側の判定だけ
//   ozaki_scale.hpp    スケール算出と FP64 の転送（scheme II と共通）
//   ozaki1_gemm.hpp    GEMM：GemmMode・L1 の見積り・構成探索（choose_cfg_1b など）・gemm_build / gemm_run
//   ozaki1_recon.hpp   再構成：ReconScale・ReconCtx / recon_*・OZ_RECON_PROBE
//   ozaki1_slice.hpp   スライス分解：SliceCtx / slice_build / slice_run（float のビット列 f32_bits は common/oz_bits.h）
// gemm / recon / slice は互いに依存せず，どれも format だけを前提にする．
// 段の上に載るもの（ozaki1_main.cpp が取り込む）：計画 ozaki1_plan.hpp，OZ_FP64 のモデル ozaki1_autop.hpp，
// 段の資源と投入の手順 ozaki1_stages.hpp，OZ_NO_FUSE_ACC の累積 ozaki1_accum.hpp，OZ_TIME_WHOLE ozaki1_whole.hpp．

#include "ozaki1/ozaki1_format.hpp"
// スケール算出（デバイス上）は scheme II と共通である．
#include "common/ozaki_scale.hpp"
#include "ozaki1/ozaki1_gemm.hpp"
#include "ozaki1/ozaki1_recon.hpp"
#include "ozaki1/ozaki1_slice.hpp"
