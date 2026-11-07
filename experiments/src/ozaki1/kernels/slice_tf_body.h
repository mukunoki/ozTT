// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 尾崎スキーム I：2 本のスライサ（slice_tf_num.cpp・slice_tf_sm8.cpp）が共有する本体（kernel_main）
// ---------------------------------------------------------------------------
// 2 本の違いは，切り出した桁 d を出力の形へ符号化する部分と，パックする DST の番号だけである．その違いを取り込む
// 側の constexpr bool SLICE_SIGNMAG8（slice_tf_sm8.cpp が true）で選び，入力の受け取り・基数倍・丸め・クランプ・
// 残差の更新・出力・残差の持ち回りはここの 1 か所に置く（片方だけ直すと，形式によって手順が黙って食い違うため）．
// 取り込む側は SLICE_SIGNMAG8 を置いてからこのヘッダを取り込む（ここが kernel_main を定義する）．
// 符号化を関数の型（テンプレートの引数）で渡す形にすると，同じ計算でもレジスタの割り当てが変わって機械語が
// 2 本に分ける前と違ったので，本体は 1 つの関数のままにして if constexpr で選ぶ（分ける前と同じ機械語）．
// 入力の受け渡しと残差の持ち回りは scheme II の桁作りとも共有する slice_tf_core.h にある．
//
// 要素ごとの処理
//   t = (hi,mid,lo)           （正規化の 2 のべきはリーダが FP64 の指数へ畳んである）
//   i = 0..NSLICE-1 について：u = radix*t ; d = clamp(round(u0)) ; d を符号化して出力 ; t = u - d
//   基数倍は基数が 2 のべき（BF16・FP16・TF32・BFP8_b の 64，FP8 の 32，BFP2_b の 2）なら
//   tfMulP2（19 演算），そうでなければ（BFP4_b の 14・INT8 の 254）tfMulR（44 演算）で，本番の値域では結果は同じ
//   （sfpu_tf_proc.h の tfMulP2 の一致の条件）．
//   round が見るのは triple-float の上位 1 語（t0）だけである．
//
// 残差の不変条件：丸めも減算も 3 本組の最上位 t0 だけに当たり，下位 2 本は手を触れずにそのまま残る．d は t0 から
// 作った整数で |t0| < 2^24 のあいだ t0-d は無誤差なので，3 本組の値としての t はちょうど d だけ減る．
// したがって不変条件は最上位についての |t0| <= R である（R = d_max/(beta-1)．既定では beta = 2*d_max なので
// R = d_max/(2*d_max-1)，すなわち 1/2 をわずかに上回る）：|t0| <= R なら
// |beta*t0 - clamp(round(beta*t0))| <= max(1/2, beta*R - d_max) = R で，繰り返しても範囲から出ない（clamp が働くのは
// beta*|t0| が d_max+1/2 を超えるか，ちょうど等しくて外側へ丸められる端の場合だけで，そのときも残差は R 以内に留まる）．
// p 段後の誤差は R*beta^{-p} 以下に落ちる．これは t0 だけについての実数上の評価である．
// 下位 2 本は，丸めの時点（基数倍が 3 本組を TwoSum で正規化した直後）では t0 の ulp 程度に収まっているので，
// この評価を丸め誤差の範囲を超えて崩さない．t0 -= d の直後は t0 が打ち消されて下位 2 本が相対的に大きく見えることが
// あるが，次の段の基数倍（tfMulR・tfMulP2 のどちらも最後に TwoSum 鎖で正規化する）が再び正規化する．
// 入口で |t0| <= R を保証するのはホスト側である（R は表せる上限 d/(β-1)．T は正規化の目標で 2 冪）．
// 正規化後の値 x について，切り捨ての形式（BF16 など）では |x| < T(1+2^-7) <= R，R が T をほとんど上回らない
// 構成（INT8・BFP2_b）ではスケールを切り上げて |x| <= T にする（ozaki1_main.cpp の bf_ceil．BFP2_b は
// 最大値がちょうど 2 のべきでも 1 段上げて |x| < T にする．同 tau_above）．t0 は x を FP32 へ丸めた上位 limb
// なので x の上界に等しくなりうる（BFP2_b の x = 1-2^-53 では t0 = 1 = T）が，どの場合も |t0| <= R は保たれる．
//
// コンパイル時引数（並びは oz1_args.h の SlicerCT）: スライスの本数 NSLICE（L+1）, radix の float ビット列, リーダが soft-float 版か,
//   桁の上限の float ビット列, その負値, OUT_HALF（桁の値のまま出すときだけ効く）
// 実行時引数: ntiles（oz1_args.h の SlicerRT）
// DST（fp32_dest_acc で 8 本．ホストの slice_build が dst_full_sync_en を立てるので 8 本を 1 区画として使える．
//   半分同期のままだと 4 本ずつに割られる）．名前は sfpu_tf_proc.h の namespace dst：t の 3 本組 = acc_hi・acc_mid・acc_lo
//   （基数倍 tfMulRadix が更新する），切り出した桁 d = keep，符号化の作業 = term・tmp0・tmp1．tfMulRadix は term・tmp0..2・
//   keep を作業に使って壊すので，d は基数倍の後に作る

#include <cstdint>
#include "../../common/kernels/slice_tf_core.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "oz1_args.h"

constexpr auto cb_out = tt::CBIndex(oz1_args::CB_SLICE_OUT);

constexpr uint32_t NSLICE  = get_compile_time_arg_val(oz1_args::SC_NSLICE);  // スライスの本数（L+1）
constexpr uint32_t RADIX_B = get_compile_time_arg_val(oz1_args::SC_RADIX);   // radix の float ビット列
// 1 ならリーダが soft-float 版で，3 本組をそのまま寄こす（測定用．論文の比較）
constexpr uint32_t SF_IN   = get_compile_time_arg_val(oz1_args::SC_SF_IN);
constexpr uint32_t DMAX_B  = get_compile_time_arg_val(oz1_args::SC_DMAX);    // 桁の上限（+）
constexpr uint32_t NDMAX_B = get_compile_time_arg_val(oz1_args::SC_NDMAX);   // 桁の上限（-）
// 1 ならスライスを IEEE binary16 のビット列（UInt16）で出力する（FP16 経路．符号-絶対値のスライサは読まない）
constexpr uint32_t OUT_HALF = get_compile_time_arg_val(oz1_args::SC_OUT_HALF);
constexpr uint32_t DF_F32   = (uint32_t)DataFormat::Float32;
constexpr uint32_t DF_U16   = (uint32_t)DataFormat::UInt16;
constexpr uint32_t DF_U8    = (uint32_t)DataFormat::UInt8;
// 出力の桁を置く DST：符号-絶対値は term，桁の値のままなら keep（d そのもの）
constexpr uint32_t OUT_REG  = SLICE_SIGNMAG8 ? dst::term : dst::keep;

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(oz1_args::SR_NTILES);
    using namespace dst;

    for (uint32_t ti = 0; ti < ntiles; ti++) {
        tf_wait_input();
        for (uint32_t i = 0; i < NSLICE; i++) {
            tile_regs_acquire();
            init_sfpu(cb_tf_hi, cb_out);
            if (i == 0) {                   // t = FP64 から組んだ 3 本組（正規化込み）
                if constexpr (SF_IN) tf_load_plain(); else tf_load_f64();
            }
            else        tf_pop_residual();
            tfMulRadix<RADIX_B>();                   // t *= radix（2 のべきなら tfMulP2．sfpu_tf.h）
            Cpy(acc_hi, keep);                       // スライス値 d = t の上位
            rounding_op_tile_init(); round_tile(keep, 0);
            clamp_tile_init();       clamp_tile(keep, NDMAX_B, DMAX_B);   // d = clamp(round(t0), ±桁上限)
            Sub(acc_hi, keep, acc_hi);               // t0 -= d（keep の d は出力用に残す．上の「残差の不変条件」）
            if constexpr (SLICE_SIGNMAG8) {
                // INT8 経路：d を符号-絶対値へ符号化して term に置く．使うのは term・tmp0・tmp1 だけなので，
                // 更新したばかりの残差（acc_hi・acc_mid・acc_lo）を壊さない．
                Cpy(keep, term); SignMag8(term, tmp0, tmp1);   // term = g(d)（sfpu_tf.h）
                typecast_tile_init<DF_F32, DF_U8>();
                typecast_tile<DF_F32, DF_U8>(term);      // term = (uint8) g(d)
            } else if constexpr (OUT_HALF) {
                // FP16 経路：d を binary16 のビット列に等しい整数値にし（HalfBits．作業に term・tmp0・tmp1），
                // それを UInt16 に型変換する．パッカは UInt16 の CB へそのまま書き，GEMM がそのバイト列を
                // Float16 として読む．
                HalfBits(keep, term, tmp0, tmp1);
                typecast_tile_init<DF_F32, DF_U16>();
                typecast_tile<DF_F32, DF_U16>(keep);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(OUT_REG, cb_out);              // スライス i を出力（cb_out の形式へ変換される）
            cb_push_back(cb_out, 1);
            if (i + 1 < NSLICE) tf_push_residual();   // 次のスライスのために残差を退避
            tile_regs_release();
        }
        tf_pop_input();
    }
}
