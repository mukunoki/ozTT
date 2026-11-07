// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：対角ごとの部分和を Int32 で K 方向へ累積する（計算側）
// ---------------------------------------------------------------------------
// 役割：S_d <- S_d + P_d を無誤差に行う．これで再構成（triple-float の Horner）を
//       K ブロックごとに繰り返す必要がなくなり，最後に 1 回だけで済む．
//
// 無誤差である理由：1 ブロック内の |P_d| は無誤差条件そのものから 2^24 以下で，
// FP32 に厳密な整数として乗っている．全ブロックを足しても nkb * K_c = K なので
//   |S_d| <= (L+1) * K * dmax^2   （dmax は桁の上限で基数ではない．BF16 は基数 64 に対して 32）
// であり，BF16 経路（L+1=10, K=14080, dmax=32）なら 1.44e8 < 2^31 に収まる
// （対角を連結するか対ごとに発行するかに依らない）．INT8 経路は K ブロックが常に
// 1 個で，この累積を通らない（ozaki1_plan.hpp の issue_candidates・plan_issue が !OZ1_INT32_PARTIAL を条件にして，
// 遅延再構成の対象から外している）．
// したがって Int32 1 語で足り，累積面は Pbig と同じ ndiag 面のままでよい．
//
// Int32 の表現に注意：Blackhole の DST 上の Int32 は経路で表現が違う．
//   ・行列エンジンがパックした Int32 を copy_tile で載せる -> 符号-絶対値
//     （INT8 経路の復号は recon_tf_i32in.cpp を SIGNMAG=1 で動かし，マスク 0x7FFFFF00 と
//       符号復元を使う）
//   ・typecast<Float32,Int32> が作る            -> 2 の補数
//     （LLK の calculate_typecast_fp32_to_int32 が明示的に 2 の補数へ落とす）
// このカーネルは後者であり，add_int_tile<Int32> も Blackhole では 2 の補数を前提に
// している（SIGN_MAGNITUDE_FORMAT=false）ので，両者は整合している．
// したがって最終段の復号は recon_tf_i32in.cpp を SIGNMAG=0 で動かさねばならない．
// 符号-絶対値版を流用すると負の値が壊れる（相対誤差 1e4 の桁で落ちる）．
//
// コンパイル時引数（並びは oz1_args.h の AccumCT）: ACC（0 なら初期化，1 なら累積），Float32 の形式値，
//                   Int32 の形式値
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/pack.h"
#include "oz1_args.h"
using namespace oz1_args;

constexpr uint32_t ACC = get_compile_time_arg_val(AC_ACC);
constexpr uint32_t F32 = get_compile_time_arg_val(AC_F32);
constexpr uint32_t I32 = get_compile_time_arg_val(AC_I32);

constexpr auto cb_p   = tt::CBIndex(CB_ACCUM_P);     // 入力：P_d（FP32）
constexpr auto cb_s   = tt::CBIndex(CB_ACCUM_S);     // 入力：前ブロックまでの S_d（Int32）
constexpr auto cb_out = tt::CBIndex(CB_ACCUM_OUT);   // 出力：S_d（Int32）

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(ACR_NTILES);
    init_sfpu(cb_p, cb_out);

    for (uint32_t t = 0; t < ntiles; t++) {
        cb_wait_front(cb_p, 1);
        if constexpr (ACC) cb_wait_front(cb_s, 1);
        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_p);
        copy_tile(cb_p, 0, 0);
        typecast_tile_init<F32, I32>();
        typecast_tile<F32, I32>(0);
        if constexpr (ACC) {
            copy_tile_to_dst_init_short(cb_s);
            copy_tile(cb_s, 0, 1);
            add_int_tile_init();
            add_int_tile<DataFormat::Int32>(0, 1, 0);
        }

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_p, 1);
        if constexpr (ACC) cb_pop_front(cb_s, 1);
    }
}
