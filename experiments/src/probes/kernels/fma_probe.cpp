// SPDX-License-Identifier: Apache-2.0
//
// SFPU に FMA があるかを実測で判定する．FMA は IEEE 754 の fusedMultiplyAdd，すなわち
// a*b+c を無限の精度と指数範囲で計算してから 1 度だけ丸める演算である．2 つの FP32 の積は
// 厳密には 48 bit あるので，これを満たすには積を丸めずに保たなければならない．
// 判定式は古典的なもので，a = b = 1+2^-23 とし c = -(1+2^-22) を足す（ホストの src/probes/fma_probe.cpp は
// これを一般化し，積の先頭から t bit 下に 1 が立つ a,b を t=20..46 の 1 bit 刻みで作って保持幅を測る）．
//   厳密な積   a*b = 1 + 2^-22 + 2^-46
//   FP32 の丸め round(a*b) = 1 + 2^-22（2^-46 は半 ulp 2^-24 より小さい）
// したがって
//   融合（積を丸めない）なら  a*b + c = 2^-46 = 1.42e-14
//   非融合（積を丸める）なら  round(a*b) + c = 0
// である．DST に a,b,c を置き，0 番と 1 番の 2 通りの書き方で同じ式を計算して読み比べる．
//   0 番  addcmul_tile（SFPMAD を 1 本使うタイル API．api/compute/eltwise_unary/addcmul.h）
//   1 番  Mul してから Add（本実装が使っている書き方）
//   2 番  積 a*b だけ（c は足さない．経路自体が壊れていないかの確認用．ホストの src/probes/fma_probe.cpp は
//         倍率ごとの掃引の前に，積が FP32 に収まる入力で 1 回流す）
// 番号と引数の並びは testkit/kernels/probe_args.h の fma_probe_args（コンパイル時引数 C_MODE = 上の番号 Mode，
// 実行時引数 R_C = 足す定数 c の float のビット列）
#include <cstdint>
#include "../../testkit/kernels/probe_args.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    using namespace fma_probe_args;
    constexpr uint32_t mode = get_compile_time_arg_val(C_MODE);   // Mode（MODE_ADDCMUL・MODE_MULADD・MODE_MUL）
    static_assert(mode < MODE_COUNT, "C_MODE は fma_probe_args::Mode");
    const uint32_t c_rt = get_arg_val<uint32_t>(R_C);             // 足す定数（float のビット列）
    constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_o = tt::CBIndex::c_16;

    unary_op_init_common(cb_a, cb_o);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_a); copy_tile(cb_a, 0, 0);   // DST0 = a
    copy_tile_to_dst_init_short(cb_b); copy_tile(cb_b, 0, 1);   // DST1 = b
    // DST2 = c：a を 0 倍してから定数を足す（ホストから 3 本目のタイルを送らずに作る）
    copy_tile_to_dst_init_short(cb_a); copy_tile(cb_a, 0, 2);
    binop_with_scalar_tile_init(); mul_unary_tile(2, 0x00000000u);
    binop_with_scalar_tile_init(); add_unary_tile(2, c_rt);

    if constexpr (mode == MODE_ADDCMUL) {
        // DST3 = DST2 + 1.0 * DST0 * DST1（SFPMAD 1 本）
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float32>(2, 0, 1, 3, 0x3F800000u);
    } else if constexpr (mode == MODE_MULADD) {
        // DST3 = DST0 * DST1 を作ってから DST2 を足す（積がここで丸められる）
        mul_binary_tile_init(); mul_binary_tile(0, 1, 3);
        add_binary_tile_init(); add_binary_tile(3, 2, 3);
    } else {
        // 確認用：積 a*b だけを出す（c は足さない）．積を丸めずに済む値を渡せば，
        // 読み込みから書き出しまでの経路自体が壊れていないことを見られる
        mul_binary_tile_init(); mul_binary_tile(0, 1, 3);
    }

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_o, 1);
    pack_reconfig_data_format(cb_o);
    pack_tile(3, cb_o);
    cb_push_back(cb_o, 1);
    tile_regs_release();
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
