// SPDX-License-Identifier: Apache-2.0
//
// SFPU の模擬（checks/sfpu_tf_emu.hpp）と実機を突き合わせる検査カーネル（ホストは checks/tfmul_probe.cpp）
// ---------------------------------------------------------------------------
// 本番の関数（common/kernels/sfpu_tf.h のタイル演算のラッパ・tfMulRadix・tfMulR・tfAddS，slice_tf_core.h の
// tf_load_f64）をそのまま呼び，DST の結果をパックして返す．ホストが同じ入力を模擬に通してビット単位で比べる．
// 引数と演算の番号は testkit/kernels/probe_args.h の tfmul_probe_args（コンパイル時引数 C_OP・C_RDX，実行時引数 R_NGROUPS = 組の数）．
// 入力は c_0 に 1 組 nin 枚（OP_LOAD だけはスライサと同じ c_0・c_4・c_1・c_2 に 1 枚ずつ），出力は c_16 へ 1 組
// nvar 回 × nout 枚．作業に DST を 8 本使うので dst_full_sync_en が要る．
#include <cstdint>
#include "../../common/kernels/slice_tf_core.h"   // sfpu_tf.h・f64_tf.h・pack.h も取り込む
#include "../../testkit/kernels/probe_args.h"
using namespace tfmul_probe_args;

constexpr auto cb_in  = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_16;
constexpr uint32_t OP  = get_compile_time_arg_val(C_OP);
constexpr uint32_t RDX = get_compile_time_arg_val(C_RDX);
static_assert(OP < OP_COUNT, "OP は tfmul_probe_args::Op");
constexpr uint32_t NIN = nin(OP), NOUT = nout(OP), NVAR = nvar(OP);

ALWI void apply(uint32_t v) {
    if constexpr (OP == OP_ARITH) {
        Add(0, 1, 2); Sub(0, 1, 3); Mul(0, 1, 4);
    } else if constexpr (OP == OP_RND) {
        Rnd(0);
    } else if constexpr (OP == OP_MUL) {
        if (v == 0) tfMulRadix<RDX>();   // 基数が 2 のべきなら tfMulP2（本番の振り分けそのもの）
        else        tfMulR(RDX);
    } else if constexpr (OP == OP_ADDS3) {
        tfAddS<3>();
    }
}

ALWI void pack_out() {
    cb_reserve_back(cb_out, NOUT);
    pack_reconfig_data_format(cb_out);
    for (uint32_t k = 0; k < NOUT; ++k) pack_tile(out_reg(OP, k), cb_out);
    cb_push_back(cb_out, NOUT);
}

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(R_NGROUPS);
    init_sfpu(cb_in, cb_out);
    for (uint32_t t = 0; t < n; ++t) {
        if constexpr (OP == OP_LOAD) {
            tf_wait_input();
            tile_regs_acquire();
            tf_load_f64();
            tile_regs_commit();
            tile_regs_wait();
            pack_out();
            tile_regs_release();
            tf_pop_input();
        } else {
            cb_wait_front(cb_in, NIN);
            for (uint32_t v = 0; v < NVAR; ++v) {   // 変種ごとに同じ入力を写し直す
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_in);
                for (uint32_t i = 0; i < NIN; ++i) copy_tile(cb_in, i, i);
                apply(v);
                tile_regs_commit();
                tile_regs_wait();
                pack_out();
                tile_regs_release();
            }
            cb_pop_front(cb_in, NIN);
        }
    }
}
