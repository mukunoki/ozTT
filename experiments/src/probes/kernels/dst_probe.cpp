// SPDX-License-Identifier: Apache-2.0
//
// DST レジスタが FP32 で何本使えるかを実機で確かめる検査カーネル
// ---------------------------------------------------------------------------
// 入力タイル 1 枚（全要素 1.0）を DST の各レジスタへ写し，繰り返し t でレジスタ i を
// t*100+(i+1) 倍してそのまま NREG 枚として書き出す．レジスタが本当に独立なら出力タイル i は
// 全要素 t*100+i+1 になる．本数が足りずに折り返していれば，どこかで値が食い違う．
//
// tensix_types.h の BIT32_DEST_REGISTER_HALF_SIZE（= 4 タイル相当）は半分同期（half-sync）
// で 1 回の acquire に使える FP32 の本数で，dst_full_sync_en を立てれば DST 全体（その 2 倍）
// を使えるはずである．本リポジトリの再構成カーネルは 8 本を使う．半分同期と全体同期の
// それぞれで何本まで独立に使えるかを，この検査で確かめる．
//
// NT 枚ぶん繰り返すのは，dst_full_sync_en を立てない（半分同期の）ときに DST の
// 2 つ目の半分を通すためである．半分同期では tile_regs_acquire のたびに使う半分が
// 入れ替わるので，1 回だけでは先頭の半分しか試せない．繰り返しの t 回目の値は
// t*100 + (i+1) とし，前の回の値が混ざれば分かるようにする．
//
// コンパイル時引数は testkit/kernels/probe_args.h の dst_probe_args（C_NREG = 試すレジスタ本数，C_NT = 繰り返す回数）

#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "../../testkit/kernels/probe_args.h"
#include "api/compute/pack.h"

constexpr auto cb_in  = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_16;
constexpr uint32_t NREG = get_compile_time_arg_val(dst_probe_args::C_NREG);
constexpr uint32_t NT   = get_compile_time_arg_val(dst_probe_args::C_NT);

void kernel_main() {
    cb_wait_front(cb_in, 1);
    init_sfpu(cb_in, cb_out);
    for (uint32_t t = 0; t < NT; t++) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_in);
        // まず全レジスタへ 1.0 を配り，それぞれを t*100+(i+1) 倍する．
        // 書き込みをすべて済ませてから読み出すので，折り返しがあれば必ず値が壊れる．
        for (uint32_t i = 0; i < NREG; i++) copy_tile(cb_in, 0, i);
        for (uint32_t i = 0; i < NREG; i++) MulS(i, (float)(t * 100 + i + 1));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, NREG);
        pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < NREG; i++) pack_tile(i, cb_out);
        cb_push_back(cb_out, NREG);
        tile_regs_release();
    }
    cb_pop_front(cb_in, 1);
}
