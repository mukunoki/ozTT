// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BF16 extended と 3xTF32（gemm_bf16ext.cpp）用の計算カーネル
// ---------------------------------------------------------------------------
// tt-metal 本番実装の PACKER_L1_ACC パターン（ttnn の
// bmm_large_block_zm_fused_bias_activation の非 FUSE_BIAS 経路）を，
// 単純な matmul_tiles サンプルの構造へ忠実に移植したもの（基準実装．論文の本文には BF16 extended の節は無く，
// スライドの第 2 部の浮動小数点分割の頁が引く）．
//
// 何が問題になるか
//   K 方向のブロック間の総和を素直に「中間 CB から DST へ読み戻して足す」形で
//   書くと，読み戻しの経路で TF32（仮数約 10 bit）へ切り詰められ，誤差が K と
//   ともに増大してしまう．
//
// どう回避しているか
//   ブロック間の累積を，DST 経由ではなく「パッカが L1 上で FP32 加算する」機構
//   （pack_reconfig_l1_acc）に任せる．中間 CB c_24 へ
//     ブロック 0        : l1_acc=0（上書き）
//     ブロック 1 以降   : l1_acc=1（L1 上で加算）
//   と pack していき，累積された部分和を DST へ読み戻すのは最後のブロックの
//   ただ 1 回だけにする（copy_block_matmul_partials．ホストが c_24 に
//   UnpackToDestFp32 を設定しているので FP32 のまま戻る）．最後のブロックの
//   行列積をその上に足し込み，結果を出力 CB c_16 へ pack する．
//
//   c_24 の容量はコアあたり出力ブロック 1 つぶんなので，下の push/pop の
//   やりくりで同じ L1 領域を使い回し，ブロック k の pack がブロック <k の
//   残り和の上に足されるようにしている．
//
// カーネルは実行時 JIT ビルドなので，変更しても tt-metal 本体の再ビルドは不要．

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "gemm_compute_args.h"   // 引数の並び（標準の 0〜11 番）と CB 番号（ホストの common/mcast_gemm.hpp と共有）
using namespace gemm_args;

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(CA_IN0_BLOCK_W);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(CA_IN0_NUM_SUBBLOCKS);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(CA_IN0_BLOCK_NUM_TILES);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(CA_IN0_SUBBLOCK_NUM_TILES);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(CA_IN1_NUM_SUBBLOCKS);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(CA_IN1_BLOCK_NUM_TILES);
    uint32_t in1_per_core_w = get_compile_time_arg_val(CA_IN1_PER_CORE_W);
    uint32_t num_blocks = get_compile_time_arg_val(CA_NUM_BLOCKS);
    uint32_t out_subblock_h = get_compile_time_arg_val(CA_OUT_SUBBLOCK_H);
    uint32_t out_subblock_w = get_compile_time_arg_val(CA_OUT_SUBBLOCK_W);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(CA_OUT_SUBBLOCK_NUM_TILES);
    uint32_t batch = get_compile_time_arg_val(CA_BATCH);

    constexpr auto cb_in0 = tt::CBIndex(CB_IN0);
    constexpr auto cb_in1 = tt::CBIndex(CB_IN1);
    constexpr auto cb_out = tt::CBIndex(CB_OUT);
    constexpr auto cb_interm = tt::CBIndex(CB_PARTIAL);

    uint32_t out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t bch = 0; bch < batch; bch++) {
        bool enable_reload = false;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            cb_wait_front(cb_in0, in0_block_num_tiles);
            cb_wait_front(cb_in1, in1_block_num_tiles);

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    if (enable_reload) {
                        // 最後のブロックのときだけ，L1 に貯めた部分和を FP32 のまま
                        // DST へ読み戻す（c_24 に UnpackToDestFp32 を指定してあるので FP32 のまま戻る）
                        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_interm);
                        cb_wait_front(cb_interm, out_subblock_num_tiles);
                        copy_block_matmul_partials(cb_interm, 0, 0, out_subblock_num_tiles);
                        cb_pop_front(cb_interm, out_subblock_num_tiles);
                        mm_init_short_with_dt(cb_in0, cb_in1, cb_interm);
                    }

                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    tile_regs_commit();

                    // パッカの設定はブロックの最初のサブブロックでだけ変える．
                    // pack_reconfig_data_format は形式を書き直すときに，L1 累積のあいだは立てておく
                    // べき「全零の面に印を付けない」設定（Disable_pack_zero_flags．LLK の
                    // pack_reconfig_l1_acc(1) が立てる）も既定へ戻してしまう．サブブロックごとに形式を
                    // 再設定し，L1 累積の設定をブロック 1 でだけ行うと，ブロック 2 以降は L1 累積がこの設定を
                    // 戻したまま走りうる．
                    // 形式は同じ出力先へ出し続けるあいだ変わらないので，再設定はブロックの頭の 1 回で足り，
                    // L1 累積の設定はその後で行う．これは公開 API の推奨する順序に合わせたもので，
                    // 誤答を防いでいるとは実証していない：サブブロックごとに再設定する形のカーネルに零の面・
                    // 零の行・零のブロックを混ぜた入力（gemm_exact_check の zface）を FP32 の中間で流しても
                    // 食い違いは観測されなかった（docs/gemm_opt_results.md の「零の面の印」）．
                    const bool first_sb = in0_subblock == 0 && in1_subblock == 0;
                    if (last_out) {
                        cb_reserve_back(cb_out, out_subblock_num_tiles);
                        tile_regs_wait();
                        if (first_sb) {
                            pack_reconfig_data_format(cb_out);
                            pack_reconfig_l1_acc(0);
                        }
                        pack_tile_block(0, cb_out, out_subblock_num_tiles);
                        tile_regs_release();
                        cb_push_back(cb_out, out_subblock_num_tiles);
                    } else {
                        cb_reserve_back(cb_interm, out_subblock_num_tiles);
                        tile_regs_wait();
                        // ブロック 0 は上書き，ブロック 1 以降は L1 上で加算．形式の再設定のあとで
                        // L1 累積を設定する（上の注記）．ブロック 2 以降は設定を変えない．
                        if (first_sb && block == 0) {
                            pack_reconfig_data_format(cb_interm);
                            pack_reconfig_l1_acc(0);
                        } else if (first_sb && block == 1) {
                            pack_reconfig_l1_acc(1);
                        }
                        pack_tile_block(0, cb_interm, out_subblock_num_tiles);
                        tile_regs_release();
                        cb_push_back(cb_interm, out_subblock_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (!last_out) {
                // c_24 の物理 L1 領域を使い回し，次のブロックの l1_acc pack が
                // 走行中の和の上に足されるようにする．最後のブロックで読み戻す
                // ため，ブロック num_blocks-2 の累積結果だけは残しておく．
                if (block < num_blocks - 2) {
                    for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                        cb_wait_front(cb_interm, out_subblock_num_tiles);
                        cb_pop_front(cb_interm, out_subblock_num_tiles);
                    }
                }
                if (block == num_blocks - 2) {
                    enable_reload = true;
                }
            }

            cb_pop_front(cb_in0, in0_block_num_tiles);
            cb_pop_front(cb_in1, in1_block_num_tiles);
        }
        pack_reconfig_l1_acc(0);  // 次のバッチのために戻す
    }
}
