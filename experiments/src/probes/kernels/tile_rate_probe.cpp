// SPDX-License-Identifier: Apache-2.0
//
// tile_rate_probe の計算カーネル：L1 に置いた 1 ブロックの行列積を NoC 転送なしに reps 回繰り返し，計時の壁時計を L1 に書く．
// 引数の並びと結果の欄の語の番号は testkit/kernels/probe_args.h の tile_rate_args（ホストの tile_rate_probe.cpp と共有）．
// CT 引数：bw，pm，pn，sh，sw，MODE（tile_rate_args::Mode の番号）
//   MODE 0  matmul_block（アンパッカの展開＋行列エンジンの積）
//   MODE 1  行列エンジンだけ（アンパッカは SrcA・SrcB の有効の印だけを立てる）
//   MODE 2  アンパッカだけ（行列エンジンは有効の印を待ってバンクを返すだけ）
//   MODE 3  MODE 0 に出力の処理を足す：サブブロックごとに K 方向 bw の積の後で DST を c_16 へパックし，2 回目以降の
//           繰り返しはパッカの L1 累積で同じ番地へ足す．c_16 は出力ブロック全体（pm x pn タイル）の大きさで，繰り返しの
//           終わりに取り出すので，次の繰り返しも同じ番地へ書く．
//           尾崎の GEMM 段の計算カーネル（common/kernels/bmm_int32_spill.cpp の BLOCKMM の枝）の内側のループ（サブブロック
//           ごとに K 方向の matmul_block を回してパックする）を書き直したもので，本番を呼ぶのではない．本番との違い：
//             ・入力は 1 回だけ L1 に置いて reps 回使い回す（本番は K ブロックごとにリーダが CB を入れ替え，cb_wait_front と
//               cb_pop_front が毎ブロック入る）
//             ・部分和の累積は繰り返しの 2 回目以降をすべて L1 累積で足す（本番は K の区間の 2 個目以降のブロックを c_24 へ
//               L1 累積で足し，区間の最後のブロックは c_16（対ループ・融合のときは c_3）へ出す．L1 累積の入り切りと
//               pack_reconfig_data_format は区間の境で行う）
//             ・本番の対ループ（PAIRLOOP．対の境で FP32 を Int32 に直して c_5 へ足す第 2 パス），出力経路への融合（FUSE．
//               ライタが先読みした DRAM の S と足す第 2 パス），c_24 からの読み戻し（enable_reload）は無い
//             ・DST の受け渡しは tile_regs_*（本番は acquire_dst・release_dst）
//           したがって値は「本番の内側のループの命令列の 1 コアの速さ」であり，本番の計算カーネルを直してもこの探針は
//           追随しない（本番を直したら，ここの内側のループと上の違いの一覧も合わせる）
// RT 引数：reps，結果の欄の L1 番地（RES_T0 = UNPACK が入力を受け取った時点の壁時計，RES_T1 = PACK が最後の
//          サブブロックを受け取った時点の壁時計，RES_DONE = PACK が終わりに書く 1）
// MODE 0--2 は出力をパックしない（PACK スレッドは DST の受け渡しの同期だけを行う）．パックの時間を含めると形式ごとの
// タイル積の比較にパッカの書き出しが混ざるため．
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "../../testkit/kernels/probe_args.h"
using namespace tile_rate_args;

constexpr uint32_t bw = get_compile_time_arg_val(C_BW), pm = get_compile_time_arg_val(C_PM),
                   pn = get_compile_time_arg_val(C_PN), sh = get_compile_time_arg_val(C_SH),
                   sw = get_compile_time_arg_val(C_SW), MODE = get_compile_time_arg_val(C_MODE);

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
// コアの壁時計（下位 32 bit）．コアのクロックで進むので，差はそのままデバイスのサイクル数になる
#define TS() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#endif

#ifdef TRISC_UNPACK
// MODE 1：実際の展開の代わりに，行列エンジンが待つ SrcA・SrcB の有効の印だけを立てる（matmul_block と同じ順序・回数）
inline void dummy_dvalid() {
    constexpr bool reuse_a = sw >= sh;
    constexpr uint32_t t_dim = reuse_a ? sh : sw, r_dim = reuse_a ? sw : sh;
    for (uint32_t t = 0; t < t_dim; t++) {
        if (reuse_a) {
            TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
            for (uint32_t u = 0; u < r_dim; u++)
                TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        } else {
            TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
            for (uint32_t u = 0; u < r_dim; u++)
                TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        }
    }
}
#endif
#ifdef TRISC_MATH
// MODE 2：積 1 回ごとに MVMUL を 1 つ出して SrcA・SrcB の有効を待ち，流したほうのバンクを返す（積の値は使わない）
inline void drain() {
    constexpr bool reuse_a = sw >= sh;
    constexpr uint32_t t_dim = reuse_a ? sh : sw, r_dim = reuse_a ? sw : sh;
    for (uint32_t t = 0; t < t_dim; t++) {
        for (uint32_t u = 0; u < r_dim; u++) {
            if (reuse_a) TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0);
            else TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
        }
        if (reuse_a) TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        else TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
}
#endif

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(R_REPS);
    const uint32_t res_addr = get_arg_val<uint32_t>(R_RES_ADDR);
    volatile uint32_t* res = reinterpret_cast<volatile uint32_t*>(res_addr);
    constexpr auto cb0 = tt::CBIndex::c_0, cb1 = tt::CBIndex::c_1, cbo = tt::CBIndex::c_16;
    mm_block_init(cb0, cb1, cbo, 0, sw, sh, bw);
    cb_wait_front(cb0, pm * bw);
    cb_wait_front(cb1, bw * pn);
    UNPACK(({ res[RES_T0] = TS(); }));
    for (uint32_t r = 0; r < reps; r++) {
        if constexpr (MODE == MODE_MMPACK) {
            if (r == 1) pack_reconfig_l1_acc(1);   // 2 回目からは前の部分和へ足す（GEMM 段の K ブロックの 2 個目以降と同じ）
        }
        for (uint32_t i0 = 0; i0 < pm / sh; i0++) {
            for (uint32_t i1 = 0; i1 < pn / sw; i1++) {
                tile_regs_acquire();
                uint32_t a = i0 * sh * bw, b = i1 * sw;
                for (uint32_t k = 0; k < bw; k++) {
                    if constexpr (MODE == MODE_MM || MODE == MODE_MMPACK) {
                        matmul_block(cb0, cb1, a, b, 0, 0, sw, sh, bw);
                    } else if constexpr (MODE == MODE_MATH) {
                        UNPACK((dummy_dvalid()));
                        MATH((llk_math_matmul<MATH_FIDELITY, 0>(0, sw, sh)));
                    } else {
                        UNPACK((llk_unpack_AB_matmul(cb0, cb1, a, b, sw, sh, bw)));
                        MATH((drain()));
                    }
                    a++;
                    b += pn;
                }
                tile_regs_commit();
                tile_regs_wait();
                if constexpr (MODE == MODE_MMPACK) {
                    cb_reserve_back(cbo, sh * sw);
                    for (uint32_t t = 0; t < sh * sw; t++) pack_tile(t, cbo);
                    cb_push_back(cbo, sh * sw);
                }
                tile_regs_release();
            }
        }
        if constexpr (MODE == MODE_MMPACK) {   // 出力ブロックを取り出し，次の繰り返しも同じ番地へ書かせる
            cb_wait_front(cbo, pm * pn);
            cb_pop_front(cbo, pm * pn);
        }
    }
    PACK(({ res[RES_T1] = TS(); res[RES_DONE] = 1; }));
    if constexpr (MODE == MODE_MMPACK) {
        if (reps > 1) pack_reconfig_l1_acc(0);
    }
}
