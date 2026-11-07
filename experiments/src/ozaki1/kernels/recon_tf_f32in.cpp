// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：デバイス上の再構成（FP32 の部分積を読む非遅延の経路，triple-float）
// ---------------------------------------------------------------------------
// 既定（PAIRLOOP・融合）は遅延再構成で Int32 の累積面を読む recon_tf_i32in.cpp を使い，既定の 16 経路の構成はここを
// 通らない．こちらを通るのは遅延再構成を使わない経路だけである：K ブロック 1 本で対ごとにも PAIRLOOP でも回さない
// FP 形式の構成（ozaki1_main.cpp の run_single_kblock．OZ_NO_PAIRLOOP・OZ_NO_FUSE_ACC・OZ_NO_DEFER で連結の K ブロックが
// 1 本になるとき）と，K ブロックが複数で遅延再構成が取れない構成（OZ_NO_DEFER，Int32 の累積が 2^31 を超えうる長い K
// （oz_defer_fits），OZ_NO_FUSE_ACC で累積面が DRAM に入らないとき．K ブロックごとに ACC で足し継ぐ）．
// 役割：対角ごとの部分和 P_d を重み付きで足し合わせ，元の高精度な積を復元する．
//       論文の ssec:nofp64（FP32 を用いた再構成と適用範囲）．
//
//   S = Σ_d radix^(L-d) P_d = (((P_0)*radix + P_1)*radix + ...) + P_L
//
//   この Horner 累積を FP32 3 本組（triple-float，作業精度約 72 bit）で行う．
//   FP32 1 本では約 24 bit，2 本でも約 48 bit で頭打ちになるが，3 本なら
//   FP64 相当の約 53 bit に届く．出力は S を FP32 3 枚（上位・中位・下位）で，
//   後段の recon_f64_writer.cpp が R = (hi+mid+lo) * radix^-(L+2) * s_A * s_B を
//   FP64 へ組み立てる（デバイス上．ホストは並べ替えるだけである．s_A・s_B は行・列の正規化に実際に用いた
//   除数 2^-dA・2^-dB で，定義は recon_f64_writer.cpp の冒頭）．
//
// Int32 入力版（recon_tf_i32in.cpp．遅延再構成と INT8）との違い
//   INT8 以外の 7 形式の P_d は FP32 の無誤差整数（2^24 以下）なので，2 本組への
//   復号は不要で，そのままスカラとしてアキュムレータに足せる．
//
// コンパイル時引数（並びは oz1_args.h の ReconCT）: ndiag(L+1), 部分積の形式値, Float32 の形式値,
//                   radix の float ビット列, ACC（0 なら初期化，1 なら累積）
//                   （Int32 入力版と共通の並び．Int32 入力版は ACC を読まず，こちらは SIGNMAG・NSINGLE_CAST を読まない）
// 実行時引数: ntiles（oz1_args.h の ReconRT）
// DST（fp32_dest_acc で 8 本．ホストの recon_build_prog が dst_full_sync_en を立てるので
//   8 本を 1 区画として使える．半分同期のままだと 4..7 番が次の区画と重なる）．名前は sfpu_tf_proc.h の namespace dst：
//   アキュムレータの 3 本組 = acc_hi・acc_mid・acc_lo，足す P_d は term（残りは tfMulRadix・tfAddS の作業用）

#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "recon_pack.h"
#include "oz1_args.h"
using namespace oz1_args;

constexpr uint32_t C_RDX = get_compile_time_arg_val(RC_RADIX);   // radix の float ビット列
// ACC=1：K ブロック化のとき，前ブロックまでの累積（hi,mid,lo）を CB_ACC から読んで足す．
// Horner は P_d について線形なので，ブロックごとに畳んでから足し合わせてよい．
constexpr uint32_t ACC = get_compile_time_arg_val(RC_ACC);

constexpr auto cb_in  = tt::CBIndex(CB_RECON_IN);   // 入力：対角ごとの部分和 P_d（FP32）
constexpr auto cb_acc = tt::CBIndex(CB_ACC);        // 入力：前ブロックまでの累積（3 タイル）
constexpr auto cb_hi  = tt::CBIndex(CB_LIMB_HI);

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(RCR_NTILES);
    constexpr uint32_t ndiag = get_compile_time_arg_val(RC_NDIAG);  // L+1

    for (uint32_t t = 0; t < ntiles; t++) {
        cb_wait_front(cb_in, ndiag);
        init_sfpu(cb_in, cb_hi);
        tile_regs_acquire();

        using namespace dst;
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 0, acc_hi);       // P_0（Horner の初項）
        Sub(acc_hi, acc_hi, acc_mid); Sub(acc_hi, acc_hi, acc_lo);   // アキュムレータを (P_0, 0, 0) に初期化

        for (uint32_t d = 1; d < ndiag; d++) {
            tfMulRadix<C_RDX>();                 // Horner の 1 段：acc *= radix（2 のべきなら tfMulP2．sfpu_tf.h）
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, d, term);      // P_d
            tfAddS<3>();                       // acc += P_d（P_d は 2^24 以下の無誤差整数）
        }
        if constexpr (ACC) {
            // 前ブロックまでの 3 本組を下位から順に足し込む
            cb_wait_front(cb_acc, 3);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 2, term); tfAddS<3>();   // lo
            copy_tile(cb_acc, 1, term); tfAddS<3>();   // mid
            copy_tile(cb_acc, 0, term); tfAddS<3>();   // hi
            cb_pop_front(cb_acc, 3);
        }

        recon_pack_triple();   // アキュムレータの 3 本組をライタへ（recon_pack.h）

        cb_pop_front(cb_in, ndiag);
    }
}
