// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II（桁分割の 7 経路）：桁積を合成して法剰余を取る段（デバイス側）
// ---------------------------------------------------------------------------
// 役割：法 1 本ぶんの GEMM の出力 np 枚（同じ位の桁積を K 方向に連結した後の本数．2 桁分割で 2〜4 枚，
//       平衡 n 桁で n 枚から n^2 枚．塊に切る構成では塊 1 つぶんか，塊ごとの部分和）から，再構成が要る
//       剰余 r ≡ D (mod m) を作る．1 桁の法（桁 32 の AWE 経路）はこの段を通らない（再構成が法へ畳む）．
//       ここで計算するのは，出力 j（位 a+b が同じ桁積 g 本の和．ozaki2_split.h の oz2_product_groups の gr[j]）
//       にその位の係数を掛けた和
//           D ≡ Σ_j gr[j].coeff_lo * P_j   (mod m)
//       である．連結する前の桁積（oz2_product_terms の並び．XY は A の桁行列 X と B の桁行列 Y の積で，
//       L・H・S は下位・上位・和の桁行列）で書くと，
//       桁分割の方式ごとに次の形になる（q は桁の基数，B は Karatsuba の基数．ozaki2_split.h の OZ2_SPLIT_*）．
//           SQ・SQS : D ≡ q*(HL + LH) + LL                   （q^2 ≡ 0 なので HH の項は無い）
//           KAR     : D ≡ (1-B)*LL + (B^2-B)*HH + B*SS       （B=16 なら -15, 240, 16．BFP4_b の B=8 なら -7, 56, 8）
//           WIDE    : D ≡ q^2*HH + q*(HL + LH) + LL
//           BAL     : D ≡ Σ_{i,j} q^(i+j) P_ij
//       位が同じ桁積は係数も同じなので，連結した 1 本の出力（SQ・WIDE の HL と LH，BAL の i+j が同じ P_ij）には
//       その位の係数を 1 回掛ければよい．
//
// なぜ D を作らずに済むのか
//   桁の上限を d とすると |D| は WIDE で K*d^2*(q+1)^2 に達し（FP8 の d=16・K=14080 で 4.2e9），
//   FP32 の無誤差整数 2^24 を超える．しかし要るのは D そのものではなく D mod m だけで，
//   合成は法について線形なので，桁積を法へ畳んでから係数を掛ければよい．
//
// 合成の方式は 2 つあり，ホストが実行時引数 CP_FOLD_FIRST で選ぶ（ozaki2_moduli.h の fold_first）．
//   fold_first=1：桁積を先に ModRound で法へ畳んでから生の係数を当てる．
//                 |P_j|+m <= 2^24 なら商と法の積（|P_j|+m/2 強）が無誤差なので剰余類が
//                 保たれ，畳んだあとは |t_j| <= m/2+2 なので総和は (Σ|c_j|)(m/2+2) で済む．
//                 ホストは出力 1 本の上界（連結した g 本ぶんの和．oz2_group_bound）に m を足して 2^24
//                 以下で，かつ実際の係数でこの総和が 2^22 に収まるときに選ぶ．分割が要らないぶん演算が少ない．K=14080 では桁分割の 7 経路すべてが
//                 こちらになる（桁の上限 32 の法は K < 16384 まで．段の時間のモデルで法を選ぶ経路は，K が大きいと
//                 上限 31 以下の法を選ぶ．ozaki2_moduli.h の累算の条件）．
//   fold_first=0：下の「桁積を畳む際に係数も一緒に畳む」．桁積に法を足すと 2^24 を
//                 超える場合でも扱えるが，1 桁積あたり上位・下位への分割が要る．
//                 どちらの方式も桁積そのものは無誤差の整数であることが前提である
//                 （ホストは K*d^2 > 2^24 を拒む．GEMM の FP32 累算で失った下位 bit は
//                 ここで 2^12 に分けても戻らない）．
//
// 桁積を畳む際に係数も一緒に畳む
//   桁積を 2^12（基数のビット数はホストと共有の OZ2_COMPOSE_SPLIT_BITS．oz2_limits.h）で上位 hi と下位 lo に分け，そこで法へ畳まずに
//   係数 c_j を掛けた形の定数をそのまま当てる：
//       c_j * P_j ≡ hi * (c_j*2^12 mod m) + lo * (c_j mod m)   (mod m)
//   ホストが A_j = (c_j*2^12) mod m と B_j = c_j mod m を渡す（ozaki2_split.h の
//   oz2_product_groups の coeff_hi・coeff_lo）．こうすると桁積 1 枚あたりの剰余計算が要らなくなり，
//   最後に 1 回だけ法へ畳めばよい．
//
// 総和の条件
//   この方式では桁積を連結しない（ホストが maxg=1 に戻す）ので，出力は桁積 1 本で
//   |P_j| <= Pmax（Pmax は出力の上界．経路一律の桁の上限 d なら K*d^2 で，本経路の無誤差条件そのもの．
//   1 桁の法を混ぜる経路と疎な桁の経路は桁行列ごとの上限の積 oz2_group_bound）．hi は切り捨てではなく
//   最近接への丸め（Rnd）なので |hi| <= floor(Pmax/2^12 + 1/2)（d=16 なら K/16 を丸めた値．K=64・d=7 の
//   P=3136 は Pmax/2^12 = 0.77 でも hi=1），|lo| <= 2^11．総和は Σ_j (|hi|*A_j + |lo|*B_j) 以下で，これに最後の
//   ModRound がかかるので，守るべき上限は FP32 の無誤差整数 2^24 ではなく，ModRound が正しい対称剰余を
//   返す 2^22 のほうである．係数は法ごとに大きく違うので，ozaki2_moduli.h の法選択（compose_sum）が
//   この丸めた |hi| の上界と実際の A_j・B_j でこれを検査する（FP8 は K=16384 でも最大 2^21.2．d=32 の経路は
//   K=16384 で超えるので，この方式に落ちる K では法選択が失敗を返す）．出力は対称剰余
//   [-m/2, m/2]（奇数の法では端に届かないので実質 (-m/2, m/2)．偶数の法では商の最近接偶数の丸めで
//   +m/2 と -m/2 の両方が出うる．sfpu_tf_proc.h の ModRound）で，再構成は符号を問わない（recon_crt.cpp）．
//
// DST レジスタ（fp32_dest_acc で 8 本）
//   0 = いま読んだ桁積   4,5 = 作業   6 = 総和   ほかは未使用
// 桁積は 1 枚ずつ DST へ載せて総和へ畳む．平衡 5 桁の経路は np = 25 になるので，
// 全部を DST に並べる形にはできない．
// 前提：ホストが fp32_dest_acc_en を立て，c_0 を UnpackToDestFp32 で受けること．
//
// 引数とバッファ
//   実行時引数（並びは oz2_args.h の ComposeRT）：ntiles, np（GEMM の出力の枚数）, m, 1/m, fold_first,
//              続いて MAXPROD 個ずつ B_j（= c_j mod m．np より後ろは 0）と A_j（= c_j*2^12 mod m）
//   入力  c_0  : GEMM の出力 np 枚（ホストの oz2_product_groups の出力 j0.. の並び．FP32 の無誤差整数．
//                平面リーダ oz2_plane_reader.cpp が積む）
//   出力  c_16 : 剰余 r（FP32．対称剰余）
#include "../../common/kernels/sfpu_tf.h"
#include "oz2_args.h"     // 引数の並びと CB 番号（上限 OZ2_MAXPROD は oz2_limits.h）

constexpr auto cb_p   = tt::CBIndex(oz2_args::CB_PLANE_IN);    // 入力（平面リーダ oz2_plane_reader.cpp が積む）
constexpr auto cb_out = tt::CBIndex(oz2_args::CB_PLANE_OUT);   // 出力（ライタ modslice_writer.cpp が書く）
// 2^12 分割の基数とその逆数（どちらも 2 のべきなので float で無誤差）
constexpr float SPLIT     = (float)(1u << OZ2_COMPOSE_SPLIT_BITS);
constexpr float SPLIT_INV = 1.0f / SPLIT;

void kernel_main() {
    using namespace oz2_args;
    const uint32_t ntiles = get_arg_val<uint32_t>(CP_NTILES);
    const uint32_t np     = get_arg_val<uint32_t>(CP_NP);
    const float m    = fbf(get_arg_val<uint32_t>(CP_MOD));
    const float rinv = fbf(get_arg_val<uint32_t>(CP_RINV));
    // 合成の方式．1 なら桁積を先に法へ畳む（下の説明）．B[] には生の係数が入る．
    const uint32_t fold_first = get_arg_val<uint32_t>(CP_FOLD_FIRST);
    constexpr uint32_t MAXPROD = OZ2_MAXPROD;   // ホストと共有の上限（oz2_limits.h）
    float B[MAXPROD], A[MAXPROD];
    // 係数は MAXPROD 個ぶんの枠で渡る（実行時引数の個数は法によらず一定．並びは oz2_args.h の cp_coeff_*）
    for (uint32_t j = 0; j < np; j++) { B[j] = fbf(get_arg_val<uint32_t>(cp_coeff_lo(j)));
                                        A[j] = fbf(get_arg_val<uint32_t>(cp_coeff_hi(j))); }

    for (uint32_t ti = 0; ti < ntiles; ti++) {
        cb_wait_front(cb_p, np);
        tile_regs_acquire();
        init_sfpu(cb_p, cb_out);
        copy_tile_to_dst_init_short(cb_p);
        // 桁積を 1 枚ずつ reg0 へ載せて総和 reg6 へ畳む（2 つの方式の条件は冒頭の説明）．
        for (uint32_t j = 0; j < np; j++) {
            copy_tile(cb_p, j, 0);
            if (fold_first) {
                ModRound(0, 4, m, rinv);                       // t_j = P_j mod m（対称剰余）
                MulS(0, B[j]);                                 // c_j * t_j（B[] に生の係数）
                if (j == 0) Cpy(0, 6); else Add(6, 0, 6);
            } else {
                Cpy(0, 4); MulS(4, SPLIT_INV); Rnd(4);         // hi = round(P_j / 2^12)
                Cpy(4, 5); MulS(5, SPLIT); Sub(0, 5, 5);       // lo = P_j - hi*2^12
                MulS(4, A[j]); MulS(5, B[j]); Add(4, 5, 4);    // c_j*P_j ≡ hi*A_j + lo*B_j
                if (j == 0) Cpy(4, 6); else Add(6, 4, 6);
            }
        }
        ModRound(6, 4, m, rinv);   // 総和を法へ畳む（対称剰余）
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(6, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_p, np);
    }
}
