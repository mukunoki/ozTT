// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：デバイス上の再構成（Int32 入力，triple-float）
// ---------------------------------------------------------------------------
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
// 入力の復号について
//   入力の Int32 は FP32 が無誤差に表せる 2^24 を超えることがある．そこで無誤差な
//   2 つの FP32（Vh, Vl）に分解してから，1 つずつアキュムレータへ足し込む
//   （どちらも単体では無誤差な FP32）．
//
//   Blackhole の DST 上の Int32 は，どこから来たかで表現が違う．どちらかを
//   コンパイル時引数 SIGNMAG で選ぶ：
//     SIGNMAG=1  符号-絶対値：行列エンジンがパックした Int32 を copy_tile で載せたもの
//                （INT8 経路の部分和 P_d）．
//                  絶対値 = (raw & 0x7FFFFF00) + (raw & 0xFF)，符号 = sign(float(raw))
//                符号を別に取り出し，分解したあとで掛け戻す．
//     SIGNMAG=0  2 の補数：typecast<Float32,Int32> が作ったもの
//                （遅延再構成が Int32 で累積した S_d．accum_i32.cpp／GEMM の融合）．
//                  v = (v & 0xFFFFFF00) + (v & 0xFF) がそのまま成り立つ．上位は
//                  256 の倍数（256*floor(v/256)）で Int32 の全域で有効桁 24 bit 以下，下位は 255 以下で，
//                  どちらも FP32 が無誤差に表せる．符号の取り出しと掛け戻しが要らない
//                  （上位・下位を FP32 へ移す型変換は SIGNMAG によらず行う）．
//                  定常段の演算数は，タイルの読み込みを数えなければ
//                  基数倍 + 復号 4 + tfAddS 15 + Cpy 1 + tfAddS 15 で，基数倍は基数が
//                  2 のべき（BF16・FP16・TF32・BFP8_b・FP8・BFP2_b）なら tfMulP2 の 19（計 54），
//                  そうでなければ（BFP4_b・INT8）tfMulR の 44（計 79）である（sfpu_tf.h の
//                  tfMulRadix．本番の値域では結果はどちらでも同じ．sfpu_tf_proc.h の tfMulP2 の一致の条件）．
//                  SIGNMAG=1 では型変換・符号判定・掛け戻しが加わって 4 増える．
//   取り違えると負の値が壊れる（相対誤差 1e4 の桁で落ちる）．マスクを積む
//   recon_reader_i32in.cpp へも，ホストが同じ SIGNMAG を渡すこと．
//
// 小さい対角の近道（NSINGLE_CAST．2 の補数の入力だけ．ホストのつまみは OZ_NSMALL）
//   遅延再構成の S_d は |S_d| <= (d+1) K d_max^2 で抑えられる（対 d+1 個 × K 項 × 桁の積）．
//   ホストがこの上界を対角ごとに評価し，2^24 以下に収まる先頭の NSINGLE_CAST 本を渡す
//   （ozaki1_plan.hpp の recon_nsmall）．そこでは S_d そのものが FP32 の無誤差整数なので，
//   2 本への分解をせず 1 回の型変換で FP32 へ移し，1 回だけ足す．マスクの読み込み 2 枚・
//   ビット積 2 回・型変換 1 回・Cpy 1 回・2 回目の tfAddS（15 演算）が消える．
//   14080^3 では BF16・FP16・TF32・BFP8_b が d=0 の 1 本，FP8 が 4 本，BFP4_b・BFP2_b は全部．
//   値の読み方が無誤差なのは分解した場合と同じだが，足し込みが 2 回から 1 回になるので，
//   triple-float の最下位 limb の丸め（tfAddS の最下位の足し込み Add(acc_lo, tmp2, acc_lo)）の回数が変わり，最下位の
//   ビットは分解した場合と一致するとは限らない（3520^3 の FP8・BFP4_b・BFP2_b で全要素の
//   0.004% 以下が FP64 の 1 ulp だけ違い，参照に近づく要素と遠ざかる要素はほぼ同数．
//   docs/nongemm_opt_results.md の §3）．
//   スキーム II の recon_crt.cpp の NSMALL_FRAG は「末尾の断片を tfAddSmall で足す本数」で別の量である
//   （ホストのつまみも OZ2_NSMALL と別）．
//
// コンパイル時引数・実行時引数の並びは oz1_args.h の ReconCT・ReconRT（このカーネルは RC_ACC を読まない．
// RC_NSINGLE_CAST は SIGNMAG=1 ではホストが 0 を渡す）
// DST（fp32_dest_acc で 8 本．ホストの recon_build_prog が dst_full_sync_en を立てるので
//   8 本を 1 区画として使える．半分同期のままだと 4..7 番が次の区画と重なる）．名前は sfpu_tf_proc.h の namespace dst：
//   アキュムレータの 3 本組 = acc_hi・acc_mid・acc_lo．復号した 2 本は Vh を term（tfAddS が足す値），Vl を keep
//   （tfAddS が壊さない）に置き，マスク 2 枚と符号は tmp0・tmp1・tmp2

#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "recon_pack.h"
#include "oz1_args.h"
using namespace oz1_args;

constexpr uint32_t I32   = get_compile_time_arg_val(RC_IN_FMT);   // Int32 の形式値
constexpr uint32_t F32   = get_compile_time_arg_val(RC_F32);      // Float32 の形式値
constexpr uint32_t C_RDX = get_compile_time_arg_val(RC_RADIX);    // radix の float ビット列
constexpr uint32_t SIGNMAG = get_compile_time_arg_val(RC_SIGNMAG); // 1 なら符号-絶対値
constexpr uint32_t NSINGLE_CAST = get_compile_time_arg_val(RC_NSINGLE_CAST);  // 先頭の何本を 1 回の型変換で読むか
static_assert(!SIGNMAG || NSINGLE_CAST == 0, "近道は 2 の補数の入力だけ（符号-絶対値は型変換で値にならない）");

constexpr auto cb_in   = tt::CBIndex(CB_RECON_IN);   // 入力：対角ごとの部分和 P_d（Int32）
constexpr auto cb_mask = tt::CBIndex(CB_MASK);       // 入力：2 本組へ分ける際のマスク
constexpr auto cb_hi   = tt::CBIndex(CB_LIMB_HI);

// cb_in[idx] の P を，符号付きで無誤差な 2 本の FP32（rh, rl）へ復号する．
// マスク mh（SIGNMAG=1 なら 0x7FFFFF00，0 なら 0xFFFFFF00）と ml(0xFF) は読み込み済みであること．sgn は作業用．
ALWI void decode(uint32_t idx, uint32_t rh, uint32_t rl,
                 uint32_t mh, uint32_t ml, uint32_t sgn) {
    if constexpr (SIGNMAG) {   // 符号を別に取っておく（下で絶対値へ掛け戻す）
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, idx, sgn);
    typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(sgn);
    sign_tile_init(); sign_tile(sgn);                       // sgn = +1/-1/0
    }
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, idx, rh);
    binary_bitwise_tile_init();
    bitwise_and_binary_tile<DataFormat::Int32>(rh, mh, rh); // raw & mh（上の注記）
    typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(rh);
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, idx, rl);
    binary_bitwise_tile_init();
    bitwise_and_binary_tile<DataFormat::Int32>(rl, ml, rl); // raw&0xFF
    typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(rl);
    if constexpr (SIGNMAG) {
        mul_binary_tile_init();
        mul_binary_tile(rh, sgn, rh);
        mul_binary_tile(rl, sgn, rl);
    }
}

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(RCR_NTILES);
    // マスク 2 枚はリーダが L1 で 1 回だけ作って積む（定数なので pop しない）
    cb_wait_front(cb_mask, 2);
    constexpr uint32_t ndiag = get_compile_time_arg_val(RC_NDIAG);  // L+1

    for (uint32_t t = 0; t < ntiles; t++) {
        cb_wait_front(cb_in, ndiag);
        init_sfpu(cb_in, cb_hi);   // タイルごとに int32 の unpack-to-dest を設定し直す
        tile_regs_acquire();

        using namespace dst;
        for (uint32_t d = 0; d < ndiag; d++) {
            if (d > 0) tfMulRadix<C_RDX>();                    // Horner の 1 段：acc *= radix（2 のべきなら tfMulP2）
            if (d < NSINGLE_CAST) {
                // |S_d| <= 2^24（ホストが示した上界）：1 回の型変換で無誤差に FP32 へ
                copy_tile_to_dst_init_short(cb_in);
                const uint32_t r = (d == 0) ? acc_hi : term;  // 初回はアキュムレータの上位へ直に置く
                copy_tile(cb_in, d, r);
                typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(r);
                if (d == 0) { Sub(acc_hi, acc_hi, acc_mid); Sub(acc_hi, acc_hi, acc_lo); }  // アキュムレータを (S_0, 0, 0) に
                else        tfAddS<3>();                      // acc += S_d
                continue;
            }
            copy_tile_to_dst_init_short(cb_mask);
            copy_tile(cb_mask, 0, tmp0); copy_tile(cb_mask, 1, tmp1);
            decode(d, term, keep, tmp0, tmp1, tmp2);      // P_d を Vh（term）・Vl（keep）に復号
            if (d == 0) { Sub(term, term, acc_hi); Sub(term, term, acc_mid); Sub(term, term, acc_lo); }  // 初回はアキュムレータを 0 に
            tfAddS<3>();                                  // acc += Vh．keep の Vl は保持される
            Cpy(keep, term);                              // term に Vl を移す
            tfAddS<3>();                                  // acc += Vl
        }

        recon_pack_triple();   // アキュムレータの 3 本組をライタへ（recon_pack.h）

        cb_pop_front(cb_in, ndiag);
    }
}
