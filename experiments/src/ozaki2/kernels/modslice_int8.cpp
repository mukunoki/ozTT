// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：デバイス上の「モジュラースライサ」（INT8 経路，FP64 相当）
// ---------------------------------------------------------------------------
// 役割：オペランドの各要素 x について，整数 A' ≈ x / 2^e * 2^b（|A' - x/2^e*2^b| <= 1 で最近接とは限らない．
//       oz2_digits.h の「桁が満たす条件」）の
//       法 m_i ごとの剰余 (A' mod m_i) を S 本ぶん求め，INT8 行列積に渡せる
//       バイト列として出力する．論文の ssec:oz2（尾崎スキーム II）に対応．b はホストが CRT の窓と入力の
//       動的域から決める（既定は 54），2^e は行／列ごとの正規化スケールである．
//
// なぜ「桁に分けて」扱うのか
//   A' は最大 100 bit の整数なので，FP32（整数を無誤差に表せるのは 2^24 まで）
//   には載らない．そこで A' を 2^20 進の ND 桁 d0..d_{ND-1}（平衡表現・符号つき）に
//   分けて扱う（各桁は 2^20 以下なので FP32 に無誤差で載る．ND は b から決まり 3〜5）．
//   本カーネルは
//       A' mod m = ( Σ_{j>0} (d_j mod m)*c_j + d0 ) mod m
//   として剰余を合成する．ここで c_j = 2^(20j) mod m（ホスト計算）．d0 は重み 1 なので
//   畳まずに足す（大きさの条件は oz2_digits.h の oz2_residue）．
//   桁を先に法で畳んでから重みを掛けるので途中の積は m^2 未満（255^2 = 65025）で，
//   すべて FP32 で無誤差に収まる．
//
//   要点は「オペランド x ではなく，丸めた整数 A' の側を桁分割する」ことである．
//   x 自身を FP32 の多倍長（limb）に分けても，|x| が 1 近傍の要素でしか
//   各 limb が整数にならず，一般の入力では破綻する．
//
//   桁そのものを作る段（正規化・整数化・桁分割）は oz2_digits.h にあり，桁分割の
//   経路（modslice_digit.cpp）と共用する．
//
// 引数とバッファ
//   コンパイル時引数 0 = S（法の本数），1 = 桁の基数 2^20 の float ビット列，
//     2 = A' の 2^20 進の桁数 ND（3〜5．oz2_digits.h）．3・4 番は桁分割経路用で
//     本カーネルは読まない（ホストは両経路に同じ並びで渡す）
//   実行時引数 0 = ntiles（処理するタイル数）
//   入力  桁を作る段の入力そのもの（oz2_digits.h と slice_tf_core.h を参照）
//         b と行／列ごとのスケール 2^e は，リーダ（slice_f64_reader.cpp）が指数差分として
//         FP64 の指数へ足し込み済み．c_2 は inv ではなく下位 limb のスケール s である
//   出力  c_16            : S 本の剰余（対称剰余．INT8 行列積向けに
//                           符号-絶対値の uint8 で符号化する）
#include "oz2_digits.h"
#include "api/compute/eltwise_unary/typecast.h"

using namespace oz2_args;
constexpr auto cb_out = tt::CBIndex(CB_PLANE_OUT);  // 出力：S 本の剰余

constexpr uint32_t S = get_compile_time_arg_val(SC_S);   // 法の本数（ホストが K から決める）
constexpr uint32_t MAXS = OZ2_MAXS;         // 法の本数の上限（配列の静的確保用．oz2_limits.h）
static_assert(S <= MAXS, "法の本数が固定配列の大きさを超える");
// 法ごとの MOD[i]，逆数 RINV[i]，桁の重み C0/C1（/C2/C3）は，すべて実行時引数
//（float のビットパターン）で渡す．これにより精度 b も法の集合も
// ホスト側だけで差し替えられる．並びは oz2_args.h の sl_*：ntiles，MOD，RINV，
// C0・C1（4 桁以上（b > 60）のときだけ C2 = 2^60 mod m，5 桁（b > 80）のときだけ C3 = 2^80 mod m）
constexpr uint32_t DF_F32 = (uint32_t)DataFormat::Float32;
constexpr uint32_t DF_U8  = (uint32_t)DataFormat::UInt8;

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(sl_ntiles());
    float MOD[MAXS], RINV[MAXS], C[MAXS][OZ2_NDIGIT - 1];
    // 実行時引数から法まわりの定数を読み出す（すべてホストが算出済み）
    constexpr uint32_t A_MOD = sl_mod(S, 0), A_RINV = sl_rinv(S, 0);
    for (uint32_t i = 0; i < S; i++) { MOD[i] = fbf(get_arg_val<uint32_t>(A_MOD + i));
                                       RINV[i] = fbf(get_arg_val<uint32_t>(A_RINV + i)); }
    // C[i][j] = 2^((j+1)*20) mod m_i（桁の重み）．4 桁なら 3 つ目まで，5 桁なら 4 つ目まで使う．
    constexpr uint32_t A_W = sl_weight(S, 0, 0);   // 桁の重みの先頭．桁 j の法 i は A_W + j*S + i
    for (uint32_t i = 0; i < S; i++)
        for (uint32_t j = 0; j + 1 < OZ2_NDIGIT; j++)
            C[i][j] = fbf(get_arg_val<uint32_t>(A_W + j * S + i));
    for (uint32_t ti = 0; ti < ntiles; ti++) {
        // オペランドのタイル 1 枚から A' の桁を作る（正規化・整数化・桁分割）
        oz2_make_digits();
        for (uint32_t i = 0; i < S; i++) {   // 法ごとに 1 タイル出力する
            tile_regs_acquire();
            init_sfpu(cb_dlo, cb_out);
            // reg2 = A' mod m（対称剰余）．ここまでは桁分割経路と共通である．
            oz2_residue(MOD[i], RINV[i], C[i]);
            SignMag8(2, 3, 4);                   // 符号-絶対値へ符号化（sfpu_tf.h）
            typecast_tile_init<DF_F32, DF_U8>(); typecast_tile<DF_F32, DF_U8>(2);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            pack_tile(2, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        oz2_digits_pop();
    }
}
