// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：デバイス上の CRT 再構成（triple-float，FP64 相当）
// ---------------------------------------------------------------------------
// 役割：法ごとの剰余を S 本受け取り（c_0．形は下の IN_RESIDUE を参照），
//       中国剰余定理の「分数形」
//           frac = ( Σ_i k_i / m_i ) mod 1,   k_i = (C_i mod m_i) * y_i mod m_i
//       を FP32 3 本組（triple-float，実効約 68 bit）で計算して出力する．
//       最終的な C = frac * KM * tau_A * tau_B（KM = M*2^-2b）は後段の
//       recon_writer.cpp がデバイス上で掛ける（ホストは並べ替えるだけである）．論文の ssec:oz2（尾崎スキーム II の CRT）と
//       ssec:nofp64（FP32 を用いた再構成．分数形と triple-float）に対応．
//
// なぜ分数形なのか
//   素直な CRT 復元 x = Σ c_i e_i は M に近い巨大整数になり，そもそも避けたかった
//   多倍長整数がここで復活してしまう．両辺を M で割った分数形なら，どの中間値も
//   高々 S 程度にしかならない．
//
// なぜ triple-float が要るのか
//   frac = C'/M は絶対値が 1 未満の小さな数で，その絶対精度が到達精度を直接決める．
//   b=54 では M が約 2^123，真の積 |C'| は最悪で M/2 直下なので frac は最大 1/2 まで
//   使われるが，典型的な要素はそれより桁が小さい．FP32 2 本組（double-single）の
//   絶対精度は約 2^-48 しかなく，これでは 53 bit に届かない．
//   3 本目の limb を足すと絶対精度が約 2^-68 になり，相対で 2^-55＝FP64 相当に届く．
//   GPU 実装（GEMMul8）はここを FP64 で計算するが，本ハードウェアは FP64 を
//   持たないので複数語の FP32 でエミュレートする．これが scheme II が本ハード
//   ウェアで非 GEMM 律速になる主因である．
//
// DST レジスタの使い分け（fp32_dest_acc 有効時は 8 本）
//   0,1,2 = triple-float アキュムレータ (a0,a1,a2)
//   3     = 加算する値 v
//   4,5,6 = 作業用
//   7     = k_i（項の計算中ずっと保持）
// 項 k_i*(1/m_i) の作り方
//   1/m_i はホスト側であらかじめ FP32 の断片 NRP 個の和に分解してある．1/m 自体は
//   2 進有限桁では表せないのでこれは近似である．断片は 12 bit の Dekker 半分なので，
//   |k_i| <= floor(m/2) + 1 <= 2049（12 bit．法の上限は桁 32 の経路（BF16・FP16・TF32・BFP8_b）の 4096．k_i は r_i*y_i の ModRound で，
//   入力が 2^22 を超えうるので対称剰余から 1 だけはみ出しうる．ozaki2_moduli.h の oz2_choose_small_tail）との積は
//   どれも 24 bit に収まって FP32 で無誤差になる．つまり項は「無誤差な FP32 スカラ NRP 個」に
//   分かれ，1 個ずつ誤差なしで足し込める．
//   【12 + 12 = 24 で FP32 の仮数とちょうど同じ】余裕は無い．法の上限を 4096 より
//   上げるか断片の幅を 12 bit より広げると，この積に丸めが入って前提が崩れる．
//   どちらかを変えるなら OZ2_RGRAN と ozaki2_moduli.h の分解も一緒に見直すこと．
//   例外は 1 パス構成（rp0）と第 2 パス（rp2）の最後の断片で，そこだけ 24 bit まるごと
//   使って幅を稼ぐ（積に丸めが入るが，その位はすでに十分小さい）．
#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "recon_args.h"
#include "oz2_limits.h"
#include "recon_crt_proc.h"   // 法ごとの手続き（k_i・断片の足し込み・整数部の削除．ホストの模擬も同じものを組む）
using namespace recon_args;
#include "api/compute/eltwise_binary.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"

constexpr auto cb_in   = tt::CBIndex(CB_IN);     // 入力：C_i
constexpr auto cb_mask = tt::CBIndex(CB_MASK);   // 入力：符号-絶対値を割るマスク 2 枚
constexpr auto cb_out  = tt::CBIndex(CB_LIMB);   // 出力：frac の各 limb

constexpr uint32_t S = get_compile_time_arg_val(C_S);     // 法の本数（ホストが K から決める）
constexpr uint32_t NLIMB = get_compile_time_arg_val(C_NLIMB); // 再構成の limb 数：2=FP32 級，3=FP64 相当
// 入力の形は経路で 2 通り．IN_RESIDUE=0 は GEMM 段が書いた符号-絶対値 int32 の C_i を
// 1 枚で受け取り，法をとるところまでここで行う（INT8 経路）．IN_RESIDUE=1 は FP32 を 1 枚
// 受け取る（桁分割経路．変換も上位／下位の分割も要らない）．中身はふつう合成段（oz2_compose.cpp）が
// 法へ畳んだ対称剰余だが，RAWMASK の立つ法（1 桁の法）は合成段を通らない GEMM の出力で，ここで畳む（下の RAWMASK）．
constexpr uint32_t IN_RESIDUE = get_compile_time_arg_val(C_IN_RESIDUE);
// 逆数の断片数．既定の再構成は 1 パス（NRP=OZ2_NRP0）で済ませる．ホストが 2 パスを選んだとき
// （OZ2_PASSES．OZ2_FP64 では 1 パスの誤差の見込みが大きいときだけ．ozaki2_autop.hpp の oz2_recon1_log2）
// だけ 2 パスに分け，同じカーネルを定数だけ替えて 2 回流す．
//   パス 1（NRP=OZ2_NRP1）：1/m_i を共通粒度 2^-64 へ丸めた R_i の断片．k_i が整数なので
//     k_i*R_i もその総和も同じ粒度の整数倍で，3 本組に収まるかぎり累算は無誤差になる．
//   パス 2（NRP=OZ2_NRP2）：丸めの残り（両符号）の断片．値そのものが 2^-64 以下なので，同じ
//     3 本組でも絶対誤差は無視できる．
// 2 つに分けるのは DST が FP32 で 8 本しかなく（src/probes/dst_probe.cpp で実測），4 本組の
// アキュムレータ（9 本要る）が載らないためである．3 本組を 2 本立てれば同じ精度に届く．
// 2 つの累算を混ぜてはならない：パス 2 の項をパス 1 の（大きい）3 本組へ足し込むと
// 1 加算あたり ULP(a2) の誤差が乗り，せっかくの無誤差性が失われる．別々に積んで
// 最後に合成する（合成は recon_writer.cpp の assemble_frac_f64．K*M 倍も同じ場所で，
// 1 パス構成では既定で recon_reader.cpp が前半の OZ2_JSPLIT 要素（既定 512）を分担する．
// 2 パス構成でも両パスで分担し，第 1 パスは自分の 3 本を 128 bit 固定小数まで足して残す）．
constexpr uint32_t NRP = get_compile_time_arg_val(C_NRP);
// 末尾の何本を tfAddSmall（第 1 段を飛ばす安い足し込み）で足すか．
// ホストが断片の大きさから数える（条件と誤差の評価は sfpu_tf_proc.h の tfAddSmall）．
constexpr uint32_t NSMALL_FRAG = get_compile_time_arg_val(C_NSMALL_FRAG);
// CPROBE【調査用】ホスト側 OZ2_CRT_PROBE．どのモードが何を残すかは下の CrtMode の表．
// 仕事を抜くモードは答えが壊れる．測った差はそのまま演算の所要時間ではない：
// リーダ・演算・ライタが重なって動くため，抜いた側が律速でなければ差は出ない．
// 単独で外した差ではなく，出口の組み立てを外す OZ_RECON_PROBE と組み合わせた
// 2x2 から読むこと（docs/recon_cost_breakdown.md の結果 3）．
constexpr uint32_t CPROBE = get_compile_time_arg_val(C_PROBE);
// RAWMASK：法 i のビットが立っていれば，剰余の平面には合成段を通らない GEMM の出力 P（桁 32 の AWE 経路の 1 桁の法．
// 桁積 1 本で |P| <= K*R^2．ホストの法選択（ozaki2_choose_moduli_w）が |P| + m <= 2^24 を課す）が入っている．ここで 1 度
// ModRound して剰余にする：P も商と法の積も 2^24 以下の整数なので FP32 で無誤差で剰余類は保たれ，結果は
// |r| <= m/2 + |P|*2^-23 <= m/2 + 2（m <= 65．tools/modround_check.cpp の (e) が総当たりで確かめる）．
// 続く r*y_i（(m/2+2)(m-1) < 2^22）も一般の保証の内側．
// 他の経路は 0（分岐はコンパイル時に消える）．ホストは ozaki2_stages.hpp の recon_make_kernels．
constexpr uint32_t RAWMASK = get_compile_time_arg_val(C_RAWMASK);
constexpr uint32_t MAXS = OZ2_MAXS;     // ホストと共有の上限（oz2_limits.h）
constexpr uint32_t MAXRP = OZ2_MAXRP;
static_assert(S <= MAXS, "法の本数が固定配列の大きさを超える");
static_assert(NRP <= MAXRP, "逆数の断片の本数が固定配列の大きさを超える");
constexpr uint32_t I32 = (uint32_t)DataFormat::Int32, F32 = (uint32_t)DataFormat::Float32;
// 法ごとの定数はすべて実行時引数（float のビットパターン）で渡す．並びは
// ozaki2/kernels/recon_args.h の crt_* が唯一の定義で，ホスト（ozaki2_stages.hpp の
// recon_set_args）と下の load_crt_constants が同じ関数で添字を作る．
// 最後の K*M 倍と正規化スケールの掛け戻しは recon_writer.cpp が行う．

// o = v mod m（対称剰余）．このカーネルが法を取る値は次の 3 通りである．
//   |C_i| の上位 15 bit・下位 16 bit（どちらも 2^16 未満．INT8 経路だけ）
//   符号を戻した |C_i| mod m（m^2/2 + m/2 未満．同じく INT8 経路（m <= 255）だけなので 2^15 未満）
//   r_i * y_i（m^2/2 未満．全経路）
// 最後の r_i * y_i は，法が 4096 まで伸びた桁 32 の経路（BF16・FP16・TF32・BFP8_b）で ModRound が一般に保証する
// |d| < 2^22 を超える（m^2/2 = 2^23）．|d| < 2^23 なので商と法の積は 2^24 以下で剰余類は保たれ，対称剰余からの
// はみ出しは |d|*2^-23 < 1 である（sfpu_tf_proc.h の ModRound の注記）．経路の上限以下の選びうる全法・全係数 y（集合を
// 変えれば y_i は 1..m_i-1 のうち m_i と互いに素な値のどれにもなりうる）について，|k| <= floor(m/2)+1（ozaki2_moduli.h の oz2_crt_k_bound）
// であることを tools/modround_check.cpp の (a') が総当たりで確かめている（奇数の法で floor(m/2) を 1 超える組がある）．
// 2 平面の復号（上の 2 つ）は桁分割の経路では通らない（剰余を FP32 で受ける．IN_RESIDUE=1）．
// 出力レジスタを取る形の ModRound を使い，入力を残したまま別レジスタへ書く．
// 代表元は [0,m) に折り返さず対称剰余のまま使う．CRT は k_i/m_i の和の小数部しか
// 見ないので，k_i を m_i の整数倍だけずらしても和はほぼ整数だけずれ，小数部の取り出しで
// 消えるからである．「ほぼ」なのは 1/m_i を FP32 5 個の和 r_i で近似しているためで，
// ずれは 1 ではなく 1 + (m_i*r_i - 1) になる．この残差は本実装の全ての法で 2^-64.5 以下，
// S 本足しても 2^-60.7 以下であり，r_i が 1/m_i でない以上どのみち各項に入っている
// 量と同じ桁なので，折り返さなくても到達精度は変わらない．
// --- 探針の契約 -------------------------------------------------------------
// CPROBE は段階の番号ではなく，別々の実験の名前である．番号の大小で分岐を書くと
// 「以降を全部外す」条件が後ろの実験を覆い隠し，後ろのモードが前のモードと同じものを
// 測ってしまう．そこで各モードが何を残すかを表として書き，下の処理はこの旗だけを見る．
//
//   モード             入力の取り込み  CRT 係数  断片の乗算  足し込み
//   Normal                  あり        あり       あり       あり
//   SkipCrt                 あり        なし       なし       なし
//   TransportOnly           なし        なし       なし       なし
//
// 断片の乗算だけ・足し込みだけを外すモードや，出力タイルごとの初期化をループの外へ括り出す
// 切り替えは持たない（それらの結果と再現の手順は docs/recon_cost_breakdown.md）．
//
// 「入力の取り込み」は，剰余入力なら copy_tile 1 回，int32 入力なら符号-絶対値の
// 復号一式を指す．どちらの入力形でも同じ意味になるよう，TransportOnly は両方を外す．
enum class CrtMode : uint32_t {
    Normal = 0, SkipCrt = 1, TransportOnly = 2,
};
constexpr CrtMode MODE = (CrtMode)CPROBE;
static_assert(CPROBE <= 2, "CPROBE は 0--2");
constexpr bool DO_LOAD   = (MODE != CrtMode::TransportOnly);
constexpr bool DO_CRT    = (MODE != CrtMode::SkipCrt && MODE != CrtMode::TransportOnly);
// 仕事を抜くモードでは 3 本組が 0 のまま出ていく．0 の 3 本組は assemble_frac_f64 が
// a128_term の入口で弾くので，ライタの FP64 組み立てまで一緒に消え，床ではなく
// 「床 - 組み立て」を測ってしまう．0 でない定数を置いて，ライタ側の仕事は残す．
constexpr bool SEED_ACC  = (MODE != CrtMode::Normal);

// --- 法ごとの定数 -----------------------------------------------------------
struct CrtConstants {
    float mod[MAXS], rinv[MAXS], yi[MAXS];   // m_i，1/m_i，y_i = (M/m_i)^{-1} mod m_i
    float rp[MAXRP][MAXS];                   // 1/m_i の 12 bit 断片
    float m16[MAXS];                         // 2^16 mod m_i（符号-絶対値の復号でだけ使う）
    ALWI float frag(uint32_t t, uint32_t i) const { return rp[t][i]; }   // 法 i の断片 t（recon_crt_proc.h の accumulate が引く）
};

ALWI void load_crt_constants(CrtConstants& c) {
    for (uint32_t i = 0; i < S; i++) {
        c.mod[i]  = fbf(get_arg_val<uint32_t>(crt_mod(S, i)));
        c.rinv[i] = fbf(get_arg_val<uint32_t>(crt_rinv(S, i)));
        c.yi[i]   = fbf(get_arg_val<uint32_t>(crt_yi(S, i)));
        for (uint32_t t = 0; t < NRP; t++)
            c.rp[t][i] = fbf(get_arg_val<uint32_t>(crt_frag(S, t, i)));
        if constexpr (!IN_RESIDUE)
            c.m16[i] = fbf(get_arg_val<uint32_t>(crt_m16(S, NRP, i)));
    }
}

// --- 入力 1 枚から r_i = C_i mod m を作り，dst::tmp0 へ置く -------------------
// 剰余入力：合成段が畳んだ対称剰余ならそのまま使う（|r_i| <= m/2 で，法の上限 4096 でも 2048．
// FP32 に無誤差で載る）．1 桁の法の GEMM の出力は，呼び出し側が RAWMASK を見て ModRound する．
ALWI void load_residue_fp32() {
    cb_wait_front(cb_in, 1);
    if constexpr (DO_LOAD) { copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, dst::tmp0); }
    cb_pop_front(cb_in, 1);
}

// int32 入力：GEMM 段が書いた符号-絶対値をここで割る．絶対値は 31 bit あって FP32 の
// 無誤差整数 2^24 を超えうるので，上位 15 bit（chi）と下位 16 bit（clo）に分けてから
// 法をとる：|C_i| = chi*2^16 + clo，|C_i| mod m = (chi mod m)*(2^16 mod m) + (clo mod m)．
// chi*2^16 は 15 bit の値に 2^16 を掛けた形なので FP32 で無誤差，clo < 2^16 も無誤差．
// 符号は最後に掛け戻す．
ALWI void decode_signmag_residue(float m, float rinv, float m16) {
    using namespace dst;
    cb_wait_front(cb_in, 1);
    if constexpr (DO_LOAD) {
        copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, tmp2);
        typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(tmp2);
        sign_tile_init(); sign_tile(tmp2);                 // sgn = +1/-1/0（tmp2 に保持）
        copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, term);
        copy_tile_to_dst_init_short(cb_mask); copy_tile(cb_mask, 0, tmp1);
        binary_bitwise_tile_init();
        bitwise_and_binary_tile<DataFormat::Int32>(term, tmp1, term);   // raw & 0x7FFF0000
        typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(term);
        MulS(term, 1.52587890625e-05f);                    // /2^16 -> chi（2 べきなので無誤差）
        ModRound(term, keep, tmp1, m, rinv);
        MulS(keep, m16);                                   // (chi mod m) * (2^16 mod m)
        copy_tile_to_dst_init_short(cb_in); copy_tile(cb_in, 0, term);
        copy_tile_to_dst_init_short(cb_mask); copy_tile(cb_mask, 1, tmp1);
        binary_bitwise_tile_init();
        bitwise_and_binary_tile<DataFormat::Int32>(term, tmp1, term);   // raw & 0x0000FFFF
        typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(term);  // clo
        ModRound(term, tmp0, tmp1, m, rinv);
        Add(tmp0, keep, tmp0);      // 2 項とも m^2/2+m/2 未満なので無誤差
        Mul(tmp0, tmp2, tmp0);      // 符号を掛け戻す（|C_i| mod m に符号を付けた形）
        ModRound(tmp0, tmp1, m, rinv);                     // r_i = C_i mod m を tmp0 に
    }
    cb_pop_front(cb_in, 1);
}

// --- 3 本組の初期化 ---------------------------------------------------------
// 直前の DST の中身に依らない定数書き込みで置く（r_i の入ったレジスタの自己減算で
// 0 を作ると，入力の取り込みを外すモードではその値が未定義になる）．
ALWI void clear_accumulator() {
    using namespace dst;
    if constexpr (SEED_ACC) { Set(acc_hi, 0.3f); Set(acc_mid, 1e-8f); Set(acc_lo, 1e-16f); }
    else                    { Set(acc_hi, 0.f);  Set(acc_mid, 0.f);   Set(acc_lo, 0.f); }
}


// --- 小数部の畳み込みはライタ側で行う -------------------------------------
// 法ごとの足し込みの後に a0 -= round(a0) を行うので（下の kernel_main），最後の法の後の a0 は
// [-1/2, 1/2] にあり，3 本組の値は [-1/2,1/2] から a1 + a2（高々 2^-18 程度）しかはみ出さない．
// これを畳み直す（(-1/2,1/2) の値は変えず，はみ出しを ±1 で戻す）のはライタ（recon_writer.cpp・recon_reader.cpp）の a128_frac_to_f64
// （common/kernels/f64_tf.h．1 パスでは assemble_frac_f64 から，2 パスの最終パスでは第 1 パスの固定小数に自分の 3 本を
// 足したあとに直接呼ぶ）で，128 bit の固定小数のまま厳密に 1 を足し引きする．
// ここで FP32 のまま畳む（t = fl(a0+a1) が 1/2 を越えたら a0 -= 1，-1/2 を下回ったら a0 += 1）と，
// 2 つの誤りが起きる（tools/crt_fold_check.cpp の FP32 模擬）：
//   (1) 見逃し：a0 = -1/2 ちょうど（a0 = 1.5 が Rnd の最近接偶数で 2 へ丸まると出る）に a1 = -2^-30 が
//       乗ると t = -1/2 で補正が入らず，値は真の frac（1/2 - 2^-30）から 1 ずれたまま出る．ライタが
//       畳まなければ，C が M*2^-2b*tau_A*tau_B だけ違う答えになる．
//   (2) 丸め：a0 が (1/4, 1/2) にあり a1 が t を 1/2 の上へ押すと，a0 - 1 は FP32 で厳密でない
//       （a0 の ulp 2^-25 が結果の ulp 2^-24 に載らない）．frac が 2^-25 まで丸まり，C の相対誤差が
//       2^-24 程度になる．1 パス構成でも 2 パス構成でも起きる．
// どちらも 1/2 - |frac| が a1 の大きさ（2^-25〜2^-18）より小さい要素でだけ起き，窓の条件
// （ozaki2_moduli.h の OZ2_WIN_EPS）が保証する余白は約 2^-31.4 なので，窓の端の要素（入力の要素が
// 行・列の最大値にそろう定数入力など）では届きうる．既定の法集合の余白の最小は BF16 で 1.5e-7 bit
// （K=12096．1/2 - |frac| >= 約 2^-24．K = 32..16384 を 32 刻みで見た tools/crt_fold_check.cpp の table）である．
// 乱数の入力では frac が 1/2 から遠く，どちらも起きない（3520^3 の CHECKSUM は FP32 で畳んでもライタで畳んでも同じ）．

ALWI void pack_fraction_limbs() {
    cb_reserve_back(cb_out, NLIMB);
    pack_reconfig_data_format(cb_out);
    pack_tile(dst::acc_hi,  cb_out);                 // frac の上位
    pack_tile(dst::acc_mid, cb_out);                 // frac の中位（2 本組なら下位）
    if constexpr (NLIMB >= 3) pack_tile(dst::acc_lo, cb_out);   // frac の下位
    cb_push_back(cb_out, NLIMB);
}

void kernel_main() {
    using namespace dst;
    SfpuDev w;   // 法ごとの手続き（recon_crt_proc.h）に渡すデバイスのラッパ
    const uint32_t ntiles = get_arg_val<uint32_t>(crt_ntiles());
    CrtConstants c; load_crt_constants(c);
    // 分割用マスク 2 枚は定数なので CB に常駐させる（リーダがループ前に 1 回だけ流す）
    if constexpr (!IN_RESIDUE) cb_wait_front(cb_mask, 2);
    for (uint32_t ti = 0; ti < ntiles; ti++) {
        tile_regs_acquire();
        // SFPU の設定は tile_regs の周期で失われるので，タイルごとに呼ぶ
        // （ループの外へ括り出すと答えが壊れる．docs/recon_cost_breakdown.md）．
        init_sfpu(cb_in, cb_out);
        for (uint32_t i = 0; i < S; i++) {           // 法ごとに 1 項を足し込む
            const float m = c.mod[i], rinv = c.rinv[i];
            if constexpr (IN_RESIDUE) load_residue_fp32();
            else                      decode_signmag_residue(m, rinv, c.m16[i]);
            if (i == 0) clear_accumulator();
            if constexpr (!DO_CRT) continue;
            // k_i を keep に（対称剰余．RAWMASK の法は合成段を通らない 1 桁の法の GEMM の出力を先に法へ畳む）
            crt_proc::term_k(w, (RAWMASK & (1u << i)) != 0u, m, rinv, c.yi[i]);
            // 項 k_i*(1/m_i) を断片に分けて足し込む．断片は k_i を掛けても無誤差な FP32 スカラ NRP 個で，1 個ずつ誤差なしで
            // 足せる．末尾の NSMALL_FRAG 本は第 1 段を飛ばして安く足す（条件と誤差は sfpu_tf_proc.h の tfAddSmall）
            crt_proc::accumulate<NLIMB>(w, NRP, NSMALL_FRAG, c, i);
            // 法ごとに整数部を落として a0 を O(1) に保つ．最後にまとめて mod 1 すると
            // 総和が S まで膨らみ，その間の加算はどれも 3 本組の最下位 limb の ULP
            // （a0 に比例）ぶんの誤差を落とす．逐次に畳めばその ULP が S 分の 1 になる．
            // 本番の手続き・断片・対称剰余で frac を測る tools/twosum_probe.cpp では最悪 2^-66.6〜2^-67.7（既定の条件と値は
            // その注記）．畳む・畳まないの比較は簡略の模擬 tools/reconsim.cpp（3 本組・法 16 本）で，畳まない RED=0 の 2^-64.4 が
            // 畳む RED=1 で 2^-67.0 になる（reconsim 16 3 0・reconsim 16 3 1 で再現できる）．
            crt_proc::drop_integer(w);
        }
        // 畳み込み（像は [-1/2,1/2]）はライタが固定小数で行う（上の「小数部の畳み込み」）
        tile_regs_commit();
        tile_regs_wait();
        pack_fraction_limbs();
        tile_regs_release();
    }
}
