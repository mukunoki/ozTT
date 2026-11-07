// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：桁分割型のモジュラースライサ（INT8 以外の 7 形式で共通）
// ---------------------------------------------------------------------------
// GEMMul8（R-CCS）の FP8 経路に対応する実装を，BF16・FP16・FP32・BFP8_b・BFP4_b・BFP2_b
// にも使う．論文の ssec:oz2（尾崎スキーム II．剰余の桁分割）と ssec:acc（桁の上限）．以下の説明は FP8 を例にとるが，形式ごとに
// 違うのは桁の上限（FP8 16（疎な桁は 32），BF16・FP16・FP32・BFP8_b 32，BFP4_b 7，BFP2_b 1）と，
// それに応じたホスト側の法の選び方だけである．
//
// なぜ桁に分けるのか
//   FP8_E4M3 が整数を連続して無誤差に表せるのは [-16,16] の範囲だけなので，剰余
//   r = A' mod m（m は 2401 まで）を 1 つの FP8 値に入れられない．そこで対称剰余 r を
//   桁に分けて，桁ごとの行列積から積を組み立てる．方式は法によって (A)〜(F) の 6 通りある
//   （ozaki2_split.h の OZ2_SPLIT_*）．うち (A)(B) は行列積 3 本で済み，GEMMul8 の
//   法テーブルはすべてこのどちらかに当たる．(C) は窓の余りを詰めるために末尾の法を選び直したときか，
//   段の時間のモデルで法を選ぶ経路（ozaki2_awe.h）が 513 を超える非平方数を引いたときに現れる．
//   (D) は BFP2_b と BFP4_b，(E) は FP8，(F) は桁 32 の AWE 経路（BF16・FP16・BFP8_b）だけが使う．
//
//   (A) 平方数の法 m = q^2：最近接丸めの平衡桁
//         h = round(r/q),  l = r - q*h        （|l| <= q/2，|h| <= round((m/2)/q)）
//       q^2 ≡ 0 (mod m) なので上位×上位の項が消え，行列積は HL, LH, LL の 3 本：
//         r_a*r_b ≡ q*(h_a l_b + l_a h_b) + l_a l_b   (mod m)
//       出力は l, h の 2 枚．
//
//   (B) 非平方数の法：基数 B の Karatsuba（B は法ごとの実行時引数．桁の上限 d が 15 以上の経路（FP8 の d=16，
//       BF16・FP16・TF32・BFP8_b の d=32）は B = 16，BFP4_b（d=7）は B = d+1 = 8．ozaki2_split.h の oz2_kar_base）
//         h = sign(r)*ceil(|r|/B),  l = r - B*h,  s = h + l
//       切り上げにするので h と l が必ず逆符号になり，|h|,|l|,|s| がそろって max(ceil((m/2)/B), B-1) 以下に
//       収まる（B = 16 で 16 以下．最近接丸めだと |h+l| が 24 まで伸び，e4m3 の無誤差整数から外れる）．
//       行列積は HH, LL, SS の 3 本で，中間項は引き算で取り出す：
//         r_a*r_b = B^2*HH + B*(SS - HH - LL) + LL
//       出力は l, h, s の 3 枚．B = 16 は m <= 513，B = 8 は m <= 113 まで桁が収まる．
//
//   (C) Karatsuba で桁が収まらない非平方数の法：最近接丸めの平衡桁（q ≒ sqrt(m)．ホストの OZ2_SPLIT_WIDE）．
//       当たるのは，桁 16・32 の経路では 513 を超える非平方数（基数 16 の Karatsuba は m <= 513 までしか
//       桁を 16 に収められない．対称剰余は |r| <= floor(m/2) なので m = 513 でも |h| <= 16），
//       BFP4_b では基数 8 の Karatsuba に収まらない 113 < m <= 225 の非平方数の法である（BFP4_b の平方数は 225 = 15^2
//       以下がすべて (A)，256 以上が (D)．非平方数も 225 を超えると (D) の平衡 3 桁）．
//       上位×上位が消えないので行列積は HH, HL, LH, LL の 4 本になる：
//         r_a*r_b ≡ q^2*HH + q*(HL+LH) + LL   (mod m)
//       桁の作り方は (A) と同じなので，本カーネルから見ると (A) と区別が要らない
//       （違うのはホスト側の合成式だけ）．出力は l, h の 2 枚．
//
//   (E) 疎な桁の平方数の法（FP8 だけ．m = q^2，q は奇数で 33 < q <= 49）：(A) と同じ最近接丸めの桁を
//       作ってから，e4m3 が無誤差に表せる整数 S = {|x| <= 16} ∪ {偶数 x : 16 < |x| <= 32} へ移す
//       （ozaki2/kernels/oz2_sparse_fix.h．SFPI）．奇数で |t| >= 17 の下位桁は q を引いて偶数にし，上位桁へ
//       1 を繰り上げ，上位桁も同様に直す（q^2 ≡ 0 なので上位桁は法 q で動かしてよい）．積・合成は (A) と同じ．
//       出力は l, h の 2 枚（|l|,|h| <= 32）．
//
//   (F) 1 桁（桁 32 の AWE 経路．ホストの OZ2_SPLIT_ONE）：全剰余が桁の上限に収まる小さい法（m <= 65）は剰余を
//       そのまま桁行列 1 枚として出す．行列積 1 本で，合成段は通らない（再構成が GEMM の出力を法へ畳む．recon_crt.cpp）．
//
//   (D) 平衡 n 桁：基数 q = 2d+1 の平衡表現 r = Σ_j x_j q^j（|x_j| <= d）を，上位へ向かって順に剥がす：
//         x_j = cur - q*round(cur/q),  cur <- round(cur/q)
//       q が奇数なので round に同点が無く（cur/q が半整数にならない），表現は一意である．
//       出力は x_0..x_{n-1} の n 枚．合成の係数は q^(i+j) で，ホストが持つ．
//       桁数 n（1..5）は法ごとの実行時引数 NDIG[i]．BFP2_b（d=1，q=3）は法ごとに m <= 3^n の最小の n を使い
//       （5 桁なら積 25 本のところ，法 81 以下は 4 桁で積 16 本，27 以下は 3 桁で 9 本で済む），
//       BFP4_b（d=7，q=15）は 2 桁の方式で運べない 225 < m <= 3375 の法を 3 桁で運ぶ（docs/awe_moduli.md）．
//       最上位の桁は剥がし終えた cur そのもの（reg2）で，下位の桁は reg 0,1,3,5 に置く．
//
//   (B) の切り上げは round で作る：整数 u >= 0 に対し
//       ceil(u/q) = round((u + q/2 - 0.5)/q)
//   （右辺の引数は (u+q-1)/q - 1/2 + 1/(2q) で，round(x) = floor(x + 1/2) より右辺は
//   floor((u+q-1)/q + 1/(2q)) になる．(u+q-1)/q は 1/q 刻みなので 1/(2q) を足しても整数を
//   またがず，右辺は floor((u+q-1)/q) = ceil(u/q) に等しい．半整数にならないので丸めの
//   同点も起きない）．
//   round は奇関数なので符号は括り出せて h = round((r + sign(r)*(q/2-0.5))/q) となる．
//
// 出力は法ごとに 2 枚（(A)(C)(E)），3 枚（(B)），n 枚（(D)），1 枚（(F)）で，全体で nplane 枚．
// 並びはホストの poff[] と揃える．
//
// 引数とバッファ
//   コンパイル時引数 0 = S（法の本数），1 = A' を分ける基数 2^20 の float ビット列，
//     2 = A' の 2^20 進の桁数，3 = NRDIG（剰余を分ける桁の数の上限．2 か 5．平衡 n 桁の法は
//     2 桁の経路でも NDIG[i] 桁で運ぶ），
//     4 = OUT_HALF（1 なら桁を binary16 のビット列で出す．FP16 経路だけが 1）
//   実行時引数 0 = ntiles（残りは下の kernel_main の冒頭）
//   入力  桁を作る段の入力そのもの（oz2_digits.h と slice_tf_core.h を参照）
//   出力  c_16          : nplane 枚の桁行列（GEMM の入力形式．FP16 経路では UInt16）
// A' の 2^20 進の桁を作る段は INT8 経路と共通で oz2_digits.h にある．
#include "oz2_digits.h"
#include "oz2_sparse_fix.h"
#include "api/compute/eltwise_unary/typecast.h"

using namespace oz2_args;
constexpr auto cb_out = tt::CBIndex(CB_PLANE_OUT);

constexpr uint32_t S = get_compile_time_arg_val(SC_S);
// SC_NRDIG＝剰余を分ける桁の数の上限．2 なら上の (A)(B)(C)(E)(F)（BFP4_b は加えて実行時に (D) の 3 桁），5 なら (D) 平衡 n 桁（n は法ごとに 5 以下）．
constexpr uint32_t NRDIG = get_compile_time_arg_val(SC_NRDIG);
// 1 なら桁を IEEE binary16 のビット列（UInt16）で出力する（FP16 経路．sfpu_tf.h の HalfBits）．
// FP32 の DST から Float16 へ直接パックすると値が壊れるための回避である．
constexpr uint32_t OUT_HALF = get_compile_time_arg_val(SC_OUT_HALF);
constexpr uint32_t DF_F32   = (uint32_t)DataFormat::Float32;
constexpr uint32_t DF_U16   = (uint32_t)DataFormat::UInt16;
// 平衡 n 桁の枝（下の方式 (D)）は桁を binary16 へ直さない（HalfBits を通らずに continue する）．FP16 経路（桁 32）は
// 平衡 n 桁を使わない（ホストの oz2_split_mode は桁 15 以上の経路に BAL を当てない）．ただし NRDIG=2 でも XMODE=3 は
// 実行時に当たりうる（BFP4_b の平衡 3 桁）ので，下の static_assert は 5 桁の構成との組み合わせを禁じるだけで，
// FP16 経路に BAL の法が無いことはホスト（ozaki2_stages.hpp の build_slice）が確かめる．
static_assert(!OUT_HALF || NRDIG <= 2, "binary16 出力は 2 桁分割の経路だけで使う");
constexpr uint32_t MAXS = OZ2_MAXS;   // oz2_limits.h
static_assert(S <= MAXS, "法の本数が固定配列の大きさを超える");
// 桁の作り方の番号（oz2_limits.h の Oz2XMode）．番号を足したら下の枝を足してからここを直す（足さないと新しい番号の法は
// 既定の枝（最近接丸めの 2 桁）へ黙って落ちる）
static_assert(OZ2_XMODE_COUNT == 5, "桁の作り方の番号が増えた：kernel_main の枝を足す");
static_assert(NRDIG == 2 || NRDIG == 5, "桁の数の上限は 2（(A)(B)(C)(E)(F)，BFP4_b は加えて 3 桁の (D)）か 5（(D)．BFP2_b）");
// 実行時引数（各 S 個ずつ，法の順．並びは oz2_args.h の sl_*）:
//   ntiles，MOD，RINV = 1/m，A' の 2^20 進の桁の重みを法で畳んだ定数 2^(20k) mod m（k = 1..NDIGIT-1），
//   続いて Q（桁分割の基数），QINV = 1/q，XMODE（桁の作り方：oz2_limits.h の Oz2XMode．RND = 最近接丸めの 2 桁（平方数・WIDE），
//   KAR = Karatsuba の切り上げ（基数 q），SQS = (E) 疎な桁の平方数，BAL = (D) 平衡 n 桁，ONE = (F) 1 桁），NDIG（(D) の桁数 n．他は 0）

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(sl_ntiles());
    float MOD[MAXS], RINV[MAXS], C[MAXS][OZ2_NDIGIT - 1], Q[MAXS], QINV[MAXS], XMODE[MAXS];
    constexpr uint32_t A_MOD = sl_mod(S, 0), A_RINV = sl_rinv(S, 0);
    for (uint32_t i = 0; i < S; i++) { MOD[i] = fbf(get_arg_val<uint32_t>(A_MOD + i));
                                       RINV[i] = fbf(get_arg_val<uint32_t>(A_RINV + i)); }
    constexpr uint32_t A_W = sl_weight(S, 0, 0);   // 桁の重みの先頭．桁 j の法 i は A_W + j*S + i
    for (uint32_t i = 0; i < S; i++)
        for (uint32_t j = 0; j + 1 < OZ2_NDIGIT; j++)
            C[i][j] = fbf(get_arg_val<uint32_t>(A_W + j * S + i));
    constexpr uint32_t A_Q = sl_q(S, OZ2_NDIGIT, 0), A_QINV = sl_qinv(S, OZ2_NDIGIT, 0),
                       A_XMODE = sl_xmode(S, OZ2_NDIGIT, 0), A_NDIG = sl_ndig(S, OZ2_NDIGIT, 0);
    for (uint32_t i = 0; i < S; i++) { Q[i] = fbf(get_arg_val<uint32_t>(A_Q + i));
                                       QINV[i] = fbf(get_arg_val<uint32_t>(A_QINV + i)); }
    uint32_t NDIG[MAXS];
    for (uint32_t i = 0; i < S; i++) XMODE[i] = fbf(get_arg_val<uint32_t>(A_XMODE + i));
    for (uint32_t i = 0; i < S; i++) NDIG[i] = get_arg_val<uint32_t>(A_NDIG + i);
    for (uint32_t ti = 0; ti < ntiles; ti++) {
        // オペランドのタイル 1 枚から A' の桁を作る（正規化・整数化・桁分割）
        oz2_make_digits();
        for (uint32_t i = 0; i < S; i++) {
            tile_regs_acquire();
            init_sfpu(cb_dlo, cb_out);
            const float q = Q[i], qinv = QINV[i];
            // reg2 = A' mod m（対称剰余）．ここまでは INT8 経路と共通である．
            oz2_residue(MOD[i], RINV[i], C[i]);
            // 桁への分割．剰余を [0,m) へ戻さず対称剰余のまま分けるのが要点で，
            // これで桁の絶対値の上限が半分になり FP8 の無誤差整数に収まる．
            const bool kar = XMODE[i] == (float)OZ2_XMODE_KAR, sqs = XMODE[i] == (float)OZ2_XMODE_SQS,
                       bal = XMODE[i] == (float)OZ2_XMODE_BAL, one = XMODE[i] == (float)OZ2_XMODE_ONE;
            if (bal) {
                // 方式 (D)：平衡 n 桁（n = NDIG[i]，1..5）．1 回の acquire の中で n 枚すべてを作る
                // （DST は release でいったん手放すと内容が残らないため）．
                // reg2 が cur（最初は剰余），reg6 が商，reg7 が作業．桁の置き場 OUT は，
                // 下位 n-1 桁がこれらと重ならないレジスタで，最上位の桁は最後に残った
                // cur（reg2）そのものである（n=1 なら剰余そのものが桁）．
                constexpr uint32_t OUT[4] = {0, 1, 3, 5};
                const uint32_t nd = NDIG[i];
                for (uint32_t j = 0; j + 1 < nd; j++) {
                    Cpy(2, 6); MulS(6, qinv); Rnd(6);        // h = round(cur/q)
                    Cpy(6, 7); MulS(7, q); Sub(2, 7, OUT[j]);// x_j = cur - q*h
                    Cpy(6, 2);                               // cur <- h
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                for (uint32_t j = 0; j < nd; j++) {
                    cb_reserve_back(cb_out, 1); pack_tile(j + 1 < nd ? OUT[j] : 2u, cb_out); cb_push_back(cb_out, 1);
                }
                tile_regs_release();
                continue;                                    // 下の 2 桁用の処理は通らない
            }
            if (one) {
                // 方式 (F)：1 桁．剰余そのもの（|r| <= 32．桁 32 の AWE 経路の m <= 65 の法）を桁行列 1 枚にする
                Cpy(2,5);
            } else if (!kar) {
                // 方式 (A)/(C)：最近接丸めの平衡桁．h = round(r/q)，l = r - q*h．
                Cpy(2,6); MulS(6, qinv); Rnd(6);                              // hi → reg6
                Cpy(6,4); MulS(4, q); Sub(2,4,5);                             // lo → reg5
                // 方式 (E)：桁を e4m3 の無誤差整数（|x| <= 16 か 32 以下の偶数）へ移す
                if (sqs) Oz2SqsFix(5, 6, q);
            } else {
                // 方式 (B)：切り上げの Karatsuba 桁．h = sign(r)*ceil(|r|/q)，
                // l = r - q*h，s = h + l．符号を括り出した
                //   h = round((r + sign(r)*(q/2-0.5))/q)
                // をそのまま計算する（sign(r) は Ltz から作る）．
                Cpy(2,4); Ltz(4); MulS(4,-2.f); AddS(4,1.f);                  // sgn = ±1 → reg4
                MulS(4, q*0.5f - 0.5f);                                       // sgn*(q/2-0.5)
                Add(2,4,6); MulS(6, qinv); Rnd(6);                            // hi → reg6
                Cpy(6,4); MulS(4, q); Sub(2,4,5);                             // lo → reg5
                Add(6,5,7);                                                   // s = hi+lo → reg7
            }
            if constexpr (OUT_HALF) {
                // FP16 経路：桁（reg5 = lo，reg6 = hi，Karatsuba なら reg7 = s）を binary16 の
                // ビット列に等しい整数値にしてから UInt16 へ型変換する．この時点で reg0..4 は
                // もう使わないので，HalfBits の作業用に reg0,1,3 を渡す．桁は高々 ±32 で
                // HalfBits の契約 |d| <= 2048 の内側にある．
                HalfBits(5, 0, 1, 3);
                if (!one) HalfBits(6, 0, 1, 3);
                if (kar) HalfBits(7, 0, 1, 3);
                typecast_tile_init<DF_F32, DF_U16>();
                typecast_tile<DF_F32, DF_U16>(5);
                if (!one) typecast_tile<DF_F32, DF_U16>(6);
                if (kar) typecast_tile<DF_F32, DF_U16>(7);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            // 1 枚ずつ確保して押し出す．法ごとに 2 枚と 3 枚が混ざるので，まとめて
            // 確保すると CB のリング長（16）を割り切らず，境界をまたいだ書き込みが
            // 壊れる（recon_crt が深さを limb 数の倍数に取っているのと同じ理由）．
            auto emit = [&](uint32_t r) {
                cb_reserve_back(cb_out, 1); pack_tile(r, cb_out); cb_push_back(cb_out, 1);
            };
            emit(5);                 // lo 桁（1 桁の方式では剰余そのもの）
            if (!one) emit(6);       // hi 桁
            if (kar) emit(7);        // 和の桁（Karatsuba の第 3 行列）
            tile_regs_release();
        }
        oz2_digits_pop();
    }
}
