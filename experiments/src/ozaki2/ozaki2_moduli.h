// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：法の自動選択（ホスト側）
// ---------------------------------------------------------------------------
// 内積長 K と法の本数から，互いに素な法 m_i と，それが覆う CRT の窓から決まる
// スライス幅 b（既定の上限 54）を選ぶ．満たすべき条件は 2 つ．
//   (i)  行列積の生の総和（INT8 経路では |C_i| = K*r_max^2，桁分割の経路では桁積 K*d^2）が
//        無誤差に読める範囲に収まること
//   (ii) 法の積 M = Π m_i が 2*max|C'| を超えること（CRT の窓が足りること）
// あわせて，スライス／再構成カーネルが必要とする法ごとの定数もここで計算する．カーネルは固定の法集合を
// 埋め込まず，法ごとの定数を実行時引数で受け取る（法の本数・A' の桁数（b で決まる）・断片の本数などはコンパイル時
// 引数なので，それらが変われば JIT が組み直す）．論文の ssec:oz2（尾崎スキーム II．法と窓）と ssec:nofp64（FP32 を用いた再構成．
// 窓の超過を抑える末尾の選び直し）．
// 経路の表・法 1 本の運び方・桁積の列挙と連結は ozaki2_split.h，段の時間のモデルで法の集合を選ぶ部分は
// ozaki2_awe.h にあり，ここで取り込む（ドライバとホスト単体の検査・模擬はこのヘッダだけを取り込めばよい）．
#pragma once
#include <algorithm>
#include <array>
#include <iterator>
#include <cstdint>
#include <quadmath.h>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <vector>
#include <functional>
#include <string>
#include "ozaki2/ozaki2_split.h"   // 経路の表・桁分割の方式・桁積の列挙と連結（OZ2_ROUTES・oz2_param を含む）
#include "ozaki2/ozaki2_awe.h"     // 法の集合を段の時間のモデルで選ぶ（oz2_awe_choose）

// 再構成の逆数定数の分け方（分け方そのものは下の oz2_crt_constants）．
//   OZ2_RGRAN : 第 1 パスの共通粒度（2^-OZ2_RGRAN）．法ごとに整数部を畳むので累算は 1+RGRAN bit 強で 3 本組に収まる．本番の法の集合・
//               対称剰余の k_i・本番の足し込みで g=68 まで無誤差，69 で崩れ始める（tools/aligned_accum.cpp．64 は 4 bit の余裕）．
//   OZ2_NRP1/2: 各パスの 12 bit 断片の本数．どの本数も recon_crt.cpp の固定配列の長さ OZ2_MAXRP（kernels/oz2_limits.h）
//               以下であること（下の static_assert）．
#define OZ2_RGRAN 64
#define OZ2_NRP1 5
#define OZ2_NRP2 3
// 1 パス構成（既定）で使う，粒度を揃えない素の分解の断片数
#define OZ2_NRP0 5
static_assert(OZ2_NRP0 <= OZ2_MAXRP && OZ2_NRP1 <= OZ2_MAXRP && OZ2_NRP2 <= OZ2_MAXRP,
              "逆数の断片の本数が recon_crt.cpp の固定配列の長さ OZ2_MAXRP を超える");
// CRT の窓の条件 |C'| < M/2 は狭義である．等号を許すと |C'| = M/2 ちょうどのとき
// +M/2 と -M/2 がすべての法で同じ剰余を与え，CRT が両者を区別できない（再構成は両者に同じ値を返すので，
// 一方は必ず誤る）．たとえば法 {1089,1024}・K=34848・b=2 で全要素が
// 正規化後 4 になる入力がこれに当たる．そこで log2 での比較にこの幅を入れて等号を弾く．
// log2 の総和の丸め（10^-13 程度）より十分大きく，かつ「本当に足りている」場合に誤って
// 弾く幅（相対 7*10^-10）より十分小さい．法の刻みは最小でも 10^-4 程度あるので，この幅に
// 落ちるのは等号ちょうどの場合だけである．
#define OZ2_WIN_EPS 1e-9
// 末尾 2 本の選び直しで，行列積の本数が同じ候補どうしを「超過」ではなく「連結後の出力・桁行列の少なさ」で
// 比べてよい超過の上限［bit］（oz2_reselect_tail）．超過 1 bit で要素ごとの最大相対誤差が
// 約 1 bit 悪くなる（ノルム比は 4 bit まで動かない）ので，0.25 bit に抑える．
#define OZ2_TAIL_EXCESS_TOL 0.25
// 法選択へ渡す acc_bits：INT8 経路の GEMM の出力 C_i を再構成側が無誤差に読み戻せる幅．
// INT8 経路の C_i は符号-絶対値 int32 を上位 15 bit と下位 16 bit の 2 平面に分けて
// 復号する（recon_crt.cpp）ので 31 bit まで読める．桁分割の経路は桁積の条件
// K*d^2 <= 2^24 を別に見るので，この値は使わない（INT8 経路の候補の上限 oz2_candidate_table と累算の条件 oz2_check_accumulation だけが読む）．
#define OZ2_ACC_BITS 31
// 精度の目標［bit］．FP64 の仮数幅で，OZ2_FP64 の幅の見積り（ozaki2_autop.hpp の choose_precision_policy）・再構成のパス数の
// 条件（同じヘッダの OZ2_RECON1_THR_LOG2）・b が FP64 相当に届かない構成を止める閾値（ozaki2_plan.hpp の oz2_check_width）が引く
inline constexpr int OZ2_TARGET_BITS = 53;
// 幅 b の上限の既定（法の選択の b_cap．ドライバは OZ2_FP64・OZ2_B で上げる）．オペランドの仮数幅 OZ2_TARGET_BITS に，正規化後の
// 最大値が (0.5,1] に入ることを見込んだ 1 bit を足した値（理由と，上げると何が変わるかは oz2_width_from_window の注記）．
// scripts/verify/check_routes.py が lib/constants.tsv の OZ2_B_DEFAULT と "OZ2_B_DEFAULT = <数>;" の字面で突き合わせるので，数で書く
inline constexpr int OZ2_B_DEFAULT = 54;
static_assert(OZ2_B_DEFAULT == OZ2_TARGET_BITS + 1, "OZ2_B_DEFAULT は精度の目標に正規化の 1 bit を足した値");

struct Ozaki2Moduli {
    bool ok = false;                        // 必要な窓幅を満たす法集合が取れたか
    int s = 0;                              // 法の本数
    int nprod = 0;                          // 行列積の本数（法 1 本につき，2 桁分割は 3 本・WIDE のみ 4 本，1 桁は 1 本，
                                            // 平衡 n 桁は n^2 本（m が q^n を割れば n(n+1)/2 本）．INT8 経路は 1 本）
    int nplane = 0;                         // 桁行列の総枚数（法 1 本につき Karatsuba 3 枚，平衡 n 桁 n 枚，1 桁 1 枚，
                                            // 他は 2 枚）
    int maxg = 1;                           // 1 本の GEMM に K 方向で連結してよい桁積の数（oz2_product_groups）．
                                            // 法ごとの値 maxg_m の最大で，スライスの置き方（連結するか）を決める
    int smax = 0;                           // 実際に使った疎な桁の上限（FP8 は K <= 16381 で 32，それより大きい K と
                                            // 他の経路は 0．ozaki2_choose_moduli_w の冒頭）
    std::vector<int>   maxg_m;              // 法ごとの連結の上限．疎な桁の経路（FP8）と 1 桁の法を混ぜる経路
                                            // （Oz2Ndig::one）は法の桁の上限で法ごとに決まり（1 桁の法は 1），
                                            // 他の経路は全部 maxg と同じ（oz2_plan_concat）
    int nout = 0;                           // GEMM の本数（連結後．桁積 nprod 本が nout 本の出力にまとまる）
    bool fold_first = false;                // 合成段で桁積を先に法へ畳むか（oz2_compose.cpp）
    // 段の時間のモデルで法の集合を選んだ経過（ozaki2_awe.h の Oz2AweInfo の要約．CONFIG 行の awe・awe_caps 欄）．
    //   awe_status  "na"（その選び方を使わない経路・OZ2_MODULI），"model"（モデルで選んだ），
    //               "table_fallback"（見つからず表の選び方へ戻った）
    //   awe_caps    探索の打ち切りに当たったもの（vec・node・budget）と DRAM の条件を外した選び直し（dram_retry）を
    //               '+' 区切りで．当たらなければ "none"．打ち切りに当たった集合はモデルの上でも最適とは限らない
    std::string awe_status = "na", awe_caps = "none";
    bool acc_ok = true;                     // 累算の条件（桁積 K*d^2 と合成の総和．下の ok 判定）を
                                            // 満たすか．false なら ok も false で，原因は本数ではなく K
    // acc_ok を下ろした条件のすべて（落ちた順で，先頭が最初に落ちた条件．満たせば空）．呼び出し側のエラー文が形式・経路ごとの
    // 本当の原因を示し，条件ごとの K の上限の最小を「通る上限」として示す（最初に落ちた条件の上限だけを示すと，その K でも
    // 別の条件で落ちることがある）．why の値：
    //   "int8"     INT8 経路で法 m の GEMM の出力の上界 K*floor(m/2)^2（val）が 2^31-1 を超える
    //   "digit"    桁積の上界 K*d^2（val）が FP32 の無誤差整数 2^24 を超える
    //   "one"      1 桁の法 m の GEMM の出力の上界 K*R^2 に法を足した値（val．R は d）が 2^24 を超える
    //   "compose"  先に法へ畳めない法があり（出力 1 本の上界に法を足して 2^24 を超えるか，畳んだ後の総和が 2^22 以上），
    //              2^12 分割の合成の総和も法 m で 2^22 以上（val）になる
    struct AccFail { std::string why; int m = 0, d = 0; double val = 0.0; };
    std::vector<AccFail> acc_fails;
    // 候補の法の上限（INT8 経路は mod_cap と K*floor(m/2)^2 <= 2^acc_bits - 1 の小さいほう．他の経路は mod_cap）と，
    // 表（INT8 経路は下の延長を含む）から取れた候補の本数（窓が余って落とす前．呼び出し側が「候補を使い切ったので
    // 本数を増やしても b は戻らない」と「本数の指定が足りない」を見分けるため．集合を外から与えたか，段の時間の
    // モデルで選べたときは選んだ本数）
    int m_cap = 0, ncand = 0;
    int b = 23;                             // スライス基数のビット幅（精度のつまみ）
    std::vector<int>   m;                   // 法の集合（互いに素．上限は mod_cap：INT8 255，FP8 2401（疎な桁．K > 16381
                                            // では 1089），BF16・FP16・FP32・BFP8_b 4096，BFP4_b 3375，BFP2_b 243）
    std::vector<int>   q;                   // 桁分割の基数（平方数なら sqrt(m)，Karatsuba なら 16（BFP4_b は 8），
                                            // WIDE なら sqrt(m) 近傍，平衡 n 桁なら 2d+1（BFP2_b 3，BFP4_b 15），
                                            // 1 桁なら 1，INT8 経路は 0）
    std::vector<int>   mode;                // 桁分割の方式（OZ2_SPLIT_{SQ,KAR,WIDE,BAL,SQS,ONE}）
    std::vector<int>   poff;                // 法 i の桁行列が始まる枚目（先頭からの累積）
    std::vector<float> rinv;                // f32(1/m_i)
    std::vector<int>   yi;                  // (M/m_i)^{-1} mod m_i（CRT の係数）
    // 1/m_i を「k_i を掛けても無誤差」な 12 bit の断片列へ分けたもの（作り方と理由は
    // oz2_crt_constants）．rp1/rp2 は 2 パス構成（OZ2_FP64）の第 1・第 2 パス，
    // rp0 は既定の 1 パス構成が使う．
    //   rp1: 1/m_i を共通の絶対粒度 2^-OZ2_RGRAN へ丸めた R_i の断片（累算が無誤差に閉じる）
    //   rp2: 丸めの残り 1/m_i - R_i（両符号）の断片
    //   rp0: 粒度を揃えない素の分解．frac の絶対精度は 2^-67 前後で頭打ちになる（本番の手続きで frac を測る
    //        tools/twosum_probe.cpp で 2^-66.6〜2^-67.7．簡略の模擬 tools/reconsim.cpp では法 16 本の最悪 2^-67.0，24 本で 2^-65.7．
    //        reconsim 16 3 1・reconsim 24 3 1 で再現できる）が，一様乱数ではそれで FP64 相当に届く．
    std::vector<std::array<float, OZ2_NRP1>> rp1;
    std::vector<std::array<float, OZ2_NRP2>> rp2;
    std::vector<std::array<float, OZ2_NRP0>> rp0;
    std::vector<float> c0, c1, c2, c3;      // 2^20/2^40/2^60/2^80 mod m_i（桁の重み．c2 は 4 桁以上，c3 は 5 桁のときだけ）
    std::vector<float> m16;                 // 2^16 mod m_i（再構成での int32 C_i の 2 平面復号用）
    double KM = 0.0;                        // M * 2^-2b（最後に掛けるスケール）
    long double KMl = 0.0L;                 // 同じものを long double で（仮数 64 bit）．
                                            // デバイス側の出口はこれを 64 bit 仮数 +
                                            // 指数に分けて整数で掛ける（double のままだと
                                            // そこで相対 2^-53 が全要素に乗る）．
};

static inline float o2_f32(double x) { return (float)x; }
// 法 mod における a の逆元（a>0，gcd(a,mod)=1 が前提．拡張ユークリッドの互除法）
static inline long  o2_inv(long a, long mod) {
    long t = 0, nt = 1, r = mod, nr = a % mod;
    while (nr != 0) {
        const long q = r / nr;
        long tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr;      r = nr; nr = tmp;
    }
    if (t < 0) t += mod;
    return t;
}

// GEMMul8（RIKEN R-CCS）の INT8 法テーブル．256 から降順に互いに素なものを並べた
// もので，利用者は先頭 num_moduli 本を使う．本実装の貪欲探索も同じ系列を作るが，
// 参照実装と同じ表を明示的に持つほうが対応を確認しやすいのでそのまま持つ．
// 先頭の 256 だけは本実装では使えない：剰余を符号-絶対値の UInt8 で運ぶため，
// 対称代表元の下端 -128 が絶対値 128 となって 7 bit に収まらない（GEMMul8 は
// int8 の [-128,127] をそのまま使えるので制約されない）．
// 代わりに 254 = 2*127 を置く．対称代表元は [-127,127] で 7 bit に収まり，2 も 127 も
// 表の他の法に現れないので互いに素である．log2(254) = 7.99 は log2(256) = 8 とほぼ
// 同じなので，法の本数も窓の幅も GEMMul8 と一致する（K=4096・20 本でどちらも b=71）．
// この 1 本は b が 54 で頭打ちなら使われないが（窓が余るので落とされる），入力の動的域に
// 応じて b を上げると 8 bit ぶんの窓＝b の 4 bit としてそのまま精度に回る．
// GEMMul8 の表は 173 までの 20 本で終わる．本実装は b を 80 より上げる入力（OZ2_FP64）のために，
// 173 より下の素数 167, 163, 157, 151 を足して 24 本（カーネルの MAXS）まで延ばす．
// 既定（16 本）と b が 71 以下で済む入力は先頭 20 本の中で選ぶので，選ばれる集合は変わらない．
// 173 の直下の 169 = 13^2 は入れてはならない：247 = 13*19 と互いに素でなく，両方が採られると
// CRT が壊れる（下の ok 判定で総当たりの gcd を見ているので黙って誤答にはならないが，そもそも入れない）．
// K が 133144 を超えると GEMM の出力の条件 K*floor(m/2)^2 <= 2^31-1 で法の上限 m_cap が 255 より下がり，表の大きい法が
// 使えなくなる（K=262144 で 181 以下の 7 本だけ）．そのときは ozaki2_choose_moduli_w が表の後ろへ，m_cap と 150 の小さいほう
// から降順に，それまでに取った法と互いに素な整数を本数 min(num_moduli, OZ2_MAXS) まで足す（GEMMul8 の表と同じ「降順に
// 互いに素」の作り方．表の延長）．m_cap = 255（K <= 133144）では表の 24 本が全部使えるので足すことはなく，論文の大きさ
// （K <= 14080）を含めて選ぶ集合は変わらない．
static constexpr int OZ2_TABLE_I8[] = {
    255, /* 256 の代わりに */ 254, 253, 251, 247, 241, 239, 233, 229, 227, 223,
    217, 211, 199, 197, 193, 191, 181, 179, 173,
    /* ここから本実装の延長 */ 167, 163, 157, 151};
static constexpr int OZ2_TABLE_I8_MIN = 151;   // 表の最小（延長はこれより小さい法から足す）

// GEMMul8 の FP8 法テーブル．こちらは剰余を桁に分けて運ぶので法を 8 bit に収める
// 必要がなく，1089 から降順に互いに素なものが並ぶ．先頭 6 本が平方数
// （33^2, 32^2, 31^2, 29^2, 25^2, 23^2）であるのは偶然ではない．法 1 本あたりの
// 行列積を 3 本に抑える道が 2 通りあり，平方数かどうかで使い分けるためである
// （GEMMul8 の not_Karatsuba = 6 はこの平方数の本数）．詳細は下の桁分割を参照．
static constexpr int OZ2_TABLE_F8[] = {
    1089, 1024, 961, 841, 625, 529, 511, 509, 503, 499,
    491, 487, 481, 479, 467, 463, 461, 457, 449, 443};
// FP8 の疎な桁（sparse_max=32）で使う表．互いに素な平方数 q^2 を降順に並べる．q > 33 は奇数に限る
// （桁の集合 S_32 が法 q の完全剰余系になるのは q <= 33 と奇数の q <= 49．下の (E)）．
// 49=7^2，47，43，41，39=3*13，37，32=2^5，31，29，25=5^2，23，19，17 と 11 は互いに素で，
// 33^2（3*11）は 39^2 と 3 を，45^2（3^2*5）は 39^2 と 3・25^2 と 5 を，35^2（5*7）は 49^2 と 7 を共有するので入れない．
// 先頭 11 本で log2(Π m) = 112.9 あり，残りは末尾の選び直しが埋める（K=14080 では 977（WIDE）の 1 本で
// 行列積 37 本．表の順と選び直しの結果は docs/fp8_sparse_digits.md）．平方数の後ろの素数（Karatsuba）は
// 既定の 14 本では使わず，b を 54 より上げる自動決定（OZ2_FP64．法を最大 24 本まで取る）のためにある．
static constexpr int OZ2_TABLE_F8S[] = {
    2401, 2209, 1849, 1681, 1521, 1369, 1024, 961, 841, 625, 529, 361, 289, 121,
    509, 503, 499, 491, 487, 479, 467, 463, 461, 457};
// 桁を 16 より広く取れる経路（d=32）用の表．TF32（FP32）の選び方で，BF16・FP16・BFP8_b は段の時間のモデルの
// 選び方（ozaki2_awe.h）が窓に収まる集合を返さなかったときだけこれへ戻る．平方数 q^2 だけを並べる．
// 平方数は OZ2_SPLIT_SQ になって積が 3 本で済み，桁の上限は ceil(q/2) なので
// q <= 64 なら 32 以下に収まる．q を互いに素に取れば法も互いに素である
// （64=2^6，63=3^2*7，55=5*11，以降は 3・5・7・11 以外の素数）．
// 先頭 11 本で log2(Π m) = 123.6 あり，b=54・K=14080 が要求する 122.8 を覆う．
// ただし表の平方数だけで終わるとは限らない：下の末尾 2 本の選び直しは上限以下の奇数を
// 候補にするので，Karatsuba・WIDE の法も入る．末尾は行列積の本数 → 超過が許容内か → 連結後の出力と
// 桁行列の少なさの順に見て選ぶので，TF32 の K=14080 では平方数 9 本に 841・961 の 11 本，K=7040 は 289・1369，
// K=3520 は 361・529（どれも平方数）で，どれも行列積 33 本になる．
// 超過だけを見ると 307（KAR）・2423（WIDE）などを引いて 34 本になる．
static constexpr int OZ2_TABLE_B8[] = {
    4096, 3969, 3721, 3481, 3025, 2809, 2209, 1849, 1681, 1369, 961,
    841, 529, 361, 289, 169};

// 桁を 16 より狭くしか取れない経路（BFP4_b，d=7）用の表．BFP4_b は段の時間のモデルで法を選ぶ（ozaki2_awe.h）ので，
// これを使うのはモデルが窓に収まる集合を返さなかったときだけである（以下はこの表の選び方の説明）．
// 平衡 2 桁で桁が d に
// 収まる法は m <= (2d+1)^2 までなので，平方数 q^2（q <= 2d+1）を先に並べ，そのあとに
// 素数を降順に並べる．非平方数は基数 16 の Karatsuba だと桁が 15 まで伸びて d=7 に
// 収まらないので，平衡桁（WIDE，積 4 本）で運ぶ．平方数 4 本＋素数 13 本の 17 本で
// log2(Π m) = 127.4 あり，b=54・K=14080 が要求する 122.8 を覆う（積は計 64 本）．
static constexpr int OZ2_TABLE_B4[] = {
    225, 196, 169, 121, 223, 211, 199, 197, 193, 191, 181, 179, 173,
    167, 163, 157, 151, 149, 139, 137, 131, 127};

// 桁が ±1 しか取れない経路（BFP2_b，d=1）用の表．BFP4_b と同じく，段の時間のモデルが窓に収まる集合を返さなかった
// ときだけ使う（以下はこの表の選び方の説明）．2 桁では法が (2d+1)^2 = 9 までで
// 窓に遠く届かないので，平衡三進法（基数 q = 2d+1 = 3）の n 桁へ広げる．n 桁で法は
// q^n まで扱えるので，n=5 なら 243 まで．下の 16 本で log2(Π m) = 124.6 あり，
// b=54・K=14080 が要求する 122.8 を覆う．法 243 = 3^5 だけは i+j >= 5 の項が
// 消えて積が 15 本，残りは 25 本ずつで計 390 本になる．
static constexpr int OZ2_TABLE_B2[] = {
    243, 242, 241, 239, 235, 233, 229, 227, 223, 221, 217, 211, 199, 197, 193, 191,
    181, 179, 173, 167};

// INT8 経路の GEMM の出力 C_i の上界 K*floor(m/2)^2 が符号-絶対値 int32 の絶対値の最大 2^acc_bits - 1 に収まるか
// （整数演算．剰余は対称剰余で |r| <= floor(m/2)）．
static inline bool oz2_int8_fits(uint32_t K, int m, int acc_bits) {
    const unsigned __int128 r = (unsigned)(m / 2);
    return (unsigned __int128)K * r * r <= (((unsigned __int128)1 << acc_bits) - 1);
}
// 上の条件を満たす最大の |r|（K*r^2 <= 2^acc_bits - 1）．
static inline long oz2_int8_rmax(uint32_t K, int acc_bits) {
    const unsigned __int128 lim = ((unsigned __int128)1 << acc_bits) - 1;
    long r = (long)std::floor(std::sqrt((double)lim / (double)K));
    while (r > 0 && (unsigned __int128)K * r * r > lim) --r;
    while ((unsigned __int128)K * (r + 1) * (r + 1) <= lim) ++r;
    return r;
}

// --- 法の選択の手順（下の ozaki2_choose_moduli_w が書いた順に 1 度ずつ呼ぶ）-----------------------------------------
// 経路の条件．選択のあいだ変わらない（smax と mod_cap は，FP8 の大きい K で稠密な桁へ戻した後の値）
struct Oz2SelRoute {
    uint32_t K;
    int mod_cap, acc_bits, digit_max, smax;
    Oz2Ndig ndig;
    // 桁分割では，桁が digit_max に収まる法しか使えない．FP8（d=16）なら平方数は m <= 1089 まで，
    // 非平方数は m <= 513 なら基数 16 の Karatsuba，それを超えると平衡桁（WIDE，積 4 本）
    // になり m <= 1089 程度まで使える（d=32 ならどちらも 4096 程度まで）．どの方式で見るかは
    // oz2_digit_bound が判断する．
    bool usable(int mm) const { return oz2_usable(mm, digit_max, ndig, smax); }
    int nprod(int mm) const { return digit_max ? oz2_digit_prods(mm, digit_max, ndig) : 1; }   // 法 1 本の行列積の本数
};

// 候補の表と法の上限 m_cap．桁分割の経路では行列積が見るのは剰余そのものではなく桁（|h|,|l| <= digit_max）なので，
// 総和の上限は K*digit_max^2 であって法の大きさでは律速されない．したがって法を mod_cap より絞る必要はなく，
// 代わりに oz2_check_accumulation がこの総和を見る．
struct Oz2Table { const int* t; int n; };
static inline Oz2Table oz2_candidate_table(const Oz2SelRoute& R, int& m_cap) {
    auto table = [](const auto& t) { return Oz2Table{t, (int)std::size(t)}; };
    m_cap = R.mod_cap;
    // 平衡 n 桁の経路（BFP2_b）．法は q^n まで
    if (R.ndig.n > 2) return table(OZ2_TABLE_B2);
    // 疎な桁の経路（FP8．(E)）．奇数 q <= 49 の平方数まで 3 本で運べる
    if (R.smax) return table(OZ2_TABLE_F8S);
    // 桁を広く取れる経路は法も広く取れる（平方数の表．末尾の選び直しで非平方数も入る．
    // 合成は K*d^2+m <= 2^24 のあいだ fold_first）
    if (R.digit_max > 16) return table(OZ2_TABLE_B8);
    // 桁が 16 より狭い経路（BFP4_b）．法は (2d+1)^2 までで，非平方数は WIDE で運ぶ
    if (R.digit_max && R.digit_max < 16) return table(OZ2_TABLE_B4);
    if (R.digit_max) return table(OZ2_TABLE_F8);                        // FP8（d=16）
    // INT8 経路は剰余をそのまま掛けるので，|C_i| = K*rmax^2 が再構成側で無誤差に
    // 読める範囲に収まる法だけを使う．上限は 2^acc_bits ではなく 2^acc_bits - 1：GEMM の出力は
    // 符号-絶対値 int32 で，絶対値は 31 bit（2^31 - 1）までしか載らない（K=2^19・法 129 で
    // |r| = 64 の全要素 1 なら K*64^2 = 2^31 ちょうどになり，格納できずに剰余が壊れる）．
    // 浮動小数の sqrt の丸めに頼らず整数で最大の rmax を求める．
    m_cap = (int)std::min<long>(R.mod_cap, 2 * oz2_int8_rmax(R.K, R.acc_bits) + 1);
    return table(OZ2_TABLE_I8);
}

// 初めの候補を o.m に置き，log2 Π m を logM に返す．集合を外から与えた（force）か段の時間のモデルで選べた（awe）なら
// その集合のまま使い，末尾を落とす・選び直すを通らない（戻り値 true）．そうでなければ表の先頭から num_moduli 本と，
// INT8 経路では表の延長．与えた集合に使えない法があれば force_bad を立てる
static inline bool oz2_pick_candidates(Ozaki2Moduli& o, double& logM, bool& force_bad, const Oz2SelRoute& R, Oz2Table tb,
                                       int m_cap, int num_moduli, int b_cap, double Wlog2, int awe,
                                       const std::vector<int>* force, uint32_t Mdim, uint32_t Ndim) {
    bool fixed_set = false;
    logM = 0.0; force_bad = false;
    if (force && !force->empty()) {
        for (int mm : *force) { o.m.push_back(mm); logM += std::log2((double)mm); if (mm < 2 || mm > m_cap || !R.usable(mm)) force_bad = true; }
        fixed_set = true;
    } else if (awe && R.digit_max) {
        const double need_awe = 2.0 * std::min(b_cap, OZ2_MAXDIGIT * OZ2_DIGIT_BITS) + 1.0 + Wlog2 + OZ2_WIN_EPS;
        Oz2AweInfo ai;
        const std::vector<int> ms = oz2_awe_choose(R.K, awe, m_cap, R.digit_max, R.ndig, R.smax, need_awe,
                                                   OZ2_TAIL_EXCESS_TOL, std::min(num_moduli, OZ2_MAXS), Mdim, Ndim, &ai);
        o.awe_status = ai.found ? "model" : "table_fallback";
        std::string caps;
        auto addc = [&](bool f, const char* nm) { if (f) caps += (caps.empty() ? "" : "+") + std::string(nm); };
        addc(ai.vec_cap, "vec"); addc(ai.node_cap, "node"); addc(ai.budget, "budget"); addc(ai.dram_retry, "dram_retry");
        o.awe_caps = caps.empty() ? "none" : caps;
        if (!ms.empty()) {
            for (int mm : ms) { o.m.push_back(mm); logM += std::log2((double)mm); }
            fixed_set = true;
        }
    }
    if (!fixed_set)
        for (int t = 0; t < tb.n; ++t) {
            int mm = tb.t[t];
            if ((int)o.m.size() >= num_moduli) break;
            if (mm > m_cap || !R.usable(mm)) continue;
            o.m.push_back(mm); logM += std::log2((double)mm);
        }
    // INT8 経路の表の延長（OZ2_TABLE_I8 の注記）：K が大きくて m_cap 以下の表の法が本数に足りないときは，151 未満の法を
    // 降順に，取った法と互いに素なものだけ足す．延ばさないと K=262144 は 181 以下の 7 本で止まり，b が 16 まで下がる．
    // 足すのは表を使い切ったときだけで，窓が余れば oz2_trim_tail が後ろ（延長の側）から落とす．
    if (!fixed_set && !R.digit_max) {
        const int want = std::min(num_moduli, OZ2_MAXS);
        for (int mm = std::min(m_cap, OZ2_TABLE_I8_MIN - 1); mm >= 3 && (int)o.m.size() < want; --mm) {
            bool cop = true;
            for (int x : o.m) if (std::gcd(x, mm) != 1) { cop = false; break; }
            if (!cop) continue;
            o.m.push_back(mm); logM += std::log2((double)mm);
        }
    }
    o.m_cap = m_cap;
    o.ncand = (int)o.m.size();
    o.s = (int)o.m.size();
    return fixed_set;
}

// スライス幅 b を CRT の窓から決める（GEMMul8 が法の本数からシフト量を導くのと
// 同じ向き）．行 i・列 j の対称なシフトを e_A(i), e_B(j) とすると，正規化後の
// 最大値が 1 以下なので
//     |C'| <= K * 2^(e_A+e_B) < M/2
// を満たせばよい．両者を等しく取れば e = b = floor((log2 M - 1 - log2 K)/2) で，
// このとき frac = C'/M は最大 1/2 近くまで使われる．窓を余らせるのも b を
// 欲張って窓を溢れさせるのも損なので，この b が唯一の正しい点である．
// 窓の条件は狭義（OZ2_WIN_EPS の説明）なので，b の上限も EPS ぶん厳しく採る．
// これで等号ちょうどの組み合わせでは b が 1 下がり，エラーにならずに済む．
// 上限が 2 つある．1 つは A' を 5 桁で送る形式が運べる幅（OZ2_MAXDIGIT*OZ2_DIGIT_BITS = 100）．もう 1 つは
// 呼び出し側が渡す b_cap で，既定は 54 である．これはオペランドの仮数幅に，正規化後の最大値が
// (0.5,1] に入ることを見込んだ 1 bit を足した値である．
// この 54 は「FP64 相当で打ち切る」という設計判断であって無損失ではない．
// 行内の全要素が共通の 2^-53 格子に載る入力（一様乱数など）なら b を 54 より
// 上げても捕捉する情報は増えないが，行内で指数が広がる入力では分割そのものが
// b に比例して伸び続ける（厳密演算での模擬 tools/splitsim.cpp で φ=2・b=60 が 56.5 bit）．
// そちらを取りにいくのが呼び出し側の自動決定（ozaki2_autop.hpp の choose_precision_policy．OZ2_FP64）で，
// 入力から必要な b を求めて b_cap に渡す．GEMMul8 も本数から同じ向きに b を導く．
// 調査用の上書き（OZ2_B）も呼び出し側が b_cap へ入れて渡す：ここで getenv を
// 読むと，同じ引数で呼んでも環境で答えが変わり，ホスト単体の検査が成り立たない．
static inline int oz2_width_from_window(double logM, double Wlog2, int b_cap) {
    const double room = logM - 1.0 - Wlog2 - OZ2_WIN_EPS;
    const int b_win = (int)std::floor(room / 2.0);
    return std::min(std::min(b_win, OZ2_MAXDIGIT * OZ2_DIGIT_BITS), b_cap);
}

// b が仮数側（54）で頭打ちになると，指定された本数ぶんの M は窓に対して余る．
// 余りは精度の足しにならない（ozaki2_choose_moduli_w の need_log2 の注記のとおり frac が小さくなるだけ）ので，
// 窓を満たしたまま落とせる法は後ろから落とす．法 1 本＝行列積 1 本（桁分割では 3〜4 本か
// 15〜25 本）ぶんのコストである．FP8 経路は法が 10 bit あるため 16 本の指定でも
// 実際には 13〜14 本で足りる．
static inline void oz2_trim_tail(std::vector<int>& m, double& logM, double need_log2) {
    while (m.size() > 2) {
        double l2 = std::log2((double)m.back());
        if (logM - l2 < need_log2) break;
        m.pop_back(); logM -= l2;
    }
}

// b が仮数側で頭打ちになったぶん M に余りが出るので，末尾 2 本を選び直して
// 必要量を最小の超過で満たす互いに素な対に置き換える．表の刻みは 1 本あたり
// 約 7.6 bit あり，そのままではその幅ぶん frac の精度を捨ててしまう．
// これは再構成が frac = C'/M を固定の絶対精度で解く本実装に固有の調整で，
// GEMMul8（FP64 の CRT）には対応物がない．
// 選ぶ順は「末尾の行列積の本数が少ない → 超過が許容（OZ2_TAIL_EXCESS_TOL）以下 → 出力と桁行列の
// 数が少ない → 超過が小さい」の辞書順である（下の better）．本数を最初に見るのは，桁分割では法ごとに
// 行列積が 3 本（平方数・Karatsuba）か 4 本（WIDE）で，超過の最小だけを見ると 4 本の法を
// 引きうるためである（d=32 の K=14080 で 307k・2423w の 7 本）．本数を先に見れば 841*・961* の 6 本で
// 済む（超過は 0.12 bit．要素ごとの最大相対誤差がほぼその分だけ悪くなりうるが，ノルム比は
// 動かない．docs/nongemm_opt_results.md の §6）．INT8 経路は法 1 本が行列積 1 本なので，
// 本数は法の本数で，1 本で足りる候補があればそちらを選ぶ．
static inline void oz2_reselect_tail(std::vector<int>& m, double& logM, double need_log2, const Oz2SelRoute& R, int m_cap) {
    if (!(m.size() >= 2 && logM > need_log2)) return;
    const uint32_t K = R.K;
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    std::vector<int> head(m.begin(), m.end() - 2);
    double base = 0.0; for (int x : head) base += std::log2((double)x);
    auto cop_head = [&](int mm) {
        for (int x : head) if (std::gcd(mm, x) != 1) return false;
        return true;
    };
    double best = logM; int bp = m[m.size()-2], bq = m.back();
    // 本数が同じ候補どうしは，超過が OZ2_TAIL_EXCESS_TOL（0.25 bit）以下なら超過で比べず，桁積を
    // K 方向に連結した後の GEMM の出力の本数（oz2_nout）→ 桁行列の枚数の少ないほうを採る．出力が
    // 減れば GEMM の発行と合成が読む平面が，桁行列が減ればスライスの書き出しが減る（3520³ の TF32 では
    // 超過だけで比べると末尾が 379k・491k（出力・桁行列 6）になるところを 361*・529*（4）にする．
    // 14080³ は maxg=1 なので変わらない．docs/nongemm_opt_results.md の §10）．
    // 超過はそのまま精度の損失なので上限を置く（上限を置かないと
    // FP8 の 14080³ が超過 4.6 bit の平方数 2 本を選ぶ）．maxg は法の上限 m_cap で見積もる
    // （実際の maxg は選んだ法の最大で決まり，これ以上になる）．
    // 疎な桁の経路（smax）と 1 桁の法を混ぜる経路（Oz2Ndig::one）は法ごとに桁の上限が違う（SQS は 18〜32，
    // 他は 16 以下．桁 32 の経路は桁行列ごとの上限）ので，連結の本数も法ごとに見積もる（l と h の積の上界．
    // oz2_plan_concat の maxg_m と同じ関数 oz2_maxg_for で，引く法だけが違う）．
    auto maxg_est_of = [&](int mm) { return digit_max ? oz2_maxg_for(mm, K, m_cap, digit_max, ndig, smax) : 1; };
    auto layout_cost_raw = [&](int mm) {
        if (!mm || !digit_max) return 0;
        return 1000 * (int)oz2_nout(mm, digit_max, ndig, maxg_est_of(mm)) + oz2_digit_planes(mm, digit_max, ndig);
    };
    // 候補ごとの値は対の総当たりの内側で何度も引くので，先に表にしておく（値は上の関数そのもの）
    std::vector<char> use_c(m_cap + 1, 0);
    std::vector<int> np_c(m_cap + 1, 0), lay_c(m_cap + 1, 0);
    for (int mm = 3; mm <= m_cap; ++mm)
        if (R.usable(mm)) { use_c[mm] = 1; np_c[mm] = R.nprod(mm); lay_c[mm] = layout_cost_raw(mm); }
    auto layout_cost = [&](int mm) { return (mm >= 3 && mm <= m_cap && use_c[mm]) ? lay_c[mm] : layout_cost_raw(mm); };
    auto usable_c = [&](int mm) { return mm >= 3 && mm <= m_cap && use_c[mm]; };
    auto nprod_c = [&](int mm) { return (mm >= 3 && mm <= m_cap && use_c[mm]) ? np_c[mm] : R.nprod(mm); };
    auto over_tol = [&](double l) { return (l - need_log2) > OZ2_TAIL_EXCESS_TOL ? 1 : 0; };
    int bcost = R.nprod(bp) + R.nprod(bq);              // 今の末尾の行列積の本数
    int blay = layout_cost(bp) + layout_cost(bq);        // 今の末尾の出力・桁行列
    // (本数, 超過が許容を超えるか, 出力・桁行列, logM) の辞書順で今より良ければ置き換える
    auto better = [&](int c, int lay, double l) {
        if (c != bcost) return c < bcost;
        if (over_tol(l) != over_tol(best)) return over_tol(l) < over_tol(best);
        return lay < blay || (lay == blay && l < best);
    };
    // 残す先頭部分に偶数の法があれば候補を奇数に限る：偶数の候補は 2 を共有して必ず弾かれるので，
    // 調べるだけ無駄である（どの表も先頭近くに偶数の法を持つ：INT8 254，FP8 1024，d=32 の表 4096，
    // BFP4_b 196，BFP2_b 242．既定の本数ではいつもこちら）．先頭に偶数が無い（先頭が短い）ときは
    // 偶数も候補にする：法がそこまで少なくても窓が足りることがあり（INT8・K=64・法 2 本では b=4），
    // 奇数だけに限ると 147*223=32781 を選んで，超過の小さい 145*226=32770 を取りこぼす．
    // 3 から総当たりするのは，本数を増やさずに超過を最小にする組を探すためで，
    // 小さい法ほど log2 の刻みが細かく必要量に寄せやすい．
    bool head_even = false; for (int x : head) head_even = head_even || (x % 2 == 0);
    const int pstep = head_even ? 2 : 1;
    for (int p2 = 3; p2 <= m_cap; p2 += pstep) {
        if (!usable_c(p2) || !cop_head(p2)) continue;
        double lp = base + std::log2((double)p2);
        if (lp >= need_log2 && better(nprod_c(p2), layout_cost(p2), lp)) {
            best = lp; bp = p2; bq = 0; bcost = nprod_c(p2); blay = layout_cost(p2); }
        // 対は積で窓に寄与するので (p,q) と (q,p) は同じ．q は p2 より上だけ見ればよい．
        for (int q = p2 + pstep; q <= m_cap; q += pstep) {   // 先頭に偶数があれば奇数だけ（p2 も奇数）
            if (!usable_c(q) || std::gcd(p2, q) != 1 || !cop_head(q)) continue;
            double lq = lp + std::log2((double)q);
            const int c = nprod_c(p2) + nprod_c(q), lay = layout_cost(p2) + layout_cost(q);
            if (lq >= need_log2 && better(c, lay, lq)) { best = lq; bp = p2; bq = q; bcost = c; blay = lay; }
        }
    }
    m = head; m.push_back(bp); if (bq) m.push_back(bq); logM = best;
}

// 累算の条件で落とす（acc_fails の末尾に足し，ok と acc_ok を下ろす）
static inline void oz2_acc_fail(Ozaki2Moduli& o, const char* why, int mm, int dd, double v) {
    o.acc_fails.push_back({why, mm, dd, v});
    o.ok = o.acc_ok = false;
}

// 累算の条件（合成段の条件は oz2_plan_concat）．どの条件で落ちても acc_ok を下ろし，落ちた条件を順に acc_fails に残す
// （呼び出し側が「本数ではなく K が原因」と，形式・経路ごとの原因を示せるように．ozaki2_plan.hpp の oz2_report_moduli_failure）．
// INT8 経路：選んだ（または与えられた）法ごとに，GEMM の出力の上界 K*floor(m/2)^2 が符号-絶対値 int32 の
// 絶対値 2^acc_bits - 1 に収まることを整数演算で確かめ直す（m_cap と同じ条件だが，表・末尾の選び直し・
// OZ2_MODULI のどこから来た法でも，浮動小数を介さずに保証する）．INT8 のスライサの剰余は対称剰余そのもの
// （|S| < 2^22 なので ModRound の一般の保証の内側．ozaki2/kernels/oz2_digits.h）で |r| <= floor(m/2)．
// 桁分割では行列積の総和が |Σ| <= K*digit_max^2 になる．桁積は FP32 に
// 累算されるので，これが無誤差整数の範囲 2^24 に収まっていなければならない．
// 上限は 2^24 ちょうどまで許してよい（FP32 は整数 2^24 を厳密に表せる）．
// digit_max=16（FP8）なら K <= 65536，32（BF16・FP16・FP32・BFP8_b）なら K <= 16384 まで
// 通る．scheme II のスライス段の CB は K に依存しない固定長なので，scheme I のような
// L1 由来の K 上限は無く，大きな K ではこの条件（と合成の条件）で最初に落ちる．
// 桁の上限 32 の法（4096 = 64^2 など）を用いると K=16384 ちょうどで先に法へ畳めなくなり（oz2_plan_concat の can_fold は
// P+m <= 2^24），2^12 分割の合成も総和が 2^22 を超えるので，実際には K < 16384 までである（表の選び方の TF32
// は K >= 16384 で必ず落ちる）．段の時間のモデルで法を選ぶ BF16・FP16・BFP8_b は，K が大きいと桁の上限 32 の法を
// 外して 31 以下の法を選ぶので K > 16352 も通りうるが，通る K は散在する（モデルの候補の検査（法の上限 mod_cap で
// 見積もる）とここ（選んだ集合の桁の上限の最大 deff で K*deff^2 <= 2^24 を見る）の上界の取り方が違い，モデルが
// 選んだ集合の deff が K に対して大きすぎる区間で落ちる．モデルは M・N から DRAM の見積りを立てるので，通る K は
// M・N でも変わる．K の区間の一覧は docs/awe_moduli.md の「K > 16352」）．
// 疎な桁の経路（smax．FP8）は法ごとの桁の上限（oz2_plane_bound．SQS は最大 32）の最大で見る．
// 1 桁の法は合成段を通らず，再構成が GEMM の出力 P（|P| <= K*R^2．R は oz2_resid_bound）を ModRound 1 回で法へ畳む
// （recon_crt.cpp の RAWMASK）．剰余類が保たれるには，丸めた商と法の積が FP32 に載るよう |P| + m <= 2^24 が要る
// （先に法へ畳む合成 fold_first と同じ形の余裕．tools/modround_check.cpp の (e) が S <= 2^24 - m を総当たりする）．
// 段の時間のモデルの選び方はこれを満たす法しか候補にしないが，表の選び方・与えた集合（OZ2_MODULI）でも保つ．
static inline void oz2_check_accumulation(Ozaki2Moduli& o, const Oz2SelRoute& R) {
    const uint32_t K = R.K;
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    if (!digit_max)
        for (int mm : o.m)
            if (!oz2_int8_fits(K, mm, R.acc_bits))
                oz2_acc_fail(o, "int8", mm, 0, (double)K * (double)(mm / 2) * (double)(mm / 2));
    auto pbound = [&](int mm, int plane) { return oz2_plane_bound(mm, plane, digit_max, ndig, smax); };
    int deff = digit_max;
    if (smax || ndig.one) {
        deff = 0;
        for (int mm : o.m)
            for (int pl = 0; pl < oz2_digit_planes(mm, digit_max, ndig); ++pl) deff = std::max(deff, pbound(mm, pl));
    }
    if (digit_max && (double)K * deff * deff > std::ldexp(1.0, 24)) oz2_acc_fail(o, "digit", 0, deff, (double)K * deff * deff);
    if (digit_max && ndig.one)
        for (int mm : o.m)
            if (oz2_split_mode(mm, digit_max, ndig, smax) == OZ2_SPLIT_ONE) {
                const double Rb = (double)oz2_resid_bound(mm);
                if ((double)K * Rb * Rb + mm > std::ldexp(1.0, 24)) oz2_acc_fail(o, "one", mm, (int)Rb, (double)K * Rb * Rb + mm);
            }
}

// 桁積の連結の上限（法ごとの maxg_m と経路の maxg）と，合成段の方式（fold_first）・条件．
// 合成段（ozaki2/kernels/oz2_compose.cpp）の方式は 2 つあり，どちらも最後の ModRound の
// 入力になる総和が，ModRound が正しい対称剰余を返す 2^22（sfpu_tf_proc.h の ModRound）未満でなければ
// ならない．係数は法ごとに大きく違うので，m と np だけの粗い上界ではなく実際の係数で見る．
//   fold_first : 桁積 P_j を先に ModRound で法へ畳んでから生の係数 c_j を当てる．
//                P_j に法 m を足しても FP32 の無誤差整数 2^24 に収まることが要る．
//                P_j < 2^24 だけでは足りない：丸めた商と法の積は |P_j|+m/2 強に達し，
//                これが 2^24 を超えると積が丸まって剰余類が変わる（d=1・K=2^24-32・m=229 で
//                P=16777184 が -44 になる．正しくは -43）．スライサの検査
//                （tools/modround_check.cpp の Smax+m）と同じ形の余裕である．
//                畳んだあとは |t_j| <= m/2+2 なので総和は (Σ|c_j|)(m/2+2) で済む．
//   2^12 分割  : 上が成り立たない場合だけこちらに落ちる．P_j を 2^12（OZ2_COMPOSE_SPLIT）の上位 h_j と下位 l_j に
//                分けて A_j = c_j*2^12 mod m，B_j = c_j mod m を当てるので，総和は
//                Σ_j (|h_j|*A_j + |l_j|*B_j) 以下で，h_j は最近接へ丸める（oz2_compose.cpp の Rnd）ので
//                |h_j| <= floor(Pmax/2^12 + 1/2)（Pmax は出力 1 本の上界 oz2_group_bound．経路一律の桁の上限なら
//                K*digit_max^2 で，桁 16 なら K/16 を丸めた値），|l_j| <= 2^11 である．FP8（d=16）なら K=16384 でも
//                最大 2^21.2 で，最初に破るのは K=45024 である．
// 桁積の連結（ozaki2_split.h の oz2_product_groups）：1 本の出力 g 本ぶんの和 g*K*d^2 に法を足しても 2^24 以下なら
// まとめる．まとめた結果で合成の条件が崩れるなら（起きないが），まとめずに（maxg=1）判定し直す．
// 疎な桁の経路（smax．FP8）と 1 桁の法を混ぜる経路（Oz2Ndig::one．桁 32 の AWE 経路）は法ごとに決める（maxg_m）：
// 2 桁分割でまとまるのは位 1 の l×h の 2 本だけなので，上界は g*K*d_l*d_h（d_l, d_h は法の l・h の桁の上限
// oz2_plane_bound．FP8 は SQS の 49^2 なら 32・32 で K=14080 では 1，SQ の 16 以下の法は 4．桁 32 の経路は
// 4096 で 32・32，3721 で 30・31）．1 桁の法は連結しないので 1．他の経路（TF32・BFP4_b・BFP2_b）は全部の法で
// 経路一律の g*K*d^2 から決める．
// 出力 1 本の上界は，連結した g 本それぞれの桁の上限の積の和 Σ_t K*d(a)*d(b)（smax=0 なら g*K*d^2）．
static inline void oz2_plan_concat(Ozaki2Moduli& o, const Oz2SelRoute& R) {
    const uint32_t K = R.K;
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    int mmax = 0; for (int mm : o.m) mmax = std::max(mmax, mm);
    o.maxg_m.assign(o.s, 1);
    if (digit_max)
        for (int i = 0; i < o.s; ++i) {
            // 1 桁の法は桁積が 1 本で連結しようがないので 1 にする（大きな値にしておくと経路の maxg がこの法で決まり，
            // 実際には連結しない構成でもスライスが連結用の置き方になる）
            if (oz2_split_mode(o.m[i], digit_max, ndig, smax) == OZ2_SPLIT_ONE) continue;
            // 疎な桁の経路（FP8）と 1 桁の法を混ぜる経路（Oz2Ndig::one）は法ごとの桁の上限で（ozaki2_split.h の oz2_maxg_for）
            o.maxg_m[i] = oz2_maxg_for(o.m[i], K, mmax, digit_max, ndig, smax);
        }
    auto sync_maxg = [&]() { o.maxg = 1; for (int g : o.maxg_m) o.maxg = std::max(o.maxg, g); };
    sync_maxg();
    if (!digit_max) return;
    auto group_bound = [&](int mm, const Oz2ProductGroup& g) {
        return oz2_group_bound(mm, g, (double)K, digit_max, ndig, smax);
    };
    // 法 i について，fold_first なら (Σ c)(m/2+2)，そうでなければ 2^12 分割の総和の上界．
    // 項は連結後の出力 1 本ごとで，上界は group_bound（smax=0 なら g*K*d^2）
    auto compose_sum = [&](int i, bool fold) {
        const int mm = o.m[i];
        Oz2ProductGroup gr[OZ2_MAXPROD];
        const uint32_t ng = oz2_product_groups(mm, gr, digit_max, ndig, o.maxg_m[i]);
        double sum = 0;
        // 係数は [0,m) か対称代表元（平衡 n 桁・桁 15 未満の Karatsuba）なので絶対値で数える
        for (uint32_t j = 0; j < ng; ++j)
            sum += fold ? std::fabs((double)gr[j].coeff_lo) * ((double)mm / 2.0 + 2.0)
                        : std::floor(group_bound(mm, gr[j]) / (double)OZ2_COMPOSE_SPLIT + 0.5) * std::fabs((double)gr[j].coeff_hi)
                          + (double)(OZ2_COMPOSE_SPLIT / 2) * std::fabs((double)gr[j].coeff_lo);
        return sum;
    };
    auto gbound_of = [&](int i) {          // 出力 1 本の上界の最大
        Oz2ProductGroup gr[OZ2_MAXPROD];
        const uint32_t ng = oz2_product_groups(o.m[i], gr, digit_max, ndig, o.maxg_m[i]);
        double gmax = 0; for (uint32_t j = 0; j < ng; ++j) gmax = std::max(gmax, group_bound(o.m[i], gr[j]));
        return gmax;
    };
    // 連結したままで判定し，崩れれば連結をやめて（maxg_m を全部 1 に）もう 1 度だけ判定する．2 度目は maxg = 1 なので
    // やり直しの条件に当たらない
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool can_fold = true, grouped_ok = true;
        for (int i = 0; i < o.s; ++i) {
            const int mm = o.m[i];
            if (oz2_nout(mm, digit_max, ndig, o.maxg_m[i]) == 0) grouped_ok = false;   // 係数が位で決まらない
            if (gbound_of(i) + mm > std::ldexp(1.0, 24) || compose_sum(i, true) >= std::ldexp(1.0, 22))
                can_fold = false;
        }
        if (o.maxg > 1 && (!can_fold || !grouped_ok)) {   // まとめずに判定し直す
            o.maxg_m.assign(o.s, 1); sync_maxg(); continue; }
        o.fold_first = can_fold;
        if (!can_fold)
            for (int i = 0; i < o.s; ++i)
                if (const double cs = compose_sum(i, false); cs >= std::ldexp(1.0, 22)) oz2_acc_fail(o, "compose", o.m[i], 0, cs);
        break;
    }
}

// 法ごとの桁分割の基数と方式，桁行列の並びの先頭位置と，行列積・桁行列・連結後の出力の本数．行列積は法 1 本あたり，
// 2 桁分割なら 3 本（SQ は HL,LH,LL／KAR は HH,LL,SS）か 4 本（WIDE），1 桁なら 1 本，
// 平衡 n 桁なら n^2 本（m が q^n を割れば n(n+1)/2 本．BFP2_b の法 243 = 3^5 は 15 本），
// 桁分割しない INT8 経路なら 1 本．
// 方式は疎な桁（SQS）も区別して記録する（スライサへ渡す旗と表示．積と桁行列は SQ と同じ）．
static inline void oz2_layout(Ozaki2Moduli& o, const Oz2SelRoute& R) {
    const int digit_max = R.digit_max, smax = R.smax;
    const Oz2Ndig ndig = R.ndig;
    o.nprod = 0; o.nplane = 0; o.nout = 0;
    for (int i = 0; i < o.s; ++i) {
        const int mm = o.m[i];
        int qq = digit_max ? oz2_digit_base(mm, digit_max, ndig) : 0;
        int md = digit_max ? oz2_split_mode(mm, digit_max, ndig, smax) : OZ2_SPLIT_SQ;
        o.q.push_back(qq); o.mode.push_back(md);
        o.poff.push_back(o.nplane);
        o.nplane += digit_max ? oz2_digit_planes(mm, digit_max, ndig) : 1;
        o.nprod  += R.nprod(mm);
        o.nout   += digit_max ? (int)oz2_nout(mm, digit_max, ndig, o.maxg_m[i]) : 1;
    }
}

// CRT とスライス・再構成のカーネルへ渡す法ごとの定数（幅 o.b で）
static inline void oz2_crt_constants(Ozaki2Moduli& o) {
    const int b = o.b;
    // CRT の係数 y_i = (Π_{j≠i} m_j)^{-1} mod m_i
    // 積を取りながら毎回 m_i で畳むので，多倍長整数を作らずに済む
    for (int i = 0; i < o.s; ++i) {
        long P = 1, mi = o.m[i];
        for (int j = 0; j < o.s; ++j) if (j != i) P = (P * (o.m[j] % mi)) % mi;
        o.yi.push_back((int)o2_inv(P, mi));
    }
    // KM = M * 2^-2b．long double で計算するのは，b=54 のとき M が約 2^125 に達し，
    // 超過分を含めると符号付き __int128 でも溢れるためである．
    long double Mld = 1.0L; for (int mm : o.m) Mld *= (long double)mm;
    o.KMl = Mld * std::pow(2.0L, -2.0 * b);
    o.KM = (double)o.KMl;
    // 法ごとに 1/m_i を「k_i を掛けても無誤差」な 12 bit 断片の列へ分ける．
    // |k_i| <= oz2_crt_k_bound(m_i) = floor(m_i/2) + 1 で，法の上限は桁 32 の経路（BF16・FP16・TF32・BFP8_b）の 4096 なので
    // |k_i| <= 2049（12 bit）．断片が 12 bit なら積は 24 bit で，FP32 の仮数とちょうど同じ幅に収まって無誤差になる．
    // 【余裕は無い】法の上限を 4096 より上げるか断片を 12 bit より広げると前提が崩れる．
    //
    // 分けかたが 2 系統あるのが要点である（recon_crt.cpp の 2 パス構成）．
    //   rp1: 1/m_i を共通の絶対粒度 2^-OZ2_RGRAN へ丸めた R_i の断片．R_i が共通粒度の
    //        整数倍で k_i が整数なので k_i*R_i も総和も同じ粒度の整数倍で，総和の大きさが
    //        3 本組の 72 bit に収まるかぎり累算は完全に無誤差になる．
    //   rp2: 丸めの残り 1/m_i - R_i（R_i は最近接へ丸めるので両符号．大きさ 2^-RGRAN 以下）の断片．こちらは値そのものが
    //        小さいので，同じ 3 本組で累算しても絶対誤差は無視できる．
    // 残差は __float128 で持つ：long double（仮数 64 bit）だと 3 本目を作る時点で残差が
    // 2^-64 までしか分からず，分解全体がそこで頭打ちになる（実測で 2^-64.5）．
    const float SPL = 4097.0f;
    auto top12 = [&](__float128 v) {                      // v の上位 12 bit を FP32 で返す
        const float p = (float)(double)v;
        if (p == 0.0f) return 0.0f;
        const float c = o2_f32((double)SPL * (double)p);
        return o2_f32((double)c - (double)o2_f32((double)c - (double)p));
    };
    for (int mm : o.m) {
        const __float128 r = (__float128)1 / (__float128)mm;
        const __float128 gran = powq((__float128)2, -(__float128)OZ2_RGRAN);
        const __float128 Rg = roundq(r / gran) * gran;    // 共通粒度へ丸めた R_i
        std::array<float, OZ2_NRP1> p1{}; std::array<float, OZ2_NRP2> p2{};
        // rp1 は全断片を 12 bit にする：k_i*断片 が無誤差でなければ第 1 パスの
        // 「共通粒度の整数倍」という性質が崩れ，累算の無誤差性が失われる．
        __float128 v = Rg;
        for (int t = 0; t < OZ2_NRP1; ++t) { p1[t] = top12(v); v -= (__float128)p1[t]; }
        const __float128 sum1 = Rg - v;               // 断片に実際に載った量
        // 断片は 1 本あたり 12 bit 強を運ぶので，1/m <= 1/3（先頭が 2^-2 以下）である
        // かぎり 5 本で R を汲み尽くす（3..1089 の全ての m で残差ゼロを確認）．とはいえ
        // これは m 依存の余裕なので，rp2 は r-R ではなく r-sum(rp1) から作る：こうすれば
        // 断片が足りなくても取りこぼしが第 2 パスへ回るだけで，2 パスの和は必ず r に戻る．
        // sum(rp1) は R の下位を落としたものなので共通粒度の整数倍のままで，第 1 パスの
        // 無誤差性は保たれる．
        // rp2 と rp0 は最後の断片だけ FP32 まるごと（24 bit）にして幅を稼ぐ．k を掛けると
        // 丸めが入るが，その位はすでに十分小さく，落ちる量は無視できる．
        v = r - sum1;
        for (int t = 0; t < OZ2_NRP2; ++t) {
            p2[t] = (t + 1 < OZ2_NRP2) ? top12(v) : (float)(double)v; v -= (__float128)p2[t]; }
        std::array<float, OZ2_NRP0> p0{};                 // 粒度を揃えない素の分解
        v = r;
        for (int t = 0; t < OZ2_NRP0; ++t) {
            p0[t] = (t + 1 < OZ2_NRP0) ? top12(v) : (float)(double)v; v -= (__float128)p0[t]; }
        o.rp1.push_back(p1); o.rp2.push_back(p2); o.rp0.push_back(p0);
        o.rinv.push_back((float)(double)r);               // 剰余計算に使う 1/m の近似
    }
    // スライス（FP64 相当．A' は 2^20 進 3〜5 桁）用の定数．スライサは A' を 2^20 進の桁として
    // デバイス上で作るので（ozaki2/kernels/oz2_digits.h），2^20 mod m・2^40 mod m（と 4・5 桁の 2^60・2^80 mod m）が要る．
    // 桁を先に法で畳んでから重みを掛けるので (d_j mod m)*c_j は floor(m/2)*(m-1) を超えない．
    // c_j は [0, m) で持つ（対称にしない）．合成の総和は m=4096 級の法で 2^23.9 に達し，
    // 最後の ModRound の一般の保証 2^22 を超えるが，2^24 以下なので無誤差で剰余類は保たれる
    // （詳細と 4・5 桁の扱いは ozaki2/kernels/oz2_digits.h の oz2_residue，検査は tools/modround_check.cpp の (c)）．
    auto pow2mod = [](int e, long mod) {                 // 2^e mod mod（e >= 0）
        long r = 1 % mod, base = 2 % mod;
        for (int i = 0; i < e; ++i) r = (r * base) % mod;
        return (float)r;
    };
    for (int mm : o.m) {
        o.c0.push_back(pow2mod(OZ2_DIGIT_BITS, mm));
        o.c1.push_back(pow2mod(2 * OZ2_DIGIT_BITS, mm));
        o.c2.push_back(pow2mod(3 * OZ2_DIGIT_BITS, mm));   // 4 桁以上のときだけ使う
        o.c3.push_back(pow2mod(4 * OZ2_DIGIT_BITS, mm));   // 5 桁のときだけ使う
        o.m16.push_back(pow2mod(16, mm));
    }
}

// num_moduli  法の本数（＝独立な行列積の本数＝コスト）．GEMMul8 と同じくこれを
//             利用者が指定し，到達精度はそこから決まる．
// mod_cap  法の上限（INT8 経路は剰余の範囲から 255）．既定は置かない．桁分割経路では剰余をそのまま運ばず
//          桁に分けるので法を 8 bit に収める必要がなく，桁が ±d に収まる範囲まで取れる：
//          FP8（d=16）は疎な桁で 49^2 = 2401（K > 16381 では 33^2 = 1089），BF16・FP16・FP32・BFP8_b（d=32）は
//          64^2 = 4096，BFP4_b（d=7）は平衡 3 桁で 15^3 = 3375，BFP2_b（d=1，平衡 5 桁）は 3^5 = 243
//          （下の digit_max を参照．値は ozaki2_split.h の経路の表 Oz2Route の mod_cap）．
// acc_bits |C_i| = K*rmax^2 をどこまで許すか，すなわち再構成側が無誤差に読み
//          戻せる幅．C_i を FP32 の無誤差整数として扱うなら 24，int32 を 2 平面に
//          分けて復号するなら 31．24 bit だと使える法の上限が K とともに痩せ，
//          K=14080 では m<=69 まで落ちる．この表の最小値は 151 なので表からは 1 本も
//          取れず，151 未満の延長（OZ2_TABLE_I8 の注記）を足しても 69 以下の互いに素な法は 16 本
//          （b=35）しか取れない．FP64 相当には 31 bit 版の再構成が必須である．
// digit_max 剰余を平衡桁に分けて運ぶ経路での桁の絶対値の上限．0 なら
//          桁分割しない（剰余をそのまま運ぶ INT8 経路）．形式が無誤差に表せる整数の上限か，
//          累算器の幅（K*d^2 <= 2^24）の小さいほうで決まる：FP8 16（形式），BF16・FP16・
//          FP32・BFP8_b 32（累算器．K=14080），BFP4_b 7，BFP2_b 1（形式）．
// b_cap  幅 b の上限（既定 54）．一様乱数のように行内の全要素が共通の 2^-53 格子に
//        載る入力では 54 で足り，それ以上は M が余って再構成の相対精度が落ちるだけだが，
//        行内で指数が広がる入力では b に比例して精度が伸びる．呼び出し側が入力の
//        動的域から決めて渡す（ozaki2_autop.hpp の choose_precision_policy．OZ2_FP64）．構造上の上限は
//        OZ2_MAXDIGIT*OZ2_DIGIT_BITS = 100 で，A' を 5 桁で送る形式が運べる幅である．
// wlog2 は CRT の窓が覆うべき |C'|/2^(2b) の上界の log2 である．w_measured=false なら
// 使わず，最悪ケース上界 |C'| <= K*2^(2b)（すなわち log2(K)）を採る．呼び出し側が
// 入力から実際の上界 W を測っていれば w_measured=true と log2(W) を渡す：正規化後の
// 各行・各列の 2 ノルムから Cauchy--Schwarz で
//     max_ij sum_k |a~_ik||b~_kj| <= (max_i ||a~_i||_2)(max_j ||b~_j||_2) = W
// と抑えられ，行の最大値と列の最大値が別の k に居る入力ほど K より小さくなる．
// 窓が縮むと M も縮み，再構成が解く frac = C'/M が大きくなって相対精度が上がる．
// 「測ったか」を値の符号で表してはならない：W は 1/4 まで下がりうる（零でない行・列は
// 最大要素が (1/2, 1] に正規化されるので ||a~_i|| >= 1/2）ので，実測の log2(W) は負になりうる．
// ndig   剰余を桁に分ける形（Oz2Ndig．桁の数と 1 桁の法を使うか），smax 疎な桁の上限（Oz2Route::sparse_max）．
// awe    法の集合の選び方（Oz2Route::awe）．0 でなければ段の時間のモデルを最小にする選び方（ozaki2_awe.h）で
//        b = b_cap の窓を覆う集合を選ぶ（本数の上限は num_moduli と OZ2_MAXS）．見つからなければ表の選び方へ戻る．
// force  法の集合を外から与える（調査用．ozaki2_plan.hpp の oz2_parse_forced_moduli．OZ2_MODULI）．与えた順のまま使い，
//        落とす・選び直すをしない（使えない法・互いに素でない組は ok=false）．
// Mdim, Ndim  出力の大きさ（awe の選び方の DRAM の見積りだけが使う．0 なら M=N=K）．
static inline Ozaki2Moduli ozaki2_choose_moduli_w(uint32_t K, int num_moduli,
                                                  int mod_cap, int acc_bits,
                                                  int digit_max, int b_cap,
                                                  bool w_measured, double wlog2, Oz2Ndig ndig,
                                                  int smax = 0, int awe = 0,
                                                  const std::vector<int>* force = nullptr,
                                                  uint32_t Mdim = 0, uint32_t Ndim = 0) {
    // 呼び出し側が測っていなければ最悪ケース上界を使う
    const double Wlog2 = w_measured ? wlog2 : std::log2((double)K);
    // 疎な桁（smax．FP8 の e4m3 で |d| <= 32）は桁積 K*smax^2 に法を足しても FP32 の無誤差整数 2^24 に
    // 収まる K（smax=32 なら K <= 16381）でだけ使う．それより大きい K は稠密な桁（|d| <= digit_max）で運ぶ
    // （表は OZ2_TABLE_F8，法の上限は (2*digit_max+1)^2 = 1089 で，どちらも GEMMul8 の FP8 と同じ）．
    if (smax && digit_max && (double)K * smax * smax + mod_cap > std::ldexp(1.0, 24)) {
        smax = 0;
        mod_cap = std::min(mod_cap, (2 * digit_max + 1) * (2 * digit_max + 1));
    }
    Ozaki2Moduli o;
    o.smax = smax;
    const Oz2SelRoute R{K, mod_cap, acc_bits, digit_max, smax, ndig};
    int m_cap = 0;
    const Oz2Table tb = oz2_candidate_table(R, m_cap);
    double logM = 0.0;
    bool force_bad = false;
    const bool fixed_set = oz2_pick_candidates(o, logM, force_bad, R, tb, m_cap, num_moduli, b_cap, Wlog2, awe, force,
                                               Mdim, Ndim);
    o.b = oz2_width_from_window(logM, Wlog2, b_cap);
    const int b = o.b;
    // 必要量に安全余裕を足してはならない．2b+1+log2(K) は最悪ケース |C'| <= K*2^2b に
    // 対する厳密な必要量なので，これを満たせば CRT の一意復元は保証される．超過は
    // 安全側ではなく精度の損失で，再構成が frac = C'/M を固定の絶対精度で解くため，
    // M が必要量を超えるほど frac が小さくなって相対精度が落ちる（K=7040・phi=0 の
    // 実測で，余剰 1 bit につき要素ごとの最大相対誤差が約 1 bit 悪化する．ノルム比は
    // 余剰 4 bit までは動かないが，8 bit で 51.3，16 bit で 46.0 bit へ落ちる）．
    // OZ2_WIN_EPS は安全余裕ではなく，狭義不等号を log2 の比較で表すための最小幅である．
    const double need_log2 = 2.0 * b + 1.0 + Wlog2 + OZ2_WIN_EPS;
    // 与えた集合とモデルの集合は落とさず選び直さない（表から取った候補だけ）
    if (!fixed_set) {
        oz2_trim_tail(o.m, logM, need_log2);
        oz2_reselect_tail(o.m, logM, need_log2, R, m_cap);
    }
    o.s = (int)o.m.size();
    // 候補を使い切っても need_log2 に届かないことがある（法の上限が小さい経路や，
    // K が大きくて 1 本あたりの幅が痩せる場合）．黙って進むと CRT の窓が足りず，
    // エラーにならないまま「もっともらしい誤答」が出るので，ここで検出する．
    // OZ2_MAXS はカーネル側の配列長（kernels/oz2_limits.h．modslice_*.cpp / recon_crt.cpp の MAXS）と同じ．
    o.ok = (logM >= need_log2) && o.s > 0 && o.s <= OZ2_MAXS
           && (o.s <= num_moduli || (force && !force->empty())) && b > 0 && !force_bad;
    // 法が互いに素でないと CRT が成り立たない（y_i の逆元が存在しない）．表と
    // 末尾の選び直しの両方で担保しているつもりだが，破れると誤答が黙って出るので
    // 総当たりで検査する．s <= 24 なので費用は無視できる．
    for (int i = 0; i < o.s; ++i)
        for (int j = i + 1; j < o.s; ++j)
            if (std::gcd(o.m[i], o.m[j]) != 1) o.ok = false;
    // 累算の条件は失敗の報告（ozaki2_plan.hpp の oz2_report_moduli_failure）が読むので，集合が使えなくても数える
    oz2_check_accumulation(o, R);
    // 与えた集合に使えない法（経路の桁に収まらない法など）があれば，ここで戻る：その法の桁積を数えると平衡 n 桁の
    // 桁数が経路の上限を超え（oz2_bal_ndig の nmax+1），固定長の積の並び（oz2_product_terms の OZ2_MAXPROD）の外へ届く．
    // 失敗の報告が読む欄（ok・acc_ok・acc_fails・m・s・b・m_cap・ncand）はここまでに埋まっている
    if (force_bad) return o;
    oz2_plan_concat(o, R);
    oz2_layout(o, R);
    oz2_crt_constants(o);
    return o;
}
// 窓を測らない呼び出し（tools/ のホスト検査）のための入口．wlog2 < 0 を
// 「測っていない」とみなす．実測の log2(W) は負になりうるので，実測値は上の
// ozaki2_choose_moduli_w へ渡すこと（ドライバの ozaki2_plan.hpp はそうしている）．
// 経路の値（法の上限・累算の幅・桁の上限・幅の上限・桁の形）には既定を置かない：既定で呼ぶと本番と違う集合になる
// （acc_bits を FP32 の無誤差整数の 24 にすると INT8 経路の K=14080 で法が 69 以下に痩せる．本番は OZ2_ACC_BITS = 31）．
// 経路の表から呼ぶなら下の ozaki2_choose_moduli_route
static inline Ozaki2Moduli ozaki2_choose_moduli(uint32_t K, int num_moduli,
                                                int mod_cap, int acc_bits,
                                                int digit_max, int b_cap,
                                                double wlog2, Oz2Ndig ndig, int smax = 0, int awe = 0) {
    return ozaki2_choose_moduli_w(K, num_moduli, mod_cap, acc_bits, digit_max, b_cap,
                                  wlog2 >= 0.0, wlog2, ndig, smax, awe);
}
// 経路の表 Oz2Route から呼ぶ入口（tools/ のホスト検査が本番と同じ引数で法を選ぶため）
static inline Ozaki2Moduli ozaki2_choose_moduli_route(uint32_t K, const Oz2Route& r, int b_cap = OZ2_B_DEFAULT,
                                                      int num_moduli = 0) {
    return ozaki2_choose_moduli(K, num_moduli ? num_moduli : (int)r.def_s, r.mod_cap, OZ2_ACC_BITS,
                                r.digit_bound, b_cap, -1.0, r.nd(), r.sparse_max, r.awe);
}

// 再構成の limb の本数：3 本（triple-float．FP64 相当）か 2 本（FP32 級）か．1 パス構成は b >= 24 なら 3 本．2 パス構成は
// 第 1 パスのライタが limb を必ず 3 本読んで 128 bit 固定小数へ足す（recon_writer.cpp の ROLE_TWO_FIRST の枝）ので，b が小さくても
// 両パスとも 3 本に揃える（揃えないと 3 本目が別の組の limb か CB の外の値になる）．ドライバ（ozaki2_plan.hpp の
// make_recon_plan）とホスト単体の模擬（tools/oz_sim.h の make_recon_kernels）が同じこの規則を使う
static inline bool oz2_recon_limb3(const Ozaki2Moduli& mo, bool two_pass) { return two_pass || mo.b >= 24; }
// 再構成の計算カーネルのコンパイル時引数 RAWMASK（recon_args.h の C_RAWMASK）：1 桁の法（OZ2_SPLIT_ONE．桁 32 の AWE 経路）は
// 剰余の平面に合成段を通らない GEMM の出力（法へ畳む前の値）が入るので，recon_crt.cpp がその法だけ先に ModRound する．
// ドライバ（ozaki2_stages.hpp の recon_make_kernels）とホスト単体の模擬（tools/oz_sim.h の make_crt_kernel）が同じこの組み立てを使う
static inline uint32_t oz2_recon_rawmask(const Ozaki2Moduli& mo) {
    uint32_t raw = 0;
    for (int i = 0; i < mo.s; i++) if (mo.mode[i] == OZ2_SPLIT_ONE) raw |= 1u << i;
    return raw;
}


// --- tfAddSmall へ回す断片の本数 --------------------------------------------
// recon_crt.cpp の足し込みは，末尾 NSMALL_FRAG 本の断片だけ tfAddS の第 1 段
// TwoSum(a0,v) を飛ばして tfAddSmall で足す（15 演算 -> 8 演算）．
// sfpu_tf_proc.h の tfAddSmall の誤差の上界は呼び出し側が |v| <= eps を守ることを前提にしていて，
// その eps と本数を決めるのがこの関数である．
//
// eps はそのパスの累算器の大きさを基準に取る．絶対値 2^-30 に固定すると
// 2 パス構成の第 2 パスで破れる：第 2 パスの断片は第 1 パスが 60 bit 取った残りで
// 2^-60 前後しかなく，a0 自体が 2^-49 程度にしかならない．そこで絶対値で測ると
// 先頭の断片まで「小さい」に数えてしまい，a0 が使われないまま全部が a1 へ入る
// （実測で limb 1 本ぶん落ちる．tools/twosum_probe.cpp の TWOSUM_PROBE_ABS）．
// 基準にするのは同じパスの最大項 vmax = max_{t,i} |k_i * RP[t][i]| で，
// 小数部へ畳むぶん 1 で頭打ちにする（|a0| <= 1）．ulp = 2^-23 に対し余裕を
// 7 bit 取り，eps = vmax * 2^-30 とする．
// 1 パスと第 1 パスでは vmax が O(1) なので末尾 2 本が該当し，第 2 パスでは 0 本になる．
//
// 【本番とホスト検査の両方がこの 1 つの実装を呼ぶ】別々に書くと，本番だけを
// 誤って変えても検査は素通りする．デバイス API に依存しない純粋な関数にしてある．
struct Oz2SmallTail {
    uint32_t n = 0;        // 末尾から何本を tfAddSmall へ回すか
    float vmax = 0.f;      // そのパスの最大項 max_{t,i} |k_i * RP[t][i]|
    float eps = 0.f;       // 判定に使ったしきい値
    float worst[8] = {0};  // 断片ごとの最大項（nrp <= 8 を想定）
};

// CRT の係数を掛けた剰余 k_i = ModRound(r_i*y_i)（recon_crt.cpp）の絶対値の上界 floor(m/2) + 1．r_i は合成段の出力（対称剰余
// |r_i| <= m/2）か，1 桁の法を再構成が畳んだ値（|r_i| <= floor(m/2)+2，m <= 65）で，0 <= y_i < m なので入力は
// |r_i*y_i| < 2^23（m <= 4096）である．ModRound が対称剰余を保証する 2^22 を超えうるので，対称剰余からのはみ出しは
// |r_i*y_i|*2^-23 < 1 で（sfpu_tf_proc.h の ModRound の注記），奇数の法で 1 だけ超える（m=3475・r=1737・y=3474 は k=1738）．
// y_i は他の法で決まるので，既定の集合の総当たりだけでは集合を変えたとき（OZ2_MODULI など）を覆えない．選びうる全法・
// 全係数での確認は tools/modround_check.cpp の (a')（同じ上界をこの関数から取る）
static inline long oz2_crt_k_bound(int m) { return m / 2 + 1; }

// rp は [i*nrp + t] の並び（法 i，断片 t）．m は法の集合．
static inline Oz2SmallTail oz2_choose_small_tail(const float* rp, uint32_t nrp,
                                                 const int* m, uint32_t s);

// 再構成の計算カーネル（recon_crt.cpp）のコンパイル時引数のうち，法の集合とパスから決まるもの（recon_args.h の CrtCT の
// C_S・C_NLIMB・C_IN_RESIDUE・C_NRP・C_NSMALL_FRAG・C_RAWMASK）．ドライバ（ozaki2_stages.hpp の recon_make_kernels）とホスト単体の
// 模擬（tools/oz_sim.h の make_crt_kernel）が同じこの組み立てを使う．調査用のつまみ（OZ2_NSMALL など）と探針（C_PROBE）は
// ドライバが重ねる．rp は逆数の断片（[i*nrp + t]），limb3 は oz2_recon_limb3
struct Oz2ReconCT { uint32_t S, NLIMB, IN_RESIDUE, NRP, NSMALL_FRAG, RAWMASK; };
static inline Oz2ReconCT oz2_recon_ct(const Ozaki2Moduli& mo, const float* rp, uint32_t nrp, bool in_residue, bool limb3) {
    return {(uint32_t)mo.s, limb3 ? 3u : 2u, in_residue ? 1u : 0u, nrp,
            oz2_choose_small_tail(rp, nrp, mo.m.data(), (uint32_t)mo.s).n, oz2_recon_rawmask(mo)};
}

static inline Oz2SmallTail oz2_choose_small_tail(const float* rp, uint32_t nrp,
                                                 const int* m, uint32_t s) {
    Oz2SmallTail r;
    if (nrp > 8) return r;                      // 呼び出し側の想定外．0 本にして安全側へ
    for (uint32_t t = 0; t < nrp; t++) {
        for (uint32_t i = 0; i < s; i++) {
            // 項の上界は |RP[t][i]| * max|k_i| で，|k_i| は oz2_crt_k_bound（floor(m/2)+1．対称剰余の m/2 では
            // 集合を変えたときの y_i で 1 超える k_i を覆えない）で数える
            const float v = std::fabs(rp[(size_t)i * nrp + t]) * (float)oz2_crt_k_bound(m[i]);
            if (v > r.worst[t]) r.worst[t] = v;
        }
        if (r.worst[t] > r.vmax) r.vmax = r.worst[t];
    }
    const float scale = r.vmax < 1.f ? r.vmax : 1.f;
    r.eps = (float)std::ldexp((double)scale, -30);
    for (uint32_t t = nrp; t-- > 0; ) {
        if (r.worst[t] <= r.eps) r.n++; else break;
    }
    return r;
}
