// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：経路の表と，剰余を桁に分ける方式・桁の上限・桁積の列挙と連結（ホスト側．デバイス API に依存しない）
// ---------------------------------------------------------------------------
// 形式ごとの経路の表 OZ2_ROUTES（oz2_param），法 1 本の運び方（OZ2_SPLIT_* と桁の上限・桁行列・桁積），同じ位の桁積を
// K 方向に連結する組（oz2_product_groups）と連結の上限（oz2_maxg_for）を置く．ozaki2_awe.h（法の集合を段の時間のモデルで
// 選ぶ）と ozaki2_moduli.h（法の表・選択・CRT の定数）がこれを取り込む．ドライバ（ozaki2_format.hpp）とホスト単体の
// 検査・模擬（tools/）は ozaki2_moduli.h を通して取り込む．
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "common/oz_format_host.h"    // OzFmt（経路の表の形式）
#include "common/oz_dram_margin.h"    // OZ_DRAM_MARGIN_BYTES（両スキームの DRAM の見積りの余裕）
#include "ozaki2/kernels/oz2_limits.h"   // OZ2_MAXPROD・OZ2_MAXS・OZ2_DIGIT_BITS・OZ2_MAXDIGIT・OZ2_COMPOSE_SPLIT_BITS・Oz2XMode（カーネルと共有）

// --- 形式ごとの法の取り方（scheme II の経路の表）------------------------------
// ドライバの形式ごとの値 Oz2Param（ozaki2_format.hpp の OZ2_PARAM）と，ホスト単体の検査（tools/modround_check.cpp・
// tools/twosum_probe.cpp など）が同じ値を見るよう，ここに 1 つだけ置く（検査側が値を書き写すと，
// 片方だけ直したときに検査が本番と違う法集合を見る）．形式から経路への対応と値の理由は下の OZ2_ROUTES の注記にある．
//   mod_cap      法の上限
//   digit_bound  実際に使う桁の上限（0 = 桁分割しない．INT8 経路）
//   nrdig        剰余を分ける桁の数（2 か 5．5 は BFP2_b の平衡三進で，桁数は法ごとに 5 以下の最小）
//   def_s        法の本数の既定値（GEMMul8 の num_moduli）．awe の経路では本数の上限
//   sparse_max   疎な桁の上限（0 = 使わない）．digit_bound までは全整数が，その先 sparse_max までは
//                形式が無誤差に表せる整数だけが桁になれる（FP8 の e4m3 は 16 まで全部，16〜32 は偶数）．
//                これが 0 でない経路では，奇数 q の平方数 q^2（q <= |S|．S は桁になれる整数の集合）を
//                方式 SQS（下の (E)）で運べる．FP8 だけが使う（docs/fp8_sparse_digits.md）
//   awe          法の集合の選び方．0 = 表の先頭から取って末尾 2 本を選び直す（表の選び方），それ以外は
//                段の時間のモデルを最小にする選び方（ozaki2_awe.h．値はモデルの係数の表の番号）．
//                BF16・FP16・BFP8_b・BFP4_b・BFP2_b が使い，INT8・FP8・TF32 は 0（docs/awe_moduli.md）
//   one_digit    1 桁の法を混ぜるか（下の Oz2Ndig::one）．桁 32 の AWE 経路（BF16・FP16・BFP8_b）だけが使う
// 剰余を桁に分ける形（Oz2Route::nd()．法ごとの関数はこれを受け取る）
//   n    桁の数（2 か 5）．スライサのコンパイル時引数 NRDIG へもこの値を渡す
//   one  1 桁で運べる小さい法（|r| <= d．d=32 なら m <= 65）を 1 桁（下の (F) OZ2_SPLIT_ONE）で運ぶ．1 桁の法は
//        桁の上限が経路の d より小さい法と並ぶので，連結の上限 maxg_m と累算の条件も経路一律の d ではなく
//        桁行列ごとの桁の上限（oz2_plane_bound）で法ごとに見る（ozaki2_choose_moduli_w・ozaki2_awe.h）
struct Oz2Ndig {
    int  n   = 2;
    bool one = false;
    constexpr Oz2Ndig(int n_ = 2, bool one_ = false) : n(n_), one(one_) {}
};
struct Oz2Route {
    int      mod_cap;
    int      digit_bound;
    int      nrdig;
    uint32_t def_s;
    int      sparse_max = 0;
    int      awe = 0;
    bool     one_digit = false;
    constexpr Oz2Ndig nd() const { return Oz2Ndig(nrdig, one_digit); }
};
// 段の時間のモデルの経路の番号（ozaki2_awe.h の係数の表の行．1 始まり）．番号を足すときは OZ2_AWE_END の前に置く（係数の表
// OZ2_AWE_COEF は第 1 次元を OZ2_AWE_COUNT で宣言するので，行が足りなければ 0 で埋まり，ozaki2_awe.h の oz2_awe_coef_rows_set が
// それを止める．多すぎれば初期化子の数でコンパイルが止まる）
enum { OZ2_AWE_NONE = 0, OZ2_AWE_BFP4 = 1, OZ2_AWE_BFP2 = 2, OZ2_AWE_BF16 = 3, OZ2_AWE_FP16 = 4, OZ2_AWE_BFP8 = 5,
       OZ2_AWE_END };
inline constexpr int OZ2_AWE_COUNT = OZ2_AWE_END - 1;
inline constexpr Oz2Route OZ2_ROUTE_INT8 = {255,  0, 2, 16u};
// FP8：e4m3 の桁 S_32 = {0..16} ∪ {18,20,..,32}（と負）で 49^2 = 2401 まで（docs/fp8_sparse_digits.md）．
// 桁を [-16,16] に限ると法の上限は 1089 = 33^2（GEMMul8 の FP8 表）で，疎な桁を使えない大きな K（K > 16381）は
// そちらになる（ozaki2_moduli.h の ozaki2_choose_moduli_w の冒頭）．
inline constexpr Oz2Route OZ2_ROUTE_FP8  = {2401, 16, 2, 14u, 32};
// 桁 32 の表の選び方（TF32．名前の D32 は桁の上限 32）．BF16・FP16・BFP8_b は下の OZ2_ROUTE_*A で，tools/awe_plan.cpp は
// 3 経路の段の時間を実測したときの選び方（較正の基準）としてもこれを使う
inline constexpr Oz2Route OZ2_ROUTE_D32 = {4096, 32, 2, 11u};
// 桁 32 の AWE 経路（BF16・FP16・BFP8_b）：平方数・Karatsuba・WIDE に 1 桁の小さい法（m <= 65）を混ぜ，連結の上限を法ごとに
// 決め，法の集合は段の時間のモデルで選ぶ（docs/awe_moduli.md の §5．TF32 は組分けがあるので表の選び方の OZ2_ROUTE_D32）
inline constexpr Oz2Route OZ2_ROUTE_BF16A = {4096, 32, 2, 24u, 0, OZ2_AWE_BF16, true};
inline constexpr Oz2Route OZ2_ROUTE_FP16A = {4096, 32, 2, 24u, 0, OZ2_AWE_FP16, true};
inline constexpr Oz2Route OZ2_ROUTE_BFP8A = {4096, 32, 2, 24u, 0, OZ2_AWE_BFP8, true};
// BFP4_b：平方数（225 まで）・基数 8 の Karatsuba（113 まで）・平衡 2 桁（225 まで）・平衡 3 桁（15^3 = 3375 まで）を
// 法ごとに使い分け，法の集合は段の時間のモデルで選ぶ（表 OZ2_TABLE_B4 の先頭 17 本，平方数 4 本＋平衡 2 桁の
// 素数 13 本で運ぶと積 64 本）．
inline constexpr Oz2Route OZ2_ROUTE_BFP4 = {3375, 7, 2, 24u, 0, OZ2_AWE_BFP4};
// BFP2_b：平衡三進 n 桁（n = 1..5．法 3^n まで）の n を法ごとに最小に取り，法の集合は段の時間のモデルで選ぶ
// （表 OZ2_TABLE_B2 の 16 本を 5 桁で運ぶと積 390 本）．
inline constexpr Oz2Route OZ2_ROUTE_BFP2 = {243,  1, 5, 24u, 0, OZ2_AWE_BFP2};

// --- 形式ごとの経路（OZ2_ROUTES．ドライバの OZ2_PARAM（ozaki2_format.hpp）はこの表を形式で引く）----------------
// 形式（OzFmt）と表示名（routes.tsv の short の II- の後ろ．TF32 は FP32 で格納する経路）と経路の値の組．
// ドライバとホスト単体の検査・模擬（tools/oz_sim.h など）が同じ表を回す（形式から経路への対応を書き写すと，
// 経路の選び方を変えたときに検査が本番と違う法の集合を見る）．
// 並びは検査の都合で FP16 を最後に置く：FP16 は運ぶ形式だけが BF16 と違い，同じ法の集合になることが多い（その K では
// 検査は BF16 の行で兼ねる）．乱数の列を経路をまたいで引く検査でも，最後に置けば他の行の入力は FP16 の有無で変わらない．
// 欄の意味（上の Oz2Route の注記に加えて）：
//   digit_bound  実際に使う桁の上限は，形式が無誤差に表せる整数の上限（common/oz_format_host.h の exact_int_max）とは
//                別で，累算器にも縛られる
//   nrdig        2 は平方数／Karatsuba／平衡 2 桁の 3 方式（BFP4_b は加えて平衡 3 桁），5 は BFP2_b の平衡 n 桁
//                （n は法ごとに 5 以下．OZ2_SPLIT_BAL）．カーネルは 2 と 5 だけを受け付ける（modslice_digit.cpp の NRDIG）
// 形式ごとの「供給上限」（アンパッカの 380.2 B/サイクルを 1 要素のバイト数で割った値）は持たない．
// 分割データ移動＋L1 累積の GEMM 段の実測がこれを超える（docs/gemm_opt_results.md の §7.3）．
// 要素あたりのバイト数は GEMM 段の速さを左右するが，上限ではない．
struct Oz2RouteRow {
    OzFmt       fmt;
    const char* name;   // 表示名（INT8・FP8・TF32・BF16・BFP8・BFP4・BFP2・FP16）
    Oz2Route    r;
};
inline constexpr Oz2RouteRow OZ2_ROUTES[] = {
    // 桁分割しない．8 bit の法では同じ窓に 16 本を要する
    {OzFmt::INT8, "INT8", OZ2_ROUTE_INT8},
    // 桁分割すれば法は桁が [-16,16] に収まる範囲まで取れる（稠密な桁の上限は形式の上限 16）．
    // さらに e4m3 は 16 から 32 までの偶数も無誤差に表せるので，桁をその疎な集合（|S| = 49）に取ると
    // 奇数 q <= 49 の平方数（2401 まで）も積 3 本で運べる（sparse_max=32．下の (E)）．
    // 法が 11 bit 級になり，14080^3 では 12 本・行列積 37 本で窓を覆う（稠密な桁だけなら法は 1089 までで 14 本・42 本）．
    // K > 16381 では桁積の上界 K*32^2 に法 2401 を足した値が 2^24 を超え（K*32^2 そのものは K <= 16384 まで
    // 2^24 以下だが，合成が法を足しても FP32 の無誤差整数に収まる必要がある），稠密な桁の法（1089 まで）に戻る．
    {OzFmt::FP8,  "FP8",  OZ2_ROUTE_FP8},
    // FP32 で格納し，行列エンジンには TF32 として渡す経路（論文での名前は TF32．common/oz_format_host.h）．
    // 桁の上限は BF16 と同じ 32 だが，法は表の選び方で選ぶ（1 桁の法も段の時間のモデルも使わない）．
    // 1 要素 4 B になる．
    // 桁面も 2 倍になり，14080^3 では A・B の桁面が各 17.4 GB（平方数 11 本で 22 枚）で，一括では DRAM 34.2 GB に
    // 収まらない．そのため 14080^3 では出力を行の帯に分けて切る（ozaki2_plan.hpp の plan_bands．A の桁面・積の器を帯 1 本
    // ぶんだけ持つ．Band 行・CONFIG の bands= に出る）．
    {OzFmt::FP32, "TF32", OZ2_ROUTE_D32},
    // BF16 は無誤差に 256 まで表せるので，BFP8_b と同じく桁の上限を決めるのは形式ではなく
    // 累積器で，K=14080 では d <= 32 である．したがって法も分割方式も GEMM の本数も
    // BFP8_b 経路とまったく同じになり，違うのは運ぶ形式だけである（1 要素 2 B 対 1.0625 B）．
    // スキーム I の BF16 対 BFP8_b と同じ関係がスキーム II にも現れる．
    // BF16・FP16・BFP8_b は 1 桁の小さい法（m <= 65）を混ぜ，連結の上限を法ごとに決め，法の集合を
    // 段の時間のモデルで選ぶ（OZ2_ROUTE_*A．docs/awe_moduli.md の §5）．TF32 は組分けがあるので表の選び方を使う．
    {OzFmt::BF16, "BF16", OZ2_ROUTE_BF16A},
    // BFP8_b は無誤差に ±127 を表せるので，桁の上限は形式ではなく累積器が決める：法 1 本の
    // GEMM は K 項の桁積を足すので 2*log2(d) + ceil(log2 K) <= 24 が要り，K=14080 では d <= 32．
    // 桁を 16 から 32 へ広げると平衡 2 桁で扱える法が 4d^2 まで伸びるので，12 bit 級の法を取れる
    // （上限 64^2 = 4096）．法の集合は段の時間のモデルで選び（BF16・FP16 と同じ OZ2_ROUTE_*A），
    // 平方数・Karatsuba・WIDE に 1 桁の小さい法を混ぜる：K=14080 では 14 本（平方数 8 本と 1 桁の法 6 本）で
    // 行列積 30 本，K=3520 では 11 本（1 桁の法 1 本）で 32 本になる（docs/awe_moduli.md の §5）．
    // 合成段は桁積を先に法へ畳む方式（fold_first）になる．桁積に法を足しても 2^24 以下なので
    // ModRound が剰余類を保ち，畳んだあとの総和は (Σ_j c_j)(m/2+2)（平方数なら c = 1, q, q）で，
    // 法選択が実際の係数で 2^22 未満を確かめる．
    // 要素あたりは 1.0625 B で FP8 の 1 B より広い．
    {OzFmt::BFP8, "BFP8", OZ2_ROUTE_BFP8A},
    // BFP4_b が無誤差に表せる整数は 7 までなので，桁の上限を決めるのは累積器ではなく形式で
    // ある．平衡 2 桁で桁が d に収まる法は (2d+1)^2 = 225 が上限になる．非平方数は基数 8 の Karatsuba
    // （m <= 113）か平衡桁（WIDE，積 4 本）で運び，225 を超える法は平衡 3 桁（15^3 = 3375 まで）で運ぶ．
    // 法の集合は段の時間のモデルで選ぶ（K=14080 で 17 本・行列積 58 本．docs/awe_moduli.md）．
    // 1 要素 0.5625 B と狭い．
    {OzFmt::BFP4, "BFP4", OZ2_ROUTE_BFP4},
    // BFP2_b が無誤差に表せる整数は ±1 だけなので，剰余を 2 桁に分けても法は (2d+1)^2 = 9 までで，
    // 窓 122.8 bit に遠く届かない．そこで平衡三進法（基数 q = 2d+1 = 3）の n 桁（n <= 5）へ広げる．
    // 5 桁なら法は 3^5 = 243 まで扱え，n は法ごとに最小に取る（法 81 以下は 4 桁で積 16 本，27 以下は
    // 3 桁で 9 本）．法の集合は段の時間のモデルで選ぶ（K=14080 で 20 本・行列積 329 本．docs/awe_moduli.md）．
    // 1 要素 0.3125 B と最も狭い．
    {OzFmt::BFP2, "BFP2", OZ2_ROUTE_BFP2},
    // 法も分割方式も GEMM の本数も BF16 経路と同じで，違うのは運ぶ形式だけである（1 要素 2 B）．
    // 桁は binary16 のビット列を UInt16 で書き出す（modslice_digit.cpp，
    // sfpu_tf.h の HalfBits．理由は common/oz_format_host.h の FP16 の項）．
    {OzFmt::FP16, "FP16", OZ2_ROUTE_FP16A},
};
// 形式の経路．表に無い形式は値が全部 0 の経路（起きない．下の static_assert）
constexpr Oz2Route oz2_param(OzFmt f) {
    for (const Oz2RouteRow& x : OZ2_ROUTES)
        if (x.fmt == f) return x.r;
    return {};
}
// 全形式がちょうど 1 回ずつ現れること
constexpr bool oz2_routes_cover() {
    for (OzFmt f : OZ_FMTS) {
        int n = 0;
        for (const Oz2RouteRow& x : OZ2_ROUTES) n += (x.fmt == f);
        if (n != 1) return false;
    }
    return sizeof(OZ2_ROUTES) / sizeof(OZ2_ROUTES[0]) == sizeof(OZ_FMTS) / sizeof(OZ_FMTS[0]);
}
static_assert(oz2_routes_cover(), "OZ2_ROUTES に無い形式か，2 回現れる形式がある");
// 再構成の入力が剰余（FP32）か：桁分割の経路は合成段が畳んだ剰余を，INT8 経路は GEMM が書いた符号-絶対値 int32 を渡す
// （ozaki2_main.cpp の run_recon_stage）
constexpr bool oz2_in_residue(const Oz2Route& r) { return r.digit_bound != 0; }

// --- 桁分割の方式（(A)〜(F) の 6 つ）------------------------------------------
// 剰余を桁に分けて桁ごとの行列積から積を組み立てる．桁は経路の許す集合に収める．
// 稠密な集合は絶対値d以下だが，FP8の疎な集合(E)は2d以下の偶数も使う．以下は FP8（d=16．fp8_e4m3 が無誤差に
// 表せる整数の上限）で説明する．BF16・FP16・FP32・BFP8_b は d=32（BF16・FP16・BFP8_b は加えて (F)），
// BFP4_b は d=7（(B) は基数 8，(D) は 3 桁），BFP2_b は d=1 で平衡 n 桁の (D) を使う．
//
// (A) OZ2_SPLIT_SQ  平方数の法 m = q^2：最近接丸めの平衡桁．行列 2 枚・積 3 本
//        h = round(r/q),  l = r - q*h      （|l| <= q/2,  |h| <= round((m/2)/q)）
//     q^2 ≡ 0 (mod m) なので上位×上位の項が消え，行列積は HL, LH, LL の 3 本．
//        r_a*r_b ≡ q*(h_a l_b + l_a h_b) + l_a l_b   (mod m)
//
// (B) OZ2_SPLIT_KAR 非平方数の法：基数 B の Karatsuba（桁 16 以上の経路は B=16，BFP4_b は B=d+1=8．
//     oz2_kar_base）．行列 3 枚・積 3 本．以下は B=16 で書く
//        h = sign(r)*ceil(|r|/16),  l = r - 16*h,  s = h + l
//     切り上げなので h と l が必ず逆符号になり，|h|,|l|,|s| がそろって 16 以下に
//     収まる（最近接丸めだと |h+l| が 24 まで伸びて e4m3 の無誤差整数から外れる）．
//     行列積は HH, LL, SS の 3 本で，中間項を引き算で取り出す：
//        r_a*r_b = 256*HH + 16*(SS - HH - LL) + LL
//     桁が 16 に収まる条件は |h| <= ceil((m/2)/16) <= 16，すなわち m <= 513．
//     GEMMul8 の FP8 テーブルの非平方数はすべて 511 以下なので，参照実装が使う
//     法にはこの方式がそのまま当たる．
//
// (C) OZ2_SPLIT_WIDE 513 を超える非平方数の法：最近接丸めの平衡桁（q ≒ sqrt(m)）．
//     行列 2 枚・積 4 本．上位×上位が消えないので HH の 1 本が余分に要る：
//        r_a*r_b ≡ q^2*HH + q*(HL+LH) + LL   (mod m)
//     GEMMul8 の法テーブルにはこの範囲の法が無いので参照実装には対応物が無い．
//     末尾2法の再選択（ozaki2_moduli.h の oz2_reselect_tail）や時間モデルによる探索で現れる．
//     この選択肢が無いと末尾再選択の候補が513以下に痩せ，法が1本余計に要って
//     スライス段と再構成段がそのぶん重くなる（実測で 3520/7040 が 5% 遅くなる）．
// (D) OZ2_SPLIT_BAL 平衡 n 桁（n = 1..5．カーネルは法ごとの実行時引数で n を受ける．ozaki2/kernels/modslice_digit.cpp）：
//     基数 q = 2d+1 の平衡表現で r = Σ_j x_j q^j，|x_j| <= d．行列 n 枚・積は n^2 本（m が q^n を割るときは
//     i+j >= n の項が消えて n(n+1)/2 本）．合成は D = Σ_{i,j} q^(i+j) P_ij である．
//     BFP2_b（d=1）は全部の法をこれで運び，桁数 n は法ごとに最小に取る（oz2_bal_ndig）．
//     BFP4_b（d=7）は 2 桁の方式で運べない 225 < m <= 3375 の法を 3 桁で運ぶ（docs/awe_moduli.md）．
// (E) OZ2_SPLIT_SQS 疎な桁の平方数の法 m = q^2（q は奇数で 33 < q <= |S|．FP8 の S_32 なら q <= 49）．
//     行列 2 枚・積 3 本で，積と合成は (A) と同じ（q^2 ≡ 0 なので HL, LH, LL）．違うのは桁の作り方だけで，
//     桁は区間 [-16,16] ではなく疎な集合 S = {|x| <= 16} ∪ {偶数 x : 16 < |x| <= 32}（e4m3 が無誤差に
//     表せる整数．arXiv 2608.06812 の「無誤差に表せる非連続な整数集合で剰余を覆う」考え方）に入れる．
//     |S| = 49 で，奇数 q <= 49 なら S は法 q の完全剰余系になる（奇数の剰余 17..q-18 を -(偶数) が覆う）．
//        h0 = round(r/q),  t = r - q*h0                  （(A) と同じ．|t| <= (q-1)/2）
//        t が奇数で |t| >= 17 なら  l = t - sgn(t)*q, h1 = h0 + sgn(t)   （r = l + q*h1 のまま）
//                             でなければ l = t, h1 = h0
//        h1 が奇数で |h1| >= 17 なら h = h1 - sgn(h1)*q，でなければ h = h1   （q*h ≡ q*h1 (mod q^2)）
//     l は |t| <= 16 か偶数ならそのまま，奇数の |t| >= 17 は q - |t| <= q - 17 <= 32 の偶数になる．h も同じ．
//     したがって l, h ∈ S で r ≡ l + q*h (mod q^2)．全法・全剰余の総当たりは tools/modround_check.cpp，
//     桁の上限（法ごとの l・h それぞれの最大）はホストが oz2_sqs_bounds で数え上げる．
// (F) OZ2_SPLIT_ONE 1 桁（桁 32 の AWE 経路．Oz2Ndig::one）：全剰余が桁の上限に収まる小さい法（|r| <= d，
//     d=32 なら m <= 65）は剰余そのものを桁行列 1 枚にし，行列積 1 本で運ぶ．合成は係数 1 の 1 本なので合成段を通さず，
//     GEMM の出力をそのまま剰余の平面へ書いて再構成が法へ畳む（ozaki2_main.cpp の run_digit_gemms・recon_crt.cpp の
//     RAWMASK．docs/awe_moduli.md）．
enum { OZ2_SPLIT_SQ = 0, OZ2_SPLIT_KAR = 1, OZ2_SPLIT_WIDE = 2, OZ2_SPLIT_BAL = 3, OZ2_SPLIT_SQS = 4, OZ2_SPLIT_ONE = 5 };
// スライサ（modslice_digit.cpp）へ渡す桁の作り方の番号（kernels/oz2_limits.h の Oz2XMode）．(A)(C) はスライサから見ると
// 同じ手続きなので同じ番号になる．知らない方式は OZ2_XMODE_COUNT を返す（呼び出し側が止める．既定の枝へ黙って落とさない）
constexpr Oz2XMode oz2_xmode(int split_mode) {
    switch (split_mode) {
    case OZ2_SPLIT_SQ:
    case OZ2_SPLIT_WIDE: return OZ2_XMODE_RND;
    case OZ2_SPLIT_KAR:  return OZ2_XMODE_KAR;
    case OZ2_SPLIT_SQS:  return OZ2_XMODE_SQS;
    case OZ2_SPLIT_BAL:  return OZ2_XMODE_BAL;
    case OZ2_SPLIT_ONE:  return OZ2_XMODE_ONE;
    }
    return OZ2_XMODE_COUNT;
}
static_assert(oz2_xmode(OZ2_SPLIT_SQ) == OZ2_XMODE_RND && oz2_xmode(OZ2_SPLIT_WIDE) == OZ2_XMODE_RND &&
              oz2_xmode(OZ2_SPLIT_KAR) == OZ2_XMODE_KAR && oz2_xmode(OZ2_SPLIT_SQS) == OZ2_XMODE_SQS &&
              oz2_xmode(OZ2_SPLIT_BAL) == OZ2_XMODE_BAL && oz2_xmode(OZ2_SPLIT_ONE) == OZ2_XMODE_ONE,
              "法の分け方からスライサの桁の作り方の番号への対応");


// --- 疎な桁（(E)．FP8 の e4m3）-------------------------------------------------
// e4m3 が無誤差に表せる整数か（有効ビットが 4 以下．|v| <= 448 の範囲で使う）
static inline bool oz2_e4m3_int(long v) {
    if (v < 0) v = -v;
    if (v == 0) return true;
    int e = 0; for (long x = v; x; x >>= 1) ++e;      // ビット長
    return e <= 4 || (v % (1L << (e - 4))) == 0;
}
// 桁の集合 S_smax = { e4m3 が無誤差に表せる整数 x : |x| <= smax }
static inline bool oz2_in_sparse(long x, int smax) { return (x < 0 ? -x : x) <= smax && oz2_e4m3_int(x); }
// S_smax が法 q の完全剰余系を含むか（q 通りの剰余をすべて覆うか）
static inline bool oz2_sparse_covers(int q, int smax) {
    if (q <= 0 || smax <= 0) return false;
    if (q > 2 * smax + 1) return false;                   // 2*smax+1 個に満たない集合は覆えない
    static signed char cache[129][129];                   // [smax][q]：0 未計算，1 覆う，-1 覆わない
    const bool cacheable = smax <= 128 && q <= 128;
    if (cacheable && cache[smax][q]) return cache[smax][q] > 0;
    std::vector<char> hit(q, 0); int n = 0;
    for (int x = -smax; x <= smax; ++x)
        if (oz2_in_sparse(x, smax)) { const int r = ((x % q) + q) % q; if (!hit[r]) { hit[r] = 1; ++n; } }
    if (cacheable) cache[smax][q] = (n == q) ? 1 : -1;
    return n == q;
}
// (E) の桁の作り方（整数で．カーネルの Rnd と FP32 の丸めは tools/modround_check.cpp が別に写して比べる）．
// r は法 q^2 の剰余（|r| <= floor(q^2/2)+1 を想定），q は奇数．
static inline void oz2_sqs_split(long r, int q, long& l, long& h) {
    // h0 = round(r/q)．q が奇数なので r/q は半整数にならず，同点は無い
    long h0 = (2 * r + (r >= 0 ? q : -q)) / (2 * q);
    long t = r - (long)q * h0;
    const long st = t > 0 ? 1 : (t < 0 ? -1 : 0);
    if ((t & 1) && (t >= 17 || t <= -17)) { t -= st * q; h0 += st; }
    const long sh = h0 > 0 ? 1 : (h0 < 0 ? -1 : 0);
    if ((h0 & 1) && (h0 >= 17 || h0 <= -17)) h0 -= sh * q;
    l = t; h = h0;
}
// (E) の桁の上限（l と h それぞれの最大絶対値）．|r| <= floor(m/2)+1 のすべての r で数え上げる
// （スライサの最後の ModRound は大きい法で 1 だけはみ出しうる．ozaki2/kernels/oz2_digits.h）．
static inline void oz2_sqs_bounds(int q, int& dl, int& dh) {
    // 法の選び直しが同じ q を何度も引くので，q ごとに 1 度だけ数える（q <= 64 は S_32 の外まで含む上位集合）
    static int cache[65][2] = {};
    static bool done[65] = {};
    if (q > 0 && q <= 64 && done[q]) { dl = cache[q][0]; dh = cache[q][1]; return; }
    const long m = (long)q * q, R = m / 2 + 1;
    dl = dh = 0;
    for (long r = -R; r <= R; ++r) {
        long l, h; oz2_sqs_split(r, q, l, h);
        dl = std::max<int>(dl, (int)(l < 0 ? -l : l));
        dh = std::max<int>(dh, (int)(h < 0 ? -h : h));
    }
    if (q > 0 && q <= 64) { cache[q][0] = dl; cache[q][1] = dh; done[q] = true; }
}

// 平衡 ndig 桁で扱える法の上限 q^ndig（q = 2*dmax+1）
static inline long oz2_bal_cap(int dmax, int ndig) {
    long q = 2L * dmax + 1, c = 1;
    for (int i = 0; i < ndig; ++i) c *= q;
    return c;
}
// スライサの最後の ModRound（ozaki2/kernels/oz2_digits.h の oz2_residue）が返しうる |r| の上界（解析）．
// |r| <= m/2 + |S|*2^-23（S は最後の ModRound の入力）で，3 桁なら |S| <= D0 + floor(m/2)(c0+c1)，
// 4 桁なら d3 の項の前に 1 度畳むので |S| <= m/2+2 + floor(m/2)*c2（c_j = 2^(20(j+1)) mod m）．
// 5 桁は d4 の項（係数 c3）を足して畳んでから d3 の項を足すので，最後の入力は 4 桁と同じ形で上界も同じ．
// D0 は最下位の桁 d0 の上界 2^19 + 2^16 + 1 である．桁を作る各段の T（3 本組）は上位 limb hi とその下の部分に分かれ，
// 丸めるのは hi だけなので，次の段は |T'| = 2^20 |T - round(hi)| <= 2^20 (1/2 + |T - hi|) になる．|T - hi| は
// 初段（FP64 を bit で割る．f64_tf.h の split_f64）では hi の 23 bit の仮数を最近接に丸めた残りで ulp_FP32(hi) 以下，
// 2 段目からは tfMulP2 の TwoSum の正規化の誤差 ulp(hi)/2 + ulp(中位+下位)/2 以下で，どちらも |hi| と |中位+下位| が
// 2^20 未満なので 2^-4 以下である．したがって |T'| <= 2^19 + 2^16 で，それを丸めた桁は 2^19 + 2^16 + 1 以下になる．
// 中位以下を無視して残差を 1/2 以下と数えると上界を下回る（b=54・行の最大値 1 で
// x = (1178*2^40 + 519718*2^20 + 532480)*2^-54 は d0 = 532480 > 2^19 + 2^9 になり，法 2357 で S = 4177212）．
// 桁の上限をこの値で数える方式（基数 B の Karatsuba と平衡 n 桁の法ごとの桁数）だけが使い，
// 平方数・WIDE は floor(m/2) で数える（実際の最大は tools/modround_check.cpp の (c) が確かめる．d0 は 2^20 まで振る）．
// 上の解析（D0 と |T - hi| <= 2^-4）は桁の基数 2^20 のもの（OZ2_DIGIT_BITS．基数を変えるなら D0 の導出も直す）
static_assert(OZ2_DIGIT_BITS == 20, "oz2_resid_bound の D0 と残差の上界は 2^20 進の桁の解析");
static inline long oz2_resid_bound(int m) {
    long c0 = 1 % m, c1, c2;
    for (int i = 0; i < OZ2_DIGIT_BITS; ++i) c0 = (c0 * 2) % m;
    c1 = (c0 * c0) % m; c2 = (c1 * c0) % m;
    const double u = std::ldexp(1.0, -23) * (1.0 + std::ldexp(1.0, -23));
    const double D0 = 524288.0 + 65536.0 + 1.0;
    const double S3 = D0 + (double)(m / 2) * (double)(c0 + c1);
    const double S4 = m / 2.0 + 2.0 + (double)(m / 2) * (double)c2;
    const double e = std::max(S3, S4) * u;
    return std::min<long>((long)std::floor(m / 2.0 + e), m / 2 + 1);
}
// 平衡 n 桁の法ごとの桁数：m <= q^n かつ |r| <= (q^n-1)/2 となる最小の n（1..nmax）．足りなければ nmax+1．
// BFP2_b（nmax=5）は法ごとに n を最小に取る（5 桁なら積 25 本のところ，法 81 以下は 4 桁で
// 積 16 本，27 以下は 3 桁で 9 本で済む．docs/awe_moduli.md）．
static inline int oz2_bal_ndig(int m, int dmax, int nmax) {
    const long R = oz2_resid_bound(m);
    for (int n = 1; n <= nmax; ++n) {
        const long c = oz2_bal_cap(dmax, n);
        if (m <= c && 2 * R + 1 <= c) return n;
    }
    return nmax + 1;
}

static inline int oz2_digit_bound_rnd(int m, int q) {    // 方式 (A)/(C)
    int rs = m / 2;                       // 対称剰余の絶対値の上限
    int h  = (rs + q / 2) / q;            // round(rs/q) の上限（切り上げ側で評価）
    return std::max(h, q / 2);
}
static inline int oz2_digit_bound_kar(int m) {           // 方式 (B)
    int rs = m / 2;
    int h  = (rs + 15) / 16;              // |h| <= ceil((m/2)/16)
    // |l| <= 15 は m に依らない（h が切り上げなので l は逆符号で 16 に届かない）．
    // |s| = |h+l| <= max(|h|,|l|) も同じ理由でこの上限に収まる．m が小さいと |h| より
    // |l| のほうが大きくなるので，max を取らないと上限を過小評価する．
    return std::max(h, 15);
}
// 基数 B の Karatsuba（切り上げ．カーネルは基数を実行時引数で受けるので B は法ごとに選べる）の桁の上限：
// |h| <= ceil(R/B)，|l| <= B-1，|s| = |h+l| <= max（h と l は逆符号）．R は oz2_resid_bound．
// 桁が 15 より狭い経路（BFP4_b の d=7）が B = d+1 = 8 で使う（m <= 2d(d+1)+1 = 113）．桁 16 以上の経路は
// 基数 16 の方式（上の oz2_digit_bound_kar）を使う．
static inline int oz2_kar_base(int dmax) { return (dmax && dmax < 15) ? dmax + 1 : 16; }
static inline int oz2_digit_bound_kar_b(int m, int B) {
    const long R = oz2_resid_bound(m);
    return std::max<int>((int)((R + B - 1) / B), B - 1);
}
static inline int oz2_isqrt(int m) {                   // floor(sqrt(m))
    int r = (int)std::sqrt((double)m);
    while ((long)(r + 1) * (r + 1) <= m) ++r;
    while ((long)r * r > m) --r;
    return r;
}
static inline int oz2_is_square(int m) {
    const int r = oz2_isqrt(m);
    return ((long)r * r == m) ? r : 0;                // 平方数なら sqrt(m)，でなければ 0
}
// 最近接丸めの平衡桁で使う基数：桁の絶対値の上限を最小にする q を sqrt(m) 近傍から選ぶ
static inline int oz2_digit_base_rnd(int m) {
    const int r = oz2_isqrt(m);
    int best = r + 1, bb = oz2_digit_bound_rnd(m, r + 1);
    for (int qq = r; qq <= r + 3; ++qq) {
        int v = oz2_digit_bound_rnd(m, qq);
        if (v < bb) { bb = v; best = qq; }
    }
    return best;
}
// dmax は運ぶ形式が無誤差に表せる桁の上限．基数 16 の Karatsuba は桁が 15 まで伸びるので，
// dmax がそれより小さい経路（BFP4_b の d=7）では選べない（そのときの選び方は下の BFP4_b の段落）．
// smax（疎な桁の上限．経路の sparse_max）が 0 でなければ，稠密な桁（dmax 以下）で運べない奇数 q の
// 平方数を (E) OZ2_SPLIT_SQS にする．積・桁行列・合成は SQ と同じなので，それらを数える関数
// （oz2_product_terms など）は smax を受け取らず SQ として扱う．SQS を区別するのは，使える法の判定
// （oz2_usable）・桁の上限（oz2_plane_bound）・スライサへ渡す方式の旗・表示だけである．
// 桁が 15 より狭い 2 桁の経路（BFP4_b，d=7）では（docs/awe_moduli.md），
// 平方数で桁が収まらなければ非平方数と同じ扱いにし，基数 B = d+1 の Karatsuba（m <= 113）→ WIDE（225 まで）→
// 平衡 3 桁（基数 2d+1 = 15．m <= 3375．OZ2_SPLIT_BAL で桁行列 3 枚・積 9 本，m | 15^3 なら 6 本）の順に当てる．
// 桁 15 以上の経路（FP8・BF16 系）にはこの規則を当てない．
static inline int oz2_split_mode(int m, int dmax = 0, Oz2Ndig ndig = {}, int smax = 0) {
    if (ndig.n > 2)              return OZ2_SPLIT_BAL;
    if (ndig.one && dmax && oz2_resid_bound(m) <= dmax) return OZ2_SPLIT_ONE;   // 1 桁（(F)）
    const bool narrow = dmax && dmax < 15;
    if (const int q = oz2_is_square(m)) {
        if (smax && dmax && oz2_digit_bound_rnd(m, q) > dmax && (q & 1) && oz2_sparse_covers(q, smax))
            return OZ2_SPLIT_SQS;
        if (!narrow || oz2_digit_bound_rnd(m, q) <= dmax) return OZ2_SPLIT_SQ;
    }
    if (narrow) {
        if (oz2_digit_bound_kar_b(m, oz2_kar_base(dmax)) <= dmax) return OZ2_SPLIT_KAR;
        if (oz2_digit_bound_rnd(m, oz2_digit_base_rnd(m)) <= dmax) return OZ2_SPLIT_WIDE;
        if (oz2_bal_ndig(m, dmax, 3) <= 3) return OZ2_SPLIT_BAL;
        return OZ2_SPLIT_WIDE;           // 使えない（oz2_usable が落とす）
    }
    const int kcap = dmax ? std::min(16, dmax) : 16;
    if (oz2_digit_bound_kar(m) <= kcap) return OZ2_SPLIT_KAR;
    return OZ2_SPLIT_WIDE;
}
// 平衡 n 桁の法の桁数（BFP2_b は法ごとに最小の n，2 桁の経路の BAL は 3 桁）．BAL 以外は 0
static inline int oz2_bal_digits(int m, int dmax, Oz2Ndig ndig) {
    if (ndig.n > 2) return oz2_bal_ndig(m, dmax, ndig.n);
    return oz2_split_mode(m, dmax, ndig) == OZ2_SPLIT_BAL ? 3 : 0;
}
static inline int oz2_digit_base(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    if (ndig.n > 2) return 2 * dmax + 1;          // 平衡表現の基数
    const int mode = oz2_split_mode(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) return 1;        // 1 桁（基数は使わない．0 は INT8 経路の印なので 1）
    if (mode == OZ2_SPLIT_BAL) return 2 * dmax + 1;
    if (mode == OZ2_SPLIT_SQ) return oz2_is_square(m);
    return (mode == OZ2_SPLIT_KAR) ? oz2_kar_base(dmax) : oz2_digit_base_rnd(m);
}
static inline int oz2_digit_bound(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    // 平衡 n 桁なら，法が q^n に収まる桁数 n があれば桁は dmax に収まる．収まらない法は
    // dmax+1 を返して usable の判定から落とす．
    if (ndig.n > 2) return (oz2_bal_ndig(m, dmax, ndig.n) <= ndig.n) ? dmax : dmax + 1;
    int mode = oz2_split_mode(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) return (int)oz2_resid_bound(m);
    if (mode == OZ2_SPLIT_BAL) return (oz2_bal_ndig(m, dmax, 3) <= 3) ? dmax : dmax + 1;
    if (mode == OZ2_SPLIT_KAR) return (dmax && dmax < 15) ? oz2_digit_bound_kar_b(m, oz2_kar_base(dmax))
                                                          : oz2_digit_bound_kar(m);
    return oz2_digit_bound_rnd(m, oz2_digit_base(m, dmax, ndig));
}
// 法 m を経路で運べるか．稠密な桁（dmax 以下）に収まるか，疎な桁の方式 (E) に当たるか．
// dmax=0（INT8 経路）は桁分割しないので常に真（法の上限は呼び出し側が見る）．
static inline bool oz2_usable(int m, int dmax, Oz2Ndig ndig, int smax) {
    if (!dmax) return true;
    if (oz2_split_mode(m, dmax, ndig, smax) == OZ2_SPLIT_SQS) return true;
    return oz2_digit_bound(m, dmax, ndig) <= dmax;
}
// 桁行列 plane（法の中の番号．0=l, 1=h, 2=s）の桁の絶対値の上限．桁積の和の上界（連結の本数 maxg・
// 合成の条件・累算の条件）に使う．1 桁の法を混ぜる経路（下）を除き，smax=0 の経路は経路の上限 dmax を
// そのまま返す（TF32・BFP4_b・BFP2_b は経路一律の上限で連結の仕方を決める）．疎な桁の経路（FP8）は法ごとに
// 見る：SQS は数え上げた l・h それぞれの最大，それ以外は oz2_digit_bound（16 以下）．
static inline int oz2_plane_bound(int m, int plane, int dmax, Oz2Ndig ndig, int smax) {
    // 1 桁の法を混ぜる経路（Oz2Ndig::one．桁 32 の AWE 経路）は桁行列ごとに見る（連結の上限を法ごとに決めるため）．剰余の上界は oz2_resid_bound
    // （超過 +1 を含む）で数える：1 桁は R，平方数・WIDE の下位桁は q/2・上位桁は round(R/q)，Karatsuba は 15・ceil(R/16)・max．
    if (ndig.one && !smax) {
        const int mode = oz2_split_mode(m, dmax, ndig), R = (int)oz2_resid_bound(m);
        if (mode == OZ2_SPLIT_ONE) return R;
        if (mode == OZ2_SPLIT_KAR) { const int h = (R + 15) / 16; return plane == 0 ? 15 : (plane == 1 ? h : std::max(h, 15)); }
        const int q = oz2_digit_base(m, dmax, ndig);
        return plane == 0 ? q / 2 : (R + q / 2) / q;
    }
    if (!smax) return dmax;
    if (oz2_split_mode(m, dmax, ndig, smax) == OZ2_SPLIT_SQS) {
        int dl, dh; oz2_sqs_bounds(oz2_is_square(m), dl, dh);
        return plane == 0 ? dl : dh;
    }
    return oz2_digit_bound(m, dmax, ndig);
}
// 法 1 本を運ぶ桁行列の枚数と，それに要する行列積の本数
static inline int oz2_digit_planes(int m, int dmax = 0, Oz2Ndig ndig = {}) {
    if (const int n = oz2_bal_digits(m, dmax, ndig)) return n;   // 平衡 n 桁（法ごとの n）
    const int mode = oz2_split_mode(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) return 1;
    return mode == OZ2_SPLIT_KAR ? 3 : 2;
}
// 桁積の本数．並びを決める oz2_product_terms（下）の長さそのものなので，
// そちらから導く．本数の規則をここへもう 1 度書くと，列挙と本数がずれうる．
static inline int oz2_digit_prods(int m, int dmax = 0, Oz2Ndig ndig = {});

// 桁積の作り方と，合成の係数．**同じ (i,j) の列挙を 1 度だけ行う．**
// 積 j は A の桁行列 a_plane と B の桁行列 b_plane の積で（0=l, 1=h, 2=s(=h+l)．
// 法の桁行列の先頭 poff[] からの相対），合成では係数 coeff_lo を掛けて足す．
// coeff_hi は同じ係数に 2^12（OZ2_COMPOSE_SPLIT）を掛けたもの（剰余を 2 段に分けて取るため）．
//   SQ/WIDE : (l,l), (h,l), (l,h) （WIDE のみ (h,h) を足して 4 本）
//   KAR     : (l,l), (h,h), (s,s)
//   平衡 n 桁 : (i,j) を総当たり．m が q^n を割るなら i+j >= n は消える
//
// 桁行列の並びと係数を別々の列挙に書くと，片方だけ直したときに桁行列と係数がずれ，
// 答えが静かに変わる．本数を返す oz2_digit_prods と，連結した組を作る oz2_product_groups（下）も
// 同じ理由でここから導く．
struct Oz2ProductTerm {
    int a_plane = 0, b_plane = 0;    // A・B の桁行列（poff からの相対）
    float coeff_lo = 0, coeff_hi = 0;  // 合成の係数と，その 2^12 倍（どちらも法で畳んである）
};
// 合成段の 2^12 分割の基数（ホストの係数 coeff_hi = c*2^12 mod m とカーネルの hi = round(P/2^12) が同じ値を見る．
// 値は oz2_limits.h の OZ2_COMPOSE_SPLIT_BITS）
inline constexpr long OZ2_COMPOSE_SPLIT = 1L << OZ2_COMPOSE_SPLIT_BITS;

static inline uint32_t oz2_product_terms(int m, Oz2ProductTerm t[OZ2_MAXPROD],
                                         int dmax = 0, Oz2Ndig ndig = {}) {
    auto rm = [&](long v) { long r = v % m; return r < 0 ? r + m : r; };
    uint32_t k = 0;
    // 対称代表元 (-m/2, m/2]．平衡 n 桁（BFP2_b・BFP4_b）と桁 15 未満の Karatsuba（BFP4_b）の係数に使う：
    // 先に法へ畳む合成の総和 Σ|c_j|(m/2+2) < 2^22 を保つため（15^3 に近い法の 15^4 mod m などは [0,m) で持つと
    // m に近い値になる）．他の経路の係数は [0,m)（結果のビット列を変えないため）．
    auto sm = [&](long v) { long r = rm(v); return r > m / 2 ? r - m : r; };
    if (const int n = oz2_bal_digits(m, dmax, ndig)) {   // 平衡 n 桁（法ごとの n）．係数は q^(i+j) mod m
        const long q = 2L * dmax + 1;
        const bool drop = (oz2_bal_cap(dmax, n) % m == 0);
        // 桁数が経路の上限を超える法（oz2_bal_ndig が nmax+1 を返す．oz2_usable が落とすので選ばれない）を渡されると，
        // 積の本数が枠 OZ2_MAXPROD を超えて t の外へ書く．呼び出し側の誤りなので止める
        const int nterm = drop ? n * (n + 1) / 2 : n * n;
        if (nterm > OZ2_MAXPROD) {
            std::fprintf(stderr, "ERROR: oz2_product_terms: 法 %d の平衡 %d 桁の積 %d 本が枠 %d を超える（使えない法）\n",
                         m, n, nterm, OZ2_MAXPROD);
            std::abort();
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (drop && i + j >= n) continue;
                long c = 1;
                for (int e = 0; e < i + j; ++e) c = rm(c * q);
                t[k++] = {i, j, (float)sm(c), (float)sm(c * OZ2_COMPOSE_SPLIT)};
            }
        return k;
    }
    const int mode = oz2_split_mode(m, dmax, ndig), q = oz2_digit_base(m, dmax, ndig);
    if (mode == OZ2_SPLIT_ONE) {          // 1 桁：(r,r)，係数 1
        t[k++] = {0, 0, 1.f, (float)rm(OZ2_COMPOSE_SPLIT)};
        return k;
    }
    if (mode == OZ2_SPLIT_KAR) {          // (l,l), (h,h), (s,s)．係数は 1-B, B^2-B, B（基数 16 なら -15, 240, 16）
        const long B = q;
        if (dmax && dmax < 15) {          // 桁 15 未満の経路（BFP4_b．基数 8）は対称代表元
            const long c[3] = {sm(1 - B), sm(B * B - B), sm(B)};
            for (int j = 0; j < 3; ++j) t[k++] = {j, j, (float)c[j], (float)sm(c[j] * OZ2_COMPOSE_SPLIT)};
            return k;
        }
        const long c[3] = {rm(1 - B), rm(B * B - B), rm(B)};
        for (int j = 0; j < 3; ++j) t[k++] = {j, j, (float)c[j], (float)rm(c[j] * OZ2_COMPOSE_SPLIT)};
        return k;
    }
    // SQ/WIDE：(l,l), (h,l), (l,h)［, (h,h)］．係数は 1, q, q［, q^2］
    const long c[4] = {rm(1), rm(q), rm(q), rm((long)q * q)};
    const int  pa[4] = {0, 1, 0, 1}, pb[4] = {0, 0, 1, 1};
    const uint32_t n = (mode == OZ2_SPLIT_WIDE) ? 4u : 3u;
    for (uint32_t j = 0; j < n; ++j) t[k++] = {pa[j], pb[j], (float)c[j], (float)rm(c[j] * OZ2_COMPOSE_SPLIT)};
    return k;
}

// 本数だけの取り出し口．並びも本数も上の 1 つで決まる．
static inline int oz2_digit_prods(int m, int dmax, Oz2Ndig ndig) {
    Oz2ProductTerm t[OZ2_MAXPROD];
    return (int)oz2_product_terms(m, t, dmax, ndig);
}

// --- 同じ係数の桁積を K 方向に連結して 1 本の GEMM にまとめる ---------------------
// 合成は D = Σ_j c_j P_j（mod m）で，係数 c_j は桁の位 i+j（2 桁分割の平方数・WIDE は q^(i+j)，
// 平衡 n 桁は q^(i+j)）だけで決まる．位が同じ桁積どうしは合成の前に足してよく，その和は
//     Σ_{i+j=k} A_i B_j = [A_{i0} | A_{i0+1} | ...] · [B_{k-i0} ; B_{k-i0-1} ; ...]
// と，A の桁行列を K 方向に並べたものと B の桁行列を逆順に K 方向に積んだものの 1 本の積になる
// （尾崎 I の対角の連結と同じ形）．GEMM の MAC 数は変わらず，出力（と合成が読む平面）の枚数が減る．
// そのためにスライスの置き方を変える（ozaki2/kernels/modslice_writer.cpp）：A は法の中の桁行列を
// 番号順に K 方向へ連結し（行 r・桁行列 p・列 c のタイルは r*(P*Kt) + p*Kt + c），B は法の中で
// 桁行列を逆順に積む（桁 j を位置 np-1-j に置く）．するとまとめる組の A の桁 i0..i0+g-1 は連続し，
// B の位置 np-1-k+i0.. も連続する．Karatsuba の 3 本（位 0,2,4）は係数が別々なのでまとまらない．
// 無誤差条件：1 本の出力は g 本の桁積の和で |Σ| <= g*K*d^2 なので，これが FP32 の無誤差整数に，
// しかも合成が先に法へ畳めるよう法を足しても 2^24 以下に収まること（g <= maxg．下の選択が決める）．
struct Oz2ProductGroup {
    int a0 = 0, b0 = 0, g = 1;         // A の桁行列 a0..a0+g-1（法の先頭 poff から），B の位置 b0..b0+g-1
    float coeff_lo = 0, coeff_hi = 0;  // 位の係数とその 2^12（OZ2_COMPOSE_SPLIT）倍（どちらも法で畳んである）
};
static inline uint32_t oz2_product_groups(int m, Oz2ProductGroup gr[OZ2_MAXPROD],
                                          int dmax, Oz2Ndig ndig, int maxg) {
    Oz2ProductTerm t[OZ2_MAXPROD];
    const uint32_t n = oz2_product_terms(m, t, dmax, ndig);
    const int np = oz2_digit_planes(m, dmax, ndig);
    uint32_t k = 0;
    // 位 key = a+b ごとに，a の昇順に並べてから maxg 本ずつに切る
    for (int key = 0; key <= 2 * (np - 1); ++key) {
        int as[OZ2_MAXPROD]; float lo = 0, hi = 0; int na = 0;
        for (uint32_t j = 0; j < n; ++j)
            if (t[j].a_plane + t[j].b_plane == key) {
                if (na && (t[j].coeff_lo != lo || t[j].coeff_hi != hi)) return 0;   // 位が同じで係数が違う（起きない）
                as[na++] = t[j].a_plane; lo = t[j].coeff_lo; hi = t[j].coeff_hi;
            }
        std::sort(as, as + na);
        for (int i = 0; i < na; ) {
            int g = 1;
            while (i + g < na && g < maxg && as[i + g] == as[i] + g) ++g;   // 連続する a だけをまとめる
            gr[k].a0 = as[i]; gr[k].b0 = np - 1 - (key - as[i]); gr[k].g = g;
            gr[k].coeff_lo = lo; gr[k].coeff_hi = hi; ++k;
            i += g;
        }
    }
    return k;
}
static inline uint32_t oz2_nout(int m, int dmax, Oz2Ndig ndig, int maxg) {
    Oz2ProductGroup gr[OZ2_MAXPROD];
    return oz2_product_groups(m, gr, dmax, ndig, maxg);
}
// 連結した出力 1 本（桁積 g 本の和）の絶対値の上界 Σ_t K*d(a0+t)*d(b)．d は桁行列ごとの桁の上限
// （oz2_plane_bound．smax=0 の経路では全部 dmax なので g*K*dmax^2）．B の位置 b0+t は法の中で桁を
// 逆順に積んだ位置なので，桁の番号は np-1-(b0+t) である．
static inline double oz2_group_bound(int m, const Oz2ProductGroup& g, double K, int dmax, Oz2Ndig ndig, int smax) {
    const int np = oz2_digit_planes(m, dmax, ndig);
    double sb = 0;
    for (int t = 0; t < g.g; ++t)
        sb += K * oz2_plane_bound(m, g.a0 + t, dmax, ndig, smax) * oz2_plane_bound(m, np - 1 - (g.b0 + t), dmax, ndig, smax);
    return sb;
}
// 連結の上限：同じ位の桁積を K 方向に何本まとめて 1 本の GEMM にしてよいか（oz2_product_groups の maxg）．1 本の出力
// （桁積 g 本の和）の上界 g*K*d_l*d_h に法 cap を足しても FP32 の無誤差整数 2^24 以下に収まる最大の g（1 以上）で，
// 合成段が先に法へ畳める余裕を残す．d_l・d_h は法 m の下位・上位の桁行列の桁の上限（oz2_plane_bound．2 桁分割で
// まとまるのは位 1 の l×h だけ）で，疎な桁（smax）も 1 桁の法を混ぜること（ndig.one）もない経路では経路一律の dmax になる．
// 引く法 cap は呼び出し側で違う：法を選ぶ途中の見積り（ozaki2_moduli.h の oz2_reselect_tail・ozaki2_awe.h の候補）は
// 法の上限 m_cap，選んだ集合の値（ozaki2_moduli.h の oz2_plan_concat の maxg_m）は集合の最大の法．1 桁の法は連結しないので，
// 呼び出し側が 1 にする
static inline int oz2_maxg_for(int m, uint32_t K, int cap, int dmax, Oz2Ndig ndig, int smax) {
    const double pb = (double)oz2_plane_bound(m, 0, dmax, ndig, smax) *
                      oz2_plane_bound(m, std::min(1, oz2_digit_planes(m, dmax, ndig) - 1), dmax, ndig, smax);
    return std::max(1, (int)std::floor((std::ldexp(1.0, 24) - cap) / ((double)K * pb)));
}

// --- 合成段の平面の枚数と，DRAM の見積り ----------------------------------------------
// ドライバ（ozaki2_plan.hpp の make_compose_plan）と，法の集合を選ぶ段の時間のモデル（ozaki2_awe.h の
// oz2_awe_choose）と tools/awe_plan.cpp が同じこの規則で数える．DRAM の見積り oz2_dram_one_group は A・B をスライスの後で
// 手放す前提のモデルの条件で，実行時の帯の数は A・B を残す同時確保で決める（ozaki2_plan.hpp の plan_bands．モデルが
// 1 本で収まると見込んだ集合も，実行時は行の帯に分けることがある．帯は答えも GEMM の本数も変えない）．
// 合成段が読む桁積の平面は，法 1 本あたりの GEMM の出力（同じ位の桁積を連結した後）の最大 mx で決まる（1 桁の法は合成段を
// 通らないので数えない）．合成する法が無ければ 0 枚．あれば下限を OZ2_COMPOSE_MIN にする（連結で出力が 2 本になる構成でも，
// 平面の枚数と合成段の CB の深さを 3 本のときと同じに保つ）．mx が OZ2_COMPOSE_NOSPLIT_MAX を超える構成（連結しない平衡 5 桁は
// 法あたり 25 本（法 q^n だけ 15 本）で，全部を同時に置くと 14080^3 で 19.8 GB）は OZ2_COMPOSE_CHUNK 本ずつに切り，部分和を
// 残す平面を足す（連結すれば位の数 9 本・5 本で 7.1 GB なので切らない．2 桁の経路は 4 本以下）．
inline constexpr uint32_t OZ2_COMPOSE_MIN = 3;
inline constexpr uint32_t OZ2_COMPOSE_NOSPLIT_MAX = 12;
inline constexpr uint32_t OZ2_COMPOSE_CHUNK = 5;   // 15 と 25 の公約数
struct Oz2ComposePlan {
    uint32_t max_nout = 0;   // 法 1 本あたりの出力の最大（下限を当てた後．合成する法が無ければ 0）
    uint32_t chunk = 0;      // 一度に合成する本数
    uint32_t planes = 0;     // 確保する平面の枚数（桁積 chunk 枚＋部分和）
};
constexpr Oz2ComposePlan oz2_compose_plan(bool composes, uint32_t mx) {
    Oz2ComposePlan p;
    if (!composes) return p;
    p.max_nout = mx > OZ2_COMPOSE_MIN ? mx : OZ2_COMPOSE_MIN;
    p.chunk = p.max_nout > OZ2_COMPOSE_NOSPLIT_MAX ? OZ2_COMPOSE_CHUNK : p.max_nout;
    p.planes = p.chunk < p.max_nout ? p.chunk + (p.max_nout + p.chunk - 1) / p.chunk : p.max_nout;
    return p;
}
// これから確保する DRAM の量［B］（FP64 のオペランドと小さい器は確保済みの時点から数える）．P は桁行列 1 枚ぶん（A と B），
// out は出力の平面 1 枚，F は FP64 のオペランド（A と B），積の器 Q =（合成段の平面 + 剰余 S 枚）× out（INT8 経路は合成段が
// 無いので C_i の S 枚）．余裕は両スキーム共通の OZ_DRAM_MARGIN_BYTES（common/oz_dram_margin.h．0.5 GiB）．
//   1 組：nplane*P + max(0, Q - F) + 余裕（スライスの後に FP64 を手放してから Q を取る）
//   n 組：組の桁行列の枚数の最大*P + Q + 余裕（FP64 は最後の組を切るまで残し，Q は先に取る）
// 値はどれも 2^53 未満の整数なので double で厳密に数えられる
inline constexpr double OZ2_DRAM_MARGIN = (double)OZ_DRAM_MARGIN_BYTES;
constexpr double oz2_dram_one_group(double nplane, double P, double compose_planes, double S, double out, double F) {
    const double Q = (compose_planes + S) * out;
    return nplane * P + (Q > F ? Q - F : 0.0) + OZ2_DRAM_MARGIN;
}
constexpr double oz2_dram_groups(double maxplanes, double P, double compose_planes, double S, double out) {
    return maxplanes * P + (compose_planes + S) * out + OZ2_DRAM_MARGIN;
}
