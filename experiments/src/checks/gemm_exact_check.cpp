// SPDX-License-Identifier: Apache-2.0
//
// GEMM 段の無誤差累積の検査（パッカの L1 累積，両端の 2 コアから配るデータ移動，対ループ・融合の確認）
// 【検査】通しの selftest の段で scripts/diagnose/gemm_exact.sh（diagnose/probes.sh の probe_gemm_exact）が回し，論文の b_acc の出どころになる．
// ---------------------------------------------------------------------------
// 尾崎の全経路が使う行列積（common/mcast_gemm.hpp の spill_gemm_spec＋build_mcast_gemm と
// common/kernels/bmm_int32_spill.cpp）と，BF16 extended・3xTF32 の計算カーネル
// （common/kernels/bmm_l1acc.cpp．l1acc_gemm_spec）を，本番と同じ構成で整数入力に対して回し，
// 出力の全要素を int64 の参照と比べる．尾崎の到達精度の検査（selftest.sh）は
// 最終結果の相対誤差しか見ないので，K 方向の累積が 1 回だけ丸まるような壊れ方は
// 埋もれうる．ここでは累積そのものを見る．
//
// 判定は「復号した整数値の一致」（出力の値が参照の整数と等しいか）であって，出力語のビット一致では
// ない：FP32 の ±0，符号-絶対値 Int32 の ±0 は区別しない．ビット一致を主張する変更には，生の出力を
// 旧版・新版で比べる別の検査を用意すること（docs/nongemm_opt_results.md の §0）．
//
//   ./build/gemm_exact_check M N K fmt pattern bw seed [mode] [runs]
//     fmt     fp32 | bf16 | fp16 | bfp8 | fp8 | fp8s | bfp4 | bfp2 | int8   （入力の形式．本番の 8 形式
//             （fp8s は e4m3 に疎な桁を入れた別の入力））
//             fp32 は TF32 として乗算される．fp16 は binary16 のビット列で運ぶ（尾崎の FP16 経路と同じく
//             GEMM は Float16 として読む）．fp8 は e4m3，bfp4・bfp2 は共有指数のブロック浮動小数．
//             fp8s は同じ e4m3 で，尾崎 II の FP8 経路の疎な桁（|x| <= 16 か 32 以下の偶数．
//             docs/fp8_sparse_digits.md）を入れる：rand はその 49 通りから一様に，d は 32，e は 15．
//     pattern 入力の作り方．e は pos・odd などに使う奇数の値（d 以下の最大の奇数：int8 127，fp32・bf16・fp16・
//             bfp8 31，fp8 15，bfp4 7，bfp2 1．奇数にしてあるのは積の和の最下位 bit まで使わせるため．32 だと和が 2^10 の倍数になり，
//             仮数が 14 bit しか無くても丸まらない）．d は rand・pm1・zface の幅（int8 127，fp32・bf16・
//             fp16・bfp8 32，fp8 16，bfp4 7，bfp2 1．本番の桁の上限）
//             rand      [-d,d] の一様乱数
//             pm1       ±d を乱数で（部分和が大きく振れる）
//             pos       すべて +e（同符号．部分和は単調に増える）
//             neg       A は +e，B は -e（pos を負側へ写したもの）
//             cancel    A はすべて +d，B は前半 +d・後半 -d（列ごとに順を逆にした列を混ぜる）．
//                       最終値は 0 だが，途中の和は K/2*d^2 まで振れる
//             odd       A はすべて +e，B は k=0 だけ +e で残りは +(e-1)．積は k=0 だけ奇数なので
//                       K 方向のどの途中の和も奇数になる（pos の途中の和は K ブロックの境界で
//                       32*e^2 の倍数なので，FP32 の加算でも 2^28 近くまで丸まらずに通ってしまう）
//             oddneg    odd を負側へ写したもの（B の符号を反転）
//             oddcancel B は k=0 が +e，前半 +(e-1)・後半 -(e-1)（列ごとに順を逆に）．途中の和は奇数のまま
//                       K/2*e(e-1) まで振れて戻る（大きい和へ逆符号のブロック和を足す）
//             maxodd    A・B とも k=0 以外は +d（桁の上限）で，k=0 の積だけを奇数の和になるように替える．
//                       d が偶数（fp32・bf16・fp16・bfp8・fp8・fp8s）なら k=0 は A・B とも +e で和は (K-1)d^2 + e^2，
//                       d が奇数（int8・bfp4・bfp2．e=d）なら k=0 は A が +d・B が +(d-1) で和は (K-1)d^2 + d(d-1)
//                       = K d^2 - d（A・B とも +d だと和 K d^2 は K が 32 の倍数なので偶数になり，奇数の検査にならない）．
//                       どちらも奇数で，桁の上限での最大の和（の近く）になる（尾崎 II の桁積の連結の上限 maxg の境目．
//                       連結した K = g*K_c で回す）
//             maxcancel A・B とも k=0 は maxodd と同じ積で，残りは A が +d，B が前半 +d・後半 -d（列ごとに順を逆に）．
//                       途中の和は K/2*d^2 まで振れて戻る．k>=1 の和は偶数列で -d^2，奇数列で +d^2 なので最終値は
//                       k=0 の積 p0（e^2 か d(d-1)）に対して p0 - d^2（偶数列）と p0 + d^2（奇数列）で，どちらも奇数
//             odd 系・maxodd・maxcancel は生成後に参照の全要素が奇数であることを確かめ，外れれば FAIL にする
//             （入力の生成を変えて検査の意図が崩れたのを見逃さないため．+re の実行は K の範囲が変わるので見ない）
//             zface     ±d の乱数に，K ブロックごとに零の面・零の行を混ぜる（ブロック b % 4 == 1 は A の
//                       タイル内の上半分の行が 0，== 2 はブロック全体が 0，== 3 は B の右半分の列が 0，
//                       K ブロックが 4 個以上なら最後のブロックも 0．A の行タイル % 3 == 1 は先頭ブロック以外が 0，% 5 == 4 は全部 0，
//                       B の列タイル % 4 == 3 は奇数ブロックが 0）．パッカの零の面の印（Disable_pack_zero_flags）が
//                       L1 累積の途中で戻ると，零の面を足さずに「0 を書いた」ことにしてしまう壊れ方を狙う
//                       （common/kernels/bmm_l1acc.cpp の注記）
//     bw      K ブロック幅［タイル］（0 なら構成探索に任せる）．K/(32*bw) がブロック数
//     seed    乱数の種
//     mode    plain（既定）  融合も対ループも使わない（尾崎 II・gemm_int8・尾崎 I の INT8 経路）
//             pl<b>          対ループ（尾崎 I の既定）．b は 1 対あたりの K ブロック数で，ブロック数を
//                            割り切ること．出力は対ごとの FP32 の和を Int32 で足したもの
//             fuse           融合（尾崎 I の OZ_NO_PAIRLOOP や長い K の経路）．実行 r の acc は
//                            r%3 != 0（0,1,1,0,...）で，acc=1 なら直前の出力に Int32 で足し込む
//             l1acc          gemm_bf16ext の計算カーネル（bmm_l1acc.cpp．HiFi4，FP32 出力．bf16・fp32 だけ）
//             bad<what>      構成をわざと壊して，組み立て（spill_gemm_spec・build_mcast_gemm）が止めるかを見る：
//                            badsh0（sh=0），badbw0（bw=0），badshdiv（sh が pm を割り切らない），
//                            badkdiv（1 < bw < Kt で bw が Kt を割り切らない．通れば K の末尾を切り捨てる形．
//                            割り切れの検査だけに掛かる．Kt <= 2 では作れないので引数の誤りとして終了コード 2 で
//                            止め REFUSED は出さない），badkbig（bw = Kt+1．K ブロックが 0 本の形），
//                            baddst（サブブロックをコアあたりの出力 pm x pn
//                            全体にして DST に載らなくする．pm*pn が DST の枚数以下の形では作れないので，引数の誤り
//                            として終了コード 2 で止め REFUSED は出さない）．
//                            それぞれ止まるべき検査の文言（下の kBadWant）で止まったときだけ "REFUSED" を出して
//                            終了コード 2．別の検査・別の例外で止まったら "REFUSED-ELSEWHERE ... FAIL" で終了コード 1
//                            （どの例外でも REFUSED にすると，別の理由で止まったのを拒否の成功と取り違える）
//             後ろに +re を付けると，同じプログラムを使い回して A/B の開始位置・K ブロック数（対ループは対の数
//             1→多数→1）・出力の面を実行ごとに付け替える（尾崎 I の gemm_run と同じ付け替え）．出力は 3 面
//             持ち，書かない面・まだ書いていない面は番兵のままかも見る（l1acc は K ブロック数がコンパイル時
//             引数なので付けられない）．+sep は c_16 と c_24 を別々の L1 領域にする（gemm_int8 と同じ）
//             pl・fuse は FP32 の部分積でしか組めないので int8 では止まる（spill_gemm_spec の検査）
//     runs    同じプログラムを，入力を替えて続けて回す回数（既定 plain・pl は 2，fuse は 4，+re は 6）．
//             乱数の入力は実行ごとに種を変え，決まった入力（pos など）は plain・pl の奇数回目で
//             B の符号を反転する．前の実行の状態（c_24・c_5 の輪，パッカの L1 累積の設定）が
//             持ち越されれば，2 回目以降が食い違う
// 出力の器は，acc=0 の実行の前にその実行が書く面を番兵で埋め直す．前の実行が書いた値を残したままだと，
// 2 回目以降の書き込みが丸ごと抜けても，実行ごとに同じ期待値になる入力（cancel の 0，zface の零の領域）では
// 一致してしまう．書かない面（+re）と acc=1 が読む面は埋め直さない（前者は書かれないこと，後者は足し込みを見る）．
// FP32 に累積する形式では，1 区間（plain・fuse は K 全体，pl は 1 対）の途中の和がすべて 2^24 以下で
// あれば無誤差のはずである．足す順序は本番のハードウェア（タイル・ブロック・行列エンジンの内部の
// 順序）と参照（k の昇順）で違うので，順序によらない十分条件 Σ|a_k b_k| <= 2^24（sum|seg|）と，参照の
// 順序での途中の和の最大（peak|seg|）を分けて出す．Int32 の累積（int8 の部分和，pl の対をまたぐ c_5，
// 融合・対ループの出力）は 2^31 未満（pl の c_5 は peak|acc|）．超える入力では範囲外と表示したうえで比べる．
// 食い違いは原因ごとに分けて数える：numeric（有限な整数の値が参照と違う．maxdiff はその |差| の最大で，
// 範囲外の入力で FP32 の丸めが入ると，その大きさでの FP32 の ulp 程度になる．2^24 を少し超えた和なら 1 か 2），
// nonnumeric（NaN・Inf・整数でない値），
// unwritten（書いたはずの面の要素が番兵のまま），overwritten（まだ書いていない面の番兵が壊れた），
// oddsum（参照の奇数性．入力の生成の検査）．範囲外の入力で食い違うことを見る検査（diagnose/gemm_exact.sh の
// mismatch）は numeric > 0 で，ほかがすべて 0 のときだけ期待どおりとみなす．
// 結果は "EXACT-CHECK ... PASS|FAIL" の 1 行（終了コードも 0/1．引数の誤りと bad<what> の拒否は 2，デバイスを開いた後の
// 想定外の例外（組み立て・転送・実行の失敗）は 3）．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat4.hpp>
#include "common/bfp2_decl.hpp"   // bfloat2.hpp の代わり（宣言だけ）
#include <tt-metalium/float8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "tt-metalium/core_coord.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

#include "common/mcast_gemm.hpp"   // build_mcast_gemm・spill_gemm_spec・l1acc_gemm_spec（ozaki_gemm_host.hpp も取り込む）
#include "common/host_eval.hpp"
#include "common/oz_driver.hpp"    // 入口の検査の並び（oz_drv::reject_env_gemm・check_env）
#include "testkit/probe_common.hpp"           // 形式の表 oz_format を名前で引く
#include "common/oz_bits.h"                  // binary16 との変換（half_bits）
#include "ozaki1/ozaki1_param.h"     // 尾崎 I の桁の上限の既定（oz1_default_digit_max．下の桁の上限の照合だけに使う）
#include "ozaki2/ozaki2_moduli.h"    // 尾崎 II の経路の表 OZ2_ROUTES・oz2_param（同上）

namespace {

// 形式ごとの入力の幅：d は rand・pm1・cancel・zface の幅（本番の桁の上限），e は pos・odd の値（奇数）．
// 形式そのもの（GEMM が読む DataFormat）は本番の表 oz_format（common/oz_format_table.hpp）の in から引く．d・e は検査の
// 期待値として独立に持ち（本番の定数に寄せると両方が同時に誤って通る），下の static_assert で本番の桁の上限と照合する
struct FmtInfo {
    const char* name;
    OzFmt fmt;
    int d, e;
    bool sparse = false;   // rand を e4m3 が無誤差に表せる整数（|x| <= d）から選ぶ（fp8s）
    tt::DataFormat df() const { return oz_format(fmt).in; }
};
constexpr FmtInfo kFmts[] = {
    {"fp32", OzFmt::FP32, 32, 31},  {"bf16", OzFmt::BF16, 32, 31},
    {"fp16", OzFmt::FP16, 32, 31},  {"bfp8", OzFmt::BFP8, 32, 31},
    {"fp8", OzFmt::FP8, 16, 15},    {"fp8s", OzFmt::FP8, 32, 15, true},
    {"bfp4", OzFmt::BFP4, 7, 7},
    {"bfp2", OzFmt::BFP2, 1, 1},    {"int8", OzFmt::INT8, 127, 127},
};
// 本番の桁の上限との照合．尾崎 I は ozaki_digit_max の既定（min(ceil(radix/2), exact_int_max)．ozaki1_param.h の
// oz1_default_digit_max），尾崎 II は形式の経路（ozaki2_split.h の oz2_param）の桁の上限 digit_bound（桁分割の 7 形式）．
// fp8s は尾崎 II の FP8 経路の疎な桁の上限（sparse_max），int8 は尾崎 II の INT8 経路の法の上限 255 の対称剰余
// （INT8 経路は桁分割しないので digit_bound は 0）
constexpr bool digits_match() {
    for (const auto& f : kFmts) {
        const Oz2Route r = oz2_param(f.fmt);
        if (f.sparse) {
            if (f.d != r.sparse_max) return false;
        } else {
            if (f.d != (int)oz1_default_digit_max(f.fmt)) return false;
            if (f.fmt == OzFmt::INT8 ? f.d != r.mod_cap / 2 : f.d != r.digit_bound) return false;
        }
        if (f.e > f.d || f.e % 2 == 0) return false;   // e は d 以下の奇数
    }
    return true;
}
static_assert(digits_match(), "gemm_exact_check の形式の表の d・e が本番の桁の上限と合わない（kFmts を直す）");

// 入力 1 組（A は M×K，B は K×N，行優先）を作る．run は何回目の実行か（上の runs の説明），
// bw は K ブロック幅［タイル］（zface がブロックの境目を知るため）．
bool make_inputs(const std::string& pat, const FmtInfo& f, uint32_t M, uint32_t N, uint32_t K, uint32_t bw,
                 uint32_t seed, uint32_t run, bool flip, std::vector<int>& A, std::vector<int>& B) {
    const int d = f.d, e = f.e;
    A.assign((size_t)M * K, 0);
    B.assign((size_t)K * N, 0);
    std::mt19937 rng(seed + 1000u * run);
    std::uniform_int_distribution<int> U(-d, d), S(0, 1);
    // fp8s：e4m3 が無誤差に表せる |x| <= d の整数（d=32 なら 49 通り）から一様に選ぶ
    std::vector<int> sparse_set;
    if (f.sparse)
        for (int x = -d; x <= d; ++x) { const int a = x < 0 ? -x : x, bl = a ? 32 - __builtin_clz((unsigned)a) : 0;
                                        if (bl <= 4 || a % (1 << (bl - 4)) == 0) sparse_set.push_back(x); }
    std::uniform_int_distribution<size_t> US(0, sparse_set.empty() ? 0 : sparse_set.size() - 1);
    auto draw = [&]() { return f.sparse ? sparse_set[US(rng)] : U(rng); };
    const bool rnd = pat == "rand" || pat == "pm1" || pat == "zface";
    const uint32_t kb = 32u * std::max(bw, 1u), nb = (K + kb - 1) / kb;
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t k = 0; k < K; k++) {
            int& a = A[(size_t)i * K + k];
            if (pat == "rand") a = draw();
            else if (pat == "pm1" || pat == "zface") a = S(rng) ? d : -d;
            else if (pat == "cancel") a = d;
            else if (pat == "maxodd" || pat == "maxcancel") a = (k == 0) ? (d % 2 ? d : e) : d;
            else a = e;   // pos・neg・odd・oddneg・oddcancel
            if (pat == "zface") {
                const uint32_t b = k / kb, it = i / 32;
                const bool zero = (b + 1 == nb && nb >= 4) || b % 4 == 2 || (b % 4 == 1 && i % 32 < 16) ||
                                  (it % 3 == 1 && b > 0) || it % 5 == 4;
                if (zero) a = 0;
            }
        }
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t j = 0; j < N; j++) {
            int& b = B[(size_t)k * N + j];
            const bool first = k < K / 2;
            const bool lead = (j & 1) ? !first : first;   // 奇数列は負から始める
            if (pat == "rand") b = draw();
            else if (pat == "pm1" || pat == "zface") b = S(rng) ? d : -d;
            else if (pat == "pos") b = e;
            else if (pat == "neg") b = -e;
            else if (pat == "cancel") b = lead ? d : -d;
            else if (pat == "odd") b = k == 0 ? e : e - 1;
            else if (pat == "oddneg") b = k == 0 ? -e : -(e - 1);
            else if (pat == "oddcancel") b = k == 0 ? e : (lead ? e - 1 : -(e - 1));
            // k=0 の積 p0：d が偶数なら e*e（奇数），奇数なら d*(d-1)（偶数．残り K-1 本の d^2 が奇数の個数だけ並ぶ）
            else if (pat == "maxodd") b = k == 0 ? (d % 2 ? d - 1 : e) : d;
            else if (pat == "maxcancel") b = k == 0 ? (d % 2 ? d - 1 : e) : (lead ? d : -d);
            else return false;
            if (pat == "zface") {
                const uint32_t blk = k / kb, jt = j / 32;
                if ((blk % 4 == 3 && j % 32 >= 16) || (jt % 4 == 3 && blk % 2 == 1)) b = 0;
            }
        }
    // 決まった入力は実行ごとに符号を替える（乱数は種を替えてあるのでそのまま）
    if (!rnd && flip)
        for (auto& b : B) b = -b;
    return true;
}

// 参照（int64）は oz_reference.hpp の int_reference_gemm（gemm_int8 と共有）．K 方向の範囲 [k0, k1) だけを足し（+re の
// 付け替え），区間の長さ seg［要素］ごとに途中の和を 0 から数え直した大きさ（peak_seg・sum_seg・peak_acc）も数える
// （対ループは 1 対ごとに FP32 のアキュムレータを積み直す）
using RefStats = oz_eval::IntRefStats;

// binary16 のビット列（整数値 |v| <= 2048 は無誤差．無誤差に表せない値はここでは作らない）．変換は common/oz_bits.h
uint16_t half_bits(float v) {
    const uint16_t h = half_from_f32(v);
    TT_FATAL(f32_bits(f32_from_half(h)) == f32_bits(v), "half_bits: {} は binary16 で無誤差に表せない", v);
    return h;
}

// 付け替え 1 回ぶん（+re）：A/B の K 方向の開始［タイル］，K ブロック数，出力の面，融合の acc
struct RunPlan { uint32_t k0 = 0, nblk = 0, plane = 0, acc = 0; };

// bad<what> ごとに，止まるべき検査の文言（common/mcast_gemm.hpp の TT_FATAL の本文の一部）．
// 例外の文字列にこれが含まれるときだけ「期待どおりに拒否した」とみなす．
struct BadWant { const char* mode; const char* want; };
const BadWant kBadWant[] = {
    {"badsh0", "形が 0 を含む"},                       // mcast_shape_check（mcast_compute_args から）
    {"badbw0", "形が 0 を含む"},
    {"badshdiv", "がコアあたりの出力"},                // 同：サブブロックが pm x pn を割り切らない
    {"badkdiv", "K ブロック幅 bw="},                   // 同：1 < bw < Kt で bw が Kt を割り切らない
    {"badkbig", "K ブロック幅 bw="},                   // 同：bw が Kt を超える
    {"baddst", "が DST の"},                           // build_mcast_gemm：サブブロックが DST に載らない
};

// 引数から決まる検査の 1 件（冒頭の使い方）
struct ExactCase {
    uint32_t M = 0, N = 0, K = 0, Mt = 0, Nt = 0, Kt = 0, forced_bw = 0, seed = 1, runs = 0, bpp = 0;
    std::string fmt, pat, mode;          // mode は接尾辞 +re・+sep を外したもの
    const FmtInfo* fi = nullptr;
    tt::DataFormat in_fmt{}, acc_fmt{};  // 入力の形式と部分積（c_24）の形式
    bool i8 = false, reprog = false, sep = false, fuse = false, l1acc = false, bad = false, pairloop = false;
    bool int_out = false;                // 出力が Int32 か（INT8 は符号-絶対値，融合・対ループは 2 の補数）
};

// 引数を読む．引数の誤りは理由を出して false（終了コード 2．デバイスを開く前）
bool parse_case(int argc, char** argv, ExactCase& c) {
    if (argc < 8 || argc > 10) {
        fprintf(stderr, "usage: %s M N K fp32|bf16|fp16|bfp8|fp8|fp8s|bfp4|bfp2|int8 "
                        "rand|pm1|pos|neg|cancel|odd|oddneg|oddcancel|maxodd|maxcancel|zface bw seed "
                        "[plain|pl<b>|fuse|l1acc|bad<what>][+re][+sep] [runs]\n", argv[0]);
        return false;
    }
    c.M = oz_env::arg_u32(argc, argv, 1, 0); c.N = oz_env::arg_u32(argc, argv, 2, 0);
    c.K = oz_env::arg_u32(argc, argv, 3, 0);
    c.fmt = argv[4]; c.pat = argv[5];
    c.forced_bw = oz_env::arg_u32(argc, argv, 6, 0); c.seed = oz_env::arg_u32(argc, argv, 7, 1);
    c.mode = argc > 8 ? argv[8] : "plain";
    if (!oz_eval::check_tile_dims(c.M, c.N, c.K, "gemm_exact_check")) return false;
    for (const auto& f : kFmts) if (c.fmt == f.name) c.fi = &f;
    if (!c.fi) { fprintf(stderr, "ERROR: fmt=%s は fp32|bf16|fp16|bfp8|fp8|fp8s|bfp4|bfp2|int8 ではない\n", c.fmt.c_str()); return false; }
    c.in_fmt = c.fi->df();
    c.i8 = c.fmt == "int8";
    // 接尾辞 +re（付け替え）と +sep（c_16 と c_24 を別々に）．'+' で区切り，先頭を構成の名前，残りを接尾辞として 1 つずつ
    // 受ける（文字列の途中から抜くと pl+reain が plain+re に化ける）．知らない接尾辞と同じ接尾辞の 2 度目は止める
    {
        const std::string whole = c.mode;
        size_t q = whole.find('+');
        c.mode = whole.substr(0, q);
        while (q != std::string::npos) {
            const size_t r = whole.find('+', q + 1);
            const std::string suf = whole.substr(q + 1, r == std::string::npos ? std::string::npos : r - q - 1);
            bool* flag = suf == "re" ? &c.reprog : suf == "sep" ? &c.sep : nullptr;
            if (!flag || *flag) {
                fprintf(stderr, "ERROR: mode=%s の接尾辞 +%s は +re・+sep のどちらでもないか，2 度目である\n", whole.c_str(), suf.c_str());
                return false;
            }
            *flag = true;
            q = r;
        }
    }
    c.fuse = c.mode == "fuse"; c.l1acc = c.mode == "l1acc";
    c.bad = c.mode.rfind("bad", 0) == 0;
    c.pairloop = c.mode != "plain" && c.mode.rfind("pl", 0) == 0;   // pl<b>（plain も pl で始まる）
    if (c.pairloop) {
        const std::string b = c.mode.substr(2);
        unsigned long long v = 0;
        if (!oz_env::to_u64(b.c_str(), v) || v < 1 || v > 65535) {
            fprintf(stderr, "ERROR: mode=%s の 1 対あたりのブロック数は 1..65535 の整数（pl3 など）\n", c.mode.c_str());
            return false;
        }
        c.bpp = (uint32_t)v;
    } else if (!c.fuse && !c.l1acc && !c.bad && c.mode != "plain") {
        fprintf(stderr, "ERROR: mode=%s は plain|pl<b>|fuse|l1acc|bad<what> ではない\n", c.mode.c_str());
        return false;
    }
    if (c.bad && c.mode != "badsh0" && c.mode != "badbw0" && c.mode != "badshdiv" && c.mode != "badkdiv" &&
        c.mode != "badkbig" && c.mode != "baddst") {
        fprintf(stderr, "ERROR: mode=%s は badsh0|badbw0|badshdiv|badkdiv|badkbig|baddst ではない\n", c.mode.c_str());
        return false;
    }
    // 融合・対ループは FP32 の部分積でしか組めない（第 2 パスが typecast<Float32,Int32> する．
    // spill_gemm_spec の TT_FATAL）．尾崎 I の INT8 経路もこの 2 つを使わない．
    if (c.i8 && (c.fuse || c.pairloop)) {
        fprintf(stderr, "ERROR: int8 では融合・対ループを組めない（mode=%s．spill_gemm_spec が止める構成）\n",
                c.mode.c_str());
        return false;
    }
    if (c.l1acc && (c.in_fmt != tt::DataFormat::Float16_b && c.in_fmt != tt::DataFormat::Float32)) {
        fprintf(stderr, "ERROR: l1acc は bf16・fp32 だけ（gemm_bf16ext の成分の形式）\n"); return false; }
    if (c.l1acc && (c.reprog || c.sep)) {
        fprintf(stderr, "ERROR: l1acc には +re・+sep を付けられない（K ブロック数がコンパイル時引数．c_16・c_24 は元から別）\n");
        return false;
    }
    if (c.bad && (c.reprog || c.sep)) { fprintf(stderr, "ERROR: bad<what> には接尾辞を付けない\n"); return false; }
    c.runs = oz_env::arg_u32(argc, argv, 9, c.reprog ? 6u : c.fuse ? 4u : 2u, 1u, 64u, "runs");
    {   // 入力の型を先に確かめる（デバイスを開く前）
        std::vector<int> a, b;
        if (!make_inputs(c.pat, *c.fi, 32, 32, 32, 1, 1, 0, false, a, b)) {
            fprintf(stderr, "ERROR: pattern=%s は知らない入力\n", c.pat.c_str()); return false; }
    }
    // 部分積（c_24）の形式．融合・対ループの出力 c_16 は Int32（2 の補数）になる
    c.acc_fmt = c.i8 ? tt::DataFormat::Int32 : tt::DataFormat::Float32;
    c.int_out = c.i8 || c.fuse || c.pairloop;
    c.Mt = c.M / 32; c.Nt = c.N / 32; c.Kt = c.K / 32;
    return true;
}

// bad<what>：構成をわざと壊して，組み立て（spill_gemm_spec・build_mcast_gemm）が止めるかを見る．
// 戻り値は終了コード（2 は止まるべき検査の文言で止まった・壊せない形で引数の誤り，1 は別の検査で止まったか止まらなかった）
template <class MakeSpec>
int run_refusal_case(std::shared_ptr<distributed::MeshDevice>& md, const ExactCase& c, OptConfig g, MakeSpec&& make_spec,
                     uint32_t in_tile, uint32_t out_tile) {
    const char* want = nullptr;
    for (const auto& b : kBadWant) if (c.mode == b.mode) want = b.want;
    TT_FATAL(want != nullptr, "gemm_exact_check: {} の期待する拒否の文言が無い", c.mode);
    const uint32_t Kt = c.Kt;
    if (c.mode == "badsh0") g.sh = 0;
    else if (c.mode == "badbw0") g.bw = 0;
    else if (c.mode == "badshdiv") g.sh = g.pm + 1;
    else if (c.mode == "badkdiv") {
        // 1 < bw < Kt で Kt を割り切らない最小の bw（Kt=32 なら 3）．mcast_shape_check の Kt >= bw には掛からず，
        // Kt % bw == 0 だけが止める形にする（bw > Kt の形は Kt % bw = Kt != 0 なので両方の条件に掛かり，
        // 割り切れの検査が抜けても前半の検査で止まってしまう）
        uint32_t bw = 2;
        while (bw < Kt && Kt % bw == 0) ++bw;
        if (bw >= Kt) {
            fprintf(stderr, "ERROR: badkdiv には 1 < bw < Kt で Kt を割り切らない bw が要る（Kt=%u．K を 96 以上に"
                            "取ること）\n", Kt);
            return 2;
        }
        g.bw = bw;
    }
    else if (c.mode == "badkbig") g.bw = Kt + 1;
    else if (c.mode == "baddst") {
        // サブブロックをコアあたりの出力全体にする（pm|pm・pn|pn なので形の検査は通り，DST の検査だけに掛かる）．
        // DST の枚数は build_mcast_gemm と同じもの（mcast_gemm.hpp の mcast_dst_tiles）．
        // pm*pn がそれ以下の形では DST に載らない構成を作れないので，検査が成り立たない（引数の誤り）
        const uint32_t dst_cap = mcast_dst_tiles(make_spec(g).ccfg);
        if (g.pm * g.pn <= dst_cap) {
            fprintf(stderr, "ERROR: baddst にはコアあたりの出力 %ux%u が DST の %u 枚を超える形が要る"
                            "（M・N を大きく取ること．例 1280 1408 で 4x4）\n", g.pm, g.pn, dst_cap);
            return 2;
        }
        g.sh = g.pm; g.sw = g.pn;
    }
    Program program{};
    auto Abuf = mesh_dram_buffer(md, (size_t)c.Mt * Kt, in_tile);
    auto Bbuf = mesh_dram_buffer(md, (size_t)Kt * c.Nt, in_tile);
    auto Cbuf = mesh_dram_buffer(md, (size_t)c.Mt * c.Nt, out_tile);
    try {
        McastGemmSpec spec = make_spec(g);
        (void)build_mcast_gemm(program, md, g, c.Mt, c.Nt, Kt, Abuf, Bbuf, Cbuf, spec);
    } catch (const std::exception& ex) {
        const bool expected = std::strstr(ex.what(), want) != nullptr;
        if (expected)
            printf("REFUSED mode=%s pm=%u pn=%u sh=%u sw=%u bw=%u: %s\n", c.mode.c_str(), g.pm, g.pn, g.sh, g.sw,
                   g.bw, ex.what());
        else
            printf("EXACT-CHECK mode=%s REFUSED-ELSEWHERE（「%s」で止まるはず） FAIL: %s\n", c.mode.c_str(), want,
                   ex.what());
        return expected ? 2 : 1;
    }
    printf("EXACT-CHECK mode=%s NOT REFUSED FAIL\n", c.mode.c_str());
    return 1;
}

// 付け替えの計画（+re でなければ K 全体・面 0）．+re は K ブロック数を 1 対（または 1 ブロック）→
// 全部 → 末尾の 1 対 → 中ほどの半分 → 2 対目 の順に替え，出力の面も替える．融合は同じ面へ acc=0,1．
// unit は K ブロック数の刻み（対ループは 1 対），nb は K ブロック数，bw は K ブロック幅［タイル］
std::vector<RunPlan> make_run_plan(const ExactCase& c, uint32_t nb, uint32_t unit, uint32_t bw, uint32_t nplane) {
    std::vector<RunPlan> plan(c.runs);
    for (uint32_t r = 0; r < c.runs; r++) {
        RunPlan& p = plan[r];
        if (!c.reprog) { p.nblk = nb; p.acc = (c.fuse && r % 3 != 0) ? 1u : 0u; continue; }
        switch (r % 5) {
        case 0: p.nblk = unit; p.k0 = 0; break;
        case 1: p.nblk = nb; p.k0 = 0; break;
        case 2: p.nblk = unit; p.k0 = (nb - unit) * bw; break;
        case 3: p.nblk = std::max(unit, (nb / 2 / unit) * unit); p.k0 = ((nb - p.nblk) / 2 / unit) * unit * bw; break;
        default: p.nblk = unit; p.k0 = (nb >= 2 * unit) ? unit * bw : 0u; break;
        }
        if (c.fuse) { p.plane = (r / 2) % nplane; p.acc = r % 2; }
        else p.plane = r % nplane;
    }
    return plan;
}

// 整数の入力 X（R×C，行優先）を形式 in_fmt へ符号化して DRAM へ書く（尾崎のスライスと同じ符号化）
void upload_exact_operand(distributed::MeshCommandQueue& cq, tt::DataFormat in_fmt, const std::vector<int>& X, uint32_t R,
                          uint32_t C, std::shared_ptr<distributed::MeshBuffer>& buf) {
    std::vector<float> f(X.begin(), X.end());
    auto t = tilize_nfaces(f, R, C);
    const tt::stl::Span<const float> sp(t.data(), t.size());
    switch (in_fmt) {
    case tt::DataFormat::Float32: distributed::EnqueueWriteMeshBuffer(cq, buf, t, true); break;
    case tt::DataFormat::Float16_b: {
        std::vector<bfloat16> h(t.size());
        for (size_t i = 0; i < t.size(); i++) h[i] = bfloat16(t[i]);
        distributed::EnqueueWriteMeshBuffer(cq, buf, h, true); break; }
    case tt::DataFormat::Float16: {   // binary16 のビット列（尾崎の FP16 経路のスライスと同じ）
        std::vector<uint16_t> h(t.size());
        for (size_t i = 0; i < t.size(); i++) h[i] = half_bits(t[i]);
        distributed::EnqueueWriteMeshBuffer(cq, buf, h, true); break; }
    case tt::DataFormat::Fp8_e4m3: {
        std::vector<uint32_t> p(t.size() / 4);
        for (size_t i = 0; i < p.size(); i++)
            p[i] = pack_four_float8_e4m3_into_uint32(float8_e4m3(t[4 * i]), float8_e4m3(t[4 * i + 1]),
                                                     float8_e4m3(t[4 * i + 2]), float8_e4m3(t[4 * i + 3]));
        // 入力がどれも e4m3 で無誤差に表せること（表せない値を丸めて渡すと参照と比べる意味が無い）
        for (size_t i = 0; i < t.size(); i++)
            TT_FATAL((float)float8_e4m3(t[i]) == t[i], "fp8: 入力 {} は e4m3 で無誤差に表せない", t[i]);
        distributed::EnqueueWriteMeshBuffer(cq, buf, p, true); break; }
    case tt::DataFormat::Bfp8_b: {
        auto p = pack_as_bfp8_tiles(sp, false, false);
        distributed::EnqueueWriteMeshBuffer(cq, buf, p, true); break; }
    case tt::DataFormat::Bfp4_b: {
        auto p = ::pack_as_bfp4_tiles(sp, false, false);
        distributed::EnqueueWriteMeshBuffer(cq, buf, p, true); break; }
    case tt::DataFormat::Bfp2_b: {
        auto p = ::pack_as_bfp2_tiles(sp, false, false);
        distributed::EnqueueWriteMeshBuffer(cq, buf, p, true); break; }
    default: {   // Int8：符号-絶対値
        std::vector<int8_t> q(t.size());
        for (size_t i = 0; i < t.size(); i++) q[i] = to_sign_mag((int)t[i]);
        distributed::EnqueueWriteMeshBuffer(cq, buf, q, true); break; }
    }
}

// 出力の器の番兵（書かれなかった要素を見分ける．FP32 なら NaN のビット列）
constexpr uint32_t SENT = 0x7FC0DEADu;

// 食い違いの原因ごとの数（冒頭の注記）と，numeric の |got - ref| の最大
struct Tally {
    size_t numeric = 0, nonnum = 0, unwritten = 0, nelem = 0, overwritten = 0;
    int64_t maxdiff = 0, peak_out = 0;
};

// 実行 r の後に読み戻した器の全面 cur を，面ごとの期待値と比べる（まだ書いていない面は番兵のままか）．
// 食い違いは原因を問わずこの実行で 5 件まで表示する（番兵の破壊は全体で 5 件まで）
void compare_planes(const ExactCase& c, uint32_t r, uint32_t nplane, const std::vector<uint32_t>& cur,
                    const std::vector<std::vector<int64_t>>& want, const std::vector<bool>& written, Tally& t) {
    const uint32_t M = c.M, N = c.N, Nt = c.Nt, MtNt = c.Mt * c.Nt;
    size_t shown = 0;
    for (uint32_t pl = 0; pl < nplane; pl++)
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t j = 0; j < N; j++) {
                const size_t off = (size_t)pl * MtNt * 1024 + ((size_t)(i / 32) * Nt + j / 32) * 1024 + nf(i % 32, j % 32);
                const uint32_t raw = cur[off];
                if (!written[pl]) {   // まだ書いていない面は番兵のまま
                    if (raw != SENT) { if (t.overwritten < 5) printf("OVERWRITE run=%u plane %u (%u,%u) raw 0x%08x\n", r, pl, i, j, raw); ++t.overwritten; }
                    continue;
                }
                const int64_t ex = want[pl][(size_t)i * N + j];
                t.peak_out = std::max<int64_t>(t.peak_out, ex < 0 ? -ex : ex);
                ++t.nelem;
                // 書いたはずの面に番兵が残っている（書き込みの欠落．Int32 の出力でも値として読まずに分ける）
                if (raw == SENT) {
                    if (shown++ < 5) printf("UNWRITTEN run=%u plane %u (%u,%u) ref %lld（番兵のまま）\n", r, pl, i, j,
                                            (long long)ex);
                    ++t.unwritten;
                    continue;
                }
                // 復号（整数値として比べる）．FP32 は有限かつ整数で，int64 に収まるときだけ変換する
                // （NaN・Inf・範囲外の float を int64 へ変換するのは未定義動作）
                int64_t got = 0;
                bool numeric = true;
                if (c.int_out) {
                    // INT8 は符号-絶対値（融合・対ループとは組まない．parse_case の検査），融合・対ループの出力は 2 の補数
                    got = c.i8 ? dec_sm32(raw) : (int64_t)(int32_t)raw;
                } else {
                    float fv; std::memcpy(&fv, &raw, 4);
                    const double v = (double)fv;
                    numeric = std::isfinite(v) && std::fabs(v) < 9.0e15 && v == std::nearbyint(v);
                    if (numeric) got = (int64_t)v;
                }
                if (!numeric) {
                    if (shown++ < 5) printf("NONNUMERIC run=%u plane %u (%u,%u) raw 0x%08x ref %lld\n", r, pl, i, j, raw,
                                            (long long)ex);
                    ++t.nonnum;
                } else if (got != ex) {
                    if (shown++ < 5) printf("MISMATCH run=%u plane %u (%u,%u) got %lld ref %lld (raw 0x%08x)\n", r, pl, i, j,
                                            (long long)got, (long long)ex, raw);
                    const int64_t d = got - ex;
                    t.maxdiff = std::max<int64_t>(t.maxdiff, d < 0 ? -d : d);
                    ++t.numeric;
                }
            }
}

// 結果の 1 行（EXACT-CHECK ... PASS|FAIL）．戻り値は PASS か
struct RunInfo { OptConfig g; uint32_t nb = 0, refills = 0; uint64_t l1b = 0; bool blockmm = true; std::string accs, nbs; };
bool print_summary(const ExactCase& c, const RunInfo& ri, const RefStats& st, const Tally& t, bool odd_pat, size_t not_odd) {
    const OptConfig& g = ri.g;
    const size_t bad_total = t.numeric + t.nonnum + t.unwritten;
    // 無誤差でいられる上限：区間の途中の和は FP32 の部分積なら 2^24，Int32（int8）なら 2^31-1．
    // 出力は FP32 の plain なら 2^24，Int32（int8・融合・対ループ）なら 2^31-1．対ループの c_5 は 2^31-1．
    const int64_t lim_seg = c.i8 ? 2147483647ll : 16777216ll;
    const int64_t lim_out = c.int_out ? 2147483647ll : 16777216ll;
    auto rng_s = [](int64_t v, int64_t lim) { return v <= lim ? "in range" : "OUT OF RANGE"; };
    const bool odd_check = odd_pat && !c.reprog;
    if (odd_check && not_odd)
        printf("PATTERN-PROPERTY pat=%s: 参照の %zu 要素が奇数でない（入力の生成が検査の意図を満たしていない）\n",
               c.pat.c_str(), not_odd);
    const bool pass = bad_total == 0 && t.overwritten == 0 && not_odd == 0;
    // blockmm は計算カーネルの発行方式：spill（bmm_int32_spill.cpp）は 1=matmul_block，0=matmul_tiles（OZ_NO_BLOCKMM）．
    // l1acc（bmm_l1acc.cpp）は OZ_NO_BLOCKMM を読まず常に matmul_tiles なので n/a
    const std::string bm = c.l1acc ? std::string("n/a") : std::to_string(ri.blockmm ? 1 : 0);
    const std::string oddsum = !odd_pat ? std::string("n/a") : c.reprog ? std::string("skipped")
                             : not_odd ? "FAIL(" + std::to_string(not_odd) + ")" : std::string("ok");
    printf("EXACT-CHECK fmt=%s pat=%s mode=%s%s%s blockmm=%s M=%u N=%u K=%u grid=%ux%u per_core=%ux%u sub=%ux%u bw=%u "
           "nblocks=%u l1=%lluKiB seed=%u runs=%u%s%s%s%s blocktiles=%ux%u "
           "peak|seg|=%lld (%s 2^%.2f) sum|seg|=%lld (%s, any order) peak|acc|=%lld (%s) peak|out|=%lld (%s) "
           "oddsum=%s mismatches=%zu/%zu numeric=%zu maxdiff=%lld nonnumeric=%zu unwritten=%zu overwritten=%zu "
           "refills=%u %s\n",
           c.fmt.c_str(), c.pat.c_str(), c.mode.c_str(), c.reprog ? "+re" : "", c.sep ? "+sep" : "", bm.c_str(),
           c.M, c.N, c.K, g.cr, g.cc, g.pm, g.pn, g.sh, g.sw, g.bw, ri.nb, (unsigned long long)(ri.l1b >> 10), c.seed, c.runs,
           c.fuse ? " acc=" : "", c.fuse ? ri.accs.c_str() : "", c.reprog ? " plan=" : "", c.reprog ? ri.nbs.c_str() : "",
           g.pm * g.bw, g.pn * g.bw,
           (long long)st.peak_seg, rng_s(st.peak_seg, lim_seg), std::log2((double)std::max<int64_t>(st.peak_seg, 1)),
           (long long)st.sum_seg, rng_s(st.sum_seg, lim_seg),
           (long long)st.peak_acc, c.pairloop ? rng_s(st.peak_acc, 2147483647ll) : "n/a",
           (long long)t.peak_out, rng_s(t.peak_out, lim_out), oddsum.c_str(), bad_total, t.nelem, t.numeric,
           (long long)t.maxdiff, t.nonnum, t.unwritten, t.overwritten, ri.refills, pass ? "PASS" : "FAIL");
    return pass;
}

// デバイスを開いた後の検査の本体．戻り値は終了コード（0 PASS，1 FAIL，2 引数の誤りか bad<what> の期待どおりの拒否）
int run_exact_case(std::shared_ptr<distributed::MeshDevice>& md, const ExactCase& c) {
    const uint32_t Mt = c.Mt, Nt = c.Nt, Kt = c.Kt, M = c.M, N = c.N, K = c.K;
    const uint32_t in_tile = tt::tile_size(c.in_fmt), out_tile = tt::tile_size(c.acc_fmt);
    // 本番と同じ仕様（spill_gemm_spec．l1acc は l1acc_gemm_spec）．fidelity は本番の規則そのもの（oz_format_table.hpp の
    // oz_fidelity_rule_for に形式と桁の上限 d を渡す：桁 <= 32 の浮動小数の形式は LoFi，INT8 は HiFi4）．l1acc は
    // gemm_bf16ext の既定の HiFi4（oz_fidelity.hpp の baseline_fidelity の既定．この検査は OZ_FIDELITY を読まない）
    SpillOpts so;
    so.in_fmt = c.in_fmt;
    so.acc_fmt = c.acc_fmt;
    so.fidelity = c.l1acc ? tt::tt_metal::MathFidelity::HiFi4 : oz_fidelity_rule_for(c.fi->fmt, (uint32_t)c.fi->d);
    so.fuse = c.fuse;
    so.pairloop = c.pairloop;
    so.blockmm = gemm_blockmm();
    so.out_shared = !c.sep;
    auto make_spec = [&](const OptConfig& g) {
        return c.l1acc ? l1acc_gemm_spec(g, Kt, c.in_fmt, so.fidelity) : spill_gemm_spec(g, Kt, so);
    };
    auto gd = md->compute_with_storage_grid_size();
    // 構成は実際に作る CB の合計（mcast_gemm_l1_bytes．融合・対ループ・l1acc の CB を含む）が予算に
    // 収まる候補から選ぶ．予算から固定の見込みを引く方式だと，対ループの c_5（pm*pn+サブブロック）を
    // 小さく見積もるうえ，予算が見込みより小さいと符号なしの引き算が下へ溢れる．
    const uint64_t budget = oz_l1_budget();
    auto accept = [&](const OptConfig& g) { return mcast_gemm_l1_bytes(g, make_spec(g)) <= budget; };
    OptConfig g = choose_cfg(Mt, Nt, Kt, gd.y, gd.x, 4, c.forced_bw, in_tile, out_tile, (uint32_t)budget, accept);
    if (!g.ok) { fprintf(stderr, "ERROR: 構成が取れない（Mt=%u Nt=%u Kt=%u bw=%u 予算 %llu KiB）\n", Mt, Nt, Kt,
                         c.forced_bw, (unsigned long long)(budget >> 10));
                 return 2; }
    if (c.bad) return run_refusal_case(md, c, g, make_spec, in_tile, out_tile);
    const uint32_t nb = Kt / g.bw;
    if (c.pairloop && nb % c.bpp != 0) {
        fprintf(stderr, "ERROR: 1 対あたり %u ブロックが K ブロック数 %u を割り切らない\n", c.bpp, nb);
        return 2;
    }
    const uint32_t unit = c.pairloop ? c.bpp : 1u;           // +re で動かす K ブロック数の刻み（対ループは 1 対）
    const uint32_t seg = c.pairloop ? c.bpp * g.bw * 32u : K; // 1 区間の長さ［要素］
    const uint32_t nplane = c.reprog ? 3u : 1u;               // 出力の面の数（+re は付け替え先を 3 面）
    const uint32_t MtNt = Mt * Nt;

    auto Abuf = mesh_dram_buffer(md, (size_t)Mt * Kt, in_tile);
    auto Bbuf = mesh_dram_buffer(md, (size_t)Kt * Nt, in_tile);
    auto Cbuf = mesh_dram_buffer(md, (size_t)nplane * MtNt, out_tile);
    auto& cq = md->mesh_command_queue();

    Program program{};
    McastGemmSpec spec = make_spec(g);   // c_16 と c_24 は既定で L1 を共有（尾崎の両スキームと同じ）
    RunInfo ri;
    ri.g = g; ri.nb = nb; ri.blockmm = so.blockmm;
    ri.l1b = mcast_gemm_l1_bytes(g, spec);
    if (!c.l1acc) {
        spill_check_run(nb, c.pairloop, c.bpp);
        spec.compute_rt = spill_compute_rt(nb, 0u, c.bpp);
    }
    McastGemm mg = build_mcast_gemm(program, md, g, Mt, Nt, Kt, Abuf, Bbuf, Cbuf, spec);
    distributed::MeshWorkload wl;
    const auto range = distributed::MeshCoordinateRange(md->shape());
    wl.add_program(range, std::move(program));
    Program& prog = wl.get_programs().at(range);

    const std::vector<RunPlan> plan = make_run_plan(c, nb, unit, g.bw, nplane);
    // cur は出力の器の中身の写しで，最初は全面が番兵，以降は各実行の後に読み戻したもの．acc=0 の実行の前に，その実行が
    // 書く面だけを番兵に戻して器へ書く（冒頭の注記．ほかの面は読み戻した値のまま書き戻すので変わらない）
    const size_t plane_words = (size_t)MtNt * out_tile / 4;
    std::vector<uint32_t> cur((size_t)nplane * plane_words, SENT);
    // 面ごとの期待値（まだ書いていない面は番兵のまま）
    std::vector<std::vector<int64_t>> want(nplane);
    std::vector<bool> written(nplane, false);
    std::vector<int64_t> Cref;
    Tally t;
    // 奇数の和を狙う入力（odd 系・maxodd・maxcancel）は，参照の全要素が奇数であることを確かめる（+re は K の範囲が
    // 変わって性質が崩れるので見ない）．外れた要素の数を数えて FAIL にする
    const bool odd_pat = c.pat == "odd" || c.pat == "oddneg" || c.pat == "oddcancel" || c.pat == "maxodd" ||
                         c.pat == "maxcancel";
    const bool odd_check = odd_pat && !c.reprog;
    size_t not_odd = 0;
    RefStats st;
    for (uint32_t r = 0; r < c.runs; r++) {
        const RunPlan& p = plan[r];
        std::vector<int> A, B;
        make_inputs(c.pat, *c.fi, M, N, K, g.bw, c.seed, r, !c.fuse && (r & 1u), A, B);
        oz_eval::int_reference_gemm(A, B, M, K, N, p.k0 * 32u, (p.k0 + p.nblk * g.bw) * 32u, seg, Cref, &st);
        if (odd_check)
            for (int64_t v : Cref) not_odd += (v % 2 == 0) ? 1u : 0u;
        upload_exact_operand(cq, c.in_fmt, A, M, K, Abuf);
        upload_exact_operand(cq, c.in_fmt, B, K, N, Bbuf);
        ri.accs += std::to_string(p.acc);
        ri.nbs += (ri.nbs.empty() ? "" : ",") + std::to_string(p.nblk) + "@" + std::to_string(p.k0) + ">" + std::to_string(p.plane);
        auto& w = want[p.plane];
        if (!written[p.plane] || !p.acc) w.assign((size_t)M * N, 0);
        TT_FATAL(!p.acc || written[p.plane], "gemm_exact_check: 書いていない面へ acc=1 で足そうとした");
        for (size_t i = 0; i < w.size(); i++) w[i] += Cref[i];
        written[p.plane] = true;
        // 付け替え（尾崎 I の gemm_run と同じ：A・B の開始位置，K ブロック数，出力の開始，acc，bpp）
        if (c.reprog || c.fuse) {
            spill_check_run(p.nblk, c.pairloop, c.bpp);   // 計算カーネルの実行時引数（どのコアも同じ値）の検査
            for (const auto& e : mg.cores) {
                const uint32_t cx = (uint32_t)e.core.x, cy = (uint32_t)e.core.y;
                mg.retarget_k(prog, e, p.k0, p.k0 * Nt, p.nblk,
                              mcast_writer_args((uint32_t)Cbuf->address(), Nt, g.pm, g.pn, g.sh, g.sw, cx, cy,
                                                p.plane * MtNt, p.acc));
                SetRuntimeArgs(prog, mg.ck, e.core, spill_compute_rt(p.nblk, p.acc, c.bpp));
            }
        }
        // acc=0 の実行が書く面を番兵に戻す（acc=1 は直前の出力を読んで足すので触らない）
        if (!p.acc) {
            std::fill(cur.begin() + (size_t)p.plane * plane_words, cur.begin() + (size_t)(p.plane + 1) * plane_words, SENT);
            distributed::EnqueueWriteMeshBuffer(cq, Cbuf, cur, true);
            ++ri.refills;
        }
        distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);
        distributed::EnqueueReadMeshBuffer(cq, cur, Cbuf, true);
        compare_planes(c, r, nplane, cur, want, written, t);
    }
    return print_summary(c, ri, st, t, odd_pat, not_odd) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    // 入口の並びは尾崎の 2 つのドライバ・基準実装と同じ（common/oz_driver.hpp の冒頭）．
    // 1. 環境だけ．入力は自前の整数の組（pattern）で，判定は int64 の参照との完全一致なので，入力の生成（OZ_PHI など）・
    //    精度の基準（OZ_MIN_BITS）・結果の書き出し（OZ_DUMP_C）のつまみは効かない．黙って無視せずデバイスを開く前に止める
    oz_drv::reject_env_gemm(oz_knobs::W_EXACT, "gemm_exact_check", "入力の作り方・精度の基準・結果の書き出しを持たない");
    if (const char* v = oz_env::get("OZ_NOVERIFY"))   // 検査そのものが目的なので省かない（安全側なので止めずに知らせる）
        fprintf(stderr, "注意：OZ_NOVERIFY=%s は gemm_exact_check では効かない（参照との比較は常に行う）\n", v);
    // 2. 位置引数
    ExactCase c;
    if (!parse_case(argc, argv, c)) return 2;
    // 3. つまみの値（OZ_L1_BUDGET_KB など）・GEMM 段（入力の生成・判定・書き出しと計時は読まない）
    oz_drv::check_env(oz_knobs::W_EXACT, c.K, true);
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    int rc = 0;
    try {
        rc = run_exact_case(md, c);
    } catch (const std::exception& ex) {
        // 想定外の例外（組み立て・転送・実行の失敗）は 3．引数の誤りと期待どおりの拒否（2）と取り違えないように分ける
        fprintf(stderr, "ERROR: %s\n", ex.what());
        md->close();
        return 3;
    }
    const bool closed = md->close();
    return (rc || !closed) ? (rc ? rc : 1) : 0;
}
