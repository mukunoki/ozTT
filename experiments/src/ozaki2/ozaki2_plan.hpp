// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎スキーム II：構成の計画（法と幅・再構成のパス数・合成段の平面・法の組分け・GEMM のパネル・再構成のパス）
// ---------------------------------------------------------------------------
// scheme I の ozaki1_plan.hpp に当たる．どれもデバイスに触らず，何も確保しない（DRAM の空き容量とコア格子の大きさだけを
// 引数で受ける）．計画は引数（Oz2Args・Oz2Knobs と入力のオペランド）だけから決まり，環境変数を読まない（つまみは main の
// 冒頭で oz2_diag_from_env・oz2_knobs_from_env が 1 回だけ読む）．確保量もカーネルの引数もこの計画から導く：決める処理と
// 確保する処理を分けておけば，「確保したあとで枚数の意味が変わる」種類のずれが起こらない．
// 依存：ozaki2_format.hpp（OZ2_PARAM・OZ2_DIGIT），ozaki2_autop.hpp（OZ2_FP64 のモデル），common/mcast_gemm.hpp
// （OptConfig・choose_cfg・DramFree・largest_div_le．ozaki_gemm_host.hpp を取り込む），common/gemm_probe.hpp（GemmProbe），
// ozaki2/kernels/recon_args.h（ReconRole），common/kernels/f64_tf.h（FP64・128 bit 固定小数のタイルの大きさ）．
// ヘッダの中では名前空間を開かない（tt-metal の名前は tt:: で修飾する）．
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#include "ozaki2/ozaki2_format.hpp"
#include "ozaki2/ozaki2_autop.hpp"
#include "common/mcast_gemm.hpp"
#include "common/gemm_probe.hpp"
#include "common/oz_knobs.hpp"
#include "common/oz_env.hpp"
#include "common/kernels/f64_tf.h"
#include "ozaki2/kernels/recon_args.h"

// ===== 位置引数とつまみ =====

// 位置引数（ozaki2_main.cpp の read_run_args が読む．読み方は scheme I と共通の oz_driver.hpp）
struct Oz2Args {
    uint32_t M = 1024, N = 1024, K = 1024;
    uint32_t s_req = OZ2_PARAM.def_s;   // 法の本数（GEMMul8 の num_moduli）
    uint32_t warmup = 1, iters = 3;
};

// OZ2_PASSES の要求（つまみの表の列挙 1|2|auto の順．未設定は Unset）
enum class Oz2PassReq { Unset, One, Two, Auto };

// 計画と段の組み方を変えるつまみ（docs/env_vars.md の A の OZ2_FP64・OZ2_NOWIN・OZ2_PASSES，B の OZ_NO_BLOCKMM・
// OZ2_MOD_GROUPS・OZ_L1_BUDGET_KB・OZ2_PANEL_FORCE，計画を変える C の OZ2_B・OZ2_MODULI・OZ2_JSPLIT・OZ2_NSMALL 系・
// OZ_FIDELITY）．scheme I の Oz1Knobs に当たる．main の冒頭で oz2_knobs_from_env が 1 回だけ読み，計画（choose_moduli_and_width・
// plan_mod_groups・plan_panels）と段の構築（ozaki2_stages.hpp）へ引数で配る（使う所で読み直さない）．
// どのつまみを読むか・範囲・読み方（有無か値か）・CONFIG 行への記録（knobs・diag）はつまみの表（common/oz_knobs.hpp）が
// 決め，構文と範囲はデバイスを開く前に oz_drv::check_env（表の preflight）が検査してある．範囲が法の本数や経路の断片数で
// 決まる OZ2_MOD_GROUPS・OZ2_NSMALL の上界だけは，法を選んだ後に main が見る．
struct Oz2Knobs {
    bool auto_b = false;                  // OZ2_FP64：幅 b を入力から決める（ozaki2_autop.hpp）
    bool nowin = false;                   // OZ2_NOWIN：OZ2_FP64 でも窓を入力から見積もらない
    Oz2PassReq passes = Oz2PassReq::Unset;   // OZ2_PASSES
    std::optional<uint32_t> b;            // OZ2_B：幅 b の上限を明示する（調査用）
    // 法の集合を外から与える（OZ2_MODULI．答えも精度も変わる．集合の解釈は oz2_parse_forced_moduli．立っていなければ nullptr）
    const char* moduli_set = nullptr;
    std::optional<uint32_t> groups;       // OZ2_MOD_GROUPS：法の組数を強制する（上限は選んだ法の本数）
    std::optional<uint32_t> bands;        // OZ_BANDS：出力を切る行の帯の本数を強制する（パネル行を等分できる本数．plan_bands）
    const char* panel_force = nullptr;    // OZ2_PANEL_FORCE=RxC［タイル］（書式は oz2_parse_panel_force．立っていなければ nullptr）
    uint32_t l1_budget = 0;               // OZ_L1_BUDGET_KB［B］（形状探索の L1 の予算．oz_l1_budget）
    bool blockmm = true;                  // OZ_NO_BLOCKMM の否定（GEMM の matmul_block．読み方は GEMM 段を組むほかのドライバと同じ）
    // FP64 組み立ての分担の境目（OZ2_JSPLIT 0--1024．0 か 1024 で分担しない．答えは変えないが測る構成を変える）
    uint32_t jsplit = 512;
    // tfAddSmall へ回す断片の本数．-1 なら自動（ホストが断片の大きさから数える），-2 なら全断片（OZ2_ALLSMALL），0 は
    // OZ2_NOSMALL．それ以外は OZ2_NSMALL の指定値で，本数は経路ごとに違うので上限の検査は使う側で行う
    // （OZ2_NSMALL=-1 を unsigned で読むと巨大値になり NRP-NSMALL が折り返すので，範囲外は表が止める）
    int nsmall = -1;
    tt::tt_metal::MathFidelity fid = tt::tt_metal::MathFidelity::LoFi;   // GEMM の MathFidelity（oz2_fidelity．OZ_FIDELITY が上書き）
};

// 答えを壊すか，段の間で読み戻す診断のつまみ（docs/env_vars.md の C の探針，D2 の書き出し）と，全体の時間の計時 OZ_TIME_WHOLE
// （A．2026-11-03 から論文の全体の時間．診断ではない）．
// 探針はどれも答えを壊すか，壊さないまでも通常と違う構成で測る．測ったログが通常の測定か調査の測定かを後から
// 見分けられないと，調査で測った値が論文の表へ入りうる（実際にこの実装では探針を立てた測定を何十回も行っている）．
// そこで解釈は入口で 1 度だけ行い，値は CONFIG 行へそのまま残す（probe=set は表の diag 欄が空でないとき．
// 検査側の scripts/verify/check_paper_numbers.py は probe が既定でないログを弾く）．0/1 のつまみは値で読む（0 なら立てていないのと同じ）．
struct Oz2Diag {
    uint32_t crt = 0;           // OZ2_CRT_PROBE   0--2（recon_crt.cpp の CPROBE）
    uint32_t recon = 0;         // OZ_RECON_PROBE  0--1（scheme II は 1 しか無い）
    bool nof64 = false;         // OZ2_RECON_NOF64：最終パスに FP64 を組み立てさせず limb 3 面を書かせる（出口 o64 を書かない）
    // 答えは変えないが，段の間で大量の読み戻しとホストの計算を挟む診断（計時の外だが，次の段の温度や実行の
    // 間隔を変える）．OZ2_CHECK_CONCAT は連結した GEMM の出力を整数の参照と比べる（桁分割経路だけ．INT8 経路では
    // main が止める），OZ2_MAXCI は GEMM の出力の最大値を読み戻して表示する（検証が有効なときだけ）
    bool check_concat = false;
    bool maxci = false;
    // OZ_TIME_WHOLE：段ごとの計時の後で全工程を続けて流す計時を足す（main の run_whole）．論文の全体の時間（表の Total）は
    // この通しの時間である（2026-11-03 から．段ごとの計時はフェーズの内訳に残る）．流し直しのために解放を遅らせるので
    // DRAM の割り付けが変わり，DRAM が足りなければ plan_bands が出力を行の帯に分ける（性能のログはどれもこれを立てる）
    bool whole = false;
    // OZ_DUMP_C（調査用の書き出し）が空でない値で立っているか（空の扱いは書き出す oz_eval::dump_result と同じ）．書き出しに
    // ホストの A・B を使うので，upload_operands が手放さずに残す
    bool dump = false;
    GemmProbe gemm_probe;       // OZ_GEMM_PROBE_*（GEMM 段の探針．立っていると CONFIG 行が probe=set になる）
};

// 範囲外・不正文字列・競合はここで止める．黙って別の意味へ倒さない（範囲はつまみの表）．main は oz2_knobs_from_env より
// 先にこちらを呼ぶ（2 つの止め方の順が入口の検査の順）．
static Oz2Diag oz2_diag_from_env() {
    using namespace oz_knobs;
    Oz2Diag d;
    d.crt = u32(OZK("OZ2_CRT_PROBE"));
    // scheme II のライタは段階を持たず，1 で組み立てを外すだけである．
    // scheme I の 2・3（表の範囲は I に合わせて 0..3）をここで黙って受け取ると 1 と同じ動作になってしまう．
    d.recon = u32(OZK("OZ_RECON_PROBE"));
    if (d.recon > 1) oz_env::fail("OZ_RECON_PROBE=" + std::to_string(d.recon) + " はスキーム II では 0 か 1（段階を持たない）");
    d.nof64 = u32(OZK("OZ2_RECON_NOF64")) != 0;
    d.check_concat = u32(OZK("OZ2_CHECK_CONCAT")) != 0;
    d.maxci = u32(OZK("OZ2_MAXCI")) != 0;
    d.whole = u32(OZK("OZ_TIME_WHOLE")) != 0;
    const char* dump = get(OZK("OZ_DUMP_C"));
    d.dump = dump && *dump;
    d.gemm_probe = oz_gemm_probe();
    return d;
}

static Oz2Knobs oz2_knobs_from_env() {
    using namespace oz_knobs;
    Oz2Knobs k;
    k.auto_b = set(OZK("OZ2_FP64"));
    k.nowin = set(OZK("OZ2_NOWIN"));
    {   // 列挙の順（つまみの表の 1|2|auto）
        static const Oz2PassReq kByIndex[] = {Oz2PassReq::One, Oz2PassReq::Two, Oz2PassReq::Auto};
        const int i = choice(OZK("OZ2_PASSES"));
        k.passes = i < 0 ? Oz2PassReq::Unset : kByIndex[i];
    }
    // OZ2_B の範囲 1..100 はつまみの表．ozaki2_moduli.h が自分で getenv すると，同じ引数で呼んでも環境で答えが変わり，
    // 法選択をホスト単体で検査できない．そこで読むのはここ 1 か所にして，値は b_cap に載せて渡す（choose_moduli_and_width）
    static_assert(TABLE[OZK("OZ2_B")].hi == OZ2_MAXDIGIT * OZ2_DIGIT_BITS,
                  "OZ2_B の上限（つまみの表）は A' を運べる幅 OZ2_MAXDIGIT*OZ2_DIGIT_BITS");
    k.b = u32_opt(OZK("OZ2_B"));
    k.moduli_set = get(OZK("OZ2_MODULI"));
    k.groups = u32_opt(OZK("OZ2_MOD_GROUPS"));
    k.bands = u32_opt(OZK("OZ_BANDS"));
    k.panel_force = get(OZK("OZ2_PANEL_FORCE"));
    k.l1_budget = oz_l1_budget();
    k.blockmm = gemm_blockmm();
    k.jsplit = u32(OZK("OZ2_JSPLIT"));
    const bool no = set(OZK("OZ2_NOSMALL")), all = set(OZK("OZ2_ALLSMALL")), n = set(OZK("OZ2_NSMALL"));
    if ((int)no + (int)all + (int)n > 1) oz_env::fail("OZ2_NOSMALL / OZ2_ALLSMALL / OZ2_NSMALL は同時に立てられない");
    if (no) k.nsmall = 0;
    else if (all) k.nsmall = -2;
    else if (n) k.nsmall = (int)*u32_opt(OZK("OZ2_NSMALL"));
    k.fid = oz2_fidelity();
    return k;
}

// ===== 計画の結果 =====
// main の実行の状態 Oz2Run（ozaki2_main.cpp）はこれを受け継ぎ，デバイスの資源と計時を足す．
struct Oz2Plan {
    // 寸法と計時回数（コマンドライン）
    uint32_t M = 0, N = 0, K = 0, warmup = 0, iters = 0;
    uint32_t S_req = 0;               // 法の本数の要求（自動決定では表の全長に置き換わる）
    // 法と幅（choose_moduli_and_width）
    bool two_pass = false;            // 2 パス再構成（OZ2_PASSES．choose_moduli_and_width の最後で決める）
    bool auto_b = false;              // OZ2_FP64：b の自動決定
    int auto_b_cap = 0;               // OZ2_FP64 の見積もりが決めた b の上限（自動決定でなければ 0．CONFIG の auto_b）
    bool win_meas = false;            // CRT の窓を入力から見積もったか（OZ2_WIN の行．CONFIG の win_meas）
    std::string pass_mode = "fixed";  // パス数の決め方：fixed（既定の 1 パス）・auto（見積りで選ぶ）・forced（OZ2_PASSES=1|2）
    double recon1_log2 = NAN;         // 1 パスの復元誤差の見込み（log2．入力の統計を測らない構成では NaN）
    Ozaki2Moduli mo;
    uint32_t b = 0, ND = 3, S = 0;    // 幅，A' を運ぶ 2^20 進の桁数，法の本数
    uint32_t Mt = 0, Nt = 0, Kt = 0, ntotA = 0, ntotB = 0, ntotC = 0;
    // 合成段の平面と切り方（make_compose_plan．法の集合から決まる．INT8 経路は全部 0）
    Oz2ComposePlan cplan;
    // 法の組分け（plan_mod_groups）．組 g は法 [gbeg[g], gbeg[g+1]) で，1 組なら全法を一度にスライスする
    uint32_t groups_req = 0;          // OZ2_MOD_GROUPS（0 = 自動）
    uint32_t ngroups = 1;
    std::vector<uint32_t> gbeg;
    bool whole_fits = true;           // OZ_TIME_WHOLE の同時確保が DRAM に収まる見積りか（plan_mod_groups）
    // GEMM 段（plan_panels）
    OptConfig g;
    uint32_t pMt = 1, pNt = 1, nbr = 1, nbc = 1;   // パネルのタイル寸法と枚数
    // 出力の行の帯（plan_bands）．帯 1 本はパネル行 bpr 行（bMt = bpr*pMt タイル行）で，nband = nbr / bpr 本（等分する）．
    // A の桁行列・積の器（桁積の平面と剰余）・再構成の器は帯 1 本ぶんだけ持ち，帯ごとに「A のその行のスライス → 全法の
    // GEMM・合成 → その帯の再構成」を回す．B の桁行列は全法ぶんを 1 回だけ作る．1 本なら帯を切らない従来の流れと同じ
    uint32_t nband = 1, bpr = 1;
    uint32_t bMt() const { return bpr * pMt; }           // 帯 1 本のタイル行
    uint32_t ntotAb() const { return bMt() * Kt; }       // 帯 1 本ぶんの A の桁行列 1 枚のタイル数
    uint32_t ntotCb() const { return bMt() * Nt; }       // 帯 1 本ぶんの出力の平面 1 枚のタイル数
};

// スライスの初手の位（ログ 2）：b - (ND-1)*20．スライサは 2^(b-(ND-1)*20-e) を掛けて A' の最下位の桁の位まで落とし
// （指数差分 expdelta_build），再構成の出口は 2 つぶん（A と B）の 2*oz2_shift を掛け戻す（run_recon_stage）
static int32_t oz2_shift(const Oz2Plan& r) { return (int32_t)r.b - (int32_t)(r.ND - 1) * (int32_t)OZ2_DIGIT_BITS; }

// ===== 合成段の平面（桁分割の 7 経路）=====
// 桁積を何枚置き，何本ずつ合成するかを決める．確保はこの結果から導く．
// 数えるのは GEMM の出力の本数（同じ位の桁積を K 方向に連結してまとめた後．oz2_product_groups）．
// 1 桁の法（OZ2_SPLIT_ONE）は合成段を通らず GEMM の出力を剰余の平面へ直接書くので数えない．合成する法が
// 1 本も無ければ桁積の平面も合成段も要らない（全部 0 で，run_digit_gemms は桁積の平面も合成段も作らない）．
// 下限と切り方の規則は，段の時間のモデルの DRAM の見積りと共有する ozaki2_split.h の oz2_compose_plan
// （OZ2_COMPOSE_MIN・OZ2_COMPOSE_NOSPLIT_MAX・OZ2_COMPOSE_CHUNK）．INT8 経路は合成段が無いので全部 0 のまま
static Oz2ComposePlan make_compose_plan(const Ozaki2Moduli& mo) {
    if constexpr (!OZ2_DIGIT) return {};
    uint32_t mx = 0; bool any = false;
    for (int i = 0; i < mo.s; i++) {
        if (i < (int)mo.mode.size() && mo.mode[i] == OZ2_SPLIT_ONE) continue;
        any = true;
        mx = std::max(mx, oz2_nout(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[i]));
    }
    return oz2_compose_plan(any, mx);
}

// 合成段の c_0 の深さの下限と，合成段を 1 つだけ組む深さの上限．2 桁経路は 3 枚・4 枚の組が来るので最低 12 枚要る．
// 公倍数が上限（64 枚．FP32 で 256 KB）を超える構成（BFP2_b の平衡 3・4・5 桁が混ざると 5・7・9 本で公倍数 315）は，1 回の
// 投入で流れる本数ごとに合成段を組む（ozaki2_main.cpp の compose_for）
constexpr uint32_t OZ2_COMPOSE_CB_MIN = 12;
constexpr uint32_t OZ2_COMPOSE_CB_MAX = 64;
// 合成段の c_0 の深さ．「1 回の投入で流れるタイル数」すべての公倍数でなければならない
// （割り切れないとリングの境界をまたいだ組が壊れる．ozaki2_stages.hpp の build_compose の注記）．
// 流れる長さは法ごと・チャンクごとに違うので，実際に発行する長さを数え上げて取る：
//   切らないとき      法ごとの桁積の本数 np
//   切るとき          通常のチャンク CH，末尾のチャンク np%CH（0 でなければ），
//                     および部分和を畳むパスの塊数 ceil(np/CH)
static inline uint32_t compose_cb_depth(const Ozaki2Moduli& mo, uint32_t CH) {
    uint32_t np_lcm = 1;
    auto take_lcm = [&np_lcm](uint32_t v) { if (v) np_lcm = (uint32_t)std::lcm(np_lcm, v); };
    for (int i = 0; i < mo.s; i++) {
        const uint32_t v = oz2_nout(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[i]);
        if (CH < v) { take_lcm(CH); take_lcm(v % CH); take_lcm((v + CH - 1) / CH); }
        else        take_lcm(v);
    }
    // 深さを下限へ上げるときも公倍数であることを崩さないよう，np_lcm の倍数のまま引き上げる．
    if (np_lcm < OZ2_COMPOSE_CB_MIN) np_lcm *= (OZ2_COMPOSE_CB_MIN + np_lcm - 1) / np_lcm;
    return np_lcm;
}

// ===== 法と幅（ホストだけで完結し，デバイスは開かない）=====

// 調査用：法の集合をカンマ区切りで与える（OZ2_MODULI．Oz2Knobs::moduli_set を解釈する）．与えた順のまま使い，
// 落とす・選び直すをしない（使えない法・互いに素でない組は法の選択の !mo.ok で止まる）．立っていなければ空のまま．
// 空文字列・空の要素・末尾の区切りも不正として止める（止めないと空文字列が「指定なし」＝自動選択に，
// "4096," が {4096} になる）．各要素は符号なしの 10 進整数だけを受け付ける．不正なら理由を出して false
static inline bool oz2_parse_forced_moduli(const char* fm, std::vector<int>& force) {
    if (!fm) return true;
    const char* q = fm;
    for (;;) {
        char* end = nullptr; errno = 0;
        const long v = (*q >= '0' && *q <= '9') ? std::strtol(q, &end, 10) : -1;
        if (v < 0 || end == q || errno == ERANGE || v < 2 || v > OZ2_PARAM.mod_cap || (*end && *end != ',')) {
            fprintf(stderr, "ERROR: OZ2_MODULI=%s は 2--%d の整数のカンマ区切りではない（空の要素・末尾の区切りも不可）\n",
                    fm, OZ2_PARAM.mod_cap); return false; }
        force.push_back((int)v);
        if (!*end) break;
        q = end + 1;
    }
    return true;
}

// 法の選択の引数（本数のほかは固定）．本数だけを変えて選び直す呼び出し（エラー文・注意が「本数を増やすと b がいくつに
// なるか」を確かめる）も同じ引数で呼ぶ．awe：段の時間のモデルで法の集合を選ぶ経路（BF16・FP16・BFP8_b・BFP4_b・BFP2_b．
// ozaki2_awe.h）．M・N は DRAM の見積りに使う
struct Oz2ModuliRequest {
    uint32_t M, N, K;
    int b_cap;
    bool w_measured;
    double wl;                       // 窓の上界の log2（実測したときだけ．幅 b_est で見積もった値）
    const std::vector<int>* force;   // 与えた集合（OZ2_MODULI．空なら選ぶ）
    Ozaki2Moduli choose(uint32_t S) const {
        return ozaki2_choose_moduli_w(K, (int)S, OZ2_PARAM.mod_cap, OZ2_ACC_BITS, OZ2_PARAM.digit_bound, b_cap,
                                      w_measured, wl, OZ2_PARAM.nd(), OZ2_PARAM.sparse_max, OZ2_PARAM.awe,
                                      force->empty() ? nullptr : force, M, N);
    }
};

// 窓を覆う法の積が取れず b が要求 target に届かないときの原因と対処．INT8 経路の法の上限，候補を使い切ったか，
// 本数を増やせば戻るか（実際に選び直して確かめる．表の選び方の経路は 1 回が速い）を書き分ける
static std::string oz2_short_cause(const Oz2Plan& r, const Ozaki2Moduli& mo, const Oz2ModuliRequest& rq, int target) {
    const bool int8 = OZ2_PARAM.digit_bound == 0;
    char buf[512]; std::string s;
    if (int8 && mo.m_cap < OZ2_PARAM.mod_cap) {
        snprintf(buf, sizeof buf, "INT8 経路は GEMM の出力の上界 K*floor(m/2)^2 を 2^%d-1 以下に保つため法が %d 以下に限られる．",
                 OZ2_ACC_BITS, mo.m_cap);
        s += buf;
    }
    if (!rq.force->empty()) {
        s += "与えた集合（OZ2_MODULI）の積で覆える最大で，本数の指定 num_moduli とは関係しない（集合を変えると戻る）";
        return s;
    }
    const uint32_t want = std::min<uint32_t>(r.S_req, OZ2_MAXS);
    if ((uint32_t)mo.ncand < want) {
        snprintf(buf, sizeof buf, "候補の法（%d 以下の表%s）を %d 本で使い切ったので，法の本数 num_moduli を増やしても戻らない",
                 mo.m_cap, int8 ? "と，その後ろへ足す 151 未満の互いに素な法" : "", mo.ncand);
        return s + buf;
    }
    if (want >= OZ2_MAXS) {
        snprintf(buf, sizeof buf, "法の本数の上限 %d 本（カーネルの MAXS）でも足りない", OZ2_MAXS);
        return s + buf;
    }
    for (uint32_t n = r.S_req + 1; n <= (uint32_t)OZ2_MAXS; n++) {
        const Ozaki2Moduli t = rq.choose(n);
        if (t.ok && t.b >= target) {
            snprintf(buf, sizeof buf, "法の本数の指定 num_moduli=%u が足りない（num_moduli=%u 以上にすると b=%d になる）",
                     r.S_req, n, t.b);
            return s + buf;
        }
    }
    const Ozaki2Moduli t = rq.choose(OZ2_MAXS);
    snprintf(buf, sizeof buf, "法の本数の指定 num_moduli=%u が足りず，上限の %d 本にしても b=%d にとどまる",
             r.S_req, OZ2_MAXS, t.ok ? t.b : 0);
    return s + buf;
}

// 累算の条件（acc_ok）で法の集合が取れないときの原因（落ちた条件ごとの文）と，落ちた条件をどれも満たす K の上限．
// 落ちた条件は法の選択が mo.acc_fails に落ちた順に残す（ozaki2_moduli.h）．条件ごとに K の上限（32 の倍数）を
// 出し，その最小を「通る上限」として示す．最初に落ちた条件の上限だけを示すと，その K でも別の条件で落ちる
// （表の選び方の TF32 の K=16416 は桁積の条件の上限 16384 を示すことになるが，16384 は合成の条件で落ち，通るのは
// 16352 まで．FP8 の 65568 も同じく 65536 ではなく 65504）．合成の条件の上限は，表の選び方（表へ戻った場合を含む）で
// 表の最大の法の桁積を先に法へ畳める K として出せる．段の時間のモデルが選んだ集合では出せないので，そのときは
// 通る上限を示さない．BFP4_b は K*7^2 > 2^24 で桁積の条件だけに掛かり，経路一律の桁の上限で決まる（法の選び方に
// 依らない）．段の時間のモデルで法を選ぶ経路は，モデルが集合を返さずに表の選び方へ戻った場合と，モデルの選んだ集合が
// 最後の検査で落ちる場合がある（K の区間は docs/awe_moduli.md の「K > 16352」）
static inline void oz2_report_accumulation_failure(const Oz2Plan& r, const Ozaki2Moduli& mo, const Oz2ModuliRequest& rq,
                                            const std::string& fset) {
    const uint32_t K = r.K;
    const bool forced = !rq.force->empty();
    int mmax = 0; for (int m : mo.m) mmax = std::max(mmax, m);
    const bool table_sel = !forced && (!OZ2_PARAM.awe || mo.awe_status == "table_fallback");
    double kmax = INFINITY; bool bounded = true, digit_only = true;
    auto lim = [&](double v) { kmax = std::min(kmax, std::floor(v / 32.0) * 32.0); };
    std::string cause; bool seen_digit = false, seen_one = false, seen_comp = false;
    char buf[512];
    for (const auto& f : mo.acc_fails) {
        if (f.why == "digit") {
            lim(std::ldexp(1.0, 24) / ((double)f.d * f.d));
            if (seen_digit) continue;
            seen_digit = true;
            snprintf(buf, sizeof buf, "桁積の上界 K*d^2 = %.0f（d=%d%s）が FP32 の無誤差整数の上限 2^24 を超える",
                     f.val, f.d, f.d != (int)OZ2_PARAM.digit_bound ? "．選んだ法の桁の上限の最大" : "");
        }
        else if (f.why == "one") {
            digit_only = false;
            lim((std::ldexp(1.0, 24) - f.m) / ((double)f.d * f.d));
            if (seen_one) continue;
            seen_one = true;
            snprintf(buf, sizeof buf, "1 桁の法 %d の GEMM の出力の上界 K*R^2 + m = %.0f（R=%d）が 2^24 を超え，"
                     "再構成が FP32 で法へ畳めない", f.m, f.val, f.d);
        }
        else {   // compose
            digit_only = false;
            if (table_sel) lim((std::ldexp(1.0, 24) - (double)mmax) /
                               ((double)OZ2_PARAM.digit_bound * OZ2_PARAM.digit_bound));
            else bounded = false;
            if (seen_comp) continue;
            seen_comp = true;
            snprintf(buf, sizeof buf, "桁積を先に法へ畳めない法があり（出力 1 本の上界に法を足すと 2^24 を超えるか，"
                     "畳んだ後の総和が 2^22 以上），2^12 分割の合成の総和の上界も法 %d で %.0f と 2^22 以上になる",
                     f.m, f.val);
        }
        cause += (cause.empty() ? "" : "．また，") + std::string(buf);
    }
    std::string bound, note;
    if (bounded && std::isfinite(kmax)) {
        // 表の選び方では経路の上限そのもの．モデルや与えた集合では，その集合の桁の上限での値（K を変えるとモデルの選ぶ
        // 集合も変わる）
        const std::string by = (table_sel && seen_comp) ? "．合成の条件は表の最大の法 " + std::to_string(mmax) + " で数える"
                             : (!table_sel ? "．この集合の桁の上限で数えた値" : "");
        snprintf(buf, sizeof buf, "（落ちた条件をどれも満たすのは K <= %.0f（32 の倍数）まで%s）", kmax, by.c_str());
        bound = buf;
    }
    // 桁積の条件だけで落ち，経路の桁の上限が法に依らない（疎な桁も 1 桁の法も使わない）なら，限界は法の選び方に依らない
    const bool route_limit = digit_only && !OZ2_PARAM.sparse_max && !OZ2_PARAM.one_digit;
    if (route_limit) {
        snprintf(buf, sizeof buf, "．この上限は経路の桁の上限 d=%d だけで決まり，法の集合の選び方に依らない",
                 (int)OZ2_PARAM.digit_bound);
        note = buf;
    }
    else if (OZ2_PARAM.awe && !forced) {
        const bool dram = mo.awe_caps.find("dram_retry") != std::string::npos;
        char nb[640];
        if (mo.awe_status == "table_fallback")
            snprintf(nb, sizeof nb, "．法の集合を選ぶ段の時間のモデル（ozaki2_awe.h）が集合を返さず，表の選び方"
                     "（法の最大 %d）へ戻った結果である（awe=table_fallback，awe_caps=%s%s）", mmax, mo.awe_caps.c_str(),
                     dram ? "．dram_retry は M・N から見積もる DRAM の条件を外した探し直しで，M・N が小さい構成なら"
                            "同じ K でもモデルが集合を選んで通ることがある" : "");
        else
            snprintf(nb, sizeof nb, "．段の時間のモデル（ozaki2_awe.h）が選んだ集合（法の最大 %d）がこの最後の検査で"
                     "落ちた（モデルの候補の検査とは上界の取り方が違う．K を少し変えると通ることがある．awe_caps=%s）",
                     mmax, mo.awe_caps.c_str());
        note = nb;
    }
    if (!forced)
        fprintf(stderr, "ERROR: K=%u ではこの経路（桁の上限 %d）の累算の条件を満たす法の集合が取れない：%s%s．"
                        "法の本数（num_moduli=%u）を変えても解決しない%s\n",
                K, (int)OZ2_PARAM.digit_bound, cause.c_str(), bound.c_str(), r.S_req, note.c_str());
    else
        fprintf(stderr, "ERROR: K=%u では %sは，この経路（桁の上限 %d）の累算の条件を満たさない：%s%s．"
                        "集合をこの K で条件を満たす法に変えるか，K を小さくする%s\n",
                K, fset.c_str(), (int)OZ2_PARAM.digit_bound, cause.c_str(), bound.c_str(), note.c_str());
}

// 法の集合が取れない（!mo.ok）ときの原因と対処をエラー文で出す（fm は与えた集合 OZ2_MODULI の文字列．無ければ nullptr）
static inline void oz2_report_moduli_failure(const Oz2Plan& r, const Ozaki2Moduli& mo, const Oz2ModuliRequest& rq,
                                      const char* fm) {
    const uint32_t K = r.K;
    const bool int8 = OZ2_PARAM.digit_bound == 0;
    const bool forced = !rq.force->empty();
    // 与えた集合（OZ2_MODULI）は落とす・選び直すをしないので，原因は本数ではなく集合である（文の主語を集合にし，
    // 本数の案内を出さない）
    const std::string fset = !forced ? std::string() : std::string("OZ2_MODULI=") + fm + " の集合";
    if (!mo.acc_ok && int8) {   // INT8 経路：GEMM の出力が符号-絶対値 int32 に収まらない法がある（表・延長・選び直しの法は
                                // m_cap 以下なので，ここへ来るのは集合を与えたときだけ）
        std::string bad;
        for (int m : mo.m) if (!oz2_int8_fits(K, m, OZ2_ACC_BITS)) bad += (bad.empty() ? "" : ",") + std::to_string(m);
        fprintf(stderr, "ERROR: K=%u では%s%s法 {%s} の GEMM の出力の上界 K*floor(m/2)^2 が符号-絶対値 int32 の"
                        "絶対値の上限 2^%d-1 を超える（%s）\n",
                K, !forced ? "" : " ", !forced ? "" : (fset + "の").c_str(), bad.c_str(), OZ2_ACC_BITS,
                !forced ? "法の本数を変えても解決しない．K を小さくするか法を小さくする"
                        : "集合をこの K で使える小さい法に変えるか，K を小さくする");
    }
    else if (!mo.acc_ok)   // 本数ではなく累算器の幅が原因（桁分割の経路で K が大きすぎる）
        oz2_report_accumulation_failure(r, mo, rq, fset);
    else if (forced)   // 与えた集合は落とす・選び直すをしないので，本数の指定とは関係なく止まる
        fprintf(stderr, "ERROR: OZ2_MODULI=%s の法の集合（%d 本，導かれた b=%d）は K=%u で使えない"
                        "（法の上限を超えるか桁に収まらない法・互いに素でない組・本数の上限 %d 超え・b=1 も取れない窓の"
                        "どれか．窓が足りないだけなら b が下がって通る）\n",
                fm, mo.s, mo.b, K, OZ2_MAXS);
    else
        fprintf(stderr, "ERROR: num_moduli=%u が K=%u で使えない（取れた法は %d 本，導かれた b=%d）：%s\n",
                r.S_req, K, mo.s, mo.b, oz2_short_cause(r, mo, rq, 1).c_str());
}

// 求めた b が窓に入り切らないことがある．CRT の窓は法の積で決まり，候補を使い切っても
// 2b+1+log2(K) bit に届かなければ b はそこで頭打ちになる（INT8 経路の K=14080 で窓を最悪ケースの
// K*2^(2b) に取ると，表 OZ2_TABLE_I8 の 24 本を使い切って b=84．GEMMul8 の表の 20 本までなら b=70）．
// 相対誤差は 2^-b に比例するので，不足 1 bit がそのまま到達精度の 1 bit である．
// 大きい K では窓 2b+1+log2 K が広がり，法の本数の指定で覆えないと b が下がる（INT8 の K=133120・法 16 本で b=53．
// 17 本以上なら 54）．INT8 経路は K > 133144 で法の上限も下がる（ozaki2_moduli.h の表の延長）．
// FP64 相当（OZ2_TARGET_BITS = 53 bit）に届かない b（既定の上限 54 と 53 の小さいほう．OZ2_FP64 の要求が 53 なら 53）になる
// 構成は，デバイスを開く前に止める（false）：出るのが CONFIG・moduli 行の b= だけだと，精度を測らない性能測定（OZ_NOVERIFY）は
// Test Skipped・終了コード 0 のまま FP64 相当に遠い構成を測る（INT8 の K=262144 は表を延ばす前は b=16）．b を明示した（OZ2_B）か
// 法の集合を与えた（OZ2_MODULI）調査用の構成は止めずに注意だけを出す．
// b が要求より下がったことは，指定元（既定・OZ2_FP64 の要求・OZ2_B）によらず明示する（OZ2_B=54 を明示しただけで
// 注意が消えると，既定と同じ指定で b=53 を黙って測ることになる）
static inline bool oz2_check_width(const Oz2Plan& r, const Ozaki2Moduli& mo, const Oz2ModuliRequest& rq, bool b_env) {
    const uint32_t K = r.K;
    const uint32_t b = (uint32_t)mo.b;
    const int b_cap = rq.b_cap;
    const int b_floor = std::min(b_cap, OZ2_TARGET_BITS);
    if (!b_env && rq.force->empty() && (int)b < b_floor) {
        fprintf(stderr, "ERROR: K=%u では法の積が足りず，窓 2b+1+log2 K を覆える b は %u で，%s（%d）にも FP64 相当の %d にも届かない"
                        "（法 %d 本）：%s．精度を下げてよい調査なら OZ2_B で b の上限を明示すると流せる\n",
                K, b, r.auto_b ? "OZ2_FP64 の要求" : "既定の上限", b_cap, b_floor, mo.s, oz2_short_cause(r, mo, rq, b_floor).c_str());
        return false;
    }
    if ((int)b < b_cap)
        printf("  注意：法の積が足りず b は%s %d ではなく %u（%d bit 少ない．法 %d 本の積で K=%u の窓 2b+1+log2 K を覆える最大）．"
               "%s：%s\n",
               b_env ? " OZ2_B の要求" : (r.auto_b ? " OZ2_FP64 の要求" : "既定の"), b_cap, b, b_cap - (int)b, mo.s, K,
               r.auto_b && !b_env ? "到達精度もおおむねそのぶん下がる（b の要求には 0.5 bit の安全余裕を含む）"
               : (b_env ? "到達精度もおおむねそのぶん下がる"
                        : "FP64 の仮数に正規化の 1 bit を足した既定の幅に届かないので，入力によっては到達精度が下がる"),
               oz2_short_cause(r, mo, rq, b_cap).c_str());
    return true;
}

// 再構成のパス数（OZ2_PASSES．未設定 preq なら OZ2_FP64 のとき auto，それ以外は既定の 1 パス．pq は解釈した後の要求）を
// 決め，見積りで決めたときは OZ2_PASSES の行を印字する
static inline void oz2_choose_passes(Oz2Plan& r, const Oz2PrecisionPolicy& pol, Oz2PassReq preq, Oz2PassReq pq) {
    r.recon1_log2 = oz2_recon1_log2(r.mo, (int)r.b, pol);
    const double pass1_max = OZ2_RECON1_THR_LOG2;
    r.two_pass = (pq == Oz2PassReq::Auto) ? !(r.recon1_log2 <= pass1_max) : (pq == Oz2PassReq::Two);
    r.pass_mode = (preq == Oz2PassReq::Unset) ? (r.auto_b ? "auto" : "fixed")
                : (preq == Oz2PassReq::Auto ? "auto" : "forced");
    // 人が読む行．同じ値（見込み・条件・回数）を CONFIG の recon1_pred=・recon1_thr=・two_pass= にも出し，照合はそちらを先に
    // 読む（この行の文面は CONFIG の欄の無い 2026-10-22 より前のログのために verify/check_paper_numbers.py の oz2_passes が，
    // 「auto … -> N パス」を scripts/selftest.sh の auto の項目が読む）
    if (r.pass_mode != "fixed")
        printf("OZ2_PASSES: %s，1 パスの復元誤差の見込み 2^%.2f（1 パスにする条件 2^%.1f 以下）-> %d パス\n",
               r.pass_mode.c_str(), r.recon1_log2, pass1_max, r.two_pass ? 2 : 1);
}

// 法の集合と幅 b を決める（ホストだけで完結し，デバイスは開かない）．
// OZ2_B の値，OZ2_FP64 の自動決定（OZ2_WIN・OZ2_FP64 の行を印字），法選択と窓の
// 見積もり直し，失敗時のエラー表示，b の頭打ちの注意，再構成のパス数（OZ2_PASSES の行を印字），合成段の平面までを決める．
// 確保も解放もしない．
// 使えない指定なら false を返す（main は 2 で終わる）．
static inline bool choose_moduli_and_width(Oz2Plan& r, const Oz2Knobs& k, const std::vector<double>& A, const std::vector<double>& B) {
    const uint32_t M = r.M, N = r.N, K = r.K;
    // OZ2_FP64：幅 b を入力から決める（scheme I の OZ_FP64 に対応する調整．誤差モデルは
    // ozaki2_autop.hpp の choose_precision_policy の前の説明）．
    // 上限 OZ2_MAXDIGIT*OZ2_DIGIT_BITS = 100 は A' を 2^20 進 5 桁で送る形式が運べる幅である．
    // b が 60 を超えるときだけスライサを 4 桁に，80 を超えるときだけ 5 桁に切り替える（既定の b=54 では 3 桁のまま）．
    int b_cap = OZ2_B_DEFAULT;
    const Oz2PassReq preq = k.passes;
    // 調査用の上書き OZ2_B（Oz2Knobs::b）は b_cap に載せて法選択へ渡す
    const bool b_env = k.b.has_value();
    if (b_env) b_cap = (int)*k.b;
    // CRT の窓は既定では最悪ケース上界 log2(K) を使い，自動決定（OZ2_FP64）が入力から
    // 実測したときだけ差し替わる（OZ2_WIN）．
    Oz2PrecisionPolicy pol; pol.b_cap = b_cap; pol.s_req = r.S_req;
    r.auto_b = k.auto_b;
    if (r.auto_b) {
        pol = choose_precision_policy(A, B, M, N, K, b_env, k.nowin, pol);
        b_cap = pol.b_cap; r.S_req = pol.s_req;
    }
    r.auto_b_cap = r.auto_b ? b_cap : 0;
    r.win_meas = pol.w_measured;
    // パス数を見積りで選ぶ（OZ2_FP64 の既定か OZ2_PASSES=auto）なら，b の自動決定をしない構成でも入力の統計を測る
    const Oz2PassReq pq = (preq == Oz2PassReq::Unset) ? (r.auto_b ? Oz2PassReq::Auto : Oz2PassReq::One) : preq;
    if (pq == Oz2PassReq::Auto && !pol.stats) oz2_measure_stats(A, B, M, N, K, pol);
    std::vector<int> force;
    if (!oz2_parse_forced_moduli(k.moduli_set, force)) return false;
    // 実測の窓は b に依る（oz2_window_log2）が，b は窓から決まる．b_cap で見積もって
    // 選び，得た b がそれより小さければその b で見積もり直す．係数は b が小さいほど
    // 大きいので b は単調に下がり，止まったときの窓は得た b そのもので見積もってある．
    // 係数が下限 1.001 に張り付く b（既定の b=54 など）では 1 回目で止まる．
    Oz2ModuliRequest rq{M, N, K, b_cap, pol.w_measured, 0.0, &force};
    Ozaki2Moduli& mo = r.mo;
    for (int b_est = b_cap;;) {   // b_est：窓の係数を見積もる幅（GEMM の K ブロック幅 bw とは無関係）
        rq.wl = pol.w_measured ? oz2_window_log2(pol.wA, pol.wB, K, b_est) : 0.0;
        mo = rq.choose(r.S_req);
        if (!pol.w_measured || !mo.ok || mo.b >= b_est) break;
        b_est = mo.b;
    }
    if (!mo.ok) { oz2_report_moduli_failure(r, mo, rq, k.moduli_set); return false; }
    if (!oz2_check_width(r, mo, rq, b_env)) return false;
    const uint32_t b = (uint32_t)mo.b;
    r.b = b;
    oz2_choose_passes(r, pol, preq, pq);
    // A' を運ぶ桁数．3 桁で 60 bit，4 桁で 80 bit まで運べるので，b がそれを超えるときだけ桁を増やす．
    r.ND = (b > 4 * OZ2_DIGIT_BITS) ? 5u : ((b > 3 * OZ2_DIGIT_BITS) ? 4u : 3u);
    r.S = mo.s;
    r.Mt = M / 32; r.Nt = N / 32; r.Kt = K / 32;
    r.ntotA = r.Mt * r.Kt; r.ntotB = r.Kt * r.Nt; r.ntotC = r.Mt * r.Nt;
    r.cplan = make_compose_plan(mo);
    return true;
}

// ===== 再構成段の計画 =====
// パス数・各パスの定数・limb を DRAM へ残すか・FP64 を組み立てるかまでを，
// バッファを確保する前にここで決め切る．確保量もカーネル引数もこの計画から導く．
//
// 【なぜ分けるか】確保のあとで診断用のつまみが出力の形を変えると，確保と書き込みがずれる．
// 例えば 1 パス構成の limb 面を 1 タイルに縮めて確保した後で OZ2_RECON_NOF64 が FINAL を
// 落とすと，ライタが !FINAL の枝に入って 1 タイルの器へ行列全体を書く．
// 計画を先に固定すれば，この種のずれは起こりようがない．
struct ReconPass {
    std::vector<float> rp;        // 逆数の断片（[i*nrp + t] の並び）
    uint32_t nrp = 0;
    bool assemble_f64 = false;    // このパスが FP64 を組み立てて書く（カーネルの FINAL）
    bool read_other = false;      // 第 1 パスの 128 bit 固定小数を読んで足す（2 パス構成の最終パス．ROLE_TWO_FINAL）
    bool pack_a128 = false;       // 自分の limb を 128 bit 固定小数へ足して書く（2 パス構成の第 1 パス．ROLE_TWO_FIRST）
    bool limb3 = false;           // limb を 3 本出す（FP64 相当，または 2 パス構成の全パス）
    // パスの役割（カーネルのコンパイル時引数 CT_ROLE．recon_args.h の ReconRole）
    recon_args::ReconRole role() const {
        return pack_a128 ? recon_args::ROLE_TWO_FIRST : (read_other ? recon_args::ROLE_TWO_FINAL : recon_args::ROLE_SINGLE);
    }
};
// 初期化子は {rp, nrp, assemble_f64, read_other, pack_a128, limb3} の順に 6 つ全部を書く（欄を足したときに
// 古い初期化子の値が 1 つずれて別の欄へ入るのを，下の 2 か所の push_back で見落とさないため）．
struct ReconPlan {
    std::vector<ReconPass> pass;
    bool save_limbs = false;      // limb 3 面を DRAM へ残すか（診断用 OZ2_RECON_NOF64 の最終パス）
    uint32_t limb_tiles = 1;      // そのための確保量［タイル］
    uint32_t a128_tiles = 1;      // 2 パス構成の第 1 パスが書く 128 bit 固定小数の確保量［16 KB ページ］
};
// 128 bit 固定小数 1 タイル（1024 要素 x 16 B）のページの大きさ（カーネルと共有の f64_tf.h の OZ_A128_TILE_BYTES）
constexpr uint32_t OZ2_A128_PAGE = OZ_A128_TILE_BYTES;

// assemble：最終パスが FP64 を組み立てるか（診断用 OZ2_RECON_NOF64 なら false．Oz2Diag::nof64 の否定）
static ReconPlan make_recon_plan(const Ozaki2Moduli& mo, bool two_pass, uint32_t ntot, bool assemble) {
    const uint32_t S = (uint32_t)mo.s;
    auto pack = [&](auto& src, uint32_t np) {
        std::vector<float> c((size_t)S * np);
        for (uint32_t i = 0; i < S; i++)
            for (uint32_t t = 0; t < np; t++) c[(size_t)i * np + t] = src[i][t];
        return c;
    };
    // 診断用（OZ2_RECON_NOF64）：最終パスに FP64 を組み立てさせず limb 3 面を書かせる（答えは壊れる）．
    // 書き出しが FP64 の 8 B/要素から limb の 12 B/要素に増えるので，測れるのは組み立てを除いた再構成段の時間の
    // 上界である．書き出し量を変えずに組み立てだけを外すのは OZ_RECON_PROBE=1（ozaki2_stages.hpp の recon_make_kernels）．
    ReconPlan p;
    if (!two_pass) {
        p.pass.push_back({pack(mo.rp0, OZ2_NRP0), OZ2_NRP0, assemble, false, false, oz2_recon_limb3(mo, false)});
    } else {
        // 第 2 パスが最終：第 1 パスは自分の 3 本を 128 bit 固定小数へ足して DRAM に残し（16 B/要素），
        // 第 2 パスはそれを読んで自分の 3 本を足して畳む．6 本をまとめて足すのとビット単位で同じで，
        // 第 2 パスの組み立ての半分を第 1 パスへ移せる
        // （第 1 パスは CRT で決まり，データ移動コアが空いている．docs/recon_cost_breakdown.md の結果 7・8）．
        // limb は両パスとも 3 本に揃える（第 1 パスのライタが必ず 3 本読む．規則は ozaki2_moduli.h の oz2_recon_limb3）．
        p.pass.push_back({pack(mo.rp1, OZ2_NRP1), OZ2_NRP1, false,    false, true,  oz2_recon_limb3(mo, true)});
        p.pass.push_back({pack(mo.rp2, OZ2_NRP2), OZ2_NRP2, assemble, true,  false, oz2_recon_limb3(mo, true)});
    }
    // limb 3 面を DRAM へ書くのは，FINAL でも第 1 パスでもないパスだけである（recon_writer.cpp の
    // !FINAL && ROLE != ROLE_TWO_FIRST の枝．診断用 OZ2_RECON_NOF64 の最終パス）．1 本でもそういうパスがあれば全面が要る．
    bool pack_any = false;
    for (const ReconPass& pass : p.pass) {
        if (!pass.assemble_f64 && !pass.pack_a128) p.save_limbs = true;
        pack_any = pack_any || pass.pack_a128;
    }
    // 書かれない場合も TensorAccessorArgs を組む先は要るので 1 タイルだけ確保する．
    // 全面を取ると 14080 角で 2.38 GB を，1 度も触らないまま DRAM から取り上げる．
    p.limb_tiles = p.save_limbs ? ntot : 1u;
    p.a128_tiles = pack_any ? ntot : 1u;
    return p;
}

// 再構成の出口のスケール（すべてのパスで同じ）：KM = M*2^-2b を 64 bit の仮数 W（上位 32 bit が Whi，下位が Wlo）と
// 指数 qw に分けたもの（double の定数として掛けると，そこで相対 2^-53 が全要素に一様に乗るので，整数の 64x64 -> 128 bit
// 乗算で掛ける．分け方は尾崎 I の recon_scale と同じ common/oz_bits.h の oz_split_scale_ld），と正規化を掛け戻す指数 sh2
// （= 2*oz2_shift．スライサが初手で落とした位の 2 つぶん）
struct ReconScale { uint32_t Whi = 0, Wlo = 0; int32_t qw = 0, sh2 = 0; };
static ReconScale make_recon_scale(const Ozaki2Moduli& mo, int32_t sh2) {
    const OzScale64 w = oz_split_scale_ld(mo.KMl);
    return {w.hi, w.lo, w.q, sh2};
}

// FP64 組み立て（2 パス構成の第 1 パスでは 128 bit 固定小数への変換）をリーダとライタで分担するか．
// 分担するのはライタが要素ごとの整数演算を行うパス（FP64 を組み立てる最終パスと，2 パス構成の第 1 パス）で，
// 境目 jsplit（Oz2Knobs::jsplit）が 0 か 1024 なら分担そのものをやめる：片方が 0 要素だと受け渡しの片側が立たず，
// 相手が永久に待つ．再構成の構築（ozaki2_stages.hpp の recon_make_cbs）と CONFIG 行の jsplit_on が同じこの 1 つの条件を見る．
static inline bool oz2_split_assembly(bool assembles, uint32_t jsplit) {
    return assembles && jsplit > 0u && jsplit < 1024u;
}

// 自動で数えた本数 auto_n に，つまみの指定（Oz2Knobs::nsmall）を重ねる．
static inline uint32_t oz2_apply_nsmall(uint32_t auto_n, uint32_t nrp, int nsmall) {
    if (nsmall == -1) return auto_n;
    if (nsmall == -2) return nrp;
    if ((uint32_t)nsmall > nrp) {
        fprintf(stderr, "ERROR: OZ2_NSMALL=%d はこの経路の断片数 %u を超える\n", nsmall, nrp);
        std::exit(2);
    }
    return (uint32_t)nsmall;
}

// ===== 法の組分け（OZ2_MOD_GROUPS で強制したときだけ．2026-11-03 から DRAM が足りないときは出力の行の帯に分ける：plan_bands）=====
// 確保も解放もしない．組が 1 つなら組に分けない経路（全法を一度にスライスする）を通る．
// 法どうしは独立なので，法の組ごとに「スライス -> GEMM -> 合成」を回せば，同時に置く桁行列は 1 組ぶんで済む（答えは変わらない）．
// ただし組ごとに FP64 の A・B を読み直してスライスし直すので，14080^3 の II-BF16 では slice が 222 ms から 402 ms に増える．
// そのため自動では選ばず，小さいサイズで組分けの経路を踏むための強制（性能だけが変わる）として残す．
// 段の時間のモデル（ozaki2_awe.h）の DRAM の見積りは ozaki2_split.h の oz2_dram_one_group（1 組で A・B を手放す前提．
// 余裕 OZ2_DRAM_MARGIN は両スキーム共通の OZ_DRAM_MARGIN_BYTES（0.5 GiB））のままである．
// 組の境目は，組の桁行列の枚数の最大（DRAM の見積りを決める量）が最小になるように置く（組は連続した法の並び）．
// 境目を「目標枚数以上になる側」へ常に切り上げる貪欲（mod_group_bounds_greedy）では最大が最小になるとは限らない
// （法ごとの枚数 [3,2,2] の 2 組を貪欲は [5,2] にするが，[3,4] なら最大 4 で済む）．法は高々 24 本なので，連続区間への分割の
// 最大を最小にする動的計画法 O(n S^2) で求める．最適な分割が複数あるときは，貪欲の境目がその 1 つなら
// それを使う（貪欲が最適な構成では貪欲の境目のままで，貪欲が最適でないときだけ境目が変わる）．

// 法 [s0,s1) の桁行列の枚数（poff の差．スライサの出力の平面数で，連結した A の行の幅 PA*Kt を決める）
static inline uint32_t group_planes(const Ozaki2Moduli& mo, uint32_t s0, uint32_t s1) {
    return (uint32_t)((s1 < (uint32_t)mo.s ? mo.poff[s1] : mo.nplane) - mo.poff[s0]);
}
static inline uint32_t mod_group_maxplanes(const Ozaki2Moduli& mo, const std::vector<uint32_t>& g) {
    uint32_t mx = 0;
    for (size_t k = 0; k + 1 < g.size(); k++) mx = std::max(mx, group_planes(mo, g[k], g[k + 1]));
    return mx;
}
static std::vector<uint32_t> mod_group_bounds_greedy(const Ozaki2Moduli& mo, uint32_t n) {
    const uint32_t S = (uint32_t)mo.s, np = (uint32_t)mo.nplane;
    std::vector<uint32_t> g(n + 1, 0); g[n] = S;
    for (uint32_t k = 1; k < n; k++) {
        uint32_t i = g[k - 1] + 1;   // 組は空にしない
        while (i < S - (n - k) && (uint64_t)mo.poff[i] * n < (uint64_t)k * np) i++;
        g[k] = i;
    }
    return g;
}
static std::vector<uint32_t> mod_group_bounds(const Ozaki2Moduli& mo, uint32_t n) {
    const uint32_t S = (uint32_t)mo.s;
    TT_FATAL(n >= 1 && n <= S, "mod_group_bounds: 組数 {} が 1..{} の外", n, S);
    // best[k][i]：先頭 i 本の法を k 組に分けたときの最大枚数の最小．cut[k][i]：そのときの最後の組の先頭
    const uint32_t INF = UINT32_MAX;
    std::vector<std::vector<uint32_t>> best(n + 1, std::vector<uint32_t>(S + 1, INF)), cut(n + 1, std::vector<uint32_t>(S + 1, 0));
    best[0][0] = 0;
    for (uint32_t k = 1; k <= n; k++)
        for (uint32_t i = k; i <= S; i++)
            for (uint32_t j = k - 1; j < i; j++) {   // 最後の組は法 [j,i)
                if (best[k - 1][j] == INF) continue;
                const uint32_t v = std::max(best[k - 1][j], group_planes(mo, j, i));
                if (v < best[k][i]) { best[k][i] = v; cut[k][i] = j; }
            }
    std::vector<uint32_t> g(n + 1, 0); g[n] = S;
    for (uint32_t k = n, i = S; k >= 1; k--) { g[k - 1] = cut[k][i]; i = cut[k][i]; }
    const std::vector<uint32_t> gg = mod_group_bounds_greedy(mo, n);
    return mod_group_maxplanes(mo, gg) == best[n][S] ? gg : g;
}

// 同時確保の見積りに足す余裕（1 GiB．小さい器と，見積りに入れない端数を受ける）
constexpr uint64_t OZ2_WHOLE_DRAM_MARGIN = 1ull << 30;

// 出力の行の帯と法の組（plan_panels の後，FP64 のオペランドを送った後に main が呼ぶ）．
// 帯と組の数は「DGEMM の 1 回の呼び出し」の同時確保で決める：A・B は呼び出しの後も使うので壊さず（FP64 のオペランドは
// 確保済みで，空き df はその後の値なので足さない），段の途中で器を手放さない（OZ_TIME_WHOLE が流し直す全工程がそのまま
// 載る）．OZ_TIME_WHOLE を立てない実行も同じ帯・組で流す（段の途中で手放すのはそのままで，構成は通しの計時と同じになる）．
//   A の桁行列（帯 1 本ぶん × 組の桁行列の枚数の最大）+ B の桁行列（全体 × 同じ枚数）+ 積の器（桁積の平面 cplan.planes
//   + 剰余 S 枚．帯 1 本ぶん）+ 出口の o64（全体．呼び出し側の C）+ 再構成の limb 3 面と 128 bit 固定小数（帯 1 本ぶん．
//   make_recon_plan）+ 余裕 OZ2_WHOLE_DRAM_MARGIN
// 帯はパネル行を等分する本数（bpr が nbr を割り切る）から，収まる最小の本数を選ぶ（OZ_BANDS で強制できる．等分できない
// 本数は止める）．法の組は OZ2_MOD_GROUPS で強制したときだけ分ける（帯とは併用しない）．
// 器は 1 つずつ全バンクへ同じ番地で置かれるので，空きの総量に加えて，最大の単一の器が最大の連続した空きに入るかも見る．
// これは必要条件にすぎず，先に確保した器が連続した空きを割った後の大きさは見ていないので，それで足りなければ確保の
// TT_FATAL で落ちる．収まる帯の本数が無い・強制した本数で収まらない構成は fits=false（main が閉じてから 2 で終わる）．
// df：DRAM の空き（main が照会する）．assemble：再構成の最終パスが FP64 を組み立てるか（make_recon_plan）．
// 戻り値：帯・組の指定が成り立つか（OZ_BANDS が等分できない，帯と組を両方立てた，なら理由を表示して false）
static inline bool plan_bands(Oz2Plan& r, const DramFree& df, bool assemble, std::optional<uint32_t> bands_req) {
    const Ozaki2Moduli& mo = r.mo;
    // 段の時間のモデル（ozaki2_awe.h）は 1 組の見積りを桁行列のタイル長 OZ2_AWE_SLICE_TILE で立てる（モデルの見積りは
    // A・B を手放す前提の oz2_dram_one_group のままで，ここでは帯に分けて収める）
    static_assert(OZ2_PARAM.awe == 0 || OZ2_AWE_SLICE_TILE[OZ2_PARAM.awe - 1] == tt::tile_size(OZ_FMT.slice_out),
                  "段の時間のモデルの桁行列のタイル長（ozaki2_awe.h の OZ2_AWE_SLICE_TILE）が形式のタイル長と違う");
    if (bands_req && r.groups_req > 1) {
        fprintf(stderr, "ERROR: OZ_BANDS と OZ2_MOD_GROUPS は併用できない（帯と法の組のどちらか一方で DRAM に収める）\n");
        return false;
    }
    const uint64_t slt = tt::tile_size(OZ_FMT.slice_out), outt = tt::tile_size(OZ_FMT.out);
    const uint64_t f32t = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t cpl = r.cplan.planes;   // 合成段の平面（INT8 経路は 0）
    const uint64_t o64B = (uint64_t)r.ntotC * OZ_F64_TILE_BYTES;
    struct Need { uint64_t a, b, q, limb, total, maxbuf; };
    auto need = [&](uint32_t bpr, uint32_t ng) {
        const uint32_t mp = mod_group_maxplanes(mo, mod_group_bounds(mo, ng));
        const uint64_t ntCb = (uint64_t)bpr * r.pMt * r.Nt;
        const ReconPlan rpl = make_recon_plan(mo, r.two_pass, (uint32_t)ntCb, assemble);
        Need w;
        w.a = (uint64_t)mp * bpr * r.pMt * r.Kt * slt;
        w.b = (uint64_t)mp * r.ntotB * slt;
        w.q = (uint64_t)(cpl + r.S) * ntCb * outt;
        w.limb = (uint64_t)rpl.limb_tiles * 3 * f32t + (uint64_t)rpl.a128_tiles * OZ2_A128_PAGE;
        w.total = w.a + w.b + w.q + o64B + w.limb + OZ2_WHOLE_DRAM_MARGIN;
        w.maxbuf = std::max({w.a, w.b, (uint64_t)cpl * ntCb * outt, (uint64_t)r.S * ntCb * outt, o64B,
                             (uint64_t)rpl.limb_tiles * f32t, (uint64_t)rpl.a128_tiles * OZ2_A128_PAGE});
        return w;
    };
    auto fits = [&](const Need& w) { return w.total <= df.total && w.maxbuf <= df.largest_block; };
    const uint32_t ng = r.groups_req ? r.groups_req : 1u;
    uint32_t bpr = r.nbr;
    if (bands_req) {
        const uint32_t nb = std::min(*bands_req, r.nbr);   // パネル行数を超える指定は 1 本 1 パネル行（scheme I と同じ）
        if (r.nbr % nb) {   // 指定した本数そのもので等分できるかを見る（切り上げた帯の高さで見ると 4 行に 3 本が 2 本で通る）
            fprintf(stderr, "ERROR: OZ_BANDS=%u ではパネル行 %u 行を等分できない（%u を割り切る本数にする）\n",
                    *bands_req, r.nbr, r.nbr);
            return false;
        }
        bpr = r.nbr / nb;
    } else if (ng == 1) {
        for (uint32_t k = r.nbr; k >= 1; k--)   // 等分できる本数のうち，収まる最小の本数（パネル行の最大の約数）
            if (r.nbr % k == 0) { bpr = k; if (fits(need(k, 1))) break; }
    }
    const Need w = need(bpr, ng), w1 = need(r.nbr, ng);
    r.bpr = bpr; r.nband = r.nbr / bpr;
    r.whole_fits = fits(w);
    // 「Band    : 出力を N 本の行の帯に分ける」は帯の数の出どころ（帯の数は CONFIG の bands= にもある）
    if (r.nband > 1)
        printf("Band    : 出力を %u 本の行の帯に分ける（帯 1 本 %u パネル行．%s．1 本の見積り %.1f GB，%u 本の見積り %.1f GB，"
               "DRAM の空き %.1f GB）\n", r.nband, r.bpr, bands_req ? "OZ_BANDS で指定" : "1 本では DRAM に収まらないため",
               w1.total / 1e9, r.nband, w.total / 1e9, df.total / 1e9);
    if (r.groups_req > 1)
        printf("Group   : 法を %u 組に分ける（OZ2_MOD_GROUPS で指定）\n", ng);
    printf("Memory  : 同時確保の見積り %.1f GB（A の桁行列 %.1f + B の桁行列 %.1f + 積の器 %.1f + o64 %.1f + limb %.2f + 余裕 %.2f．"
           "DRAM の空き %.1f GB）．最大の器 %.1f GB（最大の連続した空き %.1f GB）%s\n", w.total / 1e9, w.a / 1e9, w.b / 1e9,
           w.q / 1e9, o64B / 1e9, w.limb / 1e9, OZ2_WHOLE_DRAM_MARGIN / 1e9, df.total / 1e9, w.maxbuf / 1e9,
           df.largest_block / 1e9,
           w.total > df.total ? "  -- 収まらない" : (w.maxbuf > df.largest_block ? "  -- 最大の器が連続した空きに収まらない" : ""));
    r.ngroups = ng;
    r.gbeg = mod_group_bounds(mo, ng);
    return true;
}

// ===== 第 2 段の計画：出力パネルの分け方と GEMM のブロック構成 =====
// 出力をパネルに分けて，コアあたりの出力ブロックが L1 に収まるようにする．
// パネルの辺を 120 タイル以下に抑えると，11x10 のコア格子でコアあたり高々
// 12x11 = 132 タイル，Int32 タイルが 4 KiB なので約 540 KiB となり，入力側の CB を
// 足してもコアあたり 1.5 MiB の L1 に収まる．
// パネルが 1 枚なら行列全体の素の GEMM と同じである．
// OZ2_PANEL_FORCE=RxC（Oz2Knobs::panel_force．立っていなければ nullptr で true，a・c は 0）を読む．R が Mt を，C が Nt を割り切り，
// どちらも 2 以上でなければ false（マルチキャスト経路はコア格子の縦横に 2 コア以上を要るので，辺 1 のパネルは割れない）．
// main がデバイスを開く前に呼んで止める（構成探索まで待つと，デバイスを開いてオペランドを送りスライス段を流した後で止まる）
static inline bool oz2_parse_panel_force(uint32_t Mt, uint32_t Nt, const char* e, unsigned& a, unsigned& c) {
    a = c = 0;
    if (!e) return true;
    char tail = 0;
    if (sscanf(e, "%ux%u%c", &a, &c, &tail) != 2 || a < 2 || c < 2 || Mt % a || Nt % c) {
        fprintf(stderr, "ERROR: OZ2_PANEL_FORCE=%s は Mt=%u・Nt=%u を割り切る RxC（R・C は 2 以上）ではない\n", e, Mt, Nt);
        return false;
    }
    return true;
}
// gd はコア格子の大きさ．構成が取れなければ理由を表示して false（main がデバイスを閉じて 2 で終わる）
static inline bool plan_panels(Oz2Plan& r, tt::tt_metal::CoreCoord gd, const Oz2Knobs& k) {
    // パネルの辺は Mt・Nt の 120 タイル以下の最大の約数に固定する（共有ヘッダの largest_div_le）．scheme II
    // 固有の決め方で，辺を探さない（gemm_bf16・gemm_bf16ext の choose_panels は約数の組を面積の大きい順に探す）．
    // そのため最大の約数でコア格子に割れない形は，他の約数なら組めても止まる（M=5056 は Mt=158=2x79 の辺 79 が
    // 縦 10 コアに分けられず，下の choose_cfg が構成を返さない．src/README.md「対応する形状」）
    r.pMt = largest_div_le(r.Mt, 120); r.pNt = largest_div_le(r.Nt, 120);
    // 調査用：OZ2_PANEL_FORCE=RxC［タイル］でパネルの寸法を指定する（R が Mt を，C が Nt を割り切ること）．
    // コアあたりの出力ブロックを小さくしたときの GEMM 段の時間を見るためのもので（段をチップ内で連結する
    // 設計の見積もり．docs/fusion_estimate.md），積は変わらない．CONFIG 行の knobs 欄に残る（oz2_knobs_record）．
    {   // 値の検査は main がデバイスを開く前に済ませている（oz2_parse_panel_force）
        unsigned a = 0, c = 0;
        if (!oz2_parse_panel_force(r.Mt, r.Nt, k.panel_force, a, c)) return false;
        if (a) { r.pMt = a; r.pNt = c; }
    }
    r.nbr = r.Mt / r.pMt; r.nbc = r.Nt / r.pNt;
    // L1 はバイトで見積もる（共有ヘッダの既定はタイル数 400 の固定予算で，
    // 本経路の入力 1 B／出力 4 B という比率を見ないため L1 を使い残す）．
    // 予算は scheme I と同じつまみ（OZ_L1_BUDGET_KB．既定 1408 KiB．Oz2Knobs::l1_budget）を使う．
    // BFP8_b は 1 要素 1.0625 B で FP8 の 1 B より広く，予算が足りないと bw が
    // 1 段落ちてスピル区間が増える（予算 1280 KiB での実例は共有ヘッダの注記）．
    r.g = choose_cfg(r.pMt, r.pNt, r.Kt, gd.y, gd.x, /*area_cap=*/4, /*forced_bw=*/0,
                     (uint32_t)tt::tile_size(OZ_FMT.in),
                     (uint32_t)tt::tile_size(OZ_FMT.out),
                     k.l1_budget);
    if (!r.g.ok) {
        // 取れない理由を choose_cfg の条件の順に示す：パネルの辺がコア格子の縦（gy）・横（gx）に 2 以上の約数を持つか，
        // 持つならその分割の出力ブロックが bw=1 でも L1 の予算に収まらない
        auto has_div = [](uint32_t n, uint32_t g) { for (uint32_t d = 2; d <= std::min(n, g); d++) if (n % d == 0) return true; return false; };
        const uint32_t gy = (uint32_t)gd.y, gx = (uint32_t)gd.x;
        char why[256];
        if (!has_div(r.pMt, gy)) snprintf(why, sizeof why, "パネルの行のタイル数 %u が 2..%u（コア格子の縦）に約数を持たない", r.pMt, gy);
        else if (!has_div(r.pNt, gx)) snprintf(why, sizeof why, "パネルの列のタイル数 %u が 2..%u（コア格子の横）に約数を持たない", r.pNt, gx);
        else snprintf(why, sizeof why, "コア格子に割った出力ブロックが bw=1 でも L1 の予算 %u KiB（OZ_L1_BUDGET_KB）に収まらない",
                      k.l1_budget / 1024u);
        fprintf(stderr, "ERROR: GEMM 段の構成が取れない（no gemm cfg）：出力のパネル %ux%u タイル（Mt=%u Nt=%u．%s）で，%s"
                        "（パネルの辺は Mt・Nt の 120 以下の最大の約数に固定して探さない．src/README.md「対応する形状」）\n",
                r.pMt, r.pNt, r.Mt, r.Nt, k.panel_force ? "OZ2_PANEL_FORCE で指定" : "自動", why);
        return false;
    }
    if (r.nbr > 1 || r.nbc > 1) printf("gemm panels: %ux%u (per-core %ux%u tiles)\n", r.nbr, r.nbc, r.g.pm, r.g.pn);
    printf("gemm cfg: grid=%ux%u per-core=%ux%u subblock=%ux%u bw=%u nb=%u panels=%ux%u\n",
           r.g.cr, r.g.cc, r.g.pm, r.g.pn, r.g.sh, r.g.sw, r.g.bw, r.g.nb, r.nbr, r.nbc);
    return true;
}

// パネルはデバイス上のオフセットで指す．A の行パネルは連続なので原点は
// br*pMt*Kt，B の列パネルは行ストライドを全体の Nt に保ったまま原点を bc*pNt へ
// ずらせばよい（GEMM リーダは A/B の行ストライドを引数で受け取る）．
// A の桁行列は帯 1 本ぶん（bMt タイル行）なので，平面の間隔は ntotAb，br は帯の中の行パネルの番号である（帯 1 本なら全体）
static inline uint32_t panel_off_A(const Oz2Plan& r, uint32_t plane, uint32_t br) { return plane * r.ntotAb() + br * r.pMt * r.Kt; }
static inline uint32_t panel_off_B(const Oz2Plan& r, uint32_t plane, uint32_t bc) { return plane * r.ntotB + bc * r.pNt; }
// 桁分割経路の A は，組の桁行列 PA 枚を K 方向に連結した Mt×(PA*Kt) の 1 枚の行列である
// （ozaki2/kernels/modslice_writer.cpp の LAYOUT_A_CAT）．桁行列 plane の列の先頭と，行パネル br の行の先頭．
// B は桁行列を K 方向に積んだ (PA*Kt)×Nt の行列で，位置 plane の行の先頭は panel_off_B と同じ式になる．
static inline uint32_t panel_off_Acat(const Oz2Plan& r, uint32_t PA, uint32_t plane, uint32_t br) {
    return br * r.pMt * (PA * r.Kt) + plane * r.Kt; }
