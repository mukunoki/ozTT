// SPDX-License-Identifier: Apache-2.0
#pragma once
// 計時の手順（cold・steady）と計時区間のテレメトリの行（host_eval.hpp の傘の下．尾崎の 2 つのドライバと基準実装が使う）
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <utility>
#include <vector>
#include "telemetry.hpp"   // 同じディレクトリ（コンテナの外で tools/ を作るときも見つかるように相対で書く）
#include "oz_env.hpp"      // つまみの読み方（同じく相対）

namespace oz_eval {

// ---- 計時 --------------------------------------------------------------------
// 計時の単位は「段」である（尾崎 I のスケール算出とスライス・GEMM と再構成，尾崎 II のスケール算出・スライス・
// GEMM・合成・再構成，素の GEMM など．呼び出し側が段の名前を付ける）．1 つの段が何回の計時（パネルごと・法ごと
// など）から成っていてもよく，段の時間はその和になる．論文が報告するのはどれも計時した iters 回の算術平均で，
// 最良値ではない．dispatch は投入するだけ，sync がバリアで，iters 回の投入は途中で同期せずパイプライン化されたまま
// 流れる（同期は最後の sync の 1 回だけ）．ただし投入の関数の中で同期する段（尾崎 I の単一 K ブロックの GEMM 段・
// fill・掃き出し，尾崎 II の scale）は，iters 回が毎回同期しながら流れる（時間はやはり iters 回の和を iters で割る）．
//
// 方式は 2 つある（OZ_TIMING．CONFIG 行の timing= に残す）．
//   cold（既定）  計時しない回を warmup 回流してから iters 回を測る．warmup はドライバが渡す（位置引数か既定値．
//                 尾崎の経路は 1）．カーネルの JIT と初回アクセスの一過性は吸収するが，クロックの一過性は吸収しない．
//   steady       段ごとに，チップのクロックが負荷に追いついた状態（定常状態）まで流してから iters 回を測る．
// なぜ steady が要るか：Blackhole（p150a の既定のファームウェア）は，ボードの入力電力の 1 s の移動平均をボードの電力の
// 上限（BOARD_POWER_LIMIT 300 W）に保つ制御（Doppler．telemetry.hpp の冒頭）でクロックの上限を決める．密な GEMM を流すと
// 入力電力が上限を超え，移動平均が追いつく約 1.5 s かけて AICLK が 1350 MHz から下がる．単独で流す GEMM 以外の段（スケール・
// スライス，尾崎 II の再構成，Ozaki-I-INT8 の再構成など）は上限に届かず 1350 MHz のままで，GEMM と交互に流す段（尾崎 I の
// 浮動小数点入力 7 経路の再構成，尾崎 II の合成）は GEMM の下がったクロックで流れる．cold では計時区間が一部または全部
// この下がる前の区間に入るので，値が計時の長さ（行列の大きさ・反復回数）に依存し，大きさを振ったときの伸びにクロックの差が
// 混ざる（パネルごとの計時が数 ms と短い素の GEMM で最も大きい）．両方式で測った値とクロックの下がり方は
// docs/remeasure_other_machine.md の 6.3 にある．
// 定常状態の定義（steady のとき warm_up が満たすまで流す条件）：
//   1. 段がその群（下）で初めて流れるときは，少なくとも OZ_WARMUP_SEC 秒・warmup 回流す（温め）．
//   2. そのうえで，直近の OZ_WARMUP_WIN_SEC 秒（以上．投入の単位の切れ目で区切る）の AICLK の平均が，その前の
//      OZ_WARMUP_WIN_SEC 秒以上の窓の平均と OZ_WARMUP_TOL_PCT % 未満しか違わなくなるまで流す（電力の上限の制御が
//      落ち着いた）．2 つの窓はそれぞれ独立に単位の切れ目まで延ばすので，長さは同じとは限らない（投入の単位を 1 回ずつに
//      切り替えた直後は，前の窓が切り替える前の長い単位になる．warm_up の windows）．
//      OZ_WARMUP_MAX_SEC 秒で打ち切る（打ち切ったら WARMUP・TELEM 行に capped が残り，検査が拒む）．打ち切りは 1 の
//      warmup 回を満たしていなくても効く（満たさずに打ち切った温めも capped になる）．
//   3. 同じ群の段が続けて流れている（前の計時の終わりから OZ_WARMUP_GAP_MS ミリ秒以内に次が始まる）あいだは，
//      2 回目以降の計時は 1 からやり直さず，warmup 回と，直前の間の OZ_WARMUP_HOT_FACTOR 倍の時間以上を流してから測る
//      （これも OZ_WARMUP_MAX_SEC 秒で打ち切り，TELEM 行の capped に残す．WARMUP 行は出さない）．
//      尾崎 II の GEMM 段は法と桁積とパネルごとに数百回の計時から成り，あいだにプログラムの組み立て（数 ms）が挟まる．
//      warmup 回だけで次へ進むとデバイスの動いている割合が下がり，電力の平均が下がるぶんクロックが上がる．直前の間の
//      k 倍を流せば割合は 1/(1+1/k) 以上（既定の 20 倍で 95% 以上）に保たれ，初めの温めで落ち着いたクロックが段の終わりまで
//      続く．プログラムを組み終えてから続けて流す実際の使い方に近いのはこちらで，尾崎 I の掃き出し（計時の中で組み立てない）
//      とも同じ条件になる．段の中で負荷が変わる（法によって連結の本数 g や桁数が違い，電力が違う）とクロックも約 1 s で
//      それに追いつくので，変わり目の直後の計時は直前の 1 s から数 % から十数 % ずれる（TELEM の drift_max_pct．検査はこれを
//      見ない．print_telem の注）．
//      間が OZ_WARMUP_GAP_MS より長ければ（ホストでのパネルの切り出しと転送，別の段），2 だけを満たすまで流し直す（温め直し）．
//   群は既定で段と同じで，細かく交互に流れる段（尾崎 II の GEMM と合成）は呼び出し側が同じ群にする．
// 「定常状態」はクロックの制御についての定常であって，温度の定常ではない．GEMM を流し続けると温度は数十秒の
// 時定数で上がり続け，リーク電流が増えるぶん同じ電力の上限で回せるクロックもゆっくり下がる．窓の差の条件はこのゆっくり
// した低下（1 s あたり 0.2% 程度）を定常とみなす．温め始めの温度は測定スクリプトが揃える（60 degC 以下まで冷ましてから
// 始める．scripts/measure_lib.sh）．
// どちらの方式でも，テレメトリ（telemetry.hpp）が読めれば計時区間のクロック・電流・電力・温度と，ボードの入力電力・
// クロックを絞っている仕組み（ファームウェアのテレメトリ表）を段ごとに集め，print_telem が TELEM 行に出す．steady は
// クロックで 2 を判定するので，テレメトリ（sysfs）が読めなければ始めない（timing_check）．ファームウェアの表が読めない
// ときは TELEM の入力電力と仕組みの欄を出さずに測る（FWTELEM 行に理由が残る）．
//
// 環境変数（timing_check がデバイスを開く前に 1 度検査する）
//   OZ_TIMING=cold|steady     方式（既定 cold．測定スクリプトは steady を渡す．scripts/measure_lib.sh）
//   OZ_WARMUP_SEC=s           1 の秒数（既定 kWarmupSecDefault）
//   OZ_WARMUP_MAX_SEC=s       2 を打ち切る秒数（既定 kWarmupMaxSecDefault．OZ_WARMUP_SEC 以上）
//   OZ_WARMUP_WIN_SEC=s       2 の窓の長さ（既定 1）
//   OZ_WARMUP_TOL_PCT=p       2 の許容差［%］（既定 1．0 は 2 を判定しない：初めは 1 だけ，温め直しは窓 2 つぶんの秒数だけ流す）
//   OZ_WARMUP_GAP_MS=m        3 の間隔［ms］（既定 50．クロックの制御の応答（約 1 s）より十分短く取る：
//                             GEMM を止めてから 100 ms でクロックは約 0.5% 戻る）
//   OZ_WARMUP_HOT_FACTOR=k    3 で続けて測る回を流す時間の，直前の間に対する倍率（既定 20．0 は warmup 回だけ）
//   OZ_TELEM_TRACE=1          温めのあいだ窓ごとの AICLK・電流・電力・温度（と入力電力・仕組み）を WARMTRACE 行に出す（既定の決め方を
//                             調べるための診断．計時の値は変えない）
//   OZ_TELEM_MS・OZ_TELEM_DEV・OZ_TELEM_FW  テレメトリの採取の間隔・読むカード・ファームウェアの表を読むか（telemetry.hpp）
// 温めの秒数の既定（1）は，速い変化（GEMM を流し始めてから 2〜3 s でクロックが落ち着く）の 3 倍以上を取る．
// 打ち切りの既定（2）は，1 回が窓より長い段でも窓を数個比べられる長さにする：14080^3 の Ozaki-I-BFP2 の掃き出しは
// 1 回が 25 s を超え，1 回目は流し始めの 1350 MHz を含むので 2 回目と 3 回目を比べて落ち着く．温度の上がり方で
// 4 回目が要っても打ち切らないよう 180 s に取る（1 回が短い段は 10〜15 s で 2 を満たす）．
// scripts/remeasure_all.sh が測る rev のこの 2 行を名前で読んで表示するので，"constexpr double <名前> = <数>;" の書き方を保つ
constexpr double kWarmupSecDefault = 10.0;
constexpr double kWarmupMaxSecDefault = 180.0;

struct TimingCfg {
    bool steady = false;
    double warm_sec = kWarmupSecDefault, warm_max_sec = kWarmupMaxSecDefault;
    double win_sec = 1.0, tol_pct = 1.0, gap_sec = 0.05, hot_factor = 20.0;
    bool trace = false;
};

inline const TimingCfg& timing_cfg() {
    static const TimingCfg cfg = [] {
        TimingCfg c;
        c.steady = oz_env::choice("OZ_TIMING", {"cold", "steady"}, 0) == 1;
        c.warm_sec = oz_env::f64("OZ_WARMUP_SEC", c.warm_sec, 0.0, 3600.0);
        if (oz_env::set("OZ_WARMUP_MAX_SEC")) c.warm_max_sec = oz_env::f64("OZ_WARMUP_MAX_SEC", c.warm_max_sec, 0.0, 36000.0);
        else c.warm_max_sec = std::max(c.warm_max_sec, c.warm_sec);
        if (c.warm_max_sec < c.warm_sec) {
            std::fprintf(stderr, "ERROR: OZ_WARMUP_MAX_SEC=%g が OZ_WARMUP_SEC=%g より短い\n", c.warm_max_sec, c.warm_sec);
            std::exit(2);
        }
        c.win_sec = oz_env::f64("OZ_WARMUP_WIN_SEC", c.win_sec, 0.1, 600.0);
        c.tol_pct = oz_env::f64("OZ_WARMUP_TOL_PCT", c.tol_pct, 0.0, 100.0);
        if (oz_env::set("OZ_WARMUP_GAP_MS")) c.gap_sec = oz_env::f64("OZ_WARMUP_GAP_MS", 0.0, 0.0, 600000.0) / 1e3;
        c.hot_factor = oz_env::f64("OZ_WARMUP_HOT_FACTOR", c.hot_factor, 0.0, 1000.0);
        c.trace = oz_env::u32("OZ_TELEM_TRACE", 0u, 0u, 1u) != 0;
        return c;
    }();
    return cfg;
}

// 計時のつまみとテレメトリを，デバイスを開く前に 1 度だけ検査する（oz_driver.hpp の oz_drv::check_env が，計時する
// ドライバ（つまみの表の W_TIMED：尾崎の 2 つと基準実装）について呼ぶ）．
// steady はクロックで定常状態を判定するので，テレメトリが読めなければここで止める（exit 2）
inline void timing_check() {
    const TimingCfg& c = timing_cfg();
    const oz_telem::Sampler& t = oz_telem::Sampler::instance();
    if (c.steady && !t.available()) {
        std::fprintf(stderr, "ERROR: OZ_TIMING=steady はテレメトリ（AICLK）で定常状態を判定するが，読めない：%s"
                             "（OZ_TELEM_DEV でカードの番号を与えるか，OZ_TIMING=cold で測る）\n", t.why().c_str());
        std::exit(2);
    }
}

// TELEM・WARMUP・WARMTRACE 行の末尾に足すファームウェアの欄（print_telem の説明．区間の要約 s にファームウェアの値が
// 読めた標本が無ければ空）
inline std::string telem_fw_fields(const oz_telem::Stats& s) {
    if (!s.fw_ok) return "";
    char b[512], bpl[48];
    if (s.bpl_min == s.bpl_max) std::snprintf(bpl, sizeof bpl, "%.0f", s.bpl_min);
    else std::snprintf(bpl, sizeof bpl, "%.0f-%.0f", s.bpl_min, s.bpl_max);
    std::snprintf(b, sizeof b, ",input_power_mean=%.1f,input_power_max=%.0f,board_power_limit=%s,arb=%s,arb_top=%s,"
                  "arb_freq_mean=%.1f,arb_freq_min=%.0f,enabled_max_arb=%s",
                  s.inpow_mean, s.inpow_max, bpl, s.arb_str().c_str(), oz_telem::arb_name(s.arb_top()), s.arbf_mean,
                  s.arbf_min, s.en_arb_str().c_str());
    return b;
}
namespace timing_detail {
// 段ごとの記録．iv は計時区間（oz_telem::now() の秒），iv_ref はその区間の基準の AICLK（無ければ負）：
//   温め（初めの温め・温め直し）の直後の計時は，その温めの最後の窓の平均．
//   温めずに続けて測った計時（上の 3）は，その区間の直前 OZ_WARMUP_WIN_SEC 秒のうちの同じ群の動き（温めと計時の
//   区間）の平均．続けて流れるあいだの計時区間を，その直前の同じ負荷と比べる（温めの判定と同じ窓の長さ）．
// TELEM の ref_aiclk は iv_ref の最初の正の値（段の初めの温めの最後の窓．print_telem）
struct StageRec {
    std::string name, group;
    bool sub = false;                  // 別の段の計時区間の一部だけを記録した段（note_interval）
    uint64_t calls = 0, iters = 0, warm_iters = 0;
    double timed_s = 0.0, warm_s = 0.0;
    uint32_t warmups = 0, rewarms = 0, hot = 0, capped = 0;
    std::vector<std::pair<double, double>> iv;
    std::vector<double> iv_ref;
};
// 群ごとの状態．last_end は最後に計時か温めを終えた時刻，call_ref はいま計時しようとしている回の基準
// （直前の温めの最後の窓．温めなかった回は負），spans は群の動き（温めと計時の区間）の記録
struct GroupRec {
    std::string name;
    bool warmed = false;
    double last_end = -1.0, call_ref = -1.0;
    std::vector<std::pair<double, double>> spans;
};
inline std::deque<StageRec>& stages() { static std::deque<StageRec> v; return v; }
inline std::deque<GroupRec>& groups() { static std::deque<GroupRec> v; return v; }
inline StageRec& stage_rec(const char* name, const char* group, bool sub = false) {
    for (auto& s : stages())
        if (s.name == name) return s;
    StageRec r;
    r.name = name;
    r.group = group ? group : name;
    r.sub = sub;
    stages().push_back(r);
    return stages().back();
}
inline GroupRec& group_rec(const std::string& g) {
    for (auto& x : groups())
        if (x.name == g) return x;
    GroupRec r;
    r.name = g;
    groups().push_back(r);
    return groups().back();
}
}  // namespace timing_detail

namespace timing_detail {
// 温めの 2 つの窓（冒頭の定常状態の定義の 2）．starts は投入の単位の始まりの時刻（この温めの中だけ），t は今．
// 直近の窓 [m,e] は t から win 以上さかのぼった最後の単位の始まりから t まで，その前の窓 [a,m] はそこからさらに
// win 以上さかのぼった最後の単位の始まりから．どちらかが取れなければ false
inline bool warm_windows(const std::vector<double>& starts, double t, double win, double& a, double& m, double& e) {
    auto it = std::upper_bound(starts.begin(), starts.end(), t - win);
    if (it == starts.begin()) return false;
    m = *(it - 1);
    auto it2 = std::upper_bound(starts.begin(), starts.end(), m - win);
    if (it2 == starts.begin()) return false;
    a = *(it2 - 1);
    e = t;
    return true;
}

// 続けて測る回の温め（cold の温めと，steady で同じ群が続けて流れているとき．冒頭の定常状態の定義の 3）．
// cold は warmup 回投入して 1 回同期する．steady は warmup 回を投入の単位 b 回ずつ同期しながら流し，直前の間 gap の
// OZ_WARMUP_HOT_FACTOR 倍の時間以上流す．どちらも OZ_WARMUP_MAX_SEC で打ち切る
template <class F, class S>
inline void warm_continue(const char* stage, StageRec& st, GroupRec& g, double t0, double gap, uint32_t warmup,
                          uint32_t batch, F& dispatch, S& sync) {
    const TimingCfg& c = timing_cfg();
    const oz_telem::Sampler& tel = oz_telem::Sampler::instance();
    double t1 = t0;
    uint64_t n = 0;
    bool capped = false;
    const uint32_t b = std::max<uint32_t>(1, batch);
    if (!c.steady) {
        for (uint32_t w = 0; w < warmup; ++w) dispatch();
        sync();
        t1 = oz_telem::now();
        n = warmup;
    } else {
        // steady で続けて測る回も OZ_WARMUP_MAX_SEC で打ち切る（初めの温め・温め直しと同じく capped に残す）．
        // warmup 回は投入の単位 b 回ずつ同期しながら流し，単位ごとに上限を見る（1 回が異常に遅いと warmup 回に
        // 届く前に上限を過ぎる）．測定スクリプトの warmup は 1 回で単位 1 つに収まり，続けて流す時間（直前の間の
        // OZ_WARMUP_HOT_FACTOR 倍．既定 50 ms の 20 倍で 1 s 以下）も上限の 180 s に遠いので，普段の構成では投入の
        // 回数も同期の位置も変わらない
        while (n < warmup) {
            const uint32_t k = (uint32_t)std::min<uint64_t>(b, warmup - n);
            for (uint32_t i = 0; i < k; ++i) dispatch();
            sync();
            n += k;
            t1 = oz_telem::now();
            if (n < warmup && t1 - t0 >= c.warm_max_sec) { capped = true; break; }
        }
        if (warmup == 0) { sync(); t1 = oz_telem::now(); }
        // 直前の間（ホストでのプログラムの組み立てなど）の OZ_WARMUP_HOT_FACTOR 倍の時間以上流してから測る
        // （デバイスが動いている割合を 1/(1+1/倍率) 以上に保ち，初めの温めで落ち着いたクロックを保つ．冒頭の 3）
        if (c.hot_factor > 0)
            while (!capped && t1 - t0 < c.hot_factor * gap) {
                if (t1 - t0 >= c.warm_max_sec) { capped = true; break; }
                for (uint32_t i = 0; i < b; ++i) dispatch();
                sync();
                n += b;
                t1 = oz_telem::now();
            }
        if (capped) {
            ++st.capped;
            std::fprintf(stderr, "WARN: 段 %s の続けて測る回の温めを OZ_WARMUP_MAX_SEC=%g s で打ち切った（%llu 回．"
                                 "TELEM の capped に残る）\n", stage, c.warm_max_sec, (unsigned long long)n);
        }
    }
    st.warm_s += t1 - t0;
    st.warm_iters += n;
    if (c.steady) ++st.hot;
    // cold の基準は温めの回の平均（参考．検査は steady のログにだけ使う）．steady で続けて測る回の基準は
    // end_timed が直前の窓から取る
    g.call_ref = -1.0;
    if (!c.steady && warmup > 0 && tel.available()) g.call_ref = tel.aiclk_mean(t0, t1);
    g.spans.push_back({t0, t1});
    g.warmed = true;
    g.last_end = t1;
}

// 初めの温めと温め直しの結果（t は終えた時刻，n は流した回数，b は最後の投入の単位，m_prev・m_last は最後に比べた
// 2 つの窓の AICLK の平均．取れなければ負）
struct WarmResult {
    double t = 0.0;
    uint64_t n = 0;
    uint32_t b = 1;
    bool stable = false, capped = false;
    double m_prev = -1.0, m_last = -1.0;
};
// 初めの温め（full）と温め直し（冒頭の定常状態の定義の 1・2）：投入の単位（batch 回の投入と 1 回の同期）を繰り返し，
// 最低の回数と秒数を満たした後，50 ms ごとに 2 つの窓の AICLK の平均を比べて OZ_WARMUP_TOL_PCT % 未満の差になれば終える．
// 窓の平均は欠測（telemetry.hpp の冒頭）のある窓では負になるので，欠測のある窓どうしでは定常と認めない．
// OZ_WARMUP_MAX_SEC で打ち切る
template <class F, class S>
inline WarmResult warm_until_steady(const char* stage, bool full, double t0, uint32_t warmup, uint32_t batch,
                                    F& dispatch, S& sync) {
    const TimingCfg& c = timing_cfg();
    const oz_telem::Sampler& tel = oz_telem::Sampler::instance();
    const double min_sec = full ? c.warm_sec : (c.tol_pct > 0 ? 0.0 : 2.0 * c.win_sec);
    WarmResult r;
    // 投入の単位．1 回が窓より長い段（14080^3 の Ozaki-I-BFP2 の全工程計時など）で iters 回をまとめると窓が
    // その倍になり，窓 2 つぶんが打ち切りの秒数を超えるので，1 回目の単位を見て 1 回ずつに切り替える
    r.b = std::max<uint32_t>(1, batch);
    std::vector<double> starts;                       // 投入の単位の始まりの時刻
    double t = t0, last_eval = -1.0, next_trace = c.win_sec;
    while (true) {
        starts.push_back(t);
        for (uint32_t i = 0; i < r.b; ++i) dispatch();
        sync();
        r.n += r.b;
        const double tb = t;
        t = oz_telem::now();
        if (r.b > 1 && (t - tb) / r.b > c.win_sec) r.b = 1;
        const double el = t - t0;
        if (c.trace) {
            while (el >= next_trace) {
                const oz_telem::Stats w = tel.stats(t0 + next_trace - c.win_sec, t0 + next_trace);
                std::printf("WARMTRACE,stage=%s,t=%.2f,iters=%llu,aiclk=%.1f,curr=%.1f,power=%.1f,temp=%.2f%s\n",
                            stage, next_trace, (unsigned long long)r.n, w.aiclk_mean, w.curr_mean, w.power_mean,
                            w.temp_max, telem_fw_fields(w).c_str());
                next_trace += c.win_sec;
            }
        }
        const bool enough = r.n >= warmup && el >= min_sec;
        if (enough && c.tol_pct <= 0) {
            double a, m, e;
            if (warm_windows(starts, t, c.win_sec, a, m, e)) r.m_last = tel.aiclk_mean(m, e), r.m_prev = tel.aiclk_mean(a, m);
            else r.m_last = tel.aiclk_mean(std::max(t0, t - c.win_sec), t);
            break;
        }
        if (enough && (t - last_eval >= 0.05 || el >= c.warm_max_sec)) {
            last_eval = t;
            double a, m, e;
            if (warm_windows(starts, t, c.win_sec, a, m, e)) {
                r.m_last = tel.aiclk_mean(m, e);
                r.m_prev = tel.aiclk_mean(a, m);
                if (r.m_last > 0 && r.m_prev > 0 && 100.0 * std::fabs(r.m_last / r.m_prev - 1.0) < c.tol_pct) {
                    r.stable = true;
                    break;
                }
            }
        }
        // 打ち切りは最低回数（warmup 回）を満たしたかに依らない：1 回が異常に遅いと warmup 回に届く前に上限を過ぎ，
        // n >= warmup を待つと上限が効かない．測定スクリプトの warmup は 1 回で，投入の単位の 1 つ目で満たすので，
        // 普段の構成では判定の順序も回数も変わらない（上の安定の判定が先に通れば，上限の回でも安定として終える）
        if (el >= c.warm_max_sec) {
            r.capped = true;
            if (r.m_last <= 0) r.m_last = tel.aiclk_mean(std::max(t0, t - c.win_sec), t);
            break;
        }
    }
    r.t = t;
    return r;
}
}  // namespace timing_detail

// 段 stage（群 group．nullptr なら段と同じ）の温め．dispatch は 1 回ぶんの投入，sync はバリア．
// cold と，steady で同じ群が続けて流れているときは timing_detail::warm_continue（冒頭の 3），steady の初めの温めと
// 温め直しは warm_until_steady（冒頭の 1・2）で流す．投入の単位は batch 回の投入と 1 回の同期（計時と同じ流し方に
// するため，呼び出し側は iters を渡す）．窓の切れ目はこの単位の切れ目で，1 単位が窓より長い段では窓は単位 1 つになる．
// 初めの温めと温め直しの後に WARMUP 行を 1 行出す（欄は下の printf．末尾の miss・stale_ms は最後の窓の欠測で，
// telemetry.hpp の冒頭）
template <class F, class S>
inline void warm_up(const char* stage, const char* group, uint32_t warmup, uint32_t batch, F&& dispatch, S&& sync) {
    using namespace timing_detail;
    const TimingCfg& c = timing_cfg();
    oz_telem::Sampler& tel = oz_telem::Sampler::instance();
    tel.start();
    StageRec& st = stage_rec(stage, group);
    GroupRec& g = group_rec(st.group);
    const double t0 = oz_telem::now();
    const double gap = g.warmed ? t0 - g.last_end : -1.0;
    if (!c.steady || (g.warmed && gap <= c.gap_sec)) {
        warm_continue(stage, st, g, t0, gap, warmup, batch, dispatch, sync);
        return;
    }
    const bool full = !g.warmed;
    const WarmResult r = warm_until_steady(stage, full, t0, warmup, batch, dispatch, sync);
    st.warm_s += r.t - t0;
    st.warm_iters += r.n;
    if (full) ++st.warmups; else ++st.rewarms;
    if (r.capped) ++st.capped;
    g.call_ref = r.m_last;
    g.spans.push_back({t0, r.t});
    g.warmed = true;
    g.last_end = r.t;
    const oz_telem::Stats w = tel.stats(std::max(t0, r.t - c.win_sec), r.t);
    std::printf("WARMUP,stage=%s,group=%s,kind=%s,gap_ms=%s,sec=%.3f,iters=%llu,batch=%u,stable=%s,capped=%d,"
                "win_prev=%.1f,win_last=%.1f,curr=%.1f,power=%.1f,temp=%.2f%s,miss=%llu,stale_ms=%.3g\n",
                stage, st.group.c_str(), full ? "full" : "rewarm",
                gap < 0 ? "na" : std::to_string((long long)std::llround(gap * 1e3)).c_str(), r.t - t0,
                (unsigned long long)r.n, r.b, c.tol_pct <= 0 ? "na" : (r.stable ? "1" : "0"), r.capped ? 1 : 0,
                r.m_prev, r.m_last, w.curr_mean, w.power_mean, w.temp_max, telem_fw_fields(w).c_str(),
                (unsigned long long)w.miss, w.stale_s * 1e3);
}

// 計時区間の印．begin_timed で始め，end_timed で終える（その間の iters 回が段の計時になる）
struct TimedMark { std::string stage; double t0; };
// 区間の端は oz_telem::now() の時刻で，begin_timed は準備（採取の開始・段の記録）を済ませてから最後に取る．
// end_timed_at は終わりの時刻を呼び出し側から受ける（time_dispatch が性能の値と同じ時刻の組を TELEM の区間にするため）
inline TimedMark begin_timed(const char* stage, const char* group = nullptr) {
    oz_telem::Sampler::instance().start();
    (void)timing_detail::stage_rec(stage, group);
    return {stage, oz_telem::now()};
}
inline void end_timed_at(const TimedMark& m, uint32_t iters, double t1) {
    using namespace timing_detail;
    StageRec& st = stage_rec(m.stage.c_str(), nullptr);
    GroupRec& g = group_rec(st.group);
    double ref = g.call_ref;
    if (ref <= 0 && timing_cfg().steady) {
        // 温めずに続けて測った回：直前の窓 [t0-W, t0] に入る同じ群の動きの平均を基準にする
        const double w0 = m.t0 - timing_cfg().win_sec;
        std::vector<std::pair<double, double>> cl;
        for (auto it = g.spans.rbegin(); it != g.spans.rend() && it->second > w0; ++it)
            if (std::min(it->second, m.t0) > std::max(it->first, w0))
                cl.push_back({std::max(it->first, w0), std::min(it->second, m.t0)});
        std::reverse(cl.begin(), cl.end());
        const oz_telem::Sampler& tel = oz_telem::Sampler::instance();
        if (!cl.empty() && tel.available()) {
            const oz_telem::Stats s = tel.stats(cl);
            if (s.fresh()) ref = s.aiclk_mean;   // 欠測のある窓は基準にしない（telemetry.hpp の冒頭．その区間は drift に入らない）
        }
    }
    g.call_ref = -1.0;
    st.iv.push_back({m.t0, t1});
    st.iv_ref.push_back(ref);
    ++st.calls;
    st.iters += iters;
    st.timed_s += t1 - m.t0;
    g.spans.push_back({m.t0, t1});
    g.last_end = t1;
}
inline void end_timed(const TimedMark& m, uint32_t iters) { end_timed_at(m, iters, oz_telem::now()); }
// 別の段の計時区間の一部 [t0,t1]（oz_telem::now() の秒）を，段 stage（群 group の一部）の区間として記録する．
// 尾崎 I の掃き出しは 1 回の中で GEMM と再構成を交互に流すので，掃き出しの計時の中の GEMM・再構成の部分を
// それぞれの段のクロックとして出すのに使う（温めは掃き出しの段がまとめて行う）
inline void note_interval(const char* stage, const char* group, double t0, double t1) {
    using namespace timing_detail;
    StageRec& st = stage_rec(stage, group, true);
    st.iv.push_back({t0, t1});
    st.iv_ref.push_back(-1.0);          // 掃き出しの一部なので段としての基準は持たない（TELEM の sub=1）
    ++st.calls;
    st.timed_s += t1 - t0;
}

// 段 stage の温めと計時をまとめて行い，1 反復あたりの秒数を返す（温めは warm_up，計時は iters 回の投入と 1 回の同期）
template <class F, class S>
inline double time_dispatch(const char* stage, uint32_t warmup, uint32_t iters, F&& dispatch, S&& sync,
                            const char* group = nullptr) {
    // iters の 0 は入口で拒む（尾崎の 2 つは oz_driver.hpp の read_counts，基準実装は check_iters）．ここは 0 除算の防ぎ
    const uint32_t it = std::max<uint32_t>(1, iters);
    warm_up(stage, group, warmup, it, dispatch, sync);
    // 性能の値と TELEM の計時区間（AICLK を平均する区間）は同じ時刻の組 [m.t0, t1] から作る（別々に時計を読むと，
    // 2 つの読みの間にホストのスレッドが止まったぶん区間がずれる）．時計は単調な oz_telem::now()（steady_clock）
    const TimedMark m = begin_timed(stage, group);
    for (uint32_t i = 0; i < it; ++i) dispatch();
    sync();
    const double t1 = oz_telem::now();
    end_timed_at(m, it, t1);
    return (t1 - m.t0) / it;
}
template <class F>
inline double time_dispatch(const char* stage, uint32_t warmup, uint32_t iters, F&& dispatch) {
    return time_dispatch(stage, warmup, iters, dispatch, [] {});
}

// 段ごとの計時区間のテレメトリを 1 行ずつ出す（計時した段の初出の順．CONFIG 行の前に 1 度だけ呼ぶ）．
//   TELEM,stage=<段>,group=<群>,timing=<方式>,sub=<0|1>,calls=,iters=,timed_s=,warm_s=,warm_iters=,warmups=,rewarms=,hot=,
//         capped=,ref_aiclk=,drift_pct=,drift_max_pct=,seq_pct=,n=,aiclk_mean=,aiclk_min=,aiclk_max=,curr_mean=,curr_max=,
//         power_mean=,power_max=,vcore_mean=,temp_start=,temp_max=,telem_dev=
// timed_s は計時区間の長さの和［s］，warm_* は温めの秒数と回数（warmups は初めの温め，rewarms は温め直し，hot は
// 温めずに続けて測った計時の回数），n は標本と計時区間の重なりの数（区間ごとに数えるので，1 本の標本が 2 つの計時区間に
// 重なれば 2 と数える．採った標本の本数ではない．telemetry.hpp の Stats），aiclk_* は計時区間の AICLK［MHz］（区間との
// 重なりで重み付けした平均と最小・最大），curr［A］・power［W］・vcore［mV］・temp［degC］も同じ．
// ref_aiclk は最初の計時の基準（段の初めの温めの最後の窓の AICLK の平均）．drift_pct・drift_max_pct は各計時区間の
// AICLK の平均とその区間の基準（StageRec の iv_ref）との差［%］の重み付き平均と絶対値の最大で，計時区間が定常状態に
// あったかの目安（検査は drift_pct の絶対値だけを見る：scripts/lib/oz_logs.py の timing_reasons で STEADY_DRIFT_PCT（2%）以下．
// drift_max_pct は記録だけで，法ごとに電力の違うプログラムから成る尾崎 II の GEMM 段では 10% を超えることがある）．
// seq_pct は段全体の平均と ref_aiclk の差［%］で，何百回の計時から
// 成る段（尾崎 II の GEMM 段など）が初めの温め（1 回目の計時を詰めて繰り返した負荷）と違うクロックで続いたかを見る
// 参考値（計時のあいだにプログラムの組み立てが挟まるぶん電力の平均が下がり，クロックは高めに落ち着く）．
// sub=1 の段は掃き出しの一部で基準を持たないので na．テレメトリが読めなければ n=0 と telem_dev=none で，クロックの欄は
// 出さない．OZ_TELEM_TRACE=1 のときは計時区間ごとに TELEMIV 行（段・番号・始まりの時刻・長さ・AICLK・基準）も出す．
// 行の最後に計時区間の欠測（telemetry.hpp の冒頭の「欠測」）を足す：
//   ,miss=<sysfs が読めずに捨てた回数>,stale_ms=<保持している標本が採取の間隔の 5 倍より古かった時間［ms］>
// どちらも 0 でなければ，その段の計時区間のクロックの一部が古い標本か欠けた標本で，欠測のある区間は drift の計算と
// 温めの判定から外してある（検査は 0 でない段を拒む．欄の無い行はこの欄より前のログ）．
// ファームウェアのテレメトリ表が読めれば（telemetry.hpp の「読むもの 2」），TELEM 行の欠測の欄の前に次の欄を足す（既存の欄は
// 位置も変えない．欄の無い行は表が読めなかったか，この欄より前のログ）：
//   ,input_power_mean=,input_power_max=,board_power_limit=,arb=<番号>:<割合>|…,arb_top=<名前>,arb_freq_mean=,
//    arb_freq_min=,enabled_max_arb=0x…
// input_power_* はボードの入力電力［W］（INPUT_POWER）の平均と最大，board_power_limit はボードの電力の上限［W］（区間で
// 変わったら最小-最大），arb は AICLK を fmax より下へ絞っていた仕組み（AICLK_ARB_MAX の上位 16 bit．周波数が fmax 以上の
// 標本は 0＝fmax）ごとの時間の割合（大きい順），arb_top はその最大の仕組みの名前（oz_telem::arb_name），arb_freq_* は
// その仕組みの周波数［MHz］の平均と最小，enabled_max_arb は有効な仕組みのビットの並び（区間で変われば | で並べる）．
// 段の行の前に 1 度，ファームウェアの表の読み方と静的な値を FWTELEM 行に出す：
//   FWTELEM,reader=umd,dev=,dev_src=,fw_bundle=,update_ms=,aiclk_limit_max=,tdc_limit_max=,tdp_limit_max=,board_power_limit=,
//           enabled_max_arb=,open_ms=,samples=,errors=,read_us_mean=,read_us_max=[,why=]
//   （読めなければ FWTELEM,reader=none,dev=,dev_src=,why=<理由>）．dev_src は読んだカードの番号 dev をどこから決めたか
//   （OZ_TELEM_DEV・TT_DEVICE・sole_device＝/dev/tenstorrent/ の唯一のデバイス・default＝0．telemetry.hpp の OZ_TELEM_DEV），
//   board_power_limit は最後に読めた標本の値（無ければ開いたときの値），read_us_* は 1 標本（4 タグ）の読み出しの時間，
//   errors は例外で読めなかった回数（1 回でも出たらそれ以降は読まない）
inline void print_fwtelem() {
    const oz_telem::Sampler& tel = oz_telem::Sampler::instance();
    const oz_telem::FwInfo& f = tel.fw_info();
    const oz_telem::Sampler::FwCost fc = tel.fw_cost();
    if (!tel.fw_available() && fc.n == 0) {
        std::printf("FWTELEM,reader=none,dev=%d,dev_src=%s,why=%s\n", tel.dev(), tel.dev_source(),
                    oz_telem::field_safe(tel.fw_why()).c_str());
        return;
    }
    std::printf("FWTELEM,reader=umd,dev=%d,dev_src=%s,fw_bundle=%u.%u.%u.%u,update_ms=%u,aiclk_limit_max=%u,tdc_limit_max=%u,"
                "tdp_limit_max=%u,board_power_limit=%.0f,enabled_max_arb=0x%x,open_ms=%.1f,samples=%llu,errors=%llu,"
                "read_us_mean=%.1f,read_us_max=%.1f",
                tel.dev(), tel.dev_source(), f.bundle >> 24, (f.bundle >> 16) & 0xffu, (f.bundle >> 8) & 0xffu, f.bundle & 0xffu, f.update_ms,
                f.aiclk_limit_max, f.tdc_limit_max, f.tdp_limit_max,
                fc.last_bpl >= 0 ? fc.last_bpl : (double)f.board_power_limit, f.enabled_max_arb, f.open_ms,
                (unsigned long long)fc.n, (unsigned long long)fc.errors, fc.us_mean, fc.us_max);
    if (fc.errors) std::printf(",why=%s", oz_telem::field_safe(tel.fw_why()).c_str());
    std::printf("\n");
}
inline void print_telem() {
    using namespace timing_detail;
    const TimingCfg& c = timing_cfg();
    const oz_telem::Sampler& tel = oz_telem::Sampler::instance();
    if (tel.available()) print_fwtelem();
    for (const StageRec& st : stages()) {
        std::printf("TELEM,stage=%s,group=%s,timing=%s,sub=%d,calls=%llu,iters=%llu,timed_s=%.4f,warm_s=%.3f,"
                    "warm_iters=%llu,warmups=%u,rewarms=%u,hot=%u,capped=%u",
                    st.name.c_str(), st.group.c_str(), c.steady ? "steady" : "cold", st.sub ? 1 : 0,
                    (unsigned long long)st.calls, (unsigned long long)st.iters, st.timed_s, st.warm_s,
                    (unsigned long long)st.warm_iters, st.warmups, st.rewarms, st.hot, st.capped);
        const oz_telem::Stats s = tel.available() ? tel.stats(st.iv) : oz_telem::Stats{};
        if (!s.ok) {
            std::printf(",n=0,telem_dev=%s\n", tel.available() ? std::to_string(tel.dev()).c_str() : "none");
            continue;
        }
        double wsum = 0, dsum = 0, dmax = 0, ref0 = -1.0;
        bool have = false;
        if (!st.sub)
            for (size_t k = 0; k < st.iv.size(); ++k) {
                if (st.iv_ref[k] <= 0) continue;
                const double m = tel.aiclk_mean(st.iv[k].first, st.iv[k].second);
                if (m <= 0) continue;
                const double d = 100.0 * (m / st.iv_ref[k] - 1.0);
                const double w = std::max(st.iv[k].second - st.iv[k].first, 1e-9);
                wsum += w; dsum += w * d; dmax = std::max(dmax, std::fabs(d));
                if (!have) ref0 = st.iv_ref[k];
                have = true;
            }
        char ref[32] = "na", dm[32] = "na", dx[32] = "na", sq[32] = "na";
        if (have) {
            std::snprintf(ref, sizeof ref, "%.1f", ref0);
            std::snprintf(dm, sizeof dm, "%.2f", dsum / wsum);
            std::snprintf(dx, sizeof dx, "%.2f", dmax);
            std::snprintf(sq, sizeof sq, "%.2f", 100.0 * (s.aiclk_mean / ref0 - 1.0));
        }
        std::printf(",ref_aiclk=%s,drift_pct=%s,drift_max_pct=%s,seq_pct=%s,n=%llu,aiclk_mean=%.1f,aiclk_min=%.0f,"
                    "aiclk_max=%.0f,curr_mean=%.1f,curr_max=%.0f,power_mean=%.1f,power_max=%.0f,vcore_mean=%.0f,"
                    "temp_start=%.2f,temp_max=%.2f,telem_dev=%d%s,miss=%llu,stale_ms=%.3g\n",
                    ref, dm, dx, sq, (unsigned long long)s.n, s.aiclk_mean, s.aiclk_min, s.aiclk_max, s.curr_mean,
                    s.curr_max, s.power_mean, s.power_max, s.vcore_mean, s.temp_start, s.temp_max, tel.dev(),
                    telem_fw_fields(s).c_str(), (unsigned long long)s.miss, s.stale_s * 1e3);
        if (c.trace)
            for (size_t k = 0; k < st.iv.size(); ++k)
                std::printf("TELEMIV,stage=%s,k=%zu,t=%.4f,dur_ms=%.3f,aiclk=%.1f,ref=%.1f\n", st.name.c_str(), k,
                            st.iv[k].first, 1e3 * (st.iv[k].second - st.iv[k].first),
                            tel.aiclk_mean(st.iv[k].first, st.iv[k].second), st.iv_ref[k]);
    }
    std::fflush(stdout);
}

}  // namespace oz_eval
