// SPDX-License-Identifier: Apache-2.0
#pragma once
// ホスト側のテレメトリの採取（AICLK・コアの電流・電力・コア電圧・ASIC の温度，ボードの入力電力と周波数を絞っている仕組み）
// ---------------------------------------------------------------------------
// Blackhole p150a の既定のファームウェア（fw_table の doppler_en が真．19.6・19.15 とも）は，Doppler と呼ぶ制御で
// ボードの入力電力（INPUT_POWER．ボードの管理用 MCU（DM）が 1 ms ごとに知らせる入力電力に fw_table の
// additional_board_power（p150a で 20 W）を足した値）の 1 s（1 ms ごとの 1000 標本）の移動平均をボードの電力の上限
// （BOARD_POWER_LIMIT．ケーブルの上限と fw_table の board_power_limit 300 W の小さい方）に保つよう AICLK の上限を動かす．
// Doppler が有効なあいだはチップの電流（TDC 200 A）・電力（TDP 150 W）・ボードの電力の各制御は無効にされる
// （tt-zephyr-platforms の lib/tenstorrent/bh_arc/throttler.c の InitThrottlers）．密な GEMM を流すと入力電力が 300 W に
// 張り付き，1〜2 s で AICLK が 1350 MHz から下がり，流し続けると温度とともにさらに下がる（下がった値とファームウェアの版は
// docs/remeasure_other_machine.md の 6.1「何がクロックを下げているかの切り分け」．scripts/study/clock_study.sh）．
// 計時区間のあいだのクロックと，それを決めている仕組みをログに残すため，裏のスレッドで一定の間隔で読み，時刻つきの標本として
// 持っておく．計時の側（oz_timing.hpp の段の計時）は区間 [t0,t1] を記録しておき，後でその区間の要約（stats）を取る．
//
// 読むもの 1：tt-kmd の sysfs（コンテナの中からも同じパスで読める．run.sh はデバイスファイルだけを渡すが sysfs は見える）
//   /sys/class/tenstorrent/tenstorrent!<n>/tt_aiclk                    AICLK［MHz］
//   /sys/class/tenstorrent/tenstorrent!<n>/device/hwmon/hwmon*/curr1_input   コアの電流［mA］（テレメトリ表の TDC）
//                                                          power1_input  コアの電力［uW］（テレメトリ表の TDP）
//                                                          temp1_input   ASIC の温度［m degC］
//                                                          in0_input     コア電圧［mV］（テレメトリ表の VCORE）
// 1 ファイルの読み出しは pread で約 1.4 us（5 ファイルで 10 us 未満）なので，採取そのものの負荷は無視できる．
// ファームウェアがこれらの値を更新するのは 100 ms ごとである（1 ms 間隔で読むと値の変わり目が 100 ms おきに並ぶ）．
// したがって間隔を 100 ms より細かくしても新しい値は増えず，変わり目を捉える遅れが縮むだけである．
// 既定の 20 ms は，その遅れを更新の周期の 1/5 に抑えつつ，標本の数（1 分で 3000）を小さく保つ値である．
//
// 読むもの 2：ファームウェアのテレメトリ表（ARC のメモリ）．sysfs には入力電力と周波数の仕組みが無いので，tt-metal に
// 入っている UMD（tt::umd::TTDevice と ArcTelemetryReader．tt-smi 6.x も同じ部品で読む．UMD の版は tt-metal の版，
// つまり出どころの行の metal= で決まるので，ログには別に書かない）で読む．タグの番号は
// UMD の TelemetryTag（tt-zephyr-platforms の lib/tenstorrent/bh_arc/telemetry.h と同じ）：
//   INPUT_POWER 54        ボードの入力電力［W］
//   AICLK_ARB_MAX 66      上位 16 bit が AICLK の上限を決めている仕組み（下の arb_name），下位 16 bit がその周波数［MHz］．
//                         有効な仕組みのうち周波数の最も低いものを返し，同じ値なら番号の大きい方を返す（aiclk_ppm.c の
//                         get_aiclk_effective_arb_max）．どの仕組みも fmax より下へ絞っていないとき（負荷の無いときなど）も
//                         8（Doppler）・1350 MHz と出るので，要約では周波数が fmax 以上の標本を 0（fmax）として数える
//   ENABLED_MAX_ARB 68    有効な仕組みのビットの並び（p150a の 19.15 で 0x1d1＝fmax・温度・電圧・GDDR 温度・Doppler．
//                         入力電力が上限の 2 倍か 2.5 倍を超えると Doppler critical（bit 9）が加わる）
//   BOARD_POWER_LIMIT 53  ボードの電力の上限［W］（ケーブルの上限を DM が知らせると変わりうるので標本ごとに読む）
//   開くときに 1 度：AICLK_LIMIT_MAX 63（fmax．上の 0 の判定に使う）・TDC_LIMIT_MAX 55・TDP_LIMIT_MAX 64・
//   FLASH_BUNDLE_VERSION 28・UPDATE_TELEM_SPEED 5（表の更新の周期［ms］）．TDC 8・TDP 7 は hwmon の curr1・power1 と
//   同じ値（tt-kmd が同じタグを読む）なので読まない．
// UMD の TTDevice を，tt-metal が開くものとは別に 1 つ作る（同じデバイスファイルをもう 1 度開く．作るのに約 3 ms）．
// 別に作るのは，tt-metal の内部の UMD（MetalContext の Cluster）が公開ヘッダから取れないためと，tt-metal の入出力と
// 状態を共有しないためである：TTDevice は自分の TLB の窓（KMD に割り当てを頼む 2 MiB 1 枚）と，その窓を守る
// 自分の排他（PcieProtocol の io_lock_）を持ち，読み出しは ARC のタイルのメモリへの 4 byte の PCIe の読み出し
// （NoC 経由）である．Tensix のコア・DRAM・tt-metal の TLB には触れず，tt-smi を別のプロセスで回しながら GEMM を
// 流すのと同じ経路になる．このプロセスの中で UMD を触るのは採取のスレッドだけ（開くときの 1 度を除く）．
// 1 標本（4 タグ）の読み出しは sysfs と同じ桁（数 us から十数 us．測った値は docs/remeasure_other_machine.md の 6.3
// 「定常状態の計時」）なので，間隔を延ばす理由は無く，sysfs と同じ間隔で読む（表の更新は 100 ms ごとなので，細かく読んでも
// 新しい値は増えない）．読み出しにかかった時間の平均と最大は FWTELEM 行に残す（NoC が混んでいるときの遅れも見える）．
// 開けない・読めないときは「ファームウェアのテレメトリなし」とし（fw_available() が偽，理由は fw_why()），sysfs の
// 採取は続ける（定常状態の判定は AICLK だけを使う）．UMD のヘッダの無いビルド（tt-metal にリンクしない tools/ の検査）
// では読まない．
//
// 環境変数（値はプロセスの初めに 1 度だけ読む．打ち間違いは exit(2)）
//   OZ_TELEM_MS=m   採取の間隔［ms］（既定 20．1〜1000．0 は採取しない＝テレメトリなし）
//   OZ_TELEM_FW=0|1 ファームウェアのテレメトリ表も読むか（既定 1）
//   OZ_TELEM_DEV=n  読むカードの番号 n（tenstorrent!<n> と /dev/tenstorrent/<n>）．無ければ TT_DEVICE（/dev/tenstorrent/<n> の形）
//                   の番号，それも無ければ /dev/tenstorrent/ に番号のデバイスが 1 つだけならその番号，どれでもなければ 0．
//                   scripts/run.sh はホストの TT_DEVICE の番号をこの変数でコンテナへ渡す（tt-metal が開く
//                   デバイスと同じカードを読むため）．採取しないとき（OZ_TELEM_MS=0）はカードを決めない（値の検査だけ）
// 開くときに sysfs のファイルが無い・読めないときは「テレメトリなし」とし（available() が偽，理由は why()），標本を採らない．
// それを誤りにするかは呼び出し側が決める（oz_timing.hpp の timing_check：定常状態の計時はクロックを見て
// 温め終わりを決めるので，テレメトリなしでは始めない）．available() は開いたときの結果で，その後の採取の失敗は
// 下の「欠測」として区間ごとの要約（Stats の miss・stale_s）に出る．
//
// 区間の要約は，標本の値が次の標本まで続くとみなして（sample-and-hold）区間との重なりの長さで重み付けした
// 平均を取る．計時区間が採取の間隔より短くても，その時刻に保持している標本の値が入る．ただしファームウェアが値を
// 更新するのは 100 ms ごとなので，数 ms の計時区間の値は，区間の始まりより最大で約 120 ms（更新の周期と既定の採取の
// 間隔の和）前の状態を表す．計時区間ごとの値（TELEMIV 行・drift_max_pct）はこの遅れでぼける（段の平均への影響の
// 大きさは確かめていない）．
// 欠測：sysfs のどれかが読めなかった回は標本を作らずに捨て（take），その時刻を数える（Stats の miss は区間の中で捨てた
// 回数）．読めない回が続くと前の標本が保持され続けるので，保持している標本が採取の間隔の kStaleFactor 倍より古い部分
// （と最初の標本より前）も欠測の時間として数える（Stats の stale_s）．どちらかがあれば区間の要約は fresh() が偽になり，
// 計時の側はその区間のクロックを定常状態の判定と計時区間の基準に使わない（oz_timing.hpp の warm_up・end_timed_at．
// 要約そのものは ok なら出すので，TELEM 行の miss・stale_ms に欠測が残り，検査が拒む）．
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include "oz_env.hpp"   // つまみの読み方（同じディレクトリ．tools/ から取り込むときも見つかるように相対で書く）

#if __has_include(<umd/device/tt_device/tt_device.hpp>)
#define OZ_TELEM_HAVE_UMD 1
#include <umd/device/arc/arc_telemetry_reader.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#else
#define OZ_TELEM_HAVE_UMD 0
#endif

namespace oz_telem {

// AICLK_ARB_MAX の上位 16 bit の番号と名前（aiclk_ppm.h の enum aiclk_arb_max と同じ並び）
inline const char* arb_name(int id) {
    static const char* const kArbName[] = {"fmax", "TDP", "fast_TDC", "TDC", "thermal", "board_power", "voltage",
                                           "GDDR_thermal", "Doppler", "Doppler_critical", "host_fmax"};
    return (id >= 0 && id < (int)(sizeof kArbName / sizeof kArbName[0])) ? kArbName[id] : "unknown";
}

// 標本 1 本（t は now() の秒）
struct Sample {
    double t;
    double aiclk;    // MHz
    double curr;     // A
    double power;    // W（コア）
    double vcore;    // mV
    double temp;     // degC
    bool fw;         // 以下のファームウェアの値が読めたか
    double inpow;    // W（ボードの入力）
    uint32_t arb;    // AICLK_ARB_MAX の生の値
    uint32_t en_arb; // ENABLED_MAX_ARB
    double bpl;      // BOARD_POWER_LIMIT［W］
};

// 区間の要約．n は標本と区間の重なりの数（区間ごとに数えるので，1 本の標本が 2 つの区間に重なれば 2 と数える．
// 採った標本の本数ではない．0 なら ok=false で，値は意味を持たない）．fw_n はそのうちファームウェアの値が読めた
// 標本の重なりの数（0 なら fw_ok=false で，fw の欄は意味を持たない）．miss・stale_s は区間の中の欠測（冒頭の「欠測」：
// sysfs が読めずに捨てた回数と，保持している標本が古すぎた時間［s］．長さ 0 の区間がその部分に入れば 1e-9 s と数える）
struct Stats {
    bool ok = false;
    uint64_t n = 0;
    uint64_t miss = 0;
    double stale_s = 0;
    double aiclk_mean = 0, aiclk_min = 0, aiclk_max = 0;
    double curr_mean = 0, curr_max = 0;
    double power_mean = 0, power_max = 0;
    double vcore_mean = 0;
    double temp_start = 0, temp_max = 0;
    bool fw_ok = false;
    uint64_t fw_n = 0;
    double inpow_mean = 0, inpow_max = 0;
    double arbf_mean = 0, arbf_min = 0;                  // AICLK_ARB_MAX の周波数［MHz］
    double bpl_min = 0, bpl_max = 0;                     // BOARD_POWER_LIMIT［W］
    std::map<int, double> arb_frac;                      // 周波数を絞っている仕組み（fmax 以上は 0）ごとの時間の割合
    std::vector<uint32_t> en_arb;                        // 出てきた ENABLED_MAX_ARB（出てきた順．ふつうは 1 つ）
    // 欠測の無い要約（定常状態の判定と計時区間の基準に使ってよい）
    bool fresh() const { return ok && miss == 0 && stale_s == 0; }
    // arb_frac を "8:0.998|0:0.002" の形にする（割合の大きい順．値に ',' と '=' は入らない）
    std::string arb_str() const {
        std::vector<std::pair<double, int>> v;
        for (const auto& [id, f] : arb_frac) v.push_back({-f, id});
        std::sort(v.begin(), v.end());
        std::string s;
        char b[32];
        for (const auto& [nf, id] : v) {
            std::snprintf(b, sizeof b, "%s%d:%.3f", s.empty() ? "" : "|", id, -nf);
            s += b;
        }
        return s.empty() ? "na" : s;
    }
    // 割合の最も大きい仕組みの番号（無ければ -1）
    int arb_top() const {
        int id = -1;
        double f = -1;
        for (const auto& [k, x] : arb_frac)
            if (x > f) f = x, id = k;
        return id;
    }
    std::string en_arb_str() const {
        std::string s;
        char b[16];
        for (uint32_t x : en_arb) {
            std::snprintf(b, sizeof b, "%s0x%x", s.empty() ? "" : "|", x);
            s += b;
        }
        return s.empty() ? "na" : s;
    }
};

// 単調な時計の秒（プロセスの中で最初に呼んだ時点を 0 とする．区間の記録と標本の時刻は同じ時計で取る）
inline double now() {
    using clk = std::chrono::steady_clock;
    static const clk::time_point t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// ログの 1 欄に入れる文字列（',' '=' と制御文字を空白にする）
inline std::string field_safe(std::string s) {
    for (char& c : s)
        if (c == ',' || c == '=' || (unsigned char)c < 0x20) c = ' ';
    return s;
}

// ファームウェアのテレメトリ表を開いたときの静的な値（読めなかった欄は 0）
struct FwInfo {
    uint32_t aiclk_limit_max = 0, tdc_limit_max = 0, tdp_limit_max = 0, update_ms = 0, bundle = 0;
    uint32_t board_power_limit = 0, enabled_max_arb = 0;
    double open_ms = 0;                                  // TTDevice を作って初期化するまでの時間
};

class Sampler {
public:
    // プロセスに 1 つ（最初の呼び出しで環境変数を読み，ファイルとファームウェアのテレメトリ表を開く．採取はまだ始めない）
    static Sampler& instance() {
        static Sampler s;
        return s;
    }
    bool available() const { return ok_; }
    const std::string& why() const { return why_; }
    int dev() const { return dev_; }
    // dev をどこから決めたか（OZ_TELEM_DEV・TT_DEVICE・sole_device・default．FWTELEM 行の dev_src）
    const char* dev_source() const { return dev_src_; }
    int period_ms() const { return period_ms_; }
    bool fw_available() const { return fw_ok_.load(); }
    std::string fw_why() const { std::lock_guard<std::mutex> lk(mu_); return fw_why_; }
    const FwInfo& fw_info() const { return fw_info_; }

    // 採取を始める（既に動いていれば何もしない）．最初の 1 本はここで同期して採るので，
    // 呼んだ直後から区間の要約が取れる
    void start() {
        if (!ok_ || running_.load()) return;
        take();
        running_.store(true);
        th_ = std::thread([this] { loop(); });
    }
    // 採取を止める（標本は残る）．プロセスの終わりにデストラクタからも呼ばれる
    void stop() {
        if (!running_.exchange(false)) return;
        if (th_.joinable()) th_.join();
    }
    ~Sampler() {
        stop();
        for (int fd : fd_) if (fd >= 0) ::close(fd);
#if OZ_TELEM_HAVE_UMD
        umd_.reset();
#endif
    }

    // ファームウェアの読み出しの記録（FWTELEM 行）：読んだ標本の数，失敗した回数，1 標本（4 タグ）の読み出しの
    // 時間の平均と最大［us］
    struct FwCost { uint64_t n = 0, errors = 0; double us_mean = 0, us_max = 0; double last_bpl = -1; };
    FwCost fw_cost() const {
        std::lock_guard<std::mutex> lk(mu_);
        FwCost c;
        c.n = fw_reads_;
        c.errors = fw_errors_;
        c.us_mean = fw_reads_ ? fw_us_sum_ / fw_reads_ : 0.0;
        c.us_max = fw_us_max_;
        for (auto it = buf_.rbegin(); it != buf_.rend(); ++it)
            if (it->fw) { c.last_bpl = it->bpl; break; }
        return c;
    }

    // 区間の並び iv（[t0,t1] の組．重なっていないこと）の要約
    Stats stats(const std::vector<std::pair<double, double>>& iv) const {
        Stats s;
        std::lock_guard<std::mutex> lk(mu_);
        if (buf_.empty()) return s;
        double wsum = 0, wa = 0, wc = 0, wp = 0, wv = 0;
        double fsum = 0, fi = 0, ff = 0;
        std::map<int, double> fw_hist;
        auto add = [&](const Sample& x, double w) {
            if (s.n == 0) {
                s.temp_start = x.temp;
                s.aiclk_min = s.aiclk_max = x.aiclk; s.curr_max = x.curr; s.power_max = x.power; s.temp_max = x.temp;
            } else {
                s.aiclk_min = std::min(s.aiclk_min, x.aiclk); s.aiclk_max = std::max(s.aiclk_max, x.aiclk);
                s.curr_max = std::max(s.curr_max, x.curr); s.power_max = std::max(s.power_max, x.power);
                s.temp_max = std::max(s.temp_max, x.temp);
            }
            ++s.n;
            wsum += w; wa += w * x.aiclk; wc += w * x.curr; wp += w * x.power; wv += w * x.vcore;
            if (x.fw) {
                const double f = (double)(x.arb & 0xffffu);
                if (s.fw_n == 0) {
                    s.inpow_max = x.inpow; s.arbf_min = f; s.bpl_min = s.bpl_max = x.bpl;
                } else {
                    s.inpow_max = std::max(s.inpow_max, x.inpow); s.arbf_min = std::min(s.arbf_min, f);
                    s.bpl_min = std::min(s.bpl_min, x.bpl); s.bpl_max = std::max(s.bpl_max, x.bpl);
                }
                ++s.fw_n;
                fsum += w; fi += w * x.inpow; ff += w * f;
                fw_hist[limiting(x.arb)] += w;
                if (std::find(s.en_arb.begin(), s.en_arb.end(), x.en_arb) == s.en_arb.end()) s.en_arb.push_back(x.en_arb);
            }
        };
        // 欠測の時間：区間 [a,b] のうち [lo,hi) に入る長さ（長さ 0 の区間がそこに入れば 1e-9 s と数える）
        auto stale = [&](double a, double b, double lo, double hi) {
            if (a == b) { if (a >= lo && a < hi) s.stale_s += 1e-9; return; }
            const double w = std::min(b, hi) - std::max(a, lo);
            if (w > 0) s.stale_s += w;
        };
        for (const auto& [a, b] : iv) {
            if (!(b >= a)) continue;
            // 標本 k の値は [t_k, t_{k+1}) のあいだ続くとみなす（最初の標本は前へも，最後の標本は後へも続く）．
            // a を含む標本から，b を越える標本の手前までを重なりの長さで重み付けする．長さ 0 の区間（a==b）は
            // その時刻を含む標本を 1 本数える．保持のうち標本の時刻から stale_sec_ を過ぎた部分と，最初の標本より前は欠測
            stale(a, b, -1e300, buf_.front().t);
            for (size_t k = idx_at(a); k < buf_.size(); ++k) {
                const Sample& x = buf_[k];
                const double hs = (k == 0) ? -1e300 : x.t;
                if (hs > b) break;
                const double he = (k + 1 < buf_.size()) ? buf_[k + 1].t : 1e300;
                const double w = std::min(b, he) - std::max(a, hs);
                if (w > 0 || a == b) add(x, std::max(w, 1e-9));
                stale(a, b, x.t + stale_sec_, he);
                if (he >= b) break;
            }
            // 区間の中で sysfs が読めずに捨てた回（miss_t_ は時刻の順）
            s.miss += (uint64_t)(std::upper_bound(miss_t_.begin(), miss_t_.end(), b) -
                                 std::lower_bound(miss_t_.begin(), miss_t_.end(), a));
        }
        if (s.n == 0 || wsum <= 0) return Stats{};
        s.ok = true;
        s.aiclk_mean = wa / wsum; s.curr_mean = wc / wsum; s.power_mean = wp / wsum; s.vcore_mean = wv / wsum;
        if (s.fw_n > 0 && fsum > 0) {
            s.fw_ok = true;
            s.inpow_mean = fi / fsum;
            s.arbf_mean = ff / fsum;
            for (const auto& [id, w] : fw_hist) s.arb_frac[id] = w / fsum;
        }
        return s;
    }
    // 区間 [t0,t1] の AICLK の平均（標本が無いか，区間に欠測があれば負．定常状態の判定と計時区間の基準はこれで読む）
    double aiclk_mean(double t0, double t1) const {
        const Stats s = stats({{t0, t1}});
        return s.fresh() ? s.aiclk_mean : -1.0;
    }
    // 区間 [t0,t1] の要約
    Stats stats(double t0, double t1) const { return stats({{t0, t1}}); }

    // AICLK_ARB_MAX の値から，AICLK を fmax より下へ絞っている仕組みの番号（周波数が fmax 以上なら 0＝fmax）
    int limiting(uint32_t arb) const {
        const uint32_t f = arb & 0xffffu;
        const int id = (int)(arb >> 16);
        const uint32_t fmax = fw_info_.aiclk_limit_max;
        return (fmax > 0 && f >= fmax) ? 0 : id;
    }

private:
    Sampler() {
        period_ms_ = (int)oz_env::i64("OZ_TELEM_MS", 20, 0, 1000);
        stale_sec_ = kStaleFactor * period_ms_ / 1e3;
        const bool want_fw = oz_env::i64("OZ_TELEM_FW", 1, 0, 1) != 0;
        // 採取しないならカードを決めない（読めない TT_DEVICE で，テレメトリを使わない実行まで止めないように．
        // つまみの OZ_TELEM_DEV は打ち間違いだけを見る）
        if (period_ms_ == 0) {
            if (oz_env::set("OZ_TELEM_DEV")) (void)oz_env::i64("OZ_TELEM_DEV", 0, 0, 255);
            dev_src_ = "none";
            why_ = "OZ_TELEM_MS=0（採取しない）"; fw_why_ = "OZ_TELEM_MS が 0（採取しない）";
            return;
        }
        dev_ = resolve_dev();
        const std::string base = "/sys/class/tenstorrent/tenstorrent!" + std::to_string(dev_);
        std::string hw;
        if (DIR* d = ::opendir((base + "/device/hwmon").c_str())) {
            std::vector<std::string> names;
            while (dirent* e = ::readdir(d))
                if (std::strncmp(e->d_name, "hwmon", 5) == 0) names.push_back(e->d_name);
            ::closedir(d);
            std::sort(names.begin(), names.end());
            if (!names.empty()) hw = base + "/device/hwmon/" + names.front();
        }
        if (hw.empty()) { why_ = base + "/device/hwmon/hwmon* が無い"; fw_why_ = "sysfs のテレメトリが無い"; return; }
        const std::string path[5] = {base + "/tt_aiclk", hw + "/curr1_input", hw + "/power1_input",
                                     hw + "/in0_input", hw + "/temp1_input"};
        for (int i = 0; i < 5; ++i) {
            fd_[i] = ::open(path[i].c_str(), O_RDONLY | O_CLOEXEC);
            long v;
            if (fd_[i] < 0 || !read_long(fd_[i], v)) { why_ = path[i] + " が読めない"; fw_why_ = "sysfs のテレメトリが無い"; return; }
        }
        ok_ = true;
        if (!want_fw) fw_why_ = "OZ_TELEM_FW が 0（読まない）";
        else open_fw();
    }
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // ファームウェアのテレメトリ表を開き，静的な値を読む（上の「読むもの 2」．失敗しても sysfs の採取は続ける）
    void open_fw() {
#if OZ_TELEM_HAVE_UMD
        const auto t0 = std::chrono::steady_clock::now();
        try {
            std::unique_ptr<tt::umd::TTDevice> d = tt::umd::TTDevice::create(dev_);
            d->init_tt_device();
            tt::umd::ArcTelemetryReader* r = d->get_arc_telemetry_reader();
            if (!r) { fw_why_ = "UMD の ArcTelemetryReader が無い"; return; }
            const uint8_t need[] = {kInputPower, kArbMax, kEnabledMaxArb, kBoardPowerLimit, kAiclkLimitMax};
            for (uint8_t tg : need)
                if (!r->is_entry_available(tg)) { fw_why_ = "テレメトリ表にタグ " + std::to_string(tg) + " が無い"; return; }
            auto opt = [&](uint8_t tg) -> uint32_t { return r->is_entry_available(tg) ? r->read_entry(tg) : 0u; };
            fw_info_.aiclk_limit_max = r->read_entry(kAiclkLimitMax);
            fw_info_.tdc_limit_max = opt(kTdcLimitMax);
            fw_info_.tdp_limit_max = opt(kTdpLimitMax);
            fw_info_.update_ms = opt(kUpdateTelemSpeed);
            fw_info_.bundle = opt(kFlashBundleVersion);
            fw_info_.board_power_limit = r->read_entry(kBoardPowerLimit);
            fw_info_.enabled_max_arb = r->read_entry(kEnabledMaxArb);
            fw_info_.open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            umd_ = std::move(d);
            reader_ = r;
            fw_ok_.store(true);
        } catch (const std::exception& e) {
            fw_why_ = field_safe(std::string("UMD で開けない：") + e.what());
        } catch (...) {
            fw_why_ = "UMD で開けない（例外）";
        }
#else
        fw_why_ = "UMD のヘッダの無いビルド";
#endif
    }

    // 読むカードの番号（上の OZ_TELEM_DEV の説明の順に決める）
    int resolve_dev() {
        if (oz_env::set("OZ_TELEM_DEV")) { dev_src_ = "OZ_TELEM_DEV"; return (int)oz_env::i64("OZ_TELEM_DEV", 0, 0, 255); }
        if (const char* p = std::getenv("TT_DEVICE")) {
            const char* b = std::strrchr(p, '/');
            b = b ? b + 1 : p;
            char* end = nullptr;
            const long x = std::strtol(b, &end, 10);
            if (*b && end && *end == '\0' && x >= 0 && x <= 255) { dev_src_ = "TT_DEVICE"; return (int)x; }
            std::fprintf(stderr, "ERROR: TT_DEVICE=%s から番号が読めない（/dev/tenstorrent/<番号> の形．OZ_TELEM_DEV で与えてもよい）\n", p);
            std::exit(2);
        }
        int found = -1, count = 0;
        if (DIR* d = ::opendir("/dev/tenstorrent")) {
            while (dirent* e = ::readdir(d)) {
                char* end = nullptr;
                const long x = std::strtol(e->d_name, &end, 10);
                if (e->d_name[0] >= '0' && e->d_name[0] <= '9' && end && *end == '\0') { found = (int)x; ++count; }
            }
            ::closedir(d);
        }
        if (count == 1) { dev_src_ = "sole_device"; return found; }
        dev_src_ = "default";
        return 0;
    }
    static bool read_long(int fd, long& v) {
        char b[32];
        const ssize_t r = ::pread(fd, b, sizeof b - 1, 0);
        if (r <= 0) return false;
        b[r] = '\0';
        char* end = nullptr;
        v = std::strtol(b, &end, 10);
        return end != b;
    }
    // 1 本採る（sysfs のどれかが読めなければその回は捨て，捨てた時刻を miss_t_ に残す（冒頭の「欠測」）．ファームウェアの
    // 値が読めなければ fw=false の標本にし，例外が出たらそれ以降は読まない）
    void take() {
        long v[5];
        for (int i = 0; i < 5; ++i)
            if (!read_long(fd_[i], v[i])) {
                const double t = now();
                std::lock_guard<std::mutex> lk(mu_);
                miss_t_.push_back(t);
                return;
            }
        Sample s{now(), (double)v[0], v[1] / 1e3, v[2] / 1e6, (double)v[3], v[4] / 1e3, false, 0.0, 0u, 0u, 0.0};
        double us = -1.0;
        std::string err;
#if OZ_TELEM_HAVE_UMD
        if (fw_ok_.load()) {
            const auto a = std::chrono::steady_clock::now();
            try {
                s.inpow = (double)reader_->read_entry(kInputPower);
                s.arb = reader_->read_entry(kArbMax);
                s.en_arb = reader_->read_entry(kEnabledMaxArb);
                s.bpl = (double)reader_->read_entry(kBoardPowerLimit);
                s.fw = true;
            } catch (const std::exception& e) {
                err = field_safe(std::string("読み出しで例外：") + e.what());
            } catch (...) {
                err = "読み出しで例外";
            }
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
            if (!err.empty()) fw_ok_.store(false);
        }
#endif
        std::lock_guard<std::mutex> lk(mu_);
        buf_.push_back(s);
        if (us >= 0) {
            if (s.fw) { ++fw_reads_; fw_us_sum_ += us; fw_us_max_ = std::max(fw_us_max_, us); }
            else { ++fw_errors_; fw_why_ = err; }
        }
    }
    void loop() {
        auto next = std::chrono::steady_clock::now();
        const auto dt = std::chrono::milliseconds(period_ms_);
        while (running_.load()) {
            next += dt;
            std::this_thread::sleep_until(next);
            if (!running_.load()) break;
            take();
        }
    }
    // 時刻 t を含む標本の番号（t 以前で最後の標本．t が最初の標本より前なら 0）．mu_ を持って呼ぶ
    size_t idx_at(double t) const {
        auto it = std::upper_bound(buf_.begin(), buf_.end(), t, [](double x, const Sample& s) { return x < s.t; });
        return it == buf_.begin() ? 0 : (size_t)(it - buf_.begin()) - 1;
    }

    // テレメトリ表のタグ（UMD の TelemetryTag と同じ番号）．標本ごとに読むもの（前の 5 つ）と開くときに 1 度読むもの
    static constexpr uint8_t kBoardPowerLimit = 53, kInputPower = 54, kAiclkLimitMax = 63, kArbMax = 66, kEnabledMaxArb = 68;
    static constexpr uint8_t kTdcLimitMax = 55, kTdpLimitMax = 64, kUpdateTelemSpeed = 5, kFlashBundleVersion = 28;
    // 欠測とみなす標本の古さ（採取の間隔の倍数）．既定の 20 ms で 100 ms＝ファームウェアの更新の周期で，読めない回が
    // 5 回続く（新しい値を 1 回見落としうる）と欠測の時間になる（それより短い失敗も miss に数える）．採取のスレッドの
    // 起き上がりの遅れ（ふつうは 1 ms 未満）では届かない
    static constexpr double kStaleFactor = 5.0;

    bool ok_ = false;
    std::string why_;
    int dev_ = 0;
    const char* dev_src_ = "default";
    int period_ms_ = 20;
    double stale_sec_ = 0.1;
    int fd_[5] = {-1, -1, -1, -1, -1};
    std::atomic<bool> running_{false};
    std::thread th_;
    mutable std::mutex mu_;
    std::vector<Sample> buf_;
    std::vector<double> miss_t_;   // sysfs が読めずに捨てた回の時刻（時刻の順）
    // ファームウェアのテレメトリ表（fw_ok_ が真のあいだ採取のスレッドだけが reader_ を使う）
    std::atomic<bool> fw_ok_{false};
    std::string fw_why_;
    FwInfo fw_info_;
    uint64_t fw_reads_ = 0, fw_errors_ = 0;
    double fw_us_sum_ = 0, fw_us_max_ = 0;
#if OZ_TELEM_HAVE_UMD
    std::unique_ptr<tt::umd::TTDevice> umd_;
    tt::umd::ArcTelemetryReader* reader_ = nullptr;
#endif
};

}  // namespace oz_telem
