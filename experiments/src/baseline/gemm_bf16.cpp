// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BF16 行列積の最適化実装（論文の基準実装．素の BF16 GEMM）
// ---------------------------------------------------------------------------
// 全 110 コアを使う構成に加えて，K 方向のスピル往復を最小化するよう
// in0_block_w を自動選択する．本ハードウェアで到達しうる素の行列積性能を
// 確定するのが目的で，尾崎スキームの各実装はこの値と比較される．
//
// なぜ速くなるのか
//   計算カーネル（bmm_large_block_zm.cpp）は K を num_blocks = Kt / in0_block_w
//   個に分割し，2 つ目以降のブロックでは出力サブブロックの部分和をいったん L1
//   （CB c_24）へ退避して読み戻し，累積する．この往復は出力サブブロック 1 つに
//   つき num_blocks-1 回起き，コアあたりのオーバーヘッドを支配する．
//   in0_block_w を大きくすれば K ブロック数が減り往復も減る（num_blocks==1 なら
//   スピルは起きない）．tt-metal のサンプルは in0_block_w=2 に固定している．
//
// choose_cfg()（common/ozaki_gemm_host.hpp）の優先順位
//   1. 使用コア数 cr*cc を最大化する
//   2. そのうえで，Kt を割り切りかつ L1 に収まる最大の in0_block_w を選ぶ
//      （per_core_N <= example_max_block_dim(per_core_M, in0_block_w)．tt-metal の行列積の例の get_maximum_block_dim と
//      同じ「タイル数 400」の判定）
//      ＝ スピル回数を最小化する
//   3. それでも同点なら列挙の順（コア格子の縦 cr，次に横 cc の大きい順）で最初のもの（コアあたり出力面積
//      pm*pn = Mt*Nt/(cr*cc) はコア数が同じなら等しいので，面積では決まらない）
//
// 追加のつまみ
//   fp32_dest_acc=1  DST を FP32 で累算する（精度は上がるが DST に入るタイル数が
//                    半分になり，サブブロックの面積は half-sync なら 4，dst_full_sync=1 と併せれば 8 までに
//                    制限される）．FP32 になるのは
//                    DST（1 つの読み込みブロックの累積）だけで，読み込みブロック間の部分和（c_24）と
//                    出力（c_16）は BF16 のままなので，実効ビット数は K とともに下がる（下の「到達精度の下限」）．
//                    スライドの基準の表の 2 構成（BF16 累積と FP32 累積）はこの 0・1 で分かれる（論文の表と
//                    PDP 版の本文は FP32 累積だけ）
//   dst_full_sync=1  DST を 2 倍使えるようにする（サブブロック面積 16 まで）
//   in0_block_w（0 以外）と dst_full_sync=1 は調査用で，使うのは archive/tune_gemm_bf16.sh だけ（論文の測定は 0）
//
// 使い方:
//   gemm_bf16 [M] [N] [K] [B] [warmup] [iters] [verify]
//             [in0_block_w(0=自動)] [fp32_dest_acc] [dst_full_sync]
//   既定値:   2560 2816 2048 1 2 10 1 0 0 0
//   B はバッチ数で 1 だけ（GEMM 段はバッチを扱わない．位置引数の並びを変えないために残してある）．
//   M N K は正の 32 の倍数，warmup は 0..1000000，iters は 1..1000000（common/oz_driver.hpp の read_dims・read_counts．
//   warmup=0 でも行列積の JIT は計時に入らない），verify・fp32_dest_acc・dst_full_sync は 0 か 1．引数は 10 個まで
//   （多ければデバイスを開く前に 2 で止める）．
//   OZ_NOVERIFY が立っていれば verify より優先して参照解を省く（common/oz_reference.hpp の verify_enabled）．検証を省いた実行の最後の行は
//   "Test Skipped"（終了コード 0）で，OZ_DUMP_C とは併用できない（check_env が止める）．
//   OZ_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4 で MathFidelity を上書きできる（既定 HiFi4．BF16 の積を丸めなく完全精度で得るには
//   HiFi4 の 4 パスが要り，ピークは LoFi の 1/4．論文の fidelity の掃引で使う．common/oz_fidelity.hpp）．
//   OZ_L1_BUDGET_KB は受け付けない：構成探索は tt-metal の行列積の例の判定（タイル数 400．example_max_block_dim）で選び，
//   L1 の予算を見ない（黙って受けると，予算を変えて測ったつもりで同じ構成を測る）．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include "common/host_eval.hpp"
// 構成探索・プログラム構築（build_mcast_gemm）・行列積 1 本の実行（run_mcast_gemm）・パネルの切り出しは尾崎スキームと
// 共通なので共有ヘッダのものを使う（choose_cfg の area_cap でサブブロック面積の上限を渡す．mcast_gemm.hpp が
// ozaki_gemm_host.hpp を取り込む）
#include "common/mcast_gemm.hpp"
#include "common/oz_driver.hpp"     // 入口の検査の並び（oz_drv::reject_env_gemm・read_dims・read_counts・check_env）・最後の行（finish）・CONFIG 行（OzLine）
#include "common/oz_fidelity.hpp"   // MathFidelity の上書き・表示名・パス数・理論ピーク（基準実装 3 つと尾崎で共通）

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

// 最適構成 cfg のもとで動かす BF16 行列積の仕様．計算カーネルは tt-metal 付属の bmm_large_block_zm.cpp をそのまま使い，
// 構成（コア格子，コアあたり出力，サブブロック，in0_block_w）だけを外から与えるのが本実装の要点である．
// データ移動は尾崎の各経路と同じ common/kernels/mcast_split_dm.cpp（A/B を 2 つの RISC に分け，
// 両端の 2 コアから配る）で，tt-metal 付属のリーダ 4 種を使う構成と比べて 3520^3・14080^3 の
// どちらでも ±0.3% 以内で同じ速さである（計算カーネル側で律速している．
// docs/gemm_opt_results.md の §6）．K ブロック間の部分和は標準カーネルのまま BF16 の c_24 へ
// 退避して読み戻すので，素の BF16 GEMM の基準であることは変わらない．
// 入力 CB は「コアあたり出力の一辺 × K ブロック幅」の 2 倍（ダブルバッファ）．
// 出力 c_16 と K スピル先 c_24 は同じ領域を共有し，どちらも BF16 である（fp32_dest_acc=1 でも FP32 になるのは DST だけ）．
// num_blocks = Kt/in0_block_w が K 方向の分割数＝スピル回数を決める．
// 形の前提のうち pm|Mt・pn|Nt は，構成探索の choose_cfg が割り切る格子だけを選ぶことで保たれる
// （build_mcast_gemm は cr·pm<=Mt・cc·pn<=Nt しか見ない）．sh|pm・sw|pn・bw|Kt と DST の容量は
// build_mcast_gemm の mcast_shape_check が検査する（TT_ASSERT は Release では何もしないので，ここでは検査しない）．
static McastGemmSpec bf16_gemm_spec(const OptConfig& cfg, uint32_t Kt, tt::tt_metal::MathFidelity fid, bool fp32_dest_acc,
                                    bool dst_full_sync) {
    const tt::DataFormat df = tt::DataFormat::Float16_b;
    McastGemmSpec spec;
    spec.in_fmt = df;
    spec.in_tile = tt::tile_size(df);
    spec.c16_fmt = spec.c24_fmt = df;
    spec.out_tile = tt::tile_size(df);
    spec.out_shared = true;
    spec.compute = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp";
    spec.ccfg = tt_metal::ComputeConfig{
        .math_fidelity = fid,
        .fp32_dest_acc_en = fp32_dest_acc,
        .dst_full_sync_en = dst_full_sync,
        .compile_args = mcast_compute_args(cfg, Kt)};
    return spec;
}


// ---- 入口（デバイスを開く前．並びは尾崎の 2 つのドライバと同じ．common/oz_driver.hpp の冒頭の 1〜4）----
// 位置引数（M N K B warmup iters verify in0_block_w fp32_dest_acc dst_full_sync）と MathFidelity
struct Bf16Args {
    uint32_t M = 0, N = 0, K = 0, B = 1, warmup = 0, iters = 0;
    uint32_t forced_bw = 0;            // 0 なら自動選択
    bool verify = true;                // 参照解を求めて精度を判定するか（位置引数の verify．OZ_NOVERIFY が優先）
    bool fp32_dest_acc = false, dst_full_sync = false;
    tt::tt_metal::MathFidelity fid = tt::tt_metal::MathFidelity::HiFi4;
};
// 2〜4．不正なら理由を出して終了コード 2 で止める
static Bf16Args bf16_args(int argc, char** argv) {
    Bf16Args a;
    // 2. 位置引数．余分な位置引数は黙って読み捨てない（引数の追加・位置ずれで別の構成を測らないように）
    if (!oz_env::check_argc(argc, 10, "M N K B warmup iters verify in0_block_w fp32_dest_acc dst_full_sync")) std::exit(2);
    const oz_drv::Dims d = oz_drv::read_dims(argc, argv, {2560, 2816, 2048}, "gemm_bf16");
    a.M = d.M; a.N = d.N; a.K = d.K;
    a.B = oz_env::arg_u32(argc, argv, 4, 1, 0, UINT32_MAX, "B");
    oz_drv::read_counts(argc, argv, 5, 2, 10, a.warmup, a.iters, 0u);
    // 真偽の引数は 0 か 1 に限る（位置が 1 つずれて 10 などが入ると，表示は verify=10 のまま 1 として通ってしまう）
    a.verify = oz_env::arg_u32(argc, argv, 7, 1, 0, 1, "verify") != 0;
    a.forced_bw = oz_env::arg_u32(argc, argv, 8, 0, 0, UINT32_MAX, "in0_block_w");
    a.fp32_dest_acc = oz_env::arg_u32(argc, argv, 9, 0, 0, 1, "fp32_dest_acc") != 0;
    a.dst_full_sync = oz_env::arg_u32(argc, argv, 10, 0, 0, 1, "dst_full_sync") != 0;
    if (!oz_eval::verify_enabled()) a.verify = false;   // OZ_NOVERIFY（oz_reference.hpp）は位置引数より優先して参照解を省く
    // 3. つまみの値・入力の生成と判定と書き出し・GEMM 段・計時．K との関係（OZ_EXPK < K）もデバイスを開く前に見る．
    //    位置引数の verify=0 も渡して，OZ_DUMP_C との併用を止める（検証を省くと参照解が無く，書き出さないまま 0 で終わるため）
    oz_drv::check_env(oz_knobs::W_BF16, a.K, a.verify);
    a.fid = baseline_fidelity();   // 値は上の preflight が検査済み
    // 4. このドライバに固有の検査．B はバッチ数で，GEMM 段はバッチを扱わない（オペランドの生成も DRAM の確保も
    //    パネルの切り出しも 1 バッチぶんしかない）
    if (a.B != 1) {
        fmt::print(stderr, "ERROR: バッチは未対応である（B={} を指定．B=1 のみ）\n", a.B);
        std::exit(2);
    }
    return a;
}

// オペランドをデバイスを開く前に作り，BF16 へ丸めた値が有限か（FP32・BF16 の範囲の内側か）を見る．範囲の外
// （OZ_CONST=1e100 など）は BF16 で無限大になり，検証を省く実行（verify=0・OZ_NOVERIFY）では精度を見ないので
// 成功（Test Skipped・0）で終わってしまう．検証の有無によらず，ここで止める（参照解はデバイスを開いた後に求める）
static void bf16_operands(const Bf16Args& a, std::vector<double>& Ad, std::vector<double>& Bd) {
    oz_eval::gen_operands(Ad, Bd, a.M, a.K, a.N);
    size_t bad = 0;
    double worst = 0.0;
    for (const auto* v : {&Ad, &Bd})
        for (double x : *v)
            if (!std::isfinite(static_cast<float>(bfloat16((float)x)))) { ++bad; worst = std::max(worst, std::fabs(x)); }
    if (bad) {
        fmt::print(stderr, "ERROR: 入力の {} 要素が BF16 で有限に表せない（最大の絶対値 {:.3e}．FP32・BF16 の最大は約 3.4e38）\n",
                   bad, worst);
        std::exit(2);
    }
}

// ---- 構成探索（デバイスを開いた後．見出しの行もここで出す）----
struct Bf16Plan {
    PanelGrid pg;                      // 出力パネルの分割と最適構成
    uint32_t total_cores = 0;
    double peak_chip_tflops = 0.0;     // 理論ピーク［TFLOP/s］（common/oz_fidelity.hpp の oz_peak_tops）
};
// 構成が取れなければ理由を表示して pg.ok() が偽
static Bf16Plan bf16_plan(std::shared_ptr<distributed::MeshDevice>& md, const Bf16Args& a) {
    Bf16Plan pl;
    // コア数と周波数を実行時に取得して理論ピークを求める
    const auto grid = md->compute_with_storage_grid_size();
    pl.total_cores = grid.x * grid.y;
    const double clk_hz = (double)md->get_clock_rate_mhz() * 1e6;
    pl.peak_chip_tflops = oz_peak_tops(a.fid, clk_hz, pl.total_cores);
    // DST に入るタイル数：BF16 で half-sync なら 8，full-sync なら 16．
    // FP32 累算にすると容量が半減する．
    uint32_t area_cap = a.dst_full_sync ? 16u : 8u;
    if (a.fp32_dest_acc) area_cap /= 2;
    const uint32_t Mt = a.M / TILE_HEIGHT, Kt = a.K / TILE_WIDTH, Nt = a.N / TILE_WIDTH;
    // 出力パネル分割：コアあたりの出力ブロックを有界に保ち，110 コアを使う
    // 大きなサイズでも L1 に収まるようにする．パネルの辺は PANEL_CAP 以下で
    // Mt/Nt を割り切る数から，構成が取れる組を面積の大きい順に探す（common/mcast_gemm.hpp の
    // choose_panels．先頭の候補は「PANEL_CAP 以下の最大の約数」の組なので，その組で構成が取れる形では
    // それを選ぶ）．各パネルは小さな行列全体と同じ最適構成で
    // 走るので，Mt,Nt が PANEL_CAP 以下ならパネルは 1 枚＝分割なしと同一になる．
    const uint32_t PANEL_CAP = 120;
    pl.pg = panel_grid(choose_panels(Mt, Nt, PANEL_CAP, [&](uint32_t pm_, uint32_t pn_) {
        return choose_cfg(pm_, pn_, Kt, grid.y, grid.x, area_cap, a.forced_bw); }), Mt, Nt);

    fmt::print("================ Tenstorrent matmul (BF16, optimized) ================\n");
    fmt::print("Device      : arch={} grid={}x{} ({} cores) clock={} MHz  peak_chip={:.1f} TFLOP/s\n",
               (int)md->arch(), grid.x, grid.y, pl.total_cores,
               md->get_clock_rate_mhz(), pl.peak_chip_tflops);
    fmt::print("Datatype    : bfloat16,  MathFidelity={} (peak /{})  fp32_dest_acc={} dst_full_sync={}\n",
               oz_fidelity_name(a.fid), (int)oz_fidelity_passes(a.fid), a.fp32_dest_acc, a.dst_full_sync);
    fmt::print("Problem     : M={} N={} K={} B={}  (warmup={} iters={} verify={})\n",
               a.M, a.N, a.K, a.B, a.warmup, a.iters, a.verify ? 1 : 0);

    const OptConfig& cfg = pl.pg.g;
    if (!pl.pg.ok()) {
        fmt::print(stderr,
            "ERROR: no valid panel config for any output panel of Mt={} Nt={} tiles (sides <= {}), Kt={}.\n"
            "  Mt and Nt each need a divisor >= 2 that fits the {}x{} core grid, with the per-core block in L1.\n",
            Mt, Nt, PANEL_CAP, Kt, grid.y, grid.x);
        return pl;
    }
    const uint32_t active = cfg.cr * cfg.cc;
    fmt::print("Auto-tuned  : grid {}x{}={} cores ({:.0f}%)  per_core={}x{}  subblock={}x{}\n",
               cfg.cr, cfg.cc, active, 100.0 * active / pl.total_cores,
               cfg.pm, cfg.pn, cfg.sh, cfg.sw);
    if (pl.pg.nbr > 1 || pl.pg.nbc > 1)
        fmt::print("Panels      : {}x{} output panels of {}x{} tiles (L1-bounded)\n",
                   pl.pg.nbr, pl.pg.nbc, pl.pg.pMt, pl.pg.pNt);
    fmt::print("            : in0_block_w={}  num_blocks(K-spill)={}{}\n",
               cfg.bw, cfg.nb,
               cfg.nb == 1 ? "  (spill DISABLED)" : "");
    return pl;
}

// ---- 実行 ----
// 出力（行優先の BF16）と計時の和．Chi・Clo は参照解（検証するときだけ求める）
struct Bf16Out {
    std::vector<bfloat16> result;
    McastGemmRun st;
    std::vector<double> Chi, Clo;
};
static Bf16Out bf16_run(std::shared_ptr<distributed::MeshDevice>& md, const Bf16Args& a, const Bf16Plan& pl,
                        const std::vector<double>& Ad, const std::vector<double>& Bd) {
    const uint32_t M = a.M, N = a.N, K = a.K;
    const uint32_t Mt = M / TILE_HEIGHT, Kt = K / TILE_WIDTH, Nt = N / TILE_WIDTH;
    const PanelGrid& pg = pl.pg;
    Bf16Out o;
    // オペランドと参照解は共通の host_eval から取る．他のドライバと同じ入力・
    // 同じ指標で評価するためである．デバイスが実際に見るのはそれを BF16 に
    // 丸めた値で，出力と中間 CB（c_16・c_24）も BF16 なので，報告される精度には
    // 入力の丸めに加えて出力の丸め・K スピル時の丸め・演算誤差が入る．
    // （A・B はデバイスを開く前に作って範囲を確かめてある．bf16_operands）
    std::vector<bfloat16> av(Ad.size()), bv(Bd.size());
    for (size_t i = 0; i < Ad.size(); ++i) av[i] = bfloat16((float)Ad[i]);
    for (size_t i = 0; i < Bd.size(); ++i) bv[i] = bfloat16((float)Bd[i]);
    if (a.verify) oz_eval::reference_gemm(Ad, Bd, M, K, N, o.Chi, o.Clo);
    av = tilize_nfaces(av, M, K);
    bv = tilize_nfaces(bv, K, N);

    // 出力パネルごとに C[br,bc] = A[br 行] ・ B[bc 列] を，共通の最適構成で計算する．
    // A の行パネルはタイル順で連続だが，B の列パネルは Kt 行ごとに集める必要がある
    // （panel_rows / panel_cols，common/ozaki_gemm_host.hpp）．
    // デバイス実行時間はパネルぶんを合計する（＝デバイスが行った全仕事）．
    o.result.resize((size_t)Mt * Nt * 1024);
    const McastGemmSpec spec = bf16_gemm_spec(pg.g, Kt, a.fid, a.fp32_dest_acc, a.dst_full_sync);
    for (uint32_t br = 0; br < pg.nbr; br++) {
        for (uint32_t bc = 0; bc < pg.nbc; bc++) {
            std::vector<bfloat16> subA = panel_rows(av, br, pg.pMt, Kt);
            std::vector<bfloat16> subB = panel_cols(bv, bc, pg.pNt, Kt, Nt);
            std::vector<bfloat16> panelC;
            // パネル 1 枚の行列積（共有の run_mcast_gemm．端から端までの時間も 1 回測る）
            const McastGemmRun pst = run_mcast_gemm(md, pg.g, spec, pg.pMt, pg.pNt, Kt, subA, subB, panelC,
                                                    a.warmup, a.iters, /*e2e=*/true);
            o.st.timed_s += pst.timed_s;
            o.st.e2e_s += pst.e2e_s;
            for (uint32_t pmt = 0; pmt < pg.pMt; pmt++)
                std::copy(panelC.begin() + (size_t)pmt * pg.pNt * 1024,
                          panelC.begin() + (size_t)(pmt + 1) * pg.pNt * 1024,
                          o.result.begin() + ((size_t)(br * pg.pMt + pmt) * Nt + bc * pg.pNt) * 1024);
        }
    }
    o.result = untilize_nfaces(o.result, M, N);   // タイル順から行優先へ戻す
    return o;
}

// ---- 報告 ----
// 時間・精度・RESULT・VERIFY・テレメトリ・CONFIG の各行を出す．精度の判定と書き出しが通ったかを返す
static bool bf16_report(const Bf16Args& a, const Bf16Plan& pl, const Bf16Out& o,
                        const std::vector<double>& Ad, const std::vector<double>& Bd) {
    const uint32_t M = a.M, N = a.N, K = a.K;
    const OptConfig& cfg = pl.pg.g;
    const uint32_t active = cfg.cr * cfg.cc;
    // 性能は有効演算量 2MNK を実時間で割った値で表す（分割の仕方に依らない）
    const double flops = 2.0 * (double)M * (double)N * (double)K * (double)a.B;
    const double tflops = flops / o.st.timed_s / 1e12;
    const double e2e_tflops = flops / o.st.e2e_s / 1e12;
    const double eff_chip = 100.0 * tflops / pl.peak_chip_tflops;

    fmt::print("---------------------------------------------------------------------\n");
    // 計時の方式（steady・cold）は CONFIG 行の timing= にある
    // ホストの時計で投入から同期までを測った値（デバイスのサイクルカウンタではない．ホストからの転送は含まない）
    fmt::print("Compute time: {:.4f} ms/iter (host clock, dispatch to sync, excl. transfers; sum over panels)\n",
               o.st.timed_s * 1e3);
    fmt::print("End-to-end  : {:.4f} ms (incl. H2D+D2H, JIT excluded)\n", o.st.e2e_s * 1e3);
    // 「Performance : <TFLOP/s> TFLOP/s」は照合が空白まで固定した正規表現で読む（素の BF16 GEMM の基準性能の出どころ：
    // scripts/verify/check_paper_numbers.py・scripts/rules/probe_checks.py・scripts/lib/legacy_logs.py．同じ量は RESULT の 12 列目（RESULT と
    // 印を含めて数える）にもある）．表示を整えるときは照合の正規表現と一緒に直す
    fmt::print("Performance : {:.2f} TFLOP/s  (end-to-end {:.2f} TFLOP/s)\n", tflops, e2e_tflops);
    fmt::print("Efficiency  : {:.1f}% of chip peak ({:.1f} TFLOP/s)\n", eff_chip, pl.peak_chip_tflops);

    // 到達精度の下限．壊れた構成（設定ミス・データ移動の誤り）を拾うためのもので，構成ごとに
    // 期待値が違う：FP32 累算（fp32_dest_acc=1．出力と K ブロック間の退避は BF16）は 3520^3 で 7.4，
    // 7040^3 で 6.9，14080^3 で 6.5 bit（scripts/measure/bf16_accuracy.sh．論文の表の「単体 BF16 GEMM」と
    // PDP 版の本文の値）なので 6 bit，BF16 累算（fp32_dest_acc=0．DST も BF16）は 3520^3 で 4.1，14080^3 で 2.2 bit なので
    // 2 bit を割ったら落とす（下限ちょうどは通す）．どちらも K とともに下がるので，14080 より長い K では下限に近づく．
    // 落ちても RESULT・VERIFY・CONFIG 行は出し，デバイスを閉じてから 1 で終わる
    // （TT_FATAL で異常終了すると，行もデバイスの後始末も残らない）．
    bool acc_ok = true, dump_ok = true;
    if (a.verify) {
        std::vector<double> Cd((size_t)M * N);
        for (size_t i = 0; i < Cd.size(); ++i) Cd[i] = (double)static_cast<float>(o.result[i]);
        auto acc = oz_eval::accuracy(Cd, o.Chi, o.Clo);
        oz_eval::print_accuracy(acc);
        // OZ_DUMP_C（調査用．尾崎の各ドライバと同じ形式）．書けなければ失敗とする（デバイスを閉じてから 1 で終わる）
        dump_ok = oz_eval::dump_result(Cd, o.Chi, o.Clo, Ad, Bd, M, K, N);
        const double min_bits = a.fp32_dest_acc ? 6.0 : 2.0;
        acc_ok = std::isfinite(acc.rel_fro) && std::isfinite(acc.ebits) && acc.ebits >= min_bits;
        if (!acc_ok)
            fmt::print(stderr, "ERROR: BF16 GEMM の到達精度が {} bit で，この構成の下限 {} bit を割った"
                               "（fp32_dest_acc={}）\n", acc.ebits, min_bits, a.fp32_dest_acc);
    } else {
        fmt::print("Accuracy    : skipped\n");
    }

    // RESULT 行は照合が欄の位置で読む（scripts/lib/oz_logs.py の RESULT_SCHEMA）．欄を足すなら CONFIG に足す
    fmt::print("RESULT,{},{},{},{},{},{},{},{},{:.6e},{:.4f},{:.4f},{:.2f},{:.2f}\n",
               M, N, K, a.B, active, cfg.bw, cfg.nb,
               (uint32_t)a.fp32_dest_acc * 10 + (uint32_t)a.dst_full_sync,
               flops, o.st.timed_s * 1e3, o.st.e2e_s * 1e3, tflops, eff_chip);
    fmt::print("VERIFY,gemm_bf16,{}\n", a.verify ? (acc_ok ? "pass" : "fail") : "skipped");
    // 計時区間のクロック・電流・電力・温度（oz_timing.hpp の print_telem）と計時の方式（CONFIG の timing=．timing_desc）
    std::fflush(stdout);
    oz_eval::print_telem();
    // 設定の記録（尾崎の CONFIG 行と同じく OzLine で key=value を名前で足す．検査側は位置ではなく名前で読む）．
    // timing_scope=gemm_only：RESULT の時間はデバイス上の行列積の実行時間だけで，パネルごとに別々に
    // ウォームアップして測った時間の和である（ホストでのパネルの切り出し・転送・書き戻しは含まない）．
    // 尾崎の各経路の DRAM->DRAM（変換・スライス・再構成まで含むデバイスの段の和）とは計時の範囲が違う．
    // input=・seed= は入力の作り方（oz_eval::input_desc．尾崎の CONFIG と同じ欄．判定の基準はこのドライバ
    // 固有（上の min_bits）なので min_bits 欄は付けない），timing= からは計時の方式（timing_desc）．
    // 末尾の knobs は既定から外れたアブレーションのつまみ（つまみの表の R_KNOBS のうちこのドライバが読むもの．尾崎の
    // CONFIG と同じ書き方．今は該当が無く none），gen は入力の生成の版（oz_driver.hpp の add_common_tail と同じ欄）．
    // L1 の予算はこのドライバの構成探索が読まない（OZ_L1_BUDGET_KB は入口で拒む）ので l1_budget_kb 欄は付けない
    OzLine cfgl("CONFIG,gemm_bf16");
    cfgl.kv("fp32_dest_acc", a.fp32_dest_acc ? 1 : 0).kv("fidelity", oz_fidelity_name(a.fid)).kv("bw", cfg.bw)
        .kv("nb", cfg.nb).kv("panels", oz_drv::axb(pl.pg.nbr, pl.pg.nbc)).flag("verified", a.verify)
        .kv("dst_full_sync", a.dst_full_sync ? 1 : 0).kv("grid", oz_drv::axb(cfg.cr, cfg.cc))
        .kv("shape", oz_drv::axb(cfg.pm, cfg.pn)).kv("sub", oz_drv::axb(cfg.sh, cfg.sw)).kv("gemm_split", 1)
        .kv("timing_scope", "gemm_only").raw(oz_eval::input_desc(oz_eval::kDefaultSeed, false))
        .raw(oz_eval::timing_desc())
        .kv("knobs", oz_knobs::record(oz_knobs::W_BF16, oz_knobs::R_KNOBS)).kv("gen", oz_eval::kGenVersion);
    cfgl.print();
    return acc_ok && dump_ok;   // VERIFY 行は精度だけ．書き出しの失敗は終了コードへ
}

int main(int argc, char** argv) {
    // 入口の並びは尾崎の 2 つのドライバと同じ（common/oz_driver.hpp の冒頭）．
    // 1. 環境だけ．精度の下限はこのドライバ固有の固定値（下の min_bits．fp32_dest_acc=1 で 6 bit，0 で 2 bit）なので，
    //    OZ_MIN_BITS は効かない．黙って受けると，その基準で判定したつもりになる．L1 の予算 OZ_L1_BUDGET_KB も構成探索が
    //    見ないので止める（冒頭の使い方）
    oz_drv::reject_env_gemm(oz_knobs::W_BF16, "gemm_bf16",
                            "判定の下限はこのドライバ固有の固定値（fp32_dest_acc=1 で 6 bit，0 で 2 bit）");
    oz_knobs::reject_named("OZ_L1_BUDGET_KB", "gemm_bf16",
                          "構成探索は tt-metal の行列積の例の判定で選び，L1 の予算を見ない");
    const Bf16Args a = bf16_args(argc, argv);
    std::vector<double> Ad, Bd;
    bf16_operands(a, Ad, Bd);

    bool pass = true;
    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        const Bf16Plan pl = bf16_plan(mesh_device, a);
        if (!pl.pg.ok()) {
            mesh_device->close();
            return 2;
        }
        const Bf16Out o = bf16_run(mesh_device, a, pl, Ad, Bd);
        pass &= bf16_report(a, pl, o, Ad, Bd);
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n{}\n", e.what());
        throw;
    }

    // 検証を省いた実行（verify=0・OZ_NOVERIFY）は精度を判定していないので Test Passed と書かない（終了コードは 0）．
    // RESULT 行に精度の欄は無い
    std::fflush(stdout);
    return oz_drv::finish(pass, a.verify);
}
