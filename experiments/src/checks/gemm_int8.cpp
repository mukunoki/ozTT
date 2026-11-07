// SPDX-License-Identifier: Apache-2.0
//
// Tenstorrent Blackhole 上の INT8×INT8→INT32 行列積
// ---------------------------------------------------------------------------
// 尾崎スキームの INT8 経路が土台にする，整数行列積そのものの実装と検証．論文の値は出さない検査なので，組は CHECKS
// （src/Makefile．8197a0b までは src/baseline/ の基準実装の組に置いていた）．入口・CONFIG 行の組み方は基準実装の 2 本と同じ．
// プログラムは尾崎の各経路と同じ build_mcast_gemm（common/mcast_gemm.hpp）で組み，計算カーネルは
// Int32 の部分和を無誤差に扱える common/kernels/bmm_int32_spill.cpp を使う（標準の
// カーネルはデータ形式を再設定しないので c_24 の読み戻しで部分和が壊れる）．
//
// 整数経路は INT32 で無誤差な累算を行う（整数値として誤差がない）．どちらの mode も上の
// bmm_int32_spill.cpp を使うので K の分け方だけが違い，結果はどちらも無誤差である
// （mode 1 は K 全体を Int32 に積むので K*R^2 < 2^31 が要る．main が検査する）．
// 判定は符号-絶対値の出力語を復号し（mode 0 はホストで int64 に足し込み）int64 の参照と整数値で比べる
// もので，出力語のビット列の一致ではない（符号-絶対値の +0 と -0 は同じ 0 として通る）．mode 0 と 1 の
// 生の出力語どうしの一致もこの判定からは言えない．
// mode 0 は大きな K を K ブロック 1 つぶんの塊に分けて 1 プログラム 1 塊で実行し，
// INT32 の部分和をホスト側で足す．デバイス上でスピルを起こさない保守的な経路で，
// スピルの無誤差性に頼らずに済む代わりにプログラムを塊の数だけ起動する．
// mode 1 は K 全体を 1 プログラムで回してデバイス上でスピルさせる．起動が 1 回で
// 済むぶん約 2.4 倍速く，尾崎スキームの実装が採るのはこちらである．
//
// デバイス側の int8 入力と int32 出力は符号-絶対値表現である．
//
// 使い方: gemm_int8 [M] [N] [K] [R] [warmup] [iters] [mode]
//         既定値:   1280 1408 2048 64 1 5 0   （mode 0 = ホスト分割で無誤差）
//         M N K は正の 32 の倍数，warmup は 0..1000000（0 でも行列積の JIT は計時に入らない），iters は 1..1000000
//         （common/oz_driver.hpp の read_dims・read_counts），mode は 0 か 1．
//         R は 0..127（入力は [-R,R] の整数．符号-絶対値の int8 は -128 を表せない）．
//         OZ_NOVERIFY を立てると int64 の参照計算と一致判定を省く（スループットだけ見るとき．RESULT の一致の欄は nan，
//         最後の行は "Test Skipped"）．引数は 7 個まで（多ければデバイスを開く前に 2 で止める）．
//         入力は自前の整数乱数で結果も書き出さないので，入力の作り方（OZ_PHI など）・OZ_MIN_BITS・OZ_DUMP_C は
//         受け付けない（oz_drv::reject_env_gemm．黙って無視すると，その入力や基準で測ったつもりになる）．
//         OZ_NO_BLOCKMM を立てると計算カーネルの発行を matmul_block ではなく matmul_tiles にする（CONFIG の blockmm=0）．
// 計時の範囲（CONFIG の timing_scope=gemm_only）：RESULT の時間と TOP/s は行列積のデバイス実行時間だけで，
// mode 0 はプログラム（K ブロック）ごとに別々にウォームアップして測った時間の和である．mode 0 のホストでの
// K ブロックの切り出し・タイル化・転送・int64 への足し込みは含まない．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include "common/mcast_gemm.hpp"    // build_mcast_gemm・run_mcast_gemm（ozaki_gemm_host.hpp の nf・dec_sm32・choose_cfg も）
#include "common/host_eval.hpp"     // 整数の参照（int_reference_gemm）・完全一致（exact_check）・計時
#include "common/oz_driver.hpp"     // 入口の検査の並び（oz_drv::reject_env_gemm・read_dims・read_counts・check_env）・最後の行（finish）・CONFIG 行（OzLine）
#include "common/oz_fidelity.hpp"   // MathFidelity の上書き・表示名・パス数・理論ピーク（基準実装 3 つと尾崎で共通）

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// MathFidelity の既定は HiFi4（無誤差性の確認が目的なので落とす意味がない）．OZ_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4
// で上書きできる．LoFi は上位の区画どうしの積 x_hi*y_hi しか累積せず，絶対値の下位ビット（x_lo・y_lo の区画）を
// 含む積が欠けて無誤差でなくなる（論文の表 tab:field．値の幅で丸まるのではない）が，行列エンジンの天井が
// オペランド幅で決まるのか発行レートで決まるのかを切り分けるために，1 バイト/要素の入力で
// スループットだけを測りたいときに使う（論文の fidelity の掃引）．
// 入力の整数乱数の seed（[-R,R] の一様．CONFIG の input=・seed= に残す）
constexpr uint32_t kInt8Seed = 123;

// 共有の bmm_int32_spill.cpp の仕様（common/mcast_gemm.hpp の spill_gemm_spec）への指定．
// 構成探索の判定（accept）とプログラムの組み立て（gemm_int8）が同じものを使うように 1 か所に置く．
// 融合（CA_FUSE）と対ループ（CA_PAIRLOOP）は FP32 の部分積専用で，INT8 では組めない（spill_gemm_spec が止める）．
static SpillOpts gi8_spill_opts(tt::tt_metal::MathFidelity fid, bool blockmm) {
    SpillOpts so;
    so.in_fmt = tt::DataFormat::Int8;
    so.acc_fmt = tt::DataFormat::Int32;
    so.fidelity = fid;
    so.blockmm = blockmm;   // CA_BLOCKMM＝matmul_block（OZ_NO_BLOCKMM で切れる．main が読んで CONFIG の blockmm= に残す）
    so.out_shared = false;  // c_16 と c_24 は別々の Int32 の領域
    return so;
}
// 構成 c（K 方向 Kt タイルを 1 プログラムで回す）で実際に作る CB の合計が L1 の予算に収まるか．
// 構成探索（choose_cfg．in_tile を渡さない tt-metal の例の判定 example_max_block_dim）は c_16 と c_24 が BF16 の 1 面を
// 共有すると見積もるが，ここでは Int32 の 2 面を別々に作るので出力側は見積りの 4 倍になる．
// 見積りで選ぶ最大の bw が実際には収まらない形（10x11 コア・M=4160・N=4576・K=128 の mode 1．コアあたり
// 13x13 で bw=4 は 1560 KiB，bw=1 なら 1404 KiB）で小さい bw へ降りられるように，choose_cfg の accept に渡す．
// 見積りで選んだ構成が実際にも収まる形では，選ばれる構成は変わらない（accept は負けた候補の bw を
// 下げることはあっても，勝った候補はそのまま通す）．
static bool gi8_fits(const OptConfig& c, uint32_t Kt, const SpillOpts& so) {
    return mcast_gemm_l1_bytes(c, spill_gemm_spec(c, Kt, so)) <= (uint64_t)oz_l1_budget();
}

// INT8 行列積を 1 プログラム実行する．mode 0 では K = 32*g.bw の 1 ブロックぶん
// （num_blocks = 1），mode 1 では K 全体（num_blocks = g.nb）を渡す．
// 戻り値は温めの後に計時した 1 反復あたりの秒数（温め方は OZ_TIMING．既定の cold では定常状態ではない．
// oz_timing.hpp の「計時」）．Cint に行優先 M*N の int64 部分和を書く．
static double gemm_int8(std::shared_ptr<distributed::MeshDevice>& md, const OptConfig& g, const SpillOpts& so,
                        const std::vector<int8_t>& A_t, const std::vector<int8_t>& B_t,
                        uint32_t M, uint32_t N, uint32_t K, std::vector<int64_t>& Cint,
                        uint32_t warmup, uint32_t iters) {
    const uint32_t Mt = M / TILE_HEIGHT, Kt = K / TILE_WIDTH, Nt = N / TILE_WIDTH;
    // mode 0 は呼び出し側が K を bw 1 つぶんに切るので 1．mode 1 は K 全体を渡すので g.nb（>1 もある）
    const uint32_t num_blocks = Kt / g.bw;

    // プログラムの組み立てと実行は尾崎の各経路・基準実装と共有する（common/mcast_gemm.hpp の spill_gemm_spec・
    // build_mcast_gemm・run_mcast_gemm）．計算カーネルは Int32 の部分和を無誤差に扱える bmm_int32_spill.cpp で，
    // 標準のカーネルは _with_dt の付かない再設定呼び出しを使うため，c_24（Int32）を読むときも
    // アンパッカが INT8 設定のままになり部分和が壊れる．
    // c_24 は K ブロック間の部分和の CB（コアあたり出力ブロック 1 つぶん，Int32）で，ここでは
    // c_16 とは別に確保する（尾崎の経路は共有させる）．mode 1（num_blocks>1）ではパッカが
    // 部分和を c_24 上で L1 累積し，最後のブロックで 1 回だけ読み戻す．mode 0（num_blocks==1）では
    // その分岐は通らないが，CB 自体は存在させておく必要がある．
    // Int32 の CB に UnpackToDestFp32 を設定しても，一次資料（jit_build/data_format.cpp）を読む限り
    // src==Float32 のときにしか作用せず，Int32 では何もしない．一方 Blackhole では Int32 → DST の経路が既定の
    // 有効経路として明記されている（cunpack_common.h）ので，c_24 を Int32 のまま読み戻せる．
    McastGemmSpec spec = spill_gemm_spec(g, Kt, so);
    // main は実際に作る CB が予算に収まる構成だけを選ぶ（gi8_fits を choose_cfg の accept に渡す）ので，
    // ここは呼び出し側の誤りを CB の確保より前に止める安全網である．
    mcast_check_l1(g, spec, "gemm_int8");
    // 共有の bmm_int32_spill.cpp は num_blocks を実行時引数 CR_NUM_BLOCKS から読む
    spill_check_run(num_blocks, false, 0u);
    spec.compute_rt = spill_compute_rt(num_blocks);
    std::vector<int32_t> Ctiled;
    const double timed = run_mcast_gemm(md, g, spec, Mt, Nt, Kt, A_t, B_t, Ctiled, warmup, iters).timed_s;
    // タイル順から行優先へ戻し，符号-絶対値を復号して Cint に足し込む
    for (uint32_t tr = 0; tr < Mt; tr++)
        for (uint32_t tc = 0; tc < Nt; tc++) {
            const int32_t* t = &Ctiled[((size_t)tr * Nt + tc) * 1024];
            for (uint32_t r = 0; r < 32; r++)
                for (uint32_t c = 0; c < 32; c++)
                    Cint[(size_t)(tr * 32 + r) * N + (tc * 32 + c)] += dec_sm32(t[nf(r, c)]);
        }
    return timed;
}

// ---- 入口（デバイスを開く前．並びは尾崎の 2 つのドライバと同じ．common/oz_driver.hpp の冒頭の 1〜4）----
// 位置引数（M N K R warmup iters mode）・MathFidelity・計算カーネルの組み方
struct I8Args {
    uint32_t M = 0, N = 0, K = 0, warmup = 0, iters = 0;
    int R = 64;                        // 入力は [-R,R] の整数
    // mode 0 はホスト側で K を分割してスピルを起こさない経路（既定）．
    // mode 1 はデバイス上でスピルさせる経路で，約 2.4 倍速い．どちらも
    // common/kernels/bmm_int32_spill.cpp を使い，復号した整数値はどちらも int64 の参照と一致する
    // （生の出力語どうしを比べる検査ではない）．
    int mode = 0;
    bool verify = true;                // int64 の参照と一致判定を行うか（OZ_NOVERIFY でなければ真）
    tt::tt_metal::MathFidelity fid = tt::tt_metal::MathFidelity::HiFi4;
    bool blockmm = true;               // 計算カーネルの発行方式（尾崎の各経路と同じく OZ_NO_BLOCKMM が立っていれば matmul_tiles）
};
// 2〜4．不正なら理由を出して終了コード 2 で止める
static I8Args i8_args(int argc, char** argv) {
    I8Args a;
    // 2. 位置引数．余分な位置引数は黙って読み捨てない（引数の追加・位置ずれで別の構成を測らないように）
    if (!oz_env::check_argc(argc, 7, "M N K R warmup iters mode")) std::exit(2);
    const oz_drv::Dims d = oz_drv::read_dims(argc, argv, {1280, 1408, 2048}, "gemm_int8");
    a.M = d.M; a.N = d.N; a.K = d.K;
    // 符号-絶対値の int8 は ±127 まで．128 以上は (int8_t) で折り返し，-128 は
    // 符号-絶対値へ変換できないので，参照と食い違う原因が入力側に入る．
    a.R = (int)oz_env::arg_u32(argc, argv, 4, 64u, 0u, 127u, "R");
    oz_drv::read_counts(argc, argv, 5, 1, 5, a.warmup, a.iters, 0u);
    a.mode = (int)oz_env::arg_u32(argc, argv, 7, 0u, 0u, 1u, "mode");
    // 3. つまみの値（OZ_FIDELITY・OZ_L1_BUDGET_KB など）・GEMM 段・計時（入力の生成と判定と書き出しは読まない）
    a.verify = oz_eval::verify_enabled();
    oz_drv::check_env(oz_knobs::W_I8, a.K, a.verify);
    a.fid = baseline_fidelity();   // 値は上の preflight が検査済み
    // 4. このドライバに固有の検査．mode 1 は K 全体をデバイス上で Int32（符号-絶対値の絶対値部 31 bit）に積むので，|C| <= K*R^2 が 2^31 未満で
    // なければならない（R=127 なら K <= 133,120．超えると溢れて，OZ_NOVERIFY では壊れた値のまま終わる）．
    // mode 0 は K ブロック 1 つぶんずつ Int32 で返してホストの int64 で足すので，1 ブロックぶんで足りる．
    if (a.mode == 1 && (uint64_t)a.K * (uint64_t)a.R * (uint64_t)a.R >= (1ull << 31)) {
        fmt::print(stderr, "ERROR: mode 1 は K*R^2 < 2^31 でなければならない（K={} R={} で {}）\n",
                   a.K, a.R, (uint64_t)a.K * a.R * a.R);
        std::exit(2);
    }
    a.blockmm = gemm_blockmm();
    return a;
}

// ---- 構成探索（デバイスを開いた後．見出しの行もここで出す）----
struct I8Plan {
    OptConfig g;                       // K 全体を 1 プログラムで回す構成（mode 0 の K ブロックの構成は i8_run が引き直す）
    double peak_chip = 0.0;            // 理論ピーク［TOP/s］（common/oz_fidelity.hpp の oz_peak_tops）
    uint32_t budget = 0;               // 形状探索の L1 の予算［B］（OZ_L1_BUDGET_KB．CONFIG の l1_budget_kb）
};
// 構成が取れなければ理由を表示して g.ok が偽
static I8Plan i8_plan(std::shared_ptr<distributed::MeshDevice>& md, const I8Args& a, const SpillOpts& so) {
    I8Plan pl;
    const auto gd = md->compute_with_storage_grid_size();
    const double clk = (double)md->get_clock_rate_mhz() * 1e6;
    // 理論ピーク［演算数/サイクル/コア］は fidelity で決まる（4096 は積和を 2 演算として
    // 数えた値 2*8*16*16 なので，さらに 2 倍しない）．既定は HiFi4 = 1024．
    pl.peak_chip = oz_peak_tops(a.fid, clk, (uint32_t)(gd.x * gd.y));  // TOP/s
    const uint32_t Mt = a.M / 32, Nt = a.N / 32, Kt = a.K / 32;
    // 選び方は tt-metal の例の判定（in_tile を渡さない．example_max_block_dim）のまま，実際に作る CB が予算に収まる
    // 候補だけを受け付ける（gi8_fits）
    pl.budget = oz_l1_budget();
    pl.g = choose_cfg(Mt, Nt, Kt, gd.y, gd.x, 4u, 0u, 0u, 0u, pl.budget,
                      [&](const OptConfig& c) { return gi8_fits(c, Kt, so); });
    const OptConfig& g = pl.g;

    fmt::print("============== INT8xINT8->INT32 GEMM (multi-core) ==============\n");
    fmt::print("Device  : grid={}x{} ({} cores) {} MHz  int8 {} peak={:.1f} TOP/s\n",
               gd.x, gd.y, gd.x * gd.y, md->get_clock_rate_mhz(),
               oz_fidelity_name(a.fid), pl.peak_chip);
    if (!g.ok) {
        // 条件は src/README.md「対応する形状」：Mt・Nt がそれぞれコア格子に収まる 2 以上の約数を持ち，
        // その分割でブロックが L1 に収まること
        fmt::print(stderr, "ERROR: no valid cfg for Mt={} Nt={} Kt={} (Mt and Nt each need a divisor >= 2 "
                           "that fits the {}x{} core grid, with the per-core block fitting in L1)\n",
                   Mt, Nt, Kt, gd.y, gd.x);
        return pl;
    }
    fmt::print("Cfg     : {}x{}={} cores per_core={}x{} subblk={}x{} in0_block_w={}t num_blocks={} mode={}\n",
               g.cr, g.cc, g.cr * g.cc, g.pm, g.pn, g.sh, g.sw, g.bw, g.nb,
               a.mode == 1 ? "on-device-spill" : "host-split-K");
    return pl;
}

// ---- 実行 ----
struct I8Out {
    bool ok = false;                   // mode 0 の K ブロックの構成が取れたか（取れなければ理由を表示済み）
    OptConfig gp;                      // 1 プログラムの構成（mode 0 は K ブロック 1 つぶんで引き直したもの）
    uint32_t nprog = 0;                // プログラムの本数（mode 1 は 1，mode 0 は K ブロックの数）
    double tot = 0.0;                  // 行列積のデバイス実行時間の和［s/iter］
    std::vector<int64_t> C, Cg;        // 復号した出力と int64 の参照（検証するときだけ）
};
static I8Out i8_run(std::shared_ptr<distributed::MeshDevice>& md, const I8Args& a, const I8Plan& pl, const SpillOpts& so) {
    const uint32_t M = a.M, N = a.N, K = a.K, Mt = M / 32, Nt = N / 32, Kt = K / 32;
    const OptConfig& g = pl.g;
    const uint32_t bw = g.bw, nchunks = Kt / bw;
    I8Out o;
    // 入力は [-R,R] の整数．尾崎スキームの INT8 経路が扱うスライス値と同じ性質にする
    std::mt19937 rng(kInt8Seed);
    std::uniform_int_distribution<int> dist(-a.R, a.R);
    std::vector<int8_t> A((size_t)M * K), Bm((size_t)K * N);
    for (auto& v : A) v = (int8_t)dist(rng);
    for (auto& v : Bm) v = (int8_t)dist(rng);

    // int64 による無誤差の参照値（OZ_NOVERIFY のときは作らない．oz_reference.hpp の int_reference_gemm）
    if (a.verify) oz_eval::int_reference_gemm(A, Bm, M, K, N, o.Cg);

    o.C.assign((size_t)M * N, 0);
    o.gp = g;
    if (a.mode == 1) {
        // K 全体を 1 つのプログラムで処理する．カーネルがデバイス上で
        // Int32 の c_24 に部分和をスピル／読み戻しする（num_blocks = g.nb）
        auto At = tilize_sm8(A, M, K);
        auto Bt = tilize_sm8(Bm, K, N);
        o.tot = gemm_int8(md, g, so, At, Bt, M, N, K, o.C, a.warmup, a.iters);
        o.nprog = 1;
    } else {
        // ホスト分割：K ブロック 1 つずつ実行し，Int32 の部分和をホストで足す
        // Kt ではなく bw を渡して構成を引き直す：1 プログラムが K ブロック 1 個ぶん
        // だけを見る形にしたいからである．nb != 1 だとデバイス上でスピルが起き，
        // 無誤差にするためにホストで部分和を足すという前提が崩れるので弾く．
        const auto gd = md->compute_with_storage_grid_size();
        OptConfig gc = choose_cfg(Mt, Nt, bw, gd.y, gd.x, 4u, 0u, 0u, 0u, pl.budget,
                                  [&](const OptConfig& c) { return gi8_fits(c, bw, so); });
        if (!gc.ok || gc.nb != 1) { fmt::print(stderr, "ERROR: chunk cfg nb={}\n", gc.nb); return o; }
        o.gp = gc;
        const uint32_t Kc = bw * 32;
        for (uint32_t ch = 0; ch < nchunks; ch++) {
            std::vector<int8_t> Asl((size_t)M * Kc), Bsl((size_t)Kc * N);
            for (uint32_t i = 0; i < M; i++)
                for (uint32_t k = 0; k < Kc; k++)
                    Asl[(size_t)i * Kc + k] = A[(size_t)i * K + (ch * Kc + k)];
            for (uint32_t k = 0; k < Kc; k++)
                for (uint32_t j = 0; j < N; j++)
                    Bsl[(size_t)k * N + j] = Bm[(size_t)(ch * Kc + k) * N + j];
            auto At = tilize_sm8(Asl, M, Kc);
            auto Bt = tilize_sm8(Bsl, Kc, N);
            o.tot += gemm_int8(md, gc, so, At, Bt, M, N, Kc, o.C, a.warmup, a.iters);
        }
        o.nprog = nchunks;
    }
    o.ok = true;
    return o;
}

// ---- 報告 ----
// 一致・スループット・RESULT・VERIFY・テレメトリ・CONFIG の各行を出す．一致したか（判定を省いたなら真）を返す
static bool i8_report(const I8Args& a, const I8Plan& pl, const I8Out& o) {
    const uint32_t M = a.M, N = a.N, K = a.K;
    const OptConfig& g = pl.g;
    const OptConfig& gp = o.gp;
    // 相対誤差ではなく整数値の完全一致で判定する：整数積が 1 bit でも落ちれば
    // 尾崎スキームの前提（無誤差累算）が崩れるからである
    bool exact = true;
    const double ops = 2.0 * (double)M * N * K;   // 演算数（積和を 2 演算と数える）
    const double tops = ops / o.tot / 1e12;
    fmt::print("---------------------------------------------------------------\n");
    if (a.verify) {
        auto ex = oz_eval::exact_check(o.C, o.Cg);
        const uint64_t bad = (uint64_t)ex.mismatches;
        const int64_t maxabs = (int64_t)ex.max_abs_golden;
        exact = (bad == 0);
        // 比べるのは復号した整数値（出力語のビット列ではない）
        fmt::print("Exactness : {} (mismatch {}/{}, max|golden|={})\n",
                   exact ? "EXACT vs int64 golden (decoded integer values match)" : "MISMATCH", bad, o.C.size(),
                   (long long)maxabs);
    } else {
        // 判定していないので EXACT とは書かない（selftest の exact が拾わないように）
        fmt::print("Exactness : skipped (OZ_NOVERIFY) -- 一致は判定していない\n");
    }
    fmt::print("Throughput: {:.2f} TOP/s = {:.1f}% of int8 peak  ({} program(s), {:.3f} ms/iter total)\n",
               tops, 100.0 * tops / pl.peak_chip, o.nprog, o.tot * 1e3);
    // RESULT 行は照合が欄の位置で読む（scripts/lib/oz_logs.py の RESULT_SCHEMA）．欄を足すなら CONFIG に足す．
    // 最後の欄は一致なら 1，不一致なら 0，判定を省いたなら nan（参照が無いと求まらない欄．尾崎・基準実装の RESULT の
    // 精度欄と同じ書き方．2026-10-21 より前のログは NA）
    fmt::print("RESULT,int8,{},{},{},{},{},{:.4f},{:.4f},{}\n",
               M, N, K, g.cr * g.cc, o.nprog, o.tot * 1e3, tops,
               a.verify ? (exact ? "1" : "0") : "nan");
    fmt::print("VERIFY,gemm_int8,{}\n", a.verify ? (exact ? "pass" : "fail") : "skipped");
    // 設定の記録（尾崎の CONFIG 行と同じく OzLine で key=value を名前で足す．検査側は位置ではなく名前で読む）．
    // bw・nb・grid・shape・sub は 1 プログラムの構成（mode 0 は K ブロック 1 つぶん），
    // programs はプログラムの本数，panels は出力パネル（このドライバは切らないので常に 1x1）．
    // blockmm は計算カーネルの実際の発行方式（1=matmul_block，0=matmul_tiles．OZ_NO_BLOCKMM）．
    // timing_scope=gemm_only は冒頭の「計時の範囲」（ホストでの切り出し・転送・累積を含まない）．
    // input=・seed= は入力の作り方（尾崎・基準実装の CONFIG と同じ欄名．このドライバは自前の [-R,R] の整数乱数で，
    // 判定は int64 の参照との完全一致なので min_bits 欄は無い），timing= からは計時の方式（timing_desc）．
    // 末尾の knobs は既定から外れたアブレーションのつまみ（つまみの表の R_KNOBS のうちこのドライバが読むもの．尾崎の
    // CONFIG と同じ書き方．OZ_NO_BLOCKMM は blockmm 欄にもある），l1_budget_kb は形状探索の L1 の予算［KiB］
    // （oz_driver.hpp の add_common_tail と同じ欄）．入力は gen_operands で作らないので gen 欄は付けない
    std::fflush(stdout);
    oz_eval::print_telem();   // 計時区間のクロックなど（oz_timing.hpp）．計時の方式は CONFIG の末尾の timing=
    OzLine cfg("CONFIG,gemm_int8");
    cfg.kv("fp32_dest_acc", 1).kv("fidelity", oz_fidelity_name(a.fid)).kv("bw", gp.bw).kv("nb", gp.nb)
       .kv("panels", "1x1").flag("verified", a.verify).kv("mode", a.mode).kv("R", a.R).kv("programs", o.nprog)
       .kv("grid", oz_drv::axb(gp.cr, gp.cc)).kv("shape", oz_drv::axb(gp.pm, gp.pn)).kv("sub", oz_drv::axb(gp.sh, gp.sw))
       .kv("gemm_split", 1).flag("blockmm", a.blockmm).kv("timing_scope", "gemm_only")
       .kv("input", "int_uniform:" + std::to_string(a.R)).kv("seed", kInt8Seed).raw(oz_eval::timing_desc())
       .kv("knobs", oz_knobs::record(oz_knobs::W_I8, oz_knobs::R_KNOBS)).kv("l1_budget_kb", pl.budget / 1024u);
    cfg.print();
    return exact;
}

int main(int argc, char** argv) {
    // 入口の並びは尾崎の 2 つのドライバと同じ（common/oz_driver.hpp の冒頭）．
    // 1. 環境だけ．入力は自前の整数乱数で，判定は int64 の参照との完全一致なので，入力の作り方・精度の基準・結果の書き出しの
    //    つまみは効かない
    oz_drv::reject_env_gemm(oz_knobs::W_I8, "gemm_int8", "入力の作り方・精度の基準・結果の書き出しを持たない");
    const I8Args a = i8_args(argc, argv);
    const SpillOpts so = gi8_spill_opts(a.fid, a.blockmm);

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        const I8Plan pl = i8_plan(md, a, so);
        if (!pl.g.ok) { md->close(); return 2; }
        const I8Out o = i8_run(md, a, pl, so);
        if (!o.ok) { md->close(); return 2; }
        pass &= i8_report(a, pl, o);
        pass &= md->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed: {}\n", e.what());
        throw;
    }
    // 一致を判定しなかった実行（OZ_NOVERIFY）は Test Passed と書かない（終了コードは 0）
    std::fflush(stdout);
    return oz_drv::finish(pass, a.verify);
}
