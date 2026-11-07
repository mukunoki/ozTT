// SPDX-License-Identifier: Apache-2.0
//
// Tenstorrent Blackhole 上の尾崎スキーム I による GEMM
// ---------------------------------------------------------------------------
// scheme I の 8 形式（BF16 / FP16 / FP32(TF32) / BFP8_b / FP8 / INT8 / BFP4_b / BFP2_b）
// すべてがこの 1 つのソースから作られ，
// Makefile が渡す OZ_FMT_* マクロで切り替わる（scheme II と同じマクロ．論文の ssec:oz1（尾崎スキーム I）・
// sec:case（ハードウェア特性に応じた実装））．
// 基数 β は桁の上限 d の 2 倍に取る．fidelity は d から決まる（d<=32 なら LoFi）．
//   bf16・fp16・fp32・bfp8: 桁 32／基数 64，LoFi
//   fp8 : Fp8_e4m3，桁 16／基数 32，LoFi
//   bfp4: 桁 7／基数 14，bfp2: 桁 1／基数 2，いずれも LoFi
//   int8: Int8→Int32 + 符号-絶対値の UInt8 スライス + HiFi4，桁 127／基数 254
//         （再構成側で 2 本に復号）
// 段の部品は ozaki1/ のヘッダにあり（ozaki1_common.hpp が形式・スケール・GEMM・再構成・スライスを取りまとめる．計画は
// ozaki1_plan.hpp，OZ_FP64 のモデルは ozaki1_autop.hpp，投入の手順は ozaki1_stages.hpp，OZ_TIME_WHOLE は ozaki1_whole.hpp，
// OZ_NO_FUSE_ACC の累積は ozaki1_accum.hpp），デバイスカーネルは ozaki1/kernels/ と両スキーム共通の common/kernels/ にある．
// このファイルは段を並べて計時し，結果を表示する．
//
// 全工程をデバイス上で実行する：FP64 -> triple-float の変換・スケール算出・正規化・
// スライス分解・整数 GEMM・再構成・triple-float -> FP64 の組み立て．ホストは FP64 の
// オペランドをタイル順に並べて渡し，FP64 の結果を並べ戻すだけで，演算は一切しない
// （したがって変換のコストは計時区間の内側に入る）．
//
// 環境変数（つまみ）の名前・分類・読むプログラム・範囲と既定・CONFIG 行の knobs・diag 欄への記録の正本は
// common/oz_knobs.hpp の表で，分類ごとの説明は docs/env_vars.md（両者の名前と分類は tools/knobs_check が突き合わせる．
// ここには一覧を写さない）．このドライバは，計画を変えるつまみを Oz1Knobs に，段の組み方を変える診断と測定モードの
// つまみを Oz1Diag に，どちらも main の冒頭で 1 回だけ読んで配る（ozaki1_plan.hpp．表示だけの OZ_DBG を除く）．
// 値を持つつまみはどれもデバイスを開く前に 1 度読んで検査する（parse_run_config が呼ぶ oz_drv::check_env と，範囲が
// 形式で決まる OZ_RADIX・OZ_DMAX）．範囲がデバイスの構成で決まる OZ_NSMALL の上界だけはデバイスを開いた後に見るが，
// そのときはデバイスを閉じてから 2 で終わる（main）．
//
// 名前の一覧（ozaki1/ のヘッダとカーネルで共通）
//   p        スライス数（位置引数．OZ_FP64 が選び直しうる）    L      打ち切り対角（0..p-1．既定 p-1）
//   nslice   スライスの本数 L+1．打ち切り対角 L のもとで使う対は i+j <= L なので要るスライスは 0..L の L+1 本だけで，
//            対角の本数 ndiag もこれに等しい（RunCfg::nslice()．スライス段の本数・GEMM の連結の段数・Pbig の面の数）
//   npairs   スライス対の数 (L+1)(L+2)/2
//   Mt, Nt   出力のタイル数（M/32・N/32）                   Kto    K のタイル数 K/32（スライス 1 本ぶん）
//   Kc, nkb  無誤差な K ブロック長［要素］とその本数 K/Kc   Kbt    K ブロック 1 本ぶんの K タイル数 Kc/32
//   Kmax     スライス連結後の K (L+1)*K［要素］             Ktm    そのタイル数 Kmax/32
//   bw       GEMM の 1 回の読み出しの K 幅［タイル］（Kbt を割り切る）
//   band     出力の行バンド．Pbig はバンド 1 本ぶんの面を nslice 枚持ち，面の間隔は band_tiles［タイル］
//   radix    基数（桁の重み）．dmax は桁の上限（既定 ceil(radix/2)）

#include <limits>
#include <optional>
#include "ozaki1/ozaki1_common.hpp"
#include "ozaki1/ozaki1_plan.hpp"     // RunCfg・Oz1Knobs・plan_issue・plan_panels・recon_scale
#include "ozaki1/ozaki1_autop.hpp"    // OZ_FP64 の p の選択（oz_fp64_p）
#include "ozaki1/ozaki1_stages.hpp"   // 段の資源と投入の手順
#include "ozaki1/ozaki1_whole.hpp"    // OZ_TIME_WHOLE の流し直しと検査
#include "common/host_eval.hpp"
#include "common/oz_driver.hpp"    // 入口の検査の並び・CONFIG 行の共通の欄・最後の判定
#include "common/oz_whole.hpp"     // OZ_TIME_WHOLE の検査（WHOLE-CHECK）

namespace dist = tt::tt_metal::distributed;

// ===== 対角ごとの K 連結という要点 =====
//
// 流れは FP64 入力 → 行ごと（A）／列ごと（B）のスケール → 基数 radix の整数スライス．
// スライス対 (i,j) は対角 d = i+j でまとめられる（重み radix^-(d+2) が共通なので）．
// 同じ対角の対を K 方向に積み重ねて 1 本の論理的な GEMM にする：
//     A~_d = [A_{i0}|A_{i1}|...],   B~_d = [B_{j0};B_{j1};...]
// とすれば  A~_d ・ B~_d = Σ_{i+j=d} A_i・B_j = P_d  が一度に得られる（整数なので無誤差）．
//
// こうすると，対角ごとに 1 回のデバイス実行で P_d が出るので，中間結果の
// 読み出しが対ごとの O(p^2) 回から対角ごとの (L+1) 回に減る．
//
// 同一対角の積を 1 つの整数累算器でまとめる着想は Uchino らによる
// （Performance enhancement of the Ozaki Scheme on integer matrix multiplication
// unit, IJHPCA 39(2), 2025．同論文は群 G_g = {(i,j) | i+j=g} と呼ぶ）．
// 違いは実現の仕方で，同論文は同じ出力へ beta=1 の GEMM を反復し，まとめる本数を
// 閾値 r = max(1, 2^(31-2b-ceil(log2 n))) で打ち切る．本実装の既定FP経路はPAIRLOOPで対を順に処理し，
// 対の境界でFP32累積器をL1のINT32へ加算して初期化する．FP32の無誤差条件は1対分のKに課し，
// INT32には対角全体の上界を課す．発行回数はパネル当たりL+1になる．INT8は対角全体をINT32で
// 累積するのでK_eff=(d+1)*Kを用いる．以下のK分割・対ごとの発行は代替構成の説明である．
//
// ただし連結が常に得とは限らない．K が大きいと K ブロック化のほうが本数を押し上げる
// ので，そのときは連結をやめて対ごとに 1 本ずつ発行する（K_eff = K_c になり
// K ブロック化が消える）．本数（＝発行回数＝累積パスの面更新回数）が少ないほうを
// 自動で選ぶ：n_pw < n_cat を参照（ozaki1_plan.hpp の plan_issue）．対ごとの発行は対角あたり (d+1) 本を
// 足し込むので，Int32 累積（遅延再構成）が前提になる．

// ===== 実行の段 =====
//
// main は次の順に段を呼ぶだけである．デバイスの操作（バッファの確保・プログラムの
// 構築・転送・発行・Finish・計時）の順序は性能を左右する（DRAM の割り付け順で約 5%）ので，
// 段の順序も，段の中の操作と実行中にバッファやコンテキストを手放す点（reset の位置）も
// むやみに動かさないこと．md->close() のあとにホスト側のコンテキストが破棄される順序は，
// 段の戻り値（FillStage・InputBufs など）の宣言順に従う．
//   parse_run_config  位置引数・基数・正規化の設定（デバイスに触らない）
//   prepare_operands  入力生成・参照解・OZ_FP64 による p の選び直し（ホストのみ）
//   plan_issue        発行と累積のしかた，K ブロック長（デバイスに触らない．ozaki1_plan.hpp）
//   plan_panels       出力パネルと行バンドの分割（同）
//   alloc_pbig        対角の部分和 Pbig を確保する
//   setup_inputs      入力側のバッファを確保し，FP64 のオペランドを送る
//   run_scale_slice   スケール算出とスライス分解を構築し，計時する
//   run_single_kblock / run_sweep  GEMM と再構成を構築し，計時する
//   whole_*           OZ_TIME_WHOLE のときだけ，全工程を続けて流して計時する（ozaki1_whole.hpp）
//   report            精度と時間を表示する

// 位置引数を読んで検査する（RunCfg の M・N・K・p・L・warmup・iters・auto_p．つまみの値の検査もここ）．
// 引数・つまみが不正なら理由を表示して終了コード 2 で止める（入口の検査の並びは oz_driver.hpp．削除したつまみと
// scheme II だけのつまみは main の oz_drv::reject_env が先に止める）．
static RunCfg read_run_args(int argc, char** argv) {
    RunCfg c;
    // 余分な位置引数も止める（引数の追加・位置ずれを黙って読み捨てない．scheme II の入口と同じ）
    if (!oz_env::check_argc(argc, 7, "M N K p L warmup iters")) std::exit(2);
    // 位置引数はどれも oz_env::arg_u32 で範囲ごと読む（負値・末尾の余計な文字・範囲外はそこで止まる）
    const oz_drv::Dims d = oz_drv::read_dims(argc, argv, {256, 256, 512}, "ozaki1");
    c.M = d.M; c.N = d.N; c.K = d.K;
    // p の上限は int への変換と対の数 (L+1)(L+2)/2（count_pairs の閉じた式．int に収める）を溢れさせないための安全弁で，
    // 実際に使う p（形式ごとの値の正本は scripts/routes.tsv．OZ_FP64 が φ の広い入力で選ぶ値でも 100 前後）より十分大きい．
    const uint32_t pu = oz_env::arg_u32(argc, argv, 4, 4, 1, 4096, "p");
    // 打ち切り対角 L は 0..p-1（定義上 L < p）．スライスは L+1 本作り，連結の段数も L+1 なので
    // （RunCfg::nslice），L >= p を通すと p を L+1 にしたのと同じ計算になり，
    // 見出しや RESULT 行の p と食い違う．したがって受け付けない（count_pairs も L < p を前提にする）．
    c.p = (int)pu;
    c.L = (int)oz_env::arg_u32(argc, argv, 5, pu - 1, 0, pu - 1, "L");
    oz_drv::read_counts(argc, argv, 6, 1, 3, c.warmup, c.iters);
    // OZ_FP64 は p を入力から選び直し，L は選んだ p から p-1 に決める（prepare_operands）．L を位置引数で明示して
    // それが p-1 でなければ，指定を黙って捨てることになるので止める（測定スクリプトは p と L=p-1 を渡す）
    c.auto_p = oz_knobs::set(OZK("OZ_FP64"));
    if (c.auto_p && argc > 5 && c.L != c.p - 1) {
        fmt::print(stderr, "ERROR: OZ_FP64 では L は選び直した p から p-1 に決まる．位置引数の L={} は p-1={} でないので"
                           "受け付けない（L を省くか p-1 を渡す）\n", c.L, c.p - 1);
        std::exit(2);
    }
    // このスキームのつまみの値・入力の生成と判定と書き出し・GEMM 段・計時（oz_driver.hpp の並び）
    oz_drv::check_env(oz_knobs::W_I, c.K);
    c.Mt = c.M / 32; c.Nt = c.N / 32;
    return c;
}

// 基数・桁の上限・MathFidelity と，正規化の目標・BF16 化の丸め方向を決める（デバイスに触らない）．
//
// 尾崎スキームの基数 β．形式ごとの既定値 OZ1_PARAM.radix（桁の上限 d から β = 2d）と，その決め方は
// ozaki1_format.hpp の形式の表と ozaki_digit_max にある．
// OZ_RADIX：ビット／スライスの余裕を調べる実験用の上書き（診断用．通常の実行では使わない）．渡すのは桁の重み（基数）で，
// 桁の上限は ozaki_digit_max() が決める．上限は 2*OZ_FMT.exact_int_max とする．累算側の予算超過は ozaki_block_k が別途
// 0 を返して弾き，乗算器の限界（FP 形式の d<=63）は ozaki_check_exact が警告するので，ここでは表現限界だけを見る．
static void choose_radix_norm(RunCfg& c) {
    const int radix = (int)oz_env::u32("OZ_RADIX", OZ1_PARAM.radix, 2u, 2u * OZ_FMT.exact_int_max);
    c.radix = radix;
    // 基数 beta・桁の上限 dmax の平衡表現が表せるのは |x/tau| <= dmax/(beta-1) である．
    // beta = 2*dmax なら約 1/2 なので正規化を 1/2 に（expdelta_build へ shift=-1），
    // beta = 2 の BFP2_b だけは上限が 1 なので正規化も 1 に取る（shift=0）．
    // スケール tau はデバイスが max|x| の BF16 表現から求める（scale_f64_reader.cpp）．
    // 切り捨てで作ると mx は真の最大値より相対 2^-7 まで小さくなりうる（1 直上の BF16 の
    // 刻み幅）．そのとき指数が 1 つ小さく出て，正規化後の値が目標 2^nshift を最大で
    // 相対 2^-7 だけ超える．展開の上限 span = d/(β-1) が目標を 1/128 以上上回る形式
    // （β=64 で 1.6%，β=32 で 3.2%，β=14 で 7.7%）はその超過を表せるので切り捨てで
    // よい．INT8（127/253，目標 1/2 を 0.4% 上回るだけ）と BFP2（1 で目標 1 に等しい）は
    // 余裕が 1/128 を切るので，bf_ceil を立てて mx を切り上げ，|x/tau| <= 2^nshift を
    // 保証する（scheme II は常に切り上げる）．
    c.dmax = ozaki_digit_max((uint32_t)radix);
    const double span = (double)c.dmax / (radix - 1);           // 表せる上限
    c.nshift = (span >= 1.0) ? 0 : -1;                          // 正規化の目標 2^nshift
    const double target = std::ldexp(1.0, c.nshift);
    c.bf_ceil = span < target * (1.0 + 1.0 / 128.0);            // 切り上げが要るか
    // 目標の値そのものを桁 1 個で表せない（β·2^nshift > 桁の上限．既定では BFP2_b の 2·1 > 1 だけ）と，正規化後に
    // 目標の近くに来る要素の展開が長くなる：u = β·t が 2 に近いと桁の上限 1 へ丸められ（round(u) = 2 を 1 に切る），
    // 残差 u - 1 は 1 からの距離が倍になるだけなので，正規化後の値が 1-δ の要素は先頭に約 log2(1/δ) 個の桁 1 が並ぶ
    // （ちょうど目標の要素（δ = 0．行・列の最大値がちょうど 2 のべきで，切り上げが働かない）では展開が終わらず，全桁が 1）．
    // 桁 1 の並びどうしの積では，打ち切った対角の尾 Σ_{d>L}(対の数)·2^-(d+2) が相殺せずに足される．2 つに分けて扱う．
    //   ちょうど 2 のべき：tau_above を立て，スケールを最大値より真に大きい 2 のべき 2^(floor(log2 max)+1) に取る
    //     （scale_f64_reader.cpp の CEIL=2）．最大値がちょうど 2 のべきの行・列だけが 1 段上がって，その要素は 1/2 になり
    //     桁 1 個で厳密に表せる（その行・列は 1 bit 浅く切ることになる）．それ以外の行・列のスケールは bf_ceil だけの
    //     ときと同じなので，最大値がちょうど 2 のべきにならない一様乱数の入力では結果は変わらない．上げないと全桁 1 の
    //     打ち切りの誤差は相対 (p+2)·2^-(p+1) で，既定の p でも余裕は 1 bit ほどしかない（scripts/tests/check_adversarial.sh の入力）．
    //   2 のべきのすぐ下（δ > 0）：桁 1 の並びは高々 53 個（FP64 の仮数の長さ）で終わるので，スケールでは扱わず p で扱う．
    //     BFP2_b の既定の p は，構造のある入力の組（並びが最も長い 1-2^-53 を含む）でも 53 bit 以上に届くように
    //     選んである（scripts/diagnose/select_slice_count.sh の規則 (ii)．値の正本は scripts/routes.tsv）．
    // 目標が桁 1 個で表せる形式（BF16 系の 64·1/2 = 32 など）では，ちょうど目標の要素は先頭の桁 1 個で尽き，目標の
    // 近くの要素も桁の上限で切られない（先頭の u = β·x <= 32.25 などで丸めた桁は上限以下，以降は |t| <= 1/2）ので要らない．
    c.tau_above = (double)radix * target > (double)c.dmax;
    if (span < target)
        fmt::print(stderr, "WARNING: 基数 {} / 桁の上限 {} ではスライス展開の上限が "
                           "{:.5f} と目標 {:.5f} を下回る．正規化後の入力を表せない\n",
                   radix, c.dmax, span, target);
    // fidelity は桁の上限から決まる（OZ_FIDELITY の値は read_run_args の check_env が検査した）．以降の段はこの値を使う
    c.fid = oz1_fidelity(c.dmax);
}

static RunCfg parse_run_config(int argc, char** argv) {
    RunCfg c = read_run_args(argc, argv);
    choose_radix_norm(c);
    return c;
}

// CONFIG 行の knobs 欄の文字列（計画のつまみの構造体 Oz1Knobs・oz1_knobs_from_env とは別物）．アブレーションのつまみ
// （docs/env_vars.md の B）のうち，既定から外れているものを '+' 区切りで残す．値を持つものは 名前:値 と書く（欄の区切りの
// '=' と紛れないように）．どれも数学的には同じ積を計算し，精度の基準（到達精度 53 bit）も保つが，ビット単位で同じ結果とは
// 限らない（再構成の入力の読み方や回し方を変えるつまみは最下位のビットを変えうる．どのつまみがそうかは docs/env_vars.md の B）．
// 論文の表は既定の構成で測った値でなければならないので，検査側（scripts/lib/oz_logs.py）が既定でないログを弾けるようにする．
// OZ_DBG（進捗の表示）は計時区間の中で stderr へ書き，A のスライスの後に Finish を足すので
// 時間を変える．答えは変えないがここに残す．
// どのつまみを数えるかと並び・値の書き方は，つまみの表（common/oz_knobs.hpp の R_KNOBS）が決める．
static std::string oz1_knobs_record() { return oz_knobs::record(oz_knobs::W_I, oz_knobs::R_KNOBS); }

// デバイスの格子と理論ピーク
struct DevInfo { tt::tt_metal::CoreCoord gd; int mhz = 0; double peak_chip = 0.0; };
static DevInfo device_info(std::shared_ptr<dist::MeshDevice>& md, tt::tt_metal::MathFidelity fid) {
    DevInfo dv;
    dv.gd = md->compute_with_storage_grid_size();
    dv.mhz = md->get_clock_rate_mhz();
    const double clk = (double)dv.mhz * 1e6;
    // 理論ピーク［演算数/サイクル/コア］（積和の 2 倍は算入済み）は fidelity で決まる：
    // LoFi 1 パスが 4096 で，パス数に反比例する．既定は桁の上限から決まり（bf16/fp8 は
    // LoFi=4096，int8 は HiFi4=1024），OZ_FIDELITY を立てればピークも一緒に動く．
    // 式は基準実装 3 本と共有する（common/oz_fidelity.hpp の oz_peak_tops）．
    dv.peak_chip = oz_peak_tops(fid, clk, (uint32_t)(dv.gd.x * dv.gd.y));  // format TOP/s
    return dv;
}

static void print_header(const RunCfg& c, const DevInfo& dv, const PairCount& pc) {
    fmt::print("====== {}-Ozaki GEMM (on-device exact, per-diagonal K-stack) ======\n",
               OZ_FMT.name);
    fmt::print("Device : grid={}x{} ({} cores) {} MHz  {} {} peak={:.1f} TOP/s\n",
               dv.gd.x, dv.gd.y, dv.gd.x * dv.gd.y, dv.mhz, OZ_FMT.name,
               oz_fidelity_name(c.fid), dv.peak_chip);
    // 「Problem: M=… N=… K=…  slices p=…」の書き出しは scripts/rules/probe_checks.py の problem（上限の探針の照合．DRAM 不足で
    // 止まったログも読む）が空白 2 つまで固定した正規表現で読む（同じ値は RESULT の M,N,K と CONFIG の p= にもあるが，
    // 確保の前に止まったログにはこの行しか無い）．表示を整えるときは照合の正規表現と一緒に直す
    fmt::print("Problem: M={} N={} K={}  slices p={} radix={} (~{}b) L={}  "
               "{} pairs -> {} diagonal GEMMs (readbacks {} not {})\n",
               c.M, c.N, c.K, c.p, c.radix, (int)(std::log2((double)c.radix) * c.p),
               c.L, pc.npairs, pc.ndiag, pc.ndiag, pc.npairs);
}

// ホスト側のオペランドと参照解
struct Operands {
    std::vector<double> A, Bm;
    std::vector<double> Chi, Clo;      // 参照解（非評価和 Chi + Clo）
    bool verify = false;
};

// 入力を生成し，検証するなら参照解を計算する．OZ_FP64 が立っていれば入力から
// p を選び直し，c.p・c.L を書き換える．ホストだけで完結し，デバイスには触らない．
static Operands prepare_operands(RunCfg& c) {
    Operands op;
    oz_eval::gen_operands(op.A, op.Bm, c.M, c.K, c.N);
    op.verify = oz_eval::verify_enabled();
    if (op.verify) oz_eval::reference_gemm(op.A, op.Bm, c.M, c.K, c.N, op.Chi, op.Clo);
    // スライス数 p を入力から選び直す経路（論文の既定の測定では使わない）．
    if (c.auto_p) {
        // 正規化はデバイスと同じ目標（nshift）と BF16 化の向き（scale_build へ渡す CEIL と同じ値）で見積もる
        c.p = oz_fp64_p(op.A, op.Bm, c.M, c.N, c.K, c.radix, c.dmax, c.nshift, c.ceil_mode());
        c.L = c.p - 1;
    }
    return op;
}

// 対角ごとの部分和 Pbig（バンド 1 本ぶんの面を pbig_planes 枚）を確保する．
// DRAM の割り付け順を保つため，入力側のバッファより先に確保する．
static std::shared_ptr<dist::MeshBuffer> alloc_pbig(std::shared_ptr<dist::MeshDevice>& md, const PanelPlan& pp) {
    dist::ReplicatedBufferConfig bP{
        .size = (size_t)pp.pdiag_tile * (size_t)pp.pbig_planes * pp.band_tiles};
    dist::DeviceLocalBufferConfig dP{
        .page_size = pp.pdiag_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    return dist::MeshBuffer::create(bP, dP, md.get());
}

static void print_cfg(const PanelPlan& pp, uint32_t nkb) {
    const OptConfig& gU = pp.gU;
    const OptConfig& gR = pp.gR;
    if (pp.rem_rows)
        fmt::print("Cfg     : grid={}x{} per_core={}x{} panels={}x{} "
                   "(端数 1 行は grid={}x{} per_core={}x{} bw={}) (Kmax={}, bw={}, K blocks={})\n",
                   gU.cr, gU.cc, gU.pm, gU.pn, pp.nbr, pp.nbc, gR.cr, gR.cc, gR.pm, gR.pn, gR.bw,
                   pp.Kmax, gU.bw, nkb);
    else
        fmt::print("Cfg     : grid={}x{} per_core={}x{} panels={}x{} "
                   "(Kmax={}, bw={}, K blocks={})\n",
                   gU.cr, gU.cc, gU.pm, gU.pn, pp.nbr, pp.nbc, pp.Kmax, gU.bw, nkb);
}

// 入力側のバッファを dltAb, dltBb, rAf64, rBf64, maxAb, maxBb, flgA, flgB の順に確保し，
// 旗をゼロ化して FP64 のオペランドを送ってから，スライス連結の shA, shB を確保する．
// 確保の順序は DRAM の割り付けを左右するので変えないこと．
static InputBufs setup_inputs(std::shared_ptr<dist::MeshDevice>& md, const RunCfg& c,
                              const PanelPlan& pp, const tt::tt_metal::CoreCoord& gd, const Operands& op) {
    using tt::tt_metal::BufferType;
    InputBufs in;
    auto& cq = md->mesh_command_queue();
    const uint32_t Mt = c.Mt, Nt = c.Nt, Kto = c.K / 32;
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32);
    {
        dist::DeviceLocalBufferConfig dF{
            .page_size = f32t, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bIa{.size = (size_t)f32t * Mt};
        dist::ReplicatedBufferConfig bIb{.size = (size_t)f32t * Nt};
        in.dltAb = dist::MeshBuffer::create(bIa, dF, md.get());
        in.dltBb = dist::MeshBuffer::create(bIb, dF, md.get());
        // FP64 のオペランドは 1 要素 8 バイト＝1 ページ OZ_F64_TILE_BYTES（8192）バイト
        dist::DeviceLocalBufferConfig dD{
            .page_size = OZ_F64_TILE_BYTES, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bAd{.size = (size_t)OZ_F64_TILE_BYTES * Mt * Kto};
        dist::ReplicatedBufferConfig bBd{.size = (size_t)OZ_F64_TILE_BYTES * Kto * Nt};
        in.rAf64 = dist::MeshBuffer::create(bAd, dD, md.get());
        in.rBf64 = dist::MeshBuffer::create(bBd, dD, md.get());
        uint32_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
        dist::DeviceLocalBufferConfig dB{
            .page_size = bf16t, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bMa{.size = (size_t)bf16t * Mt};
        dist::ReplicatedBufferConfig bMb{.size = (size_t)bf16t * Nt};
        in.maxAb = dist::MeshBuffer::create(bMa, dB, md.get());
        in.maxBb = dist::MeshBuffer::create(bMb, dB, md.get());
    }
    // オペランドを FP64 のまま送る（ホストがするのはタイル順への並べ替えだけで，
    // 演算は一切しない）．論文の測定はデバイス DRAM 常駐からなので転送は計時の外
    // だが，FP64 -> triple-float の変換は計時の内側でデバイスが行う．
    // 範囲外検出の旗はオペランドを送る前にゼロ化しておく：A・B を送ってから
    // C が戻るまでホストと通信しないという仕様のため．
    // 旗はコアごとに 1 タイル（重複 NoC 書き込みの振る舞いに依存しないため）
    const uint32_t nslot = gd.x * gd.y;
    in.flgA = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
    in.flgB = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
    { std::vector<uint32_t> z((size_t)nslot * tt::tile_size(tt::DataFormat::Int32) / 4, 0);
      dist::EnqueueWriteMeshBuffer(cq, in.flgA, z, false);
      dist::EnqueueWriteMeshBuffer(cq, in.flgB, z, false);
      dist::Finish(cq); }
    upload_operand_f64(md, op.A, c.M, c.K, in.rAf64);
    upload_operand_f64(md, op.Bm, c.K, c.N, in.rBf64);

    // shA/shB にはスライス済みオペランドの全体が入り，各パネルはリーダの
    // dA/dB オフセットで部分範囲を選ぶ．したがって 1 回作って共有すればよい．
    // パネルごとに作り直すとスライス段のコストが nbr*nbc に比例して増え，
    // 1 パネルに収まらない形状すべてで実行時間を支配してしまう．
    {
        uint32_t in_tile = tt::tile_size(OZ_FMT.in);
        dist::DeviceLocalBufferConfig dIn{
            .page_size = in_tile, .buffer_type = BufferType::DRAM};
        dist::ReplicatedBufferConfig bAc{.size = (size_t)in_tile * Mt * pp.Ktm};
        dist::ReplicatedBufferConfig bBc{.size = (size_t)in_tile * Nt * pp.Ktm};
        in.shA = dist::MeshBuffer::create(bAc, dIn, md.get());
        in.shB = dist::MeshBuffer::create(bBc, dIn, md.get());
    }
    return in;
}

// スケール算出（scale）・指数差分（expdelta）・スライス分解（slice）のワークロードを
// expdelta A,B → scale A,B → slice の順に構築し，JIT prime・ウォームアップ・計時を行う
// （出力パネル・K ブロック・対角のすべてで共有）．バッファは新たに確保しない．
static FillStage run_scale_slice(std::shared_ptr<dist::MeshDevice>& md, const RunCfg& c, const Oz1Diag& dg,
                                 const PanelPlan& pp, InputBufs& in, uint32_t gw, uint32_t gi, Times& t) {
    FillStage fs;
    auto& cq = md->mesh_command_queue();
    const uint32_t Mt = c.Mt, Nt = c.Nt, Kto = c.K / 32;
    // スケール算出は最後までデバイス上で閉じている：行ごと／列ごとの
    // 最大値を求め（scale_run），そこから指数へ足す差分タイルを作る
    // （expdelta_run）．正規化はスライサ内で融合される．
    // A・B を転送してから C が戻るまでホストと通信しない仕様のため，
    // 仕上げもホストに残さない．
    fs.edA = expdelta_build(md, in.maxAb, in.dltAb, in.flgA, Mt, true,  c.nshift);
    fs.edB = expdelta_build(md, in.maxBb, in.dltBb, in.flgB, Nt, false, c.nshift);
    // 構築の順序はスケール -> スライサ -> JIT prime（t_fill_build に数える）．
    const auto tsb = now();
    fs.scA = scale_build(md, 0u, Kto, in.rAf64, in.maxAb, Mt, 0u, c.bf_ceil, c.tau_above);
    fs.scB = scale_build(md, 1u, Kto, in.rBf64, in.maxBb, Nt, Nt, c.bf_ceil, c.tau_above);
    // 桁の上限は計画が無誤差条件に使った c.dmax をそのまま渡す（slice_build の注記）
    fs.slc = slice_build(md, pp.nslice, (uint32_t)c.radix, c.dmax, dg.softfloat, in.rAf64, in.dltAb, in.shA);
    // スライサは FP64 のオペランドを直接読み，0 回目の反復で triple-float への変換と正規化（指数差分）を融合する．
    // A は Mt*Kto 枚（指数差分は行ごと），B は Kto*Nt 枚（列ごと）．どちらもブロック優先レイアウトで書く（K ブロック Kbt）
    fs.sliceA = SliceRun{in.rAf64, in.dltAb, in.shA, Mt * Kto, pp.nslice, Kto, 0u, 0u, pp.Kbt};
    fs.sliceB = SliceRun{in.rBf64, in.dltBb, in.shB, Kto * Nt, pp.nslice, Kto, Nt, 1u, pp.Kbt};
    oz_dbg("slice_build done");
    auto dscale = [&]() {
        auto td0 = now();
        fill_issue_scale(md, fs, oz1_no_note);
        dist::Finish(cq);
        auto td1 = now();
        t.dsc_scale += std::chrono::duration<double>(td1 - td0).count();
        // 最大値を 2 のべきに切り上げ，指数へ足す差分 shift-e をタイルへ展開する
        // （BF16 化の切り上げの要否は上の bf_ceil，2 のべきの最大値を 1 段上げるかは tau_above）．
        fill_issue_expdelta(md, fs, oz1_no_note);
        dist::Finish(cq);
        t.dsc_glue += std::chrono::duration<double>(now() - td1).count();
        // 正規化のための独立したパスは持たない．スライスのリーダ
        // （slice_f64_reader.cpp）が FP64 の指数へ差分を足して融合するからである．
        // スケール 2^-e 自体は指数差分タイルとしてデバイスに残り，
        // 最後の再構成のライタが FP64 を組み立てるときに掛かる
        // （recon_f64_writer.cpp．ホストは掛け算をしない）．
        t.dsc_calls++;
    };
    auto slice = [&]() {
        fill_issue_slice(md, fs, oz1_no_note,
                         [&] { if (oz_dbg_on()) { dist::Finish(cq); oz_dbg("slice A dispatched"); } });
        dist::Finish(cq);
    };
    dscale();
    oz_dbg("jit scale done");
    slice();   // JIT prime
    oz_dbg("jit slice done");
    t.fill_build += std::chrono::duration<double>(now() - tsb).count();

    // 温め（段「fill」．cold は gw 回，steady は定常状態まで．oz_timing.hpp の warm_up）．1 回ぶんの中で同期する
    // （dscale・slice がそれぞれ Finish する）ので，同期の関数は空にし，投入の単位も 1 回にする
    oz_eval::warm_up("fill", nullptr, gw, 1, [&] { dscale(); slice(); }, [] {});
    // 内訳タイマをここでリセットし，計時対象の反復だけを数える
    // （上のウォームアップが初回の JIT を吸収している）
    t.dsc_scale = t.dsc_glue = 0.0; t.dsc_calls = 0;
    // スライス処理は呼び出しごとに 1 回だけ行う．ブロック優先レイアウトの
    // おかげで，1 回のスライスで全 K ブロックぶんの連結バッファが揃う．
    // オペランドは出力パネル間でも共有されるので，これが本当のコストである．
    const oz_eval::TimedMark tm = oz_eval::begin_timed("fill");
    auto tf = now();
    for (uint32_t it = 0; it < gi; ++it) {
        auto tu = now();
        dscale();
        oz_dbg("scale done");
        t.fill_scale += std::chrono::duration<double>(now() - tu).count() / gi;
        slice();
        oz_dbg("slice+fill iter done");
    }
    t.fill += std::chrono::duration<double>(now() - tf).count() / gi;
    oz_eval::end_timed(tm, gi);
    return fs;
}

// K ブロックが 1 つで，対ごとにも PAIRLOOP でも回さない構成（IssuePlan::single_kblock．INT8 と一部のアブレーション）の
// GEMM と再構成．バンドごとに「パネルの GEMM プログラムを組んで計時し，使い切って捨てる」を
// 繰り返し，そのバンドの再構成を組んで（初回だけ再構成のバッファを確保する）計時する．
// GEMM のプログラムはパネルの終わりで解放される．
static void run_single_kblock(const StageCtx& s, BandRecon& br, Times& t) {
    auto& md = s.md;
    auto& cq = md->mesh_command_queue();
    const int L = s.c.L;
    const uint32_t gw = s.gw, gi = s.gi;
    // 対角ごとの発行は 1 パスにつき Finish 1 回だけでパイプライン化できる：
    // tt-metal は enqueue の時点で実行時引数を確定させるので，対角ごとに引数を
    // 付け替えても対角間で Finish する必要がない（結果がビット単位で一致することを確認済み）．
    // K ブロックが 1 つなら再構成は最後に 1 回で済むので，パネルのコンテキストを
    // 持ち回る必要がない．パネルごとに構築して使い切り，すぐ捨てる．同じ
    // プログラムを連続して発行できる形にしておくのが要点で，多数のプログラムを
    // 代わる代わる発行するとカーネルの再転送が挟まって発行が遅くなる（パネルが 110 枚になる 14080^3 の INT8 で
    // GEMM 段が 5.3% 遅い．docs/kblock_plan.md）．
    // バンドごとに「そのバンドの全パネルの GEMM → そのバンドの再構成」を回す．
    // バンドが 1 本なら，全パネルの GEMM の後に再構成を 1 回回すだけになる．
    for (const auto& bd : s.pp.bands) {
        for (uint32_t pi = bd.p0; pi < bd.p1; pi++) {
            auto tb0 = now();
            GemmCtx ctx = build_panel(s, pi, bd);
            t.build += std::chrono::duration<double>(now() - tb0).count();
            // 反復ごとに Finish する形なので，同期を dispatch 側へ入れる
            t.gemm += oz_eval::time_dispatch("gemm", gw, gi, [&]{
                issue_panel_diagonals(md, ctx, L, oz1_no_note);
                dist::Finish(cq); });
        }
        ReconCtx& rr = band_recon_at(s, br, t, bd, s.Pbig);
        t.recon_dev += oz_eval::time_dispatch("recon", gw, gi,
            [&]{ recon_exec(md, rr, 0u); },
            [&]{ dist::Finish(cq); });
    }
}

// 掃き出し（run_sweep）の準備：全パネルの GEMM プログラムを ctxs に組み（main の終わりまで持つ），そのあとで
// 遅延再構成の累積面 acc（別プログラムで累積する構成のみ）と再構成を確保・構築する．再構成側のバッファは GEMM の
// コンテキストを組んだあとに確保する（先に確保すると GEMM 段の DRAM の割り付けが変わって遅くなる）．
// 続行できないとき（累積面が確保できない・組めない）は理由を表示し，md->close() まで済ませて空を返す．
static std::optional<Oz1Sweep> sweep_setup(const StageCtx& s, BandRecon& br, AccumCtx& acc,
                                           std::vector<GemmCtx>& ctxs, Times& t) {
    auto& md = s.md;
    const IssuePlan& ip = s.ip;
    const PanelPlan& pp = s.pp;
    const uint32_t ndiag = s.c.nslice();
    // defer が立つのは 4 つの累積のどれか（PAIRLOOP・fuse・対ごと・スクラッチ）のときだけである
    // （plan_issue の ip.defer）．下の分岐はこれを前提にしている．
    TT_FATAL(!ip.defer || ip.pairloop || ip.fuse || ip.pairwise || ip.scratch,
             "sweep_setup: 遅延再構成なのに累積のしかたが決まっていない");
    // K ブロックが複数あるとき，または対ごとに回すときは，発行のループが
    // 出力パネルのループの外側に来る．再構成は出力全体を一度に処理するので，あるブロックの再構成へ
    // 進む前に，そのブロックについて全パネルの GEMM が終わっている必要がある
    // からである．したがって全パネルのコンテキストを保持する．
    ctxs.reserve((size_t)pp.nbr * pp.nbc);
    auto tb0 = now();
    for (const auto& bd : pp.bands)
        for (uint32_t pi = bd.p0; pi < bd.p1; pi++)
            ctxs.push_back(build_panel(s, pi, bd));
    t.build += std::chrono::duration<double>(now() - tb0).count();
    // この構成は nkb > 1 || per_pair なので，再構成は累積つきで組む（組み方は計画から．ozaki1_stages.hpp の band_recon）．
    // 入力は既定で Pbig（PAIRLOOP と fuse は累積が GEMM の出力経路に入っていて，Pbig が Int32 の累積面そのもの．そこから
    // 1 回 Horner を回す）．
    Oz1Sweep sw{s, ctxs, acc, s.Pbig};
    if (!(ip.fuse || ip.pairloop)) {
        if (ip.defer) {
            // ここへ来るのは pairwise か scratch のときだけである（plan_issue の ip.defer）．
            // 累積バッファは Pbig と同じ ndiag 面ぶん DRAM を余分に要る（14080^3 なら 1 面 0.79 GB で，
            // FP8 は p 面．8197a0b の routes.tsv の p=12 なら 9.5 GB．既定の FP8 は PAIRLOOP なので，この面を確保しない）．入るかどうかは
            // plan_issue が空き容量から見積もってあり，入らない構成は最初から defer を立てない
            // （K ブロックが複数なら K ブロックごとに Horner を回す経路になる）．
            // 見積もりに反して確保できなかったら止める．対ごとの発行も，スクラッチ 1 面の
            // 構成も，Int32 累積が前提なので後から降りられない：スクラッチでは
            // Pbig が 1 面しか無く，K ブロックごとに Horner を回す再構成（ndiag 面を読む）が成り立たない．
            const char* what = ip.pairwise ? "対ごとの発行" : "スクラッチ 1 面の構成";
            std::string why;
            // 累積面 S は Pbig と同じ形（バンド 1 本ぶんの面を ndiag 枚）
            const auto S = accum_alloc(md, pp.band_rows, s.c.Nt, ndiag, why);
            if (!S) {
                fmt::print("Recon   : 遅延再構成の累積バッファ（{} 面，{:.1f} GB）を"
                           "確保できない\n", ndiag,
                           (double)ndiag * pp.band_tiles * tt::tile_size(tt::DataFormat::Int32) / 1e9);
                fmt::print(stderr, "ERROR: {}には Int32 累積が要ります"
                                   "（累積バッファを確保できませんでした：{}）\n", what, why);
                md->close(); return std::nullopt;
            }
            try {
                acc = accum_build(md, s.Pbig, S, pp.band_rows, s.c.Nt, ndiag, pp.pbig_planes, &t.recon_build);
            } catch (const std::exception& e) {
                fmt::print(stderr, "ERROR: {}の Int32 累積のプログラムを組めない（{}）\n", what, e.what());
                md->close(); return std::nullopt;
            }
            sw.rsrc = acc.S;
        }
        // 遅延再構成を使わない経路（!ip.defer）は K ブロックごとに Pbig から Horner を回す（再構成は ip.defer で組み分ける）
    }
    // バンドの高さは 2 通りまで．最初の呼び出しが端数バンドの分（rcnT）もまとめて組む．
    band_recon(s, br, t, pp.bands[0].rows, sw.rsrc);
    fmt::print("Recon   : {}\n", ip.pairloop
        ? "対角ごとに 1 本で (d+1) 対を回し，L1 で Int32 累積して最後に 1 回 Horner"
        : ip.fuse
        ? (ip.pairwise
           ? "対ごとに発行し，GEMM の出力経路で Int32 累積して最後に 1 回 Horner"
           : "GEMM の出力経路で Int32 累積し，最後に 1 回 Horner")
        : !ip.defer
        ? "K ブロックごとに triple-float Horner"
        : (ip.pairwise
           ? "対ごとに発行し，対角ごとに Int32 で累積して最後に 1 回 Horner"
           : "遅延再構成（対角ごとに Int32 で K 累積し，最後に 1 回 Horner）"));
    return sw;
}

// K ブロックが複数ある構成，または対ごと・PAIRLOOP で回す構成の GEMM と再構成．準備（sweep_setup）の後，
// バンドごとの掃き出しをウォームアップ・計時する．投入の手順は Oz1Sweep（ozaki1_stages.hpp）で，ここは同期と計時を挟む．
// 続行できないときは md->close() まで済ませて空を返す．返すのは投入の手順（OZ_TIME_WHOLE の流し直しが使う）．
static std::optional<Oz1Sweep> run_sweep(const StageCtx& s, BandRecon& br, AccumCtx& acc,
                                         std::vector<GemmCtx>& ctxs, Times& t) {
    const std::optional<Oz1Sweep> swo = sweep_setup(s, br, acc, ctxs, t);
    if (!swo) return std::nullopt;
    const Oz1Sweep& sw = *swo;
    auto& md = s.md;
    auto& cq = md->mesh_command_queue();
    const IssuePlan& ip = s.ip;
    const PanelPlan& pp = s.pp;
    const uint32_t nkb = ip.nkb, gw = s.gw, gi = s.gi, inner = sw.inner(), ndiag = s.c.nslice();
    // 発行（SetRuntimeArgs + Enqueue）とデバイス実行の待ちを分けて計る．
    // 発行はデバイス実行と重なりうるので，発行時間がそのまま削減余地とは限らない．
    double t_issue = 0.0;
    // バンド 1 本ぶんの掃き出し（ラウンドごとに GEMM → 同期 → 累積か再構成 → 同期．ラウンドと発行の単位は Oz1Sweep）．
    // バンドが 1 本なら出力全体を 1 回で掃く．timed のときだけ時間とテレメトリの区間を記録する．
    auto sweep_band = [&](const Band& bd, bool timed) {
        ReconCtx& rr = sw.recon(br, t, bd);
        bool first = true;
        for (uint32_t b = 0; b < nkb; ++b)
        for (uint32_t a = 0; a <= inner; ++a) {
            const double sg = timed ? oz_telem::now() : 0.0;   // テレメトリの区間（段 gemm・recon）の始まり
            auto tg = now();
            sw.gemms(bd, b, a, first, oz1_no_note);
            auto ti = now();
            dist::Finish(cq);
            auto tr = now();
            const double sr = timed ? oz_telem::now() : 0.0;
            if (timed) t_issue += std::chrono::duration<double>(ti - tg).count() / gi;
            // PAIRLOOP と fuse は累積が GEMM の中で済んでいるので何も投入せず（issued が偽），2 回目の同期も
            // 再構成の区間も置かない（空の同期と区間を再構成段の時間と TELEM の recon に足さない）．
            const bool issued = sw.post(rr, b, a, first, oz1_no_note);
            first = false;
            if (issued) dist::Finish(cq);
            if (timed) {
                t.gemm += std::chrono::duration<double>(tr - tg).count() / gi;
                // 再構成の区間の終わりは記録（note_interval のホストの処理）より前に取る．後で取ると，
                // その時間が recon に入る
                const auto te = now();
                const double se = oz_telem::now();
                oz_eval::note_interval("gemm", "sweep", sg, sr);
                if (issued) {
                    t.recon_dev += std::chrono::duration<double>(te - tr).count() / gi;
                    oz_eval::note_interval("recon", "sweep", sr, se);
                }
            }
        }
        // 遅延再構成では，全ブロックを足し終えた Sbig に対して最後に 1 回だけ
        // Horner を回す（ここで FP64 まで組み立てる）．
        if (ip.defer) {
            const double sr = timed ? oz_telem::now() : 0.0;
            auto tr = now();
            sw.final(rr, oz1_no_note);
            dist::Finish(cq);
            if (timed) {
                t.recon_dev += std::chrono::duration<double>(now() - tr).count() / gi;
                oz_eval::note_interval("recon", "sweep", sr, oz_telem::now());
            }
        }
    };
    auto sweep = [&](bool timed) {
        for (const auto& bd : pp.bands) sweep_band(bd, timed);
    };
    // 掃き出しは 1 回の中で GEMM と再構成を交互に流す（段ごとに分けては流せない）ので，温めと計時は掃き出し全体を
    // 段「sweep」として行う（cold は gw 回，steady は定常状態まで．oz_timing.hpp の warm_up）．掃き出しは中で同期するので
    // 投入の単位は 1 回（14080^3 の BFP2 は 1 回 27 s あり，3 回を単位にすると窓が 80 s になる）．計時の中の GEMM と
    // 再構成の部分は，それぞれ段 gemm・recon（群 sweep の一部．TELEM の sub=1）の区間として記録する
    oz_eval::warm_up("sweep", nullptr, gw, 1, [&] { sweep(false); }, [] {});
    const oz_eval::TimedMark tm = oz_eval::begin_timed("sweep");
    for (uint32_t it = 0; it < gi; ++it) sweep(true);
    oz_eval::end_timed(tm, gi);
    // 「Issue   : GEMM の発行 <ms> ms/iter」と「(<n> パネル x <m> 本 x <k> K ブロック)」は scripts/lib/logparse.py・probe_checks.py が
    // 読む（n・m・k は CONFIG の npanels=・gemm_issues=・nkb= にもある．発行の時間は計時の値なので CONFIG には出さない）
    fmt::print("Issue   : GEMM の発行 {:.1f} ms/iter，デバイス待ち {:.1f} ms"
               " ({} パネル x {} 本 x {} K ブロック)\n",
               t_issue * 1e3, (t.gemm - t_issue) * 1e3,
               pp.panels.size(), ip.gemm_issues(ndiag), nkb);
    return swo;
}

// 精度（参照解との比べ）を表示し，OZ_DUMP_C なら書き出す．検証を省いたときは精度欄を nan にする（0 と 99 bit に
// すると誤差 0 の実行と見分けられない．欄の数と順序は変えない．gemm_bf16ext と同じ）．dump_ok は書き出せたか
static oz_eval::Acc report_accuracy(const RunCfg& c, const Operands& op, const std::vector<double>& C, bool& dump_ok) {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    oz_eval::Acc accu{qnan, qnan, qnan};
    dump_ok = true;
    if (op.verify) {
        accu = oz_eval::accuracy(C, op.Chi, op.Clo);
        oz_eval::print_accuracy(accu);
        // OZ_DUMP_C（調査用）．書けなければ失敗として返す（main がデバイスを閉じてから終わる）
        dump_ok = oz_eval::dump_result(C, op.Chi, op.Clo, op.A, op.Bm, c.M, c.K, c.N);
    } else {
        fmt::print("Accuracy : (skipped, OZ_NOVERIFY)\n");
    }
    return accu;
}

// 時間の内訳・スループット・DRAM->DRAM（と OZ_TIME_WHOLE の WHOLE-PIPELINE）の行を表示し，段ごとの計時の和 DRAM->DRAM
// （入力 A,B がデバイス DRAM に常駐した状態から出力 C が DRAM に得られるまで．A,B のアップロードと C の読み戻しは
// 含めず，デバイス上の全工程（スケール算出・正規化・スライス分解・GEMM・再構成）だけを数える）の秒を返す．論文の全体の時間は
// 全工程を続けて流した WHOLE-PIPELINE（2026-11-03 から），DRAM->DRAM の段の内訳はフェーズの内訳（図 3）に使う．
// time_whole は OZ_TIME_WHOLE（Oz1Diag）
static double report_times(const RunCfg& c, const IssuePlan& ip, const PairCount& pc, double peak_chip, const Times& t,
                           double macs, bool time_whole) {
    const uint32_t M = c.M, N = c.N, K = c.K;
    const int npairs = pc.npairs, ndiag = pc.ndiag;
    const double tops = macs / t.tot / 1e12;
    const double recon_dev_ms = t.recon_dev * 1e3;
    const double gemm_tops = macs / std::max(1e-9, t.gemm) / 1e12;
    // 1 K ブロックあたりの発行の本数は IssuePlan::gemm_issues（対ごとなら ndiag(ndiag+1)/2 本．PAIRLOOP は対を 1 本の中で
    // 回すので ndiag 本のまま）
    fmt::print("Compute  : {} {} GEMM(s) + on-device triple-float recon "
               "(single FP64 readback)\n",
               ip.gemm_issues((uint32_t)ndiag), ip.pairwise ? "pairwise" : "diagonal");
    // GEMM 段における生のスループット．対角ごとの積和数 (d+1)*M*N*K の総和は
    // スライス対の数 npairs = (L+1)(L+2)/2 で重み付けしたものに等しい．
    const double raw_macs = (double)npairs * (double)M * N * K;
    const double gemm_raw_tops = 2.0 * raw_macs / std::max(1e-9, t.gemm) / 1e12;
    fmt::print("Breakdown: GEMM build(once) {:.1f} ms "
               "| slice JIT(once) {:.1f} ms | scale+slice {:.1f} ms/iter "
               "(scale {:.1f}) | GEMM run {:.1f} ms/iter | recon dev "
               "{:.3f} ms/iter\n",
               t.build * 1e3, t.fill_build * 1e3,
               t.fill * 1e3, t.fill_scale * 1e3, t.gemm * 1e3, recon_dev_ms);
    fmt::print("Recon    : build+JIT(once) {:.1f} ms | dispatch {:.3f} "
               "ms/iter | FP64 readback {:.1f} ms\n",
               t.recon_build * 1e3, recon_dev_ms, t.recon_read * 1e3);
    // 先頭の値は RESULT の時間欄と同じ tot（1 回の構築・JIT ＋ 段の反復の平均 ＋ 読み戻し．main の t.tot）で割った値で，
    // 基準実装の End-to-end（転送込みの 1 回の端から端）とは別物．論文の全体の時間は下の WHOLE-PIPELINE（CONFIG の whole_ms），
    // フェーズの内訳は DRAM->DRAM（d2d_ms）の段の時間
    fmt::print("Throughput: build+stages+readback {:.3f} TOP/s useful fp-equiv "
               "({:.2f}% {} peak) | GEMM-phase {:.2f} TOP/s fp-equiv | "
               "GEMM-phase raw-{} {:.2f} TOP/s ({:.2f}% {} peak)\n",
               tops, 100.0 * tops / peak_chip, OZ_FMT.name, gemm_tops,
               OZ_FMT.name, gemm_raw_tops, 100.0 * gemm_raw_tops / peak_chip,
               OZ_FMT.name);
    const double t_d2d = t.fill + t.gemm + t.recon_dev;
    const double d2d_tops = macs / std::max(1e-12, t_d2d) / 1e12;
    // この行は照合が空白 2 つまで固定した正規表現で読む：表・図の値の出どころ scripts/lib/logparse.py の dram2dram
    // （と同じ実行の RESULT の段の時間との照合 stage_result_reasons．scripts/figures/update_fig_breakdown.py が使う），
    // 古いログの集計 scripts/legacy_logs.py・scripts/summarize_cool.py．scale+norm+slice の値（t.fill）は RESULT に無く，
    // 同じ値は CONFIG の slice_ms・gemm_ms・recon_ms・d2d_ms にもある（照合はそちらを先に読めばよい）．
    // 表示を整えるときは照合の正規表現と一緒に直す
    fmt::print("DRAM->DRAM : {:.3f} ms/iter  (scale+norm+slice {:.3f} | "
               "GEMM {:.3f} | recon {:.3f})  "
               "{:.4f} TFLOP/s fp64-equiv\n",
               t_d2d * 1e3, t.fill * 1e3, t.gemm * 1e3,
               t.recon_dev * 1e3, d2d_tops);
    // OZ_TIME_WHOLE：全工程を段の間で同期せずに続けて流した時間（論文の全体の時間．上の DRAM->DRAM は段ごとの
    // 計時の和で，論文はフェーズの内訳に使う）
    if (time_whole) {
        std::fflush(stdout);
        oz_drv::print_whole_pipeline(t.whole, t_d2d, macs);
    }
    if (t.dsc_calls) {
        double dc = (double)t.dsc_calls;
        fmt::print("  fill split: scale {:.3f} | glue(2^-e hop) {:.3f} | "
                   "slice {:.3f}  ms/iter (normalize fused into slice)\n",
                   t.dsc_scale / dc * 1e3, t.dsc_glue / dc * 1e3,
                   t.fill * 1e3 - (t.dsc_scale + t.dsc_glue) / dc * 1e3);
    }
    return t_d2d;
}

// 設定の記録（CONFIG 行）．RESULT 行だけでは形式も基数も分からず，あとからログを読み直す
// ときに構成を復元できない．
// 時間の指標名も明記する．RESULT の時間欄は両スキームで意味が違い，
// scheme I はプログラム構築とホストへの読み戻しまで含む tot，
// scheme II はデバイス各段の合計である．どちらでもデバイス上の全工程だけを数えた段ごとの計時の和 DRAM->DRAM を
// d2d_ms に出す（scheme II では RESULT の時間欄と一致する．論文はフェーズの内訳に使い，全体の時間は whole_ms）．
// fuse・defer は実際に採った経路を出す（OZ_NO_FUSE_ACC・OZ_NO_DEFER の状態ではない）．
// fuse は GEMM の出力経路で DRAM の S を読んで足す構成（PAIRLOOP では 0），
// defer は Int32 で累積して最後に 1 回だけ Horner を回す構成（PAIRLOOP を含む）．
// probe は OZ_RECON_PROBE に加えて，冒頭で「診断用」とした無誤差条件を破りうる
// つまみ（docs/env_vars.md の C）のどれかが立っていれば set にする．どれが
// 立っていたかは diag に '+' 区切りで残す．
// GEMM 段の探針と OZ_RECON_PROBE は環境変数の有無ではなく読んだ値で見る（=0 は既定と同じ構成）．
// 全工程を続けて流す計時（OZ_TIME_WHOLE．論文の全体の時間）は time_whole・whole_ms に残し，diag には数えない．
// どのつまみを diag に数えるかと並びは，つまみの表（common/oz_knobs.hpp の R_DIAG）が決める．
static void print_config(const RunCfg& c, const Oz1Diag& dg, const IssuePlan& ip, const PanelPlan& pp,
                         const PairCount& pc, const Times& t, double t_d2d, bool verify, const std::string& chash) {
    const std::string diag = oz_knobs::record(oz_knobs::W_I, oz_knobs::R_DIAG);
    OzLine cfg("CONFIG,ozaki_full");
    cfg.kv("fmt", OZ_FMT.name).kv("scheme", "I").kv("radix", c.radix).kv("dmax", c.dmax)
       .kv("p", c.p).kv("L", c.L).kv("npairs", pc.npairs).kv("ndiag", pc.ndiag).kv("Kc", ip.Kc).kv("nkb", ip.nkb)
       .kv("fidelity", oz_fidelity_name(c.fid)).kv("bands", pp.nband)
       .flag("pairloop", ip.pairloop).flag("fuse", ip.fuse).flag("defer", ip.defer).flag("verified", verify)
       // 調査用のつまみの状態．既定（probe=none）でないログは通常の測定ではない
       // （scripts/verify/check_paper_numbers.py が論文の表への採用を拒む）．
       .kv("probe", diag == "none" ? "none" : "set").kv("recon_probe", dg.recon_probe).kv("diag", diag)
       .kv("time_in_result", "tot_incl_build_and_readback").num("d2d_ms", "%.4f", t_d2d * 1e3);
    oz_drv::add_gemm_impl(cfg);
    // bw・shape は通常パネルの K ブロック幅とコアあたりの出力［タイル］（scheme II の欄と同じ意味．
    // 端数行のパネルは Cfg 行を見る），knobs は既定から外れたアブレーションのつまみ（oz1_knobs_record．精度の基準は保つが
    // 最下位のビットを変えうるものも含む．その注記），gemm_probe は GEMM 段の探針（組んだ Oz1Diag::gemm_probe．立っていれば
    // diag にも出る）
    cfg.kv("bw", pp.gU.bw).kv("shape", std::to_string(pp.gU.pm) + "x" + std::to_string(pp.gU.pn))
       .kv("knobs", oz1_knobs_record()).kv("gemm_probe", oz_gemm_probe_name(dg.gemm_probe));
    oz_drv::add_input_whole_timing(cfg, dg.time_whole, t.whole * 1e3);
    // 後から足した欄は末尾に置く（検査側は key=value で読むので位置に依らない）．
    // auto_p は OZ_FP64 が入力から選び直した p（選び直していなければ 0．OZ_FP64: の行の auto p= と同じ値），
    // npanels・gemm_issues は出力パネルの枚数と，パネル 1 枚・K ブロック 1 本あたりの GEMM の発行の本数（対ごとの発行なら
    // 対の数，そうでなければ対角の数．IssuePlan::gemm_issues．Issue 行の「n パネル x m 本 x k K ブロック」の n と m）
    cfg.kv("auto_p", c.auto_p ? c.p : 0).kv("npanels", pp.panels.size())
       .kv("gemm_issues", ip.gemm_issues(c.nslice()));
    oz_drv::add_common_tail(cfg);
    // 段の時間［ms/iter］（DRAM->DRAM 行の scale+norm+slice・GEMM・recon と同じ値．和が d2d_ms）．RESULT の時間欄は tot なので，
    // スライス段の値はこれまで人向けの DRAM->DRAM 行にしか無かった．計時の値なので，同じ構成の 2 本のログを比べる照合では
    // d2d_ms と同じく違ってよい欄に数える
    cfg.num("slice_ms", "%.4f", t.fill * 1e3).num("gemm_ms", "%.4f", t.gemm * 1e3).num("recon_ms", "%.4f", t.recon_dev * 1e3);
    // 結果 C のハッシュ（検証の有無に依らず出す．到達精度のログの CHECKSUM と同じ値で，照合が性能のログと突き合わせる）
    cfg.kv("chash", chash);
    // デバイスの格子と GEMM 段の格子（どちらも 列x行．Device 行・論文の「11×10」と同じ向き．gemm cfg・Cfg 行の grid= は 行x列），
    // 端数行のパネルの GEMM の格子（端数が無ければ none），計画のときの DRAM の空きの総量と最大の連続した空き［GB］
    // （2026-11-04．別のボードで構成が変わった理由をログが語るように．GEMM 以外の段はデバイスの格子の全部を使う）
    cfg.kv("dev_grid", std::to_string(pp.dev_gx) + "x" + std::to_string(pp.dev_gy))
       .kv("gemm_grid", std::to_string(pp.gU.cc) + "x" + std::to_string(pp.gU.cr))
       .kv("gemm_grid_tail", pp.rem_rows ? std::to_string(pp.gR.cc) + "x" + std::to_string(pp.gR.cr) : std::string("none"))
       .num("dram_free_gb", "%.2f", pp.dram_total / 1e9).num("dram_largest_gb", "%.2f", pp.dram_largest / 1e9);
    cfg.print();
}

// 読み戻した FP64 の結果を並べ替えて精度を測り，時間の内訳・RESULT・VERIFY・CONFIG の
// 各行を表示する．デバイスには触らない．精度の検査に通ったかを返す．
static bool report(const RunCfg& c, const Oz1Diag& dg, const IssuePlan& ip, const PanelPlan& pp,
                   const PairCount& pc, double peak_chip, const Times& t,
                   const Operands& op, const std::vector<uint32_t>& Otile) {
    const uint32_t M = c.M, N = c.N, K = c.K;
    // デバイスが FP64 まで組み立てて（radix^-(L+2) と 2 のべきの除数 s_A,s_B を
    // 含めて．main の注記）出しているので，ホストはタイル順から行優先へ並べ替えるだけである．
    const std::vector<double> C = untilize_f64(Otile, M, N);
    const double macs = 2.0 * (double)M * N * K;
    fmt::print("-------------------------------------------------------------------\n");
    // 参照解（倍々精度の Chi+Clo）に対する精度（oz_eval::accuracy）
    bool dump_ok = true;
    const oz_eval::Acc accu = report_accuracy(c, op, C, dump_ok);
    const bool pass = oz_eval::accuracy_ok(accu, op.verify);   // VERIFY 行は精度だけ．書き出しの失敗は戻り値へ
    const double t_d2d = report_times(c, ip, pc, peak_chip, t, macs, dg.time_whole);
    // RESULT 行は照合が欄の位置で読む（scripts/lib/oz_logs.py の RESULT_SCHEMA．欄の数が違う行は読めない）．時間欄は tot
    // （CONFIG の time_in_result），12・13 列目（RESULT と種類の印を含めて数える）が GEMM 段と再構成段［ms/iter］で，
    // scripts/lib/logparse.py の stage_result_reasons が DRAM->DRAM 行と突き合わせる．欄を足すなら CONFIG に足す
    fmt::print("RESULT,ozaki_full,{},{},{},{},{},{},{},{:.4f},{:.3e},{:.2f},"
               "{:.1f},{:.3f}\n",
               M, N, K, c.p, c.L, pc.npairs, pc.ndiag, t.tot * 1e3, accu.rel_fro, accu.ebits,
               t.gemm * 1e3, t.recon_dev * 1e3);
    fmt::print("VERIFY,ozaki_full,{}\n", op.verify ? (pass ? "pass" : "fail") : "skipped");
    // 段ごとの計時区間のクロック・電流・電力・温度（oz_timing.hpp の print_telem．CONFIG の timing= が計時の方式）
    std::fflush(stdout);
    oz_eval::print_telem();
    print_config(c, dg, ip, pp, pc, t, t_d2d, op.verify, oz_eval::chash_hex(C));
    return pass && dump_ok;
}

// 使い方: ozaki1_{bf16,fp16,fp32,bfp8,fp8,int8,bfp4,bfp2} [M] [N] [K] [p] [L] [warmup] [iters]
//   既定値: 256 256 512 4 (p-1) 1 3．warmup・iters は 1 以上（oz_driver.hpp の read_counts）
//   既定の p=4 は動作確認用で，53 bit の検査には届かない．形式ごとに FP64 相当に要る p と L（L=p-1）の
//   正本は scripts/routes.tsv（scripts/diagnose/select_slice_count.sh の規則で選んだ）．
//
// デバイスの資源はすべてこの関数の try ブロックが持ち，md->close() のあとで解放される．
// 確保の順序は Pbig → 入力側（setup_inputs）→ GEMM のプログラム → 累積面・再構成である
// （DRAM の割り付けが性能を左右するので，この順序は変えないこと）．
int main(int argc, char** argv) {
    // 削除したつまみと scheme II だけのつまみ（OZ2_*）を止める．II のつまみを黙って無視すると，b の選び直し（OZ2_FP64）や
    // 法の集合（OZ2_MODULI）などを入れたつもりで既定の構成を測る（scheme I でスライス数を選び直すのは OZ_FP64）
    oz_drv::reject_env(oz_knobs::W_I, argv[0]);
    RunCfg c = parse_run_config(argc, argv);   // 位置引数とこのスキームのつまみ（不正なら終了コード 2）
    const Oz1Knobs kn = oz1_knobs_from_env();  // 計画と GEMM の組み方を変えるつまみ（値は上で検査済み）
    const Oz1Diag dg = oz1_diag_from_env();    // 段の組み方・計時の流れを変える診断と測定モードのつまみ（同）

    bool pass = true;
    try {
        auto md = dist::MeshDevice::create_unit_mesh(0);
        const DevInfo dv = device_info(md, c.fid);
        // 見出しは OZ_FP64 が p・L を選び直した後に出す（前に出すと指定値の p・L が残る）
        const Operands op = prepare_operands(c);   // OZ_FP64 なら c.p・c.L が変わる
        const PairCount pc = count_pairs(c.p, c.L);
        print_header(c, dv, pc);

        // ---- 計画（デバイスの空き容量を照会するだけで，まだ何も確保しない）----
        const IssuePlan ip = plan_issue(c, dram_free_bytes(md), kn);
        if (!ip.ok) { md->close(); return 2; }
        const PanelPlan pp = plan_panels(c, ip, kn, dv.gd, dram_free(md));
        if (!pp.ok) { md->close(); return 2; }
        // 再構成の定数（ホストの計算のみ．OZ_NSMALL の上界の検査もここ．何も確保する前に止める）
        const std::optional<ReconScale> rso = recon_scale(c, ip.defer, kn.nsmall, dg);
        if (!rso) { md->close(); return 2; }
        const ReconScale& rs = *rso;

        // ---- 確保と入力の準備 ----
        const auto Pbig = alloc_pbig(md, pp);
        print_cfg(pp, ip.nkb);
        Times t;
        const uint32_t gw = c.warmup, gi = c.iters;   // どちらも 1 以上（parse_run_config）
        InputBufs in = setup_inputs(md, c, pp, dv.gd, op);
        FillStage fs = run_scale_slice(md, c, dg, pp, in, gw, gi, t);

        // ---- GEMM と再構成 ----
        // デバイスが S = Σ_d radix^(L-d) P_d を triple-float で計算し，
        // そのまま R = (hi+mid+lo) * radix^-(L+2) * s_A * s_B を FP64 へ組み立てる
        // （recon_f64_writer.cpp）．s_A = 2^-dA・s_B = 2^-dB は正規化に実際に用いた除数（dA・dB は指数差分
        // shift-e で，s = 2^(e-nshift)．e は行・列の最大値の BF16 表現（丸めの向きは bf_ceil・tau_above）を 2 のべきへ
        // 上げた 2^e の指数で，2^e は最大値以上とは限らない：切り捨ての形式で最大値が 2^k の直上（相対 2^-7 未満）の
        // 行・列では 2^k になる．OZ_FP64 のモデルの s と同じ量（ozaki1_autop.hpp の oz1_dev_exp））．ホストはタイル順から
        // 行優先へ並べ替えるだけである．
        // 工程融合：再構成の出力に行ごと・列ごとのスケールを掛けるための独立した
        // 2 パスは持たない．スケールは 2 のべきちょうどなので，FP64 を組み立てる
        // ライタが指数へ足すだけで済み，組み立てのパスに畳み込める
        // （recon_f64_writer.cpp）．
        // 再構成を組む時点は経路で違う．run_single_kblock（K ブロック 1 本で対ごとにも PAIRLOOP でも回さない．
        // INT8 と一部のアブレーション）はバンドの GEMM の後に組む．run_sweep（K ブロック複数・対ごと・PAIRLOOP．
        // 既定の FP 形式はこちら）は全パネルの GEMM のプログラムを組んだ後・GEMM を回す前に組む（K ブロックごとに
        // 再構成を呼ぶ OZ_NO_DEFER の経路がそれを要るため）．どちらも再構成の出力バッファを GEMM のプログラムより
        // 後に確保する：先に確保すると GEMM 段の DRAM の割り付けが変わり，発行が遅くなる．
        // 既定は遅延再構成（Int32 で累積して最後に 1 回 Horner）である．K ブロックごとに Horner を回して足し込むのは，
        // K ブロックが複数で遅延再構成が取れない構成だけである：OZ_NO_DEFER，Int32 の累積が 2^31 を超えうる長い K
        // （oz_defer_fits が外れる．BF16 なら K が約 2.1e5 を超えるとき．つまみによらない），OZ_NO_FUSE_ACC で
        // 累積面が DRAM に入らないとき（plan_issue の ip.defer）．
        if (dg.softfloat)
            fmt::print("SoftFloat: FP64<->triple-float の変換を C の double 式"
                       "（libgcc soft-float）で行う測定モード\n");
        BandRecon br;        // 再構成（rcn.out が FP64 の結果．最後に読み戻す）
        AccumCtx acc;        // 遅延再構成の累積面（main の終わりまで持つ）
        std::vector<GemmCtx> ctxs;   // 掃き出しの構成の全パネルの GEMM（main の終わりまで持つ）
        // 遅延再構成を使うか（ip.defer）は実行中に変わらない（累積面が取れなければ run_sweep が止まる）
        const StageCtx s{md, c, kn, dg, ip, pp, Pbig, in, rs, gw, gi};
        oz_whole::Check wck;   // OZ_TIME_WHOLE の検査（流し直しの C・番兵の検査の結果・投入の本数）
        if (ip.single_kblock()) {
            run_single_kblock(s, br, t);
            if (dg.time_whole) whole_single_kblock(s, fs, br, t, wck);
        } else {
            const std::optional<Oz1Sweep> sw = run_sweep(s, br, acc, ctxs, t);
            if (!sw) return 2;   // run_sweep が md->close() まで済ませている
            if (dg.time_whole) whole_sweep(*sw, fs, br, t, wck);
        }

        std::vector<uint32_t> Otile;
        recon_read_f64(md, br.rcn, Otile, &t.recon_read);
        // 全工程の流し直しの検査（whole_check：番兵で埋めてから 1 回流した C と投入の本数）と，計時の後の C が
        // 段ごとの計時の後の C とビット単位で同じか．identical はすべて通ったときだけ 1（oz_whole::report）
        if (dg.time_whole) {
            std::fflush(stdout);
            pass &= oz_whole::report(wck, wck.pre == Otile);
        }
        // 範囲外の入力の検査は C を読み終えてから．計時区間の外である．範囲外なら理由を
        // 表示済みで，デバイスを閉じてから 2 で終わる（selftest.sh の range guard がこの終わり方を見る）．
        const bool in_range = expdelta_check(md, fs.edA, "行", c.bf_ceil) & expdelta_check(md, fs.edB, "列", c.bf_ceil);
        if (!in_range) { md->close(); return 2; }
        const double t_recon = t.recon_build + t.recon_dev + t.recon_read;
        oz_dbg("recon done");
        t.tot = t.build + t.fill_build + t.fill + t.gemm + t_recon;

        pass &= report(c, dg, ip, pp, pc, dv.peak_chip, t, op, Otile);
        pass &= md->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed: {}\n", e.what());
        throw;
    }
    // 検証を省いた実行（OZ_NOVERIFY）は精度を判定していないので，通っても "Test Passed" とは書かない（終了コードは 0）
    std::fflush(stdout);
    return oz_drv::finish(pass, oz_eval::verify_enabled());
}
