// SPDX-License-Identifier: Apache-2.0
#pragma once
// マルチキャスト行列積のプログラム構築（全 GEMM で共有）
// ---------------------------------------------------------------------------
// 基準実装の 3 本（gemm_bf16 の bf16_gemm_spec，gemm_bf16ext の gemm_fp32out，gemm_int8 の gemm_int8．どれも下の
// run_mcast_gemm で流す），gemm_exact_check，尾崎 scheme I の gemm_build（ozaki1_gemm.hpp），scheme II の
// run_partial_gemm（ozaki2_main.cpp）が共通に組むプログラムをここに置く．
// ozaki_gemm_host.hpp（mcast_*_args・make_cb・OptConfig）を取り込む．
//
// A は同じ行のコアへ，B は同じ列のコアへマルチキャストしてオペランドを再利用し，
// DRAM 帯域を削減する（論文の並列化とデータ移動の節．\label{ssec:par}）．データ移動は kernels/mcast_split_dm.cpp の 1 本を
// define で出し分け，A と B を別々の RISC に置いて並行させる：
//   A 用（RISCV_1・NOC 1）：左端の列が 1 ブロックの前半，右端の列が後半を DRAM から読んで
//                           同じ行の自分以外の全コアへ配り，その間の列は受けるだけ．
//   B 用（RISCV_0・NOC 0）：上端の行が前半，下端の行が後半を同じ列へ配り，その間の行は
//                           受けるだけ．K ループの後に出力も書く（ライタを兼ねる）．
// A を NOC 1 に置くのは，試作でそれが最も速かったからである（A を NOC 0 にすると FP32 で 2 割遅い）．
// 比べた他の構成（tt-metal の単一 RISC リーダ 4 種＋ライタ 2 種，1 コアが全部を配る構成，
// 入力 CB の 3 段化など）は docs/gemm_opt_results.md にある．
//
// データ移動カーネルの実行時引数はリーダの 33 個（tt-metal のリーダの並びからバッチの引数と読まないブロックの高さを
// 除いたもの．mcast_reader_args）に，後半の送信用の 7 個（33..39 番）と，B 側だけライタの引数（40 番〜）を続けたもの．
// 番号は kernels/mcast_args.h の 1 つの表でホストとカーネルが共有する．
// McastGemm::set_args がこの連結を受け持ち，個数が表と合わなければ止まる．
//
// 共有の行列積カーネル kernels/bmm_int32_spill.cpp を使う実装（尾崎の両スキーム・gemm_int8・
// gemm_exact_check）は，CB・計算カーネル・コンパイル時引数の組み方を spill_gemm_spec で共有する
// （融合・対ループの CB と UnpackToDestFp32 の指定を 1 か所に置き，検査の構成と本番の構成が
// ずれないようにするため）．
//
// 実装ごとに違うのは CB の形式と構成，ライタ（標準か融合か）と計算カーネル，A/B/C の開始タイルのずらし，
// 計算カーネルの実行時引数，GEMM 段の診断用の探針（尾崎の 2 つのドライバだけ）なので，それを McastGemmSpec で渡す．
// 組み立て（build_mcast_gemm）は環境変数を読まない（spec が組む構成の全部を持つ）．
// 基準実装の 3 本（gemm_bf16・gemm_bf16ext・gemm_int8）の DRAM の器の確保から計時と読み戻しまでは run_mcast_gemm が受け持つ．
// tt-metalium の呼び出し順は CB → データ移動 最大 6 本（A の前半・後半・受信，B の前半・後半・受信．
// 受信は行／列が 3 コア以上か 1 ブロックが 1 タイルのとき，後半は 1 ブロックが 2 タイル以上のときだけ）→ 計算 →
// セマフォ 6 → コアごとの実行時引数．build_mcast_gemm はこの順に下の部品（mcast_build_check・mcast_make_cbs・
// mcast_make_kernels・mcast_make_sems・mcast_set_core_args）を呼ぶ．
// カーネル ID・セマフォ ID・CB の配置は作成順で決まるので，この順を変えないこと．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>   // TensorAccessorArgs（データ移動カーネルのコンパイル時引数）
#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/ozaki_gemm_host.hpp"
#include "common/gemm_probe.hpp"         // GEMM 段の診断用の探針（McastGemmSpec の probe）
#include "common/oz_timing.hpp"          // 計時の手順（run_mcast_gemm の time_dispatch）
#include "common/oz_paths.hpp"           // カーネルの置き場所 OZ_COMMON_KDIR
#include "common/kernels/mcast_args.h"
#include "common/kernels/gemm_compute_args.h"   // 計算カーネルの引数の並びと GEMM 段の CB 番号（カーネルと共有）

// 追加の CB 1 つ（容量 ntiles 枚，ページ page バイト）．make_cb に渡す．
struct McastCB {
    tt::CBIndex idx;
    tt::DataFormat fmt;
    uint32_t ntiles, page;
};

struct McastGemmSpec {
    // c_0・c_1：A・B の入力（2 重緩衝で pm*bw*2・pn*bw*2 枚）
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b;
    uint32_t in_tile = 0;
    // c_16：最終出力，c_24：K スピルの中間．どちらも pm*pn 枚で，ページは out_tile．
    // out_shared なら両者は 1 つの CB（同じ L1 領域）を共有し，そうでなければ別々に作る．
    tt::DataFormat c16_fmt = tt::DataFormat::Float16_b, c24_fmt = tt::DataFormat::Float16_b;
    uint32_t out_tile = 0;
    bool out_shared = true;
    std::vector<McastCB> extra_cbs;      // c_24 の後に，この順に作る
    // ライタ：false なら標準（tt-metal の writer_bmm_tile_layout.cpp と同じ書き順．引数はバッチの 2 個を除いた並びで，
    // mcast_args.h の W_*），true なら書く前に既存の S を c_2 へ先読みする融合ライタ（scheme I の FUSE．mcast_split_dm.cpp の
    // WRITER=2）
    bool writer_fused = false;
    std::string compute;                 // 計算カーネル
    tt::tt_metal::ComputeConfig ccfg;    // compile_args を含む
    // 空でなければ各コアの計算カーネルへ渡す．融合ライタ（writer_fused）では CR_ACC（既存の S を足すか）まで要り，
    // build_mcast_gemm はライタの W_ACC もここから取る（計算側とライタの acc の出どころを 1 つにする．食い違うと計算側が
    // c_2 の S を待ち，ライタが先読みしないまま止まる）
    std::vector<uint32_t> compute_rt;
    uint32_t dA = 0, dB = 0, dW = 0;     // A/B/C の開始タイルのずらし（scheme I のパネル）
    // 診断用の探針（common/gemm_probe.hpp．答えが壊れる）．既定は本番の構成（どれも 0）で，立っていると CONFIG 行へ
    // probe=set を残す尾崎の 2 つのドライバ（ozaki1_gemm.hpp の gemm_build・ozaki2_main.cpp の run_partial_gemm）だけが
    // oz_gemm_probe() で詰める
    GemmProbe probe;
};

struct McastGemm {
    // 計算カーネル（尾崎 I の gemm_run が発行ごとに実行時引数を付け替える）．データ移動カーネルと
    // セマフォは build_mcast_gemm の中だけで使う（コアごとのカーネルは下の Core が持つ）．
    tt::tt_metal::KernelHandle ck = 0;
    // コアごとのカーネルと，渡したリーダの引数（scheme I の gemm_run が付け替えに使う）．
    // 並びは cy が外側・cx が内側で，実行時引数を設定した順である．
    // ka は A 側，kb は B 側のカーネル，xa・xb は A 側・B 側の後半の送信用の 7 個（mcast_args.h の X_*）．
    struct Core {
        tt::tt_metal::CoreCoord core; tt::tt_metal::KernelHandle ka, kb; std::vector<uint32_t> ra;
        std::vector<uint32_t> xa, xb;
    };
    std::vector<Core> cores;
    // コア 1 つにリーダの引数 ra（RA_COUNT 個）とライタの引数 wa（W_COUNT 個．標準のライタは W_ACC を読まない）を設定する．
    // 個数は kernels/mcast_args.h の表で決まる．1 つずれるとカーネルは黙って隣の値を読むので，
    // ここで合わなければ止める．
    void set_args(tt::tt_metal::Program& prog, const Core& c,
                  const std::vector<uint32_t>& ra, const std::vector<uint32_t>& wa) const {
        namespace ma = mcast_args;
        TT_FATAL(ra.size() == ma::RA_COUNT, "McastGemm::set_args: リーダの引数が {} 個（{} 個のはず）",
                 ra.size(), (uint32_t)ma::RA_COUNT);
        TT_FATAL(c.xa.size() == ma::X_COUNT && c.xb.size() == ma::X_COUNT,
                 "McastGemm::set_args: 後半の送信の引数が {}/{} 個（{} 個のはず）", c.xa.size(), c.xb.size(),
                 ma::X_COUNT);
        TT_FATAL(wa.size() == ma::W_COUNT, "McastGemm::set_args: ライタの引数が {} 個（{} 個のはず）", wa.size(),
                 (uint32_t)ma::W_COUNT);
        std::vector<uint32_t> a = ra;
        a.insert(a.end(), c.xa.begin(), c.xa.end());
        tt::tt_metal::SetRuntimeArgs(prog, c.ka, c.core, a);
        std::vector<uint32_t> b = ra;
        b.insert(b.end(), c.xb.begin(), c.xb.end());
        b.insert(b.end(), wa.begin(), wa.end());
        tt::tt_metal::SetRuntimeArgs(prog, c.kb, c.core, b);
    }
    // K 方向の読みを付け替えて組み直す（尾崎 I の gemm_run，尾崎 II の run_partial_gemm，gemm_exact_check の付け替え）．
    // 構築時のリーダの引数に対して A・B の開始タイルを aoff・boff だけずらし，K ブロック数を nb にして，ライタの引数 wa と
    // 一緒にコア c へ設定する（A 側・B 側の両方のカーネルへ渡る）．計算カーネルの実行時引数（K ブロック数を含む）は，
    // 計算カーネルが実装ごとに違うので呼び出し側が設定する
    void retarget_k(tt::tt_metal::Program& prog, const Core& c, uint32_t aoff, uint32_t boff, uint32_t nb,
                    const std::vector<uint32_t>& wa) const {
        namespace ma = mcast_args;
        std::vector<uint32_t> ra = c.ra;
        ra[ma::RA_A_START] += aoff;
        ra[ma::RA_B_START] += boff;
        ra[ma::RA_NUM_BLOCKS] = nb;
        set_args(prog, c, ra, wa);
    }
};

// コアあたりの形の前提（計算カーネルの引数を作る前と build_mcast_gemm の冒頭で見る）．
// 構成探索（choose_cfg・choose_cfg_1b・choose_remainder_cfg）を通した構成は満たしているが，
// 引数を作る関数（下の mcast_compute_args）は pm/sh・pn/sw・Kt/bw で割るので，検査はその前に要る
// （build_mcast_gemm の中だけで見ると，spill_gemm_spec が先にゼロで割ってしまう）．どれも破ると
// 止まらずに黙って壊れる：
//   sh|pm・sw|pn  リーダは pm*bw 枚を push し，計算側は sh*bw*(pm/sh) 枚を pop する．
//                 割り切れないと入力の push/pop 数が合わず，c_24 の輪も元の位置へ戻らない
//   bw|Kt         Kt/bw で切り捨てるので K の末尾を黙って落とす
static inline void mcast_shape_check(const OptConfig& g, uint32_t Kt, const char* who) {
    TT_FATAL(g.pm > 0 && g.pn > 0 && g.sh > 0 && g.sw > 0 && g.bw > 0,
             "{}: 形が 0 を含む（pm={} pn={} sh={} sw={} bw={}）", who, g.pm, g.pn, g.sh, g.sw, g.bw);
    TT_FATAL(g.pm % g.sh == 0 && g.pn % g.sw == 0,
             "{}: サブブロック {}x{} がコアあたりの出力 {}x{} を割り切らない", who, g.sh, g.sw, g.pm, g.pn);
    TT_FATAL(Kt >= g.bw && Kt % g.bw == 0, "{}: K ブロック幅 bw={} が Kt={} を割り切らない", who, g.bw, Kt);
}

// 計算カーネルの形状パラメタ（compile_args の先頭 CA_STD_COUNT 個．並びは common/kernels/gemm_compute_args.h の ComputeCT で，
// tt-metal の標準の行列積カーネルと同じ）．その後ろ（CA_FUSE 以降）は呼び出し側が足す（spill_gemm_spec）．
// バッチ数 CA_BATCH は 1（データ移動カーネルがバッチを扱わない．並びを共有する標準カーネルが読むので詰める）．
static inline std::vector<uint32_t> mcast_compute_args(const OptConfig& g, uint32_t Kt) {
    mcast_shape_check(g, Kt, "mcast_compute_args");   // 下で割る前に
    namespace ga = gemm_args;
    const uint32_t pm = g.pm, pn = g.pn, sh = g.sh, sw = g.sw, bw = g.bw;
    const uint32_t in0_ns = pm / sh, in1_ns = pn / sw;
    std::vector<uint32_t> a(ga::CA_STD_COUNT);
    a[ga::CA_IN0_BLOCK_W] = bw;
    a[ga::CA_IN0_NUM_SUBBLOCKS] = in0_ns;
    a[ga::CA_IN0_BLOCK_NUM_TILES] = sh * bw * in0_ns;
    a[ga::CA_IN0_SUBBLOCK_NUM_TILES] = sh * bw;
    a[ga::CA_IN1_NUM_SUBBLOCKS] = in1_ns;
    a[ga::CA_IN1_BLOCK_NUM_TILES] = sw * bw * in1_ns;
    a[ga::CA_IN1_PER_CORE_W] = sw * in1_ns;
    a[ga::CA_NUM_BLOCKS] = Kt / bw;
    a[ga::CA_OUT_SUBBLOCK_H] = sh;
    a[ga::CA_OUT_SUBBLOCK_W] = sw;
    a[ga::CA_OUT_SUBBLOCK_NUM_TILES] = sh * sw;
    a[ga::CA_BATCH] = 1u;
    static_assert(ga::CA_STD_COUNT == 12, "ComputeCT の標準の並びに欄を足したらここで詰める");
    return a;
}

// build_mcast_gemm が spec どおりに作る CB の 1 コアあたりの合計バイト数（ozaki_gemm_host.hpp の mcast_cb_bytes に
// extra_cbs を足したもの．構成探索の見積り cfg_fits と同じ式）．build_mcast_gemm の CB の作り方と同じ数え方なので，
// 構成を選ぶ側が「実際に作る CB が予算に収まるか」を候補ごとに見るのに使う（gemm_exact_check・gemm_int8・gemm_bf16ext の
// choose_cfg の accept．融合・対ループの CB は spill_gemm_spec が extra_cbs に積むので，見積りの定数を持ち回らずに済む）．
static inline uint64_t mcast_gemm_l1_bytes(const OptConfig& g, const McastGemmSpec& s) {
    uint64_t extra = 0;
    for (const auto& e : s.extra_cbs) extra += (uint64_t)e.ntiles * e.page;
    return mcast_cb_bytes(g.pm, g.pn, g.bw, s.in_tile, s.out_tile, s.out_shared, extra);
}
// 実際に作る CB が L1 の予算に収まることを，CB の確保より前に確かめる（構成探索の accept で mcast_gemm_l1_bytes を
// 見ていない構成を止める安全網．who は止めたときの文に出す呼び出し側の名前．budget は予算［B］で，既定は共有の
// oz_l1_budget．尾崎 I は main の冒頭で 1 度読んだ値（Oz1Knobs::l1_budget）を渡す）
static inline void mcast_check_l1(const OptConfig& g, const McastGemmSpec& s, const char* who,
                                  uint32_t budget = oz_l1_budget()) {
    const uint64_t l1 = mcast_gemm_l1_bytes(g, s);
    TT_FATAL(l1 <= (uint64_t)budget,
             "{}: 構成 {}x{} コア・コアあたり {}x{}・bw={} の CB が {} KiB で L1 の予算 {} KiB を超える"
             "（c_16・c_24 の置き方と追加の CB を含めて数える．構成探索の accept で実際の CB を見ていない構成）",
             who, g.cr, g.cc, g.pm, g.pn, g.bw, l1 / 1024, budget / 1024);
}
// 計算カーネルの DST に載るタイルの枚数：16 ビットのタイルで half-sync 8 枚・full-sync 16 枚，32 ビット（fp32_dest_acc）なら
// 半分．サブブロック sh*sw 枚がこれを超えると DST の別の区画を書き潰す（build_mcast_gemm が止める．gemm_exact_check の
// baddst はこの数を超える構成をわざと作る）
static inline uint32_t mcast_dst_tiles(const tt::tt_metal::ComputeConfig& c) {
    return (c.dst_full_sync_en ? 16u : 8u) / (c.fp32_dest_acc_en ? 2u : 1u);
}

// ===== 共有の行列積カーネル kernels/bmm_int32_spill.cpp の仕様 =====
// 尾崎の両スキーム（ozaki1_gemm.hpp の gemm_build，ozaki2_main.cpp の run_partial_gemm），gemm_int8，
// gemm_exact_check が同じ CB の構成・UnpackToDestFp32・コンパイル時引数で組むように，ここで作る．
//   acc_fmt   部分積（DST と c_24）の形式．INT8 経路は Int32，それ以外は Float32
//   fuse      出力経路で DRAM の既存の S（Int32）を読んで足す（尾崎 I の遅延再構成）．
//             c_2（S の先読み，2 サブブロック）・c_3（FP32 の中間，1 サブブロック）を足し，
//             ライタを融合ライタ（mcast_split_dm.cpp の WRITER=2）にする
//   pairloop  対の境界で FP32 のアキュムレータを畳んで L1 の Int32 面 c_5 へ足す（尾崎 I）．
//             c_3 と c_5（pm*pn + 1 サブブロックの輪）を足す．fuse と排他
//   blockmm   matmul_block を使う（OZ_NO_BLOCKMM で切る）
// 融合・対ループの出力 c_16 は Int32（2 の補数）になる．
// 計算カーネルの実行時引数（K ブロック数 nb・融合の acc・対ループの bpp．gemm_compute_args.h の ComputeRT）は
// 呼び出し側が spill_compute_rt で compute_rt に入れる（発行ごとに変わるものは spill_check_run で検査してから渡す）．
struct SpillOpts {
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat acc_fmt = tt::DataFormat::Float32;
    tt::tt_metal::MathFidelity fidelity = tt::tt_metal::MathFidelity::LoFi;
    bool fuse = false, pairloop = false, blockmm = true;
    bool out_shared = true;   // c_16 と c_24 に同じ L1 領域を使わせるか（gemm_int8 だけ別々）
};
static inline McastGemmSpec spill_gemm_spec(const OptConfig& g, uint32_t Kt, const SpillOpts& o) {
    namespace tm = tt::tt_metal;
    // 融合（DRAM の S を先読みして足す）と対ループ（L1 の累積面へ足す）は足す相手が 2 つになると
    // DST（32 bit タイル 4 面）に収まらないので排他である．計算カーネルもそれを前提に 2 タイルずつ回す．
    TT_FATAL(!(o.fuse && o.pairloop), "spill_gemm_spec: fuse と pairloop は同時に立てられない");
    // 融合・対ループの第 2 パスは行列積の結果を FP32 とみなして typecast<Float32,Int32> する．
    // INT8 経路の DST は Int32（符号-絶対値）なので，そのまま通すと値が壊れる．尾崎 I の INT8 経路は
    // 融合も対ループも使わない（ozaki1/ozaki1_plan.hpp の issue_candidates の fuse_ok が OZ1_INT32_PARTIAL を除く）．
    TT_FATAL(!((o.fuse || o.pairloop) && o.acc_fmt != tt::DataFormat::Float32),
             "spill_gemm_spec: 融合・対ループは FP32 の部分積でしか組めない（部分積の形式 {}）",
             (int)o.acc_fmt);
    TT_FATAL(o.acc_fmt == tt::DataFormat::Float32 || o.acc_fmt == tt::DataFormat::Int32,
             "spill_gemm_spec: 部分積の形式は Float32 か Int32（{}）", (int)o.acc_fmt);
    const uint32_t out_tile = tt::tile_size(o.acc_fmt);
    const uint32_t snt = g.sh * g.sw;   // 1 サブブロックのタイル数
    McastGemmSpec s;
    s.in_fmt = o.in_fmt;
    s.in_tile = tt::tile_size(o.in_fmt);
    // 融合・対ループの出力は Int32（既存の S_d や c_5 へ 2 の補数で足し込んだもの）．c_24 は
    // K ブロック間の部分和なので部分積の形式のまま．out_shared なら両者は同じ L1 領域を共有する．
    s.c16_fmt = (o.fuse || o.pairloop) ? tt::DataFormat::Int32 : o.acc_fmt;
    s.c24_fmt = o.acc_fmt;
    s.out_tile = out_tile;
    s.out_shared = o.out_shared;
    // 追加の CB はこの順に作る（CB の配置は作成順で決まる）
    namespace ga = gemm_args;
    if (o.fuse)   // CB_S（c_2）：ライタが先読みする既存の S（Int32）
        s.extra_cbs.push_back({(tt::CBIndex)ga::CB_S, tt::DataFormat::Int32, snt * 2, out_tile});
    if (o.fuse || o.pairloop)   // CB_TMP（c_3）：行列積の結果をいったん FP32 のまま逃がす先
        s.extra_cbs.push_back({(tt::CBIndex)ga::CB_TMP, tt::DataFormat::Float32, snt, out_tile});
    if (o.pairloop)   // CB_SL（c_5）：対をまたぐ Int32 累積面．輪をサブブロック 1 つぶん大きくする
        // （そうしないと書き込みの reserve が読み出し中の領域と重なる）
        s.extra_cbs.push_back({(tt::CBIndex)ga::CB_SL, tt::DataFormat::Int32, g.pm * g.pn + snt, out_tile});
    s.writer_fused = o.fuse;   // 計算側の FUSE とライタの融合は必ず揃える（ずれると c_2 で止まる）
    // 計算カーネルのコンパイル時引数：標準の形状（先頭 CA_STD_COUNT 個）に，FUSE・typecast の形式（Float32 → Int32）・
    // PAIRLOOP・BLOCKMM を足す（並びは gemm_compute_args.h の ComputeCT．bmm_int32_spill.cpp の冒頭）
    std::vector<uint32_t> cargs = mcast_compute_args(g, Kt);
    cargs.resize(ga::CA_COUNT);
    cargs[ga::CA_FUSE] = o.fuse ? 1u : 0u;
    cargs[ga::CA_F32] = (uint32_t)tt::DataFormat::Float32;
    cargs[ga::CA_I32] = (uint32_t)tt::DataFormat::Int32;
    cargs[ga::CA_PAIRLOOP] = o.pairloop ? 1u : 0u;
    cargs[ga::CA_BLOCKMM] = o.blockmm ? 1u : 0u;
    static_assert(ga::CA_COUNT == 17, "ComputeCT に欄を足したらここで詰める");
    // c_24（K ブロック間の部分和）と c_3（FP32 の中間）は切り詰めずに DST へ展開させる
    // （UnpackToDestFp32．無いと TF32 へ切り詰められ，部分和が無誤差でなくなる）．Int32 の
    // c_2（S）・c_5（累積面）・INT8 経路の c_24 にも同じ指定をしている（Int32 は指定が無くても 32 bit の
    // まま DST へ入る．gemm_int8.cpp の注記）
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[ga::CB_PARTIAL] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (o.fuse) u2d[ga::CB_S] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (o.fuse || o.pairloop) u2d[ga::CB_TMP] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (o.pairloop) u2d[ga::CB_SL] = tm::UnpackToDestMode::UnpackToDestFp32;
    s.compute = OZ_COMMON_KDIR "bmm_int32_spill.cpp";
    s.ccfg = tm::ComputeConfig{.math_fidelity = o.fidelity,
                               .fp32_dest_acc_en = true,
                               .unpack_to_dest_mode = u2d,
                               .compile_args = cargs};
    return s;
}
// ===== BF16 extended・3xTF32 の計算カーネル common/kernels/bmm_l1acc.cpp の仕様 =====
// gemm_bf16ext と gemm_exact_check（l1acc）が同じ CB・UnpackToDestFp32・コンパイル時引数で組むように，
// ここで作る．入力は BF16 か FP32（TF32 として乗算），出力・K ブロック間の部分和はどちらも FP32 で，
// c_16 と c_24 は別々の領域（パッカが c_24 上で K ブロックをまたいで加算する）．K ブロック数は
// コンパイル時引数 CA_NUM_BLOCKS に焼き込まれる（bmm_int32_spill.cpp と違い実行時に付け替えられない）．
static inline McastGemmSpec l1acc_gemm_spec(const OptConfig& g, uint32_t Kt, tt::DataFormat in_fmt,
                                            tt::tt_metal::MathFidelity fidelity) {
    namespace tm = tt::tt_metal;
    TT_FATAL(in_fmt == tt::DataFormat::Float16_b || in_fmt == tt::DataFormat::Float32,
             "l1acc_gemm_spec: 入力は Float16_b か Float32（{}）", (int)in_fmt);
    const tt::DataFormat out_fmt = tt::DataFormat::Float32;
    McastGemmSpec s;
    s.in_fmt = in_fmt;
    s.in_tile = tt::tile_size(in_fmt);
    s.c16_fmt = s.c24_fmt = out_fmt;
    s.out_tile = tt::tile_size(out_fmt);
    s.out_shared = false;
    s.compute = OZ_COMMON_KDIR "bmm_l1acc.cpp";
    // 最後のブロックで c_24 の部分和を読み戻すので FP32 のまま DST へ（無いと TF32 へ切り詰められる）
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[gemm_args::CB_PARTIAL] = tm::UnpackToDestMode::UnpackToDestFp32;
    s.ccfg = tm::ComputeConfig{.math_fidelity = fidelity,
                               .fp32_dest_acc_en = true,
                               .unpack_to_dest_mode = u2d,
                               .compile_args = mcast_compute_args(g, Kt)};
    return s;
}

// ===== 出力パネルの辺の選択（gemm_bf16・gemm_bf16ext）=====
// パネルの辺（タイル数）の候補は Mt・Nt の cap 以下の約数の組（dm, dn）すべてで，面積 dm*dn の大きい順
// （同じ面積なら dm，次に dn の大きい順）に cfg_for(dm, dn) を呼び，最初に ok を返した組を採る．
// 先頭の候補は「cap 以下の最大の約数」の組（largest_div_le(Mt,cap), largest_div_le(Nt,cap)）なので，
// その組で構成が取れる形（論文の測定サイズを含む）ではそれを選ぶ．最大の約数だけを見ると，
// その約数がコア格子に割れないとき，他の約数で組めても止まってしまう
// （例：M=5056 は Mt=158=2x79 で 79 を選び，縦 10 コア以内に割れない．dm=2 なら組める）．
// 約数 1 はコア格子の縦横 2 以上（choose_cfg）を満たせないので候補に入れない．
// 見つからなければ ok=false（pMt・pNt は 0）を返す．
struct PanelChoice { uint32_t pMt = 0, pNt = 0; OptConfig g; };
template <typename CfgFor>
static inline PanelChoice choose_panels(uint32_t Mt, uint32_t Nt, uint32_t cap, CfgFor&& cfg_for) {
    std::vector<std::pair<uint32_t, uint32_t>> cand;
    for (uint32_t dm = std::min(Mt, cap); dm >= 2; --dm) {
        if (Mt % dm) continue;
        for (uint32_t dn = std::min(Nt, cap); dn >= 2; --dn)
            if (Nt % dn == 0) cand.push_back({dm, dn});
    }
    // 生成順（dm の大きい順・dn の大きい順）を保ったまま面積の大きい順へ並べる
    std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
        return (uint64_t)a.first * a.second > (uint64_t)b.first * b.second; });
    for (const auto& c : cand) {
        const OptConfig g = cfg_for(c.first, c.second);
        if (g.ok) return {c.first, c.second, g};
    }
    return {};
}
// choose_panels の結果から作る出力のパネル分割：出力 Mt×Nt［タイル］を pMt×pNt のパネル nbr×nbc 枚に分ける．基準実装の
// gemm_bf16・gemm_bf16ext が構成探索の結果を実行と報告へ渡す形（パネルの辺と枚数を別々の引数で持ち回らない）．
// 構成が取れなければ ok() が偽で，枚数は 0（pMt=pNt=0 で割らない）
struct PanelGrid {
    OptConfig g;
    uint32_t pMt = 0, pNt = 0, nbr = 0, nbc = 0;
    bool ok() const { return g.ok; }
    uint32_t Mp() const { return pMt * 32u; }   // パネルの辺［要素］
    uint32_t Np() const { return pNt * 32u; }
};
static inline PanelGrid panel_grid(const PanelChoice& pc, uint32_t Mt, uint32_t Nt) {
    PanelGrid p;
    p.g = pc.g; p.pMt = pc.pMt; p.pNt = pc.pNt;
    if (pc.g.ok) { p.nbr = Mt / pc.pMt; p.nbc = Nt / pc.pNt; }
    return p;
}

// 発行ごとに変わる計算カーネルの実行時引数の検査．nb はこのプログラムの K ブロック数，
// bpp は対ループの 1 対あたりの K ブロック数．bpp=0 や nb が bpp の倍数でないと，最後の
// ブロックが対の境界にならず c_5 の和を出力へ反映しないまま終わる（止まらずに誤答になる）．
static inline void spill_check_run(uint32_t nb, bool pairloop, uint32_t bpp) {
    TT_FATAL(nb > 0, "bmm_int32_spill: K ブロック数が 0");
    if (pairloop)
        TT_FATAL(bpp > 0 && nb % bpp == 0,
                 "bmm_int32_spill: 対ループの 1 対あたりのブロック数 bpp={} が K ブロック数 {} を割り切らない",
                 bpp, nb);
}
// bmm_int32_spill.cpp の実行時引数（gemm_compute_args.h の ComputeRT）．融合も対ループも使わない構成は先頭の
// K ブロック数だけを渡す（カーネルは FUSE のときだけ CR_ACC を，PAIRLOOP のときだけ CR_BPP を読む）
static inline std::vector<uint32_t> spill_compute_rt(uint32_t nb) {
    std::vector<uint32_t> v(gemm_args::CR_NUM_BLOCKS + 1);
    v[gemm_args::CR_NUM_BLOCKS] = nb;
    return v;
}
static inline std::vector<uint32_t> spill_compute_rt(uint32_t nb, uint32_t acc, uint32_t bpp) {
    namespace ga = gemm_args;
    std::vector<uint32_t> v(ga::CR_COUNT);
    v[ga::CR_NUM_BLOCKS] = nb; v[ga::CR_ACC] = acc; v[ga::CR_BPP] = bpp;
    static_assert(ga::CR_COUNT == 3, "ComputeRT に欄を足したらここで詰める");
    return v;
}

// ---- build_mcast_gemm の部品（呼ぶ順は冒頭の tt-metalium の呼び出し順）----
// 組む前の検査：コア格子は縦横とも 2 以上（choose_cfg・choose_cfg_1b がそう選ぶ），形の前提（mcast_shape_check の注記），
// コア格子 x コアあたりの出力が行列に収まる，サブブロック sh*sw 枚が DST に載る（載らないと DST の別の区画を書き潰す），
// CB のページ（in_tile・out_tile）が形式のタイル長と同じ（違うとページの数え方とカーネルが読むタイルの大きさが食い違う．
// c_16 と c_24 は out_tile のページで作り，out_shared なら 1 つの領域を共有する），融合ライタには S の先読み先 c_2 と
// 計算側の CR_ACC がある（無いと計算側が c_2 を待ち続けて止まる．計算側の FUSE と揃えるのは spill_gemm_spec が受け持つ）
static inline void mcast_build_check(const OptConfig& g, uint32_t Mt, uint32_t Nt, uint32_t Kt, const McastGemmSpec& s) {
    TT_FATAL(g.cr >= 2 && g.cc >= 2, "build_mcast_gemm: コア格子 {}x{} は縦横とも 2 以上でなければならない", g.cr, g.cc);
    mcast_shape_check(g, Kt, "build_mcast_gemm");
    TT_FATAL(g.cr * g.pm <= Mt && g.cc * g.pn <= Nt,
             "build_mcast_gemm: コア格子 {}x{} x コアあたり {}x{} が行列 {}x{} タイルを超える", g.cr, g.cc, g.pm, g.pn, Mt, Nt);
    const uint32_t dst_cap = mcast_dst_tiles(s.ccfg);
    TT_FATAL(g.sh * g.sw <= dst_cap, "build_mcast_gemm: サブブロック {}x{} が DST の {} 枚に載らない", g.sh, g.sw, dst_cap);
    TT_FATAL(tt::tile_size(s.in_fmt) == s.in_tile, "build_mcast_gemm: in_tile={} が入力の形式のタイル長 {} と違う",
             s.in_tile, tt::tile_size(s.in_fmt));
    TT_FATAL(tt::tile_size(s.c16_fmt) == s.out_tile && tt::tile_size(s.c24_fmt) == s.out_tile,
             "build_mcast_gemm: out_tile={} が c_16・c_24 の形式のタイル長 {}・{} と違う", s.out_tile,
             tt::tile_size(s.c16_fmt), tt::tile_size(s.c24_fmt));
    if (s.writer_fused) {
        bool has_c2 = false;
        for (const auto& e : s.extra_cbs) has_c2 = has_c2 || e.idx == (tt::CBIndex)gemm_args::CB_S;
        TT_FATAL(has_c2, "build_mcast_gemm: 融合ライタが S を先読みする c_2 が無い");
        TT_FATAL(s.compute_rt.size() > gemm_args::CR_ACC,
                 "build_mcast_gemm: 融合ライタの構成の計算カーネルの実行時引数に CR_ACC が無い（{} 個．spill_compute_rt(nb, acc, bpp)）",
                 s.compute_rt.size());
    }
}
// CB（枚数は mcast_gemm_l1_bytes と同じ数え方．mcast_io_cb_tiles）：c_0・c_1（A・B．2 重緩衝），c_16・c_24（out_shared なら
// 1 つの領域），追加の CB（extra_cbs の順）
static inline void mcast_make_cbs(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                                  const OptConfig& g, const McastGemmSpec& s) {
    namespace tm = tt::tt_metal;
    namespace ga = gemm_args;
    const McastIoTiles io = mcast_io_cb_tiles(g.pm, g.pn, g.bw);
    const tt::CBIndex cb_in0 = (tt::CBIndex)ga::CB_IN0, cb_in1 = (tt::CBIndex)ga::CB_IN1,
                      cb_out = (tt::CBIndex)ga::CB_OUT, cb_part = (tt::CBIndex)ga::CB_PARTIAL;
    make_cb(program, all_cores, cb_in0, s.in_fmt, io.c0, s.in_tile);   // A（2 重緩衝）
    make_cb(program, all_cores, cb_in1, s.in_fmt, io.c1, s.in_tile);   // B（2 重緩衝）
    if (s.out_shared) {
        tm::CreateCircularBuffer(program, all_cores,
            tm::CircularBufferConfig(g.pm * g.pn * s.out_tile,
                                     {{cb_out, s.c16_fmt}, {cb_part, s.c24_fmt}})
                .set_page_size(cb_out, s.out_tile)
                .set_page_size(cb_part, s.out_tile));
    } else {
        make_cb(program, all_cores, cb_out, s.c16_fmt, g.pm * g.pn, s.out_tile);
        make_cb(program, all_cores, cb_part, s.c24_fmt, g.pm * g.pn, s.out_tile);
    }
    for (const auto& e : s.extra_cbs) make_cb(program, all_cores, e.idx, e.fmt, e.ntiles, e.page);
}
// データ移動カーネル（kernels/mcast_split_dm.cpp）の ID．a_* は A 側（RISCV_1），b_* は B 側（RISCV_0．ライタを兼ねる）．
// *_s0 は前半の送信（A は左端の列，B は上端の行），*_s1 は後半の送信（右端の列，下端の行），*_r はその間の受けるだけの
// コア（作らない場合は 0 のまま使わない）．a2・b2 は後半を送るか（1 ブロックが 2 タイル以上）
struct McastDmKernels {
    tt::tt_metal::KernelHandle a_s0 = 0, a_s1 = 0, a_r = 0, b_s0 = 0, b_s1 = 0, b_r = 0;
    bool a2 = false, b2 = false;
    // 列 cx（格子の横 cc）のコアの A 側，行 cy（縦 cr）のコアの B 側のカーネル
    tt::tt_metal::KernelHandle a_of(uint32_t cx, uint32_t cc) const { return (cx == 0) ? a_s0 : (a2 && cx == cc - 1) ? a_s1 : a_r; }
    tt::tt_metal::KernelHandle b_of(uint32_t cy, uint32_t cr) const { return (cy == 0) ? b_s0 : (b2 && cy == cr - 1) ? b_s1 : b_r; }
};
// データ移動カーネル（最大 6 本．この順に作る）と計算カーネルを作る．コンパイル時引数は A・B・C の TensorAccessorArgs を
// この順に連結したもの．探針（spec.probe）は define で渡す．計算カーネルの ID は ck に返す
static inline McastDmKernels mcast_make_kernels(
        tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores, const OptConfig& g,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Abuf,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Bbuf,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Cbuf,
        const McastGemmSpec& s, tt::tt_metal::KernelHandle& ck) {
    namespace tm = tt::tt_metal;
    using tm::CoreRange;
    std::vector<uint32_t> ct;
    tm::TensorAccessorArgs(*Abuf).append_to(ct);
    tm::TensorAccessorArgs(*Bbuf).append_to(ct);
    tm::TensorAccessorArgs(*Cbuf).append_to(ct);
    const std::string dm_kernel = OZ_COMMON_KDIR "mcast_split_dm.cpp";
    const std::string W = s.writer_fused ? "2" : "1";
    const GemmProbe& probe = s.probe;
    auto mk = [&](const CoreRange& rng, int operand, int role) {
        const bool isB = operand == 1;
        const std::map<std::string, std::string> defs = {
            {"OPERAND", std::to_string(operand)}, {"ROLE", std::to_string(role)},
            {"WRITER", isB ? W : std::string("0")},
            {"NODRAM", std::to_string(probe.nodram)}, {"NOMCAST", std::to_string(probe.nomcast)},
            {"NOWRITE", std::to_string(probe.nowrite)}};
        return tm::CreateKernel(program, dm_kernel, rng,
            tm::DataMovementConfig{.processor = isB ? tm::DataMovementProcessor::RISCV_0
                                                    : tm::DataMovementProcessor::RISCV_1,
                                   .noc = isB ? tm::NOC::NOC_0 : tm::NOC::NOC_1,
                                   .compile_args = ct, .defines = defs});
    };
    // 1 ブロックが 1 タイルしか無いオペランド（pm*bw==1 など）は後半が空なので，後半の送信コアを
    // 置かず，前半の送信コアが全部を配る（反対の端も受けるだけのカーネルにする．mcast_split_dm.cpp の
    // half2 と同じ条件）．小さい行列ではカーネルの本数が起動の時間を左右する．
    McastDmKernels k;
    k.a2 = g.pm * g.bw >= 2;
    k.b2 = g.pn * g.bw >= 2;
    const std::size_t xl = g.cc - 1, yl = g.cr - 1;   // 右端の列・下端の行
    k.a_s0 = mk(CoreRange({0, 0}, {0, yl}), 0, 1);                                  // A：左端の列が前半を配る
    if (k.a2) k.a_s1 = mk(CoreRange({xl, 0}, {xl, yl}), 0, 3);                      //    右端の列が後半を配る
    if (g.cc > 2 || !k.a2) k.a_r = mk(CoreRange({1, 0}, {k.a2 ? xl - 1 : xl, yl}), 0, 2);
    k.b_s0 = mk(CoreRange({0, 0}, {xl, 0}), 1, 1);                                  // B：上端の行が前半を配る
    if (k.b2) k.b_s1 = mk(CoreRange({0, yl}, {xl, yl}), 1, 3);                      //    下端の行が後半を配る
    if (g.cr > 2 || !k.b2) k.b_r = mk(CoreRange({0, 1}, {xl, k.b2 ? yl - 1 : yl}), 1, 2);
    ck = tm::CreateKernel(program, s.compute, all_cores, s.ccfg);
    return k;
}
// マルチキャストの送受信を同期するセマフォ．A 用と B 用に，送信側の空き待ち（free）・前半のフラグ（flag）・
// 後半のフラグ（flag2）の 3 つずつ．ID は作成順で決まるので，この順を変えないこと
// （mcast_reader_args の {A 空き, A フラグ, B 空き, B フラグ} は tt-metal のリーダの in0/in1 の並び）．
struct McastSemIds { uint32_t A_free, A_flag, B_free, B_flag, A_flag2, B_flag2; };
static inline McastSemIds mcast_make_sems(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores) {
    namespace tm = tt::tt_metal;
    McastSemIds m{};
    m.A_free  = tm::CreateSemaphore(program, all_cores, INVALID);
    m.A_flag  = tm::CreateSemaphore(program, all_cores, INVALID);
    m.B_free  = tm::CreateSemaphore(program, all_cores, INVALID);
    m.B_flag  = tm::CreateSemaphore(program, all_cores, INVALID);
    m.A_flag2 = tm::CreateSemaphore(program, all_cores, INVALID);
    m.B_flag2 = tm::CreateSemaphore(program, all_cores, INVALID);
    return m;
}
// コアごとの実行時引数：担当する A/B の開始タイルとマルチキャストの相手先（並びは cy が外側・cx が内側）．
// 融合ライタの acc は計算カーネルの CR_ACC と同じ値（McastGemmSpec の compute_rt の注記）
static inline void mcast_set_core_args(
        tt::tt_metal::Program& program, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, McastGemm& r,
        const OptConfig& g, uint32_t Nt, uint32_t Kt, uint32_t Aaddr, uint32_t Baddr, uint32_t Caddr,
        const McastDmKernels& k, const McastSemIds& sem, const McastGemmSpec& s) {
    namespace tm = tt::tt_metal;
    using tm::CoreCoord;
    const uint32_t pm = g.pm, pn = g.pn, sh = g.sh, sw = g.sw, bw = g.bw, cr = g.cr, cc = g.cc;
    const uint32_t wacc = s.writer_fused ? s.compute_rt[gemm_args::CR_ACC] : 0u;
    for (uint32_t cy = 0; cy < cr; cy++)
        for (uint32_t cx = 0; cx < cc; cx++) {
            CoreCoord core = {(std::size_t)cx, (std::size_t)cy};
            // 相手先：行の左端・その右隣・右端，列の上端・その下隣・下端
            auto Lp  = md->worker_core_from_logical_core(CoreCoord{0, (std::size_t)cy});
            auto Lp1 = md->worker_core_from_logical_core(CoreCoord{1, (std::size_t)cy});
            auto Rp  = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cc - 1, (std::size_t)cy});
            auto Tp  = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, 0});
            auto Tp1 = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, 1});
            auto Bp  = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, (std::size_t)cr - 1});
            // 前半の送信先の矩形は行の列 1..cc-1（列の行 1..cr-1）で，tt-metal のリーダの並びのまま渡す
            auto ra = mcast_reader_args(Aaddr, Baddr, Nt, Kt, pm, pn, bw, cr, cc, cx, cy,
                                        {Lp, Lp1, Rp, Tp, Tp1, Bp},
                                        {sem.A_free, sem.A_flag, sem.B_free, sem.B_flag}, s.dA, s.dB);
            auto wa = mcast_writer_args(Caddr, Nt, pm, pn, sh, sw, cx, cy, s.dW, wacc);
            // 後半の送信用（kernels/mcast_args.h の X_*）：後半の送信コア（A は行の右端，B は列の下端）の
            // 座標と，その送信先の矩形（A は列 0..cc-2，B は行 0..cr-2．hi 隅＝座標の大きい側を先に置く），
            // 後半のフラグ．
            auto A2 = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cc - 2, (std::size_t)cy});
            auto B2 = md->worker_core_from_logical_core(CoreCoord{(std::size_t)cx, (std::size_t)cr - 2});
            // 位置は表の名前で詰める（並べ替えても値が入れ替わらない．mcast_args.h）
            auto second = [](const CoreCoord& s1, const CoreCoord& hi, const CoreCoord& lo, uint32_t fsem) {
                namespace ma = mcast_args;
                std::vector<uint32_t> x(ma::X_COUNT, 0xFFFFFFFFu);
                x[ma::X_S1_X - ma::X_S1_X] = (uint32_t)s1.x;  x[ma::X_S1_Y - ma::X_S1_X] = (uint32_t)s1.y;
                x[ma::X_HI_X - ma::X_S1_X] = (uint32_t)hi.x;  x[ma::X_HI_Y - ma::X_S1_X] = (uint32_t)hi.y;
                x[ma::X_LO_X - ma::X_S1_X] = (uint32_t)lo.x;  x[ma::X_LO_Y - ma::X_S1_X] = (uint32_t)lo.y;
                x[ma::X_FLAG_SEM - ma::X_S1_X] = fsem;
                // 詰め忘れは番兵のまま残るので止める（mcast_args.h の冒頭．ozaki_gemm_host.hpp の mcast_args_filled）
                mcast_args_filled(x, "build_mcast_gemm（後半の送信）");
                return x;
            };
            McastGemm::Core c{core, k.a_of(cx, cc), k.b_of(cy, cr), std::move(ra), second(Rp, A2, Lp, sem.A_flag2),
                              second(Bp, B2, Tp, sem.B_flag2)};
            r.set_args(program, c, c.ra, wa);
            if (!s.compute_rt.empty()) tm::SetRuntimeArgs(program, r.ck, core, s.compute_rt);
            r.cores.push_back(std::move(c));
        }
}

// program に CB・カーネル・セマフォ・コアごとの実行時引数を積む（冒頭の呼び出し順で上の部品を呼ぶ）．
// Mt・Nt・Kt はこのプログラムが見る行列のタイル数（リーダ／ライタのストライドになる）．
// A/B/C のバッファは呼び出し側が作って渡す（アドレスと TensorAccessorArgs に使う）．
// 組む構成はすべて引数と spec から決まる（環境変数を読まない．探針も spec.probe）．
static inline McastGemm build_mcast_gemm(
        tt::tt_metal::Program& program,
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
        const OptConfig& g, uint32_t Mt, uint32_t Nt, uint32_t Kt,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Abuf,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Bbuf,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Cbuf,
        const McastGemmSpec& s) {
    mcast_build_check(g, Mt, Nt, Kt, s);
    const tt::tt_metal::CoreRange all_cores({0, 0}, {(std::size_t)g.cc - 1, (std::size_t)g.cr - 1});
    mcast_make_cbs(program, all_cores, g, s);
    McastGemm r;
    const McastDmKernels k = mcast_make_kernels(program, all_cores, g, Abuf, Bbuf, Cbuf, s, r.ck);
    const McastSemIds sem = mcast_make_sems(program, all_cores);
    mcast_set_core_args(program, md, r, g, Nt, Kt, (uint32_t)Abuf->address(), (uint32_t)Bbuf->address(),
                        (uint32_t)Cbuf->address(), k, sem, s);
    return r;
}

// ===== 基準実装の行列積 1 本（gemm_bf16 のパネル 1 枚・gemm_bf16ext の gemm_fp32out・gemm_int8 の gemm_int8）=====
// 実際に作る CB が L1 の予算に収まることを確かめ（mcast_check_l1．who は止めたときの文に出す名前），A（Mt×Kt タイル）・
// B（Kt×Nt）・C（Mt×Nt）の DRAM の器をこの順に作り（タイル 1 枚 = 1 ページ．mesh_dram_buffer），spec で
// プログラムを組み，A・B（タイル順のホストの配列）を送って 1 回流す（カーネルの JIT と初回のアクセスを計時から外す）．
// e2e なら，ホストからの転送と読み戻しを含む端から端までの時間を参考値として 1 回測る（gemm_bf16）．そのうえで段「gemm」の
// 温めと計時（oz_timing.hpp の time_dispatch．計時の区間は iters 回の投入と 1 回の同期だけ）を行い，C をタイル順のまま
// 読み戻す．ホストの配列の要素の型はタイルの形式と同じ大きさであること（TIn は in_tile，TOut は out_tile の 1024 分の 1）．
// 部分積の付け替えなどで同じプログラムを何度も流す尾崎の両スキームと gemm_exact_check は使わない
struct McastGemmRun { double timed_s = 0.0, e2e_s = 0.0; };   // 1 反復あたりの秒数（計時）と端から端までの秒数（e2e のとき）
template <typename TIn, typename TOut>
static inline McastGemmRun run_mcast_gemm(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const OptConfig& g,
                                          const McastGemmSpec& s, uint32_t Mt, uint32_t Nt, uint32_t Kt,
                                          const std::vector<TIn>& A, const std::vector<TIn>& B, std::vector<TOut>& C,
                                          uint32_t warmup, uint32_t iters, bool e2e = false,
                                          const char* who = "run_mcast_gemm") {
    namespace dist = tt::tt_metal::distributed;
    TT_FATAL(s.in_tile == 1024u * sizeof(TIn) && s.out_tile == 1024u * sizeof(TOut),
             "run_mcast_gemm: ホストの配列の要素（{}・{} バイト）がタイル（in_tile={}・out_tile={}）の形式と合わない",
             sizeof(TIn), sizeof(TOut), s.in_tile, s.out_tile);
    mcast_check_l1(g, s, who);
    auto& cq = md->mesh_command_queue();
    dist::MeshWorkload workload;
    const auto range = dist::MeshCoordinateRange(md->shape());
    tt::tt_metal::Program program{};
    auto Abuf = mesh_dram_buffer(md, (size_t)Mt * Kt, s.in_tile);
    auto Bbuf = mesh_dram_buffer(md, (size_t)Kt * Nt, s.in_tile);
    auto Cbuf = mesh_dram_buffer(md, (size_t)Mt * Nt, s.out_tile);
    build_mcast_gemm(program, md, g, Mt, Nt, Kt, Abuf, Bbuf, Cbuf, s);
    workload.add_program(range, std::move(program));

    dist::EnqueueWriteMeshBuffer(cq, Abuf, A, false);
    dist::EnqueueWriteMeshBuffer(cq, Bbuf, B, false);
    dist::EnqueueMeshWorkload(cq, workload, false);
    dist::Finish(cq);   // 1 回目は計時しない（ここでカーネルの JIT ビルドと初回アクセスが起きる）

    McastGemmRun r;
    if (e2e) {
        const auto e0 = std::chrono::steady_clock::now();
        dist::EnqueueWriteMeshBuffer(cq, Abuf, A, false);
        dist::EnqueueWriteMeshBuffer(cq, Bbuf, B, false);
        dist::EnqueueMeshWorkload(cq, workload, false);
        dist::EnqueueReadMeshBuffer(cq, C, Cbuf, true);
        r.e2e_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
    }
    // パネル・部分積・K の塊ごとの計時はどれも段「gemm」．間にホストの切り出しと転送が入るので，steady ではそのたびに
    // 温め直す（oz_timing.hpp の計時の 3）
    r.timed_s = oz_eval::time_dispatch("gemm", warmup, iters,
        [&]{ dist::EnqueueMeshWorkload(cq, workload, false); },
        [&]{ dist::Finish(cq); });
    dist::EnqueueReadMeshBuffer(cq, C, Cbuf, true);
    return r;
}
