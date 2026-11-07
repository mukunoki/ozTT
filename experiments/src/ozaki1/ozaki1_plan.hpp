#pragma once
// 尾崎スキーム I：構成の計画（発行と累積のしかた，K ブロック長，出力パネルと行バンド，再構成の定数）
// ---------------------------------------------------------------------------
// どれもデバイスに触らず，何も確保しない（DRAM の空き容量とコア格子の大きさだけを引数で受ける）．計画は引数
// （RunCfg と Oz1Knobs）だけから決まり，環境変数を読まない（つまみは main の冒頭で oz1_knobs_from_env が 1 回だけ読む．
// 例外は表示だけの OZ_DBG で，DRAM の見積りの内訳を stderr へ出すかどうかしか変えない）．段の組み方を変える診断と
// 測定モードのつまみ（Oz1Diag）も同じく main の冒頭で 1 回だけ読む．
// 依存：ozaki1_format.hpp（無誤差条件），ozaki1_gemm.hpp（形状探索 choose_cfg_1b・choose_remainder_cfg・GemmMode），
// ozaki1_recon.hpp（ReconScale），common/oz_bits.h（f32_bits・oz_split_scale_ld），DRAM の見積りの余裕
// common/oz_dram_margin.h（両スキーム共通）と FP64 のタイルの大きさ common/kernels/f64_tf.h．

#include "ozaki1/ozaki1_format.hpp"
#include "ozaki1/ozaki1_gemm.hpp"
#include "ozaki1/ozaki1_recon.hpp"
#include "common/oz_bits.h"
#include "common/oz_dram_margin.h"
#include "common/kernels/f64_tf.h"   // OZ_F64_TILE_BYTES
#include <optional>
#include <string>
#include <vector>

// 位置引数と，基数から決まる正規化の設定（ozaki1_main.cpp の parse_run_config が作る）．p・L は OZ_FP64 が選び直すことがある．
struct RunCfg {
    uint32_t M = 0, N = 0, K = 0, Mt = 0, Nt = 0;
    int p = 0, L = 0;
    bool auto_p = false;       // OZ_FP64：p を入力から選び直す（prepare_operands．CONFIG の auto_p）
    uint32_t warmup = 0, iters = 0;
    int radix = 0;
    uint32_t dmax = 0;         // 桁の上限（ozaki_digit_max(radix)．OZ_DMAX が上書きしうる）
    tt::tt_metal::MathFidelity fid = tt::tt_metal::MathFidelity::LoFi;   // 行列エンジンのパス数（oz1_fidelity(dmax)）
    int32_t nshift = 0;        // 正規化の目標 2^nshift
    bool bf_ceil = false;      // max|x| の BF16 化を切り上げるか
    bool tau_above = false;    // スケールを最大値より真に大きい 2 のべきに取るか（BFP2_b．parse_run_config）
    // スライスの本数 L+1．打ち切り対角 L のもとで使う対は i+j <= L なので，要るスライスは 0..L の L+1 本だけで，
    // 対角の本数 ndiag もこれに等しい（どちらの名前もここから取る）
    uint32_t nslice() const { return (uint32_t)(L + 1); }
    uint32_t ceil_mode() const { return tau_above ? 2u : bf_ceil ? 1u : 0u; }   // scale_build へ渡す CEIL
};

// ===== アブレーションのつまみ =====
// 発行・累積・GEMM の組み方を変えるつまみ（docs/env_vars.md の B と，計画を変える C の OZ_NOBLOCK）．main の冒頭で 1 回だけ
// 読んで，計画（plan_issue・plan_panels・recon_scale）と GEMM の構築へ引数で配る（使う所で読み直さない）．どれも数学的には
// 同じ積を計算し，精度の基準（53 bit）を保つ．ビット単位で同じ C になるかはつまみと構成による（docs/env_vars.md の B）．
// 既定でないものは CONFIG 行の knobs 欄に残る（oz_knobs.hpp の表の R_KNOBS．OZ_NOBLOCK は diag 欄）．
//   OZ_NO_DEFER     遅延再構成（K ブロックごとの triple-float の Horner へ戻す）
//   OZ_NO_PAIRLOOP  対を 1 プログラムの中で回す構成（本数の少ないほうへ落ちる：対ごとの発行と融合，連結の K ブロックが
//                   複数なら連結の融合，1 本なら累積を持たない連結．plan_issue）
//   OZ_NO_FUSE_ACC  Int32 累積の GEMM の出力経路への取り込み（別プログラムの累積へ戻す．ozaki1_accum.hpp．
//                   PAIRLOOP も同じ土台に乗るので一緒に切れる）
//   OZ_NO_BLOCKMM   GEMM の計算カーネルの matmul_block（1 タイルずつの matmul_tiles へ戻す．発行が h*w*bw 回から bw 回に
//                   なる．GEMM 段の時間を何が決めているかは docs/gemm_stage_bottleneck.md）
//   OZ_NSMALL=n     遅延再構成で型変換 1 回で読む先頭の対角を n 本に減らす（上界は recon_nsmall．recon_scale が検査する）
//   OZ_BANDS=n      出力を切る行バンドの本数（既定は DRAM の空きから決める．plan_band_panel_rows）
//   OZ_L1_BUDGET_KB 形状探索の L1 の予算（choose_cfg_1b・choose_remainder_cfg の oz1_max_bw と，gemm_build の照合が使う．
//                   読み方は GEMM 段を組むほかのドライバと同じ oz_l1_budget．CONFIG 行の l1_budget_kb 欄にも残る）
//   OZ_NOBLOCK      【診断】K ブロック分割を止める（無誤差条件を意図的に破る．ozaki_block_k）
// 構文と範囲はデバイスを開く前に oz_drv::check_env（つまみの表の preflight）が検査してある．
struct Oz1Knobs {
    bool fuse_acc = true;              // OZ_NO_FUSE_ACC の否定
    bool pairloop = true;              // OZ_NO_PAIRLOOP の否定
    bool defer_recon = true;           // OZ_NO_DEFER の否定
    bool blockmm = true;               // OZ_NO_BLOCKMM の否定
    bool noblock = false;              // OZ_NOBLOCK
    std::optional<uint32_t> nsmall;    // OZ_NSMALL（未設定なら上界まで）
    std::optional<uint32_t> bands;     // OZ_BANDS（未設定なら DRAM の空きから）
    uint32_t l1_budget = 0;            // OZ_L1_BUDGET_KB の予算［B］（未設定なら表の既定）
};
static Oz1Knobs oz1_knobs_from_env() {
    Oz1Knobs k;
    k.fuse_acc = !oz_knobs::set(OZK("OZ_NO_FUSE_ACC"));
    k.pairloop = !oz_knobs::set(OZK("OZ_NO_PAIRLOOP"));
    k.defer_recon = !oz_knobs::set(OZK("OZ_NO_DEFER"));
    k.blockmm = gemm_blockmm();   // 読み方は GEMM 段を組むほかのドライバと同じ（ozaki_gemm_host.hpp）
    k.noblock = oz_knobs::set(OZK("OZ_NOBLOCK"));
    k.nsmall = oz_knobs::u32_opt(OZK("OZ_NSMALL"));
    k.bands = oz_knobs::u32_opt(OZK("OZ_BANDS"));
    k.l1_budget = oz_l1_budget();   // 読み方は GEMM 段を組むほかのドライバと同じ（ozaki_gemm_host.hpp）
    return k;
}

// ===== 診断と測定モードのつまみ =====
// 計画は変えないが，段の組み方（カーネルの選び方・コンパイル時引数）か計時の流れを変えるつまみ．Oz1Knobs と同じく main の
// 冒頭で 1 回だけ読み，段の構築（スライス・再構成・GEMM）と報告へ引数で配る（使う所で読み直さない）．OZ_TIME_WHOLE のほかは
// 論文の表・図の測定には使わない（OZ_SOFTFLOAT は knobs 欄，探針は diag 欄に残り probe=set になる．oz_knobs.hpp の表）．
//   OZ_SOFTFLOAT     FP64 <-> triple-float の変換を C の double 式（soft-float）で行う測定モード（スライスのリーダ・スライサと
//                    再構成のライタ．ozaki1_format.hpp の oz_softfloat が読む）
//   OZ_RECON_PROBE   【診断】再構成のライタの仕事を段階的に外す（答えは壊れる．ozaki1_recon.hpp の冒頭）
//   OZ_GEMM_PROBE_*  【診断】GEMM 段のデータ移動を外す探針（答えは壊れる．common/gemm_probe.hpp）
//   OZ_TIME_WHOLE    段ごとの計時の後に全工程を続けて流す計時を足す（ozaki1_whole.hpp．論文の全体の時間．2026-11-03 から
//                    診断ではなく，CONFIG の time_whole・whole_ms に残し diag には数えない）
// 表示だけの OZ_DBG はここに集めない（ozaki1_format.hpp の oz_dbg_on）．
struct Oz1Diag {
    bool softfloat = false;            // OZ_SOFTFLOAT
    uint32_t recon_probe = 0;          // OZ_RECON_PROBE（0..3．範囲外と不正な文字列は読み取りが止める）
    GemmProbe gemm_probe;              // OZ_GEMM_PROBE_*（GemmBuildSpec::probe へ詰める）
    bool time_whole = false;           // OZ_TIME_WHOLE（=0 は既定と同じ）
};
static Oz1Diag oz1_diag_from_env() {
    Oz1Diag d;
    d.softfloat = oz_softfloat() != 0u;
    d.recon_probe = oz_knobs::u32(OZK("OZ_RECON_PROBE"));
    d.gemm_probe = oz_gemm_probe();
    d.time_whole = oz_knobs::u32(OZK("OZ_TIME_WHOLE")) != 0u;
    return d;
}

// スライス対の数と対角の数．OZ_FP64 が p を選び直したら数え直す．
// 対角 d（0..L）には対 (i, d-i) が d+1 個あり，L < p（parse_run_config が検査済み）なので
// どの i も 0..p-1 に収まる．したがって対の数は (L+1)(L+2)/2 で，p^2 回の走査は要らない
// （幅の広い整数で求め，int に収まることを確かめる．p <= 4096 なら高々約 840 万）．
struct PairCount { int npairs = 0, ndiag = 0; };
static PairCount count_pairs(int p, int L) {
    TT_FATAL(L >= 0 && L < p, "count_pairs: L={} は 0..p-1（p={}）の外", L, p);
    PairCount pc;
    const int64_t np = (int64_t)(L + 1) * (int64_t)(L + 2) / 2;
    TT_FATAL(np <= INT32_MAX, "count_pairs: 対の数 {} が int に収まらない", np);
    pc.npairs = (int)np;
    pc.ndiag = L + 1;   // = RunCfg::nslice()
    return pc;
}

// ===== DRAM の見積り =====
// これから確保するものの大きさ［B］（何も確保していない時点の空きと比べる）．発行の計画（累積面が載るか）と行バンドの
// 本数の決定が同じ内訳を使う．数えないのは対角の部分和 Pbig と累積面 S（使う側が面の数を決めて足す）と，小さな器
// （範囲外の旗・limb を読まない構成の 1 タイルの器）で，後者は両スキーム共通の余裕 OZ_DRAM_MARGIN_BYTES
// （common/oz_dram_margin.h）が受ける．
struct Oz1DramNeed {
    uint64_t operands = 0;   // FP64 のオペランド A・B（8 B/要素）
    uint64_t slices = 0;     // スライス連結の A・B（L+1 本．入力形式のタイル長）
    uint64_t scale = 0;      // 指数差分（Int32 タイル）と行・列ごとの最大値（BF16 タイル）
    uint64_t output = 0;     // FP64 の出力
    uint64_t limbs = 0;      // 再構成の limb 3 面（K ブロックごとに Horner を回す構成だけ．バンド 1 本ぶんで足りるが出力全体で見る）
    uint64_t total() const { return operands + slices + scale + output + limbs; }
};
static Oz1DramNeed oz1_dram_need(const RunCfg& c, bool recon_limbs) {
    const uint64_t f32t = tt::tile_size(tt::DataFormat::Float32);
    const uint64_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    const uint64_t in_t = tt::tile_size(OZ_FMT.in);
    const uint64_t f64t = OZ_F64_TILE_BYTES;   // FP64 のタイル 1 枚（setup_inputs・recon_build が確保するページと同じ）
    const uint64_t Mt = c.Mt, Nt = c.Nt, Kto = c.K / 32, Ktm = (uint64_t)c.nslice() * Kto;
    Oz1DramNeed d;
    d.operands = f64t * Mt * Kto + f64t * Kto * Nt;
    d.slices = in_t * Mt * Ktm + in_t * Nt * Ktm;
    d.scale = f32t * (Mt + Nt) + bf16t * (Mt + Nt);
    d.output = f64t * Mt * Nt;
    d.limbs = recon_limbs ? 3 * f32t * Mt * Nt : 0u;
    return d;
}
// OZ_DBG の表示（what は見積りの用途，extra はそれに足す面の内訳）
static void oz1_dram_dbg(const char* what, size_t free_b, const Oz1DramNeed& d, const char* extra_name, uint64_t extra) {
    if (!oz_dbg_on()) return;
    fmt::print(stderr, "DBG dram({}): free={:.2f} GB need={:.2f} GB (operands {:.2f} + slices {:.2f} + scale {:.2f} + "
                       "output {:.2f} + limbs {:.2f} + {} {:.2f} + margin {:.2f})\n",
               what, free_b / 1e9, (d.total() + extra + OZ_DRAM_MARGIN_BYTES) / 1e9, d.operands / 1e9, d.slices / 1e9,
               d.scale / 1e9, d.output / 1e9, d.limbs / 1e9, extra_name, extra / 1e9, OZ_DRAM_MARGIN_BYTES / 1e9);
}

// ===== 発行と累積のしかた =====
struct IssuePlan {
    bool ok = false;            // false なら理由を表示済みで，実行を打ち切る
    bool pairloop = false;      // 対角の (d+1) 対を 1 プログラムの中で回す
    bool pairwise = false;      // 対ごとに 1 本ずつ発行する
    bool fuse = false;          // GEMM の出力経路で DRAM の S_d を読んで足す
    bool scratch = false;       // 累積を別プログラムで行い，Pbig はスクラッチ 1 面
    bool defer = false;         // Int32 で累積して最後に 1 回だけ Horner（初期値）
    uint32_t Kc = 0, nkb = 0;   // K ブロック長と K ブロック数
    // 1 本の GEMM が負う K_eff が 1 対ぶんか（対ごとの発行と PAIRLOOP）．連結だけが (L+1) 倍を負う
    bool per_pair() const { return pairwise || pairloop; }
    // 累積を別プログラム（ozaki1_accum.hpp）で行うか（対ごとかスクラッチで，GEMM の中で累積しない構成）
    bool sep_accum() const { return (pairwise || scratch) && !fuse && !pairloop; }
    // 再構成が limb 3 面（hi・mid・lo）を DRAM に置くか．K ブロックごとに Horner を回す構成（遅延再構成を
    // 使わず K ブロックが複数．OZ_NO_DEFER など）だけが，途中の K ブロックの結果を limb として次へ渡す
    // （recon_build の with_acc && !defer．遅延再構成でない構成は対ごとにも PAIRLOOP でも回さないので，
    // run_sweep に来るのは K ブロックが複数のときだけである）
    bool recon_limbs() const { return nkb > 1 && !defer; }
    // K ブロックが 1 つで，対ごとにも PAIRLOOP でも回さない構成（ozaki1_main.cpp の run_single_kblock．INT8 と一部の
    // アブレーション）．それ以外は run_sweep
    bool single_kblock() const { return nkb == 1 && !pairwise && !pairloop; }
    // パネル 1 枚・K ブロック 1 本あたりの GEMM の発行の本数：対ごとの発行なら対の数 ndiag(ndiag+1)/2，そうでなければ
    // 対角の数 ndiag（PAIRLOOP は対を 1 本の中で回すので ndiag のまま）．Issue 行・Compute 行・CONFIG の gemm_issues と
    // OZ_TIME_WHOLE の本数の期待値（ozaki1_whole.hpp）が同じこの式を使う
    uint32_t gemm_issues(uint32_t ndiag) const { return pairwise ? ndiag * (ndiag + 1) / 2 : ndiag; }
};

// 発行の 2 つの候補（連結・対ごと）の K ブロック長と本数，累積の成立条件（plan_issue の前半）
struct IssueCandidates {
    uint32_t kc_cat = 0, kc_pw = 0;     // 無誤差な K ブロック長（0 なら取れない）
    uint32_t nkb_cat = 0, nkb_pw = 0;   // その K ブロック数
    uint32_t n_cat = 0, n_pw = 0;       // 本数（発行回数＝累積パスの面更新回数）
    bool defer_ok = false;              // Int32 の累積が溢れない（oz_defer_fits）
    bool fuse_ok = false;               // 出力経路への Int32 累積の融合が使える
    bool acc_fits_pw = false, acc_fits_scr = false;   // 別に確保する累積面が DRAM に載るか（対ごと・スクラッチ）
    uint64_t acc_need = 0;              // その累積面の大きさ［B］
    uint32_t ndiag = 0;                 // 対角の本数（= スライスの本数 L+1）
};
// 無誤差条件（尾崎スキームの設計条件）を満たす K ブロック長を，連結と対ごとの 2 通りについて決める．
// K 全体で条件を満たすならブロック分割は起きない．対ごとに回すなら 1 本の K_eff は K_c のままなので，無誤差条件に
// (L+1) 倍が入らない．連結して回すなら (L+1)K_c が予算に収まるまで K を分割する．
// 対ごとに回すには Int32 の累積面（ndiag 面）を Pbig とは別に持つ必要がある．入らないと途中で詰むので，まだ何も確保して
// いないこの時点の空き容量から先に判定しておく（例えば 14080^3 の BFP2_b は累積面だけで p 面（8197a0b の routes.tsv の
// p=59 なら 46.8 GB）あり入らない）．
static IssueCandidates issue_candidates(const RunCfg& c, size_t dram_free, const Oz1Knobs& kn) {
    IssueCandidates ic;
    const uint32_t K = c.K;
    const int L = c.L;
    const uint32_t ndiag = c.nslice();
    ic.ndiag = ndiag;
    ic.acc_need = (uint64_t)ndiag * c.Mt * c.Nt * tt::tile_size(tt::DataFormat::Int32);
    // 対ごとに回すなら Pbig は ndiag 面．連結して回すならスクラッチ 1 面で済むので，同じ累積面を足しても容量に収まり
    // やすい（Pbig の差は (ndiag-1) 面．14080^3 の 1 面は 0.79 GB）．どちらの構成も遅延再構成なので limb を置かない
    const Oz1DramNeed need = oz1_dram_need(c, false);
    const uint64_t plane = (uint64_t)c.M * c.N * tt::tile_size(tt::DataFormat::Float32) / 1024u;   // Pbig の 1 面（FP32）
    oz1_dram_dbg("accum pairwise", dram_free, need, "Pbig+acc", (uint64_t)ndiag * plane + ic.acc_need);
    oz1_dram_dbg("accum scratch", dram_free, need, "Pbig+acc", plane + ic.acc_need);
    ic.acc_fits_pw  = (uint64_t)dram_free >= need.total() + (uint64_t)ndiag * plane + ic.acc_need + OZ_DRAM_MARGIN_BYTES;
    ic.acc_fits_scr = (uint64_t)dram_free >= need.total() + plane + ic.acc_need + OZ_DRAM_MARGIN_BYTES;
    // 対ごとに回すと 1 本の K は伸びるが本数が n_diag から n_diag(n_diag+1)/2 へ増える．
    // 得になるのは，連結側の K ブロック化がそれを上回って本数を増やすときだけである．
    // 本数は発行回数であると同時に累積パスの面更新回数でもあるので，これで測る．
    // Kc は 0（無誤差なブロック長が取れない）を返しうるので，割る前に必ず検査する．
    ic.kc_cat = ozaki_block_k(c.dmax, K, L, kn.noblock);
    ic.kc_pw  = ozaki_block_k(c.dmax, K, 0, kn.noblock);
    ic.nkb_cat = ic.kc_cat ? K / ic.kc_cat : 0u;
    ic.nkb_pw  = ic.kc_pw  ? K / ic.kc_pw  : 0u;
    // 連結で K ブロック長が取れない（kc_cat==0）ときは，連結の本数を無限大とみなす．
    // 0 のままだと n_pw < n_cat が常に偽になり，使えるはずの対ごとの発行まで見送って
    // 下の Kc==0 で止まってしまう．
    ic.n_cat = ic.kc_cat ? ndiag * ic.nkb_cat : UINT32_MAX;
    ic.n_pw  = ic.kc_pw  ? ndiag * (ndiag + 1) / 2 * ic.nkb_pw : 0u;
    // 対ごとの発行は対角 ℓ あたり (ℓ+1) 本を Int32 で足し込むので，連結のときと同じ
    // 上限検査 oz_defer_fits が要る（総和は nkb*Kc=K より (L+1)*K*d^2 で，d は桁の
    // 上限．発行の仕方に依らない）．ここを抜かすと，1 本ごとの無誤差条件は満たすのに
    // 累積が Int32 を溢れる構成を黙って選んでしまう．
    ic.defer_ok = oz_defer_fits(c.dmax, K, L);
    // 出力経路への Int32 累積の融合．GEMM が既存の S_d を読んで足して書くので，
    // 累積面を Pbig とは別に持つ必要がなくなる（Pbig そのものが S になる）．
    // 1 更新あたりの DRAM が 4 面から 2 面へ減り，累積プログラムも消える．
    ic.fuse_ok = kn.fuse_acc && !OZ1_INT32_PARTIAL && kn.defer_recon && ic.defer_ok;
    return ic;
}

// 発行のしかたは 2 通り．対角の (d+1) 対を 1 プログラムの中で回す（PAIRLOOP）と，
// 対ごとに 1 本ずつ発行する（pairwise）である．前者は対の境界で FP32
// アキュムレータを畳むので，無誤差条件が 1 対ぶん（K_eff = K_c）で済んで
// K ブロック化が消えるうえ，発行本数も n_diag のままで，対と対のあいだの
// 部分和が DRAM を経由しないので，後者より常に良い．K ブロックが 1 本に
// 収まるときだけ採れる（複数だと対の境界と K ブロックの境界が絡む）．
// 対ごとの発行を見送ったら（INT8 経路を除いて）理由を Issue 行に出す．
static void choose_issue(IssuePlan& ip, const IssueCandidates& ic, const Oz1Knobs& kn) {
    ip.pairloop = ic.fuse_ok && kn.pairloop && ic.kc_pw && ic.nkb_pw == 1u;
    ip.pairwise = !ip.pairloop && !OZ1_INT32_PARTIAL && kn.defer_recon
                  && (ic.acc_fits_pw || ic.fuse_ok) && ic.defer_ok && ic.kc_pw
                  && ic.n_pw < ic.n_cat;
    if (!OZ1_INT32_PARTIAL && !ip.pairwise && !ip.pairloop)
        fmt::print("Issue   : 対ごとの発行は見送る（{}）\n",
                   !kn.defer_recon ? std::string("遅延再構成を切っている（OZ_NO_DEFER）")
                   : !ic.kc_pw     ? std::string("無誤差な K ブロック長が取れない")
                   : !ic.defer_ok ? fmt::format("Int32 累積が 2^31 を超えうる（(L+1)*K*d_max^2 > 2^31-1．oz_defer_fits）")
                   : !(ic.acc_fits_pw || ic.fuse_ok)
                       ? fmt::format("Int32 累積面 {} 面 {:.1f} GB が入らない", ic.ndiag, ic.acc_need / 1e9)
                       : fmt::format("本数が {} 対 {} で連結のほうが少ない", ic.n_pw, ic.n_cat));
}

// 選んだ発行のしかたで K ブロック長 K_c が取れること・再構成が成り立つことを確かめる（取れなければ理由を stderr に
// 表示して false．plan_issue の後半）．kc_pw は対ごとの K ブロック長（エラー文の処方に使う）
static bool issue_feasible(const RunCfg& c, const IssuePlan& ip, uint32_t kc_pw) {
    if (ip.Kc == 0) {
        // ここへ来るのは連結の側で取れなかったときだけである（対ごとの発行と
        // PAIRLOOP は kc_pw != 0 を条件にしている）．
        fmt::print(stderr, "ERROR: radix={} p={} K={} では対角を連結した無誤差な K ブロック長を"
                           "取れません（K が 32 の倍数の約数を持つようにするか，基数を"
                           "小さくしてください）．{}\n", c.radix, c.p, c.K,
                   kc_pw ? fmt::format("対ごとの発行なら K_c={} で取れるが，この構成では"
                                       "使えない（OZ_NO_DEFER，または上の Issue 行の理由）", kc_pw)
                         : std::string("1 対ぶんでも取れない"));
        return false;
    }
    // INT8 経路の再構成カーネル（ozaki1/kernels/recon_tf_i32in.cpp）は ACC を
    // 実装しておらず，コンパイル時引数 RC_ACC（oz1_args.h の ReconCT）を読まない．したがって K ブロックが
    // 複数あると各ブロックが前のブロックの結果を黙って上書きし，エラーを出さない
    // まま誤答になる．基数 254（桁 127）・ACC_BITS 31 では K_c の上限が
    // 2^(31-14)/(L+1) で，本稿のサイズ（L+1=8，K<=14080）では常に nkb==1 に
    // なるので現状は到達しないが，基数や K を変えた瞬間に踏むのでここで止める．
    if (OZ1_INT32_PARTIAL && ip.nkb > 1) {
        fmt::print(stderr, "ERROR: INT8 経路は K ブロック分割に未対応です"
                           "（K_c={} で {} ブロック）．recon_tf_i32in.cpp が累積版を"
                           "持たないため，このまま進むと誤答になります．\n", ip.Kc, ip.nkb);
        return false;
    }
    // 再構成の Dekker 分割が FP32 を溢れないか．無誤差累積の条件とは別で，
    // p を手で大きくしたときに踏む（radix 32・K=512 なら p=22 で判定に掛かる）．
    // 2 のべきの基数（BFP2_b・BF16 系・FP8）の基数倍は tfMulP2 で 4097 倍をしないが，判定は
    // 基数によらず同じ式で安全側に置いている（ozaki1_format.hpp の oz_horner_fits）．下のエラー文の
    // 「Dekker 分割（4097 倍）」はそのときも同じ文言で出る（安全側に判定している旨を添える）．
    if (!oz_horner_fits((uint32_t)c.radix, c.K, c.L)) {
        fmt::print(stderr,
            "ERROR: radix={} p={} K={} では再構成の Dekker 分割（4097 倍）が FP32 を"
            "溢れます．log2(K)+(L+1)log2(radix)+12 < 128 を満たす p を使ってください"
            "（2 のべきの基数でも同じ式で安全側に判定している）．\n",
            c.radix, c.p, c.K);
        return false;
    }
    return true;
}

// 発行のしかた（連結・対ごと・PAIRLOOP）と累積のしかた（PAIRLOOP・fuse・scratch・
// 遅延再構成），および無誤差条件を満たす K ブロック長 K_c を決め，見送った理由と
// Exact 行を表示する．dram_free は何も確保していない時点の空き容量．
// デバイスには触らず，何も確保しない．構成が成り立たなければ ok=false を返す．
static IssuePlan plan_issue(const RunCfg& c, size_t dram_free, const Oz1Knobs& kn) {
    IssuePlan ip;
    const IssueCandidates ic = issue_candidates(c, dram_free, kn);
    choose_issue(ip, ic, kn);
    ip.Kc = ip.per_pair() ? ic.kc_pw : ic.kc_cat;
    ip.nkb = ip.Kc ? c.K / ip.Kc : 0u;   // K ブロック数
    // 累積のしかたは 3 通りで，上から順に良い（上が取れなければ下へ落ちる）．
    //   PAIRLOOP  対の境界で L1 の Int32 面へ畳む．Pbig がそのまま累積面になり，
    //             S_d の往復は対角あたり 1 回．既定．
    //   fuse      GEMM の出力経路で DRAM の S_d を読んで足す．Pbig が累積面を
    //             兼ねるのは同じだが，1 更新あたり 2 面の往復が残る．
    //   scratch   累積を別プログラムで行い，Pbig はスクラッチ 1 面にする．
    //             累積面 ndiag 面を別に確保するので DRAM の増分が最も大きい．
    // 既定のつまみでは PAIRLOOP が取れないとき（1 対の K が 1 ブロックに収まらない長い K．
    // selftest.sh の「I-BF16 pw+Kblock」）に fuse へ落ちる．OZ_NO_PAIRLOOP を立てたときの落ち先は本数と
    // K ブロック数で決まる：n_pw < n_cat なら対ごとの発行と fuse の組，そうでなく連結の K ブロックが複数なら
    // 連結の fuse（3520^3 の BF16 は 55 本対 50 本でこれ），連結の K ブロックが 1 本なら累積を持たない
    // run_single_kblock（3520^3 の BFP4_b・FP8，K <= 1632 の BF16 など）である．
    // scratch は fuse が取れる構成（fuse_ok）では選ばれないので，INT8 以外で通るのは
    // OZ_NO_FUSE_ACC を立てて K ブロックが複数になったときだけである．
    const bool pairloop = ip.pairloop, pairwise = ip.pairwise;
    ip.fuse = !pairloop && ic.fuse_ok && (pairwise || ip.nkb > 1);
    ip.scratch = !ip.fuse && !pairloop && !pairwise && (ip.nkb > 1)
                 && !OZ1_INT32_PARTIAL
                 && kn.defer_recon && ic.defer_ok && ic.acc_fits_scr;
    // 遅延再構成：K ブロックごとに triple-float の Horner を回す代わりに，対角ごとの
    // 部分和を Int32 で足しておき，再構成は最後に 1 回だけ行う．Horner は P_d について
    // 線形なので数学的には同じ値になる（成立条件は oz_defer_fits．int8 経路は本稿のサイズでは nkb==1）．
    // 有限精度では同じとは限らない：K ブロックごとに Horner を回すと triple-float の丸めが
    // ブロックの数だけ入るので，最下位のビットは変わりうる（精度の基準は保つ）．
    // 対ごとに回すと 1 対の出力（<=2^24）を対角あたり (d+1) 本足すので，
    // 足し込みは必ず Int32 で行う（遅延再構成が前提になる）．
    // defer が立つのは pairwise・scratch・fuse・PAIRLOOP のどれかのときだけである．
    // fuse と PAIRLOOP は Pbig を累積面に使うので別の累積バッファを要らず，
    // pairwise と scratch は Int32 累積なしでは成り立たない．したがって run_sweep で
    // 累積バッファが確保できなければ，K ブロックごとに Horner を回す経路へは落とさずエラーで止まる．
    ip.defer = (pairwise || ip.scratch || ip.fuse || pairloop) && !OZ1_INT32_PARTIAL
               && kn.defer_recon && ic.defer_ok;
    if (!issue_feasible(c, ip, ic.kc_pw)) return ip;
    // 既定のつまみでは分割後は必ず通る（通らなければ ozaki_block_k の側の誤り）．通らないのは診断のつまみ
    // （OZ_NOBLOCK・OZ_RADIX・OZ_DMAX・OZ_FIDELITY）で条件を外したときで，警告して続ける．
    // per_pair なら 1 本の K_eff は 1 対ぶんなので L ではなく 0 で検査する．
    // Exact 行は判定の結果を見てから出す（前に出すと，条件を外した構成でも stdout には「満たす」と残る）．
    // 累積の不等式の成否と乗算の条件（桁の上限と MathFidelity）の成否は分けて書く（乗算だけで落ちた構成に，
    // 成り立っている不等式を「満たさない」と残さない）．
    const OzExact ex = ozaki_check_exact((uint32_t)c.radix, c.dmax, c.fid, ip.Kc, ip.per_pair() ? 0 : c.L);
    fmt::print("Exact   : b_acc={} bit, K_c={} ({} block{}) -> "
               "2w+ceil(log2({}K_c)) <= b_acc を{}（{}）{}\n",
               OZAKI_ACC_BITS, ip.Kc, ip.nkb, ip.nkb == 1 ? "" : "s",
               ip.per_pair() ? "" : "(L+1)",
               ex.acc ? "満たす" : "満たさない．理由は stderr の [WARN]",
               pairloop ? "対を 1 本の中で回す" :
               pairwise ? "対ごとに発行" : "対角を連結",
               ex.mul ? "" : "．乗算が無誤差でない（桁の上限と MathFidelity．理由は stderr の [WARN]）");
    ip.ok = true;
    return ip;
}

// ===== 出力パネルと行バンド =====
// 出力パネル（左上タイル座標）と行バンド（タイル行の範囲とパネル番号の範囲）
struct Panel { uint32_t row0, col0; bool tail; };   // tail: 端数行の構成 gR を使う
struct Band { uint32_t row0, rows, p0, p1; };

// 出力パネルと行バンドの分割．名前は ozaki1_main.cpp の冒頭の一覧
struct PanelPlan {
    bool ok = false;
    uint32_t pdiag_tile = 0, pbig_planes = 0;   // Pbig のタイルの大きさと面数
    uint32_t nslice = 0;                        // スライスの本数 L+1（連結の段数）
    uint32_t Kmax = 0, Ktm = 0, Kbt = 0;        // 連結後の K［要素］とそのタイル数，K ブロック 1 個ぶんの K タイル数
    OptConfig gU, gR;                           // 通常パネル・端数行パネルの構成
    uint32_t nbr = 0, nbc = 0, rem_rows = 0;    // パネルの行数・列数，端数の行数［タイル］
    std::vector<Panel> panels;                  // 構築も発行もこの順で回す
    uint32_t nband = 0, band_rows = 0, band_tiles = 0;
    std::vector<Band> bands;
    // 計画に使ったデバイスの格子（compute_with_storage_grid_size．x が列）と DRAM の空き（dram_free．全バンクぶん［B］）．
    // CONFIG の dev_grid・dram_free_gb・dram_largest_gb に出す（別のボードで構成が変わった理由をログが語るように．2026-11-04）
    uint32_t dev_gx = 0, dev_gy = 0;
    uint64_t dram_total = 0, dram_largest = 0;
    const OptConfig& cfg_of(const Panel& pn) const { return pn.tail ? gR : gU; }
    uint32_t panel_rows() const { return gU.cr * gU.pm; }   // 通常パネルの高さ［タイル］
};

// GEMM の形状（コア格子・コアあたりのタイル数・K ブロック幅）と出力パネルの覆い方を決める（plan_panels の前半）．
// l1_budget は形状探索の L1 の予算［B］（Oz1Knobs）．取れなければ理由を表示して false．
static bool plan_panel_shapes(PanelPlan& pp, const RunCfg& c, const IssuePlan& ip, const tt::tt_metal::CoreCoord& gd,
                              uint32_t l1_budget) {
    const uint32_t Mt = c.Mt, Nt = c.Nt;
    // ---- 対角ごとの GEMM の出力先 Pbig の形 ----
    // Pbig には対角ごとの部分和が符号付きで入る（スロット d はタイル d*band_tiles から．バンドに切らなければ
    // band_tiles = Mt*Nt）．融合時と PAIRLOOP の Pbig は Int32 の累積面そのもの（GEMM が読んで足して書き戻す）．
    pp.pdiag_tile = tt::tile_size((ip.fuse || ip.pairloop) ? tt::DataFormat::Int32 : tt::DataFormat::Float32);
    // スクラッチ方式では対角ごとに足し込むので，部分和は 1 面で足りる．
    pp.pbig_planes = ip.scratch ? 1u : c.nslice();
    // 融合時は GEMM の CB が c_2・c_3 のぶん増えるので，形状探索の L1 予算に反映する．
    // 融合と PAIRLOOP は排他（PAIRLOOP は S の先読み c_2 を使わない）．
    const GemmMode gmode{ip.fuse, ip.pairloop};
    // 全対角で 1 つのプログラムを使い回す．対角 ℓ が持つ対の数は (ℓ+1) 個なので
    // K の長さは対角ごとに違うが，ゼロ埋めはしない．連結バッファは K_max = (L+1)*K
    // ぶんを確保し，対角ごとに違う K ブロック数 nb = (ℓ+1)*Kbt/bw は gemm_run が
    // 実行時引数（リーダの RA_NUM_BLOCKS と計算カーネルの CR_NUM_BLOCKS．common/kernels/mcast_args.h・gemm_compute_args.h）
    // で渡す．計算カーネルのコンパイル時引数 CA_NUM_BLOCKS は最大値を入れてあるだけで読まれない．
    // こうして全対角の GEMM がコンパイル時引数まで同一になり，カーネルの JIT ビルドが
    // 1 回で済む．対角ごとに組み直すと JIT 再コンパイルが実行時間を支配してしまう．
    // nslice はスライス連結レイアウトの「1 K ブロックあたりのスライス本数」でも
    // ある．打ち切り対角 L のもとで使う対は i+j <= L なので，実際に要るスライスは
    // 0..L の L+1 本だけで，i > L のスライスはどの対角にも現れない．スライス段
    // （slice_build/slice_run の nslice）と，バッファ長 Ktm，GEMM 側の段数 nslice
    // は同じ本数でなければならない：ここを p にすると L < p-1 の
    // ときだけ書き込み位置の刻みがずれ，かつ shA/shB の外へ書いて，
    // エラーを出さないまま誤った C を返す（既定の L = p-1 では p と一致する）．
    pp.nslice = c.nslice();
    pp.Kmax = pp.nslice * c.K; pp.Ktm = pp.Kmax / 32;
    pp.Kbt = ip.Kc / 32;   // K ブロック 1 個ぶんのタイル数
    const uint32_t Ktm = pp.Ktm, Kbt = pp.Kbt;
    // 出力パネル分割の構成を選ぶ．出力を (cr*pm)×(cc*pn) の nbr×nbc ブロックに
    // 分けることで，任意の（大きな）M,N を全コアで計算できる．
    // GEMM の K ブロック幅 bw は Kbt を割り切る必要がある：さもないと 1 回の
    // 読み出しが K ブロックやスライスの境界をまたいでしまう．
    // 形状は 8x8 まで広げて探す（出力パネルを何度も舐め直す構成ではパネル数を減らすほうが得で，
    // INT8 経路でも 8x8 のほうが速い．ozaki1_gemm.hpp の choose_cfg_1b の前のコメント）．
    // 探索の引数はタイル数（Kt は連結後の Ktm，Kbt は K ブロック 1 個ぶん）．
    const OptConfig gU = choose_cfg_1b(Mt, Nt, Ktm, Kbt, gd.y, gd.x, gmode, l1_budget);
    if (!gU.ok) { fmt::print(stderr, "ERROR: 1b cfg fail Mt={} Nt={} Ktm={}\n", Mt, Nt, Ktm); return false; }
    // パネルの覆い方．行方向は端数を許すので，最後の 1 行だけコア格子が小さい
    // 構成 gR で覆う．列方向は割り切れる構成だけを choose_cfg_1b が返すので
    // 端数は出ない．
    const uint32_t ph = gU.cr * gU.pm, pw = gU.cc * gU.pn;   // 通常パネルの高さ・幅
    const uint32_t nbr_full = Mt / ph, nbc = Nt / pw;
    const uint32_t rem_rows = Mt - nbr_full * ph;            // 端数の行数［タイル］
    OptConfig gR = gU;
    if (rem_rows) {
        // 端数行は (cr,pm) を取り直してコア数を最大にする．取り直しは必ず成功する：
        // choose_cfg_1b は (Mt/pm) % cr >= 2 のときだけ端数を許すので rem_rows = r*pm（2 <= r < cr <= gy）
        // で，choose_remainder_cfg の候補 cr'=r は pm'=pm（PM の表にある）・通常パネルと同じ pn と
        // サブブロック・同じ bw の候補を持つ．取れなければ前提が崩れているので，代わりの構成を探さずに止める．
        OptConfig gc = choose_remainder_cfg(rem_rows, gU, gd.y, Ktm, Kbt, gmode, l1_budget);
        TT_FATAL(gc.ok, "plan_panels: 端数行 {} タイル（pm={}）の構成が取れない", rem_rows, gU.pm);
        gR = gc;
    }
    const uint32_t nbr = nbr_full + (rem_rows ? 1u : 0u);
    pp.gU = gU; pp.gR = gR;
    pp.nbr = nbr; pp.nbc = nbc; pp.rem_rows = rem_rows;
    // パネルの左上タイル座標を並べておく（構築も発行もこの順で回す）
    pp.panels.reserve((size_t)nbr * nbc);
    for (uint32_t br = 0; br < nbr; br++)
        for (uint32_t bc = 0; bc < nbc; bc++)
            pp.panels.push_back({br * ph, bc * pw, rem_rows && br == nbr_full});
    return true;
}

// ---- 出力を行バンドへ切る（plan_panels の後半）----
// 対角ごとの累積面 Pbig は出力の全体ぶんだと ndiag 面で DRAM を使い切る
// （BFP2_b の 14080^3 で p 面．8197a0b の routes.tsv の p=59 なら 46.8 GB）．出力をタイル行のバンドへ切り，
// バンド 1 本ぶんの面だけを確保して「そのバンドの GEMM → そのバンドの再構成」
// を順に回せば，面の容量はバンド数ぶん小さくなる．演算量は変わらない．
// バンドの境界はパネル行の境界に合わせる（パネルが 2 本のバンドに
// またがらないようにするため）．バンドは行方向にだけ切るので，出力の列数 Nt は
// 変わらず，パネルの書き込み位置がバンド先頭からの相対になるだけである．
// 行の数え方：bpr はバンド 1 本あたりのパネル行数．OZ_BANDS が無ければ，dram_free（まだ何も確保していない時点の空き容量）から
// これから確保するもの（oz1_dram_need）と余裕を引いた残り avail に Pbig（と別に確保する累積面）が収まる最大の行数にする．
// どちらの決め方でも，決めた bpr のバンド 1 本ぶんの面（need）が avail に収まるか（fits）を返す：1 パネル行でも
// 収まらない・指定した本数では収まらない構成は，確保（main の alloc_pbig）で例外になってデバイスを閉じずに落ちる前に止める．
struct BandRows {
    uint32_t bpr = 1;           // バンド 1 本あたりのパネル行数
    bool fits = false;          // バンド 1 本ぶんの面が avail に収まるか
    uint64_t need = 0, avail = 0;   // その比べの両辺［B］
};
static BandRows plan_band_panel_rows(const PanelPlan& pp, const RunCfg& c, const IssuePlan& ip, const Oz1Knobs& kn,
                                     size_t dram_free) {
    const uint32_t Mt = c.Mt, Nt = c.Nt, nbr = pp.nbr, ph = pp.panel_rows();
    // タイル 1 行ぶんの Pbig（pbig_planes 面）．別プログラムで累積する構成（対ごとの発行・
    // スクラッチ）では，遅延再構成の累積面 S（Int32 で ndiag 面．ozaki1_accum.hpp の accum_alloc）もバンドの
    // 行数ぶん要る．「Pbig と同じ大きさをもう 1 組」と数えると，Pbig が 1 面の
    // スクラッチでは S の ndiag 面を 1 面と見積もってしまう．
    const size_t row_bytes = (size_t)pp.pdiag_tile * pp.pbig_planes * Nt;
    const size_t acc_row_bytes =
        ip.sep_accum() ? (size_t)tt::tile_size(tt::DataFormat::Int32) * (size_t)c.nslice() * Nt : 0u;
    const Oz1DramNeed need = oz1_dram_need(c, ip.recon_limbs());
    oz1_dram_dbg("band", dram_free, need, "Pbig+acc (1 tile row)", row_bytes + acc_row_bytes);
    const uint64_t later = need.total() + OZ_DRAM_MARGIN_BYTES;
    const uint64_t avail = ((uint64_t)dram_free > later) ? (uint64_t)dram_free - later : 0u;
    auto band_bytes = [&](uint32_t bpr) { return (uint64_t)(row_bytes + acc_row_bytes) * std::min(Mt, bpr * ph); };
    BandRows r;
    if (kn.bands) {
        // 0 や読めない値を黙って 1 本に倒さない（検査はデバイスを開く前の parse_run_config で
        // 済ませてある）．上限（つまみの表）は nbr + nb が溢れないための安全弁で，パネル行数を超える指定は
        // バンド 1 本 1 パネル行になる．
        const uint32_t nb = *kn.bands;
        r.bpr = std::max(1u, (nbr + nb - 1) / nb);
    } else {
        r.bpr = nbr;
        while (r.bpr > 1 && band_bytes(r.bpr) > avail) r.bpr--;
    }
    r.need = band_bytes(r.bpr);
    r.avail = avail;
    r.fits = r.need <= avail;
    return r;
}

// GEMM の形状と出力パネルの覆い方，および Pbig を DRAM に収めるための行バンドの切り方を決め，Band 行を表示する．
// デバイスには触らず，何も確保しない．バンド 1 本ぶんの面が DRAM に収まらなければ理由を表示して ok=false
// （main はデバイスを閉じてから 2 で終わる）．
// df は確保の前の DRAM の空き（総量と最大の連続した空き．ozaki_gemm_host.hpp の dram_free）．総量で行バンドの本数を決め，
// 大きい器（FP64 のオペランド・スライス連結・FP64 の出力・バンド 1 本ぶんの Pbig）がそれぞれ最大の連続した空きに入ることも
// 見る（入らなければ確保で例外になり，デバイスを閉じずに落ちる．スキーム II の plan_bands と同じ必要条件で，先に確保した器が
// 連続した空きを割った後の大きさは見ない）．
static PanelPlan plan_panels(const RunCfg& c, const IssuePlan& ip, const Oz1Knobs& kn,
                             const tt::tt_metal::CoreCoord& gd, const DramFree& df) {
    PanelPlan pp;
    pp.dev_gx = (uint32_t)gd.x; pp.dev_gy = (uint32_t)gd.y;
    pp.dram_total = df.total; pp.dram_largest = df.largest_block;
    const size_t dram_free = df.total;
    if (!plan_panel_shapes(pp, c, ip, gd, kn.l1_budget)) return pp;
    const uint32_t Mt = c.Mt, Nt = c.Nt, nbr = pp.nbr, nbc = pp.nbc, ph = pp.panel_rows();
    const BandRows brs = plan_band_panel_rows(pp, c, ip, kn, dram_free);
    if (!brs.fits) {
        fmt::print(stderr, "ERROR: 対角の部分和 Pbig{}のバンド 1 本（{} パネル行）が DRAM に入らない（要 {:.2f} GB，"
                           "ほかの器と余裕 {:.1f} GB を除いた空き {:.2f} GB）．{}\n",
                   ip.sep_accum() ? "と累積面 S " : "", brs.bpr, brs.need / 1e9, OZ_DRAM_MARGIN_BYTES / 1e9,
                   brs.avail / 1e9,
                   kn.bands ? "OZ_BANDS の本数を増やす" : "1 パネル行でも入らないので，M・N・K か p を小さくする");
        return pp;
    }
    const uint32_t bpr = brs.bpr;
    pp.nband = (nbr + bpr - 1) / bpr;
    pp.band_rows = std::min(Mt, bpr * ph);        // 面の確保幅［タイル行］
    pp.band_tiles = pp.band_rows * Nt;            // 面と面の間隔［タイル］
    for (uint32_t b0 = 0; b0 < nbr; b0 += bpr) {
        const uint32_t b1 = std::min(nbr, b0 + bpr);
        pp.bands.push_back({b0 * ph, std::min(Mt, b1 * ph) - b0 * ph,
                            b0 * nbc, b1 * nbc});
    }
    if (pp.nband > 1)
        fmt::print("Band    : 出力を {} 本の行バンドへ切る（1 本 {} タイル行，"
                   "対角 {} 面で {:.1f} GB．全体だと {:.1f} GB）\n",
                   pp.nband, pp.band_rows, pp.pbig_planes,
                   (double)pp.pdiag_tile * pp.pbig_planes * pp.band_tiles / 1e9,
                   (double)pp.pdiag_tile * pp.pbig_planes * Mt * Nt / 1e9);
    // 大きい器それぞれが最大の連続した空きに入るか（上の注．器は全バンクへ同じ番地で置かれるので全バンクぶんで比べる）
    const Oz1DramNeed nd = oz1_dram_need(c, ip.recon_limbs());
    const uint64_t f64t = OZ_F64_TILE_BYTES, in_t = tt::tile_size(OZ_FMT.in);
    const uint64_t Kto = c.K / 32, Ktm = (uint64_t)c.nslice() * Kto;
    const std::pair<const char*, uint64_t> big[] = {
        {"FP64 の A", f64t * Mt * Kto}, {"FP64 の B", f64t * Kto * Nt},
        {"スライス連結の A", in_t * Mt * Ktm}, {"スライス連結の B", in_t * Nt * Ktm},
        {"FP64 の出力", nd.output}, {"Pbig のバンド 1 本", (uint64_t)pp.pdiag_tile * pp.pbig_planes * pp.band_tiles}};
    for (const auto& [what, bytes] : big)
        if (bytes > df.largest_block) {
            fmt::print(stderr, "ERROR: {}（{:.2f} GB）が DRAM の最大の連続した空き（{:.2f} GB．空きの総量 {:.2f} GB）に入らない\n",
                       what, bytes / 1e9, df.largest_block / 1e9, df.total / 1e9);
            return pp;
        }
    pp.ok = true;
    return pp;
}

// ===== 再構成の定数 =====
// 遅延再構成の Int32 累積 S_d のうち，先頭から何本が FP32 に無誤差に載るか．
// 対角 d には対 (i, d-i) が d+1 個あり（d <= L = p-1），各対の K 個の積は桁の上限 d_max の
// 2 乗で抑えられるので，K ブロックを全部足した後でも
//     |S_d| <= (d+1) * K * d_max^2
// である（K は分割前の全長．oz_defer_fits の上界の対角ごと版）．右辺が 2^24 以下なら S_d は
// FP32 の無誤差整数で，recon_tf_i32in.cpp は 2 本への分解をせず 1 回の型変換で読める．
// 右辺は d について増えるので，条件を満たす対角は先頭から連続し，その本数は
// floor(2^24 / (K d_max^2)) と ndiag の小さいほうになる．14080^3 では BF16・FP16・TF32・BFP8_b
// が 1 本（d=0），FP8 が 4 本，BFP4_b と BFP2_b は全部（8197a0b の routes.tsv の p では 16 本と 59 本），INT8 は 0 本．
static uint32_t recon_nsmall(uint32_t dmax, uint32_t K, uint32_t ndiag) {
    const uint64_t dm = dmax;
    const uint64_t per = (uint64_t)K * dm * dm;          // 対 1 つぶんの上界 K*d_max^2
    if (per == 0) return 0;
    return (uint32_t)std::min<uint64_t>((1ull << 24) / per, ndiag);
}

// 基数と最終スケール radix^-(L+2) を再構成カーネルへ渡す形にする（ホストの計算のみ）．
// defer は遅延再構成を使うか（plan_issue の ip.defer）．近道 NSMALL を使うのはそのとき（2 の補数の Int32 の
// 累積面を読む recon_tf_i32in.cpp）だけで，INT8 経路（符号-絶対値）と非遅延の経路ではカーネルへ 0 が渡る
// （ozaki1_recon.hpp の recon_build）．nsmall_req（OZ_NSMALL）が上界を超えていれば理由を表示して空を返す（main は
// デバイスを閉じてから 2 で終わる）．ライタの組み方（soft-float 版か・OZ_RECON_PROBE）は dg から写す．
static std::optional<ReconScale> recon_scale(const RunCfg& c, bool defer, std::optional<uint32_t> nsmall_req,
                                             const Oz1Diag& dg) {
    const int radix = c.radix, L = c.L;
    const uint32_t K = c.K;
    ReconScale rs;
    rs.softfloat = dg.softfloat ? 1u : 0u;
    rs.probe = dg.recon_probe;
    const uint32_t bound = recon_nsmall(c.dmax, K, c.nslice());
    rs.nsmall = bound;
    // OZ_NSMALL=n（比較用．docs/env_vars.md の B）：近道で読む対角を先頭の n 本に減らす．0 なら全対角を
    // 2 本に分けて足す（近道を使わない読み方との比較に使う）．上界を満たす本数より多くは
    // 受け付けない（2^24 を超える S_d を型変換 1 回で読むと無誤差でなくなる）．近道を使わない経路では見ない．
    if (nsmall_req) {
        const uint32_t v = *nsmall_req;
        if (defer && v > bound) {
            fmt::print(stderr, "ERROR: OZ_NSMALL={} は上界を満たす本数 {}（K={}，基数 {}）を超える\n", v, bound, K, radix);
            return std::nullopt;
        }
        rs.nsmall = std::min(v, bound);
    }
    // used は実際にカーネルへ渡る本数（近道を使わない経路では 0），bound は上界を満たす本数
    fmt::print("RECON-NSMALL,used={},bound={},ndiag={},K={}\n", defer ? rs.nsmall : 0u, bound, L + 1, K);
    // 再構成カーネルへは基数を float のビット列として渡す
    rs.radixbits = f32_bits((float)radix);
    // 最終スケール radix^-(L+2) は一般には 2 のべきとは限らない（BFP4_b の基数 14，INT8 の 254）ので，全形式で
    // 共通に，ライタが整数で掛けられるよう 64 bit の仮数 W と指数 qw に分けて渡す（基数が 2 のべきなら W は
    // 2^63 ちょうど．common/oz_bits.h の oz_split_scale_ld）．FP64 の定数として乗算 1 回で済ませると，そこで相対
    // 2^-53 の誤差が全要素に一様に乗ってしまう．
    // W の誤差：long double（x86 の 64 bit 仮数）で radix を L+2 回割るので，割り算ごとの丸め
    // （相対 2^-64 以下）が積み上がり，相対で高々約 (L+2)·2^-64 になる（INT8 の L=7 で 9·2^-64）．
    // 最後の 64 bit への写しは仮数をそのまま移すだけで誤差を足さない．基数が 2 のべきなら割り算も
    // 無誤差で W は厳密である．いずれも FP64 の 2^-53 より十分小さい．
    {
        long double w = 1.0L;
        for (int i = 0; i < L + 2; i++) w /= (long double)radix;
        const OzScale64 W = oz_split_scale_ld(w);
        rs.Whi = W.hi; rs.Wlo = W.lo; rs.qw = W.q;
    }
    // soft-float 版（OZ_SOFTFLOAT=1）が使う wscale の double ビット列
    {
        const uint64_t u = f64_bits(std::pow((double)radix, -(double)(L + 2)));
        rs.Wdlo = (uint32_t)u; rs.Wdhi = (uint32_t)(u >> 32);
    }
    return rs;
}
