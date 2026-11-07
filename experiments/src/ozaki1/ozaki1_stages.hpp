#pragma once
// 尾崎スキーム I：デバイスの段の資源と投入の手順（同期も計時もしない）
// ---------------------------------------------------------------------------
// 段ごとの計時（ozaki1_main.cpp の run_scale_slice・run_single_kblock・run_sweep）と，OZ_TIME_WHOLE の全工程の流し直し
// （ozaki1_whole.hpp）が同じ手順で投入するよう，投入の手順をここに 1 つだけ置く．計時と同期（Finish）は呼び出し側が挟む．
// 投入した手順を数える note は，全工程の流し直しの検査（common/oz_whole.hpp）だけが数える関数を渡し，段ごとの計時は
// 何もしない関数を渡す（計時の経路に検査の分岐を置かない）．
// 依存：ozaki1_common.hpp（スケール・スライス・GEMM・再構成の各段），ozaki1_accum.hpp（OZ_NO_FUSE_ACC の累積），
// ozaki1_plan.hpp（RunCfg・Oz1Knobs・IssuePlan・PanelPlan），common/oz_whole.hpp（手順の種類）．

#include "ozaki1/ozaki1_common.hpp"
#include "ozaki1/ozaki1_accum.hpp"
#include "ozaki1/ozaki1_plan.hpp"
#include "common/oz_whole.hpp"

// 計時の集計（秒）．fill は iters 回の平均，GEMM・再構成は iters 回の平均の和．
struct Times {
    double build = 0.0, gemm = 0.0;
    // fill はスケール算出とスライス処理の合計，fill_scale はそのうちスケール算出
    // （この 2 つは iters 回の平均）．fill_build はスケール・スライサの構築と
    // 初回 JIT prime を 1 回だけ測ったもの（平均ではない）．
    double fill = 0.0, fill_scale = 0.0, fill_build = 0.0;
    // デバイス側スケール算出の内訳タイマ（dsc_calls 回ぶんの合計）
    double dsc_scale = 0.0, dsc_glue = 0.0;
    uint32_t dsc_calls = 0;
    double recon_dev = 0.0, recon_build = 0.0, recon_read = 0.0;
    double tot = 0.0;
    // OZ_TIME_WHOLE：全工程を続けて流した 1 回あたりの時間（秒．iters 回の平均）
    double whole = 0.0;
};

static auto now() { return std::chrono::steady_clock::now(); }

// デバイス側スライスの入力バッファ
struct InputBufs {
    // デバイス側の各段が使うバッファ：
    //   rAf64/rBf64  FP64 のオペランドそのもの（1 要素 8 バイト，タイル順）．
    //                スケール算出もスライサもここから直接読み，FP64 -> triple-float
    //                の変換はデバイス側（slice_f64_reader.cpp）が行う．
    //   maxAb/maxBb  行／列ごとの最大値（BF16）
    //   dltAb/dltBb  指数差分 shift-e のブロードキャスト（Int32）．スライサが 0 回目で
    //                FP64 の指数フィールドへ足して正規化を融合する
    //   flgA/flgB    範囲外検出の旗（コアごとに 1 タイル）
    //   shA/shB      スライス連結済みのオペランド
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> rAf64, rBf64, maxAb, maxBb, dltAb, dltBb;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> flgA, flgB, shA, shB;
};

// スケール算出とスライス分解のワークロード．C を読んだあとで旗を見るので，
// スケールの仕上げ（edA/edB）は最後まで持つ．sliceA・sliceB は A・B のスライス処理 1 回ぶんの入出力と形．
struct FillStage {
    ExpDeltaCtx edA, edB;
    ScaleCtx scA, scB;
    SliceCtx slc;
    SliceRun sliceA, sliceB;
};

// スケール算出・指数差分・スライスの投入（どれも A・B の 2 本ずつ）．順はスケール A・B → 指数差分 A・B → スライス A・B で，
// 同じコマンドキューの上では投入の順に実行されるので，同期を挟まなくても依存関係は守られる．
// OZ1_FILL_PROGRAMS はこの 3 つの投入の本数の和（OZ_TIME_WHOLE の検査が本数の期待値に使う）．
constexpr uint64_t OZ1_FILL_PROGRAMS = 6;
template <class Note>
static void fill_issue_scale(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs, Note&& note) {
    scale_run(md, fs.scA); note(oz_whole::HEAD);
    scale_run(md, fs.scB); note(oz_whole::HEAD);
}
template <class Note>
static void fill_issue_expdelta(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs, Note&& note) {
    expdelta_run(md, fs.edA); note(oz_whole::HEAD);
    expdelta_run(md, fs.edB); note(oz_whole::HEAD);
}
// between は A と B のスライスの間に挟む処理（段ごとの計時の OZ_DBG の表示と同期）
template <class Note, class Between>
static void fill_issue_slice(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs, Note&& note,
                             Between&& between) {
    slice_run(md, fs.slc, fs.sliceA); note(oz_whole::HEAD);
    between();
    slice_run(md, fs.slc, fs.sliceB); note(oz_whole::HEAD);
}

// GEMM・再構成の段が参照する構成と資源（所有はしない）
struct StageCtx {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md;
    const RunCfg& c;
    const Oz1Knobs& kn;
    const Oz1Diag& dg;
    const IssuePlan& ip;
    const PanelPlan& pp;
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig;
    const InputBufs& in;
    const ReconScale& rs;
    uint32_t gw, gi;
};

// バンドの高さは 2 通りしかない（通常のバンドと，端数の最後の 1 本）ので，
// 再構成のプログラムもその 2 つだけ組み，limb と FP64 出力のバッファは共有する．
// 最後に rcn.out から FP64 を読み戻すので，main の終わりまで持つ．
struct BandRecon {
    ReconCtx rcn, rcnT;
    uint32_t rcn_h = 0;
    bool rcn_built = false, rcnT_built = false;
};

// 高さ rows のバンド用の再構成を返す（無ければ組む）．src は再構成の入力（Pbig か累積面 S）．
// 組み方は計画から決まる：with_acc（K ブロックごとに累積する 3 つのプログラムを組むか）は掃き出しの構成（run_sweep．
// IssuePlan::single_kblock でない）で立ち，defer は IssuePlan::defer そのもの．limb 3 面を実サイズで置くのは
// with_acc && !defer で，これは IssuePlan::recon_limbs と同じ（掃き出しで遅延再構成でないのは K ブロックが複数のときだけ．
// 対ごとの発行もPAIRLOOPも使わないsingle_kblock構成は遅延再構成を使わない．plan_issue参照）．
// recon_build は JIT ビルドのために 1 回試走し，そのとき出力の先頭（オフセット 0）へ
// 書く．端数バンド用の rcnT は出力バッファを rcn と共有するので，先頭のバンドの
// 結果を書いたあとに初めて組むと，その試走が先頭のバンドの結果を上書きする．
// そこで最初の呼び出しで，高さの違う端数バンドの分もまとめて組んでおく．
// 初回の呼び出しが rcn の limb（K ブロックごとに Horner を回す構成だけ実サイズ．recon_build）と FP64 出力を
// 確保する（rcnT はそれを共有する）．
static ReconCtx& band_recon(const StageCtx& s, BandRecon& br, Times& t, uint32_t rows,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& src) {
    const uint32_t Mt = s.c.Mt, Nt = s.c.Nt, ndiag = s.c.nslice();
    const bool with_acc = !s.ip.single_kblock(), defer_in = s.ip.defer;
    if (!br.rcn_built) {
        br.rcn = recon_build(s.md, src, rows, Nt, ndiag, s.rs, with_acc, defer_in, s.in.dltAb, s.in.dltBb,
                             s.pp.band_tiles, Mt * Nt, ReconShared{}, &t.recon_build);
        br.rcn_built = true; br.rcn_h = rows;
    }
    const uint32_t tail_h = s.pp.bands.back().rows;
    if (!br.rcnT_built && tail_h != br.rcn_h) {
        br.rcnT = recon_build(s.md, src, tail_h, Nt, ndiag, s.rs, with_acc, defer_in, s.in.dltAb, s.in.dltBb,
                              s.pp.band_tiles, Mt * Nt, ReconShared{br.rcn.out, br.rcn.hi, br.rcn.mid, br.rcn.lo},
                              &t.recon_build);
        br.rcnT_built = true;
    }
    if (rows == br.rcn_h) return br.rcn;
    // plan_panels の bands の組み方から，高さは先頭と末尾の 2 通りしかない．外れたら前提の破れ（ほかの段と同じく TT_FATAL．
    // 例外は main の catch が表示する）
    TT_FATAL(rows == tail_h, "band_recon: バンドの高さ {} は想定外（{} か {}）", rows, br.rcn_h, tail_h);
    return br.rcnT;
}

// バンド bd の再構成を返し（band_recon），出力の位置をバンドの先頭へ合わせる（バンドに切らない構成では合わせない）
static ReconCtx& band_recon_at(const StageCtx& s, BandRecon& br, Times& t, const Band& bd,
                               const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& src) {
    ReconCtx& rr = band_recon(s, br, t, bd.rows, src);
    if (s.pp.nband > 1) recon_set_out_off(rr, bd.row0 * s.c.Nt);
    return rr;
}

// パネル pi（バンド bd に属する）の GEMM プログラムを組む．
static GemmCtx build_panel(const StageCtx& s, uint32_t pi, const Band& bd) {
    const Panel& pn = s.pp.panels[pi];
    GemmBuildSpec b;
    b.M = s.c.M; b.N = s.c.N; b.K = s.c.K; b.nslice = s.pp.nslice; b.Kbt = s.pp.Kbt;
    b.plane_stride = s.pp.band_tiles;
    b.mode = GemmMode{s.ip.fuse, s.ip.pairloop};
    b.fidelity = s.c.fid; b.blockmm = s.kn.blockmm; b.l1_budget = s.kn.l1_budget; b.probe = s.dg.gemm_probe;
    return gemm_build(s.md, s.pp.cfg_of(pn), b, s.Pbig, s.in.shA, s.in.shB, pn.row0, pn.col0, bd.row0);
}

// ===== K ブロックが 1 つで，対ごとにも PAIRLOOP でも回さない構成（run_single_kblock）=====
// パネル 1 枚の全対角の GEMM（対角 d は K 方向に連結した (d+1) 対を 1 本で回し，スロット d へ書く）
template <class Note>
static void issue_panel_diagonals(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, GemmCtx& ctx, int L,
                                  Note&& note) {
    for (int d = 0; d <= L; d++) {
        gemm_run(md, ctx, (uint32_t)d, (uint32_t)d);
        note(oz_whole::GEMM);
    }
}

// ===== K ブロックが複数・対ごと・PAIRLOOP の構成（run_sweep）=====
// 掃き出しはバンドごとに，ラウンド (b, a) ごとの GEMM（gemms）とその後の累積か再構成（post）を繰り返し，遅延再構成なら
// 最後に 1 回だけ再構成する（final）．ラウンドは K ブロック b ごとに a = 0..inner() で，発行の単位は 3 通りある．
//   対ごと         ラウンド a ごとに「対角 a 以上の a 番目の対」．触る面は a..L なので，累積もその範囲だけ流す．
//   スクラッチ     対角 a ごとに全パネルを回してスクラッチ面 0 へ書き，それを S の面 a へ足す．Pbig が 1 面で済む
//                  （OZ_NO_FUSE_ACC のときだけ）．
//   それ以外       K ブロックごとに全対角（Pbig は ndiag 面．a = 0 だけ）．PAIRLOOP・連結の fuse と，遅延再構成を切った
//                  経路（OZ_NO_DEFER）がこれにあたる．
// 初期化（ACC=0）を使うのは，その面に初めて触るときである．
//   対ごと     ラウンド (b=0,a=0) が全面 0..L を覆うので，そこだけ（accum_exec_range）
//   スクラッチ 各面は K ブロックごとに 1 回触るので，b==0 の回が初回（同）
//   fuse       ブロック 0 の回が全面を覆うので，そこだけ（GEMM の融合ライタの acc）
// PAIRLOOP と fuse は累積が GEMM の中で済んでいるので，post は何も投入しない．
// 遅延再構成を使わない経路（OZ_NO_DEFER）は K ブロックごとに Horner を回す（初期化・中間・最終の 3 プログラム）．
struct Oz1Sweep {
    const StageCtx& s;
    std::vector<GemmCtx>& ctxs;     // 全パネルの GEMM（組んだ順にパネル番号と同じ並び）
    AccumCtx& acc;                  // 別プログラムの累積面（対ごと・スクラッチの構成だけ．ほかは空）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> rsrc;   // 再構成の入力（Pbig か累積面 acc.S）

    uint32_t inner() const { return (s.ip.pairwise || s.ip.scratch) ? (uint32_t)s.c.L : 0u; }
    // バンド bd の再構成（初回は端数バンドの分も組む．組み方は計画から．band_recon）
    ReconCtx& recon(BandRecon& br, Times& t, const Band& bd) const {
        return band_recon_at(s, br, t, bd, rsrc);
    }
    // ラウンド (b, a) のバンド bd の全パネルの GEMM．first はバンドの最初のラウンド（融合なら面へ足さずに置く）
    template <class Note>
    void gemms(const Band& bd, uint32_t b, uint32_t a, bool first, Note&& note) const {
        const bool pairwise = s.ip.pairwise, scratch = s.ip.scratch;
        const int L = s.c.L;
        for (uint32_t pi = bd.p0; pi < bd.p1; pi++) {
            auto& ctx = ctxs[pi];
            if (scratch) {   // 対角 a をスクラッチ面 0 へ（pairwise とは排他）
                gemm_run(s.md, ctx, a, 0u, b, -1);
                note(oz_whole::GEMM);
            } else
                for (int d = (pairwise ? (int)a : 0); d <= L; d++) {
                    gemm_run(s.md, ctx, (uint32_t)d, (uint32_t)d, b,
                             pairwise ? (int)a : -1,
                             // 融合時：その面に初めて触るなら足さずに置く
                             first ? 0u : 1u);
                    note(oz_whole::GEMM);
                }
        }
    }
    // ラウンド (b, a) の後の累積か再構成を投入し，投入したら true を返す
    template <class Note>
    bool post(ReconCtx& rr, uint32_t b, uint32_t a, bool first, Note&& note) const {
        if (s.ip.fuse || s.ip.pairloop) return false;   // 累積は GEMM の中で済んでいる
        const uint32_t plane = s.pp.band_tiles, ndiag = s.c.nslice(), nkb = s.ip.nkb;
        if (s.ip.pairwise) {
            accum_exec_range(s.md, acc, first ? 0u : 1u, a * plane, a * plane, (ndiag - a) * plane);
            note(oz_whole::ACCUM);
        } else if (s.ip.scratch) {   // 対角の GEMM 直後に，その面だけを S_d へ足す
            accum_exec_range(s.md, acc, b == 0 ? 0u : 1u, 0u, a * plane, plane);
            note(oz_whole::ACCUM);
        } else {                     // 遅延再構成でない（sweep_setup の入口の TT_FATAL が前提を確かめる）
            recon_exec(s.md, rr, b == 0 ? 0u : (b + 1 == nkb ? 2u : 1u));
            note(oz_whole::RECON);
        }
        return true;
    }
    // 遅延再構成の最後の 1 回：全ブロックを足し終えた累積面から Horner を回し，FP64 まで組み立てる
    template <class Note>
    void final(ReconCtx& rr, Note&& note) const {
        recon_exec(s.md, rr, 0u);
        note(oz_whole::RECON);
    }
};

// 段ごとの計時が渡す，何も数えない note（インラインで消える）
inline constexpr auto oz1_no_note = [](oz_whole::Kind) {};
