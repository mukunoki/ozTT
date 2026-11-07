#pragma once
// 尾崎スキーム I：OZ_TIME_WHOLE の全工程の流し直し（論文の全体の時間）と，その検査（WHOLE-CHECK）
// ---------------------------------------------------------------------------
// OZ_TIME_WHOLE=1：段ごとの計時（ozaki1_main.cpp の run_scale_slice・run_single_kblock・run_sweep．DRAM->DRAM 行は
// その和で，論文の内訳に使う）のあとで，全工程（スケール算出・指数差分・スライス・全バンドの GEMM と累積と再構成）を
// 段の間で同期せずに続けて流し，段「whole」として温めてから iters 回の平均を測る（WHOLE-PIPELINE 行と CONFIG の whole_ms）．
// 段ごとの計時は段ごとにウォームアップし，K ブロック 1 本の経路では GEMM をパネルごとに反復するので，全工程を 1 回ずつ
// 流した時間とは限らない．論文の全体の時間（表の Total）はこの通しの時間である（2026-11-03 から．それより前は差を測る
// 診断で，立てた実行は probe=set・diag に OZ_TIME_WHOLE が残った）．全工程のプログラムは計時の前に組み終えておく（組み立ては両方の計時の外）．
// 流し直しは段ごとの計時と同じ投入の手順（ozaki1_stages.hpp）を，同期も段ごとの計時も挟まずに呼ぶ．発行順も段ごとの
// 計時と同じにする（違うと whole と段の和の差に発行順の差が混ざる）．
// 計時の前に，流し直しが書くはずの器を番兵で埋めてから全工程を 1 回だけ流し，その C を段ごとの計時の後の C とビット単位で
// 比べ，投入したプログラムの本数も構成から数えた本数と比べる（whole_check．手順は common/oz_whole.hpp．番兵が無いと，
// 流し直しが何もしなくても器に残った正しい C と一致してしまう）．計時の後の C も比べ，結果は WHOLE-CHECK 行（identical は
// すべて通ったときだけ 1．main が出す）．
// 数える単位はプログラムの投入 1 回：head はスケール算出・指数差分・スライス（A・B で OZ1_FILL_PROGRAMS 本），
// gemm は gemm_run，accum は accum_exec_range，recon は recon_exec（compose はこの経路に無い）．
// 投入しながら数えるので，記録した数（rec）は流した数（ran）そのもの．

#include "ozaki1/ozaki1_stages.hpp"
#include "common/host_eval.hpp"   // oz_eval::time_dispatch
#include "common/oz_whole.hpp"

// 立てたかどうかは main の冒頭で Oz1Diag::time_whole に読む（ozaki1_plan.hpp の oz1_diag_from_env）．

// 流し直しが「何もしない」「段を抜かす」「前の段の書き込みより先に読む」ことを捕まえる検査（計時の外）．流し直しが
// 書くはずの器（最大値・指数差分・スライス出力 shA/shB・対角の部分和 Pbig・累積面 S・limb 3 面・FP64 の出力 rcn.out）を
// 番兵で埋め，rcn.out に番兵が載ったことを読み戻して確かめてから，whole を 1 回だけ数えながら流して C を比べる．
// FP64 のオペランドと範囲外の旗（0 に初期化してから立てるだけ）は入力なので触らない．
// limb 3 面は K ブロックごとに Horner を回す構成（br.rcn.limbs）でだけ読み書きされる器なので，そのときだけ埋める
// （ほかの構成の hi・mid・lo はカーネルが触らない 1 タイルの器）．
// 番兵を埋めても正しい流し直しの答えは変わらない：どの器も，全工程の中で読まれる前に書かれる（Pbig・S・limb は
// その面に初めて触るとき ACC=0 で置き，pr[0] の再構成は limb を読まない）．
template <class Whole>
static void whole_check(const StageCtx& s, BandRecon& br, const AccumCtx* acc, Whole&& whole, oz_whole::Check& wck) {
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>> bufs = {
        s.in.maxAb, s.in.maxBb, s.in.dltAb, s.in.dltBb, s.in.shA, s.in.shB, s.Pbig, br.rcn.out };
    if (br.rcn.limbs)   // rcnT は rcn と limb・出力を共有する
        for (const auto& b : {br.rcn.hi, br.rcn.mid, br.rcn.lo}) bufs.push_back(b);
    if (acc && acc->S) bufs.push_back(acc->S);
    oz_whole::sentinel_run(s.md, bufs, [&](std::vector<uint32_t>& got) { recon_read_f64(s.md, br.rcn, got); }, whole,
                           [&] { tt::tt_metal::distributed::Finish(s.md->mesh_command_queue()); }, wck);
    wck.rec = wck.ran;
}

// スケール算出 → 指数差分 → スライスを同期せずに投入する（段ごとの計時の dscale と slice から Finish と内訳の計時を
// 除いたもの）
static void whole_fill(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, FillStage& fs, oz_whole::Check& wck) {
    auto note = [&](oz_whole::Kind k) { wck.note(k); };
    fill_issue_scale(md, fs, note);
    fill_issue_expdelta(md, fs, note);
    fill_issue_slice(md, fs, note, [] {});
}

// K ブロックが 1 つで，対ごとにも PAIRLOOP でも回さない構成（run_single_kblock の後に呼ぶ）．
// 全パネルの GEMM プログラムを組んでおき（計時の外），スケール → スライス → バンドごとにそのバンドの全パネルの全対角の
// GEMM → そのバンドの再構成，を同期せずに続けて流す．バンドの中の GEMM の発行順は段ごとの計時と同じ（パネルごとに
// 全対角．同じプログラムが対角の本数だけ続く）．違うのは，段の間の同期と段ごとのウォームアップが無いことと，段ごとの
// 計時の GEMM はパネルごとに反復して同期するのに対し全工程を 1 回ずつ流すことである（全工程の計時そのものも段「whole」
// として温めてから gi 回の平均を取る．oz_timing.hpp の time_dispatch）．
static void whole_single_kblock(const StageCtx& s, FillStage& fs, BandRecon& br, Times& t, oz_whole::Check& wck) {
    auto& md = s.md;
    recon_read_f64(md, br.rcn, wck.pre);
    std::vector<GemmCtx> all;
    all.reserve(s.pp.panels.size());
    for (const auto& bd : s.pp.bands)
        for (uint32_t pi = bd.p0; pi < bd.p1; pi++) all.push_back(build_panel(s, pi, bd));
    fmt::print("Whole   : 全工程の GEMM の発行順は段ごとの計時と同じ（パネルごとに全対角）\n");
    auto note = [&](oz_whole::Kind k) { wck.note(k); };
    auto whole = [&] {
        whole_fill(md, fs, wck);
        for (const auto& bd : s.pp.bands) {
            for (uint32_t pi = bd.p0; pi < bd.p1; pi++) issue_panel_diagonals(md, all[pi], s.c.L, note);
            ReconCtx& rr = band_recon_at(s, br, t, bd, s.Pbig);
            recon_exec(md, rr, 0u);
            note(oz_whole::RECON);
        }
    };
    // 本数の期待値：パネルごとに全対角（ndiag = L+1 本），バンドごとに再構成 1 本
    wck.exp = oz_whole::Counts{};
    wck.exp[oz_whole::HEAD] = OZ1_FILL_PROGRAMS;
    wck.exp[oz_whole::GEMM] = (uint64_t)s.pp.panels.size() * (uint64_t)s.c.nslice();
    wck.exp[oz_whole::RECON] = (uint64_t)s.pp.bands.size();
    whole_check(s, br, nullptr, whole, wck);
    t.whole = oz_eval::time_dispatch("whole", s.gw, s.gi, whole,
                                     [&] { tt::tt_metal::distributed::Finish(md->mesh_command_queue()); });
}

// K ブロックが複数・対ごと・PAIRLOOP の構成（run_sweep の後に呼ぶ）．スケール → スライス → バンドごとの掃き出し
// （GEMM・累積・再構成）を，段ごとの計時と同じ手順（Oz1Sweep）で同期せずに続けて流す．
static void whole_sweep(const Oz1Sweep& sw, FillStage& fs, BandRecon& br, Times& t, oz_whole::Check& wck) {
    const StageCtx& s = sw.s;
    auto& md = s.md;
    recon_read_f64(md, br.rcn, wck.pre);
    auto note = [&](oz_whole::Kind k) { wck.note(k); };
    const uint32_t nkb = s.ip.nkb, inner = sw.inner();
    auto whole = [&] {
        whole_fill(md, fs, wck);
        for (const auto& bd : s.pp.bands) {
            ReconCtx& rr = sw.recon(br, t, bd);
            bool first = true;
            for (uint32_t b = 0; b < nkb; ++b)
                for (uint32_t a = 0; a <= inner; ++a) {
                    sw.gemms(bd, b, a, first, note);
                    (void)sw.post(rr, b, a, first, note);
                    first = false;
                }
            if (s.ip.defer) sw.final(rr, note);
        }
    };
    // 本数の期待値（Oz1Sweep の発行の単位）：GEMM はパネルごと・K ブロックごとに ndiag 本（対ごとなら
    // ndiag(ndiag+1)/2 本），累積プログラムは別プログラムで累積する構成（対ごとかスクラッチで，fuse・PAIRLOOP でない．
    // IssuePlan::sep_accum）でだけバンドごと・K ブロックごとに ndiag 本（fuse と対ごとの組では累積が GEMM の出力経路で
    // 済み，累積プログラムは 0 本），再構成はバンドごとに遅延再構成なら 1 本，そうでなければ K ブロックごとに 1 本
    const uint64_t nd = (uint64_t)s.c.nslice(), nbd = (uint64_t)s.pp.bands.size();
    wck.exp = oz_whole::Counts{};
    wck.exp[oz_whole::HEAD] = OZ1_FILL_PROGRAMS;
    wck.exp[oz_whole::GEMM] = (uint64_t)s.pp.panels.size() * nkb * s.ip.gemm_issues(s.c.nslice());
    wck.exp[oz_whole::ACCUM] = s.ip.sep_accum() ? nbd * nkb * nd : 0u;
    wck.exp[oz_whole::RECON] = nbd * (s.ip.defer ? 1u : (uint64_t)nkb);
    whole_check(s, br, &sw.acc, whole, wck);
    t.whole = oz_eval::time_dispatch("whole", s.gw, s.gi, whole,
                                     [&] { tt::tt_metal::distributed::Finish(md->mesh_command_queue()); });
}
