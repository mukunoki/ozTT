#pragma once
// 尾崎スキーム I：Int32 の K 方向累積を別プログラムで行う経路（OZ_NO_FUSE_ACC 専用）
// ---------------------------------------------------------------------------
// AccumCtx と accum_alloc / accum_build / accum_exec_range を置く．
// 既定のつまみではどの形式・大きさでもここを通らない：FP 形式の遅延再構成は Int32 の累積を GEMM の中（PAIRLOOP の
// L1 の累積面か，融合 fuse の出力経路）で済ませ，INT8 は遅延再構成を使わない．ここを通るのは OZ_NO_FUSE_ACC を立てて
// 対ごとに発行する構成（pairwise）か，連結の K ブロックが複数で累積を別プログラムにする構成（scratch）だけである
// （ozaki1_plan.hpp の plan_issue）．アブレーション（diagnose/ablations.sh の OZ_NO_FUSE_ACC）と selftest の「I-BF16 nofuse」が使う．
// 依存：ozaki1_format.hpp（oz_dbg）と，それが取り込む ozaki_gemm_host.hpp（make_cb・full_grid_span），カーネルの置き場所
// oz_paths.hpp（OZ1_KDIR），カーネルの引数の並びと CB 番号（ozaki1/kernels/oz1_args.h）．

#include "ozaki1/ozaki1_format.hpp"
#include "ozaki1/kernels/oz1_args.h"
#include "common/oz_paths.hpp"
#include <string>

// --- Int32 での K 方向累積（遅延再構成）-------------------------------------
// K ブロックごとに triple-float の Horner を回す代わりに，対角ごとの部分和を
// Int32 のまま足しておき，再構成は最後に 1 回だけ行う．Horner は P_d について
// 線形なので，実数演算としては同じ値になる．有限精度では同じとは限らない：K ブロックごとに Horner を回すと
// triple-float の丸めが K ブロックの数だけ入るので，最下位のビットは変わる（3520^3 の I-BF16 で OZ_NO_DEFER
// （K ブロック 5 本）の chash は既定と違い，相対誤差は 4.72358e-17 と 4.72357e-17 で精度の基準は保つ）．
//
// 無誤差である根拠は accum_i32.cpp の冒頭に書いた．要点は，1 ブロック内の |P_ℓ| が
// 無誤差条件から 2^24 以下であり，全ブロックを足しても
//   |S_ℓ| <= nkb * (L+1) * K_c * d^2 = (L+1) * K * d^2   （d は桁の上限．oz_defer_fits）
// が 2^31 に収まること（BF16 経路の 14080^3，L+1=10・d=32 で 1.44e8，余裕約 15 倍）．
// したがって累積面 S は 1 要素 Int32 の 1 語で足り，面 1 枚は Pbig の面と同じ大きさである（幅を広げる必要は無い）．
// この経路（対ごとの発行・スクラッチ）は S の ndiag 面を Pbig とは別に確保するので，DRAM はその分だけ増える
// （ozaki1_plan.hpp の plan_issue が acc_need で見積もる）．
struct AccumCtx {
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    tt::tt_metal::distributed::MeshWorkload wl[2];      // 0 = 初期化，1 = 累積
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> S, P;
    tt::tt_metal::KernelHandle rk[2]{}, wk[2]{}, ck[2]{};
    bool ok = false;
};

// 累積の 3 本のカーネル（リーダ・計算・ライタ）の実行時引数を詰める（並びは oz1_args.h の AccumReaderRT・AccumRT・
// AccumWriterRT）．P 側 [ps, +nt) を S 側 [ss, +nt) へ足す（構築時・初回の JIT 実行・発行ごとの 3 か所が同じ形で詰める）
static void accum_set_rt(tt::tt_metal::Program& prog, tt::tt_metal::KernelHandle rk, tt::tt_metal::KernelHandle ck,
                         tt::tt_metal::KernelHandle wk, const tt::tt_metal::CoreCoord& core,
                         const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& P,
                         const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& S,
                         uint32_t nt, uint32_t ps, uint32_t ss) {
    namespace oa = oz1_args;
    std::vector<uint32_t> ra(oa::AR_COUNT), ca(oa::ACR_COUNT), wa(oa::AW_COUNT);
    ra[oa::AR_P_ADDR] = (uint32_t)P->address(); ra[oa::AR_NUM] = nt; ra[oa::AR_P_START] = ps;
    ra[oa::AR_S_ADDR] = (uint32_t)S->address(); ra[oa::AR_S_START] = ss;
    ca[oa::ACR_NTILES] = nt;
    wa[oa::AW_S_ADDR] = (uint32_t)S->address(); wa[oa::AW_NUM] = nt; wa[oa::AW_START] = ss;
    static_assert(oa::AR_COUNT == 5 && oa::ACR_COUNT == 1 && oa::AW_COUNT == 3, "累積の引数の表に欄を足したらここで詰める");
    tt::tt_metal::SetRuntimeArgs(prog, rk, core, ra);
    tt::tt_metal::SetRuntimeArgs(prog, ck, core, ca);
    tt::tt_metal::SetRuntimeArgs(prog, wk, core, wa);
}

// P 側 [p0, +ntiles) を S 側 [s0, +ntiles) へ足すように，全コアの実行時引数を詰める（担当の配り方は全段共通の
// full_grid_span）
static void accum_set_range(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, tt::tt_metal::Program& prog,
                            tt::tt_metal::KernelHandle rk, tt::tt_metal::KernelHandle ck, tt::tt_metal::KernelHandle wk,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& P,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& S,
                            uint32_t ntiles, uint32_t p0, uint32_t s0) {
    const GridSpan gs = full_grid_span(md, ntiles);
    for (size_t c = 0; c < gs.cores.size(); ++c)
        accum_set_rt(prog, rk, ck, wk, gs.cores[c], P, S, gs.count(c), p0 + gs.begin(c), s0 + gs.begin(c));
}

static tt::tt_metal::distributed::MeshWorkload accum_build_prog(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
        const tt::tt_metal::distributed::MeshCoordinateRange& device_range,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig,
        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Sbig,
        uint32_t ntiles, uint32_t ACC, AccumCtx& ctx) {
    namespace tm = tt::tt_metal;
    using tt::DataFormat;
    tm::distributed::MeshWorkload workload;
    tm::Program program{};
    const uint32_t f32_tile = tt::tile_size(DataFormat::Float32);
    const tm::CoreRange all_cores = full_grid_span(md, ntiles).range();

    // 深めの二重化にしておく：この段は DRAM 律速なので，読みと計算を重ねたい．CB 番号は oz1_args.h
    namespace oa = oz1_args;
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_ACCUM_P, DataFormat::Float32, 4, f32_tile);
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_ACCUM_S, DataFormat::Int32, 4, f32_tile);
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_ACCUM_OUT, DataFormat::Int32, 4, f32_tile);

    std::vector<uint32_t> rargs_ct(oa::AR_CT_COUNT);
    rargs_ct[oa::AR_CT_ACC] = ACC;
    tm::TensorAccessorArgs(*Pbig).append_to(rargs_ct);
    tm::TensorAccessorArgs(*Sbig).append_to(rargs_ct);
    std::vector<uint32_t> wargs_ct;
    tm::TensorAccessorArgs(*Sbig).append_to(wargs_ct);

    auto rk = tm::CreateKernel(program, OZ1_KDIR "accum_i32_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                               .noc = tm::NOC::RISCV_1_default, .compile_args = rargs_ct});
    auto wk = tm::CreateKernel(program, OZ1_KDIR "accum_i32_writer.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                               .noc = tm::NOC::RISCV_0_default, .compile_args = wargs_ct});
    // P_d は符号付き FP32 の厳密な整数なので，切り詰めずに DST へ展開させる．
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[oa::CB_ACCUM_P] = tm::UnpackToDestMode::UnpackToDestFp32;
    u2d[oa::CB_ACCUM_S] = tm::UnpackToDestMode::UnpackToDestFp32;
    std::vector<uint32_t> cargs_ct(oa::AC_COUNT);
    cargs_ct[oa::AC_ACC] = ACC; cargs_ct[oa::AC_F32] = (uint32_t)DataFormat::Float32;
    cargs_ct[oa::AC_I32] = (uint32_t)DataFormat::Int32;
    auto ck = tm::CreateKernel(program, OZ1_KDIR "accum_i32.cpp", all_cores,
        tm::ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = u2d,
                          .compile_args = cargs_ct});

    accum_set_range(md, program, rk, ck, wk, Pbig, Sbig, ntiles, 0u, 0u);
    ctx.rk[ACC] = rk; ctx.wk[ACC] = wk; ctx.ck[ACC] = ck;
    workload.add_program(device_range, std::move(program));
    return workload;
}

// 累積面 S（Int32．Mt×Nt の面を ndiag 枚．Mt はバンド 1 本の行数）を確保する．確保できなければ理由を why に入れて空を返す．
// 例外を受けるのはこの確保だけで，プログラムの構築・JIT の試走（accum_build）の失敗は理由を付けて呼び出し側へそのまま
// 上がる（どれも「確保できない」と報告すると元の例外の内容が消える）．確保の見積りは plan_issue が済ませてあるので，
// ここで失敗するのは見積りが外れたときだけである．
static std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> accum_alloc(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, uint32_t Mt, uint32_t Nt, uint32_t ndiag,
        std::string& why) {
    namespace dist = tt::tt_metal::distributed;
    const uint32_t i32_tile = tt::tile_size(tt::DataFormat::Int32);
    dist::ReplicatedBufferConfig bS{.size = (size_t)i32_tile * Mt * Nt * ndiag};
    dist::DeviceLocalBufferConfig dS{.page_size = i32_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    try {
        return dist::MeshBuffer::create(bS, dS, md.get());
    } catch (const std::exception& e) {
        why = e.what();
        return nullptr;
    }
}

// 累積のプログラム（初期化・累積の 2 つ）を S（accum_alloc が確保した面）の上に組み，初回の JIT ビルドまで済ませる．
// Pbig は ndiag 面でも 1 面のスクラッチでもよい（構築時に必要なのは形式と
// タイルサイズだけで，面数は実行時の先頭タイルで決まる）．ただし初回の JIT 実行は
// 構築時の実行時引数をそのまま使うので，Pbig が持つ面数 pbig_planes を渡して
// 読む範囲をそこへ収める．渡さないと，スクラッチ 1 面の構成で ndiag 面ぶんを
// 読みに行き，確保していない DRAM に触れる．
static AccumCtx accum_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& S,
                            uint32_t Mt, uint32_t Nt, uint32_t ndiag, uint32_t pbig_planes, double* build_s) {
    namespace dist = tt::tt_metal::distributed;
    auto t0 = std::chrono::steady_clock::now();
    AccumCtx ac;
    ac.range = dist::MeshCoordinateRange(md->shape());
    const uint32_t ntiles = Mt * Nt * ndiag;
    ac.S = S;
    ac.P = Pbig;
    ac.wl[0] = accum_build_prog(md, ac.range, Pbig, ac.S, ntiles, 0u, ac);
    ac.wl[1] = accum_build_prog(md, ac.range, Pbig, ac.S, ntiles, 1u, ac);
    auto& cq = md->mesh_command_queue();
    // 初回の JIT 実行が読む範囲を Pbig の実体に合わせる（担当の配り方は構築時と同じ full_grid_span で，枚数だけが違う）．
    {
        const uint32_t planes = std::min(pbig_planes, ndiag);
        tt::tt_metal::Program& prog = ac.wl[0].get_programs().at(ac.range);
        accum_set_range(md, prog, ac.rk[0], ac.ck[0], ac.wk[0], ac.P, ac.S, Mt * Nt * planes, 0u, 0u);
    }
    oz_dbg("accum prime enqueue");
    dist::EnqueueMeshWorkload(cq, ac.wl[0], false);   // 初回の JIT ビルド
    dist::Finish(cq);
    oz_dbg("accum prime done");
    ac.ok = true;
    *build_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return ac;
}

// P 側 [ptile0, +ntiles) を S 側 [stile0, +ntiles) へ足し込む．
// 用途は 2 つある．
//   ・対ごとに回すとき：ラウンド a では対角 a 以上しか更新されないので，
//     その範囲だけ流す（ptile0 == stile0）．
//   ・Pbig を 1 面のスクラッチにするとき：対角 d ごとに GEMM がスクラッチ面 0 を
//     書き，それを S の面 d へ足す（ptile0 = 0，stile0 = d*MtNt）．
// 累積が組まれていない（ac.ok が偽）のに呼ぶのは前提の破れ（累積を別プログラムで行う構成では sweep_setup が必ず組むか，
// 組めなければ止まる）．黙って返すと呼び出し側が投入の本数だけを数え，累積の抜けた C が出るので止める．
static void accum_exec_range(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, AccumCtx& ac,
                             uint32_t acc, uint32_t ptile0, uint32_t stile0, uint32_t ntiles) {
    TT_FATAL(ac.ok, "accum_exec_range: Int32 の累積のプログラムが組まれていない（sweep_setup の前提の破れ）");
    if (!ntiles) return;
    tt::tt_metal::Program& prog = ac.wl[acc].get_programs().at(ac.range);
    accum_set_range(md, prog, ac.rk[acc], ac.ck[acc], ac.wk[acc], ac.P, ac.S, ntiles, ptile0, stile0);
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), ac.wl[acc], false);
}
