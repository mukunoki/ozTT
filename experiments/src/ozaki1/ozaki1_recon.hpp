#pragma once
// 尾崎スキーム I：再構成段（デバイス上の triple-float の Horner と FP64 の組み立て）
// ---------------------------------------------------------------------------
// 再構成カーネルへ渡す定数 ReconScale，ReconProg / ReconCtx と recon_build_prog / recon_build / recon_exec /
// recon_read_f64 / recon_set_out_off，調査用の OZ_RECON_PROBE の説明を置く．Int32 の K 方向累積（OZ_NO_FUSE_ACC の経路）は
// ozaki1_accum.hpp．
// 依存：ozaki1_format.hpp（OZ_FMT・OZ1_INT32_PARTIAL）と，それが取り込む ozaki_gemm_host.hpp（make_cb・
// full_grid_span），カーネルの置き場所 oz_paths.hpp（OZ1_KDIR），カーネルの引数の並びと CB 番号
// （ozaki1/kernels/oz1_args.h）．GEMM 側（ozaki1_gemm.hpp）には依存しない．

#include "ozaki1/ozaki1_format.hpp"
#include "ozaki1/kernels/oz1_args.h"
#include "common/oz_paths.hpp"
#include "common/kernels/f64_tf.h"   // OZ_F64_TILE_BYTES

// OZ_RECON_PROBE=1..3【調査用．答えは壊れる】ライタの仕事を段階的に外して，
// 再構成段の固定費がどこにあるかを切り分ける（docs/recon_cost_breakdown.md）．
// 出力タイル 1 枚あたりの仕事は 3 つある：
//   (A) 行・列の指数差分タイル 2 枚の読み込み
//   (B) 1024 要素の assemble_tf_f64（3 本組を 64 bit へ集めて FP64 化．
//       scheme II の assemble_frac_f64 と違い，ここは 64 bit で足りる）
//   (C) FP64 タイル 8192 B の書き出し
//   1  (B)とlimb・指数差分のL1読出しを外す．(A)のDRAM読出しと(C)の書出し量は保つ．
//   2  さらに(A)を外す．書き出しは残る．
//   3  さらに(C)も外す．差は各処理を省いたときの短縮量で，並行する処理の単独の所要時間ではない．
// 書き出し量を変えずに (B) だけ外せるのが要点である．scheme II の
// OZ2_RECON_NOF64 は FP64 8 B/要素の代わりに limb 12 B/要素を書くので，
// 書き出し量が変わり，(B) を除いた時間の上界しか測れない．
// 値は main の冒頭で Oz1Diag::recon_probe に読み（ozaki1_plan.hpp の oz1_diag_from_env．範囲外・不正文字列は黙って 0 や
// 別の段階へ倒さず，読み取り（oz_knobs::u32．範囲はつまみの表）が止める），ReconScale::probe で渡す．

// 再構成カーネルへ渡す基数と最終スケールのビット列と，ライタの組み方（ホストの計算．ozaki1_plan.hpp の recon_scale が作る）
struct ReconScale {
    uint32_t radixbits = 0;          // 基数を float のビット列として
    uint32_t Whi = 0, Wlo = 0;       // radix^-(L+2) の 64 bit 仮数
    int32_t qw = 0;                  // その指数
    uint32_t Wdlo = 0, Wdhi = 0;     // soft-float 版が使う wscale の double ビット列
    uint32_t nsmall = 0;             // 型変換 1 回で FP32 へ移せる先頭の対角の本数（ozaki1_plan.hpp の recon_nsmall）
    uint32_t softfloat = 0;          // 1 なら FP64 の組み立てを soft-float 版で（OZ_SOFTFLOAT．ライタの RW_CT_SF）
    uint32_t probe = 0;              // 【調査用】OZ_RECON_PROBE（0..3．ライタの RW_CT_PROBE．上の注記）
};

// ===== デバイス上の再構成 =====
//
// Pbig には対角ごとの部分和が ndiag スロットぶん入っている（スロット d はタイル d*pstride から．pstride は
// 対角の面の間隔で，行バンドに切らなければ Mt*Nt，切るならバンド 1 本ぶんのタイル数 band_tiles）．
// これを Horner 法で Σ_d radix^(L-d) P_d の形に畳み，triple-float の 3 本（hi,mid,lo）としてデバイス上に残す．
//
// K ブロック化のため，同じ構成のプログラムを 3 つ組んでおく（遅延再構成なら pr[0] だけ）：
//   pr[0]  累積なし．出力を初期化する（K ブロック 0 で使う）
//   pr[1]  累積あり．いまの出力 (hi,mid,lo) を読んで足し込み，同じ場所へ書き戻す（中間ブロック）
//   pr[2]  累積あり．足し込んだうえで FP64 を組み立てて出力する（最終ブロック）
// Horner 法は P_d について線形なので，全体の総和 S = Σ_b Σ_d radix^(L-d) P_d^(b) は
// K ブロック b ごとに畳んでから足し合わせてよい．
//
// 累積は同じバッファに対する in-place だが安全である．コアごとの担当タイル範囲は
// 互いに重ならず，1 タイルの中では リーダ（読み）→ 計算 → ライタ（書き）の順序が
// CB で直列化されているので，あるタイルを書き換える前に必ずそれを読み終えている．
//
// カーネルは入力の部分和の形式で 2 組ある（どちらも ozaki1/kernels/）：
//   recon_reader_f32in.cpp・recon_tf_f32in.cpp  FP32 の P_d を 1 回ずつ足す非遅延の経路．遅延再構成を使わない構成
//     （K ブロック 1 本で対ごとにも PAIRLOOP でも回さない FP 形式と，K ブロックごとに Horner を回す構成）だけが通り，
//     既定の 16 経路の構成では通らない（長い K で遅延再構成が取れないときと OZ_NO_* のつまみだけ）
//   recon_reader_i32in.cpp・recon_tf_i32in.cpp  Int32 の部分和を無誤差な FP32 2 本へ分けて足す．INT8 経路（符号-絶対値）と
//     遅延再構成の累積（2 の補数）
struct ReconProg {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::KernelHandle wk = 0;   // ライタ（recon_set_out_off が出力の offset を差し替える）
    bool ok = false;
    // 行バンドごとに出力側の offset だけを差し替えられるよう，構築時のライタ引数を
    // コアごとに残しておく（対角 GEMM の gemm_run と同じやり方）．
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<std::vector<uint32_t>> wa;
};
struct ReconCtx {
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    // 0 = 初期化，1 = 中間ブロックの累積，2 = 最終ブロック（FP64 を組み立てる）
    ReconProg pr[3];
    // limb 3 面と FP64 の最終結果 out．limb を読み書きするのは K ブロックごとに Horner を回す構成
    // （with_acc && !defer）だけで，そのときだけ limbs が立ち，hi・mid・lo はバンド 1 本ぶんの別々の面になる．
    // ほかの構成ではカーネルの TensorAccessor の引数を作るためだけの 1 タイルの器を 3 つで共有する
    // （EMIT_LIMBS=0・ACC=0 のカーネルは触らない）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> hi, mid, lo, out;
    bool limbs = false;
    uint32_t ntiles = 0;       // 1 回の発行で畳むタイル数（バンド 1 本ぶん）
    uint32_t ntiles_out = 0;   // out の全体のタイル数（読み戻しに使う）
};

// 行バンドの出力 offset［タイル］を差し替える．バンドに切らない構成では呼ばない．
static void recon_set_out_off(ReconCtx& rc, uint32_t off) {
    for (auto& pr : rc.pr) {
        if (!pr.ok) continue;
        tt::tt_metal::Program& prog = pr.workload.get_programs().at(rc.range);
        for (size_t i = 0; i < pr.cores.size(); i++) {
            auto a = pr.wa[i];
            a[oz1_args::RW_OUT_OFF] = off;
            tt::tt_metal::SetRuntimeArgs(prog, pr.wk, pr.cores[i], a);
        }
    }
}

// 再構成のプログラム 1 つが読み書きする器
struct ReconBufs {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> P;            // 対角ごとの部分和（Pbig か累積面 S）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> hi, mid, lo;  // limb 3 面（ACC=1 なら入力にも使う）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out;          // FP64 の出力
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltA, dltB;   // 行・列の指数差分（スケール段が作る）
};
// 3 つのプログラム（ReconCtx の pr）の違い
struct ReconVariant {
    uint32_t acc = 0;          // 1 なら前ブロックまでの limb を読んで足す（fp32 入力の経路だけ）
    uint32_t emit_limbs = 0;   // limb 3 本を書き出す（K ブロックごとに Horner を回す構成の最終ブロック以外）
    uint32_t emit_f64 = 0;     // FP64 を組み立てる（最終ブロックだけ）
    bool defer = false;        // 入力が遅延再構成の Int32 累積（2 の補数）
    uint32_t nsmall = 0;       // defer のとき，型変換 1 回で読める先頭の対角の本数（符号-絶対値の入力では使わない）
};

// 再構成プログラムを 1 つ組む．Mt・Nt は 1 回の発行で畳む出力（バンド 1 本）の大きさ，pstride は対角の面の間隔［タイル］．
static ReconProg recon_build_prog(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                  const tt::tt_metal::distributed::MeshCoordinateRange& device_range,
                                  const ReconBufs& bf, uint32_t Mt, uint32_t Nt, uint32_t ndiag,
                                  const ReconScale& rs, const ReconVariant& v, uint32_t pstride) {
    namespace tm = tt::tt_metal;
    using tt::DataFormat;
    tm::distributed::MeshWorkload workload;
    tm::Program program{};

    const uint32_t ntiles = Mt * Nt;
    const uint32_t f32_tile = tt::tile_size(DataFormat::Float32);  // 4096
    // defer のときは入力が Int32 の累積 Sbig になるので，int8 経路と同じ
    // 「Int32 を無誤差な FP32 2 本へ分ける」読み方を使う．
    const bool i32_in = OZ1_INT32_PARTIAL || v.defer;
    // Int32 の表現は入力の出どころで違う．行列エンジンがパックしたもの（INT8 経路の
    // 部分和）は符号-絶対値，typecast<Float32,Int32> が作ったもの（遅延再構成の
    // 累積）は 2 の補数である．リーダのマスクと計算カーネルの符号の扱いは必ず
    // 同じ側でなければならないので，1 つの旗で両方を選ぶ．
    const uint32_t signmag = (i32_in && !v.defer) ? 1u : 0u;

    // 担当タイルの配り方は全段共通の full_grid_span（コアの並びは行優先で，先頭から per 枚ずつ）
    const GridSpan gs = full_grid_span(md, ntiles);
    const tm::CoreRange all_cores = gs.range();

    // CB 番号は oz1_args.h の表（カーネルも同じ名前で引く）．作る順は L1 の配置を決めるので変えない
    namespace oa = oz1_args;
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    make_cb(program, all_cores, cbi(oa::CB_RECON_IN), i32_in ? DataFormat::Int32 : OZ_FMT.out, ndiag * 2, f32_tile);
    if (i32_in)
        make_cb(program, all_cores, cbi(oa::CB_MASK), DataFormat::Int32, 2, f32_tile);   // Int32 を分割するためのマスクタイル
    make_cb(program, all_cores, cbi(oa::CB_LIMB_HI), DataFormat::Float32, 2, f32_tile);
    make_cb(program, all_cores, cbi(oa::CB_LIMB_LO), DataFormat::Float32, 2, f32_tile);
    make_cb(program, all_cores, cbi(oa::CB_LIMB_MID), DataFormat::Float32, 2, f32_tile);   // triple-float の中位 limb の出力
    make_cb(program, all_cores, cbi(oa::CB_F64), DataFormat::Float32, 1, OZ_F64_TILE_BYTES);   // ライタ作業：FP64 の出力タイル
    make_cb(program, all_cores, cbi(oa::CB_DLT_A), DataFormat::Int32, 1, f32_tile);   // ライタ作業：行の指数差分
    make_cb(program, all_cores, cbi(oa::CB_DLT_B), DataFormat::Int32, 1, f32_tile);   // ライタ作業：列の指数差分

    if (v.acc) {   // 前ブロックまでの累積（hi,mid,lo）を読むための CB
        make_cb(program, all_cores, cbi(oa::CB_ACC), DataFormat::Float32, 6, f32_tile);
    }
    std::vector<uint32_t> rargs_ct;
    // Int32 経路のリーダは先頭で SIGNMAG を受け取る（fp32 入力の経路は読まない）
    if (i32_in) { rargs_ct.resize(oa::RI_CT_COUNT); rargs_ct[oa::RI_CT_SIGNMAG] = signmag; }
    tm::TensorAccessorArgs(*bf.P).append_to(rargs_ct);
    if (i32_in) {
        // マスクはリーダが L1 で作るので TensorAccessor を渡さない
    } else {
        // 累積の 3 本組はリーダでは hi・mid・lo の順（ライタの hi・lo・mid と違う．oz1_args.h の冒頭）
        tm::TensorAccessorArgs(*(v.acc ? bf.hi : bf.P)).append_to(rargs_ct);
        tm::TensorAccessorArgs(*(v.acc ? bf.mid : bf.P)).append_to(rargs_ct);
        tm::TensorAccessorArgs(*(v.acc ? bf.lo : bf.P)).append_to(rargs_ct);
    }
    // ライタの TensorAccessorArgs は oz1_args.h の WriterTA の順（hi・lo・mid・out・dltA・dltB）．名前で並べてから詰める
    std::vector<uint32_t> wargs_ct;
    {
        std::shared_ptr<tm::distributed::MeshBuffer> ta[oa::TA_COUNT];
        ta[oa::TA_HI] = bf.hi; ta[oa::TA_LO] = bf.lo; ta[oa::TA_MID] = bf.mid;
        ta[oa::TA_OUT] = bf.out; ta[oa::TA_DLT_A] = bf.dltA; ta[oa::TA_DLT_B] = bf.dltB;
        for (const auto& b : ta) tm::TensorAccessorArgs(*b).append_to(wargs_ct);
    }

    auto rk = tm::CreateKernel(
        program, i32_in ? OZ1_KDIR "recon_reader_i32in.cpp" : OZ1_KDIR "recon_reader_f32in.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                               .noc = tm::NOC::RISCV_1_default, .compile_args = rargs_ct});
    auto wk = tm::CreateKernel(
        program, OZ1_KDIR "recon_f64_writer.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                               .noc = tm::NOC::RISCV_0_default,
                               // 末尾 4 つ（oz1_args.h の WriterCT）が limb を書くか／FP64 を組み立てるか／
                               // soft-float 版か／PROBE（OZ_RECON_PROBE）．
                               // FP64 の組み立てはコストがかかるので，
                               // K ブロックが複数あるときは最終ブロックだけで行う．
                               .compile_args = [&]{ auto a = wargs_ct;
                                   const size_t o = a.size(); a.resize(o + oa::RW_CT_COUNT);
                                   a[o + oa::RW_CT_EMIT_LIMBS] = v.emit_limbs; a[o + oa::RW_CT_EMIT_F64] = v.emit_f64;
                                   a[o + oa::RW_CT_SF] = rs.softfloat;
                                   a[o + oa::RW_CT_PROBE] = rs.probe;
                                   static_assert(oa::RW_CT_COUNT == 4, "WriterCT に欄を足したらここで詰める");
                                   return a; }()});
    // c_0 には符号付き FP32 の P_d が入るので，切り詰めずに DST へ展開させる．
    // 累積入力 c_2（前ブロックまでの triple-float）も同様に FP32 のまま展開する
    // 必要がある．既定のままだと展開時に切り詰められ，足し込んだ値が壊れる．
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[oa::CB_RECON_IN] = tm::UnpackToDestMode::UnpackToDestFp32;
    u2d[oa::CB_ACC] = tm::UnpackToDestMode::UnpackToDestFp32;
    if (i32_in) u2d[oa::CB_MASK] = tm::UnpackToDestMode::UnpackToDestFp32;
    std::vector<uint32_t> cargs_ct(oa::RC_COUNT);
    cargs_ct[oa::RC_NDIAG] = ndiag;
    cargs_ct[oa::RC_IN_FMT] = (uint32_t)(i32_in ? DataFormat::Int32 : OZ_FMT.out);
    cargs_ct[oa::RC_F32] = (uint32_t)DataFormat::Float32;
    cargs_ct[oa::RC_RADIX] = rs.radixbits; cargs_ct[oa::RC_ACC] = v.acc; cargs_ct[oa::RC_SIGNMAG] = signmag;
    cargs_ct[oa::RC_NSINGLE_CAST] = signmag ? 0u : std::min(v.nsmall, ndiag);
    static_assert(oa::RC_COUNT == 7, "ReconCT に欄を足したらここで詰める");
    auto ck = tm::CreateKernel(
        program, i32_in ? OZ1_KDIR "recon_tf_i32in.cpp" : OZ1_KDIR "recon_tf_f32in.cpp", all_cores,
        // dst_full_sync_en は必須：基数倍（tfMulR・tfMulP2）と復号が DST を 8 本（0..7）使う．既定の半分
        // 同期では FP32 の DST が 4 本ずつの 2 区画に分かれ，4..7 番が次の区画と重なる．
        // パッカーが前のタイルを読んでいるあいだに次のタイルの計算がそこを書き潰しうる
        // （dst_probe を 2 回以上繰り返すと 5 本目から壊れる）．
        tm::ComputeConfig{.fp32_dest_acc_en = true,
                          .dst_full_sync_en = true,
                          .unpack_to_dest_mode = u2d,
                          .compile_args = cargs_ct});

    std::vector<tm::CoreCoord> wcores;
    std::vector<std::vector<uint32_t>> wargs_rt;
    for (size_t c = 0; c < gs.cores.size(); ++c) {
        const tm::CoreCoord core = gs.cores[c];
        const uint32_t nt = gs.count(c), start = gs.begin(c);
        // Int32 入力のリーダは RR_SLOT_STRIDE までしか読まない（累積入力を持たない：
        // defer では最後に 1 回しか走らず，int8 経路は K ブロックが 1 個である）
        std::vector<uint32_t> ra(i32_in ? oa::RR_I32_COUNT : oa::RR_COUNT);
        ra[oa::RR_SRC_ADDR] = (uint32_t)bf.P->address(); ra[oa::RR_NUM] = nt; ra[oa::RR_START] = start;
        ra[oa::RR_NDIAG] = ndiag; ra[oa::RR_SLOT_STRIDE] = pstride;
        if (!i32_in) {
            ra[oa::RR_ACC_HI] = v.acc ? (uint32_t)bf.hi->address() : 0u;
            ra[oa::RR_ACC_MID] = v.acc ? (uint32_t)bf.mid->address() : 0u;
            ra[oa::RR_ACC_LO] = v.acc ? (uint32_t)bf.lo->address() : 0u;
        }
        tm::SetRuntimeArgs(program, rk, core, ra);
        std::vector<uint32_t> ca(oa::RCR_COUNT);
        ca[oa::RCR_NTILES] = nt;
        tm::SetRuntimeArgs(program, ck, core, ca);
        std::vector<uint32_t> wa(oa::RW_COUNT);
        wa[oa::RW_HI_ADDR] = (uint32_t)bf.hi->address(); wa[oa::RW_NUM] = nt; wa[oa::RW_START] = start;
        wa[oa::RW_LO_ADDR] = (uint32_t)bf.lo->address(); wa[oa::RW_MID_ADDR] = (uint32_t)bf.mid->address();
        wa[oa::RW_OUT_ADDR] = (uint32_t)bf.out->address();
        wa[oa::RW_DLT_A_ADDR] = (uint32_t)bf.dltA->address(); wa[oa::RW_DLT_B_ADDR] = (uint32_t)bf.dltB->address();
        wa[oa::RW_NT] = Nt; wa[oa::RW_WHI] = rs.Whi; wa[oa::RW_WLO] = rs.Wlo; wa[oa::RW_QW] = (uint32_t)rs.qw;
        wa[oa::RW_WDLO] = rs.Wdlo; wa[oa::RW_WDHI] = rs.Wdhi;
        wa[oa::RW_OUT_OFF] = 0u;   // 行バンドの出力 offset（recon_set_out_off が差し替える）
        static_assert(oa::RW_COUNT == 15 && oa::RR_COUNT == 8 && oa::RCR_COUNT == 1,
                      "WriterRT・ReaderRT・ReconRT に欄を足したらここで詰める");
        tm::SetRuntimeArgs(program, wk, core, wa);
        wargs_rt.push_back(std::move(wa));
        wcores.push_back(core);
    }
    workload.add_program(device_range, std::move(program));
    ReconProg pr;
    pr.workload = std::move(workload);
    pr.wk = wk; pr.ok = true;
    pr.cores = std::move(wcores); pr.wa = std::move(wargs_rt);
    return pr;
}

// 高さの違う 2 本目の再構成（端数バンド）が 1 本目と共有する器．1 本目は空で渡し，recon_build が確保する
struct ReconShared {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out, hi, mid, lo;
};

// 出力バッファと再構成プログラム（初期化・中間・最終の 3 つ）を用意し，初回の JIT
// ビルドまで済ませる．with_acc が false のとき（K ブロックが 1 個のとき）と
// defer のときは累積側（中間・最終）を組まず，limb 3 面も実サイズでは確保しない（ReconCtx の limbs）．
// Mt・Nt は 1 回の発行で畳む出力（バンド 1 本）の大きさ，pstride は対角の面の間隔［タイル］，ntiles_out は FP64 の
// 出力の全体のタイル数（バンドに切るとき out はバンド間で共有する全出力ぶんの器）．
static ReconCtx recon_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig,
                            uint32_t Mt, uint32_t Nt, uint32_t ndiag, const ReconScale& rs,
                            bool with_acc, bool defer,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& dltA,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& dltB,
                            uint32_t pstride, uint32_t ntiles_out, const ReconShared& share, double* build_s) {
    namespace dist = tt::tt_metal::distributed;
    auto rr_t0 = std::chrono::steady_clock::now();
    auto& cq = md->mesh_command_queue();
    ReconCtx rc;
    rc.range = dist::MeshCoordinateRange(md->shape());
    rc.ntiles = Mt * Nt;
    rc.ntiles_out = ntiles_out;
    const uint32_t f32_tile = tt::tile_size(tt::DataFormat::Float32);
    // limb 3 面を読み書きするのは K ブロックごとに Horner を回す構成だけ（pr[0] が EMIT_LIMBS=1，pr[1]・pr[2] が
    // ACC=1）．それ以外（K ブロック 1 個・遅延再構成）は EMIT_LIMBS=0・ACC=0 のカーネルしか組まないので，
    // ライタの TensorAccessor の引数に渡す 1 タイルの器だけを作り，3 つで共有する（14080^3 の 1 バンドなら
    // 未使用の実サイズ 3 面で 2.38 GB を DRAM から取らずに済む）．
    rc.limbs = with_acc && !defer;
    dist::ReplicatedBufferConfig bO{.size = (size_t)f32_tile * (rc.limbs ? std::max(pstride, rc.ntiles) : 1u)};
    dist::DeviceLocalBufferConfig dO{.page_size = f32_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    if (rc.limbs) {
        rc.hi  = share.hi  ? share.hi  : dist::MeshBuffer::create(bO, dO, md.get());
        rc.lo  = share.lo  ? share.lo  : dist::MeshBuffer::create(bO, dO, md.get());
        rc.mid = share.mid ? share.mid : dist::MeshBuffer::create(bO, dO, md.get());
    } else {
        rc.hi = share.hi ? share.hi : dist::MeshBuffer::create(bO, dO, md.get());
        rc.mid = share.mid ? share.mid : rc.hi;
        rc.lo = share.lo ? share.lo : rc.hi;
    }
    // FP64 の最終結果（1 要素 8 バイト，タイル順）．ホストは並べ替えるだけでよい．
    // バンドに切るときの out はバンド間で共有するので，1 本目が全出力ぶんを確保し，2 本目に渡す
    // （limb の hi/mid/lo はバンド 1 本ぶんでよい）．
    dist::ReplicatedBufferConfig bF{.size = (size_t)OZ_F64_TILE_BYTES * rc.ntiles_out};
    dist::DeviceLocalBufferConfig dFo{.page_size = OZ_F64_TILE_BYTES,
                                      .buffer_type = tt::tt_metal::BufferType::DRAM};
    rc.out = share.out ? share.out : dist::MeshBuffer::create(bF, dFo, md.get());
    // 分割用マスクはリーダがコアの L1 で作るので，DRAM にも置かず host からも送らない
    // （A・B を転送してから C が戻るまで通信しないという仕様．recon_reader_*.cpp）．
    // K ブロックが 1 つなら初期化パスがそのまま FP64 を出す．複数なら，途中の
    // ブロックは limb だけを残し，最終ブロックが FP64 を組み立てる．
    // defer では Int32 累積 Sbig を入力に，最後に 1 回だけ走らせる（ACC 側は組まない）．
    const ReconBufs bf{Pbig, rc.hi, rc.mid, rc.lo, rc.out, dltA, dltB};
    ReconVariant v0;
    v0.emit_limbs = rc.limbs ? 1u : 0u; v0.emit_f64 = rc.limbs ? 0u : 1u;
    v0.defer = defer; v0.nsmall = defer ? rs.nsmall : 0u;
    rc.pr[0] = recon_build_prog(md, rc.range, bf, Mt, Nt, ndiag, rs, v0, pstride);
    if (rc.limbs) {
        ReconVariant v1; v1.acc = 1u; v1.emit_limbs = 1u;   // 中間ブロック
        ReconVariant v2; v2.acc = 1u; v2.emit_f64 = 1u;     // 最終ブロック
        rc.pr[1] = recon_build_prog(md, rc.range, bf, Mt, Nt, ndiag, rs, v1, pstride);
        rc.pr[2] = recon_build_prog(md, rc.range, bf, Mt, Nt, ndiag, rs, v2, pstride);
    }
    // 初回はここでカーネルの JIT ビルドが走る．累積側が読む hi/mid/lo はこの時点で
    // 未初期化だが，本番の 0 番ブロックが pr[0]（初期化）で必ず上書きするので問題ない．
    dist::EnqueueMeshWorkload(cq, rc.pr[0].workload, false);
    if (rc.limbs) {   // defer と K ブロック 1 個の構成では pr[1],pr[2] を組んでいない
        dist::EnqueueMeshWorkload(cq, rc.pr[1].workload, false);
        dist::EnqueueMeshWorkload(cq, rc.pr[2].workload, false);
    }
    dist::Finish(cq);
    *build_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - rr_t0).count();
    return rc;
}

// K ブロック 1 個ぶんの再構成を発行する．which は 0 = 初期化，1 = 中間ブロックの
// 累積，2 = 最終ブロック（累積して FP64 を組み立てる）．
static void recon_exec(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ReconCtx& rc,
                       uint32_t which) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), rc.pr[which].workload, false);
}

// FP64 の最終結果をホストへ読み戻す（タイル順のまま）．組み立てとスケール倍は
// すべてデバイス側（recon_f64_writer.cpp）で済んでいるので，ホストは並べ替える
// だけでよい．read_s があれば読み戻しの時間［秒］を書く．
static void recon_read_f64(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ReconCtx& rc,
                           std::vector<uint32_t>& Ot, double* read_s = nullptr) {
    auto& cq = md->mesh_command_queue();
    Ot.resize((size_t)rc.ntiles_out * (OZ_F64_TILE_BYTES / 4u));
    auto rd0 = std::chrono::steady_clock::now();
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, Ot, rc.out, true);
    if (read_s) *read_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - rd0).count();
}
