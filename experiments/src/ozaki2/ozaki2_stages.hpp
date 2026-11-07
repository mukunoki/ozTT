// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎スキーム II：デバイスの段の資源と構築（CB・カーネル・実行時引数．同期も計時もしない）
// ---------------------------------------------------------------------------
// scheme I の ozaki1_stages.hpp に当たる．スライス段（build_slice），GEMM 段の部分行列積（gemm_build_partial），合成段
// （build_compose・compose_set_args），再構成段のパス（recon_build_pass）のプログラムを組む．投入・同期・計時と，OZ_TIME_WHOLE の
// 記録は ozaki2_main.cpp（scale_and_slice・run_partial_gemm・compose_run・run_recon_pass）が挟む．
// CB を作る順は L1 の配置を決め，確保（DRAM の器）・カーネル・セマフォを作る順は配置と番号を決めるので，どれも変えない．
// つまみの値（Oz2Knobs・Oz2Diag．ozaki2_plan.hpp）は引数で受け，環境変数を読まない．
// 依存：ozaki2_plan.hpp（OZ2_PARAM・Oz2Knobs・Oz2Diag・ReconPass・ReconScale・oz2_split_assembly・oz2_apply_nsmall），
// common/mcast_gemm.hpp（build_mcast_gemm・spill_gemm_spec．ozaki_gemm_host.hpp の make_cb・mesh_dram_buffer・full_grid_span も），
// common/slice_f64_host.hpp（両スキーム共通の FP64 リーダの引数），カーネルの引数の表 ozaki2/kernels/oz2_args.h・recon_args.h，
// common/oz_paths.hpp（カーネルの置き場所 OZ2_KDIR・OZ_COMMON_KDIR）．
// ヘッダの中では名前空間を開かない（関数の中で tm = tt::tt_metal・dist = tt::tt_metal::distributed の別名を置く）．
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include "ozaki2/ozaki2_plan.hpp"
#include "common/mcast_gemm.hpp"
#include "common/slice_f64_host.hpp"
#include "common/oz_bits.h"            // f32_bits（カーネルは実行時引数を uint32 で受け取り，ビット列のまま float に戻す）
#include "common/oz_paths.hpp"
#include "common/kernels/f64_tf.h"
#include "common/kernels/slice_f64_args.h"
#include "ozaki2/kernels/oz2_args.h"
#include "ozaki2/kernels/recon_args.h"

// ===== 第 1 段：モジュラースライス =====
// オペランド（R×C）を S 個の剰余行列へ分解する．法 s の剰余行列のタイル g は
// s*ntot+g の位置に置かれる．
//
// カーネルが行うのは，正規化 → 整数化 → 2^20 進 3〜5 桁への分解（oz2_digits.h）→
// 桁ごとの剰余の合成（modslice_int8.cpp / modslice_digit.cpp）である．入力は
// 正規化前の FP64 そのもので，リーダ（common/kernels/slice_f64_reader.cpp）が行ごと（A）／
// 列ごと（B）の指数差分を FP64 の指数へ整数で足して正規化し，triple-float に割って渡す．
// これで |A'| <= 2^b が入力の大きさに依らず成り立つ（桁の作り方は oz2_digits.h）．
//
// 実行時引数は構築時に確定するので，あとは同じワークロードを投入し直すだけでよい．
// 剰余行列はデバイスに置いたまま out で返す．GEMM 段はここからパネルをオフセットで
// 読むので，ホストへ降ろして切り出し直す必要はない．
struct Oz2Slice {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out;
    // 置き方の記録（GEMM 段が同じ前提で読むかを run_digit_gemms が確かめる）：出力の平面（桁行列）の枚数，
    // 連結の上限 maxg，スライスのライタの置き方（oz2_args.h の Layout：LAYOUT_PLANES = 平面ごと，LAYOUT_A_CAT = A を
    // K 方向に連結，LAYOUT_B_PERM = B を法の中で逆順）
    uint32_t planes = 0, layout = oz2_args::LAYOUT_PLANES;
    int maxg = 0;
};
// スライサが読む法の集合（全体か，法の組分けの組 [s0,s1)．make_slice_group）：法ごとの定数（法・逆数・桁の重み・
// 桁分割の基数と方式）と，法の本数・桁行列の枚数と並び・連結の上限．再構成が読む CRT の定数（y_i・逆数の断片・KM）は
// 持たない（組の部分集合では意味を持たないので，取り違えて使えないようにする）
struct Oz2SliceGroup {
    int s = 0, nplane = 0, maxg = 1;
    std::vector<int> m, q, mode, poff;     // poff は組の先頭からの相対
    std::vector<float> rinv, c0, c1, c2, c3;
};
// 法 [s0,s1) のスライサの入力．法を組に分けてスライスするとき（OZ2_MOD_GROUPS．plan_bands）は組ごとに，1 組なら [0,S) で作る．
// 法どうしは独立なので，部分集合を渡せば各法の桁行列は全体で切ったときと同じになる（[0,S) は全体と同じ値）
static Oz2SliceGroup make_slice_group(const Ozaki2Moduli& mo, uint32_t s0, uint32_t s1) {
    TT_FATAL(s0 < s1 && s1 <= (uint32_t)mo.s, "make_slice_group: 法の範囲 [{},{}) が本数 {} の外", s0, s1, mo.s);
    Oz2SliceGroup g;
    // 法ごとの配列はどれも法の本数ぶん埋まっている（ozaki2_choose_moduli_w）
    auto cut = [&](const auto& v) {
        TT_FATAL(v.size() == (size_t)mo.s, "make_slice_group: 法ごとの配列の長さ {} が本数 {} と違う", v.size(), mo.s);
        return std::decay_t<decltype(v)>(v.begin() + s0, v.begin() + s1); };
    g.m = cut(mo.m); g.q = cut(mo.q); g.mode = cut(mo.mode); g.rinv = cut(mo.rinv);
    g.c0 = cut(mo.c0); g.c1 = cut(mo.c1); g.c2 = cut(mo.c2); g.c3 = cut(mo.c3);
    g.s = (int)(s1 - s0);
    for (uint32_t i = s0; i < s1; i++) g.poff.push_back(mo.poff[i] - mo.poff[s0]);
    g.nplane = (int)group_planes(mo, s0, s1);
    // 部分集合でも保たれるべき前提（スライサは組の maxg で置き方を決め，GEMM は全体の maxg と
    // 組の枚数 PA で読む．run_digit_gemms が同じことをスライサの記録と突き合わせる）：
    //   maxg は全体と同じ（組の法だけで数え直すと連結の置き方が変わりうる）
    //   poff[0]=0，poff は法ごとの桁行列の枚数ずつ増え，最後の法の後ろが nplane
    g.maxg = mo.maxg;
    TT_FATAL(!g.poff.empty() && g.poff[0] == 0, "make_slice_group: 組の先頭の桁行列が 0 から始まらない");
    for (uint32_t i = 0; i < (uint32_t)g.s; i++) {
        const int npl = OZ2_DIGIT ? oz2_digit_planes(g.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd()) : 1;
        const int next = (i + 1 < (uint32_t)g.s) ? g.poff[i + 1] : g.nplane;
        TT_FATAL(next - g.poff[i] == npl, "make_slice_group: 法 {} の桁行列が {} 枚（{} 枚のはず）", g.m[i], next - g.poff[i], npl);
    }
    return g;
}

// スライス段の CB（作る順は L1 の配置を決めるので変えない）．
// c_0/c_4/c_1 がオペランドの triple-float（上位・中位・下位．中位・下位は生のビット列），
// c_2 が下位 limb のスケール s（slice_tf_core.h の tf_load_f64），
// c_16 が剰余の出力．c_24/c_25 は桁を作る途中の残差，c_26/c_27 は桁そのもの
// （並びは oz2_digits.h と common/kernels/slice_f64_reader.cpp を参照）．
// CB 番号は表から引く（入力の 4 枚・残差・リーダの作業領域は両スキーム共通の common/kernels/slice_f64_args.h，
// 桁と出力は ozaki2/kernels/oz2_args.h）．
static void slice_make_cbs(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores, uint32_t ND,
                           uint32_t slice_tile) {
    namespace sa = slice_f64_args; namespace oa = oz2_args;
    const uint32_t f32 = tt::tile_size(tt::DataFormat::Float32);
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    auto cb_f32 = [&](uint32_t idx, uint32_t depth) { make_cb(program, all_cores, cbi(idx), tt::DataFormat::Float32, depth, f32); };
    cb_f32(sa::CB_HI, 4); cb_f32(sa::CB_CRAW, 4); cb_f32(sa::CB_SCALE, 4); cb_f32(sa::CB_BRAW, 4);
    cb_f32(sa::CB_RES01, 8); cb_f32(sa::CB_RES2, 4);   // 残差（2 + 1 に分ける）
    // 桁の出力．CB_DIGIT_HI へは 1 枚ずつ押すが，1 入力タイルぶんの ND-1 枚を一組として待ち，添字で読み，まとめて捨てる
    // （oz2_digits.h の oz2_make_digits・oz2_digits_copy・oz2_digits_pop）ので，深さは
    // ND-1 の倍数でなければならない（割り切れないとリングの境界をまたいだ組が壊れる）．
    cb_f32(oa::CB_DIGIT_HI, 2 * (ND - 1)); cb_f32(oa::CB_DIGIT_LO, 2);   // 上位の桁と d0
    // FP64 リーダの作業領域：FP64 タイル 1 枚と指数差分（Int32）
    make_cb(program, all_cores, cbi(sa::CB_F64), tt::DataFormat::Float32, 1, OZ_F64_TILE_BYTES);
    make_cb(program, all_cores, cbi(sa::CB_DLT), tt::DataFormat::Int32, 1, f32);
    make_cb(program, all_cores, cbi(oa::CB_PLANE_OUT), OZ_FMT.slice_out, 16, slice_tile);
}

// スライス段の 3 本のカーネル（リーダ・ライタ・スライサ）と，ライタの置き方
struct Oz2SliceKernels {
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    uint32_t layout = oz2_args::LAYOUT_PLANES;
    std::vector<uint32_t> perm;   // LAYOUT_B_PERM の置き先（桁行列 i を平面 perm[i] へ）
};
static Oz2SliceKernels slice_make_kernels(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                                          const Oz2SliceGroup& mo, uint32_t ND, uint32_t Sout,
                                          const tt::tt_metal::distributed::MeshBuffer& f64B,
                                          const tt::tt_metal::distributed::MeshBuffer& dltB,
                                          const tt::tt_metal::distributed::MeshBuffer& outB, uint32_t mode) {
    namespace tm = tt::tt_metal;
    namespace sa = slice_f64_args; namespace oa = oz2_args;
    const uint32_t S = mo.s;
    Oz2SliceKernels k;
    // 桁分割の有無でスライサのカーネルを切り替える
    const char* SLICE_KERNEL = OZ2_DIGIT ? OZ2_KDIR "modslice_digit.cpp" : OZ2_KDIR "modslice_int8.cpp";
    // リーダは scheme I と共通：FP64 のオペランドをそのまま読み，整数のビット操作で
    // triple-float に割って渡す（正規化は指数差分として畳み込まれる）．soft-float 版は使わず，
    // 割る処理も全部リーダが行う（スライサの演算で決まる段なので分けない．SPLIT=0）．
    const std::vector<uint32_t> ra = f64_reader_ct(f64B, dltB, /*sf=*/false, /*split=*/false);
    // 置き方（oz2_args.h の Layout．ozaki2/kernels/modslice_writer.cpp）．桁分割経路は同じ位の桁積を K 方向に連結して
    // 1 本の GEMM にまとめる（ozaki2_split.h の oz2_product_groups）ために，A の桁行列を K 方向に並べ
    // （LAYOUT_A_CAT），B の桁行列を法の中で逆順に積む（LAYOUT_B_PERM）．INT8 経路は法ごとの剰余行列をそのまま並べる
    // （LAYOUT_PLANES）．まとめる桁積が無い構成（maxg=1．TF32 の 14080^3 など）は置き方を変えない．
    k.layout = (OZ2_DIGIT && mo.maxg > 1) ? ((mode == 0) ? oa::LAYOUT_A_CAT : oa::LAYOUT_B_PERM) : oa::LAYOUT_PLANES;
    std::vector<uint32_t> wa;
    tm::TensorAccessorArgs(outB).append_to(wa);
    { const size_t o = wa.size(); wa.resize(o + oa::SW_CT_COUNT); wa[o + oa::SW_CT_LAYOUT] = k.layout; }
    // B の並べ替え：法 s の桁行列 j（poff[s]+j）を位置 poff[s]+(np-1-j) へ
    k.perm.resize(Sout);
    for (uint32_t i = 0; i < Sout; i++) k.perm[i] = i;
    if constexpr (OZ2_DIGIT)
        for (uint32_t i = 0; i < S; i++) {
            const uint32_t p0 = (uint32_t)mo.poff[i], p1 = (i + 1 < S) ? (uint32_t)mo.poff[i + 1] : Sout;
            for (uint32_t j = p0; j < p1; j++) k.perm[j] = p1 - 1 - (j - p0);
        }
    k.rk = tm::CreateKernel(program, OZ_COMMON_KDIR "slice_f64_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1, .noc = tm::NOC::RISCV_1_default, .compile_args = ra});
    k.wk = tm::CreateKernel(program, OZ2_KDIR "modslice_writer.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = wa});
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    for (uint32_t idx : {sa::CB_HI, sa::CB_CRAW, sa::CB_SCALE, sa::CB_BRAW,
                         sa::CB_RES01, sa::CB_RES2, oa::CB_DIGIT_HI, oa::CB_DIGIT_LO})
        u2d[idx] = tm::UnpackToDestMode::UnpackToDestFp32;
    // dst_full_sync_en は必須：桁を作る基数倍（tfMulP2）が DST を 8 本すべて使う．
    // コンパイル時引数（法の本数 S・A' の桁数 ND・剰余の桁数の上限・binary16 で書くか）が変われば JIT が組み直す
    std::vector<uint32_t> cct(oa::SC_COUNT);
    cct[oa::SC_S] = S; cct[oa::SC_RDX] = f32_bits(std::ldexp(1.0f, OZ2_DIGIT_BITS)); cct[oa::SC_NDIGIT] = ND;
    cct[oa::SC_NRDIG] = (uint32_t)OZ2_PARAM.nrdig; cct[oa::SC_OUT_HALF] = (uint32_t)OZ_FMT.out_half;
    static_assert(oa::SC_COUNT == 5, "SlicerCT に欄を足したらここで詰める");
    k.ck = tm::CreateKernel(program, SLICE_KERNEL, all_cores,
        tm::ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = cct});
    return k;
}

// スライス段の実行時引数（コアごと）．法ごとの定数（法・逆数・桁の重み・桁分割の基数と方式）は実行時引数で渡す（並びは
// ozaki2/kernels/oz2_args.h の sl_*）．本数と桁数が同じなら，法の集合を差し替えても JIT は組み直さない
// in_off は入力の FP64 の先頭タイル（帯の A の行の先頭．リーダはタイル番号 beg+in_off から読み，指数差分も同じ番号から
// 引くので帯の行の指数差分になる．ライタは帯の中の番号 beg で書く）
static void slice_set_args(tt::tt_metal::Program& program, const GridSpan& gs, const Oz2SliceKernels& k,
                           const Oz2SliceGroup& mo, uint32_t ND, uint32_t Sout, uint32_t ntot, uint32_t Ct,
                           const tt::tt_metal::distributed::MeshBuffer& f64B,
                           const tt::tt_metal::distributed::MeshBuffer& dltB,
                           const tt::tt_metal::distributed::MeshBuffer& outB, uint32_t mode, uint32_t in_off) {
    namespace tm = tt::tt_metal;
    namespace oa = oz2_args;
    const uint32_t S = mo.s;
    for (size_t c = 0; c < gs.cores.size(); c++) { uint32_t nt = gs.count(c), beg = gs.begin(c);
        std::vector<uint32_t> ca(oa::sl_count(S, ND, OZ2_DIGIT), 0u);
        ca[oa::sl_ntiles()] = nt;
        for (uint32_t i = 0; i < S; i++) {
            ca[oa::sl_mod(S, i)] = f32_bits((float)mo.m[i]);
            ca[oa::sl_rinv(S, i)] = f32_bits(mo.rinv[i]);
            // A' の桁 d_{j+1} の重み 2^(20(j+1)) mod m（j = 0..ND-2．c2 は 4 桁以上，c3 は 5 桁のときだけ使う）
            const float wt[OZ2_MAXDIGIT - 1] = {mo.c0[i], mo.c1[i], mo.c2[i], mo.c3[i]};
            for (uint32_t j = 0; j + 1 < ND; j++) ca[oa::sl_weight(S, j, i)] = f32_bits(wt[j]);
            if constexpr (OZ2_DIGIT) {
                // 桁分割の基数と，桁の作り方の番号（カーネルと共有の Oz2XMode．ozaki2/kernels/oz2_limits.h．float の値で渡す．
                // 理由は oz2_limits.h）．
                // 方式 (A) と (C) はスライサから見ると同じ手続きである（違うのはホスト側の合成式だけ）．(E) は (A) の桁を
                // e4m3 の無誤差整数へ手直しする（FP8 だけ）．平衡 n 桁は桁数 n（BFP2_b は法ごとに最小の n，BFP4_b は 225 を
                // 超える法を 3 桁で）を続けて渡し，1 桁（桁 32 の AWE 経路の m <= 65 の法）は剰余をそのまま桁行列 1 枚にする
                ca[oa::sl_q(S, ND, i)] = f32_bits((float)mo.q[i]);
                ca[oa::sl_qinv(S, ND, i)] = f32_bits(1.0f / (float)mo.q[i]);
                const Oz2XMode xm = oz2_xmode(mo.mode[i]);
                TT_FATAL(xm < OZ2_XMODE_COUNT, "build_slice: 法 {} の分け方 {} にスライサの桁の作り方が無い", mo.m[i], mo.mode[i]);
                ca[oa::sl_xmode(S, ND, i)] = f32_bits((float)xm);
                ca[oa::sl_ndig(S, ND, i)] = (uint32_t)oz2_bal_digits(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd());
            }
        }
        // 指数差分の引き当て：A（mode 0）は行ごとなのでタイル番号を C 方向のタイル数 Ct で
        // 割り，B（mode 1）は列ごとなので同じ Ct で割った余りを使う．
        tm::SetRuntimeArgs(program, k.rk, gs.cores[c], f64_reader_rt((uint32_t)f64B.address(), (uint32_t)dltB.address(),
            nt, beg + in_off, Ct, mode));
        tm::SetRuntimeArgs(program, k.ck, gs.cores[c], ca);
        std::vector<uint32_t> wr(oa::sw_count(k.layout, Sout));
        wr[oa::SW_DST_ADDR] = (uint32_t)outB.address(); wr[oa::SW_NUM] = nt; wr[oa::SW_START] = beg;
        wr[oa::SW_PLANES] = Sout; wr[oa::SW_NTOT] = ntot;
        if (k.layout == oa::LAYOUT_A_CAT) wr[oa::SW_A_KTILES] = Ct;
        if (k.layout == oa::LAYOUT_B_PERM) for (uint32_t i = 0; i < Sout; i++) wr[oa::sw_perm(i)] = k.perm[i];
        tm::SetRuntimeArgs(program, k.wk, gs.cores[c], wr); }
}

// スライス段を組む．ND = A' を運ぶ 2^20 進の桁数（3〜5．b から計画が決める）．mode は 0 = A（行ごと），1 = B（列ごと）．
// 確保：出力の剰余行列・桁行列の器（out）．shared を渡すとそれへ書く（出力を行の帯に分けるとき全帯の A のスライサが，
// OZ_TIME_WHOLE で法を組に分けるとき全組のスライサが，1 本の器を使い回す．slice_band・slice_group．タイル長が同じで，
// この帯・組の平面が入る大きさのときだけ受ける）．in_off は入力の FP64 の先頭タイル（帯の A は帯の先頭の行．R 行を読む）．
// 投入はしない（scale_and_slice・slice_band・slice_group が投入して計時する）
static Oz2Slice build_slice(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                            const Oz2SliceGroup& mo, uint32_t ND, uint32_t R, uint32_t C,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> f64B,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltB,
                            uint32_t mode,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> shared = nullptr, uint32_t in_off = 0) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const uint32_t S = mo.s; const uint32_t Rt = R / 32, Ct = C / 32, ntot = Rt * Ct;
    // スライスを運ぶ形式のタイル長．INT8／FP8 は 1024 B だが BFP8_b は共有指数のぶん
    // 1088 B，BF16・FP16 は 2048 B，FP32 は 4096 B，BFP4_b・BFP2_b は 576 B・320 B あるので，
    // 形式から引かないと DRAM の器も CB も足りない．
    const uint32_t slice_tile = tt::tile_size(OZ_FMT.slice_out);
    // 書く形式（OZ_FMT.slice_out）と GEMM が読む形式（OZ_FMT.in）は INT8（UInt8／Int8）と
    // FP16（UInt16／Float16）で違う．同じバイト列を読み替えるので，タイル長は一致していなければならない．
    if (slice_tile != tt::tile_size(OZ_FMT.in)) { fprintf(stderr, "slice format and GEMM input format differ in tile size\n"); std::exit(2); }
    // 出力する行列の枚数．桁分割では法 1 本につき 3 枚（Karatsuba），2 枚（平方数・WIDE・疎な桁の平方数），
    // 1 枚（1 桁の法），n 枚（平衡 n 桁．BFP2_b は法ごとに 5 以下，BFP4_b の 225 を超える法は 3）で，
    // 内訳はホストが数えた mo.nplane と mo.poff に従う．INT8 経路は法 1 本につき剰余行列 1 枚．
    const uint32_t Sout = OZ2_DIGIT ? (uint32_t)mo.nplane : S;
    // FP16 経路のスライサは桁を binary16 のビット列へ直して書くが，平衡 n 桁の枝（XMODE=3）は直さずに書く
    // （ozaki2/kernels/modslice_digit.cpp）．桁 32 の経路に平衡 n 桁は当たらない（oz2_split_mode）が，規則が変わると
    // FP32 の値のまま UInt16 の CB へ書かれて黙って壊れるので，ここで止める．
    if (OZ2_DIGIT && OZ_FMT.out_half)
        for (uint32_t i = 0; i < S; i++)
            TT_FATAL(mo.mode[i] != OZ2_SPLIT_BAL, "build_slice: binary16 で書く経路に平衡 n 桁の法 {} がある", mo.m[i]);
    // 法の集合を段の時間のモデルで選ぶ経路（awe．BF16・FP16・BFP8_b・BFP4_b・BFP2_b）は DRAM の上端から取る（upload_operands の注記）
    auto outB = shared ? shared : mesh_dram_buffer(md, Sout * ntot, slice_tile, /*top_down=*/OZ2_PARAM.awe != 0);
    TT_FATAL(outB->page_size() == slice_tile && outB->num_pages() >= (uint64_t)Sout * ntot,
             "build_slice: 渡された器（ページ {} B・{} 枚）にこの組の平面 {} 枚（タイル {} B・{} 枚）が入らない",
             outB->page_size(), outB->num_pages(), Sout, slice_tile, (uint64_t)Sout * ntot);
    // スライス段はタイル単位の要素処理なので，コア格子の「形」は性能をほとんど左右しない．
    // タイルをコアへ均等に配るだけでよい．
    //
    // ただし性能はコアの「数」にほぼ比例するので，計算グリッド全体（11x10）を使う．
    // L1 も CB もコア数を絞る理由にならない．
    auto gs = full_grid_span(md, ntot); tm::CoreRange all_cores = gs.range();
    dist::MeshWorkload workload; auto range = dist::MeshCoordinateRange(md->shape()); tm::Program program{};
    slice_make_cbs(program, all_cores, ND, slice_tile);
    const Oz2SliceKernels k = slice_make_kernels(program, all_cores, mo, ND, Sout, *f64B, *dltB, *outB, mode);
    slice_set_args(program, gs, k, mo, ND, Sout, ntot, Ct, *f64B, *dltB, *outB, mode, in_off);
    workload.add_program(range, std::move(program));
    Oz2Slice s; s.workload = std::move(workload); s.out = outB;
    s.planes = Sout; s.layout = k.layout; s.maxg = mo.maxg;
    return s;
}

// ===== 第 2 段：部分行列積（1 本・出力パネル 1 枚）=====
// 部分行列積を 1 本組む．K は分割せず全体を 1 回で回す．
// 運ぶものは経路で違う（剰余行列か桁行列か）が，発行のしかたは 8 形式とも同じで，
// この 1 つの関数がすべてを受け持つ．BFP2_b の桁積もここを通るので，名前に INT8 を含めない
// （素の INT8 行列積のベンチマーク src/baseline/gemm_int8.cpp とも紛らわしい）．
//
// 出力パネル 1 枚（M×N）を計算し，呼び出し側が持つ永続バッファ Cbuf へ直接書く．
// Ntg はそのバッファ全体の列のタイル数，dW は書き出し位置（法または桁積の平面と
// パネルの原点）である．パネルを 1 枚にすれば行列全体の素の GEMM になる．パネル
// 分割はコアあたりの出力ブロック（したがって L1 使用量）を有界に保つためのもので，
// これがないと大きい立方サイズでコアあたり 1.5 MiB の L1 を溢れる．出力が
// デバイスに残るので，合成段も再構成段もそのまま読める（書き出しの散らし方は
// scheme I の gemm_run と同じで，行ストライドに全体の Ntg を使い，パネルの原点を
// dW に載せる）．
// KsA は A の行ストライド［タイル］（0 なら K/32．桁分割経路では桁行列を K 方向に連結した
// 行列の幅で，K はそのうち連結した桁積 g 本ぶんの長さ g*Kt．ozaki2_split.h の oz2_product_groups）．
// 投入も計時もしない（ozaki2_main.cpp の run_partial_gemm が投入して計時する）．
static tt::tt_metal::distributed::MeshWorkload gemm_build_partial(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const OptConfig& g,
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Abuf,
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Bbuf,
        uint32_t dA, uint32_t dB, uint32_t M, uint32_t K,
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Cbuf,
        uint32_t Ntg, uint32_t dW, uint32_t KsA, const Oz2Knobs& kn, const GemmProbe& probe) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    dist::MeshWorkload workload;
    auto device_range = dist::MeshCoordinateRange(md->shape()); tm::Program program{};
    const uint32_t Mt = M / tt::constants::TILE_HEIGHT, Kt = K / tt::constants::TILE_WIDTH;   // N のタイル数はパネル側の Ntg を使う
    const uint32_t KtS = KsA ? KsA : Kt;               // A の行ストライド（プログラムはこの幅で組む）
    TT_FATAL(KtS >= Kt, "run_partial_gemm: A の行ストライド {} が K {} より短い", KtS, Kt);
    const uint32_t num_blocks = Kt / g.bw;
    // プログラムの組み立ては scheme I・素の GEMM と共有する（common/mcast_gemm.hpp の
    // spill_gemm_spec と build_mcast_gemm）．出力経路への Int32 累積の融合と対ループ
    // （尾崎 I の遅延再構成専用．計算カーネルの CT_FUSE_ACC・CT_PAIRLOOP．gemm_compute_args.h）はここでは使わない．
    // 桁分割経路の入力は FP32 に累算される（INT8 経路の符号-絶対値 int32 ではない）ので，
    // 桁分割の 7 形式では無誤差整数を保持した Float32 として出力を読む（OZ_FMT.out）．
    // c_16（出力）と c_24（K 方向の部分和）は同じ L1 領域を共有させる．
    // 別々に確保すると同じ予算で取れる bw が半分以下になる：11x10・INT32 出力では
    // bw=8（K ブロック 55 個）が bw=20（同 22 個）になる．
    // リーダは A を Mt（このパネルの行数）× Kt，B を Kt × Ntg（全体の列数）として読み，
    // ライタは行ストライドに全体の Ntg を使ってパネルの原点 dW から書く．
    // K 方向の部分和はパッカが c_24 上で足し込み，最後のブロックで 1 回だけ読み戻す
    // （詳細は common/kernels/bmm_int32_spill.cpp）．
    SpillOpts so;
    so.in_fmt = OZ_FMT.in; so.acc_fmt = OZ_FMT.out; so.fidelity = kn.fid;
    so.blockmm = kn.blockmm;   // matmul_block（OZ_NO_BLOCKMM で切れる）
    McastGemmSpec spec = spill_gemm_spec(g, Kt, so);
    spill_check_run(num_blocks, false, 0u);
    spec.compute_rt = spill_compute_rt(num_blocks);   // 共通の行列積カーネルは num_blocks を実行時引数 CR_NUM_BLOCKS で受け取る
    spec.dA = dA; spec.dB = dB; spec.dW = dW;
    spec.probe = probe;   // OZ_GEMM_PROBE_* を立てると CONFIG 行が probe=set になる（print_config）
    McastGemm mg = build_mcast_gemm(program, md, g, Mt, Ntg, KtS, Abuf, Bbuf, Cbuf, spec);
    // K ブロック数はリーダの RA_NUM_BLOCKS（A 側・B 側のカーネルは同じ ra を受け取る．McastGemm::set_args）と
    // 計算カーネルの実行時引数 CR_NUM_BLOCKS が同じでなければならない（食い違うと少ない側が終わった後も多い側が
    // CB を待って止まる）．読む範囲が A・B のバッファに収まることも確かめる（A は行ストライド KtS の
    // Mt 行・K 方向 Kt 枚，B は Kt 行・行ストライド Ntg）．
    {
        namespace ma = mcast_args;
        namespace ga = gemm_args;
        TT_FATAL(num_blocks * g.bw == Kt && spec.compute_rt.size() > ga::CR_NUM_BLOCKS && spec.compute_rt[ga::CR_NUM_BLOCKS] == num_blocks,
                 "run_partial_gemm: K ブロック数 {}（bw={}，Kt={}）と計算カーネルの {} が合わない",
                 num_blocks, g.bw, Kt, spec.compute_rt.size() > ga::CR_NUM_BLOCKS ? spec.compute_rt[ga::CR_NUM_BLOCKS] : 0u);
        const uint64_t ta = Abuf->size() / tt::tile_size(OZ_FMT.in), tb = Bbuf->size() / tt::tile_size(OZ_FMT.in);
        TT_FATAL((uint64_t)dA + (uint64_t)(Mt - 1) * KtS + Kt <= ta, "run_partial_gemm: A の読む範囲がバッファ（{} 枚）の外", ta);
        TT_FATAL(((uint64_t)dB / Ntg + Kt) * Ntg <= tb, "run_partial_gemm: B の読む行 {}..{} がバッファ（{} 行）の外",
                 dB / Ntg, dB / Ntg + Kt, tb / Ntg);
        for (const auto& e : mg.cores)
            TT_FATAL(e.ra[ma::RA_NUM_BLOCKS] == KtS / g.bw, "run_partial_gemm: 組んだリーダのブロック数が {}（{} のはず）",
                     e.ra[ma::RA_NUM_BLOCKS], KtS / g.bw);
    }
    if (KtS != Kt) {
        // 行ストライド KtS で組んだプログラムのうち，K 方向に読むのは先頭の Kt 枚ぶん（num_blocks 個の
        // K ブロック）だけにする（尾崎 I の gemm_run と同じ差し替え．mcast_gemm.hpp の McastGemm::retarget_k）．ライタの引数は組んだときと同じ．
        // 差し替えた ra は A 側・B 側の両方のカーネルへ渡る（set_args）ので，B の供給も num_blocks 回で終わる．
        for (const auto& e : mg.cores) {
            auto wa = mcast_writer_args((uint32_t)Cbuf->address(), Ntg, g.pm, g.pn, g.sh, g.sw,
                                        (uint32_t)e.core.x, (uint32_t)e.core.y, dW, 0u);
            mg.retarget_k(program, e, 0u, 0u, num_blocks, wa);
        }
    }
    workload.add_program(device_range, std::move(program));
    return workload;
}

#ifdef OZ2_DIGITK
// ===== 第 2.5 段：桁積の合成と法剰余（桁分割の 7 経路）=====
// 法 1 本ぶんの GEMM の出力（同じ位の桁積を連結した後の本数．2 桁分割で 2〜4 枚，平衡 n 桁で 2n-1 枚から n^2 枚）
// から剰余 r ≡ D (mod m) を作り，法の平面へ書く．1 桁の法はここを通らない（run_digit_modulus）．
// 合成は法について線形なので，桁積を先に法へ畳めば |D| が 2^24 を超えても FP32 の
// ままで扱える（詳細は ozaki2/kernels/oz2_compose.cpp）．
// 再構成に融合せず独立した段にするのは，桁積の器を法ごとに使い回すためである：
// 融合すると S*np 枚の桁積を同時に持つことになり，14080^3 で 33 GB とデバイス DRAM に
// 収まらない．法ごとに実行時引数だけを差し替えて同じワークロードを投入し直す．
struct Oz2Compose {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    // 入出力は構築時に確定する．TensorAccessor の形（コンパイル時引数）は構築時の
    // バッファから作るので，発行時に別のバッファのアドレスを渡してよいのは，形が
    // 同じ（コンパイル時引数が一致し，ページ長もバッファ種別も同じ）ときだけである．
    // compose_run は塊に切るとき，出力（剰余の平面 crt_in から作ったライタ）へ桁積バッファ Dbuf の
    // アドレスを渡して部分和を落とすので，その条件を構築時に確かめて out_accepts_in に残す．
    // 保持するのはアドレスだけにする：shared_ptr で握ると呼び出し側が桁積バッファを
    // 手放しても解放されない．
    uint32_t in_addr = 0, out_addr = 0;
    bool out_accepts_in = false;   // 出力のライタへ入力バッファのアドレスを渡してよいか
    GridSpan grid;
    bool primed = false;   // 初回だけ計時の外で 1 回流して JIT を済ませる
};
static Oz2Compose build_compose(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, uint32_t ntot,
                                std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Dbuf,
                                std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> outB,
                                uint32_t np_lcm) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    namespace oa = oz2_args;
    const uint32_t f32 = tt::tile_size(tt::DataFormat::Float32);
    auto gs = full_grid_span(md, ntot); tm::CoreRange all_cores = gs.range();
    dist::MeshWorkload workload; auto range = dist::MeshCoordinateRange(md->shape()); tm::Program program{};
    // c_0 に GEMM の出力を法ごとの本数ずつ流す（連結後の本数．2 桁分割なら 2〜4 枚，平衡 n 桁なら
    // 2n-1 枚から n^2 枚）．深さはその公倍数にしなければならない：割り切れないとリングの
    // 境界をまたいだ組が壊れ，最初の数タイルだけ正しく残りが崩れる
    // （modslice_digit.cpp・recon_crt.cpp と同じ制約）．呼び出し側が最小公倍数を渡す（ozaki2_plan.hpp の compose_cb_depth）．
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_PLANE_IN, tt::DataFormat::Float32, np_lcm, f32);
    make_cb(program, all_cores, (tt::CBIndex)oa::CB_PLANE_OUT, tt::DataFormat::Float32, 4, f32);
    std::vector<uint32_t> ra, wa;
    // 平面 j はタイル j*ntot+g にある．平面リーダ（ozaki2/kernels/oz2_plane_reader.cpp）が出力タイルごとに np 枚を読む
    tm::TensorAccessorArgs(*Dbuf).append_to(ra);
    tm::TensorAccessorArgs(*outB).append_to(wa);
    // ライタは剰余の平面をそのまま並べる（LAYOUT_PLANES．ozaki2/kernels/modslice_writer.cpp）
    { const size_t o = wa.size(); wa.resize(o + oa::SW_CT_COUNT); wa[o + oa::SW_CT_LAYOUT] = oa::LAYOUT_PLANES; }
    auto rk = tm::CreateKernel(program, OZ2_KDIR "oz2_plane_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1, .noc = tm::NOC::RISCV_1_default, .compile_args = ra});
    auto wk = tm::CreateKernel(program, OZ2_KDIR "modslice_writer.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = wa});
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    u2d[oa::CB_PLANE_IN] = tm::UnpackToDestMode::UnpackToDestFp32;
    // dst_full_sync_en は必須：合成は DST を 0..6 番まで使う（再構成と同じ理由）．
    auto ck = tm::CreateKernel(program, OZ2_KDIR "oz2_compose.cpp", all_cores,
        tm::ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d, .compile_args = {}});
    Oz2Compose c; c.rk = rk; c.wk = wk; c.ck = ck; c.range = range;
    c.in_addr = (uint32_t)Dbuf->address(); c.out_addr = (uint32_t)outB->address();
    { std::vector<uint32_t> ai, ao; tm::TensorAccessorArgs(*Dbuf).append_to(ai); tm::TensorAccessorArgs(*outB).append_to(ao);
      c.out_accepts_in = ai == ao && Dbuf->page_size() == outB->page_size()
                      && Dbuf->device_local_config().buffer_type == outB->device_local_config().buffer_type; }
    c.grid = gs;
    // 実行時引数は法ごとに compose_set_args が入れ直す（ここでは形だけ作る）
    for (auto& core : c.grid.cores) {
        tm::SetRuntimeArgs(program, rk, core, std::vector<uint32_t>(oa::PR_COUNT, 0u));
        tm::SetRuntimeArgs(program, ck, core, std::vector<uint32_t>(oa::CP_COUNT, 0u));
        tm::SetRuntimeArgs(program, wk, core, std::vector<uint32_t>(oa::sw_count(oa::LAYOUT_PLANES, 1u), 0u)); }
    workload.add_program(range, std::move(program));
    c.workload = std::move(workload);
    return c;
}
// 法 s について合成段の実行時引数を詰める（投入はしない．ozaki2_main.cpp の compose_run が投入して計時し，OZ_TIME_WHOLE では
// 同じ詰め方で投入し直す手順を記録する）．
// 入力の平面 [in0, in0+jn) を合成して out_addr の平面 out_plane へ書く．
// 係数は unit なら全部 1（部分和どうしの足し合わせ），そうでなければ GEMM の出力 j0..j0+jn-1 の
// ものを使う（GEMM の出力 1 本（同じ位の桁積をまとめたもの）ごとに 1 つ．ozaki2_split.h の oz2_product_groups）．
// unit の係数の上位 coeff_hi は 2^12 mod m（OZ2_COMPOSE_SPLIT．2^12 分割の方式で lo と hi の両方に係数 1 を当てる）
static void compose_set_args(Oz2Compose& c, const Ozaki2Moduli& mo, uint32_t s, uint32_t j0, uint32_t jn, bool unit,
                             uint32_t in0, uint32_t out_addr, uint32_t out_plane) {
    namespace tm = tt::tt_metal;
    namespace oa = oz2_args;
    tm::Program& prog = c.workload.get_programs().at(c.range);
    Oz2ProductGroup gr[OZ2_MAXPROD];
    oz2_product_groups(mo.m[s], gr, OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[s]);
    float cb[OZ2_MAXPROD], ch[OZ2_MAXPROD];
    for (uint32_t j = 0; j < jn; j++) {
        cb[j] = unit ? 1.f : gr[j0 + j].coeff_lo;
        ch[j] = unit ? (float)(OZ2_COMPOSE_SPLIT % mo.m[s]) : gr[j0 + j].coeff_hi;
    }
    for (size_t i = 0; i < c.grid.cores.size(); i++) { auto& core = c.grid.cores[i];
        uint32_t nt = c.grid.count(i), beg = c.grid.begin(i);
        // 平面リーダは入力の平面 [in0, in0+jn) を出力タイルごとに読む（並びは ozaki2/kernels/oz2_args.h の PlaneReaderRT）
        std::vector<uint32_t> ra(oa::PR_COUNT, 0u);
        ra[oa::PR_IN_ADDR] = c.in_addr; ra[oa::PR_NUM] = nt; ra[oa::PR_START] = in0 * c.grid.ntiles + beg;
        ra[oa::PR_NTOT] = c.grid.ntiles; ra[oa::PR_NPLANES] = jn;
        tm::SetRuntimeArgs(prog, c.rk, core, ra);
        // 実行時引数の個数はカーネルごとに固定でなければならないので，係数は
        // OZ2_MAXPROD 個ぶんの枠を常に確保して余りを 0 で埋める（並びは ozaki2/kernels/oz2_args.h の ComposeRT）．
        // 合成の方式（先に法へ畳むか）の判断は法選択の fold_first．
        std::vector<uint32_t> ca(oa::CP_COUNT, 0u);
        ca[oa::CP_NTILES] = nt; ca[oa::CP_NP] = jn; ca[oa::CP_MOD] = f32_bits((float)mo.m[s]); ca[oa::CP_RINV] = f32_bits(mo.rinv[s]);
        ca[oa::CP_FOLD_FIRST] = mo.fold_first ? 1u : 0u;
        for (uint32_t j = 0; j < jn; j++) { ca[oa::cp_coeff_lo(j)] = f32_bits(cb[j]); ca[oa::cp_coeff_hi(j)] = f32_bits(ch[j]); }
        tm::SetRuntimeArgs(prog, c.ck, core, ca);
        // 出力の平面．ライタは dst = g をそのまま使うので開始位置に平面を載せる（平面 1 枚．LAYOUT_PLANES）
        std::vector<uint32_t> wr(oa::sw_count(oa::LAYOUT_PLANES, 1u));
        wr[oa::SW_DST_ADDR] = out_addr; wr[oa::SW_NUM] = nt; wr[oa::SW_START] = out_plane * c.grid.ntiles + beg;
        wr[oa::SW_PLANES] = 1u; wr[oa::SW_NTOT] = c.grid.ntiles;
        tm::SetRuntimeArgs(prog, c.wk, core, wr); }
}
#endif

// ===== 第 3 段：CRT による再構成（パス 1 本）=====
// 再構成のパスが読み書きするバッファ（run_recon が確保する．in・o64・dltA・dltB は呼び出し側のもの）
struct ReconBufs {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in;        // 法ごとの入力 S 平面（剰余の FP32 か C_i の符号-絶対値 int32）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> limb[3];   // frac の limb 3 面（診断用 OZ2_RECON_NOF64 の最終パスが書く）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> a128;      // 2 パス構成の第 1 パスが書き最終パスが読む 128 bit 固定小数（16 KB ページ）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> o64;       // 出口の FP64 の C（タイル順）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltA, dltB;   // 行・列の指数差分（tau_A*tau_B を指数へ足し戻す）
    // 出力を行の帯に分けるときの帯の先頭タイル（o64 と行・列の指数差分はタイル番号 + out_off で引く．入力 in・limb・a128 は
    // 帯 1 本ぶんの器で，帯の中の番号で引く．帯 1 本なら 0）
    uint32_t out_off = 0;
};

// 再構成段のリーダ・ライタのコンパイル時引数（TensorAccessorArgs の後ろの 3 語．recon_args.h の ReaderWriterCT）を
// 名前で詰める．final_or_share はライタには「FP64 を組み立てる最終パスか」，リーダには
// 「組み立ての前半を分担するか」，role はパスの役割（recon_args.h の ReconRole）
static void append_rw_ct(std::vector<uint32_t>& v, uint32_t final_or_share, recon_args::ReconRole role, uint32_t probe) {
    namespace rg = recon_args;
    const size_t o = v.size(); v.resize(o + rg::CT_COUNT);
    v[o + rg::CT_FINAL_OR_SHARE] = final_or_share; v[o + rg::CT_ROLE] = role; v[o + rg::CT_PROBE] = probe;
    static_assert(rg::CT_COUNT == 3, "ReaderWriterCT に欄を足したらここで詰める");
}

// 再構成のパス 1 本の形（パスとバッファと法の集合から決まる値）．recon_make_cbs・recon_make_kernels・recon_set_args が共有する
struct ReconPassShape {
    uint32_t S = 0, Mt = 0, Nt = 0, ntot = 0;
    uint32_t in_residue = 0;          // 1 = 合成段が法へ畳んだ剰余の FP32（桁分割経路），0 = GEMM 段の符号-絶対値 int32（INT8 経路）
    uint32_t FINAL = 0;               // このパスが FP64 を組み立てる（pass.assemble_f64）
    recon_args::ReconRole role = recon_args::ROLE_SINGLE;
    const float* rp = nullptr; uint32_t NRP = 0;   // 逆数の断片（[i*nrp + t]）
    Oz2ReconCT ct{};                  // 計算カーネルのコンパイル時引数のうち法の集合とパスから決まるもの（ozaki2_moduli.h の oz2_recon_ct）
    bool split_asm = false;           // 組み立てをリーダと分担するか（oz2_split_assembly）
    uint32_t jsplit = 0;              // 分担の境目（Oz2Knobs::jsplit）
    // frac の 3 本（上位・中位・下位）の書き先．2 パス構成の第 1 パスは 0 番を 128 bit 固定小数にする
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> out0, out1, out2;
};
static ReconPassShape recon_pass_shape(const ReconPass& pass, const ReconBufs& bf, uint32_t in_residue, uint32_t M, uint32_t N,
                                       const Ozaki2Moduli& mo, const Oz2Knobs& kn) {
    ReconPassShape p;
    p.S = mo.s; p.Mt = M / 32; p.Nt = N / 32; p.ntot = p.Mt * p.Nt;
    p.in_residue = in_residue;
    p.rp = pass.rp.data(); p.NRP = pass.nrp;
    p.FINAL = pass.assemble_f64 ? 1u : 0u;
    p.role = pass.role();
    p.out0 = pass.pack_a128 ? bf.a128 : bf.limb[0]; p.out1 = bf.limb[1]; p.out2 = bf.limb[2];
    // 計算カーネルのコンパイル時引数のうち法の集合とパスから決まるもの（ozaki2_moduli.h の oz2_recon_ct．ホスト単体の
    // 模擬と共有）：limb の本数（FP64 相当なら triple-float，FP32 以下なら 2 本組．ただし 2 パス構成では b が小さくても
    // 全パスを 3 本に揃える．判定は make_recon_plan の limb3．揃えないと，第 1 パスのライタが固定小数へ足す 3 本目
    // （c_16 の l2．recon_writer.cpp）が別の組の limb か CB の外の値になる），tfAddSmall へ回す本数，先に法へ畳む 1 桁の法
    p.ct = oz2_recon_ct(mo, p.rp, p.NRP, in_residue != 0, pass.limb3);
    // 組み立て（2 パス構成の第 1 パスでは固定小数への変換）をリーダと半分ずつ持つ．
    // 分担の境目（OZ2_JSPLIT で振れる．解釈と範囲の検査は oz2_knobs_from_env，分担する条件は oz2_split_assembly）．
    p.jsplit = kn.jsplit;
    p.split_asm = oz2_split_assembly(p.FINAL != 0 || p.role == recon_args::ROLE_TWO_FIRST, kn.jsplit);
    return p;
}

// 再構成段の CB（作る順は L1 の配置を決めるので変えない．CB 番号は ozaki2/kernels/recon_args.h の表で，3 本のカーネルも
// 同じ名前で引く）
static void recon_make_cbs(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores, const ReconPassShape& p) {
    namespace rg = recon_args;
    const uint32_t i32 = tt::tile_size(tt::DataFormat::Int32), f32 = tt::tile_size(tt::DataFormat::Float32);
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    // 剰余入力は FP32 で受けるので CB の形式もそちらに合わせる（変換が要らない）
    const tt::DataFormat in_fmt = p.in_residue ? tt::DataFormat::Float32 : tt::DataFormat::Int32;
    // リーダは 1 出力タイルにつき S 枚をまとめて確保して 1 回のバリアで読むので，
    // 深さは S の倍数でなければならない（割り切れないとリングの境界をまたいだ組が
    // 黙って壊れる．合成段の CB が 3 と 4 の公倍数 12 なのと同じ制約）．2S で二重化する．
    make_cb(program, all_cores, cbi(rg::CB_IN), in_fmt, 2 * p.S, f32);
    if (!p.in_residue) {   // マスク 2 枚（深さは 2 の倍数）
        make_cb(program, all_cores, cbi(rg::CB_MASK), tt::DataFormat::Int32, 4, i32);
    }
    make_cb(program, all_cores, cbi(rg::CB_LIMB), tt::DataFormat::Float32, (p.ct.NLIMB * 4), f32);  // 深さは NLIMB の倍数にする（リングの境界をまたがせない）
    if (p.FINAL) {   // ライタが FP64 を組み立てるための作業領域
        make_cb(program, all_cores, cbi(rg::CB_F64), tt::DataFormat::Float32, 1, OZ_F64_TILE_BYTES);
        make_cb(program, all_cores, cbi(rg::CB_DLT_A), tt::DataFormat::Int32, 1, f32);
        make_cb(program, all_cores, cbi(rg::CB_DLT_B), tt::DataFormat::Int32, 1, f32);
    }
    if (p.role != rg::ROLE_SINGLE) {   // 128 bit 固定小数 1 タイル：第 1 パスは書く側，最終パスは読んだ側（recon_writer.cpp）
        make_cb(program, all_cores, cbi(rg::CB_A128), tt::DataFormat::Float32, 1, OZ2_A128_PAGE);
    }
    // 作業領域 c_19・c_22 と差分タイル c_20・c_21 は同じコアの L1 なのでライタのものをリーダもそのまま見る．
    if (p.split_asm) {
        // 受け渡しのうち limb のアドレス 1 語．CB として確保するのは，2 つの
        // データ移動カーネルが同じ L1 アドレスを get_write_ptr で引けるからである．
        // 出し入れはしない．番号 2 本はセマフォに置く（起動のたびに 0 へ戻るため）．
        make_cb(program, all_cores, cbi(rg::CB_SYNC), tt::DataFormat::Int32, 1, rg::SYNC_PAGE);
    }
}

// 再構成段の 3 本のカーネル（リーダ・ライタ・計算）と受け渡しのセマフォ（作る順はリーダ → ライタ → セマフォ → 計算）
struct ReconPassKernels {
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    uint32_t sem_pub = 0, sem_done = 0;
};
static ReconPassKernels recon_make_kernels(tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all_cores,
                                           const ReconPassShape& p, const ReconBufs& bf, const Oz2Knobs& kn, const Oz2Diag& dg) {
    namespace tm = tt::tt_metal;
    namespace rg = recon_args;
    ReconPassKernels k;
    // 診断：出口の FP64 組み立てだけを外す（OZ_RECON_PROBE=1．答えは壊れる）．書き出し量は変えない．
    // OZ2_RECON_NOF64 は limb を書くので書き出しが増え，組み立てを除いた時間の上界にとどまる（make_recon_plan）．
    // リーダとライタの両方へ渡す：既定の分担（jsplit）では組み立ての前半をリーダが
    // 持つので，ライタだけ外しても半分が残ってしまう．
    const uint32_t rprobe = dg.recon;
    std::vector<uint32_t> ra, wa; tm::TensorAccessorArgs(*bf.in).append_to(ra);
    // リーダの CT_FINAL_OR_SHARE は「組み立ての前半を分担するか」（recon_args.h の ReaderWriterCT）
    append_rw_ct(ra, p.split_asm ? 1u : 0u, p.role, rprobe);
    tm::TensorAccessorArgs(*p.out0).append_to(wa); tm::TensorAccessorArgs(*p.out1).append_to(wa); tm::TensorAccessorArgs(*p.out2).append_to(wa);
    append_rw_ct(wa, p.FINAL, p.role, rprobe);
    tm::TensorAccessorArgs(*bf.o64).append_to(wa);
    tm::TensorAccessorArgs(*bf.dltA).append_to(wa); tm::TensorAccessorArgs(*bf.dltB).append_to(wa);
    tm::TensorAccessorArgs(*bf.a128).append_to(wa);   // 2 パス構成の最終パスが読む第 1 パスの 128 bit 固定小数
    // リーダが 1 出力タイルにつき法ごとの平面 S 枚をまとめて読み（oz2_plane_fetch.h），recon_crt が CRT を分数形で解き，
    // ライタが FP64（最終パス）・128 bit 固定小数（2 パス構成の第 1 パス）・診断の limb 3 面（OZ2_RECON_NOF64）のどれかを書く
    k.rk = tm::CreateKernel(program, OZ2_KDIR "recon_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1, .noc = tm::NOC::RISCV_1_default, .compile_args = ra});
    k.wk = tm::CreateKernel(program, OZ2_KDIR "recon_writer.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0, .noc = tm::NOC::RISCV_0_default, .compile_args = wa});
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default); u2d[rg::CB_IN] = tm::UnpackToDestMode::UnpackToDestFp32;
    // 受け渡しの番号．セマフォはプログラム起動のたびに初期値へ戻るので，
    // 同じ workload を繰り返し投入しても前回の番号が残らない（CB の中身は残る）．
    k.sem_pub  = p.split_asm ? tm::CreateSemaphore(program, all_cores, 0) : 0u;
    k.sem_done = p.split_asm ? tm::CreateSemaphore(program, all_cores, 0) : 0u;
    // 末尾の断片のうち tfAddSmall へ回してよい本数（oz2_recon_ct．判定はホスト検査 tools/twosum_probe.cpp と共有している．
    // 二重実装にすると，本番だけを誤って変えても検査が素通りする）に，調査用のつまみ（OZ2_NOSMALL / OZ2_ALLSMALL /
    // OZ2_NSMALL）を重ねる．解釈と範囲の検査は oz2_knobs_from_env が入口で済ませてある．
    const uint32_t NSMALL_FRAG = oz2_apply_nsmall(p.ct.NSMALL_FRAG, p.NRP, kn.nsmall);
    // dst_full_sync_en は必須：3 本組と作業用で DST を 8 本（0..7）使う．既定の半分同期では
    // FP32 の DST が 4 本ずつの 2 区画に分かれ，4..7 番が次の区画と重なるので，パッカーが
    // 前のタイルを読んでいるあいだに次のタイルの計算が書き潰しうる（dst_probe で確認）．
    k.ck = tm::CreateKernel(program, OZ2_KDIR "recon_crt.cpp", all_cores,
        tm::ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                          .compile_args = [&] {
                              std::vector<uint32_t> v(rg::C_COUNT, 0u);
                              v[rg::C_S]          = p.ct.S;
                              v[rg::C_NLIMB]      = p.ct.NLIMB;
                              v[rg::C_IN_RESIDUE] = p.ct.IN_RESIDUE;
                              v[rg::C_NRP]        = p.ct.NRP;
                              v[rg::C_NSMALL_FRAG] = NSMALL_FRAG;
                              v[rg::C_PROBE]      = dg.crt;
                              v[rg::C_RAWMASK]    = p.ct.RAWMASK;   // 先に法へ畳む 1 桁の法
                              return v; }()});
    return k;
}

// 再構成段の実行時引数（コアごと）．法ごとの定数（法・逆数・CRT の係数・逆数の断片）は実行時引数で渡す（並びは
// ozaki2/kernels/recon_args.h の crt_*）．法の本数 S・limb の本数・断片の本数・tfAddSmall の本数・先に畳む 1 桁の法は
// コンパイル時引数なので，それらが変われば JIT が組み直す
static void recon_set_args(tt::tt_metal::Program& program, const GridSpan& gs, const ReconPassShape& p,
                           const ReconPassKernels& k, const ReconBufs& bf, const Ozaki2Moduli& mo, const ReconScale& sc) {
    namespace tm = tt::tt_metal;
    const uint32_t S = p.S, NRP = p.NRP, NLIMB = p.ct.NLIMB, ntot = p.ntot, Nt = p.Nt;
    const float* rp = p.rp;
    for (size_t c = 0; c < gs.cores.size(); c++) { uint32_t nt = gs.count(c), beg = gs.begin(c);
        // 添字はカーネルと同じヘルパで作る．順に push_back すると，並べ方を
        // 変えたときにカーネル側の式（1 + (3+t)*S + i）とだけずれる．
        std::vector<uint32_t> ca(recon_args::crt_count(S, NRP, p.in_residue != 0), 0u);
        ca[recon_args::crt_ntiles()] = nt;
        for (uint32_t i = 0; i < S; i++) {
            ca[recon_args::crt_mod(S, i)]  = f32_bits((float)mo.m[i]);
            ca[recon_args::crt_rinv(S, i)] = f32_bits(mo.rinv[i]);
            ca[recon_args::crt_yi(S, i)]   = f32_bits((float)mo.yi[i]);
            // 逆数の断片（法ごとに NRP 個．並びは断片の位ごとに S 本ずつ）
            for (uint32_t t = 0; t < NRP; t++)
                ca[recon_args::crt_frag(S, t, i)] = f32_bits(rp[(size_t)i * NRP + t]);
            if (!p.in_residue)   // 2^16 mod m は符号-絶対値の復号でしか使わない
                ca[recon_args::crt_m16(S, NRP, i)] = f32_bits(mo.m16[i]);
        }
        // 位置ではなく名前で詰める（並びは ozaki2/kernels/recon_args.h）．
        // 無名の初期化子リストで並べると，読まない語が紛れても気づけない
        // （例えば dltA・dltB・Nt はライタの表にあってリーダの表には無い）．
        using namespace recon_args;
        std::vector<uint32_t> ra_rt(R_COUNT, 0u), wa_rt(W_COUNT, 0u);
        ra_rt[R_IN_ADDR] = (uint32_t)bf.in->address(); ra_rt[R_NUM] = nt; ra_rt[R_START] = beg;
        ra_rt[R_NTOT] = ntot; ra_rt[R_S] = S;
        ra_rt[R_MASK] = p.in_residue ? 0u : 1u; ra_rt[R_NLIMB] = NLIMB;
        ra_rt[R_WHI] = sc.Whi; ra_rt[R_WLO] = sc.Wlo; ra_rt[R_QW] = (uint32_t)sc.qw; ra_rt[R_SH2] = (uint32_t)sc.sh2;
        ra_rt[R_JSPLIT] = p.jsplit; ra_rt[R_SEM_PUB] = k.sem_pub; ra_rt[R_SEM_DONE] = k.sem_done;

        wa_rt[W_OUT0] = (uint32_t)p.out0->address(); wa_rt[W_NUM] = nt; wa_rt[W_START] = beg;
        wa_rt[W_OUT1] = (uint32_t)p.out1->address(); wa_rt[W_OUT2] = (uint32_t)p.out2->address();
        wa_rt[W_NLIMB] = NLIMB; wa_rt[W_O64] = (uint32_t)bf.o64->address();
        wa_rt[W_DLT_A] = (uint32_t)bf.dltA->address(); wa_rt[W_DLT_B] = (uint32_t)bf.dltB->address();
        wa_rt[W_NT] = Nt; wa_rt[W_WHI] = sc.Whi; wa_rt[W_WLO] = sc.Wlo;
        wa_rt[W_QW] = (uint32_t)sc.qw; wa_rt[W_SH2] = (uint32_t)sc.sh2;
        wa_rt[W_A128] = (uint32_t)bf.a128->address();
        wa_rt[W_JLO] = p.split_asm ? p.jsplit : 0u; wa_rt[W_SEM_PUB] = k.sem_pub; wa_rt[W_SEM_DONE] = k.sem_done;
        wa_rt[W_OUT_OFF] = bf.out_off;

        tm::SetRuntimeArgs(program, k.rk, gs.cores[c], ra_rt);
        tm::SetRuntimeArgs(program, k.ck, gs.cores[c], ca);
        tm::SetRuntimeArgs(program, k.wk, gs.cores[c], wa_rt); }
}

// 再構成を 1 パス組む．bf.in は法ごとの入力を載せたデバイスバッファで，平面 S 枚を
// 受け取る．in_residue=0 は GEMM 段が書いた符号-絶対値 int32 で，復号と上位／下位への
// 分割はカーネルが行う（INT8 経路）．in_residue=1 は合成段が法へ畳んだ剰余の FP32
// （桁分割経路）．
// FINAL のパス（pass.assemble_f64）は FP64 を o64 へ書く．FINAL でないパスのうち，2 パス構成の第 1 パス（ROLE_TWO_FIRST）は
// 自分の limb 3 本を 128 bit 固定小数へ足して a128 へ書き（16 KB ページ），診断用 OZ2_RECON_NOF64 の最終パスは frac の limb を
// limb[0..2] へ書く．投入も計時もしない（ozaki2_main.cpp の run_recon_pass が投入して計時する）．
static tt::tt_metal::distributed::MeshWorkload recon_build_pass(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                                                const ReconPass& pass, const ReconBufs& bf,
                                                                uint32_t in_residue, uint32_t M, uint32_t N,
                                                                const Ozaki2Moduli& mo, const ReconScale& sc,
                                                                const Oz2Knobs& kn, const Oz2Diag& dg) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const ReconPassShape p = recon_pass_shape(pass, bf, in_residue, M, N, mo, kn);
    // 配分はスライス段と同じで，計算グリッド全体を使う
    auto gs = full_grid_span(md, p.ntot); tm::CoreRange all_cores = gs.range();
    dist::MeshWorkload workload; auto range = dist::MeshCoordinateRange(md->shape()); tm::Program program{};
    recon_make_cbs(program, all_cores, p);
    const ReconPassKernels k = recon_make_kernels(program, all_cores, p, bf, kn, dg);
    recon_set_args(program, gs, p, k, bf, mo, sc);
    workload.add_program(range, std::move(program));
    return workload;
}
