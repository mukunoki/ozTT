#pragma once
// 尾崎スキーム I：スライス段（デバイス上のスライス分解）
// ---------------------------------------------------------------------------
// SliceCtx と slice_build / slice_set_args / slice_run を置く．
// 依存：ozaki1_format.hpp（OZ_FMT・OZ1_PARAM）と，それが取り込む ozaki_gemm_host.hpp（make_cb・担当の配り方
// GridSpan・full_grid_span），カーネルの置き場所 oz_paths.hpp（OZ1_KDIR・OZ_COMMON_KDIR），
// カーネルの引数の並びと CB 番号（ozaki1/kernels/oz1_args.h と，両スキーム共通の FP64 リーダの表・詰める関数
// common/slice_f64_host.hpp），カーネルへ float を渡すための f32_bits（common/oz_bits.h）．
// GEMM・再構成側には依存しない．

#include "ozaki1/ozaki1_format.hpp"
#include "ozaki1/kernels/oz1_args.h"
#include "common/oz_paths.hpp"
#include "common/oz_bits.h"
#include "common/slice_f64_host.hpp"
#include "common/kernels/f64_tf.h"   // OZ_F64_TILE_BYTES
#include <map>
#include <string>

// ===== デバイス上でのスライス分解 =====
//
// ホスト側のスライス分解ループと連結タイル化を置き換える段である．ホストは FP64 の
// オペランドをそのまま（タイル順に並べるだけで）送り，デバイスのリーダ
// slice_f64_reader が triple-float へ割って，スライサ（ozaki1/kernels/slice_tf_num.cpp・INT8 は slice_tf_sm8.cpp）が
// nslice 本の符号付き整数スライスを生成し，slice_writer が GEMM 用の連結バッファへ直接
// 書き込む．スライス処理をホストに残すと，単スレッドのスケール算出・分解・
// タイル化が実行時間を支配してしまう．


struct SliceCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    tt::tt_metal::KernelHandle rk = 0, wk = 0, ck = 0;
    std::vector<tt::tt_metal::CoreCoord> cores;
    // 構築時のスライス本数（計算カーネルのコンパイル時引数）．ライタの実行時引数のスライス本数と一致しなければ
    // ならず（ずれるとライタが存在しないスライスを永久に待つ），slice_set_args が照合する．
    uint32_t nslice = 0;
};

// 検査だけが使う口（src/checks/slice_split_check.cpp．本番は既定のまま渡す）．
//   split  0/1 なら割る処理の分担（slice_build の split）をそれに固定する．既定 -1 は本番の規則
//   probe  分担の受け渡しの順序をわざと崩す探針（common/kernels/slice_f64_split.h の SLICE_SPLIT_PROBE．答えが壊れる）．
//          検査に歯があることを示すためのもので，本番と同じカーネルの中に置き，0 なら define を足さない
struct SliceTestHooks {
    int split = -1;
    uint32_t probe = 0;
};

// スライス処理のワークロードを 1 回だけ構築する．計算グリッド全体（ほかの段と同じ full_grid_span のコアの並び）で
// リーダ → スライサ → ライタ をつなぎ，入力タイル 1 枚につき nslice 本のスライスを出す．
// dmax は桁の上限で，スライサが桁をクランプする幅になる．計画が無誤差条件・fidelity・NSINGLE_CAST の上界に使った値
// （RunCfg::dmax）と同じものを呼び出し側が渡す（ここで求め直すと，dmax の決め方を変えたときに計画が保証した桁と
// 違う桁がスライサから出て，無誤差条件が黙って崩れる）．softfloat は OZ_SOFTFLOAT（main の冒頭で読んだ Oz1Diag）．
//
// 入口の工程融合：リーダ（slice_f64_reader.cpp）は DRAM 上の FP64 をそのまま読み，
// triple-float への変換と，行ごと（A）／列ごと（B）の 2 のべき正規化（指数への
// 整数差分 refDlt）を 1 パスに畳んでスライサへ渡す．したがってホスト側には
// 変換も正規化パスも中間 DRAM バッファも残らない．
static SliceCtx slice_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                            uint32_t nslice, uint32_t radix, uint32_t dmax, bool softfloat,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& refF64,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& refDlt,
                            const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& refDst,
                            const SliceTestHooks& hooks = {}) {
    namespace tm = tt::tt_metal;
    using tt::DataFormat;
    tm::distributed::MeshWorkload workload;
    auto device_range = tm::distributed::MeshCoordinateRange(md->shape());
    tm::Program program{};
    const uint32_t f32_tile = tt::tile_size(DataFormat::Float32);
    // コアの並びと範囲だけを使う（担当の枚数は発行ごとに slice_run が入力の枚数から決める）
    const GridSpan gs = full_grid_span(md, 0u);
    const tm::CoreRange all_cores = gs.range();
    // CB 番号は表から引く（入力の 4 枚・残差・リーダの作業領域・郵便受けは両スキーム共通の slice_f64_args.h，
    // 出力は oz1_args.h）．作る順は L1 の配置を決めるので変えない
    namespace sa = slice_f64_args; namespace oa = oz1_args;
    auto cbi = [](uint32_t n) { return (tt::CBIndex)n; };
    make_cb(program, all_cores, cbi(sa::CB_HI), DataFormat::Float32, 4, f32_tile);     // 入力：上位 limb
    make_cb(program, all_cores, cbi(sa::CB_CRAW), DataFormat::Float32, 4, f32_tile);   // 入力：下位 limb
    make_cb(program, all_cores, cbi(sa::CB_SCALE), DataFormat::Float32, 4, f32_tile);  // 入力：s = ±2^(elsb+7)（f64_tf.h の split_f64）
    make_cb(program, all_cores, cbi(sa::CB_F64), DataFormat::Float32, 1, OZ_F64_TILE_BYTES);   // リーダ作業：FP64 タイル
    make_cb(program, all_cores, cbi(sa::CB_DLT), DataFormat::Int32, 1, f32_tile);   // リーダ作業：指数差分（Int32）
    make_cb(program, all_cores, cbi(sa::CB_BRAW), DataFormat::Float32, 8, f32_tile);   // 入力：中位 limb
    // スライスは桁の範囲の符号付き整数で，多くの形式ではそのまま GEMM の入力形式で書き出し，
    // GEMM もその形式で読む．int8 は符号-絶対値の UInt8，fp16 は binary16 のビット列の UInt16 で
    // 書き，GEMM はそれぞれ Int8・Float16 として読む．書く形式と読む形式でタイル長が違うと
    // DRAM 上の器の大きさが合わなくなるので，ここで確かめておく．
    TT_FATAL(tt::tile_size(OZ_FMT.slice_out) == tt::tile_size(OZ_FMT.in),
             "スライスの書き出し形式と GEMM の読み込み形式でタイル長が違う");
    make_cb(program, all_cores, cbi(oa::CB_SLICE_OUT), OZ_FMT.slice_out, 8, (uint32_t)tt::tile_size(OZ_FMT.slice_out));   // 出力：スライス
    make_cb(program, all_cores, cbi(sa::CB_RES01), DataFormat::Float32, 8, f32_tile);   // 作業：残差の limb 0,1
    make_cb(program, all_cores, cbi(sa::CB_RES2), DataFormat::Float32, 4, f32_tile);    // 作業：残差の limb 2（3 タイル同時の往復を避けるため分離）
    // 郵便受け（1 ページ MBOX_PAGE B）：リーダが FP64 のタイルの後半を割る仕事をライタへ渡す
    // （common/kernels/slice_f64_split.h）．soft-float 版のリーダは全部を自分で割るので使わない．
    const uint32_t split = (hooks.split >= 0) ? (uint32_t)(hooks.split != 0 && !softfloat)
                                              : (softfloat ? 0u : 1u);
    TT_FATAL(hooks.probe == 0 || split, "slice_build: 受け渡しの探針は分担するとき（split=1）だけ");
    // 探針のときだけ define を足す（本番の JIT のハッシュは変えない）
    std::map<std::string, std::string> pdefs;
    if (hooks.probe) pdefs["SLICE_SPLIT_PROBE"] = std::to_string(hooks.probe);
    make_cb(program, all_cores, cbi(sa::CB_MBOX), DataFormat::UInt32, 1, sa::MBOX_PAGE);
    // リーダ：soft-float 版に切り替えるか・後半をライタに割らせるか（common/slice_f64_host.hpp）
    const std::vector<uint32_t> rargs_ct = f64_reader_ct(*refF64, *refDlt, softfloat, split != 0);
    std::vector<uint32_t> wargs_ct;
    tm::TensorAccessorArgs(*refDst).append_to(wargs_ct);
    { const size_t o = wargs_ct.size(); wargs_ct.resize(o + oa::SW_CT_COUNT); wargs_ct[o + oa::SW_CT_SPLIT] = split; }
    auto rk = tm::CreateKernel(program,
        // リーダは INT8 以外の 7 形式と INT8 経路で完全に同一（CB のデータ形式はホストが
        // 決めるので，カーネル側に経路の区別が無い）．よって 1 本を共用する．
        OZ_COMMON_KDIR "slice_f64_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                               .noc = tm::NOC::RISCV_1_default,
                               .compile_args = rargs_ct, .defines = pdefs});
    auto wk = tm::CreateKernel(program,
        OZ1_KDIR "slice_writer.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                               .noc = tm::NOC::RISCV_0_default,
                               .compile_args = wargs_ct, .defines = pdefs});
    const uint32_t rb = f32_bits((float)radix);
    // 桁の上限は基数と別に渡す（INT8 は重み 254／桁 ±127）．
    const uint32_t db = f32_bits((float)dmax), ndb = f32_bits(-(float)dmax);
    std::vector<tm::UnpackToDestMode> u2d(64, tm::UnpackToDestMode::Default);
    for (uint32_t idx : {sa::CB_HI, sa::CB_CRAW, sa::CB_SCALE, sa::CB_RES01, sa::CB_BRAW, sa::CB_RES2})
        u2d[idx] = tm::UnpackToDestMode::UnpackToDestFp32;
    std::vector<uint32_t> cargs_ct(oa::SC_COUNT);
    cargs_ct[oa::SC_NSLICE] = nslice; cargs_ct[oa::SC_RADIX] = rb; cargs_ct[oa::SC_SF_IN] = softfloat ? 1u : 0u;
    cargs_ct[oa::SC_DMAX] = db; cargs_ct[oa::SC_NDMAX] = ndb; cargs_ct[oa::SC_OUT_HALF] = (uint32_t)OZ_FMT.out_half;
    static_assert(oa::SC_COUNT == 6, "SlicerCT に欄を足したらここで詰める");
    // スライサは桁の符号化で 2 本ある：INT8 は符号-絶対値の UInt8，ほかの 7 形式は数そのもの（ozaki1_param.h の signmag_slices）
    auto ck = tm::CreateKernel(program,
        OZ1_PARAM.signmag_slices ? OZ1_KDIR "slice_tf_sm8.cpp" : OZ1_KDIR "slice_tf_num.cpp", all_cores,
        tm::ComputeConfig{.fp32_dest_acc_en = true,
                          .dst_full_sync_en = true,  // 8 本の DST を 1 バンクとして使う
                          // （triple-float の残差と作業用で 8 本を同時に使うため）
                          .unpack_to_dest_mode = u2d,
                          .compile_args = cargs_ct});
    SliceCtx s;
    for (const tm::CoreCoord& core : gs.cores) {   // 並びは行優先（ほかの段と同じ）．slice_set_args の cnt もこの順
        tm::SetRuntimeArgs(program, rk, core, std::vector<uint32_t>(sa::R_COUNT, 0u));
        tm::SetRuntimeArgs(program, ck, core, std::vector<uint32_t>(oa::SR_COUNT, 0u));
        tm::SetRuntimeArgs(program, wk, core, std::vector<uint32_t>(oa::SW_COUNT, 0u));
        s.cores.push_back(core);
    }
    workload.add_program(device_range, std::move(program));
    s.workload = std::move(workload); s.range = device_range;
    s.rk = rk; s.wk = wk; s.ck = ck; s.nslice = nslice;
    return s;
}

// スライス処理 1 回ぶんの入力・出力と書き先の形（slice_run・slice_set_args の引数）
//   mode 0 = A：ライタの s1 = Kto，リーダの dlt_stride = Kto（差分の番号は i/Kto）
//   mode 1 = B：ライタの s1 = Kto, s2 = Nt，リーダの dlt_stride = Nt（番号は i%Nt）
// 指数差分 shift-e（expdelta.cpp．既定の shift=-1 なら -1-e）は A なら行ごと，B なら列ごとのブロードキャスト値である．
// Kbt は K ブロック 1 個ぶんの K タイル数（Kto を割り切る．ブロック分割なしなら Kto）．
// ライタはこれを使って出力をブロック優先レイアウトに並べる．
struct SliceRun {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> f64, dlt, dst;   // FP64 のオペランド・指数差分・スライス連結の書き先
    uint32_t ntiles = 0;    // 入力タイルの枚数
    uint32_t nslice = 0;    // スライスの本数（slice_build と同じでなければならない）
    uint32_t s1 = 0, s2 = 0, mode = 0, Kbt = 0;
};

// コアごとの担当枚数 cnt[c]（並びは s.cores）でリーダ・スライサ・ライタの実行時引数を詰める（発行はしない）．
// slice_run（タイルを均等に配る）と，配り方を変えて受け渡しを調べる検査（src/checks/slice_split_check.cpp）が使う．
// 並びは oz1_args.h と common/kernels/slice_f64_args.h の表．
static inline void slice_set_args(SliceCtx& s, const SliceRun& j, const std::vector<uint32_t>& cnt) {
    // 計算側のスライス本数は構築時の定数，ライタ側は実行時引数．食い違うとライタが来ないスライスを
    // 永久に待つ（または計算側が出したスライスを読み残す）ので，発行の前に止める．
    TT_FATAL(j.nslice == s.nslice, "slice_set_args: スライス本数 {} が slice_build の {} と違う", j.nslice, s.nslice);
    TT_FATAL(cnt.size() == s.cores.size(), "slice_set_args: 枚数 {} 個がコア {} 個と違う", cnt.size(), s.cores.size());
    TT_FATAL(j.Kbt > 0 && j.s1 % j.Kbt == 0, "slice_set_args: K ブロック {} タイルが Kto={} を割り切らない", j.Kbt, j.s1);
    tt::tt_metal::Program& prog = s.workload.get_programs().at(s.range);
    const uint32_t dlt_stride = (j.mode == 0) ? j.s1 : j.s2;
    namespace oa = oz1_args;
    uint32_t start = 0;
    for (size_t c = 0; c < s.cores.size(); ++c) {
        const uint32_t nt = cnt[c];
        tt::tt_metal::SetRuntimeArgs(prog, s.rk, s.cores[c],
            f64_reader_rt((uint32_t)j.f64->address(), (uint32_t)j.dlt->address(), nt, start, dlt_stride, j.mode));
        std::vector<uint32_t> ca(oa::SR_COUNT);
        ca[oa::SR_NTILES] = nt;
        tt::tt_metal::SetRuntimeArgs(prog, s.ck, s.cores[c], ca);
        std::vector<uint32_t> wa(oa::SW_COUNT);
        wa[oa::SW_DST_ADDR] = (uint32_t)j.dst->address(); wa[oa::SW_NUM] = nt; wa[oa::SW_START] = start;
        wa[oa::SW_NSLICE] = j.nslice; wa[oa::SW_S1] = j.s1; wa[oa::SW_S2] = j.s2; wa[oa::SW_MODE] = j.mode; wa[oa::SW_KBT] = j.Kbt;
        static_assert(oa::SW_COUNT == 8 && oa::SR_COUNT == 1, "SliceWriterRT・SlicerRT に欄を足したらここで詰める");
        tt::tt_metal::SetRuntimeArgs(prog, s.wk, s.cores[c], wa);
        start += nt;
    }
}

// スライス処理を j.ntiles 枚の入力タイルに対して 1 回発行する（タイルはほかの段と同じ full_grid_span の配り方で
// コアへ先頭から均等に配る．検査 slice_split_check の even も同じ GridSpan::counts で配る）
static inline void slice_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, SliceCtx& s, const SliceRun& j) {
    slice_set_args(s, j, full_grid_span(md, j.ntiles).counts());
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), s.workload, false);
}
