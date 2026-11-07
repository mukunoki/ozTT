// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 探針と検査のプログラム（本番でない src/probes/ の探針と src/checks/ の検査）が共有するホスト側の部品
// ---------------------------------------------------------------------------
//   probe_format          形式の名前（fp32・bf16 など）から形式を引く．表は本番の oz_format（common/oz_format_table.hpp）
//   ProbeLine             機械が読む結果の行 "PROBE,<探針の名前>,<欄>=<値>,..."（testkit/probe_line.hpp．tt-metal に依存しないので
//                         tools/ の道具も同じものを使う）
//   run_single_core_mm    tt-metal 付属の単一コアの行列積（matmul_single_core のリーダ・ライタ）で 1 回流す
//   add_tile_reader/writer  探針のタイル運びのカーネル（kernels/probe_tile_reader.cpp・probe_tile_writer.cpp）を置く
// 本番のドライバはこのヘッダを取り込まない．
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "testkit/kernels/probe_args.h"
#include "testkit/probe_line.hpp"
#include "common/oz_paths.hpp"   // カーネルの置き場所（OZ_TESTKIT_KDIR．探針・検査の OZ_PROBES_KDIR・OZ_CHECKS_KDIR も）

// 形式の表（選択マクロ OZ_FMT_* に依らない．形式を 1 つに決めない探針・検査はこれを取り込む）
#include "common/oz_format_table.hpp"

namespace probe {

// --- 形式の表 -------------------------------------------------------------------
// 名前（OzFormat::name．CONFIG・RESULT 行の fmt と同じ）から形式を引く．知らない名前は nullopt
inline std::optional<OzFmt> probe_format(const std::string& name) {
    OzFmt f;
    if (oz_fmt_by_name(name.c_str(), f)) return f;
    return std::nullopt;
}

// --- 単一コアの行列積 -----------------------------------------------------------------
// tt-metal 付属の matmul_single_core のリーダ・ライタで，A（Mt×Kt タイル）と B（Kt×Nt タイル）を 1 回流して C を読み戻す．
// 計算カーネルは付属の mm.cpp（compile_args {Mt, Kt, Nt}）か，探針のカーネル（fma_probe）．カーネルとの契約
// （引数の順序）は tt-metal のソースで確認したもの：
//   reader : {A_addr, B_addr, Mt, Kt, Nt}，compile args に A・B の TensorAccessorArgs
//   writer : {C_addr, Mt, Nt}
// 入力 2 本（c_0・c_1）と出力（c_16）の CB は深さ 2 タイル（計算カーネルは 1 タイルずつ wait/pop する）．
inline const std::string kMatmulSingleCoreDir = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/";
inline const std::string kMatmulSingleCoreCompute = kMatmulSingleCoreDir + "compute/mm.cpp";

struct SingleCoreMM {
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b;   // A・B の形式
    tt::DataFormat out_fmt = tt::DataFormat::Float32;    // C の形式
    uint32_t Mt = 1, Kt = 1, Nt = 1;
    std::string compute = kMatmulSingleCoreCompute;      // 計算カーネルのパス
    tt::tt_metal::ComputeConfig ccfg{};                  // compile_args も呼び出し側が入れる（mm.cpp は {Mt, Kt, Nt}）
    std::vector<uint32_t> compute_rt;                    // 計算カーネルの実行時引数（mm.cpp は無し）
};

// A・B はタイル順に並べた語の列（大きさは形式のタイル長 × タイル数）．戻り値は C の語の列（タイル順）
inline std::vector<uint32_t> run_single_core_mm(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                                const SingleCoreMM& mm, const std::vector<uint32_t>& A,
                                                const std::vector<uint32_t>& B) {
    using namespace tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const uint32_t in_ts = tt::tile_size(mm.in_fmt), out_ts = tt::tile_size(mm.out_fmt);
    TT_FATAL(A.size() * 4 == (size_t)in_ts * mm.Mt * mm.Kt && B.size() * 4 == (size_t)in_ts * mm.Kt * mm.Nt,
             "run_single_core_mm: A・B の語数がタイル数と合わない");
    auto& cq = md->mesh_command_queue();
    dist::DeviceLocalBufferConfig dIn{.page_size = in_ts, .buffer_type = BufferType::DRAM};
    dist::DeviceLocalBufferConfig dOut{.page_size = out_ts, .buffer_type = BufferType::DRAM};
    auto Ab = dist::MeshBuffer::create(dist::ReplicatedBufferConfig{.size = (size_t)in_ts * mm.Mt * mm.Kt}, dIn, md.get());
    auto Bb = dist::MeshBuffer::create(dist::ReplicatedBufferConfig{.size = (size_t)in_ts * mm.Kt * mm.Nt}, dIn, md.get());
    auto Cb = dist::MeshBuffer::create(dist::ReplicatedBufferConfig{.size = (size_t)out_ts * mm.Mt * mm.Nt}, dOut, md.get());
    Program program{};
    const CoreCoord core{0, 0};
    auto mkcb = [&](uint32_t idx, tt::DataFormat f, uint32_t ts) {
        CircularBufferConfig c(2 * ts, {{idx, f}});
        c.set_page_size(idx, ts);
        CreateCircularBuffer(program, core, c);
    };
    mkcb(tt::CBIndex::c_0, mm.in_fmt, in_ts);
    mkcb(tt::CBIndex::c_1, mm.in_fmt, in_ts);
    mkcb(tt::CBIndex::c_16, mm.out_fmt, out_ts);
    std::vector<uint32_t> rca, wca;
    TensorAccessorArgs(*Ab).append_to(rca);
    TensorAccessorArgs(*Bb).append_to(rca);
    TensorAccessorArgs(*Cb).append_to(wca);
    auto rk = CreateKernel(program, kMatmulSingleCoreDir + "dataflow/reader_single_core_mm.cpp", core,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = rca});
    auto wk = CreateKernel(program, kMatmulSingleCoreDir + "dataflow/writer_single_core_mm.cpp", core,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = wca});
    auto ck = CreateKernel(program, mm.compute, core, mm.ccfg);
    SetRuntimeArgs(program, rk, core, {(uint32_t)Ab->address(), (uint32_t)Bb->address(), mm.Mt, mm.Kt, mm.Nt});
    SetRuntimeArgs(program, wk, core, {(uint32_t)Cb->address(), mm.Mt, mm.Nt});
    if (!mm.compute_rt.empty()) SetRuntimeArgs(program, ck, core, mm.compute_rt);
    dist::EnqueueWriteMeshBuffer(cq, Ab, A, false);
    dist::EnqueueWriteMeshBuffer(cq, Bb, B, false);
    dist::MeshWorkload wl;
    wl.add_program(dist::MeshCoordinateRange(md->shape()), std::move(program));
    dist::EnqueueMeshWorkload(cq, wl, false);
    std::vector<uint32_t> C(((size_t)out_ts * mm.Mt * mm.Nt + 3) / 4);
    dist::EnqueueReadMeshBuffer(cq, C, Cb, true);
    return C;
}

// 型 T の要素の列を語の列へ（タイルのバイト列をそのまま運ぶ．端数のバイトは 0 で埋める）
template <class T>
inline std::vector<uint32_t> to_words(const std::vector<T>& v) {
    std::vector<uint32_t> w((v.size() * sizeof(T) + 3) / 4, 0u);
    std::memcpy(w.data(), v.data(), v.size() * sizeof(T));
    return w;
}
template <class T>
inline std::vector<T> from_words(const std::vector<uint32_t>& w, size_t n) {
    TT_FATAL(n * sizeof(T) <= w.size() * 4, "from_words: 語が足りない");
    std::vector<T> v(n);
    std::memcpy(static_cast<void*>(v.data()), w.data(), n * sizeof(T));
    return v;
}

// --- 探針のタイル運び ------------------------------------------------------------------
// リーダ（kernels/probe_tile_reader.cpp）を RISCV_1 に置く．組ごとに平面 p（タイル p*ntot + g）を cbs[p] へ積む
inline tt::tt_metal::KernelHandle add_tile_reader(tt::tt_metal::Program& program, const tt::tt_metal::CoreCoord& core,
                                                  const tt::tt_metal::distributed::MeshBuffer& in,
                                                  const std::vector<uint32_t>& cbs, uint32_t ntiles, uint32_t ntot) {
    using namespace tt::tt_metal;
    namespace pa = probe_tile_args;
    TT_FATAL(!cbs.empty() && cbs.size() <= pa::R_MAX_PLANES, "add_tile_reader: 平面の数は 1..{}", pa::R_MAX_PLANES);
    std::vector<uint32_t> ct;
    TensorAccessorArgs(in).append_to(ct);
    std::vector<uint32_t> tail(pa::R_CT_COUNT, 0u);
    tail[pa::R_NPL] = (uint32_t)cbs.size();
    for (size_t p = 0; p < cbs.size(); ++p) tail[pa::R_CB0 + p] = cbs[p];
    ct.insert(ct.end(), tail.begin(), tail.end());
    auto k = CreateKernel(program, OZ_TESTKIT_KDIR "probe_tile_reader.cpp", core,
                          DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                             .noc = NOC::RISCV_1_default, .compile_args = ct});
    std::vector<uint32_t> rt(pa::R_COUNT, 0u);
    rt[pa::R_IN_ADDR] = (uint32_t)in.address();
    rt[pa::R_NTILES] = ntiles;
    rt[pa::R_NTOT] = ntot;
    SetRuntimeArgs(program, k, core, rt);
    return k;
}
// ライタ（kernels/probe_tile_writer.cpp）を RISCV_0 に置く．組 g の k 枚目（k < nout）をタイル k*ntot + g へ書く
inline tt::tt_metal::KernelHandle add_tile_writer(tt::tt_metal::Program& program, const tt::tt_metal::CoreCoord& core,
                                                  const tt::tt_metal::distributed::MeshBuffer& out,
                                                  uint32_t ntiles, uint32_t ntot, uint32_t nout) {
    using namespace tt::tt_metal;
    namespace pa = probe_tile_args;
    std::vector<uint32_t> ct;
    TensorAccessorArgs(out).append_to(ct);
    auto k = CreateKernel(program, OZ_TESTKIT_KDIR "probe_tile_writer.cpp", core,
                          DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                             .noc = NOC::RISCV_0_default, .compile_args = ct});
    std::vector<uint32_t> rt(pa::W_COUNT, 0u);
    rt[pa::W_OUT_ADDR] = (uint32_t)out.address();
    rt[pa::W_NTILES] = ntiles;
    rt[pa::W_NTOT] = ntot;
    rt[pa::W_NOUT] = nout;
    SetRuntimeArgs(program, k, core, rt);
    return k;
}

}  // namespace probe
