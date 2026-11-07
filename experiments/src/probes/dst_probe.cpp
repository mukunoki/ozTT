// SPDX-License-Identifier: Apache-2.0
//
// DST レジスタが FP32 で何本使えるかを実機で確かめる（kernels/dst_probe.cpp の駆動）
// 【探針．本番の 16 経路のバイナリではない】remeasure_all.sh の probes の段（scripts/diagnose/probes.sh の probe_dst）が回す．
// ---------------------------------------------------------------------------
//     ./build/dst_probe [最大本数=16] [繰り返し回数=64]
// レジスタ本数 N と dst_full_sync_en の有無を振り，繰り返し t の出力タイル i が全要素
// t*100+i+1 になっているかを見る．N は 2 から試す．最初に壊れる本数が N（> 2）なら，確認できた上限は N-1 本である
// （N-1 本の回が通っている）．最初の N=2 で壊れたときは N=1 の回を流していないので，確認できたのは壊れたレジスタより
// 前の本数（reg 1 で壊れれば 1 本）だけで，reg 0 から壊れれば経路そのものの故障として 0 本・終了コード 1 にする．
// 最後に機械が読む 1 行 PROBE,dst_probe,...（欄は main の末尾）を出す．
//
// 再構成カーネルを FP32 4 本組へ拡張できるか（レジスタが 9 本以上要る）を決めるための検査．

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include "testkit/probe_common.hpp"
#include "common/oz_knobs.hpp"   // DST_PROBE_VERBOSE（つまみの表）と位置引数の読み方（oz_env.hpp）

using namespace tt;
using namespace tt::tt_metal;
namespace dpa = dst_probe_args;   // 計算カーネルのコンパイル時引数（testkit/kernels/probe_args.h）

// N 本のレジスタを試し，壊れた最初のレジスタ番号を返す（壊れなければ -1）
static int try_regs(std::shared_ptr<distributed::MeshDevice>& md, uint32_t N, bool full_sync,
                 uint32_t NT) {
    auto& cq = md->mesh_command_queue();
    const uint32_t tile = tt::tile_size(tt::DataFormat::Float32);
    auto mk = [&](size_t nt) {
        distributed::ReplicatedBufferConfig bc{.size = (size_t)tile * nt};
        distributed::DeviceLocalBufferConfig dc{.page_size = tile,
                                                .buffer_type = tt_metal::BufferType::DRAM};
        return distributed::MeshBuffer::create(bc, dc, md.get());
    };
    auto in = mk(1), out = mk((size_t)N * NT);
    std::vector<float> ones(1024, 1.0f);
    distributed::EnqueueWriteMeshBuffer(cq, in, ones, false);

    distributed::MeshWorkload workload;
    auto range = distributed::MeshCoordinateRange(md->shape());
    Program program{};
    tt::tt_metal::CoreRange core({0, 0}, {0, 0});
    CreateCircularBuffer(program, core,
        CircularBufferConfig(tile * 2, {{CBIndex::c_0, DataFormat::Float32}})
            .set_page_size(CBIndex::c_0, tile));
    CreateCircularBuffer(program, core,
        CircularBufferConfig(tile * 2 * N, {{CBIndex::c_16, DataFormat::Float32}})
            .set_page_size(CBIndex::c_16, tile));
    // リーダ：入力 1 枚を 1 回だけ c_0 へ（組 1 つ・平面 1 枚）．ライタ：繰り返し t のレジスタ i をタイル i*NT + t に置く
    // （組 NT 個・1 組 N 枚）．どちらも探針のタイル運び（probe_common.hpp）で，本番のリーダ・ライタは借りない
    const tt_metal::CoreCoord c00{0, 0};
    probe::add_tile_reader(program, c00, *in, {(uint32_t)CBIndex::c_0}, 1u, 1u);
    probe::add_tile_writer(program, c00, *out, NT, NT, N);
    std::vector<tt_metal::UnpackToDestMode> u2d(64, tt_metal::UnpackToDestMode::Default);
    u2d[(uint32_t)CBIndex::c_0] = tt_metal::UnpackToDestMode::UnpackToDestFp32;
    std::vector<uint32_t> ct(dpa::C_COUNT, 0u);
    ct[dpa::C_NREG] = N;
    ct[dpa::C_NT] = NT;
    CreateKernel(program, OZ_PROBES_KDIR "dst_probe.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = full_sync,
                      .unpack_to_dest_mode = u2d, .compile_args = ct});
    workload.add_program(range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<float> got((size_t)N * NT * 1024);
    distributed::EnqueueReadMeshBuffer(cq, got, out, true);
    for (uint32_t t = 0; t < NT; t++)
        for (uint32_t i = 0; i < N; i++)
            for (uint32_t e = 0; e < 1024; e++)
                if (got[((size_t)i * NT + t) * 1024 + e] != (float)(t * 100 + i + 1)) {
                    if (oz_knobs::u32(OZK("DST_PROBE_VERBOSE")))
                        printf("[t=%u reg=%u elem=%u: %g, expected %u] ", t, i, e,
                               got[((size_t)i * NT + t) * 1024 + e], t * 100 + i + 1);
                    return (int)i;
                }
    return -1;
}

int main(int argc, char** argv) {
    // 余分な位置引数は黙って読み捨てない（引数の追加・位置ずれで，既定の構成のまま測ったログが残らないように）
    if (argc > 3) {
        std::fprintf(stderr, "ERROR: 引数が多すぎる（使い方: dst_probe [本数の上限 2..64] [繰り返し 1..10000]．%d 個）\n", argc - 1);
        return 2;
    }
    // 本数は 2 から試す．期待値 t*100+i+1 が FP32 で厳密に表せる範囲に収まるよう上限を置く．
    // 読めない・範囲外なら止める（0 本や 0 回で何も検査せずに正常終了するのを防ぐ）
    const uint32_t NMAX = oz_env::arg_u32(argc, argv, 1, 16, 2, 64, "本数の上限");
    // 繰り返し回数．1 だと DST の先頭の半分しか通らない（kernels/dst_probe.cpp の説明）．
    const uint32_t NT = oz_env::arg_u32(argc, argv, 2, 64, 1, 10000, "繰り返し");
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    printf("FP32 の DST レジスタ本数を実測（%u 回繰り返し，出力タイルが期待値になるか）\n", NT);
    // 同期の方式ごとに，確かめた上限の本数（ok），最初に壊れた本数（ng．NMAX まで壊れなければ 0）とそのレジスタ番号（reg．
    // 壊れなければ -1）
    uint32_t ok[2] = {0, 0}, ng[2] = {0, 0};
    int reg[2] = {-1, -1};
    bool path_bad = false;   // 最初の N=2 で reg 0 から壊れた（経路そのものの故障．冒頭）
    for (int fs = 0; fs < 2; fs++) {
        printf("  dst_full_sync_en=%d : ", fs);
        ok[fs] = NMAX;
        for (uint32_t n = 2; n <= NMAX; n++) {
            const int bad = try_regs(md, n, fs != 0, NT);
            printf("N=%u %s  ", n, bad < 0 ? "OK" : "NG");
            if (bad >= 0) {
                // 確認できた上限：N-1 本の回が通っていればそれ（n > 2），最初の回なら壊れたレジスタより前の本数
                const uint32_t conf = n > 2 ? n - 1 : (uint32_t)bad;
                if (conf == 0) {
                    printf("(reg 0 から壊れる．1 本も確かめられない：経路の故障)");
                    path_bad = true;
                } else {
                    printf("(reg %d から壊れる．確認できた上限は %u 本)", bad, conf);
                }
                ok[fs] = conf; ng[fs] = n; reg[fs] = bad;
                break;
            }
        }
        printf("\n");
        fflush(stdout);
    }
    const bool closed = md->close();
    // 機械が読む行．half_* は dst_full_sync_en=0（半分同期），full_* は 1（全体同期）．*_ok は確かめた上限の本数で，
    // *_ng が 0 なら上限 nmax まで壊れなかった（*_ok は下界）
    probe::ProbeLine("dst_probe").add("nmax", "%u", NMAX).add("iters", "%u", NT)
        .add("half_ok", "%u", ok[0]).add("half_ng", "%u", ng[0]).add("half_reg", "%d", reg[0])
        .add("full_ok", "%u", ok[1]).add("full_ng", "%u", ng[1]).add("full_reg", "%d", reg[1]).print();
    return closed && !path_bad ? 0 : 1;
}
