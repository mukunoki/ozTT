// SPDX-License-Identifier: Apache-2.0
//
// 単コアのタイル積サイクル数を測る
// 【探針．本番の 16 経路のバイナリではない】remeasure_all.sh の probes の段（scripts/diagnose/probes.sh の probe_tile_*・
// probe_tilecfg_*．冷ましてから回す）が回し，論文とスライドの単コアの値を出す．
// ---------------------------------------------------------------------------
// 1 コア（論理座標 (0,0)，格子 1x1）の L1 に置いた A ブロック（pm x bw タイル）と B ブロック（bw x pn タイル）の
// 行列積を NoC 転送なしに reps 回繰り返し，タイル積 1 回（32x32x32）あたりのサイクル数を出す．
// 論文の表 tab:fmt（行列エンジンが受け付ける入力の格納形式）のタイル積の欄と「GEMM 段の効率とパス数」の節（ssec:gemm．
// 1 コアの値は下の本番の構成を mmpack で測ったもの），スライドの単コアの参照値の出どころ（LoFi の積の RESULT の
// cyc_per_prod．値は measure_logs/probe_tile_<形式>_LoFi_mm.log と docs/gemm_stage_bottleneck.md の 9 節）．
// 既定の構成：格子 1x1，出力 8x8 タイル，K 方向 4 タイル，サブブロック 2x2，FP32 の DST
// （fp32_dest_acc_en．FP32 格納の入力はアンパッカが TF32 へ縮める），出力はパックしない（表 tab:fmt とスライドの単コアの
// 参照値はこの構成）．
// 「展開のみ」も同じ格子 1x1 で測る（FP32 の展開のみを格子 1x2 の 1 コアの値で代用すると，ほかの値と系列が揃わない）．
// 出力ブロック pm x pn・K 方向 bw を与えると，尾崎の GEMM 段のコアあたりの構成（スキーム II は 11x10・サブブロック 1x2，
// スキーム I は 8x8・2x2．bw は形式ごと）を 1 コアで再現できる（110 コアの GEMM 段とコアあたりのサイクル数を比べるため）．
//
//   tile_rate_probe <形式 fp32|bf16|fp16|bfp8|fp8|bfp4|bfp2> <fidelity LoFi|HiFi2|HiFi3|HiFi4> <mode mm|math|unpack|mmpack>
//                   [reps=40000] [sh=2] [sw=2] [launches=3] [pm=8] [pn=8] [bw=4]
//   mode  mm      matmul_block（アンパッカの展開＋行列エンジンの積）．タイル積の値
//         math    行列エンジンだけ（アンパッカは SrcA・SrcB の有効の印だけを立てる）．演算のみの値
//         unpack  アンパッカだけ（行列エンジンは有効の印を待ってバンクを返すだけ）．展開のみの値
//         mmpack  mm に，尾崎の GEMM 段と同じ出力の処理を足す：サブブロックごとに K 方向 bw の積の後で DST をパックし，
//                 2 回目以降の繰り返しはパッカの L1 累積で同じ番地へ足す（GEMM 段の K ブロックごとの部分和の累積と同じ形）．
//                 出力の CB は出力ブロック全体（pm x pn タイル）で，1 回の繰り返しの終わりに取り出す
//
// 計時：1 回の起動（ウォームアップ 1 回の後に launches 回）ごとに，計算カーネルの UNPACK スレッドが入力の CB が
// 揃った時点の壁時計（RISCV_DEBUG_REG_WALL_CLOCK．コアのクロックで進む）を，PACK スレッドが最後のサブブロックを
// 受け取った時点の壁時計を L1 に書き，その差をデバイスのサイクル数とする．サイクル数はクロックが下がっても
// 変わらない量なので，ホストの時間に公称クロック 1350 MHz を掛ける方法（起動と同期の時間とクロックの低下を含む）
// より直接である．ホストの時間からの値（host_cyc_per_prod）と，両者の比から出る実効クロック（eff_mhz）も並べて出す．
// 壁時計は下位 32 bit だけを読むので，1 回の起動が 2^32 サイクルを超えると差が 1 周して小さく出る．起動ごとにホストで
// 測った時間（起動と同期を含むので，その間のサイクル数の上界になる）に公称クロック（最大のクロック）を掛けた値が 2^32 に
// 届けば，1 周しなかったと言えないので失敗にする（引数の段の reps の上限は 1 回 70 サイクルを見込んだ粗い前の検査）．
// 出力：起動ごとの TILERATE 行，launches 回の平均・最小・最大の RESULT 行と CONFIG 行（key=value．どれも common/oz_logline.hpp
// の OzLine で欄を名前で足して組む），人が読む TILE-RATE 行．
// 値は測った命令列での参照値であって，行列エンジンの上限ではない．
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "testkit/probe_common.hpp"
#include "common/oz_env.hpp"     // 引数の読み方
#include "common/oz_logline.hpp" // 機械が読む行（TILERATE・RESULT・CONFIG）
#include "common/oz_fidelity.hpp"  // fidelity の表示名（ドライバの CONFIG 行と同じ表）
using namespace tt;
using namespace tt::tt_metal;

// カーネルの引数の並びと L1 の結果の欄の語の番号は，カーネルと共有の表（testkit/kernels/probe_args.h の tile_rate_args）
namespace tra = tile_rate_args;
using tra::RES_T0; using tra::RES_T1; using tra::RES_DONE; using tra::RES_WORDS;

[[noreturn]] static void usage(const char* why) {
    std::fprintf(stderr, "%s\n使い方: tile_rate_probe <fp32|bf16|fp16|bfp8|fp8|bfp4|bfp2> <LoFi|HiFi2|HiFi3|HiFi4> "
                         "<mm|math|unpack|mmpack> [reps=40000] [sh=2] [sw=2] [launches=3] [pm=8] [pn=8] [bw=4]\n", why);
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc < 4) usage("引数が足りない");
    // 余分な位置引数は黙って読み捨てない（引数の追加・位置ずれで，既定の構成のまま測ったログが残らないように）
    if (argc > 11) usage("引数が多すぎる（形式 fidelity mode と省略できる 7 個の計 10 個まで）");
    // 未知の値は既定に落とさず止める（別の構成を測ったログが残らないように）
    const std::string fs = argv[1], fids = argv[2], ms = argv[3];
    // 形式は本番の表（common/oz_format_table.hpp の oz_format．GEMM が読むスライスの形式 in）から名前で引く．INT8 は測らない
    const auto of = probe::probe_format(fs);
    if (!of || *of == OzFmt::INT8) usage("形式は fp32|bf16|fp16|bfp8|fp8|bfp4|bfp2");
    const DataFormat fmt = oz_format(*of).in;
    const auto fido = oz_fidelity_by_name(fids.c_str());
    if (!fido) usage("fidelity は LoFi|HiFi2|HiFi3|HiFi4");
    const tt::tt_metal::MathFidelity fid = *fido;
    uint32_t mode;
    if (ms == "mm") mode = tra::MODE_MM;
    else if (ms == "math") mode = tra::MODE_MATH;
    else if (ms == "unpack") mode = tra::MODE_UNPACK;
    else if (ms == "mmpack") mode = tra::MODE_MMPACK;
    else usage("mode は mm|math|unpack|mmpack");
    // 数の引数は oz_env の読み方（10 進の整数として全体を読み切り，範囲の外は理由を出して 2 で止める）
    const uint32_t reps = oz_env::arg_u32(argc, argv, 4, 40000, 1, 1000000, "reps");
    const uint32_t sh = oz_env::arg_u32(argc, argv, 5, 2, 1, 8, "sh");
    const uint32_t sw = oz_env::arg_u32(argc, argv, 6, 2, 1, 8, "sw");
    const uint32_t launches = oz_env::arg_u32(argc, argv, 7, 3, 1, 100, "launches");
    const uint32_t pm = oz_env::arg_u32(argc, argv, 8, 8, 1, 16, "pm");
    const uint32_t pn = oz_env::arg_u32(argc, argv, 9, 8, 1, 16, "pn");
    const uint32_t bw = oz_env::arg_u32(argc, argv, 10, 4, 1, 64, "bw");
    if (pm % sh || pn % sw || sh * sw > 4) usage("サブブロックは出力ブロック pm x pn を割り切り，面積 4 以下（FP32 の DST に載る大きさ）");
    // 壁時計は下位 32 bit だけを読むので，1 回の起動が 2^32 サイクル（1350 MHz で 3.2 s）を超えないこと．ここは見込みでの
    // 前の検査（起動ごとのタイル積は reps*pm*pn*bw 回，1 回は多くて 70 サイクル程度）で，起動ごとにホストの時間でも確かめる
    if ((double)reps * pm * pn * bw * 70.0 >= 4.0e9) usage("reps が大きすぎる（壁時計の下位 32 bit が 1 周する）");

    const DataFormat ofmt = DataFormat::Float32;   // fp32_dest_acc_en の DST（mmpack 以外はパックしないが CB の形式として要る）
    const uint32_t it = tile_size(fmt), ot = tile_size(ofmt), st = sh * sw;
    // 出力の CB：mmpack は出力ブロック全体（繰り返しごとに同じ番地へ L1 累積で足す），ほかはサブブロック 2 つ分
    const uint32_t out_tiles = mode == tra::MODE_MMPACK ? pm * pn : 2 * st;
    // L1 に収まるか（入力は 1 重．尾崎の GEMM 段は 2 重緩衝なので，同じ pm・pn・bw でも本番のほうが多く使う）
    const uint64_t l1_need = (uint64_t)(pm * bw + bw * pn) * it + (uint64_t)out_tiles * ot;
    if (l1_need > 1300ull * 1024) usage("入力と出力の CB が L1（1300 KiB を上限とする）に収まらない");
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    const int clk = md->get_clock_rate_mhz();
    auto& cq = md->mesh_command_queue();
    auto dev = md->get_devices()[0];
    // 結果の欄：L1 のインタリーブのバッファを 1 本取り，その番地をコア (0,0) の L1 の中で使う（どのバンクでも同じ番地が
    // 確保されるので，コア (0,0) のその番地は他の用途と重ならない）
    distributed::ReplicatedBufferConfig bc{.size = (size_t)64 * 200};
    distributed::DeviceLocalBufferConfig dc{.page_size = 64, .buffer_type = BufferType::L1};
    auto rb = distributed::MeshBuffer::create(bc, dc, md.get());
    const uint32_t raddr = rb->address();
    const tt::tt_metal::CoreCoord core{0, 0};

    Program program{};
    CreateCircularBuffer(program, core, CircularBufferConfig(pm * bw * it, {{CBIndex::c_0, fmt}}).set_page_size(CBIndex::c_0, it));
    CreateCircularBuffer(program, core, CircularBufferConfig(bw * pn * it, {{CBIndex::c_1, fmt}}).set_page_size(CBIndex::c_1, it));
    CreateCircularBuffer(program, core, CircularBufferConfig(out_tiles * ot, {{CBIndex::c_16, ofmt}}).set_page_size(CBIndex::c_16, ot));
    // 入力の CB は 1 回だけ埋めて使い回す（値は計時に関係しないので，形式によらず同じバイト列を置く）
    std::vector<uint32_t> fct(tra::F_COUNT), cct(tra::C_COUNT), crt(tra::R_COUNT);
    fct[tra::F_N0] = pm * bw; fct[tra::F_N1] = bw * pn; fct[tra::F_B0] = pm * bw * it; fct[tra::F_B1] = bw * pn * it;
    cct[tra::C_BW] = bw; cct[tra::C_PM] = pm; cct[tra::C_PN] = pn; cct[tra::C_SH] = sh; cct[tra::C_SW] = sw;
    cct[tra::C_MODE] = mode;
    crt[tra::R_REPS] = reps; crt[tra::R_RES_ADDR] = raddr;
    static_assert(tra::F_COUNT == 4 && tra::C_COUNT == 6 && tra::R_COUNT == 2, "tile_rate_args に欄を足したらここで詰める");
    CreateKernel(program, OZ_PROBES_KDIR "tile_rate_fill.cpp", core,
                 DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                    .compile_args = fct});
    auto ck = CreateKernel(program, OZ_PROBES_KDIR "tile_rate_probe.cpp", core,
                           ComputeConfig{.math_fidelity = fid, .fp32_dest_acc_en = true,
                                         .compile_args = cct});
    SetRuntimeArgs(program, ck, core, crt);
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));

    const double prods = (double)reps * pm * pn * bw;     // 1 回の起動のタイル積の回数
    std::vector<double> cdev, chost, emhz;
    bool ok = true;
    for (uint32_t i = 0; i <= launches; i++) {           // i=0 はウォームアップ（JIT・初回の起動を計時から外す）
        std::vector<uint32_t> z(RES_WORDS, 0);
        detail::WriteToDeviceL1(dev, core, raddr, z);
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<uint32_t> rv;
        detail::ReadFromDeviceL1(dev, core, raddr, RES_WORDS * 4, rv);
        if (rv.size() < RES_WORDS || rv[RES_DONE] != 1) {
            std::fprintf(stderr, "ERROR: 起動 %u で計算カーネルの計時の欄が書かれていない（done=%u）\n", i,
                         rv.size() > RES_DONE ? rv[RES_DONE] : 0u);
            ok = false;
            break;
        }
        const uint32_t cyc = rv[RES_T1] - rv[RES_T0];    // 下位 32 bit の差（1 周未満なので符号なしの引き算で足りる．下で確かめる）
        const double cd = cyc / prods, ch = s * clk * 1e6 / prods, mhz = cyc / s / 1e6;
        if (i == 0) continue;                            // ウォームアップは JIT を含むのでホストの時間で周回を見ない（値も使わない）
        // 周回の検査（冒頭）：ホストの時間に公称クロックを掛けたサイクル数が 2^32 に届けば，壁時計の差が 1 周しなかったと
        // 言えない（クロックは公称より上がらないので，届かなければ 1 周していない）
        if (s * clk * 1e6 >= 4294967296.0) {
            std::fprintf(stderr, "ERROR: 起動 %u のホストの時間 %.3f s に公称クロック %d MHz を掛けると壁時計の下位 32 bit の 1 周"
                                 "（2^32 サイクル）に届き，デバイスのサイクル数 %u が 1 周したか分からない（reps を減らす）\n",
                         i, s, clk, cyc);
            ok = false;
            break;
        }
        cdev.push_back(cd); chost.push_back(ch); emhz.push_back(mhz);
        OzLine("TILERATE").kv("launch", i).kv("dev_cycles", cyc).num("host_ms", "%.4f", s * 1e3)
            .num("cyc_per_prod", "%.3f", cd).num("host_cyc_per_prod", "%.3f", ch).num("eff_mhz", "%.1f", mhz).print();
    }
    auto mean = [](const std::vector<double>& v) { double a = 0; for (double x : v) a += x; return v.empty() ? 0.0 : a / v.size(); };
    auto lo = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };
    auto hi = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    if (ok) {
        std::printf("TILE-RATE : %s %s %s  %.2f cyc/prod (min %.2f max %.2f, %zu launches)  host@%dMHz %.2f  eff %.0f MHz\n",
                    fs.c_str(), fids.c_str(), ms.c_str(), mean(cdev), lo(cdev), hi(cdev), cdev.size(), clk, mean(chost), mean(emhz));
        OzLine("RESULT,tile_rate_probe").kv("fmt", fs).kv("fidelity", fids).kv("mode", ms)
            .num("cyc_per_prod", "%.3f", mean(cdev)).num("cyc_min", "%.3f", lo(cdev)).num("cyc_max", "%.3f", hi(cdev))
            .num("host_cyc_per_prod", "%.3f", mean(chost)).num("host_min", "%.3f", lo(chost)).num("host_max", "%.3f", hi(chost))
            .num("eff_mhz", "%.1f", mean(emhz)).kv("launches", cdev.size()).num("prods_per_launch", "%.0f", prods).print();
    }
    // 構成の記録（検査側は名前で読む）．timing=device_cycles は RESULT の cyc_per_prod がデバイスの壁時計の差であること，
    // clk_nominal_mhz は host_cyc_per_prod に掛けた公称クロック，noc=0 は NoC の転送をしない（入力も出力もこのコアの
    // L1 に置く．NoC の番号ではない．欄の名前は前のログと揃えてある）
    OzLine("CONFIG,tile_rate_probe").kv("fmt", fs).kv("fidelity", fids).kv("mode", ms).kv("fp32_dest_acc", 1)
        .dims("grid", 1, 1).dims("core", 0, 0).kv("pm", pm).kv("pn", pn).kv("bw", bw).dims("sub", sh, sw)
        .kv("reps", reps).kv("launches", launches).kv("warmup", 1).kv("noc", 0)
        .flag("pack", mode == tra::MODE_MMPACK).flag("l1acc", mode == tra::MODE_MMPACK && reps > 1)
        .kv("timing", "device_cycles").kv("clk_nominal_mhz", clk).print();
    const bool closed = md->close();
    return ok && closed ? 0 : 1;
}
