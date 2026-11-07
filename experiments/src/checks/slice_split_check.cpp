// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I のスライス段で，FP64 を割る処理を 2 つの RISC-V に分けた経路（SPLIT=1）が，
// 分けない経路（SPLIT=0．リーダが全部を割る）と全要素でビット単位に同じスライスを出すかを見る
// 【検査】scripts/tests/check_slice_split.sh が回す（手で流す．通しでは段 checks の tests/run_checks.sh だけ）．
// ---------------------------------------------------------------------------
// 分ける経路はリーダ（slice_f64_reader.cpp）とライタ（slice_writer.cpp）が 1 ページの郵便受け
// （CB c_7）で仕事と完了を受け渡す（common/kernels/slice_f64_split.h）．受け渡しの順序が崩れると
// （知らせが L1 への書き込みより先に着くと），ライタは古い仕事の番地を読み，スライサは後半の古い
// 値を読み，リーダは読み終わっていない FP64 のタイルを上書きする．どれも「前のタイル・前の実行の
// 値」を読む誤りなので，
//   ・入力はタイルごとに値も指数差分も違い，実行ごとにすべて作り直す（前の値を読めば必ず食い違う）
//   ・同じプログラム（同じ L1 の配置）を入力だけ替えて何度も回す
//   ・担当の枚数をコアごとに 0・1・複数と振り分ける（skew）．1 コアに 65536 枚を超えて寄せると
//     CB の 16 bit の数え（tiles_received／acked）も一周する（one）
//   ・A（行ごとの差分）と B（列ごとの差分）を同じプログラムで続けて流す（ab．本番と同じ使い方）
// として，SPLIT=0 と SPLIT=1 の 2 つのプログラムの出力を毎回全語で比べる．毎回，流す前に 2 つの出力を別々の番兵で
// 埋めるので，両方の経路で同じタイルの書き込みが抜けても（前の実行の値が両方に残っても）番兵のまま食い違う．
// 検査に歯があることは探針で示す（第 6 引数 probe．slice_f64_split.h の SLICE_SPLIT_PROBE）：
//   1  ライタが後半を割る前に郵便受けを返す，2  リーダが仕事の中身を書く前に知らせる．
// どちらも同じ種類の誤りを大きな時間差でわざと起こすもので，この検査は食い違いとして捕まえるはずである
// （scripts/tests/check_slice_split.sh は探針ありで FAIL，なしで PASS を要求する）．
//
//   ./build/slice_split_check R C a|b|ab iters even|skew|one [probe] [p]
//     R C    行列の大きさ（a は R×C の A，b は C×R の B（K=C・N=R．ab の B と同じ形），ab は R×C の A と C×R の B）
//     iters  入力を作り直して回す回数
//     probe  第 6 引数．0（既定）・1・2（上の探針）
//     p      第 7 引数．スライス数 1..64（既定は形式の既定：BF16 10，INT8 8）．既定より大きい p では FP64 の情報が
//            先頭の数枚で尽き，残りのスライスは正しく 0 になるので，出力が退化していないかは先頭の
//            min(p, 既定の p) 枚で見る（全スライスの 2 経路の一致はそのまま比べる）
// 出力は "SPLIT-CHECK ... PASS|FAIL" の 1 行（終了コードも 0/1．引数の誤りは 2）．

#include "ozaki1/ozaki1_format.hpp"
#include "common/ozaki_scale.hpp"
#include "ozaki1/ozaki1_slice.hpp"
#include "common/host_eval.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

// 尾崎 I のヘッダは名前空間を開かない（tt:: で修飾して書く）ので，この翻訳単位の中だけで開く
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// 入力 1 組：タイル順の FP64（1 要素 2 語）と，指数差分のタイル（Int32．行ごとか列ごと）
struct Operand {
    std::vector<uint32_t> f64;   // tilize_f64 の結果
    std::vector<int32_t> dlt;    // 差分タイル（A は Rt 枚，B は Ct 枚）
};

// R×C の行列を作る．rowwise なら差分は行ごと（A），そうでなければ列ごと（B）．
// 値は実行・タイル・要素ごとに違う：符号・仮数は乱数，指数は行（列）ごとの基準に要素ごとの揺れを足す．
// 5% は 0．差分は d = -1 - ceil(log2 max|x|)（正規化後 |x 2^d| <= 1/2．本番の nshift=-1 と同じ）．
Operand make_operand(uint32_t R, uint32_t C, bool rowwise, uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.5, 1.0);
    std::uniform_int_distribution<int> E(-30, 30), J(-8, 0), Z(0, 19), S(0, 1);
    const uint32_t nv = rowwise ? R : C;
    std::vector<int> base(nv);
    for (auto& b : base) b = E(rng);
    std::vector<double> X((size_t)R * C);
    std::vector<double> vmax(nv, 0.0);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t c = 0; c < C; ++c) {
            const uint32_t v = rowwise ? r : c;
            double x = 0.0;
            if (Z(rng) != 0) x = (S(rng) ? -1.0 : 1.0) * std::ldexp(U(rng), base[v] + J(rng));
            X[(size_t)r * C + c] = x;
            vmax[v] = std::max(vmax[v], std::fabs(x));
        }
    Operand o;
    o.f64 = tilize_f64(X, R, C);
    const uint32_t nt = nv / 32;
    o.dlt.assign((size_t)nt * 1024, 0);
    for (uint32_t t = 0; t < nt; ++t)
        for (uint32_t k = 0; k < 32; ++k) {
            const double m = vmax[t * 32 + k];
            int e = 0;
            if (m > 0) { int ex; const double f = std::frexp(m, &ex); e = (f == 0.5) ? ex - 1 : ex; }   // ceil(log2 m)
            const int32_t d = -1 - e;
            for (uint32_t j = 0; j < 32; ++j)
                o.dlt[(size_t)t * 1024 + (rowwise ? nf(k, j) : nf(j, k))] = d;
        }
    return o;
}

// 形式の既定のスライス数（scripts/routes.tsv の p 列．尾崎 I の BF16 は 10，INT8 は 8）．第 7 引数の既定と，退化の判定に
// 使う先頭のスライスの枚数の上限
constexpr uint32_t P_DEF = OZ_FMT_ID == OzFmt::INT8 ? 8u : 10u;

// ntiles 枚をコアへ配る．even は本番（slice_run）と同じ均等割り（呼び出し側が full_grid_span(md, ntiles).counts() で
// 作って渡す），skew はコア番号 %5 が 0 のコアに 0 枚，1 のコアに 1 枚，残りのコアに残りを均等に（端数は前から），
// one は全部をコア 0 へ．
std::vector<uint32_t> distribute(const std::string& how, uint32_t ntiles, uint32_t ncores,
                                 const std::vector<uint32_t>& even) {
    std::vector<uint32_t> cnt(ncores, 0);
    if (how == "one") { cnt[0] = ntiles; return cnt; }
    if (how == "skew") {
        uint32_t rest = ntiles, nbig = 0;
        for (uint32_t c = 0; c < ncores; ++c) {
            if (c % 5 == 1 && rest) { cnt[c] = 1; --rest; }
            else if (c % 5 >= 2) ++nbig;
        }
        if (nbig) {
            const uint32_t q = rest / nbig, rmd = rest % nbig;
            uint32_t i = 0;
            for (uint32_t c = 0; c < ncores; ++c)
                if (c % 5 >= 2) { cnt[c] = q + (i < rmd ? 1u : 0u); ++i; }
        }
        return cnt;
    }
    return even;   // 本番の slice_run と同じ配り方（GridSpan::counts．ozaki1_slice.hpp）
}

}  // namespace

int main(int argc, char** argv) {
    oz_knobs::reject_removed();
    if (argc < 6 || argc > 8) {
        fprintf(stderr, "usage: %s R C a|b|ab iters even|skew|one [probe 0|1|2] [p]\n", argv[0]);
        return 2;
    }
    const uint32_t R = oz_env::arg_u32(argc, argv, 1, 0), C = oz_env::arg_u32(argc, argv, 2, 0);
    const std::string which = argv[3];
    const uint32_t iters = oz_env::arg_u32(argc, argv, 4, 1);
    const std::string dist = argv[5];
    const uint32_t probe = oz_env::arg_u32(argc, argv, 6, 0);
    const uint32_t P = oz_env::arg_u32(argc, argv, 7, P_DEF);
    if (!R || !C || R % 32 || C % 32) { fprintf(stderr, "ERROR: R,C は正で 32 の倍数\n"); return 2; }
    if (which != "a" && which != "b" && which != "ab") { fprintf(stderr, "ERROR: a|b|ab\n"); return 2; }
    if (dist != "even" && dist != "skew" && dist != "one") { fprintf(stderr, "ERROR: even|skew|one\n"); return 2; }
    if (probe > 2 || iters < 1 || P < 1 || P > 64) { fprintf(stderr, "ERROR: probe 0..2, iters>=1, p 1..64\n"); return 2; }
    if (oz_softfloat()) { fprintf(stderr, "ERROR: OZ_SOFTFLOAT では分担しない（比べる相手が無い）\n"); return 2; }
    const bool doA = which != "b", doB = which != "a";
    const uint32_t radix = OZ1_PARAM.radix;

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto gd = md->compute_with_storage_grid_size();
    auto& cq = md->mesh_command_queue();
    const uint32_t ncores = gd.x * gd.y;
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t out_t = tt::tile_size(OZ_FMT.slice_out);

    // A：R×C（Mt=R/32，Kto=C/32），B：C×R（Kto=C/32，Nt=R/32）．どちらも入力 R*C/1024 枚．
    const uint32_t Rt = R / 32, Ct = C / 32, ntiles = Rt * Ct;
    auto fA = mesh_dram_buffer(md, ntiles, 2 * f32t), fB = mesh_dram_buffer(md, ntiles, 2 * f32t);
    auto dA = mesh_dram_buffer(md, Rt, f32t), dB = mesh_dram_buffer(md, Rt, f32t);
    std::shared_ptr<distributed::MeshBuffer> oA[2], oB[2];
    for (int v = 0; v < 2; ++v) {
        oA[v] = mesh_dram_buffer(md, (size_t)ntiles * P, out_t);
        oB[v] = mesh_dram_buffer(md, (size_t)ntiles * P, out_t);
    }
    // SPLIT=0 と SPLIT=1（探針つき）の 2 つのプログラム．どちらも本番と同じ slice_build で組む．
    // 桁の上限は本番のドライバの RunCfg::dmax と同じ決め方（ozaki_digit_max．OZ_DMAX も読む）．OZ_SOFTFLOAT は上で拒んだので false
    const uint32_t dmax = ozaki_digit_max(radix);
    SliceCtx sc[2] = {slice_build(md, P, radix, dmax, false, fA, dA, oA[0], SliceTestHooks{0, 0}),
                      slice_build(md, P, radix, dmax, false, fA, dA, oA[1], SliceTestHooks{1, probe})};
    const auto cnt = distribute(dist, ntiles, ncores, full_grid_span(md, ntiles).counts());
    uint32_t cmin = UINT32_MAX, cmax = 0, c0 = 0, c1 = 0;
    for (auto c : cnt) { cmin = std::min(cmin, c); cmax = std::max(cmax, c); c0 += c == 0; c1 += c == 1; }
    // mode 0 = A（差分は i/Kto 番，s1=Kto），mode 1 = B（差分は i%Nt 番，s1=Kto，s2=Nt）．引数は本番の slice_run と同じ
    // slice_set_args で詰め，コアごとの枚数だけを cnt で与える（K ブロックに分けない：Kbt = Kto）
    auto set_args = [&](SliceCtx& s, uint32_t mode, std::shared_ptr<distributed::MeshBuffer>& f,
                        std::shared_ptr<distributed::MeshBuffer>& d, std::shared_ptr<distributed::MeshBuffer>& o) {
        const uint32_t Kto = Ct;                          // A は R×C，B は C×R で，どちらも K=C
        const uint32_t Nt = (mode == 0) ? 0u : Rt;
        slice_set_args(s, SliceRun{f, d, o, ntiles, P, Kto, Nt, mode, Kto}, cnt);
    };
    size_t bad_words = 0, bad_tiles = 0, words = 0, nonzero = 0, lead_words = 0, same_as_prev = 0;
    std::vector<uint32_t> prevA, prevB;
    // 退化の判定に使う先頭のスライスの枚数（上の p の説明）
    const uint32_t P_lead = std::min(P, P_DEF);
    // 2 つの出力を埋める番兵（違う値にしておけば，どちらの経路も書かなかったタイルは食い違う）
    const uint32_t SENT[2] = {0xA5A5A5A5u, 0x5A5A5A5Au};
    for (uint32_t it = 0; it < iters; ++it) {
        Operand A, B;
        if (doA) {
            A = make_operand(R, C, true, 1000003ull * it + 17);
            distributed::EnqueueWriteMeshBuffer(cq, fA, A.f64, false);
            distributed::EnqueueWriteMeshBuffer(cq, dA, A.dlt, false);
        }
        if (doB) {   // B は C×R（K×N）で差分は列ごと
            B = make_operand(C, R, false, 7777777ull * it + 29);
            distributed::EnqueueWriteMeshBuffer(cq, fB, B.f64, false);
            distributed::EnqueueWriteMeshBuffer(cq, dB, B.dlt, false);
        }
        distributed::Finish(cq);
        for (int v = 0; v < 2; ++v) {   // 前の実行の値を残さない（冒頭の注）．fill_mesh_buffer_u32 は同期して戻る
            if (doA) fill_mesh_buffer_u32(md, oA[v], SENT[v]);
            if (doB) fill_mesh_buffer_u32(md, oB[v], SENT[v]);
        }
        for (int v = 0; v < 2; ++v) {   // 本番と同じく，同じプログラムで A を流してから B を流す
            if (doA) { set_args(sc[v], 0, fA, dA, oA[v]); distributed::EnqueueMeshWorkload(cq, sc[v].workload, false); }
            if (doB) { set_args(sc[v], 1, fB, dB, oB[v]); distributed::EnqueueMeshWorkload(cq, sc[v].workload, false); }
        }
        distributed::Finish(cq);
        // 出力タイル t のスライス番号（slice_writer.cpp の置き方．ここは K をブロックに分けない Kbt = Kto）：
        //   A は t = mt*P*Kto + s*Kto + kk，B は t = row*Nt + n で row = (P-1-s)*Kto + kk
        auto slice_of = [&](bool isA, size_t t) -> uint32_t {
            if (isA) return (uint32_t)((t / Ct) % P);
            return P - 1 - (uint32_t)((t / Rt) / Ct);
        };
        auto cmp = [&](std::shared_ptr<distributed::MeshBuffer>* o, std::vector<uint32_t>& prev, const char* nm) {
            std::vector<uint32_t> x0, x1;
            distributed::EnqueueReadMeshBuffer(cq, x0, o[0], true);
            distributed::EnqueueReadMeshBuffer(cq, x1, o[1], true);
            const size_t wpt = out_t / 4, nt = x0.size() / wpt;
            size_t bw = 0, bt = 0;
            for (size_t t = 0; t < nt; ++t) {
                size_t b = 0;
                for (size_t w = 0; w < wpt; ++w) b += x0[t * wpt + w] != x1[t * wpt + w];
                if (b && bt < 3)
                    printf("MISMATCH it=%u %s out-tile %zu: %zu/%zu words differ\n", it, nm, t, b, wpt);
                bw += b; bt += b != 0;
            }
            const bool isA = nm[0] == 'A';
            for (size_t t = 0; t < nt; ++t) {
                if (slice_of(isA, t) >= P_lead) continue;
                for (size_t w = 0; w < wpt; ++w) nonzero += x0[t * wpt + w] != 0;
                lead_words += wpt;
            }
            if (!prev.empty() && prev == x0) ++same_as_prev;   // 入力を替えたのに出力が同じ＝検査が働いていない
            prev = std::move(x0);
            bad_words += bw; bad_tiles += bt; words += x1.size();
        };
        if (doA) cmp(oA, prevA, "A");
        if (doB) cmp(oB, prevB, "B");
    }
    // 出力のほとんどが 0 なら比べても意味が無い．スライスは符号付きの桁で，乱数入力なら先頭の P_lead 枚は大半が非零
    const bool degenerate = nonzero * 4 < lead_words || same_as_prev > 0;
    const bool pass = bad_words == 0 && !degenerate;
    printf("SPLIT-CHECK fmt=%s p=%u R=%u C=%u which=%s iters=%u dist=%s tiles/core=%u..%u (0-tile cores %u, "
           "1-tile cores %u) probe=%u words=%zu nonzero=%.1f%%（先頭 %u 枚） repeated_outputs=%zu mismatched_words=%zu "
           "mismatched_tiles=%zu %s\n",
           OZ_FMT.name, P, R, C, which.c_str(), iters, dist.c_str(), cmin, cmax, c0, c1, probe, words,
           100.0 * (double)nonzero / (double)std::max<size_t>(lead_words, 1), P_lead, same_as_prev, bad_words, bad_tiles,
           pass ? "PASS" : "FAIL");
    const bool closed = md->close();
    return (pass && closed) ? 0 : 1;
}
