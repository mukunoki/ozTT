// SPDX-License-Identifier: Apache-2.0
//
// SFPU の模擬（checks/sfpu_tf_emu.hpp）が実機とビット単位で一致するかを，本番の関数を実機で呼んで確かめる
// 【検査】（kernels/tfmul_probe.cpp の駆動．scripts/selftest.sh が毎回まわす）
// ---------------------------------------------------------------------------
//     ./build/tfmul_probe [1 種類あたりのタイル数=4]
// ホスト単体の検査（tools/tfmul_check.cpp の FastTwoSum の前提など）は模擬の上で結論を出すので，その模擬が実機と同じ
// 値を返すことをここで押さえる．照合する演算（実機は本番の関数そのもの，模擬は sfpu_tf_emu.hpp の同じ名前の手続き）：
//   arith  タイル演算のラッパ Add・Sub・Mul（sfpu_tf.h．模擬 fadd・fsub・fmul）
//   rnd    Rnd（sfpu_tf.h．模擬 frnd）
//   mul    基数倍．1 回目は tfMulRadix<RDX>（2 のべきなら tfMulP2），2 回目は tfMulR（模擬 tfMulRadix・tfMulR）．
//          基数は 2・32・64（2 のべき．両者のビットの食い違いも数える）と 14・254（BFP4_b・INT8 の tfMulR）
//   adds3  tfAddS<3>（模擬 tfAddS3）
//   load   tf_load_f64（slice_tf_core.h．リーダが split_f64 で割った 4 枚から 3 本組を作る SFPU の部分．模擬 load_f64）
// tfAddSmall・ModRound は実機で呼んでいない：模擬も本番と同じ手続き（common/kernels/sfpu_tf_proc.h のテンプレート．
// sfpu_tf_emu.hpp の同じ名前の関数）で，違いうるのは中のタイル演算の模型（Add・Sub・Cpy・MulS・Rnd）だけなので，その模型が
// 上の演算で実機と一致すれば手続きも一致する（MulS・Cpy は mul の tfMulR・tfMulP2 の中で通る．AddS・Ltz はこの検査では通らず，
// 2 つの手続きも使わない）．
// 入力の種類は演算ごとに下の表（kArithCats・kRndCats・kMulCats・kAddsCats・kLoadCats）にある．本番の値域に当たる種類（judged．
// ±0 の並び zeros を含む）を判定に使い，値域の外の種類（非正規化数，溢れの近く，任意のビット列，load の wide）は模擬の
// 適用範囲を記録するための参考として数えるだけにする．出力の器は番兵で埋めてから流し，番兵が残った語（カーネルが書かなかった
// 語）は判定に依らず失敗に数える（PROBE 行の unwritten）．
// mul の canon はリーダの split_f64 の形（sfpu_emu::load_f64 で作る．尾の絶対値の和が ulp(hi) を超える
// tools/tfmul_check.cpp [6] の例 x = 2^-2 + 2^-25 + 127·2^-54 を先頭に置く），nearest は FP64 を FP32 へ順に
// 最近接で割った非重複の 3 本組である．
// 出力は種類ごとに人向けの 1 行と機械が読む行（PROBE,tfmul_probe,...．欄は下の report），最後に全体の PROBE 行（op=all）と
// "TFMUL-PROBE per=<n> PASS|FAIL"（判定する種類で，模擬とのビットの不一致と，2 のべきの基数での tfMulP2 と tfMulR の
// ビットの食い違いが 0 で，どの種類にも書かれなかった語が無ければ PASS．±0 の符号も比べる．終了コード 0/1，引数の誤りは 2）．
// 照合（scripts/rules/probe_checks.py．diagnose/probes.sh の probe_tfmul のログ）は PROBE 行を欄の名前で読む．
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include "testkit/probe_common.hpp"
#include "common/oz_env.hpp"     // 引数の読み方
#include "checks/sfpu_tf_emu.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace tpa = tfmul_probe_args;

namespace {

constexpr size_t TE = 1024;   // 1 タイルの要素数

struct Cat { const char* name; bool judged; };
const Cat kArithCats[] = {{"normal", true}, {"tie", true}, {"cancel", true}, {"zeros", true}, {"denorm", false},
                          {"big", false}};
const Cat kRndCats[] = {{"tie", true}, {"frac", true}, {"small", true}, {"int", true}, {"zeros", true}, {"bits", false}};
const Cat kMulCats[] = {{"canon", true}, {"nearest", true}, {"resid", true}, {"overlap", true}, {"horner", true},
                        {"zeros", true}, {"tiny", false}, {"big", false}, {"bits", false}};
const Cat kAddsCats[] = {{"horner", true}, {"crt", true}, {"split", true}, {"cancel", true}, {"zeros", true},
                         {"bits", false}};
const Cat kLoadCats[] = {{"slice1", true}, {"slice2", true}, {"range", true}, {"wide", false}};

// tools/tfmul_check.cpp [6] の例（split_f64 で (2^-2 + 2^-24, -2^-25, 127·2^-54)．尾の和が ulp(hi) を超える）
const double kX6 = 0x1p-2 + 0x1p-25 + 127 * 0x1p-54;

double U1(std::mt19937_64& g) { return std::uniform_real_distribution<double>(-1.0, 1.0)(g); }
double H1(std::mt19937_64& g) { return std::uniform_real_distribution<double>(0.5, 1.0)(g); }
int Ei(std::mt19937_64& g, int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
double sgn(std::mt19937_64& g) { return (g() & 1) ? -1.0 : 1.0; }
float finite_bits(std::mt19937_64& g) {
    uint32_t u;
    do { u = (uint32_t)g(); } while (((u >> 23) & 0xFFu) == 0xFFu);
    return bits_f32(u);
}
// FP64 を FP32 へ順に最近接で割った非重複の 3 本組
void nearest3(double x, float* t) {
    t[0] = (float)x; const double r1 = x - (double)t[0];
    t[1] = (float)r1; t[2] = (float)(r1 - (double)t[1]);
}
// 整数 N（|N| < 2^bits）の最近接の 3 本組（再構成の Horner の累積の形．各 limb は整数）
void int3(std::mt19937_64& g, int bits, float* t) {
    __int128 N = 0;
    for (int b = 0; b < bits; b += 32) N = (N << 32) | (uint32_t)g();
    N &= (((__int128)1) << bits) - 1;
    if (g() & 1) N = -N;
    t[0] = (float)N; const __int128 r1 = N - (__int128)t[0];
    t[1] = (float)r1; t[2] = (float)(r1 - (__int128)t[1]);
}
// split_f64 の形の 3 本組の元になる FP64（尾崎 I のスライサの入口 |x| <= 1/2，T を 2^20 にすれば尾崎 II の桁作りの入口）．
// i = 0,1 は [6] の例とその符号違い，ほかは sfpu_emu::gen_x の 5 種類（一様・桁の近くでほぼ整数・最近接の同点・長い
// 1 か 0 の並び・尾が ulp(hi) を超える形）を順に回す
double split_x(std::mt19937_64& g, size_t i, double beta, double T) {
    if (i < 2) return (i ? -kX6 : kX6) * (T / 0.5);
    return sfpu_emu::gen_x(g, (int)(i % 5), beta, 10, T);
}

// 実機と模擬の 1 組の比べ：ビットが 1 つでも違えば真．±0 の符号も比べる（模擬の fmul・frnd は実機と同じく，積の入力に 0 が
// あれば積を +0，Rnd の結果が 0 なら +0 にする．sfpu_tf_emu.hpp）
bool cmp(const uint32_t* dev, const float* emu, int k) {
    for (int i = 0; i < k; ++i) if (dev[i] != f32_bits(emu[i])) return true;
    return false;
}

// 種類ごとの数．emu は模擬とビットが違う要素．
// mul は 2 回目（tfMulR）の同じ数と，2 のべきの基数での両者のビットの食い違い（diff．diff_zs は ±0 の符号だけのもの），
// 3 本の和が x*rdx に厳密に等しくない要素（inex・inex_r）も数える．unwritten は出力の語のうち番兵（kUnwritten）が
// 残ったもの（カーネルが書かなかった語．要素ごとではなく語ごとに数える）
struct Tally {
    size_t n = 0, emu = 0;
    size_t emu_r = 0, diff = 0, diff_zs = 0, inex = 0, inex_r = 0;
    size_t unwritten = 0;
    size_t sub[3] = {0, 0, 0};   // arith の Add・Sub・Mul ごとの不一致
    int shown = 0;
    size_t failed() const {      // 判定に入れる不一致の要素の数
        return emu + emu_r;
    }
};

// 出力の器を埋める番兵（FP32 の静かな NaN．本番の値域の入力からも，参考の種類の有限な入力からも実機はこの語を作らない）．
// 読み戻した語にこれが残っていれば，カーネルがその語を書かなかった（前の演算の出力や DRAM の 0 と期待値が偶然一致して
// 見逃すのを防ぐ．gemm_exact_check.cpp の出力の器の番兵と同じ理由）
constexpr uint32_t kUnwritten = 0x7FC0DEADu;
// 出力の語 o[w]（w は nvar*nout 平面を並べた番号）が書かれていなければ数える．z は平面の中の要素の番号，N は平面の大きさ
size_t count_unwritten(const std::vector<uint32_t>& o, size_t z, size_t N, uint32_t nplanes) {
    size_t u = 0;
    for (uint32_t k = 0; k < nplanes; ++k) u += o[(size_t)k * N + z] == kUnwritten;
    return u;
}

// 1 つの演算を実機で回す．in は平面ごとの FP32 のビット列（どの平面も ntiles 枚），戻り値は nvar*nout 平面（出力の器は
// 番兵 kUnwritten で埋めてから流す）
std::vector<uint32_t> run_op(std::shared_ptr<distributed::MeshDevice>& md, uint32_t op, float rdx,
                             const std::vector<std::vector<uint32_t>>& in, uint32_t ntiles) {
    auto& cq = md->mesh_command_queue();
    const uint32_t tile = tt::tile_size(tt::DataFormat::Float32);
    auto mk = [&](size_t nt) {
        distributed::ReplicatedBufferConfig bc{.size = (size_t)tile * nt};
        distributed::DeviceLocalBufferConfig dc{.page_size = tile, .buffer_type = tt_metal::BufferType::DRAM};
        return distributed::MeshBuffer::create(bc, dc, md.get());
    };
    const uint32_t nin = tpa::nin(op), nout = tpa::nout(op) * tpa::nvar(op);
    TT_FATAL(in.size() == nin, "run_op: 平面の数が演算と合わない");
    auto ib = mk((size_t)nin * ntiles), ob = mk((size_t)nout * ntiles);
    {
        std::vector<uint32_t> v;
        for (const auto& p : in) { TT_FATAL(p.size() == (size_t)ntiles * TE, "run_op: 平面の大きさ"); v.insert(v.end(), p.begin(), p.end()); }
        distributed::EnqueueWriteMeshBuffer(cq, ib, v, true);
        const std::vector<uint32_t> fill((size_t)nout * ntiles * TE, kUnwritten);
        distributed::EnqueueWriteMeshBuffer(cq, ob, fill, true);
    }
    Program program{};
    const tt_metal::CoreCoord core{0, 0};
    auto mkcb = [&](CBIndex idx, uint32_t ntile) {
        CreateCircularBuffer(program, core, CircularBufferConfig(tile * ntile, {{idx, DataFormat::Float32}})
                                                .set_page_size(idx, tile));
    };
    // tf_load_f64 はスライサと同じ CB（slice_tf_core.h：c_0 hi・c_4 braw・c_1 craw・c_2 s）から読む
    std::vector<uint32_t> cbs;
    if (op == tpa::OP_LOAD) {
        for (CBIndex c : {CBIndex::c_0, CBIndex::c_4, CBIndex::c_1, CBIndex::c_2}) { mkcb(c, 2); cbs.push_back((uint32_t)c); }
    } else {
        mkcb(CBIndex::c_0, 2 * nin);
        cbs.assign(nin, (uint32_t)CBIndex::c_0);
    }
    mkcb(CBIndex::c_16, 2 * tpa::nout(op));
    probe::add_tile_reader(program, core, *ib, cbs, ntiles, ntiles);
    probe::add_tile_writer(program, core, *ob, ntiles, ntiles, nout);
    std::vector<tt_metal::UnpackToDestMode> u2d(64, tt_metal::UnpackToDestMode::Default);
    for (CBIndex c : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_2, CBIndex::c_4})
        u2d[(uint32_t)c] = tt_metal::UnpackToDestMode::UnpackToDestFp32;   // FP32 のまま DST へ（TF32 に縮めない）
    std::vector<uint32_t> ct(tpa::C_COUNT, 0u);
    ct[tpa::C_OP] = op;
    ct[tpa::C_RDX] = f32_bits(rdx);
    auto ck = CreateKernel(program, OZ_CHECKS_KDIR "tfmul_probe.cpp", core,
                           ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .compile_args = ct});
    std::vector<uint32_t> rt(tpa::R_COUNT, 0u);
    rt[tpa::R_NGROUPS] = ntiles;
    SetRuntimeArgs(program, ck, core, rt);
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    std::vector<uint32_t> o;
    distributed::EnqueueReadMeshBuffer(cq, o, ob, true);
    TT_FATAL(o.size() == (size_t)nout * ntiles * TE, "run_op: 出力の大きさ");
    return o;
}

const char* op_name(uint32_t op) {
    switch (op) {
    case tpa::OP_ARITH: return "arith";
    case tpa::OP_RND: return "rnd";
    case tpa::OP_MUL: return "mul";
    case tpa::OP_ADDS3: return "adds3";
    default: return "load";
    }
}

struct Summary { size_t cats = 0, judged = 0, failed = 0, unwritten = 0; };

// 種類 1 つぶんの報告（人向けの行と PROBE 行）．判定する種類で不一致があれば，また種類によらず書かれなかった語があれば
// failed に数える
void report(Summary& S, uint32_t op, float rdx, const Cat& c, const Tally& t) {
    const bool p2 = op == tpa::OP_MUL && (f32_bits(rdx) & 0x7FFFFFu) == 0u;
    const bool bad = (c.judged && (t.failed() || (p2 && t.diff))) || t.unwritten > 0;
    ++S.cats; S.judged += c.judged; S.failed += bad; S.unwritten += t.unwritten;
    const char* tag = c.judged ? "" : "  [参考．判定しない]";
    if (t.unwritten) printf("  %-8s カーネルが書かなかった語 %zu 個（出力の器の番兵が残った）\n", c.name, t.unwritten);
    probe::ProbeLine pl("tfmul_probe");
    pl.add("op", "%s", op_name(op));
    if (op == tpa::OP_MUL) pl.add("rdx", "%g", rdx);
    pl.add("cat", "%s", c.name).add("judged", "%d", c.judged ? 1 : 0).add("n", "%zu", t.n)
      .add("emu", "%zu", t.emu);
    if (op == tpa::OP_MUL) {
        printf("  %-8s elements=%zu bit-differences=%zu (zero-sign only %zu) sum!=x*rdx: %s=%zu r=%zu "
               "emu-mismatch: first=%zu r=%zu%s\n", c.name, t.n, p2 ? t.diff : 0, p2 ? t.diff_zs : 0,
               p2 ? "p2" : "r1", t.inex, t.inex_r, t.emu, t.emu_r, tag);
        pl.add("emu_r", "%zu", t.emu_r);
        if (p2) pl.add("diff", "%zu", t.diff).add("diff_zsign", "%zu", t.diff_zs);
        else    pl.add("diff", "n/a").add("diff_zsign", "n/a");
        pl.add("inexact", "%zu", t.inex).add("inexact_r", "%zu", t.inex_r);
    } else if (op == tpa::OP_ARITH) {
        printf("  %-8s elements=%zu emu-mismatch=%zu (Add %zu・Sub %zu・Mul %zu)%s\n", c.name, t.n,
               t.emu, t.sub[0], t.sub[1], t.sub[2], tag);
        pl.add("emu_add", "%zu", t.sub[0]).add("emu_sub", "%zu", t.sub[1]).add("emu_mul", "%zu", t.sub[2]);
    } else {
        printf("  %-8s elements=%zu emu-mismatch=%zu%s\n", c.name, t.n, t.emu, tag);
    }
    pl.add("unwritten", "%zu", t.unwritten).add("result", "%s", bad ? "FAIL" : !c.judged ? "n/a" : "PASS");
    pl.print();
}

// 不一致の例を種類ごとに 2 件まで出す
void show(Tally& t, const char* what, const char* cat, const float* in, int nin, const uint32_t* dev, const float* emu,
          int nout) {
    if (t.shown >= 2) return;
    ++t.shown;
    printf("    EMU %s %s in=(", what, cat);
    for (int i = 0; i < nin; ++i) printf("%s%a", i ? "," : "", in[i]);
    printf(") dev=(");
    for (int i = 0; i < nout; ++i) printf("%s%a", i ? "," : "", bits_f32(dev[i]));
    printf(") emu=(");
    for (int i = 0; i < nout; ++i) printf("%s%a", i ? "," : "", emu[i]);
    printf(")\n");
}

// 平面の組を作る道具：種類 c・要素 i（その種類の中の番号）ごとに f(c, i, 入力) を呼ぶ
template <size_t NC, class F>
std::vector<std::vector<float>> make_planes(const Cat (&)[NC], uint32_t per, int nin, F&& f) {
    std::vector<std::vector<float>> P(nin, std::vector<float>((size_t)NC * per * TE));
    std::vector<float> v(nin);
    for (size_t c = 0; c < NC; ++c)
        for (size_t i = 0; i < (size_t)per * TE; ++i) {
            f(c, i, v.data());
            for (int p = 0; p < nin; ++p) P[p][c * per * TE + i] = v[p];
        }
    return P;
}
std::vector<std::vector<uint32_t>> to_bits(const std::vector<std::vector<float>>& P) {
    std::vector<std::vector<uint32_t>> W;
    for (const auto& p : P) { std::vector<uint32_t> w(p.size()); for (size_t i = 0; i < p.size(); ++i) w[i] = f32_bits(p[i]); W.push_back(std::move(w)); }
    return W;
}

// --- arith：Add・Sub・Mul ----------------------------------------------------------------
void probe_arith(std::shared_ptr<distributed::MeshDevice>& md, uint32_t per, Summary& S) {
    std::mt19937_64 g(1001);
    auto P = make_planes(kArithCats, per, 2, [&](size_t c, size_t i, float* v) {
        float& a = v[0]; float& b = v[1];
        switch (c) {
        case 0: {   // 正規化数どうし（積も正規化数に収まる指数）．桁ずらしは ±24 まで
            const int ea = Ei(g, -50, 50);
            a = (float)(sgn(g) * std::ldexp(H1(g), ea)); b = (float)(sgn(g) * std::ldexp(H1(g), ea + Ei(g, -24, 24)));
            break; }
        case 1: {   // 最近接の同点：偶数番は和（1 + k 2^-23 に ±2^-24），奇数番は積（12 bit と 13 bit の奇数の仮数）
            const int e = Ei(g, -40, 40);
            if (i % 2 == 0) {
                const double k = (double)(2 * (g() % (1u << 22)) + 1);
                a = (float)(sgn(g) * std::ldexp(1.0 + k * 0x1p-23, e)); b = (float)(sgn(g) * std::ldexp(1.0, e - 24));
            } else {
                a = (float)(sgn(g) * std::ldexp((double)(2048 + 2 * (g() % 1024) + 1), e - 11));
                b = (float)(sgn(g) * std::ldexp((double)(4096 + 2 * (g() % 2048) + 1), -12));
            }
            break; }
        case 2: {   // 打ち消し：b は a の (1 ± 2^-d) 倍（d = 1..23）を FP32 へ丸めたもの（偶数番は同符号，奇数番は逆符号）
            a = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -40, 40)));
            b = (float)((double)a * (1.0 + sgn(g) * std::ldexp(1.0, -Ei(g, 1, 23)))) * ((i % 2) ? -1.0f : 1.0f);
            break; }
        case 3: {   // ±0 の組み合わせと，片方だけ非零
            const float z[2] = {0.0f, -0.0f};
            a = z[i & 1]; b = z[(i >> 1) & 1];
            if (i % 8 >= 4) { const float x = (float)U1(g); if (i % 2) a = x; else b = x; }
            break; }
        case 4:     // 非正規化数の入力・出力（2^-149..2^-110）
            a = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -149, -110)));
            b = (i % 2) ? (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -149, -110))) : (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -20, 0)));
            break;
        default:    // 溢れの近く（和・積が 2^128 を超える）
            a = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, 100, 127))); b = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, 0, 127)));
            break;
        }
    });
    const uint32_t ntiles = (uint32_t)(std::size(kArithCats) * per);
    const auto o = run_op(md, tpa::OP_ARITH, 0.0f, to_bits(P), ntiles);
    const size_t N = (size_t)ntiles * TE;
    printf("op=arith（Add・Sub・Mul）\n");
    for (size_t c = 0; c < std::size(kArithCats); ++c) {
        Tally t;
        for (size_t i = 0; i < (size_t)per * TE; ++i) {
            const size_t z = c * per * TE + i;
            const float a = P[0][z], b = P[1][z];
            const float e[3] = {sfpu_emu::fadd(a, b), sfpu_emu::fsub(a, b), sfpu_emu::fmul(a, b)};
            const uint32_t d[3] = {o[z], o[N + z], o[2 * N + z]};
            ++t.n;
            t.unwritten += count_unwritten(o, z, N, 3);
            if (cmp(d, e, 3)) {
                ++t.emu;
                for (int k = 0; k < 3; ++k) t.sub[k] += d[k] != f32_bits(e[k]);
                const float in[2] = {a, b};
                show(t, "arith", kArithCats[c].name, in, 2, d, e, 3);
            }
        }
        report(S, tpa::OP_ARITH, 0.0f, kArithCats[c], t);
    }
}

// --- rnd：Rnd ----------------------------------------------------------------------
void probe_rnd(std::shared_ptr<distributed::MeshDevice>& md, uint32_t per, Summary& S) {
    std::mt19937_64 g(2002);
    auto P = make_planes(kRndCats, per, 1, [&](size_t c, size_t i, float* v) {
        switch (c) {
        case 0: {   // 最近接の同点 k + 1/2（先頭は ±1/2・±3/2・±5/2，以降 |k| < 2^22）
            const double k = i < 6 ? (double)(i / 2) : (double)(int64_t)(g() % (1u << 23)) - (double)(1u << 22);
            v[0] = (float)((i % 2 ? -1.0 : 1.0) * (k + 0.5));
            break; }
        case 1: v[0] = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -3, 22))); break;   // 小数部のある値（|v| < 2^22）
        case 2: v[0] = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -30, 0))); break;   // |v| < 1（結果は ±0 か ±1）
        case 3: {   // 2^23 の前後（2^23 - 1/2 は同点で 2^23 へ）と，それ以上の整数（そのまま）
            const float edge[] = {8388607.5f, 8388606.5f, 8388608.0f, 8388609.0f, 16777215.0f, 16777216.0f};
            v[0] = i < 12 ? (i % 2 ? -edge[i / 2] : edge[i / 2]) : (float)(sgn(g) * std::ldexp(H1(g), Ei(g, 24, 100)));
            break; }
        case 4: {   // ±0 と非正規化数
            const float z[2] = {0.0f, -0.0f};
            v[0] = (i % 4 < 2) ? z[i & 1] : (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -149, -127)));
            break; }
        default: v[0] = finite_bits(g); break;
        }
    });
    const uint32_t ntiles = (uint32_t)(std::size(kRndCats) * per);
    const auto o = run_op(md, tpa::OP_RND, 0.0f, to_bits(P), ntiles);
    printf("op=rnd（Rnd）\n");
    for (size_t c = 0; c < std::size(kRndCats); ++c) {
        Tally t;
        for (size_t i = 0; i < (size_t)per * TE; ++i) {
            const size_t z = c * per * TE + i;
            const float e = sfpu_emu::frnd(P[0][z]);
            ++t.n;
            t.unwritten += o[z] == kUnwritten;
            if (cmp(&o[z], &e, 1)) { ++t.emu; show(t, "rnd", kRndCats[c].name, &P[0][z], 1, &o[z], &e, 1); }
        }
        report(S, tpa::OP_RND, 0.0f, kRndCats[c], t);
    }
}

// --- mul：tfMulRadix<RDX>・tfMulR ---------------------------------------------------------
void probe_mul(std::shared_ptr<distributed::MeshDevice>& md, uint32_t per, float rdx, Summary& S) {
    std::mt19937_64 g(3003 + f32_bits(rdx));
    auto P = make_planes(kMulCats, per, 3, [&](size_t c, size_t i, float* t) {
        switch (c) {
        case 0: {   // split_f64 の形（偶数番は尾崎 I のスライサの入口 |x| <= 1/2，奇数番は尾崎 II の桁作りの入口 |x| <= 2^20）
            sfpu_emu::Dst D;
            sfpu_emu::load_f64(D, split_x(g, i / 2, rdx, (i % 2) ? 0x1p20 : 0.5));
            t[0] = D.r[0]; t[1] = D.r[1]; t[2] = D.r[2];
            break; }
        case 1: nearest3(std::ldexp(U1(g), Ei(g, -30, 30)), t); break;
        case 2: t[0] = (float)(0.5 * U1(g)); t[1] = (float)std::ldexp(U1(g), -22); t[2] = (float)std::ldexp(U1(g), -46); break;
        case 3: { const int e = Ei(g, -30, 30); for (int k = 0; k < 3; ++k) t[k] = (float)std::ldexp(U1(g), e); break; }
        case 4: int3(g, Ei(g, 1, 76), t); break;   // 再構成の Horner の累積（整数．BF16 の 14080^3 で 2^74 程度）
        case 5: {   // 8 通りの ±0 の並びと，1 本だけ非零の並び
            const int k = (int)(i % 16);
            const float z[2] = {0.0f, -0.0f};
            t[0] = z[k & 1]; t[1] = z[(k >> 1) & 1]; t[2] = z[(k >> 2) & 1];
            if (k >= 8) { const float v = (float)U1(g); t[k % 3] = v; }
            break; }
        case 6: for (int k = 0; k < 3; ++k) t[k] = (float)std::ldexp(U1(g), Ei(g, -149, -100)); break;
        case 7: { const int e = Ei(g, 100, 127); for (int k = 0; k < 3; ++k) t[k] = (float)std::ldexp(U1(g), e - 24 * k); break; }
        default: for (int k = 0; k < 3; ++k) t[k] = finite_bits(g); break;
        }
    });
    const bool p2 = (f32_bits(rdx) & 0x7FFFFFu) == 0u;
    const uint32_t ntiles = (uint32_t)(std::size(kMulCats) * per);
    const auto o = run_op(md, tpa::OP_MUL, rdx, to_bits(P), ntiles);
    const size_t N = (size_t)ntiles * TE;
    printf("rdx=%g\n", rdx);
    for (size_t c = 0; c < std::size(kMulCats); ++c) {
        Tally t;
        for (size_t i = 0; i < (size_t)per * TE; ++i) {
            const size_t z = c * per * TE + i;
            uint32_t a[3], b[3];
            for (int k = 0; k < 3; ++k) { a[k] = o[(size_t)k * N + z]; b[k] = o[(size_t)(3 + k) * N + z]; }
            const float in[3] = {P[0][z], P[1][z], P[2][z]};
            ++t.n;
            t.unwritten += count_unwritten(o, z, N, 6);
            // tfMulP2 と tfMulR のビットの食い違い（2 のべきの基数だけ．そうでなければ 2 回とも tfMulR）
            if (p2) {
                int r = 0;
                for (int k = 0; k < 3; ++k) if (a[k] != b[k]) r = std::max(r, bits_f32(a[k]) == bits_f32(b[k]) ? 1 : 2);
                t.diff += r != 0; t.diff_zs += r == 1;
            }
            sfpu_emu::Dst Da, Db;
            for (int k = 0; k < 3; ++k) Da.r[k] = Db.r[k] = in[k];
            sfpu_emu::tfMulRadix(Da, rdx);
            sfpu_emu::tfMulR(Db, rdx);
            const bool ra = cmp(a, Da.r, 3), rb = cmp(b, Db.r, 3);
            t.emu += ra; t.emu_r += rb;
            if (ra) show(t, p2 ? "mul(tfMulP2)" : "mul(tfMulR)", kMulCats[c].name, in, 3, a, Da.r, 3);
            if (rb) show(t, "mul(tfMulR)", kMulCats[c].name, in, 3, b, Db.r, 3);
            // 3 本の和が x*rdx に厳密に等しいか（__float128．tiny・bits では桁が広すぎて判定の意味が薄い）
            const __float128 x = ((__float128)in[0] + in[1] + in[2]) * rdx;
            t.inex += ((__float128)bits_f32(a[0]) + bits_f32(a[1]) + bits_f32(a[2])) != x;
            t.inex_r += ((__float128)bits_f32(b[0]) + bits_f32(b[1]) + bits_f32(b[2])) != x;
        }
        report(S, tpa::OP_MUL, rdx, kMulCats[c], t);
    }
}

// --- adds3：tfAddS<3> --------------------------------------------------------------------
void probe_adds3(std::shared_ptr<distributed::MeshDevice>& md, uint32_t per, Summary& S) {
    std::mt19937_64 g(4004);
    auto P = make_planes(kAddsCats, per, 4, [&](size_t c, size_t i, float* t) {
        switch (c) {
        case 0:     // 尾崎 I の再構成の Horner：整数の累積に整数の桁の和（|v| < 2^24）を足す
            int3(g, Ei(g, 1, 76), t);
            t[3] = (float)((double)(int64_t)(g() % (1u << 24)) * sgn(g));
            break;
        case 1:     // 尾崎 II の CRT の累積：|acc| <= 2 に小さい項（|v| < 2^-k）を足す
            nearest3(2.0 * U1(g), t);
            t[3] = (float)std::ldexp(U1(g), -Ei(g, 0, 40));
            break;
        case 2: {   // split_f64 の形の 3 本組に，それより小さい値を足す
            sfpu_emu::Dst D;
            sfpu_emu::load_f64(D, split_x(g, i, 64.0, 0.5));
            t[0] = D.r[0]; t[1] = D.r[1]; t[2] = D.r[2];
            t[3] = (float)(sgn(g) * std::ldexp(H1(g), Ei(g, -30, 0)));
            break; }
        case 3:     // 上位 limb をほぼ打ち消す項（3 つに 1 つはちょうど -hi）
            nearest3(std::ldexp(U1(g), Ei(g, -20, 20)), t);
            t[3] = (i % 3 == 0) ? -t[0] : (float)(-(double)t[0] * (1.0 + sgn(g) * std::ldexp(1.0, -Ei(g, 1, 24))));
            break;
        case 4: {   // ±0 の並びと，項だけ非零
            const float z[2] = {0.0f, -0.0f};
            for (int k = 0; k < 4; ++k) t[k] = z[(i >> k) & 1];
            if (i % 32 >= 16) t[3] = (float)U1(g);
            break; }
        default: for (int k = 0; k < 4; ++k) t[k] = finite_bits(g); break;
        }
    });
    const uint32_t ntiles = (uint32_t)(std::size(kAddsCats) * per);
    const auto o = run_op(md, tpa::OP_ADDS3, 0.0f, to_bits(P), ntiles);
    const size_t N = (size_t)ntiles * TE;
    printf("op=adds3（tfAddS<3>）\n");
    for (size_t c = 0; c < std::size(kAddsCats); ++c) {
        Tally t;
        for (size_t i = 0; i < (size_t)per * TE; ++i) {
            const size_t z = c * per * TE + i;
            sfpu_emu::Dst D;
            float in[4];
            for (int k = 0; k < 4; ++k) in[k] = D.r[k] = P[k][z];
            sfpu_emu::tfAddS3(D);
            const uint32_t d[3] = {o[z], o[N + z], o[2 * N + z]};
            ++t.n;
            t.unwritten += count_unwritten(o, z, N, 3);
            if (cmp(d, D.r, 3)) { ++t.emu; show(t, "adds3", kAddsCats[c].name, in, 4, d, D.r, 3); }
        }
        report(S, tpa::OP_ADDS3, 0.0f, kAddsCats[c], t);
    }
}

// --- load：tf_load_f64（split_f64 は f64_tf.h の同じ関数をホストで呼んで 4 枚を作る）-------------------------
void probe_load(std::shared_ptr<distributed::MeshDevice>& md, uint32_t per, Summary& S) {
    std::mt19937_64 g(5005);
    std::vector<double> X(std::size(kLoadCats) * per * TE);
    auto P = make_planes(kLoadCats, per, 4, [&](size_t c, size_t i, float* t) {
        double x;
        switch (c) {
        case 0: x = split_x(g, i, 64.0, 0.5); break;       // 尾崎 I のスライサの入口
        case 1: x = split_x(g, i, 64.0, 0x1p20); break;    // 尾崎 II の桁作りの入口
        case 2: x = sgn(g) * std::ldexp(H1(g), Ei(g, -73, -1)); break;   // f64_tf.h の適用範囲（正規化の目標から 73 binade 以内）
        default: {  // 範囲の外（s・hi が落ちる下側，2^1..2^60），0・-0・FP64 の非正規化数
            const int k = (int)(i % 8);
            if (k == 0) x = (i % 16) ? -0.0 : 0.0;
            else if (k == 1) { const uint64_t b = (g() & 0xFFFFFFFFFFFFFull) | ((g() & 1) << 63); std::memcpy(&x, &b, 8); }
            else x = sgn(g) * std::ldexp(H1(g), (k % 2) ? Ei(g, -220, -74) : Ei(g, 1, 60));
            break; }
        }
        X[c * per * TE + i] = x;
        uint64_t b; std::memcpy(&b, &x, 8);
        uint32_t hi, braw, craw, s;
        split_f64((uint32_t)(b >> 32), (uint32_t)b, 0, hi, braw, craw, s);
        t[0] = bits_f32(hi); t[1] = bits_f32(braw); t[2] = bits_f32(craw); t[3] = bits_f32(s);
    });
    const uint32_t ntiles = (uint32_t)(std::size(kLoadCats) * per);
    const auto o = run_op(md, tpa::OP_LOAD, 0.0f, to_bits(P), ntiles);
    const size_t N = (size_t)ntiles * TE;
    printf("op=load（tf_load_f64）\n");
    for (size_t c = 0; c < std::size(kLoadCats); ++c) {
        Tally t;
        for (size_t i = 0; i < (size_t)per * TE; ++i) {
            const size_t z = c * per * TE + i;
            sfpu_emu::Dst D;
            sfpu_emu::load_f64(D, X[z]);
            const uint32_t d[3] = {o[z], o[N + z], o[2 * N + z]};
            ++t.n;
            t.unwritten += count_unwritten(o, z, N, 3);
            if (cmp(d, D.r, 3)) {
                ++t.emu;
                const float in[4] = {P[0][z], P[1][z], P[2][z], P[3][z]};
                show(t, "load", kLoadCats[c].name, in, 4, d, D.r, 3);
            }
        }
        report(S, tpa::OP_LOAD, 0.0f, kLoadCats[c], t);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // 余分な位置引数は黙って読み捨てない（引数の追加・位置ずれで，既定の構成のまま測ったログが残らないように）
    if (argc > 2) { fprintf(stderr, "ERROR: 引数が多すぎる（使い方: tfmul_probe [タイル数 1..64]．%d 個）\n", argc - 1); return 2; }
    const uint32_t per = oz_env::arg_u32(argc, argv, 1, 4, 1, 64, "タイル数");
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    Summary S;
    probe_arith(md, per, S);
    probe_rnd(md, per, S);
    // 基数は本番の基数倍が使うもの（スキーム I の形式ごとの基数．2 のべきの 2・32・64 と，BFP4_b の 14・INT8 の 254．
    // スキーム II の 2^20 の tfMulP2 は 2 のべきの基数と同じ手続きなので回さない．sfpu_tf_emu.hpp の production_radices）
    for (float rdx : sfpu_emu::production_radices(false)) probe_mul(md, per, rdx, S);
    probe_adds3(md, per, S);
    probe_load(md, per, S);
    const bool closed = md->close();
    const bool pass = S.failed == 0 && closed;
    probe::ProbeLine("tfmul_probe").add("op", "all").add("per", "%u", per).add("cats", "%zu", S.cats)
        .add("judged", "%zu", S.judged).add("failed", "%zu", S.failed).add("unwritten", "%zu", S.unwritten)
        .add("result", "%s", pass ? "PASS" : "FAIL").print();
    printf("TFMUL-PROBE per=%u %s\n", per, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
