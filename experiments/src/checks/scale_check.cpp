// SPDX-License-Identifier: Apache-2.0
//
// スケール段（行／列ごとの最大絶対値 → BF16 → 指数差分）の出力を，全タイルの有効な 32 要素と
// 指数差分タイルの全要素について，ホストの参照と比べる
// 【検査】scripts/tests/check_scale.sh が回す（手で流す．通しでは段 checks の tests/run_checks.sh だけ）．
// ---------------------------------------------------------------------------
// 尾崎の両スキームのスケール段（common/ozaki_scale.hpp の scale_build・expdelta_build）を
// 本番と同じ関数で組み，出力タイルを読み戻して比べる．最終結果 C の精度では，最大値の BF16 が
// 1 ulp ずれても正規化の 2 のべきが変わらなければ見えないので，段の出力そのものを見る．
// scale_f64_reader.cpp は同じカーネルを 2 つのデータ移動 RISC-V に置き，コアの担当する出力タイルを
// 前半（切り上げ）を RISCV_1，残りを RISCV_0 が受け持つ．コアあたり 1 枚では RISCV_0 は働かないので，
// 担当が 2・3・4・5 枚になる大きさ（出力 220・330・440・550 枚．11x10 コア）と，端数のコアが出る
// 大きさを流して RISCV_0 の経路（作業領域 c_4・c_17，開始位置 start+n1，出力先）を踏む．
//
// 参照：行（列）の |x| の最大値を FP64 で取り（カーネルの upd16 と同じく，符号を落としたビット列の大小で比べる．NaN は inf より
// 大きく並ぶ），カーネルが使う common/kernels/scale_bits.h の oz_absf64_to_bf16 と同じ式（独立な参照として自前に持つ．
// scale_bits.h の冒頭．切り捨て，CEIL なら
// 落とすビットがあれば 1 つ上へ．CEIL=2（尾崎 I の BFP2_b）はちょうど 2 のべきのときも 1 つ上へ．0 は 0，非零で下端 2^-126 未満は範囲外の印 0xFF80，
// Inf・NaN は印 0xFFC0，上端は inf）で BF16 にしたもの．要素ごとに BF16 にしてから行列エンジンの MAX リダクションで最大値を取る実装
// （b7a7ee8 の scale_reduce.cpp）と同じ値になるのは，最大値が 2^-126 以上か 0 の行（列）だけである：2^-126 未満の
// 非零の最大値を，その実装は 0 に，このカーネルは印 0xFF80 にする（0xFF80 は順序を保たないので f は
// 単調でない．scale_f64_reader.cpp の冒頭）．その実装との直接の比較には第 7 引数で出力の生のバイト列を
// 書き出して突き合わせる（docs/nongemm_opt_results.md の §4）．そのときは SCALE_CHECK_TINY=0 で印の行を混ぜない
// （下端の境目の行のうち 2^-126 ちょうどは残るが，CEIL=0・1 なら両者とも 0x0080 になる）．その実装は 2 のべきの
// 1 段上げを持たないので，突き合わせは CEIL=0・1 に限る（CEIL=2 は 2 のべきの最大値の行がすべて 1 つ上になり，
// 2^-126 ちょうども 0x0081 になる）．
// 指数差分の参照は expdelta.cpp と同じ式 d = shift - e（e = ceil(log2 mx)，mx の指数部 0 は e=0）．
//
// 入力は行（列）ごとに性質を変える：乱数，全零，最大値の上位語が同じで下位語だけ違う要素の組
// （upd16 が下位語を比べる分岐），BF16 の丸めの境目（仮数の落とすビットが全部 1 で，切り上げが指数へ
// 繰り上がる），落とすビットが下位語にだけある値，2^-126 未満の非零の最大値（範囲外の印 0xFF80 になり，
// 旗のビット 1 が立つ），±Inf か NaN を 1 要素含む行（列）（印 0xFFC0 になり，旗のビット 2 が立つ．kind 5 の半分），負の最大値，下端の境目（最大値がちょうど 2^-126（0x0080 で旗なし．CEIL=2 は 2 のべきなので 0x0081），その FP64 の 1 ulp 下
// （[2^-127, 2^-126) で e8 = 0．印），FP64 の最小の非正規化数 2^-1074（上位語 0・下位語だけ非零．印），上位語に仮数の
// ある非正規化数 2^-1040（印）），上端の境目（最大値がちょうど BF16 の最大の有限値 2^128 - 2^120（0x7F7F．旗なし），
// その FP64 の 1 ulp 上（切り捨てなら 0x7F7F，切り上げ（CEIL=1・2）なら inf 0x7F80 で旗のビット 0），1 ulp 下（切り捨てなら
// 0x7F7E，切り上げなら 0x7F7F），2^128 ちょうど（どの CEIL でも inf で旗のビット 0））．旗（expdelta_flags）も参照と比べる：
// inf（2^128 以上）の行（列）があればビット 0，2^-126 未満の行（列）があればビット 1，Inf・NaN の行（列）があればビット 2 が
// 立ち，無ければ旗は 0 でなければならない．環境変数 SCALE_CHECK_TINY=0 で印・旗になる種類の行（列）を
// 乱数の行に替え，旗が立たないことを確かめる（既定は混ぜる．2^-126 ちょうど・2^128 - 2^120 ちょうどとその 1 ulp 下の行は
// TINY=0 でも残す）．
// 行（列）の数が種類 10 通り × 境目 4 通りに足りる大きさ（M + N >= 1280）では，境目を実際に踏んだことも PASS の
// 条件にする（下端は 2^-126 ちょうどの最大値，FP64 の非正規化数の最大値，[2^-127, 2^-126) の最大値の件数，上端は
// 2^128 - 2^120 ちょうどとその 1 ulp 下・1 ulp 上・2^128 の最大値の件数が 0 でない．TINY=0 なら印・旗になるものは 0）．
// 同じ大きさでは Inf・NaN の行（列）を実際に踏んだこと（nonfinite_max が 0 でない．TINY=0 なら 0）も条件にする．
//
//   ./build/scale_check M N K ceil shift seed [dump_prefix]   （ceil は 0 切り捨て・1 切り上げ・2 切り上げと 2 のべきの 1 段上げ，
//                                                              shift は 0（正規化の目標 2^0）か 1（2^-1．expdelta の shift=-1））
//   ./build/scale_check empty     空の行列（K=0・出力 0 枚）を組み立てが止めるか
// 出力は "SCALE-CHECK ... PASS|FAIL" の 1 行（終了コードも 0/1．引数の誤りは 2）．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

#include "common/ozaki_gemm_host.hpp"
#include "common/ozaki_scale.hpp"
#include "common/host_eval.hpp"

namespace {

uint64_t bits(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
double from_bits(uint64_t u) { double x; std::memcpy(&x, &u, 8); return x; }

// common/kernels/scale_bits.h の oz_absf64_to_bf16（scale_f64_reader.cpp が呼ぶ）と同じ式を独立に持つ（|x| の上位語 h・下位語 l から）
uint16_t to_bf16(double ax, uint32_t ceil_) {
    const uint64_t u = bits(ax);
    const uint32_t h = (uint32_t)(u >> 32) & 0x7FFFFFFFu, l = (uint32_t)u;
    const int32_t e8 = (int32_t)((h >> 20) & 0x7FFu) - 896;
    if (((h >> 20) & 0x7FFu) == 0x7FFu) return 0xFFC0u;   // Inf・NaN の印
    if (e8 <= 0) return (h | l) ? 0xFF80u : 0u;   // 非零で 2^-126 未満は範囲外の印
    if (e8 >= 255) return 0x7F80u;
    uint32_t v = ((uint32_t)e8 << 7) | ((h >> 13) & 0x7Fu);
    if (ceil_ == 1 && ((h & 0x1FFFu) || l)) v += 1u;
    if (ceil_ == 2 && ((h & 0x1FFFu) || l || !(v & 0x7Fu))) v += 1u;   // ちょうど 2 のべきも 1 つ上へ
    return (uint16_t)v;
}
// expdelta.cpp と同じ式
int32_t delta_of(uint16_t u, int32_t shift) {
    const uint32_t E = (u >> 7) & 0xFFu, m = u & 0x7Fu;
    int32_t e = 0;
    if (E == 0u) e = 0;
    else if (E == 0xFFu) e = 0;
    else e = (int32_t)E - 127 + (m ? 1 : 0);
    return shift - e;
}

// 印・旗になる最大値の行（列）を混ぜるか（SCALE_CHECK_TINY．既定 1）
bool g_tiny = true;

// 行（列）の性質の種類の数（下の element の kind 0..NKIND-1）．行 v の種類は v の一次式を NKIND で割った余りで決め
// （main），同じ種類の行は v が NKIND おきに並ぶので，境目の種類の中の 4 通りは (v / NKIND) % 4 で順に回る
constexpr uint32_t NKIND = 10;
// BF16 の最大の有限値 (2 - 2^-7) 2^127（0x7F7F）
const double kBf16Max = std::ldexp(255.0, 120);

// 行（列）v の 1 要素．kind は v ごとの性質（上の説明）．k は縮約方向の位置，n はその長さ．
double element(uint32_t kind, uint32_t v, uint32_t k, uint32_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const int sc = (int)(v % 41) - 20;                          // 行ごとの大きさ 2^-20..2^20
    const uint32_t peak = (v * 7919u) % n;                      // 最大値を置く位置（行ごとに違う）
    switch (kind) {
    case 0: return std::ldexp(U(rng), sc);                      // 乱数
    case 1: return 0.0;                                         // 全零
    case 2: {   // 上位語が同じで下位語だけが違う要素を並べ，最大は下位語が最大のもの（位置は peak）
        const uint64_t hi = (uint64_t)bits(std::ldexp(1.25, sc)) & 0xFFFFFFFF00000000ull;
        const uint32_t lo = (k == peak) ? 0xFFFFFFF0u : (uint32_t)(rng() & 0x7FFFFFFFu);
        const double x = from_bits(hi | lo);
        return (k & 1) ? -x : x;
    }
    case 3: {   // BF16 の丸めの境目：仮数の上位 7 bit が全部 1 で落とすビットも全部 1（CEIL で指数へ繰り上がる）
        const uint64_t e = (uint64_t)(1023 + sc) << 52;
        const double x = from_bits(e | 0x000FFFFFFFFFFFFFull);   // (2 - 2^-52) * 2^sc
        return (k == peak) ? x : std::ldexp(U(rng), sc - 1);
    }
    case 4: {   // 落とすビットが下位語にだけある：上位語の仮数の下 13 bit は 0，下位語は 1
        const uint64_t e = (uint64_t)(1023 + sc) << 52;
        const double x = from_bits(e | (0x5ull << 45) | 0x1ull);
        return (k == peak) ? -x : std::ldexp(U(rng), sc - 2);
    }
    case 5: {   // 範囲外の印になる最大値．(v/NKIND)%4 が 0・1 なら最大値が 2^-126 未満（印 0xFF80・旗のビット 1），2 なら ±Inf，
                // 3 なら NaN を位置 peak に 1 つ含む（印 0xFFC0・旗のビット 2．ほかの要素は通常の乱数）
        if (!g_tiny) return std::ldexp(U(rng), sc);             // SCALE_CHECK_TINY=0 なら乱数の行に替える
        const uint32_t sub = (v / NKIND) % 4u;
        if (sub < 2) return std::ldexp(U(rng), -140);
        if (k != peak) return std::ldexp(U(rng), sc);
        if (sub == 2) return (v & 1) ? -INFINITY : INFINITY;
        return std::numeric_limits<double>::quiet_NaN();
    }
    case 6: {   // 最大の要素が負で，ちょうど 2 のべき（仮数 0．CEIL=1 では上がらず，CEIL=2 では 1 つ上がる）
        return (k == peak) ? -std::ldexp(1.0, sc) : std::ldexp(U(rng), sc - 1);
    }
    case 7: {   // 同じ最大値が行の中に複数ある（全要素が同じ大きさで符号だけ違う）
        const double x = std::ldexp(1.0 + 1.0 / 3.0, sc);
        return (k & 2) ? -x : x;
    }
    case 8: {   // 下端の境目．行（列）ごとに (v/NKIND)%4 で 4 通り
        const uint32_t sub = (v / NKIND) % 4u;
        if (!g_tiny && sub != 0) return std::ldexp(U(rng), sc);   // TINY=0：印になる 3 通りは乱数の行に替える
        double mx;
        switch (sub) {
        case 0:  mx = std::ldexp(1.0, -126); break;                          // ちょうど 2^-126：0x0080（CEIL=2 は 0x0081．旗なし）
        case 1:  mx = std::nextafter(std::ldexp(1.0, -126), 0.0); break;     // 2^-126 の 1 ulp 下：e8 = 0 で印
        case 2:  mx = from_bits(1ull); break;                                // 2^-1074：上位語 0・下位語 1 で印
        default: mx = std::ldexp(1.0, -1040); break;                         // 上位語に仮数のある非正規化数：印
        }
        if (k == peak) return (k & 1) ? -mx : mx;
        return sub == 2 ? 0.0 : mx * 0.5 * U(rng);   // 他の要素は最大値より小さく（2^-1074 より小さい非零は無い）
    }
    default: {  // 上端の境目（BF16 の最大の有限値 2^128 - 2^120 の前後）．行（列）ごとに (v/NKIND)%4 で 4 通り
        const uint32_t sub = (v / NKIND) % 4u;
        if (!g_tiny && (sub == 1 || sub == 2)) return std::ldexp(U(rng), sc);   // TINY=0：inf・旗になりうる 2 通りは乱数の行に
        double mx;
        switch (sub) {
        case 0:  mx = kBf16Max; break;                                      // ちょうど：0x7F7F（どの CEIL でも旗なし）
        case 1:  mx = std::nextafter(kBf16Max, INFINITY); break;            // 1 ulp 上：切り上げなら inf 0x7F80・旗のビット 0
        case 2:  mx = std::ldexp(1.0, 128); break;                          // 2^128：どの CEIL でも inf 0x7F80・旗のビット 0
        default: mx = std::nextafter(kBf16Max, 0.0); break;                 // 1 ulp 下：0x7F7E，切り上げなら 0x7F7F
        }
        if (k == peak) return (k & 1) ? -mx : mx;
        return mx * 0.5 * U(rng);
    }
    }
}

// ---- 空の行列の拒否（第 1 引数 empty）----
// scale_build・expdelta_build が空の行列（K=0 と出力 0 枚）を止めるか（止めないと K=0 ではカーネルが読みの完了を待たずに
// 終わり，出力 0 枚ではコア数 0 で割る）．戻り値は終了コード
int run_empty() {
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto gd = md->compute_with_storage_grid_size();
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32), bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    auto f = mesh_dram_buffer(md, 1, 2 * f32t), m = mesh_dram_buffer(md, 1, bf16t);
    auto d = mesh_dram_buffer(md, 1, f32t), g = mesh_dram_buffer(md, gd.x * gd.y, f32t);
    int refused = 0;
    // 数えるのは，空の寸法を止める検査そのものの例外（文言に needle を含む．ozaki_scale.hpp の TT_FATAL）だけ．
    // 別の原因の例外（組み立ての失敗など）まで拒否と数えると，空の寸法の検査が無くても PASS になる
    auto expect_refusal = [&](const char* what, const char* needle, auto&& fn) {
        try { fn(); printf("NOT REFUSED: %s\n", what); }
        catch (const std::exception& ex) {
            const bool mine = std::strstr(ex.what(), needle) != nullptr &&
                              std::strstr(ex.what(), "空の行列は受け付けない") != nullptr;
            if (mine) ++refused;
            printf("%s %s: %.160s\n", mine ? "REFUSED" : "UNEXPECTED EXCEPTION", what, ex.what());
        }
    };
    expect_refusal("scale_build Kt=0", "scale_build: 出力",
                   [&] { (void)scale_build(md, 0u, 0u, f, m, 1u, 0u, false); });
    expect_refusal("scale_build nout=0", "scale_build: 出力",
                   [&] { (void)scale_build(md, 0u, 1u, f, m, 0u, 0u, false); });
    expect_refusal("expdelta_build ntiles=0", "expdelta_build: 出力",
                   [&] { (void)expdelta_build(md, m, d, g, 0u, true, 0); });
    printf("SCALE-CHECK empty refused=%d/3 %s\n", refused, refused == 3 ? "PASS" : "FAIL");
    const bool closed = md->close();
    return (refused == 3 && closed) ? 0 : 1;
}

// ---- 引数 ----
struct Args {
    uint32_t M = 0, N = 0, K = 0, ceil_mode = 0, seed = 1;
    int32_t shift = 0;
    std::string dump;   // 書き出しの接頭辞（第 7 引数．空なら書き出さない）
};
// 引数を読む．誤りは理由を出して false（終了コード 2．デバイスを開く前）
bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 7 || argc > 8) {
        fprintf(stderr, "usage: %s M N K ceil(0|1|2) shift(0|1 -> 0|-1) seed [dump_prefix]\n", argv[0]);
        return false;
    }
    a.M = oz_env::arg_u32(argc, argv, 1, 0); a.N = oz_env::arg_u32(argc, argv, 2, 0);
    a.K = oz_env::arg_u32(argc, argv, 3, 0);
    a.ceil_mode = oz_env::arg_u32(argc, argv, 4, 0);
    if (a.ceil_mode > 2) { fprintf(stderr, "ERROR: ceil=%u は 0・1・2\n", a.ceil_mode); return false; }
    const uint32_t shift_arg = oz_env::arg_u32(argc, argv, 5, 0);
    if (shift_arg > 1) { fprintf(stderr, "ERROR: shift=%u は 0（正規化の目標 2^0）か 1（2^-1）\n", shift_arg); return false; }
    a.shift = shift_arg ? -1 : 0;
    a.seed = oz_env::arg_u32(argc, argv, 6, 1);
    a.dump = argc > 7 ? argv[7] : "";
    return oz_eval::check_tile_dims(a.M, a.N, a.K, "scale_check");
}

// ---- 照合の数 ----
// 不一致（bad_max・bad_dlt）と，境目の網羅を確かめるための最大値の種類ごとの数（どれも A・B の両方を足す）
struct Counts {
    size_t bad_max = 0, bad_dlt = 0, nzero = 0, ntiny = 0, nceil_up = 0;
    size_t n126 = 0, nbelow = 0, nsub = 0;     // 下端の境目：ちょうど 2^-126，[2^-127, 2^-126)，FP64 の非正規化数
    size_t npow2 = 0;                          // CEIL=2 で，ちょうど 2 のべきのために 1 段上がった最大値（CEIL=1 との差）
    size_t nnonfin = 0;                        // Inf・NaN の最大値（印 0xFFC0）
    size_t ntop = 0, ntopb = 0, nabove = 0, ninf = 0;   // 上端の境目：ちょうど 2^128 - 2^120，その 1 ulp 下，その上で 2^128 未満，2^128 以上
};
// 片側（A は rowwise）の最大値タイルの有効 32 要素（A は各行の 0 列目 nf(k,0)，B は 0 行目の各列 nf(0,k)）と，指数差分
// タイルの全 1024 要素を参照と比べて n に足す．xm・xd は読み戻した最大値・指数差分，mx は参照の最大値（行・列ごと），nt はタイル数
void check_side(Counts& n, const std::vector<uint16_t>& xm, const std::vector<int32_t>& xd, const std::vector<double>& mx,
                uint32_t nt, bool rowwise, const char* nm, uint32_t ceil_mode, int32_t shift) {
    const bool bf_ceil = ceil_mode != 0;
    for (uint32_t t = 0; t < nt; ++t)
        for (uint32_t k = 0; k < 32; ++k) {
            const double m = mx[t * 32 + k];
            const uint16_t want = to_bf16(m, ceil_mode);
            n.nzero += want == 0;
            n.ntiny += want == 0xFF80u;
            n.nnonfin += want == 0xFFC0u;
            n.nceil_up += bf_ceil && want != to_bf16(m, 0u);
            n.npow2 += ceil_mode == 2 && want != to_bf16(m, 1u);
            n.n126 += m == std::ldexp(1.0, -126);
            n.nbelow += m >= std::ldexp(1.0, -127) && m < std::ldexp(1.0, -126);
            n.nsub += m > 0.0 && m < std::ldexp(1.0, -1022);
            n.ntop += m == kBf16Max;
            n.ntopb += m == std::nextafter(kBf16Max, 0.0);
            n.nabove += m > kBf16Max && m < std::ldexp(1.0, 128);
            n.ninf += std::isfinite(m) && m >= std::ldexp(1.0, 128);
            const uint16_t got = xm[(size_t)t * 1024 + (rowwise ? nf(k, 0) : nf(0, k))];
            if (got != want) {
                if (n.bad_max < 5) printf("MISMATCH %s max tile %u k=%u got 0x%04x want 0x%04x (max|x|=%.17g)\n",
                                          nm, t, k, got, want, m);
                ++n.bad_max;
            }
            const int32_t d = delta_of(want, shift);
            for (uint32_t j = 0; j < 32; ++j) {
                const int32_t gd2 = xd[(size_t)t * 1024 + (rowwise ? nf(k, j) : nf(j, k))];
                if (gd2 != d) {
                    if (n.bad_dlt < 5) printf("MISMATCH %s delta tile %u k=%u j=%u got %d want %d\n", nm, t, k, j, gd2, d);
                    ++n.bad_dlt;
                }
            }
        }
}

// 旗の期待値：2^-126 未満の行（列）があればビット 1，inf（2^128 以上）があればビット 0，Inf・NaN の行（列）があればビット 2
uint32_t want_flags(const std::vector<double>& mx, uint32_t ceil_mode) {
    uint32_t w = 0;
    for (double m : mx) {
        const uint16_t u = to_bf16(m, ceil_mode);
        w |= (u == 0xFF80u) ? 2u : (u == 0x7F80u) ? 1u : (u == 0xFFC0u) ? 4u : 0u;
    }
    return w;
}

// R×C（行優先）の X の行ごと（rowwise）か列ごとの |x| の最大値．カーネルの upd16 と同じく，符号を落とした 63 bit の
// 大小で取る（有限の非負値は値の順と同じで，NaN は inf より大きく並ぶ．std::max と fabs で取ると NaN を落としてしまう）
std::vector<double> max_abs(const std::vector<double>& X, uint32_t R, uint32_t C, bool rowwise) {
    auto bmaxabs = [](double cur, double x) {
        const uint64_t c = bits(cur), u = bits(x) & 0x7FFFFFFFFFFFFFFFull;
        return u > c ? from_bits(u) : cur;
    };
    std::vector<double> mx(rowwise ? R : C, 0.0);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t c = 0; c < C; ++c) {
            double& m = mx[rowwise ? r : c];
            m = bmaxabs(m, X[(size_t)r * C + c]);
        }
    return mx;
}

}  // namespace

int main(int argc, char** argv) {
    oz_knobs::reject_removed();
    if (argc == 2 && std::string(argv[1]) == "empty") return run_empty();
    Args a;
    if (!parse_args(argc, argv, a)) return 2;
    const uint32_t M = a.M, N = a.N, K = a.K, ceil_mode = a.ceil_mode, seed = a.seed;
    const int32_t shift = a.shift;
    const bool bf_ceil = ceil_mode != 0, above_pow2 = ceil_mode == 2;   // scale_build の bf_ceil・above_pow2
    g_tiny = oz_knobs::u32(OZK("SCALE_CHECK_TINY")) != 0;   // 0 か 1（既定 1．つまみの表）
    const uint32_t Mt = M / 32, Nt = N / 32, Kt = K / 32;

    // A（M×K，行ごと）と B（K×N，列ごと）．行（列）v の性質（kind 0..NKIND-1）は v の一次式を NKIND で割った余りで決める
    // （係数 3・7 は 10 と互いに素なので，どちらも NKIND 通りを順に回る）
    std::mt19937_64 rng(seed);
    std::vector<double> A((size_t)M * K), B((size_t)K * N);
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t k = 0; k < K; ++k) A[(size_t)i * K + k] = element((i * 3u + seed) % NKIND, i, k, K, rng);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t j = 0; j < N; ++j) B[(size_t)k * N + j] = element((j * 7u + seed) % NKIND, j, k, K, rng);
    const std::vector<double> amax = max_abs(A, M, K, true), bmax = max_abs(B, K, N, false);

    auto md = distributed::MeshDevice::create_unit_mesh(0);
    auto gd = md->compute_with_storage_grid_size();
    auto& cq = md->mesh_command_queue();
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32), bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    auto fA = mesh_dram_buffer(md, (size_t)Mt * Kt, 2 * f32t), fB = mesh_dram_buffer(md, (size_t)Kt * Nt, 2 * f32t);
    auto mA = mesh_dram_buffer(md, Mt, bf16t), mB = mesh_dram_buffer(md, Nt, bf16t);
    auto dA = mesh_dram_buffer(md, Mt, f32t), dB = mesh_dram_buffer(md, Nt, f32t);
    const uint32_t nslot = gd.x * gd.y;
    auto gA = mesh_dram_buffer(md, nslot, f32t), gB = mesh_dram_buffer(md, nslot, f32t);
    {   // 出力の器を番兵で埋めておく（書かれなかったタイル・要素を見分ける）
        std::vector<uint32_t> s1((size_t)Mt * bf16t / 4, 0xA5A5A5A5u), s2((size_t)Nt * bf16t / 4, 0xA5A5A5A5u);
        std::vector<uint32_t> s3((size_t)Mt * f32t / 4, 0x5A5A5A5Au), s4((size_t)Nt * f32t / 4, 0x5A5A5A5Au);
        std::vector<uint32_t> z((size_t)nslot * f32t / 4, 0);
        distributed::EnqueueWriteMeshBuffer(cq, mA, s1, false); distributed::EnqueueWriteMeshBuffer(cq, mB, s2, false);
        distributed::EnqueueWriteMeshBuffer(cq, dA, s3, false); distributed::EnqueueWriteMeshBuffer(cq, dB, s4, false);
        distributed::EnqueueWriteMeshBuffer(cq, gA, z, false); distributed::EnqueueWriteMeshBuffer(cq, gB, z, false);
        distributed::Finish(cq);
    }
    upload_operand_f64(md, A, M, K, fA);
    upload_operand_f64(md, B, K, N, fB);
    // 本番と同じ組み方（ozaki1_main.cpp の run_scale_slice・ozaki2_main.cpp の scale_and_slice）
    auto edA = expdelta_build(md, mA, dA, gA, Mt, true, shift);
    auto edB = expdelta_build(md, mB, dB, gB, Nt, false, shift);
    auto scA = scale_build(md, 0u, Kt, fA, mA, Mt, 0u, bf_ceil, above_pow2);
    auto scB = scale_build(md, 1u, Kt, fB, mB, Nt, Nt, bf_ceil, above_pow2);
    scale_run(md, scA); scale_run(md, scB); distributed::Finish(cq);
    expdelta_run(md, edA); expdelta_run(md, edB); distributed::Finish(cq);

    std::vector<uint16_t> xmA, xmB; std::vector<int32_t> xdA, xdB;
    distributed::EnqueueReadMeshBuffer(cq, xmA, mA, true); distributed::EnqueueReadMeshBuffer(cq, xmB, mB, true);
    distributed::EnqueueReadMeshBuffer(cq, xdA, dA, true); distributed::EnqueueReadMeshBuffer(cq, xdB, dB, true);
    bool dump_ok = true;   // 書き出し（第 7 引数）．開けない・書けない・閉じられなければ失敗にする（古いファイルを残して PASS としない）
    if (!a.dump.empty()) {
        auto w = [&](const std::string& nm, const void* p, size_t n) {
            const std::string path = a.dump + nm;
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (f) f.write((const char*)p, (std::streamsize)n);
            if (f) f.close();
            if (!f) { fprintf(stderr, "ERROR: 書き出せない：%s\n", path.c_str()); dump_ok = false; }
        };
        w("_maxA.bin", xmA.data(), xmA.size() * 2); w("_maxB.bin", xmB.data(), xmB.size() * 2);
        w("_dltA.bin", xdA.data(), xdA.size() * 4); w("_dltB.bin", xdB.data(), xdB.size() * 4);
    }
    Counts n;
    check_side(n, xmA, xdA, amax, Mt, true, "A", ceil_mode, shift);
    check_side(n, xmB, xdB, bmax, Nt, false, "B", ceil_mode, shift);
    // RISCV_0 が受け持つ枚数（scale_build と同じ配り方）
    auto per0 = [&](uint32_t nout) {
        auto gs = full_grid_span(md, nout);
        uint32_t mn = UINT32_MAX, mxv = 0, cores0 = 0;
        for (size_t c = 0; c < gs.cores.size(); ++c) {
            const uint32_t nt = gs.count(c), n0 = nt - (nt + 1) / 2;
            mn = std::min(mn, n0); mxv = std::max(mxv, n0); cores0 += n0 > 0;
        }
        return std::to_string(gs.per) + "/core,riscv0=" + std::to_string(mn) + ".." + std::to_string(mxv) +
               " on " + std::to_string(cores0) + " cores";
    };
    const uint32_t gfA = expdelta_flags(md, edA), gfB = expdelta_flags(md, edB);
    const uint32_t wfA = want_flags(amax, ceil_mode), wfB = want_flags(bmax, ceil_mode);
    const bool flag_ok = gfA == wfA && gfB == wfB;
    // 境目の網羅（行（列）が足りる大きさのときだけ）：TINY=1 なら下端の 3 通りと上端の 4 通りとも，TINY=0 なら印・旗に
    // ならないもの（2^-126 ちょうど，2^128 - 2^120 ちょうどとその 1 ulp 下）だけがある
    const bool big = M + N >= 32u * NKIND * 4u;
    const bool cover_ok = !big || (g_tiny ? (n.n126 > 0 && n.nbelow > 0 && n.nsub > 0 && n.nnonfin > 0)
                                          : (n.n126 > 0 && n.nbelow == 0 && n.nsub == 0 && n.nnonfin == 0));
    const bool top_ok = !big || (n.ntop > 0 && n.ntopb > 0 && (g_tiny ? (n.nabove > 0 && n.ninf > 0) : (n.nabove == 0 && n.ninf == 0)));
    // CEIL=2 では，ちょうど 2 のべきの最大値（kind 6 など）を 1 段上げる分岐を実際に踏んだことも条件にする
    const bool pow2_ok = ceil_mode != 2 || n.npow2 > 0;
    const bool pass = n.bad_max == 0 && n.bad_dlt == 0 && flag_ok && cover_ok && top_ok && pow2_ok && dump_ok;
    printf("SCALE-CHECK M=%u N=%u K=%u ceil=%d shift=%d seed=%u tiny=%d A:%s B:%s zero_max=%zu tiny_max=%zu nonfinite_max=%zu ceil_raised=%zu pow2_raised=%zu "
           "edge126_max=%zu below126_max=%zu f64sub_max=%zu top_max=%zu topm1_max=%zu above_top_max=%zu inf_max=%zu "
           "max_mismatch=%zu/%u delta_mismatch=%zu/%u flags=A%u/%u,B%u/%u(got/want)%s%s%s%s%s %s\n",
           M, N, K, (int)ceil_mode, shift, seed, (int)g_tiny, per0(Mt).c_str(), per0(Nt).c_str(), n.nzero, n.ntiny, n.nnonfin,
           n.nceil_up, n.npow2, n.n126, n.nbelow, n.nsub, n.ntop, n.ntopb, n.nabove, n.ninf,
           n.bad_max, 32u * (Mt + Nt), n.bad_dlt, 1024u * (Mt + Nt), gfA, wfA, gfB, wfB,
           flag_ok ? "" : " FLAG-MISMATCH", cover_ok ? "" : " EDGE-NOT-COVERED", top_ok ? "" : " TOP-NOT-COVERED",
           pow2_ok ? "" : " POW2-NOT-COVERED", dump_ok ? "" : " DUMP-FAILED",
           pass ? "PASS" : "FAIL");
    const bool closed = md->close();
    return (pass && closed) ? 0 : 1;
}
