// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎スキーム II：診断 OZ2_CHECK_CONCAT=1【診断】連結した GEMM の出力・剰余・CRT 復元を独立な整数の参照と全要素で比べる
// （桁分割の経路の ozaki2_main.cpp が取り込む．scripts/tests/check_concat.sh が回す．呼ぶのは ozaki2_main.cpp の run_digit_modulus・
// run_digit_gemms・read_back_and_verify）
// ---------------------------------------------------------------------------
// 同じ位の桁積を K 方向に連結した GEMM（ozaki2_split.h の oz2_product_groups・スライスのライタの LAYOUT_A_CAT・LAYOUT_B_PERM）は，
// 出力 1 本が g 本の桁積の和 Σ_{t<g} A_{a0+t} B_{key-a0-t} になる．デバイスのスライス出力（桁行列）を
// 読み戻して桁の整数に直し，その和をホストで int64 で作って，桁積の平面を全要素で比べる．参照は桁の
// 定義（A の桁 a と B の桁 key-a）から作るので，B を法の中で逆順に積む置き方（桁 j を位置 np-1-j）と
// GEMM の開始位置・K の長さ（g*K）の両方を確かめる．答えは変えない（読むだけ）．1 パネルの構成だけ．
// 出力は法・塊ごとに CONCAT-CHECK 行（最後に合計）．和の上界 g*K*d^2（+法 <= 2^24 が maxg の条件）と
// 実際の最大 |Σ| も出す．
// 上の比べは，デバイスが書いた桁行列と本番と同じ積の列挙 gr を正解側にも使うので，スライサの合同式や積の列挙・
// 合成の係数そのものの誤りは捕まえられない（桁が丸まっても，丸まった桁どうしの GEMM は通る）．そこで本番の積の列挙と
// 合成を使わずに作った整数の参照（concat_ref_* と concat_residue_check・concat_crt_check の前の注記）と段ごとに比べる．
// この独立な参照は本番の積の列挙へ統合しない（検査の独立性を保つ）．
// 検査が読む実行の状態は Oz2ConcatRun（ドライバが Oz2Run から詰める）だけで，ドライバの内部の型には頼らない．
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <quadmath.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/float8.hpp>
#include "common/bfp2_decl.hpp"        // unpack_bfp2_tiles_into_float_vec（bfloat2.hpp の代わり．宣言だけ）
#include "common/kernels/tile_nf.h"    // nf（タイル内の要素の位置）
#include "common/oz_bits.h"            // f32_from_half
#include "ozaki2/ozaki2_moduli.h"
#include "ozaki2/kernels/oz2_args.h"   // スライスの置き方（Layout）

// 検査が読む実行の状態（ドライバの Oz2Run から呼ぶたびに詰める．スライス出力は法の組ごとに作り直される）
struct Oz2ConcatRun {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> md;
    uint32_t M, N, K, Nt, Kt, ntotA, ntotB, ntotC;
    int b;                    // 幅
    uint32_t ND;              // A' を運ぶ 2^20 進の桁数
    bool two_pass;            // 2 パス再構成
    const Ozaki2Moduli* mo;
    Oz2Route rt;              // 経路（桁の上限・桁の形）
    tt::DataFormat slice_fmt; // スライス出力の形式（OZ_FMT.slice_out）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> sliceA, sliceB;   // 今の組のスライス出力（桁行列）
    uint32_t layoutA, layoutB;                                               // その置き方（oz2_args.h の Layout）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltA, dltB;       // 行・列の指数差分
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> prod;             // 桁積の平面（GEMM の出力）
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> resid;            // 法ごとの剰余の平面（1 桁の法は GEMM の出力）
    const std::vector<double>* A;                                            // ホストの FP64 のオペランド
    const std::vector<double>* B;
};

struct Oz2ConcatCheck {
    bool ok = true;
    uint64_t outputs = 0, elems = 0, bad = 0;
    int64_t peak = 0;
    uint32_t grp = UINT32_MAX;               // 桁行列を読み込んである組
    std::vector<int8_t> Ad, Bd;              // 組の桁行列（A は [平面][M][K]，B は [平面][K][N]．平面はスライスの出力順）
    // 独立な整数参照の整数化（concat_ref_init・concat_ref_digits・concat_residue_check）．要素ごとに
    // X = x 2^(d + (ND-1)*20) の rint を基点に持ち，候補 c = rint(X) + δ（δ = -1,0,1 をビット δ+1）のうち
    // |c - X| <= 1 かつ |c| <= 2^b で，ここまでに見た法すべての桁行列の合成と合同なものを残す．
    bool ref_init = false, ref_skip = false;
    std::vector<int64_t> Abase, Bbase;       // rint(X)（|X| <= 2^b を確かめてから整数にする）
    std::vector<uint8_t> Acand, Bcand;       // 残っている候補．REF_RANGE は |X| <= 2^b を満たさない要素（整数にしない）
    uint64_t ref_range = 0, ref_kar = 0;     // |X| > 2^b か有限でない要素，Karatsuba の第 3 の桁が l + h でない要素
    static constexpr uint8_t REF_RANGE = 0x80;
    // 独立な整数参照の積 C' = A' B'（__int128）と，行・列の指数差分（concat_residue_check が作り，concat_crt_check が出口の C と
    // 比べる．ref_ready はそれが済んだか）
    std::vector<__int128> Cp;
    std::vector<int32_t> Da, Db;
    bool ref_ready = false;
};

// デバイスのバッファ b の全体をホストへ読む（EnqueueReadMeshBuffer は shared_ptr を非 const の参照で受けるので写しを渡す）
template <class T>
static void concat_read(const Oz2ConcatRun& r, std::vector<T>& dst, std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> b) {
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(r.md->mesh_command_queue(), dst, b, true);
}
// スライス出力（タイル順，形式 fmt）を要素ごとの浮動小数へ（タイル順のまま）
static std::vector<float> concat_decode_slice_tiles(tt::DataFormat fmt, const std::vector<uint32_t>& w) {
    std::vector<float> f;
    const tt::stl::Span<const uint32_t> sp(w.data(), w.size());
    switch (fmt) {
    case tt::DataFormat::Float32: f.resize(w.size()); std::memcpy(f.data(), w.data(), w.size() * 4); break;
    case tt::DataFormat::Float16_b: f.resize(w.size() * 2);
        for (size_t i = 0; i < w.size(); ++i) { uint32_t lo = (w[i] & 0xFFFFu) << 16, hi = w[i] & 0xFFFF0000u;
            std::memcpy(&f[2 * i], &lo, 4); std::memcpy(&f[2 * i + 1], &hi, 4); } break;
    case tt::DataFormat::UInt16: f.resize(w.size() * 2);   // FP16 経路：binary16 のビット列
        for (size_t i = 0; i < w.size(); ++i) { f[2 * i] = f32_from_half((uint16_t)w[i]); f[2 * i + 1] = f32_from_half((uint16_t)(w[i] >> 16)); } break;
    case tt::DataFormat::Fp8_e4m3: f.resize(w.size() * 4);
        for (size_t i = 0; i < w.size(); ++i)
            for (int b = 0; b < 4; ++b) f[4 * i + b] = (float)float8_e4m3::from_bits((uint8_t)(w[i] >> (8 * b)));
        break;
    case tt::DataFormat::Bfp8_b: f = unpack_bfp8_tiles_into_float_vec(sp, false, false); break;
    case tt::DataFormat::Bfp4_b: f = unpack_bfp4_tiles_into_float_vec(sp, false, false); break;
    case tt::DataFormat::Bfp2_b: f = unpack_bfp2_tiles_into_float_vec(sp, false, false); break;
    default: TT_FATAL(false, "OZ2_CHECK_CONCAT: スライスの形式 {} は読めない", (int)fmt);
    }
    return f;
}
static int8_t concat_digit_of(float v) {
    TT_FATAL(std::isfinite(v) && v == std::nearbyint(v) && std::fabs(v) <= 127.0f,
             "OZ2_CHECK_CONCAT: 桁 {} が整数でない", v);
    return (int8_t)v;
}
// 組 grp の桁行列を読み込む．A（行 R=M，列 K）は LAYOUT_A_CAT なら行ごとに平面を K 方向へ連結
// （タイル rt*(P*Kt) + p*Kt + c），LAYOUT_PLANES なら平面ごと（p*ntotA + rt*Kt + c）．B（行 K，列 N）は平面 p の
// タイルが p*ntotB + kt*Nt + nt（LAYOUT_B_PERM でも位置の並びは同じで，法の中の桁の順が逆になるだけ）．
static void concat_load(const Oz2ConcatRun& r, Oz2ConcatCheck& cc, uint32_t grp, uint32_t P) {
    if (cc.grp == grp) return;
    const uint32_t M = r.M, N = r.N, K = r.K, Nt = r.Nt, Kt = r.Kt;
    std::vector<uint32_t> wa, wb;
    concat_read(r, wa, r.sliceA);
    concat_read(r, wb, r.sliceB);
    const auto fa = concat_decode_slice_tiles(r.slice_fmt, wa), fb = concat_decode_slice_tiles(r.slice_fmt, wb);
    cc.Ad.assign((size_t)P * M * K, 0); cc.Bd.assign((size_t)P * K * N, 0);
    const bool catA = r.layoutA == oz2_args::LAYOUT_A_CAT;
#pragma omp parallel for
    for (int64_t p = 0; p < (int64_t)P; ++p)
        for (uint32_t i = 0; i < M; ++i)
            for (uint32_t k = 0; k < K; ++k) {
                const size_t t = catA ? ((size_t)(i / 32) * (P * Kt) + (size_t)p * Kt + k / 32)
                                      : ((size_t)p * r.ntotA + (size_t)(i / 32) * Kt + k / 32);
                cc.Ad[((size_t)p * M + i) * K + k] = concat_digit_of(fa[t * 1024 + nf(i % 32, k % 32)]);
            }
#pragma omp parallel for
    for (int64_t p = 0; p < (int64_t)P; ++p)
        for (uint32_t k = 0; k < K; ++k)
            for (uint32_t j = 0; j < N; ++j) {
                const size_t t = (size_t)p * r.ntotB + (size_t)(k / 32) * Nt + j / 32;
                cc.Bd[((size_t)p * K + k) * N + j] = concat_digit_of(fb[t * 1024 + nf(k % 32, j % 32)]);
            }
    cc.grp = grp;
}

// ---- 独立な整数参照 ----------------------------------------------------------------------------------------
//   整数化：ホストの FP64 の x とデバイスの指数差分 d から，正規化後の値 X = x 2^(d + (ND-1)*20)（|X| <= 2^b）を作る．
//   スライサが保証するのは |A' - X| <= 1 であって最近接性ではない（ozaki2/kernels/oz2_digits.h．丸めるのは 3 本組の
//   上位 limb だけなので，最後の桁で上位 limb がちょうど半整数なら，中位以下の符号によらず最近接偶数の側を選ぶ．
//   例えば b=54 で行の最大値が 1 の行の x = 2^-55 + 2^-80 は X = 1/2 + 2^-26 で，A' = 0，rint(X) = 1 になる）．
//   そこで A' を rint(X) とは決めず，|c - X| <= 1 かつ |c| <= 2^b の整数 c（高々 3 個）のうち，各法の桁行列の
//   合成 Σ_p w_p d_p（桁の重み w_p は concat_ref_digits）と合同なものを残す．候補どうしの差は 2 以下なので，
//   3 以上の法を 1 本見れば候補は高々 1 個になる．全法を見て 1 個だけ残った c を A' とし，残らない要素
//   （桁の合同式が崩れているか |A' - X| > 1）は不一致として数える．B' も同じ．C' = A' B'（__int128）．
//   (1) 剰余（concat_residue_check）：全法の GEMM と合成が済んだ剰余の平面 s が C' mod m_s と合同で，
//       合成した法では対称剰余から高々 1 はみ出すだけ（1 桁の法は GEMM の出力そのもので，上界 K R^2 の内側で
//       合同かを見る）．
//   (2) CRT 復元（concat_crt_check）：出口の FP64 の C が C' * 2^-(dA + dB + 2(ND-1)*20) に一致する
//       （frac の絶対誤差 2^-60 の M 倍と FP64 の丸め 2^-51 相対まで許す．±M ずれた答えや法 1 本の誤りは
//       桁違いに大きく外れる）．
// 候補を選ぶのにデバイスの桁行列を使うが，受け入れるのは契約 |A' - X| <= 1 の内側で全法が同じ整数を表す
// 桁だけである．(1) が確かめる積の列挙・合成の係数・GEMM は，この選び方に依らない．
// C' が __int128 に収まる 2b + log2 K <= 125 の構成だけ（既定の b <= 54 は収まる）．
// デバイスの指数差分（行ごとなら dltA の各タイルの列 0，列ごとなら dltB の行 0）を要素ごとに読む
static void concat_read_delta(const Oz2ConcatRun& r, bool rowwise, std::vector<int32_t>& d) {
    const uint32_t n = rowwise ? r.M : r.N;
    std::vector<int32_t> w;
    concat_read(r, w, rowwise ? r.dltA : r.dltB);
    d.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i)
        d[i] = w[(size_t)(i / 32) * 1024 + (rowwise ? nf(i % 32, 0) : nf(0, i % 32))];
}
// 整数化の基点 rint(X) と最初の候補を作る（最初の法の桁行列を見る前に 1 度だけ）．
static void concat_ref_init(const Oz2ConcatRun& r, Oz2ConcatCheck& cc) {
    cc.ref_init = true;
    if (2.0 * r.b + std::log2((double)r.K) > 125.0) { cc.ref_skip = true; return; }
    concat_read_delta(r, true, cc.Da); concat_read_delta(r, false, cc.Db);
    const uint32_t M = r.M, N = r.N, K = r.K;
    const int off = (int)(r.ND - 1) * OZ2_DIGIT_BITS;
    const int64_t lim = (int64_t)1 << r.b;       // 上の条件で b <= 60
    const double limd = std::ldexp(1.0, r.b);
    auto init = [&](double x, int sh, int64_t& base, uint8_t& cand) {
        const double X = std::ldexp(x, sh);
        // 正規化の前提 |X| <= 2^b を整数に直す前に見る（有限でない値もここで落ちる）．外れた要素は整数に直さない
        if (!(std::fabs(X) <= limd)) { base = 0; cand = Oz2ConcatCheck::REF_RANGE; return; }
        base = (int64_t)std::rint(X);
        cand = 0;
        for (int dl = -1; dl <= 1; ++dl) {
            const int64_t c = base + dl;
            if (c <= lim && c >= -lim && fabsq((__float128)c - (__float128)X) <= 1) cand |= (uint8_t)(1u << (dl + 1));
        }
    };
    cc.Abase.resize((size_t)M * K); cc.Acand.resize((size_t)M * K);
    cc.Bbase.resize((size_t)K * N); cc.Bcand.resize((size_t)K * N);
    const std::vector<double>& A = *r.A;
    const std::vector<double>& B = *r.B;
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)M; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            const size_t o = (size_t)i * K + k;
            init(A[o], cc.Da[i] + off, cc.Abase[o], cc.Acand[o]);
        }
#pragma omp parallel for
    for (int64_t k = 0; k < (int64_t)K; ++k)
        for (uint32_t j = 0; j < N; ++j) {
            const size_t o = (size_t)k * N + j;
            init(B[o], cc.Db[j] + off, cc.Bbase[o], cc.Bcand[o]);
        }
    for (uint8_t c : cc.Acand) cc.ref_range += (c == Oz2ConcatCheck::REF_RANGE);
    for (uint8_t c : cc.Bcand) cc.ref_range += (c == Oz2ConcatCheck::REF_RANGE);
}
// 法 s の桁行列で整数化の候補を絞る（法の最初の塊で 1 度）．桁の重みは w_p = q^p で，q は法の桁の基数
// （oz2_digit_base）：平方数・WIDE・疎な桁の平方数は (l, h) に 1, q，平衡 n 桁は q^p，1 桁の法は 1．
// Karatsuba は (l, h, s) のうち l + B h が値で，第 3 の桁行列 s は l + h そのもの（ozaki2_split.h の (B)）
// なので，重みには入れずに s = l + h を整数のまま確かめる．
static void concat_ref_digits(const Oz2ConcatRun& r, Oz2ConcatCheck& cc, uint32_t s, uint32_t po, uint32_t npl, bool revB) {
    if (!cc.ref_init) concat_ref_init(r, cc);
    if (cc.ref_skip) return;
    const uint32_t M = r.M, N = r.N, K = r.K;
    const int64_t m = r.mo->m[s];
    const bool kar = r.mo->mode[s] == OZ2_SPLIT_KAR;
    const int64_t q = oz2_digit_base((int)m, r.rt.digit_bound, r.rt.nd());
    const uint32_t nw = kar ? 2u : npl;
    constexpr uint32_t NPL_MAX = 8;   // 法 1 本の桁行列の枚数の上限（平衡 n 桁の n <= 5 を覆う）
    TT_FATAL(npl <= NPL_MAX && (!kar || npl == 3), "concat_ref_digits: 法 {} の桁行列が {} 枚", m, npl);
    int64_t w[NPL_MAX]; w[0] = 1;
    for (uint32_t p = 1; p < nw; ++p) w[p] = w[p - 1] * q;
    uint32_t pa[NPL_MAX], pb[NPL_MAX];
    for (uint32_t p = 0; p < npl; ++p) { pa[p] = po + p; pb[p] = po + (revB ? (npl - 1 - p) : p); }
    uint64_t kbad = 0;
    auto narrow = [&](const std::vector<int8_t>& D, size_t n, const uint32_t* pl,
                      const std::vector<int64_t>& base, std::vector<uint8_t>& cand) {
#pragma omp parallel for reduction(+ : kbad)
        for (int64_t e = 0; e < (int64_t)n; ++e) {
            if (cand[e] == Oz2ConcatCheck::REF_RANGE) continue;
            int64_t v = 0;
            for (uint32_t p = 0; p < nw; ++p) v += w[p] * D[(size_t)pl[p] * n + e];
            if (kar && D[(size_t)pl[2] * n + e] != D[(size_t)pl[0] * n + e] + D[(size_t)pl[1] * n + e]) ++kbad;
            for (int dl = -1; dl <= 1; ++dl)
                if ((base[e] + dl - v) % m != 0) cand[e] &= (uint8_t)~(1u << (dl + 1));
        }
    };
    narrow(cc.Ad, (size_t)M * K, pa, cc.Abase, cc.Acand);
    narrow(cc.Bd, (size_t)K * N, pb, cc.Bbase, cc.Bcand);
    cc.ref_kar += kbad;
}

// 法 s の出力 j0..j0+jn-1（桁積の平面 j*ntotC）を参照と比べる．po は組の中での法 s の先頭の平面，
// npl は法 s の桁行列の枚数．B の桁 b は，LAYOUT_B_PERM なら位置 po+(npl-1-b)，LAYOUT_PLANES なら位置 po+b に置かれている．
// raw（1 桁の法）は GEMM の出力を合成段を通さず剰余の平面 s へ書くので，そちらを読む．
static void concat_check_chunk(const Oz2ConcatRun& r, Oz2ConcatCheck& cc, const Oz2ProductGroup* gr, uint32_t j0, uint32_t jn,
                               uint32_t po, uint32_t npl, uint32_t s, bool raw) {
    const Ozaki2Moduli& mo = *r.mo;
    const uint32_t M = r.M, N = r.N, K = r.K, Nt = r.Nt;
    std::vector<float> D;
    concat_read(r, D, raw ? r.resid : r.prod);
    const size_t plane0 = raw ? (size_t)s : 0;
    const bool revB = r.layoutB == oz2_args::LAYOUT_B_PERM;
    uint64_t bad = 0; int64_t peak = 0; int gmax = 1; double bound = 0;
    for (uint32_t jj = 0; jj < jn; ++jj) {
        const Oz2ProductGroup& q = gr[j0 + jj];
        gmax = std::max(gmax, q.g);
        // 和の上界（連結した g 本の桁の上限の積の和．smax=0 の経路では g*K*d^2．ozaki2_split.h の oz2_group_bound）
        bound = std::max(bound, oz2_group_bound(mo.m[s], q, (double)K, r.rt.digit_bound, r.rt.nd(), mo.smax));
        // 出力 jj = Σ_t A の桁 (a0+t) × B の桁 (key-a0-t)．gr の b0 は「B の位置」なので桁に直す：
        // LAYOUT_B_PERM では位置 np-1-桁，それ以外（maxg=1）はドライバの GEMM の発行と同じく桁 np-1-b0 を読む
        std::vector<int64_t> ref((size_t)M * N, 0);
        for (int t = 0; t < q.g; ++t) {
            const uint32_t pa = po + (uint32_t)(q.a0 + t);
            const uint32_t bdig = (uint32_t)(npl - 1 - (q.b0 + t));   // B の桁の番号（位 key = a + b）
            const uint32_t pb = po + (revB ? (npl - 1 - bdig) : bdig);
            const int8_t* Ap = &cc.Ad[(size_t)pa * M * K];
            const int8_t* Bp = &cc.Bd[(size_t)pb * K * N];
#pragma omp parallel for
            for (int64_t i = 0; i < (int64_t)M; ++i)
                for (uint32_t k = 0; k < K; ++k) {
                    const int64_t a = Ap[(size_t)i * K + k];
                    if (!a) continue;
                    for (uint32_t j = 0; j < N; ++j) ref[(size_t)i * N + j] += a * Bp[(size_t)k * N + j];
                }
        }
        for (uint32_t i = 0; i < M; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                const size_t off = ((plane0 + jj) * r.ntotC + (size_t)(i / 32) * Nt + j / 32) * 1024 + nf(i % 32, j % 32);
                const double v = (double)D[off];
                const int64_t want = ref[(size_t)i * N + j];
                peak = std::max<int64_t>(peak, want < 0 ? -want : want);
                if (!(std::isfinite(v) && v == (double)want)) {
                    if (bad < 3) printf("CONCAT-MISMATCH m=%d out %u (%u,%u) got %.1f want %lld\n", mo.m[s], j0 + jj, i, j,
                                        v, (long long)want);
                    ++bad;
                }
            }
    }
    printf("CONCAT-CHECK m=%d outputs=%u..%u gmax=%d maxg=%d K=%u peak|D|=%lld bound g*K*d^2=%.0f (+m %s 2^24) mismatches=%llu/%llu\n",
           mo.m[s], j0, j0 + jn - 1, gmax, mo.maxg_m[s], K, (long long)peak, bound,
           bound + mo.m[s] <= std::ldexp(1.0, 24) ? "<=" : ">", (unsigned long long)bad, (unsigned long long)jn * M * N);
    // 桁行列そのものの検査（法の最初の塊でだけ）：桁が形式で無誤差に表せる整数（FP8 なら e4m3．疎な桁の
    // 法 SQS では 32 以下の偶数を含む）で，法・桁行列ごとの桁の上限（oz2_plane_bound．上の上界の根拠）に
    // 収まるか．A と B の両方の全要素を見る．桁の位置は A も B も po + (LAYOUT_B_PERM の B は逆順の) 桁．
    if (j0 == 0) {
        uint64_t dbad = 0; int dmaxA = 0, dmaxB = 0;
        for (uint32_t d = 0; d < npl; ++d) {
            const int lim = oz2_plane_bound(mo.m[s], (int)d, r.rt.digit_bound, r.rt.nd(), mo.smax);
            const uint32_t pa = po + d, pb = po + (revB ? (npl - 1 - d) : d);
            auto scan = [&](const int8_t* p, size_t n, int& mx) {
                for (size_t i = 0; i < n; ++i) {
                    const int v = p[i], a = v < 0 ? -v : v;
                    mx = std::max(mx, a);
                    const bool rep_ok = r.slice_fmt == tt::DataFormat::Fp8_e4m3 ? oz2_e4m3_int(v) : true;
                    if (a > lim || !rep_ok) { if (dbad < 3) printf("CONCAT-DIGIT m=%d plane %u digit %d (limit %d)\n",
                                                                   mo.m[s], d, v, lim); ++dbad; }
                }
            };
            scan(&cc.Ad[(size_t)pa * M * K], (size_t)M * K, dmaxA);
            scan(&cc.Bd[(size_t)pb * K * N], (size_t)K * N, dmaxB);
        }
        printf("CONCAT-DIGITS m=%d mode=%d max|digit| A=%d B=%d bad=%llu\n", mo.m[s], mo.mode[s], dmaxA, dmaxB,
               (unsigned long long)dbad);
        if (dbad) { cc.bad += dbad; cc.ok = false; }
        concat_ref_digits(r, cc, s, po, npl, revB);   // 独立な整数参照の候補を法 s の桁行列の合同式で絞る
    }
    cc.outputs += jn; cc.elems += (uint64_t)jn * M * N; cc.bad += bad; cc.peak = std::max(cc.peak, peak);
    if (bad) cc.ok = false;
}

static void concat_residue_check(const Oz2ConcatRun& r, Oz2ConcatCheck& cc) {
    const Ozaki2Moduli& mo = *r.mo;
    const uint32_t M = r.M, N = r.N, K = r.K, Nt = r.Nt;
    if (!cc.ref_init) concat_ref_init(r, cc);
    if (cc.ref_skip) {
        printf("CONCAT-RESIDUE skipped（2b + log2 K = %.1f > 125 で C' が __int128 に収まらない）\n",
               2.0 * r.b + std::log2((double)K));
        return;
    }
    // 候補が 1 個だけ残った要素はその整数に，残らない要素は 0 にして不一致に数える
    uint64_t nochoice = 0, offnear = 0;
    auto pick = [&](const std::vector<int64_t>& base, const std::vector<uint8_t>& cand, std::vector<__int128>& out) {
        out.assign(base.size(), 0);
        for (size_t e = 0; e < base.size(); ++e) {
            const uint8_t c = cand[e];
            if (c == 1u || c == 2u || c == 4u) {
                const int dl = c == 1u ? -1 : (c == 2u ? 0 : 1);
                out[e] = (__int128)base[e] + dl;
                offnear += (dl != 0);
            } else if (c != Oz2ConcatCheck::REF_RANGE) ++nochoice;
        }
    };
    std::vector<__int128> Ap, Bp;
    pick(cc.Abase, cc.Acand, Ap); pick(cc.Bbase, cc.Bcand, Bp);
    printf("CONCAT-RESIDUE integers |X|>2^b=%llu no-unique-candidate=%llu karatsuba-s!=l+h=%llu not-rint(X)=%llu"
           "（|A'-X| <= 1 の内側で rint(X) と違う整数を選んだ要素．不一致ではない）\n",
           (unsigned long long)cc.ref_range, (unsigned long long)nochoice, (unsigned long long)cc.ref_kar,
           (unsigned long long)offnear);
    cc.Cp.assign((size_t)M * N, 0);
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)M; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            const __int128 a = Ap[(size_t)i * K + k];
            if (!a) continue;
            for (uint32_t j = 0; j < N; ++j) cc.Cp[(size_t)i * N + j] += a * Bp[(size_t)k * N + j];
        }
    std::vector<float> Rv;
    concat_read(r, Rv, r.resid);
    uint64_t bad_all = 0;
    for (int s = 0; s < mo.s; ++s) {
        const long m = mo.m[s];
        const bool raw = mo.mode[s] == OZ2_SPLIT_ONE;
        // 剰余の平面の値の許容範囲：合成した法は対称剰余から高々 1 はみ出すだけ（floor(m/2)+1），
        // 1 桁の法は GEMM の出力そのものなので桁積の和の上界 K R^2（oz2_group_bound．2^24 以下）
        double vmax = (double)(m / 2 + 1);
        if (raw) {
            Oz2ProductGroup g1[OZ2_MAXPROD];
            oz2_product_groups((int)m, g1, r.rt.digit_bound, r.rt.nd(), mo.maxg_m[s]);
            vmax = oz2_group_bound((int)m, g1[0], (double)K, r.rt.digit_bound, r.rt.nd(), mo.smax);
        }
        uint64_t bad = 0, wide = 0;
        for (uint32_t i = 0; i < M; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                const size_t o = ((size_t)s * r.ntotC + (size_t)(i / 32) * Nt + j / 32) * 1024 + nf(i % 32, j % 32);
                const float v = Rv[o];
                // 有限・整数・許容範囲を整数に直す前に見る．外れた値は変換せずに不一致とする
                const bool inr = std::isfinite(v) && v == std::nearbyint(v) && std::fabs((double)v) <= vmax;
                const bool ok = inr && ((__int128)(int64_t)v - cc.Cp[(size_t)i * N + j]) % m == 0;
                if (!ok) {
                    if (bad < 3) printf("CONCAT-RESIDUE-MISMATCH m=%ld (%u,%u) got %.1f want ≡ %lld (mod m)%s\n", m, i, j,
                                        (double)v, (long long)(cc.Cp[(size_t)i * N + j] % m), inr ? "" : "（許容範囲の外）");
                    ++bad; wide += !inr;
                }
            }
        printf("CONCAT-RESIDUE m=%ld mode=%d %s |v|<=%.0f mismatches=%llu/%llu（うち許容範囲の外 %llu）\n", m, mo.mode[s],
               raw ? "raw" : "composed", vmax, (unsigned long long)bad, (unsigned long long)M * N, (unsigned long long)wide);
        bad_all += bad;
    }
    const uint64_t bad_int = cc.ref_range + nochoice + cc.ref_kar;
    printf("CONCAT-RESIDUE total moduli=%d elements=%llu integer-failures=%llu mismatches=%llu %s\n", mo.s,
           (unsigned long long)M * N * mo.s, (unsigned long long)bad_int, (unsigned long long)bad_all,
           (bad_all || bad_int) ? "FAIL" : "PASS");
    if (bad_all || bad_int) { cc.bad += bad_all + bad_int; cc.ok = false; }
    cc.ref_ready = true;
}
// CRT-CHECK が照合できる構成か．許容 M 2^-60 は frac を limb 3 本で持つ構成の精度である．1 パス構成で b < 24 のときは
// limb 2 本（ozaki2_moduli.h の oz2_recon_limb3．FP32 級の到達精度で足りる構成）で，frac の絶対誤差がこの前提の外になるので
// 比べない
static bool concat_crt_ref_runs(const Oz2ConcatRun& r) { return oz2_recon_limb3(*r.mo, r.two_pass); }
// 出口の FP64 の C（行優先）を C' * 2^-(dA + dB + 2(ND-1)*20) と比べる．外れた要素の数を返す．
static uint64_t concat_crt_check(const Oz2ConcatRun& r, const Oz2ConcatCheck& cc, const std::vector<double>& C) {
    const uint32_t M = r.M, N = r.N;
    if (!concat_crt_ref_runs(r)) {
        printf("CRT-CHECK skipped（b=%d の 1 パス構成は frac を limb 2 本で持ち，許容 M 2^-60 の前提の外）\n", r.mo->b);
        return 0;
    }
    const int off = (int)(r.ND - 1) * OZ2_DIGIT_BITS;
    __float128 Mq = 1; for (int mm : r.mo->m) Mq *= (__float128)mm;
    const __float128 tolM = Mq * powq(2.0Q, -60);
    uint64_t bad = 0; double worst = 0;
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            const __float128 cp = (__float128)cc.Cp[(size_t)i * N + j];
            const __float128 got = ldexpq((__float128)C[(size_t)i * N + j], cc.Da[i] + cc.Db[j] + 2 * off);
            const __float128 err = fabsq(got - cp), tol = tolM + fabsq(cp) * powq(2.0Q, -51);
            worst = std::max(worst, (double)(err / tol));
            if (!(err <= tol)) {
                if (bad < 3) printf("CRT-CHECK-MISMATCH (%u,%u) got %.17g want %.17g\n", i, j, C[(size_t)i * N + j],
                                    (double)ldexpq(cp, -(cc.Da[i] + cc.Db[j] + 2 * off)));
                ++bad;
            }
        }
    printf("CRT-CHECK elements=%llu worst|err|/tol=%.3g (tol = M 2^-60 + |C'| 2^-51) mismatches=%llu %s\n",
           (unsigned long long)M * N, worst,
           (unsigned long long)bad, bad ? "FAIL" : "PASS");
    return bad;
}

// ---- ドライバが呼ぶ 3 つの入口 ----------------------------------------------------------------------------
// GEMM の塊（法 s の出力 j0..j0+jn-1）が済んだところで：組 grp の桁行列（P 枚）を読み，塊の出力を参照と比べる
inline void concat_after_chunk(const Oz2ConcatRun& r, Oz2ConcatCheck& cc, uint32_t grp, uint32_t P, uint32_t s,
                               const Oz2ProductGroup* gr, uint32_t j0, uint32_t jn, uint32_t po, bool raw) {
    tt::tt_metal::distributed::Finish(r.md->mesh_command_queue());
    concat_load(r, cc, grp, P);
    const uint32_t npl = (uint32_t)oz2_digit_planes(r.mo->m[s], r.rt.digit_bound, r.rt.nd());
    concat_check_chunk(r, cc, gr, j0, jn, po, npl, s, raw);
}
// 全法の GEMM と合成が済んだところで：独立な整数参照と剰余を比べ，合計の行を出す．外れたらデバイスを閉じて終了コード 1
inline void concat_after_compose(const Oz2ConcatRun& r, Oz2ConcatCheck& cc) {
    tt::tt_metal::distributed::Finish(r.md->mesh_command_queue());
    concat_residue_check(r, cc);   // 独立な整数参照との剰余の比較（全法の合成の後）
    // 独立な参照を飛ばした構成は，判定の後ろに「reference skipped: 何を・なぜ」を付ける（行は PASS で終わらなくなる．
    // scripts/tests/check_concat.sh は PASS で終わる行だけを通すので，参照を回すはずの構成で飛ばせば FAIL になる）．
    // CRT-CHECK は出口の C を読んだ後（concat_after_output）に走るが，走るかどうかはここで決まっている
    char skip[256] = "";
    if (cc.ref_skip)
        snprintf(skip, sizeof skip, "（reference skipped: 剰余と CRT．2b + log2 K = %.1f > 125 で C' が __int128 に収まらない）",
                 2.0 * r.b + std::log2((double)r.K));
    else if (!concat_crt_ref_runs(r))
        snprintf(skip, sizeof skip, "（reference skipped: CRT．b=%d の 1 パス構成は frac を limb 2 本で持つ）", r.mo->b);
    printf("CONCAT-CHECK total outputs=%llu elements=%llu peak|D|=%lld mismatches=%llu %s%s\n",
           (unsigned long long)cc.outputs, (unsigned long long)cc.elems, (long long)cc.peak,
           (unsigned long long)cc.bad, cc.ok ? "PASS" : "FAIL", skip);
    if (!cc.ok) { fprintf(stderr, "ERROR: OZ2_CHECK_CONCAT で連結した GEMM の出力が参照と違う\n"); r.md->close(); std::exit(1); }
}
// 出口の C（行優先）を読み戻したところで：独立な整数参照 C' と比べる．外れたらデバイスを閉じて終了コード 1
inline void concat_after_output(const Oz2ConcatRun& r, const Oz2ConcatCheck& cc, const std::vector<double>& C) {
    if (cc.ref_ready && concat_crt_check(r, cc, C)) {
        fprintf(stderr, "ERROR: OZ2_CHECK_CONCAT で CRT 復元の C が整数の参照と違う\n"); r.md->close(); std::exit(1); }
}
