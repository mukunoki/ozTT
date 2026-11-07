// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// tools のホスト単体の検査・模擬が共有する部品（本番のドライバは取り込まない）
// ---------------------------------------------------------------------------
// 検査ごとに本番の値や手続きを書き写すと，本番だけを直したときに検査が古い写しのまま通る．本番の表・関数は
// tt-metal に依存しないヘッダから取り込んで使う：形式の表（common/oz_format_host.h），スキーム I の基数と桁の上限
// （ozaki1/ozaki1_param.h の oz1_param・oz1_default_digit_max），スキーム II の経路の表と法の選択・断片の本数・limb の本数
// （ozaki2/ozaki2_moduli.h が取り込む ozaki2_split.h の OZ2_ROUTES・oz2_param・oz2_in_residue と，ozaki2_moduli.h の
// oz2_choose_small_tail・oz2_recon_limb3・oz2_recon_rawmask），FP64 と triple-float の変換（f64_tf.h）．SFPU の演算の模擬と
// triple-float・法の剰余の手続き（本番と同じ common/kernels/sfpu_tf_proc.h）は checks/sfpu_tf_emu.hpp．ここに置くのは次のものだけ：
//   ・つまみの読み方（env_flag01．名前・範囲はつまみの表 common/oz_knobs.hpp）
//   ・経路の表 scripts/routes.tsv のスキーム I の行（p・L）を読み，本番の基数・桁の上限と組にする（load_routes1・scheme1_configs．
//     本番の 8 形式がちょうど 1 回ずつ現れることも見る）．スキーム II の行の法の本数が本番の既定と同じことを確かめる
//     （check_routes2_def_s．回すのは tools/routes_check.cpp）
//   ・スキーム II の経路を表示名で引く（route2）．FP16 の行を BF16 の行で兼ねるか（fp16_same_as_bf16）
//   ・行・列の 2 のべきのスケールの指数（exp_ceil．splitsim と study/oz2_quantsim が同じ量を求める）
//   ・再構成 ozaki2/kernels/recon_crt.cpp の 1 要素（crt_*．法ごとの k_i・断片の足し込み・整数部の削除は本番と同じ
//     ozaki2/kernels/recon_crt_proc.h のテンプレートを Dst の上で組む．写しは入力の読み込み crt_load と法のループの並び
//     crt_element だけ．tools/README.md の「本番の写し」）と，ozaki2_main.cpp の run_recon_pass がカーネルへ渡す定数の組
//     （make_crt_kernel．コンパイル時引数は本番と同じ ozaki2_moduli.h の oz2_recon_ct で決める）
// ビルドは他の道具と同じく src を取り込み先に入れる（-Isrc．src/Makefile の HOSTRULE は src から -I.）．
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <strings.h>
#include <utility>
#include <vector>
#include "common/oz_format_host.h"
#include "ozaki1/ozaki1_param.h"
#include "ozaki2/ozaki2_moduli.h"
#include "checks/sfpu_tf_emu.hpp"
#include "ozaki2/kernels/recon_crt_proc.h"   // 再構成の法ごとの手続き（本番の recon_crt.cpp と同じテンプレート）
#include "common/oz_env.hpp"   // つまみの読み方（ドライバと同じ 1 系統）
#include "common/oz_knobs.hpp"   // つまみの表（env_flag01 の名前と範囲）

namespace oz_sim {

// ---- つまみ ----------------------------------------------------------------------------------------
// 0/1 で読む：未設定と 0 は偽，1 は真，それ以外（空・"yes"・"2" など）は打ち間違いとして止める（終了コード 2．
// 読み方は oz_env.hpp，名前・範囲・既定はつまみの表 src/common/oz_knobs.hpp の F の行で，ここに範囲を書き写さない）．
// 有無だけで読むと，=0 と書いても比較用の版が有効になる．表に無い名前を渡すのは道具の誤り（終了コード 2）
inline bool env_flag01(const char* name) {
    const int id = oz_knobs::index_of(name);
    if (id < 0) {
        std::fprintf(stderr, "ERROR: つまみ %s はつまみの表（src/common/oz_knobs.hpp）に無い（道具の誤り）\n", name);
        std::exit(2);
    }
    return oz_knobs::u32(id) != 0;
}

// ---- 2 のべきのスケール ----------------------------------------------------------------------------------
// 2^e >= m となる最小の e（m は 0 より大きい有限な数）．ilogb と 2 のべきとの厳密な比較で求める（ceil(log2(m)) は
// 2 のべきの直上の数個の double で log2 が整数へ丸まり，m より小さい 2^e を返す．例えば m = 16(1 + 2^-52)）．
// スケール段（scale_f64_reader.cpp・expdelta.cpp）は最大値を BF16 へ切り上げてから同じ e を取る（尾崎 II は CEIL を立てる．
// ozaki2_main.cpp の scale_build）が，2^e は BF16 で表せるので切り上げても 2^e を越えず，e はこれと同じになる（最大値が
// 2^-126 未満か BF16 の範囲を超える行・列は，スケール段が範囲外として止める．expdelta_check）
inline int exp_ceil(double m) {
    const int e = std::ilogb(m);
    return (std::ldexp(1.0, e) == m) ? e : e + 1;
}

// ---- 経路の表 scripts/routes.tsv ------------------------------------------------------------------------
// 置き場所は OZ_ROUTES_TSV で与えられ，無ければ scripts/routes.tsv（リポジトリ直下から回す前提．scripts/selftest.sh・
// diagnose/probes.sh はそう回す）．列の意味は routes.tsv の冒頭．読めない・列が足りないときは止める（終了コード 2）．
inline const char* routes_path() {
    const char* v = oz_env::get("OZ_ROUTES_TSV");
    return (v && *v) ? v : "scripts/routes.tsv";
}
// 1 行の列の値（見出しの列名で引く）
struct RouteTsvRow { std::string line; std::vector<std::string> head, cell; };
inline std::string route_col(const RouteTsvRow& r, const char* name, const char* who) {
    for (size_t i = 0; i < r.head.size(); i++)
        if (r.head[i] == name) {
            if (i < r.cell.size()) return r.cell[i];
            break;
        }
    std::fprintf(stderr, "ERROR: %s：%s の行に列 %s が無い：%s\n", who, routes_path(), name, r.line.c_str());
    std::exit(2);
}
inline std::vector<RouteTsvRow> load_routes_tsv(const char* who) {
    const char* path = routes_path();
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "ERROR: %s：経路の表 %s を読めない（リポジトリ直下から回すか，OZ_ROUTES_TSV で置き場所を与える）\n",
                     who, path);
        std::exit(2);
    }
    auto split = [](const std::string& s) {
        std::vector<std::string> v; std::string x; std::istringstream is(s);
        while (std::getline(is, x, '\t')) v.push_back(x);
        return v;
    };
    std::vector<std::string> head;
    std::vector<RouteTsvRow> out;
    for (std::string line; std::getline(f, line); ) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> c = split(line);
        if (head.empty()) { head = c; continue; }
        out.push_back({line, head, std::move(c)});
    }
    return out;
}
inline int route_int(const std::string& s, const char* col, const std::string& key, const char* who) {
    char* e = nullptr; errno = 0;
    const long v = std::strtol(s.c_str(), &e, 10);
    if (s.empty() || *e || errno || v < 1 || v > 1000) {
        std::fprintf(stderr, "ERROR: %s：%s の %s の %s=%s は 1--1000 の整数ではない\n", who, routes_path(), key.c_str(), col, s.c_str());
        std::exit(2);
    }
    return (int)v;
}

// スキーム I の行（key・fmt・short・p・L）．p と L が L = p-1 を満たさないときは止める（古い既定の p で黙って回らないように）
struct Route1 { std::string key, fmt, name; int p = 0, L = 0; };   // name は short から "I-" を除いたもの（BF16 など）
inline std::vector<Route1> load_routes1(const char* who) {
    std::vector<Route1> out;
    for (const RouteTsvRow& row : load_routes_tsv(who)) {
        if (route_col(row, "scheme", who) != "I") continue;
        Route1 r;
        r.key = route_col(row, "key", who); r.fmt = route_col(row, "fmt", who);
        const std::string sh = route_col(row, "short", who);
        r.name = sh.rfind("I-", 0) == 0 ? sh.substr(2) : sh;
        r.p = route_int(route_col(row, "p", who), "p", r.key, who); r.L = route_int(route_col(row, "L", who), "L", r.key, who);
        if (r.L != r.p - 1) {
            std::fprintf(stderr, "ERROR: %s：%s の %s は L=%d が p-1=%d でない\n", who, routes_path(), r.key.c_str(), r.L, r.p - 1);
            std::exit(2);
        }
        out.push_back(r);
    }
    if (out.empty()) { std::fprintf(stderr, "ERROR: %s：%s にスキーム I の行が無い\n", who, routes_path()); std::exit(2); }
    // 本番の 8 形式がちょうど 1 回ずつ（行が欠けると検査が黙って少ない構成を回し，重なると同じ構成を 2 度回す）
    int seen[sizeof(OZ_FMTS) / sizeof(OZ_FMTS[0])] = {};
    for (const Route1& r : out) {
        OzFmt f;
        if (!oz_fmt_by_name(r.fmt.c_str(), f)) {
            std::fprintf(stderr, "ERROR: %s：%s の %s の形式 %s は本番の形式の表に無い\n", who, routes_path(), r.key.c_str(), r.fmt.c_str());
            std::exit(2);
        }
        seen[(int)f]++;
    }
    for (OzFmt f : OZ_FMTS)
        if (seen[(int)f] != 1) {
            std::fprintf(stderr, "ERROR: %s：%s のスキーム I の行に形式 %s が %d 回ある（1 回のはず）\n", who, routes_path(),
                         oz_fmt_host(f).name, seen[(int)f]);
            std::exit(2);
        }
    return out;
}

// スキーム II の行の法の本数 s（測定スクリプトがバイナリへ渡す本数）が，本番の既定（ozaki2_split.h の OZ2_ROUTES の
// def_s．引数を省いた実行が使う本数）と同じこと，本番の 8 形式がちょうど 1 回ずつ現れることを確かめる．食い違えば止める
// （終了コード 2．手で引数を省いた実行だけが別の構成になるのを防ぐ）．確かめた本数を 1 行で出す
inline void check_routes2_def_s(const char* who) {
    int seen[sizeof(OZ_FMTS) / sizeof(OZ_FMTS[0])] = {};
    std::string msg;
    for (const RouteTsvRow& row : load_routes_tsv(who)) {
        if (route_col(row, "scheme", who) != "II") continue;
        const std::string key = route_col(row, "key", who), fmt = route_col(row, "fmt", who);
        OzFmt f;
        if (!oz_fmt_by_name(fmt.c_str(), f)) {
            std::fprintf(stderr, "ERROR: %s：%s の %s の形式 %s は本番の形式の表に無い\n", who, routes_path(), key.c_str(), fmt.c_str());
            std::exit(2);
        }
        const int s = route_int(route_col(row, "s", who), "s", key, who);
        if (s != (int)oz2_param(f).def_s) {
            std::fprintf(stderr, "ERROR: %s：%s の %s は法の本数 s=%d が本番の既定 def_s=%u（ozaki2_split.h の OZ2_ROUTES）と違う\n",
                         who, routes_path(), key.c_str(), s, oz2_param(f).def_s);
            std::exit(2);
        }
        seen[(int)f]++;
        msg += (msg.empty() ? "" : "，") + key + " s=" + std::to_string(s);
    }
    for (OzFmt f : OZ_FMTS)
        if (seen[(int)f] != 1) {
            std::fprintf(stderr, "ERROR: %s：%s のスキーム II の行に形式 %s が %d 回ある（1 回のはず）\n", who, routes_path(),
                         oz_fmt_host(f).name, seen[(int)f]);
            std::exit(2);
        }
    std::printf("経路の表 %s（スキーム II）の法の本数が本番の既定（OZ2_ROUTES の def_s）と一致：%s\n", routes_path(), msg.c_str());
}

// ---- スキーム I の形式：本番の基数（ozaki1_param.h の oz1_param），桁の上限（ozaki_digit_max の既定
// oz1_default_digit_max = min(ceil(基数/2), exact_int_max)），部分積が Int32 か（oz_format_host.h の acc_int32．INT8 経路だけ）----
struct Fmt1 { OzFmt fmt; int radix, dmax; bool int32_partial; };
inline Fmt1 fmt1(const std::string& name, const char* who) {
    OzFmt f;
    if (!oz_fmt_by_name(name.c_str(), f)) {
        std::fprintf(stderr, "ERROR: %s：形式 %s は本番の形式の表（common/oz_format_host.h）に無い\n", who, name.c_str());
        std::exit(2);
    }
    return {f, (int)oz1_param(f).radix, (int)oz1_default_digit_max(f), oz_fmt_host(f).acc_int32};
}
// routes.tsv のスキーム I の経路を，基数・桁の上限・p（L）が同じものどうしでまとめた構成（表の順．name は
// "BF16/FP16/TF32/BFP8" のように経路の短い名をつなぐ）
struct Cfg1 { std::string name; int radix, dmax, p, L; bool int32_partial; };
inline std::vector<Cfg1> scheme1_configs(const char* who) {
    std::vector<Cfg1> out;
    for (const Route1& r : load_routes1(who)) {
        const Fmt1& f = fmt1(r.fmt, who);
        bool merged = false;
        for (Cfg1& c : out)
            if (c.radix == f.radix && c.dmax == f.dmax && c.p == r.p && c.int32_partial == f.int32_partial) {
                c.name += "/" + r.name; merged = true; break;
            }
        if (!merged) out.push_back({r.name, f.radix, f.dmax, r.p, r.L, f.int32_partial});
    }
    return out;
}
// 読んだ表を 1 行で出す（ログに p・L の出どころを残す）
inline void print_scheme1(const std::vector<Cfg1>& cs) {
    std::printf("経路の表 %s（スキーム I）：", routes_path());
    for (size_t i = 0; i < cs.size(); i++)
        std::printf("%s%s 基数 %d 桁 %d p=%d L=%d", i ? "，" : "", cs[i].name.c_str(), cs[i].radix, cs[i].dmax, cs[i].p, cs[i].L);
    std::printf("\n");
}

// ---- スキーム II の本番の経路（ozaki2_split.h の OZ2_ROUTES．並びは FP16 が最後）を表示名で引く（大文字小文字は問わない）----
inline const Oz2RouteRow* route2(const char* name) {
    for (const Oz2RouteRow& x : OZ2_ROUTES) if (strcasecmp(name, x.name) == 0) return &x;
    return nullptr;
}
// 2 つの法の取り方が，再構成の模擬の使う定数まで同じか（FP16 を BF16 の行で兼ねる条件）
inline bool same_moduli(const Ozaki2Moduli& a, const Ozaki2Moduli& b) {
    return a.ok == b.ok && a.s == b.s && a.b == b.b && a.m == b.m && a.mode == b.mode && a.q == b.q && a.yi == b.yi &&
           a.rinv == b.rinv && a.m16 == b.m16 && a.rp0 == b.rp0 && a.rp1 == b.rp1 && a.rp2 == b.rp2 && a.KMl == b.KMl;
}
// FP16 の経路 r の行を，K で同じ法の取り方になる BF16 の行（OZ2_ROUTE_BF16A）で兼ねるか（FP16 と BF16 は段の時間のモデルの
// 係数だけが違い，同じ集合を選ぶ K では再構成の模擬が同じになる）．mo は r の K での法の取り方．FP16 でない経路は false
inline bool fp16_same_as_bf16(const Oz2Route& r, uint32_t K, const Ozaki2Moduli& mo) {
    return r.awe == OZ2_AWE_FP16 && same_moduli(mo, ozaki2_choose_moduli_route(K, OZ2_ROUTE_BF16A));
}
inline bool fp16_same_as_bf16(const Oz2Route& r, uint32_t K) {
    return r.awe == OZ2_AWE_FP16 && fp16_same_as_bf16(r, K, ozaki2_choose_moduli_route(K, r));
}

// ---- 再構成（ozaki2/kernels/recon_crt.cpp）の 1 要素 -----------------------------------------------------------
// コンパイル時引数と実行時引数（法ごとの定数）を構造体で持つ
struct CrtKernel {
    uint32_t S = 0, NLIMB = 3, IN_RESIDUE = 1, NRP = 0, NSMALL_FRAG = 0, RAWMASK = 0;
    std::vector<float> mod, rinv, yi, m16, rp;   // rp は [i*NRP + t]（ホストの並び）
    float frag(uint32_t t, uint32_t i) const { return rp[(size_t)i * NRP + t]; }   // 法 i の断片 t（crt_proc::accumulate が引く）
};
// 法の集合 mo と断片 rpv から，ozaki2_main.cpp の run_recon_pass と同じ定数を組む（コンパイル時引数は本番と同じ
// ozaki2_moduli.h の oz2_recon_ct．limb の本数は呼び出し側が oz2_recon_limb3 で決める．調査用の OZ2_NSMALL などは重ねない）
template <size_t NP>
inline CrtKernel make_crt_kernel(const Ozaki2Moduli& mo, const std::vector<std::array<float, NP>>& rpv, bool in_res, bool limb3) {
    CrtKernel k;
    for (int i = 0; i < mo.s; i++) {
        k.mod.push_back((float)mo.m[i]); k.rinv.push_back(mo.rinv[i]); k.yi.push_back((float)mo.yi[i]);
        k.m16.push_back(mo.m16[i]);
        for (size_t t = 0; t < NP; t++) k.rp.push_back(rpv[i][t]);
    }
    const Oz2ReconCT ct = oz2_recon_ct(mo, k.rp.data(), (uint32_t)NP, in_res, limb3);
    k.S = ct.S; k.NLIMB = ct.NLIMB; k.IN_RESIDUE = ct.IN_RESIDUE; k.NRP = ct.NRP; k.NSMALL_FRAG = ct.NSMALL_FRAG; k.RAWMASK = ct.RAWMASK;
    return k;
}
// 1 パス（rp0）と 2 パスの第 1（rp1）・第 2（rp2）のカーネル
struct ReconKernels { CrtKernel one, first, second; };
inline ReconKernels make_recon_kernels(const Ozaki2Moduli& mo, bool in_res) {
    // limb の本数は本番と同じ規則（ozaki2_moduli.h の oz2_recon_limb3）
    return {make_crt_kernel(mo, mo.rp0, in_res, oz2_recon_limb3(mo, false)),
            make_crt_kernel(mo, mo.rp1, in_res, oz2_recon_limb3(mo, true)),
            make_crt_kernel(mo, mo.rp2, in_res, oz2_recon_limb3(mo, true))};
}

using sfpu_emu::Dst;
// 法 i の入力を r_i にして tmp0 へ置く（recon_crt.cpp の load_residue_fp32／decode_signmag_residue）．
// 剰余入力（IN_RESIDUE=1）は FP32 の値 resf[i]，INT8 経路（IN_RESIDUE=0）は符号-絶対値 int32 の語 raw[i]
inline void crt_load(Dst& D, const CrtKernel& k, uint32_t i, const float* resf, const uint32_t* raw) {
    using namespace sfpu_emu::dst;
    const float m = k.mod[i], ri = k.rinv[i];
    if (k.IN_RESIDUE) { D.r[tmp0] = resf[i]; return; }
    const uint32_t w = raw[i], mag = w & 0x7FFFFFFFu;
    const float v = (w >> 31) ? -(float)mag : (float)mag;              // typecast（符号だけ使う）
    D.r[tmp2] = (v > 0.f) ? 1.f : (v < 0.f) ? -1.f : 0.f;              // sign_tile
    D.r[term] = (float)(w & 0x7FFF0000u);                               // and + typecast
    D.MulS(term, 1.52587890625e-05f);
    sfpu_emu::ModRound(D, term, keep, tmp1, m, ri);
    D.MulS(keep, k.m16[i]);
    D.r[term] = (float)(w & 0x0000FFFFu);
    sfpu_emu::ModRound(D, term, tmp0, tmp1, m, ri);
    D.Add(tmp0, keep, tmp0);
    D.Mul(tmp0, tmp2, tmp0);
    sfpu_emu::ModRound(D, tmp0, tmp1, m, ri);
}
// tmp0 の r_i から k_i = ModRound(r_i*y_i) を keep へ（1 桁の法は先に法へ畳む．RAWMASK．本番と同じ crt_proc::term_k）
inline void crt_term_k(Dst& D, const CrtKernel& k, uint32_t i) {
    crt_proc::term_k(D, (k.RAWMASK & (1u << i)) != 0u, k.mod[i], k.rinv[i], k.yi[i]);
}
// keep の k_i に断片を掛けて足し込み（末尾 NSMALL_FRAG 本は tfAddSmall），法ごとに a0 -= round(a0)
// （本番と同じ crt_proc::accumulate・drop_integer．limb の本数はカーネルではコンパイル時引数なので，ここで振り分ける）
inline void crt_accumulate(Dst& D, const CrtKernel& k, uint32_t i) {
    if (k.NLIMB >= 3) crt_proc::accumulate<3>(D, k.NRP, k.NSMALL_FRAG, k, i);
    else              crt_proc::accumulate<2>(D, k.NRP, k.NSMALL_FRAG, k, i);
    crt_proc::drop_integer(D);
}
// 1 要素ぶん（recon_crt.cpp の kernel_main の法のループ．出口の畳み込みはライタの assemble_frac_f64 が行う）
inline void crt_element(Dst& D, const CrtKernel& k, const float* resf, const uint32_t* raw) {
    using namespace sfpu_emu::dst;
    for (uint32_t i = 0; i < k.S; i++) {
        crt_load(D, k, i, resf, raw);
        if (i == 0) { D.Set(acc_hi, 0.f); D.Set(acc_mid, 0.f); D.Set(acc_lo, 0.f); }   // clear_accumulator
        crt_term_k(D, k, i);
        crt_accumulate(D, k, i);
    }
}
// pack_fraction_limbs（FP32 の DST を Float32 の CB へ：ビット列のまま．2 本組なら 3 本目は 0）
inline void crt_pack(const Dst& D, const CrtKernel& k, uint32_t out[3]) {
    using namespace sfpu_emu::dst;
    out[0] = f32_bits(D.r[acc_hi]); out[1] = f32_bits(D.r[acc_mid]);
    out[2] = (k.NLIMB >= 3) ? f32_bits(D.r[acc_lo]) : 0u;
}

}  // namespace oz_sim
