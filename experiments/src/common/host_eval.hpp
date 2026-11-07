// SPDX-License-Identifier: Apache-2.0
#pragma once
// ホスト側の測定・評価の部品の傘（尾崎の 2 つのドライバ・基準実装・GEMM 段の検査と一部の検査・探針，ホスト単体の道具
// tools/dgemmcheck・dot2check・splitsim が取り込む）
// ---------------------------------------------------------------------------
// 部品ごとに 1 つのヘッダに分けてあり，ここはそれを取り込んで，部品をまたぐ入口の検査（check_env・check_iters・
// check_tile_dims）と CONFIG 行の欄の書き方（rt_double・input_desc・timing_desc）だけを持つ．
//   oz_input.hpp      入力 A・B の生成 gen_operands と入力のつまみ（OZ_PHI・OZ_CONST など）
//   oz_reference.hpp  参照解（FP64 の Dot2 の reference_gemm と整数の int_reference_gemm）と OZ_NOVERIFY
//   oz_accuracy.hpp   精度の指標 accuracy（相対誤差・実効ビット数・結果のビット列のハッシュ）と合否の基準 OZ_MIN_BITS，
//                     整数の完全一致 exact_check
//   oz_timing.hpp     計時の手順（cold・steady．time_dispatch など）と TELEM・WARMUP・FWTELEM 行
//                     （telemetry.hpp を取り込むのはここだけ）
//   oz_dump.hpp       結果の書き出し OZ_DUMP_C（調査用）
// 入力・参照解・精度の指標は CPU の DGEMM の参照（tools/dgemmcheck．scripts/measure/cpu_dgemm_reference.sh が測る）と共有する．照合
// （scripts/lib/peer_logs.py）は DGEMM 参照のログを尾崎のログと並べる前に，両方の rev でこのファイルから #include "…" を
// たどり，gen_operands・reference_gemm・accuracy とその本体が名前で使う定義が同じことを要る．そのため，
//   ・傘の下のヘッダは src/common に置き，同じディレクトリから相対で取り込む（measure/cpu_dgemm_reference.sh はこのファイルの置き場所の
//     ディレクトリだけをコンテナへ渡す．コンテナの外で tools/ を作るときも見つかる．絶対パスの取り込みは照合がたどれない）
//   ・つまみの表（oz_knobs.hpp）は取り込まない（表につまみを 1 つ足すだけで DGEMM 参照を測り直すことにならないように．
//     取り込むのは読み方の oz_env.hpp だけ）
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "oz_env.hpp"         // つまみと位置引数の読み方
#include "oz_input.hpp"
#include "oz_reference.hpp"
#include "oz_accuracy.hpp"
#include "oz_timing.hpp"
#include "oz_dump.hpp"

namespace oz_eval {

// ---- 引数の検査：どのドライバも位置引数で M N K ... を取る（読むのは oz_env::arg_u32）-----------------
// 計時の反復回数 iters は 1 以上（time_dispatch は 0 を 1 回に読み替えるので，0 を受けると記録した回数
// （基準実装の Problem 行・CONFIG 行の iters=0）と実際に測った回数（TELEM の iters）が食い違う）．
// 基準実装（gemm_bf16・gemm_bf16ext・gemm_int8）が使う（尾崎の 2 つのドライバは oz_driver.hpp の read_counts で範囲として見る）
inline bool check_iters(uint32_t iters, const char* who) {
    if (iters == 0) {
        std::fprintf(stderr, "ERROR: %s の iters は 1 以上でなければならない\n", who);
        return false;
    }
    return true;
}

// 全工程がタイル（32x32）単位で動くので，端数があると黙って切り捨てた行列を
// 計算してしまう．しかも参照解はフルの M,N,K で計算するので，誤った精度が警告なしに
// 報告される．尾崎の 2 つ以外のドライバと検査が使う（尾崎の 2 つのドライバは oz_driver.hpp の read_dims から）．
inline bool check_tile_dims(uint32_t M, uint32_t N, uint32_t K, const char* who) {
    if (!M || !N || !K || (M % 32) || (N % 32) || (K % 32)) {
        std::fprintf(stderr, "ERROR: %s の M,N,K は正で 32 の倍数でなければならない"
                             "（M=%u N=%u K=%u）\n", who, M, N, K);
        return false;
    }
    return true;
}

// v を読み戻すと同じ double になる最短の %.<n>g（n = 6..17．6 桁で足りれば %g と同じ文字列）
inline std::string rt_double(double v) {
    char b[40];
    for (int prec = 6; prec <= 17; ++prec) {
        std::snprintf(b, sizeof b, "%.*g", prec, v);
        if (std::strtod(b, nullptr) == v) break;
    }
    return b;
}

// CONFIG 行へ足す計時の方式の欄（"timing=steady,warmup_sec=10,…,telem_ms=20,telem_dev=0" の形．値に ',' と '=' は無い）．
// cold のときは温めの欄を出さない（その方式では使わない）．テレメトリが読めなければ telem_dev=none
inline std::string timing_desc() {
    const TimingCfg& c = timing_cfg();
    const oz_telem::Sampler& t = oz_telem::Sampler::instance();
    std::string s = std::string("timing=") + (c.steady ? "steady" : "cold");
    if (c.steady)
        s += ",warmup_sec=" + rt_double(c.warm_sec) + ",warmup_max_sec=" + rt_double(c.warm_max_sec) +
             ",warmup_win_sec=" + rt_double(c.win_sec) + ",warmup_tol_pct=" + rt_double(c.tol_pct) +
             ",warmup_gap_ms=" + rt_double(c.gap_sec * 1e3) + ",warmup_hot_factor=" + rt_double(c.hot_factor);
    s += ",telem_ms=" + std::to_string(t.period_ms());
    s += ",telem_dev=" + (t.available() ? std::to_string(t.dev()) : std::string("none"));
    return s;
}

// 実際に使われた入力の作り方（oz_input.hpp の gen_operands の優先順位を当てた後）と seed・精度の基準を，CONFIG 行へそのまま
// 足せる形 "input=<作り方>,seed=<n>,min_bits=<b>" で返す（これが無いと乱数入力と OZ_CONST=1 が同じ
// CONFIG になり，定数入力の 99 bit を乱数入力の精度と取り違えうる）．作り方は
//   expdiff:<d>:k<k>   phi:<φ>   const:<v>[:<符号の付け方>]   uniform|pos:<lo> に +wide:<R>・+rowscale:<R> を重ねたもの
// で，値の中に ',' と '=' は入らない（検査側は key=value で読む）．優先順位で使われないつまみは出さない．
// 実数の値（φ・定数・下限・幅・合否の基準）は読み戻すと同じ double になる桁数で書く（rt_double）．既定の 6 桁
// （%g）では OZ_CONST=0.4921875 が 0.492188，1 と 1+2^-30 がともに 1，OZ_MIN_BITS=52.999999 が 53 になり，
// 違う設定が同じ CONFIG になる．%g で読み戻せる値は %g のままの書き方になる（phi:4・min_bits=53 など）．
// with_min_bits=false は min_bits 欄を付けない（基準実装の gemm_bf16・gemm_bf16ext は OZ_MIN_BITS を判定に使わない
// ので，53 と書くと別の基準で判定したように読める）．
inline std::string input_desc(uint32_t seed = kDefaultSeed, bool with_min_bits = true) {
    const InputEnv e = input_env();
    char b[160];
    std::string m;
    if (e.expdiff)      { std::snprintf(b, sizeof b, "expdiff:%d:k%d", e.d, e.k2); m = b; }
    else if (e.phi)     m = "phi:" + rt_double(e.phiv);
    else if (e.cst)     { m = "const:" + rt_double(e.c); if (!e.sign.empty()) m += ":" + e.sign; }
    else {
        m = e.pos ? "pos:" + rt_double(e.lo) : std::string("uniform");
        if (e.wide) m += "+wide:" + rt_double(e.R);
        if (e.rows) m += "+rowscale:" + rt_double(e.Rs);
    }
    if (with_min_bits) std::snprintf(b, sizeof b, ",seed=%u,min_bits=%s", seed, rt_double(min_bits()).c_str());
    else               std::snprintf(b, sizeof b, ",seed=%u", seed);
    return "input=" + m + b;
}

// 入力の生成（input_env）・精度の判定（min_bits）・結果の書き出し（OZ_DUMP_C）のつまみを，デバイスを開く前に
// 1 度読んで検査する（不正なら exit(2)）．どれも使う場所で同じ検査をするが，尾崎 I は入力の生成も判定も
// デバイスを開いた後なので，そこで初めて読むと打ち間違いがデバイスを閉じずに終わる．入力の生成・判定・書き出しを読む
// ドライバ（つまみの表の W_INPUT：尾崎の 2 つ・基準実装の gemm_bf16・gemm_bf16ext と tools/dgemmcheck）が呼ぶ（読まない
// gemm_int8・gemm_exact_check と tools の dot2check・splitsim は，代わりに oz_knobs.hpp の拒否で止める）．
// OZ_DUMP_C は書き出し先をファイルを作らずに見る（oz_dump.hpp の check_dump_path）．書き出すのは参照解（Chi・Clo）を
// 含むので，検証を省く実行（verify=false．既定は OZ_NOVERIFY の有無．gemm_bf16 は位置引数の verify=0 も渡す）との
// 併用は止める：止めないと何も書かずに 0 で終わり，同じ置き場所に残った前の実行のファイルを今回の結果として
// tools/cmp_dump.cpp が比べてしまう．
// K（0 でなければ）との関係も見る：OZ_EXPDIFF の第 2 項の位置 OZ_EXPK は 1 以上 K 未満（gen_operands と同じ条件．
// 入力の生成まで待つと，尾崎 I ではデバイスを開いた後に exit(2) することになる）．
inline void check_env(uint32_t K = 0, bool verify = verify_enabled()) {
    const InputEnv e = input_env();
    if (K) check_expk(e, K);   // oz_input.hpp（gen_operands と同じ検査）
    (void)min_bits();
    if (const char* path = oz_env::get("OZ_DUMP_C"); path && *path) {
        if (!verify) {
            std::fprintf(stderr, "ERROR: OZ_DUMP_C=%s は参照解を書き出すので，検証を省く実行（%s）とは併用できない\n", path,
                         verify_enabled() ? "verify=0" : "OZ_NOVERIFY");
            std::exit(2);
        }
        check_dump_path(path);
    }
}
}  // namespace oz_eval
