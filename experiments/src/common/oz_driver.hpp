// SPDX-License-Identifier: Apache-2.0
#pragma once
// 尾崎の 2 つのドライバの枠：入口（位置引数とデバイスを開く前の検査の並び）と，ログの共通の部分（CONFIG 行の共通の欄・
// WHOLE-PIPELINE 行・最後の判定の行）．入口の検査の並びは尾崎以外のドライバ（基準実装と gemm_exact_check）も同じ関数で揃える（末尾）
// ---------------------------------------------------------------------------
// 入口の検査（デバイスを開く前）は両スキームとも次の並びで行う．どれも打ち間違いを理由つきの終了コード 2 で止める
// （デバイスを開いた後に見つかると閉じずに終わる）．
//   1. reject_env   削除したつまみ・もう一方のスキームだけのつまみ・このドライバが読まないつまみ（表の who）・表に無い OZ2_ の
//                   名前（環境だけを見る．位置引数より先）
//   2. 位置引数     余分な引数（oz_env::check_argc）・M N K（read_dims）・各スキームの引数・warmup と iters（read_counts）
//   3. check_env    このスキームが読む表のつまみの値（範囲・列挙．oz_knobs::preflight．GEMM 段の L1 の予算と探針も）→ 入力の
//                   生成・判定・書き出し（oz_eval::check_env．OZ_EXPK と K の関係も）→ 計時とテレメトリ（oz_eval::timing_check）
//   4. 各スキームに固有の検査（I は p・L・基数，II は OZ2_PANEL_FORCE・INT8 経路の OZ2_CHECK_CONCAT など）
// CONFIG 行は oz_logline.hpp の OzLine で欄を名前で足して組む．両スキームに共通の欄はここの関数が足す（欄の中身と並びは
// どちらのスキームでも同じ）．
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "common/host_eval.hpp"     // check_env・timing_check・check_tile_dims・input_desc・timing_desc・kGenVersion
#include "common/ozaki_gemm_host.hpp"   // oz_l1_budget（CONFIG 行の l1_budget_kb）
#include "common/oz_env.hpp"
#include "common/oz_knobs.hpp"
#include "common/oz_logline.hpp"

namespace oz_drv {

// argv[0] の最後の要素（誤りの文に出すドライバの名前）
inline const char* drv_name(const char* argv0) {
    const char* s = std::strrchr(argv0, '/');
    return s ? s + 1 : argv0;
}

// 1. 削除したつまみ，もう一方のスキームだけのつまみ（案内つき），このスキームが読まないつまみ（表の who．もう一方の
//    スキームとの区別は上が先に止める．oz_knobs::reject_unread），表に無い OZ2_ の名前（スキーム I は上の規則が OZ2_ の
//    変数をすべて止める）．self は oz_knobs::W_I か W_II．入力の組はどちらのスキームも全部読むので，理由の文は使わない
inline void reject_env(uint16_t self, const char* argv0) {
    const char* drv = drv_name(argv0);
    oz_knobs::reject_removed();
    oz_knobs::reject_other_scheme(self, drv);
    oz_knobs::reject_unread(self, drv, "");
    if (self != oz_knobs::W_I) oz_knobs::reject_unknown_oz2(drv);
}

// 3. このドライバが読む表のつまみの値（範囲・列挙．GEMM 段の L1 の予算 OZ_L1_BUDGET_KB と探針もここで読む：表の Grp::Gemm は
//    範囲を表で持つ U32 なので preflight が読む．table_ok）→ 入力の生成と判定と書き出し（表の W_INPUT のドライバだけ．K は
//    位置引数の K，verify は参照解を求めるか）→ 計時とテレメトリ（表の W_TIMED のドライバだけ）．self は表の読むプログラムの
//    ビット（尾崎の W_I・W_II と，下の尾崎以外のドライバの W_BF16 など）
inline void check_env(uint16_t self, uint32_t K, bool verify) {
    oz_knobs::preflight(self);
    if (self & oz_knobs::W_INPUT) oz_eval::check_env(K, verify);
    if (self & oz_knobs::W_TIMED) oz_eval::timing_check();
}
inline void check_env(uint16_t self, uint32_t K) { check_env(self, K, oz_eval::verify_enabled()); }

// 位置引数 1〜3 の M N K（無ければ dflt）．1 以上 2^32 未満で，32 の倍数でなければ止める：全工程がタイル（32x32）単位で
// 動くので，端数があると黙って切り捨てた行列を計算し，しかも参照解はフルの M,N,K で計算するので誤った精度が警告なしに
// 報告される．0 は法の選択で K による除算に届き，2^32+64 のような値は uint32_t へ落とすと 64 に化けて 32 の倍数の検査を
// 素通りする（oz_env::arg_u32 が縮める前に範囲を見る）
struct Dims { uint32_t M = 0, N = 0, K = 0; };
inline Dims read_dims(int argc, char** argv, Dims dflt, const char* who) {
    Dims d;
    d.M = oz_env::arg_u32(argc, argv, 1, dflt.M, 1u, UINT32_MAX, "M");
    d.N = oz_env::arg_u32(argc, argv, 2, dflt.N, 1u, UINT32_MAX, "N");
    d.K = oz_env::arg_u32(argc, argv, 3, dflt.K, 1u, UINT32_MAX, "K");
    if (!oz_eval::check_tile_dims(d.M, d.N, d.K, who)) std::exit(2);
    return d;
}

// 位置引数の warmup・iters（warmup は wmin 以上，iters は 1 以上，ともに 1000000 以下）．warmup は計時の前に段ごとに流す回数（cold の計時．steady は
// 定常状態まで流す）である．尾崎 I の GEMM のプログラムは最初の投入で JIT を済ませる（ほかの段は計時の前に 1 回流して
// 済ませる）ので，0 回にすると JIT が計時に入る．以前は I が 0 を黙って 1 に読み替え，II は 0 回のまま流した（同じ位置
// 引数の意味がスキームで違った）．両スキームとも 1 以上を要る．iters は 0 だと time_dispatch が 1 回に読み替えて，記録した
// 回数と測った回数が食い違う．wmin は warmup の下限で，尾崎の 2 つは既定の 1．基準実装（gemm_bf16・gemm_bf16ext・gemm_int8）は
// 0 を渡して warmup=0 を受ける：行列積 1 本を流す run_mcast_gemm（mcast_gemm.hpp）は計時の前に必ず 1 回流して JIT を
// 済ませるので，0 回でも JIT は計時に入らない（scripts/diagnose/probes.sh の probe_micro・probe_ext_k は gemm_bf16ext に 0 を渡す）
inline void read_counts(int argc, char** argv, int iw, uint32_t dw, uint32_t di, uint32_t& w, uint32_t& it,
                        uint32_t wmin = 1u) {
    w = oz_env::arg_u32(argc, argv, iw, dw, wmin, 1000000u, "warmup");
    it = oz_env::arg_u32(argc, argv, iw + 1, di, 1u, 1000000u, "iters");
}

// ---- CONFIG 行の共通の欄 ----
// GEMM 段の実装の印（docs/gemm_opt_results.md）．GEMM 段は分割データ移動＋パッカの L1 累積の 1 通りだけなので常に 1．
// この欄が無いログは，この GEMM 段で測ったものではない
inline void add_gemm_impl(OzLine& l) { l.kv("gemm_split", 1).kv("gemm_l1acc", 1); }
// 入力の作り方・seed・精度の基準（oz_eval::input_desc），全工程を続けて流す計時（OZ_TIME_WHOLE．whole_ms はその 1 回あたり
// ［ms］で，立てていなければ 0），計時の方式（oz_eval::timing_desc）
inline void add_input_whole_timing(OzLine& l, bool whole, double whole_ms) {
    l.raw(oz_eval::input_desc()).flag("time_whole", whole).num("whole_ms", "%.4f", whole_ms).raw(oz_eval::timing_desc());
}
// 末尾の共通の欄：形状探索の L1 の予算［KiB］（既定でなければ knobs にも出る），入力の生成の版（oz_eval::kGenVersion．
// 照合 scripts/rules/probe_checks.py の PHI_GEN が φ の入力 a=(u-1/2)exp(φz) を生成する版として登録したものか見る）
inline void add_common_tail(OzLine& l) {
    l.kv("l1_budget_kb", oz_l1_budget() / 1024u).kv("gen", oz_eval::kGenVersion);
}

// 全工程を段の間で同期せずに続けて流した時間（OZ_TIME_WHOLE．論文の全体の時間．DRAM->DRAM は段ごとの計時の和で内訳に使う）．
// whole_s・sum_s は 1 回あたりの秒，flop は演算数（2MNK）
inline void print_whole_pipeline(double whole_s, double sum_s, double flop) {
    std::printf("WHOLE-PIPELINE : %.3f ms/iter  (stage sum %.3f ms; whole/sum-1 = %+.2f%%)  %.4f TFLOP/s fp64-equiv\n",
                whole_s * 1e3, sum_s * 1e3, 100.0 * (whole_s / std::max(1e-12, sum_s) - 1.0),
                flop / std::max(1e-12, whole_s) / 1e12);
}

// "AxB" の形の欄の値（CONFIG 行の panels=・grid=・shape=・sub= など）
inline std::string axb(uint64_t a, uint64_t b) { return std::to_string(a) + "x" + std::to_string(b); }

// ---- 尾崎以外のドライバ（基準実装 gemm_bf16・gemm_bf16ext・gemm_int8 と GEMM 段の検査 gemm_exact_check）の入口 ----
// 検査の並びは尾崎の 2 つと同じ（冒頭の 1〜4）で，3 は上の check_env を使う．位置引数（2）の並びと読み方は各ドライバが持つ
// （read_dims・read_counts・oz_env::check_argc は使ってよい）．1 は環境だけを見る：削除したつまみ，このドライバが読まない
// つまみ（表の who に self が無い A〜E のもの．入力の生成・判定・書き出しの組には why をその理由として誤りの文に添える．
// GEMM 段の探針は，基準実装と検査が McastGemmSpec の probe を詰めないので効かない．oz_knobs::reject_unread），表に無い OZ2_ の
// 名前．黙って無視すると，その構成・入力・基準・書き出しで測ったつもりになる．drv はドライバの名前
inline void reject_env_gemm(uint16_t self, const char* drv, const char* why) {
    oz_knobs::reject_removed();
    oz_knobs::reject_unread(self, drv, why);
    oz_knobs::reject_unknown_oz2(drv);
}

// 最後の判定の行（空行の後に Test Passed・Test Failed・Test Skipped）を出し，終了コード（通れば 0，通らなければ 1）を返す．
// 検証を省いた実行（OZ_NOVERIFY）は精度を判定していないので，通っても Test Passed とは書かない（終了コードは 0）．
// passed は通ったときの文で，精度に合否の基準を持たないドライバ（gemm_bf16ext）は "Test Passed" の代わりにそれを書く
// （精度を判定していない実行に Test Passed と書かないのは OZ_NOVERIFY と同じ理由）
inline int finish(bool pass, bool verified, const char* passed = "Test Passed") {
    std::printf("\n%s\n", !pass ? "Test Failed" : verified ? passed : "Test Skipped");
    return pass ? 0 : 1;
}

}  // namespace oz_drv
