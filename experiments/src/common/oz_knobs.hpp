// SPDX-License-Identifier: Apache-2.0
#pragma once
// つまみ（環境変数）の表：名前・分類・読むプログラム・読み方・CONFIG 行への記録
// ---------------------------------------------------------------------------
// つまみの一覧を手で持つ場所をここの 1 つにする．次のものはどれもこの表から作る：
//   ・尾崎の 2 つのドライバの CONFIG 行の knobs 欄（性能だけを変えるつまみ）・diag 欄（診断のつまみ．立っていれば probe=set）
//   ・スキームをまたぐ拒否（スキーム II は I だけのつまみを止める．I は OZ2_ の接頭辞で止める）
//   ・削除したつまみの拒否（最後に含むコミットを添える）
//   ・ドライバが読まないつまみの拒否（reject_unread．表の who にそのドライバが無い A〜E のつまみ．入力の作り方・精度の
//     基準・書き出しの組（gemm_bf16・gemm_bf16ext の OZ_MIN_BITS，gemm_int8・gemm_exact_check・tools の dot2check・splitsim は
//     組の全部）と，GEMM 段の探針を記録できない基準実装の拒否はその特別な場合）と，表に無い OZ2_ の名前の拒否
//   ・入口の検査（preflight：ドライバが読む，範囲を表で持つつまみを，デバイスを開く前に 1 度ずつ読んで検査する）
// 分類（cls）は docs/env_vars.md の節（A 論文の測定・B アブレーション・C 診断・D 入力の作り方・D2 検査の書き出し・E その他・
// F 検査・探針・道具のつまみ）と同じで，その一致は tools/knobs_check.cpp が確かめる（selftest.sh の hostcheck．表の名前と
// 節が docs と食い違う，src・tools が表に無い名前を読む，のどれでも失敗する）．
// tt-metal に依存しない（tools/ からも取り込む）．値の読み方の規則（整数・負号・範囲・終了コード 2）は oz_env.hpp．
//
// 範囲（ranged=true）を表で持つつまみは oz_knobs::u32 などで読む（範囲はここにしか書かない）．範囲が形式で決まるもの
// （OZ_RADIX・OZ_DMAX）と，入力の生成・計時・テレメトリのつまみ（host_eval.hpp の傘の下の oz_input.hpp・oz_timing.hpp と telemetry.hpp が oz_env で読む．
// CPU の DGEMM の参照と共有する読み取りなので，この表を取り込まない）は ranged=false で，読む所が範囲を持つ．
// 「既定から外れている」（active）の決め方：既定値を持つ（has_dflt）なら読んだ値が既定と違うとき，持たないなら立っているとき
// （OZ_RADIX=<既定の基数> も立てたものとして記録する）．
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unistd.h>    // environ（スキームをまたぐ拒否と表に無い OZ2_ の名前の拒否）
#include "oz_env.hpp"   // 同じディレクトリ（tools/ から取り込むときも見つかるように相対で書く）

namespace oz_knobs {

// docs/env_vars.md の節（Removed は末尾の「以下は削除した」の段落）
enum class Cls : uint8_t { A, B, C, D, D2, E, F, Removed };
inline const char* cls_name(Cls c) {
    switch (c) {
        case Cls::A: return "A";   case Cls::B: return "B";   case Cls::C: return "C";  case Cls::D: return "D";
        case Cls::D2: return "D2"; case Cls::E: return "E";   case Cls::F: return "F";  default: return "removed";
    }
}
// 読むプログラム（ビットの和）
enum Who : uint16_t {
    W_I = 1,          // ozaki1_*
    W_II = 2,         // ozaki2_*
    W_BF16 = 4,       // gemm_bf16
    W_EXT = 8,        // gemm_bf16ext
    W_I8 = 16,        // gemm_int8
    W_EXACT = 32,     // gemm_exact_check
    W_CHECK = 64,     // src/checks の scale_check・slice_split_check
    W_PROBE = 128,    // src/probes
    W_TOOL = 256,     // tools/（ホスト単体の道具）
    W_DGEMM = 512,    // tools/dgemmcheck（CPU の DGEMM の参照）
};
constexpr uint16_t W_OZ = W_I | W_II;
constexpr uint16_t W_GEMM = W_I | W_II | W_BF16 | W_EXT | W_I8 | W_EXACT;     // build_mcast_gemm を使うドライバ
constexpr uint16_t W_INPUT = W_I | W_II | W_BF16 | W_EXT | W_DGEMM;           // 入力の生成・判定・書き出し（oz_input.hpp・oz_accuracy.hpp・oz_dump.hpp）
constexpr uint16_t W_TIMED = W_I | W_II | W_BF16 | W_EXT | W_I8;              // 計時とテレメトリ（oz_timing.hpp・telemetry.hpp）
// 読み方（oz_env.hpp）．Set は有無，U32・F64 は範囲つきの数，Choice は列挙，Str は文字列（書式は使う側が見る）
enum class Kind : uint8_t { Set, U32, I64, F64, Choice, Str };
// CONFIG 行への記録（ビットの和）．尾崎の 2 つのドライバだけが使う
enum Rec : uint8_t {
    R_NONE = 0,
    R_KNOBS = 1,   // knobs 欄（性能だけを変える．値を持つものは 名前:値）
    R_DIAG = 2,    // diag 欄（診断．立っていれば probe=set）
    R_FIELD = 4,   // 専用の欄（input=・timing=・pass_mode= など．knobs・diag には入れない）
};
// 拒否の組
enum class Grp : uint8_t {
    None,
    Input,   // 入力の作り方・精度の基準・結果の書き出し（読まないドライバでは reject_unread が理由 why_input を添えて止める）
    Gemm,    // GEMM 段（GEMM を組むドライバ W_GEMM が読む．C の探針は尾崎の 2 つだけが読み，基準実装・検査では reject_unread が止める）
};

struct Knob {
    std::string_view name;
    Cls cls;
    uint16_t who;
    Kind kind;
    bool ranged;          // 範囲・列挙を表で持つ（false：読む所が持つ）
    double lo, hi;        // U32・I64・F64 の範囲
    bool has_dflt;        // 既定値を持つ（持たなければ「立っていない」が既定）
    double dflt;
    uint8_t rec;          // Rec の和
    Grp grp;
    const char* choices;  // Choice の名前（"|" 区切り．表示と検査）
    const char* note;     // Removed：最後に含むコミット．スキームをまたぐ拒否の案内（無ければ nullptr）
};

// 表の行の書き方を短くする（Set・Str は範囲を持たない）
constexpr Knob set_(std::string_view n, Cls c, uint16_t w, uint8_t rec, Grp g = Grp::None, const char* note = nullptr) {
    return {n, c, w, Kind::Set, false, 0, 0, false, 0, rec, g, nullptr, note};
}
constexpr Knob str_(std::string_view n, Cls c, uint16_t w, uint8_t rec, Grp g = Grp::None) {
    return {n, c, w, Kind::Str, false, 0, 0, false, 0, rec, g, nullptr, nullptr};
}
constexpr Knob u32_(std::string_view n, Cls c, uint16_t w, double lo, double hi, uint8_t rec, Grp g = Grp::None,
                    const char* note = nullptr) {
    return {n, c, w, Kind::U32, true, lo, hi, false, 0, rec, g, nullptr, note};
}
constexpr Knob u32d(std::string_view n, Cls c, uint16_t w, double lo, double hi, double d, uint8_t rec,
                    Grp g = Grp::None) {
    return {n, c, w, Kind::U32, true, lo, hi, true, d, rec, g, nullptr, nullptr};
}
constexpr Knob val_(std::string_view n, Cls c, uint16_t w, Kind k, uint8_t rec, Grp g = Grp::None) {   // 範囲は読む所
    return {n, c, w, k, false, 0, 0, false, 0, rec, g, nullptr, nullptr};
}
constexpr Knob chc_(std::string_view n, Cls c, uint16_t w, const char* choices, uint8_t rec, bool ranged = true) {
    return {n, c, w, Kind::Choice, ranged, 0, 0, false, 0, rec, Grp::None, choices, nullptr};
}
constexpr Knob gone(std::string_view n, const char* commit) {
    return {n, Cls::Removed, 0, Kind::Set, false, 0, 0, false, 0, R_NONE, Grp::None, nullptr, commit};
}

// 形状探索の L1 の予算の既定［KiB］（OZ_L1_BUDGET_KB．ozaki_gemm_host.hpp の oz_l1_budget）．根拠：予算を上げていくと
// 14080^3 の Ozaki-II-BFP4 は 1470 KiB まで通り 1480 KiB で CB の確保に失敗する（L1 は 1536 KiB なので，予約領域と
// カーネル側の CB に 64 KiB 強が要る）．失敗はプログラム生成の時点で明示的に出るので，取りすぎても黙って壊れることはない．
// 1408 は，この境界に 64 KiB の余裕を残したうえで狭い形式の bw を一段上げられる値である（1280 KiB では 1 要素 1.0625 B の
// BFP8_b が bw を一段落とす．docs/gemm_stage_bottleneck.md の §1）．scripts/rules/probe_checks.py が測った rev のこの定義を
// 名前で読むので，"OZ_L1_BUDGET_KB_DEFAULT = <数>u" の書き方を保つ
inline constexpr uint32_t OZ_L1_BUDGET_KB_DEFAULT = 1408u;

// 表の並びが knobs 欄・diag 欄の名前の並びになる（スキーム I は OZ_NO_DEFER から OZ_DBG まで，II は OZ_NO_BLOCKMM から
// OZ2_PANEL_FORCE までの knobs と，OZ2_JSPLIT から OZ2_CHECK_CONCAT までの diag の順．並べ替えるとログの欄の文字列が変わる）
inline constexpr Knob TABLE[] = {
    // ---- A 論文の測定 ----
    val_("OZ_PHI", Cls::A, W_INPUT, Kind::F64, R_FIELD, Grp::Input),                 // input=phi:<φ>
    set_("OZ_NOVERIFY", Cls::A, W_OZ | W_BF16 | W_EXT | W_I8 | W_EXACT, R_FIELD),    // verified=0（gemm_exact_check は注意を出すだけ）
    set_("OZ_FP64", Cls::A, W_I, R_FIELD, Grp::None, "スキーム II の自動決定は OZ2_FP64"),   // auto_p=
    set_("OZ2_FP64", Cls::A, W_II, R_FIELD, Grp::None, "スキーム I の自動決定は OZ_FP64"),   // auto_b=
    set_("OZ2_NOWIN", Cls::A, W_II, R_FIELD),                                          // win_meas=0
    chc_("OZ2_PASSES", Cls::A, W_II, "1|2|auto", R_FIELD),                            // pass_mode=・two_pass=
    chc_("OZ_EXT_FMT", Cls::A, W_EXT, "bf16|tf32", R_FIELD),                          // ext_fmt=
    u32d("OZ_EXT_SIG", Cls::A, W_EXT, 8, 11, 11, R_FIELD),                            // ext_sig=
    val_("OZ_MIN_BITS", Cls::A, W_OZ, Kind::F64, R_FIELD, Grp::Input),               // min_bits=
    chc_("OZ_TIMING", Cls::A, W_TIMED, "cold|steady", R_FIELD, false),               // timing=（oz_timing.hpp が読む）
    val_("OZ_WARMUP_SEC", Cls::A, W_TIMED, Kind::F64, R_FIELD),
    val_("OZ_WARMUP_WIN_SEC", Cls::A, W_TIMED, Kind::F64, R_FIELD),
    val_("OZ_WARMUP_TOL_PCT", Cls::A, W_TIMED, Kind::F64, R_FIELD),
    val_("OZ_WARMUP_MAX_SEC", Cls::A, W_TIMED, Kind::F64, R_FIELD),
    val_("OZ_WARMUP_GAP_MS", Cls::A, W_TIMED, Kind::F64, R_FIELD),
    val_("OZ_WARMUP_HOT_FACTOR", Cls::A, W_TIMED, Kind::F64, R_FIELD),
    val_("OZ_TELEM_MS", Cls::A, W_TIMED, Kind::I64, R_FIELD),                         // telemetry.hpp が読む
    val_("OZ_TELEM_DEV", Cls::A, W_TIMED, Kind::I64, R_FIELD),
    val_("OZ_TELEM_FW", Cls::A, W_TIMED, Kind::I64, R_NONE),
    val_("OZ_TELEM_TRACE", Cls::A, W_TIMED, Kind::U32, R_NONE),                       // 0 か 1（oz_timing.hpp の timing_cfg）
    // 全体の時間（DGEMM 1 回の呼び出しの通し．2026-11-03 から論文の Total．それより前は診断で diag に数え probe=set だった）
    u32d("OZ_TIME_WHOLE", Cls::A, W_OZ, 0, 1, 0, R_FIELD),                           // time_whole=・whole_ms=
    // ---- B アブレーション（knobs 欄．この並びが欄の中の並び）----
    set_("OZ_NO_DEFER", Cls::B, W_I, R_KNOBS),
    set_("OZ_NO_FUSE_ACC", Cls::B, W_I, R_KNOBS),
    set_("OZ_NO_PAIRLOOP", Cls::B, W_I, R_KNOBS),
    set_("OZ_NO_BLOCKMM", Cls::B, W_OZ | W_I8 | W_EXACT, R_KNOBS),
    set_("OZ_SOFTFLOAT", Cls::B, W_I, R_KNOBS),
    u32_("OZ_NSMALL", Cls::B, W_I, 0, 65535, R_KNOBS, Grp::None, "スキーム II の断片の本数は OZ2_NSMALL"),   // 上界は recon_scale
    u32_("OZ_BANDS", Cls::B, W_OZ, 1, 65535, R_KNOBS),        // スキーム II はパネル行を等分できる本数だけ（plan_bands）
    // OZ2_JSPLIT は答えを変えないが FP64 組み立ての分担を変える診断（docs の C）で，knobs にも数える
    u32d("OZ2_JSPLIT", Cls::C, W_II, 0, 1024, 512, R_KNOBS | R_DIAG),
    u32_("OZ2_MOD_GROUPS", Cls::B, W_II, 1, 1024, R_KNOBS | R_FIELD),                  // 上限は法の本数（main が見る）．groups=
    // gemm_bf16 は読まない（構成探索が tt-metal の例の判定で L1 の予算を見ない．立っていれば main の入口で止める）
    u32d("OZ_L1_BUDGET_KB", Cls::B, (uint16_t)(W_GEMM & ~W_BF16), 1, 1536, OZ_L1_BUDGET_KB_DEFAULT, R_KNOBS, Grp::Gemm),
    str_("OZ2_PANEL_FORCE", Cls::B, W_II, R_KNOBS),                                    // RxC（書式は oz2_parse_panel_force）
    // ---- C 診断（diag 欄．この並びが欄の中の並び）----
    val_("OZ_RADIX", Cls::C, W_I, Kind::U32, R_DIAG),            // 範囲は形式から（parse_run_config）．radix=
    val_("OZ_DMAX", Cls::C, W_I, Kind::U32, R_DIAG),             // 範囲は形式から（ozaki_digit_max）．dmax=
    chc_("OZ_FIDELITY", Cls::C, W_OZ | W_BF16 | W_EXT | W_I8, "LoFi|HiFi2|HiFi3|HiFi4", R_DIAG),   // fidelity=
    set_("OZ_NOBLOCK", Cls::C, W_I, R_DIAG),
    u32_("OZ2_B", Cls::C, W_II, 1, 100, R_DIAG),                 // 上限は A' を 2^20 進 5 桁で運べる幅（ozaki2_main.cpp が確かめる）
    str_("OZ2_MODULI", Cls::C, W_II, R_DIAG | R_FIELD),          // 書式は choose_moduli_and_width．forced_moduli=
    u32d("OZ2_CRT_PROBE", Cls::C, W_II, 0, 2, 0, R_DIAG | R_FIELD),                  // crt_probe=
    u32d("OZ_GEMM_PROBE_NODRAM", Cls::C, W_OZ, 0, 1, 0, R_DIAG | R_FIELD, Grp::Gemm),   // gemm_probe=
    u32d("OZ_GEMM_PROBE_NOMCAST", Cls::C, W_OZ, 0, 1, 0, R_DIAG | R_FIELD, Grp::Gemm),
    u32d("OZ_GEMM_PROBE_NOWRITE", Cls::C, W_OZ, 0, 1, 0, R_DIAG | R_FIELD, Grp::Gemm),
    u32d("OZ_RECON_PROBE", Cls::C, W_OZ, 0, 3, 0, R_DIAG | R_FIELD),   // スキーム II は 0・1 だけ（oz2_probes が止める）．recon_probe=
    u32d("OZ2_RECON_NOF64", Cls::C, W_II, 0, 1, 0, R_DIAG | R_FIELD),                // nof64=
    u32_("OZ2_NSMALL", Cls::C, W_II, 0, 64, R_DIAG | R_FIELD),    // 上限は経路の断片数（main が見る）．nsmall_req=
    set_("OZ2_NOSMALL", Cls::C, W_II, R_DIAG | R_FIELD),
    set_("OZ2_ALLSMALL", Cls::C, W_II, R_DIAG | R_FIELD),
    u32d("OZ2_MAXCI", Cls::C, W_II, 0, 1, 0, R_DIAG | R_FIELD),                      // maxci=
    // ---- D 入力の作り方（oz_input.hpp の input_env が読む．input= に残る）----
    val_("OZ_WIDE", Cls::D, W_INPUT, Kind::F64, R_FIELD, Grp::Input),
    val_("OZ_POS", Cls::D, W_INPUT, Kind::F64, R_FIELD, Grp::Input),
    val_("OZ_CONST", Cls::D, W_INPUT, Kind::F64, R_FIELD, Grp::Input),
    {"OZ_CONST_SIGN", Cls::D, W_INPUT, Kind::Choice, false, 0, 0, false, 0, R_FIELD, Grp::Input, "col|half|halfeps", nullptr},
    val_("OZ_ROWSCALE", Cls::D, W_INPUT, Kind::F64, R_FIELD, Grp::Input),
    val_("OZ_EXPDIFF", Cls::D, W_INPUT, Kind::I64, R_FIELD, Grp::Input),
    val_("OZ_EXPK", Cls::D, W_INPUT, Kind::I64, R_FIELD, Grp::Input),
    // ---- D2 検査の書き出し ----
    str_("OZ_DUMP_C", Cls::D2, W_INPUT, R_NONE, Grp::Input),     // DUMP 行（空の値は「書き出さない」）
    u32d("OZ2_CHECK_CONCAT", Cls::D2, W_II, 0, 1, 0, R_DIAG | R_FIELD),             // check_concat=
    // ---- E その他 ----
    set_("OZ_DBG", Cls::E, W_I, R_KNOBS),                         // 計時区間の中で表示と同期を足すので knobs に残す
    // ---- F 検査・探針・道具のつまみ（尾崎のドライバは読まない）----
    u32d("TWOSUM_PROBE_ABS", Cls::F, W_TOOL, 0, 1, 0, R_NONE),
    u32d("TWOSUM_PROBE_FAST", Cls::F, W_TOOL, 0, 1, 0, R_NONE),
    u32d("TWOSUM_PROBE_ALL", Cls::F, W_TOOL, 0, 1, 0, R_NONE),
    u32d("TFMUL_VERBOSE", Cls::F, W_TOOL, 0, 1, 0, R_NONE),
    u32d("AWE_DEBUG", Cls::F, W_TOOL, 0, 1, 0, R_NONE),
    str_("OZ_ROUTES_TSV", Cls::F, W_TOOL, R_NONE),
    u32d("PROBE_FP32ACC", Cls::F, W_PROBE, 0, 1, 1, R_NONE),
    u32d("PROBE_DUMP", Cls::F, W_PROBE, 0, 1, 0, R_NONE),
    u32d("PROBE_MISMATCH", Cls::F, W_PROBE, 0, 1, 0, R_NONE),
    u32d("DST_PROBE_VERBOSE", Cls::F, W_PROBE, 0, 1, 0, R_NONE),
    u32d("SCALE_CHECK_TINY", Cls::F, W_CHECK, 0, 1, 1, R_NONE),
    // ---- 削除したつまみ（最後に含むコミット．立っていれば止める）----
    // 調査の済んだ実験のつまみ
    gone("OZ_AUTO", "edb23d3"), gone("OZ_NO_PAIRWISE", "edb23d3"), gone("OZ_NO_RAGGED", "edb23d3"),
    gone("OZ_PER_CORE_M", "edb23d3"), gone("OZ_PER_CORE_N", "edb23d3"), gone("OZ_SPILL_BW", "edb23d3"),
    gone("OZ2_CRT_HOIST", "edb23d3"), gone("OZ2_CHUNK", "edb23d3"),
    // GEMM 段の別の実装と調査用の切り替え（GEMM 段は分割データ移動＋両端の 2 コアから配る構成＋パッカの L1 累積の
    // 1 通りだけを持つ．docs/gemm_opt_results.md）
    gone("OZ_GEMM_LEGACY", "e3becaa"), gone("OZ_NO_SPLITRD", "e3becaa"), gone("OZ_NO_L1ACC", "e3becaa"),
    gone("OZ_GEMM_DUAL", "e3becaa"), gone("OZ_GEMM_NBUF", "e3becaa"), gone("OZ_GEMM_SB8", "e3becaa"),
    gone("OZ2_PANEL", "e3becaa"), gone("OZ2_BW", "e3becaa"), gone("OZ1_PC", "e3becaa"),
    // スキーム I の発行順と同期回数を測る計測用のつまみ（INT8 の ratio_iss_*．docs/kblock_plan.md だけが引いた）と，
    // どのスクリプトも立てなかった無誤差条件の厳格モード．論文の値を測った tag measured-20261021 の版（042dc00）に含む
    gone("OZ_SYNC_EACH", "042dc00"), gone("OZ_PANEL_RR", "042dc00"), gone("OZ_STRICT_EXACT", "042dc00"),
    // どのスクリプトも立てず，用途を文書にしていない調査のつまみ
    gone("OZ2_DBGSMALL", "58dce32"),
};
inline constexpr int NKNOB = (int)(sizeof(TABLE) / sizeof(TABLE[0]));

// 名前から表の行の番号（無ければ -1）．OZK("名前") はコンパイル時に引き，表に無い名前はコンパイルを止める
constexpr int index_of(std::string_view n) {
    for (int i = 0; i < NKNOB; ++i)
        if (TABLE[i].name == n) return i;
    return -1;
}
template <int I> constexpr int checked() {
    static_assert(I >= 0, "oz_knobs.hpp の表に無いつまみの名前");
    return I;
}
#define OZK(name) (::oz_knobs::checked<::oz_knobs::index_of(name)>())

// 表の形の検査（名前の重複・範囲・既定・スキームをまたぐ拒否の前提）．コンパイル時に確かめる
constexpr bool table_ok() {
    for (int i = 0; i < NKNOB; ++i) {
        const Knob& k = TABLE[i];
        for (int j = 0; j < i; ++j)
            if (TABLE[j].name == k.name) return false;                     // 名前の重複
        if (k.ranged && (k.kind == Kind::U32 || k.kind == Kind::I64 || k.kind == Kind::F64)) {
            if (!(k.lo <= k.hi)) return false;
            if (k.has_dflt && (k.dflt < k.lo || k.dflt > k.hi)) return false;
        }
        if (k.kind == Kind::U32 && k.ranged && k.lo < 0) return false;
        // 範囲を表で持つのは U32 と Choice だけ（ranged の I64・F64 を読む関数も preflight の枝も無い．足すなら両方を足す）
        if (k.ranged && (k.kind == Kind::I64 || k.kind == Kind::F64)) return false;
        // GEMM 段のつまみは範囲を表で持つ U32 で，BF16 の固定予算の基準実装以外の GEMM ドライバが読むか（L1 の予算），尾崎の 2 つだけが
        // 読む（探針）．GEMM 段のつまみの値は入口の preflight が検査する（範囲を表で持つ U32 だけを読むので，この形に限る）
        if (k.grp == Grp::Gemm && (k.kind != Kind::U32 || !k.ranged || !(k.who == (W_GEMM & ~W_BF16) || k.who == W_OZ)))
            return false;
        if (k.kind == Kind::Choice && !k.choices) return false;
        // スキーム I は II のつまみを OZ2_ の接頭辞で止めるので，II だけが読むつまみはその名前でなければならない
        if ((k.who & W_OZ) == W_II && k.name.substr(0, 4) != "OZ2_") return false;
        // 逆に OZ2_ のつまみを I が読むと，I の入口が自分のつまみを止めてしまう
        if ((k.who & W_I) && k.name.substr(0, 4) == "OZ2_") return false;
        // 記録はつまみを読む尾崎のドライバにだけ意味がある
        if ((k.rec & (R_KNOBS | R_DIAG)) && !(k.who & W_OZ)) return false;
        // knobs・diag に値を書くには表が範囲を持つか，有無か文字列で読むこと
        if ((k.rec & R_KNOBS) && !(k.kind == Kind::Set || k.kind == Kind::Str || k.ranged)) return false;
        if (k.cls == Cls::Removed && (!k.note || k.who)) return false;
    }
    return true;
}
static_assert(table_ok(), "oz_knobs.hpp の表の形が崩れている（名前の重複・範囲・接頭辞・記録の前提．table_ok の各行）");

inline const Knob& at(int id) { return TABLE[id]; }
inline std::string name(int id) { return std::string(TABLE[id].name); }
inline const char* get(int id) { return std::getenv(name(id).c_str()); }
inline bool set(int id) { return get(id) != nullptr; }

// 既定値のある整数のつまみ．既定値を持たないものは u32_opt で未設定を区別する．
inline uint32_t u32(int id) {
    const Knob& k = TABLE[id];
    if (k.kind != Kind::U32 || !k.ranged || !k.has_dflt) {
        std::fprintf(stderr, "ERROR: oz_knobs::u32 は既定値と範囲を持つ整数だけを読む（%s）\n", name(id).c_str());
        std::abort();
    }
    const char* v = get(id);
    if (!v) return (uint32_t)k.dflt;
    return oz_env::parse_u32(name(id).c_str(), v, (uint32_t)k.lo, (uint32_t)k.hi);
}
// 既定を持たない整数のつまみ（OZ2_B・OZ_BANDS・OZ2_MOD_GROUPS・OZ_NSMALL・OZ2_NSMALL など）を読む：未設定なら nullopt，
// 立っていれば範囲を検査した値（読めなければ u32 と同じく止める）
inline std::optional<uint32_t> u32_opt(int id) {
    const Knob& k = TABLE[id];
    if (k.kind != Kind::U32 || !k.ranged || k.has_dflt) {
        std::fprintf(stderr, "ERROR: oz_knobs::u32_opt は既定を持たないつまみだけを読む（%s は u32 で読む）\n", name(id).c_str());
        std::abort();
    }
    const char* v = get(id);
    if (!v) return std::nullopt;
    return oz_env::parse_u32(name(id).c_str(), v, (uint32_t)k.lo, (uint32_t)k.hi);
}
// 列挙のつまみ：値が choices の何番目か（未設定なら -1）．どれでもなければ止める（照合と誤りの文は oz_env::choice_in）
inline int choice(int id) {
    const char* v = get(id);
    if (!v) return -1;
    return oz_env::choice_in(name(id).c_str(), v, TABLE[id].choices);
}

// 既定から外れているか（冒頭の決め方）
inline bool active(int id) {
    const Knob& k = TABLE[id];
    if (k.has_dflt && k.kind == Kind::U32) return u32(id) != (uint32_t)k.dflt;
    return set(id);
}
// knobs 欄の 名前:値 の値の部分（有無のつまみは空）
inline std::string value_text(int id) {
    const Knob& k = TABLE[id];
    if (k.kind == Kind::Set) return "";
    if (k.kind == Kind::U32) {
        if (k.has_dflt) return std::to_string(u32(id));
        const auto v = u32_opt(id);
        return v ? std::to_string(*v) : std::string();
    }
    const char* v = get(id);
    return v ? std::string(v) : std::string();
}

// CONFIG 行の knobs 欄（rec=R_KNOBS．既定から外れたものを表の並びに '+' でつなぎ，値を持つものは 名前:値）・diag 欄
// （rec=R_DIAG．名前だけ）の値．どちらも無ければ none
inline std::string record(uint16_t who, Rec rec) {
    std::string out;
    for (int i = 0; i < NKNOB; ++i) {
        const Knob& k = TABLE[i];
        if (!(k.who & who) || !(k.rec & rec) || !active(i)) continue;
        std::string t = name(i);
        if (rec == R_KNOBS) {
            const std::string v = value_text(i);
            if (!v.empty()) t += ":" + v;
        }
        out += (out.empty() ? "" : "+") + t;
    }
    return out.empty() ? std::string("none") : out;
}

// 入口の検査：who が読み，範囲・列挙を表で持つつまみを全部読んで検査する（不正なら終了コード 2）．
// 使う所で初めて読むと，打ち間違いがデバイスを開いた後に見つかって閉じずに終わる
inline void preflight(uint16_t who) {
    for (int i = 0; i < NKNOB; ++i) {
        const Knob& k = TABLE[i];
        if (!(k.who & who) || !k.ranged) continue;
        if (k.kind == Kind::U32) {
            if (k.has_dflt) (void)u32(i);
            else (void)u32_opt(i);
        }
        else if (k.kind == Kind::Choice) (void)choice(i);
    }
}

// 削除したつまみ（docs/env_vars.md の末尾）が残っていたら止める（黙って無視すると，旧設定のつもりで既定の構成を測る）．
// デバイスを開く前に呼ぶのは，oz_driver.hpp の 2 つの入口（尾崎の 2 つのドライバの reject_env と，基準実装・gemm_exact_check の
// reject_env_gemm）と，その入口を通らない検査 scale_check・slice_split_check
inline void reject_removed() {
    bool bad = false;
    for (int i = 0; i < NKNOB; ++i)
        if (TABLE[i].cls == Cls::Removed)
            if (const char* v = get(i)) {
                std::fprintf(stderr, "ERROR: %s=%s は削除したつまみで，もう効かない（docs/env_vars.md．再現は git checkout %s）\n",
                             name(i).c_str(), v, TABLE[i].note);
                bad = true;
            }
    if (bad) std::exit(2);
}

// もう一方のスキームだけが読むつまみを止める（黙って無視すると，OZ_FP64 で選び直したつもりの性能測定が既定の構成を測る．
// docs/env_vars.md は OZ_FP64 と OZ2_FP64 を 1 行に並べる）．スキーム I は OZ2_ の接頭辞の変数をすべて止める（表に無い
// 打ち間違いも止める．表は II だけのつまみを OZ2_ の名前にしてある．table_ok）．スキーム II は表で I だけが読むものを止める．
// どちらも空の値も立っているとみなす．drv はドライバの名前（argv[0] の最後の要素）
inline void reject_other_scheme(uint16_t self, const char* drv) {
    bool bad = false;
    auto hint = [](const char* nm) -> std::string {
        const int i = index_of(nm);
        return (i >= 0 && TABLE[i].note && TABLE[i].cls != Cls::Removed) ? std::string("（") + TABLE[i].note + "）"
                                                                       : std::string();
    };
    if (self == W_I) {
        for (char** e = ::environ; e && *e; ++e)
            if (std::strncmp(*e, "OZ2_", 4) == 0) {
                const char* eq = std::strchr(*e, '=');
                const std::string nm = eq ? std::string(*e, (size_t)(eq - *e)) : std::string(*e);
                std::fprintf(stderr, "ERROR: %s はスキーム II（ozaki2_*）だけのつまみで，%s では効かない%s\n", *e, drv,
                             hint(nm.c_str()).c_str());
                bad = true;
            }
    } else {
        for (int i = 0; i < NKNOB; ++i)
            if ((TABLE[i].who & W_OZ) == W_I)
                if (const char* v = get(i)) {
                    std::fprintf(stderr, "ERROR: %s=%s はスキーム I（ozaki1_*）だけのつまみで，%s では効かない%s\n",
                                 name(i).c_str(), v, drv, hint(name(i).c_str()).c_str());
                    bad = true;
                }
    }
    if (bad) std::exit(2);
}

// 名前を指定して拒否する（使うつまみの読み取りとは区別する）．空の DUMP_C だけは「書き出さない」．
// who にこのプログラムを足してはいけない：拒否するための検査であり，計算には使わない．
inline void reject_named(const char* nm, const char* drv, const char* why) {
    const int id = index_of(nm);
    if (id < 0) { std::fprintf(stderr, "ERROR: unknown rejected knob %s\n", nm); std::exit(2); }
    if (const char* v = get(id); v && (*v || TABLE[id].name != "OZ_DUMP_C")) {
        std::fprintf(stderr, "ERROR: %s=%s は %s では効かない（%s）\n", nm, v, drv, why);
        std::exit(2);
    }
}

// ---- 読まないつまみの拒否 ----
// self（読むプログラムのビット．0 なら「どのプログラムでもない」）が読まない A〜E のつまみ（表の who に self が無いもの）が
// 既定から外れて立っていれば，理由を添えてデバイスを開く前に止める（黙って無視すると，そのつまみの構成・入力・基準・
// 書き出しで測ったつもりになる．HiFi2 で確かめたつもりの gemm_exact_check が LoFi で通る，など）．「既定から外れて」は
// active の決め方で，既定を持つ 0/1 などのつまみは既定の値なら通す（GEMM 段の探針の =0 は既定と同じ構成）．
// sel は数える組（下の S_*．S_ALL が入口の規則 reject_unread で，S_INPUT の reject_unread_input はその特別な場合）．例外：
//   ・OZ_TELEM_DEV は scripts/run.sh がコンテナのどのコマンドにも渡す（テレメトリを読むカードの番号）ので，読まない
//     プログラムでも止めない
//   ・空の OZ_DUMP_C は「書き出さない」（check_env・dump_result）なので止めない．ほかの名前は空でも止める（読むドライバでは
//     空は値の検査で止まるので，読まないドライバでも通さない）
//   ・尾崎の 2 つのドライバ（self が W_I か W_II）では，もう一方のスキームだけのつまみは数えない（reject_other_scheme が
//     案内つきで止める）
//   ・F（検査・探針・道具のつまみ）と削除したつまみ（reject_removed）は数えない
// 誤りの文は組で分ける：入力の生成・判定・書き出しの組（Grp::Input）は why_input を添え，答えを壊す GEMM 段の探針
// （Grp::Gemm の C）は「尾崎のドライバでだけ使える」（立てると CONFIG 行へ probe=set を残せるのは尾崎の 2 つだけ．素の GEMM の
// ドライバは探針を McastGemmSpec の probe に詰めないので，止めなくても答えは壊れないが，探針を立てたつもりの測定を黙って
// 通さない），ほかは「読まない」．drv はドライバの名前
enum Sel : uint8_t { S_INPUT = 1, S_PROBES = 2, S_OTHER = 4, S_ALL = 7 };   // 数える組（入力・GEMM 段の探針・ほか）
inline void reject_unread_sel(uint16_t self, const char* drv, const char* why_input, uint8_t sel) {
    bool bad = false;
    for (int i = 0; i < NKNOB; ++i) {
        const Knob& k = TABLE[i];
        if (k.cls == Cls::F || k.cls == Cls::Removed || (k.who & self)) continue;
        if (k.name == "OZ_TELEM_DEV") continue;
        if ((self == W_I || self == W_II) && (k.who & W_OZ) && (k.who & W_OZ) != self) continue;
        const bool input = k.grp == Grp::Input, probe = k.grp == Grp::Gemm && k.cls == Cls::C;
        if (!(sel & (input ? S_INPUT : probe ? S_PROBES : S_OTHER))) continue;
        const char* v = get(i);
        if (!v || (!*v && k.name == "OZ_DUMP_C") || !active(i)) continue;
        if (input)
            std::fprintf(stderr, "ERROR: %s=%s は %s では効かない（%s）\n", name(i).c_str(), v, drv, why_input);
        else if (probe)
            std::fprintf(stderr, "ERROR: %s=%s は尾崎のドライバでだけ使える（%s は答えが壊れたことをログへ probe=set と"
                                 "残せない）\n", name(i).c_str(), v, drv);
        else
            std::fprintf(stderr, "ERROR: %s=%s は %s では読まない（つまみの表 common/oz_knobs.hpp の読むプログラムに無い．"
                                 "黙って無視すると，その構成で測ったつもりになる）\n", name(i).c_str(), v, drv);
        bad = true;
    }
    if (bad) std::exit(2);
}
// 入口の規則（組の全部）．尾崎の 2 つのドライバと，尾崎以外のドライバ（基準実装・gemm_exact_check）が oz_driver.hpp の入口で呼ぶ
inline void reject_unread(uint16_t self, const char* drv, const char* why_input) { reject_unread_sel(self, drv, why_input, S_ALL); }
// 入力の生成・判定・書き出しの組（表の Grp::Input：入力の作り方・OZ_PHI・OZ_MIN_BITS・OZ_DUMP_C）だけ
inline void reject_unread_input(uint16_t self, const char* drv, const char* why) { reject_unread_sel(self, drv, why, S_INPUT); }
// 入力を自前で作り，Dot2 の参照解も結果の書き出しも持たない道具用：組の全部を止める（tools の dot2check・splitsim．
// 入力を条件ごとに自前で作る）
inline void reject_unused_input(const char* drv) {
    reject_unread_input(0, drv, "入力の作り方・精度の基準・結果の書き出しを持たない");
}

// 表に無い OZ2_ の名前（打ち間違い）を止める．OZ2_ のつまみはどれも表にあるので，表に無い名前は読まれずに黙って無視される．
// OZ_ の名前は止めない：表に無い OZ_ の変数をスクリプトがコンテナへ渡す（run.sh の OZ_IMAGE_ID，build_all.sh・
// scripts/study/measure_paper.sh に渡す OZ_PROVENANCE）ので，接頭辞では打ち間違いと区別できない（区別するには表の外の名前の
// 一覧が要り，手で持つ一覧が 2 つになる）．スキーム I の入口は OZ2_ の変数をすべて止める（reject_other_scheme）ので，
// これを呼ぶのはスキーム II と尾崎以外のドライバ
inline void reject_unknown_oz2(const char* drv) {
    bool bad = false;
    for (char** e = ::environ; e && *e; ++e) {
        if (std::strncmp(*e, "OZ2_", 4) != 0) continue;
        const char* eq = std::strchr(*e, '=');
        const std::string nm = eq ? std::string(*e, (size_t)(eq - *e)) : std::string(*e);
        if (index_of(nm) >= 0) continue;
        std::fprintf(stderr, "ERROR: %s はつまみの表（common/oz_knobs.hpp・docs/env_vars.md）に無い名前で，%s は読まない（打ち間違い？）\n",
                     *e, drv);
        bad = true;
    }
    if (bad) std::exit(2);
}

}  // namespace oz_knobs
