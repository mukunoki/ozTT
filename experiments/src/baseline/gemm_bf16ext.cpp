// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// BF16 extended と 3xTF32（オペランドの分割による行列積）
// ---------------------------------------------------------------------------
// 論文「累積経路の制約と累積幅」の節（ssec:where）の 3xBF16．FP32 のエミュレーションではなく，あくまで
// 低精度形式の「拡張」である．FP32 のオペランドを低精度の成分に分けて精度を上げる手法だが，
// 本ハードウェアでは FP32 の約 24 bit に届かない．どこで止まるかは成分の形式で違う．
//   BF16 成分（既定）  : 約 12.5 bit で頭打ち．行列エンジンの積和が FP32 累算器の前段で
//                        8 項ごとに最大指数へ整列され，そこで下位が落ちる（累算側の制約）．
//   TF32 成分（OZ_EXT_FMT=tf32）: 約 11.1 bit とさらに低い．乗算器は SrcA 側の仮数の最下位
//                        bit をどのパスでも読まない（論文の表 tab:field）ので，11 bit の仮数を
//                        持つ TF32 は最上位の部分積 Ah*Bh の時点で無誤差でない（乗算側の制約）．
//   TF32 成分を有効仮数 10 bit に丸めたもの（OZ_EXT_SIG=10）: 12.6 bit．乗算側の制約を
//                        避けても，累算側の制約で BF16 と同じ位置に止まる．
// 値は本プログラムの乱数入力・HiFi4・FP32 累算での測定値で，普遍的な上限ではないが，
// 止まる原因はどちらも実装ではなくハードウェア側にある．本研究が FP64 相当の精度を得る
// 手段として，整数化して無誤差累算に持ち込む尾崎スキームを用いるのはこのためである．
// 測定値は docs/exponent_format_constraint.md 追記 6．
//
// 原理（NVIDIA の 3xTF32 と同型のものを Tensix の行列エンジンへ写したもの．BF16 の場合で書く）
//   FP32 の値を BF16 3 成分の和に無誤差分割する（仮数 8 bit × 3 = 24 bit で FP32 を覆う．
//   前提は下の「分割の前提」）．積は 9 本の BF16 行列積の和になり，
//   そのうち大きい方から 6 本（約 2^-16 まで）を採る：
//       A = Ah + Am + Al ,  B = Bh + Bm + Bl
//       C ≒ Ah*Bh                                  （レベル 0，約 2^0）
//          + Ah*Bm + Am*Bh                          （レベル 1，約 2^-8）
//          + Ah*Bl + Am*Bm + Al*Bh                  （レベル 2，約 2^-16）
//       （Am*Bl, Al*Bm の約 2^-24 と Al*Bl の約 2^-32 は捨てる）
//   各部分積は BF16×BF16 の素の行列積を FP32 累算で行う．部分積はホスト側で
//   小さいレベルから順に足して丸めを抑える．要素ごとの多倍長演算は使わない：
//   分割の粒度を行列積にするのが，行列エンジンにとって都合がよいからである．
//
// プログラムの組み立て（データ移動は A/B を 2 つの RISC に分けて両端の 2 コアから配る
// common/kernels/mcast_split_dm.cpp）と構成の自動選択は gemm_bf16・尾崎の各経路と共有する
// （common/mcast_gemm.hpp の build_mcast_gemm と choose_cfg）が，計算カーネルは専用の
// common/kernels/bmm_l1acc.cpp に差し替える．標準のカーネルとの本質的な違いは出力形式ではなく，K ブロックの部分和を
// パッカの L1 累積で足し込み，読み戻しを最後の 1 回にすることである（詳細はそのカーネルの冒頭）．
//
// 使い方:
//   gemm_bf16ext [M] [N] [K] [warmup] [iters] [terms]
//   既定値:        1280 1408 1024 1 3 6
//   M N K は正の 32 の倍数，warmup は 0..1000000（0 でも行列積の JIT は計時に入らない），iters は 1..1000000
//   （common/oz_driver.hpp の read_dims・read_counts）．引数は 6 個まで（多ければデバイスを開く前に 2 で止める）．
//   terms: 1（Ah*Bh のみ＝単一 GEMM）| 3（レベル 0+1．成分 2 つ）| 6（レベル 0+1+2．成分 3 つ）
//   レベルの大きさは BF16 で 2^0, 2^-8, 2^-16，TF32 で 2^0, 2^-11, 2^-22 程度．
//
// 環境変数
//   OZ_EXT_FMT=tf32  成分を TF32 にする．TF32 は CB 形式として存在しないので，成分を FP32 で
//                    格納し，fp32_dest_acc_en の下でアンパッカに TF32 へ展開させる（成分は
//                    最初から TF32 の値なので，展開で値は変わらない）．成分 2 つ・部分積 3 本
//                    （terms=3）が NVIDIA の 3xTF32 と同じ構成である．
//   OZ_EXT_SIG=n     TF32 成分の有効仮数幅（暗黙の 1 を含む．8..11，既定 11）．10 にすると
//                    SrcA が読まない最下位 bit を使わない成分になり，乗算側の制約を除いて
//                    累算側の限界だけを見られる．11 を超える幅はアンパッカが TF32 へ縮めて
//                    しまい，8 未満では 3 成分で FP32 の 24 bit を覆えないので受け付けない．
//   OZ_FIDELITY      MathFidelity（既定 HiFi4．縮約が乗算側か累算側かを切り分けるマイクロベンチ用）．
//                    ピーク比もこの fidelity のピークに対して出す（common/oz_fidelity.hpp）
//   OZ_NOVERIFY      参照解の計算と精度の評価を省く（スループットだけ見るとき．RESULT の精度欄は nan，最後の行は
//                    "Test Skipped"．OZ_DUMP_C とは併用できない（check_env が止める））
//
// 計時の範囲（CONFIG の timing_scope=gemm_only）：RESULT の時間と TFLOP/s は部分積の行列積のデバイス実行時間の
// 和から出す．ホストでの成分への分割・タイル化・パネルの切り出し・部分積の足し込みは含まない．
// 終了コード：到達精度に合否の基準は無い（精度が測定の対象）ので最後の行は "Completed"（検証を省いたときは
// "Test Skipped"）で，失敗（1）にするのは出力か精度の指標が有限でないときと，OZ_DUMP_C を書けなかったときと，
// デバイスを閉じられなかったときだけ．入力が FP32 で有限に表せない（OZ_CONST=1e100 など．FP32 の最大は約 3.4e38）ときは，
// 検証の有無によらずデバイスを開く前に 2 で止める（gemm_bf16 と同じ．検証を省く実行では精度を見ないので，止めないと
// 成功で終わってしまう）．
//
// 分割の前提：FP32 の値を丸めた成分と残差に分け，残差は FP32 で無誤差に求まる（成分は x を
// 最近接丸めした値で x の 2 倍以内にあるので，Sterbenz の補題により x - 成分 は厳密）．
// ただし成分の和が x に等しくなるには，残差も成分の形式で表せなければならない．これは
// 通常範囲の有限値に限る（FLT_MAX 付近は丸めの繰り上がりで無限大になり，最小の正規数
// 付近では 2^-126 + 2^-149 の残差 2^-149 のように成分に載らない値が出る）．host_eval の
// 乱数オペランドはその範囲に入らない．また入力は FP64 を float に丸めてから分けるので，参照の FP64 に対する誤差には，一般の入力では
// FP32 への丸め誤差（約 2^-24）も含まれる．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include "common/host_eval.hpp"
#include "common/oz_bits.h"         // float のビット列（round_to_prec）
#include "common/oz_fidelity.hpp"   // MathFidelity の上書き・表示名・パス数・理論ピーク（基準実装 3 つと尾崎で共通）
// 行列積の実行時パラメタ選択（OptConfig / choose_cfg）・プログラム構築（build_mcast_gemm）・行列積 1 本の実行
// （run_mcast_gemm）・パネルの切り出しは尾崎スキームと共通なので共有ヘッダのものを使う（mcast_gemm.hpp が
// ozaki_gemm_host.hpp を取り込む）．
#include "common/mcast_gemm.hpp"
#include "common/oz_driver.hpp"     // 入口の検査の並び（oz_drv::reject_env_gemm・read_dims・read_counts・check_env）・最後の行（finish）・CONFIG 行（OzLine）

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

// ---- FP32 の値を BF16 3 成分へ無誤差分割する ----
//   hi = RN_bf16(x)，r = x - (float)hi，mid = RN_bf16(r)，lo = RN_bf16(r - (float)mid)
//   hi は x を丸めた値で x の 2 倍以内にあるので，x - hi は FP32 で厳密（Sterbenz の補題）．
static inline void split3(float x, bfloat16& hi, bfloat16& mid, bfloat16& lo) {
    hi = bfloat16(x);
    float r = x - static_cast<float>(hi);
    mid = bfloat16(r);
    float r2 = r - static_cast<float>(mid);
    lo = bfloat16(r2);
}

// ---- 成分の有効仮数幅（暗黙の 1 を含む）．既定は TF32 の 11 ----
// 分割は要素ごとに呼ばれるので，環境変数は最初の 1 回だけ読む（範囲 8..11 と既定はつまみの表）．
static int ext_sig() {
    static const int n = (int)oz_knobs::u32(OZK("OZ_EXT_SIG"));
    return n;
}
// ---- FP32 の値を精度 prec bit（暗黙の 1 を含む仮数の桁数．binary32 は 24，TF32 は 11）へ最近偶数丸めする ----
// ビット列の整数加算で丸める：落とす下位 drop bit に「半分 - 1」と，残す最下位 bit
// （同点のとき偶数へ寄せるため）を足してから切り捨てる．仮数が桁あふれすれば指数へ
// 繰り上がり，値として正しい（2 のべきになる）．
static inline float round_to_prec(float x, int prec) {
    if (!std::isfinite(x) || prec >= 24) return x;
    const int drop = 24 - prec;                      // 落とす明示仮数のビット数（TF32 なら 13）
    uint32_t u = f32_bits(x);
    u += ((1u << (drop - 1)) - 1u) + ((u >> drop) & 1u);
    u &= ~((1u << drop) - 1u);
    return bits_f32(u);
}
static inline float round_tf32(float x) { return round_to_prec(x, ext_sig()); }
// ---- FP32 の値を TF32（既定）3 成分へ分割する ----
//   hi = RN(x)，r = x - hi，mid = RN(r)，lo = RN(r - mid)（RN は ext_sig() bit への最近偶数丸め）
//   hi は x を最近接丸めした値で x の 2 倍以内にあるので，x - hi は FP32 で厳密（Sterbenz）．
//   各段で有効 ext_sig() bit（8..11）ずつ取り出し，3 段で 24 bit 以上を覆うので，冒頭の
//   「分割の前提」が満たされる限り 3 成分の和は x に等しい．
static inline void split3(float x, float& hi, float& mid, float& lo) {
    hi = round_tf32(x);
    float r = x - hi;
    mid = round_tf32(r);
    lo = round_tf32(r - mid);
}

// 成分の格納形式．OZ_EXT_FMT=tf32 なら FP32 で格納して TF32 として掛ける（列挙 bf16|tf32 はつまみの表．既定 bf16）
static bool ext_tf32() {
    static const bool on = oz_knobs::choice(OZK("OZ_EXT_FMT")) == 1;
    return on;
}


// 成分どうしの行列積を 1 本，FP32 出力で実行する．T は bfloat16（BF16 格納）か
// float（FP32 格納，TF32 として乗算）．
// 戻り値は温めの後に計時した 1 反復あたりの秒数（温め方は OZ_TIMING．既定の cold では定常状態ではない．
// oz_timing.hpp の「計時」）．Cf（行優先 M*N，FP32）に結果を書く．
template <typename T>
static double gemm_fp32out(
    std::shared_ptr<distributed::MeshDevice>& md, const OptConfig& g, tt::tt_metal::MathFidelity fid,
    const std::vector<T>& A_tilized, const std::vector<T>& B_tilized,
    uint32_t M, uint32_t N, uint32_t K, std::vector<float>& Cf,
    uint32_t warmup, uint32_t iters) {
    constexpr bool kF32 = std::is_same_v<T, float>;
    const tt::DataFormat in_fmt = kF32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t Mt = M / TILE_HEIGHT, Kt = K / TILE_WIDTH, Nt = N / TILE_WIDTH;

    // プログラムの組み立てと実行は gemm_bf16 などと共有する（common/mcast_gemm.hpp の build_mcast_gemm・run_mcast_gemm）．
    // c_16（最終出力）と c_24（中間の FP32 アキュムレータ）は別々の L1 領域で，
    // それぞれコアあたり出力ブロック 1 つぶんを占める．c_24 へはパッカが
    // K ブロックをまたいで加算していく（l1_acc）．
    // 計算カーネルは tt-metal 本番実装の PACKER_L1_ACC パターンを移植したもの．
    // K ブロック 0..N-2 はパッカが c_24 上で FP32 加算していき，累積結果を読み戻すのは
    // 最後のブロックの 1 回だけにする．その読み戻しは FP32 でなければならないので，
    // c_24 に UnpackToDestFp32 を設定する（詳細は common/kernels/bmm_l1acc.cpp）．
    // CB・UnpackToDestFp32・コンパイル時引数は gemm_exact_check（l1acc）と共有する
    // （common/mcast_gemm.hpp の l1acc_gemm_spec）．
    const McastGemmSpec spec = l1acc_gemm_spec(g, Kt, in_fmt, fid);
    // BF16 成分の構成探索（main の choose_cfg．in_tile を渡さない tt-metal の例の判定）は c_16 と c_24 が BF16 の 1 面を
    // 共有すると見積もるが，ここでは FP32 の 2 面を別々に作るので出力側は見積りの 4 倍になる．main は実際に作る
    // CB の合計（mcast_gemm_l1_bytes）が予算に収まる構成だけを受け付けて選ぶ（choose_cfg の accept．main の fits_l1）ので，
    // ここは呼び出し側の誤りを CB の確保より前に止める安全網である．
    mcast_check_l1(g, spec, "gemm_fp32out");
    std::vector<float> Ctiled;
    const double timed = run_mcast_gemm(md, g, spec, Mt, Nt, Kt, A_tilized, B_tilized, Ctiled, warmup, iters).timed_s;
    Cf = untilize_nfaces<float>(Ctiled, M, N);  // タイル順（4 面）から行優先へ戻す
    return timed;
}

// 成分への分割から部分積の足し込みまで．T は bfloat16 か float（TF32）．pg は出力のパネル分割と構成（main の構成探索）．
// 採った部分積の本数を返し，Cacc（行優先 M*N，FP32）と tot（GEMM の合計秒数）を書く．
template <typename T>
static size_t run_split(std::shared_ptr<distributed::MeshDevice>& md, const PanelGrid& pg, tt::tt_metal::MathFidelity fid,
                        const std::vector<double>& Ad, const std::vector<double>& Bd,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t terms,
                        uint32_t warmup, uint32_t iters, std::vector<float>& Cacc, double& tot) {
    const uint32_t Kt = K / 32, Nt = N / 32;
    const OptConfig& g = pg.g;
    const uint32_t nbr = pg.nbr, nbc = pg.nbc, pMt = pg.pMt, pNt = pg.pNt, Mp = pg.Mp(), Np = pg.Np();
    // ---- ホスト側で 3 成分へ分割する ----
    std::vector<T> Ah(Ad.size()), Am(Ad.size()), Al(Ad.size());
    std::vector<T> Bh(Bd.size()), Bm(Bd.size()), Bl(Bd.size());
    for (size_t i = 0; i < Ad.size(); ++i) split3((float)Ad[i], Ah[i], Am[i], Al[i]);
    for (size_t i = 0; i < Bd.size(); ++i) split3((float)Bd[i], Bh[i], Bm[i], Bl[i]);

    // 6 つの成分行列をそれぞれタイル順に並べ替える（GEMM ごとに使い回す）
    std::vector<T> AhT = tilize_nfaces(Ah, M, K), AmT = tilize_nfaces(Am, M, K), AlT = tilize_nfaces(Al, M, K);
    std::vector<T> BhT = tilize_nfaces(Bh, K, N), BmT = tilize_nfaces(Bm, K, N), BlT = tilize_nfaces(Bl, K, N);

    // 部分積の一覧．小さいレベルから並べる（レベル 2 → 1 → 0）ことで，
    // 足し込みの丸めを抑える
    struct Term { const std::vector<T>* a; const std::vector<T>* b; const char* name; int lvl; };
    std::vector<Term> all = {
        {&AlT, &BhT, "Al*Bh", 2}, {&AmT, &BmT, "Am*Bm", 2}, {&AhT, &BlT, "Ah*Bl", 2},
        {&AmT, &BhT, "Am*Bh", 1}, {&AhT, &BmT, "Ah*Bm", 1},
        {&AhT, &BhT, "Ah*Bh", 0},
    };
    // 末尾から terms 個を採る（terms=1 なら Ah*Bh のみ，3 ならレベル 1 まで，6 なら全部）
    std::vector<Term> use(all.end() - std::min<size_t>(terms, all.size()), all.end());

    // 採用した部分積を小さいレベルから順に実行し，FP32 で足し込んでいく．
    // オペランドはタイル化済みの全体から出力パネルぶんを切り出して渡す
    // （panel_rows / panel_cols，common/ozaki_gemm_host.hpp）．
    Cacc.assign((size_t)M * N, 0.0f);
    std::vector<float> Cp;
    tot = 0.0;
    for (auto& t : use) {
        for (uint32_t br = 0; br < nbr; ++br)
            for (uint32_t bc = 0; bc < nbc; ++bc) {
                auto Ap = panel_rows(*t.a, br, pMt, Kt); auto Bp = panel_cols(*t.b, bc, pNt, Kt, Nt);
                tot += gemm_fp32out<T>(md, g, fid, Ap, Bp, Mp, Np, K, Cp, warmup, iters);
                // Mp×Np のパネルを全体の Cacc へ書き戻す（FP32 で累積）
                for (uint32_t r = 0; r < Mp; ++r)
                    for (uint32_t c = 0; c < Np; ++c)
                        Cacc[(size_t)(br * Mp + r) * N + (bc * Np + c)] += Cp[(size_t)r * Np + c];
            }
        // 表示は最初の 1 本（最も小さい項）と主項（レベル 0）だけにする（ログを短く保つため．採った本数は戻り値）
        if (&t == &use.front() || t.lvl == 0)
            fmt::print("  + {} (level {})\n", t.name, t.lvl);
    }
    return use.size();
}

// ---- 入口（デバイスを開く前．並びは尾崎の 2 つのドライバと同じ．common/oz_driver.hpp の冒頭の 1〜4）----
// 位置引数（M N K warmup iters terms．バッチは扱わないので B は取らない）と MathFidelity
struct ExtArgs {
    uint32_t M = 0, N = 0, K = 0, warmup = 0, iters = 0, terms = 6;
    bool verify = true;                // 参照解を求めて精度を評価するか（OZ_NOVERIFY でなければ真）
    tt::tt_metal::MathFidelity fid = tt::tt_metal::MathFidelity::HiFi4;
};
// 2〜4．不正なら理由を出して終了コード 2 で止める
static ExtArgs ext_args(int argc, char** argv) {
    ExtArgs a;
    // 2. 位置引数．余分な位置引数は黙って読み捨てない
    if (!oz_env::check_argc(argc, 6, "M N K warmup iters terms")) std::exit(2);
    const oz_drv::Dims d = oz_drv::read_dims(argc, argv, {1280, 1408, 1024}, "gemm_bf16ext");
    a.M = d.M; a.N = d.N; a.K = d.K;
    oz_drv::read_counts(argc, argv, 4, 1, 3, a.warmup, a.iters, 0u);
    a.terms = oz_env::arg_u32(argc, argv, 6, 6, 0, UINT32_MAX, "terms");
    a.verify = oz_eval::verify_enabled();
    // 3. つまみの値（OZ_EXT_FMT・OZ_EXT_SIG・OZ_FIDELITY・OZ_L1_BUDGET_KB など）・入力の生成と判定と書き出し・GEMM 段・計時．
    //    K との関係（OZ_EXPK < K）もデバイスを開く前に見る
    oz_drv::check_env(oz_knobs::W_EXT, a.K, a.verify);
    a.fid = baseline_fidelity();   // 値は上の preflight が検査済み
    // 4. このドライバに固有の検査．OZ_EXT_SIG は TF32 の成分の幅なので，成分が BF16（OZ_EXT_FMT が bf16 か未設定）のときに
    //    立っていれば止める（黙って無視すると，幅を変えたつもりで BF16 の成分を測る．CONFIG には ext_fmt=bf16・ext_sig=8 が
    //    残るだけ）
    if (a.terms != 1 && a.terms != 3 && a.terms != 6) {
        fmt::print(stderr, "terms は 1, 3, 6 のいずれか（レベルの切れ目でしか切れない）\n");
        std::exit(2);
    }
    if (!ext_tf32() && oz_knobs::set(OZK("OZ_EXT_SIG"))) {
        fmt::print(stderr, "ERROR: OZ_EXT_SIG={} は OZ_EXT_FMT=tf32 のときだけ効く（成分が BF16 の構成では止める）\n",
                   oz_knobs::get(OZK("OZ_EXT_SIG")));
        std::exit(2);
    }
    return a;
}

// オペランドをデバイスを開く前に作り（共通の host_eval．尾崎の各ドライバと同一），FP32 へ丸めた値が有限か（FP32 の範囲の内側か）を
// 見る．範囲の外（OZ_CONST=1e100 など）は FP32 で無限大になり，成分への分割も精度の評価も意味を持たない．検証を省く実行
// （OZ_NOVERIFY）では精度を見ないので成功（Test Skipped・0）で終わってしまう．検証の有無によらず，ここで止める（gemm_bf16 と同じ．
// 参照解はデバイスを開いた後に求める）
static void ext_operands(const ExtArgs& a, std::vector<double>& Ad, std::vector<double>& Bd) {
    oz_eval::gen_operands(Ad, Bd, a.M, a.K, a.N);
    size_t bad = 0;
    double worst = 0.0;
    for (const auto* v : {&Ad, &Bd})
        for (double x : *v)
            if (!std::isfinite((float)x)) { ++bad; worst = std::max(worst, std::fabs(x)); }
    if (bad) {
        fmt::print(stderr, "ERROR: 入力の {} 要素が FP32 で有限に表せない（最大の絶対値 {:.3e}．FP32 の最大は約 3.4e38）\n",
                   bad, worst);
        std::exit(2);
    }
}

// ---- 構成探索（デバイスを開いた後．見出しの行もここで出す）----
struct ExtPlan {
    PanelGrid pg;                      // 出力パネルの分割と構成
    double peak_chip = 0.0;            // 理論ピーク［TFLOP/s］（common/oz_fidelity.hpp の oz_peak_tops）
    uint32_t l1_budget = 0;            // 形状探索の L1 の予算［B］（OZ_L1_BUDGET_KB．CONFIG の l1_budget_kb）
};
// 構成が取れなければ理由を表示して pg.ok() が偽
static ExtPlan ext_plan(std::shared_ptr<distributed::MeshDevice>& md, const ExtArgs& a) {
    ExtPlan pl;
    const auto gd = md->compute_with_storage_grid_size();
    const double clk = (double)md->get_clock_rate_mhz() * 1e6;
    // LoFi 1 パスが 4096 演算/サイクル/コアで，パス数に反比例する（既定 HiFi4 なら 1024．oz_fidelity.hpp）
    pl.peak_chip = oz_peak_tops(a.fid, clk, (uint32_t)(gd.x * gd.y));
    const uint32_t Mt = a.M / 32, Nt = a.N / 32, Kt = a.K / 32;
    // 出力パネル分割．パッカ L1 累積の経路はコアあたりの出力ブロックを FP32 で 2 面（c_16 の最終出力と
    // c_24 のアキュムレータ）常駐させるので，コアあたりの L1 は出力の 2*(pm*pn)*4 KiB に入力 CB（A・B の
    // ブロックの 2 重緩衝．bw と入力の形式で決まる）を足した量になる．収まるかは固定の N の境界では決めず，
    // 候補の構成ごとに実際に作る CB の合計（mcast_gemm_l1_bytes）を L1 の予算（OZ_L1_BUDGET_KB．既定 1408 KiB）と
    // 比べて決める（下の fits_l1）．
    // パネルの辺は Mt・Nt の CAP 以下の約数から，構成が取れて実際の CB が予算に収まる組を面積の大きい順に探し
    // （common/mcast_gemm.hpp の choose_panels．先頭の候補は「CAP 以下の最大の約数」の組なので，その組で構成が
    // 取れる形ではそれを選ぶ），出力をパネルに分けて順に回す．
    // CAP=80 はパネルの辺の固定の上限（方針）で，容量から導いた値ではない．Mt・Nt が 80 を超えれば，1 枚で予算に
    // 収まる形でもパネルに分ける（3520 は Mt=110 の 80 以下の最大の約数 55 で 2x2 枚）．パネルに分けるので，
    // 行列全体を 1 枚にするとコアあたりの出力が予算を超える大きさ（14080 など）でも実行でき，測りたい到達精度が
    // L1 の容量で測れる大きさに縛られない．
    const uint32_t CAP = 80;
    // TF32 成分は FP32 で格納するので入力タイルが 4 KB になる．L1 の見積りに実サイズを
    // 渡す（出力は c_16 と c_24 の 2 面ぶん）．BF16 は in_tile を渡さない判定（c_16 と c_24 が BF16 の 1 面を
    // 共有すると見積もる）で選び，実際に作る CB（FP32 の 2 面）の合計が予算に収まる bw だけを受け付ける
    // （accept）．その判定で選ばれた構成が実際にも収まる形（測定済みの構成）では，accept は
    // 選ばれる構成を変えない：他の候補の bw を下げることはあっても，勝った候補はそのまま通るため．
    // accept が無いと，収まらない構成を選んだまま gemm_fp32out の TT_FATAL で止まる．
    const tt::DataFormat in_fmt_sel = ext_tf32() ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    pl.l1_budget = oz_l1_budget();
    const uint64_t budget = pl.l1_budget;
    auto fits_l1 = [&](const OptConfig& c) {
        return mcast_gemm_l1_bytes(c, l1acc_gemm_spec(c, Kt, in_fmt_sel, a.fid)) <= budget; };
    // TF32 の見積り：入力 1 枚と，c_16・c_24 の FP32 の 2 面をまとめて 1 面とみなした出力 1 枚のバイト数
    const uint32_t f32_tile = tt::tile_size(tt::DataFormat::Float32);
    pl.pg = panel_grid(choose_panels(Mt, Nt, CAP, [&](uint32_t pm_, uint32_t pn_) {
        return ext_tf32()
            ? choose_cfg(pm_, pn_, Kt, gd.y, gd.x, 4u, 0u, f32_tile, 2u * f32_tile, (uint32_t)budget, fits_l1)
            : choose_cfg(pm_, pn_, Kt, gd.y, gd.x, 4u, 0u, 0u, 0u, (uint32_t)budget, fits_l1);
    }), Mt, Nt);
    const OptConfig& g = pl.pg.g;

    fmt::print("============== {}-extended matmul ({} split) ==============\n",
               ext_tf32() ? "TF32" : "BF16", ext_tf32() ? "tf32x2/x3" : "bf16x2/x3");
    fmt::print("Device      : grid={}x{} ({} cores) {} MHz  chip_peak(bf16/{})={:.1f} TFLOP/s\n",
               gd.x, gd.y, gd.x * gd.y, md->get_clock_rate_mhz(), oz_fidelity_name(a.fid), pl.peak_chip);
    fmt::print("Problem     : M={} N={} K={}  terms={} (1=single,3=lvl0-1,6=lvl0-2)  fmt={}\n",
               a.M, a.N, a.K, a.terms, ext_tf32() ? fmt::format("TF32 (FP32 storage, sig={})", ext_sig()) : "BF16");
    if (!pl.pg.ok()) {
        // 条件は src/README.md「対応する形状」（Mt・Nt がそれぞれコア格子に収まる 2 以上の約数を持ち，
        // その分割でブロックが L1 に収まること．パネルの辺はその約数から選べる）
        fmt::print(stderr, "ERROR: no valid cfg for any output panel of Mt={} Nt={} tiles (sides <= {}), Kt={} "
                           "(Mt and Nt each need a divisor >= 2 that fits the {}x{} core grid, with the per-core "
                           "block fitting in L1)\n", Mt, Nt, CAP, Kt, gd.y, gd.x);
        return pl;
    }
    fmt::print("Cfg         : {}x{}={} cores per_core={}x{} subblock={}x{} in0_block_w={} num_blocks={} (fp32_dest_acc, out=FP32)\n",
               g.cr, g.cc, g.cr * g.cc, g.pm, g.pn, g.sh, g.sw, g.bw, g.nb);
    if (pl.pg.nbr > 1 || pl.pg.nbc > 1)
        fmt::print("Panels      : {}x{} output panels of {}x{} tiles (per-core L1 bounded to {} out-tiles)\n",
                   pl.pg.nbr, pl.pg.nbc, pl.pg.pMt, pl.pg.pNt, g.pm * g.pn);
    // 「Kernel      : common/kernels/bmm_l1acc.cpp (packer L1 accumulate, num_blocks=<n>, accumulating)」は
    // scripts/rules/probe_checks.py の kblock_acc（論文の「読み込みブロック間の L1 累積」の照合）が正規表現で読む（カーネルの
    // パスは移す前の kernels/bmm_l1acc.cpp と書いた 042dc00 のログも読めるようにしてある．num_blocks は CONFIG の nb= にもある）．
    // 表示を整えるときは照合の正規表現と一緒に直す
    fmt::print("Kernel      : common/kernels/bmm_l1acc.cpp (packer L1 accumulate, "
               "num_blocks={}{})\n", g.nb, g.nb > 1 ? ", accumulating" : ", single block");
    return pl;
}

// ---- 実行（成分への分割・部分積の行列積と FP32 での足し込み）----
struct ExtOut {
    std::vector<float> Cacc;           // 行優先 M*N の FP32 の和
    double tot = 0.0;                  // 部分積の行列積のデバイス実行時間の和［s/iter］
    size_t nterms = 0;                 // 採った部分積の本数
    std::vector<double> Chi, Clo;      // 参照解（検証するときだけ）
};
static ExtOut ext_run(std::shared_ptr<distributed::MeshDevice>& md, const ExtArgs& a, const ExtPlan& pl,
                      const std::vector<double>& Ad, const std::vector<double>& Bd) {
    ExtOut o;
    if (a.verify) oz_eval::reference_gemm(Ad, Bd, a.M, a.K, a.N, o.Chi, o.Clo);
    if (ext_tf32())
        o.nterms = run_split<float>(md, pl.pg, a.fid, Ad, Bd, a.M, a.N, a.K, a.terms, a.warmup, a.iters, o.Cacc, o.tot);
    else
        o.nterms = run_split<bfloat16>(md, pl.pg, a.fid, Ad, Bd, a.M, a.N, a.K, a.terms, a.warmup, a.iters, o.Cacc, o.tot);
    return o;
}

// ---- 報告 ----
// 精度・時間・RESULT・テレメトリ・CONFIG の各行を出す．出力と精度の指標が有限で，書き出しが通ったかを返す
static bool ext_report(const ExtArgs& a, const ExtPlan& pl, const ExtOut& o,
                       const std::vector<double>& Ad, const std::vector<double>& Bd) {
    const uint32_t M = a.M, N = a.N, K = a.K;
    const OptConfig& g = pl.pg.g;
    // ---- 共通の参照解に対する精度 ----
    // OZ_NOVERIFY のときは評価せず，RESULT の精度欄を nan にする
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    oz_eval::Acc acc{qnan, qnan, qnan};
    bool dump_ok = true;
    if (a.verify) {
        std::vector<double> Cd(o.Cacc.begin(), o.Cacc.end());
        acc = oz_eval::accuracy(Cd, o.Chi, o.Clo);
        // OZ_DUMP_C（調査用．尾崎の各ドライバと同じ形式）．書けなければ失敗とする（デバイスを閉じてから 1 で終わる）
        dump_ok = oz_eval::dump_result(Cd, o.Chi, o.Clo, Ad, Bd, M, K, N);
    }
    const double rel_fro = acc.rel_fro, mre = acc.maxrel;
    // 有限性の判定．到達精度に合否の基準は無い（精度そのものが測定の対象）が，出力や精度の指標が有限でなければ
    // 壊れた実行なので失敗にする（入力が FP32 で有限なことは ext_operands がデバイスを開く前に確かめてある）．
    size_t nonfinite = 0;
    for (float v : o.Cacc) nonfinite += std::isfinite(v) ? 0u : 1u;
    const bool acc_finite = !a.verify || (std::isfinite(rel_fro) && std::isfinite(mre));
    const bool finite_ok = nonfinite == 0 && acc_finite;
    if (!finite_ok)
        fmt::print(stderr, "ERROR: 有限な入力に対して出力に非有限値が {} 個{}\n", nonfinite,
                   acc_finite ? "" : "，精度の指標が有限でない");

    // 有効演算量は 2MNK（部分積の本数に依らない）．したがって実効スループットは
    // 部分積の本数ぶんだけ素の BF16 より低くなる
    const double flops = 2.0 * (double)M * N * K;
    const double eff_tflops = flops / (o.tot) / 1e12;
    fmt::print("-----------------------------------------------------------------\n");
    fmt::print("Partials    : {} GEMM(s), total compute {:.4f} ms/iter\n", o.nterms, o.tot * 1e3);
    fmt::print("Throughput  : {:.2f} TFLOP/s (useful fp32-equiv)  = {:.1f}% of bf16 chip peak\n",
               eff_tflops, 100.0 * eff_tflops / pl.peak_chip);
    if (a.verify) oz_eval::print_accuracy(acc);
    else          fmt::print("Accuracy : skipped (OZ_NOVERIFY) -- 精度は評価していない\n");
    // OZ_EXPDIFF（各内積が 1 と 2^-d の 2 項だけの入力）のときは，出力から 1 を引いた値の最小・最大も出す．
    // 相対誤差だけでは「小さい項が消えた（C-1=0）」と「2^-(d-1) へ切り上がった」を区別できない（どちらも
    // |誤差| = 2^-d）ので，論文の「d=11 で 2^-10 へ丸められ，d>=12 で完全に失われる」はこの値で読む
    const oz_eval::InputEnv ie = oz_eval::input_env();
    if (ie.expdiff) {
        double dmin = INFINITY, dmax = -INFINITY;
        for (float v : o.Cacc) { const double dv = (double)v - 1.0; dmin = std::min(dmin, dv); dmax = std::max(dmax, dv); }
        fmt::print("EXPDIFF-OUT,d={},k2={},K={},nb={},c_minus_1_min={:.9e},c_minus_1_max={:.9e},small_term={:.9e}\n",
                   ie.d, ie.k2, K, g.nb, dmin, dmax, std::exp2(-ie.d));
    }
    // RESULT 行は照合が欄の位置で読む（scripts/lib/oz_logs.py の RESULT_SCHEMA）．欄を足すなら CONFIG に足す
    fmt::print("RESULT,{},{},{},{},{},{:.6e},{:.4f},{:.4f},{:.3e},{:.3e},{:.1f}\n",
               M, N, K, (uint32_t)o.nterms, g.cr * g.cc, flops, o.tot * 1e3,
               eff_tflops, rel_fro, mre, acc.ebits);
    // 設定の記録（尾崎の CONFIG 行と同じく OzLine で key=value を名前で足す．検査側は位置ではなく名前で読む）．
    // ext_sig は成分の有効仮数幅（暗黙の 1 を含む．BF16 成分は 8）．到達精度に合否の基準は
    // 無いので VERIFY 行は出さない（verified は精度を測ったかどうか）．
    // timing_scope=gemm_only：RESULT の時間（total compute）は部分積の行列積のデバイス実行時間の和で，
    // 部分積・パネルごとに別々にウォームアップして測ったもの．ホストでの成分への分割・タイル化・
    // パネルの切り出し・部分積の FP32 での足し込みは含まない．尾崎の各経路の DRAM->DRAM（分割・
    // 再構成までデバイスの段の和）と同じ計時の範囲ではないので，全体性能として並べて比べないこと．
    // input=・seed= は入力の作り方（oz_eval::input_desc．尾崎の CONFIG と同じ欄．合否の基準は無いので
    // min_bits 欄は付けない），timing= からは計時の方式（timing_desc）．
    // 末尾の knobs は既定から外れたアブレーションのつまみ（つまみの表の R_KNOBS のうちこのドライバが読むもの．尾崎の
    // CONFIG と同じ書き方），l1_budget_kb は形状探索の L1 の予算［KiB］，gen は入力の生成の版（後の 2 つは
    // oz_driver.hpp の add_common_tail と同じ欄）
    std::fflush(stdout);
    oz_eval::print_telem();   // 計時区間のクロックなど（oz_timing.hpp）．計時の方式は CONFIG の timing=
    OzLine cfg("CONFIG,gemm_bf16ext");
    cfg.kv("fp32_dest_acc", 1).kv("fidelity", oz_fidelity_name(a.fid)).kv("bw", g.bw).kv("nb", g.nb)
       .kv("panels", oz_drv::axb(pl.pg.nbr, pl.pg.nbc)).flag("verified", a.verify)
       .kv("ext_fmt", ext_tf32() ? "tf32" : "bf16").kv("ext_sig", ext_tf32() ? ext_sig() : 8).kv("terms", (uint32_t)o.nterms)
       .kv("grid", oz_drv::axb(g.cr, g.cc)).kv("shape", oz_drv::axb(g.pm, g.pn)).kv("sub", oz_drv::axb(g.sh, g.sw))
       .kv("gemm_split", 1).kv("timing_scope", "gemm_only").raw(oz_eval::input_desc(oz_eval::kDefaultSeed, false))
       .raw(oz_eval::timing_desc())
       .kv("knobs", oz_knobs::record(oz_knobs::W_EXT, oz_knobs::R_KNOBS)).kv("l1_budget_kb", pl.l1_budget / 1024u)
       .kv("gen", oz_eval::kGenVersion);
    cfg.print();
    return finite_ok && dump_ok;
}

int main(int argc, char** argv) {
    // 入口の並びは尾崎の 2 つのドライバと同じ（common/oz_driver.hpp の冒頭）．
    // 1. 環境だけ．到達精度に合否の基準を持たない（精度そのものが測定の対象．冒頭の「終了コード」）ので，OZ_MIN_BITS は効かない
    oz_drv::reject_env_gemm(oz_knobs::W_EXT, "gemm_bf16ext", "到達精度に合否の基準を持たない（精度そのものが測定の対象）");
    const ExtArgs a = ext_args(argc, argv);
    std::vector<double> Ad, Bd;
    ext_operands(a, Ad, Bd);

    bool pass = true;
    try {
        auto md = distributed::MeshDevice::create_unit_mesh(0);
        const ExtPlan pl = ext_plan(md, a);
        if (!pl.pg.ok()) {
            md->close();
            return 2;
        }
        const ExtOut o = ext_run(md, a, pl, Ad, Bd);
        pass &= ext_report(a, pl, o, Ad, Bd);
        pass &= md->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed: {}\n", e.what());
        throw;
    }
    // 到達精度に合否の基準は無いので "Test Passed" とは書かない．成功（終了コード 0）が意味するのは，
    // 構成が取れて最後まで走り，出力と精度の指標が有限だったことだけである．
    // 精度を評価しなかった実行（OZ_NOVERIFY）は，ほかのドライバと同じく "Test Skipped" とする（終了コードは 0）
    std::fflush(stdout);
    return oz_drv::finish(pass, a.verify, "Completed (no accuracy threshold; outputs finite)");
}
