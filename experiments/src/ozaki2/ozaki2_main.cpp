// SPDX-License-Identifier: Apache-2.0
//
// Blackhole 上の尾崎スキーム II（CRT 方式）の一気通貫パイプライン
// ---------------------------------------------------------------------------
// FP64 のオペランド A,B から FP64 相当の精度の C = A*B を得る．論文の ssec:oz2（尾崎スキーム II）と
// ssec:nofp64（FP32 を用いた再構成）．
//   第 0 段 SCALE : 行ごと（A）／列ごと（B）の最大絶対値 -> 2 のべきの正規化係数
//   第 1 段 SLICE : A -> {A_i}, B -> {B_i}（S 個の剰余行列）．正規化・整数化
//                   A' ≈ x/2^e * 2^b（|A' - X| <= 1 で最近接とは限らない．ozaki2/kernels/oz2_digits.h）・2^20 進 3〜5 桁への
//                   分解もここに含む
//   第 2 段 GEMM  : C_i = A_i ・ B_i．INT8 経路は法ごとに 1 本（S 本）の独立な INT8→INT32 行列積，
//                   桁分割の 7 経路は法ごとに桁積 3〜4 本（1 桁の法は 1 本，平衡 n 桁は n^2 本まで．計 nprod 本）の
//                   FP32 累算の行列積．位の同じ桁積は K 方向に連結して 1 本の GEMM にまとめる
//                   （無誤差条件の許す範囲で．出力は計 nout 本．ozaki2_split.h の oz2_product_groups）
//   第 2.5 段     : 桁分割の 7 経路だけ．桁積を法へ畳んで剰余にする（ozaki2/kernels/oz2_compose.cpp）．
//                   1 桁の法（桁 32 の AWE 経路）は通らず，GEMM の出力をそのまま剰余の平面へ書き，再構成が法へ畳む
//   第 3 段 RECON : {C_i} -> C = frac*KM（中国剰余定理を分数形で，triple-float）
// 全段をデバイス上で実行する．ホストが行うのは FP64 のオペランドをタイル順に並べて
// 送ることと，FP64 のまま戻る結果を並べ戻すことだけで，正規化スケールの算出も
// FP64 と triple-float の相互変換も掛け戻しもデバイス側にある．これは scheme I と
// 同じ分担である（論文の sec:cond（測定条件））．ただし入力からパラメータを決める構成（OZ2_FP64 の b・窓・パス数，
// OZ2_PASSES=auto のパス数）では，デバイスを開く前にホストがオペランドを走査して統計を測る
// （ozaki2_plan.hpp の choose_moduli_and_width．段の計時の外）．
// 参照実装 GEMMul8 との対応は docs/gemmul8_parity.md にある．
//
// 法（ビット幅と本数 s）は K から ozaki2_choose_moduli_w（ozaki2_moduli.h）が自動で選ぶ．条件は
// |C_i| = |Σ_k A_i B_i| が再構成側で無誤差に読める範囲に収まり，かつ法の積が
// CRT に必要な窓を覆うこと．法を固定すると K に上限が付く（INT8 経路は GEMM の出力が符号-絶対値 int32
// なので K*floor(m/2)^2 <= 2^31-1 が要り，法 255 なら K <= 133144）が，K から選ぶので，大きい K では
// この条件を満たす小さい法を取る（INT8 は表の後ろへ 151 未満の互いに素な法を足す．ozaki2_moduli.h の表の延長）．
// ただし任意の K で動くわけではない：K が大きいと法 1 本の幅が痩せて窓 2b+1+log2 K を覆えなくなり，b が FP64 相当の
// 53 を下回る構成は choose_moduli_and_width がデバイスを開く前に止める（OZ2_B・OZ2_MODULI で強制した調査用の構成を除く）．
// 桁分割の経路は桁積の累算の条件（ozaki2_moduli.h の ok 判定）で止まる．
//
// 段の部品は ozaki2/ のヘッダにある（scheme I の ozaki1/ と同じ分け方）：形式ごとの値・桁分割の有無・MathFidelity は
// ozaki2_format.hpp，つまみ（Oz2Knobs・Oz2Diag）と計画（法と幅・再構成のパス数・合成段の平面・法の組分け・GEMM のパネル・
// 再構成のパス）は ozaki2_plan.hpp，OZ2_FP64 の誤差のモデルは ozaki2_autop.hpp，段の資源と構築は ozaki2_stages.hpp，
// OZ_TIME_WHOLE の記録と流し直しは ozaki2_whole.hpp，診断 OZ2_CHECK_CONCAT は ozaki2_concat_check.hpp．法の表と選び方は
// ozaki2_moduli.h（経路の表と法 1 本の運び方 ozaki2_split.h・段の時間のモデル ozaki2_awe.h を取り込む）．デバイスカーネルは
// ozaki2/kernels/ と両スキーム共通の common/kernels/ にある．このファイルは段を並べて投入・計時し，結果を表示する．
//
// 使い方: ozaki2_int8 [M] [N] [K] [num_moduli] [warmup] [iters]
//         既定値   1024 1024 1024 def_s 1 3（warmup・iters は 1 以上．scheme I と同じ読み方．common/oz_driver.hpp）
//         def_s（経路の表 ozaki2_split.h の OZ2_ROUTES）は形式ごと：INT8 16，FP8 14，FP32（TF32）11，BF16・FP16・BFP8_b・BFP4_b・BFP2_b 24
//         （24 の 5 形式は段の時間のモデルで法の集合を選び，num_moduli はその本数の上限．ozaki2_awe.h）
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/host_eval.hpp"
#include "common/oz_bits.h"             // f32_bits
#include "ozaki2/ozaki2_format.hpp"     // OZ2_PARAM・OZ2_DIGITK・OZ2_DIGIT・oz2_fidelity
#include "ozaki2/ozaki2_plan.hpp"       // Oz2Args・Oz2Knobs・Oz2Diag・Oz2Plan と計画の関数
#include "ozaki2/ozaki2_autop.hpp"      // OZ2_FP64 の誤差のモデル（OZ2_RECON1_*）
#include "ozaki2/ozaki2_stages.hpp"     // 段の資源と構築
#include "ozaki2/ozaki2_whole.hpp"      // OZ_TIME_WHOLE の記録と流し直し
// 行／列ごとの正規化スケールをデバイスで求める段（scale_build・expdelta_build・upload_operand_f64・untilize_f64）は scheme I と共通
#include "common/ozaki_scale.hpp"
// FP64・128 bit 固定小数のタイルの大きさ
#include "common/kernels/f64_tf.h"
// 入口の検査の並び・CONFIG 行の共通の欄・最後の判定（scheme I と共通）と，OZ_TIME_WHOLE の検査（WHOLE-CHECK）
#include "common/oz_driver.hpp"
#include "common/oz_whole.hpp"
#ifdef OZ2_DIGITK
// 診断 OZ2_CHECK_CONCAT（連結した GEMM の出力・剰余・CRT 復元を独立な整数の参照と比べる．桁分割の経路だけ）
#include "ozaki2/ozaki2_concat_check.hpp"
#endif

namespace ttm = tt::tt_metal;
namespace dist = tt::tt_metal::distributed;

// ---------------- 実行の状態 ----------------
// main は下の段の関数を順に呼ぶだけにしてある．出力が帯 1 本・法が 1 組（14080^3 より小さい大きさの既定）のとき，デバイス上の
// 確保・プログラムの構築・転送・投入・同期・計時・解放は，呼ぶ順に 1 本の流れとして並ぶ
// （出力を行の帯に分ける経路では，main が帯ごとに「A のスライス（slice_band）→ 第 2 段 → 第 3 段」を回し，器は初めの帯で
// 取って全帯で使い回す．法を組に分ける経路（OZ2_MOD_GROUPS）では，組ごとのスライスの確保と解放が第 2 段の中に入る．slice_group）．
// 14080^3 では DRAM がほぼ埋まるので，reset() の位置が 1 つずれるだけで確保に失敗する．
// 確保の順序も配置を通して性能を左右する．そのためバッファはすべて Oz2Run が持ち，
// 手放す位置は各段の関数の中の reset() でそのまま示す．
//
// 計画（法と幅・組分け・パネル・合成段の平面）は Oz2Plan（ozaki2_plan.hpp）から受け継ぐ．つまみ（kn・dg）は main の冒頭で
// 1 度だけ読んだ値で，段の関数はここから引く．
// メンバの宣言順は 1 組のときの確保順に揃えてある．構造体はメンバの逆順に破棄されるので，
// md->close() のあとまで残ったバッファは確保の逆順に解放される．
// MeshWorkload を含む型（ScaleCtx・ExpDeltaCtx・Oz2Slice・Oz2Compose）は std::optional で
// 持ち，構築する段で emplace する．MeshWorkload は既定構築でも MetalContext を引き，
// ワークロードの通し番号を 1 つ消費するので，デバイスを開く前に作ってはいけない．
struct Oz2Run : Oz2Plan {
    Oz2Knobs kn;
    Oz2Diag dg;
    // デバイスと参照解（open_device・compute_reference）
    std::shared_ptr<dist::MeshDevice> md;
    ttm::CoreCoord gd;
    bool verify = false;            // 検証が有効か（OZ_NOVERIFY が立っていない．common/oz_reference.hpp の verify_enabled）
    // 出口の FP64 o64 を最終パスが書くか（診断用 OZ2_RECON_NOF64 なら書かない．Oz2Diag::nof64 の否定）
    bool out_written = true;
    // 精度を求めて判定するか．検証が有効で，出口の FP64 が書かれるときだけ（RESULT・VERIFY・CONFIG の verified と最後の行）
    bool checked() const { return verify && out_written; }
    std::vector<double> Chi, Clo;
    // オペランドと第 0 段の中間（upload_operands）
    std::shared_ptr<dist::MeshBuffer> rAf64, rBf64, maxAb, maxBb, dltAb, dltBb, flgA, flgB;
    // 第 0・1 段のワークロード（scale_and_slice）．剰余行列は slA->out・slB->out
    std::optional<ScaleCtx> scCA, scCB;
    std::optional<ExpDeltaCtx> edA, edB;
    std::optional<Oz2Slice> slA, slB;
    // 全帯・全組のスライサが使い回す剰余行列・桁行列の器．出力を行の帯に分けるとき A は帯 1 本ぶん（slice_band が初めの帯で
    // 取る），OZ_TIME_WHOLE で法を組に分けるとき A・B とも組の枚数の最大（slice_group が初めの組で取る．段の途中で手放さないので，
    // 組ごとに取ると全組ぶんが同時に載ってしまう）
    std::shared_ptr<dist::MeshBuffer> slAsh, slBsh;
    // 出力を行の帯に分けるとき（nband > 1）の今の帯（main が回す．帯 1 本なら 0）．帯の先頭のパネル行は br0 = band*bpr
    uint32_t band = 0;
    uint32_t br0() const { return band * bpr; }
    uint32_t aOff() const { return br0() * pMt * Kt; }   // 帯の A の FP64 の先頭タイル（slice_band）
    uint32_t cOff() const { return br0() * pMt * Nt; }   // 帯の C（o64）の先頭タイル（run_recon）
    bool last_band() const { return band + 1 == nband; }
    // 第 2 段（run_gemm_stage）
    bool want_maxci = false;
#ifdef OZ2_DIGITK
    std::shared_ptr<dist::MeshBuffer> Dbuf;   // 桁積の平面
#endif
    // 再構成の入力（法ごとの平面 S 枚）：桁分割経路は合成段が法へ畳んだ剰余（1 桁の法は GEMM の出力），INT8 経路は GEMM の
    // 出力 C_i（符号-絶対値 int32）．run_recon_stage がそのまま読む
    std::shared_ptr<dist::MeshBuffer> crt_in;
#ifdef OZ2_DIGITK
    std::optional<Oz2Compose> cmp;
    // 法ごとの出力の本数が混ざって c_0 の深さ（公倍数）が大きくなりすぎる構成（BFP2_b の平衡 3・4・5 桁が
    // 混ざると 5・7・9 本で公倍数 315）では，1 回の投入で流れる本数 jn ごとに深さ jn の倍数の合成段を別に組む
    // （compose_for）．1 回の投入の中では本数が一定なので，それで足りる．
    std::map<uint32_t, std::unique_ptr<Oz2Compose>> cmp_by_jn;
    bool cmp_split = false;
    std::optional<Oz2ConcatCheck> cc;   // 診断 OZ2_CHECK_CONCAT（立っていなければ空．ozaki2/ozaki2_concat_check.hpp）
#endif
    // 第 3 段（run_recon_stage）．limb 3 面と 128 bit 固定小数は帯 1 本ぶんで，初めの帯で取って全帯で使い回す（run_recon）
    std::shared_ptr<dist::MeshBuffer> o64;
    std::shared_ptr<dist::MeshBuffer> rlimb[3], ra128;
    // 計時［秒］と診断値．tcp は桁分割経路の合成段（INT8 経路は 0）．帯に分けるときは全帯の和．tr1・tr2 は 2 パス構成の
    // 再構成の第 1・第 2 パス（recon pass の行）
    double tsc = 0, tsl = 0, tg = 0, tcp = 0, tr = 0, tr1 = 0, tr2 = 0;
    // OZ_TIME_WHOLE：全工程を続けて流した 1 回あたりの時間［秒］と，段ごとの計時の後の C（比較用）
    double twhole = 0;
    std::vector<uint32_t> o64_first;
    int whole_same = -1;
    uint64_t maxCi = 0;
    // OZ2_MAXCI の桁積（桁分割経路）で最大値に入れなかった要素の数：有限でない・整数でない・2^24 を超える
    uint64_t maxci_nonfinite = 0, maxci_nonint = 0, maxci_over = 0;
    // 精度（検証したときだけ求める．検証を省いた実行の RESULT の精度欄は nan．print_result）
    oz_eval::Acc acc{NAN, NAN, NAN};
    bool dump_ok = true;            // OZ_DUMP_C の書き出し（read_back_and_verify．書けなければ終了コード 1）
    std::string chash = "na";       // 結果 C のハッシュ（CONFIG の chash．oz_accuracy.hpp の chash_hex．読み戻さない診断 OZ2_RECON_NOF64 は na）
    DramFree dram0;                 // 計画（plan_bands）のときの DRAM の空き（CONFIG の dram_free_gb・dram_largest_gb）
    // ホストの FP64 の A・B（OZ2_CHECK_CONCAT の独立な整数参照と，OZ_DUMP_C の書き出し（dump_result．A の行ごと・
    // B の列ごとの最大絶対値を書く）だけが使う．upload_operands）
    std::vector<double> Ahost, Bhost;
    // OZ_TIME_WHOLE（論文の全体の時間）：段ごとの計時で投入したワークロードを投入の順に記録する器（ozaki2/ozaki2_whole.hpp．立っていなければ空）．
    // 段の関数は記録の 1 行（whole->record・record_workload）と，バッファを手放す所の may_release だけを持つ．
    // 中身はデバイスを閉じる前に main が手放す（閉じた後の破棄で MeshBuffer・MeshWorkload を壊さないように）
    std::optional<Oz2Whole> whole;
    // 段の途中でバッファを手放してよいか．OZ_TIME_WHOLE は記録した全工程を流し直すので，流し直しが読む FP64 のオペランド・
    // スライス出力・桁積の平面などを手放さない（手放す所はどれもこれを見る）
    bool may_release() const { return !whole; }
};

// ---------------- 段の投入と計時（構築は ozaki2_stages.hpp）----------------

// 部分行列積を 1 本組んで（gemm_build_partial）投入し，1 反復あたりの平均秒数を返す．引数の意味は gemm_build_partial．
// 計時は全ドライバ共通の手順（温めは計時せず，iters 回の算術平均．oz_timing.hpp の time_dispatch）．
// 法・桁積・パネルごとの計時はどれも段「gemm」で，続けて流れるあいだは初めの 1 回だけが定常状態まで温める．
// 出力は Cbuf に載ったまま返す（後段がデバイス上で直接読む）．
static double run_partial_gemm(Oz2Run& r, std::shared_ptr<dist::MeshBuffer> Abuf, std::shared_ptr<dist::MeshBuffer> Bbuf,
                               uint32_t dA, uint32_t dB, uint32_t M, uint32_t K, std::shared_ptr<dist::MeshBuffer> Cbuf,
                               uint32_t Ntg, uint32_t dW, uint32_t KsA = 0) {
    auto& cq = r.md->mesh_command_queue();
    dist::MeshWorkload workload = gemm_build_partial(r.md, r.g, Abuf, Bbuf, dA, dB, M, K, Cbuf, Ntg, dW, KsA, r.kn, r.dg.gemm_probe);
    dist::EnqueueMeshWorkload(cq, workload, false); dist::Finish(cq);   // 初回はここで JIT ビルド
    const double sec_iter = oz_eval::time_dispatch("gemm", r.warmup, r.iters,
        [&] { dist::EnqueueMeshWorkload(cq, workload, false); },
        [&] { dist::Finish(cq); });
    if (r.whole) r.whole->record_workload(oz_whole::GEMM, cq, std::move(workload));   // 全工程の計時のために取っておく（投入の順に）
    return sec_iter;
}

#ifdef OZ2_DIGITK
// バッファのタイル [first, first+ntiles)（直前の GEMM が書いた桁積の平面か，1 桁の法の剰余の平面）の最大絶対値
// （診断用．OZ2_MAXCI）．
// 桁積は FP32 に累算されるので，無誤差でいられる上限 2^24 に対してどこまで
// 使っているかを実行時に表示する．
// 出力は 2^24 以下の整数のはずなので，そうでない要素は最大値に入れずに数える：有限でない値（NaN・Inf），
// 整数でない値，2^24 を超える値．整数への変換はこの 3 つを除いた値（2^24 以下の整数）だけに行う
// （Inf や 2^64 以上の値を整数へ変換すると未定義動作で，NaN は std::max で比べると黙って落ちる）．
// EnqueueReadMeshBuffer はバッファ全体を読む（渡した vector を全体の長さへ広げる）ので，
// 読む量は減らせない．見る範囲だけをその平面に絞る（ほかの平面には部分和や前の法の桁積，ほかの法の剰余が残っている）．
// 参照計算を行うときだけ呼ぶので，性能測定の経路には転送が一切残らない．
struct Oz2PlaneMax { uint64_t max = 0, nonfinite = 0, nonint = 0, over = 0; };
static Oz2PlaneMax plane_absmax(std::shared_ptr<dist::MeshDevice>& md,
                                std::shared_ptr<dist::MeshBuffer> buf, size_t first, size_t ntiles) {
    std::vector<float> raw;
    dist::EnqueueReadMeshBuffer(md->mesh_command_queue(), raw, buf, true);
    const size_t n = std::min(raw.size(), (first + ntiles) * 1024);
    Oz2PlaneMax mx;
    for (size_t i = first * 1024; i < n; i++) {
        const float a = std::fabs(raw[i]);
        if (!std::isfinite(a))      { ++mx.nonfinite; continue; }
        if (a != std::floor(a))     { ++mx.nonint; continue; }
        if (a > 16777216.0f)        { ++mx.over; continue; }
        mx.max = std::max<uint64_t>(mx.max, (uint64_t)a);
    }
    return mx;
}

// 法 s について合成段を 1 回発行する（実行時引数を詰めて投入する．compose_set_args）．戻り値は 1 反復あたりの秒数．
// 入力の平面 [in0, in0+jn) を合成して out_addr の平面 out_plane へ書く．
// 桁積の平面を全部同時に持てない構成（連結しない平衡 5 桁は法あたり 25 枚で 14080^3 なら
// 19.8 GB．既定では連結で 9 本以下になるので切らない）は，CH 本ずつに切って部分和を別々の平面へ落とし，最後にその部分和を
// 足し合わせる（unit）．合成は法について線形で，途中で ModRound を挟んでも剰余類は変わらない
// ので答えは一度に合成した場合と同じである．どの回も入力と出力が別の平面なので，
// 計時のために同じワークロードを何度投入しても結果が変わらない（冪等である）．
static double compose_run(Oz2Run& r, Oz2Compose& c, uint32_t s, uint32_t j0, uint32_t jn, bool unit, uint32_t in0,
                          uint32_t out_addr, uint32_t out_plane) {
    auto& cq = r.md->mesh_command_queue();
    // ライタの TensorAccessor は構築時の出力バッファから作ってある．入力バッファ（桁積）へ
    // 書かせるのは，両者の形が同じと構築時に確かめてあるときだけ（Oz2Compose の注記）．
    if (out_addr != c.out_addr && !(out_addr == c.in_addr && c.out_accepts_in)) {
        fprintf(stderr, "ERROR: 合成段の出力先が構築時のバッファと形の違うバッファを指している\n");
        std::exit(2);
    }
    compose_set_args(c, r.mo, s, j0, jn, unit, in0, out_addr, out_plane);
    if (!c.primed) {   // 初回だけ JIT を計時の外で済ませる
        dist::EnqueueMeshWorkload(cq, c.workload, false); dist::Finish(cq); c.primed = true; }
    // 合成は法ごとに GEMM と交互に流れる（段としては GEMM 段と別の合成段で，breakdown の compose の行．DRAM->DRAM・
    // RESULT の GEMM の欄は両方の和．論文は合成段を GEMM 段と分けて書く）ので，温めの群だけを GEMM と同じにする
    // （oz_timing.hpp の計時の 3．50 ms・20 倍の規則は直前の GEMM の計時からの間隔で当たる）
    const double sec = oz_eval::time_dispatch("compose", r.warmup, r.iters,
        [&] { dist::EnqueueMeshWorkload(cq, c.workload, false); },
        [&] { dist::Finish(cq); }, "gemm");
    // OZ_TIME_WHOLE のときは同じ詰め方で投入し直す手順を記録する（合成段は 1 つのワークロードを引数だけ差し替えて使い回すので，
    // 記録するのは「詰めて投入する」ことそのもの．引数の設定はホストの仕事で，全工程の計時の中にも入る）
    if (r.whole) {
        auto* q = &cq; Oz2Compose* cp = &c; const Ozaki2Moduli* mop = &r.mo;
        r.whole->record(oz_whole::COMPOSE, [q, cp, mop, s, j0, jn, unit, in0, out_addr, out_plane] {
            compose_set_args(*cp, *mop, s, j0, jn, unit, in0, out_addr, out_plane);
            dist::EnqueueMeshWorkload(*q, cp->workload, false); });
    }
    return sec;
}
#endif

// 再構成を 1 パス組んで（recon_build_pass）投入し，1 反復あたりの秒数を返す．
// 2 パス構成では逆数の断片列 pass.rp（1 法あたり NRP 個）を差し替えて 2 回呼ぶ：
//   パス 1 = 共通粒度へ丸めた R_i（累算が無誤差），パス 2 = 丸めの残り（両符号．ozaki2_moduli.h の rp2）．
// 2 つを混ぜて積むと 1 加算ごとに ULP(a2) が乗って無誤差性が消えるので，別々に積み，
// パス 2 が a128（パス 1 の 128 bit 固定小数）を読んで自分の 3 本を足し，出口で畳む．
static double run_recon_pass(Oz2Run& r, const ReconPass& pass, const ReconBufs& bf, uint32_t in_residue, const ReconScale& sc) {
    auto& cq = r.md->mesh_command_queue();
    dist::MeshWorkload workload = recon_build_pass(r.md, pass, bf, in_residue, r.bMt() * 32, r.N, r.mo, sc, r.kn, r.dg);
    dist::EnqueueMeshWorkload(cq, workload, false); dist::Finish(cq);
    const double sec = oz_eval::time_dispatch("recon", r.warmup, r.iters,
        [&] { dist::EnqueueMeshWorkload(cq, workload, false); },
        [&] { dist::Finish(cq); });
    // 全工程の計時のために取っておく（limb の面と 128 bit 固定小数の器は Oz2Run が持つ．run_recon）
    if (r.whole) r.whole->record_workload(oz_whole::RECON, cq, std::move(workload));
    return sec;
}

// 再構成を 1 パスか 2 パス（r.two_pass．make_recon_plan）回して FP64 の C をデバイス上で組み立てる．
// 合成には 90 bit を超える幅が要る（2 パス構成の frac は 2^-99 程度の絶対精度を持つ
// のに対し frac 自体は 2^-8 前後）ので，ライタは 128 bit の固定小数に集める．
// 出力 o64 はタイル順の FP64 で，ホストは並べ替えるだけでよい．出力を行の帯に分けるときは帯 r.band の bMt タイル行を
// 組み立て，o64 の帯の位置（cOff）へ書く（入力 inB・limb・a128 は帯 1 本ぶんの器）．
static void run_recon(Oz2Run& r, std::shared_ptr<dist::MeshBuffer> inB, uint32_t in_residue,
                      std::shared_ptr<dist::MeshBuffer> o64,
                      std::shared_ptr<dist::MeshBuffer> dltA,
                      std::shared_ptr<dist::MeshBuffer> dltB,
                      int32_t sh2, double& sec) {
    const uint32_t f32 = tt::tile_size(tt::DataFormat::Float32);
    const ReconScale sc = make_recon_scale(r.mo, sh2);

    // 計画 -> 確保 -> 実行の順に進む．確保量は計画から取るので，つまみが
    // 出力の形を変えても確保がそれに追随する（逆の順序で起きるずれは ReconPlan の注記）．
    const ReconPlan plan = make_recon_plan(r.mo, r.two_pass, r.ntotCb(), r.out_written);
    // frac の limb 3 面（診断用 OZ2_RECON_NOF64 の最終パスが書く）と，2 パス構成の第 1 パスが書き第 2 パスが読む
    // 128 bit 固定小数（16 KB ページ 1 面）．帯 1 本ぶんで，初めの帯で取って全帯で使い回す（同じキューなので，次の帯の
    // 第 1 パスは前の帯の最終パスが読み終えてから書く）
    if (!r.ra128) {
        for (auto& l : r.rlimb) l = mesh_dram_buffer(r.md, plan.limb_tiles, f32);
        r.ra128 = mesh_dram_buffer(r.md, plan.a128_tiles, OZ2_A128_PAGE);
        if (r.whole) {   // 全工程の計時で番兵で埋める（run_whole．器そのものは Oz2Run が持つ）
            for (auto& l : r.rlimb) r.whole->keep(l);
            r.whole->keep(r.ra128);
        }
    }
    ReconBufs bf;
    bf.in = inB; bf.o64 = o64; bf.dltA = dltA; bf.dltB = dltB;
    for (int i = 0; i < 3; i++) bf.limb[i] = r.rlimb[i];
    bf.a128 = r.ra128;
    bf.out_off = r.cOff();

    double s1 = 0, s2 = 0;
    for (size_t i = 0; i < plan.pass.size(); i++)
        (i ? s2 : s1) = run_recon_pass(r, plan.pass[i], bf, in_residue, sc);
    sec = s1 + s2;
    r.tr1 += s1; r.tr2 += s2;
    if (r.tr2 > 0 && r.last_band()) printf("  recon pass  : %.4f + %.4f ms（逆数の断片 %u 本 + %u 本）\n",
                                           r.tr1 * 1e3, r.tr2 * 1e3, (unsigned)OZ2_NRP1, (unsigned)OZ2_NRP2);
}

// ---------------- 入口と準備（ホストのみ）----------------

// コマンドラインの読み取り．引数の検査を main の先頭に置くと，計算の話と入り混じって読みにくい．
// ここは「受け取った文字列が使える形か」だけを見る（読み方は scheme I と共通の oz_driver.hpp．不正なら終了コード 2）．
static Oz2Args read_run_args(int argc, char** argv) {
    Oz2Args o;
    // 余分な位置引数も止める（引数の追加・位置ずれを黙って読み捨てない）
    if (!oz_env::check_argc(argc, 6, "M N K num_moduli warmup iters")) std::exit(2);
    // 引数は 10 進の整数として全体を読み切れたときだけ受け付ける（oz_env::arg_u32）．M,N,K は正で 2^32 未満の 32 の倍数
    const oz_drv::Dims d = oz_drv::read_dims(argc, argv, {o.M, o.N, o.K}, "ozaki2");
    o.M = d.M; o.N = d.N; o.K = d.K;
    // 法の本数は 1 以上．表の長さを超える指定は法選択が表を使い切ったところで止まる．
    o.s_req = oz_env::arg_u32(argc, argv, 4, OZ2_PARAM.def_s, 1u, 1024u, "num_moduli");
    // 全段で同じ計時回数（scheme I と同じ読み方）
    oz_drv::read_counts(argc, argv, 5, 1, 3, o.warmup, o.iters);
    return o;
}

// デバイスを開き，選んだ法を印字する（参照解は compute_reference）．
// 確保するのはデバイス（md）だけで，何も解放しない（md は main の終わりで閉じる）．
static void open_device(Oz2Run& r) {
    const Ozaki2Moduli& mo = r.mo;
    r.md = dist::MeshDevice::create_unit_mesh(0);
    r.gd = r.md->compute_with_storage_grid_size();
    printf("moduli: K=%u num_moduli=%u -> s=%u b=%u  m={", r.K, r.S_req, r.S, r.b);
    // 桁分割の方式を印で示す：* = 平方数，s = 疎な桁の平方数（FP8．積は * と同じ 3 本），k = Karatsuba，
    // w = 平衡桁 4 積，S = 1 桁（桁 32 の AWE 経路．積 1 本で合成段を通らない），b<n> = 平衡 n 桁（BFP2_b・BFP4_b．
    // 積 n^2 本か n(n+1)/2 本），Karatsuba は基数が 16 でなければ k<基数>（BFP4_b の k8），無印は INT8 経路．
    // scripts/verify/check_derivation.py はこの印から行列積の本数を数えるので，印を変えたら向こうも直すこと
    for (int i = 0; i < mo.s; i++) { char tag[16] = "";
        if (mo.q[i]) {
            if (mo.mode[i] == OZ2_SPLIT_SQ) snprintf(tag, sizeof tag, "*");
            else if (mo.mode[i] == OZ2_SPLIT_SQS) snprintf(tag, sizeof tag, "s");
            else if (mo.mode[i] == OZ2_SPLIT_KAR) { if (mo.q[i] == 16) snprintf(tag, sizeof tag, "k"); else snprintf(tag, sizeof tag, "k%d", mo.q[i]); }
            else if (mo.mode[i] == OZ2_SPLIT_BAL) snprintf(tag, sizeof tag, "b%d", oz2_bal_digits(mo.m[i], OZ2_PARAM.digit_bound, OZ2_PARAM.nd()));
            else if (mo.mode[i] == OZ2_SPLIT_ONE) snprintf(tag, sizeof tag, "S");
            else snprintf(tag, sizeof tag, "w");
        }
        printf("%d%s%s", mo.m[i], tag, i + 1 < mo.s ? "," : ""); }
    // gemms は桁積（MAC で数えた行列積）の本数，outputs は同じ位の桁積を K 方向に連結したあとの
    // GEMM の出力の本数（maxg 本まで連結する．ozaki2_split.h の oz2_product_groups）
    printf("}  KM=%.6f  gemms=%d  outputs=%d  maxg=%d", mo.KM, mo.nprod, mo.nout, mo.maxg);
    // 疎な桁の経路（FP8）と 1 桁の法を混ぜる経路は連結の上限が法ごとに違う（ozaki2_moduli.h の maxg_m）ので並べて出す
    if (std::any_of(mo.maxg_m.begin(), mo.maxg_m.end(), [&](int g) { return g != mo.maxg; })) {
        printf("  maxg_m={");
        for (int i = 0; i < mo.s; i++) printf("%d%s", mo.maxg_m[i], i + 1 < mo.s ? "," : "}");
    }
    printf("\n");
}
// 検証が有効なら参照解 Chi・Clo をホストで求める（GEMM 段の構成が取れることを確かめた後．取れない形で長い参照計算を
// 待たせない）．出口の FP64 を書かない診断（OZ2_RECON_NOF64）は精度を求めないので求めない．何も確保・解放しない
// （ホストの Chi・Clo だけ）
static void compute_reference(Oz2Run& r, const std::vector<double>& A, const std::vector<double>& B) {
    r.verify = oz_eval::verify_enabled();
    if (r.checked()) oz_eval::reference_gemm(A, B, r.M, r.K, r.N, r.Chi, r.Clo);
}

// ---- オペランドをデバイスへ載せる（計時区間の外）----
// 確保（この順）：rAf64・rBf64（FP64 のオペランド），maxAb・maxBb（行／列ごとの最大値），
// dltAb・dltBb（指数差分），flgA・flgB（範囲外検出の旗）．旗をゼロ化してから A・B を送る．
// 解放：ホスト側の A・B（デバイス側は何も手放さない）．
// 正規化前の値を FP64 のまま（1 要素 8 バイト，タイル順）送る．ホストがするのは
// タイル順への並べ替えだけで，FP64 -> triple-float の変換も，スケール算出用の |x| の
// BF16 化も，デバイス側のリーダが計時区間の内側で行う（scale_f64_reader.cpp は
// bf_ceil を立てるので，リダクションの結果が真の最大値を下回らない）．
static void upload_operands(Oz2Run& r, std::vector<double>& A, std::vector<double>& B) {
    auto& md = r.md; auto& cq = md->mesh_command_queue();
    const uint32_t f32t = tt::tile_size(tt::DataFormat::Float32), bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    auto mkb = [&](size_t nt, uint32_t ps) { return mesh_dram_buffer(md, nt, ps); };
    // 法の集合を段の時間のモデルで選ぶ経路（awe．BF16・FP16・BFP8_b・BFP4_b・BFP2_b．ozaki2_awe.h）は確保の順序を変える：
    // 手放さない小さい器（最大値・指数差分・旗）を FP64 のオペランドより先に取り，桁行列は DRAM の上端から取る
    // （build_slice）．FP64 のオペランドを先に取る順序だと，スライスの後で FP64 を手放した穴（14080^3 で 3.2 GB）が
    // 後から取った小さい器と桁行列に挟まれて残り，剰余の器（法の本数 S 平面．BFP2_b の S=20 で 15.9 GB）が連続した
    // 空きに収まらず確保に失敗する．この順序なら FP64 の穴が桁積と剰余を置く空きとつながる．他の経路は FP64 の
    // オペランドを先に取り，桁行列も上端からは取らない．
    const uint32_t nslot = r.gd.x * r.gd.y;
    auto mk_small = [&]() {
        r.maxAb = mkb(r.Mt, bf16t); r.maxBb = mkb(r.Nt, bf16t);     // 行／列ごとの最大値
        r.dltAb = mkb(r.Mt, f32t); r.dltBb = mkb(r.Nt, f32t);       // スライサへ渡す指数差分（Int32）
        // 範囲外検出の旗はオペランドを送る前にゼロ化しておく（仕様：A・B を送ってから
        // C が戻るまでホストと通信しない）．
        // 旗はコアごとに 1 タイル（重複 NoC 書き込みの振る舞いに依存しないため）
        r.flgA = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32));
        r.flgB = mesh_dram_buffer(md, nslot, tt::tile_size(tt::DataFormat::Int32)); };
    if (OZ2_PARAM.awe) mk_small();
    // FP64 のオペランドは 1 要素 8 バイト＝1 ページ OZ_F64_TILE_BYTES（8192）バイト
    r.rAf64 = mkb(r.ntotA, OZ_F64_TILE_BYTES); r.rBf64 = mkb(r.ntotB, OZ_F64_TILE_BYTES);
    if (!OZ2_PARAM.awe) mk_small();
    { std::vector<uint32_t> z((size_t)nslot * tt::tile_size(tt::DataFormat::Int32) / 4, 0);
      dist::EnqueueWriteMeshBuffer(cq, r.flgA, z, false);
      dist::EnqueueWriteMeshBuffer(cq, r.flgB, z, false);
      dist::Finish(cq); }
    upload_operand_f64(md, A, r.M, r.K, r.rAf64);
    upload_operand_f64(md, B, r.K, r.N, r.rBf64);
    // オペランドはデバイスに載った．参照解はすでに計算済みで，スケールを掛け戻すのも
    // デバイス側なので，ホスト側の FP64 は手放す（14080^3 で 3.2 GB）．
    // OZ2_CHECK_CONCAT（出力パネル 1 枚の小さい構成だけ）は独立な整数参照を作るので残す．OZ_DUMP_C（調査用．
    // 検証を省く実行との併用は check_env が，OZ2_RECON_NOF64 との併用は main が止める）も書き出しに A・B を使うので残す．
    if (r.dg.check_concat || (r.dg.dump && r.checked())) { r.Ahost = std::move(A); r.Bhost = std::move(B); }
    std::vector<double>().swap(A); std::vector<double>().swap(B);
}

// ---------------- 第 0 段：正規化スケール，第 1 段：モジュラースライス（デバイス）----------------
// 構築（この順）：scCA・scCB（最大値），edA・edB（指数差分と範囲外の旗），slA・slB
// （スライサ．剰余行列の器 slA->out・slB->out を確保する．ozaki2_stages.hpp の build_slice）．初回の投入で JIT を済ませた
// のち 2 つの段を別々に計時する（tsc・tsl）．
// 解放：rAf64・rBf64・maxAb・maxBb（dltAb・dltBb は再構成が使うので残す）．
// 法を組に分けるとき（ngroups > 1）は第 0 段だけを行い，スライスと FP64 の解放は
// 第 2 段が組ごとに行う（slice_group）．出力を行の帯に分けるとき（nband > 1）は第 0 段と B のスライスだけを行い，
// A は帯ごとに slice_band が切る（rAf64 はそこで最後の帯を切ってから手放す）．
static void scale_and_slice(Oz2Run& r) {
    auto& md = r.md; auto& cq = md->mesh_command_queue();
    const uint32_t Mt = r.Mt, Nt = r.Nt, Kt = r.Kt;
    // A は行ごと，B は列ごとに最大絶対値を取り，2 のべきへ切り上げる（scheme I と同じ）．
    // これが無いと |x| <= 1 の入力しか扱えない．最大値からべき指数 e を求めて
    // ブロードキャストタイルへ展開する仕上げもデバイス上である（expdelta.cpp）：
    // A・B を送ってから C が戻るまでホストと通信しない仕様のため．
    // |x| の BF16 化は切り上げる：最大値が真値を下回ると正規化後の値が 1 を超え，
    // クランプを持たない scheme II では |A'| <= 2^b を破って CRT の窓から出る．
    r.scCA.emplace(scale_build(md, 0u, Kt, r.rAf64, r.maxAb, Mt, 0u, true));
    r.scCB.emplace(scale_build(md, 1u, Kt, r.rBf64, r.maxBb, Nt, Nt, true));
    // スライサが使うのは 2^(b-(ND-1)*G-e)（oz2_shift）．初手で最下位の桁の位まで落としておくと，
    // カーネルは桁を ND 個取り出すだけで済む（oz2_digits.h）．
    const int32_t sh_i = oz2_shift(r);
    r.edA.emplace(expdelta_build(md, r.maxAb, r.dltAb, r.flgA, Mt, true, sh_i));
    r.edB.emplace(expdelta_build(md, r.maxBb, r.dltBb, r.flgB, Nt, false, sh_i));
    auto run_scale = [&] {
        scale_run(md, *r.scCA); scale_run(md, *r.scCB);
        dist::Finish(cq);
        expdelta_run(md, *r.edA); expdelta_run(md, *r.edB);
        dist::Finish(cq); };
    if (r.ngroups > 1) {
        run_scale();   // 初回はここで JIT ビルド
        r.tsc = oz_eval::time_dispatch("scale", r.warmup, r.iters, run_scale);
        if (r.whole) {   // 全工程の計時の先頭：最大値 → 指数差分（スライスは組ごとに slice_group が記録する）
            Oz2Run* rp = &r;
            r.whole->record(oz_whole::HEAD, [rp] {
                Oz2Run& r = *rp;
                scale_run(r.md, *r.scCA); scale_run(r.md, *r.scCB);
                expdelta_run(r.md, *r.edA); expdelta_run(r.md, *r.edB); });
        }
        return;
    }

    const Oz2SliceGroup all = make_slice_group(r.mo, 0u, r.S);
    if (r.nband > 1) {   // B だけを全法ぶん切る（A は帯ごとに slice_band）
        r.slB.emplace(build_slice(md, all, r.ND, r.K, r.N, r.rBf64, r.dltBb, 1u));
        auto run_sliceB = [&] { dist::EnqueueMeshWorkload(cq, r.slB->workload, false); };
        run_scale(); run_sliceB(); dist::Finish(cq);   // 初回はここで JIT ビルド
        r.tsc = oz_eval::time_dispatch("scale", r.warmup, r.iters, run_scale);
        r.tsl = oz_eval::time_dispatch("slice", r.warmup, r.iters, run_sliceB, [&] { dist::Finish(cq); });
        if (r.whole) {   // 全工程の計時の先頭：最大値 → 指数差分 → B のスライス（A のスライスは帯ごとに slice_band が記録する）
            Oz2Run* rp = &r;
            r.whole->record(oz_whole::HEAD, [rp] {
                Oz2Run& r = *rp; auto& cq = r.md->mesh_command_queue();
                scale_run(r.md, *r.scCA); scale_run(r.md, *r.scCB);
                expdelta_run(r.md, *r.edA); expdelta_run(r.md, *r.edB);
                dist::EnqueueMeshWorkload(cq, r.slB->workload, false); });
        }
        if (r.may_release()) { r.rBf64.reset(); r.maxAb.reset(); r.maxBb.reset(); }
        return;
    }

    // A と B をモジュラースライスする（全法を 1 組で）
    r.slA.emplace(build_slice(md, all, r.ND, r.M, r.K, r.rAf64, r.dltAb, 0u));
    r.slB.emplace(build_slice(md, all, r.ND, r.K, r.N, r.rBf64, r.dltBb, 1u));
    auto run_slices = [&] {
        dist::EnqueueMeshWorkload(cq, r.slA->workload, false);
        dist::EnqueueMeshWorkload(cq, r.slB->workload, false); };
    run_scale(); run_slices(); dist::Finish(cq);   // 初回はここで JIT ビルド
    r.tsc = oz_eval::time_dispatch("scale", r.warmup, r.iters, run_scale);   // 内部で同期する
    r.tsl = oz_eval::time_dispatch("slice", r.warmup, r.iters, run_slices,
                                   [&] { dist::Finish(cq); });
    if (r.whole) {   // 全工程の計時の先頭：最大値 → 指数差分 → スライス（同期なし．同じキューなので投入順に実行される）
        Oz2Run* rp = &r;
        r.whole->record(oz_whole::HEAD, [rp] {
            Oz2Run& r = *rp; auto& cq = r.md->mesh_command_queue();
            scale_run(r.md, *r.scCA); scale_run(r.md, *r.scCB);
            expdelta_run(r.md, *r.edA); expdelta_run(r.md, *r.edB);
            dist::EnqueueMeshWorkload(cq, r.slA->workload, false);
            dist::EnqueueMeshWorkload(cq, r.slB->workload, false); });
    }
    if (!r.may_release()) return;   // 全工程を流し直すので FP64 のオペランドを手放さない
    // スライスが済んだらオペランド側のバッファを手放す．14080^3 では FP64 の A・B で
    // 3.2 GB あり，このあと確保する C_i（12.7 GB）と同居させる理由がない．
    // 以降このワークロードを投入し直すことはないので，実行時引数に焼き込んだ
    // アドレスが無効になっても構わない．
    r.rAf64.reset(); r.rBf64.reset(); r.maxAb.reset(); r.maxBb.reset();
    // dltAb/dltBb は手放さない：再構成のライタが tau_A*tau_B を指数へ足し戻すのに使う
    // （Mt + Nt タイルなので容量は問題にならない）．
}

// ---- 法の組 g をスライスする（ngroups > 1 のときだけ．第 2 段から呼ぶ）----
// 構築：slA・slB（組 g の法だけのスライサ．剰余行列の器を確保する）．計時は scale_and_slice の
// スライスと同じ手順（初回の投入で JIT を済ませ，warmup・iters で測る）で，tsl に積む．
// 解放：最後の組を切ったら rAf64・rBf64・maxAb・maxBb．組のスライス出力は呼び出し側が
// その組の GEMM を済ませてから手放す．
// OZ_TIME_WHOLE では何も手放さず，全組のスライサが同じ器 slAsh・slBsh（組の枚数の最大で取る）へ書く（組ごとに取ると全組ぶんの
// 桁行列が同時に載る）．組 g のスライスは，その組の GEMM と合成の前に流す手順として記録する（同じキューなので，次の組の
// スライスは前の組の GEMM・合成が読み終えてから器を上書きする）．
// 組の本数 S がコンパイル時引数なので，本数の違う組ではスライサが別にビルドされる．
static void slice_group(Oz2Run& r, uint32_t g) {
    auto& md = r.md; auto& cq = md->mesh_command_queue();
    const Oz2SliceGroup sub = make_slice_group(r.mo, r.gbeg[g], r.gbeg[g + 1]);
    if (r.whole && !r.slAsh) {
        uint32_t mx = 0;   // 組の出力の平面の枚数の最大（build_slice の Sout．桁分割は桁行列，INT8 は剰余行列の枚数）
        for (uint32_t k = 0; k < r.ngroups; k++) {
            const Oz2SliceGroup gk = make_slice_group(r.mo, r.gbeg[k], r.gbeg[k + 1]);
            mx = std::max<uint32_t>(mx, OZ2_DIGIT ? (uint32_t)gk.nplane : (uint32_t)gk.s);
        }
        const uint32_t slt = tt::tile_size(OZ_FMT.slice_out);
        r.slAsh = mesh_dram_buffer(md, (size_t)mx * r.ntotA, slt, /*top_down=*/OZ2_PARAM.awe != 0);
        r.slBsh = mesh_dram_buffer(md, (size_t)mx * r.ntotB, slt, /*top_down=*/OZ2_PARAM.awe != 0);
    }
    r.slA.reset(); r.slB.reset();
    r.slA.emplace(build_slice(md, sub, r.ND, r.M, r.K, r.rAf64, r.dltAb, 0u, r.slAsh));
    r.slB.emplace(build_slice(md, sub, r.ND, r.K, r.N, r.rBf64, r.dltBb, 1u, r.slBsh));
    auto run_slices = [&] {
        dist::EnqueueMeshWorkload(cq, r.slA->workload, false);
        dist::EnqueueMeshWorkload(cq, r.slB->workload, false); };
    run_slices(); dist::Finish(cq);   // 初回はここで JIT ビルド
    r.tsl += oz_eval::time_dispatch("slice", r.warmup, r.iters, run_slices,
                                    [&] { dist::Finish(cq); });
    if (r.whole) {   // 全工程の計時のために取っておく（スライサの器は slAsh・slBsh が生かしておく）
        auto wa = std::make_shared<dist::MeshWorkload>(std::move(r.slA->workload));
        auto wb = std::make_shared<dist::MeshWorkload>(std::move(r.slB->workload));
        dist::MeshCommandQueue* q = &cq;
        r.whole->record(oz_whole::HEAD, [q, wa, wb] {
            dist::EnqueueMeshWorkload(*q, *wa, false);
            dist::EnqueueMeshWorkload(*q, *wb, false); });
    }
    if (g + 1 == r.ngroups && r.may_release()) { r.rAf64.reset(); r.rBf64.reset(); r.maxAb.reset(); r.maxBb.reset(); }
}

// ---- 帯 r.band の A をスライスする（nband > 1 のときだけ．main の帯の回し方から呼ぶ）----
// 構築：slA（帯の行 bMt タイル行だけを全法ぶん切るスライサ．全帯が 1 本の器 slAsh（帯 1 本ぶん）へ書く．FP64 は帯の先頭の
// タイル aOff から読む）．計時は scale_and_slice のスライスと同じ手順（初回の投入で JIT を済ませ，warmup・iters で測る）で，
// tsl に積む．解放：最後の帯を切ったら rAf64（段の途中で手放せるときだけ）．
// 同じキューなので，次の帯のスライスは前の帯の GEMM が器を読み終えてから上書きする
static void slice_band(Oz2Run& r) {
    auto& md = r.md; auto& cq = md->mesh_command_queue();
    const Oz2SliceGroup all = make_slice_group(r.mo, 0u, r.S);
    if (!r.slAsh)
        r.slAsh = mesh_dram_buffer(md, (size_t)(OZ2_DIGIT ? (uint32_t)all.nplane : (uint32_t)all.s) * r.ntotAb(),
                                   tt::tile_size(OZ_FMT.slice_out), /*top_down=*/OZ2_PARAM.awe != 0);
    r.slA.reset();
    r.slA.emplace(build_slice(md, all, r.ND, r.bMt() * 32, r.K, r.rAf64, r.dltAb, 0u, r.slAsh, r.aOff()));
    auto run_sliceA = [&] { dist::EnqueueMeshWorkload(cq, r.slA->workload, false); };
    run_sliceA(); dist::Finish(cq);   // 初回はここで JIT ビルド
    r.tsl += oz_eval::time_dispatch("slice", r.warmup, r.iters, run_sliceA, [&] { dist::Finish(cq); });
    if (r.whole) {   // 全工程の計時のために取っておく（スライサの器は slAsh が生かしておく）
        auto wa = std::make_shared<dist::MeshWorkload>(std::move(r.slA->workload));
        dist::MeshCommandQueue* q = &cq;
        r.whole->record(oz_whole::HEAD, [q, wa] { dist::EnqueueMeshWorkload(*q, *wa, false); });
    }
    if (r.last_band() && r.may_release()) r.rAf64.reset();
}

#ifdef OZ2_DIGITK
// 1 回の投入で jn 本を流す合成段（cmp_split のときは本数ごとに深さ jn*ceil(OZ2_COMPOSE_CB_MIN/jn) で組む）
static Oz2Compose& compose_for(Oz2Run& r, uint32_t jn) {
    if (!r.cmp_split) return *r.cmp;
    auto it = r.cmp_by_jn.find(jn);
    if (it == r.cmp_by_jn.end()) {
        const uint32_t depth = jn * ((OZ2_COMPOSE_CB_MIN + jn - 1) / jn);
        it = r.cmp_by_jn.emplace(jn, std::make_unique<Oz2Compose>(build_compose(r.md, r.ntotCb(), r.Dbuf, r.crt_in, depth))).first;
    }
    return *it->second;
}

// 連結の検査（OZ2_CHECK_CONCAT）が読む実行の状態（呼ぶたびに詰める．スライス出力は法の組ごとに作り直される）
static Oz2ConcatRun concat_view(const Oz2Run& r) {
    return {r.md, r.M, r.N, r.K, r.Nt, r.Kt, r.ntotA, r.ntotB, r.ntotC, (int)r.b, r.ND, r.two_pass, &r.mo, OZ2_PARAM, OZ_FMT.slice_out,
            r.slA ? r.slA->out : nullptr, r.slB ? r.slB->out : nullptr,
            r.slA ? r.slA->layout : (uint32_t)oz2_args::LAYOUT_PLANES, r.slB ? r.slB->layout : (uint32_t)oz2_args::LAYOUT_PLANES,
            r.dltAb, r.dltBb, r.Dbuf, r.crt_in, &r.Ahost, &r.Bhost};
}

// ---------------- 第 2 段と第 2.5 段（桁分割の 7 経路）：桁積の GEMM と，法ごとの合成 ----------------
// 確保（この順）：Dbuf（桁積の平面 cplan.planes 枚），crt_in（法ごとの剰余 S 平面），
// 続いて合成段のワークロード cmp を構築する．法ごと・塊ごとに GEMM と合成を発行して
// tg・tcp に積む．解放：Dbuf（剰余は crt_in に移っており，crt_in は再構成が読むので残す）．
// 合成済みの剰余には対称範囲からのはみ出しがある．1桁経路は未還元の整数積を渡す（RAWMASK）．
// いずれもホストがFP32の無誤差整数域を確認するので，法あたりFP32平面1枚でよい．
// 桁積は 1 本のバッファに平面として並べる．法 1 本あたりの GEMM の出力（連結後）の最大が OZ2_COMPOSE_NOSPLIT_MAX 本以下なら
// 切らずに置き，それを超える構成（連結しない平衡 5 桁の法あたり 25 本など）は OZ2_COMPOSE_CHUNK 本ずつ切って，
// 桁積の塊と部分和の平面を置く（ozaki2_plan.hpp の make_compose_plan）．
// 法 1 本あたりの桁行列は 2 枚（平方数と WIDE），3 枚（Karatsuba），1 枚（1 桁），n 枚（平衡 n 桁）で，
// 積はその組み合わせを 3〜4 本（1 桁は 1 本，平衡 n 桁は n^2 本まで）とる．
// 位（桁の番号の和）が同じ積は合成の係数も同じなので，A の桁行列を K 方向に並べ B の桁行列を
// 法の中で逆順に積んだ置き方（スライスのライタの LAYOUT_A_CAT・LAYOUT_B_PERM）の上で，K 方向の連続した範囲の
// 1 本の GEMM にまとめる（g 本の和 g*K*d^2 に法を足しても 2^24 以下に収まる範囲で．maxg）．
// まとめ方と合成の係数は ozaki2_split.h の oz2_product_groups が持つ．
// 合成は法について線形なので D 自体は作らず，桁積を法へ畳んでから係数を掛ける
// （ozaki2/kernels/oz2_compose.cpp）．桁積が FP32 の無誤差整数に載ること（桁の上限 d に対し
// |P_j| <= K*d^2 <= 2^24．本経路の無誤差条件そのもの）に加え，合成の総和が剰余演算の
// 範囲 2^22 に収まることが要る．どちらも ozaki2_moduli.h の法選択が検査する．
// 合成段のワークロードは 1 回だけ組み，法ごとに実行時引数を差し替えて投入する．
// 法を組に分けるとき（ngroups > 1）は，Dbuf・crt_in・cmp のあと組ごとに slice_group で
// スライス出力を確保し，その組の GEMM と合成を済ませてから手放す．

// 法 s の GEMM の出力 j0..j0+jn-1（塊 1 つ）を，出力パネルごとに発行する（パネル行 br → パネル列 bc → 出力 j の順）．
// 出力 j は A の桁行列 a0..a0+g-1 と B の位置 b0..b0+g-1 の積の和で，K の長さは g*K になる（ozaki2_split.h の
// oz2_product_groups．まとめない（maxg=1）と桁積 1 本ずつ）．置き方（LAYOUT）は全体の maxg で決まるので，連結しない法も
// 連結した置き方の上で g=1 で読む．PA は組の桁行列の枚数，po は組の中での法 s の先頭の桁行列．
// 1 桁の法（raw）は GEMM の出力を剰余の平面 s へ直接書き（再構成が RAWMASK で法へ畳む．ozaki2/kernels/recon_crt.cpp），
// ほかは桁積の平面 j へ書く
static void issue_digit_chunk(Oz2Run& r, uint32_t s, uint32_t PA, uint32_t po, const Oz2ProductGroup* gr,
                              uint32_t j0, uint32_t jn, bool raw) {
    const Ozaki2Moduli& mo = r.mo;
    auto& sA = r.slA->out; auto& sB = r.slB->out;
    // 出力の平面と A の桁行列は帯 1 本ぶん（帯 1 本なら全体）で，br は帯の中の行パネルの番号
    const uint32_t K = r.K, Nt = r.Nt, ntotC = r.ntotCb();
    for (uint32_t br = 0; br < r.bpr; br++) {
        for (uint32_t bc = 0; bc < r.nbc; bc++) {
            uint32_t Mp = r.pMt * 32, dW = br * r.pMt * Nt + bc * r.pNt;
            for (uint32_t j = 0; j < jn; j++) {
                const Oz2ProductGroup& q = gr[j0 + j];
                auto& Cb = raw ? r.crt_in : r.Dbuf;
                const uint32_t cw = (raw ? s : j) * ntotC + dW;
                if (mo.maxg > 1)   // 連結した置き方（スライスのライタの LAYOUT_A_CAT・LAYOUT_B_PERM）
                    r.tg += run_partial_gemm(r, sA, sB,
                              panel_off_Acat(r, PA, (uint32_t)(po + q.a0), br),
                              panel_off_B(r, (uint32_t)(po + q.b0), bc),
                              Mp, (uint32_t)q.g * K, Cb, Nt, cw, PA * r.Kt);
                else {             // 連結しない（g=1）：桁行列ごとの平面のまま．B の位置 b0 は桁 np-1-b0
                    const uint32_t npl = (uint32_t)oz2_digit_planes(mo.m[s], OZ2_PARAM.digit_bound, OZ2_PARAM.nd());
                    r.tg += run_partial_gemm(r, sA, sB,
                              panel_off_A(r, (uint32_t)(po + q.a0), br),
                              panel_off_B(r, (uint32_t)(po + (npl - 1 - q.b0)), bc),
                              Mp, K, Cb, Nt, cw);
                }
            }
        }
    }
}

// 法 s（組 grp の中）の GEMM と合成．出力 np 本を CH 本ずつの塊に切って，塊ごとに GEMM を発行し（issue_digit_chunk），
// 合成段で剰余の平面 s へ畳む．切るのは法あたりの出力が OZ2_COMPOSE_NOSPLIT_MAX 本を超える構成（連結しない平衡 5 桁の
// 15・25 本など）だけで，CH=OZ2_COMPOSE_CHUNK はその約数になっている（割り切れなくても最後の塊だけ本数が半端になるだけで
// 動く）．切るときは部分和を桁積の平面 CH+nch へ落としておき，全部の塊が終わってから係数 1 で足して法へ畳む．
// 切らない経路では CH は法ごとの本数の最大値で，CH >= np なのでどの法も 1 回で終わる（CH が np の約数とは限らない）．
// 1 桁の法（OZ2_SPLIT_ONE．桁積 1 本で係数 1）は塊に切らず，合成段を通さない
static void run_digit_modulus(Oz2Run& r, uint32_t grp, uint32_t s, uint32_t PA, uint32_t po) {
    auto& md = r.md; const Ozaki2Moduli& mo = r.mo;
    const uint32_t ntotC = r.ntotCb(), CH = r.cplan.chunk;   // 平面は帯 1 本ぶん
    Oz2ProductGroup gr[OZ2_MAXPROD];
    // 連結の上限は法ごと（maxg_m．疎な桁の経路（FP8）と 1 桁の法を混ぜる経路は法の桁の上限で違い（1 桁の法は 1），
    // 他の経路は全部 maxg）
    const uint32_t np = oz2_product_groups(mo.m[s], gr, OZ2_PARAM.digit_bound, OZ2_PARAM.nd(), mo.maxg_m[s]);
    const bool raw = mo.mode[s] == OZ2_SPLIT_ONE;
    TT_FATAL(!raw || np == 1, "run_digit_modulus: 1 桁の法 {} の出力が {} 本", mo.m[s], np);
    // 合成する法が 1 本も無い構成では CH = 0 なので，CH で刻むと進まない（make_compose_plan）
    const uint32_t CHs = raw ? np : CH;
    uint32_t nch = 0;
    for (uint32_t j0 = 0; j0 < np; j0 += CHs, nch++) {
        const uint32_t jn = std::min(CHs, np - j0);
        issue_digit_chunk(r, s, PA, po, gr, j0, jn, raw);
        // GEMM の出力の生の総和の最大値は診断用の表示なので，参照計算を行うときだけ見る（OZ2_MAXCI）．
        // 塊に切る経路では次の塊が同じ平面を上書きするので，塊ごとに見る（そのぶん読み戻しが塊の数だけ増える）．
        // 1 桁の法は剰余の平面 s に GEMM の出力が入っている
        if (r.want_maxci) {
            const Oz2PlaneMax mx = raw ? plane_absmax(md, r.crt_in, (size_t)s * ntotC, ntotC) : plane_absmax(md, r.Dbuf, 0, (size_t)jn * ntotC);
            r.maxCi = std::max(r.maxCi, mx.max);
            r.maxci_nonfinite += mx.nonfinite; r.maxci_nonint += mx.nonint; r.maxci_over += mx.over;
        }
        // 診断：この塊の出力を整数の参照と比べる（OZ2_CHECK_CONCAT）
        if (r.cc) concat_after_chunk(concat_view(r), *r.cc, grp, PA, s, gr, j0, jn, po, raw);
        if (raw) continue;   // 合成段を通らない（剰余の平面 s には GEMM の出力が入っている）
        // 切らずに済むならそのまま剰余の平面 s へ．切るときは部分和を桁積の平面 CH+nch へ落としておく
        const bool single_chunk = (np <= CH);
        r.tcp += compose_run(r, compose_for(r, jn), s, j0, jn, false, 0u,
                             single_chunk ? (uint32_t)r.crt_in->address() : (uint32_t)r.Dbuf->address(),
                             single_chunk ? s : (CH + nch));
    }
    if (!raw && np > CH)   // 部分和 nch 枚（平面 CH..CH+nch-1）を係数 1 で足して法へ畳む
        r.tcp += compose_run(r, compose_for(r, nch), s, 0u, nch, true, CH,
                             (uint32_t)r.crt_in->address(), s);
}

static void run_digit_gemms(Oz2Run& r) {
    auto& md = r.md; const Ozaki2Moduli& mo = r.mo;
    const uint32_t S = r.S, ntotC = r.ntotCb();   // 桁積の平面と剰余は帯 1 本ぶん（帯 1 本なら全体）
    const uint32_t oz2_out_tile = tt::tile_size(OZ_FMT.out);
    // 枚数と切り方は計画で決めてあり（ozaki2_plan.hpp の make_compose_plan），確保はその結果から取る
    const uint32_t nD = r.cplan.planes, CH = r.cplan.chunk;
    if (r.band == 0) {   // 器と合成段は初めの帯で作り，全帯で使い回す
        // 合成する法が無い（全部 1 桁の法）なら桁積の平面も合成段も作らない（make_compose_plan）
        if (nD) r.Dbuf = mesh_dram_buffer(md, (size_t)nD * ntotC, oz2_out_tile);
        r.crt_in = mesh_dram_buffer(md, (size_t)S * ntotC, oz2_out_tile);
        if (nD) {
            const uint32_t depth = compose_cb_depth(mo, CH);
            // 公倍数が上限を超えるときだけ本数ごとに組む．それ以下（深さ 12〜45）は合成段を 1 つだけ組む
            r.cmp_split = depth > OZ2_COMPOSE_CB_MAX;
            if (!r.cmp_split) r.cmp.emplace(build_compose(md, ntotC, r.Dbuf, r.crt_in, depth));
        }
        // 診断 OZ2_CHECK_CONCAT（出力パネル 1 枚の構成だけ．それ以外は main が plan_panels の直後に止める）
        if (r.dg.check_concat) r.cc.emplace();
    }
    // 法の組ごとに回す（1 組なら全法を 1 回で回す）．組が複数なら組ごとに切り，
    // 組の法の GEMM と合成を済ませてからスライス出力を手放す．剰余は大域の法番号 s の平面へ書く．
    for (uint32_t grp = 0; grp < r.ngroups; grp++) {
        if (r.ngroups > 1) slice_group(r, grp);
        const int p0 = mo.poff[r.gbeg[grp]];   // 組の先頭の桁行列（スライス出力の平面 0）
        // 組の桁行列の枚数（A を K 方向に連結した行列の幅 PA*Kt を決める）
        const uint32_t PA = group_planes(mo, r.gbeg[grp], r.gbeg[grp + 1]);
        // スライサが書いた置き方と，ここで読む前提が同じか（組分けの部分集合でも）：平面の枚数，
        // 連結の上限，置き方（連結するなら A は LAYOUT_A_CAT・B は LAYOUT_B_PERM，しないなら両方 LAYOUT_PLANES）
        TT_FATAL(r.slA->planes == PA && r.slB->planes == PA,
                 "run_digit_gemms: 組 {} のスライスの平面が A {}・B {} 枚（GEMM は {} 枚で読む）", grp, r.slA->planes, r.slB->planes, PA);
        TT_FATAL(r.slA->maxg == mo.maxg && r.slB->maxg == mo.maxg,
                 "run_digit_gemms: スライスの maxg {}/{} が GEMM の {} と違う", r.slA->maxg, r.slB->maxg, mo.maxg);
        TT_FATAL(r.slA->layout == (mo.maxg > 1 ? oz2_args::LAYOUT_A_CAT : oz2_args::LAYOUT_PLANES) &&
                 r.slB->layout == (mo.maxg > 1 ? oz2_args::LAYOUT_B_PERM : oz2_args::LAYOUT_PLANES),
                 "run_digit_gemms: スライスの置き方 A {}・B {} が maxg={} と合わない", r.slA->layout, r.slB->layout, mo.maxg);
        for (uint32_t s = r.gbeg[grp]; s < r.gbeg[grp + 1]; s++)
            run_digit_modulus(r, grp, s, PA, (uint32_t)(mo.poff[s] - p0));
        if (r.ngroups > 1 && r.may_release()) { r.slA->out.reset(); r.slB->out.reset(); }
    }
    // 診断：独立な整数参照と剰余を比べる（全法の合成の後．OZ2_CHECK_CONCAT）
    if (r.cc) concat_after_compose(concat_view(r), *r.cc);
    // 桁積はここで用済みになる（剰余は crt_in に移っている）．手放すのは桁積の平面 nD 枚で，
    // 14080^3 なら 1 枚 0.79 GB（4 枚で 3.2 GB）．crt_in は S 枚で，14080^3 の BF16 の S=14 なら 11.1 GB．
    // 帯に分けるときは次の帯が使い回すので，最後の帯の後で手放す
    if (!r.last_band()) return;
    if (r.whole) r.whole->keep(r.Dbuf);
    r.Dbuf.reset();
}
#else
// ---------------- 第 2 段（INT8 経路）：法ごとに 1 本の行列積 ----------------
// 確保：crt_in（S 本ぶんの C_i．符号-絶対値 int32 で法あたり 1 平面，14080^3・法 16 本で
// 12.7 GB）．C_i をそのままデバイスに残し，再構成カーネルが直接読む．解放はしない．
// 法どうしは完全に独立なので依存関係はない．C_i は法ごとに crt_in の平面 s へ直接書く．
// ホストへは一切降ろさない：符号-絶対値 int32 の復号も上位／下位への分割も再構成
// カーネルが行う．法を組に分けるときは crt_in のあと組ごとに slice_group でスライスする．
static void run_int8_gemms(Oz2Run& r) {
    auto& md = r.md;
    const uint32_t S = r.S, K = r.K, Nt = r.Nt, ntotC = r.ntotCb();   // C_i の平面は帯 1 本ぶん（帯 1 本なら全体）
    if (r.band == 0) r.crt_in = mesh_dram_buffer(md, S * ntotC, tt::tile_size(OZ_FMT.out));   // 全帯で使い回す
    // 法の組ごとに回す（桁分割経路の run_digit_gemms と同じ．1 組なら全法を 1 回で回す）．
    // スライス出力の平面は組の先頭からの相対（s - s0），C_i は大域の法番号 s の平面へ書く．
    for (uint32_t grp = 0; grp < r.ngroups; grp++) {
        if (r.ngroups > 1) slice_group(r, grp);
        auto& sA = r.slA->out; auto& sB = r.slB->out;
        const uint32_t s0 = r.gbeg[grp];
        for (uint32_t s = s0; s < r.gbeg[grp + 1]; s++) {
            for (uint32_t br = 0; br < r.bpr; br++) {   // br は帯の中の行パネルの番号
                for (uint32_t bc = 0; bc < r.nbc; bc++) {
                    uint32_t dW = (uint32_t)s * ntotC + br * r.pMt * Nt + bc * r.pNt;
                    r.tg += run_partial_gemm(r, sA, sB, panel_off_A(r, s - s0, br), panel_off_B(r, s - s0, bc),
                                             r.pMt * 32, K, r.crt_in, Nt, dW);
                }
            }
        }
        if (r.ngroups > 1 && r.may_release()) { r.slA->out.reset(); r.slB->out.reset(); }
    }
    // max|C_i| は診断用の表示なので，参照計算を行うときだけ読み戻して求める
    // （OZ_NOVERIFY を立てる性能測定では計時区間の外にも一切の転送を残さない）．
    if (r.want_maxci) {
        // 最大値を見るだけなのでタイル配置を解く必要はない．全平面を一度に読む．
        std::vector<int32_t> raw((size_t)S * ntotC * 1024);
        dist::EnqueueReadMeshBuffer(md->mesh_command_queue(), raw, r.crt_in, true);
        for (auto v : raw) r.maxCi = std::max<uint64_t>(r.maxCi, (uint64_t)std::llabs((long long)dec_sm32(v)));
    }
}
#endif

// ---- 第 2 段：S 本の独立な行列積 C_i = A_i ・ B_i ----
// 経路ごとの本体（桁分割は run_digit_gemms，INT8 は run_int8_gemms）を呼ぶ．確保と
// 解放の内訳はそれぞれの関数の説明にある．済んだらスライス出力 slA->out・slB->out を
// 手放す．
static void run_gemm_stage(Oz2Run& r) {
    // max|C_i| は「桁積が累算器の無誤差範囲に収まっているか」を見るための表示であって，
    // 精度の判定には要らない．求めるには GEMM の出力を全部ホストへ降ろして単一スレッドで
    // 走査することになる．14080^3 の FP8 経路では桁積 1 本が 14080^2*4 B = 0.79 GB で，
    // 全部の法の桁積では数十 GB を読み戻すので，精度検証そのもの（参照解が 16 スレッドで
    // 115 秒）より桁違いに重く，この段が終わらなくなる．
    // そこで既定では求めず，OZ2_MAXCI=1 のときだけ求める（0 は立てていないのと同じ．検証が有効なときだけ）．
    r.want_maxci = r.verify && r.dg.maxci;
#ifdef OZ2_DIGITK
    run_digit_gemms(r);
#else
    run_int8_gemms(r);
#endif
    // スライスしたオペランドは GEMM が済めば用済みで，再構成が読むのは積のほうだけ
    // である．14080^3 では A・B 合わせて，INT8 経路の剰余行列（法 16 本）が 6.3 GB，II-BF16 の桁行列（22 枚）が
    // 17.4 GB ある（本数・枚数は rev 042dc00 の全測定の 14080^3 の moduli 行．同じ構成で流せば同じ行が出る）ので，再構成の
    // 出力を確保する前に手放す．帯に分けるときは次の帯が使い回すので，最後の帯の後で手放す
    // （o64 は初めの帯の再構成で取るので，帯に分ける構成はスライス出力と同居する．plan_bands の見積り）
    if (r.last_band() && r.may_release()) { r.slA->out.reset(); r.slB->out.reset(); r.slAsh.reset(); }
}

// ---- 第 3 段：CRT による再構成 ----
// 確保：o64（FP64 の C，タイル順）．run_recon が limb 面 3 枚（全面は診断用 OZ2_RECON_NOF64 のときだけで，
// ほかは 1 タイルずつ）と 128 bit 固定小数の器（全面は 2 パス構成のときだけ）を初めの帯で確保して Oz2Run の rlimb・ra128 に持ち，
// 全帯で使い回して main の終わりまで残す（plan_bands の w.limb もそれを数える）．
// ここでは何も手放さない（crt_in と dltAb・dltBb は main の終わりまで残る）．
// 出力は FP64（タイル順）で，正規化スケール tau_A*tau_B と KM もデバイスが掛ける．
// 入力はホストを一切経由しない：桁分割経路は合成段が畳んだ剰余（FP32），INT8 は GEMM が書いた C_i（符号-絶対値 int32）
static void run_recon_stage(Oz2Run& r) {
    if (!r.o64) r.o64 = mesh_dram_buffer(r.md, r.ntotC, OZ_F64_TILE_BYTES);   // 出口の C は全体（帯に分けても 1 本）
    double sec = 0;
    run_recon(r, r.crt_in, oz2_in_residue(OZ2_PARAM) ? 1u : 0u, r.o64, r.dltAb, r.dltBb, 2 * oz2_shift(r), sec);
    r.tr += sec;
}

// ---- 結果の読み戻しと，共通の参照解との突き合わせ ----
// o64 を読み戻し，範囲外の入力を検査し（expdelta_check），行優先へ並べ戻して精度を
// 求める（検証が有効なときだけ．結果は acc）．デバイス側の確保・解放はしない．
// デバイスが FP64 まで組み立てているので，ホストは並べ替えるだけである．
// 診断 OZ2_RECON_NOF64 では最終パスが o64 を書かない（limb 3 面を書く）ので，o64 を読み戻さず（DRAM に前から残っていた
// 値になる），精度・CHECKSUM・書き出しを省く（入力の範囲の検査は続ける）．
// 範囲外の入力があれば理由を表示して false を返す（main がデバイスを閉じてから 2 で終わる）．
static bool read_back_and_verify(Oz2Run& r) {
    const uint32_t M = r.M, N = r.N, ntotC = r.ntotC;
    std::vector<uint32_t> Ot;
    if (r.out_written) {
        Ot.resize((size_t)ntotC * (OZ_F64_TILE_BYTES / 4));
        dist::EnqueueReadMeshBuffer(r.md->mesh_command_queue(), Ot, r.o64, true);
    }
    // 範囲外の入力の検査は C を読み終えてから．計時区間の外である．
    const bool in_range = expdelta_check(r.md, *r.edA, "行") & expdelta_check(r.md, *r.edB, "列");
    if (!in_range) return false;
    if (!r.out_written) {
        printf("Accuracy : skipped (OZ2_RECON_NOF64) -- 最終パスが出口の FP64 を書かないので，読み戻さず精度・CHECKSUM を求めない\n");
        return true;
    }
    if (r.whole) r.o64_first = Ot;   // 全工程を流し直した後の C と比べる（run_whole）
    const std::vector<double> Crm = untilize_f64(Ot, M, N);   // 行優先へ（尾崎 I と同じ ozaki_scale.hpp の並べ戻し）
    r.chash = oz_eval::chash_hex(Crm);                        // 検証を省く性能測定でも出す（CONFIG の chash）
    if (r.verify) {
        r.acc = oz_eval::accuracy(Crm, r.Chi, r.Clo); oz_eval::print_accuracy(r.acc);
        // OZ_DUMP_C（調査用．尾崎 I・基準実装と同じ形式）．書けなければ失敗とする（main がデバイスを閉じてから 1 で終わる）
        r.dump_ok = oz_eval::dump_result(Crm, r.Chi, r.Clo, r.Ahost, r.Bhost, M, r.K, N);
    }
#ifdef OZ2_DIGITK
    // 診断：出口の C を独立な整数参照 C' と比べる（OZ2_CHECK_CONCAT）
    if (r.cc) concat_after_output(concat_view(r), *r.cc, Crm);
#endif
    return true;
}

// ---- OZ_TIME_WHOLE：記録した全工程を同期せずに続けて流して計時し，C を段ごとの計時の後と比べる ----
// 記録・検査・計時の手順は ozaki2/ozaki2_whole.hpp．ここは経路の構造に依る 2 つを渡す：流し直しが書くはずの器（番兵で埋める）
// と，種類ごとの本数の期待値を数える構成．
// 埋める器：最大値 maxAb・maxBb，指数差分 dltAb・dltBb（HEAD が書く），スライス出力 slA/slB->out（HEAD），
//   再構成の入力 crt_in（桁分割は COMPOSE と 1 桁の法の GEMM，INT8 は GEMM が書く），出力 o64（最終パス）と，Oz2Run::whole が
//   生かしておいた器（桁積の平面．GEMM が書き COMPOSE が読む．128 bit 固定小数の器．2 パスの第 1 パスが書き第 2 パスが読む．
//   limb 3 面．診断用 OZ2_RECON_NOF64 の最終パスが書く）．
// FP64 のオペランドと範囲外の旗（0 に初期化してから立てるだけ）は入力なので触らない．
// WHOLE-CHECK の identical は，番兵の検査・本数の突き合わせ・計時の後の C の比較がすべて通ったときだけ 1 になる．
static void run_whole(Oz2Run& r) {
    auto& cq = r.md->mesh_command_queue();
    std::vector<std::shared_ptr<dist::MeshBuffer>> bufs = {r.maxAb, r.maxBb, r.dltAb, r.dltBb, r.slA->out, r.slB->out, r.o64,
                                                           r.crt_in};
    auto read_out = [&](std::vector<uint32_t>& Ot) {
        Ot.assign((size_t)r.ntotC * (OZ_F64_TILE_BYTES / 4), 0u);
        dist::EnqueueReadMeshBuffer(cq, Ot, r.o64, true); };
    const uint32_t npass = (uint32_t)make_recon_plan(r.mo, r.two_pass, r.ntotC, r.out_written).pass.size();
    const oz_whole::Check w = r.whole->check(r.md, bufs, read_out, r.o64_first,
                                             oz2_whole_expected(r.mo, OZ2_PARAM, r.nbr * r.nbc, r.cplan.chunk, npass, r.ngroups, r.nband));
    r.twhole = r.whole->time(r.warmup, r.iters, cq);
    std::vector<uint32_t> Ot;
    read_out(Ot);
    r.whole_same = oz_whole::report(w, Ot == r.o64_first) ? 1 : 0;
}

// ---------------- 報告（デバイスには触れない）----------------

// 性能だけを変えるつまみ（docs/env_vars.md の B）のうち，既定から外れているものを CONFIG 行の knobs 欄へ
// '+' 区切りで残す．値を持つものは 名前:値 と書く（欄の区切りの '=' と紛れないように）．
// どれを数えるかと並びはつまみの表（common/oz_knobs.hpp の R_KNOBS）が決める．OZ2_JSPLIT は答えを変えないが
// FP64 組み立ての分担を変える診断（C）なので，ここに加えて diag にも数え probe=set にする．
// 法の集合を外から与える OZ2_MODULI は答えを変える調査用（docs/env_vars.md の C）なので，ここには数えず
// diag（probe=set）と forced_moduli=1 で残す（Oz2Knobs::moduli_set）．
// 計画のつまみの構造体 Oz2Knobs（ozaki2_plan.hpp）とは別物で，こちらは記録の文字列を表から作る．
static std::string oz2_knobs_record() { return oz_knobs::record(oz_knobs::W_II, oz_knobs::R_KNOBS); }

// 行列積の生の総和の実測最大値（OZ2_MAXCI．検証が有効なときだけ）．無誤差でいられる上限と並べて出す．
// INT8 経路は int32 の符号-絶対値を 2 平面で復号するので 2^31，
// 桁分割経路は積が FP32 に累算されるので 2^24 が上限である．診断が通ったか（2^24 以下の整数でない出力が無いか）を返す
static bool print_maxci(const Oz2Run& r) {
    bool maxci_ok = true;
    if (!r.want_maxci) return maxci_ok;
    if constexpr (OZ2_DIGIT) {
        printf("max|sum| (桁積)=%llu (FP32 無誤差上限 2^24=%d)\n", (unsigned long long)r.maxCi, 1 << 24);
        // 2^24 以下の整数でない出力があれば診断は失敗（最大値はそれを除いた値から取ってある）
        maxci_ok = r.maxci_nonfinite == 0 && r.maxci_nonint == 0 && r.maxci_over == 0;
        if (!maxci_ok) {
            printf("MAXCI FAIL: 桁積の出力に有限でない値 %llu 個，整数でない値 %llu 個，2^24 を超える値 %llu 個"
                   "（上の max|sum| はそれらを除いた値の最大）\n", (unsigned long long)r.maxci_nonfinite,
                   (unsigned long long)r.maxci_nonint, (unsigned long long)r.maxci_over);
            fprintf(stderr, "ERROR: OZ2_MAXCI で桁積の出力に 2^24 以下の整数でない値がある\n");
        }
    } else {
        printf("max|C_i|=%llu (2 平面復号の上限 2^31=%lld)\n", (unsigned long long)r.maxCi, (long long)1 << 31);
    }
    return maxci_ok;
}

// 段ごとの時間の内訳（breakdown の行）．total は全段の和［秒］．
// 照合はこの人向けの行を空白まで固定した正規表現で読む（表示を整える直しで照合が外れる．外れれば照合が止まるので黙っては
// 通らない）：scale・slice A+B・N GEMMs・compose・recon (DS)・TOTAL の行は scripts/lib/logparse.py の dram2dram
// （_OZ2_PARTS・_OZ2_COMPOSE．表と図の値）と scripts/lib/oz_logs.py の oz2_stages（verify/check_paper_numbers.py の load_phi_nowin）が，
// 「slice A+B    :」は verify/check_paper_numbers.py の Derived.oz2_slice_ab と probe_checks.py も読む．
// 同じ値の機械向けの欄は RESULT の slice_ms（scale+slice A+B）・gemm_ms（合成段を含む）・recon_ms（lib/constants.tsv の
// RESULT_ozaki2_full）だけで，scale と slice A+B，GEMM と compose の内訳はこの行にしか無い
static void print_breakdown(const Oz2Run& r, double total) {
    const double tsc = r.tsc, tsl = r.tsl, tg = r.tg, tcp = r.tcp, tr = r.tr;
    const double tslice = tsc + tsl;      // スケール算出＋正規化＋整数化＋スライス
    printf("breakdown (DRAM->DRAM device time, ms):\n");
    printf("  scale        : %.4f  (行／列ごとの最大値と 2 のべきへの丸め)\n", tsc * 1e3);
    printf("  slice A+B    : %.4f  (正規化・整数化・桁分解・剰余)\n", tsl * 1e3);
    // nGemm は「アルゴリズム上の行列積の本数」（法 1 本あたり，2 桁分割では 3 本・WIDE のみ 4 本，1 桁は 1 本，
    // 平衡 n 桁は n^2 本（m が q^n を割れば n(n+1)/2 本．BFP2_b の法 243 は 15 本））．
    // 実際のカーネル発行はこれを出力パネル数 nbr*nbc 倍したものなので，1 回あたりの
    // 時間を出すときはそちらで割る（両者を混ぜると 14080^3 では 16 倍ずれる）．
    // 桁分割経路では同じ位の桁積を K 方向に連結するので，発行する GEMM は出力の本数 nout 本である
    // （MAC は桁積 nGemm 本ぶんで変わらない）．
    const uint32_t nGemm = (uint32_t)r.mo.nprod, nOut = (uint32_t)r.mo.nout, nCall = nOut * r.nbr * r.nbc;
    printf("  %u GEMMs      : %.4f  (%.4f/gemm, 出力 %u 本を %u パネル発行で %.4f/call)\n",
           nGemm, tg * 1e3, tg * 1e3 / nGemm, nOut, nCall, tg * 1e3 / nCall);
    if constexpr (OZ2_DIGIT) printf("  compose      : %.4f  (桁積の合成と法剰余)\n", tcp * 1e3);
    // 見出しの「recon (DS)」は scripts/lib/oz_logs.py などが読む名前なので変えない．DS（double-single）は古い実装の名残で，
    // 今の累積は limb 3 本の triple-float（b < 24 の 1 パス構成だけ 2 本）なので，実際のパス数と limb の本数を添える
    {
        const ReconPlan rpl = make_recon_plan(r.mo, r.two_pass, 1, r.out_written);
        printf("  recon (DS)   : %.4f  (CRT 再構成 %zu パス．累積は FP32 の limb %u 本．DS は表示名の名残)\n",
               tr * 1e3, rpl.pass.size(), rpl.pass.back().limb3 ? 3u : 2u);
    }
    printf("  ---------------------------\n");
    printf("  TOTAL        : %.4f ms  (recon %.1f%%, gemm %.1f%%, scale+slice %.1f%%)\n",
           total * 1e3, 100 * tr / total, 100 * (tg + tcp) / total, 100 * tslice / total);
}

// 精度の判定（Verify 行）．出口の FP64 を書かない診断（OZ2_RECON_NOF64）は精度を求めていないので判定しない
static bool judge_accuracy(const Oz2Run& r) {
    if (!r.out_written) {
        printf("Verify   : skipped (OZ2_RECON_NOF64) -- 精度は判定していない\n");
        return true;
    }
    return oz_eval::accuracy_ok(r.acc, r.verify);
}

// 統一指標の DRAM->DRAM 行（scheme I と同じ表記・同じ測定区間）と，集計用の RESULT・VERIFY 行．acc_ok は judge_accuracy の判定．
// DRAM->DRAM 行は scripts/lib/logparse.py の dram2dram（_D2D_II．TFLOP/s）・legacy_logs.py・summarize_cool.py・
// figures/update_fig_ozscale.py・measure_lib.sh の RUN_PAT が空白まで固定した正規表現で読む．同じ値の機械向けの欄は，全体が
// CONFIG と RESULT の d2d_ms，slice・GEMM（合成段を含む）・recon が RESULT の slice_ms・gemm_ms・recon_ms（欄の名前は
// lib/constants.tsv の RESULT_ozaki2_full）で，TFLOP/s は RESULT の M・N・K と d2d_ms から求まる
static void print_result(const Oz2Run& r, double total, bool acc_ok) {
    const uint32_t M = r.M, N = r.N, K = r.K, S = r.S, b = r.b;
    const bool checked = r.checked(); const oz_eval::Acc& acc = r.acc;
    const double tslice = r.tsc + r.tsl;
    const double d2d_tops = 2.0 * (double)M * N * K / std::max(1e-12, total) / 1e12;
    printf("DRAM->DRAM : %.3f ms/iter  (slice %.3f | GEMM %.3f | recon %.3f)  %.4f TFLOP/s fp64-equiv\n",
           total * 1e3, tslice * 1e3, (r.tg + r.tcp) * 1e3, r.tr * 1e3, d2d_tops);
    // OZ_TIME_WHOLE：全工程を段の間で同期せずに続けて流した時間（論文の全体の時間．上の DRAM->DRAM は段ごとの計時の和で，
    // 論文はフェーズの内訳に使う）
    if (r.dg.whole) oz_drv::print_whole_pipeline(r.twhole, total, 2.0 * (double)M * N * K);
    // 精度を判定しない実行では精度欄を nan にする（参照解が無いと求まらない．0・99 と書くと誤差 0 の実行と読める）
    const double rel = checked ? acc.rel_fro : NAN, ebits = checked ? acc.ebits : NAN;
    const uint32_t nGemm = (uint32_t)r.mo.nprod;
    // 集計スクリプト用の CSV 出力（先頭のタグで scheme I と区別する．欄の名前は lib/constants.tsv の RESULT_ozaki2_full で，
    // scripts/verify/check_routes.py がこのファイルの書式と突き合わせる）
    printf("RESULT,ozaki2_full,%u,%u,%u,%u,%u,%u,%.4f,%.3e,%.2f,%.4f,%.4f,%.4f\n",
           M, N, K, b, S, nGemm, total * 1e3, rel, ebits, tslice * 1e3, (r.tg + r.tcp) * 1e3, r.tr * 1e3);
    printf("VERIFY,ozaki2_full,%s\n", checked ? (acc_ok ? "pass" : "fail") : "skipped");
}

// 設定の記録（CONFIG 行と，段ごとの計時区間のテレメトリ）．RESULT 行だけでは形式も法の構成も分からず，あとからログを読み直す
// ときに構成を復元できない．時間の指標名も明記する：scheme II の RESULT の
// 時間欄はデバイス各段の合計で，DRAM->DRAM（フェーズの内訳）と同じ値である（論文の全体の時間は whole_ms）
// （scheme I の RESULT は構築と読み戻しを含む tot なので意味が違う）．
static void print_config(const Oz2Run& r, double total) {
    const Oz2Knobs& kn = r.kn; const Oz2Diag& dg = r.dg;
    const int cfg_chunk = (int)r.cplan.chunk;   // 桁分割しない経路では塊分割そのものが無い（0）
    // FP64 組み立ての分担が実際に行われたか（1 パス構成の最終パスは OZ2_RECON_NOF64 が
    // 無ければ FP64 を組み立てる．2 パス構成は第 1 パスが固定小数への変換を分担するので OZ2_RECON_NOF64 に
    // 依らない．make_recon_plan）．
    const bool jsplit_on = oz2_split_assembly(!dg.nof64 || r.two_pass, kn.jsplit);
    // 探針は環境変数の有無ではなく読んだ値で見る（OZ_GEMM_PROBE_NODRAM=0 は既定と同じ構成なので none）．
    // 診断のつまみ（つまみの表の R_DIAG．docs/env_vars.md の C・D2．2026-11-03 より前は E の OZ_TIME_WHOLE も）のどれかが既定から外れていれば
    // probe=set にし，どれが立っていたかを diag に '+' 区切りで残す（scheme I と同じ決め方．2026-10-22 より前のスキーム II は
    // diag を持たず，OZ_NO_BLOCKMM も probe=set に数え，OZ_FIDELITY・OZ2_B は数えなかった）
    const std::string diag = oz_knobs::record(oz_knobs::W_II, oz_knobs::R_DIAG);
    const std::string knobs = oz2_knobs_record();
    char recon1_txt[32] = "na";
    if (!std::isnan(r.recon1_log2)) snprintf(recon1_txt, sizeof recon1_txt, "%.2f", r.recon1_log2);
    // 段ごとの計時区間のクロック・電流・電力・温度（oz_timing.hpp の print_telem．CONFIG の timing= が計時の方式）
    oz_eval::print_telem();
    OzLine cfg("CONFIG,ozaki2_full");
    cfg.kv("fmt", OZ_FMT.name).kv("scheme", "II").kv("b", r.b).kv("s", r.S).kv("ngemm", (uint32_t)r.mo.nprod)
       .kv("digit_max", (int)OZ2_PARAM.digit_bound).kv("nrdig", (int)OZ2_PARAM.nrdig).kv("chunk", cfg_chunk)
       .kv("fidelity", oz_fidelity_name(kn.fid)).kv("shape", std::to_string(r.g.pm) + "x" + std::to_string(r.g.pn))
       .kv("bw", r.g.bw).kv("panels", std::to_string(r.nbr) + "x" + std::to_string(r.nbc)).flag("verified", r.checked())
       // 調査用のつまみの状態．既定（probe=none）でないログは通常の測定ではない．
       // scripts/verify/check_paper_numbers.py はここを見て，論文の表への採用を拒む．
       .kv("probe", diag == "none" ? "none" : "set").kv("crt_probe", dg.crt)
       .kv("recon_probe", dg.recon).kv("nsmall_req", kn.nsmall).flag("nof64", dg.nof64)
       .kv("time_in_result", "dram2dram_device_stages").num("d2d_ms", "%.4f", total * 1e3)
       // 欄を足すときは末尾に置く（検査側は key=value で読むので位置に依らない）
       .kv("jsplit", kn.jsplit).flag("noblockmm", !kn.blockmm).flag("two_pass", r.two_pass)
       .flag("jsplit_on", jsplit_on).kv("groups", r.ngroups).kv("bands", r.nband);
    oz_drv::add_gemm_impl(cfg);
    // nout は GEMM の出力の本数（同じ位の桁積を K 方向に連結した後），maxg は連結の上限．
    // gemm_probe は GEMM 段の探針（立っていれば probe=set），knobs は性能だけを変えるつまみ（oz2_knobs_record）．
    // sparse_max は実際に使った疎な桁の上限（FP8 の e4m3 で K <= 16381 なら 32．0 は使わない経路・構成），
    // nsqs はそれで運んだ法の本数，nsingle は 1 桁の法（合成段を通らない．桁 32 の AWE 経路）の本数，
    // forced_moduli は法の集合を OZ2_MODULI で与えたか（立っていれば probe=set）．
    // 段の間で読み戻す診断 check_concat・maxci（立っていれば probe=set），段の時間のモデルで法を選んだ経過（ozaki2_awe.h の
    // Oz2AweInfo．awe=na|model|table_fallback，awe_caps は打ち切りに当たったもの）
    cfg.kv("nout", (uint32_t)r.mo.nout).kv("maxg", r.mo.maxg).kv("gemm_probe", oz_gemm_probe_name(dg.gemm_probe)).kv("knobs", knobs)
       .kv("sparse_max", r.mo.smax)
       .kv("nsqs", (int)std::count(r.mo.mode.begin(), r.mo.mode.end(), (int)OZ2_SPLIT_SQS))
       .kv("nsingle", (int)std::count(r.mo.mode.begin(), r.mo.mode.end(), (int)OZ2_SPLIT_ONE))
       .flag("forced_moduli", kn.moduli_set != nullptr).flag("check_concat", dg.check_concat)
       .flag("maxci", dg.maxci).kv("awe", r.mo.awe_status).kv("awe_caps", r.mo.awe_caps);
    oz_drv::add_input_whole_timing(cfg, dg.whole, r.twhole * 1e3);
    // 再構成のパス数の決め方（fixed・auto・forced）と 1 パスの復元誤差の見込み（log2．測らない構成は na），
    // 1 パスにする条件（見込みがこれ以下なら 1 パス．-(53+OZ2_RECON1_MARGIN)），見込みの式の η の実効値
    // （OZ2_RECON1_EPS_LOG2．どれも OZ2_PASSES 行と同じ値で，照合はこちらを先に読む），
    // auto_b は OZ2_FP64 の見積もりが決めた b の上限（自動決定でなければ 0．OZ2_FP64 行の b<=），
    // win_meas は CRT の窓を入力から見積もったか（OZ2_WIN 行の有無），diag は上の診断のつまみ
    cfg.kv("pass_mode", r.pass_mode).kv("recon1_pred", recon1_txt)
       .num("recon1_thr", "%.2f", OZ2_RECON1_THR_LOG2).num("recon1_eps", "%.2f", OZ2_RECON1_EPS_LOG2)
       .kv("auto_b", r.auto_b_cap).flag("win_meas", r.win_meas).kv("diag", diag);
    oz_drv::add_common_tail(cfg);
    cfg.kv("chash", r.chash);   // 結果 C のハッシュ（到達精度のログの CHECKSUM と同じ値．照合が性能のログと突き合わせる）
    // デバイスの格子と GEMM 段の格子（どちらも 列x行．END-TO-END 行・論文の「11×10」と同じ向き．gemm cfg 行の grid= は 行x列），
    // 計画のときの DRAM の空きの総量と最大の連続した空き［GB］（2026-11-04．スキーム I と同じ欄．GEMM 以外の段はデバイスの格子の全部を使う）
    cfg.kv("dev_grid", std::to_string((uint32_t)r.gd.x) + "x" + std::to_string((uint32_t)r.gd.y))
       .kv("gemm_grid", std::to_string(r.g.cc) + "x" + std::to_string(r.g.cr))
       .num("dram_free_gb", "%.2f", r.dram0.total / 1e9).num("dram_largest_gb", "%.2f", r.dram0.largest_block / 1e9);
    cfg.print();
}

// ---- 段ごとの時間と，集計用の RESULT・VERIFY・CONFIG 行を印字する ----
// デバイスには触れない．精度の判定と診断（OZ2_MAXCI・OZ_TIME_WHOLE の WHOLE-CHECK）がすべて通ったかを返す．
static bool print_report(const Oz2Run& r) {
    const double total = r.tsc + r.tsl + r.tg + r.tcp + r.tr;   // tcp は桁分割経路の桁積合成（INT8 経路は 0）
    printf("========= Ozaki-II END-TO-END pipeline  M=%u N=%u K=%u S=%u grid=%ux%u =========\n",
           r.M, r.N, r.K, r.S, (uint32_t)r.gd.x, (uint32_t)r.gd.y);
    const bool maxci_ok = print_maxci(r);
    print_breakdown(r, total);
    const bool acc_ok = judge_accuracy(r);
    print_result(r, total, acc_ok);
    print_config(r, total);
    return acc_ok && maxci_ok && (r.whole_same != 0);
}

int main(int argc, char** argv) {
    // 入口の検査（デバイスを開く前．並びは scheme I と同じ oz_driver.hpp）：削除したつまみとスキーム I だけのつまみ
    // （OZ_FP64・OZ_RADIX など．黙って無視すると，OZ_FP64 で b を選び直したつもりの性能測定（OZ_NOVERIFY）が既定の構成を
    // 測る）→ 位置引数 → このスキームのつまみの値・入力の生成と判定・GEMM 段（OZ_L1_BUDGET_KB と 3 つの探針は GEMM の構成
    // 探索・構築で初めて読むので，ここで先に読んでおかないとデバイスを開いたまま止まる）・計時とテレメトリ
    oz_drv::reject_env(oz_knobs::W_II, argv[0]);
    const Oz2Args opt = read_run_args(argc, argv);
    oz_drv::check_env(oz_knobs::W_II, opt.K);
    // つまみはここで 1 度だけ読み，以降は Oz2Run の kn・dg から引く（ozaki2_plan.hpp．診断のほうを先に読む）
    const Oz2Diag dg = oz2_diag_from_env();
    const Oz2Knobs kn = oz2_knobs_from_env();
    // OZ2_PANEL_FORCE の書式と割り切りも開く前に見る（構成探索まで待つと，スライス段を流した後で止まる）
    { unsigned a = 0, c = 0; if (!oz2_parse_panel_force(opt.M / 32, opt.N / 32, kn.panel_force, a, c)) return 2; }
    // OZ2_CHECK_CONCAT の検査（連結した GEMM の出力・剰余・CRT の独立な整数参照）は桁分割経路のものである．INT8 経路には
    // 連結した GEMM も桁行列も無く，受け付けると CONFIG に check_concat=1・probe=set を付けてホストの A・B を保持する
    // だけで何も検査しない．黙って通さずに止める
    if (!OZ2_DIGIT && dg.check_concat) {
        fprintf(stderr, "ERROR: OZ2_CHECK_CONCAT は桁分割の経路（ozaki2_int8 以外）でだけ使える（INT8 経路には連結した GEMM が無い）\n");
        return 2;
    }
    // OZ_TIME_WHOLE の検査（ozaki2/ozaki2_whole.hpp）は出口 o64 を番兵で埋めて流し直し，段ごとの計時の後の o64 と比べる．
    // OZ2_RECON_NOF64 の最終パスは o64 を書かない（limb 3 面を書く）ので，比べるのは DRAM に前から残っていた値になり，
    // WHOLE-CHECK と CHECKSUM が前の実行で変わる．黙って通さずに止める
    if (dg.whole && dg.nof64) {
        fprintf(stderr, "ERROR: OZ_TIME_WHOLE は OZ2_RECON_NOF64 と併用できない（最終パスが出口 o64 を書かないので，"
                        "流し直しの検査が成り立たない）\n");
        return 2;
    }
    // 出口の C を読む診断（OZ2_CHECK_CONCAT の CRT 復元の比較・OZ_DUMP_C の書き出し）も，出口 o64 を書かない OZ2_RECON_NOF64
    // とは併用できない（o64 を読み戻さないので，比べも書き出しもできない．黙って省かずに止める）
    if (dg.nof64 && (dg.check_concat || dg.dump)) {
        fprintf(stderr, "ERROR: %s は OZ2_RECON_NOF64 と併用できない（最終パスが出口 o64 を書かないので，出口の C を読めない）\n",
                dg.check_concat ? "OZ2_CHECK_CONCAT" : "OZ_DUMP_C");
        return 2;
    }
    Oz2Run r;
    r.kn = kn; r.dg = dg;
    r.out_written = !dg.nof64;
    if (dg.whole) r.whole.emplace();
    r.M = opt.M; r.N = opt.N; r.K = opt.K; r.S_req = opt.s_req;
    r.warmup = opt.warmup; r.iters = opt.iters;
    // オペランドの生成と参照解は共通の host_eval を使う．これにより scheme I と
    // scheme II がビット単位で同一の入力を，同じ指標で評価される（論文の sec:cond（測定条件））．
    // オペランドは終始 FP64 で，入口では FP32 3 本組（72 bit）へ分けるだけである．
    // 2^-97 を下回る要素では下位 limb が FP32 の非正規化数から外れて桁が落ちるが，
    // 落ちる量は正規化後の桁の 2^-22 ULP 以下で結果に影響しない（ozaki_scale.hpp．下端の実測は tools/study/tfrange.cpp）．
    std::vector<double> A, B; oz_eval::gen_operands(A, B, r.M, r.K, r.N);

    if (!choose_moduli_and_width(r, kn, A, B)) return 2;   // 法と幅 b（ホストのみ．ozaki2_plan.hpp）
    // 法の組数の強制（OZ2_MOD_GROUPS．範囲は法の本数で決まるのでここで見る．不正なら exit 2）
    if (kn.groups) {
        r.groups_req = *kn.groups;
        if (r.groups_req > r.S) oz_env::fail("OZ2_MOD_GROUPS=" + std::to_string(r.groups_req) + " は 1 以上 " + std::to_string(r.S)
                                             + " 以下の整数ではない（上限は選んだ法の本数）");
    }
    // OZ2_NSMALL の上限は使うパスの断片数（1 パスは OZ2_NRP0，2 パスは第 1・第 2 パスの小さいほう）で，
    // 再構成の構築（oz2_apply_nsmall）まで待つとデバイスを開いた後で分かる．パス数は法の選択で決まっているので，
    // デバイスを開く前にここで見る．
    {
        const int w = kn.nsmall;
        const uint32_t nrp = r.two_pass ? std::min<uint32_t>(OZ2_NRP1, OZ2_NRP2) : (uint32_t)OZ2_NRP0;
        if (w >= 0 && (uint32_t)w > nrp) {
            fprintf(stderr, "ERROR: OZ2_NSMALL=%d はこの構成の断片数 %u を超える\n", w, nrp); return 2; }
    }
    open_device(r);                                 // デバイスを開き，法を印字する
    // 第 2 段のパネルと構成（ホストだけ）．取れない形は，参照解の計算・オペランドの転送・スライス段より前に止める
    if (!plan_panels(r, r.gd, kn)) { r.md->close(); return 2; }
    // OZ2_CHECK_CONCAT の整数の参照は出力パネル 1 枚の構成だけを扱う（ozaki2/ozaki2_concat_check.hpp）．パネルの枚数は
    // plan_panels で決まるので，ここで参照解の計算・オペランドの転送・スライス段より前に止める
    if (dg.check_concat && (r.nbr != 1 || r.nbc != 1)) {
        fprintf(stderr, "ERROR: OZ2_CHECK_CONCAT は出力パネル 1 枚の構成だけ（いまは %ux%u 枚）\n", r.nbr, r.nbc);
        r.md->close(); return 2;
    }
    compute_reference(r, A, B);                     // 検証が有効なら参照解を求める
    upload_operands(r, A, B);                       // FP64 の A・B を送り，ホスト側を手放す
    // 出力の行の帯（DRAM に収まらないときだけ）と法の組（OZ2_MOD_GROUPS のときだけ）．A・B を残し段の途中で器を手放さない
    // 同時確保で決める（ozaki2_plan.hpp の plan_bands）
    r.dram0 = dram_free(r.md);
    if (!plan_bands(r, r.dram0, r.out_written, kn.bands)) { r.md->close(); return 2; }
    if (!r.whole_fits) {   // 帯をパネル行 1 行まで細かくしても（強制した帯・組では）DRAM に載らない
        fprintf(stderr, "ERROR: A・B を残して全工程の器を同時に置くと DRAM に収まらないか，最大の器が連続した空きに"
                        "収まらない（上の Memory 行の見積り）\n");
        r.md->close(); return 2; }
    scale_and_slice(r);                             // 第 0 段と第 1 段（帯に分けるときは B だけ）
    for (r.band = 0; r.band < r.nband; r.band++) {  // 帯ごとに A の帯を切り，その帯の積と再構成を済ませる（帯 1 本なら 1 回）
        if (r.nband > 1) slice_band(r);             // 第 1 段（帯の A）
        run_gemm_stage(r);                          // 第 2 段（桁分割経路は第 2.5 段も）
        run_recon_stage(r);                         // 第 3 段
    }
    if (!read_back_and_verify(r)) { r.whole.reset(); r.md->close(); return 2; }   // 範囲外の入力
    if (r.whole) run_whole(r);                      // 全工程を続けて流す計時（論文の全体の時間．計時の定義は run_whole の前）
    const bool acc_ok = print_report(r);
    // 全工程の計時で取っておいたワークロードとバッファは，デバイスを閉じる前に手放す
    // （閉じた後の破棄で MeshBuffer・MeshWorkload を壊すと落ちる）
    r.whole.reset();
    const bool closed = r.md->close();
    const bool pass = acc_ok && r.dump_ok && closed;   // VERIFY 行は精度だけ．書き出しの失敗は終了コードへ
    // 精度を判定していない実行（OZ_NOVERIFY・OZ2_RECON_NOF64）は Test Passed と書かない（終了コードは 0 のまま）
    return oz_drv::finish(pass, r.checked());
}
