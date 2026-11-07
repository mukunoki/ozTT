// SPDX-License-Identifier: Apache-2.0
#pragma once
// オペランドの転送と正規化スケール（尾崎スキーム I・II 共通）
// ---------------------------------------------------------------------------
// 両スキームともスライス段の手前で，行ごと（A）／列ごと（B）の最大絶対値を求め，それを経路ごとの向き
// （切り捨て・切り上げ・2 のべきの 1 段上げ．scale_f64_reader.cpp の CEIL）で BF16 にし，その BF16 を 2 のべきへ
// 切り上げた 2^e から正規化の指数シフトを作る（デバイスの上だけで行う．論文の測定条件の節 \label{sec:cond} の計時区間は
// オペランドがデバイス DRAM にある状態から結果がデバイス DRAM に揃うまでで，その間ホストと通信しない）．2^e は切り上げる経路
// （scheme II と scheme I の INT8・BFP2_b）では最大値以上（BFP2_b は真に大きい）だが，切り捨てる経路では最大値が
// 2^k の直上（相対 2^-7 未満）のとき 2^k になり，最大値を下回る（そのぶんの超過は展開の上限の余裕が受ける．
// ozaki1_main.cpp の bf_ceil）．最大値のリダクションは mode 0 が A（行方向），mode 1 が B（列方向）で，
// 2 のべきに丸めるのでスライサ側の正規化は指数シフトだけの無誤差演算になる．
//
// この段は最後までデバイス上で閉じている：行・列ごとの最大値は
// common/kernels/scale_f64_reader.cpp（2 つのデータ移動 RISC-V に 1 本ずつ）が FP64 のビット列のまま
// 求めて BF16 にして書き（計算カーネルは使わない），最大値から「指数へ足す差分」を作る仕上げが
// common/kernels/expdelta.cpp である（expdelta_build／expdelta_run）．
// A・B を送ってから C が戻るまでホストと通信しないという仕様のため，
// 仕上げもホストに残さない．範囲外の入力だけは旗で受け，C を読んだあとに
// expdelta_check が見る．
//
// 転送はどちらのスキームも FP64 のままである（tilize_f64 / upload_operand_f64）：
// FP64 -> triple-float の変換はデバイス側（common/kernels/slice_f64_reader.cpp）が行い，
// 正規化は FP64 の指数への整数差分として同じパスに融合される．
// したがってホストは並べ替えるだけで，演算をしない．
// このヘッダはそれだけで翻訳単位として通る（tt-metal の名前は tt:: で修飾し，取り込む側の using に頼らない）．

#include "common/ozaki_gemm_host.hpp"   // nf・GridSpan・make_cb・mesh_dram_buffer
#include "common/oz_paths.hpp"          // カーネルの置き場所 OZ_COMMON_KDIR
#include "common/kernels/scale_args.h"   // スケール段のカーネルの引数の並びと CB 番号（カーネルと共有）
#include "common/kernels/f64_tf.h"       // FP64 タイルの大きさ OZ_F64_TILE_BYTES
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>


// ===== オペランドを FP64 のまま送る =====
// 変換はデバイス側で行うので，ホストがするのは「タイル順への並べ替え」だけである
// （演算は一切しない）．1 要素 8 バイト，1 ページ 1024 要素 = 8192 バイト．
// 下位語を先に置く（little endian の double のメモリ像そのまま）．
inline std::vector<uint32_t> tilize_f64(const std::vector<double>& s,
                                        uint32_t R, uint32_t C) {
    const uint32_t Rt = R / 32, Ct = C / 32;
    std::vector<uint32_t> o((size_t)Rt * Ct * 2048);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t tr = 0; tr < (int64_t)Rt; tr++)
        for (uint32_t tc = 0; tc < Ct; tc++) {
            uint32_t* t = &o[((size_t)tr * Ct + tc) * 2048];
            for (uint32_t r = 0; r < 32; r++)
                for (uint32_t c = 0; c < 32; c++) {
                    uint64_t u;
                    const double v = s[(size_t)(tr * 32 + r) * C + (tc * 32 + c)];
                    std::memcpy(&u, &v, 8);
                    const uint32_t off = nf(r, c);
                    t[2 * off]     = (uint32_t)u;
                    t[2 * off + 1] = (uint32_t)(u >> 32);
                }
        }
    return o;
}

// tilize_f64 の逆：タイル順の FP64（1 要素 2 語，下位語が先）を行優先の R×C へ戻す．
// デバイスが FP64 まで組み立てた結果を読み戻したあとに使う（ホストは並べ替えるだけ）．
inline std::vector<double> untilize_f64(const std::vector<uint32_t>& t,
                                        uint32_t R, uint32_t C) {
    const uint32_t Rt = R / 32, Ct = C / 32;
    std::vector<double> o((size_t)R * C, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t tr = 0; tr < (int64_t)Rt; tr++)
        for (uint32_t tc = 0; tc < Ct; tc++) {
            const uint32_t* s = &t[((size_t)tr * Ct + tc) * 2048];
            for (uint32_t r = 0; r < 32; r++)
                for (uint32_t c = 0; c < 32; c++) {
                    const uint32_t off = nf(r, c);
                    const uint64_t u = (uint64_t)s[2 * off] | ((uint64_t)s[2 * off + 1] << 32);
                    std::memcpy(&o[(size_t)(tr * 32 + r) * C + (tc * 32 + c)], &u, 8);
                }
        }
    return o;
}

// FP64 のオペランドを 1 本のバッファへ送る．デバイス側の
// scale_f64_reader / slice_f64_reader がここから直接読む．
// 論文の測定区間はデバイス DRAM 常駐からなので，この転送は計時の外である．
inline void upload_operand_f64(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                               const std::vector<double>& X, uint32_t R, uint32_t C,
                               std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buf) {
    auto& cq = md->mesh_command_queue();
    auto t = tilize_f64(X, R, C);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, buf, t, false);
    tt::tt_metal::distributed::Finish(cq);
}

// ===== スケールの仕上げ（デバイス上）=====
// 最大値タイルから「指数へ足す差分」タイルを作る．中身は expdelta.cpp を参照．
// A・B を転送してから C が戻るまでホストと通信しない仕様のため，ホストで std::ceil(std::log2(.)) を
// 取るのではなく，デバイス上で作る．
struct ExpDeltaCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> flag;
    uint32_t nslot = 0;      // 旗のスロット数（＝コア数）
};

// プログラムは端末の全格子に置き（担当タイルを配る full_grid_span と同じ格子），旗はコアごとに 1 スロット（1 タイル）を
// 持つ（重複した NoC の書き込みの振る舞いに依存しないため）．flagbuf は全格子のコア数ぶんのタイルを持つこと．
inline ExpDeltaCtx expdelta_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> maxbuf,
                                  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dltbuf,
                                  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> flagbuf,
                                  uint32_t ntiles, bool rowwise, int32_t shift) {
    namespace tm = tt::tt_metal;
    // 空の行列は受け付けない（ntiles=0 だと担当タイルの配分が意味を持たない）．ドライバは M・N・K が
    // 正であることを先に確かめているが，この関数を直接呼ぶ側のためにここでも止める．
    TT_FATAL(ntiles > 0, "expdelta_build: 出力が 0 枚（空の行列は受け付けない）");
    const uint32_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t i32t  = tt::tile_size(tt::DataFormat::Int32);
    // 担当タイルの配り方は全段共通の full_grid_span（コアの並びは行優先で，旗のスロットの番号もその並び）
    const GridSpan gs = full_grid_span(md, ntiles);
    const tm::CoreRange all_cores = gs.range();
    ExpDeltaCtx ec;
    ec.range = tm::distributed::MeshCoordinateRange(md->shape());
    ec.flag = flagbuf;
    ec.nslot = (uint32_t)gs.cores.size();
    TT_FATAL(flagbuf->size() >= (uint64_t)ec.nslot * i32t, "expdelta_build: 旗の器 {} B がコア {} 個ぶんのスロットに足りない",
             flagbuf->size(), ec.nslot);
    tm::distributed::MeshWorkload workload;
    tm::Program program{};
    namespace ea = expdelta_args;
    make_cb(program, all_cores, (tt::CBIndex)ea::CB_MAX, tt::DataFormat::Float16_b, 1, bf16t);
    make_cb(program, all_cores, (tt::CBIndex)ea::CB_OUT, tt::DataFormat::Int32, 1, i32t);
    std::vector<uint32_t> ct(ea::CT_COUNT);
    ct[ea::CT_ROWWISE] = rowwise ? 1u : 0u;
    tm::TensorAccessorArgs(*maxbuf).append_to(ct);
    tm::TensorAccessorArgs(*dltbuf).append_to(ct);
    tm::TensorAccessorArgs(*flagbuf).append_to(ct);
    auto k = tm::CreateKernel(program,
        OZ_COMMON_KDIR "expdelta.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                               .noc = tm::NOC::RISCV_0_default, .compile_args = ct});
    for (size_t c = 0; c < gs.cores.size(); ++c) {
        std::vector<uint32_t> rt(ea::R_COUNT);
        rt[ea::R_MAX_ADDR] = (uint32_t)maxbuf->address(); rt[ea::R_OUT_ADDR] = (uint32_t)dltbuf->address();
        rt[ea::R_FLAG_ADDR] = (uint32_t)flagbuf->address(); rt[ea::R_NUM] = gs.count(c); rt[ea::R_START] = gs.begin(c);
        rt[ea::R_SHIFT] = (uint32_t)shift;
        rt[ea::R_SLOT] = (uint32_t)c;   // 旗はコアごとに専用スロットへ
        static_assert(ea::R_COUNT == 7, "expdelta_args の RT に欄を足したらここで詰める");
        tm::SetRuntimeArgs(program, k, gs.cores[c], rt);
    }
    workload.add_program(ec.range, std::move(program));
    ec.workload = std::move(workload);
    return ec;
}

inline void expdelta_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ExpDeltaCtx& ec) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), ec.workload, false);
}

// 旗の語（全スロットの OR）を読む．ビット 0 = 最大値が BF16 の範囲を超えた（境目は expdelta_check），
// ビット 1 = 非零の最大値が 2^-126 を下回った，ビット 2 = Inf・NaN の要素がある（expdelta.cpp）．
// 組まれていない構成では 0．
inline uint32_t expdelta_flags(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const ExpDeltaCtx& ec) {
    if (!ec.flag) return 0u;
    const uint32_t wpt = tt::tile_size(tt::DataFormat::Int32) / 4;
    std::vector<uint32_t> f(wpt * ec.nslot);
    auto fb = ec.flag;
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(md->mesh_command_queue(), f, fb, true);
    uint32_t bits = 0;
    for (uint32_t i = 0; i < ec.nslot; i++) bits |= f[(size_t)i * wpt];
    return bits;
}

// C を読み終えたあとで呼ぶ．範囲外の入力があれば理由を表示して false を返す（呼び出し側がデバイスを
// 閉じてから 2 で終わる．ここで exit するとデバイスが開いたまま終わる）．範囲は行（列）ごとの最大絶対値が
// 2^-126 以上で，上端は最大値の BF16 化の向きで違う（全零の行・列は可．scale_f64_reader.cpp）：
// 切り上げる経路（bf_ceil．scheme II と scheme I の INT8・BFP2_b）は BF16 の最大有限値 2^128-2^120 以下，
// 切り捨てる経路（scheme I のほかの 6 形式）は 2^128 未満．エラー文にはその経路の境目を書く．
// Inf・NaN の要素は大きさの範囲とは別の理由として示す（2 のべきで割っても直らない）．
[[nodiscard]] static inline bool expdelta_check(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                         const ExpDeltaCtx& ec, const char* what, bool bf_ceil = true) {
    const uint32_t bits = expdelta_flags(md, ec);
    if (bits & 1u)
        std::fprintf(stderr, "ERROR: %s の最大絶対値が BF16 の範囲を超えている（この経路が受け付けるのは %s．"
                     "最大値の BF16 化を%sため）．オペランド全体を 2 のべきで割ってから渡すこと\n", what,
                     bf_ceil ? "BF16 の最大有限値 2^128-2^120 以下" : "2^128 未満",
                     bf_ceil ? "切り上げる" : "切り捨てる");
    if (bits & 2u)
        std::fprintf(stderr, "ERROR: %s の非零の最大絶対値が 2^-126 を下回るものがある（BF16 の正規化数で表せず，"
                     "その%sが黙って零になるので止める）．オペランド全体を 2 のべきで掛けてから渡すこと\n", what, what);
    if (bits & 4u)
        std::fprintf(stderr, "ERROR: %s に Inf・NaN の要素がある（受け付けるのは有限の FP64 だけ．2 のべきで割っても"
                     "直らない）\n", what);
    return bits == 0u;
}

// この段は呼び出し間で変わるものが何も無い（入出力アドレス・走査の向き・タイル配分は
// すべて構築時に確定する）ので，実行時引数は構築時に焼き込み，発行だけを scale_run が行う．
// バッファは shared_ptr で持たない：持つと呼び出し側が解放しても実体が残ってしまう．寿命は呼び出し側が握る．
// この段が DRAM に持つのは入力と最大値だけである（リーダが FP64 のビット列のまま最大値を取るので，|x| のコピーは作らない）．
struct ScaleCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
};
// nout 枚の出力タイルを作る．出力 1 枚あたり Kt 枚の入力を畳み，入力の番号は
// mode 0（A，行方向）が o*Kt+kt，mode 1（B，列方向）が kt*stride+o である．
// 入力は FP64 のオペランドそのもので，scale_f64_reader.cpp が |x| の最大値を FP64 のビット列の
// まま求め，出力の 32 個だけを BF16 にしたタイルを書く．要素ごとに BF16 の |x| を作って計算カーネル
// （行列エンジンの MAX リダクション）で畳むと，BF16 化をデータ移動の RISC-V が 1 要素ずつ行うので，
// この段の時間がその処理で決まる（両方式の時間は docs/nongemm_opt_results.md の §4）．
// bf_ceil は BF16 化を切り上げるか（scheme II は必須．scheme I は展開の上限の余裕が
// 1/128 を切る INT8 と BFP2 で立て，ほかの形式は切り捨てでよい．ozaki1_main.cpp の bf_ceil）．
// above_pow2 は切り上げに加えて，最大値がちょうど 2 のべきのときも BF16 を 1 つ上げ，スケールを最大値より
// 真に大きい 2 のべきにするか（bf_ceil が要る．scheme I の BFP2_b だけが立てる．ozaki1_main.cpp の tau_above と
// scale_f64_reader.cpp の CEIL=2）．
// プログラムは端末の全格子に置く（担当タイルを配る full_grid_span と同じ格子）．
inline ScaleCtx scale_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                            uint32_t mode, uint32_t Kt,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> refIn,
                            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> refOut,
                            uint32_t nout, uint32_t stride, bool bf_ceil = false,
                            bool above_pow2 = false) {
    namespace tm = tt::tt_metal;
    // 空の行列は受け付けない：Kt=0 だとカーネル（scale_f64_reader.cpp）はタイル 0 を先読みしたまま
    // 読みの完了を待たずに終わり，nout=0 ではコアの割り当てが意味を持たない．ドライバは M・N・K が
    // 正であることを先に確かめているが，この関数を直接呼ぶ側のためにここでも止める．
    TT_FATAL(nout > 0 && Kt > 0, "scale_build: 出力 {} 枚・K {} タイルのどちらかが 0（空の行列は受け付けない）", nout, Kt);
    TT_FATAL(!above_pow2 || bf_ceil, "scale_build: above_pow2 は bf_ceil（切り上げ）と組で使う");
    tm::distributed::MeshWorkload workload;
    auto device_range = tm::distributed::MeshCoordinateRange(md->shape());
    tm::Program program{};
    uint32_t bf16t = tt::tile_size(tt::DataFormat::Float16_b);
    // 担当タイルの配分は全段共通の full_grid_span に任せ，CB とカーネルも同じ格子に置く．
    // リーダとライタは start_o から全体でのタイル番号を計算するので，TensorAccessor の
    // 基準アドレスはバッファ全体の先頭のままでよい．
    const GridSpan gs = full_grid_span(md, nout);
    const tm::CoreRange all_cores = gs.range();
    // 同じカーネル（scale_f64_reader.cpp）を 2 つのデータ移動 RISC-V に置き，コアの担当する
    // 出力タイルを半分ずつ受け持たせる．作業領域はそれぞれ専用：FP64 タイル 2 枚（8192 B ずつ．
    // 1 枚を調べるあいだに次を読む）と，結果の BF16 タイル 1 枚（expdelta.cpp が BF16 として
    // 読む．スケールは max|A| の指数さえ分かればよいので BF16 で十分である）．
    namespace sc = scale_args;
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_F64_R1, tt::DataFormat::Float32, 2, OZ_F64_TILE_BYTES);    // RISCV_1 の FP64
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_F64_R0, tt::DataFormat::Float32, 2, OZ_F64_TILE_BYTES);    // RISCV_0 の FP64
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_OUT_R1, tt::DataFormat::Float16_b, 1, bf16t);   // RISCV_1 の結果
    make_cb(program, all_cores, (tt::CBIndex)sc::CB_OUT_R0, tt::DataFormat::Float16_b, 1, bf16t);   // RISCV_0 の結果
    auto ct_for = [&](uint32_t cbf, uint32_t cbo) {
        std::vector<uint32_t> a;
        tm::TensorAccessorArgs(*refIn).append_to(a);
        tm::TensorAccessorArgs(*refOut).append_to(a);
        const size_t o = a.size();
        a.resize(o + sc::CT_COUNT);
        a[o + sc::CT_CEIL] = above_pow2 ? 2u : bf_ceil ? 1u : 0u;   // scale_f64_reader.cpp の CEIL
        a[o + sc::CT_MODE] = mode;
        a[o + sc::CT_CB_F64] = cbf; a[o + sc::CT_CB_OUT] = cbo;
        static_assert(sc::CT_COUNT == 4, "scale_args の CT に欄を足したらここで詰める");
        return a;
    };
    auto rk = tm::CreateKernel(program,
        OZ_COMMON_KDIR "scale_f64_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_1,
                               .noc = tm::NOC::RISCV_1_default,
                               .compile_args = ct_for(sc::CB_F64_R1, sc::CB_OUT_R1)});
    auto wk = tm::CreateKernel(program,
        OZ_COMMON_KDIR "scale_f64_reader.cpp", all_cores,
        tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                               .noc = tm::NOC::RISCV_0_default,
                               .compile_args = ct_for(sc::CB_F64_R0, sc::CB_OUT_R0)});
    const uint32_t in_addr = (uint32_t)refIn->address(), out_addr = (uint32_t)refOut->address();
    for (size_t c = 0; c < gs.cores.size(); ++c) {
        const uint32_t nt = gs.count(c), start = gs.begin(c);
        // 前半（奇数なら 1 枚多いほう）を RISCV_1，残りを RISCV_0 が受け持つ（1 枚なら RISCV_1 だけ）
        const uint32_t n1 = (nt + 1) / 2, n0 = nt - n1;
        auto rt = [&](uint32_t n, uint32_t s0) {
            std::vector<uint32_t> a(sc::R_COUNT);
            a[sc::R_IN_ADDR] = in_addr; a[sc::R_OUT_ADDR] = out_addr; a[sc::R_NOUT] = n; a[sc::R_KT] = Kt;
            a[sc::R_STRIDE] = stride; a[sc::R_START_O] = s0;
            static_assert(sc::R_COUNT == 6, "scale_args の RT に欄を足したらここで詰める");
            return a;
        };
        tm::SetRuntimeArgs(program, rk, gs.cores[c], rt(n1, start));
        tm::SetRuntimeArgs(program, wk, gs.cores[c], rt(n0, start + n1));
    }
    ScaleCtx s;
    workload.add_program(device_range, std::move(program));
    s.workload = std::move(workload); s.range = device_range;
    return s;
}
// 1 回発行する（実行時引数は構築時に焼き込み済み）．
inline void scale_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, ScaleCtx& s) {
    tt::tt_metal::distributed::EnqueueMeshWorkload(md->mesh_command_queue(), s.workload, false);
}
