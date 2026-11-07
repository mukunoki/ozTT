// SPDX-License-Identifier: Apache-2.0
//
// スライサへの入力を読むデータ移動カーネル（FP64 入力版．尾崎スキーム I・II 共通）
// ---------------------------------------------------------------------------
// ホストは FP64 のオペランドをそのまま（8 バイト要素のまま）タイル順に送ってくる．
// このカーネルが FP64 -> triple-float の変換を担い，計算カーネルへ渡す．
// 変換は整数演算だけで行う：デバイスの RISC-V は FP64 の演算器を持たず（単精度の F 拡張も一部だけで，倍精度の D 拡張は無い．Blackhole の ISA 資料 BabyRISCV/InstructionSet.md），double を書くと
// soft-float（約 76 ns/op）になってしまうためである．
//
// FP64 -> triple-float の割り方と，正規化を指数へ畳む理屈は共通ヘッダ
// common/kernels/f64_tf.h にある（尾崎スキーム II のリーダも同じものを使う）．
//
// 正規化（行ごと／列ごとの 2 のべきスケール）は，FP64 の指数フィールドへ整数の
// 差分 d を足す形でここに融合してある．したがって inv（FP32 の 2^-e）をタイルで掛ける
// 乗算は要らず，スケールが FP32 の指数範囲に収まらなければならないという制約も
// 無い（正規化後の値は，スキーム I では |x/tau| <= 1，スキーム II では桁の位置合わせを含めて 2^20 以下
// （sfpu_tf_proc.h の入口の条件）なので，どちらでも limb は FP32 に収まる）．
//
// コンパイル時引数 SPLIT を立てると，1 タイルの後半 512 要素を同じコアのライタ（slice_writer.cpp）に
// 割らせる（slice_f64_split.h）．1 つの RISC-V で割るとスキーム I のスライス段の時間はこの処理で決まるので，
// 2 つの RISC-V に分けると縮む（docs/nongemm_opt_results.md の §5b．動機は §5）．スキーム II はスライサの
// SFPU 演算で決まっていてライタも別物（modslice_writer.cpp）なので立てない．
//
// 比較用に soft-float 版も持つ（コンパイル時引数 SF）．こちらは C の double 式で
// 素直に書いたもので，libgcc の soft-float 呼び出しが 1 要素あたり 8 回入る．
// 論文で「整数ビット演算でなければどれだけ遅いか」を示すためだけのもので，本番では
// 使わない（ホスト側の OZ_SOFTFLOAT=1 で切り替わる）．
//
// 実行時引数: f64_addr, dlt_addr, num_tiles, start_id, dlt_stride, mode（並びと CB 番号は slice_f64_args.h．
// ホストは common/slice_f64_host.hpp の f64_reader_ct・f64_reader_rt で詰める）

#include "api/dataflow/dataflow_api.h"
#include "f64_tf.h"
#include "slice_f64_split.h"
#include "l1_order.h"
#include "slice_f64_args.h"
using namespace slice_f64_args;

constexpr uint32_t cb_h  = CB_HI;      // hi
constexpr uint32_t cb_l  = CB_CRAW;    // craw
constexpr uint32_t cb_s  = CB_SCALE;   // s（下位 limb のスケール）
constexpr uint32_t cb_m  = CB_BRAW;    // braw
constexpr uint32_t cb_f  = CB_F64;     // 作業：FP64 タイル（OZ_F64_TILE_BYTES）
constexpr uint32_t cb_d  = CB_DLT;     // 作業：指数差分タイル（Int32）

// soft-float 版：double で x*2^d を求め，float への丸めと差で 3 本に割る素直な手順を C の double 式で書く．
// x*2^d の正規化 1 回と，3 本組への分割で計 8 回の soft-float 演算が入る．
inline void split_softfloat(uint32_t w1, uint32_t w0, int32_t d,
                            uint32_t& hb, uint32_t& mb, uint32_t& lb) {
    union { double d; uint64_t u; } xu;  xu.u = ((uint64_t)w1 << 32) | w0;
    union { double d; uint64_t u; } su;                 // 2^d（正規化係数）
    const int32_t eb = d + 1023;
    su.u = (eb > 0 && eb < 2047) ? ((uint64_t)(uint32_t)eb << 52) : 0ull;
    const double x   = xu.d * su.d;                     // 1: muldf3
    const float  hi  = (float)x;                        // 2: truncdfsf2
    const double rem = x - (double)hi;                  // 3,4: extendsfdf2, subdf3
    const float  mid = (float)rem;                      // 5: truncdfsf2
    const float  lo  = (float)(rem - (double)mid);      // 6,7,8: extend, sub, trunc
    union { float f; uint32_t u; } c;
    c.f = hi;  hb = c.u;
    c.f = mid; mb = c.u;
    c.f = lo;  lb = c.u;
}

void kernel_main() {
    const uint32_t f64_addr   = get_arg_val<uint32_t>(R_F64_ADDR);
    const uint32_t dlt_addr   = get_arg_val<uint32_t>(R_DLT_ADDR);
    const uint32_t num_tiles  = get_arg_val<uint32_t>(R_NUM);
    const uint32_t start_id   = get_arg_val<uint32_t>(R_START);
    const uint32_t dlt_stride = get_arg_val<uint32_t>(R_DLT_STRIDE);
    const uint32_t mode       = get_arg_val<uint32_t>(R_MODE);

    constexpr auto f_args = TensorAccessorArgs<0>();
    const auto fa = TensorAccessor(f_args, f64_addr);
    constexpr auto d_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    const auto da = TensorAccessor(d_args, dlt_addr);
    // 1 なら soft-float 版（測定用）．既定は 0（整数ビット演算）．
    constexpr uint32_t SF = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + CT_SF);
    // 1 なら後半をライタに割らせる（上の説明．soft-float 版では使わない）
    constexpr uint32_t SPLIT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + CT_SPLIT) && !SF;

    // c_5 / c_6 は push/pop しない作業領域で，書き込み位置を固定して使い回す（CB の生成はホスト側 slice_build が
    // 行う）．SPLIT=1 ではライタも郵便受けで受け取った番地から後半を読むので，上書きしてよいのはライタの pop の後
    // （下の reserve．slice_f64_split.h）．
    const uint32_t fp = get_write_ptr(cb_f);
    const uint32_t dp = get_write_ptr(cb_d);

    // 指数差分タイルは A（行方向）なら同じ行タイルの Kt 枚で共通なので，変わった
    // ときだけ読む．14080^3 で 1.59 GB のうち A 側が消えるが，この段は DRAM 律速では
    // ないので時間は動かない（実効 13.9 GB/s）．速さは変わらないが，読みの量が減るだけで害は無い．
    uint32_t last_vi = 0xFFFFFFFFu;
    const uint32_t end_id = start_id + num_tiles;
    if constexpr (SPLIT && MBOX_PREFILL)   // 探針 2 だけ（slice_f64_split.h）
        mbox_write_job((volatile tt_l1_ptr SliceJob*)get_write_ptr(SLICE_MBOX_CB), fp, dp, cb_h, cb_m, cb_l, cb_s);
    for (uint32_t i = start_id; i < end_id; ++i) {
        noc_async_read_tile(i, fa, fp);
        // 指数差分は行（A）または列（B）ごとに 1 枚なので，タイル番号から引き当てる
        const uint32_t vi = (mode == 0) ? (i / dlt_stride) : (i % dlt_stride);
        if (vi != last_vi) { noc_async_read_tile(vi, da, dp); last_vi = vi; }
        noc_async_read_barrier();

        cb_reserve_back(cb_h, 1); cb_reserve_back(cb_m, 1);
        cb_reserve_back(cb_l, 1); cb_reserve_back(cb_s, 1);
        if constexpr (SPLIT) {
            // 郵便受けへ仕事を置き（前の仕事はライタが pop 済み：下の reserve で待った），前半を割る
            cb_reserve_back(SLICE_MBOX_CB, 1);
            volatile tt_l1_ptr SliceJob* job = (volatile tt_l1_ptr SliceJob*)get_write_ptr(SLICE_MBOX_CB);
            mbox_reader_before_write();
            // 仕事の中身が L1 に届いてから知らせる（fence だけでは届いたことにならない．l1_order.h）
            mbox_write_job(job, fp, dp, cb_h, cb_m, cb_l, cb_s);
            mbox_reader_notify();
            slice_split_range(fp, dp, get_write_ptr(cb_h), get_write_ptr(cb_m),
                              get_write_ptr(cb_l), get_write_ptr(cb_s), 0, SLICE_SPLIT_J);
            // ライタが後半を割り終えて郵便受けを返すのを待つ．ライタは後半が L1 に届いてから
            // pop する（slice_writer.cpp）ので，返れば後半も L1 にある．
            // 返ったページはそのまま次のタイルの仕事に使う．
            cb_reserve_back(SLICE_MBOX_CB, 1);
            // 自分の前半も L1 に届いてからスライサへ渡す（最後に書いたのは s の 511 番）
            l1_writes_landed(get_write_ptr(cb_s) + 4u * (SLICE_SPLIT_J - 1));
            cb_push_back(cb_h, 1); cb_push_back(cb_m, 1);
            cb_push_back(cb_l, 1); cb_push_back(cb_s, 1);
            continue;
        }
        const uint32_t ohp = get_write_ptr(cb_h), omp = get_write_ptr(cb_m), olp = get_write_ptr(cb_l),
                       osp = get_write_ptr(cb_s);
        if constexpr (SF) {                     // 測定用：C の double 式（soft-float）
            volatile tt_l1_ptr uint32_t* src = (volatile tt_l1_ptr uint32_t*)fp;
            volatile tt_l1_ptr int32_t*  dl  = (volatile tt_l1_ptr int32_t*)dp;
            volatile tt_l1_ptr uint32_t* oh  = (volatile tt_l1_ptr uint32_t*)ohp;
            volatile tt_l1_ptr uint32_t* om  = (volatile tt_l1_ptr uint32_t*)omp;
            volatile tt_l1_ptr uint32_t* ol  = (volatile tt_l1_ptr uint32_t*)olp;
            volatile tt_l1_ptr uint32_t* os  = (volatile tt_l1_ptr uint32_t*)osp;
            for (uint32_t j = 0; j < 1024; ++j) {
                const uint32_t w0 = src[2 * j];     // little endian：下位語が先
                const uint32_t w1 = src[2 * j + 1];
                uint32_t hi, braw, craw;
                split_softfloat(w1, w0, dl[j], hi, braw, craw);
                oh[j] = hi; om[j] = braw; ol[j] = craw; os[j] = 0x3F800000u;   // s は使わないが形を揃えて置く
            }
        } else {
            // 1 タイル全部を整数ビット演算で割る（分担版の前半・ライタの後半と同じ手続き．slice_f64_split.h）
            slice_split_range(fp, dp, ohp, omp, olp, osp, 0, 1024);
        }
        l1_writes_landed(osp + 4u * 1023);   // 4 枚が L1 に届いてからスライサへ渡す（最後に書いたのは s の 1023 番．l1_order.h）
        cb_push_back(cb_h, 1); cb_push_back(cb_m, 1);
        cb_push_back(cb_l, 1); cb_push_back(cb_s, 1);
    }
}
