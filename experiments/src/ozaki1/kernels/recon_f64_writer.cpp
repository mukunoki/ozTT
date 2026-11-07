// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：再構成の結果を FP64 にして書き出すデータ移動カーネル
// ---------------------------------------------------------------------------
// 計算カーネルが出す triple-float の 3 本（c_16 上位／c_18 中位／c_17 下位）を
// FP64 に組み立てて出力する．整数演算だけで行う：デバイスの RISC-V は FP64 の演算器を
// 持たず（単精度の F 拡張も一部だけで，倍精度の D 拡張は無い），double を書くと soft-float（約 76 ns/op）になってしまうためである．
//
// 最終値は
//     C = (hi + mid + lo) * radix^-(L+2) * s_A * s_B
// で，s_A = 2^-dA・s_B = 2^-dB は正規化に実際に用いた除数（dA・dB は行ごと・列ごとの指数差分タイルの値 shift-e．
// e は行・列の最大値の BF16 表現（丸めの向きは経路ごと．scale_f64_reader.cpp の CEIL）を 2 のべきへ上げた 2^e の指数で，
// s はその 2^-shift 倍，既定の shift=-1 なら 2^(e+1)）である．2 のべきなので
// 指数への加算 -(dA+dB) で済む．radix^-(L+2) は一般には 2 のべきとは限らない（BFP4_b の基数 14，INT8 の 254）ので，
// 全形式で共通に 64 bit の仮数 W と指数 qw に分けてホストから受け取る（基数が 2 のべきなら W は 2^63 ちょうど）．
// 集めたあと W を掛けて丸める段は scheme II の
// ライタと共通で，common/kernels/f64_tf.h の pack_f64 にある（手続きの正しさは
// tools/tfconv.cpp が検査している）．
//
// コンパイル時引数の末尾 4 つで挙動が変わる：
//   EMIT_LIMBS  3 本の limb を書き出す．K ブロック化のとき，次のブロックの再構成が
//               これを読んで足し込むので，最終ブロック以外で立てる．
//   EMIT_F64    FP64 を組み立てる．要素あたり数十サイクルかかるので最終ブロックだけ．
//   SF          組み立てを C の double 式（soft-float）で行う測定モード
//               （ホスト側 OZ_SOFTFLOAT=1．論文の ssec:nofp64 の「FP64 との変換は整数命令のビット操作で行う」
//               ことの根拠を測る比較に使う）．
//   PROBE       0 以外なら，このカーネルの仕事を段階的に外す【調査用．答えは壊れる】．
//               1 は組み立て（limb と指数差分を L1 から読む部分を含む）を外す（書き出し量は変えない），
//               2 は指数差分の DRAM からの読みも外す，
//               3 は書き出しも外す．ホスト側 OZ_RECON_PROBE．
//               docs/recon_cost_breakdown.md の切り分けに使う．
//
// 実行時引数（並びは oz1_args.h の WriterRT．limb は hi・lo・mid の順）: hi_addr, num_tiles, start_id, lo_addr, mid_addr,
//             out_addr, dltA_addr, dltB_addr, Nt, Whi, Wlo, qw,
//             Wdlo, Wdhi（soft-float 版の wscale），out_off（行バンドの出力 offset）
// コンパイル時引数：TensorAccessorArgs が oz1_args.h の WriterTA の順（hi・lo・mid・out・dltA・dltB）に並び，
//             その後ろに上の 4 つ（WriterCT）

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/l1_order.h"
#include "../../common/kernels/f64_tf.h"
#include "oz1_args.h"
using namespace oz1_args;

constexpr uint32_t cb_hi  = CB_LIMB_HI;
constexpr uint32_t cb_lo  = CB_LIMB_LO;
constexpr uint32_t cb_mid = CB_LIMB_MID;
constexpr uint32_t cb_out = CB_F64;      // 作業：FP64 タイル（OZ_F64_TILE_BYTES）
constexpr uint32_t cb_da  = CB_DLT_A;    // 作業：行の指数差分
constexpr uint32_t cb_db  = CB_DLT_B;    // 作業：列の指数差分

// 比較用の soft-float 版：C の double 式で素直に書く．3 本の拡張・2 回の加算・
// 2 回の乗算で計 7 回の soft-float 呼び出しが入る．定数 wscale は double なので，
// この経路は速度だけでなく精度でも整数版に劣る（相対 2^-53 が一様に乗る）．
inline uint64_t assemble_softfloat(uint32_t hb, uint32_t mb, uint32_t lb, int32_t g,
                                   uint32_t Wdlo, uint32_t Wdhi) {
    union { float f; uint32_t u; } a, b, c;
    a.u = hb; b.u = mb; c.u = lb;
    union { double d; uint64_t u; } w, s, r;
    w.u = ((uint64_t)Wdhi << 32) | Wdlo;                 // wscale（double）
    const int32_t eb = g + 1023;
    s.u = (eb > 0 && eb < 2047) ? ((uint64_t)(uint32_t)eb << 52) : 0ull;   // 2^g
    r.d = (((double)a.f + (double)b.f) + (double)c.f) * w.d * s.d;
    return r.u;
}

void kernel_main() {
    const uint32_t hi_addr   = get_arg_val<uint32_t>(RW_HI_ADDR);
    const uint32_t num_tiles = get_arg_val<uint32_t>(RW_NUM);
    const uint32_t start_id  = get_arg_val<uint32_t>(RW_START);
    const uint32_t lo_addr   = get_arg_val<uint32_t>(RW_LO_ADDR);
    const uint32_t mid_addr  = get_arg_val<uint32_t>(RW_MID_ADDR);
    const uint32_t out_addr  = get_arg_val<uint32_t>(RW_OUT_ADDR);
    const uint32_t dA_addr   = get_arg_val<uint32_t>(RW_DLT_A_ADDR);
    const uint32_t dB_addr   = get_arg_val<uint32_t>(RW_DLT_B_ADDR);
    const uint32_t Nt        = get_arg_val<uint32_t>(RW_NT);
    const uint32_t Whi       = get_arg_val<uint32_t>(RW_WHI);
    const uint32_t Wlo       = get_arg_val<uint32_t>(RW_WLO);
    const int32_t  qw        = (int32_t)get_arg_val<uint32_t>(RW_QW);
    const uint32_t Wdlo      = get_arg_val<uint32_t>(RW_WDLO);   // soft-float 版が使う
    const uint32_t Wdhi      = get_arg_val<uint32_t>(RW_WDHI);   // wscale の double ビット列
    // RW_OUT_OFF＝出力側のタイル offset．出力全体を行バンドに切って再構成するとき（対角の
    // 累積面を全出力ぶん持たずに済ませる構成），limb バッファはバンド内の局所番号で
    // 引くが，最終出力と行・列の指数差分は全体の番号で引く必要がある．その差分を渡す．
    // バンドに切らない既定の構成では 0 で，局所番号と全体の番号が一致する．
    const uint32_t out_off   = get_arg_val<uint32_t>(RW_OUT_OFF);

    // TensorAccessorArgs は WriterTA の順（TA_HI・TA_LO・TA_MID・TA_OUT・TA_DLT_A・TA_DLT_B）
    constexpr auto h_args = TensorAccessorArgs<0>();
    const auto ah = TensorAccessor(h_args, hi_addr);
    constexpr auto l_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto al = TensorAccessor(l_args, lo_addr);
    constexpr auto m_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    const auto am = TensorAccessor(m_args, mid_addr);
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto ao = TensorAccessor(o_args, out_addr);
    constexpr auto da_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto ada = TensorAccessor(da_args, dA_addr);
    constexpr auto db_args = TensorAccessorArgs<da_args.next_compile_time_args_offset()>();
    const auto adb = TensorAccessor(db_args, dB_addr);
    static_assert(TA_COUNT == 6, "WriterTA の数と上の TensorAccessorArgs の数が合わない");
    // 上の並び（hi→lo→mid→out→dltA→dltB）は手で書いたものなので，表の番号の順と同じであることも確かめる（ホストの
    // recon_build_prog は表の名前の添字で並べる．表の 2 つを入れ替えると数は 6 のままで，limb を違う面へ黙って書く）
    static_assert(TA_HI == 0 && TA_LO == 1 && TA_MID == 2 && TA_OUT == 3 && TA_DLT_A == 4 && TA_DLT_B == 5,
                  "WriterTA の並びと上の TensorAccessorArgs の並びが合わない");
    // 末尾 4 つのコンパイル時引数（WriterCT）：limb を書くか（K ブロック累積のため），
    // FP64 を組み立てるか（最終ブロックだけ），soft-float 版か（測定用），PROBE（調査用）
    constexpr uint32_t CT0 = db_args.next_compile_time_args_offset();
    constexpr uint32_t EMIT_LIMBS = get_compile_time_arg_val(CT0 + RW_CT_EMIT_LIMBS);
    constexpr uint32_t EMIT_F64   = get_compile_time_arg_val(CT0 + RW_CT_EMIT_F64);
    constexpr uint32_t SF         = get_compile_time_arg_val(CT0 + RW_CT_SF);
    constexpr uint32_t PROBE      = get_compile_time_arg_val(CT0 + RW_CT_PROBE);

    // 作業領域（push/pop せず書き込み位置を固定して使い回す）
    const uint32_t op = get_write_ptr(cb_out);
    const uint32_t ap = get_write_ptr(cb_da);
    const uint32_t bp = get_write_ptr(cb_db);

    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_hi, 1);
        cb_wait_front(cb_mid, 1);
        cb_wait_front(cb_lo, 1);
        // limb 3 本は次の K ブロックが読むときだけ残す
        if constexpr (EMIT_LIMBS) {
            noc_async_write_tile(i, ah, get_read_ptr(cb_hi));
            noc_async_write_tile(i, am, get_read_ptr(cb_mid));
            noc_async_write_tile(i, al, get_read_ptr(cb_lo));
        }
        if constexpr (!EMIT_F64) {
            noc_async_write_barrier();
            cb_pop_front(cb_hi, 1); cb_pop_front(cb_mid, 1); cb_pop_front(cb_lo, 1);
            continue;
        }
        // 行／列の指数差分（shift-e）を引き当てる（全体の番号で引く）
        const uint32_t gi = i + out_off;
        if constexpr (PROBE < 2) {   // PROBE>=2 はこの読みを外す
            noc_async_read_tile(gi / Nt, ada, ap);
            noc_async_read_tile(gi % Nt, adb, bp);
            noc_async_read_barrier();
        }
        // 計算カーネル（パッカ）が書いた limb（cb_hi・cb_mid・cb_lo）をこの RISC-V で読むので，L0 の古い内容を捨てる．
        // cb_wait_front は L0 を捨てない．上の noc_async_read_barrier も中で捨てるが，それに頼ると指数差分の読みを
        // 省く変更（PROBE>=2 や，行ごとに読みを使い回す最適化）で黙って前提が消えるので，ここで明示する．
        invalidate_l1_cache();

        volatile tt_l1_ptr uint32_t* ph = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_hi);
        volatile tt_l1_ptr uint32_t* pm = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_mid);
        volatile tt_l1_ptr uint32_t* pl = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_lo);
        volatile tt_l1_ptr int32_t*  da = (volatile tt_l1_ptr int32_t*)ap;
        volatile tt_l1_ptr int32_t*  db = (volatile tt_l1_ptr int32_t*)bp;
        volatile tt_l1_ptr uint32_t* po = (volatile tt_l1_ptr uint32_t*)op;
        if constexpr (PROBE == 0) {
            for (uint32_t j = 0; j < 1024; ++j) {
                // 差分タイルには shift-e が入っているので，掛け戻す指数は -(dA+dB)
                const int32_t g = -(da[j] + db[j]);
                uint64_t u;
                if constexpr (SF) u = assemble_softfloat(ph[j], pm[j], pl[j], g, Wdlo, Wdhi);
                else              u = assemble_tf_f64(ph[j], pm[j], pl[j], g, Whi, Wlo, qw);
                po[2 * j]     = (uint32_t)u;        // little endian：下位語が先
                po[2 * j + 1] = (uint32_t)(u >> 32);
            }
        } else {
            // 組み立てを外す．limb の CB は待って捨てるだけで（計算カーネルとの依存を保つ）中身は読まないので，
            // limb と指数差分の L1 からの読みも組み立てと一緒に外れる．出力には定数を詰める．書き出す量は
            // PROBE==0 と同じ 1 タイル（OZ_F64_TILE_BYTES）である．
            (void)ph; (void)pm; (void)pl; (void)da; (void)db;
            for (uint32_t j = 0; j < 1024; ++j) { po[2 * j] = 0u; po[2 * j + 1] = 0u; }
        }
        if constexpr (PROBE < 3) {   // PROBE==3 は書き出しも外す
            l1_writes_landed(op + OZ_F64_TILE_BYTES - 4u);   // 組み立てた FP64 が L1 に届いてから NoC に読ませる（l1_order.h）
            noc_async_write_tile(i + out_off, ao, op);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_mid, 1);
        cb_pop_front(cb_lo, 1);
    }
}
