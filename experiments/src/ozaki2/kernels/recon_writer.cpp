// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：CRT 再構成の結果を DRAM へ書き出すデータ移動カーネル
// ---------------------------------------------------------------------------
// FINAL でないパスのうち，2 パス構成の第 1 パス（ROLE_TWO_FIRST）は，出力タイル g ごとに c_16 の
// triple-float の frac 3 本を 128 bit 固定小数（frac * 2^124．要素あたり 4 語＝16 B，w[0] が最下位）へ足し，
// 1 タイル 16 KB のページとして out0[g] へ書く（最終パスはこれを読んで自分の 3 本を足す．6 本をまとめて
// 足すのとビット単位で同じ値になる．f64_tf.h の a128_add_limbs）．診断用の OZ2_RECON_NOF64 の最終パスは
// FP32 タイル nlimb 枚（frac の上位・中位・下位）をそのまま out0[g], out1[g], out2[g] へ書き出す．
//
// FINAL が立っているパスは limb を書かず，frac から FP64 の C を組み立てて out64 へ書く
// （既定の 1 パス構成はこちらだけ）．
// 整数演算だけで行う：データ移動を担う RISC-V は FP64 の演算器を持たず（単精度の F 拡張も一部だけで，倍精度の D 拡張は無い．Blackhole の ISA 資料 BabyRISCV/InstructionSet.md），double を書くと
// soft-float（約 76 ns/op）になってしまうためである．
//
//   C = frac * KM * tau_A * tau_B
// で，tau は 2 のべきなので指数への加算で済む（行・列の指数差分タイルから引く）．
// KM は 2 のべきではないので，64 bit の仮数 W と指数 qw に分けて受け取る．
//
// scheme I のライタと違うのは「集める幅」だけである．frac の絶対精度がそのまま
// 到達精度を決める（2 パス構成で 2^-99 程度）一方 frac 自体は 2^-8 前後なので，
// 集めるには 90 bit を超える幅が要り，scheme I の 64 bit では足りない．そこで
//   acc = frac * 2^124   （128 bit 固定小数，4 語）
// に集める．limb の指数 e に対しシフト量は e-26 で，上端 2^-1 の limb でも 2^124 を
// 越えず，下端 2^-97 の limb も 1 bit も落ちない．2 パス構成の最終パス（ROLE_TWO_FINAL）は，第 1 パスが
// DRAM に残した 128 bit 固定小数を読み，そこへ自分の 3 本を足す．集めた値は 1 パスでも 2 パスでも固定小数のまま畳む（(-1/2,1/2) の値は
// 変えず，はみ出しを ±1 で戻す．像は [-1/2,1/2]．f64_tf.h）．計算カーネル（recon_crt.cpp）は畳まず，法ごとに整数部を
// 除く a0 -= round(a0) だけを行うので，各パスの limb の和は [-1/2,1/2] から a1 + a2 ぶん（尾の大きさ）
// はみ出しうる．2 パスではその和どうしを足すので，はみ出しも足し合わさる．いずれも 1 回の ±1 で戻る
// （和は (-3/2,3/2) の中．f64_tf.h の assemble_frac_f64）．計算カーネルが FP32 の和で畳むと窓の端のはみ出しを
// 見逃すので（recon_crt.cpp の「小数部の畳み込み」），ここで畳む．
// 要素の手続きはリーダの前半と同じ recon_elem.h，集める手続きも W を掛けて丸める段も common/kernels/f64_tf.h にあり
// （assemble_frac_f64），tools/tfconv.cpp がそれを直接 include して検査している．
//
// 実行時引数（並びは ozaki2/kernels/recon_args.h の W_*）:
//             out0_addr, num_tiles, start_id, out1_addr, out2_addr, nlimb,
//             out64_addr, dltA_addr, dltB_addr, Nt, Whi, Wlo, qw, sh2,
//             a128_addr（ROLE_TWO_FINAL のとき第 1 パスの 128 bit 固定小数．16 KB ページ），
//             jlo（組み立てをリーダへ渡す前半の要素数．分担しないなら 0），
//             sem_pub, sem_done（リーダとの受け渡しのセマフォ），
//             out_off（出力を行の帯に分けるときの帯の先頭タイル．o64 と指数差分の番号に足す）

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/f64_tf.h"
#include "recon_args.h"
#include "recon_elem.h"
#include "../../common/kernels/l1_order.h"
using namespace recon_args;

constexpr uint32_t cb_out = CB_LIMB;
constexpr uint32_t cb_f64 = CB_F64;    // 作業：FP64 タイル（OZ_F64_TILE_BYTES）
constexpr uint32_t cb_sync = CB_SYNC;  // リーダとの受け渡し（limb のアドレス 1 語．SYNC_PAGE B の CB）
constexpr uint32_t cb_da  = CB_DLT_A;  // 作業：行の指数差分
constexpr uint32_t cb_db  = CB_DLT_B;  // 作業：列の指数差分
constexpr uint32_t cb_q   = CB_A128;   // 作業：128 bit 固定小数 1 タイル（OZ_A128_TILE_BYTES）．第 1 パスは書く側，最終パスは読んだ側

void kernel_main() {
    const uint32_t a0    = get_arg_val<uint32_t>(W_OUT0);
    const uint32_t num   = get_arg_val<uint32_t>(W_NUM);
    const uint32_t start = get_arg_val<uint32_t>(W_START);
    const uint32_t a1    = get_arg_val<uint32_t>(W_OUT1);
    const uint32_t a2    = get_arg_val<uint32_t>(W_OUT2);
    const uint32_t nlimb = get_arg_val<uint32_t>(W_NLIMB);
    const uint32_t o64   = get_arg_val<uint32_t>(W_O64);
    const uint32_t dAa   = get_arg_val<uint32_t>(W_DLT_A);
    const uint32_t dBa   = get_arg_val<uint32_t>(W_DLT_B);
    const uint32_t Nt    = get_arg_val<uint32_t>(W_NT);
    const uint32_t Whi   = get_arg_val<uint32_t>(W_WHI);
    const uint32_t Wlo   = get_arg_val<uint32_t>(W_WLO);
    const int32_t  qw    = (int32_t)get_arg_val<uint32_t>(W_QW);
    const int32_t  sh2   = (int32_t)get_arg_val<uint32_t>(W_SH2);   // 2*(b-(ND-1)*G)
    const uint32_t q0a   = get_arg_val<uint32_t>(W_A128);
    // 組み立ての分担：jlo 未満はリーダ（recon_reader.cpp）が済ませている．
    // 分けないときは jlo=0 で 1024 要素すべてをここで組み立てる．
    const uint32_t jlo   = get_arg_val<uint32_t>(W_JLO);
    // 帯の先頭タイル：o64 と行・列の指数差分は C 全体の番号 g + off で引く（limb・a128 は帯 1 本ぶんの器なので g のまま）．
    // off は Nt の倍数（帯は行で切る）なので，列の指数差分 (g + off) % Nt は g % Nt と同じ
    const uint32_t off   = get_arg_val<uint32_t>(W_OUT_OFF);
    // 番号はセマフォに置く：プログラム起動のたびに初期値へ戻るのがセマフォだけで，
    // CB の中身は起動をまたいで残る（recon_reader.cpp の注記）．
    volatile tt_l1_ptr uint32_t* pub =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(W_SEM_PUB));
    volatile tt_l1_ptr uint32_t* done =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(W_SEM_DONE));

    constexpr auto t0a = TensorAccessorArgs<0>();
    const auto o0 = TensorAccessor(t0a, a0);
    constexpr auto t1a = TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    const auto o1 = TensorAccessor(t1a, a1);
    constexpr auto t2a = TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    const auto o2 = TensorAccessor(t2a, a2);
    // FINAL / ROLE はコンパイル時に決まる（プログラムがパスごとに別なので）．ROLE は recon_args.h の ReconRole
    // （ROLE_SINGLE = 1 パス構成，ROLE_TWO_FINAL = 2 パス構成の最終パス（第 1 パスの固定小数を読んで足す），
    // ROLE_TWO_FIRST = 2 パス構成の第 1 パス（自分の 3 本を固定小数へ足して書く．FINAL=0））
    constexpr uint32_t FINAL = get_compile_time_arg_val(t2a.next_compile_time_args_offset() + CT_FINAL_OR_SHARE);
    constexpr uint32_t ROLE  = get_compile_time_arg_val(t2a.next_compile_time_args_offset() + CT_ROLE);
    static_assert((ROLE == ROLE_SINGLE || ROLE == ROLE_TWO_FINAL || ROLE == ROLE_TWO_FIRST) && !(FINAL && ROLE == ROLE_TWO_FIRST),
                  "パスの役割は ReconRole で，第 1 パス（ROLE_TWO_FIRST）は FINAL でない");
    // PROBE【調査用．答えは壊れる】1 なら assemble_frac_f64 だけ外す．出力には定数を
    // 詰めるので指数差分の読みも書き出し量も変わらない．差は省略時の短縮量で，組み立て単体の費用ではない
    // （ホスト側 OZ_RECON_PROBE．docs/recon_cost_breakdown.md）．
    constexpr uint32_t PROBE = get_compile_time_arg_val(t2a.next_compile_time_args_offset() + CT_PROBE);
    // 上の 3 語ぶん，以降の TensorAccessor は CT_COUNT から始まる
    constexpr auto foa = TensorAccessorArgs<t2a.next_compile_time_args_offset() + CT_COUNT>();
    const auto of = TensorAccessor(foa, o64);
    constexpr auto daa = TensorAccessorArgs<foa.next_compile_time_args_offset()>();
    const auto ada = TensorAccessor(daa, dAa);
    constexpr auto dba = TensorAccessorArgs<daa.next_compile_time_args_offset()>();
    const auto adb = TensorAccessor(dba, dBa);
    constexpr auto qa = TensorAccessorArgs<dba.next_compile_time_args_offset()>();
    const auto aq0 = TensorAccessor(qa, q0a);

    const uint32_t fp = get_write_ptr(cb_f64);
    // リーダと分担するときの受け渡し（recon_reader.cpp の注記）
    volatile tt_l1_ptr uint32_t* sy = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_sync);
    const uint32_t ap = get_write_ptr(cb_da);
    const uint32_t bp = get_write_ptr(cb_db);
    const uint32_t qp = get_write_ptr(cb_q);

    for (uint32_t g = start; g < start + num; ++g) {
        cb_wait_front(cb_out, nlimb);
        const uint32_t p = get_read_ptr(cb_out), ts = get_tile_size(cb_out);
        const uint32_t tag = g - start + 1;
        // limb 3 面を DRAM へ書くのは診断用 OZ2_RECON_NOF64 の最終パスだけである．最終パスは FP64 を，
        // 2 パス構成の第 1 パスは 128 bit 固定小数を書くので，limb を DRAM へ返す必要が無い
        // （どちらも下の l0/l1/l2 のとおり L1 の CB を直接読む．limb 面を読む処理はホスト側にも
        //   最終パスにも無い）．14080 角で 2.38 GB の書き込みを省く．
        if constexpr (!FINAL && ROLE != ROLE_TWO_FIRST) {
            noc_async_write_tile(g, o0, p);
            noc_async_write_tile(g, o1, p + ts);
            if (nlimb >= 3) noc_async_write_tile(g, o2, p + 2 * ts);
        }
        if constexpr (ROLE == ROLE_TWO_FIRST) {   // 第 1 パス：自分の 3 本を固定小数へ（前半はリーダ）
            if (jlo) { sy[0] = p; *pub = tag; }
            volatile tt_l1_ptr uint32_t* l0 = (volatile tt_l1_ptr uint32_t*)p;
            volatile tt_l1_ptr uint32_t* l1 = (volatile tt_l1_ptr uint32_t*)(p + ts);
            volatile tt_l1_ptr uint32_t* l2 = (volatile tt_l1_ptr uint32_t*)(p + 2 * ts);
            volatile tt_l1_ptr uint32_t* pa = (volatile tt_l1_ptr uint32_t*)qp;
            for (uint32_t j = jlo; j < 1024; ++j) recon_elem::pack_a128<PROBE>(l0, l1, l2, pa, j);
            if (jlo) { while (*done != tag) { invalidate_l1_cache(); } }
            l1_writes_landed(qp + OZ_A128_TILE_BYTES - 4u);
            noc_async_write_tile(g, o0, qp);
        }
        if constexpr (FINAL) {
            noc_async_read_tile((g + off) / Nt, ada, ap);
            noc_async_read_tile((g + off) % Nt, adb, bp);
            if constexpr (ROLE == ROLE_TWO_FINAL) {   // 第 1 パスの固定小数（16 KB）を読む
                noc_async_read_tile(g, aq0, qp);
            }
            noc_async_read_barrier();
            // 差分タイルはここで読み終えているので，リーダは c_20/c_21 を
            // そのまま見ればよい（同じコアの L1）．重複して読ませない．
            if (jlo) { sy[0] = p; *pub = tag; }      // リーダへ limb のアドレスを渡す
            volatile tt_l1_ptr uint32_t* l0 = (volatile tt_l1_ptr uint32_t*)p;
            volatile tt_l1_ptr uint32_t* l1 = (volatile tt_l1_ptr uint32_t*)(p + ts);
            volatile tt_l1_ptr uint32_t* l2 = (volatile tt_l1_ptr uint32_t*)(p + 2 * ts);
            volatile tt_l1_ptr uint32_t* qa4 = (volatile tt_l1_ptr uint32_t*)qp;
            volatile tt_l1_ptr int32_t*  da = (volatile tt_l1_ptr int32_t*)ap;
            volatile tt_l1_ptr int32_t*  db = (volatile tt_l1_ptr int32_t*)bp;
            volatile tt_l1_ptr uint32_t* po = (volatile tt_l1_ptr uint32_t*)fp;
            for (uint32_t j = jlo; j < 1024; ++j)
                recon_elem::assemble_f64<ROLE, PROBE>(l0, l1, l2, nlimb, da, db, qa4, po, j, sh2, Whi, Wlo, qw);
            // リーダの前半を待つ（L0 を捨てながら．リーダは前半を書いてから同じ L1 のセマフォを書くので，
            // 1 つの L1 アクセスポートを順に通り，done が見えた時点で前半も L1 にある）
            if (jlo) { while (*done != tag) { invalidate_l1_cache(); } }
            // 自分の後半が L1 に届いてから NoC に読ませる（common/kernels/l1_order.h）
            l1_writes_landed(fp + OZ_F64_TILE_BYTES - 4u);
            noc_async_write_tile(g + off, of, fp);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, nlimb);
    }
}
