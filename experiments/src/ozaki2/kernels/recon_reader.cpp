// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム II：再構成の入力（法ごとの平面 S 枚）を読み，
// 最終パスでは FP64 の組み立てを，2 パス構成の第 1 パスでは 128 bit 固定小数への変換を半分だけ手伝うデータ移動カーネル
// ---------------------------------------------------------------------------
// 出力タイル g ごとに，平面 s のタイル s*ntot+g を s = 0..S-1 の順で c_0 へ読み込む（oz2_plane_fetch.h．
// 合成段の入力は別のリーダ oz2_plane_reader.cpp が同じ手続きで読む）．
//
// INT8 経路の再構成は符号-絶対値 int32 の C_i を受け取り，絶対値を上位 15 bit と
// 下位 16 bit に割ってから法をとる（FP32 が無誤差に表せる 2^24 を超えるため）．その
// ためのマスク 2 枚（0x7FFF0000, 0x0000FFFF）を mask=1 のとき c_1 へ流す．マスクは
// 定数なので計算カーネルは pop せず，ここでもループの前に 1 回だけ作る．
//
// 最終パスの FP64 組み立てを分担する理由
//   組み立てをライタだけで行うと，再構成段の律速は CRT（SFPU）ではなくライタのスカラの FP64 組み立てになる
//   （法の本数を減らしても再構成段が縮まない．測定は docs/recon_cost_breakdown.md の結果 7 の表と，その後の INT8 経路の注）．
//   このコアは S 枚を読むだけで大半を遊んでいるので，1 タイル 1024
//   要素のうち前半をここで組み立てて，ライタと半分ずつ持つ（要素の手続きはライタと同じ recon_elem.h）．
//   同じコアの L1 を共有するので，作業領域（c_19）はライタのものをそのまま使う．
//   c_16 の出し入れはライタだけが行い（1 つの CB に consumer を 2 つ置くと，
//   読み出しポインタの持ち方が RISC-V ごとに別なので黙って壊れる），こちらは
//   ライタが渡してきた L1 アドレスを見るだけにする．受け渡しは，番号 2 本を
//   セマフォに，limb のアドレスを共有スクラッチ（c_26）の 1 語に置いて行う．
//   番号は 1 から単調に増やすので取りこぼしも二重取りも起きない．
//   番号をセマフォに置くのは，プログラム起動のたびに初期値へ戻るのがセマフォ
//   だけであるためである：CB の中身は起動をまたいで残るので，そこに番号を置くと
//   前回の値（そのコアの担当が 1 タイルなら 1）が残り，次の実行で待たずに
//   古い limb を読んでしまう．
//   差分タイル（c_20/c_21）はライタが読んだものをそのまま見る：両方で読むと
//   1 出力タイルあたり 2 枚ぶん（14080 角で 1.59 GB）の重複になる．
//   計算カーネルを止めないよう，入力の読みは 1 タイル先行させる（c_0 は深さ 2S）．
//   2 パス構成でも同じ形で分担する（パスの役割 ROLE はライタと同じ recon_args.h の ReconRole）．
//   第 1 パスは自分の 3 本を 128 bit 固定小数へ足して作業領域 c_22 の前半へ置き，最終パスはライタが c_22 へ
//   読んだ第 1 パスの固定小数に自分の 3 本を足して畳む（どちらも同じコアの L1 なので，ライタの CB をそのまま見る）．
//   2 パスの最終パスはライタだけで 6 本を組み立てると再構成段の律速になる（docs/recon_cost_breakdown.md の結果 7）．
//
// 実行時引数（並びは ozaki2/kernels/recon_args.h の R_*）:
//             in_addr, num_tiles, start_id, ntot, S, mask,
//             nlimb, Whi, Wlo, qw, sh2, jsplit, sem_pub, sem_done
//   SHARE（コンパイル時引数 CT_FINAL_OR_SHARE．組み立てを分担するか）が 0 のとき（分担しない構成）に読むのは先頭 6 語だけである．

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/f64_tf.h"
#include "recon_args.h"
#include "recon_elem.h"
#include "oz2_plane_fetch.h"
#include "../../common/kernels/l1_order.h"
using namespace recon_args;

void kernel_main() {
    // 引数の並びは ozaki2/kernels/recon_args.h の 1 つの表による
    uint32_t in_addr   = get_arg_val<uint32_t>(R_IN_ADDR);
    uint32_t num       = get_arg_val<uint32_t>(R_NUM);
    uint32_t start     = get_arg_val<uint32_t>(R_START);
    uint32_t ntot      = get_arg_val<uint32_t>(R_NTOT);
    uint32_t S         = get_arg_val<uint32_t>(R_S);
    uint32_t mask      = get_arg_val<uint32_t>(R_MASK);   // 1 ならマスクを流す
    constexpr uint32_t cb_in   = CB_IN;
    constexpr uint32_t cb_mask = CB_MASK;
    constexpr uint32_t cb_out  = CB_LIMB;
    constexpr uint32_t cb_f64  = CB_F64;
    constexpr uint32_t cb_da   = CB_DLT_A;   // ライタが読んだ差分をそのまま見る
    constexpr uint32_t cb_db   = CB_DLT_B;
    constexpr uint32_t cb_sync = CB_SYNC;
    constexpr uint32_t cb_q    = CB_A128;    // ライタの作業領域：128 bit 固定小数 1 タイル（OZ_A128_TILE_BYTES）
    constexpr auto ia = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(ia, in_addr);
    // SHARE（ホストが oz2_split_assembly で決める：1 パス構成の最終パスと 2 パス構成の両パスで分担するとき）だけ
    // 組み立て（2 パス構成の第 1 パスでは 128 bit 固定小数への変換）の前半を持つ
    constexpr uint32_t SHARE = get_compile_time_arg_val(ia.next_compile_time_args_offset() + CT_FINAL_OR_SHARE);
    // パスの役割（recon_args.h の ReconRole．ROLE_SINGLE = 1 パス構成の最終パス（3 本を組み立て），ROLE_TWO_FINAL = 2 パス構成の
    // 最終パス（第 1 パスの固定小数に 3 本を足して組み立て），ROLE_TWO_FIRST = 2 パス構成の第 1 パス（3 本を固定小数へ足すだけ））
    constexpr uint32_t ROLE  = get_compile_time_arg_val(ia.next_compile_time_args_offset() + CT_ROLE);
    static_assert(ROLE == ROLE_SINGLE || ROLE == ROLE_TWO_FINAL || ROLE == ROLE_TWO_FIRST, "パスの役割は ReconRole");
    // PROBE【調査用．答えは壊れる】1 なら assemble_frac_f64 だけ外す．ライタ側と同じ番号で，
    // 同じホスト側のつまみ（OZ_RECON_PROBE）から来る．既定の分担では組み立ての前半を
    // こちらが持つので，両方に入れないと半分が残る（docs/recon_cost_breakdown.md）．
    constexpr uint32_t PROBE = get_compile_time_arg_val(ia.next_compile_time_args_offset() + CT_PROBE);

    // マスクを流さない経路では c_1 自体が作られないので，参照もしない．
    // 定数なのでこのコアの L1 で作る：DRAM に置かず，ホストからも送らない
    // （A・B を転送してから C が戻るまで通信しないという仕様）．
    if (mask) {
        cb_reserve_back(cb_mask, 2);
        volatile tt_l1_ptr uint32_t* mp = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_mask);
        for (uint32_t z = 0; z < 1024; ++z) { mp[z] = 0x7FFF0000u; mp[1024 + z] = 0x0000FFFFu; }
        l1_writes_landed((uint32_t)&mp[2047]);   // L1 に届いてから計算側へ渡す（common/kernels/l1_order.h）
        cb_push_back(cb_mask, 2);
    }
    const uint32_t tb = get_tile_size(cb_in);
    // 1 出力タイルに対し S 枚の平面をまとめて確保し，1 回のバリアで読み終える
    // （c_0 の深さは 2S なので 1 タイル先行できる）．
    auto fetch = [&](uint32_t g) { oz2_fetch_planes(cb_in, in, g, S, ntot, tb); };

    if constexpr (!SHARE) {
        for (uint32_t g = start; g < start + num; ++g) fetch(g);
        return;
    }

    const uint32_t nlimb = get_arg_val<uint32_t>(R_NLIMB);
    const uint32_t Whi = get_arg_val<uint32_t>(R_WHI);
    const uint32_t Wlo = get_arg_val<uint32_t>(R_WLO);
    const int32_t  qw  = (int32_t)get_arg_val<uint32_t>(R_QW);
    const int32_t  sh2 = (int32_t)get_arg_val<uint32_t>(R_SH2);
    // 分担の境目．こちらは入力の読みも持つので，半分より少なめが釣り合う．
    const uint32_t jsplit = get_arg_val<uint32_t>(R_JSPLIT);
    volatile tt_l1_ptr uint32_t* sy = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_sync);
    volatile tt_l1_ptr uint32_t* pub =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(R_SEM_PUB));
    volatile tt_l1_ptr uint32_t* done =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(R_SEM_DONE));
    // ライタと同じ L1 の作業領域．第 1 パス（ROLE_TWO_FIRST）は c_19〜c_21 を作らないので参照しない
    const uint32_t fp = (ROLE == ROLE_TWO_FIRST) ? 0u : get_write_ptr(cb_f64);
    const uint32_t ap = (ROLE == ROLE_TWO_FIRST) ? 0u : get_write_ptr(cb_da);
    const uint32_t bp = (ROLE == ROLE_TWO_FIRST) ? 0u : get_write_ptr(cb_db);
    const uint32_t qp = (ROLE == ROLE_SINGLE) ? 0u : get_write_ptr(cb_q);

    if (num) fetch(start);
    for (uint32_t g = start; g < start + num; ++g) {
        if (g + 1 < start + num) fetch(g + 1);   // 計算カーネルを止めない
        const uint32_t tag = g - start + 1;
        // ライタが limb のアドレスを置くまで待つ．L0 データキャッシュは他の RISC-V の書き込みで
        // 捨てられないので，回すたびに捨てて L1 を読み，抜けた後にも捨ててから limb・差分を読む
        // （tt-isa-documentation の MemoryOrdering.md「L0 data cache」）．
        while (*pub != tag) { invalidate_l1_cache(); }
        invalidate_l1_cache();
        const uint32_t p = sy[0], ts = get_tile_size(cb_out);
        const tt_l1_ptr uint32_t* l0 = (const tt_l1_ptr uint32_t*)p;
        const tt_l1_ptr uint32_t* l1p = (const tt_l1_ptr uint32_t*)(p + ts);
        const tt_l1_ptr uint32_t* l2 = (const tt_l1_ptr uint32_t*)(p + 2 * ts);
        const tt_l1_ptr int32_t*  da = (const tt_l1_ptr int32_t*)ap;
        const tt_l1_ptr int32_t*  db = (const tt_l1_ptr int32_t*)bp;
        tt_l1_ptr uint32_t* po = (tt_l1_ptr uint32_t*)fp;
        tt_l1_ptr uint32_t* qa4 = (tt_l1_ptr uint32_t*)qp;
        for (uint32_t j = 0; j < jsplit; ++j) {  // 前半だけ受け持つ
            if constexpr (ROLE == ROLE_TWO_FIRST) recon_elem::pack_a128<PROBE>(l0, l1p, l2, qa4, j);   // 固定小数へ足して置くだけ
            else recon_elem::assemble_f64<ROLE, PROBE>(l0, l1p, l2, nlimb, da, db, qa4, po, j, sh2, Whi, Wlo, qw);
        }
        *done = tag;                             // 前半が済んだ
    }
}
