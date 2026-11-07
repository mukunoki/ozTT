// SPDX-License-Identifier: Apache-2.0
//
// マルチキャスト行列積のデータ移動カーネル（オペランドごとに RISC を分け，両端の 2 コアから配る）
// ---------------------------------------------------------------------------
// tt-metal の matmul_common にあるマルチキャスト版リーダ 4 種
// （reader_bmm_tile_layout_in0_{sender,receiver}_in1_{sender,receiver}.cpp）は，
// 1 つの RISC が A のブロックを読んで配り終えてから B のブロックを読んで配る．
// その方式では送信コアが各ブロック周期の 5〜6 割を DRAM 読みの完了待ちに費やし，A と B の
// 読みが直列になるぶん周期が延びる（docs/gemm_opt_results.md）．
//
// 1 つのソースから 2 つのカーネルを作る：
//   OPERAND=0 : A（c_0）だけを扱う．RISCV_1 に置き，NOC 1 を使う．
//   OPERAND=1 : B（c_1）だけを扱い，K ループの後に出力を書く（WRITER）．
//               RISCV_0 に置き，NOC 0 を使う．
// こうすると A と B の DRAM 読み・マルチキャストが 2 つの RISC で並行に進む．
// 出力の書き出しは全ブロックの後にしか起きないので，B の RISC に同居させても
// B の供給とは時間が重ならない（バッチは扱わない：tt-metal のリーダ・ライタのバッチの引数とループは除いてある）．
//
// 1 ブロックのタイルは前半・後半に分け，行（列）の両端の 2 コアがそれぞれ半分を DRAM から
// 読んで，自分以外の全コアへ配る．送信コアあたりの DRAM 読みとマルチキャストの量が半分になり，
// 2 コアが並行に進む．端のコアは送信側であると同時に，反対の端が配る半分の受信側でもある．
// 受信側は 2 つのフラグ（前半・後半）を待つ．
//   ROLE=1 : 前半の送信側（A は行の左端，B は列の上端）
//   ROLE=3 : 後半の送信側（A は行の右端，B は列の下端）
//   ROLE=2 : 受けるだけのコア（その間．行／列が 3 コア以上のときだけ居る）
// 1 ブロックが 1 タイルのときは後半が 0 枚になる．そのときホストは後半の送信側を置かず（反対の端も
// ROLE=2），前半の送信側が全部を配る．後半のフラグの受け渡しもしない（half2）．
//   WRITER=0 : 書かない（A 側）
//   WRITER=1 : 標準のライタ（tt-metal の writer_bmm_tile_layout.cpp と同じ書き順．引数はバッチの 2 個を除いた 11 個．
//              mcast_args.h の W_*）
//   WRITER=2 : 融合ライタ．書き出す前に同じ位置の既存の S を c_2 へ先読みする（尾崎スキーム I の
//              遅延再構成で，計算カーネル bmm_int32_spill.cpp の FUSE が Int32 で足し込む相手）．
//   NODRAM・NOMCAST・NOWRITE : 診断用の探針（答えが壊れる．下の各 define）．既定の 0 が本番の構成で，ホストは
//              McastGemmSpec の probe（common/gemm_probe.hpp）から渡す．
//
// 実行時引数はリーダの 33 個（0..32．tt-metal のリーダの並びからバッチの 4 個とブロックの高さ 2 個を除いたもの．
// mcast_reader_args）の後ろに，後半の送信用の 7 個（33..39：後半の送信コアの座標，その送信先の矩形，後半のフラグの
// セマフォ）が続き，WRITER のときはさらに 40 番からライタの引数（mcast_writer_args）が続く．番号はホストと共有する
// mcast_args.h の表（RA_*・X_*・W_*）で読む．
// リーダの並びは tt-metal のリーダから写したものだが，その並びに依存する箇所は無い：付け替え（scheme I の gemm_run
// など）は mcast_args.h の表の名前（RA_A_START・RA_B_START・RA_NUM_BLOCKS）で行う．
// コンパイル時引数は A・B・C の TensorAccessorArgs をこの順に連結したもの．
//
// マルチキャストの矩形はホストが (hi, lo) の 2 隅で渡す（hi は座標の大きい側．tt-metal のリーダの
// 引数名では start）．NOC 0 は座標の増える向きに (lo, hi)，NOC 1 は逆向きなので (hi, lo) の順で
// 宛先を組む（MCAST_ADDR）．

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "mcast_args.h"
#include "gemm_compute_args.h"   // GEMM 段の CB 番号（計算カーネルとホストの common/mcast_gemm.hpp と共有）
#include "l1_order.h"   // l1_writes_landed（下のフラグ・セマフォの store の後）

#ifndef OPERAND
#error "OPERAND (0=A, 1=B) must be defined"
#endif
#ifndef ROLE
#error "ROLE (1=first-half sender, 2=receiver, 3=second-half sender) must be defined"
#endif
#ifndef WRITER
#define WRITER 0
#endif
// NODRAM=1：診断用．送信側は最初の 2 ブロックだけ DRAM から読み，以降は CB に残った
// 古いデータをそのまま配る（答えは壊れる）．DRAM 読みを除いたときの GEMM 段の時間を見る．
#ifndef NODRAM
#define NODRAM 0
#endif
// NOMCAST=1：診断用．A/B の供給（DRAM 読み・マルチキャスト・同期）をせず，入力 CB の reserve/push だけを行う
// （計算カーネルは CB に残った値で回るので答えは壊れる）．出力の書き出し（下の write_output の DRAM への書き込みと
// その完了待ち．融合ライタ WRITER=2 なら既存の S の先読みも）は残るので，測れるのは「計算パイプライン＋出力の転送」
// の時間であって計算だけの時間ではない（名前は「A/B のマルチキャストなし」の意味．NoC を一切使わないのではない）．
#ifndef NOMCAST
#define NOMCAST 0
#endif
// NOWRITE=1：診断用．出力の DRAM への書き込み（融合ライタなら既存の S の先読みも）をせず，c_16 を受け取って捨てる
// だけにする（出力は書かれないので答えは壊れる）．出力の書き出しを除いたときの GEMM 段の時間を見る．
#ifndef NOWRITE
#define NOWRITE 0
#endif

// 宛先の矩形 (hi 隅, lo 隅) から NOC の向きに合わせたマルチキャストの宛先を作る
#define MCAST_ADDR(hx, hy, lx, ly, a) \
    (noc_index == 0 ? get_noc_multicast_addr(lx, ly, hx, hy, a) : get_noc_multicast_addr(hx, hy, lx, ly, a))

namespace ma = mcast_args;   // 区画どうしの関係の検査は mcast_args.h の static_assert が受け持つ

#if WRITER
// 出力を書く．引数 W0 からライタの引数が並ぶ．
template <uint32_t FUSED, typename Acc>
static inline void write_output(const Acc& s, uint32_t start_tile) {
    constexpr uint32_t w0 = ma::W0;
    const uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(w0 + ma::W_STRIDE_W);
    const uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(w0 + ma::W_STRIDE_H);
    const uint32_t next_sb_w = get_arg_val<uint32_t>(w0 + ma::W_NEXT_SB_W);
    const uint32_t next_sb_h = get_arg_val<uint32_t>(w0 + ma::W_NEXT_SB_H);
    const uint32_t out_subblock_w = get_arg_val<uint32_t>(w0 + ma::W_SB_W);
    const uint32_t out_subblock_h = get_arg_val<uint32_t>(w0 + ma::W_SB_H);
    const uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(w0 + ma::W_SB_TILES);
    const uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(w0 + ma::W_NSB_W);
    const uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(w0 + ma::W_NSB_H);
    // 融合ライタだけが W_ACC の acc を読む（ホストはどのライタにも acc まで含めた W_COUNT 個（mcast_args.h）を
    // 渡し，McastGemm::set_args がその個数を検査する．標準のライタは W_ACC を読まない）
    const uint32_t acc = FUSED ? get_arg_val<uint32_t>(w0 + ma::W_ACC) : 0u;
    constexpr uint32_t cb_out = gemm_args::CB_OUT;
    constexpr uint32_t cb_s = gemm_args::CB_S;   // 融合ライタ：既存の S の先読み先
    const uint32_t tsz = get_tile_size(cb_out);

    uint32_t sbh_start = start_tile;
    for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
        uint32_t sbw_start = sbh_start;
        for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
            if (FUSED && acc && !NOWRITE) {
                // 既存の S を先に読む（計算カーネルが Int32 で足して c_16 へ出す）
                cb_reserve_back(cb_s, out_subblock_tile_count);
                uint32_t l1w = get_write_ptr(cb_s);
                uint32_t row = sbw_start;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t id = row;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_read_page(id, s, l1w);
                        l1w += tsz;
                        id += out_tensor_stride_w;
                    }
                    row += out_tensor_stride_h;
                }
                noc_async_read_barrier();
                cb_push_back(cb_s, out_subblock_tile_count);
            }
            if (FUSED && acc && NOWRITE) {   // 計算カーネルが待つ S の CB を中身なしで渡す
                cb_reserve_back(cb_s, out_subblock_tile_count);
                cb_push_back(cb_s, out_subblock_tile_count);
            }
            cb_wait_front(cb_out, out_subblock_tile_count);
            if (!NOWRITE) {
                uint32_t l1r = get_read_ptr(cb_out);
                uint32_t row = sbw_start;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t id = row;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        noc_async_write_page(id, s, l1r);
                        l1r += tsz;
                        id += out_tensor_stride_w;
                    }
                    row += out_tensor_stride_h;
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, out_subblock_tile_count);
            sbw_start += next_sb_w;
        }
        sbh_start += next_sb_h;
    }
}
#endif

void kernel_main() {
    // A は引数 0..6，B は 7..13 が同じ並び（アドレス・開始タイル・刻み・ブロックの形）
    constexpr uint32_t T = OPERAND == 0 ? ma::RA_A : ma::RA_B;
    const uint32_t tensor_addr = get_arg_val<uint32_t>(T + ma::OP_ADDR);
    const uint32_t start_tile = get_arg_val<uint32_t>(T + ma::OP_START);
    const uint32_t stride_w = get_arg_val<uint32_t>(T + ma::OP_STRIDE_W);
    const uint32_t stride_h = get_arg_val<uint32_t>(T + ma::OP_STRIDE_H);
    const uint32_t next_block_stride = get_arg_val<uint32_t>(T + ma::OP_NEXT_BLOCK);
    const uint32_t block_w = get_arg_val<uint32_t>(T + ma::OP_BLOCK_W);
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(T + ma::OP_BLOCK_TILES);
    const uint32_t num_blocks = get_arg_val<uint32_t>(ma::RA_NUM_BLOCKS);
    // マルチキャストの引数：A は 15..23，B は 24..32．矩形は前半の送信先（ROLE=1 以外の全コア）
    constexpr uint32_t Mc = OPERAND == 0 ? ma::RA_MC_A : ma::RA_MC_B;
    const uint32_t mc_hx = get_arg_val<uint32_t>(Mc + ma::MC_HI_X);
    const uint32_t mc_hy = get_arg_val<uint32_t>(Mc + ma::MC_HI_Y);
    const uint32_t mc_lx = get_arg_val<uint32_t>(Mc + ma::MC_LO_X);
    const uint32_t mc_ly = get_arg_val<uint32_t>(Mc + ma::MC_LO_Y);
    const uint32_t mc_num_dests = get_arg_val<uint32_t>(Mc + ma::MC_NUM_DESTS);
    const uint32_t sender_x = get_arg_val<uint32_t>(Mc + ma::MC_SENDER_X);
    const uint32_t sender_y = get_arg_val<uint32_t>(Mc + ma::MC_SENDER_Y);
    const uint32_t sender_sem_addr = get_semaphore(get_arg_val<uint32_t>(Mc + ma::MC_SENDER_SEM));
    const uint32_t receiver_sem_addr = get_semaphore(get_arg_val<uint32_t>(Mc + ma::MC_FLAG_SEM));
    // 後半の送信コア（行／列の反対の端）とその送信先の矩形（hi 隅・lo 隅），後半のフラグ
    const uint32_t s1_x = get_arg_val<uint32_t>(ma::X_S1_X), s1_y = get_arg_val<uint32_t>(ma::X_S1_Y);
    const uint32_t r1_hx = get_arg_val<uint32_t>(ma::X_HI_X), r1_hy = get_arg_val<uint32_t>(ma::X_HI_Y);
    const uint32_t r1_lx = get_arg_val<uint32_t>(ma::X_LO_X), r1_ly = get_arg_val<uint32_t>(ma::X_LO_Y);
    const uint32_t flag1_addr = get_semaphore(get_arg_val<uint32_t>(ma::X_FLAG_SEM));

    constexpr uint32_t cb = OPERAND == 0 ? gemm_args::CB_IN0 : gemm_args::CB_IN1;
    constexpr bool IS_S0 = ROLE == 1, IS_S1 = ROLE == 3;
    // 前半のフラグは receiver_sem，後半のフラグは flag1．送信コアは自分の空き待ちに sender_sem を使う
    volatile tt_l1_ptr uint32_t* rsem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    volatile tt_l1_ptr uint32_t* flag1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag1_addr);
    volatile tt_l1_ptr uint32_t* ssem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
#if OPERAND == 0
    const auto src = TensorAccessor(a_args, tensor_addr);
#else
    const auto src = TensorAccessor(b_args, tensor_addr);
#endif
#if WRITER
    const uint32_t out_addr = get_arg_val<uint32_t>(ma::W0 + ma::W_ADDR);
    const uint32_t out_start = get_arg_val<uint32_t>(ma::W0 + ma::W_START);
    const auto dst = TensorAccessor(c_args, out_addr);
#endif

    const uint32_t tile_bytes = get_tile_size(cb);
    const uint32_t n0 = (block_num_tiles + 1) / 2;   // 前半のタイル数（ROLE=1 が受け持つ）
    const uint64_t s0_ready_noc = get_noc_addr(sender_x, sender_y, sender_sem_addr);
    const uint64_t s1_ready_noc = get_noc_addr(s1_x, s1_y, sender_sem_addr);
    // 送信コアが配るフラグの元（自分の L1 に VALID を置いておく）．下のマルチキャストは NoC でここを読むが，
    // L1 への store とその後の NoC の発行の順序は保証されない（l1_order.h）ので，store が L1 に届いてから進む
    if constexpr (IS_S0) { *rsem = VALID; l1_writes_landed(receiver_sem_addr); }
    if constexpr (IS_S1) { *flag1 = VALID; l1_writes_landed(flag1_addr); }
    // 送信コアの受け持ち：送信先の矩形・配るフラグ・タイルの範囲 [t_beg, t_end)
    const uint32_t my_hx = IS_S1 ? r1_hx : mc_hx, my_hy = IS_S1 ? r1_hy : mc_hy;
    const uint32_t my_lx = IS_S1 ? r1_lx : mc_lx, my_ly = IS_S1 ? r1_ly : mc_ly;
    const uint32_t my_flag = IS_S1 ? flag1_addr : receiver_sem_addr;
    const uint32_t t_beg = IS_S1 ? n0 : 0u, t_end = IS_S1 ? block_num_tiles : n0;
    // 後半があるか（1 タイルのブロックでは無い．そのとき後半の送信コアは居らず，フラグの受け渡しも省く．
    // ホストの build_mcast_gemm が後半の送信コアを置くかどうかと同じ条件）
    const bool half2 = block_num_tiles > n0;

    uint32_t cur = start_tile;
    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb, block_num_tiles);
#if !NOMCAST
        const uint32_t slot = get_write_ptr(cb);
        // 受け取る側のフラグを倒してから，自分に書き込む送信コアへ空きを知らせる．INVALID は L1 への store，
        // 知らせは NoC の発行で，この順序は保証されない（l1_order.h）．知らせが先に出て，送信コアの VALID の後に
        // INVALID が届くと下の待ちが終わらないので，INVALID が L1 に届いてから知らせる（ブロックごとに数十サイクル）
        if constexpr (!IS_S0) {
            noc_semaphore_set(rsem, INVALID); l1_writes_landed(receiver_sem_addr); noc_semaphore_inc(s0_ready_noc, 1); }
        if (!IS_S1 && half2) {
            noc_semaphore_set(flag1, INVALID); l1_writes_landed(flag1_addr); noc_semaphore_inc(s1_ready_noc, 1); }
        if (IS_S0 || (IS_S1 && half2)) {
            // 自分の受け持ち（前半か後半）のタイルを読む．タイル t はブロック内の
            // (h, w) = (t / block_w, t % block_w) で，CB の中でも t 番目に置かれる．
            uint32_t h = t_beg / block_w, w = t_beg % block_w;
            uint32_t l1 = slot + t_beg * tile_bytes;
            for (uint32_t t = t_beg; t < t_end; t++) {
                if (!NODRAM || block < 2) noc_async_read_page(cur + h * stride_h + w * stride_w, src, l1);
                l1 += tile_bytes;
                if (++w == block_w) { w = 0; h++; }
            }
            noc_async_read_barrier();
            // 自分以外の全コアが空きを知らせてくるのを待ってから配る
            noc_semaphore_wait(ssem, mc_num_dests);
            // 0 に戻す store が L1 に届いてから配る（同じく順序は保証されず，届く前に配った先から次のブロックの
            // 知らせが来ると，それを 0 で消してしまう．l1_order.h）
            noc_semaphore_set(ssem, 0);
            l1_writes_landed(sender_sem_addr);
            const uint32_t a = slot + t_beg * tile_bytes;
            noc_async_write_multicast(a, MCAST_ADDR(my_hx, my_hy, my_lx, my_ly, a),
                                      (t_end - t_beg) * tile_bytes, mc_num_dests);
#ifdef ARCH_BLACKHOLE
            // Blackhole ではコマンドが別々の FIFO に入り順序が保証されないので，
            // フラグより先にデータを送り出しておく（tt-metal のリーダと同じ）
            noc_async_writes_flushed();
#endif
            noc_semaphore_set_multicast(my_flag, MCAST_ADDR(my_hx, my_hy, my_lx, my_ly, my_flag), mc_num_dests);
        }
        cur += next_block_stride;
        if constexpr (!IS_S0) noc_semaphore_wait(rsem, VALID);
        if (!IS_S1 && half2) noc_semaphore_wait(flag1, VALID);
#endif
        cb_push_back(cb, block_num_tiles);
    }
#if WRITER
    write_output<WRITER == 2 ? 1u : 0u>(dst, out_start);
#endif
#if ROLE == 1 || ROLE == 3
    // 送ったマルチキャストが出切るのを待つ（カーネル終了後に L1 を再利用されないように）
    noc_async_write_barrier();
#endif
}
