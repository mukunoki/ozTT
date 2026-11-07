// SPDX-License-Identifier: Apache-2.0
//
// K スピルの読み戻しを無誤差にした行列積カーネル（全経路で共用）
// ---------------------------------------------------------------------------
// tt-metal 標準の bmm_large_block_zm を，c_24 経由のスピルが無誤差になるよう
// 手直ししたもの．論文の「累積経路の制約と累積幅」の節（\label{ssec:where}）．
//
// 標準カーネルのままでは何が起きるか
//   標準カーネルは copy_tile_to_dst_init_short / mm_init_short を使うが，これらは
//   アンパッカのデータ形式を再設定しない．そのため INT8 用のアンパック設定が
//   残ったまま c_24（Int32）を読み戻すことになり，32 bit の部分和が壊れる．
//
// 本カーネルの対処
//   _with_dt が付いた版（copy_tile_to_dst_init_short_with_dt /
//   mm_init_short_with_dt）を使い，読み戻しの前後でデータ形式を切り替える．
//   ホスト側は c_24 を部分積の形式（INT8 経路は Int32，それ以外は Float32．
//   common/mcast_gemm.hpp の spill_gemm_spec の acc_fmt）にし，unpack_to_dest_mode[c_24] に
//   UnpackToDestFp32 を設定しておく必要がある（どちらも spill_gemm_spec が行う）．
//
//   コンパイル時引数 0-11 番と全体構造は標準カーネルと同じで，データ形式の再設定呼び出し
//   を足してある．以下 **標準との差分** と書いた箇所がそれである（12 番以降と，下の
//   L1 累積・FUSE・PAIRLOOP は本リポジトリで足したもの）．
//
// K ブロック間の累積（パッカの L1 累積）
//   K ブロック間の部分和はパッカが L1（c_24）上で足し込み（pack_reconfig_l1_acc．
//   tt-metal の PACKER_L1_ACC と同じ），DST へ読み戻すのは区間（PAIRLOOP なら 1 対，
//   そうでなければ K 全体）の最後のブロックの 1 回だけにする．
//   毎ブロック・毎サブブロックでアンパッカを c_24 へ再設定して部分和を読み戻し，
//   行列積の設定へ戻す方式では，その往復が行列積そのものと同程度の時間を食う
//   （docs/gemm_opt_results.md）．
//   無誤差性：DST の部分和は FP32（INT8 経路は Int32）で，L1 の加算器も同じ形式で
//   足す．整数の部分和がすべて 2^24 以下（FP32 は 2^24 を無誤差に表せる）なら各加算は丸めを生まない．
//   この条件は DST 内の累積と同じもの（1 区間の K_eff*d^2 <= 2^24．ozaki1_format.hpp の ozaki_max_digit）で，
//   ホストがすでに課している．Int32 の L1 加算は符号-絶対値の整数加算で，途中の和が
//   奇数になる入力・負側・逆符号のブロック和を含めて 2^31 の手前（2,147,092,480）まで
//   無誤差であることを実機で確かめた（scripts/diagnose/gemm_exact.sh の int8 の項目．
//   FP32 の部分積で同じ入力を 2^24 の先まで回すと最近接偶数へ丸まって食い違うので，
//   この検査は両者を見分けられる．docs/gemm_opt_results.md の §4.1）．ホストが INT8 経路に
//   許す部分和の上界は，既定の桁（スキーム I の d=127，スキーム II の法 255 以下で 127）ではその内側にある：
//   スキーム II は K*127^2 < 2^31 で K が 32 の倍数なので最大でちょうどこの値（133120·127^2），スキーム I は
//   無誤差条件が K_eff <= 2^17 で止める（ozaki_block_k）ので 131072·127^2 = 2,114,060,288 まで（さらに内側）．
//   OZ_RADIX で桁を変えたスキーム I の診断の構成は INT32_MAX まで許すので，確かめた範囲を超えうる
//   （基数 64・K=2097120 で 2,147,450,880）．
//
// 出力経路への Int32 累積の融合（FUSE）
//   尾崎スキーム I の遅延再構成では，対角 d の部分和 P_d を DRAM の累積面 S_d へ
//   足し込む．これを別プログラムで行うと 1 更新あたり DRAM を 4 面（P 書き・P 読み・
//   S 読み・S 書き）通るが，本カーネルの最終ブロックへ融合すれば 2 面（S 読み・
//   S 書き）で済む．P が DRAM を経由しなくなるためである．
//   FUSE を立てると，最終ブロックの出力を Int32 へ直し，ライタが先読みした
//   既存の S タイル（c_2）と足してから Int32 でパックする．
//   DST は fp32_dest_acc のとき 32 bit タイルを 4 面しか持たず，出力サブブロックが
//   それを使い切るので，S を同時に載せられない．そこで行列積の結果をいったん
//   FP32 のまま c_3 へ出し，2 タイルずつ載せ直して足す．
//   Int32 の表現は 2 の補数である（typecast<Float32,Int32> と add_int_tile が
//   ともに 2 の補数）ので，最終段の復号は recon_tf_i32in.cpp を SIGNMAG=0 で
//   （＝2 の補数として）動かさなければならない．
//
// 対を 1 プログラムの中で回す（PAIRLOOP）
//   対角 d の (d+1) 対は，スライス連結レイアウトの上では K 方向に連続している
//   （対 a は A スライス a と B スライス d-a の積で，a が 1 進むと A も B も
//   ちょうどスライス 1 本ぶん進む）．したがって連結して発行した 1 本の GEMM が
//   そのまま (d+1) 対を流している．PAIRLOOP なしで連結すると不利なのは，計算カーネルが全対を
//   ひとつの FP32 アキュムレータに積むため無誤差条件が K_eff=(d+1)K_c になり，
//   K を分割せざるをえないからである．
//   PAIRLOOP は対の境界（bpp ブロックごと）で FP32 アキュムレータを畳んで
//   L1 の Int32 面（c_5）へ足し，次の対を 0 から積み直す．こうすると
//   ・無誤差条件は 1 対ぶんの K_eff=K_c だけで済み，K ブロック化が消える
//   ・対と対のあいだの部分和が DRAM を経由しない（S_d の往復が 1 回になる）
//   L1 の累積面は書き込み中に読み出し中の領域と重ならないよう
//   pm*pn + out_subblock の輪にしてある．
//
// 部品：kernel_main は K ブロックとサブブロックのループ（読み戻し・行列積・出力・第 2 パスの順．同期の位置は標準カーネルと
// 同じ）を持ち，サブブロックを出力先へ出す pack_subblock と，第 2 パス second_pass を呼ぶ．どちらも always_inline で，
// 読む状態と変える状態を引数と注記で示す（ループの状態 enable_reload・l1acc_on は kernel_main だけが持つ）．

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/add_int_sfpu.h"
#include "gemm_compute_args.h"   // 引数の並びと CB 番号（ホストの common/mcast_gemm.hpp と共有）
using namespace gemm_args;

// ---- 構成（コンパイル時引数．どれも定数で，下の部品はこれを読むだけ）----
// FUSE を立てると最終ブロックの出力を既存の S へ Int32 で足し込む（上の注記）．F32・I32 は typecast の形式
constexpr uint32_t FUSE = get_compile_time_arg_val(CA_FUSE);
constexpr uint32_t F32 = get_compile_time_arg_val(CA_F32);
constexpr uint32_t I32 = get_compile_time_arg_val(CA_I32);
// PAIRLOOP（コンパイル時引数 CA_PAIRLOOP）：0 なら標準カーネルと同じく全ブロックをひとつの FP32 アキュムレータに積み，
// 実行時引数 CR_BPP は読まない．1 なら CR_BPP の bpp が 1 対あたりの K ブロック数で，bpp は 1 以上で
// K ブロック数を割り切らねばならない（ホストの spill_check_run が止める．bpp=0 は「対ループを切る」
// 指定ではなく，下の剰余がゼロ除算になる不正な値である．対ループを切るのは PAIRLOOP=0）．
constexpr uint32_t PAIRLOOP = get_compile_time_arg_val(CA_PAIRLOOP);
// BLOCKMM：サブブロックを 1 タイルずつ発行する matmul_tiles ではなく，
// (rt,ct) の外積をまとめて渡す matmul_block を使う．アンパッカが
// サブブロック内の再利用を知るので，発行あたりの手数が減りうる．
constexpr uint32_t BLOCKMM = get_compile_time_arg_val(CA_BLOCKMM);

constexpr auto cb_in0 = tt::CBIndex(CB_IN0);
constexpr auto cb_in1 = tt::CBIndex(CB_IN1);
constexpr auto cb_out = tt::CBIndex(CB_OUT);
constexpr auto cb_acc = tt::CBIndex(CB_PARTIAL);
constexpr auto cb_s   = tt::CBIndex(CB_S);      // 融合時：既存の S_d（Int32）
constexpr auto cb_tmp = tt::CBIndex(CB_TMP);    // 融合時：行列積の結果（FP32）
constexpr auto cb_sl  = tt::CBIndex(CB_SL);     // PAIRLOOP：対をまたぐ Int32 累積面

// 出力サブブロックの形（コンパイル時引数から kernel_main が組む．部品は読むだけ）
struct SubblockShape { uint32_t h, w, ntiles, in0_block_w; };

// アンパッカ（と数値ユニット）を行列積の設定へ戻す．old_cb は直前にアンパッカが読んでいた CB（_with_dt の旧の形式）
ALWI void mm_restore_with_dt(const SubblockShape& sb, uint32_t old_cb) {
    if constexpr (BLOCKMM)
        mm_block_init_short_with_dt(cb_in0, cb_in1, old_cb, 0, sb.w, sb.h, sb.in0_block_w);
    else
        mm_init_short_with_dt(cb_in0, cb_in1, old_cb);
}

// DST のサブブロックを出力先へ出す（区間の最後のブロックは c_16・c_3・c_5，それ以外は c_24）．
// 読む状態：pair_end・first_pair・last_out・segj・block（このブロックの区間の中での位置），first_sb（このブロックの最初の
// サブブロックか）．変える状態：out_num_tiles_to_wait（最初のブロックで c_16 を予約した枚数），各 CB の受け渡しと
// パッカの形式．最初の対の typecast では数値ユニットの設定を行列積へ戻す（アンパッカは触らない）
ALWI void pack_subblock(const SubblockShape& sb, bool pair_end, bool first_pair, bool last_out, uint32_t segj,
                        uint32_t block, bool first_sb, uint32_t& out_num_tiles_to_wait) {
    const uint32_t out_subblock_num_tiles = sb.ntiles;
    if (pair_end && PAIRLOOP && first_pair) {
        // 最初の対：足す相手が無いので，その場で Int32 へ直して置く
        // （c_3 への退避が要らない）．対角 0 は対が 1 つしか無いので，
        // これが最後の対でもあり，そのときは累積面ではなく出力へ出す．
        // このブロックのサブブロックはすべて同じ cb_dst へ出し，間にほかの
        // pack は無い（第 2 パスは最初の対では通らない）ので，形式の再設定は
        // 最初のサブブロックだけでよい．
        const auto cb_dst = last_out ? cb_out : cb_sl;
        typecast_tile_init<F32, I32>();
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            typecast_tile<F32, I32>(i);
        cb_reserve_back(cb_dst, out_subblock_num_tiles);
        if (first_sb) pack_reconfig_data_format(cb_dst);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            pack_tile(i, cb_dst);
        cb_push_back(cb_dst, out_subblock_num_tiles);
        // typecast で数値ユニットの設定が変わっているので，
        // 次の対の行列積に入る前に戻す（アンパッカは触っていない）．
        if constexpr (BLOCKMM)
            mm_block_init_short(cb_in0, cb_in1, 0, sb.w, sb.h, sb.in0_block_w);
        else
            mm_init_short(cb_in0, cb_in1);
    } else if (pair_end && (PAIRLOOP || FUSE)) {
        // 行列積の結果をいったん FP32 のまま c_3 へ逃がす．
        // DST を解放してから載せ直さないと足す相手を置く面が取れない．
        // 形式の再設定はサブブロックごとに要る（間の第 2 パスが Int32 の c_5・c_16 へ出す）．
        cb_reserve_back(cb_tmp, out_subblock_num_tiles);
        pack_reconfig_data_format(cb_tmp);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, cb_tmp);
        }
        cb_push_back(cb_tmp, out_subblock_num_tiles);
    } else if (last_out) {
        cb_reserve_back(cb_out, out_subblock_num_tiles);
        // **標準との差分**：出力 CB に合わせてパッカを再設定．最後のブロックの
        // サブブロックはすべて c_16 へ出し，間にほかの pack は無いので最初の 1 回でよい
        if (first_sb) pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, cb_out);
        }
        cb_push_back(cb_out, out_subblock_num_tiles);
    } else {
        if (block == 0) {
            cb_reserve_back(cb_out, out_num_tiles_to_wait);
            out_num_tiles_to_wait += out_subblock_num_tiles;
        }
        cb_reserve_back(cb_acc, out_subblock_num_tiles);
        // **標準との差分**：c_24（Int32）に合わせてパッカを再設定
        // c_24 以外へ出すのは区間の最後だけなので，形式の再設定は
        // 区間の最初のブロックの最初のサブブロックだけでよい（再設定は STALL を伴う）
        if (segj == 0 && first_sb)
            pack_reconfig_data_format(cb_acc);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, cb_acc);
        }
        cb_push_back(cb_acc, out_subblock_num_tiles);
    }
}

// 第 2 パス（区間の最後のブロックで PAIRLOOP か FUSE のとき．最初の対を除く）：c_3 の FP32 を Int32 へ直し，足すべき相手と
// 足して出力先へ出す．
//   PAIRLOOP の途中の対 : L1 の累積面 c_5（出力先も c_5）
//   PAIRLOOP の最後の対 : c_5 と足して c_16 へ
//   融合（FUSE）        : ライタが先読みした DRAM の S（c_2）と足して c_16 へ（acc が 0 なら足さずに初期化）
// 足す相手は高々 1 つである：融合と対ループはホスト側で排他にしてある（common/mcast_gemm.hpp の spill_gemm_spec の
// TT_FATAL）．DST は fp32_dest_acc のとき 32 bit タイルを 4 面しか持たず，結果 n 面＋相手 n 面なので 2 タイルずつ回す．
// 読む状態：last_out（最後の対か）・acc．変える状態：c_3・c_5・c_2・c_16 の受け渡しと，アンパッカ・パッカ・数値ユニットの
// 設定（最後にアンパッカを行列積の設定へ戻す）．DST はこの中で 2 タイルごとに取って返す
ALWI void second_pass(const SubblockShape& sb, bool last_out, uint32_t acc) {
    const uint32_t out_subblock_num_tiles = sb.ntiles;
    const bool use_sl  = PAIRLOOP;                 // c_5 と足すか
    const bool to_out  = !PAIRLOOP || last_out;    // 出力先が c_16 か
    const bool use_dram = FUSE && acc && to_out;   // c_2 と足すか
    const bool add_any = use_sl || use_dram;
    const auto cb_dst = to_out ? cb_out : cb_sl;
    const auto cb_add = use_sl ? cb_sl : cb_s;     // 足す相手の CB
    cb_wait_front(cb_tmp, out_subblock_num_tiles);
    if (add_any) cb_wait_front(cb_add, out_subblock_num_tiles);
    cb_reserve_back(cb_dst, out_subblock_num_tiles);
    // パッカの出力先はこのパスのあいだ cb_dst だけなので，形式の再設定は 1 回でよい
    // （2 タイル刻みのループの中に置くと，4 タイルのサブブロックで 2 回再設定することになる）
    pack_reconfig_data_format(cb_dst);
    // アンパッカの形式の再設定は「直前に設定していた CB」から切り替える
    // （_with_dt は旧と新の形式が同じなら何もしない）．ループの 1 回目の直前は行列積の
    // 設定（SrcA は cb_in1），2 回目以降は前の回の最後に読んだ cb_add か cb_tmp である．
    // 毎回 cb_in1 から切り替えると，cb_in1 の形式が c_3 と同じ構成では 2 回目の
    // 再設定が飛ばされ，Int32（c_5・c_2）の設定のまま c_3 の FP32 を読んでしまう．
    // I-FP32（TF32）では GEMM の入力 cb_in1 も c_3 も Float32 なので，これは実際に起きる
    // （PAIRLOOP が既定の経路）．prev_cb で直前に読んだ CB を追うのは省けない．
    uint32_t prev_cb = cb_in1;
    for (uint32_t i = 0; i < out_subblock_num_tiles; i += 2) {
        uint32_t n = out_subblock_num_tiles - i;
        if (n > 2) n = 2;
        acquire_dst();
        copy_tile_to_dst_init_short_with_dt(prev_cb, cb_tmp);
        for (uint32_t j = 0; j < n; j++) copy_tile(cb_tmp, i + j, j);
        prev_cb = cb_tmp;
        typecast_tile_init<F32, I32>();
        for (uint32_t j = 0; j < n; j++) typecast_tile<F32, I32>(j);
        if (add_any) {   // 結果は 0..n-1，相手は n..2n-1 に置く
            copy_tile_to_dst_init_short_with_dt(cb_tmp, cb_add);
            for (uint32_t j = 0; j < n; j++) copy_tile(cb_add, i + j, n + j);
            prev_cb = cb_add;
            add_int_tile_init();
            for (uint32_t j = 0; j < n; j++)
                add_int_tile<DataFormat::Int32>(j, n + j, j);
        }
        for (uint32_t j = 0; j < n; j++) pack_tile(j, cb_dst);
        release_dst();
    }
    cb_push_back(cb_dst, out_subblock_num_tiles);
    cb_pop_front(cb_tmp, out_subblock_num_tiles);
    // c_5 は読んだぶんを捨てる（輪に out_subblock ぶんの余裕を
    // 持たせてあるので，reserve が読み出し中の領域と重ならない）
    if (add_any) cb_pop_front(cb_add, out_subblock_num_tiles);
    // アンパッカを行列積の設定へ戻す（次のサブブロック・次の対のため）
    mm_restore_with_dt(sb, add_any ? cb_add : cb_tmp);
}

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(CA_IN0_BLOCK_W);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(CA_IN0_NUM_SUBBLOCKS);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(CA_IN0_BLOCK_NUM_TILES);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(CA_IN0_SUBBLOCK_NUM_TILES);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(CA_IN1_NUM_SUBBLOCKS);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(CA_IN1_BLOCK_NUM_TILES);
    uint32_t in1_per_core_w = get_compile_time_arg_val(CA_IN1_PER_CORE_W);
    // num_blocks だけはコンパイル時引数 CA_NUM_BLOCKS（7 番）を読まず，実行時引数 CR_NUM_BLOCKS にしてある：
    // 尾崎スキームの GEMM 段は対角 d ごとに K の長さ (d+1)*K が変わるが，コンパイル済みプログラムを
    // 1 つ使い回せるようにするためである（7 番は標準カーネル・bmm_l1acc.cpp と並びを共有するために残る．
    // gemm_compute_args.h の冒頭）．
    uint32_t num_blocks = get_arg_val<uint32_t>(CR_NUM_BLOCKS);
    uint32_t out_subblock_h = get_compile_time_arg_val(CA_OUT_SUBBLOCK_H);
    uint32_t out_subblock_w = get_compile_time_arg_val(CA_OUT_SUBBLOCK_W);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(CA_OUT_SUBBLOCK_NUM_TILES);
    uint32_t batch = get_compile_time_arg_val(CA_BATCH);
    const SubblockShape sb{out_subblock_h, out_subblock_w, out_subblock_num_tiles, in0_block_w};
    // acc は実行時引数 CR_ACC：その面への最初の書き込みなら 0（足さずに初期化する．FUSE のときだけ読む）．
    const uint32_t acc = FUSE ? get_arg_val<uint32_t>(CR_ACC) : 0u;
    // PAIRLOOP でないときも 1 を入れておく：0 だと下の剰余が定数畳み込みで
    // ゼロ除算と判定され，コンパイルが通らない（分岐は死んでいても評価される）．
    // PAIRLOOP のときホストが渡すのは必ず 1 以上である．
    const uint32_t bpp = PAIRLOOP ? get_arg_val<uint32_t>(CR_BPP) : 1u;
    // 1 区間（アキュムレータを 0 から積み直す単位）のブロック数
    const uint32_t seglen = PAIRLOOP ? bpp : num_blocks;
    const uint32_t out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    bool l1acc_on = false;   // パッカの L1 累積が今立っているか（再設定は STALL を伴うので状態を持つ）

    if constexpr (BLOCKMM) mm_block_init(cb_in0, cb_in1, cb_out, 0,
                                         out_subblock_w, out_subblock_h, in0_block_w);
    else                    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t b = 0; b < batch; b++) {
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            // PAIRLOOP では対の境界ごとにアキュムレータを畳む．
            // pair_end はその境界，last_out はプログラム全体の最後の対の境界．
            const bool last_out = block == (num_blocks - 1);
            const bool pair_end = PAIRLOOP ? ((block + 1) % bpp == 0) : last_out;
            const bool first_pair = PAIRLOOP ? (block < bpp) : true;
            const uint32_t segj = PAIRLOOP ? (block % bpp) : block;   // 区間内の位置
            // 区間の最後のブロックは c_24 以外（c_16・c_3・c_5）へ出すので L1 累積を切る．
            // 区間内の 2 番目以降のブロックは c_24 上の和へ足し込む．
            const bool want_l1acc = !pair_end && segj > 0;
            if (want_l1acc != l1acc_on) { pack_reconfig_l1_acc(want_l1acc ? 1 : 0); l1acc_on = want_l1acc; }

            cb_wait_front(cb_in0, in0_block_num_tiles);
            cb_wait_front(cb_in1, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    // acquire_dst/release_dst は tt-metal v0.73.1 で [[deprecated]]（reg_api.h）だが，これを使っている．
                    // tile_regs_* へ置き換えるには pack の位置（commit と wait の間）の見直しが要り，論文の測定に
                    // 使った GEMM 段の命令列が変わるので，置き換えていない．
                    acquire_dst();

                    if (enable_reload) {
                        // **標準との差分**：c_24（Int32 または Float32）を読むためアンパッカを再設定
                        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_acc);
                        cb_wait_front(cb_acc, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(cb_acc, i, i);
                        }
                        cb_pop_front(cb_acc, out_subblock_num_tiles);
                        // **標準との差分**：アンパッカを入力形式（INT8 経路なら Int8）へ戻す
                        mm_restore_with_dt(sb, cb_acc);
                    }

                    if constexpr (BLOCKMM) {
                        // 内側の K タイルごとに (out_subblock_h x out_subblock_w) の
                        // 外積をまとめて DST へ足し込む．A は 1 タイルずつ右へ，
                        // B は 1 行ぶん（in1_per_core_w）下へ進む．
                        uint32_t in0_index = (uint32_t)in0_index_subblock_offset;
                        uint32_t in1_index = (uint32_t)in1_index_subblock_offset;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                            matmul_block(cb_in0, cb_in1, in0_index, in1_index, 0, 0,
                                         out_subblock_w, out_subblock_h, in0_block_w);
                            in0_index++;
                            in1_index += in1_per_core_w;
                        }
                    } else {
                        int dst_index = 0;
                        int in0_index_h_offset = 0;
                        for (uint32_t h = 0; h < out_subblock_h; h++) {
                            for (uint32_t w = 0; w < out_subblock_w; w++) {
                                int in1_index_inner_dim_offset = 0;
                                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                    int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                    int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                    matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index);
                                    in1_index_inner_dim_offset += in1_per_core_w;
                                }
                                dst_index++;
                            }
                            in0_index_h_offset += in0_block_w;
                        }
                    }

                    // このブロックの最初のサブブロックか（パッカの形式の再設定は STALL を伴うので，
                    // 同じ出力先へ出し続けるあいだは 1 回にする）
                    const bool first_sb = in0_subblock == 0 && in1_subblock == 0;
                    pack_subblock(sb, pair_end, first_pair, last_out, segj, block, first_sb, out_num_tiles_to_wait);

                    release_dst();

                    // 最初の対の終わり（pair_end && PAIRLOOP && first_pair）は上で c_5 へ置いたので第 2 パスは無く，
                    // アンパッカは行列積のままでよい．
                    if (pair_end && (PAIRLOOP || FUSE) && !(PAIRLOOP && first_pair)) second_pass(sb, last_out, acc);
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            // c_24 は pm*pn 枚ちょうどの輪なので，書いたぶんを捨てれば次のブロックの
            // pack は同じ L1 位置（＝走行中の和）へ足し込まれる．区間の最後から 2 番目の
            // ブロックの和だけは残し，最後のブロックでサブブロックごとに読み戻す．
            // 区間が切り替わる（pair_end）とアキュムレータを 0 から積み直す．
            // 捨てるのはブロックぶん（pm*pn 枚＝c_24 の容量）をまとめて 1 回にする．
            // サブブロックごとに wait/pop すると（尾崎 II の 11x10・1x2 なら 1 ブロック 55 回），
            // 14080^3 の GEMM 段で尾崎 II が 1.7〜2.9% 遅くなる（尾崎 I は差が雑音の内．
            // docs/gemm_opt_results.md の追記の性能の表）．待つ枚数はこのブロックで pack した
            // pm*pn 枚そのもので，c_24 の容量を超えない．
            if (pair_end) {
                enable_reload = false;
            } else if (segj + 2 < seglen) {
                cb_wait_front(cb_acc, out_block_num_tiles);
                cb_pop_front(cb_acc, out_block_num_tiles);
            } else {
                enable_reload = true;
            }

            cb_pop_front(cb_in0, in0_block_num_tiles);
            cb_pop_front(cb_in1, in1_block_num_tiles);
        }
    }
    if (l1acc_on) pack_reconfig_l1_acc(0);
}
