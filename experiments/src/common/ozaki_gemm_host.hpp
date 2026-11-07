// SPDX-License-Identifier: Apache-2.0
#pragma once
// GEMM を組む全ドライバで共通のホスト側ヘルパ
// ---------------------------------------------------------------------------
// 尾崎の両スキーム（scheme I は ozaki1_common.hpp 経由，scheme II は ozaki2_main.cpp）と基準実装
// （gemm_bf16・gemm_bf16ext・gemm_int8），GEMM 段の検査 gemm_exact_check が使う．置いてあるのは，タイル内の座標変換（nf），
// 符号-絶対値の int8/int32 符号化・復号，計算グリッドへのタイルの配り方（GridSpan），CB と DRAM バッファの作成
// （make_cb・mesh_dram_buffer），DRAM の空き（dram_free），L1 の予算（oz_l1_budget）と CB の大きさの数え方（mcast_cb_bytes），コアあたりの行列積
// 構成の探索（choose_cfg），マルチキャストの引数（mcast_*_args），パネルの切り出し（つまみの読み取りは oz_knobs.hpp の表と
// oz_env.hpp）．これを使うマルチキャスト行列積のプログラムの構築は mcast_gemm.hpp（これを取り込む）にある．
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>   // DRAM の空きの照会（dram_free）
#include <cstdlib>
#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include <functional>
#include "common/kernels/mcast_args.h"   // 引数の表（デバイス API に依存しない）
#include "common/oz_knobs.hpp"           // つまみの表（OZ_L1_BUDGET_KB）

// (行, 列) → タイル内の通し番号 nf（タイルの要素を直に読み書きするカーネルと同じ 1 つの関数）
#include "common/kernels/tile_nf.h"
// 整数 → 符号-絶対値の int8（bit7 が符号，下位 7 bit が絶対値）
// -128 は絶対値が 7 bit に収まらないので -127 に丸める（桁は ±127 まで，基数は 254
// 以下なので実際には現れない）．
static inline int8_t to_sign_mag(int v) {
    if (v < 0) { if (v == -128) v = -127; v = (~v) + 1; v |= 0x80; }
    return (int8_t)v;
}
// 符号-絶対値の int32 → 符号付き int64 へ戻す
static inline int64_t dec_sm32(int32_t v) {
    uint32_t u = (uint32_t)v;
    int64_t m = (int64_t)(u & 0x7fffffffu);
    return (u & 0x80000000u) ? -m : m;
}

// 行優先の R×C 配列をタイル順・面順（16x16 の面 4 つ）へ並べ替え，各要素を符号-絶対値の int8 へ
// 符号化する（INT8 行列積の入力形式）．
inline std::vector<int8_t> tilize_sm8(const std::vector<int8_t>& s, uint32_t R, uint32_t C) {
    uint32_t Rt = R / 32, Ct = C / 32;
    std::vector<int8_t> o((size_t)Rt * Ct * 1024, 0);
    for (uint32_t tr = 0; tr < Rt; tr++)
        for (uint32_t tc = 0; tc < Ct; tc++) {
            int8_t* t = &o[((size_t)tr * Ct + tc) * 1024];
            for (uint32_t r = 0; r < 32; r++)
                for (uint32_t c = 0; c < 32; c++)
                    t[nf(r, c)] = to_sign_mag(s[(size_t)(tr * 32 + r) * C + (tc * 32 + c)]);
        }
    return o;
}

// 計算グリッド全体へ ntiles 枚を均等に配る．どの段も「コアを全部使い，タイルを
// 前から順に切って渡す」という同じ配り方をするので，コアの列挙・コア範囲・担当区間を
// ここに 1 つだけ置く．i 番目のコアの担当は [begin(i), begin(i)+count(i))．
struct GridSpan {
    std::vector<tt::tt_metal::CoreCoord> cores;
    uint32_t ntiles = 0, per = 0, ncx = 0, ncy = 0;
    uint32_t begin(size_t i) const { return std::min(per * (uint32_t)i, ntiles); }
    uint32_t count(size_t i) const { return std::min(per, ntiles - begin(i)); }
    // コアごとの担当の枚数（cores の順．尾崎 I のスライス段 slice_run が詰める形で，配り方を変える検査
    // slice_split_check はこれを書き換えて渡す）
    std::vector<uint32_t> counts() const {
        std::vector<uint32_t> c(cores.size());
        for (size_t i = 0; i < c.size(); ++i) c[i] = count(i);
        return c;
    }
    tt::tt_metal::CoreRange range() const {
        return tt::tt_metal::CoreRange({0, 0}, {(std::size_t)ncx - 1, (std::size_t)ncy - 1});
    }
};
static inline GridSpan full_grid_span(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                      uint32_t ntiles) {
    auto gd = md->compute_with_storage_grid_size();
    uint32_t ncx = (uint32_t)gd.x, ncy = (uint32_t)gd.y;
    if (!ncx) ncx = 1;
    if (!ncy) ncy = 1;
    GridSpan g;
    g.ncx = ncx; g.ncy = ncy;
    for (uint32_t y = 0; y < ncy; y++)
        for (uint32_t x = 0; x < ncx; x++) g.cores.push_back({(std::size_t)x, (std::size_t)y});
    uint32_t ncores = std::min<uint32_t>((uint32_t)g.cores.size(), ntiles ? ntiles : 1);
    g.ntiles = ntiles;
    g.per = (ntiles + ncores - 1) / ncores;
    return g;
}

// CB を 1 つ作る．素で書くと CircularBufferConfig と set_page_size の両方へ
// 同じインデックスを渡すことになり，片方だけ直して静かに壊す事故が起きやすい．
// 容量は ntiles 枚ぶん，ページはタイル 1 枚ぶんが既定（page で上書きできる：
// 1 タイルを丸ごと 1 ページとして扱う作業領域がいくつかある）．
static inline void make_cb(tt::tt_metal::Program& program,
                           const tt::tt_metal::CoreRange& cores,
                           tt::CBIndex idx, tt::DataFormat fmt,
                           uint32_t ntiles, uint32_t page = 0) {
    const uint32_t ps = page ? page : (uint32_t)tt::tile_size(fmt);
    tt::tt_metal::CreateCircularBuffer(
        program, cores,
        tt::tt_metal::CircularBufferConfig(ntiles * ps, {{idx, fmt}}).set_page_size(idx, ps));
}

// タイル 1 枚 = 1 ページの DRAM バッファを 1 つ作る．top_down なら DRAM の上端から取る（尾崎 II の awe の経路の
// 桁行列．ozaki2_main.cpp の build_slice）．既定は確保の向きを指定しない（アロケータの既定＝下から）．
static inline std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> mesh_dram_buffer(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
        size_t ntiles, uint32_t page_size, bool top_down = false) {
    tt::tt_metal::distributed::ReplicatedBufferConfig bc{.size = (size_t)page_size * ntiles};
    tt::tt_metal::distributed::DeviceLocalBufferConfig dc{
        .page_size = page_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    if (top_down) dc.bottom_up = false;
    return tt::tt_metal::distributed::MeshBuffer::create(bc, dc, md.get());
}

// DRAM の空き［B］．get_statistics はバンクあたりの値を返すので，バンク数を掛けて全バンクの総量にする（器は全バンクへ
// 等分に，同じ番地で置かれる）．largest_block は最大の連続した空き（同じく全バンクぶん．1 つの器が入るかの必要条件）．
// 照会するだけで確保はしない．尾崎の両スキームが，確保の前に段の組み方（スキーム I の発行の方式とバンドの本数，
// スキーム II の法の組分けと OZ_TIME_WHOLE の同時確保）を決めるのに使う
struct DramFree { uint64_t total = 0, largest_block = 0; };
static inline DramFree dram_free(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md) {
    using tt::tt_metal::BufferType;
    const auto st = md->allocator()->get_statistics(BufferType::DRAM);
    const uint64_t nbank = md->allocator()->get_num_banks(BufferType::DRAM);
    return {(uint64_t)st.total_free_bytes * nbank, (uint64_t)st.largest_free_block_bytes * nbank};
}
static inline uint64_t dram_free_bytes(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md) {
    return dram_free(md).total;
}

// バッファ全体を 32 bit の語 word で埋めて同期する（計時の外で使う．OZ_TIME_WHOLE の番兵 oz_whole.hpp の OZ_WHOLE_SENTINEL）．
// 14080^3 では 1 本で 10 GB を超える器があるので，ホストに全体ぶんの配列は作らず，ページの倍数の区間
// （64 MiB 前後）ずつ同じ配列から書く（BufferRegion．書き込みはホストの配列をコマンドキューへ写してから
// 戻るので，同じ配列を続けて渡してよい）．
static inline void fill_mesh_buffer_u32(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md,
                                        const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& buf,
                                        uint32_t word) {
    namespace dist = tt::tt_metal::distributed;
    if (!buf) return;
    auto& cq = md->mesh_command_queue();
    const uint64_t total = buf->size(), page = buf->page_size();
    const uint64_t chunk = std::max<uint64_t>(page, ((64ull << 20) / page) * page);
    const std::vector<uint32_t> src((size_t)((chunk + 3) / 4), word);
    const dist::MeshCoordinateRange all(md->shape());
    for (uint64_t off = 0; off < total; off += chunk) {
        const uint64_t n = std::min<uint64_t>(chunk, total - off);   // 端も page の倍数（size = page * 枚数）
        cq.enqueue_write_shard_to_sub_grid(*buf, src.data(), all, false, tt::tt_metal::BufferRegion(off, n));
    }
    dist::Finish(cq);
}

struct OptConfig {
    bool ok = false;
    uint32_t cr = 0, cc = 0, pm = 0, pn = 0, sh = 0, sw = 0, bw = 0, nb = 0;
};
// n の約数を降順に列挙する
inline std::vector<uint32_t> divisors_desc(uint32_t n) {
    std::vector<uint32_t> d;
    for (uint32_t i = n; i >= 1; --i) if (n % i == 0) d.push_back(i);
    return d;
}
// サブブロック (h,w) の候補（面積の大きい順）．DST に何タイル入るかで使える上限が
// 変わるので，choose_cfg の area_cap で絞り込む．
//   無誤差累算（int32 / fp32_dest_acc）  : DST 半減で面積 4 まで（既定．half-sync）
//   fp32_dest_acc・dst_full_sync        : 面積 8 まで（gemm_bf16 で両方を指定したとき）
//   BF16 単体・half-sync                : 面積 8 まで
//   BF16 単体・dst_full_sync            : 面積 16 まで
static constexpr std::array<std::pair<uint32_t, uint32_t>, 23> SB_CHOICES = {{
    {4, 4}, {8, 2}, {2, 8},                                        // 面積 16
    {4, 2}, {2, 4}, {8, 1}, {1, 8},                                // 面積 8
    {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},                // 面積 6-7
    {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4},                        // 面積 4-5
    {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},                        // 面積 1-3
}};
// 候補表 SB_CHOICES の中で，(pm,pn) を割り切り面積が上限以内の最初のもの（面積の大きい順に並べてあるので，表の中で
// 最大のもの）を選ぶ．表に無い形（3x3・5x2 など）は条件を満たしても選ばない（pick_subblock(3,3,16) は 3x1）
inline bool pick_subblock(uint32_t pm, uint32_t pn, uint32_t area_cap,
                          uint32_t& sh, uint32_t& sw) {
    for (const auto& s : SB_CHOICES) {
        if (s.first * s.second > area_cap) continue;
        if (pm % s.first == 0 && pn % s.second == 0) { sh = s.first; sw = s.second; return true; }
    }
    return false;
}

// 形状探索が使う L1 の予算［B］．OZ_L1_BUDGET_KB［KiB］で変えられる．scheme I・II の双方が参照する．
// 既定（OZ_L1_BUDGET_KB_DEFAULT）とその根拠，上書きの範囲（L1 の全容量 1536 KiB までの正の整数）はつまみの表
// （oz_knobs.hpp）にある．
inline uint32_t oz_l1_budget() { return oz_knobs::u32(OZK("OZ_L1_BUDGET_KB")) * 1024u; }
// GEMM 段の計算カーネル bmm_int32_spill.cpp を matmul_block で発行するか（OZ_NO_BLOCKMM が立っていれば 1 タイルずつの
// matmul_tiles．有無で読む．整数の積は同じで発行のしかただけが違う）．spill_gemm_spec の SpillOpts::blockmm に渡す
static inline bool gemm_blockmm() { return !oz_knobs::set(OZK("OZ_NO_BLOCKMM")); }

// build_mcast_gemm（mcast_gemm.hpp）が作る入力の CB の枚数（c_0・c_1．2 重緩衝で 1 ブロックの 2 倍）
struct McastIoTiles { uint32_t c0, c1; };
static inline McastIoTiles mcast_io_cb_tiles(uint32_t pm, uint32_t pn, uint32_t bw) {
    return {pm * bw * 2u, pn * bw * 2u};
}
// build_mcast_gemm が作る CB の 1 コアあたりの合計バイト数：c_0・c_1（入力形式．mcast_io_cb_tiles），c_16 と c_24
// （out_shared なら同じ L1 領域の pm*pn 枚，そうでなければ別々に pm*pn 枚ずつ．ページは out_tile），それに追加の CB の
// extra バイト．構成探索の見積り（下の cfg_fits）と，spec どおりに実際に作る CB の数え方（mcast_gemm.hpp の
// mcast_gemm_l1_bytes）がこの 1 つの式を使う
static inline uint64_t mcast_cb_bytes(uint32_t pm, uint32_t pn, uint32_t bw, uint32_t in_tile, uint32_t out_tile,
                                      bool out_shared, uint64_t extra = 0) {
    const McastIoTiles io = mcast_io_cb_tiles(pm, pn, bw);
    return ((uint64_t)io.c0 + io.c1) * in_tile + (out_shared ? 1ull : 2ull) * pm * pn * out_tile + extra;
}
// tt-metal の行列積の例（tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp の get_maximum_block_dim）と同じ
// 判定：コアあたり「タイル数 400」の固定の予算の中で，K ブロック幅 in0_block_w と一辺 block_dim のブロックに対して取れる
// もう一辺の最大（0 なら取れない）．BF16 の 2048 B/タイルを前提にした値で，データ形式を見ない．基準実装の構成をこの例の
// 規則で選ぶ（下の cfg_fits の in_tile=0）ために式を写してある（例のヘッダは大域の CoreCoord を宣言し直すなど，共通の
// ヘッダが取り込むには余分が多い）
static inline uint32_t example_max_block_dim(int32_t block_dim, int32_t in0_block_w) {
    const int32_t other_dim = (400 - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    return other_dim > 0 ? (uint32_t)other_dim : 0u;
}
// L1 に収まるかの判定．in_tile を渡すとバイトで見積もり（mcast_cb_bytes．c_16 と c_24 は同じ領域を共有し，追加の CB は
// 無いとみなす．尾崎の両スキームの GEMM 段の置き方），渡さなければ tt-metal の例の判定（example_max_block_dim）に従う．
// 狭い形式では前者でないと L1 を使い残す．
// c_16 と c_24 を別々に作る呼び出し元（gemm_int8 は Int32，gemm_bf16ext は FP32 の 2 面）と，融合・対ループの CB を足す
// 構成には，この判定は過小である（例の判定の出力側は 4 倍になる）．そういう呼び出し元はこの判定のまま，実際に作る CB
// （mcast_gemm_l1_bytes）が予算に収まる候補だけを choose_cfg の accept で受け付ける（見積りでは収まっても実際には
// 収まらない bw を飛ばして，収まる小さい bw へ降りる）．
inline bool cfg_fits(uint32_t pm, uint32_t pn, uint32_t bw,
                     uint32_t in_tile, uint32_t out_tile, uint32_t budget) {
    if (!in_tile) {   // tt-metal の例に従う判定
        uint32_t pnmax = example_max_block_dim((int32_t)pm, (int32_t)bw);
        return pnmax != 0 && pn <= pnmax;
    }
    return mcast_cb_bytes(pm, pn, bw, in_tile, out_tile, /*out_shared=*/true) <= (uint64_t)budget;
}

// 行列積の実行時パラメタを自動選択する．
// 優先順位は，(a) 使用コア数の最大化 → (b) in0_block_w（K 方向のブロック幅）の
// 最大化＝スピル往復の最小化．それでも同点なら列挙の順（cr，次に cc の大きい順）で最初のものになる（cr・cc は Mt・Nt を
// 割る値だけを取るので，コアあたり出力面積 pm*pn = Mt*Nt/(cr*cc) はコア数が同じなら等しく，面積では決まらない）．
//   area_cap  サブブロックの面積上限（上の表を参照．既定は無誤差累算向けの 4）
//   forced_bw 0 以外なら in0_block_w をその値に固定する（手動チューニング用）
//
// 対応する形状の条件：M,N,K はタイル幅 32 の倍数で，さらにタイル数 Mt=M/32 が 2..gy，Nt=N/32 が 2..gx の
// 範囲に約数を持つ必要がある（gy・gx はコア格子の縦横．マルチキャスト経路が cr,cc >= 2 を要求する）．
// Kt は bw の候補を約数から選び，約数 1（bw=1）も候補に入るので，K/32=1 を含めて
// Kt への条件は無い．
// 条件を満たさない形（例：M=96・N=416・K=32．Nt=13 は横 11 を超える素数なので割れない．Nt=11 なら通る）では
// 構成が見つからず，呼び出し側が ok=false を受けて止まる．誤答にはならない．
// 論文の測定サイズ（3520・7040・14080 の立方）はいずれも条件を満たす．
//   accept    空でなければ，上の見積り（cfg_fits）に加えて候補ごとに呼ぶ判定（実際に作る CB の合計が
//             予算に収まるか，など．gemm_exact_check・gemm_int8・gemm_bf16ext が mcast_gemm_l1_bytes で使う）．
//             サブブロックは bw によらないので先に選び，候補の構成を丸ごと渡す．
inline OptConfig choose_cfg(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t gy, uint32_t gx,
                            uint32_t area_cap = 4, uint32_t forced_bw = 0,
                            uint32_t in_tile = 0, uint32_t out_tile = 0,
                            uint32_t budget = oz_l1_budget(),   // 既定は共有の予算に従う
                            const std::function<bool(const OptConfig&)>& accept = {}) {
    OptConfig best;
    uint32_t bc = 0, bbw = 0;
    // コア格子は縦横とも 2 以上とする（マルチキャスト経路が cr,cc >= 2 を要求する）
    for (uint32_t cr = std::min(gy, Mt); cr >= 2; --cr) {
        if (Mt % cr) continue;
        uint32_t pm = Mt / cr;
        for (uint32_t cc = std::min(gx, Nt); cc >= 2; --cc) {
            if (Nt % cc) continue;
            uint32_t pn = Nt / cc;
            uint32_t sh = 0, sw = 0;
            if (!pick_subblock(pm, pn, area_cap, sh, sw)) continue;
            uint32_t bw = 0;
            // L1 に収まる範囲で最大の in0_block_w を選ぶ
            for (uint32_t cand : divisors_desc(Kt)) {
                if (forced_bw != 0 && cand != forced_bw) continue;
                if (!cfg_fits(pm, pn, cand, in_tile, out_tile, budget)) continue;
                if (accept && !accept(OptConfig{true, cr, cc, pm, pn, sh, sw, cand, Kt / cand})) continue;
                bw = cand; break;
            }
            if (!bw) continue;
            const uint32_t cores = cr * cc;
            if (cores > bc || (cores == bc && bw > bbw)) {
                bc = cores; bbw = bw;
                best = {true, cr, cc, pm, pn, sh, sw, bw, Kt / bw};
            }
        }
    }
    return best;
}

// ===== マルチキャスト行列積カーネルの実行時引数 =====
// 並びは tt-metalium の matmul_common/kernels/dataflow/ のリーダとライタの並びから，バッチの引数（リーダの末尾 4 個と
// ライタの 11・12 番．GEMM 段はバッチを扱わない）を除いたもの（データ移動カーネル kernels/mcast_split_dm.cpp が読む）．
// build_mcast_gemm を使う全実装で同一なので，ここに 1 つだけ持つ．番号の表は
// kernels/mcast_args.h（RA_*・W_*）にあり，カーネルはその名前で読む．
// 1 要素ずれてもコンパイルは通り，実行時に黙って壊れるだけなので，McastGemm::set_args が
// 個数を表と突き合わせる．
// マルチキャストの矩形は (hi, lo) の 2 隅で，tt-metal の引数名の start が hi（座標の大きい側），
// end が lo にあたる（A は行の列 1..cc-1 の R と L1，B は列の行 1..cr-1 の B と T1）．
//
// dA / dB / dW は，A/B/C の読み書き開始位置をタイル単位でずらす量（scheme I の
// K ブロック化とスライス連結が使う）．ずらさない実装は 0 のままでよい．
struct McastPeers { tt::tt_metal::CoreCoord L, L1, R, T, T1, B; };  // マルチキャストの相手先コア
struct McastSems  { uint32_t s0s, s0r, s1s, s1r; };   // A/B それぞれの送信・受信セマフォ

// 配列は表の名前（mcast_args.h の RA_*・OP_*・MC_*・W_*）で詰める．並びを変えても値が入れ替わらず，
// 詰め忘れは番兵 0xFFFFFFFF のまま残るので下の検査で止まる．
static inline void mcast_args_filled(const std::vector<uint32_t>& a, const char* who) {
    for (size_t i = 0; i < a.size(); ++i)
        TT_FATAL(a[i] != 0xFFFFFFFFu, "{}: 引数 {} 番を詰めていない（mcast_args.h の表と食い違い）", who, i);
}
static inline std::vector<uint32_t> mcast_reader_args(
    uint32_t Aaddr, uint32_t Baddr, uint32_t Nt, uint32_t Kt,
    uint32_t pm, uint32_t pn, uint32_t bw, uint32_t cr, uint32_t cc,
    uint32_t cx, uint32_t cy, const McastPeers& p, const McastSems& s,
    uint32_t dA = 0, uint32_t dB = 0) {
    namespace ma = mcast_args;
    using tt::tt_metal::CoreCoord;
    TT_FATAL(bw > 0, "mcast_reader_args: bw=0");
    std::vector<uint32_t> a(ma::RA_COUNT, 0xFFFFFFFFu);
    // オペランド 7 個（A は行 cy のブロック，B は列 cx のブロック．ブロックの高さはカーネルが読まないので渡さない）
    auto operand = [&](uint32_t base, uint32_t addr, uint32_t start, uint32_t sw_, uint32_t sh_,
                       uint32_t next, uint32_t bw_, uint32_t bt) {
        a[base + ma::OP_ADDR] = addr;       a[base + ma::OP_START] = start;
        a[base + ma::OP_STRIDE_W] = sw_;    a[base + ma::OP_STRIDE_H] = sh_;
        a[base + ma::OP_NEXT_BLOCK] = next; a[base + ma::OP_BLOCK_W] = bw_;
        a[base + ma::OP_BLOCK_TILES] = bt;
    };
    operand(ma::RA_A, Aaddr, Kt * pm * cy + dA, 1u, Kt, bw, bw, bw * pm);
    operand(ma::RA_B, Baddr, pn * cx + dB, 1u, Nt, bw * Nt, pn, pn * bw);
    a[ma::RA_NUM_BLOCKS] = Kt / bw;
    // マルチキャスト 9 個（矩形の hi・lo 隅，宛先数，前半の送信コア，セマフォ 2 つ）
    auto mcast = [&](uint32_t base, const CoreCoord& hi, const CoreCoord& lo, uint32_t ndest,
                     const CoreCoord& snd, uint32_t ssem, uint32_t fsem) {
        a[base + ma::MC_HI_X] = (uint32_t)hi.x;   a[base + ma::MC_HI_Y] = (uint32_t)hi.y;
        a[base + ma::MC_LO_X] = (uint32_t)lo.x;   a[base + ma::MC_LO_Y] = (uint32_t)lo.y;
        a[base + ma::MC_NUM_DESTS] = ndest;
        a[base + ma::MC_SENDER_X] = (uint32_t)snd.x; a[base + ma::MC_SENDER_Y] = (uint32_t)snd.y;
        a[base + ma::MC_SENDER_SEM] = ssem;       a[base + ma::MC_FLAG_SEM] = fsem;
    };
    mcast(ma::RA_MC_A, p.R, p.L1, cc - 1, p.L, s.s0s, s.s0r);
    mcast(ma::RA_MC_B, p.B, p.T1, cr - 1, p.T, s.s1s, s.s1r);
    mcast_args_filled(a, "mcast_reader_args");
    return a;
}
// 11 番の acc は融合ライタ（mcast_split_dm.cpp の WRITER=2）だけが読む．
// 標準のライタは 11 個しか読まないので，どのライタにも 12 個のまま渡す（McastGemm::set_args が個数を見る）．
// 出力の行の刻みは Nt だけで決まる（行数は要らない．バッチの出力の刻み Mt*Nt は GEMM 段がバッチを扱わないので渡さない）．
static inline std::vector<uint32_t> mcast_writer_args(
    uint32_t Caddr, uint32_t Nt, uint32_t pm, uint32_t pn,
    uint32_t sh, uint32_t sw, uint32_t cx, uint32_t cy, uint32_t dW = 0,
    uint32_t acc = 0) {
    namespace ma = mcast_args;
    TT_FATAL(sh > 0 && sw > 0, "mcast_writer_args: サブブロック {}x{} が 0 を含む", sh, sw);
    std::vector<uint32_t> w(ma::W_COUNT, 0xFFFFFFFFu);
    w[ma::W_ADDR] = Caddr;
    w[ma::W_START] = cx * pn + cy * pm * Nt + dW;
    w[ma::W_STRIDE_W] = 1u;       w[ma::W_STRIDE_H] = Nt;
    w[ma::W_NEXT_SB_W] = sw;      w[ma::W_NEXT_SB_H] = sh * Nt;
    w[ma::W_SB_W] = sw;           w[ma::W_SB_H] = sh;           w[ma::W_SB_TILES] = sw * sh;
    w[ma::W_NSB_W] = pn / sw;     w[ma::W_NSB_H] = pm / sh;
    w[ma::W_ACC] = acc;
    mcast_args_filled(w, "mcast_writer_args");
    return w;
}

// ===== 出力パネル分割 =====
// n 以下かつ cap 以下で n を割り切る最大の数（パネルの辺をタイル数で選ぶ）
static inline uint32_t largest_div_le(uint32_t n, uint32_t cap) {
    for (uint32_t d = std::min(n, cap); d >= 1; --d) if (n % d == 0) return d;
    return 1u;
}
// タイル化済みのオペランドから出力パネルぶんを切り出す．
// A（Mt x Kt タイル）の行パネル br はタイル [br*pMt, (br+1)*pMt) x Kt で連続だが，
// B（Kt x Nt タイル）の列パネル bc は Kt x [bc*pNt, (bc+1)*pNt) をとびとびに集める．
template <typename T>
inline std::vector<T> panel_rows(const std::vector<T>& At, uint32_t br, uint32_t pMt, uint32_t Kt) {
    std::vector<T> o((size_t)pMt * Kt * 1024);
    std::copy(At.begin() + (size_t)br * pMt * Kt * 1024,
              At.begin() + (size_t)(br + 1) * pMt * Kt * 1024, o.begin());
    return o;
}
template <typename T>
inline std::vector<T> panel_cols(const std::vector<T>& Bt, uint32_t bc, uint32_t pNt,
                                 uint32_t Kt, uint32_t Nt) {
    std::vector<T> o((size_t)Kt * pNt * 1024);
    for (uint32_t kt = 0; kt < Kt; ++kt)
        std::copy(Bt.begin() + ((size_t)kt * Nt + bc * pNt) * 1024,
                  Bt.begin() + ((size_t)kt * Nt + (bc + 1) * pNt) * 1024,
                  o.begin() + (size_t)kt * pNt * 1024);
    return o;
}

