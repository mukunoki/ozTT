#pragma once
// 尾崎スキーム I：GEMM 段の構成探索と発行
// ---------------------------------------------------------------------------
// GemmMode，L1 の見積り oz1_gemm_l1_bytes（実際に作る CB の合計．mcast_gemm.hpp の mcast_gemm_l1_bytes）と K ブロック幅の
// 探索 oz1_max_bw，出力パネルの構成探索（choose_cfg_1b・choose_remainder_cfg），対角 GEMM のプログラム構築 gemm_build と
// 発行 gemm_run を置く．
// 依存：ozaki1_format.hpp（OZ_FMT）と，それが取り込む ozaki_gemm_host.hpp（OptConfig・pick_subblock・divisors_desc・
// mcast_*_args），mcast_gemm.hpp（build_mcast_gemm・spill_gemm_spec・mcast_gemm_l1_bytes・mcast_check_l1）．
// L1 の予算（OZ_L1_BUDGET_KB）は環境から読まず，呼び出し側が Oz1Knobs::l1_budget を引数で渡す（ozaki1_plan.hpp）．

#include "ozaki1/ozaki1_format.hpp"
#include "common/mcast_gemm.hpp"


// ===== プログラム再利用：1 回だけ構築し，対角ごとに実行する =====
// 対角ごとに変わるのは K の長さ（連結するスライス数 ℓ+1）と出力先だけで，
// どちらも実行時引数で渡せるので，コンパイル済みプログラムを 1 つ作れば
// どの対角にも使い回せる．対角ごとに差し替えるのはリーダ・計算カーネルのブロック数 nb と
// ライタの出力スロットなど少数の実行時引数だけ（gemm_run が SetRuntimeArgs する）．
// これにより，対角ごとの CreateKernel／CB 作成／JIT のコストが丸ごと消える．

// GEMM の組み方．形状探索の L1 見積りが CB の構成に依存するので，探索へ引数で渡す．
// ファイル内のグローバルにして呼び出し側が探索の直前に代入する形にすると，探索の関数は
// 引数だけを見て決まるように見えるのに実際は代入の順序に依存し，順序を崩すと静かに別の構成を選ぶ．
struct GemmMode {
    bool fuse = false;      // Int32 累積を GEMM へ取り込む（c_2・c_3 が増える）
    bool pairloop = false;  // 対を 1 プログラムの中で回す（c_5・c_3 が増える．fuse と排他）
};

// GEMM の CB・計算カーネルの組み方（spill_gemm_spec へ渡す）．構築（gemm_build）と L1 の見積り（oz1_gemm_l1_bytes）が
// 同じものを使うので，見積りは実際に作る CB と食い違わない．fidelity と blockmm は CB の大きさに関係しない．
inline SpillOpts oz1_spill_opts(GemmMode mode, tt::tt_metal::MathFidelity fid = tt::tt_metal::MathFidelity::LoFi,
                                bool blockmm = true) {
    SpillOpts so;
    so.in_fmt = OZ_FMT.in;
    so.acc_fmt = OZ_FMT.out;
    so.fidelity = fid;
    so.fuse = mode.fuse;
    so.pairloop = mode.pairloop;
    so.blockmm = blockmm;
    return so;
}

// コアあたりの形 (pm,pn)・サブブロック (sh,sw)・K ブロック幅 bw の GEMM が 1 コアに作る CB の合計［B］．
// gemm_build と同じ spill_gemm_spec で組んだ仕様を mcast_gemm.hpp の mcast_gemm_l1_bytes が数える（c_0・c_1 が
// 入力形式の 2 重緩衝，c_16 と c_24 が同じ L1 領域を共有する出力 pm*pn 枚，融合・対ループの c_2・c_3・c_5）．
// tt-metal の get_maximum_block_dim の「タイル数 400」という固定予算はデータ形式を見ない（BF16 の 2048 B/タイルを
// 前提にした値）ので使わない：入力が BF16 2048 B・BFP8_b 1088 B・INT8/FP8 1024 B とばらつくので，狭い形式ほど
// L1 を使い残す（INT8 の pm=pn=8・bw=22 は実バイトで 960 KiB と予算に収まるのに，タイル数では 768 > 400 で弾かれる）．
// L1 の大きさは K の長さに依らないので，仕様は K ブロック 1 本ぶん（Kt = bw）で組む．
inline uint64_t oz1_gemm_l1_bytes(uint32_t pm, uint32_t pn, uint32_t sh, uint32_t sw, uint32_t bw, GemmMode mode) {
    OptConfig g;
    g.ok = true; g.cr = 2; g.cc = 2; g.pm = pm; g.pn = pn; g.sh = sh; g.sw = sw; g.bw = bw; g.nb = 1;
    return mcast_gemm_l1_bytes(g, spill_gemm_spec(g, bw, oz1_spill_opts(mode)));
}

// K ブロック幅 bw の候補（Kbt の約数＝スライス・K ブロックの境界に揃う）のうち，L1 の予算 l1_budget［B］
// （Oz1Knobs::l1_budget．OZ_L1_BUDGET_KB で変えられる）に CB が収まる最大のもの．無ければ 0．bw が Kbt を割り切る必要が
// あるのは，さもないと 1 回の読み出しが K ブロックやスライスの境界をまたぎ，A のスライス q と B のスライス ℓ-q という
// 対応が崩れるため．
inline uint32_t oz1_max_bw(uint32_t pm, uint32_t pn, uint32_t sh, uint32_t sw, uint32_t Kbt, GemmMode mode,
                           uint32_t l1_budget) {
    for (uint32_t cand : divisors_desc(Kbt))
        if (oz1_gemm_l1_bytes(pm, pn, sh, sw, cand, mode) <= (uint64_t)l1_budget) return cand;
    return 0;
}

// ===== 出力パネル分割の構成探索 =====
// コアあたりの出力 (pm,pn) は L1 の予算で決まり（最大 8×8），DST の上限（int32 で
// 4 タイル）はサブブロックの大きさだけを制約する（K ブロックの中で部分和を c_24 へスピルする．
// choose_cfg_1b のコメント）．コア格子が覆うのは Mt×Nt の出力のうち
// (cr*pm)×(cc*pn) の 1 ブロックだけでよい．ホスト側で出力全体を
// (Mt/(cr*pm))×(Nt/(cc*pn)) 個のブロック（パネル）に分け，ブロックごとに
// K を流し直す（入力は読み直すが，パネル間で部分和を持ち越すことはない）．これで，コア格子に割り切れないサイズでも
// 全 110 コアを使える．形状の制約は行と列で違う（gy・gx はコア格子の縦横）．
//   列方向（割り切りを要求する）：Nt/pn が 2..gx の約数を持つ pn（1・2・4・8 のうち Nt を割るもの）があること．
//   行方向（端数パネル行を許す）：Mt を割る pm と 2 <= cr <= min(gy, Mt/pm) で (Mt/pm) % cr != 1 となるものが
//     あればよい（余りのコア行数 1 だけが組めない）．Mt >= 2 ならほぼ常に満たし，Mt <= gy なら cr = Mt/pm で
//     必ず取れる（例：Mt=13 は pm=1・cr=10 と残り 3 行）．外れるのは Mt が奇数で 3..gy のすべてで割って
//     余り 1 になる場合（gy=10 なら Mt ≡ 1 mod 2520．Mt=2521 など）だけである．
// scheme II の choose_cfg（common/ozaki_gemm_host.hpp）は行も列も割り切りを要求するので条件が違う．
//
// 探索の関数（choose_cfg_1b・choose_remainder_cfg）の引数はすべてタイル数である（名前は ozaki1_main.cpp の冒頭の一覧）．
// Kt   対角 1 本の GEMM が見る K タイル総数（呼び出し側の Ktm = (L+1)*K/32．
//      スライス連結後）．構成の nb = Kt/bw（OptConfig の欄．scheme I は読まず，K ブロック数は gemm_run が付け替える）
// Kbt  K ブロック 1 個ぶんの K タイル数（分割なしなら K/32）．bw はこれを割り切る（oz1_max_bw）．
// l1_budget は形状探索の L1 の予算［B］（Oz1Knobs::l1_budget）．
// コアあたりの形状 (pm,pn) はどの構成でも 8 まで探す．対ループも対ごとの発行も使わない INT8 経路も
// 同じで，14080^3 の I-INT8 は 4 で止めるより 8x8 のほうが速い（docs/nongemm_opt_results.md の §7）．
// サブブロックは共有ヘッダの pick_subblock で面積 4 以下の最大を選ぶ（pm・pn は
// どちらの探索でも 2 のべきなので，候補表の 3 を含む形は選ばれない）．

// 端数パネル行の構成を選ぶ．通常パネルの (cc,pn) はそのままに，残りの行数 rem を
// (cr,pm) でどう割るかだけを決め直す．cr を大きく取るほど使うコアが増えるので，
// コア数最大で選ぶ：14080^3 の BFP8 では rem=40 行を pm=8/cr=5（55 コア）ではなく
// pm=4/cr=10（110 コア）で覆える．サブブロックと bw は新しい pm で取り直す（nb も取り直した bw から Kt/bw）．
static OptConfig choose_remainder_cfg(uint32_t rem, const OptConfig& gU,
                                      uint32_t gy, uint32_t Kt, uint32_t Kbt, GemmMode mode, uint32_t l1_budget) {
    // 端数の行は choose_cfg_1b が作る r*pm（2 <= r < cr <= gy，pm は 1・2・4・8）なので 72 行以下で，
    // どの行数でもこの 4 通りから取れる（16 は，取れる行数では必ずより大きい cr の候補に負けるので置かない）．
    static constexpr std::array<uint32_t, 4> PM = {{8, 4, 2, 1}};
    OptConfig best; uint32_t bcore = 0;
    for (uint32_t cr = std::min(gy, rem); cr >= 2; --cr) {
        if (rem % cr) continue;
        const uint32_t pm = rem / cr;
        if (std::find(PM.begin(), PM.end(), pm) == PM.end()) continue;
        uint32_t sh = 0, sw = 0;
        if (!pick_subblock(pm, gU.pn, 4, sh, sw)) continue;
        const uint32_t bw = oz1_max_bw(pm, gU.pn, sh, sw, Kbt, mode, l1_budget);
        if (!bw) continue;
        if (cr * gU.cc > bcore) {
            bcore = cr * gU.cc;
            best = {true, cr, gU.cc, pm, gU.pn, sh, sw, bw, Kt / bw};
        }
    }
    return best;
}

static OptConfig choose_cfg_1b(uint32_t Mt, uint32_t Nt, uint32_t Kt,
                               uint32_t Kbt, uint32_t gy, uint32_t gx,
                               GemmMode mode, uint32_t l1_budget) {
    // スピルありの構成：コアあたりの (pm,pn) は DST の上限を超えてよい．
    // カーネルが K ブロックの中で (sh,sw) のサブブロックを順に回し，部分和を
    // c_24 経由でスピルするからである．ただしサブブロックの面積は DST に収まる
    // 必要がある（fp32_dest_acc_en のとき int32 で 4 タイルまで）．
    // 候補の (pm,pn) を面積の大きい順に試し，それぞれで最大の
    // サブブロック（sh|pm, sw|pn, sh*sw<=4）を選ぶ．
    // 形状の上限は L1 で決まる（oz1_max_bw が CB の合計を予算と比べる）．pm=pn=8・bw=2・BF16 入力／FP32 出力の
    // 対ループで約 400 KiB と予算に収まる．
    static constexpr std::array<std::pair<uint32_t, uint32_t>, 16> PC_ALL = {{
        {8, 8}, {8, 4}, {4, 8}, {8, 2}, {2, 8}, {4, 4}, {8, 1}, {1, 8},
        {4, 2}, {2, 4}, {4, 1}, {1, 4}, {2, 2}, {2, 1}, {1, 2}, {1, 1}}};
    OptConfig best;
    uint32_t bcore = 0, bcov = 0;
    for (auto& a : PC_ALL) {
        uint32_t pm = a.first, pn = a.second;
        if (Mt % pm || Nt % pn) continue;
        // (pm,pn) を割り切る中で面積 4 以下の最大のサブブロックを選ぶ
        uint32_t sh = 0, sw = 0;
        if (!pick_subblock(pm, pn, 4, sh, sw)) continue;
        // bw はコア格子に依らず (pm,pn) とサブブロックで決まる
        const uint32_t bw = oz1_max_bw(pm, pn, sh, sw, Kbt, mode, l1_budget);
        if (!bw) continue;
        // コア格子は縦横とも 2 以上とする．データ移動は行／列の両端の 2 コアが送信側に
        // なる構成なので，cr==1 や cc==1 では組めない（共有側 choose_cfg と同じ制約）．
        for (uint32_t cr = std::min(gy, Mt / pm); cr >= 2; --cr) {
            // 行方向は端数パネルを許す：最後のパネル行だけ小さいコア格子で覆えばよい
            // （呼び出し側が別の構成で組む）．ただし端数もコア格子 2 行以上を要するので，
            // 余りが pm の倍数で cr_rem >= 2 になるときだけ認める．
            // 列方向は割り切れることを要求する：端数を両方向に許すとパネルの種類が
            // 4 つに増え，マルチキャストの構成もそれだけ持つことになる．
            // 余り［コア行数］が 1 だと格子 1 行になり，mcast が組めない．
            if ((Mt / pm) % cr == 1) continue;
            for (uint32_t cc = std::min(gx, Nt / pn); cc >= 2; --cc) {
                if ((Nt / pn) % cc) continue;
                const uint32_t cores = cr * cc, cov = cr * pm * cc * pn;
                // 優先順位：コア数 → 1 パネルの面積（同点は列挙の順で先のもの）．
                // コア数を先に置くのは，端数パネルを許すと「大きいが半分の
                // コアしか使わない」構成も候補に入るからである
                // （14080^3 で pm=8 は Mt/pm=55 となり，2..10 の約数が 5 しかないので
                //  割り切る構成だけでは 5x11=55 コアに落ちる）．
                // 1 パネルの面積は cov = コア数 × コアあたり面積 pm*pn なので，コア数と 1 パネルの面積が同じならコアあたり
                // 面積も同じで，3 段目に置いても決まらない（共有の choose_cfg の注記と同じ理由．ozaki_gemm_host.hpp）．
                if (cores > bcore || (cores == bcore && cov > bcov)) {
                    bcore = cores; bcov = cov;
                    best = {true, cr, cc, pm, pn, sh, sw, bw, Kt / bw};
                }
            }
        }
    }
    return best;
}

struct GemmCtx {
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange range{tt::tt_metal::distributed::MeshCoordinate(0, 0),
                                                         tt::tt_metal::distributed::MeshCoordinate(0, 0)};
    // A・B はスライス連結済みのオペランド全体（どのパネルも同じものを共有し，持つのは寿命のため），C は対角ごとの部分和 Pbig
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> Abuf, Bbuf, Cbuf;
    uint32_t pm = 0, pn = 0, sh = 0, sw = 0, Nt = 0;
    uint32_t dWblock = 0;  // このブロック (br,bc) 用に焼き込んだ出力オフセット
    // 対角の面と面の間隔［タイル］．行バンドに切らなければ Mt*Nt，切るならバンド 1 本ぶんのタイル数．
    uint32_t planeStride = 0;
    // 対角ごとに差し替えるリーダの状態．nslice は 1 K ブロックあたりのスライス本数
    // L+1（p ではない），Kbt は K ブロック 1 個ぶんの K タイル数（分割なしなら K/32）．
    // B の開始位置とブロック数を対角 ℓ・K ブロック b ごとに付け替える．
    uint32_t bw = 0, nslice = 0, Kbt = 0;
    bool fuse = false;          // 出力経路へ Int32 累積を融合しているか
    bool pairloop = false;      // 対を 1 プログラムの中で回しているか
    // コアごとのカーネルと構築時のリーダ引数．引数の設定は mg.set_args に任せる
    // （データ移動を A/B に分けた構成ではライタの引数が B 側のカーネルに載るため）．
    McastGemm mg;
};

// 対角 GEMM のプログラムの組み方（gemm_build の引数のうち，パネルによらず決まるもの）
struct GemmBuildSpec {
    uint32_t M = 0, N = 0, K = 0;   // K はスライス 1 本ぶん（もとの K）［要素］
    uint32_t nslice = 0;        // スライスの本数 L+1（連結の段数．連結後の K は nslice*K）
    uint32_t Kbt = 0;           // K ブロック 1 個ぶんの K タイル数（K/32 を割り切る）
    uint32_t plane_stride = 0;  // 対角の面と面の間隔［タイル］（行バンドに切らなければ Mt*Nt）
    GemmMode mode;
    tt::tt_metal::MathFidelity fidelity = tt::tt_metal::MathFidelity::LoFi;
    bool blockmm = true;        // matmul_block を使う（OZ_NO_BLOCKMM の否定．Oz1Knobs）
    uint32_t l1_budget = 0;     // 形状探索の L1 の予算［B］（Oz1Knobs::l1_budget．作った CB の合計をこれと照合する）
    GemmProbe probe;            // GEMM 段の探針（Oz1Diag::gemm_probe．立てると CONFIG 行が probe=set になる．ozaki1_main.cpp）
};

// gemm_build は，ブロック (br,bc) に対するリーダの A オフセットとライタの基準
// アドレスを構築時に焼き込む．A/B はスライス連結済みのバッファ（呼び出し側が 1 回だけ確保・スライスして全パネルで
// 共有する．パネルごとに作り直すとスライス段のコストが nbr*nbc 倍になる）．対角 d ごとに gemm_run がリーダの
// B 開始位置，リーダと計算カーネルのブロック数，ライタのスロットを付け替える．
// (row0, col0) はこのパネルの左上のタイル座標，band_row0 はこのパネルが属する行バンドの先頭タイル行．
// 構築の後に，作った CB の合計が L1 の予算に収まることを確かめる（形状探索の見積りと同じ数え方なので，探索を
// 通した構成なら必ず通る．食い違えば探索か CB の作り方の片方だけを変えた誤り）．
static GemmCtx gemm_build(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, const OptConfig& g,
                          const GemmBuildSpec& b,
                          const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Pbig,
                          const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Abuf,
                          const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& Bbuf,
                          uint32_t row0, uint32_t col0, uint32_t band_row0) {
    namespace dist = tt::tt_metal::distributed;
    using tt::constants::TILE_HEIGHT; using tt::constants::TILE_WIDTH;
    // 融合と対ループの排他，およびその 2 つが FP32 の部分積でしか組めないこと（INT8 経路は使わない）は
    // spill_gemm_spec が検査する．
    dist::MeshWorkload workload;
    auto device_range = dist::MeshCoordinateRange(md->shape());
    tt::tt_metal::Program program{};

    const uint32_t Mt = b.M / TILE_HEIGHT, Kt = b.nslice * (b.K / TILE_WIDTH), Nt = b.N / TILE_WIDTH;
    const uint32_t bw = g.bw;
    TT_FATAL(b.plane_stride > 0, "gemm_build: 対角の面の間隔が 0");
    // PAIRLOOP：1 対あたりの K ブロック数（対の境界をカーネルに教える）．
    // bw は K ブロック 1 個ぶん（1 対ぶん）の K タイル数を割り切らなければならない：さもないと
    // 1 回の読み出しがスライスの境界をまたぎ，対角ごとの付け替え（gemm_run）も端数を落とす．
    TT_FATAL(b.Kbt > 0 && bw > 0 && b.Kbt % bw == 0 && Kt % bw == 0,
             "gemm_build: bw={} が 1 対の K タイル数 {} と連結後の Kt={} を割り切らない", bw, b.Kbt, Kt);
    const uint32_t bpp = b.Kbt / bw;

    // CB・計算カーネル・コンパイル時引数は gemm_int8・gemm_exact_check・scheme II と共有する
    // （common/mcast_gemm.hpp の spill_gemm_spec）．融合時の出力は Int32（既存の S_d へ 2 の補数で
    // 足し込んだもの）で，c_24 は K ブロック間の部分和なので FP32 のまま．両者は同じ L1 領域を
    // 共有する．融合時は出力を書く前に同じ位置の既存 S を c_2 へ読むライタを使う．
    // K ブロック間の部分和はパッカが c_24 上で足し込み（L1 累積），DST へ読み戻すのは区間
    // （対ループなら 1 対，そうでなければ K 全体）の最後の 1 回だけである．読み戻しは c_24 の
    // UnpackToDestFp32 で FP32 のまま戻るので無誤差である（common/kernels/bmm_int32_spill.cpp の冒頭）．
    // fidelity は桁の上限から決めたもの（ozaki1_format.hpp の oz1_fidelity．桁 <= 32 なら LoFi で積が
    // 無誤差，それを超えると HiFi4．INT8 経路は常に HiFi4）．
    McastGemmSpec spec = spill_gemm_spec(g, Kt, oz1_spill_opts(b.mode, b.fidelity, b.blockmm));
    // 作った CB の合計が L1 の予算に収まるか（共有の安全網．形状探索の見積り oz1_gemm_l1_bytes と同じ数え方なので，食い違えば
    // 探索か CB の作り方の片方だけを変えた誤り）
    mcast_check_l1(g, spec, "gemm_build", b.l1_budget);
    // 計算カーネルの num_blocks は実行時引数 CR_NUM_BLOCKS（common/kernels/gemm_compute_args.h）で，対角ごとに
    // gemm_run が付け替える．ここでは最大値を入れておく．
    spill_check_run(Kt / bw, b.mode.pairloop, bpp);
    spec.compute_rt = spill_compute_rt(Kt / bw, 0u, bpp);
    // このパネルの A/B/出力オフセットを構築時に焼き込む．パネルの左上を
    // (row0, col0)［タイル単位］で受け取るので，パネルの大きさが揃っていなくてよい：
    // 最後のパネル行だけ小さいコア格子で覆う構成（ozaki1_plan.hpp の端数パネル）が取れる．
    spec.dA = Kt * row0;                      // A の開始タイル
    spec.dB = col0;                           // B の開始タイル
    // 出力の基準位置．Pbig はバンド 1 本ぶんしか無いのでバンド先頭からの相対位置で書く
    // （A の読み出し dA は全体の row0 のままである）．
    const uint32_t dWblock = col0 + (row0 - band_row0) * Nt;
    spec.dW = dWblock;
    spec.probe = b.probe;   // OZ_GEMM_PROBE_*（main の冒頭で Oz1Diag に読んだもの）
    McastGemm mg = build_mcast_gemm(program, md, g, Mt, Nt, Kt, Abuf, Bbuf, Pbig, spec);

    workload.add_program(device_range, std::move(program));
    GemmCtx c;
    c.workload = std::move(workload);
    c.range = device_range;
    c.Abuf = Abuf; c.Bbuf = Bbuf; c.Cbuf = Pbig;
    c.pm = g.pm; c.pn = g.pn; c.sh = g.sh; c.sw = g.sw;
    c.Nt = Nt; c.dWblock = dWblock; c.planeStride = b.plane_stride;
    c.bw = bw; c.Kbt = b.Kbt; c.fuse = b.mode.fuse; c.pairloop = b.mode.pairloop;
    c.nslice = b.nslice;
    c.mg = std::move(mg);
    return c;
}

// 対角 d（論文の ℓ）・K ブロック blk の GEMM を発行する．スライス連結済みの A/B
// バッファは 1 回だけアップロード済みで，ここで付け替えるのは次のものだけである：
//   リーダの A 開始位置   RA_A_START    += (blk*nslice + a0)*Kbt  （ブロック blk の先頭．対ごとなら
//                                                             A スライス a0 = pair の位置）
//   リーダの B 開始位置   RA_B_START    += (blk*nslice + (nslice-1-d) + a0)*Kbt*Nt （スライス d..0 の窓）
// nslice は 1 K ブロックあたりのスライス本数で，スライス段が書いた刻みと一致する（L+1．p ではない）．
//   リーダのブロック数     RA_NUM_BLOCKS  = nb = n*Kbt/bw（連結なら n = d+1，対ごとなら 1）
//   計算カーネルの実行時引数 = nb，融合時の acc，1 対あたりのブロック数 bpp（common/kernels/gemm_compute_args.h の ComputeRT）
//   ライタの出力スロット   W_START       += slot*plane（plane は面の間隔．バンドに切るとバンド幅）
//   融合ライタの acc       W_ACC
// （名前は common/kernels/mcast_args.h の表．ライタの引数は mcast_writer_args で作り直し，リーダの付け替えと一緒に
// McastGemm::retarget_k で設定する）
// そのうえで，使い回しているワークロードを投入する．
// A/B はブロック優先レイアウト（slice_writer 参照）なので，ブロック blk の
// スライス連結一式は連続した K 範囲に載っており，開始位置をずらすだけで読める．
// Kbt == K/32（分割なし）かつ blk == 0 のときは分割前の式に一致する．
// pair < 0 なら対角 d の (d+1) 対を K 方向に連結して 1 本で回す（既定）．
// pair >= 0 ならその 1 対だけを回す：A スライス pair と B スライス (d-pair) の積で，
// 1 本あたりの K が (d+1)K_c ではなく K_c になる．連結をやめるぶん本数は
// p(p+1)/2 に増えるが，どの本も K が最大になるので 1 本あたりの効率が上がり，
// 無誤差条件も (d+1) 倍を負わずに済む（論文の ssec:noblock のスライス対ループの段落）．
// acc は融合時のみ意味を持つ：その累積面への最初の書き込みなら 0（足さずに置く），
// 2 回目以降は 1（既存の S を読んで Int32 で足す）．
static void gemm_run(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md, GemmCtx& c,
                     uint32_t d, uint32_t slot, uint32_t blk = 0, int pair = -1,
                     uint32_t acc = 0) {
    auto& cq = md->mesh_command_queue();
    tt::tt_metal::Program& prog = c.workload.get_programs().at(c.range);
    const uint32_t plane = c.planeStride;
    const uint32_t n = (pair < 0) ? (d + 1) : 1u;               // 連結する対の数
    const uint32_t a0 = (pair < 0) ? 0u : (uint32_t)pair;       // 先頭の A スライス
    // 対角 d・対 pair が連結の窓（スライス 0..nslice-1）の中にあること．外れると (nslice-1-d) が
    // uint32 で下位あふれして範囲外を読む（parse_run_config が L < p を検査している）．
    TT_FATAL(d < c.nslice && (pair < 0 || (uint32_t)pair <= d),
             "gemm_run: 対角 d={} 対 {} が連結の段数 {} の外", d, pair, c.nslice);
    uint32_t nb = n * c.Kbt / c.bw;                   // このプログラムの K ブロック数
    uint32_t aoff = (blk * c.nslice + a0) * c.Kbt;    // A の開始 K 列のずれ
    uint32_t boff = (blk * c.nslice + (c.nslice - 1 - d) + a0) * c.Kbt * c.Nt;  // B の開始 K 行のずれ
    const uint32_t bpp = c.Kbt / c.bw;      // 1 対あたりの K ブロック数（gemm_build が bw | Kbt を検査済み）
    spill_check_run(nb, c.pairloop, bpp);
    for (const auto& e : c.mg.cores) {
        const uint32_t cx = (uint32_t)e.core.x, cy = (uint32_t)e.core.y;
        auto wa = mcast_writer_args((uint32_t)c.Cbuf->address(), c.Nt,
                                    c.pm, c.pn, c.sh, c.sw, cx, cy,
                                    c.dWblock + slot * plane, c.fuse ? acc : 0u);
        c.mg.retarget_k(prog, e, aoff, boff, nb, wa);
        tt::tt_metal::SetRuntimeArgs(prog, c.mg.ck, e.core, spill_compute_rt(nb, c.fuse ? acc : 0u, bpp));
    }
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, c.workload, false);
}
