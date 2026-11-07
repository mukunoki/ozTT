#pragma once
// 尾崎スキーム I：形式の定義と無誤差条件（ホスト側の判定だけで，デバイスを触らない）
// ---------------------------------------------------------------------------
// scheme I の形式ごとの値 OZ1_PARAM（表 Oz1Param は ozaki1_param.h．形式そのものの性質 OZ_FMT は oz_format.hpp），
// 桁の上限 ozaki_digit_max・MathFidelity の規則，無誤差条件（ozaki_check_exact / ozaki_block_k /
// oz_defer_fits / oz_horner_fits）と，OZ_SOFTFLOAT の読み取り・OZ_DBG の表示をここに置く（理論ピークは common/oz_fidelity.hpp）．
// 無誤差条件の関数は環境変数も大域の状態も読まず，桁の上限・MathFidelity・OZ_NOBLOCK を引数で受ける（発行の計画
// plan_issue が RunCfg と Oz1Knobs だけから決まるように．ozaki1_plan.hpp）．
// 依存：tt-metalium のホスト API（MathFidelity・tt::DataFormat・TILE_WIDTH・fmt）と
// ozaki_gemm_host.hpp（つまみの表 oz_knobs.hpp・読み方 oz_env.hpp も取り込む）．ozaki1 のほかのヘッダはどれもこれを取り込む．
// ヘッダの中では名前空間を開かない（tt-metal の名前は tt:: で修飾する．大域の非推奨の別名を使わないためでもある）．

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>

// nf / to_sign_mag / dec_sm32 / OptConfig / divisors_desc / choose_cfg とつまみの読み取りは
// scheme II と共通なので別ヘッダにまとめてある．下の ozaki_digit_max がつまみを読むので，
// 形式の定義より前に取り込む．
#include "common/ozaki_gemm_host.hpp"

// 形式の選択（-DOZ_FMT_*）と，形式そのものの性質（OZ_FMT.in / slice_out / out / out_half /
// exact_int_max / name）は両スキーム共通の oz_format.hpp にある．
#include "common/oz_format.hpp"

// scheme I の形式ごとの値の表（Oz1Param・oz1_param）と桁の上限の規則（oz1_digit_max）は，ホスト単体の道具
// （tools/oz_sim.h）も取り込めるよう tt-metal に依存しない ozaki1_param.h に置く．
#include "ozaki1/ozaki1_param.h"
inline constexpr Oz1Param OZ1_PARAM = oz1_param(OZ_FMT_ID);
// 部分積が Int32（INT8 経路）なら 1：再構成が Int32 の P_d を無誤差な FP32 2 本に分ける．
// 値は形式の out から導く（表に別に書かない）．
inline constexpr bool OZ1_INT32_PARTIAL = (OZ_FMT.out == tt::DataFormat::Int32);

// 行列エンジンのパス数は桁の上限だけで決まる．形式によらず一つの規則で，scheme II と
// 共通の oz_fidelity_rule（oz_format.hpp）が実装する：
// LoFi（1 パス）の乗算器は SrcA が 5 bit なので，有効ビットが 5 以下の整数（0..31 と 32）
// までが無誤差に掛かる．すなわち桁が 32 以下なら LoFi で積が無誤差になり，
// それを超える場合は完全精度側の HiFi4（4 パス）を採る．重み（基数）はその 2 倍まで
// 取れるので，基数 64 でも LoFi のままである．
//
// 境目は総当たりで測った（docs/fidelity_radix_sweep.md）：BF16 は LoFi で桁の上限 32
// （33 で破綻），HiFi2 で 63 に達し，HiFi3・HiFi4 でも 63 のままである．FP8 は全
// fidelity で 16（形式が [-16,16] しか無誤差に表せないため．乗算器の幅ではない）．
//
// HiFi3 は本方式では使い道がない：4 つの部分積のうち LoFi は A_hi*B_hi，HiFi2 はこれに
// A_lo*B_hi，HiFi3 はさらに A_hi*B_lo を足すので，厳密になる条件はそれぞれ
// A_lo=B_lo=0／B_lo=0／A_lo*B_lo=0 である．個々の積では HiFi3 が HiFi2 に勝つ（A_lo=0 かつ B_lo!=0）
// ことはあるが，要るのは使う桁の集合全体での保証である．A と B は同じ集合（基数が同じ）から桁を取るので，
// HiFi3 の保証には同じ値 v どうしの積も含めて A_lo(v)*B_lo(v)=0 が要り，FP 経路は値の先頭が最上位ビットへ
// 正規化されるので A_lo(v)=0 なら B_lo(v)=0 である．したがってそれは集合のすべての v で B_lo(v)=0，
// すなわち HiFi2 の保証の条件と同じになる．
//
// HiFi2 を使えば桁 63（基数 126）が載り，スライスが 10 枚から 9 枚（本数は 55 から 45）
// へ減るが，2w=12 で無誤差条件が厳しくなり K ブロック化が要る．行列エンジンのパス数も
// 2 倍になるので，14080^3 では割に合わない．既定は基数 64・桁 32・LoFi とする
// （K から選び分ける自動化は未実装）．
//
// INT8 だけは別で，基数を 5 bit に落としても低 fidelity では答えが出ない
// （LoFi 0／HiFi2 1.0／HiFi3 1.0／HiFi4 53.8 bit）．ttsim の INT8 経路は FP とは別の
// マスクで読み，SrcA の上位区画が絶対値 bit7-5 の 3 bit しかないためである
// （docs/int8_fidelity_mechanism.md）．幅の問題ではないので基数では回避できない．
//
// 判定に使うのは重み（基数）ではなく桁の上限 ozaki_digit_max である．結果として
// BF16／BFP8（基数 64・桁 32）と FP8（基数 32・桁 16）と BFP4（基数 14・桁 7）と
// BFP2（基数 2・桁 1）が LoFi，INT8（基数 254・桁 127）が HiFi4 になる．
// 2 のべきの基数を採るのは，2 進の仮数が有限桁で尽き，基数倍が tfMulP2（sfpu_tf_proc.h．Dekker 分割が要らない）で
// 済むからである．無誤差条件の側の損得は無い：基数 64 と 63 はどちらも桁の上限が 32（下の ceil(radix/2)）で
// 桁積の上界 2^10 も同じなので，64 にしても条件は変わらず，スライスあたりの取り出しは log2(64/63)，約 0.023 bit
// 多いだけである．
// 桁の上限の既定の規則（ceil(radix/2) と形式の上限の小さいほう）とその理由は ozaki1_param.h の oz1_digit_max．
// OZ_DMAX は上の規則を上書きする．調査用で，桁の上限を (radix-1)/2 に落とすと
// K が大きいところで clamp が発火する様子を見られる（scripts/diagnose/radix_odd.sh）．
// OZ_DMAX は 1..OZ_FMT.exact_int_max の整数に限る．0 や読めない値を黙って無視しない．
static inline uint32_t ozaki_digit_max(uint32_t radix) {
    static const uint32_t dmax_env =
        oz_env::u32("OZ_DMAX", 0u, 1u, OZ_FMT.exact_int_max);   // 0 = 未設定（範囲は形式から決まるので表は持たない）
    return oz1_digit_max(radix, OZ_FMT.exact_int_max, dmax_env);
}

// 桁の上限 dmax で掛ける MathFidelity．OZ_FIDELITY があればそれを，無ければ規則（oz_format.hpp の oz_fidelity_rule）で
// 決める．上書きの検査を先に行う（打ち間違いは桁の計算より前に止める）．ドライバは基数を決めたところで 1 度だけ呼び，
// 結果を RunCfg に持って配る（ozaki1_main.cpp の parse_run_config）．
inline tt::tt_metal::MathFidelity oz1_fidelity(uint32_t dmax) {
    if (const auto f = oz_fidelity_env()) return *f;
    return oz_fidelity_rule(dmax);
}
// 尾崎スキームの無誤差条件
// ---------------------------------------------------------------------------
// 各積は |a^(i) b^(j)| <= d_max^2（d_max は桁の上限 ozaki_digit_max．2w = ceil(log2 d_max^2) が
// ozaki_prod_bits で，基数 64・桁 32 なら 10 bit）で，1 本の GEMM がそれを K_eff 項ぶん累算する．
// 累算器が丸めなく保持できる整数ビット幅を b_acc として
//
//     2w + ceil(log2 K_eff) <= b_acc
//
// を満たせば，同符号の最大値でも無誤差累積を保証できる．満たさない場合はこの十分条件では保証できない．
// これは尾崎らの原論文が課す設計条件で，基数は K から決まる（K が大きいほど小さくする）．
//
// 同一対角のスライス対を K 方向に連結して 1 本の GEMM にする発行では K_eff = (L+1) * K_c である
// （対ごとに GEMM を分ける原典より (L+1) 倍きびしい）．既定の PAIRLOOP（対の境界で Int32 面へ畳む）と
// 対ごとの発行では K_eff = K_c（K_c は K ブロック長．ozaki1_plan.hpp の plan_issue）．
//
// b_acc は経路ごとに異なる：
//   INT8 以外の 7 形式 -> FP32 累算 : 24（FP32 が整数を無誤差に表せる上限 2^24）
//   INT8       -> INT32 累算 : 31（符号-絶対値 int32 の絶対値部）
//
// 乱数入力では部分和が符号の相殺でランダムウォークになるため，この条件を破っても
// 見かけ上は正しい値が出る．しかし非負行列のように相殺が起きない入力では実際に
// 破綻する（radix 63 の BF16 は K=7040 で 53 bit -> 17 bit）．
// したがって「実測で問題ないから良い」とはせず，構成時に必ず検査する．
static constexpr uint32_t OZAKI_ACC_BITS = OZ1_INT32_PARTIAL ? 31u : 24u;

// OZ_SOFTFLOAT=1 で，FP64 <-> triple-float の変換を C の double 式（libgcc の
// soft-float）で書いた版に切り替える．「整数ビット演算でなければどれだけ遅いか」を示すための測定専用の経路である
// （スライス段のリーダ・計算カーネルと再構成段のライタが変わる）．読むのは入口だけ：ドライバは main の冒頭で
// Oz1Diag に読み（ozaki1_plan.hpp の oz1_diag_from_env），段の構築は引数で受ける．slice_split_check は入口で拒む．
static inline uint32_t oz_softfloat() {
    static const uint32_t v = oz_knobs::set(OZK("OZ_SOFTFLOAT")) ? 1u : 0u;
    return v;
}

// OZ_DBG：進捗を stderr へ出す（答えは変えない．docs/env_vars.md）．表示だけのつまみなので，Oz1Knobs・Oz1Diag に集めず
// 表示する所で読む（計画と段の組み方はこれで変わらない）．
static inline bool oz_dbg_on() {
    static const bool v = oz_knobs::set(OZK("OZ_DBG"));
    return v;
}
static inline void oz_dbg(const char* msg) {
    if (!oz_dbg_on()) return;
    fmt::print(stderr, "DBG: {}\n", msg);
    std::fflush(stderr);
}

// 桁の上限 dmax のスライス積 |d_i d_j| <= dmax^2 が占めるビット数 ceil(log2(dmax^2))．
// 無誤差条件 2w + ceil(log2 K_eff) <= b_acc の左辺 2w はこれである．桁の上限が dmax=2^s でも
// 2^s-1 でも 2s になる（(2^s-1)^2 < 2^{2s} = (2^s)^2）ので，桁の上限が 2 のべき（基数 64 の 32 など）でも
// 1 bit 損をすることはない（「dmax <= 2^s-1 となる最小の s」で数えると，2 のべきの dmax を 1 bit 余分に
// 見積もってしまう）．基数 64（dmax 32）なら 10 である．
static inline uint32_t ozaki_prod_bits(uint32_t dmax) {
    const uint64_t r2 = (uint64_t)dmax * dmax;
    uint32_t b = 0;
    while ((1ull << b) < r2) ++b;
    return b;
}

// K_eff 項を無誤差に累算できる最大の「桁」：d^2 * 2^ceil(log2 K_eff) <= 2^b_acc（すなわち
// 2w + ceil(log2 K_eff) <= b_acc．INT8 経路はさらに d^2 * K_eff <= 2^31-1）を満たす最大の d．
// K_eff が 2 のべきでないときは d^2 * K_eff <= 2^b_acc の最大解より小さいことがある（FP32 累算・
// K_eff=12288 では 32 を返すが，d^2 * K_eff なら 36 まで載る）．無誤差条件の式と同じ側に揃えた安全側である．
// 返すのは桁の上限であって基数ではない（基数はここから beta = 2d と決まる）．
// 予算 2^(b_acc-bits) が 1 や 2 でも桁 1 は載るので，0 を返すのは K_eff > 2^b_acc の
// ときだけである（ozaki_block_k の keff_max と同じ境界．bits+1 >= b_acc で 0 を返すと，
// BFP2 の K_eff=2^23 のような成立する構成まで違反と判定してしまう）．
// INT8 経路の部分積は符号-絶対値 Int32 で，2^31 ちょうどは表せない（最大 2^31-1）．
// FP32 は 2^24 を無誤差に表せるので，等号を許すのは FP32 側だけである．
static inline uint32_t ozaki_max_digit(uint64_t k_eff) {
    uint32_t bits = 0;
    while ((1ull << bits) < k_eff) ++bits;          // ceil(log2 k_eff)
    if (OZAKI_ACC_BITS < bits) return 0;
    const uint64_t budget = 1ull << (OZAKI_ACC_BITS - bits);   // d^2 の上限
    auto fits = [&](uint64_t d) {
        if (d * d > budget) return false;
        return !OZ1_INT32_PARTIAL || d * d * k_eff <= (uint64_t)INT32_MAX;
    };
    if (!fits(1)) return 0;
    uint32_t r = 1;
    while (fits(r + 1)) ++r;
    return r;
}

// 構成が無誤差条件を満たすか検査する．満たさない条件があれば理由と処方を stderr に表示する（止めはしない．
// 既定のつまみでは ozaki_block_k が条件を満たす K ブロック長を選ぶので必ず通り，外れるのは条件を意図的に外す
// 診断のつまみ（docs/env_vars.md の C）を立てたときだけである）．
// 無誤差には二つの条件が要る（論文の ssec:exact（無誤差条件））ので，両方を見て別々に返す（呼び出し側の Exact 行が
// どちらで落ちたかを書き分ける．ozaki1_plan.hpp の plan_issue）：
//   (a) mul：選んだ形式とモードで桁どうしの積が無誤差に生成されること
//   (b) acc：その積和が累積器の無誤差整数領域に収まること
// radix は表示だけに使う（判定は桁の上限 dmax と MathFidelity fid で行う）．
struct OzExact {
    bool mul = true, acc = true;
    bool ok() const { return mul && acc; }
};
static inline OzExact ozaki_check_exact(uint32_t radix, uint32_t dmax, tt::tt_metal::MathFidelity fid, uint32_t K, int L) {
    using MF = tt::tt_metal::MathFidelity;
    OzExact r;
    // (a) 乗算側．LoFi の乗算器は SrcA の 5 bit しか読まないので桁が 32 を超えると
    // 積が丸まる．HiFi2 以上でも，FP 経路は K 方向 8 項ごとに積を組の中の最大の指数 E
    // （両オペランドの指数の和）に揃えて 2^{E-10} の位へ丸める（論文の「8 項の指数合わせ」）．
    // 桁の指数が 5 以下（d <= 63）なら E <= 10 で整数の積は失われないが，d >= 64（積が
    // 2^12 以上）では無誤差になるとは限らないので，桁 63 が上限である（ozaki1_param.h の
    // BF16／FP16／FP32 の説明と docs/fidelity_radix_sweep.md．指数合わせは FP 経路の性質なので，
    // 形式の表現限界が 63 を超える BF16・FP16・FP32・BFP8_b に同じく当てはまる）．OZ_RADIX で桁を 64 以上に
    // すると，どの fidelity でも無誤差にならない．
    // 桁 33..63 で HiFi2 が足りることは BF16（単一タイルで桁 63 が 1024/1024 一致，
    // 3520^3 の radix63・HiFi2 で 53.5 bit）と BFP8_b（同 53.5 bit）で実測してある．
    // FP16・FP32 も単一タイルでは HiFi2 で桁 63 まで無誤差になる（diagnose/probes.sh の probe_fp8_digit）が，
    // 全体の経路の HiFi2 は測っていないので，そちらは HiFi4 を要求したままにする．
    // 規則による自動選択（oz1_fidelity）は桁 33 以上では形式によらず HiFi4 を採る（安全側）．
    // INT8 経路は区画配置の都合で基数によらず HiFi4 を要する
    // （docs/int8_fidelity_mechanism.md）．規則はそれを選ぶが，OZ_FIDELITY で上書きされている場合はここで検出する．
    {
        // HiFi2 で桁 33..63 が足りると実測してあるのは BF16 と BFP8_b だけ
        constexpr bool is_int8   = (OZ_FMT_ID == OzFmt::INT8);
        constexpr bool hifi2_ok  = (OZ_FMT_ID == OzFmt::BF16 || OZ_FMT_ID == OzFmt::BFP8);
        const bool mul_ok = is_int8  ? (fid == MF::HiFi4)
                          : (dmax <= 32) ? true
                          : hifi2_ok ? (dmax <= 63 && fid != MF::LoFi)
                                     : (dmax <= 63 && fid == MF::HiFi4);
        const char* need  = is_int8  ? "HiFi4（INT8 経路は基数によらず要する）"
                          : hifi2_ok ? "桁 <= 32 なら LoFi 以上，33..63 なら HiFi2 以上．64 以上はどの fidelity でも不可"
                                     : "桁 <= 32 なら LoFi 以上，33..63 なら HiFi4（HiFi2 は未測定）．64 以上はどの fidelity でも不可";
        if (!mul_ok) {
            r.mul = false;
            fmt::print(stderr,
                "\n[WARN] 乗算が無誤差になりません: 桁の上限 {}・MathFidelity {}\n"
                "        要件は {} です．積が丸まるため到達精度が大きく落ちます．\n"
                "        この警告を出して続行する（条件を外す診断のつまみの構成．docs/env_vars.md の C）．\n\n",
                dmax, oz_fidelity_name(fid), need);
        }
    }
    // (b) 累積側．
    const uint64_t k_eff = (uint64_t)(L + 1) * K;
    const uint32_t rmax = ozaki_max_digit(k_eff);
    if (dmax <= rmax) return r;
    r.acc = false;
    fmt::print(stderr,
        "\n[WARN] 無誤差条件を満たしていません: radix={} K={} L={} (K_eff=(L+1)*K={})\n"
        "        累算器の予算 b_acc={} bit を b_prod+ceil(log2 K_eff) が超過します\n"
        "        （b_prod=ceil(log2 d^2) は桁積のビット数）．\n"
        "        この K で無誤差を保証できる最大の桁は {} です（基数はその 2 倍）．\n"
        "        乱数入力では符号の相殺で見かけ上通ることがありますが，非負行列のような\n"
        "        入力では部分和が累算器の整数域を出て精度が崩れます．\n"
        "        この警告を出して続行する（条件を外す診断のつまみの構成．docs/env_vars.md の C）．\n"
        "        本来は K をブロック化して 1 ブロックあたりの K を予算内に収める．\n\n",
        radix, K, L, k_eff, OZAKI_ACC_BITS, rmax);
    return r;
}

// 無誤差条件を満たす K ブロック長 K_c を決める（ozaki_max_digit の逆問題）．
// 条件 2w + ceil(log2((L+1)*K_c)) <= b_acc を満たす最大の K_c を求め，そこから
// 「K の約数かつタイル幅の倍数」である最大のものを選ぶ（ブロック長を揃えると
// 全ブロックで同じプログラムを使い回せる）．K 全体で条件を満たすなら K を返す
// （＝ブロック分割なし．INT8 経路は本稿のサイズでは常にこちら）．
// 取れない場合は 0 を返す．dmax は桁の上限．
// noblock（OZ_NOBLOCK=1）：ブロック分割を止めて K 全体を 1 本の GEMM に載せる．無誤差条件を意図的に破って
// 何が起きるかを見るための調査用スイッチで（非負入力の実験は docs/fidelity_radix_sweep.md の「NOBLOCK が速いのは」の節），
// 通常の実行では使わない．
static inline uint32_t ozaki_block_k(uint32_t dmax, uint32_t K, int L, bool noblock) {
    if (noblock) return K;
    const uint32_t pb = ozaki_prod_bits(dmax);   // 2w：スライス積のビット数
    if (OZAKI_ACC_BITS < pb + 1) return 0;            // この基数は累算器に載らない
    uint64_t keff_max = 1ull << (OZAKI_ACC_BITS - pb);
    // INT8 経路の部分積は符号-絶対値 Int32 で 2^31 ちょうどを表せない．d^2 が 2 のべき
    // （OZ_RADIX=64 の d=32 など）だと上の予算は 2^31 ちょうどを許すので，整数で詰める．
    // 既定の d=127 では INT32_MAX/d^2 のほうが大きく，値は変わらない．
    if (OZ1_INT32_PARTIAL) {
        const uint64_t d = dmax;
        keff_max = std::min<uint64_t>(keff_max, (uint64_t)INT32_MAX / (d * d));
    }
    const uint64_t kc_max = keff_max / (uint64_t)(L + 1);
    if (kc_max >= (uint64_t)K) return K;              // 分割不要
    uint32_t best = 0;
    for (uint32_t c = tt::constants::TILE_WIDTH; c <= kc_max; c += tt::constants::TILE_WIDTH)
        if (K % c == 0) best = c;
    return best;
}

// Horner の指数予算．tfMulR は上位 limb を Dekker 分割するために 4097 倍（2^12.0）
// するので，累積が 2^116 を超えると FP32（上限 2^128）を溢れる．基数が 2 のべきのときの
// tfMulP2 はこの 4097 倍をしないが，判定は基数によらずこの式のまま（安全側）にしてある．
// Horner（acc = P_0; acc = acc*radix + P_d，d = 1..L）で 4097 倍されるのは最後の基数倍の直前の値
// H_{L-1} = sum_{d<L} P_d radix^(L-1-d) である．|P_d| <= (d+1) K dmax^2（dmax = ceil(radix/2)）なので
//     |H_{L-1}| <= K dmax^2 sum_{d<L} (d+1) radix^(L-1-d) < K dmax^2 radix^(L+1)/(radix-1)^2 <= K radix^(L+1)
// （radix >= 2 で dmax <= radix-1．OZ_DMAX で桁の上限を ceil(radix/2) より上げた診断の構成は除く）．
// そこで条件を
//     log2(K) + (L+1) log2(radix) + 12 < 128
// とする．K radix^(L+1) は再構成に入る値そのものの上限ではない：基数倍した後の最終値 H_L は
// H_{L-1} のおよそ radix 倍（L=0 なら H_0 = P_0 の上限 K dmax^2 が，基数 64 で K radix の 16 倍）だが，
// この条件のもとで radix <= 4096 なら 2^128 未満に収まる．既定の構成（形式ごとの基数と p は scripts/routes.tsv，
// K<=14080）では大きく余裕があるが，p を手で増やすと判定に掛かる：基数 64・K=512 なら
// p=18 で左辺が 129 になる（2 のべきの基数は tfMulP2 なので，実際に溢れるのはもっと先である）．
// 無誤差累積の条件（oz_defer_fits）とは別物で，そちらを満たしても溢れる．
static inline bool oz_horner_fits(uint32_t radix, uint32_t K, int L) {
    const double v = std::log2((double)K) + (double)(L + 1) * std::log2((double)radix);
    return v + 12.0 < 128.0;
}

// Int32 1 語に全ブロックの和が収まるか．nkb 個のブロックを足した上限は
// nkb * (L+1) * K_c * d^2 = (L+1) * K * d^2 である（d は桁の上限 dmax で，基数ではない）．
// 符号つき Int32 の正側の上端は 2^31-1 なので，条件は (L+1)*K*d^2 <= 2^31-1 である．
// ビット数で 2w + ceil(log2((L+1)K)) <= 31 と書くと，等号のところで 2^31 ちょうどを
// 許してしまう（例：d=32・L=0・K=2097152 は 10+21=31 を満たすが総和は 2^31）ので，
// 整数のまま比較する．すなわち「ブロック化しない K に対する
// b_acc=31 の無誤差条件」そのものになる．
static inline bool oz_defer_fits(uint32_t dmax, uint32_t K, int L) {
    const uint64_t k_eff = (uint64_t)(L + 1) * K;
    const uint64_t dsq   = (uint64_t)dmax * dmax;
    // k_eff * dsq <= INT32_MAX を，掛け算で溢れさせずに判定する．
    return dsq != 0 && k_eff <= (uint64_t)INT32_MAX / dsq;
}
