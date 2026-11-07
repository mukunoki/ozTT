// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 尾崎スキーム I：形式ごとの既定の基数とスライスの符号化の表，桁の上限の規則（デバイス API に依存しない）
// ---------------------------------------------------------------------------
// ドライバ（ozaki1_format.hpp の OZ1_PARAM・ozaki_digit_max）とホスト単体の検査・模擬（tools/oz_sim.h）が
// 同じ表を見るよう，tt-metal を取り込まない形でここに 1 つだけ置く（取り込むのは common/oz_format_host.h だけ）．
#include <cstdint>
#include "common/oz_format_host.h"

// scheme I の形式ごとの値（Oz1Param）．スライス／再構成のデバイスカーネル自体は形式に
// 依存せず，形式で変わるのはこの表と OZ_FMT だけである．
// 桁の上限 d と基数（重み）beta は別の量で，beta = 2d と取る（下の oz1_digit_max）．
//   bf16: 入力 Float16_b，部分積 Float32，既定 radix 64（桁 32．形式の上限は 256）
//   fp16: 入力 Float16（スライスは binary16 のビット列を UInt16 で出す），部分積 Float32，radix 64
//   fp32: 入力 Float32（行列エンジンには TF32 として渡る．論文での名前は TF32），
//         部分積 Float32，radix 64（1 要素 4 B）
//   bfp8: 入力 Bfp8_b   ，部分積 Float32，既定 radix 64（桁 32．形式の上限は 127）
//   fp8 : 入力 Fp8_e4m3 ，部分積 Float32，既定 radix 32（桁 16）
//   bfp4: 入力 Bfp4_b   ，部分積 Float32，既定 radix 14（桁 7）
//   bfp2: 入力 Bfp2_b   ，部分積 Float32，既定 radix 2 （桁 1）
//   int8: 入力 Int8     ，部分積 Int32（符号-絶対値のバイトスライス，
//         再構成側で FP32 2 本に復号），既定 radix 254（桁 127）
// fidelity は表に無い：桁の上限から ozaki1_format.hpp の oz1_fidelity が決める（規則は oz_format.hpp の
// oz_fidelity_rule．桁 <= 32 なら LoFi，超えれば HiFi4，INT8 は常に HiFi4）ので，既定では
// bf16/fp16/fp32/bfp8/fp8/bfp4/bfp2 が LoFi，int8 だけが HiFi4 になる．
// 形式の exact_int_max（common/oz_format_host.h）は「桁 |a| <= この値をその形式が無誤差に表せる」上限である
// （基数の上限ではない．基数はここから beta = 2d と決まる）．
// これを超えると値が黙って別の数に化ける（累算側の予算は ozaki_block_k が別途見る）．
struct Oz1Param {
    uint32_t radix;          // 既定の基数（桁の重み）．OZ_RADIX で上書きできる
    // スライスを符号-絶対値の UInt8 で書くか（INT8 経路だけ．スライサは ozaki1/kernels/slice_tf_sm8.cpp）．ほかの
    // 7 形式は桁の値そのものを GEMM の入力形式で書く（slice_tf_num.cpp．ozaki1_slice.hpp の slice_build）．再構成の
    // カーネルは形式ではなく部分和の形式で選ぶ（ozaki1_recon.hpp の recon_build_prog）
    bool signmag_slices;
};
constexpr Oz1Param oz1_param(OzFmt f) {
    switch (f) {
    // 桁の重み 254（桁自体は ±127．下の oz1_digit_max）．スライサは符号-絶対値の UInt8 で
    // 書き（slice_tf_sm8.cpp），部分積は Int32 で，再構成が無誤差な FP32 2 本に復号する
    // （recon_tf_i32in.cpp を SIGNMAG=1 で）
    case OzFmt::INT8: return {254, true};
    case OzFmt::FP8:  return {32,  false};
    // 基数 64・桁 ±32（積の絶対値は 2^10 以下）．桁が 32 以下なので
    // LoFi の乗算器（SrcA 5 bit）に収まり，無誤差条件 2w+ceil(log2 K) <= 24 は w=5 で
    // K <= 2^14 まで満たす（14080 はその内側）．基数 32 から 64 へ上げると桁の上限も 2 倍に
    // なって累積の予算を 2 bit 多く使うが，1 スライスあたりの取り出しが 1 bit 増えるので，
    // スライス数と部分積の本数が減る（基数を選び直したときの p と本数・実測は docs/kblock_plan.md の
    // 「基数の選び直し」，今の p の正本は scripts/routes.tsv，今の性能は docs/results.md の表）．
    case OzFmt::BF16: return {64,  false};
    // IEEE binary16．BF16 経路と基数・スライス数・部分積の本数がすべて同じで，運ぶ形式だけが違う．
    // 形式が無誤差に表せる整数は 2048 までと BF16 の 256 より広いが，桁の上限を
    // 決めるのは形式ではなく乗算器の区画（LoFi で 32）と累積幅なので，桁も基数も BF16 と同じ
    // d <= 32・radix 64 になる．1 要素 2 B も BF16 と同じ．つまり BF16 との差は
    // 仮数長だけで，それが性能にも精度にも現れないことを確かめる経路である（論文「格納形式だけを
    // 変えた比較」）．
    // 11 bit の仮数を使い切る値（1025 のような整数も含む）の積は HiFi4 でも無誤差でない
    // （common/oz_format_host.h の FP16 の項）．掛けられる桁の上限は BF16 と同じく LoFi 32・HiFi4 63 で
    // （docs/fidelity_radix_sweep.md），既定の桁 ±32 はその内側にある．OZ_RADIX で桁を 63 より
    // 広げる設定は，BF16 でも FP16 でも壊れる（ozaki_check_exact が警告する）．
    case OzFmt::FP16: return {64,  false};
    // FP32 で格納し，行列エンジンには TF32 として渡す経路（論文での名前は TF32．common/oz_format_host.h）．
    // 桁の上限も基数もスライス数も BF16 と同じで，1 要素が 4 B になり，DRAM・L1 で扱う量が
    // BF16 の 2 倍になる．1 タイル積の展開も BF16 の約 1.8 倍のサイクルを要する（単コアの実測．
    // docs/gemm_stage_bottleneck.md の §2．同 §1 の「供給上限」のモデルは撤回した）．
    // exact_int_max の意味と，無誤差に掛けられる桁の上限（LoFi 32・HiFi4 63）との関係は FP16 と同じである．
    case OzFmt::FP32: return {64,  false};
    // BF16 経路と基数・スライス数・累算器は同じで，スライスを運ぶ形式だけ block-float 8 に
    // 変えたもの．スライスは整数なので 16 要素が指数を共有しても厳密に表せる
    //   （共有指数は面内の最大値から決まり，量子化幅は 2 のべき．整数 0..127 はどの面でも厳密）
    // 要素あたり 2 B -> 1.0625 B になるので，DRAM・NoC で運ぶ量と L1 の入力 CB がほぼ半分になり，
    // 同じ L1 の予算で bw を BF16 の 11 から 22 へ広げられる．単コアの 1 タイル積のサイクル数は
    // BF16 とほぼ同じである（docs/gemm_stage_bottleneck.md の §2）．何が変わって何が変わらないかは
    // 論文の ssec:bw（データ移動の削減）と ssec:fmtcmp（格納形式の比較）．
    case OzFmt::BFP8: return {64,  false};
    // block-float 4．0.5625 B/要素と BFP8_b よりさらに狭く，DRAM・L1 で扱う量が減る．
    // 代償は表せる整数の狭さで，仮数 3 bit なので基数は小さくなり，
    // スライス数 p が増える．重み 14（桁は ±7．INT8 と同じ分け方）．
    // 基数 15 で桁の上限を 8 にすると（common/oz_format_host.h の BFP4 の exact_int_max（7）を一時的に上げたとき），BFP4_b が
    // 表せない ±8 の桁が出て，到達精度は 2.9 bit まで落ちる（実測）．
    case OzFmt::BFP4: return {14,  false};
    // block-float 2．0.3125 B/要素と，本ハードウェアが入力に取れる最も狭い形式である．
    // 共有指数のもとで厳密に表せる整数は [-1,1] しかないので，重み 2・桁 ±1 の符号つき
    // 2 進展開で，1 スライスが 1 bit しか取り出せない．
    // FP64 相当に要る p は 1 スライス 1 bit なので 53 を超え（値の正本は scripts/routes.tsv．構造のある入力の組でも
    // 53 bit に届く値．scripts/diagnose/select_slice_count.sh），部分積 p(p+1)/2 は BF16 の数十倍の本数になる（8197a0b の routes.tsv では
    // p=59 で 1770 本，BF16 は 55 本）．要素が狭くても本数の差が大きく，全体では大きく負ける．それを実測で示すための経路である．
    case OzFmt::BFP2: return {2,   false};
    }
    return {};
}

// 桁の上限の既定の規則：min(ceil(radix/2), 形式の上限 exact_int_max)．dmax_override（OZ_DMAX．0 = 未設定）は
// ceil(radix/2) の代わりに使う（調査用．形式の上限は超えない．ozaki1_format.hpp の ozaki_digit_max）．
// 桁の重み（基数 radix）と，1 桁が取りうる絶対値の上限は別の量である．INT8 経路は
// 符号-絶対値バイトの絶対値部が 7 bit しか無いので桁は ±127 に収まり，下の規則から
// 重みは 254 になる（先行実装 ozIMMU は 2^7 を重みに取り，桁を 7 bit に収める）．無誤差条件や fidelity を決めるのは乗算器が実際に
// 受け取る桁の値なので，以下ではこの上限で数える．
// スライサは最近接丸めで符号つきの桁を作る（ozaki1/kernels/slice_tf_num.cpp・slice_tf_sm8.cpp の
//   t *= radix ; d = clamp(round(t)) ; t -= d
// ）ので，桁は平衡表現になり ceil(radix/2) まで出る．残差は原理上 ±1/2 に収まるが，
// 上限を (radix-1)/2 に取ると範囲 R = 桁/(基数-1) が正規化の目標 T = 1/2 と等しくなって余裕が 0 になり，
// K が大きいときに到達精度が落ちる（radix 65・桁 32 は K=1792・4096・16384 で 40.3・35.3・34.7 bit，
// 桁 33 なら試した K=512〜16384 で 54.2 bit．M=N=512・p=10．docs/fidelity_radix_sweep.md）．
// triple-float の上位 1 語だけを丸めることで clamp が発火して残差が縮まなくなると考えられるが，
// 境界を超える要素の数は測っておらず，原因は特定していない．したがって桁の上限は ceil(radix/2)
// であり，逆に形式が表せる整数 D に対して採れる基数は 2D である．
constexpr uint32_t oz1_digit_max(uint32_t radix, uint32_t exact_int_max, uint32_t dmax_override = 0) {
    const uint32_t half = dmax_override ? dmax_override : (radix + 1) / 2;   // ceil(radix/2)
    return half < exact_int_max ? half : exact_int_max;
}
// 形式 f の既定の基数での桁の上限（OZ_DMAX なし）
constexpr uint32_t oz1_default_digit_max(OzFmt f) { return oz1_digit_max(oz1_param(f).radix, oz_fmt_host(f).exact_int_max); }

// 遅延再構成の Int32 累積 S_d のうち，先頭から何本が FP32 に無誤差に載るか（ozaki1_plan.hpp の recon_nsmall と同じ式．
// 導き方はそちらの注記）：|S_d| <= (d+1) * K * d_max^2 が 2^24 以下の対角は先頭から連続し，その本数は
// floor(2^24 / (K d_max^2)) と ndiag の小さいほう．tt-metal に依存しないここに置き，ホストの検査（tools/tfmul_check.cpp の [3]）も
// 同じ関数を呼ぶ（写しを持たない）
constexpr uint32_t oz1_recon_nsmall(uint32_t dmax, uint32_t K, uint32_t ndiag) {
    const uint64_t dm = dmax;
    const uint64_t per = (uint64_t)K * dm * dm;          // 対 1 つぶんの上界 K*d_max^2
    if (per == 0) return 0;
    const uint64_t n = (1ull << 24) / per;
    return (uint32_t)(n < ndiag ? n : ndiag);
}
