// SPDX-License-Identifier: Apache-2.0
//
// 行列エンジンの入力形式 × MathFidelity × 桁の上限で，整数スライスの積が無誤差になるかを測る
// 【探針．本番の 16 経路のバイナリではない】remeasure_all.sh の probes の段（scripts/diagnose/probes.sh の probe_fp8_digit・
// probe_fp8_rand）が回し，論文・スライドの値（無誤差の境界・乱数入力の到達精度・パッカの丸め）を出す．
// ---------------------------------------------------------------------------
// 単一コアで A（Mt×Kt タイル）と B（Kt×Nt タイル）の行列積を 1 回流し，ホストで計算した
// 参照値と比べる．調べたいのは 1 回の積和が丸められるかどうかなので，並列性能は見ない．
// 計算カーネルは tt-metal 付属の単一コア版（matmul_single_core）をそのまま使い，
// 変えるのは入力形式・出力形式・math_fidelity・DST の形式だけである．
//
// モード（第 3 引数）
//   整数 R      : A・B に [0,R] の整数を入れ，整数として計算した C=A*B と全要素を比べる．
//                 一致すれば「この試行では全要素が一致した」と報告する（対象範囲の全入力に
//                 対する保証ではない．境界は R を振って探す）．入力は形式へ丸めずに流すので，
//                 R は形式が [0,R] の整数をすべて無誤差に表せる範囲（common/oz_format_host.h の exact_int_max．
//                 FP8 E4M3 は 16，BF16 は 256，FP16 と TF32 は 2048）に限る．超えると入力の丸めが積や累算の丸めと
//                 区別できないので止める．出力 fp32 で不一致があれば終了コード 1 を返す．
//   rand        : [-1,1] の一様乱数を入力形式へ丸めて流し，その値の高精度（double）な積に
//                 対する相対 Frobenius 誤差を報告する（無誤差性ではなく到達精度）．
//                 fp32 入力では参照を FP32 の値で作るので，アンパッカが TF32 へ縮める分も
//                 誤差に入る（「FP32 を入れたら FP32 の積に対してどこまで合うか」を見る）．
//   randbf16    : 乱数を BF16 へ丸めてから bf16/fp16/fp32 のいずれかに格納する．3 形式が
//                 同じ値を持つので，格納形式の違いだけを共通の参照に対して比べられる
//                 （FP16 の非正規域で仮数が欠けるなど，FP16 との往復で値が変わる値が
//                 出た場合は止める）．
//   randtf32    : 乱数を TF32（明示仮数 10 bit，暗黙の 1 を含めて 11 bit）へ丸めてから fp32 に
//                 格納する．アンパッカが FP32 を TF32 へ縮めても値が変わらない入力なので，
//                 入力の量子化を除いた演算経路（乗算器の区画と累算の整列）の精度が見える．
//   randa10 / randb10 : 行列 A または B の片側だけ有効仮数 10 bit（暗黙の 1 を含む．TF32 の
//                 最下位 bit を 0 にした値）へ丸め，もう片側は TF32 のまま流す．乗算器の区画が
//                 SrcA の bit 0 を読まないことの切り分け用．行列積では A（in0）が SrcB，B（in1）が
//                 SrcA へ展開される（llk_unpack_AB_matmul_api.h "In0 -> srcB"）ので，意味を持つのは randb10．
// 出力形式（第 5 引数）を fp32 以外にすると，パッカが DST からその形式へ正しく
// 書けるかを見る．その場合は参照を出力形式へ丸めたものとも比べ，整数積の判定とは分ける．
// 整数モードで出力が 16 bit のとき，3 つの丸め方（最近接偶数・最近接で同点は 0 から遠い側・切り捨て）のどれとも
// 全要素が一致しなければ終了コード 1（積和が丸まったか，パッカが知らない変換をした．どちらでもパッカの判定は成り立たない）．
// DST の形式は既定で FP32（fp32_dest_acc_en）．環境変数 PROBE_FP32ACC=0 で 16 bit の DST にする（値は 0 か 1）．
// 診断の表示：PROBE_DUMP=1 で出力の先頭の語を，PROBE_MISMATCH=1 でパッカの検査の不一致の中身を出す（どちらも 0 か 1）．
//
// 測定結果と機構の説明は docs/fidelity_radix_sweep.md と docs/exponent_format_constraint.md に
// ある（コードに書くと実装の変更で古くなるため）．
//
// 使い方: fp8_probe <入力形式 bf16|fp16|fp32|fp8> <fidelity 1..4> <R|rand|randbf16|randtf32|randa10|randb10> [Kt>=1] [出力形式 fp32|bf16|fp16]
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include "common/host_eval.hpp"
#include "testkit/probe_common.hpp"
#include "common/oz_bits.h"
#include "common/oz_knobs.hpp"   // PROBE_FP32ACC・PROBE_DUMP・PROBE_MISMATCH（つまみの表）
#include "common/oz_fidelity.hpp"   // fidelity の番号と表示名（ドライバの CONFIG 行と同じ表）
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// 行優先の並びをタイル順（tt-metal の tilize_nfaces：タイル行→タイル列→16x16 の面 4 つ（左上・右上・左下・右下）→
// 面内の行優先．出力の読み戻しの untilize_nfaces と同じ並び）へ詰め替える．tilize_nfaces は float・bfloat16・uint16_t・
// uint32_t でだけ実体化されていて，FP8 の 1 バイトの要素（uint8_t）には使えないので，要素の番号（uint32_t）を
// tilize_nfaces で並べ替えた順を作り，それをどの型にも当てる（並べ方を自前に持たない）．
template<class T> static std::vector<T> to_tiles(const std::vector<T>& lin, uint32_t rows, uint32_t cols){
  std::vector<uint32_t> id(lin.size());
  std::iota(id.begin(), id.end(), 0u);
  const std::vector<uint32_t> perm = tilize_nfaces(id, rows, cols);
  std::vector<T> out(lin.size());
  for(size_t i=0;i<out.size();i++) out[i]=lin[perm[i]];
  return out;
}

static void usage_and_exit(const char* why){
  std::fprintf(stderr,"%s\n使い方: fp8_probe <bf16|fp16|fp32|fp8> <fidelity 1..4> <R|rand|randbf16|randtf32|randa10|randb10> [Kt>=1] [fp32|bf16|fp16]\n",why);
  std::exit(2);
}

// ---- 引数（未知の値は既定に落とさず，止める）----
struct Config {
  std::string fmt, mode, ofmt;        // 入力形式・モード・出力形式の名前（表示と PROBE 行に使う）
  DataFormat in_fmt, out_fmt;
  long fid; tt::tt_metal::MathFidelity mf;
  bool rand_a10, rand_b10, randmode, round_input_to_bf16, round_input_to_tf32;
  long R;                             // 整数モードの上限（乱数モードは 0）
  uint32_t Kt, Mt, Nt, M, N, K;
  bool fp32acc;                       // DST が FP32 か（PROBE_FP32ACC）
  std::string mode_label() const { return randmode? mode : "R="+std::to_string(R); }
};

// 形式の名前から形式の表（common/oz_format_table.hpp の oz_format）を引く．allowed にない名前は止める
static OzFormat format_of(const std::string& name, std::initializer_list<const char*> allowed, const char* why){
  bool ok=false; for(const char* a: allowed) ok |= name==a;
  const auto f = probe::probe_format(name);
  if(!ok || !f) usage_and_exit(why);
  return oz_format(*f);
}

static Config parse_args(int argc, char** argv){
  if(argc<4) usage_and_exit("引数が足りない");
  // 余分な位置引数は黙って読み捨てない（引数の追加・位置ずれで，既定の構成のまま測ったログが残らないように）
  if(argc>6) usage_and_exit("引数が多すぎる（5 個まで）");
  Config c;
  c.fmt=argv[1]; c.mode=argv[3]; c.ofmt= argc>5? argv[5] : "fp32";
  const OzFormat fi = format_of(c.fmt, {"bf16","fp16","fp32","fp8"}, "入力形式が不明");
  c.in_fmt = fi.in;
  // 数の引数は oz_env の読み方（10 進の整数として全体を読み切り，範囲の外は理由を出して 2 で止める）
  c.fid=(long)oz_env::arg_i64(argc,argv,2,0,1,4,"fidelity");
  const auto mf=oz_fidelity_by_passes(c.fid);   // 1..4 = LoFi・HiFi2・HiFi3・HiFi4（パス数）
  if(!mf) usage_and_exit("fidelity は 1..4");
  c.mf=*mf;
  c.rand_a10 = (c.mode=="randa10"); c.rand_b10 = (c.mode=="randb10");
  c.randmode = (c.mode=="rand"||c.mode=="randbf16"||c.mode=="randtf32"||c.rand_a10||c.rand_b10);
  c.round_input_to_bf16 = (c.mode=="randbf16");
  c.round_input_to_tf32 = (c.mode=="randtf32"||c.rand_a10||c.rand_b10);
  if(c.round_input_to_bf16 && c.in_fmt==DataFormat::Fp8_e4m3) usage_and_exit("randbf16 は bf16/fp16/fp32 用（fp8 は BF16 の値を保てない）");
  if(c.round_input_to_tf32 && c.in_fmt!=DataFormat::Float32) usage_and_exit("randtf32/randa10/randb10 は fp32 用");
  c.R=0;
  if(!c.randmode) c.R=(long)oz_env::arg_i64(argc,argv,3,0,0,1L<<24,"R（rand 系のモード名でなければ 0..2^24 の整数）");
  // [0,R] の整数をすべて無誤差に表せる上限は形式の表の exact_int_max（暗黙の 1 を含む仮数幅 t に対して 2^t）．
  // fp32 はアンパッカが TF32（仮数 11 bit）へ縮めるので FP32 ではなく TF32 で数える（表の FP32 の項と同じ理由）．
  const long R_exact = (long)fi.exact_int_max;
  if(!c.randmode && c.R>R_exact){
    std::fprintf(stderr,"R=%ld は %s が無誤差に表せる整数の上限 %ld を超える（入力の丸めと積和の丸めが区別できない）\n",
                 c.R, c.fmt.c_str(), R_exact);
    usage_and_exit("R が形式の表せる範囲を超える");
  }
  c.Kt=(uint32_t)oz_env::arg_i64(argc,argv,4,1,1,4096,"Kt"); c.Mt=1; c.Nt=1;   // M,N は 1 タイルで十分
  c.M=c.Mt*32; c.N=c.Nt*32; c.K=c.Kt*32;
  c.out_fmt = format_of(c.ofmt, {"fp32","bf16","fp16"}, "出力形式が不明").in;
  // 整数モードの参照は double で累算する．各要素は高々 K*R^2 なので，これが 2^53 に
  // 収まることを条件にする（収まらなければ参照が無誤差でない）．
  if(!c.randmode && (long double)c.K*(long double)c.R*(long double)c.R > 9007199254740992.0L) usage_and_exit("K*R^2 が 2^53 を超える");
  // 既定は FP32 の DST．PROBE_FP32ACC=0 で 16 bit の DST にして，パッカの挙動の違いを見る（0・1 以外は止める）
  c.fp32acc = oz_knobs::u32(OZK("PROBE_FP32ACC")) != 0;
  return c;
}

// ---- 入力 A(M×K), B(K×N)（行優先の float）----
// 整数モードは [0,R] の整数のみ．負や小数を混ぜると「丸めのせいか値域のせいか」が
// 切り分けられなくなる．乱数モードはデバイスが実際に見る値（入力形式へ丸めた値）を
// 参照にも使う．測りたいのは演算経路の精度であって，入力の丸めではない．
static void make_inputs(const Config& c, std::vector<float>& Af, std::vector<float>& Bf){
  std::mt19937 rng(3); std::uniform_int_distribution<long> U(0,c.R); std::uniform_real_distribution<float> UR(-1.f,1.f);
  auto round_sig=[](float v,int keep)->float{ return f32_round_to_prec(v,keep+1); };   // 明示仮数を keep bit に最近接偶数丸め（oz_bits.h）
  auto to_tf32=[&](float v)->float{ return round_sig(v,10); };
  auto quantize=[&](float v)->float{
    if(c.round_input_to_bf16) return (float)bfloat16(v);      // 3 形式とも BF16 の値（fp16/fp32 で厳密に表せる）
    if(c.round_input_to_tf32) return to_tf32(v);
    switch(c.in_fmt){
      case DataFormat::Float16_b: return (float)bfloat16(v);
      case DataFormat::Float16:   return f32_from_half(half_from_f32(v));
      case DataFormat::Fp8_e4m3:  return (float)float8_e4m3(v);
      default:                    return v;
    }
  };
  Af.assign(c.M*c.K,0.f); Bf.assign(c.K*c.N,0.f);
  for(auto&x:Af)x= c.randmode? quantize(UR(rng)) : (float)U(rng);
  for(auto&x:Bf)x= c.randmode? quantize(UR(rng)) : (float)U(rng);
  // randbf16 を fp16 に格納するときは，値が FP16 で厳密に表せることを確かめる
  // （BF16 は指数が広いので，|v| < 2^-14 の非正規域では仮数が欠けうる．2^-15 のように
  // 欠けない値もあるので，範囲ではなく往復で判定する）．
  if(c.round_input_to_bf16 && c.in_fmt==DataFormat::Float16)
    for(const auto* v : {&Af,&Bf}) for(float x : *v)
      if(f32_from_half(half_from_f32(x))!=x){ std::fprintf(stderr,"randbf16: FP16 で表せない値 %g\n",x); std::exit(2); }
  // 片側だけ有効仮数 10 bit（明示 9 bit）にする：TF32 の最下位 bit を 0 にした値．
  // TF32 へ丸めた値をもう一度丸めるので直接の丸めと 1 ulp 違いうるが，調べたいのは
  // 「最下位 bit が 0 の値」を流すことなので構わない．
  if(c.rand_a10) for(auto&x:Af) x=round_sig(x,9);
  if(c.rand_b10) for(auto&x:Bf) x=round_sig(x,9);
}

// ---- 入力形式へ符号化し，タイル順へ並べ替えて uint32 語に詰める（語への詰め方は probe_common.hpp の to_words）----
static std::vector<uint32_t> encode_tile_words(const Config& c, const std::vector<float>& v, uint32_t rows, uint32_t cols){
  const size_t n=v.size();
  switch(c.in_fmt){
    case DataFormat::Float32: return probe::to_words(to_tiles(v,rows,cols));
    case DataFormat::Float16_b: { std::vector<bfloat16> l(n); for(size_t i=0;i<n;i++) l[i]=bfloat16(v[i]);
      return probe::to_words(to_tiles(l,rows,cols)); }
    case DataFormat::Float16: { std::vector<uint16_t> l(n); for(size_t i=0;i<n;i++) l[i]=half_from_f32(v[i]);
      return probe::to_words(to_tiles(l,rows,cols)); }
    default: { std::vector<uint8_t> l(n); for(size_t i=0;i<n;i++) l[i]=float8_e4m3(v[i]).to_bits();
      return probe::to_words(to_tiles(l,rows,cols)); }
  }
}

// ---- 単一コアの行列積を 1 回流す（組み立ては probe_common.hpp の run_single_core_mm）----
static std::vector<uint32_t> run_device(std::shared_ptr<distributed::MeshDevice>& md, const Config& c,
                                        const std::vector<uint32_t>& Aw, const std::vector<uint32_t>& Bw){
  probe::SingleCoreMM mm;
  mm.in_fmt=c.in_fmt; mm.out_fmt=c.out_fmt; mm.Mt=c.Mt; mm.Kt=c.Kt; mm.Nt=c.Nt;
  mm.ccfg=ComputeConfig{.math_fidelity=c.mf, .fp32_dest_acc_en=c.fp32acc, .compile_args={c.Mt,c.Kt,c.Nt}};
  return probe::run_single_core_mm(md, mm, Aw, Bw);
}

// ---- 読み戻し：出力形式から float へ，タイル順から行優先へ ----
static std::vector<float> decode_output(const Config& c, const std::vector<uint32_t>& Cw){
  if(oz_knobs::u32(OZK("PROBE_DUMP"))){                                // 先頭の語をそのまま見る（符号化の確認用）
    std::printf("raw (%s, tile order):", c.ofmt.c_str());
    if(c.out_fmt==DataFormat::Float32) for(int i=0;i<8;i++) std::printf(" %08x",Cw[i]);
    else { uint16_t h[8]; std::memcpy(h,Cw.data(),sizeof h); for(int i=0;i<8;i++) std::printf(" %04x",h[i]); }
    std::printf("\n");
  }
  std::vector<float> Ct(c.M*c.N);
  if(c.out_fmt==DataFormat::Float32) Ct=probe::from_words<float>(Cw,(size_t)c.M*c.N);
  else {
    const std::vector<uint16_t> h=probe::from_words<uint16_t>(Cw,(size_t)c.M*c.N);
    for(size_t i=0;i<(size_t)c.M*c.N;i++){
      if(c.out_fmt==DataFormat::Float16) Ct[i]=f32_from_half(h[i]);
      else Ct[i]=bits_f32((uint32_t)h[i]<<16);
    }
  }
  return untilize_nfaces(Ct,c.M,c.N);
}

// ---- パッカの検査（整数モードで出力が 16 bit のとき）----
// 積和が正確でも書き出しで整数を表せないことがあるので，その丸めを期待どおりに受けた参照とも比べ，整数積の判定と
// パッカの判定を分ける．パッカの丸め方向は文書に無いので，3 種の参照（最近接・同点は偶数（IEEE），最近接・同点は 0 から
// 遠い側，切り捨て）を用意してどれに一致するかを報告する（どれにも一致しなければ変換そのものが疑わしい）．
// BF16 の丸めは自前で行う（tt-metal の bfloat16(float) の丸め方向に依存しないため）．
// status：nonfinite（非有限の出力がある），ambiguous（複数の方式に全要素が一致して区別できない），nearest-even・
// nearest-away・truncate（その方式だけに一致），none（どれとも一致しない），n/a（検査しない構成）
struct PackCheck { std::string status="n/a"; uint32_t me=0, ma=0, mz=0; };
static PackCheck pack_check(const Config& c, const std::vector<float>& Cu, const std::vector<double>& Cg, uint32_t nonfinite){
  const uint32_t tot=c.M*c.N;
  // bf16_even は f32_round_to_prec(f, 8) と同じ値だが，3 つの丸め方を同じ形で並べて比べられるようにビット列で書く
  auto bf16_even =[](float f)->float{ uint32_t b=f32_bits(f); b+=0x7FFFu+((b>>16)&1u); b&=0xFFFF0000u; return bits_f32(b); };
  auto bf16_away =[](float f)->float{ uint32_t b=f32_bits(f); b+=0x8000u; b&=0xFFFF0000u; return bits_f32(b); };
  auto bf16_trunc=[](float f)->float{ return bits_f32(f32_bits(f) & 0xFFFF0000u); };
  auto half_away =[](float f)->float{ uint32_t b=f32_bits(f); const uint32_t e=(b>>23)&0xFF;
    if(e>=127-14 && e<=127+15){ b+=0x1000u; b&=~0x1FFFu; }                    // 正規化数の範囲だけ同点を遠い側へ
    return f32_from_half(half_from_f32(bits_f32(b))); };
  auto half_trunc=[](float f)->float{ return f32_from_half(half_from_f32(bits_f32(f32_bits(f) & ~0x1FFFu))); };
  std::vector<double> Ce(tot), Ca(tot), Cz(tot);
  for(uint32_t g=0;g<tot;g++){
    const float v=(float)Cg[g];
    if(c.out_fmt==DataFormat::Float16){ Ce[g]=f32_from_half(half_from_f32(v)); Ca[g]=half_away(v); Cz[g]=half_trunc(v); }
    else { Ce[g]=bf16_even(v); Ca[g]=bf16_away(v); Cz[g]=bf16_trunc(v); }
  }
  auto matches=[&](const std::vector<double>& ref){ return (uint32_t)(tot-oz_eval::exact_check(Cu,ref).mismatches); };
  PackCheck p; p.me=matches(Ce); p.ma=matches(Ca); p.mz=matches(Cz);
  if(oz_knobs::u32(OZK("PROBE_MISMATCH"))){                         // 不一致の中身を見る（診断用）
    int shown=0; for(uint32_t g=0;g<tot && shown<8;g++) if(Cu[g]!=Ce[g]){
      std::printf("    [%u] device=%.9g exact=%.9g nearest=%.9g trunc=%.9g\n",g,(double)Cu[g],Cg[g],Ce[g],Cz[g]); shown++; }
  }
  // 複数の方式に全要素が一致したら，この入力では区別できない（例：出力が全部 0 や，
  // 丸めの要らない値だけのとき）．最初の候補に断定せず，一致した方式を並べて報告する．
  std::string both;
  if(p.me==tot) both+=" nearest-even";
  if(p.ma==tot) both+=" nearest-away";
  if(p.mz==tot) both+=" truncate";
  const int nfull=(p.me==tot)+(p.ma==tot)+(p.mz==tot);
  p.status = nonfinite ? "nonfinite" : nfull>1 ? "ambiguous" : p.me==tot ? "nearest-even" : p.ma==tot ? "nearest-away"
           : p.mz==tot ? "truncate" : "none";
  std::printf("  pack check: match vs nearest-even %u/%u, nearest-away %u/%u, truncate %u/%u  => %s%s\n", p.me,tot, p.ma,tot, p.mz,tot,
              nonfinite ? "non-finite output" : nfull>1 ? "inputs do not distinguish; consistent with:" : p.me==tot ? "packer rounds to nearest even" : p.ma==tot ? "packer rounds to nearest, ties away from zero"
              : p.mz==tot ? "packer truncates" : "packer output matches none of the three roundings",
              (!nonfinite && nfull>1) ? both.c_str() : "");
  return p;
}

int main(int argc, char** argv){
  const Config c=parse_args(argc,argv);
  std::vector<float> Af, Bf;
  make_inputs(c,Af,Bf);
  std::vector<double> Cg(c.M*c.N,0.0);                         // 参照：double で累算した C=A*B
  for(uint32_t i=0;i<c.M;i++)for(uint32_t k=0;k<c.K;k++){double a=Af[i*c.K+k];for(uint32_t j=0;j<c.N;j++)Cg[i*c.N+j]+=a*Bf[k*c.N+j];}
  const auto Aw=encode_tile_words(c,Af,c.M,c.K), Bw=encode_tile_words(c,Bf,c.K,c.N);

  auto md = distributed::MeshDevice::create_unit_mesh(0);
  const auto Cw = run_device(md,c,Aw,Bw);
  const auto Cu = decode_output(c,Cw);

  // --- 比較 ---
  const uint32_t tot=c.M*c.N;
  uint32_t nonfinite=0; for(uint32_t g=0;g<tot;g++) if(!std::isfinite(Cu[g])) nonfinite++;
  auto ex=oz_eval::exact_check(Cu,Cg); const uint32_t exact=(uint32_t)(tot-ex.mismatches);
  double maxabs=0; for(uint32_t g=0;g<tot;g++){ double d=std::abs((double)Cu[g]-Cg[g]); if(std::isnan(d)) d=INFINITY; maxabs=std::max(maxabs,d); }
  std::printf("fmt=%s out=%s fidelity=%s(pass=%ld) mode=%s Kt=%u  M=%u N=%u K=%u fp32acc=%d\n",
              c.fmt.c_str(), c.ofmt.c_str(), oz_fidelity_name(c.mf), c.fid, c.mode_label().c_str(), c.Kt, c.M,c.N,c.K, c.fp32acc ? 1 : 0);
  if(nonfinite) std::printf("  非有限の出力が %u 要素ある（検査失敗）\n",nonfinite);
  double rel=std::nan(""), bits=std::nan("");                   // 乱数モードの相対 Frobenius 誤差とそのビット数
  if(c.randmode){
    double num=0,den=0; for(uint32_t g=0;g<tot;g++){ double d=(double)Cu[g]-Cg[g]; num+=d*d; den+=Cg[g]*Cg[g]; }
    if(den==0) std::printf("  参照が 0 なので相対誤差を定義できない\n");
    else { rel=std::sqrt(num/den); bits= rel==0 ? INFINITY : -std::log2(rel);   // 非有限の出力で rel が nan なら bits も nan
      if(rel==0) std::printf("  rand mode: rel Frobenius err=0 (exact)\n");
      else       std::printf("  rand mode: rel Frobenius err=%.3e  => %.1f bits\n", rel, bits); }
  }
  PackCheck pk;
  if(!c.randmode){
    std::printf("  integer ref: match %u/%u (%.1f%%)   max|err|=%.6g\n", exact,tot, 100.0*exact/tot, maxabs);
    if(c.out_fmt==DataFormat::Float32){
      std::printf("  => %s\n", (exact==tot && !nonfinite)
        ? "BIT-EXACT in this trial (no mismatch seen for these inputs)" : "NOT exact (a product or an accumulation was rounded)");
    } else {
      pk=pack_check(c,Cu,Cg,nonfinite);
    }
  }
  // 機械が読む 1 行（scripts/diagnose/probes.sh のログを scripts/rules/probe_checks.py が読む．verify/check_paper_numbers.py が取り込む）．
  // chash は出力の語列（タイル順）のハッシュ（oz_accuracy.hpp の oz_eval::fnv64．FNV-1a の更新式で，初期値は標準の
  // FNV-1a と違う）で，「同じ値を 3 形式に格納した積の誤差がビット単位で一致する」（randbf16）を相対誤差の表示桁ではなく
  // 出力そのもので比べるためのもの．整数モードの relerr・bits は nan（相対誤差ではなく一致の数で判定する）．
  // fp32acc は DST が FP32 か（PROBE_FP32ACC），pack はパッカの検査の結果（上の PackCheck の status．検査しない構成は n/a），
  // pack_even・pack_away・pack_trunc は 3 つの丸め方の参照に一致した要素の数（n/a のときは 0）
  {
    const uint64_t h=oz_eval::fnv64(Cw.data(), (size_t)tile_size(c.out_fmt)*c.Mt*c.Nt);
    probe::ProbeLine("fp8_probe").add("fmt","%s",c.fmt.c_str()).add("out","%s",c.ofmt.c_str()).add("fidelity","%s",oz_fidelity_name(c.mf))
      .add("mode","%s",c.mode_label().c_str()).add("Kt","%u",c.Kt).add("match","%u",exact).add("tot","%u",tot)
      .add("nonfinite","%u",nonfinite).add("relerr","%.6e",rel).add("bits","%.3f",bits).add("chash","%016llx",(unsigned long long)h)
      .add("fp32acc","%d",c.fp32acc ? 1 : 0).add("pack","%s",pk.status.c_str())
      .add("pack_even","%u",pk.me).add("pack_away","%u",pk.ma).add("pack_trunc","%u",pk.mz).print();
  }
  const bool closed = md->close();
  if(!closed) return 1;
  // 整数モード・出力 fp32 では不一致も失敗として返す（出力が 16 bit のときは書き出しの丸めを
  // 受けるので，整数の参照との不一致は失敗にしない．代わりに 3 つの丸め方のどれとも一致しなければ失敗）．
  if(!c.randmode && c.out_fmt==DataFormat::Float32 && exact!=tot) return 1;
  if(pk.status=="none") return 1;
  return nonfinite? 1 : 0;
}
