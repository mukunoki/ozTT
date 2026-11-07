// SPDX-License-Identifier: Apache-2.0
//
// FP32 累算の検査：Tenstorrent 提供のサンプル（nyoshifujiTT/tt-metal PR #3，
// matmul_fp32_acc_check）と同じ単一コア 32x32x32・HiFi4・fp32_dest_acc_en=true の
// 構成で，(a) 同じ 32 項のベクトルと，(b) 指数差 d の 2 項だけからなるベクトルを
// 流し，FP32 で読み戻して累算経路が保持する幅を見る．(c) は出力をサンプルと同じ BF16 の
// まま受け，大きい項どうしを相殺させて小さい項が残るかを見る．
// 【探針．本番の 16 経路のバイナリではない】remeasure_all.sh の probes の段（scripts/diagnose/probes.sh の probe_fp32acc）が回す．
//
// サンプルとの違いは 2 点：出力を BF16 ではなく FP32 で受ける（BF16 出力では
// 2^-8 より細かい差が見えない），および 2 項の指数差を 2^-8 より広げる．
//
// A は全要素 1，B の列はどれも同じ 32 項なので，C の 1024 要素は全部同じ値になるはずである．読み戻した全要素が
// 有限で，ビット単位で揃っていなければ経路の異常として数え（観測したい累算の精度の低下とは分ける），1 で終わる．
// 揃っていても一様に誤った値（全部 0 など）が返る経路は見分けられないので，最初に答えが厳密に決まる入力（1 項だけ 1．
// 和は FP32 でも BF16 でもちょうど 1）を流し，ちょうど 1 でなければ同じく経路の異常に数える．
// 結果は人向けの表と，回ごとに機械が読む行 PROBE,fp32acc_probe,...（欄は run の後の注記）．
//
// 使い方: fp32acc_probe            # サンプルのベクトル + d と k2 の掃引 + 相殺ベクトル
// 出所:   https://github.com/nyoshifujiTT/tt-metal/pull/3
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "testkit/probe_common.hpp"
using namespace tt; using namespace tt::tt_metal; using namespace tt::constants;

static int g_path_bad = 0;   // 出力の要素が有限で揃っていない回の数

// A=1（32x32），B の各列 = terms（32 項）．出力 [0] = Σ terms を FP32 で返す（BF16 出力なら BF16 の値）．
static float run(const std::vector<float>& terms, std::shared_ptr<distributed::MeshDevice>& md, bool out_f32) {
    const uint32_t M=32,N=32,K=32, Mt=1,Kt=1,Nt=1;
    std::vector<bfloat16> a(M*K, bfloat16(1.0f)), b(K*N);
    for(uint32_t k=0;k<K;k++) for(uint32_t n=0;n<N;n++) b[k*N+n]=bfloat16(terms[k]);
    probe::SingleCoreMM mm;
    mm.in_fmt = DataFormat::Float16_b;
    mm.out_fmt = out_f32 ? DataFormat::Float32 : DataFormat::Float16_b;
    mm.ccfg = ComputeConfig{.math_fidelity=tt::tt_metal::MathFidelity::HiFi4,.fp32_dest_acc_en=true,.math_approx_mode=false,.compile_args={Mt,Kt,Nt}};
    const auto w = probe::run_single_core_mm(md, mm, probe::to_words(tilize_nfaces(a,M,K)), probe::to_words(tilize_nfaces(b,K,N)));
    std::vector<float> o(M*N);
    if(out_f32){ o = untilize_nfaces(probe::from_words<float>(w, M*N), M, N); }
    else { const auto h = untilize_nfaces(probe::from_words<bfloat16>(w, M*N), M, N); for(size_t i=0;i<o.size();i++) o[i]=(float)h[i]; }
    uint32_t b0; std::memcpy(&b0, &o[0], 4);
    bool ok = true;
    for(float x : o){ uint32_t u; std::memcpy(&u, &x, 4); ok &= std::isfinite(x) && u == b0; }
    if(!ok){ ++g_path_bad; printf("  経路の異常：出力の要素が有限で揃っていない（先頭 %a）\n", (double)o[0]); }
    return o[0];
}

// 機械が読む行（PROBE,fp32acc_probe,case=...）．値はどれも %a で厳密に出す．
//   case=pr       (a)：ref（FP32 で足した参照）・out_bf16・out_fp32
//   case=pair     (b)：d・k2・out・state（kept 保持，lost 消失，undecidable 判別不能，partial 部分的に残った）
//   case=cancel   (c)：d・group（same は同じ 8 項群 k=2，other は別の群 k=8）・out・true（真値 2^-d）
//   case=summary  path_bad（出力の要素が有限で揃っていなかった回の数．0 でなければ終了コード 1）
int main(int argc, char**){
    // 引数は取らない．渡されたら黙って読み捨てずに止める（引数で構成を変えたつもりのログが残らないように）
    if(argc>1){ fprintf(stderr,"ERROR: fp32acc_probe は引数を取らない（%d 個）\n",argc-1); return 2; }
    auto md=distributed::MeshDevice::create_unit_mesh(0);
    // 経路の確認（冒頭）：1 項だけ 1 のベクトルの和は FP32 出力でも BF16 出力でもちょうど 1
    {
        std::vector<float> one(32, 0.f); one[0] = 1.0f;
        const float o32 = run(one,md,true), o16 = run(one,md,false);
        const bool ok = o32 == 1.0f && o16 == 1.0f;
        if(!ok) ++g_path_bad;
        printf("[経路の確認] 1 項だけ 1 の和  FP32 出力=%a  BF16 出力=%a  %s\n", (double)o32, (double)o16,
               ok ? "一致" : "異常（ちょうど 1 でない）");
    }
    // (a) PR のベクトル
    std::vector<float> pr={1.0f,0.25f,-0.00390625f,-0.03125f,-0.00390625f,1.0f,-1.0f,-0.5f,
        -0.00390625f,-1.0f,0.5f,1.0f,-0.5f,-1.0f,-0.5f,-0.5f, 0.03125f,-0.5f,-0.00390625f,-0.25f,0.00390625f,0.25f,-0.5f,0.03125f,
        -0.25f,0.5f,0.25f,0.00390625f,0.5f,1.0f,-0.25f,0.03125f};
    float ref=0.f; for(float t:pr) ref+=t;
    const float pr_bf = run(pr,md,false), pr_f32 = run(pr,md,true);
    printf("[PR のベクトル] FP32 参照=%.9g  BF16 出力=%.9g  FP32 出力=%.9g\n", ref, pr_bf, pr_f32);
    probe::ProbeLine("fp32acc_probe").add("case", "pr").add("ref", "%a", (double)ref).add("out_bf16", "%a", (double)pr_bf)
        .add("out_fp32", "%a", (double)pr_f32).print();
    // (b) 2 項：k1=0 に 1.0，k2 に 2^-d．真値 1+2^-d．FP32 出力で 1.0 なら小さい項が落ちている．
    //     d >= 24 では真値 1+2^-d が FP32 に載らない（d=24 は 1 と 1+2^-23 のちょうど中間）ので，出力が 1.0 でも
    //     小さい項が累算の途中で落ちたのか最後の FP32 への丸めで消えたのかを区別できない．そこは「判別不能」と出す
    //     （相殺ベクトルの (c) なら d が大きくても見える）．真値は double で持つ（float にすると d=24 で 1.0 になり，
    //     出力 1.0 を「保持」と取り違える）
    printf("[2 項 1 + 2^-d，FP32 出力]  行=d，列=k2（k2<=7 は同じ 8 項群，k2>=8 は別の群）\n");
    printf("%4s","d"); for(int k2: {1,4,7,8,16,31}) printf("  k2=%-2d ",k2); printf("\n");
    std::vector<probe::ProbeLine> pls;
    for(int d: {4,8,9,10,11,12,16,20,23,24}){
        printf("%4d",d);
        for(int k2: {1,4,7,8,16,31}){
            std::vector<float> v(32,0.f); v[0]=1.0f; v[k2]=std::ldexp(1.0f,-d);
            const float o=run(v,md,true);
            const double tr=1.0+std::ldexp(1.0,-d);
            const bool repr=(double)(float)tr==tr;    // 真値が FP32 で表せるか（d <= 23）
            const char* st;
            if(repr && (double)o==tr){ printf("  保持   "); st="kept"; }
            else if(o==1.0f){ printf(repr ? "  消失   " : "  判別不能"); st = repr ? "lost" : "undecidable"; }
            else { printf(" %+.1e", (double)o-1.0); st="partial"; }   // 部分的に残った値（1 からの差）
            pls.push_back(probe::ProbeLine("fp32acc_probe").add("case", "pair").add("d", "%d", d).add("k2", "%d", k2)
                              .add("out", "%a", (double)o).add("state", "%s", st));
        }
        printf("\n");
        fflush(stdout);
    }
    // 機械が読む行は表を書き終えてから出す（表の行の途中に挟まると，表を読む照合の正規表現が外れる）
    for(const auto& l : pls) l.print();
    // (c) 出力を BF16 のまま（サンプルを変えずに）でも見える形：大きい項を相殺させる．
    //     {1, -1, 2^-d} なら真値は 2^-d で，BF16 でも表せる．整列で 2^-d が落ちれば 0 が返る．
    printf("[相殺ベクトル {1, -1, 2^-d}，サンプルと同じ BF16 出力]  真値は 2^-d\n");
    std::vector<probe::ProbeLine> cls;
    for(int d: {8,10,11,12,16}){
        std::vector<float> s1(32,0.f); s1[0]=1.0f; s1[1]=-1.0f; s1[2]=std::ldexp(1.0f,-d);   // 同じ群
        std::vector<float> s2(32,0.f); s2[0]=1.0f; s2[1]=-1.0f; s2[8]=std::ldexp(1.0f,-d);   // 別の群
        const float o1 = run(s1,md,false), o2 = run(s2,md,false);
        printf("  d=%2d  同じ群(k=2): %.4e   別の群(k=8): %.4e   真値 %.4e\n", d, o1, o2, std::ldexp(1.0,-d));
        fflush(stdout);
        for(int gi=0; gi<2; ++gi)
            cls.push_back(probe::ProbeLine("fp32acc_probe").add("case", "cancel").add("d", "%d", d).add("group", "%s", gi ? "other" : "same")
                              .add("out", "%a", (double)(gi ? o2 : o1)).add("true", "%a", std::ldexp(1.0,-d)));
    }
    for(const auto& l : cls) l.print();
    const bool closed = md->close();
    probe::ProbeLine("fp32acc_probe").add("case", "summary").add("path_bad", "%d", g_path_bad).print();
    return (g_path_bad == 0 && closed) ? 0 : 1;
}
