// 再構成（frac = sum k_i/m_i の小数部）を FP32 triple-float で模擬し，厳密値と比べて
// 実効精度を測る．どの要素が床を決めているかを切り分けるための調査用である．
//
// NL で累積の組み方を選ぶ．NL=3 が現行の実装（src/ozaki2/kernels/recon_crt.cpp の既定の 1 パス）と同じ構造（3 本組の
// 累積に，逆数の断片 5 本をそれぞれ k_i 倍して足す）で，残りは「もし別の組み方にしたら床がどこまで下がるか」を
// 見るための仮想の構成である：
//   NL=3  3 本組・5 項（r0h,r0l,r1h,r1l,r2）        ← 現行と同じ構造
//   NL=4  3 本組・6 項（r2 も Dekker 分割する）
//   NL=5  3 本組・7 項（さらに第 4 limb r3 を足す）
//   NL=6  4 本組・7 項（累積を 4 本にする．DST を 9 本使う想定なので実装していない）
// RED を立てると法ごとに小数部へ畳む（現行の実装はこれを行う）．
// 「現行」は構造が同じという意味で，断片の値と入力は本番と同じではない（模擬を簡略化してある）：
//   断片  本番（src/ozaki2/ozaki2_moduli.h の rp0．OZ2_NRP0=5）は厳密な残差（__float128）から上位 12 bit ずつ 4 本を取り，
//         最後の 1 本を FP32 まるごと（24 bit）にする．ここは FP32 に丸めた 1/m とその残差をそれぞれ Dekker 分割した
//         半分（r0h・r0l・r1h・r1l）と残り（r2）で，切り方が違う
//   k_i   本番の k_i は r_i*y_i の ModRound の出力で |k_i| <= floor(m/2)+1（ozaki2_moduli.h の oz2_crt_k_bound．
//         tools/modround_check.cpp の (a')．tools/twosum_probe.cpp は対称剰余 |k_i| <= floor(m/2) から振る）だが，ここは [0,m) から引く
//   法    INT8 経路の表（OZ2_TABLE_I8）だけ．本番が末尾の断片を tfAddSmall で足す扱い（NSMALL_FRAG）も模擬しない
// 出力の「（現行）」の表記もこの意味である（scripts/rules/probe_checks.py は probe_reconsim の NL=3 の回をスライドの記録の
// 照合に使う）．本番の構成（本番の断片 rp0・rp1・rp2，対称剰余の k_i，本番の NSMALL_FRAG と 8 経路の法の集合）での frac の
// 最悪誤差は tools/twosum_probe.cpp が測る．この道具の値は，組み方（NL）と法ごとに畳むか（RED=1）畳まないか（RED=0）の
// 比べ（簡略の模擬）として読む（scripts/diagnose/probes.sh の probe_reconsim は RED=1 だけを回す）．
// SFPU の演算（加減乗算・Rnd・非正規化数の扱い）は共通の模擬（checks/sfpu_tf_emu.hpp の Dst．tools/oz_sim.h を通す）を使い，
// 3 本組の足し込みは本番と同じ tfAddS<3>（NL=3..5）．4 本組（NL=6）は DST に載らない仮想の構成なので，同じ演算の
// TwoSum を 1 段延ばした手続きをここに置く．
#include <initializer_list>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iterator>
#include <random>
#include <vector>
#include <quadmath.h>
#include "oz_sim.h"
typedef float f32;
using sfpu_emu::Dst;
// 演算の器（DST の模擬．レジスタ 0..5 を作業に使う）で x + y を TwoSum する（本番の sfpu_tf_proc.h の TwoSum を模擬の上で呼ぶ）
static inline void twosum(Dst& W, f32 x, f32 y, f32& s, f32& e) {
    W.r[0] = x; W.r[1] = y; W.TwoSum(0, 1, 2, 3, 4, 5); s = W.r[2]; e = W.r[3];
}
// 4 本組版（最下位に 1 本足して誤差を回収する．仮想の構成）
static inline void tfadd4(Dst& W, f32& a0, f32& a1, f32& a2, f32& a3, f32 v) {
    f32 s, e; twosum(W, a0, v, s, e); a0 = s;
    f32 s2, e2; twosum(W, a1, e, s2, e2); a1 = s2;
    f32 s3, e3; twosum(W, a2, e2, s3, e3); a2 = s3;
    W.r[0] = a3; W.r[1] = e3; W.Add(0, 1, 0); a3 = W.r[0];
}
// 演算の器で v = k * 断片（SFPU の MulS）と a0 -= round(a0)（Rnd）
static inline f32 mul(Dst& W, f32 k, f32 c) { W.r[0] = k; W.MulS(0, c); return W.r[0]; }
static inline f32 frac0(Dst& W, f32 a0) { W.r[0] = a0; W.Cpy(0, 1); W.Rnd(1); W.Sub(0, 1, 0); return W.r[0]; }
// 3 本組（DST の 0..2）へ v を tfAddS<3> で足す（最下位は近似）
static inline void tfadd(Dst& D, f32 v) { D.r[sfpu_emu::dst::term] = v; sfpu_emu::tfAddS<3>(D); }
int main(int argc,char**argv){
    if(argc<3||argc>4){ fprintf(stderr,"使い方: reconsim S NL [RED]  （S=法の本数 1..24，NL=3..6，RED=0|1）\n"); return 2; }
    // 法テーブルは本番の INT8 経路の表そのもの（src/ozaki2/ozaki2_moduli.h の OZ2_TABLE_I8）の先頭 S 本
    const int* mods = OZ2_TABLE_I8;
    const int nmods = (int)std::size(OZ2_TABLE_I8);
    // 引数は 10 進の整数として全体を読み切れて範囲に入るときだけ受け付ける（src/common/oz_env.hpp．atoi では "16x" が 16 に，
    // "x" が 0 になる）．範囲の外は終了コード 2
    const int S=(int)oz_env::arg_i64(argc,argv,1,0,1,nmods,"S");
    const int NL=(int)oz_env::arg_i64(argc,argv,2,0,3,6,"NL");
    const int RED=(int)oz_env::arg_i64(argc,argv,3,0,0,1,"RED");
    // 逆数を FP32 の limb へ厳密有理数で分解する（残差を厳密に持つのは本番と同じだが，limb を FP32 まるごとに取ってから
    // Dekker 分割するのは，残差から上位 12 bit ずつ取る本番の rp0 と違う．冒頭の注）
    std::vector<f32> r0(S),r1(S),r2(S),r3(S);
    std::vector<f32> r0h(S),r0l(S),r1h(S),r1l(S),r2h(S),r2l(S);
    for(int i=0;i<S;i++){
        // 残差は __float128（113 bit）で持つ．FP32 4 本＝96 bit なので十分．
        __float128 rem=(__float128)1/(__float128)mods[i];
        r0[i]=(f32)(double)rem; rem-=(__float128)r0[i];
        r1[i]=(f32)(double)rem; rem-=(__float128)r1[i];
        r2[i]=(f32)(double)rem; rem-=(__float128)r2[i];
        r3[i]=(f32)(double)rem;
        // 本番は k*断片 を無誤差にするため断片を 12 bit に取る（ここでは FP32 の limb を 12 bit ずつに Dekker 分割して近づける）
        auto dek=[](f32 x,f32&h,f32&l){ f32 c=(f32)(4097.0*(double)x); h=(f32)((double)c-(double)(f32)((double)c-(double)x)); l=(f32)((double)x-(double)h); };
        dek(r0[i],r0h[i],r0l[i]); dek(r1[i],r1h[i],r1l[i]); dek(r2[i],r2h[i],r2l[i]);
    }
    std::mt19937 rng(11); std::uniform_int_distribution<int> U(0,1<<20);
    __float128 worst=0; double sum=0; int n=0;
    for(int t=0;t<20000;t++){
        std::vector<int> k(S); for(int i=0;i<S;i++) k[i]=U(rng)%mods[i];
        // 厳密な frac
        __float128 ex=0; for(int i=0;i<S;i++) ex+=(__float128)k[i]/(__float128)mods[i];
        ex-=floorq(ex);
        // 模擬（3 本組は D の 0..2，4 本組と演算の器は W）
        Dst D, W;
        f32 a0=0,a1=0,a2=0,a3=0;
        for(int i=0;i<S;i++){
            f32 ki=(f32)k[i];
            // NL=3 が現行の 5 項（r0h,r0l,r1h,r1l,r2）．他は上の表のとおり仮想の構成
            if(NL==6){   // 4 本組（DST 9 本要る想定．実装していない）
                for(f32 c:{r0h[i],r0l[i],r1h[i],r1l[i],r2h[i],r2l[i],r3[i]}) tfadd4(W,a0,a1,a2,a3,mul(W,ki,c));
                if(RED) a0=frac0(W,a0);
                continue; }
            tfadd(D,mul(W,ki,r0h[i])); tfadd(D,mul(W,ki,r0l[i]));
            tfadd(D,mul(W,ki,r1h[i])); tfadd(D,mul(W,ki,r1l[i]));
            if(NL==3)      tfadd(D,mul(W,ki,r2[i]));                                         // 現行：5 項
            else if(NL==4){ tfadd(D,mul(W,ki,r2h[i])); tfadd(D,mul(W,ki,r2l[i])); }           // r2 も分割：6 項
            else          { tfadd(D,mul(W,ki,r2h[i])); tfadd(D,mul(W,ki,r2l[i]));             // NL=5
                            tfadd(D,mul(W,ki,r3[i])); }                                        // + 第 4 limb：7 項
            if(RED) D.r[0]=frac0(W,D.r[0]);   // 法ごとに小数部へ畳む
        }
        if(NL!=6){ a0=D.r[0]; a1=D.r[1]; a2=D.r[2]; }
        // 小数部を取る（カーネルと同じく a0 を round して引く）
        a0=frac0(W,a0);
        __float128 got=(__float128)a0+(__float128)a1+(__float128)a2+(__float128)a3;
        if(got>(__float128)0.5) got-=1;
        if(got<-(__float128)0.5) got+=1;
        __float128 exs=ex; if(exs>(__float128)0.5) exs-=1;
        __float128 d=fabsq(got-exs);
        if(d>worst) worst=d;
        sum+=(double)d; n++;
    }
    const char* tag = NL==6?"4 本組 7 項（仮想）":(NL==3?(RED?"3 本組 5 項（現行）":"3 本組 5 項（畳みなし）"):(NL==4?"r2 分割 6 項（仮想）":"r2 分割+第4limb 7 項（仮想）"));
    printf("S=%-3d %-22s frac の絶対誤差: 最悪 2^%.1f  平均 2^%.1f\n",
           S,tag,worst>0?log2((double)worst):-999.0,sum>0?log2(sum/n):-999.0);
    return 0;
}
