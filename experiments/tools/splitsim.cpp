// scheme II の「分割だけ」の限界を測る．行/列ごとに 2 のべき τ で正規化し，
// A' = round(A/τ * 2^b) の内積を__float128の補正和で求め，スケールを戻す．
// 剰余・CRT再構成は模型に含めない．補正和等の有限精度誤差は残るが，主に量子化の誤差を見る．
// round の同点は最近接偶数へ倒す（rintq．既定の丸めモード）．本番の桁作り（ozaki2/kernels/oz2_digits.h）の
// Rnd（tt-llk の round_tile．最近接偶数）と同じ向きである（本番は 3 本組の上位 limb だけを丸めるので，
// 保証は |A' - X| <= 1 で最近接とは限らないが，3 本組が X を厳密に持つときの同点はこの向きになる）．
// 「ほぼ」なのは総和が __float128 の補正和のためで，厳密な累算器ではない（下）．
//     ./splitsim phi b [K=1024] [出力の辺 n=K]
// 引数は 2〜4 個（それより多いか少ない，整数・数でない引数は終了コード 2）．
// 入力はドライバと同じもの：oz_input.hpp の gen_operands（seed 7，OZ_PHI=φ．φ=0 も OZ_PHI=0 として渡す）で K×K の A・B を作る．
// これは scripts/measure/exponent_sweep.sh の tab:phib（ozaki2_int8 の K^3．K=1024）が実機で測る行列そのもので，誤差は出力の先頭の
// n×n の区画（A の行 0..n-1 と B の列 0..n-1．n=K なら全要素）の ||C'-C||_F/||C||_F である．n=K で tab:phib の実機の値と
// 同じ入力・同じ指標になる．ただしこれは同じ入力に対する「最近接の量子化」の模型で，実機の値との差は再構成の丸めの
// 有無だけではない：本番の量子化は上の注記のとおり最近接とは限らず（|A' - X| <= 1），その差も残る．φ が大きいと
// ||C||_F は少数の要素に支配されるので，出力の一部の要素だけで測った値は実機の値から数 bit 外れうる（内積を 512 本だけ
// 抜いた以前の版は φ=4 で 7 bit 違った）．
// ほかの入力のつまみ（OZ_EXPDIFF など）・OZ_MIN_BITS・OZ_DUMP_C は持たないので，立っていれば止める
// （oz_knobs.hpp の reject_unused_input）．K は必要な b を決める側の量（窓は 2b+1+log2 K bit 要る）でもあり，実機と突き合わせる
// ときは合わせること．n^2 K 回の __float128 の積和なので，-fopenmp でビルドする（src/Makefile の HOST_TOOLS はそうしている）．
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common/host_eval.hpp"
#include "common/oz_knobs.hpp"   // reject_unused_input（つまみの表の入力・判定・書き出しの組）
#include "oz_sim.h"              // exp_ceil（2 のべきのスケールの指数．study/oz2_quantsim と同じ関数）
int main(int argc,char**argv){
    if(argc<3||argc>5){ fprintf(stderr,"使い方: splitsim phi b [K=1024] [出力の辺 n=K]\n"); return 2; }
    double phi=0.0;
    if(!oz_env::to_f64(argv[1],phi)||!(phi>=0&&phi<=100)){ fprintf(stderr,"ERROR: phi=%s は 0--100 の数ではない\n",argv[1]); return 2; }
    // 整数の引数は 10 進の整数として全体を読み切れて範囲に入るときだけ受け付ける（src/common/oz_env.hpp．atoi では "56x" が
    // 56 に，"x" が 0 になる）．b は 2^b が __float128 の指数に収まり，A'*B' の補正和が意味を持つ範囲（積は 2^2b まで．
    // b > 56 では積が 113 bit を超えて丸まる．下の参照解の説明）
    const int b=(int)oz_env::arg_i64(argc,argv,2,0,1,200,"b");
    const int K=(int)oz_env::arg_i64(argc,argv,3,1024,1,8192,"K");   // K×K の A・B（8192 で 1 GB）
    const int NO=(int)oz_env::arg_i64(argc,argv,4,K,1,K,"出力の辺 n");
    oz_knobs::reject_unused_input("splitsim");
    setenv("OZ_PHI",oz_eval::rt_double(phi).c_str(),1);
    std::vector<double> A,B;
    oz_eval::gen_operands(A,B,(uint32_t)K,(uint32_t)K,(uint32_t)K);   // seed は既定（kDefaultSeed = 7）
    // 行ごと（A）・列ごと（B）の 2 のべきのスケール τ（最大絶対値以上の最小の 2 のべき．全要素が 0 なら 1．oz_sim.h の exp_ceil）
    auto pow2ceil=[](double m){ return m>0 ? std::ldexp(1.0,oz_sim::exp_ceil(m)) : 1.0; };
    std::vector<double> ta(NO), tb(NO);
    for(int i=0;i<NO;i++){ double m=0; for(int k=0;k<K;k++) m=std::fmax(m,std::fabs(A[(size_t)i*K+k])); ta[i]=pow2ceil(m); }
    for(int j=0;j<NO;j++){ double m=0; for(int k=0;k<K;k++) m=std::fmax(m,std::fabs(B[(size_t)k*K+j])); tb[j]=pow2ceil(m); }
    // 分割：A' = round(a/τ * 2^b)（同点は最近接偶数．冒頭の説明）．a/τ*sc は 2 のべきの倍なので厳密．A' は |A'| <= 2^b の整数で
    // __float128 に厳密に載る（b <= 112）．A は行ごと，B は列ごとに並べ直して持つ
    const __float128 sc=ldexpq((__float128)1.0,b);
    std::vector<__float128> Aq((size_t)NO*K), Bq((size_t)NO*K);
    for(int i=0;i<NO;i++) for(int k=0;k<K;k++) Aq[(size_t)i*K+k]=rintq((__float128)A[(size_t)i*K+k]/ta[i]*sc);
    for(int j=0;j<NO;j++) for(int k=0;k<K;k++) Bq[(size_t)j*K+k]=rintq((__float128)B[(size_t)k*K+j]/tb[j]*sc);
    // 個々の積は入力が binary64 なので高々 106 bit で __float128 に厳密に載るが，
    // K 項の総和は 106+log2(K) bit を要して 113 bit を超える．素直に足すと，桁落ちの
    // 強い内積ではその欠落が「分割だけの誤差」に混ざる．Neumaier の補正和で回収する
    //（残差 c を別に持てば実効の作業幅が倍になり，K=1024 でも余裕がある）．
    auto csum=[](__float128& s,__float128& c,__float128 x){
        const __float128 t=s+x;
        c += (fabsq(s)>=fabsq(x)) ? ((s-t)+x) : ((x-t)+s);
        s = t; };
    // 誤差ノルム^2 と解ノルム^2 はスレッドごとに足してから合わせる（__float128 は OpenMP の reduction に載らない）
    int nth=1;
#ifdef _OPENMP
    nth=omp_get_max_threads();
#endif
    std::vector<__float128> ens(nth,0), cns(nth,0);
#pragma omp parallel for schedule(dynamic,1)
    for(int i=0;i<NO;i++){
        int tid=0;
#ifdef _OPENMP
        tid=omp_get_thread_num();
#endif
        std::vector<double> bcol(K);
        for(int j=0;j<NO;j++){
            for(int k=0;k<K;k++) bcol[k]=B[(size_t)k*K+j];
            // 参照解：積は厳密で，K 項の足し込みの誤差は補正項 exc に回収される．ただし最後の ex+=exc で __float128 1 語
            // （113 bit）に丸めるので，参照解の相対誤差は 2^-113 程度である（2 項 1 と 2^-113 の和は 1 になる）．
            // 分割の誤差（2^-40..-70）より十分小さいので，到達精度の表示（%.1f）には効かない
            __float128 ex=0, exc=0;
            for(int k=0;k<K;k++) csum(ex,exc,(__float128)A[(size_t)i*K+k]*(__float128)bcol[k]);
            ex+=exc;
            // 二冪でスケールしたFP64入力を整数化した各値は高々53有効bitで，個々の積は106bit以内．総和等では丸めが残る．
            __float128 acc=0, accc=0;
            const __float128* ai=&Aq[(size_t)i*K]; const __float128* bj=&Bq[(size_t)j*K];
            for(int k=0;k<K;k++) csum(acc,accc,ai[k]*bj[k]);
            acc+=accc;
            const __float128 ap = acc*(__float128)ta[i]*(__float128)tb[j]/(sc*sc);
            const __float128 e = ap-ex;
            ens[tid] += e*e; cns[tid] += ex*ex;
        }
    }
    __float128 en=0, cn=0;
    for(int t=0;t<nth;t++){ en+=ens[t]; cn+=cns[t]; }
    printf("phi=%-4g b=%-3d  分割だけの到達精度 = %.1f bit\n",
           phi, b, -log2((double)sqrtq(en/cn)));
    return 0;
}
