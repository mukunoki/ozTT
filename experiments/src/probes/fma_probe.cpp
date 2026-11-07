// SPDX-License-Identifier: Apache-2.0
//
// SFPU の積和が融合か（積を FP32 へ丸めずに加算するか）を実機で判定する．
// 【探針．本番の 16 経路のバイナリではない】remeasure_all.sh の probes の段（scripts/diagnose/probes.sh の probe_fma）が回す．
// 詳細は kernels/fma_probe.cpp の冒頭．積の先頭から t bit 下（2^-t）に 1 が立つ入力を
// t=20..46 の 1 bit 刻みで作り，積に打ち消しの定数 c を足す．2^-t が出れば積の上位 t+1 bit を
// 保っている，0 なら 2^-t の位置を丸め込んでいる．
//   t が偶数（t=2j）  a = b = 1+2^-j            ：a*b = 1 + 2^-(j-1) + 2^-2j
//   t が奇数（t=2j+1）a = 1+2^-j, b = 1+2^-(j+1)：a*b = 1 + 2^-j + 2^-(j+1) + 2^-(2j+1)
// どちらも c = -(積から末尾の項を除いた部分) は FP32 で厳密に表せる．末尾の判定は掃引の結果から
// 組み立てる．
// 観測したい精度の低下と経路の異常を分ける：全要素に同じ入力を流すので，読み戻した 1024 要素がビット単位で同じで
// なければ経路の異常とする．Mul->Add（本実装の書き方）は積を FP32 へ最近接で丸めてから足すので，ホストの
// fl(fl(a*b)+c) と一致しなければ同じく異常とする．各倍率の掃引の前に積だけを出す経路の確認（カーネルのモード 2．
// 積が FP32 に収まる入力）を 1 回流し，積と一致することを見る．異常があれば保持幅を断定せずに 1 で終わる．
// 最後に機械が読む 1 行 PROBE,fma_probe,...（欄は main の末尾）を出す．
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "testkit/probe_common.hpp"
#include "common/oz_bits.h"
using namespace tt; using namespace tt::tt_metal; using namespace tt::constants;
namespace fpa = fma_probe_args;   // 計算カーネルの引数と書き方の番号（testkit/kernels/probe_args.h）


// 1 回の実行の結果：先頭の要素と，全要素がそれとビット単位で同じか
struct Out { float v; bool uniform; };

// 書き方 mode（fpa::MODE_ADDCMUL・MODE_MULADD（Mul->Add）・MODE_MUL（積だけ））で a*b(+c) を 1 タイルの全要素に計算させる
static Out run(fpa::Mode mode, float av, float bv, uint32_t cbits, std::shared_ptr<distributed::MeshDevice>& md) {
    const uint32_t M=32,N=32;
    std::vector<float> a(M*N, av), b(M*N, bv);
    probe::SingleCoreMM mm;
    mm.in_fmt = mm.out_fmt = DataFormat::Float32;
    mm.compute = OZ_PROBES_KDIR "fma_probe.cpp";
    std::vector<tt_metal::UnpackToDestMode> u2d(64, tt_metal::UnpackToDestMode::UnpackToDestFp32);
    std::vector<uint32_t> ct(fpa::C_COUNT, 0u), rt(fpa::R_COUNT, 0u);
    ct[fpa::C_MODE] = mode;
    rt[fpa::R_C] = cbits;
    mm.ccfg = ComputeConfig{.math_fidelity=tt::tt_metal::MathFidelity::HiFi4,.fp32_dest_acc_en=true,
                            .unpack_to_dest_mode=u2d,.math_approx_mode=false,.compile_args=ct};
    mm.compute_rt = rt;
    const auto w = probe::run_single_core_mm(md, mm, probe::to_words(tilize_nfaces(a,M,N)), probe::to_words(tilize_nfaces(b,M,N)));
    const auto o = untilize_nfaces(probe::from_words<float>(w, M*N), M, N);
    bool uniform = true;
    for (float x : o) uniform &= f32_bits(x) == f32_bits(o[0]);
    return {o[0], uniform};
}

int main(int argc, char**){
    // 引数は取らない．渡されたら黙って読み捨てずに止める（引数で構成を変えたつもりのログが残らないように）
    if (argc > 1) { std::fprintf(stderr, "ERROR: fma_probe は引数を取らない（%d 個）\n", argc - 1); return 2; }
    auto md = distributed::MeshDevice::create_unit_mesh(0);
    std::printf("SFPU の積和（SFPMAD）が 48 bit の積のうち何ビットを保つかを 1 bit 刻みで測る．\n");
    std::printf("積の先頭から t bit 下に 1 が立つ a,b を作り，残りを打ち消す c を足す．1 度だけ丸める（FMA）なら\n"
                "2^-t が出る．積を FP32 へ丸めてから足すなら t>=24 で 0 が出る（t<=23 は積が FP32 に収まる）．\n\n");
    // 各指数で「t=20 から続けて保っている」最大の t．2^-t を保ち 2^-(t+1) を失うなら，
    // 積の上位 t+1 bit をちょうど保つ．3 つの指数の最小を判定に使う．
    // 「保つ」は出力がちょうど 2^-t のとき（FMA の正解 2^-t は FP32 の正規数で厳密なので，許容誤差で比べると数 ulp
    // ずれた出力も保つと数えてしまう）．保てなかった t の出力として固定の幅の丸めが返しうるのは，2^-t を丸め捨てた 0 と，
    // 保つ幅の最下位の位（最後に保った t の 2^-t）へ丸め上げた値の 2 つだけである（実機は t=27 以降で 2^-26 を返す）．
    // 一度失った後の t で再び保つ・それ以外の有限な値（符号の反転など）・有限でない出力が出るなら，固定の幅で丸めるとは
    // 言えないので判定を分け（irregular），保持幅を断定せずに 1 で終わる
    const int T0 = 20, T1 = 46;
    const int KS[3] = {0, 10, -10};
    int tkeep = T1, tk_of[3] = {0, 0, 0};
    bool irregular = false;
    int t_irr = 0;
    // 経路の異常（全要素が同じでない・Mul->Add がホストの fl(fl(a*b)+c) と違う・積だけの確認が積と違う）の件数と最初の例
    int path_bad = 0;
    char path_why[256] = "";
    for (int ki = 0; ki < 3; ++ki) {
        const int k = KS[ki];
        // 経路の確認：a = b = (1+2^-10) 2^k の積（1 + 2^-9 + 2^-20 の 2^2k 倍は FP32 に収まる）だけを出す
        {
            const float a = (float)std::ldexp(bits_f32(f32_bits(1.0f) + (1u << 13)), k);
            const Out r2 = run(fpa::MODE_MUL, a, a, 0u, md);
            const bool ok = r2.uniform && f32_bits(r2.v) == f32_bits(a * a);
            std::printf("  経路の確認（積だけ．2^%d 倍の (1+2^-10)^2）：%s（出力 %a，積 %a）\n", k, ok ? "一致" : "異常",
                        (double)r2.v, (double)(a * a));
            if (!ok && path_bad++ == 0)
                std::snprintf(path_why, sizeof path_why, "2^%d 倍の積だけの確認が積と違う（出力 %a，要素が揃う %d）", k,
                              (double)r2.v, (int)r2.uniform);
        }
        int tk = T0 - 1;
        bool run_ok = true;
        std::printf("  --- オペランドを 2^%d 倍したとき（積の指数は 2^%d）---\n", k, 2*k);
        std::printf("   t   期待（1 度丸め）addcmul（SFPMAD）        Mul->Add     判定\n");
        for (int t = T0; t <= T1; ++t) {
            const int j = t / 2;
            const int jb = (t % 2) ? j + 1 : j;           // b の立てるビット
            const float a = (float)std::ldexp(bits_f32(f32_bits(1.0f) + (1u << (23 - j))), k);
            const float b = (float)std::ldexp(bits_f32(f32_bits(1.0f) + (1u << (23 - jb))), k);
            // c = -(1 + 2^-j + 2^-jb)．偶数のとき j = jb なので 1 + 2^-(j-1) になる
            const double cm = 1.0 + std::ldexp(1.0, -j) + std::ldexp(1.0, -jb);
            const float c = -(float)std::ldexp(cm, 2*k);
            if ((double)(float)cm != cm) { std::printf("内部エラー：c が FP32 で表せない（t=%d）\n", t); return 1; }
            const double want = std::ldexp(1.0, -t + 2 * k);
            const Out o0 = run(fpa::MODE_ADDCMUL, a, b, f32_bits(c), md);
            const Out o1 = run(fpa::MODE_MULADD, a, b, f32_bits(c), md);
            const float r0 = o0.v, r1 = o1.v;
            int e0 = (r0 == 0.0f || !std::isfinite(r0)) ? 0 : (int)std::lround(std::log2(std::fabs((double)r0)));
            const bool kept = (double)r0 == want;
            // 最後に保った t（tk．続けて保っているあいだは t-1）の位 2^-tk へ丸め上げた値
            const bool up = !kept && (double)r0 == std::ldexp(1.0, -tk + 2 * k);
            const bool odd = !std::isfinite(r0) || (!kept && r0 != 0.0f && !up);
            const char* verdict = !std::isfinite(r0) ? "有限でない" : (r0 == 0.0f) ? "丸めている" : kept ? "保っている"
                                : up ? "一部だけ保つ" : "異常な値";
            std::printf("  %2d   2^%-5d      %-12.6g (2^%-4d)  %-12.6g %s\n",
                        t, -t + 2*k, (double)r0, e0, (double)r1, verdict);
            if (odd || (!run_ok && kept)) { if (!irregular) t_irr = t; irregular = true; }
            if (run_ok && kept) tk = t; else run_ok = false;
            // 経路の異常：全要素が同じでない，Mul->Add がホストの fl(fl(a*b)+c)（-ffp-contract=off で積を FP32 へ丸める）と違う
            const float p = a * b;
            const float want1 = p + c;
            if ((!o0.uniform || !o1.uniform) && path_bad++ == 0)
                std::snprintf(path_why, sizeof path_why, "t=%d（2^%d 倍）で出力の要素が揃わない（addcmul %d・Mul->Add %d）",
                              t, k, (int)o0.uniform, (int)o1.uniform);
            if (f32_bits(r1) != f32_bits(want1) && path_bad++ == 0)
                std::snprintf(path_why, sizeof path_why, "t=%d（2^%d 倍）の Mul->Add %a がホストの fl(fl(a*b)+c) %a と違う",
                              t, k, (double)r1, (double)want1);
        }
        std::printf("\n");
        tk_of[ki] = tk;
        tkeep = std::min(tkeep, tk);
    }
    // 判定はこの実行の掃引から出す（固定の文言にすると，実機の挙動が変わっても同じ結論を出し続ける）
    if (path_bad) {
        std::printf("判定（今回の測定）：経路の異常が %d 件ある（最初は %s）．精度の低下と区別できないので\n"
                    "      保持幅を断定しない．\n", path_bad, path_why);
    } else if (irregular) {
        std::printf("判定（今回の測定）：t=%d で，一度失った後に再び 2^-t を保ったか，固定の幅の丸めでは出ない値（0 でも\n"
                    "      最後に保った位への丸め上げでもない有限な値か，有限でない値）が出た．固定の幅で丸めるとは\n"
                    "      言えないので保持幅を断定しない（経路の誤りを疑う）．\n", t_irr);
    } else if (tkeep >= T1) {
        std::printf("判定（今回の測定）：t=%d まで 2^-t を保った．積の上位 %d bit 以上を保っており，\n"
                    "      この掃引では 1 度だけ丸める FMA と区別できない．\n", T1, T1 + 1);
    } else if (tkeep < T0) {
        std::printf("判定（今回の測定）：t=%d でも 2^-%d を保てなかった．保つのは積の上位 %d bit 未満で，\n"
                    "      FP32（24 bit）へ丸めるより粗いか，経路が正しく動いていない．\n", T0, T0, T0 + 1);
    } else {
        std::printf("判定（今回の測定）：SFPMAD は 2^-%d を保ち 2^-%d を失う．保つのは 48 bit の積のうち\n"
                    "      上位 %d bit（FP32 の 24 bit + %d bit）で，それより下は丸め込む．\n",
                    tkeep, tkeep + 1, tkeep + 1, tkeep + 1 - 24);
        if (tkeep <= 23)
            std::printf("      これは積を FP32（24 bit）へ丸めてから足す場合と区別できない．\n");
        else
            std::printf("      1 度だけ丸める演算ではないので，IEEE 754 の意味での FMA は無い．\n"
                        "      したがって a*b - fl(a*b) を 1 命令で取り出せず，Dekker 分割が要る．\n");
    }
    std::printf("      Mul->Add の列は，積を FP32 へ丸めてから足す書き方（本実装）の結果である．\n");
    const bool closed = md->close();
    // 機械が読む行．held_p0・held_p10・held_m10 はオペランドを 2^0・2^10・2^-10 倍したときに t=20 から続けて保った最大の t
    // （t0-1 なら t0 も保てなかった），held はその最小（判定の行の「2^-held を保ち 2^-(held+1) を失う」）．
    // irregular=1 は固定の幅の丸めでない掃引，path_bad は経路の異常の件数（どちらかが 0 でなければ保持幅を断定しない）
    probe::ProbeLine("fma_probe").add("t0", "%d", T0).add("t1", "%d", T1)
        .add("held_p0", "%d", tk_of[0]).add("held_p10", "%d", tk_of[1]).add("held_m10", "%d", tk_of[2])
        .add("held", "%d", tkeep).add("irregular", "%d", irregular ? 1 : 0).add("path_bad", "%d", path_bad).print();
    return (!irregular && !path_bad && closed) ? 0 : 1;
}
