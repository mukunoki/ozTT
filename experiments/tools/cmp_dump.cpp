// cmp_dump.cpp -- 2 つの実行の結果 C（OZ_DUMP_C の書き出し）を要素ごとに比べる（デバイス不要）
// ---------------------------------------------------------------------------
//     make -C src cmp_dump                （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     ./src/build/cmp_dump old.bin new.bin
// 実装の変更で結果の最下位ビットが変わりうるとき（docs/nongemm_opt_results.md の §3：遅延再構成の
// 小さい対角の近道で足し込みの回数が変わる），「異なる要素の数」「差の大きさ」「どちらが参照に近いか」を
// 数える．参照は同じ入力の Dot2 の非評価和 Chi+Clo で，__float128 で足して使う．
// 2 つの書き出しが同じ入力から来たことは，参照 Chi・Clo と A の行ごと・B の列ごとの最大絶対値が全要素で
// ビット単位に一致することで確かめる（違えば終了コード 2．浮動小数の == ではないので，+0 と -0 は違い，同じビット列の
// NaN は同じとみなす）．書き出しは入力そのものもその要約（hash）も
// 持たないので，これは必要条件にすぎない：同じ積と同じ最大値を与える別の A・B は区別できない
// （例えば M=N=1・K=2 の a=(1,1/2)・b=(1/2,1) と a=(1/2,1)・b=(1,1/2)）．入力を変えずに実装だけを変えた 2 回の実行を比べる道具である．
// 差の大きさは 2 つの単位で出す：
//   ulp       結果そのものの ulp（大きい方の |C| の ulp．非正規化数と 0 では 2^-1074）．出力が相殺で小さいと，
//             同じ絶対誤差でも ulp の数は大きくなる
//   scale     K max|a_i| max|b_j|（その要素の積の和の大きさの上界）に対する相対で，2^-53 を 1 とした値．
//             尾崎スキームの誤差はこの尺度で決まるので，相殺の有無によらず比べられる
// 有限でない要素（inf・NaN）は差の大きさと近づいた／遠ざかったの数から外し，nonfinite に各側の個数を出す（その側の
// rel_fro は nan．評価できない）．参照 Chi+Clo が有限でない要素（溢れた入力．OZ_CONST=1e300 など）も数えて nonfinite の
// ref に出し，その要素は誤差・近づいた／遠ざかったの数から外す．1 個でもあれば参照に対する指標（rel_fro・max_err の両側）は
// 評価できないので nan と出す（両方の Chi が同じ NaN だと参照のビット照合は通るので，ここで数えないと誤差の和が NaN に
// なっても rel_fro が 0 と出る）．
// 差・尺度・誤差の途中の計算は __float128 で行い，比を求めてから double にする（double のままだと，有限の結果どうしの
// 差 2^1023 - (-2^1023) や尺度 K max|a| max|b| の途中の K max|a| が溢れて無限大になる）．
// 出力は 1 行（CMP-DUMP ...）．読めない・形が違う・入力が違うときは終了コード 2．形は書き出し（oz_dump.hpp の
// dump_result）と同じかを見る：見出しの 4 語目が 0 であること，ファイルの長さが M・N から決まる長さとちょうど
// 同じであること（末尾に余分なデータがあれば，形式の違う書き出しとして断る）．
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <quadmath.h>
#include <vector>

namespace {
struct Dump {
    uint32_t M = 0, N = 0, K = 0;
    std::vector<double> C, Chi, Clo, am, bm;
};
bool load(const char* path, Dump& d) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "ERROR: %s を開けない\n", path); return false; }
    char magic[8]; uint32_t hdr[4];
    bool told = false;   // 理由を表示したか（したら末尾の一般の文を出さない）
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "OZDUMP1", 8) == 0 &&
              std::fread(hdr, 4, 4, f) == 4 && hdr[3] == 0;
    // 長さは M・N から決まる：8 + 16 + 8*(3*M*N + M + N) バイト．読む前に比べる（大きすぎる M・N で配列を取りに行かない．
    // 末尾の余分なデータも断る）
    if (ok) {
        const unsigned __int128 want = 24 + (unsigned __int128)8 * (3 * (unsigned __int128)hdr[0] * hdr[1] + hdr[0] + hdr[1]);
        const off_t here = ftello(f);
        ok = here >= 0 && fseeko(f, 0, SEEK_END) == 0;
        const off_t size = ok ? ftello(f) : -1;
        ok = ok && size >= 0 && (unsigned __int128)size == want && fseeko(f, here, SEEK_SET) == 0;
        if (!ok && size >= 0) {
            std::fprintf(stderr, "ERROR: %s の長さ %lld バイトが見出し（M=%u N=%u）から決まる長さと違う\n", path,
                         (long long)size, hdr[0], hdr[1]);
            told = true;
        }
    }
    if (ok) {
        d.M = hdr[0]; d.N = hdr[1]; d.K = hdr[2];
        const size_t n = (size_t)d.M * d.N;
        for (auto* v : {&d.C, &d.Chi, &d.Clo}) { v->resize(n); ok = ok && std::fread(v->data(), 8, n, f) == n; }
        d.am.resize(d.M); d.bm.resize(d.N);
        ok = ok && std::fread(d.am.data(), 8, d.M, f) == d.M && std::fread(d.bm.data(), 8, d.N, f) == d.N;
        ok = ok && std::fgetc(f) == EOF;
    }
    std::fclose(f);
    if (!ok && !told) std::fprintf(stderr, "ERROR: %s は OZ_DUMP_C の書き出しではない\n", path);
    return ok;
}
// 有限の x の ulp．frexp の指数 e から 2^(e-53) とするのは正規数（|x| >= 2^-1022）だけで，非正規化数と 0 は
// 2^-1074 で一定である（2^(e-53) のままだと 2^-1074 で e = -1073 となり 2^-1126 が 0 へ下位あふれして，1 ulp の差を
// 無限大と数える）
double ulp_of(double x) {
    x = std::fabs(x);
    const double tiny = std::ldexp(1.0, -1074);
    if (x < std::ldexp(1.0, -1022)) return tiny;
    int e; std::frexp(x, &e);
    return std::ldexp(1.0, e - 53);
}
// 2 つの配列がビット単位で同じか（NaN も同じビット列なら同じとみなす）
bool same_bits(const std::vector<double>& x, const std::vector<double>& y) {
    return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * 8) == 0);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s old.bin new.bin\n", argv[0]); return 2; }
    Dump a, b;
    if (!load(argv[1], a) || !load(argv[2], b)) return 2;
    if (a.M != b.M || a.N != b.N || a.K != b.K) { std::fprintf(stderr, "ERROR: 形が違う\n"); return 2; }
    const size_t n = (size_t)a.M * a.N;
    if (!same_bits(a.Chi, b.Chi) || !same_bits(a.Clo, b.Clo)) {
        std::fprintf(stderr, "ERROR: 参照が違う（入力が違う）\n"); return 2; }
    // 参照が同じでも入力が同じとは限らないので，行・列の最大絶対値も比べる（尺度 K max|a| max|b| は古い側の値で取る）
    if (!same_bits(a.am, b.am) || !same_bits(a.bm, b.bm)) {
        std::fprintf(stderr, "ERROR: A の行ごと・B の列ごとの最大絶対値が違う（入力が違う）\n"); return 2; }
    size_t ndiff = 0, closer = 0, farther = 0, tie = 0, nfa = 0, nfb = 0, nfr = 0;
    // 最大は __float128 で持ち，出力の直前に double にする（2^-53 を単位にした比は double の範囲を超えうる．そのときは inf）
    __float128 max_ulp = 0, max_scale = 0, max_err_a = 0, max_err_b = 0;
    __float128 num_a = 0, num_b = 0, den = 0;
    const __float128 u53 = 0x1p-53Q;
    for (size_t z = 0; z < n; ++z) {
        const bool fa = std::isfinite(a.C[z]), fb = std::isfinite(b.C[z]);
        nfa += !fa; nfb += !fb;
        const __float128 ref = (__float128)a.Chi[z] + (__float128)a.Clo[z];
        const bool fr = std::isfinite(a.Chi[z]) && std::isfinite(a.Clo[z]) && finiteq(ref);
        nfr += !fr;
        const __float128 ea = (__float128)a.C[z] - ref, eb = (__float128)b.C[z] - ref;
        const __float128 sc = (__float128)a.K * (__float128)a.am[z / a.N] * (__float128)a.bm[z % a.N];
        if (fr) {
            num_a += ea * ea; num_b += eb * eb; den += ref * ref;
            if (sc > 0) {
                if (fa) max_err_a = fmaxq(max_err_a, fabsq(ea) / sc / u53);
                if (fb) max_err_b = fmaxq(max_err_b, fabsq(eb) / sc / u53);
            }
        }
        if (std::memcmp(&a.C[z], &b.C[z], 8) == 0) continue;
        ++ndiff;
        if (!fa || !fb) continue;           // 有限でない側があれば差の大きさは測らない（nonfinite に数える）
        const __float128 d = fabsq((__float128)a.C[z] - (__float128)b.C[z]);
        max_ulp = fmaxq(max_ulp, d / (__float128)ulp_of(std::max(std::fabs(a.C[z]), std::fabs(b.C[z]))));
        if (sc > 0) max_scale = fmaxq(max_scale, d / sc / u53);
        if (!fr) continue;                  // 参照が有限でなければ近づいた／遠ざかったは決められない
        if (fabsq(eb) < fabsq(ea)) ++closer; else if (fabsq(eb) > fabsq(ea)) ++farther; else ++tie;
    }
    // 相対誤差（rel_fro）は有限でない要素が 1 個でもあれば評価できないので nan とする（先に分けないと，参照が全零のとき
    // NaN の誤差の和が「0 より大きい」に当たらず 0 と出る）．参照が全零なら，誤差が 0 のときだけ 0，そうでなければ inf
    auto rel_fro = [&](size_t nonfinite, __float128 num) {
        if (nonfinite || nfr) return (double)NAN;
        if (den > 0) return (double)sqrtq(num / den);
        return num > 0 ? (double)INFINITY : 0.0;
    };
    const double rf_a = rel_fro(nfa, num_a), rf_b = rel_fro(nfb, num_b);
    const double me_a = nfr ? (double)NAN : (double)max_err_a, me_b = nfr ? (double)NAN : (double)max_err_b;
    std::printf("CMP-DUMP M=%u N=%u K=%u differing=%zu/%zu (%.4f%%) max_diff=%.3g ulp (of C) =%.3g x2^-53 "
                "(of K max|a| max|b|) new_closer=%zu new_farther=%zu tie=%zu rel_fro old=%.10e new=%.10e "
                "max_err old=%.3g new=%.3g x2^-53 (of K max|a| max|b|) nonfinite old=%zu new=%zu ref=%zu\n",
                a.M, a.N, a.K, ndiff, n, 100.0 * (double)ndiff / (double)n, (double)max_ulp, (double)max_scale, closer,
                farther, tie, rf_a, rf_b, me_a, me_b, nfa, nfb, nfr);
    return 0;
}
