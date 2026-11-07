// SPDX-License-Identifier: Apache-2.0
//
// knobs_check：つまみの表（src/common/oz_knobs.hpp）が docs/env_vars.md と src・tools のソースと食い違わないかを確かめる
// ---------------------------------------------------------------------------
//     make -C src knobs_check && ./src/build/knobs_check [docs/env_vars.md [src tools ...]]   （リポジトリ直下から．selftest.sh）
//     ./src/build/knobs_check --dump                                                       （表を TSV で出す）
// 確かめること（どれか 1 つでも外れれば終了コード 1．読めない引数・ファイル・ディレクトリの列挙の誤りは 2）：
//   1. docs/env_vars.md の節（## A.〜## F.）の表の 1 列目と D の節の 1 段落目に挙げたつまみの名前と，末尾の「削除した」
//      段落に挙げた名前が，表の名前と分類にちょうど一致する（docs にだけある名前・表にだけある名前・分類の違いを出す）
//   2. src・tools のソースがつまみを読む所（getenv・oz_env::<関数>・OZK・env_flag01 に名前の文字列を渡す所）の名前が
//      どれも表にある（表に無い名前を読んでいれば，knobs・diag・スキームをまたぐ拒否・削除の拒否から漏れる）．
//      表にあってどのソースも読まない名前も出す（削除したつまみは除く．表は読む所の写しではなく正本なので，読まなく
//      なったつまみは表から外すか削除したつまみへ移す）
//      reject_named に渡す名前は拒否の対象なので，表の存在だけを見て，who の読者には数えない．
//   3. 読む所の置き場所から決まるプログラム（下の reader_of．src/ozaki1 は ozaki1_*，src/ozaki2 は ozaki2_*，src/baseline の
//      各ファイル，src/checks の gemm_exact_check とそれ以外，src/probes，tools の dgemmcheck とそれ以外）が，表の読むプログラム
//      （who）に入っている．who は入口の拒否（oz_knobs::reject_unread・reject_other_scheme）と preflight を決めるので，読む
//      プログラムが who に無いと，そのプログラムが読むつまみを入口が「読まない」として止めるか，範囲の検査を飛ばす．複数の
//      プログラムが取り込む共通の部品（src/common・src/testkit・src/checks/sfpu_tf_emu.hpp）の読み取りは置き場所から
//      決まらないので，この項では見ない
//   4. 範囲を表で持たない列挙（ranged=false の Choice．読む所が oz_env::choice に名前を並べる）の並びが，表の choices と
//      同じ（片方だけを直すと，記録・検査と読み取りの受け付ける名前が食い違う）
// ソースの読み取りの数え方は，コメントと #if 0 の中を除いた文字列の字面だけを見る（名前を実行時に組む読み方は数えない．表の
// 名前でまとめて読む oz_knobs.hpp の関数自体は名前を書かないので数えない）．コメントの除き方は scripts/lib/cpp_defs.py の
// strip_comments と同じ規則（文字列・文字の定数の中は残す．// の行末がバックスラッシュなら次の行もコメント）．
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "common/oz_knobs.hpp"

namespace fs = std::filesystem;
using oz_knobs::Cls;

namespace {

const char* kind_name(oz_knobs::Kind k) {
    switch (k) {
        case oz_knobs::Kind::Set: return "set";   case oz_knobs::Kind::U32: return "u32";
        case oz_knobs::Kind::I64: return "i64";   case oz_knobs::Kind::F64: return "f64";
        case oz_knobs::Kind::Choice: return "choice"; default: return "str";
    }
}

std::string who_text(uint16_t w) {
    static const std::pair<uint16_t, const char*> names[] = {
        {oz_knobs::W_I, "ozaki1"}, {oz_knobs::W_II, "ozaki2"}, {oz_knobs::W_BF16, "gemm_bf16"},
        {oz_knobs::W_EXT, "gemm_bf16ext"}, {oz_knobs::W_I8, "gemm_int8"}, {oz_knobs::W_EXACT, "gemm_exact_check"},
        {oz_knobs::W_CHECK, "checks"}, {oz_knobs::W_PROBE, "probes"}, {oz_knobs::W_TOOL, "tools"},
        {oz_knobs::W_DGEMM, "dgemmcheck"}};
    std::string s;
    for (const auto& [b, n] : names)
        if (w & b) s += (s.empty() ? "" : "+") + std::string(n);
    return s.empty() ? "-" : s;
}

void dump() {
    std::printf("name\tclass\twho\tkind\trange\tdefault\trecord\tnote\n");
    for (int i = 0; i < oz_knobs::NKNOB; ++i) {
        const oz_knobs::Knob& k = oz_knobs::TABLE[i];
        std::string range = "-", dflt = "-", rec;
        if (k.kind == oz_knobs::Kind::Choice && k.choices) range = k.choices;
        else if (k.ranged) range = std::to_string((long long)k.lo) + ".." + std::to_string((long long)k.hi);
        else if (k.kind != oz_knobs::Kind::Set && k.kind != oz_knobs::Kind::Str) range = "reader";
        if (k.has_dflt) dflt = std::to_string((long long)k.dflt);
        if (k.rec & oz_knobs::R_KNOBS) rec += "knobs";
        if (k.rec & oz_knobs::R_DIAG) rec += (rec.empty() ? "" : "+") + std::string("diag");
        if (k.rec & oz_knobs::R_FIELD) rec += (rec.empty() ? "" : "+") + std::string("field");
        std::printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", std::string(k.name).c_str(), oz_knobs::cls_name(k.cls),
                    who_text(k.who).c_str(), kind_name(k.kind), range.c_str(), dflt.c_str(), rec.empty() ? "-" : rec.c_str(),
                    k.note ? k.note : "-");
    }
}

// `NAME` か `NAME=…` の名前の部分（大文字で始まり英大文字・数字・_ だけ．末尾は _ でない）
std::vector<std::string> backticked_names(const std::string& s) {
    static const std::regex re("`([A-Z][A-Z0-9_]*[A-Z0-9])(=[^`]*)?`");
    std::vector<std::string> out;
    for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it)
        out.push_back((*it)[1].str());
    return out;
}

// C++ のソースからコメントと #if 0 の中を除く（上の冒頭の規則．改行は残す）
std::string strip_cpp(const std::string& src) {
    std::string out;
    const size_t n = src.size();
    size_t i = 0;
    auto lit_end = [&](size_t j) -> size_t {   // src[j] が '"' か '\'' なら，その定数の終わりの次の位置（生の文字列は R"(…)"）
        if (src[j] == 'R' && j + 1 < n && src[j + 1] == '"') {
            const size_t q = src.find('(', j + 2);
            if (q == std::string::npos) return n;
            const std::string end = ")" + src.substr(j + 2, q - (j + 2)) + "\"";
            const size_t e = src.find(end, q + 1);
            return e == std::string::npos ? n : e + end.size();
        }
        const char qc = src[j];
        size_t k = j + 1;
        while (k < n && src[k] != qc) k += (src[k] == '\\') ? 2 : 1;
        return std::min(n, k + 1);
    };
    while (i < n) {
        if (src.compare(i, 2, "//") == 0) {
            size_t j = i;
            while (true) {
                j = src.find('\n', j);
                if (j == std::string::npos) break;
                size_t b = j;
                while (b > i && src[b - 1] == '\r') --b;
                if (b > i && src[b - 1] == '\\') { ++j; continue; }
                break;
            }
            i = (j == std::string::npos) ? n : j;
        } else if (src.compare(i, 2, "/*") == 0) {
            const size_t j = src.find("*/", i + 2);
            for (size_t k = i; k < (j == std::string::npos ? n : j); ++k) if (src[k] == '\n') out += '\n';
            out += ' ';
            i = (j == std::string::npos) ? n : j + 2;
        } else if (src[i] == '"' || src[i] == '\'' || (src[i] == 'R' && i + 1 < n && src[i + 1] == '"' &&
                                                         (i == 0 || !(std::isalnum((unsigned char)src[i - 1]) || src[i - 1] == '_')))) {
            const size_t e = lit_end(i);
            out.append(src, i, e - i);
            i = e;
        } else {
            out += src[i++];
        }
    }
    // #if 0 … #endif（入れ子の #if を数える．#else 以降は残す）
    std::istringstream is(out);
    std::string res, line;
    int skip = 0;   // #if 0 の中の深さ（0 なら外）
    while (std::getline(is, line)) {
        std::string t = line;
        t.erase(0, t.find_first_not_of(" \t"));
        const bool pp = !t.empty() && t[0] == '#';
        std::string d;
        if (pp) { t.erase(0, 1); t.erase(0, t.find_first_not_of(" \t")); d = t; }
        if (skip) {
            if (pp && d.rfind("if", 0) == 0) ++skip;
            else if (pp && d.rfind("endif", 0) == 0) --skip;
            else if (pp && skip == 1 && (d.rfind("else", 0) == 0 || d.rfind("elif", 0) == 0)) skip = 0;
            res += '\n';
            continue;
        }
        if (pp && (d == "if 0" || d.rfind("if 0 ", 0) == 0)) { skip = 1; res += '\n'; continue; }
        res += line + '\n';
    }
    return res;
}

// 読む所の置き場所から決まるプログラム（表の Who のビット．上の冒頭の 3．共通の部品なら 0）
uint16_t reader_of(const fs::path& p) {
    const std::string s = p.generic_string();
    auto has = [&](const char* d) { return s.find(d) != std::string::npos; };
    const std::string f = p.filename().string();
    if (has("/common/") || has("/testkit/") || f == "sfpu_tf_emu.hpp") return 0;
    if (has("/ozaki1/")) return oz_knobs::W_I;
    if (has("/ozaki2/")) return oz_knobs::W_II;
    if (has("/baseline/")) {
        if (f == "gemm_bf16.cpp") return oz_knobs::W_BF16;
        if (f == "gemm_bf16ext.cpp") return oz_knobs::W_EXT;
        if (f == "gemm_int8.cpp") return oz_knobs::W_I8;
        return 0;
    }
    if (has("/checks/")) return f == "gemm_exact_check.cpp" ? oz_knobs::W_EXACT : oz_knobs::W_CHECK;
    if (has("/probes/")) return oz_knobs::W_PROBE;
    if (s.rfind("tools/", 0) == 0 || has("/tools/")) return f == "dgemmcheck.cpp" ? oz_knobs::W_DGEMM : oz_knobs::W_TOOL;
    return 0;
}

bool section_of(const std::string& line, std::string& sec) {
    static const std::regex re("^## (A|B|C|D2|D|E|F)\\.");
    std::smatch m;
    if (line.rfind("## ", 0) != 0) return false;
    sec = std::regex_search(line, m, re) ? m[1].str() : std::string();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--dump")) { dump(); return 0; }
    const std::string docs = argc > 1 ? argv[1] : "docs/env_vars.md";
    std::vector<std::string> roots;
    for (int i = 2; i < argc; ++i) roots.push_back(argv[i]);
    if (roots.empty()) roots = {"src", "tools"};
    int bad = 0;
    auto report = [&](const std::string& why) { std::printf("MISMATCH %s\n", why.c_str()); ++bad; };

    // ---- 1. docs/env_vars.md ----
    std::ifstream f(docs);
    if (!f) { std::fprintf(stderr, "ERROR: %s を読めない（リポジトリ直下から回す）\n", docs.c_str()); return 2; }
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l); ) lines.push_back(l);
    size_t last_rule = 0;   // 最後の "---" の行（その後ろに「削除した」段落がある）
    for (size_t i = 0; i < lines.size(); ++i) if (lines[i] == "---") last_rule = i;
    std::map<std::string, std::string> doc_cls;   // 名前 -> 節
    auto put = [&](const std::string& n, const std::string& c) {
        auto it = doc_cls.find(n);
        if (it != doc_cls.end() && it->second != c)
            report("docs で " + n + " が 2 つの節（" + it->second + "・" + c + "）に挙がっている");
        doc_cls[n] = c;
    };
    std::string sec;
    bool d_para = false;   // D の節の 1 段落目を読んだか
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& l = lines[i];
        std::string s2;
        if (section_of(l, s2)) { sec = s2; d_para = false; continue; }
        if (i > last_rule && last_rule) { sec.clear(); }
        if (sec.empty()) continue;
        if (l.rfind("| `", 0) == 0) {          // 表の行：1 列目の名前
            const size_t e = l.find('|', 1);
            for (const auto& n : backticked_names(l.substr(1, e == std::string::npos ? std::string::npos : e - 1))) put(n, sec);
        } else if (sec == "D" && !d_para && l.rfind("`", 0) == 0) {   // D の節は表を持たず，1 段落目に名前を並べる
            for (const auto& n : backticked_names(l)) put(n, sec);
            d_para = true;
        }
    }
    // 削除したつまみ：最後の "---" より後ろの「削除した」段落の，一覧（「削除した…:」の後から最初の「。」まで）．値の範囲だけを
    // 削除したもの（`OZ2_CRT_PROBE=3..5` のように今もあるつまみに = を付けた名前）は数えない
    std::set<std::string> doc_removed;
    for (size_t i = last_rule + 1; last_rule && i < lines.size(); ++i) {
        const size_t k = lines[i].find("削除した");
        if (k == std::string::npos) continue;
        const size_t c = lines[i].find(": ", k);
        if (c == std::string::npos) continue;
        const size_t e = lines[i].find("。", c);
        const std::string list = lines[i].substr(c, e == std::string::npos ? std::string::npos : e - c);
        static const std::regex re("`([A-Z][A-Z0-9_]*[A-Z0-9])(=[^`]*)?`");
        for (auto it = std::sregex_iterator(list.begin(), list.end(), re); it != std::sregex_iterator(); ++it) {
            const std::string n = (*it)[1].str();
            const int id = oz_knobs::index_of(n);
            if ((*it)[2].matched && id >= 0 && oz_knobs::TABLE[id].cls != Cls::Removed) continue;
            doc_removed.insert(n);
        }
    }
    int ntab = 0;
    for (int i = 0; i < oz_knobs::NKNOB; ++i) {
        const oz_knobs::Knob& k = oz_knobs::TABLE[i];
        const std::string n(k.name);
        ++ntab;
        if (k.cls == Cls::Removed) {
            if (!doc_removed.count(n)) report("表の削除したつまみ " + n + " が docs の「削除した」段落に無い");
            if (doc_cls.count(n)) report("表の削除したつまみ " + n + " が docs の節 " + doc_cls[n] + " にも挙がっている");
            continue;
        }
        auto it = doc_cls.find(n);
        if (it == doc_cls.end()) report("表の " + n + "（" + oz_knobs::cls_name(k.cls) + "）が docs のどの節の表にも無い");
        else if (it->second != oz_knobs::cls_name(k.cls))
            report("表の " + n + " の分類 " + oz_knobs::cls_name(k.cls) + " と docs の節 " + it->second + " が違う");
    }
    for (const auto& [n, c] : doc_cls)
        if (oz_knobs::index_of(n) < 0) report("docs の節 " + c + " の " + n + " が表に無い");
    for (const auto& n : doc_removed) {
        const int id = oz_knobs::index_of(n);
        if (id < 0 || oz_knobs::TABLE[id].cls != Cls::Removed) report("docs の削除した " + n + " が表の削除したつまみに無い");
    }

    // ---- 2. ソースがつまみを読む所 ----
    // 表に無くてよいもの：TT_DEVICE（run.sh・tt-metal のデバイスの指定．telemetry.hpp がカードの番号を読む）
    static const std::set<std::string> external = {"TT_DEVICE"};
    static const std::regex rd("(getenv|oz_env::[A-Za-z0-9_]+|OZK|env_flag01|oz_knobs::reject_named)\\s*\\(\\s*\"([A-Z][A-Z0-9_]*[A-Z0-9])\"");
    // 範囲を表で持たない列挙の読み取り：oz_env::choice("NAME", {"a", "b"} か {arr[0], arr[1]}, …)
    static const std::regex rc("oz_env::choice\\s*\\(\\s*\"([A-Z][A-Z0-9_]*[A-Z0-9])\"\\s*,\\s*\\{([^}]*)\\}");
    std::map<std::string, std::string> read_at;   // 名前 -> 最初に読む所
    std::map<std::string, std::pair<std::string, std::string>> choices_at;   // 名前 -> (並び，読む所)
    int nread = 0;
    for (const auto& r : roots) {
        std::error_code ec;
        if (!fs::is_directory(r, ec)) { std::fprintf(stderr, "ERROR: %s はディレクトリでない\n", r.c_str()); return 2; }
        auto it = fs::recursive_directory_iterator(r, ec);
        if (ec) { std::fprintf(stderr, "ERROR: %s を列挙できない（%s）\n", r.c_str(), ec.message().c_str()); return 2; }
        for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) { std::fprintf(stderr, "ERROR: %s の列挙が途中で失敗した（%s）\n", r.c_str(), ec.message().c_str()); return 2; }
            const fs::path& p = it->path();
            const std::string ext = p.extension().string();
            if (!it->is_regular_file() || (ext != ".cpp" && ext != ".hpp" && ext != ".h")) continue;
            if (p.filename() == "oz_knobs.hpp" || p.filename() == "knobs_check.cpp") continue;   // 表とこの検査自身
            std::ifstream g(p, std::ios::binary);
            if (!g) { std::fprintf(stderr, "ERROR: %s を開けない\n", p.string().c_str()); return 2; }
            std::stringstream ss; ss << g.rdbuf();
            if (g.bad() || (!g.eof() && g.fail())) { std::fprintf(stderr, "ERROR: %s を読み切れない\n", p.string().c_str()); return 2; }
            const std::string t = strip_cpp(ss.str());
            const uint16_t prog = reader_of(p);
            for (auto m = std::sregex_iterator(t.begin(), t.end(), rd); m != std::sregex_iterator(); ++m) {
                const std::string n = (*m)[2].str();
                ++nread;
                if ((*m)[1].str() != "oz_knobs::reject_named" && !read_at.count(n)) read_at[n] = p.string();
                if (external.count(n)) continue;
                const int id = oz_knobs::index_of(n);
                if (id < 0) report(p.string() + " が表に無いつまみ " + n + " を読む");
                else if (oz_knobs::TABLE[id].cls == Cls::Removed) report(p.string() + " が削除したつまみ " + n + " を読む");
                else if ((*m)[1].str() != "oz_knobs::reject_named" && prog && !(oz_knobs::TABLE[id].who & prog))
                    report(p.string() + "（" + who_text(prog) + "）が " + n + " を読むが，表の読むプログラム（" +
                           who_text(oz_knobs::TABLE[id].who) + "）に無い");
            }
            // 列挙の並び（上の冒頭の 4）．要素が配列の添字（kSigns[0]）なら同じファイルの配列の定義から引く
            for (auto m = std::sregex_iterator(t.begin(), t.end(), rc); m != std::sregex_iterator(); ++m) {
                const std::string n = (*m)[1].str(), body = (*m)[2].str();
                std::string list;
                static const std::regex item("\\s*(?:\"([^\"]*)\"|([A-Za-z_][A-Za-z0-9_]*)\\s*\\[\\s*([0-9]+)\\s*\\])\\s*(,|$)");
                bool ok = true;
                for (auto q = std::sregex_iterator(body.begin(), body.end(), item); q != std::sregex_iterator(); ++q) {
                    std::string v;
                    if ((*q)[1].matched) v = (*q)[1].str();
                    else {
                        const std::regex arr((*q)[2].str() + "\\s*\\[\\s*\\]\\s*=\\s*\\{([^}]*)\\}");
                        std::smatch am;
                        std::vector<std::string> el;
                        if (std::regex_search(t, am, arr)) {
                            const std::string ab = am[1].str();
                            static const std::regex sl("\"([^\"]*)\"");
                            for (auto e = std::sregex_iterator(ab.begin(), ab.end(), sl); e != std::sregex_iterator(); ++e)
                                el.push_back((*e)[1].str());
                        }
                        const size_t k = (size_t)std::stoul((*q)[3].str());
                        if (k >= el.size()) { ok = false; break; }
                        v = el[k];
                    }
                    list += (list.empty() ? "" : "|") + v;
                }
                if (!ok || list.empty()) { report(p.string() + " の " + n + " の列挙の並びが読めない"); continue; }
                choices_at[n] = {list, p.string()};
            }
        }
    }
    for (const auto& [n, lp] : choices_at) {
        const int id = oz_knobs::index_of(n);
        if (id < 0) continue;   // 表に無い名前は上で出した
        const oz_knobs::Knob& k = oz_knobs::TABLE[id];
        if (k.kind != oz_knobs::Kind::Choice || k.ranged)
            report(lp.second + " が " + n + " を oz_env::choice で読むが，表では範囲を表で持たない列挙（ranged=false の Choice）でない");
        else if (!k.choices || lp.first != k.choices)
            report(lp.second + " の " + n + " の列挙 " + lp.first + " が表の choices " + (k.choices ? k.choices : "（無し）") + " と違う");
    }
    for (int i = 0; i < oz_knobs::NKNOB; ++i) {
        const oz_knobs::Knob& k = oz_knobs::TABLE[i];
        if (k.cls != Cls::Removed && k.kind == oz_knobs::Kind::Choice && !k.ranged && !choices_at.count(std::string(k.name)))
            report("表の " + std::string(k.name) + " は範囲を表で持たない列挙だが，oz_env::choice に名前を並べて読む所が無い");
    }
    for (int i = 0; i < oz_knobs::NKNOB; ++i) {
        const oz_knobs::Knob& k = oz_knobs::TABLE[i];
        if (k.cls != Cls::Removed && !read_at.count(std::string(k.name)))
            report("表の " + std::string(k.name) + " をどのソースも読まない（" + [&] {
                std::string s; for (const auto& r : roots) s += (s.empty() ? "" : "・") + r; return s; }() + "）");
    }

    std::printf("合計 %d 通り（表 %d 本，docs の名前 %zu 本と削除した %zu 本，ソースの読み取り %d か所）%s\n",
                ntab + (int)doc_cls.size() + (int)doc_removed.size() + nread, ntab, doc_cls.size(), doc_removed.size(), nread,
                bad ? "" : "．食い違い無し");
    if (bad) std::printf("食い違い %d 件（表 src/common/oz_knobs.hpp・docs/env_vars.md・ソースのどれかを直す）\n", bad);
    return bad ? 1 : 0;
}
