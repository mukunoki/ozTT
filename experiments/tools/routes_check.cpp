// SPDX-License-Identifier: Apache-2.0
//
// 経路の表 scripts/routes.tsv が本番の表と食い違わないかを確かめる【検査用】
// ---------------------------------------------------------------------------
//     make -C src routes_check            （コンテナの中で．フラグは src/Makefile の HOST_CXXFLAGS．build_all.sh も作る）
//     ./src/build/routes_check            （リポジトリ直下から．scripts/selftest.sh の hostcheck．引数は取らない）
// routes.tsv は測定スクリプトがバイナリへ渡す構成の正本で，ホストの検査（tools/tfmul_check.cpp・tfconv.cpp など）もそれを読んで
// 本番の構成を回す．読み方は tools/oz_sim.h（置き場所は OZ_ROUTES_TSV で変えられる）で，ここは次の 2 つを確かめる：
//   ・スキーム I の行が本番の 8 形式 1 回ずつで，L = p-1（oz_sim.h の load_routes1．欠けると検査が黙って少ない構成を回す）
//   ・スキーム II の行が本番の 8 形式 1 回ずつで，法の本数 s が本番の既定（ozaki2_split.h の OZ2_ROUTES の def_s．引数を
//     省いた実行が使う本数）と同じ（oz_sim.h の check_routes2_def_s．手で引数を省いた実行だけが別の構成になるのを防ぐ）
// 以前はこの 2 つを tools/tfmul_check.cpp（基数倍の FastTwoSum の前提の検査）が回していたので，外れると selftest に
// 「FastTwoSum の前提 FAIL」と出た．表の検査として独立させた．
// 終了コード：通れば 0，表が読めない・食い違えば 2（oz_sim.h の読み方は誤りを終了コード 2 で止める），引数があれば 2．
// 表と 16 経路を組むスクリプト・表を読まずに書いた既定の構成との突き合わせは scripts/verify/check_routes.py が行う．
#include <cstdio>
#include "oz_sim.h"

int main(int argc, char** argv) {
    if (argc != 1) { std::fprintf(stderr, "usage: %s（引数は取らない）\n", argv[0]); return 2; }
    const std::vector<oz_sim::Route1> r1 = oz_sim::load_routes1("routes_check");   // 8 形式 1 回ずつ・L = p-1（違えば 2）
    std::printf("経路の表 %s（スキーム I）：%zu 行が本番の 8 形式 1 回ずつで L = p-1：", oz_sim::routes_path(), r1.size());
    for (size_t i = 0; i < r1.size(); i++) std::printf("%s%s p=%d", i ? "，" : "", r1[i].key.c_str(), r1[i].p);
    std::printf("\n");
    oz_sim::check_routes2_def_s("routes_check");   // スキーム II（違えば 2．一致すれば 1 行を出す）
    std::printf("ROUTES-CHECK PASS\n");
    return 0;
}
