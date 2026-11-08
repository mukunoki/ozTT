#!/bin/bash
# measure/accuracy.sh -- FP64 相当の全実装の精度を，110 コア整合サイズで確認する
# ---------------------------------------------------------------------------
#     ./scripts/measure/accuracy.sh [サイズ…]                  # ホストから（既定 3520 7040 14080）
#     ONLY='oz1bf16|oz1fp8' ./scripts/measure/accuracy.sh       # ログ名の正規表現に当たる項目だけを測る
#     ./scripts/measure/accuracy.sh --list [サイズ…]           # 測らずに，項目ごとにログ名・表示名・コンテナへ渡すコマンドを出す
# 1 項目ずつコンテナを起こして流す（冷却は待たない．到達精度は温度に依らない．measure_lib.sh の run_item）．出どころの行・
# 実行ファイルの行・流した後の確認と完了の記録は他のホストの測定と同じ（scripts/README.md「ログの出どころ」）．
# accuracy_logs/ はリポジトリが追跡しているので，各項目は一時ファイルに書いて最後まで書けたときだけログへ置き換える
# （run_item --replace．途中で止められても前の完全なログを残す）．
# 論文用の測り直しは STEPS="acc" ./scripts/remeasure_all.sh（測った後に元の作業木でコミットしたなら
# REMEASURE_AT=<測った rev> も付ける）．
#
# 参照は host_eval.hpp の Dot2（誤差なし変換による補償付き総和）で，全サイズで同じ
# ものを使う．並列化・ブロック化してあるので 14080^3 でも 2 分程度で終わる．
#
# 参照の精度は phi=0--16・K=1024/14080 で相対誤差 2^-85 以下（tools/dot2check.cpp）で，判定したい 53 bit に対して
# 32 bit の余裕がある．したがって，検査した入力条件（ここで使う一様乱数と phi=0--16 の入力）では参照の誤差は
# 報告値に対して十分小さい．未検査の入力（強い相殺で条件数が大きいものなど）で参照の誤差が増えないことまでは
# 保証しない（一般的な保証には条件数を含む誤差評価が要る）．
#
# II-TF32 の 14080^3 は，法の組分け（ozaki2_main.cpp の plan_mod_groups．自動で 2 組）で DRAM に
# 収まる．落ちれば他の経路と同じく FAILED（ONLY=oz2fp32_14080 でこの 1 本だけも測れる．remeasure_all.sh からは
# STEPS="acc" REMEASURE_ONLY='^oz2fp32_14080$'）．
set -u
cd "$(dirname "$0")/../.."
LIST=0
if [ "${1:-}" = --list ]; then LIST=1; shift; fi
SIZES="${*:-3520 7040 14080}"
for N in $SIZES; do
  case $N in *[!0-9]*|0) echo "ERROR: 大きさ $N は正の整数でない（使い方: measure/accuracy.sh [--list] [サイズ…]）" >&2; exit 2 ;; esac
done
[ -n "${SIZES// /}" ] || { echo "ERROR: 大きさが空（使い方: measure/accuracy.sh [--list] [サイズ…]）" >&2; exit 2; }
OUT=accuracy_logs; [ "$LIST" = 1 ] || mkdir -p "$OUT" || exit 2
# 測る前の確認（provenance_check：src/build が今の rev のビルドか）・デバイスの排他・ONLY が正規表現として読めるかの確認は
# measure_lib.sh が読み込んだ時点で行う（一覧だけなら定義だけを読む）
if [ "$LIST" = 1 ]; then . ./scripts/measure_lib.sh --defs-only; else . ./scripts/measure_lib.sh; fi   # prov_begin・prov_bin・prov_exit・only_summary
. ./scripts/routes.sh        # routes_each_in・routes_row_cmd・OZ_MEAS_ORDER

# run ラベル ログ名 -- コマンド...
#   コマンドをコンテナで実行し，全出力を accuracy_logs/ログ名.log に残して精度行を表示する
#   終了コードと精度行の両方を見て，どちらか欠けたら fail を立てる（最後に非零で終わる）．
#   検査側は PROVENANCE 行のあるログに EXIT,rc=0 と VERIFY,…,pass の両方を要る（検証を省く OZ_NOVERIFY などの
#   つまみは，run.sh がコンテナの中でコマンドの前に外す）
fail=0
nrun=0
run() {
  label=$1; log=$2; shift 3
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$log" | grep -Eq -- "$ONLY"; then return; fi
  nrun=$((nrun + 1))
  if [ "$LIST" = 1 ]; then printf '%s\t%s\t%s\n' "$log" "$label" "cd /home/user/tt && $*"; return; fi
  run_item --replace "$OUT/$log.log" "cd /home/user/tt && $*"
  rc=$RI_RC
  acc=$(grep -ohE '\|\|C-(ref|fp64)\|\|.*' "$OUT/$log.log" | head -1)
  # 終了コードと VERIFY 行の両方を見る．プログラム側が到達精度を基準（既定 53 bit）と
  # 突き合わせて終了コードへ反映するので，精度不足はここで落ちる．VERIFY 行が無い・
  # skipped（検証を省いた）など pass 以外はすべて失敗とする．
  ver=$(grep -ohE '^VERIFY,[^,]+,[a-z]+' "$OUT/$log.log" | head -1 | cut -d, -f3)
  if [ $rc -ne 0 ] || [ -z "$acc" ] || [ "$ver" != "pass" ]; then
    fail=1
    printf '  %-14s FAILED (exit %s, verify %s, see %s)\n' "$label" "$rc" "${ver:-なし}" "$OUT/$log.log"
  else
    printf '  %-14s %s\n' "$label" "$acc"
  fi
}

# 16 経路は routes.tsv から引く（表示名・ログ名の幹と，表の既定の構成 p・L か法の本数．ウォームアップ 1・計時 1）．
# 回す順は measure/performance.sh と同じ routes.sh の OZ_MEAS_ORDER（到達精度は順に依らない）．ログ名は <表の log>_<N>
acc_route() { run "$r_label" "${r_log}_$N" -- $(routes_row_cmd "$N" 1 1); }
for N in $SIZES; do
  [ "$LIST" = 1 ] || echo "===== N=$N  (cubic M=N=K, reference = Dot2) ====="
  routes_each_in "$OZ_MEAS_ORDER" acc_route || exit 2
done
[ "$LIST" = 1 ] && exit 0
echo "(全出力は $OUT/ に残る)"
# ONLY に 1 項目も当たらなかったときは注意を出す（終了コードは変えない：remeasure_all.sh の REMEASURE_ONLY は
# 段ごとの測定スクリプトへ同じ ONLY を渡し，当たる項目の無い段があってよい．何も測らなかったことは remeasure_all.sh が
# 書いたログの本数で見分ける）．ONLY を与えずに 1 項目も流れないのは表の誤りなので止める
only_summary
if [ "$nrun" = 0 ] && [ -z "${ONLY:-}" ]; then echo "ERROR: 流した項目が無い" >&2; exit 2; fi
exit $fail
