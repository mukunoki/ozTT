#!/bin/bash
# measure/exponent_adaptive.sh -- 指数部の範囲を広げた入力（φ）で p・b を入力から選び直した構成の到達精度と性能（14080^3）を，
#                         スキーム II の窓を表の構成と同じ最悪の上界 K 2^{2b} に固定して（OZ2_NOWIN=1）測る
# ---------------------------------------------------------------------------
#     ./scripts/measure/exponent_adaptive.sh                  # ホストから（冷却に tt-smi を使う）．既定の φ は 0.5 1 2 4 8
#                                                     # （lib/constants.tsv の PHIN_PHIS．所要時間は remeasure_all.sh の冒頭の実績）
#     PHIS="0.5 1" ./scripts/measure/exponent_adaptive.sh     # φ を選ぶ
#     ONLY='^acc_' ./scripts/measure/exponent_adaptive.sh     # ログ名の正規表現で項目を選ぶ（他の測定スクリプトと同じ）
#     ./scripts/measure/exponent_adaptive.sh --list           # 既定の φ で測る項目のログ名を 1 行ずつ出すだけ（測らない．
#                                                     # remeasure_all.sh が組の項目が揃ったかを見るのに使う）
#     ./scripts/measure/exponent_adaptive.sh --commands       # 測らずに，項目ごとにログ名・形・コンテナへ渡すコマンドを出す
#                                                     # （排他も出どころの確認もしない．直す前後で流すコマンドを比べる）
#     ./scripts/measure/exponent_adaptive.sh --part accuracy --commands # 精度だけ（performance は性能だけ）
#     DGEMMREF_CORETYPE=ZEN DGEMMREF_SET=phi14080 ./scripts/measure/cpu_dgemm_reference.sh
#                                                     # 比べる CPU の DGEMM（measure_logs/probe_dgemm_phi_14080.log へ書く）
#   remeasure_all.sh の phi_nowin の段がこれを，dgemm の段が CPU の DGEMM を流す．
#   ONLY は測った rev のままの測り直しにだけ使う．ログは rev ごとのディレクトリに置き，検査（verify/check_paper_numbers.py の
#   load_phi_nowin）は同じ項目のログが 2 つの rev にあればどちらも使わず，組の中で rev が揃っていなければ組全体を拒む．
#   したがって rev が変わった後に一部の項目だけを測ると組が壊れる．rev が変わったら全項目を測り直し，古い rev の
#   ディレクトリ measure_logs/phi_nowin/<古い rev7>/ を measure_logs/phi_nowin の外へ退避する（remeasure_all.sh の
#   phi_nowin の段は，段が成功し，この rev の組の全項目（--list）が揃ったときだけ退避する）．
#   <rev7> は HEAD の先頭ちょうど 7 桁（git rev-parse HEAD | cut -c1-7．--short=7 は他のオブジェクトと重なると 8 桁
#   以上になり，検査がちょうど 7 桁を要るのと食い違う）
#
# 論文（PDP 版の 7.2 節・フル版の 8.1 節 ssec:accu の φ の段落）とスライドの「（参考）φ を広げた入力で…」の頁が引く，φ を広げた入力での p・b と到達精度・再構成の回数・実効
# スループットの出どころ．
# 表の構成（既定の入力．measure/performance.sh）のスキーム II は，窓を |C'| の最悪の上界 K 2^{2b} として法の集合を決める．
# 以前の φ の測定（diagnose/probes.sh の probe_phiacc_*・probe_phiperf_*．OZ2_FP64=1）は窓も入力から見積もった
# （Cauchy--Schwarz．ozaki2_main.cpp の choose_precision_policy）ので法が小さく，φ=0.5 の Ozaki-II-BFP8 が表の 4.64 を
# 上回る 4.75 TFlops/s になり，論文は「既定の構成で最大 4.64」と書き分けていた（利用者の判断 2026-10-20：再実験で 1 つの
# 構成にそろえる）．ここでは窓だけを表の構成と同じ上界に戻し，入力に合わせて変えるのを p・b（OZ_FP64・OZ2_FP64 の規則）と
# スキーム II の再構成の回数だけにする．スキーム I には窓が無いが，同じビルドでそろえるため Ozaki-I-BFP8 も測る．
#
# 項目（ログは measure_logs/phi_nowin/<rev7>/<項目>.log．1 項目 1 本．1 行目は出どころの PROVENANCE 行，2 行目は
# 実行するバイナリの BINARY 行，最後に EXIT,rc=…）
#   acc_oz1bfp8_phi<φ>         Ozaki-I-BFP8 の到達精度（OZ_FP64=1．冷まさない．検証あり．参照は host_eval の Dot2．
#                              II と同じく OZ_MIN_BITS=1 にして，53 bit を割っても値を残し，判定は検査側に任せる．
#                              付けずに測った前の rev のログもあったが，どの φ も 53 bit 以上で値は同じだった）
#   acc_oz2bfp8_phi<φ>_p<n>    Ozaki-II-BFP8 の到達精度（OZ2_FP64=1・OZ2_NOWIN=1，再構成を n 回に固定（OZ2_PASSES=n））．
#                              1 回で 53 bit を割る φ があるので OZ_MIN_BITS=1 にして，精度不足を異常終了にしない
#                              （FP64 相当かどうかは検査側が CPU の DGEMM の値と比べて判定する）
#   perf_oz1bfp8_phi<φ>        Ozaki-I-BFP8 の性能（measure_lib.sh の run_dram：冷ましてから定常計時．OZ_NOVERIFY=1）
#   perf_oz2bfp8_phi<φ>_p<n>   Ozaki-II-BFP8 の性能（再構成を n 回に固定．どの回数で精度が足りるかは acc_* で決まるので両方測る）
# 引数は diagnose/probes.sh の φ の項目と同じ（I は p の初期値，II は法の最大本数で，どちらも routes.tsv の既定の構成
# （routes.sh の routes_args．今は I-BFP8 の 10・9，II-BFP8 の 24）．精度は warmup 1・iters 1，性能は warmup 1・iters 3）．同じ入力の CPU の DGEMM の値は DGEMMREF_CORETYPE=ZEN DGEMMREF_SET=phi14080 ./scripts/measure/cpu_dgemm_reference.sh
# （出力は measure_logs/probe_dgemm_phi_14080.log．検査は FMA を用いるカーネルで測ったものだけを使うので，
# OpenBLAS 0.3.20 が Zen 5 に選ぶ FMA の無い Barcelona を DGEMMREF_CORETYPE=ZEN で避ける）．
set -u
cd "$(dirname "$0")/../.."
# 既定の φ の点は lib/constants.tsv の PHIN_PHIS（比べる CPU の DGEMM の measure/cpu_dgemm_reference.sh の phi14080 と照合の
# verify/check_paper_numbers.py の PHIN_PHIS も同じ行を読む）
. ./scripts/lib/consts.sh
PHIN_DEFAULT=$(const_get PHIN_PHIS) || exit 2
PHIS="${PHIS:-$PHIN_DEFAULT}"
# φ は空白区切りの 0 以上の数で，少なくとも 1 つ（空白だけの PHIS で 1 項目も流さずに 0 で終わらないように）
read -r -a _p <<<"$PHIS"
[ ${#_p[@]} -gt 0 ] || { echo "ERROR: PHIS が空（φ を空白区切りで与える．既定 \"$PHIN_DEFAULT\"）" >&2; exit 2; }
for _x in "${_p[@]}"; do
  [[ $_x =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ERROR: PHIS の $_x は 0 以上の数でない" >&2; exit 2; }
done
unset _p _x
# 項目のログ名（下の 2 つのループと同じ並び．--list と remeasure_all.sh の揃ったかの判定が使う）
PART=all
if [ "${1:-}" = --part ]; then
  PART=${2:-}; [ "$#" -ge 2 ] || { echo 'ERROR: --part に accuracy または performance が必要' >&2; exit 2; }; shift 2
fi
case $PART in all|accuracy|performance) ;; *) echo "ERROR: 未知の測定種別 $PART" >&2; exit 2 ;; esac
items() {
  local phi np
  if [ "$PART" != performance ]; then
    for phi in $PHIS; do echo "acc_oz1bfp8_phi${phi}"; for np in 1 2; do echo "acc_oz2bfp8_phi${phi}_p${np}"; done; done
  fi
  if [ "$PART" != accuracy ]; then
    for phi in $PHIS; do echo "perf_oz1bfp8_phi${phi}"; for np in 1 2; do echo "perf_oz2bfp8_phi${phi}_p${np}"; done; done
  fi
}
CMDS=0
[ "$#" -le 1 ] || { echo 'ERROR: 引数は --list または --commands のみ' >&2; exit 2; }
case "${1:-}" in
  '') ;;
  --list) items; exit 0 ;;
  --commands) CMDS=1 ;;
  *) echo "ERROR: 引数 $1 は知らない（--list か --commands）" >&2; exit 2 ;;
esac
if [ "$CMDS" = 1 ]; then
  . ./scripts/measure_lib.sh --defs-only   # 一覧だけ：ビルドの記録の確認と排他は要らない（timing_env などの定義だけ）
else
  . ./scripts/measure_lib.sh   # provenance_line・bin_line・run_dram・wait_cool（ビルドの記録の確認とデバイスの排他もここで取る）
fi
. ./scripts/routes.sh        # routes_args（既定の構成の引数．表の p・法の本数を変えたときに書き下しが古いまま残らないように）
read -r I_BIN I_PA <<<"$(routes_args oz1bfp8)" && [ -n "${I_PA:-}" ] || { echo "ERROR: routes.tsv から I-BFP8 の既定の引数を読めない" >&2; exit 2; }
read -r II_BIN II_PA <<<"$(routes_args oz2bfp8)" && [ -n "${II_PA:-}" ] || { echo "ERROR: routes.tsv から II-BFP8 の既定の引数を読めない" >&2; exit 2; }
# 測ったログを置く measure_logs/phi_nowin/<rev7>/ の rev．--commands はログを書かないので rev が要らない（selftest がコンテナの中で
# 流すと，remeasure_all.sh のスナップショットは git worktree で .git が元のリポジトリを指し，その先がマウントされていないので
# git が読めない）．測るときに読めなければ止める
if ! REV7=$(git rev-parse HEAD 2>/dev/null); then
  [ "$CMDS" = 1 ] || { echo "ERROR: git rev-parse HEAD が失敗した（ログを置く measure_logs/phi_nowin/<rev7>/ を決められない）" >&2; exit 2; }
  REV7=unknown
fi
REV7=${REV7:0:7}
OUT=measure_logs/phi_nowin/$REV7; [ "$CMDS" = 1 ] || mkdir -p "$OUT" || exit 2
LABEL_W=40
fail=0
want() { [ -z "${ONLY:-}" ] || printf '%s\n' "$1" | grep -Eq -- "$ONLY"; }

# acc ログ名 ラベル バイナリ 環境変数… -- 引数…   到達精度の項目（冷まさない）
acc() {
  local log=$1 label=$2 bin=$3 env="" rc a; shift 3
  while [ "$1" != "--" ]; do env="$env $1"; shift; done; shift
  want "$log" || return 0
  local cmd="cd /home/user/tt && env$env ./src/build/$bin $*"
  if [ "$CMDS" = 1 ]; then printf '%s\tacc\t%s\n' "$log" "$cmd"; return 0; fi
  run_item --bins "$bin" "$OUT/$log.log" "$cmd"   # measure_lib.sh（流している途中で変わっていれば EXIT の代わりに INVALIDATED）
  rc=$RI_RC
  a=$(grep -m1 '^Accuracy' "$OUT/$log.log" | grep -oE 'effective ~[-0-9.]+ bit')
  if [ "$rc" -ne 0 ] || [ -z "$a" ]; then fail=1; a="FAILED (exit $rc, see $OUT/$log.log)"; fi
  printf "  %-${LABEL_W}s %s\n" "$label" "$a"
}
# perf ラベル ログ名 環境変数… -- コマンド…   性能の項目（run_dram．--commands なら run_dram が run.sh へ渡す文字列を出す）
perf() {
  want "$2" || return 0
  if [ "$CMDS" = 1 ]; then
    local log=$2 env=""; shift 2
    while [ "$1" != "--" ]; do env="$env $1"; shift; done; shift
    printf '%s\tperf\t%s\n' "$log" "cd /home/user/tt && OZ_NOVERIFY=1 $(timing_env) $env $*"
    return 0
  fi
  run_dram "$@"
}

[ "$PART" = performance ] || . ./scripts/measure/exponent_accuracy.sh
[ "$PART" = accuracy ] || . ./scripts/measure/exponent_performance.sh
[ "$CMDS" = 1 ] && exit 0
echo "(全出力は $OUT/*.log に残る)"
only_summary
exit $fail
