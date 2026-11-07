#!/bin/bash
# diagnose/ablations.sh -- 本文が挙げる「工夫を外したときの値」を測り直す
# ---------------------------------------------------------------------------
#     ./scripts/diagnose/ablations.sh 14080              # ホストから（tt-smi を使う）．大きさの既定は 14080
#     ./scripts/diagnose/ablations.sh --list 7040        # 測らずに，項目ごとにログ名・役割・コンテナへ渡すコマンドを 1 行ずつ出す
#     ./scripts/diagnose/ablations.sh --group k_block --list 14080 # K ブロック化の比較だけ
#                                               # （排他も出どころの確認もしない．remeasure_all.sh が段の項目を役割で選ぶ）
# measure/performance.sh と同じく 1 項目ずつ冷ましてから，段ごとに定常状態まで温めて測る（measure_lib.sh の run_dram．計時の方式
# OZ_TIMING の既定は steady）ので，並べた値を直接比べられる．
#
#   (1) K ブロック化の除去（8.3 節）：遅延再構成・融合・対ループを外した版との差．既定の構成は K を分割しないので
#       融合は働かず（abl_kb_on の CONFIG は fuse=0），差はスライス対ループと遅延再構成による
#   (2) スライス数の感度（8 節）：Ozaki-I-INT8 を既定の p=8 から 9 へ 1 段増やしたときの差
#   (3) 行バンドの確保：Ozaki-I-BFP2 を一括確保とバンド 6 本で比べる（7040．14080 の一括確保は DRAM に収まらないことの
#       確認）．既定の自動の本数は cool_oz1bfp2_<N> と同じ構成なので，ここでは測らない
# 既定の側（工夫を入れた側）の構成は経路の表 routes.tsv の既定（routes.sh の routes_args）から引き，実験で変える値
# （p=9 の p・L，バンドの本数）だけを書き下す．
#
# 項目の役割（--list の 2 列目．どの原稿が引くか．remeasure_all.sh の abl の段はこれで項目を選び，失敗の数え方を決める）：
#   paper  本文・スライドが引く（14080 の (1)・(2)）
#   docs   本文から外した文（原稿の %コメント）だけが引く（7040 の (3)）
#   oom    DRAM に収まらないことの確認（14080 の一括確保．落ちるのが期待どおり）
#   -      その大きさの値はどの原稿も引かない（通しの測定では流さない．手で流せば測る）
set -u
cd "$(dirname "$0")/../.."
LIST=0
GROUP=all
if [ "${1:-}" = --group ]; then
  GROUP=${2:-}; [ "$#" -ge 2 ] || { echo 'ERROR: --group に比較の名前が必要' >&2; exit 2; }; shift 2
fi
case $GROUP in all|k_block|slice_count|row_bands) ;; *) echo "ERROR: 未知の比較 $GROUP" >&2; exit 2 ;; esac
if [ "${1:-}" = --list ]; then LIST=1; shift; fi
N="${1:-14080}"
[ "$#" -le 1 ] || { echo 'ERROR: サイズは 1 つだけ指定する' >&2; exit 2; }
case $N in ''|*[!0-9]*|0) echo "ERROR: 大きさ $N は正の整数でない（使い方: diagnose/ablations.sh [--list] [サイズ]）" >&2; exit 2 ;; esac
OUT=measure_logs; [ "$LIST" = 1 ] || mkdir -p "$OUT"
# 冷却待ちと 1 項目の実行（run_dram）は measure_lib.sh にある（冷却未達・終了コードの扱いと ONLY による絞り込みもそちらの注記）
if [ "$LIST" = 1 ]; then . ./scripts/measure_lib.sh --defs-only; else . ./scripts/measure_lib.sh; fi
. ./scripts/routes.sh        # routes_args（既定の側の構成）
read -r BF16_BIN BF16_PA <<<"$(routes_args oz1bf16)" && [ -n "${BF16_PA:-}" ] || { echo "ERROR: routes.tsv から I-BF16 の既定の引数を読めない" >&2; exit 2; }
read -r INT8_BIN INT8_PA <<<"$(routes_args oz1int8)" && [ -n "${INT8_PA:-}" ] || { echo "ERROR: routes.tsv から I-INT8 の既定の引数を読めない" >&2; exit 2; }
read -r BFP2_BIN BFP2_PA <<<"$(routes_args oz1bfp2)" && [ -n "${BFP2_PA:-}" ] || { echo "ERROR: routes.tsv から I-BFP2 の既定の引数を読めない" >&2; exit 2; }
LABEL_W=40
fail=0
# run ラベル ログ名 役割 環境変数… -- コマンド…   1 項目（--list なら測らずに 1 行を出す．run_dram が run.sh へ渡すのと同じ文字列）
run() {
  local label=$1 log=$2 role=$3 env="" a; shift 3
  if [ "$LIST" = 1 ]; then
    for a in "$@"; do [ "$a" = -- ] && break; env="$env $a"; done
    while [ "$1" != "--" ]; do shift; done; shift
    printf '%s\t%s\t%s\n' "$log" "$role" "cd /home/user/tt && OZ_NOVERIFY=1 $(timing_env) $env $*"
    return 0
  fi
  run_dram "$label" "$log" "$@"
}
# role 14080 の役割 7040 の役割   この大きさ N での役割（ほかの大きさは -）
role() { case $N in 14080) echo "$1" ;; 7040) echo "$2" ;; *) echo - ;; esac; }

for group in k_block slice_count row_bands; do
  [ "$GROUP" = all ] || [ "$GROUP" = "$group" ] || continue
  . "./scripts/diagnose/${group}.sh" || exit 2
done
[ "$LIST" = 1 ] && exit 0
echo "(全出力は $OUT/abl_*.log に残る)"
only_summary
exit $fail
