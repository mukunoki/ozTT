#!/bin/bash
# measure/performance.sh -- 1 項目ずつチップを冷ましてから測る（サーマルによる系統誤差を除く）
# ---------------------------------------------------------------------------
# study/measure_paper.sh は 18 項目を冷まさずに続けて回すので，後半の項目ほどチップが温まった状態から始まり，
# 値が項目の順序に依存する．論文の表の値はこちらで測る．各項目の前に温度がしきい値へ下がるまで待ち，実行前後の温度とクロックを
# ログの末尾の COOLCOND 行（その後に最後の行の EXIT）に残す．計時は既定で steady（各段をクロックが負荷に追いつくまで流してから測る．
# 段ごとの計時区間の AICLK などは TELEM 行．measure_lib.sh の OZ_TIMING）で，冷ますのはその温めを始める
# 条件を揃えるためである．尾崎の経路は OZ_TIME_WHOLE=1 で，段ごとの計時（内訳．DRAM->DRAM 行）の後に DGEMM 1 回の
# 呼び出しの通しを温めてから測る（WHOLE-PIPELINE 行．論文の全体の時間．2026-11-03 から）．ホスト側で tt-smi を呼ぶのでコンテナの外から実行する：
#     ./scripts/measure/performance.sh 14080
#     ./scripts/measure/performance.sh --list 14080   # 測らずに，項目ごとにログ名・表示名・コンテナへ渡すコマンドを 1 行ずつ出す
#                                              （ONLY・COOL_ENV・COOL_TAG も反映する．verify/check_routes.py が経路の表と突き合わせる）
set -u
cd "$(dirname "$0")/../.."
LIST=0
if [ "${1:-}" = --list ]; then LIST=1; shift; fi
[ "$#" -le 1 ] || { echo "ERROR: サイズは1つずつ指定する（measure/performance.sh [--list] [サイズ]）" >&2; exit 2; }
N="${1:-14080}"
case $N in ''|*[!0-9]*|0) echo "ERROR: 大きさ $N は正の整数でない（使い方: measure/performance.sh [--list] [サイズ]）" >&2; exit 2 ;; esac
OUT=measure_logs; [ "$LIST" = 1 ] || mkdir -p "$OUT"
# COOL_ENV を与えると各項目の環境へ足す（例：COOL_ENV=OZ_NO_BLOCKMM=1 で matmul_tiles の GEMM）．
# COOL_TAG はログ名の末尾に付ける印（例：_noblockmm → cool_oz1bf16_14080_noblockmm.log）．
# 付けたログは cool_*_<N>.log の形から外れるので，論文の表を起こすスクリプトは読まない．
# COOL_ENV だけを与えると論文の表が読む cool_*_<N>.log を環境を足した変種で上書きするので，
# 印の無い COOL_ENV は測り始める前に止める．与えた COOL_ENV はログにも COOLENV 行として残し，
# 検査側（oz_logs.py）はその行のあるログを表・図の出どころにしない．
if [ -n "${COOL_ENV:-}" ] && [ -z "${COOL_TAG:-}" ]; then
  echo "ERROR: COOL_ENV（$COOL_ENV）を与えるときは COOL_TAG も与えること" \
       "（無いと論文の表が読む $OUT/cool_*_$N.log を上書きする）" >&2
  exit 2
fi
# TMAX・TWAIT・temp・clk・wait_cool・coolcond・clk_begin/clk_end・provenance_line・bin_line．
# 読み込んだ時点でビルドの記録を確かめ（provenance_check），デバイスの排他を取る（一覧だけなら定義だけを読む）
if [ "$LIST" = 1 ]; then . ./scripts/measure_lib.sh --defs-only; else . ./scripts/measure_lib.sh; fi

# 名前 ログ名 引数...
#   終了コードはログへ EXIT,rc=… として残し，非零なら指標の行が出ていても FAILED と表示する
#   （全項目のあと非零で終わる）．冷却がしきい値へ届かないまま測った項目も失敗に数える．
#   --list のときは測らずに，ログ名・表示名とコンテナへ渡すコマンド（測るときと同じ文字列）を出すだけ
. ./scripts/routes.sh        # routes_each_in・routes_row_cmd・OZ_MEAS_ORDER
fail=0
run() {
  label=$1; log=$2${COOL_TAG:-}; shift 2
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$log" | grep -Eq -- "$ONLY"; then return; fi
  # 計時の方式（measure_lib.sh の timing_env．既定 steady：段ごとに定常状態まで温めてから測る）
  cmd="cd /home/user/tt && OZ_NOVERIFY=1 $(timing_env) ${COOL_ENV:-} $*"
  if [ "$LIST" = 1 ]; then printf '%s\t%s\t%s\n' "$log" "$label" "$cmd"; return; fi
  # 冷ましてから（しきい値へ届いたかを cooled に残す．届かないまま測った値は失敗に数え，検査側も弾く），出どころの行・
  # 実行ファイルの行・（COOL_ENV があれば）COOLENV 行を書いて流し，実行中のクロックの標本（$OUT/$log.clk と AICLK 行）と，
  # 冷却の条件に前後の温度・クロックと標本の要約を足した COOLCOND 行，流した後の確認と完了の記録を残す
  # （measure_lib.sh の run_item --cool-detail．aiclk_* は起動・JIT・準備を含む区間の参考値で，正規化には使わない）
  if [ -n "${COOL_ENV:-}" ]; then
    run_item --cool-detail --line "COOLENV,$COOL_ENV" "$OUT/$log.log" "$cmd"
  else
    run_item --cool-detail "$OUT/$log.log" "$cmd"
  fi
  rc=$RI_RC; post=$RI_POST
  # 尾崎の各実装は全体の時間（WHOLE-PIPELINE 行．通し）の "TFLOP/s fp64-equiv"，素の GEMM は "Performance : ... TFLOP/s" を出す
  case " $* " in
    *" OZ_TIME_WHOLE=1 "*) v=$(grep -E '^WHOLE-PIPELINE : ' "$OUT/$log.log" | grep -oE '[0-9.]+ TFLOP/s fp64-equiv' | head -1) ;;
    *) v=$(grep -oE '[0-9.]+ TFLOP/s fp64-equiv|Performance : [0-9.]+ TFLOP/s' "$OUT/$log.log" | head -1) ;;
  esac
  if [ "$rc" -ne 0 ] || [ -z "$v" ]; then
    fail=1; v="FAILED (exit $rc, see $OUT/$log.log)"
  elif [ "$cooled" != 1 ]; then
    fail=1; v="FAILED（冷却未達．値は $v）"
  fi
  # GEMM 段の計時区間の AICLK（TELEM 行）と，実行中の標本（起動・JIT・準備を含み計時区間と同期しない参考値）
  printf '  %-14s %-28s (start %s degC%s, end %s degC / GEMM 段の計時区間 %s MHz%s / 実行中の標本 %s-%s MHz 平均 %s．参考)\n' \
         "$label" "$v" "$pre" "$(uncooled_mark)" "$post" "$(telem_clk "$OUT/$log.log")" "$(telem_lim "$OUT/$log.log")" \
         "$cmin" "$cmax" "$cmean"
}

[ "$LIST" = 1 ] || echo "===== N=$N  1 項目ずつ ${TMAX} degC 以下から測る ====="
# 尾崎の 16 経路は routes.tsv から引く（表示名・ログ名の幹と，表の既定の構成 p・L か法の本数）．回す順は routes.sh の
# OZ_MEAS_ORDER（論文の値を測ってきた順．表の並びと違う）．ログ名は cool_<表の log>_<N>，ウォームアップ 1・計時 3．
# 全体の時間は全工程の通し（OZ_TIME_WHOLE=1）
cool_route() { run "$r_label" "cool_${r_log}_$N" OZ_TIME_WHOLE=1 $(routes_row_cmd "$N" 1 3); }
routes_each_in "$OZ_MEAS_ORDER" cool_route || exit 2
# 素の GEMM（表に無い．引数は lib/paper_cfg.py の BASELINE_CFG と同じ構成で，verify/check_routes.py が突き合わせる）
run "BF16 single"   "cool_bf16single_$N" ./src/build/gemm_bf16 $N $N $N 1 1 3 0 0 0 0
run "BF16x2/PP"     "cool_bf16x2_$N"     ./src/build/gemm_bf16 $N $N $N 1 1 3 0 0 1 0
[ "$LIST" = 1 ] && exit 0
echo "(全出力は $OUT/cool_*.log に残る)"
only_summary
exit $fail
