#!/bin/bash
# diagnose/reconstruction_components.sh -- 再構成段の 2x2（CRT の演算 × FP64 の組み立て）を冷却して測る
# ---------------------------------------------------------------------------
#     ./scripts/diagnose/reconstruction_components.sh [N]        # ホストから（tt-smi を使う）
#     SPLIT_ENV="OZ2_PASSES=2" SPLIT_TAG=p2 SPLIT_FMTS=bfp8 ./scripts/diagnose/reconstruction_components.sh   # 2 パスの再構成で同じ 2x2
#       SPLIT_ENV はセルの実行に足す環境変数（空白区切り），SPLIT_TAG はログ名の末尾の印（split_<形式>_<N>_c<>r<>_<印>.log），
#       SPLIT_FMTS は測る形式（既定 "bfp8 int8"）．2 パスでは各パスの時間を「recon pass」の行から表示にも出す．
#       SPLIT_ENV を与えるときは SPLIT_TAG も与える（印が無いと，論文が読む既定のログ split_<形式>_<N>_c<>r<>.log を
#       環境を足した変種で上書きするので，measure/performance.sh の COOL_ENV と同じく測り始める前に止める）
#     ONLY='^split_bfp8_14080_c1r0$' ./scripts/diagnose/reconstruction_components.sh   # ログ名（拡張子を除く）の正規表現でセルを選ぶ
#       （他の測定スクリプトと同じ．当たらないセルは測らず，1 つも当たらなければ最後に注意を出す）
#
# なぜ専用のスクリプトが要るか．
# この 2x2 は論文 7.5 節・8.4 節（14080）と docs/recon_cost_breakdown.md の結果 3（14080・3520）が引く値で（3520 の II-BFP8 は
# スライドと照合も引く．remeasure_all.sh の split の段），
# 差ではなく交互作用から読む．ところが 4 つのセルのうち「CRT を外して組み立てを残す」
# 構成だけがクロックに敏感で，連続実行では 214/220/243 ms と 13% ばらついた．
# 他の 3 セルは 0.1% で再現する．差を取る前に，全セルを同じ条件で採る必要がある．
#
# そこで measure/performance.sh と同じ冷却待ちを使う：1 セルごとに温度がしきい値へ下がるまで待ち，
# 到達したかどうかを表示とログ末尾の COOLCOND 行（その後に最後の行の EXIT）に残す（届かなければそのセルを失敗に数える）．
# 実行中のクロックの標本は split_*.clk に採り，要約を AICLK 行に残す（参考値．measure_lib.sh）．
# 計時は既定で steady（段ごとに定常状態まで温めてから測る．計時区間の AICLK は TELEM 行．measure_lib.sh の OZ_TIMING）．
# ログの 1 行目は出どころの PROVENANCE 行，2 行目は実行したバイナリの BINARY 行．
# 終了コードもログの最後の行に EXIT,rc=… として残し（流した後の出どころの確認を通ってから．measure_lib.sh の prov_exit），
# 非零なら FAILED として最後に非零で終わる．
#
# II-BFP8 は法の集合を段の時間のモデルで選ぶ（docs/awe_moduli.md §5）：
# 14080^3 で法は 14 本で，そのうち 6 本が 1 桁の法（剰余をそのまま運び，合成段を通らない）．1 桁の法の
# GEMM の出力を法へ畳む ModRound（recon_crt.cpp の RAWMASK）は再構成の計算カーネルの CRT 側にあり，
# `OZ2_CRT_PROBE` で CRT を外すと一緒に外れる．したがって「CRT 抜き」のセルの差には，この畳み込みの分も
# 含まれる（全部平方数の 11 本の表の法集合なら畳み込みは無い）．法の本数は各ログの CONFIG の s= にある
# （verify/check_paper_numbers.py が本文の s= と照合する）．II-INT8 は法の集合を表から取る（法 16 本，分割しない）．
set -u
cd "$(dirname "$0")/../.."
N="${1:-14080}"
OUT=measure_logs; mkdir -p "$OUT"
if [ -n "${SPLIT_ENV:-}" ] && [ -z "${SPLIT_TAG:-}" ]; then
  echo "ERROR: SPLIT_ENV（$SPLIT_ENV）を与えるときは SPLIT_TAG も与えること" \
       "（無いと論文が読む $OUT/split_*_${N}_c*r*.log を上書きする）" >&2
  exit 2
fi
# 形式の名前の誤りは排他を取る前に止める（測り始めた後に 2 つ目の形式で止まらないように．空白だけの SPLIT_FMTS で
# 1 セルも流さずに 0 で終わらないように）
read -r -a _f <<<"${SPLIT_FMTS:-bfp8 int8}"
[ ${#_f[@]} -gt 0 ] || { echo "ERROR: SPLIT_FMTS が空（bfp8・int8 を空白区切りで与える．既定 \"bfp8 int8\"）" >&2; exit 2; }
for f in "${_f[@]}"; do
  case " bfp8 int8 " in *" $f "*) ;; *) echo "ERROR: SPLIT_FMTS の $f は bfp8・int8 のどちらでもない" >&2; exit 2 ;; esac
done
unset _f
. ./scripts/measure_lib.sh   # TMAX・TWAIT・wait_cool・clk_begin/clk_end・provenance_line（デバイスの排他もここで取る）

EXTRA_ENV="${SPLIT_ENV:-}"
TAG="${SPLIT_TAG:+_$SPLIT_TAG}"
FMTS="${SPLIT_FMTS:-bfp8 int8}"
fail=0
cell() {   # cell <形式> <法の本数> <CRT抜き> <組立抜き>
  local fmt=$1 s=$2 cp=$3 rp=$4
  local pre cooled
  local log="$OUT/split_${fmt}_${N}_c${cp}r${rp}${TAG}.log"
  local name; name=$(basename "$log" .log)
  # ONLY（ログ名の正規表現．他の測定スクリプトと同じ）に当たらないセルは測らない
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$name" | grep -Eq -- "$ONLY"; then return 0; fi
  # 冷ましてから流し，標本・冷却の条件・流した後の確認と完了の記録を残す（measure_lib.sh の run_item --cool）
  local rc
  run_item --cool "$log" "cd /home/user/tt && OZ_NOVERIFY=1 $(timing_env) $EXTRA_ENV OZ2_CRT_PROBE=$cp OZ_RECON_PROBE=$rp \
    ./src/build/ozaki2_$fmt $N $N $N $s 1 3"
  rc=$RI_RC
  local v; v=$(result_value "$(cat "$log")" ozaki2_full recon_ms)   # 列は lib/constants.tsv の RESULT_ozaki2_full（lib/common.sh）
  if [ "$rc" -ne 0 ] || [ -z "$v" ]; then fail=1; v="FAILED(exit $rc)"
  elif [ "$cooled" != 1 ]; then fail=1; v="FAILED(冷却未達．値 $v)"; fi
  local pp; pp=$(grep -m1 -oE 'recon pass  : [0-9.]+ \+ [0-9.]+ ms' "$log" | sed 's/recon pass  : /パス別 /')
  printf '  %-6s CRT抜き=%s 組立抜き=%s  recon %9s ms %s (start %s degC%s, 再構成段の計時区間 %s MHz（TELEM）)\n' \
         "$fmt" "$cp" "$rp" "$v" "${pp:+（$pp）}" "$pre" "$(uncooled_mark)" "$(telem_clk "$log" recon)"
}

echo "===== 再構成段の 2x2（N=$N，1 セルずつ ${TMAX} degC 以下から測る．追加の環境：${EXTRA_ENV:-なし}） ====="
for f in $FMTS; do
  case $f in bfp8) s=24 ;; int8) s=16 ;; *) echo "ERROR: SPLIT_FMTS の $f は bfp8・int8 のどちらでもない" >&2; exit 2 ;; esac
  for cfg in "0 0" "1 0" "0 1" "1 1"; do
    set -- $f $s $cfg
    cell "$1" "$2" "$3" "$4"
  done
done
echo "(全出力は $OUT/split_*.log に残る)"
only_summary
exit $fail
