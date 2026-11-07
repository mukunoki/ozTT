#!/bin/bash
# phi_sweep.sh -- 指数部レンジを振って到達精度とスライス数の依存を測る（調査用．論文の表には直接入らない）
# ---------------------------------------------------------------------------
#     ./scripts/run.sh "bash /home/user/tt/scripts/study/phi_sweep.sh [N] [phi ...]"          # 既定 N=4096，phi は下の PHIS
#     ./scripts/run.sh "bash /home/user/tt/scripts/study/phi_sweep.sh --phib"                  # Ozaki-II の幅 b の表（1024^3）
#
# 入力は尾崎スキームの評価で標準的に使われる
#     a_ij = (rand - 0.5) * exp(phi * randn)      （rand は [0,1) の一様乱数，randn は標準正規乱数）
# である（src/common/oz_input.hpp の gen_operands の OZ_PHI）．phi が指数部レンジを決める：因子 exp(phi * randn) の
# 自然対数は標準偏差 phi の正規分布に従い，±1 標準偏差の幅はおおよそ 2*phi/ln2 ≈ 2.9*phi bit になる（要素の対数には
# log|rand - 0.5| も加わるので，要素の対数そのものは正規分布ではなく，この幅も標本の動的域そのものではない目安）．
# phi=0 は [-0.5, 0.5) の一様乱数（OZ_PHI を立てない既定の入力 [-1, 1) の一様乱数とは大きさだけが違う）．
#
# 見るもの：
#   - 固定スライス数（論文の測定に使う p=10/8/12，Ozaki-II は s=16/14）での到達精度
#   - Ozaki-I を誤差の期待値のモデル（ozaki1/ozaki1_autop.hpp の oz_fp64_p．保証ではない）で自動決定させたとき
#     （OZ_FP64=1）の p と到達精度
#   - --phib：Ozaki-II の幅 b に対する到達精度（1024^3，法は 24 本要求，2 パス再構成．分割だけの厳密模擬
#     diagnose/probes.sh の probe_splitsim と並べる実機側の列．もとは measure/exponent_sweep.sh の tab:phib．どの原稿も載せていない）
# Ozaki-II の自動決定（OZ2_FP64）込みの表は measure/exponent_sweep.sh にある．
# スライス分解はデータ依存なので，行内の最大値から大きく下回る要素ほど多くのスライスを
# 要する．固定 p のまま phi を上げれば精度は落ち，p を増やせば戻る．
# 精度の合否の基準は OZ_MIN_BITS=1 に下げて流す（phi を上げると固定 p の到達精度は 53 bit を割るので，既定の基準では
# 精度不足と異常終了を終了コードで区別できない）．終了コードが 0 でない実行（起動できない・クラッシュ・入口の検査で
# 止まった）の欄は FAIL(rc=<n>) と出し，最後に非零で終わる．
set -u
cd "$(dirname "$0")/../.."
. ./scripts/lib/common.sh  # oz_env_clean・run_capture
# 継承した環境で比較条件が変わらないようにする：固定側に自動決定（OZ_FP64／OZ2_FP64）が混ざらないように，
# 精度測定なので検証（OZ_NOVERIFY）も省かせない．自動側は下で明示して立てる．
oz_env_clean
export OZ_MIN_BITS=1
nfail=0

bits() { grep -oE 'effective ~[-0-9.]+ bit' | head -1 | grep -oE '[-0-9.]+'; }
# acc 出力の変数 -- コマンド…   実行して到達精度の表示（effective）を置く．終了コードが 0 でなければ FAIL(rc=<n>) を置いて数える
acc() {
  local __av=$1 __o __rc __v; shift; [ "$1" = -- ] && shift
  run_capture __o __rc -- "$@"
  __v=$(printf '%s\n' "$__o" | bits)
  if [ "$__rc" != 0 ] || [ -z "$__v" ]; then __v="FAIL(rc=$__rc)"; nfail=$((nfail + 1)); fi
  printf -v "$__av" '%s' "$__v"
  _last_out=$__o
}

if [ "${1:-}" = --phib ]; then
  # 法の本数は既定の構成（routes.tsv の II-INT8 の 16）ではなく上限の 24 を要求する（OZ2_B で与えた b の窓を覆う本数を
  # 実装が選ぶので，上限を渡す．verify/check_routes.py は既定の構成で回す行だけを突き合わせる）
  S_REQ=24
  echo "===== Ozaki-II の b 依存（1024^3，法は $S_REQ 本要求，2 パス再構成）====="
  printf '%-5s' 'b'; for phi in 0 2 4; do printf ' phi=%-6s' "$phi"; done; echo
  for b in 48 52 56 60 64 68; do
    printf '%-5s' "$b"
    for phi in 0 2 4; do
      acc v -- env OZ_PHI=$phi OZ2_B=$b OZ2_FP64=1 OZ2_PASSES=2 ./src/build/ozaki2_int8 1024 1024 1024 "$S_REQ" 1 1
      printf ' %-11s' "$v"
    done
    echo
  done
  [ "$nfail" = 0 ] || { echo "ERROR: 終了コードが 0 でない実行が $nfail 回あった" >&2; exit 1; }
  exit 0
fi

N=${1:-4096}
case $N in ''|*[!0-9]*|0) echo "ERROR: 大きさ N=$N は正の整数でない（使い方: phi_sweep.sh [N] [phi ...]）" >&2; exit 2 ;; esac
[ $# -gt 0 ] && shift
PHIS=${*:-"0 0.5 1 2 3 4 6 8 16"}

printf '入力 a_ij=(rand-0.5)*exp(phi*randn)   N=%s（立方）\n\n' "$N"
printf '%-5s %-8s | %-14s %-14s %-14s | %-7s %-7s\n' \
       phi 動的域 I-BF16 I-INT8 I-FP8 "II-INT8" II-FP8
printf '%-5s %-8s | %-14s %-14s %-14s | %-7s %-7s\n' \
       '' '' '固定p=10 / 自動' '固定p=8 / 自動' '固定p=12 / 自動' 's=16' 's=14'
echo "-------------------------------------------------------------------------------------"
for phi in $PHIS; do
  export OZ_PHI=$phi
  # scheme I：固定スライス数（論文の測定値）と，誤差の期待値のモデルで決め直させた場合（OZ_FP64=1）
  #           自動決定は形式によらず共通の ozaki1/ozaki1_autop.hpp にあるので 3 形式とも使える
  cell() {  # $1=バイナリ $2=固定p
    local fx au pa
    acc fx -- ./src/build/$1 $N $N $N $2 $(($2-1)) 1 1
    acc au -- env OZ_FP64=1 ./src/build/$1 $N $N $N $2 $(($2-1)) 1 1
    pa=$(printf '%s' "$_last_out" | grep -oE 'auto p=[0-9]+' | grep -oE '[0-9]+')
    printf '%s / %s(p=%s)' "$fx" "$au" "$pa"
  }
  # 列ごとに経路を書き下す（表の見出しと対）．p・法の本数の正本は routes.tsv で，
  # 食い違いは verify/check_routes.py（selftest.sh が呼ぶ）が拾う．
  # cell はコマンド置換の中で流れるので，失敗の数は列の FAIL(rc=…) から数える
  c16=$(cell ozaki1_bf16 10)
  ci8=$(cell ozaki1_int8  8)
  cf8=$(cell ozaki1_fp8  12)
  for c in "$c16" "$ci8" "$cf8"; do case $c in *FAIL*) nfail=$((nfail + 1)) ;; esac; done
  acc o2i -- ./src/build/ozaki2_int8 $N $N $N 16 1 1
  acc o2f -- ./src/build/ozaki2_fp8  $N $N $N 14 1 1
  dyn=$(awk -v p="$phi" 'BEGIN { printf "~%.0f bit", 2 * p / 0.6931 }')
  printf '%-5s %-8s | %-14s %-14s %-14s | %-7s %-7s\n' \
         "$phi" "$dyn" "$c16" "$ci8" "$cf8" "$o2i" "$o2f"
done
unset OZ_PHI
echo
echo "scheme I の自動決定（OZ_FP64=1）は 3 形式とも共通の ozaki1/ozaki1_autop.hpp にあり，"
echo "誤差の期待値のモデル（保証ではない）からスライス数を決め直す．scheme II の自動決定は OZ2_FP64 で，"
echo "||E||_F/||C||_F から幅 b を決める（||C||_F は Hutchinson のスケッチで測る）．"
echo "上の II の列は固定 s（既定 b<=54）なので，自動決定込みの値は measure/exponent_sweep.sh を見ること．"
echo "（参照解 Dot2 は，phi=0--16・K=1024/14080 で __float128 の参照和に対し相対誤差 2^-85 以下．53 bit の判定には十分である）"
[ "$nfail" = 0 ] || { echo "ERROR: 終了コードが 0 でない実行がある（FAIL(rc=…) の欄）" >&2; exit 1; }
exit 0
