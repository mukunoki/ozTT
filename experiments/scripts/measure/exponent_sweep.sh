#!/bin/bash
# measure/exponent_sweep.sh -- 指数部レンジ φ に対する到達精度（論文とスライドの fig:phi）の値を測り直す
# ---------------------------------------------------------------------------
#     STEPS="phi figs check" ./scripts/remeasure_all.sh         # 論文用の測り直し（測った後に元の作業木でコミットしたなら
#                                                               # REMEASURE_AT=<測った rev> も付ける）
#     ./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/measure/exponent_sweep.sh" >/tmp/phi.log
#                                                               # 手で試すとき（出力を追跡しているログ accuracy_logs/phi_sweep.log へ
#                                                               # 直接 tee しない：途中で止まると前の完全なログを失う．
#                                                               # remeasure_all.sh の phi の段は lib/common.sh の tee_replace で書く）
#   出どころの行はホストで作って OZ_PROVENANCE で渡す（渡さないと image=unknown になり，figures/update_fig_phi.py の
#   既定の検査が拒む）．
#
# fig:phi ：phi に対する到達精度（4096^3）を，図に載せる経路（routes.tsv の figs 列に phi を持つ 12 経路．
#           6 形式 × 2 方式）で測る．Ozaki-I は固定 p と自動 p，Ozaki-II は既定（b<=54）と自動 b（OZ2_FP64）．
#           FP16・TF32 の 4 経路は図の系列を増やさないよう含めない（I-FP16・I-TF32・II-FP16 は桁・基数・p・法が
#           BF16 と同じで行列エンジンに渡るスライスも同じ整数値なので，到達精度は BF16 の行と同じになる．
#           II-TF32 は法の集合が II-BF16 と別なので同じとは限らないが，図に無いので回さない．
#           acc_as 列で選ぶと acc_as が - の II-TF32 も入り，図が使わない 12 回を余計に回すので，figs 列で選ぶ）．
#           出力は 1 行 1 測定の CSV（PHI,...）で，fig:phi（paper/fig/fig_phi.gp）のデータ部を
#           ここから起こす（figures/update_fig_phi.py．系列も同じ figs 列で選ぶ）．スライドがこの図（fig/fig_phi.eps）を載せ，
#           論文（フル版）の fig:phi（fig_phiacc）も同じログから起こす（figures/update_fig_phiacc.py）．
# Ozaki-II の幅 b に対する到達精度の表（もと tab:phib．どの原稿も載せず検査も読まない）は phi_sweep.sh --phib に移した
# （読まない測定の失敗で fig:phi のログの EXIT が非零になり，図の段を止めないように）．
#
# 引数で fig:phi の phi を選べる：measure/exponent_sweep.sh [phi ...]（既定は 0 1 2 4 8 16．accuracy_logs/phi_sweep.log）．
# φ=0.5 の点は accuracy_logs/phi_sweep_05.log に分けて残す（lib/phi_logs.py の PHI_LOGS）ので，remeasure_all.sh の phi の段は
# measure/exponent_sweep.sh 0.5 も流す．引数は 0 以上の数だけを受け付け，ほかは測る前に止める．2 つのログの点と大きさ（4096）の正本は
# lib/constants.tsv の PHI_SWEEP・PHI_SWEEP_05・PHI_N（照合の lib/phi_logs.py も同じ行を読む）．
#
# 出力の 1 行目は出どころの PROVENANCE 行（scripts/provenance.sh．ホストから OZ_PROVENANCE で渡すのが
# 正式），最後の行は EXIT,rc=<失敗した実行の数>,post_check=0（0 なら全実行が正常に終わった．post_check=0 はコンテナの中で
# 書いた完了の記録で，流した後のホストの出どころの確認を経ていない印．scripts/README.md「ログの出どころ」）．
# figures/update_fig_phi.py は rev の分からない古いログや，最後まで走らなかった（EXIT の無い・非零の）ログを既定では図の出どころにしない．
# fig:phi の各実行は PHIRUN,<実装>,<phi>,<fixed|auto>,rc=<終了コード>,verify=<VERIFY の結果> の行，実行した
# バイナリの BINARY 行（provenance.sh の bin_line），その実行が出した RESULT・VERIFY・CONFIG の行と自動決定の
# 行（OZ_FP64:・OZ2_FP64:）をそのまま残す．失敗した回（終了コードが非零か VERIFY が pass でない）だけは，なぜその点を
# 測れなかったか（入口の拒否・確保の失敗・JIT のエラーなど）を辿れるよう，捕まえた全出力を PHIDIAG,begin,<実装>,<phi>,<種別> と
# PHIDIAG,end の区画に，各行の頭に PHIDIAG, を付けて残す（頭を付けるので，区画の中の行が RESULT・VERIFY・CONFIG の行として
# 読まれることは無い．EXIT が非零になるのでそのログは図の出どころにならない）．figures/update_fig_phi.py は各実行の完了と構成（探針・既定でないつまみ・
# gemm_split=1 でない実行・4096^3 以外の大きさを拒む），固定側・自動側の実際の設定（自動決定の行の有無，
# scheme II は CONFIG の two_pass），PHI 行の値がその実行の RESULT の到達精度と選んだ p・b に一致することを確かめる．
# 精度の合否の基準は OZ_MIN_BITS=1 に下げて測る．phi を上げると固定 p・固定 b の到達精度は 53 bit を割るので，
# 既定の基準（53 bit）のままでは終了コードが精度不足と異常終了を区別できない．基準を下げても到達精度の値は
# 変わらず，非有限の精度や異常終了は引き続き非零になる．
# コンテナの中で 1 回で 144 回の実行を流すので，各回の後に実行ファイルの sha256 を取り直し，その実行ファイルの最初の
# BINARY 行の値と違えば（流しているあいだに src/build が差し替わった），INVALIDATED 行を出して EXIT を書かずに止まる
# （figures/update_fig_phi.py は EXIT の無いログを拒む）．作業木の変更はコンテナの中では見えないので，それは呼び出し側
# （remeasure_all.sh の段の後の確認とスナップショット）が守る．
set -u
cd "$(dirname "$0")/../.."
. ./scripts/provenance.sh
. ./scripts/lib/common.sh    # oz_env_clean
provenance_check || exit 2    # src/build が出どころの行の rev のビルドか
provenance_line
# 継承した環境で比較条件が変わらないようにする：固定側に自動決定（OZ_FP64／OZ2_FP64）や幅の指定（OZ2_B）が
# 混ざらないように，精度測定なので検証（OZ_NOVERIFY）も省かせない（ホストで作った出どころの行 OZ_PROVENANCE は残す）．
# 自動側の設定は下で明示して立てる．
oz_env_clean OZ_PROVENANCE
export OZ_MIN_BITS=1
bits() { grep -oE 'effective ~[-0-9.]+ bit' | head -1 | grep -oE '[-0-9.]+'; }
nfail=0

PHI_N=$(const_get PHI_N) && PHI_DEFAULT=$(const_get PHI_SWEEP) || exit 2   # 大きさと既定の点（上の注記）
for _a in "$@"; do   # 引数は φ の値だけ（知らない引数・選択肢を黙って φ の点にしない）
  [[ $_a =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ERROR: 引数 $_a は φ の値（0 以上の数）でない（使い方: measure/exponent_sweep.sh [phi ...]）" >&2; exit 2; }
done
unset _a
PHI_LIST=${*:-$PHI_DEFAULT}
if [ "$PHI_LIST" = "$PHI_DEFAULT" ]; then
  echo "===== phi 依存（${PHI_N}^3，fig:phi の 12 経路：6 形式 × 2 方式）====="
else
  echo "===== phi 依存（${PHI_N}^3，fig:phi の 12 経路：6 形式 × 2 方式．phi=${PHI_LIST// /,} の追加）====="
fi
# 1 行 1 測定で残す（fig:phi のデータ部と docs/accuracy.md の表はここから起こす）：
#   PHI,<実装>,<phi>,<固定 p/b での到達ビット>,<自動決定での到達ビット>,<自動が選んだ p または b>
printf 'PHI,impl,phi,fixed_bit,auto_bit,auto_p_or_b\n'
# 経路・バイナリ・p（スキーム I）／法の本数（スキーム II）は routes.tsv から読む．
# 回すのは figs 列に phi を持つ経路だけ（上の注記．figures/update_fig_phi.py の系列と同じ選び方）．
. ./scripts/routes.sh      # routes_each
# phi_run 種別 環境変数… -- コマンド…   1 回の実行．PHIRUN 行・BINARY 行と RESULT・VERIFY・CONFIG・自動決定の行を出し，
#   出力を _out に置く．流した後の実行ファイルの sha256 がその実行ファイルの最初の値と違えば止まる（上の注記）
declare -A BIN_SHA0
phi_run() {
  local kind=$1 env=() rc v bl name h; shift
  while [ "$1" != "--" ]; do env+=("$1"); shift; done; shift
  if _out=$(env "${env[@]}" "$@" 2>&1); then rc=0; else rc=$?; fi
  v=$(printf '%s\n' "$_out" | grep -m1 -oE '^VERIFY,[^,]+,[a-z]+' | cut -d, -f3)
  [ "$rc" -eq 0 ] && [ "$v" = pass ] || nfail=$((nfail + 1))
  printf 'PHIRUN,%s,%s,%s,rc=%s,verify=%s\n' "$r_short" "$phi" "$kind" "$rc" "${v:-none}"
  bl=$(bin_line "$@")
  printf '%s\n' "$bl"
  name=$(printf '%s' "$bl" | cut -d, -f2); h=$(prov_field "$bl" sha256)
  [ -n "$name" ] || { echo "ERROR: 実行のコマンドに src/build のバイナリが無い: $*" >&2; exit 2; }
  [ -n "${BIN_SHA0[$name]:-}" ] || BIN_SHA0[$name]=$h
  # 測る前の provenance_check で各バイナリはビルドの記録どおりだったので，記録の値（stamp_sha256）とも比べる
  # （1 回目の実行のあいだの差し替えも捕まえる）
  if [ -z "$h" ] || [ "$h" = none ] || [ "$h" != "${BIN_SHA0[$name]}" ] || [ "$h" != "$(prov_field "$bl" stamp_sha256)" ]; then
    printf 'INVALIDATED,item=phi_tables,when=after_run,changed=bin_%s:%s->%s,reason=流しているあいだに実行ファイルが最初の BINARY 行のときと変わった\n' \
      "$name" "${BIN_SHA0[$name]:0:16}" "${h:0:16}"
    echo "ERROR: src/build/$name が流しているあいだに変わった．EXIT を書かずに止める" >&2
    exit 2
  fi
  printf '%s\n' "$_out" | grep -E '^(RESULT|VERIFY|CONFIG),|^OZ2?_FP64:'
  if [ "$rc" -ne 0 ] || [ "$v" != pass ]; then   # 失敗した回の全出力（上の注記）
    printf 'PHIDIAG,begin,%s,%s,%s\n' "$r_short" "$phi" "$kind"
    printf '%s\n' "$_out" | sed 's/^/PHIDIAG,/'
    printf 'PHIDIAG,end\n'
  fi
}
phi_row() {    # 1 経路 1 行（routes_each が r_short・r_binary・r_p・r_L・r_s・r_figs を置いて呼ぶ）
  local fx au pa o
  case ",$r_figs," in *,phi,*) ;; *) return ;; esac
  if [ "$r_scheme" = I ]; then
    phi_run fixed OZ_PHI=$phi -- ./src/build/$r_binary $PHI_N $PHI_N $PHI_N $r_p $r_L 1 1; fx=$(printf '%s\n' "$_out" | bits)
    phi_run auto OZ_PHI=$phi OZ_FP64=1 -- ./src/build/$r_binary $PHI_N $PHI_N $PHI_N $r_p $r_L 1 1; o=$_out
    pa=$(printf '%s' "$o" | grep -oE 'auto p=[0-9]+' | grep -oE '[0-9]+' | head -1)
  else
    phi_run fixed OZ_PHI=$phi -- ./src/build/$r_binary $PHI_N $PHI_N $PHI_N $r_s 1 1; fx=$(printf '%s\n' "$_out" | bits)
    phi_run auto OZ_PHI=$phi OZ2_FP64=1 OZ2_PASSES=2 -- ./src/build/$r_binary $PHI_N $PHI_N $PHI_N $r_s 1 1; o=$_out
    pa=$(printf '%s' "$o" | grep -oE '\-> s=[0-9]+ b=[0-9]+' | grep -oE 'b=[0-9]+' | grep -oE '[0-9]+' | head -1)
  fi
  au=$(printf '%s' "$o" | bits)
  [ -n "$fx" ] && [ -n "$au" ] || nfail=$((nfail + 1))
  printf 'PHI,%s,%s,%s,%s,%s\n' "$r_short" "$phi" "${fx:-NA}" "${au:-NA}" "${pa:-NA}"
}
for phi in $PHI_LIST; do
  routes_each phi_row || exit 2   # figs 列に phi を持つ経路：scheme I の 6 形式，続いて scheme II の 6 形式（表の順）
done

echo "EXIT,rc=$nfail,post_check=0"
exit $((nfail > 0))
