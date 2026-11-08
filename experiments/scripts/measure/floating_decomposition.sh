#!/bin/bash
# measure/floating_decomposition.sh -- 無誤差分割（gemm_bf16ext）の到達精度のログ ext_split_<N>.log を作る
# ---------------------------------------------------------------------------
#     ./scripts/measure/floating_decomposition.sh [サイズ...]   # 既定 14080 3520．ホストから
#
# 論文「基準性能と無誤差分割の到達精度」の節と docs/exponent_format_constraint.md（追記 6）が引く値
# （BF16x3 の 12.5 bit，3xTF32 の 11.1 bit，有効仮数 10 bit の TF32 成分の 12.6 bit，単一 GEMM の
# 8.9／10.9 bit）．verify/check_paper_numbers.py の check_ext_prose が ext_split_14080.log を本文・スライドの
# 記述と照合する．
#
# 1 サイズ 1 本のログに，成分の形式 × 部分積の本数ごとの実行を順に足していく．各実行の前に
#     === fmt=<bf16|tf32> [sig=<有効仮数幅>] terms=<本数>
# の区切り行を置く（verify/check_paper_numbers.py の load_ext はこの行で形式と sig を知る．sig を書かない
# 区切りは既定の 11）．ログの 1 行目は出どころの PROVENANCE 行，2 行目は実行するバイナリの BINARY 行，
# 各実行の後ろに EXIT,rc=… を残す（検査側は区切りごとに EXIT,rc=0 を要る．最後の実行の後に EXIT が
# 無いログは途中で止まったものとして拒む）．
# 精度を測るので OZ_NOVERIFY は立てない（参照は host_eval の Dot2．14080^3 では 1 回数分かかる）．
# 到達精度は温度に依らないので冷却は待たない（スループットの欄は論文に使わない）．
#
#   EXT_TERMS  部分積の本数（既定 "1 3 6"．1 = Ah*Bh だけの単一 GEMM，3 = 成分 2 つ，6 = 成分 3 つ）
#   ONLY       区切り行の文字列に対する正規表現（例 ONLY='fmt=tf32'）．当たった実行だけを回し，
#              既存のログを消さずに後ろへ足す（当たる実行のある大きさのログにだけ，最初の実行の前に PROVENANCE 行と
#              BINARY 行も足すので，rev が変わっていれば検査側がログの中の混在として拒む）．ONLY が無ければログを作り直す
#
# 出どころの確認（measure_lib.sh）：大きさごとのログの PROVENANCE 行は prov_begin で書き（測り始めと出どころが
# 違えばログに触れずに止める），BINARY 行は最初の実行の前に prov_bin で書く．2 回目以降の実行の前には実行ファイルが
# その BINARY 行のときのままかを確かめ（prov_bin_check），各実行の後に prov_end で出どころと実行ファイルを確かめ直して
# から，その実行の EXIT を書く（prov_exit．実行の途中や実行の間で作業木・ビルド・実行ファイルなどが変わっていれば，
# EXIT の代わりにそのログの末尾に INVALIDATED 行を足して止める．ログには 9 回の実行が入るので，どの回が汚れても
# その大きさのログ全体を拒ませる）．
# 形式と有効仮数幅の変数（OZ_EXT_FMT・OZ_EXT_SIG）は区切り行のとおりに与える（コンテナの初期化ファイル（run.sh の
# bash -l が読む .profile など）に残った値は，run.sh がコマンドの前に外す）
set -u
cd "$(dirname "$0")/../.."
SIZES="${*:-14080 3520}"
TERMS="${EXT_TERMS:-1 3 6}"
# 本数は空白区切りの正の整数で，少なくとも 1 つ（空白だけの EXT_TERMS で 1 回も流さずに 0 で終わらないように）
read -r -a _t <<<"$TERMS"
[ ${#_t[@]} -gt 0 ] || { echo "ERROR: EXT_TERMS が空（部分積の本数を空白区切りで与える．既定 \"1 3 6\"）" >&2; exit 2; }
for _x in "${_t[@]}"; do
  [[ $_x =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: EXT_TERMS の $_x は正の整数でない" >&2; exit 2; }
done
unset _t _x
OUT=measure_logs; mkdir -p "$OUT"
. ./scripts/measure_lib.sh   # provenance_line・bin_line（ビルドの記録の確認とデバイスの排他もここで取る）
fail=0

# one ログ 形式 sig 本数
#   1 回の実行を区切り行つきでログへ足し，到達精度の行を表示する．その大きさの最初の実行の前に
#   PROVENANCE 行（prov_begin．ONLY のときは後ろへ足す）と BINARY 行を書く（started）
one() {
  local log=$1 fmt=$2 sig=$3 t=$4 head env rc acc
  head="=== fmt=$fmt${sig:+ sig=$sig} terms=$t"
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$head" | grep -Eq -- "$ONLY"; then return; fi
  env=""
  [ "$fmt" = tf32 ] && env="OZ_EXT_FMT=tf32"
  [ -n "$sig" ] && env="$env OZ_EXT_SIG=$sig"
  # その大きさの最初の実行は出どころの行（ONLY のときは後ろへ足す）と実行ファイルの行から，2 回目以降は実行ファイルが
  # 最初の BINARY 行のときのままかを確かめてから（違えば INVALIDATED を足して止める），区切り行を書いて流し，
  # 確かめてからこの実行の EXIT を書く（measure_lib.sh の run_item）
  local mode=--cont
  if [ "$started" = 0 ]; then
    mode=""; [ -n "${ONLY:-}" ] && mode=--append
    started=1
  fi
  run_item $mode --bins gemm_bf16ext --line "$head" "$log" "cd /home/user/tt && $env ./src/build/gemm_bf16ext $N $N $N 1 3 $t"
  rc=$RI_RC
  # 同じ区切り行が複数あれば（ONLY で足したとき）最後の実行の精度行を拾う
  acc=$(awk -v h="$head" '$0 == h {on = 1; a = ""; next} on && a == "" && /^Accuracy/ {a = $0} END {print a}' "$log" \
        | grep -oE 'effective ~[-0-9.]+ bit')
  if [ "$rc" -ne 0 ] || [ -z "$acc" ]; then
    fail=1; acc="FAILED (exit $rc, see $log)"
  fi
  printf '  %-30s %s\n' "$head" "$acc"
}

for N in $SIZES; do
  log="$OUT/ext_split_$N.log"
  echo "===== N=$N  無誤差分割の到達精度（gemm_bf16ext，HiFi4・FP32 累積）====="
  started=0
  for t in $TERMS; do one "$log" bf16 "" "$t"; done
  for t in $TERMS; do one "$log" tf32 "" "$t"; done
  for t in $TERMS; do one "$log" tf32 10 "$t"; done
done
echo "(全出力は $OUT/ext_split_*.log に残る)"
only_summary
exit $fail
