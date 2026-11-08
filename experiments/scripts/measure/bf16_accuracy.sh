#!/bin/bash
# measure/bf16_accuracy.sh -- 単体の BF16 GEMM（性能の表の行と同じ構成）の到達精度のログを作る
# ---------------------------------------------------------------------------
#     ./scripts/measure/bf16_accuracy.sh [サイズ...]   # 既定 14080 7040 3520．ホストから
#     ONLY='^bf16_verify_14080$' ./scripts/measure/bf16_accuracy.sh   # ログ名（拡張子を除く）の正規表現で大きさを選ぶ
#     ./scripts/measure/bf16_accuracy.sh --list        # 既定で測る項目のログ名を 1 行ずつ出すだけ（測らない．
#                                                    # remeasure_all.sh が組の項目が揃ったかを見るのに使う）
#
# 論文の表「到達精度と性能の概要」（tab:flops）の「単体 BF16 GEMM」の行と PDP 版 7.3 節が引く実効ビット数
# （14080^3 で約 6.5 bit）と，スライドの問題サイズ別の表の行名（3 サイズで 6.5--7.4 bit）の出どころ．
# 性能の行（measure/performance.sh の cool_bf16x2_<N>．gemm_bf16 の HiFi4・fp32_dest_acc=1）は検証を止めて測るので
# （verify=0），同じ引数で検証だけを有効にして（第 7 引数を 1）測り直す．
# fp32_dest_acc=1 で FP32 になるのは DST（1 つの読み込みブロックの累積）の中だけで，K 方向の読み込みブロック間の
# 部分和（c_24）と出力（c_16）は BF16 である（src/baseline/gemm_bf16.cpp の冒頭）．そのため実効ビット数は無誤差分割の
# 1 項（gemm_bf16ext．部分和・出力とも FP32．measure/floating_decomposition.sh の 8.9 bit）より低く，K が大きいほど低い．
#
# ログは measure_logs/bf16_verify/<rev7>/bf16_verify_<N>.log（1 サイズ 1 本）．1 行目は出どころの PROVENANCE 行，
# 2 行目は実行するバイナリの BINARY 行，最後に EXIT,rc=…．性能の表のログ（cool_*）とは測った rev が違うので，
# verify/check_paper_numbers.py はこのログを rev 別の組として読み（load_bf16_verify），このログ自身の出どころ
# （dirty=0・ALLOW_DIRTY の印なし・build_rev=rev・実行したバイナリがビルドの記録どおり）と，性能の行のログと
# 構成（bw・nb・パネル・コアの格子など）が同じこと，同じマシンで測ったことを確かめる．
# 精度を測るので OZ_NOVERIFY は立てない（参照は host_eval の Dot2．14080^3 では約 2 分かかる）．
# 到達精度は温度に依らないので冷却は待たない（スループットの欄は論文に使わない）．
# rev が変わったら全サイズを測り直し，古い rev のディレクトリ measure_logs/bf16_verify/<古い rev7>/ を
# measure_logs/bf16_verify の外へ退避する（同じ大きさのログが 2 つの rev にあると検査はどちらも使わない．
# remeasure_all.sh の bf16_verify の段は，段が成功し，この rev の組の全項目（--list）が揃ったときだけ退避する）．
# ONLY で一部の大きさだけを測るのは，同じ rev の組の一部の測り直しにだけ使う．
# <rev7> は HEAD の先頭ちょうど 7 桁（git rev-parse HEAD | cut -c1-7．--short=7 は他のオブジェクトと重なると 8 桁
# 以上になり，検査がちょうど 7 桁を要るのと食い違う）
set -u
cd "$(dirname "$0")/../.."
DEF_SIZES="14080 7040 3520"
if [ "${1:-}" = --list ]; then
  for N in $DEF_SIZES; do echo "bf16_verify_$N"; done
  exit 0
fi
SIZES="${*:-$DEF_SIZES}"
. ./scripts/measure_lib.sh   # provenance_line・bin_line（ビルドの記録の確認とデバイスの排他もここで取る）
REV7=$(git rev-parse HEAD) || exit 2
REV7=${REV7:0:7}
OUT=measure_logs/bf16_verify/$REV7; mkdir -p "$OUT" || exit 2
fail=0
for N in $SIZES; do
  log="$OUT/bf16_verify_$N.log"
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "bf16_verify_$N" | grep -Eq -- "$ONLY"; then continue; fi
  # 引数は measure/performance.sh の cool_bf16x2_<N> と同じ（M N K B warmup iters verify bw fp32_dest_acc dst_full_sync）で，
  # verify だけを 1 にする（measure_lib.sh の run_item．流している途中（参照の計算を含め約 2 分）で出どころか実行ファイルが
  # 変わっていれば，EXIT の代わりに INVALIDATED を足して止める）
  run_item "$log" "cd /home/user/tt && ./src/build/gemm_bf16 $N $N $N 1 1 3 1 0 1 0"
  rc=$RI_RC
  acc=$(grep -m1 '^Accuracy' "$log" | grep -oE 'effective ~[-0-9.]+ bit')
  if [ "$rc" -ne 0 ] || [ -z "$acc" ]; then
    fail=1; acc="FAILED (exit $rc, see $log)"
  fi
  printf '  N=%-6s %s\n' "$N" "$acc"
done
echo "(全出力は $OUT/bf16_verify_*.log に残る)"
only_summary
exit $fail
