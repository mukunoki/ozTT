#!/bin/bash
# diagnose/radix_odd.sh -- beta=2d+1 が K に依存して破れることを確かめる
# ---------------------------------------------------------------------------
#     ./scripts/run.sh "bash /home/user/tt/scripts/diagnose/radix_odd.sh"
# 桁の上限が ceil(beta/2) であること（docs/fidelity_radix_sweep.md の追記）の根拠．
# OZ_RADIX=65 は桁 33 を要求するので，OZAKI_MAX_RADIX が 33 以上ある BF16 で見る．
# 桁を 32 に抑えた場合との差を出すため，桁の上限は OZ_DMAX で上書きする．
# 出力（probe_checks.py が読む）：ODDR,<radix>,<dmax>,<K>,<bit>（bit は effective の表示．実行が期待どおりに
# 終わらなければ NA）．破れる構成は検証に落ちて終了コード 1 で終わるのが期待どおりなので 0 と 1 を受け付け，
# それ以外（起動できない・入口の検査で止まった・クラッシュ・検証の状態と終了コードが食い違う）は値を NA にして，
# このスクリプトを終了コード 1 で終える（diagnose/probes.sh の probe_radix_odd はその回を失敗に数える）．
set -u
cd "$(dirname "$0")/../.."
. ./scripts/lib/common.sh  # oz_env_clean・run_capture・verify_state
# 固定の p・既定の入力で精度を測るので，入力を変える・p を選び直す・検証を省く設定を引き継がない（OZ_NOVERIFY が
# 残っていると精度の欄が番兵の値になる）
oz_env_clean
eff_bits_display() { printf '%s\n' "$1" | grep -oE 'effective ~[-0-9.]+ bit' | head -1 | grep -oE '[-0-9.]+'; }
nerr=0
printf 'ODDR,radix,dmax,K,bit\n'
for K in 512 1024 1792 4096 16384; do
  for cfg in "64 32" "65 32" "65 33"; do
    set -- $cfg
    run_capture o rc -- env OZ_RADIX=$1 OZ_DMAX=$2 ./src/build/ozaki1_bf16 512 512 $K 10 9 1 1
    v=$(eff_bits_display "$o"); vs=$(verify_state "$o")
    if ! { [ "$rc" = 0 ] && [ "$vs" = pass ]; } && ! { [ "$rc" = 1 ] && [ "$vs" = fail ]; }; then
      echo "ERROR: OZ_RADIX=$1 OZ_DMAX=$2 K=$K の実行が期待どおりに終わらない（終了コード $rc，VERIFY $vs）" >&2
      v=""; nerr=$((nerr + 1))
    fi
    printf 'ODDR,%s,%s,%s,%s\n' "$1" "$2" "$K" "${v:-NA}"
  done
done
[ "$nerr" = 0 ] || { echo "ERROR: 期待どおりに終わらない実行が $nerr 回あった（その値は NA）" >&2; exit 1; }
exit 0
