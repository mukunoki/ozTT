# lib/probes/exact.sh -- diagnose/probes.sh の群 (8) GEMM 段の無誤差の境界（diagnose/gemm_exact.sh）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
# GEMM 段の無誤差の境界（b_acc）：diagnose/gemm_exact.sh の出力に出どころの行と完了の記録を付けて残す（約 10 分）
gemm_exact() {
  local log=probe_gemm_exact rc t=$SECONDS
  want "$log" || return 0
  [ "$LIST" = 1 ] && { list_item "$log" "check(bins=gemm_exact_check)" "cd /home/user/tt && bash scripts/diagnose/gemm_exact.sh"; return 0; }
  section "(8) GEMM 段の無誤差の境界（diagnose/gemm_exact.sh）"
  run_item --bins gemm_exact_check "$OUT/$log.log" "cd /home/user/tt && bash scripts/diagnose/gemm_exact.sh"   # measure_lib.sh
  rc=$RI_RC
  if [ "$rc" -ne 0 ]; then fail=1; printf "  %-${LABEL_W}s FAILED（exit %s．%s）\n" "diagnose/gemm_exact.sh" "$rc" "$OUT/$log.log"
  else printf "  %-${LABEL_W}s %s（%d s）\n" "diagnose/gemm_exact.sh" "$(grep -E 'ALL PASS|SOME FAILED' "$OUT/$log.log" | tail -1)" $((SECONDS - t)); fi
}
gemm_exact
