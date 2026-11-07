# lib/probes/limits.sh -- diagnose/probes.sh の群 (6) DRAM と L1 の上限
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(6) DRAM と L1 の上限"
# beta=3 の 14080^3 が DRAM に収まらないこと：本文から外した文（原稿の %コメント）と docs/kblock_plan.md だけが引く（PROBES_DOCS）
IN_DOCS=1
dram_item oom probe_rdx3_14080 "beta=3・p=34 の 14080^3" OZ_RADIX=3 -- ./src/build/ozaki1_bf16 14080 14080 14080 34 33 1 1
IN_DOCS=0
# 長い K と立方の上限は Ozaki-I-BF16 で見る．スキーム II の桁分割の経路は，桁積の和を FP32 で先に法へ畳むので
# K <= 16352 で止まり（ozaki2_main.cpp が確保の前に exit 2），DRAM の上限の前に K の上限が来る
oz_set oz1bf16
dram_item limit probe_limit_oz1bf16_176128 "oz1bf16 の M=N=3520・K=176128" -- ./src/build/$OZ_BIN 3520 3520 176128 $OZ_PA 1 1
dram_item limit probe_limit_oz1bf16_17600 "oz1bf16 の 17600^3" -- ./src/build/$OZ_BIN 17600 17600 17600 $OZ_PA 1 1
dram_item limit probe_limit_oz1bf16_21120 "oz1bf16 の 21120^3" -- ./src/build/$OZ_BIN 21120 21120 21120 $OZ_PA 1 1

# L1 の空き：OZ_L1_BUDGET_KB を二分探索して CB の確保が通る最大の予算を探す（経路は II-BFP4 の 7040^3．bw の候補が
# Kt=220 の約数なので予算の境目は bw の段で決まる）．予算そのものは bw の段の位置でしか決まらないので，失敗した回の
# 「CB が grow to X B which is beyond max L1 size of Y B」から CB の置き場の先頭（X から式の CB の合計を引いたもの）と
# L1 の終わり Y を読み，CB に使える L1（Y − 先頭）を出す（II-BFP4 は式の外の CB を作らないので，式の合計が CB の全部）．
# 式の合計は 2*bw*(pm+pn)*W_in + pm*pn*W_out（BFP4_b の 1 タイル 576 B，出力 FP32 4096 B．gemm cfg 行の per-core と bw）
# 終わり方（EXIT）：0 は境目を挟んだ（L1CAP 行に容量），3 は既定の予算 lo で通らない，2 は CB の確保以外で止まった，
# 4 は上端 hi でも通って境目が無い（失敗した回が無いので容量を出せない．L1CAP 行は budget_ng=none だけ），
# 5 は境目を挟んだが容量を出すのに要る行（通った回 lo と失敗した回 hi の gemm cfg 行の per-core・bw，失敗した回の
# 「grow to X B which is beyond max L1 size of Y B」の X・Y）が欠けた，または読めた値が整合しない（X > Y，
# CB の置き場の先頭 X − 式の合計が正，空き Y − 先頭が正，を満たさない）．5 のときは容量を出さない（L1CAP 行を書かず，
# 理由を ERROR 行に残す．架空の容量を残さない）．どの終わり方も流した後の確認（prov_exit）を通してから EXIT を書く
l1cap() {
  local log=probe_l1cap lo=1408 hi=1536 mid rc t=$SECONDS sum
  want "$log" || return 0
  if [ "$LIST" = 1 ]; then   # 予算は二分探索で決まるので，探索の範囲と 1 回のコマンドの形を出す
    list_item "$log" "l1cap(lo=$lo,hi=$hi)" "cd /home/user/tt && OZ_NOVERIFY=1 OZ_L1_BUDGET_KB=<予算> ./src/build/ozaki2_bfp4 7040 7040 7040 24 1 1"
    return 0
  fi
  prov_begin "$OUT/$log.log"
  prov_bin "$OUT/$log.log" ./src/build/ozaki2_bfp4
  try_budget() {   # try_budget 予算 → 0 通った，1 CB の確保に失敗，2 それ以外で止まった
    local b=$1 r out
    log_put "$OUT/$log.log" "=== probe: budget=$b"
    if out=$(./scripts/run.sh "cd /home/user/tt && OZ_NOVERIFY=1 OZ_L1_BUDGET_KB=$b ./src/build/ozaki2_bfp4 7040 7040 7040 24 1 1" 2>&1); then
      r=0; else r=$?; fi
    log_put "$OUT/$log.log" "$(printf '%s\n' "$out" | grep -vE '^\[[0-9a-f]+:[0-9]+\] ')"   # 例外のバックトレースの行は落とす
    log_put "$OUT/$log.log" "RC,$r"
    if [ "$r" -eq 0 ]; then return 0; fi
    printf '%s\n' "$out" | grep -q 'beyond max L1 size' && return 1
    return 2
  }
  try_budget $lo; rc=$?
  if [ $rc -ne 0 ]; then prov_exit "$OUT/$log.log" 3; fail=1; printf "  %-${LABEL_W}s FAILED（既定の予算 %s で通らない）\n" "L1 の空き" $lo; return; fi
  try_budget $hi; rc=$?
  if [ $rc -eq 0 ]; then
    # 上端でも通った：失敗した回が無いので CB の終端・L1 の終わりが読めず，下の要約は架空の値になる．境目を挟めて
    # いないので失敗として止める（探索の上端 hi を広げて測り直す）
    log_put "$OUT/$log.log" "L1CAP,route=oz2bfp4_7040,budget_ok=$hi,budget_ng=none"
    prov_exit "$OUT/$log.log" 4; fail=1
    printf "  %-${LABEL_W}s FAILED（上端の予算 %s でも通り，境目を挟めない）\n" "L1 の空き" $hi; return
  elif [ $rc -eq 1 ]; then
    while [ $((hi - lo)) -gt 1 ]; do
      mid=$(( (lo + hi) / 2 ))
      try_budget $mid; rc=$?
      if [ $rc -eq 0 ]; then lo=$mid; elif [ $rc -eq 1 ]; then hi=$mid; else break; fi
    done
  fi
  if [ $rc -eq 2 ]; then prov_exit "$OUT/$log.log" 2; fail=1; printf "  %-${LABEL_W}s FAILED（CB の確保以外で止まった）\n" "L1 の空き"; return; fi
  # 通った最大の予算の回（lo）と失敗した最小の予算の回（hi）の bw と CB の終端をログから拾って要約する．
  # 要る欄が 1 つでも欠けるか，値が整合しなければ awk は理由を出して 1 で終わる（容量の行は出さない）
  if sum=$(awk -v lo=$lo -v hi=$hi '
    /^=== probe: budget=/ { b = substr($0, 19) + 0 }
    /^gemm cfg:/ { if (match($0, /per-core=[0-9]+x[0-9]+/)) { split(substr($0, RSTART + 9, RLENGTH - 9), pc, "x"); pm[b] = pc[1]; pn[b] = pc[2] }
                   if (match($0, / bw=[0-9]+/)) bw[b] = substr($0, RSTART + 4, RLENGTH - 4) }
    /grow to [0-9]+ B which is beyond max L1 size of [0-9]+ B/ {
      if (match($0, /grow to [0-9]+/)) gt[b] = substr($0, RSTART + 8, RLENGTH - 8)
      if (match($0, /max L1 size of [0-9]+/)) mx[b] = substr($0, RSTART + 15, RLENGTH - 15) }
    END {
      miss = ""
      if (!(lo in pm) || !(lo in bw)) miss = miss " lo=" lo "の gemm cfg（per-core・bw）"
      if (!(hi in pm) || !(hi in bw)) miss = miss " hi=" hi "の gemm cfg（per-core・bw）"
      if (!(hi in gt) || !(hi in mx)) miss = miss " hi=" hi "の grow to・max L1 size"
      if (miss != "") { print "要る行が無い:" miss; exit 1 }
      cbok = 2 * bw[lo] * (pm[lo] + pn[lo]) * 576 + pm[lo] * pn[lo] * 4096
      cbng = 2 * bw[hi] * (pm[hi] + pn[hi]) * 576 + pm[hi] * pn[hi] * 4096
      base = gt[hi] - cbng
      if (!(gt[hi] > mx[hi]) || !(base > 0) || !(mx[hi] - base > 0)) {
        printf "値が整合しない: cb_end_ng=%s l1_size=%s cb_ng=%d cb_base=%d\n", gt[hi], mx[hi], cbng, base; exit 1 }
      printf "L1CAP,route=oz2bfp4_7040,budget_ok=%d,budget_ng=%d,bw_ok=%s,bw_ng=%s,cb_ok=%d,cb_ng=%d,cb_end_ng=%s,l1_size=%s,cb_base=%d,cb_free=%d,cb_free_kib=%.2f\n",
             lo, hi, bw[lo], bw[hi], cbok, cbng, gt[hi], mx[hi], base, mx[hi] - base, (mx[hi] - base) / 1024.0 }' \
    "$OUT/$log.log"); then
    log_put "$OUT/$log.log" "$sum"
    prov_exit "$OUT/$log.log" 0
    printf "  %-${LABEL_W}s %s（%d s）\n" "L1 の空き" "$(grep '^L1CAP' "$OUT/$log.log" | tail -1)" $((SECONDS - t))
  else
    log_put "$OUT/$log.log" "ERROR: L1 の空きを出せない（budget_ok=$lo・budget_ng=$hi．$sum）"
    prov_exit "$OUT/$log.log" 5; fail=1
    printf "  %-${LABEL_W}s FAILED（容量を出せない：%s）\n" "L1 の空き" "$sum"
  fi
}
l1cap
