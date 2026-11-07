# lib/probes/host.sh -- diagnose/probes.sh の群 (2) ホストだけの検査（コンテナの中で．デバイスは使わない）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(2) ホストだけの検査（コンテナの中で．デバイスは使わない）"
acc_item probe_dot2 "参照解 Dot2 の誤差（12 条件）" dot2check 'sub dot2check -- ./src/build/dot2check'
acc_item probe_awe "法の選び方のモデルの見込み" awe_plan 'sub awe_plan fp8 tf32 -- ./src/build/awe_plan fp8 tf32'
# 誤差は K=1024 の K^3 の出力の全要素で測る（study/phi_sweep.sh --phib の b の表が実機で測る行列と同じ量．一部の要素だけでは
# φ が大きいと ||C||_F が少数の要素に支配されて数 bit 外れる．tools/splitsim.cpp の冒頭）．見出しの K・n は照合が見る
acc_item probe_splitsim "分割だけの到達精度（phi x b）" splitsim \
  'for phi in 0 2 4; do for b in 48 52 56 60 64 68; do sub phi=$phi b=$b K=1024 n=1024 -- ./src/build/splitsim $phi $b 1024; done; done'
# NL=3 は 3 本組・5 項（今の実装と同じ構造），NL=6 は 4 本組・7 項（tools/reconsim.cpp の冒頭）
acc_item probe_reconsim "再構成の limb の丸め（3 本組／4 本組）" reconsim \
  'for nl in 3 6; do sub S=16 NL=$nl RED=1 -- ./src/build/reconsim 16 $nl 1; done'
acc_item probe_twosum "tfAddSmall の判定と第 2 段" twosum_probe \
  'for e in "" TWOSUM_PROBE_ABS=1 TWOSUM_PROBE_FAST=1 "TWOSUM_PROBE_ABS=1 TWOSUM_PROBE_FAST=1"; do
     for s in 11 16; do sub env=${e:-none} seed=$s -- env $e ./src/build/twosum_probe 14080 $s 20000; done; done' \
  "0 1"   # 誤った規則（TWOSUM_PROBE_ABS）の回は 1 で終わるのが期待どおり
acc_item probe_tfconv "triple-float 変換の 8 検査" tfconv 'sub tfconv -- ./src/build/tfconv'

acc_item probe_modround "ModRound の総当たり" modround_check \
  'sub K=14080 -- ./src/build/modround_check 14080'
