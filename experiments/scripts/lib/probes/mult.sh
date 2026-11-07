# lib/probes/mult.sh -- diagnose/probes.sh の群 (4) 乗算側：形式 x fidelity の無誤差境界と乱数入力
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(4) 乗算側：形式 x fidelity の無誤差境界と乱数入力（fp8_probe，単一コア 32x32x32）"
acc_item probe_fp8_digit "無誤差境界（整数 R の掃引）" fp8_probe \
  'for f in bf16 fp16 fp32; do for fid in 1 2 3 4; do for R in 31 32 33 62 63 64 65; do
     sub fmt=$f fid=$fid R=$R -- ./src/build/fp8_probe $f $fid $R 1; done; done; done
   for fid in 3 4; do for R in 127 128; do sub fmt=bf16 fid=$fid R=$R -- ./src/build/fp8_probe bf16 $fid $R 1; done; done
   for fid in 1 2 3 4; do for R in 15 16; do sub fmt=fp8 fid=$fid R=$R -- ./src/build/fp8_probe fp8 $fid $R 1; done; done' \
  "0 1"   # 境界を越えた回は無誤差でなく 1 で終わる
acc_item probe_fp8_rand "乱数入力の到達精度" fp8_probe \
  'for fid in 1 2 3 4; do
     for m in "bf16 randbf16" "fp16 randbf16" "fp32 randbf16" "fp32 randtf32" "fp32 randa10" "fp32 randb10" "bf16 rand" "fp16 rand" "fp32 rand"; do
       set -- $m; sub fmt=$1 fid=$fid mode=$2 fp32acc=1 -- ./src/build/fp8_probe $1 $fid $2 1; done
     sub fmt=fp32 fid=$fid mode=rand fp32acc=0 -- env PROBE_FP32ACC=0 ./src/build/fp8_probe fp32 $fid rand 1
   done
   sub fmt=fp16 fid=4 R=8 out=fp16 fp32acc=0 -- env PROBE_FP32ACC=0 ./src/build/fp8_probe fp16 4 8 1 fp16'
