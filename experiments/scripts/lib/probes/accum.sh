# lib/probes/accum.sh -- diagnose/probes.sh の群 (3) 累積側：整列窓の 2 項実験・無誤差分割の K 依存・DST
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(3) 累積側：整列窓の 2 項実験・無誤差分割の K 依存・DST"
acc_item probe_micro "2 項実験（512^3．d・k2・fidelity・K）" gemm_bf16ext \
  'for fid in LoFi HiFi2 HiFi3 HiFi4; do sub fid=$fid d=12 k2=1 K=512 -- env OZ_FIDELITY=$fid OZ_EXPDIFF=12 OZ_EXPK=1 ./src/build/gemm_bf16ext 512 512 512 0 1 1; done
   for K in 512 1024 2048 4096; do sub d=12 k2=1 K=$K -- env OZ_EXPDIFF=12 OZ_EXPK=1 ./src/build/gemm_bf16ext 512 512 $K 0 1 1; done
   for d in 8 10 11 12 16 23 24; do for k in 1 4 7 8 16 31 64; do
     sub d=$d k2=$k K=512 -- env OZ_EXPDIFF=$d OZ_EXPK=$k ./src/build/gemm_bf16ext 512 512 512 0 1 1; done; done'
acc_item probe_fp32acc "2 項実験（単一コア・K=32）" fp32acc_probe 'sub fp32acc_probe -- ./src/build/fp32acc_probe'
acc_item probe_ext_k "BF16x2 の到達精度の K 依存（3520x3520xK）" gemm_bf16ext \
  'for K in 1024 3520 7040 14080; do sub K=$K terms=3 -- ./src/build/gemm_bf16ext 3520 3520 $K 0 1 3; done'
acc_item probe_dst "DST の FP32 の本数" dst_probe 'sub dst_probe -- ./src/build/dst_probe 16 64'
acc_item probe_fma "SFPMAD の保持幅" fma_probe 'sub fma_probe -- ./src/build/fma_probe'
acc_item probe_bf16acc "BF16 累積の単体 GEMM の到達精度" gemm_bf16 \
  'for N in 3520 7040 14080; do sub N=$N -- ./src/build/gemm_bf16 $N $N $N 1 1 1 1 0 0 0; done'

acc_item probe_tfmul "SFPU の模擬と実機の照合" tfmul_probe \
  'sub tfmul_probe per=4 -- ./src/build/tfmul_probe'
