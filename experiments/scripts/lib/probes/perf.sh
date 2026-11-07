# lib/probes/perf.sh -- diagnose/probes.sh の群 (7) 性能の項目（1 項目ずつ冷ましてから）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(7) 性能の項目（1 項目ずつ ${TMAX} degC 以下から．OZ_NOVERIFY=1・OZ_TIMING=$OZ_TIMING）"
# L1 予算 1280 KiB と既定（同じ回に並べて測る）
for r in oz1bf16 oz2bfp8; do
  oz_set $r
  perf "$r 既定の L1 予算" "probe_l1b_default_${r}_14080" -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
  perf "$r L1 予算 1280 KiB" "probe_l1b_1280_${r}_14080" OZ_L1_BUDGET_KB=1280 -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
done
# スキーム I の再構成段：ライタの仕事を段階的に外す（OZ_RECON_PROBE=1 組み立て，2 指数差分の読み，3 書き出し）
oz_set oz1bf16
for N in 3520 14080; do
  for r in 0 1 2 3; do
    perf "I-BF16 $N OZ_RECON_PROBE=$r" "probe_rprobe${r}_$N" OZ_RECON_PROBE=$r -- ./src/build/$OZ_BIN $N $N $N $OZ_PA 1 3
  done
done
# 組み立てを除いた再構成段と対角の本数（p を振る．近道 NSMALL を切って全対角を同じ読み方にする）
for P in 4 8 12 16; do
  perf "I-BF16 3520 PROBE=3 p=$P" "probe_ndiag_p${P}_3520" OZ_RECON_PROBE=3 OZ_NSMALL=0 -- ./src/build/ozaki1_bf16 3520 3520 3520 $P $((P-1)) 1 3
done
# 長い K（M=N=3520）：本文の上限 K=140,800（上の oz_set oz1bf16 のまま）
perf "I-BF16 3520x3520x140800" probe_longk_oz1bf16_140800 -- ./src/build/$OZ_BIN 3520 3520 140800 $OZ_PA 1 3
# 対ループ・K ブロック化の切り分け（既定側は measure/performance.sh の cool_oz1{bf16,bfp8}_14080，強制側は diagnose/ablations.sh の abl_kb_off_14080）
perf "I-BF16 対ごと＋融合（NO_PAIRLOOP）" probe_nopl_oz1bf16_14080 OZ_NO_PAIRLOOP=1 -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
oz_set oz1bfp8
perf "I-BFP8 対ごと＋融合（NO_PAIRLOOP）" probe_nopl_oz1bfp8_14080 OZ_NO_PAIRLOOP=1 -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
perf "I-BFP8 K ブロックごとの Horner" probe_kboff_oz1bfp8_14080 OZ_NO_DEFER=1 OZ_NO_FUSE_ACC=1 OZ_NO_PAIRLOOP=1 \
  -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
perf "I-BFP8 bw=11（L1 予算 1024 KiB）" probe_bw11_oz1bfp8_14080 OZ_L1_BUDGET_KB=1024 -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
oz_set oz2int8
perf "II-INT8 2 パス再構成（b=54）" probe_2pass_14080 OZ2_FP64=1 OZ2_PASSES=2 OZ2_B=54 -- ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
# 素の BF16 GEMM：MathFidelity 別（FP32 累積．measure/fidelity_comparison.sh の gemmfid_* と同じ引数）と形状の掃引（K=14080）
RUN_PAT='Performance : [0-9.]+ TFLOP/s'
for N in 3520 7040; do
  for fid in LoFi HiFi2 HiFi3 HiFi4; do
    perf "素の GEMM $fid $N" "probe_gemmfid_${fid}_$N" OZ_FIDELITY=$fid -- ./src/build/gemm_bf16 $N $N $N 1 1 3 0 0 1 0
  done
done
for mn in "640 704 LoFi" "1280 1408 LoFi" "1280 2816 LoFi" "2560 1408 LoFi" "2560 2816 LoFi" "3520 3520 LoFi"; do
  set -- $mn
  perf "素の GEMM ${1}x${2}x14080 $3" "probe_shape_${1}x${2}_$3" OZ_FIDELITY=$3 -- ./src/build/gemm_bf16 $1 $2 14080 1 1 3 0 0 1 0
done
unset RUN_PAT
