# lib/probes/ext.sh -- diagnose/probes.sh の群 (拡張) どの原稿も今は引いていない項目（PROBES_EXTENDED）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
if [ "$EXT" != 0 ]; then
  IN_EXT=1
  section "(拡張) どの原稿も今は引いていない項目"
  # soft-float の倍率の代案（S3．原稿は diagnose/implementation_ratios.sh の段別の倍率を引く）
  oz_set oz1int8
  perf "I-INT8 4096 整数" probe_sf_off_int8_4096 -- ./src/build/$OZ_BIN 4096 4096 4096 $OZ_PA 1 3
  perf "I-INT8 4096 soft-float" probe_sf_on_int8_4096 OZ_SOFTFLOAT=1 -- ./src/build/$OZ_BIN 4096 4096 4096 $OZ_PA 1 3
fi
