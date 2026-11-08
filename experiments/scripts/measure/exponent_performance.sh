#!/bin/bash
# 指数部の範囲を広げた入力で p・b を選び直したときの性能。
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  cd "$(dirname "$0")/../.." || exit 2
  exec bash scripts/measure/exponent_adaptive.sh --part performance "$@"
fi
[ "$CMDS" = 1 ] || echo "===== 性能（冷ましてから定常計時．全体の時間は全工程の通し OZ_TIME_WHOLE=1）====="
# 表示する指標の行は全体の時間（measure_lib.sh の run_dram の RUN_PAT．無ければ FAILED）
RUN_PAT='WHOLE-PIPELINE : .*'
for phi in $PHIS; do
  perf "I-BFP8 phi=$phi" "perf_oz1bfp8_phi${phi}" OZ_PHI=$phi OZ_FP64=1 OZ_TIME_WHOLE=1 -- ./src/build/$I_BIN 14080 14080 14080 $I_PA 1 3
  for np in 1 2; do
    perf "II-BFP8 phi=$phi 再構成 $np 回" "perf_oz2bfp8_phi${phi}_p${np}" OZ_PHI=$phi OZ2_FP64=1 OZ2_NOWIN=1 OZ2_PASSES=$np OZ_TIME_WHOLE=1 -- \
      ./src/build/$II_BIN 14080 14080 14080 $II_PA 1 3
  done
done
