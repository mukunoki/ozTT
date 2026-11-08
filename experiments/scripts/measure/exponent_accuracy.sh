#!/bin/bash
# 指数部の範囲を広げた入力で p・b を選び直したときの精度。
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  cd "$(dirname "$0")/../.." || exit 2
  exec bash scripts/measure/exponent_adaptive.sh --part accuracy "$@"
fi
[ "$CMDS" = 1 ] || echo "===== 到達精度（$OUT）====="
for phi in $PHIS; do
  acc "acc_oz1bfp8_phi${phi}" "I-BFP8 phi=$phi" "$I_BIN" OZ_PHI=$phi OZ_FP64=1 OZ_MIN_BITS=1 -- 14080 14080 14080 $I_PA 1 1
  for np in 1 2; do
    acc "acc_oz2bfp8_phi${phi}_p${np}" "II-BFP8 phi=$phi 再構成 $np 回" "$II_BIN" \
      OZ_PHI=$phi OZ2_FP64=1 OZ2_NOWIN=1 OZ2_PASSES=$np OZ_MIN_BITS=1 -- 14080 14080 14080 $II_PA 1 1
  done
done
