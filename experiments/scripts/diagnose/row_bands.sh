#!/bin/bash
# 出力行バンドの確保方法の比較。14080 の一括確保は容量不足の診断。
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  cd "$(dirname "$0")/../.." || exit 2
  exec bash scripts/diagnose/ablations.sh --group row_bands "$@"
fi
[ "$LIST" = 1 ] || echo "===== (3) 行バンドの確保（$N^3，Ozaki-I-BFP2）====="
run "一括確保（バンド 1 本）" "abl_band_one_$N" "$(role oom docs)" OZ_BANDS=1 -- ./src/build/$BFP2_BIN $N $N $N $BFP2_PA 1 3
run "バンド 6 本" "abl_band_six_$N" "$(role - docs)" OZ_BANDS=6 -- ./src/build/$BFP2_BIN $N $N $N $BFP2_PA 1 3
