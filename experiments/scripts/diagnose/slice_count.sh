#!/bin/bash
# スライス数を 1 段増やしたときの感度。単独実行と一括実行で同じ計時部品を使う。
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  cd "$(dirname "$0")/../.." || exit 2
  exec bash scripts/diagnose/ablations.sh --group slice_count "$@"
fi
[ "$LIST" = 1 ] || echo "===== (2) スライス数の感度（$N^3，Ozaki-I-INT8）====="
run "p=${INT8_PA%% *}（既定）" "abl_p8_$N" "$(role paper -)" -- ./src/build/$INT8_BIN $N $N $N $INT8_PA 1 3
run "p=9" "abl_p9_$N" "$(role paper -)" -- ./src/build/$INT8_BIN $N $N $N 9 8 1 3
