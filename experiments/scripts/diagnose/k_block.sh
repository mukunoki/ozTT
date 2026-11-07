#!/bin/bash
# K ブロック化を外した構成と既定構成の比較。単独実行も一括実行も同じ計時部品を使う。
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  cd "$(dirname "$0")/../.." || exit 2
  exec bash scripts/diagnose/ablations.sh --group k_block "$@"
fi
[ "$LIST" = 1 ] || echo "===== (1) K ブロック化の除去（$N^3，Ozaki-I-BF16）====="
run "既定（対ループ＋遅延再構成．融合は働かない）" "abl_kb_on_$N" "$(role paper -)" -- ./src/build/$BF16_BIN $N $N $N $BF16_PA 1 3
run "三つとも外す（K ブロックごとに Horner）" "abl_kb_off_$N" "$(role paper -)" \
    OZ_NO_DEFER=1 OZ_NO_FUSE_ACC=1 OZ_NO_PAIRLOOP=1 -- ./src/build/$BF16_BIN $N $N $N $BF16_PA 1 3
