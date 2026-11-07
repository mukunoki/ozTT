# lib/probes/bound.sh -- diagnose/probes.sh の群 (5) 尾崎の到達精度で見る境界（冷まさない）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(5) 尾崎の到達精度で見る境界（冷まさない．OZ_MIN_BITS=1 は精度不足を異常終了と分けるため）"
acc_item probe_int8_fid "INT8 の fidelity 別（基数 254・1280^3・p=8）" ozaki1_int8 \
  'for fid in LoFi HiFi2 HiFi3 HiFi4; do sub radix=254 fid=$fid -- env OZ_FIDELITY=$fid OZ_MIN_BITS=1 ./src/build/ozaki1_int8 1280 1280 1280 8 7 1 1; done' \
  "0 1"   # 足りない fidelity の破綻を見る掃引
acc_item probe_int8_zero "INT8 で出力が 0 になる基数（1280^3・p=8）" ozaki1_int8 \
  'for r in 31 15 62 30; do for fid in LoFi HiFi2 HiFi3 HiFi4; do
     sub radix=$r fid=$fid -- env OZ_RADIX=$r OZ_FIDELITY=$fid OZ_MIN_BITS=1 ./src/build/ozaki1_int8 1280 1280 1280 8 7 1 1; done; done' \
  "0 1"   # 出力が厳密に 0 になる回は 1 bit にも届かず 1 で終わる
acc_item probe_digit_lofi "I-BF16 の桁 32／33 を LoFi で（3520^3）" ozaki1_bf16 \
  'for r in 64 66; do sub radix=$r fid=LoFi -- env OZ_RADIX=$r OZ_FIDELITY=LoFi OZ_MIN_BITS=1 ./src/build/ozaki1_bf16 3520 3520 3520 10 9 1 1; done' \
  "0 1"   # 桁 33 の破綻を見る掃引
acc_item probe_radix_odd "beta=2d+1 の破れ（512x512xK）" ozaki1_bf16 'sub radix_odd_check -- bash scripts/diagnose/radix_odd.sh'
acc_item probe_radix_d "beta=d のときの p（512^3）" "ozaki1_bf16 ozaki1_bfp8 ozaki1_fp8 ozaki1_bfp4" \
  'for c in "bf16 32 10" "bf16 32 11" "bfp8 32 10" "bfp8 32 11" "fp8 16 13" "fp8 16 14" "bfp4 7 19" "bfp4 7 20"; do set -- $c
     sub fmt=$1 radix=$2 dmax=$2 p=$3 -- env OZ_RADIX=$2 OZ_DMAX=$2 OZ_MIN_BITS=1 ./src/build/ozaki1_$1 512 512 512 $3 $(($3-1)) 1 1; done'
acc_item probe_pickp "スライス数 p の決め方（M=N=512，K=512・16384）" "$PICKP_BINS" \
  'sub pick_p -- bash scripts/diagnose/select_slice_count.sh'
acc_item probe_psweep_1280 "I-BF16 の p 掃引（1280^3）" ozaki1_bf16 \
  'for p in 6 7 8 9 10 11; do sub p=$p -- env OZ_MIN_BITS=1 ./src/build/ozaki1_bf16 1280 1280 1280 $p $((p-1)) 1 1; done'
oz_set oz1int8   # 既定の構成（下の単引用符の断片はコンテナの中で展開するので，既定の引数だけをここで埋める）
acc_item probe_int8_softfloat "I-INT8 の最終スケールを double で（3 サイズ）" "$OZ_BIN" \
  'for N in 3520 7040 14080; do sub N=$N softfloat=1 -- env OZ_SOFTFLOAT=1 OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' $N $N $N '"$OZ_PA"' 1 1; done'
# 最後の回は I-INT8 の既定の構成（上の oz_set oz1int8 のまま）
acc_item probe_nonneg_7040 "非負入力と無誤差条件（桁 63・HiFi2・p=9）" "ozaki1_bf16 $OZ_BIN" \
  'for inp in pos uni; do for nb in 0 1; do e="OZ_RADIX=126 OZ_FIDELITY=HiFi2 OZ_MIN_BITS=1"
     [ $inp = pos ] && e="$e OZ_POS=0.9"; [ $nb = 1 ] && e="$e OZ_NOBLOCK=1"
     sub input=$inp noblock=$nb radix=126 -- env $e ./src/build/ozaki1_bf16 7040 7040 7040 9 8 1 1; done; done
   sub input=pos int8 -- env OZ_POS=0.9 OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 7040 7040 7040 '"$OZ_PA"' 1 1' \
  "0 1"   # 無誤差条件を破る構成（NOBLOCK）を含む
oz_set oz2int8
acc_item probe_rowscale_7040 "行・列スケールの入力（II-INT8，7040^3）" "$OZ_BIN" \
  'sub rowscale=1 -- env OZ_ROWSCALE=1 OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 7040 7040 7040 '"$OZ_PA"' 1 1'
# 1 パスの回の法の集合は，同じ φ・b で OZ2_FP64 に選ばせた集合を準備の実行（検証なし）で取って与える．準備の実行は
# sub の外なので，終了コード 0・moduli 行がちょうど 1 本・法が空でない，を prep で確かめ，外れれば SUB_BAD に足して
# （EXIT,rc=3）その集合を使う回を流さない
acc_item probe_phi16 "phi=16 の b 掃引と 1 パス／2 パス（II-INT8，4096^3）" ozaki2_int8 \
  'prep() { local o r n; o=$(env "$@" 2>/dev/null); r=$?; n=$(printf "%s\n" "$o" | grep -c "^moduli: .* m={")
     m=$(printf "%s\n" "$o" | grep "^moduli: " | head -1 | grep -oE " m=\{[^}]*\}" | sed -e "s/^ m={//" -e "s/}$//" -e "s/[^0-9,]//g")
     if [ "$r" = 0 ] && [ "$n" = 1 ] && [ -n "$m" ]; then return 0; fi
     echo "ERROR: 法の集合を取る準備の実行が期待どおりでない（終了コード $r．moduli 行 $n 本．法 ${m:-なし}）．続く回を流さない: $*" >&2
     SUB_BAD="${SUB_BAD:+$SUB_BAD }prep:$r"; m=""; return 1; }
   for b in 54 60 65 71; do sub phi=16 b=$b pass=2 -- env OZ_PHI=16 OZ2_FP64=1 OZ2_PASSES=2 OZ2_B=$b OZ_MIN_BITS=1 ./src/build/ozaki2_int8 4096 4096 4096 24 1 1; done
   if prep OZ_PHI=16 OZ2_FP64=1 OZ2_B=71 OZ_NOVERIFY=1 ./src/build/ozaki2_int8 4096 4096 4096 24 1 1; then
     sub phi=16 b=71 pass=1 moduli=$m -- env OZ_PHI=16 OZ2_B=71 OZ2_MODULI=$m OZ_MIN_BITS=1 ./src/build/ozaki2_int8 4096 4096 4096 24 1 1
   fi
   if prep OZ_PHI=8 OZ2_FP64=1 OZ2_B=69 OZ_NOVERIFY=1 ./src/build/ozaki2_int8 4096 4096 4096 24 1 1; then
     sub phi=8 b=69 pass=2 moduli=$m -- env OZ_PHI=8 OZ2_FP64=1 OZ2_PASSES=2 OZ2_B=69 OZ2_MODULI=$m OZ_MIN_BITS=1 ./src/build/ozaki2_int8 4096 4096 4096 24 1 1
     sub phi=8 b=69 pass=1 moduli=$m -- env OZ_PHI=8 OZ2_B=69 OZ2_MODULI=$m OZ_MIN_BITS=1 ./src/build/ozaki2_int8 4096 4096 4096 24 1 1
   fi'
