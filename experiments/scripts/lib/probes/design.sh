# lib/probes/design.sh -- diagnose/probes.sh の群 (9) 設計の比較（PROBES_DESIGN．今の実装と代案を同じ回に並べて測る）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
# ---------------------------------------------------------------------------
# (9) 設計の比較：今の実装と代案（つまみで作る）を同じ回に並べて測る（PROBES_DESIGN．ログは probe_design_*）
#   ログ名の変種と，比べる相手：
#   betad            β=d（基数＝桁の上限．p は probe_radix_d が選ぶ値）と既定の β=2d
#   noblockmm        matmul_tiles（OZ_NO_BLOCKMM=1）と既定の matmul_block
#   kboff            K ブロックごとに Horner を回す構成（遅延再構成・融合・対ループを外す）．I-INT8 は既定で K ブロック 1
#   jsplit0          II の FP64 組み立てをリーダと分担しない（OZ2_JSPLIT=0）．II-INT8 は組み立てあり／なし（OZ_RECON_PROBE=1）
#                        × 法 16・12・8 本（既定の集合の先頭から．窓が足りず b は下がるが時間だけを読む）
#   l1b1280          L1 予算 1280 KiB と既定の 1408 KiB（II-BFP8 と I-BF16 の対は既定の群の probe_l1b_*）
#   nodram・nomcast   GEMM 段の探針（送信側の DRAM 読みを除く・A と B の供給を除く）
#   r64hifi2・r126・r126nb・r126pos・r126posnb
#                        HiFi2 で基数を上げる選び分けと，K ブロック化の切り分け：既定（β=64・LoFi・p=10）を A，
#                        同じ構成を HiFi2 にしたものを C，β=126（桁 63）・HiFi2・p=9 を B（FP32 累積の無誤差条件から
#                        K>4096 で K ブロックに分かれる），B から K ブロック化を外したもの（OZ_NOBLOCK．無誤差条件を破る）を D．
#                        pos は非負入力（OZ_POS=0.9．probe_nonneg_7040 と同じ入力）
#   nopl・scratch     対を 1 本のプログラムで回さず対ごとに発行する構成（OZ_NO_PAIRLOOP=1．Int32 の累積は GEMM の
#                        出力経路で足す）と，さらに累積を別プログラムで DRAM の累積面へ足す構成（OZ_NO_FUSE_ACC=1 も．足し込み
#                        55 回）．既定（対ループ．累積面の更新は対角ごとの 10 回で再構成段は最後の Horner だけ）と比べて，
#                        1 回の面更新・足し込みの値段と固定費に分ける
#   modtable         法の集合を表の選び方（段の時間のモデルの前の選び方．OZ2_MODULI で与える）にしたもの
#   digit<d>_7040    桁 63・32・31・15（β=126 HiFi2 p=9，既定，β=62 p=10，β=30 p=12．p は到達精度が 53.9 bit を上回る
#                        最小の値で，probe_design_digit_acc_7040 がその 1 段下と並べて確かめる）
#   ssweep           II-BFP8 の法を平方数だけの 7・9・11 本に固定して S だけを振る（先行工程と格子は同じ）
#   nsmall           II-BFP8 3520^3 で tfAddSmall で足す断片の本数（OZ2_NSMALL=0..5）
#   longk            M=N=3520 で K を伸ばす（K=3520 は base_oz1bf16_3520）
#   kgemm            素の GEMM（LoFi，FP32 累積，3520x3520）の K=1408（旧設計の K ブロック 1 個ぶん）と K=14080
#   精度の項目（冷まさない）：digit64（桁 63／64 の破綻），digit_acc_7040，r126_acc，winexcess_7040（法 16 本のまま
#   窓の過剰を振る），nsmall_acc（OZ2_NSMALL の各構成と 2 パス），modtable_acc
# 効果が数 % の比較のために，既定の構成（base）も経路ごとに同じ回で測り，代案をその直後に並べる．
if [ "$DES" != 0 ] && [ "$EXT" != only ]; then
  IN_DESIGN=1
  section "(9) 設計の比較（今の実装と代案を同じ回に並べて測る）"
  # dperf 変種 経路 N 環境変数… [-- 引数の上書き]   経路の既定の構成（oz_set．routes.tsv の p・L・法の本数）で 1 項目
  #   （ログ probe_design_<変種>_<経路>_<N>）．-- の後に引数を与えると p・L（法の本数）をそれに替える
  dperf() {
    local v=$1 r=$2 N=$3 env="" args; shift 3
    oz_set "$r"; args=$OZ_PA
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do env="$env $1"; shift; done
    if [ "${1:-}" = "--" ]; then shift; args="$*"; fi
    perf "$r $v $N" "probe_design_${v}_${r}_$N" $env -- ./src/build/$OZ_BIN $N $N $N $args 1 3
  }
  # 表の選び方の法の集合（14080^3．OZ2_TABLE_B8 の平方数から末尾 2 本を選び直したもの（II-TF32 の既定と同じ），
  # OZ2_TABLE_B4 の先頭 17 本，OZ2_TABLE_B2 の先頭 16 本）．スキーム II の BF16・FP16・BFP8_b は同じ桁 32 の表を使う
  MOD_TBL_D32=4096,3969,3721,3481,3025,2809,2209,1849,1681,961,841
  MOD_TBL_B4=225,196,169,121,223,211,199,197,193,191,181,179,173,167,163,157,151
  MOD_TBL_B2=243,242,241,239,235,233,229,227,223,221,217,211,199,197,193,191
  # II-INT8 の既定の法（14080^3，16 本）の先頭 12 本・8 本
  MOD_I8_12=255,254,253,251,247,241,239,233,229,227,223,217
  MOD_I8_8=255,254,253,251,247,241,239,233
  N=14080
  # スキーム I
  dperf base     oz1bf16 $N
  dperf betad    oz1bf16 $N OZ_RADIX=32 OZ_DMAX=32 -- 11 10
  dperf noblockmm oz1bf16 $N OZ_NO_BLOCKMM=1
  dperf nodram   oz1bf16 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz1bf16 $N OZ_GEMM_PROBE_NOMCAST=1
  dperf kboff    oz1bf16 $N OZ_NO_DEFER=1 OZ_NO_FUSE_ACC=1 OZ_NO_PAIRLOOP=1
  dperf nopl     oz1bf16 $N OZ_NO_PAIRLOOP=1
  dperf scratch  oz1bf16 $N OZ_NO_PAIRLOOP=1 OZ_NO_FUSE_ACC=1
  dperf r64hifi2 oz1bf16 $N OZ_FIDELITY=HiFi2
  dperf r126     oz1bf16 $N OZ_RADIX=126 OZ_FIDELITY=HiFi2 -- 9 8
  dperf r126nb   oz1bf16 $N OZ_RADIX=126 OZ_FIDELITY=HiFi2 OZ_NOBLOCK=1 -- 9 8
  dperf r126pos  oz1bf16 $N OZ_RADIX=126 OZ_FIDELITY=HiFi2 OZ_POS=0.9 -- 9 8
  dperf r126posnb oz1bf16 $N OZ_RADIX=126 OZ_FIDELITY=HiFi2 OZ_POS=0.9 OZ_NOBLOCK=1 -- 9 8
  dperf base     oz1bfp8 $N
  dperf betad    oz1bfp8 $N OZ_RADIX=32 OZ_DMAX=32 -- 11 10
  dperf noblockmm oz1bfp8 $N OZ_NO_BLOCKMM=1
  dperf nodram   oz1bfp8 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz1bfp8 $N OZ_GEMM_PROBE_NOMCAST=1
  dperf kboff    oz1bfp8 $N OZ_NO_DEFER=1 OZ_NO_FUSE_ACC=1 OZ_NO_PAIRLOOP=1
  dperf base     oz1fp8 $N
  dperf betad    oz1fp8 $N OZ_RADIX=16 OZ_DMAX=16 -- 14 13
  dperf noblockmm oz1fp8 $N OZ_NO_BLOCKMM=1
  dperf kboff    oz1fp8 $N OZ_NO_DEFER=1 OZ_NO_FUSE_ACC=1 OZ_NO_PAIRLOOP=1
  dperf base     oz1bfp4 $N
  dperf betad    oz1bfp4 $N OZ_RADIX=7 OZ_DMAX=7 -- 20 19
  dperf nodram   oz1bfp4 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz1bfp4 $N OZ_GEMM_PROBE_NOMCAST=1
  dperf base     oz1int8 $N
  dperf noblockmm oz1int8 $N OZ_NO_BLOCKMM=1
  dperf base     oz1fp32 $N
  dperf nodram   oz1fp32 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz1fp32 $N OZ_GEMM_PROBE_NOMCAST=1
  # スキーム II
  dperf base     oz2int8 $N
  dperf noblockmm oz2int8 $N OZ_NO_BLOCKMM=1
  dperf l1b1280  oz2int8 $N OZ_L1_BUDGET_KB=1280
  dperf jsplit0_s16    oz2int8 $N OZ2_JSPLIT=0
  dperf jsplit0rp1_s16 oz2int8 $N OZ2_JSPLIT=0 OZ_RECON_PROBE=1
  dperf jsplit0_s12    oz2int8 $N OZ2_JSPLIT=0 OZ2_MODULI=$MOD_I8_12
  dperf jsplit0rp1_s12 oz2int8 $N OZ2_JSPLIT=0 OZ_RECON_PROBE=1 OZ2_MODULI=$MOD_I8_12
  dperf jsplit0_s8     oz2int8 $N OZ2_JSPLIT=0 OZ2_MODULI=$MOD_I8_8
  dperf jsplit0rp1_s8  oz2int8 $N OZ2_JSPLIT=0 OZ_RECON_PROBE=1 OZ2_MODULI=$MOD_I8_8
  dperf base     oz2fp8 $N
  dperf noblockmm oz2fp8 $N OZ_NO_BLOCKMM=1
  dperf l1b1280  oz2fp8 $N OZ_L1_BUDGET_KB=1280
  dperf jsplit0  oz2fp8 $N OZ2_JSPLIT=0
  dperf base     oz2bfp8 $N
  dperf l1b1280  oz2bfp8 $N OZ_L1_BUDGET_KB=1280
  dperf nodram   oz2bfp8 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz2bfp8 $N OZ_GEMM_PROBE_NOMCAST=1
  dperf modtable oz2bfp8 $N OZ2_MODULI=$MOD_TBL_D32
  dperf base     oz2bf16 $N
  dperf nodram   oz2bf16 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz2bf16 $N OZ_GEMM_PROBE_NOMCAST=1
  dperf modtable oz2bf16 $N OZ2_MODULI=$MOD_TBL_D32
  dperf base     oz2fp16 $N
  dperf modtable oz2fp16 $N OZ2_MODULI=$MOD_TBL_D32
  dperf base     oz2bfp4 $N
  dperf l1b1280  oz2bfp4 $N OZ_L1_BUDGET_KB=1280
  dperf modtable oz2bfp4 $N OZ2_MODULI=$MOD_TBL_B4
  dperf base     oz2bfp2 $N
  dperf modtable oz2bfp2 $N OZ2_MODULI=$MOD_TBL_B2
  dperf base     oz2fp32 $N
  dperf nodram   oz2fp32 $N OZ_GEMM_PROBE_NODRAM=1
  dperf nomcast  oz2fp32 $N OZ_GEMM_PROBE_NOMCAST=1
  # 残りの 9 経路の GEMM 段の探針（既定の構成は同じ回に隣で測る sbase．上の base を持つ経路は base と比べる）
  for r in oz2fp8 oz2int8 oz2bfp4 oz2bfp2 oz2fp16 oz1fp8 oz1int8 oz1bfp2 oz1fp16; do
    dperf sbase   $r $N
    dperf nodram  $r $N OZ_GEMM_PROBE_NODRAM=1
    dperf nomcast $r $N OZ_GEMM_PROBE_NOMCAST=1
  done
  # 出力の書き出しを外す探針（単独と，A・B の供給も外したもの）．既定の構成は同じ回に隣で測る（wbase）
  for r in oz2bfp8 oz2bf16 oz2fp32 oz2bfp2 oz1bfp8 oz1bf16 oz1fp32 oz1bfp4; do
    dperf wbase   $r $N
    dperf nowrite $r $N OZ_GEMM_PROBE_NOWRITE=1
    dperf nmnw    $r $N OZ_GEMM_PROBE_NOMCAST=1 OZ_GEMM_PROBE_NOWRITE=1
  done
  # phi を広げた入力で入力から p・b を選び直した構成（OZ_FP64／OZ2_FP64．II は 2 パス再構成に固定）の 14080^3．
  # 到達精度（Dot2．冷まさない）と性能．CPU の DGEMM の同じ入力の値は DGEMMREF_CORETYPE=ZEN DGEMMREF_SET=phi14080
  # ./scripts/measure/cpu_dgemm_reference.sh（II のパス数を自動で選ぶ構成は下の probe_phiacc_auto_*・probe_phiperf_auto_*）．
  # この項目と下の probe_phiacc_*・probe_phiperf_*・probe_passcal_* は，スキーム II の窓も入力から見積もる（OZ2_NOWIN を
  # 付けない）ので，表の構成（窓は最悪の上界 K 2^{2b}）より法が少なくなりうる．論文・スライドの φ の値はこれらではなく
  # measure/exponent_adaptive.sh（OZ2_NOWIN=1．ログは measure_logs/phi_nowin/<rev7>/ の今の組）から引き，
  # verify/check_paper_numbers.py・probe_checks.py はこれらのログを読まない（docs/accuracy.md の再構成のパス数の記録などが
  # 引く参考の値として残す）．したがってこの項目から probe_phiperf_p1_* までは docs だけが引く群（PROBES_DOCS）．
  # その前に，1 パスの復元誤差の見込みの係数を較正した測定（4096^3 で 1 パスと 2 パスを強制して比べる）を置く：論文の本文が
  # 引く「η の実効値（1 回と 2 回の再構成の誤差を比べた値）」の出どころなので，docs だけの群に入れない
  oz_set oz2bfp8
  acc_item probe_passcal_oz2bfp8_4096 "II-BFP8 4096^3 の 1 パス・2 パスの到達精度（パス数の見込みの較正）" "$OZ_BIN" \
    'for phi in 0 0.5 1 2 4 8; do for np in 1 2; do
       sub phi=$phi passes=$np -- env OZ_PHI=$phi OZ2_FP64=1 OZ2_PASSES=$np OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 4096 4096 4096 '"$OZ_PA"' 1 1; done; done'
  IN_DOCS=1
  for phi in 0 0.5 1 2 4 8; do
    for rn in oz1bfp8 oz2bfp8; do
      oz_set $rn   # 引数の p・法の本数は選び直しの初期値（既定の構成と同じ）
      if [ "${rn:0:3}" = oz1 ]; then aenv="OZ_FP64=1"; else aenv="OZ2_FP64=1 OZ2_PASSES=2"; fi
      acc_item "probe_phiacc_${rn}_phi${phi}_14080" "$rn phi=$phi 選び直しの到達精度" "$OZ_BIN" \
        "sub $rn phi=$phi -- env OZ_PHI=$phi $aenv ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 1" \
        "0 1"   # OZ_MIN_BITS を付けないので，53 bit を割る φ は 1 で終わる（参考の値としてそのまま残す）
      perf "$rn phi=$phi 選び直し" "probe_phiperf_${rn}_phi${phi}_14080" OZ_PHI=$phi $aenv -- \
        ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
    done
  done
  # 再構成のパス数の自動選択（OZ2_PASSES．OZ2_FP64 の既定）で，自動で選んだ構成の 14080^3 の到達精度と性能（docs/accuracy.md）
  oz_set oz2bfp8   # この節の II-BFP8 の項目（自動選択・1 パスの固定・S の掃引・断片の本数）は既定の法の本数で
  for phi in 0 0.5 1 2 4 8; do
    acc_item "probe_phiacc_auto_oz2bfp8_phi${phi}_14080" "oz2bfp8 phi=$phi 選び直し（パス数も自動）の到達精度" "$OZ_BIN" \
      "sub oz2bfp8 phi=$phi passes=auto -- env OZ_PHI=$phi OZ2_FP64=1 ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 1" \
      "0 1"   # 同上（53 bit を割る φ も記録として残す）
    perf "oz2bfp8 phi=$phi 選び直し（パス数も自動）" "probe_phiperf_auto_oz2bfp8_phi${phi}_14080" OZ_PHI=$phi OZ2_FP64=1 -- \
      ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
  done
  # 1 パスで足りる範囲の境目：phi=4 は自動選択では 2 パス（見込み 2^-52.3．余裕 2 bit）だが 1 パスでも 53 bit を超えるので，
  # 1 パスに固定した到達精度と性能を測る．phi=8 は 1 パスでは 53 bit に届かないことを 14080^3 でも確かめる
  for phi in 4 8; do
    acc_item "probe_phiacc_p1_oz2bfp8_phi${phi}_14080" "oz2bfp8 phi=$phi 選び直し（1 パスに固定）の到達精度" "$OZ_BIN" \
      "sub oz2bfp8 phi=$phi passes=1 -- env OZ_PHI=$phi OZ2_FP64=1 OZ2_PASSES=1 OZ_MIN_BITS=1 ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 1"
  done
  perf "oz2bfp8 phi=4 選び直し（1 パスに固定）" probe_phiperf_p1_oz2bfp8_phi4_14080 OZ_PHI=4 OZ2_FP64=1 OZ2_PASSES=1 -- \
    ./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA 1 3
  IN_DOCS=0
  # S だけを振る（II-BFP8，平方数だけの法 7・9・11 本．b は窓から決まる）
  oz_set oz2bfp8
  SQ11=(4096 3969 3721 3481 3025 2809 2209 1849 1681 1369 961)
  for n in 3520 14080; do for S in 7 9 11; do
    m=$(IFS=,; echo "${SQ11[*]:0:$S}")
    perf "II-BFP8 $n 平方数 $S 本" "probe_design_ssweep_s${S}_oz2bfp8_$n" OZ2_MODULI=$m -- ./src/build/$OZ_BIN $n $n $n $OZ_PA 1 3
  done; done
  # 小さいサイズ：HiFi2 で基数を上げる選び分け（3520^3・7040^3）と桁 63/32/31/15（7040^3）
  dperf base oz1bf16 3520
  dperf r126 oz1bf16 3520 OZ_RADIX=126 OZ_FIDELITY=HiFi2 -- 9 8
  dperf base oz1bfp8 3520
  dperf r126 oz1bfp8 3520 OZ_RADIX=126 OZ_FIDELITY=HiFi2 -- 9 8
  # 行は「桁 p 環境」．桁 32 は既定の構成そのもの（つまみを立てない）
  for spec in "63 9 OZ_RADIX=126 OZ_FIDELITY=HiFi2" "32 10" "31 10 OZ_RADIX=62" "15 12 OZ_RADIX=30"; do
    set -- $spec; d=$1 P=$2; shift 2
    perf "I-BF16 7040 桁 $d p=$P" "probe_design_digit${d}_oz1bf16_7040" "$@" -- ./src/build/ozaki1_bf16 7040 7040 7040 $P $((P-1)) 1 3
  done
  oz_set oz2bfp8
  for n in 0 1 2 3 4 5; do
    perf "II-BFP8 3520 OZ2_NSMALL=$n" "probe_design_nsmall${n}_oz2bfp8_3520" OZ2_NSMALL=$n -- ./src/build/$OZ_BIN 3520 3520 3520 $OZ_PA 1 3
  done
  oz_set oz1bf16
  for K in 22016 44032 88064 140800; do
    perf "I-BF16 3520x3520x$K" "probe_design_longk_oz1bf16_$K" -- ./src/build/$OZ_BIN 3520 3520 $K $OZ_PA 1 3
  done
  RUN_PAT='Performance : [0-9.]+ TFLOP/s'
  for K in 1408 14080; do
    perf "素の GEMM LoFi 3520x3520x$K" probe_design_kgemm_3520x3520x$K OZ_FIDELITY=LoFi -- ./src/build/gemm_bf16 3520 3520 $K 1 1 3 0 0 1 0
  done
  unset RUN_PAT
  # 精度の項目（冷まさない．OZ_MIN_BITS=1 は精度不足を異常終了と分けるため）
  acc_item probe_design_digit64_acc "桁 63／64（β=126／128，1280^3，p=8）" ozaki1_bf16 \
    'for r in 126 128; do for fid in HiFi2 HiFi4; do sub radix=$r fid=$fid p=8 -- env OZ_RADIX=$r OZ_FIDELITY=$fid OZ_MIN_BITS=1 ./src/build/ozaki1_bf16 1280 1280 1280 8 7 1 1; done; done' \
    "0 1"   # 桁 64 の破綻を見る掃引
  acc_item probe_design_digit_acc_7040 "桁 63/31/15 の p と 1 段下（7040^3）" ozaki1_bf16 \
    'for c in "126 HiFi2 8" "126 HiFi2 9" "62 LoFi 9" "62 LoFi 10" "30 LoFi 11" "30 LoFi 12"; do set -- $c
       sub radix=$1 fid=$2 p=$3 -- env OZ_RADIX=$1 OZ_FIDELITY=$2 OZ_MIN_BITS=1 ./src/build/ozaki1_bf16 7040 7040 7040 $3 $(($3-1)) 1 1; done'
  acc_item probe_design_r126_acc "β=126・HiFi2・p=9 の到達精度（3520^3，14080^3 の乱数・非負 x NOBLOCK）" "ozaki1_bf16 ozaki1_bfp8" \
    'for f in bf16 bfp8; do sub fmt=$f N=3520 input=uni noblock=0 -- env OZ_RADIX=126 OZ_FIDELITY=HiFi2 OZ_MIN_BITS=1 ./src/build/ozaki1_$f 3520 3520 3520 9 8 1 1; done
     for inp in uni pos; do for nb in 0 1; do e="OZ_RADIX=126 OZ_FIDELITY=HiFi2 OZ_MIN_BITS=1"
       [ $inp = pos ] && e="$e OZ_POS=0.9"; [ $nb = 1 ] && e="$e OZ_NOBLOCK=1"
       sub fmt=bf16 N=14080 input=$inp noblock=$nb -- env $e ./src/build/ozaki1_bf16 14080 14080 14080 9 8 1 1; done; done' \
    "0 1"   # 無誤差条件を破る構成（NOBLOCK）を含む
  # 窓の過剰：II-BF16 の 7040^3 で，互いに素な素数 16 本の積が b=54 の窓の必要量 2*54+1+log2(7040) を
  # 約 0.05・2・4・8・16 bit 上回る集合を与える（本数は 16 のまま．過剰の実際の値は検査側が法から計算する）
  oz_set oz2bf16   # 法の集合は与えるので，法の本数の要求値は既定のまま
  acc_item probe_design_winexcess_7040 "窓の過剰（II-BF16，7040^3，法 16 本）" "$OZ_BIN" \
    'for m in 431,349,337,331,281,269,239,211,163,157,137,131,127,113,109,103 \
              409,337,331,313,281,271,251,211,191,181,179,163,151,149,139,101 \
              463,433,431,409,349,271,257,223,197,181,179,167,149,139,137,127 \
              571,461,433,431,389,337,313,293,283,233,229,211,191,173,149,127 \
              773,733,701,673,659,547,499,457,311,283,271,263,239,233,199,197; do
       sub moduli=$m -- env OZ2_MODULI=$m OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 7040 7040 7040 '"$OZ_PA"' 1 1; done'
  oz_set oz2bfp8
  acc_item probe_design_nsmall_acc "II-BFP8 3520 の逆数の断片と到達精度（1 パス・2 パス）" "$OZ_BIN" \
    'for e in OZ2_NSMALL=0 OZ2_NSMALL=1 OZ2_NSMALL=2 OZ2_NSMALL=3 OZ2_NSMALL=4 OZ2_NSMALL=5 OZ2_NOSMALL=1; do
       sub pass=1 $e -- env $e OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 3520 3520 3520 '"$OZ_PA"' 1 1; done
     sub pass=2 nsmall=default -- env OZ2_FP64=1 OZ2_PASSES=2 OZ2_B=54 OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 3520 3520 3520 '"$OZ_PA"' 1 1
     sub pass=2 OZ2_NOSMALL=1 -- env OZ2_FP64=1 OZ2_PASSES=2 OZ2_B=54 OZ2_NOSMALL=1 OZ_MIN_BITS=1 ./src/build/'"$OZ_BIN"' 3520 3520 3520 '"$OZ_PA"' 1 1' \
    "0 1"   # 断片を増やしすぎた構成（OZ2_NSMALL=5）は 1 で終わる
  oz_set oz2bf16; MT_BF16="./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA"
  oz_set oz2bfp4; MT_BFP4="./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA"
  oz_set oz2bfp2; MT_BFP2="./src/build/$OZ_BIN 14080 14080 14080 $OZ_PA"
  acc_item probe_design_modtable_acc "表の選び方の法の集合の到達精度（14080^3）" "ozaki2_bf16 ozaki2_bfp4 ozaki2_bfp2" \
    "sub fmt=bf16 table=d32 -- env OZ2_MODULI=$MOD_TBL_D32 OZ_MIN_BITS=1 $MT_BF16 1 1
     sub fmt=bfp4 table=b4 -- env OZ2_MODULI=$MOD_TBL_B4 OZ_MIN_BITS=1 $MT_BFP4 1 1
     sub fmt=bfp2 table=b2 -- env OZ2_MODULI=$MOD_TBL_B2 OZ_MIN_BITS=1 $MT_BFP2 1 1"
  IN_DESIGN=0
fi
