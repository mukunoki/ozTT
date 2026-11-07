# lib/probes/tile.sh -- diagnose/probes.sh の群 (1)〜(1c) 単コアのタイル積と GEMM 段のコアあたりの構成（性能．冷ましてから）
# ---------------------------------------------------------------------------
# diagnose/probes.sh が群の順に source する（単独では実行しない）．項目の形の部品（want・perf・acc_item・dram_item・
# section・oz_set など）と群の選び方，項目のログ名と論文・スライドの値の対応は diagnose/probes.sh の冒頭
section "(1) 単コアのタイル積（tile_rate_probe．格子 1x1・8x8・K 4・サブブロック 2x2・3 回の起動の平均）"
RUN_PAT='TILE-RATE : .*'
for f in fp32 bf16 fp16 bfp8; do
  for fid in LoFi HiFi2 HiFi3 HiFi4; do
    reps=40000; [ "$fid" = LoFi ] || reps=20000       # 1 回の起動を 0.1〜0.3 s に揃える
    perf "$f $fid 積" "probe_tile_${f}_${fid}_mm" -- ./src/build/tile_rate_probe $f $fid mm $reps
  done
  perf "$f LoFi 展開のみ" "probe_tile_${f}_LoFi_unpack" -- ./src/build/tile_rate_probe $f LoFi unpack 40000
  perf "$f LoFi 演算のみ" "probe_tile_${f}_LoFi_math" -- ./src/build/tile_rate_probe $f LoFi math 40000
done
for f in fp32 bf16; do
  perf "$f LoFi 積 サブブロック 1x1" "probe_tile_${f}_LoFi_mm_sub1x1" -- ./src/build/tile_rate_probe $f LoFi mm 40000 1 1
done
# 尾崎の GEMM 段のコアあたりの構成を 1 コアで再現する（出力ブロック・サブブロック・bw を各経路の 14080^3 の既定に揃える．
# スキーム II は 11x10・1x2，スキーム I は 8x8・2x2．i12 は I の出力ブロックに II のサブブロック 1x2 を当てた比較）．
# 1 回の起動のタイル積を約 1000 万回に揃える．INT8（HiFi4．Int32 の DST）はこの探針が扱わない
section "(1b) 単コアで GEMM 段のコアあたりの構成（tile_rate_probe．LoFi・3 回の起動の平均）"
tilecfg() {   # tilecfg 名前 pm pn sh sw 形式 bw モード…
  local nm=$1 pm=$2 pn=$3 sh=$4 sw=$5 f=$6 bw=$7 mode reps; shift 7
  reps=$(( (10240000 + pm * pn * bw / 2) / (pm * pn * bw) ))
  for mode in "$@"; do
    perf "$nm $f bw=$bw $mode" "probe_tilecfg_${nm}_${f}_${mode}" -- \
      ./src/build/tile_rate_probe $f LoFi $mode $reps $sh $sw 3 $pm $pn $bw
  done
}
# mmpack（展開・積・パック）は全形式を原稿の 1 コアの値として照合が読み，unpack（展開だけ）はスライドの GEMM 段の表に
# 載せる 8 実装の形式だけを測る（残りの形式の unpack と i12 は docs だけが引くので下の (1c)）
for fb in bf16:11 fp16:11 fp32:5 bfp8:20 fp8:22 bfp4:40 bfp2:55; do
  tilecfg ii 11 10 1 2 ${fb%:*} ${fb#*:} mmpack
done
for fb in bf16:11 fp32:5 bfp8:20 bfp2:55; do tilecfg ii 11 10 1 2 ${fb%:*} ${fb#*:} unpack; done
for fb in bf16:11 fp16:11 fp32:5 bfp8:22 fp8:22 bfp4:44 bfp2:55; do
  tilecfg i 8 8 2 2 ${fb%:*} ${fb#*:} mmpack
done
for fb in bf16:11 fp32:5 bfp8:22 bfp4:44; do tilecfg i 8 8 2 2 ${fb%:*} ${fb#*:} unpack; done
# docs/gemm_stage_bottleneck.md（9.1・9.2 節）だけが引く 1 コアの値（PROBES_DOCS の群）
IN_DOCS=1
section "(1c) 単コアの構成のうち docs だけが引く値（tile_rate_probe．展開だけの残りの形式と，I の出力ブロックにサブブロック 1x2）"
for fb in fp16:11 fp8:22 bfp4:40; do tilecfg ii 11 10 1 2 ${fb%:*} ${fb#*:} unpack; done
for fb in fp16:11 fp8:22 bfp2:55; do tilecfg i 8 8 2 2 ${fb%:*} ${fb#*:} unpack; done
for fb in bf16:11 bfp8:22; do tilecfg i12 8 8 1 2 ${fb%:*} ${fb#*:} mmpack; done
IN_DOCS=0
unset RUN_PAT
