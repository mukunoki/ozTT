#!/bin/bash
# measure/fidelity_comparison.sh -- 論文 tab:pass（パス数と GEMM 段）の元ログを 1 項目ずつ冷ましてから測る
# ---------------------------------------------------------------------------
#     ./scripts/measure/fidelity_comparison.sh [サイズ...]       # 既定 3520 7040 14080．ホストから（tt-smi を使う）
#
# 出力は measure_logs/ の次のログで，名前は verify/check_paper_numbers.py が tab:pass に読むもの：
#   pass_LoFi_<N>.log         Ozaki-I-BF16 を OZ_FIDELITY=LoFi で（既定と同じパス数を明示して立てる）
#   pass_HiFi2_<N>.log        同じく OZ_FIDELITY=HiFi2
#   fid_hifi4_<N>.log         同じく OZ_FIDELITY=HiFi4．サイズは HIFI4_SIZES（既定 7040 14080．
#                             論文の表は 3520 を --- にしているので，増やすなら表も直す）
#   gemmfid_<fid>_<N>.log     単体の BF16 GEMM を MathFidelity 4 通りで．引数は cool_bf16x2 と同じ
#                             FP32 累積（gemm_bf16 の fp32_dest_acc=1）．サイズは GEMMFID_SIZES（既定 14080）
# 尾崎の行の引数は measure/performance.sh の Ozaki-I-BF16 と同じ（p=10・L=9・1 3），素の GEMM の行は
# measure/performance.sh の BF16x2/PP と同じ（verify/check_routes.py が突き合わせる）．
#
# OZ_FIDELITY は診断用のつまみ（docs/env_vars.md の C）なので，尾崎の CONFIG 行は probe=set・
# diag=OZ_FIDELITY になる．verify/check_paper_numbers.py は tab:pass に限り，diag が OZ_FIDELITY だけで
# CONFIG の fidelity がログ名と合うログを受け付ける（ほかの探針が立っていれば拒む）．
# このスクリプトは measure/performance.sh と同じく 1 項目ずつ冷まし（届かなければその項目を失敗に数える），段ごとに定常状態まで
# 温めてから測る（measure_lib.sh の OZ_TIMING．既定 steady）．論文の表の
# 注記は測り方を述べているので，測り直した後にこの条件と合っているかを人が確かめる（verify/check_paper_numbers.py は
# 注記の文を照合しない．remeasure_all.sh が最後に表示する「次にすること」の 2）．
# 各ログには PROVENANCE・BINARY・EXIT・COOLCOND・AICLK の各行が残る（measure_lib.sh の run_dram）．
#   ONLY='pass_LoFi|gemmfid' のようにログ名の正規表現を与えると，その項目だけを測る．
set -u
cd "$(dirname "$0")/../.."
SIZES="${*:-3520 7040 14080}"
HIFI4_SIZES="${HIFI4_SIZES:-7040 14080}"
GEMMFID_SIZES="${GEMMFID_SIZES:-14080}"
OUT=measure_logs; mkdir -p "$OUT"
. ./scripts/measure_lib.sh   # wait_cool・run_dram・provenance_line（デバイスの排他もここで取る）
LABEL_W=26
fail=0
run() { run_dram "$@"; }
has() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

for N in $SIZES; do
  echo "===== N=$N  Ozaki-I-BF16 の GEMM 段をパス数別に（1 項目ずつ ${TMAX} degC 以下から）====="
  run "LoFi（1 パス）"  "pass_LoFi_$N"  OZ_FIDELITY=LoFi  -- ./src/build/ozaki1_bf16 $N $N $N 10 9 1 3
  run "HiFi2（2 パス）" "pass_HiFi2_$N" OZ_FIDELITY=HiFi2 -- ./src/build/ozaki1_bf16 $N $N $N 10 9 1 3
  if has "$N" "$HIFI4_SIZES"; then
    run "HiFi4（4 パス）" "fid_hifi4_$N" OZ_FIDELITY=HiFi4 -- ./src/build/ozaki1_bf16 $N $N $N 10 9 1 3
  fi
  if has "$N" "$GEMMFID_SIZES"; then
    echo "===== N=$N  単体の BF16 GEMM（FP32 累積）を MathFidelity 別に ====="
    for fid in LoFi HiFi2 HiFi3 HiFi4; do
      RUN_PAT='Performance : [0-9.]+ TFLOP/s' \
        run "単体 GEMM $fid" "gemmfid_${fid}_$N" OZ_FIDELITY=$fid -- ./src/build/gemm_bf16 $N $N $N 1 1 3 0 0 1 0
    done
  fi
done
echo "(全出力は $OUT/pass_*.log・fid_hifi4_*.log・gemmfid_*.log に残る)"
only_summary
exit $fail
