#!/bin/bash
# diagnose/gemm_exact.sh -- GEMM 段の K 方向の累積が無誤差かを，敵対的な整数入力で全要素について見る
# ---------------------------------------------------------------------------
#     ./scripts/run.sh "bash /home/user/tt/scripts/diagnose/gemm_exact.sh"
# src/build/gemm_exact_check（尾崎の両スキームと同じ spill_gemm_spec＋build_mcast_gemm＋bmm_int32_spill，
# BF16 extended の l1acc_gemm_spec＋bmm_l1acc）を形式 × 入力の型 × K ブロック数 × 乱数の種 × 構成
# （素・対ループ・融合・付け替え）で回し，int64 の参照と「整数値として」完全一致かを判定する
# （出力語のビット一致ではない．±0 は区別しない）．各実行は同じプログラムを入力を替えて 2 回以上続けて回し
# （融合は acc を 0→1→1→0 と替えて 4 回），前の実行の状態の持ち越しも見る．
# 計算カーネルの発行方法は 2 通り（matmul_block と OZ_NO_BLOCKMM=1 の matmul_tiles）で，既定の側は
# 親の環境の OZ_NO_BLOCKMM を外してから回し，EXACT-CHECK 行の blockmm= が側と合うことも確かめる
# （引き継ぐと，matmul_block を一度も通らずに ALL PASS になりうる）．
# l1acc（bmm_l1acc.cpp）は OZ_NO_BLOCKMM を読まないので blockmm=n/a で，既定の側だけで回す．
# 形式は本番の 8 つ（fp32（TF32）・bf16・fp16・bfp8・fp8・bfp4・bfp2・int8）で，入力の幅は形式の桁の上限．
# fp8s は同じ e4m3 に尾崎 II の FP8 経路の疎な桁（|x| <= 16 か 32 以下の偶数．積は 32^2 = 1024 まで．
# docs/fp8_sparse_digits.md）を入れたもの．
#   rand/pm1  乱数（種 3 つ）      pos/neg  同符号の最大値（FP32 累積では 2^24 の手前まで）
#   cancel    途中で大きく振れて最後に 0 へ戻る
#   odd 系    途中の和がすべて奇数（INT8 の L1 累積が 2^24 を超えて丸めずに足すかを見る．
#             INT8 は 2^31 の手前まで．pos は K ブロックの和が 32*e^2 の倍数なので FP32 の加算でも
#             2^28 近くまで丸まらず，これだけでは Int32 で足していることの証拠にならない）
#   maxodd/maxcancel  桁の上限 d での最大の和（奇数）と大きな振れ．尾崎 II の桁積の連結の上限 maxg の
#             境目（FP8 は K=32736 で maxg=2 → 連結した K=65472，d=32 は K=8160 → 16320）で無誤差であること，
#             その少し先（和が 2^24 を超える K）では FP32 の丸めで食い違うこと（検査の感度）を見る．
#             d が奇数の形式（INT8・BFP4_b・BFP2_b）は k=0 の積を d(d-1) にして和 K d^2 - d を奇数にする
#             （gemm_exact_check の maxodd の注記．odd 系と maxodd・maxcancel は参照の全要素が奇数かも確かめる：oddsum=ok）
#   zface     K ブロックごとに零の面・零の行・零のブロックを混ぜる（パッカの零の面の印と L1 累積）
#   ブロック数 1・2・3・多数（c_24 の輪を何周もする）
#   小さい格子  2x2・3x3・3x4 コア（データ移動の役割の境目．受けるだけのコアの有無）
#   1 ブロックのタイル数  1（後半の送信コアを置かず前半の送信コアが全部を配る），3 と 15（前半と後半で
#             枚数が違い，後半が行の途中から始まる），A だけ 2 枚で B は 1 枚（片側だけ 2 コアから配る）
#   対ループ  1 対あたり 1・2・3 ブロック，複数の対（c_5 の輪の周回），対の和の合計が 2^24 を超えるもの
#   融合      acc=0→1→1→0（直前の出力へ Int32 で足す／上書き），合計が 2^24 を超えるもの
#   付け替え（+re）  同じプログラムで A/B の開始位置・K ブロック数・対の数（1→多数→1）・出力の面を替え，
#             書かない面は番兵のままかも見る（尾崎 I の gemm_run と同じ付け替え）
#   l1acc     gemm_bf16ext の計算カーネル（bmm_l1acc.cpp）．4 ブロック以上・サブブロック 4 つで零の面を混ぜる
#   組み立ての拒否  形が 0・割り切れない（K 方向は Kt より小さく割り切らない bw と Kt を超える bw）・DST に載らない
#             構成（bad<what>），L1 の予算に収まらない構成
#   int8 で対ループ・融合を頼むと止まること（融合・対ループは FP32 の部分積専用）
# 判定は「終了コード 0 かつ PASS 行」．どちらかが欠ければ FAIL（PASS を出したあとの異常終了や，
# 同期の誤りによる停止を見逃さない）．範囲外の入力で食い違うことを見る件（mismatch）は，終了コード 1・
# 範囲外の表示・FAIL に加えて，有限な数値の食い違いがあり（numeric > 0），NaN などの値（nonnumeric）・
# 書き込みの欠落（unwritten）・番兵の破壊（overwritten）・奇数性の崩れ（oddsum）が無いことを要る
# （出力が全部 NaN になる壊れ方を，期待した食い違いと取り違えないため）．
# 1 件ごとに EXACT_TIMEOUT 秒（既定 600）で打ち切る．
# 側は EXACT_SIDES（既定 "blockmm noblockmm"）で選ぶ．値は改行を含めて全体を空白区切りで読み，空の並び・知らない名前・
# 重複は止める（空白だけの値では外のループが一度も回らず，GEMM を 1 件も実行せずに ALL PASS になる．1 行目だけを
# 読むと，改行の後ろの側は検査も実行もされない）．最後に側ごとの GEMM の実行件数を
# 決まった件数（EXPECT_RUNS）と，拒否の件数を REFUSE_RUNS と比べる．件を足したり外したりしたら両方を直すこと．
# 終了コード（scripts/README.md「装置の検査（精査）」の規則）：0 は両側を全件確かめて通った（最後の行 ALL PASS），1 は
# 通らない件がある（SOME FAILED），3 は通らない件は無いが EXACT_SIDES で側を省いた（最後の行 PARTIAL．省いた側の
# GEMM を確かめていないので ALL PASS と出さない）．
# 全出力は EXACT_LOG_DIR（既定 measure_logs/gemm_exact_<日時>）に 1 件 1 ファイルで残す．
set -u
cd "$(dirname "$0")/../.."
# 親の環境の尾崎のつまみ（OZ_*・OZ2_*）を引き継がない（検査したい分岐や既定の構成を変えたまま通る・落ちるのを防ぐ．
# 各件に要る値はその行で与える）
. ./scripts/lib/common.sh; oz_env_clean   # OZ_TELEM_DEV・OZ_IMAGE_ID（run.sh が渡す）は残す
X=./src/build/gemm_exact_check
TMO=${EXACT_TIMEOUT:-600}
LOGD=${EXACT_LOG_DIR:-measure_logs/gemm_exact_$(date +%Y%m%d_%H%M%S)}
# 側ごとの GEMM の実行件数（run の呼び出し数）と拒否の件数．l1acc の 10 件は blockmm の側だけ
declare -A EXPECT_RUNS=([blockmm]=235 [noblockmm]=225)
REFUSE_RUNS=11
# 側の並びの検査（空・知らない名前・重複は，デバイスに触る前に止める）．read -d '' は NUL まで（値の全体）を読み，
# 改行も IFS の空白として区切る（read -r -a だけでは 1 行目で止まる）．NUL が来ないので終了コードは 1 になる
read -r -d '' -a SIDES <<<"${EXACT_SIDES-blockmm noblockmm}" || true
if [ ${#SIDES[@]} -eq 0 ]; then echo "ERROR: EXACT_SIDES が空（blockmm・noblockmm の 1 つ以上）"; exit 2; fi
declare -A seen=()
for side in "${SIDES[@]}"; do
  case $side in blockmm|noblockmm) ;; *) echo "ERROR: EXACT_SIDES の $side は blockmm|noblockmm ではない"; exit 2 ;; esac
  [ -n "${seen[$side]:-}" ] && { echo "ERROR: EXACT_SIDES に $side が 2 回ある"; exit 2; }
  seen[$side]=1
done
mkdir -p "$LOGD"
fail=0; n=0; nref=0
declare -A nrun=() nok=()
# 側（blockmm / noblockmm）ごとの環境．既定の側は親の OZ_NO_BLOCKMM を外す
side_env() { if [ "$1" = noblockmm ]; then echo "OZ_NO_BLOCKMM=1"; else echo "-u OZ_NO_BLOCKMM"; fi; }
side_bm()  { if [ "$1" = noblockmm ]; then echo 0; else echo 1; fi; }
# run 側 期待(pass|mismatch) 引数...   mismatch は「範囲外の入力で食い違う」ことを期待する（検査の感度）
run() {
  local side=$1 want=$2; shift 2
  n=$((n+1)); nrun[$side]=$(( ${nrun[$side]:-0} + 1 ))
  local log; log=$(printf '%s/%04d.log' "$LOGD" "$n")
  local rc envs; envs=$(side_env "$side")
  { echo "CMD: env $envs $X $*"; } >"$log"
  # shellcheck disable=SC2086
  if timeout "$TMO" env $envs $X "$@" >>"$log" 2>&1; then rc=0; else rc=$?; fi
  echo "EXIT,rc=$rc" >>"$log"
  local line; line=$(grep -E '^EXACT-CHECK' "$log" | tail -1)
  local ok=0 bm; bm=$(side_bm "$side")
  case " $* " in *" l1acc "*) bm=n/a ;; esac   # bmm_l1acc.cpp は OZ_NO_BLOCKMM を読まない
  printf '%s' "$line" | grep -q " blockmm=$bm " || line="$line [blockmm が側 $side と合わない]"
  if [ "$want" = pass ]; then
    [ "$rc" -eq 0 ] && printf '%s' "$line" | grep -qE ' PASS$' && ok=1
  else
    # 範囲外で食い違うのは有限な数値としてだけ（ほかの原因が 1 つでもあれば期待した食い違いではない）
    [ "$rc" -eq 1 ] && printf '%s' "$line" | grep -qE 'OUT OF RANGE.* FAIL$' \
      && printf '%s' "$line" | grep -qE ' numeric=[1-9][0-9]* ' \
      && printf '%s' "$line" | grep -qE ' nonnumeric=0 unwritten=0 overwritten=0 ' \
      && printf '%s' "$line" | grep -qE ' oddsum=(ok|n/a) ' && ok=1
  fi
  if [ $ok = 1 ]; then
    nok[$side]=$(( ${nok[$side]:-0} + 1 ))
    printf '  %-9s %-8s %s\n' "$side" "$want" "$(echo "$line" | sed 's/EXACT-CHECK //')"
  else
    local why="exit $rc"; [ "$rc" -eq 124 ] && why="timeout ${TMO}s"
    printf '  %-9s FAIL (%s，%s を期待): %s\n' "$side" "$why" "$want" "$*"
    grep -E '^(EXACT-CHECK|MISMATCH|NONNUMERIC|UNWRITTEN|OVERWRITE|PATTERN-PROPERTY|ERROR)' "$log" | head -8 | sed 's/^/      /'
    [ -z "$line" ] && tail -5 "$log" | sed 's/^/      | /'
    echo "      （全出力 $log）"
    fail=1
  fi
}
# 止まるべき実行が，終了コード 2 と決まった文言で止まるか
refuse() {   # refuse 文言 [env...] -- 引数...
  local msg=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  n=$((n+1)); nref=$((nref+1))
  local log; log=$(printf '%s/%04d.log' "$LOGD" "$n")
  local rc
  if timeout "$TMO" env "${envs[@]}" $X "$@" >"$log" 2>&1; then rc=0; else rc=$?; fi
  echo "EXIT,rc=$rc" >>"$log"
  if [ "$rc" -eq 2 ] && grep -qF "$msg" "$log"; then
    printf '  %-9s refused as expected: %s %s\n' "" "${envs[*]}" "$*"
  else
    printf '  %-9s FAIL (exit %s．「%s」で止まるはず): %s %s\n' "" "$rc" "$msg" "${envs[*]}" "$*"; fail=1
  fi
}

for side in "${SIDES[@]}"; do
  echo "== $side =="
  for f in fp32 bf16 bfp8 int8; do
    # 同符号の最大値：FP32 累積の形式は K=16352（31^2*K = 15,714,272 = 2^24 - 1.06e6），INT8 も同じ K=16352
    # （127^2*K = 263,741,408 = 2^27.97）．
    # 2 回目の実行は B の符号を反転する（負側の最大値）
    run $side pass 1280 1408 16352 $f pos 0 1
    run $side pass 1280 1408 16384 $f cancel 0 1
    # 途中の和がすべて奇数（FP32 系は 30*31*K + 31^2 が 2^24 の手前，INT8 は 2^27.97）
    run $side pass 1280 1408 16352 $f odd 0 1
    run $side pass 1280 1408 16352 $f oddcancel 0 1
    for s in 1 2 3; do run $side pass 1280 1408 8192 $f rand 0 $s; done
    run $side pass 1280 1408 8192 $f pm1 0 1
    # ブロック数 1・2・3・12（Kt=12）
    for bw in 12 6 4 1; do run $side pass 640 704 384 $f pm1 $bw 5; done
    # 零の面・零の行・零のブロック（16 ブロック・サブブロック 4 つ）
    run $side pass 1280 1408 1024 $f zface 2 11
    # 小さい格子（2x2・3x4 コア．両端の送信コアが隣り合う／受信だけのコアが無い行・列）
    run $side pass 64 64 512 $f pm1 2 7
    run $side pass 96 128 512 $f rand 4 8
    # 1 タイルのブロック（2x2 コアで K ブロック 1 つ，2x3 コアで 16 個）
    run $side pass 64 64 32 $f pm1 0 7
    run $side pass 64 96 512 $f pm1 1 7
    # 前半と後半の枚数が違うブロック．3 枚（2+1）：2x2 コアで 6 ブロック，3x3 コア（受けるだけのコアあり．
    # B の後半は h=2 から）で 1 ブロック．15 枚（8+7．A の後半は行の途中 w=3 から，B は h=2,w=2 から）
    run $side pass 64 64 576 $f pm1 3 7
    run $side pass 96 96 96 $f pm1 3 7
    run $side pass 960 1056 1280 $f pm1 5 9
    # A のブロックは 2 枚（両端の 2 コアから配る）で B は 1 枚（前半の送信コアが全部を配る）
    run $side pass 640 352 128 $f pm1 1 7
    # 同じプログラムの付け替え（A/B の開始位置・K ブロック数 1→16→1→8→1・出力の面．書かない面は番兵のまま）
    run $side pass 1280 1408 1024 $f pm1 2 3 plain+re
    if [ "$f" = int8 ]; then
      # Int32 の L1 累積を 2^31 の手前まで（尾崎 I の INT8 は対角 7 で 2^30.8，II は K に応じて 2^31 の手前まで
      # 許す）．64x64 は 2x2 コア・コアあたり 1 タイルで，K ブロックは既定の bw で 13 個，bw=45 で 91 個．
      run $side pass 64 64 131040 int8 pos 0 1           # 2,113,544,160（2^30.98）．2 回目は負側
      run $side pass 64 64 133120 int8 pos 0 1           # 2,147,092,480（2^31 - 3.9e5）
      run $side pass 64 64 131040 int8 odd 0 1           # 途中の和がすべて奇数で 2^30.97 まで
      run $side pass 64 64 131040 int8 oddneg 0 1
      run $side pass 64 64 131040 int8 odd 45 1
      run $side pass 64 64 131040 int8 cancel 0 1        # 2^30 まで振れて 0 へ戻る
      run $side pass 64 64 131040 int8 oddcancel 0 1     # 同じく奇数のまま（大きい和へ逆符号のブロック和）
      run $side pass 640 704 32768 int8 odd 0 1          # 尾崎 II の INT8 の K=32768（2^28.97．10x11 コア・2x2）
      run $side pass 64 64 133120 int8 maxodd 0 1        # 桁の上限 127 での奇数の和 133120*127^2-127 = 2,147,092,353
      run $side pass 64 64 133120 int8 maxcancel 0 1     # 2^30 まで振れて奇数の -127 / 2*127^2-127 へ戻る
      run $side pass 640 704 384 int8 pm1 4 5 plain+sep  # c_16 と c_24 を別々に（gemm_int8 と同じ）
      continue
    fi
    # 対ループ（尾崎 I の既定．FP32 の部分積の形式だけ）．1280x1408 は 10x11 コア・コアあたり 4x4
    # （サブブロック 2x2 が 4 つ）で，c_5 は 20 枚の輪なので対ごとに位置がずれて何周もする．
    run $side pass 1280 1408 960  $f pm1 2 3 pl3         # 1 対 3 ブロック × 5 対
    run $side pass 1280 1408 1024 $f rand 4 5 pl2        # 1 対 2 ブロック × 4 対
    run $side pass 1280 1408 1536 $f pm1 8 7 pl1         # 1 対 1 ブロック × 6 対（c_24 を使わない）
    run $side pass 96 128 512 $f pm1 4 8 pl2             # 3x4 コア
    run $side pass 96 96 576 $f pm1 3 7 pl3              # 3x3 コア・3 枚のブロック × 2 対
    # 1 対の和は 2^24 未満だが全体は 2^24 を超える（対の和は Int32 で足す）．2x2 コア・8 対
    run $side pass 64 64 65536 $f pos 32 1 pl8           # 全体 62,980,096（2^25.9）
    run $side pass 64 64 65536 $f odd 32 1 pl8
    # 対ループの付け替え：対の数 1（最初の対がそのまま出力：first_pair && last_out）→ 8 → 1 → 4 → 1
    run $side pass 1280 1408 1024 $f pm1 2 3 pl2+re
    run $side pass 1280 1408 1024 $f zface 2 3 pl1+re
    # 融合（DRAM の既存の出力へ Int32 で足す．acc=0,1,1,0）
    run $side pass 1280 1408 1024 $f rand 4 5 fuse       # 8 ブロック
    run $side pass 1280 1408 16352 $f pos 0 1 fuse       # 1 回の和は 2^24 の手前，3 回ぶんで 2^25.5
    run $side pass 64 64 32 $f pm1 0 7 fuse              # 1 ブロック 1 タイル
    run $side pass 96 96 96 $f pm1 3 7 fuse              # 3x3 コア・3 枚のブロック
    run $side pass 1280 1408 1024 $f rand 2 5 fuse+re    # 融合の付け替え（面ごとに acc=0 → 1）
  done
  # 本番のほかの形式（FP16 は binary16 のビット列，FP8 は e4m3，BFP4_b・BFP2_b は共有指数）．入力は形式の桁の上限．
  for f in fp16 fp8 fp8s bfp4 bfp2; do
    for s in 1 2; do run $side pass 1280 1408 4096 $f rand 0 $s; done
    run $side pass 1280 1408 4096 $f pm1 0 3
    run $side pass 1280 1408 4096 $f cancel 0 1
    run $side pass 1280 1408 4096 $f odd 0 1
    for bw in 12 4 1; do run $side pass 640 704 384 $f pm1 $bw 5; done
    run $side pass 1280 1408 1024 $f zface 2 11
    run $side pass 64 64 512 $f pm1 2 7
    run $side pass 960 1056 1280 $f pm1 5 9
    run $side pass 1280 1408 1024 $f pm1 2 3 pl2+re
    run $side pass 1280 1408 1024 $f rand 4 5 fuse
  done
  # 同符号の最大値を形式ごとの 2^24 の手前まで（FP16 は 31^2，FP8 は 15^2，BFP4_b は 7^2，BFP2_b は 1）
  run $side pass 1280 1408 16352 fp16 pos 0 1
  run $side pass 64 64 65472 fp8 pos 0 1                 # 65472*225 = 14,731,200
  run $side pass 64 64 327680 bfp4 pos 0 1               # 327680*49 = 16,056,320
  run $side pass 64 64 327680 bfp2 pos 0 1               # 327680（BFP2_b の桁は ±1．長い K の累積）
  # d が奇数の形式の桁の上限での奇数の和（k=0 の積を d(d-1) に替える）．BFP4_b は 327680*49-7 = 16,056,313，
  # BFP2_b は 327679．その少し先（BFP4_b の K=344064 で 16,859,129 > 2^24）では FP32 の丸めで食い違う
  run $side pass 64 64 327680 bfp4 maxodd 0 1
  run $side pass 64 64 327680 bfp4 maxcancel 0 1
  run $side pass 64 64 327680 bfp2 maxodd 0 1
  run $side mismatch 64 64 344064 bfp4 maxodd 0 1
  # 尾崎 II の桁積の連結の境目：連結した K = maxg*K_c で桁の上限の最大の和（maxodd は奇数，maxcancel は大きく振れる）
  run $side pass 64 64 65472 fp8 maxodd 0 1              # FP8 K_c=32736（maxg=2）→ (65471)*256+225 = 16,760,801
  run $side pass 64 64 65472 fp8 maxcancel 0 1
  run $side pass 64 64 16320 bf16 maxodd 0 1             # d=32 K_c=8160（maxg=2）→ 16319*1024+961 = 16,711,617
  run $side pass 64 64 16320 bf16 maxcancel 0 1
  # FP8 の疎な桁（桁 32 を e4m3 で）：連結しない最大の K=16352（法 2401（=49^2）の桁積 1 本）と，K=8160 の 2401 の
  # 連結した 2 本（K=16320）で，桁の上限の最大の和．乱数（疎な集合の 49 通り）も同じ K で
  run $side pass 64 64 16352 fp8s maxodd 0 1             # 16351*1024+225 = 16,743,649
  run $side pass 64 64 16352 fp8s maxcancel 0 1
  run $side pass 64 64 16320 fp8s maxodd 0 1             # 16319*1024+225 = 16,710,881
  run $side pass 1280 1408 16352 fp8s rand 0 1
  # その少し先では和が 2^24 を超えて FP32 の丸めで食い違う（検査がこの境目を見分けられること）
  run $side mismatch 64 64 65568 fp8 maxodd 0 1          # 16,785,377 > 2^24
  run $side mismatch 64 64 16416 bf16 maxodd 0 1         # 16,809,921 > 2^24
  run $side mismatch 64 64 16416 fp8s maxodd 0 1         # 16415*1024+225 = 16,809,185 > 2^24
  if [ "$side" = blockmm ]; then
    # BF16 extended・3xTF32 の計算カーネル（bmm_l1acc.cpp．OZ_NO_BLOCKMM は読まないので片側だけ）．
    # 16 ブロック・4 ブロック，サブブロック 4 つ，零の面・零のブロックを混ぜる
    for f in bf16 fp32; do
      run $side pass 1280 1408 1024 $f zface 2 11 l1acc
      run $side pass 1280 1408 512 $f zface 4 13 l1acc
      run $side pass 1280 1408 1024 $f pm1 2 3 l1acc
      run $side pass 1280 1408 8192 $f pos 0 1 l1acc
      run $side pass 640 704 384 $f rand 1 5 l1acc
    done
  fi
done
echo "== 組み立ての拒否 =="
# 融合・対ループは FP32 の部分積専用で，int8 では組めない
refuse "int8 では融合・対ループを組めない" -- 64 64 512 int8 pm1 2 7 pl2
refuse "int8 では融合・対ループを組めない" -- 64 64 512 int8 pm1 2 7 fuse
# 形が 0・割り切れない・DST に載らない構成は，引数を作る前（spill_gemm_spec）か組み立て（build_mcast_gemm）で止まる．
# REFUSED は止まるべき検査の文言で止まったときだけ出る（別の検査で止まれば REFUSED-ELSEWHERE で終了コード 1）．
# 1280x1408 は 10x11 コア・コアあたり 4x4 で，baddst のサブブロック 4x4 は DST の 4 枚を超える．
# K=1024 は Kt=32 で，badkdiv は bw=3（Kt >= bw は満たし，Kt % bw == 0 だけに掛かる），badkbig は bw=33
for b in badsh0 badbw0 badshdiv badkdiv badkbig baddst; do refuse "REFUSED mode=$b" -- 1280 1408 1024 bf16 pm1 2 3 $b; done
# コアあたり 1x1（64x64）では DST に載らない構成を作れない．REFUSED を出さず引数の誤りとして止まること
refuse "baddst にはコアあたりの出力" -- 64 64 32 bf16 pm1 0 7 baddst
# L1 の予算に収まる構成が無い（予算から固定の見込みを引く方式では符号なしの引き算が下へ溢れる）
refuse "構成が取れない" OZ_L1_BUDGET_KB=1 -- 1280 1408 1024 bf16 pm1 2 3 pl1
refuse "構成が取れない" OZ_L1_BUDGET_KB=1 -- 1280 1408 1024 bf16 pm1 2 3
# 実行した件数が決まった件数と合うか（ループの取り違えや件の抜けで，少ない件数のまま ALL PASS にしない）
echo "== 件数 =="
for side in "${SIDES[@]}"; do
  if [ "${nrun[$side]:-0}" -eq "${EXPECT_RUNS[$side]}" ]; then
    printf '  %-9s GEMM %s 件（決まった件数 %s．期待どおり %s 件）\n' "$side" "${nrun[$side]:-0}" "${EXPECT_RUNS[$side]}" "${nok[$side]:-0}"
  else
    printf '  %-9s FAIL GEMM %s 件（%s 件のはず）\n' "$side" "${nrun[$side]:-0}" "${EXPECT_RUNS[$side]}"; fail=1
  fi
done
if [ "$nref" -eq "$REFUSE_RUNS" ]; then printf '  %-9s 拒否 %s 件\n' "" "$nref"
else printf '  %-9s FAIL 拒否 %s 件（%s 件のはず）\n' "" "$nref" "$REFUSE_RUNS"; fail=1; fi
echo "($n 件．全出力は $LOGD)"
# 省いた側（EXACT_SIDES に無い側）．全部を確かめたときだけ ALL PASS と出す
omitted=""
for side in blockmm noblockmm; do [ -n "${seen[$side]:-}" ] || omitted="$omitted${omitted:+・}$side"; done
if [ $fail -ne 0 ]; then echo "SOME FAILED"; exit 1
elif [ -n "$omitted" ]; then echo "PARTIAL（EXACT_SIDES で側 $omitted を省いた．通らない件は無いが，省いた側は確かめていない）"; exit 3
fi
echo "ALL PASS"
exit 0
