#!/bin/bash
# measure/cpu_dgemm_reference.sh -- ネイティブの FP64 DGEMM（ホストの CPU の OpenBLAS）の到達精度を，尾崎の実装と同じ入力・参照解・指標で測る
# ---------------------------------------------------------------------------
#     DGEMMREF_CORETYPE=ZEN ./scripts/measure/cpu_dgemm_reference.sh                          # 表の DGEMM → accuracy_logs/dgemm_ref.log
#     DGEMMREF_CORETYPE=ZEN DGEMMREF_SET=phi14080 ./scripts/measure/cpu_dgemm_reference.sh    # φ の DGEMM → measure_logs/probe_dgemm_phi_14080.log
# 出力先は測る点の組（DGEMMREF_SET）で決まり（下の DGEMMREF_OUT），同じものを画面にも出す（tee で別のファイルへ
# 書かなくてよい．前の版の使い方 `| tee accuracy_logs/dgemm_ref.log` のまま phi14080 を流すと表の DGEMM のログを
# φ の 5 点で潰した）．
# 道具は tools/dgemmcheck.cpp（入力 gen_operands・参照解 reference_gemm（Dot2）・指標 accuracy はどれも
# src/common/host_eval.hpp が取り込むもの（oz_input.hpp・oz_reference.hpp・oz_accuracy.hpp）で，尾崎の実装の
# 「effective ~X bit」と同じ）．OpenBLAS は tt-metal のイメージに無いので，
# scripts/Dockerfile.dgemmref のイメージ（DGEMMREF_IMAGE）を使い，無ければ作る．コンパイルするソース（host_eval.hpp の
# ディレクトリ（今は src/common）と dgemmcheck.cpp）は，出どころの行に記録する rev のコミットの中身を git archive で固めて
# コンテナの標準入力へ渡し，
# コンテナの中の /tmp で展開・コンパイルして走らせる（作業木をマウントしてコンパイルすると，出どころを取ってから
# コンパイルするまでのあいだに作業木を直したとき，記録した rev・host_eval・telemetry と実行ファイルが食い違う）．
# dirty が 0 でないとき（試し用．検査はそのログを拒む）だけは作業木の今のファイルを固めて渡す．デバイスは使わない．
# ソースの置き場所は固定のパスでなく，測る rev の木（dirty なら作業木）で名前から引く（lib/revtree.sh の rev_path．src を
# 並べ替えた rev でも黙って読み損なわない）．取り込みの根（-I）は dgemmcheck.cpp が host_eval.hpp を取り込む書き方
# （#include "common/host_eval.hpp"）から決める．コンパイルのフラグは src/Makefile のホストの道具と同じもの（HOST_OPT．
# フラグの正本を Makefile の 1 か所にする）で，デバイスの排他と rev を決めた後に，その rev の木の src/Makefile（dirty なら
# 作業木のもの）をホストの make に渡して print-host-flags を読む（排他を待つあいだのコミットで，記録する rev と別の Makefile の
# フラグにならないように）．
#
# 測る点（GEMMul8 の比べ方：構成を固定した尾崎の到達精度を，同じ入力に対するネイティブの DGEMM と並べる）：
#   OZ_PHI=φ（φ = 0, 0.5, 1, 2, 4, 8, 16）の 4096^3   accuracy_logs/phi_sweep.log・phi_sweep_05.log と同じ大きさ・入力
#   OZ_PHI=0 の 3520^3・7040^3・14080^3
#   既定の一様乱数（OZ_PHI なし）の 3520^3・7040^3・14080^3   表（tab:flops）の到達精度と同じ入力（参考）
# 14080^3 は Dot2 の参照に 16 スレッドで約 110 s かかり，全体で約 6 分（Ryzen 7 9700X）．行列 5 枚（A・B・C・Chi・Clo）で約 8 GB を使う．
#
# 出力：1 行目は出どころの PROVENANCE 行，最後の行は EXIT,rc=<失敗した点の数>（0 なら全点が正常に終わった）．
#   PROVENANCE,rev=…,dirty=…,date=…,image=…,dockerfile=…,coretype=…,cflags=…,openblas=…,blas_config=…,blas_core=…,blas_threads=…,
#              blas_parallel=…,omp_threads=…,cxx=…,libc=…,libstdcxx=…,libgcc=…,libgomp=…,host_eval=…,telemetry=…,oz_input=…,
#              oz_reference=…,oz_accuracy=…,host=…,cpu=…,kmd=…,dev=…,board=…,…（prov_machine の欄）
#     rev・dirty・date   ホストの git で scripts/provenance.sh と同じ規則（dirty は src/・scripts/・tools/ の未コミットの変更）
#     image              このイメージの ID の先頭 12 桁
#     dockerfile         このイメージを作った scripts/Dockerfile.dgemmref の sha256 の先頭 12 桁（作るときに付けたラベル
#                        ttgemm.dockerfile．ラベルの無い前の版のイメージは unknown）
#     coretype           DGEMMREF_CORETYPE（OpenBLAS の OPENBLAS_CORETYPE）．与えなければ auto（OpenBLAS が CPU から選ぶ）
#     cflags             コンパイルのフラグ（上の print-host-flags の 1 行．末尾の改行を除く）の sha256 の先頭 12 桁
#     openblas           libopenblas0-pthread の版（dpkg），blas_*・omp_threads は dgemmcheck info の行（OpenBLAS の
#                        openblas_get_config・openblas_get_corename・スレッド数，Dot2 の OpenMP のスレッド数）
#     cxx                g++ の版（dpkg）．入力の乱数は libstdc++ のヘッダで決まるので測定のイメージと同じ 12.3 にしてある
#     libc・libstdcxx・libgcc・libgomp
#                        libc6・libstdc++6・libgcc-s1・libgomp1 の版（dpkg）．入力の exp は glibc で，乱数と Dot2 の OpenMP は
#                        これらの実行時のライブラリで決まる（Dockerfile.dgemmref は依存まで版で固定するが，与えたイメージ
#                        DGEMMREF_IMAGE やタグの付け替えでも食い違いが記録に残るように）
#     host_eval・telemetry・oz_input・oz_reference・oz_accuracy
#                        src/common/host_eval.hpp（入力生成・参照解・指標のヘッダを取り込む傘）・telemetry.hpp と，定義の本体を持つ
#                        傘の下の 3 本の sha256 の先頭 12 桁（照合は加えて下の定義の比べで見る）
#     host 以降          provenance.sh の prov_machine（測った計算機とボード．論文の値と同じ計算機で測ったことの記録）
#   各点は見出し「=== dgemmref: M=… N=… K=… input=…」，dgemmcheck の出力（Reference: Dot2 …・Accuracy : …・CHECKSUM,…・
#   DGEMMREF,…．形式は tools/dgemmcheck.cpp），終了コードの行 RC,<n> の順．DGEMMREF 行の bits が到達精度［bit］．
#
# 出どころの規則（scripts/README.md「φ の掃引と DGEMM 参照」）：このログは尾崎のログ（表の rev）と別の rev で
# 測ってよい（道具を足したコミットより後．rev は揃えなくてよい）．入力・参照解・指標は host_eval.hpp と，それが取り込む
# ヘッダの定義で決まるので，verify/check_paper_numbers.py（probe_checks.py の dgemm_log）は，このログの rev と表のログの rev
# とで入力生成・参照解・指標の定義（lib/peer_logs.py の DGEMM_ROOTS の gen_operands・reference_gemm・accuracy と，その本体が
# 名前で使う取り込みの連鎖の中の定義．コメントを除いて比べる．本体は違うが同じ値を返すと確かめた版は DGEMM_SAME に理由とともに
# 書く）が同じことを要る（ファイル全体の一致は要らないので，コメントや計時だけを変えた rev の表とも組める．定義が
# 変わったら測り直す）．あわせて，ログに記録したソースの sha256（host_eval・telemetry・傘の下の 3 本）がログ自身の rev のファイルと同じ，
# dirty=0・allow_dirty なし・INVALIDATED 行なし，g++ が測定のイメージと同じ 12.3，最後の行が EXIT,rc=0，全点がそろう，
# ことを要る．したがって正式な測定は，道具をコミットしてから作業木を汚さずに流す．
#
# 環境変数
#   DGEMMREF_IMAGE     イメージの名前（既定 localhost/ttgemm-dgemmref:df-<Dockerfile.dgemmref の sha256 の先頭 12 桁>．
#                      Dockerfile を直せばタグが変わって作り直す（同じタグの古いイメージを使い続けない））
#   DGEMMREF_THREADS   OpenBLAS と OpenMP（Dot2）のスレッド数（既定は与えない＝どちらも論理 CPU の数）
#   DGEMMREF_SET       測る点の組（既定 paper＝上の「測る点」．phi14080 は 14080^3 の φ=0.5,1,2,4,8 で，
#                      measure/exponent_adaptive.sh（と diagnose/probes.sh の probe_phiacc_*・probe_phiperf_*）の比較の相手．
#                      表の DGEMM と同じカーネルで測るため DGEMMREF_CORETYPE=ZEN を付けて流す（verify/check_paper_numbers.py の
#                      phin_dgemm も probe_checks.py の表の DGEMM も FMA のカーネルを要る））
#   DGEMMREF_OUT       出力のログ（既定は組で決める：paper は accuracy_logs/dgemm_ref.log，phi14080 は
#                      measure_logs/probe_dgemm_phi_14080.log．- なら画面にだけ出す）．一時ファイル（.<名前>.tmp.<pid>）に
#                      書いてから置き換える．置き換えるのは，一時ファイルへの書き込み（tee）とソースを固める処理（git archive・
#                      tar）がどちらも成功し，コンテナが最後まで走り（終了コード 0 か 1．1 は失敗した点がある記録），1 行目が
#                      PROVENANCE 行で最後の行が EXIT,rc= のときだけ．それ以外（コンパイルの前などで止まった，途中で止められた，
#                      書き込みに失敗した）は前のログを残し，書きかけを <ログ>.incomplete に置いて非零で終わる
#   DGEMMREF_NO_LOCK=1 デバイスの排他（device_lock.sh）を取らない．既定では取る：デバイスは使わないが全コアを使うので，
#                      ホストの CPU で投入・参照解を計算する別の測定と重ねないため
#   DGEMMREF_CORETYPE  OpenBLAS の DGEMM のカーネルを固定する（OPENBLAS_CORETYPE．既定は与えない＝OpenBLAS が CPU から選ぶ）．
#                      Ubuntu 22.04 の OpenBLAS 0.3.20 は Ryzen 7 9700X（Zen 5）を知らず Barcelona（SSE2．FMA なし）の
#                      カーネルを選ぶ（blas_core=Barcelona）．Haswell・Zen（AVX2＋FMA）・SkylakeX（AVX-512）も動き，到達精度は
#                      4096^3・φ=0 で 50.76（Barcelona）・50.68（Haswell・Zen）・50.93（SkylakeX）bit．Cooperlake は誤った C を返す
#                      （到達精度 -0.5 bit．dgemmcheck が失敗にする）．どれでもスレッド数を変えても C はビット単位で同じ
set -u
cd "$(dirname "$0")/../.."
REPO=$(pwd)
. ./scripts/provenance.sh
DF12=$(sha256sum scripts/Dockerfile.dgemmref 2>/dev/null | cut -c1-12)
case $DF12 in *[!0-9a-f]*|'') echo "ERROR: scripts/Dockerfile.dgemmref の sha256 を取れない" >&2; exit 2 ;; esac
IMG=${DGEMMREF_IMAGE:-localhost/ttgemm-dgemmref:df-$DF12}

if ! podman image exists "$IMG"; then
  echo "イメージ $IMG が無いので作る（scripts/Dockerfile.dgemmref）" >&2
  podman build --label "ttgemm.dockerfile=$DF12" -t "$IMG" -f scripts/Dockerfile.dgemmref scripts >&2 \
    || { echo "ERROR: イメージを作れない（Dockerfile.dgemmref で固定した版がアーカイブに無いなら，版を確かめてから直す）" >&2; exit 2; }
fi
# 走らせるのは，ここで読んだイメージの ID そのもの（タグ $IMG で走らせると，排他を待つあいだに別の実行がタグを付け替えたとき，
# 記録した ID と違うイメージで測る）
img=$(podman image inspect --format '{{.Id}}' "$IMG" 2>/dev/null | head -1)
img=${img#sha256:}
case $img in *[!0-9a-f]*|'') echo "ERROR: イメージ $IMG の ID が読めない（${img:-空}）" >&2; exit 2 ;; esac
img_df=$(podman image inspect --format '{{index .Labels "ttgemm.dockerfile"}}' "$img" 2>/dev/null | head -1)
case $img_df in *[!0-9a-f]*|'') img_df=unknown ;; esac

# つまみの検査（排他を取る前に）
SET=${DGEMMREF_SET:-paper}
case $SET in
  paper) out_def=accuracy_logs/dgemm_ref.log ;;
  phi14080) out_def=measure_logs/probe_dgemm_phi_14080.log ;;
  *) echo "ERROR: DGEMMREF_SET=$SET は paper か phi14080" >&2; exit 2 ;;
esac
OUT=${DGEMMREF_OUT:-$out_def}
thr=()
case "${DGEMMREF_CORETYPE:-}" in
  ('') coretype=auto ;;
  (*[!A-Za-z0-9]*) echo "ERROR: DGEMMREF_CORETYPE=$DGEMMREF_CORETYPE は OpenBLAS のコアの名前（英数字）でない" >&2; exit 2 ;;
  (*) coretype=$DGEMMREF_CORETYPE; thr+=(-e "OPENBLAS_CORETYPE=$DGEMMREF_CORETYPE") ;;
esac
if [ -n "${DGEMMREF_THREADS:-}" ]; then
  case "$DGEMMREF_THREADS" in (*[!0-9]*|0) echo "ERROR: DGEMMREF_THREADS=$DGEMMREF_THREADS は正の整数でない" >&2; exit 2 ;; esac
  thr+=(-e "OPENBLAS_NUM_THREADS=$DGEMMREF_THREADS" -e "OMP_NUM_THREADS=$DGEMMREF_THREADS")
fi

# コンパイルのフラグは排他と rev の後に読む（冒頭の注記）．make が無ければ排他を取る前に止める
command -v make >/dev/null 2>&1 || { echo "ERROR: ホストに make が無い（コンパイルのフラグを src/Makefile の print-host-flags から読む）" >&2; exit 2; }

if [ "${DGEMMREF_NO_LOCK:-0}" != 1 ]; then
  . ./scripts/device_lock.sh
  device_lock || exit $?
fi

# 出どころの行のうちホストで決まる欄（コンテナの中で OpenBLAS とコンパイラの欄を足して 1 行目に出す）
rev=$(prov_try git -C "$REPO" rev-parse HEAD) || rev=unknown
if [ "$rev" = unknown ]; then dirty=unknown; else dirty=$(prov_dirty "$REPO"); fi
[ "$dirty" = 0 ] || echo "注意: dirty=$dirty（src/・scripts/・tools/ に未コミットの変更がある）．このログは検査に拒まれる（試し用）" >&2
# コンパイルするソース（冒頭の注記）．dirty=0 なら rev のコミットの中身を固め，記録するソースの sha256 も
# そのコミットの中身から取る（記録・コンパイル・検査の比べる相手がどれも同じ rev の中身になる）
. ./scripts/lib/revtree.sh   # rev_path
TREE=$rev; [ "$dirty" = 0 ] || TREE=-
MK=$(rev_path "$TREE" Makefile src) || exit 2         # コンパイルのフラグ（冒頭の注記）
HE=$(rev_path "$TREE" host_eval.hpp src) || exit 2
TE=$(rev_path "$TREE" telemetry.hpp src) || exit 2
OI=$(rev_path "$TREE" oz_input.hpp src) || exit 2      # 入力生成（傘の下）
ORF=$(rev_path "$TREE" oz_reference.hpp src) || exit 2 # 参照解（同）
OA=$(rev_path "$TREE" oz_accuracy.hpp src) || exit 2   # 指標（同）
DC=$(rev_path "$TREE" dgemmcheck.cpp tools) || exit 2
if [ "$dirty" = 0 ]; then
  src_tar() { git -C "$REPO" archive --format=tar "$rev" -- $SRC_PATHS; }
  src_cat() { git -C "$REPO" show "$rev:$1"; }
  sha12() { git -C "$REPO" cat-file -e "$rev:$1" 2>/dev/null && git -C "$REPO" show "$rev:$1" | sha256sum | cut -c1-12; }
else
  src_tar() { tar -C "$REPO" -cf - $SRC_PATHS; }
  src_cat() { cat "$REPO/$1"; }
  sha12() { sha256sum "$REPO/$1" 2>/dev/null | cut -c1-12; }
fi
# コンパイルのフラグ：測る rev の src/Makefile（dirty なら作業木のもの）を make に渡して print-host-flags を読む（冒頭の注記）
DGC_FLAGS=$(src_cat "$MK" | make -s -C "$(dirname "$MK")" -f - print-host-flags 2>/dev/null) || DGC_FLAGS=""
case $DGC_FLAGS in
  ''|*[!A-Za-z0-9=+._\ -]*) echo "ERROR: rev ${rev:0:12} の $MK の print-host-flags でコンパイルのフラグを読めない（${DGC_FLAGS:-空}）" >&2; exit 2 ;;
esac
CF12=$(printf '%s' "$DGC_FLAGS" | sha256sum | cut -c1-12)
# 取り込みの根：dgemmcheck.cpp の #include "<書き方>" が host_eval.hpp のパスの末尾になる（"common/host_eval.hpp" なら src）
inc=$(src_cat "$DC" | sed -n 's/^#include "\([^"]*host_eval\.hpp\)".*/\1/p' | head -1)
case $HE in
  */"$inc") DGC_INC=${HE%/"$inc"} ;;
  *) echo "ERROR: $DC の host_eval.hpp の取り込み（${inc:-無い}）が rev ${rev:0:12} の $HE と合わない" >&2; exit 2 ;;
esac
# 固めるのは host_eval.hpp のディレクトリ（同じ場所から相対で取り込む telemetry.hpp などを含む）と dgemmcheck.cpp
SRC_PATHS="$(dirname "$HE") $DC"
[ "$(dirname "$TE")" = "$(dirname "$HE")" ] || SRC_PATHS="$SRC_PATHS $(dirname "$TE")"
tail_f="host_eval=$(sha12 "$HE"),telemetry=$(sha12 "$TE"),oz_input=$(sha12 "$OI"),oz_reference=$(sha12 "$ORF"),oz_accuracy=$(sha12 "$OA"),$(prov_machine)"

# コンテナの中：標準入力のソースを展開 → コンパイル → 出どころの行 → 各点．コンパイルの出力は標準エラーへ（ログを汚さない）
read -r -d '' INNER <<'EOS'
set -u
mkdir -p /tmp/work && tar -xf - -C /tmp/work || { echo "ERROR: 渡したソースを展開できない" >&2; exit 2; }
g++-12 $DGC_FLAGS -I"/tmp/work/$DGC_INC" "/tmp/work/$DGC_SRC" \
  -o /tmp/dgemmcheck -lopenblas >&2 || { echo "ERROR: $DGC_SRC をコンパイルできない" >&2; exit 2; }
info=$(/tmp/dgemmcheck info) || { echo "ERROR: dgemmcheck info が失敗した" >&2; exit 2; }
case $info in BLASINFO,*) ;; *) echo "ERROR: dgemmcheck info の行が読めない: $info" >&2; exit 2 ;; esac
ver() { local v; v=$(dpkg-query -W -f='${Version}' "$1" 2>/dev/null | tr -c 'A-Za-z0-9._:+\n-' _); printf '%s' "${v:-unknown}"; }
printf '%s,openblas=%s,%s,cxx=%s,libc=%s,libstdcxx=%s,libgcc=%s,libgomp=%s,%s\n' "$HEAD_F" "$(ver libopenblas0-pthread)" \
  "${info#BLASINFO,}" "$(ver g++-12)" "$(ver libc6)" "$(ver libstdc++6)" "$(ver libgcc-s1)" "$(ver libgomp1)" "$TAIL_F"
nfail=0
pt() {   # pt <M=N=K> <φ|uniform>
  local n=$1 phi=$2 env=() out rc
  [ "$phi" = uniform ] || env=("OZ_PHI=$phi")
  echo "=== dgemmref: M=$n N=$n K=$n input=$([ "$phi" = uniform ] && echo uniform || echo "phi:$phi")"
  if out=$(env "${env[@]}" /tmp/dgemmcheck "$n" "$n" "$n" 2>&1); then rc=0; else rc=$?; fi
  printf '%s\n' "$out"
  echo "RC,$rc"
  if [ "$rc" -ne 0 ] || ! printf '%s\n' "$out" | grep -q '^DGEMMREF,'; then nfail=$((nfail + 1)); fi
}
case "${DGEMMREF_SET:-paper}" in
  paper)
    for phi in @PHI_PAPER@; do pt @PHI_N@ "$phi"; done
    for n in 3520 7040 14080; do pt "$n" 0; done
    for n in 3520 7040 14080; do pt "$n" uniform; done ;;
  phi14080)   # 14080^3 で phi を広げた入力（measure/exponent_adaptive.sh・diagnose/probes.sh の probe_phiacc_*・probe_phiperf_* と同じ入力）
    for phi in @PHIN_PHIS@; do pt 14080 "$phi"; done ;;
  *) echo "ERROR: DGEMMREF_SET=$DGEMMREF_SET は paper か phi14080" >&2; exit 2 ;;
esac
echo "EXIT,rc=$nfail,post_check=0"
exit $((nfail > 0))
EOS
# φ の点と大きさ（@PHI_PAPER@・@PHI_N@・@PHIN_PHIS@）はホストでここに埋める（コンテナへ渡すのは値を並べた文字列）．正本は
# lib/constants.tsv の PHI_SWEEP・PHI_SWEEP_05（paper の組は 2 つを小さい順に）・PHI_N と PHIN_PHIS で，比べる尾崎の掃引
# measure/exponent_sweep.sh・measure/exponent_adaptive.sh と照合 lib/phi_logs.py・lib/peer_logs.py・verify/check_paper_numbers.py も同じ行を読む
phs=$(const_get PHI_SWEEP) && ph05=$(const_get PHI_SWEEP_05) && phin=$(const_get PHI_N) && phn=$(const_get PHIN_PHIS) || exit 2
phi_paper=$(printf '%s\n' $phs $ph05 | LC_ALL=C sort -g | tr '\n' ' ')
INNER=${INNER//@PHI_PAPER@/${phi_paper% }}
INNER=${INNER//@PHI_N@/$phin}
INNER=${INNER//@PHIN_PHIS@/$phn}
case $INNER in *@PHI*) echo "ERROR: measure/cpu_dgemm_reference.sh の φ の点の埋め込みが残っている" >&2; exit 2 ;; esac

head_f="PROVENANCE,rev=$rev,dirty=$dirty,date=$(date -Iseconds),image=${img:0:12},dockerfile=$img_df,coretype=$coretype,cflags=$CF12"

# run_inner   ソースを固めてコンテナへ渡し，コンテナの終了コードを返す．ソースを固める処理（src_tar）が失敗したら
#   （コンテナが途中までの tar を展開して走ったとしても）理由を出して 2 を返す
run_inner() {
  local st
  src_tar | podman run --rm -i -w /tmp "${thr[@]}" -e "HEAD_F=$head_f" -e "TAIL_F=$tail_f" \
    -e "DGEMMREF_SET=$SET" -e "DGC_INC=$DGC_INC" -e "DGC_SRC=$DC" -e "DGC_FLAGS=$DGC_FLAGS" "$img" bash -c "$INNER" 9>&-
  st=("${PIPESTATUS[@]}")
  if [ "${st[0]}" != 0 ]; then
    echo "ERROR: コンテナへ渡すソースを固められなかった（git archive・tar の終了コード ${st[0]}）．このログは使えない" >&2
    return 2
  fi
  return "${st[1]}"
}
if [ "$OUT" = - ]; then
  run_inner; exit $?
fi
mkdir -p "$(dirname "$OUT")" || exit 2
tmp="$(dirname "$OUT")/.$(basename "$OUT").tmp.$$"
run_inner | tee "$tmp"
ps=("${PIPESTATUS[@]}")
rc=${ps[0]}; trc=${ps[1]}
# 置き換えるのは，書き込みも走らせるのも最後まで済んだときだけ（冒頭の DGEMMREF_OUT）．終了コードの条件（0 か 1）はここで見て，
# 書き込みの失敗と中身の形の規則は lib/common.sh の keep_log・replace_log に任せる（tee_replace と同じ規則）
. ./scripts/lib/common.sh    # keep_log・replace_log
if [ "$trc" != 0 ]; then
  keep_log "$tmp" "$OUT" "一時ファイル $tmp へ書き込めなかった（tee の終了コード $trc．空き容量・権限）"; [ "$rc" -ne 0 ] || rc=2
elif [ "$rc" != 0 ] && [ "$rc" != 1 ]; then
  keep_log "$tmp" "$OUT" "最後まで走らなかった（終了コード $rc）"
elif replace_log "$tmp" "$OUT"; then
  echo "（出力は $OUT．終了コード $rc）" >&2
else
  [ "$rc" -ne 0 ] || rc=2
fi
exit "$rc"
