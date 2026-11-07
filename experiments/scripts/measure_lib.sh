# measure_lib.sh -- ホストで 1 項目ずつ測るスクリプトが共有する部品（source して使う．単独では実行しない）
# ---------------------------------------------------------------------------
# リポジトリ直下へ cd してから
#     . ./scripts/measure_lib.sh               # 測る：読み込んだ時点で測る前の確認とデバイスの排他まで行う（下の 0〜3）
#     . ./scripts/measure_lib.sh --defs-only   # 一覧（--list・--commands）と調査：関数と変数の定義だけ（確認も排他もしない）
# 1 項目（ログ 1 本）をコンテナで流す手続きは run_item（冷ます・冷まさない・同じログへの追記の 3 形と，追跡するログの
# 置き換え），冷ましてから性能を測る項目の手続きと表示は run_dram．どちらもホストで動き（コンテナの外から実行する），
# 性能を測る項目はホストの tt-smi で冷ましてから測る．ログの行の形と出どころの確認の規則（測る前の provenance_check，
# 測っている途中の prov_guard・prov_bin・prov_end，完了の記録 prov_exit）は scripts/README.md の「ログの出どころ」が正本で，
# ここには関数ごとに何をするかだけを書く（読み込む側は `command grep -l measure_lib.sh scripts` で分かるので一覧を書かない）．
#
# 読み込んだ時点で（--defs-only でなければ）
#   0. 環境に残った OZ_PROVENANCE を外す（コンテナで動くスクリプトへ渡すための変数で，残っているとホストの
#      provenance_line がその古い行を返し，全ログの rev・dirty・date が古いまま，下の確認も同じ文字列どうしを比べて働かない），
#   1. 今の src/build が今のソースのビルドかを確かめ（provenance.sh の provenance_check．外れれば止める．ALLOW_DIRTY=1 は
#      試し用で，理由を表示して続けるが，そのログには allow_dirty=1 の印が付き既定の検査を通らない），
#      確かめたその行を測り始めの出どころ（PROV_START_LINE・PROV_START）とし，
#   2. デバイスの排他を取り（device_lock.sh），そのシェルが終わるまで持つ（冷却待ちのあいだに別の実行が
#      チップを温めないように），
#   3. 排他を待つあいだ（最長 TT_DEVICE_LOCK_WAIT 秒）に作業木・ビルド・tt-metal・イメージ・マシンが変わっていないかを
#      確かめ直す（provenance_check と prov_guard．変わっていれば何も測らずに終了コード 2 で止める）．
#
# 冷却待ち（wait_cool）は温度を COOL_POLL 秒おきに読み，しきい値の判定に使った標本をそのまま COOLCOND 行の
# start に残す（判定の後にもう一度読み直すと，start が判定に使った温度と食い違いうるので，読み直さない）．
# ONLY によるログ名の絞り込みは各スクリプトが項目ごとに掛ける．ONLY を環境に残したまま走らせると，
# 外れた項目は黙って飛ばされる（remeasure_all.sh は呼び出し元の ONLY を引き継がない）．ONLY は読み込んだ時点で
# 正規表現（grep -E）として読めるかを確かめる（読めないと grep の失敗を「当たらない」と読んで何も測らずに 0 で終わる）．
# 1 項目も当たらなかったときは only_summary が注意を出す．
#
# 環境変数（名前・読む所・remeasure_all.sh が引き継ぐかの正本は lib/knobs.tsv．ここには値の決め方の理由を書く）
#   計時の方式（コンテナへは run.sh が渡さないので，下の timing_env がコマンドの文字列に書いて渡す）
#   OZ_TIMING  steady（既定）：段ごとに，クロックが負荷に追いつくまで流してから測る（定常状態．論文の測定）．
#              cold：温めを warmup 回（尾崎の経路は 1 回）だけにした従来の計時（比べるときだけ．検査は論文用に拒む）
#   lib/knobs.tsv の kind=timing の名前（OZ_WARMUP_*・OZ_TELEM_MS・OZ_TELEM_TRACE・OZ_TELEM_FW）
#              定常状態の条件とテレメトリ（既定はバイナリの既定．src/common/oz_timing.hpp）．立っていればそのまま渡す．
#              冷却（TMAX）はそのまま各項目の始めの条件で，温めはその後に段ごとに行う
#
#   COOL_POLL  冷却待ちで温度を読む間隔［s．0.1〜3600 の数］（既定 5）．始める温度がしきい値のすぐ下に揃う（30 s おきだと 56〜60 degC に
#           ばらつく）．定常状態の計時でも始める温度が 4.5 degC 低いと，14080^3 の I-BF16 の GEMM 段の AICLK が約 1.3% 高く，
#           DRAM->DRAM が約 1.2% 短く出る（docs/remeasure_other_machine.md の 6.3）
#   TMAX    実行開始を許す温度［degC．小数も可］．アイドルの底が 56 degC 付近なので 60 に採る（論文の「60℃以下」）．
#           60 より大きくすると，そのログは論文用の検査に拒まれる．数でなければ読み込んだ時点で止める
#   TWAIT   冷却待ちの上限［s．0〜86400 の整数］．届かないまま上限に達したら測りはするが（記録のため），その項目は失敗に数える．
#           既定 1800．定常状態の計時（各段を 10 s 以上流す）ではチップが 80 degC 台まで上がり，600 s では 60 degC に
#           戻りきらない項目があった．整数でなければ読み込んだ時点で止める（小数や誤字・64 bit を超える桁のまま流すと
#           wait_cool の比較が誤り，上限が働かずに待ち続ける）
#           温度が読めない（3 回続けて）ときは待たずにスクリプトを止める（下の wait_cool）
#   CLK_DT  クロックを採る間隔［s．0.1〜3600 の数］（既定 2．tt-smi 1 回の照会にも 1 s 近くかかる）．
#           COOL_POLL・CLK_DT は数でなければ読み込んだ時点で止める（0 や誤字だと sleep がすぐ返り，tt-smi を休みなく呼ぶ）
#   TT_SMI  温度とクロックを読む tt-smi（provenance.sh の PROV_SMI．既定 ~/.venvs/tt-smi/bin/tt-smi，無ければ PATH の tt-smi）
#   TT_DEVICE  測る Blackhole（provenance.sh の PROV_DEVICE．既定 /dev/tenstorrent/0）．温度とクロックは tt-smi -s の
#           出力のうちこのデバイスのボードの値を読む（カードが複数ある計算機で別のカードの温度で冷却を判定しないように）
. ./scripts/lib/common.sh    # require_number・const_get
. ./scripts/lib/knobs.sh     # knobs_names（lib/knobs.tsv の計時のつまみ）
# 冷却と標本の既定（上の注記）．既定の正本はここ（remeasure_all.sh は --defs-only で読み込んでこの値を使う）．
# TMAX の既定は論文の冷却条件そのもの（lib/constants.tsv の APPROVED_TMAX．検査側の oz_logs.py も同じ行を読む）
APPROVED_TMAX=$(const_get APPROVED_TMAX) || exit 2
TMAX="${TMAX:-$APPROVED_TMAX}"
TWAIT="${TWAIT:-1800}"
COOL_POLL="${COOL_POLL:-5}"
CLK_DT="${CLK_DT:-2}"
if ! [[ $TMAX =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: TMAX=$TMAX は温度［degC］の数として読めない" >&2; exit 2
fi
# TWAIT は 64 bit を超えると [ -ge ] の比較が誤りになり上限が働かないので，桁も確かめる（86400 s＝1 項目に 1 日）．
# COOL_POLL・CLK_DT は 0 や誤字だと sleep がすぐ返り，tt-smi を休みなく呼び続ける（lib/common.sh の require_number．
# 値の全体が数の形であることを要る）
require_number TWAIT "$TWAIT" 0 86400 int || exit 2
require_number COOL_POLL "$COOL_POLL" 0.1 3600 || exit 2
require_number CLK_DT "$CLK_DT" 0.1 3600 || exit 2
. ./scripts/provenance.sh    # provenance_line・provenance_check・bin_line・prov_smi_get（PROV_SMI・PROV_DEVICE）
. ./scripts/device_lock.sh   # device_lock
SMI=$PROV_SMI
# 温度は小数まで読む（整数部だけ読むと 60.9 degC を 60 として通してしまう）．tt-smi -s の telemetry の値
# （asic_temperature は "75.7" のような小数の文字列，aiclk は MHz の整数の文字列）を数の形のときだけ返し，
# 読めなければ空を返す（呼び出し側が読めなかった回として扱う）
temp() { prov_smi_get telemetry.asic_temperature 2>/dev/null | grep -oE '^[0-9]+(\.[0-9]+)?$' | head -1; }
clk()  { prov_smi_get telemetry.aiclk 2>/dev/null | grep -oE '^[0-9]+' | head -1; }
# le 値 しきい値   小数どうしの比較（値が空なら偽）
le() { [ -n "$1" ] && awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 <= b + 0) }'; }
# 計時の方式（上の注記）．値はコマンドの文字列へそのまま埋め込むので，数の形（と steady・cold）だけを受け付ける
OZ_TIMING="${OZ_TIMING:-steady}"
case $OZ_TIMING in steady|cold) ;; *) echo "ERROR: OZ_TIMING=$OZ_TIMING は steady か cold" >&2; exit 2 ;; esac
# 計時のつまみ（lib/knobs.tsv の kind=timing．値の全体が 0 以上の数であることを要る：改行や空白を含む値はコマンドの文字列の中で
# 別のコマンドになる）
TIMING_VARS=$(knobs_names kind=timing) && [ -n "$TIMING_VARS" ] || { echo "ERROR: lib/knobs.tsv から計時のつまみを読めない" >&2; exit 2; }
for _v in $TIMING_VARS; do
  if [ -n "${!_v:-}" ] && ! [[ ${!_v} =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: $_v=${!_v} は 0 以上の数でない" >&2; exit 2
  fi
done
unset _v
# telem_clk ログ [段]   ログの TELEM 行のうち段（既定 gemm）の計時区間の AICLK の平均［MHz］（無ければ -）．表示用
telem_clk() {
  local v
  v=$(grep -E "^TELEM,stage=${2:-gemm}," "$1" 2>/dev/null | head -1 | grep -oE 'aiclk_mean=[0-9.]+' | cut -d= -f2)
  printf '%s' "${v:--}"
}
# telem_lim ログ [段]   同じ段の計時区間で AICLK を最も長く絞っていた仕組みとボードの入力電力の平均（TELEM 行の末尾の
#   arb_top・arb・input_power_mean．"，Doppler 100%・入力 309 W" の形．欄が無ければ空）．表示用．
#   割合は arb（"<番号>:<割合>|…"）のうち arb_top の名前の番号の項目から取る（先頭の項目を取ると，並びが変わったときに
#   別の仕組みの割合を付けてしまう）．名前と番号の対応は lib/constants.tsv の ARB_NAMES（src/common/telemetry.hpp の arb_name と
#   同じ並びであることは verify/check_routes.py が確かめる）
TELEM_ARB_NAMES=$(const_get ARB_NAMES) || exit 2
telem_lim() {
  local ln top frac pin arb n id="" i=0
  ln=$(grep -E "^TELEM,stage=${2:-gemm}," "$1" 2>/dev/null | head -1)
  top=$(printf '%s' "$ln" | grep -oE 'arb_top=[A-Za-z_]+' | cut -d= -f2)
  [ -n "$top" ] || return 0
  for n in $TELEM_ARB_NAMES; do
    [ "$n" = "$top" ] && { id=$i; break; }
    i=$((i + 1))
  done
  arb=$(printf '%s' "$ln" | grep -oE ',arb=[0-9:.|]+' | head -1 | cut -d= -f2)
  frac=""
  [ -n "$id" ] && frac=$(printf '%s\n' "$arb" | tr '|' '\n' | sed -n "s/^$id:\([0-9.]*\)$/\1/p" | head -1 | awk '{printf "%.0f", 100*$1}')
  pin=$(printf '%s' "$ln" | grep -oE 'input_power_mean=[0-9.]+' | cut -d= -f2 | awk '{printf "%.0f", $1}')
  printf '，%s %s%%・入力 %s W' "$top" "${frac:-?}" "${pin:-?}"
}
# timing_env   コンテナの中のコマンドの前に置く計時の方式の変数（"OZ_TIMING=steady OZ_WARMUP_SEC=…" の形）
timing_env() {
  local v out="OZ_TIMING=$OZ_TIMING"
  for v in $TIMING_VARS; do [ -n "${!v:-}" ] && out="$out $v=${!v}"; done
  printf '%s' "$out"
}
# ONLY（ログ名の正規表現．各スクリプトは grep -Eq -- で読む）が正規表現として読めるか．読めないと grep は終了コード 2 で，
# 呼び出し側はそれを「当たらない」と読むので，全項目を黙って飛ばして 0 で終わる．
# 当たらない（grep の 1）は || で受ける（呼び出し元が set -e でも，正しい正規表現をここで止めない）
if [ -n "${ONLY:-}" ]; then
  _r=0; printf 'x\n' | grep -Eq -- "$ONLY" 2>/dev/null || _r=$?
  [ "$_r" -le 1 ] || { echo "ERROR: ONLY='$ONLY' は正規表現（grep -E）として読めない" >&2; exit 2; }
  unset _r
fi
PROV_ITEMS=0          # prov_begin で書き始めた項目のログの数（only_summary が見る）
PROV_BINS=""          # 流す項目の実行ファイルと sha256（"名前=sha256 …"．prov_bin が足し，prov_begin が空にする）

# log_put ログ 行…   ログの末尾へ行を足す．書けなければ（空き容量・権限・置き場所が無い）理由を出してスクリプトを
#   終了コード 2 で止める（書けなかった行を書いたことにしない．EXIT が書けなければ，そのログは完了の記録が無いので
#   検査に拒まれる）
log_put() {
  local log=$1; shift
  if ! printf '%s\n' "$@" >>"$log"; then
    echo "ERROR: $log に書き込めない（空き容量・権限を確かめること）．この項目から先を測らずに止める" >&2
    exit 2
  fi
}

# prov_guard 行 [ログ]
#   これから項目のログに書く出どころの行（provenance_line の出力）が測り始め（PROV_START）と同じか
#   （PROV_IDENT_KEYS：rev・dirty・tt-metal・イメージ・ビルドの記録．provenance.sh の prov_ident）と，マシンの欄
#   （prov_machine_diff）が測り始めの行と同じかを確かめ，違えば違う欄を出してスクリプトを終了コード 2 で止める．
#   測っている途中で src/・scripts/・tools/ を書き換えたりコミットしたり（paper/ だけのコミットでも HEAD が変わる），
#   src/build を作り直したり，共有の tt-metal を直したりイメージを入れ替えたりすると，それ以降の項目が別の出どころに
#   なり，検査に拒まれる（2026-10-20 に 29 本を測り直した）．気づかないまま全部を流さないよう，その項目を測る前に止める．
#   ログには触れずに止めるので（呼び出し側の prov_begin はこれを通ってから書く），前に測った正しいログは残る．
#   ログを与えなければ（読み込み時の排他の後の確認）「何も測らずに止める」と出す．ALLOW_DIRTY=1（試し用）のときは見ない
prov_guard() {
  local line=$1 log=${2:-} d md bad=0
  [ -n "${PROV_START:-}" ] && [ "${ALLOW_DIRTY:-0}" != 1 ] || return 0
  d=$(prov_ident_diff "$PROV_START" "$line") || bad=1
  md=$(prov_machine_diff "$PROV_START_LINE" "$line") || bad=1
  [ "$bad" = 0 ] && return 0
  if [ -n "$log" ]; then
    printf 'ERROR: 測り始めから HEAD・作業木（src/・scripts/・tools/）・ビルド・tt-metal・イメージ・マシンのどれかが変わった:\n%s\n' \
      "$(printf '%s\n%s' "$d" "$md" | grep .)" >&2
    printf '  %s は書き換えずに，この項目から先を測らずに止める（それより前に測り終えた項目のログは測り始めの出どころのまま）\n' "$log" >&2
  else
    printf 'ERROR: デバイスの排他を待っているあいだに HEAD・作業木・ビルド・tt-metal・イメージ・マシンのどれかが変わった:\n%s\n  何も測らずに止める\n' \
      "$(printf '%s\n%s' "$d" "$md" | grep .)" >&2
  fi
  exit 2
}

# prov_begin ログ [append]
#   項目のログの 1 行目に置く出どころの行を作り，prov_guard を通ってからログへ書く（append なら後ろへ足す．
#   measure/floating_decomposition.sh の ONLY）．通らなければログに触れずに止まる（前の正しいログを PROVENANCE 1 行の切れ端で
#   上書きしない．切れ端が別の rev のディレクトリに残ると，phi_nowin などの組全体が「rev が揃っていない」で拒まれる）
prov_begin() {
  local line
  line=$(provenance_line)
  prov_guard "$line" "$1"
  if [ "${2:-}" = append ]; then printf '%s\n' "$line" >>"$1"; else printf '%s\n' "$line" >"$1"; fi || {
    echo "ERROR: $1 に出どころの行を書き込めない（空き容量・権限・置き場所を確かめること）．この項目から先を測らずに止める" >&2
    exit 2
  }
  PROV_ITEMS=$((PROV_ITEMS + 1))
  PROV_BINS=""
}

# prov_invalidate ログ when changed 理由   出どころの確認に落ちた項目のログの末尾に
#     INVALIDATED,item=<ログ名>,when=<when>,run=<OZ_RUN_ID>,date=<時刻>,changed=<changed>,reason=<理由>
#   を足し（EXIT は書かない），終了コード 2 で止める．足せなかった（書き込めない）ときもそう表示して止める
#   （EXIT が無いので検査は「完了の記録なし」で拒むが，そのログは使わないこと）
#   run_item --replace の一時ファイル（RI_TMP）なら，項目の名前は置き換え先のログの名前にし，足した後にその一時ファイルを
#   <ログ>.incomplete へ移す（置き換え先の前の完全なログは残す）
prov_invalidate() {
  local log=$1 when=$2 ch=$3 why=$4 item
  item=$(basename "$log" .log)
  [ -n "${RI_TMP:-}" ] && [ "$log" = "$RI_TMP" ] && item=$(basename "$RI_REPLACE_TO" .log)
  if printf 'INVALIDATED,item=%s,when=%s,run=%s,date=%s,changed=%s,reason=%s\n' \
       "$item" "$when" "${OZ_RUN_ID:-none}" "$(date -Iseconds)" "$ch" "$why" >>"$log"; then
    if [ -n "${RI_TMP:-}" ] && [ "$log" = "$RI_TMP" ] && mv -f "$log" "$RI_REPLACE_TO.incomplete"; then
      log="$RI_REPLACE_TO.incomplete（$RI_REPLACE_TO は前のまま）"
    fi
    printf 'ERROR: %s（%s）．\n  %s の末尾に INVALIDATED 行を足した（EXIT 行は書かない．検査はそのログを使わない）．この項目から先を測らずに止める\n' \
      "$why" "$ch" "$log" >&2
  else
    printf 'ERROR: %s（%s）．\n  !!! %s の末尾に INVALIDATED 行を足せなかった（書き込めない）．EXIT 行も書いていないので検査は完了の記録なしとして拒むが，このログは使わないこと．この項目から先を測らずに止める\n' \
      "$why" "$ch" "$log" >&2
  fi
  exit 2
}

# prov_bin ログ コマンド…
#   コマンドの中の ./src/build/<名前> の BINARY 行（provenance.sh の bin_line）をログへ足し，その名前と sha256 を
#   PROV_BINS に覚える（流した後に prov_end が取り直して比べる）．今のファイルの sha256 がビルドの記録（BUILD_STAMP）の
#   値と違えば（測り始めの provenance_check の後に，スタンプを残して実行ファイルだけが差し替わった），流す前に
#   INVALIDATED（when=before_run）を足して止める．ALLOW_DIRTY=1（試し用）のときは比べない
prov_bin() {
  local log=$1 ln name h sh; shift
  ln=$(bin_line "$*")
  [ -n "$ln" ] || return 0
  log_put "$log" "$ln"
  name=$(printf '%s' "$ln" | cut -d, -f2); h=$(prov_field "$ln" sha256); sh=$(prov_field "$ln" stamp_sha256)
  PROV_BINS="$PROV_BINS${PROV_BINS:+ }$name=$h"
  [ -n "${PROV_START:-}" ] && [ "${ALLOW_DIRTY:-0}" != 1 ] || return 0
  if [ "$h" = none ] || [ "$h" != "$sh" ]; then
    prov_invalidate "$log" before_run "bin_$name:${sh:0:16}->${h:0:16}" "流す前の実行ファイル src/build/$name がビルドの記録と違う"
  fi
}

# prov_bin_diff   PROV_BINS の各実行ファイルの今の sha256 を取り直し，覚えた値（流す前の BINARY 行の値）とビルドの記録の
#   値の両方と比べる．違うものを "bin_<名前>:<前>-><今>"（sha256 の先頭 16 桁．無ければ none）で ; につないで出す
prov_bin_diff() {
  local kv name h now st out=""
  for kv in $PROV_BINS; do
    name=${kv%%=*}; h=${kv#*=}
    now=$(sha256sum "$PROV_ROOT/src/build/$name" 2>/dev/null | cut -d' ' -f1)
    st=$(grep -m1 "^BIN,$name," "$PROV_ROOT/src/build/BUILD_STAMP" 2>/dev/null | cut -d, -f3)
    now=${now:-none}; st=${st:-none}
    if [ "$now" != "$h" ]; then out="$out${out:+;}bin_$name:${h:0:16}->${now:0:16}"
    elif [ "$now" != "$st" ]; then out="$out${out:+;}bin_${name}_stamp:${st:0:16}->${now:0:16}"; fi
  done
  printf '%s' "$out"
}

# prov_bin_check ログ   1 本のログに何回も流す項目（measure/floating_decomposition.sh）の 2 回目以降の実行の前に，実行ファイルが
#   最初の BINARY 行のときのままかを確かめる（違えば INVALIDATED（when=before_run）を足して止める）
prov_bin_check() {
  local d
  [ -n "${PROV_START:-}" ] && [ "${ALLOW_DIRTY:-0}" != 1 ] || return 0
  d=$(prov_bin_diff)
  [ -z "$d" ] || prov_invalidate "$1" before_run "$d" "流す前に実行ファイルが最初の BINARY 行のときと変わった"
}

# prov_end ログ
#   項目を流した後（AICLK・COOLCOND を書いた後，EXIT を書く前）に出どころ（prov_ident：rev・dirty・tt-metal・イメージ・
#   ビルドの記録）を今の作業木から取り直し，実行ファイルの sha256 も取り直して（prov_bin_diff），測り始め・流す前と
#   違えば，そのログの末尾に
#     INVALIDATED,item=<ログ名>,when=after_run,run=<OZ_RUN_ID>,date=<時刻>,changed=<欄>:<始め>-><今>[;…],reason=…
#   を足して終了コード 2 で止める（EXIT は書かない．prov_invalidate）．カーネルは実行時に src/ から JIT でビルドされるので，
#   項目を流している途中で直したソースはその項目に入るが，PROVENANCE 行は流す前の状態なので，それだけでは dirty=0 のまま
#   検査を通る．実行ファイルは git の外（src/build）なので dirty には出ず，スタンプを残して差し替えれば build_sha も
#   変わらない．検査側（oz_logs.reasons）は INVALIDATED 行のあるログを「無効化」で拒む．途中で直して元に戻した場合は
#   見分けられない（それを防ぐのは remeasure_all.sh のスナップショット）．マシンの欄は次の項目の prov_guard が見る
#   （tt-smi を呼ばない）．ALLOW_DIRTY=1（試し用）のときは見ない．
#   通れば 0 を返す（呼び出し側はその後に EXIT を書く．ふつうは下の prov_exit を使う）．確かめたかを PROV_END_CHECKED（1 か 0．
#   測り始めの出どころが無い・ALLOW_DIRTY=1 で見なかったら 0）に置く
PROV_END_CHECKED=0
prov_end() {
  local log=$1 now ch="" k a b bd
  PROV_END_CHECKED=0
  [ -n "${PROV_START:-}" ] && [ "${ALLOW_DIRTY:-0}" != 1 ] || return 0
  now=$(prov_ident)
  if [ "$now" != "$PROV_START" ]; then
    for k in $PROV_IDENT_KEYS; do
      a=$(prov_field "$PROV_START" "$k"); b=$(prov_field "$now" "$k")
      [ "$a" = "$b" ] || ch="$ch${ch:+;}$k:${a:-none}->${b:-none}"
    done
  fi
  bd=$(prov_bin_diff)
  [ -n "$bd" ] && ch="$ch${ch:+;}$bd"
  [ -z "$ch" ] && { PROV_END_CHECKED=1; return 0; }
  prov_invalidate "$log" after_run "$ch" "項目を流している途中で出どころか実行ファイルが測り始めと変わった"
}

# prov_exit ログ 終了コード   項目の最後：流した後の確認（prov_end）を通ってから，完了の記録
#     EXIT,rc=<終了コード>,post_check=<1|0>
#   を末尾に書く（書けなければ止める）．post_check=1 はホストで流した後の確認を通った印（0 は確認を見なかった：ALLOW_DIRTY=1 の
#   試し．コンテナの中で書く完了の記録も 0．scripts/README.md「ログの出どころ」）．確認に落ちれば EXIT を書かずに止まる．
#   AICLK・COOLCOND などの行はこれより前に書いておく
prov_exit() {
  prov_end "$1"
  log_put "$1" "EXIT,rc=$2,post_check=$PROV_END_CHECKED"
}

# only_summary   ONLY を与えたのに 1 項目も測らなかったら注意を出す（打ち間違いに気づけるように．終了コードは変えない：
#   remeasure_all.sh の REMEASURE_ONLY は段ごとの測定スクリプトへ同じ ONLY を渡し，当たる項目の無い段があってよい）
only_summary() {
  [ -n "${ONLY:-}" ] && [ "$PROV_ITEMS" = 0 ] || return 0
  echo "注意: ONLY='$ONLY' に当たる項目が無く，何も測らなかった" >&2
}

# --defs-only（一覧・調査と remeasure_all.sh．remeasure_all.sh は冷却の既定と値の確かめと tt-smi の温度の読み方だけを使う）は
# ビルドの記録の確認と排他を飛ばす（関数と変数の定義だけ）．
# 引数なしで読み込むと呼び出し元の位置引数（サイズなど）が見えるが，それが --defs-only になることは無い
if [ "${1:-}" != --defs-only ]; then
  # 冒頭の注記の 0〜3．測り始めの出どころは排他を待つ前に取り，provenance_check で確かめたその行にする
  # （排他の後に取ると，待っているあいだに汚れた作業木や新しい rev がそのまま基準になり，prov_guard が止めない）
  if [ -n "${OZ_PROVENANCE:-}" ]; then
    echo "注意: 環境の OZ_PROVENANCE を外した（ホストで動く測定は出どころの行を今の作業木から作る）" >&2
    unset OZ_PROVENANCE
  fi
  PROV_START_LINE=$(provenance_line)
  provenance_check "$PROV_START_LINE" || exit 2
  PROV_START=$(prov_ident "$PROV_START_LINE")
  device_lock || exit $?
  _pl=$(provenance_line)
  provenance_check "$_pl" || { echo "ERROR: デバイスの排他を待っているあいだに src/build が今のソースの測定に使えなくなった．何も測らずに止める" >&2; exit 2; }
  prov_guard "$_pl"
  unset _pl
fi

# wait_cool
#   温度が TMAX 以下になるまで COOL_POLL 秒おきに待つ（最長 TWAIT 秒）．判定に使った温度を pre に，
#   しきい値へ届いたかを cooled（1=届いた，0=届かないまま上限に達した）に置く．
#   届かないまま測った値は他と同じ条件ではないので，呼び出し側がログの COOLCOND 行へ cooled を残し，
#   その項目を失敗に数える（fail=1）．呼び出し側の関数で local pre cooled と宣言しておけば，その局所変数へ書かれる．
#   温度が読めない（tt-smi が壊れた・出力の形式が変わった）ときは 10 秒おきに読み直し，3 回続けて読めなければ
#   測らずにこのスクリプトを終了コード 4 で止める（TWAIT まで待ってから冷却未達のまま測ると，
#   remeasure_all.sh の約 120 項目のそれぞれで 10 分ずつ失い，全部が失敗になる）．止まるのは項目のログを
#   書き始める前なので，半端なログは残らない
wait_cool() {
  local t0=$SECONDS t bad=0
  cooled=0
  while :; do
    t=$(temp)
    if [ -z "$t" ]; then
      bad=$((bad + 1))
      if [ "$bad" -ge 3 ]; then
        echo "ERROR: tt-smi（$SMI）が温度を返さない（$bad 回続けて）．冷却の条件を確かめられないので測らずに止める" >&2
        exit 4
      fi
      sleep 10; continue
    fi
    bad=0
    if le "$t" "$TMAX"; then cooled=1; break; fi
    [ $((SECONDS - t0)) -ge "$TWAIT" ] && break
    sleep "$COOL_POLL"
  done
  pre=$t
}
# 表示用の印．cooled=0 のときだけ「!冷却未達」を返す
uncooled_mark() { [ "$cooled" = 1 ] || echo ' !冷却未達'; }
# coolcond ログ名 [key=value…]   COOLCOND 行（start・cooled・tmax・twait と，足したい欄）
coolcond() {
  local name=$1; shift
  local IFS=,
  printf 'COOLCOND,%s,start=%s,cooled=%s,tmax=%s,twait=%s%s\n' "$name" "$pre" "$cooled" "$TMAX" "$TWAIT" "${*:+,$*}"
}

# clk_begin 採取ファイル
#   実行中のクロックを CLK_DT 秒ごとに採取ファイルへ 1 行 1 値で採り始める．
# clk_end ログ名 ログ
#   採るのを止め，要約を cmin cmax cmean cn（全標本）と bmean（800 MHz より上の標本の平均）に置き，
#   ログへ次の 1 行を足す（値は MHz．標本が無ければ 0）．
#     AICLK,<ログ名>,window=process,n=,min=,max=,mean=,median=,busy_n=,busy_min=,busy_max=,busy_mean=,busy_median=,idle=800,dt=2
#   標本を採る区間はプロセスの実行中全体（window=process）で，コンテナの起動・JIT・入力の準備・ウォームアップを
#   含み，計時区間とは同期していない．busy_* は 800 MHz より上の標本だけの値だが，短い計時区間を標本が捉えて
#   いる保証は無く，準備中の値だけのこともある．したがって「実行中の 800 MHz 超の標本」として参考に表示する
#   だけで，性能の正規化には使わない．計時区間のクロックはバイナリが計時区間に合わせて採り，TELEM 行に出す
#   （src/common/oz_timing.hpp の print_telem．scripts/README.md の「ログの出どころ」）．ここの標本はそれと別の，同期していない参考値である．
IDLE_MHZ=$(const_get IDLE_MHZ) || exit 2   # lib/constants.tsv（oz_logs.aiclk の要約も同じ行を読む）
#   採取のループは親（$$．このスクリプト）が生きているあいだだけ回る．スクリプトの中の非同期のコマンドは
#   SIGINT を無視するので，親の生存を見ないと Ctrl-C でスクリプトが止まった後もループが残り，tt-smi を呼び続け，
#   受け継いだデバイスの排他（fd 9）を持ち続けて次の実行を待たせてしまう
clk_begin() {
  _clk_file=$1; : >"$_clk_file"
  ( while kill -0 $$ 2>/dev/null; do c=$(clk); [ -n "$c" ] && echo "$c" >>"$_clk_file"; sleep "$CLK_DT"; done ) &
  _clk_pid=$!
}
clk_end() {
  local name=$1 log=$2 bn bmin bmax bmed cmed
  # 止めた子の wait は非零（シグナルで終わった状態）を返すので受けておく（呼び出し元が set -e でも EXIT と prov_end へ進むように）
  kill "$_clk_pid" 2>/dev/null || :
  wait "$_clk_pid" 2>/dev/null || :
  read -r cn cmin cmax cmean cmed bn bmin bmax bmean bmed <<<"$(sort -n "$_clk_file" | awk -v idle="$IDLE_MHZ" '
    function med(a, k, lo) { if (k == 0) return 0; return (k % 2) ? a[lo + (k + 1) / 2] : (a[lo + k / 2] + a[lo + k / 2 + 1]) / 2 }
    { v[NR] = $1; s += $1; if ($1 > idle) { if (!nb) fb = NR; nb++; sb += $1 } }
    END {
      if (!NR) { print "0 0 0 0 0 0 0 0 0 0"; exit }
      printf "%d %d %d %.0f %.0f ", NR, v[1], v[NR], s / NR, med(v, NR, 0)
      if (nb) printf "%d %d %d %.0f %.0f\n", nb, v[fb], v[NR], sb / nb, med(v, nb, fb - 1)
      else print "0 0 0 0 0"
    }')"
  log_put "$log" "$(printf 'AICLK,%s,window=process,n=%s,min=%s,max=%s,mean=%s,median=%s,busy_n=%s,busy_min=%s,busy_max=%s,busy_mean=%s,busy_median=%s,idle=%s,dt=%s' \
         "$name" "$cn" "$cmin" "$cmax" "$cmean" "$cmed" "$bn" "$bmin" "$bmax" "$bmean" "$bmed" "$IDLE_MHZ" "$CLK_DT")"
}

# run_item [--cool|--cool-detail] [--append|--cont] [--replace] [--bins "名前…"|--bins all] [--line 行]… ログ コマンドの文字列
#   1 項目（ログ 1 本）をコンテナで 1 回流す手続き．流す順は（--cool なら冷ます）→ 出どころの行（prov_begin）→ 実行ファイルの行
#   （prov_bin）→ --line の行 → 流す（run.sh．出力はログの後ろへ）→（--cool なら AICLK 行と COOLCOND 行）→ 流した後の確認と完了の
#   記録（prov_exit）．測定スクリプトは項目ごとにこれを呼び，表示と合否は呼び出し側が RI_RC（コンテナの終了コード）と，--cool の
#   ときの pre・cooled（wait_cool）・cmin などの標本の要約（clk_end）で決める．ONLY による絞り込みも呼び出し側が掛ける．
#     --cool         冷ましてから測る（wait_cool，実行中のクロックの標本 <ログの幹>.clk と AICLK 行，COOLCOND 行）
#     --cool-detail  --cool に加えて，流す前のクロック（RI_CLK0）・流した後の温度とクロック（RI_POST・RI_CLK1）と実行中の標本の
#                    要約も COOLCOND 行に足す（measure/performance.sh の表の項目）
#     --append       出どころの行をログの後ろへ足す（前のログを消さない．measure/floating_decomposition.sh の ONLY）
#     --cont         同じログへの 2 回目以降の実行：出どころの行と実行ファイルの行を書かず，実行ファイルが最初の BINARY 行の
#                    ときのままかを確かめる（prov_bin_check．measure/floating_decomposition.sh）
#     --replace      一時ファイル（.<名前>.tmp.<pid>．*.log でないので検査は読まない）に書き，完了の記録の後にログへ置き換える
#                    （lib/common.sh の replace_log．git で追跡するログ accuracy_logs/ を途中で止められても前の完全なログを残す．
#                    流した後の確認に落ちたときは一時ファイルを <ログ>.incomplete へ移す．prov_invalidate）
#     --bins         実行ファイルの行を書く src/build の実行ファイルの名前（既定はコマンドの文字列の最初の ./src/build/<名前>．
#                    all ならビルドの記録 src/build/BUILD_STAMP に載った全部）
#     --line 行      実行ファイルの行の後（流す前）にログへ足す行（COOLENV 行，measure/floating_decomposition.sh の区切り行）
#   コマンドの文字列の中で環境を外す（unset）必要は無い（run.sh がコンテナの中でコマンドの前に lib/common.sh の oz_env_clean で外す）
RI_RC=0
RI_TMP=""; RI_REPLACE_TO=""
run_item() {
  local cool=0 detail=0 mode=new replace=0 bins="" lines=() log cmd tgt name t b l
  while :; do
    case "${1:-}" in
      --cool) cool=1; shift ;;
      --cool-detail) cool=1; detail=1; shift ;;
      --append) mode=append; shift ;;
      --cont) mode=cont; shift ;;
      --replace) replace=1; shift ;;
      --bins) [ -n "${2:-}" ] || { echo "ERROR: run_item の --bins に名前が無い" >&2; exit 2; }; bins=$2; shift 2 ;;
      --line) lines+=("$2"); shift 2 ;;
      -*) echo "ERROR: run_item の知らない印 $1" >&2; exit 2 ;;
      *) break ;;
    esac
  done
  [ $# = 2 ] || { echo "ERROR: run_item の引数はログとコマンドの文字列の 2 つ（$#）" >&2; exit 2; }
  log=$1; cmd=$2; name=$(basename "$log" .log); tgt=$log
  if [ "$replace" = 1 ]; then
    [ "$mode" = new ] || { echo "ERROR: run_item の --replace は --append・--cont と組まない" >&2; exit 2; }
    tgt="$(dirname "$log")/.$(basename "$log").tmp.$$"
    RI_TMP=$tgt; RI_REPLACE_TO=$log
  fi
  if [ "$cool" = 1 ]; then
    wait_cool
    # 冷却未達が 2 項目続いたら，その計算機は冷却の条件を満たせないとみなし，次の項目を測らずに終了コード 5 で止める
    # （1 項目ずつ TWAIT 待ってから失敗に数えると，冷ます百数十項目で数十時間を失う．remeasure_all.sh の do_step は
    # 5 を受けると KEEP_GOING によらず全体を止める．2026-11-04 の点検．1 項目だけの未達は今までどおり失敗に数えて続ける）
    if [ "$cooled" = 1 ]; then cool_miss_run=0; else cool_miss_run=$(( ${cool_miss_run:-0} + 1 )); fi
    if [ "$cool_miss_run" -ge 2 ]; then
      echo "ERROR: 2 項目続けて温度が TMAX=$TMAX degC まで下がらなかった（TWAIT=$TWAIT s 待って $pre degC）．"\
"冷却の条件を満たせないので，この項目から先を測らずに終了コード 5 で止める" >&2
      exit 5
    fi
    [ "$detail" = 1 ] && RI_CLK0=$(clk)
  fi
  case $mode in
    new) prov_begin "$tgt" ;;
    append) prov_begin "$tgt" append ;;
    cont) prov_bin_check "$tgt" ;;
  esac
  if [ "$mode" != cont ]; then
    case $bins in
      '') prov_bin "$tgt" "$cmd" ;;
      all) while IFS=, read -r t b _; do [ "$t" = BIN ] && prov_bin "$tgt" "./src/build/$b"; done <src/build/BUILD_STAMP ;;
      *) for b in $bins; do prov_bin "$tgt" "./src/build/$b"; done ;;
    esac
  fi
  for l in "${lines[@]}"; do log_put "$tgt" "$l"; done
  [ "$cool" = 1 ] && clk_begin "${log%.log}.clk"
  if ./scripts/run.sh "$cmd" >>"$tgt" 2>&1; then RI_RC=0; else RI_RC=$?; fi
  if [ "$cool" = 1 ]; then
    clk_end "$name" "$tgt"
    if [ "$detail" = 1 ]; then
      RI_POST=$(temp); RI_CLK1=$(clk)
      log_put "$tgt" "$(coolcond "$name" "clk0=$RI_CLK0" "end=$RI_POST" "clk1=$RI_CLK1" \
               "aiclk_min=$cmin" "aiclk_max=$cmax" "aiclk_mean=$cmean" "aiclk_n=$cn")"
    else
      log_put "$tgt" "$(coolcond "$name")"
    fi
  fi
  prov_exit "$tgt" "$RI_RC"
  if [ "$replace" = 1 ]; then
    RI_TMP=""; RI_REPLACE_TO=""
    replace_log "$tgt" "$log" || { echo "ERROR: $log を置き換えられない．この項目から先を測らずに止める" >&2; exit 2; }
  fi
}

# run_dram ラベル ログ名 環境変数… -- コマンド…
#   冷ましてから性能を測る 1 項目（diagnose/ablations.sh・diagnose/implementation_ratios.sh・measure/fidelity_comparison.sh・diagnose/probes.sh・measure/exponent_adaptive.sh）．
#   OZ_NOVERIFY=1 と計時の方式（timing_env．既定 steady）で run_item --cool に流し，全出力を $OUT/ログ名.log に残して指標の行を
#   表示する（実行中のクロックの標本は $OUT/ログ名.clk）．終了コードが非零か指標の行が無ければ FAILED とし，冷却がしきい値へ
#   届かなかったときも失敗として，どちらも fail=1 を立てる（呼び出し側が最後に非零で終わる）．指標の行は RUN_PAT（既定
#   'DRAM->DRAM : .*'．素の GEMM なら 'Performance : [0-9.]+ TFLOP/s'）で拾う．ラベルの表示幅は LABEL_W（既定 34）．
#   ONLY='abl_p8|abl_p9' のようにログ名の正規表現を与えると，その項目だけを測る．
run_dram() {
  local label=$1 log=$2 env="" line rc pre cooled pat="${RUN_PAT:-DRAM->DRAM : .*}"; shift 2
  while [ "$1" != "--" ]; do env="$env $1"; shift; done; shift
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$log" | grep -Eq -- "$ONLY"; then return; fi
  run_item --cool "$OUT/$log.log" "cd /home/user/tt && OZ_NOVERIFY=1 $(timing_env) $env $*"
  rc=$RI_RC
  line=$(grep -oE -- "$pat" "$OUT/$log.log" | head -1)
  if [ "$rc" -ne 0 ] || [ -z "$line" ]; then
    fail=1; line="FAILED (exit $rc${line:+, $line}, see $OUT/$log.log)"
  elif [ "$cooled" != 1 ]; then
    fail=1; line="FAILED（冷却未達：start $pre degC > TMAX $TMAX．値は $line）"
  fi
  printf "  %-${LABEL_W:-34}s %s%s  [GEMM 段の計時区間 %s MHz（TELEM）%s]\n" "$label" "$line" "$(uncooled_mark)" \
         "$(telem_clk "$OUT/$log.log")" "$(telem_lim "$OUT/$log.log")"
}
