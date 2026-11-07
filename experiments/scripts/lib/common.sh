# common.sh -- scripts/ のシェルが共有する小さな部品（source して使う．単独では実行しない）
# ---------------------------------------------------------------------------
#     . ./scripts/lib/common.sh       # リポジトリ直下へ cd してから（コンテナの中でも同じ）
#
# 読み込んでも何も実行しない（関数の定義だけ）．デバイスの排他・出どころの確認は measure_lib.sh・provenance.sh が持つ．
# 定数の表（lib/constants.tsv）の読み手 const_get も一緒に読み込む（lib/consts.sh）．
. "$(dirname "${BASH_SOURCE[0]}")/consts.sh"

# oz_env_clean [残す名前…]
#   親の環境から受け継いだ尾崎のつまみ（OZ_*・OZ2_*）とホストの道具（tools/）のつまみ（TWOSUM_PROBE_*・TFMUL_*・AWE_DEBUG）を
#   外す．run.sh がコンテナへ渡す OZ_TELEM_DEV（テレメトリを読むカード）と OZ_IMAGE_ID（走らせたイメージ）はいつも残し，
#   引数に与えた名前も残す．固定の構成・既定の入力で測る検査が，親のシェルに残った値（自動決定 OZ_FP64，入力の作り方
#   OZ_PHI・OZ_CONST，検証を省く OZ_NOVERIFY，判定の基準 OZ_MIN_BITS，診断のつまみ OZ_NO_*，道具の規則を誤った版に替える
#   TWOSUM_PROBE_ABS・_FAST など）で別の構成・別の入力になったまま通らないように．外したい名前を手で並べると，
#   つまみを足したときに漏れる
oz_env_clean() {
  local _v _k _keep
  for _v in $(compgen -e | grep -E '^(OZ2?_|TWOSUM_PROBE_|TFMUL_|AWE_DEBUG$)'); do
    _keep=0
    case $_v in OZ_TELEM_DEV|OZ_IMAGE_ID) _keep=1 ;; esac
    for _k in "$@"; do [ "$_v" = "$_k" ] && _keep=1; done
    [ "$_keep" = 1 ] || unset "$_v"
  done
}

# run_capture 出力の変数 終了コードの変数 [--] コマンド…
#   コマンドの標準出力と終了コードを，パイプを通さずに 2 つの変数へ置く（標準エラーは捨てる）．
#   `v=$(cmd | awk …)` は cmd の終了コードを捨て，RESULT を出した後に非零で終わった回も値だけで通してしまう．
#   pipefail を立てても，awk が先に exit すると cmd の SIGPIPE が残るので，出力を取ってから読む．
#   環境変数を与えるときは `run_capture o rc -- env OZ_CONST=1 ./src/build/…` とする
#   局所変数は呼び出し側の変数の名前と重ならない名前にする（重なると printf -v が呼び出し側でなくここの変数に書く）
run_capture() {
  local __rcap_o=$1 __rcap_r=$2 __rcap_out __rcap_rc; shift 2
  [ "${1:-}" = -- ] && shift
  if __rcap_out=$("$@" 2>/dev/null); then __rcap_rc=0; else __rcap_rc=$?; fi
  printf -v "$__rcap_o" '%s' "$__rcap_out"
  printf -v "$__rcap_r" '%s' "$__rcap_rc"
}

# result_field 種類 列の名前
#   ドライバの RESULT 行（CONFIG の種類 ozaki_full・ozaki2_full・gemm_bf16 など）の列 名前 が，, で分けて何列目か（RESULT が
#   1 列目）と，その行の 2 列目の種類の印（無ければ -）を「列の番号 印」の 1 行で出す．列の並びの正本は lib/constants.tsv の
#   RESULT_<種類>（oz_logs.py の RESULT_SCHEMA も同じ行から作る）で，列を番号で書き下さない（列を足す・並べ替えたときに
#   黙って別の値を読まないように）．種類・名前が表に無ければ ERROR を出して 2
result_field() {
  local row i=0 w col=""
  row=$(const_get "RESULT_$1") || return 2
  set -- "$1" "$2" $row
  local tag=$3 kind=$1 name=$2; shift 3
  for w in "$@"; do
    i=$((i + 1))
    [ "${w%%:*}" = "$name" ] && { col=$i; break; }
  done
  if [ -z "$col" ]; then
    echo "ERROR: lib/constants.tsv の RESULT_$kind に列 $name が無い" >&2
    return 2
  fi
  [ "$tag" = - ] || col=$((col + 1))
  printf '%s %s\n' $((col + 1)) "$tag"
}

# result_value 出力 種類 列の名前   出力の最初の RESULT,<種類の印>, 行の列 名前 の値（無ければ何も出さない）．
#   種類の印の無い RESULT 行（素の GEMM など）は RESULT,<数>, で始まる行を読む
result_value() {
  local f c tag
  f=$(result_field "$2" "$3") || return 2
  read -r c tag <<<"$f"
  if [ "$tag" = - ]; then
    printf '%s\n' "$1" | awk -F, -v c="$c" '/^RESULT,[0-9]/ { print $c; exit }'
  else
    printf '%s\n' "$1" | awk -F, -v c="$c" -v re="^RESULT,$tag," '$0 ~ re { print $c; exit }'
  fi
}

# result_bits 出力   尾崎の実行の出力（RESULT,ozaki_full,… か RESULT,ozaki2_full,…）の最初の RESULT 行から実効ビット数
#   （小数第 2 位まで．列 bits）を出す．無ければ何も出さない．effective ~X bit の表示（小数第 1 位）とは別のもの
result_bits() {
  local f1 f2 c1 c2 t1 t2
  f1=$(result_field ozaki_full bits) && f2=$(result_field ozaki2_full bits) || return 2
  read -r c1 t1 <<<"$f1"; read -r c2 t2 <<<"$f2"
  printf '%s\n' "$1" | awk -F, -v c1="$c1" -v r1="^RESULT,$t1," -v c2="$c2" -v r2="^RESULT,$t2," '
    $0 ~ r1 { print $c1; exit } $0 ~ r2 { print $c2; exit }'
}

# verify_state 出力   尾崎の実行の出力の検証の結果（VERIFY,ozaki_full,… の pass・fail・skipped．無ければ none）
verify_state() {
  local v
  v=$(printf '%s\n' "$1" | sed -n 's/^VERIFY,ozaki2\{0,1\}_full,\([a-z]*\).*/\1/p' | head -1)
  printf '%s\n' "${v:-none}"
}

# accuracy_verdict 出力 終了コード   尾崎の実行を OZ_MIN_BITS=<基準> で流した 1 回の判定を 1 語で出す．
#   pass    終了コード 0・VERIFY が pass・Test Passed（プログラムが丸めずに基準と比べて通した）
#   fail    終了コード 1・VERIFY が fail（基準を割った．値はある）
#   reject  終了コード 2 で RESULT 行が無い（入口の検査がその構成を組めないとして止めた．例：INT8 は K ブロック分割に
#           未対応なので，K が大きく p も大きい構成を拒む）
#   error   それ以外（検証を省いた・起動できない・クラッシュ・RESULT を出した後に落ちた・終了コードと VERIFY が食い違う）
#   判定はプログラム自身の検証に任せる：RESULT・Verify 行の精度は 2 桁に丸めた表示なので，基準の近くで食い違う
accuracy_verdict() {
  local out=$1 rc=$2 v
  v=$(verify_state "$out")
  if [ "$rc" = 0 ] && [ "$v" = pass ] && printf '%s\n' "$out" | grep -q '^Test Passed'; then echo pass
  elif [ "$rc" = 1 ] && [ "$v" = fail ]; then echo fail
  elif [ "$rc" = 2 ] && ! printf '%s\n' "$out" | grep -q '^RESULT,'; then echo reject
  else echo error; fi
}

# require_number 名前 値 下限 上限 [int]
#   値の全体が 0 以上の数（int を付ければ整数）で，下限以上・上限以下か．外れれば理由を出して 1（呼び出し側が止める）．
#   形は値の全体に当てる（grep は行ごとに当てるので，改行を含む値 $'1\njunk' がどれか 1 行の数で通ってしまう）．
#   比べるのは awk の浮動小数点（bash の [ -ge ] は 64 bit を超える整数で誤り，上限が働かなくなる）
require_number() {
  local name=$1 v=$2 lo=$3 hi=$4 kind=${5:-}
  local re='^[0-9]+(\.[0-9]+)?$'
  [ "$kind" = int ] && re='^[0-9]+$'
  if ! [[ $v =~ $re ]]; then
    echo "ERROR: $name=$v は$([ "$kind" = int ] && echo ' 0 以上の整数' || echo ' 0 以上の数')でない" >&2
    return 1
  fi
  if ! awk -v v="$v" -v lo="$lo" -v hi="$hi" 'BEGIN { exit !(v + 0 >= lo + 0 && v + 0 <= hi + 0) }'; then
    echo "ERROR: $name=$v が範囲 [$lo, $hi] の外にある" >&2
    return 1
  fi
}

# 最後まで書けたログだけを置き換える規則（リポジトリが追跡するログ accuracy_logs/ を tee で直接上書きすると，途中で止まった
# ときに前の完全なログを失うため）．書くのは一時ファイル（.<名前>.tmp.<pid>．*.log でないので検査は読まない）で，置き換えるのは
#   (1) 一時ファイルへの書き込みが失敗していない（tee_replace・measure/cpu_dgemm_reference.sh．失敗した一時ファイルは中身の形が揃っていても使わない），
#   (2) 書いた中身の 1 行目が PROVENANCE 行で最後の行が EXIT,rc= 行（replace_log），
# の両方を満たすときだけ．満たさなければ前のログを残し，書きかけを <ログ>.incomplete に置いて 1（keep_log）．
# 流したコマンドの終了コードで決める条件は呼び出し側が先に見る（measure/cpu_dgemm_reference.sh は 0 か 1 だけを置き換える）

# keep_log 一時ファイル ログ 理由   置き換えずに前のログを残し，書きかけを <ログ>.incomplete に置いて 1 を返す
keep_log() {
  local tmp=$1 out=$2 why=$3
  if [ ! -e "$tmp" ]; then
    echo "ERROR: $why．$out は前のままにした（書きかけの一時ファイルは無い）" >&2
  elif mv -f "$tmp" "$out.incomplete"; then
    echo "ERROR: $why．$out は前のままにし，書きかけを $out.incomplete に置いた" >&2
  else
    echo "ERROR: $why．$out は前のままにした（書きかけ $tmp を $out.incomplete へ移せなかった）" >&2
  fi
  return 1
}

# replace_log 一時ファイル ログ   書き終えた一時ファイルを，上の (2) を満たすときだけログへ置き換える（mv）．
#   満たさなければ（測る前の確認で止まった・途中で止められた）keep_log
replace_log() {
  local tmp=$1 out=$2 why=""
  if [ ! -s "$tmp" ]; then why="何も書かれていない"
  elif ! head -1 "$tmp" | grep -q '^PROVENANCE,'; then why="1 行目が PROVENANCE 行でない（測り始める前に止まった）"
  elif ! tail -1 "$tmp" | grep -q '^EXIT,rc='; then why="最後の行が EXIT 行でない（途中で止まった）"
  fi
  [ -n "$why" ] && { keep_log "$tmp" "$out" "$why"; return 1; }
  mv -f "$tmp" "$out" && return 0
  echo "ERROR: $tmp を $out へ置き換えられない" >&2
  return 1
}

# tee_replace ログ   標準入力を画面へ出しながら一時ファイルへ書き，入力が終わったら上の規則でログへ置き換える．
#   `… | tee ログ` の代わりに使う（pipefail と合わせれば，流したコマンドの失敗・書き込みの失敗・置き換えなかったことの
#   どれも終了コードに出る）
tee_replace() {
  local out=$1 tmp rc=0
  tmp="$(dirname "$out")/.$(basename "$out").tmp.$$"
  tee "$tmp" || rc=$?
  if [ "$rc" != 0 ]; then
    keep_log "$tmp" "$out" "一時ファイル $tmp への書き込み（tee）が失敗した（終了コード $rc．空き容量・権限）"
    return 1
  fi
  replace_log "$tmp" "$out"
}
