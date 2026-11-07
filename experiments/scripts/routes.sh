# routes.sh -- scripts/routes.tsv（16 経路の正本）を読む bash の部品（source して使う．単独では実行しない）
# ---------------------------------------------------------------------------
#     . ./scripts/routes.sh
#     cb() { echo "$r_label $r_binary $r_p"; }
#     routes_each cb scheme=I acc_as=-
#     set -- $(routes_args oz1bf16)          # ozaki1_bf16 10 9（スキーム II は ozaki2_int8 16 のようにバイナリと法の本数）
#     cb() { echo "$r_label $(routes_row_cmd 3520 1 3)"; }
#     routes_each_in "$OZ_MEAS_ORDER" cb      # 16 経路を測る順に．./src/build/ozaki1_bf16 3520 3520 3520 10 9 1 3 など
# 列の意味は routes.tsv の先頭の注記を見ること．
ROUTES_TSV="$(dirname "${BASH_SOURCE[0]}")/routes.tsv"

# スライス数 p（routes.tsv の p 列）を決める 2 つの規則（routes.tsv の p 列の注記）の基準と入力．diagnose/select_slice_count.sh（p を選び直す）と
# tests/check_adversarial.sh（今の p が規則 (ii) を満たすか）はこの変数で読み，scripts/README.md・study/measure_paper.sh の説明も
# この値で書く．値の正本は lib/constants.tsv の RULE_I_BITS・STRUCT_BITS・STRUCT_CONSTS・STRUCT_SIGNS（照合の probe_checks.py も
# 同じ行を読む）．読めなければ（表が壊れた・無い）ここで止める（空の基準で p を選ばない）
. "$(dirname "${BASH_SOURCE[0]}")/lib/consts.sh"
OZ_RULE_I_BITS=$(const_get RULE_I_BITS) || exit 2       # (i) 一様乱数 [-1,1) の入力で到達精度がこれを上回る［bit］
OZ_STRUCT_BITS=$(const_get STRUCT_BITS) || exit 2       # (ii) 構造のある入力でこれ以上［bit］（53 ちょうどでは余裕が無いので）
# (ii) の入力：OZ_CONST の定数 1/3・2/3・1/7・0.7・0.9・1-2^-53・0.99999999・0.75・1 を，符号の付け方 OZ_STRUCT_SIGNS
# （none は OZ_CONST_SIGN を立てない，col は OZ_CONST_SIGN=col）のそれぞれで
OZ_STRUCT_CONSTS=$(const_get STRUCT_CONSTS) || exit 2
OZ_STRUCT_SIGNS=$(const_get STRUCT_SIGNS) || exit 2

# routes_each 関数 [列=値 ...]
#   表の順に 1 行ずつ，各列を r_<列名>（r_key・r_label・r_p など）に置いて 関数 を呼ぶ．
#   列=値 を与えると，すべて一致する行だけを回す．関数は呼び出し元と同じシェルで動くので
#   fail=1 などの代入はそのまま残る．行を飛ばすときは関数から return する．
#   表は fd 3 で読むので，関数の中のコマンドの標準入力は奪わない．
#   表が読めない・見出し行が無い・行の列の数が見出しと違う・空欄がある（空欄は作らず - と書く）・行の数が注記の
#   「# rows=<n>」と違う（行が消えた・増えたのに気づかず，回す経路が黙って減らないように）・条件の列名が見出しに無い，
#   のどれかなら，1 行も回さずに ERROR を出して 2 を返す（表を全部確かめてから回すので，途中まで回してから止まることも，
#   1 経路も回さずに成功扱いになることもない）．呼び出し側は `routes_each ... || exit 2` で止めること．
#   区切りのタブは bash の IFS では空白の扱いになり，続いたタブ（空欄）が 1 つに潰れて後ろの列が前へずれるので，
#   空白でない区切り（\x1f）に置き換えてから分ける．
routes_each() {
  local _fn=$1 _line _i _ok _c _v _want="" _rows=() _head=() _f=(); shift
  if [ ! -r "$ROUTES_TSV" ]; then
    printf 'ERROR: 経路の表 %s を読めない\n' "$ROUTES_TSV" >&2
    return 2
  fi
  _want=$(sed -n 's/^# rows=\([0-9][0-9]*\)$/\1/p' "$ROUTES_TSV" | head -1)
  if [ -z "$_want" ]; then
    printf 'ERROR: 経路の表 %s に行の数の注記「# rows=<n>」が無い\n' "$ROUTES_TSV" >&2
    return 2
  fi
  # 1 回目：表を全部読んで確かめる
  while IFS= read -r _line <&3; do
    case "$_line" in ''|'#'*) continue ;; esac
    IFS=$'\x1f' read -r -a _f <<<"${_line//$'\t'/$'\x1f'}"
    if [ ${#_head[@]} -eq 0 ]; then _head=("${_f[@]}"); continue; fi
    _rows+=("$_line")
    _ok=1; for _v in "${_f[@]}"; do [ -n "$_v" ] || _ok=0; done
    if [ ${#_f[@]} -ne ${#_head[@]} ] || [ $_ok = 0 ]; then
      printf 'ERROR: 経路の表 %s の %d 行目の列が %d 個か空欄がある（見出しは %d 個．空欄は - と書く）: %s\n' \
        "$ROUTES_TSV" ${#_rows[@]} ${#_f[@]} ${#_head[@]} "$_line" >&2
      return 2
    fi
  done 3<"$ROUTES_TSV"
  if [ ${#_head[@]} -eq 0 ]; then
    printf 'ERROR: 経路の表 %s に見出し行が無い\n' "$ROUTES_TSV" >&2
    return 2
  fi
  if [ ${#_rows[@]} != "$_want" ]; then
    printf 'ERROR: 経路の表 %s の経路の行が %d 行で，注記の rows=%s と違う（行を足した・消したなら注記も直す）\n' \
      "$ROUTES_TSV" ${#_rows[@]} "$_want" >&2
    return 2
  fi
  for _c in "$@"; do
    case " ${_head[*]} " in *" ${_c%%=*} "*) ;; *)
      printf 'ERROR: routes_each の条件 %s の列が経路の表 %s の見出しに無い\n' "$_c" "$ROUTES_TSV" >&2
      return 2 ;;
    esac
  done
  # 2 回目：表の順に回す
  local "${_head[@]/#/r_}"
  for _line in "${_rows[@]}"; do
    IFS=$'\x1f' read -r -a _f <<<"${_line//$'\t'/$'\x1f'}"
    for _i in "${!_head[@]}"; do printf -v "r_${_head[$_i]}" '%s' "${_f[$_i]}"; done
    _ok=1
    for _c in "$@"; do
      _i="r_${_c%%=*}"; [ "${!_i}" = "${_c#*=}" ] || { _ok=0; break; }
    done
    if [ $_ok = 1 ]; then "$_fn"; fi
  done
  return 0   # 最後の行が条件から外れても呼び出し元の終了コードを汚さない
}

# routes_args ログ名の幹（oz1bf16・oz2int8 など．routes.tsv の log 列）
#   その経路のバイナリと既定の引数を 1 行で出す：スキーム I は「バイナリ p L」，II は「バイナリ s」（法の本数の要求値）．
#   表を読まずに経路の既定の構成を書き下すと，表の p・法の本数を変えたときに直し忘れた所だけ黙って既定でない構成で
#   測るので，既定の構成で回す所はここから引く（diagnose/probes.sh の設計の比較・measure/exponent_adaptive.sh など）．
#   表が読めなければ 2，その経路の行が無ければ 1 を返す（呼び出し側は `a=$(routes_args oz1bf16) || exit 2`）
#   1 行の形は routes_row_args（routes_each の関数の中で，今の行について同じ 1 行を出す）
routes_row_args() { if [ "$r_scheme" = I ]; then printf '%s %s %s\n' "$r_binary" "$r_p" "$r_L"; else printf '%s %s\n' "$r_binary" "$r_s"; fi; }
routes_args() {
  local _out=""
  _ra() { _out=$(routes_row_args); }
  routes_each _ra log="$1" || return 2
  if [ -z "$_out" ]; then
    printf 'ERROR: 経路の表 %s に log=%s の行が無い\n' "$ROUTES_TSV" "$1" >&2
    return 1
  fi
  printf '%s\n' "$_out"
}

# routes_row_cmd N ウォームアップ 計時
#   routes_each の関数の中で，今の行の経路を M=N=K=N の立方・既定の構成（表の p・L か法の本数）で流すコマンドを 1 行で出す
#   （"./src/build/<バイナリ> N N N <p L か法の本数> <ウォームアップ> <計時>"）．16 経路を全部回すスクリプト（measure/performance.sh・
#   measure/accuracy.sh・study/measure_paper.sh・selftest.sh）は経路を書き下さずにこれで組む（表の p・法の本数を変えたとき，
#   4 か所の写しを直し忘れて既定でない構成で測ることが無いように）．呼び出し側はコマンドを語に分けて渡す（$(…) を引用符で囲まない）
routes_row_cmd() {
  local _a; _a=$(routes_row_args)
  printf './src/build/%s %s %s %s %s %s %s\n' "${_a%% *}" "$1" "$1" "$1" "${_a#* }" "$2" "$3"
}

# OZ_MEAS_ORDER   16 経路を全部測るスクリプト（measure/performance.sh・measure/accuracy.sh・study/measure_paper.sh）が回す経路の順（表の key）．
#   表の並び（論文の表の並び）と違い，I-INT8 が I-FP8 の前で，II は INT8・FP8・BFP8 から始まる．論文の値を測ってきた
#   measure/performance.sh の順で，1 項目ずつ冷ましてから測っても，前の項目の熱が基板・電源に残る分は順に依りうるので変えない
#   （表の並びに替えるなら全部の段を測り直す）．到達精度は順に依らないが，同じ並びにそろえておく
OZ_MEAS_ORDER="I-bf16 I-fp16 I-fp32 I-bfp8 I-int8 I-fp8 I-bfp4 I-bfp2 II-int8 II-fp8 II-bfp8 II-bf16 II-fp16 II-fp32 II-bfp4 II-bfp2"

# routes_each_in 順 関数
#   routes_each と同じく各行の列を r_<列名> に置いて関数を呼ぶが，「順」（表の key を空白で並べたもの．OZ_MEAS_ORDER など）の
#   とおりに回す．順が表の全経路をちょうど 1 回ずつ並べたものでなければ（表に無い key・重複・欠け），1 行も回さずに ERROR を
#   出して 2 を返す（経路を黙って飛ばす・2 回測ることが無いように）．呼び出し側は `routes_each_in "$OZ_MEAS_ORDER" 関数 || exit 2`
routes_each_in() {
  local _order=$1 _fn=$2 _k _all="" _seen=" "
  _rei_key() { _all="$_all $r_key"; }
  routes_each _rei_key || return 2
  for _k in $_order; do
    case "$_all " in *" $_k "*) ;; *)
      printf 'ERROR: 回す順の %s が経路の表 %s に無い\n' "$_k" "$ROUTES_TSV" >&2; return 2 ;;
    esac
    case "$_seen" in *" $_k "*)
      printf 'ERROR: 回す順に %s が 2 回ある\n' "$_k" >&2; return 2 ;;
    esac
    _seen="$_seen$_k "
  done
  for _k in $_all; do
    case "$_seen" in *" $_k "*) ;; *)
      printf 'ERROR: 経路の表 %s の %s が回す順に無い\n' "$ROUTES_TSV" "$_k" >&2; return 2 ;;
    esac
  done
  for _k in $_order; do routes_each "$_fn" key="$_k" || return 2; done
  return 0
}
