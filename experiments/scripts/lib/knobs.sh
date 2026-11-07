# knobs.sh -- scripts/lib/knobs.tsv（scripts/ のシェルと照合の Python が読む環境変数の正本）を読む部品と，表とスクリプトの検査
# ---------------------------------------------------------------------------
#     . ./scripts/lib/knobs.sh                       # source して使う（どこからでも．表の場所はこのファイルの隣で決める）
#     INHERIT_UNSET=$(knobs_names remeasure=drop) || exit 2
#     bash scripts/lib/knobs.sh --check              # 表の検査と，表とスクリプトの突き合わせ（0 で一致．selftest.sh・remeasure_all.sh）
# 列の意味は knobs.tsv の注記．読み込んでも何も実行しない（関数と表の場所の定義だけ）．
KNOBS_TSV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/knobs.tsv"
KNOBS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# _knobs_table   表を全部確かめてから「名前<TAB>kind<TAB>remeasure<TAB>read_by」を表の順に出す．表が読めない・見出しが違う・
#   5 列でない・空欄・kind と remeasure が決まった値でない・同じ名前が 2 行・行の数が注記「# rows=<n>」と違う，のどれかなら
#   ERROR を出して何も出さずに 2（lib/consts.sh の const_get と同じ規則．壊れた表から一部の名前だけを読まない）
_knobs_table() {
  if [ ! -r "$KNOBS_TSV" ]; then
    printf 'ERROR: 環境変数の表 %s を読めない\n' "$KNOBS_TSV" >&2
    return 2
  fi
  awk -F'\t' -v path="$KNOBS_TSV" '
    function bad(m) { printf "ERROR: 環境変数の表 %s %s\n", path, m > "/dev/stderr"; err = 1; exit 2 }
    /^# rows=[0-9]+$/ { if (rows == "") rows = substr($0, 8) + 0; next }
    /^#/ || /^$/ { next }
    !head { if ($0 != "name\tkind\tremeasure\tread_by\tmeaning") bad("の見出しが name・kind・remeasure・read_by・meaning でない: " $0); head = 1; next }
    { if (NF != 5) bad(sprintf("の %d 行目が 5 列でない: %s", NR, $0))
      for (i = 1; i <= 5; i++) if ($i == "") bad(sprintf("の %d 行目に空欄がある: %s", NR, $0))
      if ($1 !~ /^[A-Z][A-Z0-9_]*$/) bad(sprintf("の %d 行目の名前 %s が大文字の変数の名前でない", NR, $1))
      if ($2 !~ /^(knob|timing|container|pass|local|sys)$/) bad(sprintf("の %d 行目の kind=%s が knob・timing・container・pass・local・sys のどれでもない", NR, $2))
      if ($3 !~ /^(drop|keep|self|-)$/) bad(sprintf("の %d 行目の remeasure=%s が drop・keep・self・- のどれでもない", NR, $3))
      if ($1 in seen) bad("に " $1 " の行が 2 つある")
      seen[$1] = 1; n++; out = out $1 "\t" $2 "\t" $3 "\t" $4 "\n" }
    END {
      if (err) exit 2
      if (!head) bad("に見出し行が無い")
      if (rows == "") bad("に行の数の注記「# rows=<n>」が無い")
      if (n != rows) bad(sprintf("の名前の行が %d 行で，注記の rows=%d と違う（行を足した・消したなら注記も直す）", n, rows))
      printf "%s", out
    }' "$KNOBS_TSV"
}

# knobs_names 列=値…   条件（kind=… remeasure=… のすべて）に当たる名前を表の順に空白で区切って 1 行で出す．表が壊れていれば 2
knobs_names() {
  local t
  t=$(_knobs_table) || return 2
  awk -F'\t' -v conds="$*" '
    BEGIN { n = split(conds, c, " "); for (i = 1; i <= n; i++) { split(c[i], kv, "="); want[kv[1]] = kv[2] } col["kind"] = 2; col["remeasure"] = 3 }
    { ok = 1; for (k in want) if ($(col[k]) != want[k]) ok = 0; if (ok) printf "%s%s", (m++ ? " " : ""), $1 }
    END { print "" }' <<<"$t"
}

# knobs_check   表の検査（_knobs_table）と，表とスクリプトの突き合わせ．食い違いを 1 行ずつ出し，あれば 1，無ければ 0：
#   - scripts/ の直下・lib・tests・study の *.sh の注記でない行で，既定値つきの展開（${名前:-…}・${名前-…}・${名前:+…}・
#     ${名前+…}・${名前:=…}・${名前:?…}）で読む大文字の名前がすべて表にあり，そのファイルが read_by に挙がっている
#     （つまみを足して表に載せ忘れると，remeasure_all.sh が呼び出し元の値を外さずに引き継ぐ）
#   - 同じく *.py が os.environ.get("名前")・os.environ["名前"]・os.getenv("名前") で読む名前も
#   - read_by の各ファイルがあり，その名前が語として出る（kind=timing はその名前を knobs_names kind=timing で引く）
knobs_check() {
  local t bad=0 f rel n kind rem rb x
  t=$(_knobs_table) || return 2
  declare -A RB=()
  while IFS=$'\t' read -r n kind rem rb; do RB[$n]=",$rb,"; done <<<"$t"
  # 読む側から：既定値つきの展開と os.environ
  while IFS= read -r f; do
    rel=${f#"$KNOBS_SCRIPTS"/}
    case $f in
      *.sh) x=$(grep -vE '^[[:space:]]*#' "$f" | grep -oE '\$\{[A-Z][A-Z0-9_]*(:-|-|:\+|\+|:=|:\?)' | sed -E 's/^\$\{//; s/(:-|-|:\+|\+|:=|:\?)$//' | sort -u) ;;
      *.py) x=$(grep -vE '^[[:space:]]*#' "$f" | grep -oE "(environ\.get\(|environ\[|getenv\()[\"'][A-Z][A-Z0-9_]*[\"']" | sed -E "s/.*[\"']([A-Z0-9_]+)[\"']/\1/" | sort -u) ;;
    esac
    for n in $x; do
      if [ -z "${RB[$n]:-}" ]; then
        echo "  $rel が $n を読むが，lib/knobs.tsv に $n の行が無い"; bad=1
      else
        case ${RB[$n]} in *",$rel,"*) ;; *) echo "  $rel が $n を読むが，lib/knobs.tsv の $n の read_by（${RB[$n]//,/ }）に $rel が無い"; bad=1 ;; esac
      fi
    done
  done < <(find "$KNOBS_SCRIPTS" "$KNOBS_SCRIPTS/lib" "$KNOBS_SCRIPTS/tests" "$KNOBS_SCRIPTS/study" -maxdepth 1 -type f \( -name '*.sh' -o -name '*.py' \) | LC_ALL=C sort)
  # 表の側から：read_by のファイルにその名前が出るか（kind=timing の名前は，ファイルが表から knobs_names kind=timing で
  # 引いて間接に読むので，その引き方が出るか）
  while IFS=$'\t' read -r n kind rem rb; do
    for f in ${rb//,/ }; do
      if [ ! -f "$KNOBS_SCRIPTS/$f" ]; then echo "  lib/knobs.tsv の $n の read_by の $f が無い"; bad=1
      elif [ "$kind" = timing ]; then
        grep -q -- 'knobs_names kind=timing' "$KNOBS_SCRIPTS/$f" || { echo "  lib/knobs.tsv の $n（kind=timing）の read_by の $f が knobs_names kind=timing で表を引かない"; bad=1; }
      elif ! grep -qw -- "$n" "$KNOBS_SCRIPTS/$f"; then echo "  lib/knobs.tsv の $n の read_by の $f に $n が出ない"; bad=1; fi
    done
  done <<<"$t"
  return $bad
}

# 実行されたとき（source ではなく）
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    --check)
      if out=$(knobs_check); then
        echo "lib/knobs.tsv とスクリプトは一致（$(knobs_names | wc -w) 個の名前）"; exit 0
      else
        rc=$?
        [ "$rc" = 1 ] && printf 'lib/knobs.tsv とスクリプトの食い違い:\n%s\n' "$out"
        exit "$rc"
      fi ;;
    *) echo "使い方: bash scripts/lib/knobs.sh --check（ほかは source して knobs_names を使う）" >&2; exit 2 ;;
  esac
fi
