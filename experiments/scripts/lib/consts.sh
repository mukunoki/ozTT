# consts.sh -- scripts/lib/constants.tsv（測る側と検査する側が同じ値を使う定数の正本）を読む bash の部品（source して使う．
#              単独では実行しない）
# ---------------------------------------------------------------------------
#     . ./scripts/lib/consts.sh              # どこからでも（表の場所はこのファイルの隣で決める．コンテナの中でも同じ）
#     TMAX_OK=$(const_get APPROVED_TMAX) || exit 2
# 値の意味は constants.tsv の注記．Python の読み手は lib/consts.py．lib/common.sh と provenance.sh がこれを読み込む
# （provenance.sh は run.sh・device_lock.sh と一緒に旧 rev の木へ写されるので，写すときはこのファイルと表も写す．
# legacy/measure.sh の overlay_spec）．
# 読み込んでも何も実行しない（関数と表の場所の定義だけ）．
CONSTS_TSV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/constants.tsv"

# const_get 名前
#   表の 名前 の行の値を 1 行で出す．表が読めない・見出し（name<TAB>value）が無い・2 列でない行・空の値・同じ名前が 2 行・
#   名前の行の数が注記「# rows=<n>」と違う・名前が無い，のどれかなら ERROR を出して何も出さずに 2 を返す（表を全部確かめて
#   から出すので，壊れた表から一部の値だけを読むことが無い．lib/consts.py と同じ規則）．
#   呼び出し側は `v=$(const_get 名前) || exit 2` のように止める（空の値のまま測り始めない）
const_get() {
  if [ ! -r "$CONSTS_TSV" ]; then
    printf 'ERROR: 定数の表 %s を読めない\n' "$CONSTS_TSV" >&2
    return 2
  fi
  awk -F'\t' -v want="$1" -v path="$CONSTS_TSV" '
    function bad(msg) { printf "ERROR: 定数の表 %s %s\n", path, msg > "/dev/stderr"; err = 1; exit 2 }
    /^# rows=[0-9]+$/ { if (rows == "") rows = substr($0, 8) + 0; next }
    /^#/ || /^$/ { next }
    !head { if ($0 != "name\tvalue") bad("の見出しが name<TAB>value でない: " $0); head = 1; next }
    { if (NF != 2 || $1 == "" || $2 == "") bad(sprintf("の %d 行目が 名前<TAB>値 の 2 列でない（値を空にしない）: %s", NR, $0))
      if ($1 in val) bad("に " $1 " の行が 2 つある")
      val[$1] = $2; n++ }
    END {
      if (err) exit 2
      if (!head) bad("に見出し行が無い")
      if (rows == "") bad("に行の数の注記「# rows=<n>」が無い")
      if (n != rows) bad(sprintf("の名前の行が %d 行で，注記の rows=%d と違う（行を足した・消したなら注記も直す）", n, rows))
      if (!(want in val)) bad("に " want " の行が無い")
      print val[want]
    }' "$CONSTS_TSV"
}
