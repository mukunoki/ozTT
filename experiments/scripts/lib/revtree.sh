# revtree.sh -- 測る rev のソースの置き場所を名前で引く部品（ホストで source して使う．単独では実行しない）
# ---------------------------------------------------------------------------
#     . ./scripts/lib/revtree.sh      # リポジトリ直下へ cd してから
#
# 読み込んでも何も実行しない（関数の定義だけ）．リポジトリの履歴を読むので，コンテナの中では使わない
# （コンテナの中で動く検査が読む lib/common.sh と分けてある）．

# rev_path rev 名前 [接頭辞…]
#   rev の木でファイル名がちょうど「名前」のファイルのパスを 1 つ出す（接頭辞を与えればその下だけを探す．例 src tools）．
#   0 個か 2 個以上なら理由を出して 1．rev が - なら作業木（追跡しているファイルと，無視しない追跡外のファイル）．
#   測る rev のソースを固定のパス（src/common/host_eval.hpp など）で読むと，置き場所を変えた rev で黙って読めなくなるので，
#   rev ごとに名前で引く（scripts/README.md「ログの出どころ」）．照合の Python の側は lib/srcrev.py の src_path が同じ考えで
#   引く（当て方は少し違う：こちらはファイル名に当て，src_path は「common/host_eval.hpp」のようなパスの末尾にも当てる）
rev_path() {
  local rev=$1 name=$2 hits n; shift 2
  if [ "$rev" = - ]; then hits=$(git ls-files --cached --others --exclude-standard 2>/dev/null) || hits=""
  else hits=$(git ls-tree -r --name-only "$rev" 2>/dev/null) || hits=""; fi
  hits=$(printf '%s\n' "$hits" | awk -F/ -v b="$name" -v pre="$*" '
    $NF != b { next }
    pre == "" { print; next }
    { n = split(pre, p, " "); for (i = 1; i <= n; i++) if (index($0, p[i] "/") == 1) { print; next } }')
  n=$(printf '%s' "$hits" | grep -c .)
  if [ "$n" != 1 ]; then
    echo "ERROR: rev ${rev:0:12} の木に ${*:+$* の下の }$name が $n 個ある（ちょうど 1 個のはず${hits:+：$(echo $hits)}）" >&2
    return 1
  fi
  printf '%s\n' "$hits"
}
