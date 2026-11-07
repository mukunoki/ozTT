#!/bin/bash
# build_all.sh -- src/Makefile の全部の組（本番・比較対象・探針・検査・ホストの道具）を作り直し，ビルドの記録 src/build/BUILD_STAMP を書く
# コンテナの中で実行する： ./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/build_all.sh"
#   （OZ_PROVENANCE はホストで作った出どころの行．remeasure_all.sh はこうして呼ぶ．無ければコンテナの中の git で作る）
#
# BUILD_STAMP の 1 行目はビルドした時点の rev・dirty・tt-metal・コンテナのイメージ，続く行は src/build/ の各バイナリの
# sha256（provenance.sh の write_build_stamp）．測定スクリプトは測る前に provenance_check で，スタンプの rev が
# 今の rev と同じで各バイナリが差し替わっておらず，tt-metal とイメージが今のものと同じことを確かめ，外れれば
# 測らない（古いバイナリのまま STEPS="cool" と流しても，今の HEAD の測定として記録されないように．イメージを
# 入れ替えた後の一部の段の測り直しも止まる）．イメージはホストでしか読めないので，OZ_PROVENANCE を渡さずに
# ビルドするとスタンプの image が unknown になり，測定スクリプトは測らない．make の前に古いスタンプを
# BUILD_STAMP.prev へ退けるので，make が失敗すればスタンプは無く，測定は止まる．
# スタンプは今の環境（rev・tt-metal・イメージ・計算機）を書くので，そこに載るバイナリはこの実行で作ったものでなければ
# ならない．make -B で全部を作り直す：時刻だけで判断すると，別の計算機から時刻を保って写した src/build や，
# イメージを入れ替える前のバイナリを「最新」とみなして残し，今の環境のビルドとして記録してしまう（src/Makefile の
# 前提の印もコンパイラ・フラグ・tt-metal の変化で作り直すが，イメージの中身のすべては見ない）．所要は 1 分足らず．
# write_build_stamp は src/build の実行ファイルをすべて載せるので，今の Makefile のターゲットでない実行ファイル
# （ターゲットを改名・削除する前のバイナリ）が残っていれば，スタンプを書かずに止める（make はそれを作り直さない）．
set -eu
cd "$(dirname "$0")/.."          # リポジトリ直下へ
. ./scripts/provenance.sh        # write_build_stamp
mkdir -p src/build
[ -e src/build/BUILD_STAMP ] && mv -f src/build/BUILD_STAMP src/build/BUILD_STAMP.prev
echo ">>> building all src/ targets"   # デバイスカーネルは実行時 JIT なのでここでは作らない
( cd src && make -B -j"$(nproc)" >/dev/null )
targets=" $(make -s -C src print-targets | tr '\n' ' ') "
extra=""
for f in src/build/*; do
  [ -f "$f" ] && [ -x "$f" ] || continue
  case $targets in *" ${f##*/} "*) ;; *) extra="$extra ${f##*/}" ;; esac
done
if [ -n "$extra" ]; then
  echo "ERROR: src/build に今の src/Makefile のターゲットでない実行ファイルがある:$extra（この実行で作ったものではないので，" >&2
  echo "       退避してから流し直す．例 D=~/.claude/trash_bin/\$(date +%Y%m%d); mkdir -p \"\$D\"; mv src/build/<名前> \"\$D/\"）" >&2
  exit 1
fi
write_build_stamp
echo ">>> all built  ($(head -1 src/build/BUILD_STAMP))"
