#!/bin/bash
# run.sh -- Blackhole を繋いだ tt-metalium コンテナの中でコマンドを実行する
# ---------------------------------------------------------------------------
# このスクリプトが置かれたリポジトリ（git worktree でもよい）がコンテナ内の /home/user/tt に
# マウントされる．バイナリはビルドした src の絶対パス（ここで作れば /home/user/tt/src/．src/common/oz_paths.hpp）から
# カーネルを読むので，別の worktree から呼べばその worktree のソースが使われる．使い方：
#     ./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/build_all.sh"
#     ./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/study/measure_paper.sh"
#     ./scripts/run.sh "bash /home/user/tt/scripts/selftest.sh"
# ビルドと測定には，ホストで作った出どころの行を OZ_PROVENANCE で渡す（コンテナの中ではイメージが読めず，
# worktree では rev も読めないので，渡さないと測定スクリプトが測る前の確認で止まる．provenance.sh の注記）．
# ホストの環境はコンテナへ渡らない．渡すのは下のデバイスの番号（OZ_TELEM_DEV．TT_DEVICE）と，走らせたイメージの ID
# （OZ_IMAGE_ID）だけなので，渡したい変数はコマンドの文字列に書く（測定スクリプトは計時の方式 OZ_TIMING なども
# 文字列に書いて渡す．measure_lib.sh の timing_env）．コンテナの中の bash は -l でマウントしたホームの .profile
# （と，それが読む .bashrc）を読むので，コマンドの前に lib/common.sh の oz_env_clean で尾崎のつまみ（OZ_*・OZ2_*）と
# ホストの道具のつまみ（TWOSUM_PROBE_*・TFMUL_*・AWE_DEBUG）を外す（ホストで未設定のつまみが .profile の値のまま
# 測定に入らないように．外す範囲の正本は oz_env_clean の 1 か所．文字列の中で後から与える代入は残る）．
# 2 つの値は .profile が触らない名前（RUNSH_TELEM_DEV・RUNSH_IMAGE_ID）でコンテナへ渡し，外した後に OZ_TELEM_DEV・
# OZ_IMAGE_ID へ置く（.profile が同じ名前を export しても run.sh の値が勝つ）．
#
# 計算機ごとに違いうるものはホストの環境変数で与える（既定はこの研究で測ってきた計算機の配置．
# docs/remeasure_other_machine.md）：
#   TT_DEVICE      コンテナへ渡す Blackhole のデバイスファイル（既定 /dev/tenstorrent/0．provenance.sh の PROV_DEVICE）．
#                  コンテナの中でも同じパスに見え，tt-metal はそれだけを見つけて使う．カードが複数あれば測るカードを選ぶ．
#                  番号（/dev/tenstorrent/<番号>）は OZ_TELEM_DEV としてもコンテナへ渡し，計時区間のテレメトリも同じカードから読む
#   TT_HUGEPAGES   1 GiB の巨大ページの hugetlbfs（既定 /dev/hugepages-1G）．コンテナの /dev/hugepages-1G へマウントする
#                  （tt-metal はその場所を探すので，コンテナ側のパスは変えない）
#   TT_HOME_MOUNT  コンテナの /home/user（=$HOME）にマウントするホストのディレクトリ（既定 ホストの $HOME）．
#                  カーネルの JIT キャッシュ（~/.cache/tt-metal-cache．legacy/measure.sh は ~/.cache/tt-metal-cache-legacy）
#                  がここに残る．コンテナは bash -l で始まるので，このディレクトリの .profile（と，それが読む .bashrc）も
#                  コンテナの中で読まれる
#   TT_METAL_SRC   ビルド済みの tt-metal の木（既定 ~/tt-metal-src．provenance.sh の PROV_METAL_SRC）．/tt-metal へマウントする
#   TT_IMAGE       コンテナのイメージ（provenance.sh の PROV_IMAGE）．排他を取った後にそのタグのイメージの ID を 1 度読み，
#                  その ID で走らせる（排他を待つあいだに podman pull などでタグが付け替わっても，出どころの行に記録した
#                  イメージと走らせるイメージが黙って食い違わないよう，ID を OZ_IMAGE_ID でコンテナへ渡し，コンテナの中の
#                  provenance_check が出どころの行の image と比べる）．イメージが無ければ（ID が読めなければ）走らせない
#   TT_DEVICE_LOCK デバイスの排他のロックファイル（device_lock.sh．既定はデバイスごとに /tmp/tt-blackhole-<番号>.lock）
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# イメージは provenance.sh の PROV_IMAGE（TT_IMAGE で差し替えられる）．出どころの行の image= と同じものを使う．
# tt-metal も同じく provenance.sh の PROV_METAL_SRC（下の -v），デバイスも PROV_DEVICE（出どころの行のボードの欄と同じ）
. "$REPO/scripts/provenance.sh"
IMG=$PROV_IMAGE
DEV=$PROV_DEVICE
HUGE="${TT_HUGEPAGES:-/dev/hugepages-1G}"
HOME_MNT="${TT_HOME_MOUNT:-$HOME}"
# 渡すものが無ければ排他を取る前に止める（podman や tt-metal の分かりにくい失敗より先に，どの変数を直すかを出す）
[ -c "$DEV" ] || { echo "ERROR: デバイス $DEV が無い（TT_DEVICE で Blackhole のデバイスファイルを与える．ls /dev/tenstorrent/）" >&2; exit 2; }
[ -d "$HUGE" ] || { echo "ERROR: 1 GiB の巨大ページ $HUGE が無い（TT_HUGEPAGES で hugetlbfs（pagesize=1G）のマウント先を与える）" >&2; exit 2; }
[ -d "$HOME_MNT" ] || { echo "ERROR: コンテナの /home/user にマウントする $HOME_MNT が無い（TT_HOME_MOUNT）" >&2; exit 2; }
[ -d "$PROV_METAL_SRC/build_Release/lib" ] || { echo "ERROR: ビルド済みの tt-metal $PROV_METAL_SRC/build_Release/lib が無い（TT_METAL_SRC．docs/remeasure_other_machine.md）" >&2; exit 2; }
# デバイスの排他（scripts/device_lock.sh）．別の実行がデバイスを使っていれば空くまで待つ．
# 冷ましてから測るスクリプトのように呼び出し側が既に持っていれば（TT_DEVICE_LOCK_HELD=1 で，受け継いだ fd 9 が
# そのロックを持っている）取り直さない．
. "$REPO/scripts/device_lock.sh"
device_lock || exit $?
# 走らせるイメージの ID（上の TT_IMAGE の注記．provenance.sh の _prov_image_id を一時的な失敗をやり直して読む）
IMG_ID=$(prov_try _prov_image_id) || { echo "ERROR: イメージ $IMG の ID が読めない（podman image inspect．イメージが無ければ podman pull すること）" >&2; exit 2; }
case $IMG_ID in *[!0-9a-f]*|'') echo "ERROR: イメージ $IMG の ID が読めない（${IMG_ID:-空}）" >&2; exit 2 ;; esac
# デバイスと巨大ページ，ホームとリポジトリ，tt-metal 本体をマウントして実行する．
# tt-metal は provenance.sh の PROV_METAL_SRC（ホストの TT_METAL_SRC．既定 ~/tt-metal-src）をマウントする．
# 出どころの metal= も同じ変数の木から取るので，記録した tt-metal と実行する tt-metal が食い違わない．
# コンテナの中の provenance.sh も /tt-metal を見るよう TT_METAL_SRC を渡す．
# ロックの fd 9 はコンテナへ渡さない（podman の補助プロセスが残ってロックを握り続けないように）．
# テレメトリ（src/common/telemetry.hpp．計時区間の AICLK などを sysfs の tenstorrent!<番号> から読む）が tt-metal の開く
# カードと同じカードを読むよう，デバイスファイルの番号を OZ_TELEM_DEV で渡す（sysfs はコンテナの中でも全カードが見える）
TELEM_DEV=${DEV##*/}
case "$TELEM_DEV" in (''|*[!0-9]*) TELEM_DEV="" ;; esac   # 番号でなければ渡さない（テレメトリはバイナリの既定のカード）
# コンテナの中で，.profile・.bashrc が export したつまみを外してから，run.sh が渡した 2 つを置く（上の注記）．
# tt-metal の場所（TT_METAL_SRC・TT_METAL_RUNTIME_ROOT・TT_METAL_HOME）・ARCH_NAME・HOME も .profile の後で置き直し，ほかの
# TT_METAL_*（watcher・dprint・遅い dispatch など．黙って遅くなる）が立っていれば止める（2026-11-04 の点検．別の計算機の
# ホームの .profile が -e で渡した値を上書きしても，出どころの metal= はホストの木のままで気づけないので）．
# lib/common.sh はマウントしたこの木のもの（読めなければコマンドを流さずに止める）
CLEAN='if ! . /home/user/tt/scripts/lib/common.sh; then echo "ERROR: run.sh: コンテナの中で scripts/lib/common.sh を読めない" >&2; exit 2; fi
oz_env_clean; unset OZ_TELEM_DEV OZ_IMAGE_ID
[ -z "$RUNSH_TELEM_DEV" ] || export OZ_TELEM_DEV=$RUNSH_TELEM_DEV
export OZ_IMAGE_ID=$RUNSH_IMAGE_ID; unset RUNSH_TELEM_DEV RUNSH_IMAGE_ID
export TT_METAL_SRC=/tt-metal TT_METAL_RUNTIME_ROOT=/tt-metal TT_METAL_HOME=/tt-metal ARCH_NAME=blackhole HOME=/home/user
x=$(env | sed -n "s/^\(TT_METAL_[A-Za-z0-9_]*\)=.*/\1/p" | grep -vxE "TT_METAL_(SRC|RUNTIME_ROOT|HOME)" | tr "\n" " ")
if [ -n "$x" ]; then echo "ERROR: run.sh: コンテナの中に tt-metal のつまみ $x が立っている（マウントしたホームの .profile・.bashrc か．TT_HOME_MOUNT に空のディレクトリを与える）" >&2; exit 2; fi
'
podman run --rm \
  --device="$DEV" \
  -e "RUNSH_TELEM_DEV=$TELEM_DEV" \
  -e "RUNSH_IMAGE_ID=$IMG_ID" \
  -v "$HUGE":/dev/hugepages-1G \
  -v "$HOME_MNT":/home/user \
  -v "$REPO":/home/user/tt \
  -v "$PROV_METAL_SRC":/tt-metal \
  -e TT_METAL_SRC=/tt-metal \
  -e TT_METAL_RUNTIME_ROOT=/tt-metal \
  -e TT_METAL_HOME=/tt-metal \
  -e ARCH_NAME=blackhole \
  -e HOME=/home/user \
  -w /home/user/tt \
  "$IMG_ID" \
  bash -lc "$CLEAN$*" 9>&-
