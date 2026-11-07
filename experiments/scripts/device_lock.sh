# device_lock.sh -- Blackhole を 1 度に 1 つの実行だけが使うための排他（source して使う）
# ---------------------------------------------------------------------------
#     . ./scripts/device_lock.sh; device_lock || exit $?
#
# 同じデバイスを 2 つのコンテナが同時に開くと，片方が落ちるか，落ちなくても測定値が互いの負荷で
# 壊れる（別の worktree や別の作業者が同じ Blackhole を使うことがある）．flock(1) で 1 つのロック
# ファイルを取り合い，空くまで待つ．
#
#   run.sh             コンテナを 1 回走らせるあいだ取る．
#   measure_lib.sh     冷ましてから測るスクリプト（measure/performance.sh など）は，読み込んだ時点で取り，
#                      終わるまで持ち続ける．冷却待ちのあいだに別の実行がチップを温めると，
#                      COOLCOND の cooled=1 が嘘になるため．
#   remeasure_all.sh   全段を通して持つ．
# 取ったシェルは TT_DEVICE_LOCK_HELD=1 を環境へ置き，その子の run.sh は取り直さない（同じファイルを
# 別に開いて flock すると，自分が持っているロックを待って止まる）．ロックは fd 9 に開いたファイルに
# 掛かり，取ったシェルが終われば外れる．
# 取り直さないのは，TT_DEVICE_LOCK_HELD がちょうど 1 で，かつ受け継いだ fd 9 がこのロックファイルを指し，
# その fd でロックが取れている（flock -n 9 が通る．同じ開いたファイルを親と共有していれば，親が持つロックを
# そのまま取り直せる）ときだけである．値が 1 以外（=0 など）や，環境に 1 が残っているだけで fd 9 が別の
# ファイル・閉じている・別の実行が持っているときは，普通に取る（変数が残っただけで排他を外さないように）．
#
#   TT_DEVICE_LOCK       ロックファイル（既定 /tmp/tt-blackhole-<番号>.lock．<番号> は使うデバイスファイル TT_DEVICE
#                        （既定 /dev/tenstorrent/0．run.sh がコンテナへ渡すもの）の名前で，既定のデバイスなら
#                        /tmp/tt-blackhole-0.lock．worktree をまたいで共通．カードが複数ある計算機では，別のカードを
#                        使う実行どうしは待ち合わせない）．/tmp を別の作業者と共有しない計算機（systemd の PrivateTmp など）や
#                        別の作業者が作ったファイルに書けない計算機では，全員が書ける場所を与える
#   TT_DEVICE_LOCK_WAIT  空くのを待つ上限［s］（既定 86400）．0 なら待たずに失敗する
device_lock() {
  local f="${TT_DEVICE_LOCK:-/tmp/tt-blackhole-$(basename "${TT_DEVICE:-/dev/tenstorrent/0}").lock}" w="${TT_DEVICE_LOCK_WAIT:-86400}" fd9 want
  if [ "${TT_DEVICE_LOCK_HELD:-}" = 1 ]; then
    fd9=$(readlink /proc/self/fd/9 2>/dev/null)
    want=$(readlink -f -- "$f" 2>/dev/null)
    if [ -n "$fd9" ] && [ "$fd9" = "$want" ] && flock -n 9; then
      return 0
    fi
    echo "注意: TT_DEVICE_LOCK_HELD=1 だが，fd 9 がロック $f を持っていない（${fd9:-閉じている}）．取り直す" >&2
  fi
  unset TT_DEVICE_LOCK_HELD
  if ! exec 9>>"$f"; then
    echo "ERROR: ロックファイル $f を開けない" >&2
    return 2
  fi
  if ! flock -n 9; then
    echo "デバイスは別の実行が使用中（$f）．空くまで待つ（最長 ${w} s）" >&2
    if ! flock -w "$w" 9; then
      echo "ERROR: ${w} s 待ってもデバイスが空かない（$f）" >&2
      return 75
    fi
  fi
  export TT_DEVICE_LOCK_HELD=1
}
