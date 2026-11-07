# provenance.sh -- 測定ログの先頭に置く PROVENANCE 行を作り，ビルドの記録を書く・確かめる（source しても実行してもよい）
# ---------------------------------------------------------------------------
#     . ./scripts/provenance.sh; provenance_line >"$log"      # ログの 1 行目として書く
#     OZ_PROVENANCE=$(bash scripts/provenance.sh)              # 1 行だけ得る
#     provenance_check || exit 2                               # 測る前：今の src/build が今の rev のビルドか
#     write_build_stamp                                        # build_all.sh が make の直後に呼ぶ
#     bin_line ./src/build/ozaki1_bf16 ... >>"$log"            # 実行するバイナリの BINARY 行
#
# 形式（key=value を , でつなぐ．検査側は key で読むので，欄は後ろへ足してよい）：
#     PROVENANCE,rev=…,dirty=…,date=…,metal=…,metal_dirty=…,metal_diff=…,image=…,
#                build_rev=…,build_dirty=…,build_sha=…,run=…,host=…,cpu=…,kmd=…,dev=…,board=…,board_id=…,
#                asic_id=…,fw=…,cm_fw=…,dm_fw=…,eth_fw=…,gddr_fw=…,smi=…[,allow_dirty=1]
# 欄の意味と，測る前の確認・測っている途中の確認・検査側の受け入れの規則は scripts/README.md の「ログの出どころ」が
# 正本（ここに写さない）．ここでは関数ごとに何を作るか・確かめるかだけを書く．git・podman の問い合わせは，失敗しても
# PROV_TRIES 回（既定 3）まで PROV_RETRY_DT 秒（既定 1）おきにやり直してから unknown にする（下の prov_try）．
# マシンとボードの欄は scripts/lib/prov_machine.py が読む（下の prov_machine）．
#
# なぜ要るか：ログのファイル名には「いつ・どのソースで・どのビルドで測ったか」が無く，実装を替える前と後の
# ログが同じ名前で並ぶと見分けられない（例えば GEMM 段を替えた前後．docs/gemm_opt_results.md）．
# また HEAD だけでは，実行したバイナリがそのソースから作られたかは分からない（古いバイナリを残したまま
# STEPS="cool" と流すと，今の HEAD の測定として記録されてしまう）．verify/check_paper_numbers.py・update_fig_*.py は
# この行を見て，rev・ビルド・tt-metal・イメージ・測ったマシン（計算機・ボード・ファームウェア）の揃わないログの混在を
# 拒む（scripts/README.md「ログの出どころ」）．マシンを記録するのは，同じ rev でも計算機とボードが変われば
# クロックの出方も性能も変わり，別の計算機で測り直した値と元の計算機の値が 1 つの表に混ざりうるため．
#
# OZ_PROVENANCE が立っていればそれを使う（ALLOW_DIRTY=1 なら allow_dirty=1 だけを足す）．コンテナの中で動くスクリプト
# （measure/exponent_sweep.sh・study/measure_paper.sh・build_all.sh）には，ホストで作った行をこの変数で渡す：
#     ./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/measure/exponent_sweep.sh"
# （remeasure_all.sh の in_container がこうして組む）．渡さなければコンテナの中の git で作る．git worktree では
# .git がホストのパスを指すので rev=unknown になり，image も unknown になるので，検査は既定でそのログを拒む．
# ホストで 1 項目ずつ流す測定（measure_lib.sh を読み込むもの）は，測る直前にホストで作る．
PROV_IMAGE="${TT_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc}"
PROV_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# tt-metal のソース．run.sh はこの木をコンテナの /tt-metal へマウントし，出どころの metal= もこの木から取る
# （マウントする木と記録する木を 1 つの変数で決め，別の checkout の rev を記録したまま固定の木で測る，という
# 食い違いを起こさない）．ホストでは TT_METAL_SRC（既定 ~/tt-metal-src）．TT_METAL_HOME は見ない
# （ホストの環境に別の用途で残っていることがあるため）．コンテナの中では run.sh が TT_METAL_SRC=/tt-metal を渡す．
PROV_METAL_SRC="${TT_METAL_SRC:-$HOME/tt-metal-src}"
# 使う Blackhole のデバイスファイル（ホストの TT_DEVICE．既定 /dev/tenstorrent/0）．run.sh はこれだけをコンテナへ渡し，
# 温度・クロック（measure_lib.sh の temp・clk）と出どころの行のボードの欄もこのデバイスのボードから読む
# （カードが複数ある計算機で，測るカードと別のカードの温度や製造番号を記録しないように）
PROV_DEVICE="${TT_DEVICE:-/dev/tenstorrent/0}"
# tt-smi（ホストで温度・クロック・ファームウェアの版を読む）．ホストの TT_SMI があればそれ，無ければ
# ~/.venvs/tt-smi/bin/tt-smi（この計算機で入れた場所），それも無ければ PATH の tt-smi
if [ -n "${TT_SMI:-}" ]; then
  PROV_SMI=$TT_SMI
elif [ -x "$HOME/.venvs/tt-smi/bin/tt-smi" ]; then
  PROV_SMI=$HOME/.venvs/tt-smi/bin/tt-smi
else
  PROV_SMI=$(command -v tt-smi 2>/dev/null || echo tt-smi)
fi
# マシンの欄のうち，ログの組の中で同じ値であることを要るもの（PROV_MACHINE_CORE は sysfs・/proc から読むので読み損ない
# が無く，1 つでも違えば別のマシンとみなす．PROV_MACHINE_DETAIL は tt-smi でしか読めず，一時的に読めないことが
# あるので，両方が分かっているときだけ比べる）と，検査が分かっていることを要る欄（PROV_MACHINE_REQUIRED）．
# 値の正本は lib/constants.tsv の MACHINE_CORE・MACHINE_DETAIL・MACHINE_REQUIRED（検査側の oz_logs.py も同じ行を読む）．
# 読めなければ（表が壊れた・無い）ここで止める（欄の並びが空のまま比べると，別のマシンのログを黙って通す）
. "$PROV_ROOT/scripts/lib/consts.sh"
PROV_MACHINE_CORE=$(const_get MACHINE_CORE) && PROV_MACHINE_DETAIL=$(const_get MACHINE_DETAIL) \
  && PROV_MACHINE_REQUIRED=$(const_get MACHINE_REQUIRED) || exit 2

# マシンとボードを読む Python（scripts/lib/prov_machine.py．tt-smi -s の JSON を標準入力に取る．空でもよい）．
#   ident はマシンの欄の 1 行，get はこのデバイスの項目から与えた欄の値（使い方・このデバイスの項目の選び方はその冒頭）
PROV_MACHINE_PY="$PROV_ROOT/scripts/lib/prov_machine.py"

# _prov_smi_json   tt-smi -s の JSON（{ を含まない出力・失敗・PROV_SMI_TIMEOUT 秒（既定 60）を超えたら非零）．
#   時間の上限を置くのは，デバイスが止まったときに tt-smi が戻らず，測定と出どころの行の作成が止まらないように
_prov_smi_json() {
  local js
  js=$(timeout "${PROV_SMI_TIMEOUT:-60}" "$PROV_SMI" -s 2>/dev/null) || return 1
  case $js in *'{'*) printf '%s\n' "$js" ;; *) return 1 ;; esac
}

# prov_smi_get 欄…   tt-smi -s を 1 回呼び，このデバイス（PROV_DEVICE）の項目から欄の値を 1 行ずつ出す．
#   tt-smi が失敗した・このデバイスの項目が決まらないときは何も出さずに非零（measure_lib.sh の temp・clk が使う．
#   やり直さないのは，呼び出し側が読めなかった回を数えて扱うため）
prov_smi_get() {
  local js
  js=$(_prov_smi_json) || return 1
  printf '%s\n' "$js" | python3 -B "$PROV_MACHINE_PY" get "$PROV_DEVICE" "$@"
}

# prov_machine   マシンとボードの欄を 1 行（host=…,cpu=…,…,smi=…）で出す．tt-smi の JSON は prov_try でやり直してから
#   諦め（そのときは tt-smi でしか読めない欄が unknown になる），python3 が無ければ host だけを埋める
prov_machine() {
  local js out k
  js=$(prov_try _prov_smi_json) || js=""
  if out=$(printf '%s' "$js" | python3 -B "$PROV_MACHINE_PY" ident "$PROV_DEVICE" 2>/dev/null) && [ -n "$out" ]; then
    printf '%s\n' "$out"; return
  fi
  out="host=$(hostname 2>/dev/null | tr -c 'A-Za-z0-9._:+\n-' _)"
  for k in cpu kmd dev board board_id asic_id fw cm_fw dm_fw eth_fw gddr_fw smi; do out="$out,$k=unknown"; done
  printf '%s\n' "$out"
}

# prov_machine_diff 行A 行B   2 つの行（PROVENANCE 行でも prov_machine の行でもよい）のマシンの欄が同じマシンを指すか．
#   PROV_MACHINE_CORE は値が同じこと（unknown も値として比べる），PROV_MACHINE_DETAIL は両方が分かっていれば同じこと．
#   違えば違う欄を「欄: A → B」で 1 行ずつ出して 1，同じなら何も出さずに 0
prov_machine_diff() {
  local k a b bad=0
  for k in $PROV_MACHINE_CORE; do
    a=$(prov_field "$1" "$k"); b=$(prov_field "$2" "$k")
    [ "${a:-unknown}" = "${b:-unknown}" ] || { echo "  $k: ${a:-（無い）} → ${b:-（無い）}"; bad=1; }
  done
  for k in $PROV_MACHINE_DETAIL; do
    a=$(prov_field "$1" "$k"); b=$(prov_field "$2" "$k")
    if prov_known "$a" && prov_known "$b" && [ "$a" != "$b" ]; then echo "  $k: $a → $b"; bad=1; fi
  done
  return $bad
}

# prov_machine_missing 行   検査が必ず要るマシンの欄（PROV_MACHINE_REQUIRED．host・board_id・fw）のうち分からないものを
#   空白で区切って出す（すべて分かれば何も出さない）
prov_machine_missing() {
  local k out=""
  for k in $PROV_MACHINE_REQUIRED; do prov_known "$(prov_field "$1" "$k")" || out="$out${out:+ }$k"; done
  printf '%s' "$out"
}

# prov_field 行 キー   PROVENANCE 行（や BUILD 行）から 1 つの値を取り出す（無ければ空）
prov_field() {
  printf '%s\n' "$1" | tr ',' '\n' | sed -n "s/^$2=//p" | head -1
}

# prov_known 値   分かっている値（空・unknown・none でない）なら 0
prov_known() {
  case "$1" in ''|unknown|none) return 1 ;; esac
  return 0
}

# prov_try コマンド…   コマンドが成功するまで PROV_TRIES 回（既定 3）まで PROV_RETRY_DT 秒（既定 1）おきにやり直し，
#   成功した回の標準出力を出して 0 を返す．すべて失敗すれば何も出さずに 1．
#   podman と git の問い合わせは一時的に失敗しうる（段の最後の podman run --rm の後始末とぶつかる，別のプロセスが
#   git の index を書き換えている，など）．1 回の失敗をそのまま unknown にすると，guard がそれを識別子の変化と
#   みなして終わった段を無効化し，その値を刻んだログは検査に拒まれるので，やり直してから判断する
prov_try() {
  local i out n="${PROV_TRIES:-3}"
  for ((i = 1; i <= n; i++)); do
    if out=$("$@" 2>/dev/null); then printf '%s\n' "$out"; return 0; fi
    [ "$i" -lt "$n" ] && sleep "${PROV_RETRY_DT:-1}"
  done
  return 1
}

# prov_dirty ディレクトリ   src/・scripts/・tools/ の未コミットの変更：0／1／unknown（git status の失敗）
prov_dirty() {
  local out
  if ! out=$(prov_try git -C "$1" status --porcelain --untracked-files=normal -- src scripts tools); then
    echo unknown; return
  fi
  if [ -n "$out" ]; then echo 1; else echo 0; fi
}

# _prov_diffsum 木   追跡しているファイルの変更（git diff HEAD）の sha256 の先頭 12 桁．git が失敗すれば非零
#   （prov_try のサブシェルの中で呼ぶので，pipefail は呼び出し元へ漏れない）
_prov_diffsum() {
  set -o pipefail
  git -C "$1" diff HEAD | sha256sum | cut -c1-12
}

# prov_metal   tt-metal（PROV_METAL_SRC）の "rev dirty diff" を 1 行で
prov_metal() {
  local mh="$PROV_METAL_SRC" rev st d
  rev=$(prov_try git -C "$mh" rev-parse --short=12 HEAD) || { echo "unknown unknown unknown"; return; }
  if ! st=$(prov_try git -C "$mh" status --porcelain --untracked-files=no); then
    echo "$rev unknown unknown"; return
  fi
  if [ -z "$st" ]; then echo "$rev 0 none"; return; fi
  d=$(prov_try _prov_diffsum "$mh") || d=unknown
  echo "$rev 1 ${d:-unknown}"
}

# _prov_image_id   podman image inspect の .Id の 1 行目（空なら非零．prov_try がやり直す）
_prov_image_id() {
  local id
  id=$(podman image inspect --format '{{.Id}}' "$PROV_IMAGE") || return 1
  id=${id%%$'\n'*}; id=${id#sha256:}
  [ -n "$id" ] || return 1
  printf '%s\n' "$id"
}

# prov_image   コンテナのイメージ ID の先頭 12 桁（podman が無い・イメージが無い・やり直しても読めなければ unknown）
prov_image() {
  local id
  command -v podman >/dev/null 2>&1 || { echo unknown; return; }   # コンテナの中（やり直しても変わらない）
  if id=$(prov_try _prov_image_id); then printf '%s\n' "${id:0:12}"; else echo unknown; fi
}

# prov_stamp   "build_rev build_dirty build_sha"（src/build/BUILD_STAMP．無ければ none none none．欄が読めなければその欄を none）
prov_stamp() {
  local st="$PROV_ROOT/src/build/BUILD_STAMP" head r d s
  if [ ! -r "$st" ]; then echo "none none none"; return; fi
  head=$(grep -m1 '^BUILD,' "$st")
  r=$(prov_field "$head" rev); d=$(prov_field "$head" dirty); s=$(sha256sum "$st" | cut -c1-16)
  printf '%s %s %s\n' "${r:-none}" "${d:-none}" "${s:-none}"
}

# _prov_core   出どころの行のうち作業木・tt-metal・イメージ・ビルドの記録で決まる欄を，PROV_IDENT_KEYS の順に
#   空白で区切って出す（provenance_line と prov_ident が同じ問い合わせで作る．tt-smi は呼ばない）
_prov_core() {
  local rev dirty m b
  rev=$(prov_try git -C "$PROV_ROOT" rev-parse HEAD) || rev=unknown
  if [ "$rev" = unknown ]; then dirty=unknown; else dirty=$(prov_dirty "$PROV_ROOT"); fi
  read -r -a m <<<"$(prov_metal)"
  read -r -a b <<<"$(prov_stamp)"
  printf '%s %s %s %s %s %s %s %s %s\n' "$rev" "$dirty" "${m[0]:-unknown}" "${m[1]:-unknown}" "${m[2]:-unknown}" \
         "$(prov_image)" "${b[0]:-none}" "${b[1]:-none}" "${b[2]:-none}"
}
# 測っている途中で変わってはならない出どころの欄（measure_lib.sh の prov_guard・prov_end が測り始めと比べる．
# remeasure_all.sh の guard が比べる rev・dirty・build_sha・metal・metal_dirty・metal_diff・image と同じものに，
# ビルドの記録の rev・dirty を足したもの）．マシンの欄は prov_machine_diff で別に比べる
PROV_IDENT_KEYS="rev dirty metal metal_dirty metal_diff image build_rev build_dirty build_sha"

provenance_line() {
  local line c
  if [ -n "${OZ_PROVENANCE:-}" ]; then
    line=$OZ_PROVENANCE
  else
    read -r -a c <<<"$(_prov_core)"
    line=$(printf 'PROVENANCE,rev=%s,dirty=%s,date=%s,metal=%s,metal_dirty=%s,metal_diff=%s,image=%s,build_rev=%s,build_dirty=%s,build_sha=%s,run=%s,%s' \
           "${c[0]}" "${c[1]}" "$(date -Iseconds)" "${c[2]}" "${c[3]}" "${c[4]}" "${c[5]}" \
           "${c[6]}" "${c[7]}" "${c[8]}" "${OZ_RUN_ID:-none}" "$(prov_machine)")
  fi
  # ALLOW_DIRTY=1 は provenance_check の外れを許すので，そのログには拒むための印を残す
  if [ "${ALLOW_DIRTY:-0}" = 1 ]; then
    case ",$line," in *,allow_dirty=1,*) ;; *) line="$line,allow_dirty=1" ;; esac
  fi
  printf '%s\n' "$line"
}

# prov_ident [行]   PROV_IDENT_KEYS の欄を "rev=…,dirty=…,…" の 1 行にする．行（PROVENANCE 行）を与えればその行から，
#   与えなければ今の作業木・tt-metal・イメージ・ビルドの記録から作る（OZ_PROVENANCE は見ない．tt-smi も呼ばないので速い）
prov_ident() {
  local k out="" c i=0
  if [ $# -gt 0 ]; then
    for k in $PROV_IDENT_KEYS; do out="$out${out:+,}$k=$(prov_field "$1" "$k")"; done
  else
    read -r -a c <<<"$(_prov_core)"
    for k in $PROV_IDENT_KEYS; do out="$out${out:+,}$k=${c[$i]:-}"; i=$((i + 1)); done
  fi
  printf '%s\n' "$out"
}

# prov_ident_diff 行A 行B   2 つの行（PROVENANCE 行でも prov_ident の行でもよい）の PROV_IDENT_KEYS の欄が同じか．
#   違えば違う欄を「  欄: A → B」で 1 行ずつ出して 1，同じなら何も出さずに 0（unknown も値として比べる）
prov_ident_diff() {
  local k a b bad=0
  for k in $PROV_IDENT_KEYS; do
    a=$(prov_field "$1" "$k"); b=$(prov_field "$2" "$k")
    [ "$a" = "$b" ] || { echo "  $k: ${a:-（無い）} → ${b:-（無い）}"; bad=1; }
  done
  return $bad
}

# write_build_stamp   make の直後に src/build/BUILD_STAMP を書く（build_all.sh が呼ぶ）．
#   1 行目 BUILD,rev=…,dirty=…,date=…,metal=…,metal_dirty=…,metal_diff=…,image=…,host=…,cpu=…（date 以外はビルドした
#   時点の provenance_line の値．コンテナの中では OZ_PROVENANCE で渡されたホストの行），続いて src/build/ の実行ファイル
#   ごとに BIN,<名前>,<sha256>．provenance_check はこのスタンプの rev が今の rev と同じで，各バイナリの sha256 が
#   変わっておらず，tt-metal・コンテナのイメージ・計算機（host・cpu）がビルドしたときと同じことを確かめる．
#   計算機を比べるのは，ホスト側は -march=native でビルドするので（src/Makefile），別の計算機で作った src/build を
#   写して測ると，今の計算機のために作ったものでないバイナリで測ることになるため．一時ファイルに書いてから
#   置き換える（途中で止まっても半端なスタンプを残さない）．
#   どれかのバイナリの sha256 を取れなければ（読めない・sha256sum が失敗した），スタンプを書かずに理由を出して 1 を返す
#   （`$(sha256sum … | cut …)` は sha256sum が失敗しても cut と printf が成功し，空のハッシュのスタンプを書いてしまう．
#   空どうしは確かめる側でも一致して見える）．どの行の書き込みが失敗しても（空き容量），書いた BIN 行の数が src/build の
#   実行ファイルの数と違っても，置き換えずに 1 を返す（BIN 行の欠けたスタンプを正式名へ移さない）．書きかけの一時ファイルは
#   .BUILD_STAMP.<pid> に残る
write_build_stamp() {
  local dir="$PROV_ROOT/src/build" line f tmp h bad="" werr=0 nexe=0 nbin
  line=$(provenance_line)
  tmp="$dir/.BUILD_STAMP.$$"
  printf 'BUILD,rev=%s,dirty=%s,date=%s,metal=%s,metal_dirty=%s,metal_diff=%s,image=%s,host=%s,cpu=%s\n' \
         "$(prov_field "$line" rev)" "$(prov_field "$line" dirty)" "$(date -Iseconds)" \
         "$(prov_field "$line" metal)" "$(prov_field "$line" metal_dirty)" "$(prov_field "$line" metal_diff)" \
         "$(prov_field "$line" image)" "$(prov_field "$line" host)" "$(prov_field "$line" cpu)" >"$tmp" || werr=1
  for f in "$dir"/*; do
    [ -f "$f" ] && [ -x "$f" ] || continue
    nexe=$((nexe + 1))
    if h=$(prov_sha256 "$f"); then printf 'BIN,%s,%s\n' "$(basename "$f")" "$h" >>"$tmp" || werr=1
    else bad="$bad $(basename "$f")"; fi
  done
  nbin=$(grep -c '^BIN,' "$tmp" 2>/dev/null)
  if [ -n "$bad" ] || [ "$werr" = 1 ] || [ "${nbin:-0}" != "$nexe" ]; then
    echo "ERROR: ビルドの記録 $dir/BUILD_STAMP を書けない（sha256 を取れないバイナリ:${bad:- なし}．書き込みの失敗:$([ "$werr" = 1 ] && echo ' あり' || echo ' なし')．BIN 行 ${nbin:-0} 本／実行ファイル $nexe 本）" >&2
    return 1
  fi
  mv "$tmp" "$dir/BUILD_STAMP"
}

# prov_sha256 ファイル   ファイルの sha256（64 桁の 16 進）を出す．読めない・sha256sum が失敗した・形が違えば何も出さずに 1
prov_sha256() {
  local h
  h=$(sha256sum "$1" 2>/dev/null) || return 1
  h=${h%% *}
  [[ $h =~ ^[0-9a-f]{64}$ ]] || return 1
  printf '%s\n' "$h"
}

# provenance_check   今の src/build が今のソースのビルドかを確かめる（測る前に呼ぶ）．
#   provenance_line（OZ_PROVENANCE が立っていればその行）の rev が 40 桁の rev で dirty=0，
#   src/build/BUILD_STAMP がありその rev が同じで dirty=0，スタンプに載った各バイナリの sha256 が今のファイルと
#   同じ，スタンプの tt-metal（metal・metal_dirty・metal_diff．ビルドしたときの tt-metal）が PROVENANCE 行の
#   tt-metal（今マウントする木）と同じ，スタンプのコンテナのイメージ（image．ビルドしたときのイメージ）が分かっていて
#   PROVENANCE 行の image（今使うイメージ）と同じ，スタンプの計算機（host・cpu．ビルドした計算機）が PROVENANCE 行の
#   host・cpu（今の計算機）と同じ，PROVENANCE 行のマシンの欄のうち検査が要る host・board_id・fw が分かる，
#   PROVENANCE 行の build_sha がこのスタンプのもの，のすべてを満たせば 0．外れれば理由を標準エラーへ出して 1．
#   マシンの欄が分からない（TT_DEVICE のデバイスが無い・tt-kmd の sysfs も tt-smi も読めない）まま測ったログは
#   検査に拒まれるので，測る前に止める．
#   イメージを比べるのは，前に測った rev の一部の段を測り直すとき（remeasure_all.sh の REMEASURE_AT）に，
#   その間に podman pull し直した・TT_IMAGE の違うシェルで流した，という食い違いを測る前に止めるため
#   （段をまたぐ識別子の比較は remeasure_all.sh の 1 回の実行の中だけなので，それが無いと最後の検査まで気づかない）．
#   image が分からない（podman が読めない，コンテナの中で出どころの行を作った）ときも，そのログは検査に拒まれる
#   ので測らない．
#   ALLOW_DIRTY=1（試し用）のときは理由を出すが 0 を返す．そのときの PROVENANCE 行には allow_dirty=1 が付き
#   （provenance_line），検査側は既定でそのログを拒む．
#   引数に PROVENANCE 行を与えると，その行で確かめる（measure_lib.sh が測り始めの行 PROV_START_LINE を確かめた行と
#   同じにするため）．無ければ provenance_line で作る
provenance_check() {
  local line st rev dirty head brev bdirty bsha bi li cur tag name h bad=""
  line=${1:-}
  [ -n "$line" ] || line=$(provenance_line)
  st="$PROV_ROOT/src/build/BUILD_STAMP"
  rev=$(prov_field "$line" rev); dirty=$(prov_field "$line" dirty)
  echo "$rev" | grep -Eq '^[0-9a-f]{40}$' || bad="$bad\n  rev が分からない（rev=$rev）"
  [ "$dirty" = 0 ] || bad="$bad\n  src/・scripts/・tools/ に未コミットの変更がある，または git status が失敗した（dirty=$dirty）"
  if [ ! -r "$st" ]; then
    bad="$bad\n  ビルドの記録 src/build/BUILD_STAMP が無い（build_all.sh でビルドし直すこと）"
  else
    head=$(grep -m1 '^BUILD,' "$st")
    brev=$(prov_field "$head" rev); bdirty=$(prov_field "$head" dirty)
    [ "$brev" = "$rev" ] || bad="$bad\n  src/build のバイナリは rev=$brev のソースから作ったもので，今の rev=$rev と違う（build_all.sh でビルドし直すこと）"
    [ "$bdirty" = 0 ] || bad="$bad\n  src/build のバイナリは未コミットの変更がある状態（dirty=$bdirty）でビルドした"
    while IFS=, read -r tag name h; do
      [ "$tag" = BIN ] || continue
      # 記録の値も今の値も 64 桁の 16 進であることを要る（取れなかったハッシュを空どうしで一致と見ない）
      cur=$(prov_sha256 "$PROV_ROOT/src/build/$name") || cur=none
      if ! [[ $h =~ ^[0-9a-f]{64}$ ]]; then
        bad="$bad\n  ビルドの記録の src/build/$name の sha256 が 64 桁の 16 進でない（$h．build_all.sh でビルドし直すこと）"
      elif [ "$cur" != "$h" ]; then
        bad="$bad\n  src/build/$name がビルドの記録と違う（ビルドの後に差し替わった・消えた・読めない）"
      fi
    done <"$st"
    local k bm lm mw
    for k in metal metal_dirty metal_diff; do
      bm=$(prov_field "$head" "$k"); lm=$(prov_field "$line" "$k")
      if [ -z "$bm" ]; then
        bad="$bad\n  ビルドの記録に tt-metal の $k が無い（build_all.sh でビルドし直すこと）"
      elif [ "$bm" != "$lm" ]; then
        bad="$bad\n  src/build は tt-metal の $k=$bm でビルドしたもので，今の tt-metal（$PROV_METAL_SRC）の $k=$lm と違う（build_all.sh でビルドし直すこと）"
      fi
    done
    # 値の形（image と同じく，分からない値どうしの一致で通さない）：metal は分かる rev，metal_dirty は 0 か 1，metal_diff は
    # metal_dirty=0 なら none，1 なら変更の sha256 の先頭 12 桁（検査側が組の中で比べる形）
    lm=$(prov_field "$line" metal); mw=""
    prov_known "$lm" || mw="metal=${lm:-（無い）} が分からない"
    case "$(prov_field "$line" metal_dirty):$(prov_field "$line" metal_diff)" in
      0:none) ;;
      1:*) [[ $(prov_field "$line" metal_diff) =~ ^[0-9a-f]{12}$ ]] || mw="$mw${mw:+．}metal_diff=$(prov_field "$line" metal_diff) が変更の sha256 の先頭 12 桁でない" ;;
      *) mw="$mw${mw:+．}metal_dirty=$(prov_field "$line" metal_dirty)・metal_diff=$(prov_field "$line" metal_diff) が 0・none か 1・<12 桁> でない" ;;
    esac
    [ -z "$mw" ] || bad="$bad\n  今の tt-metal（$PROV_METAL_SRC）の出どころが読めない（$mw．git が失敗した・木が無い）"

    bi=$(prov_field "$head" image); li=$(prov_field "$line" image)
    if [ -z "$bi" ]; then
      bad="$bad\n  ビルドの記録にコンテナのイメージ（image）が無い（build_all.sh でビルドし直すこと）"
    elif ! prov_known "$bi"; then
      bad="$bad\n  src/build はコンテナのイメージが分からない状態（image=$bi）でビルドした（ホストで作った出どころの行を OZ_PROVENANCE で渡して build_all.sh でビルドし直すこと）"
    elif ! prov_known "$li"; then
      bad="$bad\n  今のコンテナのイメージが分からない（image=$li．podman image inspect $PROV_IMAGE が読めない，またはコンテナの中で出どころの行を作った．ホストで作って OZ_PROVENANCE で渡すこと）"
    elif [ "$bi" != "$li" ]; then
      bad="$bad\n  src/build はコンテナのイメージ $bi でビルドしたもので，今のイメージ $li（$PROV_IMAGE）と違う（イメージを入れ替えたなら build_all.sh でビルドし直し，全段を測り直すこと）"
    fi
    # コンテナの中では，run.sh が実際に走らせたイメージの ID を OZ_IMAGE_ID で渡す．出どころの行（ホストで作ってから
    # run.sh が排他を待つことがある）の image と違えば，待つあいだにタグ（$PROV_IMAGE）が付け替わったので測らない
    if [ -n "${OZ_IMAGE_ID:-}" ] && [ "${OZ_IMAGE_ID:0:12}" != "$li" ]; then
      bad="$bad\n  走っているコンテナのイメージ ${OZ_IMAGE_ID:0:12}（run.sh が渡した OZ_IMAGE_ID）が出どころの行の image=$li と違う（出どころの行を作った後にタグ $PROV_IMAGE が付け替わった）"
    fi
    for k in host cpu; do
      bm=$(prov_field "$head" "$k"); lm=$(prov_field "$line" "$k")
      if [ -z "$bm" ]; then
        bad="$bad\n  ビルドの記録にビルドした計算機（$k）が無い（build_all.sh でビルドし直すこと）"
      elif [ "$bm" != "$lm" ]; then
        bad="$bad\n  src/build は $k=$bm の計算機でビルドしたもので，今の計算機（$k=${lm:-（無い）}）と違う（別の計算機の src/build を写したなら，この計算機の build_all.sh でビルドし直すこと）"
      fi
    done
    bsha=$(prov_field "$line" build_sha); cur=$(sha256sum "$st" | cut -c1-16)
    [ "$bsha" = "$cur" ] || bad="$bad\n  PROVENANCE 行の build_sha=$bsha が今の src/build/BUILD_STAMP（$cur）と違う"
  fi
  local mm; mm=$(prov_machine_missing "$line")
  [ -z "$mm" ] || bad="$bad\n  測るマシンの $mm が分からない（TT_DEVICE=$PROV_DEVICE のデバイスが無い，tt-kmd の sysfs も tt-smi（$PROV_SMI）も読めない，またはコンテナの中で出どころの行を作った．そのログは検査に拒まれる）"
  [ -z "$bad" ] && return 0
  if [ "${ALLOW_DIRTY:-0}" = 1 ]; then
    printf '注意（ALLOW_DIRTY=1 なので続ける．このログは既定の検査を通らない）:%b\n' "$bad" >&2
    return 0
  fi
  printf 'ERROR: 今の src/build を今のソースの測定に使えない:%b\n' "$bad" >&2
  return 1
}

# bin_line コマンド…   コマンドの中の ./src/build/<名前> について
#     BINARY,<名前>,sha256=<今のファイルの値>,stamp_sha256=<src/build/BUILD_STAMP に載った値>
#   を出す（コマンドにバイナリが無ければ何も出さない．ファイル・スタンプの行が無ければその値は none）．
#   検査側（oz_logs.py の LogSet）は 2 つが同じであることと，同じ名前のバイナリの sha256 がログの組の中で
#   1 つであることを要る（測る前の確認の後にバイナリだけを差し替えて測ったログを拒む）．
#   1 つの文字列で渡されたコマンドにも使える（最初に現れたバイナリ 1 つ）．
bin_line() {
  local b name h sh
  b=$(printf '%s\n' "$*" | grep -oE 'src/build/[A-Za-z0-9_]+' | head -1)
  [ -n "$b" ] || return 0
  name=${b#src/build/}
  h=$(prov_sha256 "$PROV_ROOT/$b") || h=""
  sh=$(grep -m1 "^BIN,$name," "$PROV_ROOT/src/build/BUILD_STAMP" 2>/dev/null | cut -d, -f3)
  printf 'BINARY,%s,sha256=%s,stamp_sha256=%s\n' "$name" "${h:-none}" "${sh:-none}"
}

# 実行されたとき（source ではなく）は 1 行を出して終わる
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  provenance_line
fi
