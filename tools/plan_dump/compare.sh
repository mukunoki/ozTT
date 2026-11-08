#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compares every decision of the scheme I and II host planners between a git
# revision and the working tree.  The two dump programs in this directory are
# built for all eight formats against both source trees (no device, no
# libtt_metal), run for every part of their case lists, and the outputs are
# diffed per format and part.  Exits 0 when all outputs are identical, 1 on a
# difference or a build/run failure, 2 on a usage error.
#   tools/plan_dump/compare.sh [REV]      default REV is HEAD
# Environment (as in the Makefile):
#   TT_BUILD         Metalium build directory (default $TT_METAL_HOME/build_Release,
#                    then /tt-metal/build_Release); headers and lib/libspdlog.a
#                    are used
#   OUT              work directory; its base/, bin/ and out/ are emptied first
#                    and kept (default: a temporary directory, removed at exit)
#   JOBS             parallel builds and runs (default 16)
#   CXX              compiler (default g++)
set -uo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
rev=${1:-HEAD}
if (( $# > 1 )) || [[ $rev == -* ]]; then echo "usage: $0 [REV]" >&2; exit 2; fi
TB=${TT_BUILD:-${TT_METAL_HOME:-/tt-metal}/build_Release}
CXX=${CXX:-g++}
JOBS=${JOBS:-16}
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo "compare.sh: JOBS must be a positive integer (JOBS=$JOBS)" >&2; exit 2; }
[[ -f $TB/include/tt-metalium/host_api.hpp && -f $TB/lib/libspdlog.a ]] ||
  { echo "compare.sh: Metalium headers or lib/libspdlog.a not found under $TB (set TT_BUILD or TT_METAL_HOME)" >&2; exit 2; }
cd "$root"
git rev-parse --verify --quiet "$rev^{commit}" >/dev/null ||
  { echo "compare.sh: unknown revision '$rev'" >&2; exit 2; }
if [[ -n ${OUT:-} ]]; then
  mkdir -p -- "$OUT" && work=$(cd -- "$OUT" && pwd) || exit 2
  rm -rf -- "$work/base" "$work/bin" "$work/out"   # no stale binary or output may stand in
  where="work directory $work"
else
  work=$(mktemp -d) || exit 2
  trap 'rm -rf -- "$work"' EXIT
  where="set OUT to keep the work directory"
fi
mkdir -p "$work/base" "$work/bin" "$work/out"
# Paths in the archive are relative to this directory (the repository root).
git archive --format=tar "$rev" include src | tar -x -C "$work/base" ||
  { echo "compare.sh: git archive of $rev failed" >&2; exit 2; }

formats="BF16 FP16 TF32 BFP8 FP8 INT8 BFP4 BFP2"
parts1="1 2 3"; parts2="1 2 3 4"
# A simple job limiter: at most JOBS background jobs at once.
limit() { while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done; }

build() {  # build SIDE SCHEME FORMAT
  local side=$1 sch=$2 fmt=$3 src macro=$3
  [[ $side == base ]] && src=$work/base || src=$root
  [[ $fmt == TF32 ]] && macro=FP32
  $CXX -std=gnu++20 -O2 -DNDEBUG -fno-fast-math -ffp-contract=off -fopenmp -Wno-int-to-pointer-cast -w \
    -I"$src/include" -I"$src/src" -I"$src/src/backend" \
    -isystem "$TB/include" -isystem "$TB/include/metalium-thirdparty" -isystem "$TB/libexec/tt-metalium" \
    -isystem "$TB/libexec/tt-metalium/tt_metal/hostdevcommon/api" \
    -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 -DFMT_HEADER_ONLY=1 -DNTEST -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL \
    -DTRACY_IMPORTS -DTRACY_TIMER_FALLBACK -DTT_UMD_BUILD_SIMULATION \
    "-DOZ_FMT_$macro=1" -DOZTTGEMM_BACKEND_NS=plan_dump "-DOZTTGEMM_PUBLIC_FORMAT=::ozTTgemm::Format::$fmt" \
    "$here/ozaki${sch}_plan_dump.cpp" -o "$work/bin/${side}_ozaki${sch}_$fmt" \
    "$TB/lib/libspdlog.a" -lquadmath > "$work/out/build_${side}_ozaki${sch}_$fmt.log" 2>&1
  echo "$?" > "$work/out/build_${side}_ozaki${sch}_$fmt.rc"
}

echo "building 32 dump programs (base: $rev, new: working tree) ..."
for side in base new; do for sch in 1 2; do for fmt in $formats; do
  limit; build "$side" "$sch" "$fmt" &
done; done; done
wait
fail=0
for side in base new; do for sch in 1 2; do for fmt in $formats; do
  tag=${side}_ozaki${sch}_$fmt
  [[ -x $work/bin/$tag && $(cat "$work/out/build_$tag.rc" 2>/dev/null) == 0 ]] && continue
  echo "BUILD FAILED ${side} ozaki${sch} $fmt:"; head -n 20 "$work/out/build_$tag.log"; fail=1
done; done; done
(( fail )) && { echo "plan_dump: build failure ($where)"; exit 1; }

echo "running ..."
run() {  # run SIDE SCHEME FORMAT PART
  local tag=${1}_ozaki${2}_$3.$4
  OMP_NUM_THREADS=4 timeout 3000 "$work/bin/${1}_ozaki${2}_$3" "$4" > "$work/out/$tag.txt" 2> "$work/out/$tag.err"
  echo "$?" > "$work/out/$tag.rc"
}
for side in base new; do for fmt in $formats; do
  for p in $parts1; do limit; run "$side" 1 "$fmt" "$p" & done
  for p in $parts2; do limit; run "$side" 2 "$fmt" "$p" & done
done; done
wait

same=0 differ=0
for sch in 1 2; do
  [[ $sch == 1 ]] && parts=$parts1 || parts=$parts2
  for fmt in $formats; do for p in $parts; do
    tag=ozaki${sch}_$fmt.$p
    rcb=$(cat "$work/out/base_$tag.rc") rcn=$(cat "$work/out/new_$tag.rc")
    if [[ $rcb != 0 || $rcn != 0 ]]; then
      echo "DIFF $tag: exit status base=$rcb new=$rcn"; differ=$((differ + 1))
    elif cmp -s "$work/out/base_$tag.txt" "$work/out/new_$tag.txt"; then
      same=$((same + 1))
    else
      echo "DIFF $tag: $(diff "$work/out/base_$tag.txt" "$work/out/new_$tag.txt" | grep -c '^[<>]') differing lines"
      diff "$work/out/base_$tag.txt" "$work/out/new_$tag.txt" | head -n 6 | cut -c1-200 | sed 's/^/    /'
      differ=$((differ + 1))
    fi
  done; done
done
if (( differ == 0 )); then
  echo "plan_dump: identical ($same outputs, base $rev vs working tree)"
  exit 0
fi
echo "plan_dump: $differ differing, $same identical (base $rev vs working tree; $where)"
exit 1
