#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Host-only compile check of the library sources: every translation unit of
# the Makefile is compiled (not linked) with the Makefile's flags plus
# -Wall -Wextra -Wno-missing-field-initializers, all units in parallel.
# No device is used.  Prints the return code of every unit and every
# diagnostic; exits 1 if any unit failed or produced a warning.
#   tools/hostcc.sh [1|2|all]   all (default): scheme I and II drivers for
#                               the eight formats, plus context, dispatch,
#                               evaluator, tests and the example
# Environment:
#   TT_METAL_BUILD   Metalium build directory (default /tt-metal/build_Release)
#   OBJ              directory for objects and logs (default: a temporary
#                    directory that is removed at exit)
#   CXX, CXXFLAGS    compiler and optimization flags (g++, -O3 -DNDEBUG)
set -uo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
what=${1:-all}
case $what in 1|2|all) ;; *) echo "usage: $0 [1|2|all]" >&2; exit 2 ;; esac
TB=${TT_METAL_BUILD:-/tt-metal/build_Release}
[[ -f $TB/include/tt-metalium/host_api.hpp ]] ||
  { echo "hostcc.sh: Metalium headers not found under TT_METAL_BUILD=$TB" >&2; exit 2; }
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -DNDEBUG}
if [[ -n ${OBJ:-} ]]; then mkdir -p -- "$OBJ"; obj=$(cd -- "$OBJ" && pwd)
else obj=$(mktemp -d); trap 'rm -rf -- "$obj"' EXIT; fi

# The Makefile's flags (LOCAL_CPPFLAGS, METAL_CPPFLAGS, WARNINGS, REQUIRED_CXXFLAGS).
FL="-I$root/include -I$root/src -I$root/src/backend -I$obj
    -isystem $TB/include -isystem $TB/include/metalium-thirdparty
    -isystem $TB/libexec/tt-metalium -isystem $TB/libexec/tt-metalium/tt_metal/hostdevcommon/api
    -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 -DFMT_HEADER_ONLY=1 -DNTEST
    -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DTRACY_IMPORTS
    -DTRACY_TIMER_FALLBACK -DTT_UMD_BUILD_SIMULATION
    $CXXFLAGS -Wall -Wextra -Wno-missing-field-initializers
    -std=gnu++20 -fno-fast-math -ffp-contract=off -fPIC -fopenmp -Wno-int-to-pointer-cast"
formats="BF16 FP16 TF32 BFP8 FP8 INT8 BFP4 BFP2"
: > "$obj/rc.txt"
compile() {  # compile NAME SOURCE [FLAGS...]
  local name=$1 src=$2; shift 2
  ( $CXX $FL "$@" -c "$src" -o "$obj/$name.o" > "$obj/$name.log" 2>&1
    echo "$name rc=$?" >> "$obj/rc.txt" ) &
}
cd "$root"
for sch in 1 2; do
  [[ $what == all || $what == "$sch" ]] || continue
  for f in $formats; do
    macro=$f; [[ $f == TF32 ]] && macro=FP32
    name=$(echo "$f" | tr '[:upper:]' '[:lower:]')
    compile "ozaki${sch}_$name" "src/ozaki$sch.cpp" "-DOZ_FMT_$macro=1" \
      "-DTTGEMM_BACKEND_NS=ozaki${sch}_$name" "-DTTGEMM_PUBLIC_FORMAT=::ttgemm::Format::$f"
  done
done
if [[ $what == all ]]; then
  # What the Makefile generates: build/formats.inc with one DECLARE per format.
  for f in $formats; do echo "DECLARE($f, $(echo "$f" | tr '[:upper:]' '[:lower:]'))"; done > "$obj/formats.inc"
  compile context src/context.cpp "-DTTGEMM_SOURCE_KERNEL_DIR=\"$root/src/backend\"" \
    "-DTTGEMM_INSTALL_KERNEL_DIR=\"/usr/local/share/ttgemm/kernels\""
  compile dispatch src/dispatch.cpp
  compile evaluation evaluation/evaluate.cpp -march=native
  compile test tests/correctness.cpp
  compile example examples/gemm.cpp
fi
wait
sort "$obj/rc.txt" | tr '\n' ' '; echo
bad=0
for log in "$obj"/*.log; do
  [[ -s $log ]] || continue
  echo "== $(basename "$log" .log)"; cat "$log"; bad=1
done
grep -qv 'rc=0$' "$obj/rc.txt" && bad=1
if [[ $bad == 0 ]]; then echo "hostcc: clean ($(wc -l < "$obj/rc.txt") units, no diagnostics)"
else echo "hostcc: FAILED (see above)"; fi
exit $bad
