#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Two paper evaluations, without the research repository's experiment harness.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
usage() {
  cat <<'HELP'
Usage: ./evaluate.sh {quick|table|phi|all} [options]
  quick   Small 128^3 checks, no cooldown or steady-state warmup.
  table   16 Ozaki I/II implementations at 3520, 7040, 14080.
  phi     BFP8, phi=0.5,1,2,4,8 at 14080^3; Ozaki II uses 1 and 2 passes.
  all     Both paper evaluations.
Options:
  --list                  Print the case list without using a device.
  --sizes 3520,7040        Table sizes (multiples of 64).
  --formats BF16,BFP8      Table formats (all eight by default).
  --phi-size 14080        Matrix size for the phi experiment.
  --phis 0.5,1,2,4,8       Phi points.
  --repeats 3             Timed repetitions per phase.
  --warmup 10             Minimum initial warmup seconds; 0 disables steady checks.
  --cooldown 60           Start each case at or below this temperature; 0 disables.
  --timeout 300           Cooldown timeout in seconds.
  --device 0              Blackhole device ID.
  --sysfs-device N        Telemetry index (/sys/class/tenstorrent/tenstorrent!N) if it differs.
  --blas PATH             Optional path to LP64 libopenblas.so.0.
  --no-cpu                Omit CPU DGEMM comparison (FP64 equivalence is unknown).
  --no-reference          Performance only; omit Dot2 and CPU comparison.
Paper-sized runs take a long time. Start with quick or --list.
JOBS controls build parallelism; FORMATS and BUILD_DIR are Makefile settings.
HELP
}
if [[ $# == 0 || $1 == --help || $1 == -h ]]; then usage; exit 0; fi
mode=$1; shift
extra=()
case $mode in
  quick) mode=all; extra=(--quick) ;;
  table|phi|all) ;;
  *) usage >&2; exit 2 ;;
esac
for arg in "$@"; do
  if [[ $arg == --case ]]; then
    echo "evaluate.sh: --case is reserved for the per-case processes this script starts;" \
         "run the evaluator binary directly to select one case of the plan." >&2
    exit 2
  fi
done
make -s -C "$root" -j"${JOBS:-2}" evaluation >&2
build=${BUILD_DIR:-build}
[[ $build == /* ]] || build="$root/$build"
for arg in "$@"; do
  if [[ $arg == --list ]]; then
    exec "$build/ozTTgemm_evaluate" --suite "$mode" "${extra[@]}" "$@"
  fi
done
mkdir -p "$root/evaluation_results"
out=$(mktemp -d "$root/evaluation_results/$(date +%Y%m%d-%H%M%S)-XXXXXX")
printf 'Results directory: %s\n' "$out"
{
  printf 'Date: %s\n' "$(date -Iseconds)"
  printf 'Command:'; printf ' %q' "$build/ozTTgemm_evaluate" --suite "$mode" "${extra[@]}" "$@"; printf '\n'
  sha256sum "$build/libozTTgemm.a"
  cat "$build/.build-config"
} > "$out/run.log"
if [[ ${#extra[@]} -gt 0 ]]; then
  # quick: all cases in one process (no cooling between cases).
  "$build/ozTTgemm_evaluate" --suite "$mode" "${extra[@]}" --output "$out/results.csv" "$@" 2>&1 | tee -a "$out/run.log"
else
  # One process per case: the device returns to its idle clock, where it can cool,
  # only when the process that opened it exits. A failed case stops the run.
  count=$("$build/ozTTgemm_evaluate" --suite "$mode" "$@" --list | tail -n +2 | wc -l)
  for ((i = 0; i < count; i++)); do
    "$build/ozTTgemm_evaluate" --suite "$mode" --output "$out/results.csv" "$@" --case "$i" 2>&1 | tee -a "$out/run.log"
  done
fi
