# Paper evaluation

The entry point is `../evaluate.sh`. It builds a small C++ evaluator with GNU
Make and runs only the two main PDP evaluations. The source copy is standalone;
no files from the parent research repository are read.

```sh
./evaluate.sh quick                   # small checks first
./evaluate.sh all --list              # show the 63 paper-sized cases; no device use
./evaluate.sh table                   # 16 implementations, three sizes
./evaluate.sh phi                     # BFP8 exponent-range experiment
./evaluate.sh all                     # both
```

Run these commands from the release directory in a working TT-Metalium runtime
environment. `quick` uses 128^3, three phi points, one timed repetition and no
steady-state warmup or cooling. It evaluates 25 cases; its timings are **not**
paper performance measurements. A first run also compiles device kernels.
Paper-sized runs include CPU reference calculations and take hours depending
on cooling, clock settling, host CPU and the selected cases. This repository
update was verified with `quick` and one run of the complete 63-case plan (see
`../VALIDATION.md`); it does not replace the saved paper measurements.

`paper_reference.csv` contains the 63 corresponding measurements from the
paper-data snapshot `63e84a6` (48 table rows and 15 phi rows). It is supplied
for manual comparison with new `results.csv` files, not as a pass/fail threshold.
`total_ms` is the whole-pass time (the paper's Total) and `phase_sum_ms` the sum
of the phase times, both at the saved logs' printed precision; `effective_tflops`
is 2N^3 over the whole-pass time, and effective bits are calculated from the
full-precision relative Frobenius errors. No saved value is substituted into new
evaluation results. `moduli` is the selected count, whereas `--list` shows the
allowed maximum.

## Cases and parameters

`table` uses square matrices of sizes 3520, 7040 and 14080, FP64 uniform input
in [-1,1), seed 7, and all eight formats with both schemes. Scheme I retains
all diagonals through p-1, with these fixed slice counts:

| Format | BF16 | FP16 | TF32 | BFP8 | FP8 | INT8 | BFP4 | BFP2 |
|---|---|---|---|---|---|---|---|---|
| p | 10 | 10 | 10 | 10 | 12 | 8 | 16 | 59 |
| II maximum moduli | 24 | 24 | 11 | 24 | 14 | 16 | 24 | 24 |

Scheme II fixes b=54 and uses one reconstruction pass. Its planner selects the
actual moduli within the listed limit. INT8 uses HiFi4; the other formats use
LoFi. The results cover the throughput/accuracy comparison, size scaling and
phase-time breakdown for the Ozaki implementations.

`phi` uses BFP8 at 14080^3 with phi=0.5,1,2,4,8 and input
`(u-0.5)*exp(phi*z)`, with u uniform on [0,1) and z standard normal. The input
order and seed match the research generator `ozgen1`. Slice count/quantization
width are reselected from the input; Scheme II retains the conservative CRT
window. It runs Ozaki I and both one-pass and two-pass Ozaki II. A one-pass
result below 53 bits is an expected measurement, so it is recorded rather than
treated as an execution failure. The driver does not automatically discard the
less accurate pass count when comparing speed.

Examples for selecting a smaller part of the experiments:

```sh
./evaluate.sh table --sizes 3520 --formats BF16,BFP8
./evaluate.sh phi --phis 0.5,8
./evaluate.sh table --sizes 128 --formats BFP8 --warmup 2 --cooldown 0 --no-cpu
```

Overrides and the exact command are recorded in the run log. Use the default
paper sizes/conditions for a comparison with the publication. Custom dimensions
must be multiples of 64, so padding does not change the evaluated problem.
All formats must have been built for the default table; use `FORMATS` and
`--formats` consistently if building only a subset. Phi requires BFP8.

## Accuracy and the CPU comparison

The evaluator uses the paper's compensated Dot2 algorithm with an unrounded
FP64 two-word reference sum. Effective bits are
`-log2(||C-Cref||_F / ||Cref||_F)`, using the same reference normalization and
scaled Frobenius-norm calculation. The selected parameters and output hash are
also recorded.

Normal runs require a separate LP64 OpenBLAS installation for the CPU DGEMM
comparison. The executable dynamically loads `libopenblas.so.0`; `--blas PATH`
selects another installation. OpenBLAS is not bundled. The publication used
OpenBLAS 0.3.20 (Ubuntu 22.04 `libopenblas0-pthread`) on an AMD Ryzen Threadripper
PRO 9975WX with its Zen kernel; when reproducing that combination, set
`OPENBLAS_CORETYPE=ZEN` before invoking the script. The loaded OpenBLAS
configuration is printed in the run log. Other CPUs/libraries can give
different CPU results.

`fp64_equivalent=yes` requires both at least 53 effective bits and accuracy no
worse than the CPU DGEMM on the same input. `--no-cpu` skips the CPU comparison,
and a result at or above 53 bits then has equivalence `unknown`. `quick` uses
this setting by default. `--no-reference` skips both reference computations for
a performance-only run and writes `nan` in the accuracy columns; `output_hash`
is still written, so such a run can be compared bit for bit with another.

## Timing and output

Each case starts at or below 60 degrees C by default, with a 300-second cooling
timeout. Except for `quick`, `evaluate.sh` runs every case in its own process
(as the research harness does): an open device idles at the full clock and
only returns to the idle clock, where it can cool, when the process that opened
it exits. Inputs and reference values are therefore recomputed for each case. Each load group is warmed initially for at least 10 seconds and until
the time-average AICLK in two adjacent windows of at least one second, aligned
to the starts of the issued units, differs by less than 1 percent. Warmup is limited to 180 seconds. Short gaps are covered by additional
execution; a gap above 50 ms triggers a new clock-stability check. Normal runs
require readable device temperature/AICLK sysfs data. Failed cooling, clock
settling, execution, or reference calculation stops the run with a nonzero status.

Three repetitions are timed. The evaluator excludes allocation, parameter
selection, PCIe transfer, tiling/untiling, JIT compilation, initial execution,
CPU reference work and warmup from the reported times. Each scheme uses
the same partial-product/reconstruction order as its normal calculation. Two
timings are taken in one profiled call, as in the publication:

* **phase times**: each phase is warmed and repeated on its own with
  synchronization after every phase; `phase_sum_ms` is the sum of the phase
  means and is the breakdown only;
* **whole pass** (`total_ms`, the paper's Total): the buffers normally released
  between stages are kept, and after the last phase the complete sequence
  (scale, exponent deltas, slicing, all partial products, composition and
  reconstruction) is issued again without synchronization between stages,
  warmed as the group `whole`, and repeated. Ozaki II records every device step
  it issues during the phase timing and replays the records; Ozaki I issues its
  stage loops a second time with the same programs (with a single K block, the
  per-panel GEMM programs, which the phase timing built one at a time, are
  built again beforehand). The host program construction is outside this time;
  Ozaki II's composition arguments are set inside it, as in the research driver.
  Before timing, the intermediate and output buffers are filled with a
  sentinel, the pass is issued once and its output must be bit-identical to
  the phase-timed output, and the number of issued steps per kind must equal
  the count derived from the configuration; a failed check stops the run.

The whole pass is usually shorter than the phase sum. The compact evaluator does
not reproduce the research harness's full provenance checks or firmware
power/current telemetry. Device/runtime versions and thermal control can still
change measured numbers.

A new directory under `evaluation_results/` contains `results.csv` and `run.log`.
The log preserves the command, library hash, build flags, progress and errors.
Rows are flushed as cases finish; a failed run can leave a partial CSV.

Main CSV columns:

| Columns | Meaning |
|---|---|
| `suite,n,scheme,format,phi` | Evaluated case; phi=-1 denotes uniform input |
| `slices,integer_bits,moduli,passes,partial_products,bw` | Actual selected parameters |
| `scale_ms,slice_ms,gemm_ms,compose_ms,recon_ms,phase_sum_ms` | Mean phase times and their sum; Scheme I includes scale in slice |
| `total_ms` | Mean time of one uninterrupted whole pass (the paper's Total) |
| `effective_tflops` | 2*N^3 / whole-pass time |
| `gemm_tops` | 2*N^3*Npp / GEMM phase time |
| `nominal_peak_percent` | GEMM throughput / the used mode's peak at 1350 MHz and the device core count |
| `gemm_aiclk_mhz` | Time-average driver-reported AICLK during GEMM intervals |
| `effective_bits,relative_fro,cpu_dgemm_bits,fp64_equivalent` | Accuracy and the paper's measured-input criterion |
| `output_hash` | FNV-1a-style hash of the output bit patterns with the research implementation's initial value; written in every mode, also with `--no-reference` |
| `evaluation_wall_ms` | Diagnostic duration of the whole profiled call, including warmup/repetitions and host work |
| `timing,reference,start_temperature_c` | Timing mode, reference type, initial temperature |

`--help` on the shell entry point lists the available options. On a host where
the Metalium device ID differs from the sysfs index of the card (for example only
`/dev/tenstorrent/1` is visible in a container), pass `--sysfs-device 1`. Measurement
numbers should only be compared when the recorded configurations agree.
