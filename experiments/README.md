# Experiments of the paper (research code, as run)

This directory is a verbatim copy of the research code that produced the numbers of
the paper, at the paper's measurement revision (research snapshot
`63e84a6c179bcf3e2865fb04e82b756aab55d3d6`, the same snapshot the library was adapted
from). It is kept separate from the library: `src/` here is the research
implementation (one program per scheme and format, with the measurement switches,
probes and baseline programs), not the library `../src/`. For the 63 cases of the
library's paper plan the two give bit-identical output (`../VALIDATION.md` compares the
library's output hashes with `../evaluation/paper_reference.csv`, which was taken from
the logs in `paper_logs/`). Comments and script messages are in Japanese; this README
is the English guide.

Use the library (`../evaluate.sh`) for the main results, Table III, Figs. 2-4 and the
exponent-range experiment. Use this directory for the numbers the library's evaluator
does not produce:

| Paper statement (English version) | Script here | Programs | Paper logs (`paper_logs/`) |
|---|---|---|---|
| Section VII-B: one BF16 GEMM (TT-Metalium matmul kernel, HiFi4, FP32 accumulation in DST), 81.3 TFlop/s and its peak ratios | `ONLY='^cool_bf16x2_' scripts/measure/performance.sh 14080` (the item's name is historical; it is one GEMM with FP32 accumulation, not a 2-term split) | `gemm_bf16` | `measure_logs/cool_bf16x2_14080.log` |
| Section VII-B: effective bits of that BF16 GEMM (about 6.5) | `scripts/measure/bf16_accuracy.sh` | `gemm_bf16` with verification | `measure_logs/bf16_verify/63e84a6/bf16_verify_14080.log` |
| Section VII-B: LoFi and HiFi2 of Ozaki-I-BF16 (1.07-1.54 times), one BF16 GEMM with LoFi and HiFi2 (AICLK 1146 and 941 MHz) | `ONLY='^(pass_\|gemmfid_(LoFi\|HiFi2)_)' scripts/measure/fidelity_comparison.sh` | `ozaki1_bf16`, `gemm_bf16` with `OZ_FIDELITY` | `measure_logs/pass_{LoFi,HiFi2}_{3520,7040,14080}.log`, `gemmfid_{LoFi,HiFi2}_14080.log` |
| Section VII-B: 1-core tile products (17.7-18.5 cycles; HiFi2 about 2 times LoFi) | `ONLY='^probe_tile_.*_mm$\|^probe_tilecfg_.*_mmpack$' scripts/diagnose/probes.sh` | `tile_rate_probe` | `measure_logs/probe_tile_*_mm.log`, `probe_tilecfg_*_mmpack.log` |
| Section VII-B: ablation runs (skipping the input supply or the product write-out; cycles per tile product per core) | `ONLY='^probe_design_(base\|nomcast\|nowrite\|nmnw\|sbase\|wbase)_' scripts/diagnose/design_comparisons.sh` | the 16 drivers with `OZ_*` switches | `measure_logs/probe_design_*.log` |
| Section VII-C: reconstruction of Ozaki-I-BF16 without the FP64 conversion (49-57 %) | `ONLY='^probe_rprobe[01]_' scripts/diagnose/probes.sh` | `ozaki1_bf16` with `OZ_RECON_PROBE` | `measure_logs/probe_rprobe[01]_{3520,14080}.log` |
| Section VII-C: reconstruction of Ozaki-II-BFP8 without CRT and/or FP64 assembly (about 1/6) | `SPLIT_FMTS=bfp8 scripts/diagnose/reconstruction_components.sh 14080` | `ozaki2_bfp8` with `OZ2_CRT_PROBE`, `OZ_RECON_PROBE` | `measure_logs/split_bfp8_14080_c?r?.log` |
| Section VII-A: choice of the slice count p (margin over 53 bits, constants, p+-1, p+-2) | `ONLY='^probe_pickp$' scripts/diagnose/probes.sh` (runs `scripts/diagnose/select_slice_count.sh`) | the scheme I drivers at 512 x 512 | `measure_logs/probe_pickp.log` |
| Section VII-A: GEMM-phase outputs error-free for integer inputs | `ONLY='^probe_gemm_exact$' scripts/diagnose/probes.sh` (runs `scripts/diagnose/gemm_exact.sh`) | `gemm_exact_check` | `measure_logs/probe_gemm_exact.log` |
| Section VII-A: the 4096^3 exponent-range check of 12 implementations with Table III's parameters (51.9 and 52.6 bits at phi = 0.5) | `scripts/measure/exponent_sweep.sh` inside the container (see below; no argument = phi 0 1 2 4 8 16, argument `0.5` = the phi = 0.5 point) | the 12 drivers | `accuracy_logs/phi_sweep.log`, `phi_sweep_05.log` |
| Section VII-A: CPU DGEMM reference (OpenBLAS): 50.58-50.71 effective bits at the 3 sizes, 50.59-53.75 bits at 14080^3 for phi = 0.5-8 | `DGEMMREF_CORETYPE=ZEN scripts/measure/cpu_dgemm_reference.sh` and `DGEMMREF_CORETYPE=ZEN DGEMMREF_SET=phi14080 scripts/measure/cpu_dgemm_reference.sh` (podman; builds `scripts/Dockerfile.dgemmref`; no device) | `tools/dgemmcheck.cpp` | `accuracy_logs/dgemm_ref.log`, `measure_logs/probe_dgemm_phi_14080.log` |
| Section VII-B: Ozaki-I-BFP8 with the same bw (11) as Ozaki-I-BF16 (13 % faster per clock) | `ONLY='^probe_bw11_' scripts/diagnose/probes.sh` | `ozaki1_bfp8` | `measure_logs/probe_bw11_oz1bfp8_14080.log` |
| Section V: the 8-term exponent alignment (two-term experiments: where a small product is rounded or lost) | `ONLY='^probe_micro$' scripts/diagnose/probes.sh` | `gemm_bf16ext` | `measure_logs/probe_micro.log` |
| Section V: 3xBF16 stays at 12.5 effective bits with HiFi4 at 14080^3 | `scripts/measure/floating_decomposition.sh 14080` (the paper's number is the block `=== fmt=bf16 terms=3`) | `gemm_bf16ext` | `measure_logs/ext_split_14080.log` |
| Section V (error-free input bound 32 with LoFi, 63 with HiFi2 and above) and Table I (d_fmt = 16 of FP8): sweep of the largest integer multiplied exactly per format and fidelity; Section VI-D: FP32 tiles of DST used by one reconstruction | `ONLY='^probe_(fp8_digit\|dst)$' scripts/diagnose/probes.sh` | `fp8_probe`, `dst_probe` | `measure_logs/probe_fp8_digit.log`, `probe_dst.log` |
| Table III, Figs. 2-4, phi experiment (also produced by the library) | `scripts/measure/performance.sh`, `scripts/measure/accuracy.sh`, `scripts/measure/exponent_adaptive.sh` | the 16 drivers | `measure_logs/cool_*.log`, `accuracy_logs/oz*.log`, `measure_logs/phi_nowin/63e84a6/*.log` |

`ONLY` is a regular expression on the log name (`grep -E`); the patterns above select
the items the paper's logs came from (the tile and design patterns also take in the same
measurement at formats and sizes the paper did not keep). Every item ends up as one log under
`measure_logs/` or `accuracy_logs/` (`measure_logs/bf16_verify/<rev7>/` and
`measure_logs/phi_nowin/<rev7>/` for rows 2 and 16, where `<rev7>` is the first 7
digits of the commit of the directory the scripts run in).

## What the scripts run

`scripts/measure/performance.sh`, `accuracy.sh`, `scripts/diagnose/probes.sh` and
`design_comparisons.sh` print their items with `--list` (or `ONLY=... --list`) without
touching the device: one line per item with the log name, the kind of check, and the
exact command that is run inside the container (`cd /home/user/tt && <environment>
./src/build/<program> <arguments>`). `exponent_adaptive.sh --commands` does the same;
`exponent_adaptive.sh --list` and `bf16_accuracy.sh --list` print the log names only.
The other scripts have no listing mode (they take sizes as arguments); their commands
are, with `N` in 3520 7040 14080:

    fidelity_comparison.sh        OZ_NOVERIFY=1 OZ_TIMING=steady OZ_FIDELITY={LoFi,HiFi2} ./src/build/ozaki1_bf16 N N N 10 9 1 3
                                  OZ_NOVERIFY=1 OZ_TIMING=steady OZ_FIDELITY={LoFi,HiFi2} ./src/build/gemm_bf16 14080 14080 14080 1 1 3 0 0 1 0
    reconstruction_components.sh  OZ_NOVERIFY=1 OZ_TIMING=steady OZ2_CRT_PROBE={0,1} OZ_RECON_PROBE={0,1} ./src/build/ozaki2_bfp8 14080 14080 14080 24 1 3
    floating_decomposition.sh     ./src/build/gemm_bf16ext 14080 14080 14080 1 3 {1,3,6}      (3xBF16 is terms=3)
    bf16_accuracy.sh              ./src/build/gemm_bf16 N N N 1 1 3 1 0 1 0
    exponent_sweep.sh             OZ_PHI=<phi> [OZ_FP64=1 | OZ2_FP64=1 OZ2_PASSES=2] <driver> 4096 4096 4096 <p L | s> 1 1 for the 12 drivers
    cpu_dgemm_reference.sh        compiles tools/dgemmcheck.cpp against OpenBLAS in the Dockerfile's image and runs it (no device)

These commands are what to run by hand in any TT-Metalium environment after building
`src/`; the scripts only add cooling, steady-state warm-up, telemetry, the provenance
line and the checks of the build around them. The header comment of each script
(Japanese) is the authoritative description of its arguments, environment variables and
output lines.

## Layout

    src/            research programs: ozaki1_<fmt>, ozaki2_<fmt> (the 16 implementations),
                    baseline/ (gemm_bf16, gemm_bf16ext), probes/, checks/, testkit/, common/;
                    one GNU Makefile (src/README.md, Japanese)
    tools/          host-only tools (tools/README.md, Japanese); those the Makefile builds are
                    its TOOLS targets, tools/study/ is not built by anything here
    scripts/        the measurement scripts of the table above and what they source:
                    run.sh (container launcher), device_lock.sh, provenance.sh,
                    build_all.sh, measure_lib.sh (cooling, steady-state timing),
                    routes.sh/routes.tsv (the 16 implementations' arguments),
                    lib/ (shared parts; knobs.tsv = the scripts' own variables, e.g. ONLY,
                    TMAX, OZ_TIMING), measure/, diagnose/
    paper_logs/     the 215 logs the paper's numbers were checked against
                    (measure_logs/..., accuracy_logs/..., same paths as in the research
                    repository) and manifest.tsv (SHA-256, size, path, revision and the
                    checker's name for each log; the header says the columns;
                    cd paper_logs && awk -F'\t' '!/^#/{print $1"  "$3}' manifest.tsv | sha256sum -c
                    verifies them)

The research repository's other parts (paper sources, figure scripts, the checker
that matches the paper's text against these logs, tests, older experiments) are not
included. Comments in the code and the scripts still refer to some of them
(`docs/*.md`, `scripts/README.md`, `scripts/selftest.sh`, `scripts/tests/*`,
`scripts/verify/*`, `scripts/rules/*`, `scripts/remeasure_all.sh`, `paper/*.tex`;
`scripts/dgemm_ref.sh` is the former name of `scripts/measure/cpu_dgemm_reference.sh`).
None of them is needed to build or to run the commands above.

## Building

Inside a TT-Metalium environment (the container of the Metalium release, or a host
with `TT_METAL_HOME` set and the Metalium build at `$TT_METAL_HOME/build_Release`):

```sh
cd experiments/src
make -j release        # the 16 implementations and the two baselines
make -j probes checks  # tile_rate_probe, fp8_probe, dst_probe, gemm_exact_check, ...
make -j                # everything, including the host tools under ../tools
```

The programs read their device kernels from the absolute path of this `src/` at build
time (`OZ_SRC_DIR` in the Makefile), so build them where they will run. Example runs
(the arguments of the two example rows of Table III):

```sh
OZ_NOVERIFY=1 OZ_TIMING=steady OZ_TIME_WHOLE=1 ./build/ozaki1_bf16 14080 14080 14080 10 9 1 3   # Ozaki-I-BF16
OZ_NOVERIFY=1 OZ_TIMING=steady ./build/gemm_bf16 14080 14080 14080 1 1 3 0 0 1 0              # one BF16 GEMM
./build/gemm_exact_check                                                                     # prints its usage
```

Positional arguments: scheme I drivers take `M N K p L warmup iters` (`p` slices,
`L = p-1` the last diagonal of slice products kept), scheme II drivers `M N K s warmup
iters` (`s` = number of moduli requested), `gemm_bf16` `M N K B warmup iters verify
in0_block_w fp32_dest_acc dst_full_sync` (`B = 1`; the paper's GEMM has
`fp32_dest_acc = 1`), `gemm_bf16ext` `M N K warmup iters terms`. `scripts/routes.tsv`
holds the values for the 16 implementations. `OZ_NOVERIFY=1` skips the reference
computation (performance runs), `OZ_TIMING=steady` warms up to the steady state,
`OZ_TIME_WHOLE=1` times the whole call (the paper's Total). The drivers' switches
(`OZ_FIDELITY`, `OZ_RECON_PROBE`, `OZ2_CRT_PROBE`, `OZ_PHI`, ...) are tabulated in
`src/common/oz_knobs.hpp`.

## Running the scripts as the paper did

The scripts were written for the research machine: the repository is mounted into a
Metalium container at `/home/user/tt` by `scripts/run.sh` (podman; the image, the
built Metalium tree (`$TT_METAL_SRC/build_Release`), the device file and the huge-page
mount are given by `TT_IMAGE`, `TT_METAL_SRC`, `TT_DEVICE`, `TT_HUGEPAGES`, see the
header of `run.sh`; the paper used the Metalium v0.73.1 image, see the `image=` and
`metal=` fields of the logs' provenance lines), cooling between items reads the chip
temperature with `tt-smi` on the host, and every log starts with a provenance line
(git revision, Metalium revision, image, board).

To run them, copy this directory to the host that has the Blackhole and make the copy
its own git repository (`git init && git add -A && git commit -m experiments`): the
scripts record the HEAD of the repository they are in as `rev=` in every log and refuse
to measure while `src/`, `scripts/` or `tools/` have uncommitted or untracked changes,
and `cpu_dgemm_reference.sh` reads its sources with `git show`/`git archive` relative to
the repository root, so they do not work while `experiments/` is a subdirectory of the
library's checkout. `rev=` in new logs is the copy's commit, not the research snapshot.
Then build inside the container with

    ./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/build_all.sh"

and call the scripts of the table from the copy's root on the host (`exponent_sweep.sh`
runs inside the container:
`./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/measure/exponent_sweep.sh [0.5]"`).
The host needs podman, python3, flock and tt-smi. Logs are written to `measure_logs/`
and `accuracy_logs/` of the copy (not into `paper_logs/`). Without that setup, run the
commands of "What the scripts run" by hand from the directory that holds `src/`
(replace `/home/user/tt` by it).

Timing and clock values depend on the board, firmware, Metalium version and the power
limit (`../README.md`, `../VALIDATION.md`). The accuracy values and the integer checks
do not need cooling, but they are results of the device's arithmetic and the Metalium
kernels of the recorded environment (the provenance line of each log), not constants.
