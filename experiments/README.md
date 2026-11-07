# Experiments of the paper (research code, as run)

This directory is a verbatim copy of the research code that produced the numbers of
the paper, at the paper's measurement revision (research snapshot
`63e84a6c179bcf3e2865fb04e82b756aab55d3d6`, the same snapshot the library was adapted
from). It is kept separate from the library: `src/` here is the research
implementation (one program per scheme and format, with the measurement switches,
probes and baseline programs), not the library `../src/`. The two give bit-identical
results for the paper's cases (`../VALIDATION.md` compares the library's output hashes
with the logs in `paper_logs/`). Comments and script messages are in Japanese; this
README is the English guide.

Use the library (`../evaluate.sh`) for the main results, Table III, Figs. 2-4 and the
exponent-range experiment. Use this directory for the numbers the library's evaluator
does not produce:

| Paper statement (English version) | Script here | Programs | Paper logs (`paper_logs/`) |
|---|---|---|---|
| Section VII-B: one BF16 GEMM (TT-Metalium matmul kernel, HiFi4, FP32 accumulation in DST), 81.3 TFlop/s and its peak ratios | `scripts/measure/performance.sh --list 14080` (items `cool_bf16x2`, `cool_bf16single`) | `gemm_bf16` | `measure_logs/cool_bf16x2_14080.log` |
| Section VII-B: effective bits of that BF16 GEMM (about 6.5) | `scripts/measure/bf16_accuracy.sh` | `gemm_bf16` with verification | `measure_logs/bf16_verify/63e84a6/bf16_verify_14080.log` |
| Section VII-B: LoFi and HiFi2 of Ozaki-I-BF16 (1.07-1.54 times), one BF16 GEMM under the 4 fidelities (1146 and 941 MHz) | `scripts/measure/fidelity_comparison.sh` | `ozaki1_bf16`, `gemm_bf16` with `OZ_FIDELITY` | `measure_logs/pass_*.log`, `fid_hifi4_*.log`, `gemmfid_*.log` |
| Section VII-B: 1-core tile products (17.7-18.5 cycles; HiFi2 about 2 times LoFi) | `ONLY='^probe_tile(cfg)?_' scripts/diagnose/probes.sh` | `tile_rate_probe` | `measure_logs/probe_tile_*.log`, `probe_tilecfg_*.log` |
| Section VII-B: ablation runs (skipping the input supply or the product write-out; cycles per tile product per core) | `scripts/diagnose/design_comparisons.sh` (variants `base`, `nomcast`, `nowrite`, `nmnw`, `sbase`, `wbase`) | the 16 drivers with `OZ_*` switches | `measure_logs/probe_design_*.log` |
| Section VII-C: reconstruction of Ozaki-I-BF16 without the FP64 conversion (49-57 %) | `ONLY='^probe_rprobe' scripts/diagnose/probes.sh` | `ozaki1_bf16` with `OZ_RECON_PROBE` | `measure_logs/probe_rprobe[01]_{3520,14080}.log` |
| Section VII-C: reconstruction of Ozaki-II-BFP8 without CRT and/or FP64 assembly (about 1/6) | `SPLIT_FMTS=bfp8 scripts/diagnose/reconstruction_components.sh 14080` | `ozaki2_bfp8` | `measure_logs/split_bfp8_14080_c?r?.log` |
| Section VII-A: choice of the slice count p (margin over 53 bits, constants, p+-1, p+-2) | `ONLY='^probe_pickp$' scripts/diagnose/probes.sh` (runs `scripts/diagnose/select_slice_count.sh`) | the scheme I drivers at 512 x 512 | `measure_logs/probe_pickp.log` |
| Section VII-A: GEMM-phase outputs error-free for integer inputs | `ONLY='^probe_gemm_exact$' scripts/diagnose/probes.sh` (runs `scripts/diagnose/gemm_exact.sh`) | `gemm_exact_check` | `measure_logs/probe_gemm_exact.log` |
| Section VII-A: the 4096^3 exponent-range check of 12 implementations with Table III's parameters (51.9 and 52.6 bits) | `scripts/study/phi_sweep.sh` (inside the container) | the drivers | `accuracy_logs/phi_sweep*.log` |
| Section VII-A: CPU DGEMM reference (OpenBLAS) effective bits | `scripts/measure/cpu_dgemm_reference.sh` (builds `scripts/Dockerfile.dgemmref`) | `tools/dgemmcheck.cpp` | `accuracy_logs/dgemm_ref.log` |
| Section VII-B: Ozaki-I-BFP8 with the same bw (11) as Ozaki-I-BF16 (13 % faster per clock) | `ONLY='^probe_bw11_' scripts/diagnose/probes.sh` | `ozaki1_bfp8` | `measure_logs/probe_bw11_oz1bfp8_14080.log` |
| Section V: the 8-term exponent alignment (two-term experiments: where a small product is rounded or lost) | `ONLY='^probe_micro$' scripts/diagnose/probes.sh` | `gemm_bf16ext` | `measure_logs/probe_micro.log` |
| Section V: 3xBF16 stays at 12.5 effective bits with HiFi4 at 14080^3 | `scripts/measure/floating_decomposition.sh 14080` | `gemm_bf16ext` | `measure_logs/ext_split_14080.log` |
| Section III (Table II): the largest digit multiplied exactly in each format and fidelity (sweep); Section VI-D: FP32 tiles of DST used by one reconstruction | `ONLY='^probe_(fp8_digit|dst)$' scripts/diagnose/probes.sh` | `fp8_probe`, `dst_probe` | `measure_logs/probe_fp8_digit.log`, `probe_dst.log` |
| Table III, Figs. 2-4, phi experiment (also produced by the library) | `scripts/measure/performance.sh`, `scripts/measure/accuracy.sh`, `scripts/measure/exponent_adaptive.sh` | the 16 drivers | `measure_logs/cool_*.log`, `accuracy_logs/oz*.log`, `measure_logs/phi_nowin/63e84a6/*.log` |

Every script prints its items with `--list` (or `ONLY=... --list`) without touching the
device: one line per item with the log name, the kind of check, and the exact command
that is run inside the container (`cd /home/user/tt && <environment> ./src/build/<program>
<arguments>`). That command is what to run by hand in any TT-Metalium environment
after building `src/`; the scripts only add cooling, steady-state warm-up, telemetry
and provenance around it. The header comment of each script (Japanese) is the
authoritative description of its arguments, environment variables and output lines.

## Layout

    src/            research programs: ozaki1_<fmt>, ozaki2_<fmt> (the 16 implementations),
                    gemm_bf16, gemm_bf16ext (baselines), probes/, checks/, testkit/;
                    one GNU Makefile (src/README.md, Japanese)
    tools/          host-only tools built by src/Makefile (tools/README.md, Japanese)
    scripts/        the measurement scripts of the table above and what they source:
                    run.sh (container launcher), device_lock.sh, provenance.sh,
                    build_all.sh, measure_lib.sh (cooling, steady-state timing),
                    routes.sh/routes.tsv (the 16 implementations' arguments),
                    lib/ (shared parts, knobs.tsv = environment variables),
                    measure/, diagnose/, study/
    paper_logs/     the 215 logs the paper's numbers were checked against
                    (measure_logs/..., accuracy_logs/..., same paths as in the research
                    repository) and manifest.tsv (SHA-256, size, path, revision and the
                    checker's name for each log)

The research repository's other parts (paper sources, figure scripts, the checker
that matches the paper's text against these logs, older experiments) are not included.

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
(arguments are M N K and the implementation's parameters; see `scripts/routes.tsv`):

```sh
OZ_TIMING=steady ./build/ozaki1_bf16 14080 14080 14080 10 9 1 3      # Ozaki-I-BF16, p=10
OZ_TIMING=steady ./build/gemm_bf16 14080 14080 14080 1 1 3 0 0 1 0   # one BF16 GEMM (the baseline row)
./build/gemm_exact_check                                              # prints its usage
```

`OZ_NOVERIFY=1` skips the reference computation (performance runs); `OZ_FIDELITY`,
`OZ_RECON_PROBE` and the other switches are listed in `scripts/lib/knobs.tsv` and in the
drivers' `src/common/oz_knobs.hpp`.

## Running the scripts as the paper did

The scripts were written for the research machine: the repository is mounted into a
Metalium container at `/home/user/tt` by `scripts/run.sh` (podman; the image, the
Metalium tree, the device file and the huge-page mount are given by `TT_IMAGE`,
`TT_METAL_SRC`, `TT_DEVICE`, `TT_HUGEPAGES`, see the header of `run.sh`), cooling
between items reads the chip temperature with `tt-smi` on the host, and every log
starts with a provenance line (git revision of this directory's repository, Metalium
revision, image, board). To run them, copy this directory to the host that has the
Blackhole, build inside the container with
`./scripts/run.sh "OZ_PROVENANCE='$(bash scripts/provenance.sh)' bash /home/user/tt/scripts/build_all.sh"`,
then call the scripts of the table from this directory on the host. Logs are written
to `measure_logs/` and `accuracy_logs/` here (not into `paper_logs/`). Without that
setup, run the commands that `--list` prints by hand.

The measured values depend on the board, firmware, Metalium version and the power
limit (`../README.md`, `../VALIDATION.md`); the accuracy values and the integer
checks do not.
