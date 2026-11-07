# Maintainer tools

Checks used while the library was cleaned up for distribution, kept so that a
later change can be validated the same way. Nothing here is part of the
library build or installation; the Makefile does not use this directory.
The programs need GNU bash, Python 3, git and (for the compile checks) GCC and
a built TT-Metalium tree, but no device: run them inside the Metalium
container, or on any host that has a Metalium build directory. Point
`TT_METAL_BUILD` at that directory; the default `/tt-metal/build_Release` is
the container's path.

```sh
export TT_METAL_BUILD=/path/to/tt-metal/build_Release
tools/check_kernels.sh [REV]              # kernels unchanged except comments
tools/hostcc.sh                           # host sources compile without warnings
tools/plan_dump/compare.sh [REV]          # host planners decide the same
tools/compare_results.py OLD.csv NEW.csv  # evaluator results bit-identical
tools/compare_reference.py NEW.csv        # evaluator results vs the paper
```

The scripts locate the release directory from their own path, which must be
inside the git repository; `REV` is any git revision (default `HEAD`) and the
other side is always the working tree. All scripts exit with 0 on success, 1 on
a finding, and 2 on a usage error.

## check_kernels.sh and codecheck.py

`check_kernels.sh REV` extracts `src/backend` of `REV` with `git archive` and
runs `codecheck.py` on every file under `src/backend/*/kernels/` of the working
tree. A file is OK when it is pure ASCII and its token stream, with comments
removed and whitespace ignored, equals that of `REV`; string literals that
contained non-ASCII text in `REV` may be reworded. Files added or removed also
count as differences. The check is textual (no preprocessor is run): any change
outside comments, including a changed macro or constant, is reported, and a
reported difference means the device kernels must be tested again.
`codecheck.py BASELINE NEW` can be used on one file pair directly.

## hostcc.sh

Compiles (without linking) every translation unit that the Makefile builds:
`src/ozaki1.cpp` and `src/ozaki2.cpp` for all eight formats, `src/context.cpp`,
`src/dispatch.cpp`, the evaluator, the test program and the example, all in
parallel, with the Makefile's flags plus `-Wall -Wextra
-Wno-missing-field-initializers`. It prints the exit status of every unit and
every diagnostic; the result is clean when every unit compiles with no output.
`OBJ` names a directory for objects and logs (default: temporary), `CXX` and
`CXXFLAGS` override the compiler and optimization flags (`g++`, `-O3 -DNDEBUG`,
as in the Makefile). `tools/hostcc.sh 1` or `2` compiles only one scheme's
drivers.

## plan_dump/compare.sh

The planners of both schemes run on the host and decide every parameter of a
call (slice counts, moduli, K blocks, panel shapes, bands, reconstruction
passes, scales) before anything reaches the device. `ozaki1_plan_dump.cpp` and
`ozaki2_plan_dump.cpp` link against the backend headers and print all of these
decisions for a fixed list of cases (fixed and automatic precision, small and
paper-sized shapes, several device grids and amounts of free DRAM, including
cases that must be rejected). `compare.sh REV` builds both programs for all
eight formats against `include/` and `src/` of `REV` and against the working
tree, runs every part of the case lists, and diffs the outputs per format and
part. "Identical" means that every one of the 56 outputs (2 schemes x 8
formats x 3 or 4 parts) of the working tree is byte-for-byte equal to the
baseline's, so the host change did not alter any planning decision; the device
kernels, which the dumps do not run, are checked separately. A differing output
is listed with its first lines of the diff, and the work directory with all
outputs is kept (`OUT` selects it beforehand). The link needs
`lib/libspdlog.a` from the Metalium build directory and `libquadmath`; no
device library is used. `JOBS` limits parallel builds and runs (default 16).
A full comparison takes about a minute on a 64-core host (9 CPU minutes); the
outputs are about 200 MB per side.

When the planner is changed on purpose, the differences are the review
material: check that every differing line is explained by the intended change,
then run the device tests. The dumps print internal structures, so a change of
a backend header can also require an update of the dump sources; they must
compile against both sides (a `BUILD FAILED` report shows the first errors and
keeps the work directory). A baseline older than the dump sources needs the
dump sources of its time.

## compare_results.py and compare_reference.py

`compare_results.py BASELINE.csv NEW.csv` compares two evaluator
`results.csv` files case by case: the selected parameters (`slices`,
`integer_bits`, `moduli`, `passes`, `partial_products`, `bw`), `output_hash`
and `effective_bits` must be equal; timings are ignored. Identical hashes show
that a change left the computed results bit-identical. The results of the last
validated revision are in `validated/`: `all.csv` (the complete paper plan, 63
cases), `quick.csv` (`evaluate.sh quick`, 25 cases) and `subset.csv` (seven
paper-plan cases); see `../VALIDATION.md`. Cases missing from the new file are
listed, not counted as failures, so a partial run can be compared with `all.csv`.

`compare_reference.py [--reference REF.csv] RESULTS.csv...` compares results
with `evaluation/paper_reference.csv` (the default) and prints the whole-call
time, the phase sum and their deviation from the paper in percent, the effective
bits of both, and `PARAMS DIFFER` when the selected parameters differ. Only
differing bits or parameters fail; timing deviations are informative.

## Workflow

Host code changed (`src/*.cpp`, `src/backend/*/*.hpp`):

1. `tools/hostcc.sh` must be clean.
2. `tools/plan_dump/compare.sh` against the last validated revision must report
   identical, or every difference must be explained by the change.
3. `make test` on a device, then `./evaluate.sh quick` and
   `tools/compare_results.py tools/validated/quick.csv <new results.csv>`: the
   hashes must be unchanged unless the numerical behavior was changed on
   purpose. For a change that can affect the paper-sized plans, run those cases
   (`./evaluate.sh table ...`, `./evaluate.sh phi ...`) and compare with
   `tools/validated/all.csv`. Record the result in `../VALIDATION.md` and
   replace the files in `validated/` when the validated revision moves.

Kernel comment or formatting changed (`src/backend/*/kernels/`):

1. `tools/check_kernels.sh` against the last validated revision must report
   all OK. Any `DIFF` means the kernel code changed and the device checks of
   the previous list are required.

Kernel code changed: `make test`, `./evaluate.sh quick`, `compare_results.py`,
and the paper-sized evaluation if performance may have changed.
