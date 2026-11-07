# Validation record

Validated on 2026-11-06 with the sources as distributed, using a Tenstorrent Blackhole
P150a (110 compute cores), firmware 19.15.0.0 and KMD 2.11.0 on an AMD Ryzen
Threadripper PRO 9975WX host. Host compilation used GCC 12.3.0 in the Metalium Ubuntu
22.04 container. The built TT-Metalium checkout was
`ec685d2af0a5374016b76f67b5677a12b360988a`; this is the tested API version, not a claim
of compatibility with other releases.

## What was checked

* **Build.** GNU Make compilation of all eight formats for both Ozaki schemes into one
  static library, the usage example, the evaluator and the test program, without
  warnings under `-Wall -Wextra -Wno-missing-field-initializers`. `make install` into a
  prefix.
* **API input validation** without opening the device (`make test-host`).
* **Device correctness** (`make test`): 50 GEMM cases. 48 are compared with a
  long-double CPU reference, with all relative Frobenius errors at most 6.77e-17: every
  scheme/format combination with uniform and with nonnegative inputs and padding; zero
  matrices, exact cancellation, row/column scaling, odd dimensions, both ends of the
  normalization range, a fixed slice count, forced two-pass reconstruction, a scheme II
  panel fallback, and a scheme I calculation crossing the K-block limit. The other two
  check overlapping input/output and that a moduli limit too small for FP64-equivalent
  precision is rejected with `std::runtime_error` without touching the output.
* **Evaluator hooks.** Profiling with repeated execution and the whole-pass replay
  produced exactly the same result bits as ordinary library calls for both schemes; the
  replay passed its sentinel, bit-identity and step-count checks; phase timing without
  the replay, the fixed b=54 override and timing-stage attribution were checked
  separately.
* **`evaluate.sh quick`**: 25 cases at 128^3 (all 16 fixed-parameter scheme/format
  combinations and nine adaptive BFP8 phi cases), each with the whole-pass checks.
* **The complete paper plan**, `evaluate.sh all`: 63 cases under the paper conditions
  (cooling to 60 degrees C, clock-settled warmup, three repetitions, Dot2 reference,
  OpenBLAS 0.3.20 Zen-kernel CPU DGEMM), one process per case, in a single run: the 48
  table cases (16 scheme/format combinations at 3520^3, 7040^3 and 14080^3) and the 15
  BFP8 phi cases at 14080^3 (phi = 0.5, 1, 2, 4, 8; Ozaki I and Ozaki II with one and two
  passes). Ozaki II at 14080^3 with BF16, FP16, BFP2 and TF32 does not fit in device DRAM
  at once and is processed in output row bands by the planner.

## Results of the paper plan

Against `evaluation/paper_reference.csv` (measured with the research implementation on
the same machine):

* The selected parameters (slices, integer bits, moduli, passes, partial products) were
  equal in all 63 cases.
* The output hashes were identical, in all 63 cases, to those of the research
  implementation's results, so the effective bits were equal as well.
* The whole-pass total differed by at most 2.0 percent in 60 cases and by at most 3.1
  percent in all; the phase sum differed by at most 4.6 percent in 62 cases and by 5.5
  percent in one (Ozaki I-INT8 at 14080^3). In an earlier run of the same plan two
  timing outliers occurred that did not reproduce when the cases were run again (a
  whole pass 28 percent longer, a phase sum 18 percent longer); the cause was not
  identified. The evaluator records the clock only for the GEMM phase.
* 59 cases met the measured-input FP64-equivalence criterion. The four that did not are
  the one-pass Ozaki II cases at phi = 1, 2, 4 and 8 (52.7, 49.2, 43.8 and 34.5 bits),
  where one pass is expected to fall short; they are recorded as such.

## Relation to the research implementation

The device kernels were compared file by file with the research snapshot named in
`ORIGIN.md` after removing comments and whitespace: the code is identical except for
the argument table of an unused accumulation program removed from
`ozaki1/kernels/args.h`. The host code was restructured for the library; its
equivalence rests on the identical output hashes above and on the correctness checks.

After the run of the paper plan, two comment lines pointing to `docs/sfpu_lemmas.md`
were added to one kernel header. The code was unchanged by the same comparison, and
the device correctness checks and `evaluate.sh quick` were repeated with the final
sources (same output hashes as before).

## Re-validation after the review changes (2026-11-07)

Five independent reviews of the sources above led to host-side simplifications
(removal of an unreachable measured-window path and of an always-whole-set slicer
abstraction in scheme II, of constant parameters and fields in both schemes, the
evaluator's recorded steps discarded at the end of every profiled call, validation
before allocation in `multiply`, the input contract stated in the public header) and
to comment corrections in the kernels. The kernel code is still identical to the
snapshot by the comparison above. The host planners were compared without a device
before and after the changes: every planning decision (slices, K blocks, panels,
bands, GEMM configurations and offsets of scheme I for 8 formats x 1003 cases x 3 core
grids x 8 DRAM sizes; moduli, width, passes, panels, bands and reconstruction
constants of scheme II for 8 formats x 1712 cases) was identical. On the device, the
build was warning-free, the correctness checks passed (the 50 cases, with an added check that
an evaluator scope with zero repetitions is rejected), `evaluate.sh quick`
produced the same 25 output hashes and parameters as before, and five cases of the
paper plan chosen to exercise the changed paths (14080^3 Ozaki I-BF16 and the banded
Ozaki II-BF16; phi = 2 with Ozaki I, one-pass and two-pass Ozaki II) reproduced the
output hashes and parameters of the complete run, with times within 0.4 percent
(whole pass) and 1.5 percent (phase sum) of `paper_reference.csv`. `make test-host`
and `make install` were repeated.

A second review round on that result led to further simplifications (scheme I: one
slice count instead of three copies, format-fixed values as compile-time constants,
unreachable fidelity/digit-bound branches and a linear search removed, the three format
headers merged into one, an unused configuration field and parameter removed, the L1
estimate no longer builds a kernel specification, the reconstruction programs are no
longer executed once at construction, exceptions in the OpenMP region are propagated;
scheme II: an always-true reconstruction flag, duplicated modulus-limit plumbing, an
unused pass estimate, a constant runtime argument and a duplicate enumeration removed;
interface: records of the evaluator discarded at the start of every call, evaluator
settings reset per scope, value validation before the output allocation in `multiply`,
host assumptions stated) and to comment corrections in the kernels and in
`docs/sfpu_lemmas.md`. The same checks were repeated: kernel code identical to the
snapshot; host planner dumps byte-identical before and after (scheme I 8 formats x 3
parts, scheme II 8 formats x 1712 cases); on the device the build was warning-free, the
correctness checks passed (with an added check that evaluator settings reused in a
second scope start from zero), `evaluate.sh quick` reproduced the 25 output hashes and
parameters, and seven paper-plan cases (14080^3 with BF16 and INT8 for both schemes;
phi = 2 with Ozaki I, one-pass and two-pass Ozaki II) reproduced the output hashes and
parameters of the complete run, with times within 1.1 percent (whole pass) and 2.6
percent (phase sum) of `paper_reference.csv`. The sequence of device programs of scheme
I is shorter by the removed construction-time executions; the results are unchanged.

On 2026-11-07 the library, its namespace, header, macros and build products were
renamed from `ttgemm` to `ozTTgemm` (`OZTTGEMM_*`); the kernels were not touched and
the host code changed only in the names and in line breaks. The kernel check reported
all 45 files unchanged, the host compile check was clean, the planner dumps were
byte-identical to the revision before the rename (56 outputs), and on the device the
correctness checks passed and `evaluate.sh quick` reproduced the 25 output hashes and
parameters of `tools/validated/quick.csv`.

The evaluator output files behind these statements are kept in `tools/validated/`:
`all.csv` (the complete paper plan, 63 cases, run on 2026-11-06 after the clean-up;
its hashes equal those of the first complete run), `quick.csv` (the 25 cases of
`evaluate.sh quick`) and `subset.csv` (the seven paper-plan cases), both from the
re-validation of 2026-11-07. `tools/compare_results.py` compares a new results file
with them.

## Not repeated in this validation

These were checked on 2026-10-27 with an earlier state of this directory; the build
system has changed only in details since (object names, warning flags, the installed
documents):

* Incremental builds left the archive untouched when settings did not change; switching
  to one format removed the other archive members, and restoring the full format list
  restored all 16 backends.
* An independently copied source tree built outside its original location, was
  installed, the install prefix was moved, the source tree was hidden, and a separate
  C++ consumer linked using the installed pkg-config metadata and ran a GEMM from
  another working directory using the installed kernels.
* The evaluator's input generator, Dot2 high/low arrays and accuracy metrics matched
  the research implementation bit for bit on small inputs.
* Invalid evaluator arguments were rejected before device use.

## Scope

These are correctness checks and one reproduction run, not a performance study. They
do not establish an error guarantee for arbitrary input, compatibility with other
Metalium releases, or production performance. The values in
`evaluation/paper_reference.csv` come from the research implementation, not from this
evaluator.

Operational note: do not kill an evaluation while device programs are running. In an
earlier validation a killed run was followed, without a board reset, by a run that
stalled, after which device initialization failed until a board-level reset.
