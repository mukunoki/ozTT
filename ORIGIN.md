# Source and copy boundaries

This library was created from the authors' research repository (not public; revision
`a2f0301c3576c6c22345d9da928c764e05a40b1c`) and re-synchronized with its paper-data
snapshot `63e84a6c179bcf3e2865fb04e82b756aab55d3d6`. It is a separate copy: there are
no symlinks, includes, or build dependencies pointing back into the research
repository, and updating the research sources does not update this library. It is
maintained independently from here on (it was split out of that repository's
`release/` directory at commit c029f626).

`SOURCE_MAP.tsv` lists, for every library file that has a research origin, the
research file and its SHA-256 at the snapshot. The hashes identify the originals;
the library files are adapted as described below.

## What was taken

* The device kernels (`src/backend/*/kernels/`). Their code is the snapshot's code.
  Changes: comments and `static_assert` messages were translated to English, some
  files were renamed (and the `#include` lines that name them), and the argument
  table of an accumulation program that the library never issues was removed from
  `ozaki1/kernels/args.h` together with its three kernels. A comparison of every
  kernel file with the snapshot after removing comments and whitespace shows no other
  difference. Diagnostic selectors that the kernels still accept are always passed as
  zero.
* The host-side calculation code (`src/backend/*/*.hpp`): planning, exactness
  conditions, program construction, per-format constants, and the fitted coefficients
  of the Ozaki II stage-time model (`ozaki2/awe_coef.inc`, kept as algorithm data).
  Changes: private namespaces so that all schemes and formats coexist in one binary;
  research settings that were read from environment variables are fixed to their
  defaults and the branches that can then no longer be taken are removed; errors are
  reported as exceptions instead of messages on stderr; English comments; one code
  formatter for all host files.
* For the optional evaluator: the input generator, the Dot2 reference and the accuracy
  metric (`evaluation/numerics.hpp`).

## What was written for the library

* `include/ozTTgemm/ozTTgemm.hpp`, `src/context.cpp`, `src/dispatch.cpp`,
  `src/backend_interface.hpp`: the public interface, validation, padding, kernel
  lookup and device lifetime.
* `src/ozaki1.cpp`, `src/ozaki2.cpp`: the execution sequence of one GEMM call. They
  replace the research benchmark drivers and preserve the order of the partial
  products and of the reconstruction. Two deliberate differences from the research
  drivers: the scheme II panel planner also tries smaller divisors if the largest
  candidate cannot be mapped to the core grid (padded general shapes), and
  `Options::max_moduli` limits the number of moduli also when the precision is selected
  automatically.
* `src/evaluation_hooks.hpp`, `evaluation/evaluate.cpp`, `evaluate.sh`: the evaluator
  for the paper's two main experiments. The hooks are active only inside an explicit
  evaluator scope; ordinary library calls execute every device step once. Inside that
  scope the drivers repeat and time each stage and keep the buffers they would
  otherwise release, so that the complete sequence can be issued again without
  synchronization between stages (the paper's whole-pass total): the scheme II driver
  records every device step and replays the records, the scheme I driver issues its
  stage loops a second time. The evaluator can fix scheme II to the paper's b = 54
  without changing the public API's input-dependent default.

## What was left out

Research argument parsing and environment-variable readers, benchmark drivers,
telemetry, output dumps, forced configurations for experiments, host-side emulations
and check programs, and everything related to the manuscript. No TT-Metalium source or
binary is bundled.
