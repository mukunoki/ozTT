# ozTTgemm

C++20 library for FP64 matrix multiplication using Ozaki schemes I and II on
Tenstorrent Blackhole. The low-precision matrix engine computes exact integer
partial products; device kernels split and reconstruct the FP64 values.

This repository is a complete project: all required source files are included, and
TT-Metalium is the only external dependency. The library, its namespace, and its
header are named `ozTTgemm`. Version 0.1.0 (the macros `OZTTGEMM_VERSION_*` in the
public header).

## Build

Requirements: Linux, a Blackhole device with its driver/hugepages configured,
GCC with C++20, GNU Make, OpenMP, libquadmath, and a built TT-Metalium tree.
The implementation is tied to Metalium's low-level API; the version used for
validation (release `v0.73.1`) is recorded in [VALIDATION.md](VALIDATION.md).
Paths of the checkout and of the Metalium tree must not contain spaces (the
Makefile does not quote them).

The host code assumes what was validated: an x86-64 Linux host whose `long double`
is the 80-bit extended format with a 64-bit significand (the conversion of the
scales relies on it, and the build checks it), a round-to-nearest floating-point
environment on the host (the automatic precision selection uses `std::nearbyint`),
and GCC (the device kernels, which Metalium compiles with its RISC-V GCC, read bit
patterns through unions, a GCC-supported extension).

In an environment where Metalium's runtime can use the device:

```sh
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export ARCH_NAME=blackhole
make -j2
./build/ozTTgemm_example
```

The example prints `58 64` and `139 154`. Make produces `build/libozTTgemm.a`.
All eight internal formats are built by default: BF16, FP16, TF32, BFP8, FP8,
INT8, BFP4, and BFP2. BFP8/BFP4/BFP2 mean Metalium's block-floating formats.
To build a subset, use e.g. `make -j2 FORMATS="BF16 BFP8 INT8"`.
`ozTTgemm::is_available()` reports which formats were included.

The build disables implicit FMA contraction and fast-math because the
error-free transformations depend on the original evaluation order.

## Use from C++

```cpp
#include <ozTTgemm/ozTTgemm.hpp>
#include <vector>

int main() {
    ozTTgemm::Context context;  // default device 0; opens on the first valid call
    ozTTgemm::Options options;
    options.scheme = ozTTgemm::Scheme::II;
    options.format = ozTTgemm::Format::BFP8;
    std::vector<double> a{1, 2, 3, 4, 5, 6};  // 2 x 3, row-major
    std::vector<double> b{7, 8, 9, 10, 11, 12}; // 3 x 2, row-major
    auto c = context.multiply(a, b, 2, 2, 3, options);
}
```

For an existing output buffer, use
`context.gemm(a, b, c, m, n, k, options)`. Inputs and output are contiguous
row-major `double` spans with exactly `m*k`, `k*n`, and `m*n` elements.
The operation overwrites C with A*B. The returned `Info` reports the selected
precision and padded dimensions. Input/output spans may overlap because input
is copied before output is written. The method throws `std::invalid_argument` for
invalid input (before the device is opened and before C is written),
`std::runtime_error`, with the reason in the message, for configurations that
cannot be executed and for failures of the TT-Metalium runtime, and
`std::bad_alloc` when host memory runs out. The library itself never terminates
the process and writes nothing to stdout or stderr (TT-Metalium logs as
configured by its own settings). The constructor of `Context` resolves the kernel
directory and throws if no kernels are found; the device is opened by the first
valid call. Keep one `Context` alive for repeated calls to reuse the device.
Calls through this library are serialized within a process; coordinate device
ownership with other applications separately.

M and N are zero-padded to multiples of 64, K to a multiple of 32. The result is
cropped back to the requested shape. Device buffers and programs are rebuilt
for each call; this first API does not provide persistent device matrices or a
cached GEMM plan. It supports contiguous A*B only, without transpose, alpha/beta,
or batched operations.

### Precision and input range

* Scheme I: `slices=0` selects the number of slices from the input using the
  implementation's error model. Set a positive `slices` value (up to 256) to
  use a fixed number; all diagonals through `slices-1` are retained.
  The internal reconstruction range can impose a smaller limit for a given
  format and inner dimension; a count beyond it, fixed or selected
  automatically, is rejected with `std::runtime_error`.
* Scheme II: precision is selected from the input. `max_moduli=0` leaves the
  number of CRT moduli to the library; a positive value up to 24 limits it, and
  a limit too small for FP64-equivalent precision is rejected.
  The CRT window uses the conservative bound for the padded inner dimension.
  `reconstruction` selects automatic, one-pass, or two-pass reconstruction. The
  automatic choice uses an error estimate from the input statistics; when the
  estimated product norm is zero (strong cancellation or underflow in the
  estimate) it has no estimate and takes one pass. Use `TwoPass` when that
  matters.
* Automatic precision targets approximately FP64 precision using a model. It
  does not guarantee correctly rounded FP64 GEMM, nor uniform relative error
  for arbitrary cancellation patterns. Validate accuracy on your application's
  inputs. Fixed slices or forced one-pass reconstruction may reduce accuracy.
* Inputs must be finite. Every nonzero row maximum of A and column maximum of B
  must be between 2^-126 and the largest finite BF16 value, inclusive, because
  device normalization uses BF16 maxima. Zero rows and columns are supported.
  A row or column whose largest magnitude is subnormal is therefore rejected;
  subnormal elements of an admissible row or column are treated as zero.
* Each row of A and column of B is aligned to its maximum: an element many
  binades below that maximum keeps fewer bits, and one further below it than the
  retained width (about 54 to 64 bits with the default settings) is dropped
  entirely, even when the matching element of the other operand is large. No
  check detects this; it is a property of the scaling per row and column.
* A result that would be an FP64 subnormal is returned as zero (with its sign).
* The exact-accumulation bounds, available memory, and 32-bit device indexing
  limit supported sizes. A configuration that does not fit is rejected.

The library reads no environment variables of its own. Metalium's runtime
environment settings still apply.

## Link and install

For a library kept in its source tree, obtain the public compile/link flags
from the Makefile. For example, build `my_application.cpp` from another directory:

```sh
c++ $(make -s -C /path/to/ozTTgemm print-cflags) my_application.cpp \
  $(make -s -C /path/to/ozTTgemm print-libs) -o my_application
```

`make library` builds only the static library; `make example` builds the usage
example. `CXX`, `CPPFLAGS`, `CXXFLAGS`, `LDFLAGS`, `LDLIBS`, and `BUILD_DIR` can
be overridden. `TT_BUILD` can point directly to the built Metalium distribution
if it is not under `$TT_METAL_HOME/build_Release`.

To install the library, public header, runtime kernels, and documentation:

```sh
make -j2 PREFIX=/desired/prefix
make PREFIX=/desired/prefix install
```

Use the same `PREFIX` for build and install, and the same `TT_METAL_HOME` (or
`TT_BUILD`): the build configuration is recorded and a different value at install
time triggers a rebuild, so with `sudo` pass them explicitly (`sudo make
TT_METAL_HOME=... PREFIX=... install`) or use `sudo -E`. `DESTDIR=/staging/root` is supported
for staged installation. Build flags and enabled formats are tracked, so a
subsequent `make` with changed settings updates the affected outputs. The
documents go to `share/ozTTgemm/` in the layout of this repository; the evaluator,
`tools/` and their READMEs are part of the source tree only, so the links to them
in an installed copy of this file lead back to the repository.

The installation provides `lib/pkgconfig/ozTTgemm.pc`. With pkg-config available,
a separate application can link with:

```sh
export PKG_CONFIG_PATH=/desired/prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
c++ $(pkg-config --cflags ozTTgemm) my_application.cpp \
  $(pkg-config --libs ozTTgemm) -o my_application
pkg-config --variable=kernelsdir ozTTgemm
```

The pkg-config file derives its prefix from its own location, so it follows a
moved installation. Metalium's library path is external and remains the one
selected at build time.

Kernels are compiled by Metalium at runtime, so the installed sources must
remain available. The default lookup checks this library's source directory and
configured install prefix. When moving the installation, pass the installed
kernel directory explicitly (the value of `pkg-config --variable=kernelsdir ozTTgemm`):

```cpp
ozTTgemm::DeviceOptions device;
device.device_id = 0;
device.kernel_directory = "/desired/prefix/share/ozTTgemm/kernels";
ozTTgemm::Context context(device);
```

Default kernel lookup is independent of the process's working directory; an
explicit relative `kernel_directory` is resolved when the context is created.
Copying source after building requires a rebuild or an explicit
`kernel_directory`. Absolute kernel paths are resolved for each context.

## Paper evaluation

This library accompanies a paper on DGEMM emulation with the Ozaki schemes on
Blackhole (under review at the time of writing; the reference is added here when
it is published). "The paper" below means that paper. An optional script runs its
two main evaluations, the 16-implementation comparison and the BFP8 exponent-range
experiment:

```sh
./evaluate.sh quick
./evaluate.sh all --list
./evaluate.sh table
./evaluate.sh phi
```

It writes the whole-pass total (the time of one uninterrupted GEMM call, the
paper's "Total"), the phase timings and the accuracy against a compensated
reference (Dot2) to CSV. Paper-sized runs are long; start with `quick` or `--list`. OpenBLAS is needed for the CPU comparison in normal
runs. Parameters, timing definitions and output fields are described in
[evaluation/README.md](evaluation/README.md).

## Correctness checks

```sh
make test-host
make test
```

Use the same `FORMATS` setting as the library build when running these targets.
The first check validates API errors without opening a device; the executable
still links against Metalium. The second runs small correctness cases against a
long-double CPU reference, plus one small-output case with a long inner dimension
that exercises scheme I's K blocking. These checks are not performance benchmarks.

### Maintainer checks

`tools/` holds checks for changes to this library that need no device: a
warning-free host compile of every unit, a comparison of every host planning
decision between a git revision and the working tree, a check that kernel files
changed only in comments, and comparisons of evaluator CSVs. They are not part
of the build; [tools/README.md](tools/README.md) describes them and the workflow.

## Source layout

```
include/ozTTgemm/ozTTgemm.hpp     public API (the only installed header)
src/context.cpp               validation, padding, kernel lookup, device lifetime
src/dispatch.cpp              (format, scheme) -> backend
src/backend_interface.hpp     interface between the two above and the backends
src/ozaki1.cpp, ozaki2.cpp    one GEMM call of scheme I / II; compiled once per format
src/evaluation_hooks.hpp      timing hooks used only by the optional evaluator
src/backend/common/           shared host code: formats, GEMM program builder, scaling
src/backend/ozaki1/           scheme I host code: planning, slicing, GEMMs, reconstruction
src/backend/ozaki2/           scheme II host code: moduli, planning, stages
src/backend/*/kernels/        device kernels (compiled by Metalium at run time)
evaluation/, evaluate.sh      paper evaluator
tests/, examples/             correctness checks, usage example
docs/sfpu_lemmas.md           proofs for the error-free steps of the triple-float kernels
tools/                        maintainer checks (not built); tools/validated/ holds the
                              evaluator results of the last validated revision
```

In `src/backend/`, the directory names the scope, so file names carry no scheme
prefix (`ozaki1/plan.hpp`, `ozaki2/plan.hpp`). Host headers are `.hpp`; kernel
sources are `.cpp` with `.h` headers. Every file starts with a comment that says
what it provides; the kernels document the numerical and hardware reasoning in
detail. Host sources are formatted with the `.clang-format` in the repository root;
the kernels are not reformatted.

## Origin and license

The calculation code derives from a research implementation; see
[ORIGIN.md](ORIGIN.md) for what was taken and changed, [SOURCE_MAP.tsv](SOURCE_MAP.tsv)
for the original files, and [VALIDATION.md](VALIDATION.md) for what was tested.
Code is distributed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
