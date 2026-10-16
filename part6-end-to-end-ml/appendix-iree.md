# Appendix: IREE, a Production MLIR-Based ML Compiler

> Reworked from ML-compiler-exercise's IREE appendix for the unified repo.

Part 6 hand-assembled an ML compiler: import (ch27), lowering pipelines and
an ABI (ch28), a library backend (ch29), a GPU backend (ch31). This
appendix is a short survey of [IREE](https://iree.dev/) — an MLIR-based
compiler *and runtime* built by people doing the same job at production
scale, aimed at real-time inference from servers down to edge and mobile
devices. Reading its architecture after building the toy version is the
payoff: every component maps onto something you now know by name.

## The compiler

IREE accepts models through framework importers that produce MLIR in an
input dialect — Torch models via
[iree-turbine](https://github.com/iree-org/iree-turbine) (the same
torch-dialect route as ch27), TensorFlow Lite via `tosa`, plus plain
`linalg`/`arith` input like our snapshots. From there the descent runs
through three IREE-specific dialect layers
([architecture overview](https://iree.dev/#project-architecture)):

- **`flow`** — captures high-level structure and partitions the program
  into *dispatch regions*: chunks of computation that can be scheduled as
  units. This is our ch28 pipeline's role, done with a real cost model —
  and where our per-op `gpu.launch_func`s (ch31) would be fused into far
  fewer dispatches.
- **`stream`** — models the partitioned work as asynchronous streams of
  data and execution, so independent dispatches overlap and transfers hide
  behind compute. The toy pipeline has no analogue: our host code launches
  everything synchronously in order.
- **`hal`** — the Hardware Abstraction Layer: explicit buffers, command
  buffers, and executables, still as IR. It is the compiler-side twin of
  the runtime API below — roughly our MemRefDescriptor ABI plus the
  `mgpu*` wrapper layer, made into a first-class dialect.

Backends then lower each executable for its target: the LLVM backend for
CPUs (x86, ARM — our ch28 flow), SPIR-V for Vulkan/OpenCL GPUs, NVVM/PTX
for CUDA (our ch31 flow), and others. One program can carry executables
for several targets side by side — the generalization of our
`gpu.binary` + host-code split.

## The runtime

The compiler's output is a *module* — a shared-library-like artifact
holding the executable binaries (execution logic) plus the scheduling
program (host logic), which runs on a small VM inside the IREE runtime.
The runtime dynamically loads modules into isolated contexts, drives
devices through the same HAL, and exposes API bindings (C, Python, and a
growing list). Contrast with Part 6's runtime story — a C++ `main`, a
hand-written descriptor struct, and `-lopenblas`/`-lmlir_cuda_runtime` —
and you see what a real deployment story adds: versioned artifacts,
multi-device scheduling, and language bindings, without recompiling the
model per host program.

## Why this matters for you

The skills transfer one-to-one. `flow`'s dispatch formation is dialect
design plus the tiling/fusion reasoning of ch22; `stream` is an IR for the
asynchrony ch31 hid inside `mgpu*` calls; `hal` is ABI design as a
dialect; the backends are ch18/ch19 conversions at scale. If Part 6 was
readable, the IREE codebase is approachable — start at the
[architecture page](https://iree.dev/#project-architecture) and follow one
of your models through `iree-compile --mlir-print-ir-after-all`, the same
way ch02 taught you to watch a pipeline work.
