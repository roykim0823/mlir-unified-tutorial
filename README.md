# MLIR Unified Tutorial

A single organized MLIR tutorial, consolidating five source repos:
[j2kun/mlir-tutorial (fork)](https://github.com/roykim0823/mlir-tutorial),
mlir-tutor (PPoPP-2026 exercises), ML-compiler-exercise (end-to-end pipelines),
mlir-beginner-friendly-tutorial (user-view intro), and mlir-ir-lowering
(hand-written IR, after Stephen Diehl's "MLIR" article series).

Everything builds against a **prebuilt MLIR pip wheel** — no LLVM source build.

## Setup

```sh
uv venv && source .venv/bin/activate   # or python3 -m venv .venv
uv pip sync requirements.txt           # or pip install -r requirements.txt
uv pip install -e python/              # tiny DSL frontend (Parts 4-5); also pulls in numpy
python -m mlir_wheel --root-dir        # sanity check: prints the wheel's path (used below)

mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) -DLLVM_EXTERNAL_LIT=$(which lit)
cmake --build . -j
cmake --build . --target check-all
```

> The wheel index (`https://llvm.github.io/eudsl`) is a rolling nightly feed that
> keeps roughly one month of builds. If `uv pip sync` reports the pinned version
> missing, pick the newest version present for both `mlir-wheel` and
> `mlir-python-bindings` on the index and update `requirements.txt`.

## Roadmap

| Part | Chapters | Source material |
|---|---|---|
| [1. Using MLIR](part1-using-mlir/README.md) | ch01 why-mlir/reading IR · ch02 lowering pipelines · ch03 LLVM IR & native code · ch04 memory & the C ABI · ch05 affine optimization · ch06 affine maps & loop transforms · ch07 parallel loops & OpenMP · ch08 linalg by hand · ch09 testing with lit | beginner-friendly demos 1–4, mlir-tutorial ch1–2, mlir-ir-lowering 1–4 |
| [2. Writing passes](part2-writing-passes/README.md) | ch10 first pass in C++ · ch11 tablegen passes | mlir-tutorial ch3–4 |
| [3. Building a dialect](part3-building-a-dialect/README.md) | ch12 defining poly · ch13 traits · ch14 folders · ch15 verifiers · ch16 canonicalizers/DRR · ch17 PDLL | mlir-tutorial ch5–9, 13 |
| [4. Lowering](part4-lowering/README.md) | ch18 dialect conversion · ch19 to LLVM & native binary · ch20 region ops → SCF | mlir-tutorial ch10–11, mlir-tutor ch1–2 |
| [5. Structured & parallel](part5-structured-and-parallel/README.md) | ch21 op interfaces & GPU tiles · ch22 linalg + transform dialect · ch23 dataflow analysis · ch24 equality saturation & e-graphs | mlir-tutor ch3–4, mlir-tutorial ch12, mlir-ir-lowering 6 |
| [6. End-to-end ML (capstone)](part6-end-to-end-ml/README.md) | ch25 neural nets from scratch · ch26 transformers & softmax · ch27 torch-mlir import · ch28 CPU pipeline & ABI · ch29 BLAS integration · ch30 real models · ch31 GPU/NVVM · IREE appendix | mlir-ir-lowering 5, 7, 8; ML-compiler-exercise (own Python env) |

### Where each chapter comes from

The five source repos, with the short names used below:

| Short name | Repo | What it is |
|---|---|---|
| **beginner** | [mlir-beginner-friendly-tutorial](https://github.com/AlexandreSinger/mlir-beginner-friendly-tutorial) (Alexandre Singer, MIT) | Demos 0–4: a user's-eye introduction |
| **j2kun** | [mlir-tutorial fork](https://github.com/roykim0823/mlir-tutorial) of Jeremy Kun's [MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) | 13 chapters on passes, the `poly` dialect and lowering |
| **tutor** | [mlir-tutor](https://github.com/Groverkss/mlir-tutor) (PPoPP-2026 tutorial) | 4 chapters building the `tiny` DSL dialects, with exercise branches |
| **diehl** | mlir-ir-lowering, in [roykim0823/mlir](https://github.com/roykim0823/mlir), after Stephen Diehl's [MLIR articles](https://www.stephendiehl.com/tags/mlir/) | 8 chapters of hand-written IR, Parts 0–8 of the article series |
| **mlce** | [ML-compiler-exercise](https://github.com/DavidGinten/ML-compiler-exercise) (David Ginten) | 6 chapters + IREE appendix: PyTorch models end to end |

| Ch | Title | Source | Notes |
|---|---|---|---|
| 01 | [Why MLIR, and How to Read Its IR](part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md) | beginner Demo 1 "Motivating MLIR", Demo 2 "Entering MLIR" | + sigil table from diehl's top README |
| 02 | [Lowering Pipelines with mlir-opt](part1-using-mlir/ch02-lowering-pipelines/README.md) | j2kun ch1 "MLIR Basics and Running a Lowering" (Part A); beginner Demo 3 "Lowering MLIR" (Part B) | |
| 03 | [LLVM IR and Native Code](part1-using-mlir/ch03-llvm-ir-and-native-code/README.md) | diehl `1_intro` (Part 1 "Introduction to MLIR") | + top README's compiler-stack, standard-dialect and toolchain sections; `mlir-runner` → ExecutionEngine |
| 04 | [Memory: Tensors, Memrefs and the C ABI](part1-using-mlir/ch04-memory-and-the-c-abi/README.md) | diehl `2_memory` (Part 2 "Memory in MLIR") | llvmlite JIT → ExecutionEngine |
| 05 | [Optimizing with the Affine Dialect](part1-using-mlir/ch05-affine-optimization/README.md) | beginner Demo 4 "Optimizing MLIR with Affine Analysis" | |
| 06 | [Affine Maps and Loop Transformations](part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md) | diehl `3_parallel` steps 1–4 (Part 3 "Affine Dialect and OpenMP") | parallelization moved to ch07 |
| 07 | [Parallel Loops and OpenMP](part1-using-mlir/ch07-parallel-loops-and-openmp/README.md) | diehl `3_parallel`: `--affine-parallelize` from steps 1 and 4, and step 5 (OpenMP) | |
| 08 | [Linalg by Hand](part1-using-mlir/ch08-linalg-by-hand/README.md) | diehl `4_linalg` (Part 4 "Linear Algebra in MLIR") | |
| 09 | [Testing with lit and FileCheck](part1-using-mlir/ch09-testing-with-lit/README.md) | j2kun ch2 "Testing a Lowering" | |
| 10 | [Writing Your First Pass in C++](part2-writing-passes/ch10-first-pass-cpp/README.md) | j2kun ch3 "Writing Our First Pass" | |
| 11 | [Tablegen for Passes](part2-writing-passes/ch11-tablegen-passes/README.md) | j2kun ch4 "Using Tablegen for Passes" | |
| 12 | [Defining the Poly Dialect](part3-building-a-dialect/ch12-defining-poly/README.md) | j2kun ch5 "Defining a New Dialect" | |
| 13 | [Using Traits](part3-building-a-dialect/ch13-traits/README.md) | j2kun ch6 "Using Traits" | |
| 14 | [Folders and Constant Propagation](part3-building-a-dialect/ch14-folders/README.md) | j2kun ch7 "Folders and Constant Propagation" | |
| 15 | [Verifiers](part3-building-a-dialect/ch15-verifiers/README.md) | j2kun ch8 "Verifiers" | |
| 16 | [Canonicalizers and DRR](part3-building-a-dialect/ch16-canonicalizers-drr/README.md) | j2kun ch9 "Canonicalizers and Declarative Rewrite Patterns" | |
| 17 | [Patterns with PDLL](part3-building-a-dialect/ch17-pdll/README.md) | j2kun ch13 "Defining Patterns with PDLL" | moved next to the other pattern chapters |
| 18 | [Dialect Conversion](part4-lowering/ch18-dialect-conversion/README.md) | j2kun ch10 "Dialect Conversion"; tutor ch1 "CPU Vector DSL" (TinyToArith) | + tutor ch1 exercises (ODS ops, TinyToArith) |
| 19 | [Lowering to LLVM and a Native Binary](part4-lowering/ch19-to-llvm-native-binary/README.md) | j2kun ch11 "Lowering through LLVM"; tutor ch1 "CPU Vector DSL" (TinyToLLVM, `square.py`) | + tutor ch1 TinyToLLVM exercise and matmul challenge |
| 20 | [Region Ops and Lowering to SCF](part4-lowering/ch20-region-ops-to-scf/README.md) | tutor ch2 "CPU Vector DSL with Loops" | |
| 21 | [Op Interfaces and a GPU Tile DSL](part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md) | tutor ch3 "GPU Vector Tile DSL" | |
| 22 | [Linalg and the Transform Dialect](part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md) | tutor ch4 "Linalg and Transform Dialect" | |
| 23 | [Dataflow Analysis and Global Optimization](part5-structured-and-parallel/ch23-dataflow-analysis/README.md) | j2kun ch12 "A Global Optimization and Dataflow Analysis" | |
| 24 | [Equality Saturation and E-graphs](part5-structured-and-parallel/ch24-equality-saturation-egraphs/README.md) | diehl `6_egraph` (Part 6 "Specializing Python with E-graphs") | |
| 25 | [Neural Networks from Scratch, then Compiled](part6-end-to-end-ml/ch25-neural-nets-from-scratch/README.md) | diehl `5_neural_network` (Part 5 "Neural Networks") | |
| 26 | [Transformers and a Compiled Softmax](part6-end-to-end-ml/ch26-transformers-and-softmax/README.md) | diehl `7_transformer` (Part 7 "Transformers") | |
| 27 | [Importing PyTorch Models with torch-mlir](part6-end-to-end-ml/ch27-torch-mlir-import/README.md) | mlce ch1 "Introduction", ch2 "Getting started and project setup", ch3 "Importing PyTorch models to torch-mlir" | + T5 export and lifted buffers from mlce ch3–4 |
| 28 | [The CPU Pipeline and the C ABI](part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md) | mlce ch4 "Lowering models to x86 machine code" | |
| 29 | [Calling External Libraries — matmul to OpenBLAS](part6-end-to-end-ml/ch29-external-libs-blas/README.md) | mlce ch5 "Integration of OpenBLAS for Matrix Multiplications" | |
| 30 | [Scaling Up to Real Models](part6-end-to-end-ml/ch30-real-models/README.md) | mlce ch4 (bert, flan-t5 execution) and its per-model READMEs, scripts and benchmarks (`src/{cnn,resnet18,bert-base-uncased,flan-t5-small}`) | |
| 31 | [Targeting an NVIDIA GPU through NVVM](part6-end-to-end-ml/ch31-gpu-nvvm/README.md) | mlce ch6 "Targeting an Nvidia GPU"; diehl `8_gpu` (Part 8 "GPU Compilation with MLIR") | + Diehl's [gpu-offload](https://github.com/sdiehl/gpu-offload) Python pipeline |

Appendices:

| Appendix | Source |
|---|---|
| [IREE, a Production MLIR-Based ML Compiler](part6-end-to-end-ml/appendix-iree.md) | mlce "An overview of IREE" appendix |
| [Other Ways to Install MLIR](docs/appendix-installing-mlir.md) | diehl top README "Installation and setup" (Part 0 "Installing MLIR") |
| [Building MLIR from Source](docs/appendix-source-build.md) | written for this repo; its torch-mlir in-tree recipe is mlce ch2's build |
| [Glossary](docs/glossary.md), [Learning paths](docs/learning-paths.md) | written for this repo |

All 31 chapters are written. Not sure where to start? See the
[learning paths](docs/learning-paths.md); there is also a
[glossary](docs/glossary.md), a
[source-build appendix](docs/appendix-source-build.md) and an
[appendix on other ways to install MLIR](docs/appendix-installing-mlir.md).

Each chapter directory contains `README.md` (the chapter text), worked
examples, and where applicable `exercises/`, all validated by per-part lit
targets (`check-part1` ... `check-part6`, or `check-all`).

## Layout

```
lib/        dialect/pass/conversion sources (poly, tiny, tiny_loop, tiny_tile, noisy)
tools/      tutorial-opt driver; ll_fp_compat.py (LLVM 24 .ll -> older clang);
            np_memref.py (NumPy <-> memref descriptors for ctypes);
            mlir_to_object.py (LLVM-dialect module -> .o, for huge models)
test/       lit suites, one directory per part
partN-*/    chapter text, examples, exercises
python/     tiny Python DSL frontend (drives tutorial-opt via TUTORIAL_OPT)
docs/       appendices (source build, other installs, glossary, learning paths)
```

## Notes

- The wheel ships `mlir-opt`, `mlir-translate`, `mlir-pdll`, `FileCheck`, etc.,
  but **no `llc` and no `mlir-runner`** — native-binary chapters compile LLVM
  IR with `clang -x ir`, and JIT examples use the Python bindings'
  `ExecutionEngine`. Because the wheel is LLVM 24 and the system clang is
  usually older, `.ll` files go through `tools/ll_fp_compat.py` first.
- Chapters 3–8 and 24–26 run their kernels from Python (NumPy + ctypes) or
  small C drivers. ch07's OpenMP examples need `libomp`; its runtime lit
  test is `REQUIRES: libomp` and is skipped where none is found.
- The ch23 noisy-dialect optimizer needs or-tools; it is an optional
  component so `check-all` stays laptop-friendly. Configure a second build
  directory with `-DENABLE_SOLVER=ON` (see ch23's README) to enable the
  pass and its `REQUIRES: solver` lit test.
- Part 6's Python side (torch + torch-mlir) has its own environment —
  torch-mlir wheels are Linux-only, so on macOS the import scripts run in a
  container (`part6-end-to-end-ml/generate_snapshots.sh`). The C++ side and
  its lit tests run everywhere via checked-in IR snapshots; the BLAS tests
  link Accelerate on macOS and OpenBLAS elsewhere. ch30's four models
  (cnn, resnet18, bert, flan_t5) are ported but **their snapshots have not
  been generated yet** — run `./generate_snapshots.sh cnn resnet18 bert
  flan_t5` (Docker) before following that chapter's compiled steps.
- **Known gap:** the GPU chapters (ch21, ch31) are verified to IR and real
  PTX (ch31 now covers sample and mnist, with cnn/resnet18 pending their
  snapshots), but nothing has been *executed* on an NVIDIA GPU — this repo was
  built on Apple silicon. ch31's `compile.sh` +
  [the source-build appendix](docs/appendix-source-build.md) document the
  CUDA path; running it and reporting drift is the most valuable
  contribution available.

## References

- [Official MLIR website](https://mlir.llvm.org/)
- [MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/)
  by Jeremy Kun (Parts 2–5 grew out of it), especially
  [MLIR lowering through LLVM](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/)
- [Stephen Diehl's MLIR articles](https://www.stephendiehl.com/tags/mlir/),
  including GPU compilation (the basis of mlir-ir-lowering, i.e. ch03–08,
  ch24–26 and parts of ch31)
- [torch-mlir](https://github.com/llvm/torch-mlir) and its
  [development guide](https://github.com/llvm/torch-mlir/blob/main/docs/development.md)
