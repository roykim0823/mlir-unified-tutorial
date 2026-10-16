# Part 1: Using MLIR

The user's-eye view: read IR fluently and drive existing tools — no C++
anywhere. Everything runs off the prebuilt wheel (`mlir-opt`,
`mlir-translate`, `FileCheck` all ship in it); setup is the repo
[README](../README.md)'s `pip install` step. Chapters 3–8 also run what they
compile, so they need a C compiler (`clang`), NumPy in the venv, and for
ch07's OpenMP examples `libomp` (`brew install libomp` / `apt install
libomp-dev`).

| Chapter | One line |
|---|---|
| [ch01: Why MLIR, and How to Read Its IR](ch01-why-mlir-and-reading-ir/README.md) | Ops, values, types, attributes, regions — the grammar of every `.mlir` file you'll ever see |
| [ch02: Lowering Pipelines with mlir-opt](ch02-lowering-pipelines/README.md) | Chain `--convert-*` passes and watch a program descend dialect by dialect |
| [ch03: LLVM IR and Native Code](ch03-llvm-ir-and-native-code/README.md) | Hand-written LLVM IR, a tour of the standard dialects, and the last step to objects, shared libraries and a JIT |
| [ch04: Memory: Tensors, Memrefs and the C ABI](ch04-memory-and-the-c-abi/README.md) | Bufferization up close, the memref descriptor, and calling compiled kernels from Python and C with NumPy data |
| [ch05: Optimizing with the Affine Dialect](ch05-affine-optimization/README.md) | Detour through `affine` while lowering: loop fusion and cache tiling, by flag |
| [ch06: Affine Maps and Loop Transformations](ch06-affine-maps-and-loop-transforms/README.md) | Affine maps in depth and the rest of the affine pass catalog — LICM, unrolling, scalar replacement, vectorization, interchange, skewing, conv2d |
| [ch07: Parallel Loops and OpenMP](ch07-parallel-loops-and-openmp/README.md) | Dependences, `--affine-parallelize`, and `scf.parallel` → `omp` → libomp, reductions included |
| [ch08: Linalg by Hand](ch08-linalg-by-hand/README.md) | Write linalg ops yourself — generic, map, named ops, reductions, broadcasts — then fuse and tile them with passes |
| [ch09: Testing with lit and FileCheck](ch09-testing-with-lit/README.md) | The `// RUN:` + `// CHECK:` idiom every LLVM-family project tests with |

If this is all you need (you *use* MLIR-based compilers rather than build
them), you can stop after this part — see the
[learning paths](../docs/learning-paths.md). Otherwise,
[Part 2](../part2-writing-passes/README.md) starts writing C++.
