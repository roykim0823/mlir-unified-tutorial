# Learning Paths

Thirty-one chapters is a book. Not everyone needs the whole book, and the
parts were designed so you don't have to read it as one. Pick the row that
matches why you're here.

| You are... | Read | Skip-ahead allowed? |
|---|---|---|
| **An MLIR user** — you run compilers built on MLIR (IREE, torch-mlir, HEIR) and want to read the IR and drive the tools | Part 1 (ch01–09) | ch01, ch02, ch05 and ch09 are the core; ch03–04 and ch06–08 go deeper. Stop after Part 1; return for ch22 if you meet the transform dialect in the wild |
| **A dialect author** — you're building a compiler *on* MLIR: your own ops, types, rewrites | Parts 1→2→3→4, then ch21, ch23 + ch24 | ch01–02 skimmable if you already read IR fluently; ch06–08 optional |
| **An ML compiler engineer** — you care about models-in, binaries-out | Part 1 (incl. ch04, ch07, ch08), Part 2, ch18–19, ch22, then Part 6 | Part 3 on demand (it's the "custom ops" part); ch21/ch31 when GPU work starts |

## The MLIR user path (1–2 days)

Part 1 never writes a line of C++.
- [ch01](../part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md)
  teaches reading IR.
- [ch02](../part1-using-mlir/ch02-lowering-pipelines/README.md) drives
  lowering pipelines with `mlir-opt`.
- [ch03](../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md)
  follows the result out of MLIR into LLVM IR, objects and a JIT.
- [ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md) explains
  what bufferization produces and how to call compiled code with real data.
- [ch05](../part1-using-mlir/ch05-affine-optimization/README.md) and
  [ch06](../part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md)
  apply loop optimizations by flag.
- [ch07](../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md)
  parallelizes loops with OpenMP.
- [ch08](../part1-using-mlir/ch08-linalg-by-hand/README.md) writes the
  structured linalg ops that ML frontends emit.
- [ch09](../part1-using-mlir/ch09-testing-with-lit/README.md) teaches
  lit/FileCheck, the skill you need to *read* any MLIR project's tests.

Everything runs off the prebuilt wheel plus a C compiler; no build step
beyond `pip install`. In a hurry: ch01, ch02, ch05, ch09 (half a day).

## The dialect author path (1–2 weeks)

The poly-dialect arc:
- **Part 2 (ch10–11)** writes passes.
- **Part 3 (ch12–17)** is the deepest content in the repo: a full dialect
  with types, traits, folders, verifiers, and canonicalizers in three
  pattern languages (C++, DRR, PDLL).
- **Part 4 (ch18–20)** lowers it out of existence, ending in a native
  binary.

Then three Part 5 chapters complete the toolbox:
- [ch21](../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md):
  designing your own op interface.
- [ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md):
  whole-program analysis, the finale of the arc.
- [ch24](../part5-structured-and-parallel/ch24-equality-saturation-egraphs/README.md):
  equality saturation, the alternative to the greedy rewriting you used in
  Part 3.

[ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)
is optional here but recommended: it shows what upstream already provides
before you build your own, and how to extend the transform dialect in C++.

## The ML compiler engineer path (about 2 weeks)

1. **Part 1 for fluency**, with special attention to:
   - [ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md): the
     memref ABI every compiled model uses.
   - [ch07](../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md):
     CPU parallelism.
   - [ch08](../part1-using-mlir/ch08-linalg-by-hand/README.md): the linalg
     ops importers emit.
2. **Part 2**, then the lowering spine without the poly detours:
   - [ch18](../part4-lowering/ch18-dialect-conversion/README.md): dialect
     conversion. Read it even though it uses poly; the machinery is what
     Part 6 runs on.
   - [ch19](../part4-lowering/ch19-to-llvm-native-binary/README.md): to
     LLVM and a native binary.
   - [ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md):
     linalg and scheduling, the lingua franca of ML compilers.
3. **All of [Part 6](../part6-end-to-end-ml/README.md):**
   - ch25–26 hand-write a dense layer and a softmax.
   - ch27–29 take real PyTorch models through torch-mlir to CPU binaries
     and BLAS.
   - ch30 scales up to resnet18, bert and flan-t5 and benchmarks against
     PyTorch.
   - ch31 compiles to PTX.
4. **When GPU work starts**, add
   [ch21](../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md)
   and [ch31](../part6-end-to-end-ml/ch31-gpu-nvvm/README.md) together.
   They're two views (DSL-down and loops-down) of the same SIMT target.
5. **Finish with the [IREE appendix](../part6-end-to-end-ml/appendix-iree.md)**
   to map what you built onto a production compiler.

## Cross-cutting references

Whatever the path:
- the [glossary](glossary.md), when a term rings half a bell;
- the [source-build appendix](appendix-source-build.md), the day the wheel
  stops being enough (custom LLVM, CUDA runner, debugging upstream);
- [other ways to install MLIR](appendix-installing-mlir.md), when you want
  system-wide tools such as `llc` and `mlir-runner`.
