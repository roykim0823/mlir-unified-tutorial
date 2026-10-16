# Glossary

Terms as this tutorial uses them, with the chapter that introduces each
(linked on the chapter number). Definitions are deliberately informal;
the linked chapter has the precise story.

**affine dialect** — Loops and memory accesses restricted to affine
(linear-plus-constant) index expressions, which is what makes tiling,
fusion, and dependence analysis decidable. The subject of
[ch05](../part1-using-mlir/ch05-affine-optimization/README.md) and
[ch06](../part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md); revisited
as a GPU on-ramp in [ch31](../part6-end-to-end-ml/ch31-gpu-nvvm/README.md).

**affine map** — A function from dimension and symbol identifiers to index
expressions built from `+`, `*` by constants, `floordiv`, `ceildiv` and
`mod`, e.g. `(d0, d1)[s0] -> (d0 * s0 + d1)`. Loop bounds, subscripts and
linalg indexing maps are all affine maps.
([ch06](../part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md))

**attribute** — Compile-time constant data attached to an op (shapes,
integer values, affine maps). Contrast *operand*: runtime value.
([ch01](../part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md))

**block / region** — A block is a list of ops with block arguments (MLIR's
phi nodes); a region is a list of blocks owned by an op. Ops-with-regions
are how MLIR nests structure (`func.func`, `scf.for`, `linalg.generic`).
([ch01](../part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md),
[ch20](../part4-lowering/ch20-region-ops-to-scf/README.md))

**big-M trick** — ILP encoding of a conditional as linear inequalities
using a large constant. Used to model "either reduced noise or
input-dependent noise" in
[ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md).

**bufferization (one-shot)** — The whole-module conversion from
value-semantics `tensor`s to memory-semantics `memref`s, including
allocation placement and copy avoidance. `bufferize-function-boundaries`
extends it across function signatures.
([ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md),
[ch28](../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md))

**canonicalization** — Rewriting IR toward a preferred equivalent form
(`x*2` → `x<<1`), run as the `--canonicalize` pass from patterns registered
per-op. ([ch16](../part3-building-a-dialect/ch16-canonicalizers-drr/README.md))

**conversion pattern** — A rewrite pattern that may change *types*, used
with a type converter under the dialect-conversion framework; receives both
original and type-converted operands.
([ch18](../part4-lowering/ch18-dialect-conversion/README.md), applied to
FFI in [ch29](../part6-end-to-end-ml/ch29-external-libs-blas/README.md))

**dataflow analysis** — Computing facts about every SSA value by iterating
transfer functions to a fixed point over a lattice. MLIR packages it as
`DataFlowSolver`.
([ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md))

**destination-passing style** — The caller supplies the buffer a result is
written into (linalg's `outs`, an out-parameter memref), instead of the
callee allocating and returning one. It lets bufferization write in place.
([ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md),
[ch28](../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md))

**dense_resource** — A named binary blob stored in a module's
`dialect_resources` section, referenced by constants; how imported model
weights live inside `.mlir` files.
([ch27](../part6-end-to-end-ml/ch27-torch-mlir-import/README.md))

**dialect** — A namespace of ops, types, and attributes loaded into the
context; MLIR's unit of extensibility. Defining one is Part 3
([ch12](../part3-building-a-dialect/ch12-defining-poly/README.md)).

**dialect conversion** — The framework for lowering with type changes:
conversion target (legal/illegal ops), type converter, patterns, and
`applyPartialConversion`/`applyFullConversion`.
([ch18](../part4-lowering/ch18-dialect-conversion/README.md))

**DRR (Declarative Rewrite Rules)** — Tablegen syntax for source→target
pattern rewrites; generates C++ patterns.
([ch16](../part3-building-a-dialect/ch16-canonicalizers-drr/README.md))

**e-graph / equality saturation** — An e-graph stores many equivalent
programs at once by grouping e-nodes into equivalence classes. Equality
saturation applies every rewrite until nothing new appears, then extracts
the cheapest program, so rewrite order no longer matters.
([ch24](../part5-structured-and-parallel/ch24-equality-saturation-egraphs/README.md))

**ExecutionEngine** — The MLIR Python bindings' JIT: it compiles an
LLVM-dialect module in-process and lets Python call its functions. This
repo uses it in place of `mlir-runner`, which the wheel doesn't ship.
([ch03](../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md),
[ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md))

**FileCheck** — LLVM's output-matching tool (`// CHECK:` lines); with lit,
the testing idiom of every LLVM-family project.
([ch09](../part1-using-mlir/ch09-testing-with-lit/README.md))

**folding** — An op computing its own result at compile time when operands
are constant (`fold()` methods); the engine behind constant propagation
and a prerequisite for canonicalization.
([ch14](../part3-building-a-dialect/ch14-folders/README.md))

**FX importer / `fx.export_and_import`** — torch-mlir's recommended entry:
`torch.export` traces a PyTorch module, the importer turns the FX graph
into MLIR at a chosen `OutputType`.
([ch27](../part6-end-to-end-ml/ch27-torch-mlir-import/README.md))

**gpu dialect / kernel outlining** — Target-independent GPU concepts:
`gpu.launch` regions, `gpu.module` device code, `gpu.launch_func`
host-side calls; `gpu-kernel-outlining` performs the host/device split.
([ch31](../part6-end-to-end-ml/ch31-gpu-nvvm/README.md); the SIMT mental
model is [ch21](../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md))

**ILP (integer linear programming)** — Optimization over integer/continuous
variables under linear constraints; ch23 uses a solver (or-tools) as a
compiler analysis to place ops optimally.
([ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md))

**interface (op/dialect)** — A named set of methods an op (or dialect)
implements so generic code can query unknown ops; "a trait with methods."
Defining your own:
[ch21](../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md);
implementing upstream's:
[ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md).

**lattice** — The ordered set of possible facts in a dataflow analysis
(here: integer ranges from uninitialized to unbounded), with a join that
guarantees convergence.
([ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md))

**linalg dialect / `linalg.generic`** — Structured ops that *are* loop
nests, with the loop structure in `indexing_maps` + `iterator_types`
attributes rather than control flow.
([ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md))

**linalg-on-tensors** — The linalg+arith+tensor form that ML frontends emit
and backends consume; torch-mlir's backend-contract output and Part 6's
interchange format.
([ch27](../part6-end-to-end-ml/ch27-torch-mlir-import/README.md))

**lit** — LLVM's test runner: each test file carries its own `// RUN:`
command lines. ([ch09](../part1-using-mlir/ch09-testing-with-lit/README.md))

**lowering** — Rewriting IR from a higher-level dialect to a lower one,
usually via dialect conversion; "the descent." (Part 4, starting at
[ch18](../part4-lowering/ch18-dialect-conversion/README.md))

**MemRefDescriptor** — The struct a `memref` becomes in LLVM: allocated +
aligned pointers, offset, per-dim sizes and strides. The ABI between
MLIR-compiled code and C/C++ (and, via `tools/np_memref.py`, NumPy).
([ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md),
[ch19](../part4-lowering/ch19-to-llvm-native-binary/README.md),
[ch28](../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md))

**memref vs tensor** — `tensor` is an SSA value with no memory identity;
`memref` is a view of actual memory. Bufferization crosses the bridge.
([ch18](../part4-lowering/ch18-dialect-conversion/README.md),
[ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md))

**`_mlir_ciface_` wrapper** — The C-friendly entry point emitted for
functions carrying `llvm.emit_c_interface` (or for all functions with
`--llvm-request-c-wrappers`), passing each memref as a pointer-to-descriptor
instead of exploded scalars.
([ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md),
[ch28](../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md))

**NVVM dialect** — NVIDIA's LLVM-IR flavor as an MLIR dialect: PTX special
registers (`nvvm.read.ptx.sreg.*`), barriers, etc.; the last MLIR stop
before the NVPTX backend.
([ch31](../part6-end-to-end-ml/ch31-gpu-nvvm/README.md))

**ODS (Operation Definition Specification)** — Defining ops/types/dialects
in tablegen, generating the C++ classes.
([ch12](../part3-building-a-dialect/ch12-defining-poly/README.md))

**omp dialect / OpenMP** — MLIR's model of OpenMP parallel regions and
worksharing loops (`omp.parallel`, `omp.wsloop`). `--convert-scf-to-openmp`
produces it from `scf.parallel`, and it lowers to calls into the libomp
runtime (`__kmpc_*`).
([ch07](../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md))

**op (operation)** — MLIR's universal IR node: operands, results,
attributes, regions, all uniformly structured. Everything — modules,
functions, constants — is an op.
([ch01](../part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md))

**pass / pass pipeline** — A unit of IR transformation, run by the pass
manager; pipelines compose passes under one flag
(`PassPipelineRegistration`).
([ch10](../part2-writing-passes/ch10-first-pass-cpp/README.md),
[ch11](../part2-writing-passes/ch11-tablegen-passes/README.md),
[ch28](../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md))

**PDLL** — The pattern language that compiles to `pdl` dialect IR and runs
on a bytecode interpreter — patterns as data.
([ch17](../part3-building-a-dialect/ch17-pdll/README.md))

**PTX / cubin** — NVIDIA's virtual assembly (text, from LLVM's NVPTX
backend) and the assembled device binary (from `ptxas`). `cubin-format=isa`
stops at PTX and needs no CUDA toolkit.
([ch31](../part6-end-to-end-ml/ch31-gpu-nvvm/README.md))

**rewrite pattern** — A matchAndRewrite unit applied greedily
(`applyPatternsGreedily`) or via drivers; the workhorse of ch10's
`MulToAdd`, canonicalizers, and lowerings.
([ch10](../part2-writing-passes/ch10-first-pass-cpp/README.md))

**scf dialect** — Structured control flow: `scf.for`, `scf.if`,
`scf.forall` — loops as ops with regions, between high-level dialects and
the CFG world. ([ch20](../part4-lowering/ch20-region-ops-to-scf/README.md))

**scf.parallel** — A loop whose iterations may run in any order or at the
same time, optionally with reductions. The common starting point for both
CPU threads (OpenMP) and GPU kernels.
([ch07](../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md),
[ch31](../part6-end-to-end-ml/ch31-gpu-nvvm/README.md))

**SIMT** — Single instruction, multiple threads: the GPU execution model
where a kernel body runs per-thread, indexed by block/thread ids.
([ch21](../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md))

**SSA (static single assignment)** — Every value defined exactly once;
merges happen via block arguments.
([ch01](../part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md))

**tablegen** — LLVM's DSL for generating C++ from declarative records;
MLIR uses it for ops (ODS), passes, patterns (DRR), interfaces.
([ch11](../part2-writing-passes/ch11-tablegen-passes/README.md))

**tiling / fusion** — Splitting an iteration space into cache- or
tile-sized blocks; fusing producer loops into consumers. By flags in
[ch05](../part1-using-mlir/ch05-affine-optimization/README.md), under
precise control in
[ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md).

**torch dialect** — torch-mlir's MLIR mirror of PyTorch's ATen operator
set (`torch.aten.*` on `!torch.vtensor`); the "backend contract" form.
([ch27](../part6-end-to-end-ml/ch27-torch-mlir-import/README.md))

**trait** — A compile-time marker attached to an op in ODS (`Pure`,
`Commutative`, `SameOperandsAndResultType`) that generic code can test;
"an interface with no methods."
([ch13](../part3-building-a-dialect/ch13-traits/README.md))

**transform dialect** — Scheduling as IR: a transform script (tiling,
fusion, vectorization directives) that an interpreter applies to payload
IR; extensible with your own transform ops.
([ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md))

**type converter** — The dialect-conversion component mapping illegal types
to legal ones (e.g. `!poly.poly<10>` → `tensor<10xi32>`, memref → LLVM
struct). ([ch18](../part4-lowering/ch18-dialect-conversion/README.md))

**vector dialect** — Fixed-size SIMD values (`vector<8xf32>`) and ops on
them; it lowers to LLVM vector types. Produced by vectorization passes or
written by hand.
([ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md),
[ch06](../part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md),
[ch22](../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md))

**verifier** — Structural validity checks an op runs after parsing and
after each pass; the reason broken IR fails loudly instead of miscompiling.
([ch15](../part3-building-a-dialect/ch15-verifiers/README.md))

**walk** — Recursively visiting every op nested under an op
(`op->walk([](...))`); the simplest way a pass touches IR.
([ch10](../part2-writing-passes/ch10-first-pass-cpp/README.md))

**wheel (MLIR wheel)** — The prebuilt MLIR distribution on the
[eudsl index](https://llvm.github.io/eudsl) this repo pins: headers, libs,
and tools (`mlir-opt`, `mlir-translate`, `FileCheck`, ...) via pip, no LLVM
build. (Repo [README](../README.md); escape hatch:
[source-build appendix](appendix-source-build.md))
