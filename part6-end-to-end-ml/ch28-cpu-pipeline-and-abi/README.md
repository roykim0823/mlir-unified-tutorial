# Chapter 28: The CPU Pipeline and the C ABI

> Reworked from ML-compiler-exercise chapter 4 for the unified repo.

Chapter 27 ended with `sample_model_linalg.mlir` — a real PyTorch model as
linalg-on-tensors IR. This chapter turns it into machine code and calls it
from a C++ program, and the compiled binary prints *exactly* the numbers
PyTorch printed. Two things are genuinely new relative to Chapter 19's
poly-to-binary flow: the lowering is packaged as two reusable **pass
pipelines** inside `tutorial-opt`, and the calling convention is the
**MemRefDescriptor ABI** with `_mlir_ciface_` wrappers — the actual
mechanism by which the outside world hands tensors to MLIR-compiled code.

## What you will learn

- Registering multi-pass pipelines (`PassPipelineRegistration`) so a whole
  lowering is one flag
- The two-stage descent: tensors → memrefs (one-shot bufferization +
  deallocation), then memrefs → LLVM dialect
- What a lowered function's ABI looks like: the memref "exploded" argument
  list, and the `_mlir_ciface_` struct-passing wrapper
- Writing the C++ side: `MemRefDescriptor<T, N>`, offsets, sizes, strides —
  and who allocates the output buffer (not who you'd think)
- Real-world toolchain skew: LLVM 24 float literals vs older clang, and the
  shim that bridges them
- Scale check: the same two flags chew through a 950 MB GPT-2 module in
  seconds

## Prerequisites

- [Chapter 19](../../part4-lowering/ch19-to-llvm-native-binary/README.md)
  (mlir-translate, `clang -x ir`, the native-binary flow) and
  [Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)
  (one-shot bufferization).
- A built `build/tools/tutorial-opt` and the wheel venv on PATH:

  ```sh
  source .venv/bin/activate
  export MLIR_ROOT=$(python -m mlir_wheel --root-dir)
  export PATH=$MLIR_ROOT/bin:$PATH
  ```

All commands run from the repo root. Verified end-to-end by
[`test/part6/sample_e2e.mlir`](../../test/part6/sample_e2e.mlir) and
[`test/part6/mnist_e2e.mlir`](../../test/part6/mnist_e2e.mlir).

## 1. Two pipelines instead of twenty flags

Chapter 22 lowered linalg by chaining a dozen `--convert-*` flags by hand.
For a workflow you run on every model, that belongs in the driver
([`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp)):

```cpp
void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);
}
```

registered under a name, exactly like `--poly-to-llvm` was:

```cpp
mlir::PassPipelineRegistration<>(
    "linalg-to-bufferization",
    "Run passes to lower the linalg dialect to bufferized memrefs",
    linalgToBufferizationPipelineBuilder);
```

Stage one is deliberately *just* bufferization, so you can stop and read
the buffer-world program:

```sh
build/tools/tutorial-opt --linalg-to-bufferization \
    part6-end-to-end-ml/src/sample/sample_model_linalg.mlir
```

```mlir
memref.global "private" constant @__constant_3x4xf32 : memref<3x4xf32> =
    dense_resource<torch_tensor_3_4_torch.float32> {alignment = 64 : i64}
...
func.func @sample_model(%arg0: memref<3x4xf32, strided<[?, ?], offset: ?>>)
    -> memref<3x5xf32> {
  ...
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x4xf32>
  linalg.generic {...} ins(%arg0, %2 : ...) outs(%alloc : memref<3x4xf32>) { ... }
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<4x5xf32>
  linalg.transpose ins(%1 : memref<5x4xf32>) outs(%alloc_1 : memref<4x5xf32>) ...
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<3x5xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc_2 : memref<3x5xf32>)
  linalg.matmul ins(%alloc, %alloc_1 : ...) outs(%alloc_2 : memref<3x5xf32>)
  ...
```

Three things to notice. The weight constants became `memref.global`s. The
function argument grew a `strided<[?, ?], offset: ?>` layout —
`bufferizeFunctionBoundaries` assumes nothing about how callers lay out
their buffers. And the function *returns* a fresh `memref<3x5xf32>` it
allocated itself; hold that thought for section 3.

Stage two is the descent to the LLVM dialect — Chapter 19's ladder plus a
few rungs a real model needs (`math.erf` → libm for bert, affine fusion),
with one slot reserved for Chapter 29:

```cpp
void bufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Replace matmuls with BLAS calls AFTER bufferization (the pass needs
  // memrefs) but BEFORE the loop lowering (which would dissolve the matmul).
  manager.addPass(mlir::tutorial::createMatmulToBlas());

  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createConvertMathToLibmPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}
```

(One drift from the source repo: its pipeline also ran
`affine-super-vectorize`. Without vector-size options that pass is a no-op,
so this port drops it — Chapter 22 showed the modern way to vectorize.)

## 2. The ABI: what a memref becomes

Run stage two with one extra flag:

```sh
build/tools/tutorial-opt --linalg-to-bufferization \
    part6-end-to-end-ml/src/sample/sample_model_linalg.mlir \
| build/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm
```

and look at the function signatures in the result:

```mlir
llvm.func @sample_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64,
                        %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64)
    -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> ...

llvm.func @_mlir_ciface_sample_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr) ...
```

The first is the native calling convention: the rank-2 memref argument
**exploded** into seven scalars — allocated pointer, aligned pointer,
offset, two sizes, two strides — and the memref result returned as the
matching struct. Calling that from C means passing seven arguments per
tensor; mnist's rank-4 input would take eleven. That is what
`--llvm-request-c-wrappers` fixes: it asks for a second function, prefixed
`_mlir_ciface_`, where every memref travels as a *pointer to the struct* —
and a returned memref becomes a leading output-pointer parameter. Two
pointers total. That is the function we call.

On the C++ side the struct is spelled out in
[`src/sample/sample_call.cpp`](../src/sample/sample_call.cpp):

```cpp
template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_sample_model(MemRefDescriptor<float, 2> *output,
                               MemRefDescriptor<float, 2> *input);
}
```

The caller fills a descriptor for the 3×4 input — both pointers to the same
stack array, offset 0, sizes {3, 4}, strides {4, 1} for row-major — and
passes a second descriptor for the output:

```cpp
MemRefDescriptor<float, 2> inputMemRef = {
    (float *)inputData, (float *)inputData, /*offset=*/0,
    /*sizes=*/{3, 4}, /*strides=*/{4, 1}};

_mlir_ciface_sample_model(&outputMemRef, &inputMemRef);

float *output = (float *)outputMemRef.aligned;
```

## 3. Who allocates the output

Here is the subtlety the bufferized IR foreshadowed: `sample_model`
*returns* a buffer it `memref.alloc`ed itself. The output descriptor you
pass is not a buffer the model writes into — it is an **out-slot that the
wrapper overwrites** with the descriptor of the model-allocated result.
Whatever pointer you put in it is simply replaced. That is why the read
goes through `outputMemRef.aligned` *after* the call (now pointing at the
model's buffer), and why the print loop uses `outputMemRef.sizes` and
`.strides` rather than its own constants. Corollary: the result was
malloc'd by generated code and is leaked by this toy caller — a real
integration would either free it (the `allocated` pointer is there for
exactly that) or use bufferization options to make results write into
caller-provided buffers.

## 4. To machine code — with one toolchain speed bump

Chapter 19's recipe: `mlir-translate` out of MLIR, `clang -x ir` into an
object file (the wheel ships no `llc`). One new obstacle, kept here
because it is the most honest lesson in the chapter: the wheel's LLVM 24
prints float constants in a syntax older clangs cannot read —

```
@__constant_3x4xf32 = ... [float f0x3EA3B35C, float -4.291610e-01, ...]
```

`f0x...` typed hex literals are new; and even the decimal form is a
"shortest round-trip" spelling that old parsers reject as inexact for
`float`. If your system clang is LLVM ≥ 24 none of this bites; this
machine's clang 20 needed the twenty-line filter
[`tools/ll_fp_compat.py`](../../tools/ll_fp_compat.py), which rewrites both
forms into the legacy double-widened hex encoding (`float
0x3FD4766B80000000`) that every LLVM since forever accepts. Toolchain
version skew between IR producer and consumer is a fact of life in ML
compiler work; now you have seen one concrete instance and one concrete
fix.

The full flow, exactly as the lit test runs it:

```sh
cd part6-end-to-end-ml/src/sample
../../../build/tools/tutorial-opt --linalg-to-bufferization sample_model_linalg.mlir \
| ../../../build/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm \
| mlir-translate --mlir-to-llvmir \
| python3 ../../../tools/ll_fp_compat.py \
| clang -x ir - -c -o sample_model.o

c++ -c sample_call.cpp -o sample_call.o
c++ sample_call.o sample_model.o -o sample -framework Accelerate  # macOS
# on Linux: c++ ... -o sample -lopenblas
./sample
```

(The BLAS link flag is Chapter 29's doing — the pipeline already rewrote
the matmul into a `cblas_sgemm` call.) Output:

```
0.3844 0.0000 0.2425 0.2203 0.0000
0.9251 0.0000 0.6383 0.4868 0.0000
0.9336 0.0000 0.3170 0.4453 0.0000
```

Compare with Chapter 27's PyTorch reference: identical, digit for digit. A
PyTorch model, compiled by your own pass driver, running as native code.

## 5. MNIST, and the runtime library

The same commands with `mnist` substituted run the real MLP — with one new
link error worth understanding:

```
Undefined symbols for architecture arm64:
  "_memrefCopy", referenced from: _mnist_model in mnist_model.o
```

MNIST's `flatten` became `tensor.collapse_shape`; on a
function-boundary memref with dynamic strides, its lowering cannot always
be a no-op view, so `--finalize-memref-to-llvm` emits a call to
`memrefCopy` — a helper in MLIR's small **runner-utils runtime library**,
which the wheel ships:

```sh
c++ mnist_call.o mnist_model.o -o mnist -framework Accelerate \
    -L$MLIR_ROOT/lib -lmlir_c_runner_utils -Wl,-rpath,$MLIR_ROOT/lib
./mnist
```

```
0.0149 -0.0860 0.1660 -0.1409 -0.1119 -0.0749 -0.1015 -0.1354 -0.0231 0.0546
```

Again identical to PyTorch's logits. So: some lowerings are pure code, some
lean on a tiny runtime — know which libraries your pipeline implies.

## 6. Scale check: GPT-2

Do the two pipeline flags survive contact with a 124M-parameter
transformer? Regenerate the full `gpt2_linalg.mlir` (950 MB — see the
[part README](../README.md)), then:

```sh
build/tools/tutorial-opt --linalg-to-bufferization gpt2_linalg.mlir -o gpt2_buf.mlir
build/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm \
    gpt2_buf.mlir -o gpt2_llvm.mlir
```

Measured on this port's machine: about **4 seconds per stage, ~1.4 GB peak
memory** — almost all of it the weight blobs streaming through — and the
result contains **49 `cblas_sgemm` calls** (12 layers × 4 projection
matmuls, plus the output head). The attention `batch_matmul`s stay as
loops; making the BLAS pass catch those is a Chapter 29 exercise. Actually
*executing* GPT-2 from C++ (feeding token ids, decoding logits in a
generation loop) follows the source repo's flan-t5/pybind11 harness and is
left as the capstone exercise below.

To go one step further, to a linkable object file, run
[`src/gpt2/run_mlir_pipeline.sh`](../src/gpt2/run_mlir_pipeline.sh) next to
the regenerated export. Section 4's route doesn't survive this scale: the
`.ll` that `mlir-translate` prints is 2.4 GB (3.2 GB after
`ll_fp_compat.py`, which spells every float constant in hex), and clang
gives up on any single source file past about 2 GB:

```
fatal error: translation unit is too large for Clang to process: ran out of source locations
```

So the script compiles the LLVM-dialect module in process instead, with
[`tools/mlir_to_object.py`](../../tools/mlir_to_object.py). The MLIR Python
bindings' `ExecutionEngine` translates and codegens the module with the
wheel's own LLVM, so no textual IR and no version skew are involved, and
`dump_to_object_file` writes the result. `ExecutionEngine` links before it
dumps, so it needs the libraries that provide external symbols
(`libmlir_c_runner_utils` for `memrefCopy`). Measured here: about 32 seconds
and 3.5 GB peak memory end to end, giving a 498 MB `gpt2.o` that exports
`_mlir_ciface_gpt2_model` and leaves `cblas_sgemm`, `memrefCopy`, `expf` and
`tanhf` for the final link.

## 7. Several results, and one op the pipeline does not lower

Two more ABI facts come up as soon as the models get real (Chapter 30's
bert and flan-t5). Both were checked on a toy function with two results,
`(tensor<2x3xf32>, tensor<3xf32>)`, sent through the same two stages and
called from C++.

**Several results → one struct.** The lowered native function returns an
`!llvm.struct` *of* memref structs, one per result. The `_mlir_ciface_`
wrapper turns that into a single leading out-pointer to the same nested
struct:

```mlir
llvm.func @_mlir_ciface_two(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
//                          ^ results struct  ^ input 0       ^ input 1
```

So on the C++ side you declare a struct holding one descriptor per result,
in order, and pass its address first:

```cpp
struct Output {
  MemRefDescriptor<float, 2> a;   // result #0
  MemRefDescriptor<float, 1> c;   // result #1
};
extern "C" void _mlir_ciface_two(Output *, MemRefDescriptor<float, 2> *,
                                 MemRefDescriptor<int64_t, 1> *);
```

Both descriptors get overwritten with model-allocated buffers, exactly as
in section 3. bert's [`bert_call.cpp`](../src/bert/bert_call.cpp) is this pattern with a rank-3 and a
rank-2 result.

**`bufferization.clone`.** If a function returns a buffer it does not
own, such as a constant (a `memref.global`) or one of its own
arguments, the deallocation pipeline in stage one inserts a
`bufferization.clone` so the caller receives something it may free. Stage
two has no pass that lowers `clone`, so the op reaches `mlir-translate`
unconverted:

```
error: Dialect `bufferization' not found for custom op 'bufferization.clone'
```

and the object file never gets the function. Upstream's
`--convert-bufferization-to-memref` (`clone` → `memref.alloc` +
`memref.copy`, the latter from the runner-utils library of section 5)
closes the gap, run between the two stages:

```sh
tutorial-opt --linalg-to-bufferization model.mlir \
| mlir-opt --convert-bufferization-to-memref \
| tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm | ...
```

On sample and mnist it is a no-op (same binaries, same numbers; checked),
so Chapter 30's scripts always include it. sample and mnist do not need
it, since they return freshly computed tensors. The bigger models might:
any result that bufferizes to a view or a global would.

## Where to go next

The matmul-to-BLAS pass this pipeline quietly used is
[Chapter 29](../ch29-external-libs-blas/README.md)'s subject — the full
conversion-pattern walkthrough. Then
[Chapter 31](../ch31-gpu-nvvm/README.md) retargets the same model at a GPU.

## Exercises

1. Delete `--llvm-request-c-wrappers` from the flow and look at the
   generated `.ll`: find `sample_model`'s exploded 7-argument signature and
   adapt `sample_call.cpp` to call it directly. (You'll pass the struct
   *fields* as arguments, and receive a struct by value.)
2. Fix the leak: extend `sample_call.cpp` to `free()` the model-allocated
   result. Which descriptor field is the right one to free, and why must
   it be `allocated` rather than `aligned`?
3. The reproducibility bug from Chapter 27, live: remove the
   `torch.manual_seed(41)` inside `lower_sample_model.py`'s `export()` (the
   one right before `Sample()`), regenerate, and re-run the lit test.
   Explain the failure from the RNG-stream mechanics.
4. Feed a *wrong* stride: change the input descriptor's strides to {1, 4}
   (column-major) without changing the data. Predict the three output rows
   before running.
5. Capstone: write `gpt2_call.cpp`. Descriptors are rank-2 i64 for
   `input_ids`/`attention_mask` and rank-3 f32 out; implement the greedy
   loop from `run_gpt2_model.py` in C++ and compare your first generated
   token against PyTorch's.
