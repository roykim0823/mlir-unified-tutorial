# Chapter 29: Calling External Libraries — matmul to OpenBLAS

> Reworked from ML-compiler-exercise chapter 5 for the unified repo.

Chapter 28's pipeline compiled every op to loops — including the matrix
multiplications, which is exactly the part you should *not* compile
yourself. Decades of engineering live in tuned BLAS libraries; a real ML
compiler's job for a matmul is to **get out of the way and call one**. This
chapter writes that pass: `--convert-matmul-to-blas` rewrites each 2-D f32
`linalg.matmul` into a call to `cblas_sgemm`. It is also this tutorial's
final dialect-conversion workout: a `ConversionPattern` that straddles two
type worlds (memrefs and LLVM structs) at once.

## What you will learn

- The FFI mindset: lowering an op to a *declared external function* plus a
  call, instead of to code
- A `ConversionPattern` with an `LLVMTypeConverter`: why the pattern reads
  the *original* operands for types and dimensions but the *converted*
  operands for data pointers
- Handling static and dynamic shapes in one pattern (`memref.dim` +
  `index_cast` as the dynamic fallback)
- `getOrCreate`-style declaration injection, and mapping a C signature
  (`cblas_sgemm`'s 14 parameters) onto LLVM types
- Pass placement: why this runs after bufferization but before
  linalg-to-loops
- Linking: OpenBLAS on Linux, Accelerate on macOS, and the lit substitution
  that papers over the difference

## Prerequisites

[Chapter 18](../../part4-lowering/ch18-dialect-conversion/README.md)
(dialect conversion, type converters) is assumed cold. The pass lives in
[`lib/Conversion/MatmulToBlas/`](../../lib/Conversion/MatmulToBlas/),
registered tablegen-style like every pass since Chapter 11 (the source repo
uses the older `PassWrapper` idiom — exercise 5 compares them). Tests:
[`test/part6/matmul_to_blas.mlir`](../../test/part6/matmul_to_blas.mlir).

## 1. Getting a BLAS

Any library exposing the CBLAS interface works:

- **Linux:** OpenBLAS, via package manager (`apt install libopenblas-dev`)
  or from source (recipe below).
- **macOS:** nothing to install — Apple's **Accelerate** framework ships
  `cblas_sgemm`; link with `-framework Accelerate`. (Homebrew OpenBLAS
  works too.)

Building OpenBLAS from source (the source repo's recipe; see OpenBLAS's
[install docs](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/install.md#linux)
and [user manual](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/user_manual.md#compiling-openblas)).
`make` auto-detects the CPU architecture; the default install prefix is
`/opt/OpenBLAS`, so pick your own with `PREFIX`:

```sh
curl -LO https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.31/OpenBLAS-0.3.31.tar.gz
tar -xzf OpenBLAS-0.3.31.tar.gz && cd OpenBLAS-0.3.31
make -j$(nproc)                       # macOS: make -j$(sysctl -n hw.ncpu)
make PREFIX=$HOME/opt/openblas install
```

Link with `-L$HOME/opt/openblas/lib -lopenblas`, and make sure the dynamic
linker finds the library at run time, either per shell:

```sh
export LD_LIBRARY_PATH=$HOME/opt/openblas/lib:$LD_LIBRARY_PATH     # macOS: DYLD_LIBRARY_PATH
```

or system-wide on Linux:

```sh
echo "$HOME/opt/openblas/lib" | sudo tee /etc/ld.so.conf.d/openblas.conf
sudo ldconfig
```

(An `-Wl,-rpath,$HOME/opt/openblas/lib` link flag avoids both.)

The repo's lit config picks per-platform
([`lit.cfg.py`](../../lit.cfg.py)): `%blas_link_flags` expands to
`-framework Accelerate` on Darwin and `-lopenblas` elsewhere.

The function we are targeting — single-precision general matrix multiply,
computing **C = alpha·A·B + beta·C**:

```c
void cblas_sgemm(Order, TransA, TransB,      // layout & transpose flags
                 M, N, K,                    // C is MxN, K is the shared dim
                 alpha, A, lda, B, ldb,      // inputs + leading dimensions
                 beta, C, ldc);              // output
```

## 2. The pattern: one foot in each type world

The core of
[`MatmulToBlas.cpp`](../../lib/Conversion/MatmulToBlas/MatmulToBlas.cpp) is
a `ConversionPattern` — the raw, type-changing kind from Chapter 18 —
constructed with an `LLVMTypeConverter`:

```cpp
struct MatmulOpToBlasLibraryCall : public ConversionPattern {
  explicit MatmulOpToBlasLibraryCall(MLIRContext *context,
                                     TypeConverter &converter)
      : ConversionPattern(converter, linalg::MatmulOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
```

`matchAndRewrite` receives the op **twice**, in effect. `op` is the
original `linalg.matmul` whose operands still have memref types — that is
where legality checks and shape questions go:

```cpp
    auto lhsType = dyn_cast<MemRefType>(originalLhs.getType());
    ...
    if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outputType.getRank() != 2 ||
        !lhsType.getElementType().isF32())
      return failure();
```

(returning `failure()` = "not my case, leave it alone" — batch matmuls,
f64, higher ranks all fall through to the loop lowering). `operands`, by
contrast, holds the **converted** values: the type converter has already
turned each memref into the LLVM descriptor struct from Chapter 28, and
that is where the raw data pointers come from:

```cpp
    Value lhsPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, lhs, ArrayRef<int64_t>{1});
```

— field 1 of the struct is the aligned pointer, exactly the
`MemRefDescriptor::aligned` you wrote by hand in C++ last chapter, now read
from the other side of the ABI.

## 3. Dimensions, static or dynamic

`cblas_sgemm` wants M, N, K as i32. For a static shape they are attributes
turned constants; for a `?` the pattern asks the *original* memref at
runtime:

```cpp
    if (lhsShape[0] != ShapedType::kDynamic) {
      M = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[0]));
    } else {
      Value dimM = rewriter.create<memref::DimOp>(loc, originalLhs, 0);
      M = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimM);
    }
```

The same dance yields the leading dimensions (`ldA` = K, `ldB` = `ldC` = N
for row-major, non-transposed operands). Alpha is 1. Beta deserves a
sentence: `linalg.matmul` *accumulates into* its output, so the fully
general translation is beta = 1 — but every pipeline in this repo
materializes the accumulator with `linalg.fill 0` first (you saw it in both
snapshots), so the pass emits **beta = 0** and spares BLAS a read of C.

## 4. Declaring the callee, building the call

You cannot call a function the module has never heard of. The
`getOrCreateSgemmFunc` helper checks `module.lookupSymbol` first (patterns
run once per matmul; only the first may create the declaration), then
builds the 14-parameter LLVM function type and inserts a declaration at
module scope:

```cpp
    SmallVector<Type> argTypes = {i32Type, i32Type, i32Type, i32Type, i32Type,
                                  i32Type, f32Type, ptrType, i32Type, ptrType,
                                  i32Type, f32Type, ptrType, i32Type};
    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), argTypes);

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto sgemmFunc = rewriter.create<LLVM::LLVMFuncOp>(...);
```

A declaration with no body is precisely a promise the *linker* will keep —
that is the whole FFI trick. Then the rewrite finishes anticlimactically:

```cpp
    SmallVector<Value> args = {order, transA, transB,    M,   N,
                               K,     alpha,  lhsPtr,    ldA, rhsPtr,
                               ldB,   beta,   outputPtr, ldC};
    rewriter.create<LLVM::CallOp>(loc, sgemmFunc, args);
    rewriter.eraseOp(matmulOp);
```

Note there is no `replaceOp`: the matmul writes through a buffer and
produces no SSA results in memref-world, so erasing it after planting the
call is the entire rewrite.

## 5. The pass shell and its placement

```cpp
    ConversionTarget target(context);
    target.addLegalDialect<func::FuncDialect, LLVM::LLVMDialect,
                           memref::MemRefDialect, arith::ArithDialect>();
    target.addIllegalOp<linalg::MatmulOp>();

    RewritePatternSet patterns(&context);
    LLVMTypeConverter typeConverter(&context);
    patterns.add<MatmulOpToBlasLibraryCall>(patterns.getContext(),
                                            typeConverter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
```

Chapter 18's machinery, one illegal op. `applyPartialConversion` tolerates
the other linalg ops still in the module — they are none of this pass's
business. Placement in the Chapter 28 pipeline is load-bearing and the
comment in `tutorial-opt.cpp` says why: **after bufferization** (the
pattern needs memrefs — on tensors there is no data pointer to extract) and
**before `convert-linalg-to-loops`** (which would dissolve the matmul into
loops and leave nothing to match).

## 6. Watch it fire

On a dynamic-shaped matmul
([`test/part6/matmul_to_blas.mlir`](../../test/part6/matmul_to_blas.mlir)):

```sh
build/tools/tutorial-opt --linalg-to-bufferization --convert-matmul-to-blas \
    test/part6/matmul_to_blas.mlir
```

```mlir
llvm.func @cblas_sgemm(i32, i32, i32, i32, i32, i32, f32, !llvm.ptr, i32,
                       !llvm.ptr, i32, f32, !llvm.ptr, i32)
func.func @dynamic(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, ...) {
  %0 = builtin.unrealized_conversion_cast %arg2 : memref<?x?xf32, ...>
       to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  ...
  %dim = memref.dim %arg0, %c0 : memref<?x?xf32, ...>
  %6 = arith.index_cast %dim : index to i32
  ...
  llvm.call @cblas_sgemm(%3, %4, %5, %6, %7, %8, %11, %13, %8, %14, %9, %12, %15, %10)
```

The `unrealized_conversion_cast`s are Chapter 18's familiar seams — the
function signature is still memref-world, the call is LLVM-world, and
`--finalize-memref-to-llvm` + `--reconcile-unrealized-casts` later in the
pipeline dissolve them.

The proof it all works is Chapter 28's numbers: the sample binary (1 sgemm
call) and mnist binary (3 — one per Linear layer) both match PyTorch
exactly, with the actual multiplication done by Accelerate/OpenBLAS. And at
scale: lowering the full GPT-2 module plants **49** `cblas_sgemm` calls,
one per projection/head matmul.

## Where to go next

CPU: done for the small models — model, pipeline, library, binary.
[Chapter 30](../ch30-real-models/README.md) runs the same pipeline on real
models (a CNN, resnet18, bert, flan-t5) and benchmarks it, and
[Chapter 31](../ch31-gpu-nvvm/README.md) points the same linalg IR at an
NVIDIA GPU through the gpu and nvvm dialects.

## Exercises

1. The pattern silently skips `linalg.batch_matmul` — GPT-2's 24 attention
   matmuls stay as loops. Extend the pass with a second pattern that
   lowers a static-shaped `batch_matmul` to a loop over `cblas_sgemm`
   calls (or to one call per batch element at rewrite time).
2. f64 support: add a twin pattern targeting `cblas_dgemm`. How much can
   you share? (Consider a templated helper or a constructor parameter.)
3. Wrong-beta experiment: change beta to 1.0, rebuild, and run the mnist
   e2e test. It still passes — why? Now construct an input IR where
   beta = 1 vs beta = 0 *does* matter (hint: an accumulator that isn't
   zero-filled) and check which one matches `linalg.matmul` semantics.
4. Benchmark it: time the mnist binary against a build where you comment
   the pass out of the pipeline (pure loops). On top of that, compare
   against PyTorch's own timing (the `benchmark*` scripts in
   `src/*/`; see [Chapter 30](../ch30-real-models/README.md)).
5. Style archaeology: the source repo implements this pass as a
   `PassWrapper` with hand-written `getArgument()`/`getDescription()`.
   Diff that against this repo's tablegen registration (Chapter 11) and
   list what tablegen generates for free.
