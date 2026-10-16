# Chapter 19: Lowering to LLVM and a Native Binary

> Reworked from the mlir-tutorial fork's chapter 11, with the tiny dialect
> example from mlir-tutor's chapter 1. All commands and IR listings have been
> re-run against this repo's prebuilt binaries and the MLIR wheel (LLVM 24 at
> the time of writing), so what you see here is what you will get — including
> one deliberate deviation from the original: this repo compiles LLVM IR with
> `clang -x ir` instead of `llc`, because the wheel ships no `llc`.

This is Part 4's payoff chapter: a program written in our own `poly`
dialect becomes a **native executable**, called from ordinary C code, and
prints the right answer. Chapter 18 took one step down the staircase
(`poly` → `tensor`/`scf`/`arith`); this chapter takes all the rest — and
explains the last unexplained piece of `tutorial-opt.cpp`, the
`--poly-to-llvm` mega-pipeline. A second, shorter thread then revisits the
same destination from the `tiny` dialect's side: what dialect conversion
looks like when a *custom type* (`!tiny.ptr`) survives all the way to the
LLVM boundary and must be converted there.

**What you will learn:**

- How to register a named pass *pipeline* (`PassPipelineRegistration`),
  and a practical methodology for composing one.
- Two genuinely new lowering concepts: **linalg** as the gateway for
  tensor arithmetic, and **bufferization** — the tensor-to-memref
  world-switch from values to memory.
- The complete descent to the `llvm` exit dialect, pass by pass, with
  real intermediate IR at each interesting stage.
- How MLIR hands off to LLVM proper: `mlir-translate`, then `clang -x ir`
  to native code, linking against a C `main` — and what the memref
  calling convention means for your exported functions.
- A pipeline *gap* you can hit today (unlowered `bufferization.clone`),
  reproduced and explained.
- Type conversion at the LLVM boundary: the `tiny` dialect's
  `TypeConverter` turning `!tiny.ptr` into `!llvm.ptr`, including inside
  function signatures — and a Python DSL that drives the whole descent.

**Prerequisites:**
[Chapter 18](../ch18-dialect-conversion/README.md) (dialect conversion —
this chapter's Stage 1 is its `--poly-to-standard` pass, and the tiny
thread assumes you know what a `ConversionPattern` and adaptor are), plus
Chapter 2's ctlz-to-LLVM staircase and matmul+relu bufferization demo —
reread [Chapter 2](../../part1-using-mlir/ch02-lowering-pipelines/README.md)
Part B if it's hazy; everything there returns here at full scale.
Tools: besides `build/tools/tutorial-opt`, this chapter uses
`mlir-translate` (from the wheel — on `PATH` once the repo's virtualenv is
activated) and your system `clang`. Commands are run from the repository
root.

---

## 1. Concepts: pipelines and the exit dialect

Chapter 2 lowered `math.ctlz` to executable form by chaining a handful of
`--convert-*` flags (and mentioned the `-pass-pipeline`
[textual DSL](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification)
for the same thing). For `poly` the journey is ~18 passes, and typing
that string is not a lifestyle. The fix is the last feature of
`tutorial-opt.cpp` (Chapter 10 showed everything else):

***tools/tutorial-opt.cpp*** (excerpt)
```cpp
mlir::PassPipelineRegistration<>(
    "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
    polyToLLVMPipelineBuilder);
```

which turns a C++ function that appends passes to an `OpPassManager` into
a single `--poly-to-llvm` flag
([pass pipeline registration](https://mlir.llvm.org/docs/PassManagement/#pass-pipeline-registration)
in the docs). The pipeline's destination is the
[`llvm` dialect](https://mlir.llvm.org/docs/Dialects/LLVM/)
— the **exit dialect**: a faithful MLIR mirror of LLVM IR, from
which a mechanical translation leaves MLIR entirely. (Other exits exist —
SPIR-V for GPU, EmitC for C source — but LLVM is the classic one.)

How do you *compose* such a pipeline? The honest method by which the
original author built it — worth learning *as method* — is: start from
an empty pipeline;
look at the highest-level op remaining in your IR; add the pass that
lowers it (the upstream [passes list](https://mlir.llvm.org/docs/Passes/)
has too many lowerings to scan linearly — search it for the dialect's
name); when that pass complains, figure out what it needed *first*;
repeat until only the exit dialect remains. The pipeline below is the
fossil record of exactly that process — including the fixes from its
war stories (section 8).

## 2. The pipeline, stage by stage

A pipeline builder is exactly the kind of function section 1's
registration wraps: it receives an `OpPassManager` and appends passes
with `addPass`, in the order they will run — so reading it top to bottom
is reading the compilation strategy. Here is `polyToLLVMPipelineBuilder`
from [`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp), verbatim:

***tools/tutorial-opt.cpp*** (excerpt)
```cpp
void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize, from
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Needed to lower memref.subview
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}
```

Read it as five logical stages:

1. **Leave poly** — Chapter 18's `--poly-to-standard`, plus a
   canonicalize to sweep up.
2. **Tensor arithmetic → linalg** — the two `*-to-linalg` passes.
3. **Bufferize** — tensors become memrefs, then compiler-inserted
   deallocations.
4. **Linalg → loops**, plus one memref cleanup pass.
5. **The LLVM descent** — Chapter 2's staircase (`scf` → `cf` → `llvm`),
   extended for memory, ending in `reconcile-unrealized-casts`.

The final cleanup quartet you know from Part 3 (canonicalize and sccp
lean on Chapter 14's folders; cse is generic dedup; symbol-dce deletes
now-unused private functions). Stages 2–5 need explaining, and the best
way is to watch IR move through them. Running example (degree 4 to stay
readable) — save it as `/tmp/add_ct.mlir`:

```mlir
func.func @add_ct(%arg : !poly.poly<4>) -> !poly.poly<4> {
  %0 = poly.constant dense<[1, 2, 3, 4]> : tensor<4xi32> : !poly.poly<4>
  %1 = poly.add %0, %arg : !poly.poly<4>
  return %1 : !poly.poly<4>
}
```

Because every pass in the pipeline is also registered as an individual
flag on `tutorial-opt`, you can replay the pipeline *prefix by prefix* —
that's how each listing below was produced. (The lazier alternative:
append `--mlir-print-ir-after-all` to the full pipeline run and get all
25 IR snapshots on stderr at once; exercise 2 does exactly that.)

### Stage 2: why linalg?

After stage 1 (run `build/tools/tutorial-opt /tmp/add_ct.mlir
--poly-to-standard --canonicalize` to see it) our add is Chapter 18's
output: an `arith.addi` *on tensors*, `%0 = arith.addi %arg0, %cst :
tensor<4xi32>`. And here's the snag:
[`convert-arith-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-llvm)
only handles scalar arith. LLVM has no
tensors; someone must decide what a tensor `addi` *means* in terms of
loops and memory. That someone is
**[linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)**, MLIR's
structured linear-algebra dialect.
[`convert-elementwise-to-linalg`](https://mlir.llvm.org/docs/Passes/#-convert-elementwise-to-linalg)
(legal because `arith.addi` implements
`ElementwiseMappable` — Chapter 13's traits paying rent again) rewrites
the tensor addi. Append `--convert-elementwise-to-linalg` to the command
and the output becomes (real output):

```mlir
#map = affine_map<(d0) -> (d0)>
func.func @add_ct(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
       ins(%arg0, %cst : tensor<4xi32>, tensor<4xi32>)
       outs(%arg0 : tensor<4xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.addi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
```

Read `linalg.generic` as "a loop nest as data": the `indexing_maps` say
how each input/output is indexed (here identity — elementwise), the
`iterator_types` say the loop is `parallel` (no cross-iteration
dependence), and the region holds the *scalar* body. High-level enough to
optimize (fuse, tile, vectorize — Part 5's subject), mechanical enough to
lower. `convert-tensor-to-linalg` does the same for `tensor.pad`
(Chapter 18's from_tensor lowering), which becomes a `linalg.fill` plus
insertion.

### Stage 3: bufferization — values become memory

Everything so far — SSA values, `tensor`s, Chapter 18's "functional
tensors" — lives in a world of immutable *values*. Real machines
have mutable *memory*. **Bufferization** is the world-switch: every
`tensor<4xi32>` (a value) becomes a `memref<4xi32>` (a typed view of a
buffer), and ops that *produced* new tensors become ops that *write
into* buffers. Extend the prefix through one-shot-bufferize:

```sh
build/tools/tutorial-opt /tmp/add_ct.mlir --poly-to-standard --canonicalize \
  --convert-elementwise-to-linalg --convert-tensor-to-linalg \
  --one-shot-bufferize=bufferize-function-boundaries
```

```mlir
memref.global "private" constant @__constant_4xi32 : memref<4xi32>
    = dense<[1, 2, 3, 4]> {alignment = 64 : i64}
func.func @add_ct(%arg0: memref<4xi32, strided<[?], offset: ?>>)
    -> memref<4xi32, strided<[?], offset: ?>> {
  %0 = memref.get_global @__constant_4xi32 : memref<4xi32>
  linalg.generic ... ins(%arg0, %0 : ...) outs(%arg0 : ...) { ... }
  return %arg0 : ...
}
```

Worth a slow look:

- The constant became a **global** in read-only memory, fetched with
  `get_global` — exactly what a C compiler would do with a static array.
- `bufferizeFunctionBoundaries = true` (the
  `bufferize-function-boundaries` flag above) converted the *signature*
  too (tensor arguments → memref arguments) — the same "types hide in
  signatures" issue as Chapter 18, handled by an option here. The
  `strided<[?], offset: ?>` layout means "accept any stride/offset" — a
  caller-friendly ABI for function arguments.
- The `linalg.generic` now writes its result **into `%arg0`** — the
  "one-shot" analysis proved it could reuse the input buffer instead of
  allocating. Bufferization is an optimization problem (minimize copies
  and allocations), and one-shot-bufferize solves it whole-module in one
  analysis — its predecessor was a fleet of per-dialect passes
  coordinated through a dedicated
  [`bufferization` dialect](https://mlir.llvm.org/docs/Dialects/BufferizationOps/)
  of intermediate ops, which is why the name celebrates being *one* pass.
- Where buffers *are* allocated, someone must free them:
  `buildBufferDeallocationPipeline` appends the official
  [ownership-based buffer deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/)
  passes that insert `dealloc`s — compiler-generated
  memory management. (The helper is the official bundle of what would
  otherwise be five hand-listed passes; on the command line it's the
  single flag `--buffer-deallocation-pipeline`.)

### Stages 4–5: down the familiar staircase

With buffer semantics,
[`convert-linalg-to-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-loops)
turns each `linalg.generic` into `scf.for` loops over memrefs (it
*requires* bufferized input — see section 8's war story; sibling passes
emit
[affine loops](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-affine-loops)
or
[parallel loops](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-parallel-loops)
instead). Appending it to the stage-3 command yields (real output):

```mlir
func.func @add_ct(%arg0: memref<4xi32, strided<[?], offset: ?>>)
    -> memref<4xi32, strided<[?], offset: ?>> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_4xi32 : memref<4xi32>
  scf.for %arg1 = %c0 to %c4 step %c1 {
    %1 = memref.load %arg0[%arg1] : memref<4xi32, strided<[?], offset: ?>>
    %2 = memref.load %0[%arg1] : memref<4xi32>
    %3 = arith.addi %1, %2 : i32
    memref.store %3, %arg0[%arg1] : memref<4xi32, strided<[?], offset: ?>>
  }
  return %arg0 : memref<4xi32, strided<[?], offset: ?>>
}
```

From here it's Chapter 2's staircase verbatim, extended for memory:
`scf` → `cf` → `llvm`
([`convert-scf-to-cf`](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-cf),
[`convert-cf-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-cf-to-llvm));
scalar arith → `llvm`; `func` → `llvm.func`
([`convert-func-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-func-to-llvm));
[`finalize-memref-to-llvm`](https://mlir.llvm.org/docs/Passes/#-finalize-memref-to-llvm)
lowers memref ops
to pointer arithmetic (and the `index` type to `i64`); and
[`reconcile-unrealized-casts`](https://mlir.llvm.org/docs/Passes/#-reconcile-unrealized-casts)
sweeps the temporary casts — Chapter 18
taught you exactly what those are.
[`expand-strided-metadata`](https://mlir.llvm.org/docs/Passes/#-expand-strided-metadata),
the one
oddball, pre-chews fancy memref ops (like the subviews `tensor.pad`
lowering produces) into primitive ones the finalize pass can handle.

## 3. Step: run the pipeline

```sh
build/tools/tutorial-opt /tmp/add_ct.mlir --poly-to-llvm
```

The output is pure `llvm` dialect (almost — see below). The top of our
small example (real output):

```mlir
llvm.mlir.global private constant @__constant_4xi32(dense<[1, 2, 3, 4]>
    : tensor<4xi32>) {addr_space = 0 : i32, alignment = 64 : i64}
    : !llvm.array<4 x i32>
llvm.func @add_ct(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64,
                  %arg3: i64, %arg4: i64)
    -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
```

Look at that signature: one `memref` argument exploded into **five**
LLVM arguments (allocated pointer, aligned pointer, offset, size,
stride), and the memref return became a five-field struct. This is the
*memref calling convention*, and it's why the repo's runnable tests
export functions with **scalar** signatures (`(i32) -> i32`) — plain C
can call those without knowing MLIR's struct layout.

One more sharp edge, reproducible today: functions that *return* a
polynomial (like `add_ct`) make the deallocation pipeline insert a
`bufferization.clone` op — and nothing later in the pipeline lowers it.
Scroll to the bottom of the output above and you'll find, sitting among
the `llvm.*` ops:

```mlir
%22 = bufferization.clone %10 : memref<4xi32, strided<[?], offset: ?>>
    to memref<4xi32, strided<[?], offset: ?>>
```

so the next step's `mlir-translate` fails on this example (verified):

```
error: Dialect `bufferization' not found for custom op 'bufferization.clone'
```

The pipeline is honest teaching code: scalar-out functions work end to
end; buffer-returning ABIs are left as real-world homework (exercise 4
fixes it with one upstream pass).

## 4. Step: out of MLIR, into an executable

The test program,
[`test/part4/poly_to_llvm.mlir`](../../test/part4/poly_to_llvm.mlir) —
note it exercises *every* `poly` op, and returns a scalar:

***test/part4/poly_to_llvm.mlir*** (excerpt)
```mlir
func.func @test_poly_fn(%arg : i32) -> i32 {
  %tens = tensor.splat %arg : tensor<10xi32>
  %input = poly.from_tensor %tens : tensor<10xi32> -> !poly.poly<10>
  %0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  %1 = poly.add %0, %input : !poly.poly<10>
  %2 = poly.mul %1, %1 : !poly.poly<10>
  %3 = poly.sub %2, %input : !poly.poly<10>
  %4 = poly.eval %3, %arg: (!poly.poly<10>, i32) -> i32
  return %4 : i32
}
```

and a perfectly ordinary C caller,
[`test/part4/poly_to_llvm_main.c`](../../test/part4/poly_to_llvm_main.c):

***test/part4/poly_to_llvm_main.c***
```c
#include <stdio.h>

// This is the function we want to call from LLVM
int test_poly_fn(int x);

int main(int argc, char *argv[]) {
  int i = 1;
  int result = test_poly_fn(i);
  printf("Result: %d\n", result);

  return 0;
}
```

Four commands connect them (each verified, run from the repo root with
the virtualenv active so `mlir-translate` is on `PATH`):

```sh
# 1. MLIR -> LLVM IR (textual): leave MLIR-land
build/tools/tutorial-opt test/part4/poly_to_llvm.mlir --poly-to-llvm \
  | mlir-translate --mlir-to-llvmir > /tmp/poly_fn.ll

# 2. LLVM IR -> native object file
clang -x ir /tmp/poly_fn.ll -c -o /tmp/poly_fn.o

# 3. Compile the C caller; link the two
clang -c test/part4/poly_to_llvm_main.c -o /tmp/main.o
clang /tmp/main.o /tmp/poly_fn.o -o /tmp/a.out

# 4. Run it
/tmp/a.out
```

```
Result: 351
```

`mlir-translate` is the border crossing: not a pass but a *translation*
(`llvm` dialect → real LLVM IR text — the official docs on
[generating LLVM IR](https://mlir.llvm.org/docs/TargetLLVMIR/) cover this
handoff); past it, MLIR is out of the picture and the classic toolchain
takes over. Before running step 4, check the number: at x=1, the splat
polynomial p has ten 1-coefficients so p(1) = 10; c = 2+3x+4x² gives
c(1) = 9; the program computes (c+p)² − p, so (9+10)² − 10 = **351**.
The machine agrees with the algebra.

### Where's llc?

If you've seen this journey elsewhere (including the original
mlir-tutorial), step 2 was `llc --relocation-model=pic -filetype=obj`:
`llc` is LLVM's standalone IR-to-machine-code compiler. This repo's MLIR
wheel deliberately ships no `llc` (check `ls $(python -m mlir_wheel
--root-dir)/bin` — translation tools, no backend tools), and you don't
need it: `clang -x ir` accepts textual LLVM IR and runs *the same LLVM
backend* that `llc` wraps, plus its driver already knows your platform's
conventions — the original's hand-passed `--relocation-model=pic`
(current linkers, notably macOS/arm64, demand position-independent
objects) is clang's default on those platforms, so the flag disappears.
One cosmetic difference: clang warns `overriding the module target
triple`, because `mlir-translate` emits IR without a target triple and
clang stamps in the host's — also your cue that cross-compiling would
need `--target=...` here. If you do want `llc` (or `mlir-runner`) next to
the wheel, [Other Ways to Install MLIR](../../docs/appendix-installing-mlir.md)
covers getting them and the version skew that comes with mixing LLVMs.

### Reading the compiler's work

It's worth admiring what step 1's textual LLVM IR looks like for the
eval loop, from the smaller
[`test/part4/poly_to_llvm_eval.mlir`](../../test/part4/poly_to_llvm_eval.mlir)
(real output):

```llvm
@__constant_3xi32 = private constant [3 x i32] [i32 2, i32 3, i32 4], align 64

define i32 @test_poly_fn(i32 %0) {
  br label %2

2:
  %3 = phi i64 [ %12, %6 ], [ 1, %1 ]
  %4 = phi i32 [ %11, %6 ], [ 0, %1 ]
  %5 = icmp slt i64 %3, 4
  br i1 %5, label %6, label %13

6:
  %7 = sub i64 3, %3
  %8 = mul i32 %0, %4
  %9 = getelementptr inbounds nuw i32, ptr @__constant_3xi32, i64 %7
  %10 = load i32, ptr %9, align 4
  %11 = add i32 %8, %10
  %12 = add i64 %3, 1
  br label %2

13:
  ret i32 %4
}
```

Chapter 18's Horner `scf.for` is now branches and **phi nodes** — where
MLIR used block arguments (`^bb1(%12: i64)`), LLVM uses `phi`; same
concept, older notation — and the coefficient tensor is a
`getelementptr` + `load` from a global. You can read your whole
compiler's work in one screen.

## 5. The lit test: four RUN lines to a running binary

Recall from Chapter 9 that lit executes each `// RUN:` line as a shell
command in a per-test sandbox, substituting placeholders like `%s` and
`%t` — so a sequence of RUN lines is a small script. The RUN header of
`poly_to_llvm.mlir` uses that to script exactly section 4 — compile,
link, run, check the printed output:

***test/part4/poly_to_llvm.mlir*** (excerpt)
```mlir
// RUN: tutorial-opt --poly-to-llvm %s | mlir-translate --mlir-to-llvmir | clang -x ir - -c -o %t.obj
// RUN: clang -c %project_source_dir/test/part4/poly_to_llvm_main.c -o %t.main.o
// RUN: clang %t.main.o %t.obj -o %t.exe
// RUN: %t.exe | FileCheck %s

// CHECK: 351
```

Every lit feature here is from Chapter 9 — `%t`, chained RUN lines,
FileCheck on the *runtime* output — plus one: `%project_source_dir`, the
repo-defined substitution (declared in [`lit.cfg.py`](../../lit.cfg.py))
that locates the `.c` file from the test's sandbox. (The `-` in
`clang -x ir -` means "read stdin", which is why `-x ir` is mandatory
there: with no file extension to sniff, clang must be told the
language.) This is the fully-armed version of functional testing: not
syntax checks this time, but an actual linked binary. Run both e2e tests:

```sh
source .venv/bin/activate && lit build/test --filter=poly_to_llvm -v
```

```
PASS: MLIRUnifiedTutorial :: part4/poly_to_llvm_eval.mlir (1 of 2)
PASS: MLIRUnifiedTutorial :: part4/poly_to_llvm.mlir (2 of 2)
```

## 6. Second thread: type conversion at the LLVM boundary

The poly pipeline never had to invent an LLVM story for `!poly.poly<10>`:
Chapter 18 converted the type to `tensor<10xi32>` early, and from there
every type problem (tensor → memref → ptr+struct) was solved by
*upstream* passes. The `tiny` dialect from Chapter 18's second thread is
the opposite case study: its custom type `!tiny.ptr` (a raw pointer to
f16 memory) has no standard-dialect equivalent, so it rides along
untouched until the very last step — and the `--tiny-to-llvm` pass must
convert the type itself, right at the boundary.

The whole pass lives in
[`lib/Dialect/Tiny/TinyToLLVM.cpp`](../../lib/Dialect/Tiny/TinyToLLVM.cpp).
Its heart is a `TypeConverter` — three lines of policy:

***lib/Dialect/Tiny/TinyToLLVM.cpp*** (excerpt)
```cpp
/// Type converter that converts tiny.ptr to llvm.ptr.
class TinyToLLVMTypeConverter : public TypeConverter {
public:
  TinyToLLVMTypeConverter() {
    // Identity conversion for all types (fallback).
    addConversion([](Type type) { return type; });

    // Convert tiny.ptr to llvm.ptr.
    addConversion([](PtrType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
  }
};
```

Chapter 18 explained the mechanics; the one thing to re-notice is the
*shape* of the rule: everything maps to itself except `!tiny.ptr`, which
becomes the opaque `!llvm.ptr`. The conversion patterns then consume
already-converted operands through the adaptor — here is the load
lowering, `tiny.load %ptr, %offset` → GEP + load:

***lib/Dialect/Tiny/TinyToLLVM.cpp*** (excerpt, condensed)
```cpp
struct LoadOpToLLVMLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The adaptor provides the converted operands (ptr is now !llvm.ptr).
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    // ... index_cast the offset to i64, then:
    // GEP %ptr[%offset] with f16 element type computes
    // ptr + offset * sizeof(f16).
    Value gep = LLVM::GEPOp::create(rewriter, loc, llvmPtrType, f16Type, ptr,
                                    ValueRange{offsetI64});
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), gep);
    return success();
  }
};
```

Two details reward attention:

- **The offset is in elements, not bytes.** `llvm.getelementptr` with an
  `f16` element type multiplies by `sizeof(f16)` for you — the same
  pointer arithmetic C does with a typed pointer. No manual `* 2`.
- **`adaptor.getPtr()` is already `!llvm.ptr`.** The pattern never sees
  a `!tiny.ptr` value; the conversion framework inserted the
  materializations (Chapter 18's `unrealized_conversion_cast`s) and will
  reconcile them once both sides are converted.

And because types hide in *function signatures* — the recurring lesson
of this Part — the pass wires the type converter into `func.func` and
`func.return` legality, using upstream helpers instead of hand-written
signature patterns:

***lib/Dialect/Tiny/TinyToLLVM.cpp*** (excerpt)
```cpp
    // Mark func dialect operations as dynamically legal if their types are
    // converted.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });
    ...
    // Add function signature conversion patterns.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
```

Watch it work on the lit test input
[`test/part4/lower-to-llvm.mlir`](../../test/part4/lower-to-llvm.mlir):

```sh
build/tools/tutorial-opt --tiny-to-llvm test/part4/lower-to-llvm.mlir
```

```mlir
func.func @test_load_lowering(%arg0: !llvm.ptr, %arg1: index) -> vector<4xf16> {
  %0 = arith.index_cast %arg1 : index to i64
  %1 = llvm.getelementptr %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f16
  %2 = llvm.load %1 : !llvm.ptr -> vector<4xf16>
  return %2 : vector<4xf16>
}
...
func.func @test_type_conversion(%arg0: !llvm.ptr) -> !llvm.ptr {
  return %arg0 : !llvm.ptr
}
```

The signature said `(%ptr: !tiny.ptr) -> !tiny.ptr` in the input; both
occurrences became `!llvm.ptr`, including in a function whose *body*
contains no tiny op at all — pure signature conversion. The lit version:

```sh
source .venv/bin/activate && lit build/test --filter=lower-to-llvm -v
```

### The same exit door

Note what `--tiny-to-llvm` deliberately does *not* do: the `func` and
`arith` ops are still there. Finishing the job is, once again, someone
else's code — the wheel's `--convert-to-llvm` umbrella pass (a
one-shot "convert everything that registered an LLVM lowering
interface") mops up `func`, `arith`, and `vector`. Save the square
function from Chapter 18's tiny thread as `/tmp/square.mlir`:

```mlir
func.func @square(%a: !tiny.ptr, %result: !tiny.ptr, %offset: index) {
  %x = tiny.load %a, %offset : vector<16xf16>
  %sq = tiny.mulf %x, %x : vector<16xf16>
  tiny.store %sq, %result, %offset : vector<16xf16>
  return
}
```

and push it through the tiny pipeline and the *same* border crossing as
poly (verified):

```sh
build/tools/tutorial-opt /tmp/square.mlir \
    --tiny-to-arith --canonicalize --cse --tiny-to-llvm --convert-to-llvm \
  | mlir-translate --mlir-to-llvmir \
  | clang -x ir - -c -o /tmp/square.o
```

The intermediate LLVM IR is four instructions of pure vector code:

```llvm
define void @square(ptr %0, ptr %1, i64 %2) {
  %4 = getelementptr half, ptr %0, i64 %2
  %5 = load <16 x half>, ptr %4, align 32
  %6 = fmul <16 x half> %5, %5
  %7 = getelementptr half, ptr %1, i64 %2
  store <16 x half> %6, ptr %7, align 32
  ret void
}
```

Two dialects, two very different type stories, one exit dialect, one
`mlir-translate | clang` tail. That reuse is the whole sales pitch of
lowering *through* LLVM.

## 7. Bonus: a Python DSL driving the descent

The tiny dialect exists to be a compiler *target*, and this repo ships
the front end too: a small Python-embedded DSL
([`python/tiny/`](../../python/tiny/)) that builds tiny IR via MLIR's
Python bindings and shells out to your `tutorial-opt` for every lowering
stage. The chapter's [`examples/square.py`](examples/square.py) is the
whole program:

***examples/square.py***
```python
"""Example: Vectorized square function using ch1 tiny dialect."""

from tiny.ch1 import Ptr, Index, compile_and_print

@compile_and_print
def square(a: Ptr, result: Ptr, offset: Index):
    """Square a vector of 16 f16 elements."""
    x = a.load(offset, num_elements=16)
    x_squared = x * x
    result.store(offset, x_squared)
```

With the tiny package installed in the venv (`pip install -e python/`,
part of the root README setup) and `TUTORIAL_OPT` pointing at your
binary, run it from the repo root (verified):

```sh
TUTORIAL_OPT=$PWD/build/tools/tutorial-opt \
  python part4-lowering/ch19-to-llvm-native-binary/examples/square.py
```

It prints three stages: the generated tiny IR (exactly section 6's
`/tmp/square.mlir` — the DSL emitted it), the IR after `--tiny-to-arith`,
and the final LLVM dialect after `--tiny-to-llvm --convert-to-llvm`:

```
=== Tiny Dialect ===
...
=== After tiny-to-arith ===
...
=== LLVM Dialect ===
  llvm.func @square(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) {
    %0 = llvm.getelementptr %arg0[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %1 = llvm.load %0 : !llvm.ptr -> vector<16xf16>
    %2 = llvm.fmul %1, %1 : vector<16xf16>
    %3 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %2, %3 : vector<16xf16>, !llvm.ptr
    llvm.return
  }
```

The decorator (`compile_and_print` in
[`python/tiny/ch1/dsl.py`](../../python/tiny/ch1/dsl.py)) reads the
function's type annotations to build a `func.func`, executes the Python
body to emit ops, then calls `tutorial-opt` as a subprocess for each
pass list — a complete, if miniature, language front end. Use it as a
playground: edit the function, rerun, watch every layer of this chapter
respond. (Try an elementwise add of two pointers.)

## 8. War stories (read before composing your own pipeline)

The original author documented the dead ends met while composing the
poly pipeline, and they're the most reusable part:

- **Pass order: `func-to-llvm` wants to go last.** Placed early, it
  converts function types before `scf`/`cf` are lowered, and the
  leftovers can't legalize. The symptom is baffling; the fix
  (section 2's stage 5 order) looks obvious only afterward.
- **"expected linalg op with buffer semantics."**
  `convert-linalg-to-loops` silently requires bufferization *before* it.
  The error names neither bufferization nor the fix — recognize it by
  symptom.
- **A plague of `unrealized_conversion_cast i64 → index`.** Cured by
  `finalize-memref-to-llvm` (which lowers `index` itself) followed by
  `reconcile-unrealized-casts` — Chapter 2's mysterious final pass, now
  fully demystified.
- **The best one: an off-by-one caught by running the program.** The
  original eval lowering computed 320 instead of 351 — a wrong loop
  bound in Horner's method. Every syntactic test passed; the *binary*
  didn't. That is the whole argument for section 5's run-the-binary lit
  tests: lowerings lie until you execute them.

## Exercises

1. Reproduce section 4 end to end. Then change the constant to
   `dense<[1, 2, 3]>` and *predict* the output before running — c(1)
   becomes 6, so (6+10)² − 10 = 246 (verified). If your prediction was
   wrong, so was the original author's once; that's what the test is
   for.
2. Rerun with `--mlir-print-ir-after-all` and skim the 25 snapshots.
   Find the exact pass where (a) the last `tensor` type disappears,
   (b) the first `llvm.` op appears, (c) the function signature changes.
3. The wheel ships no `mlir-runner` either, but you don't need C to get
   a runnable binary: append a zero-argument entry point to
   `poly_to_llvm.mlir`

   ```mlir
   func.func @main() -> i32 {
     %c1 = arith.constant 1 : i32
     %r = func.call @test_poly_fn(%c1) : (i32) -> i32
     return %r : i32
   }
   ```

   then compile it with `clang -x ir - -o a.out` (no `-c`, no C file)
   and check the *exit code*: `./a.out; echo $?` prints 95 — which is
   351 mod 256, because exit codes are one byte (verified). A useful
   trick for smoke-testing lowerings without writing a caller.
4. Reproduce section 3's `bufferization.clone` failure, then find *two*
   ways to make it compile: (a) change the function's ABI to scalar-out,
   (b) replay the pipeline as individual flags with
   `--convert-bufferization-to-memref` inserted right after
   `--buffer-deallocation-pipeline`, and confirm `mlir-translate` now
   succeeds (verified; the emitted `.ll` gains `malloc` and `memrefCopy`
   declarations — the clone became a real copy).
5. The pipeline runs canonicalize twice and cleanup at the end. Delete
   the final cleanup quartet (conceptually) — which artifacts would
   survive in the LLVM IR? Check your answer against the actual `.ll`
   with and without `--mlir-print-ir-before=symbol-dce`.
6. In `square.py`, write `accumulate(a, result, offset)` that loads two
   16-element vectors at `offset` and `offset + 16` and stores their sum
   — you'll need the DSL's index arithmetic (`offset +
   Index.constant(16)` emits `tiny.constant` + `tiny.addi`; verified).
   Follow `!tiny.ptr` through all three printed stages: where exactly
   does it die?
7. **Lower `tiny.store` to LLVM** *(mlir-tutor chapter 1, exercise 3)*.
   [`exercises/skeletons/TinyToLLVM.cpp`](exercises/skeletons/TinyToLLVM.cpp)
   is the original exercise file. `tiny.load` → `llvm.getelementptr` +
   `llvm.load` is implemented, and `StoreOpToLLVMLowering` is a TODO. Write
   it as an `OpConversionPattern<StoreOp>`, following the load pattern
   and section 6. Take `value`/`ptr`/`offset` from the *adaptor*, where
   `ptr` has already been converted to `!llvm.ptr` by the TypeConverter.
   `arith.index_cast` the offset to `i64`. Build an `LLVM::GEPOp` with
   element type `f16`, so that the GEP scales the offset by
   `sizeof(f16)`. Finish with `replaceOpWithNewOp<LLVM::StoreOp>`. Then
   follow the skeleton's other two TODOs: add `StoreOp` to
   `addIllegalOp<...>`, and add the pattern to `patterns.add<...>`.
   The skeletons are not compiled. Copy it over the solution, rebuild and
   test:

   ```bash
   cp part4-lowering/ch19-to-llvm-native-binary/exercises/skeletons/TinyToLLVM.cpp lib/Dialect/Tiny/
   cmake --build build -j
   .venv/bin/lit build/test -v --filter 'part4/lower-to-llvm'
   git checkout -- lib/Dialect/Tiny/TinyToLLVM.cpp   # restore the solution
   ```

   The skeleton builds as shipped, and the test fails at
   `@test_store_lowering`. Before you fix it, read the failure twice. First
   read it as shipped, with `StoreOp` still legal, so the partial conversion
   is allowed to leave `tiny.store` behind even though its `!tiny.ptr`
   operand has become an `!llvm.ptr` function argument. Then read it after
   adding `StoreOp` to `addIllegalOp` but before adding the pattern. Which
   message names the real problem? (Compare Chapter 18, section 6.)
8. **Challenge: a 64×64×64 matmul in the Python DSL** *(mlir-tutor
   chapter 1, optional challenge)*. Using only `tiny.ch1` (`Ptr`,
   `Index`, `F16Vector`) and **vectors of width 8**, write
   `matmul(a: Ptr, bt: Ptr, c: Ptr)` computing C = A × B for M = N = K = 64.
   The ch1 DSL has no loop construct, so use ordinary Python `for`
   loops: they run at *trace time* and unroll completely into straight-line
   IR. Hints: pass B transposed (`bt`) so both operands of each dot
   product are contiguous; for each `(i, j)`, accumulate
   `a.load(Index.constant(i*K + k), num_elements=8) * bt.load(...)` over
   `k` in steps of 8 into an `F16Vector.constant([0.0] * 8)`, then store
   `acc.sum()` (a `vector<1xf16>`) at `i*N + j`. Use `print_ir` while
   developing. A version written this way produces about 213,000
   lines of `tiny` IR, and `compile_and_print` takes several seconds to
   produce 32,768 `llvm.fmul`s (64·64·8, verified). That size is the
   argument for the next chapter. Compare your program with
   [Chapter 20's `matmul.py`](../ch20-region-ops-to-scf/examples/matmul.py),
   which writes the same computation with `tiny_loop.accumulate`. There
   the loops survive into the IR as `scf.for` instead of being unrolled
   by Python.

## Where to go next

The `poly` story is complete: from custom syntax (Chapter 12) to a number
printed by a C program — and `tiny` reached the same exit carrying its
own pointer type to the border. But the tiny thread skipped something:
this chapter's `square` has no loops. Real kernels iterate, and the
`tiny_loop` dialect (already visible in
[`test/part4/full-pipeline.mlir`](../../test/part4/full-pipeline.mlir))
provides a custom `accumulate` op with a *region* — lowering ops with
regions and loop-carried values onto `scf.for` is its own set of
techniques, and the subject of
[Chapter 20: Region Ops to SCF](../ch20-region-ops-to-scf/README.md).
(The other thread teased by the original — a `noisy` dialect analyzed
and optimized rather than lowered — returns in
[Chapter 23: Dataflow Analysis](../../part5-structured-and-parallel/ch23-dataflow-analysis/README.md).)
