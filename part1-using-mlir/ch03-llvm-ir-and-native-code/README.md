# Chapter 3: LLVM IR and Native Code

> Adapted from mlir-ir-lowering's `1_intro` chapter and the overview sections of
> its top-level README ("Key components of the modern compiler stack",
> "Standard MLIR dialects", "The toolchain"), themselves based on Stephen
> Diehl's "MLIR" article series. Everything has been re-run with this repo's
> toolchain: the MLIR wheel (LLVM 24) plus Homebrew clang 20 on Apple
> Silicon. Hex addresses and register choices may differ on your machine.

Chapter 2 stopped at a `.ll` file. This chapter finishes the job: it turns LLVM
IR into object files, shared libraries and executables, runs them, calls them
from Python, reads the machine code they contain, and finally JIT-compiles a
module in memory without writing any file.

A compiler is a **staircase of small, meaning-preserving translations**, not a
black box. At the top sits a representation that is comfortable for *humans*:
loops, functions, types, tensors. At the bottom sits one that is comfortable
for *silicon*: registers, branches, fixed-width integers, memory addresses.
Each step down gives up a little abstraction for a little concreteness, and
every step preserves what the program means. LLVM has essentially one landing
on that staircase, LLVM IR. MLIR's bet is that the staircase should have as many
landings as you need, each a dialect pitched at the right altitude, with
lowering as a series of short hops between neighbours.

To make that concrete, this chapter walks the **complete pipeline twice**:

- **Part A** starts near the bottom, with a program written directly in LLVM IR,
  and walks it down to native code. This establishes the destination and the
  tools that reach it.
- **Part B** starts higher up, with a loop written in the `scf` dialect, and lets
  `mlir-opt` lower it to the `llvm` dialect, where it rejoins Part A's path.
  You will watch a structured `scf.for` turn into basic blocks, then phi nodes,
  then ARM64 registers.

Put side by side, the two halves answer the question behind the whole tutorial:
*what does MLIR buy us over plain LLVM IR?* Starting higher up the staircase,
with the descent handled for us. Later chapters (memrefs, linalg, GPUs) only add
taller landings at the top. The bottom steps never change, so we nail them down
here.

## What you will learn

- Where MLIR sits in a modern compiler stack, and which tools make up the
  toolchain (`mlir-opt`, `mlir-translate`, `llc`, `clang`, `mlir-runner`), including
  what the pip wheel lacks and what replaces it in this repo.
- How to read and hand-write a minimal LLVM IR module, and compile it into an
  object file, a shared library and an executable with `clang`.
- How to check a program through its exit code, and how to call a shared
  library from Python with `ctypes`.
- The "standard" low-level dialects (`func`, `scf`/`cf`, `arith`/`math`,
  `index`, `llvm`) and the conversion pass that lowers each one.
- Why `--convert-index-to-llvm` is needed, and what goes wrong without it.
- How to read the generated assembly and `objdump` disassembly, and how much
  the optimization level changes it.
- AOT versus JIT, and how to JIT-run MLIR from Python with
  `mlir.execution_engine.ExecutionEngine` instead of `mlir-runner`.

## Prerequisites

- [Chapter 1](../ch01-why-mlir-and-reading-ir/README.md): reading MLIR
  (operations, SSA values, types, regions, blocks).
- [Chapter 2](../ch02-lowering-pipelines/README.md): running `mlir-opt` pass
  pipelines, `--reconcile-unrealized-casts`, and `mlir-translate --mlir-to-llvmir`.
- The repo's virtual environment set up as in the [root README](../../README.md),
  and a `clang` on your `PATH` (any recent clang works; the outputs below come
  from Homebrew clang 20.1.8). Every `build.sh` below finds the wheel's tools by
  itself, so you can run them from any directory.

## The modern compiler stack

Recall from Chapter 1 why MLIR exists. LLVM IR is powerful but low-level.
Expressing a loop nest, a tensor operation or a parallel region directly in it
is tedious, and it throws away structure the optimizer could have used. So
language after language invented its own *mid-level IR* to bridge the gap:
Rust has MIR, Swift has SIL, TensorFlow had XLA HLO. They all solved the same
problem independently. MLIR generalizes the idea into a framework for many
coexisting IRs (dialects), all of which eventually lower to the `llvm` dialect:

```
  Source        Mid-level IR         Unified backend         Targets
  ──────        ────────────         ───────────────         ───────
  C / C++ ─┐                                               ┌─ x86
  Rust   ──┤    (MIR, SIL, …)                              ├─ ARM
  Swift  ──┤         ──►        MLIR dialects ──► LLVM IR ─┤─ RISC-V
  Julia  ──┤                    (lowering passes)          ├─ NVPTX (GPU)
  PyTorch ─┘                                               └─ WASM
```

This matters because traditional compilers, while excellent on conventional
CPUs, struggle with the specialized hardware of modern AI systems:

- **Heterogeneous hardware.** Systems mix CPUs, GPUs, TPUs and custom ASICs;
  one extensible infrastructure can target all of them.
- **The right level of abstraction.** Dialects let each optimization run at
  the level where it is easy, from ML graphs down to hardware instructions.
- **One unified infrastructure** instead of a separate compiler per
  accelerator or framework, with first-class support for tensors and neural
  networks.
- **"Weird domains."** Spinning up a domain-specific compiler is cheap, so MLIR
  is used well beyond ML: signal processing, quantum computing, homomorphic
  encryption, FPGAs and custom silicon.

### Key components

A modern compiler weaves a few reusable pieces together:

1. **LLVM.** Its IR is a low-level, assembly-like language that still preserves
   important high-level information (types, functions, SSA). It is the bridge
   between source languages and dozens of target architectures.
2. **MLIR.** A multi-level IR, more expressive than LLVM IR, that represents
   higher-level concepts (structured control flow, dataflow, parallelism)
   before being lowered into LLVM IR.
3. **Optimization passes.** Both MLIR and LLVM ship large libraries of passes,
   so any language built on them inherits research-grade optimizations for
   free.
4. **E-graphs.** *Equality saturation*, a technique from automated theorem
   proving, applies rewrite rules via e-matching until the e-graph is
   saturated, then extracts the best term according to a cost function. It
   explores a broad space of transformations at once instead of committing
   to one rewrite order.
   [Chapter 24](../../part5-structured-and-parallel/ch24-equality-saturation-egraphs/README.md)
   builds one.

### The pipeline these components form

1. Source code is parsed into a surface language.
2. The surface language is translated to a core language.
3. The core language is optimized by core-to-core transformations (optionally
   with e-graphs / equality saturation).
4. The core language is lowered to MLIR.
5. MLIR performs high-level optimizations.
6. MLIR is lowered to LLVM IR.
7. LLVM performs low-level optimizations and code generation.

This chapter covers steps 6 and 7. Parts 2–5 of the tutorial cover step 5, and
Part 6 runs the whole pipeline on real models.

## The toolchain

The upstream MLIR tutorials, and the source of this chapter, drive five tools:

| Tool | Role | In this repo's wheel? |
| --- | --- | --- |
| `mlir-opt` | Run lowering and optimization passes on `.mlir` files | Yes |
| `mlir-translate` | Translate the `llvm` dialect into textual LLVM IR (`.ll`) | Yes |
| `llc` | LLVM's static compiler: LLVM IR → native object (`.o`) or assembly (`.s`) | **No**: use `clang -x ir -c` / `clang -x ir -S` |
| `clang` | Link objects into executables or shared libraries | System/Homebrew clang |
| `mlir-runner` | JIT-execute an llvm-dialect module directly | **No**: use the Python bindings' `ExecutionEngine` |

`clang` contains the same LLVM back end as `llc`, so it can compile `.ll`
files directly (`-x ir` says the input is LLVM IR). The MLIR Python bindings,
installed with the wheel, provide `mlir.execution_engine.ExecutionEngine`, the
same ORC-JIT-based engine `mlir-runner` uses internally. The pipeline used
throughout this repo is therefore:

```
 .mlir ──mlir-opt──► llvm dialect ──mlir-translate──► .ll ──ll_fp_compat.py──► .ll ──clang──► .o ──clang──► .dylib / executable
                          │
                          └──── Python: PassManager + ExecutionEngine (JIT, no files) ────► result
```

The `ll_fp_compat.py` step bridges a version gap. The wheel's `mlir-translate`
is LLVM 24, which writes two things clang 20 cannot parse: float constants in
a new syntax, and a new attribute (`nocreateundeforpoison`) on intrinsic
declarations. [`tools/ll_fp_compat.py`](../../tools/ll_fp_compat.py) rewrites
the first to the old form and drops the second. Other modules pass through
unchanged, but piping every `.ll` through it is a good habit.

### Driving mlir-opt: pipelines, pass order and debugging

Every `--convert-X-to-Y` flag is one registered pass, and repeated flags run in
order. For precise control, the same passes can be written as a single
textual pipeline:

```bash
mlir-opt --pass-pipeline="builtin.module(convert-scf-to-cf,convert-cf-to-llvm)" in.mlir
```

**Pass order matters.** `convert-cf-to-llvm` only lowers `cf` ops, so it has to
run *after* `convert-scf-to-cf` produces them. Swap the two and the conversion
silently does nothing: `cf.br` and `cf.cond_br` survive into the output, and
`mlir-translate` rejects the module later.

`--convert-to-llvm` lowers everything it can in one pass, via dialect
interfaces. In practice we prefer the granular passes so we control exactly
what happens. The common granular conversions are:

| Pass | Effect |
| --- | --- |
| `--convert-func-to-llvm` | `func` ops → `llvm.func` / `llvm.call` / `llvm.return` |
| `--convert-arith-to-llvm` | `arith` ops → LLVM instructions |
| `--convert-math-to-llvm` | `math` ops → LLVM intrinsics |
| `--convert-index-to-llvm` | `index` ops → `i64` arithmetic |
| `--convert-scf-to-cf` | structured control flow → `cf` branches |
| `--convert-cf-to-llvm` | `cf` branches → `llvm.br` / `llvm.cond_br` |
| `--finalize-memref-to-llvm` | `memref` ops → pointer arithmetic (Chapter 4) |
| `--convert-vector-to-llvm` | `vector` ops → LLVM vector instructions (Chapter 4) |
| `--convert-linalg-to-loops` | `linalg` ops → `scf` loop nests (Chapter 2) |
| `--reconcile-unrealized-casts` | Remove the leftover bridge casts (Chapter 2) |

When a pipeline misbehaves, these flags print the IR as it changes (to
stderr):

| Flag | What it shows |
| --- | --- |
| `--mlir-print-ir-after-all` | The IR after every pass |
| `--mlir-print-ir-after-change` | Combined with the above: only after passes that changed the IR |
| `--mlir-print-ir-after-failure` | The IR after a pass that failed |
| `--mlir-print-ir-tree-dir=<dir>` | Write each snapshot to a file tree instead of stderr |

Each snapshot starts with a header naming the pass and its options, e.g.:

```
// -----// IR Dump After SCFToControlFlowPass: convert-scf-to-cf{allow-pattern-rollback=true} //----- //
```

### A roadmap of dialects

Chapters 1 and 2 introduced the dialects of the matmul example. Here is the
wider set this tutorial uses, ordered from high to low level, with the chapter
where each one is used most:

| Level | Dialect | Purpose | Key ops | Main chapters |
| --- | --- | --- | --- | --- |
| High | `tensor` | Immutable, value-semantic arrays with **no memory location**; bufferization later turns them into memrefs. Shapes may be static or dynamic (`tensor<?x?xf32>`). | `tensor.empty`, `tensor.from_elements`, `tensor.extract`, `tensor.insert`, `tensor.generate` | 1, 4 |
| High | `linalg` | Structured linear-algebra ops (pointwise, matmul, convolution) that keep the math intact so the compiler can fuse, tile and vectorize. Every named op is a specialization of `linalg.generic`. | `linalg.generic`, `linalg.matmul`, `linalg.fill` | 1, 2, 8, 22 |
| High | `affine` | Loop nests and accesses whose bounds and subscripts are affine functions of loop indices, built on the polyhedral model: dependence analysis, tiling, fusion, parallelization. Lowered to `scf` by `--lower-affine`. | `affine.for`, `affine.parallel`, `affine.load`, `affine.store`, `affine.apply` | 5, 6 |
| High | `omp` | The OpenMP programming model: explicit thread teams, work-sharing loops, synchronization. Lowers to calls into `libomp`. | `omp.parallel`, `omp.wsloop`, `omp.loop_nest` | 7 |
| High | `gpu` | Target-agnostic kernel launches: grids, blocks, threads, device memory. Lowered to a vendor dialect such as `nvvm`. | `gpu.launch`, `gpu.launch_func`, `gpu.thread_id` | 21, 31 |
| Mid | `vector` | Hardware-independent SIMD vectors (`vector<4xf32>`, `vector<2x4xf32>`), mapped by LLVM onto AVX/NEON or scalarized. | `vector.load`, `vector.extract`, `vector.shuffle`, `vector.reduction`, `vector.broadcast` | 4, 5 |
| Low | `scf` | Structured control flow: loops and conditionals with nested regions. | `scf.for`, `scf.if`, `scf.while`, `scf.parallel`, `scf.yield` | 2, 3 |
| Low | `cf` | Unstructured control flow: branches between blocks, with block arguments in place of phi nodes. | `cf.br`, `cf.cond_br`, `cf.switch` | 2, 3 |
| Low | `func` | Function definitions, calls and returns. | `func.func`, `func.call`, `func.return` | all |
| Low | `arith` | Integer and float arithmetic, comparisons, bitwise logic and casts on scalars, vectors and tensors. | `arith.constant`, `arith.addi`, `arith.mulf`, `arith.cmpi`, `arith.sitofp` | all |
| Low | `math` | Math functions beyond arithmetic (trig, exp/log, pow, special functions). Lowered to LLVM intrinsics or libm calls. | `math.sin`, `math.exp`, `math.sqrt`, `math.tanh` | 3, 26 |
| Low | `index` | Operations on the platform-sized `index` type. | `index.constant`, `index.add`, `index.mul`, `index.cmp` | 3 |
| Low | `memref` | Concrete memory buffers: pointer plus offset, sizes and strides. The bridge between NumPy arrays and raw pointers. | `memref.alloc`, `memref.load`, `memref.store`, `memref.subview`, `memref.dim` | 2, 4 |
| Target | `nvgpu` / `nvvm` | NVIDIA-specific: `nvgpu` exposes tensor cores and TMA; `nvvm` mirrors the NVPTX intrinsics and lowers to PTX. | `nvvm.read.ptx.sreg.tid.x`, `nvvm.barrier0`, `nvgpu.mma.sync` | 31 |
| Lowest | `llvm` | A near 1:1 mirror of LLVM IR, plus C-like structs, arrays and raw pointers. Every CPU pipeline ends here. | `llvm.func`, `llvm.call`, `llvm.getelementptr`, `llvm.load`, `llvm.insertvalue` | all |

These are a fraction of what ships upstream. `mlir-opt --show-dialects` on
this wheel lists 48. Roughly grouped:

| Group | Upstream dialects in this wheel |
| --- | --- |
| Core / structural | `builtin`, `func`, `cf`, `scf`, `ub`, `dlti` |
| Computation | `arith`, `math`, `index`, `complex` |
| Data and memory | `tensor`, `memref`, `bufferization`, `sparse_tensor`, `vector`, `ptr`, `shape` |
| Structured / ML | `linalg`, `affine`, `tosa`, `ml_program`, `quant`, `shard` |
| Parallelism and runtime | `omp` (OpenMP), `acc` (OpenACC), `async`, `mpi` |
| GPU / accelerators | `gpu`, `nvgpu`, `nvvm`, `amdgpu`, `rocdl`, `xegpu`, `xevm`, `spirv` |
| CPU vector ISAs | `x86`, `arm_neon`, `arm_sve`, `arm_sme` |
| Exit points | `llvm` (→ LLVM IR), `emitc` (→ C/C++), `spirv` (→ SPIR-V), `wasmssa` (→ WebAssembly) |
| Meta / tooling | `transform`, `pdl`, `pdl_interp`, `irdl`, `smt` |

Projects outside LLVM add their own: `torch` (Torch-MLIR, [Chapter 27](../../part6-end-to-end-ml/ch27-torch-mlir-import/README.md)),
`stablehlo` (XLA), `triton`, CIRCT's hardware dialects, and more.

---

## Part A: raw LLVM IR

Before reaching for MLIR, look at the layer underneath it. MLIR's job is to
eventually produce LLVM IR, so once you know how a tiny piece of LLVM IR
becomes a callable native function, Part B is the same path with extra
lowering stages in front. Part A also sets the end goal: **a shared library
Python can call**, the bridge into the ML ecosystem used throughout this
tutorial.

> **What is a module?** Every LLVM IR file is a *module*: the top-level
> container of functions, globals and metadata, roughly one translation unit
> (one `.c` file). MLIR borrows the idea: its top-level container is also a
> `module`. Compilation at every level is a series of transformations from
> one module to another.

### The source

[`examples/01-llvm-modules/simple.ll`](examples/01-llvm-modules/simple.ll):

```llvm
define i32 @main() {
  ret i32 42
}
```

This is hand-written LLVM IR, equivalent to the C program:

```c
int main() { return 42; }
```

`define i32 @main()` declares a function named `main` returning a 32-bit
integer; `ret i32 42` returns the constant `42`.

Three conventions here recur in MLIR:

- **`@` is a symbol (global) name.** `@main` is a globally visible function,
  the same name the linker and `ctypes` look up later. Local SSA values use
  `%`, as we will see in Part B.
- **Types are explicit everywhere.** `i32` is a 32-bit integer type. LLVM
  has `i1`, `i8`, `i64` and so on; there is no implicit `int`. Writing the
  width out is part of what lets the same IR target many architectures.
- **It is already SSA.** Every value is assigned exactly once. That
  constraint is invisible here, but it becomes central in Part B.

### The build

[`examples/01-llvm-modules/build.sh`](examples/01-llvm-modules/build.sh) runs
these steps (paths shortened):

```bash
clang -x ir -c -fPIC simple.ll -o build/simple.o        # 1. LLVM IR -> object
clang -shared -fPIC build/simple.o -o build/libsimple.dylib  # 2. object -> shared library
clang build/simple.o -o build/simple                    # 3. object -> executable
build/simple; echo "exit code: $?"                      # 4. run it
python simple.py                                        # 5. call it from Python
```

```
$ bash part1-using-mlir/ch03-llvm-ir-and-native-code/examples/01-llvm-modules/build.sh
warning: overriding the module target triple with arm64-apple-macosx16.0.0 [-Woverride-module]
1 warning generated.
exit code: 42
from Python: 42
```

**Step 1: LLVM IR → object code.** The source used `llc`, LLVM's static
compiler. It is the back end: instruction selection, register allocation,
and emission of a native object file (`.o`) full of machine code. `clang -x ir
-c` runs the very same back end. `-fPIC` (the `llc` spelling is
`--relocation-model=pic`) emits *position-independent code*, which works at
whatever address it is loaded. A shared library needs that, because the OS
may map it anywhere in a process's address space.

The warning is harmless. `simple.ll` names no target, so clang fills in the
host's triple (`arm64-apple-macosx16.0.0` here) and says so. Later scripts
pass `-Wno-override-module` to silence it.

**Step 2: object → shared library.** A bare `.o` is not loadable. The linker
packages it into `libsimple.dylib` (a `.so` on Linux): a self-contained unit
that other programs, including the Python interpreter, can `dlopen` at
runtime and call into.

**Step 3: object → executable.** The same object also links into a standalone
binary.

**Step 4: run it.** The program only returns `42`, which becomes the process
**exit code**. `$?` holds the exit code of the last command.

**Step 5: call it from Python.**
[`simple.py`](examples/01-llvm-modules/simple.py) uses the standard-library
`ctypes` module:

```python
module = ctypes.CDLL(os.path.join(HERE, "build", lib_name))

# Declare the C signature of `main` so ctypes marshals values correctly.
module.main.argtypes = []            # main takes no arguments
module.main.restype = ctypes.c_int   # main returns a C int (i32)

print(module.main())
```

- `ctypes.CDLL(path)` `dlopen`s the library and returns a handle whose
  attributes are its exported symbols (`module.main` is the C `main`).
- `argtypes` / `restype` tell ctypes how to convert Python values to and
  from C. Without `restype`, ctypes assumes a C `int`. That happens to be
  right here, but setting it is essential once return types are not `int`.
- `module.main()` invokes the native code and returns `42` as a Python `int`.
  Unlike step 4, this `42` is a *return value* we print, not an exit code.

That is the point of Part A: **anything compiled through LLVM becomes a plain
shared library callable from Python.**

---

## Part B: from high-level MLIR

Returning 42 is trivial in LLVM IR. Real programs have loops, and writing a
loop in raw LLVM IR means managing basic blocks and **phi nodes** by hand.

> **Why loops are awkward in SSA.** SSA says every value is assigned exactly
> once, but a loop counter by definition changes. LLVM reconciles the two with
> a **phi node**: a pseudo-instruction at the top of a block whose value is
> "whichever incoming value belongs to the edge we arrived on" (`0` on entry,
> `previous + 1` on each back edge). It is correct but fiddly bookkeeping. Part
> B lets MLIR generate all of it from a plain `scf.for`.

### The source

[`examples/02-mlir-to-native/example.mlir`](examples/02-mlir-to-native/example.mlir):

```mlir
// loop_add: sum the integers 0..9 using a structured (scf) for-loop.
// Returns `index` (a platform-sized integer, like size_t).
func.func @loop_add() -> (index) {
  // `index` dialect: loop bounds and the running total are index-typed.
  %init = index.constant 0   // initial accumulator value
  %lb = index.constant 0     // loop lower bound (inclusive)
  %ub = index.constant 10    // loop upper bound (exclusive)
  %step = index.constant 1   // loop step

  // scf.for is a *structured* loop. `iter_args` threads a loop-carried value
  // (%acc) through each iteration; whatever we scf.yield becomes %acc next
  // time, and the final value is bound to %sum.
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (index) {
    %sum_next = arith.addi %acc, %iv : index   // arith dialect: acc + iv
    scf.yield %sum_next : index                // carry sum_next into next iter
  }
  return %sum : index
}

// main: call loop_add and return the result as a C-style i32 exit code.
func.func @main() -> i32 {
  %out = call @loop_add() : () -> index
  // index is i64 here; narrow it to i32 so main can return a normal exit code.
  %out_i32 = arith.index_cast %out : index to i32
  return %out_i32 : i32
}
```

In C:

```c
long loop_add(void) {
  long sum = 0;
  for (long iv = 0; iv < 10; iv += 1)
    sum = sum + iv;
  return sum;            // 0+1+...+9 = 45
}
int main(void) { return (int)loop_add(); }
```

This one file uses **four dialects**: `func` (functions and `call`), `index`
(loop bounds), `scf` (the structured loop) and `arith` (`addi`,
`index_cast`). Mixing them is the point of MLIR, not an accident. Dialects are
designed to coexist in one module, each modeling the part of the program it
suits best, and each is lowered independently. Two type details stand out:

- **`index` is platform-sized.** Like C's `size_t`, it is the target's natural
  pointer-width integer (64 bits on Apple Silicon), the right type for loop
  counters and array offsets. That is why `main` needs `arith.index_cast` to
  narrow it to a fixed-width `i32` before returning it as an exit code.
- **`scf.for` returns a value.** A C `for` loop is a statement that mutates a
  variable; `scf.for` with `iter_args` is an expression that yields its final
  loop-carried value. This functional framing keeps the IR in SSA form without
  any phi nodes at this level.

The loop sums 0..9, so this program returns **45**, deliberately different
from Part A's 42.

### The standard dialects, one by one

`example.mlir` uses four dialects, and lowering it passes through two more
(`cf`, then `llvm`). Add `math` and you have the six dialects Stephen Diehl's
article calls the **"standard MLIR dialects"**: the low-level floor almost
every pipeline lowers through on its way to `llvm`. Add `func`, which every
program uses, and you have the set this section tours.

> **Is "standard" an official term?** Not anymore. MLIR once had a single
> catch-all **`std` dialect** holding arithmetic, branches, calls, memory ops
> and more. Around 2021–2022 (LLVM 14–15) it was split into focused dialects:
> `arith`, `math`, `func`, `cf`, `memref`, `tensor` and others. "Standard" here
> just means the common low-level floor.

Every function below lives in
[`examples/03-dialect-tour/dialect_tour.mlir`](examples/03-dialect-tour/dialect_tour.mlir),
and [`build.sh`](examples/03-dialect-tour/build.sh) lowers them together,
compiles them, and calls each one from a C driver
([`driver.c`](examples/03-dialect-tour/driver.c)):

```
$ bash part1-using-mlir/ch03-llvm-ir-and-native-code/examples/03-dialect-tour/build.sh
arithmetic_example(1, 2.9) = 45
math_example(2.0)          = 1.285941
compute_offset(2, 3, 10)   = 23
select(7, 9, true/false)   = 7 9
classify(0, 1, 5)          = 10 20 -1
abs_diff(3, 10), (10, 3)   = 7 7
collatz_steps(27)          = 111
```

(Python agrees: `sin(2)·√2 = 1.2859407…`, and 27 takes 111 Collatz steps.)

#### `func`: functions and calls

`func.func` defines a function with a typed signature, `func.call` (printed
`call` inside a function) invokes one, and `func.return` (`return`) hands back
results. `--convert-func-to-llvm` turns them into `llvm.func`, `llvm.call` and
`llvm.return`. The lowered signatures map directly onto C prototypes, which is
how `driver.c` can declare and call them:

```c
int32_t arithmetic_example(int32_t a, float b);   // (i32, f32) -> i32
int64_t compute_offset(int64_t i, int64_t j, int64_t stride);  // index -> i64
int32_t select(int32_t a, int32_t b, bool flag);  // i1 -> bool
```

#### `scf` and `cf`: two views of control flow

**`scf`** (structured control flow) offers the constructs of a real
language: `scf.if`, `scf.for`, `scf.while`. Each has one entry and one exit,
with its body nested as a **region**, so an optimizer sees the bounds, step
and body of a loop as one op:

```mlir
scf.if %b {
  // true region
} else {
  // false region
}

scf.for %iv = %lb to %ub step %step {
  // loop region
}
```

`scf.if` can return values too: each branch `scf.yield`s one, and the `if` is
an expression. `@abs_diff` in the tour uses that. `@collatz_steps` uses
`scf.while`, whose "before" region computes the condition with
`scf.condition` and whose "after" region is the body.

**`cf`** (unstructured control flow) is what those structures compile into:
plain branches between basic blocks.

| Op | Meaning |
| --- | --- |
| `cf.br ^bb1` | Unconditional jump |
| `cf.cond_br %c, ^bb1, ^bb2` | Jump to one of two blocks based on an `i1` |
| `cf.switch %i : i32, [...]` | Multi-way jump selected by an integer |

You can watch `example.mlir` take exactly this step by running *only*
`--convert-scf-to-cf` (`build.sh` saves this as `build/example_cf.mlir`):

```mlir
module {
  func.func @loop_add() -> index {
    %idx0 = index.constant 0
    %idx0_0 = index.constant 0
    %idx10 = index.constant 10
    %idx1 = index.constant 1
    cf.br ^bb1(%idx0_0, %idx0 : index, index)
  ^bb1(%0: index, %1: index):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %0, %idx10 : index
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %3 = arith.addi %1, %0 : index
    %4 = arith.addi %0, %idx1 : index
    cf.br ^bb1(%4, %3 : index, index)
  ^bb3:  // pred: ^bb1
    return %1 : index
  }
  ...
}
```

The `scf.for` is gone. The loop is now explicit blocks (`^bb1` header, `^bb2`
body, `^bb3` exit), and the `iter_args` have become **block arguments** (`%0`
= `iv`, `%1` = `acc`). Everything else (`index`, `arith`, `func`) is
untouched, because this pass only knows about `scf`. Block arguments can also
select between values, the way a phi node does (`@select` in the tour):

```mlir
func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)
^bb1(%x: i32):
  return %x : i32
}
```

`--convert-cf-to-llvm` maps `cf.br` / `cf.cond_br` / `cf.switch` one-to-one
onto `llvm.br` / `llvm.cond_br` / `llvm.switch`. The tour's `@classify` ends
up as a real LLVM `switch` instruction:

```llvm
define i32 @classify(i32 %0) {
  switch i32 %0, label %4 [
    i32 0, label %2
    i32 1, label %3
  ]
```

#### `arith` and `math`: computation

**`arith`** holds the basic integer and floating-point operations, on
scalars as well as vectors and tensors:

| Family | Operations |
| --- | --- |
| Constants | `constant` |
| Integer arithmetic | `addi`, `subi`, `muli`, `divsi` (signed), `divui` (unsigned), `remsi`, `remui` |
| Float arithmetic | `addf`, `subf`, `mulf`, `divf` |
| Comparisons | `cmpi` (integer), `cmpf` (float) |
| Conversions | `extsi`/`extui` (sign/zero extend), `trunci`, `fptosi`/`fptoui`, `sitofp`/`uitofp`, `index_cast` |
| Bitwise | `andi`, `ori`, `xori` |

The op name carries the type family (`addi` vs `addf`) and, where it matters,
the signedness (`divsi` vs `divui`). Integer *types* like `i32` have no sign;
as in LLVM IR, the operation decides how to read the bits.

```mlir
func.func @arithmetic_example(%a: i32, %b: f32) -> i32 {
  %1 = arith.constant 42 : i32       // an integer constant
  %2 = arith.addi %a, %1 : i32       // integer addition
  %3 = arith.fptosi %b : f32 to i32  // float -> signed integer (truncates)
  %4 = arith.addi %2, %3 : i32
  return %4 : i32
}
```

With `a = 1`, `b = 2.9` that is `1 + 42 + 2 = 45`: `fptosi` truncates toward
zero.

**`math`** adds more elaborate functions:

| Family | Operations |
| --- | --- |
| Trigonometric | `sin`, `cos`, `tan`, `atan2` |
| Exponential / log | `exp`, `exp2`, `log`, `log2`, `log10` |
| Power | `powf`, `sqrt`, `rsqrt` |
| Special | `erf`, `tanh` |

```mlir
func.func @math_example(%x: f32) -> f32 {
  %1 = math.sin %x : f32        // sin(x) * sqrt(x)
  %2 = math.sqrt %x : f32
  %3 = arith.mulf %1, %2 : f32
  return %3 : f32
}
```

`--convert-arith-to-llvm` maps `arith` onto plain LLVM instructions (`addi` →
`llvm.add`, `cmpi slt` → `llvm.icmp "slt"`, `index_cast` → `llvm.trunc` or
`llvm.sext`). `--convert-math-to-llvm` maps `math` onto LLVM **intrinsics**:

```mlir
  llvm.func @math_example(%arg0: f32) -> f32 {
    %0 = llvm.intr.sin(%arg0) : (f32) -> f32
    %1 = llvm.intr.sqrt(%arg0) : (f32) -> f32
    %2 = llvm.fmul %0, %1 : f32
    llvm.return %2 : f32
  }
```

The back end then decides how to implement each intrinsic. On ARM64,
`sqrt` is a single instruction while `sin` becomes a call into libm:

```asm
_math_example:
	stp	d9, d8, [sp, #-32]!
	stp	x29, x30, [sp, #16]
	fmov	s8, s0
	bl	_sinf
	fsqrt	s1, s8
	fmul	s0, s0, s1
	...
```

`example.mlir` has no `math` ops, so `--convert-math-to-llvm` has nothing to
do in its pipeline. It stays there so the same pipeline works on programs that
do use `math`.

#### `index`: addressing and loop counters

The `index` dialect works on the `index` **type**. Its ops mirror `arith`
but take and return `index`:

| Operation | Purpose |
| --- | --- |
| `index.constant` | Create an index constant |
| `index.add` / `index.sub` | Add / subtract |
| `index.mul` | Multiply |
| `index.divs` / `index.rems` | Signed division / remainder |
| `index.cmp` | Compare |

A typical use is flattening multi-dimensional coordinates into a memory
offset, here the row-major formula `i * stride + j`:

```mlir
func.func @compute_offset(%i: index, %j: index, %stride: index) -> index {
  %1 = index.mul %i, %stride
  %2 = index.add %1, %j
  return %2 : index
}
```

`arith` ops also accept `index` operands: `example.mlir` uses
`arith.addi ... : index` in the loop body. `--convert-index-to-llvm` replaces
`index` with the target's pointer-width integer, which is why every value in
the lowered output below is `i64`. The `index` type comes into its own with
memrefs in Chapter 4.

#### `llvm`: the bottom landing

The `llvm` dialect is a near 1:1 mirror of LLVM IR (`llvm.func`, `llvm.add`,
`llvm.icmp`, `llvm.br`, `llvm.call`, `llvm.mlir.constant`, …) and the lowest
level of the hierarchy. Once a module is entirely in it, `mlir-translate` can
pass it straight through to LLVM IR. Every dialect above reaches it one
`--convert-X-to-llvm` pass at a time:

| Dialect | Lowering pass | Ends up as |
| --- | --- | --- |
| `func` | `--convert-func-to-llvm` | `llvm.func` / `llvm.call` / `llvm.return` |
| `scf` | `--convert-scf-to-cf` | `cf` branches (then ↓) |
| `cf` | `--convert-cf-to-llvm` | `llvm.br` / `llvm.cond_br` / `llvm.switch` |
| `arith` | `--convert-arith-to-llvm` | LLVM instructions (`llvm.add`, `llvm.icmp`, …) |
| `math` | `--convert-math-to-llvm` | LLVM intrinsics (`llvm.intr.sin`, …) |
| `index` | `--convert-index-to-llvm` | `i64` arithmetic in `llvm` |

That table *is* the `mlir-opt` command of the next step.

### Step 1: mlir-opt lowers everything to the llvm dialect

[`examples/02-mlir-to-native/build.sh`](examples/02-mlir-to-native/build.sh)
runs the whole pipeline; from the repo root:

```
$ bash part1-using-mlir/ch03-llvm-ir-and-native-code/examples/02-mlir-to-native/build.sh
exit code: 45
from Python (ctypes): 45
JIT (ExecutionEngine): 45
```

Its first real step is:

```bash
mlir-opt example.mlir \
    --convert-func-to-llvm \
    --convert-math-to-llvm \
    --convert-index-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    -o build/example_opt.mlir
```

**What lowering means.** Lowering translates an op from a higher-level dialect
into one or more ops of a lower-level dialect with the same runtime behavior
and less abstraction. You never lower the whole program in one leap. A
sequence of small passes, each responsible for one dialect, runs until
everything has bottomed out in `llvm`. That design is what makes the
machinery reusable: each dialect only has to know how to take one step down.
While a conversion is in progress the module is mixed, and MLIR bridges the
two worlds with `unrealized_conversion_cast` placeholders, which
`reconcile-unrealized-casts` removes once they cancel out (Chapter 2).

The result, `build/example_opt.mlir`, is entirely in the `llvm` dialect:

```mlir
module {
  llvm.func @loop_add() -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(10 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb1(%1, %0 : i64, i64)
  ^bb1(%4: i64, %5: i64):  // 2 preds: ^bb0, ^bb2
    %6 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %7 = llvm.add %5, %4 : i64
    %8 = llvm.add %4, %3 : i64
    llvm.br ^bb1(%8, %7 : i64, i64)
  ^bb3:  // pred: ^bb1
    llvm.return %5 : i64
  }
  llvm.func @main() -> i32 {
    %0 = llvm.call @loop_add() : () -> i64
    %1 = llvm.trunc %0 : i64 to i32
    llvm.return %1 : i32
  }
}
```

The loop-carried values became the block arguments of `^bb1` (`%4` = induction
variable, `%5` = accumulator). `%6 = llvm.icmp "slt" %4, %2` is the `i < 10`
test, `%7 = llvm.add %5, %4` is `sum += iv`, and `%8 = llvm.add %4, %3` is
`iv += 1`. The `index` type is gone: every value is an `i64`, and
`arith.index_cast` became `llvm.trunc`.

#### Why `--convert-index-to-llvm` matters

Drop that one flag and the pipeline still "succeeds", but the `index.constant`
ops survive, each wrapped in a cast that `reconcile-unrealized-casts` cannot
remove because its other half never got converted:

```mlir
  llvm.func @loop_add() -> i64 {
    %idx0 = index.constant 0
    %0 = builtin.unrealized_conversion_cast %idx0 : index to i64
    ...
```

`mlir-translate` then refuses the module, because only LLVM-translatable
dialects are loaded on its side:

```
<stdin>:3:13: error: Dialect `index' not found for custom op 'index.constant'
    %idx0 = index.constant 0
            ^
```

The `arith` ops on `index` values do not need this pass; `convert-arith-to-llvm`
handles their types itself. It is the `index` *dialect* ops that need their
own conversion. That is the general rule: every dialect left in the module
needs a pass that lowers it.

### Step 2: mlir-translate leaves MLIR

```bash
mlir-translate build/example_opt.mlir --mlir-to-llvmir -o build/example.ll
```

> **The `llvm` dialect is not LLVM IR.** This catches everyone at first. After
> step 1 the module is in the *`llvm` dialect*: still MLIR data structures,
> parsed and printed by MLIR tools, just with ops that mirror LLVM IR
> one-to-one. `mlir-translate` is the bridge that leaves MLIR and emits real
> LLVM IR text that `clang`, `opt` and `llc` understand. The split lets MLIR
> keep a faithful in-memory model of LLVM IR, and run its own passes on it,
> without depending on LLVM's data structures until the last moment.

`build/example.ll`:

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @loop_add() {
  br label %1

1:                                                ; preds = %5, %0
  %2 = phi i64 [ %7, %5 ], [ 0, %0 ]
  %3 = phi i64 [ %6, %5 ], [ 0, %0 ]
  %4 = icmp slt i64 %2, 10
  br i1 %4, label %5, label %8

5:                                                ; preds = %1
  %6 = add i64 %3, %2
  %7 = add i64 %2, 1
  br label %1

8:                                                ; preds = %1
  ret i64 %3
}

define i32 @main() {
  %1 = call i64 @loop_add()
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

Here the loop is expressed with real **phi nodes**, the bookkeeping MLIR
saved us from writing. LLVM IR has no block arguments, so `mlir-translate`
converts `^bb1(%4, %5)` into the two `phi i64` nodes `%2` (induction variable)
and `%3` (accumulator). Each lists one incoming value per predecessor: `0`
from the entry block `%0`, the updated value from the body block `%5`. The
constants were folded into the instructions (`icmp slt i64 %2, 10`).

`build.sh` then pipes the file through `tools/ll_fp_compat.py` into
`build/example.compat.ll`. This module has no float constants or intrinsics,
so nothing changes, but the step is kept so the script stays correct when you
edit it.

### Step 3: clang compiles and links

```bash
LLC_LIKE="-Wno-override-module -O2 -Xclang -disable-llvm-passes"
clang $LLC_LIKE -x ir -c -fPIC build/example.compat.ll -o build/example.o
clang -shared -fPIC build/example.o -o build/libexample.dylib   # shared library
clang build/example.o -o build/example                          # executable
build/example; echo "exit code: $?"
```

These are Part A's final stages, now with an MLIR program on top. The
`LLC_LIKE` flags reproduce what `llc` does by default, explained in step 4.
The object is linked two ways:

- `-shared` gives `libexample.dylib`, loadable from Python just like
  `simple.py` (`build.sh` calls `ctypes.CDLL(...).main()` and prints `45`).
- No `-shared` gives a standalone executable whose `main` return value is the
  process exit code: `exit code: 45`.

The high-level MLIR loop has become an ordinary native program.

### Step 4: read the machine code

Looking at the assembly is the final check that the chain produced *good*
code, not just *correct* code. This is where the optimization level matters
a lot, and where `clang` and `llc` differ:

- `llc` defaults to `-O2` **for the back end only**: instruction selection
  and register allocation are optimized, but the IR is compiled as given.
- `clang -O2` *also* runs LLVM's IR optimizer (inlining, loop analysis,
  constant folding) before the back end.
- `-Xclang -disable-llvm-passes` tells clang to skip the IR optimizer, which
  makes `clang -O2` behave like `llc`.

`build.sh` writes the assembly three ways (`clang -x ir -S`). **Back end
only** (`build/example.s`, the `llc` equivalent; directives elided):

```asm
_loop_add:                              ; @loop_add
	mov	x8, #0                          ; iv  = 0
	mov	x0, #0                          ; sum = 0
	cmp	x8, #9
	b.gt	LBB0_2
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	add	x0, x0, x8                      ; sum += iv
	add	x8, x8, #1                      ; iv  += 1
	cmp	x8, #9
	b.le	LBB0_1
LBB0_2:
	ret                                     ; return sum (in x0)
_main:                                  ; @main
	stp	x29, x30, [sp, #-16]!
	bl	_loop_add
	ldp	x29, x30, [sp], #16
	ret
```

(Comments after the `;` on instruction lines were added by hand.) This is the
payoff of the real LLVM back end. The loop bound `10` became the immediate
`#9` in `cmp` / `b.le` (`iv <= 9`), the induction variable and accumulator
live entirely in registers (`x8`, `x0`) with no memory traffic, and the body
is a tight loop with a single conditional branch. The first `cmp`/`b.gt` is
the guard for a loop that might run zero times; the back end does not know it
never does. The result is already in `x0`, the ARM64 return register.

**No optimization** (`build/example.O0.s`): the same loop, but every value
round-trips through the stack:

```asm
_loop_add:
	sub	sp, sp, #48
	mov	x8, #0
	mov	x9, x8
	str	x9, [sp, #32]                   ; 8-byte Folded Spill
	str	x8, [sp, #40]                   ; 8-byte Folded Spill
	b	LBB0_1
LBB0_1:
	ldr	x9, [sp, #32]                   ; 8-byte Folded Reload
	ldr	x8, [sp, #40]                   ; 8-byte Folded Reload
	...
```

**Full `clang -O2`** (`build/example.O2.s`): the IR optimizer notices the loop
has constant bounds and no inputs, computes the answer at compile time, and
deletes the loop and the call:

```asm
_loop_add:                              ; @loop_add
	mov	w0, #45                         ; =0x2d
	ret
_main:                                  ; @main
	mov	w0, #45                         ; =0x2d
	ret
```

Three settings, three very different binaries from the same `.ll`. When you
read compiler output, always know which optimizations ran.

**Disassembly.** `build.sh` also disassembles the object and the library with
`objdump -d --no-show-raw-insn`, which shows addresses. The object file
(`build/example.dis`):

```
0000000000000000 <ltmp0>:
       0:      	mov	x8, #0x0                ; =0
       4:      	mov	x0, #0x0                ; =0
       8:      	cmp	x8, #0x9
       c:      	b.gt	0x20 <ltmp0+0x20>
      10:      	add	x0, x0, x8
      14:      	add	x8, x8, #0x1
      18:      	cmp	x8, #0x9
      1c:      	b.le	0x10 <ltmp0+0x10>
      20:      	ret

0000000000000024 <_main>:
      24:      	stp	x29, x30, [sp, #-0x10]!
      28:      	bl	0x28 <_main+0x4>
      2c:      	ldp	x29, x30, [sp], #0x10
      30:      	ret
```

The shared library (`build/libexample.dis`):

```
00000000000003a8 <_loop_add>:
     3a8:      	mov	x8, #0x0                ; =0
     ...
00000000000003cc <_main>:
     3cc:      	stp	x29, x30, [sp, #-0x10]!
     3d0:      	bl	0x3a8 <_loop_add>
     3d4:      	ldp	x29, x30, [sp], #0x10
     3d8:      	ret
```

Two things are visible only at this level:

- **Relocations.** In the object, `bl 0x28` branches to *itself*. The call
  target is not known yet, so the assembler left a placeholder plus a
  relocation record ("patch this with the address of `_loop_add`"). The linker
  resolved it in the dylib: `bl 0x3a8 <_loop_add>`. `objdump -dr` prints the
  relocation records next to the instructions.
- **Symbols.** Mach-O (macOS) prefixes C symbols with `_`, so `@loop_add`
  becomes `_loop_add`; ctypes and C callers still use the plain name. In the
  object, objdump labels the first function with the assembler's local
  `ltmp0` symbol at the same address. `nm -gU build/libexample.dylib` lists
  the exported symbols: `_loop_add` and `_main`, both `T` (in the text
  section).

### Step 5: JIT instead of AOT

There are two ways to run compiled code, and later chapters use both:

- **AOT (ahead of time)**, steps 2–4: produce a `.dylib` or executable on disk
  now and run it later. This is what you ship.
- **JIT (just in time)**: compile to machine code *in memory* when you run,
  execute immediately, and write no files. This is what you use while
  iterating.

The source tutorial JIT-ran the lowered module with `mlir-runner -e main
-entry-point-result=i32 build/example_opt.mlir`. The wheel has no
`mlir-runner`, but it ships the MLIR **Python bindings**, which expose the same
JIT engine. [`examples/02-mlir-to-native/jit.py`](examples/02-mlir-to-native/jit.py)
does the whole thing in-process, without even shelling out to `mlir-opt`:

```python
from mlir.execution_engine import ExecutionEngine
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

PIPELINE = (
    "builtin.module("
    "convert-func-to-llvm,convert-math-to-llvm,convert-index-to-llvm,"
    "convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,"
    "reconcile-unrealized-casts)"
)

with Context():
    module = Module.parse(open(src).read())
    PassManager.parse(PIPELINE).run(module.operation)

    # Compile the whole module to machine code in memory.
    engine = ExecutionEngine(module, opt_level=2)

    # For every function @f, ExecutionEngine also emits a "packed" wrapper
    # `_mlir_f(void **args)`: args[i] points at the i-th argument, and the
    # slots after the arguments point at where results should be written.
    result = ctypes.c_int32(-1)
    packed_args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.pointer(result), ctypes.c_void_p))
    main = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(engine.raw_lookup("main"))
    main(packed_args)

print(result.value)   # 45
```

`PassManager.parse` takes exactly the textual pipeline syntax of `mlir-opt
--pass-pipeline`. `ExecutionEngine` runs the equivalent of `mlir-translate`
and the LLVM back end internally. The *packed* calling convention, a single
`void**` array of argument and result pointers, is how a generic runner calls
functions of any signature. It is also what `mlir-runner` uses under the
hood. Chapter 4 meets the friendlier `engine.invoke(...)`, which works on
functions that have a C interface.

When the MLIR calls runtime helpers such as `printMemref`, the runtime
libraries must be loaded into the JIT, as `mlir-runner -shared-libs=...` did.
With `ExecutionEngine`, pass
`shared_libs=[".../mlir_wheel/lib/libmlir_c_runner_utils.dylib", ...]`. The
wheel ships `libmlir_c_runner_utils` and `libmlir_runner_utils` in
`.venv/lib/python3.12/site-packages/mlir_wheel/lib`. This example does not
need them.

The JIT is a shortcut for *running* code, not part of the build-to-native
path. It consumes `example.mlir` alongside steps 2–4, not after them.

---

## The big picture

The two parts are the same staircase entered at different heights:

```text
   Part B (02-mlir-to-native)                  Part A (01-llvm-modules)
   high-level MLIR    example.mlir   ─┐
     scf / index / arith / func       │  mlir-opt (lowering passes)
   llvm dialect       example_opt.mlir│
     │  mlir-translate                ▼
   LLVM IR            example.ll  ◄─────────  simple.ll   ← Part A starts here
     │  clang -x ir -c                        │  clang -x ir -c
   object (.o)        example.o               simple.o
     │  clang                                 │  clang
   native .dylib / executable  ◄──────────────┘   → callable from Python (ctypes)
```

| | Part A (`01-llvm-modules`) | Part B (`02-mlir-to-native`) |
| --- | --- | --- |
| You write | raw LLVM IR | high-level MLIR (`scf.for`) |
| Loops / phi nodes | by hand | generated by lowering |
| Lowering tool | none (already LLVM IR) | `mlir-opt` passes |
| To LLVM IR | it *is* LLVM IR | `mlir-translate` |
| To native object | `clang -x ir -c` (was `llc`) | `clang -x ir -c` (was `llc`) |
| Shared library + executable | `clang` | `clang` |
| JIT option | none | `ExecutionEngine` (was `mlir-runner`) |

Both end up as ordinary native code, but MLIR let us express the loop at a
human level and lowered it mechanically, phi nodes included. Every later
chapter adds higher dialects (`memref`, `linalg`, `gpu`) on top of this same
bottom.

The key results are covered by lit tests:
[`test/part1/ch03_llvm_ir_and_native_code.mlir`](../../test/part1/ch03_llvm_ir_and_native_code.mlir)
(exit codes 42 and 45, the lowered IR, and the two assembly variants) and
[`test/part1/ch03_dialect_tour.mlir`](../../test/part1/ch03_dialect_tour.mlir).

## Exercises

1. **Hand-write a loop in LLVM IR.** Extend `simple.ll` with a function
   `@sum_to(i64 %n)` that returns `0 + 1 + … + (n-1)`, using `phi` nodes
   yourself. Use `build/example.ll` as a crib, compile it with the
   `01-llvm-modules` steps, and call it from Python with
   `argtypes = [ctypes.c_int64]` and `restype = ctypes.c_int64`.
2. **Parameterize `loop_add`.** Change `example.mlir` so `@loop_add` takes
   `%ub: index` as an argument. Rebuild, then call it from Python through the
   dylib with `ctypes.c_int64` arguments. Look at the `-O2` assembly
   (`example.O2.s`): what can the optimizer still fold, and what not?
3. **Break the pass order.** Run the step 1 pipeline with
   `--convert-cf-to-llvm` *before* `--convert-scf-to-cf`, and feed the result
   to `mlir-translate`. Which ops survive, and what is the error? Then use
   `--mlir-print-ir-after-all` to see which pass did nothing.
4. **Count the casts.** Run the pipeline without `--convert-index-to-llvm` and
   without `--reconcile-unrealized-casts`, and count the
   `unrealized_conversion_cast` ops. Add `--reconcile-unrealized-casts` back:
   which casts disappear, and why do the `index.constant` ones remain?
5. **Try the umbrella pass.** Replace the six conversion passes with
   `--convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts` and
   diff the result against `build/example_opt.mlir`.
6. **Extend the dialect tour.** Add `@hypot(%x: f32, %y: f32) -> f32` using
   `math.sqrt`, and a function using `arith.divsi` vs `arith.divui` on `-7`
   and `2` (as `i32`). Predict both results before calling them from
   `driver.c`.

## Notes on reproducing with current MLIR

The source chapter targeted Homebrew LLVM 20.1.8. With the wheel (LLVM 24) and
Homebrew clang 20:

- **Unchanged:** every pass name, the lowered llvm-dialect output and the
  translated LLVM IR of `example.mlir` are identical to the LLVM 20 output.
- **No `llc`:** `clang -x ir -c` / `-S` replaces it. Plain `clang -O2` also runs
  the IR optimizer and folds the loop to `mov w0, #45`. Add `-Xclang
  -disable-llvm-passes` to get `llc`'s back-end-only `-O2`, which reproduces
  the source's ARM64 assembly instruction for instruction.
- **No `mlir-runner`:** JIT through the Python bindings' `ExecutionEngine`
  instead (`jit.py`). Runtime libraries go in `shared_libs=[...]` instead of
  `-shared-libs=`.
- **`.dylib`, not `.so`:** the source named its macOS shared libraries `.so`.
  That works with `ctypes`, but the scripts here use the platform suffix.
- **Version skew between `mlir-translate` (LLVM 24) and clang (20):** float
  literals need `tools/ll_fp_compat.py`. Intrinsic declarations, such as
  `llvm.sin` from `math.sin`, carry the attribute `nocreateundeforpoison`,
  which clang 20 rejects with `error: unterminated attribute group`.
  `ll_fp_compat.py` strips that attribute too.
- **Print-IR flags:** `--mlir-print-ir-after-change` only takes effect
  together with `--mlir-print-ir-after-all`.
- **Upstream dialects:** the wheel lists 48. Compared with LLVM 20, `mesh` is
  now `shard`, `x86vector` and `amx` are now `x86`, `polynomial` is gone, and
  `smt`, `wasmssa` and `xevm` are new.

## Where to go next

Every value in this chapter was a scalar in a register. Real programs pass
arrays, and an array has to live somewhere in memory.
[Chapter 4: Memory: Tensors, Memrefs and the C ABI](../ch04-memory-and-the-c-abi/README.md)
follows the *data* down the staircase: from `tensor` values with no address,
through `memref` buffers that NumPy can share without copying, down to LLVM
structs, SIMD vectors and the C calling convention.
