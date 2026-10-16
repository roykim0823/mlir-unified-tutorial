# Chapter 7: Parallel Loops and OpenMP

> Adapted from mlir-ir-lowering's `3_parallel` chapter (steps `1_affine_matmul`
> and `5_openmp`), itself based on Stephen Diehl's "MLIR Part 3 — Affine
> Dialect and OpenMP" article.

## What you will learn

- Why a sequential loop says more than it means, and how **dependences**
  (read-after-write, write-after-read, reductions) decide which loops may run
  in parallel.
- How `--affine-parallelize` proves loops independent and rewrites them into
  `affine.parallel`, including parallel **reductions**.
- How `-lower-affine` keeps that parallelism as `scf.parallel`, and why the
  default CPU pipeline then quietly throws it away.
- The directive-based OpenMP model in C (`#pragma omp parallel for`) and its
  op-for-op mirror in MLIR, the **`omp` dialect**.
- The route to real threads: `scf.parallel` → `--convert-scf-to-openmp` →
  `--convert-openmp-to-llvm` → `mlir-translate`, and which `__kmpc_*` OpenMP
  runtime calls come out the other end.
- How to link against `libomp` on macOS and Linux, run the kernels, check the
  results, and see which thread ran which iteration.

## Prerequisites

- Chapter 4: the memref descriptor and the `_mlir_ciface_*` C interface. Every
  C driver in this chapter passes memrefs as pointers to descriptor structs.
- Chapters 5 and 6: the affine dialect, affine maps, the polyhedral view of
  loops, and the sequential affine matmul (Chapter 6, example
  `01-affine-matmul`, compiles and runs it). This chapter picks up that same
  kernel and parallelizes it.
- An OpenMP runtime. The MLIR wheel does not ship one:
  - macOS: `brew install libomp`
  - Debian/Ubuntu: `apt install libomp-dev` (Fedora: `dnf install libomp-devel`)

  The example scripts find it through
  [`examples/find_libomp.sh`](examples/find_libomp.sh) (details in Step 7).

## A loop says more than it means

Write a `for` loop and you have, without meaning to, made a promise:
*iteration 0 happens, then 1, then 2, …*, strictly in order. But look at a
matrix multiply, blurring an image, or scaling a vector: the iterations do not
actually *depend* on each other. Computing `C[0][0]` does not need `C[0][1]`
to have happened first. The sequential order is an **accident of syntax**, not
a requirement of the math.

That gap is where all parallel performance lives. A modern CPU has many cores,
a GPU has thousands; the whole game is to notice which work is *independent*
and let it run at the same time. Earlier chapters walked computation *down*
the staircase (loops → branches → registers) and asked *where data lives*.
This chapter is about a third question: **when does work run, and what is
allowed to run at once?**

MLIR gives you two routes to an answer, and this chapter walks both:

- **Let the compiler find the parallelism**: the `affine` dialect. Because
  affine loops and accesses are restricted to affine expressions of the loop
  indices, the compiler can *prove* that iterations are independent and
  rewrite the nest automatically.
- **State the parallelism yourself**: the `omp` (OpenMP) dialect. Sometimes
  you would rather just say "run these iterations on a team of threads", the
  directive-based model familiar from `#pragma omp parallel for` in C. MLIR has
  it as first-class ops.

Both meet in the same place: `scf.parallel`, lowered to the OpenMP runtime.

### Dependences, in one paragraph

Chapter 6 introduced the polyhedral model: each execution of a loop body is a
point in an iteration domain, each point touches memory through an access
relation, and a schedule orders the points. A *valid* reordering is one that
preserves every **dependence**: a pair of points where one writes a location
the other reads or writes. Running a loop's iterations in parallel is the most
aggressive reordering there is (any order, or all at once), so it is legal
exactly when **no dependence is carried from one iteration of that loop to
another**. There are three flavours to watch for:

| Dependence | Pattern | Example |
| --- | --- | --- |
| read-after-write (true) | a later iteration reads what an earlier one wrote | `A[i] += A[i-1]` |
| write-after-read (anti) | a later iteration overwrites what an earlier one still had to read | `A[i] = A[i+1]` in place |
| write-after-write (output) | two iterations write the same location | `C[i][j] += …` across the `k` loop |

A **reduction** is a special loop-carried dependence: every iteration feeds
one accumulator (`sum += A[i]`). It is a dependence, but if the combining
operation is associative, the iterations can still be split among threads and
the partial results combined at the end.

Because affine accesses are affine functions of the loop indices, MLIR's
dependence analysis can decide these questions *exactly*. A general `scf.for`
with arbitrary index arithmetic, or C code with pointers, gives it nothing to
reason about.

## Running the examples

Each example directory has a `build.sh` that runs from any directory and puts
its outputs in a `build/` subdirectory. All listings and outputs below come
from running them with the MLIR wheel (LLVM 24), Homebrew clang 20 and
Homebrew libomp on a 12-core Apple Silicon Mac.

| Example | What it shows | Needs libomp |
| --- | --- | --- |
| [`01-affine-parallelize`](examples/01-affine-parallelize/) | dependence analysis, `--affine-parallelize`, `affine.parallel` → `scf.parallel`, serialization, GPU preview | no |
| [`02-c-openmp`](examples/02-c-openmp/) | the C `#pragma omp` reference program | yes |
| [`03-omp-dialect`](examples/03-omp-dialect/) | a hand-written `omp` dialect program lowered to a native executable | yes |
| [`04-scf-to-openmp`](examples/04-scf-to-openmp/) | `scf.parallel` → `omp`, called from C, thread ownership per iteration | yes |
| [`05-parallel-matmul`](examples/05-parallel-matmul/) | the affine matmul, auto-parallelized and run on all cores | yes |
| [`06-parallel-reduction`](examples/06-parallel-reduction/) | sum and max reductions through OpenMP | yes |

```bash
for d in part1-using-mlir/ch07-parallel-loops-and-openmp/examples/0*/; do
    bash "$d"build.sh
done
```

The lit tests `test/part1/ch07_*.mlir` check the key IR at each stage and
that the parallel matmul compiles to an object file with the expected runtime
references (`.venv/bin/lit build/test -v --filter ch07_`). The wheel ships no
`libomp`, so `ch07_parallel_matmul_run.mlir`, which links and runs the
parallel matmul on 4 threads, carries `REQUIRES: libomp`. `lit.cfg.py`
enables that feature when it finds Homebrew's `libomp` or a system one, and
reports the test as unsupported otherwise. The `build.sh` scripts run every
example.

## Step 1: `--affine-parallelize`

[`examples/01-affine-parallelize/deps.mlir`](examples/01-affine-parallelize/deps.mlir)
holds six small affine loops, one per situation from the table above. Running
`--affine-parallelize` on it, `build.sh` prints which loops came out parallel:

```text
=== loop kinds after --affine-parallelize (parallel-reductions=true) ===
  @scale                 affine.parallel
  @prefix_sum            affine.for
  @shift_out_of_place    affine.parallel
  @shift_in_place        affine.for
  @gather                affine.for
  @sum                   affine.parallel
```

- `@scale` (`B[i] = 2*A[i]`): each iteration touches only its own elements.
  Parallel.
- `@prefix_sum` (`A[i] += A[i-1]`): a true dependence from `i-1` to `i`.
  Stays sequential.
- `@shift_out_of_place` (`B[i] = A[i+1]`): the offset read is harmless because
  reads and writes go to different buffers. Parallel.
- `@shift_in_place` (`A[i] = A[i+1]`): the same shift in place has an anti
  dependence. Iteration `i` must read `A[i+1]` before iteration `i+1`
  overwrites it. Sequential.
- `@gather` (`B[i] = A[Idx[i]]`): a data-dependent index cannot be an
  `affine.load`, so it is a plain `memref.load`. This loop only *reads* `A`
  and would be safe, but the analysis cannot see through a non-affine access
  and stays conservative. (Any `memref.load` in the body has the same effect,
  even `A[%i]`.)
- `@sum`: a reduction through `iter_args`. Parallel only when you ask for it
  (Step 2).

The two parallel loops look like this (`build/deps_parallel.mlir`):

```mlir
  func.func @scale(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    %cst = arith.constant 2.000000e+00 : f32
    affine.parallel (%arg2) = (0) to (100) {
      %0 = affine.load %arg0[%arg2] : memref<100xf32>
      %1 = arith.mulf %0, %cst : f32
      affine.store %1, %arg1[%arg2] : memref<100xf32>
    }
    return
  }
  func.func @prefix_sum(%arg0: memref<100xf32>) {
    affine.for %arg1 = 1 to 100 {
      %0 = affine.load %arg0[%arg1 - 1] : memref<100xf32>
      %1 = affine.load %arg0[%arg1] : memref<100xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg0[%arg1] : memref<100xf32>
    }
    return
  }
```

`affine.parallel` says "these iterations may run in any order, or all at
once."

### The matmul

Now the real kernel, the classic product $C_{ij} = \sum_k A_{ik} B_{kj}$ as a
three-deep affine nest
([`matmul.mlir`](examples/01-affine-parallelize/matmul.mlir), the same kernel
Chapter 6 compiles and runs sequentially):

```mlir
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%i, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %j] : memref<?x?xf32>
        %c = affine.load %C[%i, %j] : memref<?x?xf32>
        %prod = arith.mulf %a, %b : f32
        %sum  = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
```

The `%i` and `%j` loops are independent: each `(i, j)` writes a distinct
`C[i, j]`. The `%k` loop is not: every `k` reads and writes the *same*
`C[i, j]`. The analysis sees exactly that (`build/matmul_parallel.mlir`):

```mlir
    affine.parallel (%arg3) = (0) to (symbol(%dim)) {          // i: parallel
      affine.parallel (%arg4) = (0) to (symbol(%dim_0)) {      // j: parallel
        affine.for %arg5 = 0 to %dim_1 {                       // k: stays sequential
          %0 = affine.load %arg0[%arg3, %arg5] : memref<?x?xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<?x?xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<?x?xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<?x?xf32>
        }
      }
    }
```

Two things to notice:

- **Two nested `affine.parallel`s, not one 2-D band.** The op *can* express a
  hyper-rectangular band such as `affine.parallel (%i, %j) = (0, 0) to (%M, %N)`
  (one op, two induction variables), but `--affine-parallelize` converts each
  loop in place ("Convert affine.for ops into 1-D affine.parallel", says its
  help text), so you get an outer parallel loop wrapping an inner one. Both
  forms mean the same thing. Its `max-nested` option limits how many nested
  levels it parallelizes: `--affine-parallelize=max-nested=1` makes only the
  `i` loop parallel (`build/matmul_parallel_outer.mlir`). One level is usually
  all a CPU needs; nested parallel regions mostly add overhead.
- **No `affine.yield` here.** This `affine.parallel` returns *no results*:
  each iteration accumulates straight into the `C` buffer with
  `affine.store`, so there is nothing to combine across iterations.
  `affine.yield` only appears when a parallel loop itself *produces* a value,
  i.e. a true parallel reduction (next step). The `k` loop reduces too, but
  through memory, which is why it has no yield either.

### The convolution

[Chapter 6](../ch06-affine-maps-and-loop-transforms/README.md)'s conv2d
(`examples/05-conv2d/conv2d.mlir` there) accumulates in *registers*: the two
filter loops carry the running sum as `iter_args`, and only the final value
is stored. `build.sh` runs `--affine-parallelize` on it too
(`build/conv2d_parallel.mlir`):

```mlir
affine.parallel (%arg3) = (0) to (symbol(%dim)) {
  affine.parallel (%arg4) = (0) to (symbol(%dim_0)) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.for %arg5 = 0 to %dim_1 iter_args(%arg6 = %cst) -> (f32) {
      %1 = affine.for %arg7 = 0 to %dim_2 iter_args(%arg8 = %arg6) -> (f32) {
        %2 = affine.load %arg1[%arg5, %arg7] : memref<?x?xf32>
        %3 = affine.load %arg0[%arg3 + %arg5, %arg4 + %arg7] : memref<?x?xf32>
        %4 = arith.mulf %3, %2 : f32
        %5 = arith.addf %arg8, %4 : f32
        affine.yield %5 : f32
      }
      affine.yield %1 : f32
    }
    affine.store %0, %arg2[%arg3, %arg4] : memref<?x?xf32>
  }
}
```

The output loops are independent (each writes its own `out[i, j]`), so
they become `affine.parallel`. The filter loops stay sequential: their
`iter_args` chain is a loop-carried dependence, and by default the pass
doesn't turn reductions into parallel loops (Step 2 shows the
`parallel-reductions` option that does). This is the shape you want for a
CPU: a parallel band on the outside and a tight sequential reduction
inside each thread. `--lower-affine --canonicalize` keeps it, merging the
two output loops into one 2-D `scf.parallel`
(`build/conv2d_scf_parallel.mlir`). Exercise 6 takes it all the way to
OpenMP.

## Step 2: Reductions in `affine.parallel`

By default `@sum` stays sequential: its `iter_args` value is a loop-carried
dependence. The `parallel-reductions=true` option tells the pass to recognize
reductions whose combiner it knows (`addf`, `mulf`, `maximumf`, `addi`, …) and
parallelize them anyway (`build/deps_parallel_reductions.mlir`):

```mlir
  func.func @sum(%arg0: memref<100xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.parallel (%arg1) = (0) to (100) reduce ("addf") -> (f32) {
      %2 = affine.load %arg0[%arg1] : memref<100xf32>
      affine.yield %2 : f32
    }
    %1 = arith.addf %cst, %0 : f32
    return %1 : f32
  }
```

Each iteration now just *yields its element*; the `reduce ("addf")` clause
says how to combine the yielded values, in any order. The original initial
value (`%cst`) is folded in once, after the loop. Note the opt-in: for floats,
reassociating a sum changes the rounding, which is why this is not on by
default.

## Step 3: Keeping the parallelism — `affine.parallel` → `scf.parallel`

`-lower-affine` turns `affine.parallel` into `scf.parallel`, the
dialect-neutral "parallel loop" that every backend understands
(`build/matmul_scf_parallel.mlir`, after `--lower-affine --canonicalize`):

```mlir
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%dim, %dim_0) step (%c1, %c1) {
      scf.for %arg5 = %c0 to %dim_1 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg5] : memref<?x?xf32>
        %1 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf32>
        %2 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        memref.store %4, %arg2[%arg3, %arg4] : memref<?x?xf32>
      }
      scf.reduce
    }
```

`-lower-affine` itself produces two nested 1-D `scf.parallel`s;
`--canonicalize` merges perfectly nested parallel loops into the single 2-D
band above. The body ends in `scf.reduce`, the terminator of every
`scf.parallel` (empty here, since nothing is reduced). The `@sum` reduction
lowers to the value-carrying form:

```mlir
    %0 = scf.parallel (%arg1) = (%c0) to (%c100) step (%c1) init (%cst) -> f32 {
      %2 = memref.load %arg0[%arg1] : memref<100xf32>
      scf.reduce(%2 : f32) {
      ^bb0(%arg2: f32, %arg3: f32):
        %3 = arith.addf %arg2, %arg3 : f32
        scf.reduce.return %3 : f32
      }
    }
```

Here the combiner is an explicit region: given two partial results, produce
one.

### The trap: the default pipeline serializes it

The CPU pipeline from Chapters 2–6 runs `--convert-scf-to-cf` next. That pass
happily accepts `scf.parallel`, and lowers it **into ordinary sequential
branches**. `build.sh` counts the parallel ops at each stage:

```text
=== parallel ops per stage (matmul) ===
  matmul_parallel          affine.parallel=2 scf.parallel=0 cf.cond_br=0
  matmul_parallel_outer    affine.parallel=1 scf.parallel=0 cf.cond_br=0
  matmul_scf_parallel      affine.parallel=0 scf.parallel=1 cf.cond_br=0
  matmul_serialized        affine.parallel=0 scf.parallel=0 cf.cond_br=3
```

After `--convert-scf-to-cf` the parallel band is just three nested
compare-and-branch loops, identical to what the sequential kernel produces.
That is why Chapter 6's affine matmul runs single-threaded even if you add
`--affine-parallelize`: the parallelism is real in the IR and then discarded.
To get parallel execution you must hand `scf.parallel` to a lowering that
*keeps* it: `--convert-scf-to-openmp` for CPU threads (the rest of this
chapter) or the GPU passes (Chapter 31).

## Step 4: OpenMP in C — the reference

The affine dialect had the *compiler* discover parallelism. OpenMP is the
opposite instinct: sometimes you want **direct, explicit control** over how
work is parallelized. OpenMP is a **directive-based** model — you annotate
ordinary loops with pragmas and a runtime spreads the iterations across a team
of threads — and it is a mainstay of high-performance computing.

Its native form is C. This kernel
([`02-c-openmp/omp_double.c`](examples/02-c-openmp/omp_double.c)) doubles a
10-element array: `#pragma omp parallel` opens a team of threads, and
`#pragma omp for` shares the loop's iterations among them. We also record
`omp_get_thread_num()` per iteration to see the split:

```c
void kernel(const float *input, float *output) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      output[i] = input[i] * 2.0f;
      owner[i] = omp_get_thread_num();
    }
  }
}
```

`build.sh` compiles it with `clang -fopenmp` against libomp and runs it with
`OMP_NUM_THREADS=4` and `=1`:

```text
=== OpenMP runtime calls clang emitted ===
@__kmpc_barrier
@__kmpc_for_static_fini
@__kmpc_for_static_init_4
@__kmpc_fork_call
=== run with 4 threads ===
2.000000 4.000000 6.000000 8.000000 10.000000 12.000000 14.000000 16.000000 18.000000 20.000000
threads available: 4
iteration -> thread: 0 0 0 1 1 1 2 2 3 3
=== run with 1 thread ===
2.000000 4.000000 6.000000 8.000000 10.000000 12.000000 14.000000 16.000000 18.000000 20.000000
threads available: 1
iteration -> thread: 0 0 0 0 0 0 0 0 0 0
```

The default *static schedule* cuts the 10 iterations into contiguous chunks,
one per thread (3, 3, 2, 2). Keep the four `__kmpc_*` names in mind: the MLIR
route below produces the same calls. Clang's pragmas are a frontend feature;
MLIR exposes the *same* primitives as ops in the `omp` dialect.

## Step 5: The `omp` dialect by hand

| Op | Meaning |
| --- | --- |
| `omp.parallel` | Fork a team of threads; its region runs on each (`#pragma omp parallel`) |
| `omp.wsloop` + `omp.loop_nest` | A worksharing loop: split the iterations across the team (`#pragma omp for`). `omp.wsloop` carries the scheduling/reduction clauses, the `omp.loop_nest` inside it is the loop itself |
| `omp.barrier` | Make all threads wait until everyone arrives |
| `omp.terminator` | Closes an `omp.parallel` (and similar) region |
| `omp.yield` | Closes the `omp.loop_nest` body: the end of *one loop iteration* |

MLIR regions must end in a terminator op (a block cannot just "fall
through"), so `omp.yield` and `omp.terminator` are bookkeeping, not
computation. The distinction is which region they close: `omp.yield` ends the
body of the `omp.loop_nest` — "this iteration is done" — analogous to
`scf.yield`/`affine.yield` (here it carries no value, since the loop just
writes to memory); `omp.terminator` ends the outer `omp.parallel` region.

[`03-omp-dialect/omp_double.mlir`](examples/03-omp-dialect/omp_double.mlir)
is the C program above, written in MLIR. The parallel region:

```mlir
func.func private @kernel(%input: memref<10xf32>, %output: memref<10xf32>) {
  %ub   = llvm.mlir.constant(9 : i32) : i32     // inclusive upper bound
  %lb   = llvm.mlir.constant(0 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) inclusive step (%step) {
        %ix  = arith.index_cast %i : i32 to index
        %v   = memref.load %input[%ix] : memref<10xf32>
        %two = arith.constant 2.0 : f32
        %r   = arith.mulf %v, %two : f32
        memref.store %r, %output[%ix] : memref<10xf32>
        omp.yield
      }
    }
    omp.barrier
    omp.terminator
  }
  return
}
```

The loop uses `i32` bounds with an `inclusive` upper bound, `0` to `9`, the
same iteration space as C's `i < 10`. `main` then prints the output array.

> **Aside: variadic `printf`.** `main` prints via C's `printf`, declared as
> `llvm.func @printf(!llvm.ptr, ...) -> i32` and called with
> `llvm.call @printf(%fmt, %x) vararg(!llvm.func<i32 (ptr, ...)>)`. One gotcha
> is baked into the example: C varargs promote `float` to `double`, so each
> `f32` is `arith.extf`'d to `f64` before the call (the format string is
> `"%f\n"`).

## Step 6: From `omp` to LLVM IR and the OpenMP runtime

`03-omp-dialect/build.sh` lowers the program with the usual conversion passes
plus one new one:

```bash
mlir-opt omp_double.mlir \
    --finalize-memref-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-openmp-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o build/omp_double_llvm.mlir
mlir-translate build/omp_double_llvm.mlir --mlir-to-llvmir -o build/omp_double.ll
```

Two things are worth knowing about how the work is split between these tools:

1. **`--convert-openmp-to-llvm` does not remove the `omp` ops.** It converts
   the types and operations *inside* their regions to the llvm dialect. The
   llvm-dialect file still contains them:

   ```mlir
     llvm.func @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, ...) {
       ...
       omp.parallel {
         omp.wsloop {
           omp.loop_nest (%arg10) : i32 = (%13) to (%12) inclusive step (%14) {
           ...
         omp.barrier
         omp.terminator
   ```

2. **`mlir-translate` does the real OpenMP lowering.** Its OpenMP translation
   is built on LLVM's `OpenMPIRBuilder` — the same component clang uses for
   `#pragma omp` — which outlines the parallel region into its own function
   and emits the runtime calls.

The result:

```text
=== OpenMP runtime calls in the LLVM IR ===
@__kmpc_barrier
@__kmpc_for_static_fini
@__kmpc_for_static_init_4u
@__kmpc_fork_call
@__kmpc_global_thread_num
=== outlined parallel region ===
define void @kernel(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
define internal void @kernel..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
define i32 @main() {
```

`@kernel` itself shrank to packing its arguments into a struct and calling
the runtime:

```llvm
omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded7 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %.reloaded7, ptr %gep_.reloaded7, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @kernel..omp_par, ptr %structArg)
```

and the body moved into `@kernel..omp_par`, which every thread runs. Inside,
each thread asks the runtime for its share of the iteration space:

```llvm
omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i32 0, ptr %p.lowerbound, align 4
  store i32 9, ptr %p.upperbound, align 4
  store i32 1, ptr %p.stride, align 4
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_4u(ptr @2, i32 %omp_global_thread_num, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i32 1, i32 0)
  ...
omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @2, i32 %omp_global_thread_num)
  %omp_global_thread_num5 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @3, i32 %omp_global_thread_num5)
  ...
omp.region.cont3:                                 ; preds = %omp_loop.after
  %omp_global_thread_num6 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @4, i32 %omp_global_thread_num6)
```

What each runtime entry point does (these are libomp's `kmpc` interface; all
of this chapter's examples use only these):

| Call | Role |
| --- | --- |
| `__kmpc_fork_call(loc, nargs, fn, args...)` | Start a team of threads and run the outlined `fn` on each; returns when all are done (end of `omp.parallel`) |
| `__kmpc_global_thread_num(loc)` | The calling thread's runtime id |
| `__kmpc_for_static_init_{4,4u,8,8u}(...)` | Static worksharing: overwrite this thread's lower/upper bound with its chunk. The suffix is the induction variable type: 32/64-bit, signed or `u`nsigned. `34` is `kmp_sch_static`, the default static schedule |
| `__kmpc_for_static_fini(loc, tid)` | End of the worksharing loop |
| `__kmpc_barrier(loc, tid)` | Wait for the whole team |
| `__kmpc_reduce` / `__kmpc_end_reduce` | Combine per-thread partial results of a reduction (Step 10) |

There are **two barriers**: a worksharing loop ends with an *implicit* barrier
unless it has a `nowait` clause (C's `#pragma omp for` has one too — that is
the `__kmpc_barrier` in clang's output), and our explicit `omp.barrier` adds a
second, redundant one. The MLIR loop uses `_4u` (i32 bounds, emitted as
unsigned); clang uses `_4` for the `int i`; an `index`-typed loop gets `_8u`.

## Step 7: Linking against libomp

The object file has undefined references to `__kmpc_*`, so the final link
needs `-lomp`. `03-omp-dialect/build.sh` finishes like every native build in
this tutorial (LLVM 24 float syntax rewritten for clang 20, then `clang -x
ir`), and links:

```bash
python3 tools/ll_fp_compat.py < build/omp_double.ll > build/omp_double.compat.ll
clang -Wno-override-module -x ir -c build/omp_double.compat.ll -o build/omp_double.o
clang build/omp_double.o -L$(brew --prefix libomp)/lib -lomp \
    -Wl,-rpath,$(brew --prefix libomp)/lib -o build/omp_double
```

```text
=== running (input doubled, one value per line) ===
2.000000
4.000000
6.000000
8.000000
10.000000
12.000000
14.000000
16.000000
18.000000
20.000000
```

Same result as the C version. Where libomp lives depends on the platform;
[`examples/find_libomp.sh`](examples/find_libomp.sh), sourced by every
`build.sh` that links, sets `OMP_CFLAGS`/`OMP_LDFLAGS` accordingly:

- **macOS (Homebrew).** libomp is keg-only, so nothing is on the default
  search paths: `-I$(brew --prefix libomp)/include` to compile code that
  includes `omp.h`, `-L$(brew --prefix libomp)/lib -lomp` to link, and
  `-Wl,-rpath,$(brew --prefix libomp)/lib` to bake the location into the
  binary so it loads without `DYLD_LIBRARY_PATH`. Homebrew builds libomp for
  the newest macOS, so the linker warns `building for macOS-16.0, but linking
  with dylib ... which was built for newer version 26.0` (Homebrew clang 20
  predates the macOS 26 version numbering). The warning is harmless; the
  script passes `-Wl,-w` to silence it.
- **Linux.** Install `libomp-dev` (Debian/Ubuntu) or `libomp-devel`
  (Fedora); plain `-lomp` then works, and `clang -fopenmp` finds `omp.h` by
  itself. Note that GCC's `libgomp` is a different runtime that does not
  implement the `__kmpc_*` entry points MLIR emits. (The Linux path is the
  standard package layout; this chapter was verified on macOS.)
- **Anywhere else**, set `LIBOMP_PREFIX` to a directory with
  `lib/libomp.{dylib,so}` and `include/omp.h`.

`-Wno-override-module` silences clang's note that it is replacing the
module's (empty) target triple with the host's.

## Step 8: `scf.parallel` → OpenMP, and who ran what

**You rarely write the `omp` dialect by hand.** The usual route is to produce
`scf.parallel` (by writing it, or via `--affine-parallelize` + `-lower-affine`)
and let `--convert-scf-to-openmp` rewrite it.
[`04-scf-to-openmp/scf_parallel.mlir`](examples/04-scf-to-openmp/scf_parallel.mlir)
has two plain parallel loops:

```mlir
func.func @add(%A: memref<100xf32>, %B: memref<100xf32>)
    attributes {llvm.emit_c_interface} {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c100 = arith.constant 100 : index
  scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
    %v = memref.load %A[%i] : memref<100xf32>
    %r = arith.addf %v, %v : f32
    memref.store %r, %B[%i] : memref<100xf32>
    scf.reduce
  }
  return
}

// The OpenMP runtime's own API is just C functions; declare one and call it.
func.func private @omp_get_thread_num() -> i32

func.func @which_thread(%owner: memref<?xi32>)
    attributes {llvm.emit_c_interface} {
  ...
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %t = func.call @omp_get_thread_num() : () -> i32
    memref.store %t, %owner[%i] : memref<?xi32>
    scf.reduce
  }
  return
}
```

`mlir-opt scf_parallel.mlir --convert-scf-to-openmp` turns each
`scf.parallel` into an `omp.parallel` region wrapping an
`omp.wsloop`/`omp.loop_nest`, with the bounds and step carried over
(`build/scf_parallel_omp.mlir`):

```mlir
  func.func @add(%arg0: memref<100xf32>, %arg1: memref<100xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg2) : index = (%c0) to (%c100) step (%c1) {
          memref.alloca_scope  {
            %1 = memref.load %arg0[%arg2] : memref<100xf32>
            %2 = arith.addf %1, %1 : f32
            memref.store %2, %arg1[%arg2] : memref<100xf32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
```

The loop body is wrapped in `memref.alloca_scope`, so any stack allocation in
an iteration is released at the end of that iteration instead of piling up
on the thread's stack. `--finalize-memref-to-llvm` lowers it to an
`llvm.intr.stacksave`/`llvm.intr.stackrestore` pair, and it must do so
**before** `--convert-scf-to-cf`: an `alloca_scope` region may only have one
block, and turning an `scf.for` inside it into branches (as happens in Step 9)
breaks that rule. That is why every pipeline in this chapter runs
`--finalize-memref-to-llvm` first.

`driver.c` calls both kernels through their `_mlir_ciface_*` wrappers
(Chapter 4's descriptor structs), and `build.sh` runs it with 4, 2 and 1
threads:

```text
=== OpenMP runtime calls in the LLVM IR ===
@__kmpc_barrier
@__kmpc_for_static_fini
@__kmpc_for_static_init_8u
@__kmpc_fork_call
@__kmpc_global_thread_num
=== OMP_NUM_THREADS=4 ===
add: ok (B[0]=0, B[99]=198)
threads available: 4
iteration -> thread: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
=== OMP_NUM_THREADS=2 ===
add: ok (B[0]=0, B[99]=198)
threads available: 2
iteration -> thread: 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
=== OMP_NUM_THREADS=1 ===
add: ok (B[0]=0, B[99]=198)
threads available: 1
iteration -> thread: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

The 16 iterations are split into equal contiguous chunks, exactly as clang's
static schedule did. `OMP_NUM_THREADS` is read by libomp at run time; without
it, the team uses every core. (`--convert-scf-to-openmp=num-threads=N` would
instead hard-code a `num_threads` clause into the IR.)

## Step 9: The whole road — a parallel matmul

[`05-parallel-matmul`](examples/05-parallel-matmul/) puts all of it together
on the affine matmul. One `mlir-opt` call goes from sequential affine loops to
OpenMP:

```bash
mlir-opt matmul.mlir \
    --affine-parallelize \
    --lower-affine \
    --canonicalize \
    --convert-scf-to-openmp \
    -o build/matmul_omp.mlir
```

```mlir
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg3, %arg4) : index = (%c0, %c0) to (%dim, %dim_0) step (%c1, %c1) collapse(2) {
          memref.alloca_scope  {
            scf.for %arg5 = %c0 to %dim_1 step %c1 {
              %1 = memref.load %arg0[%arg3, %arg5] : memref<?x?xf32>
              %2 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf32>
              %3 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
              %4 = arith.mulf %1, %2 : f32
              %5 = arith.addf %3, %4 : f32
              memref.store %5, %arg2[%arg3, %arg4] : memref<?x?xf32>
            }
          }
          omp.yield
        }
      }
      omp.terminator
    }
```

The 2-D `scf.parallel` band became **one** worksharing loop with
`collapse(2)`: the `(i, j)` space is flattened into a single iteration range
and divided among threads, like C's `#pragma omp parallel for collapse(2)`.
(Without `--canonicalize`, the two nested `scf.parallel`s become two nested
`omp.parallel` regions instead — try it in the exercises.)

`driver.c` fills two random 512x512 matrices, runs a plain C triple loop as
reference, calls `_mlir_ciface_matmul`, and times both with
`omp_get_wtime()`. `build.sh` runs it on 1 thread and on all cores:

```text
=== OpenMP runtime calls in the LLVM IR ===
@__kmpc_barrier
@__kmpc_for_static_fini
@__kmpc_for_static_init_8u
@__kmpc_fork_call
@__kmpc_global_thread_num
N=512 threads=1  C reference 0.120s  MLIR+OpenMP 0.107s  max abs error 2.86e-06
parallel matmul matches
N=512 threads=12  C reference 0.122s  MLIR+OpenMP 0.017s  max abs error 2.86e-06
parallel matmul matches
```

On one thread the MLIR kernel performs like the C loop (both compiled at
`-O2`); on 12 threads (6 performance + 6 efficiency cores) it runs about 6x
faster, with no change to the kernel source — the compiler proved the
parallelism and OpenMP delivered it. The tiny error is not a race: clang
contracts the C reference's `a*b + c` into fused multiply-adds, which round
once instead of twice. Building `driver.c` with `-ffp-contract=off` makes the
error exactly `0`.

## Step 10: Parallel reductions through OpenMP

[`06-parallel-reduction/reduce.mlir`](examples/06-parallel-reduction/reduce.mlir)
contains a sum written as a sequential `affine.for` with `iter_args` (it goes
through Steps 2 and 3 automatically) and a max written directly as an
`scf.parallel` reduction. `--convert-scf-to-openmp` turns each `scf.reduce`
combiner into an `omp.declare_reduction` and attaches it to the worksharing
loop (`build/reduce_omp.mlir`):

```mlir
  omp.declare_reduction @__scf_reduction : f32 init {
  ^bb0(%arg0: f32):
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    omp.yield(%0 : f32)
  } combiner {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    omp.yield(%0 : f32)
  } atomic {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg1 : !llvm.ptr -> f32
    %1 = llvm.atomicrmw fadd %arg0, %0 monotonic : !llvm.ptr, f32
    omp.yield
  }
  func.func @sum(%arg0: memref<?xf32>) -> f32 attributes {llvm.emit_c_interface} {
    ...
    %1 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    llvm.store %cst, %1 : f32, !llvm.ptr
    omp.parallel {
      omp.wsloop reduction(@__scf_reduction %1 -> %arg1 : !llvm.ptr) {
        omp.loop_nest (%arg2) : index = (%c0) to (%dim) step (%c1) {
          memref.alloca_scope  {
            %4 = memref.load %arg0[%arg2] : memref<?xf32>
            %5 = llvm.load %arg1 : !llvm.ptr -> f32
            %6 = arith.addf %5, %4 : f32
            llvm.store %6, %arg1 : f32, !llvm.ptr
          }
          omp.yield
        }
      }
      omp.terminator
    }
    %2 = llvm.load %1 : !llvm.ptr -> f32
    %3 = arith.addf %2, %cst : f32
    return %3 : f32
  }
```

Read it as OpenMP's `reduction(+:sum)` clause spelled out:

- The **`init`** region gives each thread a private accumulator, set to the
  identity (`0.0` for addition). Inside the loop, `%arg1` points at *this
  thread's* private copy, so the body updates it with no synchronization.
- The **`combiner`** region merges two partial results; at the end of the
  loop the runtime (`__kmpc_reduce`/`__kmpc_end_reduce`) combines the
  per-thread partials into the shared variable `%1`.
- The optional **`atomic`** region lets the runtime instead combine with an
  atomic read-modify-write; it is only emitted for combiners that map to one
  (`fadd` here, none for the max).

The max reduction uses `arith.cmpf ogt` + `arith.select` as its combiner.
That spelling is not a style choice: `--convert-scf-to-openmp` only
recognizes `addf`/`addi`, `mulf`/`muli`, `andi`/`ori`/`xori`, and min/max
written as compare + select. The `arith.maximumf` combiner that
`--affine-parallelize` produces for a max loop is left unconverted, and the
pass fails with `unconverted operation found` (`ch07_scf_to_openmp.mlir`
tests this).

`driver.c` sums `0, 1, …, 4095` with one element replaced by `99999`. Every
partial sum is an integer below 2^24, so float addition is exact in *any*
order and the parallel result must match exactly:

```text
=== OpenMP runtime calls in the LLVM IR ===
@__kmpc_barrier
@__kmpc_end_reduce
@__kmpc_for_static_fini
@__kmpc_for_static_init_8u
@__kmpc_fork_call
@__kmpc_global_thread_num
@__kmpc_reduce
threads=1  sum=8485325.0 (expect 8485325.0)  max=99999.0 (expect 99999.0)
reductions match
threads=4  sum=8485325.0 (expect 8485325.0)  max=99999.0 (expect 99999.0)
reductions match
```

With arbitrary floats the parallel sum would differ from the sequential one
in the last bits, and the difference would change with the thread count —
the price of the reassociation you opted into in Step 2.

## Two philosophies, one backend — and a preview of the GPU

Automatic (affine) and manual (OpenMP) parallelism both meet at
`scf.parallel` and lower through the same `llvm` machinery you already know.
And `scf.parallel` is not tied to CPU threads. A GPU kernel is, conceptually,
*one iteration of a parallel loop, run on one thread*. Hand the **same**
`matmul_scf_parallel.mlir` to the GPU passes instead of OpenMP
(`01-affine-parallelize/build.sh`, step 5):

```bash
mlir-opt build/matmul_scf_parallel.mlir \
    --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --canonicalize
```

```mlir
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %dim, %arg10 = %dim_0, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      scf.for %arg15 = %c0 to %dim_1 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg15] : memref<?x?xf32>
        ...
      }
      gpu.terminator
    }
```

The parallel `(i, j)` band became the GPU grid (one block per output element
in this naive mapping), and the sequential `k` loop is what each block runs.
One high-level program, two backends: Chapter 31 takes this route all the way
to NVIDIA hardware.

## Key takeaways

- **A loop over-specifies order.** Parallelization is recovering the
  independence the sequential syntax hid, and the affine dialect is what lets
  the compiler *see* that independence.
- **`affine.parallel` / `scf.parallel`** express independent iterations; their
  `reduce` forms express reductions, which you opt into for floats.
- **The default lowering serializes.** `--convert-scf-to-cf` turns
  `scf.parallel` into plain loops. Real parallelism needs a backend that keeps
  it: `--convert-scf-to-openmp` (CPU threads) or the GPU passes.
- **OpenMP is the explicit alternative.** `omp.parallel` + `omp.wsloop`
  declare parallelism directly, mirroring `#pragma omp parallel for`;
  `--convert-scf-to-openmp` gets you there from `scf.parallel` without writing
  the dialect by hand.
- **`mlir-translate` does the OpenMP heavy lifting**: it outlines regions and
  emits `__kmpc_*` calls, the same ones clang emits, so the final binary just
  links against `libomp`.

## Exercises

1. **Break a dependence.** In `deps.mlir`, rewrite `@shift_in_place` so it is
   parallel again without changing its result (hint: copy to a second buffer
   first — two loops, each of which is parallel). Confirm with
   `--affine-parallelize`.
2. **Interchange vs. parallelism.** Write a 2-D loop nest
   `A[i][j] = A[i-1][j] + 1`. Which of the two loops does
   `--affine-parallelize` make parallel? What changes if the dependence is
   `A[i][j-1]` instead? Relate the answer to the dependence direction.
3. **Nested teams.** Drop `--canonicalize` from the first `mlir-opt` command
   in `05-parallel-matmul/build.sh`. How does `build/matmul_omp.mlir` change?
   Run the binary: is it faster or slower than the collapsed version? Then try
   `--affine-parallelize=max-nested=1` instead and compare again.
4. **Pin the thread count in IR.** Add `--convert-scf-to-openmp=num-threads=2`
   in `04-scf-to-openmp/build.sh`. Which clause appears on `omp.parallel`,
   which extra runtime call (`grep __kmpc`) appears in the LLVM IR, and what
   does the iteration → thread line print now, even with
   `OMP_NUM_THREADS=4`?
5. **Make the max automatic.** Rewrite `@max` in `reduce.mlir` as an
   `affine.for` with `iter_args` and `arith.maximumf`, and run
   `--affine-parallelize=parallel-reductions=true`, then `--lower-affine`,
   then `--convert-scf-to-openmp`. Read the error. Can you get the automatic
   path to succeed with a compare + select combiner? (Look at which
   combiners `--affine-parallelize` recognizes.)
6. **Parallelize the convolution.** Take
   [The convolution](#the-convolution)'s `build/conv2d_parallel.mlir` through
   the Step 9 pipeline and call it from C, reusing Chapter 6's
   `conv2d_main.c` for the reference result. Then rerun with
   `--affine-parallelize=parallel-reductions=true`: which filter loop becomes
   an `affine.parallel` with a reduction, and would that help on a CPU
   compared with the outer parallel band?

## Notes on reproducing with current MLIR

The source chapter was written against Homebrew LLVM 20.1.8 with `llc`;
this port uses the MLIR wheel (LLVM 24) and Homebrew clang 20.

- **Pass names and behaviour are unchanged**: `--affine-parallelize` (with
  `max-nested` and `parallel-reductions`), `-lower-affine`,
  `--convert-scf-to-openmp`, `--convert-openmp-to-llvm`. The
  `--affine-parallelize` matmul output and the `scf_parallel.mlir` →
  `omp` output are identical to the LLVM 20 listings, including the
  `memref.alloca_scope` wrapper and the stray `llvm.mlir.constant(1 : i64)`
  `--convert-scf-to-openmp` leaves at the top of each function.
- **No `llc`.** Native code goes `mlir-translate` → `tools/ll_fp_compat.py` →
  `clang -x ir -c`, as everywhere in this tutorial.
- **Pass order.** The source pipeline ran `--finalize-memref-to-llvm` *after*
  `--convert-scf-to-cf`. That works for the hand-written `omp_double.mlir` and
  the flat `scf_parallel.mlir`, but fails on the matmul with
  `'memref.alloca_scope' op expects region #0 to have 0 or 1 blocks`, because
  the `scf.for` inside the scope has already been turned into branches. All
  pipelines here run `--finalize-memref-to-llvm` first.
- **`-inf` literals.** LLVM 24 prints a negative-infinity float constant as
  `float -inf`, which clang 20 rejects ("expected value token").
  `tools/ll_fp_compat.py` now rewrites it to hex, but the max reduction
  still starts from `-FLT_MAX` (`0xFF7FFFFF`), which works with any clang.
  Changing the init to `0xFF800000` (`-inf`) is a good exercise in checking
  that the filter handles it.
- **`maximumf` reductions** are produced by `--affine-parallelize` but not
  accepted by `--convert-scf-to-openmp` (Step 10).
- **libomp.** The source linked Homebrew's libomp with hard-coded
  `brew --prefix libomp`; here `examples/find_libomp.sh` also handles Linux
  and a `LIBOMP_PREFIX` override, and silences the macOS
  "built for newer version" linker warning.
- **Additions over the source:** the dependence examples (`deps.mlir`), the
  reduction examples and `omp.declare_reduction`, running the parallelized
  matmul (the source only inspected it and ran the sequential kernel), the
  thread-ownership kernels, the runtime-call walkthrough, and the GPU preview.

## Where to go next

- **[Chapter 8: Linalg by hand](../ch08-linalg-by-hand/)** — one level up:
  instead of writing loop nests and asking the compiler to find their
  parallelism, `linalg` ops declare it outright through their iterator types
  (`parallel` vs `reduction`), and lower to parallel loops directly.
- **[Chapter 31: GPU via NVVM](../../part6-end-to-end-ml/ch31-gpu-nvvm/)** —
  the other backend for `scf.parallel`, taken all the way to PTX.
