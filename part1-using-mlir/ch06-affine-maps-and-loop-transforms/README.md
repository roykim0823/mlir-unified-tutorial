# Chapter 6: Affine Maps and Loop Transformations

> Adapted from mlir-ir-lowering's `3_parallel` chapter (steps 1–4), itself
> based on Stephen Diehl's "MLIR Part 3 — Affine Dialect and OpenMP" article.
> The parallelization half of that chapter (`--affine-parallelize`,
> `scf.parallel`, the `omp` dialect) is
> [Chapter 7](../ch07-parallel-loops-and-openmp/README.md).

## What you will learn

- The **polyhedral model**: why a loop nest whose bounds and subscripts are
  affine can be treated as a geometric object, and what that buys over `scf`.
- How to **write affine code by hand** — `affine.for`, `affine.load`,
  `affine.store` — and run a hand-written matmul from Python against NumPy.
- **Affine maps in depth**: dims vs. symbols, `floordiv`/`ceildiv`/`mod`,
  `affine.apply` folding and composition, maps as loop bounds (`min`/`max`),
  and what `-lower-affine` turns each of them into.
- **Every affine loop pass worth knowing** besides the fusion and tiling you
  met in Chapter 5: LICM, unrolling, unroll-and-jam, normalization, scalar
  replacement, coalescing and super-vectorization, each with before/after IR.
- How to do the transformations MLIR has **no pass** for — loop interchange,
  skewing, and the **wavefront** — by hand, and prove them correct by running
  them.
- A real kernel: **2-D convolution** with `iter_args` reductions, optimized
  with affine passes and checked against NumPy.

## Prerequisites

- [Chapter 2](../ch02-lowering-pipelines/README.md): the `scf` → `cf` → `llvm`
  pipeline and `scf.for`.
- [Chapter 3](../ch03-llvm-ir-and-native-code/README.md): getting from the
  `llvm` dialect to a native object with `mlir-translate` and `clang`
  (including the `ll_fp_compat.py` bridge for LLVM 24 output).
- [Chapter 4](../ch04-memory-and-the-c-abi/README.md): memref descriptors,
  `llvm.emit_c_interface` / `_mlir_ciface_*` wrappers, and the
  `tools/np_memref.py` helper this chapter uses to pass NumPy arrays.
- [Chapter 5](../ch05-affine-optimization/README.md): what makes a loop or an
  access affine, `--convert-linalg-to-affine-loops`, `--affine-loop-fusion`,
  `--affine-loop-tile`, and `--lower-affine`. This chapter builds on those and
  does not repeat them.
- The repository's `.venv` with the MLIR wheel and NumPy (see the root
  README), and `clang` on your `PATH`.

## A loop says more than it means

Write a `for` loop and you have, without meaning to, made a promise:
iteration 0 happens, then 1, then 2, strictly in order. But look at a matrix
multiply, an image blur, or a vector scale: the iterations do not actually
*depend* on each other. Computing `C[0][0]` does not need `C[0][1]` to have
happened first. The sequential order is an accident of syntax, not a
requirement of the math. The loop says more than it means.

That gap is where loop optimization lives. Once the compiler knows which
orderings are *required* (the data dependences) and which are accidental, it
may reorder, block, flatten, unroll, vectorize — or run iterations at the same
time, which is Chapter 7's topic. The `affine` dialect is MLIR's tool for
making those dependences visible.

## The polyhedral model — loops as geometry

Why can a compiler reason about an `affine.for` nest but not a general
`scf.for` or C loop? Because affine constraints let it adopt the **polyhedral
model**: representing a loop nest as a geometric object.

Each execution of a statement inside a 2-D loop nest is a point `(i, j)`; the
set of all such points is a polygon (in higher dimensions, a polyhedron) carved
out by the loop bounds. Once your program is a shape, loop optimization becomes
geometry: parallelizing is slicing the shape into independent pieces, tiling
is gridding it into blocks, interchange is reflecting it, skewing is shearing
it. The model has three pieces:

- **Iteration domains** — the set of points a loop nest visits, bounded by
  affine inequalities (e.g. `0 ≤ i < M`).
- **Access relations** — which memory each point touches (e.g. point `(i, j)`
  reads `A[i, k]`).
- **Scheduling functions** — the order points execute in. A *valid*
  reordering is one that preserves every dependence.

MLIR builds three concepts on top of this:

| Concept | What it is | Example |
| --- | --- | --- |
| **Affine map** | A quasi-linear function from dims/symbols to results | `(d0, d1)[s0] -> (d0 + d1, s0 * d1)` |
| **Integer set** | Affine constraints carving out a domain | `(i)[N] : (i >= 0, N - i - 1 >= 0)` = `0 ≤ i < N` |
| **Affine ops** | `affine.for`, `affine.if`, `affine.parallel`, `affine.apply` | loop nests and index math |

The price of all this power is that affine expressions may use **only** `+`,
`-`, `*` by a constant, and `floordiv`/`ceildiv`/`mod` by a constant. No
data-dependent indices (`A[B[i]]`), no multiplying two loop variables. That
restriction is exactly what keeps the geometry tractable.

### `affine` vs `scf`

| | `scf` (Chapter 2) | `affine` (this chapter) |
| --- | --- | --- |
| Loop op | `scf.for` / `scf.parallel` | `affine.for` / `affine.parallel` |
| Index/access form | anything | affine expressions only |
| Memory ops | `memref.load`/`store` | `affine.load`/`store` (analyzable) |
| Compiler can tile / fuse / unroll-jam / vectorize / parallelize automatically? | no | **yes** |
| Lower it with | (already low) | `-lower-affine` → `scf` |

You write (or generate) high-level `affine`, let the optimizer transform it,
then `-lower-affine` drops it to `scf`, and from there it is the same pipeline
as every earlier chapter.

## Running the examples

Each example directory has a `build.sh` that works from any directory and
writes its outputs to `examples/<NN-topic>/build/`:

| Directory | Topic | Runs code? |
| --- | --- | --- |
| [`examples/01-affine-matmul/`](examples/01-affine-matmul/) | hand-written `affine.for` matmul, called from Python | yes |
| [`examples/02-affine-maps/`](examples/02-affine-maps/) | `affine_map`, `affine.apply`, folding, lowering | inspect |
| [`examples/03-affine-passes/`](examples/03-affine-passes/) | one file per affine optimization pass | inspect |
| [`examples/04-interchange-skewing/`](examples/04-interchange-skewing/) | manual interchange, skewing, wavefront + correctness check | yes |
| [`examples/05-conv2d/`](examples/05-conv2d/) | 2-D convolution, plain and optimized, vs NumPy | yes |

```bash
for d in part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/0*/; do
  bash "$d/build.sh"
done
```

The runnable examples share [`examples/common.sh`](examples/common.sh), whose
`lower_to_shared_lib` function is the whole CPU pipeline in one place:

```bash
mlir-opt kernel.mlir \
  -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
  -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm \
  -convert-func-to-llvm -reconcile-unrealized-casts -o kernel_llvm.mlir
mlir-translate --mlir-to-llvmir kernel_llvm.mlir -o kernel.ll
python3 tools/ll_fp_compat.py < kernel.ll > kernel.compat.ll
clang -shared -fPIC -O2 -Wno-override-module -x ir kernel.compat.ll -o libkernel.dylib
```

The only new step compared with Chapter 2 is the first pass,
`-lower-affine`. The Python drivers open the library with `ctypes` and pass
NumPy arrays through `tools/np_memref.py` (Chapter 4).

## Step 1 — A hand-written affine matmul

Chapter 5 got its affine loops from `--convert-linalg-to-affine-loops`. Here
we write them ourselves. The product $C_{ij} \mathrel{+}= \sum_k A_{ik}
B_{kj}$ as a three-deep affine nest
([`matmul.mlir`](examples/01-affine-matmul/matmul.mlir)):

```mlir
func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>   // rows of A (and C)
  %N = memref.dim %B, %c1 : memref<?x?xf32>   // cols of B (and C)
  %K = memref.dim %A, %c1 : memref<?x?xf32>   // shared/contraction dimension

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
  return
}
```

Unlike `scf.for`, `affine.for` is a "first-class loop": its bounds and the
memory accesses inside it are restricted to affine expressions of the loop
indices. `affine.load`/`affine.store` carry the same restriction, so the
dependence analysis can reason about them exactly. Two things to notice:

- The upper bounds `%M`, `%N`, `%K` are runtime values. They are legal because
  they are **symbols** (defined at the top of the function, never changed in
  the nest) — more on dims vs. symbols in Step 2. One compiled kernel handles
  any matrix size.
- `%i` and `%j` are independent across iterations: each `(i, j)` writes a
  distinct `C[i, j]`. Only `%k` carries a dependence (the accumulation into
  `C[i, j]`). That is precisely what `--affine-parallelize` proves in
  [Chapter 7](../ch07-parallel-loops-and-openmp/README.md).

Build and run it:

```bash
bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/01-affine-matmul/build.sh
```

```
Affine matmul successful! (max abs error 2.38e-07)
[[0.7453293  1.4563159  1.5619537 ]
 [0.22123039 0.9140139  0.15340269]
 [0.99109054 2.257992   1.7315115 ]
 [0.997431   2.1437762  1.5663465 ]]
```

The tiny error is ordinary float rounding: the kernel and NumPy sum in
different orders. The driver,
[`aot_main.py`](examples/01-affine-matmul/aot_main.py), zero-fills `C`
(the kernel accumulates into it), wraps the three arrays with
`numpy_to_memref`, and calls `_mlir_ciface_matmul` with a pointer to each
descriptor.

### What the lowering produces

`-lower-affine` alone (saved as `build/matmul_scf.mlir`) is unsurprising: each
`affine.for` becomes an `scf.for` with its bounds spelled out as
`arith.constant`s, and `affine.load`/`store` become `memref.load`/`store`:

```mlir
    scf.for %arg3 = %c0_2 to %dim step %c1_3 {
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg4 = %c0_4 to %dim_0 step %c1_5 {
        %c0_6 = arith.constant 0 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg5 = %c0_6 to %dim_1 step %c1_7 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<?x?xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
```

(The duplicated constants are what `-canonicalize` would clean up.) The full
lowering to the `llvm` dialect is left in `build/matmul_llvm.mlir`. It is
verbose, but every part of it is something Chapters 2–4 introduced, and it is
worth opening once to connect the dots:

**The signature explodes.** `@matmul` takes **21 scalar arguments**, not three
memrefs: `-finalize-memref-to-llvm` unrolls each `memref<?x?xf32>` into
`(allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1)`. The
body starts by `llvm.insertvalue`-ing those scalars back into descriptor
structs:

```mlir
llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, ..., %arg20: i64) attributes {llvm.emit_c_interface} {
  %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<...>   // C: allocated ptr
  %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<...>   // C: aligned ptr
  // ... [2] = offset, [3, 0..1] = sizes, [4, 0..1] = strides; then B, then A
```

**The `?` dimensions become runtime loads.** `memref.dim` is gone; each size
is pulled out of the descriptor's size array at run time:

```mlir
%27 = llvm.extractvalue %23[3] : !llvm.struct<...>            // A's [size0, size1]
%28 = llvm.alloca %26 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
llvm.store %27, %28 : !llvm.array<2 x i64>, !llvm.ptr
%29 = llvm.getelementptr %28[0, %24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i64>
%30 = llvm.load %29 : !llvm.ptr -> i64                         // %M
```

**The loops became blocks and branches.** No loop ops remain: the three
`affine.for`s are `^bb1`…`^bb9` wired with `llvm.cond_br`/`llvm.br` (the
`affine → scf → cf` lowering):

```mlir
  llvm.br ^bb1(%41 : i64)
^bb1(%43: i64):  // 2 preds: ^bb0, ^bb8          the i loop
  %44 = llvm.icmp "slt" %43, %30 : i64           // i < M ?
  llvm.cond_br %44, ^bb2, ^bb9
// ^bb3(%47) is the j loop, ^bb5(%51) the k loop, each the same shape
```

**`affine.load`/`store` became pointer math.** Each access computes the
row-major offset `i*stride0 + k`, a `getelementptr`, and a `load`/`store`;
`arith.mulf`/`addf` became `llvm.fmul`/`fadd`:

```mlir
%53 = llvm.extractvalue %23[1] : !llvm.struct<...>     // A aligned ptr
%54 = llvm.extractvalue %23[4, 0] : !llvm.struct<...>  // A stride0
%55 = llvm.mul %43, %54 overflow<nsw> : i64            // i * stride0
%56 = llvm.add %55, %51 overflow<nsw> : i64            // + k
%57 = llvm.getelementptr inbounds %53[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
%58 = llvm.load %57 : !llvm.ptr -> f32                 // A[i, k]
// ... likewise B[k, j] and C[i, j]; then:
%71 = llvm.fmul %58, %64 : f32                         // A[i,k] * B[k,j]
%72 = llvm.fadd %70, %71 : f32                         // C[i,j] + ...
```

**There are two functions.** The real `@matmul` (21 arguments) plus a
generated `@_mlir_ciface_matmul` that takes three descriptor *pointers*,
loads each struct, unpacks its fields and calls `@matmul` — the C-interface
wrapper from Chapter 4, and what `aot_main.py` actually calls:

```mlir
llvm.func @_mlir_ciface_matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
  %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %1 = llvm.extractvalue %0[0] : !llvm.struct<...>   // unpack A's 7 fields
  // ... unpack B and C too, then:
  llvm.call @matmul(%1, %2, %3, %4, %5, %6, %7, %9, ..., %23) : (...) -> ()
```

In short: affine loops → scalarized loops over raw pointers, ready for
`mlir-translate`. Nothing in this pipeline is parallel or vectorized; the
compiled kernel is a plain sequential triple loop.

## Step 2 — Affine maps and `affine.apply`

Affine maps are the dialect's workhorse: multi-dimensional affine functions
that spell out how loop indices map to the memory each iteration touches.
Once the compiler knows an access pattern as an affine map, it can reason
about layout and locality, which is what performance on CPUs and GPUs lives or
dies by. Every subscript you write in `affine.load %A[%i + %fi, %j]` is an
affine map in disguise, and every loop bound is one too.

An **affine map** is a function from *dimensions* and *symbols* to one or more
`index` results; `affine.apply` evaluates a single-result map.
[`affine_apply.mlir`](examples/02-affine-maps/affine_apply.mlir) works through
the forms you will meet:

```mlir
#tile_map    = affine_map<(d0) -> (d0 floordiv 32)>          // which 32-wide tile?
#offset_map  = affine_map<(d0)[s0] -> (d0 + s0)>             // shift by a symbol
#complex_map = affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 floordiv 4 + s0)>

func.func @affine_apply_examples() -> (index, index, index, index, index) {
  %c0   = arith.constant 0   : index
  %c42  = arith.constant 42  : index
  %c128 = arith.constant 128 : index

  // 1. Tiling: 42 floordiv 32 = 1 (element 42 lives in tile #1).
  %tile = affine.apply #tile_map(%c42)
  // 2. An inline (anonymous) map for an offset: 42 + 10 = 52.
  %off = affine.apply affine_map<(i) -> (i + 10)>(%c42)
  // 3. A symbol operand: 42 + 128 = 170. Dims go in (), symbols in [].
  %shift = affine.apply #offset_map(%c42)[%c128]
  // 4. Two dims and a symbol: 42*2 + 128 floordiv 4 + 0 = 116.
  %cplx = affine.apply #complex_map(%c42, %c128)[%c0]
  // 5. Composition: feed one apply's result into the next. (42*2) + 5 = 89.
  %t     = affine.apply affine_map<(i) -> (i * 2)>(%c42)
  %final = affine.apply affine_map<(i) -> (i + 5)>(%t)

  return %tile, %off, %shift, %cplx, %final : index, index, index, index, index
}
```

The rules that pin down what an affine map can be:

- **`(d0)[s0]` = one dimension `d0` plus one symbol `s0`.** Dims go in `()`,
  symbols in `[]`.
- **Only a fixed vocabulary is allowed:** `+`, `-`, multiplication by a
  constant, and `floordiv` / `ceildiv` / `mod` by a constant. No two variables
  multiplied, no data-dependent terms. That linearity is what makes the maps
  analyzable. (MLIR's parser also accepts *semi-affine* maps that multiply or
  divide by a **symbol**, such as `(d0)[s0] -> (d0 mod s0)`; you will see one
  come out of the coalescing pass in Step 3. They are fine to compute with,
  but the dependence analysis cannot reason about them until the symbol is
  folded to a constant.)
- **`floordiv` rounds toward −∞, `ceildiv` toward +∞, and `mod` is the
  matching non-negative remainder.** This differs from C's `/` and `%`, which
  truncate toward zero.
- **The result is always `index`.** Maps compute positions, never data.

### Dims vs. symbols

The split matters because it tells the analysis what can change where.
A **dimension** is a value that varies inside the region being analyzed —
typically a loop induction variable. A **symbol** is a value that is fixed for
the whole region — a function argument, a constant, a `memref.dim` of a
function argument, or an `affine.apply` of other symbols. The analysis can
treat symbols as unknown-but-constant parameters, which is why `affine.for %i
= 0 to %M` is a legal bound in Step 1 and the iteration domain `0 ≤ i < M` is
still a polyhedron.

Which values qualify is enforced by the verifier.
[`invalid_index.mlir`](examples/02-affine-maps/invalid_index.mlir) subscripts
a load with `arith.muli %i, %j` — the product of two loop variables — and
`mlir-opt` refuses it:

```
invalid_index.mlir:11:12: error: 'affine.load' op operand cannot be used as a dimension id
      %v = affine.load %A[%ij] : memref<16xf32>
           ^
```

### Folding and composition

`build.sh` round-trips the file through `mlir-opt` (into
`build/affine_apply_roundtrip.mlir`) and then runs `-canonicalize` (into
`build/affine_apply_folded.mlir`). The round trip shows that `mlir-opt`
hoists every map, named or inline, into a numbered alias and deduplicates
them — your `#tile_map` comes back as `#map`, and the `(i) -> (i * 2)` used in
two functions appears once:

```mlir
#map = affine_map<(d0) -> (d0 floordiv 32)>
#map1 = affine_map<(d0) -> (d0 + 10)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 floordiv 4 + s0)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0) -> (d0 + 5)>
#map6 = affine_map<(d0) -> (d0 floordiv 4)>
#map7 = affine_map<(d0) -> (d0 ceildiv 4)>
#map8 = affine_map<(d0) -> (d0 mod 4)>
```

After `-canonicalize` every apply on constants has been evaluated, confirming
the arithmetic in the comments — and the negative-number cases from the
`@rounding` function (`-7 floordiv 4`, `-7 ceildiv 4`, `-7 mod 4`):

```mlir
#map = affine_map<()[s0] -> (s0 * 2 + 5)>
module {
  func.func @affine_apply_examples() -> (index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c52 = arith.constant 52 : index
    %c170 = arith.constant 170 : index
    %c116 = arith.constant 116 : index
    %c89 = arith.constant 89 : index
    return %c1, %c52, %c170, %c116, %c89 : index, index, index, index, index
  }
  func.func @rounding() -> (index, index, index) {
    %c-2 = arith.constant -2 : index
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    return %c-2, %c-1, %c1 : index, index, index
  }
  func.func @compose(%arg0: index) -> index {
    %0 = affine.apply #map()[%arg0]
    return %0 : index
  }
  func.func @dead_apply() {
    return
  }
}
```

Three things happened:

- **Constant folding.** C would give `-7 / 4 = -1` and `-7 % 4 = -3`; affine
  semantics give `-2`, `-1` and `1`.
- **Composition.** In `@compose` the operand is a runtime value, so nothing
  folds — but the two chained applies `(i * 2)` then `(i + 5)` merged into one
  map, `s0 * 2 + 5`. Notice the canonicalizer also *promoted* the function
  argument from a dim to a symbol (`()[%arg0]`), since a function argument is
  a valid symbol. Composition is why chains of index computations stay cheap
  in the affine dialect: they collapse into a single map that the analysis
  sees whole.
- **Dead code elimination.** `affine.apply` has no side effects, so in
  `@dead_apply`, whose result is unused, it is simply deleted.

### Maps as loop bounds

[`tiled_loop.mlir`](examples/02-affine-maps/tiled_loop.mlir) shows the
everyday use of `affine.apply` — turning a `(tile, within-tile)` pair into a
flat array index inside a tiled loop nest:

```mlir
func.func @tiled_loop(%arg0: memref<256xf32>) {
  affine.for %i = 0 to 256 step 32 {       // outer: one iteration per tile
    affine.for %j = 0 to 32 {              // inner: elements within the tile
      %idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
      %val = memref.load %arg0[%idx] : memref<256xf32>
      %dbl = arith.addf %val, %val : f32
      memref.store %dbl, %arg0[%idx] : memref<256xf32>
    }
  }
  return
}
```

When the size is only known at run time and need not be a multiple of the
tile, the last tile is partial. Then the inner loop's bounds become affine maps
of the outer index (a dim) and the size (a symbol), and the upper bound takes
the `min` of the map's two results — `min(i + 32, n)`:

```mlir
#tile_lb = affine_map<(d0) -> (d0)>
#tile_ub = affine_map<(d0)[s0] -> (d0 + 32, s0)>
func.func @tiled_loop_dynamic(%arg0: memref<?xf32>, %n: index) {
  affine.for %i = 0 to %n step 32 {
    affine.for %ii = #tile_lb(%i) to min #tile_ub(%i)[%n] {
      %val = affine.load %arg0[%ii] : memref<?xf32>
      %dbl = arith.addf %val, %val : f32
      affine.store %dbl, %arg0[%ii] : memref<?xf32>
    }
  }
  return
}
```

This is exactly the shape `--affine-loop-tile` generates for dynamic sizes —
you will see it again in Step 5's convolution. (`max` works the same way for
lower bounds; Step 4's wavefront uses both.)

### What `-lower-affine` makes of maps

`build/tiled_loop_lowered.mlir` shows maps turning into ordinary index
arithmetic: `(d0 + d1)` becomes an `arith.addi`, and the `min` bound becomes an
`arith.minsi`:

```mlir
  func.func @tiled_loop_dynamic(%arg0: memref<?xf32>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    scf.for %arg2 = %c0 to %arg1 step %c32 {
      %c32_0 = arith.constant 32 : index
      %0 = arith.addi %arg2, %c32_0 : index
      %1 = arith.minsi %0, %arg1 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %arg2 to %1 step %c1 {
        %2 = memref.load %arg0[%arg3] : memref<?xf32>
        %3 = arith.addf %2, %2 : f32
        memref.store %3, %arg0[%arg3] : memref<?xf32>
      }
    }
    return
  }
```

`floordiv` is more expensive than it looks, because hardware signed division
(`arith.divsi`) truncates toward zero. `build/floordiv_lowered.mlir`
([`floordiv.mlir`](examples/02-affine-maps/floordiv.mlir) lowered) wraps the
divide in a sign fix-up — for negative `x` it computes `-1 - ((-1 - x) / 4)`:

```mlir
  func.func @floordiv4(%arg0: index) -> index {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %0 = arith.cmpi slt, %arg0, %c0 : index
    %1 = arith.subi %c-1, %arg0 : index
    %2 = arith.select %0, %1, %arg0 : index
    %3 = arith.divsi %2, %c4 : index
    %4 = arith.subi %c-1, %3 : index
    %5 = arith.select %0, %4, %3 : index
    return %5 : index
  }
```

So the pair of outputs shows both directions: `affine.apply` is *erasable*
when its result is unused or constant, and *lowers to ordinary index
arithmetic* when it is needed.

## Step 3 — The affine optimization passes

This is the payoff of first-class loops: because the compiler can treat a loop
nest as analyzable geometry, MLIR ships a catalog of passes that rewrite the
nest to run faster without changing what it computes. Each subsection pairs a
minimal source file from
[`examples/03-affine-passes/`](examples/03-affine-passes/) with the exact
command and the real output. `build.sh` applies every pass and writes
`build/<name>_after.mlir`, so you can also
`diff licm.mlir build/licm_after.mlir` yourself.

> The `module { … }` wrapper in these outputs is the implicit top-level module
> every `.mlir` file has; the sources omit it, but `mlir-opt` always prints it.

### Loop-invariant code motion

Code inside a loop that does not depend on the loop index is recomputed every
iteration for no reason. LICM lifts it out — and by clearing index-independent
work out of the body, it often exposes other optimizations too. Here
`%x = 42.0` is the same in every iteration
([`licm.mlir`](examples/03-affine-passes/licm.mlir)):

```mlir
func.func @licm(%A: memref<10xf32>, %B: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %x = arith.constant 42.0 : f32           // loop-invariant — gets hoisted
    %v = affine.load %A[%i] : memref<10xf32>
    %s = arith.addf %v, %x : f32
    affine.store %s, %B[%i] : memref<10xf32>
  }
  return
}
```

```bash
mlir-opt licm.mlir -affine-loop-invariant-code-motion
```

The constant now sits before the loop, computed once instead of ten times:

```mlir
module {
  func.func @licm(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %arg0[%arg2] : memref<10xf32>
      %1 = arith.addf %0, %cst : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

The affine version can also hoist `affine.load`s whose address is invariant,
when it can prove no store in the loop writes there. (The dialect-independent
`-loop-invariant-code-motion` handles side-effect-free ops in any loop.)

### Unrolling

Replicating the body N times cuts loop overhead (fewer bound checks and
branches) and hands the CPU more independent work to pipeline
([`unroll.mlir`](examples/03-affine-passes/unroll.mlir)):

```mlir
func.func @unroll(%A: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i] : memref<8xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<8xf32>
  }
  return
}
```

`affine-loop-unroll` runs on functions and has to be nested explicitly with
`--pass-pipeline` (the plain `-affine-loop-unroll` flag fails, see the notes at
the end):

```bash
mlir-opt unroll.mlir \
  --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))"
```

The loop now steps by 4 (2 trips instead of 8) and the body appears four
times, the copies indexing `i`, `i+1`, `i+2`, `i+3` via `affine.apply`:

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @unroll(%arg0: memref<8xf32>) {
    affine.for %arg1 = 0 to 8 step 4 {
      %0 = affine.load %arg0[%arg1] : memref<8xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg1] : memref<8xf32>
      %2 = affine.apply #map(%arg1)
      %3 = affine.load %arg0[%2] : memref<8xf32>
      %4 = arith.addf %3, %3 : f32
      affine.store %4, %arg0[%2] : memref<8xf32>
      %5 = affine.apply #map1(%arg1)
      %6 = affine.load %arg0[%5] : memref<8xf32>
      %7 = arith.addf %6, %6 : f32
      affine.store %7, %arg0[%5] : memref<8xf32>
      %8 = affine.apply #map2(%arg1)
      %9 = affine.load %arg0[%8] : memref<8xf32>
      %10 = arith.addf %9, %9 : f32
      affine.store %10, %arg0[%8] : memref<8xf32>
    }
    return
  }
```

When the trip count is not a multiple of the factor, the pass splits off a
**cleanup loop**. The same body over `0 to 10` (`@unroll_cleanup` in the same
file) becomes the unrolled loop over `0 to 8` followed by:

```mlir
    affine.for %arg1 = 8 to 10 {
      %0 = affine.load %arg0[%arg1] : memref<10xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg1] : memref<10xf32>
    }
```

With a dynamic trip count the split point itself is an affine map — Step 5
shows `(s0 floordiv 2) * 2`.

### Unroll-and-jam

`-affine-loop-unroll-jam` unrolls an **outer** loop and then fuses ("jams")
the resulting copies of the inner loop back into one. For a matrix-vector
product `y += A·x` ([`unroll_jam.mlir`](examples/03-affine-passes/unroll_jam.mlir)):

```mlir
func.func @unroll_jam(%A: memref<8x16xf32>, %x: memref<16xf32>, %y: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 16 {
      %a = affine.load %A[%i, %j] : memref<8x16xf32>
      %b = affine.load %x[%j] : memref<16xf32>
      %c = affine.load %y[%i] : memref<8xf32>
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      affine.store %s, %y[%i] : memref<8xf32>
    }
  }
  return
}
```

```bash
mlir-opt unroll_jam.mlir \
  --pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))"
```

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @unroll_jam(%arg0: memref<8x16xf32>, %arg1: memref<16xf32>, %arg2: memref<8xf32>) {
    affine.for %arg3 = 0 to 8 step 2 {
      affine.for %arg4 = 0 to 16 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<8x16xf32>
        %1 = affine.load %arg1[%arg4] : memref<16xf32>
        %2 = affine.load %arg2[%arg3] : memref<8xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %arg2[%arg3] : memref<8xf32>
        %5 = affine.apply #map(%arg3)
        %6 = affine.load %arg0[%5, %arg4] : memref<8x16xf32>
        %7 = affine.load %arg1[%arg4] : memref<16xf32>
        %8 = affine.load %arg2[%5] : memref<8xf32>
        %9 = arith.mulf %6, %7 : f32
        %10 = arith.addf %8, %9 : f32
        affine.store %10, %arg2[%5] : memref<8xf32>
      }
    }
    return
  }
}
```

Each inner iteration now works on rows `i` and `i+1`, so the two rows share
one pass over `x`. Note that `x[j]` is still loaded twice (`%1` and `%7`):
the store to `y` sits between them, and since `%x` and `%y` are both function
arguments the compiler cannot rule out that they alias. When `y` is a local
`memref.alloc` instead, `-affine-scalrep` (below) does forward `%1` into the
second row — try it as an exercise.

### Normalization

Rewrite a loop to start at 0 and step 1, folding the old bounds into the
index. Many passes assume normalized loops, so this is a common pre-pass
([`normalize.mlir`](examples/03-affine-passes/normalize.mlir)):

```mlir
func.func @normalize(%A: memref<100xf32>) {
  affine.for %i = 10 to 100 step 5 {
    %v = affine.load %A[%i] : memref<100xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<100xf32>
  }
  return
}
```

```bash
mlir-opt normalize.mlir -affine-loop-normalize
```

`10 to 100 step 5` has `(100 − 10) / 5 = 18` iterations, so it becomes
`0 to 18`, with the real index recovered as `i * 5 + 10`:

```mlir
#map = affine_map<(d0) -> (d0 * 5 + 10)>
module {
  func.func @normalize(%arg0: memref<100xf32>) {
    affine.for %arg1 = 0 to 18 {
      %0 = affine.apply #map(%arg1)
      %1 = affine.load %arg0[%0] : memref<100xf32>
      %2 = arith.addf %1, %1 : f32
      affine.store %2, %arg0[%0] : memref<100xf32>
    }
    return
  }
}
```

Chapter 5 used this pass for a different reason: after tiling it turns the
odd-looking tile bounds back into readable `0 to N` loops.

### Scalar replacement — `-affine-scalrep`

When a value is stored and then loaded straight back from the same place, the
load is redundant: the analysis forwards the stored value and deletes the
load. A store that is overwritten before anyone can read it is dead and goes
too ([`scalrep.mlir`](examples/03-affine-passes/scalrep.mlir)):

```mlir
func.func @scalrep(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %c = arith.constant 1.0 : f32
    affine.store %c, %A[%i] : memref<10xf32>         // store ...
    %v = affine.load %A[%i] : memref<10xf32>         // ... and load right back
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<10xf32>
  }
  return
}
```

```bash
mlir-opt scalrep.mlir -affine-scalrep
```

The load is gone — `%cst` is used directly — and the now-dead first store is
dropped, leaving one store per iteration:

```mlir
module {
  func.func @scalrep(%arg0: memref<10xf32>) {
    affine.for %arg1 = 0 to 10 {
      %cst = arith.constant 1.000000e+00 : f32
      %0 = arith.addf %cst, %cst : f32
      affine.store %0, %arg0[%arg1] : memref<10xf32>
    }
    return
  }
}
```

### Coalescing

The opposite of tiling: *flatten* a perfectly nested loop into one, recovering
the original indices with `floordiv`/`mod`. Handy when you want a single flat
iteration space — e.g. to map onto a 1-D thread grid
([`coalescing.mlir`](examples/03-affine-passes/coalescing.mlir)):

```mlir
func.func @coalescing(%A: memref<4x8xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %A[%i, %j] : memref<4x8xf32>
      %t = arith.addf %v, %v : f32
      affine.store %t, %A[%i, %j] : memref<4x8xf32>
    }
  }
  return
}
```

```bash
mlir-opt coalescing.mlir -affine-loop-coalescing
```

The 4×8 nest becomes one loop with `4 * 8` trips, with row and column rebuilt
inside via `floordiv` and `mod` (`#map4` and `#map3`). These are the
semi-affine maps mentioned in Step 2 — the divisor is a symbol:

```mlir
#map = affine_map<() -> (4)>
#map1 = affine_map<() -> (8)>
#map2 = affine_map<(d0)[s0] -> (d0 * s0)>
#map3 = affine_map<(d0)[s0] -> (d0 mod s0)>
#map4 = affine_map<(d0)[s0] -> (d0 floordiv s0)>
module {
  func.func @coalescing(%arg0: memref<4x8xf32>) {
    %0 = affine.apply #map()
    %1 = affine.apply #map1()
    %2 = affine.apply #map2(%0)[%1]
    affine.for %arg1 = 0 to %2 {
      %3 = affine.apply #map3(%arg1)[%1]
      %4 = affine.apply #map4(%arg1)[%1]
      %5 = affine.load %arg0[%4, %3] : memref<4x8xf32>
      %6 = arith.addf %5, %5 : f32
      affine.store %6, %arg0[%4, %3] : memref<4x8xf32>
    }
    return
  }
}
```

Run `-canonicalize` after it (`build/coalescing_canonical_after.mlir`) and the
folding and composition from Step 2 take over: the trip count becomes `32`,
the symbol becomes the constant `8`, and the maps are folded straight into the
subscripts:

```mlir
    affine.for %arg1 = 0 to 32 {
      %0 = affine.load %arg0[%arg1 floordiv 8, %arg1 mod 8] : memref<4x8xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg1 floordiv 8, %arg1 mod 8] : memref<4x8xf32>
    }
```

### Super-vectorization

The affine dialect's own route to SIMD: a scalar loop becomes a loop over
vectors ([`vectorize.mlir`](examples/03-affine-passes/vectorize.mlir)):

```mlir
func.func @vectorize(%A: memref<256xf32>, %B: memref<256xf32>) {
  affine.for %i = 0 to 256 {
    %a = affine.load %A[%i] : memref<256xf32>
    %b = affine.load %B[%i] : memref<256xf32>
    %s = arith.addf %a, %b : f32
    affine.store %s, %A[%i] : memref<256xf32>
  }
  return
}
```

```bash
mlir-opt vectorize.mlir -affine-super-vectorize="virtual-vector-size=8"
```

The loop now steps by 8, and the element loads/add/store become
`vector.transfer_read` / `arith.addf : vector<8xf32>` /
`vector.transfer_write` — 8 elements per iteration:

```mlir
module {
  func.func @vectorize(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
    affine.for %arg2 = 0 to 256 step 8 {
      %0 = ub.poison : f32
      %1 = vector.transfer_read %arg0[%arg2], %0 {in_bounds = [true]} : memref<256xf32>, vector<8xf32>
      %2 = ub.poison : f32
      %3 = vector.transfer_read %arg1[%arg2], %2 {in_bounds = [true]} : memref<256xf32>, vector<8xf32>
      %4 = arith.addf %1, %3 : vector<8xf32>
      vector.transfer_write %4, %arg0[%arg2] {in_bounds = [true]} : vector<8xf32>, memref<256xf32>
    }
    return
  }
}
```

The "virtual" in `virtual-vector-size` means the size need not match the
hardware: `vector<8xf32>` is later split or widened to the target's real
registers. `in_bounds = [true]` records that the analysis proved every
8-element read stays inside the 256-element buffer (256 is a multiple of 8),
so no masking is needed, and the `ub.poison` padding value is never used.

### Fusion and tiling, briefly

Chapter 5 already showed both on the matmul+ReLU kernel; the minimal versions
are here for completeness. **Fusion** merges two adjacent loops over the same
range — the first writes `B` from `A`, the second reads `B` to write `C` — so
each `B[i]` is consumed while it is still hot
([`fusion.mlir`](examples/03-affine-passes/fusion.mlir); this is the loop-level
cousin of the `linalg` fusion in
[Chapter 8](../ch08-linalg-by-hand/README.md)):

```bash
mlir-opt fusion.mlir -affine-loop-fusion
```

```mlir
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %arg0[%arg3] : memref<10xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg3] : memref<10xf32>
      %2 = affine.load %arg1[%arg3] : memref<10xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg2[%arg3] : memref<10xf32>
    }
```

Passes compose: the fused body now stores `B[i]` and immediately reloads it,
which is exactly what scalar replacement removes.
`-affine-loop-fusion -affine-scalrep` (`build/fusion_scalrep_after.mlir`)
uses `%1` directly. The store to `B` stays, because `B` is a function
argument the caller can observe:

```mlir
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg3] : memref<10xf32>
      %2 = arith.addf %1, %1 : f32
      affine.store %2, %arg2[%arg3] : memref<10xf32>
```

**Tiling** with a fixed size turns a 32×32 nest into four loops: outer *tile*
loops stepping by 8, inner *point* loops walking the 8 elements of each tile
([`tiling.mlir`](examples/03-affine-passes/tiling.mlir)):

```bash
mlir-opt tiling.mlir -affine-loop-tile="tile-size=8"
```

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 8)>
module {
  func.func @tiling(%arg0: memref<32x32xf32>) {
    affine.for %arg1 = 0 to 32 step 8 {
      affine.for %arg2 = 0 to 32 step 8 {
        affine.for %arg3 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
            %0 = affine.load %arg0[%arg3, %arg4] : memref<32x32xf32>
            %1 = arith.addf %0, %0 : f32
            affine.store %1, %arg0[%arg3, %arg4] : memref<32x32xf32>
          }
        }
      }
    }
    return
  }
}
```

32 is a multiple of 8, so no `min` bound is needed; compare with Step 2's
`@tiled_loop_dynamic`.

### The rest of the catalog

`-affine-parallelize` (turn provably independent loops into
`affine.parallel`) is covered in
[Chapter 7](../ch07-parallel-loops-and-openmp/README.md).
`-affine-pipeline-data-transfer` is specialized: it overlaps `affine.dma`
transfers with compute, so it only does anything on code that already uses
the DMA ops. The full list in the current `mlir-opt --help`:

| Pass | Transformation |
| --- | --- |
| `-affine-data-copy-generate` | Copy the data a loop nest touches into a faster memory space and back |
| `-affine-expand-index-ops` | Lower `affine.delinearize_index` / `linearize_index` to arithmetic |
| `-affine-fold-memref-alias-ops` | Fold `memref.subview` & co. into affine load/store maps |
| `-affine-loop-coalescing` | Flatten nested loops into one |
| `-affine-loop-fusion` | Merge loop nests for locality (Chapter 5) |
| `-affine-loop-invariant-code-motion` | Hoist code that does not depend on the loop index |
| `-affine-loop-normalize` | Normalize bounds/steps to start at 0, step 1 |
| `-affine-loop-tile` | Block loops into cache-friendly tiles (Chapter 5) |
| `-affine-loop-unroll` / `-affine-loop-unroll-jam` | Replicate the body (and jam the inner loop copies) |
| `-affine-parallelize` | Turn independent loops into `affine.parallel` (Chapter 7) |
| `-affine-pipeline-data-transfer` | Pipeline DMA between memory levels |
| `-affine-raise-from-memref` | Turn `memref.load`/`store` into affine ops where the indices allow |
| `-affine-scalrep` | Forward stores to loads, remove redundant loads and dead stores |
| `-affine-simplify-min-max`, `-affine-simplify-with-bounds` | Simplify `affine.min`/`max`/`apply` using known bounds |
| `-affine-simplify-structures` | Simplify affine expressions in maps/sets and normalize memrefs |
| `-affine-super-vectorize` | Vectorize to an n-D vector abstraction |

There is also `-convert-affine-for-to-gpu`, which turns top-level affine loop
nests straight into GPU kernels — the bridge to
[Chapter 21](../../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md)
and [Chapter 31](../../part6-end-to-end-ml/ch31-gpu-nvvm/README.md).

## Step 4 — Done by hand: interchange, skewing and the wavefront

Two classic transformations have **no one-flag pass** in `mlir-opt` — there
is no `-affine-loop-interchange` or `-affine-loop-skew` in the list above — so
we perform them by rewriting loop order, bounds and index maps ourselves.
[`build.sh`](examples/04-interchange-skewing/build.sh) compiles each file and
runs [`aot_main.py`](examples/04-interchange-skewing/aot_main.py) to confirm
that every version produces identical results.

### Interchange

Interchange swaps the loop nesting order. Visiting `(i, j)` in a different
order cannot change a copy `B = A` — every element is written once and the
iterations are independent — but row-major traversal (`i` outer) touches
memory contiguously, while column-major (`j` outer) jumps a whole row between
consecutive accesses. From
[`interchange_manual.mlir`](examples/04-interchange-skewing/interchange_manual.mlir),
before (`j` outer, column-major walk):

```mlir
func.func @copy_ji(%A: memref<4x5xf32>, %B: memref<4x5xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %j = 0 to 5 {
    affine.for %i = 0 to 4 {
      %v = affine.load %A[%i, %j] : memref<4x5xf32>
      affine.store %v, %B[%i, %j] : memref<4x5xf32>
    }
  }
  return
}
```

after (`i` outer, row-major walk):

```mlir
func.func @copy_ij(%A: memref<4x5xf32>, %B: memref<4x5xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 5 {
      %v = affine.load %A[%i, %j] : memref<4x5xf32>
      affine.store %v, %B[%i, %j] : memref<4x5xf32>
    }
  }
  return
}
```

Interchange is legal here because the iterations are independent. It is not
always: in the stencil below, swapping the loops would be fine too, but in a
nest where iteration `(i, j)` reads what `(i-1, j+1)` wrote, interchanging
would read the value before it is written.

### Skewing

Skewing shears the iteration space. In the stencil
`A[i,j] = A[i-1,j] + A[i,j-1]`, each cell needs its **upper** (`i-1, j`) and
**left** (`i, j-1`) neighbour, so **both** loops carry a dependence — nothing
is parallel yet.
[`skewing_manual.mlir`](examples/04-interchange-skewing/skewing_manual.mlir)
has three versions that all compute the same array. The plain sequential
stencil (the reference):

```mlir
func.func @stencil(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 1 to 8 {
    affine.for %j = 1 to 8 {
      %a = affine.load %A[%i - 1, %j] : memref<8x8xf32>
      %b = affine.load %A[%i, %j - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
```

Skewing *alone* shifts the inner loop's range by `i` (it now runs from
`i + 1` to `i + 8`) and recovers the original column as `j - i`, visiting the
same cells with the same update:

```mlir
func.func @stencil_skewed(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 1 to 8 {
    affine.for %j = affine_map<(d) -> (d + 1)>(%i) to affine_map<(d) -> (d + 8)>(%i) {
      %jn = affine.apply affine_map<(d0, d1) -> (d1 - d0)>(%i, %j)   // original column = j - i
      %a = affine.load %A[%i - 1, %jn] : memref<8x8xf32>
      %b = affine.load %A[%i, %jn - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %jn] : memref<8x8xf32>
    }
  }
  return
}
```

> **Skewing alone gives no speedup.** It is just a reindexing: `i` is still
> the outer loop, and each inner step still reads the cell the previous inner
> step wrote (`A[i, jn - 1]`), so the inner loop stays sequential. Skewing's
> real job is to make a **follow-up interchange** legal and useful. That pair
> is the wavefront.

### The wavefront: skew + interchange

Make the skewed dimension — the anti-diagonal `t = i + j` — the *outer* loop,
and iterate the cells *on* each diagonal in the inner loop. Every cell
`(i, t − i)` depends only on diagonal `t − 1`, so no two cells of the same
diagonal depend on each other, and the **inner loop is genuinely parallel**:

```mlir
func.func @stencil_wavefront(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %t = 2 to 15 {                                  // outer: over diagonals t = i + j
    // cells on diagonal t: i in [max(1, t-7), min(7, t-1)] so j = t-i stays in [1,7]
    affine.parallel (%i) = (max(1, %t - 7)) to (min(8, %t)) {   // inner: PARALLEL
      %j = affine.apply affine_map<(d0, d1) -> (d1 - d0)>(%i, %t)   // j = t - i
      %a = affine.load %A[%i - 1, %j] : memref<8x8xf32>
      %b = affine.load %A[%i, %j - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
```

The outer `affine.for` walks the diagonals in order (diagonal `t` still needs
`t − 1`); the inner `affine.parallel` is the reward — a whole diagonal at once.
The `max`/`min` bounds are the Step 2 map bounds again: they clip each
diagonal to the 7×7 interior. `-lower-affine` (`build/skewing_scf.mlir`)
turns them into `arith.maxsi`/`arith.minsi` and the parallel loop into an
`scf.parallel`:

```mlir
    scf.for %arg1 = %c2 to %c15 step %c1 {
      %c1_0 = arith.constant 1 : index
      %c-7 = arith.constant -7 : index
      %0 = arith.addi %arg1, %c-7 : index
      %1 = arith.maxsi %c1_0, %0 : index
      %c8 = arith.constant 8 : index
      %2 = arith.minsi %c8, %arg1 : index
      %c1_1 = arith.constant 1 : index
      scf.parallel (%arg2) = (%1) to (%2) step (%c1_1) {
```

Our CPU pipeline then serializes that `scf.parallel` (`-convert-scf-to-cf`
turns it into ordinary branches); making the threads real is Chapter 7's job.
The point here is that the inner loop is now *legally* parallel, which the
original nest never was.

**The picture.** Lay the grid out with row `i` going down and column `j` going
right, and label each cell with its wavefront `t = i + j`:

```text
         j=1  j=2  j=3  j=4
  i=1  [  2    3    4    5 ]        cells with the SAME label lie on one
  i=2  [  3    4    5    6 ]        anti-diagonal, a "/" line running NE-SW.
  i=3  [  4    5    6    7 ]        That line is one affine.parallel
  i=4  [  5    6    7    8 ]        (all of its cells at once).
        (rows i down, cols j right)

  the outer loop marches t = 2, 3, 4, ...: the "/" wavefront slides from the
  NW corner (1,1) to the SE corner.
```

Why every cell on a `/` is independent of the others on the same line: cell
`(i, j)` reads only its **north** neighbour `A[i-1, j]` and its **west**
neighbour `A[i, j-1]`, and both of those have `i + j = t − 1`. They live on
the *previous* diagonal, never the current one:

```text
        A[i-1,j]        <- north neighbour, on diagonal t-1
           |
 A[i,j-1] -+-> A[i,j]   <- west neighbour, on diagonal t-1
           (current cell, on diagonal t)
```

So each diagonal depends only on the one before it (that is why the outer
`t` loop stays sequential), while the cells within a diagonal touch disjoint
data (that is why the inner loop is parallel).

> **Why `affine.apply (d1 - d0)` and not `arith.subi %t, %i`?** Every index
> passed to `affine.load`/`store` must be a valid affine dim or symbol — a loop
> induction variable, a symbol, or an `affine.apply` of those. A plain
> `arith.subi` result is opaque to the affine machinery and is rejected, just
> like `invalid_index.mlir` in Step 2. `affine.apply` is the sanctioned way to
> name an index expression. For a one-off you do not even need it: since `%i`
> and `%t` are dims, you can fold the subtraction straight into the access,
> `affine.load %A[%i - 1, %t - %i]`. We use `affine.apply` only to name `j`
> once and reuse it in all three accesses.

### Run the check

```bash
bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/04-interchange-skewing/build.sh
```

```
interchange:  copy_ij == copy_ji == A
stencil:      matches the NumPy reference (A[7,7] = 3432)
skewing:      stencil == stencil_skewed
wavefront:    stencil == stencil_wavefront (parallel inner loop)
Manual transforms preserve results.
```

Starting from all ones, the stencil counts lattice paths, so `A[7,7]` is the
binomial coefficient C(14, 7) = 3432 — any reordering that broke a dependence
would read a stale neighbour and miss it.

> **So is there really no pass?** Not a stock one-flag *affine* pass. Both
> transformations are only *legal* when they preserve every dependence and
> only *profitable* in specific cases, so MLIR treats them as building blocks
> a scheduler invokes after analysis, not blind global rewrites.
> Compiler-driven routes do exist:
>
> - **Interchange** — the Transform dialect offers
>   `transform.structured.interchange`, which permutes a `linalg` op's
>   iterator order *before* it lowers to loops (the modern, scriptable way to
>   choose loop order;
>   [Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)
>   introduces the Transform dialect). MLIR also has C++ utilities
>   (`interchangeLoops`, `permuteLoops`) that a custom pass can call, and LLVM
>   runs its own `LoopInterchange` pass late in the back end.
> - **Skewing** — no user-facing pass or transform op; it exists only as an
>   internal C++ utility (`affineForOpBodySkew`, used by
>   `-affine-pipeline-data-transfer`). From `mlir-opt` alone you do it by
>   hand, as shown here.

## Step 5 — 2-D convolution

A real kernel, and the affine version of `scf.for`'s loop-carried values.

### How 2-D convolution works

Convolution (really *cross-correlation*, as in every CNN framework) slides a
small **filter** over an **input**. At each position it lays the filter over a
window of the input, multiplies the overlapping cells, and sums them into one
**output** cell:

$$\text{output}[i,j] = \sum_{fi}\sum_{fj} \text{filter}[fi,fj]\cdot\text{input}[i+fi,\, j+fj]$$

```text
   input (in_h x in_w)               filter (KH x KW)         output (OH x OW)
  +-----------------------+          +---------------+       +---------------+
  |+-------+              |          | f00 f01 f02   |       | o00 ...       |
  ||a  b  c| d  e  ...    |          | f10 f11 f12   |  ==>  |               |
  ||g  h  i| j  k  ...    |    *     | f20 f21 f22   |       |               |
  ||m  n  o| p  q  ...    |          +---------------+       |               |
  |+-------+              |                                  |               |
  | ...                   |                                  +---------------+
  +-----------------------+
   o00 = a*f00 + b*f01 + c*f02 + g*f10 + h*f11 + i*f12 + m*f20 + n*f21 + o*f22

  slide the window one step right -> output[i, j+1]; down -> output[i+1, j].
  output size = (in_h - KH + 1) x (in_w - KW + 1)  (only positions where the
                                                    window fully fits)
```

So there are **four** loops: two outer ones over the output cells `(i, j)`,
and two inner ones that reduce over the filter window `(fi, fj)` to build each
cell's sum.

### The kernel

The inner reduction accumulates with **`iter_args`** — the same loop-carried
value mechanism as `scf.for`, now on `affine.for`, with each loop handing its
running sum to the next iteration through `affine.yield`
([`conv2d.mlir`](examples/05-conv2d/conv2d.mlir)):

```mlir
func.func @conv_2d(%input: memref<?x?xf32>, %filter: memref<?x?xf32>, %output: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %OH = memref.dim %output, %c0 : memref<?x?xf32>   // output rows  = in_h - k_h + 1
  %OW = memref.dim %output, %c1 : memref<?x?xf32>   // output cols  = in_w - k_w + 1
  %KH = memref.dim %filter, %c0 : memref<?x?xf32>   // filter rows  (window height)
  %KW = memref.dim %filter, %c1 : memref<?x?xf32>   // filter cols  (window width)

  // outer loops: one iteration per OUTPUT cell (i, j) -- independent
  affine.for %i = 0 to %OH {
    affine.for %j = 0 to %OW {
      %zero = arith.constant 0.0 : f32

      // inner loops: a REDUCTION over the filter window (fi, fj)
      %acc = affine.for %fi = 0 to %KH iter_args(%a = %zero) -> (f32) {
        %acc2 = affine.for %fj = 0 to %KW iter_args(%b = %a) -> (f32) {
          %fv = affine.load %filter[%fi, %fj] : memref<?x?xf32>          // filter[fi, fj]
          %iv = affine.load %input[%i + %fi, %j + %fj] : memref<?x?xf32> // input[i+fi, j+fj]
          %p  = arith.mulf %iv, %fv : f32
          %n  = arith.addf %b, %p : f32
          affine.yield %n : f32                                         // carry sum to next fj
        }
        affine.yield %acc2 : f32                                        // carry sum to next fi
      }
      affine.store %acc, %output[%i, %j] : memref<?x?xf32>   // write the finished output[i, j]
    }
  }
  return
}
```

The subscript `%input[%i + %fi, %j + %fj]` is an affine function of four loop
variables — that is what keeps even a sliding-window access analyzable. The
two outer loops are independent (each writes a distinct `output[i, j]`), so
they can be tiled freely, and Chapter 7 parallelizes them. The two inner loops
build *one* value; they are a reduction, so they stay sequential. Carrying the
sum in `iter_args` instead of storing it to memory every step keeps it in a
register.

### Optimizing it

[`build.sh`](examples/05-conv2d/build.sh) compiles the kernel twice: as
written, and after a short affine pipeline that combines what this chapter
covered:

```bash
mlir-opt conv2d.mlir \
  --pass-pipeline="builtin.module(func.func(affine-loop-invariant-code-motion,affine-loop-tile{tile-size=4},affine-loop-unroll{unroll-factor=2},canonicalize))"
```

The result (`build/conv2d_opt.mlir`):

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (d0 + 4, s0)>
#map2 = affine_map<()[s0] -> ((s0 floordiv 2) * 2)>
module {
  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg2, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg1, %c1 : memref<?x?xf32>
    affine.for %arg3 = 0 to %dim step 4 {
      affine.for %arg4 = 0 to %dim_0 step 4 {
        affine.for %arg5 = #map(%arg3) to min #map1(%arg3)[%dim] {
          affine.for %arg6 = #map(%arg4) to min #map1(%arg4)[%dim_0] {
            %0 = affine.for %arg7 = 0 to %dim_1 iter_args(%arg8 = %cst) -> (f32) {
              %1 = affine.for %arg9 = 0 to #map2()[%dim_2] step 2 iter_args(%arg10 = %arg8) -> (f32) {
                %3 = affine.load %arg1[%arg7, %arg9] : memref<?x?xf32>
                %4 = affine.load %arg0[%arg5 + %arg7, %arg6 + %arg9] : memref<?x?xf32>
                %5 = arith.mulf %4, %3 : f32
                %6 = arith.addf %arg10, %5 : f32
                %7 = affine.load %arg1[%arg7, %arg9 + 1] : memref<?x?xf32>
                %8 = affine.load %arg0[%arg5 + %arg7, %arg6 + %arg9 + 1] : memref<?x?xf32>
                %9 = arith.mulf %8, %7 : f32
                %10 = arith.addf %6, %9 : f32
                affine.yield %10 : f32
              }
              %2 = affine.for %arg9 = #map2()[%dim_2] to %dim_2 iter_args(%arg10 = %1) -> (f32) {
                %3 = affine.load %arg1[%arg7, %arg9] : memref<?x?xf32>
                %4 = affine.load %arg0[%arg5 + %arg7, %arg6 + %arg9] : memref<?x?xf32>
                %5 = arith.mulf %4, %3 : f32
                %6 = arith.addf %arg10, %5 : f32
                affine.yield %6 : f32
              }
              affine.yield %2 : f32
            }
            affine.store %0, %arg2[%arg5, %arg6] : memref<?x?xf32>
          }
        }
      }
    }
    return
  }
}
```

Every piece is something you have now seen:

- **LICM** hoisted the zero constant `%cst` out of the whole nest.
- **Tiling** split each output loop into a tile loop (`step 4`) and a point
  loop whose upper bound is `min(i + 4, OH)` (`#map1`) — Step 2's dynamic
  tiled loop, generated for you, because the output size is a runtime symbol.
  The pass tiled only the two outer loops: a band of perfectly nested loops
  ends where the `iter_args` loop begins.
- **Unrolling** the innermost (filter-column) loop by 2: the main loop runs to
  `(KW floordiv 2) * 2` (`#map2`, a map on the symbol `%dim_2`) and carries
  *two* multiply-adds per trip, chaining the running sum through `%6`; a
  cleanup loop handles the odd column when `KW` is odd. Unrolling threaded the
  `iter_args` through correctly — `%1` feeds the cleanup loop's initial value.

### Run it

[`aot_main.py`](examples/05-conv2d/aot_main.py) runs both libraries on an
11×10 input with a 3×3 filter and compares them with a NumPy reference. The
sizes are deliberately awkward: the 9×8 output is not a multiple of the 4×4
tile, and the 3-wide filter is not a multiple of the unroll factor, so the
partial tiles and the cleanup loop are exercised too.

```bash
bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/05-conv2d/build.sh
```

```
conv2d     matches NumPy (output 9x8, max abs error 3.81e-06)
conv2d_opt matches NumPy (output 9x8, max abs error 3.81e-06)
```

## Tests

The key results are pinned by lit tests in [`test/part1/`](../../test/part1/):

| Test | What it checks |
| --- | --- |
| `ch06_affine_maps.mlir` | folded constants (incl. negative `floordiv`/`ceildiv`/`mod`), composition, dead-apply removal, `floordiv` and `min`-bound lowering, the verifier error |
| `ch06_affine_passes.mlir` | the before/after shape of every pass in Step 3, including fusion + scalrep |
| `ch06_affine_matmul_run.mlir` | compiles the matmul and runs it from a C driver |
| `ch06_interchange_skewing_run.mlir` | all five manual versions agree; the wavefront lowers to `scf.parallel` with `max`/`min` bounds |
| `ch06_conv2d_run.mlir` | plain and optimized conv2d give identical, exact results from a C driver |

```bash
.venv/bin/lit build/test -v --filter ch06_
```

The runtime tests use small C drivers (`matmul_main.c`, `transforms_main.c`,
`conv2d_main.c`, sharing [`examples/memref2d.h`](examples/memref2d.h)) instead
of Python, with integer-valued inputs so the expected output is exact.

## Key takeaways

- **A loop over-specifies order.** Loop optimization is recovering the freedom
  the sequential syntax hid, and the `affine` dialect is what lets the
  compiler *see* which orderings are required.
- **`affine` = analyzable loops.** Restricting bounds and subscripts to affine
  maps of dims and symbols buys the polyhedral model, and with it automatic
  tiling, fusion, unrolling, scalar replacement, coalescing and
  vectorization. Write `affine`, transform it, then `-lower-affine` to `scf`
  and reuse the Chapter 2–4 pipeline.
- **Affine maps fold, compose and lower.** `affine.apply` chains collapse into
  one map, constants fold away, and what remains becomes plain index
  arithmetic — with a sign fix-up for `floordiv`.
- **Passes compose.** Fusion exposes scalar replacement; canonicalize cleans up
  after coalescing; LICM, tiling and unrolling stack on a real kernel.
- **Some transformations are yours to write.** Interchange, skewing and the
  wavefront need dependence reasoning MLIR leaves to you (or to the Transform
  dialect); running the versions side by side is how you prove them correct.

## Exercises

1. **Floordiv by hand.** Add a function to `affine_apply.mlir` that returns
   `affine.apply affine_map<(d0) -> ((d0 + 3) floordiv 4 * 4)>` applied to
   `-5`, `0` and `13`. Predict the results (round up to a multiple of 4), then
   check with `-canonicalize`. Rewrite the map with `ceildiv` and confirm
   the canonical form.
2. **Scalar replacement and aliasing.** Copy `unroll_jam.mlir`, make `%y` a
   local `memref.alloc() : memref<8xf32>` that the function returns, write the
   jammed loop body by hand (or edit the pass output), and run
   `-affine-scalrep`. Confirm the second load of `x[j]` disappears, and
   explain why it did not when `%y` was an argument.
3. **Unroll a dynamic loop.** Run `affine-loop-unroll{unroll-factor=4}` on
   `@tiled_loop_dynamic` from `tiled_loop.mlir`. Which loop gets unrolled,
   what does its cleanup bound look like, and why does it involve `floordiv`?
4. **Vectorize something harder.** Change `vectorize.mlir` to a 250-element
   buffer. What does `-affine-super-vectorize="virtual-vector-size=8"` do to
   the `in_bounds` attribute, and why? Then try a reduction
   (`iter_args` sum over the array) and see whether the pass vectorizes it.
5. **A wavefront for a different stencil.** Change the stencil to
   `A[i,j] = A[i-1,j-1] + A[i-1,j+1]` (keep `j` inside `[1, 6]`). Which loop
   carries the dependence now, and is the inner `j` loop already parallel
   without skewing? Add the new version to `skewing_manual.mlir` and the
   driver to prove your answer.
6. **Tile the convolution differently.** Change the conv2d pipeline to
   `affine-loop-tile{tile-sizes=2,4}` and add `affine-loop-normalize` at the
   end. Re-run `build.sh` and check that the result still matches NumPy.

## Notes on reproducing with current MLIR

The source chapter was written against Homebrew LLVM 20.1.8. Rerunning it with
the MLIR wheel (LLVM 24) and this repository's toolchain:

- **No `llc`.** The source compiled with `llc -filetype=obj`. Here
  `mlir-translate` output goes through `tools/ll_fp_compat.py`, which rewrites
  LLVM 24's float literal syntax and drops `nocreateundeforpoison` — an
  attribute LLVM 24 attaches to `@llvm.smax`/`@llvm.smin` (used by the
  wavefront's `max`/`min` bounds) that clang 20 rejects with "unterminated
  attribute group". Then `clang -shared -x ir` builds the library.
  `-Wno-override-module` silences clang replacing the module's target triple.
- **`affine-loop-unroll` and `affine-loop-unroll-jam` must be nested.**
  `mlir-opt -affine-loop-unroll` now fails with "unable to schedule pass
  'AffineLoopUnroll' on a PassManager intended to run on 'builtin.module'"; use
  `--pass-pipeline="builtin.module(func.func(affine-loop-unroll{...}))"`.
  The other affine passes still work as plain flags.
- **Super-vectorization output changed.** The padding value of
  `vector.transfer_read` is now `ub.poison` instead of
  `arith.constant 0.0`, and the transfers carry `in_bounds = [true]`.
- **LLVM-dialect cosmetics.** Descriptors start from `llvm.mlir.poison`
  (was `llvm.mlir.undef`); index arithmetic carries `overflow<nsw>` and
  GEPs are `inbounds`.
- **`affine_apply.mlir` now returns its results.** In the source, every
  `affine.apply` was unused, so `-canonicalize` simply deleted them all and
  the folded file was an empty function. Returning the values shows the
  folding; `@dead_apply` keeps the original dead-code point.
- **New examples.** `@rounding`, `@compose`, `floordiv.mlir`,
  `invalid_index.mlir`, `@tiled_loop_dynamic`, `@unroll_cleanup`,
  `unroll_jam.mlir`, the fusion + scalrep and coalescing + canonicalize
  combinations, and the optimized conv2d were added for this port; the LICM,
  unroll, normalize, scalrep, coalescing, fusion and tiling outputs are
  otherwise unchanged from LLVM 20.
- **One NumPy helper.** The source's `common/np_memref2d.py` (a 2-D-only
  descriptor) is replaced by the rank-generic `tools/np_memref.py`.
- **Parallelization moved.** The source's `--affine-parallelize` outputs for
  matmul and conv2d, and its OpenMP step, are in Chapter 7.

## Where to go next

- **[Chapter 7: Parallel Loops and OpenMP](../ch07-parallel-loops-and-openmp/README.md)**
  — the loops this chapter showed to be independent (matmul's `i`/`j`,
  conv2d's output loops, the wavefront's diagonals) become `affine.parallel`,
  then `scf.parallel`, then real threads through the `omp` dialect.
- [Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)
  revisits tiling, fusion and loop ordering with the Transform dialect, where
  you script exactly which loops get transformed instead of running
  whole-module passes.
