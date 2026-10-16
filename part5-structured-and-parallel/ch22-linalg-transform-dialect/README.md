# Chapter 22: Linalg and the Transform Dialect

> Reworked from mlir-tutor's chapter 4 for the unified repo.

For four parts of this tutorial we have been *building* IR machinery: custom
dialects, custom passes, custom lowerings. This chapter flips the perspective.
Upstream MLIR already ships an industrial-strength stack for exactly the kind
of tensor code we have been compiling — the **linalg dialect** for describing
structured computations, and the **transform dialect** for *scheduling* them:
tiling, fusing, and vectorizing under precise, scriptable control.

Back in [Chapter 5](../../part1-using-mlir/ch05-affine-optimization/README.md)
we tiled and fused loops by passing flags to monolithic passes
(`--affine-loop-tile="cache-size=512"`, `--affine-loop-fusion`) and hoping the
heuristics inside would do the right thing. This chapter is the modern,
controllable way to do the same job: instead of flags, you write a small
*program* — in IR, shipped alongside the payload — that says "tile **this** op
with **these** sizes, fuse **that** producer into the loop you just made, then
vectorize". And because the scheduling language is itself an MLIR dialect, we
can extend it: the second half of the chapter writes two custom transform ops
in C++ and registers them into `tutorial-opt`.

## What you will learn

- The anatomy of `linalg.generic`: how `indexing_maps` and `iterator_types`
  encode an entire loop nest, and how to read one as "matmul" or "reduction"
  at a glance
- Why a scheduling *dialect* beats pass flags: transform scripts as IR,
  `transform.named_sequence`, handles (`!transform.any_op`), and
  `transform.structured.match`
- Tiling with `transform.structured.tile_using_for` and the tensor slice ops
  (`tensor.extract_slice` / `tensor.insert_slice`) that tiling produces
- The tile-and-fuse recipe: tile the *consumer* into an `scf.forall`, then
  `fuse_into_containing_op` the producers — and why consumer-first is the
  right order
- Vectorizing tiled loop bodies into `vector.contract` with
  `vectorize_children_and_apply_patterns`
- The full descent: one-shot bufferization (tensors → memrefs) and the pass
  pipeline down to the LLVM dialect
- Extending the transform dialect in C++: `TransformEachOpTrait` /
  `applyToOne`, handle memory effects, `TilingInterface`,
  `scf::tileUsingSCF`, and registering a `TransformDialectExtension`
- Verifying custom transform ops with lit tests

## Prerequisites

- **Chapter 18** (dialect conversion) and **Chapter 19** (lowering to LLVM):
  the final example reuses the same "walk down the dialect ladder" mindset,
  and you should recognize `--reconcile-unrealized-casts` when it reappears.
- **Chapter 20** (region ops): `linalg.generic` and `scf.forall` are both
  region-carrying ops; block arguments and terminators (`linalg.yield`) will
  look familiar.
- **Chapter 5** for contrast: re-skim what `--affine-loop-tile` and
  `--affine-loop-fusion` did there. This chapter achieves the same
  transformations, but *you* pick the ops, the order, and the sizes.
- **Chapter 21** introduced op interfaces; here the star interface is
  upstream's `TilingInterface`.
- A built `build/tools/tutorial-opt` and the Python venv from the repo
  README (lit tests need it).

All files for this chapter:

- Worked examples: [`examples/`](examples/) — five `.mlir` files, each with
  the transform script embedded in the same file
- Exercises: [`exercises/`](exercises/) — two-level cache tiling, with and
  without fusion (solved); [`exercises/skeletons/`](exercises/skeletons/)
  holds the TODO versions (see [Exercises](#exercises))
- Custom transform ops:
  [`lib/Transform/TinyTransform/`](../../lib/Transform/TinyTransform/) —
  `TinyTransformOps.td`, `TinyTransformOps.cpp`, `TinyTransformExtension.cpp`
- Tests: [`test/part5/loop-unroll-full.mlir`](../../test/part5/loop-unroll-full.mlir),
  [`test/part5/tile-to-l2-l1.mlir`](../../test/part5/tile-to-l2-l1.mlir)

Run every command below from the repo root (`mlir-unified-tutorial/`).

## The linalg dialect: loops as data

The linalg dialect's central abstraction is the **structured op**: an
operation that *is* a perfectly nested loop, with the loop structure encoded
in attributes instead of written out as control flow. The fundamental
building block is `linalg.generic`.

Here is a matrix multiplication, `C[m,n] += A[m,k] * B[k,n]`:

```mlir
#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>

%result = linalg.generic {
  indexing_maps = [#map_A, #map_B, #map_C],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
  outs(%C : tensor<64x64xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %prod = arith.mulf %a, %b : f32
    %sum = arith.addf %c, %prod : f32
    linalg.yield %sum : f32
} -> tensor<64x64xf32>
```

Three pieces work together:

**The iteration space.** The op implicitly iterates over a 3-D space
`(m, n, k)`. Where does the space come from? From the affine maps and the
operand shapes together: `m` ranges over `A`'s rows (64), `n` over `B`'s
columns (64), `k` over the shared dimension (128). There is no loop syntax —
the space is *inferred*.

**The `indexing_maps`.** One affine map per operand (inputs first, then
outputs), each mapping a point of the iteration space to the coordinates that
operand is accessed at. At the iteration point `(m=2, n=3, k=4)` the body
receives:

```mlir
%a = %A[2, 4]   // #map_A: (m, n, k) -> (m, k)
%b = %B[4, 3]   // #map_B: (m, n, k) -> (k, n)
%c = %C[2, 3]   // #map_C: (m, n, k) -> (m, n)
```

**The `iterator_types`.** One entry per iteration-space dimension. `parallel`
means iterations along that dimension are independent — they can run in any
order, or simultaneously. `reduction` means the dimension accumulates into
the output (note that `k` does not appear in `#map_C`: every `k` iteration
writes the *same* output element, which is exactly what makes it a
reduction).

**The body.** The region defines what happens at *one* point of the
iteration space. Block arguments are the loaded elements (inputs first, then
the current value of each output); `linalg.yield` produces the value stored
back to the output. Our body is one multiply and one accumulate — a matmul's
innermost statement.

This encoding is the payoff: a transformation like tiling does not need to
pattern-match loop nests, bounds, and subscript expressions the way the
affine passes in Chapter 5 did. The structure is right there in the
attributes — *loops as data*. And the same `linalg.generic` op with different
maps and iterator types expresses elementwise addition, row sums,
convolutions, transposes... so every transformation written against
`linalg.generic` works for all of them.

Two more spellings you will meet below: `linalg.fill`, a named op that
initializes a tensor with a scalar, and the fact that `linalg.generic` works
on both `tensor` (value semantics, SSA) and `memref` (buffer semantics)
operands. Example 1 uses memrefs; Examples 2-5 use tensors, because fusion
and bufferization want value semantics.

### Example 1: seeing the implied loops

[`examples/01-linalg-to-loops.mlir`](examples/01-linalg-to-loops.mlir)
contains three memref-based generics — the matmul above, an elementwise add,
and a row-sum reduction. `--convert-linalg-to-loops` materializes the loop
nest each one implies:

```bash
build/tools/tutorial-opt --convert-linalg-to-loops \
    part5-structured-and-parallel/ch22-linalg-transform-dialect/examples/01-linalg-to-loops.mlir
```

The matmul becomes exactly the triple loop you would write by hand:

```mlir
func.func @matmul(%arg0: memref<64x128xf32>, %arg1: memref<128x64xf32>, %arg2: memref<64x64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c64 step %c1 {
      scf.for %arg5 = %c0 to %c128 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg5] : memref<64x128xf32>
        %1 = memref.load %arg1[%arg5, %arg4] : memref<128x64xf32>
        %2 = memref.load %arg2[%arg3, %arg4] : memref<64x64xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        memref.store %4, %arg2[%arg3, %arg4] : memref<64x64xf32>
      }
    }
  }
  return
}
```

Loop bounds come from the shapes, loads come from the indexing maps, the body
is the region. Look at `@row_sum` in the output too — its output map
`(i, j) -> (i)` drops the `j` dimension, so the store target
`%arg1[%arg2]` is invariant in the inner loop: a reduction, exactly as its
`iterator_types = ["parallel", "reduction"]` promised.

So far this is Chapter 2/3 material in reverse. The interesting part is what
we can do *before* materializing loops.

## Why a scheduling dialect beats pass flags

Think about what Chapter 5's `--affine-loop-tile="cache-size=512"` really
was: one knob, applied by a heuristic, to every loop nest in the module. You
could not say "tile the matmul 32×32×16 but leave the bias add alone", or
"fuse these two ops but not that one", or control the *order* of
transformations within one pass pipeline position. Production compilers need
exactly that control — and encoding it as ever-more pass flags does not
scale.

The transform dialect's answer: make the *schedule* itself IR. A transform
script is an MLIR program that manipulates another MLIR program (the
"payload"):

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Find payload operations by matching on their name.
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Apply tiling to the matched operation.
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul
        tile_sizes [16, 16, 8]
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

Key concepts:

- **Handles.** A value of type `!transform.any_op` is not a tensor or a
  number — it is a *reference to a list of payload ops*. `%matmul` above
  points at every `linalg.generic` that the match found; `%tiled` points at
  the residual (tiled) op the tiling created, and `%loops:3` at the three new
  loops. Subsequent transform ops consume handles, so transformations
  *compose*: the output of one step is the input of the next.
- **`transform.named_sequence @__transform_main`.** The entry point. The
  `--transform-interpreter` pass looks for this symbol (in a module tagged
  `transform.with_named_sequence`) and executes it with `%arg0` bound to the
  payload root. In this chapter we keep the script in the *same file* as the
  payload, which is convenient for tutorials and lit tests; in production
  flows the script often lives in a separate library file.
- **Failure is loud.** If a match finds nothing, or a tiling is invalid, the
  interpreter reports a diagnostic pointing at the transform op that failed —
  instead of a heuristic silently declining, you get told exactly which step
  of your schedule broke.

Run any script with:

```bash
build/tools/tutorial-opt --transform-interpreter input.mlir
```

This is the same philosophy as Halide's algorithm/schedule split, rebuilt
inside MLIR: `linalg.generic` is the algorithm; the transform script is the
schedule. The upstream
[transform dialect tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
goes much deeper than we do here — this chapter gives you the working core.

## Example 2: tiling a single op

[`examples/02-tile-single-op.mlir`](examples/02-tile-single-op.mlir) is the
64×128×64 matmul on tensors, plus the script you just saw with tile sizes
`[32, 32, 16]`:

```bash
build/tools/tutorial-opt --transform-interpreter \
    part5-structured-and-parallel/ch22-linalg-transform-dialect/examples/02-tile-single-op.mlir
```

Output (trimmed):

```mlir
%0 = scf.for %arg3 = %c0 to %c64 step %c32 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
  %1 = scf.for %arg5 = %c0_0 to %c64_2 step %c32_3 iter_args(%arg6 = %arg4) -> (tensor<64x64xf32>) {
    %2 = scf.for %arg7 = %c0_1 to %c128 step %c16 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [32, 16] [1, 1]
          : tensor<64x128xf32> to tensor<32x16xf32>
      %extracted_slice_4 = tensor.extract_slice %arg1[%arg7, %arg5] [16, 32] [1, 1]
          : tensor<128x64xf32> to tensor<16x32xf32>
      %extracted_slice_5 = tensor.extract_slice %arg8[%arg3, %arg5] [32, 32] [1, 1]
          : tensor<64x64xf32> to tensor<32x32xf32>
      %3 = linalg.generic { ... iterator_types = ["parallel", "parallel", "reduction"] }
          ins(%extracted_slice, %extracted_slice_4 : tensor<32x16xf32>, tensor<16x32xf32>)
          outs(%extracted_slice_5 : tensor<32x32xf32>) {
        ...
      } -> tensor<32x32xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg8[%arg3, %arg5] [32, 32] [1, 1]
          : tensor<32x32xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    scf.yield %2 : tensor<64x64xf32>
  }
  scf.yield %1 : tensor<64x64xf32>
}
```

Read this carefully — it is the shape every later example builds on:

- Three `scf.for` loops appeared, stepping by the tile sizes (32, 32, 16),
  threading the output tensor through `iter_args` exactly as Chapter 20's
  loop-carried values did.
- Inside, `tensor.extract_slice` carves out the operand tiles
  (`32x16` of A, `16x32` of B, `32x32` of C), and `tensor.insert_slice`
  writes the computed tile back.
- Most importantly: **the tiled computation is still a `linalg.generic`** —
  the same matmul, now on `32x16 × 16x32` tiles. Tiling did not lower
  anything; it rewrote one structured op into loops around a smaller
  structured op. That closure property is what lets us keep scheduling:
  tile again (Exercises), fuse (Example 3), vectorize (Example 4).
- The transform script survives in the output, unchanged. The interpreter
  transforms the payload, not the schedule. (Upstream provides
  `--transform-preload-library` and related mechanisms to keep schedules out
  of the payload file entirely.)

## Example 3: tile and fuse

Tiling one op is not yet a win. Real programs are *chains* of ops, and the
performance killer is materializing full intermediate tensors between them —
the exact problem `--affine-loop-fusion` attacked in Chapter 5 with a
heuristic. Here we do it by hand, precisely.

[`examples/03-tile-and-fuse.mlir`](examples/03-tile-and-fuse.mlir) computes
`relu`-less "matmul + bias": a producer chain
`linalg.fill → matmul (linalg.generic) → bias add (linalg.generic)`.

The schedule follows the canonical **tile-the-consumer-first** recipe:

```mlir
transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
  // 1. Find the add (consumer) - it has only parallel iterators
  %add = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                   #linalg.iterator_type<parallel>]} in %arg0
      : (!transform.any_op) -> !transform.any_op

  // 2. Tile add with scf.forall (parallel tiling)
  %tiled_add, %forall = transform.structured.tile_using_forall %add
      tile_sizes [16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // 3. Find matmul (producer of add) - it has a reduction iterator
  %matmul = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = [#linalg.iterator_type<parallel>,
                                   #linalg.iterator_type<parallel>,
                                   #linalg.iterator_type<reduction>]} in %arg0
      : (!transform.any_op) -> !transform.any_op

  // 4. Fuse matmul into the forall
  %fused_matmul, %new_forall = transform.structured.fuse_into_containing_op
      %matmul into %forall
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // 5. Find and fuse fill (producer of matmul)
  %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
  %fused_fill, %final_forall = transform.structured.fuse_into_containing_op
      %fill into %new_forall
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // 6. Tile reduction dimension of the fused matmul
  %inner_matmul = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = [...]} in %final_forall
      : (!transform.any_op) -> !transform.any_op
  %tiled_k, %k_loop = transform.structured.tile_using_for %inner_matmul
      tile_sizes [0, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  transform.yield
}
```

New moves, one by one:

- **Matching by attributes.** The file has *two* `linalg.generic` ops, so
  matching by name alone is ambiguous. The add is the one whose
  `iterator_types` are all-parallel; the matmul is the one with a
  `reduction`. Structural matching like this is the transform dialect's
  bread and butter (upstream also has `transform.structured.match ops`
  with interface and result-shape predicates).
- **`tile_using_forall`.** Instead of sequential `scf.for` loops this
  produces one `scf.forall` — a multi-dimensional *parallel* loop whose
  iterations may run concurrently and which writes results via
  `tensor.parallel_insert_slice`. Legal here precisely because the add's
  iterators are all `parallel`. This is the op Chapter 21's GPU tiles
  foreshadowed, and the natural handoff point to GPU/thread distribution.
- **`fuse_into_containing_op`.** The fusion workhorse. Given a producer
  handle and a loop handle, it clones the producer *inside* the loop and
  shrinks it to compute only the tile the loop body actually consumes.
  Consumer-first ordering is what makes this work: once the add is tiled,
  each `forall` iteration demands exactly one 16×16 tile of the matmul
  result, so the fused matmul knows what slice to compute.
- **Tiling only the reduction.** Tile sizes `[0, 0, 8]` mean "leave m and n
  alone, tile k by 8" — a size of 0 skips a dimension. The parallel dims are
  already 16×16 from fusion; this adds the sequential k-loop inside.

Run it (`--canonicalize` folds away leftover dead ops — see the note below):

```bash
build/tools/tutorial-opt --transform-interpreter --canonicalize \
    part5-structured-and-parallel/ch22-linalg-transform-dialect/examples/03-tile-and-fuse.mlir
```

The payload becomes (trimmed):

```mlir
%1 = scf.forall (%arg3, %arg4) in (4, 4) shared_outs(%arg5 = %0) -> (tensor<64x64xf32>) {
  %extracted_slice = tensor.extract_slice %arg0[%4, 0] [16, 128] [1, 1]
      : tensor<64x128xf32> to tensor<16x128xf32>
  %extracted_slice_0 = tensor.extract_slice %arg1[0, %5] [128, 16] [1, 1]
      : tensor<128x64xf32> to tensor<128x16xf32>
  %extracted_slice_1 = tensor.extract_slice %arg5[%6, %7] [16, 16] [1, 1]
      : tensor<64x64xf32> to tensor<16x16xf32>

  // fused fill: initialize just this 16x16 tile
  %8 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>

  // fused matmul, reduction tiled by 8
  %9 = scf.for %arg6 = %c0 to %c128 step %c8 iter_args(%arg7 = %8) -> (tensor<16x16xf32>) {
    %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, %arg6] [16, 8] [1, 1] ...
    %extracted_slice_5 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [8, 16] [1, 1] ...
    %11 = linalg.generic { ... ["parallel", "parallel", "reduction"] }
        ins(%extracted_slice_4, %extracted_slice_5 : tensor<16x8xf32>, tensor<8x16xf32>)
        outs(%arg7 : tensor<16x16xf32>) { ... } -> tensor<16x16xf32>
    scf.yield %11 : tensor<16x16xf32>
  }

  // tiled bias add consumes the tile while it is hot
  %10 = linalg.generic { ... ["parallel", "parallel"] }
      ins(%9, %extracted_slice_2 : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%extracted_slice_3 : tensor<16x16xf32>) { ... } -> tensor<16x16xf32>

  scf.forall.in_parallel {
    tensor.parallel_insert_slice %10 into %arg5[%2, %3] [16, 16] [1, 1]
        : tensor<16x16xf32> into tensor<64x64xf32>
  }
}
```

One `scf.forall` over a 4×4 grid of tiles; inside each iteration the fill,
the matmul (with its own k-loop), and the bias add all operate on the same
16×16 tile. No full 64×64 intermediate ever exists between matmul and add —
that is fusion.

> **Why `--canonicalize`?** `fuse_into_containing_op` *clones* the producer
> into the loop; the original op is only deleted if it becomes dead and
> something cleans it up. Run the example without `--canonicalize` and you
> will see the original, untiled matmul still sitting in front of the
> `scf.forall` as dead code. This is a general transform-dialect habit:
> schedules leave canonicalization work behind, and you finish with a
> cleanup (either the pass, or `transform.apply_patterns` ops inside the
> schedule itself).

## Example 4: vectorization

Tiles of 16×16 are exactly the granularity SIMD units want.
[`examples/04-vectorize.mlir`](examples/04-vectorize.mlir) is Example 3's
payload and schedule, plus two lines at the end:

```mlir
// 5. Vectorize all linalg operations in the function
%func = transform.structured.match ops{["func.func"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
transform.structured.vectorize_children_and_apply_patterns %func
    : (!transform.any_op) -> !transform.any_op
```

`vectorize_children_and_apply_patterns` walks everything under the matched
function, replaces each linalg op with vector-dialect ops of the tile's
shape, and runs the standard simplification patterns afterwards (which, as a
side effect, also cleans up the dead untiled matmul that Example 3 needed
`--canonicalize` for).

```bash
build/tools/tutorial-opt --transform-interpreter \
    part5-structured-and-parallel/ch22-linalg-transform-dialect/examples/04-vectorize.mlir
```

The k-loop body is now (trimmed):

```mlir
%8 = scf.for %arg6 = %c0 to %c128 step %c8 iter_args(%arg7 = %7) -> (tensor<16x16xf32>) {
  %16 = vector.transfer_read %arg0[%15, %arg6], %0 {in_bounds = [true, true]}
      : tensor<64x128xf32>, vector<16x8xf32>
  %18 = vector.transfer_read %arg1[%arg6, %17], %0 {in_bounds = [true, true]}
      : tensor<128x64xf32>, vector<8x16xf32>
  %19 = vector.transfer_read %arg7[%c0, %c0], %0 {in_bounds = [true, true]}
      : tensor<16x16xf32>, vector<16x16xf32>
  %20 = vector.contract {indexing_maps = [#map1, #map2, #map3],
                         iterator_types = ["parallel", "parallel", "reduction"],
                         kind = #vector.kind<add>}
      %16, %18, %19 : vector<16x8xf32>, vector<8x16xf32> into vector<16x16xf32>
  %21 = vector.transfer_write %20, %arg7[%c0, %c0] {in_bounds = [true, true]}
      : vector<16x16xf32>, tensor<16x16xf32>
  scf.yield %21 : tensor<16x16xf32>
}
```

Three things worth savoring:

- The `linalg.generic` matmul became a single `vector.contract` — and its
  attributes are the *same* `indexing_maps` and `iterator_types` we started
  with. The structured-op abstraction carries all the way into the vector
  dialect; only the container changed (tensor tile → virtual register).
- The fill became a `vector.transfer_write` of a `dense<0.0>` constant; the
  bias add became an `arith.addf` on `vector<16x16xf32>` values — arith ops
  are elementwise on vectors for free.
- `in_bounds = [true, true]` is the tiling paying rent: because 16 and 8
  divide the problem sizes evenly, the vectorizer can prove no masking is
  needed. Ragged tiles would produce masked transfers instead.

Note the order of operations, which is the whole point of the chapter:
**tile first, then vectorize**. Vectorizing the original 64×128×64 matmul
directly would produce absurd `vector<64x128xf32>` values; after tiling, the
vector sizes are exactly the tile sizes.

## Example 5: bufferize and lower to LLVM

The IR still computes on tensors — immutable SSA values with no memory
address. To execute on a CPU we must descend the same ladder as
Chapter 19, with one big new rung at the top: **bufferization**, the
assignment of memory buffers (memrefs) to tensor values.

[`examples/05-bufferize-to-llvm.mlir`](examples/05-bufferize-to-llvm.mlir)
has the same payload and schedule as Example 4. Bufferization and the final
lowering are ordinary passes, run *after* the transform interpreter — a
reminder that the two mechanisms compose; a real pipeline interleaves
transform scripts and passes freely. The pipeline from the source file:

```bash
build/tools/tutorial-opt --transform-interpreter \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --convert-vector-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    part5-structured-and-parallel/ch22-linalg-transform-dialect/examples/05-bufferize-to-llvm.mlir
```

Stage by stage:

1. `--one-shot-bufferize="bufferize-function-boundaries"` — tensors become
   memrefs, in one whole-module analysis that reuses buffers in place where
   it can prove it safe. `tensor.empty` becomes `memref.alloc`,
   `tensor.extract_slice` becomes `memref.subview`, and
   `tensor.insert_slice`/`parallel_insert_slice` mostly *disappear* — with
   buffers, writing the tile in place is the natural thing. The
   `bufferize-function-boundaries` option bufferizes function signatures
   too (`tensor<64x128xf32>` arguments become strided memrefs).
2. `--convert-linalg-to-loops` — any linalg ops that survived (none here,
   we vectorized them all — but keep it in the pipeline for schedules that
   vectorize less aggressively) become loops.
3. `--convert-scf-to-cf` — structured loops become blocks and branches
   (Chapter 2's unstructured control flow).
4. `--convert-vector-to-llvm` — `vector.contract` and the transfers become
   LLVM vector instructions (`llvm.insertelement`, `arith.mulf` on 1-D
   vectors, `llvm.intr.vector.reduce.fadd` intrinsics, ...).
5. `--convert-func-to-llvm` — functions and calls; memref arguments explode
   into the (ptr, ptr, offset, sizes, strides) descriptor structs you met in
   Chapter 19.
6. `--reconcile-unrealized-casts` — erases the now-matching cast pairs the
   partial conversions left behind.

The output opens like this — compare with Chapter 19's descriptors:

```mlir
llvm.func @matmul_bias(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64,
                       %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, ...)
    -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
  ...
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
  cf.br ^bb1(%c0 : index)
^bb1(%32: index):  // 2 preds: ^bb0, ^bb8
  %33 = arith.cmpi slt, %32, %c4 : index
  cf.cond_br %33, ^bb2, ^bb9
  ...
```

Notice `memref.alloc` and `cf.br` in the middle of an `llvm.func`: this
six-pass pipeline gets you *most* of the way, but not to pure LLVM dialect.
Count the surviving ops and you find leftovers from five dialects —
`affine.apply` (from tile-index arithmetic), `cf` branches, `memref`
subviews, a couple of `vector.transfer`s on strided memrefs, `ub.poison`,
and the `unrealized_conversion_cast`s that could not reconcile around them.
This is the normal texture of building a lowering pipeline (Chapter 19
hit the same wall): each leftover op names the conversion pass you still
owe. Paying all the debts:

```bash
build/tools/tutorial-opt --transform-interpreter \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --convert-vector-to-scf \
    --convert-scf-to-cf \
    --convert-vector-to-llvm \
    --expand-strided-metadata \
    --lower-affine \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-ub-to-llvm \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    part5-structured-and-parallel/ch22-linalg-transform-dialect/examples/05-bufferize-to-llvm.mlir
```

The additions: `--convert-vector-to-scf` legalizes the awkward
transfers into loops of legal ones *before* the vector-to-LLVM conversion;
`--expand-strided-metadata` + `--finalize-memref-to-llvm` lower subviews and
allocs to descriptor arithmetic; `--lower-affine` (after
`expand-strided-metadata`, which itself emits `affine.apply`) turns index
maps into `arith`; and the `arith`/`ub`/`cf` conversions sweep up the rest.
Counting ops per dialect in the result:

```
1803 llvm
   8 transform      <- the embedded schedule, still along for the ride
```

Every payload op is now LLVM dialect. (The transform module survives because
nothing lowers a schedule — delete it, or keep schedules in separate files,
before `mlir-translate --mlir-to-llvmir` as in Chapter 19.)

## Extending the transform dialect in C++

Everything above used upstream transform ops. Because the scheduling
language is just another dialect, you can add your own ops to it, and
`tutorial-opt` already ships two: `transform.tiny.loop_unroll_full`
(walked through here) and `transform.tiny.tile_to_l2_l1` (built in
[Exercise 3](#exercise-3-a-custom-transform-op-transformtinytile_to_l2_l1)).
The code lives in [`lib/Transform/TinyTransform/`](../../lib/Transform/TinyTransform/):

| File | Role |
|---|---|
| `TinyTransformOps.td` | ODS definitions of both ops |
| `TinyTransformOps.cpp` | their `applyToOne` and `getEffects` implementations |
| `TinyTransformExtension.cpp` | registers the ops into the transform dialect |

### Defining the op

```tablegen
def TinyLoopUnrollFullOp : TransformDialectOp<"tiny.loop_unroll_full",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

`TransformDialectOp<"tiny.loop_unroll_full", ...>` puts the op in the
`transform` dialect's namespace, which is why it is spelled
`transform.tiny.loop_unroll_full`. The traits do the heavy lifting:

- **`TransformOpInterface`** makes it something the transform interpreter
  can execute.
- **`TransformEachOpTrait`** supplies the interpreter-facing `apply` method
  for you. A handle can point at many payload ops (`structured.match` may
  find several `scf.for`s), and the trait loops over them, calling your
  `applyToOne` once per payload op. You only write the single-op logic.
- **`DeclareOpInterfaceMethods<MemoryEffectsOpInterface>`** obliges you to
  write `getEffects`, which describes what the op does to *handles* and to
  the *payload IR* (see "Declaring effects on handles" below).
- **`ReportTrackingListenerFailuresOpTrait`** reports it when a rewrite
  replaces a payload op in a way the interpreter cannot track, instead of
  leaving a dangling handle.

### Implementing `applyToOne`

```cpp
DiagnosedSilenceableFailure
TinyLoopUnrollFullOp::applyToOne(TransformRewriter &rewriter, Operation *target,
                                 ApplyToEachResultList &results,
                                 TransformState &state) {
  auto forOp = dyn_cast<scf::ForOp>(target);
  if (!forOp)
    return emitSilenceableError()
           << "expected scf.for, but got " << target->getName();
  if (failed(loopUnrollFull(forOp)))
    return emitSilenceableError()
           << "failed to fully unroll loop (bounds may not be constant)";
  return DiagnosedSilenceableFailure::success();
}
```

The return type is the transform dialect's three-way result:
- **Success.**
- **A silenceable failure:** the schedule couldn't apply here, but the
  payload IR is still valid. An enclosing `transform.sequence
  failures(suppress)` or an alternatives op can recover from it.
- **A definite failure:** the IR may be broken, so the interpreter aborts.

Failing to unroll a loop with dynamic bounds is the textbook silenceable
case. The real work is done by upstream's `loopUnrollFull` from the SCF
utilities; the transform op is a thin wrapper around it that you can call
from a schedule.

On the first test case in `test/part5/loop-unroll-full.mlir` (a 4-trip
loop adding 1), `tutorial-opt --transform-interpreter` produces:

```mlir
func.func @unroll_simple_loop() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  ...
  %0 = arith.addi %c0_i32, %c1_i32 : i32
  %1 = arith.addi %0, %c1_i32 : i32
  %2 = arith.addi %1, %c1_i32 : i32
  %3 = arith.addi %2, %c1_i32 : i32
  return %3 : i32
}
```

Point the same schedule at a loop whose upper bound is a function argument,
and the silenceable error is reported at the transform op that failed:

```
error: failed to fully unroll loop (bounds may not be constant)
    transform.tiny.loop_unroll_full %loop : !transform.any_op
    ^
```

### Declaring effects on handles

```cpp
void TinyLoopUnrollFullOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}
```

Handles are the transform dialect's references into the payload IR, and
the interpreter uses these effects to keep them valid:

- `onlyReadsHandle` says `%loop` is still usable afterwards. The
  alternative, `consumesHandle`, invalidates it (and every handle nested
  inside it), so any later use is an error. Ops that erase or replace their
  target usually consume it.
- `producesHandle` declares the op's results as new handles.
  `tile_to_l2_l1` uses it, because it returns the tiled op.
- `modifiesPayload` records that the payload IR changes. This matters to
  analyses that check whether a schedule is side-effect free.

### Registering the ops

The ops don't form a new dialect. They are added to the existing transform
dialect through a `TransformDialectExtension`:

```cpp
class TinyTransformExtension
    : public transform::TransformDialectExtension<TinyTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TinyTransformExtension)
  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "Transform/TinyTransform/TinyTransformOps.cpp.inc"
        >();
  }
};

void mlir::tiny::registerTinyTransformExtension(DialectRegistry &registry) {
  registry.addExtensions<TinyTransformExtension>();
}
```

`GET_OP_LIST` expands to the comma-separated list of every op class
generated from the `.td`, so adding a new op needs no change here. The
driver calls `mlir::tiny::registerTinyTransformExtension(registry)` next
to the upstream registrations in
[`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp). Extensions are
applied lazily, when the transform dialect is loaded.

`loop_unroll_full` is the reference implementation. Exercise 3 below uses
the same skeleton for a harder op, which:
- takes attributes (`l2 [...] l1 [...]`),
- drives upstream's `TilingInterface` twice,
- returns a handle.

## Exercises

> Ported from mlir-tutor's chapter 4 exercises (its `ch4-exercise` branch).

The original repo shipped these exercises as TODO files on a separate git
branch. Here the **solutions are in-tree** — they are what `tutorial-opt` is
built from and what lit runs — and the original TODO versions sit next to
them in [`exercises/skeletons/`](exercises/skeletons/):

| Skeleton | Copy it over (the solution) | Checked by |
| --- | --- | --- |
| `skeletons/ex1-l2-l1-no-fusion.mlir` | `exercises/ex1-l2-l1-no-fusion.mlir` | its own RUN line |
| `skeletons/ex2-l2-l1-with-fusion.mlir` | `exercises/ex2-l2-l1-with-fusion.mlir` | its own RUN line |
| `skeletons/TinyTransformOps.td` | `lib/Transform/TinyTransform/TinyTransformOps.td` | `test/part5/tile-to-l2-l1.mlir` |
| `skeletons/TinyTransformOps.cpp` | `lib/Transform/TinyTransform/TinyTransformOps.cpp` | `test/part5/tile-to-l2-l1.mlir` |

The workflow is the same for every exercise in this tutorial that ships a
skeleton: copy the skeleton over the solution, fill in the TODOs, rebuild if
you touched C++ or TableGen, run the test, and use `git` to get the solution
back when you want to compare:

```bash
CH=part5-structured-and-parallel/ch22-linalg-transform-dialect
cp $CH/exercises/skeletons/ex1-l2-l1-no-fusion.mlir $CH/exercises/
# ... edit $CH/exercises/ex1-l2-l1-no-fusion.mlir ...
.venv/bin/lit build/test -v --filter ex1-l2-l1     # fails until you are done
git diff $CH/exercises/ex1-l2-l1-no-fusion.mlir    # compare with the solution
git checkout -- $CH/exercises/ex1-l2-l1-no-fusion.mlir   # restore it
```

lit never runs the skeletons themselves: `lit.cfg.py` excludes every
directory named `skeletons`. (The chapter's `examples/` and `exercises/`
reach the suite through the symlink `test/part5/ch22-linalg-transform-dialect`,
so they show up as `part5/ch22-linalg-transform-dialect/...` in lit output.)

### Exercise 1: L2/L1 tiling without fusion

Open `exercises/ex1-l2-l1-no-fusion.mlir` (after copying the skeleton over
it). The payload is Example 3's `fill → matmul → bias add` chain at
128×256×128; the `@__transform_main` body is empty. Write a schedule that
tiles for two cache levels, with matmul and add tiled *separately*:

1. Tile the matmul with L2 sizes `[64, 64, 32]` using `scf.for`
   (`transform.structured.tile_using_for`).
2. Tile the resulting matmul again with L1 sizes `[16, 16, 8]`, again with
   `scf.for`.
3. Tile the add with L2 sizes `[64, 64]` using `scf.for`.
4. Tile it again with L1 sizes `[16, 16]`.

Pick the matmul and the add apart with `transform.structured.match` on their
`iterator_types` (the matmul is the one with a `reduction`), as Example 3
does. Any L2/L1 sizes and any problem shape are fine — the FileCheck lines
are structural: six nested `scf.for` around the matmul, then a separate
four-deep nest around the add.

Then look at the output and ask what the separate nests cost: the whole
128×128 matmul result is materialized before the first add tile runs.

### Exercise 2: L2/L1 tiling with fusion

Same payload, in `exercises/ex2-l2-l1-with-fusion.mlir`, but this time fuse:

1. Tile the add (the *consumer*) with L2 sizes using
   `transform.structured.tile_using_forall`.
2. Fuse the matmul (its producer) into that `scf.forall` with
   `transform.structured.fuse_into_containing_op`.
3. Fuse the `linalg.fill` (the matmul's producer) into the same loop.
4. Tile the reduction dimension of the fused matmul (`tile_sizes [0, 0, 32]`
   — a 0 means "leave this dimension alone").
5. Apply L1 tiling to the matmul and to the add separately.

Steps 4 and 5 need handles to the *fused* ops, which live inside the
`scf.forall`: match again with `... in %forall` rather than `in %arg0`.
The checks expect one `scf.forall` containing the fill, a four-deep
matmul nest, and a two-deep add nest. Any sizes and shape are fine.

### Exercise 3: a custom transform op, `transform.tiny.tile_to_l2_l1`

Exercises 1 and 2 spell the two-level tiling out as two
`tile_using_for` calls per op. Package it as one transform op of our own:

```mlir
%matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
%tiled = transform.tiny.tile_to_l2_l1 %matmul
    l2 [64, 64, 32] l1 [16, 16, 8]
    : (!transform.any_op) -> !transform.any_op
```

The op takes a handle to the ops to tile and two integer arrays, and returns
a handle to the innermost tiled op. The `tiny.` prefix is only a naming
convention: the op belongs to the transform dialect, added to it through a
`TransformDialectExtension`. All the code is in
[`lib/Transform/TinyTransform/`](../../lib/Transform/TinyTransform/), next to a
finished reference op, `transform.tiny.loop_unroll_full`
(tested by [`test/part5/loop-unroll-full.mlir`](../../test/part5/loop-unroll-full.mlir)).
Read that one first; the exercise is to write its sibling.

**The ODS half** (`skeletons/TinyTransformOps.td` →
`lib/Transform/TinyTransform/TinyTransformOps.td`). Define
`TinyTileToL2L1Op` as a `TransformDialectOp<"tiny.tile_to_l2_l1", ...>` with:

- traits `TransformOpInterface` (it is a transform op),
  `TransformEachOpTrait` (the interpreter calls you once per payload op in
  the handle, through `applyToOne`, so you never loop over the handle
  yourself), `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>` (you
  write `getEffects` by hand), and `ReportTrackingListenerFailuresOpTrait`;
- arguments `TransformHandleTypeInterface:$target`,
  `DenseI64ArrayAttr:$l2_tile_sizes`, `DenseI64ArrayAttr:$l1_tile_sizes`,
  and one result, `TransformHandleTypeInterface:$tiled_op`;
- `assemblyFormat` =
  ``$target `l2` $l2_tile_sizes `l1` $l1_tile_sizes attr-dict `:` functional-type(operands, results)``
  (a `DenseI64ArrayAttr` prints as `[64, 64, 32]`, which is where the
  bracketed syntax above comes from);
- an `extraClassDeclaration` declaring `applyToOne(TransformRewriter &,
  Operation *target, ApplyToEachResultList &results, TransformState &)`,
  copied from `TinyLoopUnrollFullOp`.

No registration change is needed: `TinyTransformExtension.cpp` registers
every op in the generated `GET_OP_LIST`, so a new `def` is picked up
on the next build.

**The C++ half** (`skeletons/TinyTransformOps.cpp` →
`lib/Transform/TinyTransform/TinyTransformOps.cpp`). Implement
`TinyTileToL2L1Op::applyToOne` with upstream's interface-driven tiling,
which works on any op that implements `TilingInterface` (every linalg op
does), not only on `linalg.generic`:

1. `dyn_cast<TilingInterface>(target)`; if that fails, return
   `emitSilenceableError() << "target does not implement TilingInterface"`.
   A *silenceable* error lets an enclosing `transform.sequence` with
   `failures(suppress)` carry on. A definite error would abort the script.
2. Convert `getL2TileSizes()` (an `ArrayRef<int64_t>`) into
   `SmallVector<OpFoldResult>` with `rewriter.getIndexAttr(size)`, and put
   them in a `scf::SCFTilingOptions` with `setTileSizes(...)`.
   (`SCFTilingOptions` also chooses the loop kind; the default is `scf.for`.)
3. `rewriter.setInsertionPoint(target)`, then call
   `scf::tileUsingSCF(rewriter, tilingInterface, l2Options)`. It returns
   `FailureOr<scf::SCFTilingResult>`; on success the result carries the new
   `loops`, the `tiledOps` (the smaller op inside the loops) and the
   `replacements` for the original op's results.
4. `rewriter.replaceOp(target, l2Result->replacements)`. Going through the
   `TransformRewriter` is what keeps other handles that point at `target`
   up to date. That tracking is also why the op carries
   `ReportTrackingListenerFailuresOpTrait`.
5. Repeat steps 1–4 on `l2Result->tiledOps[0]` with the L1 sizes. This is
   the same "tiling produces a smaller structured op you can tile again"
   closure you saw in Example 2.
6. `results.push_back(l1Result->tiledOps[0])` to fill the `$tiled_op` handle,
   and return `DiagnosedSilenceableFailure::success()`.

Then write `getEffects`: `onlyReadsHandle(getTargetMutable(), effects)` (we
do not consume the input handle), `producesHandle(getOperation()->getOpResults(), effects)`,
and `modifiesPayload(effects)`. The skeleton's comments contain this as a
hint. Upstream's `transform.structured.tile_using_for` in
`mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp` is the
production-grade version of the same recipe.

Copy both skeletons over, then rebuild:

```bash
cp part5-structured-and-parallel/ch22-linalg-transform-dialect/exercises/skeletons/TinyTransformOps.{td,cpp} \
   lib/Transform/TinyTransform/
cmake --build build -j
```

The skeletons build as shipped: the op is simply missing, so
`test/part5/tile-to-l2-l1.mlir` fails to parse `transform.tiny.tile_to_l2_l1`.
Once both halves are in, that test sees six nested `scf.for` around a
`linalg.generic`. Restore the solution with
`git checkout -- lib/Transform/TinyTransform/` when you are done.

### Testing

Everything in this chapter runs under Part 5's target:

```bash
cmake --build build --target check-part5
# or just this chapter's files and its two custom-op tests:
.venv/bin/lit build/test -v --filter 'ch22-linalg|tile-to-l2-l1|loop-unroll-full'
```

`check-part5` covers `examples/01`–`05` (they must run without errors),
`exercises/ex1` and `ex2` (structural FileCheck), and
`test/part5/tile-to-l2-l1.mlir` / `loop-unroll-full.mlir` for the custom
transform ops.
