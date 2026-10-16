# Chapter 8: Linalg by Hand

> Adapted from mlir-ir-lowering's 4_linalg chapter, itself based on Stephen
> Diehl's "MLIR Part 4 — Linear Algebra in MLIR" article.

**Stop writing loops. Name the operation.**

[Chapter 6](../ch06-affine-maps-and-loop-transforms/README.md) made loops
*first-class* so the compiler could optimize them, but you still **wrote the
loops**: you spelled out the three nested `affine.for`s of a matrix multiply,
and only then did the optimizer get to work. `linalg` asks a sharper question:
why write loops at all? A matrix multiply *is* a matrix multiply. Say **that**,
and let the compiler generate the loops, choose their order, tile them, and
fuse them with their neighbours.

That is the idea behind the `linalg` dialect: a small set of **structured
operations** that describe linear-algebra computations *declaratively*, the
**what** and not the **how**. `linalg.matmul`, `linalg.add`,
`linalg.broadcast`, convolutions, reductions: each is a single op that still
carries enough structure for the compiler to reason about. This is the level
real ML compilers target, because chains of these ops (a neural network is
little else) can be fused and tiled *as a group* in ways hand-written loops
never could.

Think of it as one more rung above Chapters 5 and 6. There, an `affine.for`
nest was a *shape* the polyhedral model could slice and tile, but the shape
was still implied by loops you typed out. Here the operation names itself, and
the loop nest is something the compiler *derives on demand*. You have climbed
from "loops the compiler can analyze" to "loops the compiler writes for you."

You have already met two linalg ops: [Chapter 1](../ch01-why-mlir-and-reading-ir/README.md)
read a `linalg.matmul` (and its `outs` quirk) and wrote ReLU as a
`linalg.generic`, and [Chapter 2](../ch02-lowering-pipelines/README.md)
lowered that pair stage by stage. This chapter writes linalg *by hand*, op by
op, compiles every runnable example to a shared library, and checks it from
Python against NumPy.

## What you will learn

- What a structured op is: *payload* (scalar math in a region) plus
  *structure* (`indexing_maps` and `iterator_types`), and how the block
  arguments and `linalg.yield` of a `linalg.generic` region work
- The elementwise ladder: `linalg.generic` → `linalg.map` → named ops like
  `linalg.add`, and the passes that move between the rungs
  (`--linalg-generalize-named-ops`, `--linalg-specialize-generic-ops`)
- Reductions: the `reduction` iterator, why the *output* map drops a
  dimension, and the named `linalg.reduce`
- How `linalg.matmul` combines `parallel` and `reduction`, and the difference
  between lowering it to `scf` loops and to `affine` loops
- Broadcasting with `linalg.broadcast`, and the same thing as a single
  indexing map that ignores a loop dimension
- Elementwise fusion through passes (`--linalg-generalize-named-ops`,
  `--linalg-fuse-elementwise-ops`) and running a fused tensor kernel with
  `--buffer-results-to-out-params`
- Tiling a matmul with plain passes, and where the scriptable alternative
  lives
- When linalg ops on tensors need bufferization and when ops on memrefs do
  not

## Prerequisites

- Chapter 1 (reading linalg IR, `ins`/`outs`) and Chapter 2 (bufferization
  and the `scf` → `cf` → `llvm` pipeline)
- Chapter 3 (turning the LLVM dialect into a native library with
  `mlir-translate` and `clang`, including the `tools/ll_fp_compat.py` filter)
- Chapter 4 (memref descriptors and the C ABI): every driver here hands NumPy
  arrays to compiled code through [`tools/np_memref.py`](../../tools/np_memref.py),
  which that chapter explains field by field
- Chapters 5 and 6 (affine loops, affine maps, `--affine-loop-tile`,
  `--affine-loop-fusion`)
- The repo's `.venv` with the MLIR wheel and NumPy (see the root README), and
  `clang` on your `PATH`

## Running the examples

Each step has its own folder under [`examples/`](examples/) with a `build.sh`
that writes everything into a local `build/` directory. The scripts share
[`examples/common.sh`](examples/common.sh), which puts the wheel's tools on
`PATH` and defines one helper, `compile_kernel`, for the memref-to-library
pipeline every runnable step uses:

```bash
mlir-opt kernel.mlir \
    --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
    --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts -o kernel_llvm.mlir
mlir-translate kernel_llvm.mlir --mlir-to-llvmir -o kernel.ll
python3 tools/ll_fp_compat.py < kernel.ll > kernel.compat.ll
clang -shared -fPIC -x ir kernel.compat.ll -o libkernel.so
```

Every kernel function carries `attributes {llvm.emit_c_interface}`, so the
library exports a `_mlir_ciface_<name>` wrapper that takes one pointer per
memref descriptor. Each `run.py` builds those descriptors over NumPy arrays
with `numpy_to_memref` (no copy: the kernel writes straight into the NumPy
array) and checks the result against NumPy.

| Step | Folder | Topic | Runnable? |
| --- | --- | --- | --- |
| 1 | [`01-elementwise/`](examples/01-elementwise/) | `linalg.generic`, `linalg.map`, named ops | yes (Python) |
| 2 | [`02-reduce/`](examples/02-reduce/) | `linalg.reduce` and the `reduction` iterator | yes (Python) |
| 3 | [`03-matmul/`](examples/03-matmul/) | `linalg.matmul`, its generic form, `scf` vs `affine` | yes (Python) |
| 4 | [`04-broadcast/`](examples/04-broadcast/) | `linalg.broadcast` and broadcast indexing maps | yes (Python) |
| 5 | [`05-fusion/`](examples/05-fusion/) | elementwise fusion with passes | yes (Python) |
| 6 | [`06-tiling/`](examples/06-tiling/) | tiling a matmul with passes | inspect only |

Run them all from the repo root:

```bash
for d in part1-using-mlir/ch08-linalg-by-hand/examples/0*/; do
  echo "=== $d"; bash "$d/build.sh"
done
```

Every IR listing and program output below is copied from those `build/`
directories, produced with the MLIR wheel (LLVM 24).

## What a structured op actually is

A `linalg` op is **payload + structure**. The *payload* is the scalar math (an
`addf`, a `mulf`) written in a region; the *structure* is the iteration space
and how each operand is indexed within it. Nothing in between: no loop
keywords, no index arithmetic.

```text
   ┌─────────────────────── linalg.generic ────────────────────────┐
   │  STRUCTURE                          PAYLOAD                    │
   │  indexing_maps  = [...]   ─┐        ^bb0(%a, %b, %out):        │
   │  iterator_types = [...]    ├──────►   %s = arith.addf %a, %b   │
   │  ins(...) outs(...)       ─┘          linalg.yield %s          │
   │  "which element, in what order"     "what to compute per elt"  │
   └────────────────────────────────────────────────────────────────┘
      the compiler expands this into a loop nest on demand
```

The most general op, `linalg.generic`, spells the structure out with two
attributes:

- **`indexing_maps`**: one affine map per operand (inputs *and* outputs),
  saying how that operand is indexed by the loop variables. For elementwise
  add, all three maps are `(d0) -> (d0)`: element `i` of each operand maps to
  loop index `i`. For matmul they are `(i,j,k)->(i,k)`, `(i,j,k)->(k,j)`,
  `(i,j,k)->(i,j)`, exactly $A_{ik}$, $B_{kj}$, $C_{ij}$. These are the same
  affine maps as in Chapter 6, now used to *describe* an op rather than to
  index a hand-written loop.
- **`iterator_types`**: one per loop dimension, classifying it:

  | Iterator | Meaning | Example |
  | --- | --- | --- |
  | `parallel` | iterations are independent | elementwise ops; the `i, j` of matmul |
  | `reduction` | iterations accumulate into the same output element | the `k` sum of matmul; the window dimensions of a convolution |

  (The original article lists a third kind, `window`, for sliding windows.
  Current MLIR rejects it: `unexpected iterator_type (window)`. A convolution
  is now a contraction whose input map adds the window offset to the output
  position. Generalizing `linalg.conv_2d_nhwc_hwcf` shows the input map
  `(d0, ..., d6) -> (d0, d1 + d4, d2 + d5, d6)` with `d4..d6` marked
  `reduction`.)

Everything else in the dialect is `linalg.generic` with the boilerplate
hidden:

| Level | Op | What it is |
| --- | --- | --- |
| Most general | `linalg.generic` | the raw payload + structure form |
| Elementwise | `linalg.map { op }`, `linalg.add`/`sub`/`mul`/`div`, ... | apply a scalar op per element |
| Reduction | `linalg.reduce { op }` | collapse a dimension (sum, product, max, ...) |
| Contraction | `linalg.matmul` (and friends) | the named matrix multiply |
| Shape | `linalg.broadcast` | stretch a smaller operand to a larger shape |

Because the structure is explicit, lowering is just *generating the loop nest
the maps imply*: `--convert-linalg-to-loops` (to `scf`) or
`--convert-linalg-to-affine-loops` (to `affine`). From there it is the same
pipeline as Chapter 2. [Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md#the-linalg-dialect-loops-as-data)
dissects the same encoding from the transformation side (how the iteration
space is inferred from shapes, and why "loops as data" makes tiling easy);
this chapter stays on the user's side and writes the ops.

### Tensors vs. memrefs at this level

Linalg ops work on both **tensors** (immutable SSA values) and **memrefs**
(buffers). The rule is the one from Chapter 2:

- A linalg op on memrefs writes its caller-provided `outs` buffer *in place*.
  It needs no allocation and lowers straight to loops. That is why the
  runnable named ops in Steps 1-4 are written on memrefs and skip
  bufferization.
- A linalg op on tensors *returns* a new tensor. Before it can run it must be
  **bufferized** (tensors → memrefs) with `--one-shot-bufferize`. The
  `linalg.generic` and `linalg.map` forms inspected in Steps 1-2, the fused
  kernel in Step 5 and the matmul in Step 6 all go through that step. Step 5
  returns a tensor from a function we call from Python, so it also needs
  `--buffer-results-to-out-params` to turn the returned buffer into an
  in-place output argument.

Same ops, different data model. Transformations such as fusion prefer
tensors (value semantics make producer/consumer chains explicit); calling
compiled code from C or Python wants memrefs.

## Step 1: Elementwise — `generic`, `map`, named ops

**Goal:** meet the fundamental `linalg.generic`, then watch it collapse to a
one-liner.

[`generic_add.mlir`](examples/01-elementwise/generic_add.mlir) writes an
elementwise tensor add **the long way**: three identity `indexing_maps`, one
`parallel` iterator, and a region that does the `arith.addf`. It is verbose on
purpose; every other linalg op is this op with the boilerplate removed.

```mlir
#map_1d = affine_map<(d0) -> (d0)>

func.func @add_tensors(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %result = tensor.empty() : tensor<10xf32>
  %0 = linalg.generic {
      indexing_maps  = [#map_1d, #map_1d, #map_1d],   // arg0, arg1, result
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
    outs(%result : tensor<10xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
      %1 = arith.addf %a, %b : f32
      linalg.yield %1 : f32
    } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
```

**Reading the region.** The block `^bb0(%a, %b, %out)` has **one argument per
operand**: one for each `ins()` value *and* one for each `outs()` value, in
order:

| Block arg | Comes from | Role |
| --- | --- | --- |
| `%a` | `ins` `%arg0` | current element of `arg0` |
| `%b` | `ins` `%arg1` | current element of `arg1` |
| `%out` | `outs` `%result` | current value of the output element |

Two things trip people up:

- **`%out` looks unused, and that is correct.** `%out` is the *current* value
  of the output element. It matters when you accumulate into it (the running
  sum of a reduction in Step 2, the `%acc` of matmul in Step 3). A plain
  elementwise write never reads the old value, so `%out` is unused. It must
  still be declared, because the block's argument count has to match the
  operand count.
- **The result is written by `linalg.yield`, not by assigning to `%out`.**
  You compute a *fresh* value `%1 = arith.addf %a, %b`, and `linalg.yield %1`
  stores it into the output element. You cannot write
  `%out = arith.addf %a, %b`: `%out` is already defined (it is a block
  argument) and SSA values are immutable. `mlir-opt` rejects it:

  ```text
  error: redefinition of SSA value '%out'
      %out = arith.addf %x, %y : f32
      ^
  ```

  The flow is always: read the arguments → compute a new value → `yield` it.
  The yield is the store back to the `outs` element.

"Structured" means the loops are *implied*, not written. Ask the compiler to
make them concrete and the maps become a plain loop nest, with one catch:
`--convert-linalg-to-loops` only lowers linalg ops on **memrefs**. This op
works on `tensor<10xf32>`, so we **bufferize first** to give the loops memory
to load from and store into:

```bash
mlir-opt generic_add.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops
```

Run `--convert-linalg-to-loops` *without* the bufferize step and nothing
happens: the tensor op round-trips unchanged. With it, `build.sh` saves
`build/generic_add_loops.mlir`:

```mlir
module {
  func.func @add_tensors(%arg0: memref<10xf32, strided<[?], offset: ?>>, %arg1: memref<10xf32, strided<[?], offset: ?>>) -> memref<10xf32> {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    scf.for %arg2 = %c0 to %c10 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<10xf32, strided<[?], offset: ?>>
      %1 = memref.load %arg1[%arg2] : memref<10xf32, strided<[?], offset: ?>>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%arg2] : memref<10xf32>
    }
    return %alloc : memref<10xf32>
  }
}
```

This is exactly the elementwise `scf.for` you would write by hand. The
identity maps became the `memref.load`/`memref.store` at index `i`, the
region's `arith.addf` sits in the loop body, `%out` was the element at
`%alloc[i]`, and `linalg.yield` became the `memref.store` into it. The
`tensor.empty` became the `memref.alloc`. The `strided<[?], offset: ?>`
layouts on the arguments are bufferization being cautious: a function
argument could be any view of a buffer (Chapter 4's descriptor), so it does
not assume contiguity.

### One rung of sugar up: `linalg.map`

For a *pure elementwise* op all of that structure is implied (every operand
is indexed `(d0) -> (d0)` and every dimension is `parallel`), so you can drop
it and give only the scalar payload. `linalg.map` is that shorthand
([`map_add.mlir`](examples/01-elementwise/map_add.mlir)):

```mlir
func.func @map_add(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %init = tensor.empty() : tensor<10xf32>
  %mapped = linalg.map { arith.addf }
      ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
      outs(%init : tensor<10xf32>)
  return %mapped : tensor<10xf32>
}
```

No `indexing_maps`, no `iterator_types`, no explicit region: `{ arith.addf }`
*is* the payload. Its operands are the `ins()` elements in order (`%arg0[i]`,
`%arg1[i]`), and its result is stored to the `outs()` element, the same
read → compute → store flow the generic region spelled out. It is still on
tensors, so it needs the same bufferize step, and it produces the *identical*
loop nest. `build.sh` saves it to `build/map_add_loops.mlir`; the only
difference from `build/generic_add_loops.mlir` is the function name:

```text
$ diff build/generic_add_loops.mlir build/map_add_loops.mlir
2c2
<   func.func @add_tensors(%arg0: memref<10xf32, strided<[?], offset: ?>>, %arg1: memref<10xf32, strided<[?], offset: ?>>) -> memref<10xf32> {
---
>   func.func @map_add(%arg0: memref<10xf32, strided<[?], offset: ?>>, %arg1: memref<10xf32, strided<[?], offset: ?>>) -> memref<10xf32> {
```

### Fully sugared: the named ops

A **named op** is the whole thing as a single line: payload, structure and
all. Elementwise arithmetic has a family of them, each a `linalg.map` with
the payload baked into the name:

| Named op | Payload | NumPy |
| --- | --- | --- |
| `linalg.add` | `arith.addf` / `arith.addi` | `a + b` |
| `linalg.sub` | `arith.subf` / `arith.subi` | `a - b` |
| `linalg.mul` | `arith.mulf` / `arith.muli` | `a * b` |
| `linalg.div` | `arith.divf` / `arith.divsi` | `a / b` |

The op picks float or integer arithmetic from its operand types; a separate
`linalg.div_unsigned` gives `arith.divui`. There are matching unary families:
`linalg.exp` (`math.exp`), `linalg.log` (`math.log`), `linalg.abs`
(`math.absf`), `linalg.negf` (`arith.negf`), and so on. (All of these
payloads were confirmed by generalizing each op with the wheel.)

[`add.mlir`](examples/01-elementwise/add.mlir) is the runnable version. It
writes into a caller-provided memref, so unlike the tensor forms above it
needs no bufferization:

```mlir
func.func @addv(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.add ins(%a, %b : memref<10xf32>, memref<10xf32>)
             outs(%c : memref<10xf32>)
  return
}
```

You can walk the ladder in both directions with two passes.
`--linalg-generalize-named-ops` rewrites a named op into its generic form
(`build/add_generalized.mlir`):

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @addv(%arg0: memref<10xf32>, %arg1: memref<10xf32>, %arg2: memref<10xf32>) attributes {llvm.emit_c_interface} {
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<10xf32>, memref<10xf32>) outs(%arg2 : memref<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return
  }
}
```

and `--linalg-specialize-generic-ops` recognizes a generic and turns it back
into a named op. Applied to `generic_add.mlir`:

```mlir
module {
  func.func @add_tensors(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }
}
```

Generalizing is how transformations get one uniform form to work on (Step 5
depends on it); specializing is how a pipeline gets named ops back, for
example to hand a recognized matmul to a library (Chapter 29).

**Run:** `bash examples/01-elementwise/build.sh`

```text
linalg.add matches NumPy a + b:
[10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]
```

## Step 2: Reductions — the `reduction` iterator

**Goal:** meet the second iterator type, the one that makes `%out` come
alive.

Every op in Step 1 was `["parallel"]`: iterations independent, `%out` never
read. A **reduction** is the opposite: many inputs collapse into one output,
so each iteration reads the running total in `%out`, combines it with the
next input, and writes it back. This is the accumulation the "unused `%out`"
in Step 1 was hinting at. We sum each row of a 3×4 matrix into a length-3
vector (NumPy `a.sum(axis=1)`):

```text
   input 3×4              collapse dim 1 (columns)        output (3,)
   [ 0  1  2  3]          Σ over each row                 [ 6]
   [ 4  5  6  7]   ───────────────────────────────►       [22]
   [ 8  9 10 11]          i = parallel, j = reduction     [38]
```

The `linalg.generic` form ([`reduce_generic.mlir`](examples/02-reduce/reduce_generic.mlir))
makes both iterators explicit. The output map drops `j`, and that missing
index is what marks the collapsed axis. The region now **reads `%out`**:

```mlir
#map_in  = affine_map<(i, j) -> (i, j)>
#map_out = affine_map<(i, j) -> (i)>

func.func @row_sum(%input: tensor<3x4xf32>, %init: tensor<3xf32>) -> tensor<3xf32> {
  %0 = linalg.generic {
      indexing_maps  = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%input : tensor<3x4xf32>)
    outs(%init : tensor<3xf32>) {
      ^bb0(%in: f32, %out: f32):
      %s = arith.addf %out, %in : f32    // read %out (running sum), accumulate
      linalg.yield %s : f32
    } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
```

(Map dimensions can be named anything; `(i, j)` reads better than the
`(d0, d1)` that `mlir-opt` prints back.) Lowering (bufferize first, since it
is on tensors) makes the accumulation concrete, `build/reduce_generic_loops.mlir`:

```mlir
module {
  func.func @row_sum(%arg0: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3xf32, strided<[?], offset: ?>>) -> memref<3xf32, strided<[?], offset: ?>> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<3x4xf32, strided<[?, ?], offset: ?>>
        %1 = memref.load %arg1[%arg2] : memref<3xf32, strided<[?], offset: ?>>
        %2 = arith.addf %1, %0 : f32
        memref.store %2, %arg1[%arg2] : memref<3xf32, strided<[?], offset: ?>>
      }
    }
    return %arg1 : memref<3xf32, strided<[?], offset: ?>>
  }
}
```

The outer `scf.for` is the `parallel` row loop; the inner one is the
`reduction` sweep over columns. Its body loads `%arg1[i]` (the running sum,
that is `%out`), adds the element, and stores it back. Because the output
index does not depend on the inner loop variable, every inner iteration hits
the same element. Notice also that no buffer was allocated this time:
bufferization wrote the result into `%init`'s buffer in place and returned
it.

The **named op** is one line ([`reduce.mlir`](examples/02-reduce/reduce.mlir)).
`dimensions = [1]` names the axis to collapse. Because it accumulates into the
caller's `%init` buffer in place, that buffer must arrive **pre-zeroed** (the
driver passes `np.zeros`), just like matmul's `C`:

```mlir
func.func @row_sum(%input: memref<3x4xf32>, %init: memref<3xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.reduce { arith.addf }
      ins(%input : memref<3x4xf32>)
      outs(%init : memref<3xf32>)
      dimensions = [1]
  return
}
```

`--linalg-generalize-named-ops` confirms it is the same op
(`build/reduce_generalized.mlir`): maps `(d0, d1) -> (d0, d1)` and
`(d0, d1) -> (d0)`, iterators `["parallel", "reduction"]`, and a body
`arith.addf %out, %in`. Swap the payload for `arith.mulf` (product),
`arith.maximumf` (max), and so on; the structure is identical, only the
combine op changes.

**Run:** `bash examples/02-reduce/build.sh`

```text
linalg.reduce matches NumPy matrix.sum(axis=1):
[ 6. 22. 38.]
```

With `parallel` (Step 1) and `reduction` (here) in hand, matmul is just the
two combined.

## Step 3: `linalg.matmul`, and `scf` vs. `affine` loops

**Goal:** a whole matrix multiply as one op, and the two ways to turn it into
loops. (Chapter 1 introduced `linalg.matmul` and why it needs an initialized
`outs`; Chapter 2 lowered it stage by stage. This step is deliberately
brief.)

[`matmul.mlir`](examples/03-matmul/matmul.mlir):

```mlir
func.func @matmul(%A: memref<8x10xf32>, %B: memref<10x16xf32>, %C: memref<8x16xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.matmul ins(%A, %B : memref<8x10xf32>, memref<10x16xf32>)
                outs(%C : memref<8x16xf32>)
  return
}
```

That single line is sugar for the `linalg.generic` whose maps and iterators
encode $C_{ij} = \sum_k A_{ik}B_{kj}$, a contraction where two indices are
free and one is summed away:

```text
        k                j                    j
      ┌────┐          ┌──────┐             ┌──────┐
   i  │ A  │    ×   k │  B   │     =    i  │  C   │
      └────┘          └──────┘             └──────┘
     (i,k)            (k,j)                (i,j)

   iterator_types: i = parallel   j = parallel   k = reduction
   C[i,j] = Σ_k A[i,k]·B[k,j]   — sweep k, accumulate; i,j independent
```

Written out ([`matmul_generic.mlir`](examples/03-matmul/matmul_generic.mlir)):

```mlir
#map_A = affine_map<(i, j, k) -> (i, k)>
#map_B = affine_map<(i, j, k) -> (k, j)>
#map_C = affine_map<(i, j, k) -> (i, j)>

func.func @matmul(%A: memref<8x10xf32>, %B: memref<10x16xf32>, %C: memref<8x16xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.generic {
      indexing_maps  = [#map_A, #map_B, #map_C],
      iterator_types = ["parallel", "parallel", "reduction"]
    }
    ins(%A, %B : memref<8x10xf32>, memref<10x16xf32>)
    outs(%C : memref<8x16xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %acc, %p : f32    // read %acc (current C[i,j]), accumulate
      linalg.yield %s : f32
    }
  return
}
```

This is Step 1's `parallel` and Step 2's `reduction` in one op: `i, j` are
the parallel output dimensions, `k` is the reduced one (absent from `C`'s
map), and `%acc` is the read-and-write accumulator from the reduction, here
holding the running dot product. `--linalg-specialize-generic-ops` turns this
generic straight back into `linalg.matmul`.

Two passes generate the loop nest; they differ only in which loop dialect
they emit. `build.sh` runs the generic form through both:

```bash
mlir-opt matmul_generic.mlir --convert-linalg-to-loops         # -> scf
mlir-opt matmul_generic.mlir --convert-linalg-to-affine-loops  # -> affine
```

`build/matmul_generic_scf.mlir`:

```mlir
module {
  func.func @matmul(%arg0: memref<8x10xf32>, %arg1: memref<10x16xf32>, %arg2: memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c10 = arith.constant 10 : index
    scf.for %arg3 = %c0 to %c8 step %c1 {
      scf.for %arg4 = %c0 to %c16 step %c1 {
        scf.for %arg5 = %c0 to %c10 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<8x10xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<10x16xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<8x16xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<8x16xf32>
        }
      }
    }
    return
  }
}
```

`build/matmul_generic_affine.mlir`:

```mlir
module {
  func.func @matmul(%arg0: memref<8x10xf32>, %arg1: memref<10x16xf32>, %arg2: memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 16 {
        affine.for %arg5 = 0 to 10 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<8x10xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<10x16xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<8x16xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<8x16xf32>
        }
      }
    }
    return
  }
}
```

Same arithmetic, two dialects. The `scf` form needs explicit `%c0`…`%c16`
index constants for its bounds; the `affine` form writes `0 to 8` directly:

| | `--convert-linalg-to-loops` (`scf`) | `--convert-linalg-to-affine-loops` (`affine`) |
| --- | --- | --- |
| Loop op | `scf.for` (bounds are SSA values: `%c0` to `%c8`) | `affine.for %i = 0 to 8` (bounds are affine expressions) |
| Memory | `memref.load` / `memref.store` | `affine.load` / `affine.store` |
| Constraints | none: bounds and indices can be *any* values | bounds and subscripts must be **affine** in the loop variables |
| Analyzability | opaque to the polyhedral model | fully analyzable: tiling, fusion, interchange, vectorization |
| Level | **lower** (closer to raw imperative loops) | **higher** (carries structure for optimization) |

**Which is "higher"?** `affine` is the more abstract form: it adds the
promise that every bound and subscript is an affine function of the loop
variables, which is exactly what the polyhedral transforms of Chapters 5 and
6 need. `scf` is more general and lower-level: it can express loops `affine`
cannot (data-dependent bounds, arbitrary indexing), and that generality is
what stops the optimizer from reasoning about it. The ladder runs
**linalg → affine → scf → cf → llvm**; `affine` sits *above* `scf` and lowers
to it with `--lower-affine`.

**When to use each:**

- **`--convert-linalg-to-affine-loops`** when you still want to *optimize the
  loop nest*: tile, fuse, interchange or vectorize it. Step 6 uses it because
  `--affine-loop-tile` only works on `affine.for`; Chapter 5 used it for its
  "affine pit stop". It is also how you confirm the op *is* Chapter 6's
  matmul: this `affine.for` nest is the one you wrote by hand there.
- **`--convert-linalg-to-loops`** when you are done optimizing and just want
  runnable loops on the way to LLVM (or when an op's access pattern is not
  affine). That is why every runnable step here takes the `scf` route.

One line upstream, but by the time it is loops it is exactly what you would
have hand-written, and the compiler was free to choose the order. The driver
checks the result against NumPy's `@` (the kernel accumulates into `C`, so
`C` is passed zeroed).

**Run:** `bash examples/03-matmul/build.sh`

```text
linalg.matmul matches NumPy A @ B (8x10 @ 10x16, max abs error 4.77e-07)
```

The inputs come from a seeded generator, so this exact error reproduces. It
is float rounding: the kernel and NumPy sum the `k` dimension in different
orders.

## Step 4: Broadcasting

**Goal:** NumPy-style broadcasting, combining operands of different shapes.

Broadcasting lets you combine tensors of *different* shapes in one
operation, following NumPy's rules. The smaller operand is implicitly
stretched along the missing dimensions, so you can add a vector to every row
of a matrix, or fold a scalar into a whole tensor, without writing out the
expanded copy. In NumPy, the stretch alone is `np.broadcast_to`:

```python
arg0 = np.array([1, 2, 3])
result = np.broadcast_to(arg0, (3, 3))
# array([[1, 2, 3],
#        [1, 2, 3],
#        [1, 2, 3]])
```

That is what `linalg.broadcast` does. Adding a length-4 vector to every row
of a 3×4 matrix is `matrix + vector` in NumPy; the shapes do not match, so
something has to *stretch* the vector over the missing dimension:

```text
   vector (4,)        broadcast dim 0       stretched (3,4)     matrix (3,4)    result (3,4)
   [1 2 3 4]   ────────────────────────►    [1 2 3 4]      +    [1 1 1 1]   =   [2 3 4 5]
                 replicate across           [1 2 3 4]           [1 1 1 1]       [2 3 4 5]
                 the 3 rows                 [1 2 3 4]           [1 1 1 1]       [2 3 4 5]
```

### Two named ops

In linalg that is `linalg.broadcast` (stretch the vector across the rows)
followed by an elementwise add, both writing the caller's `%out` buffer in
place ([`add_vec_to_mat.mlir`](examples/04-broadcast/add_vec_to_mat.mlir)):

```mlir
func.func @add_vec_to_mat(%matrix: memref<3x4xf32>, %vector: memref<4xf32>, %out: memref<3x4xf32>)
    attributes {llvm.emit_c_interface} {
  // 1. broadcast the vector across the 3 rows of %out
  linalg.broadcast ins(%vector : memref<4xf32>)
                   outs(%out : memref<3x4xf32>)
                   dimensions = [0]
  // 2. add the matrix elementwise into %out (in place)
  linalg.add ins(%matrix, %out : memref<3x4xf32>, memref<3x4xf32>)
             outs(%out : memref<3x4xf32>)
  return
}
```

`dimensions = [0]` names the dimension the *input lacks*, dimension 0 (the
rows), so the vector is replicated along it. Note that `%out` is both an
input and the output of the `linalg.add`: on memrefs that is fine, each
element is read before it is written.

### What a broadcast really is: an indexing map

Generalize the pair (`build/add_vec_to_mat_generalized.mlir`) and the
broadcast turns out to be nothing but an indexing map:

```mlir
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_vec_to_mat(%arg0: memref<3x4xf32>, %arg1: memref<4xf32>, %arg2: memref<3x4xf32>) attributes {llvm.emit_c_interface} {
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : memref<4xf32>) outs(%arg2 : memref<3x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : memref<3x4xf32>, memref<3x4xf32>) outs(%arg2 : memref<3x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return
  }
}
```

The vector's map `(d0, d1) -> (d1)` *ignores* `d0`: whichever row the loop is
on, it reads `vector[j]`. Compare Step 2, where it was the **output** map
that dropped a dimension (a reduction); here an **input** map drops one (a
broadcast), and both iterators stay `parallel` because nothing is summed.

That means the copy into `%out` is unnecessary. One generic can read the
vector through the broadcasting map and add in a single pass
([`add_vec_to_mat_generic.mlir`](examples/04-broadcast/add_vec_to_mat_generic.mlir)):

```mlir
#mat = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d1)>

func.func @add_vec_to_mat_generic(%matrix: memref<3x4xf32>, %vector: memref<4xf32>,
                                  %out: memref<3x4xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.generic {
      indexing_maps  = [#mat, #row, #mat],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%matrix, %vector : memref<3x4xf32>, memref<4xf32>)
    outs(%out : memref<3x4xf32>) {
  ^bb0(%m: f32, %v: f32, %o: f32):
    %s = arith.addf %m, %v : f32
    linalg.yield %s : f32
  }
  return
}
```

Step 5 shows that the fusion pass derives exactly this generic from the two
named ops by itself. The driver runs both versions and checks them against
NumPy's own broadcasting (`matrix + vector`).

**Run:** `bash examples/04-broadcast/build.sh`

```text
linalg.broadcast + linalg.add matches NumPy matrix + vector:
[[2. 3. 4. 5.]
 [2. 3. 4. 5.]
 [2. 3. 4. 5.]]
single linalg.generic with a (d0, d1) -> (d1) map: same result
```

## Step 5: Elementwise fusion with passes

**Goal:** the optimization that makes structured ops pay off.

[`separate_ops.mlir`](examples/05-fusion/separate_ops.mlir) computes
`(a + b) * c` as **two** named ops on tensors. Run as written, that is two
loops: the add materializes a whole intermediate to memory, then the mul
reads it back.

```mlir
module {
  // This is the unfused version, which we will fuse with a pass.
  func.func @addmul(%a: tensor<10xf32>, %b: tensor<10xf32>, %c: tensor<10xf32>) -> tensor<10xf32>
      attributes {llvm.emit_c_interface} {
    // 1. Create an empty tensor to hold the intermediate result of the addition.
    %0 = tensor.empty() : tensor<10xf32>

    // 2. Add the two input tensors and store the result in the intermediate tensor.
    %1 = linalg.add ins(%a, %b : tensor<10xf32>, tensor<10xf32>)
                    outs(%0 : tensor<10xf32>) -> tensor<10xf32>

    // 3. Create an empty tensor to hold the final result.
    %2 = tensor.empty() : tensor<10xf32>

    // 4. Multiply the intermediate result by the third input tensor and store the result.
    %3 = linalg.mul ins(%1, %c : tensor<10xf32>, tensor<10xf32>)
                    outs(%2 : tensor<10xf32>) -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}
```

**Fusion** merges the two into a single loop that reads each input once and
writes the result directly: no intermediate, no second pass over memory.

```text
   separate                              fused
   ┌────────────────┐                    ┌───────────────────────┐
   │ for i: t[i]=a+b│  writes t          │ for i:                │
   └────────────────┘  ───► memory ───►  │   r[i] = (a[i]+b[i])  │
   ┌────────────────┐  reads t           │          * c[i]       │
   │ for i: r[i]=t*c│                    │  (t never exists)     │
   └────────────────┘                    └───────────────────────┘
   two loops + temp array                one loop, no temp
```

The fusion pipeline (`build.sh` writes the result to `build/fused_ops.mlir`):

```bash
mlir-opt separate_ops.mlir \
    --canonicalize \
    --linalg-fuse-elementwise-ops \
    --cse \
    --linalg-generalize-named-ops \
    --linalg-fuse-elementwise-ops
```

| Pass | Role |
| --- | --- |
| `--canonicalize` | Standard cleanup: fold constants, drop dead ops, put the IR in a normal form so later patterns match reliably. |
| `--linalg-fuse-elementwise-ops` | The fusion pass. It merges producer → consumer elementwise ops, **but only `linalg.generic` ops**, so on the named `add`/`mul` it does nothing yet. |
| `--cse` | Common-subexpression elimination: merges the two identical `tensor.empty()` ops into one. |
| `--linalg-generalize-named-ops` | Rewrites `linalg.add` / `linalg.mul` into their `linalg.generic` form, the form fusion understands. |
| `--linalg-fuse-elementwise-ops` | Run **again**: now both ops are generic, and this is the run that fuses them. |

The order is the point: fusion works on `linalg.generic`, so the named ops
must be *generalized* first. After the first three passes the IR still holds
two named ops (CSE has only shared the `tensor.empty`):

```mlir
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
    %2 = linalg.mul ins(%1, %arg2 : tensor<10xf32>, tensor<10xf32>) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
```

After the last two, `build/fused_ops.mlir`:

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @addmul(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<10xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) outs(%0 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.mulf %2, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }
}
```

The two named ops have become **one `linalg.generic`** whose body does the
`addf` and the `mulf` before a single `linalg.yield`. It has four
`indexing_maps` (`a`, `b`, `c` and the output, all `(d0) -> (d0)`), and the
intermediate `t` is gone: its value only exists as `%2` inside the body.
(Only `--linalg-generalize-named-ops --linalg-fuse-elementwise-ops` is
strictly needed; on this input those two passes alone give the same result.
The cleanup passes matter on bigger graphs.)

This is the bread-and-butter optimization for deep-learning graphs, which
are mostly long chains of elementwise ops. It is the linalg cousin of Chapter
5's `--affine-loop-fusion`, which fused *loops*; this fuses *ops*, before any
loop exists. And it happens **on tensors, before bufferization**: the
intermediate is fused away while it is still an SSA value, so no buffer is
ever allocated for it.

Fusion also understands broadcasts. Feed it Step 4's two named ops, written
on tensors ([`broadcast_add.mlir`](examples/05-fusion/broadcast_add.mlir)),
and the `linalg.broadcast` disappears into the add's indexing maps
(`build/broadcast_add_fused.mlir`):

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @add_vec_to_mat(%arg0: tensor<3x4xf32>, %arg1: tensor<4xf32>) -> tensor<3x4xf32> {
    %0 = tensor.empty() : tensor<3x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<4xf32>) outs(%0 : tensor<3x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<3x4xf32>
    return %1 : tensor<3x4xf32>
  }
}
```

This is the generic we wrote by hand in Step 4: the compiler found the
`(d0, d1) -> (d1)` map on its own.

### Running the fused kernel

Fusion is an IR-to-IR transform, but we can prove it computes the right
thing by compiling it. The fused function *returns* a tensor, so it needs
bufferizing, and the returned buffer must become an in-place `out` argument
so that the C interface is `addmul(a, b, c, out)` like the other steps:

```bash
mlir-opt build/fused_ops.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-results-to-out-params="modify-public-functions=true hoist-static-allocs=true" \
    --canonicalize \
    -o build/fused_memref.mlir
```

- `--one-shot-bufferize` turns the tensors into memrefs. The result becomes a
  freshly allocated `memref<10xf32>` that the function returns.
- `--buffer-results-to-out-params` rewrites the signature so that returned
  memref becomes a trailing argument. Two options matter with the current
  wheel. `modify-public-functions=true` is required because `@addmul` is
  public: without it the pass leaves the function alone (see the drift
  notes). `hoist-static-allocs=true` makes the kernel write straight into the
  caller's buffer; without it the kernel computes into its own `memref.alloc`
  and then `memref.copy`s it into the out-parameter, and the allocation is
  never freed.

`build/fused_memref.mlir`:

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @addmul(%arg0: memref<10xf32, strided<[?], offset: ?>>, %arg1: memref<10xf32, strided<[?], offset: ?>>, %arg2: memref<10xf32, strided<[?], offset: ?>>, %arg3: memref<10xf32>) attributes {llvm.emit_c_interface} {
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : memref<10xf32, strided<[?], offset: ?>>, memref<10xf32, strided<[?], offset: ?>>, memref<10xf32, strided<[?], offset: ?>>) outs(%arg3 : memref<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      %1 = arith.mulf %0, %in_1 : f32
      linalg.yield %1 : f32
    }
    return
  }
}
```

and its loop form, `build/fused_loops.mlir`, reads `a[i]`, `b[i]`, `c[i]`
once each and writes the result, exactly the fused nest from the diagram:

```mlir
    scf.for %arg4 = %c0 to %c10 step %c1 {
      %0 = memref.load %arg0[%arg4] : memref<10xf32, strided<[?], offset: ?>>
      %1 = memref.load %arg1[%arg4] : memref<10xf32, strided<[?], offset: ?>>
      %2 = memref.load %arg2[%arg4] : memref<10xf32, strided<[?], offset: ?>>
      %3 = arith.addf %0, %1 : f32
      %4 = arith.mulf %3, %2 : f32
      memref.store %4, %arg3[%arg4] : memref<10xf32>
    }
```

For contrast, `build.sh` also pushes the *unfused* `separate_ops.mlir`
through the same bufferization (`build/unfused_loops.mlir`): two loops and a
temporary buffer for `a + b`:

```mlir
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    scf.for %arg4 = %c0 to %c10 step %c1 {
      %0 = memref.load %arg0[%arg4] : memref<10xf32, strided<[?], offset: ?>>
      %1 = memref.load %arg1[%arg4] : memref<10xf32, strided<[?], offset: ?>>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%arg4] : memref<10xf32>
    }
    scf.for %arg4 = %c0 to %c10 step %c1 {
      %0 = memref.load %alloc[%arg4] : memref<10xf32>
      %1 = memref.load %arg2[%arg4] : memref<10xf32, strided<[?], offset: ?>>
      %2 = arith.mulf %0, %1 : f32
      memref.store %2, %arg3[%arg4] : memref<10xf32>
    }
```

**Run:** `bash examples/05-fusion/build.sh`

```text
fused kernel matches NumPy (a + b) * c:
  a   = [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
  b   = [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
  c   = [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
  out = [  2.   6.  12.  20.  30.  42.  56.  72.  90. 110.]
```

Reading the first column: `(a[0] + b[0]) * c[0] = (0 + 2) * 1 = 2`; the
last: `(9 + 2) * 10 = 110`.

### Passes vs. the transform dialect

`--linalg-fuse-elementwise-ops` is a whole-module "hammer": it fuses every
elementwise producer/consumer pair it can find, by its own heuristics, and
only that one kind of fusion. [Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md#example-3-tile-and-fuse)
does fusion the other way: a transform-dialect script *tiles* a consumer and
then fuses named producers (a fill and a matmul, not just elementwise ops)
into the tile loop with `transform.structured.fuse_into_containing_op`. The
two are complementary: elementwise fusion merges whole ops into one bigger
generic before any loop exists (good for long elementwise chains), and
tile-and-fuse keeps ops separate but computes them tile by tile (needed when
a contraction such as matmul is involved, since it cannot be merged into an
elementwise body).

## Step 6: Tiling with passes

**Goal:** restructure a matmul for cache locality. This step is brief:
Chapter 5 explains cache tiling in depth, and Chapter 22 tiles linalg ops
directly with the transform dialect.

On modern hardware memory access, not arithmetic, is usually the bottleneck.
When a computation sweeps a large matrix linearly it keeps evicting and
re-fetching data. **Tiling** breaks the computation into small **blocks**
that fit in cache, so each block is fully reused before the next one is
loaded. This matters even more on **GPUs**, whose memory hierarchy (global
memory, shared memory, registers) is explicit: the classic CUDA pattern loads
a tile into shared memory, synchronizes, and has each thread work inside that
tile (Chapters 21 and 31).

We start from a one-line `linalg.matmul` on **tensors**
([`matmul_tile.mlir`](examples/06-tiling/matmul_tile.mlir)):

```mlir
module {
  func.func @matmul(%a: tensor<10x10xf32>, %b: tensor<10x10xf32>,
                    %c: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<10x10xf32>, tensor<10x10xf32>)
                       outs(%c : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
}
```

The pipeline bufferizes it, lowers to affine loops, then tiles:

```bash
mlir-opt matmul_tile.mlir \
    --convert-tensor-to-linalg \
    --linalg-generalize-named-ops \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-deallocation-pipeline \
    --convert-bufferization-to-memref \
    --convert-linalg-to-affine-loops \
    --affine-loop-tile="tile-size=5" \
    --canonicalize --cse
```

| Pass | Role |
| --- | --- |
| `--convert-tensor-to-linalg` | Rewrites remaining `tensor` ops (e.g. `tensor.pad`) into linalg form. A no-op on this input; kept so the pipeline also handles inputs that have them. |
| `--linalg-generalize-named-ops` | `linalg.matmul` → `linalg.generic`. |
| `--one-shot-bufferize="bufferize-function-boundaries"` | Tensors → memrefs, including the function's arguments and result (needed because the function *returns* a tensor). |
| `--buffer-deallocation-pipeline` | Makes buffer ownership explicit and inserts deallocations so nothing leaks. |
| `--convert-bufferization-to-memref` | Lowers leftover `bufferization` ops (here a `bufferization.clone`) into `memref` ops. |
| `--convert-linalg-to-affine-loops` | Lowers the generic to an `affine.for` nest; the **affine** form, because tiling needs analyzable loops (Step 3). |
| `--affine-loop-tile="tile-size=5"` | The tiling: splits each loop into a *tile* loop (step 5) and a *point* loop, turning the 3-deep nest into 6. |
| `--canonicalize --cse` | Cleanup. |

The order is the point: generalize → bufferize → **affine** loops → tile.
`--affine-loop-tile` cannot touch an `scf.for` nest. `build.sh` writes
`build/matmul_tiled.mlir`:

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 5)>
module {
  func.func @matmul(%arg0: memref<10x10xf32, strided<[?, ?], offset: ?>>, %arg1: memref<10x10xf32, strided<[?, ?], offset: ?>>, %arg2: memref<10x10xf32, strided<[?, ?], offset: ?>>) -> memref<10x10xf32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 10 step 5 {
      affine.for %arg4 = 0 to 10 step 5 {
        affine.for %arg5 = 0 to 10 step 5 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<10x10xf32, strided<[?, ?], offset: ?>>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<10x10xf32, strided<[?, ?], offset: ?>>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<10x10xf32, strided<[?, ?], offset: ?>>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                affine.store %4, %arg2[%arg6, %arg7] : memref<10x10xf32, strided<[?, ?], offset: ?>>
              }
            }
          }
        }
      }
    }
    %alloc = memref.alloc() : memref<10x10xf32>
    %cast = memref.cast %alloc : memref<10x10xf32> to memref<10x10xf32, strided<[?, ?], offset: ?>>
    memref.copy %arg2, %alloc : memref<10x10xf32, strided<[?, ?], offset: ?>> to memref<10x10xf32>
    return %cast : memref<10x10xf32, strided<[?, ?], offset: ?>>
  }
}
```

The 3-loop matmul became a **6-deep nest**: three outer *tile* loops stepping
by 5 over blocks, and three inner *point* loops (bounds `#map = (d0)->(d0)`
to `#map1 = (d0)->(d0+5)`) walking the 5 elements inside each tile. It is the
tile/point split of Chapter 5's `--affine-loop-tile`, applied to a nest the
compiler generated from one `linalg.matmul`:

```text
   before: sweep the whole            after: 5×5×5 blocks, one at a time
   10×10×10 space
   for i in 0..10                     for i0 in 0..10 step 5      ┐
     for j in 0..10                     for j0 in 0..10 step 5    │ tile loops
       for k in 0..10                     for k0 in 0..10 step 5  ┘ (step 5)
         C[i,j] += A[i,k]*B[k,j]            for i in i0..i0+5     ┐
                                              for j in j0..j0+5   │ point loops
                                                for k in k0..k0+5 ┘ (walk the tile)
                                                  C[i,j] += A[i,k]*B[k,j]
```

Two things in the output are bufferization's doing, not tiling's:

- The `strided<[?, ?], offset: ?>` types: bufferization cannot assume the
  layout of a buffer passed across a function boundary, so it works through a
  generic strided view. The `?`s are the runtime strides of Chapter 4's
  descriptor.
- The trailing `alloc` + `copy`: one-shot bufferization computed the matmul
  in place in `%c`'s buffer (`%arg2`) and returned that argument. The
  deallocation pipeline then saw a function returning a buffer it does not
  own, and inserted a `bufferization.clone` so the caller receives a buffer
  it may free; `--convert-bufferization-to-memref` lowered that clone to
  `memref.alloc` + `memref.copy`. Drop `--buffer-deallocation-pipeline` and
  the function simply returns `%arg2`.

Because linalg keeps the op abstract until lowering, the compiler picks the
tile structure from a single `linalg.matmul`. The pass route has limits,
though: `--affine-loop-tile` tiles every affine nest in the module with the
same sizes, and it only works after you have given up the linalg op.
Chapter 22's `transform.structured.tile_using_for` tiles a *chosen* op with
*chosen* per-dimension sizes and keeps a smaller linalg op inside the loops,
so you can keep transforming it (fuse, vectorize).

**Run:** `bash examples/06-tiling/build.sh` (then read
`build/matmul_tiled.mlir`).

## Tests

The key results are pinned by lit tests in [`test/part1/`](../../test/part1/):

| Test | Checks |
| --- | --- |
| `ch08_elementwise_ladder.mlir` | generic and map lower to the identical loop; no loops without bufferization; `linalg.add` generalizes, the generic specializes back |
| `ch08_reduce.mlir` | the parallel/reduction loop nest; `linalg.reduce` generalizes to the `(d0, d1) -> (d0)` output map |
| `ch08_matmul_loops.mlir` | the `scf` and `affine` nests; the generic specializes to `linalg.matmul` |
| `ch08_broadcast.mlir` | the `(d0, d1) -> (d1)` broadcast map; both broadcast kernels compiled and run from a C driver |
| `ch08_fusion.mlir` | no fusion on named ops; the fused 4-map generic; broadcast folded into maps; the out-param signature; the fused kernel compiled and run from C |
| `ch08_tiling.mlir` | the 6-deep tiled nest |

The runtime tests use small C drivers
([`bcast_main.c`](examples/04-broadcast/bcast_main.c),
[`addmul_main.c`](examples/05-fusion/addmul_main.c)) instead of the Python
drivers, because lit's `python3` may not have NumPy. Run them with:

```bash
.venv/bin/lit build/test -v --filter ch08_
```

## Exercises

1. **Another elementwise rung.** Write `(a - b) * a` three ways: as one
   `linalg.generic`, as `linalg.map`s, and as named `linalg.sub` +
   `linalg.mul` on tensors. Run all three through
   `--linalg-generalize-named-ops --linalg-fuse-elementwise-ops` and confirm
   they end up as the same single generic. (Hint: `--linalg-fuse-elementwise-ops`
   alone leaves `linalg.map` untouched too; `--linalg-generalize-named-ops`
   rewrites it into a generic just like the named ops.)
2. **Column sums and max.** Change [`reduce.mlir`](examples/02-reduce/reduce.mlir)
   to sum over `dimensions = [0]` (NumPy `a.sum(axis=0)`), fixing the output
   shape, and update `run.py`. Then make a row-max with `arith.maximumf`:
   what must the driver now pass as the initial `out` instead of zeros?
3. **Forget to zero.** In [`02-reduce/run.py`](examples/02-reduce/run.py),
   initialize `out` with `np.ones(3)` instead of zeros and explain the
   result from the loop nest in `build/reduce_generic_loops.mlir`.
4. **Transposed matmul by maps.** Copy
   [`matmul_generic.mlir`](examples/03-matmul/matmul_generic.mlir) and change
   only `#map_B` so the kernel computes `A @ B.T` with `B` given as a 16×10
   memref. Compile it with `compile_kernel` and check against NumPy. Does
   `--linalg-specialize-generic-ops` still recognize it, and as which op?
5. **Column broadcast.** Add a length-3 vector to every *column* of the 3×4
   matrix (NumPy `matrix + vector[:, None]`). Write it once with
   `linalg.broadcast` (which `dimensions`?) and once as a single generic
   (which map?), and extend `run.py` to check both.
6. **Fusion meets reduction.** Append a row-sum `linalg.reduce` to
   `separate_ops.mlir`'s output (make the inputs 2-D first). Run the fusion
   pipeline: does the elementwise generic fuse into the reduction? Compare
   what Chapter 22's `fuse_into_containing_op` would do with the same chain.

## Notes on reproducing with current MLIR

The source chapter was written against Homebrew LLVM 20.1.8. Rerunning it
with the MLIR wheel (LLVM 24):

- **No `llc`.** The source compiled with `llc -filetype=obj` and then linked
  with `clang -shared`. The wheel ships no `llc`, so `compile_kernel` hands
  the LLVM IR straight to `clang -shared -fPIC -x ir`, after
  `tools/ll_fp_compat.py` rewrites LLVM 24's float-literal syntax for the
  older clang (Chapter 3) and strips LLVM 24-only attributes such as
  `nocreateundeforpoison` that clang 20 cannot parse. None of these kernels
  contain float constants or math intrinsics, so the filter changes nothing
  here, but it keeps `compile_kernel` safe for the exercises (e.g. a row-max
  with `arith.maximumf` declares `@llvm.maximum.f32`).
- **`--buffer-results-to-out-params` skips public functions by default.**
  With LLVM 20 the source's plain `-buffer-results-to-out-params` turned the
  returned tensor into a fourth argument. With the wheel the same command
  silently does nothing: `@addmul` still returns a `memref<10xf32>`, and
  calling it through a four-argument C signature would be wrong. The pass now
  has a `modify-public-functions` option (default off); Step 5 sets it, plus
  `hoist-static-allocs=true` to avoid an internal allocation and copy.
- **The `window` iterator type is gone.** The source's iterator table listed
  `parallel`, `reduction` and `window`; `mlir-opt` now rejects
  `"window"` (`unexpected iterator_type (window)`). Convolution windows are
  expressed as `reduction` dimensions with `d1 + d4`-style input maps.
- **Fusion keeps the C-interface attribute.** The source's
  `build/fused_ops.mlir` listing showed `@addmul` without attributes; the
  wheel keeps `attributes {llvm.emit_c_interface}` through fusion (which is
  what the runnable step needs anyway).
- **New: `--linalg-specialize-generic-ops`.** Not used in the source; the
  wheel can turn generics back into named ops (`linalg.add`,
  `linalg.matmul`), which this chapter uses to show the ladder in both
  directions.
- Everything else reproduced unchanged: all loop nests in Steps 1-3, the
  fused generic, and the tiled nest (including the trailing alloc + copy) are
  identical to the LLVM 20 outputs, and every runnable check passes. The
  matmul driver now uses a seeded generator, so its error figure is
  reproducible (`4.77e-07`).
- The tiling step's `alloc` + `copy` explanation differs from the source,
  which attributed it to bufferization in general. Running the pipeline
  without `--buffer-deallocation-pipeline` shows it comes from the
  deallocation pipeline's `bufferization.clone` of the returned argument.

## Where to go next

- **[Chapter 9: Testing with lit and FileCheck](../ch09-testing-with-lit/README.md)**:
  how the `ch08_*.mlir` tests above work, and how to write your own.
- **[Chapter 22: Linalg and the Transform Dialect](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)**:
  the same structured ops, scheduled precisely: tile a chosen op, fuse
  producers into its tile loop, and vectorize.
- **[Chapter 25: Neural Networks from Scratch](../../part6-end-to-end-ml/ch25-neural-nets-from-scratch/README.md)**:
  chains of these ops (matmul, broadcast bias add, elementwise activations,
  reductions) are a neural network.
