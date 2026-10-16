# Chapter 4: Memory: Tensors, Memrefs and the C ABI

> Adapted from mlir-ir-lowering's `2_memory` chapter, itself based on Stephen
> Diehl's "MLIR Part 2: Memory in MLIR" article. Every command, listing and
> output below was re-run with this repo's toolchain: the MLIR wheel (LLVM 24),
> its Python bindings, and Homebrew clang 20 on Apple Silicon.

*Where does a number live?* In mathematics, nowhere. The entry A<sub>ij</sub>
of a matrix is a pure value with no address, no lifetime and no notion of being
overwritten. In hardware, the exact opposite holds: a number is nothing until
it is *somewhere*, as bytes at an address in a register, a cache line or a page
of RAM. The whole performance story of a program (copies, alignment, locality,
aliasing) is the story of where its data lives and when.

[Chapter 3](../ch03-llvm-ir-and-native-code/README.md) walked *computation* down
the compiler staircase: a loop became branches, then phi nodes, then registers.
Every value there was a scalar that lived in a register and evaporated. This
chapter covers the other half of the descent: the **data**. Real programs pass
arrays, and an array is precisely the thing that has to live somewhere. MLIR
gives data its own ladder, parallel to the one for code, and the rungs are
distinguished by one question: *does this value have an address yet?*

## What you will learn

- MLIR's layered memory model: `tensor` (a value with no address) → `memref`
  (an addressable but target-independent buffer) → LLVM structs, arrays and
  pointers, plus `vector` for SIMD registers.
- The `tensor` type system and its core ops, and why a pure tensor cannot be
  executed.
- Bufferization: what `one-shot-bufferize` does, why it sometimes copies, and
  how destination-passing style lets it write straight into a caller's buffer.
- The memref descriptor (allocated pointer, aligned pointer, offset, sizes,
  strides), how it relates to NumPy's array layout, and how
  [`tools/np_memref.py`](../../tools/np_memref.py) passes NumPy arrays to
  compiled kernels without copying.
- Calling kernels ahead of time through `ctypes`, and just in time through the
  MLIR `ExecutionEngine`.
- LLVM structs and arrays, and why passing them *by value* does not follow the
  C ABI.
- The `vector` dialect by hand, down to NEON instructions.
- What `llvm.emit_c_interface` and `--llvm-request-c-wrappers` generate, field
  by field, on a 2-D broadcast kernel called from Python and from C.

## Prerequisites

- [Chapter 2](../ch02-lowering-pipelines/README.md): bufferization and
  `--finalize-memref-to-llvm` were introduced there in passing.
- [Chapter 3](../ch03-llvm-ir-and-native-code/README.md): turning a module into a
  shared library with `mlir-translate`, `tools/ll_fp_compat.py` and `clang`,
  calling it with `ctypes`, and JIT-compiling with `ExecutionEngine`.
- The repo's `.venv` (it has NumPy and the MLIR Python bindings) and a `clang`
  on your `PATH`. Every `build.sh` finds the wheel's tools itself and runs from
  any directory.

## MLIR's layered memory model

MLIR's memory architecture balances **high-level abstraction** (easy to write,
easy to optimize) against **low-level control** (precise layout and access).
Instead of forcing a choice, it lets you start at whatever altitude suits the
problem and lower progressively toward the metal. It is the same philosophy
you saw for control flow in Chapter 3, applied to data. There are three
altitudes, and at each one the question is: does this value have an address
yet, and how hardware-specific is that address?

- **Top: `tensor`, a number that lives nowhere.** A pure mathematical value:
  immutable, with a shape and an element type but **no location**. Matrix math
  wants to live here. A value with no address cannot be secretly mutated or
  aliased, so the compiler can freely reorder, fuse and optimize it.
- **Middle: `memref`, a number with a home, but an abstract one.** A concrete
  buffer (pointer, offset, sizes, strides), so it finally *has* an address that
  the CPU and NumPy can reach. It is still a high-level, **target-independent**
  description, though. It carries rich metadata (dynamic dimensions, strided
  layouts) and knows nothing about registers or a specific instruction set. Data
  becomes *addressable* here without yet becoming *hardware*, which is why a
  memref is not the bottom.
- **Bottom: `llvm` structs, arrays and pointers, and `vector`: raw machine
  storage.** The metadata is gone. An `!llvm.struct` or `!llvm.array` is
  exactly a C aggregate (fixed size, no shape or strides), and a
  `vector<8xf32>` is a SIMD register. A memref descriptor is ultimately
  *built out of* these once it finishes lowering.

```text
   more abstract  ┌───────────────────────────────────────────────┐  "lives nowhere"
        ▲         │ tensor<?x?xf32>     a value: shape, no address │  (SSA, immutable)
        │         └───────────────────────────────────────────────┘
        │            │  bufferization   ← the one big crossing (top → middle)
        │         ┌───────────────────────────────────────────────┐  "has a home,
        │         │ memref<?x?xf32>     ptr+offset+sizes+strides   │   but abstract"
        │         └───────────────────────────────────────────────┘  (addressable)
        │            │  --finalize-memref-to-llvm / --convert-vector-to-llvm
        ▼         ┌───────────────────────────────────────────────┐  "raw storage"
   more concrete  │ !llvm.struct/array, !llvm.ptr, vector<8xf32>   │  (C aggregate,
                  └───────────────────────────────────────────────┘   SIMD register)
```

| Layer | What it is | Mutable? | Has an address? | Hardware-specific? | Dialect |
| --- | --- | --- | --- | --- | --- |
| **Tensor** | A pure value (a NumPy array *as a value*) | No (SSA) | No | No | `tensor` |
| **MemRef** | A buffer: pointer + sizes + strides | Yes | Yes | No (still abstract) | `memref` |
| **LLVM struct/array/ptr** | C-level aggregates and raw pointers, no metadata | Yes (through pointers) | Yes | Yes | `llvm` |
| **Vector** | A SIMD register of N lanes | No (SSA) | No (registers) | Yes | `vector` |

The most important crossing on this ladder is the one between the top and the
middle: turning an address-less tensor *value* into an addressable memref
*buffer*. It is called **bufferization**, and it is the data-world equivalent
of Chapter 3's lowering passes. Everything below memref is further
concretization. Two bridges tie the layers together, and this chapter uses
each before dissecting it:

- **Bufferization** (tensor → memref): used in step 1, dissected in step 3.
- **The C interface** (`llvm.emit_c_interface`), which makes a memref kernel
  callable from C and Python: used in steps 1–3, dissected in step 4.

The practical payoff: once data reaches the memref rung, handing arrays back
and forth with **NumPy costs zero copies**. A NumPy array and a memref are,
underneath, the same handful of bytes describing a pointer (step 2).

## The examples

Each step has a directory under [`examples/`](examples/) with a `build.sh` that
compiles everything and runs it:

| Step | Directory | Topic | Runs |
| --- | --- | --- | --- |
| 1 | [`01-tensor/`](examples/01-tensor/) | Tensor types and ops; three identity-matrix kernels | Python |
| 2 | [`02-array-add/`](examples/02-array-add/) | Memrefs and NumPy: AOT (ctypes) and JIT (ExecutionEngine) | Python |
| 3 | [`03-low-level-llvm/`](examples/03-low-level-llvm/) | LLVM structs and arrays, vectors, bufferization, C interface, a SIMD kernel | C and Python |
| 4 | [`04-c-wrappers/`](examples/04-c-wrappers/) | `emit_c_interface` vs `--llvm-request-c-wrappers` on a 2-D broadcast | Python and C |

The shared NumPy ↔ memref helper is
[`tools/np_memref.py`](../../tools/np_memref.py), explained in step 2. The source
tutorial marked several files as inspect-only snippets. Here every snippet
except the pure-tensor catalog is also compiled and executed, because running
code is what exposed the ABI pitfall in step 3.

---

## Step 1: Tensors (`01-tensor/`)

**Goal:** understand the tensor, MLIR's highest-level data type, and why
running one forces it down into a memref.

### What a tensor is

A tensor is an **immutable, side-effect-free value**. It has a shape and an
element type but **no memory address**: you cannot take a pointer to it, alias
it or mutate it. "Updating" an element produces a *new* tensor. This value
semantics is what makes tensors easy to analyze and reorder, and why they are
the natural representation for linear algebra and ML.

### The tensor type system

Shapes use `x` as the delimiter between dimensions and the element type. A
number is a **static** dimension (fixed at compile time); `?` is a **dynamic**
dimension (known only at run time). `!Name = ...` defines a type alias:

```mlir
!TensorStatic = tensor<10x10x10x10xf32>   // fully static 4-D
!TensorDyn    = tensor<?x?x10x10xf32>      // first two dims dynamic, last two static
```

### The core tensor ops

| Op | Meaning |
| --- | --- |
| `tensor.empty(%d0, %d1)` | An uninitialized tensor; one operand per `?` dimension |
| `tensor.from_elements %a, %b, …` | A small tensor literal built from SSA values |
| `tensor.extract %t[%i, %j]` | Read one element (a pure read) |
| `tensor.insert %x into %t[%i, %j]` | A **new** tensor with one element changed |
| `tensor.generate %m, %n { … }` | Build a tensor by running a region once per index tuple |
| `tensor.yield %v` | Inside `generate`: the value of the current element |

[`tensor_ops.mlir`](examples/01-tensor/tensor_ops.mlir) is a commented catalog
of these ops (`@make`, `@access`). It is inspect-only: a pure tensor has no
memory, so there is nothing to execute. `build.sh` round-trips it through
`mlir-opt` and then canonicalizes it. Canonicalization deletes the two unused
`tensor.empty` ops from `@make` as dead code. Because tensors have no side
effects, an unused tensor op can always be removed:

```mlir
  func.func @make(%arg0: index, %arg1: index, %arg2: f32) -> tensor<3xf32> {
    %from_elements = tensor.from_elements %arg2, %arg2, %arg2 : tensor<3xf32>
    return %from_elements : tensor<3xf32>
  }
```

### Why running a tensor needs bufferization

Because a tensor has no address, you cannot hand one to C or Python, or even
execute it. To run a tensor program you must **bufferize** it: rewrite the
value-semantic tensor ops into pointer-semantic memref ops backed by real
buffers. The crossing is the whole point, not an incidental detail. The pass
is **one-shot bufferization**. With `bufferize-function-boundaries=true` it
also rewrites function *signatures*, so a `tensor<?x?xi32>` argument or result
becomes a `memref<?x?xi32>` that C can pass.

### Three identity-matrix kernels

The example is the **identity matrix**, the simplest computation that is truly
index-dependent:

A<sub>ij</sub> = 1 if i = j, else 0

All three kernels compute it and differ only in how the result crosses the C
boundary, i.e. their **calling convention**.

**[`identity_return.mlir`](examples/01-tensor/identity_return.mlir): return
the tensor (callee allocates).** This is the article's form: take the
dimensions, return a fresh tensor.

```mlir
func.func @identity(%m : index, %n : index) -> tensor<?x?xi32> attributes {llvm.emit_c_interface} {
  %out = tensor.generate %m, %n {
  ^bb0(%i : index, %j : index):                  // region runs for every (i, j)
    %ni = arith.index_cast %i : index to i32      // index -> i32 so we can compare
    %nj = arith.index_cast %j : index to i32
    %eq = arith.cmpi eq, %ni, %nj : i32           // i == j ?  -> i1 (1 on the diagonal)
    %v  = arith.extui %eq : i1 to i32             // widen i1 to i32 (1 or 0)
    tensor.yield %v : i32                          // value of this (i, j) element
  } : tensor<?x?xi32>
  return %out : tensor<?x?xi32>
}
```

`tensor.generate` shows what "side-effect-free" means: you declare *what each
element is*, not *how to write it into memory*. After one-shot-bufferize
(`build/identity_return_bufferized.mlir`), the result has become a heap
allocation that the function returns:

```mlir
  func.func @identity(%arg0: index, %arg1: index) -> memref<?x?xi32> attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc(%arg0, %arg1) {alignment = 64 : i64} : memref<?x?xi32>
    linalg.map outs(%alloc : memref<?x?xi32>)
      (%init: i32) {
        %0 = linalg.index 0 : index
        %1 = linalg.index 1 : index
        ...
        linalg.yield %5 : i32
      }
    ...
    return %alloc : memref<?x?xi32>
  }
```

The `tensor.generate` became a `linalg.map` that writes into the new buffer.
The caller now owns a `malloc`ed buffer and must `free` it.

**[`identity_fill.mlir`](examples/01-tensor/identity_fill.mlir): fill the
caller's buffer (destination-passing).** The function takes the output as a
`memref<?x?xi32>` and returns nothing:

```mlir
  func.func @identity(%out: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %out, %c0 : memref<?x?xi32>
    %n = memref.dim %out, %c1 : memref<?x?xi32>

    %t = tensor.generate %m, %n {
      ^bb0(%i: index, %j: index):
        ...
        tensor.yield %v : i32
    } : tensor<?x?xi32>

    bufferization.materialize_in_destination %t in writable %out
      : (tensor<?x?xi32>, memref<?x?xi32>) -> ()
    return
  }
```

The dimensions are no longer arguments; `memref.dim` reads them from the
buffer. `bufferization.materialize_in_destination` says that the value `%t`
must end up in `%out`. The source tutorial expected this to write in place.
With today's MLIR, the bufferized output says otherwise:

```mlir
  func.func @identity(%arg0: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
    ...
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xi32>
    linalg.map outs(%alloc : memref<?x?xi32>)
      ...
    memref.copy %alloc, %arg0 : memref<?x?xi32> to memref<?x?xi32>
    return
  }
```

`tensor.generate` has no *destination operand*: nothing in the op says where
its result should be built. So bufferization allocates a temporary, fills it,
and copies it into `%out`. Notice also that the temporary is never freed:
**one-shot-bufferize inserts no deallocations**. `build.sh` therefore runs
`--buffer-deallocation-pipeline` next, which adds the missing
`memref.dealloc %alloc` after the copy (lowered to a call to `free`).

**[`identity_fill_inplace.mlir`](examples/01-tensor/identity_fill_inplace.mlir):
destination-passing done right.** This variant is new in this port. It builds
the result with an op that *does* take a destination:

```mlir
func.func @identity(%out: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
  %dest = bufferization.to_tensor %out restrict writable
      : memref<?x?xi32> to tensor<?x?xi32>
  %t = linalg.map outs(%dest : tensor<?x?xi32>) (%init: i32) {
    %i = linalg.index 0 : index                  // current row
    %j = linalg.index 1 : index                  // current column
    %eq = arith.cmpi eq, %i, %j : index          // i == j ?
    %v = arith.extui %eq : i1 to i32             // 1 on the diagonal, else 0
    linalg.yield %v : i32
  }
  bufferization.materialize_in_destination %t in restrict writable %out
      : (tensor<?x?xi32>, memref<?x?xi32>) -> ()
  return
}
```

`to_tensor ... restrict writable` promises that this tensor *is* `%out`, that
nothing else aliases it, and that it may be overwritten. `linalg.map outs(...)`
names that tensor as the place the result lives. One-shot-bufferize can now
write directly into the caller's buffer:

```mlir
  func.func @identity(%arg0: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
    linalg.map outs(%arg0 : memref<?x?xi32>)
      (%init: i32) {
        ...
      }
    memref.copy %arg0, %arg0 : memref<?x?xi32> to memref<?x?xi32>
    return
  }
```

The remaining self-copy is removed by `--canonicalize`. The compiled LLVM IR of
this kernel contains no `malloc`, `memcpy` or `free` at all, while
`identity_fill`'s contains all three. This is the general lesson of
**destination-passing style**, and the reason linalg ops carry `outs`
operands (Chapter 1's "outs quirk"): an op that names its destination lets
bufferization reuse that buffer.

**Comparison.**

| | `identity_return` | `identity_fill` | `identity_fill_inplace` |
| --- | --- | --- | --- |
| Convention | returns the tensor | destination-passing | destination-passing |
| Signature | `(index, index) -> tensor<?x?xi32>` | `(memref<?x?xi32>) -> ()` | `(memref<?x?xi32>) -> ()` |
| Where `m`, `n` come from | arguments | `memref.dim` on the buffer | the `linalg.map` iteration space |
| Who allocates the result | the callee (`malloc`) | the caller (`np.zeros`) | the caller (`np.zeros`) |
| Hidden temporary + copy | no | **yes** (`alloc` + `memref.copy`) | no |
| Who frees what | caller frees the result | kernel frees its temporary | nothing to free |
| C signature of the wrapper | `void (MemRef2D *result, int64_t m, int64_t n)` | `void (MemRef2D *out)` | `void (MemRef2D *out)` |

Neither convention is better in general; they are the two standard ways MLIR
kernels exchange arrays with a caller. Destination-passing is more common in
real pipelines because the caller controls allocation and copies can be
avoided, *provided* the computation names its destination. Returning a tensor
reads more naturally and matches how you would write the math on paper.

### The pipeline and the Python driver

[`build.sh`](examples/01-tensor/build.sh) lowers each kernel in two stages,
saving the bufferized IR in between:

```bash
mlir-opt identity_fill.mlir \
    --one-shot-bufferize="bufferize-function-boundaries=true" \
    -o build/identity_fill_bufferized.mlir
mlir-opt build/identity_fill_bufferized.mlir \
    --buffer-deallocation-pipeline \
    --canonicalize \
    --convert-linalg-to-loops \
    --convert-bufferization-to-memref \
    --finalize-memref-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o build/identity_fill_llvm.mlir
```

It then runs `mlir-translate`, `ll_fp_compat.py` and `clang -shared` as in
Chapter 3. [`aot_main.py`](examples/01-tensor/aot_main.py) loads all three
libraries and calls them through two helpers, one per convention:

- **`run_fill(lib, m, n)`**: Python allocates `out = np.zeros((m, n))`, wraps
  it in a 2-D descriptor pointing **at NumPy's buffer** (no copy), and passes
  one pointer. The kernel writes into the array Python already holds, so there
  is nothing to read back and nothing to free.
- **`run_return(lib, m, n)`**: the kernel `malloc`s the buffer and writes the
  *descriptor* back through the result out-pointer (a returned memref becomes
  an extra first argument in the C interface; step 4 shows why). Python views
  that memory as a NumPy array, **copies it out**, and only then calls
  `free(result.allocated)`. The copy matters because the view aliases the
  buffer that is about to be freed.

```python
def run_return(lib, m, n):
    fn = lib._mlir_ciface_identity
    fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.c_int64, ctypes.c_int64]
    fn.restype = None
    result = MemRef2D()                        # zero-initialized; callee fills it
    fn(ctypes.byref(result), m, n)
    arr = memref_to_numpy(result, np.int32).copy()   # copy BEFORE freeing
    _libc.free(result.allocated)                     # free the base pointer
    return arr
```

Output:

```
$ bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/build.sh
identity_fill         (caller's buffer, via a temporary)
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
identity_fill_inplace (caller's buffer, in place)
[[1 0 0 0 0]
 ...
identity_return       (returns a tensor)
  returned descriptor: sizes=[5, 5] strides=[5, 1] offset=0 aligned-allocated=32 bytes
[[1 0 0 0 0]
 ...
all identity kernels agree with np.eye
```

Note `aligned-allocated=32 bytes`: the two pointers in the returned
descriptor differ. (The gap depends on the address `malloc` returns, so it
varies between runs and is sometimes 0.) The lowered `@identity` shows why:

```llvm
  %6 = add i64 %5, 64                ; requested bytes + 64 bytes of slack
  %7 = call ptr @malloc(i64 %6)
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 63
  %10 = urem i64 %9, 64
  %11 = sub i64 %9, %10              ; round up to the next multiple of 64
  %12 = inttoptr i64 %11 to ptr
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0    ; allocated
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, ptr %12, 1      ; aligned
```

`memref.alloc {alignment = 64}` over-allocates and rounds the pointer up.
Loads and stores use the **aligned** pointer, but `free` must receive the
**allocated** one, the exact address `malloc` returned. Freeing the aligned
pointer here would crash or corrupt the heap.

---

## Step 2: Memrefs and NumPy, AOT and JIT (`02-array-add/`)

**Goal:** the workhorse pattern. A memref kernel called from Python on NumPy
arrays, with **no copying**.

A **memref** is MLIR's abstraction for a memory buffer, the concrete
counterpart of a tensor. Where a tensor is a value with no address, a memref
*is* the address: a pointer carried together with enough metadata to describe
the buffer's structure. That makes it the bridge between high-level arrays (a
NumPy array, a tensor) and low-level memory operations.

It is **not** a raw `float*`. It is a small struct, the **descriptor**, whose
fields are either *data* (pointers into the buffer) or *metadata* (how to
interpret what the pointers reach):

| Field | Kind | Meaning | Why it exists |
| --- | --- | --- | --- |
| `allocated` | data | Start of the underlying allocation | What you `free()`; may sit *before* the first element |
| `aligned` | data | The aligned pointer that accesses start from | SIMD wants aligned data, so it can differ from `allocated` (step 1) |
| `offset` | metadata | Elements from `aligned` to element 0 | Lets a memref be a *view* into a bigger buffer without copying |
| `sizes[rank]` | metadata | Size of each dimension | Dynamic `?` dimensions get their run-time values here |
| `strides[rank]` | metadata | Elements to step per dimension | Encodes the layout: a transpose or slice is just different strides |

Element `[i, j]` lives at `aligned + offset + i*strides[0] + j*strides[1]`,
counted in elements. So **reshaping, slicing and transposing are metadata
edits, not memory copies**. The layout can also be spelled out in the type.
These three are the same type, increasingly explicit:

```mlir
memref<3x4xf32>                                  // implicit row-major ("identity layout")
memref<3x4xf32, strided<[4, 1], offset: 0>>      // strides written out
memref<3x4xf32, affine_map<(i, j) -> (i, j)>>    // layout as an affine map
```

`mlir-opt` prints them back as `memref<3x4xf32>`, `memref<3x4xf32,
strided<[4, 1]>>` and `memref<3x4xf32>`: the identity map is the default
layout, and `offset: 0` is the default offset. A transposed view of a
row-major 3x4 buffer is `memref<4x3xf32, strided<[1, 4]>>`, with the same
pointer and different strides.

The type also decides **which descriptor fields the compiled code trusts**.
For an identity-layout `memref<?x?xf32>`, the code knows statically that the
innermost stride is 1 and the offset is 0, so it reads only `strides[0]` from
the descriptor. Step 4's kernel computes `aligned + i*strides[0] + j`. Passing
it a transposed descriptor would silently read the wrong elements. Only a
`strided<[?, ?], offset: ?>` type makes the code read every field.

### The kernel

[`array_add.mlir`](examples/02-array-add/array_add.mlir) computes
`c[i] = a[i] + b[i]` over three `memref<1024xf32>` buffers:

```mlir
module {
  func.func @array_add(%arg0: memref<1024xf32>,    // input  a
                       %arg1: memref<1024xf32>,     // input  b
                       %arg2: memref<1024xf32>)     // output c (written in place)
      attributes {llvm.emit_c_interface} {
    // Loop bounds are `index`-typed (platform-sized integers).
    %c0    = arith.constant 0    : index   // start
    %c1024 = arith.constant 1024 : index   // end (exclusive)
    %c1    = arith.constant 1    : index   // step

    // One element per iteration: load a[i] and b[i], add, store into c[i].
    scf.for %arg3 = %c0 to %c1024 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<1024xf32>   // a[i]
      %1 = memref.load %arg1[%arg3] : memref<1024xf32>   // b[i]
      %2 = arith.addf %0, %1 : f32                        // a[i] + b[i]
      memref.store %2, %arg2[%arg3] : memref<1024xf32>   // c[i] = ...
    }
    return
  }
}
```

The `llvm.emit_c_interface` attribute makes the library export a C-callable
`_mlir_ciface_array_add` (step 4 explains it). The one new pass compared with
Chapter 3 is `--finalize-memref-to-llvm`, which turns `memref.load`/`store` and
the descriptor itself into LLVM pointer arithmetic. In the lowered loop
(`build/array_add_llvm.mlir`), each `memref.load` becomes "take field 1
(`aligned`), index it, load":

```mlir
  ^bb2:  // pred: ^bb1
    %23 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %24 = llvm.getelementptr inbounds|nuw %23[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %25 = llvm.load %24 : !llvm.ptr -> f32
```

The static shape `1024` and the identity layout tell the compiler that offset
is 0 and the stride is 1, so neither is read at run time.

### NumPy's array layout

A NumPy array carries the *same* information as a memref descriptor. It is a
thin Python object around a raw buffer plus four pieces of metadata:

- **Data buffer**: the contiguous block of memory holding the values.
- **Shape**: a tuple of dimensions.
- **Strides**: a tuple of steps **in bytes** to move one element along each
  dimension. (NumPy strides are in *bytes*; memref strides are in *elements*.)
- **Dtype**: the element type.

```python
>>> a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
>>> a.shape, a.strides, a.dtype, a.itemsize, a.flags['C_CONTIGUOUS']
((2, 3), (12, 4), dtype('float32'), 4, True)
>>> a.T.shape, a.T.strides, a.T.flags['C_CONTIGUOUS']
((3, 2), (4, 12), False)
```

NumPy arrays are **C-contiguous** (row-major) by default, so element `[i, j]`
lives at `a.ctypes.data + i*strides[0] + j*strides[1]`. That is a memref's
addressing, just counted in bytes. And `a.T` is a transpose done purely by
swapping strides, exactly the metadata trick memrefs use.

### `tools/np_memref.py`: the bridge

To hand a NumPy array to a kernel we need a Python object whose memory layout
matches the descriptor struct exactly. That is
[`tools/np_memref.py`](../../tools/np_memref.py), this chapter's helper, which
every later chapter that runs kernels from Python imports:

```python
def memref_descriptor_type(rank):
    """Return (and cache) the ctypes Structure for a rank-`rank` memref."""
    if rank not in _TYPES:
        _TYPES[rank] = type(
            f"MemRef{rank}D",
            (Structure,),
            {
                "_fields_": [
                    ("allocated", c_void_p),  # base pointer (the one free() takes)
                    ("aligned", c_void_p),  # aligned data pointer
                    ("offset", c_longlong),  # in elements
                    ("sizes", c_longlong * rank),
                    ("strides", c_longlong * rank),
                ]
            },
        )
    return _TYPES[rank]
```

This is the C struct `{ T *allocated; T *aligned; int64_t offset; int64_t
sizes[N]; int64_t strides[N]; }`, i.e. the LLVM type `!llvm.struct<(ptr, ptr,
i64, array<N x i64>, array<N x i64>)>` that `--finalize-memref-to-llvm`
produces. The source tutorial hard-coded a 1-D struct and wrote a second 2-D
one for step 1. Generating the class per rank fixes the classic bug that the
struct must change with the memref rank, or the wrapper reads past the end of
your struct.

The module provides four functions:

- **`numpy_to_memref(arr)`** fills a descriptor of rank `arr.ndim` that points
  at NumPy's buffer: `allocated = aligned = arr.ctypes.data`, `offset = 0`,
  `sizes = arr.shape`, `strides = arr.strides // arr.itemsize` (bytes →
  elements). `allocated` and `aligned` are the same because NumPy gives us one
  buffer and nothing on the MLIR side will ever free it. A non-contiguous array
  is **rejected**, not silently copied. A copy would make the kernel write
  into a temporary instead of the caller's array, and identity-layout kernels
  ignore inner strides anyway.
- **`memref_to_numpy(desc, dtype)`** goes the other way: it views the memory a
  descriptor points at (`aligned + offset`, sizes, strides scaled back to
  bytes) as a NumPy array, without copying. Use `.copy()` if the buffer is
  about to be freed (step 1).
- **`expand(desc)`** returns the descriptor's fields as a flat list,
  `[allocated, aligned, offset, *sizes, *strides]`, for calling a function
  that has **no** C wrapper (step 4).
- **`memref_descriptor_type(rank)`** gives the class itself, for `argtypes`
  and for result descriptors that the callee fills in.

```text
   NumPy ndarray            ONE buffer in RAM            memref descriptor
   ┌──────────────┐        ┌───────────────────┐        ┌───────────────────┐
   │ .ctypes.data ├───────►│ f32 f32 f32 … f32  │◄───────┤ allocated, aligned │
   │ .shape (N,)  │        └───────────────────┘        │ sizes[0] = N       │
   │ .strides (4,)│         (no copy: both               │ strides[0] = 1     │
   └──────────────┘          describe the same bytes)    │ offset = 0         │
                                                         └───────────────────┘
```

Because both sides describe the same bytes, the kernel writes its results
**directly into the array Python is holding**. That is the reason MLIR is a
practical back end for the NumPy and ML ecosystem.

### AOT: ctypes on a prebuilt library

[`aot_main.py`](examples/02-array-add/aot_main.py):

```python
lib = ctypes.CDLL(os.path.join(HERE, "build", f"libarray_add.{SO}"))

# The C-interface wrapper takes one descriptor pointer per memref:
#   void _mlir_ciface_array_add(MemRef1D *a, MemRef1D *b, MemRef1D *c);
array_add = lib._mlir_ciface_array_add
array_add.argtypes = [ctypes.POINTER(MemRef1D)] * 3
array_add.restype = None

a = np.ones(size, dtype=np.float32)
b = np.ones(size, dtype=np.float32) * 2
c = np.zeros(size, dtype=np.float32)

# Descriptors point straight at NumPy's buffers (no copy).
a_desc, b_desc, c_desc = (numpy_to_memref(x) for x in (a, b, c))
array_add(ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(c_desc))
```

### JIT: ExecutionEngine

The source's `jit_main.py` shelled out to `mlir-opt` and `mlir-translate` and
JIT-compiled the resulting LLVM IR with **llvmlite**. The port,
[`jit_main.py`](examples/02-array-add/jit_main.py), keeps everything in-process
with the MLIR Python bindings from the wheel, as in Chapter 3:

```python
PIPELINE = (
    "builtin.module("
    "convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,"
    "convert-func-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)"
)

def compile_and_load(mlir_path):
    with Context():
        module = Module.parse(open(mlir_path).read())
        PassManager.parse(PIPELINE).run(module.operation)
        return ExecutionEngine(module, opt_level=3)

engine = compile_and_load(os.path.join(HERE, "array_add.mlir"))
...
arg = lambda d: ctypes.pointer(ctypes.pointer(d))
engine.invoke("array_add", arg(a_desc), arg(b_desc), arg(c_desc))
```

`engine.invoke(name, ...)` looks up `_mlir_ciface_<name>` (so the function
needs a C interface) and calls it through its *packed* wrapper, which takes
an array of pointers to the arguments. Each C-interface argument is already
a pointer to a descriptor, hence the double `ctypes.pointer`.

```
$ bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/02-array-add/build.sh
python aot_main.py
Array addition successful!
First few elements: [3. 3. 3. 3. 3.]
python jit_main.py
Array addition successful!
First few elements: [3. 3. 3. 3. 3.]
```

(`a = 1`, `b = 2` everywhere, so `c = 3`.)

> **AOT vs JIT is a real trade-off, not a style choice.** Both call the
> identical compiled kernel; they differ in *when* compilation happens.
>
> - **AOT** compiles once at build time and ships a library. It is simpler,
>   needs no compiler at run time, and suits production: distribute the
>   `.dylib`/`.so` and `ctypes.CDLL` it.
> - **JIT** compiles when the program runs. The MLIR can be **generated or
>   modified at run time**, there is **no separate library** to build, ship
>   and version, and code can be **optimized for the exact host CPU**.
>   The costs are **compilation time at startup**, a **heavier runtime
>   dependency** (here the MLIR Python bindings; in the source, the MLIR
>   command-line tools plus llvmlite), and **compilation errors at run time**
>   instead of build time (`PassManager.run` raises `MLIRError`).

---

## Step 3: Below the memref (`03-low-level-llvm/`)

**Goal:** the C-level building blocks *beneath* memref and tensor. This is the
bottom rung of the memory ladder, where the comfortable abstractions run out
and you see what everything above is built from.

| File | Shows |
| --- | --- |
| [`structs_arrays.mlir`](examples/03-low-level-llvm/structs_arrays.mlir) | `!llvm.struct<(i32, f32)>`, `!llvm.array<N x T>`, `insertvalue` / `extractvalue`, and the by-value ABI pitfall |
| [`vectors.mlir`](examples/03-low-level-llvm/vectors.mlir) | `vector<4xf32>`: `arith` elementwise ops, `vector.extract` / `shuffle` / `reduction` / `broadcast` |
| [`bufferization.mlir`](examples/03-low-level-llvm/bufferization.mlir) | Tensor → memref three ways: manual, `--one-shot-bufferize`, `materialize_in_destination` |
| [`c_interface.mlir`](examples/03-low-level-llvm/c_interface.mlir) | The `_mlir_ciface_` wrapper, previewed |
| [`array_add_vec.mlir`](examples/03-low-level-llvm/array_add_vec.mlir) | Step 2's kernel processing 8 floats per iteration |

[`build.sh`](examples/03-low-level-llvm/build.sh) compiles the struct and vector
files and calls every function from [`driver.c`](examples/03-low-level-llvm/driver.c),
writes the bufferization and C-interface outputs to `build/`, and runs the SIMD
kernel from Python:

```
$ bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/build.sh
== structs_arrays: llvm-dialect structs and arrays ==
== vectors: the vector dialect ==
make_pair_into(&p) = {42, 3.14}
pair_second(&p)    = 3.14
array_demo()       = 3
pair_array_first_a = 7
vadd(a, b)   = [11, 22, 33, 44]
vmul(a, b)   = [10, 40, 90, 160]
vextract(b)  = 10
vshuffle     = [1, 2, 10, 20]
vreduce_sum  = 100
vsplat(7)    = [7, 7, 7, 7]
== bufferization: tensor -> memref ==
wrote build/bufferization_{canonical,oneshot,oneshot_identity}.mlir
== c_interface: the _mlir_ciface_ wrapper ==
wrote build/c_interface_llvm.mlir and build/c_interface.ll
== array_add_vec: runnable SIMD kernel ==
Vectorized array addition successful!
First few elements: [2. 3. 4. 5. 6.]
Last element: 1025.0
```

### LLVM structs and arrays: the C floor

In the `llvm` dialect you get exactly C's aggregate types. Both are **rigid**
(fixed size known at compile time, with **no** shape, offset or stride
metadata, unlike a memref) and both are **values**. The dialect is SSA, so you
never mutate one in place: `llvm.insertvalue` builds a *new* aggregate and
`llvm.extractvalue` reads a field.

**Structs.** `!llvm.struct<(i32, f32)>` is the C struct

```c
struct { int a; float b; };
```

in LLVM-dialect form:

```mlir
  llvm.func @make_pair() -> !llvm.struct<(i32, f32)> {
    // Zero-initialized struct, then fill each field.
    %z  = llvm.mlir.zero     : !llvm.struct<(i32, f32)>
    %a  = llvm.mlir.constant(42 : i32) : i32
    %b  = llvm.mlir.constant(3.140000e+00 : f32) : f32
    %s0 = llvm.insertvalue %a, %z [0] : !llvm.struct<(i32, f32)>
    %s1 = llvm.insertvalue %b, %s0[1] : !llvm.struct<(i32, f32)>
    llvm.return %s1 : !llvm.struct<(i32, f32)>
  }
```

**Arrays.** `!llvm.array<N x T>` is fixed-size sequential storage, the rigid
sibling of a memref, like a C array. Element types compose, so arrays of
structs work too:

```mlir
    // Build a constant array {1, 2, 3, 4}.
    %arr = llvm.mlir.constant(dense<[1, 2, 3, 4]> : tensor<4xi32>)
         : !llvm.array<4 x i32>
    // Read element index 2 (== 3).
    %v   = llvm.extractvalue %arr[2] : !llvm.array<4 x i32>
    // Write that value back into slot 0; produces a NEW array value.
    %arr2 = llvm.insertvalue %v, %arr[0] : !llvm.array<4 x i32>
```

A **nested index path** reaches a field inside a struct inside an array:
`llvm.insertvalue %k, %z[0, 0]` writes element 0, field 0 of an
`!llvm.array<2 x !llvm.struct<(i32, f32)>>`. These two types are the raw
material of a memref descriptor, `!llvm.struct<(ptr, ptr, i64, array<N x i64>,
array<N x i64>)>`: a struct whose last two fields are arrays.

The file is already entirely in the `llvm` dialect, so `mlir-translate` takes
it straight to LLVM IR, where `!llvm.struct<(i32, f32)>` becomes `{ i32, float
}`. The translator also constant-folds the array demos into `ret i32 3` and
`ret i32 7`:

```llvm
define { i32, float } @make_pair() {
  ret { i32, float } { i32 42, float 3.140000e+00 }
}

define i32 @pair_first({ i32, float } %0) {
  %2 = extractvalue { i32, float } %0, 0
  ret i32 %2
}

define i32 @array_demo() {
  ret i32 3
}
```

**The by-value ABI pitfall.** The first version of `driver.c` declared
`Pair make_pair(void);` and printed `{42, 0.00}`: the float was lost. The
assembly shows why. LLVM returns the aggregate `{ i32, float }` field by field
in two registers:

```asm
_make_pair:
	mov	w8, #62915
	movk	w8, #16456, lsl #16
	fmov	s0, w8                 ; b in s0
	mov	w0, #42                ; a in w0
	ret
```

A C function returning the same 8-byte struct follows the platform ABI, which
packs both fields into one 64-bit register (clang emits `define i64 @cpair()`
for it):

```asm
_cpair:
	mov	x0, #42
	movk	x0, #62915, lsl #32
	movk	x0, #16456, lsl #48
	ret
```

**LLVM IR aggregates passed or returned by value are not C ABI-compatible.**
Clang performs "ABI lowering" when it compiles C; MLIR's LLVM dialect does not.
What *does* match C is the **in-memory layout**. So `structs_arrays.mlir` adds
two pointer-based functions, `@make_pair_into(ptr)` (store the struct) and
`@pair_second(ptr) -> f32` (load it), and the driver calls only those. This
small bug is the whole reason MLIR's C interface passes memref descriptors
**by pointer** (step 4). Vectors are different: `<4 x float>` and the C type
`float __attribute__((vector_size(16)))` both travel in one 128-bit register,
so the vector functions can be called by value.

### Vectors: SIMD without intrinsics

`vector<4xf32>` is MLIR's SIMD type: a bundle of lanes operated on together.
The point of the `vector` dialect is **portability**: write the operation once
and let LLVM map it onto the target, as one AVX instruction on x86, NEON
instructions on ARM, or a scalar loop on hardware without SIMD.

Elementwise math is plain `arith` on a vector type; there is no `vector.add`.
The lane-manipulation ops live in the `vector` dialect:

| Op | What it does |
| --- | --- |
| `arith.addf` / `arith.mulf` | Elementwise add / multiply across all lanes |
| `vector.extract` | Pull one lane out as a scalar |
| `vector.shuffle` | Pick lanes from two vectors (indices 0..3 = first, 4..7 = second) |
| `vector.reduction` | Horizontally reduce a vector to a scalar |
| `vector.broadcast` | Splat a scalar across every lane |

```mlir
%c    = arith.addf %a, %b : vector<4xf32>                              // elementwise
%lane = vector.extract %v[0] : f32 from vector<4xf32>                  // one lane out
%p    = vector.shuffle %a, %b [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>  // a0,a1,b0,b1
%sum  = vector.reduction <add>, %v : vector<4xf32> into f32            // horizontal sum
%spl  = vector.broadcast %x : f32 to vector<4xf32>                     // splat
```

After `--convert-vector-to-llvm` (plus the arith and func conversions) and
`mlir-translate`, each maps onto a native LLVM vector construct:

```llvm
define <4 x float> @vadd(<4 x float> %0, <4 x float> %1) {
  %3 = fadd <4 x float> %0, %1
  ...
define <4 x float> @vshuffle(<4 x float> %0, <4 x float> %1) {
  %3 = shufflevector <4 x float> %0, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ...
define float @vreduce_sum(<4 x float> %0) {
  %2 = call float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> %0)
  ...
define <4 x float> @vsplat(float %0) {
  %2 = insertelement <4 x float> poison, float %0, i32 0
  %3 = shufflevector <4 x float> %2, <4 x float> poison, <4 x i32> zeroinitializer
```

### Bufferization, dissected

[`bufferization.mlir`](examples/03-low-level-llvm/bufferization.mlir) shows the
tensor → memref crossing three ways:

1. **Manual primitives.** `bufferization.to_buffer` (tensor → memref; called
   `to_memref` in older MLIR) and `bufferization.to_tensor` (memref →
   tensor). `restrict` promises the analysis that no other tensor aliases the
   buffer. A `to_buffer`/`to_tensor` round trip is a no-op, and
   `--canonicalize` folds `@manual` down to `return %arg0 : tensor<10xf32>`.
2. **Automatic, program-wide:** `--one-shot-bufferize`. With
   `bufferize-function-boundaries`, signatures are converted too.
3. **In-place destination:** `bufferization.materialize_in_destination`, the
   usual final step of a kernel that writes into a caller-owned buffer (step 1).

The payoff is in `@insert_scalar`. Before:

```mlir
  func.func @insert_scalar(%arg0: tensor<8xf32>, %idx: index) -> tensor<8xf32> {
    %f = arith.constant 5.000000e+00 : f32
    %r = tensor.insert %f into %arg0[%idx] : tensor<8xf32>
    return %r : tensor<8xf32>
  }
```

After `--one-shot-bufferize="bufferize-function-boundaries"`
(`build/bufferization_oneshot.mlir`):

```mlir
  func.func @insert_scalar(%arg0: memref<8xf32, strided<[?], offset: ?>>, %arg1: index) -> memref<8xf32, strided<[?], offset: ?>> {
    %cst = arith.constant 5.000000e+00 : f32
    memref.store %cst, %arg0[%arg1] : memref<8xf32, strided<[?], offset: ?>>
    return %arg0 : memref<8xf32, strided<[?], offset: ?>>
  }
```

`tensor.insert`, which *logically* creates a brand-new tensor, became a
`memref.store` into the **input buffer**: no copy, no allocation. Tensor-level
immutability costs nothing at run time once the analysis proves the old value
is dead. `@materialize` became a plain `memref.copy %arg0, %arg1`.

Note the signature types. By default, function-boundary bufferization gives
arguments a **fully dynamic layout**, `strided<[?], offset: ?>`, because a
caller could pass any view. That is the safe choice, and it makes the code
read every descriptor field. Adding `function-boundary-type-conversion=identity-layout-map`
gives plain `memref<8xf32>` signatures instead
(`build/bufferization_oneshot_identity.mlir`). That is cheaper and simpler for
C callers, but it promises contiguous inputs. Step 4 uses it.

### The C interface, previewed

[`c_interface.mlir`](examples/03-low-level-llvm/c_interface.mlir) is an
identity function on `memref<3xf32>` carrying `llvm.emit_c_interface`. Lowering
it (`build/c_interface_llvm.mlir`) shows the two functions that step 4
dissects: `@identity`, with the descriptor **unrolled** into five scalar
arguments, and `@_mlir_ciface_identity`, which takes descriptor **pointers**.

### Runnable: vectorized `array_add`

[`array_add_vec.mlir`](examples/03-low-level-llvm/array_add_vec.mlir) is step
2's kernel stepping by 8, with `vector.load` / `vector.store`:

```mlir
    scf.for %i = %c0 to %c1024 step %c8 {
      %va = vector.load %arg0[%i] : memref<1024xf32>, vector<8xf32>
      %vb = vector.load %arg1[%i] : memref<1024xf32>, vector<8xf32>
      %vc = arith.addf %va, %vb : vector<8xf32>
      vector.store %vc, %arg2[%i] : memref<1024xf32>, vector<8xf32>
    }
```

The LLVM IR works on `<8 x float>` (`load <8 x float>`, `fadd <8 x float>`).
ARM64's NEON registers hold 4 floats, so the back end **splits** each 8-wide
operation into two 128-bit halves (`build/array_add_vec.s`, back-end-only
`-O2` as in Chapter 3):

```asm
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	add	x11, x1, x8
	ldp	q1, q0, [x11]          ; a[i..i+7] into two q registers
	add	x11, x6, x8
	ldp	q3, q2, [x11]          ; b[i..i+7]
	fadd.4s	v1, v1, v3             ; two 4-lane adds
	fadd.4s	v0, v0, v2
	add	x11, x10, x8
	stp	q1, q0, [x11]          ; c[i..i+7]
	add	x9, x9, #8             ; i += 8
	add	x8, x8, #32            ; byte offset += 32
	cmp	x9, #1023
	b.le	LBB0_1
```

On x86 with AVX the same IR becomes a single 256-bit `vaddps`. The registers
also show the unrolled descriptors: `x1` and `x6` are the `aligned` fields of
the first two memrefs (arguments 1 and 6). The third memref's `aligned`
pointer is argument 11, past the eight integer argument registers, so it is
loaded from the stack (`ldr x10, [sp, #24]`).

---

## Step 4: C-compatible wrappers (`04-c-wrappers/`)

**Goal:** see what `llvm.emit_c_interface` generates, why it has to exist, and
how `--llvm-request-c-wrappers` differs.

**The problem.** When MLIR lowers a function taking memrefs, it does not pass
each descriptor as one struct. It **unrolls** every memref into its fields as
separate scalar arguments: `ptr, ptr, i64, i64, i64` for a 1-D memref, plus two
more `i64`s per extra dimension. That is efficient for MLIR-to-MLIR calls but
awkward from C or ctypes, where you would hand-pack 5–7 arguments per array.
Returning a memref is worse: it comes back as an LLVM struct *by value*, which
step 3 showed does not follow the C ABI. C wants one pointer to a struct per
array.

`llvm.emit_c_interface` bridges the two by keeping the original function and
emitting a wrapper `_mlir_ciface_<name>` with a C-friendly signature. The
wrapper takes descriptor pointers, loads the structs, extracts the fields,
calls the real function, and stores a returned descriptor through an
out-pointer.

### The kernel: a 2-D broadcast

The source tutorial's `add_vector_to_matrix` was an identity function (its
point was the calling convention), which step 3's `c_interface.mlir` already
covers. Here the name is taken at its word: add a vector to every row of a
matrix, NumPy's `M + v` broadcast. [`add_vector_to_matrix.mlir`](examples/04-c-wrappers/add_vector_to_matrix.mlir)
holds two conventions for the same math:

```mlir
#row = affine_map<(i, j) -> (i, j)>   // M and out: element (i, j)
#bcast = affine_map<(i, j) -> (j)>    // v: the same v[j] for every row i

func.func @add_vector_to_matrix(%M: tensor<?x?xf32>, %v: tensor<?xf32>)
    -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %M, %c0 : tensor<?x?xf32>
  %n = tensor.dim %M, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%m, %n) : tensor<?x?xf32>
  %r = linalg.generic {indexing_maps = [#row, #bcast, #row],
                       iterator_types = ["parallel", "parallel"]}
      ins(%M, %v : tensor<?x?xf32>, tensor<?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %unused: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

func.func @add_vector_to_matrix_into(%M: memref<?x?xf32>, %v: memref<?xf32>,
                                     %out: memref<?x?xf32>) {
  linalg.generic {indexing_maps = [#row, #bcast, #row], ...}
      ins(%M, %v : memref<?x?xf32>, memref<?xf32>)
      outs(%out : memref<?x?xf32>) { ... }
  return
}
```

The broadcast lives entirely in the indexing map `#bcast = (i, j) -> (j)`:
for every point `(i, j)` of the iteration space, `v` is read at `j` alone, so
each row sees the same vector. No data is replicated.

- `@add_vector_to_matrix` returns a new matrix (callee allocates) and has the
  attribute.
- `@add_vector_to_matrix_into` writes into a caller buffer and has **no**
  attribute.

[`build.sh`](examples/04-c-wrappers/build.sh) bufferizes with
`identity-layout-map` (plain `memref<?x?xf32>` signatures), then lowers the
result twice: as written, and with `--llvm-request-c-wrappers`. That pass adds
`llvm.emit_c_interface` to *every* function, and it must run before
`--convert-func-to-llvm`, the pass that actually emits the wrappers. Exported
symbols (Mach-O adds a leading `_`, so the wrapper shows as `__mlir_ciface_…`):

```
exported symbols:
  00000000000005b0 T __mlir_ciface_add_vector_to_matrix
  00000000000003f0 T _add_vector_to_matrix
  000000000000077c T _add_vector_to_matrix_into
with --llvm-request-c-wrappers:
  00000000000005b8 T __mlir_ciface_add_vector_to_matrix
  00000000000008fc T __mlir_ciface_add_vector_to_matrix_into
  00000000000003f8 T _add_vector_to_matrix
  0000000000000784 T _add_vector_to_matrix_into
```

| | `llvm.emit_c_interface` | `--llvm-request-c-wrappers` |
| --- | --- | --- |
| Where | an attribute on a `func.func` in the source | a pass on the pipeline command line |
| Scope | only the functions that carry it | every function in the module |
| Typical use | a hand-written kernel that must be callable from C | compiling a whole module for a C/C++ host (e.g. [Chapter 28](../../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md)) |

### What the wrapper looks like

From `build/add_vector_to_matrix_llvm.mlir`, the signatures:

```mlir
  llvm.func @add_vector_to_matrix(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
  llvm.func @_mlir_ciface_add_vector_to_matrix(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
  llvm.func @add_vector_to_matrix_into(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64) {
```

The raw `@add_vector_to_matrix` takes `M` unrolled as 7 scalars (`%arg0`–`%arg6`:
2 pointers, offset, 2 sizes, 2 strides) and `v` as 5 (`%arg7`–`%arg11`), and
returns the result descriptor **by value**. The wrapper takes three pointers:

```mlir
  llvm.func @_mlir_ciface_add_vector_to_matrix(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    ...
    (call @add_vector_to_matrix with the 12 fields, then
     llvm.store the returned struct through %arg0)
```

So the rules of the C interface are:

1. Every memref argument becomes **one pointer to its descriptor**. The wrapper
   loads the struct and unpacks it: field 0 = allocated, 1 = aligned,
   2 = offset, `[3, k]` = `sizes[k]`, `[4, k]` = `strides[k]`.
2. A memref **result** becomes an extra **first** argument: a pointer to a
   caller-provided descriptor that the wrapper fills (`llvm.store`). The C
   return type becomes `void`. Scalar arguments (like step 1's `index`
   dimensions) stay scalars after it.
3. The original unrolled function is kept, so MLIR-to-MLIR calls stay fast.

In C, the wrapper is just ([`main.c`](examples/04-c-wrappers/main.c)):

```c
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef2D;

void _mlir_ciface_add_vector_to_matrix(MemRef2D *result, MemRef2D *M, MemRef1D *v);
```

### Calling it three ways

[`aot_main.py`](examples/04-c-wrappers/aot_main.py) uses all of
`tools/np_memref.py`:

```python
# 1. Callee allocates. void _mlir_ciface_add_vector_to_matrix(
#        MemRef2D *result, MemRef2D *M, MemRef1D *v);
result = MemRef2D()
fn(ctypes.byref(result), ctypes.byref(m_desc), ctypes.byref(v_desc))
out1 = memref_to_numpy(result, np.float32).copy()  # copy, then free
_libc.free(result.allocated)

# 2. No C wrapper: each memref is expanded into its fields -- allocated,
#    aligned, offset, sizes[rank], strides[rank] -- as separate scalars.
def field_types(rank):
    return [ctypes.c_void_p] * 2 + [ctypes.c_int64] * (1 + 2 * rank)

raw = lib.add_vector_to_matrix_into
raw.argtypes = field_types(2) + field_types(1) + field_types(2)
args = expand(m_desc) + expand(v_desc) + expand(numpy_to_memref(out2))
raw(*args)

# 3. The library built with --llvm-request-c-wrappers has a wrapper for it.
fn = lib_all._mlir_ciface_add_vector_to_matrix_into
fn(ctypes.byref(m_desc), ctypes.byref(v_desc), ctypes.byref(numpy_to_memref(out3)))
```

Calling the unrolled function from C or ctypes is ABI-safe because every
argument is a plain pointer or integer. The danger is only in by-value
*aggregates*, which is why the returning variant is called through its
wrapper. `main.c` makes the same two kinds of call from C. Output:

```
== Python (aot_main.py) ==
  M descriptor: offset=0 sizes=[2, 3] strides=[3, 1]
  v descriptor: offset=0 sizes=[3] strides=[1]
  result descriptor: offset=0 sizes=[2, 3] strides=[3, 1]
1. _mlir_ciface_add_vector_to_matrix (returns a memref):
[[10. 21. 32.]
 [13. 24. 35.]]
2. add_vector_to_matrix_into, 19 scalar arguments:
[[10. 21. 32.]
 [13. 24. 35.]]
3. _mlir_ciface_add_vector_to_matrix_into (--llvm-request-c-wrappers):
[[10. 21. 32.]
 [13. 24. 35.]]
all three match NumPy's M + v
== C (main.c) ==
result: sizes=[2, 3] strides=[3, 1] offset=0
_mlir_ciface_add_vector_to_matrix:
  10.0  21.0  32.0
  13.0  24.0  35.0
add_vector_to_matrix_into (expanded descriptors):
  10.0  21.0  32.0
  13.0  24.0  35.0
```

### Getting the descriptor right

The caller's struct must mirror the LLVM struct field for field. Three
details bite if you get them wrong:

- **`allocated` vs `aligned`.** MLIR separates the pointer to `free` from the
  pointer to access through, because an aligned allocation can hand back a
  different address (step 1 measured a 32-byte gap; the size varies from run
  to run and can be 0). When wrapping NumPy
  memory, set both to the array's data pointer; nothing on the MLIR side
  frees it. When the kernel returns memory, read through `aligned + offset`
  and `free(allocated)`.
- **`offset` and `strides` are in elements, not bytes.** A contiguous float
  vector has `strides[0] == 1`, not 4; a row-major 2x3 matrix has strides
  `[3, 1]`.
- **The struct's size depends on the rank.** A 2-D memref needs `sizes[2]` and
  `strides[2]`. If the MLIR signature's rank changes and the caller's struct
  does not, the wrapper reads garbage past the end of your struct.
  `memref_descriptor_type(rank)` keeps them in sync.

Two pipeline points matter as well. `--finalize-memref-to-llvm` is what turns
the descriptor into the struct and unrolls it; it leaves
`unrealized_conversion_cast`s behind, so it must run before
`--reconcile-unrealized-casts`. And objects destined for a shared library are
compiled with `-fPIC`, because the library is loaded with `dlopen` at an
arbitrary address.

[Chapter 28](../../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md) uses
exactly this ABI from C++ to call a whole compiled PyTorch model.

The results of all four steps are pinned by lit tests:
[`ch04_tensor_bufferization.mlir`](../../test/part1/ch04_tensor_bufferization.mlir),
[`ch04_array_add.mlir`](../../test/part1/ch04_array_add.mlir),
[`ch04_low_level_llvm.mlir`](../../test/part1/ch04_low_level_llvm.mlir) and
[`ch04_c_wrappers.mlir`](../../test/part1/ch04_c_wrappers.mlir), each
compiling the kernels and running them from a C driver.

## Key takeaways

- **The whole chapter is one question: does this value have an address yet?**
  A tensor lives nowhere; a memref has a home. The crossing is bufferization,
  the data-world counterpart of Chapter 3's lowering.
- **Destination-passing decides whether bufferization copies.** An op that
  names its output (`outs`) can write into an existing buffer; one that does
  not (`tensor.generate`) gets a temporary and a copy.
- **One-shot-bufferize does not free.** Run `--buffer-deallocation-pipeline`.
- **A memref is just a struct** (allocated, aligned, offset, sizes, strides).
  Point it at NumPy's buffer and data crosses for free
  (`tools/np_memref.py`). The memref *type* decides which fields the code
  reads.
- **LLVM aggregates by value are not the C ABI.** Exchange structs through
  memory. `llvm.emit_c_interface` / `--llvm-request-c-wrappers` do exactly
  that for memrefs.
- **The `vector` dialect gives portable SIMD.** LLVM splits, maps or
  scalarizes it for the target.

## Exercises

1. **Strides by hand.** Build a descriptor for the transpose of a C-contiguous
   2x3 `float32` array without copying (`sizes = [3, 2]`, `strides = [1, 3]`).
   Pass it to `add_vector_to_matrix_into` with a 2-element `v`, and explain the
   wrong answer from the lowered address computation. Then change the
   kernel's argument type to `memref<?x?xf32, strided<[?, ?], offset: ?>>`,
   rebuild, and check that the transpose now works.
2. **Free the right pointer.** In `01-tensor/aot_main.py`, change `run_return`
   to free `result.aligned` instead of `result.allocated`, and print the gap
   between them. Run it for a few sizes (3x3, 5x5, 7x7). The gap depends on
   where `malloc` happened to put the buffer: when it is 0 the bug is silent,
   and when it is not, the process aborts inside `free`. Why is a bug that
   only *sometimes* crashes the worse kind?
3. **Make the fill in place.** Rewrite `identity_fill.mlir` without
   `tensor.generate`, so that one-shot-bufferize produces no `memref.alloc`.
   (One answer is `identity_fill_inplace.mlir`; try `linalg.generic` with an
   `outs` operand, or `tensor.empty` plus `linalg.map`, and see which
   bufferizes in place.)
4. **Vector width.** Change `array_add_vec.mlir` to `vector<4xf32>` and to
   `vector<16xf32>` (adjusting the step), and compare the ARM64 loops in
   `build/array_add_vec.s`. How many `fadd.4s` per iteration, and why?
5. **A ciface for a scalar result.** Write a function
   `@sum(%a: memref<?xf32>) -> f32` with `llvm.emit_c_interface`, call it from
   Python with `numpy_to_memref`, and read the wrapper's signature: which
   arguments are pointers, and does the scalar result use an out-pointer?
6. **JIT the broadcast.** Port `jit_main.py` to `04-c-wrappers`: run the
   bufferization and lowering pipeline with `PassManager`, then call
   `engine.invoke("add_vector_to_matrix", ...)`. The result descriptor is the
   *first* argument; remember that every argument passed to `invoke` is a
   pointer to the C-interface argument.

## Notes on reproducing with current MLIR

The source chapter targeted Homebrew LLVM 20.1.8, `llc`, and llvmlite. With the
wheel (LLVM 24), Homebrew clang 20, and the MLIR Python bindings:

- **`bufferization.to_memref` is now `bufferization.to_buffer`.** The old
  spelling fails with `custom op 'bufferization.to_memref' is unknown`.
  `bufferization.mlir` uses the new name.
- **`identity_fill` is not in place.** One-shot-bufferize allocates a
  temporary for `tensor.generate` and `memref.copy`s it into the destination,
  and without `--buffer-deallocation-pipeline` the temporary leaks. The
  source described this kernel as copy-free. `identity_fill_inplace.mlir` was
  added to show the in-place version.
- **Function-boundary bufferization defaults to fully dynamic layouts**
  (`memref<8xf32, strided<[?], offset: ?>>`). Pass
  `function-boundary-type-conversion=identity-layout-map` for plain memrefs.
- **llvmlite replaced by `ExecutionEngine`.** `jit_main.py` lowers in-process
  with `PassManager` and calls `engine.invoke`, with no tools invoked at run
  time.
- **`llc` replaced by `clang -x ir`**, with `tools/ll_fp_compat.py` bridging
  LLVM 24's float-literal syntax and the `nocreateundeforpoison` intrinsic
  attribute (which appears here on `llvm.vector.reduce.fadd`).
- **Printer drift:** a struct's float constant now translates to
  `float 3.140000e+00` instead of the hex `0x40091EB860000000`, and
  `mlir-translate` constant-folds the array demos to `ret i32 3` / `ret i32 7`.
  The lowered descriptor code uses `llvm.mlir.poison` where LLVM 20 printed
  `llvm.mlir.undef`.
- **A latent ABI bug in the source:** `structs_arrays.mlir` was inspect-only
  there. Running it from C exposed that by-value structs do not follow the C
  ABI; the pointer-based functions were added.
- **One shared, rank-generic helper:** the source kept a 1-D
  `common/np_memref.py` and a private 2-D struct in step 1. This port uses
  `tools/np_memref.py` everywhere, whose fields are named `sizes`/`strides`
  (the source used `shape`/`stride`).
- **The C-wrapper kernel now computes something.** The source's
  `add_vector_to_matrix` was an identity function on `memref<3xf32>`; that
  version survives as `03-low-level-llvm/c_interface.mlir`. Step 4 uses a real
  2-D broadcast.

## Where to go next

- **[Chapter 5: Optimizing with the Affine Dialect](../ch05-affine-optimization/README.md)**
  optimizes loops over memrefs, with fusion and tiling driven by exact
  knowledge of the access patterns.
- **[Chapter 28](../../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md)**
  comes back to the memref ABI from C++, at the scale of a whole PyTorch
  model.
