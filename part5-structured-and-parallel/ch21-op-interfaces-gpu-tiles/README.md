# Chapter 21: Op Interfaces and a GPU Tile DSL

> Reworked from mlir-tutor's chapter 3 for the unified repo.

Part 4 ended with a working scalar/vector language: the `tiny` dialect, a loop
construct, and a lowering to LLVM. This chapter points the same machinery at a
GPU-shaped problem. We build `tiny_tile`, a miniature **tile-based programming
language** in the style of [Triton](https://triton-lang.org) and NVIDIA's
TileIR: the programmer writes operations on whole *tiles* (statically shaped
2-D arrays) at the level of a **workgroup**, and the compiler lowers them to
**SIMT** code — the per-thread program that actually runs on a GPU, where
`gpu.thread_id` decides which slice of the tile each thread touches.

Along the way this chapter introduces three pieces of MLIR we have not built
before:

- a **parameterized attribute** (`#tiny_tile.layout<...>`) that carries the
  thread-distribution scheme inside the type system,
- a **parameterized type** (`!tiny_tile.tile<HxW, layout>`) with a
  hand-written parser, and
- a custom **op interface** (`TinyTileLoweringOpInterface`), used to drive an
  entire conversion pass with a *single* generic pattern instead of one
  pattern per op.

Back in [Chapter 13](../../part3-building-a-dialect/ch13-traits/README.md) we
*used* upstream traits and interfaces; here we *define* an interface of our
own in ODS and watch what tablegen generates for it.

## What you will learn

- The tile/workgroup programming model and how it relates to SIMT — what
  Triton-style compilers actually do when they "distribute" a tile
- Defining a parameterized attribute with `AttrDef` (parameters,
  `assemblyFormat`, `genVerifyDecl`, `extraClassDeclaration`)
- Defining a parameterized type with `TypeDef`, and when you must abandon the
  declarative `assemblyFormat` for a hand-written `parse`/`print`
  (spoiler: the lexer eats `64x128` as a single token)
- Declaring an `OpInterface` in ODS, attaching it to ops with
  `DeclareOpInterfaceMethods`, and reading the generated
  Concept/Model dispatch code
- Writing an interface-driven dialect conversion: one
  `OpInterfaceConversionPattern` + a `TypeConverter` that maps
  `!tiny_tile.tile<...>` to a per-thread `vector<Nxf16>`
- Lowering workgroup-level ops to per-thread code with `gpu.thread_id`,
  `gpu.subgroup_reduce`, and offset arithmetic derived from the layout

## Prerequisites

- **Chapters 18-20** (Part 4): this chapter lowers *onto* the `tiny` dialect
  from Chapters 18-19 (`tiny.load`/`tiny.store`/`tiny.addf`/`tiny.sum` on
  `vector<Nxf16>` and the `!tiny.ptr` type), and the DSL demo reuses
  Chapter 20's `tiny_loop.accumulate` for its reduction loop.
- **Chapter 13** for the trait/interface mental model — the difference between
  a trait (compile-time mixin) and an interface (runtime-dispatchable API) is
  the heart of this chapter.
- A built `build/tools/tutorial-opt` and the Python venv from the repo README.

All code for this chapter lives in
[`lib/Dialect/TinyTile/`](../../lib/Dialect/TinyTile/):

| File | Contents |
| --- | --- |
| `TinyTileDialect.td` | ODS for the dialect, layout attribute, tile type, and five ops |
| `TinyTileLoweringInterface.td` | The `TinyTileLoweringOpInterface` definition |
| `TinyTileLoweringInterface.h` | Pulls in the tablegen'd interface declarations |
| `TinyTileDialect.cpp` | Verifiers, the tile type parser/printer, and every `convertToSIMT` implementation |
| `TinyTilePasses.td` | The `--tiny-tile-to-tiny` pass declaration |
| `TinyTileToTiny.cpp` | Type converter + interface-driven conversion pass |

Tests are in [`test/part5/`](../../test/part5/): `basic-ops.mlir`
(parse/print round-trips) and `lower-to-tiny.mlir` (the SIMT lowering).

## The tile programming model

A GPU executes thousands of threads grouped into **workgroups** (CUDA: thread
blocks). Writing correct per-thread code — "thread 17 loads elements 136-143"
— is tedious and error-prone, so tile languages raise the altitude: the
programmer writes *one* program per workgroup, operating on tiles, and the
compiler decides how the tile's elements are divided among the workgroup's
threads.

Our language has exactly one aggregate: a 2-D, statically shaped tile of
`f16` values. Five operations work on it:

| Op | Meaning |
| --- | --- |
| `tiny_tile.splat` | Create a tile filled with one constant |
| `tiny_tile.elementwise` | `add`/`sub`/`mul`/`div` two tiles element-wise |
| `tiny_tile.load` | Load a tile from memory at a (row, col) offset |
| `tiny_tile.store` | Store a tile to memory |
| `tiny_tile.sum` | Reduce a whole tile to a `vector<1xf16>` |

Here is the motivating program, a dot product, in the Python DSL (full source
in [`examples/gpu_dot_product.py`](examples/gpu_dot_product.py)):

```python
# 1x256 tile = 256 elements
# 32 threads (1x32), each handles 8 elements
LAYOUT = Layout(thread=(1, 32), vector_size=8)
TILE_H, TILE_W = 1, 256


def dot(a: Ptr, b: Ptr, out: Ptr, M: Index, K: Index):
    # Each workgroup handles one row of the MxK inputs.
    bid = block_id_x()
    tile_w = Index.constant(TILE_W)

    acc_init = Tile.zeros(TILE_H, TILE_W, LAYOUT)

    # March over the K dimension, 256 elements per step.
    @accumulate(K, tile_w, inits=[acc_init])
    def _(tile_idx: Index, acc: Tile):
        a_tile = load_tile(a, bid, tile_idx, K, TILE_H, TILE_W, LAYOUT)
        b_tile = load_tile(b, bid, tile_idx, K, TILE_H, TILE_W, LAYOUT)
        return acc + a_tile * b_tile

    # Reduce the accumulated tile across all 32 threads.
    result = _[0].sum()
    out.store(bid, result)
```

Notice what is *absent*: no thread IDs, no per-thread offsets, no cross-thread
reduction protocol. The program is written as if one processor owned the whole
1x256 tile. The only concession to the hardware is the `LAYOUT` constant —
and that is the next topic.

## The layout attribute

A tile value never materializes as 512 contiguous elements anywhere; it is a
fiction maintained by the compiler. The **layout** records how the fiction
maps to reality: which thread physically holds which elements. We encode it
as a dialect attribute so it can live inside the tile type:

```mlir
#tiny_tile.layout<thread = [16, 4], vector_size = 8>
```

This says: the workgroup is a 16x4 grid of 64 threads, and each thread owns a
contiguous run of 8 elements — so the layout covers 64 * 8 = 512 elements,
exactly one `32x16` tile. Flattening the tile row-major, thread `t` owns
elements `[t * 8, t * 8 + 8)`, where `t` is the linearized thread ID. The
linearization used by the lowering (we will meet it again in
`computeThreadOffset`) is:

```
linear_tid     = tid_y * thread[0] + tid_x
thread_offset  = linear_tid * vector_size
```

The layout is a distribution *contract*, not data — which is why an attribute
(compile-time constant, uniqued in the context) is the right vehicle, just
like `affine_map` or `dense<...>` upstream. In ODS
([`TinyTileDialect.td`](../../lib/Dialect/TinyTile/TinyTileDialect.td)):

```tablegen
def TinyTile_LayoutAttr : AttrDef<TinyTile_Dialect, "Layout"> {
  let mnemonic = "layout";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$thread,
    "int64_t":$vectorSize
  );

  let assemblyFormat = "`<` `thread` `=` `[` $thread `]` `,` `vector_size` `=` $vectorSize `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the total number of threads.
    int64_t getNumThreads() const { ... }

    /// Returns the total number of elements covered by this layout.
    int64_t getNumElements() const {
      return getNumThreads() * getVectorSize();
    }
  }];
}
```

Everything here should look familiar from defining ops, transposed to
attributes:

- `parameters` play the role of `arguments` — tablegen generates a storage
  class, `get(...)` factory, and `getThread()`/`getVectorSize()` accessors.
- `assemblyFormat` gives us the printer and parser for free. This works for
  the attribute (unlike the tile type below) because every token is
  parser-friendly.
- `genVerifyDecl = 1` declares a static `verify` hook we implement in C++.
  Attribute verifiers run at *construction* time — an invalid layout cannot
  exist, even transiently.
- `extraClassDeclaration` splices helper methods into the generated class;
  the lowering calls `getNumElements()` to check tile/layout agreement.

The verifier itself, in
[`TinyTileDialect.cpp`](../../lib/Dialect/TinyTile/TinyTileDialect.cpp),
enforces exactly-2 thread dimensions and powers of two:

```cpp
LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> thread, int64_t vectorSize) {
  // Thread dimensions must be exactly 2.
  if (thread.size() != 2) { ... }

  // Each thread dimension must be a power of 2.
  for (auto [idx, dim] : llvm::enumerate(thread)) {
    if (!llvm::isPowerOf2_64(dim)) {
      return emitError() << "thread dimension " << idx << " must be a power of 2, got "
                         << dim;
    }
  }
  ...
```

and it fires with a proper source location when parsing bad input:

```
$ build/tools/tutorial-opt bad-layout.mlir
bad-layout.mlir:1:23: error: thread dimension 0 must be a power of 2, got 3
#L = #tiny_tile.layout<thread = [3, 4], vector_size = 8>
                      ^
```

## The tile type

With the layout in hand, the tile type is a `TypeDef` with three parameters —
height, width, and the layout attribute itself:

```mlir
!tiny_tile.tile<64x128, #tiny_tile.layout<thread = [16, 4], vector_size = 8>>
```

```tablegen
def TinyTile_TileType : TypeDef<TinyTile_Dialect, "Tile"> {
  let mnemonic = "tile";

  let parameters = (ins
    "int64_t":$height,
    "int64_t":$width,
    "LayoutAttr":$layout
  );

  // Use custom parser/printer because the lexer treats "64x128" as a dimension
  // list token. We want `<64x128, layout>` syntax.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the total number of elements in the tile.
    int64_t getNumElements() const { return getHeight() * getWidth(); }

    /// Returns the per-thread vector type.
    VectorType getPerThreadVectorType() const;
  }];
}
```

Two things deserve attention.

**Why `hasCustomAssemblyFormat`?** We want the familiar `64x128` shape syntax,
but MLIR's lexer tokenizes `64x128` as a single *dimension-list* token, not as
"integer, `x`, integer" — so a declarative
`assemblyFormat = "`<` $height `x` $width ... "` cannot parse it. Instead we
write `parse`/`print` by hand and use the parser's dedicated hook
(from `TinyTileDialect.cpp`):

```cpp
Type TileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse "HxW" as a dimension list. MLIR's lexer treats "64x128" as a single
  // dimension list token, so we must use parseDimensionList.
  SmallVector<int64_t, 2> dims;
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false,
                                /*withTrailingX=*/false))
    return Type();
  ...
  LayoutAttr layout;
  if (parser.parseAttribute(layout))
    return Type();

  if (parser.parseGreater())
    return Type();

  return TileType::get(parser.getContext(), dims[0], dims[1], layout);
}
```

This is the same escape hatch Chapter 20 used for a region op's syntax, now
applied to a type. `parser.parseAttribute(layout)` recursively invokes the
layout attribute's *generated* parser — parameterized attributes and types
compose.

**The verifier ties tile to layout.** `TileType::verify` requires power-of-two
dimensions and, crucially, that the tile's element count equals the layout's
coverage:

```cpp
  // Verify tile elements match layout elements.
  int64_t tileElements = height * width;
  int64_t layoutElements = layout.getNumElements();
  if (tileElements != layoutElements) {
    return emitError() << "tile has " << tileElements << " elements but layout covers "
                       << layoutElements << " elements";
  }
```

This one invariant is what makes the whole lowering story trivial later: a
well-formed tile *always* divides evenly into `numThreads` vectors of
`vectorSize` elements. The bridge between the two worlds is a one-liner:

```cpp
VectorType TileType::getPerThreadVectorType() const {
  auto layout = getLayout();
  auto elementType = Float16Type::get(getContext());
  return VectorType::get({layout.getVectorSize()}, elementType);
}
```

A `tile<32x16, layout<thread=[16,4], vector_size=8>>` is, from any single
thread's point of view, just a `vector<8xf16>`. Keep that in mind — it is the
entire type-conversion story of this chapter.

## The ops, briefly

The five ops are plain ODS at this point in the tutorial series; see
[`TinyTileDialect.td`](../../lib/Dialect/TinyTile/TinyTileDialect.td) for all
of them. Two representative excerpts. `elementwise` packs the four arithmetic
kinds into one op with an `EnumAttr` (the same `I32EnumAttr` machinery as
upstream `arith.cmpi` predicates):

```tablegen
def TinyTile_ElementwiseOp : TinyTile_Op<"elementwise",
    [Pure, DeclareOpInterfaceMethods<TinyTileLoweringOpInterface>]> {
  let summary = "Element-wise binary operation on tiles.";

  let arguments = (ins
    TinyTile_ElementwiseKindAttr:$kind,
    TinyTile_Tile:$lhs,
    TinyTile_Tile:$rhs
  );
  let results = (outs TinyTile_Tile:$result);

  let assemblyFormat = "$kind $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
}
```

`load` and `store` mix our new tile type with Part 4's `!tiny.ptr`. Since
`PtrType` lives in another dialect's C++, the `.td` file wraps it in a local
`Type<CPred<...>>` constraint before using it as an operand type:

```tablegen
def Tiny_PtrType : Type<
  CPred<"::llvm::isa<::mlir::tiny::PtrType>($_self)">,
  "tiny.ptr type",
  "::mlir::tiny::PtrType"
>;
```

Parse/print round-trips for every op are checked by
[`test/part5/basic-ops.mlir`](../../test/part5/basic-ops.mlir).

The interesting part of both definitions is the trait list:
`DeclareOpInterfaceMethods<TinyTileLoweringOpInterface>`. Every one of the
five ops carries it. Time to define that interface.

## Defining an op interface in ODS

Think ahead to the lowering pass. The classic approach (Chapter 18) is one
`OpConversionPattern` subclass per op: `SplatOpLowering`,
`ElementwiseOpLowering`, `LoadOpLowering`, ... five patterns whose
`matchAndRewrite` bodies share nothing but boilerplate. The alternative is to
flip the dependency around: let *each op* describe its own lowering, and have
the pass ask for it through a common API. "A common API implemented
per-op, dispatched at runtime on an `Operation*`" is precisely what an
**op interface** is.

The whole definition, from
[`TinyTileLoweringInterface.td`](../../lib/Dialect/TinyTile/TinyTileLoweringInterface.td):

```tablegen
def TinyTileLoweringOpInterface : OpInterface<"TinyTileLoweringOpInterface"> {
  let description = [{
    Interface for TinyTile operations that can be converted to SIMT operations.

    Operations implementing this interface provide their own conversion logic,
    which is invoked by a generic conversion pattern. The SIMT operands
    (with TileType already converted to per-thread VectorType) are passed
    to the method.
  }];

  let cppNamespace = "::mlir::tiny_tile";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Convert this operation to SIMT (per-thread) operations.
        ...
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"convertToSIMT",
      /*args=*/(ins
        "::mlir::RewriterBase &":$rewriter,
        "::mlir::ValueRange":$simtOperands
      )
    >
  ];
}
```

An `OpInterface` is a list of `InterfaceMethod`s, each giving a return type,
name, and argument list as C++ strings. Ours has a single method: "rewrite
yourself into per-thread form, given your already-type-converted operands."

The build hooks it up with two extra tablegen invocations
([`CMakeLists.txt`](../../lib/Dialect/TinyTile/CMakeLists.txt)):

```cmake
set(LLVM_TARGET_DEFINITIONS TinyTileLoweringInterface.td)
mlir_tablegen(TinyTileLoweringInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TinyTileLoweringInterface.cpp.inc -gen-op-interface-defs)
```

### What tablegen generates

It pays to read the generated code once in your life; it is short and it
demystifies every interface you will ever use. Look at
`build/lib/Dialect/TinyTile/TinyTileLoweringInterface.h.inc`. The core is a
hand-rolled vtable, the classic **Concept/Model** idiom:

```cpp
struct TinyTileLoweringOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::LogicalResult (*convertToSIMT)(const Concept *impl,
        ::mlir::Operation *, ::mlir::RewriterBase &, ::mlir::ValueRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept { ... };
  ...
};
```

- `Concept` is a struct of function pointers — one per interface method, all
  taking an opaque `Operation*`.
- `Model<ConcreteOp>` fills those pointers for one specific op class by
  casting the opaque pointer down and forwarding:

```cpp
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val))
      .convertToSIMT(rewriter, simtOperands);
```

- The public class `TinyTileLoweringOpInterface` wraps an `Operation*` plus a
  pointer to the right `Concept`, and its `convertToSIMT` (in the `.cpp.inc`)
  is one line: `return getImpl()->convertToSIMT(getImpl(), getOperation(),
  rewriter, simtOperands);`.

When an op lists `DeclareOpInterfaceMethods<TinyTileLoweringOpInterface>` in
its traits, two things happen in the generated op code
(`build/lib/Dialect/TinyTile/TinyTileOps.h.inc`): the op class inherits
`TinyTileLoweringOpInterface::Trait`, which registers a `Model<ElementwiseOp>`
so `isa`/`cast` to the interface succeed — and, because of the *Declare*
prefix, the method is declared on the op class for us to implement:

```cpp
class ElementwiseOp : public ::mlir::Op<ElementwiseOp, ...,
    ::mlir::tiny_tile::TinyTileLoweringOpInterface::Trait> {
  ...
  ::mlir::LogicalResult convertToSIMT(::mlir::RewriterBase &rewriter,
                                      ::mlir::ValueRange simtOperands);
```

That declaration is a link-time promise: forget to implement `convertToSIMT`
for an op that declares the interface and you get an undefined-symbol error,
not a runtime surprise. (The alternatives, for reference: an
`InterfaceMethod` can carry a default implementation, and an
`ExternalModel` can attach an interface to an op you don't own — that is how
upstream retrofits interfaces onto builtin ops.)

Now revisit Chapter 13's trait/interface distinction with implementer's eyes:
a *trait* is static — the op class mixes in behavior, but you must know the
concrete op class to use it. An *interface* additionally registers that
function-pointer table so code holding a bare `Operation*` can query and call
it at runtime:

```cpp
if (auto lowerable = dyn_cast<TinyTileLoweringOpInterface>(op))
  (void)lowerable.convertToSIMT(rewriter, operands);
```

That is exactly the shape a generic conversion pattern needs.

## The conversion pass

The pass is declared in
[`TinyTilePasses.td`](../../lib/Dialect/TinyTile/TinyTilePasses.td) as
`--tiny-tile-to-tiny` and implemented in
[`TinyTileToTiny.cpp`](../../lib/Dialect/TinyTile/TinyTileToTiny.cpp). It is a
Chapter 18-style dialect conversion with three components, and thanks to the
interface, all three are tiny.

**1. The type converter.** The observation from earlier — "a tile, seen from
one thread, is a `vector<vectorSize x f16>`" — becomes code:

```cpp
class TinyTileToTinyTypeConverter : public TypeConverter {
public:
  TinyTileToTinyTypeConverter() {
    // Identity conversion for all types (fallback).
    addConversion([](Type type) { return type; });

    // Convert TileType to VectorType (layout is always present).
    addConversion([](TileType tileType) -> Type {
      return tileType.getPerThreadVectorType();
    });
  }
};
```

**2. One pattern for five ops.** Instead of `OpConversionPattern<SplatOp>`
etc., we use `OpInterfaceConversionPattern`, which matches any op
implementing the given interface and hands the type-converted operands
straight through:

```cpp
struct TinyTileLoweringPattern
    : public OpInterfaceConversionPattern<TinyTileLoweringOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(TinyTileLoweringOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return op.convertToSIMT(rewriter, operands);
  }
};
```

This is the payoff of the interface design. The pass knows nothing about
splat vs. load; adding a sixth op to the dialect requires *zero* changes
here — the op brings its own lowering. Upstream MLIR plays the same trick at
scale: `LoopLikeOpInterface` lets `--loop-invariant-code-motion` hoist out of
any loop op in any dialect, and bufferization, inlining, and tiling are all
interface-driven passes.

**3. Target and driver.** `tiny_tile` becomes illegal; the SIMT-level
dialects are legal; SCF ops are legal only once their types are converted
(the demo's accumulator loop carries a tile as an `iter_arg`, and
`populateSCFStructuralTypeConversions` rewrites `scf.for`/`scf.yield`
signatures through the type converter — compare Chapter 18, where we did the
same for `func.func`):

```cpp
    ConversionTarget target(getContext());
    target.addIllegalDialect<TinyTileDialect>();
    target.addLegalDialect<tiny::TinyDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addDynamicallyLegalDialect<scf::SCFDialect>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());
    patterns.add<TinyTileLoweringPattern>(typeConverter, &getContext());
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
```

One deliberate gap: the pass converts ops and SCF structure but *not*
function signatures, so a tile cannot cross a `func.func` boundary. Feed it a
function with a tile argument and the materialization machinery from
Chapter 18 reports exactly what is missing:

```
error: failed to legalize unresolved materialization from
('!tiny_tile.tile<32x16, ...>') to ('vector<8xf16>') that remained live after
conversion
note: see current operation: %0 = "builtin.unrealized_conversion_cast"(%arg0) ...
```

In this language that is fine — a kernel produces tiles with `load`/`splat`
and consumes them with `store`/`sum` — but [Exercise 2](#exercise-2-let-tiles-cross-function-boundaries)
closes the gap.

## The `convertToSIMT` implementations

All five implementations live at the bottom of
[`TinyTileDialect.cpp`](../../lib/Dialect/TinyTile/TinyTileDialect.cpp).
We walk through them from trivial to subtle, each with real IR produced by
`build/tools/tutorial-opt --tiny-tile-to-tiny`. Throughout, the running
layout is

```mlir
#L = #tiny_tile.layout<thread = [16, 4], vector_size = 8>   // 64 threads x 8 elems
```

on a `32x16` tile (512 elements), so every tile becomes a `vector<8xf16>`.

### splat: constants are free

Per thread, "a tile full of 1.0" is just "a vector<8xf16> full of 1.0". The
implementation builds a splat `DenseElementsAttr` of the per-thread vector
type (converting the stored f64 down to f16) and replaces the op with a
`tiny.constant`:

```cpp
LogicalResult SplatOp::convertToSIMT(RewriterBase &rewriter,
                                     ValueRange simtOperands) {
  auto tileType = cast<TileType>(getResult().getType());
  VectorType vectorType = tileType.getPerThreadVectorType();

  APFloat f64Value = getSplatValue();
  ... // convert to IEEEhalf
  auto splatAttr = DenseElementsAttr::get(vectorType, f64Value);
  rewriter.replaceOpWithNewOp<tiny::ConstantOp>(*this, splatAttr);
  return success();
}
```

```mlir
// before
%tile = tiny_tile.splat 1.0 : !tiny_tile.tile<32x16, #L>
// after
%0 = tiny.constant dense<1.000000e+00> : vector<8xf16>
```

Note what did *not* appear: any mention of thread IDs. Every thread runs the
identical instruction — SIMT code is still one program.

### elementwise: the layout guarantees alignment

Because the verifier forced both operands onto the *same* layout, thread `t`
of the lhs holds exactly the elements that thread `t` of the rhs holds. An
element-wise op therefore never needs communication — it is a per-thread
vector op, and `convertToSIMT` is a switch over the kind:

```cpp
  switch (getKind()) {
  case ElementwiseKind::add:
    rewriter.replaceOpWithNewOp<tiny::AddFOp>(*this, lhs, rhs);
    break;
  case ElementwiseKind::sub:
    rewriter.replaceOpWithNewOp<tiny::SubFOp>(*this, lhs, rhs);
    break;
  ...
```

`lhs` and `rhs` come from `simtOperands` — the pattern already remapped them
to `vector<8xf16>` values, which is why the method wants the converted
operands rather than calling `getLhs()` (whose type is still a tile).

```mlir
// before
%c = tiny_tile.elementwise mul %a, %b : ..., ... -> !tiny_tile.tile<32x16, #L>
// after
%14 = tiny.mulf %6, %13 : vector<8xf16>
```

### load and store: where thread IDs enter

A tile load is the first op whose SIMT form differs per thread: each thread
must load *its* 8 elements. Two shared helpers compute the offsets. The
memory-side offset is plain strided addressing, `row * stride + col`. The
thread-side offset implements the layout contract from earlier:

```cpp
static Value computeThreadOffset(OpBuilder &builder, Location loc,
                                 LayoutAttr layout) {
  // Get thread IDs.
  Value tidX = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value tidY = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::y);
  ...
  // Compute linear thread ID: tid_y * thread[0] + tid_x
  Value threadRowsConst =
      arith::ConstantOp::create(builder, loc, builder.getIndexAttr(threadRows));
  Value linearTid = arith::AddIOp::create(
      builder, loc, tidX,
      arith::MulIOp::create(builder, loc, tidY, threadRowsConst));

  // Compute thread offset: linear_tid * vector_size
  Value vectorSizeConst =
      arith::ConstantOp::create(builder, loc, builder.getIndexAttr(vectorSize));
  return arith::MulIOp::create(builder, loc, linearTid, vectorSizeConst);
}
```

`gpu.thread_id` is an upstream op (dialect `gpu`) that reads the invocation's
position in the workgroup — the moment it appears, the code has committed to
the SIMT model. `LoadOp::convertToSIMT` glues the pieces together and emits a
per-thread `tiny.load` of the per-thread vector type:

```cpp
  Value baseOffset = computeBaseOffset(rewriter, loc, row, col, stride);
  Value threadOffset = computeThreadOffset(rewriter, loc, layout);
  Value finalOffset =
      arith::AddIOp::create(rewriter, loc, baseOffset, threadOffset);

  VectorType vectorType = tileType.getPerThreadVectorType();
  rewriter.replaceOpWithNewOp<tiny::LoadOp>(*this, vectorType, ptr,
                                            finalOffset);
```

`StoreOp` is the mirror image with one extra `simtOperand` (the value).
Before/after for a load+store pair, after `--canonicalize --cse` folds the
duplicated offset math:

```mlir
// before
%a = tiny_tile.load %ptr, %row, %col stride %stride
     : !tiny.ptr -> !tiny_tile.tile<32x16, #L>
tiny_tile.store %a, %ptr, %row, %col stride %stride
     : !tiny_tile.tile<32x16, #L>, !tiny.ptr

// after: --tiny-tile-to-tiny --canonicalize --cse
%0 = arith.muli %arg1, %arg3 : index          // row * stride
%1 = arith.addi %0, %arg2 : index             //   + col
%thread_id_x = gpu.thread_id x
%thread_id_y = gpu.thread_id y
%2 = arith.muli %thread_id_y, %c16 : index    // tid_y * thread[0]
%3 = arith.addi %thread_id_x, %2 : index      //   + tid_x
%4 = arith.muli %3, %c8 : index               //   * vector_size
%5 = arith.addi %1, %4 : index                // final per-thread offset
%6 = tiny.load %arg0, %5 : vector<8xf16>
tiny.store %6, %arg0, %5 : vector<8xf16>
```

One tile op became one memory op *per thread*, 64 threads covering 512
elements — the compiler wrote the addressing code the CUDA programmer would
have written by hand. (Without `--cse` you would see the offset chain twice:
each `convertToSIMT` emits its own `gpu.thread_id`s and arithmetic, and
cleanup is deliberately left to the canonicalizer, as usual in MLIR.)

### sum: a two-level reduction

`tiny_tile.sum` is the only op that needs *cross-thread* communication: the
512 elements to be summed live in 64 different threads' registers. The SIMT
recipe is the standard two-level reduction:

1. each thread reduces its own `vector<8xf16>` to a scalar (`tiny.sum` +
   `vector.extract`), then
2. the threads combine their 64 partial sums with `gpu.subgroup_reduce add`,
   a hardware-backed primitive (warp shuffles on NVIDIA, DPP on AMD), and
3. `vector.broadcast` wraps the scalar back into the `vector<1xf16>` the
   result type promises.

```cpp
LogicalResult SumOp::convertToSIMT(RewriterBase &rewriter,
                                   ValueRange simtOperands) {
  Location loc = getLoc();
  Value input = simtOperands[0];

  // First, reduce the per-thread vector to a scalar using tiny.sum.
  VectorType vec1Type = VectorType::get({1}, rewriter.getF16Type());
  Value localSum = tiny::SumOp::create(rewriter, loc, vec1Type, input);

  // Extract the scalar from vector<1xf16>.
  Value scalar =
      vector::ExtractOp::create(rewriter, loc, localSum, ArrayRef<int64_t>{0});

  // Perform cross-thread reduction using gpu.subgroup_reduce.
  Value subgroupSum = gpu::SubgroupReduceOp::create(
      rewriter, loc, scalar, gpu::AllReduceOperation::ADD, /*uniform=*/true);

  // Wrap result back to vector<1xf16> for compatibility with tiny.store.
  Value result =
      vector::BroadcastOp::create(rewriter, loc, vec1Type, subgroupSum);

  rewriter.replaceOp(*this, result);
  return success();
}
```

```mlir
// before
%s = tiny_tile.sum %a : !tiny_tile.tile<32x16, #L> -> vector<1xf16>
// after
%7 = tiny.sum %6 : vector<8xf16> -> vector<1xf16>
%8 = vector.extract %7[0] : f16 from vector<1xf16>
%9 = gpu.subgroup_reduce add %8 uniform : (f16) -> f16
%10 = vector.broadcast %9 : f16 to vector<1xf16>
```

A scoping caveat: `gpu.subgroup_reduce` reduces across a *subgroup* (warp),
typically 32 threads. For our demo layout (32 threads total) that is the whole
workgroup, so the result is the full tile sum; for the 64-thread layout above,
a production compiler would insert a second stage through workgroup shared
memory. Layout-aware choices like this are exactly what Triton's reduction
codegen spends its complexity budget on.

## Playing with the DSL

As in Chapters 19 and 20, a Python DSL drives the whole descent.
[`examples/gpu_dot_product.py`](examples/gpu_dot_product.py) writes an SPMD
dot product: each block loads `1x256` tiles of `a` and `b` (layout
`thread=(1, 32), vector_size=8`, so 32 threads × 8 elements), accumulates
`a * b` over K, sums the tile and stores one value per block. Run it from
the repo root (after `uv pip install -e python/`):

```sh
TUTORIAL_OPT=$PWD/build/tools/tutorial-opt \
  python part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/examples/gpu_dot_product.py
```

It prints four stages:

1. **`TinyTile Dialect`**: the tile-level program, one op per line of
   Python (`tiny_tile.splat`, `tiny_tile.load`, `tiny_tile.elementwise
   mul/add`, `tiny_tile.sum`) inside a `tiny_loop.accumulate`.
2. **`After tiny-loop-to-scf`**: the accumulate loop is now an `scf.for`
   carrying the tile as `iter_args` (Chapter 20's lowering).
3. **`After tiny-tile-to-tiny`**: this chapter's pass. The tile type is
   gone; each thread works on a `vector<8xf16>`, the load address includes
   `gpu.thread_id` times the vector size, and the sum became a per-thread
   `tiny.sum` plus a `gpu.subgroup_reduce`:

   ```mlir
   %2 = scf.for %arg5 = %c0 to %arg4 step %0 iter_args(%arg6 = %1) -> (vector<8xf16>) {
     ...
     %thread_id_x = gpu.thread_id x
     %thread_id_y = gpu.thread_id y
     %9 = arith.addi %thread_id_x, %thread_id_y : index
     %10 = arith.muli %9, %c8 : index
     %11 = arith.addi %8, %10 : index
     %12 = tiny.load %arg0, %11 : vector<8xf16>
     %13 = tiny.load %arg1, %11 : vector<8xf16>
     %14 = tiny.mulf %12, %13 : vector<8xf16>
     %15 = tiny.addf %arg6, %14 : vector<8xf16>
     scf.yield %15 : vector<8xf16>
   }
   %3 = tiny.sum %2 : vector<8xf16> -> vector<1xf16>
   %4 = vector.extract %3[0] : f16 from vector<1xf16>
   %5 = gpu.subgroup_reduce add %4 uniform : (f16) -> f16
   ```
4. **`After tiny-to-arith`**: plain `arith`/`vector`/`scf`/`gpu` ops. From
   here the usual GPU path applies: kernel outlining and NVVM to PTX
   ([Chapter 31](../../part6-end-to-end-ml/ch31-gpu-nvvm/README.md)), or the
   ROCDL dialect for AMDGPU.

Try modifying the tile layout or dimensions and watch the SIMT code change.
With `Layout(thread=(1, 64), vector_size=4)` (still 256 elements), every
`vector<8xf16>` becomes `vector<4xf16>` and the thread offset is scaled by 4
instead of 8. With `Layout(thread=(1, 32), vector_size=4)` the layout covers
only 128 of the tile's 256 elements, and the type's verifier rejects it:

```
error: tile has 256 elements but layout covers 128 elements
Assertion failed: (succeeded( ConcreteT::verifyInvariants(...))), function get, ...
```

The error is right, but it is followed by a crash: `TileType::parse` in
[`TinyTileDialect.cpp`](../../lib/Dialect/TinyTile/TinyTileDialect.cpp)
builds the type with `TileType::get`, which asserts that verification
succeeds. Exercise 3 below fixes that.

## Exercises

> Ported from mlir-tutor's chapter 3 exercises (its `ch3-exercise` branch).

The solutions are in-tree: `lib/Dialect/TinyTile/` is what `tutorial-opt`
is built from. The original TODO versions are in
[`exercises/skeletons/`](exercises/skeletons/). They are not compiled. To
work an exercise, copy them over the solution, rebuild, run the test, and
restore the solution with `git` afterwards:

| Skeleton | Copy it over | Checked by |
| --- | --- | --- |
| `skeletons/TinyTileDialect.td` | `lib/Dialect/TinyTile/TinyTileDialect.td` | `test/part5/lower-to-tiny.mlir` |
| `skeletons/TinyTileDialect.cpp` | `lib/Dialect/TinyTile/TinyTileDialect.cpp` | `test/part5/lower-to-tiny.mlir` |

```bash
cp part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/exercises/skeletons/TinyTileDialect.{td,cpp} \
   lib/Dialect/TinyTile/
cmake --build build -j
.venv/bin/lit build/test -v --filter 'part5/(basic-ops|lower-to-tiny)'
git checkout -- lib/Dialect/TinyTile/     # back to the solution
```

The two skeleton files build as shipped. `basic-ops.mlir` still passes,
because parsing and printing never touch the interface. `lower-to-tiny.mlir`
fails until you finish Exercise 1.

### Exercise 1: implement `convertToSIMT` for load, store and sum

In the skeleton, only `elementwise` and `splat` carry
`TinyTileLoweringOpInterface`, and only those two have `convertToSIMT`
bodies. Finish the other three ops. If you can, do it before reading
[The `convertToSIMT` implementations](#the-converttosimt-implementations)
above, which walks through the answers.

1. In `TinyTileDialect.td`, add
   `DeclareOpInterfaceMethods<TinyTileLoweringOpInterface>` to the trait
   lists of `TinyTile_LoadOp`, `TinyTile_StoreOp` and `TinyTile_SumOp`
   (keep `Pure` on `sum`).
2. In `TinyTileDialect.cpp`, implement `convertToSIMT` for each, using the
   existing `ElementwiseOp`/`SplatOp` versions as a model and the
   `computeBaseOffset`/`computeThreadOffset` helpers already in the file:
   - `LoadOp` → `gpu.thread_id` + `tiny.load`. The operands are
     `[ptr, row, col, stride]`. The offset is `row * stride + col` plus the
     per-thread offset from the layout, and the result type is
     `tileType.getPerThreadVectorType()`.
   - `StoreOp` → `gpu.thread_id` + `tiny.store`, with operands
     `[value, ptr, row, col, stride]` and the same offset arithmetic.
   - `SumOp` → `tiny.sum` + `vector.extract` + `gpu.subgroup_reduce add` +
     `vector.broadcast`. This is the two-level reduction: each thread reduces
     its own vector, then the threads combine their partial sums.

Two intermediate states are worth seeing on purpose, because each one fails in
a way you will meet again in real projects:

- **Neither half done (the skeleton as shipped).** The build succeeds, but
  `--tiny-tile-to-tiny` reports `failed to legalize operation
  'tiny_tile.load' that was explicitly marked illegal`. The op does not
  implement the interface, so the single `OpInterfaceConversionPattern`
  never matches it, and the conversion target still forbids it.
- **Trait added, method missing.** The build fails at *link* time with
  an undefined `LoadOp::convertToSIMT`. `DeclareOpInterfaceMethods` declared
  the method on the op class, and the interface's Model calls it, but
  nothing defines it. (The reverse, a method body without the trait, does
  not even compile, because the op class has no such member to define.)

### Exercise 2: let tiles cross function boundaries

[The conversion pass](#the-conversion-pass) section showed that a
`func.func` taking a `!tiny_tile.tile<...>` argument fails with an
unresolved materialization. Fix it in
[`TinyTileToTiny.cpp`](../../lib/Dialect/TinyTile/TinyTileToTiny.cpp) the way
Chapter 19's `TinyToLLVM.cpp` handles `!tiny.ptr`:

- add `populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter)`
  and `populateReturnOpTypeConversionPattern(patterns, typeConverter)`
  (add `populateCallOpTypeConversionPattern` too if you want calls);
- mark `func.func` dynamically legal when
  `typeConverter.isSignatureLegal(op.getFunctionType()) && typeConverter.isLegal(&op.getBody())`,
  and `func.return` when its operand types are legal.

Then add a test case to `test/part5/lower-to-tiny.mlir` (it uses
`--split-input-file`, so append a `// -----` section). Write a function that
takes a tile and returns `tiny_tile.sum` of it, and check that its signature
becomes `vector<Nxf16>`. Think about what this means for a real GPU kernel
ABI: a per-thread vector is now an *argument*, so every thread must already
hold its slice when the function is called.

### Exercise 3: make a bad layout an error, not a crash

[Playing with the DSL](#playing-with-the-dsl) showed that a tile whose
layout doesn't cover it trips an assertion. `TileType` already has a
verifier (the `error: tile has 256 elements but layout covers 128 elements`
line comes from it); the problem is that `TileType::parse` calls
`TileType::get`, which asserts that verification passes. Change the
parser's last line to build the type with `parser.getChecked<TileType>(...)`
instead, which reports the verifier's diagnostic at the parser's location
and returns a null type. Rebuild and rerun the bad-layout DSL program: you
should get the error pointing at the offending `!tiny_tile.tile<...>` and
no stack dump. Then add a `--verify-diagnostics` test to
`test/part5/basic-ops.mlir` (a function using
`!tiny_tile.tile<1x256, #tiny_tile.layout<thread = [1, 32], vector_size = 4>>`
with an `expected-error` line) so the fix stays fixed.

### Challenge: `tiny_tile.matmul`

Add a `tiny_tile.matmul` op that multiplies two tiles and produces a tile,
and lower it through the same interface. Decide first what the op means at
the workgroup level: which shapes must agree, and what the result layout is.
Then decide what each thread has to compute. With the layouts in this
chapter, a thread holds a contiguous `vector_size` run of one row, so for
`C = A × B` it owns part of a row of C. It needs the matching row of A,
which other threads hold, and a column of B, which is spread across all the
threads. That forces you to decide how data moves between threads (repeated
`gpu.subgroup_reduce`/shuffles, or a trip through memory with `tiny.load`),
or to choose a different layout for matmul operands. This is the problem
that pushes Triton-style compilers to give dot operands and accumulators
their own layout kinds. Add the op to `basic-ops.mlir` and a lowering test
to `lower-to-tiny.mlir`.
