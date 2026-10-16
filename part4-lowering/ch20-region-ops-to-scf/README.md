# Chapter 20: Region Ops and Lowering to SCF

> Reworked from mlir-tutor's chapter 2 for the unified repo.

Every op we have defined so far — `poly` in Part 3, `tiny` in Chapters 18-19 —
was a "flat" op: operands in, results out. Real compilers need more than that:
loops, conditionals, and function bodies all contain *nested code*, and MLIR's
answer to nested code is the **region** — an op can carry one or more regions,
each holding blocks of further ops.

This chapter builds a region-carrying op from scratch: `tiny_loop.accumulate`,
a counted for-loop with **loop-carried SSA values** — then lowers it onto
upstream `scf.for` by cloning the region body into the new op.

## What you will learn

- Regions, block arguments, and how loop-carried values
  (`init` args / `iter_args`) work in an SSA IR
- Declaring regions in ODS (`SizedRegion<1>` vs `AnyRegion`), the
  `SingleBlock`, `SingleBlockImplicitTerminator`, and
  `RecursiveMemoryEffects` traits, and why a companion `yield` op is needed
- Building, parsing (`parseRegion`, `ensureTerminator`), printing, and
  verifying an op whose region takes block arguments
- Lowering a region op to `scf.for` with a rewrite pattern that clones the
  body block through an `IRMapping`
- Running the full pipeline `tiny_loop` + `tiny` → `scf` → `arith`/LLVM,
  and driving it all from the Python DSL

## Prerequisites

- **Chapter 18** (dialect conversion) and **Chapter 19** (lowering to LLVM):
  this chapter reuses the `tiny` dialect and its `--tiny-to-arith` /
  `--tiny-to-llvm` passes for the loop *bodies*, and adds loops around them.
- **Part 3** for ODS basics. Here we go past the declarative assembly format
  and write the parser and printer by hand.
- `scf.for` and its `iter_args` were previewed back in Part 1,
  [Chapter 2](../../part1-using-mlir/ch02-lowering-pipelines/README.md), when
  we lowered *through* SCF. Now we lower *onto* it.
- A built `build/tools/tutorial-opt` and the Python venv from the repo README.

All code for this chapter lives in
[`lib/Dialect/TinyLoop/`](../../lib/Dialect/TinyLoop/):

| File | Contents |
| --- | --- |
| `TinyLoopDialect.td` | ODS for the dialect, `accumulate`, and `yield` |
| `TinyLoopDialect.cpp` | Custom builder, parser, printer, verifier |
| `TinyLoopPasses.td` | The `--tiny-loop-to-scf` pass declaration |
| `TinyLoopToSCF.cpp` | The lowering pattern and pass implementation |

Tests are in [`test/part4/`](../../test/part4/): `basic-loop.mlir`,
`accumulate.mlir`, `lower-to-scf.mlir`, `full-pipeline.mlir`.

## The language feature: `accumulate`

In the Python DSL from Chapters 18-19 we could express straight-line vector
code. This chapter extends the language with a loop construct. Here is the
loop nest from the motivating program — a tiled, vectorized matrix multiply
`C[M,N] = A[M,K] * B[K,N]^T`, full source in
[`examples/matmul.py`](examples/matmul.py) (`one` and `vstep` are `Index`
constants 1 and 16):

```python
@accumulate(M, one)
def _(i: Index):
    @accumulate(N, one)
    def _(j: Index):
        # Initialize accumulator vector
        acc_init = F16Vector.constant([0.0])

        @accumulate(K, vstep, inits=[acc_init])
        def _(k: Index, acc: F16Vector):
            # Load a[i, k:k+16] and b[j, k:k+16]
            a_idx = i * K + k
            b_idx = j * K + k
            a_vec = a.load(a_idx, num_elements=vec_size)
            b_vec = b.load(b_idx, num_elements=vec_size)
            # Accumulate element-wise product
            return acc + (a_vec * b_vec).sum()

        # Sum the accumulator and store result
        result = _[0]
        c_idx = i * N + j
        c.store(c_idx, result)
```

`accumulate(size, step)` is a counted loop equivalent to Python's
`for i in range(0, size, step)`; the decorated function is the loop *body*,
whose first argument is the induction variable.

The interesting part is the innermost loop. Our IR is SSA — you cannot
"update a variable" across iterations by assignment. Instead the loop supports
**loop-carried values**: initial values are passed as `inits=[...]`, the body
receives each carried variable's current value as an extra argument (`acc`
above) and *returns* its value for the next iteration, and after the loop the
final values are available as the loop's results.

## Designing a region op

How do we model this loop as an MLIR op? A flat op with operands and results
cannot hold the body, so we give the op a **region**:

```
tiny_loop.accumulate %n step(%step) init(%init) -> index {   <- the op
^bb0(%i: index, %acc: index):                                <- entry block args
  %next = arith.addi %acc, %i : index                        <- body
  tiny_loop.yield %next : index                              <- terminator
}
```

The design decisions, one by one:

- **Operands.** `%bound` and `%step` (both `index`), plus a variadic list of
  `init` values of any type. The lower bound is fixed at 0, matching the DSL's
  `range(0, size, step)` — one less operand to carry around.
- **Results.** One result per `init` value, with matching types: the final
  values of the carried variables.
- **Region.** Exactly one region with exactly one block, whose arguments are
  `[%i: index, %acc0, %acc1, ...]` — induction variable first, then one per
  carried value. Block arguments are how a region receives values that change
  on every entry; they are the region-op analogue of function parameters.
- **Terminator.** Every block must end in a terminator op, so we define a
  companion `tiny_loop.yield` whose operands are the next-iteration values of
  the carried variables — the same pattern as upstream's `scf.for`/`scf.yield`
  and `linalg.generic`/`linalg.yield`. A dedicated yield op per dialect keeps
  the pairing verifiable (`HasParent`) and the semantics unambiguous.

The chain of custody for a carried value is worth spelling out, because the
verifier and the lowering both hinge on it: the `init` operand enters the
block argument on the first iteration, the `yield` operand re-enters it on
every following iteration, and the last iteration's `yield` operand becomes
the op result. Four value lists — `init` operands, trailing block arguments,
`yield` operands, op results — must all agree in count and type.

## ODS: declaring an op with a region

The full dialect definition is in
[`lib/Dialect/TinyLoop/TinyLoopDialect.td`](../../lib/Dialect/TinyLoop/TinyLoopDialect.td).
The `accumulate` op:

```tablegen
def TinyLoop_AccumulateOp : TinyLoop_Op<"accumulate", [
    // The region has exactly one block.
    SingleBlock,
    // The block is implicitly terminated by tiny_loop.yield.
    SingleBlockImplicitTerminator<"YieldOp">,
    // The loop may have memory effects through its body.
    RecursiveMemoryEffects
  ]> {
  let summary = "Accumulate loop from 0 to N.";

  let arguments = (ins
    Index:$bound,
    Index:$step,
    Variadic<AnyType>:$initArgs
  );

  let results = (outs Variadic<AnyType>:$results);

  // SizedRegion<1> means the region has exactly one block.
  let regions = (region SizedRegion<1>:$region);
```

The new pieces relative to Part 3:

- **`let regions = (region SizedRegion<1>:$region);`** declares that the op
  carries one region, constrained to exactly one block. The looser constraint
  `AnyRegion` accepts any number of blocks — what you would use for an op
  whose body may branch internally, like `func.func`. For a structured loop
  one block is what we want, and `SizedRegion<1>` plus the `SingleBlock`
  trait let the rest of the code call `getRegion().front()` without checking.
- **`SingleBlockImplicitTerminator<"YieldOp">`** says the block is terminated
  by `tiny_loop.yield`, and that the terminator may be *implicit*: when a
  loop carries no values, the textual IR may omit the empty `tiny_loop.yield`
  and the parser inserts it (via `ensureTerminator`, below) — which is why
  `test/part4/basic-loop.mlir` can write a side-effect-only loop with no
  visible terminator.
- **`RecursiveMemoryEffects`** tells MLIR the op's memory behavior is the
  union of the memory behavior of everything *inside* its region. Without it,
  an op with no declared effects and no results (our side-effect-only loop)
  looks trivially dead, and canonicalization will happily delete it — stores
  and all.

Four more declarations complete the op:

```tablegen
  let hasCustomAssemblyFormat = 1;

  // Custom builder to set up the block arguments correctly.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$bound, "Value":$step,
      CArg<"ValueRange", "{}">:$initArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];
```

The generated default builders know nothing about entry-block arguments, so
`skipDefaultBuilders` suppresses them and we declare our own. The
`bodyBuilder` callback is an upstream idiom (compare `scf::ForOp::build`):
callers pass a lambda that receives the induction variable and the carried
values and emits the body — exactly the hook the Python DSL needs. An
`extraClassDeclaration` adds accessors (`getInductionVar()`,
`getRegionIterArgs()`, `getBody()`) so C++ code reads like the op's semantics
instead of block-argument index arithmetic, and `let hasVerifier = 1;`
declares the verifier we implement below.

The companion terminator `TinyLoop_YieldOp` takes
`Variadic<AnyType>:$results` and three traits: `Terminator` marks it as a
block terminator, `HasParent<"AccumulateOp">` rejects a stray
`tiny_loop.yield` anywhere else, and `Pure` says the yield itself touches no
memory. Having no region of its own, `yield` gets away with a one-line
declarative format:

```tablegen
let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
```

## The custom builder

The builder implementation in
[`lib/Dialect/TinyLoop/TinyLoopDialect.cpp`](../../lib/Dialect/TinyLoop/TinyLoopDialect.cpp)
shows what "setting up a region" means mechanically — create the region,
create its entry block, add block arguments in the agreed order, then hand
control to the caller's body builder:

```cpp
  result.addOperands({bound, step});
  result.addOperands(initArgs);
  result.addTypes(initArgs.getTypes());

  // Create the body region with a single block.
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = new Block();
  bodyRegion->push_back(bodyBlock);

  // Add the induction variable as the first block argument.
  bodyBlock->addArgument(builder.getIndexType(), result.location);

  // Add block arguments for each init arg (the accumulators).
  for (Value arg : initArgs) {
    bodyBlock->addArgument(arg.getType(), result.location);
  }
```

If the caller passed a `bodyBuilder` lambda, the builder then sets the
insertion point to the start of `bodyBlock` (under an
`OpBuilder::InsertionGuard`) and invokes the lambda with the induction
variable and the remaining block arguments.

Note the result types are derived from `initArgs.getTypes()` — the builder
*enforces* the init/result pairing by construction, so C++ callers cannot get
it wrong. The parser and verifier have to enforce the same pairing for IR that
arrives as text.

## Custom assembly: parsing an op with a region

The declarative `assemblyFormat` DSL can handle regions in simple cases, but
our format has coupled optional pieces — `init(...)` and `-> types` must
appear together, and the region's entry-block argument *types* are determined
by them. So `TinyLoopDialect.td` sets `hasCustomAssemblyFormat = 1` and we
implement `AccumulateOp::parse` and `AccumulateOp::print` in
`TinyLoopDialect.cpp`, for the format

```
tiny_loop.accumulate %bound step(%step) [init(%args...)] [-> types] { body }
```

The parse method reads the operands and types first, then parses the region.
The region-specific part is at the end — we pre-compute the entry block's
argument types (`index` for the induction variable, then the result types for
the accumulators) and hand them to `parseRegion`:

```cpp
  // Parse the region.
  // The region has block arguments: [induction_var: index, acc0, acc1, ...]
  Region *body = result.addRegion();

  // We need to know the block argument types before parsing.
  // Block args: index (for induction var) + result types (for accumulators).
  SmallVector<OpAsmParser::Argument> regionArgs;
  // ... push one Argument with type = builder.getIndexType() (the IV),
  // ... then one Argument per entry of result.types (the accumulators).

  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Ensure the body has a terminator.
  AccumulateOp::ensureTerminator(*body, builder, result.location);
```

`ensureTerminator` is provided by the `SingleBlockImplicitTerminator` trait:
if the parsed block has no terminator (the side-effect-only loop case), it
appends an empty `tiny_loop.yield`. Two smaller details worth noticing in the
full function: the parser rejects an `init`/result count mismatch with a
*parse-time* diagnostic, and attributes use
`parseOptionalAttrDictWithKeyword` (`attributes { ... }`) because a bare `{`
must unambiguously start the region.

The printer is the mirror image; the region-relevant line is:

```cpp
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/!getInitArgs().empty());
```

We always print the entry block arguments (the parser needs the names, even
though it already knows the types), and we *hide* the terminator exactly when
it is the implicit empty `tiny_loop.yield` — keeping print and parse
round-trippable. Round-tripping is tested by `test/part4/basic-loop.mlir`
and `test/part4/accumulate.mlir`, which run `tutorial-opt` with no passes and
FileCheck the printed output; `accumulate.mlir` covers single, vector-typed,
and multiple accumulators
(`init(%init_a, %init_b) -> (i32, f32)` with a two-value yield).

## The verifier

`hasVerifier = 1` in ODS declares `LogicalResult AccumulateOp::verify()`,
implemented in `TinyLoopDialect.cpp`. It checks precisely the "four lists
agree" invariant from the design section — init count vs result count, block
argument count and types, yield operand count and types — plus the region's
shape (single block, induction variable of `index` type). A representative
check:

```cpp
  // Check that the yield has the right number and types of operands.
  auto yield = cast<YieldOp>(body->getTerminator());
  if (yield.getResults().size() != getNumResults())
    return emitOpError("yield has ")
           << yield.getResults().size() << " operands, but loop has "
           << getNumResults() << " results";
```

The `cast<YieldOp>` is safe because the
`SingleBlockImplicitTerminator<"YieldOp">` trait already guarantees the
terminator's kind by the time `verify()` runs — trait verifiers run first.

Try it — put a loop that carries a value but yields nothing inside a
`func.func` in `bad.mlir`:

```mlir
%result = tiny_loop.accumulate %n step(%step) init(%init) -> index {
^bb0(%i: index, %acc: index):
  tiny_loop.yield
}
```

```
$ build/tools/tutorial-opt bad.mlir
bad.mlir:2:13: error: 'tiny_loop.accumulate' op yield has 0 operands, but loop has 1 results
```

A note on what we did *not* have to verify: uses of values from *outside*
the loop inside the body are fine as-is. Our region is
[SSACFG-style](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions),
so ops in the body may reference any value that dominates the
`tiny_loop.accumulate` op itself — that is how `%ptr` flows into the loop
bodies in the tests without being a loop argument. MLIR's core verifier
enforces dominance; we only verify what is specific to our op.

## Lowering to `scf.for`

Now the payoff. `tiny_loop.accumulate` maps almost one-to-one onto upstream
[`scf.for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop):
our `init` args are `scf.for`'s `iter_args`, our yield is `scf.yield`, and
the only mismatch is that `scf.for` wants an explicit lower bound. The pass
is declared in
[`lib/Dialect/TinyLoop/TinyLoopPasses.td`](../../lib/Dialect/TinyLoop/TinyLoopPasses.td):

```tablegen
def TinyLoopToSCF : Pass<"tiny-loop-to-scf"> {
  let summary = "Lower TinyLoop operations to SCF dialect.";
  ...
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}
```

`dependentDialects` matters here for the same reason as in Chapter 18: the
pass creates `scf.for` and `arith.constant` ops in a module that may contain
neither dialect yet, so both must be loaded before the pass runs.

The pattern lives in
[`lib/Dialect/TinyLoop/TinyLoopToSCF.cpp`](../../lib/Dialect/TinyLoop/TinyLoopToSCF.cpp).
In Chapter 18 we lowered flat ops, where `matchAndRewrite` only had to compute
replacement values. A region op adds a new problem: the body block — with its
own block arguments — has to move into the region of the *new* op, and every
value reference inside it has to be rewired. Here is the body of
`AccumulateOpLowering::matchAndRewrite(AccumulateOp op, PatternRewriter
&rewriter)`, an `OpRewritePattern<AccumulateOp>`:

```cpp
    Location loc = op.getLoc();
    // Create the lower bound constant (always 0).
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto forOp = scf::ForOp::create(rewriter, loc, zero, op.getBound(),
                                    op.getStep(), op.getInitArgs());

    // Clone each operation to the forOp's body.
    rewriter.setInsertionPointToStart(forOp.getBody());

    IRMapping mapping;
    for (int64_t i = 0; i < op.getBody()->getNumArguments(); ++i) {
      mapping.map(op.getBody()->getArgument(i),
                  forOp.getBody()->getArgument(i));
    }

    for (Operation &bodyOp : op.getBody()->getOperations()) {
      if (isa<tiny_loop::YieldOp>(bodyOp)) {
        continue;
      }
      rewriter.clone(bodyOp, mapping);
    }

    SmallVector<Value> newOperands;
    for (Value val : op.getBody()->getTerminator()->getOperands()) {
      newOperands.push_back(mapping.lookup(val));
    }

    if (!newOperands.empty()) {
      scf::YieldOp::create(rewriter, loc, newOperands);
    }

    // Replace the accumulate op results with the scf.for results.
    rewriter.replaceOp(op, forOp.getResults());

    return success();
```

Walking through it:

1. **Create the target op first.** `scf::ForOp::create(...)` builds a
   `scf.for` whose body block already has the right block arguments (`%iv`
   plus one per iter_arg) — `scf.for`'s builder does for us exactly what our
   own custom builder does for `accumulate`.
2. **Map old block arguments to new ones.** An
   [`IRMapping`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/IRMapping.h)
   is just a value-to-value map, seeded with
   `old body arg i -> new body arg i`; the argument orders line up by design.
3. **Clone the body ops through the mapping.** `rewriter.clone(bodyOp,
   mapping)` copies each op into the `scf.for` body, replacing any operand
   found in the mapping with its image, *and* records
   `old result -> cloned result` as it goes — so later clones automatically
   reference earlier clones. The old `tiny_loop.yield` is skipped.
4. **Emit the new terminator.** The yield's operands are translated with
   `mapping.lookup`. One quirk: `scf::ForOp`'s builder already inserts an
   empty `scf.yield` when there are no iter_args, so we only create one when
   there are carried values — a second terminator would be invalid IR.
5. **Replace the op.** `rewriter.replaceOp(op, forOp.getResults())` reroutes
   all users of the loop results and erases the old op, region and all.

An alternative to the clone loop is to *move* the block instead of copying
it: `rewriter.inlineBlockBefore` splices the ops across and remaps the block
arguments in one call, leaving only the terminator to fix up. That is what a
production pass would do, but the explicit `IRMapping` version makes the
mechanics visible, which is why the tutorial keeps it (see exercise 2).

The pass itself is the same greedy-driver boilerplate as Chapter 18's
`--tiny-to-arith` — a `RewritePatternSet` with one pattern, applied with
`applyPatternsGreedily`. Since `tiny_loop.yield` is consumed while rewriting
its parent, one pattern handles both ops.

### Seeing it run

```sh
$ build/tools/tutorial-opt --tiny-loop-to-scf sum.mlir
```

Before (`sum.mlir`, a sum of squares):

```mlir
func.func @sum_squares(%n: index, %step: index) -> index {
  %zero = arith.constant 0 : index
  %result = tiny_loop.accumulate %n step(%step) init(%zero) -> index {
  ^bb0(%i: index, %acc: index):
    %sq = arith.muli %i, %i : index
    %next = arith.addi %acc, %sq : index
    tiny_loop.yield %next : index
  }
  return %result : index
}
```

After:

```mlir
func.func @sum_squares(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %0 = scf.for %arg2 = %c0 to %arg0 step %arg1 iter_args(%arg3 = %c0) -> (index) {
    %1 = arith.muli %arg2, %arg2 : index
    %2 = arith.addi %arg3, %1 : index
    scf.yield %2 : index
  }
  return %0 : index
}
```

Note `%c0` serves as both the lower bound and the init value: the pattern
did create its own `arith.constant 0`, but the greedy driver deduplicates
identical constants, merging it with the `%zero` already in the function.

The FileCheck version of this is `test/part4/lower-to-scf.mlir`, which also
covers the no-accumulator and two-accumulator cases
(`iter_args(%a = ..., %b = ...)` and the `%results:2` destructuring). If your
rewrite produces invalid IR while you are experimenting,
`--mlir-print-ir-after-failure` shows the IR the verifier rejected.

## The full pipeline: loops + vectors down to LLVM

The point of lowering onto SCF instead of straight to branches is that
everything downstream is free. Chapter 19's passes lower the `tiny` body ops,
and upstream passes take SCF the rest of the way.
`test/part4/full-pipeline.mlir` runs the dialect-level pipeline on a
vector-sum loop:

```mlir
// RUN: tutorial-opt --split-input-file --tiny-loop-to-scf --tiny-to-arith --tiny-to-llvm %s | FileCheck %s

func.func @vector_sum_pipeline(%ptr: !tiny.ptr, %n: index) -> vector<4xf16> {
  %c4 = tiny.constant 4 : index
  %zero = tiny.constant dense<0.0> : vector<4xf16>
  %result = tiny_loop.accumulate %n step(%c4) init(%zero) -> vector<4xf16> {
  ^bb0(%i: index, %acc: vector<4xf16>):
    %vec = tiny.load %ptr, %i : vector<4xf16>
    %sum = tiny.addf %acc, %vec : vector<4xf16>
    tiny_loop.yield %sum : vector<4xf16>
  }
  return %result : vector<4xf16>
}
```

Note the ordering: `--tiny-loop-to-scf` runs *first*, while the loop body
still contains `tiny.load`/`tiny.addf` — the pattern never inspects the
body's ops, so it does not care. Running `--tiny-to-arith` first would work
too; the passes are independent because each pattern only touches its own
dialect.

Appending the upstream conversions from Chapter 19 takes the structured loop
all the way to LLVM dialect branches:

```sh
$ build/tools/tutorial-opt --tiny-loop-to-scf --tiny-to-arith --tiny-to-llvm \
    --convert-scf-to-cf --convert-to-llvm test/part4/full-pipeline.mlir
```

```mlir
llvm.func @vector_sum_pipeline(%arg0: !llvm.ptr, %arg1: i64) -> vector<4xf16> {
  ...
  llvm.br ^bb1(%0, %2 : i64, vector<4xf16>)
^bb1(%3: i64, %4: vector<4xf16>):  // 2 preds: ^bb0, ^bb2
  %5 = llvm.icmp "slt" %3, %arg1 : i64
  llvm.cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %6 = llvm.getelementptr %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f16
  %7 = llvm.load %6 : !llvm.ptr -> vector<4xf16>
  %8 = llvm.fadd %4, %7 : vector<4xf16>
  %9 = llvm.add %3, %1 : i64
  llvm.br ^bb1(%9, %8 : i64, vector<4xf16>)
^bb3:  // pred: ^bb1
  llvm.return %4 : vector<4xf16>
}
```

It is worth pausing on `^bb1(%3: i64, %4: vector<4xf16>)`: after
`--convert-scf-to-cf` destroys the structured loop, the loop-carried value
survives as a **block argument on the loop-header block** — MLIR's equivalent
of an LLVM phi node. `init` args, `iter_args`, and block-argument phis are
the same concept at three levels of the stack; from here Chapter 19's
`mlir-translate`/clang steps would carry this to a native binary unchanged.

Run all four of this chapter's tests with lit:

```sh
$ source .venv/bin/activate
$ lit build/test --filter='accumulate|basic-loop|lower-to-scf|full-pipeline' -v
...
PASS: MLIRUnifiedTutorial :: part4/full-pipeline.mlir (4 of 4)
```

## Driving it from the Python DSL

The `accumulate` decorator you saw at the top is implemented in
[`python/tiny/ch2/dsl.py`](../../python/tiny/ch2/dsl.py) with MLIR's Python
bindings, and it is a compact re-telling of this whole chapter:

```python
op = Operation.create(
    "tiny_loop.accumulate",
    results=result_types,
    operands=[bound._value, step._value] + init_values,
    regions=1,  # One region for the body
)

# Set up the block with arguments: (index, *iter_args)
block = Block.create_at_start(op.regions[0], [IndexType.get()] + init_types)

with InsertionPoint(block):
    ...  # run the user's body function, then:
    Operation.create("tiny_loop.yield", operands=yield_values)
```

Run the matmul demo (from the repo root, venv active):

```sh
$ TUTORIAL_OPT=$PWD/build/tools/tutorial-opt \
    python part4-lowering/ch20-region-ops-to-scf/examples/matmul.py
```

It prints four stages — `tiny`+`tiny_loop`, after `--tiny-loop-to-scf`, after
`--tiny-to-arith`, and LLVM dialect. For the inner loop, the first stage
prints

```mlir
%3 = tiny_loop.accumulate %arg5 step(%1) init(%2) -> vector<1xf16> { ... }
```

and the second stage

```mlir
%3 = scf.for %arg8 = %c0 to %arg5 step %1 iter_args(%arg9 = %2) -> (vector<1xf16>) { ... }
```

The three nested `@accumulate` decorators become three nested `scf.for` ops;
by the LLVM stage the reduction is an `llvm.intr.vector.reduce.fadd` inside
a branch-based loop nest. Use the demo as a playground — a good warm-up is a
dot product `sum(a * b)`, which is exactly the innermost matmul loop without
the two outer ones.

## Exercises

Unlike the original mlir-tutor repo, this repo keeps the solutions in-tree,
so the exercises are "delete and rebuild" or extensions. Exercises 1 and 4
are the original chapter 2 exercise and challenge. For exercise 1, the
original TODO file is also available as a skeleton.

1. **Reimplement the lowering.** Delete the body of
   `AccumulateOpLowering::matchAndRewrite` in
   `lib/Dialect/TinyLoop/TinyLoopToSCF.cpp` (keep a copy!) and rewrite it
   from the numbered recipe in the comment above it. Rebuild, then check
   yourself with `lit build/test --filter='lower-to-scf|full-pipeline' -v`.
   To do the exercise exactly as the original had it (its `ch2-exercise`
   branch), start from
   [`exercises/skeletons/TinyLoopToSCF.cpp`](exercises/skeletons/TinyLoopToSCF.cpp).
   It removes the whole pattern struct *and* the pass body, so you also
   write the `RewritePatternSet` + `applyPatternsGreedily` boilerplate.
   The skeleton is not compiled. Copy it over
   `lib/Dialect/TinyLoop/TinyLoopToSCF.cpp`, rebuild (it builds as
   shipped, and the tests fail because the pass does nothing), and use
   `git checkout -- lib/Dialect/TinyLoop/TinyLoopToSCF.cpp` to restore the
   solution. When the output looks wrong, `--mlir-print-ir-after-failure`
   shows the IR your pattern produced at the point of failure.
2. **Swap clone for inline.** Change the pattern to use
   `rewriter.inlineBlockBefore` instead of the per-op clone loop, replacing
   the old block arguments with the `scf.for` body arguments. You will need
   to erase or rewrite the `tiny_loop.yield` — and mind the empty-terminator
   quirk from step 4 above. The lit tests must still pass unchanged.
3. **Probe the parser.** The verifier catches a yield whose types disagree
   with the results — but what rejects
   `tiny_loop.accumulate %n step(%s) init(%x) -> f32` when `%x` is an
   `index`? Find where in `AccumulateOp::parse` that mismatch surfaces
   (hint: `resolveOperand` checks the type it is given against the value's
   prior uses), then write a negative test with
   `// RUN: tutorial-opt %s --verify-diagnostics` and an `// expected-error`
   annotation, as covered in Part 1, Chapter 9.
4. **Challenge: `tiny_loop.if_zero` → `scf.if`.** Add a new op

   ```mlir
   tiny_loop.if_zero %i {
     // then-region
   } else {
     // else-region
   }
   ```

   that executes the `then` region when the given index value is 0 and the
   `else` region otherwise — see
   [`scf.if`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-scfifop)
   for motivation. You will need two regions in ODS:

   ```tablegen
   let regions = (region SizedRegion<1>:$thenRegion,
                         SizedRegion<1>:$elseRegion);
   ```

   Then a verifier that both regions yield matching types, and a lowering
   that emits `arith.cmpi eq, %i, %c0` followed by an `scf.if` — cloning
   *two* region bodies this time. Add a lit test in `test/part4/`.

## Where to go next

`tiny_loop.accumulate` hard-codes one loop shape. Upstream MLIR instead
abstracts over "things that loop" with **op interfaces** like
`LoopLikeOpInterface`, so passes can hoist code out of *any* loop without
knowing its dialect. Interfaces are where Part 5 begins:
[Part 5, Chapter 21](../../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md)
extends the tiny DSL with op interfaces and GPU-style tile operations, and
the chapters after it climb from hand-written loops to `linalg` and the
transform dialect — where the loops in this chapter's matmul are *generated*
rather than written.
