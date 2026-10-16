# Chapter 18: Dialect Conversion

> Reworked from the mlir-tutorial fork's chapter 10, with the tiny dialect example from mlir-tutor's chapter 1.

Six chapters of Part 3 made `poly` a complete high-level IR — and
completely unable to run. This chapter starts the descent: **lowering**
`poly` into upstream dialects (`arith`, `tensor`, `scf`), so that
[Chapter 19](../ch19-to-llvm-native-binary/README.md) can carry it the
rest of the way to an executable. On the way we meet the one piece of
pass machinery Chapter 10 couldn't teach, because it only matters when
*types* change: the **dialect conversion framework**.

This chapter also introduces the repo's *second* dialect, `tiny` — a
small fp16-vector DSL ported from the mlir-tutor exercises — as the
control group. Its arithmetic lowering, `--tiny-to-arith`, changes no
types, and therefore needs *none* of the framework: it is a plain
Chapter 10 greedy-rewrite pass. Seeing both side by side is the fastest
way to learn which tool a lowering actually needs.

The code lives in
[`lib/Conversion/PolyToStandard/`](../../lib/Conversion/PolyToStandard/)
(the `Conversion/` directory promised by Chapter 10 §2's layout tour;
flag: `--poly-to-standard`) and
[`lib/Dialect/Tiny/TinyToArith.cpp`](../../lib/Dialect/Tiny/TinyToArith.cpp)
(flag: `--tiny-to-arith`).

## What you will learn

- Why type changes make lowering fundamentally harder than rewriting —
  and how the conversion framework solves it.
- The four moving parts: `TypeConverter`, `OpConversionPattern`,
  `ConversionTarget`, and `applyPartialConversion`.
- How each `poly` op lowers — including polynomial multiplication as a
  loop nest and evaluation as Horner's method, with real output.
- Why `func.func` needs special treatment (types hide in signatures).
- How to read the framework's signature error — "failed to legalize
  unresolved materialization" — with a live reproduction.
- The contrast case: when your ops carry no custom types, a plain
  `OpRewritePattern` lowering suffices (`tiny` → `arith`), and what you
  give up by skipping the framework (the legality contract).

## Prerequisites

- Part 3 in full — [Chapter 12](../../part3-building-a-dialect/ch12-defining-poly/README.md)
  through [Chapter 17](../../part3-building-a-dialect/ch17-pdll/README.md).
  This chapter leans especially on Chapter 12 (the `!poly.poly<N>` type
  and its coefficient-tensor representation), Chapter 14 §3 (the
  compile-time convolution we now emit as runtime code), and Chapter 15
  (verifiers — why type mismatches can't be left lying around).
- [Chapter 10](../../part2-writing-passes/ch10-first-pass-cpp/README.md)'s
  pattern-rewrite machinery: `OpRewritePattern`, `matchAndRewrite`, the
  greedy driver. Conversion patterns are those patterns, re-based.
- [Chapter 2](../../part1-using-mlir/ch02-lowering-pipelines/README.md)'s
  pipeline vocabulary — including `scf.for`'s `iter_args` idiom and the
  `builtin.unrealized_conversion_cast` ops that appeared in its Part B
  pipeline. This chapter finally explains who makes those.
- A configured build in `build/` per the top-level
  [README](../../README.md). No rebuilding is needed — `tutorial-opt` is
  already at `build/tools/tutorial-opt`. Run every command from the repo
  root with the virtualenv active (`source .venv/bin/activate` — needed
  for `lit`; the `tutorial-opt` runs work without it).

---

## 1. Concepts: what makes lowering hard

Here is the plan, and it sounds like a Chapter 10 exercise: rewrite
`poly.add` into `arith.addi`, `poly.mul` into loops, and so on. As this
codebase's author puts it: *if not for types, dialect conversion would be
essentially the same as a normal pass*.

But types ruin everything. Our `TypeConverter` will say a
`!poly.poly<10>` becomes a `tensor<10xi32>` (ten 32-bit coefficients —
Chapter 12's representation, made literal). Now rewrite one op:

```mlir
%2 = poly.add %0, %1 : !poly.poly<10>      // becomes...
%2 = arith.addi %0, %1 : tensor<10xi32>    // ...this
```

The moment you do, the IR is inconsistent: `%0` and `%1` are still
`!poly.poly<10>` (their defining ops aren't converted yet), so the new
`arith.addi` is ill-typed; and every not-yet-converted *user* of `%2`
still expects a `!poly.poly<10>`. Any conversion order you pick leaves a
frontier of type mismatches — and Chapter 15 taught us mismatches don't
survive a verifier. A plain greedy rewrite (Chapter 10) has no answer;
you'd need every pattern to defensively cast, and patterns would have to
fire in dependency order.

The **dialect conversion framework**
([official docs](https://mlir.llvm.org/docs/DialectConversion/)) exists to
manage that frontier. The idea:

- You declare which types change and how (**TypeConverter**).
- You write patterns that see their operands *as if already converted*
  (**OpConversionPattern** — the framework rewrites the graph in a sorted
  order and hands each pattern the in-progress converted values).
- Type mismatches on the frontier are patched with temporary
  [`builtin.unrealized_conversion_cast`](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinunrealized_conversion_cast-unrealizedconversioncastop)
  ops — essentially a forced type coercion, the internal stand-in for a
  type conflict (you met these in Chapter 2's pipeline — now you know
  who makes them) — that the framework removes as conversion completes.
- You declare what "done" means (**ConversionTarget**), and the framework
  checks it — any op still illegal at the end is an error.

That last point is the second big idea: **legality**. A conversion pass
doesn't just rewrite opportunistically; it has a *contract* — "after me,
no `poly` ops remain" — and the framework enforces it. Hold that thought
for section 7, where a lowering *without* a contract turns out to be
exactly what the `tiny` dialect wants.

## 2. The pass shell and the TypeConverter

The pass is declared in
[`PolyToStandard.td`](../../lib/Conversion/PolyToStandard/PolyToStandard.td)
(Chapter 11 machinery):

***lib/Conversion/PolyToStandard/PolyToStandard.td*** (excerpt)
```tablegen
def PolyToStandard : Pass<"poly-to-standard"> {
  let summary = "Lower `poly` to standard MLIR dialects.";
  ...
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tutorial::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
  ];
}
```

Note the four `dependentDialects`: the pass *creates* `arith`, `tensor`,
and `scf` ops, so those dialects must be loaded before it runs —
Chapter 11's rule, now with real stakes.

The interesting code is in
[`PolyToStandard.cpp`](../../lib/Conversion/PolyToStandard/PolyToStandard.cpp),
starting with the type rule. A `TypeConverter` is the first of section
1's four moving parts — the object that "declares which types change and
how" — and using one amounts to subclassing it and registering one
callback per type rule via `addConversion`:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](PolynomialType type) -> Type {
      int degreeBound = type.getDegreeBound();
      IntegerType elementTy =
          IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
      return RankedTensorType::get({degreeBound}, elementTy);
    });

    // We don't include any custom materialization hooks because this lowering
    // is all done in a single pass. ...
  }
};
```

Two `addConversion` calls, tried in reverse order of registration: the
`PolynomialType` one maps `!poly.poly<N>` → `tensor<Nxi32>` (the degree
bound *from the type parameter* becomes the static tensor length — this
is where Chapter 12's decision to put N in the type pays off); the
identity conversion says every other type is fine as-is. Note the
signless `i32` — Chapter 15 §2's distinction, consciously chosen.

The comment cut short above is worth reading in full in the source: no
custom **materializations** are registered. Materialization hooks tell
the framework how to build real ops bridging old→new types when a
mismatch must persist — say, because you split your lowering across
several passes and a `!poly.poly<10>` value must legitimately flow into
not-yet-lowered code. Because this lowering converts everything in one
pass, the temporary `unrealized_conversion_cast`s all cancel out by the
end, and no custom hook is needed. (Section 6 shows what happens when
they *don't* cancel. A version-drift note if you go exploring these
hooks: their upstream API has shifted over the years — argument
materializations were merged into source materializations in later LLVM
releases.)

## 3. Conversion patterns

A conversion pattern is Chapter 10's rewrite pattern re-based onto the
framework: still one class per op, still a `matchAndRewrite` whose job
is to build replacement ops and replace the root — but it subclasses
`OpConversionPattern` instead of `OpRewritePattern`, and that changes
the method's signature. Here's the add lowering, in full:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op.getOperation(), addOp);
    return success();
  }
};
```

Both differences sit in that `matchAndRewrite` signature, and they carry
the whole framework:

- **The `OpAdaptor` parameter** (an alias for `AddOp::Adaptor`, generated
  code you met in Chapter 14 as `FoldAdaptor`'s sibling). Its accessors —
  `adaptor.getLhs()` — return the *already-type-converted* operands: by
  the time this runs, they are `tensor<10xi32>` values, no matter what
  the original IR looked like. The `op` parameter still has the original
  (poly-typed) operands, for when you need to inspect pre-conversion
  facts. That split is the framework's central service: patterns never
  see the inconsistent frontier.
- **`ConversionPatternRewriter`** — a `PatternRewriter` with extra
  conversion-aware methods. Rule of thumb: inside a conversion pattern,
  do *everything* through it, even more strictly than Chapter 10's
  "technically not allowed" warning about mutating IR behind the
  rewriter's back.

Why does `arith.addi` accept tensors at all? Because it's
`ElementwiseMappable` — Chapter 13's trait, on the *other* side of the
trade. Polynomial addition is coefficient-wise addition, so the whole
lowering is one op. `ConvertSub` is identical with `subi`;
`ConvertToTensor` is even better — `to_tensor` becomes *nothing*
(`rewriter.replaceOp(op, adaptor.getInput())`): after conversion a
polynomial already *is* its coefficient tensor. `ConvertFromTensor` only
has work when the input tensor is shorter than the degree bound, in
which case it zero-pads with `tensor.pad` (visible in section 5's test).

### Multiplication: creating a loop nest

`poly.mul` is a cyclic convolution (Chapter 14 §3 computed it at compile
time; now we emit code to do it at *runtime*). The pattern —
`ConvertMul` in the source — builds Chapter 14's double loop as IR. It's
long but made of three readable pieces:

1. an all-zeros result tensor: `arith.constant dense<0> : tensor<Nxi32>`;
2. loop scaffolding:
   [`scf.for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)
   ops built with `b.create<scf::ForOp>`, with the accumulating tensor
   threaded through `iter_args` (Chapter 2's idiom, now built
   programmatically — the lambda you pass `create<ForOp>` receives the
   loop induction variable and loop state and fills the body);
3. the body: extract both coefficients, multiply, add into
   `result[(i + j) mod N]` — `tensor.extract`, `arith.remui`,
   `tensor.insert`.

One style note you'll use forever: `ImplicitLocOpBuilder b(op.getLoc(),
rewriter);` wraps the rewriter so you stop passing the location to every
`create` — builders with 10+ creates get much cleaner.

Run it (all output in this chapter is real, from a degree-4 example to
keep it readable — put this in `/tmp/lower_mul.mlir`):

```mlir
func.func @lower_mul(%p: !poly.poly<4>, %q: !poly.poly<4>) -> !poly.poly<4> {
  %0 = poly.mul %p, %q : !poly.poly<4>
  return %0 : !poly.poly<4>
}
```

```bash
build/tools/tutorial-opt --poly-to-standard /tmp/lower_mul.mlir
```

```mlir
func.func @lower_mul(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %cst) -> (tensor<4xi32>) {
    %1 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4xi32>) {
      %2 = arith.addi %arg2, %arg4 : index
      %3 = arith.remui %2, %c4 : index
      %extracted = tensor.extract %arg0[%arg2] : tensor<4xi32>
      %extracted_0 = tensor.extract %arg1[%arg4] : tensor<4xi32>
      %4 = arith.muli %extracted, %extracted_0 : i32
      %extracted_1 = tensor.extract %arg5[%3] : tensor<4xi32>
      %5 = arith.addi %4, %extracted_1 : i32
      %inserted = tensor.insert %5 into %arg5[%3] : tensor<4xi32>
      scf.yield %inserted : tensor<4xi32>
    }
    scf.yield %1 : tensor<4xi32>
  }
  return %0 : tensor<4xi32>
}
```

Read the inner body against Chapter 14's fold —
`result[(i + j) % degree] += lhs[i] * rhs[j]` — it's the same algorithm,
emitted instead of executed. (A "functional tensors" note: `tensor.insert`
doesn't mutate; it yields a *new* tensor value, which is why the
accumulator threads through `iter_args`. Turning this into actual
in-place memory is Chapter 19's
[bufferization](https://mlir.llvm.org/docs/Bufferization/) step — one of
the harder lowering pipelines in upstream MLIR, and part of why the
dialect conversion framework carries the complexity it does.)

### Evaluation: Horner's method

`ConvertEval` lowers `poly.eval` to the classic trick for evaluating a
polynomial with N multiplies instead of N²:

> accum = 0; for i = 1..N: accum = point·accum + coeff[N−i]

i.e. c₀ + x(c₁ + x(c₂ + ⋯)) evaluated inside-out. Lowered (real output):

```mlir
func.func @lower_eval(%arg0: tensor<10xi32>, %arg1: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = scf.for %arg2 = %c1 to %c11 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
    %1 = arith.subi %c10, %arg2 : index
    %2 = arith.muli %arg1, %arg3 : i32
    %extracted = tensor.extract %arg0[%1] : tensor<10xi32>
    %3 = arith.addi %2, %extracted : i32
    scf.yield %3 : i32
  }
  return %0 : i32
}
```

One loop, one multiply per iteration, coefficients read
highest-to-lowest via the `subi`. (Note what's *missing*: complex-typed
points. This lowering only handles the i32 case — a gap with
consequences in section 6.)

### Constants: lower by delegation

`ConvertConstant` is the cleverest three lines in the pass: it rewrites
`poly.constant` into `arith.constant` **+ `poly.from_tensor`** — a *poly*
op! That's legal mid-conversion: the framework simply keeps converting,
and the freshly created `from_tensor` is picked up by
`ConvertFromTensor`. Lowerings can be compositional; you don't have to
re-implement your own dialect's plumbing in every pattern. Watch the
whole chain fire on a short constant in a `!poly.poly<10>` (real
output):

```mlir
%0 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
// becomes: ConvertConstant made the arith.constant, then
// ConvertFromTensor zero-padded 3 coefficients out to degree bound 10.
%cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
%c0_i32 = arith.constant 0 : i32
%padded = tensor.pad %cst low[0] high[7] { ... tensor.yield %c0_i32 ... }
    : tensor<3xi32> to tensor<10xi32>
```

## 4. Legality: the ConversionTarget

The pass body (`runOnOperation`) is where section 1's moving parts
meet: it states the legality contract as a `ConversionTarget`, collects
section 3's patterns into a `RewritePatternSet` (constructed with the
type converter), and hands the lot to `applyPartialConversion`, the
framework's driver:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
ConversionTarget target(*context);
target.addLegalDialect<arith::ArithDialect>();
target.addIllegalDialect<PolyDialect>();

RewritePatternSet patterns(context);
PolyToStandardTypeConverter typeConverter(context);
patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertEval,
             ConvertMul, ConvertFromTensor, ConvertToTensor>(typeConverter,
                                                             context);
...
if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
  signalPassFailure();
}
```

`addIllegalDialect<PolyDialect>()` is the contract: when the dust
settles, zero `poly` ops. The framework drives patterns *toward* that
goal and fails the pass if it can't get there — unlike Chapter 10's
greedy driver, which happily stops at any fixed point.
(`addLegalDialect<arith::ArithDialect>()` is the mirror declaration —
`arith` ops need no legalization at all.)

Then comes everyone's first stumble: **ops you don't own carry your
types**. Our test functions have signatures like
`(%arg0: !poly.poly<10>)` — the type hides inside `func.func`'s
*attributes* (remember Chapter 2's generic-form dump: `function_type` is
just data). No `poly` op is involved, so our patterns never fire, but
the contract isn't met until those signatures change too. Upstream
provides drop-in helpers, one per structural-op family:

***lib/Conversion/PolyToStandard/PolyToStandard.cpp*** (excerpt)
```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
    patterns, typeConverter);
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return typeConverter.isSignatureLegal(op.getFunctionType()) &&
         typeConverter.isLegal(&op.getBody());
});

populateReturnOpTypeConversionPattern(patterns, typeConverter);
target.addDynamicallyLegalOp<func::ReturnOp>(
    [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
```

plus the same pairing for `func.call`
(`populateCallOpTypeConversionPattern`) and branch-interface ops
(`populateBranchOpInterfaceTypeConversionPattern` with
`markUnknownOpDynamicallyLegal`) — read them in the source. Each pairing
has a pattern-half and a legality-half. The pattern-half
(`populate...TypeConversionPattern`) rewrites signatures and operand
lists through the type converter; the legality-half
(`addDynamicallyLegalOp`) says a `func.func` is legal *only if* its
signature contains no convertible-but-unconverted types. Dynamic
legality is the third legality mode after legal/illegal-by-dialect — a
callback deciding per op. If you write a conversion with a new type,
budget a few lines for each structural dialect your types can flow
through; forgetting one is the classic source of section 6's error.

Finally, `applyPartialConversion` vs `applyFullConversion`: *full*
requires every op in the module be legal afterward; *partial* only
requires that no *illegal* op remain, leaving unknown ops alone. The
advice this repo adopts: partial is a strict generalization, use it
(and it produces far better errors).

## 5. Step: run the conversion

[`test/part4/poly_to_standard.mlir`](../../test/part4/poly_to_standard.mlir)
has one function per pattern. The whole file converts with:

```bash
build/tools/tutorial-opt --poly-to-standard test/part4/poly_to_standard.mlir
```

Highlights of the real output — add becomes one op, with the *signature*
converted too (section 4's machinery at work):

```mlir
func.func @test_lower_add(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
  %0 = arith.addi %arg0, %arg1 : tensor<10xi32>
  return %0 : tensor<10xi32>
}
```

`to_tensor`/`from_tensor` at matching sizes vanish entirely:

```mlir
func.func @test_lower_to_tensor(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  return %arg0 : tensor<10xi32>
}
```

the size-extending `from_tensor` case shows the zero-pad
(`tensor<10xi32>` into a `!poly.poly<20>`):

```mlir
%padded = tensor.pad %arg0 low[0] high[10] { ... tensor.yield %c0_i32 ... }
    : tensor<10xi32> to tensor<20xi32>
```

and `@test_lower_add_and_fold` shows Chapter 14 cooperating: the
`poly.add` of two constants never reaches `ConvertAdd` — the folder got
it first, mid-conversion — so the output has a folded
`arith.constant dense<[5, 7, 9]>` (padded to degree 10) and no `addi`.

Run the suite entry (the `RUN` line is
`tutorial-opt --poly-to-standard %s | FileCheck %s`):

```bash
source .venv/bin/activate && lit build/test --filter=poly_to_standard -v
```

```
-- Testing: 1 of 24 tests, 1 workers --
PASS: MLIRUnifiedTutorial :: part4/poly_to_standard.mlir (1 of 1)
```

## 6. Step: read a legality failure

Every conversion author meets this error; better to meet it on purpose.
`ConvertEval` doesn't handle complex points (section 3), so feed it one
(`/tmp/complex_eval.mlir` — Chapter 15's verifier admits complex points
precisely so Chapter 16's conjugation canonicalization has something to
chew on):

```mlir
func.func @f(%p: !poly.poly<10>, %z: complex<f64>) -> complex<f64> {
  %0 = poly.eval %p, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}
```

```bash
build/tools/tutorial-opt --poly-to-standard /tmp/complex_eval.mlir
```

```
complex_eval.mlir:2:8: error: failed to legalize unresolved materialization from ('i32') to
('complex<f64>') that remained live after conversion
  %0 = poly.eval %p, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>
       ^
note: see current operation: %5 = "builtin.unrealized_conversion_cast"(%4) : (i32) -> complex<f64>
note: see existing live user here: "func.return"(%5) : (complex<f64>) -> ()
```

Decoding it: `ConvertEval` fired anyway and produced its i32 Horner loop;
the framework patched the resulting i32-vs-complex mismatch with an
`unrealized_conversion_cast` (section 1's frontier mechanism); at
end-of-conversion the cast was still there — no pattern or
materialization could "realize" it — so the contract failed, and the
error shows you the surviving cast *and its user*. When you see this in
your own work, the question to ask is: which pattern produced a value of
the wrong type, or which structural op did I forget to populate patterns
for?

The framework will also narrate its whole decision process — this
repo's `tutorial-opt` is built with assertions on, so
`--debug-only=dialect-conversion` works and prints one
`Legalizing operation` block per op, ending in SUCCESS or FAILURE.
Inside the `poly.eval` block you can watch the miscompile live: the
frontier is visibly ill-typed, exactly as section 1 promised —

```
%9 = "arith.addi"(%7, %8) ... : (complex<f64>, i32) -> complex<f64>
```

— Horner's `point·accum + coeff` add, built with the *original* complex
point where an i32 belongs. (And note the pattern *should* have refused
to match a complex eval with `return failure()` — as written it happily
miscompiles-then-fails; making it decline politely is exercise 3.)

## 7. Contrast: when a plain RewritePattern is enough

Now meet the control group. The **`tiny` dialect**
([`lib/Dialect/Tiny/TinyDialect.td`](../../lib/Dialect/Tiny/TinyDialect.td),
from the mlir-tutor exercises) is a small CPU-vector DSL: element-wise
fp16 arithmetic on vectors (`tiny.constant`, `tiny.{addf,subf,mulf,divf}`),
a horizontal reduction (`tiny.sum`), `index` arithmetic for address math
(`tiny.{addi,subi,muli,divi}`), and memory ops (`tiny.load`/`tiny.store`)
built around one custom type, `!tiny.ptr`. It exists to express little
kernels like:

```python
def square(a: Ptr, result: Ptr, offset: Index):
    x = a.load(offset, num_elements=16)
    x_squared = x * x
    result.store(offset, x_squared)
```

(Chapters 19 and 20 pick `tiny` back up: lowering its memory ops to
LLVM, and a structured-control-flow sibling, `tiny_loop`.)

Look at the type signatures, though: every *arithmetic* `tiny` op
operates on ordinary builtin types — `vector<16xf16>`, `index`. When we
lower `tiny.mulf` to `arith.mulf`, **no type changes**. And section 1
was explicit: types are the *only* reason dialect conversion differs
from a normal pass. So
[`TinyToArith.cpp`](../../lib/Dialect/Tiny/TinyToArith.cpp) uses none of
this chapter's framework — it is a Chapter 10 pass, verbatim:
`OpRewritePattern`, no adaptor, no `TypeConverter`, no target:

***lib/Dialect/Tiny/TinyToArith.cpp*** (excerpts)
```cpp
/// Lower tiny.constant to arith.constant.
struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getType(),
                                                   op.getValue());
    return success();
  }
};

/// Lower tiny.addf to arith.addf.
struct AddFOpLowering : public OpRewritePattern<AddFOp> {
  using OpRewritePattern<AddFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddFOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};
```

Note the differences from section 3, all in the direction of *less*:
operands come straight off `op` (`op.getLhs()` — there is no frontier to
hide, so no adaptor), and `replaceOpWithNewOp` collapses
create-then-replace into one call. Eight of the ten patterns are this
shape — a 1:1 rename into `arith` (with one semantic decision hiding in
`DivIOpLowering`: `tiny.divi` → `arith.divsi`, *signed* division). The
only 1:N pattern is the reduction:

***lib/Dialect/Tiny/TinyToArith.cpp*** (excerpt)
```cpp
/// Lower tiny.sum to vector.reduction<add> + vector.broadcast.
struct SumOpLowering : public OpRewritePattern<SumOp> {
  ...
    // vector.reduction<add> returns a scalar f16.
    Value scalarSum = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::ADD, input);

    // Broadcast the scalar to vector<1xf16>.
    VectorType resultType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resultType, scalarSum);
  ...
};
```

(Incidental style drift: `vector::ReductionOp::create(rewriter, loc, ...)`
is the newer op-creation spelling upstream is migrating toward;
`rewriter.create<ReductionOp>(loc, ...)` — `PolyToStandard.cpp`'s style —
is equivalent.)

And the pass driver is Chapter 10's greedy loop, nothing more:

***lib/Dialect/Tiny/TinyToArith.cpp*** (excerpt)
```cpp
void runOnOperation() override {
  // Set up rewrite patterns.
  RewritePatternSet patterns(&getContext());
  patterns.add<ConstantOpLowering, AddFOpLowering, SubFOpLowering,
               MulFOpLowering, DivFOpLowering, AddIOpLowering, SubIOpLowering,
               MulIOpLowering, DivIOpLowering, SumOpLowering>(&getContext());

  // Apply patterns greedily to lower all matching operations.
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
```

Run it on the `square` kernel from above, written in IR
(`/tmp/tiny_square.mlir`):

```mlir
func.func @square(%ptr: !tiny.ptr, %result: !tiny.ptr, %offset: index) {
  %c2 = tiny.constant 2 : index
  %i = tiny.muli %offset, %c2
  %x = tiny.load %ptr, %i : vector<16xf16>
  %xx = tiny.mulf %x, %x : vector<16xf16>
  tiny.store %xx, %result, %i : vector<16xf16>
  return
}
```

```bash
build/tools/tutorial-opt --tiny-to-arith /tmp/tiny_square.mlir
```

```mlir
func.func @square(%arg0: !tiny.ptr, %arg1: !tiny.ptr, %arg2: index) {
  %c2 = arith.constant 2 : index
  %0 = arith.muli %arg2, %c2 : index
  %1 = tiny.load %arg0, %0 : vector<16xf16>
  %2 = arith.mulf %1, %1 : vector<16xf16>
  tiny.store %2, %arg1, %0 : vector<16xf16>
  return
}
```

Real output, and it teaches three things at once:

- **The arithmetic lowered** with zero framework involvement — constant,
  `muli`, `mulf` all became `arith` ops, and every SSA value kept its
  type, so no frontier ever existed.
- **`tiny.load` and `tiny.store` survived**, and so did the `!tiny.ptr`
  arguments in the signature. This pass has no patterns for them — they
  involve the custom `!tiny.ptr` type, whose lowering (to `llvm.ptr`)
  *does* change types and is therefore Chapter 19's job, back in the
  conversion framework with a `TypeConverter`. The two-pass split is
  drawn exactly along section 1's fault line.
- **Nothing checked that anything happened.** There is no
  `ConversionTarget`, so "after me, no `tiny` arithmetic remains" is a
  hope, not a contract. Delete `MulFOpLowering` from the list and the
  pass silently returns IR with a `tiny.mulf` still in it — some later
  pass fails instead, further from the cause. That's the trade: the
  greedy version is shorter and simpler, but the framework version
  *fails where the bug is*.

(For the 1:2 pattern in action, feed it a `tiny.sum` — it comes back as
`vector.reduction <add>` feeding a `vector.broadcast`, exactly as
`SumOpLowering` promised.)

The suite entry is
[`test/part4/lower-to-arith.mlir`](../../test/part4/lower-to-arith.mlir)
— note its `CHECK: tiny.load`/`CHECK: tiny.store` lines *asserting* the
memory ops survive:

```bash
source .venv/bin/activate && lit build/test --filter=lower-to-arith -v
```

```
-- Testing: 1 of 24 tests, 1 workers --
PASS: MLIRUnifiedTutorial :: part4/lower-to-arith.mlir (1 of 1)
```

### Choosing your tool

| Question about your lowering | If yes... |
|---|---|
| Do any SSA value types change? | Framework. Non-negotiable — a greedy pass can't manage the frontier. |
| Do your types appear in signatures of ops you don't own (`func.func`, calls, branches)? | Framework, plus section 4's `populate...` helpers. |
| Do you need a checked guarantee that certain ops are gone afterward? | Framework (`ConversionTarget`), even if types don't change — ops can be declared illegal without any type conversion. |
| None of the above? | A plain `OpRewritePattern` pass (`applyPatternsGreedily`) is simpler, easier to debug, and composes with folders for free. |

`--poly-to-standard` answers yes to the first three questions;
`--tiny-to-arith` answers no. Both are called "lowerings" — the word describes the
direction of travel (abstraction going down), not the machinery.

## Where to go next

We're one flight down the staircase: `poly` programs are now `tensor` +
`scf` + `arith` programs, and `tiny` kernels are `arith` + `vector`
programs with only their memory ops still dialect-specific. But none of
that is executable yet — tensors must become memory (bufferization),
loops must become branches, `!tiny.ptr` must become a real pointer, and
everything must reach the `llvm` dialect and beyond. That's
[Chapter 19: Lowering to LLVM and a Native Binary](../ch19-to-llvm-native-binary/README.md)
— where the `--poly-to-llvm` mega-pipeline and `--tiny-to-llvm` finally
run, and a real `main.c` calls a compiled polynomial evaluator.

**Exercises**

1. Reproduce section 6's failure, then run it again with
   `--debug-only=dialect-conversion` and trace the frontier: find the
   ill-typed `(complex<f64>, i32) -> complex<f64>` ops inside the
   `poly.eval` legalization block, and the point where `func.return`
   is "marked legal by the target" even though its operand will turn
   out to be an unresolved materialization.
2. Predict, then verify: what does
   `poly.constant dense<[1,2,3]> : tensor<3xi32> : !poly.poly<10>`
   lower to? (Chase it through `ConvertConstant` → `ConvertFromTensor`:
   how does the 3-vs-10 size mismatch resolve? Section 3's last excerpt
   is the answer key — but predict before you scroll.)
3. Fix `ConvertEval`'s manners on paper: add the early
   `return failure()` for non-i32 points. What does the *pass* then do
   with a complex eval — and is that outcome (a surviving illegal
   `poly.eval` and a cleaner error) actually better? What would a real
   fix — lowering complex evals to `complex` dialect ops — need from the
   TypeConverter?
4. The mul lowering is O(N²) with a `remui` in the hot loop. Sketch (IR
   on paper, not C++) a version that splits the inner loop into the
   no-wraparound prefix and the wraparound suffix, eliminating `remui`.
   Which Chapter 10 pass idea would you use to check your hand-derived
   IR computes the same thing? (Hint: Chapter 19 gives you a better
   tool — running it.)
5. Section 4 said `partial` beats `full` conversion. Change one line in
   your head: with `applyFullConversion`, what *additional* ops in
   `test/part4/poly_to_standard.mlir` would need legality declarations,
   and what would break first?
6. *(In the spirit of the original mlir-tutor exercise, which shipped
   `TinyToArith.cpp` with only `constant`/`addf`/`sum`/`addi` implemented
   and made you write the rest.)* Delete the `subf`/`mulf`/`divf`/
   `subi`/`muli`/`divi` patterns from
   [`lib/Dialect/Tiny/TinyToArith.cpp`](../../lib/Dialect/Tiny/TinyToArith.cpp)
   (structs *and* the `patterns.add<...>` list), rebuild
   (`cmake --build build -j`), and confirm
   `lit build/test --filter=lower-to-arith -v` fails — noting *how*:
   FileCheck complains, not the pass. Reimplement the six patterns from
   scratch, without peeking, until the test passes again. Framework
   postscript: rewrite the pass using this chapter's machinery — a
   `ConversionTarget` declaring the ten lowered `tiny` ops illegal but
   `tiny.load`/`tiny.store` legal, and `applyPartialConversion` — and
   check that the missing-pattern failure mode from section 7 now points
   at the guilty op.
   Instead of deleting the patterns yourself, you can start from the
   original file:
   [`exercises/skeletons/TinyToArith.cpp`](exercises/skeletons/TinyToArith.cpp)
   (see exercise 7 for how the skeletons work).
7. **Op definitions in ODS** *(mlir-tutor chapter 1, exercise 1)*. The
   `tiny` dialect in section 7 was handed to you finished. Write half of it
   yourself.
   [`exercises/skeletons/TinyDialect.td`](exercises/skeletons/TinyDialect.td)
   is the original exercise file. It keeps the dialect, the `!tiny.ptr`
   type, `tiny.constant`, `tiny.addf`, `tiny.sum`, `tiny.addi` and
   `tiny.load`, and leaves TODOs for `tiny.subf`, `tiny.mulf`,
   `tiny.divf`, `tiny.subi`, `tiny.muli`, `tiny.divi` and `tiny.store`.
   Define each one by copying the pattern of its neighbour:
   `Tiny_VectorBinaryOp<"subf">` for the vector ops,
   `Tiny_IndexBinaryOp<"subi">` for the index ops, and `[Commutative]`
   on `mulf`/`muli` as on `addf`/`addi`. `tiny.store` follows `tiny.load`
   but takes `(ins Tiny_VectorF16:$value, Tiny_PtrType:$ptr, Index:$offset)`,
   has no results, and uses the assembly format
   ``$value `,` $ptr `,` $offset attr-dict `:` type($value)``. Everyone copies
   TableGen definitions from somewhere; the original author said so after
   five years of working on MLIR.

   The skeletons are not compiled. To work the exercise, copy the skeleton
   over the solution and rebuild:

   ```bash
   cp part4-lowering/ch18-dialect-conversion/exercises/skeletons/TinyDialect.td lib/Dialect/Tiny/
   cmake --build build -j
   ```

   Here the unified repo behaves differently from the original. There,
   chapter 1 was built on its own. Here, the *solutions* that use these ops
   are compiled in the same build: `TinyToArith.cpp`, `TinyToLLVM.cpp`
   (Chapter 19) and `lib/Dialect/TinyTile/TinyTileDialect.cpp` (Chapter 21).
   So the build first fails with errors like `unknown type name 'SubFOp'`
   and `no member named 'getOffset' in StoreOpAdaptor`. Treat those errors
   as your checklist. When every op is defined with the right name and
   operand names (`lhs`/`rhs`, `value`/`ptr`/`offset`), the errors are
   gone, and `.venv/bin/lit build/test -v --filter 'part4/basic-ops'`
   (which round-trips all seven ops) passes. To do exercises 6 and 7
   together, copy both skeletons; then `lower-to-arith` fails until the
   patterns are back too. Restore the solutions with
   `git checkout -- lib/Dialect/Tiny/`.
