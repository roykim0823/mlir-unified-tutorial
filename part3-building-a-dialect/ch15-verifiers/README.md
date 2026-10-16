# Chapter 15: Verifiers

> Reworked from the mlir-tutorial fork's chapter 8 for the unified repo.

[Chapter 14](../ch14-folders/README.md) taught `poly` to compute; this
chapter teaches it to say **no**. Verifiers are the checks that keep
every MLIR program well-formed — and they finally explain the last
mysteries in `PolyOps.td`: `let hasVerifier = 1` and that custom
`Has32BitArguments` trait spliced into `poly.eval`'s trait list back in
[Chapter 13](../ch13-traits/README.md).

The centerpiece of this chapter is an experiment: feeding `poly.eval`
three *differently* wrong inputs to catch each of MLIR's **three
verification layers** red-handed, one at a time.

## What you will learn

- What verifiers are and when they run (more often than you'd think).
- The three layers of verification — ODS constraints, trait verifiers,
  custom op verifiers — and their firing order, demonstrated empirically.
- How to write an op verifier (`hasVerifier`, `verify()`, `emitOpError`).
- How to write a *custom trait* with a verifier (`NativeOpTrait`,
  `verifyTrait`).
- How to test error messages with lit + FileCheck — including the `not`
  tool for tests whose subject *fails*.
- How to choose between a type constraint, a trait, and an op verifier.

## Prerequisites

- [Chapter 12](../ch12-defining-poly/README.md) (the `poly` dialect and
  its `poly.eval` op), [Chapter 13](../ch13-traits/README.md) (traits),
  and [Chapter 14](../ch14-folders/README.md) (folders — we reuse its
  `LogicalResult` idiom).
- A configured build in `build/` per the top-level
  [README](../../README.md), with `tutorial-opt` built at
  `build/tools/tutorial-opt`.

Run every command from the repo root, with the virtualenv active
(`source .venv/bin/activate`) so `lit` is on your `PATH`.

---

## 1. Concepts: what verifiers are and when they run

A **verifier** checks that an operation is well-formed: operand types
make sense, attributes are in range, invariants hold. You have been
watching well-formedness checks work since Part 1, where `mlir-opt`
with *no passes at all* still rejected bad IR. This repo keeps a minimal
specimen of that around as a real test,
[`test/part3/wrong_type.mlir`](../../test/part3/wrong_type.mlir):

***test/part3/wrong_type.mlir***
```mlir
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s

// CHECK: use of value '%b' expects different type than prior uses: 'i32' vs 'i64'
func.func @bad(%a: i32, %b: i64) -> i32 {
  %0 = arith.muli %a, %b : i32
  func.return %0 : i32
}
```

Run its subject by hand — no passes requested, error anyway:

```bash
build/tools/tutorial-opt test/part3/wrong_type.mlir
```

```
test/part3/wrong_type.mlir:5:23: error: use of value '%b' expects different type than prior uses: 'i32' vs 'i64'
  %0 = arith.muli %a, %b : i32
                      ^
test/part3/wrong_type.mlir:4:25: note: prior use here
func.func @bad(%a: i32, %b: i64) -> i32 {
```

To be precise, *this particular* diagnostic comes from the parser
itself — call it layer 0: `%b` was declared `i64`, the `: i32` asks the
parser to resolve it as `i32`, and SSA use/def resolution refuses before
any op is even constructed. Verifiers proper are the layers that run on
IR that *does* parse: the module is built, then every operation's
invariants are checked. (We'll return to this file's `RUN` line — the
`not` tool and the pipe into FileCheck — in section 5.)

The detail that makes verifiers strategically important is *when* they
run: **after parsing, and before and after every pass** (in debug
builds; release builds check after passes). They are not just input
validation — they check that every pass, folder, and rewrite pattern
*produces* legal IR too. This buys two things:

- Bugs surface at the pass that introduced them, not ten passes later
  as a mysterious crash.
- Passes get to be *simpler*: Chapter 14's `EvalOp` handling never
  worried about a floating-point evaluation point, because verified IR
  can't contain one. An invariant checked once in a verifier is an edge
  case deleted from every pass that touches the op.

The word "verifier" actually names three layers, all of which you have
already written without necessarily noticing:

1. **ODS invariants** — generated from your tablegen `arguments` /
   `results` constraints (Chapter 12). `IntOrComplex:$point` *is* a
   verifier.
2. **Trait verifiers** — any trait's `verifyTrait` hook (Chapter 13);
   `SameOperandsAndResultType` rejecting mixed types *is* a verifier.
3. **Custom op verifiers** — arbitrary C++ you write for one op:
   `let hasVerifier = 1`. This chapter's new material.

They run in that order, and the order is observable — section 4 proves
it with an experiment. (The official docs cover
[verifiers for operations](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-verifier-code)
and, separately,
[verifiers for attributes and types](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#verification)
— this chapter's subject is the former.)

## 2. A custom op verifier for `poly.eval`

`poly.eval` evaluates a polynomial at a point (Chapter 12). Its ODS
constraint `IntOrComplex` admits *any* integer width — but our
semantics (coefficients mod 2³²) only make sense evaluating at a 32-bit
point, or a complex number (for Chapter 16's conjugation identities).
Widths are not something a simple type constraint expresses, so
[`lib/Dialect/Poly/PolyOps.td`](../../lib/Dialect/Poly/PolyOps.td)
declares:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
def IntOrComplex : AnyTypeOf<[AnyInteger, AnyComplex]>;

def Poly_EvalOp : Op<Poly_Dialect, "eval", [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$polynomial, IntOrComplex:$point);
  let results = (outs IntOrComplex:$output);
  let assemblyFormat = "$polynomial `,` $point attr-dict `:` `(` qualified(type($polynomial)) `,` type($point) `)` `->` type($output)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

(`hasCanonicalizer` is Chapter 16's business; ignore it today.)

`hasVerifier = 1` makes tablegen emit a declaration —
`::llvm::LogicalResult verify();` on the generated `EvalOp` class —
that *you* must implement. The contract is small: `verify()` is an
ordinary member method on the op class — so it has full access to typed
accessors like `getPoint()` — that inspects the op and returns a
`LogicalResult`. The implementation, from
[`lib/Dialect/Poly/PolyOps.cpp`](../../lib/Dialect/Poly/PolyOps.cpp):

***lib/Dialect/Poly/PolyOps.cpp*** (excerpt)
```cpp
LogicalResult EvalOp::verify() {
  auto pointTy = getPoint().getType();
  bool isSignlessInteger = pointTy.isSignlessInteger(32);
  auto complexPt = dyn_cast<ComplexType>(pointTy);
  return isSignlessInteger || complexPt ? success()
                                        : emitOpError(
                                              "argument point must be a 32-bit "
                                              "integer, or a complex number");
}
```

The body is one straight-line check, so read it in execution order.
`getPoint().getType()` fetches the point operand's type through the
typed accessor. The first test, `isSignlessInteger(32)`, hides a
vocabulary point: MLIR integers come in *signless* (`i32` — no inherent
signedness; ops decide, like `arith.addi` vs comparison flavors),
*signed* (`si32`), and *unsigned* (`ui32`) variants; almost all real
dialects use signless, and this verifier demands specifically a
*signless* 32-bit integer. Keep that distinction in mind for section
4's experiment. The second test, the `dyn_cast<ComplexType>`, admits a
complex point instead. If either passes, `success()`; otherwise the
error branch runs, and there **`emitOpError` is load-bearing**:
returning bare `failure()` fails verification *silently* — the pipeline
aborts with no explanation — while `emitOpError` attaches a proper
diagnostic (`'poly.eval' op` prefix, source location, caret) and
conveniently *returns* a failure, so the idiom is
`return emitOpError("...")` or the conditional above.

## 3. A custom trait verifier

The width rule "integer operands must be 32-bit" isn't really about
`eval` — it's a policy you might want on many ops. Chapter 13 said traits
carry verification routines; now we write one. Two pieces. In
[`PolyOps.td`](../../lib/Dialect/Poly/PolyOps.td), declare the trait's
existence and C++ home:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
// Inject verification that all integer-like arguments are 32-bits
def Has32BitArguments : NativeOpTrait<"Has32BitArguments"> {
  let cppNamespace = "::mlir::tutorial::poly";
}
```

`NativeOpTrait` means "the implementation is hand-written C++ —
tablegen, just splice the name into the generated op's trait list" (you
saw the mixin lists in Chapter 13's generated-code tour; `eval` is the
only op that lists this one).

On the C++ side, a hand-written trait has a fixed shape: a class
template over `ConcreteType` (the same CRTP pattern as Part 2's pass
base classes), deriving from `OpTrait::TraitBase`, that the generated op
class mixes in — and its verification hook is a *static*
`verifyTrait(Operation *op)`, which receives a **generic** `Operation*`,
not an `EvalOp`, because the same trait must work on any op that lists
it. The implementation,
[`lib/Dialect/Poly/PolyTraits.h`](../../lib/Dialect/Poly/PolyTraits.h)
in full (minus the include guard):

***lib/Dialect/Poly/PolyTraits.h***
```cpp
#include "mlir/IR/OpDefinition.h"

namespace mlir::tutorial::poly {

template <typename ConcreteType>
class Has32BitArguments : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // OK to skip non-integer operand types
      if (!type.isIntOrIndex()) continue;

      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }

    return success();
  }
};

}
```

Compare it with section 2's op verifier — the differences are the whole
trait-vs-verifier tradeoff:

- The genericity has its price right there in the body: no `getPoint()`;
  the trait can only loop over `getOperandTypes()` and apply type-level
  checks. Generic, hence reusable on any op — and blind to any op's
  specifics.
- `emitOpError` here is called on the `Operation*` and streamed into
  (`<< "..."`), same diagnostic machinery, different spelling.
- Note the deliberate skips: non-integer types `continue` (so the
  polynomial operand and complex points sail through), and
  `isIntOrIndex` + `isInteger(32)` checks *width only* — `si32` is a
  32-bit integer to this trait. That looseness is about to become
  visible.

## 4. Step: catch all three layers in the act

Here is the experiment. `poly.eval %p, %x` with three different illegal
point types, each chosen to slip past the earlier layers and be caught
by the next one. All errors below are real output from this repo's
`tutorial-opt`; the scratch files live in `/tmp` because they are
deliberately broken IR, not tests we want to keep. (As a control, first
swap in a signless `i32` point — `tutorial-opt` round-trips the module
silently, all three layers satisfied.)

**Layer 1 — ODS constraint.** A floating-point point violates
`IntOrComplex` before traits or verifiers get a look:

```bash
cat > /tmp/eval_f32.mlir <<'EOF'
func.func @main(%p: !poly.poly<10>, %x: f32) -> f32 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, f32) -> f32
  return %0 : f32
}
EOF
build/tools/tutorial-opt /tmp/eval_f32.mlir
```

```
/tmp/eval_f32.mlir:2:8: error: 'poly.eval' op operand #1 must be integer or complex-type, but got 'f32'
  %0 = poly.eval %p, %x : (!poly.poly<10>, f32) -> f32
       ^
```

That message was *generated* — nobody wrote it; it's the `IntOrComplex`
constraint from section 2's tablegen doing its verifier job, with the
constraint description interpolated.

**Layer 2 — trait verifier.** An `i16` point is an integer (ODS ✓) but
not 32 bits wide:

```bash
cat > /tmp/eval_i16.mlir <<'EOF'
func.func @main(%p: !poly.poly<10>, %x: i16) -> i16 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, i16) -> i16
  return %0 : i16
}
EOF
build/tools/tutorial-opt /tmp/eval_i16.mlir
```

```
/tmp/eval_i16.mlir:2:8: error: 'poly.eval' op requires each numeric operand to be a 32-bit integer
  %0 = poly.eval %p, %x : (!poly.poly<10>, i16) -> i16
       ^
```

Section 3's `verifyTrait`, verbatim.

**Layer 3 — custom op verifier.** Now the sneaky one: a *signed*
`si32`. It's an integer (ODS ✓), it's 32 bits wide (trait ✓ — remember,
`isInteger(32)` checks width only), but it is not a *signless* i32:

```bash
cat > /tmp/eval_si32.mlir <<'EOF'
func.func @main(%p: !poly.poly<10>, %x: si32) -> si32 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, si32) -> si32
  return %0 : si32
}
EOF
build/tools/tutorial-opt /tmp/eval_si32.mlir
```

```
/tmp/eval_si32.mlir:2:8: error: 'poly.eval' op argument point must be a 32-bit integer, or a complex number
  %0 = poly.eval %p, %x : (!poly.poly<10>, si32) -> si32
       ^
```

Section 2's `EvalOp::verify`, and only it, catches this — proof both
that the layers check subtly different things, and that the custom
verifier runs *last*: for `i16`, where both layer 2 and layer 3 would
object, layer 2's message is the one you get. (Run the `i16` case
yourself and check which error prints — that's the ordering, observed.)

There's a mildly uncomfortable consequence, called out in the repo's own
test comment (next section): for plain signless inputs the two
hand-written checks overlap almost completely, and which message a user
sees depends on ordering trivia. Real dialects periodically audit their
layers to keep each one's job crisp.

## 5. Step: testing error messages

Verifier tests differ from every previous test in one way: the
interesting output is on **stderr**, and `tutorial-opt` exits nonzero.
This repo's tests show two harness idioms for that, one per file.

**Idiom 1: redirect stderr, split the commands.**
[`test/part3/poly_verifier.mlir`](../../test/part3/poly_verifier.mlir),
in full:

***test/part3/poly_verifier.mlir***
```mlir
// RUN: tutorial-opt %s 2>%t; FileCheck %s < %t

func.func @test_invalid_evalop(%arg0: !poly.poly<10>, %cst: i64) -> i64 {
  // This is a little brittle, since it matches both the error message
  // emitted by Has32BitArguments as well as that of EvalOp::verify.
  // I manually tested that they both fire when the input is as below.
  // CHECK: to be a 32-bit integer
  %0 = poly.eval %arg0, %cst : (!poly.poly<10>, i64) -> i64
  return %0 : i64
}
```

`2>%t` redirects stderr to lit's temp file, and the command is separated
from FileCheck by `;` rather than `|` — because `tutorial-opt` *fails*
(nonzero exit), and with a plain pipe lit would report the RUN line
itself as failed. The `;` swallows the expected failure, then FileCheck
judges the diagnostic text.

**Idiom 2: the `not` tool.** Section 1's `wrong_type.mlir` used the
other spelling:

```mlir
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s
```

`not` is a tiny LLVM utility (shipped in the MLIR pip wheel, next to
`FileCheck`) that runs its argument and *inverts* the exit code —
failure becomes success, so the ordinary pipe works, and the test
additionally *asserts* that the tool failed. `2>&1` folds stderr into
the pipe. This is the more idiomatic form in upstream MLIR; prefer it
for new failure tests.

Two more details worth stealing for your own tests:

- The CHECK lines match a *substring* of the error, loosely — Chapter
  9's loose-vs-strict tradeoff, applied to diagnostics (error wording
  changes more often than op syntax).
- `poly_verifier.mlir`'s comment is candid about its brittleness: with
  an `i64` input both layer 2 and layer 3 would fire, and the test
  matches text common to both messages (`to a 32-bit integer`). Section
  4's experiment is the sharper instrument.

Upstream MLIR has a more precise tool worth knowing:
`mlir-opt --verify-diagnostics` with `// expected-error@...` annotations
pinned to lines. This repo's simpler stderr-plus-FileCheck approach
needs no new machinery, at the cost of the looseness above.

Run both tests through lit (from the repo root, virtualenv active):

```bash
lit build/test --filter='verifier|wrong_type' -v
```

```
-- Testing: 2 of 17 tests, 2 workers --
PASS: MLIRUnifiedTutorial :: part3/wrong_type.mlir (1 of 2)
PASS: MLIRUnifiedTutorial :: part3/poly_verifier.mlir (2 of 2)
```

And, as always (Chapter 9), when a test misbehaves, replay its RUN line
by hand with the wheel tools on `PATH`:

```bash
export PATH=$(.venv/bin/python -m mlir_wheel --root-dir)/bin:$PATH
not build/tools/tutorial-opt test/part3/wrong_type.mlir 2>&1 | FileCheck test/part3/wrong_type.mlir
```

Silence is FileCheck's way of passing.

## 6. Choosing your layer

With three places to put a check, a decision rule (distilled from
experience plus the experiment):

- **Type constraint (ODS)** — when the rule is "operand must be
  such-and-such type", expressible with existing
  [constraint](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints)
  combinators (the docs describe traits as subclasses of a `Constraint`
  base class — constraints and traits are one family in ODS).
  Free error messages, visible in the `.td`, also enforced at *build*
  sites (generated builders). First choice.
- **Trait verifier** — when the rule spans *multiple ops* and needs only
  generic `Operation*`-level inspection ("all integer operands 32-bit").
  Reusable; but casting-averse and blind to op specifics — supporting
  op-specific arguments from a trait requires awkward casting, at which
  point...
- **Op verifier** — when the rule is op-specific, multi-operand, or
  needs the typed accessors (`getPoint()`). Maximum power, zero reuse.

And one non-choice: never enforce semantic invariants in *passes*. The
verifier is the single place an invariant lives; passes assume it.

## Where to go next

`poly` now computes and self-checks. The remaining static-power move is
teaching MLIR *algebraic identities* — that `x² − y²` is `(x+y)(x−y)`
(one multiplication instead of two), or that conjugation commutes with
evaluation. Those are rewrite patterns attached to canonicalization, and
half of them are written in *tablegen*, not C++:
[Chapter 16: Canonicalizers and DRR](../ch16-canonicalizers-drr/README.md)
— where `PolyOps.td`'s `hasCanonicalizer = 1` flags and
`lib/Dialect/Poly/PolyPatterns.td` get their turn.

## Exercises

1. Reproduce section 4's three-layer experiment. Then find a fourth
   illegal input that layer 1 rejects with a *different* message (hint:
   what about the result type? `AllTypesMatch` is also a verifier).
2. The trait skips `index`-typed operands oddly: `isIntOrIndex` passes,
   but check what `isInteger(32)` returns for `index`. Determine by
   experiment whether
   `poly.eval %p, %i : (!poly.poly<10>, index) -> index` even reaches
   the trait, and which layer ultimately rejects it.
3. Strengthen `test/part3/poly_verifier.mlir` (in a scratch copy): add a
   second function using the `si32` trick so the file distinguishes the
   two messages, with separate `--check-prefix` prefixes (Chapter 9's
   technique) — and rewrite its RUN line in `wrong_type.mlir`'s
   `not ... | FileCheck` style while you're there.
4. Write (on paper) a `verifyTrait` for a hypothetical
   `HasMatchingDegrees` trait that checks all `!poly.poly<N>` operands
   share one `N`. What stops you from using `getDegreeBound()` directly,
   and how do you get it anyway? (Hint: `dyn_cast<PolynomialType>` on
   the types — generic traits can still cast *types*, just not *ops*.)
5. Predict, then check: does the verifier run on IR that only *parses*
   (`tutorial-opt` with no passes, as in section 4), and does it run
   again after `--canonicalize`? Design an experiment with a folder that
   would produce illegal IR if unguarded (Chapter 14's `from_tensor` fold
   is a good subject: what if the tensor is longer than the degree
   bound?).
