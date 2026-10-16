# Chapter 16: Canonicalizers and DRR

> Reworked from the mlir-tutorial fork's chapter 9 for the unified repo.

[Chapter 14](../ch14-folders/README.md) gave `poly` folders — single-op,
constants-only simplifications. This chapter adds the general case:
**canonicalization patterns**, rewrites spanning several ops, hooked into
the same `--canonicalize` pass. Along the way we learn a second authoring
language for rewrite patterns — **DRR** (declarative rewrite rules),
tablegen instead of C++ — and resolve the last unexplained lines in the
`poly` codebase: `let hasCanonicalizer = 1` and
[`PolyPatterns.td`](../../lib/Dialect/Poly/PolyPatterns.td).

The running examples are two genuinely mathematical rewrites:

- **Difference of squares**: x² − y² = (x + y)(x − y) — trades two
  (expensive) multiplications for one, plus cheap additions. Chapter 10's
  `MulToAdd` economics, at the polynomial level.
- **Conjugate through evaluation**: for polynomials with real
  coefficients, f(z̄) = f(z)̄ — evaluating at a conjugate equals
  conjugating the evaluation. Rewriting the former to the latter puts IR
  in a *normal form* so equivalent computations look identical.

## What you will learn

- What canonicalization is, and how it differs from folding.
- `hasCanonicalizer` / `getCanonicalizationPatterns` — attaching rewrite
  patterns (Chapter 10's kind!) to an op's canonical form.
- The same pattern written twice: C++ (the original version, preserved in
  this chapter's [`code/`](code/) snapshot) and DRR (the `lib/` version)
  — and how to read the tablegen pattern language (`Pat`, `Pattern`,
  bound variables, constraints, `CPred`).
- How `-gen-rewriters` turns DRR into the C++ you would have written.
- When to choose folding vs. canonicalization vs. a standalone pass.

## Prerequisites

- [Chapter 12](../ch12-defining-poly/README.md) (the `poly` dialect and
  its tablegen op definitions), [Chapter 13](../ch13-traits/README.md)
  (traits), [Chapter 14](../ch14-folders/README.md) (folding,
  `--canonicalize`), and [Chapter 15](../ch15-verifiers/README.md)
  (verifiers — one of this chapter's rewrites is the reason `poly.eval`
  admits complex points at all).
- [Chapter 10](../../part2-writing-passes/ch10-first-pass-cpp/README.md)'s
  pattern-rewrite machinery: `OpRewritePattern`, `matchAndRewrite`, the
  greedy driver. Canonicalization patterns *are* those patterns.
- A configured build in `build/` per the top-level
  [README](../../README.md). No rebuilding is needed — `tutorial-opt`
  and this chapter's `tutorial-opt-ch16` are already in your build tree —
  but we will run `mlir-tblgen` by hand, and it ships inside the MLIR
  pip wheel.

Run every command in this chapter from the repo root, with the
virtualenv active and the wheel's `bin/` on your `PATH`:

```bash
source .venv/bin/activate
export WHEEL_ROOT=$(python -m mlir_wheel --root-dir)
export PATH=$WHEEL_ROOT/bin:$PATH
```

---

> **Heads-up before you read any code:** this chapter presents the
> canonicalization pattern twice — section 3 in C++ (the original
> version, which is **not what `lib/` contains**; it is preserved,
> buildable, in this chapter's [`code/`](code/) directory), and
> section 4 in DRR, which *is* the main tree's version
> ([`lib/Dialect/Poly/PolyPatterns.td`](../../lib/Dialect/Poly/PolyPatterns.td)).
> Both produce the same rewrites.

## 1. Concepts: canonicalization vs. folding

Chapter 14 drew folding's box deliberately small: one op, no new ops,
return an attribute or existing value. Plenty of valuable rewrites don't
fit:

- x² − y² → (x + y)(x − y) touches *three* ops (a sub and two muls) and
  creates three new ones.
- f(conj(z)) → conj(f(z)) *reorders* two ops.

These are **DAG-to-DAG rewrites** — replacing one connected piece of the
IR graph with another — and MLIR's home for them is the
**canonicalization** hook: patterns an op contributes to the
`--canonicalize` pass. That pass, which you've been using since
Chapter 14, is really three things braided together, run by the greedy
driver (Chapter 10) until a fixed point:

1. every op's **folders**,
2. every op's **canonicalization patterns** (this chapter),
3. generic cleanups (dead-op removal — the reason canonicalize deletes
   what sccp leaves behind).

Why "canonical"? The goal isn't only optimization: it's putting IR into a
*normal form* so that later passes (and folders, and CSE) see one shape
instead of five equivalent ones. The conjugation rewrite is pure
normal-form play: `conj` ops migrate to a consistent position, so
downstream code needs one pattern, not two. (In practice — as this
codebase's author notes, fairly — canonicalize accretes minor
optimizations too and becomes "a heavyweight and powerful pass"; the
boundary between "canonical form" and "optimization" is community
convention.)

Mechanically you already know everything involved: canonicalization
patterns *are* Chapter 10's `RewritePattern`s. The only new machinery is
how an op volunteers them.

## 2. Attaching patterns to an op

In [`PolyOps.td`](../../lib/Dialect/Poly/PolyOps.td) (Chapter 12), the
binop base class and `eval` declare:

***lib/Dialect/Poly/PolyOps.td*** (excerpts)
```tablegen
class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  ...
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

```tablegen
def Poly_EvalOp : Op<Poly_Dialect, "eval", [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
  ...
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

`hasCanonicalizer = 1` generates one static-method declaration on each op
class — you can see it in `build/lib/Dialect/Poly/PolyOps.h.inc`, once
per op that opted in (the official docs for canonicalizing this way are
[here](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-rewritepatterns)):

```cpp
static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
```

The `--canonicalize` pass calls this on every registered op type and
collects the patterns. The implementations at the bottom of
[`PolyOps.cpp`](../../lib/Dialect/Poly/PolyOps.cpp) are one line each:

***lib/Dialect/Poly/PolyOps.cpp*** (excerpt)
```cpp
void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {
  results.add<DifferenceOfSquares>(context);
}

void EvalOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.add<LiftConjThroughEval>(context);
}
```

(`AddOp`'s and `MulOp`'s are registered but empty — hooks waiting for
future patterns.) A design question worth pausing on: each pattern is
attached to its *root* op — the op at the "top" of the matched DAG, the
one being replaced. Difference-of-squares roots at the `sub`; the
conjugation rewrite roots at the `eval`.

## 3. The pattern in C++ (the original version)

`DifferenceOfSquares` was first written exactly the way Chapter 10
taught: an ordinary `OpRewritePattern<SubOp>` — rooted at the `sub`, as
section 2 chose — following the two-phase discipline, a match phase that
only inspects and bails out, then a rewrite phase that builds the
replacement. It's worth reading in full because the DRR version must be
understood *as shorthand for this*:

***part3-building-a-dialect/ch16-canonicalizers-drr/code/DifferenceOfSquares.cpp*** (excerpt)
```cpp
struct DifferenceOfSquares : public OpRewritePattern<SubOp> {
  DifferenceOfSquares(mlir::MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
      return failure();
    }

    auto rhsMul = rhs.getDefiningOp<MulOp>();
    auto lhsMul = lhs.getDefiningOp<MulOp>();
    if (!rhsMul || !lhsMul) {
      return failure();
    }

    bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
    bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();
    if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
      return failure();
    }

    auto x = lhsMul.getLhs();
    auto y = rhsMul.getLhs();

    AddOp newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
    SubOp newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
    MulOp newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);

    // The original version wrote `replaceOp(op, {newMul})`; the braced form
    // is ambiguous (ValueRange vs. Operation*) in current MLIR.
    rewriter.replaceOp(op, newMul);
    return success();
  }
};
```

All Chapter 10 vocabulary, and one short linear path — read it in
execution order. The match phase is three bailouts, each returning
`failure()` before anything is mutated. The first — the **`hasOneUse`
guard** — is the one genuinely new, *semantic* check: if `x²` has
*another* consumer, rewriting the sub doesn't retire the mul — it would
still be computed, and we'd have *added* ops for nothing. "Is this
rewrite profitable?" often reduces to use-count checks. The second climbs
the SSA graph with `getDefiningOp<MulOp>`: both operands must be
products. The third, the `getLhs() == getRhs()` comparisons, requires
each product to be a *square* — both factors the same value. Only once
all three pass does the rewrite phase run: bind `x` and `y`, `create`
the new add, sub, and mul, and `replaceOp` the root.

The other semantic detail is **what's *not* here: erasing the muls.**
The pattern replaces only the root `sub`; the old muls become dead and
the canonicalizer's built-in dead-code cleanup (section 1, item 3)
collects them. Rewrite the root, let the driver sweep.

> **Want to run this exact pattern?** It is kept buildable in this
> chapter's [`code/`](code/) directory, wrapped as a standalone pass
> (flag: `--difference-of-squares`) so the Poly dialect in `lib/`
> needn't be edited. The normal build already produced its driver:
>
> ```bash
> build/part3-building-a-dialect/ch16-canonicalizers-drr/code/tutorial-opt-ch16 \
>   --difference-of-squares test/part3/poly_canonicalize.mlir
> ```
>
> (To rebuild only it: `cmake --build build --target tutorial-opt-ch16`.)
> On the section 5 test file it produces the same rewrite as the DRR
> version, and honors the same `hasOneUse` refusal. One line had to
> change to compile against current MLIR — see the directory's
> [README](code/README.md).

## 4. The same pattern in DRR (the repo's version)

Look closely at section 3: about 30 lines, of which perhaps 6 carry the
*idea* — "match sub(mul(x,x), mul(y,y)), emit mul(add(x,y), sub(x,y))".
The rest is casts, null checks, and plumbing.
**[DRR](https://mlir.llvm.org/docs/DeclarativeRewrites/)** (Declarative
Rewrite Rules) is a tablegen language for exactly this shape of pattern:
say the source DAG, say the target DAG, let a generator write the
plumbing. The repo's current implementation:

***lib/Dialect/Poly/PolyPatterns.td*** (excerpt)
```tablegen
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

Reading the language:

- **The source pattern** (first argument) is the DAG to match, written as
  nested op applications: a `Poly_SubOp` whose operands are two
  `Poly_MulOp`s. Names with `$` are *bound variables*: `$x` appearing
  twice in `(Poly_MulOp:$lhs $x, $x)` means "both operands of this mul
  must be the *same* value" — the C++ version's `getLhs() == getRhs()`
  check, expressed by repetition. `OpName:$name` binds the matched *op's
  result* itself (`$lhs`, `$rhs`) for use in constraints.
- **The result pattern(s)** (second argument) build the replacement.
  Because this replacement is a *list* of new ops (add, sub, mul), the
  `Pattern` class is used; the last entry (`$res`) replaces the root op's
  result. A single-op replacement can use the simpler `Pat<source,
  result>` form — see the next pattern.
- **Constraints** (third argument) are extra predicates. There is no
  built-in DRR way to say "has one use", so `HasOneUse` *injects C++*
  via `CPred`: `$_self` expands to the constrained value, and the string
  is pasted into the generated match code. DRR's escape hatch is
  literally "write the C++ inline" — a candid note from this codebase's
  author: fluency in *both* styles is required, since real patterns keep
  one foot in C++.

The conjugation rewrite is the simple form — one op in, one op out,
`Pat`:

***lib/Dialect/Poly/PolyPatterns.td*** (excerpt)
```tablegen
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
```

Match "eval of f at conj(z)", emit "conj of eval of f at z" — the
math identity f(z̄) = f(z)̄, as two lines of tree surgery. (`ConjOp` comes
from including `mlir/Dialect/Complex/IR/ComplexOps.td` — DRR patterns mix
dialects freely. `$fastmath` binds `conj`'s fastmath *attribute* and
carries it to the new op: attributes bind just like operands. The
`fastmath` attribute is a relatively recent upstream addition to
`ConjOp` — an older two-argument version of this pattern, without
`$fastmath`, no longer compiles.)

### From DRR to C++

The build runs the `-gen-rewriters` tablegen backend over
`PolyPatterns.td`, producing `PolyCanonicalize.cpp.inc`, which
`PolyOps.cpp` includes at the top (with a warning-comment that the
include must come *after* `PatternMatch.h`, whose declarations the
generated code uses). The CMake wiring in
[`lib/Dialect/Poly/CMakeLists.txt`](../../lib/Dialect/Poly/CMakeLists.txt)
is one more `mlir_tablegen` stanza, Chapter 11's mechanism verbatim:

***lib/Dialect/Poly/CMakeLists.txt*** (excerpt)
```cmake
set(LLVM_TARGET_DEFINITIONS PolyPatterns.td)
mlir_tablegen(PolyCanonicalize.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRPolyCanonicalizationIncGen)
```

Run the generator yourself and skim (Chapter 11's habit):

```bash
mlir-tblgen -gen-rewriters lib/Dialect/Poly/PolyPatterns.td \
  -I $WHEEL_ROOT/include -I lib/Dialect/Poly | wc -l
```

```
197
```

Twenty-six lines of tablegen in, 197 lines of C++ out — and they are the
build tree's bytes. Diffed against `build/lib/Dialect/Poly/`, the only
difference is the `Generated from:` path comment (CMake invokes tblgen
from inside the source directory); run it from there and the match is
exact:

```bash
(cd lib/Dialect/Poly && mlir-tblgen -gen-rewriters PolyPatterns.td \
  -I $WHEEL_ROOT/include -I .) \
  | diff - build/lib/Dialect/Poly/PolyCanonicalize.cpp.inc && echo IDENTICAL
```

```
IDENTICAL
```

The output is one `RewritePattern` subclass per `def`. The header alone
repays the trip — DRR computed a pattern *benefit* of 3 from the source
DAG's depth (section 3's hand-written version guessed `benefit=1`), and
declared up front which ops the rewrite can create:

***build/lib/Dialect/Poly/PolyCanonicalize.cpp.inc*** (excerpts)
```cpp
struct DifferenceOfSquares : public ::mlir::RewritePattern {
  DifferenceOfSquares(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("poly.sub", 3, context, {"poly.add", "poly.mul", "poly.sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
```

Inside `matchAndRewrite` you can watch each piece of the tablegen become
the C++ you would have written: the repeated `$x` becomes an equality
check, and the `CPred` string is pasted in verbatim (with `$_self`
expanded):

```cpp
    if (!(*x.begin() == *x0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'x' and 'x0' must be equal";
      });
    }
    ...
    if (!(((*lhs.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities '' failed to satisfy constraint: 'has one use'";
      });
    }
```

One upgrade over section 3's `return failure()`: every generated bailout
is a `notifyMatchFailure` with a human-readable reason, which
`--debug`-style pattern logging can surface. The file ends with a
convenience `populateWithGenerated(RewritePatternSet&)` that registers
everything at once (this repo registers by name instead, in section 2's
hooks, since different patterns belong to different ops). As with every
tablegen backend since Chapter 11: it's a see-through generator, and when
a DRR pattern misbehaves, you debug by reading this file.

## 5. Step: watch the canonicalizations

[`test/part3/poly_canonicalize.mlir`](../../test/part3/poly_canonicalize.mlir)
has four functions; take them in turn with

```bash
build/tools/tutorial-opt --canonicalize test/part3/poly_canonicalize.mlir
```

(all output below is real).

**Difference of squares, firing:**

***test/part3/poly_canonicalize.mlir*** (excerpt)
```mlir
func.func @test_difference_of_squares(
    %0: !poly.poly<3>, %1: !poly.poly<3>) -> !poly.poly<3> {
  %2 = poly.mul %0, %0 : !poly.poly<3>
  %3 = poly.mul %1, %1 : !poly.poly<3>
  %4 = poly.sub %2, %3 : !poly.poly<3>
  %5 = poly.add %4, %4 : !poly.poly<3>
  return %5 : !poly.poly<3>
}
```

becomes

```mlir
func.func @test_difference_of_squares(%arg0: !poly.poly<3>, %arg1: !poly.poly<3>) -> !poly.poly<3> {
  %0 = poly.add %arg0, %arg1 : !poly.poly<3>
  %1 = poly.sub %arg0, %arg1 : !poly.poly<3>
  %2 = poly.mul %0, %1 : !poly.poly<3>
  %3 = poly.add %2, %2 : !poly.poly<3>
  return %3 : !poly.poly<3>
}
```

Two muls in, one mul out — and notice the original muls are simply
*gone*: the dead-code sweep from section 3's last paragraph, observed.

**Difference of squares, declining.** The third test function,
`@test_difference_of_squares_other_uses`, adds one line —
`%5 = poly.add %4, %2` — giving `x²` a second use. Run it: the IR comes
back *unchanged*. The `HasOneUse` constraint working as designed; a
rewrite that fires only when profitable.

**Conjugate through eval:**

***test/part3/poly_canonicalize.mlir*** (excerpt)
```mlir
%z_bar = complex.conj %z : complex<f64>
%evaled = poly.eval %f, %z_bar : (!poly.poly<3>, complex<f64>) -> complex<f64>
```

becomes

```mlir
%0 = poly.eval %arg0, %arg1 : (!poly.poly<3>, complex<f64>) -> complex<f64>
%1 = complex.conj %0 : complex<f64>
```

The conjugation lifted through the evaluation — and this is why
Chapter 15's verifier admits complex points at all.

**And the first function** is Chapter 14's fold demo (constants all the
way down to a `poly.constant dense<[2, 4, 6]>`) — a reminder that
`--canonicalize` is folders *and* patterns *and* cleanup in one loop.

Now the comparison this chapter was built for — run section 3's
preserved C++ pattern on the *same* file:

```bash
build/part3-building-a-dialect/ch16-canonicalizers-drr/code/tutorial-opt-ch16 \
  --difference-of-squares test/part3/poly_canonicalize.mlir
```

`@test_difference_of_squares` comes out identical to the DRR version's
output above, and the `other_uses` function is likewise refused. Two
differences, both instructive: the conj-through-eval function is
*untouched* (that pattern belongs to `EvalOp`'s canonicalizer, which this
standalone pass deliberately doesn't include), and `@test_simple` is
*still* fully folded — the greedy driver runs folders even when no
pattern matches, Chapter 10's "AndFold" behavior.

Finally, the suite entry (the `RUN` line is
`tutorial-opt --canonicalize %s | FileCheck %s`):

```bash
source .venv/bin/activate && lit build/test --filter=poly_canonicalize -v
```

```
-- Testing: 1 of 17 tests, 1 workers --
PASS: MLIRUnifiedTutorial :: part3/poly_canonicalize.mlir (1 of 1)
```

## 6. Choosing your tool, final table

Three chapters of IR-simplification machinery, one decision rule:

| Mechanism | Scope | Creates ops? | Runs |
|---|---|---|---|
| **Folder** (Ch. 14) | one op | never (attribute/value out) | constantly: greedy driver, canonicalize, sccp, after every `create` in some drivers |
| **Canonicalization pattern** (here) | a DAG | freely | inside `--canonicalize` |
| **Standalone pass** (Ch. 10) | whole module/function | freely | when *you* schedule it |

Prefer the smallest box that fits: folders if it's one op and constants;
canonicalization if it's a local identity that should *always* hold
(cheap, universally beneficial, direction-agreed); a dedicated pass if
it's a strategy — profitable only sometimes, needing analysis or
configuration (Chapter 10's full-unroll is a pass, not a
canonicalization, because unrolling everything is not always an
improvement).

There is also a *third* authoring language for patterns beyond C++ and
DRR: [PDLL](https://mlir.llvm.org/docs/PDLL/), which is the very next
chapter (via the repo's `--mul-to-add-pdll` pass and
[`lib/Transform/Arith/MulToAdd.pdll`](../../lib/Transform/Arith/MulToAdd.pdll)).

## Where to go next

[Chapter 17: PDLL](../ch17-pdll/README.md) closes out pattern authoring
with that third language. After it, the `poly` dialect is
feature-complete as a *high-level* IR: syntax, semantics, verification,
folding, canonicalization. What it cannot yet do is *run*. The remaining
arc is downhill — Part 1's lowering staircase, but for our own dialect:
convert `poly` ops into `arith` and `tensor` ops
([Chapter 18: Dialect Conversion](../../part4-lowering/ch18-dialect-conversion/README.md)),
then to LLVM and an actual executable
([Chapter 19](../../part4-lowering/ch19-to-llvm-native-binary/README.md)).
The `lib/Conversion/PolyToStandard/` directory has been waiting since
the project-layout tour.

**Exercises**

1. Reproduce section 5, then break the `HasOneUse` guard: delete the
   constraint list from `DifferenceOfSquares` in a *copy* of
   `PolyPatterns.td`, regenerate with `-gen-rewriters`, and read the
   generated `matchAndRewrite` diff. (No rebuild needed — this is a
   tablegen-only exercise. Verified: exactly two `if`-blocks — the
   pasted `hasOneUse()` checks — disappear; the match logic is otherwise
   untouched.)
2. The conjugation identity f(z̄) = f(z)̄ requires *real* coefficients —
   ours are integers, so it holds. Write the one-line DRR `Pat` for the
   related identity eval(f, conj(conj(z))) → eval(f, z)... then explain
   why you don't need to: which existing upstream canonicalization
   already handles conj∘conj? (Check: run two nested `complex.conj`
   through `--canonicalize` — verified, the pair vanishes entirely.)
3. Write x² − y² with `x = y` (both operands the same value, squared by
   two separate muls) and canonicalize. The pattern fires and leaves
   `poly.sub %x, %x` in the output (verified) — semantically zero, but
   nothing simplifies it, because Chapter 14's `SubOp::fold` only handles
   *constant* operands. Sketch the fold (or DRR pattern) that would
   finish the job: `x − x →` what, exactly? (Careful: the answer is a
   *constant zero polynomial* — which existing op materializes it, and
   what attribute does its folder need to return?)
4. In Chapter 10, `PowerOfTwoExpand` and `PeelFromMul` were *pass*
   patterns, not canonicalizations. Given section 6's table, argue both
   sides: what would go wrong (or right) if `mul-to-add` rewrites were
   attached to `arith.muli`'s canonicalizer?
5. DRR can't express "has one use" natively, but it *can* express
   attribute equality and type constraints. Skim the
   [DRR docs](https://mlir.llvm.org/docs/DeclarativeRewrites/) and find
   one feature not used in `PolyPatterns.td` (e.g. `NativeCodeCall`) —
   sketch a use for it in `poly`.
