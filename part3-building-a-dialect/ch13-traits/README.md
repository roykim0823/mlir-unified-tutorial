# Chapter 13: Using Traits

> Reworked from the mlir-tutorial fork's chapter 6 for the unified repo.

[Chapter 12](../ch12-defining-poly/README.md) left the `poly` dialect able
to parse and print — and do absolutely nothing else. This chapter makes
the first payments on that: by declaring **traits** on our ops, a fleet
of upstream optimization passes — CSE, loop-invariant code motion,
control-flow sinking — starts working on `poly` code *without us writing
a line of pass logic*. This is also where the bracketed lists we skipped
in Chapter 12's `PolyOps.td` get explained, and where we meet this repo's
one *custom* trait, `Has32BitArguments`.

## What you will learn

- What traits are, and how they differ from interfaces.
- What `Pure` actually promises (two separate things!), and which
  upstream passes it unlocks — demonstrated live on `poly` ops.
- What `ElementwiseMappable` + container type constraints buy, and what
  they cost in the generated API.
- What `SameOperandsAndResultType` verifies and the syntax it enables.
- How to define a custom trait with a `verifyTrait` hook
  ([`lib/Dialect/Poly/PolyTraits.h`](../../lib/Dialect/Poly/PolyTraits.h)).
- How traits appear in generated C++, and where their verification lives.
- That *missing* traits have visible consequences too (why CSE won't
  touch `poly.eval`).
- How to audit the rest of the upstream pass list: most entries need
  nothing from `poly`, one crashes the *tool* (a registration lesson),
  and one waits for Chapter 14.

## Prerequisites

- [Chapter 12](../ch12-defining-poly/README.md) (the `poly` dialect
  shell) and, for how `tutorial-opt` came to be, Part 2's
  [Chapter 10](../../part2-writing-passes/ch10-first-pass-cpp/README.md).
- A configured build in `build/` per the top-level
  [README](../../README.md). Run every command in this chapter from the
  repo root with the virtualenv active
  (`source .venv/bin/activate`); the driver binary is
  `build/tools/tutorial-opt`.

---

## 1. Concepts: traits and interfaces

Generic passes have a problem: `--cse` (common subexpression
elimination) was written years before `poly.mul` existed. For it to
safely merge two identical `poly.mul` ops, it must know that `poly.mul`
has no side effects — computing it once instead of twice must be
unobservable. But a generic pass can't know that about an op it's never
heard of, and the only safe default is the conservative one: *assume the
worst and touch nothing*. That's why, out of the box, a freshly minted
dialect is invisible to the entire upstream optimization arsenal.

MLIR's fix is a declarative contract system with two closely related
mechanisms:

- An **[interface](https://mlir.llvm.org/docs/Interfaces/)** is a set of
  function signatures an op implements, giving passes a way to query or
  manipulate it without knowing its concrete type (think of
  `MemoryEffectOpInterface`, which lets any pass ask "what do you
  touch?").
- A **[trait](https://mlir.llvm.org/docs/Traits/)** is — pithily — *an
  interface with no methods*: a marker you "slap on" an operation
  asserting some property — "I have no memory effects", "all my operands
  and results have one type". Many traits also carry a *verification*
  routine, so the invariant they assert is checked on every parse and
  after every pass.

You apply traits primarily *to enable passes to do things with your
ops* — and secondarily to get free verification. Both dimensions show up
below.

## 2. The traits on `poly`'s ops

Mechanically, traits occupy the third template parameter of `Op` — a
bracketed list of trait names (it defaults to empty, which is why
Chapter 12 could ignore it). Each name in the list is one contract in
section 1's sense: a property the op asserts to every pass, often with a
verification routine attached. Here is Chapter 12's binop base again,
this time reading the bracket:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

(The `hasFolder`/`hasCanonicalizer` lines are *not* traits — this repo's
code already carries the machinery of
[Chapter 14](../ch14-folders/README.md) and
[Chapter 16](../ch16-canonicalizers-drr/README.md); ignore them for
now.) Taking the three traits in turn:

### `Pure` — the workhorse

`Pure` is actually a bundle of two independent promises (both defined in
upstream's
[SideEffectInterfaces.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td),
the file to read for the full effect vocabulary):

- **`NoMemoryEffect`** — executing this op neither reads nor writes any
  observable state. This is what lets CSE deduplicate it and dead-code
  elimination delete it when unused.
- **`AlwaysSpeculatable`** — this op may be executed *earlier* than
  written, or even when it wouldn't have executed at all (e.g. hoisted
  out of a conditional or a zero-iteration loop), without changing
  program behavior. Divisions are the classic counterexample:
  speculating a division can introduce a divide-by-zero that the
  original program guarded against.

The distinction is real and bites: the original tutorial's author first
applied only `AlwaysSpeculatable` and was baffled that
`--loop-invariant-code-motion` did nothing — reading
[the pass source](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp)
revealed LICM requires *both* `isSpeculatable` *and*
`isMemoryEffectFree` (in today's source the gate is literally a call to
a helper named `isPure`). Hoisting an op out of a loop moves it to
different *and* possibly more/fewer executions; each half of `Pure`
covers one of those hazards.

### `ElementwiseMappable` — scalars to containers

Polynomial programs will want *vectors* of polynomials.
`ElementwiseMappable` declares that the op applied to containers means
"map me over the elements", letting one op definition serve
`!poly.poly<10>`, `tensor<2x!poly.poly<10>>`, and vectors alike. It
works together with the argument constraint:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
def PolyOrContainer : TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;
```

which relaxes "a polynomial" to "a polynomial, or a tensor/vector of
them". (Upstream once called this helper `TypeOrContainer`; it was
renamed to `TypeOrValueSemanticsContainer` with the same meaning.) From
[`test/part3/poly_syntax.mlir`](../../test/part3/poly_syntax.mlir)
(Chapter 12's round-trip test):

***test/part3/poly_syntax.mlir*** (excerpt)
```mlir
%7 = tensor.from_elements %arg0, %arg1 : tensor<2x!poly.poly<10>>
%8 = poly.add %7, %7 : tensor<2x!poly.poly<10>>
```

There is a cost, visible in the generated code: with the wider
constraint, the generated accessors can no longer return
`::mlir::TypedValue<PolynomialType>` — `getLhs()` returns a
`::mlir::TypedValue<::mlir::Type>` (morally a plain `Value`, since
*every* value's type satisfies `Type`), and C++ code that needs the poly
type must `dyn_cast` for itself. Declarative generality, paid for in
static type information. (Section 5 shows how to regenerate the C++ and
see this yourself.)

### `SameOperandsAndResultType` — one type to rule them

Verifies that both operands and the result all have exactly the same
type — for `poly.add`, that you never add a `!poly.poly<10>` to a
`!poly.poly<11>`. It also has two pleasant side effects:

- With all types provably equal, the assembly format only needs to
  spell one of them — this is why the binop syntax is the terse
  `poly.add %a, %b : !poly.poly<10>` rather than the verbose generic
  `(type, type) -> type`.
- The trait implies **type inference** (you'll see
  `InferTypeOpInterface::Trait` in the generated code below): given
  operand types, MLIR can construct the op without being told the
  result type.

> **Note:** an earlier version of these ops used the looser
> `SameOperandsAndResultElementType` (same *element* type only —
> containers could mix with scalars), which admitted a mixed
> scalar-tensor add like
> `poly.add %tensor, %scalar : (tensor<2x!poly.poly<10>>, !poly.poly<10>) -> ...`.
> With `SameOperandsAndResultType` that is now **rejected**: feeding it
> in generic form produces
> `'poly.add' op requires the same type for all operands and results`
> (verified against this repo's build). Containers still work; they
> just must match on both sides.

`poly.eval` uses the parameterized cousin visible in `PolyOps.td` —
`AllTypesMatch<["point", "output"]>` — same idea, scoped to a named
subset of operands/results. Its *other* trait is not from upstream at
all, which brings us to:

## 3. A custom trait: `Has32BitArguments`

Look at `poly.eval`'s declaration:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
// Inject verification that all integer-like arguments are 32-bits
def Has32BitArguments : NativeOpTrait<"Has32BitArguments"> {
  let cppNamespace = "::mlir::tutorial::poly";
}

def Poly_EvalOp : Op<Poly_Dialect, "eval", [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$polynomial, IntOrComplex:$point);
  let results = (outs IntOrComplex:$output);
  ...
}
```

`NativeOpTrait` is tablegen's escape hatch: "there is a C++ class by
this name in this namespace; mix it into the op". The class itself lives
in [`lib/Dialect/Poly/PolyTraits.h`](../../lib/Dialect/Poly/PolyTraits.h),
in its entirety:

***lib/Dialect/Poly/PolyTraits.h*** (excerpt)
```cpp
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
```

Three things to notice:

- **The shape is CRTP again** — `TraitBase<ConcreteType, ...>`, the
  same mixin pattern as Part 2's pass classes. A trait is just a class
  template threaded into the op's inheritance chain (section 5 makes
  this literal).
- **`verifyTrait` is the hook.** A trait may define it, and op
  verification calls every trait's `verifyTrait` automatically — this
  is the "free verification" half of the trait contract. Upstream's
  `SameOperandsAndResultType` rejects mismatched types through exactly
  this mechanism.
- **The wiring is one `#include`.**
  [`PolyOps.h`](../../lib/Dialect/Poly/PolyOps.h) includes
  `PolyTraits.h` before the generated op declarations, so when tablegen
  emits `::mlir::tutorial::poly::Has32BitArguments` into `EvalOp`'s
  base class list, the definition is in scope.

Watch it fire — evaluate a polynomial at an `i64` point:

```mlir
func.func @eval_64bit(%p: !poly.poly<10>, %x: i64) -> i64 {
  %0 = poly.eval %p, %x : (!poly.poly<10>, i64) -> i64
  return %0 : i64
}
```

```bash
build/tools/tutorial-opt eval_64bit.mlir
```

```
error: 'poly.eval' op requires each numeric operand to be a 32-bit integer
```

No pass was requested — verification alone runs on parse. The test
[`test/part3/poly_verifier.mlir`](../../test/part3/poly_verifier.mlir)
pins this error down with FileCheck. Custom traits like this are the
lightweight end of a spectrum whose heavyweight end — `hasVerifier` and
custom diagnostics in general — is
[Chapter 15](../ch15-verifiers/README.md)'s subject.

## 4. Step: watch the upstream passes work

Time to collect. Three test files, three upstream passes, zero lines of
`poly`-specific pass code. (All passes here are registered by
`registerAllPasses()` in
[`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp).)

### Common subexpression elimination

[`test/part2/cse.mlir`](../../test/part2/cse.mlir) computes the same
product twice:

***test/part2/cse.mlir*** (excerpt)
```mlir
%2 = poly.mul %p0, %p0 : !poly.poly<10>
%3 = poly.mul %p0, %p0 : !poly.poly<10>
%4 = poly.add %2, %3 : !poly.poly<10>
```

```bash
build/tools/tutorial-opt -cse test/part2/cse.mlir
```

```mlir
%0 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
%1 = poly.mul %0, %0 : !poly.poly<10>
%2 = poly.add %1, %1 : !poly.poly<10>
```

One `poly.mul` remains, feeding both sides of the add. Only
`NoMemoryEffect` makes this legal — if `poly.mul` wrote to a log file,
deduplicating it would halve the log.

### Loop-invariant code motion

The upstream
[general transformation passes list](https://mlir.llvm.org/docs/Passes/#general-transformation-passes)
includes
[loop invariant code motion](https://mlir.llvm.org/docs/Passes/#-loop-invariant-code-motion),
which checks loop bodies for operations that don't need to be in the
loop and moves them out.
[`test/part2/code_motion.mlir`](../../test/part2/code_motion.mlir)
multiplies two loop-constant polynomials *inside* a loop:

***test/part2/code_motion.mlir*** (excerpt)
```mlir
%ret_val = affine.for %i = 0 to 100 iter_args(%sum_iter = %p0) -> !poly.poly<10> {
  %2 = poly.mul %p0, %p1 : !poly.poly<10>
  %sum_next = poly.add %sum_iter, %2 : !poly.poly<10>
  affine.yield %sum_next : !poly.poly<10>
}
```

```bash
build/tools/tutorial-opt --loop-invariant-code-motion test/part2/code_motion.mlir
```

```mlir
%0 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
%1 = poly.from_tensor %cst_0 : tensor<3xi32> -> !poly.poly<10>
%2 = poly.mul %0, %1 : !poly.poly<10>
%3 = affine.for %arg0 = 0 to 100 iter_args(%arg1 = %0) -> (!poly.poly<10>) {
  %4 = poly.add %arg1, %2 : !poly.poly<10>
  ...
```

The `poly.mul` is hoisted above the loop — computed 1× instead of 100×.
This is the pass that needs the *whole* of `Pure` (section 2's war
story). Note the `poly.add` correctly stays inside: it consumes
`%sum_iter`, which changes every iteration — LICM checks invariance of
operands, traits or no traits.

### Control-flow sinking

The dual of hoisting. Where LICM rescues work from a region that runs
*many* times,
[control-flow sink](https://mlir.llvm.org/docs/Passes/#-control-flow-sink)
pushes work into a region that may run *zero* times: an operation whose
only uses sit inside one conditionally-executed region is moved into
that region, so it executes only when its result is actually consumed.
[`test/part2/control_flow_sink.mlir`](../../test/part2/control_flow_sink.mlir)
builds two polynomials *before* a branch, but each is used in only one
arm of an `scf.if` (Part 1's structured conditional):

***test/part2/control_flow_sink.mlir*** (excerpt)
```mlir
%p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
%p1 = poly.from_tensor %1 : tensor<3xi32> -> !poly.poly<10>
%4 = scf.if %arg0 -> (!poly.poly<10>) {
  %2 = poly.mul %p0, %p0 : !poly.poly<10>
  scf.yield %2 : !poly.poly<10>
} else {
  %3 = poly.mul %p1, %p1 : !poly.poly<10>
  scf.yield %3 : !poly.poly<10>
}
```

```bash
build/tools/tutorial-opt -control-flow-sink test/part2/control_flow_sink.mlir
```

```mlir
func.func @test_simple_sink(%arg0: i1) -> !poly.poly<10> {
  %0 = scf.if %arg0 -> (!poly.poly<10>) {
    %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %1 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
    %2 = poly.mul %1, %1 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  } else {
    %cst = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
    %1 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
    %2 = poly.mul %1, %1 : !poly.poly<10>
    scf.yield %2 : !poly.poly<10>
  }
  return %0 : !poly.poly<10>
}
```

Comparing before and after:

- **Nothing is left in front of the branch.** Each `poly.from_tensor`
  moved into the arm that uses it, so whichever way `%arg0` goes at
  runtime, the untaken arm's polynomial is never materialized — the
  work runs "only if used" instead of "always".
- **The `arith.constant`s went along for the ride.** Sinking chases
  whole def chains: once a `from_tensor` sinks, its constant's only use
  sits inside that arm as well, making the constant sinkable too. (It
  qualifies for the same reason — `arith.constant` carries `Pure`
  upstream.)
- **The licensing trait is `Pure` again, seen from the other side.**
  LICM moved an op to *more* executions — that's why it also demands
  the speculatability half (section 2's war story). Sinking moves an op
  to *fewer*, and the half doing the work is `NoMemoryEffect`: skipping
  an execution is only invisible when the op has no effects to skip. If
  `poly.from_tensor` appended to a log file, sinking it would silence
  the log every time the other arm ran.

One thing sinking will *not* do is duplicate work. Rewrite the test so
*both* arms use the same polynomial — `%p0` squared in one arm, added
to itself in the other:

```mlir
%p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
%4 = scf.if %arg0 -> (!poly.poly<10>) {
  %2 = poly.mul %p0, %p0 : !poly.poly<10>
  scf.yield %2 : !poly.poly<10>
} else {
  %3 = poly.add %p0, %p0 : !poly.poly<10>
  scf.yield %3 : !poly.poly<10>
}
```

and the pass leaves the IR exactly as it found it (verified). The pass
only *moves* ops, never clones them, so an op can sink only when
**all** of its uses land in a single region — `%p0`'s uses span both
arms, so it stays where it is, traits or no traits. Same lesson as LICM
declining to hoist the `poly.add`: traits grant *permission*; the
pass's own profitability and correctness analysis still decides.

Run all three as tests (from the repo root, virtualenv active):

```bash
lit build/test --filter='cse|code_motion|control_flow_sink' -v
```

```
PASS: MLIRUnifiedTutorial :: part2/cse.mlir (1 of 3)
PASS: MLIRUnifiedTutorial :: part2/control_flow_sink.mlir (2 of 3)
PASS: MLIRUnifiedTutorial :: part2/code_motion.mlir (3 of 3)
```

### The dog that didn't bark

Traits you *don't* declare also teach. `poly.eval`'s trait list has no
`Pure`. Duplicate an eval and run CSE:

```mlir
%0 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
%1 = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
%2 = arith.addi %0, %1 : i32
```

Both `poly.eval` ops survive `-cse` (verified). Nothing is wrong with
`eval` — it's a pure function mathematically — but nobody *told* MLIR
that, so the conservative default rules. When your shiny new op
mysteriously escapes optimization, the first place to look is its trait
list.

### The rest of the pass list

The three demos weren't cherry-picked — they're what remains after
reading the
[general transformation passes list](https://mlir.llvm.org/docs/Passes/#general-transformation-passes)
end to end and asking, for each entry, *what does this pass need from
`poly`?* The others sort into three buckets, and the sorting itself is
a skill worth having whenever you bring up a new dialect:

- **Nothing — the pass works on structures `poly` doesn't have.**
  [`-mem2reg`](https://mlir.llvm.org/docs/Passes/#-mem2reg) (promotes
  memory slots into SSA values) and
  [`-sroa`](https://mlir.llvm.org/docs/Passes/#-sroa) (scalar
  replacement of aggregates) rewrite memory operations, and `poly`
  programs contain none;
  [`-remove-dead-values`](https://mlir.llvm.org/docs/Passes/#-remove-dead-values)
  trims unused function arguments and results;
  [`-symbol-dce`](https://mlir.llvm.org/docs/Passes/#-symbol-dce)
  deletes private functions nobody references. Stack all four onto a
  `poly` file and nothing changes (verified):

  ```bash
  build/tools/tutorial-opt -mem2reg -sroa -remove-dead-values -symbol-dce test/part2/cse.mlir
  ```

  They aren't *failing* on `poly` — they have nothing to do here, and
  no trait would change that.

- **Something from the *tool*, not the dialect.**
  [`-inline`](https://mlir.llvm.org/docs/Passes/#-inline) also has
  nothing `poly`-specific to do — it rearranges function calls, and
  `poly` ops would just ride along — but try it and this repo's binary
  *crashes*:

  ```bash
  build/tools/tutorial-opt -inline test/part2/cse.mlir
  ```

  ```
  LLVM ERROR: checking for an interface (`mlir::DialectInlinerInterface`)
  that was promised by dialect 'func' but never implemented. This is
  generally an indication that the dialect extension implementing the
  interface was never registered.
  ```

  Decode it with section 1's vocabulary: inlining is powered by an
  *interface* — attached to a whole dialect rather than a single op —
  that answers questions like "may this body be inlined into that call
  site?". The `func` dialect *promises* that interface but ships the
  implementation in a separate extension, which the wheel's stock
  `mlir-opt` registers and `tutorial-opt` never does — the same
  two-function example inlines fine under
  `$(python -m mlir_wheel --root-dir)/bin/mlir-opt -inline` (verified).
  A useful failure to have met: "the pass doesn't work" sometimes means
  a missing trait, and sometimes a missing *registration* — Chapter 12's
  registration lesson resurfacing one level up.

- **Folding — the next chapter's subject.**
  [`-sccp`](https://mlir.llvm.org/docs/Passes/#-sccp) (sparse
  conditional constant propagation) propagates *values*: to push
  constants through `poly.mul` it must ask the op to compute its
  result, and nothing taught so far provides that. The machinery is
  called a *folder*, and it's Chapter 14. (If you can't wait: this
  repo's ops already carry Chapter 14's folders, so
  `build/tools/tutorial-opt -sccp test/part2/sccp.mlir` already
  rewrites the constant `poly` arithmetic into `poly.constant`s today —
  verified.)

## 5. Under the hood

What does the bracket in tablegen become? Every trait is a C++
*template mixin* threaded into the op's base class — the same CRTP
dance as Part 2's passes, at industrial scale. To see it, regenerate
the op declarations with the wheel's `mlir-tblgen` (with the virtualenv
active, from the repo root):

```bash
export PATH=$(python -m mlir_wheel --root-dir)/bin:$PATH
mlir-tblgen --gen-op-decls lib/Dialect/Poly/PolyOps.td \
  -I lib/Dialect/Poly -I $(python -m mlir_wheel --root-dir)/include
```

and look at `AddOp`'s actual base class (verified, one line, wrapped
here):

```cpp
class AddOp : public ::mlir::Op<AddOp,
    ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl,
    ::mlir::OpTrait::ZeroSuccessors,
    ::mlir::OpTrait::NOperands<2>::Impl,
    ::mlir::OpTrait::OpInvariants,
    ::mlir::ConditionallySpeculatable::Trait,
    ::mlir::OpTrait::AlwaysSpeculatableImplTrait,
    ::mlir::MemoryEffectOpInterface::Trait,
    ::mlir::OpTrait::Elementwise,
    ::mlir::OpTrait::Scalarizable,
    ::mlir::OpTrait::Vectorizable,
    ::mlir::OpTrait::Tensorizable,
    ::mlir::OpTrait::SameOperandsAndResultType,
    ::mlir::InferTypeOpInterface::Trait> {
```

You can see the expansions, declaration by declaration:

- `Pure` became the
  `ConditionallySpeculatable`/`AlwaysSpeculatableImplTrait` pair plus
  `MemoryEffectOpInterface` (whose generated `getEffects` body is empty
  — "no effects" as literal code).
- `ElementwiseMappable` became
  `Elementwise` + `Scalarizable` + `Vectorizable` + `Tensorizable`.
- `SameOperandsAndResultType` dragged in `InferTypeOpInterface` as
  promised.
- Even "trivia" you never declared is trait-encoded: `ZeroRegions`,
  `OneResult`, `NOperands<2>` come from the `arguments`/`results` you
  wrote in Chapter 12.

And in the same output, `EvalOp`'s base class list contains, verbatim,
`::mlir::tutorial::poly::Has32BitArguments` — our section 3 class
template, mixed in exactly like the upstream ones. There is no
privileged machinery behind upstream traits; `verifyTrait` runs for all
of them as part of op verification, which is how
`SameOperandsAndResultType` rejects mismatched types and how
`Has32BitArguments` rejects `i64` points.

A practical note, still true: there is no single documented list of all
upstream traits. The Traits documentation's
[operation traits list](https://mlir.llvm.org/docs/Traits/#operation-traits-list)
covers many, but some (e.g. `ConstantLike`, `Involution`, `Idempotent`)
you discover only by reading `OpBase.td` and the pass sources. A few
worth knowing exist, even though `poly`'s binops don't use them:
`Commutative` (operand reordering; canonicalization uses it to move
constants rightward — the fact `PowerOfTwoExpand` relied on in
Chapter 10!), `Involution` (`f(f(x)) = x`, would auto-cancel a
hypothetical double `poly.neg`), and `Idempotent` (`f(f(x)) = f(x)`).
One member of the family you *can* already spot in this repo's
`PolyOps.td`: `poly.constant` carries `ConstantLike` — a promise whose
payoff ("this op materializes a compile-time constant") is central to
Chapter 14.

Skimming the list also turns up whole families for capabilities
`poly`'s ops don't have — regions and symbols — though you have been
*consuming* them since Part 1: `IsolatedFromAbove` (the isolation
contract that lets `func.func` anchor a parallel pass, as in Part 2),
`Terminator` plus `SingleBlock`/`SingleBlockImplicitTerminator` (the
rules that every block ends in a `return`/`scf.yield`/`affine.yield`),
`AffineScope` (also on `func.func`: the boundary inside which
`affine`'s symbol-and-dimension rules apply), and `SymbolTable` (on
`builtin.module`; what makes lookups of `@main`-style names work). None
of these belong on `poly.add`; recognizing their names in other
dialects' tablegen is the point.

## Where to go next

Traits let *existing* ops be moved, merged, and deleted — but nothing
yet *computes* with polynomials at compile time.
[`test/part2/sccp.mlir`](../../test/part2/sccp.mlir) shows the goal:
sparse conditional constant propagation replacing `poly.mul` of known
constants with a `poly.constant` of the product — possible only because
the ops know how to **fold**. How folders work, and what the
`hasFolder = 1` lines we skipped past actually generate, is
[Chapter 14: Folders and Constant Propagation](../ch14-folders/README.md).

**Exercises**

1. Reproduce the "dog that didn't bark": write the duplicate-`poly.eval`
   function from section 4, run `-cse`, and confirm both evals survive.
   Then find, in `PolyOps.td`, exactly what you would change so CSE
   could merge them.
2. Make LICM refuse: change `test/part2/code_motion.mlir`'s `poly.mul`
   to depend on `%sum_iter` (work on a copy outside the repo, or revert
   after — it's a checked-in test) and confirm the mul stays in the
   loop — trait or not, invariance is about operands.
3. Verify the stricter trait yourself: run this chapter's mixed
   scalar/tensor `poly.add` (section 2's note; use the generic
   `"poly.add"(...)` syntax from Part 1, since the pretty syntax can't
   even express it) and read the verifier error.
4. Run `-cse` and `--loop-invariant-code-motion` *together* on
   `test/part2/cse.mlir` and `test/part2/code_motion.mlir` — does the
   order of the two flags matter for these two files? Predict, then
   check.
5. Skim [mlir.llvm.org/docs/Traits](https://mlir.llvm.org/docs/Traits/)
   and pick one trait not used by `poly` (e.g. `Involution`). Sketch
   the `.td` for a `poly.neg` op that would exploit it, and what
   canonicalization it would enable for `poly.neg (poly.neg %x)`.
6. Trigger `Has32BitArguments` from section 3 yourself, then explain
   why `test/part3/poly_syntax.mlir`'s `poly.eval` at a `complex<f64>`
   point does *not* trigger it. Read `verifyTrait` in `PolyTraits.h`
   and find the exact line that lets complex points through.
