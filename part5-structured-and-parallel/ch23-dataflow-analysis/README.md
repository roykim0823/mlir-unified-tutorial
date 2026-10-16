# Chapter 23: Dataflow Analysis and Global Optimization

> Reworked from the mlir-tutorial fork's chapter 12 for the unified repo.

Every transformation in this tutorial so far — folds, canonicalizations,
lowerings, even Chapter 22's transform scripts — was *local*: look at an op
and its neighbors, rewrite, repeat. This chapter, the finale of the
`poly`/`noisy` thread that began in Part 3, changes register entirely. A
second dialect, **`noisy`**, models the central constraint of FHE (noise
growth), and the compiler's job becomes a *placement problem*: insert as few
expensive `noisy.reduce_noise` ops as possible while keeping every value's
noise under budget. Solving it takes two tools no previous chapter needed — a
**dataflow analysis** (MLIR's framework for whole-program facts) and an
**integer linear program** solved by Google's or-tools. Yes: this one pass is
why this repo's build has a `-DENABLE_SOLVER` option that downloads half of
operations research (more on that under Prerequisites).

## What you will learn

- The `noisy` dialect's noise model, worked by hand before any code.
- Implementing an *upstream* op interface (`InferIntRangeInterface`) — the
  complement of Chapter 21, where you defined your own.
- MLIR's dataflow framework: transfer functions, joins, fixed points — and
  reusing upstream's integer-range analysis for a custom purpose.
- How to encode "where should I insert ops?" as an ILP: decision variables,
  big-M conditional constraints, and a solver as a compiler analysis.
- Verified end-to-end runs where the solver beats any greedy strategy.

## Prerequisites

- [Chapter 12](../../part3-building-a-dialect/ch12-defining-poly/README.md)
  (dialect machinery — `noisy` reuses all of it) and
  [Chapter 13](../../part3-building-a-dialect/ch13-traits/README.md), which
  defined a trait as "an interface with no methods" and promised the
  with-methods half. [Chapter 21](../ch21-op-interfaces-gpu-tiles/README.md)
  delivered one direction — *defining* your own interface so your pass can
  dispatch through it. This chapter is the other direction: *implementing* an
  interface that upstream MLIR defined, so upstream's machinery can drive
  your ops.

**Build note.** This is the one chapter that needs **or-tools**, and it is
deliberately kept out of the default build so `check-all` stays
laptop-friendly. The `noisy` *dialect* is always built — the default
`build/tools/tutorial-opt` parses and prints noisy IR — but the
`--noisy-reduce-noise-optimizer` *pass* only exists when the solver is
enabled. Set up a second build directory (the first configure fetches and
builds or-tools v9.11; expect a long wait and a few gigabytes):

```sh
source .venv/bin/activate
mkdir build-solver && cd build-solver
cmake .. -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) \
         -DLLVM_EXTERNAL_LIT=$(which lit) \
         -DENABLE_SOLVER=ON
cmake --build . -j
```

All files for this chapter:

- Dialect: [`lib/Dialect/Noisy/`](../../lib/Dialect/Noisy/) —
  `NoisyDialect.td`, `NoisyTypes.td`, `NoisyOps.td` and their `.cpp` files
- The ILP analysis:
  [`lib/Analysis/ReduceNoiseAnalysis/`](../../lib/Analysis/ReduceNoiseAnalysis/)
- The pass: [`lib/Transform/Noisy/`](../../lib/Transform/Noisy/) —
  `ReduceNoiseOptimizer.cpp`
- Tests: [`test/part5/noisy_syntax.mlir`](../../test/part5/noisy_syntax.mlir)
  (runs in any build),
  [`test/part5/noisy_reduce_noise.mlir`](../../test/part5/noisy_reduce_noise.mlir)
  (gated on the solver build — see section 5)

Run every command below from the repo root (`mlir-unified-tutorial/`).

## 1. The noise model, by hand

The `noisy` dialect is a toy model of FHE's defining problem. In
lattice-based homomorphic encryption, every ciphertext carries random noise;
each homomorphic operation *grows* it; and if noise ever exceeds a threshold,
decryption returns garbage. Here, that becomes: a `!noisy.i32` holds a small
(5-bit) message plus a noise level measured in bits, with two constants (from
[`NoisyDialect.h`](../../lib/Dialect/Noisy/NoisyDialect.h)):

***lib/Dialect/Noisy/NoisyDialect.h*** (excerpt)
```cpp
constexpr int INITIAL_NOISE = 12;
constexpr int MAX_NOISE = 26;
```

and six ops ([`NoisyOps.td`](../../lib/Dialect/Noisy/NoisyOps.td)) with
simple noise arithmetic:

| op | meaning | noise of result |
|---|---|---|
| `noisy.encode` | i5 → noisy.i32 | 12 (fresh) |
| `noisy.add` / `sub` | arithmetic | max(inputs) + 1 |
| `noisy.mul` | arithmetic | sum of inputs |
| `noisy.reduce_noise` | **expensive** cleanup | 12 (reset) |
| `noisy.decode` | noisy.i32 → i5 | — fails if input noise > 26 |

Work an example by hand — repeated squaring, from the test file (the comments
in `test/part5/noisy_reduce_noise.mlir` do this arithmetic too):

```
%2 = encode %0     -> noise 12
%4 = mul %2, %3    -> 12 + 12 = 24   (fine, ≤ 26)
%5 = mul %4, %4    -> 24 + 24 = 48   (BOOM)
```

Addition is far gentler: a chain of four adds starting from fresh values goes
12 → 13 → 14 → 15 → 16 — never in danger. So multiplication is the
budget-eater (true in real FHE too), and programs need `reduce_noise` — but
each one is expensive, so we want the *minimum* number, placed *optimally*.

Why is this not a Chapter-11-style local rewrite? Two reasons. First, whether
an op's noise is dangerous depends on *everything upstream* of it — a per-op
pattern can't see that. Second, the best placement depends on everything
*downstream*: the star example below has two branches that each individually
exceed budget, where one `reduce_noise` placed *before the split* fixes both
— a fact no per-branch (greedy) strategy can discover. (A design aside worth
absorbing: you might propose encoding noise in the *type* — `!noisy.i32<24>`
— and letting Chapter 15's verifiers do the work. But then inserting one op
would change the types of everything downstream, and the IR fights you.
Analysis, not types, is the right home for this kind of fact.)

## 2. The dialect, and implementing an upstream interface

`noisy`'s tablegen is Chapter 12 material at this point — a dialect shell, a
parameterless type, ops with constraints (`AnyIntOfWidths<[1, 2, 3, 4, 5]>`
for the message type is the only new constraint trick). Confirm the plumbing
works in the *default* build — the dialect is always registered:

```sh
build/tools/tutorial-opt test/part5/noisy_syntax.mlir
```

One tablegen line is genuinely new:

***lib/Dialect/Noisy/NoisyOps.td*** (excerpt)
```tablegen
class Noisy_BinOp<string mnemonic> : Op<Noisy_Dialect, mnemonic, [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
]> {
```

`DeclareOpInterfaceMethods<InferIntRangeInterface, ...>` attaches an **op
interface** — a named set of function signatures — and asks tablegen to
declare the method on the op class for you to implement. Any pass can then
ask an unknown op, dynamically: "do you implement `InferIntRangeInterface`?
Then tell me your result's range." In Chapter 21 you *authored* the
interface and wrote the generic pass that consumes it; here upstream authored
`InferIntRangeInterface`, upstream wrote the analysis that consumes it, and
our only job is the per-op implementations. Interfaces are how upstream
analyses work with ops they've never heard of — the same open-extensibility
trick as traits, but carrying code. (Upstream's own customer is the
[`int-range-optimizations`](https://mlir.llvm.org/docs/Passes/#-int-range-optimizations)
pass, which uses the inferred ranges to replace `arith.cmpi` ops with
constants when the ranges decide the comparison.)

The implementations are the noise semantics of section 1 as code. Every
`inferResultRanges` has the same contract: it receives the already-inferred
ranges of the op's inputs (one `ConstantIntRanges` — a bounding interval —
per input) and reports its result's range through the `setResultRange`
callback. Lightly trimmed — the file also has `SubOp`, identical to `AddOp`:

***lib/Dialect/Noisy/NoisyOps.cpp*** (excerpt)
```cpp
ConstantIntRanges initialNoiseRange() {
  return ConstantIntRanges::fromUnsigned(APInt(32, 0),
                                         APInt(32, INITIAL_NOISE));
}

ConstantIntRanges unionPlusOne(ArrayRef<ConstantIntRanges> inputRanges) {
  auto lhsRange = inputRanges[0];
  auto rhsRange = inputRanges[1];
  auto joined = lhsRange.rangeUnion(rhsRange);
  return ConstantIntRanges::fromUnsigned(joined.umin(), joined.umax() + 1);
}

void EncodeOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                 SetIntRangeFn setResultRange) {
  setResultRange(getResult(), initialNoiseRange());
}

void AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), unionPlusOne(inputRanges));
}

void MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                              SetIntRangeFn setResultRange) {
  auto lhsRange = inputRanges[0];
  auto rhsRange = inputRanges[1];
  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(
                                  lhsRange.umin() + rhsRange.umin(),
                                  lhsRange.umax() + rhsRange.umax()));
}

void ReduceNoiseOp::inferResultRanges(ArrayRef<ConstantIntRanges> inputRanges,
                                      SetIntRangeFn setResultRange) {
  setResultRange(getResult(), initialNoiseRange());
}
```

Matching them to section 1's table:

- `EncodeOp` and `ReduceNoiseOp` are the same helper: `initialNoiseRange()`
  is `[0, INITIAL_NOISE]` = [0, 12] — the table's "fresh" and "reset" rows
  are literally one function.
- `AddOp` (and the trimmed `SubOp`) delegate to `unionPlusOne`: union the two
  input ranges, then add 1 to the upper bound — the table's max(inputs) + 1,
  expressed on ranges.
- `MulOp` adds the endpoints (`umin + umin`, `umax + umax`) — the table's
  "sum of inputs", again as a range.

The clever reuse: these methods were designed for upstream's
integer-*value* range analysis, but nothing says the integer being bounded
must be the op's value. Here the range `[0, N]` tracks **bits of noise** —
we've hijacked a stock analysis as a noise meter.

## 3. Dataflow analysis: whole-program facts

Given those per-op rules, something must *propagate* them through the
program. That is a classic **dataflow analysis** (Kildall's method — one part
of Frances Allen's Turing Award;
[this clang-docs article](https://clang.llvm.org/docs/DataFlowAnalysisIntro.html)
is a good standalone introduction):

- each SSA value carries a lattice element (here: its possible noise range,
  from "uninitialized" up to `[0, ∞)`);
- each op has a *transfer function* (our `inferResultRanges`) mapping input
  facts to output facts;
- where control flow merges, facts *join* (range union — which must be
  associative, commutative, idempotent so the iteration converges);
- iterate until a **fixed point** — no fact changes.

MLIR packages this as the `DataFlowSolver`
([official docs on dataflow analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/);
the current solver framework was introduced in
[this RFC](https://discourse.llvm.org/t/rfc-a-dataflow-analysis-framework/63340)).
The validation half of the pass
([`ReduceNoiseOptimizer.cpp`](../../lib/Transform/Noisy/ReduceNoiseOptimizer.cpp))
shows the whole API:

***lib/Transform/Noisy/ReduceNoiseOptimizer.cpp*** (excerpt)
```cpp
DataFlowSolver solver;
// The IntegerRangeAnalysis depends on DeadCodeAnalysis, but this
// dependence is not automatic and fails silently.
solver.load<dataflow::DeadCodeAnalysis>();
solver.load<dataflow::IntegerRangeAnalysis>();
if (failed(solver.initializeAndRun(module))) { ... }
```

then walks the noisy ops, looks up each result's inferred range in the
solver's lattice, and errors if `range.umax() > MAX_NOISE` — the "noise
exceeds the allowable maximum" diagnostic:

```cpp
const dataflow::IntegerValueRangeLattice *opRange =
    solver.lookupState<dataflow::IntegerValueRangeLattice>(op->getResult(0));
...
ConstantIntRanges range = opRange->getValue().getValue();
if (range.umax().getZExtValue() > MAX_NOISE) { ... }
```

Two field notes, both preserved as comments in the code: the
`DeadCodeAnalysis` line is **load-bearing** — the range analysis silently
computes nothing without it (an hour of anyone's life, donated by the
original author so you keep yours); and the author also had to patch
upstream LLVM to make `IntegerRangeAnalysis` reusable outside its own pass
at all — sometimes the tutorial writes the infrastructure. (That fix has
long since landed — you inherit it silently, via the wheel.)

## 4. The optimization, as an integer linear program

Now the placement problem. The insight is that "where do `reduce_noise` ops
go?" can be written as constraints over variables
([`ReduceNoiseAnalysis.cpp`](../../lib/Analysis/ReduceNoiseAnalysis/ReduceNoiseAnalysis.cpp)):

- For each noisy arithmetic op x, a **binary decision variable**
  `InsertReduceNoise_x` ∈ {0, 1} — "insert a cleanup after x?"
- For each SSA value v, a **continuous variable** `NoiseAt_v` ∈ [0, 26] —
  the upper bound implied so far. The `≤ 26` bound *is* the correctness
  constraint, baked into the variable's domain.
- **Encode results** (and block arguments — see exercise 4): `NoiseAt = 12`.
- **Mul:** result noise is *either* lhs+rhs (no cleanup) *or* 12 (cleanup) —
  a conditional, which ILPs can't say directly. The standard **big-M trick**
  encodes it as four linear inequalities (with `C = IF_THEN_AUX = 100`, any
  constant safely above all feasible noise):

  ```
  NoiseAt_RES ≥ 12·d              NoiseAt_RES ≤ 12 + C·(1−d)
  NoiseAt_RES ≥ lhs+rhs − C·d     NoiseAt_RES ≤ lhs+rhs + C·d
  ```

  When the decision d = 1, the right column pins the result near 12 and the
  big-M slack disables the lhs+rhs pair; when d = 0, vice versa. A fifth
  constraint (`DecisionDynamics_..._5` in the code) bounds the *pre-cleanup*
  noise `lhs+rhs ≤ 26` too — the cleanup op's own input must be valid.
- **Add/sub** need max(lhs, rhs)+1, encoded with an auxiliary variable
  `Z_v ≥ 1 + lhs`, `Z_v ≥ 1 + rhs`, given a *small penalty* (coefficient
  0.1) in the objective so the solver keeps it at the max rather than
  inflating it. (A carefulness worth copying: whenever you use the penalty
  trick you must argue the solver can't profitably inflate Z — here, larger
  Z only tightens downstream constraints, forcing *more* insertions, so
  minimization keeps it honest. The code says exactly this in a comment.)
- **Objective:** minimize Σ decision variables (+ the tiny Z penalties) —
  the fewest cleanups that keep every domain constraint satisfiable.

The or-tools mechanics take a page of glue: build the model with
`MPSolver::CreateSolver("SCIP")`, `MakeIntVar(0, 1, name)` /
`MakeNumVar(0, MAX_NOISE, name)`, `SetCoefficient(...)` per constraint —
then `Solve()`, and read the solution into a map. The analysis class wraps
all of it in the shape MLIR expects — construct it on an op, then query it:

***lib/Analysis/ReduceNoiseAnalysis/ReduceNoiseAnalysis.h*** (excerpt)
```cpp
class ReduceNoiseAnalysis {
 public:
  ReduceNoiseAnalysis(Operation *op);   // builds & solves the ILP

  /// Return true if a reduce_noise op should be inserted after the given
  /// operation, according to the solution to the optimization problem.
  bool shouldInsertReduceNoise(Operation *op) const {
    return solution.lookup(op);
  }

 private:
  llvm::DenseMap<Operation *, bool> solution;
};
```

Note the constructor-does-everything pattern: constructing the analysis
builds *and solves* the ILP on the spot, and `shouldInsertReduceNoise` is
afterwards a mere lookup in the stored solution map. The original author
grumbles about this pattern — analyses can't easily signal failure from a
constructor, hence the pass's `FIXME` about infeasible models (a second
standing `FIXME`: the analysis assumes no `reduce_noise` ops pre-exist in
the input). This construct-then-query shape is all MLIR's analysis
infrastructure asks for: the pass-management docs cover
[storing and caching a constructed analysis within a pass](https://mlir.llvm.org/docs/PassManagement/#analysis-management)
and
[when an analysis must be recomputed](https://mlir.llvm.org/docs/PassManagement/#preserving-analyses)
(by default: always, between passes).

And the pass itself is almost anticlimactic — solve, walk, insert:

***lib/Transform/Noisy/ReduceNoiseOptimizer.cpp*** (excerpt)
```cpp
ReduceNoiseAnalysis analysis(module);
OpBuilder b(&getContext());

module->walk([&](Operation *op) {
  if (!analysis.shouldInsertReduceNoise(op))
    return;

  b.setInsertionPointAfter(op);
  auto reduceOp = b.create<ReduceNoiseOp>(op->getLoc(), op->getResult(0));
  op->getResult(0).replaceAllUsesExcept(reduceOp.getResult(), {reduceOp});
});
```

In execution order: constructing the analysis solves the ILP (as above); the
walk then visits every op and skips any the solution didn't mark; for a
marked op, the builder's insertion point moves to just after it, a
`reduce_noise` consuming the op's result is created there, and
`replaceAllUsesExcept` — the one new IR-surgery verb — rewires every use of
that result to the cleanup op, *except* the cleanup op's own input. Then
section 3's dataflow validation runs as a built-in self-check: the solver's
plan is re-verified by an independent analysis before the pass declares
success. Belt, suspenders, and a good example to copy.

## 5. Step: watch the solver work

All outputs real, via the solver build:

```sh
build-solver/tools/tutorial-opt --noisy-reduce-noise-optimizer \
    test/part5/noisy_reduce_noise.mlir
```

**The squaring chain.** Four muls; noise would go 24, 48, 96, 192:

```mlir
%2 = noisy.mul %0, %1 : !noisy.i32
%3 = noisy.reduce_noise %2 : !noisy.i32
%4 = noisy.mul %3, %3 : !noisy.i32
%5 = noisy.reduce_noise %4 : !noisy.i32
%6 = noisy.mul %5, %5 : !noisy.i32
%7 = noisy.reduce_noise %6 : !noisy.i32
%8 = noisy.mul %7, %7 : !noisy.i32
%9 = noisy.decode %8 : !noisy.i32 -> i5
```

Three cleanups, not four: the *last* mul's inputs are fresh (12), so its
result is 24 ≤ 26 and `decode` accepts it directly. The minimum, found — a
greedy "reduce after every mul" wastes one.

**The all-adds chain** (12 → 13 → ... → 16): the solver inserts **nothing**.
Knowing when to do nothing is also optimization.

**The star exhibit — one cleanup, two branches.** A mul (noise 24) feeds
*two* separate four-op chains, each of which would end at noise 27 on its
own:

```mlir
%2 = noisy.mul %0, %1 : !noisy.i32
%3 = noisy.reduce_noise %2 : !noisy.i32
%4 = noisy.add %3, %1 : !noisy.i32      // branch 1: 13, 14, 15, 16
...
%8 = noisy.sub %3, %0 : !noisy.i32      // branch 2: 13, 14, 15, 16
...
```

One `reduce_noise`, placed *before the fork*, rescues both branches — the
globally-optimal move that no local pattern and no per-branch greedy pass
can see, because its justification lives in two different futures. This is
the whole chapter in one line of IR. (The test file's fifth function is the
control experiment: same shape, but the branches *remultiply* at the end,
and the solver correctly pays for **two more** cleanups, one per branch —
the `CHECK-COUNT-2` in the test pins down the count. And look *where* it
puts them: after each branch's third add, not its fourth, leaving each
branch at noise 13 so the final mul lands at 13 + 13 = 26 — exactly on
budget. The solver spends nothing on slack it doesn't need.)

Two debugging aids, both verified against the wheel (which ships with
assertions enabled, so `--debug-only` works):

```sh
# The ILP side: solve time and objective value (debug output is on stderr)
$ build-solver/tools/tutorial-opt --debug-only=ReduceNoiseAnalysis \
      --noisy-reduce-noise-optimizer test/part5/noisy_reduce_noise.mlir \
      2>&1 | head -3
Problem solved in 22 milliseconds
Solution:
Objective value = 4.220000e+01

# The dataflow side: upstream's per-op inference log
build-solver/tools/tutorial-opt --debug-only=int-range-analysis \
    --noisy-reduce-noise-optimizer test/part5/noisy_reduce_noise.mlir
```

That objective 42.2 decomposes as 8 insertions (3 + 0 + 1 + 1 + 3 across the
five functions) plus 0.1 × the sum of the add/sub `Z` variables — the
penalty trick from section 4, visible in the wild. (There is also a
commented-out `ExportModelAsLpFormat` block in the analysis if you want the
full model as text.)

**Testing.** The lit test carries `// REQUIRES: solver`; `lit.cfg.py` adds
the `solver` feature only when the build was configured with
`-DENABLE_SOLVER=ON`. So the same suite does the right thing in both builds:

```sh
cmake --build build --target check-part5          # noisy_reduce_noise: UNSUPPORTED
cmake --build build-solver --target check-part5   # all 15 tests run, incl. the solver one
```

## Where to go next

This completes the `poly`/`noisy` arc — from "what is a dialect" in Chapter
12 to a compiler that does operations-research-grade optimization on a domain
model. One more optimization idea closes Part 5:
[Chapter 24](../ch24-equality-saturation-egraphs/README.md) replaces the
greedy, order-sensitive rewriting of Chapters 16–17 with equality saturation
over an e-graph. After that, [Part 6](../../part6-end-to-end-ml/README.md)
leaves toy dialects behind and compiles *real ML models* (PyTorch, via
torch-mlir) down the linalg-and-bufferization highway you drove in
Chapter 22. And if the FHE
thread hooked you: everything here is a toy of [HEIR](https://heir.dev/),
the real compiler this material was warming up for — where `poly` grew into
a full `polynomial` dialect (upstreamed to MLIR for a time, now maintained
in HEIR — Chapter 12's fun fact) and noise management is a research area, not
a six-op toy.

## Exercises

1. Recompute section 5 by hand: annotate every value in
   `test_add_after_mul` (the file's third function) with its noise, and
   predict the single insertion point before looking — then check against
   the pass output (it goes right after the mul; can you argue from the
   noise numbers why *no other single location* works?).
2. The last mul in the squaring chain kept noise 24 ≤ 26 without help. Add a
   fifth `noisy.mul %8, %8` (plus a decode) to a copy of the test and
   predict: how many cleanups now, and where? Run it.
3. Prove (informally) that with this op set the ILP is *never* infeasible —
   i.e. the pass's FIXME about infeasibility can't trigger for
   verifier-legal input. (Sketch: inserting a cleanup after every arithmetic
   op bounds every value by max(12+1, 12+12) = 24 ≤ 26... almost — which
   op's constraint do you still need to check?)
4. The analysis treats *every* block argument as a fresh input at noise 12
   (the code comment calls this "a bit sloppy"). Construct a program where
   that assumption is wrong (hint: a function called with an already-noisy
   value) and explain what the fix would require (whole-program vs
   per-function analysis — the same tension as
   [Chapter 18](../../part4-lowering/ch18-dialect-conversion/README.md)'s
   function-boundary handling).
5. Estimate the cost of doing this "properly by hand": sketch what a greedy
   noise-tracking pass (insert when the *next* op would overflow) does on
   `test_single_insertion_branching`, and count its insertions vs the
   solver's one. Then read the ILP's variables for that function (uncomment
   the variable-printing loop in `ReduceNoiseAnalysis.cpp`, or just
   enumerate on paper) — how many variables and constraints did that one
   function generate?
