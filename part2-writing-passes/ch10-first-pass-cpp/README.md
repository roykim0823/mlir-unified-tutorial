# Chapter 10: Writing Your First Pass in C++

> Reworked from the mlir-tutorial fork's chapter 3 for the unified repo.

Part 1 ended with us running *upstream* passes through `mlir-opt` and
testing them with lit and FileCheck. This chapter is where we switch
sides of the tool: we write passes of our own, organize them into a
project, and drive them from `tutorial-opt`, this repo's own version of
`mlir-opt`.

> **Heads-up before you read any code:** the pass code this chapter
> teaches is the original **hand-written** form — a `PassWrapper`
> subclass per pass, with an explicit `PassRegistration` line in the
> driver. That is *not* what the main tree's `lib/Transform/` contains:
> those directories hold the same passes migrated to tablegen-generated
> boilerplate, and that migration is
> [Chapter 11](../ch11-tablegen-passes/README.md)'s subject. The
> hand-written form is preserved, buildable, in this chapter's
> [`code/`](code/) directory, and `cmake --build build -j` produces it as
> its own binary,
> `build/part2-writing-passes/ch10-first-pass-cpp/code/tutorial-opt-ch10`.
> It registers the same flags and produces byte-identical output to the
> main `tutorial-opt` on this chapter's tests, so every command below
> works with either binary. A short
> [What the repo has now](#what-the-repo-has-now) note at the end of
> section 4 summarizes how `lib/`'s versions differ.

## What you will learn

- How an MLIR project is organized on disk, and why an out-of-tree
  project needs its own `opt` tool.
- The anatomy of a pass: `runOnOperation`, pass *anchoring*, and why
  anchoring exists (hint: parallelism).
- Two ways to implement a pass: walking the IR directly, and using the
  **pattern rewrite engine**.
- The core `PatternRewriter` API: matching, creating, replacing, and
  erasing operations.
- How the greedy rewrite driver applies competing patterns to a fixed
  point — with folding thrown in for free.
- How a pass becomes a `--flag`: `PassRegistration`, `MlirOptMain`, and
  the CMake wiring in between.

## Prerequisites

- Part 1, especially
  [Chapter 9](../../part1-using-mlir/ch09-testing-with-lit/README.md)
  (lit and FileCheck — this chapter's tests use them without
  re-explanation).
  [Chapter 2](../../part1-using-mlir/ch02-lowering-pipelines/README.md)
  introduced pass pipelines and the `iter_args` loop-carried-value
  idiom, and
  [Chapter 5](../../part1-using-mlir/ch05-affine-optimization/README.md)
  the `affine` dialect — both reappear below.
- A configured build in `build/` per the top-level
  [README](../../README.md). `cmake --build build -j` builds everything
  in this chapter, including `tutorial-opt-ch10`; no LLVM source build
  is involved (the passes compile and link against the MLIR pip wheel).

Run every command from the repo root, with the virtualenv active
(`source .venv/bin/activate`). If you also want the wheel's own tools
(`mlir-opt`, `FileCheck`, ...) on your `PATH` for side experiments:

```bash
export PATH=$(python -m mlir_wheel --root-dir)/bin:$PATH
```

Set a shell variable once so this chapter's commands stay short:

```bash
CH05_OPT=build/part2-writing-passes/ch10-first-pass-cpp/code/tutorial-opt-ch10
```

(Substituting `build/tools/tutorial-opt` — the main driver, with the
Chapter 11 versions of these passes compiled in — gives the same output
for every command below.)

---

## 1. Concepts: passes, and why loop unrolling is the example

A **pass** is a unit of work that traverses the IR and analyzes or
transforms it. Everything a compiler does between parsing and emitting
machine code is organized as a pipeline of passes — the lowerings of
Part 1 were passes; so are classic optimizations. A pass doesn't have to
lower anything: it can rewrite IR *within* one abstraction level, which
is what both passes in this chapter do.

Our first example operates on the
**[`affine` dialect](https://mlir.llvm.org/docs/Dialects/Affine/)** you
met in Chapter 5 — the highest-level loop dialect in MLIR, and the
reason Part 1 kept saying "keep the program at the right abstraction
level for each optimization." Affine loops are loops whose bounds and
index expressions are *affine functions* (sums of variables times
constants, like `2*i + j + 5`) of loop variables and symbols. That
restriction is what the dialect sells: it was originally built for
polyhedral loop analysis — a mathematical framework that studies loop
nests via lattice theory — and it makes questions like "what is this
loop's trip count?" statically answerable. An `scf.for` loop can have
any dynamic bound; an
[`affine.for`](https://mlir.llvm.org/docs/Dialects/Affine/#affinefor-affineaffineforop)
loop is analyzable by construction.

We exploit exactly that: our pass **fully unrolls** every affine loop —
replacing the loop with one copy of its body per iteration. Unrolling is
a real optimization (it removes branch overhead and exposes parallelism,
and in some domains — like fully homomorphic encryption (FHE) compilers
— loops must be unrolled entirely because the target can't branch at
all). But honestly, we picked it because MLIR provides a one-call
utility, `loopUnrollFull`, that does the work. Both passes in this
chapter are deliberately thin: the point is to learn how to *set up* a
pass, navigate the IR through the C++ API, and modify it — not to write
a clever algorithm.

## 2. Project organization

Before writing code, decide where it goes. A typical MLIR codebase (the
LLVM monorepo included) splits code into two parallel directory trees:

- `include/` — header files and tablegen files, defining the public
  interface;
- `lib/` — the implementation code.

Within each, conventional subdirectories signal a component's role:

- `Transform/` — passes that transform code *within* a dialect (this
  chapter);
- `Conversion/` — passes that convert *between* dialects (the lowerings
  of Part 1; this repo's `lib/Conversion/PolyToStandard` is Part 4's
  subject);
- `Analysis/` — passes that compute information without modifying IR;
- `Dialect/` — definitions of the dialects themselves (Part 3).

This repo deviates in one deliberate way: **`include/` and `lib/` are
merged** — every header sits next to its implementation file. The
parallel tree exists in ordinary C++ projects partly to separate public
interface from private implementation, but navigating mirrored trees is
tedious, and this is a teaching codebase — simpler is better. So the
layout relevant to this chapter is:

```
lib/Transform/Affine/
├── AffineFullUnroll.h                    # walk-based pass (tablegen'd form, Ch. 11)
├── AffineFullUnroll.cpp
├── AffineFullUnrollPatternRewrite.h      # pattern-based pass (tablegen'd form, Ch. 11)
├── AffineFullUnrollPatternRewrite.cpp
├── Passes.td, Passes.h                   # tablegen boilerplate (Chapter 11)
└── CMakeLists.txt
lib/Transform/Arith/
├── MulToAdd.h, MulToAdd.cpp              # tablegen'd form (Ch. 11)
├── ...                                   # PDLL variant, from Chapter 17
tools/
└── tutorial-opt.cpp                      # the project's `opt` binary
test/part2/
├── affine_loop_unroll.mlir
└── mul_to_add.mlir
part2-writing-passes/ch10-first-pass-cpp/code/
├── AffineFullUnroll.{h,cpp}              # THIS chapter: hand-written form
├── AffineFullUnrollPatternRewrite.{h,cpp}
├── MulToAdd.{h,cpp}
├── tutorial-opt.cpp                      # pre-tablegen driver
└── CMakeLists.txt                        # builds tutorial-opt-ch10
```

The `tools/` convention: binaries live apart from libraries, and each
binary is a thin shell over library code — the drivers here are well
under a hundred lines each.

This chapter quotes its code from the `code/` snapshot — the
hand-written form, kept buildable so you can run exactly what you read.
The `lib/` twins have the same pass *bodies* inside generated shells;
section 4's closing note has the details.

## 3. Anatomy of a pass

A pass is a C++ class that supplies three functions — one that does the
work, two that name and describe its CLI flag — and inherits everything
else the pass API requires from a base class:

***part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnroll.h***
```cpp
class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "affine-full-unroll"; }

  StringRef getDescription() const final {
    return "Fully unroll all affine loops";
  }
};
```

The three functions we need to implement are:

- `runOnOperation()` is the heart of every pass — the method the pass
  manager calls to do the work.
- `getArgument()` is the CLI flag name.
- `getDescription()` is the CLI description shown by `--help` on the
  `mlir-opt`-like tool.

Everything else is inherited. `PassWrapper` fills in the rest of the
required pass API (such as a compliant clone/copy method — why a pass
must be *copyable* becomes clear just below) using the *Curiously
Recurring Template Pattern* — that's why the class passes itself as the
first template argument. CRTP lets the base class call methods of the
derived class without virtual dispatch; if it looks odd, treat it as an
idiom to copy, not master.

The second template argument deserves its own paragraph.
`OperationPass<mlir::func::FuncOp>` **anchors** the pass to function
operations: the pass manager will invoke `runOnOperation()` once per
`func.func` in the module, and inside the pass, `getOperation()` returns
the `FuncOp` currently being processed. (The MLIR docs formally describe
[what is required of an
OperationPass](https://mlir.llvm.org/docs/PassManagement/#operation-pass-static-filtering-by-op-type),
and in particular limit this anchoring to specific operations like
functions and modules.)

### Why anchor at all? Parallelism

Be careful what "parallelism" means here: it is *not* that different
passes run at the same time — a pipeline is strictly sequential
(canonicalize finishes, *then* cse starts). What runs concurrently is
**one pass, applied to many separate ops of its anchor type**. Anchoring
on `func.func` turns "run this pass" into one independent job per
function in the module, and the pass manager hands those jobs to a
thread pool (MLIR's context owns one; threading is on by default):

```
module { func.func @f   func.func @g   func.func @h }

thread 1:  runOnOperation() on @f
thread 2:  runOnOperation() on @g    <- same pass, three simultaneous calls
thread 3:  runOnOperation() on @h
────────────────── join ──────────────────
(only then does the next pass in the pipeline start)
```

Each thread does ordinary single-threaded work on its own private
subtree of the IR — the model is `make -j` compiling independent `.c`
files, with the pass manager as the scheduler. Two things exist
*because* of this fan-out: the pass manager doesn't share one pass
object across threads, it **clones** the pass once per thread — that is
what `PassWrapper`'s copy machinery above is for — and every
`MlirOptMain` tool accepts `--mlir-disable-threading` to force one
function at a time (handy when three threads' debug output interleaves).

And like `make -j`, no locks are taken anywhere: the *independence of
the jobs* is the entire safety argument. Two conditions make the jobs
truly independent, and the pass infrastructure enforces both:

1. **Your pass code must stay inside the op it was given.** Thread 1 is
   processing `@f`; if the pass reached over to inspect or edit `@g`
   ("let me check how `@g` calls this..."), it would be touching IR that
   thread 2 is mutating at that very moment — a data race. A pass that
   genuinely needs a whole-program view (inlining, say, which edits
   callers and callees together) anchors on the module instead: there is
   only one module, so one job on one thread. Correct, but serial —
   that is the price.

2. **Nothing inside the anchor op may be wired to anything outside it.**
   This one is subtle, because in MLIR "editing inside X" can physically
   write to memory *outside* X. SSA edges are two-way links: every
   `Value` keeps a **use-list**, and creating or erasing an op that uses
   `%x` writes to `%x`'s use-list. So if an op inside `@f`'s body could
   use a value defined outside `@f`, then a pass editing only `@f`'s
   interior would still be mutating shared state — even though its code
   never "looked" outside. The trait that rules this out is
   `IsolatedFromAbove`: nothing inside a `func.func` may reference an
   SSA value defined above it, so a function's interior is genuinely
   private to whichever thread holds it. The pass manager refuses to
   run a parallel pass anchored on any op without this trait.

The second condition explains why we can't just anchor the pass on
`affine.for` and skip the walk below. A loop is wired to its
surroundings in every direction (a sketch; section 5's real test input
has all three edges):

```mlir
%c = arith.constant 0 : i32               // defined outside the loop...
%r = affine.for %i = 0 to 4 iter_args(%acc = %c) -> i32 {
  %t = arith.addi %acc, %c : i32          // ...used inside the body
  affine.yield %t : i32
}
return %r : i32                           // loop result used outside
```

— and unrolling doesn't even stay inside the loop: it *deletes the
`affine.for` op itself* and splices the copied bodies into the parent
block, rewriting the uses of `%r`. Two threads doing that to two loops
in the same function would be editing the same block and the same
use-lists concurrently. So `affine.for` is not `IsolatedFromAbove`, and
MLIR won't let you build a parallel pass on it. Hence the shape of
passes like ours: anchor at the isolation boundary (`func.func`), then
*walk* — inside your own function, where you are the only thread — down
to the non-isolated ops you actually care about.

One footnote so the trait doesn't overpromise: "a function can't touch
outside variables" sounds wrong if you're thinking of C, where functions
mutate globals all the time. `IsolatedFromAbove` seals only the **SSA
graph** (the `%`-values). Memory is a separate channel — two functions
can happily store to the same `memref` — which is part of why condition
1 remains a rule your code must follow, rather than something the IR's
structure could enforce for you.

## 4. Implementing the pass: walking the IR

The implementation has exactly two jobs: visit every `affine.for` nested
anywhere in the function, and invoke MLIR's unroll utility on each one
it finds. Both come as ready-made API — a traversal method and a
one-call utility — which is why the entire body fits in seven lines:

***part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnroll.cpp***
```cpp
using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

// A pass that manually walks the IR
void AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}
```

`getOperation()` returns the `FuncOp` the pass is anchored on (section
3), though we don't use any specific information about it being a
function. We instead call its `walk` method — present on all `Operation`
instances — which traverses the abstract syntax tree (AST) of the
operation (here, the function body) in *post-order*: children before
parents, so for nested loops the inner loop is visited first, which is
exactly the order unrolling wants. For each operation `walk` encounters,
if that operation's type matches the input type of the callback, the
callback is executed — because our lambda takes an `AffineForOp`, it
runs only for `affine.for` ops, and everything else is skipped. This
"walk and match by type" idiom is the simplest way to visit IR.

Inside the callback, we attempt to unroll the loop, and if that fails we
quit with a diagnostic error. `loopUnrollFull(op)` is the upstream
utility doing the real work; it returns a `LogicalResult`, MLIR's
success/failure type (produce one with `success()`/`failure()`, test one
with `failed(...)`). On failure, `op.emitError(...)` attaches a
diagnostic to the offending operation — MLIR prints it with its source
location, like a compiler error — and `signalPassFailure()` is the
official way to tell the pass manager this pass failed, aborting the
pipeline.

Note what we did *not* write: no iteration over functions, no manual
recursion into loop bodies, no worklist. `walk` handles traversal, and
the pass manager handles everything outside `runOnOperation`.

### What the repo has now

The main tree,
[`lib/Transform/Affine/AffineFullUnroll.cpp`](../../lib/Transform/Affine/AffineFullUnroll.cpp),
has this same body inside a *tablegen-generated* shell —
`struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll>`
— where the generated base class supplies the flag name, description,
and registration, declared in
[`Passes.td`](../../lib/Transform/Affine/Passes.td) instead of C++. One
visible difference in the body: the generated pass is *generic* (not
anchored on `func.func`), so `getOperation()` returns an `Operation*`
and the walk is spelled `getOperation()->walk` (arrow, not dot). The
same goes for the other two passes in this chapter; the anchoring
*concept* from section 3 is unchanged and returns in later chapters. How
and why to migrate is [Chapter 11](../ch11-tablegen-passes/README.md)'s
whole subject — nothing more about it is needed today.

## 5. Step: run the unrolling pass

The test input
[`test/part2/affine_loop_unroll.mlir`](../../test/part2/affine_loop_unroll.mlir)
sums the 4 entries of a buffer (the `iter_args` loop-carried-sum idiom
is explained in Chapter 2; it exists to keep loops in compliance with
SSA form — for more on SSA in MLIR, see
[this MLIR doc](https://mlir.llvm.org/docs/LangRef/#high-level-structure)):

***test/part2/affine_loop_unroll.mlir*** (excerpt)
```mlir
func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
```

(`memref` is a new dialect/type: a *memory reference*, MLIR's typed view
of a buffer in memory — here, 4 `i32`s — with `affine.load` reading an
element at an index. It's the bridge between SSA values and actual
memory.)

Run the pass on it:

```bash
$CH05_OPT test/part2/affine_loop_unroll.mlir --affine-full-unroll
```

Output for that function:

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.load %arg0[%c0] : memref<4xi32>
    %1 = arith.addi %c0_i32, %0 : i32
    %2 = affine.apply #map(%c0)
    %3 = affine.load %arg0[%2] : memref<4xi32>
    %4 = arith.addi %1, %3 : i32
    %5 = affine.apply #map1(%c0)
    %6 = affine.load %arg0[%5] : memref<4xi32>
    %7 = arith.addi %4, %6 : i32
    %8 = affine.apply #map2(%c0)
    %9 = affine.load %arg0[%8] : memref<4xi32>
    %10 = arith.addi %7, %9 : i32
    return %10 : i32
  }
  ...
}
```

The loop is gone: four loads and four adds, one per iteration. The
`affine_map` lines at the top — those affine functions from section 1,
now in the flesh — and the `affine.apply` ops express the per-iteration
index arithmetic (`i+1`, `i+2`, `i+3`) symbolically. They're clutter
here (`%c0` is a constant, so each index is too), but the unroll utility
doesn't clean up after itself. Hold that thought for the next section.

(The test file's second function, `@test_doubly_nested_loop`, unrolls
into all 12 loads — the post-order walk reached the inner loop first.
Exercise 5 asks why that ordering matters.)

## 6. The same pass with the pattern rewrite engine

Walking the IR by hand scales poorly: real rewrites must worry about
*order* (does unrolling one loop expose another?) and *iteration* (keep
going until nothing changes). MLIR's **pattern rewrite engine** takes
over that bookkeeping. You provide *patterns* — small classes that say
"I match this kind of op, and here is how I rewrite it" — and a *driver*
applies them until no pattern matches anymore.

One thing to know before reading any code: this section's example is
**deliberately degenerate**. It re-does exactly the job of section 4 —
fully unrolling affine loops, by calling the very same upstream
`loopUnrollFull` utility — with a single pattern whose entire body is
one line. That is on purpose: with the *task* held constant, everything
new in the listing must be the *machinery*, and at the end you can diff
the two passes' outputs to see what the engine adds. Read this section
for the skeleton and that comparison; section 7 then writes patterns
that do real work through the rewriter API, and that is where
pattern-*writing* is actually taught.

Concretely, a rewrite pattern is a subclass of `OpRewritePattern` with a
single method to override, `matchAndRewrite`, which performs the
transformation. Its return value is the `LogicalResult` from section 4 —
a wrapper around a boolean, with `success()`/`failure()` to construct
one and `failed(...)` to test one. A relative worth meeting here is
`FailureOr<T>`: it interoperates with `LogicalResult` through the
presence or absence of a value — "either failure, or the thing you
computed."

Two contract points before reading the code. First, inside
`matchAndRewrite`, every mutation of the IR is supposed to go through
the
[`PatternRewriter`](https://mlir.llvm.org/docs/PatternRewriter/#pattern-rewriter)
argument: the rewriter is what makes a pattern *atomic*, guaranteeing
the changes take effect only if the method runs to the end and succeeds.
(Our pattern below violates this — `loopUnrollFull` has no variant that
accepts a `PatternRewriter` — and gets away with it for our limited test
cases.) Second, the *pass* is still anchored on `func.func`, but a
pattern can match *any* op type: the rewrite engine performs the walk we
wrote by hand in section 4 (an optional configuration struct can choose
the walk order), collecting any number of patterns from a
`RewritePatternSet` and greedily applying whichever ones match — in
[an order related to their `benefit`](https://mlir.llvm.org/docs/PatternRewriter/#greedy-pattern-rewrite-driver)
— until no operation matches, every applicable pattern returns failure,
or a large iteration limit trips to avoid infinite loops.

### The skeleton: which lines are boilerplate, which line is yours

Every pattern-based pass in MLIR — this section's, section 7's, and the
ones in the rest of this tutorial — is assembled from the same two
pieces of scaffolding, each with exactly one slot for your logic (a
sketch, not a repo file):

```cpp
// Piece 1: a pattern class. Fixed shape; only the body of
// matchAndRewrite is yours.
struct SomePattern : public OpRewritePattern<SomeOp> {
  SomePattern(mlir::MLIRContext *context)
      : OpRewritePattern<SomeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(SomeOp op,
                                PatternRewriter &rewriter) const override {
    // <-- your logic: inspect `op`; mutate through `rewriter`;
    //     return success()/failure()
  }
};

// Piece 2: the pass body. Fixed three lines; only the list of
// patterns you add is yours.
void SomePass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<SomePattern>(&getContext());   // one line per pattern
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
```

The scaffolding lines are worth understanding once, so that from now on
you can skip over them and read only the payload:

- **The constructor** is pure ceremony: it forwards the `MLIRContext`
  and a `benefit` to the base class and does nothing else. The context
  is the per-tool "global state" object that owns every uniqued type,
  attribute, and registered op — the base class needs it to identify
  which op the pattern matches. `benefit` is the pattern's priority:
  when several patterns match the same op, the driver tries
  higher-benefit ones first. With a single pattern it is moot (any
  constant works); it earns its keep in section 7, where two patterns
  compete for the same op.
- **`matchAndRewrite` does two jobs in one method**: decide whether this
  op is one the pattern applies to, and if so transform it. Return
  `failure()` to say "I don't match" — which must leave the IR
  completely untouched — or rewrite the IR and return `success()`.
- **`RewritePatternSet patterns(&getContext())`** is a container for
  however many patterns the pass wants applied together. `getContext()`
  here is the *pass's* accessor for the same `MLIRContext`.
- **`patterns.add<SomePattern>(&getContext())`** constructs one instance
  of the pattern inside the set, forwarding its arguments to the
  pattern's constructor — which is why `&getContext()` appears twice:
  once for the set itself, once passed through to `SomePattern`'s
  constructor above.
- **`applyPatternsGreedily(getOperation(), std::move(patterns))`** is
  the driver invocation. `getOperation()` is the op the pass is anchored
  on — for this pass, the `func.func` currently being processed
  (section 3) — and the driver confines itself to everything nested
  inside it, which is exactly the region section 3 argued a parallel
  pass may touch. The `std::move` is required because the driver takes
  ownership of the patterns (it "freezes" the set into an immutable,
  lookup-optimized form before applying it). Its `LogicalResult` return
  value reports whether the rewriting *converged* — reached a fixed
  point before the driver's iteration cap — and the `(void)` discards
  it; a production pass would test it and call `signalPassFailure()`
  (section 4) instead.

When you read any pattern-based pass from here on, look for three
things and ignore the rest: (a) which op type each pattern matches,
(b) what each `matchAndRewrite` body does, (c) which patterns the pass
registers.

This chapter's second pass re-implements unrolling this way (the main
tree's
[twin](../../lib/Transform/Affine/AffineFullUnrollPatternRewrite.cpp)
has the identical pattern inside the generated shell — see section 4's
note):

***part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnrollPatternRewrite.cpp***
```cpp
// A pattern that matches on AffineForOp and unrolls it.
struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp> {
  AffineFullUnrollPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    // This is technically not allowed, since in a RewritePattern all
    // modifications to the IR are supposed to go through the `rewriter` arg,
    // but it works for our limited test cases.
    return loopUnrollFull(op);
  }
};

// A pass that invokes the pattern rewrite engine.
void AffineFullUnrollPassAsPatternRewrite::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<AffineFullUnrollPattern>(&getContext());
  // One could use GreedyRewriteConfig here to slightly tweak the behavior of
  // the pattern application.
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
```

Hold it against the skeleton and the whole file is the template plus
one payload line:

```cpp
return loopUnrollFull(op);
```

Reading answers to the three questions: (a) the pattern matches
`affine.for`; (b) its body
delegates to the same utility section 4 called — there is no explicit
match logic because `loopUnrollFull` handles *any* `affine.for`, and its
`LogicalResult` doubles as the pattern's verdict; (c) the pass registers
that one pattern. What's left is what the engine itself contributes:

- The *greedy driver* behind `applyPatternsGreedily` keeps a worklist of
  operations. It pops an op, tries the patterns that could match it
  (highest benefit first), and — whenever a rewrite changes the IR —
  puts the affected ops back on the worklist, repeating until a **fixed
  point**: a state in which no pattern matches anything anymore. That is
  the "order" and "iteration" bookkeeping from this section's opening
  paragraph, and it is why neither the pattern nor the pass contains a
  loop: unrolling an outer `affine.for` exposes the inner copies, and
  the driver simply finds them on the next round.
- The driver also runs op *folders* along the way: built-in per-op
  simplifications like `0 + x → x` or evaluating constant expressions.
  You'll see their fingerprints in the output below. (Older MLIR — and
  older tutorials — spell this function `applyPatternsAndFoldGreedily`,
  advertising the folding in the name; upstream renamed it, and the
  folding behavior stayed. If you meet the long name in pre-2025 pass
  code, it's the same function.)
- Note the source comment: an honest rewrite pattern performs all IR
  changes through the `rewriter` argument (as we do in section 7), so
  the driver can track what changed and re-examine affected ops. Calling
  `loopUnrollFull` behind the driver's back is a shortcut that happens
  to survive our tests.

Run it and compare with section 5:

```bash
$CH05_OPT test/part2/affine_loop_unroll.mlir --affine-full-unroll-rewrite
```

```mlir
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<4xi32>
    %1 = affine.load %arg0[%c1] : memref<4xi32>
    %2 = arith.addi %0, %1 : i32
    %3 = affine.load %arg0[%c2] : memref<4xi32>
    %4 = arith.addi %2, %3 : i32
    %5 = affine.load %arg0[%c3] : memref<4xi32>
    %6 = arith.addi %4, %5 : i32
    return %6 : i32
  }
  ...
}
```

Same unrolling — but cleaner. The driver's folding already collapsed the
`affine.apply` index arithmetic into constants and deleted the useless
`0 + %0` add from the first iteration. You get standard cleanups for
free just by going through the engine.

## 7. A real rewrite: multiplications into additions

The unroll pattern delegated to an upstream utility. Now we build
patterns that construct IR themselves, using the `rewriter` API properly
— this is where you learn the vocabulary used by every rewrite pattern
ever written.

The `--mul-to-add` pass replaces multiplication-by-constant with
repeated addition (imagine a target where multiplication is much more
expensive than addition). Two cooperating patterns live in
[`code/MulToAdd.cpp`](code/MulToAdd.cpp) (main-tree twin:
[`lib/Transform/Arith/MulToAdd.cpp`](../../lib/Transform/Arith/MulToAdd.cpp)
— same two patterns, generated shell):

- **PowerOfTwoExpand** (benefit 2): `y = C*x` → `y = (C/2)*x + (C/2)*x`
  when `C` is a power of two.
- **PeelFromMul** (benefit 1): `y = C*x` → `y = (C-1)*x + x` otherwise.

Applied repeatedly, `9*x` becomes `8*x + x`, then the `8*x` halves its
way down to single additions — about `log C` additions total rather than
the `C` you'd get from peeling alone.

One general principle organizes every pattern body, and it's worth
having in mind before reading the code: a pattern runs in two phases.
The **match phase** may only *inspect* the IR — walk operands, check
properties — and must bail out with `failure()` before anything has been
touched; the **rewrite phase**, entered only once the match is certain,
makes every change through the `rewriter` argument (section 6's
atomicity contract). Here is `PowerOfTwoExpand`, read with that split in
mind:

***part2-writing-passes/ch10-first-pass-cpp/code/MulToAdd.cpp***
```cpp
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);

    // canonicalization patterns ensure the constant is on the right, if there
    // is a constant
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;

    if (!is_power_of_two) {
      return failure();
    }

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value / 2));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};
```

The specifics, phase by phase:

**Match phase** — may only *inspect*:

- `op.getOperand(n)` fetches an input `Value`, the type representing an
  SSA value (i.e., an MLIR variable).
- `rhs.getDefiningOp<OpTy>()` walks *backwards* through the SSA graph to
  the operation that produced the value — returning null if that op is
  not an `OpTy`. Here it asks: "is the right operand a constant
  integer?" (The source comment explains why checking only the *right*
  operand suffices: MLIR's
  [global canonicalization rules](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules)
  move constants to the right of commutative ops, so `9*x` and `x*9`
  both arrive as `muli %x, %c9`.)
- `(value & (value - 1)) == 0` is the classic power-of-two bit trick.
- Every non-match exits with `failure()` **before any mutation**. This
  is the pattern-writing contract: a pattern must not touch the IR and
  then bail, or the driver's bookkeeping breaks.

**Rewrite phase** — every change goes through `rewriter`:

- `rewriter.create<OpTy>(loc, ...)` builds a new op. Every op carries a
  *location* (`getLoc()`) — MLIR's source-tracking breadcrumb, used to
  produce meaningful diagnostics after many rounds of rewriting; reusing
  the original ops' locations keeps that chain intact.
- `rewriter.getIntegerAttr(type, value)` builds the compile-time
  constant
  [*attribute*](https://mlir.llvm.org/docs/LangRef/#attributes) the new
  `ConstantOp` holds (attributes are how ops store static data, as
  opposed to runtime SSA operands — you can put strings or dictionaries
  in an attribute, but for `ConstantOp` it's just an int).
- `rewriter.replaceOp(op, newAdd)` rewires all uses of `op`'s results to
  `newAdd`'s results and deletes `op` — the moral equivalent of "return
  the new expression."
- `rewriter.eraseOp(rhsDefiningOp)` deletes the now-unused constant. (If
  it were still used elsewhere this would be wrong — a real pass would
  let dead-code elimination handle it; here it's safe for the test
  cases.)

`PeelFromMul` is nearly identical, with a lower benefit and no
power-of-two check:

***part2-writing-passes/ch10-first-pass-cpp/code/MulToAdd.cpp*** (excerpt)
```cpp
    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value - 1));  // value - 1 instead of value / 2
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);
```

The **benefit values encode the strategy**: when `C` is a power of two,
*both* patterns match the op, and benefit makes the driver prefer
halving (log-many steps) over peeling (linearly many). It also lets
`PeelFromMul` skip re-checking: its body notes it is *guaranteed* `C` is
not a power of two, because the higher-benefit pattern already had its
chance. The pass itself has the same shape you saw in section 6 —
collect the patterns, hand them to the greedy driver:

***part2-writing-passes/ch10-first-pass-cpp/code/MulToAdd.cpp***
```cpp
void MulToAddPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<PowerOfTwoExpand>(&getContext());
  patterns.add<PeelFromMul>(&getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
```

Run it on
[`test/part2/mul_to_add.mlir`](../../test/part2/mul_to_add.mlir), which
multiplies by 8 and by 9:

```bash
$CH05_OPT test/part2/mul_to_add.mlir --mul-to-add
```

```mlir
module {
  func.func @just_power_of_two(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    return %2 : i32
  }
  func.func @power_of_two_plus_one(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    %3 = arith.addi %2, %arg0 : i32
    return %3 : i32
  }
}
```

Trace the second function yourself: `9*x` → peel → `8*x + x` → three
halvings → the doubling chain `%0, %1, %2`, plus the final `+ %arg0`
from the peel. All multiplications are gone, reached purely by local
patterns running to a fixed point.

## 8. Walk or pattern rewrite?

With two ways to define a pass — walk the entire IR from the root
operation, or match and rewrite patterns through the rewrite engine —
the natural question is when to use which.

Notice first that the choice is not about *power*. The pattern engine
expresses a convenient subset of what a pass can do — conceptually
trivially so: anyone who can walk the whole tree can, with enough
effort, do anything at all, up to and including reimplementing the
pattern rewrite engine. The engine's case rests on convenience, and on
history: MLIR's own
[Generic DAG Rewriter Infrastructure Rationale](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/)
lays out the motivation, distilled from a long line of pattern-matching
systems in the LLVM project and elsewhere.

What the engine is convenient *for* is **local** transformations.
"Local" means the situation you're rewriting can be recognized from a
small subset of the IR viewed as a directed acyclic graph —
pragmatically, anything you can detect by looking around at an op's
neighbors in the same block and applying some filtering logic. "Is this
`exp` followed by a `log`, with no other uses of the `exp`'s result?" is
local. `PowerOfTwoExpand`'s "is my right operand a constant, and is it a
power of two?" is local — one hop up the SSA graph and a bit test.

Some analyses and optimizations, by contrast, must construct the entire
dataflow of a program before they can act. Common subexpression
elimination is the classic example: deciding whether it's
*cost-effective* to pull a subexpression used in multiple places into
one variable depends on the operation's cost and on what keeping an
extra value alive does to memory access and register availability at
that point in the program — information no amount of local
pattern-matching can see. For work like that — or for custom traversal
orders, or one-shot structural surgery — you walk, and you own the
iteration logic yourself.

The accumulated wisdom: when the transformation fits, the pattern engine
is usually the *easier* path. You don't write sprawling case/switch
logic over everything that might appear in the IR; you don't hand-roll
the "keep going until nothing changes" loop (the engine re-applies
patterns for you, with section 6's freebies — fixed-point iteration,
benefit prioritization, folding); and each pattern can be written in
isolation, trusting the engine to combine them appropriately — exactly
how `PowerOfTwoExpand` and `PeelFromMul` teamed up in section 7. Most
rewrites in the wild are pattern-based; the walk is the escape hatch.

## 9. How the pass gets into a binary: registration and the driver

So far the passes have simply *been there* behind `$CH05_OPT` flags.
Time to close the loop: why do we need our own binary at all, and what
turns a C++ class into a `--flag`?

Why can't we keep using `mlir-opt`? Because a binary only contains the
passes *compiled into it*. `mlir-opt` is built inside the LLVM project
from upstream code only; it has never heard of `affine-full-unroll`. In
an out-of-tree project (one living outside the LLVM monorepo, like this
one), you build your own tool.

MLIR makes this nearly free. It provides *registration hooks* to plug in
dialects and passes, and a library entry point, `MlirOptMain`, that
provides the entire `mlir-opt` command-line experience — parsing input,
verifying it, exposing every registered pass as a `--flag`, `--help`,
`--pass-pipeline`, IR printing options — on top of whatever you
registered. Here is this chapter's driver in full (minus the orientation
comment at the top of the file):

***part2-writing-passes/ch10-first-pass-cpp/code/tutorial-opt.cpp***
```cpp
#include "part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnroll.h"
#include "part2-writing-passes/ch10-first-pass-cpp/code/AffineFullUnrollPatternRewrite.h"
#include "part2-writing-passes/ch10-first-pass-cpp/code/MulToAdd.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();
  mlir::PassRegistration<mlir::tutorial::MulToAddPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```

Line by line:

- `DialectRegistry` collects the dialects the tool is allowed to
  *parse*. `registerAllDialects` throws in every upstream dialect —
  convenient for a tutorial; a production tool would register only what
  it handles, to keep the binary small.
- `mlir::PassRegistration<PassT>()` makes one pass visible to the pass
  registry — this is what turns `getArgument()`'s string into an
  `--affine-full-unroll` style flag. One line per pass; forgetting the
  line is the classic "my pass builds but the flag doesn't exist"
  mistake.
- `MlirOptMain(...)` runs the whole show; `asMainReturnCode` converts
  its `LogicalResult` into a process exit code.

The effect is visible immediately:

```bash
$CH05_OPT --help | grep -E 'affine-full-unroll|mul-to-add'
```

```
      --affine-full-unroll                                   -   Fully unroll all affine loops
      --affine-full-unroll-rewrite                           -   Fully unroll all affine loops using the pattern rewrite engine
      --mul-to-add                                           -   Convert multiplications to repeated additions
```

Those descriptions are `getDescription()` verbatim — and they are the
*only* passes this driver has: no `registerAllPasses()` call, so its
`--help` is roughly 300 lines of general options plus our three flags.

Compare the main driver,
[`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp): it calls
`mlir::registerAllPasses()` — every stock MLIR pass, pushing its
`--help` past a thousand lines; your tool becomes a superset of
`mlir-opt` — and instead of `PassRegistration` lines it calls generated
functions, `mlir::tutorial::registerAffinePasses()` and
`registerArithPasses()`, that come out of the tablegen machinery
(Chapter 11). Running the same `--help | grep` against it also shows a
fourth flag, `--mul-to-add-pdll`, which belongs to a much later chapter
(Chapter 17).

### The build wiring

The chain from `.cpp` to `--flag` is worth seeing once. In
[`code/CMakeLists.txt`](code/CMakeLists.txt), the three passes form one
library, and the driver is an executable linking it plus MLIR's
opt-tool machinery:

***part2-writing-passes/ch10-first-pass-cpp/code/CMakeLists.txt*** (excerpt)
```cmake
add_mlir_library(TutorialCh05Passes
    PARTIAL_SOURCES_INTENDED
    AffineFullUnroll.cpp
    AffineFullUnrollPatternRewrite.cpp
    MulToAdd.cpp
    ...
)

add_llvm_executable(tutorial-opt-ch10 tutorial-opt.cpp)

target_link_libraries(tutorial-opt-ch10 PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    TutorialCh05Passes
    MLIROptLib          # provides MlirOptMain
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions
    MLIRPass
)
```

`add_mlir_library`/`add_llvm_executable` are the wrappers MLIR's CMake
exports provide (the wheel ships them along with the headers and
libraries — that's what the top-level `CMAKE_PREFIX_PATH` setup found).
The main tree's equivalents are
[`lib/Transform/Affine/CMakeLists.txt`](../../lib/Transform/Affine/CMakeLists.txt)
and [`tools/CMakeLists.txt`](../../tools/CMakeLists.txt).

When you add a new pass of your own, you touch: the pass files, a
`CMakeLists.txt` library entry, and a registration call in the driver —
that's the whole checklist. Then rebuild:

```bash
cmake --build build -j
```

One quality-of-life note while you're editing pass code: configuring
CMake with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` generates the
[`compile_commands.json`](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
that clangd needs — C++ editing against MLIR's headers without a
language server is painful.

## 10. Step: test the passes

The lit/FileCheck machinery from Chapter 9 applies unchanged — the only
news is that `RUN` lines now invoke `tutorial-opt`, which is on lit's
`PATH` because the lit config ([`lit.cfg.py`](../../lit.cfg.py))
prepends the build's `tools/` directory. (The lit tests exercise the
*main* driver; this chapter's `tutorial-opt-ch10` is flag- and
output-compatible, as the demos above showed.) The test file's RUN
header:

***test/part2/affine_loop_unroll.mlir*** (excerpt)
```mlir
// RUN: tutorial-opt %s --affine-full-unroll > %t
// RUN: FileCheck %s < %t

// RUN: tutorial-opt %s --affine-full-unroll-rewrite > %t
// RUN: FileCheck %s < %t

func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  // CHECK-LABEL: test_single_nested_loop
  // CHECK-NOT: affine.for
  ...
```

One file tests both pass variants: two RUN pipelines, same CHECK
assertions — "whatever else you produce, no `affine.for` may survive." A
deliberately loose test (Chapter 9's trade-off discussion): it stays
green across both implementations and any future cleanup changes. The
`mul_to_add.mlir` test is the opposite call — it pins the exact addition
chain with capture variables (`%[[SUM_0:.*]]` and friends), because the
addition structure *is* the pass's contract. The unroll file also has a
second test function with a *nested* loop, checking that `walk` reaches
inner loops.

Run them:

```bash
source .venv/bin/activate
lit build/test --filter=part2 -v
# equivalently: cmake --build build --target check-part2
```

```
-- Testing: 6 of 33 tests, 6 workers --
PASS: MLIRUnifiedTutorial :: part2/control_flow_sink.mlir (1 of 6)
PASS: MLIRUnifiedTutorial :: part2/cse.mlir (2 of 6)
PASS: MLIRUnifiedTutorial :: part2/sccp.mlir (3 of 6)
PASS: MLIRUnifiedTutorial :: part2/code_motion.mlir (4 of 6)
PASS: MLIRUnifiedTutorial :: part2/mul_to_add.mlir (5 of 6)
PASS: MLIRUnifiedTutorial :: part2/affine_loop_unroll.mlir (6 of 6)

Total Discovered Tests: 33
  Excluded: 27 (81.82%)
  Passed  :  6 (18.18%)
```

(The other four `part2` tests are Part 1 Chapter 5's upstream-pass
demos, which live in the same lit suite.)

## Where to go next

You have now written (well, read — the exercises fix that) both styles
of pass, know why passes are anchored, and have the rewriter vocabulary:
match via `getDefiningOp`, rewrite via
`rewriter.create`/`replaceOp`/`eraseOp`, and let the greedy driver
handle iteration. You also know the full checklist that turns a class
into a flag: `PassWrapper` + `getArgument` + `PassRegistration` + CMake.

Next, [Chapter 11: Tablegen for Passes](../ch11-tablegen-passes/README.md)
replaces the by-hand parts of that checklist with generated code —
`Passes.td`, `Passes.h.inc`, and the `registerAffinePasses()` you saw in
the main driver — which is exactly the difference between this chapter's
`code/` snapshot and the main tree's `lib/Transform/` versions.

**Exercises**

1. Predict the output of `--mul-to-add` for multiplication by 7 (write
   the pattern applications out by hand: which pattern fires at each
   step, and why?). Then check yourself: change the constant in a copy
   of `test/part2/mul_to_add.mlir` and run `$CH05_OPT` on it. (To check
   your reasoning, not your reading: three peels and two halvings, five
   `arith.addi` ops total.)
2. What about `C = 1` or `C = 0`? Trace the code:
   `(value & (value - 1)) == 0` is true for both, so *PowerOfTwoExpand*
   claims them and would rewrite `1*x` into `0*x + 0*x` — heading
   somewhere unpleasant. Now run it. The actual output is just
   `return %arg0` (resp. a constant 0): the greedy driver's built-in
   **folders** simplify `1*x` and `0*x` away before the pattern ever
   fires. Safety nets like this are part of why rewrites go through the
   driver.
3. Rerun `--mul-to-add` with `--mlir-print-ir-after-all` to see the IR
   after each pass; then try `--debug` if your build has assertions
   enabled to see each pattern application the greedy driver attempts.
4. Modify the walk-based pass (in a scratch copy of `code/` — the
   snapshot itself should stay as the chapter presents it) to unroll
   only loops whose trip count exceeds a threshold (peek at
   `AffineForOp`'s methods like `getConstantUpperBound` in the upstream
   headers), rebuild, and watch the test fail — then adjust the test.
5. The doubly nested loop in `affine_loop_unroll.mlir` unrolls to 12
   loads. Why does post-order traversal matter for it? What would happen
   on a pre-order walk that unrolled the outer loop first? (The answer
   is "it still works, but think about why" — the unrolled copies of the
   inner loop are new ops the walk has to find.)
