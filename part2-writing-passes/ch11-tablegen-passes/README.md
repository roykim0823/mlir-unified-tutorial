# Chapter 11: Tablegen for Passes

> Reworked from the mlir-tutorial fork's chapter 4 for the unified repo.

[Chapter 10](../ch10-first-pass-cpp/README.md) ended with an IOU: it taught
the hand-written `PassWrapper` form of the passes — preserved verbatim in
[`ch10-first-pass-cpp/code/`](../ch10-first-pass-cpp/code/) — while this
repo's `lib/` holds the *same* passes in a form we only glimpsed: a
mysterious `impl::AffineFullUnrollBase` base class, and, in the real
[`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp), a
`registerAffinePasses()` registration function that nobody ever wrote.
This chapter pays the debt — those are all *generated* by a tool called
**tablegen**, and here we read the 22 lines that produce them, run the
generator by hand, and read every line it emits.

A word of expectation-setting: the original author of these chapters
describes a **love-hate relationship** with tablegen — the hate earned
mostly by what happens when something goes wrong (poor diagnostics, thin
documentation), the love recovered by a change of mindset that section 1
hands you up front and section 8 stress-tests against real error
messages. Going in with the right expectations is half of using tablegen
well. The ch10 snapshot preserves the pre-migration state (and builds as
its own binary, `tutorial-opt-ch10`), so before and after can be laid
side by side — which is exactly what section 5 does; the `lib/` files we
read here are that migration's end state.

## What you will learn

- What tablegen is, and the right mental model for it ("see-through code
  generator", not abstraction layer).
- The anatomy of `Passes.td` — `def`, `Pass<"...">`, `summary`,
  `description`, `dependentDialects`.
- How to run `mlir-tblgen` yourself (it ships in the pip wheel), and how
  this repo's CMake wires it into the build.
- How to read the generated `Passes.h.inc`: the `GEN_PASS_DECL`,
  `GEN_PASS_DEF`, and `GEN_PASS_REGISTRATION` sections, and the
  `#define`/`#include` idiom that consumes them.
- What the migration deleted from Chapter 10's hand-written passes, and
  what tablegen offers beyond them (doc generation, pass options).
- What tablegen's failure modes look like — with real error messages.

## Prerequisites

- [Chapter 10](../ch10-first-pass-cpp/README.md), including its `code/`
  snapshot ([`code/README.md`](../ch10-first-pass-cpp/code/README.md)
  maps each file to the chapter section it came from).
- A configured build in `build/` per the top-level
  [README](../../README.md). No rebuilding is needed in this chapter —
  the generated files already sit in your build tree — but we will run
  `mlir-tblgen` by hand, and it ships inside the MLIR pip wheel.

Run every command in this chapter from the repo root, with the
virtualenv active and the wheel's `bin/` on your `PATH`:

```bash
source .venv/bin/activate
export WHEEL_ROOT=$(python -m mlir_wheel --root-dir)
export PATH=$WHEEL_ROOT/bin:$PATH
```

---

## 1. Concepts: what tablegen is

Nobody writing MLIR passes hand-implements the pass interface the way
Chapter 10 showed. They write a few lines in a **tablegen** file and let a
generator produce the boilerplate, keeping only `runOnOperation` (and
whatever else is genuinely custom) in C++.

Tablegen is LLVM's domain-specific language for *structured records*
(the language itself is documented in the
[TableGen Programmer's Reference](https://llvm.org/docs/TableGen/ProgRef.html)).
A `.td` file declares records with typed fields; a backend of the
`mlir-tblgen` tool walks those records and prints C++ code (or markdown,
or anything else a backend chooses). It is used everywhere in the LLVM
universe — instruction definitions for CPU backends, clang diagnostics —
and MLIR leans on it harder than anyone: passes (this chapter), dialects,
ops, types, and rewrite patterns (Part 3) are all declared in tablegen —
MLIR's specialized flavor for ops, the
[Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
(ODS), is Chapter 12's territory.

One mindset note before we start, straight from the original author's
battle scars: tablegen is **not an abstraction layer**. You cannot use it
well while ignoring what it generates — when something goes wrong, the
errors surface in the *generated* C++, and the documentation for "what
fields can I set?" is largely the upstream `.td` base files themselves.
Treat it as a totally see-through code generator: write the `.td`, then
go *read the output* (we will). Once you approach it that way, it's
genuinely pleasant — the generated code is more complete and more correct
than what you'd write by hand.

The pain concentrates in specific, predictable places:

- Tablegen won't clearly tell you **which functions are left for you to
  implement**, nor spell out their contracts — the answers live in the
  upstream sources and, not rarely, in
  [forum threads](https://discourse.llvm.org/c/mlir/31).
- The practical way to discover what's missing is to **build anyway and
  read the C++ compiler errors** — which, for template-heavy generated
  code, can run long (section 8 shows a real specimen).
- The generated code lives in **namespaces you have to manage**: using it
  means knowing which generated symbols to pull in, where. Section 4's
  `GEN_PASS_*` / `#include` gates are exactly this bookkeeping.

## 2. The tablegen file

Here is [`lib/Transform/Affine/Passes.td`](../../lib/Transform/Affine/Passes.td)
in full — 22 lines that replace roughly a hundred hand-written ones. To
read it you need only section 1's model plus two keywords: `def`
*instantiates* a record from a record class (here the upstream `Pass`
class — so one `def` per pass), and each `let` fills in one of that
class's typed fields. Everything between the guard lines is just records
and field assignments:

***lib/Transform/Affine/Passes.td***
```tablegen
#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite engine.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

#endif  // LIB_TRANSFORM_AFFINE_PASSES_TD_
```

Field by field:

- `include "mlir/Pass/PassBase.td"` — imports the upstream `Pass` record
  class. `PassBase.td` is also the de-facto documentation for what you
  may `let`: the wheel ships it at
  `$WHEEL_ROOT/include/mlir/Pass/PassBase.td`, so open it there when you
  wonder what else is settable. The official prose counterpart is the
  [declarative pass specification](https://mlir.llvm.org/docs/PassManagement/#declarative-pass-specification)
  section of the pass docs. (The `#ifndef` guard works like a C header
  guard; `.td` files use the C preprocessor conventions.)
- The record header:

  ```tablegen
  def AffineFullUnroll : Pass<"affine-full-unroll"> {
  ```

  `def` *instantiates* a record (as opposed to
  `class`, which declares a reusable template like `Pass` itself). The
  record's name, `AffineFullUnroll`, becomes the generated class-name stem
  (`AffineFullUnrollBase`) and the macro suffix
  (`GEN_PASS_DEF_AFFINEFULLUNROLL`).
- `Pass<"affine-full-unroll">` — the template argument is the CLI flag,
  what Chapter 10's hand-written `getArgument()` returned. `Pass` takes an
  optional second argument naming the anchor op (e.g.
  `Pass<"...", "mlir::func::FuncOp">`); omitted here, so these are generic
  passes anchored on any op — this is exactly where the repo's passes lost
  the `FuncOp` anchoring that Chapter 10's hand-written code had (and why
  `lib/`'s `runOnOperation` writes `getOperation()->walk` with an arrow
  where the ch10 snapshot writes `getOperation().walk` with a dot).
- `let summary = ...` — the `--help` one-liner (`getDescription()` in the
  hand-written version — confusingly, tablegen's `summary` maps to the C++
  `getDescription()`).
- The long-form documentation:

  ```tablegen
  let description = [{
    Fully unroll all affine loops.
  }];
  ```

  `[{` `}]` is tablegen's heredoc syntax for multi-line strings. Used by
  the doc generator (section 7), not by `--help`.
- `let dependentDialects = [...]` — the important one. An `MLIRContext`
  loads dialects lazily, and a pass may only create ops from dialects that
  are already loaded. Unrolling creates fresh `affine.apply` ops, so the
  pass must declare the affine dialect, and the generated code will load
  it before the pass runs. Forgetting a dependent dialect is a classic
  runtime crash ("op created with unregistered dialect") that this field
  exists to prevent.

## 3. Step: run the generator by hand

The build runs `mlir-tblgen` for you, but running it manually makes the
whole mechanism concrete. The only flags are the backend selector
(`-gen-pass-decls`), a `-name` for the pass *group*, and an include path
so tablegen can find `PassBase.td` — the wheel's `include/` directory
serves for that:

```bash
mlir-tblgen -gen-pass-decls -name Affine \
  lib/Transform/Affine/Passes.td -I $WHEEL_ROOT/include | wc -l
```

```
221
```

Twenty-two lines in, 221 lines of C++ out. The `-name Affine` argument is
what groups the two passes: it names the umbrella registration function
`registerAffinePasses()`.

In the real build you never see this command. The CMake in
[`lib/Transform/Affine/CMakeLists.txt`](../../lib/Transform/Affine/CMakeLists.txt)
runs it:

***lib/Transform/Affine/CMakeLists.txt*** (excerpt)
```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRAffineFullUnrollPasses)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
```

Each `mlir_tablegen`/`add_mlir_doc` line is one `mlir-tblgen` run with
those flags. The output lands in the build tree
(`build/lib/Transform/Affine/Passes.h.inc`), and because the build tree
is on the include path, C++ can write

```cpp
#include "lib/Transform/Affine/Passes.h.inc"
```

as if the file lived in the source tree. The
`add_mlir_library(AffineFullUnroll ...)` entry above
these lines lists `DEPENDS MLIRAffineFullUnrollPasses`, so generation
runs before the `.cpp` files compile.

The manual command and the build produce the same bytes — verify it:

```bash
mlir-tblgen -gen-pass-decls -name Affine \
  lib/Transform/Affine/Passes.td -I $WHEEL_ROOT/include \
  | diff - build/lib/Transform/Affine/Passes.h.inc && echo IDENTICAL
```

```
IDENTICAL
```

## 4. Reading the generated code

Open `build/lib/Transform/Affine/Passes.h.inc` (or pipe the section-3
command through `less`) and you'll find it is *three* files in one, each
gated by a preprocessor macro so that consumers opt in to exactly the
part they need. All excerpts below are from the real generated file (the
wheel's `mlir-tblgen`, LLVM 24 at the time of writing — other versions
differ cosmetically), lightly trimmed.

### `GEN_PASS_DECL_*` — the public face

```cpp
#ifdef GEN_PASS_DECL_AFFINEFULLUNROLL
std::unique_ptr<::mlir::Pass> createAffineFullUnroll();
#undef GEN_PASS_DECL_AFFINEFULLUNROLL
#endif // GEN_PASS_DECL_AFFINEFULLUNROLL
```

Just a factory function declaration — all that other code needs in order
to *use* the pass. This is what the pass's header,
[`AffineFullUnroll.h`](../../lib/Transform/Affine/AffineFullUnroll.h),
consumes; the entire header is:

***lib/Transform/Affine/AffineFullUnroll.h***
```cpp
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir
```

Note the idiom: `#define` the gate, then `#include` the `.inc`. The
generated file `#undef`s the gate after use, so several gates can be
pulled from the same file in sequence. Note also that the `.inc` is
included *inside* `namespace mlir::tutorial` — the generated code
deliberately contains no namespace of its own, and lands in whatever
namespace you include it into. (Get this wrong and you'll enjoy some of
section 8.)

### `GEN_PASS_DEF_*` — the base class

This is the replacement for Chapter 10's hand-written `PassWrapper` class,
consumed at the top of
[`AffineFullUnroll.cpp`](../../lib/Transform/Affine/AffineFullUnroll.cpp)
with `#define GEN_PASS_DEF_AFFINEFULLUNROLL`:

```cpp
namespace impl {

template <typename DerivedT>
class AffineFullUnrollBase : public ::mlir::OperationPass<> {
public:
  using Base = AffineFullUnrollBase;

  AffineFullUnrollBase() : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}

  /// Returns the command-line argument attached to this pass.
  ::llvm::StringRef getArgument() const override { return "affine-full-unroll"; }

  ::llvm::StringRef getDescription() const override { return R"PD(Fully unroll all affine loops)PD"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>();
  }

private:
  friend std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
    return std::make_unique<DerivedT>();
  }
};
} // namespace impl
```

Everything from Chapter 10's hand-written class is here, written for you.
Piece by piece:

- **Each `Passes.td` field maps to one method:** the `Pass<"...">` flag
  became `getArgument()`, `summary` became `getDescription()`, and
  `dependentDialects` became `getDependentDialects()` (there's the lazy
  dialect loading from section 2).
- **You get things the hand-written version skipped:** `classof` enables
  MLIR's `isa`/`dyn_cast` on passes via a per-class `TypeID`, and
  `clonePass` is required for the pass manager to run pass instances on
  multiple functions safely — the boilerplate `PassWrapper` used to hide,
  now merely *generated* instead of hidden. (The untrimmed file goes
  further still — a copy constructor, a `getName()` for diagnostics, an
  explicit `TypeID` declaration — read it in your build tree.)
- **The base is
  [`::mlir::OperationPass<>`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Pass/Pass.h)**
  — empty template argument, the generic anchor from section 2. (That
  header holds the `Pass`/`OperationPass` definitions and
  `runOnOperation`'s declaration — the authoritative reference for the
  pass API the generated code plugs into.)
- **The CRTP from Chapter 10 is still here** (`template <typename
  DerivedT>` + `std::make_unique<DerivedT>()` in the friend factory): the
  base class must construct and clone *your* derived class, which it can
  only name via the template parameter.

What's left for the human is exactly the part Chapter 10 taught:

***lib/Transform/Affine/AffineFullUnroll.cpp*** (excerpt)
```cpp
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() { ... }
};
```

### `GEN_PASS_REGISTRATION` — the hookup

The third gate holds the registration code: small inline functions, one
per pass plus one umbrella for the whole group, that hand the pass
factory to MLIR's pass registry:

```cpp
#ifdef GEN_PASS_REGISTRATION
// Generate registrations for all passes.
#define GEN_PASS_REGISTRATION_AFFINEFULLUNROLL
#define GEN_PASS_REGISTRATION_AFFINEFULLUNROLLPATTERNREWRITE
#endif // GEN_PASS_REGISTRATION

#ifdef GEN_PASS_REGISTRATION_AFFINEFULLUNROLL

inline void registerAffineFullUnroll() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAffineFullUnroll();
  });
}
// ... registerAffineFullUnrollPatternRewrite() has the same shape ...

inline void registerAffinePasses() {
  registerAffineFullUnroll();
  registerAffineFullUnrollPatternRewrite();
}
```

This is consumed by
[`Passes.h`](../../lib/Transform/Affine/Passes.h) (the "whole group"
header), and `registerAffinePasses()` is precisely the function
[`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp) calls — mystery
resolved. Each `register*` wraps the same `createAffineFullUnroll()`
factory in `mlir::registerPass`, which is what makes the
`--affine-full-unroll` flag exist:

```bash
build/tools/tutorial-opt --help | grep -E 'affine-full-unroll|mul-to-add'
```

```
      --affine-full-unroll             -   Fully unroll all affine loops
      --affine-full-unroll-rewrite     -   Fully unroll all affine loops using the pattern rewrite engine
      --mul-to-add                     -   Convert multiplications to repeated additions
      --mul-to-add-pdll                -   Convert multiplications to repeated additions using pdll
```

(The generated file also contains `registerAffineFullUnrollPass()`-style
variants marked "Old registration code, kept for temporary backwards
compatibility" — generated code has legacy baggage too.)

One deliberate omission: the repo's *second* generated file,
`build/lib/Transform/Arith/Passes.h.inc`, gets no walkthrough — it is
the same backend applied to the same kind of records, so it has exactly
the shape you just read with the names swapped (`GEN_PASS_DECL_MULTOADD`,
`impl::MulToAddBase`, `registerArithPasses()`), once per pass in
[`lib/Transform/Arith/Passes.td`](../../lib/Transform/Arith/Passes.td).
Verifying that yourself — section 3's command with `-name Arith` — is a
better use of it than a second walkthrough, and its `.td` holds the one
genuinely new wrinkle (`MulToAddPdll`'s non-empty `dependentDialects`),
which exercise 4 takes up.

## 5. The migration, before and after

Put Chapter 10's hand-written class next to what this repo keeps on disk,
and the trade is clear.

Before — the preserved
[`ch10-first-pass-cpp/code/AffineFullUnroll.h`](../ch10-first-pass-cpp/code/AffineFullUnroll.h),
~15 lines of class *per pass*, plus a manual registration line per pass
in the driver
([`code/tutorial-opt.cpp`](../ch10-first-pass-cpp/code/tutorial-opt.cpp)):

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
// ... and in tutorial-opt.cpp:
mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
```

After — the current
[`lib/Transform/Affine/`](../../lib/Transform/Affine/): a 6-line `def` in
`Passes.td`, a header that is nothing but the `GEN_PASS_DECL` idiom, and
a `.cpp` whose class definition shrinks to a 2-line frame around
`runOnOperation`:

```cpp
#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

// A pass that manually walks the IR
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() {
    getOperation()->walk([&](AffineForOp op) {
      if (failed(loopUnrollFull(op))) {
        op.emitError("unrolling failed");
        signalPassFailure();
      }
    });
  }
};
```

plus one `registerAffinePasses()` call in `tutorial-opt.cpp` that never
changes as the group grows. Diff-worthy details of the trade:

- `getArgument`/`getDescription` moved into `Passes.td` (as the `Pass<>`
  template argument and `summary`); `PassRegistration<...>` per pass
  became one generated `registerAffinePasses()` per group.
- The `runOnOperation` *body* is character-for-character the same walk;
  only `getOperation().walk` became `getOperation()->walk`, because the
  tablegen'd pass is a generic `OperationPass<>` rather than
  `OperationPass<FuncOp>` (section 2's anchor note).
- The same holds for the other two passes: diff
  [`code/AffineFullUnrollPatternRewrite.cpp`](../ch10-first-pass-cpp/code/AffineFullUnrollPatternRewrite.cpp)
  against
  [its `lib/` twin](../../lib/Transform/Affine/AffineFullUnrollPatternRewrite.cpp),
  and [`code/MulToAdd.cpp`](../ch10-first-pass-cpp/code/MulToAdd.cpp)
  against [its](../../lib/Transform/Arith/MulToAdd.cpp) — the
  `OpRewritePattern`s and the
  `applyPatternsGreedily(getOperation(), std::move(patterns))` driver
  call are untouched; only the class frame around them changed.
- The `dependentDialects` declaration is *new* — the hand-written
  versions never declared it and got away with it only because the input
  IR already forced the dialects to load.

The behavior is identical, and because the ch10 snapshot builds as its
own binary, that's checkable rather than takeable on faith:

```bash
diff <(build/part2-writing-passes/ch10-first-pass-cpp/code/tutorial-opt-ch10 \
        test/part2/affine_loop_unroll.mlir --affine-full-unroll) \
     <(build/tools/tutorial-opt \
        test/part2/affine_loop_unroll.mlir --affine-full-unroll) \
  && echo IDENTICAL-OUTPUT
```

```
IDENTICAL-OUTPUT
```

and the part 2 test suite (which exercises the tablegen'd passes through
`tutorial-opt`) still passes:

```bash
source .venv/bin/activate && lit build/test --filter=part2 -v
```

```
PASS: MLIRUnifiedTutorial :: part2/mul_to_add.mlir (5 of 6)
PASS: MLIRUnifiedTutorial :: part2/affine_loop_unroll.mlir (6 of 6)
...
  Passed  :  6 (35.29%)
```

Chapter 10's "add a pass" checklist now has a tablegen flavor: add a `def`
to `Passes.td`, write the struct with `runOnOperation` in a `.cpp`, add
the file to the `add_mlir_library` in that directory's `CMakeLists.txt` —
and registration is already handled if the group is registered.

## 6. Step: watch a change flow through

To feel the "see-through generator" property, make a fresh pass
definition and run tablegen on it — no build system, no C++, just text in
and text out:

```bash
cat > /tmp/extra.td <<'EOF'
include "mlir/Pass/PassBase.td"
def AffineCountLoops : Pass<"affine-count-loops"> {
  let summary = "Count affine loops";
}
EOF
mlir-tblgen -gen-pass-decls -name Affine \
  /tmp/extra.td -I $WHEEL_ROOT/include | grep AFFINECOUNTLOOPS
```

```
#define GEN_PASS_DECL_AFFINECOUNTLOOPS
#ifdef GEN_PASS_DECL_AFFINECOUNTLOOPS
#undef GEN_PASS_DECL_AFFINECOUNTLOOPS
#endif // GEN_PASS_DECL_AFFINECOUNTLOOPS
#ifdef GEN_PASS_DEF_AFFINECOUNTLOOPS
#undef GEN_PASS_DEF_AFFINECOUNTLOOPS
#endif // GEN_PASS_DEF_AFFINECOUNTLOOPS
#define GEN_PASS_REGISTRATION_AFFINECOUNTLOOPS
#ifdef GEN_PASS_REGISTRATION_AFFINECOUNTLOOPS
#undef GEN_PASS_REGISTRATION_AFFINECOUNTLOOPS
#endif // GEN_PASS_REGISTRATION_AFFINECOUNTLOOPS
```

A new `def` materializes a new family of gates, a new base class, a new
factory, and a new entry inside `registerAffinePasses()` — all from four
lines of tablegen.

## 7. What else the pass generator gives you

Two features beyond what this repo's passes use, worth knowing exist:

- **Documentation generation.** The `add_mlir_doc(... -gen-pass-doc)`
  line from section 3's CMake excerpt renders the `summary`/`description`
  fields to markdown. Run that backend by hand on our `Passes.td`:

  ```bash
  mlir-tblgen -gen-pass-doc lib/Transform/Affine/Passes.td -I $WHEEL_ROOT/include
  ```

  ```markdown
  ### `-affine-full-unroll`

  _Fully unroll all affine loops_

  Fully unroll all affine loops.

  ### `-affine-full-unroll-rewrite`
  ...
  ```

  Upstream MLIR's entire [passes documentation
  page](https://mlir.llvm.org/docs/Passes/) is generated this way — the
  `description` field you write *is* the documentation.
- **Pass options and statistics.** A `def` can declare
  `let options = [Option<...>]` to generate CLI-configurable pass
  parameters (upstream passes like
  `convert-math-to-funcs{convert-ctlz}` get their sub-flags this way),
  and `let statistics = [...]` for counters reported with
  `--mlir-pass-statistics`. There is also `let constructor = ...` to
  supply a custom factory. `PassBase.td` documents all of these — in the
  source-as-documentation sense from section 2.

## 8. When tablegen goes wrong

The failure modes are where the intro's love-hate relationship earned its
"hate", so let's provoke two, for calibration.

Misspell a field in a copy of the `.td` (`let summry = ...`):

```
broken.td:7:7: error: Value 'summry' unknown!
  let summry = "Fully unroll all affine loops";
      ^
```

That one is fine — tablegen's own parser has decent errors. The real pain
lives on the *C++ side* of the boundary. Forget to implement
`runOnOperation` in your struct and the compiler says (trimmed):

```
unique_ptr.h:767:30: error: allocating an object of abstract class type
      'mlir::tutorial::AffineFullUnroll'
mlir/Pass/Pass.h:179:16: note: unimplemented pure virtual method
      'runOnOperation' in 'AffineFullUnroll'
```

Modern clang does name the missing method (older toolchains were far
less helpful, and earned tablegen much of its grumbling) — but notice
*where* the error points: inside `unique_ptr.h`, reached through the
generated friend factory, two layers away from anything you wrote. This
is the norm for tablegen mistakes: wrong namespace around the
`#include`, a missing `GEN_PASS_DEF` gate, a `dependentDialects` typo —
each surfaces as C++ errors *in or through generated code*. The survival
skill is always the same: open the generated `.inc` in your build tree
and read what is actually there. It's short, it's commented, and it's the
ground truth.

## Where to go next

Passes were tablegen's warm-up act. In
[Chapter 12: Defining the Poly Dialect](../../part3-building-a-dialect/ch12-defining-poly/README.md),
tablegen defines an entire **dialect** — the `poly` dialect for
polynomial arithmetic, with custom types and ops
([`lib/Dialect/Poly/`](../../lib/Dialect/Poly/)) — and the generated code
grows from one base class to whole op definitions with verifiers,
parsers, and printers. The read-the-generated-code habit from section 8
becomes essential there.

**Exercises**

1. Do section 6's experiment, then extend it: give `AffineCountLoops` a
   `description` and `dependentDialects`, regenerate, and find each
   field's landing spot in the output.
2. Delete the `include "mlir/Pass/PassBase.td"` line from a copy of
   `Passes.td` and regenerate. The error —
   `error: Couldn't find class 'Pass'` — is your reminder that `Pass` is
   not a keyword, just a record class defined in the upstream `.td`.
3. Run section 3's `diff` for the Arith group: generate with
   `-name Arith` from `lib/Transform/Arith/Passes.td` and compare against
   `build/lib/Transform/Arith/Passes.h.inc`.
4. Read [`lib/Transform/Arith/Passes.td`](../../lib/Transform/Arith/Passes.td)
   and explain why `MulToAddPdll` declares `dependentDialects` (the PDL
   and PDLInterp dialects) while `MulToAdd` doesn't — section 2's rule
   answers it. (Hint: which pass creates ops from dialects that might not
   be loaded yet? `MulToAdd` only creates `arith` ops, and its input
   necessarily contains `arith` ops already.)
5. In `PassBase.td` (`$WHEEL_ROOT/include/mlir/Pass/PassBase.td`), find
   the `Option` class and sketch — on paper only — what adding an
   `unroll-factor` option to `AffineFullUnroll` would look like. Part 3's
   dialect work will make you write such things for real.
