# Chapter 17: Patterns with PDLL

> Reworked from the mlir-tutorial fork's chapter 13 for the unified repo.

This chapter doubles as a language tutorial built on the official
[PDLL documentation](https://mlir.llvm.org/docs/PDLL/): section 3
teaches PDLL from scratch, hands-on, before section 4 applies it to this
repo. It introduces a *third* way to write rewrite patterns, after C++
([Chapter 10](../../part2-writing-passes/ch10-first-pass-cpp/README.md))
and DRR ([Chapter 16](../ch16-canonicalizers-drr/README.md)), applied to
an old friend — Chapter 10's mul-to-add transformation, reborn as
`--mul-to-add-pdll`. PDLL is worth a whole chapter not because the
syntax is nicer (though it is), but because of a genuinely different
*execution model*: PDLL patterns compile to **IR** — the `pdl` dialect —
and are *interpreted* at runtime by a bytecode engine. Patterns become
data. We will watch every stage of that story in real output.

> **Tooling note:** all tools in this chapter (`mlir-pdll`, `mlir-opt`)
> come from the prebuilt MLIR wheel installed in the repo's virtual
> environment — no LLVM source build required. Follow the setup in the
> [repo root README](../../README.md) first. All commands below are run
> from the repo root after:
>
> ```sh
> source .venv/bin/activate
> export MLIR_ROOT=$(python -m mlir_wheel --root-dir)
> export PATH=$MLIR_ROOT/bin:$PATH
> ```

## What you will learn

- Why a third pattern language exists (what DRR structurally can't say).
- The PDLL language, built up step by step with runnable examples:
  patterns, variables, operation expressions, constraints, rewrites.
- The runtime story: PDLL → `pdl` IR → `pdl_interp` matcher automaton →
  bytecode — all made visible with `mlir-pdll` and `mlir-opt`.
- The repo's `MulToAdd.pdll` line by line, its native C++ constraints,
  and the CMake build wiring.
- How to choose among C++, DRR, and PDLL — and PDLL's current limits.

## Prerequisites

[Chapter 16](../ch16-canonicalizers-drr/README.md) (DRR — the contrast
makes both languages clearer) and
[Chapter 10](../../part2-writing-passes/ch10-first-pass-cpp/README.md)
(the original C++ `MulToAdd`, which you should have fresh in mind).

---

## 1. Why a third pattern language?

Chapter 16 ended with DRR looking pretty good — so why did the MLIR
community build a replacement (and put DRR into maintenance mode)?
Because DRR inherited tablegen's DAG-expression syntax, and several
ordinary MLIR shapes don't fit a tablegen DAG:

- **Multi-result operations.** DRR binds all results to one name with a
  `__N` suffix convention; PDLL writes `threeResultOp.output1` (or `.0`).
- **Variadic operands and result groups** — awkward to impossible in DRR.
- **Regions** — ops with bodies can't be matched in DRR at all.
- **Arithmetic on static values** — Chapter 10's `value / 2` had no DRR
  spelling whatsoever; recall Chapter 16's `CPred` escape hatch was
  string-pasted C++.
- **Readability at scale** — DRR constraints live in a list far from the
  thing they constrain; PDLL attaches them "directly on or next to the
  entities they apply to."

The designers explicitly considered embedding pattern-writing in Python
and rejected it (dependency and tooling consistency); PDLL is a
purpose-built language with its own compiler, `mlir-pdll`.

### Patterns as IR

The deeper novelty is *how patterns run*. Compare the three generations:

| | authored as | becomes | executed as |
|---|---|---|---|
| C++ (Ch. 10) | `matchAndRewrite` | compiled object code | native code |
| DRR (Ch. 16) | tablegen DAGs | *generated* C++ | native code |
| PDLL | `.pdll` source | **`pdl` dialect IR** | **interpreted bytecode** |

PDLL patterns compile to operations in the
[`pdl` dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/) (Pattern
Descriptor Language — PDLL humorously stands for Pattern Descriptor
Language *Language*) — MLIR IR that *describes matching and rewriting
MLIR IR* (`pdl.operation`, `pdl.replace` model "an operation", "a
replacement"). At pass-build time that IR is lowered (like any other
dialect!) to
[`pdl_interp`](https://mlir.llvm.org/docs/Dialects/PDLInterpOps/), a
dialect of primitive match instructions, then to a compact bytecode that
a small interpreter executes inside the greedy driver. Only *native
constraints* (section 3.7) remain as C++.

Why interpret patterns instead of compiling them? Three arguments:

- **Extensibility**: patterns-as-data can be loaded at *runtime* — a user
  can hand your compiler new rewrite rules without rebuilding it.
- **Size**: PDL bytecode is roughly 10× smaller than the equivalent
  compiled C++ pattern object code.
- **Speed, surprisingly**: all patterns merge into one decision automaton
  (you'll *see* this in section 6), sharing common checks across
  patterns — per a 2019 MLIR talk (linked from
  [this discussion](https://discourse.llvm.org/t/what-is-the-benefit-of-interpreting-pdl/80331/1)),
  constructing the merged matcher is ~15× faster than compiling the same
  patterns as C++, and matching itself is claimed competitive or better.

## 2. Your lab bench: the `mlir-pdll` tool

Everything in section 3 is meant to be *run*, and the whole loop needs
one tool. `mlir-pdll` ships in the wheel, next to
[Chapter 11](../../part2-writing-passes/ch11-tablegen-passes/README.md)'s
`mlir-tblgen` (both live in `$MLIR_ROOT/bin`). Its interface:

```sh
mlir-pdll -x=mlir -I $MLIR_ROOT/include yourfile.pdll
```

- `-I` points at the directory containing `mlir/Dialect/...` tablegen
  files, so your `.td` includes resolve — same idea as Chapter 11's
  include path. The wheel installs the full MLIR include tree, so
  `$MLIR_ROOT/include` is all you need.
- `-x=` selects the output: **`mlir`** (the compiled `pdl` IR — the best
  way to *check* a pattern while learning), **`cpp`** (the generated C++
  used in real builds, section 7), or **`ast`** (the parsed syntax tree,
  for debugging "why doesn't this mean what I wrote").
- One cosmetic wrinkle: `-x=mlir` decorates every line with `loc(...)`
  source locations (useful — they travel all the way into pass
  diagnostics). To read the IR without the noise, pipe through
  `mlir-opt`, which drops locations by default; the examples below all
  do `mlir-pdll -x=mlir ... | mlir-opt`.

Two tooling notes for later: when a pattern doesn't mean what you
wrote, `-x=ast` shows the parse — variables, constraints, inline code
blocks — in tree form; and PDLL has its own language server,
`mlir-pdll-lsp-server` (op-name completion straight out of included
ODS), though the wheel ships `mlir-pdll` only — the LSP needs a full
LLVM distribution.

Put each snippet below in a scratch `.pdll` file, run it, and read what
comes out. All outputs shown are real, produced with the repo's wheel
(LLVM 24) — generated-code details vary slightly by LLVM version, and
where the official docs disagree with observed behavior (e.g. printed
default benefits), the outputs here are what the tools actually printed.

## 3. The PDLL language, step by step

### 3.1 The smallest possible pattern

A `.pdll` file is includes followed by definitions. Here is a complete,
compilable file — one anonymous pattern that erases every
`arith.constant` (not a *useful* rewrite; a minimal one):

```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Pattern => erase op<arith.constant>;
```

Compile it (`mlir-pdll -x=mlir -I $MLIR_ROOT/include ... | mlir-opt`)
and PDLL shows you what it understood (verified):

```mlir
pdl.pattern : benefit(0) {
  %0 = operands
  %1 = types
  %2 = operation "arith.constant"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
  rewrite %2 {
    erase %2
  }
}
```

Already visible: the pattern *is IR*; the match section binds
placeholders (`pdl.operands`, `pdl.types`, an `arith.constant`
operation); the rewrite region does the mutation. The `Pattern => ...`
form is the single-statement lambda shorthand; we'll use full bodies from
here on.

### 3.2 Includes: how PDLL knows your ops

That first line matters more than it looks: including a **tablegen**
file imports its ODS wholesale — every op, attribute constraint, and
interface defined there becomes available to PDLL *by name*.
Chapters 11–12's op definitions are a machine-readable contract, and PDLL
is a consumer. You can watch the import happen: run `-x=ast` on the file
and the dump *begins* with dozens of `UserConstraintDecl`s
(`APIntAttr`, ...) synthesized from arith's ODS (verified). The other
include form,

```pdll
#include "other.pdll"
```

is plain textual inclusion of more PDLL, for sharing constraint libraries
across pattern files.

### 3.3 Pattern anatomy and metadata

The general form:

```pdll
Pattern SomeName with benefit(10), recursion {
  // --- match section ---
  ...statements describing the input IR...

  // --- rewrite section ---
  replace ... with ...;   // or: erase ...;  or: rewrite ... with { ... };
}
```

- The name is optional (3.1's pattern had none) but names show up in the
  generated `pdl.pattern @Name` and in diagnostics — use them.
- The body splits at the first
  **[operation rewrite statement](https://mlir.llvm.org/docs/PDLL/#operation-rewrite-statements)**
  (`replace` / `erase` / `rewrite ... with {}`), which must be the last
  statement: everything before it matches, everything inside it
  rewrites. Chapter 10's two-phase discipline (inspect, *then* mutate),
  enforced grammatically.
- **`benefit(N)`** — the pattern priority from Chapters 10 and 16. If
  omitted, it defaults to the number of ops matched (deeper wins ties).
- **`with recursion`** — opt-in for patterns that can safely apply to
  their own output; without it the driver guards against infinite
  self-application.

### 3.4 Variables and the core types

Every PDLL variable is typed by the kind of IR entity it stands for,
mirroring the C++ handle types you already know: `Attr`
(`mlir::Attribute`, Chapter 14's fold values), `Op` or `Op<arith.muli>`
(`mlir::Operation*` / that op's generated class), `Type`/`TypeRange`,
and `Value`/`ValueRange` (Chapter 10's daily bread).

Variables get **bound** — attached to a piece of the matched IR — either
positionally inside a match expression (`x` in `op<...>(x: Value)` means
"whatever the first operand is, call it x") or by assignment
(`let halved: Attr = Halve(const);`). Three refinements:

```pdll
let value: Value;                  // declare, bind later
let value: [Value, HasOneUse];     // declare with a constraint list
let input: Value<someType>;        // constrain the value's MLIR type
_: Value                           // wildcard: match, bind nothing
```

That constraint-list form is worth savoring after Chapter 16:
`HasOneUse` — a `CPred` C++ string in a separate list in DRR — sits *on
the variable declaration itself*.

### 3.5 Operation expressions

The op expression mirrors the *generic form* of an operation from
[Chapter 1](../../part1-using-mlir/ch01-why-mlir-and-reading-ir/README.md)
— which you now get to write by hand:

```pdll
let root = op<my_dialect.foo>(operands: ValueRange)
    {attr1 = someAttr: Attr} -> (resultTypes: TypeRange);
```

Piece by piece:

- **Name** in angle brackets. In a match section you may omit it —
  `op<>` matches *any* operation; in a rewrite section the name is
  mandatory (you can't create "some op").
- **Operands** in parens: one `ValueRange` for everything, or individual
  `Value`s/`ValueRange`s lining up with the op's ODS operand groups.
- **Attributes** in braces, `{name = var: Attr}` — matching an attribute
  binds its *payload*, exactly how Chapter 14's folders picked apart
  constant attributes.
- **Result types** after `->`. **Results** of a bound op are accessed as
  `someOp.resultName` (ODS names) or `someOp.0`; and an `Op` variable
  auto-converts to its result `Value` where a value is expected. That
  conversion is what makes *nesting* work: in

  ```pdll
  op<arith.muli>(op<arith.constant> {value = c: Attr}, rhs: Value)
  ```

  the inner op expression stands for "the result of an arith.constant" —
  compare Chapter 10's `getDefiningOp<ConstantIntOp>()` dance: the
  backward walk through the SSA graph became *syntax*.

### 3.6 Step: grow a real pattern

Time to build one incrementally, the way you actually would. Goal:
eliminate `x + 0`. The naive version —
`let root = op<arith.addi>(x: Value, op<arith.constant>); replace root
with x;` — compiles (verified) and is wrong: it rewrites `x + 7` to `x`.
We matched the *shape* but not the *content*. To inspect content we bind
the constant's payload attribute and constrain it — and "is this
attribute zero?" is a question about a C++ object, so we reach for a
**native constraint** (full taxonomy next section):

```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Constraint IsZero(attr: Attr) [{
  return success(cast<::mlir::IntegerAttr>(attr).getValue().isZero());
}];

Pattern EliminateAddZero {
  let root = op<arith.addi>(x: Value, op<arith.constant> {value = zeroAttr: Attr});
  IsZero(zeroAttr);
  replace root with x;
}
```

Verified: this compiles to a `pdl.pattern` whose match section carries
`apply_native_constraint "IsZero"` — the C++ block will run during
matching, and a `failure()` from it simply means "this pattern doesn't
apply here" (Chapter 10's `return failure()`, relocated).

One more feature, discovered by trying (the docs are quiet on it, the
compiler is not): **repetition means equality**. Match `x − x` by
binding the same variable twice, and rewrite it to a constant zero using
an **attribute literal** — `attr<"...">` parses any textual MLIR
attribute:

```pdll
Pattern ZeroSelfSub {
  let root = op<arith.subi>(x: Value, x);
  rewrite root with {
    let zero = op<arith.constant> {value = attr<"0 : i32">};
    replace root with zero;
  };
}
```

Verified — the generated PDL matches `"arith.subi"(%0, %0)`: one
placeholder used twice, equality by construction. (DRR had the same
trick in Chapter 16's `(Poly_MulOp $x, $x)`; now you know both
spellings. Also note this pattern hardcodes `i32` — handling any integer
type needs the result type queried from `x`, a good exercise.)

### 3.7 Constraints: the full menu

Three species, in increasing order of C++ involvement (the
[constraints section](https://mlir.llvm.org/docs/PDLL/#constraints-1) of
the PDLL docs has the full details of what can be constrained):

1. **PDLL-defined** — written in the language, composing match logic,
   optionally returning values (and tuples, with named elements accessed
   as `.0` or `.name`):

   ```pdll
   Constraint UsedByFooOp(value: Value) {
     op<my_dialect.foo>(value);
   }
   ```

2. **Native, inline** — C++ in a `[{ ... }]` block (3.6's `IsZero`),
   compiled into the generated code; `rewriter` is in scope and the
   return type is implicitly `LogicalResult`. Chapter 16's `CPred`
   comparison: same escape hatch, but a *typed function* instead of a
   string pasted into a condition.

3. **Native, external** — declaration only; implementation registered at
   pass construction:

   ```pdll
   Constraint HasOneUse(value: Value);
   Constraint Halve(attr: Attr) -> Attr;   // constraints may return values
   ```

   ```cpp
   patterns.getPDLPatterns().registerConstraintFunction("HasOneUse", impl);
   ```

   The implementation can use a friendly typed signature
   (`LogicalResult f(PatternRewriter&, Value)`) or the interpreter's
   generic one (`LogicalResult f(PatternRewriter&, PDLResultList&,
   ArrayRef<PDLValue>)`), where `PDLValue` is a dynamically-typed box you
   `.cast<Attribute>()` out of — a real one appears in section 5.

Apply single-entity constraints in a variable's bracket list, and
multi-entity or value-returning ones with call syntax:

```pdll
let v: [Value, HasOneUse];   // single-entity constraints
IsZero(zeroAttr);            // multi-entity constraint
let halved = Halve(const);   // value-returning constraint
```

### 3.8 Rewrites: the full menu

Symmetric story on the other side of the boundary. Inside patterns,
three statements:

```pdll
replace root with x;              // rewire uses, delete root
replace root with (a, b);         // multi-result replacement (tuple)
erase root;                       // just delete
rewrite root with { ... };        // a block of rewrite steps
```

In a `rewrite ... with {}` block, nothing is implicit — build ops with
`let`, and finish with a `replace`/`erase` (3.6's `ZeroSelfSub` did
exactly this). And mirroring constraints, there are function-like
**`Rewrite`** definitions in the same three flavors — PDLL-defined,
native-inline `[{ }]`, native-external via `registerRewriteFunction` —
with tuple returns and a lambda form:

```pdll
Rewrite Double(v: Value) -> Op {
  return op<arith.addi>(v, v);
}
Rewrite EraseOp(op: Op) => erase op;
```

(Verified: a pattern calling `Double(x)` inside its rewrite block
compiles, the helper inlined into the generated PDL.) One rule ties the
section together: **Constraints run while matching and may fail;
Rewrites run after the match is committed and must succeed** — and a
`Rewrite` can only be *called* from the rewrite section; the compiler
enforces the phase discipline both ways.

---

## 4. The repo's patterns: MulToAdd in PDLL

Now the destination — and with section 3 behind you,
[`lib/Transform/Arith/MulToAdd.pdll`](../../lib/Transform/Arith/MulToAdd.pdll)
reads like review. The transformation is Chapter 10's: expand
power-of-two multiplications by halving, peel everything else. The core
(one variant of each pattern; the file has both — see below):

***lib/Transform/Arith/MulToAdd.pdll*** (excerpt)
```pdll
#include "mlir/Dialect/Arith/IR/ArithOps.td"

Constraint IsPowerOfTwo(attr: Attr) [{
  int64_t value = cast<::mlir::IntegerAttr>(attr).getValue().getSExtValue();
  return success((value & (value - 1)) == 0);
}];

// Currently, constraints that return values must be defined in C++
Constraint Halve(atttr: Attr) -> Attr;
Constraint MinusOne(attr: Attr) -> Attr;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
Pattern PowerOfTwoExpandRhs with benefit(2) {
  let root = op<arith.muli>(op<arith.constant> {value = const: Attr}, rhs: Value);
  IsPowerOfTwo(const);
  let halved: Attr = Halve(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = halved};
    let newMul = op<arith.muli>(newConst, rhs);
    let newAdd = op<arith.addi>(newMul, newMul);
    replace root with newAdd;
  };
}

// Replace y = 9*x with y = 8*x + x
Pattern PeelFromMulRhs with benefit(1) {
  let root = op<arith.muli>(lhs: Value, op<arith.constant> {value = const: Attr});

  // We are guaranteed `value` is not a power of two, because the greedy
  // rewrite engine ensures the PowerOfTwoExpand pattern is run first, since
  // it has higher benefit.
  let minusOne: Attr = MinusOne(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = minusOne};
    let newMul = op<arith.muli>(lhs, newConst);
    let newAdd = op<arith.addi>(newMul, lhs);
    replace root with newAdd;
  };
}
```

(Plus `PowerOfTwoExpandLhs` and `PeelFromMulLhs` twins in the file.)
You can name every construct now: an ODS include (3.2); a native inline
constraint, `IsPowerOfTwo` (3.7 #2); two external constraints *with
results*, `Halve` and `MinusOne` (3.7 #3 — and there's PDLL's sharpest
current limit: *arithmetic on static values has no in-language
spelling*, so `value / 2` must live in C++; adding arithmetic, boolean
logic, and comparisons
[has an RFC](https://discourse.llvm.org/t/rfc-add-arithmetic-logical-and-comparison-expressions-into-pdll/78251));
nested op matching with an attribute bind (3.5); benefits dividing labor
exactly as in Chapter 10 (halving at 2 beats peeling at 1); and
`rewrite ... with {}` blocks building the replacement (3.8). Hold it
against Chapter 10's 35-line `matchAndRewrite`: the match is the *shape*,
no `getDefiningOp`, no null checks, constraints beside the things they
constrain.

Two things deserve stories:

**Why `Rhs` and `Lhs` variants of everything?** Chapter 10's C++ checked
only one operand, on the stated assumption that canonicalization moves
constants rightward. The original author got bitten here: that
normalization is a *canonicalization pattern* of `arith.muli`, and this
pass's driver only runs *these* patterns — nobody normalizes on its
behalf. The PDLL file owns the problem by spelling out both operand
orders. (A [Chapter 13](../ch13-traits/README.md) aside: this is what
`Commutative`-aware matching would obviate.)

**An easter egg proving 3.7's point.** The repo's `Halve` declaration
misspells its parameter `atttr` — three t's — and nothing cares: for
external constraints only the *types* travel (via the documented
[PDLL-to-C++ type mapping](https://mlir.llvm.org/docs/PDLL/#native-constraint-type-translations));
parameter names are documentation. (Verified: fixing the typo and
regenerating produces identical output, modulo source locations.)

## 5. The pass: parsing patterns at runtime

The C++ side
([`lib/Transform/Arith/MulToAddPdll.cpp`](../../lib/Transform/Arith/MulToAddPdll.cpp))
has two jobs. First, the external constraint implementations — here is
`Halve`, in the generic `PDLValue` signature from 3.7:

***lib/Transform/Arith/MulToAddPdll.cpp*** (excerpt)
```cpp
LogicalResult halveImpl(PatternRewriter &rewriter, PDLResultList &results,
                        ArrayRef<PDLValue> args) {
  Attribute attr = args[0].cast<Attribute>();
  IntegerAttr cAttr = cast<IntegerAttr>(attr);
  int64_t value = cAttr.getValue().getSExtValue();
  results.push_back(rewriter.getIntegerAttr(cAttr.getType(), value / 2));
  return success();
}

void registerNativeConstraints(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("Halve", halveImpl);
  patterns.getPDLPatterns().registerConstraintFunction("MinusOne", minusOneImpl);
}
```

Read it in execution order: the interpreter hands the constraint its
arguments as `PDLValue` boxes, so `args[0].cast<Attribute>()` unboxes
the attribute (3.7's dynamically-typed box, met for real); ordinary C++
extracts the integer and builds the halved attribute; and because
`Halve` *returns a value*, the result travels back by pushing onto
`results`, with `success()` meaning the constraint held.
`registerNativeConstraints` ties the `.pdll` declarations to these
implementations by name.

Second, the pass body, near-identical to Chapter 10's:

***lib/Transform/Arith/MulToAddPdll.cpp*** (excerpt)
```cpp
struct MulToAddPdll : impl::MulToAddPdllBase<MulToAddPdll> {
  using MulToAddPdllBase::MulToAddPdllBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    registerNativeConstraints(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
```

`populateGeneratedPDLLPatterns` comes from the generated
`MulToAddPdll.h.inc` (in `build/lib/Transform/Arith/` after a build).
What did the generator actually emit? Look — this is the real file,
locations and template boilerplate elided:

***build/lib/Transform/Arith/MulToAddPdll.h.inc*** (excerpt)
```cpp
static ::llvm::LogicalResult IsPowerOfTwoPDLFn(::mlir::PatternRewriter &rewriter, ::mlir::Attribute attr) {
  int64_t value = cast<::mlir::IntegerAttr>(attr).getValue().getSExtValue();
  return success((value & (value - 1)) == 0);
}

struct PowerOfTwoExpandRhs : ::mlir::PDLPatternModule {
  PowerOfTwoExpandRhs(::mlir::MLIRContext *context, ...)
    : ::mlir::PDLPatternModule(::mlir::parseSourceString<::mlir::ModuleOp>(
R"mlir(pdl.pattern @PowerOfTwoExpandRhs : benefit(2) {
  ...
    )mlir", context), ...) {
    registerConstraintFunction("IsPowerOfTwo", IsPowerOfTwoPDLFn);
  }
};
```

The "compiled" pattern is a **string of `pdl` IR** embedded in a C++
raw-string literal, parsed when the pass is constructed — with the
*inline* native constraint compiled into a typed C++ function
(`PatternRewriter&, Attribute` — the friendly signature from 3.7) and
registered automatically; only the external ones needed our manual
`registerNativeConstraints`. That's why the pass's tablegen declaration
(Chapter 11 machinery, in
[`lib/Transform/Arith/Passes.td`](../../lib/Transform/Arith/Passes.td))
lists `dependentDialects = ["mlir::pdl::PDLDialect",
"mlir::pdl_interp::PDLInterpDialect"]` — the pass *creates `pdl` IR*
(by parsing that string), so those dialects must be loaded before it
runs. Hold on to that rule: every pass must declare the dialects it
creates ops from, and it returns with force in
[Chapter 18](../../part4-lowering/ch18-dialect-conversion/README.md).

## 6. Under the hood: watching a pattern become bytecode

Everything in this section is real output. First, PDLL → PDL — section
2's command on the repo's file, from the repo root:

```sh
mlir-pdll -x=mlir -I $MLIR_ROOT/include lib/Transform/Arith/MulToAdd.pdll | mlir-opt
```

```mlir
pdl.pattern @PowerOfTwoExpandRhs : benefit(2) {
  %1 = attribute
  %3 = operation "arith.constant"(%0 : !pdl.range<value>) {"value" = %1} -> (%2 : !pdl.range<type>)
  %4 = result 0 of %3
  %5 = operand
  %7 = operation "arith.muli"(%4, %5 : !pdl.value, !pdl.value) -> (%6 : !pdl.range<type>)
  apply_native_constraint "IsPowerOfTwo"(%1 : !pdl.attribute)
  %8 = apply_native_constraint "Halve"(%1 : !pdl.attribute) : !pdl.attribute
  rewrite %7 {
    %9 = operation "arith.constant"  {"value" = %8}
    %11 = operation "arith.muli"(%10, %5 : !pdl.value, !pdl.value)
    %14 = operation "arith.addi"(%12, %13 : !pdl.value, !pdl.value)
    replace %7 with %14
  }
}
```

By now you can read this fluently — it's 3.1's toy output at full scale,
with the native constraints as named calls.

Second, PDL → `pdl_interp` — run the standard lowering by appending
`--convert-pdl-to-pdl-interp` to the `mlir-opt` in the pipe:

```mlir
pdl_interp.func @matcher(%arg0: !pdl.operation) {
  pdl_interp.check_operation_name of %arg0 is "arith.muli" -> ^bb2, ^bb1
^bb1:  // 10 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb6, ^bb7, ^bb8, ^bb9, ^bb11, ^bb12
  pdl_interp.finalize
^bb2:  // pred: ^bb0
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb3, ^bb1
  ...
^bb10:  // pred: ^bb9
  pdl_interp.apply_constraint "IsPowerOfTwo"(%4 : !pdl.attribute) -> ^bb13, ^bb11
^bb11:  // 3 preds: ^bb10, ^bb13, ^bb14
  %6 = pdl_interp.apply_constraint "MinusOne"(%4 : !pdl.attribute) : !pdl.attribute -> ^bb12, ^bb1
^bb12:  // pred: ^bb11
  pdl_interp.record_match @rewriters::@PeelFromMulRhs(...) : benefit(1), ...
```

This is the **merged matcher automaton** from section 1's performance
claims, observable: all *four* patterns in the file became a *single*
`@matcher` function — one shared
`check_operation_name ... "arith.muli"` serving every pattern, one
shared failure block (`^bb1`, with ten predecessors!), branching apart
only where the patterns genuinely differ (which operand is the
constant). Common work is done once per candidate op, not once per
pattern — that's what a C++ pattern set, where each `matchAndRewrite`
starts from scratch, structurally cannot do. Note the fallthrough
`^bb10`→`^bb11`: if `IsPowerOfTwo` fails, control falls straight into
trying the peel pattern on the *same* bindings.

Finally, at `FrozenRewritePatternSet` construction (inside the pass),
the `pdl_interp` form is serialized to `PDLByteCode`, and the greedy
driver calls the interpreter's hooks like any other pattern. Five
representations of one idea — `.pdll` text, AST, `pdl` IR, `pdl_interp`
automaton, bytecode — and you've now seen all five.

## 7. Build integration and running it

The CMake wiring
([`lib/Transform/Arith/CMakeLists.txt`](../../lib/Transform/Arith/CMakeLists.txt))
is Chapter 11's tablegen pattern with one twist — a dedicated helper,
`add_mlir_pdll_library`, provided by the wheel's MLIR CMake package,
which swaps the generator binary: under the hood it invokes
`mlir-pdll -x=cpp` instead of `mlir-tblgen`:

***lib/Transform/Arith/CMakeLists.txt*** (excerpt)
```cmake
add_mlir_pdll_library(MulToAddPdllIncGen
  MulToAdd.pdll
  MulToAddPdll.h.inc
)
```

The library target then lists `MulToAddPdllIncGen` under `DEPENDS` so
the header is generated before `MulToAddPdll.cpp` compiles; the header
is included from
[`MulToAddPdll.h`](../../lib/Transform/Arith/MulToAddPdll.h), and the
pass registers through the same `Passes.h` umbrella as Chapter 10's
`--mul-to-add`, so `tutorial-opt` picks it up with no further ceremony.

Run the pass on
[`test/part3/mul_to_add_pdll.mlir`](../../test/part3/mul_to_add_pdll.mlir)
(a clone of Chapter 10's test with the new flag) — verified:

```sh
build/tools/tutorial-opt --mul-to-add-pdll test/part3/mul_to_add_pdll.mlir
```

```mlir
func.func @just_power_of_two(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  %1 = arith.addi %0, %0 : i32
  %2 = arith.addi %1, %1 : i32
  return %2 : i32
}
```

Byte-identical behavior to Chapter 10's `--mul-to-add` — same doubling
chain, same peel for `9*x` — arrived at through an interpreter instead
of compiled pattern code. The lit test locks it in
([Chapter 9](../../part1-using-mlir/ch09-testing-with-lit/README.md)'s
workflow) — verified,
`PASS: MLIRUnifiedTutorial :: part3/mul_to_add_pdll.mlir`:

```sh
source .venv/bin/activate
lit build/test --filter=pdll -v
```

## 8. Choosing among three languages

The decision table:

| | C++ (Ch. 10) | DRR (Ch. 16) | PDLL |
|---|---|---|---|
| expressiveness | everything | DAGs, no regions/variadics/multi-result | rich matching; no static arithmetic (yet), no regions (yet) |
| escape hatch | is the escape hatch | `CPred` strings | typed native constraints/rewrites |
| debugging | debugger-friendly | read generated C++ | `-x=ast`, read `pdl`/`pdl_interp` IR |
| runtime model | compiled | compiled (generated C++) | interpreted bytecode |
| runtime-loadable patterns | no | no | possible by design |
| status | forever | **maintenance mode** | active development |

Known PDLL gaps as of this writing (RFC linked in section 4): no
in-language arithmetic/boolean logic/comparisons (hence `Halve` in C++),
no region support, no dialect-conversion (type-converting) patterns —
and the docs still warn that designs "are not necessarily final." The
practical advice: shapes-of-ops rewrites in PDLL or DRR, anything clever
in C++, and expect to *read* all three in the wild.

## Exercises

1. Do section 3 for real: build `EliminateAddZero` step by step in a
   scratch file, checking each stage with `-x=mlir`. Then extend it with
   an `EliminateAddZeroLhs` twin (section 4 explains why) and a
   `Rewrite`-based `ZeroSelfSub` that works for any integer type (hint:
   a native rewrite building a zero attribute from `x`'s type).
2. The capstone: port Chapter 16's `DifferenceOfSquares` from DRR to
   PDLL on paper. You'll need a `HasOneUse` external constraint, the
   repetition trick from 3.6 for `op<poly.mul>(x: Value, x)`, and a
   `#include` of `PolyOps.td` (mind the `-I` paths: your own tablegen
   lives in this repo, not the wheel). Compare all three spellings of
   this one pattern — C++, DRR, yours — side by side.
3. Run `mlir-pdll -x=ast -I $MLIR_ROOT/include` on `MulToAdd.pdll` and
   find (a) the imported arith ODS constraints at the top, (b) the four
   `PatternDecl`s with their benefits, (c) how the inline `IsPowerOfTwo`
   C++ appears (a `Code<...>` payload) vs the bodiless `Halve`.
4. Count the sharing: how many `check_operation_name` checks exist in
   the `pdl_interp` output for four patterns? (Grep says three.) Which
   block do all failures funnel into? Sketch the automaton if
   `PeelFromMul` matched `arith.subi` instead.
5. Design exercise from section 1's extensibility promise: sketch a
   `--load-patterns=<file.pdl>` flag for `tutorial-opt`. The pass
   already parses `pdl` IR from a string — what changes if the string
   comes from a user's file? What about native constraints the file
   references, and what does that imply about which patterns are safely
   user-loadable?

## Where to go next

You now have three pattern languages and one recurring limitation: none
of them can change *types*. Every rewrite so far replaced ops whose
results had the same types as before. Lowering `poly` out of existence —
turning `!poly.poly<10>` into tensors an actual machine can chew on —
needs a fundamentally different engine:
[Part 4, Chapter 18: Dialect Conversion](../../part4-lowering/ch18-dialect-conversion/README.md)
introduces type converters, conversion targets, and the
`ConversionPattern` machinery that PDLL (section 8's table) doesn't yet
speak.
