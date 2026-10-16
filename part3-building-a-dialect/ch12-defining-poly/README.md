# Chapter 12: Defining the Poly Dialect

> Reworked from the mlir-tutorial fork's chapter 5 for the unified repo.

So far we have consumed dialects other people defined. In this chapter the
training wheels come off: we define **`poly`**, a dialect for polynomial
arithmetic, with its own type (`!poly.poly<10>`) and operations
(`poly.add`, `poly.mul`, `poly.eval`, ...). This dialect is Part 3's and
Part 4's protagonist — every remaining chapter of those parts builds on it.

One reading note up front: the repo's `poly` files have accumulated
machinery from Chapters 13–16 (traits, folders, verifiers, canonicalizers).
This chapter walks the *current* files but clearly marks each
forward-reference — you'll know exactly which `let` lines to ignore until
their chapter comes.

## What you will learn

- What it takes to define a dialect: the dialect shell, custom types,
  custom ops — all in tablegen — plus a thin layer of C++.
- How ODS ("operation definition spec") describes ops: `arguments`,
  `results`, type constraints, and `assemblyFormat`.
- How MLIR types carry compile-time parameters (the `10` in
  `!poly.poly<10>`), and what type *uniquing* means.
- How the generated classes look and how the dialect registers itself into
  `tutorial-opt`.
- How to test syntax round-tripping with lit and FileCheck.

## Prerequisites

- Part 2 — especially
  [Chapter 11](../../part2-writing-passes/ch11-tablegen-passes/README.md):
  dialect definition is tablegen-heavy, and we lean on the "read the
  generated code" habit throughout. Chapter 10's `tutorial-opt` driver is
  the tool we extend.
- A configured build in `build/` per the top-level
  [README](../../README.md), so that `build/tools/tutorial-opt` exists.

Run every command from the repo root with the virtualenv active, and put
the MLIR wheel's binaries (`mlir-tblgen`, `mlir-opt`, ...) on your `PATH`:

```bash
source .venv/bin/activate
export PATH=$(python -m mlir_wheel --root-dir)/bin:$PATH
```

---

## 1. Concepts: what's in a dialect, and why polynomials

Chapter 1 described a dialect as a self-contained set of operations and
types with defined semantics. Defining one means providing:

1. a **dialect shell** — the namespace, its registration, and bookkeeping;
2. **types** — here, "a polynomial" (`!poly.poly<N>`);
3. **ops** — here, arithmetic on polynomials;
4. eventually, the *behaviors* hanging off those ops (verification,
   folding, lowering) — that's Chapters 13–19.

### A taxonomy of dialects: where `poly` sits

There is a large surface area of design options when writing a dialect,
and it helps to first see the landscape of dialects that already exist.
Upstream MLIR ships [around fifty](https://mlir.llvm.org/docs/Dialects/),
and downstream projects (TensorFlow, IREE, torch-mlir, onnx-mlir) define
many more — a zoo, until you sort the dialects by the *role* each plays
in a compilation pipeline. The EuroLLVM 2023 talk
["MLIR Dialect Design and Composition for Front-End Compilers"](https://www.youtube.com/watch?v=hIt6J1_E21c)
by Jeff Niu and Mehdi Amini does exactly that. The first two columns
below reproduce the talk's slide (asterisks included; TFG, TFSavedModel,
TFE, tf_type, and TFRT are TensorFlow-family dialects); the third column
explains each role:

| Category | Dialects | What the role means |
|---|---|---|
| Input | TFG, TFSavedModel, "Input" (iree), Torch, ONNX | Mirror an external system's format so its programs can *enter* MLIR: `torch` models PyTorch programs, `onnx` models ONNX graphs. The design brief is faithful import, not optimization. |
| Hourglass | GPU, LLVM, SPIRV, arith | The pipeline's narrow waist: many higher-level dialects lower *into* them, and one set of lowerings continues *out* to many targets. Chapter 2's lowering chain (`math` → `scf`/`arith` → `cf` → `llvm`) showed the funnel — arithmetic from any source ends up as `arith` ops, and a single `arith`-to-`llvm` conversion serves everything above it. |
| Optimizer | linalg, affine, mhlo, shape, flow (iree) | Representations chosen to make a class of transformations tractable: `affine` (Chapter 5) restricts loops to affine forms precisely so polyhedral loop analysis applies; `linalg` and `shape` do the analogous thing for tensor programs. |
| Computation | arith, tensor, vector, index | The ordinary number-crunching that the other roles arrange and shuttle around. |
| Output | RISCV, TFRT, LLVM, SPIRV | Mirror an external *target* so programs can leave MLIR: the `llvm` dialect is a transliteration of LLVM IR that exists to be translated out (Chapter 19 does it); `spirv` likewise for SPIR-V binaries. |
| Paradigm | scf, linalg, openmp, cf, affine, func, builtin, gpu | Encode a programming model — the *shape* of a program rather than its math: `func` (functions and calls), `scf`/`cf` (Chapter 2's structured vs unstructured control flow), `openmp` (shared-memory parallelism), `gpu` (the kernel-launch model). |
| Glue | builtin\*, TFE\*, tf_type, DLTI\* | Shared vocabulary and metadata that let the others compose: `builtin` is the types and attributes every dialect borrows (`i32`, `tensor<3xi32>`, `dense<...>`), and [`dlti`](https://mlir.llvm.org/docs/Dialects/DLTIDialect/) attaches data-layout and target information to modules. |
| Meta | PDL, IRDL, Transform | IR *about* IR: [`pdl`](https://mlir.llvm.org/docs/Dialects/PDLOps/) represents rewrite patterns themselves as MLIR programs (Chapter 17's PDLL compiles to it), [`irdl`](https://mlir.llvm.org/docs/Dialects/IRDL/) defines dialects as IR — a runtime alternative to this chapter's tablegen — and [`transform`](https://mlir.llvm.org/docs/Dialects/Transform/) scripts entire transformation pipelines as IR (Part 5 uses it). |

Two reading notes. First, the categories overlap on purpose — they
classify *roles*, not dialects, and a dialect earns every role it plays:
`llvm` and `spirv` are hourglass *and* output, `arith` is hourglass
*and* computation, `affine` and `linalg` are optimizer *and* paradigm.
Second, the zoo moves fast — `mhlo` (XLA's HLO in MLIR form) has since
been succeeded by StableHLO — but the roles are stable even as the
names rotate.

In this taxonomy `poly` is squarely a **computation** dialect: it exists
so that a program can say "multiply these polynomials" without saying
how, keeping the mathematical intent visible for optimization — exactly
the trick Part 1 promised dialects could pull off.

### What `poly` represents, concretely

A value of the `poly` dialect is a **single-variable polynomial**, stored
as its list of coefficients, where **index i holds the coefficient of
xⁱ**:

```
[1, 2, 3]   ↔   1 + 2x + 3x²
[5]         ↔   the constant polynomial 5
[0, 1]      ↔   x
```

Arithmetic is what you learned in school, on those lists: **addition**
is coefficient-wise (`[1, 2, 3] + [2, 3, 4] = [3, 5, 7]`), and
**multiplication** multiplies every term by every term, with products of
xⁱ and xʲ landing at x^(i+j): `[1, 1] · [1, 1] = [1, 2, 1]`, i.e.
(1 + x)² = 1 + 2x + x². (Chapter 14 works a bigger example digit by digit
when the compiler starts doing this arithmetic itself.)

Two wraparound rules make the type finite, and both are visible in the
type syntax `!poly.poly<10>`:

- **Coefficients** are 32-bit integers with wraparound — the ring
  ℤ/2³²ℤ. (That's the `i32` you'll see in coefficient tensors.)
- **Powers** wrap at the *degree bound*, the `10` in the type: x¹⁰ ≡ 1,
  so x¹² is the same element as x², and every polynomial is
  representable with 10 coefficient slots. Mathematicians write this ring
  ℤ/2³²ℤ[x]/(x¹⁰ − 1).

So a tiny `poly` program reads like annotated math — build
p = 1 + 2x + 3x², square it, evaluate at x = 7:

```mlir
%coeffs = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
%p  = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<10>    // p = 1 + 2x + 3x²
%sq = poly.mul %p, %p : !poly.poly<10>                              // p² = 1 + 4x + 10x² + 12x³ + 9x⁴
%c7 = arith.constant 7 : i32
%v  = poly.eval %sq, %c7 : (!poly.poly<10>, i32) -> i32             // p²(7) = 162² = 26244
```

(Wrap that in a `func.func` and `build/tools/tutorial-opt` already
parses and re-prints it — by the end of this chapter you'll know why.)

Why give this its own dialect instead of writing loops over `tensor`s of
coefficients? The same answer as Chapter 5's `affine` story: at this
level, the compiler can use *polynomial* facts — fold a `poly.mul` of
constants by actually multiplying polynomials (Chapter 14), simplify with
algebraic identities (Chapter 16) — none of which are visible once the
program is a soup of loops and loads; only when the polynomial-level
optimizations are done will we lower to that soup (Chapters 18–19).
Polynomial arithmetic is also the computational heart of FHE (fully
homomorphic encryption), the original mlir-tutorial's end goal:
ciphertexts in lattice-based cryptography *are* vectors of polynomials
in exactly this kind of ring. (The design deliberately prioritizes "get
*some* dialect defined" over optimal design.)

Everything lives in [`lib/Dialect/Poly/`](../../lib/Dialect/Poly/),
following Chapter 10's layout conventions: three tablegen files (dialect,
types, ops), thin `.h` wrappers, a `.cpp` or two, and a `CMakeLists.txt`.

## 2. The dialect shell

In tablegen, a dialect shell is a single record: one `def` deriving from
the upstream `Dialect` class, whose `let` fields tell the generators
everything they need — the dialect's textual name, where the generated
C++ should live, and which optional machinery to emit. Here is
[`PolyDialect.td`](../../lib/Dialect/Poly/PolyDialect.td) (header guard
elided):

***lib/Dialect/Poly/PolyDialect.td***
```tablegen
include "mlir/IR/OpBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let summary = "A dialect for polynomial math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::tutorial::poly";

  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}
```

- `name` — the namespace prefix in IR text: ops print as `poly.add`, types
  as `!poly.poly<10>`.
- `cppNamespace` — where the generated C++ lands. Unlike the pass
  generator (Chapter 11), which emitted namespace-less code for you to
  wrap, the dialect generators emit these namespaces themselves.
- `useDefaultTypePrinterParser` — asks tablegen to generate the dialect's
  type parser/printer from each type's `assemblyFormat` (section 4).
  Without it, you write `parseType`/`printType` by hand — and forgetting
  it entirely is a classic silent failure where your type won't parse.
- `hasConstantMaterializer` — a **forward reference to Chapter 14**;
  ignore it today.

Run the generator yourself (Chapter 11's habit — `-I` paths let the `.td`
includes resolve: one for the wheel's upstream `.td` files, and
`-I lib/Dialect/Poly` because these files include each other by bare
name):

```bash
mlir-tblgen --gen-dialect-decls \
  -I $(python -m mlir_wheel --root-dir)/include \
  -I lib/Dialect/Poly lib/Dialect/Poly/PolyDialect.td
```

The interesting part of the output (real, verified):

```cpp
namespace mlir::tutorial::poly {

/// A dialect for polynomial math
/// ...
class PolyDialect : public ::mlir::Dialect {
  explicit PolyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~PolyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("poly");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

  ::mlir::Operation *materializeConstant(...) override;
};

} // namespace mlir::tutorial::poly
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tutorial::poly::PolyDialect)
```

A `Dialect` subclass whose `initialize()` we must fill in (section 6) —
that hook is where types and ops get attached. `parseType`/`printType`
are there because of `useDefaultTypePrinterParser`;
`materializeConstant` (its real signature spelled out in the actual
output) because of `hasConstantMaterializer` (Chapter 14).
The matching `--gen-dialect-defs` output provides constructor/destructor
and TypeID plumbing. The consuming header,
[`PolyDialect.h`](../../lib/Dialect/Poly/PolyDialect.h), is two
`#include`s: one the generated code needs, plus the `.h.inc` — the same
`.inc`-consumption pattern as Chapter 11, minus the `#define` gates
(dialect decls have no sections to select). You don't have to rerun
tablegen to read any of this, by the way: the build already generated
everything under `build/lib/Dialect/Poly/*.inc`.

## 3. Registering the dialect

One line in [`tools/tutorial-opt.cpp`](../../tools/tutorial-opt.cpp),
next to the pass registrations from Chapter 10:

***tools/tutorial-opt.cpp*** (excerpt)
```cpp
mlir::DialectRegistry registry;
registry.insert<mlir::tutorial::poly::PolyDialect>();
mlir::registerAllDialects(registry);
```

And the tool now understands `poly` (from
`./build/tools/tutorial-opt --help`):

```
Available Dialects: acc, affine, ..., pdl, pdl_interp, poly, ptr, quant, ...
```

Two fun details about that list. First, why registration matters — feed a
`poly` file to the wheel's stock `mlir-opt` and you get:

```
test/part3/poly_syntax.mlir:6:44: error: `!"poly"<"poly<10>">` type created
with unregistered dialect. If this is intended, please call
allowUnregisteredDialects() on the MLIRContext, or use
-allow-unregistered-dialect with the MLIR opt tool used
```

(MLIR can optionally round-trip unknown-dialect IR as opaque text — that's
the `-allow-unregistered-dialect` escape hatch it mentions — but nothing
can be *done* with such ops.) Second, a bit of history: when the original
mlir-tutorial was written, upstream's `polynomial` dialect sat right next
to `poly` in this list — the tutorial's author had contributed `poly`
upstream in expanded form. Upstream has since removed it again (an
archived copy of its docs survives — see exercise 5), but the point
stands: you are studying the prototype of what was, for a while, a real
upstream dialect.

## 4. The type: `!poly.poly<10>`

A custom type gets the same treatment as the dialect: one tablegen
record — a `TypeDef` tied to its dialect — from which the generators
produce a C++ class. Three `let` fields carry the substance, and the
bullets below unpack each: the type's syntax keyword (its *mnemonic*),
the compile-time data it carries (its *parameters* — the `10` in
`!poly.poly<10>`), and the textual format that
`useDefaultTypePrinterParser` turns into a parser and printer. Here is
[`PolyTypes.td`](../../lib/Dialect/Poly/PolyTypes.td) (header guard
elided):

***lib/Dialect/Poly/PolyTypes.td***
```tablegen
include "PolyDialect.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coefficients";

  let description = [{
    A type for polynomials with integer coefficients in a single-variable polynomial ring.
  }];

  let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";
}
```

- The `class`/`def` split is Chapter 11's lesson applied as a convention:
  `Poly_Type` is a reusable template tying every type in this dialect to
  `Poly_Dialect`; `def Polynomial` instantiates the one concrete type.
  (A warning from experience: accidentally writing `def` where you
  mean `class`, or vice versa, produces spectacularly unhelpful errors.)
- `TypeDef<Poly_Dialect, "Polynomial">` generates a class named
  `PolynomialType` (C++ name = record name + `Type`).
- `mnemonic = "poly"` is the syntax keyword. Types in MLIR are written
  with a leading `!` (`!poly.poly<10>`), distinguishing them from ops
  (`poly.add`) and attributes (`#...`) — you have been reading builtin
  types (`i32`, `memref<4xi32>`) without prefixes only because builtins
  get that privilege.
- [`parameters`](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#parameters)
  — the compile-time data the type carries. Here one `int` named
  `degreeBound`: the `10` in `!poly.poly<10>`, section 1's
  power-wraparound bound. Nothing in *this* chapter enforces those
  semantics — a type is just data; they become real when Chapter 14's
  folders compute with it. Encoding the bound in the *type* means degree
  compatibility is checked statically, the same way `tensor<3xi32>` vs
  `tensor<4xi32>` mismatches are.
- ``assemblyFormat = "`<` $degreeBound `>`"`` — the textual syntax between
  the mnemonic and the end of the type, as a template with literals in
  backticks. This is what `useDefaultTypePrinterParser` turns into a
  parser and printer.

The generated class, from `--gen-typedef-decls` (run it like section 2's
command, on `PolyTypes.td`, adding `-typedefs-dialect=poly` as the
repo's CMake does; verified):

```cpp
class PolynomialType : public ::mlir::Type::TypeBase<PolynomialType,
                           ::mlir::Type, detail::PolynomialTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "poly.poly";
  static constexpr ::llvm::StringLiteral dialectName = "poly";
  static PolynomialType get(::mlir::MLIRContext *context, int degreeBound);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"poly"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getDegreeBound() const;
};
```

Note there is no public constructor — only a static
`get(MLIRContext*, int)`. MLIR types are **uniqued**: for a given context,
`!poly.poly<10>` exists exactly once, `get` returns the canonical instance,
and comparing types is pointer comparison. The machinery making that work
is `PolynomialTypeStorage`, generated into the `--gen-typedef-defs`
output (read it in `build/lib/Dialect/Poly/PolyTypes.cpp.inc`):

```cpp
struct PolynomialTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<int>;
  ...
  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }
  ...
  int degreeBound;
};
```

A hash-consed
[storage class](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#storage-classes):
the parameters form a key, the key is hashed, and `get` looks up (or
creates) the one instance per key. Tablegen writes it for free for simple
parameters like `int` that have trivial construction/destruction
semantics; wilder parameter types (an array needing allocation, say)
eventually require hand-written storage.

## 5. The ops

[`PolyOps.td`](../../lib/Dialect/Poly/PolyOps.td) defines the operations,
one ODS record per op, with the same record-plus-`let`-fields shape as
the dialect and type above: an `Op<...>` ties the op to its dialect and
names its mnemonic, and `let`s declare what the op takes (`arguments`),
what it produces (`results`), and how it prints (`assemblyFormat`). The
binary ops come from a shared base — shown here as the current repo has
it, with the forward references annotated:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic,
    [Pure, ElementwiseMappable, SameOperandsAndResultType]> {  // Chapter 13
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  let hasFolder = 1;         // Chapter 14
  let hasCanonicalizer = 1;  // Chapter 16
}

def Poly_AddOp : Poly_BinOp<"add"> {
  let summary = "Addition operation between polynomials.";
}
def Poly_SubOp : Poly_BinOp<"sub"> { ... }
def Poly_MulOp : Poly_BinOp<"mul"> { ... }
```

In its first incarnation this base was simpler — no trait list, no folder,
and plain `Polynomial:$lhs` arguments; the shape to internalize is:

- `Op<Poly_Dialect, "add">` — ties the op to the dialect and names its
  mnemonic, like `TypeDef` did for the type.
- `arguments = (ins <constraint>:$name, ...)` — the operands, each with a
  **type constraint** and a name; `results = (outs ...)` likewise. The
  names aren't decoration: tablegen generates accessors `getLhs()`,
  `getRhs()`, `getOutput()` from them, plus typed `build(...)` overloads
  used by `rewriter.create<AddOp>(...)` — Chapter 10's rewriter API was
  consuming generated methods like these all along.
- Constraints compose: `Polynomial` (our own `def` from PolyTypes.td!) is
  itself a constraint meaning "must be a PolynomialType";
  `TensorOf<[AnyInteger]>` below means "tensor of any integer type"; the
  current file's `PolyOrContainer` relaxes "a poly" to "a poly, or a
  tensor/vector of polys" (that generalization is Chapter 13 business).
- `assemblyFormat` — same idea as for types: `$lhs`, `$rhs` splice
  operands, `attr-dict` is a mandatory slot for discretionary attributes,
  and `type(...)` directives say where types appear. Because
  `SameOperandsAndResultType` (Chapter 13) guarantees all three types
  match, one type suffices: `poly.add %a, %b : !poly.poly<10>`. The
  earlier pre-trait version spelled all types:
  `... : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>`.

Then the ops that cross the dialect boundary — polynomials have to come
from somewhere and produce usable numbers:

***lib/Dialect/Poly/PolyOps.td*** (excerpt)
```tablegen
def Poly_FromTensorOp : Op<Poly_Dialect, "from_tensor", [Pure]> {
  let summary = "Creates a Polynomial from integer coefficients stored in a tensor.";
  let arguments = (ins TensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";
  let hasFolder = 1;  // Chapter 14
}

def Poly_EvalOp : Op<Poly_Dialect, "eval",
    [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {  // Chapter 13/10
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$polynomial, IntOrComplex:$point);
  let results = (outs IntOrComplex:$output);
  let assemblyFormat = "$polynomial `,` $point attr-dict `:` `(` qualified(type($polynomial)) `,` type($point) `)` `->` type($output)";
  let hasVerifier = 1;      // Chapter 15
  let hasCanonicalizer = 1; // Chapter 16
}
```

`from_tensor` bridges *in* (a `tensor<3xi32>` of coefficients becomes a
polynomial), `eval` bridges *out* (a polynomial and a point produce a
number). The current file also has `to_tensor` and a `poly.constant` op —
both arrive in later chapters.

What does an op record generate? The habit from Chapter 11 pays off again —
here is the verified `--gen-op-decls` output for `AddOp`, heavily trimmed
(read the full version in `build/lib/Dialect/Poly/PolyOps.h.inc`; the
trait list on the base class is Chapter 13's subject):

```cpp
class AddOp : public ::mlir::Op<AddOp, ::mlir::OpTrait::ZeroRegions,
                  ::mlir::OpTrait::OneResult, /* ... */
                  ::mlir::OpTrait::SameOperandsAndResultType, /* ... */> {
public:
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("poly.add");
  }
  ::mlir::TypedValue<::mlir::Type> getLhs();     // operand 0
  ::mlir::TypedValue<::mlir::Type> getRhs();     // operand 1
  ::mlir::TypedValue<::mlir::Type> getOutput();  // result 0
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::Type output, ::mlir::Value lhs, ::mlir::Value rhs);
  // ... many more build/create overloads, parse, print, fold, verify ...
};
```

(One drift from older MLIR worth knowing: each `build` overload now also
gets a static `AddOp::create(builder, loc, ...)` twin — the
now-preferred spelling of `builder.create<AddOp>(loc, ...)`.)

## 6. Wiring it into C++ and the build

[`PolyDialect.cpp`](../../lib/Dialect/Poly/PolyDialect.cpp) is where all
the generated definitions get compiled and attached (trimmed to today's
scope — the file's `materializeConstant` body is Chapter 14's):

***lib/Dialect/Poly/PolyDialect.cpp*** (excerpt)
```cpp
#include "lib/Dialect/Poly/PolyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Poly/PolyOps.cpp.inc"

namespace mlir {
namespace tutorial {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Poly/PolyOps.cpp.inc"
      >();
}

} // namespace poly
} // namespace tutorial
} // namespace mlir
```

The gates are doing something cute: the same `PolyTypes.cpp.inc` is
included *twice* — once with `GET_TYPEDEF_CLASSES` to emit the class
definitions, once with `GET_TYPEDEF_LIST` where it expands to just a
comma-separated list of type names (for us, exactly
`::mlir::tutorial::poly::PolynomialType`), which lands inside
`addTypes<...>()` as its template arguments. That's the whole
registration: `initialize()` hands the dialect its types and ops. (The
mild ugliness of how much must funnel into this one `.cpp` is real —
dialect `.cpp` files in real projects are long.)

The build side lives in
[`lib/Dialect/Poly/CMakeLists.txt`](../../lib/Dialect/Poly/CMakeLists.txt).
MLIR's CMake helpers provide a one-call `add_mlir_dialect(...)`, but this
repo inlines the calls it bundles, to control the generated file names:

***lib/Dialect/Poly/CMakeLists.txt*** (excerpt)
```cmake
set(LLVM_TARGET_DEFINITIONS PolyOps.td)
mlir_tablegen(PolyOps.h.inc -gen-op-decls)
mlir_tablegen(PolyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PolyTypes.h.inc -gen-typedef-decls -typedefs-dialect=poly)
mlir_tablegen(PolyTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=poly)
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs -dialect=poly)
add_public_tablegen_target(MLIRPolyOpsIncGen)

add_mlir_dialect_library(MLIRPoly
    PolyDialect.cpp
    PolyOps.cpp
    ...)
```

Each `mlir_tablegen` line is one backend invocation — the same
`mlir-tblgen` runs you did by hand in sections 2 and 4, six of them, with
the same pattern as Chapter 11's pass generation. One subtlety: all six
run on *`PolyOps.td`* (the `LLVM_TARGET_DEFINITIONS`). That works because
`PolyOps.td` transitively includes the other two files, so tablegen sees
one merged record universe, and the `-dialect=poly` /
`-typedefs-dialect=poly` flags tell each backend which records to emit.

### The whole picture

That's a lot of files playing relay — sections 2–6 introduced each one
where it mattered, but the dependency structure is easier to see drawn
once. Read the chart top to bottom: each of the three tablegen files
heads its own vertical track (definitions → generated `.inc` pair →
hand-written wrapper header), and everything funnels into the single
`PolyDialect.cpp`:

```
         ┌────────────── include ──────────────┐
         │                                     ▼
   PolyDialect.td  ──▶  PolyTypes.td  ──▶  PolyOps.td        .td layer (you write;
         │                    │                 │             A ──▶ B: B includes A)
         │ mlir-tblgen        │ mlir-tblgen     │ mlir-tblgen
         │ -gen-dialect-      │ -gen-typedef-   │ -gen-op-
         │   decls / -defs    │   decls / -defs │   decls / -defs
         ▼                    ▼                 ▼
   PolyDialect.h.inc    PolyTypes.h.inc    PolyOps.h.inc     generated into build/
   PolyDialect.cpp.inc  PolyTypes.cpp.inc  PolyOps.cpp.inc   (never edited)
         │                    │                 │
         │ #include           │ #include        │ #include
         ▼                    ▼                 ▼
   PolyDialect.h        PolyTypes.h        PolyOps.h         .h layer (you write;
   (wraps h.inc)        (wraps h.inc)      (wraps h.inc;     each a thin wrapper
         │                    │             also includes    around its h.inc)
         │                    │             PolyDialect.h,
         │                    │             PolyTypes.h)
         │                    │                 │
         └──────────┬─────────┴─────────────────┘
                    │ #include — all three .h, plus all three .cpp.inc
                    │ (PolyTypes/PolyOps .cpp.inc twice, gated: once for
                    │  the class bodies, once as the list handed to
                    │  addTypes<...> / addOperations<...>)
                    ▼
              PolyDialect.cpp — initialize() attaches types and ops
                    │
                    │ compiled into the MLIRPoly library
                    │ (CMakeLists.txt: mlir_tablegen ×6 +
                    │  add_mlir_dialect_library)
                    ▼
              tools/tutorial-opt.cpp — registry.insert<PolyDialect>()
```

Things the chart makes visible:

- **The top row's arrows are the `-I` flags.** `PolyTypes.td` includes
  `PolyDialect.td` (its `TypeDef<Poly_Dialect, ...>` must name the
  dialect record), and `PolyOps.td` includes both (its ops name
  `Poly_Dialect` *and* the `Polynomial` constraint). That's why
  section 2's manual `mlir-tblgen` runs needed `-I lib/Dialect/Poly` —
  tablegen re-resolves the includes every time it processes a file, so
  each generator invocation sees one merged record universe (the property
  that also lets the CMake run all six backends on `PolyOps.td` alone).
- **The hand-written layer is thin on purpose.** All three `.h` files
  are a few lines each; the substance lives in the generated `.inc`s.
  The only hand-written files with real content are `PolyDialect.cpp`
  and `PolyOps.cpp` — the latter would be empty today and only exists to
  hold Chapters 14–16's folders and canonicalizers. There is no
  `PolyTypes.cpp` at all: the generated type definitions are compiled via
  `PolyDialect.cpp`'s includes. (`PolyOps.h` also pulls in `PolyTraits.h`,
  and the directory has a `PolyPatterns.td` — Chapters 13 and 16's
  subjects, ignore them today.)
- **Registration is a two-step chain.** `initialize()` attaches types
  and ops to the *dialect* (bottom of the funnel), and section 3's
  `registry.insert<PolyDialect>()` attaches the dialect to the *tool* —
  break either link and `tutorial-opt` is back to section 3's
  "unregistered dialect" error.

## 7. Step: exercise the syntax

[`test/part3/poly_syntax.mlir`](../../test/part3/poly_syntax.mlir) is a
pure round-tripping test — no passes, just "does this parse and
re-print". Its `RUN` lines are the Chapter 9 pattern:

```mlir
// RUN: tutorial-opt %s > %t
// RUN: FileCheck %s < %t
```

Run the underlying command by hand first:

```bash
./build/tools/tutorial-opt test/part3/poly_syntax.mlir
```

```mlir
module {
  func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    return %arg0 : !poly.poly<10>
  }
  func.func @test_op_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    %0 = poly.add %arg0, %arg1 : !poly.poly<10>
    %1 = poly.sub %arg0, %arg1 : !poly.poly<10>
    %2 = poly.mul %arg0, %arg1 : !poly.poly<10>
    %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %3 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
    %c7_i32 = arith.constant 7 : i32
    %4 = poly.eval %3, %c7_i32 : (!poly.poly<10>, i32) -> i32
    %cst_0 = complex.constant [1.000000e+00, 2.000000e+00] : complex<f64>
    %5 = poly.eval %3, %cst_0 : (!poly.poly<10>, complex<f64>) -> complex<f64>
    %from_elements = tensor.from_elements %arg0, %arg1 : tensor<2x!poly.poly<10>>
    %6 = poly.add %from_elements, %from_elements : tensor<2x!poly.poly<10>>
    %7 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
    %8 = poly.constant dense<[2, 3, 4]> : tensor<3xi8> : !poly.poly<10>
    %9 = poly.constant dense<[2, 3, 4]> : tensor<3xi8> : !poly.poly<10>
    %10 = poly.constant dense<4> : tensor<100xi32> : !poly.poly<10>
    %11 = poly.to_tensor %1 : !poly.poly<10> -> tensor<10xi32>
    return %3 : !poly.poly<10>
  }
}
```

Parsing back what the printer emits is exactly what the test locks in —
if your `assemblyFormat` prints something its own parser rejects, this is
where you find out. (The printer also renames: the source file's
`%3`-through-`%14` numbering comes back as `%cst`, `%c7_i32`,
`%from_elements`, and a fresh `%0, %1, ...` sequence — SSA names are not
semantic, as Chapter 9's capture variables already insisted.)

### What do these values *mean*?

A fair question at this point: what is the *value* of `%0`? The honest
answer: **this test computes nothing** — `%arg0` and `%arg1` are runtime
values and no pass runs here. But each SSA value has a precise *meaning*
under section 1's semantics (ℤ/2³²ℤ[x]/(x¹⁰ − 1), coefficient list
indexed by power), and reading the function through that lens is good
practice. Writing `p₀ = %arg0` and `p₁ = %arg1`:

| Value | Meaning |
|-------|---------|
| `%0`  | `p₀ + p₁` — coefficient-wise addition, mod 2³² |
| `%1`  | `p₀ − p₁` |
| `%2`  | `p₀ · p₁` — cyclic convolution: products of xⁱ·xʲ land at x^((i+j) mod 10) |
| `%3`  | the concrete polynomial **1 + 2x + 3x²** (coefficients `[1,2,3]`, index = power of x) |
| `%4`  | `%3` evaluated at 7: 1 + 2·7 + 3·7² = **162** : i32 |
| `%5`  | `%3` evaluated at 1+2i: 1 + 2(1+2i) + 3(1+2i)² = **−6 + 16i** (eval accepts complex points — that's the `IntOrComplex` constraint) |
| `%6`  | elementwise (Chapter 13): the tensor `[2p₀, 2p₁]` |
| `%7`, `%8`, `%9` | all the *same* polynomial **2 + 3x + 4x²** — stored as i32, as i8, and (in the source file) as the hex blob `dense<"0x020304">`; the printer proving the point by rendering all three as `[2, 3, 4]` |
| `%10` | 100 coefficients, all 4, in a 10-slot ring: x¹⁰ ≡ 1 reduces it to **40·(1 + x + ... + x⁹)**. Note nothing *checks* coefficient count against the degree bound — the attribute happily stores 100 entries; reduction is a semantic fact, not a syntactic one (verification is Chapter 15's business) |
| `%11` | the coefficient tensor of `%1`, now with its static length 10 visible in the type `tensor<10xi32>` |

None of these "happen" today — but they're not hypothetical either: once
Chapter 14 gives the ops folders, the compiler itself performs exactly
this arithmetic on the constant-valued ones at compile time (and you can
check the numbers above against it).

A cautionary tale from porting this chapter: an earlier version of this
test file read `// RUN FileCheck %s < %t` — missing the colon after
`RUN`. Per Chapter 9, lit only executes `RUN:` lines; without the colon
the line is an ordinary comment, so the test ran `tutorial-opt` and
*never checked its output*. Morals: a test that can't fail is not a
test, and typos in magic comments fail *silently* — when adding a lit
test, break it once on purpose (Chapter 9's habit) to prove it's alive.

Bad inputs are worth trying too. Feed `tutorial-opt` a file containing

```mlir
func.func @f(%a: !poly.poly<x>) { return }
```

and the generated parser — section 4's `PolynomialType::parse`, doing
exactly what the `assemblyFormat` said — reports (verified):

```
/tmp/badpoly.mlir:1:29: error: expected integer value
func.func @f(%a: !poly.poly<x>) { return }
                            ^
/tmp/badpoly.mlir:1:29: error: failed to parse Polynomial parameter 'degreeBound'
which is to be a `int`
```

Finally, run the test through lit the Chapter 9 way:

```bash
lit build/test --filter=poly_syntax -v
```

```
PASS: MLIRUnifiedTutorial :: part3/poly_syntax.mlir (1 of 1)
```

(`cmake --build build --target check-part3` runs it together with the
rest of Part 3's suite — those neighbors, `poly_canonicalize.mlir` and
`poly_verifier.mlir`, belong to later chapters.)

## Exercises

1. Run the section-2 and section-4 `mlir-tblgen` commands and read both
   full outputs — or read the pre-generated versions in
   `build/lib/Dialect/Poly/`. Find where `degreeBound` is stored in
   `PolynomialTypeStorage`, and how `hashKey` uses it.
2. Add a negation op to a *copy* of `PolyOps.td` (say in `/tmp`):

   ```tablegen
   def Poly_NegOp : Op<Poly_Dialect, "neg"> {
     let arguments = (ins Polynomial:$input);
     let results = (outs Polynomial:$output);
   }
   ```

   Regenerate with `--gen-op-decls`. Find its generated accessor and
   builders. What `assemblyFormat` would make it print as the following?

   ```mlir
   poly.neg %x : !poly.poly<10>
   ```

3. Foreshadowing Chapter 13: evaluate a polynomial at an `i64` point —
   `poly.eval %p, %c : (!poly.poly<10>, i64) -> i64` — and run it through
   `tutorial-opt`. The error (`'poly.eval' op requires each numeric operand
   to be a 32-bit integer` — verified) comes from that mysterious
   `Has32BitArguments` trait in `PolyOps.td`.
4. Feed the printer's own output back in:
   `./build/tools/tutorial-opt test/part3/poly_syntax.mlir | ./build/tools/tutorial-opt`
   — the round-trip property, composed (verified to succeed). Then break
   `assemblyFormat` mentally: if `from_tensor`'s format omitted
   `type($input)`, could the parser know what tensor type to expect?
5. Compare `poly` with upstream's short-lived `polynomial` dialect
   ([archived docs](https://web.archive.org/web/20241226114248/https://mlir.llvm.org/docs/Dialects/PolynomialDialect/))
   — how did the upstream version generalize the coefficient type that
   our `poly` hardcodes as 32-bit?

## Where to go next

The dialect parses, prints, and round-trips — but it doesn't *do*
anything yet: nothing checks semantic invariants beyond types, nothing
simplifies `poly.add` of constants, and nothing lowers `poly` to real
arithmetic. That begins with
[Chapter 13: Using Traits](../ch13-traits/README.md): those bracketed
lists (`[Pure, ElementwiseMappable, ...]`) we skipped over are
declarative hooks into upstream passes — and the reason `--cse` and
friends will just work on `poly` ops.
