# Chapter 24: Equality Saturation and E-graphs

> Adapted from mlir-ir-lowering's 6_egraph chapter, itself based on Stephen
> Diehl's "MLIR" article series (*"MLIR Part 6 — Specializing Python with
> E-graphs"*).

### The optimization that happens *before* lowering

Diehl's series sketches a modern compiler pipeline in which *"the core
language is optimized by core-to-core transformations, optionally using
e-graphs"* before anything is handed to MLIR. This chapter builds that step.
It's a different kind of optimization from most of what this tutorial has
done so far — it happens at the **high level**, on the math itself, *before*
anything is lowered to `linalg` or `llvm`.

Why bother, when LLVM and MLIR already optimize? Because they don't know your
*algebra*. LLVM will not turn `transpose(transpose(A))` into `A`, or
`e^x · e^x · e^x` into `e^{3x}`, and under default IEEE float semantics
neither LLVM nor MLIR will cancel the 2s in `(a*2)/2` (you will watch MLIR's
canonicalizer decline to in section 4) — those are domain identities, and by
the time the code is loops over memory the structure needed to spot them is
gone. Applying them up front, on the expression tree, is "low-hanging fruit"
that can delete whole matrix operations before codegen even starts.

The obstacle is the **phase-ordering problem**: rewrites interfere. Given
`x*2 → x<<1` and `(x*y)/y → x`, the expression `(a*2)/2` *should* become `a` —
but if you greedily apply the first rule you get the stuck term `(a<<1)/2` and
miss the cancellation. Which rule, in which order? **Equality saturation**
sidesteps the question entirely.

Diehl's article uses the `egglog` library (whose API churns); to keep the
chapter self-contained and runnable, we build a small e-graph engine from
scratch in pure Python — then emit MLIR from the optimized expression,
compile it to native code and execute it, exactly as the article's pipeline
does.

## What you will learn

- Why destructive rewriting — the strategy behind MLIR's greedy pattern
  driver, canonicalization, DRR and PDLL — suffers from **phase ordering**,
  and a runnable demonstration of a rewrite getting stuck.
- What an **e-graph** is: e-nodes, e-classes, **union-find**, the
  **hashcons**, and why **congruence closure** (`rebuild`) is needed.
- **E-matching** of rewrite patterns against a graph of equivalence classes.
- **Equality saturation**: apply every rule to a fixed point, never deleting.
- **Cost-based extraction**: pick the cheapest equivalent term, even through
  cycles.
- How to emit MLIR (`func` + `arith`) from the optimized expression, compile
  original and optimized versions to shared libraries, and check that they
  agree.

## Prerequisites

- [Chapter 16](../../part3-building-a-dialect/ch16-canonicalizers-drr/README.md)
  (canonicalization patterns and DRR) and
  [Chapter 17](../../part3-building-a-dialect/ch17-pdll/README.md) (PDLL):
  this chapter is about what those rewrite systems *can't* do, so you
  should know what they do. Both run on the greedy driver from
  [Chapter 10](../../part2-writing-passes/ch10-first-pass-cpp/README.md).
- [Chapter 23](../ch23-dataflow-analysis/README.md)'s fixed-point
  iteration — saturation and extraction are both "iterate until nothing
  changes" loops.
- The MLIR → LLVM IR → `clang` path for native code from
  [Chapter 3](../../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md),
  and the `ll_fp_compat.py` filter explained in
  [Chapter 28](../../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md).
- Python 3 (no packages needed for the e-graph itself) and the repo's
  `.venv` with the MLIR wheel, plus a system `clang`. No C++ build is needed.

Everything runs from one script, from the repo root:

```bash
bash part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/build.sh
```

| File (in [`examples/`](examples/)) | Role |
| --- | --- |
| [`egraph.py`](examples/egraph.py) | the engine: `add` / `merge` / `rebuild` / `ematch` / `saturate` / `extract` (+ `dump` for inspection) |
| [`greedy_demo.py`](examples/greedy_demo.py) | the phase-ordering problem, with a destructive rewriter |
| [`optimize_demo.py`](examples/optimize_demo.py) | the three concept demos (pure Python) |
| [`inspect_egraph.py`](examples/inspect_egraph.py) | prints every e-class before and after saturation |
| [`to_mlir.py`](examples/to_mlir.py) | emit a `func.func` of `arith` ops from an optimized term |
| [`capstone.py`](examples/capstone.py) | optimize an expression, emit MLIR for original & optimized |
| [`run_compiled.py`](examples/run_compiled.py) | run both compiled functions, confirm they agree |
| [`build.sh`](examples/build.sh) | run the demos, then compile + execute the capstone |

Outputs land in `examples/build/` (gitignored). The lit tests
[`test/part5/ch24_egraph_demos.mlir`](../../test/part5/ch24_egraph_demos.mlir)
and
[`test/part5/ch24_egraph_capstone.mlir`](../../test/part5/ch24_egraph_capstone.mlir)
check every output shown below:

```bash
.venv/bin/lit build/test -v --filter ch24_
```

---

## 1. The phase-ordering problem

Every rewriting system you have built in this tutorial is **destructive**.
Chapter 10's greedy driver pops an op off a worklist, tries the patterns
(highest benefit first), and when one matches it *replaces* the op — the old
form is gone. Chapter 16's canonicalization patterns and DRR rules, and
Chapter 17's PDLL patterns, all feed the same driver. That design is fast
and works beautifully when the rules form a well-behaved *normal form* system
(that was Chapter 16's point about "canonical"). But as soon as two rules
compete for the same piece of IR, *which one fires first decides the result*.

[`greedy_demo.py`](examples/greedy_demo.py) makes this concrete with two
rules on `(a*2)/2`:

```
strength reduction   x*2     -> x<<1
cancellation         (x*y)/y -> x
```

It applies them destructively, first as two separate "passes" each run to a
fixed point (the classic phase-ordering setting — a pass pipeline), then as
one driver holding both rules, visiting the expression in two different
orders:

```
Greedy, destructive rewriting of ((a * 2) / 2):
  as two passes (each run to a fixed point, in sequence):
    cancel; then strength-reduce   ->  a
    strength-reduce; then cancel   ->  ((a shl 1) / 2)   (stuck)
  as one driver holding both rules:
    visiting parents first         ->  a
    visiting operands first        ->  ((a shl 1) / 2)   (stuck)
```

Once `a*2` has been destructively replaced by `a<<1`, the cancellation
pattern `(x*y)/y` has nothing to match: the `*` it needed no longer exists.
No amount of further iteration recovers it. Even inside a single driver,
the *traversal order* (which Chapter 10 noted an optional config struct can
choose) changes the answer. Real compilers paper over this with carefully
tuned pass orders, pattern benefits, and "run canonicalize again" — and
Chapter 17's `MulToAdd` comment ("we are guaranteed ... because the greedy
rewrite engine ensures the PowerOfTwoExpand pattern is run first, since it
has higher benefit") is exactly this kind of ordering argument.

## 2. How an e-graph works

An **e-graph** (equality graph) stores **many equivalent expressions at
once**. Rather than rewriting destructively (and losing the original), it
records that two forms are *equal* and keeps both:

- An **e-class** is a set of expressions known to be equivalent.
- An **e-node** is one operator applied to e-classes (`mul(class3, class7)`),
  hash-consed so shared subexpressions are stored once.
- A rewrite `lhs → rhs` doesn't replace anything; it **adds** `rhs` and
  **merges** its e-class with the one that matched `lhs`.

```text
   destructive rewrite            e-graph: keep both, mark them equal
   (a*2)/2  ──►  a                 ┌─ e-class ───────────────┐
   (original is gone)              │  (a*2)/2   ≡   a<<1/2    │  all equal;
                                   │      ≡        ≡   a      │  extraction
                                   └─────────────────────────┘  picks cheapest
```

That "keep both" is the whole trick: applying `x*2 → x<<1` no longer
*destroys* the `(a*2)/2` form, so `(x*y)/y → x` can still fire on it. The
stuck-term problem disappears because nothing ever gets stuck — every form
coexists.

The algorithm is three phases ([`egraph.py`](examples/egraph.py)):

1. **Saturate** — apply *every* rule everywhere, repeatedly, until no new
   equivalence appears. Order no longer matters: all reachable forms coexist.
2. **Rebuild** — restore congruence (if `a = b`, then `f(a)` and `f(b)` must
   share an e-class too).
3. **Extract** — walk the graph picking, for each e-class, the lowest-cost
   e-node according to a **cost model**, and read out the cheapest
   equivalent term.

The rest of this section walks through the engine piece by piece.

### 2.1 Terms, e-nodes, and the hashcons

Terms (and rewrite patterns) are nested tuples:
`("*", ("sym", "a"), ("lit", 2))` is `a * 2`; leaves are `("lit", value)` or
`("sym", name)`, and a *pattern* may also contain `Var("x")` placeholders. An
e-node is `(op, tuple-of-child-eclass-ids, data)` — note the children are
**e-class ids, not terms**. That is what lets one e-node stand for
exponentially many terms: `*(c3, c1)` means "anything in class 3 times
anything in class 1".

Adding a term is bottom-up, and every e-node goes through the **hashcons**,
a dictionary from e-node to e-class id. If an identical e-node already
exists, you get its class back instead of a new one:

***examples/egraph.py*** (excerpt)
```python
  def add(self, term):
    """Insert a term, returning the id of the e-class that represents it."""
    op = term[0]
    if op in ("lit", "sym"):
      enode = (op, (), term[1])
    else:
      enode = (op, tuple(self.add(arg) for arg in term[1:]), None)
    return self._add_enode(enode)

  def _add_enode(self, enode):
    enode = self._canon(enode)
    if enode in self.hashcons:
      return self.find(self.hashcons[enode])
    i = self._new_class()
    self.hashcons[enode] = i
    self.nodes[i].add(enode)
    return i
```

If that reminds you of MLIR's **CSE** pass, it should: hash-consing *is*
common-subexpression elimination, applied on insertion. In the capstone
below, the two identical halves of `((a*2)/2) * ((a*2)/2)` become one e-class
the moment they are added, just as `mlir-opt --cse` merges them in the IR.

### 2.2 Union-find: what "merge" means

Equivalence is tracked with a **union-find** (disjoint-set) structure:
every e-class id has a `parent`, and the *canonical* id of a class is found by
following parents to a root (with path compression, so chains stay short):

***examples/egraph.py*** (excerpt)
```python
  def find(self, i):
    while self.parent[i] != i:
      self.parent[i] = self.parent[self.parent[i]]   # path compression
      i = self.parent[i]
    return i
```

A rewrite therefore never deletes; it only ever `merge`s the matched class
with a freshly-added right-hand side:

***examples/egraph.py*** (a rewrite adds `rhs` and merges — never replaces)
```python
  def merge(self, a, b):
    """Declare two e-classes equivalent."""
    a, b = self.find(a), self.find(b)
    if a == b:
      return a
    self.parent[b] = a
    self.nodes[a] |= self.nodes[b]
    self.nodes[b] = set()
    return a
```

### 2.3 Rebuild: congruence closure

Merging creates a subtle obligation. Suppose classes `x` and `y` are merged,
and the graph contains both `f(x)` and `f(y)`. Those two e-nodes are now
*the same* e-node (their children are equal), so their e-classes must be
merged too — and that merge may make *another* pair of parents identical,
and so on up the graph. Keeping the graph closed under this rule ("equal
inputs ⇒ equal outputs") is **congruence closure**. `rebuild` re-canonicalizes
every e-node's children through `find`, rebuilds the hashcons, and merges any
classes that now contain the same e-node, repeating until stable:

***examples/egraph.py*** (excerpt)
```python
  def rebuild(self):
    """Restore congruence: if two e-nodes became identical after merges, their
    e-classes must merge too. Repeat until stable."""
    changed = True
    while changed:
      changed = False
      self.hashcons = {}
      self.nodes = {c: {self._canon(en) for en in s}
                    for c, s in self.nodes.items() if self.find(c) == c}
      for c, s in self.nodes.items():
        for en in s:
          if en in self.hashcons and self.find(self.hashcons[en]) != self.find(c):
            self.merge(self.hashcons[en], c)
            changed = True
          self.hashcons[en] = self.find(c)
```

Doing this once per saturation round, rather than after every single merge,
is the "deferred rebuilding" idea popularized by the `egg` library: merges
are cheap, and the invariant is restored in bulk.

### 2.4 E-matching and saturation

To apply a rule, you must find where its left-hand side occurs — but in an
e-graph, "occurs" means *in any of the equivalent forms*. `_match` walks the
pattern and the graph together: a `Var` binds to (or must agree with) an
e-class; an operator pattern tries **every e-node in the class** with that
operator, recursing into the children's classes. So one pattern can match
the same class in several ways, and `ematch` returns every
`(eclass, substitution)` pair. Note the non-linear pattern `(x*y)/y`: the
second `y` must bind to the *same e-class* as the first, which is how "the
divisor equals the multiplier" is checked — equality up to everything the
graph already knows.

`saturate` is then a short loop: collect all matches of all rules (*before*
changing anything, so rule order inside a round can't matter), instantiate
each right-hand side with `_instantiate` and merge it with the matched
class, `rebuild`, and stop when a round adds no new e-nodes — **saturation**,
a fixed point in the same sense as Chapter 23's dataflow solver. A right-hand
side may also be a Python callable (`rhs(egraph, subst)`), the hook for
dynamic rewrites such as constant folding. Because some rule sets never
saturate (associativity and commutativity generate ever more forms — see
exercise 3), the loop has an iteration cap, `max_iters=30`.

### 2.5 Extraction and the cost model

After saturation, the root's e-class holds every equivalent form found.
**Extraction** picks one: the cost of an e-node is its operator's cost plus
the best cost of each child class, and a class's best is its cheapest e-node.

***examples/egraph.py*** (excerpt)
```python
  def extract(self, eclass, costs):
    """Return the lowest-cost term in `eclass`, given per-op `costs`.

    cost(node) = costs[op] + sum of best child costs; best(eclass) = cheapest
    node. Iterated to a fixpoint so cyclic equivalences converge."""
    best = {}
    for _ in range(200):
      changed = False
      for c in list(self.parent):
        if self.find(c) != c:
          continue
        for op, ch, data in self.nodes[c]:
          if any(self.find(x) not in best for x in ch):
            continue
          cost = costs.get(op, 1) + sum(best[self.find(x)][0] for x in ch)
          if c not in best or cost < best[c][0]:
            term = ((op, data) if op in ("lit", "sym")
                    else (op,) + tuple(best[self.find(x)][1] for x in ch))
            best[c] = (cost, term)
            changed = True
      if not changed:
        break
    return best[self.find(eclass)][1]
```

The cost model is just a number per operation — `exp` is dear, `+` is cheap
— so extraction naturally prefers `e^{3x}` (one `exp`) over `e^x·e^x·e^x`
(three). Why iterate to a fixpoint instead of one recursive pass? Because
e-graphs can contain **cycles** — and the next subsection shows a real one.
(This greedy per-class choice is optimal for tree-shaped costs like ours; if
you want to account for sharing — a subterm used twice is paid for once —
extraction becomes NP-hard, and serious tools reach for ILP solvers, the
same hammer Chapter 23 used for noise placement.)

### 2.6 Look inside

[`inspect_egraph.py`](examples/inspect_egraph.py) saturates `(a*2)/2` with
both rules from section 1 and prints every live e-class (`cN` is an e-class
id; inside the braces are its e-nodes, whose children are e-class ids):

```
after add: root = c3
  c0: { a }
  c1: { 2 }
  c2: { *(c0, c1) }
  c3: { /(c2, c1) }

after saturate: root = c3
  c1: { 2 }
  c2: { *(c3, c1), shl(c3, c4) }
  c3: { /(c2, c1), a }
  c4: { 1 }

extract(root) = a
```

Read the saturated graph carefully:

- `c3` now holds both `/(c2, c1)` and `a`: the cancellation rule merged the
  `a` class (`c0`) into the root, so `c0` disappears as a separate class.
- `c2` holds both `*(...)` **and** `shl(...)`: strength reduction *added* a
  form; it did not replace `a*2`. That is why cancellation could still fire.
- Because `a` now lives in `c3`, `c2`'s children point at `c3` — the class
  that contains `c2`'s own parent. The graph has a **cycle**: `a = (a*2)/2 =
  ((a*2)/2*2)/2 = ...`, infinitely many terms in four classes. Extraction
  handles it by building up costs from the leaves (`a` costs 0) until
  nothing improves.

## 3. The concept demos (`optimize_demo.py`)

Each builds an expression, saturates a small rule set, and extracts the
cheapest form:

```
Equality saturation — extracting the cheapest equivalent form:
  (a*2)/2           ->  a
  e^x * e^x * e^x   ->  exp(((x + x) + x))
  (A^T)^T           ->  A
All e-graph demos succeeded.
```

The first is the phase-ordering example: the e-graph holds `a*2`, `a<<1`,
*and* the cancelled `a` simultaneously, so extraction simply reads off `a` —
no rule ordering required. The second shows the **cost model** doing the
deciding: with `exp` priced at 40 and `+` and `*` at 1, the single-`exp` form
`exp((x+x)+x)` (cost 42) wins over three exps (cost 122), and over the
in-between `exp(x+x) * exp(x)` (82) — even though it *adds* cheap `+`s. The
third is the kind of tensor identity that lets you delete an operation
`linalg` would otherwise dutifully compute.

## 4. Capstone: optimize → emit MLIR → execute

This closes the loop from expression to running code — the toy compiler
Diehl's article builds, in miniature. We optimize

```
f(a) = ((a*2)/2) * ((a*2)/2)      # 5 ops: 2 mul + 2 div + 1 mul
```

The rule `(x*y)/y → x` cancels each half to `a`, so the e-graph extracts

```
f(a) = a * a                      # 1 op
```

([`capstone.py`](examples/capstone.py) uses costs `*` = 1, `/` = 5, so the
original costs 13 and the extracted form 1.)

### Emitting MLIR

[`to_mlir.py`](examples/to_mlir.py) is the last stage of the toy compiler:
free symbols become `f32` arguments (in order of first appearance), literals
become `arith.constant`, and each binary node becomes the matching `arith`
op (`+ - * /` → `addf subf mulf divf`), emitted bottom-up with fresh SSA
names:

***examples/to_mlir.py*** (excerpt)
```python
  def emit(t):
    if t[0] == "sym":
      return "%" + t[1]
    if t[0] == "lit":
      r = fresh()
      lines.append(f"  {r} = arith.constant {float(t[1]):.6e} : f32")
      return r
    op = _BINOP[t[0]]
    a, b = emit(t[1]), emit(t[2])
    r = fresh()
    lines.append(f"  {r} = {op} {a}, {b} : f32")
    return r
```

`capstone.py` emits MLIR for **both** forms. The contrast is the whole point
— five arithmetic ops become one. Before:

***examples/build/orig.mlir*** (the 5-op original)
```mlir
func.func @f(%a: f32) -> f32 {
  %v1 = arith.constant 2.000000e+00 : f32
  %v2 = arith.mulf %a, %v1 : f32
  %v3 = arith.constant 2.000000e+00 : f32
  %v4 = arith.divf %v2, %v3 : f32
  %v5 = arith.constant 2.000000e+00 : f32
  %v6 = arith.mulf %a, %v5 : f32
  %v7 = arith.constant 2.000000e+00 : f32
  %v8 = arith.divf %v6, %v7 : f32
  %v9 = arith.mulf %v4, %v8 : f32
  return %v9 : f32
}
```

after:

***examples/build/opt.mlir*** (the extracted 1-op form)
```mlir
func.func @f(%a: f32) -> f32 {
  %v1 = arith.mulf %a, %a : f32
  return %v1 : f32
}
```

(The emitter walks the *term*, not the e-graph, so the original repeats the
shared half; `to_mlir` does no CSE of its own.)

### What MLIR's own rewriter makes of the original

For contrast, `build.sh` also runs MLIR's greedy canonicalizer and CSE over
the original:

```bash
mlir-opt build/orig.mlir --canonicalize --cse
```

```mlir
module {
  func.func @f(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = arith.mulf %arg0, %cst : f32
    %1 = arith.divf %0, %cst : f32
    %2 = arith.mulf %1, %1 : f32
    return %2 : f32
  }
}
```

CSE did the hash-consing part (one constant, one copy of the half), but the
`mulf`/`divf` pair survives. That is not timidity: under IEEE semantics
`(a*2)/2 → a` is **not** a valid rewrite — for `a = 3e38`, `a*2` overflows
to `inf` and `inf/2` is `inf`, not `3e38`. (We also tried adding
`fastmath<fast>` to the ops; the canonicalizer still leaves them alone.) The
e-graph applied the rule because *we told it* the rule holds — e-graphs make
rules safe to *combine*, not safe to *have*. Choosing sound rules for your
domain (exact reals? floats in a known range? integers that can't overflow?)
is still your job, exactly as with a Chapter 16 DRR pattern. For this
capstone's inputs, the identity holds.

### Compiling and running

`build.sh` compiles each version to a shared library with the now-familiar
pipeline — `mlir-opt` (arith and func to the `llvm` dialect) →
`mlir-translate --mlir-to-llvmir` → `ll_fp_compat.py` → `clang -shared`:

***examples/build.sh*** (excerpt)
```bash
for name in orig opt; do
    mlir-opt "$BUILD/${name}.mlir" \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$BUILD/${name}_llvm.mlir"
    mlir-translate "$BUILD/${name}_llvm.mlir" --mlir-to-llvmir -o "$BUILD/${name}.ll"
    python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$BUILD/${name}.ll" > "$BUILD/${name}.compat.ll"
    clang -shared -fPIC -Wno-override-module -x ir "$BUILD/${name}.compat.ll" -o "$BUILD/lib${name}.so"
done
```

The optimized function's LLVM IR is as small as you'd hope:

***examples/build/opt.compat.ll*** (excerpt)
```llvm
define float @f(float %0) {
  %2 = fmul float %0, %0
  ret float %2
}
```

Since `f` takes and returns a scalar `f32` — no memrefs — there is no
descriptor to build: [`run_compiled.py`](examples/run_compiled.py) loads
both libraries with `ctypes`, declares `f` as `float(float)`, and calls both
over a range of inputs to confirm the rewrite **preserved meaning** while
cutting 5 operations to 1.

The capstone part of `build.sh`'s output:

```
=== capstone: optimize an expression, emit MLIR ===
original :  (((a * 2) / 2) * ((a * 2) / 2))   (5 ops)
optimized:  (a * a)   (1 ops)
Wrote orig.mlir and opt.mlir to part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/build/
```

```
=== run both compiled functions and compare ===
  f(  -3.00):  original =     9.0000   optimized =     9.0000
  f(   0.00):  original =     0.0000   optimized =     0.0000
  f(   2.50):  original =     6.2500   optimized =     6.2500
  f(   7.00):  original =    49.0000   optimized =    49.0000
  f( 100.00):  original = 10000.0000   optimized = 10000.0000
Capstone OK: optimized f(a) == original f(a) == a*a for all test inputs.
```

## 5. E-graphs and the rest of this tutorial

- **Versus greedy rewriting (Chapters 10, 16, 17).** A pattern for the
  greedy driver and a rule for the e-graph look alike — a left-hand shape, a
  right-hand shape — and the rules in this chapter could be written as DRR
  `Pat`s or PDLL `Pattern`s. The difference is the *driver*: greedy
  rewriting commits to each match immediately and needs benefits, pass
  order, and canonical forms to steer it; saturation explores all matches and
  defers the decision to extraction, where a global cost model makes it.
  The price is memory and time: the e-graph holds every form at once, and
  some rule sets (associativity + commutativity) blow up exponentially.
- **Versus canonicalization.** Canonicalization (Chapter 16) picks one normal
  form *per rule* so that equivalent IR looks identical. An e-graph doesn't
  need a normal form — equivalent forms share an e-class, and "looks
  identical" is answered by `find`.
- **Versus dataflow analysis (Chapter 23).** Both are fixed-point
  iterations over a growing body of facts. Dataflow facts are lattice values
  attached to SSA values that only move up the lattice; e-graph facts are
  equalities that only merge classes. Neither ever "un-learns" a fact, which
  is why both terminate (when the e-graph's rules permit it).
- **Where it lives in practice.** Equality saturation was introduced as a
  compiler optimization technique by Tate, Stepp, Tatlock and Lerner (POPL
  2009). The `egg` library (Willsey et al., POPL 2021) made it fast and
  practical with deferred rebuilding and e-class analyses, and `egglog`
  (PLDI 2023) — the library Diehl's article uses — unifies it with Datalog.
  In production compilers, Cranelift's mid-end optimizer is built on a
  restricted variant, the *acyclic* e-graph ("aegraph"). MLIR's core
  rewriting infrastructure is greedy and destructive, as you have seen;
  representing e-graphs inside MLIR itself has been explored in research,
  but this chapter's approach — saturate on a small expression language, then
  *emit* MLIR — needs nothing beyond what upstream MLIR already provides.

## Key takeaways

- **E-graphs solve phase ordering** by keeping every equivalent form at once,
  then extracting the best — so rewrite order stops mattering.
- **Equality saturation** = apply all rules to a fixpoint (`saturate`);
  **extraction** = pick the cheapest form under a cost model (`extract`); a
  rewrite only ever `merge`s, never deletes.
- **Union-find + hashcons + rebuild** are the data-structure core: classes
  merge cheaply, identical e-nodes are shared (CSE for free), and congruence
  is restored in bulk.
- **This is high-level, domain-aware optimization** that LLVM/`linalg` can't
  do because they lack the algebraic identities (and, for floats, rightly
  refuse unsound ones) — it runs on the core language *before* lowering.
- **It still ends in MLIR**: the optimized expression is emitted as `arith`
  ops and compiled, the same "core → MLIR → LLVM" path as the rest of this
  tutorial.

## Exercises

1. **Commutativity rescues a miss.** Change the capstone's expression to
   `((2*a)/2) * ((2*a)/2)` (constant on the left). With only the
   cancellation rule, the e-graph extracts the original unchanged — the
   pattern `(x*y)/y` needs the divisor on the right of the `*`. Add the rule
   `x*y → y*x` and rerun: it now extracts `(a * a)`. Why is adding
   commutativity *safe* in an e-graph but an infinite loop for a naive
   greedy rewriter? (Compare Chapter 17's `Lhs`/`Rhs` pattern twins, which
   exist precisely because the greedy driver can't afford this rule.)
2. **Fix the stuck rewriter.** In [`greedy_demo.py`](examples/greedy_demo.py),
   add a third rule `(x<<1)/2 → x`. Both orders now reach `a`. Then explain
   why this "patch the stuck term" approach doesn't scale: how many extra
   rules would you need for `(a*4)/4` with a strength reduction `x*4 → x<<2`?
3. **Measure the blowup.** Saturate a left-nested sum
   `x0 + x1 + ... + x(n-1)` with commutativity `x+y → y+x` and associativity
   `(x+y)+z → x+(y+z)`, and print the number of live e-classes and e-nodes
   for n = 2..6. You should see 3, 7, 15, 31, 63 classes (2ⁿ − 1, one per
   non-empty subset of the variables) and 4, 15, 54, 185, 608 e-nodes. How
   far can you push n before `saturate` hits its 30-iteration cap or becomes
   too slow?
4. **Constant folding as a dynamic rule.** `_instantiate` accepts a callable
   right-hand side, `rhs(egraph, subst)`, which must return an e-class id.
   Write a rule with left-hand side `("*", Var("p"), Var("q"))` whose callable
   looks for `lit` e-nodes in the two bound classes and, if both have one,
   adds their product as a new `lit` (otherwise it can just re-add the
   matched `*` node). Together with an associativity rule
   `(x*p)*q → x*(p*q)`, `(a*2)*3` should extract `(a * 6)`. Why is the
   associativity rule necessary — what does the e-graph look like without it?
5. **Soundness.** Emit MLIR for just one half, `(a*2)/2`, compile both
   versions as in `build.sh`, and call them at `a = 3e38`. The original
   returns `inf`; the "optimized" `a` returns `3e38`. (The full capstone
   happens to agree at that input — work out why.) Then find an input where
   the rule `(x*y)/y → x` is wrong because of `y = 0`, and propose a guard
   (a condition on the rule, like a DRR constraint) that makes it sound.
6. **Add a lowering for the tensor demo.** The `(A^T)^T → A` demo never
   reaches MLIR. Extend `to_mlir.py` to emit `linalg.transpose` on a fixed
   `tensor<4x8xf32>` argument for `T`, and confirm that the optimized term
   produces a function with no `linalg` ops at all.

## Notes on reproducing with current MLIR

The source chapter was written against LLVM 20 (Homebrew `llvm@20`). With
this repo's MLIR wheel (LLVM 24):

- **No `llc`.** The source compiled `.ll` → `.o` with `llc -filetype=obj
  --relocation-model=pic`, then linked with `clang -shared`. The wheel ships
  no `llc`, so `build.sh` hands the IR directly to
  `clang -shared -fPIC -x ir`, after
  [`tools/ll_fp_compat.py`](../../tools/ll_fp_compat.py) (LLVM 24 prints
  float literals in a syntax older system clangs reject — see Chapter 28).
  For this capstone the filter happens to be a no-op (the only float
  literal, `2.0`, appears as `fdiv float %2, 2.000000e+00`, a form older
  clangs accept) but is kept so the pipeline is the standard one. `-Wno-override-module` silences clang's note that it replaces the
  module's target triple with the host's.
- **Pass names unchanged.** `--convert-arith-to-llvm`,
  `--convert-func-to-llvm` and `--reconcile-unrealized-casts` still exist
  with the same spelling, and the emitted `orig.mlir`/`opt.mlir` are
  byte-identical to the source's.
- **Paths and names.** Libraries are named `lib*.so` on every platform
  (`ctypes` loads them fine on macOS), the source's `aot_main.py` is
  `run_compiled.py`, and `capstone.py`/`run_compiled.py` take the output
  directory as an optional argument (so the lit test can use `%t`).
- **Additions.** `greedy_demo.py`, `inspect_egraph.py`, `EGraph.dump()`, the
  `--canonicalize --cse` comparison, and the per-input printout in
  `run_compiled.py` are new in this port; the e-graph algorithm itself is
  unchanged.

## Where to go next

That ends Part 5. [Part 6](../../part6-end-to-end-ml/README.md) turns to
end-to-end ML, starting with
[Chapter 25: Neural Networks from Scratch](../../part6-end-to-end-ml/ch25-neural-nets-from-scratch/README.md)
— and when you reach
[Chapter 26](../../part6-end-to-end-ml/ch26-transformers-and-softmax/README.md)'s
transformers, keep this chapter's question in mind: which algebraic
identities (`(A^T)^T = A`, `e^p · e^q = e^{p+q}`, softmax's shift
invariance) would you want applied *before* the graph becomes loops?
