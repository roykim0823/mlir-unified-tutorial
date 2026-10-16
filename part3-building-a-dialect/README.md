# Part 3: Building a Dialect

The heart of the tutorial: `poly`, a dialect for polynomial arithmetic,
built up one capability at a time — and with it, most of MLIR's
dialect-craftsmanship surface. Each chapter adds machinery the previous
one made you want.

| Chapter | One line |
|---|---|
| [ch12: Defining the Poly Dialect](ch12-defining-poly/README.md) | ODS: a dialect, a custom type, ops, and the generated C++ |
| [ch13: Using Traits](ch13-traits/README.md) | `Pure`, `Commutative`, verifier-carrying markers — and what interfaces are |
| [ch14: Folders and Constant Propagation](ch14-folders/README.md) | `fold()` methods; how `--sccp` and canonicalization eat constants |
| [ch15: Verifiers](ch15-verifiers/README.md) | Structural invariants that make broken IR fail loudly |
| [ch16: Canonicalizers and DRR](ch16-canonicalizers-drr/README.md) | Preferred-form rewrites, in C++ and in tablegen's pattern language |
| [ch17: Patterns with PDLL](ch17-pdll/README.md) | The third pattern language: patterns compiled to IR and interpreted |

**Prerequisites:** Part 2. **Next:**
[Part 4](../part4-lowering/README.md) lowers `poly` out of existence — the
one thing no pattern language here could do (none of them change types).
The arc's finale is
[ch23](../part5-structured-and-parallel/ch23-dataflow-analysis/README.md),
where a second dialect gets an operations-research-grade optimizer.
