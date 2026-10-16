# Chapter 16 code: the C++ pattern, before DRR

[`DifferenceOfSquares.cpp`](DifferenceOfSquares.cpp) contains the C++
`DifferenceOfSquares` rewrite pattern exactly as
[Chapter 16 §3](../README.md) presents it — the original version, which
the repo has since replaced with the DRR pattern in
`lib/Dialect/Poly/PolyPatterns.td` (Chapter 16 §4).

Two pieces of scaffolding around the pattern are **not** part of the
original:

- The pattern was originally registered in
  `SubOp::getCanonicalizationPatterns`, so it ran inside `--canonicalize`.
  Here it is wrapped in a standalone `PassWrapper` pass (flag:
  `--difference-of-squares`) so you can run it without editing the Poly
  dialect in `lib/`.
- One line differs from the original listing:

  ```cpp
  rewriter.replaceOp(op, {newMul});  // original
  rewriter.replaceOp(op, newMul);    // here
  ```

  The braced form is ambiguous (`ValueRange` vs. `Operation*`) against
  current MLIR and no longer compiles. A source comment marks the spot.

Run (verified; the normal top-level build already produced the binary,
or rebuild only it with `cmake --build build --target tutorial-opt-ch16`):

```bash
build/part3-building-a-dialect/ch16-canonicalizers-drr/code/tutorial-opt-ch16 \
  --difference-of-squares test/part3/poly_canonicalize.mlir
```

On that test file, `@test_difference_of_squares` is rewritten to
`poly.add` / `poly.sub` / `poly.mul` — the same result as the DRR version
under `--canonicalize` — and `@test_difference_of_squares_other_uses` is
left untouched (the `hasOneUse` guard declines). The conj-through-eval
function is also untouched: that is `EvalOp`'s canonicalization pattern,
which this pass deliberately does not include. `@test_simple` still gets
constant-folded, because the greedy driver runs folders even when no
pattern matches — Chapter 10's "AndFold" behavior.
