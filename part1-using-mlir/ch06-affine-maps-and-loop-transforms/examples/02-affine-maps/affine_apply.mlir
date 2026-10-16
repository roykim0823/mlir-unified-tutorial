// Affine maps and affine.apply.
//
// An affine map is a multi-dimensional quasi-linear function from dimensions
// and symbols to `index` results. Notation: (d0, d1)[s0] -> (...) has dimension
// operands d0, d1 and symbol operand s0. Maps may use only +, -, * by a
// constant, and floordiv / ceildiv / mod by a constant.
//
// `affine.apply` evaluates a map on concrete operands -- tile indices, offsets,
// array addresses. build.sh runs this file through -canonicalize: every apply
// on constant operands folds to a constant, and the chained applies on a
// runtime value compose into a single map.

// Reusable named maps (the `#name` attribute alias from Chapter 1).
#tile_map    = affine_map<(d0) -> (d0 floordiv 32)>          // which 32-wide tile?
#offset_map  = affine_map<(d0)[s0] -> (d0 + s0)>             // shift by a symbol
#complex_map = affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 floordiv 4 + s0)>

func.func @affine_apply_examples() -> (index, index, index, index, index) {
  %c0   = arith.constant 0   : index
  %c42  = arith.constant 42  : index
  %c128 = arith.constant 128 : index

  // 1. Tiling: 42 floordiv 32 = 1 (element 42 lives in tile #1).
  %tile = affine.apply #tile_map(%c42)

  // 2. An inline (anonymous) map for an offset: 42 + 10 = 52.
  %off = affine.apply affine_map<(i) -> (i + 10)>(%c42)

  // 3. A symbol operand: 42 + 128 = 170. Dims go in (), symbols in [].
  %shift = affine.apply #offset_map(%c42)[%c128]

  // 4. Two dims and a symbol: 42*2 + 128 floordiv 4 + 0 = 116.
  %cplx = affine.apply #complex_map(%c42, %c128)[%c0]

  // 5. Composition: feed one apply's result into the next. (42*2) + 5 = 89.
  %t     = affine.apply affine_map<(i) -> (i * 2)>(%c42)
  %final = affine.apply affine_map<(i) -> (i + 5)>(%t)

  return %tile, %off, %shift, %cplx, %final : index, index, index, index, index
}

// floordiv rounds toward -infinity, ceildiv toward +infinity, and mod is the
// matching (always non-negative) remainder. Try them on a negative number:
//   -7 floordiv 4 = -2,  -7 ceildiv 4 = -1,  -7 mod 4 = 1
func.func @rounding() -> (index, index, index) {
  %m7 = arith.constant -7 : index
  %fd = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%m7)
  %cd = affine.apply affine_map<(d0) -> (d0 ceildiv 4)>(%m7)
  %md = affine.apply affine_map<(d0) -> (d0 mod 4)>(%m7)
  return %fd, %cd, %md : index, index, index
}

// With a runtime operand nothing can be constant-folded, but the two applies
// still compose into one map: (d0 * 2) + 5.
func.func @compose(%n: index) -> index {
  %t     = affine.apply affine_map<(i) -> (i * 2)>(%n)
  %final = affine.apply affine_map<(i) -> (i + 5)>(%t)
  return %final : index
}

// affine.apply has no side effects: if nobody uses the result, -canonicalize
// deletes it, and this function collapses to a bare `return`.
func.func @dead_apply() {
  %c42 = arith.constant 42 : index
  %unused = affine.apply #tile_map(%c42)
  return
}
