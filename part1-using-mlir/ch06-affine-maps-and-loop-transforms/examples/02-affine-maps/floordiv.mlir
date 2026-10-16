// What floordiv costs once it leaves the affine dialect.
//
// Hardware integer division (arith.divsi) truncates toward zero, but affine
// floordiv rounds toward -infinity. -lower-affine therefore wraps the divide
// in a sign fix-up; build.sh writes the result to build/floordiv_lowered.mlir.
func.func @floordiv4(%x: index) -> index {
  %r = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%x)
  return %r : index
}
