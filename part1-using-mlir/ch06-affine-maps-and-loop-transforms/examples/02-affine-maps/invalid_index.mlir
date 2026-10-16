// NOT a valid affine program -- build.sh expects mlir-opt to reject it.
//
// Every index of an affine.load/affine.store must be a valid affine dim or
// symbol (a loop induction variable, a function argument, a constant, or an
// affine.apply of those). An arith.muli result is opaque to the affine
// machinery, so the verifier refuses the access.
func.func @not_affine(%A: memref<16xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %ij = arith.muli %i, %j : index       // i * j: two variables multiplied
      %v = affine.load %A[%ij] : memref<16xf32>
      affine.store %v, %A[%ij] : memref<16xf32>
    }
  }
  return
}
