// Unroll-and-jam.
//
// Unroll the OUTER loop by 2, then "jam" the two copies of the inner loop
// back into one inner loop. Each inner iteration now handles rows i and i+1,
// so x[j] is loaded once per pair of rows instead of once per row -- in
// principle. (Matrix-vector product y += A * x.)
//
//   mlir-opt unroll_jam.mlir \
//     --pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))"
func.func @unroll_jam(%A: memref<8x16xf32>, %x: memref<16xf32>, %y: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 16 {
      %a = affine.load %A[%i, %j] : memref<8x16xf32>
      %b = affine.load %x[%j] : memref<16xf32>
      %c = affine.load %y[%i] : memref<8xf32>
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      affine.store %s, %y[%i] : memref<8xf32>
    }
  }
  return
}
