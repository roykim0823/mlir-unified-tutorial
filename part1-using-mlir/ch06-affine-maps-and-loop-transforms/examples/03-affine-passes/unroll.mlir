// Loop unrolling.
//
// Unrolling replicates the loop body to cut loop overhead and expose
// instruction-level parallelism. affine-loop-unroll is a function-level pass,
// so it must be nested under func.func:
//
//   mlir-opt unroll.mlir \
//     --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))"
//
// @unroll: 8 trips / 4 = 2 trips, body appears 4 times.
// @unroll_cleanup: 10 trips is not a multiple of 4, so the pass emits the
// unrolled main loop for 0..8 plus a 2-trip cleanup (epilogue) loop for 8..10.
func.func @unroll(%A: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i] : memref<8xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<8xf32>
  }
  return
}

func.func @unroll_cleanup(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %v = affine.load %A[%i] : memref<10xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<10xf32>
  }
  return
}
