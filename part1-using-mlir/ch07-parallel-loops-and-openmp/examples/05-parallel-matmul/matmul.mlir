// Matrix multiply with the affine dialect: C += A * B.
//
// The same kernel as ../01-affine-parallelize/matmul.mlir (and Chapter 6's
// sequential affine matmul). This time build.sh does not stop at reading the
// parallel IR: it parallelizes the i/j loops, lowers them through scf.parallel
// to the omp dialect, and runs the result on several threads.
//
// The %i and %j loops are independent across iterations (each writes a
// distinct C[i, j]); the %k loop accumulates into C[i, j] and stays sequential.
//
// Exported as `_mlir_ciface_matmul` (see driver.c). The caller passes a
// zero-initialized C, since the kernel accumulates into it.
func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>   // rows of A (and C)
  %N = memref.dim %B, %c1 : memref<?x?xf32>   // cols of B (and C)
  %K = memref.dim %A, %c1 : memref<?x?xf32>   // shared/contraction dimension

  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%i, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %j] : memref<?x?xf32>
        %c = affine.load %C[%i, %j] : memref<?x?xf32>
        %prod = arith.mulf %a, %b : f32
        %sum  = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return
}
