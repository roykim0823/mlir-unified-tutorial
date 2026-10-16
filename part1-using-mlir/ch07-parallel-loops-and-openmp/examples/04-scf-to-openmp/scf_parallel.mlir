// scf.parallel -> OpenMP.
//
// You rarely write the omp dialect by hand. The usual route is to express
// parallelism with scf.parallel (by hand, or via --affine-parallelize followed
// by --lower-affine, as in example 01) and let --convert-scf-to-openmp rewrite
// it into omp.parallel + omp.wsloop:
//
//   mlir-opt scf_parallel.mlir --convert-scf-to-openmp
//
// Both functions get C interface wrappers (_mlir_ciface_*) so driver.c can
// call them with memref descriptors.

// B[i] = A[i] + A[i] for 100 elements.
func.func @add(%A: memref<100xf32>, %B: memref<100xf32>)
    attributes {llvm.emit_c_interface} {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c100 = arith.constant 100 : index
  scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
    %v = memref.load %A[%i] : memref<100xf32>
    %r = arith.addf %v, %v : f32
    memref.store %r, %B[%i] : memref<100xf32>
    scf.reduce
  }
  return
}

// The OpenMP runtime's own API is just C functions; declare one and call it.
func.func private @omp_get_thread_num() -> i32

// owner[i] = the id of the thread that ran iteration i. Makes the work
// distribution visible.
func.func @which_thread(%owner: memref<?xi32>)
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n  = memref.dim %owner, %c0 : memref<?xi32>
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %t = func.call @omp_get_thread_num() : () -> i32
    memref.store %t, %owner[%i] : memref<?xi32>
    scf.reduce
  }
  return
}
