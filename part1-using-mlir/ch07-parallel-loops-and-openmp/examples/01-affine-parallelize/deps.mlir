// Six small affine loops. Which ones may run their iterations in parallel?
//
// --affine-parallelize answers that with exact dependence analysis: a loop
// becomes affine.parallel only if no iteration writes memory that another
// iteration reads or writes.
//
//   mlir-opt deps.mlir --affine-parallelize
//   mlir-opt deps.mlir --affine-parallelize=parallel-reductions=true

// B[i] = 2 * A[i]: every iteration touches its own element. PARALLEL.
func.func @scale(%A: memref<100xf32>, %B: memref<100xf32>) {
  %two = arith.constant 2.0 : f32
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i] : memref<100xf32>
    %r = arith.mulf %v, %two : f32
    affine.store %r, %B[%i] : memref<100xf32>
  }
  return
}

// A[i] += A[i-1]: iteration i reads what iteration i-1 wrote (a true,
// read-after-write dependence carried by the loop). SEQUENTIAL.
func.func @prefix_sum(%A: memref<100xf32>) {
  affine.for %i = 1 to 100 {
    %prev = affine.load %A[%i - 1] : memref<100xf32>
    %cur  = affine.load %A[%i] : memref<100xf32>
    %s = arith.addf %prev, %cur : f32
    affine.store %s, %A[%i] : memref<100xf32>
  }
  return
}

// B[i] = A[i+1]: reads and writes go to different buffers. PARALLEL.
func.func @shift_out_of_place(%A: memref<101xf32>, %B: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i + 1] : memref<101xf32>
    affine.store %v, %B[%i] : memref<100xf32>
  }
  return
}

// A[i] = A[i+1]: the same shift in place. Iteration i reads A[i+1] *before*
// iteration i+1 overwrites it (a write-after-read, or anti, dependence).
// Running the iterations in a different order would read already-shifted
// values. SEQUENTIAL.
func.func @shift_in_place(%A: memref<101xf32>) {
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i + 1] : memref<101xf32>
    affine.store %v, %A[%i] : memref<101xf32>
  }
  return
}

// B[i] = A[Idx[i]]: a data-dependent (gather) index is not affine, so it must
// be a plain memref.load. Here the loop only *reads* A, so it would be safe,
// but the analysis cannot see through memref.load and stays conservative.
// SEQUENTIAL.
func.func @gather(%A: memref<100xf32>, %Idx: memref<100xindex>, %B: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    %j = affine.load %Idx[%i] : memref<100xindex>
    %v = memref.load %A[%j] : memref<100xf32>
    affine.store %v, %B[%i] : memref<100xf32>
  }
  return
}

// sum(A): a reduction. The loop-carried value %acc is a dependence, but an
// associative one (addf), so the iterations CAN be split up if the partial
// sums are combined at the end. PARALLEL only with parallel-reductions=true.
func.func @sum(%A: memref<100xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %s = affine.for %i = 0 to 100 iter_args(%acc = %zero) -> (f32) {
    %v = affine.load %A[%i] : memref<100xf32>
    %n = arith.addf %acc, %v : f32
    affine.yield %n : f32
  }
  return %s : f32
}
