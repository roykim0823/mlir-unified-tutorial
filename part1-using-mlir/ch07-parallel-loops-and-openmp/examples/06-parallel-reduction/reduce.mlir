// Reductions: loops whose iterations all feed ONE value.
//
// A reduction carries a value from iteration to iteration (affine.for
// iter_args + affine.yield), which is a dependence -- so plain
// --affine-parallelize leaves such loops alone. But when the combining
// operation is associative (and, for floats, we accept reassociation), the
// iterations can be split among threads: each thread folds its share into a
// private partial result, and the partials are combined at the end.

// sum(A), written as a sequential affine loop. build.sh parallelizes it
// automatically:
//   --affine-parallelize=parallel-reductions=true
//       affine.for iter_args  ->  affine.parallel ... reduce ("addf")
//   --lower-affine
//       ->  scf.parallel ... init (...) { scf.reduce(...) { arith.addf } }
//   --convert-scf-to-openmp
//       ->  omp.declare_reduction + omp.wsloop reduction(...)
func.func @sum(%A: memref<?xf32>) -> f32 attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %n = memref.dim %A, %c0 : memref<?xf32>
  %zero = arith.constant 0.0 : f32
  %s = affine.for %i = 0 to %n iter_args(%acc = %zero) -> (f32) {
    %v = affine.load %A[%i] : memref<?xf32>
    %r = arith.addf %acc, %v : f32
    affine.yield %r : f32
  }
  return %s : f32
}

// max(A), written directly as an scf.parallel reduction.
//
// Why not as an affine.for like @sum? --affine-parallelize would happily
// produce `reduce ("maximumf")`, but --convert-scf-to-openmp only recognizes
// a handful of combiners: addf/addi, mulf/muli, andi/ori/xori, and min/max
// spelled as cmpf/cmpi + select. An arith.maximumf combiner is left
// unconverted and the pass fails. So we spell the max as compare + select.
func.func @max(%A: memref<?xf32>) -> f32 attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %A, %c0 : memref<?xf32>
  // Start from the lowest finite float, -FLT_MAX. (-infinity is the
  // natural identity; LLVM 24 prints it as `float -inf`, which clang 20
  // only accepts after tools/ll_fp_compat.py rewrites it.)
  %lowest = arith.constant 0xFF7FFFFF : f32
  %m = scf.parallel (%i) = (%c0) to (%n) step (%c1) init (%lowest) -> f32 {
    %v = memref.load %A[%i] : memref<?xf32>
    scf.reduce(%v : f32) {
    ^bb0(%lhs: f32, %rhs: f32):
      %gt = arith.cmpf ogt, %lhs, %rhs : f32
      %r = arith.select %gt, %lhs, %rhs : f32
      scf.reduce.return %r : f32
    }
  }
  return %m : f32
}
