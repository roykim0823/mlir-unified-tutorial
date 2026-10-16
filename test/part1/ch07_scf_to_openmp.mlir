// Chapter 7: scf.parallel -> omp dialect with --convert-scf-to-openmp,
// including a collapsed 2-D band and reductions.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/04-scf-to-openmp/scf_parallel.mlir --convert-scf-to-openmp | FileCheck %s --check-prefix=ADD
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/05-parallel-matmul/matmul.mlir --affine-parallelize --lower-affine --canonicalize --convert-scf-to-openmp | FileCheck %s --check-prefix=MM
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/06-parallel-reduction/reduce.mlir --affine-parallelize=parallel-reductions=true --lower-affine --canonicalize --convert-scf-to-openmp | FileCheck %s --check-prefix=RED
// RUN: mlir-opt %s --convert-scf-to-openmp --verify-diagnostics

// ADD-LABEL: func.func @add
// ADD:         omp.parallel {
// ADD-NEXT:      omp.wsloop {
// ADD-NEXT:        omp.loop_nest (%{{.*}}) : index = (%c0) to (%c100) step (%c1) {
// ADD-NEXT:          memref.alloca_scope
// ADD:               omp.yield
// ADD:           omp.terminator

// MM: omp.loop_nest (%{{.*}}, %{{.*}}) : index = (%c0, %c0) to (%dim, %dim_0) step (%c1, %c1) collapse(2)
// MM:   scf.for

// RED:       omp.declare_reduction @__scf_reduction : f32 init
// RED:       } combiner {
// RED-NEXT:  ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// RED-NEXT:    arith.addf
// RED:       } atomic {
// RED:         llvm.atomicrmw fadd
// RED-LABEL: func.func @sum
// RED:         omp.wsloop reduction(@__scf_reduction %{{.*}} -> %{{.*}} : !llvm.ptr)
// RED:       omp.declare_reduction @__scf_reduction_0 : f32 init
// RED:       } combiner {
// RED:         arith.cmpf ogt
// RED-NEXT:    arith.select
// RED-LABEL: func.func @max
// RED:         omp.wsloop reduction(@__scf_reduction_0

// A max reduction spelled arith.maximumf (what --affine-parallelize emits) is
// not recognized by --convert-scf-to-openmp: the pass fails.
func.func @max_unsupported(%A: memref<?xf32>, %init: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %A, %c0 : memref<?xf32>
  %m = scf.parallel (%i) = (%c0) to (%n) step (%c1) init (%init) -> f32 {
    %v = memref.load %A[%i] : memref<?xf32>
    scf.reduce(%v : f32) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.maximumf %a, %b : f32
      // expected-error @+1 {{unconverted operation found}}
      scf.reduce.return %r : f32
    }
  }
  return %m : f32
}
