// Chapter 7: dependence analysis with --affine-parallelize, and keeping the
// parallelism through -lower-affine (affine.parallel -> scf.parallel).
// Inputs live in the chapter's examples/01-affine-parallelize directory.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/deps.mlir --affine-parallelize | FileCheck %s --check-prefix=DEPS
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/deps.mlir --affine-parallelize=parallel-reductions=true | FileCheck %s --check-prefix=RED
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/matmul.mlir --affine-parallelize | FileCheck %s --check-prefix=MM
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/matmul.mlir --affine-parallelize=max-nested=1 | FileCheck %s --check-prefix=OUTER
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/matmul.mlir --affine-parallelize --lower-affine --canonicalize | FileCheck %s --check-prefix=SCF
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/matmul.mlir --affine-parallelize --lower-affine --canonicalize --convert-scf-to-cf | FileCheck %s --check-prefix=SERIAL

// Independent iterations become affine.parallel; loop-carried dependences,
// non-affine accesses and (by default) reductions stay affine.for.
// DEPS-LABEL: func.func @scale
// DEPS:         affine.parallel (%{{.*}}) = (0) to (100)
// DEPS-LABEL: func.func @prefix_sum
// DEPS-NOT:     affine.parallel
// DEPS:         affine.for
// DEPS-LABEL: func.func @shift_out_of_place
// DEPS:         affine.parallel
// DEPS-LABEL: func.func @shift_in_place
// DEPS-NOT:     affine.parallel
// DEPS:         affine.for
// DEPS-LABEL: func.func @gather
// DEPS-NOT:     affine.parallel
// DEPS:         affine.for
// DEPS-LABEL: func.func @sum
// DEPS-NOT:     affine.parallel
// DEPS:         affine.for {{.*}} iter_args

// RED-LABEL: func.func @sum
// RED:         affine.parallel (%{{.*}}) = (0) to (100) reduce ("addf") -> (f32)
// RED:           affine.yield

// i and j are parallel, the k reduction (through memory) stays sequential.
// MM:      affine.parallel
// MM-NEXT:   affine.parallel
// MM-NEXT:     affine.for

// OUTER:     affine.parallel
// OUTER-NOT: affine.parallel
// OUTER:       affine.for
// OUTER-NEXT:    affine.for

// lower-affine keeps the parallelism; canonicalize merges the nest into a 2-D band.
// SCF:     scf.parallel (%{{.*}}, %{{.*}}) = (%c0, %c0) to (%dim, %dim_0) step (%c1, %c1)
// SCF-NEXT:  scf.for
// SCF:       scf.reduce

// The default CPU route serializes it.
// SERIAL-NOT: scf.parallel
// SERIAL:     cf.cond_br

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/matmul.mlir --affine-parallelize --lower-affine --canonicalize --gpu-map-parallel-loops --convert-parallel-loops-to-gpu | FileCheck %s --check-prefix=GPU
// The same scf.parallel can instead become a GPU launch (Chapter 31).
// GPU: gpu.launch blocks
// GPU:   scf.for
// GPU:   gpu.terminator
