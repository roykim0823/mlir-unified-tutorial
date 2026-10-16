// Chapter 8: the generic matmul lowers to the same 3-deep nest in two loop
// dialects -- scf (bounds are SSA constants) or affine (bounds are literals).
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/03-matmul/matmul_generic.mlir --convert-linalg-to-loops | FileCheck %s --check-prefix=SCF
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/03-matmul/matmul_generic.mlir --convert-linalg-to-affine-loops | FileCheck %s --check-prefix=AFFINE
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/03-matmul/matmul_generic.mlir --linalg-specialize-generic-ops | FileCheck %s --check-prefix=SPEC

// SCF:      scf.for %[[I:.*]] = %c0 to %c8 step %c1 {
// SCF-NEXT:   scf.for %[[J:.*]] = %c0 to %c16 step %c1 {
// SCF-NEXT:     scf.for %[[K:.*]] = %c0 to %c10 step %c1 {
// SCF-NEXT:       memref.load %arg0[%[[I]], %[[K]]]
// SCF-NEXT:       memref.load %arg1[%[[K]], %[[J]]]
// SCF-NEXT:       memref.load %arg2[%[[I]], %[[J]]]

// AFFINE:      affine.for %[[I:.*]] = 0 to 8 {
// AFFINE-NEXT:   affine.for %[[J:.*]] = 0 to 16 {
// AFFINE-NEXT:     affine.for %[[K:.*]] = 0 to 10 {
// AFFINE-NEXT:       affine.load %arg0[%[[I]], %[[K]]]
// AFFINE-NEXT:       affine.load %arg1[%[[K]], %[[J]]]
// AFFINE-NEXT:       affine.load %arg2[%[[I]], %[[J]]]

// The generic is recognized as a matmul again.
// SPEC: linalg.matmul ins(%arg0, %arg1
