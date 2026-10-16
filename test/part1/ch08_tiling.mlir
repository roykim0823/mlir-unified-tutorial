// Chapter 8: tiling a linalg.matmul with plain passes -- bufferize, lower to
// affine loops, then --affine-loop-tile turns the 3-deep nest into 6 loops.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/06-tiling/matmul_tile.mlir --convert-tensor-to-linalg --linalg-generalize-named-ops --one-shot-bufferize="bufferize-function-boundaries" --buffer-deallocation-pipeline --convert-bufferization-to-memref --convert-linalg-to-affine-loops --affine-loop-tile="tile-size=5" --canonicalize --cse | FileCheck %s

// CHECK-DAG:  #[[LB:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:  #[[UB:.*]] = affine_map<(d0) -> (d0 + 5)>
// CHECK:      affine.for %[[I0:.*]] = 0 to 10 step 5 {
// CHECK-NEXT:   affine.for %[[J0:.*]] = 0 to 10 step 5 {
// CHECK-NEXT:     affine.for %[[K0:.*]] = 0 to 10 step 5 {
// CHECK-NEXT:       affine.for %{{.*}} = #[[LB]](%[[I0]]) to #[[UB]](%[[I0]]) {
// CHECK-NEXT:         affine.for %{{.*}} = #[[LB]](%[[J0]]) to #[[UB]](%[[J0]]) {
// CHECK-NEXT:           affine.for %{{.*}} = #[[LB]](%[[K0]]) to #[[UB]](%[[K0]]) {
// CHECK:                  arith.mulf
// CHECK:                  arith.addf
// The deallocation pipeline's clone of the returned argument, lowered:
// CHECK:      memref.alloc
// CHECK:      memref.copy %arg2
