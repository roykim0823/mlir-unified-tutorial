// Chapter 8: a reduction is a generic whose OUTPUT map drops a dimension and
// whose region reads %out. The generic lowers to an outer parallel loop and an
// inner accumulating loop; the named linalg.reduce generalizes to the same maps.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/02-reduce/reduce_generic.mlir --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-loops | FileCheck %s --check-prefix=LOOPS
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/02-reduce/reduce.mlir --linalg-generalize-named-ops | FileCheck %s --check-prefix=GEN

// LOOPS:      scf.for %[[I:.*]] = %c0 to %c3 step %c1 {
// LOOPS-NEXT:   scf.for %[[J:.*]] = %c0 to %c4 step %c1 {
// LOOPS-NEXT:     %[[X:.*]] = memref.load %arg0[%[[I]], %[[J]]]
// LOOPS-NEXT:     %[[ACC:.*]] = memref.load %arg1[%[[I]]]
// LOOPS-NEXT:     %[[S:.*]] = arith.addf %[[ACC]], %[[X]] : f32
// LOOPS-NEXT:     memref.store %[[S]], %arg1[%[[I]]]

// GEN-DAG: #[[IN:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// GEN-DAG: #[[OUT:.*]] = affine_map<(d0, d1) -> (d0)>
// GEN:     linalg.generic {indexing_maps = [#[[IN]], #[[OUT]]], iterator_types = ["parallel", "reduction"]}
// GEN:     ^bb0(%[[IN_E:.*]]: f32, %[[OUT_E:.*]]: f32):
// GEN-NEXT:  arith.addf %[[OUT_E]], %[[IN_E]] : f32
