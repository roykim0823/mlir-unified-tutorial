// Chapter 6: affine maps and affine.apply (examples/02-affine-maps).
//
// -canonicalize folds every affine.apply on constants, composes chained maps
// on a runtime value, and deletes unused applies.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/02-affine-maps/affine_apply.mlir -canonicalize | FileCheck %s --check-prefix=FOLD
// FOLD: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 2 + 5)>
// FOLD-LABEL: func.func @affine_apply_examples
// FOLD-DAG: %[[C1:.*]] = arith.constant 1 : index
// FOLD-DAG: %[[C52:.*]] = arith.constant 52 : index
// FOLD-DAG: %[[C170:.*]] = arith.constant 170 : index
// FOLD-DAG: %[[C116:.*]] = arith.constant 116 : index
// FOLD-DAG: %[[C89:.*]] = arith.constant 89 : index
// FOLD: return %[[C1]], %[[C52]], %[[C170]], %[[C116]], %[[C89]]
// FOLD-LABEL: func.func @rounding
// FOLD-DAG: %[[FD:.*]] = arith.constant -2 : index
// FOLD-DAG: %[[CD:.*]] = arith.constant -1 : index
// FOLD-DAG: %[[MD:.*]] = arith.constant 1 : index
// FOLD: return %[[FD]], %[[CD]], %[[MD]]
// FOLD-LABEL: func.func @compose
// FOLD-NEXT: affine.apply #[[MAP]]()[%arg0]
// FOLD-LABEL: func.func @dead_apply
// FOLD-NEXT: return

// floordiv lowers to a truncating divide wrapped in a sign fix-up.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/02-affine-maps/floordiv.mlir -lower-affine | FileCheck %s --check-prefix=FLOOR
// FLOOR: arith.cmpi slt
// FLOOR: arith.select
// FLOOR: arith.divsi
// FLOOR: arith.select

// A min(...) upper bound becomes arith.minsi.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/02-affine-maps/tiled_loop.mlir -lower-affine | FileCheck %s --check-prefix=TILED
// TILED-LABEL: func.func @tiled_loop(
// TILED: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %c32
// TILED: arith.addi %[[I]], %{{.*}} : index
// TILED-LABEL: func.func @tiled_loop_dynamic
// TILED: scf.for %[[T:.*]] = %{{.*}} to %arg1 step %c32
// TILED: %[[UB:.*]] = arith.minsi %{{.*}}, %arg1 : index
// TILED: scf.for %{{.*}} = %[[T]] to %[[UB]]

// A non-affine subscript is rejected by the verifier.
// RUN: not mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/02-affine-maps/invalid_index.mlir 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: error: 'affine.load' op operand cannot be used as a dimension id
