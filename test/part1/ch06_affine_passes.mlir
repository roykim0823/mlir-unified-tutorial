// Chapter 6: the affine optimization passes (examples/03-affine-passes).
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/licm.mlir -affine-loop-invariant-code-motion | FileCheck %s --check-prefix=LICM
// LICM: %[[CST:.*]] = arith.constant 4.200000e+01 : f32
// LICM-NEXT: affine.for
// LICM: arith.addf %{{.*}}, %[[CST]]

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/unroll.mlir --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))" | FileCheck %s --check-prefix=UNROLL
// UNROLL-LABEL: func.func @unroll(
// UNROLL: affine.for %{{.*}} = 0 to 8 step 4
// UNROLL-COUNT-4: arith.addf
// UNROLL-NOT: affine.for
// UNROLL-LABEL: func.func @unroll_cleanup(
// UNROLL: affine.for %{{.*}} = 0 to 8 step 4
// UNROLL: affine.for %{{.*}} = 8 to 10 {

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/unroll_jam.mlir --pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))" | FileCheck %s --check-prefix=JAM
// JAM: affine.for %[[I:.*]] = 0 to 8 step 2
// JAM-NEXT: affine.for %{{.*}} = 0 to 16 {
// JAM-COUNT-2: affine.store
// JAM-NEXT: }

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/normalize.mlir -affine-loop-normalize | FileCheck %s --check-prefix=NORM
// NORM: #[[MAP:.*]] = affine_map<(d0) -> (d0 * 5 + 10)>
// NORM: affine.for %[[I:.*]] = 0 to 18 {
// NORM-NEXT: affine.apply #[[MAP]](%[[I]])

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/scalrep.mlir -affine-scalrep | FileCheck %s --check-prefix=SCALREP
// SCALREP: %[[C:.*]] = arith.constant 1.000000e+00 : f32
// SCALREP-NOT: affine.load
// SCALREP: arith.addf %[[C]], %[[C]]
// SCALREP-COUNT-1: affine.store
// SCALREP-NOT: affine.store

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/coalescing.mlir -affine-loop-coalescing | FileCheck %s --check-prefix=COAL
// COAL: affine.for %[[K:.*]] = 0 to %{{.*}} {
// COAL-NEXT: %[[J:.*]] = affine.apply #{{.*}}(%[[K]])
// COAL-NEXT: %[[I:.*]] = affine.apply #{{.*}}(%[[K]])
// COAL-NEXT: affine.load %arg0[%[[I]], %[[J]]]
// COAL-NOT: affine.for

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/vectorize.mlir -affine-super-vectorize="virtual-vector-size=8" | FileCheck %s --check-prefix=VEC
// VEC: affine.for %{{.*}} = 0 to 256 step 8
// VEC: vector.transfer_read {{.*}} vector<8xf32>
// VEC: arith.addf {{.*}} : vector<8xf32>
// VEC: vector.transfer_write

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/fusion.mlir -affine-loop-fusion | FileCheck %s --check-prefix=FUSE
// FUSE-COUNT-1: affine.for
// FUSE-NOT: affine.for
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/fusion.mlir -affine-loop-fusion -affine-scalrep | FileCheck %s --check-prefix=FUSESR
// FUSESR: %[[M:.*]] = arith.mulf
// FUSESR-NEXT: affine.store %[[M]], %arg1
// FUSESR-NEXT: arith.addf %[[M]], %[[M]]

// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/tiling.mlir -affine-loop-tile="tile-size=8" | FileCheck %s --check-prefix=TILE
// TILE: affine.for %{{.*}} = 0 to 32 step 8
// TILE-NEXT: affine.for %{{.*}} = 0 to 32 step 8
// TILE-NEXT: affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}})
// TILE-NEXT: affine.for %{{.*}} = #{{.*}}(%{{.*}}) to #{{.*}}(%{{.*}})
