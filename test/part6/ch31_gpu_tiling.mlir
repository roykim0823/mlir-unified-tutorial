// ch31: tile-size tuning on the affine GPU route. The source repo tiled
// before mapping (--affine-loop-tile, "play with the tile size"); this checks
// that tiling itself is fine under LLVM 24 and pins down what actually breaks
// it: running affine-loop-invariant-code-motion first (which the source's
// double --pass-pipeline flag silently skipped - mlir-opt keeps only the last).
//
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/src/sample/sample_model_linalg.mlir --convert-tensor-to-linalg --linalg-generalize-named-ops --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation-pipeline --convert-bufferization-to-memref --llvm-request-c-wrappers --convert-linalg-to-affine-loops --affine-loop-tile="tile-size=4" --canonicalize --cse > %t.tiled.mlir
// RUN: mlir-opt %t.tiled.mlir --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' | FileCheck %s
// RUN: mlir-opt %t.tiled.mlir --pass-pipeline='builtin.module(func.func(affine-loop-invariant-code-motion))' --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' | FileCheck %s
// RUN: not mlir-opt %t.tiled.mlir --pass-pipeline='builtin.module(func.func(affine-loop-invariant-code-motion,convert-affine-for-to-gpu))' 2>&1 | FileCheck %s --check-prefix=LICM

// Tiled by 4: six nests, six launches; each thread runs the intra-tile loops.
// CHECK-COUNT-6: gpu.launch blocks
// CHECK-NOT: gpu.launch blocks

// LICM: error: expected perfectly nested loops in the body
