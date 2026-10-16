// Chapter 8: elementwise fusion through plain passes. The named add and mul
// only fuse after --linalg-generalize-named-ops; the result is one generic
// with four identity maps and both payload ops. Bufferized with
// --buffer-results-to-out-params, it is compiled and called from C.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/05-fusion/separate_ops.mlir --linalg-fuse-elementwise-ops | FileCheck %s --check-prefix=NOFUSE
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/05-fusion/separate_ops.mlir --canonicalize --linalg-fuse-elementwise-ops --cse --linalg-generalize-named-ops --linalg-fuse-elementwise-ops > %t.fused.mlir
// RUN: FileCheck %s --check-prefix=FUSED < %t.fused.mlir
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/05-fusion/broadcast_add.mlir --linalg-generalize-named-ops --linalg-fuse-elementwise-ops | FileCheck %s --check-prefix=BCAST
// RUN: mlir-opt %t.fused.mlir --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params="modify-public-functions=true hoist-static-allocs=true" --canonicalize > %t.memref.mlir
// RUN: FileCheck %s --check-prefix=OUTPARAM < %t.memref.mlir
// RUN: mlir-opt %t.memref.mlir --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -Wno-override-module -x ir %t.compat.ll -c -o %t.kernel.o
// RUN: clang %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/05-fusion/addmul_main.c %t.kernel.o -o %t.exe
// RUN: %t.exe | FileCheck %s --check-prefix=EXEC

// Fusion alone does nothing to named ops.
// NOFUSE: linalg.add
// NOFUSE: linalg.mul

// FUSED:     #[[ID:.*]] = affine_map<(d0) -> (d0)>
// FUSED:     tensor.empty
// FUSED-NOT: tensor.empty
// FUSED:     linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]], #[[ID]]], iterator_types = ["parallel"]}
// FUSED:     %[[SUM:.*]] = arith.addf
// FUSED:     %[[PROD:.*]] = arith.mulf %[[SUM]]
// FUSED:     linalg.yield %[[PROD]]
// FUSED-NOT: linalg.generic

// BCAST-DAG: #[[ID2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// BCAST-DAG: #[[ROW:.*]] = affine_map<(d0, d1) -> (d1)>
// BCAST:     linalg.generic {indexing_maps = [#[[ID2]], #[[ROW]], #[[ID2]]]
// BCAST-NOT: linalg.broadcast

// The returned tensor became a fourth (out) parameter and nothing is allocated.
// OUTPARAM:     func.func @addmul({{.*}}, %arg3: memref<10xf32>) attributes
// OUTPARAM-NOT: memref.alloc
// OUTPARAM:     outs(%arg3 : memref<10xf32>)
// OUTPARAM-NOT: memref.alloc

// EXEC: out = 2 6 12 20 30 42 56 72 90 110
// EXEC: fused addmul OK
