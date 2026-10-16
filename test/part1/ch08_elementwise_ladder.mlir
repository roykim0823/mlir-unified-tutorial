// Chapter 8: linalg.generic, linalg.map and linalg.add are three spellings of
// one elementwise add. The two tensor forms bufferize to the identical scf
// loop; the named memref op generalizes back to the identity-map generic.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/01-elementwise/generic_add.mlir --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-loops | FileCheck %s --check-prefix=LOOPS
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/01-elementwise/map_add.mlir --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-loops | FileCheck %s --check-prefix=LOOPS
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/01-elementwise/generic_add.mlir --convert-linalg-to-loops | FileCheck %s --check-prefix=NOBUF
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/01-elementwise/add.mlir --linalg-generalize-named-ops | FileCheck %s --check-prefix=GEN
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/01-elementwise/generic_add.mlir --linalg-specialize-generic-ops | FileCheck %s --check-prefix=SPEC

// LOOPS:      %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
// LOOPS-NEXT: scf.for %[[I:.*]] = %c0 to %c10 step %c1 {
// LOOPS-NEXT:   %[[A:.*]] = memref.load %arg0[%[[I]]]
// LOOPS-NEXT:   %[[B:.*]] = memref.load %arg1[%[[I]]]
// LOOPS-NEXT:   %[[S:.*]] = arith.addf %[[A]], %[[B]] : f32
// LOOPS-NEXT:   memref.store %[[S]], %[[ALLOC]][%[[I]]] : memref<10xf32>
// LOOPS-NEXT: }
// LOOPS-NEXT: return %[[ALLOC]] : memref<10xf32>

// Without bufferization the tensor op is left alone.
// NOBUF-NOT: scf.for
// NOBUF:     linalg.generic

// GEN:     #[[ID:.*]] = affine_map<(d0) -> (d0)>
// GEN:     linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]], iterator_types = ["parallel"]}
// GEN:     arith.addf
// GEN-NOT: linalg.add

// SPEC:     linalg.add ins(%arg0, %arg1
// SPEC-NOT: linalg.generic
