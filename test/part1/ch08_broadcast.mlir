// Chapter 8: linalg.broadcast is a generic whose input map ignores a loop
// dimension, (d0, d1) -> (d1). Both broadcast kernels -- named ops and the
// single hand-written generic -- are compiled and called from C.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/04-broadcast/add_vec_to_mat.mlir --linalg-generalize-named-ops | FileCheck %s --check-prefix=GEN
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/04-broadcast/add_vec_to_mat.mlir --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.add_vec_to_mat.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.add_vec_to_mat.ll > %t.add_vec_to_mat.compat.ll
// RUN: clang -Wno-override-module -x ir %t.add_vec_to_mat.compat.ll -c -o %t.add_vec_to_mat.o
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/04-broadcast/add_vec_to_mat_generic.mlir --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.add_vec_to_mat_generic.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.add_vec_to_mat_generic.ll > %t.add_vec_to_mat_generic.compat.ll
// RUN: clang -Wno-override-module -x ir %t.add_vec_to_mat_generic.compat.ll -c -o %t.add_vec_to_mat_generic.o
// RUN: clang %project_source_dir/part1-using-mlir/ch08-linalg-by-hand/examples/04-broadcast/bcast_main.c %t.add_vec_to_mat.o %t.add_vec_to_mat_generic.o -o %t.exe
// RUN: %t.exe | FileCheck %s --check-prefix=EXEC

// GEN-DAG: #[[ROW:.*]] = affine_map<(d0, d1) -> (d1)>
// GEN-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// GEN:     linalg.generic {indexing_maps = [#[[ROW]], #[[ID]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : memref<4xf32>)
// GEN:     linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]], iterator_types = ["parallel", "parallel"]}

// EXEC: named  : [2 3 4 5] [2 3 4 5] [2 3 4 5]
// EXEC: generic: [2 3 4 5] [2 3 4 5] [2 3 4 5]
