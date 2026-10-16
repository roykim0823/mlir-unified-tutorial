// Chapter 4, step 4: llvm.emit_c_interface vs --llvm-request-c-wrappers on the
// 2-D broadcast kernel, and calling both conventions from C.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/04-c-wrappers/add_vector_to_matrix.mlir \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   -o %t.buf.mlir
// RUN: mlir-opt %t.buf.mlir --buffer-deallocation-pipeline --canonicalize --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm \
// RUN:   --finalize-memref-to-llvm --reconcile-unrealized-casts -o %t.llvm.mlir
// RUN: FileCheck %s --check-prefix=ATTR < %t.llvm.mlir
// ATTR-DAG: llvm.func @_mlir_ciface_add_vector_to_matrix(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
// ATTR-DAG: llvm.func @add_vector_to_matrix_into({{.*}}%arg18: i64) {
// ATTR-NOT: @_mlir_ciface_add_vector_to_matrix_into
//
// RUN: mlir-opt %t.buf.mlir --llvm-request-c-wrappers --buffer-deallocation-pipeline --canonicalize \
// RUN:   --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts \
// RUN:   | FileCheck %s --check-prefix=ALL
// ALL-DAG: llvm.func @_mlir_ciface_add_vector_to_matrix(
// ALL-DAG: llvm.func @_mlir_ciface_add_vector_to_matrix_into(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
//
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir | python3 %project_source_dir/tools/ll_fp_compat.py > %t.ll
// RUN: clang -Wno-override-module -O2 -c -x ir %t.ll -o %t.o
// RUN: clang -O2 %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/04-c-wrappers/main.c %t.o -o %t.exe
// RUN: %t.exe | FileCheck %s
// CHECK:      result: sizes=[2, 3] strides=[3, 1] offset=0
// CHECK-NEXT: _mlir_ciface_add_vector_to_matrix:
// CHECK-NEXT:   10.0  21.0  32.0
// CHECK-NEXT:   13.0  24.0  35.0
// CHECK-NEXT: add_vector_to_matrix_into (expanded descriptors):
// CHECK-NEXT:   10.0  21.0  32.0
// CHECK-NEXT:   13.0  24.0  35.0
