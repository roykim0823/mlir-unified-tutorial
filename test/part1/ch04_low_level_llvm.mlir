// Chapter 4, step 3: LLVM structs/arrays and the vector dialect, compiled and
// called from C; bufferization and the C-interface wrapper, inspected.
//
// RUN: mlir-translate --mlir-to-llvmir %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/structs_arrays.mlir \
// RUN:   | python3 %project_source_dir/tools/ll_fp_compat.py > %t.structs.ll
// RUN: FileCheck %s --check-prefix=STRUCT < %t.structs.ll
// STRUCT: define { i32, float } @make_pair()
// STRUCT: define void @make_pair_into(ptr %0)
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/vectors.mlir --convert-vector-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir | python3 %project_source_dir/tools/ll_fp_compat.py > %t.vectors.ll
// RUN: FileCheck %s --check-prefix=VECTOR < %t.vectors.ll
// VECTOR: fadd <4 x float>
// VECTOR: shufflevector <4 x float> %0, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
// VECTOR: call float @llvm.vector.reduce.fadd.v4f32
//
// RUN: clang -Wno-override-module -O2 -c -x ir %t.structs.ll -o %t.structs.o
// RUN: clang -Wno-override-module -O2 -c -x ir %t.vectors.ll -o %t.vectors.o
// RUN: clang -O2 %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/driver.c %t.structs.o %t.vectors.o -o %t.exe
// RUN: %t.exe | FileCheck %s
// CHECK: make_pair_into(&p) = {42, 3.14}
// CHECK: pair_second(&p)    = 3.14
// CHECK: array_demo()       = 3
// CHECK: pair_array_first_a = 7
// CHECK: vadd(a, b)   = [11, 22, 33, 44]
// CHECK: vmul(a, b)   = [10, 40, 90, 160]
// CHECK: vextract(b)  = 10
// CHECK: vshuffle     = [1, 2, 10, 20]
// CHECK: vreduce_sum  = 100
// CHECK: vsplat(7)    = [7, 7, 7, 7]
//
// Bufferization: to_buffer/to_tensor round trips fold away ...
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/bufferization.mlir --canonicalize | FileCheck %s --check-prefix=CANON
// CANON-LABEL: func.func @manual(%arg0: tensor<10xf32>) -> tensor<10xf32>
// CANON-NEXT:    return %arg0 : tensor<10xf32>
// ... and one-shot-bufferize turns tensor.insert into an in-place store.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/bufferization.mlir --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=ONESHOT
// ONESHOT-LABEL: func.func @insert_scalar(%arg0: memref<8xf32, strided<[?], offset: ?>>, %arg1: index) -> memref<8xf32, strided<[?], offset: ?>>
// ONESHOT-NOT:     memref.alloc
// ONESHOT:         memref.store %{{.*}}, %arg0[%arg1]
// ONESHOT:         return %arg0
// ONESHOT-LABEL: func.func @materialize
// ONESHOT:         memref.copy %arg0, %arg1
//
// The C interface: an unrolled function plus a pointer-taking wrapper.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/c_interface.mlir --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=CIFACE
// CIFACE:      llvm.func @identity(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CIFACE:      llvm.func @_mlir_ciface_identity(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
// CIFACE-NEXT:   %[[IN:.*]] = llvm.load %arg1
// CIFACE:        %[[R:.*]] = llvm.call @identity(
// CIFACE-NEXT:   llvm.store %[[R]], %arg0
