// Chapter 4, steps 2-3: the scalar and the vectorized array_add kernels,
// compiled and called from C through their _mlir_ciface_ wrappers.
//
// RUN: split-file %s %t
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/02-array-add/array_add.mlir --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm \
// RUN:   --reconcile-unrealized-casts -o %t/scalar.mlir
// RUN: FileCheck %s --check-prefix=SCALAR < %t/scalar.mlir
// The raw function takes three expanded 1-D descriptors (5 fields each) ...
// SCALAR:      llvm.func @array_add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr,
// SCALAR-SAME:   %arg14: i64) attributes {llvm.emit_c_interface}
// ... and accesses memory through the descriptor's ALIGNED pointer (field 1).
// SCALAR:        llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// SCALAR-NEXT:   llvm.getelementptr inbounds|nuw
// SCALAR-NEXT:   llvm.load %{{.*}} : !llvm.ptr -> f32
// The wrapper takes one pointer per memref.
// SCALAR:      llvm.func @_mlir_ciface_array_add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/array_add_vec.mlir --convert-vector-to-llvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts -o %t/vec.mlir
// RUN: FileCheck %s --check-prefix=VEC < %t/vec.mlir
// VEC: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr -> vector<8xf32>
// VEC: llvm.fadd %{{.*}}, %{{.*}} : vector<8xf32>
// VEC: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<8xf32>, !llvm.ptr
//
// RUN: mlir-translate --mlir-to-llvmir %t/scalar.mlir | python3 %project_source_dir/tools/ll_fp_compat.py > %t/scalar.ll
// RUN: mlir-translate --mlir-to-llvmir %t/vec.mlir | python3 %project_source_dir/tools/ll_fp_compat.py > %t/vec.ll
// RUN: clang -Wno-override-module -O2 -c -x ir %t/scalar.ll -o %t/scalar.o
// RUN: clang -Wno-override-module -O2 -c -x ir %t/vec.ll -o %t/vec.o
// RUN: clang %t/driver.c %t/scalar.o -o %t/scalar.exe
// RUN: clang %t/driver.c %t/vec.o -o %t/vec.exe
// RUN: %t/scalar.exe | FileCheck %s
// RUN: %t/vec.exe | FileCheck %s
// CHECK: c[0]=2 c[1]=3 c[1023]=1025 errors=0

//--- driver.c
#include <stdint.h>
#include <stdio.h>
typedef struct { float *allocated, *aligned; int64_t offset, sizes[1], strides[1]; } MemRef1D;
void _mlir_ciface_array_add(MemRef1D *a, MemRef1D *b, MemRef1D *c);
int main(void) {
  static float a[1024], b[1024], c[1024];
  for (int i = 0; i < 1024; ++i) { a[i] = (float)i; b[i] = 2.0f; c[i] = -1.0f; }
  MemRef1D A = {a, a, 0, {1024}, {1}}, B = {b, b, 0, {1024}, {1}}, C = {c, c, 0, {1024}, {1}};
  _mlir_ciface_array_add(&A, &B, &C);
  int errors = 0;
  for (int i = 0; i < 1024; ++i) errors += c[i] != a[i] + b[i];
  printf("c[0]=%g c[1]=%g c[1023]=%g errors=%d\n", c[0], c[1], c[1023], errors);
  return 0;
}
