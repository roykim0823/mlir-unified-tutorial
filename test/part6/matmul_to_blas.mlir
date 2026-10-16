// RUN: tutorial-opt --linalg-to-bufferization --convert-matmul-to-blas %s | FileCheck %s

// CHECK: llvm.func @cblas_sgemm(i32, i32, i32, i32, i32, i32, f32, !llvm.ptr, i32, !llvm.ptr, i32, f32, !llvm.ptr, i32)

// CHECK-LABEL: @dynamic
// Dynamic dimensions are read back out of the memrefs at runtime...
// CHECK-COUNT-5: memref.dim
// ...and the matmul itself is gone, replaced by the library call.
// CHECK-NOT: linalg.matmul
// CHECK: llvm.call @cblas_sgemm
func.func @dynamic(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                   %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%acc: tensor<?x?xf32>)
  -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

// CHECK-LABEL: @static
// Static dimensions become constants: M, N, K for a 3x4 * 4x5 product.
// CHECK-DAG: llvm.mlir.constant(3 : i32)
// CHECK-DAG: llvm.mlir.constant(5 : i32)
// CHECK-DAG: llvm.mlir.constant(4 : i32)
// CHECK-NOT: linalg.matmul
// CHECK-NOT: memref.dim
// CHECK: llvm.call @cblas_sgemm
func.func @static(%lhs: memref<3x4xf32>, %rhs: memref<4x5xf32>,
                  %acc: memref<3x5xf32>) {
  linalg.matmul
    ins(%lhs, %rhs: memref<3x4xf32>, memref<4x5xf32>)
    outs(%acc: memref<3x5xf32>)
  return
}
