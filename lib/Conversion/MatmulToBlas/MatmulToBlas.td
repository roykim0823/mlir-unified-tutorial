#ifndef LIB_CONVERSION_MATMULTOBLAS_MATMULTOBLAS_TD_
#define LIB_CONVERSION_MATMULTOBLAS_MATMULTOBLAS_TD_

include "mlir/Pass/PassBase.td"

def MatmulToBlas : Pass<"convert-matmul-to-blas", "mlir::ModuleOp"> {
  let summary = "Convert linalg.matmul operations to cblas_sgemm library calls.";

  let description = [{
    Replaces each 2-D f32 `linalg.matmul` (on memrefs, i.e. after
    bufferization) with a call to the C BLAS single-precision GEMM routine
    `cblas_sgemm`, declared as an external LLVM function. The resulting
    module must be linked against a BLAS implementation (OpenBLAS, Apple
    Accelerate, ...) that provides the CBLAS interface.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
  ];
}

#endif  // LIB_CONVERSION_MATMULTOBLAS_MATMULTOBLAS_TD_
