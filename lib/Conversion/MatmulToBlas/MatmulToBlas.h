#ifndef LIB_CONVERSION_MATMULTOBLAS_MATMULTOBLAS_H_
#define LIB_CONVERSION_MATMULTOBLAS_MATMULTOBLAS_H_

#include "mlir/Pass/Pass.h"  // from @llvm-project

// Extra includes needed for dependent dialects
#include "mlir/Dialect/Arith/IR/Arith.h"        // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"       // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"    // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"      // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"      // from @llvm-project

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL
#include "lib/Conversion/MatmulToBlas/MatmulToBlas.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/MatmulToBlas/MatmulToBlas.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_CONVERSION_MATMULTOBLAS_MATMULTOBLAS_H_
