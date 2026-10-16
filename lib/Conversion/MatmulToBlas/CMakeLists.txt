add_mlir_library(MatmulToBlas
  MatmulToBlas.cpp

  ${PROJECT_SOURCE_DIR}/lib/Conversion/MatmulToBlas/
  ADDITIONAL_HEADER_DIRS

  DEPENDS
  MatmulToBlasPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  )

set(LLVM_TARGET_DEFINITIONS MatmulToBlas.td)
mlir_tablegen(MatmulToBlas.h.inc -gen-pass-decls -name MatmulToBlas)
add_public_tablegen_target(MatmulToBlasPassIncGen)
add_mlir_doc(MatmulToBlas MatmulToBlas MatmulToBlas/ -gen-pass-doc)
