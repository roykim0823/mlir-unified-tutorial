//===- TinyDialect.cpp - Tiny dialect implementation ------------*- C++ -*-===//
//
// Implements the Tiny dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Tiny/TinyDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tiny;

//===----------------------------------------------------------------------===//
// Tiny dialect implementation
//===----------------------------------------------------------------------===//

#include "Dialect/Tiny/TinyOpsDialect.cpp.inc"

// Include the auto-generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "Dialect/Tiny/TinyTypes.cpp.inc"

void TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Tiny/TinyOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Tiny/TinyTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd operation definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Tiny/TinyOps.cpp.inc"
