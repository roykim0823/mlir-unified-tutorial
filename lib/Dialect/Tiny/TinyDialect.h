//===- TinyDialect.h - Tiny dialect header ----------------------*- C++ -*-===//
//
// Declares the Tiny dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH1_TINY_DIALECT_H
#define TUTORIAL_CH1_TINY_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated dialect declaration.
#include "Dialect/Tiny/TinyOpsDialect.h.inc"

// Include the auto-generated type declarations.
#define GET_TYPEDEF_CLASSES
#include "Dialect/Tiny/TinyTypes.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "Dialect/Tiny/TinyOps.h.inc"

#endif // TUTORIAL_CH1_TINY_DIALECT_H
