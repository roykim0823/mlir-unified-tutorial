//===- TinyLoopDialect.h - TinyLoop dialect header --------------*- C++ -*-===//
//
// Declares the TinyLoop dialect and its operations (Chapter 2).
// This dialect introduces MLIR Regions and Basic Blocks.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH2_TINY_LOOP_DIALECT_H
#define TUTORIAL_CH2_TINY_LOOP_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated dialect declaration.
#include "Dialect/TinyLoop/TinyLoopOpsDialect.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "Dialect/TinyLoop/TinyLoopOps.h.inc"

#endif // TUTORIAL_CH2_TINY_LOOP_DIALECT_H
