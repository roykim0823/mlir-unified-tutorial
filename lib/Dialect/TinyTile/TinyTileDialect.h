//===- TinyTileDialect.h - TinyTile dialect header --------------*- C++ -*-===//
//
// Declares the TinyTile dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIAL_CH3_TINY_TILE_DIALECT_H
#define TUTORIAL_CH3_TINY_TILE_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the Tiny dialect for !tiny.ptr type.
#include "Dialect/Tiny/TinyDialect.h"

// Include the lowering interface.
#include "Dialect/TinyTile/TinyTileLoweringInterface.h"

// Include the auto-generated dialect declaration.
#include "Dialect/TinyTile/TinyTileOpsDialect.h.inc"

// Include the auto-generated enum declarations (must come before attrs).
#include "Dialect/TinyTile/TinyTileEnums.h.inc"

// Include the auto-generated attribute declarations.
#define GET_ATTRDEF_CLASSES
#include "Dialect/TinyTile/TinyTileAttrs.h.inc"

// Include the auto-generated type declarations.
#define GET_TYPEDEF_CLASSES
#include "Dialect/TinyTile/TinyTileTypes.h.inc"

// Include the auto-generated operation declarations.
#define GET_OP_CLASSES
#include "Dialect/TinyTile/TinyTileOps.h.inc"

#endif // TUTORIAL_CH3_TINY_TILE_DIALECT_H
