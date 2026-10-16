#!/bin/bash
# Inspect-only: tile a tensor linalg.matmul with plain passes. Bufferize,
# lower to affine loops, then --affine-loop-tile 5x5x5. Writes
# build/matmul_tiled.mlir; look for the outer loops stepping by 5.
#
#   bash part1-using-mlir/ch08-linalg-by-hand/examples/06-tiling/build.sh
#
# Adapted from mlir-ir-lowering's 4_linalg/6_tiling.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../common.sh"

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

#   --convert-tensor-to-linalg         remaining tensor ops -> linalg
#   --linalg-generalize-named-ops      linalg.matmul -> linalg.generic
#   --one-shot-bufferize=...           tensors -> memrefs
#   --buffer-deallocation-pipeline     insert deallocs for temporary buffers
#   --convert-bufferization-to-memref  leftover bufferization ops -> memref
#   --convert-linalg-to-affine-loops   linalg -> affine.for nest
#   --affine-loop-tile="tile-size=5"   split every loop into tile + point loops
#   --canonicalize --cse               cleanup
mlir-opt "$SCRIPT_DIR/matmul_tile.mlir" \
    --convert-tensor-to-linalg \
    --linalg-generalize-named-ops \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-deallocation-pipeline \
    --convert-bufferization-to-memref \
    --convert-linalg-to-affine-loops \
    --affine-loop-tile="tile-size=5" \
    --canonicalize \
    --cse \
    -o "$BUILD/matmul_tiled.mlir"

echo "Wrote $BUILD/matmul_tiled.mlir: the 3-loop matmul is now a 6-deep tiled nest."
