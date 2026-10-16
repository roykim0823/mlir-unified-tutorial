#!/bin/bash
# Elementwise add three ways: linalg.generic -> linalg.map -> linalg.add.
#
# Saves the loop form of the two tensor versions for inspection, lowers the
# named memref version to a shared library, and runs it from Python against
# NumPy. Run from anywhere:
#
#   bash part1-using-mlir/ch08-linalg-by-hand/examples/01-elementwise/build.sh
#
# Adapted from mlir-ir-lowering's 4_linalg/1_elementwise.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../common.sh"

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

# (inspect) linalg.generic on tensors -> scf loops. --convert-linalg-to-loops
# only lowers linalg ops on memrefs, so bufferize first; without it the tensor
# op round-trips unchanged.
mlir-opt "$SCRIPT_DIR/generic_add.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    -o "$BUILD/generic_add_loops.mlir"

# (inspect) linalg.map is shorthand for the same generic: identical loops.
mlir-opt "$SCRIPT_DIR/map_add.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    -o "$BUILD/map_add_loops.mlir"

# (inspect) the named op, generalized back to the generic it stands for.
mlir-opt "$SCRIPT_DIR/add.mlir" --linalg-generalize-named-ops \
    -o "$BUILD/add_generalized.mlir"

# (runnable) the named linalg.add on memrefs needs no bufferization.
compile_kernel "$SCRIPT_DIR/add.mlir" add "$BUILD"

"$PYTHON" "$SCRIPT_DIR/run.py" "$BUILD/libadd.so"
