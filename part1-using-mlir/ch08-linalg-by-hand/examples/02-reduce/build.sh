#!/bin/bash
# Row sums two ways: linalg.generic with a "reduction" iterator, and the named
# linalg.reduce. Saves the generic's loop nest for inspection, compiles the
# named memref version, and checks it from Python against a.sum(axis=1).
#
#   bash part1-using-mlir/ch08-linalg-by-hand/examples/02-reduce/build.sh
#
# Adapted from mlir-ir-lowering's 4_linalg/2_reduce.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../common.sh"

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

# (inspect) outer parallel loop over rows, inner reduction loop that
# accumulates into %out. On tensors, so bufferize first.
mlir-opt "$SCRIPT_DIR/reduce_generic.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    -o "$BUILD/reduce_generic_loops.mlir"

# (inspect) linalg.reduce generalized: the same maps and iterators.
mlir-opt "$SCRIPT_DIR/reduce.mlir" --linalg-generalize-named-ops \
    -o "$BUILD/reduce_generalized.mlir"

# (runnable) the named linalg.reduce on memrefs.
compile_kernel "$SCRIPT_DIR/reduce.mlir" reduce "$BUILD"

"$PYTHON" "$SCRIPT_DIR/run.py" "$BUILD/libreduce.so"
