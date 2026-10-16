#!/bin/bash
# Broadcasting: add a length-4 vector to every row of a 3x4 matrix, once with
# linalg.broadcast + linalg.add and once with a single linalg.generic whose
# indexing map does the broadcast. Both are checked from Python against
# NumPy's own broadcasting (matrix + vector).
#
#   bash part1-using-mlir/ch08-linalg-by-hand/examples/04-broadcast/build.sh
#
# Adapted from mlir-ir-lowering's 4_linalg/4_broadcast.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../common.sh"

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

# (inspect) what linalg.broadcast means as a generic: the (d0, d1) -> (d1) map.
mlir-opt "$SCRIPT_DIR/add_vec_to_mat.mlir" --linalg-generalize-named-ops \
    -o "$BUILD/add_vec_to_mat_generalized.mlir"

# (runnable) both versions.
compile_kernel "$SCRIPT_DIR/add_vec_to_mat.mlir" bcast "$BUILD"
compile_kernel "$SCRIPT_DIR/add_vec_to_mat_generic.mlir" bcast_generic "$BUILD"

"$PYTHON" "$SCRIPT_DIR/run.py" "$BUILD/libbcast.so" "$BUILD/libbcast_generic.so"
