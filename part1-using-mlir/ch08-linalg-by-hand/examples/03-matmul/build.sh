#!/bin/bash
# linalg.matmul and its generic form. Lowers the generic both to scf loops and
# to affine loops for comparison, compiles the named op, and checks it from
# Python against NumPy's A @ B.
#
#   bash part1-using-mlir/ch08-linalg-by-hand/examples/03-matmul/build.sh
#
# Adapted from mlir-ir-lowering's 4_linalg/3_matmul.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../common.sh"

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

# (inspect) same nest, two loop dialects:
#   --convert-linalg-to-loops         -> scf.for    (plain structured loops)
#   --convert-linalg-to-affine-loops  -> affine.for (analyzable, tileable)
mlir-opt "$SCRIPT_DIR/matmul_generic.mlir" --convert-linalg-to-loops \
    -o "$BUILD/matmul_generic_scf.mlir"
mlir-opt "$SCRIPT_DIR/matmul_generic.mlir" --convert-linalg-to-affine-loops \
    -o "$BUILD/matmul_generic_affine.mlir"

# (runnable) the named op through the scf route.
compile_kernel "$SCRIPT_DIR/matmul.mlir" matmul "$BUILD"

"$PYTHON" "$SCRIPT_DIR/run.py" "$BUILD/libmatmul.so"
