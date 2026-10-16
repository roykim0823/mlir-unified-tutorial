#!/usr/bin/env bash
# Chapter 6, example 01: a hand-written affine matmul, compiled to a shared
# library and called from Python (aot_main.py checks it against NumPy).
#
#   bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/01-affine-matmul/build.sh
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../common.sh"

OUT="$SCRIPT_DIR/build"
rm -rf "$OUT" && mkdir -p "$OUT"

# Only the first step differs from Chapter 2's pipeline: -lower-affine turns
# affine.for/load/store/apply into scf.for + memref + arith.
mlir-opt "$SCRIPT_DIR/matmul.mlir" -lower-affine -o "$OUT/matmul_scf.mlir"

lower_to_shared_lib "$SCRIPT_DIR/matmul.mlir" "$OUT" matmul
"$PYTHON" "$SCRIPT_DIR/aot_main.py" "$OUT"
