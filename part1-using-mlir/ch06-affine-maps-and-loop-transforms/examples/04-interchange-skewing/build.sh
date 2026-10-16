#!/usr/bin/env bash
# Chapter 6, example 04: loop interchange, skewing and the wavefront, written
# by hand (MLIR has no one-flag affine pass for them). Each file is compiled to
# a shared library and aot_main.py checks that every rewritten version computes
# exactly what the original does.
#
#   bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/04-interchange-skewing/build.sh
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../common.sh"

OUT="$SCRIPT_DIR/build"
rm -rf "$OUT" && mkdir -p "$OUT"

# For reading: what -lower-affine makes of the wavefront's max/min bounds and
# of its affine.parallel (an scf.parallel, which the CPU pipeline serializes).
mlir-opt "$SCRIPT_DIR/skewing_manual.mlir" -lower-affine -o "$OUT/skewing_scf.mlir"

lower_to_shared_lib "$SCRIPT_DIR/interchange_manual.mlir" "$OUT" interchange_manual
lower_to_shared_lib "$SCRIPT_DIR/skewing_manual.mlir" "$OUT" skewing_manual
"$PYTHON" "$SCRIPT_DIR/aot_main.py" "$OUT"
