#!/usr/bin/env bash
# Chapter 6, example 02: affine maps and affine.apply.
#
# Nothing here is executed; we parse, fold and lower the files and save the
# results in build/ so you can read them next to the sources:
#
#   affine_apply_roundtrip.mlir   parse + print (inline maps become #mapN aliases)
#   affine_apply_folded.mlir      -canonicalize (constant folding + composition)
#   floordiv_lowered.mlir         -lower-affine (floordiv -> divsi + sign fix-up)
#   tiled_loop_lowered.mlir       -lower-affine (affine.for -> scf.for, min bound)
#   invalid_index.err             the verifier error for a non-affine subscript
#
#   bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/02-affine-maps/build.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
  export PATH="$WHEEL_BIN:$PATH"
elif ! command -v mlir-opt &> /dev/null; then
  echo "error: mlir-opt not found (create .venv with the MLIR wheel, see the root README)" >&2
  exit 1
fi

OUT="$SCRIPT_DIR/build"
rm -rf "$OUT" && mkdir -p "$OUT"

echo "== affine_apply: round-trip and fold =="
mlir-opt "$SCRIPT_DIR/affine_apply.mlir" -o "$OUT/affine_apply_roundtrip.mlir"
mlir-opt "$SCRIPT_DIR/affine_apply.mlir" -canonicalize -o "$OUT/affine_apply_folded.mlir"

echo "== floordiv: lower-affine =="
mlir-opt "$SCRIPT_DIR/floordiv.mlir" -lower-affine -o "$OUT/floordiv_lowered.mlir"

echo "== tiled_loop: lower-affine =="
mlir-opt "$SCRIPT_DIR/tiled_loop.mlir" -lower-affine -o "$OUT/tiled_loop_lowered.mlir"

echo "== invalid_index: must be rejected by the verifier =="
if mlir-opt "$SCRIPT_DIR/invalid_index.mlir" -o /dev/null 2> "$OUT/invalid_index.err"; then
  echo "error: invalid_index.mlir was unexpectedly accepted" >&2
  exit 1
fi
head -n 1 "$OUT/invalid_index.err"

echo "Done. Read the files in $OUT."
