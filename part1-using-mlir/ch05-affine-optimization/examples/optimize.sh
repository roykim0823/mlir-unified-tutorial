#!/bin/bash
# Chapter 5: lower linalg through the affine dialect, optimizing along the way.
#
# Regenerates the numbered stage files in this directory from
# 0-linalg-on-tensor.mlir. Run it as:
#
#   bash part1-using-mlir/ch05-affine-optimization/examples/optimize.sh

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Prefer mlir-opt from the MLIR wheel inside the repo's .venv (see the root
# README for setup); fall back to whatever is on PATH. Putting the wheel first
# keeps the output stable even if another LLVM (e.g. from Homebrew) is
# installed system-wide.
REPO_ROOT=$SCRIPT_DIR/../../..
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi

# Start from the same linalg-on-tensor kernel used in Chapters 1 and 2
# (matmul followed by ReLU) and bufferize it: tensors become memrefs.
mlir-opt \
    "$SCRIPT_DIR/0-linalg-on-tensor.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries=true" \
    -o "$SCRIPT_DIR/1-linalg-on-memref.mlir"

# Lower to the affine level of abstraction (instead of going straight to
# scf loops). The affine dialect provides analytical models of how memory
# is accessed in a kernel, which is what enables the optimizations below.
mlir-opt \
    "$SCRIPT_DIR/1-linalg-on-memref.mlir" \
    --convert-linalg-to-affine-loops \
    -o "$SCRIPT_DIR/2-affine.mlir"

# Loop fusion. The kernel has three loop nests with compatible iteration
# spaces (zero-fill, matmul, ReLU). Affine analysis proves the fusion is
# safe and, as a bonus, shrinks the 1 MB intermediate buffer down to a
# single element.
mlir-opt \
    "$SCRIPT_DIR/2-affine.mlir" \
    --affine-loop-fusion \
    -o "$SCRIPT_DIR/3-affine-fused.mlir"

# Loop tiling for cache locality. We tell the pass the cache size we are
# targeting (in KiB) and it picks tile sizes so the footprint of the inner
# loops fits. The raw output uses tiled induction variables with strange
# bounds, so we normalize and canonicalize to keep the IR readable.
mlir-opt \
    "$SCRIPT_DIR/3-affine-fused.mlir" \
    --affine-loop-tile="cache-size=1024" \
    --affine-loop-normalize \
    --canonicalize \
    -o "$SCRIPT_DIR/4-affine-fused-tiled.mlir"

# Done with affine analysis: lower out of the affine dialect into the scf
# and memref dialects. From here the pipeline continues exactly as in
# Chapter 2 (scf -> cf -> llvm).
mlir-opt \
    "$SCRIPT_DIR/4-affine-fused-tiled.mlir" \
    --lower-affine \
    --canonicalize \
    -o "$SCRIPT_DIR/5-loops.mlir"

# For comparison: the same kernel lowered directly to scf loops without the
# stop in the affine dialect (this is the Chapter 2 result).
mlir-opt \
    "$SCRIPT_DIR/1-linalg-on-memref.mlir" \
    --convert-linalg-to-loops \
    -o "$SCRIPT_DIR/6-loops-unoptimized.mlir"
