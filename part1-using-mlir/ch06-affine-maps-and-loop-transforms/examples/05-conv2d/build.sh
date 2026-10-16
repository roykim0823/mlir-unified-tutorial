#!/usr/bin/env bash
# Chapter 6, example 05: 2-D convolution in the affine dialect.
#
# 1. Compile conv2d.mlir as written            -> build/libconv2d.{dylib,so}
# 2. Optimize it with affine passes             -> build/conv2d_opt.mlir
#      LICM, 4x4 tiling of the output loops, 2x unrolling of the innermost
#      (filter-column) reduction loop, then -canonicalize to tidy up.
# 3. Compile the optimized version              -> build/libconv2d_opt.{dylib,so}
# 4. aot_main.py runs both and checks them against a NumPy reference.
#
#   bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/05-conv2d/build.sh
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../common.sh"

OUT="$SCRIPT_DIR/build"
rm -rf "$OUT" && mkdir -p "$OUT"

lower_to_shared_lib "$SCRIPT_DIR/conv2d.mlir" "$OUT" conv2d

# The affine passes all run on functions, so they are nested under func.func.
mlir-opt "$SCRIPT_DIR/conv2d.mlir" \
  --pass-pipeline="builtin.module(func.func(affine-loop-invariant-code-motion,affine-loop-tile{tile-size=4},affine-loop-unroll{unroll-factor=2},canonicalize))" \
  -o "$OUT/conv2d_opt.mlir"
lower_to_shared_lib "$OUT/conv2d_opt.mlir" "$OUT" conv2d_opt

"$PYTHON" "$SCRIPT_DIR/aot_main.py" "$OUT"
