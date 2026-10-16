#!/usr/bin/env bash
# Chapter 6, example 03: the affine optimization passes, one file per pass.
#
# Each <name>.mlir is run through its pass and the result saved as
# build/<name>_after.mlir, so you can `diff <name>.mlir build/<name>_after.mlir`.
#
#   bash part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/03-affine-passes/build.sh
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../common.sh"

OUT="$SCRIPT_DIR/build"
rm -rf "$OUT" && mkdir -p "$OUT"

# run <name> <mlir-opt pass arguments...>
run() {
  local name=$1
  shift
  echo "== $name: $*"
  mlir-opt "$SCRIPT_DIR/$name.mlir" "$@" -o "$OUT/${name}_after.mlir"
}

run licm       -affine-loop-invariant-code-motion
run unroll     --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))"
run unroll_jam --pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))"
run normalize  -affine-loop-normalize
run scalrep    -affine-scalrep
run coalescing -affine-loop-coalescing
run vectorize  -affine-super-vectorize="virtual-vector-size=8"
run fusion     -affine-loop-fusion
run tiling     -affine-loop-tile="tile-size=8"

# Passes compose: after fusion, B[i] is stored and immediately reloaded in the
# same iteration, which is exactly the pattern scalar replacement forwards.
echo "== fusion + scalrep"
mlir-opt "$SCRIPT_DIR/fusion.mlir" -affine-loop-fusion -affine-scalrep \
  -o "$OUT/fusion_scalrep_after.mlir"

# Coalescing leaves the trip counts as affine.apply of constant maps;
# -canonicalize folds them and inlines the div/mod maps into the subscripts.
echo "== coalescing + canonicalize"
mlir-opt "$SCRIPT_DIR/coalescing.mlir" -affine-loop-coalescing -canonicalize \
  -o "$OUT/coalescing_canonical_after.mlir"

echo "Done. Compare each <name>.mlir with $OUT/<name>_after.mlir."
