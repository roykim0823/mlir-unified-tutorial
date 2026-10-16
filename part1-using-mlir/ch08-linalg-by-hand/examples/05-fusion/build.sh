#!/bin/bash
# Elementwise fusion with plain passes: fuse (a + b) * c from two named ops
# into one linalg.generic, bufferize the fused kernel, turn its returned
# tensor into an out-parameter, compile it, and check it from Python against
# NumPy's (a + b) * c.
#
#   bash part1-using-mlir/ch08-linalg-by-hand/examples/05-fusion/build.sh
#
# Adapted from mlir-ir-lowering's 4_linalg/5_fusion.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../common.sh"

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

# (inspect) the fusion pipeline.
#   --canonicalize                 general cleanup
#   --linalg-fuse-elementwise-ops  fusion -- a no-op here: the ops are still named
#   --cse                          merges the two identical tensor.empty ops
#   --linalg-generalize-named-ops  linalg.add / linalg.mul -> linalg.generic
#   --linalg-fuse-elementwise-ops  now the generics fuse into one
mlir-opt "$SCRIPT_DIR/separate_ops.mlir" \
    --canonicalize \
    --linalg-fuse-elementwise-ops \
    --cse \
    --linalg-generalize-named-ops \
    --linalg-fuse-elementwise-ops \
    -o "$BUILD/fused_ops.mlir"

# (inspect) the same two passes fold a broadcast into its consumer's maps.
mlir-opt "$SCRIPT_DIR/broadcast_add.mlir" \
    --linalg-generalize-named-ops \
    --linalg-fuse-elementwise-ops \
    -o "$BUILD/broadcast_add_fused.mlir"

# (runnable) the fused op is on tensors and returns a tensor:
#   --one-shot-bufferize            tensors -> memrefs; the result becomes a
#                                   freshly allocated memref
#   --buffer-results-to-out-params  the returned memref becomes a trailing
#                                   out-parameter. modify-public-functions is
#                                   needed because @addmul is public;
#                                   hoist-static-allocs writes straight into
#                                   the caller's buffer instead of alloc+copy.
mlir-opt "$BUILD/fused_ops.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-results-to-out-params="modify-public-functions=true hoist-static-allocs=true" \
    --canonicalize \
    -o "$BUILD/fused_memref.mlir"
mlir-opt "$BUILD/fused_memref.mlir" --convert-linalg-to-loops \
    -o "$BUILD/fused_loops.mlir"

# (inspect) for contrast: the UNFUSED kernel through the same bufferization.
# Two loops and a 10-element temporary buffer for a + b.
mlir-opt "$SCRIPT_DIR/separate_ops.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-results-to-out-params="modify-public-functions=true hoist-static-allocs=true" \
    --canonicalize \
    --convert-linalg-to-loops \
    -o "$BUILD/unfused_loops.mlir"

compile_kernel "$BUILD/fused_memref.mlir" fused "$BUILD"

"$PYTHON" "$SCRIPT_DIR/run.py" "$BUILD/libfused.so"
