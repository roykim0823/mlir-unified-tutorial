#!/bin/sh
# GPU lowering pipeline for the MNIST MLP (ch31). Ported from
# ML-compiler-exercise src/mnist/gpu/run_mlir_pipeline.sh.
#
# Needs the wheel's mlir-opt / mlir-translate on PATH (see the repo root
# README). Reads ../mnist_model_linalg.mlir (the checked-in snapshot) and
# writes mnist_{1_loops,2_gpu,3_outlined,nvvm}.mlir + mnist.ll (one file per stage)
# into the CURRENT directory, so run it
# from a scratch directory if you don't want outputs in the source tree:
#
#   mkdir -p /tmp/mnist-gpu && cd /tmp/mnist-gpu
#   sh $REPO/part6-end-to-end-ml/src/mnist/gpu/run_mlir_pipeline.sh       # PTX
#
# Knobs (environment variables / first argument):
#   $1 = FORMAT  isa (default): PTX text via LLVM's NVPTX backend - works on
#                any machine, no CUDA toolkit. At run time the CUDA driver
#                JIT-compiles the PTX (mgpuModuleLoadJIT).
#                bin: a real cubin - needs ptxas (CUDA toolkit) on PATH.
#   TILE   affine tile size, default 16 (the source repo's value for mnist).
#          It sets how much work each GPU thread does: the outer tile loop
#          becomes the block index, the next tile loop the thread index, and
#          the TILExTILE intra-tile loops run sequentially inside one thread.
#          "Play with the tile size" (16, 32, 64, ...) - smaller models want
#          smaller tiles. TILE=0 disables tiling (one element per thread).
#   CHIP   GPU architecture, default sm_90 (H100). sm_80 = A100, sm_75 = T4...
#   PTXV   PTX ISA feature, default +ptx80.
#
# Drift notes vs the source script:
#  - The source passed TWO --pass-pipeline flags to one mlir-opt
#    (affine-loop-invariant-code-motion, then convert-affine-for-to-gpu).
#    mlir-opt keeps only the last one, so LICM never ran there; it is
#    dropped here. Running LICM for real breaks the mapping: it hoists the
#    inner nests out of the trip-count-1 batch tile loop, leaving an empty
#    affine.for that convert-affine-for-to-gpu rejects with
#    "expected perfectly nested loops in the body".
#  - --gpu-module-to-binary after --gpu-to-llvm is dropped: the
#    gpu-lower-to-nvvm-pipeline already ran it.
#  - llc is replaced by the compile.sh step (the wheel ships no llc).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
FORMAT="${1:-isa}"
TILE="${TILE:-16}"
CHIP="${CHIP:-sm_90}"
PTXV="${PTXV:-+ptx80}"
IN="${IN:-$HERE/../mnist_model_linalg.mlir}"

[ -f "$IN" ] || { echo "error: input $IN not found (set IN=...)" >&2; exit 1; }

if [ "$TILE" -gt 0 ]; then
  TILE_FLAG="--affine-loop-tile=tile-size=$TILE"
else
  TILE_FLAG=""
fi

# shellcheck disable=SC2086  # TILE_FLAG is intentionally word-split (may be empty)
mlir-opt "$IN" \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-affine-loops \
  $TILE_FLAG \
  --canonicalize \
  --cse \
  -o mnist_1_loops.mlir
mlir-opt mnist_1_loops.mlir \
  --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' \
  -o mnist_2_gpu.mlir
mlir-opt mnist_2_gpu.mlir --gpu-kernel-outlining --lower-affine -o mnist_3_outlined.mlir
mlir-opt mnist_3_outlined.mlir \
  --gpu-decompose-memrefs \
  --expand-strided-metadata \
  --normalize-memrefs \
  --convert-index-to-llvm \
  --arith-expand \
  --memref-expand \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=$CHIP cubin-features=$PTXV opt-level=3 cubin-format=$FORMAT" \
  --convert-nvvm-to-llvm \
  --reconcile-unrealized-casts \
  --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true' \
  -o mnist_nvvm.mlir

mlir-translate --mlir-to-llvmir mnist_nvvm.mlir -o mnist.ll

echo "wrote mnist_nvvm.mlir and mnist.ll (format=$FORMAT tile=$TILE chip=$CHIP)"
