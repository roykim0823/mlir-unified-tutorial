#!/bin/sh
# GPU lowering pipeline for ResNet-18 (ch31). Ported from
# ML-compiler-exercise src/resnet18/gpu/run_mlir_pipeline.sh.
#
# Unlike sample/mnist/cnn, this model takes the *scf.parallel* route to the
# GPU instead of the affine one:
#
#   linalg --(linalg-fuse-elementwise-ops)--> fewer, fused linalg.generics
#          --(convert-linalg-to-parallel-loops)--> scf.parallel nests
#          --(gpu-map-parallel-loops)--> mapping attrs: outer -> blocks, nested -> threads
#          --(convert-parallel-loops-to-gpu)--> gpu.launch
#
# and it keeps full memref descriptors at the kernel boundary
# (use-bare-pointers-*=false) where the other models pass bare pointers.
#
# Source author's tuning note, kept verbatim in spirit:
#   Try --pass-pipeline='builtin.module(func.func(affine-loop-tile{tile-size=16}))'
#   for performance (16, 32, 64, 128, 256, ...).
#   Currently this yields CUDA_ERROR_ILLEGAL_ADDRESS.
# (For the scf route the matching knob is --scf-parallel-loop-tiling, which
# gives the mapper a nested scf.parallel to put on threads; see ch31.)
#
# STATUS: PENDING. Needs ../resnet18_model_linalg.mlir with the parameters
# inserted (ch30's import + get_buffers_in_mlir_format.py flow), which was not
# in this repo when this script was ported, so this exact script is untested.
# The same pass sequence is verified on the mnist snapshot
# (test/part6/ch31_mnist_gpu_ptx.mlir, PAR prefix).
#
# Usage / knobs: as in ../../mnist/gpu/run_mlir_pipeline.sh ($1 = isa|bin,
# CHIP, PTXV, IN). Outputs resnet18_{2_gpu,3_outlined,nvvm}.mlir + resnet18.ll in the CURRENT
# directory.
#
# Drift notes vs the source script: its second stage
# (affine-loop-invariant-code-motion / convert-affine-for-to-gpu via two
# --pass-pipeline flags) is dropped - only the last flag took effect, and
# after convert-parallel-loops-to-gpu there are no affine.for loops left to
# map. The trailing --gpu-module-to-binary is dropped too (already part of
# gpu-lower-to-nvvm-pipeline), and llc moved to compile.sh.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
FORMAT="${1:-isa}"
CHIP="${CHIP:-sm_90}"
PTXV="${PTXV:-+ptx80}"
IN="${IN:-$HERE/../resnet18_model_linalg.mlir}"

[ -f "$IN" ] || { echo "error: input $IN not found (set IN=...)" >&2; exit 1; }

mlir-opt "$IN" \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --canonicalize \
  --cse \
  -o resnet18_2_gpu.mlir
mlir-opt resnet18_2_gpu.mlir --gpu-kernel-outlining --lower-affine -o resnet18_3_outlined.mlir
mlir-opt resnet18_3_outlined.mlir \
  --gpu-decompose-memrefs \
  --expand-strided-metadata \
  --normalize-memrefs \
  --convert-index-to-llvm \
  --arith-expand \
  --memref-expand \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=$CHIP cubin-features=$PTXV opt-level=3 cubin-format=$FORMAT" \
  --convert-nvvm-to-llvm \
  --reconcile-unrealized-casts \
  --gpu-to-llvm='use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false' \
  -o resnet18_nvvm.mlir

mlir-translate --mlir-to-llvmir resnet18_nvvm.mlir -o resnet18.ll

echo "wrote resnet18_nvvm.mlir and resnet18.ll (format=$FORMAT chip=$CHIP)"
