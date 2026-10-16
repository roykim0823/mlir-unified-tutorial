#!/usr/bin/env bash
# The CUDA-box recipe for the square example - Linux + NVIDIA GPU only.
# Ported from mlir-ir-lowering 8_gpu/run_on_gpu.sh.
#
# Needs an MLIR built with the CUDA runner (-DMLIR_ENABLE_CUDA_RUNNER=ON, see
# docs/appendix-source-build.md) - or the reference series' docker image
# ghcr.io/sdiehl/docker-mlir-cuda:main - with its bin/ on PATH and
# LLVM=<that build dir>. It continues past build.sh: lower to NVVM, serialize
# the kernel to a GPU binary (gpu-module-to-binary), lower the host side to
# LLVM, and JIT-run it with mlir-runner linked against the CUDA runtime.
#
# Fixes vs the 8_gpu original (found while verifying this port):
#  - runs square_runnable.mlir (square.mlir + a @main that allocates,
#    gpu.host_register's and prints the buffer); square.mlir alone has no entry
#    point mlir-runner can call.
#  - adds --lower-affine before --convert-gpu-to-nvvm: the kernel's index math
#    is an affine.apply, and without it serialization fails with
#    "LLVM Translation failed for operation: builtin.unrealized_conversion_cast".
#  - adds --convert-scf-to-cf / --expand-strided-metadata for @main's loop and
#    subview.
#
# STATUS: everything up to the final mlir-runner call is verified on a machine
# WITHOUT a GPU using FORMAT=isa (PTX text); FORMAT=fatbin (default, needs ptxas)
# and the run itself are NOT verified in this port.
set -euo pipefail
cd "$(dirname "$0")"

CHIP="${1:-sm_80}"       # your GPU's architecture: sm_75 Turing, sm_80 Ampere, sm_90 Hopper, ...
FORMAT="${FORMAT:-fatbin}"   # fatbin/bin need ptxas; isa = PTX only, JIT-ed by the driver at load time

mlir-opt square_runnable.mlir \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --lower-affine \
  --convert-gpu-to-nvvm \
  "--nvvm-attach-target=chip=${CHIP}" \
  --gpu-to-llvm \
  "--gpu-module-to-binary=format=${FORMAT}" \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --convert-to-llvm \
  --reconcile-unrealized-casts \
  -o square_gpu_ready.mlir
echo "wrote square_gpu_ready.mlir (chip=${CHIP}, format=${FORMAT})"

if [ "${NO_RUN:-0}" = 1 ]; then exit 0; fi

: "${LLVM:?set LLVM to your CUDA-runner-enabled LLVM/MLIR build directory}"
mlir-runner square_gpu_ready.mlir \
  --shared-libs="$LLVM/lib/libmlir_cuda_runtime.so" \
  --shared-libs="$LLVM/lib/libmlir_runner_utils.so" \
  --entry-point-result=void
# expected: [0,  1,  4,  9,  16,  25,  36,  49]
