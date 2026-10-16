#!/usr/bin/env bash
# INSPECT-ONLY on a machine without an NVIDIA GPU: lower square.mlir from a
# high-level parallel loop down to GPU code, saving each stage so you can read
# the transformation. Ported from mlir-ir-lowering 8_gpu/build.sh; stage 4 and
# stage 5 are additions of this port.
#
# These passes are pure IR->IR transforms and run on any CPU (including Apple
# silicon). What you CANNOT do without NVIDIA hardware is *execute* the result -
# see run_on_gpu.sh for that. Needs the wheel's mlir-opt on PATH.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build
mkdir -p build

echo "== stage 1: scf.parallel  ->  gpu.module (outlined kernel + host launch) =="
mlir-opt square.mlir \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  -o build/square_gpu.mlir
echo "   look for: gpu.launch_func, gpu.module @square_kernel, gpu.block_id/thread_id"

echo "== stage 2: device kernel  ->  NVVM dialect (PTX-level) =="
mlir-opt build/square_gpu.mlir --convert-gpu-to-nvvm -o build/square_nvvm.mlir
echo "   look for: nvvm.read.ptx.sreg.tid.x / ctaid.x  (= threadIdx / blockIdx)"

echo "== stage 3: attach a target GPU architecture (e.g. sm_80 = Ampere) =="
mlir-opt build/square_gpu.mlir "--nvvm-attach-target=chip=sm_80" \
  -o build/square_target.mlir
echo "   look for: #nvvm.target<chip = \"sm_80\"> on the gpu.module"

echo "== stage 4: tile first, so the kernel gets 256 threads per block like square.cu =="
mlir-opt square.mlir \
  --scf-parallel-loop-tiling="parallel-loop-tile-sizes=256" \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --canonicalize \
  -o build/square_gpu_tiled.mlir
echo "   look for: blocks in (%c4, ...) threads in (%c256, ...)  (= square<<<4, 256>>>)"

echo "== stage 5: all the way to PTX text (no CUDA toolkit needed) =="
mlir-opt build/square_gpu_tiled.mlir \
  --lower-affine \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-format=isa" \
  -o build/square_ptx.mlir
echo "   look for: .visible .entry square_kernel, mov.u32 %r, %ctaid.x / %tid.x"

echo
echo "Wrote build/square_{gpu,nvvm,target,gpu_tiled,ptx}.mlir."
echo "To actually RUN on a GPU, see run_on_gpu.sh (needs CUDA hardware + MLIR CUDA runner)."
