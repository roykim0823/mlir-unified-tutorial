#!/bin/sh
# Compile and link the GPU-lowered sample model - Linux + NVIDIA GPU + CUDA only.
# Ported from ML-compiler-exercise src/sample/gpu/compile.sh; the RWTH
# /cvmfs/.../CUDA/12.6.3 paths are replaced by $CUDA_HOME.
#
# Run in the directory where run_mlir_pipeline.sh wrote sample.ll.
# Prerequisites (see ch31 section "Running it" and docs/appendix-source-build.md):
#   CUDA_HOME  your CUDA toolkit, e.g. /usr/local/cuda (lib64/ must hold
#              libcudart; libcuda comes with the driver)
#   MLIR_LIB   lib/ of an MLIR build configured with -DMLIR_ENABLE_CUDA_RUNNER=ON
#              (holds libmlir_cuda_runtime + libmlir_(c_)runner_utils; the wheel
#              does not ship libmlir_cuda_runtime)
#   llc        from that same build (preferred; same LLVM version as the .ll),
#              otherwise clang is used on the .ll
# STATUS: not verified in this port (no NVIDIA GPU on the porting machine).
set -eu

: "${CUDA_HOME:?set CUDA_HOME to your CUDA toolkit, e.g. /usr/local/cuda}"
: "${MLIR_LIB:?set MLIR_LIB to the lib/ dir of a CUDA-runner-enabled MLIR build}"
CUDA_LIB="${CUDA_LIB:-$CUDA_HOME/lib64}"
HERE=$(cd "$(dirname "$0")" && pwd)
CALLER="${CALLER:-$HERE/../sample_call.cpp}"
OUT="${OUT:-sample_gpu}"

if command -v llc >/dev/null 2>&1; then
  llc -filetype=obj -O3 -relocation-model=pic sample.ll -o sample.o
else
  clang -x ir sample.ll -c -O3 -fPIC -o sample.o
fi
c++ -O3 -c "$CALLER" -o $(basename "$CALLER" .cpp).o
c++ -O3 $(basename "$CALLER" .cpp).o sample.o -o "$OUT" \
  -L"$MLIR_LIB" -lmlir_c_runner_utils -lmlir_runner_utils -lmlir_cuda_runtime \
  -L"$CUDA_LIB" -lcuda -lcudart \
  -Wl,-rpath,"$MLIR_LIB" -Wl,-rpath,"$CUDA_LIB"

echo "built ./$OUT   (watch the GPU with: watch -n 0.5 nvidia-smi)"
