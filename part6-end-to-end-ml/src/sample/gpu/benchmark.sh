#!/bin/sh
# Build the GPU benchmark binary for the sample model - Linux + NVIDIA GPU + CUDA only.
# Ported from ML-compiler-exercise src/sample/gpu/benchmark.sh (RWTH /cvmfs CUDA
# paths replaced by $CUDA_HOME). Same prerequisites as compile.sh; this just
# swaps in the timing caller. Run ./bench_gpu afterwards.
# STATUS: not verified in this port (no NVIDIA GPU on the porting machine).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CALLER="$HERE/sample_call_benchmark.cpp" OUT=bench_gpu sh "$HERE/compile.sh"
