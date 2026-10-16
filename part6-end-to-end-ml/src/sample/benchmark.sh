#!/bin/sh
# Build and run the C++ benchmark for the sample model (CPU, MLIR-compiled).
#
# Usage (from part6-end-to-end-ml/src/sample/, after building the repo):
#   sh benchmark.sh
#
# Lowers the checked-in sample_model_linalg.mlir with the Chapter 28
# pipeline (unless sample_model.o already exists), compiles
# sample_call_benchmark.cpp with -O3, links against BLAS, and runs it.
# Compare with: python benchmark_sample_model.py (needs torch).
set -eu
MODEL=sample_model
CALLER=sample_call_benchmark

ROOT=$(cd ../../.. && pwd)
OPT=$ROOT/build/tools/tutorial-opt
PY=python3
if [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python; fi
MLIR_ROOT=${MLIR_ROOT:-$("$PY" -m mlir_wheel --root-dir)}

if [ ! -f "$MODEL.o" ]; then
  "$OPT" --linalg-to-bufferization "${MODEL}_linalg.mlir" \
  | "$OPT" --llvm-request-c-wrappers --bufferization-to-llvm \
  | "$MLIR_ROOT/bin/mlir-translate" --mlir-to-llvmir \
  | python3 "$ROOT/tools/ll_fp_compat.py" \
  | clang -O3 -x ir - -c -o "$MODEL.o"
fi

if [ "$(uname)" = Darwin ]; then BLAS="-framework Accelerate"; else BLAS="-lopenblas"; fi

c++ -O3 -std=c++17 -c "$CALLER.cpp" -o "$CALLER.o"
# shellcheck disable=SC2086  # $BLAS is deliberately word-split
c++ "$CALLER.o" "$MODEL.o" -o bench.out $BLAS \
    -L"$MLIR_ROOT/lib" -lmlir_c_runner_utils -Wl,-rpath,"$MLIR_ROOT/lib"
./bench.out
