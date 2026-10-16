#!/bin/sh
# Build and run the C++ benchmark for the CNN (CPU, MLIR-compiled).
#
# Usage (from part6-end-to-end-ml/src/cnn/):
#   sh run_mlir_pipeline.sh   # once: produces cnn_model.o
#   sh benchmark.sh
# Compare with: python benchmark_cnn_model.py (needs torch).
set -eu
MODEL=cnn_model
CALLER=cnn_call_benchmark

ROOT=$(cd ../../.. && pwd)
PY=python3
if [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python; fi
MLIR_ROOT=${MLIR_ROOT:-$("$PY" -m mlir_wheel --root-dir)}

if [ ! -f "$MODEL.o" ]; then sh run_mlir_pipeline.sh; fi

if [ "$(uname)" = Darwin ]; then BLAS="-framework Accelerate"; else BLAS="-lopenblas"; fi

c++ -O3 -std=c++17 -c "$CALLER.cpp" -o "$CALLER.o"
# shellcheck disable=SC2086  # $BLAS is deliberately word-split
c++ "$CALLER.o" "$MODEL.o" -o bench.out $BLAS \
    -L"$MLIR_ROOT/lib" -lmlir_c_runner_utils -Wl,-rpath,"$MLIR_ROOT/lib"
./bench.out
