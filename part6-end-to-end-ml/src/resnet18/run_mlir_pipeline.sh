#!/bin/sh
# Lower ResNet-18's linalg snapshot to a native object and link the C++ caller.
#
# Usage (from part6-end-to-end-ml/src/resnet18/, after ./generate_snapshots.sh resnet18):
#   sh run_mlir_pipeline.sh && ./resnet18
#
# The Chapter 28 pipeline, plus one extra step (see Chapter 30, section 6):
# mlir-opt --convert-bufferization-to-memref lowers any bufferization.clone
# the deallocation pipeline left behind; it is a no-op when there is none.
set -eu
MODEL=resnet18_model

ROOT=$(cd ../../.. && pwd)
OPT=$ROOT/build/tools/tutorial-opt
PY=python3
if [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python; fi
MLIR_ROOT=${MLIR_ROOT:-$("$PY" -m mlir_wheel --root-dir)}

"$OPT" --linalg-to-bufferization "${MODEL}_linalg.mlir" \
| "$MLIR_ROOT/bin/mlir-opt" --convert-bufferization-to-memref \
| "$OPT" --llvm-request-c-wrappers --bufferization-to-llvm \
| "$MLIR_ROOT/bin/mlir-translate" --mlir-to-llvmir \
| python3 "$ROOT/tools/ll_fp_compat.py" \
| clang -O3 -x ir - -c -o "$MODEL.o"

if [ "$(uname)" = Darwin ]; then BLAS="-framework Accelerate"; else BLAS="-lopenblas"; fi

c++ -std=c++17 -c resnet18_call.cpp -o resnet18_call.o
# shellcheck disable=SC2086  # $BLAS is deliberately word-split
c++ resnet18_call.o "$MODEL.o" -o resnet18 $BLAS \
    -L"$MLIR_ROOT/lib" -lmlir_c_runner_utils -Wl,-rpath,"$MLIR_ROOT/lib"
echo "built ./resnet18"
