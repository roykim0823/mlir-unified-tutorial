#!/bin/sh
# Build the pybind11 extension module `flan_call` from flan_call.cpp and the
# compiled model object (flan_t5_model.o, from run_mlir_pipeline.sh).
#
# Usage (from part6-end-to-end-ml/src/flan_t5/):
#   PYTHON=/path/to/python sh build_flan_call.sh
# PYTHON must be the interpreter that will import the module, with pybind11
# installed (pip install pybind11 transformers — no torch needed).
set -eu
PYTHON=${PYTHON:-python3}

ROOT=$(cd ../../.. && pwd)
PY=python3
if [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python; fi
MLIR_ROOT=${MLIR_ROOT:-$("$PY" -m mlir_wheel --root-dir)}

INCLUDES=$("$PYTHON" -m pybind11 --includes)
SUFFIX=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

if [ "$(uname)" = Darwin ]; then
  BLAS="-framework Accelerate"
  # Python symbols are resolved when the interpreter loads the module.
  PYLINK="-undefined dynamic_lookup"
else
  BLAS="-lopenblas"
  PYLINK=""
fi

# shellcheck disable=SC2086  # $INCLUDES, $BLAS, $PYLINK are word-split on purpose
c++ -O3 -Wall -shared -std=c++17 -fPIC $INCLUDES \
    flan_call.cpp flan_t5_model.o -o "flan_call$SUFFIX" \
    $BLAS $PYLINK -lm \
    -L"$MLIR_ROOT/lib" -lmlir_c_runner_utils -Wl,-rpath,"$MLIR_ROOT/lib"
echo "built flan_call$SUFFIX"
