#!/bin/sh
# Lower flan-t5-small's linalg export to a position-independent object file.
#
# Usage (from part6-end-to-end-ml/src/flan_t5/, after
# ./generate_snapshots.sh flan_t5):
#   sh run_mlir_pipeline.sh    # -> flan_t5_model.o
#   sh build_flan_call.sh      # -> flan_call<ext-suffix>.so (pybind11)
#   python run_flan_t5_model_mlir.py
#
# The object is compiled as position-independent code (-fPIC) so it can be
# linked into a shared library — the Python extension module. (The source
# tutorial used `llc -relocation-model=pic`; the wheel ships no llc.)
#
# The Chapter 28 pipeline, plus one extra step (see Chapter 30, section 6):
# mlir-opt --convert-bufferization-to-memref lowers any bufferization.clone
# the deallocation pipeline left behind; it is a no-op when there is none.
set -eu
MODEL=flan_t5

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
| clang -O3 -fPIC -x ir - -c -o "${MODEL}_model.o"
echo "built ${MODEL}_model.o"
