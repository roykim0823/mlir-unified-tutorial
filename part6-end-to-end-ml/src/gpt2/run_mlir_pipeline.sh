#!/bin/sh
# Lower the full GPT-2 linalg export to a native object file (Chapter 28, §6).
#
# Usage (from part6-end-to-end-ml/src/gpt2/, after ./generate_snapshots.sh gpt2,
# which writes the ~950 MB gpt2_linalg.mlir):
#   sh run_mlir_pipeline.sh
#
# There is no C++ caller for GPT-2: the result is gpt2.o, whose
# _mlir_ciface_gpt2 entry point a generation loop could call (Chapter 28's
# capstone exercise; Chapter 30's flan_t5 harness shows the pattern).
# Stages are written to files so a failing step stops the script instead of
# feeding an empty module to the next one.
set -eu
MODEL=gpt2

ROOT=$(cd ../../.. && pwd)
OPT=$ROOT/build/tools/tutorial-opt
PY=python3
if [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python; fi
MLIR_ROOT=${MLIR_ROOT:-$("$PY" -m mlir_wheel --root-dir)}

"$OPT" --linalg-to-bufferization "${MODEL}_linalg.mlir" -o "${MODEL}_buf.mlir"
"$MLIR_ROOT/bin/mlir-opt" --convert-bufferization-to-memref "${MODEL}_buf.mlir" \
    -o "${MODEL}_buf2.mlir"
"$OPT" --llvm-request-c-wrappers --bufferization-to-llvm "${MODEL}_buf2.mlir" \
    -o "${MODEL}_llvm.mlir"
# The textual-IR route (mlir-translate | ll_fp_compat.py | clang -x ir) fails
# here: GPT-2's .ll is >3 GB and clang stops at ~2 GB per source file. Compile
# the LLVM-dialect module in process instead (see tools/mlir_to_object.py).
"$PY" "$ROOT/tools/mlir_to_object.py" "${MODEL}_llvm.mlir" "${MODEL}.o" \
    "$MLIR_ROOT/lib/libmlir_c_runner_utils$( [ "$(uname)" = Darwin ] && echo .dylib || echo .so )"
rm -f "${MODEL}_buf.mlir" "${MODEL}_buf2.mlir" "${MODEL}_llvm.mlir"

echo "built ${MODEL}.o"
nm "${MODEL}.o" | grep ' T .*_mlir_ciface' || true
echo "undefined symbols (link these in):"
nm "${MODEL}.o" | grep ' U ' || true
