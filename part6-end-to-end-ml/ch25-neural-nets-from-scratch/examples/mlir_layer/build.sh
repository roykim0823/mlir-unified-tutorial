#!/usr/bin/env bash
# Build the MLIR dense+ReLU layer (dense_relu.mlir) into a shared library and
# run the NumPy driver against it. Run it from anywhere:
#
#   bash part6-end-to-end-ml/ch25-neural-nets-from-scratch/examples/mlir_layer/build.sh
#
# Outputs go to build/ next to this script (gitignored).
#
# Adapted from mlir-ir-lowering's 5_neural_network/mlir_layer/build.sh. The
# source used `llc` to make an object file; the MLIR wheel ships no llc, so we
# hand the textual LLVM IR straight to clang (`-x ir`) after passing it through
# tools/ll_fp_compat.py, which rewrites LLVM 24's float-literal syntax for an
# older clang (see Chapter 28). It also drops `nocreateundeforpoison`, an
# intrinsic attribute LLVM 24 attaches to @llvm.maximum.f32 that clang <= 20
# does not know ("unterminated attribute group"); it is only an optimization
# hint, so removing it is safe.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$SCRIPT_DIR/../../../.." && pwd )

# Prefer the MLIR wheel's tools in the repo's .venv (see the root README).
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi
PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

# linalg (on memrefs) -> scf loops -> cf branches -> llvm dialect.
mlir-opt "$SCRIPT_DIR"/dense_relu.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "$BUILD"/dense_relu_opt.mlir

# Leave MLIR: llvm dialect -> LLVM IR text -> native shared library.
mlir-translate "$BUILD"/dense_relu_opt.mlir --mlir-to-llvmir -o "$BUILD"/dense_relu.ll
python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$BUILD"/dense_relu.ll > "$BUILD"/dense_relu.compat.ll
clang -shared -fPIC -Wno-override-module -x ir "$BUILD"/dense_relu.compat.ll -o "$BUILD"/libdense.so

"$PYTHON" "$SCRIPT_DIR"/aot_main.py "$BUILD"/libdense.so
