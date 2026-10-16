#!/bin/bash
# Compile array_add.mlir to a shared library and call it from Python twice:
# ahead of time through ctypes (aot_main.py), and just in time with the MLIR
# ExecutionEngine (jit_main.py, which does not use the build/ files at all).
# Run from anywhere:
#
#   bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/02-array-add/build.sh
#
# Adapted from mlir-ir-lowering's 2_memory/2_array_add/build.sh.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi
PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}
case "$(uname)" in Darwin) SO=dylib ;; *) SO=so ;; esac
B=$SCRIPT_DIR/build
mkdir -p "$B"

# The one new pass compared with Chapter 3 is --finalize-memref-to-llvm: it
# turns memref.load/store and the memref descriptor itself into plain LLVM
# pointer arithmetic.
mlir-opt "$SCRIPT_DIR/array_add.mlir" \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$B/array_add_llvm.mlir"
mlir-translate "$B/array_add_llvm.mlir" --mlir-to-llvmir -o "$B/array_add.ll"
python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$B/array_add.ll" > "$B/array_add.compat.ll"
clang -Wno-override-module -O2 -x ir -c -fPIC "$B/array_add.compat.ll" -o "$B/array_add.o"
clang -shared -fPIC "$B/array_add.o" -o "$B/libarray_add.$SO"

echo "python aot_main.py"
"$PYTHON" "$SCRIPT_DIR/aot_main.py"

echo "python jit_main.py"
"$PYTHON" "$SCRIPT_DIR/jit_main.py"
