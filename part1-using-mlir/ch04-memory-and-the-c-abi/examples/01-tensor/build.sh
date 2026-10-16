#!/bin/bash
# Bufferize, lower and compile the three identity-matrix kernels into shared
# libraries, then call them from Python (aot_main.py). Run from anywhere:
#
#   bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/build.sh
#
# Adapted from mlir-ir-lowering's 2_memory/1_tensor/build.sh.

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

# (Inspection) the pure-tensor catalog: parse/verify, then canonicalize to
# watch the unused tensor.empty ops disappear as dead code.
mlir-opt "$SCRIPT_DIR/tensor_ops.mlir" -o "$B/tensor_ops.roundtrip.mlir"
mlir-opt "$SCRIPT_DIR/tensor_ops.mlir" --canonicalize -o "$B/tensor_ops.canonical.mlir"

lower() {  # $1 = basename without .mlir
    # Stage 1 -- bufferize: tensors become memrefs, including in the function
    # signature (bufferize-function-boundaries). Saved for reading.
    mlir-opt "$SCRIPT_DIR/$1.mlir" \
        --one-shot-bufferize="bufferize-function-boundaries=true" \
        -o "$B/$1_bufferized.mlir"
    # Stage 2 -- free temporaries (one-shot-bufferize inserts no deallocs),
    # tidy up, and lower everything to the llvm dialect.
    mlir-opt "$B/$1_bufferized.mlir" \
        --buffer-deallocation-pipeline \
        --canonicalize \
        --convert-linalg-to-loops \
        --convert-bufferization-to-memref \
        --finalize-memref-to-llvm \
        --convert-scf-to-cf \
        --convert-cf-to-llvm \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$B/$1_llvm.mlir"
    mlir-translate "$B/$1_llvm.mlir" --mlir-to-llvmir -o "$B/$1.ll"
    python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$B/$1.ll" > "$B/$1.compat.ll"
    clang -Wno-override-module -O2 -x ir -c -fPIC "$B/$1.compat.ll" -o "$B/$1.o"
    clang -shared -fPIC "$B/$1.o" -o "$B/lib$1.$SO"
}

lower identity_fill
lower identity_fill_inplace
lower identity_return

"$PYTHON" "$SCRIPT_DIR/aot_main.py"
