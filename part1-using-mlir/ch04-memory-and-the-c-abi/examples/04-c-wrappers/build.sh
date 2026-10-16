#!/bin/bash
# Lower add_vector_to_matrix.mlir twice -- once as written (only the function
# carrying llvm.emit_c_interface gets a _mlir_ciface_ wrapper) and once with
# --llvm-request-c-wrappers (every function gets one) -- then call the
# results from Python (aot_main.py) and from C (main.c). Run from anywhere:
#
#   bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/04-c-wrappers/build.sh
#
# Adapted from mlir-ir-lowering's 2_memory/4_C_compatible_wrappers/build.sh.

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

# Tensors -> memrefs. identity-layout-map keeps the signatures as plain
# memref<?x?xf32> instead of memref<?x?xf32, strided<[?, ?], offset: ?>>.
mlir-opt "$SCRIPT_DIR/add_vector_to_matrix.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    -o "$B/add_vector_to_matrix_bufferized.mlir"

lower() {  # $1 = output name, remaining args = extra leading passes
    local name=$1; shift
    # --llvm-request-c-wrappers (when given) must run BEFORE
    # --convert-func-to-llvm, which is the pass that emits the wrappers.
    mlir-opt "$B/add_vector_to_matrix_bufferized.mlir" "$@" \
        --buffer-deallocation-pipeline \
        --canonicalize \
        --convert-linalg-to-loops \
        --convert-scf-to-cf \
        --convert-cf-to-llvm \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        --finalize-memref-to-llvm \
        --reconcile-unrealized-casts \
        -o "$B/${name}_llvm.mlir"
    mlir-translate "$B/${name}_llvm.mlir" --mlir-to-llvmir -o "$B/$name.ll"
    python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$B/$name.ll" > "$B/$name.compat.ll"
    clang -Wno-override-module -O2 -x ir -c -fPIC "$B/$name.compat.ll" -o "$B/$name.o"
    clang -shared -fPIC "$B/$name.o" -o "$B/lib$name.$SO"
}

lower add_vector_to_matrix
lower add_vector_to_matrix_all --llvm-request-c-wrappers

echo "exported symbols:"
nm -gU "$B/libadd_vector_to_matrix.$SO" | sed 's/^/  /'
echo "with --llvm-request-c-wrappers:"
nm -gU "$B/libadd_vector_to_matrix_all.$SO" | sed 's/^/  /'

echo "== Python (aot_main.py) =="
"$PYTHON" "$SCRIPT_DIR/aot_main.py"

echo "== C (main.c) =="
clang -O2 "$SCRIPT_DIR/main.c" "$B/add_vector_to_matrix.o" -o "$B/main"
"$B/main"
