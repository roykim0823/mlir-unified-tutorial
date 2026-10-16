#!/bin/bash
# Build every example in this directory into build/:
#
#   structs_arrays.mlir, vectors.mlir -- lowered, translated, compiled, and
#       called from driver.c (the C types match the LLVM types exactly)
#   bufferization.mlir   -- tensor -> memref three ways (inspect the output)
#   c_interface.mlir     -- the _mlir_ciface_ wrapper (inspect the output)
#   array_add_vec.mlir   -- SIMD array_add, compiled and called from Python
#
# Run from anywhere:
#
#   bash part1-using-mlir/ch04-memory-and-the-c-abi/examples/03-low-level-llvm/build.sh
#
# Adapted from mlir-ir-lowering's 2_memory/3_low_level_llvm/build.sh and
# build_array_add_vec.sh.

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

to_object() {  # $1 = .ll file, $2 = .o file
    python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$1" > "${1%.ll}.compat.ll"
    clang -Wno-override-module -O2 -x ir -c -fPIC "${1%.ll}.compat.ll" -o "$2"
}

echo "== structs_arrays: llvm-dialect structs and arrays =="
# Already entirely in the llvm dialect, so it translates straight to LLVM IR.
mlir-translate "$SCRIPT_DIR/structs_arrays.mlir" --mlir-to-llvmir -o "$B/structs_arrays.ll"
to_object "$B/structs_arrays.ll" "$B/structs_arrays.o"

echo "== vectors: the vector dialect =="
mlir-opt "$SCRIPT_DIR/vectors.mlir" \
    --convert-vector-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$B/vectors_llvm.mlir"
mlir-translate "$B/vectors_llvm.mlir" --mlir-to-llvmir -o "$B/vectors.ll"
to_object "$B/vectors.ll" "$B/vectors.o"

clang -O2 "$SCRIPT_DIR/driver.c" "$B/structs_arrays.o" "$B/vectors.o" -o "$B/driver"
"$B/driver"

echo "== bufferization: tensor -> memref =="
mlir-opt "$SCRIPT_DIR/bufferization.mlir" --canonicalize -o "$B/bufferization_canonical.mlir"
mlir-opt "$SCRIPT_DIR/bufferization.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries" \
    -o "$B/bufferization_oneshot.mlir"
mlir-opt "$SCRIPT_DIR/bufferization.mlir" \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    -o "$B/bufferization_oneshot_identity.mlir"
echo "wrote build/bufferization_{canonical,oneshot,oneshot_identity}.mlir"

echo "== c_interface: the _mlir_ciface_ wrapper =="
mlir-opt "$SCRIPT_DIR/c_interface.mlir" \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$B/c_interface_llvm.mlir"
mlir-translate "$B/c_interface_llvm.mlir" --mlir-to-llvmir -o "$B/c_interface.ll"
echo "wrote build/c_interface_llvm.mlir and build/c_interface.ll"

echo "== array_add_vec: runnable SIMD kernel =="
mlir-opt "$SCRIPT_DIR/array_add_vec.mlir" \
    --convert-vector-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$B/array_add_vec_llvm.mlir"
mlir-translate "$B/array_add_vec_llvm.mlir" --mlir-to-llvmir -o "$B/array_add_vec.ll"
to_object "$B/array_add_vec.ll" "$B/array_add_vec.o"
clang -shared -fPIC "$B/array_add_vec.o" -o "$B/libarray_add_vec.$SO"
# Back-end-only -O2 (as llc would do), so the loop is not unrolled away.
clang -Wno-override-module -O2 -Xclang -disable-llvm-passes -x ir -S \
    "$B/array_add_vec.compat.ll" -o "$B/array_add_vec.s"
"$PYTHON" "$SCRIPT_DIR/aot_main.py"
