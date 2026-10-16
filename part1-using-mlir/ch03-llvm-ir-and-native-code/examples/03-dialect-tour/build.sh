#!/bin/bash
# Lower dialect_tour.mlir (func, arith, math, index, cf, scf) to the llvm
# dialect, compile it, link it with a C driver, and run it. Run from anywhere:
#
#   bash part1-using-mlir/ch03-llvm-ir-and-native-code/examples/03-dialect-tour/build.sh
#
# The functions come from the "Standard MLIR dialects" tour in the top-level
# README of mlir-ir-lowering, made runnable here.

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

B=$SCRIPT_DIR/build
mkdir -p "$B"

# One conversion pass per dialect. Order matters only where one pass produces
# what another consumes: scf-to-cf must come before cf-to-llvm.
mlir-opt "$SCRIPT_DIR/dialect_tour.mlir" \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --convert-math-to-llvm \
    --convert-index-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    -o "$B/dialect_tour_llvm.mlir"

mlir-translate "$B/dialect_tour_llvm.mlir" --mlir-to-llvmir -o "$B/dialect_tour.ll"
# Bridge LLVM 24 -> older clang: ll_fp_compat.py rewrites float literals and
# drops `nocreateundeforpoison`, a function attribute LLVM 24 puts on
# intrinsic declarations (here llvm.sin / llvm.sqrt) that clang 20 rejects.
python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$B/dialect_tour.ll" > "$B/dialect_tour.compat.ll"

clang -Wno-override-module -O2 -x ir -c "$B/dialect_tour.compat.ll" -o "$B/dialect_tour.o"
clang -O2 "$SCRIPT_DIR/driver.c" "$B/dialect_tour.o" -lm -o "$B/driver"
"$B/driver"
