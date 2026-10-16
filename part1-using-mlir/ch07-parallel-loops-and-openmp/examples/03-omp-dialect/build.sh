#!/usr/bin/env bash
# Chapter 7, example 03: a hand-written `omp` dialect program, lowered to LLVM
# IR, compiled with clang, linked against libomp, and run.
#
#   bash part1-using-mlir/ch07-parallel-loops-and-openmp/examples/03-omp-dialect/build.sh
#
# Adapted from mlir-ir-lowering's 3_parallel/5_openmp.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi

OUT=$SCRIPT_DIR/build
rm -rf "$OUT"
mkdir -p "$OUT"

# 1. Lower everything to the llvm dialect (the pass order is explained in
#    ../04-scf-to-openmp/build.sh). --convert-openmp-to-llvm does not
#    remove the omp ops: it converts the types and ops *inside* their regions
#    to the llvm dialect, so that mlir-translate can handle them next.
mlir-opt "$SCRIPT_DIR"/omp_double.mlir \
    --finalize-memref-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-openmp-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$OUT"/omp_double_llvm.mlir

# 2. mlir-translate's OpenMP translation (built on LLVM's OpenMPIRBuilder) is
#    what outlines the parallel region and emits the __kmpc_* runtime calls.
mlir-translate "$OUT"/omp_double_llvm.mlir --mlir-to-llvmir -o "$OUT"/omp_double.ll

echo "=== OpenMP runtime calls in the LLVM IR ==="
grep -o '@__kmpc_[a-z_0-9]*' "$OUT"/omp_double.ll | sort -u
echo "=== outlined parallel region ==="
grep '^define' "$OUT"/omp_double.ll

# 3. Native code: LLVM 24 float syntax -> older clang, then compile and link.
python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$OUT"/omp_double.ll > "$OUT"/omp_double.compat.ll
clang -Wno-override-module -x ir -c "$OUT"/omp_double.compat.ll -o "$OUT"/omp_double.o

source "$SCRIPT_DIR"/../find_libomp.sh
# shellcheck disable=SC2086
clang "$OUT"/omp_double.o $OMP_LDFLAGS -o "$OUT"/omp_double

echo "=== running (input doubled, one value per line) ==="
OMP_NUM_THREADS=4 "$OUT"/omp_double
