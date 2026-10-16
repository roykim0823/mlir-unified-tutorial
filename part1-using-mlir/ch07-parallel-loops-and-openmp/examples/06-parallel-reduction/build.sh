#!/usr/bin/env bash
# Chapter 7, example 06: parallel reductions through OpenMP.
#
#   bash part1-using-mlir/ch07-parallel-loops-and-openmp/examples/06-parallel-reduction/build.sh
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

# 1. Parallelize the reduction (only @sum is affine) and keep the stages.
mlir-opt "$SCRIPT_DIR"/reduce.mlir --affine-parallelize=parallel-reductions=true \
    -o "$OUT"/reduce_affine_parallel.mlir
mlir-opt "$OUT"/reduce_affine_parallel.mlir --lower-affine --canonicalize \
    -o "$OUT"/reduce_scf_parallel.mlir
mlir-opt "$OUT"/reduce_scf_parallel.mlir --convert-scf-to-openmp \
    -o "$OUT"/reduce_omp.mlir

# 2. Down to LLVM IR (same pass list as examples 03-05).
mlir-opt "$OUT"/reduce_omp.mlir \
    --finalize-memref-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-openmp-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$OUT"/reduce_llvm.mlir
mlir-translate "$OUT"/reduce_llvm.mlir --mlir-to-llvmir -o "$OUT"/reduce.ll

echo "=== OpenMP runtime calls in the LLVM IR ==="
grep -o '@__kmpc_[a-z_0-9]*' "$OUT"/reduce.ll | sort -u

# 3. Compile, link, run.
python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$OUT"/reduce.ll > "$OUT"/reduce.compat.ll
clang -O2 -Wno-override-module -x ir -c "$OUT"/reduce.compat.ll -o "$OUT"/reduce.o
clang -O2 -c "$SCRIPT_DIR"/driver.c -o "$OUT"/driver.o

source "$SCRIPT_DIR"/../find_libomp.sh
# shellcheck disable=SC2086
clang "$OUT"/driver.o "$OUT"/reduce.o $OMP_LDFLAGS -o "$OUT"/reduce

for t in 1 4; do
    OMP_NUM_THREADS=$t "$OUT"/reduce
done
