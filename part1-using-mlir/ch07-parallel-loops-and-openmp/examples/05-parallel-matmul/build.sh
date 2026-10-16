#!/usr/bin/env bash
# Chapter 7, example 05: the affine matmul, automatically parallelized and run
# on CPU threads through OpenMP.
#
#   affine.for --affine-parallelize--> affine.parallel --lower-affine-->
#   scf.parallel --convert-scf-to-openmp--> omp.wsloop --...--> LLVM IR + libomp
#
#   bash part1-using-mlir/ch07-parallel-loops-and-openmp/examples/05-parallel-matmul/build.sh
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

# 1. Find the parallelism and keep it all the way to the omp dialect.
#    --canonicalize merges the two nested scf.parallel loops into one 2-D
#    scf.parallel, which becomes a single collapse(2) worksharing loop.
mlir-opt "$SCRIPT_DIR"/matmul.mlir \
    --affine-parallelize \
    --lower-affine \
    --canonicalize \
    --convert-scf-to-openmp \
    -o "$OUT"/matmul_omp.mlir

# 2. Down to the llvm dialect and LLVM IR (same pass list as examples 03/04).
mlir-opt "$OUT"/matmul_omp.mlir \
    --finalize-memref-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-openmp-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$OUT"/matmul_llvm.mlir
mlir-translate "$OUT"/matmul_llvm.mlir --mlir-to-llvmir -o "$OUT"/matmul.ll

echo "=== OpenMP runtime calls in the LLVM IR ==="
grep -o '@__kmpc_[a-z_0-9]*' "$OUT"/matmul.ll | sort -u

# 3. Native code. -O2 on both sides so the timing comparison is fair.
python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$OUT"/matmul.ll > "$OUT"/matmul.compat.ll
clang -O2 -Wno-override-module -x ir -c "$OUT"/matmul.compat.ll -o "$OUT"/matmul.o
clang -O2 -c "$SCRIPT_DIR"/driver.c -o "$OUT"/driver.o

source "$SCRIPT_DIR"/../find_libomp.sh
# shellcheck disable=SC2086
clang "$OUT"/driver.o "$OUT"/matmul.o $OMP_LDFLAGS -o "$OUT"/matmul

NPROC=$(getconf _NPROCESSORS_ONLN 2> /dev/null || echo 4)
for t in 1 "$NPROC"; do
    OMP_NUM_THREADS=$t "$OUT"/matmul 512
done
