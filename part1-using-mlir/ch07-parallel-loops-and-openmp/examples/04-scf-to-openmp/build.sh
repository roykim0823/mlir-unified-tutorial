#!/usr/bin/env bash
# Chapter 7, example 04: scf.parallel -> omp dialect -> LLVM IR -> native code
# linked against libomp, driven from C.
#
#   bash part1-using-mlir/ch07-parallel-loops-and-openmp/examples/04-scf-to-openmp/build.sh
#
# Adapted from mlir-ir-lowering's 3_parallel/5_openmp/scf_parallel.mlir.
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

# 1. scf.parallel -> omp.parallel { omp.wsloop { omp.loop_nest } }.
mlir-opt "$SCRIPT_DIR"/scf_parallel.mlir --convert-scf-to-openmp \
    -o "$OUT"/scf_parallel_omp.mlir

# 2. The rest of the CPU pipeline, with --convert-openmp-to-llvm added (the
#    same pass list as examples 03 and 05). finalize-memref-to-llvm runs
#    FIRST: it lowers the memref.alloca_scope that convert-scf-to-openmp
#    wraps around each loop body (to llvm.intr.stacksave/stackrestore), and
#    must do so before convert-scf-to-cf turns any scf.for inside that scope
#    into multiple blocks, which alloca_scope does not allow (example 05).
mlir-opt "$OUT"/scf_parallel_omp.mlir \
    --finalize-memref-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-openmp-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    -o "$OUT"/scf_parallel_llvm.mlir
mlir-translate "$OUT"/scf_parallel_llvm.mlir --mlir-to-llvmir -o "$OUT"/scf_parallel.ll

echo "=== OpenMP runtime calls in the LLVM IR ==="
grep -o '@__kmpc_[a-z_0-9]*' "$OUT"/scf_parallel.ll | sort -u

# 3. Compile, link with the C driver and libomp, run.
python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$OUT"/scf_parallel.ll > "$OUT"/scf_parallel.compat.ll
clang -Wno-override-module -x ir -c "$OUT"/scf_parallel.compat.ll -o "$OUT"/scf_parallel.o
clang -c "$SCRIPT_DIR"/driver.c -o "$OUT"/driver.o

source "$SCRIPT_DIR"/../find_libomp.sh
# shellcheck disable=SC2086
clang "$OUT"/driver.o "$OUT"/scf_parallel.o $OMP_LDFLAGS -o "$OUT"/scf_parallel

for t in 4 2 1; do
    echo "=== OMP_NUM_THREADS=$t ==="
    OMP_NUM_THREADS=$t "$OUT"/scf_parallel
done
