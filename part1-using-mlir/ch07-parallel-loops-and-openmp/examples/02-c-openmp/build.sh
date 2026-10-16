#!/usr/bin/env bash
# Chapter 7, example 02: the plain-C OpenMP reference program.
#
#   bash part1-using-mlir/ch07-parallel-loops-and-openmp/examples/02-c-openmp/build.sh
#
# Needs LLVM's OpenMP runtime, libomp: `brew install libomp` on macOS,
# `apt install libomp-dev` on Debian/Ubuntu (see ../find_libomp.sh).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
OUT=$SCRIPT_DIR/build
rm -rf "$OUT"
mkdir -p "$OUT"

# Sets OMP_CFLAGS (compile flags for omp.h) and OMP_LDFLAGS (link flags).
source "$SCRIPT_DIR"/../find_libomp.sh

# clang -fopenmp turns the pragmas into calls into the OpenMP runtime.
# shellcheck disable=SC2086
clang -O1 -fopenmp $OMP_CFLAGS "$SCRIPT_DIR"/omp_double.c $OMP_LDFLAGS \
    -o "$OUT"/omp_double_c

# Also keep clang's LLVM IR, to compare with what MLIR emits in 03/04.
# shellcheck disable=SC2086
clang -O1 -fopenmp $OMP_CFLAGS -S -emit-llvm "$SCRIPT_DIR"/omp_double.c \
    -o "$OUT"/omp_double_c.ll

echo "=== OpenMP runtime calls clang emitted ==="
grep -o '@__kmpc_[a-z_0-9]*' "$OUT"/omp_double_c.ll | sort -u

echo "=== run with 4 threads ==="
OMP_NUM_THREADS=4 "$OUT"/omp_double_c
echo "=== run with 1 thread ==="
OMP_NUM_THREADS=1 "$OUT"/omp_double_c
