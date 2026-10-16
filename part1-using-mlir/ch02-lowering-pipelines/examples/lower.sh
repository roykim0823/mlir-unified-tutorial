#!/bin/bash
# Stage-by-stage lowering of a linalg matmul+relu program down to LLVM IR.
#
# Regenerates the numbered stage files in this directory from
# 0-linalg-on-tensor.mlir. Every IR listing in the chapter README comes from
# these files. Run it as:
#
#   bash part1-using-mlir/ch02-lowering-pipelines/examples/lower.sh
#
# Adapted from Alexandre Singer's mlir-beginner-friendly-tutorial (MIT),
# demo3-lowering-mlir/lower.sh.

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Prefer mlir-opt/mlir-translate from the MLIR wheel inside the repo's .venv
# (see the root README for setup); fall back to whatever is on PATH. Putting
# the wheel first keeps the output stable even if another LLVM (e.g. from
# Homebrew) is installed system-wide.
REPO_ROOT=$SCRIPT_DIR/../../..
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi

# Stage 0 -> 1: lower tensors to memrefs. Since we are targeting a CPU, every
# chunk of data needs an allocated buffer somewhere in memory. The
# one-shot-bufferize pass converts tensors into memrefs (pointers with shape
# information attached); canonicalize then cleans up the result.
mlir-opt \
    "$SCRIPT_DIR"/0-linalg-on-tensor.mlir \
    --one-shot-bufferize="bufferize-function-boundaries=true" \
    --canonicalize \
    -o "$SCRIPT_DIR"/1-linalg-on-memref.mlir

# Stage 1 -> 2: convert the linalg operations into loops in the scf dialect.
# Our target CPU cannot compute matmul and relu directly, so the named kernels
# become explicit loop nests -- much closer to what a CPU can execute.
mlir-opt \
    "$SCRIPT_DIR"/1-linalg-on-memref.mlir \
    --convert-linalg-to-loops \
    -o "$SCRIPT_DIR"/2-scf-loops.mlir

# Stage 2 -> 3: convert structured loops into basic blocks and branches
# (the cf dialect), which is how CPUs actually represent control flow.
mlir-opt \
    "$SCRIPT_DIR"/2-scf-loops.mlir \
    --convert-scf-to-cf \
    -o "$SCRIPT_DIR"/3-cf-branches.mlir

# Stage 3 -> 4: convert everything that remains (func, cf, memref, arith)
# into the llvm dialect. reconcile-unrealized-casts removes the temporary
# type-bridging casts the partial conversions leave behind.
mlir-opt \
    "$SCRIPT_DIR"/3-cf-branches.mlir \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    --canonicalize \
    -o "$SCRIPT_DIR"/4-llvm-dialect.mlir

# Stage 4 -> 5: leave MLIR. mlir-translate turns llvm-dialect MLIR into real
# LLVM IR, which clang/llc can compile the rest of the way to machine code.
mlir-translate \
    "$SCRIPT_DIR"/4-llvm-dialect.mlir \
    --mlir-to-llvmir \
    -o "$SCRIPT_DIR"/5-llvm-ir.ll
