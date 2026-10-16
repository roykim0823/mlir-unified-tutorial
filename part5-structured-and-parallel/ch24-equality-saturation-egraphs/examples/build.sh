#!/bin/bash
# Chapter 24: e-graph optimization, then emit MLIR, compile it, and run it.
#
#   bash part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/build.sh
#
# Everything is written to build/ next to this script (gitignored).
#
# Adapted from mlir-ir-lowering's 6_egraph/build.sh. The original used
# `llc` to make an object file; the MLIR wheel ships no llc, so we hand the
# LLVM IR straight to clang (after tools/ll_fp_compat.py, which rewrites
# LLVM 24's float-literal syntax for older clangs -- see Chapter 28).

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$SCRIPT_DIR/../../.." &> /dev/null && pwd )

# Prefer the tools from the MLIR wheel inside the repo's .venv.
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi

BUILD=$SCRIPT_DIR/build
rm -rf "$BUILD"
mkdir -p "$BUILD"

echo "=== the phase-ordering problem (greedy, destructive rewriting) ==="
python3 "$SCRIPT_DIR"/greedy_demo.py

echo
echo "=== concept demos (pure-Python e-graph) ==="
python3 "$SCRIPT_DIR"/optimize_demo.py

echo
echo "=== inside the e-graph for (a*2)/2 ==="
python3 "$SCRIPT_DIR"/inspect_egraph.py

echo
echo "=== capstone: optimize an expression, emit MLIR ==="
python3 "$SCRIPT_DIR"/capstone.py "$BUILD"   # writes orig.mlir and opt.mlir

echo
echo "=== for comparison: what MLIR's own greedy canonicalizer + CSE do ==="
mlir-opt "$BUILD"/orig.mlir --canonicalize --cse -o "$BUILD"/orig_canon.mlir
cat "$BUILD"/orig_canon.mlir

# Compile both the original and the e-graph-optimized MLIR to shared libraries.
for name in orig opt; do
    mlir-opt "$BUILD/${name}.mlir" \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$BUILD/${name}_llvm.mlir"
    mlir-translate "$BUILD/${name}_llvm.mlir" --mlir-to-llvmir -o "$BUILD/${name}.ll"
    python3 "$REPO_ROOT"/tools/ll_fp_compat.py < "$BUILD/${name}.ll" > "$BUILD/${name}.compat.ll"
    clang -shared -fPIC -Wno-override-module -x ir "$BUILD/${name}.compat.ll" -o "$BUILD/lib${name}.so"
done

echo
echo "=== run both compiled functions and compare ==="
python3 "$SCRIPT_DIR"/run_compiled.py "$BUILD"
