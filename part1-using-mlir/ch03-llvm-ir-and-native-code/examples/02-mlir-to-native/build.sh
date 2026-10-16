#!/bin/bash
# Lower example.mlir (func/index/scf/arith) all the way to native code:
#
#   example.mlir --mlir-opt--> example_opt.mlir (llvm dialect)
#                --mlir-translate--> example.ll (LLVM IR)
#                --clang--> example.o --clang--> libexample.dylib / example
#
# then run the executable, call the shared library from Python, dump the
# assembly and the disassembly, and finally JIT the same module in-process.
# Run from anywhere:
#
#   bash part1-using-mlir/ch03-llvm-ir-and-native-code/examples/02-mlir-to-native/build.sh
#
# Adapted from mlir-ir-lowering's 1_intro/2_mlir/build.sh. `llc` and
# `mlir-runner` from the original are replaced by `clang -x ir` and the MLIR
# Python bindings' ExecutionEngine (jit.py), since the wheel ships neither.

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

# 0. (Inspection) Only the scf -> cf step, to see the loop become blocks.
mlir-opt "$SCRIPT_DIR/example.mlir" --convert-scf-to-cf -o "$B/example_cf.mlir"

# 1. Lower every dialect to the llvm dialect, one conversion pass each.
mlir-opt "$SCRIPT_DIR/example.mlir" \
    --convert-func-to-llvm \
    --convert-math-to-llvm \
    --convert-index-to-llvm \
    --convert-scf-to-cf \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    -o "$B/example_opt.mlir"

# 2. Leave MLIR: llvm dialect -> textual LLVM IR.
mlir-translate "$B/example_opt.mlir" --mlir-to-llvmir -o "$B/example.ll"

# 2b. The wheel's mlir-translate is LLVM 24; clang may be older. This filter
#     rewrites LLVM 24's float-literal syntax for older parsers (a no-op for
#     this integer-only module, but harmless -- every later chapter needs it).
python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$B/example.ll" > "$B/example.compat.ll"

# 3. LLVM IR -> object, then link it two ways.
#    LLC_LIKE = what `llc` does by default: optimize instruction selection and
#    register allocation (-O2) but run NO IR-level optimization passes, so the
#    machine code still has the loop we wrote. (`clang -O2` alone would also
#    run LLVM's IR optimizer -- see step 6.) -Wno-override-module silences
#    clang's note that the module has no target triple and gets the host's.
LLC_LIKE="-Wno-override-module -O2 -Xclang -disable-llvm-passes"
clang $LLC_LIKE -x ir -c -fPIC "$B/example.compat.ll" -o "$B/example.o"
clang -shared -fPIC "$B/example.o" -o "$B/libexample.$SO"
clang "$B/example.o" -o "$B/example"

# 4. Run the executable: main() returns 45 (0+1+...+9) as the exit code.
set +e
"$B/example"
echo "exit code: $?"
set -e

# 5. Same function through ctypes.
echo "from Python (ctypes): $("$PYTHON" -c "
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.main.restype = ctypes.c_int
print(lib.main())" "$B/libexample.$SO")"

# 6. (Inspection) Native assembly at three optimization settings:
#    example.s     -- llc-like: back-end optimization only (the loop survives)
#    example.O0.s  -- no optimization at all (every value spilled to the stack)
#    example.O2.s  -- full clang -O2: the IR optimizer folds the loop to 45
clang $LLC_LIKE -x ir -S "$B/example.compat.ll" -o "$B/example.s"
clang -Wno-override-module -x ir -S -O0 "$B/example.compat.ll" -o "$B/example.O0.s"
clang -Wno-override-module -x ir -S -O2 "$B/example.compat.ll" -o "$B/example.O2.s"

# 7. (Inspection) Disassemble the object and the shared library.
objdump -d --no-show-raw-insn "$B/example.o" > "$B/example.dis"
objdump -d --no-show-raw-insn "$B/libexample.$SO" > "$B/libexample.dis"

# 8. JIT: compile and run the module in memory, no files involved.
echo "JIT (ExecutionEngine): $("$PYTHON" "$SCRIPT_DIR/jit.py")"
