#!/bin/bash
# Compile a hand-written LLVM IR module (simple.ll) three ways -- object file,
# shared library, executable -- then run the executable and call the shared
# library from Python. Run from anywhere:
#
#   bash part1-using-mlir/ch03-llvm-ir-and-native-code/examples/01-llvm-modules/build.sh
#
# Adapted from mlir-ir-lowering's 1_intro/1_llvm_modules/build.sh. The original
# used `llc` for step 1; the MLIR wheel ships no llc, so clang (which contains
# the same LLVM back end) compiles the .ll directly.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
[ -n "$WHEEL_BIN" ] && export PATH=$WHEEL_BIN:$PATH
PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}

case "$(uname)" in Darwin) SO=dylib ;; *) SO=so ;; esac
B=$SCRIPT_DIR/build
mkdir -p "$B"

# 1. LLVM IR -> native object file. `-x ir` tells clang the input is LLVM IR
#    (it would guess from the .ll suffix too). -fPIC = position-independent
#    code, required for a shared library.
clang -x ir -c -fPIC "$SCRIPT_DIR/simple.ll" -o "$B/simple.o"

# 2. Object -> shared library that other programs (Python!) can dlopen.
clang -shared -fPIC "$B/simple.o" -o "$B/libsimple.$SO"

# 3. Object -> standalone executable.
clang "$B/simple.o" -o "$B/simple"

# 4. Run it. main's return value becomes the process exit code.
set +e
"$B/simple"
echo "exit code: $?"
set -e

# 5. Call main() from Python through ctypes.
echo "from Python: $("$PYTHON" "$SCRIPT_DIR/simple.py")"
