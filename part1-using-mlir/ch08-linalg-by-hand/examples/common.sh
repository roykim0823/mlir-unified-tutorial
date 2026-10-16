# Shared setup for the Chapter 8 build scripts. Source it, don't run it:
#
#   source "$SCRIPT_DIR/../common.sh"
#
# It puts the MLIR wheel's tools first on PATH, picks a Python with NumPy,
# and defines compile_kernel, which takes a linalg-on-memref file all the way
# to a shared library that run.py loads with ctypes.

EXAMPLES_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$EXAMPLES_DIR/../../.." &> /dev/null && pwd )

# Prefer mlir-opt/mlir-translate from the MLIR wheel inside the repo's .venv
# (see the root README); fall back to whatever is on PATH.
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
    export PATH=$WHEEL_BIN:$PATH
elif ! command -v mlir-opt &> /dev/null; then
    echo "error: mlir-opt not found (activate the venv or install the wheel)" >&2
    exit 1
fi

# The drivers need NumPy; the repo's venv has it.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON=$REPO_ROOT/.venv/bin/python
else
    PYTHON=python3
fi

# compile_kernel <input.mlir> <name> <build-dir>
#
# Lowers a function whose linalg ops work on memrefs (and that carries
# `llvm.emit_c_interface`) to <build-dir>/lib<name>.so:
#   linalg -> scf loops -> cf -> llvm dialect  (mlir-opt)
#   llvm dialect -> LLVM IR                    (mlir-translate)
#   LLVM IR -> shared library                  (clang)
# ll_fp_compat.py rewrites LLVM 24 float literals and strips LLVM-24-only
# attributes so older clangs accept the IR
# (Chapter 3 explains why). The .so suffix is used on every platform; dlopen
# does not care about the extension.
compile_kernel() {
    local input=$1 name=$2 out=$3
    mlir-opt "$input" \
        --convert-linalg-to-loops \
        --convert-scf-to-cf \
        --convert-cf-to-llvm \
        --convert-arith-to-llvm \
        --finalize-memref-to-llvm \
        --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$out/${name}_llvm.mlir"
    mlir-translate "$out/${name}_llvm.mlir" --mlir-to-llvmir -o "$out/$name.ll"
    python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$out/$name.ll" > "$out/$name.compat.ll"
    clang -shared -fPIC -Wno-override-module -x ir "$out/$name.compat.ll" -o "$out/lib$name.so"
}
