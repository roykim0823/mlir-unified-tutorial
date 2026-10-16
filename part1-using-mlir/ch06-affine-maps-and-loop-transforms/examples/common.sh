# Shared helpers for the Chapter 6 example scripts (sourced, not executed).
#
# Expects the caller to have set SCRIPT_DIR. Defines REPO_ROOT, puts the MLIR
# wheel's tools first on PATH, and provides `lower_to_shared_lib`.

REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
WHEEL_BIN=$(cd "$REPO_ROOT"/.venv/lib/python3*/site-packages/mlir_wheel/bin 2> /dev/null && pwd || true)
if [ -n "$WHEEL_BIN" ]; then
  export PATH="$WHEEL_BIN:$PATH"
elif ! command -v mlir-opt &> /dev/null; then
  echo "error: mlir-opt not found (create .venv with the MLIR wheel, see the root README)" >&2
  exit 1
fi

# Prefer the venv's python (it has numpy); fall back to python3 on PATH.
PYTHON="$REPO_ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

# lower_to_shared_lib <input.mlir> <out_dir> <name>
#
# affine -> scf -> cf -> llvm dialect -> LLVM IR -> lib<name>.dylib/.so.
# Writes <name>_llvm.mlir, <name>.ll and the library into <out_dir>. The
# ll_fp_compat.py filter rewrites LLVM 24 output for the older clang (see
# Chapter 3): float literal syntax, and the `nocreateundeforpoison` attribute
# on intrinsics such as @llvm.smax.i64 that clang <= 20 cannot parse.
lower_to_shared_lib() {
  local in=$1 out=$2 name=$3
  mlir-opt "$in" \
    -lower-affine \
    -convert-scf-to-cf \
    -convert-cf-to-llvm \
    -convert-arith-to-llvm \
    -convert-index-to-llvm \
    -finalize-memref-to-llvm \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o "$out/${name}_llvm.mlir"
  mlir-translate --mlir-to-llvmir "$out/${name}_llvm.mlir" -o "$out/$name.ll"
  python3 "$REPO_ROOT/tools/ll_fp_compat.py" < "$out/$name.ll" > "$out/$name.compat.ll"
  local ext=so
  [ "$(uname)" = Darwin ] && ext=dylib
  clang -shared -fPIC -O2 -Wno-override-module -x ir "$out/$name.compat.ll" -o "$out/lib$name.$ext"
}
