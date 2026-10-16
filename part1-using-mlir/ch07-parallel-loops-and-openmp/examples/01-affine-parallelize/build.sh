#!/usr/bin/env bash
# Chapter 7, example 01: which loops are parallel, and what --affine-parallelize
# does with them. Everything here is IR -> IR; nothing is compiled or run.
#
#   bash part1-using-mlir/ch07-parallel-loops-and-openmp/examples/01-affine-parallelize/build.sh
#
# Adapted from mlir-ir-lowering's 3_parallel/1_affine_matmul.
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

# 1. Dependence analysis on six small loops. Without parallel-reductions the
#    @sum loop (a loop-carried iter_args value) stays sequential.
mlir-opt "$SCRIPT_DIR"/deps.mlir --affine-parallelize \
    -o "$OUT"/deps_parallel.mlir
mlir-opt "$SCRIPT_DIR"/deps.mlir --affine-parallelize=parallel-reductions=true \
    -o "$OUT"/deps_parallel_reductions.mlir

echo "=== loop kinds after --affine-parallelize (parallel-reductions=true) ==="
awk '/func.func/ { split($2, a, "("); name = a[1] }
     /affine\.(for|parallel)/ { match($0, /affine\.(for|parallel)/);
                                printf "  %-22s %s\n", name, substr($0, RSTART, RLENGTH) }' \
    "$OUT"/deps_parallel_reductions.mlir

# 2. The matmul: i and j become affine.parallel, the k reduction stays an
#    affine.for (it accumulates through memory into C[i, j]).
mlir-opt "$SCRIPT_DIR"/matmul.mlir --affine-parallelize \
    -o "$OUT"/matmul_parallel.mlir
#    max-nested=1 parallelizes only the outermost parallel loop.
mlir-opt "$SCRIPT_DIR"/matmul.mlir --affine-parallelize=max-nested=1 \
    -o "$OUT"/matmul_parallel_outer.mlir

# 3. -lower-affine keeps the parallelism: affine.parallel -> scf.parallel.
mlir-opt "$OUT"/matmul_parallel.mlir --lower-affine --canonicalize \
    -o "$OUT"/matmul_scf_parallel.mlir
mlir-opt "$OUT"/deps_parallel_reductions.mlir --lower-affine --canonicalize \
    -o "$OUT"/deps_scf_parallel.mlir

# 4. ...but the default CPU route does not: -convert-scf-to-cf turns
#    scf.parallel into ordinary sequential branches.
mlir-opt "$OUT"/matmul_scf_parallel.mlir --convert-scf-to-cf \
    -o "$OUT"/matmul_serialized.mlir

echo "=== parallel ops per stage (matmul) ==="
for f in matmul_parallel matmul_parallel_outer matmul_scf_parallel matmul_serialized; do
    printf "  %-24s affine.parallel=%s scf.parallel=%s cf.cond_br=%s\n" "$f" \
        "$(grep -c 'affine.parallel' "$OUT/$f.mlir" || true)" \
        "$(grep -c 'scf.parallel' "$OUT/$f.mlir" || true)" \
        "$(grep -c 'cf.cond_br' "$OUT/$f.mlir" || true)"
done
# 5. Preview of Chapter 31: the SAME scf.parallel handed to the GPU passes
#    becomes a gpu.launch (one block per (i, j) here; nothing is run).
mlir-opt "$OUT"/matmul_scf_parallel.mlir \
    --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --canonicalize \
    -o "$OUT"/matmul_gpu_preview.mlir
echo "=== GPU preview ==="
grep -o 'gpu.launch blocks.*threads' "$OUT"/matmul_gpu_preview.mlir

# 6. Chapter 6's conv2d: the two output loops become affine.parallel, the
#    filter loops stay sequential affine.for with iter_args (a reduction
#    through registers, not memory).
CONV="$SCRIPT_DIR"/../../../ch06-affine-maps-and-loop-transforms/examples/05-conv2d/conv2d.mlir
mlir-opt "$CONV" --affine-parallelize -o "$OUT"/conv2d_parallel.mlir
mlir-opt "$OUT"/conv2d_parallel.mlir --lower-affine --canonicalize \
    -o "$OUT"/conv2d_scf_parallel.mlir
echo "=== conv2d ==="
printf "  affine.parallel=%s affine.for(iter_args)=%s scf.parallel=%s\n" \
    "$(grep -c 'affine.parallel' "$OUT"/conv2d_parallel.mlir || true)" \
    "$(grep -c 'affine.for .*iter_args' "$OUT"/conv2d_parallel.mlir || true)" \
    "$(grep -c 'scf.parallel' "$OUT"/conv2d_scf_parallel.mlir || true)"
echo "outputs in $OUT"
