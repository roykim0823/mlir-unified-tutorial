"""Prove the hand-written interchange, skewing and wavefront rewrites are
correct: each transformed kernel must produce exactly the same array as its
original. build.sh compiles each .mlir file into its own shared library.

    python aot_main.py <build dir>
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402

PTR = ctypes.POINTER(memref_descriptor_type(2))


def load(build_dir, name, funcs):
    """Open lib<name> and declare `_mlir_ciface_<f>(MemRef2D*, ...)` for each f."""
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(os.path.join(build_dir, f"lib{name}.{ext}"))
    for fname, nargs in funcs:
        fn = getattr(lib, "_mlir_ciface_" + fname)
        fn.argtypes = [PTR] * nargs
        fn.restype = None
    return lib


def call(fn, *arrays):
    descs = [numpy_to_memref(a) for a in arrays]  # keep alive during the call
    fn(*[ctypes.byref(d) for d in descs])


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")

    # --- interchange: both loop orders copy B = A ---------------------------
    ic = load(build_dir, "interchange_manual", [("copy_ij", 2), ("copy_ji", 2)])
    A = np.arange(20, dtype=np.float32).reshape(4, 5)
    B_ij = np.zeros_like(A)
    B_ji = np.zeros_like(A)
    call(ic._mlir_ciface_copy_ij, A, B_ij)
    call(ic._mlir_ciface_copy_ji, A, B_ji)
    assert np.array_equal(B_ij, B_ji) and np.array_equal(B_ij, A)
    print("interchange:  copy_ij == copy_ji == A")

    # --- skewing + wavefront: same stencil, three iteration orders ----------
    sk = load(build_dir, "skewing_manual",
              [("stencil", 1), ("stencil_skewed", 1), ("stencil_wavefront", 1)])
    S = np.ones((8, 8), dtype=np.float32)
    S_skew = S.copy()
    S_wave = S.copy()
    call(sk._mlir_ciface_stencil, S)
    call(sk._mlir_ciface_stencil_skewed, S_skew)
    call(sk._mlir_ciface_stencil_wavefront, S_wave)

    # Plain-Python reference for A[i,j] = A[i-1,j] + A[i,j-1].
    ref = np.ones((8, 8), dtype=np.float32)
    for i in range(1, 8):
        for j in range(1, 8):
            ref[i, j] = ref[i - 1, j] + ref[i, j - 1]
    assert np.array_equal(S, ref)
    assert np.array_equal(S, S_skew)
    assert np.array_equal(S, S_wave)
    print("stencil:      matches the NumPy reference (A[7,7] = %g)" % S[7, 7])
    print("skewing:      stencil == stencil_skewed")
    print("wavefront:    stencil == stencil_wavefront (parallel inner loop)")
    print("Manual transforms preserve results.")


if __name__ == "__main__":
    main()
