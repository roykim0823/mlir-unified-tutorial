"""Call the compiled affine matmul from Python and check it against NumPy.

The kernel computes C += A @ B, so we pass a zero-initialized C. Each NumPy
array is wrapped in a rank-2 memref descriptor by tools/np_memref.py (Chapter 4
explains the layout) and passed by pointer to the `_mlir_ciface_matmul`
wrapper that `llvm.emit_c_interface` generated.

    python aot_main.py <build dir>
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402


def load_lib(build_dir, name):
    ext = "dylib" if sys.platform == "darwin" else "so"
    return ctypes.CDLL(os.path.join(build_dir, f"lib{name}.{ext}"))


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")
    lib = load_lib(build_dir, "matmul")
    matmul = lib._mlir_ciface_matmul
    matmul.argtypes = [ctypes.POINTER(memref_descriptor_type(2))] * 3
    matmul.restype = None

    rng = np.random.default_rng(0)
    M, K, N = 4, 5, 3
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)  # the kernel accumulates into C

    matmul(ctypes.byref(numpy_to_memref(A)),
           ctypes.byref(numpy_to_memref(B)),
           ctypes.byref(numpy_to_memref(C)))

    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-5)
    print(f"Affine matmul successful! (max abs error {np.abs(C - expected).max():.2e})")
    print(C)


if __name__ == "__main__":
    main()
