"""Call the linalg.matmul kernel (matmul.mlir) and check it against NumPy's @.

The kernel accumulates into C, so C is passed in zeroed.

Usage: run.py <path to libmatmul.so>   (build.sh passes it)
"""

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402


def main():
    lib = ctypes.CDLL(sys.argv[1])
    matmul = lib._mlir_ciface_matmul
    matmul.argtypes = [ctypes.POINTER(memref_descriptor_type(2))] * 3
    matmul.restype = None

    rng = np.random.default_rng(0)
    A = rng.random((8, 10), dtype=np.float32)
    B = rng.random((10, 16), dtype=np.float32)
    C = np.zeros((8, 16), dtype=np.float32)

    matmul(ctypes.byref(numpy_to_memref(A)),
           ctypes.byref(numpy_to_memref(B)),
           ctypes.byref(numpy_to_memref(C)))

    expected = A @ B
    np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-5)
    print("linalg.matmul matches NumPy A @ B "
          f"(8x10 @ 10x16, max abs error {np.abs(C - expected).max():.2e})")


if __name__ == "__main__":
    main()
