"""Call the named linalg.reduce kernel (reduce.mlir) and check it against
NumPy's a.sum(axis=1).

Usage: run.py <path to libreduce.so>   (build.sh passes it)
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
    row_sum = lib._mlir_ciface_row_sum
    row_sum.argtypes = [ctypes.POINTER(memref_descriptor_type(2)),
                        ctypes.POINTER(memref_descriptor_type(1))]
    row_sum.restype = None

    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    # linalg.reduce accumulates into its outs buffer, so it must start at 0.
    out = np.zeros(3, dtype=np.float32)

    row_sum(ctypes.byref(numpy_to_memref(matrix)),
            ctypes.byref(numpy_to_memref(out)))

    np.testing.assert_array_equal(out, matrix.sum(axis=1))
    print("linalg.reduce matches NumPy matrix.sum(axis=1):")
    print(out)


if __name__ == "__main__":
    main()
