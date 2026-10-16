"""Run both broadcast kernels and check them against NumPy's matrix + vector.

Usage: run.py <libbcast.so> <libbcast_generic.so>   (build.sh passes them)
"""

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402


def run(lib_path, symbol):
    fn = getattr(ctypes.CDLL(lib_path), symbol)
    fn.argtypes = [ctypes.POINTER(memref_descriptor_type(2)),
                   ctypes.POINTER(memref_descriptor_type(1)),
                   ctypes.POINTER(memref_descriptor_type(2))]
    fn.restype = None

    matrix = np.ones((3, 4), dtype=np.float32)
    vector = np.array([1, 2, 3, 4], dtype=np.float32)
    out = np.zeros((3, 4), dtype=np.float32)

    fn(ctypes.byref(numpy_to_memref(matrix)),
       ctypes.byref(numpy_to_memref(vector)),
       ctypes.byref(numpy_to_memref(out)))

    np.testing.assert_array_equal(out, matrix + vector)  # NumPy broadcasting
    return out


def main():
    out = run(sys.argv[1], "_mlir_ciface_add_vec_to_mat")
    print("linalg.broadcast + linalg.add matches NumPy matrix + vector:")
    print(out)
    run(sys.argv[2], "_mlir_ciface_add_vec_to_mat_generic")
    print("single linalg.generic with a (d0, d1) -> (d1) map: same result")


if __name__ == "__main__":
    main()
