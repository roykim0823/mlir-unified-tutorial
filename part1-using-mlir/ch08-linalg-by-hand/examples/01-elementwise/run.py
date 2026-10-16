"""Call the named linalg.add kernel (add.mlir) and check c == a + b.

Usage: run.py <path to libadd.so>   (build.sh passes it)
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
    addv = lib._mlir_ciface_addv
    addv.argtypes = [ctypes.POINTER(memref_descriptor_type(1))] * 3
    addv.restype = None

    a = np.arange(10, dtype=np.float32)
    b = np.full(10, 10, dtype=np.float32)
    c = np.zeros(10, dtype=np.float32)

    # The descriptors alias NumPy's buffers, so the kernel's writes land in c.
    addv(ctypes.byref(numpy_to_memref(a)),
         ctypes.byref(numpy_to_memref(b)),
         ctypes.byref(numpy_to_memref(c)))

    np.testing.assert_array_equal(c, a + b)
    print("linalg.add matches NumPy a + b:")
    print(c)


if __name__ == "__main__":
    main()
