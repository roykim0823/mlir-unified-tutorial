"""Call the fused addmul kernel and check it against NumPy's (a + b) * c.

The fused function returned a tensor; build.sh bufferized it and ran
--buffer-results-to-out-params, so the C interface is the in-place shape
addmul(a, b, c, out).

Usage: run.py <path to libfused.so>   (build.sh passes it)
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
    addmul = lib._mlir_ciface_addmul
    addmul.argtypes = [ctypes.POINTER(memref_descriptor_type(1))] * 4  # a, b, c, out
    addmul.restype = None

    a = np.arange(10, dtype=np.float32)
    b = np.full(10, 2, dtype=np.float32)
    c = np.arange(10, dtype=np.float32) + 1
    out = np.zeros(10, dtype=np.float32)

    addmul(ctypes.byref(numpy_to_memref(a)),
           ctypes.byref(numpy_to_memref(b)),
           ctypes.byref(numpy_to_memref(c)),
           ctypes.byref(numpy_to_memref(out)))

    np.testing.assert_allclose(out, (a + b) * c)
    print("fused kernel matches NumPy (a + b) * c:")
    print("  a   =", a)
    print("  b   =", b)
    print("  c   =", c)
    print("  out =", out)


if __name__ == "__main__":
    main()
