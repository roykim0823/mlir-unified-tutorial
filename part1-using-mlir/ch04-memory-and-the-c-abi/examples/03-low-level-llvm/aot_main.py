"""Call the vectorized array_add (8 floats per iteration) from Python."""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402

MemRef1D = memref_descriptor_type(1)
SO = "dylib" if sys.platform == "darwin" else "so"


def main():
    lib = ctypes.CDLL(os.path.join(HERE, "build", f"libarray_add_vec.{SO}"))
    array_add = lib._mlir_ciface_array_add
    array_add.argtypes = [ctypes.POINTER(MemRef1D)] * 3   # a, b, c
    array_add.restype = None

    size = 1024
    a = np.arange(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2
    c = np.zeros(size, dtype=np.float32)
    a_desc, b_desc, c_desc = (numpy_to_memref(x) for x in (a, b, c))
    array_add(ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(c_desc))

    np.testing.assert_array_almost_equal(c, a + b)
    print("Vectorized array addition successful!")
    print(f"First few elements: {c[:5]}")
    print(f"Last element: {c[-1]}")


if __name__ == "__main__":
    main()
