"""Ahead-of-time: load the prebuilt library and call array_add on NumPy arrays."""

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
    lib = ctypes.CDLL(os.path.join(HERE, "build", f"libarray_add.{SO}"))

    # The C-interface wrapper takes one descriptor pointer per memref:
    #   void _mlir_ciface_array_add(MemRef1D *a, MemRef1D *b, MemRef1D *c);
    array_add = lib._mlir_ciface_array_add
    array_add.argtypes = [ctypes.POINTER(MemRef1D)] * 3
    array_add.restype = None

    size = 1024
    a = np.ones(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2
    c = np.zeros(size, dtype=np.float32)

    # Descriptors point straight at NumPy's buffers (no copy).
    a_desc, b_desc, c_desc = (numpy_to_memref(x) for x in (a, b, c))
    array_add(ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(c_desc))

    np.testing.assert_array_almost_equal(c, a + b)
    print("Array addition successful!")
    print(f"First few elements: {c[:5]}")


if __name__ == "__main__":
    main()
