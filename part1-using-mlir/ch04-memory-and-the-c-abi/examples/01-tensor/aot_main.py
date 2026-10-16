"""Call the three compiled identity-matrix kernels from Python (AOT).

build.sh compiles each .mlir file into a shared library exporting a
C-callable `_mlir_ciface_identity` (generated because of the
`llvm.emit_c_interface` attribute). All three compute the same m x n identity
matrix; they differ only in the CALLING CONVENTION, i.e. in how the result
memref crosses the C boundary:

  identity_return        "callee allocates, caller frees"
      void _mlir_ciface_identity(MemRef2D *result, int64_t m, int64_t n);
      The kernel malloc()s the buffer and writes its DESCRIPTOR through the
      `result` out-pointer. We copy the data out, then free(allocated).

  identity_fill          "destination-passing" (with a hidden temporary)
  identity_fill_inplace  "destination-passing" (truly in place)
      void _mlir_ciface_identity(MemRef2D *out);
      WE allocate the output (np.zeros) and pass a descriptor pointing at
      NumPy's buffer; the kernel writes into it. Nothing to free.

The memref descriptor <-> NumPy glue is tools/np_memref.py (Chapter 4).
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, memref_to_numpy, numpy_to_memref  # noqa: E402

MemRef2D = memref_descriptor_type(2)
SO = "dylib" if sys.platform == "darwin" else "so"

# The callee-allocates kernel malloc()s inside the library; we must give the
# buffer back to the same C allocator. ctypes.CDLL(None) is "this process",
# whose symbols include libc's free.
_libc = ctypes.CDLL(None)
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def load(name):
    return ctypes.CDLL(os.path.join(HERE, "build", f"lib{name}.{SO}"))


def run_fill(lib, m, n):
    """Destination-passing: we allocate, the kernel fills in place."""
    fn = lib._mlir_ciface_identity
    fn.argtypes = [ctypes.POINTER(MemRef2D)]   # one in/out descriptor
    fn.restype = None
    out = np.zeros((m, n), dtype=np.int32)     # caller-owned buffer
    fn(ctypes.byref(numpy_to_memref(out)))     # kernel writes into `out`
    return out                                 # nothing to free


def run_return(lib, m, n):
    """Callee allocates and returns the buffer; we copy it out and free it."""
    fn = lib._mlir_ciface_identity
    # Result descriptor out-pointer first, then the two index args (i64).
    fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.c_int64, ctypes.c_int64]
    fn.restype = None
    result = MemRef2D()                        # zero-initialized; callee fills it
    fn(ctypes.byref(result), m, n)
    print(f"  returned descriptor: sizes={list(result.sizes)} "
          f"strides={list(result.strides)} offset={result.offset} "
          f"aligned-allocated={result.aligned - result.allocated} bytes")
    arr = memref_to_numpy(result, np.int32).copy()   # copy BEFORE freeing
    _libc.free(result.allocated)                     # free the base pointer
    return arr


def main():
    m, n = 5, 5
    expected = np.eye(m, n, dtype=np.int32)
    cases = [
        ("identity_fill         (caller's buffer, via a temporary)", "identity_fill", run_fill),
        ("identity_fill_inplace (caller's buffer, in place)", "identity_fill_inplace", run_fill),
        ("identity_return       (returns a tensor)", "identity_return", run_return),
    ]
    for label, name, run in cases:
        print(label)
        arr = run(load(name), m, n)
        np.testing.assert_array_equal(arr, expected)
        print(arr)
    # Non-square works too: the shape is only known at run time.
    np.testing.assert_array_equal(run_fill(load("identity_fill_inplace"), 2, 4),
                                  np.eye(2, 4, dtype=np.int32))
    print("all identity kernels agree with np.eye")


if __name__ == "__main__":
    main()
