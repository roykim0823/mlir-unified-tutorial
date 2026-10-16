"""Call add_vector_to_matrix three ways from Python, through tools/np_memref.py.

  1. _mlir_ciface_add_vector_to_matrix       (emit_c_interface; returns a
     memref, so the wrapper's FIRST argument is a result-descriptor pointer)
  2. add_vector_to_matrix_into               (no wrapper: every descriptor is
     expanded into its 3 + 2*rank scalar fields)
  3. _mlir_ciface_add_vector_to_matrix_into  (wrapper added to every function
     by --llvm-request-c-wrappers, in the second library)
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import (expand, memref_descriptor_type,  # noqa: E402
                       memref_to_numpy, numpy_to_memref)

MemRef1D = memref_descriptor_type(1)
MemRef2D = memref_descriptor_type(2)
SO = "dylib" if sys.platform == "darwin" else "so"

_libc = ctypes.CDLL(None)
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def show(desc, label):
    print(f"  {label}: offset={desc.offset} sizes={list(desc.sizes)} "
          f"strides={list(desc.strides)}")


def main():
    lib = ctypes.CDLL(os.path.join(HERE, "build", f"libadd_vector_to_matrix.{SO}"))
    lib_all = ctypes.CDLL(os.path.join(HERE, "build", f"libadd_vector_to_matrix_all.{SO}"))

    M = np.arange(6, dtype=np.float32).reshape(2, 3)   # [[0 1 2] [3 4 5]]
    v = np.array([10, 20, 30], dtype=np.float32)
    expected = M + v                                    # NumPy's broadcast
    m_desc, v_desc = numpy_to_memref(M), numpy_to_memref(v)
    show(m_desc, "M descriptor")
    show(v_desc, "v descriptor")

    # 1. Callee allocates. void _mlir_ciface_add_vector_to_matrix(
    #        MemRef2D *result, MemRef2D *M, MemRef1D *v);
    fn = lib._mlir_ciface_add_vector_to_matrix
    fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef2D),
                   ctypes.POINTER(MemRef1D)]
    fn.restype = None
    result = MemRef2D()
    fn(ctypes.byref(result), ctypes.byref(m_desc), ctypes.byref(v_desc))
    show(result, "result descriptor")
    out1 = memref_to_numpy(result, np.float32).copy()  # copy, then free
    _libc.free(result.allocated)
    np.testing.assert_allclose(out1, expected)
    print("1. _mlir_ciface_add_vector_to_matrix (returns a memref):")
    print(out1)

    # 2. No C wrapper: the raw function takes each memref EXPANDED into its
    #    fields -- allocated, aligned, offset, sizes[rank], strides[rank] --
    #    as separate scalar arguments (2 pointers + 1 + 2*rank integers).
    def field_types(rank):
        return [ctypes.c_void_p] * 2 + [ctypes.c_int64] * (1 + 2 * rank)

    out2 = np.zeros_like(M)
    raw = lib.add_vector_to_matrix_into
    raw.argtypes = field_types(2) + field_types(1) + field_types(2)
    raw.restype = None
    args = expand(m_desc) + expand(v_desc) + expand(numpy_to_memref(out2))
    print(f"2. add_vector_to_matrix_into, {len(args)} scalar arguments:")
    raw(*args)
    np.testing.assert_allclose(out2, expected)
    print(out2)

    # 3. Same function, but from the library built with
    #    --llvm-request-c-wrappers: now it has a wrapper too.
    out3 = np.zeros_like(M)
    fn = lib_all._mlir_ciface_add_vector_to_matrix_into
    fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef1D),
                   ctypes.POINTER(MemRef2D)]
    fn.restype = None
    fn(ctypes.byref(m_desc), ctypes.byref(v_desc), ctypes.byref(numpy_to_memref(out3)))
    np.testing.assert_allclose(out3, expected)
    print("3. _mlir_ciface_add_vector_to_matrix_into (--llvm-request-c-wrappers):")
    print(out3)
    print("all three match NumPy's M + v")


if __name__ == "__main__":
    main()
