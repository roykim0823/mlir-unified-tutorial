"""Driver for dense_relu.mlir — runs one neural-network dense layer
(out = relu(X @ W + b)) and checks it against a NumPy reference.

Usage: python aot_main.py [path/to/libdense.so]   (build.sh passes the path)
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared NumPy <-> memref bridge (Chapter 4) lives in <repo>/tools.
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402

MemRef1D = memref_descriptor_type(1)
MemRef2D = memref_descriptor_type(2)


def main():
  lib_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build", "libdense.so")
  lib = ctypes.CDLL(lib_path)
  dense = lib._mlir_ciface_dense_relu
  dense.argtypes = [ctypes.POINTER(MemRef2D),   # X
                    ctypes.POINTER(MemRef2D),   # W
                    ctypes.POINTER(MemRef1D),   # b
                    ctypes.POINTER(MemRef2D)]   # out
  dense.restype = None

  rng = np.random.default_rng(0)                 # fixed seed: reproducible output
  N, K, M = 4, 3, 5                              # batch=4, in=3, out=5
  X = rng.standard_normal((N, K)).astype(np.float32)
  W = rng.standard_normal((K, M)).astype(np.float32)
  b = rng.standard_normal(M).astype(np.float32)
  out = np.zeros((N, M), dtype=np.float32)       # matmul accumulates -> zero it

  # Keep the descriptors alive in variables while the kernel runs.
  dX, dW, db, dout = (numpy_to_memref(a) for a in (X, W, b, out))
  dense(ctypes.byref(dX), ctypes.byref(dW), ctypes.byref(db), ctypes.byref(dout))

  expected = np.maximum(X @ W + b, 0.0)          # relu(X@W + b)
  np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
  print("MLIR dense layer  out = relu(X @ W + b)  successful! "
        f"(max abs error {np.abs(out - expected).max():.2e})")
  np.set_printoptions(precision=4, suppress=True)
  print(out)


if __name__ == "__main__":
  main()
