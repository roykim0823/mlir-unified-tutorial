"""Driver for softmax.mlir — runs the MLIR row-wise softmax and checks it against
the NumPy softmax from the model (gpt2/model.py).

Usage: python aot_main.py [path/to/libsoftmax.so]   (build.sh passes the path)
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The shared NumPy <-> memref bridge (Chapter 4) lives in <repo>/tools; the
# NumPy GPT-2 package is one directory up.
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, ".."))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402
from gpt2 import softmax as np_softmax                         # noqa: E402

MemRef2D = memref_descriptor_type(2)


def main():
  lib_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build", "libsoftmax.so")
  lib = ctypes.CDLL(lib_path)
  fn = lib._mlir_ciface_softmax
  fn.argtypes = [ctypes.POINTER(MemRef2D)] * 2
  fn.restype = None

  # Stand-in for attention scores: an N x N matrix of scaled dot products.
  rng = np.random.default_rng(0)                 # fixed seed: reproducible output
  x = (rng.standard_normal((6, 6)) * 3.0).astype(np.float32)
  out = np.zeros_like(x)

  dx, dout = numpy_to_memref(x), numpy_to_memref(out)
  fn(ctypes.byref(dx), ctypes.byref(dout))

  expected = np_softmax(x)                       # the model's own softmax
  np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(out.sum(axis=1), np.ones(6), atol=1e-5)
  print("MLIR softmax successful! "
        f"(max abs error {np.abs(out - expected).max():.2e}; every row sums to 1)")


if __name__ == "__main__":
  main()
