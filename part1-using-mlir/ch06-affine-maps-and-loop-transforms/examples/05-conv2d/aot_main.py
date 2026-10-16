"""Run the affine 2-D convolution (plain and optimized) and check both against
a NumPy reference (cross-correlation, as in CNN frameworks).

The output shape is (in_h - k_h + 1, in_w - k_w + 1); the caller allocates it.
The sizes are deliberately awkward: the 9x8 output is not a multiple of the 4x4
tile, and the 3-wide filter is not a multiple of the unroll factor 2, so the
partial tiles and the unroll cleanup loop are exercised too.

    python aot_main.py <build dir>
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import memref_descriptor_type, numpy_to_memref  # noqa: E402


def reference_conv2d(inp, flt):
    """NumPy cross-correlation, matching the MLIR kernel exactly."""
    kh, kw = flt.shape
    oh, ow = inp.shape[0] - kh + 1, inp.shape[1] - kw + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[i, j] = (inp[i:i + kh, j:j + kw] * flt).sum()
    return out


def run(build_dir, name, inp, flt):
    ext = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(os.path.join(build_dir, f"lib{name}.{ext}"))
    conv = lib._mlir_ciface_conv_2d
    conv.argtypes = [ctypes.POINTER(memref_descriptor_type(2))] * 3
    conv.restype = None
    oh, ow = inp.shape[0] - flt.shape[0] + 1, inp.shape[1] - flt.shape[1] + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    conv(ctypes.byref(numpy_to_memref(inp)),
         ctypes.byref(numpy_to_memref(flt)),
         ctypes.byref(numpy_to_memref(out)))
    return out


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")
    rng = np.random.default_rng(0)
    inp = rng.random((11, 10), dtype=np.float32)
    flt = np.arange(9, dtype=np.float32).reshape(3, 3)  # a simple 3x3 kernel
    expected = reference_conv2d(inp, flt)

    for name in ("conv2d", "conv2d_opt"):
        out = run(build_dir, name, inp, flt)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
        print(f"{name:<10} matches NumPy (output {out.shape[0]}x{out.shape[1]}, "
              f"max abs error {np.abs(out - expected).max():.2e})")


if __name__ == "__main__":
    main()
