"""Load the compiled original and optimized capstone functions and confirm they
compute the same values — i.e. the e-graph rewrite preserved meaning.

Usage: python3 run_compiled.py [BUILD_DIR]   (default: build/ next to this file)
build.sh compiles build/orig.mlir and build/opt.mlir to liborig.so / libopt.so.
"""

import ctypes
import os
import sys


def load(build, name):
  lib = ctypes.CDLL(os.path.join(build, f"lib{name}.so"))
  fn = lib.f                      # scalar f32 -> f32; no memref descriptor needed
  fn.argtypes = [ctypes.c_float]
  fn.restype = ctypes.c_float
  return fn


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  build = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "build")
  orig = load(build, "orig")
  opt = load(build, "opt")
  for x in (-3.0, 0.0, 2.5, 7.0, 100.0):
    a, b = orig(x), opt(x)
    print(f"  f({x:7.2f}):  original = {a:10.4f}   optimized = {b:10.4f}")
    assert abs(a - b) < 1e-4, f"mismatch at {x}: {a} vs {b}"
    assert abs(b - x * x) < 1e-4, f"optimized wrong at {x}: {b}"
  print("Capstone OK: optimized f(a) == original f(a) == a*a for all test inputs.")


if __name__ == "__main__":
  main()
