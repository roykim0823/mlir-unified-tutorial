"""Compare the numbers in two output files within a tolerance.

Usage:
  ./cnn > cnn_c_out.txt
  python run_cnn_model.py > cnn_py_out.txt    # (in the torch environment)
  python ../compare_outputs.py cnn_c_out.txt cnn_py_out.txt [atol]

Every number in each file is extracted in order (labels, brackets, "tensor(",
"grad_fn=..." and other text are ignored), so a compiled binary's plain
listing can be held against PyTorch's printed tensor. Exits 0 on a match,
1 otherwise. Default absolute tolerance: 1e-4 — float32 results from
different kernels (BLAS vs ATen, different summation orders) agree to a few
ulps, not bit-for-bit.

Needs only numpy. Shapes in the PyTorch printout ("torch.Size([4, 10])") are
numbers too, so print only the values you want compared — the run_*.py
scripts here label shapes on separate lines you can drop, e.g. with
`grep -v shape`.
"""

import re
import sys

import numpy as np

# A number not glued to a word: skips the 0 in "Backward0", the 32 in
# "float32", the 1 in "arg1".
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")


def numbers(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array([float(x) for x in NUMBER.findall(f.read())])


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    a, b = numbers(sys.argv[1]), numbers(sys.argv[2])
    atol = float(sys.argv[3]) if len(sys.argv) == 4 else 1e-4

    if a.shape != b.shape:
        print(f"Outputs differ: {a.size} vs {b.size} numbers")
        return 1
    if np.allclose(a, b, atol=atol):
        print(f"Outputs match ({a.size} values, atol={atol:g})")
        return 0
    diff = np.abs(a - b)
    i = int(diff.argmax())
    print(f"Outputs differ: max |diff| = {diff[i]:.3g} at value #{i} "
          f"({a[i]} vs {b[i]}); {int((diff > atol).sum())} of {a.size} "
          f"values outside atol={atol:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
