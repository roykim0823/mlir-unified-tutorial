"""Run the sample model in PyTorch with a tensor of ones.

The printed output is the reference the compiled C++ binary (ch28) must
reproduce.
"""

import os

# Limit to a single thread for reproducibility.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

from model import Sample

if __name__ == "__main__":
    torch.manual_seed(41)
    model = Sample()
    x = torch.ones((3, 4))
    print("Input:", x)
    print("Output:", model(x))
