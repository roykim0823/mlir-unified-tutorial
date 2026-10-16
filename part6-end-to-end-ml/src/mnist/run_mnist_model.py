"""Run the MNIST MLP in PyTorch with a tensor of ones.

The printed logits are the reference the compiled C++ binary (ch28) must
reproduce.
"""

import os

# Limit to a single thread for reproducibility.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

from model import MnistNetwork

if __name__ == "__main__":
    torch.manual_seed(41)
    model = MnistNetwork()
    x = torch.ones((1, 1, 28, 28))
    print("Output:", model(x))
