"""Run the CNN in PyTorch on a batch of four all-ones images.

The printed log-probabilities are the reference the compiled C++ binary
(cnn_call.cpp) must reproduce.
"""

import os

# Limit to a single thread for reproducibility.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

from model import ConvolutionalNetwork

if __name__ == "__main__":
    torch.manual_seed(41)
    model = ConvolutionalNetwork()
    x = torch.ones((4, 1, 28, 28))
    out = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
    torch.set_printoptions(precision=5)
    print("Output:", out)
