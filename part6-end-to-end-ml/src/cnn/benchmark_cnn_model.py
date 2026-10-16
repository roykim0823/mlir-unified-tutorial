"""Benchmark the CNN's inference time in PyTorch (CPU, single thread).

10 warm-up calls, then the average over 100 timed calls — the same protocol
as cnn_call_benchmark.cpp, so the two numbers are comparable.
"""

import os

# Must be set before torch is imported to take effect.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

import torch

from model import ConvolutionalNetwork

torch.set_num_threads(1)


def benchmark(warmup: int = 10, runs: int = 100) -> None:
    torch.manual_seed(41)
    model = ConvolutionalNetwork().eval()
    x = torch.ones((4, 1, 28, 28))
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        end = time.perf_counter()
    print(f"Avg inference time: {(end - start) / runs:.9f} sec")


if __name__ == "__main__":
    benchmark()
