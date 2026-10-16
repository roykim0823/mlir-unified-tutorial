"""Benchmark ResNet-18's inference time in PyTorch (CPU, single thread).

10 warm-up calls, then the average over 100 timed calls. (The C++
benchmark uses 5 + 10 — the compiled model is far slower; see Chapter 30.)
"""

import os

# Must be set before torch is imported to take effect.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

import torch
from transformers import AutoModelForImageClassification

torch.set_num_threads(1)


def benchmark(warmup: int = 10, runs: int = 100) -> None:
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    model.eval()
    x = torch.ones((1, 3, 224, 224))
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        end = time.perf_counter()
    print(f"Avg inference time: {(end - start) / runs:.6f} sec")


if __name__ == "__main__":
    benchmark()
