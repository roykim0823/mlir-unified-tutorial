"""Benchmark flan-t5-small greedy generation in PyTorch (CPU).

Times a full model.generate() — encoder once, then one decoder step per
output token — averaged over 10 runs after 3 warm-ups. flan_call_benchmark.cpp
times the same greedy loop over the compiled model. (PyTorch's generate
uses a KV cache; the compiled model recomputes the decoder prefix every
step — part of the gap.)
"""

import time

from transformers import AutoTokenizer, T5ForConditionalGeneration

INPUT_TEXT = "translate English to German: How are you?"


def benchmark(warmup: int = 3, runs: int = 10) -> None:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.eval()
    input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids

    for _ in range(warmup):
        model.generate(input_ids)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model.generate(input_ids)
        times.append(time.perf_counter() - start)

    print(f"Generated {out.shape[1]} tokens: {out.tolist()}")
    print(f"Avg generation time: {sum(times) / len(times):.4f} s")
    print(f"Min: {min(times):.4f} s, Max: {max(times):.4f} s")


if __name__ == "__main__":
    benchmark()
