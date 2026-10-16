// Benchmark greedy generation with the compiled flan-t5-small (CPU).
//
// Times one full generation for "translate English to German: How are
// you?" — the same work benchmark_flan_t5_model.py times with
// model.generate(): 3 warm-ups, then the average over 10 runs. Also
// reports the per-step (per model call) average.
//
// Build and run: sh benchmark.sh (Chapter 30). No Python needed.
#include <chrono>
#include <iomanip>
#include <iostream>

#include "flan_generate.h"

int main() {
  // AutoTokenizer("google/flan-t5-small")("translate English to German: How
  // are you?")
  const std::vector<int64_t> input_ids = {13959, 1566, 12, 2968, 10,
                                          571,   33,   25, 58,   1};
  const std::vector<int64_t> attention_mask(input_ids.size(), 1);

  const int warmup = 3, runs = 10;
  std::vector<int64_t> tokens;
  for (int i = 0; i < warmup; ++i)
    tokens = generate_tokens(input_ids, attention_mask, /*verbose=*/false);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i)
    tokens = generate_tokens(input_ids, attention_mask, /*verbose=*/false);
  auto end = std::chrono::steady_clock::now();

  double seconds = std::chrono::duration<double>(end - start).count() / runs;
  size_t steps = tokens.size() - 1; // one model call per generated token
  std::cout << "Generated " << tokens.size() << " tokens: [";
  for (size_t i = 0; i < tokens.size(); ++i)
    std::cout << (i ? ", " : "") << tokens[i];
  std::cout << "]\n"
            << "Avg generation time: " << std::fixed << std::setprecision(4)
            << seconds << " s (" << steps << " steps, "
            << seconds / steps << " s per model call)\n";
  return 0;
}
