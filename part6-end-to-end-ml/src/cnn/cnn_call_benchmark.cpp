// Benchmark the compiled cnn model from C++ (CPU, MLIR-compiled).
//
// Same descriptors as cnn_call.cpp; times the _mlir_ciface_ call: 10 warm-up
// calls, then the average over 100 timed calls (the protocol of
// benchmark_cnn_model.py). Build and run with: sh benchmark.sh
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_cnn_model(MemRefDescriptor<float, 2> *output,
                     MemRefDescriptor<float, 4> *input);
}

static float inputData[4 * 1 * 28 * 28];

int main() {
  for (float &v : inputData)
    v = 1.0f;

  MemRefDescriptor<float, 4> inputMemRef = {
      inputData, inputData, /*offset=*/0, /*sizes=*/{4, 1, 28, 28}, /*strides=*/{784, 784, 28, 1}};

  // The model allocates its own result buffer and overwrites this out-slot
  // with its descriptor (Chapter 28, section 3), so no output storage is
  // needed here — but each call's result must be freed, or a 100-call loop
  // leaks 100 buffers.
  MemRefDescriptor<float, 2> outputMemRef = {};

  auto call = [&] {
    _mlir_ciface_cnn_model(&outputMemRef, &inputMemRef);
    free(outputMemRef.allocated);
  };

  const int warmup = 10, runs = 100;
  for (int i = 0; i < warmup; ++i)
    call();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i)
    call();
  auto end = std::chrono::steady_clock::now();

  double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << "Avg inference time: " << std::fixed << std::setprecision(9)
            << (seconds / runs) << " sec\n";
  return 0;
}
