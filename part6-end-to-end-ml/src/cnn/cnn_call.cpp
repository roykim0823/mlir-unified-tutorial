// Call the compiled CNN from C++ through the _mlir_ciface_ wrapper.
// Same structure as mnist_call.cpp; the input is a batch of four 1x28x28
// images (NCHW) and the output is 4x10 log-probabilities.
//
// Build: sh run_mlir_pipeline.sh (lowers cnn_model_linalg.mlir and links
// this file into ./cnn) — see Chapter 30.
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

int main() {
  const int batch_size = 4;
  float inputData[batch_size][1][28][28];

  for (int k = 0; k < batch_size; k++)
    for (int i = 0; i < 28; i++)
      for (int j = 0; j < 28; j++)
        inputData[k][0][i][j] = 1.0;

  MemRefDescriptor<float, 4> inputMemRef = {
      (float *)inputData, (float *)inputData, /*offset=*/0,
      /*sizes=*/{batch_size, 1, 28, 28},
      /*strides=*/{784, 784, 28, 1}};  // row-major layout

  // Out-slot: the model allocates the result and writes its descriptor
  // here (Chapter 28, section 3).
  MemRefDescriptor<float, 2> outputMemRef = {};

  _mlir_ciface_cnn_model(&outputMemRef, &inputMemRef);

  float *output = outputMemRef.aligned + outputMemRef.offset;
  for (int64_t i = 0; i < outputMemRef.sizes[0]; ++i) {
    for (int64_t j = 0; j < outputMemRef.sizes[1]; ++j)
      std::cout << std::fixed << std::setprecision(5)
                << output[i * outputMemRef.strides[0] +
                          j * outputMemRef.strides[1]]
                << ' ';
    std::cout << "\n";
  }
  free(outputMemRef.allocated);
  return 0;
}
