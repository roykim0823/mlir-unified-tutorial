// Call the compiled MNIST MLP from C++ through the _mlir_ciface_ wrapper.
// Same structure as sample_call.cpp; the input is now rank 4 (NCHW).
#include <cstdint>
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
void _mlir_ciface_mnist_model(MemRefDescriptor<float, 2> *output,
                              MemRefDescriptor<float, 4> *input);
}

int main(int argc, char *argv[]) {
  float inputData[1][1][28][28];
  float outputData[1][10];

  for (int i = 0; i < 28; i++)
    for (int j = 0; j < 28; j++)
      inputData[0][0][i][j] = 1.0;

  MemRefDescriptor<float, 4> inputMemRef = {
      (float *)inputData, (float *)inputData, /*offset=*/0,
      /*sizes=*/{1, 1, 28, 28},
      /*strides=*/{784, 784, 28, 1}};  // row-major layout

  MemRefDescriptor<float, 2> outputMemRef = {
      (float *)outputData, (float *)outputData, /*offset=*/0,
      /*sizes=*/{1, 10},
      /*strides=*/{10, 1}};

  _mlir_ciface_mnist_model(&outputMemRef, &inputMemRef);

  float *output = (float *)outputMemRef.aligned;
  for (int64_t i = 0; i < outputMemRef.sizes[0]; ++i) {
    for (int64_t j = 0; j < outputMemRef.sizes[1]; ++j)
      std::cout << std::fixed << std::setprecision(4)
                << output[i * outputMemRef.strides[0] + j] << ' ';
    std::cout << "\n";
  }
  return 0;
}
