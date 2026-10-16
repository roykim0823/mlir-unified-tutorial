// Call the compiled ResNet-18 from C++ through the _mlir_ciface_ wrapper.
//
// Expects the one-argument function from lower_resnet18_model.py (buffers
// frozen). Input: one all-ones 3x224x224 image (NCHW); output: 1x1000
// ImageNet logits. Prints the first ten and the arg-max class, which
// run_resnet18_model.py prints too.
//
// Build: sh run_mlir_pipeline.sh  — see Chapter 30.
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
void _mlir_ciface_resnet18(MemRefDescriptor<float, 2> *output,
                           MemRefDescriptor<float, 4> *input);
}

// 600 KB: too big to keep on the stack comfortably.
static float inputData[1][3][224][224];

int main() {
  for (int k = 0; k < 3; k++)
    for (int i = 0; i < 224; i++)
      for (int j = 0; j < 224; j++)
        inputData[0][k][i][j] = 1.0;

  MemRefDescriptor<float, 4> inputMemRef = {
      (float *)inputData, (float *)inputData, /*offset=*/0,
      /*sizes=*/{1, 3, 224, 224},
      /*strides=*/{3 * 224 * 224, 224 * 224, 224, 1}};  // row-major layout

  MemRefDescriptor<float, 2> outputMemRef = {};  // out-slot, see Chapter 28

  _mlir_ciface_resnet18(&outputMemRef, &inputMemRef);

  float *logits = outputMemRef.aligned + outputMemRef.offset;
  int64_t n = outputMemRef.sizes[1], stride = outputMemRef.strides[1];

  std::cout << "Output shape: [" << outputMemRef.sizes[0] << ", " << n
            << "]\nFirst 10 logits: ";
  int64_t best = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i < 10)
      std::cout << std::fixed << std::setprecision(5) << logits[i * stride]
                << ' ';
    if (logits[i * stride] > logits[best * stride])
      best = i;
  }
  std::cout << "\nTop class index: " << best << "\n";
  free(outputMemRef.allocated);
  return 0;
}
