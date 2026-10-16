// Call the compiled sample model from C++ through the _mlir_ciface_ wrapper.
//
// Build (from src/sample/, after running the lowering pipeline — see ch28):
//   c++ -c sample_call.cpp -o sample_call.o
//   c++ sample_call.o sample_model.o -o sample -lopenblas   # or -framework Accelerate
#include <cstdint>
#include <iomanip>
#include <iostream>

// The C-compatible shape of a lowered MLIR memref: two data pointers
// (allocated for freeing, aligned for access), an element offset, and one
// size and stride per dimension. This must match what
// --finalize-memref-to-llvm produces for a rank-N memref.
template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
// Emitted by --llvm-request-c-wrappers: outputs first, then inputs, all
// passed by pointer to a descriptor.
void _mlir_ciface_sample_model(MemRefDescriptor<float, 2> *output,
                               MemRefDescriptor<float, 2> *input);
}

int main(int argc, char *argv[]) {
  float inputData[3][4];
  float outputData[3][5];

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      inputData[i][j] = 1.0;

  MemRefDescriptor<float, 2> inputMemRef = {
      (float *)inputData, (float *)inputData, /*offset=*/0,
      /*sizes=*/{3, 4},
      /*strides=*/{4, 1}};  // row-major layout

  MemRefDescriptor<float, 2> outputMemRef = {
      (float *)outputData, (float *)outputData, /*offset=*/0,
      /*sizes=*/{3, 5},
      /*strides=*/{5, 1}};

  _mlir_ciface_sample_model(&outputMemRef, &inputMemRef);

  float *output = (float *)outputMemRef.aligned;
  for (int64_t i = 0; i < outputMemRef.sizes[0]; ++i) {
    for (int64_t j = 0; j < outputMemRef.sizes[1]; ++j)
      std::cout << std::fixed << std::setprecision(4)
                << output[i * outputMemRef.strides[0] + j] << ' ';
    std::cout << "\n";
  }
  return 0;
}
