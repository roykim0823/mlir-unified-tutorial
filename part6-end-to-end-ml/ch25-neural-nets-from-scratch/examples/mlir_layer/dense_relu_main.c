// C driver for dense_relu.mlir, used by test/part6/ch25_dense_relu.mlir.
// Same check as aot_main.py, without NumPy: fixed inputs, a plain-C reference
// relu(X @ W + b), and the compiled kernel's output printed for FileCheck.
#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct { float *allocated, *aligned; int64_t offset, sizes[1], strides[1]; } MemRef1D;
typedef struct { float *allocated, *aligned; int64_t offset, sizes[2], strides[2]; } MemRef2D;

void _mlir_ciface_dense_relu(MemRef2D *X, MemRef2D *W, MemRef1D *b, MemRef2D *out);

enum { N = 2, K = 3, M = 4 };

int main(void) {
  float X[N][K] = {{1, 2, 3}, {-1, 0.5f, 2}};
  float W[K][M] = {{0.5f, -1, 0, 2}, {1, 0.25f, -0.5f, 0}, {-0.5f, 1, 1, -1}};
  float b[M] = {0.1f, -0.2f, 0.3f, -4};
  float out[N][M] = {{0}};  // linalg.matmul accumulates: zero it first

  MemRef2D dX = {&X[0][0], &X[0][0], 0, {N, K}, {K, 1}};
  MemRef2D dW = {&W[0][0], &W[0][0], 0, {K, M}, {M, 1}};
  MemRef1D db = {b, b, 0, {M}, {1}};
  MemRef2D dout = {&out[0][0], &out[0][0], 0, {N, M}, {M, 1}};
  _mlir_ciface_dense_relu(&dX, &dW, &db, &dout);

  float max_err = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      float z = b[j];
      for (int k = 0; k < K; ++k) z += X[i][k] * W[k][j];
      float ref = z > 0 ? z : 0;
      max_err = fmaxf(max_err, fabsf(out[i][j] - ref));
      printf("%.4f%s", out[i][j], j + 1 < M ? " " : "\n");
    }
  }
  printf("max abs error vs C reference: %s\n", max_err < 1e-5f ? "ok" : "FAIL");
  return max_err < 1e-5f ? 0 : 1;
}
