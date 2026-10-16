// C driver for softmax.mlir, used by test/part6/ch26_softmax.mlir.
// Same check as aot_main.py, without NumPy: compares the compiled kernel with
// a plain-C stable softmax and prints each row and its sum for FileCheck.
// Row 2 has huge scores: a naive exp(x) would overflow to inf there, the
// max-subtracted version must not.
#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct { float *allocated, *aligned; int64_t offset, sizes[2], strides[2]; } MemRef2D;

void _mlir_ciface_softmax(MemRef2D *x, MemRef2D *out);

enum { N = 3, M = 4 };

int main(void) {
  float x[N][M] = {{1, 2, 3, 4}, {0, 0, 0, 0}, {1000, 1001, 1002, 1003}};
  float out[N][M];
  MemRef2D dx = {&x[0][0], &x[0][0], 0, {N, M}, {M, 1}};
  MemRef2D dout = {&out[0][0], &out[0][0], 0, {N, M}, {M, 1}};
  _mlir_ciface_softmax(&dx, &dout);

  float max_err = 0;
  for (int i = 0; i < N; ++i) {
    float mx = x[i][0], sum = 0, row = 0;
    for (int j = 1; j < M; ++j) mx = fmaxf(mx, x[i][j]);
    for (int j = 0; j < M; ++j) sum += expf(x[i][j] - mx);
    for (int j = 0; j < M; ++j) {
      max_err = fmaxf(max_err, fabsf(out[i][j] - expf(x[i][j] - mx) / sum));
      row += out[i][j];
      printf("%.4f ", out[i][j]);
    }
    printf("| sum %.4f\n", row);
  }
  printf("max abs error vs C reference: %s\n", max_err < 1e-6f ? "ok" : "FAIL");
  return max_err < 1e-6f ? 0 : 1;
}
