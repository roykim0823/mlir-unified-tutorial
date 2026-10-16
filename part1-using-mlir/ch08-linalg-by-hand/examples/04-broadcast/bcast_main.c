// C driver for both broadcast kernels, used by test/part1/ch08_broadcast.mlir
// (lit cannot rely on NumPy). Adds [1 2 3 4] to every row of a 3x4 matrix of
// ones with each kernel and prints the result.
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef1D;
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef2D;

void _mlir_ciface_add_vec_to_mat(MemRef2D *matrix, MemRef1D *vector, MemRef2D *out);
void _mlir_ciface_add_vec_to_mat_generic(MemRef2D *matrix, MemRef1D *vector,
                                         MemRef2D *out);

static MemRef2D wrap2(float *data) {
  MemRef2D m = {data, data, 0, {3, 4}, {4, 1}};
  return m;
}

static void print(const char *name, const float *out) {
  printf("%s:", name);
  for (int i = 0; i < 3; ++i) {
    printf(" [");
    for (int j = 0; j < 4; ++j) printf("%s%g", j ? " " : "", out[i * 4 + j]);
    printf("]");
  }
  printf("\n");
}

int main(void) {
  float matrix[12], vector[4] = {1, 2, 3, 4}, out1[12] = {0}, out2[12] = {0};
  for (int i = 0; i < 12; ++i) matrix[i] = 1.0f;
  MemRef1D mv = {vector, vector, 0, {4}, {1}};
  MemRef2D mm = wrap2(matrix), mo1 = wrap2(out1), mo2 = wrap2(out2);

  _mlir_ciface_add_vec_to_mat(&mm, &mv, &mo1);
  _mlir_ciface_add_vec_to_mat_generic(&mm, &mv, &mo2);
  print("named  ", out1);
  print("generic", out2);
  return 0;
}
