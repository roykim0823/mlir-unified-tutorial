/* C driver for matmul.mlir, used by test/part1/ch06_affine_matmul_run.mlir.
 * Multiplies a 2x3 by a 3x2 matrix with integer-valued entries (so the result
 * is exact) and prints C. */
#include <stdio.h>

#include "../memref2d.h"

void _mlir_ciface_matmul(MemRef2D *A, MemRef2D *B, MemRef2D *C);

int main(void) {
  float a[2 * 3] = {1, 2, 3,
                    4, 5, 6};
  float b[3 * 2] = {7, 8,
                    9, 10,
                    11, 12};
  float c[2 * 2] = {0};
  MemRef2D A = memref2d(a, 2, 3), B = memref2d(b, 3, 2), C = memref2d(c, 2, 2);
  _mlir_ciface_matmul(&A, &B, &C);
  for (int i = 0; i < 2; ++i)
    printf("%g %g\n", c[i * 2], c[i * 2 + 1]);
  return 0;
}
