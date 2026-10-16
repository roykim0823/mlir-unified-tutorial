// Runs the two OpenMP reductions compiled from reduce.mlir and checks them.
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef1D;

float _mlir_ciface_sum(MemRef1D *A);
float _mlir_ciface_max(MemRef1D *A);
int omp_get_max_threads(void);

enum { N = 4096 };

int main(void) {
  static float a[N];
  // 0 + 1 + ... + 4095 = 8386560. Every partial sum is an integer below 2^24,
  // so float addition is exact in ANY order: the parallel result must match.
  for (int i = 0; i < N; i++)
    a[i] = (float)i;
  // Hide the maximum somewhere in the middle (not in the last chunk).
  a[1234] = 99999.0f;
  float expect_sum = 8386560.0f - 1234.0f + 99999.0f;

  MemRef1D A = {a, a, 0, {N}, {1}};
  float s = _mlir_ciface_sum(&A), m = _mlir_ciface_max(&A);
  printf("threads=%d  sum=%.1f (expect %.1f)  max=%.1f (expect 99999.0)\n",
         omp_get_max_threads(), s, expect_sum, m);
  int ok = s == expect_sum && m == 99999.0f;
  printf("%s\n", ok ? "reductions match" : "MISMATCH");
  return !ok;
}
