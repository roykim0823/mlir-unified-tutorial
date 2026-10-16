// Runs the OpenMP-parallel matmul compiled from matmul.mlir, checks it against
// a plain C triple loop, and times both.
//
//   usage: matmul [N]        (square N x N matrices, default 512)
//
// The memref<?x?xf32> arguments are passed as pointers to rank-2 descriptors
// (Chapter 4): {allocated, aligned, offset, sizes[2], strides[2]}.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef2D;

void _mlir_ciface_matmul(MemRef2D *A, MemRef2D *B, MemRef2D *C);
int omp_get_max_threads(void);
double omp_get_wtime(void);

static MemRef2D view(float *p, int64_t rows, int64_t cols) {
  MemRef2D m = {p, p, 0, {rows, cols}, {cols, 1}};
  return m;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 512;
  float *a = malloc(sizeof(float) * n * n), *b = malloc(sizeof(float) * n * n);
  float *c = calloc((size_t)n * n, sizeof(float));
  float *ref = calloc((size_t)n * n, sizeof(float));
  srand(0);
  for (int i = 0; i < n * n; i++) {
    a[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    b[i] = (float)rand() / (float)RAND_MAX - 0.5f;
  }

  // Reference: the same i-j-k loop order, sequential, in C.
  double t0 = omp_get_wtime();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        ref[i * n + j] += a[i * n + k] * b[k * n + j];
  double t_ref = omp_get_wtime() - t0;

  MemRef2D A = view(a, n, n), B = view(b, n, n), C = view(c, n, n);
  t0 = omp_get_wtime();
  _mlir_ciface_matmul(&A, &B, &C);
  double t_mlir = omp_get_wtime() - t0;

  float err = 0;
  for (int i = 0; i < n * n; i++)
    err = fmaxf(err, fabsf(c[i] - ref[i]));
  printf("N=%d threads=%d  C reference %.3fs  MLIR+OpenMP %.3fs  max abs error %.2e\n",
         n, omp_get_max_threads(), t_ref, t_mlir, err);
  int ok = err < 1e-3f;
  printf("%s\n", ok ? "parallel matmul matches" : "MISMATCH");
  free(a); free(b); free(c); free(ref);
  return !ok;
}
