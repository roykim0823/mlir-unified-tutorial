/* C driver for interchange_manual.mlir + skewing_manual.mlir, used by
 * test/part1/ch06_interchange_skewing_run.mlir. Runs every version and prints
 * whether it matches the original, plus the bottom-right stencil value. */
#include <stdio.h>
#include <string.h>

#include "../memref2d.h"

void _mlir_ciface_copy_ij(MemRef2D *A, MemRef2D *B);
void _mlir_ciface_copy_ji(MemRef2D *A, MemRef2D *B);
void _mlir_ciface_stencil(MemRef2D *A);
void _mlir_ciface_stencil_skewed(MemRef2D *A);
void _mlir_ciface_stencil_wavefront(MemRef2D *A);

static void ones(float *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = 1.0f;
}

int main(void) {
  float a[4 * 5], bij[4 * 5] = {0}, bji[4 * 5] = {0};
  for (int i = 0; i < 20; ++i)
    a[i] = (float)i;
  MemRef2D A = memref2d(a, 4, 5);
  MemRef2D Bij = memref2d(bij, 4, 5), Bji = memref2d(bji, 4, 5);
  _mlir_ciface_copy_ij(&A, &Bij);
  _mlir_ciface_copy_ji(&A, &Bji);
  printf("interchange: %s\n",
         memcmp(bij, a, sizeof a) == 0 && memcmp(bji, a, sizeof a) == 0
             ? "same" : "DIFFERENT");

  float s[64], sk[64], wf[64];
  ones(s, 64), ones(sk, 64), ones(wf, 64);
  MemRef2D S = memref2d(s, 8, 8), SK = memref2d(sk, 8, 8), WF = memref2d(wf, 8, 8);
  _mlir_ciface_stencil(&S);
  _mlir_ciface_stencil_skewed(&SK);
  _mlir_ciface_stencil_wavefront(&WF);
  printf("stencil: A[7][7] = %g\n", s[63]);
  printf("skewed: %s\n", memcmp(s, sk, sizeof s) == 0 ? "same" : "DIFFERENT");
  printf("wavefront: %s\n", memcmp(s, wf, sizeof s) == 0 ? "same" : "DIFFERENT");
  return 0;
}
