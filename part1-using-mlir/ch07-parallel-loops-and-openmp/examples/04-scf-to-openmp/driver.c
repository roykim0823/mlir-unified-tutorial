// Calls the two kernels in scf_parallel.mlir through their C interface
// wrappers. A ranked memref is passed as a pointer to its descriptor struct
// (Chapter 4 explains the layout).
#include <stdint.h>
#include <stdio.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef1DF32;

typedef struct {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef1DI32;

void _mlir_ciface_add(MemRef1DF32 *A, MemRef1DF32 *B);
void _mlir_ciface_which_thread(MemRef1DI32 *owner);
int omp_get_max_threads(void);

int main(void) {
  float a[100], b[100];
  for (int i = 0; i < 100; i++) {
    a[i] = (float)i;
    b[i] = -1.0f;
  }
  MemRef1DF32 A = {a, a, 0, {100}, {1}}, B = {b, b, 0, {100}, {1}};
  _mlir_ciface_add(&A, &B);
  int bad = 0;
  for (int i = 0; i < 100; i++)
    bad += b[i] != 2.0f * a[i];
  printf("add: %s (B[0]=%g, B[99]=%g)\n", bad ? "MISMATCH" : "ok", b[0], b[99]);

  enum { N = 16 };
  int32_t owner[N];
  MemRef1DI32 O = {owner, owner, 0, {N}, {1}};
  _mlir_ciface_which_thread(&O);
  printf("threads available: %d\niteration -> thread:", omp_get_max_threads());
  for (int i = 0; i < N; i++)
    printf(" %d", owner[i]);
  printf("\n");
  return bad != 0;
}
