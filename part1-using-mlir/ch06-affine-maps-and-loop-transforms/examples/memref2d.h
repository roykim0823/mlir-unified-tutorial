/* The rank-2 memref descriptor that -finalize-memref-to-llvm produces for
 * memref<?x?xf32> (Chapter 4): { ptr, ptr, i64, [2 x i64], [2 x i64] }.
 * Shared by the Chapter 6 C drivers that the lit tests compile. */
#ifndef CH06_MEMREF2D_H
#define CH06_MEMREF2D_H

#include <stdint.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D;

/* Wrap a row-major rows x cols buffer (no copy). */
static inline MemRef2D memref2d(float *data, int64_t rows, int64_t cols) {
  MemRef2D m = {data, data, 0, {rows, cols}, {cols, 1}};
  return m;
}

#endif
