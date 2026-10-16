// C driver for the fused addmul kernel, used by test/part1/ch08_fusion.mlir
// (lit cannot rely on NumPy, so the test calls the kernel from C instead of
// run.py). Same inputs as run.py; prints out = (a + b) * c.
#include <stdint.h>
#include <stdio.h>

// The struct --finalize-memref-to-llvm uses for a rank-1 memref (Chapter 4).
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1D;

void _mlir_ciface_addmul(MemRef1D *a, MemRef1D *b, MemRef1D *c, MemRef1D *out);

static MemRef1D wrap(float *data, int64_t n) {
  MemRef1D m = {data, data, 0, {n}, {1}};
  return m;
}

int main(void) {
  float a[10], b[10], c[10], out[10] = {0};
  for (int i = 0; i < 10; ++i) {
    a[i] = (float)i;
    b[i] = 2.0f;
    c[i] = (float)(i + 1);
  }
  MemRef1D ma = wrap(a, 10), mb = wrap(b, 10), mc = wrap(c, 10), mo = wrap(out, 10);
  _mlir_ciface_addmul(&ma, &mb, &mc, &mo);

  int ok = 1;
  printf("out =");
  for (int i = 0; i < 10; ++i) {
    printf(" %g", out[i]);
    ok &= out[i] == (a[i] + b[i]) * c[i];
  }
  printf("\n%s\n", ok ? "fused addmul OK" : "fused addmul MISMATCH");
  return ok ? 0 : 1;
}
