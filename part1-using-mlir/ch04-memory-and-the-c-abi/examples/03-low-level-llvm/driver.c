// Calls the "inspect-only" snippets of structs_arrays.mlir and vectors.mlir
// after they have been compiled, to show that the LLVM-level types are
// exactly C types:
//   !llvm.struct<(i32, f32)>  <->  struct { int32_t a; float b; }
//   vector<4xf32>             <->  a 16-byte SIMD vector of 4 floats
#include <stdint.h>
#include <stdio.h>

typedef struct { int32_t a; float b; } Pair;
typedef float v4f __attribute__((vector_size(16)));

// make_pair / pair_first take and return the struct BY VALUE, which does not
// follow the C ABI (see structs_arrays.mlir), so C calls the pointer-based
// variants instead.
void make_pair_into(Pair *out);
float pair_second(const Pair *p);
int32_t array_demo(void);
int32_t pair_array_first_a(void);

v4f vadd(v4f a, v4f b);
v4f vmul(v4f a, v4f b);
float vextract(v4f v);
v4f vshuffle(v4f a, v4f b);
float vreduce_sum(v4f v);
v4f vsplat(float x);

static void print4(const char *name, v4f v) {
  printf("%-12s = [%g, %g, %g, %g]\n", name, v[0], v[1], v[2], v[3]);
}

int main(void) {
  Pair p;
  make_pair_into(&p);
  printf("make_pair_into(&p) = {%d, %.2f}\n", p.a, p.b);
  printf("pair_second(&p)    = %.2f\n", pair_second(&p));
  printf("array_demo()       = %d\n", array_demo());
  printf("pair_array_first_a = %d\n", pair_array_first_a());

  v4f a = {1, 2, 3, 4}, b = {10, 20, 30, 40};
  print4("vadd(a, b)", vadd(a, b));
  print4("vmul(a, b)", vmul(a, b));
  printf("%-12s = %g\n", "vextract(b)", vextract(b));
  print4("vshuffle", vshuffle(a, b));
  printf("%-12s = %g\n", "vreduce_sum", vreduce_sum(b));
  print4("vsplat(7)", vsplat(7));
  return 0;
}
