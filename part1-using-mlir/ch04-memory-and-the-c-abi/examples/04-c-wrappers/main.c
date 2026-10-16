// The same calls as aot_main.py, from C: the memref descriptor is just a C
// struct, and _mlir_ciface_ functions take pointers to it. Chapter 28 does the
// same from C++ for a whole compiled model.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// memref<?xf32> and memref<?x?xf32> after --finalize-memref-to-llvm.
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef1D;
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef2D;

// Generated because of llvm.emit_c_interface. A returned memref becomes an
// out-pointer in FRONT of the arguments.
void _mlir_ciface_add_vector_to_matrix(MemRef2D *result, MemRef2D *M, MemRef1D *v);
// No wrapper: the descriptors are expanded field by field.
void add_vector_to_matrix_into(
    float *M_alloc, float *M_align, int64_t M_off, int64_t M_s0, int64_t M_s1,
    int64_t M_st0, int64_t M_st1,
    float *v_alloc, float *v_align, int64_t v_off, int64_t v_s0, int64_t v_st0,
    float *o_alloc, float *o_align, int64_t o_off, int64_t o_s0, int64_t o_s1,
    int64_t o_st0, int64_t o_st1);

static void print(const char *label, const float *p, int64_t rows, int64_t cols,
                  int64_t row_stride) {
  printf("%s\n", label);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) printf(" %5.1f", p[i * row_stride + j]);
    printf("\n");
  }
}

int main(void) {
  float m[2][3] = {{0, 1, 2}, {3, 4, 5}};
  float v[3] = {10, 20, 30};
  MemRef2D M = {&m[0][0], &m[0][0], 0, {2, 3}, {3, 1}};
  MemRef1D V = {v, v, 0, {3}, {1}};

  MemRef2D r;
  _mlir_ciface_add_vector_to_matrix(&r, &M, &V);
  printf("result: sizes=[%lld, %lld] strides=[%lld, %lld] offset=%lld\n",
         (long long)r.sizes[0], (long long)r.sizes[1], (long long)r.strides[0],
         (long long)r.strides[1], (long long)r.offset);
  print("_mlir_ciface_add_vector_to_matrix:", r.aligned + r.offset, r.sizes[0],
        r.sizes[1], r.strides[0]);
  free(r.allocated);  // callee allocated with malloc; free the BASE pointer

  float out[2][3] = {{0}};
  add_vector_to_matrix_into(&m[0][0], &m[0][0], 0, 2, 3, 3, 1,
                            v, v, 0, 3, 1,
                            &out[0][0], &out[0][0], 0, 2, 3, 3, 1);
  print("add_vector_to_matrix_into (expanded descriptors):", &out[0][0], 2, 3, 3);
  return 0;
}
