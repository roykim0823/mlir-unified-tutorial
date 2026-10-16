// Calls every function of dialect_tour.mlir after it has been compiled to
// native code. The prototypes are what the lowered llvm.func signatures mean
// in C: i32 -> int32_t, f32 -> float, index -> int64_t (on a 64-bit target),
// i1 -> bool.
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

int32_t arithmetic_example(int32_t a, float b);
float math_example(float x);
int64_t compute_offset(int64_t i, int64_t j, int64_t stride);
int32_t select(int32_t a, int32_t b, bool flag);
int32_t classify(int32_t v);
int32_t abs_diff(int32_t a, int32_t b);
int64_t collatz_steps(int64_t n);

int main(void) {
  printf("arithmetic_example(1, 2.9) = %d\n", arithmetic_example(1, 2.9f));
  printf("math_example(2.0)          = %.6f\n", math_example(2.0f));
  printf("compute_offset(2, 3, 10)   = %lld\n", (long long)compute_offset(2, 3, 10));
  printf("select(7, 9, true/false)   = %d %d\n", select(7, 9, true), select(7, 9, false));
  printf("classify(0, 1, 5)          = %d %d %d\n", classify(0), classify(1), classify(5));
  printf("abs_diff(3, 10), (10, 3)   = %d %d\n", abs_diff(3, 10), abs_diff(10, 3));
  printf("collatz_steps(27)          = %lld\n", (long long)collatz_steps(27));
  return 0;
}
