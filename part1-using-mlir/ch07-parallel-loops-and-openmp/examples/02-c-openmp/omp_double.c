// The C reference for this chapter: double a 10-element array with OpenMP.
//
//   #pragma omp parallel   opens a team of threads (the region runs on each)
//   #pragma omp for        shares the loop's iterations across the team
//
// MLIR's `omp` dialect mirrors these directives op for op (see 03-omp-dialect),
// and both lower to the same calls into the OpenMP runtime (libomp).
//
// Besides the doubled array, the program records which thread executed each
// iteration, so you can see the work being split up. build.sh compiles it
// with `clang -fopenmp` against Homebrew's (or the system's) libomp.
#include <omp.h>
#include <stdio.h>

static int owner[10];

void kernel(const float *input, float *output) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      output[i] = input[i] * 2.0f;
      owner[i] = omp_get_thread_num();
    }
  }
}

int main(void) {
  float input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float output[10];
  kernel(input, output);
  for (int i = 0; i < 10; i++)
    printf("%f ", output[i]);
  printf("\nthreads available: %d\niteration -> thread:", omp_get_max_threads());
  for (int i = 0; i < 10; i++)
    printf(" %d", owner[i]);
  printf("\n");
  return 0;
}
