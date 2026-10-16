/* C driver for conv2d.mlir (plain or optimized), used by
 * test/part1/ch06_conv2d_run.mlir. Input is 6x5 with input[r][c] = r*5 + c,
 * the filter is 3x3 with filter[r][c] = r*3 + c, so every output value is an
 * exact integer: output[i][j] = 312 + 180*i + 36*j. Prints the 4x3 output. */
#include <stdio.h>

#include "../memref2d.h"

void _mlir_ciface_conv_2d(MemRef2D *input, MemRef2D *filter, MemRef2D *output);

int main(void) {
  float in[6 * 5], flt[3 * 3], out[4 * 3] = {0};
  for (int k = 0; k < 6 * 5; ++k)
    in[k] = (float)k;
  for (int k = 0; k < 3 * 3; ++k)
    flt[k] = (float)k;
  MemRef2D I = memref2d(in, 6, 5), F = memref2d(flt, 3, 3), O = memref2d(out, 4, 3);
  _mlir_ciface_conv_2d(&I, &F, &O);
  for (int i = 0; i < 4; ++i)
    printf("%g %g %g\n", out[i * 3], out[i * 3 + 1], out[i * 3 + 2]);
  return 0;
}
