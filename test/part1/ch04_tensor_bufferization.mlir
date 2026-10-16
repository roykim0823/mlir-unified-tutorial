// Chapter 4, step 1: the three identity-matrix kernels. Checks what
// one-shot-bufferize makes of each calling convention, then compiles all
// three and runs them from C.
//
// RUN: split-file %s %t
//
// identity_fill: tensor.generate has no destination, so bufferization
// allocates a temporary and copies it into the caller's buffer.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_fill.mlir --one-shot-bufferize="bufferize-function-boundaries=true" | FileCheck %s --check-prefix=FILL
// FILL-LABEL: func.func @identity(%arg0: memref<?x?xi32>)
// FILL:         %[[TMP:.*]] = memref.alloc(
// FILL:         linalg.map outs(%[[TMP]]
// FILL:         memref.copy %[[TMP]], %arg0
//
// identity_fill_inplace: linalg.map writes straight into %arg0; no alloc.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_fill_inplace.mlir --one-shot-bufferize="bufferize-function-boundaries=true" --canonicalize | FileCheck %s --check-prefix=INPLACE
// INPLACE-LABEL: func.func @identity(%arg0: memref<?x?xi32>)
// INPLACE-NOT:     memref.alloc
// INPLACE:         linalg.map outs(%arg0
// INPLACE-NOT:     memref.copy
// INPLACE:         return
//
// identity_return: the callee allocates the result and returns it.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_return.mlir --one-shot-bufferize="bufferize-function-boundaries=true" | FileCheck %s --check-prefix=RETURN
// RETURN-LABEL: func.func @identity(%arg0: index, %arg1: index) -> memref<?x?xi32>
// RETURN:         %[[R:.*]] = memref.alloc(%arg0, %arg1)
// RETURN:         return %[[R]] : memref<?x?xi32>
//
// The temporary of identity_fill is freed only after the deallocation pipeline:
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_fill.mlir --one-shot-bufferize="bufferize-function-boundaries=true" --buffer-deallocation-pipeline | FileCheck %s --check-prefix=DEALLOC
// DEALLOC: memref.copy
// DEALLOC: memref.dealloc
//
// Compile each kernel on its own and link it with the C driver below, built
// once per calling convention (all three export _mlir_ciface_identity).
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_fill.mlir --one-shot-bufferize="bufferize-function-boundaries=true" --buffer-deallocation-pipeline --canonicalize --convert-linalg-to-loops --convert-bufferization-to-memref --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir | python3 %project_source_dir/tools/ll_fp_compat.py > %t/identity_fill.ll
// RUN: clang -Wno-override-module -O2 -c -x ir %t/identity_fill.ll -o %t/identity_fill.o
// RUN: clang -DMODE_FILL %t/driver.c %t/identity_fill.o -o %t/identity_fill.exe
// RUN: %t/identity_fill.exe | FileCheck %s --check-prefix=EXE-FILL
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_fill_inplace.mlir --one-shot-bufferize="bufferize-function-boundaries=true" --buffer-deallocation-pipeline --canonicalize --convert-linalg-to-loops --convert-bufferization-to-memref --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir | python3 %project_source_dir/tools/ll_fp_compat.py > %t/identity_fill_inplace.ll
// RUN: clang -Wno-override-module -O2 -c -x ir %t/identity_fill_inplace.ll -o %t/identity_fill_inplace.o
// RUN: clang -DMODE_INPLACE %t/driver.c %t/identity_fill_inplace.o -o %t/identity_fill_inplace.exe
// RUN: %t/identity_fill_inplace.exe | FileCheck %s --check-prefix=EXE-INPLACE
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch04-memory-and-the-c-abi/examples/01-tensor/identity_return.mlir --one-shot-bufferize="bufferize-function-boundaries=true" --buffer-deallocation-pipeline --canonicalize --convert-linalg-to-loops --convert-bufferization-to-memref --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-translate --mlir-to-llvmir | python3 %project_source_dir/tools/ll_fp_compat.py > %t/identity_return.ll
// RUN: clang -Wno-override-module -O2 -c -x ir %t/identity_return.ll -o %t/identity_return.o
// RUN: clang -DMODE_RETURN %t/driver.c %t/identity_return.o -o %t/identity_return.exe
// RUN: %t/identity_return.exe | FileCheck %s --check-prefix=EXE-RETURN
// EXE-FILL:    fill: 10000 01000 00100 00010 00001
// EXE-INPLACE: inplace: 10000 01000 00100 00010 00001
// EXE-RETURN:  return: 10000 01000 00100 00010 00001 sizes=5x5 strides=5,1

//--- driver.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { int32_t *allocated, *aligned; int64_t offset, sizes[2], strides[2]; } MemRef2D;
static void dump(const char *label, const int32_t *p, int64_t rs) {
  printf("%s", label);
  for (int i = 0; i < 5; ++i) { printf(" "); for (int j = 0; j < 5; ++j) printf("%d", p[i * rs + j]); }
}
#ifdef MODE_RETURN
void _mlir_ciface_identity(MemRef2D *result, int64_t m, int64_t n);
int main(void) {
  MemRef2D r;
  _mlir_ciface_identity(&r, 5, 5);
  dump("return:", r.aligned + r.offset, r.strides[0]);
  printf(" sizes=%lldx%lld strides=%lld,%lld\n", (long long)r.sizes[0], (long long)r.sizes[1],
         (long long)r.strides[0], (long long)r.strides[1]);
  free(r.allocated);
  return 0;
}
#else
void _mlir_ciface_identity(MemRef2D *out);
int main(void) {
  int32_t buf[5][5];
  for (int i = 0; i < 25; ++i) (&buf[0][0])[i] = 7;   // garbage the kernel must overwrite
  MemRef2D d = {&buf[0][0], &buf[0][0], 0, {5, 5}, {5, 1}};
  _mlir_ciface_identity(&d);
#ifdef MODE_FILL
  dump("fill:", &buf[0][0], 5);
#else
  dump("inplace:", &buf[0][0], 5);
#endif
  printf("\n");
  return 0;
}
#endif
