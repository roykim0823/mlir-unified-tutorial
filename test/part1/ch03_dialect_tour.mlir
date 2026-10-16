// Chapter 3: the standard-dialect tour (func, arith, math, index, cf, scf)
// lowered to LLVM, compiled, and called from C.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch03-llvm-ir-and-native-code/examples/03-dialect-tour/dialect_tour.mlir \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm \
// RUN:   --convert-math-to-llvm --convert-index-to-llvm --convert-arith-to-llvm \
// RUN:   --reconcile-unrealized-casts -o %t.llvm.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.llvm.mlir
// LOWERED-LABEL: llvm.func @math_example
// LOWERED:         llvm.intr.sin
// LOWERED:         llvm.intr.sqrt
// LOWERED-LABEL: llvm.func @compute_offset(%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64) -> i64
// LOWERED:         llvm.mul
// LOWERED-LABEL: llvm.func @classify
// LOWERED:         llvm.switch
//
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -Wno-override-module -O2 -x ir -c %t.compat.ll -o %t.o
// RUN: clang -O2 %project_source_dir/part1-using-mlir/ch03-llvm-ir-and-native-code/examples/03-dialect-tour/driver.c %t.o -lm -o %t.exe
// RUN: %t.exe | FileCheck %s
//
// CHECK: arithmetic_example(1, 2.9) = 45
// CHECK: math_example(2.0)          = 1.285941
// CHECK: compute_offset(2, 3, 10)   = 23
// CHECK: select(7, 9, true/false)   = 7 9
// CHECK: classify(0, 1, 5)          = 10 20 -1
// CHECK: abs_diff(3, 10), (10, 3)   = 7 7
// CHECK: collatz_steps(27)          = 111
