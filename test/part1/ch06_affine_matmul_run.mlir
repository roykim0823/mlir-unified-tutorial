// Chapter 6: the hand-written affine matmul, lowered, compiled and called
// from C (examples/01-affine-matmul). ll_fp_compat.py bridges LLVM 24's
// .ll output to the older clang.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/01-affine-matmul/matmul.mlir -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -Wno-override-module -x ir %t.compat.ll -c -o %t.obj
// RUN: clang %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/01-affine-matmul/matmul_main.c %t.obj -o %t.exe
// RUN: %t.exe | FileCheck %s

// [1 2 3; 4 5 6] x [7 8; 9 10; 11 12]
// CHECK: 58 64
// CHECK-NEXT: 139 154
