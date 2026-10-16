// Chapter 25: the hand-written dense+ReLU layer (linalg.matmul + a fused
// bias/ReLU linalg.generic) is lowered to LLVM, compiled to a native object,
// and called from a C driver that checks it against a plain-C reference.
//
// ll_fp_compat.py bridges LLVM 24's .ll output to the older clang
// (float literal syntax; the `nocreateundeforpoison` intrinsic attribute).
//
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch25-neural-nets-from-scratch/examples/mlir_layer/dense_relu.mlir --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -Wno-override-module -x ir %t.compat.ll -c -o %t.obj
// RUN: clang -c %project_source_dir/part6-end-to-end-ml/ch25-neural-nets-from-scratch/examples/mlir_layer/dense_relu_main.c -o %t.main.o
// RUN: clang %t.main.o %t.obj -o %t.exe
// RUN: %t.exe | FileCheck %s

// CHECK: 1.1000 2.3000 2.3000 0.0000
// CHECK-NEXT: 0.0000 2.9250 2.0500 0.0000
// CHECK-NEXT: max abs error vs C reference: ok
