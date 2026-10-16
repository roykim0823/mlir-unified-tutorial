// Chapter 26: the numerically stable row-wise softmax (three scf.for passes
// per row, math.exp lowered to a libm expf call) is compiled to a native
// object and called from a C driver that checks it against a plain-C
// reference. The third row's scores (~1000) would overflow a naive exp().
//
// ll_fp_compat.py bridges LLVM 24's .ll output to the older clang
// (here the -FLT_MAX literal f0xFF7FFFFF; the `nocreateundeforpoison`
// intrinsic attribute).
//
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch26-transformers-and-softmax/examples/mlir_attention/softmax.mlir --convert-scf-to-cf --convert-cf-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -Wno-override-module -x ir %t.compat.ll -c -o %t.obj
// RUN: clang -c %project_source_dir/part6-end-to-end-ml/ch26-transformers-and-softmax/examples/mlir_attention/softmax_main.c -o %t.main.o
// RUN: clang %t.main.o %t.obj -o %t.exe -lm
// RUN: %t.exe | FileCheck %s

// CHECK: 0.0321 0.0871 0.2369 0.6439 | sum 1.0000
// CHECK-NEXT: 0.2500 0.2500 0.2500 0.2500 | sum 1.0000
// CHECK-NEXT: 0.0321 0.0871 0.2369 0.6439 | sum 1.0000
// CHECK-NEXT: max abs error vs C reference: ok
