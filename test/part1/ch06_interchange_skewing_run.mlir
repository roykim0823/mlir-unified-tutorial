// Chapter 6: hand-written interchange, skewing and wavefront must compute
// exactly what the original loops compute (examples/04-interchange-skewing).
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/04-interchange-skewing/interchange_manual.mlir -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.ic.ll
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/04-interchange-skewing/skewing_manual.mlir -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.sk.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ic.ll > %t.ic.compat.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.sk.ll > %t.sk.compat.ll
// RUN: clang -Wno-override-module -x ir %t.ic.compat.ll -c -o %t.ic.obj
// RUN: clang -Wno-override-module -x ir %t.sk.compat.ll -c -o %t.sk.obj
// RUN: clang %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/04-interchange-skewing/transforms_main.c %t.ic.obj %t.sk.obj -o %t.exe
// RUN: %t.exe | FileCheck %s

// CHECK: interchange: same
// A[7][7] of the all-ones stencil is the binomial coefficient C(14, 7).
// CHECK-NEXT: stencil: A[7][7] = 3432
// CHECK-NEXT: skewed: same
// CHECK-NEXT: wavefront: same

// The wavefront's inner loop really is parallel, with max/min bounds.
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/04-interchange-skewing/skewing_manual.mlir -lower-affine | FileCheck %s --check-prefix=WAVE
// WAVE-LABEL: func.func @stencil_wavefront
// WAVE: %[[LB:.*]] = arith.maxsi
// WAVE: %[[UB:.*]] = arith.minsi
// WAVE: scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]])
