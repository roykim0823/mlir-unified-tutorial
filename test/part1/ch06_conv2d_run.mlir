// Chapter 6: affine conv2d, compiled as written and after affine optimization
// (LICM + 4x4 tiling + 2x unroll); both must give the same exact output
// (examples/05-conv2d).
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/05-conv2d/conv2d.mlir -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.plain.ll
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/05-conv2d/conv2d.mlir --pass-pipeline="builtin.module(func.func(affine-loop-invariant-code-motion,affine-loop-tile{tile-size=4},affine-loop-unroll{unroll-factor=2},canonicalize))" > %t.opt.mlir
// RUN: FileCheck %s --check-prefix=OPT < %t.opt.mlir
// RUN: mlir-opt %t.opt.mlir -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.opt.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.plain.ll > %t.plain.compat.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.opt.ll > %t.opt.compat.ll
// RUN: clang -Wno-override-module -x ir %t.plain.compat.ll -c -o %t.plain.obj
// RUN: clang -Wno-override-module -x ir %t.opt.compat.ll -c -o %t.opt.obj
// RUN: clang %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/05-conv2d/conv2d_main.c %t.plain.obj -o %t.plain.exe
// RUN: clang %project_source_dir/part1-using-mlir/ch06-affine-maps-and-loop-transforms/examples/05-conv2d/conv2d_main.c %t.opt.obj -o %t.opt.exe
// RUN: %t.plain.exe | FileCheck %s
// RUN: %t.opt.exe | FileCheck %s

// CHECK: 312 348 384
// CHECK-NEXT: 492 528 564
// CHECK-NEXT: 672 708 744
// CHECK-NEXT: 852 888 924

// The optimized kernel: hoisted zero, tiled output loops with min bounds, and
// an unrolled filter-column loop plus its cleanup loop.
// OPT: arith.constant 0.000000e+00 : f32
// OPT: affine.for %{{.*}} = 0 to %{{.*}} step 4
// OPT: affine.for %{{.*}} = 0 to %{{.*}} step 4
// OPT: to min #{{.*}}
// OPT: to min #{{.*}}
// OPT: step 2 iter_args
// OPT: affine.for {{.*}} iter_args
