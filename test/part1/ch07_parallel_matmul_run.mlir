// Chapter 7: run the auto-parallelized matmul on the OpenMP runtime.
// --affine-parallelize turns the i/j loops into affine.parallel, which
// lowers to scf.parallel and then to an omp.parallel + omp.wsloop nest;
// the C driver checks the result against a sequential C triple loop.
// Only the verdict line is checked: the timings vary from run to run.
//
// REQUIRES: libomp
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/05-parallel-matmul/matmul.mlir --affine-parallelize --lower-affine --canonicalize --convert-scf-to-openmp | mlir-opt --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-openmp-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -O2 -Wno-override-module -x ir -c %t.compat.ll -o %t.matmul.o
// RUN: clang -O2 -c %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/05-parallel-matmul/driver.c -o %t.driver.o
// RUN: clang %t.driver.o %t.matmul.o %omp_link_flags -o %t.exe
// RUN: env OMP_NUM_THREADS=4 %t.exe 64 | FileCheck %s

// CHECK: N=64 threads=4
// CHECK-NEXT: parallel matmul matches
