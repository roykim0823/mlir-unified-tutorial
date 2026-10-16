// Chapter 7: omp dialect -> LLVM IR -> object file. Checks the outlined
// parallel region and the OpenMP runtime (__kmpc_*) calls, and that the
// result compiles. Linking and running need libomp, which the MLIR wheel
// does not ship, so the run lives in the chapter's build.sh scripts.
//
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/03-omp-dialect/omp_double.mlir --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-openmp-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=DIALECT
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/03-omp-dialect/omp_double.mlir --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-openmp-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.double.ll
// RUN: FileCheck %s --check-prefix=DOUBLE < %t.double.ll
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/06-parallel-reduction/reduce.mlir --affine-parallelize=parallel-reductions=true --lower-affine --canonicalize --convert-scf-to-openmp --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-openmp-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=RED
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch07-parallel-loops-and-openmp/examples/05-parallel-matmul/matmul.mlir --affine-parallelize --lower-affine --canonicalize --convert-scf-to-openmp --finalize-memref-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-openmp-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t.matmul.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.matmul.ll > %t.matmul.compat.ll
// RUN: clang -Wno-override-module -x ir -c %t.matmul.compat.ll -o %t.matmul.o
// RUN: nm %t.matmul.o | FileCheck %s --check-prefix=SYMS

// The omp ops survive --convert-openmp-to-llvm; only their contents change.
// DIALECT: llvm.func @kernel
// DIALECT:   omp.parallel
// DIALECT:     omp.wsloop
// DIALECT:       omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step
// DIALECT:     omp.barrier

// mlir-translate outlines the region and calls the runtime.
// DOUBLE-LABEL: define void @kernel(
// DOUBLE:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @kernel..omp_par, ptr %structArg)
// DOUBLE-LABEL: define internal void @kernel..omp_par(
// DOUBLE:         call void @__kmpc_for_static_init_4u(ptr @{{.*}}, i32 %omp_global_thread_num, i32 34,
// DOUBLE:         call void @__kmpc_for_static_fini(
// DOUBLE:         call void @__kmpc_barrier(
// DOUBLE:         call void @__kmpc_barrier(

// RED: call i32 @__kmpc_reduce(
// RED: call void @__kmpc_end_reduce(

// SYMS-DAG: U {{_*}}__kmpc_fork_call
// SYMS-DAG: U {{_*}}__kmpc_for_static_init_8u
// SYMS-DAG: T {{_*}}_mlir_ciface_matmul
