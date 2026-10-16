// End-to-end test for the ch28/ch29 CPU pipeline: the checked-in
// linalg-on-tensors snapshot of the sample model (exported from PyTorch by
// part6-end-to-end-ml/src/sample/lower_sample_model.py) is lowered to LLVM,
// compiled to a native object, linked against a CBLAS implementation, and
// called from C++. The expected numbers are PyTorch's own output for a
// tensor of ones (see run_sample_model.py).
//
// The ll_fp_compat.py filter bridges LLVM 24's typed float literals (f0x...)
// to the older syntax the system clang accepts — see ch28.
//
// RUN: tutorial-opt --linalg-to-bufferization %project_source_dir/part6-end-to-end-ml/src/sample/sample_model_linalg.mlir > %t.buf.mlir
// RUN: tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm %t.buf.mlir | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -x ir %t.compat.ll -c -o %t.obj
// RUN: clang++ -c %project_source_dir/part6-end-to-end-ml/src/sample/sample_call.cpp -o %t.main.o
// RUN: clang++ %t.main.o %t.obj -o %t.exe %blas_link_flags
// RUN: %t.exe | FileCheck %s

// CHECK: 0.3844 0.0000 0.2425 0.2203 0.0000
// CHECK: 0.9251 0.0000 0.6383 0.4868 0.0000
// CHECK: 0.9336 0.0000 0.3170 0.4453 0.0000
