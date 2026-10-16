// End-to-end test for a real model shape: the MNIST MLP (784->512->512->10)
// goes from its checked-in linalg-on-tensors snapshot to a native binary.
// Its three linalg.matmul ops become three cblas_sgemm calls, and the
// lowered module calls memrefCopy from the MLIR runner-utils library (the
// flatten of the NCHW input), hence the extra link flags. Expected logits
// are PyTorch's own output for a tensor of ones (see run_mnist_model.py).
//
// RUN: tutorial-opt --linalg-to-bufferization %project_source_dir/part6-end-to-end-ml/src/mnist/mnist_model_linalg.mlir > %t.buf.mlir
// RUN: tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm %t.buf.mlir | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -x ir %t.compat.ll -c -o %t.obj
// RUN: clang++ -c %project_source_dir/part6-end-to-end-ml/src/mnist/mnist_call.cpp -o %t.main.o
// RUN: clang++ %t.main.o %t.obj -o %t.exe %blas_link_flags -L%mlir_libs_dir -lmlir_c_runner_utils -Wl,-rpath,%mlir_libs_dir
// RUN: %t.exe | FileCheck %s

// CHECK: 0.0149 -0.0860 0.1660 -0.1409 -0.1119 -0.0749 -0.1015 -0.1354 -0.0231 0.0546
