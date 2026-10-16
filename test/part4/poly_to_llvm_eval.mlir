// RUN: tutorial-opt --poly-to-llvm %s | mlir-translate --mlir-to-llvmir | clang -x ir - -c -o %t.obj
// RUN: clang -c %project_source_dir/test/part4/poly_to_llvm_main.c -o %t.main.o
// RUN: clang %t.main.o %t.obj -o %t.exe
// RUN: %t.exe | FileCheck %s

// CHECK: 9
func.func @test_poly_fn(%arg : i32) -> i32 {
  // 2 + 3x + 4x^2 evaluated at x=1, should be 2+3+4
  %input = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<3>
  %0 = poly.eval %input, %arg: (!poly.poly<3>, i32) -> i32
  return %0 : i32
}
