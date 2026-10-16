// RUN: not tutorial-opt %s 2>&1 | FileCheck %s

// CHECK: use of value '%b' expects different type than prior uses: 'i32' vs 'i64'
func.func @bad(%a: i32, %b: i64) -> i32 {
  %0 = arith.muli %a, %b : i32
  func.return %0 : i32
}
