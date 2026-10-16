// Chapter 3: hand-written LLVM IR and the scf/index/arith example, compiled
// all the way to native executables (clang stands in for llc, which the MLIR
// wheel does not ship).
//
// Part A -- simple.ll returns 42 as the process exit code:
// RUN: clang -Wno-override-module -x ir -c %project_source_dir/part1-using-mlir/ch03-llvm-ir-and-native-code/examples/01-llvm-modules/simple.ll -o %t.simple.o
// RUN: clang %t.simple.o -o %t.simple
// RUN: bash -c '%t.simple; echo "exit=$?"' | FileCheck %s --check-prefix=SIMPLE
// SIMPLE: exit=42
//
// Part B -- example.mlir lowered to the llvm dialect:
// RUN: mlir-opt %project_source_dir/part1-using-mlir/ch03-llvm-ir-and-native-code/examples/02-mlir-to-native/example.mlir \
// RUN:   --convert-func-to-llvm --convert-math-to-llvm --convert-index-to-llvm \
// RUN:   --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
// RUN:   --reconcile-unrealized-casts -o %t.opt.mlir
// RUN: FileCheck %s --check-prefix=LLVMDIALECT < %t.opt.mlir
// LLVMDIALECT:      llvm.func @loop_add() -> i64 {
// LLVMDIALECT:        llvm.br ^bb1(%{{.*}}, %{{.*}} : i64, i64)
// LLVMDIALECT:      ^bb1(%[[IV:.*]]: i64, %[[ACC:.*]]: i64):
// LLVMDIALECT-NEXT:   llvm.icmp "slt" %[[IV]]
// LLVMDIALECT:        llvm.add %[[ACC]], %[[IV]] : i64
// LLVMDIALECT:      llvm.func @main() -> i32 {
// LLVMDIALECT:        llvm.trunc %{{.*}} : i64 to i32
// LLVMDIALECT-NOT:  index.
// LLVMDIALECT-NOT:  scf.
//
// ... then to LLVM IR, where block arguments become phi nodes:
// RUN: mlir-translate --mlir-to-llvmir %t.opt.mlir -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVMIR < %t.ll
// LLVMIR:      define i64 @loop_add() {
// LLVMIR:        phi i64
// LLVMIR-NEXT:   phi i64
// LLVMIR-NEXT:   icmp slt i64 %{{.*}}, 10
//
// ... then to a native executable whose exit code is 0+1+...+9 = 45:
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t.ll > %t.compat.ll
// RUN: clang -Wno-override-module -O2 -Xclang -disable-llvm-passes -x ir -c %t.compat.ll -o %t.example.o
// RUN: clang %t.example.o -o %t.example
// RUN: bash -c '%t.example; echo "exit=$?"' | FileCheck %s --check-prefix=EXAMPLE
// EXAMPLE: exit=45
//
// Back-end-only optimization (what llc does by default) keeps the loop, in
// registers; the full clang -O2 IR optimizer folds it away entirely.
// RUN: clang -Wno-override-module -O2 -Xclang -disable-llvm-passes -x ir -S %t.compat.ll -o - | FileCheck %s --check-prefix=LLCLIKE
// LLCLIKE-LABEL: _loop_add:
// LLCLIKE:         add x0, x0, x8
// LLCLIKE:         b.le
// RUN: clang -Wno-override-module -O2 -x ir -S %t.compat.ll -o - | FileCheck %s --check-prefix=O2
// O2-LABEL: _loop_add:
// O2-NEXT:  ; %bb.0:
// O2-NEXT:    mov w0, #45
