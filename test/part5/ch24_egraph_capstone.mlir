// Chapter 24 capstone: the e-graph turns ((a*2)/2) * ((a*2)/2) into a*a, the
// result is emitted as MLIR, compiled to a native shared library, and run
// against the compiled original. %t is used as an output directory.
//
// The ll_fp_compat.py filter bridges LLVM 24's typed float literals to the
// older syntax the system clang accepts — see ch28.
//
// RUN: rm -rf %t
// RUN: python3 %project_source_dir/part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/capstone.py %t | FileCheck %s --check-prefix=CAP
// RUN: FileCheck %s --check-prefix=ORIG < %t/orig.mlir
// RUN: FileCheck %s --check-prefix=OPT < %t/opt.mlir
//
// MLIR's own greedy canonicalizer + CSE share the duplicated halves but
// (correctly, under IEEE semantics) do not cancel the multiply/divide.
// RUN: mlir-opt %t/orig.mlir --canonicalize --cse | FileCheck %s --check-prefix=CANON
//
// RUN: mlir-opt %t/orig.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t/orig.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t/orig.ll > %t/orig.compat.ll
// RUN: clang -shared -fPIC -Wno-override-module -x ir %t/orig.compat.ll -o %t/liborig.so
// RUN: mlir-opt %t/opt.mlir --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > %t/opt.ll
// RUN: python3 %project_source_dir/tools/ll_fp_compat.py < %t/opt.ll > %t/opt.compat.ll
// RUN: clang -shared -fPIC -Wno-override-module -x ir %t/opt.compat.ll -o %t/libopt.so
// RUN: python3 %project_source_dir/part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/run_compiled.py %t | FileCheck %s --check-prefix=EXEC

// CAP: original :  (((a * 2) / 2) * ((a * 2) / 2))   (5 ops)
// CAP-NEXT: optimized:  (a * a)   (1 ops)

// ORIG-LABEL: func.func @f(%a: f32) -> f32
// ORIG-COUNT-2: arith.divf
// ORIG: return %v9 : f32

// OPT-LABEL: func.func @f(%a: f32) -> f32 {
// OPT-NEXT:   %v1 = arith.mulf %a, %a : f32
// OPT-NEXT:   return %v1 : f32
// OPT-NEXT: }

// CANON: arith.mulf
// CANON-NEXT: arith.divf
// CANON-NEXT: arith.mulf
// CANON-NOT: arith.divf

// EXEC: f( 100.00):  original = 10000.0000   optimized = 10000.0000
// EXEC-NEXT: Capstone OK: optimized f(a) == original f(a) == a*a for all test inputs.
