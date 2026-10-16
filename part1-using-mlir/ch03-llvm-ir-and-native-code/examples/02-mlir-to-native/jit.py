"""JIT-compile example.mlir in-process and call @main -- no files, no clang.

This replaces the original tutorial's

    mlir-runner -e main -entry-point-result=i32 build/example_opt.mlir

(the MLIR wheel ships no mlir-runner). It uses the MLIR Python bindings that
come with the wheel: parse the module, run the same lowering pipeline as
build.sh with an in-process PassManager, hand the llvm-dialect module to
ExecutionEngine (LLVM's ORC JIT), and call the compiled function.

Usage:  .venv/bin/python jit.py [path/to/module.mlir]
"""

import ctypes
import os
import sys

from mlir.execution_engine import ExecutionEngine
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "example.mlir")

# The same passes as step 1 of build.sh, written as a textual pipeline.
PIPELINE = (
    "builtin.module("
    "convert-func-to-llvm,convert-math-to-llvm,convert-index-to-llvm,"
    "convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,"
    "reconcile-unrealized-casts)"
)

with Context():
    module = Module.parse(open(src).read())
    PassManager.parse(PIPELINE).run(module.operation)

    # Compile the whole module to machine code in memory.
    engine = ExecutionEngine(module, opt_level=2)

    # For every function @f, ExecutionEngine also emits a "packed" wrapper
    # `_mlir_f(void **args)`: args[i] points at the i-th argument, and the
    # slots after the arguments point at where results should be written.
    # raw_lookup("main") returns that wrapper's address. @main has no
    # arguments and one i32 result, so the array holds one result pointer.
    result = ctypes.c_int32(-1)
    packed_args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.pointer(result), ctypes.c_void_p))
    main = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(engine.raw_lookup("main"))
    main(packed_args)

print(result.value)
