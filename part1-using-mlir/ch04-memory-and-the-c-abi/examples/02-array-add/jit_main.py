"""Just-in-time: compile array_add.mlir at run time and call it -- no files.

Ported from mlir-ir-lowering's jit_main.py, which shelled out to mlir-opt and
mlir-translate and JIT-compiled the LLVM IR with llvmlite. Here everything
stays in-process with the MLIR Python bindings shipped in the wheel:

  Module.parse          -- read the .mlir text
  PassManager           -- the same lowering as build.sh, as a pass pipeline
  ExecutionEngine       -- translate to LLVM IR and JIT it (LLVM ORC)
  engine.invoke(name)   -- call _mlir_ciface_<name> through its packed wrapper
"""

import ctypes
import os
import sys

import numpy as np
from mlir.execution_engine import ExecutionEngine
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "tools"))
from np_memref import numpy_to_memref  # noqa: E402

PIPELINE = (
    "builtin.module("
    "convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,"
    "convert-func-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)"
)


def compile_and_load(mlir_path):
    """Lower the module to the llvm dialect and JIT-compile it."""
    with Context():
        module = Module.parse(open(mlir_path).read())
        # A failing pass raises MLIRError here, at run time -- one of the JIT
        # trade-offs (an AOT build would have failed at build time instead).
        PassManager.parse(PIPELINE).run(module.operation)
        return ExecutionEngine(module, opt_level=3)


def main():
    engine = compile_and_load(os.path.join(HERE, "array_add.mlir"))

    size = 1024
    a = np.ones(size, dtype=np.float32)
    b = np.ones(size, dtype=np.float32) * 2
    c = np.zeros(size, dtype=np.float32)
    a_desc, b_desc, c_desc = (numpy_to_memref(x) for x in (a, b, c))

    # invoke("array_add", ...) calls _mlir_ciface_array_add through its packed
    # wrapper, so every argument is a pointer TO the C-interface argument --
    # and the C-interface argument is itself a pointer to a descriptor.
    arg = lambda d: ctypes.pointer(ctypes.pointer(d))  # noqa: E731
    engine.invoke("array_add", arg(a_desc), arg(b_desc), arg(c_desc))

    np.testing.assert_array_almost_equal(c, a + b)
    print("Array addition successful!")
    print(f"First few elements: {c[:5]}")


if __name__ == "__main__":
    main()
