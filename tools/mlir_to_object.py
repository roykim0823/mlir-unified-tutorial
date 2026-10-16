#!/usr/bin/env python3
"""Compile an LLVM-dialect .mlir module straight to a native object file.

    python tools/mlir_to_object.py in_llvm.mlir out.o [shared_lib ...]

The usual route (mlir-translate -> tools/ll_fp_compat.py -> clang -x ir)
goes through textual LLVM IR. That breaks down for very large modules:
GPT-2's .ll is over 3 GB, and clang refuses any source file past ~2 GB
("ran out of source locations"). This tool skips the text entirely: the MLIR
Python bindings' ExecutionEngine translates and codegens the module in
process with the wheel's own LLVM (so there is no version skew either), and
dump_to_object_file writes the result.

ExecutionEngine links the module before it can dump it, so every external
symbol must resolve. Pass the shared libraries that provide them, e.g. the
wheel's libmlir_c_runner_utils for memrefCopy. libc/libm symbols resolve
from the running process. The object still lists them as undefined, so link
the final binary against the same libraries (plus BLAS) as usual.
"""

import sys
import time

from mlir import ir
from mlir.execution_engine import ExecutionEngine


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out, libs = sys.argv[1], sys.argv[2], sys.argv[3:]
    start = time.time()
    with ir.Context():
        with open(src) as f:
            module = ir.Module.parse(f.read())
        engine = ExecutionEngine(module, opt_level=2, shared_libs=libs)
        engine.dump_to_object_file(out)
    print(f"wrote {out} in {time.time() - start:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
