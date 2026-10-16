"""Load build/libsimple.dylib with ctypes and call its `main` function."""

import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
lib_name = "libsimple.dylib" if sys.platform == "darwin" else "libsimple.so"

# Load the shared library we just built into this process (dlopen).
module = ctypes.CDLL(os.path.join(HERE, "build", lib_name))

# Declare the C signature of `main` so ctypes marshals values correctly.
module.main.argtypes = []            # main takes no arguments
module.main.restype = ctypes.c_int   # main returns a C int (i32)

# Call the native function and print its return value.
print(module.main())
