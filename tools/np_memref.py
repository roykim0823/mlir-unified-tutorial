"""NumPy <-> MLIR memref bridge over plain ctypes (no MLIR Python bindings).

A ranked memref<d0 x ... x dN-1 x T> lowered by --finalize-memref-to-llvm is
the struct

    { T* allocated, T* aligned, i64 offset, i64 sizes[N], i64 strides[N] }

This module builds that struct with ctypes so a NumPy array can be handed to a
compiled kernel without copying. Chapter 4 explains the layout field by field;
every later chapter that runs kernels from Python imports it:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "<relative path to repo>", "tools"))
    from np_memref import numpy_to_memref, memref_descriptor_type

`offset`, `sizes` and `strides` are in ELEMENTS, not bytes.

Calling conventions: a function compiled with `llvm.emit_c_interface` (or
--llvm-request-c-wrappers) gets a `_mlir_ciface_<name>` wrapper that takes one
*pointer* per memref descriptor -- pass `ctypes.byref(desc)`. Without the
wrapper, each descriptor is expanded into its 3 + 2*rank scalar fields; use
`expand(desc)` to get them in order.
"""

import ctypes
from ctypes import Structure, c_longlong, c_void_p

import numpy as np

_TYPES = {}


def memref_descriptor_type(rank):
    """Return (and cache) the ctypes Structure for a rank-`rank` memref."""
    if rank not in _TYPES:
        _TYPES[rank] = type(
            f"MemRef{rank}D",
            (Structure,),
            {
                "_fields_": [
                    ("allocated", c_void_p),  # base pointer (the one free() takes)
                    ("aligned", c_void_p),  # aligned data pointer
                    ("offset", c_longlong),  # in elements
                    ("sizes", c_longlong * rank),
                    ("strides", c_longlong * rank),
                ]
            },
        )
    return _TYPES[rank]


def numpy_to_memref(arr):
    """Wrap a NumPy array in a memref descriptor of matching rank (no copy).

    The descriptor points straight at NumPy's buffer, so writes made by the
    kernel are visible in `arr`. Non-contiguous arrays are rejected rather
    than silently copied, because the kernel would then write into the copy.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("pass np.ascontiguousarray(arr): kernel writes must reach arr")
    desc = memref_descriptor_type(arr.ndim)()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    for i in range(arr.ndim):
        desc.sizes[i] = arr.shape[i]
        desc.strides[i] = arr.strides[i] // arr.itemsize  # bytes -> elements
    return desc


def memref_to_numpy(desc, dtype=np.float32):
    """View the memory a descriptor points at as a NumPy array (no copy)."""
    rank = len(desc.sizes)
    shape = tuple(desc.sizes[i] for i in range(rank))
    strides = tuple(desc.strides[i] * np.dtype(dtype).itemsize for i in range(rank))
    nbytes = (max((s - 1) * st for s, st in zip(shape, strides)) if rank else 0) + np.dtype(dtype).itemsize
    base = (ctypes.c_char * (nbytes + desc.offset * np.dtype(dtype).itemsize)).from_address(desc.aligned)
    flat = np.frombuffer(base, dtype=dtype, offset=desc.offset * np.dtype(dtype).itemsize)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=strides)


def expand(desc):
    """The descriptor's fields as the flat argument list used without C wrappers."""
    rank = len(desc.sizes)
    return [desc.allocated, desc.aligned, desc.offset,
            *[desc.sizes[i] for i in range(rank)],
            *[desc.strides[i] for i in range(rank)]]
