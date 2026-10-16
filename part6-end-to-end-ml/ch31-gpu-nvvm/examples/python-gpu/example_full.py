"""End-to-end "check your CUDA toolchain" example: linalg.square on a 10x10
matrix, compiled MLIR -> PTX and launched through the CUDA driver API.

  pip install cuda-python   # match your CUDA 12.x, e.g. cuda-python==12.6.0
  python example_full.py

Port addition: without cuda-python (or with --compile-only) it stops after
printing the PTX, which is the part verified in this port on a machine with no
NVIDIA GPU. The kernel launch and the numeric check are NOT verified here.
"""
# Adapted from Stephen Diehl's gpu-offload (https://github.com/sdiehl/gpu-offload),
# companion code to "GPU Compilation with MLIR"
# (https://www.stephendiehl.com/posts/mlir_gpu/), via ML-compiler-exercise
# src/python/gpu/. Ported into the unified tutorial for ch31.

import sys

import numpy as np

from compile import compile_mlir_to_ptx

try:
    from run import CudaContext
except ImportError:  # cuda-python not installed
    CudaContext = None

# Example MLIR module for a matrix squaring operation
SQUARE_MLIR = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""

# Input data: 10x10 random matrix
size = 10
input_data = np.random.randn(size, size).astype(np.float32)

# Expected output for verification
expected_output = input_data * input_data

# Step 1: Compile MLIR to PTX
print("Compiling MLIR to PTX...")
ptx_code = compile_mlir_to_ptx(SQUARE_MLIR)

if not ptx_code:
    raise RuntimeError("PTX compilation failed.")

if CudaContext is None or "--compile-only" in sys.argv:
    print(ptx_code)
    print("Compile-only: PTX generated; install cuda-python on a CUDA machine to run it.")
    sys.exit(0)

# Step 2: Execute the kernel using the CudaContext
with CudaContext() as ctx:
    print("Running kernel on GPU...")

    # Create device arrays
    d_input = ctx.array(input_data)
    d_output = ctx.array(shape=(size, size), dtype=np.float32)

    # Execute kernel
    # The kernel expects 4 parameters based on the PTX:
    # - param_0 and param_1: scalar offsets (typically 0 for direct access)
    # - param_2 and param_3: pointers to input and output arrays
    ctx.run_kernel(
        ptx_code,
        "square_kernel",
        [0, 0, d_input, d_output],
        grid_dims=(size, 1),     # 10 blocks in x-dimension
        block_dims=(size, 1, 1), # 10 threads in x-dimension
    )

    # Get results back
    d_output.copy_device_to_host()
    result = d_output.host_array

    # Verify results
    print("Verifying results...")
    np.testing.assert_allclose(result, expected_output, rtol=1e-5)
    print(result)
    print("Success! Results verified.")
