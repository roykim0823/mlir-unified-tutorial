"""MLIR (linalg on tensors) -> GPU kernel -> PTX, with the MLIR Python bindings.

The pass pipeline and PTX step are Diehl's: mlir-translate to LLVM IR, then
`llc -march=nvptx64`. Port addition: when `llc` is not on PATH (the MLIR wheel
used by this repo ships none), generate_ptx_mlir() produces the same PTX inside
MLIR via gpu-module-to-binary{format=isa} - no CUDA toolkit required.
"""
# Adapted from Stephen Diehl's gpu-offload (https://github.com/sdiehl/gpu-offload),
# companion code to "GPU Compilation with MLIR"
# (https://www.stephendiehl.com/posts/mlir_gpu/), via ML-compiler-exercise
# src/python/gpu/. Ported into the unified tutorial for ch31.

import shutil
import subprocess

from mlir.ir import Context, Module
from mlir.passmanager import PassManager


def compile_mlir_to_ptx(mlir_module_str: str, chip_type="sm_90"):
    """Compiles MLIR module string to PTX code."""
    with Context():
        # Parse the input module
        module = Module.parse(mlir_module_str)

        # Apply GPU compilation pipeline
        module, gpu_module = apply_gpu_pipeline(module, chip_type)

        # Generate PTX from the GPU module
        if shutil.which("llc"):
            ptx = generate_ptx(str(gpu_module), chip_type)
        else:  # port addition: no llc -> let MLIR's NVPTX serializer emit PTX
            ptx = generate_ptx_mlir(module)

    return ptx


def apply_gpu_pipeline(module, chip_type="sm_90"):
    """Applies the GPU compilation pipeline to the MLIR module."""
    pm = PassManager()
    pm.enable_ir_printing(print_after_change=True)
    pm.add("canonicalize")
    pm.add(
        "one-shot-bufferize{ bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map }"
    )
    pm.add("canonicalize")
    pm.add("convert-linalg-to-affine-loops")
    pm.add("func.func(affine-loop-invariant-code-motion)")
    pm.add("func.func(convert-affine-for-to-gpu)")
    pm.add("gpu-kernel-outlining")
    pm.add("lower-affine")
    pm.add("gpu-decompose-memrefs")
    pm.add("expand-strided-metadata")
    pm.add("normalize-memrefs")
    pm.add(
        "gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv })"
    )
    pm.add(f"nvvm-attach-target{{chip={chip_type} features=+ptx80 O=3}}")
    pm.add("convert-nvvm-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.add("gpu-to-llvm { use-bare-pointers-for-host use-bare-pointers-for-kernels }")
    pm.run(module.operation)

    # Extract the GPU module
    gpu_module = extract_gpu_module(module)

    return module, gpu_module


def extract_gpu_module(module: Module) -> Module:
    """Extracts the GPU module from a transformed MLIR module."""
    # Navigate the operation tree to find the GPU module
    # Structure: module -> region[0] -> block[0] -> operations[1] (GPU host-device code)
    # -> region[0] -> block[0] -> operations[0] (GPU module)
    try:
        main_func_op = module.operation.regions[0].blocks[0].operations[1]
        gpu_module_op = main_func_op.regions[0].blocks[0].operations[0]

        # Create a new module from the GPU module operation
        gpu_module = Module.parse(str(gpu_module_op))
        return gpu_module
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to extract GPU module: {e}") from e


def generate_ptx(gpu_module_str, chip_type="sm_90"):
    """Generates PTX from an MLIR GPU module string."""
    # First convert MLIR to LLVM IR
    llvm_ir_result = subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir", "-"],
        input=gpu_module_str,
        capture_output=True,
        text=True,
    )

    if llvm_ir_result.returncode != 0:
        print("Error generating LLVM IR:")
        print(llvm_ir_result.stderr)
        return None

    llvm_ir = llvm_ir_result.stdout

    # Then convert LLVM IR to PTX
    ptx_result = subprocess.run(
        ["llc", "-march=nvptx64", f"-mcpu={chip_type}", "-"],
        input=llvm_ir,
        capture_output=True,
        text=True,
    )

    if ptx_result.returncode != 0:
        print("Error generating PTX:")
        print(ptx_result.stderr)
        return None

    return ptx_result.stdout


def generate_ptx_mlir(module):
    """Port addition: PTX via MLIR's own NVPTX serialization (no llc needed).

    Runs gpu-module-to-binary{format=isa} on the module apply_gpu_pipeline()
    produced (its gpu.module already carries the #nvvm.target from
    nvvm-attach-target) and returns the PTX text stored in the gpu.binary op.
    """
    from mlir.dialects import gpu
    from mlir.passmanager import PassManager

    PassManager.parse("builtin.module(gpu-module-to-binary{format=isa})").run(
        module.operation
    )
    for op in module.body.operations:
        if op.operation.name == "gpu.binary":
            obj = gpu.ObjectAttr(op.attributes["objects"][0])
            return obj.object.decode("utf-8")
    raise RuntimeError("no gpu.binary produced")
