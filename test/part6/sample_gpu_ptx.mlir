// The ch31 GPU pipeline, verified as far as a machine without an NVIDIA GPU
// can go: the sample model's loops are mapped to GPU blocks/threads, kernels
// are outlined, lowered through NVVM, and compiled to PTX assembly by LLVM's
// NVPTX backend (cubin-format=isa needs no CUDA toolkit; assembling to a
// real cubin would need ptxas).
//
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/src/sample/sample_model_linalg.mlir --convert-tensor-to-linalg --linalg-generalize-named-ops --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation-pipeline --convert-bufferization-to-memref --llvm-request-c-wrappers --convert-linalg-to-affine-loops --canonicalize --cse > %t.tiled.mlir
// RUN: mlir-opt %t.tiled.mlir --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' > %t.gpu.mlir
// RUN: mlir-opt %t.gpu.mlir --gpu-kernel-outlining --lower-affine > %t.outlined.mlir
// RUN: mlir-opt %t.outlined.mlir --gpu-decompose-memrefs --expand-strided-metadata --normalize-memrefs --convert-index-to-llvm --arith-expand --memref-expand --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 cubin-features=+ptx80 opt-level=3 cubin-format=isa" --convert-nvvm-to-llvm --reconcile-unrealized-casts --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true' | FileCheck %s

// Six loop nests -> six kernels, each embedded as PTX in a gpu.binary.
// CHECK-COUNT-6: gpu.binary
// The PTX itself is quoted inside the binary op:
// CHECK: .visible .entry sample_model_kernel
