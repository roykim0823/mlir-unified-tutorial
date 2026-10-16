// ch31: the MNIST MLP snapshot taken to PTX by both GPU routes, on a machine
// without an NVIDIA GPU (cubin-format=isa needs no CUDA toolkit). Mirrors
// part6-end-to-end-ml/src/mnist/gpu/run_mlir_pipeline.sh (affine route, the
// source repo's tile size 16) and src/resnet18/gpu/run_mlir_pipeline.sh
// (scf.parallel route, run here on the mnist snapshot).
//
// Affine route: tile, map outer tile loop -> blocks / next -> threads, outline.
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/src/mnist/mnist_model_linalg.mlir --convert-tensor-to-linalg --linalg-generalize-named-ops --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation-pipeline --convert-bufferization-to-memref --llvm-request-c-wrappers --convert-linalg-to-affine-loops --affine-loop-tile="tile-size=16" --canonicalize --cse > %t.loops.mlir
// RUN: mlir-opt %t.loops.mlir --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' > %t.gpu.mlir
// RUN: FileCheck %s --check-prefix=MAP < %t.gpu.mlir
// RUN: mlir-opt %t.gpu.mlir --gpu-kernel-outlining --lower-affine > %t.outlined.mlir
// RUN: mlir-opt %t.outlined.mlir --gpu-decompose-memrefs --expand-strided-metadata --normalize-memrefs --convert-index-to-llvm --arith-expand --memref-expand --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 cubin-features=+ptx80 opt-level=3 cubin-format=isa" --convert-nvvm-to-llvm --reconcile-unrealized-casts --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true' > %t.nvvm.mlir
// RUN: FileCheck %s < %t.nvvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.nvvm.mlir | FileCheck %s --check-prefix=HOST
//
// scf.parallel route (fuse elementwise ops first, full memref descriptors).
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/src/mnist/mnist_model_linalg.mlir --convert-tensor-to-linalg --linalg-generalize-named-ops --linalg-fuse-elementwise-ops --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation-pipeline --convert-bufferization-to-memref --llvm-request-c-wrappers --convert-linalg-to-parallel-loops --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --canonicalize --cse | mlir-opt --gpu-kernel-outlining --lower-affine | mlir-opt --gpu-decompose-memrefs --expand-strided-metadata --normalize-memrefs --convert-index-to-llvm --arith-expand --memref-expand --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 cubin-features=+ptx80 opt-level=3 cubin-format=isa" --convert-nvvm-to-llvm --reconcile-unrealized-casts --gpu-to-llvm='use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false' | FileCheck %s --check-prefix=PAR

// The first nest (the 512x784 -> 784x512 weight transpose) tiled by 16: the
// grid is ceil(784/16) blocks of ceil(512/16) threads, and each thread walks
// a 16x16 tile sequentially.
// MAP: arith.ceildivsi
// MAP: gpu.launch blocks
// MAP: affine.for {{.*}} to #map1(
// MAP: affine.for {{.*}} to #map1(
// MAP: memref<512x784xf32>

// 13 loop nests -> 13 kernels, each embedded as PTX text.
// CHECK-COUNT-13: gpu.binary @mnist_model_kernel
// CHECK: .visible .entry mnist_model_kernel

// Host side after translation: kernels are JIT-loaded from PTX and launched
// through the mgpu* wrappers of libmlir_cuda_runtime. No mgpuMemAlloc/Memcpy:
// the kernels receive the caller's host pointers directly.
// HOST-DAG: define void @_mlir_ciface_mnist_model(ptr %0, ptr %1)
// HOST-DAG: call ptr @mgpuModuleLoadJIT
// HOST-DAG: call void @mgpuLaunchKernel
// HOST-NOT: mgpuMemAlloc

// Fusion leaves 6 parallel nests -> 6 kernels.
// PAR-COUNT-6: gpu.binary @mnist_model_kernel
// PAR-NOT: gpu.binary
