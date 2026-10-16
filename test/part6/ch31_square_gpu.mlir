// ch31: the square example (examples/square.mlir, after mlir-ir-lowering
// 8_gpu) - the scf.parallel -> gpu route, stage by stage as build.sh runs it,
// plus the run_on_gpu.sh pipeline up to the point that needs a GPU.
//
// Stage 1: map + convert + outline.
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch31-gpu-nvvm/examples/square.mlir --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --gpu-kernel-outlining > %t.gpu.mlir
// RUN: FileCheck %s --check-prefix=GPU < %t.gpu.mlir
// Stage 2: device side to NVVM.
// RUN: mlir-opt %t.gpu.mlir --convert-gpu-to-nvvm | FileCheck %s --check-prefix=NVVM
// Stage 3: attach a target.
// RUN: mlir-opt %t.gpu.mlir --nvvm-attach-target=chip=sm_80 | FileCheck %s --check-prefix=TARGET
// Stage 4 (port addition): tile 256 first -> square<<<4, 256>>> like square.cu.
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch31-gpu-nvvm/examples/square.mlir --scf-parallel-loop-tiling="parallel-loop-tile-sizes=256" --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --gpu-kernel-outlining --canonicalize > %t.tiled.mlir
// RUN: FileCheck %s --check-prefix=TILED < %t.tiled.mlir
// Ragged tile (100): default min-bounds are silently left unconverted;
// no-min-max-bounds gives 11 x 100 plus an scf.if bounds guard.
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch31-gpu-nvvm/examples/square.mlir --scf-parallel-loop-tiling="parallel-loop-tile-sizes=100" --gpu-map-parallel-loops --convert-parallel-loops-to-gpu | FileCheck %s --check-prefix=RAGGED
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch31-gpu-nvvm/examples/square.mlir --scf-parallel-loop-tiling="parallel-loop-tile-sizes=100 no-min-max-bounds=true" --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --gpu-kernel-outlining --canonicalize | FileCheck %s --check-prefix=GUARD
// Stage 5: to PTX.
// RUN: mlir-opt %t.tiled.mlir --lower-affine --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-format=isa" | FileCheck %s --check-prefix=PTX
// run_on_gpu.sh (fixed) with format=isa; only the mlir-runner call is left out.
// RUN: mlir-opt %project_source_dir/part6-end-to-end-ml/ch31-gpu-nvvm/examples/square_runnable.mlir --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --gpu-kernel-outlining --lower-affine --convert-gpu-to-nvvm --nvvm-attach-target=chip=sm_80 --gpu-to-llvm --gpu-module-to-binary=format=isa --convert-scf-to-cf --expand-strided-metadata --convert-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=RUNNABLE

// One block per element, one thread per block; the index math is synthesized.
// GPU: gpu.launch_func @square_kernel::@square_kernel blocks in ({{.*}}) threads in ({{.*}})
// GPU: gpu.module @square_kernel
// GPU: gpu.func @square_kernel({{.*}}) kernel
// GPU: %block_id_x = gpu.block_id x
// GPU: affine.apply #map1(%block_id_x)
// GPU: arith.mulf

// NVVM: llvm.func @square_kernel({{.*}}) attributes {{{.*}}nvvm.kernel
// NVVM: nvvm.read.ptx.sreg.ctaid.x
// NVVM: llvm.fmul

// TARGET: gpu.module @square_kernel [#nvvm.target<chip = "sm_80">]

// TILED: gpu.launch_func @square_kernel::@square_kernel blocks in (%c4, %c1, %c1) threads in (%c256, %c1, %c1)
// TILED: gpu.block_id x
// TILED: gpu.thread_id x

// RAGGED-NOT: gpu.launch
// RAGGED: affine.min
// RAGGED: scf.parallel
// RAGGED-NOT: gpu.launch

// GUARD: gpu.launch_func @square_kernel::@square_kernel blocks in (%c11, %c1, %c1) threads in (%c100, %c1, %c1)
// GUARD: arith.cmpi ult
// GUARD-NEXT: scf.if

// PTX: gpu.binary @square_kernel
// PTX-SAME: .target sm_80
// PTX-SAME: .visible .entry square_kernel(
// PTX-SAME: %ctaid.x
// PTX-SAME: %tid.x
// PTX-SAME: mul.rn.f32

// RUNNABLE-DAG: define void @main()
// RUNNABLE-DAG: call void @mgpuMemHostRegisterMemRef
// RUNNABLE-DAG: call void @mgpuLaunchKernel
// RUNNABLE-DAG: call void @printMemrefF32
