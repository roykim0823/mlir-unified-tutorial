# Chapter 31: Targeting an NVIDIA GPU through NVVM

> Reworked from ML-compiler-exercise chapter 6 for the unified repo, with
> the GPU work model, the `square` example and the scf.parallel route from
> mlir-ir-lowering chapter 8 (itself after Stephen Diehl's *MLIR Part 8 —
> GPU Compilation with MLIR*), and Diehl's Python GPU pipeline from his
> [gpu-offload](https://github.com/sdiehl/gpu-offload) repo.

Chapter 21 built a GPU *programming model* — tiles, layouts, a SIMT
lowering — but stopped at IR. This chapter drives upstream's actual NVIDIA
backend: the same `sample_model_linalg.mlir` from Chapter 27 gets its loops
mapped onto GPU blocks and threads, outlined into kernels, lowered through
the **nvvm** dialect (NVIDIA's LLVM-IR flavor), and compiled by LLVM's
NVPTX backend into **PTX assembly** — all verifiable on any machine with
this repo's wheel, no GPU required. The final hop (PTX → cubin → running
binary) needs NVIDIA hardware and the CUDA toolkit; it is documented and
scripted here, but — honesty first — **not verified in this port**, which
was done on an Apple-silicon machine.

## What you will learn

- The loop-to-GPU story: `convert-affine-for-to-gpu` (outer loop → blocks,
  inner loop → threads) and `gpu-kernel-outlining`
- Reading `gpu.launch_func` / `gpu.module` / `gpu.binary` — the host/device
  split as IR
- The `gpu-lower-to-nvvm-pipeline` mega-pass and its `cubin-format` knob:
  `isa` (PTX text, toolkit-free) vs `bin` (real cubin, needs `ptxas`)
- What NVVM IR and PTX actually look like for a kernel you know
- The runtime side: `mgpu*` wrappers, `libmlir_cuda_runtime`, and exactly
  what a CUDA machine needs to run the result
- How the GPU sees the work (grid → blocks → threads) and how a CUDA
  kernel like `square.cu` lines up with the `gpu` and `nvvm` dialects
- The *other* loop-to-GPU route: `scf.parallel` → `gpu-map-parallel-loops`
  → `convert-parallel-loops-to-gpu` — the same loop Chapter 7 handed to
  OpenMP
- Tile size as the tuning knob: what `--affine-loop-tile` /
  `--scf-parallel-loop-tiling` change about blocks, threads, and work per
  thread
- The pipeline on real models (MNIST, CNN, ResNet-18) and a Python
  "check your CUDA toolchain" example

## Prerequisites

- [Chapter 21](../../part5-structured-and-parallel/ch21-op-interfaces-gpu-tiles/README.md)
  for the SIMT mental model (blocks/threads, why layouts matter).
- [Chapter 7](../../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md)
  for `scf.parallel` — section 6 sends the very same op to the GPU.
- [Chapter 28](../ch28-cpu-pipeline-and-abi/README.md) — this chapter
  reuses its bufferization stage and its caller.
- Wheel tools on PATH (`mlir-opt` here is the wheel's, not
  `tutorial-opt` — every pass involved is upstream).

Script: [`src/sample/gpu/run_mlir_pipeline.sh`](../src/sample/gpu/run_mlir_pipeline.sh)
(plus `src/{mnist,cnn,resnet18}/gpu/` for the real models) and
[`examples/`](examples/) for the `square` kernel and the Python pipeline.
Verified by [`test/part6/sample_gpu_ptx.mlir`](../../test/part6/sample_gpu_ptx.mlir),
[`ch31_gpu_tiling.mlir`](../../test/part6/ch31_gpu_tiling.mlir),
[`ch31_square_gpu.mlir`](../../test/part6/ch31_square_gpu.mlir) and
[`ch31_mnist_gpu_ptx.mlir`](../../test/part6/ch31_mnist_gpu_ptx.mlir)
(run them with `lit build/test -v --filter ch31_`).

## 0. How the GPU sees the work — and the same kernel in CUDA

A GPU runs a **grid** of **thread blocks**, each block a group of
**threads**. You launch a *kernel* across this grid
(`kernel<<<blocks, threads>>>(...)` in CUDA), and every thread runs the same
code on different data, locating its element from its coordinates:

```text
   grid ───────────────────────────────────────────────
   ┌── block 0 ──┐ ┌── block 1 ──┐ ┌── block 2 ──┐ ...
   │ t0 t1 … t255│ │ t0 t1 … t255│ │ t0 t1 … t255│
   └─────────────┘ └─────────────┘ └─────────────┘
       each thread computes ONE element:
       tid = blockDim.x · blockIdx.x + threadIdx.x
             └ block size ┘ └ which block ┘ └ within block ┘
```

That index arithmetic — built from `blockIdx`, `blockDim`, `threadIdx` — is
the heart of GPU programming. [`examples/square.cu`](examples/square.cu) is
the full CUDA C version of "square every element"; its kernel is exactly
that snippet:

```c
__global__ void square(float *array, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;   // this thread's global index
  if (tid < n)                                       // guard the array bounds
    array[tid] = array[tid] * array[tid];
}
// host: square<<<(N + 255) / 256, 256>>>(a, N);   -> 4 blocks of 256 for N = 1024
```

(`nvcc square.cu -o square && ./square` on a CUDA machine prints
`a[2] = 4.000000 (expected 4)`; `nvcc -ptx square.cu` shows its PTX — not run in this
port.)

NVIDIA's toolchain has its own IR ladder, parallel to LLVM's:

| Stage | What it is |
| --- | --- |
| CUDA C++ | `__global__` kernels, launched with `<<<…>>>` |
| **PTX** | architecture-neutral parallel-thread assembly (stable across GPU generations) |
| **CUBIN** | device-specific machine code for one `sm_XX` architecture |
| **SASS** | the actual GPU shader assembly a CUBIN contains |
| FATBIN | a bundle of PTX + several CUBINs, so one binary supports many GPUs |

GPU architectures are named `sm_<N>`: `sm_75` (Turing), `sm_80` (Ampere,
A100), `sm_90` (Hopper, H100), and so on. A CUBIN runs only on its
generation; PTX is JIT-compiled by the driver to fill the gaps — which is
exactly what happens to this chapter's `cubin-format=isa` output at load
time (`mgpuModuleLoadJIT`, section 4).

MLIR does not make you write the index math. You state parallelism at a
higher level (an affine loop nest in sections 1–3, an `scf.parallel` in
section 6), and passes manufacture the kernel. The **`gpu` dialect** mirrors
the CUDA model — `gpu.module` (a container of device kernels), `gpu.func`
(a kernel), `gpu.launch_func` (the host-side `<<<…>>>`), and
`gpu.block_id` / `gpu.thread_id` / `gpu.block_dim` (`blockIdx` /
`threadIdx` / `blockDim`). Lowering further produces the **NVVM** dialect —
MLIR's mirror of NVIDIA's device IR, where those coordinate ops become reads
of PTX special registers: `nvvm.read.ptx.sreg.ctaid.x` is `blockIdx.x`,
`nvvm.read.ptx.sreg.tid.x` is `threadIdx.x`.

## 1. From loops to `gpu.launch`

The pipeline's front half is Chapter 28 with one twist — we lower linalg to
**affine** loops instead of straight to LLVM, because the GPU mapper works
on affine loop nests:

```sh
mlir-opt sample_model_linalg.mlir \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-affine-loops \
  --canonicalize --cse \
| mlir-opt --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))'
```

`convert-affine-for-to-gpu` takes each perfectly-nested affine loop band
and rewrites it as a `gpu.launch`: by default the outermost loop becomes
the **block** index and the next one the **thread** index — a grid-stride
mapping of the iteration space onto the SIMT hierarchy Chapter 21 drew
diagrams of. (`identity-layout-map` at the function boundary keeps the
memrefs plain — the GPU path is much less forgiving of `strided<[?, ?]>`
arguments than the CPU path was.)

**A drift note, corrected.** The source repo tiled first
(`--affine-loop-tile="tile-size=4"`) and mapped the tiled nest. An earlier
version of this port concluded that LLVM 24's `convert-affine-for-to-gpu`
rejects tiled loops (`error: expected perfectly nested loops in the body`)
and dropped tiling. Re-checking showed the culprit is a different pass: the
source script also asked for `affine-loop-invariant-code-motion`, but passed
it as a *second* `--pass-pipeline` flag to the same `mlir-opt` — and
`mlir-opt` keeps only the last one, so LICM never actually ran there. Run
LICM for real and it hoists inner nests out of trip-count-1 loops, leaving
an empty `affine.for` behind, which is what the mapper rejects. Tiling by
itself maps fine. [`ch31_gpu_tiling.mlir`](../../test/part6/ch31_gpu_tiling.mlir)
pins all three facts down. So `run_mlir_pipeline.sh` tiles again — by 4 for
the sample model, the source's value — without LICM. `TILE=0` turns tiling
off and gives you the untiled pipeline shown above. Section 5 covers what
the tile size does.

## 2. Outlining: the host/device split becomes IR

```sh
... | mlir-opt --gpu-kernel-outlining --lower-affine
```

`gpu-kernel-outlining` hoists every `gpu.launch` region into a proper
kernel function inside a `gpu.module`, leaving a call-site behind. The
sample model's six loop nests (add, transpose, fill, matmul, bias,
clamp) become **six kernels**:

```mlir
func.func @sample_model(...) {
  ...
  gpu.launch_func @sample_model_kernel::@sample_model_kernel
      blocks in (%10, %9, %9) threads in (%8, %9, %9)
      args(%13 : i64, %13 : i64, %arg0 : !llvm.ptr, ...)
  ...
}
gpu.module @sample_model_kernel {
  gpu.func @sample_model_kernel(%arg0: index, %arg1: index,
      %arg2: memref<3x4xf32>, %arg3: memref<3x4xf32>,
      %arg4: memref<3x4xf32>) kernel {
    ...
  }
}
```

One IR file now contains two programs: the host function (launches) and the
device modules (kernels). Everything downstream treats them differently —
which is the whole point of the `gpu` dialect.

## 3. Through NVVM to PTX — on any machine

The back half:

```sh
... | mlir-opt \
  --gpu-decompose-memrefs --expand-strided-metadata --normalize-memrefs \
  --convert-index-to-llvm --arith-expand --memref-expand \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_90 cubin-features=+ptx80 opt-level=3 cubin-format=isa" \
  --convert-nvvm-to-llvm --reconcile-unrealized-casts \
  --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true'
```

`gpu-lower-to-nvvm-pipeline` is upstream's packaged descent for the device
side: kernel bodies go to the **nvvm** dialect — where Chapter 21's
`gpu.thread_id`-style ops become reads of PTX special registers —

```mlir
%1 = "nvvm.read.ptx.sreg.ctaid.x"() : () -> i32     // block index
%3 = "nvvm.read.ptx.sreg.tid.x"() : () -> i32       // thread index
...
%13 = "llvm.fadd"(%10, %12) : (f32, f32) -> f32
```

— and then **out of MLIR entirely**: the pipeline hands each kernel to
LLVM's NVPTX backend and stores the result in a `gpu.binary` op. The
`cubin-format` knob decides what "the result" is:

- **`isa`** — PTX assembly text. This is pure LLVM codegen, needs no CUDA
  toolkit, and works on this repo's wheel on any OS. The lit test pins it.
- **`bin`** (default) — a real cubin, assembled by `ptxas`. Without the
  CUDA toolkit this fails with
  `error: Couldn't find the 'ptxas' binary` — on a non-NVIDIA machine
  that error is your signal to switch to `isa`.

Inside the emitted `gpu.binary` sits recognizable PTX:

```
// Generated by LLVM NVPTX Back-End
.version 8.0
.target sm_90
.address_size 64

.visible .entry sample_model_kernel(
    .param .u64 sample_model_kernel_param_0,
    .param .u64 .ptr .align 1 sample_model_kernel_param_2,
    ...
```

Run it yourself: `./run_mlir_pipeline.sh` (defaults to `isa`) in
[`src/sample/gpu/`](../src/sample/gpu/), then grep `sample_nvvm.mlir` for
`.visible .entry`. Six kernels, six PTX modules, generated on a laptop
with no GPU. Meanwhile `--gpu-to-llvm` rewrote the host side's
`gpu.launch_func` into calls to `mgpu*` runtime wrappers
(`mgpuLaunchKernel`, `mgpuStreamCreate`, ...) — plain LLVM calls into a
runtime library, morally identical to Chapter 29's `cblas_sgemm`
declaration trick.

## 4. Running it — NVIDIA hardware from here

On a Linux box with CUDA and a GPU, the remaining steps are Chapter 28
verbatim plus two libraries
([`src/sample/gpu/compile.sh`](../src/sample/gpu/compile.sh)):

```sh
export CUDA_HOME=/usr/local/cuda             # your toolkit (source repo: an RWTH /cvmfs path)
export MLIR_LIB=/path/to/llvm-project/build/lib   # CUDA-runner-enabled MLIR build
mkdir -p /tmp/sample-gpu && cd /tmp/sample-gpu
sh $REPO/part6-end-to-end-ml/src/sample/gpu/run_mlir_pipeline.sh bin   # ptxas now required; writes sample.ll
sh $REPO/part6-end-to-end-ml/src/sample/gpu/compile.sh                 # llc/clang + link -> ./sample_gpu
./sample_gpu                       # same caller, same expected numbers
```

`compile.sh` boils down to:

```sh
llc -filetype=obj -O3 -relocation-model=pic sample.ll -o sample.o
c++ -c ../sample_call.cpp -o sample_call.o
c++ sample_call.o sample.o -o sample_gpu \
  -L"$MLIR_LIB" -lmlir_c_runner_utils -lmlir_runner_utils -lmlir_cuda_runtime \
  -L"$CUDA_HOME/lib64" -lcuda -lcudart -Wl,-rpath,...
```

`benchmark.sh` next to it builds `bench_gpu` from
`sample_call_benchmark.cpp` (10 warm-up calls, then the mean of 100 timed
calls) the same way. To watch the GPU while it runs:
`watch -n 0.5 nvidia-smi`.

The `mgpu*` wrappers live in **`libmlir_cuda_runtime`** — which the wheel
does *not* ship (it must be compiled against a CUDA install). You need an
MLIR build with `-DMLIR_ENABLE_CUDA_RUNNER=ON`; the
[source-build appendix](../../docs/appendix-source-build.md) covers the
setup, and the source repo's cluster notes list the CUDA-enabled cmake
cache. The C++ caller is unchanged from Chapter 28 — MLIR's generated host
code does every kernel launch behind the same `_mlir_ciface_` signature.
**Where does the data live?** Look at the translated host IR
(`mnist.ll` / `sample.ll`): the only runtime calls are
`mgpuModuleLoadJIT`/`mgpuModuleGetFunction` (run from a global constructor,
before `main`), `mgpuStreamCreate`, `mgpuLaunchKernel`,
`mgpuStreamSynchronize` and their teardown — **no `mgpuMemAlloc`, no
`mgpuMemcpy`**. Intermediate buffers come from plain `malloc`, and the
kernels receive the caller's host pointers directly (bare pointers, no
`gpu.host_register`). That only works on systems where the GPU can
dereference ordinary pageable host memory — e.g. Linux with HMM
(heterogeneous memory management) on a recent driver, or Grace Hopper's
coherent memory. The source repo's H100 numbers suggest its cluster was one
of these. On a GPU that cannot do that, expect `CUDA_ERROR_ILLEGAL_ADDRESS`.
This is our reading of the IR, **not verified on hardware**. The `square`
example in section 6 shows the explicit alternative
(`gpu.host_register`). **Status of this port:** everything through PTX
is machine-verified (lit test `sample_gpu_ptx.mlir`); the
cubin/execution steps above follow the source repo and upstream docs but
awaits a CUDA machine — if you run one, expect Chapter 28's exact numbers,
and please report drift.

## 5. Tile size: the tuning knob

The source repo's one piece of performance advice: **tile size matters**.
Smaller models want smaller tiles, bigger ones larger tiles; experiment and
compare. Its scripts used 4 (sample), 16 (MNIST) and 64 (CNN, marked "not
tuned"), with the note *"Play with tile size for performance (16, 32, 64,
128, 256, ...)"*. Here is what the number actually controls on the affine
route. Take MNIST's first nest (transposing the 512×784 weight) tiled by 16
(simplified from `mnist_2_gpu.mlir`):

```mlir
%7 = arith.ceildivsi %c784, %c16      // 49 tiles along dim 0
%9 = arith.ceildivsi %c512, %c16      // 32 tiles along dim 1
gpu.launch blocks(%bx, ...) in (%7, %c1, %c1) threads(%tx, ...) in (%9, %c1, %c1) {
  %i0 = arith.muli %c16, %bx          // this block's tile
  %j0 = arith.muli %c16, %tx          // this thread's tile
  affine.for %i = #map(%i0) to #map1(%i0) {      // 16 x 16 elements,
    affine.for %j = #map(%j0) to #map1(%j0) {    // sequentially, in ONE thread
      ...
```

The mapper puts the first **tile** loop on blocks and the second on
threads. The intra-tile loops stay inside the kernel as plain sequential
loops. So the tile size T trades parallelism for work per thread:
(784/T)·(512/T) threads, each doing T² elements. T=1 (or `TILE=0`, no
tiling) is maximal parallelism with one element per thread. Large T means
few, fat threads. Blocks of 32 threads (for T=16 here) are also well below
what a GPU wants per block (multiples of the 32-wide warp, typically
128–256). That is why tuning this one number moves performance so much,
and why it is a crude knob: it ties block count, threads per block and
work per thread together. Decoupling them is what
[Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)'s
transform dialect is for (tile, then map tile loops to blocks and threads
explicitly).

Try it: `TILE=4`, `TILE=16`, `TILE=64` on
[`src/mnist/gpu/run_mlir_pipeline.sh`](../src/mnist/gpu/run_mlir_pipeline.sh)
and compare the `gpu.launch` sizes in `mnist_2_gpu.mlir`. Then, on a CUDA
box, compare `bench_gpu` times. The source's measured GPU timings (RWTH
H100, untuned tiles, seconds per inference): sample 0.001746, mnist
0.005568, cnn 0.035221, resnet18 0.619498, versus 0.00000068546 /
0.000853 / 0.001693 / 6.889084 for its CPU pipeline. Tiny models lose to
the launch overhead, and ResNet-18 wins by 11×. These are not reproduced
here.

## 6. The other route: `scf.parallel` → GPU

Sections 1–3 reached the GPU from **affine** loops. The second upstream
route starts from `scf.parallel`, the "these iterations are independent" op
that [Chapter 7](../../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md)
handed to OpenMP. Hand the *same* loop to different passes and it becomes
a GPU kernel. A GPU kernel is, conceptually, one iteration of a parallel
loop run on one thread. [`examples/square.mlir`](examples/square.mlir) is
`square.cu` written that way, with no index math at all:

```mlir
func.func @square(%a: memref<1024xf32>) {
  %c0    = arith.constant 0    : index
  %c1    = arith.constant 1    : index
  %c1024 = arith.constant 1024 : index
  scf.parallel (%i) = (%c0) to (%c1024) step (%c1) {
    %v = memref.load %a[%i] : memref<1024xf32>
    %t = arith.mulf %v, %v : f32        // square this element
    memref.store %t, %a[%i] : memref<1024xf32>
    scf.reduce
  }
  return
}
```

[`examples/build.sh`](examples/build.sh) lowers it in stages, saving each
to `examples/build/`:

**Stage 1 — `scf.parallel` → a GPU kernel + host launch.**

```sh
mlir-opt square.mlir \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining
```

`gpu-map-parallel-loops` only *annotates*: it attaches a mapping attribute
saying the outermost `scf.parallel` goes to the grid (blocks) and a nested
one to threads. `convert-parallel-loops-to-gpu` then rewrites the annotated
loop into a `gpu.launch` and synthesizes the index math. Outlining moves
the body into a `gpu.func`:

```mlir
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<1024xf32>) {
    ...
    %0 = affine.apply #map(%c1024)[%c0, %c1]
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%0, %c1_0, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<1024xf32>)
    return
  }
  gpu.module @square_kernel {
    gpu.func @square_kernel(%arg0: index, %arg1: index, %arg2: memref<1024xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      ...   (thread_id / grid_dim / block_dim x,y,z - unused here)
      %0 = affine.apply #map1(%block_id_x)[%arg0, %arg1]     // i = blockIdx.x * step + lb
      %1 = memref.load %arg2[%0] : memref<1024xf32>
      %2 = arith.mulf %1, %1 : f32
      memref.store %2, %arg2[%0] : memref<1024xf32>
      gpu.return
    }
  }
}
```

The loop is gone. Its body is a kernel indexed by `gpu.block_id`, launched
by `gpu.launch_func`. The `affine.apply #map1` is the compiler's version of
`blockIdx·blockDim + threadIdx`. (The 8_gpu source credits outlining with
synthesizing it; it is actually `convert-parallel-loops-to-gpu`, and
outlining only moves it.) Note the shape: **1024 blocks of 1 thread**. A
single-level `scf.parallel` maps to the grid only.

**Stage 2 — device kernel → NVVM.** `mlir-opt build/square_gpu.mlir
--convert-gpu-to-nvvm` turns `gpu.block_id x` into
`nvvm.read.ptx.sreg.ctaid.x` (literally the `blockIdx` register), the
memref accesses into `llvm.getelementptr` + `llvm.load`/`llvm.store`, and
the kernel into an `llvm.func` tagged `nvvm.kernel`:

```mlir
llvm.func @square_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 1, 1, 1>} {
  ...
  %8 = nvvm.read.ptx.sreg.ctaid.x : i32
  %9 = llvm.sext %8 : i32 to i64
  ...
  %11 = affine.apply #map1(%10)[%1, %0]
  ...
  %14 = llvm.getelementptr inbounds|nuw %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %15 = llvm.load %14 : !llvm.ptr -> f32
  %16 = llvm.fmul %15, %15 : f32
  ...
  llvm.store %16, %18 : f32, !llvm.ptr
  llvm.return
}
```

The memref is passed exploded into its descriptor fields (allocated ptr,
aligned ptr, offset, size, stride). That is the non-bare-pointer
convention, the one ResNet-18's script below uses.

**Stage 3 — pin a target architecture.** `mlir-opt build/square_gpu.mlir
--nvvm-attach-target=chip=sm_80` changes exactly one line. The module now
says which `sm_XX` machine code to emit later:

```mlir
  gpu.module @square_kernel [#nvvm.target<chip = "sm_80">] {
```

**Stage 4 (added in this port) — get real thread blocks.** To launch like
`square.cu`'s `<<<4, 256>>>`, give the mapper a *nested* parallel loop:

```sh
mlir-opt square.mlir --scf-parallel-loop-tiling="parallel-loop-tile-sizes=256" \
  --gpu-map-parallel-loops --convert-parallel-loops-to-gpu --gpu-kernel-outlining --canonicalize
```

```mlir
gpu.launch_func @square_kernel::@square_kernel blocks in (%c4, %c1, %c1) threads in (%c256, %c1, %c1) ...
```

The outer tile loop goes to blocks and the inner one to threads. This is
the scf-route counterpart of section 5's tile size. One trap: running
`--canonicalize` *between* tiling and mapping fuses the two nested loops
back into one 2-D `scf.parallel`, which then lands on blocks x and y, still
with 1 thread per block. Since 1024 divides by 256, no bounds guard (`if
(tid < n)`) is needed. A second trap is a ragged tile size such as 100. By
default the inner loop then gets an `affine.min` upper bound, and
`convert-parallel-loops-to-gpu` **silently leaves that loop unconverted**:
no `gpu.launch`, and no error either. With
`--scf-parallel-loop-tiling="parallel-loop-tile-sizes=100 no-min-max-bounds=true"`,
the inner bound stays the constant 100, and the leftover iterations are
masked by an `scf.if (%i < %c1024)`. You get 11 blocks × 100 threads,
plus exactly `square.cu`'s `if (tid < n)`.

**Stage 5 — PTX.** `--lower-affine --gpu-lower-to-nvvm-pipeline=
"cubin-chip=sm_80 cubin-format=isa"` produces a `gpu.binary` whose PTX
reads (abridged to the lines the test pins):

```ptx
.target sm_80
.visible .entry square_kernel(
  ...
  mov.u32 %r1, %ctaid.x;
  ...
  mov.u32 %r2, %tid.x;
  ...
  mul.rn.f32 ...
```

That is `square.cu` again, compiled from a loop.

All five stages are pinned by
[`ch31_square_gpu.mlir`](../../test/part6/ch31_square_gpu.mlir).

**The scf route on a real model.** Upstream's
`--convert-linalg-to-parallel-loops` gives every linalg op an
`scf.parallel` nest over its parallel dimensions (reductions stay `scf.for`
inside). The source repo's ResNet-18 GPU script uses exactly this route,
after `--linalg-fuse-elementwise-ops` to cut the kernel count. On the MNIST
snapshot it produces **6** kernels where the affine route produces 13
(`PAR` prefix in `ch31_mnist_gpu_ptx.mlir`).

## 7. The real models: MNIST, CNN, ResNet-18

The source repo ran the GPU pipeline on four models (all but FLAN-T5, which
it marks "not yet working"). Ported scripts, one directory per model:

| Model | Script dir | Route | Tile | Status here |
|---|---|---|---|---|
| sample | [`src/sample/gpu/`](../src/sample/gpu/) | affine | 4 | PTX verified (`sample_gpu_ptx.mlir`, `ch31_gpu_tiling.mlir`) |
| MNIST MLP | [`src/mnist/gpu/`](../src/mnist/gpu/) | affine | 16 | PTX verified (`ch31_mnist_gpu_ptx.mlir`): 13 kernels |
| small CNN | [`src/cnn/gpu/`](../src/cnn/gpu/) | affine | 64 ("not tuned") | **pending** the `cnn_model_linalg.mlir` snapshot (ch30) |
| ResNet-18 | [`src/resnet18/gpu/`](../src/resnet18/gpu/) | scf.parallel, fused, non-bare pointers | — | **pending** the `resnet18_model_linalg.mlir` snapshot with parameters (ch30) |

Each directory has `run_mlir_pipeline.sh` (to PTX/cubin + `.ll`, any
machine; `TILE`, `CHIP`, `PTXV`, `IN` knobs), `compile.sh` and
`benchmark.sh` (CUDA box), and the C++ callers. The CUDA paths that pointed
at the RWTH cluster (`/cvmfs/.../CUDA/12.6.3/lib64`) are now `$CUDA_HOME`.
The CNN and ResNet-18 callers live in `gpu/` because the source had GPU
copies. The CNN caller runs a batch of 4. The ResNet-18 benchmark times a
single call, as in the source ("benchmark: one call only"). The pipeline
scripts write one file per stage (`*_1_loops`, `*_2_gpu`, `*_3_outlined`,
`*_nvvm.mlir`) instead of piping. A pipe let a failing middle stage hand an
empty module to the next stage and still "succeed".

Three caveats the porting uncovered, all **unverified on hardware**:

- **ResNet-18 + tiling.** The source author reports that tiling ResNet-18
  ("Try `affine-loop-tile{tile-size=16}` … for performance") currently
  yields `CUDA_ERROR_ILLEGAL_ADDRESS`, which is why that script has no tile
  step.
- **libdevice.** A model whose kernels use `math` ops (the CNN's
  `log_softmax` has an `exp`) lowers them to calls like `__nv_expf`. When
  MLIR finds a CUDA toolkit, it links NVIDIA's libdevice into the kernel
  during serialization. On the wheel without a toolkit, the PTX keeps them
  as unresolved `.extern .func __nv_expf`. We saw this on a hand-written
  CNN-shaped smoke test. The CUDA driver will refuse such PTX at load time,
  so on a CUDA box make sure `CUDA_HOME` is set when you run the pipeline.
  The ReLU-only sample and MNIST models are not affected.
- **Host memory**, as discussed at the end of section 4.

The ported scripts were exercised to PTX on the MNIST snapshot (both
routes) and on a hand-written conv/pool/softmax module standing in for the
CNN. The `.ll` they emit cross-compiles to Linux x86-64 objects that define
`_mlir_ciface_<model>` and reference only `mgpu*` symbols.

## 8. Check your CUDA toolchain first: the Python pipeline

Before debugging a whole model on a new CUDA machine, run the smallest
possible end-to-end example. [`examples/python-gpu/`](examples/python-gpu/)
is adapted from Stephen Diehl's
[GPU compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)
([source: sdiehl/gpu-offload](https://github.com/sdiehl/gpu-offload/tree/main)),
by way of ML-compiler-exercise. It compiles a 10×10 `linalg.square` to a
kernel, launches it through the CUDA driver API from Python, and checks the
result against NumPy:

- [`compile.py`](examples/python-gpu/compile.py): the MLIR Python
  bindings run one `PassManager` (bufferize → affine loops → LICM →
  `convert-affine-for-to-gpu` → outline → `convert-gpu-to-nvvm` →
  `nvvm-attach-target` → `gpu-to-llvm`), extract the `gpu.module`, and
  turn it into PTX via `mlir-translate --mlir-to-llvmir` + `llc
  -march=nvptx64 -mcpu=sm_90`. (Here LICM runs for real and is harmless:
  the 10×10 nest is untiled, so there is nothing to hoist.)
- [`run.py`](examples/python-gpu/run.py): a small `CudaContext` /
  `CudaArray` wrapper over `cuda-python` (`cuInit`, `cuMemAlloc`,
  `cuMemcpyHtoD`, `cuModuleLoadData` of the PTX string, `cuLaunchKernel`).
  Unlike the model pipelines, it copies data to and from device memory
  explicitly.
- [`example_full.py`](examples/python-gpu/example_full.py): ties them
  together. It launches `square_kernel` with the 4 parameters the PTX
  declares (two index offsets, two pointers) as 10 blocks of 10 threads.
  The mapper put the row loop on blocks and the column loop on threads.

```sh
pip install "cuda-python==12.6.0"    # match your CUDA 12.x (see the appendix)
cd part6-end-to-end-ml/ch31-gpu-nvvm/examples/python-gpu
python example_full.py               # on a CUDA box: prints the squares and "Success!"
python example_full.py --compile-only   # anywhere: stops after printing the PTX
```

Port additions: without `cuda-python` (or with `--compile-only`) the
example stops after printing the PTX. Without `llc` on PATH (this repo's
wheel ships none), `compile.py` gets the same PTX from MLIR itself via
`gpu-module-to-binary{format=isa}`. **Verified here:** the compile half, by
both paths (Homebrew `llc` 20, and the no-`llc` fallback). The `.entry
square_kernel` with 4 parameters comes out as the launch code expects.
**Not verified:** the launch and the NumPy check, which need a GPU.

## 9. On a CUDA box: the `square` recipe

[`examples/run_on_gpu.sh`](examples/run_on_gpu.sh) continues past
`build.sh`. On a Linux machine with an NVIDIA GPU and a CUDA-runner MLIR
build ([appendix](../../docs/appendix-source-build.md), or Diehl's
`ghcr.io/sdiehl/docker-mlir-cuda:main` image), it lowers to NVVM, serializes
the kernel with `gpu-module-to-binary`, lowers the host side to LLVM, and
JIT-runs the result with `mlir-runner` against `libmlir_cuda_runtime`:

```sh
LLVM=/path/to/llvm-project/build ./run_on_gpu.sh sm_80   # your GPU's sm_XX
# expected: [0,  1,  4,  9,  16,  25,  36,  49]
```

It runs [`square_runnable.mlir`](examples/square_runnable.mlir): `square.mlir`
plus a `@main` that allocates the array, `gpu.host_register`s it (pins the
host buffer and maps it for the GPU), calls `@square`, and prints the first
8 elements. The original 8_gpu script pointed `mlir-runner` at
`square.mlir`, whose only function takes a memref and so cannot be an entry
point. It also failed before reaching the GPU at all: without
`--lower-affine`, the kernel's `affine.apply` broke serialization
(`LLVM Translation failed for operation: builtin.unrealized_conversion_cast`).
Both are fixed. `NO_RUN=1 FORMAT=isa ./run_on_gpu.sh` runs everything but
`mlir-runner` on any machine, and the lit test does the same. The actual
run is **unverified**.

## Where to go next

This closes the capstone: one PyTorch model, two backends, every stage
inspectable. Two codas remain. If you want to see how a *production*
MLIR-based ML compiler arranges these same moves — dispatch regions,
streams, a hardware abstraction layer, dozens of targets — read the
[IREE appendix](../appendix-iree.md). And if the GPU thread left you
wanting real execution work, the declared gap in this repo (issue: run
ch21/ch31 kernels on actual hardware) is the natural contribution.

## Exercises

1. Read one of the six PTX kernels end to end and match each instruction
   back to the linalg op it came from. Which kernel is the matmul, and how
   can you tell from the loop structure inside it?
2. The mapper put loop 0 on blocks and loop 1 on threads. For the 3×5
   clamp kernel of the untiled pipeline (`TILE=0`), how many blocks and threads does the launch request?
   (Read the `gpu.launch_func` operands back to their constants.) What is
   wasteful about this for a 3×5 problem, and what would Chapter 21's
   layout machinery do differently?
3. Swap `convert-affine-for-to-gpu` for the scf path:
   `--convert-linalg-to-parallel-loops --gpu-map-parallel-loops
   --convert-parallel-loops-to-gpu`. Compare kernel count and shapes
   against the affine path.
4. `cubin-chip=sm_90` targets Hopper. Retarget to an older GPU
   (`sm_75`, `+ptx63`) and diff the PTX. What changed besides the header?
5. (Hardware required.) Run `compile.sh` on a CUDA machine and verify the
   output matches Chapter 28's. Then profile with `nsys` / `nvprof`: how
   much of the runtime is the six kernel launches vs the per-launch stream
   create/synchronize/destroy? Section 4 says there are no memcpys at all.
   Does the profiler agree, and where do page migrations show up instead?
6. Sweep `TILE` over 1, 2, 4, 8, 16, 32, 64 on the MNIST pipeline and
   tabulate blocks × threads × elements-per-thread for the matmul kernel
   from `mnist_2_gpu.mlir`. Which value gives full 32-thread warps? On a
   CUDA box, add `bench_gpu` times to the table.
7. In `examples/`, insert `--canonicalize` between
   `--scf-parallel-loop-tiling` and `--gpu-map-parallel-loops` (stage 4) and
   explain the launch shape you get. Then tile by 100 instead of 256, with
   and without `no-min-max-bounds=true`: in which case is there no
   `gpu.launch` at all, and where does the `if (tid < n)` guard of
   `square.cu` reappear in the other?
8. Write a `square_kernel` launch for `examples/python-gpu` that uses the
   PTX from `build/square_ptx.mlir` (stage 5) instead of the
   `linalg.square` one. What parameters does that kernel take, and why
   more than four?
