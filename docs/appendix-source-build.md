# Appendix: Building MLIR from Source

Everything in this repo builds against the prebuilt
[MLIR wheel](https://llvm.github.io/eudsl) precisely so you never need
what this appendix describes. (For prebuilt alternatives — Homebrew, apt,
conda, Docker — see [Other Ways to Install MLIR](appendix-installing-mlir.md).) But three situations put you back in
source-build territory:

1. **The CUDA runner.** Chapter 31 stops at PTX on the wheel because
   `libmlir_cuda_runtime` must be compiled against a CUDA install — the
   wheel cannot ship it.
2. **Version pinning or patching.** You need an MLIR older/newer than the
   wheel feed's ~1-month window, or you are debugging/patching upstream
   MLIR itself.
3. **torch-mlir on macOS.** Its wheels are Linux-only (Part 6 sidesteps
   this with a container); a native macOS setup means building torch-mlir
   and its pinned LLVM.

Budget accordingly: an LLVM+MLIR build is **tens of minutes to more than an
hour** on a fast machine, and tens of GB of disk in Debug. This page gives
the recipes and the flags that matter; the authoritative reference is
[mlir.llvm.org/getting_started](https://mlir.llvm.org/getting_started/).

## Plain MLIR (replaces the wheel)

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -GNinja -Bbuild llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON

ninja -C build
```

Flag notes, in the order they will save you pain:

- `LLVM_ENABLE_ASSERTIONS=ON` even in Release — it is what makes
  `--debug-only=...` and the good error messages work (the wheel is built
  this way too, as ch23 exploits).
- `LLVM_TARGETS_TO_BUILD=host` skips a dozen backends you don't need. Add
  `;NVPTX` if GPU work is the point (see below).
- `lld` + `ccache` turn rebuild cycles from minutes into seconds; drop
  those two lines if you don't have them installed.
- For hacking on MLIR itself, add
  `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (debuggable without Debug's disk
  bill) and `-DBUILD_SHARED_LIBS=ON` (much faster relinks).

Point this repo at the result instead of the wheel:

```sh
cmake .. -DCMAKE_PREFIX_PATH=/path/to/llvm-project/build \
         -DLLVM_EXTERNAL_LIT=/path/to/llvm-project/build/bin/llvm-lit
```

## The CUDA runner build (for ch31 execution)

On a Linux machine with an NVIDIA GPU and the CUDA toolkit:

```sh
export CUDA_HOME=/usr/local/cuda        # adapt to your install

cmake -GNinja -Bbuild llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
  -DMLIR_ENABLE_NVPTXCOMPILER=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDACXX=$CUDA_HOME/bin/nvcc \
  -DCUDA_PATH=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON     # only needed for ch31's Python example
```

`MLIR_ENABLE_CUDA_RUNNER=ON` is the line that produces
**`libmlir_cuda_runtime`** — the library holding the `mgpu*` wrappers that
ch31's host code calls. `NVPTX` in the targets list is what lets
`gpu-module-to-binary` (via `ptxas`) produce real cubins. With this build
on PATH/`MLIR_LIB`, ch31's `run_mlir_pipeline.sh bin` + `compile.sh` flow
runs end to end.

The other CUDA flags, which the ML-compiler-exercise GPU build also sets:

- `MLIR_ENABLE_CUDA_CONVERSIONS=ON` builds MLIR's NVPTX-dependent pieces
  (the NVVM target's serialization to PTX/cubin). MLIR's CMake turns it on
  by itself when `NVPTX` is in `LLVM_TARGETS_TO_BUILD`; setting it
  explicitly documents the intent.
- `MLIR_ENABLE_NVPTXCOMPILER=ON` links NVIDIA's `nvptxcompiler` static
  library into MLIR, so `gpu-module-to-binary` assembles PTX into a cubin
  in-process instead of shelling out to the `ptxas` binary. It needs a
  toolkit that ships `lib64/libnvptxcompiler_static.a` (recent CUDA
  toolkits do); if yours does not, leave the flag off and keep `ptxas` on
  PATH instead.
- `CUDACXX` / `CUDA_PATH` point CMake's CUDA detection and MLIR's toolkit
  lookup at the same toolkit as `CMAKE_CUDA_COMPILER`. On machines with
  several CUDA installs (clusters with `module load`), leaving them out can
  make CMake pick a different `nvcc` than the toolkit whose `libcudart` you
  link later. At compile time, MLIR also finds `ptxas` and libdevice through
  the toolkit path (`NVVM::getCUDAToolkitPath()`: the CMake-detected
  toolkit, or `CUDA_ROOT` / `CUDA_HOME` / `CUDA_PATH`).
- `CMAKE_CUDA_ARCHITECTURES` is your GPU's compute capability without the
  dot: `90` for H100 (the source repo's cluster), `80` for A100, `86` for
  RTX 30xx, `89` for RTX 40xx / L4, `75` for T4. Use the same value as the
  `sm_XX` you pass to ch31's pipelines (`CHIP=sm_90` / `cubin-chip=sm_90`).
  `nvidia-smi --query-gpu=compute_cap --format=csv` tells you yours.

For ch31's Python "check your CUDA toolchain" example
(`part6-end-to-end-ml/ch31-gpu-nvvm/examples/python-gpu/`), also install
NVIDIA's Python driver bindings into the environment that has the MLIR
Python bindings:

```sh
pip install "cuda-python==12.6.0"   # match your CUDA 12.x toolkit
```

Pin a 12.x release: the example imports `cuda.cuda` / `cuda.cudart`, the
module layout cuda-python 12 uses (later releases move these under
`cuda.bindings`). The source repo got the MLIR Python bindings from
`uv pip install mlir_python_bindings -f
https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest`;
this repo's wheel already provides them, as does a build with
`MLIR_ENABLE_BINDINGS_PYTHON=ON`.

## torch-mlir in-tree (the Part 6 frontend, natively)

Part 6's recommended route is the Linux wheels (see the
[part README](../part6-end-to-end-ml/README.md)); build from source only
if you must. torch-mlir builds as an LLVM external project against *its
own* pinned llvm-project submodule (see
[torch-mlir's development guide](https://github.com/llvm/torch-mlir/blob/main/docs/development.md)).
This is the setup ML-compiler-exercise used, with the fixes needed in 2026.
The macOS configure step below was run successfully on Apple silicon; the
full multi-hour `ninja` build and the Linux and CUDA variants were not run
for this appendix.

**1. Checkout and Python environment.** Run everything from inside the
torch-mlir checkout (ML-compiler-exercise had it at `externals/torch-mlir`).
`--depth 1` skips LLVM's multi-GB history:

```sh
git clone https://github.com/llvm/torch-mlir.git && cd torch-mlir   # or your fork
git submodule update --init --recursive --depth 1
uv venv venv --python 3.12 && source venv/bin/activate              # or python3 -m venv
```

torch-mlir pins an exact PyTorch *nightly* in `pytorch-requirements.txt` /
`torchvision-requirements.txt`. PyTorch keeps only about 60 days of
nightlies, so an older pin stops resolving ("no version of
torch==2.12.0.dev20260218+cpu"). When that happens, install the matching
stable release instead and keep the rest of the requirements:

```sh
uv pip install -r build-requirements.txt -r test-requirements.txt \
    torch==2.12.0 torchvision==0.27.0      # nearest stable to the pinned nightly
uv pip install transformers pybind11 lit  # used by Part 6's scripts
```

With uv, pass `--prerelease=allow` on the command line if you do install
from the nightly files (uv ignores the `--pre` line inside them).

**2. Configure and build.** Linux, CPU only (needs `lld` and `ccache`, or
set both options to `OFF`):

```sh
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
  externals/llvm-project/llvm
ninja -C build
```

macOS (Apple silicon) differs in three places:
- Use Apple's clang (`/usr/bin/clang`), not a Homebrew LLVM clang that may
  be first on `PATH`; it matches the system SDK and C++ library.
- Drop `LLVM_ENABLE_LLD`; macOS uses Apple's linker.
- ccache is optional (`brew install ccache`, then `LLVM_CCACHE_BUILD=ON`).

```sh
brew install cmake ninja
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DLLVM_CCACHE_BUILD=OFF \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
  externals/llvm-project/llvm
ninja -C build
```

Configure warnings about `ld-classic` and `library kineto not found` are
harmless. If a configure attempt fails, delete `build/CMakeCache.txt`
before retrying with different flags.

**With CUDA (Linux + NVIDIA).** The torch-mlir build accepts the same CUDA
options as [the CUDA runner build](#the-cuda-runner-build-for-ch31-execution)
above, so one build gives you both the frontend and the GPU runtime. Add
to the Linux command (and drop the `LLVM_TARGETS_TO_BUILD=host` line):

```sh
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="90" \
  -DCUDACXX=$CUDA_HOME/bin/nvcc \
  -DCUDA_PATH=$CUDA_HOME \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
  -DMLIR_ENABLE_NVPTXCOMPILER=ON \
```

(`90` is an H100; use your GPU's compute capability. Don't put a comment
after a `\` line continuation: the shell ends the command there.)

**3. Environment.** Put the tools and all three Python package directories
on your paths. Appending to the venv's activate script makes it stick:

```sh
TM=$PWD
cat >> venv/bin/activate <<EOF
export PATH="$TM/build/bin:\$PATH"     # torch-mlir-opt, mlir-opt, mlir-translate, llc
export PYTHONPATH="$TM/build/tools/mlir/python_packages/mlir_core:$TM/build/tools/torch-mlir/python_packages/torch_mlir:$TM/test/python/fx_importer"
EOF
source venv/bin/activate
```

**4. Building this repo against it** (optional; not tested). Point CMake at
the in-tree build's package config instead of the wheel:

```sh
cmake -S /path/to/mlir-unified-tutorial -B build-torchmlir \
  -DMLIR_DIR=$TM/build/lib/cmake/mlir -DLLVM_DIR=$TM/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit)
```

torch-mlir's pinned LLVM is not the wheel's, so `tutorial-opt` may not
compile against it without source changes (the
[installation appendix](appendix-installing-mlir.md) shows what breaks
against LLVM 20). Mind the **version-skew rule of thumb** from ch28: IR
produced by one LLVM version may not parse under another
(`tools/ll_fp_compat.py` exists because of exactly this), so keep the
torch-mlir side and the lowering side close in vintage, or pass `.mlir`
files — not `.ll` files — across the boundary.
