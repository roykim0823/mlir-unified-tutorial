# Appendix: Other Ways to Install MLIR

For this tutorial, the supported path is the pinned pip wheel described in the
[root README](../README.md#setup). `requirements.txt` pins `mlir-wheel` and
`mlir-python-bindings` from the
[eudsl index](https://llvm.github.io/eudsl). Every chapter, script and lit
test was checked against that wheel, which is an LLVM 24 development snapshot
built with assertions. If the wheel installs, stay with it.

This appendix covers the other ways to get MLIR, for cases where you want:

1. **System-wide tools** that work outside this repo's `.venv`, e.g. for
   other projects or for an editor's `mlir-lsp-server`.
2. **Tools the wheel leaves out.** The wheel has `mlir-opt`, `mlir-translate`,
   `mlir-pdll`, `mlir-tblgen`, `FileCheck` and the runtime libraries, but no
   `llc`, no `mlir-runner` and no `mlir-transform-opt`. The chapters work
   around these gaps (`clang -x ir` in place of `llc`, the Python
   `ExecutionEngine` in place of `mlir-runner`), but you may want the real
   tools.
3. **CUDA.** The wheel can't run GPU code (ch31 stops at PTX). Running it
   needs a source build. See
   [Building MLIR from Source](appendix-source-build.md#the-cuda-runner-build-for-ch31-execution).
4. **A different LLVM version.** You may need one that matches another
   project, a release instead of a nightly, or a version that has dropped out
   of the wheel feed's ~1-month window.

Before you replace the wheel, read
[Version skew with the tutorial](#version-skew-with-the-tutorial). An older
MLIR can't build the C++ half of this repo, and some Part 1 lit tests fail on
it. A safe pattern is to keep the wheel for the repo and add a second
install only for the tools you're missing.

Most of this page is adapted from the installation guide in the
mlir-ir-lowering source repo (itself based on Stephen Diehl's "MLIR Part 0 —
Installing MLIR"). Each recipe has a **Status** line. "Checked" means it was
run or inspected on an Apple-silicon Mac with Homebrew in October 2026.
"Not verified" means the recipe is kept as the source had it (or with the
fix noted) but was not run here.

## Background: what are we installing?

MLIR is not distributed on its own. It lives in the `mlir/` directory of the
`llvm-project` monorepo and is built together with LLVM, so "installing MLIR"
means getting an LLVM build configured with `-DLLVM_ENABLE_PROJECTS=mlir`.
The pieces this tutorial uses, and where each install method gets them:

| Component | What it is | Wheel | Homebrew `llvm@20` |
| --- | --- | --- | --- |
| `mlir-opt` | Runs passes; lowers one dialect to another. | yes | yes |
| `mlir-translate` | MLIR ↔ other formats, notably LLVM IR (`.ll`). | yes | yes |
| `mlir-runner` | JIT-executes MLIR (formerly `mlir-cpu-runner`). | **no** | yes |
| `mlir-transform-opt` | Driver for the Transform dialect. | **no** | yes |
| `llc` | LLVM IR → object file / assembly. | **no** | yes |
| `clang` | C front end; also compiles `.ll` with `-x ir`. | no (use the system one) | yes |
| `FileCheck`, `not`, `count` | lit test utilities. | yes | yes |
| `libmlir_runner_utils`, `libmlir_c_runner_utils` | Runtime support libraries. | yes | yes |
| `libomp` | OpenMP runtime (ch07). | no | no (`brew install libomp`) |
| Assertions | Needed for `--debug-only=...` (ch23). | on | **off** |
| MLIR Python bindings | `import mlir`. | separate pinned wheel | no |

**Status:** checked. The wheel column comes from listing
`$(python -m mlir_wheel --root-dir)/bin` and `lib`, and the Homebrew column
from listing `/opt/homebrew/opt/llvm@20`. Assertion modes come from
`llvm-config --assertion-mode` (wheel `ON`, Homebrew `OFF`).

The pipeline from the source repo, using the full toolchain:

```
 .mlir ──mlir-opt──► (llvm dialect) ──mlir-translate──► .ll ──llc──► .o ──clang──► .so / executable
                            │
                            └──────── mlir-runner (JIT, no codegen) ───────► result
```

This repo's version replaces `llc` with `clang -x ir`, adds the
`tools/ll_fp_compat.py` step, and replaces `mlir-runner` with the Python
`ExecutionEngine`. See
[ch03](../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md).

## Which method should I pick?

| Your situation | Method |
| --- | --- |
| Following this tutorial | **The wheel** ([root README](../README.md#setup)) |
| macOS, want `llc` / `mlir-runner` / system-wide tools as well | **Homebrew**, alongside the wheel |
| Ubuntu/Debian, want prebuilt tools | **Apt** or **llvm.sh** |
| Want an official release tarball | **Precompiled packages** |
| Only use MLIR from Python | **pip / uv / Poetry / Conda** |
| Need CUDA, need to modify MLIR, or want an exact revision | **From source** (and [the source-build appendix](appendix-source-build.md)) |
| Want a reproducible, disposable environment | **Docker** |

The trade-off is time and disk against control. Prebuilt packages take
minutes. A source build takes an hour or more and tens of GB of disk, but
you choose the exact revision and features (assertions, Python bindings, the
CUDA runner).

> **MLIR has no stable API or IR guarantee.** Dialect syntax, pass names,
> pass options and C++ APIs all change between major releases. For example,
> `mlir-cpu-runner` became `mlir-runner`, and `bufferization.to_memref`
> became `bufferization.to_buffer`. Pin one version and keep to it. For this
> repo, that version is the wheel.

## From Homebrew (macOS)

This is the simplest way to get the complete command-line toolchain on a Mac:

```sh
brew install llvm@20
```

**Status:** checked. `llvm@20` 20.1.8 is installed at
`/opt/homebrew/opt/llvm@20`, and `mlir-opt --version` reports
`Homebrew LLVM version 20.1.8`.

### Why `llvm@20` and not plain `llvm`?

Homebrew's `llvm` formula follows the latest major release (23.1.2 as of
this writing), so it moves forward under you and can change pass flags. A
versioned formula pins you. The source repo targeted LLVM 20; `llvm@21` and
`llvm@22` also exist (`brew info llvm@22`). For this repo, no Homebrew
version matches the wheel's LLVM 24 snapshot, so treat any Homebrew LLVM as
a companion toolset. See
[Version skew](#version-skew-with-the-tutorial).

### Homebrew is "keg-only": fix your PATH, carefully

Homebrew doesn't symlink `llvm@20` into `/opt/homebrew/bin`, because it
would clash with Apple's toolchain. To find `mlir-opt`, `mlir-runner` and
the other tools, add the keg to PATH:

```sh
# Apple Silicon: prefix is /opt/homebrew
echo 'export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Intel Macs: prefix is /usr/local instead
# echo 'export PATH="/usr/local/opt/llvm@20/bin:$PATH"' >> ~/.zshrc
```

`brew info llvm@20` prints the exact lines for your machine, including the
`CMAKE_PREFIX_PATH` hint. Trust its output over this page.

**Watch the shadowing.** With the keg on PATH, a bare `mlir-opt` in your
shell is LLVM 20, even with the repo's venv activated. The venv doesn't put
the wheel's `bin/` on PATH. The chapter `build.sh` scripts prepend the
wheel's `bin/` themselves, and lit puts the configured build's tools first,
so scripts and tests are unaffected. Commands you type by hand from a
chapter are affected. Check with `which mlir-opt`
([ch09](../part1-using-mlir/ch09-testing-with-lit/README.md) shows the
expected answer), or put the wheel first:

```sh
export PATH="$(python -m mlir_wheel --bin-dir):$PATH"
```

### The runtime libraries

Examples that print from MLIR or call C helpers need MLIR's runtime support
libraries. Homebrew installs them next to the tools:

```
/opt/homebrew/opt/llvm@20/lib/libmlir_runner_utils.dylib
/opt/homebrew/opt/llvm@20/lib/libmlir_c_runner_utils.dylib
```

which is what `mlir-runner` needs, e.g.:

```sh
mlir-runner -e main -entry-point-result=i32 \
  -shared-libs=/opt/homebrew/opt/llvm@20/lib/libmlir_runner_utils.dylib \
  ./build/example_opt.mlir
```

Use runtime libraries from the same install as the runner. The wheel has
its own copies under `$(python -m mlir_wheel --lib-dir)`, which the lit
tests use through `%mlir_libs_dir`.

**Status:** checked (both dylibs exist; the smoke test in
[Verifying your installation](#verifying-your-installation) runs with this
`mlir-runner`).

Two more Homebrew notes from `brew info llvm@20`. `lld` is now a separate
formula (`brew install lld@20`) if you want `-DLLVM_ENABLE_LLD=ON` in a
source build. And Homebrew's `clang` needs the Command Line Tools
(`xcode-select --install`).

## Using Apt (Ubuntu 24.04)

The official LLVM APT repository provides prebuilt MLIR packages. Ubuntu
24.04's codename is **noble**, so the repo line uses
`noble`/`llvm-toolchain-noble-20` (22.04 is `jammy`).

```sh
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
sudo add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
sudo apt-get update
sudo apt-get install -y libmlir-20-dev mlir-20-tools
```

The tools are installed with version suffixes, e.g. `mlir-opt-20`. Call them
by that name, or add `/usr/lib/llvm-20/bin` to PATH, or create unsuffixed
names with `update-alternatives`. For this repo's CMake, the prefix is
`/usr/lib/llvm-20` (see [below](#pointing-this-repos-cmake-at-a-non-wheel-mlir)).

**Status:** not verified (no apt on this machine). The
`llvm-toolchain-noble-20` repository exists, and so do `-21`, `-22` and
`-23`; their `Release` files are reachable. The source's commands omitted
`sudo`, which is added here.

## Using `llvm.sh`

LLVM's convenience script sets up the APT repo and installs LLVM, on
Ubuntu/Debian:

```sh
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

To install a specific version:

```sh
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20
```

`llvm.sh` installs the core LLVM/clang packages. If `mlir-20-tools` and
`libmlir-20-dev` are missing afterwards, install them with the apt line
above.

**Status:** not verified (the script URL is live; it was not run).

## Precompiled packages

The LLVM project publishes official release archives on the
[releases page](https://github.com/llvm/llvm-project/releases). They help
when you need a specific release without building it. They are best-effort,
and the Linux ones may not match your distribution's glibc.

The source guide pointed at the `mlir-<version>.src.tar.xz` files. Those are
**source** tarballs with no `bin/` directory. For binaries, download the
platform archive instead: for 23.1.3, the latest release, these are
`LLVM-23.1.3-macOS-ARM64.tar.xz`, `LLVM-23.1.3-Linux-X64.tar.xz`,
`LLVM-23.1.3-Linux-ARM64.tar.xz` and Windows installers.

1. Download the archive for your platform and version.
2. Extract it:
   ```sh
   tar xf LLVM-<version>-<OS>-<arch>.tar.xz
   ```
3. Put its `bin/` on PATH, either by moving it to a system location:
   ```sh
   sudo mv LLVM-<version>-<OS>-<arch> /usr/local/llvm-<version>
   echo 'export PATH=/usr/local/llvm-<version>/bin:$PATH' >> ~/.bashrc
   ```
   or by pointing PATH at where it is:
   ```sh
   echo 'export PATH=/path/to/LLVM-<version>-<OS>-<arch>/bin:$PATH' >> ~/.bashrc
   ```
4. Verify:
   ```sh
   mlir-opt --version
   ```

**Status:** partly checked. The asset names above come from the GitHub
releases API. The 23.1.3 release configuration
(`clang/cmake/caches/Release.cmake`) includes `mlir` in the projects it
builds, so the tarballs should contain `mlir-opt`. The archives (~1.5 GB)
were not downloaded to confirm this.

> Precompiled packages save build time but may lack options a custom
> build gives you, notably assertions and the CUDA runner.

## From source (macOS)

A source build gives you the most control, at the cost of time and disk
(expect 30–60+ minutes and tens of GB). This is the source repo's recipe.
[Building MLIR from Source](appendix-source-build.md) has this repo's
recipe, explains the flags, and covers the CUDA and torch-mlir builds. Use
that page if the build is meant to replace the wheel here.

```sh
brew install cmake ccache ninja
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CCACHE_BUILD=ON
cmake --build . -t mlir-opt mlir-translate mlir-transform-opt mlir-runner
cmake --build . -t install
```

### What the important flags mean

| Flag | Why |
| --- | --- |
| `-G Ninja` | Ninja instead of Make: much faster incremental builds. |
| `-DLLVM_ENABLE_PROJECTS=mlir` | The flag that enables MLIR. Without it you get plain LLVM. |
| `-DLLVM_TARGETS_TO_BUILD="Native"` | Only your CPU's code generator. Add `NVPTX` for GPU work (see the [CUDA build](appendix-source-build.md#the-cuda-runner-build-for-ch31-execution)). |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build. `Debug` is far larger and slower; `RelWithDebInfo` is in between. |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Keeps MLIR's assertions in Release, which catches malformed IR and makes `--debug-only` work. |
| `-DLLVM_CCACHE_BUILD=ON` | Caches objects so rebuilds after a `git pull` are fast. Requires `ccache`. |

Notes on this recipe:

- The first `cmake --build` builds only the named tools. `install` then
  builds **everything** (it depends on `all`) and installs to
  `/usr/local` unless you pass `-DCMAKE_INSTALL_PREFIX=...`. If you only
  want the build tree, skip `install` and use `llvm-project/build` directly.
- To use an install prefix with this repo's lit tests, also pass
  `-DLLVM_INSTALL_UTILS=ON` so `FileCheck`, `not` and `count` are installed.
  A build tree has them without this flag.
- To match the wheel exactly, check out the commit in its version string
  before configuring. `pip show mlir-wheel` prints e.g.
  `20260727+abd6e745e`, which means llvm-project `abd6e745e`.
- To check that the build works before installing, run MLIR's own tests:
  ```sh
  cmake --build . --target check-mlir
  ```

**Status:** not verified (not built on this machine; `cmake`, `ninja` and
`ccache` are available through Homebrew).

## From source (Ubuntu 24.04)

This is the same recipe on Ubuntu. Building from source doesn't depend on
the release codename, only on having `clang`, `lld`, `cmake` and `ninja`.

```sh
sudo apt-get install clang lld cmake ninja-build ccache
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CCACHE_BUILD=ON
cmake --build . -t mlir-opt mlir-translate mlir-transform-opt mlir-runner
cmake --build . -t install
```

The macOS notes above apply here too.

**Status:** not verified. The source's `apt-get` line listed only `clang lld`.
The other three packages are added here because the recipe uses them.

## From wheels (pip, uv, Poetry)

There are two wheel feeds, and they are different projects:

| Feed | Packages | State |
| --- | --- | --- |
| [`llvm.github.io/eudsl`](https://llvm.github.io/eudsl) | `mlir-wheel` (tools + CMake package), `mlir-python-bindings`, `mlir-native-tools` | Rolling nightly, ~1 month retained. **This repo uses it.** |
| [`makslevental/mlir-wheels`](https://github.com/makslevental/mlir-wheels) | `mlir`, `mlir-python-bindings`, `mlir-native-tools` | **Archived** on GitHub; newest builds are 22.0.0 snapshots from Nov 2025. |

The source guide used the second feed, which has since been archived. It
still resolves, but it will not get newer builds. Prefer the eudsl feed, and
use the root README's `requirements.txt` for this repo.

### pip

```sh
# This repo's feed (what requirements.txt does):
pip install mlir-wheel mlir-python-bindings -f https://llvm.github.io/eudsl

# The source guide's feed (archived):
pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

These wheels are community-built, not official LLVM releases. The
`mlir-wheel` package puts its tools under `$(python -m mlir_wheel --bin-dir)`,
not on PATH.

**Status:** the eudsl line is checked (it is this repo's setup). The
`makslevental` line was checked only for resolution: `uv pip compile`
resolves `mlir==22.0.0.2025112901+9bae84b0`.

### uv

The source's `uv add mlir --index mlir=<url>` **fails**: a GitHub release
page is a flat list of links, not a package index, and uv reports "The
index returned metadata for the wrong package". Use `--find-links`:

```sh
uv add mlir-wheel mlir-python-bindings --find-links https://llvm.github.io/eudsl
```

Or, declaratively in `pyproject.toml`, mark the index `format = "flat"`:

```toml
[project]
dependencies = ["mlir-wheel", "mlir-python-bindings"]

[tool.uv.sources]
mlir-wheel = { index = "eudsl" }
mlir-python-bindings = { index = "eudsl" }

[[tool.uv.index]]
name = "eudsl"
url = "https://llvm.github.io/eudsl"
format = "flat"
explicit = true
```

For the archived feed, use the `makslevental` URL and the package name
`mlir` instead of `mlir-wheel`. Check that it works:

```sh
uv run python -m mlir_wheel --bin-dir     # eudsl feed
uv run mlir-opt --version                 # makslevental feed (puts mlir-opt on PATH)
```

**Status:** partly checked. `uv lock` resolved both `pyproject.toml` variants:
`mlir-wheel 20261012+6115c6d6d` from eudsl, and `mlir 22.0.0…` plus bindings
from the archived feed. Nothing was installed, and `uv run` was not tested.

### Poetry

```toml
[tool.poetry.dependencies]
python = "^3.12"
mlir-wheel = { version = "*", source = "eudsl" }

[[tool.poetry.source]]
name = "eudsl"
url = "https://llvm.github.io/eudsl"
priority = "explicit"
```

The source used `version = "latest"`, which is not a valid Poetry
constraint; it is `"*"` here. Check it works:

```sh
poetry run python -m mlir_wheel --bin-dir
```

**Status:** not verified (Poetry isn't installed here). Poetry needs to read
a flat link page as a source for this to work, which was not confirmed.

## Using Conda

conda-forge packages both the MLIR tools and the Python bindings:

```sh
conda install conda-forge::mlir
conda install conda-forge::mlir-python-bindings
```

**Status:** partly checked. The anaconda.org API lists `mlir` 23.1.3 and
`mlir-python-bindings` 23.1.2, both with `osx-arm64` builds. Conda isn't
installed here, so the package contents (e.g. whether `mlir-runner` is
included) were not inspected. For this repo's CMake, the prefix is
`$CONDA_PREFIX`.

## Using Docker

For a containerized, reproducible environment, Stephen Diehl maintains
prebuilt images at
[sdiehl/docker-mlir-cuda](https://github.com/sdiehl/docker-mlir-cuda). They
come in CUDA and non-CUDA variants for Ubuntu 22.04 and 24.04:

```sh
# Ubuntu 24.04 with CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-cuda-ubuntu24.04
# Ubuntu 24.04 without CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-ubuntu24.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-ubuntu24.04

# Ubuntu 22.04 with CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu22.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-cuda-ubuntu22.04
# Ubuntu 22.04 without CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-ubuntu22.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-ubuntu22.04
```

Run one interactively:

```sh
docker run -it ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04 bash
```

or use one as a base image:

```dockerfile
FROM ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04
```

These images contain MLIR 19/20. That is older than this repo's wheel, so
the [skew caveats](#version-skew-with-the-tutorial) apply. They are not the
container Part 6 uses for torch-mlir (`generate_snapshots.sh`).

> **macOS:** Docker on a Mac has no GPU/CUDA passthrough. The CUDA images are
> only useful on a Linux host with an NVIDIA GPU. On macOS, use the non-CUDA
> tags, which run under virtualization.

**Status:** not verified. The tag list matches the image repo's README, but
no image was pulled or run.

## Installing the MLIR Python bindings

This repo's `requirements.txt` already installs pinned
`mlir-python-bindings` that match the wheel. The options below are for
other setups. Bindings must come from the same LLVM build or version as the
tools you pair them with.

```sh
pip install mlir-python-bindings -f https://llvm.github.io/eudsl
pip install mlir-python-bindings -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest   # archived feed
uv add mlir-python-bindings --find-links https://llvm.github.io/eudsl
poetry add mlir-python-bindings --source eudsl      # after the Poetry source above
conda install conda-forge::mlir-python-bindings
```

If none of those cover your platform, build the bindings in the LLVM source
tree with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`.

On Ubuntu, install the build dependencies first:

```sh
sudo apt-get install -y \
  bash-completion ca-certificates ccache clang cmake cmake-curses-gui \
  git lld man-db ninja-build pybind11-dev python3 python3-numpy \
  python3-pybind11 python3-yaml unzip wget xz-utils
```

Build [nanobind](https://github.com/wjakob/nanobind) (the C++↔Python glue
the bindings use) and install it locally:

```sh
git clone https://github.com/wjakob/nanobind && \
  cd nanobind && \
  git submodule update --init --recursive && \
  cmake \
    -G Ninja \
    -B build \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_INSTALL_PREFIX=$HOME/usr && \
  cmake --build build --target install
```

Then configure and build LLVM/MLIR with the bindings enabled (from the
`llvm-project` checkout):

```sh
cmake llvm \
  -G Ninja \
  -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_PREFIX_PATH=$HOME/usr \
  -DLLVM_BUILD_EXAMPLES=On \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_CCACHE_BUILD=On \
  -DLLVM_CCACHE_DIR=$HOME/ccache \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_LLD=On \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
  -DLLVM_USE_SPLIT_DWARF=On \
  -DMLIR_ENABLE_BINDINGS_PYTHON=On \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=On
cmake --build build -t mlir-opt mlir-translate mlir-transform-opt mlir-runner
cmake --build build -t install
```

> Set `-DPython3_EXECUTABLE` to the Python you will use (`which python3`;
> inside a venv, the venv's interpreter). A mismatched interpreter is the
> most common reason `import mlir` fails later.

The bindings are built into:

```
build/tools/mlir/python_packages/mlir_core/mlir
```

Package them as a wheel:

```sh
cd build/tools/mlir/python_packages/mlir_core
python setup.py bdist_wheel
pip install dist/mlir_core-*.whl
```

or add them to `PYTHONPATH` (adjust the path to your build location):

```sh
export PYTHONPATH=$PYTHONPATH:$HOME/llvm-project/build/tools/mlir/python_packages/mlir_core
```

**Status:** not verified (no bindings build was run here). The source's
`PYTHONPATH` example used `$HOME/build/...`; it now points into the
`llvm-project` checkout to match the commands above.

## Version skew with the tutorial

`mlir-opt --version` for the pinned wheel prints `LLVM version 24.0.0`
(`Optimized build with assertions`). No released LLVM is that new: Homebrew,
apt.llvm.org, conda-forge and the GitHub releases currently top out at 23.x.
Skew therefore shows up two ways.

### Wheel-produced IR into an older clang or llc

This is the skew the tutorial already handles. The wheel's
`mlir-translate` writes LLVM 24 textual IR, and `clang`/`llc` 20 (and
Apple's clang) can't read all of it:

- float constants as typed hex literals (`float f0x3EA3B35C`) or
  shortest-round-trip decimals (`float -4.291610e-01`), which older parsers
  reject as "floating point constant invalid for type";
- `inf`/`nan` float literals;
- a new function attribute, `nocreateundeforpoison`, on intrinsic
  declarations such as `@llvm.exp.f32`, which older parsers reject as
  "unterminated attribute group".

[`tools/ll_fp_compat.py`](../tools/ll_fp_compat.py) is a stdin→stdout filter
that rewrites the float forms to the legacy double-widened hex encoding and
drops the attribute. The chapters that turn wheel output into native code
pipe every `.ll` through it before `clang -x ir`:
[ch03](../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md),
[ch04](../part1-using-mlir/ch04-memory-and-the-c-abi/README.md),
[ch06](../part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md),
[ch07](../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md),
[ch08](../part1-using-mlir/ch08-linalg-by-hand/README.md) and
[ch24](../part5-structured-and-parallel/ch24-equality-saturation-egraphs/README.md).
[ch28](../part6-end-to-end-ml/ch28-cpu-pipeline-and-abi/README.md) explains
the problem in detail. If you install Homebrew `llc` and use it in place of
`clang -x ir`, the same filter applies:

```sh
mlir-translate --mlir-to-llvmir kernel.mlir -o kernel.ll           # wheel, LLVM 24
python3 tools/ll_fp_compat.py < kernel.ll > kernel.compat.ll
llc -filetype=obj -relocation-model=pic kernel.compat.ll -o kernel.o  # Homebrew, LLVM 20
```

**Status:** checked. A module that calls `math.exp`, lowered with the wheel,
was rejected by Homebrew clang 20.1.8 and by Apple clang 21.0.0 with
"unterminated attribute group". After `ll_fp_compat.py`, both compiled it and
the program returned the expected value.

**Coverage:** the filter rewrites float literals wherever LLVM 24 can print
them without a preceding `float` keyword too: the second operand of
`fadd`/`fsub`/`fmul`/`fdiv`/`frem`/`fcmp` (as in `fmul float %0,
3.200000e-01`, which clang rejects because `0.32` isn't exactly
representable in f32) and `phi float` incoming values. It also rewrites
`inf`/`-inf`/`nan`. It is still a text filter, not a parser: if a newer LLVM
prints some other construct an older clang rejects, use an `llc`/`clang` at
least as new as the IR (when one exists), or pass `.mlir` across the version
boundary instead of `.ll`.

A rule of thumb from ch28 and the source-build appendix: tools from
different LLVM versions should exchange **`.mlir` source files, not
generated `.ll` or lowered MLIR**. This also applies to `mlir-runner`. To
JIT with Homebrew's runner, lower with Homebrew's `mlir-opt` too, starting
from the chapter's input file. Don't feed it the wheel's output.

### Running the repo against an older MLIR

This direction is harder to work around. Pass names, pass options, op
syntax, diagnostics and the C++ API all drift between releases. The lit
tests' `CHECK` lines match the wheel's output exactly, so a different
version can fail them even when its output is correct. To measure this,
the repo was configured against Homebrew LLVM 20.1.8 (as described in the
next section) and the lit suite was run with LLVM 20's tools:

| Area | Result on LLVM 20 | Why |
| --- | --- | --- |
| C++: `tutorial-opt`, so all of Parts 2–5 and Part 6's C++ tests | **Does not compile** | Newer APIs: `OpTy::create(...)` builders, `constFoldBinaryOp` signature, `scf::SCFTilingResult::replacements`, `loopUnrollFull`, `mlir/Dialect/Affine/Transforms/Passes.h` |
| ch02 (`ctlz*`), ch03, ch04 C wrappers, ch06 passes and run tests, ch07 `affine-parallelize` and parallel matmul, ch08 elementwise/matmul/reduce/tiling/broadcast, ch24, ch25, ch26, ch31 `square_gpu` | Pass | mlir-opt-only pipelines whose output is unchanged |
| ch04 array add, low-level LLVM, tensor bufferization | Fail | `bufferization.to_buffer` (LLVM 20 spells it `to_memref`), `linalg.map` syntax, output text |
| ch06 affine maps | Fail | Diagnostic wording changed |
| ch07 `scf-to-openmp`, OpenMP to LLVM | Fail | Lowered output differs from the `CHECK` lines |
| ch08 fusion | Fail | Pass option `modify-public-functions` doesn't exist in 20 |
| ch31 GPU tiling / MNIST / sample PTX | Fail | LLVM 20's affine dimension/symbol checks reject the tiled IR (`operand cannot be used as a dimension id`) |

**Status:** checked (configure, build and lit run in a scratch build
directory). 21 tests passed, 46 failed, and 1 was skipped. Most failures
come from the C++ side failing to build. LLVM 21–23 were not tested. Expect
fewer failures as versions approach the wheel's, but only the wheel's
commit is known to give a clean `check-all`.

So the chapters that assume the wheel's exact pass names and output are
**every C++ chapter (ch10–ch24 and Part 6's `tutorial-opt` flows)**, plus
the Part 1 chapters ch04, ch06, ch07 and ch08 and the ch31 GPU pipelines.
ch23's `--debug-only` walkthrough also needs an assertions-enabled MLIR,
which Homebrew's is not.

## Pointing this repo's CMake at a non-wheel MLIR

[`CMakeLists.txt`](../CMakeLists.txt) uses a plain
`find_package(MLIR CONFIG)`. It has nothing wheel-specific: any MLIR that
ships `lib/cmake/mlir/MLIRConfig.cmake` works, and `MLIRConfig` locates the
matching LLVM. Pass the **prefix** (CMake appends `lib/cmake/mlir`) or point
at the directory directly:

```sh
mkdir build-other && cd build-other
cmake .. -G Ninja \
  -DCMAKE_PREFIX_PATH=<prefix> \
  -DLLVM_EXTERNAL_LIT=$(which lit)
# equivalently: -DMLIR_DIR=<prefix>/lib/cmake/mlir
```

| Install | `<prefix>` |
| --- | --- |
| The wheel (default) | `$(python -m mlir_wheel --root-dir)` |
| Homebrew | `/opt/homebrew/opt/llvm@20` (`brew --prefix llvm@20`) |
| apt.llvm.org | `/usr/lib/llvm-20` |
| Conda | `$CONDA_PREFIX` |
| Source build tree | `/path/to/llvm-project/build` |
| Source install | your `CMAKE_INSTALL_PREFIX` |

Things to check:

- **Confirm which MLIR was found.** The configure log prints
  `Using MLIRConfig.cmake in: ...`. If `CMAKE_PREFIX_PATH` from your
  environment (as `brew info` suggests setting) also points at another
  install, the first match wins. Use `-DMLIR_DIR` to be unambiguous, and use
  a fresh build directory whenever you switch.
- **`LLVM_EXTERNAL_LIT` is required.** Homebrew, apt and release tarballs
  don't ship `llvm-lit`. The pip `lit` from `requirements.txt` works with
  any of them. A source build tree has `build/bin/llvm-lit`.
- **lit runs the found install's tools.** `test/lit.site.cfg.py` sets the
  tools directory to that install's `bin/` and puts it at the front of
  PATH, so tests run its `mlir-opt`, `FileCheck`, etc. no matter what your
  shell's PATH says. That `bin/` must contain `FileCheck`, `not` and
  `count`. Homebrew and the wheel ship them; for a source *install*, build
  with `-DLLVM_INSTALL_UTILS=ON`.
- **Version.** Configure succeeds against LLVM 20, but the build does not
  (see the table above). In practice the non-wheel MLIR must be close to the
  wheel's commit, and a source build at that commit is the reliable route
  ([source-build appendix](appendix-source-build.md#plain-mlir-replaces-the-wheel)).
- **Static vs shared.** The repo's libraries are built static to match the
  wheel (`ENABLE_SOLVER` forces `BUILD_SHARED_LIBS=OFF` for this reason).
  An MLIR built with `-DBUILD_SHARED_LIBS=ON` was not tested.

**Status:** checked for Homebrew `llvm@20`. Configure found
`/opt/homebrew/opt/llvm@20/lib/cmake/mlir`, `lit.site.cfg.py` pointed at
`/opt/homebrew/opt/llvm@20/bin`, and the build failed as described.

## Verifying your installation

Whatever method you used, check that the tools are on PATH and are the ones
you expect:

```sh
mlir-opt --version
mlir-translate --version
mlir-runner --version    # older builds: mlir-cpu-runner; the wheel has neither
which mlir-opt           # make sure this is the install you meant
```

For the wheel, the tools are not on PATH by default:

```sh
ls "$(python -m mlir_wheel --bin-dir)"
"$(python -m mlir_wheel --bin-dir)/mlir-opt" --version   # LLVM version 24.0.0
```

An end-to-end smoke test: write a trivial module and JIT it.

```sh
cat > /tmp/smoke.mlir <<'EOF'
func.func @main() -> i32 {
  %0 = arith.constant 42 : i32
  return %0 : i32
}
EOF

mlir-opt /tmp/smoke.mlir \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts | \
mlir-runner -e main -entry-point-result=i32
# prints: 42
```

If that runs, the install works. Then check the Python bindings:

```sh
python -c "import mlir; from mlir.ir import Context; print('mlir bindings OK')"
```

**Status:** checked with Homebrew `llvm@20` (prints `42`). With the wheel,
use [ch03](../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md)'s
`jit.py` in place of `mlir-runner`.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `command not found: mlir-opt` | PATH not updated. Re-check the Homebrew `export PATH=...` line and `source` your shell rc. For the wheel, use `python -m mlir_wheel --bin-dir`. |
| `mlir-cpu-runner: command not found` | Renamed to `mlir-runner` in recent LLVM. Use the new name (or the old one on older installs). The wheel ships neither. |
| `which mlir-opt` shows Homebrew while you're following a chapter | The keg is shadowing the wheel; see [the PATH note](#homebrew-is-keg-only-fix-your-path-carefully). Scripts and lit are unaffected; commands typed by hand are. |
| Unknown pass or `no such option`, e.g. `--convert-cf-to-llvm` or `modify-public-functions` | Version mismatch. Pass names and options differ across LLVM majors. Use the version the examples target: the wheel for this repo. |
| `custom op 'bufferization.to_buffer' is unknown` | IR from a newer MLIR fed to an older one (here, LLVM 20). Exchange `.mlir` *source*, not lowered output, across versions. |
| `floating point constant invalid for type` / `unterminated attribute group` from `clang` or `llc` | LLVM 24 IR into an older backend. Pipe the `.ll` through `tools/ll_fp_compat.py` (see [what it covers](#wheel-produced-ir-into-an-older-clang-or-llc)). |
| `library not found` / `dlopen` error from `mlir-runner` | Wrong path to `libmlir_runner_utils`. On Apple silicon it is `/opt/homebrew/opt/llvm@20/lib/...`; on Intel, `/usr/local/opt/...`. Use the runtime libraries from the same install as the runner. |
| `-debug-only` / `--debug-only` is an unknown option | MLIR built without assertions (e.g. Homebrew). Use the wheel or a source build with `-DLLVM_ENABLE_ASSERTIONS=ON`. |
| CMake picks the wrong MLIR, or the build fails with `no member named 'create'` | `CMAKE_PREFIX_PATH` points at an older MLIR (check `Using MLIRConfig.cmake in:`). Use a fresh build dir with `-DMLIR_DIR=...`. |
| `import mlir` fails in Python | Bindings built against a different Python than the one you're running, or `PYTHONPATH` not set. Rebuild with the right `-DPython3_EXECUTABLE`. |
| uv: `The index returned metadata for the wrong package` | A flat link page was used as an `--index`. Use `--find-links`, or `format = "flat"` in `pyproject.toml`. |
| Source build runs out of disk / RAM | Use `-DCMAKE_BUILD_TYPE=Release` and `-DLLVM_TARGETS_TO_BUILD="Native"`, and link with `lld` (`-DLLVM_ENABLE_LLD=On`) to reduce memory use during linking. |
