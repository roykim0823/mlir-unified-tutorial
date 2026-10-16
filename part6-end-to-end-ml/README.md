# Part 6: End-to-End ML Capstone

> Reworked from [ML-compiler-exercise](https://github.com/DavidGinten/ML-compiler-exercise)
> for the unified repo; ch25–26 come from mlir-ir-lowering (after Stephen
> Diehl's "MLIR" article series).

Part 6 starts where the earlier parts left off, with IR written by hand:
ch25 builds a neural network from scratch and compiles one dense layer, and
ch26 does the same for a transformer's softmax. Then it switches to
**real PyTorch models**: a toy two-layer module, the classic MNIST MLP,
HuggingFace's GPT-2, and in ch30 a CNN, resnet18, bert-base-uncased and
flan-t5-small. The frontend is
[torch-mlir](https://github.com/llvm/torch-mlir), which turns a
`torch.export`ed program into the linalg-on-tensors IR you know from
Chapter 22 — and from there it is *this repo's own* `tutorial-opt`, the
same wheel-built binary as everywhere else, all the way down to a native
binary you call from C++.

| Chapter | What happens |
|---|---|
| [ch25](ch25-neural-nets-from-scratch/README.md) | autodiff and an MLP in pure Python, then one dense+ReLU layer written in linalg, compiled and checked against NumPy |
| [ch26](ch26-transformers-and-softmax/README.md) | GPT-2 in NumPy, and a numerically stable softmax compiled from MLIR |
| [ch27](ch27-torch-mlir-import/README.md) | PyTorch → torch dialect → linalg-on-tensors, via the FX importer |
| [ch28](ch28-cpu-pipeline-and-abi/README.md) | linalg → LLVM → object file; the MemRefDescriptor ABI and `_mlir_ciface_` wrappers |
| [ch29](ch29-external-libs-blas/README.md) | a custom pass rewrites `linalg.matmul` into OpenBLAS `cblas_sgemm` calls |
| [ch30](ch30-real-models/README.md) | cnn, resnet18, bert and flan-t5 (pybind11 greedy decoding) through the same pipeline, plus benchmarks against PyTorch |
| [ch31](ch31-gpu-nvvm/README.md) | the same model mapped to GPU kernels and compiled to PTX via NVVM |
| [appendix](appendix-iree.md) | survey: how IREE, a production MLIR-based ML compiler, is put together |

## The environment split

Part 6 deliberately splits in two, so the heavyweight ML frontend cannot
contaminate the rest of the repo:

ch25 and ch26 need only NumPy and the repo's normal venv. From ch27 on:

**The Python side** (this directory's `src/*/lower_*.py`) needs torch +
torch-mlir. torch-mlir publishes nightly wheels **for Linux only**
(x86_64 and aarch64) — no macOS wheels, and building it from source means a
multi-hour LLVM build. So the import scripts run in their own environment
described by [`requirements.txt`](requirements.txt) (separate from the repo
root's), and on macOS they run in a Linux container:

```sh
cd part6-end-to-end-ml
./generate_snapshots.sh            # sample mnist gpt2 (Docker; ~10 min + downloads)
./generate_snapshots.sh sample     # just one
./generate_snapshots.sh cnn resnet18 bert flan_t5   # the ch30 models
```

**The C++/MLIR side** — everything from linalg downward — uses the ordinary
repo build against the MLIR wheel. To make that side (and its lit tests in
`test/part6/`) work *without* torch installed, the import results are
checked in as snapshots:

- `src/sample/sample_model_{torch,linalg}.mlir` — complete (the model is tiny)
- `src/mnist/mnist_model_linalg.mlir` — complete (5.1 MB; weights included so
  the end-to-end test can run)
- `src/gpt2/gpt2_linalg_stripped.mlir` — weights stripped (the full file is
  ~950 MB and is gitignored; regenerate it if you want to lower GPT-2 whole)
- **ch30's models (cnn, resnet18, bert, flan_t5): not generated yet.** Their
  scripts, C++ callers and pipeline scripts are ported, but no snapshot has
  been exported, so they have no lit tests and their compiled outputs are
  unverified. Run the `generate_snapshots.sh` line above; `cnn_model_linalg.mlir`
  is small enough to check in, the others are gitignored.

Every reference output quoted in these chapters (PyTorch tensor values, the
compiled binaries' stdout, GPT-2's greedy decoding) was produced by the
scripts in this directory — the lit tests check the compiled binaries
against PyTorch's own numbers.

## What changed from the source repo

All of ML-compiler-exercise's models are here: `sample`, `mnist` and `gpt2`
carry the walkthrough chapters (ch27–29, ch31), and `cnn`, `resnet18`,
`bert` and `flan_t5` are in ch30, along with the benchmarks and
`compare_outputs.py`. What was dropped is the cluster-specific build: the
in-tree torch-mlir/LLVM build, `module load` lines and hard-coded CUDA paths
are replaced by the wheel, the container above and `$CUDA_HOME`. The
in-tree build is kept for reference in
[the source-build appendix](../docs/appendix-source-build.md).
