# Chapter 30: Scaling Up to Real Models

> Reworked from ML-compiler-exercise chapters 3, 4 and 7 (model reference
> and benchmarks) for the unified repo.

Chapters 27–29 took three models through the pipeline: a toy, an MLP, and
GPT-2 (lowered to an object file but never called). This chapter runs the
same machinery on four more models. Each one adds a single new wrinkle at
the boundary between PyTorch, MLIR and C++:

| Model | New wrinkle | Function signature after import |
|---|---|---|
| **cnn** (LeNet-style, random weights) | convolutions and pooling | `(4x1x28x28 f32) -> 4x10` |
| **resnet18** (HF `microsoft/resnet-18`) | BatchNorm *buffers* lifted to arguments | 61 args as exported; **1** after `freeze_buffers()` |
| **bert-base-uncased** | two buffer arguments fed at run time; **two results** | `(2 buffers, input_ids, token_type_ids, attention_mask) -> (1x12x768, 1x768)` |
| **flan-t5-small** | encoder-decoder, two dynamic dims, a **generation loop** in C++ called from Python via pybind11 | `(1x?, 1x?, 1x? i64) -> 1x?x32128` |

Then we time everything against PyTorch, which is a humbling exercise and
the right note to end the CPU half of the capstone on.

> **Status: scripts ported, snapshots not yet generated.** The four models'
> MLIR exports need torch-mlir (Linux-only), and this port was done on a
> macOS machine without Docker. Run
>
> ```sh
> cd part6-end-to-end-ml
> ./generate_snapshots.sh cnn resnet18 bert flan_t5
> ```
>
> to produce them. Until then, **nothing in this chapter that starts from
> an `*_linalg.mlir` file has been run**. What *has* been verified, and
> how, is marked in each section: the PyTorch reference outputs (run with
> torch 2.12.0 + transformers 5.19 on macOS), the export signatures (via
> `torch.export` with the torch-mlir importer's freezing logic applied),
> every C++ caller compiled, the C ABI patterns on toy functions, the
> flan-t5 harness against a fake model, and the sample/mnist benchmarks
> end to end.

## What you will learn

- Running the Chapter 28 pipeline on convolutional, residual and
  transformer models, and what changes in the C++ caller each time
- Feeding lifted buffers from a file at run time, and freezing them away
  at export time
- Calling a function with **two results** through `_mlir_ciface_`
- Driving an encoder-decoder's **autoregressive loop** from C++, and
  exposing it to Python as a pybind11 extension module (position-independent
  object code, `.so` linking)
- How to benchmark compiled models against PyTorch fairly, and how to read
  the gap

## Prerequisites

- [Chapter 27](../ch27-torch-mlir-import/README.md), especially sections
  6–8 (FLAN-T5 export, lifted buffers, `aten.any.dim`)
- [Chapter 28](../ch28-cpu-pipeline-and-abi/README.md), including section 7
  (several results; the `bufferization.clone` gap), and
  [Chapter 29](../ch29-external-libs-blas/README.md) (BLAS linking)
- A built `build/tools/tutorial-opt` and the repo's `.venv` (the scripts
  find the wheel's `mlir-opt`/`mlir-translate`/runtime libraries through
  `python -m mlir_wheel --root-dir`, or `$MLIR_ROOT` if set)

All files: [`src/cnn/`](../src/cnn/), [`src/resnet18/`](../src/resnet18/),
[`src/bert/`](../src/bert/), [`src/flan_t5/`](../src/flan_t5/),
[`src/compare_outputs.py`](../src/compare_outputs.py), and the
`benchmark*` files in every model directory.

## 1. One script per model

Each new model directory follows the convention `generate_snapshots.sh`
expects:

| File | Runs where | What it does |
|---|---|---|
| `lower_<m>_model.py` | container (torch-mlir) | export to `*_linalg.mlir` (+ a `_stripped` copy for the big ones) |
| `run_<m>_model.py` | anywhere with torch | the PyTorch reference output |
| `run_mlir_pipeline.sh` | host | Chapter 28 pipeline → `.o` → linked caller |
| `<m>_call.cpp` | host | the C++ caller (bert: `bert_call.cpp`; flan_t5: `flan_call.cpp` + `flan_generate.h`) |
| `benchmark_<m>_model.py` / `<m>_call_benchmark.cpp` / `benchmark.sh` | torch / host | section 7 |

`run_mlir_pipeline.sh` is the Chapter 28 flow plus one step from Chapter 28
section 7:

```sh
tutorial-opt --linalg-to-bufferization <m>_linalg.mlir \
| mlir-opt --convert-bufferization-to-memref \
| tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm \
| mlir-translate --mlir-to-llvmir \
| python3 tools/ll_fp_compat.py \
| clang -O3 -x ir - -c -o <m>.o
```

(`--convert-bufferization-to-memref` lowers any leftover
`bufferization.clone`; `ll_fp_compat.py` adapts LLVM 24's `.ll` syntax for
older clangs.) The script then links with `%blas_link_flags`' equivalent (`-framework Accelerate` on
macOS, `-lopenblas` elsewhere) plus `-lmlir_c_runner_utils` (for
`memrefCopy`, see Chapter 28 section 5). Two differences from the source
tutorial: the wheel has no `llc`, so `clang -x ir` does code generation
(note the **`-O3`**: without it clang compiles IR at `-O0`), and the
OpenBLAS path is no longer hard-coded to `$REPO/openblas`.

## 2. cnn: convolutions

[`src/cnn/model.py`](../src/cnn/model.py) is a LeNet-style network:
two blocks of `Conv2d` → ReLU → 2×2 max-pool (1→6→16 channels), a flatten
to 400, then Linear 400→120→84→10 and a `log_softmax`. The batch size is
4. Weights are random (seed 41), as for mnist. Import is the mnist recipe
verbatim ([`lower_cnn_model.py`](../src/cnn/lower_cnn_model.py)); the
export should be under a megabyte (~60k parameters), small enough to
check in like mnist's.

What to look for in `cnn_model_linalg.mlir` once generated: the
convolutions arrive as linalg named convolution ops (torch-mlir's
TorchToLinalg emits `linalg.conv_2d_nchw_fchw`), and the max-pools as
`linalg.pooling_nchw_max` with a `-inf` init. The `ll_fp_compat.py` filter
now also rewrites infinite float literals for older clangs. Neither op is
a `linalg.matmul`, so Chapter 29's pass leaves them alone and they become
plain loop nests. Only the three Linear layers become `cblas_sgemm` calls.
That matters in section 7.

The C++ side ([`cnn_call.cpp`](../src/cnn/cnn_call.cpp)) is mnist's with a
batch of 4. The port reads the result entirely through the returned
descriptor (sizes, strides, offset) and frees it, rather than trusting
its own constants. PyTorch reference ([`run_cnn_model.py`](../src/cnn/run_cnn_model.py)),
with every image all ones, so all four rows match:

```
Output: tensor([[-2.39254, -2.33461, -2.23863, -2.27202, -2.22305, -2.25292, -2.35712,
         -2.40054, -2.26724, -2.30542],
        ... (three more identical rows) ...], grad_fn=<LogSoftmaxBackward0>)
```

Build and run (pending the snapshot):

```sh
cd part6-end-to-end-ml/src/cnn
sh run_mlir_pipeline.sh && ./cnn > cnn_c_out.txt
python ../compare_outputs.py cnn_c_out.txt cnn_py_out.txt   # see section 8
```

## 3. resnet18: buffers, frozen

ResNet-18 (HuggingFace `microsoft/resnet-18`, ImageNet weights) adds
residual connections, 20 BatchNorms and an adaptive average pool. None of
that is new to the pipeline. What *is* new is that BatchNorm's
running statistics are **buffers**, and torch-mlir leaves buffers as
function arguments: 61 of them, 20 never used
([Chapter 27, section 7](../ch27-torch-mlir-import/README.md#7-lifted-buffers-and-the-dead-argument-problem)).

The source tutorial dumped them with `get_buffers_in_mlir_format.py` and
pasted them into the torch-dialect file by hand, then lowered with
`torch-mlir-opt`. This port keeps that script for reference but automates
the step in
[`lower_resnet18_model.py`](../src/resnet18/lower_resnet18_model.py):

```python
def freeze_buffers(model):
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            del module._buffers[name]
            if name == "num_batches_tracked":
                module._buffers[name] = None          # training-only
            else:
                module.register_parameter(
                    name, torch.nn.Parameter(buf, requires_grad=False))
    return model
```

The script writes both versions: `resnet18_model_torch_lifted.mlir` (the
61-argument problem, to look at) and `resnet18_model_linalg.mlir` (the
frozen, one-argument function that
[`resnet18_call.cpp`](../src/resnet18/resnet18_call.cpp) and Chapter 31's
`src/resnet18/gpu/` scripts consume). Verified with torch 2.12: after
freezing, `torch.export` leaves exactly one input (`pixel_values`) and
the logits are bit-identical. The torch-mlir import itself is pending.

Reference ([`run_resnet18_model.py`](../src/resnet18/run_resnet18_model.py),
an all-ones 3×224×224 image; the caller prints the same two lines):

```
First 10 logits:     tensor([-0.03914,  0.11446, -1.79676, -1.23429, -0.81900,  0.32396, -2.18660,
        -1.28766, -1.90192, -0.73148])
Top class index:     623
```

```sh
cd part6-end-to-end-ml/src/resnet18
sh run_mlir_pipeline.sh && ./resnet18
```

The linalg export embeds 11.7M parameters as hex, so expect ~90 MB of
text. Keep it out of git; the `_stripped` copy is for reading.

## 4. bert: buffers at run time, two results

bert-base-uncased is exported through a wrapper returning a **tuple**,
`(last_hidden_state, pooler_output)`, and lowered two-step
([`lower_bert_model.py`](../src/bert/lower_bert_model.py); Chapter 27,
section 8 explains why and tells the `aten.any.dim` story). The resulting
function has five arguments, the two buffers first:

```
bert_model(position_ids [1,512] i64, token_type_ids [1,512] i64,   ← buffers
           input_ids [1,12] i64, token_type_ids [1,12] i64,
           attention_mask [1,12] i64) -> ([1,12,768] f32, [1,768] f32)
```

This time the buffers are *not* frozen. [`bert_call.cpp`](../src/bert/bert_call.cpp)
(the source's `call_bert.cpp`) shows the runtime alternative, which you
need whenever state cannot or should not be baked into the binary:

1. [`get_buffers.py`](../src/bert/get_buffers.py) writes both buffers to
   `tensor.csv`, one row each, in `named_buffers()` order (which is the
   argument order). `generate_snapshots.sh` runs it automatically.
2. `load_tensor()` reads the CSV back, and each row's `std::vector<int64_t>`
   becomes the data of a `[1, 512]` descriptor.
3. The **two results** come back through one out-pointer to a struct
   holding one descriptor per result, in result order (Chapter 28,
   section 7, where this ABI was verified on a two-result toy function):

   ```cpp
   struct Output {
     MemRefDescriptor<float, 3> last_hidden_state; // [1, 12, 768]
     MemRefDescriptor<float, 2> pooler_output;     // [1, 768]
   };
   extern "C" void _mlir_ciface_bert_model(Output *result,
       MemRefDescriptor<int64_t, 2> *buffer_embeddings_position_ids,
       MemRefDescriptor<int64_t, 2> *buffer_embeddings_token_type_ids,
       MemRefDescriptor<int64_t, 2> *input_ids,
       MemRefDescriptor<int64_t, 2> *token_type_ids,
       MemRefDescriptor<int64_t, 2> *attention_mask);
   ```

Fixes relative to the source caller: `<sstream>` was missing and
`std::stoi` became `std::stoll`; a missing `tensor.csv` is an error
rather than a crash; and the last-hidden-state descriptor's strides
(`{768, 12, 1}` in the source, which is wrong though harmless, since the
descriptor is overwritten) are no longer hand-written at all. The token
ids for `"This is a sample input for BERT model export."` are
`[101, 2023, 2003, 1037, 7099, 7953, 2005, 14324, 2944, 9167, 1012, 102]`,
confirmed with the tokenizer. BERT's GELU uses `math.erf`, which
`--convert-math-to-libm` turns into a libm call. That rung of the
Chapter 28 pipeline was put there for this model.

Reference ([`run_bert_model.py`](../src/bert/run_bert_model.py) prints in
the caller's format):

```
Last hidden state (first 10 values of each of the 12 vectors):
-0.53822 -0.49217 -0.02191 -0.36287 -0.21428 -0.35426 -0.08411 0.29183 -0.20049 0.35593
-0.60307 -0.88999 0.00697 -0.71585 0.80235 0.22622 -0.08629 1.24849 0.01254 0.56988
...
-0.25144 0.02528 -0.21374 0.15785 -0.11461 -1.38765 0.58753 -0.46022 0.13321 0.46256

Pooler output:
-0.90762 -0.51741 -0.84328 0.74621 0.73548 -0.35998 0.83880 0.36642 -0.74345 -0.99999
```

```sh
cd part6-end-to-end-ml/src/bert
sh run_mlir_pipeline.sh && ./bert      # reads ./tensor.csv
```

## 5. flan-t5-small: a generation loop in C++, called from Python

[Chapter 27, section 6](../ch27-torch-mlir-import/README.md#6-an-encoder-decoder-flan-t5)
exported `flan_decoder(input_ids, attention_mask, decoder_input_ids) ->
logits [1, dec_len, 32128]` with both sequence lengths dynamic. One call
is one decoder step. *Generating* means looping, and this is the first
model where the compiled function is not the whole program. The source's
design, kept here:

- **Python** tokenizes the prompt, calls `flan_call.generate_tokens(ids,
  mask)`, and detokenizes the result
  ([`run_flan_t5_model_mlir.py`](../src/flan_t5/run_flan_t5_model_mlir.py)).
  The host needs `transformers` for the tokenizer, but not torch.
- **C++** ([`flan_generate.h`](../src/flan_t5/flan_generate.h)) runs the
  greedy loop: start the decoder with `[0]` (`<pad>`, T5's start token);
  each step, call the model, take the logits row of the last real
  position, skip NaN/inf values, print the top 5, append the arg-max; stop
  at `</s>` (id 1) or after 20 steps.
- **pybind11** ([`flan_call.cpp`](../src/flan_t5/flan_call.cpp)) turns that
  function into a Python extension module. `pybind11/stl.h` converts
  Python lists to and from `std::vector<int64_t>`.

```cpp
PYBIND11_MODULE(flan_call, m) {
  m.def("generate_tokens", &generate_tokens, ...,
        pybind11::arg("input_ids"), pybind11::arg("attention_mask"),
        pybind11::arg("verbose") = true);
}
```

**Padding the decoder.** Every step passes the decoder input padded to its
maximum, 32 tokens, and reads row `current_len - 1`. Is that sound? Yes:
T5's decoder self-attention is *causal*, so the logits at position *t*
depend only on tokens 0..*t*, and trailing pads cannot change the row we
read. Checked in PyTorch: greedy decoding with the 32-padded input
reproduces `model.generate`'s ids exactly (Chapter 27, section 6). It is
also wasteful, since every step computes 32 positions (exercise 3).

**Two bugs fixed from the source harness.**

1. The source declared a fourth model input, a `decoder_attention_mask`,
   that the exported function does not have. The C ABI silently ignored
   the extra pointer. This port declares exactly the three inputs.
2. The source pre-filled its output buffer with `-999` and then "checked
   the buffer was written". But the model allocates its own result and
   overwrites the out-slot (Chapter 28, section 3), so the check could
   never fail, and every step leaked a 4 MB logits buffer (1×32×32128
   floats). The port reads through the returned descriptor and `free()`s
   it each step.

**Building.** A shared library needs position-independent code, so the
model object is compiled with `-fPIC` (the source used
`llc -relocation-model=pic`):

```sh
cd part6-end-to-end-ml/src/flan_t5
sh run_mlir_pipeline.sh                 # flan_t5_linalg.mlir -> flan_t5_model.o
PYTHON=python3 sh build_flan_call.sh    # -> flan_call.cpython-312-darwin.so (or ...-linux-gnu.so)
python3 run_flan_t5_model_mlir.py
```

`build_flan_call.sh` gets include flags from `python -m pybind11
--includes` and the module suffix from `sysconfig`. On macOS it links
with `-undefined dynamic_lookup`, because Python's symbols are resolved
when the interpreter loads the module. Install `pybind11` and
`transformers` into whichever Python will import the module.

Expected result (PyTorch's own greedy `generate`, from
[`run_flan_t5_model.py`](../src/flan_t5/run_flan_t5_model.py)):

```
Input text: "translate English to German: What is your name?"
Input IDs:      [[13959, 1566, 12, 2968, 10, 363, 19, 39, 564, 58, 1]]
Output IDs:     [[0, 12625, 7, 229, 1197, 7, 10016, 58, 1]]
Decoded output: <pad> Welches ist Ihres Namen?</s>
```

Verification so far: `flan_call.cpp` and the benchmark were compiled and
**run against a stand-in `_mlir_ciface_flan_decoder`**. It is a few lines
of C++ that allocates its result as compiled code would (with
`allocated != aligned`) and plants a logit peak at a scripted token per
position. Through the real pybind11 module, Python got back
`[0, 12625, 7, 229, 1197, 7, 10016, 58, 1]`. So the harness works:
argument marshalling, the loop, descriptor offsets and strides, frees, and
the list conversion. Whether the *compiled model* produces those logits
is pending the snapshot.

## 6. When the pipeline itself needs a patch

The source's flan-t5 script piped its already-linalg file through
`torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline`
once more before `tutorial-opt`. That was harmless (the pipeline is
idempotent on linalg) and is dropped here.

The one change this port makes to the *pipeline* is the
`mlir-opt --convert-bufferization-to-memref` step between the two
`tutorial-opt` stages (Chapter 28, section 7). It is there defensively. A
model that returns a buffer it does not own (a constant, an argument, a
view whose ownership is unknown) gets a `bufferization.clone` from the
deallocation pipeline, which stage two cannot lower. On sample and mnist
the step changes nothing (checked: identical outputs). Whether any of the
four big models needs it is a question for the pending snapshots. The
clean fix is to add the pass to `--linalg-to-bufferization` itself.

## 7. Benchmarking

Every model directory now has a PyTorch and a compiled benchmark with
**the same protocol**: warm up, then average a timed loop.

| | PyTorch | compiled (C++) |
|---|---|---|
| sample, mnist, cnn | `python benchmark_<m>_model.py`: 10 warm-up + 100 timed calls | `sh benchmark.sh`: same counts |
| resnet18 | 10 + 100 | 5 + 10 (the compiled model is much slower) |
| flan_t5 | `model.generate()`: 3 warm-up + 10 timed full generations | `flan_call_benchmark.cpp`: the same greedy loop, 3 + 10, also reports per-call time |

`benchmark.sh` lowers the model if its `.o` is missing, compiles
`<m>_call_benchmark.cpp` with `-O3`, links, and runs `./bench.out`.

**Fixes relative to the source benchmarks**, each of which affected the
numbers:

- The Python scripts set `OMP_NUM_THREADS=1` *after* `import torch`,
  where it has no effect. Now it is set before the import, plus
  `torch.set_num_threads(1)`, so PyTorch really is single-threaded like
  the compiled code (BLAS libraries may still use their own threads on
  both sides). The timer is `time.perf_counter()` rather than
  `time.time()`.
- The C++ loops leaked one model-allocated result per call. They now free
  it, which also makes the timing include the result allocation that real
  use pays for.
- The source's `flan_call_benchmark.cpp` called a stale
  `_mlir_ciface_transformer_model` from an earlier export (2-D output,
  one decoder token) that no longer matches `flan_decoder`. The port times
  a full greedy generation through the same `generate_tokens()` as the
  Python module, which is comparable to `model.generate()`.
- The model objects are compiled with `-O3` (the source's `llc` used its
  default `-O2`; plain `clang -x ir` would use `-O0`).

**Source numbers.** Measured in the source repo on the RWTH cluster
(Linux, OpenBLAS, the source's pipeline and benchmark code with the
issues above). **Not re-measured here.** Average inference time in
seconds:

| Model | PyTorch (CPU) | MLIR (CPU) | MLIR (GPU, Chapter 31) |
|---|---|---|---|
| sample | 0.000023 | 0.00000068546 | 0.001746 |
| mnist | 0.000151 | 0.000853 | 0.005568 |
| cnn | 0.000319 | 0.001693 | 0.035221 |
| resnet18 | 0.045876 | 6.889084 | 0.619498 |
| flan-t5-small | 0.1178 | 0.397444 | — |

(The source's flan-t5 MLIR number timed single calls of its older export,
and its PyTorch number timed whole `generate()` runs, so that row
compares different things.)

**This machine.** Apple M3 Pro, macOS, Homebrew clang 20, Accelerate as
BLAS. PyTorch 2.12.0 (CPU, single thread), so not the container's
2.13. Compiled numbers are from this port's `benchmark.sh`, and each figure
is the typical value of three runs:

| Model | PyTorch (CPU) | MLIR (CPU) | MLIR / PyTorch |
|---|---|---|---|
| sample | 0.0000050 s | 0.00000014 s | **0.03×** (35× faster) |
| mnist | 0.000028 s | 0.00033 s | **12× slower** |
| cnn | 0.00012 s | *pending snapshot* | |
| resnet18 | 0.0145 s | *pending snapshot* | |
| flan-t5-small (full greedy generation, "How are you?" → 6 tokens) | 0.0286 s | *pending snapshot* | |

**Reading the numbers.**

- *sample*: the compiled function wins by more than an order of magnitude
  because there is almost nothing to compute (a 3×4·4×5 matmul).
  PyTorch's per-call cost is dispatcher, autograd bookkeeping and Python,
  while the compiled call is a plain function call. For tiny models,
  framework overhead *is* the runtime.
- *mnist*: the matmuls go to Accelerate in both worlds, yet the compiled
  version is ~12× slower. The rest of the work happens in loops
  `--convert-linalg-to-loops` produced: bias adds, ReLUs, and the
  `memrefCopy` of the flatten, all scalar, unfused and unvectorized, with
  a fresh `malloc` per intermediate. Finding out which of these dominates
  is exercise 1.
- *resnet18 / cnn* (source numbers): the gap explodes, 150× for resnet18,
  because **convolutions are not matmuls**. The BLAS pass never sees them,
  and a naive 7-deep loop nest over 224×224 images is exactly what
  decades of library engineering exist to avoid. The GPU column (Chapter 31)
  is ~11× faster than the CPU loops even with untuned tiles.
- *flan-t5*: PyTorch's `generate()` reuses a KV cache, so step *n* costs
  one token's work. The compiled model recomputes the whole decoder
  prefix, and with this harness's padding, all 32 positions, every step.

None of this is a defect of the tutorial. The source said it plainly: apart
from the BLAS matmul, the CPU pipeline uses naive loops, and the GPU tile
sizes are untuned. The point is to see *where* the gap comes from. The
fixes are Part 5's subject: tiling, fusion and vectorization in
[Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md),
and convolution-to-matmul rewriting (im2col).

## 8. Comparing outputs

[`compare_outputs.py`](../src/compare_outputs.py) checks two output files
number by number within an absolute tolerance (default 1e-4):

```sh
./cnn > cnn_c_out.txt
python run_cnn_model.py | grep -v shape > cnn_py_out.txt   # in the torch env
python ../compare_outputs.py cnn_c_out.txt cnn_py_out.txt [atol]
```

It extracts every number not glued to a word, so PyTorch's
`tensor([[...]], grad_fn=<LogSoftmaxBackward0>)` and a plain listing from
C++ compare cleanly. It exits 0 on a match and otherwise reports the worst
difference and how many values fall outside the tolerance. (The source
used the deprecated `np.fromstring`, which fails on any non-numeric text.)
Why a tolerance at all: Accelerate/OpenBLAS and ATen sum in different
orders, so float32 results agree to a few ulps, not bit for bit. The
sample and mnist lit tests get away with exact matching only because they
print four digits.

## 9. Known issues

Carried over from the source where they still apply, plus this port's own:

- **Snapshots pending** for all four models (see the status note at the
  top). No lit tests exist for them yet. Once the snapshots exist, cnn
  (small, full weights) is a candidate for an e2e test like `mnist_e2e.mlir`.
- **Dead arguments:** bert still carries its unused `token_type_ids`
  buffer argument. No pass removes dead arguments from a public function
  (Chapter 27, section 7). resnet18 avoids the problem only by freezing
  buffers at export time.
- **Buffers must be in sync with the binary:** `tensor.csv` must come from
  the same model revision as `bert_linalg.mlir`.
  `generate_snapshots.sh bert` writes both together.
- **GPT-2 has no C++ caller** (Chapter 28, exercise 5). The flan-t5
  harness in section 5 is the template.
- **PyTorch versions:** the source's torch-mlir pinned a PyTorch nightly
  that has aged off the index. This port pins the container to
  torch-mlir 20260531.828 + torch 2.13.0 (`requirements.txt`). The
  references above came from torch 2.12.0 on macOS. Regenerate them in the
  container if the last digit disagrees.
- **Big files:** the full bert (~880 MB), flan-t5 (~600 MB) and resnet18
  (~90 MB) exports must stay out of git, like GPT-2's.

## Where to go next

[Chapter 31](../ch31-gpu-nvvm/README.md) sends the same linalg files to an
NVIDIA GPU. Its `src/cnn/gpu/` and `src/resnet18/gpu/` scripts read the
`cnn_model_linalg.mlir` and `resnet18_model_linalg.mlir` this chapter
produces.

## Exercises

1. **Where does mnist's time go?** Build a variant of the mnist object
   without the BLAS pass (comment it out of the pipeline in a scratch copy
   of `tutorial-opt.cpp`, or lower with plain `mlir-opt` passes) and time
   both. Then profile the BLAS build (`perf` or Instruments): how much is
   `cblas_sgemm`, and how much is the elementwise loops?
2. **Free the right pointer:** in `flan_generate.h`, change
   `free(outputMemRef.allocated)` to `free(outputMemRef.aligned)`. Run
   against the stand-in model (which allocates with `allocated !=
   aligned`, just as `--finalize-memref-to-llvm` does for aligned
   allocations) and explain the result.
3. **Stop padding:** the decoder dimension is dynamic, so pass exactly
   `current_len` tokens instead of 32. What goes wrong on the first step,
   when `current_len == 1` (hint: Chapter 27, section 6, step 3), and how
   would you work around it? Measure the speedup.
4. **Freeze bert:** apply `freeze_buffers()` to bert (Chapter 27,
   exercise 7), drop `get_buffers.py`, and simplify `bert_call.cpp`
   accordingly.
5. **Run-time input for resnet18:** extend `resnet18_call.cpp` to read a
   raw 3×224×224 float file (normalize a real image in Python with the
   `AutoImageProcessor`), then check that the predicted class matches
   PyTorch's.
6. **Convolutions to BLAS:** the cnn's convolutions stay as loops. Sketch
   how an im2col rewrite would turn `linalg.conv_2d_nchw_fchw` into a
   `linalg.matmul` that Chapter 29's pass then catches. Upstream MLIR has a
   transform for this; find it.
