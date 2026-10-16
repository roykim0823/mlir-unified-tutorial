# Chapter 27: Importing PyTorch Models with torch-mlir

> Reworked from ML-compiler-exercise chapters 1–3 for the unified repo.

Until now we compiled IR that we wrote, or that a fifty-line Python DSL
wrote; Chapters 25–26 hand-wrote even the neural-network layers.
This chapter starts the capstone: taking a model *someone actually trained
in PyTorch* and producing the linalg-on-tensors IR that the rest of the
descent (Chapters 28–31) consumes. The importer is
[torch-mlir](https://github.com/llvm/torch-mlir), the LLVM-incubator bridge
between PyTorch and MLIR, and the workflow is three models of increasing
difficulty: a toy module, the MNIST MLP, and GPT-2 with a dynamic sequence
length.

## What you will learn

- The import path: `torch.export` → FX graph → torch dialect →
  linalg-on-tensors, and what each level looks like on a real model
- `fx.export_and_import` and its `OutputType`s (RAW, TORCH,
  LINALG_ON_TENSORS)
- Where the weights go: `dense_resource` blobs and why the exported file is
  as big as the model
- What breaks on a real transformer — dictionary outputs, KV caches,
  optional kwargs — and the wrapper pattern that fixes it
- Dynamic dimensions with `torch.export.Dim`, and the two-step lowering
  used for models the direct path can't handle

## Prerequisites

- [Chapter 22](../../part5-structured-and-parallel/ch22-linalg-transform-dialect/README.md)
  (or [Chapter 8](../../part1-using-mlir/ch08-linalg-by-hand/README.md)):
  you must be able to read `linalg.generic` — the importer's output is
  exactly that.
- Recommended: [Chapter 25](../ch25-neural-nets-from-scratch/README.md) and
  [Chapter 26](../ch26-transformers-and-softmax/README.md), which write a
  dense layer and a softmax by hand. The importer emits the same ops, so
  you will recognise them in its output.
- The Part 6 environment: see the [part README](../README.md). torch-mlir
  wheels are Linux-only; on macOS run the scripts via
  `./generate_snapshots.sh`. Everything this chapter *prints* is also
  checked in under [`src/`](../src/), so you can follow along with no torch
  install at all.

All files: [`src/sample/`](../src/sample/), [`src/mnist/`](../src/mnist/),
[`src/gpt2/`](../src/gpt2/).

## 1. The sample model

The whole pipeline is easiest to watch on a module small enough to print
([`src/sample/model.py`](../src/sample/model.py)):

```python
class Sample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)
```

Add a learned bias, apply a linear layer (y = xWᵀ + b), clamp to [0, 1].
Input 3×4, output 3×5. [`run_sample_model.py`](../src/sample/run_sample_model.py)
executes it in PyTorch on a tensor of ones and prints the reference output
we will hold every compiled artifact to:

```
Output: tensor([[0.3844, 0.0000, 0.2425, 0.2203, 0.0000],
        [0.9251, 0.0000, 0.6383, 0.4868, 0.0000],
        [0.9336, 0.0000, 0.3170, 0.4453, 0.0000]], grad_fn=<ClampBackward1>)
```

(A reproducibility detail that cost this port a debugging session: the
weights are random, so **every** script re-seeds with
`torch.manual_seed(41)` immediately before constructing the model. Seeding
once per file is not enough — a second `Sample()` in the same process draws
*different* weights, and your compiled model silently disagrees with your
reference.)

## 2. `fx.export_and_import`

The recommended torch-mlir API is the FX-based exporter
(`torch_mlir.compile()` is deprecated). The entire import is one call
([`lower_sample_model.py`](../src/sample/lower_sample_model.py)):

```python
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

m = fx.export_and_import(
    Sample(),
    torch.randn(3, 4),                       # example input: shapes only
    output_type=OutputType.LINALG_ON_TENSORS,
    func_name="sample_model",
)
```

You pass the model, an example input (used by `torch.export` to trace —
its *values* don't matter, its *shapes and dtypes* do), the target
abstraction level, and the name the generated `func.func` gets.
`OutputType` selects how far down the importer lowers:

- **RAW** — the FX graph translated op-for-op, before any simplification.
- **TORCH** — the **torch dialect**: MLIR ops named `torch.aten.*`,
  mirroring PyTorch's ATen operator set, on `!torch.vtensor` types. This is
  the "backend contract" form other compilers hook into.
- **LINALG_ON_TENSORS** — no torch ops left: just linalg/arith/tensor, the
  upstream dialects Part 5 taught.

## 3. Reading the two snapshots

The torch-dialect output
([`sample_model_torch.mlir`](../src/sample/sample_model_torch.mlir), checked
in) is PyTorch semantics made explicit — you can read `forward()` line by
line:

```mlir
func.func @sample_model(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,5],f32> {
  ...
  %3 = torch.aten.add.Tensor %arg0, %2, %int1 : ... -> !torch.vtensor<[3,4],f32>
  %4 = torch.aten.transpose.int %1, %int0, %int1 : ... -> !torch.vtensor<[4,5],f32>
  %5 = torch.aten.mm %3, %4 : ... -> !torch.vtensor<[3,5],f32>
  %6 = torch.aten.add.Tensor %5, %0, %int1 : ... -> !torch.vtensor<[3,5],f32>
  %7 = torch.aten.clamp %6, %float0.000000e00, %float1.000000e00 : ... -> !torch.vtensor<[3,5],f32>
  return %7 : !torch.vtensor<[3,5],f32>
}
```

`x + param` became `aten.add.Tensor`; `Linear` decomposed into
`aten.transpose.int` + `aten.mm` + a bias `aten.add.Tensor`; `clamp` stayed
`aten.clamp`. Note the types: `!torch.vtensor<[3,5],f32>` is a
*value-semantics* tensor — the importer has already converted PyTorch's
in-place, aliasing world into SSA.

The linalg-on-tensors output
([`sample_model_linalg.mlir`](../src/sample/sample_model_linalg.mlir)) is
the same program in Chapter 22's vocabulary — worth reading in full, since
Chapters 28–31 start from it:

```mlir
func.func @sample_model(%arg0: tensor<3x4xf32>) -> tensor<3x5xf32> {
  %cst   = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>
  %cst_1 = arith.constant dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>
  %cst_2 = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
  ...
  %1 = linalg.generic {...} ins(%arg0, %cst_2 : ...) ...   // x + param
  %transposed = linalg.transpose ins(%cst_1 : tensor<5x4xf32>) ... permutation = [1, 0]
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<3x5xf32>) ...
  %5 = linalg.matmul ins(%1, %transposed : ...) outs(%4 : ...) ...
  %6 = linalg.generic {...}                                 // + bias (broadcast #map1)
  %7 = linalg.generic {...} {
    %8 = arith.cmpf ult, %in, %cst : f32                    // clamp, as cmpf+select
    %9 = arith.select %8, %cst, %in : f32
    %10 = arith.cmpf ugt, %9, %cst_3 : f32
    %11 = arith.select %10, %cst_3, %9 : f32
    linalg.yield %11 : f32
  } -> tensor<3x5xf32>
  return %7 : tensor<3x5xf32>
}
```

Everything Chapter 22 taught is here in the wild: the elementwise add as a
`linalg.generic` with identity maps, the bias add broadcasting via
`#map1 = affine_map<(d0, d1) -> (d1)>`, `linalg.fill` zero-initializing the
matmul accumulator.

**Where the weights went.** The trailing block of the file is the
`dialect_resources` section:

```mlir
{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_torch.float32: "0x040000006061C8BC30FC8ABD...",
      ...
    }
  }
#-}
```

Every parameter tensor is embedded in the IR as a named hex blob that the
`dense_resource` constants reference. The program *is* the model, weights
included — which is why GPT-2's export below is a ~950 MB text file, and
why this repo checks in a stripped variant for reading.

## 4. MNIST: same recipe, real shape

[`src/mnist/model.py`](../src/mnist/model.py) is the classic MLP —
flatten 28×28, then Linear(784→512), ReLU, Linear(512→512), ReLU,
Linear(512→10). The export
([`lower_mnist_model.py`](../src/mnist/lower_mnist_model.py)) is identical
to the sample's, with input `torch.randn(1, 1, 28, 28)`. The linalg
snapshot ([`mnist_model_linalg.mlir`](../src/mnist/mnist_model_linalg.mlir),
5.1 MB — those 670k weights) contains three `linalg.matmul` ops, and the
PyTorch reference logits for a ones input are:

```
Output: tensor([[ 0.0149, -0.0860,  0.1660, -0.1409, -0.1119, -0.0749, -0.1015, -0.1354,
         -0.0231,  0.0546]], grad_fn=<AddmmBackward0>)
```

Ten numbers you will meet again, digit for digit, coming out of a native
binary in Chapter 28.

## 5. GPT-2: wrappers, dynamic shapes, and the two-step lowering

A HuggingFace transformer will not export as-is. Its `forward()` takes a
pile of optional keyword arguments, returns a dictionary-like object
carrying caches and hidden states, and maintains a KV cache — side state
that `torch.export` cannot represent as a pure function. The fix is a
wrapper that pins the interface down to tensors
([`lower_gpt2_model.py`](../src/gpt2/lower_gpt2_model.py)):

```python
class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
```

plus `model.config.use_cache = False` to disable the KV cache entirely.
GPT-2 is a *causal* (decoder-only) model: to generate, you call it, take
the logits of the last position, append the winning token to `input_ids`,
and call it again. So the sequence length must be a **dynamic dimension**:

```python
seq_len = torch.export.Dim("seq_len", max=128)
ep = torch.export.export(
    wrapped,
    (encoded["input_ids"], encoded["attention_mask"]),
    dynamic_shapes={
        "input_ids": {1: seq_len},        # dim 1 is dynamic...
        "attention_mask": {1: seq_len},   # ...and shared between both inputs
    },
)
ep = ep.run_decompositions()
```

`torch.export.Dim` names the symbolic dimension; using the same object for
both inputs tells the exporter they are equal. `run_decompositions()` then
breaks composite ATen ops into primitives the importer knows.

For the final descent this chapter uses the **two-step** path — import to
TORCH, then run the backend pipeline explicitly:

```python
m = fx.export_and_import(ep, output_type=OutputType.TORCH, func_name="gpt2_model")
run_pipeline_with_repro_report(
    m, "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
    "lowering torch backend contract to linalg-on-tensors")
```

which is exactly what `OutputType.LINALG_ON_TENSORS` runs internally. The
source repo needed this split because the direct path died on
`failed to legalize operation 'torch.constant.int'` for bert/gpt2 (and
bert additionally needed a patched torch-mlir); with the 2026 wheels the
two-step path works out of the box. The split is worth keeping anyway: it
gives you the torch-dialect file as a debugging waypoint.

The result — this is real, from the checked-in stripped snapshot
([`gpt2_linalg_stripped.mlir`](../src/gpt2/gpt2_linalg_stripped.mlir)):

```mlir
func.func @gpt2_model(%arg0: tensor<1x?xi64>, %arg1: tensor<1x?xi64>)
    -> tensor<1x?x50257xf32> {
  %dim = tensor.dim %arg1, %c1 : tensor<1x?xi64>
  ...
```

The dynamic dimension survived as `?`, queried at runtime with
`tensor.dim`; the output is logits over GPT-2's 50,257-token vocabulary.
The full function is ~5,500 lines: 49 `linalg.matmul`, 24
`linalg.batch_matmul` (12 attention layers × 2), 786 `linalg.generic` —
and ~950 MB of file, nearly all of it the weight resources. (A drift note:
the source repo complained about 20+ unused `tensor<1x1x1024x1024xi1>`
mask arguments in GPT-2's signature; current torch-mlir exports the clean
two-argument signature you see above.)

For reference, PyTorch's greedy decoding
([`run_gpt2_model.py`](../src/gpt2/run_gpt2_model.py)), 10 steps from "The
capital of France is":

```
The capital of France is the capital of the French Republic, and the capital
```

Vintage GPT-2. The point of the capstone is the compiler, not the poetry.

## 6. An encoder-decoder: FLAN-T5

GPT-2 is decoder-only, so one dynamic dimension was enough. Google's
**flan-t5-small** is an *encoder-decoder*: the encoder reads the prompt
once, and the decoder generates the answer token by token while attending
to the encoder's output. Its raw interface is the HuggingFace kitchen sink:

```python
model(input_ids=..., attention_mask=..., decoder_input_ids=..., labels=...,
      output_hidden_states=..., output_attentions=..., return_dict=True, ...)
```

and it returns a `Seq2SeqLMOutput` carrying `encoder_last_hidden_state`,
`cross_attentions`, caches, and more. `torch.export` copes badly with
exactly these things: several optional keyword arguments,
dictionary-style outputs, side data (states, caches, hidden states), and
multiple heads. The fix is the GPT-2 fix, one input wider
([`src/flan_t5/lower_flan_t5_model.py`](../src/flan_t5/lower_flan_t5_model.py)):

```
(input_ids, attention_mask, decoder_input_ids) → logits
```

```python
class T5Wrapper(torch.nn.Module):
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids)
        return outputs.logits      # (batch, decoder_len, vocab=32128)
```

`export_flan_t5()` then goes through six steps:

1. Load the tokenizer and encode an example prompt
   (`"translate English to German: How are you?"`) to `input_ids` and
   `attention_mask`, the sample inputs `torch.export` traces with.
2. Load the model, set `use_cache = False` (the KV cache is side state
   again), and wrap it.
3. Build a decoder start sequence of **two** `<pad>` tokens (id 0, T5's
   decoder start token). Two, not one, because `torch.export`
   *specializes* any dimension whose example size is 0 or 1. Trace with a
   length-1 decoder input and that dimension is baked in as 1, whatever
   `dynamic_shapes` says.
4. Call `torch.export.export` with dynamic-shape specs: batch fixed at 1,
   one `Dim` for the encoder length shared by `input_ids` and
   `attention_mask` (≤ 128), and a second `Dim` for the decoder length
   (≤ 32), which grows by one every generation step:

   ```python
   encoder_seq_len = torch.export.Dim("encoder_seq_len", max=128)
   decoder_seq_len = torch.export.Dim("decoder_seq_len", max=32)
   ep = torch.export.export(
       wrapped,
       (encoded["input_ids"], encoded["attention_mask"], decoder_start),
       dynamic_shapes={
           "input_ids": {1: encoder_seq_len},
           "attention_mask": {1: encoder_seq_len},
           "decoder_input_ids": {1: decoder_seq_len},
       },
   )
   ```
5. `ep.run_decompositions()`, as for GPT-2.
6. `fx.export_and_import(ep, output_type=OutputType.LINALG_ON_TENSORS,
   func_name="flan_decoder")`. T5 imports straight to linalg. We called
   `torch.export` ourselves only to pass `dynamic_shapes` and inspect the
   program; handed an `nn.Module`, `export_and_import` would run it
   internally.

(Drift notes: the source tutorial's prose gives the bounds as "encoder up to
10, decoder ≤ 16" but its code uses 128 and 32. This port follows the
code. The source also loaded `T5Tokenizer`; the port uses `AutoTokenizer`,
which needs no `sentencepiece`.)

Checked on the torch side only (torch 2.12 + transformers 5.19 on macOS,
since torch-mlir cannot run here): the export succeeds with exactly three
user inputs, no buffers, and range constraints `[0, 128]` / `[0, 32]`.
`model.generate` translates `"translate English to German: What is your
name?"` to token ids `[0, 12625, 7, 229, 1197, 7, 10016, 58, 1]`, which
decode to `<pad> Welches ist Ihres Namen?</s>`. Running the wrapper
greedily on a decoder input zero-padded to 32 tokens yields **the same
ids**. Chapter 30's C++ harness relies on that.

**Status:** the `flan_t5_linalg.mlir` export itself is pending snapshot
generation (`./generate_snapshots.sh flan_t5`).

## 7. Lifted buffers and the dead-argument problem

`torch.export` turns all module state into graph **inputs** (placeholders)
and records what each one is in `ep.graph_signature`:
`inputs_to_parameters`, `inputs_to_buffers`, and
`inputs_to_lifted_tensor_constants`. Parameters are weights and biases.
*Buffers* are state that is not trained: BatchNorm's `running_mean`,
`running_var` and `num_batches_tracked`, or BERT's precomputed
`position_ids`. torch-mlir's `fx.export_and_import` then calls
`FxImporter.import_frozen_program`, which is meant to inline that state as
`dense_resource` literals. Here is the relevant part of the 2026 wheel
(`torch_mlir/extras/fx_importer.py`, torch-mlir 20260531.828):

```python
if hasattr(prog, "constants"):
    constants = prog.constants
    # Lift tensor constants.
    for input_name, state_name in sig.inputs_to_lifted_tensor_constants.items():
        ...
else:
    # Lift buffers.
    for input_name, state_name in sig.inputs_to_buffers.items():
        ...
# Lift parameters.
for input_name, state_name in sig.inputs_to_parameters.items():
    ...
```

The buffer branch runs only on PyTorch versions *without*
`ExportedProgram.constants`, which every PyTorch since early 2024 has. So
on any current PyTorch, **parameters and constants are frozen, buffers are
not**: they stay function arguments, ahead of the user inputs, in module
order. torch-mlir's own
[resnet18 example](https://github.com/llvm/torch-mlir/blob/main/projects/pt1/examples/fximporter_resnet18.py)
extracts the buffers and passes them in for the same reason. The source
tutorial hit this behavior and the 2026 wheel still has it. To get the
signatures without torch-mlir, mirror that logic on the exported program
(torch 2.12):

| Model | `torch.export` placeholders | Function args after import | Of which unused |
|---|---|---|---|
| resnet18 (HF `microsoft/resnet-18`) | 123 (62 params, 60 buffers, 1 input) | **61**: 60 BatchNorm buffers + the image | **20**: every third, `num_batches_tracked` (`si64` scalar, read only in training) |
| bert-base-uncased | 204 (199 params, 2 buffers, 3 inputs) | **5**: `position_ids`, `token_type_ids` (both `[1,512]` i64), then the three inputs | **1**: the `token_type_ids` buffer, because we pass explicit token type ids |
| flan-t5-small | 3 user inputs, no buffers | **3** | 0 |

(The source tutorial says resnet18 grows "from 1 to over 500 inputs".
The count measured here is 61. Its observation that *every third*
argument is an unused `!torch.vtensor<[],si64>` matches exactly.)

This also explains a drift note in section 5. The source's GPT-2 export
had 20+ unused `tensor<1x1x1024x1024xi1>` arguments: GPT-2's causal-mask
`attn.bias` **buffers**, one per layer. Current `transformers` no longer
registers that buffer (it builds masks on the fly), so the 2026 export has
the clean two-argument signature. The importer is unchanged; the model
changed. It is also why the source repo's `src/gpt/get_buffers.py` (which
dumped those buffers to a CSV, like bert's) has no counterpart here: there
is nothing left to dump.

There are three ways to deal with buffers, and this part uses all three:

- **Paste them in by hand** (the source's resnet18 method):
  [`get_buffers_in_mlir_format.py`](../src/resnet18/get_buffers_in_mlir_format.py)
  dumps each buffer as a resource hex blob (`"0x04000000..."`, a 4-byte
  alignment header followed by the raw little-endian floats) and as a
  `%argN = torch.vtensor.literal(dense_resource<...>)` line. You paste the
  blobs into `dialect_resources`, put the literals at the top of the
  function body, and delete the arguments from the signature.
- **Pass them in at run time** (bert): [`get_buffers.py`](../src/bert/get_buffers.py)
  writes the two buffers to `tensor.csv`, and the C++ caller loads and
  passes them (Chapter 30).
- **Freeze them before export** (this port's resnet18):
  `freeze_buffers()` in
  [`lower_resnet18_model.py`](../src/resnet18/lower_resnet18_model.py)
  re-registers every buffer as a `requires_grad=False` `Parameter`, which
  moves it into the category the importer *does* freeze, and drops
  `num_batches_tracked`. Checked with torch 2.12: the export is left with
  the single `pixel_values` input, and the logits are bit-identical to the
  unfrozen model's.

**Dead arguments.** The source notes that no pass in `torch-mlir-opt` or
`mlir-opt` removes them. Upstream MLIR does have `--remove-dead-values`,
but it does not help here, and the reason is instructive. On this repo's
wheel it drops an unused argument from a *private* function (it can
rewrite every call site) and leaves a *public* one's signature alone
(verified on a two-function example). The exported model function is
public, and its callers live in C++, outside the module.

## 8. The two-step lowering on the command line, and `aten.any.dim`

Section 5 ran the TORCH → linalg step inside Python with
`run_pipeline_with_repro_report`. The source tutorial ran it as a separate
tool invocation instead. That works from any torch-dialect file and needs
no Python, but does need `torch-mlir-opt` (shipped by source builds of
torch-mlir; the Linux wheels expose the same pipeline through Python only):

```sh
torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline \
    bert_torch.mlir > bert_linalg.mlir
```

The source's bert, gpt2 and resnet18 pipeline scripts all started with
that line. Why bert needed it at all is a small story about how lowering
failures surface. The direct LINALG_ON_TENSORS import died with

```
error: failed to legalize operation 'torch.constant.int'
```

which looks like a problem with integer constants. The real culprit was
upstream of it. BERT's attention-mask handling contains `aten.any.dim`
(still present in the exported graph on torch 2.12; checked), and the
TorchToLinalg reduction patterns of the time had no lowering for it. The
`aten.any.dim` op survived, so the `torch.constant.int` feeding its `dim`
operand stayed alive too, and since constants are only legalized once
their users are, *that* is the op the conversion reported. The source
fixed it by adding `AtenAnyDimOp` to the reduction lowering in
`lib/Conversion/TorchToLinalg/Reduction.cpp` of a
[torch-mlir fork](https://github.com/DavidGinten/torch-mlir/blob/97bf9d2e6313abae5eb890748207baa178caeddf/lib/Conversion/TorchToLinalg/Reduction.cpp),
which is why its submodule pointed there. Even patched, only the two-step
path worked for bert.

This is history now. The 2026 wheel's TorchToLinalg contains the
reduction-info instantiation for `AtenAnyDimOp`, found by searching the
symbols of `libTorchMLIRAggregateCAPI.so`
(`ConvertReductionOp::computeReductionOpInfoForDimVariantOp<AtenAnyDimOp>`),
so stock torch-mlir lowers `aten.any.dim`. The bert script still uses the
two-step path, to match GPT-2 and to keep the torch-dialect file as a
waypoint. Whether the direct path now works for bert too is untested here
(exercise 6). The lesson generalizes: in dialect conversion, **the op an
error names is often a victim, not the culprit**. Look for the
unconverted op that keeps it alive.

## Where to go next

You now hold `*_linalg.mlir` files for three real models.
[Chapter 28](../ch28-cpu-pipeline-and-abi/README.md) lowers them to native
code and calls them from C++ — and checks the numbers against the
references above.

## Exercises

1. Run `lower_sample_model.py` with `output_type=OutputType.RAW` and
   compare against the TORCH output: what did the backend-contract
   simplification remove?
2. Export the sample model with a *dynamic batch dimension* (Dim on dim 0
   of the input). What changes in the linalg signature, and which new ops
   appear in the body?
3. The mnist export's `flatten` becomes which tensor op in
   `mnist_model_linalg.mlir`? (Look for `collapse_shape`.) Why does this
   op have no computation in it?
4. The `cnn` model ([Chapter 30](../ch30-real-models/README.md)) is
   mnist-plus-convolutions. Export it (`./generate_snapshots.sh cnn`) and
   find which linalg op in `cnn_model_linalg.mlir` carries the convolution.
5. In the GPT-2 export, change `Dim("seq_len", max=128)` to `max=1024` and
   re-export. Does the IR change at all? Where does the bound actually
   live (hint: look for `cf.assert` / guards near the top of the
   function)?
6. Change `lower_bert_model.py` to import directly with
   `OutputType.LINALG_ON_TENSORS`. Does the 2026 wheel get through
   `aten.any.dim` (section 8)? If it fails, which op does the error name,
   and which op is really to blame?
7. Apply `freeze_buffers()` from `lower_resnet18_model.py` to bert before
   exporting. How many arguments does `bert_model` have now, and what would
   change in `bert_call.cpp`? (Careful: `position_ids` is int64. Check
   that a `requires_grad=False` int Parameter is accepted.)
8. Export flan-t5 with a one-token decoder start (`[[pad]]`) instead of
   two, keeping the `dynamic_shapes` spec. What does `torch.export` do,
   and why?
