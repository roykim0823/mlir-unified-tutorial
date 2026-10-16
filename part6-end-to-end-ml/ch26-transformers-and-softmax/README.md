# Chapter 26: Transformers and a Compiled Softmax

> Adapted from mlir-ir-lowering's `7_transformer` chapter, itself based on
> Stephen Diehl's *"MLIR Part 7 — Transformers"* article. That article is a
> ~100-line **NumPy** GPT-2 with no MLIR; the source chapter reproduced it as
> a runnable model **and** compiled the softmax — attention's defining op —
> to MLIR.

## What you will learn

- The architecture of a **decoder-only transformer** (GPT-2), and why every
  piece of it is an op you already know: matmuls, elementwise activations,
  bias adds, reductions, residual adds.
- **Self-attention**: Q/K/V projections, scaled dot-product scores, causal
  masking, softmax, weighted sum — and multiple heads.
- How the whole GPT-2 forward pass reads in ~60 lines of NumPy, and how to run
  it on a tiny random-weight configuration.
- How to write a **numerically stable softmax** in MLIR with three `scf.for`
  passes per row, two of them reductions carried by `iter_args`, and
  `math.exp` lowered to a libm call — compiled and checked against NumPy.
- How the softmax you wrote by hand shows up, op for op, in the real GPT-2
  that Chapter 27 imports from PyTorch.

## Prerequisites

- [Chapter 25](../ch25-neural-nets-from-scratch/README.md): dense layers
  (`x @ w + b`), activations, and the build-and-check workflow for a
  hand-written kernel.
- [Chapter 8](../../part1-using-mlir/ch08-linalg-by-hand/README.md): the
  linalg ops a transformer is made of (matmul, elementwise generics,
  reductions).
- Part 1's loop chapters ([Chapter 6](../../part1-using-mlir/ch06-affine-maps-and-loop-transforms/README.md),
  [Chapter 7](../../part1-using-mlir/ch07-parallel-loops-and-openmp/README.md)):
  loop-carried values with `iter_args`.
- [Chapter 4](../../part1-using-mlir/ch04-memory-and-the-c-abi/README.md):
  the memref descriptor and `_mlir_ciface_` wrappers used by the driver.
- The repo's `.venv` (NumPy only). **No torch and no weight download** — the
  demo uses a tiny random-weight config built in-process.

All code lives in [`examples/`](examples/):

```
examples/
├── gpt2/model.py        # the NumPy GPT-2 forward pass (the article's content)
├── demo.py              # run it on a tiny random-weight config
└── mlir_attention/      # softmax compiled to MLIR
    ├── softmax.mlir
    ├── build.sh
    ├── aot_main.py        # NumPy driver
    └── softmax_main.c     # C driver (used by the lit test)
```

## The architecture this whole part is aiming at

This is the model Part 6 builds toward: a **decoder-only transformer**
(GPT-2). And here is the encouraging part — by now you have already met every
piece. A transformer is not one exotic operation; it is a *graph of the
primitives from earlier chapters*: matrix multiplies (Chapter 8), elementwise
activations and bias adds (Chapters 8 and 25), reductions and softmaxes, and
residual adds — stacked a dozen times. The "magic" is purely in how they are
wired.

The one genuinely new idea is **self-attention**: when processing a token,
the model dynamically weights how much to "attend" to every other token.
Processing *"bank"* in *"deposit money at the bank"*, attention lets it lean
on *"deposit"* and *"money"* to read "bank" as a financial institution.
Mechanically that is three projections (Query, Key, Value), a scaled
dot-product `QKᵀ/√d`, a **softmax** to turn scores into weights, and a
weighted sum of Values — all matmuls and a softmax.

## Part A — GPT-2 in NumPy (`gpt2/model.py`)

The full forward pass, every function mapping to ops you already know:

| Component | What it is | Built from |
| --- | --- | --- |
| token + position embeddings | look up `wte[token] + wpe[pos]` | gather + add |
| `layer_norm` | normalize across features, scale + shift | mean/var reductions |
| `linear` | `x @ w + b` | **matmul + bias** (Ch 8/25) |
| `gelu` | smooth activation | elementwise |
| `ffn` | `linear → gelu → linear` | two dense layers (Ch 25) |
| `softmax` | scores → probabilities | exp + row-sum reduction |
| `attention` | `softmax(QKᵀ/√d + mask) · V` | **matmul + softmax + matmul** |
| `mha` | run attention in parallel heads, concat, project | the above × heads |
| `transformer_block` | pre-LN → MHA → residual; pre-LN → FFN → residual | all of it |
| `gpt2` | embed → N blocks → final LN → project to vocab | the stack |

Notice how directly the code reads as that graph — `attention` is literally
its one-line formula, every operation something from Chapters 8 and 25:

*examples/gpt2/model.py* (excerpt — attention and its softmax)
```python
def softmax(x):
  """Row-wise softmax (numerically stable). Turns scores into probabilities."""
  ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return ex / np.sum(ex, axis=-1, keepdims=True)

# ... layer_norm, linear, ffn ...

def attention(q, k, v, mask):
  """Scaled dot-product attention: softmax(QKᵀ/√d + mask) · V."""
  scores = (q @ k.T) / np.sqrt(q.shape[-1]) + mask
  return softmax(scores) @ v
```

```text
   Q ─┐                                                  weighted
      ├─► Q·Kᵀ ─► ÷√d ─► + mask ─► softmax ─► weights ──►  sum   ─► out
   K ─┘   scores     scale    hide      row-wise         · V
   V ────────────────────────future    probabilities
        (matmul)                       (Part B compiles this)   (matmul)
```

The rest of the model is just as literal:

*examples/gpt2/model.py* (excerpt — the other building blocks)
```python
def gelu(x):
  """Gaussian Error Linear Unit (tanh approximation) — the FFN non-linearity."""
  return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def layer_norm(x, g, b, eps=1e-5):
  """Normalize across the feature dim, then scale (g) and shift (b)."""
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return g * (x - mean) / np.sqrt(var + eps) + b

def ffn(x, c_fc, c_proj):
  """Position-wise feed-forward: expand -> GELU -> project back."""
  return linear(gelu(linear(x, *c_fc)), *c_proj)

def mha(x, c_attn, c_proj, n_head):
  """Multi-head attention: project to Q,K,V, split into heads, attend, merge."""
  x = linear(x, *c_attn)                                   # [N, 3*d]
  qkv = np.split(x, 3, axis=-1)                            # 3 x [N, d]
  qkv_heads = [np.split(part, n_head, axis=-1) for part in qkv]  # 3 x n_head x [N, d/h]
  n = x.shape[0]
  causal = (1 - np.tri(n, dtype=x.dtype)) * -1e10          # hide future tokens
  heads = [attention(q, k, v, causal) for q, k, v in zip(*qkv_heads)]
  x = np.hstack(heads)                                     # [N, d]
  return linear(x, *c_proj)

def transformer_block(x, blk: Block, n_head):
  # pre-LN attention sub-layer with a residual connection
  x = x + mha(layer_norm(x, blk.ln1_g, blk.ln1_b),
              blk.attn_c_attn, blk.attn_c_proj, n_head)
  # pre-LN feed-forward sub-layer with a residual connection
  x = x + ffn(layer_norm(x, blk.ln2_g, blk.ln2_b),
              blk.mlp_c_fc, blk.mlp_c_proj)
  return x

def gpt2(inputs, model: Model, n_head):
  """Forward pass: token IDs -> logits over the vocabulary for each position."""
  x = model.wte[inputs] + model.wpe[range(len(inputs))]   # embed tokens + positions
  for blk in model.blocks:                                # the transformer stack
    x = transformer_block(x, blk, n_head)
  x = layer_norm(x, model.lnf_g, model.lnf_b)             # final layer norm
  return x @ model.wte.T                                  # project back to vocab logits
```

Note the last line: GPT-2 **ties** its output projection to the token
embedding table (`wte.T`), so there is no separate vocab matrix. The
parameters of one block live in two dataclasses, `Block` and `Model`, whose
field names (`ln1_g`, `attn_c_attn`, `mlp_c_fc`, …) mirror the names in the
real GPT-2 checkpoint.

Two transformer-specific details worth calling out:

- **Causal masking.** A language model must not peek at future tokens, so
  before the softmax we add a mask that sets "future" scores to `-1e10` —
  after softmax those weights become ~0. (`mha` builds it with `np.tri`; for
  four tokens it is the strictly upper triangle:)

  ```
  [[-0.e+00 -1.e+10 -1.e+10 -1.e+10]
   [-0.e+00 -0.e+00 -1.e+10 -1.e+10]
   [-0.e+00 -0.e+00 -0.e+00 -1.e+10]
   [-0.e+00 -0.e+00 -0.e+00 -0.e+00]]
  ```

- **Multiple heads.** The `d=768` embedding is split into 12 heads of 64,
  each attending independently ("from different perspectives"), then
  concatenated back — `768 / 12 = 64`.

### Running it

[`demo.py`](examples/demo.py) runs the whole thing on a **tiny random-weight**
config. Real GPT-2 small is `d=768, heads=12, layers=12, vocab=50257` and its
weights are hundreds of MB; the demo instead builds `d=12, heads=3,
layers=2, vocab=32, ctx=16, d_ff=48` with small seeded random weights
(layer-norm gains 1, biases 0) and **downloads nothing**. With random weights
the prediction is meaningless, but the plumbing — embeddings, the block
stack, attention, FFN, residuals, layer norm, and the vocab projection — is
exactly the real architecture:

```sh
cd part6-end-to-end-ml/ch26-transformers-and-softmax/examples
python demo.py
```
```
GPT-2-style forward pass: 5 tokens, d=12, heads=3, layers=2
logits shape: (5, 32)  (sequence_length x vocab)
next-token distribution sums to 1.0000; argmax -> token 9
Transformer forward pass successful!
```

The last row of the logits predicts the *next* token, so the demo softmaxes
it and takes the argmax — one step of the greedy decoding loop that
Chapter 27 runs with real weights.

## Part B — softmax, compiled (`mlir_attention/`)

Attention's defining step is the **softmax** over scores. It is also the
most interesting transformer op to compile, because it is not purely
elementwise — it needs two **reductions** per row (a max for stability, then
a sum) with passes between them:

```
out[i,j] = exp(x[i,j] - max_j x[i,:]) / Σ_j exp(x[i,:] - max_j x[i,:])
```

Why subtract the max? `exp` of a large number overflows: a row of scores
around 1000 gives `exp(1000) = inf` in float32, and `inf / inf = nan`.
Softmax is invariant to shifting a row by a constant, so subtracting the row
max changes nothing mathematically while keeping every exponent `≤ 0`:

```
naive  exp(x)/sum: [[nan nan nan]]
stable softmax:    [[0.09003057 0.24472848 0.66524094]]     for x = [1000, 1001, 1002]
```

[`softmax.mlir`](examples/mlir_attention/softmax.mlir) implements it with
three sequential `scf.for` passes per row — a `maximumf` accumulator (the
row max, subtracted for numerical stability), then an `addf` accumulator
over `math.exp`, then a divide — the same stable formula as the NumPy
`softmax` above, lowered to native code:

*examples/mlir_attention/softmax.mlir*
```mlir
func.func @softmax(%x: memref<?x?xf32>, %out: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %N    = memref.dim %x, %c0 : memref<?x?xf32>
  %M    = memref.dim %x, %c1 : memref<?x?xf32>
  %ninf = arith.constant -3.40282347E+38 : f32   // -FLT_MAX, the max identity
  %zero = arith.constant 0.0 : f32

  scf.for %i = %c0 to %N step %c1 {
    // 1. row maximum
    %mx = scf.for %j = %c0 to %M step %c1 iter_args(%m = %ninf) -> (f32) {
      %v  = memref.load %x[%i, %j] : memref<?x?xf32>
      %mm = arith.maximumf %m, %v : f32
      scf.yield %mm : f32
    }
    // 2. exp(x - max), stored into out, accumulating the row sum
    %sum = scf.for %j = %c0 to %M step %c1 iter_args(%s = %zero) -> (f32) {
      %v = memref.load %x[%i, %j] : memref<?x?xf32>
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      memref.store %e, %out[%i, %j] : memref<?x?xf32>
      %ss = arith.addf %s, %e : f32
      scf.yield %ss : f32
    }
    // 3. normalize the row by its sum
    scf.for %j = %c0 to %M step %c1 {
      %e = memref.load %out[%i, %j] : memref<?x?xf32>
      %r = arith.divf %e, %sum : f32
      memref.store %r, %out[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
```

The two `iter_args` accumulators are loop-carried values again (the
`maximumf`/`addf` reductions): each iteration receives the running value as
a block argument and hands the next one over with `scf.yield`, and the loop's
result is the final value. Step 2 also reuses `out` as scratch space for the
unnormalized exponentials, so the kernel needs no temporary buffer.

### What `math.exp` turns into

`--convert-math-to-llvm` rewrites `math.exp` into the LLVM intrinsic
`llvm.intr.exp`, and `arith.maximumf` into `llvm.intr.maximum` (output of
the first few passes on the current wheel):

```mlir
    %7 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
    %17 = llvm.intr.maximum(%13, %16) : (f32, f32) -> f32
    %27 = llvm.intr.exp(%26) : (f32) -> f32
```

The backend then lowers `@llvm.exp.f32` to a call to libm's `expf` — which is
why the shared library has one undefined symbol (`nm build/libsoftmax.so`):

```
0000000000000680 T __mlir_ciface_softmax
                 U _expf
0000000000000488 T _softmax
```

`clang` links libm by default, so nothing extra is needed. And the
`-FLT_MAX` constant is a nice example of the LLVM 24 float-literal drift that
`tools/ll_fp_compat.py` exists for (Chapter 28): `mlir-translate` prints it as
`f0xFF7FFFFF`, which the filter rewrites to the legacy
`0xC7EFFFFFE0000000` that clang 20 accepts.

### Building and running it

[`build.sh`](examples/mlir_attention/build.sh) runs `convert-scf-to-cf`,
`convert-cf-to-llvm`, `convert-math-to-llvm`, the arith/index/memref/func
`*-to-llvm` conversions and `reconcile-unrealized-casts`, then
`mlir-translate --mlir-to-llvmir` and `clang -shared -fPIC -x ir`.
[`aot_main.py`](examples/mlir_attention/aot_main.py) feeds a seeded `6×6`
matrix of stand-in attention scores (random, scaled by 3) through the kernel,
checks it against the model's own `gpt2.softmax`, and confirms every row sums
to 1:

```sh
bash part6-end-to-end-ml/ch26-transformers-and-softmax/examples/mlir_attention/build.sh
```
```
MLIR softmax successful! (max abs error 0.00e+00; every row sums to 1)
```

The error is exactly zero on this input: for a row this short NumPy sums
sequentially too, so both sides do the same float32 operations in the same
order. (Expect last-bit differences on longer rows, where NumPy switches to
pairwise summation — hence the `1e-5` tolerance in the check.)

Put it together with Chapter 8's `linalg.matmul`, and you have all of
`attention = softmax(Q @ Kᵀ / √d) @ V` in MLIR — which is what a real ML
compiler fuses and ships to a GPU.

### Testing it

[`test/part6/ch26_softmax.mlir`](../../test/part6/ch26_softmax.mlir)
compiles the kernel under lit and links it with a C driver
([`softmax_main.c`](examples/mlir_attention/softmax_main.c)) that compares
every entry against a plain-C stable softmax. One of its rows is
`[1000, 1001, 1002, 1003]` — the overflow case — and must produce the same
probabilities as `[1, 2, 3, 4]`:

```sh
.venv/bin/lit build/test -v --filter 'ch26_'
```

## The same softmax, in the real GPT-2

Chapter 27 imports HuggingFace's GPT-2 through torch-mlir. Open its
checked-in linalg snapshot,
[`gpt2_linalg_stripped.mlir`](../src/gpt2/gpt2_linalg_stripped.mlir), and
the attention you just built is there, op for op. The 12 heads × 64 appear
as `linalg.batch_matmul` over a leading head dimension of 12:

```mlir
    %79 = linalg.batch_matmul ins(%collapsed_175, %collapsed_176 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
```

(the `÷√d` is split as `64^(-1/4) = 0.35355…` applied to both Q and K before
this matmul). Then comes the softmax — the same three steps as
`softmax.mlir`, written as `linalg.generic`s on tensors instead of `scf.for`
on memrefs (abridged from the snapshot; `#map7` is the identity on 4-D, while
`#map15` drops the last dimension and `#map6` pins it to 0 — the reduced
outputs):

```mlir
    // 1. row max: a "reduction" over the last dim, starting from -inf
    %97:2 = linalg.generic {... iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%92 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
      %1192 = arith.maximumf %in, %out : f32
      ...
    // 2a. x - max
      %1190 = arith.subf %in, %in_495 : f32
    // 2b. exp
      %1190 = math.exp %in : f32
    // 2c. row sum: another reduction
    %102 = linalg.generic {... iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ... {
      %1190 = arith.addf %in, %out : f32
    // 3. normalize
      %1190 = arith.divf %in, %in_495 : f32
```

The snapshot has exactly **12 `math.exp`** — one softmax per layer — next to
the 24 `batch_matmul`s (`QKᵀ` and `·V`, per layer) that Chapter 27 counted,
and **12 `math.tanh`**, the GELU of each layer's FFN. Two differences from
our hand-written kernel are instructive:

- PyTorch starts the max from `-inf` (`0xFF800000`) rather than `-FLT_MAX`,
  and then adds a guard (`%105`–`%118` in the snapshot) that detects rows
  where *every* score was `-inf` and replaces them with zeros instead of the
  `nan` that `-inf - (-inf)` would produce. Our `-1e10` mask never creates
  such rows; a mask with true `-inf` can.
- The importer emits each step as its own `linalg.generic`, materializing a
  full `1×12×N×N` tensor between them. Our kernel fused steps 2a–2c into one
  loop. Recovering that fusion is the compiler's job — the transform-dialect
  fusion of Chapter 22, or what Chapter 28's pipeline does (and does not do)
  when it lowers this model.

## Key takeaways

- **A transformer is a graph of familiar ops** — matmul, layer norm, GELU,
  softmax, residual adds — stacked. Nothing here is outside Chapters 8 and 25
  plus softmax.
- **Self-attention** = `softmax(QKᵀ/√d + mask) · V`: project to Q/K/V,
  score, mask the future, softmax, weight the Values; run it in parallel
  **heads**.
- **Softmax is the interesting op to compile** because of its per-row
  reductions; `math.exp` + `scf.for` reductions (with `iter_args`) reproduce
  the stable formula exactly.
- **The model is a compiler workload.** Frameworks lower exactly this graph
  to `linalg`/`tensor` and optimize it — and Chapter 27 shows the real GPT-2
  arriving as precisely those ops.

## Exercises

1. **Watch it overflow.** Make a copy of `softmax.mlir` without the max
   subtraction (`%d = %v`) and feed it the C driver's row
   `[1000, 1001, 1002, 1003]`. What does the kernel return, and at which
   step does the first `inf` appear?
2. **Fuse passes 1 and 2: online softmax.** Compute the max and the sum in
   one pass by rescaling the running sum whenever the max grows
   (`s ← s·exp(m_old − m_new) + exp(x − m_new)`); the loop then carries two
   `iter_args`. Check it with `aot_main.py`. This trick is the heart of
   FlashAttention.
3. **Scaled, masked attention in MLIR.** Extend the kernel into
   `@attention_scores(%q, %k, %out)` that computes `Q·Kᵀ/√d` with a causal
   mask (`j > i` → `-1e10`) and then the softmax, and check it against
   `gpt2.model`'s `attention` with `v = I` (identity) so the result *is* the
   weight matrix.
4. **Layer norm.** Write `@layer_norm(%x, %g, %b, %out)` in the same style
   (a mean reduction, a variance reduction, then an elementwise pass with
   `math.rsqrt`) and check it against `gpt2.model.layer_norm`. How many
   passes over each row do you need, and can you get it down to two?
5. **Grow the demo.** Set `demo.py` to GPT-2 small's shapes
   (`D=768, N_HEAD=12, N_LAYER=12, D_FF=3072, VOCAB=50257, CTX=1024`) and
   count the parameters of the resulting `Model`. Compare with the
   124M-parameter figure in Chapter 28's scale check — what accounts for any
   difference?
6. **Read the real one.** In `gpt2_linalg_stripped.mlir`, find the first
   layer norm (hint: look for a reduction followed by `math.rsqrt`). Map each
   `linalg.generic` to a line of `layer_norm` in `gpt2/model.py`.

## Notes on reproducing with current MLIR

The source chapter was verified on Homebrew LLVM 20.1.8 with `llc`. Rerun
with this repo's MLIR wheel (LLVM 24):

- `softmax.mlir` and the pass pipeline are **unchanged**; the result is still
  bit-identical to NumPy on the test input.
- No `llc` in the wheel: `build.sh` compiles the `.ll` with `clang -x ir`.
- The `.ll` needs `tools/ll_fp_compat.py` (the `f0xFF7FFFFF` literal for
  `-FLT_MAX`) and removal of LLVM 24's `nocreateundeforpoison` intrinsic
  attribute (on the `@llvm.exp.f32` / `@llvm.maximum.f32` declarations)
  before clang 20 accepts it; `ll_fp_compat.py` does both. See
  Chapter 25's notes for details.
- The driver uses the shared `tools/np_memref.py`, takes the library path as
  an argument, and seeds its RNG so the output is reproducible.
- `demo.py` and `gpt2/model.py` are unchanged and need only NumPy.

## Where to go next

You have now written a dense layer and a softmax by hand — and seen, in the
last section, that the real GPT-2 is ~1,100 linalg ops of exactly that kind.
Nobody writes those by hand.

- **[Chapter 27: Importing PyTorch Models with torch-mlir](../ch27-torch-mlir-import/README.md)**
  gets them from PyTorch instead. Its
  [§5](../ch27-torch-mlir-import/README.md#5-gpt-2-wrappers-dynamic-shapes-and-the-two-step-lowering)
  exports the real 124M-parameter GPT-2 — the same architecture as
  `gpt2/model.py`, with trained weights and a dynamic sequence length — and
  runs the greedy decoding loop that `demo.py` took one step of.
- **[Chapter 28](../ch28-cpu-pipeline-and-abi/README.md)** lowers it to
  native code; its
  [§6 "Scale check: GPT-2"](../ch28-cpu-pipeline-and-abi/README.md#6-scale-check-gpt-2)
  pushes the whole model through the pipeline and shows where its matmuls
  (and the attention `batch_matmul`s around your softmax) end up.
