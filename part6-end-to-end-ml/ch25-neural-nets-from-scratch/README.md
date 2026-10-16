# Chapter 25: Neural Networks from Scratch, then Compiled

> Adapted from mlir-ir-lowering's `5_neural_network` chapter, itself based on
> Stephen Diehl's *"MLIR Part 5 — Neural Networks"* article. That article is a
> ground-up **Python** implementation with no MLIR code; the source chapter
> reproduced it as a small runnable library **and added** an MLIR forward-pass
> kernel to tie it back to the compiler.

## What you will learn

- Why a neural network is "just" a big differentiable function, and why
  training it reduces to gradient descent.
- How **reverse-mode automatic differentiation** works, by building a
  micrograd-style engine (`Value`, `.backward()`) in ~100 lines of Python.
- How neurons, layers and a multi-layer perceptron (MLP) stack on top of that
  engine, and how an SGD optimizer trains one on a toy dataset.
- Why the autodiff graph is the same kind of object as an MLIR module.
- How the workhorse of every network — a dense layer `relu(X @ W + b)` — is
  written by hand in `linalg`, with the bias broadcast and ReLU **fused** into
  one `linalg.generic`, then compiled to a native library and checked against
  NumPy.

## Prerequisites

- [Chapter 4](../../part1-using-mlir/ch04-memory-and-the-c-abi/README.md):
  memref descriptors and `llvm.emit_c_interface` — the driver hands NumPy
  arrays to the kernel through `tools/np_memref.py`.
- [Chapter 8](../../part1-using-mlir/ch08-linalg-by-hand/README.md):
  `linalg.matmul`, `linalg.generic`, indexing maps, broadcasting, and fusion.
  Part B of this chapter is built entirely from those pieces.
- [Chapter 3](../../part1-using-mlir/ch03-llvm-ir-and-native-code/README.md):
  the `mlir-translate` → `clang` path to a native shared library.
- The repo's `.venv` (see the root README) — NumPy is the only Python
  dependency. **No torch is needed in this chapter.**

All code lives in [`examples/`](examples/):

```
examples/
├── tinynn/            # the pure-Python library (the article's content)
│   ├── engine.py      #   Value: reverse-mode autodiff
│   ├── nn.py          #   Module / Neuron / Layer / MLP
│   └── optim.py       #   SGD (+ L2 weight decay)
├── train_demo.py      # train on synthetic two-moons (NumPy only)
└── mlir_layer/        # the MLIR bridge: a dense layer in linalg
    ├── dense_relu.mlir
    ├── build.sh
    ├── aot_main.py        # NumPy driver
    └── dense_relu_main.c  # C driver (used by the lit test)
```

## A neural network is just a big differentiable function

Parts 1–5 built machinery: lowering, memory, parallelism, structured linear
algebra, passes, dialects, conversions. This chapter turns to the
**application** the whole tutorial is aiming at — deep learning — but before
compiling neural networks it pays to understand, from first principles, what
one *is*.

Strip away the mystique and a neural network is a large function `f(x; θ)`
with many tunable parameters `θ` (weights and biases). "Learning" is just
**gradient descent**: measure how wrong the output is with a loss `L`,
compute `∇L` (how each parameter should change), nudge every parameter a
little against its gradient, repeat. The only non-obvious part is computing
`∇L` for millions of parameters efficiently — and the answer, **automatic
differentiation**, is the conceptual heart of PyTorch, JAX, and TensorFlow
alike.

Here is the connection that makes this a compiler chapter: autodiff works by
recording every operation into a **computation graph** — a DAG of ops with
data flowing between them. That is *the same idea as an MLIR module*. The
frameworks build an IR of your computation so they can differentiate it and
optimize it, exactly as MLIR builds an IR so it can lower and optimize it.
We build that graph from scratch in pure Python (Part A), then compile one
layer of it with the `linalg` dialect from Chapter 8 (Part B).

## Part A — the ideas, in pure Python (`tinynn/`)

### Automatic differentiation (`tinynn/engine.py`)

Every number is wrapped in a `Value` that remembers the operation that
produced it. Building an expression therefore builds a graph; each op also
stores a tiny `_backward` closure encoding its local derivative (the chain
rule, one edge at a time). Multiplication is the clearest case — the product
rule scales each input's gradient by the *other's* value:

*examples/tinynn/engine.py* (one op's forward + local backward)
```python
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      # product rule: each input's grad is scaled by the other's value
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
```

The engine defines only four primitive ops this way — `+`, `*`, `**`
(constant power) and `relu` — and derives everything else from them
(`-x` is `x * -1`, `a - b` is `a + (-b)`, `a / b` is `a * b**-1`, and the
reflected `__radd__`/`__rmul__` let plain floats appear on the left). Note
the `+=` in every `_backward`: a value used twice (like `a` below) receives
gradient from both uses.

`.backward()` on the final node topologically sorts the graph, seeds the
output gradient to `1`, and runs every `_backward` in reverse — gradients
flow from the loss back to every parameter. That reverse sweep *is*
**backpropagation**:

*examples/tinynn/engine.py* (the reverse sweep)
```python
  def backward(self):
    """Fill in .grad for every node feeding into this one."""
    topo, visited = [], set()

    def build(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build(child)
        topo.append(v)
    build(self)

    self.grad = 1.0                    # seed: d(output)/d(output) = 1
    for v in reversed(topo):           # reverse-topological = outputs first
      v._backward()
```

```text
   forward: build the graph          backward: chain rule, in reverse
   a ─┐                              a.grad += b·out.grad  ◄─┐
      ×──► out ─► L                                          out.grad ◄─ L.grad=1
   b ─┘                              b.grad += a·out.grad  ◄─┘
   each op stores _backward          run every _backward outputs-first
```

Try it on `e = relu(a·b + c) · a` with `a=2, b=-3, c=10` (run from
`examples/`):

```sh
python -c "
from tinynn import Value
a=Value(2.0);b=Value(-3.0);c=Value(10.0)
d=a*b+c; e=d.relu()*a
e.backward()
print(e, a, b, c, d)
"
```
```
Value(data=8.0000, grad=1.0000) Value(data=2.0000, grad=-2.0000) Value(data=-3.0000, grad=4.0000) Value(data=10.0000, grad=2.0000) Value(data=4.0000, grad=2.0000)
```

Check by hand: `d = 4 > 0`, so ReLU passes it through and `e = 4·2 = 8`.
`∂e/∂a = b·a + relu(d) = -6 + 4 = -2` — the two contributions are the two
uses of `a`, accumulated by `+=`. `∂e/∂b = a·a = 4`, `∂e/∂c = a = 2`.

### Neurons, layers, the MLP (`tinynn/nn.py`)

The standard hierarchy, every value a `Value` so gradients are automatic:

- **`Neuron`** — `relu(w·x + b)`; the weighted sum plus bias, then an
  activation.
- **`Layer`** — a row of neurons sharing the same inputs (a fully-connected
  layer).
- **`MLP`** — a stack of layers; the last is linear, the rest use **ReLU**
  (`max(0, z)`), the non-linearity that lets a deep net represent more than a
  single linear map.

All three derive from a `Module` base class that provides `parameters()` and
`zero_grad()`, mirroring how PyTorch's `nn.Module` is structured — minus the
performance. The whole hierarchy is tiny because each level just delegates to
the one below:

*examples/tinynn/nn.py* (the neuron — `w·x + b`, then activation)
```python
class Neuron(Module):
  """One neuron: a = activation( sum_i w_i * x_i + b )."""

  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(0.0)
    self.nonlin = nonlin

  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w·x + b
    return act.relu() if self.nonlin else act
```

In matrix form each layer is `A = σ(W·A_prev + b)` — keep that equation in
mind; Part B compiles exactly it.

### Optimizer & training (`tinynn/optim.py`, `train_demo.py`)

`SGD` applies `p ← p − lr · p.grad` (with optional L2 weight decay
`p ← p − lr·(grad + 2λp)`, which shrinks weights toward zero to discourage
overfitting) — the entire optimizer is one line:

*examples/tinynn/optim.py* (the update rule)
```python
  def step(self, lambda_reg=0.0):
    """Take one gradient-descent step. lambda_reg>0 adds L2 weight decay,
    which shrinks weights toward zero to discourage overfitting."""
    for p in self.parameters:
      p.data -= self.lr * (p.grad + 2.0 * lambda_reg * p.data)
```

[`train_demo.py`](examples/train_demo.py) puts it all together on a
synthetic **two-moons** dataset — two interleaving half-circles that
*require* a non-linear boundary — and trains an `MLP(2, [8, 8, 1])` with an
MSE loss against ±1 targets, one example at a time (forward, loss,
`zero_grad`, `backward`, `step`). The dataset is generated with NumPy (no
scikit-learn), and both RNGs are seeded, so the run is reproducible:

```sh
cd part6-end-to-end-ml/ch25-neural-nets-from-scratch/examples
python train_demo.py
```
```
Training MLP(2, [8, 8, 1]) — 105 parameters
  epoch  10: loss=0.0766  acc=1.00
  epoch  20: loss=0.0243  acc=1.00
  epoch  30: loss=0.0176  acc=1.00
  epoch  40: loss=0.0096  acc=1.00
Final training accuracy: 1.00
Neural network trained successfully!
```

> **Why it's slow.** Each `Value` is one Python object and each op one
> Python call, so a forward+backward pass is hundreds to thousands of
> interpreter operations per example. Concretely, one example through this
> 105-parameter MLP builds a graph of **318 `Value` nodes**; 40 epochs × 100
> examples is ~1.3 million node creations plus their backward closures, and
> the run takes about **1.7 s** for a model whose forward pass is three
> tiny matrix multiplies (2→8→8→1) that compiled code evaluates in
> microseconds. That sluggishness is the entire
> motivation for the rest of this part: the *same* graph, compiled to
> vectorized, parallel, GPU code, runs orders of magnitude faster.

## Part B — one layer, compiled (`mlir_layer/`)

A neural network is mostly the same kernel repeated: a **dense layer**
`out = relu(X @ W + b)`. That decomposes cleanly into pieces you already
built in Chapter 8:

```text
   X (N×K)     W (K×M)            b (M)             out (N×M)
   ┌────┐   ┌──────┐          ┌──────────┐       ┌──────────┐
   │batch  ×  weights   =   X@W  + bias(row) ─►  relu(·)  =  activations
   └────┘   └──────┘          └──────────┘       └──────────┘
   linalg.matmul            ── one fused linalg.generic ──
                            (add b[j] across rows, then max(·,0))
```

| Layer math | Linalg op |
| --- | --- |
| `X @ W` (the `W·A_prev`) | `linalg.matmul` |
| `+ b` broadcast over the batch | a `linalg.generic` reading `b[j]` via `(i,j)->(j)` |
| `relu(·)` | `arith.maximumf z, 0` inside that same generic |

(The batch is the *rows* of `X`, so the math reads `X @ W` rather than
`W · A_prev` — same layer, transposed convention.)

The bias-add and the ReLU are **fused into one `linalg.generic`** — a single
pass over the output instead of three — which is exactly the kernel-fusion
idea from Chapter 8 applied to a real layer. The whole kernel:

*examples/mlir_layer/dense_relu.mlir*
```mlir
func.func @dense_relu(%X: memref<?x?xf32>, %W: memref<?x?xf32>,
                      %b: memref<?xf32>, %out: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  // out = X @ W   (accumulates into the caller's zeroed buffer)
  linalg.matmul ins(%X, %W : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out : memref<?x?xf32>)

  // out[i,j] = relu(out[i,j] + b[j])   — bias broadcast + activation, fused
  linalg.generic {
    indexing_maps  = [affine_map<(i, j) -> (j)>,        // b: depends only on column j
                      affine_map<(i, j) -> (i, j)>],    // out: full 2-D
    iterator_types = ["parallel", "parallel"]
  } ins(%b : memref<?xf32>) outs(%out : memref<?x?xf32>) {
  ^bb0(%bias: f32, %acc: f32):
    %z    = arith.addf %acc, %bias : f32
    %zero = arith.constant 0.0 : f32
    %relu = arith.maximumf %z, %zero : f32              // ReLU = max(z, 0)
    linalg.yield %relu : f32
  }
  return
}
```

The bias's indexing map `(i, j) -> (j)` is the broadcast: for every row `i`
the same `b[j]` is read, stretching the length-`M` bias across all `N` rows
exactly like Chapter 8's `linalg.broadcast` — but here folded straight into
the activation loop. Three more details worth noticing:

- The kernel works on **memrefs** (already-bufferized buffers), with dynamic
  `?` shapes, so the same compiled code serves any batch and layer size.
- `linalg.matmul` *accumulates* into `outs`, so the caller must pass a zeroed
  `out`. (A tensor-level version would start from `linalg.fill`; Chapter 27's
  importer output does exactly that.)
- `llvm.emit_c_interface` asks for a `_mlir_ciface_dense_relu` wrapper that
  takes one pointer per memref descriptor (Chapter 4), which is what the
  Python driver calls.

### What it lowers to

`--convert-linalg-to-loops` alone shows the two ops as the loops a CPU will
run — the matmul's triple loop, then **one** double loop doing bias + ReLU
(output from the current wheel):

```mlir
    scf.for %arg4 = %c0 to %dim step %c1 {
      scf.for %arg5 = %c0 to %dim_1 step %c1 {
        scf.for %arg6 = %c0 to %dim_0 step %c1 {
          %0 = memref.load %arg0[%arg4, %arg6] : memref<?x?xf32>
          %1 = memref.load %arg1[%arg6, %arg5] : memref<?x?xf32>
          %2 = memref.load %arg3[%arg4, %arg5] : memref<?x?xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg3[%arg4, %arg5] : memref<?x?xf32>
        }
      }
    }
    %dim_2 = memref.dim %arg2, %c0 : memref<?xf32>
    %dim_3 = memref.dim %arg3, %c0 : memref<?x?xf32>
    scf.for %arg4 = %c0 to %dim_3 step %c1 {
      scf.for %arg5 = %c0 to %dim_2 step %c1 {
        %0 = memref.load %arg2[%arg5] : memref<?xf32>
        %1 = memref.load %arg3[%arg4, %arg5] : memref<?x?xf32>
        %2 = arith.addf %1, %0 : f32
        %3 = arith.maximumf %2, %cst : f32
        memref.store %3, %arg3[%arg4, %arg5] : memref<?x?xf32>
      }
    }
```

The `(i, j) -> (j)` map became `memref.load %arg2[%arg5]` — the bias is
indexed by the column loop variable only. Notice also that the bias loop runs
to `memref.dim %arg2` (the bias length) while the matmul's column loop ran to
`memref.dim %arg1, 1`; linalg trusts the shapes to agree, and nothing checks
that at runtime.

### Building and running it

[`build.sh`](examples/mlir_layer/build.sh) runs the standard memref-level
pipeline — `convert-linalg-to-loops`, `convert-scf-to-cf`, then the
`*-to-llvm` conversions and `reconcile-unrealized-casts` — then
`mlir-translate --mlir-to-llvmir`, and compiles the LLVM IR with
`clang -shared -fPIC -x ir` into `build/libdense.so`.
[`aot_main.py`](examples/mlir_layer/aot_main.py) feeds in NumPy `X`, `W`,
`b` (via `tools/np_memref.py`) and checks the output against
`np.maximum(X @ W + b, 0)`:

```sh
bash part6-end-to-end-ml/ch25-neural-nets-from-scratch/examples/mlir_layer/build.sh
```
```
MLIR dense layer  out = relu(X @ W + b)  successful! (max abs error 5.96e-08)
[[0.     0.     0.5044 0.     0.    ]
 [0.     0.     0.     0.     0.    ]
 [0.     0.     0.     0.     0.8985]
 [2.1901 0.     1.1844 0.001  0.    ]]
```

The inputs are seeded random `4×3`, `3×5` and `5` arrays; the error is pure
float rounding (the compiled loop sums in a different order than NumPy's
BLAS). All the zeros are ReLU at work: with zero-mean random weights about
half the pre-activations are negative.

This is the unit deep-learning compilers stamp out and optimize; the next
chapter wires many such layers into a transformer.

### Testing it

[`test/part6/ch25_dense_relu.mlir`](../../test/part6/ch25_dense_relu.mlir)
runs the same pipeline under lit, links the kernel against a small C driver
([`dense_relu_main.c`](examples/mlir_layer/dense_relu_main.c)) that compares
it with a plain-C `relu(X @ W + b)`, and FileChecks the printed rows:

```sh
.venv/bin/lit build/test -v --filter 'ch25_'
```

## Key takeaways

- **Training = autodiff + gradient descent.** Record the computation as a
  graph, run the chain rule backward over it (`backward()`), step the
  parameters. That graph is the same kind of IR MLIR manipulates.
- **A net is a stack of dense layers**, each `A = σ(W·A_prev + b)` — matmul,
  bias, activation.
- **That layer is pure Chapter 8 Linalg**: `linalg.matmul` + a fused
  bias/ReLU `linalg.generic`, the broadcast bias riding in via a `(i,j)->(j)`
  indexing map. Frameworks lower whole models to graphs of exactly these ops.
- **The scalar Python version is slow on purpose** — it motivates compiling
  the graph, which is what the rest of Part 6 does.

## Exercises

1. **A new op for the engine.** Add `tanh()` to `Value` (forward
   `math.tanh`, local derivative `1 - tanh²`), then switch the MLP's hidden
   activation to it. Does `train_demo.py` still reach accuracy 1.00? Check
   your derivative numerically first: compare `a.grad` with
   `(f(a + h) - f(a - h)) / 2h`.
2. **The `+=` matters.** In `__mul__`'s `_backward`, change both `+=` to
   `=` and rerun the hand-checked `e = relu(a·b + c)·a` example above. Which
   gradient goes wrong, which of its two contributions survives, and why
   do `b` and `c` stay correct?
3. **Weight decay.** Pass `lambda_reg=1e-3` to `opt.step()` in
   `train_demo.py` and print the largest `|p.data|` at the end with and
   without it. What does L2 decay do to the weights, and to the loss?
4. **A second layer in MLIR.** Add a function `@mlp2` to
   `dense_relu.mlir` that calls `@dense_relu` twice (`X → hidden → out`, the
   second layer without the ReLU), with the caller passing all buffers. Extend
   `aot_main.py` to check it against NumPy. Where does the intermediate
   buffer have to come from, given the kernel only works on memrefs?
5. **Tensors instead of memrefs.** Rewrite the kernel on `tensor<?x?xf32>`
   values, starting the matmul from a `linalg.fill` of zero so the caller no
   longer has to zero `out`, and lower it with `--one-shot-bufferize` first
   (Chapter 2). Compare the result with the memref version — and with what
   Chapter 27's importer emits for a `torch.nn.Linear` + ReLU.
6. **Skip the shared library.** Load `dense_relu.mlir` with the MLIR Python
   bindings, run the same pass pipeline with
   `mlir.passmanager.PassManager`, and execute it with
   `mlir.execution_engine.ExecutionEngine` (`engine.invoke("dense_relu", …)`
   with `ctypes.pointer(ctypes.pointer(desc))` per memref). Is the output
   bit-identical to the `clang`-built library's?

## Notes on reproducing with current MLIR

The source chapter was verified on Homebrew LLVM 20.1.8 with `llc`. Rerun
with this repo's MLIR wheel (LLVM 24):

- `dense_relu.mlir` and the pass pipeline are **unchanged**; every pass name
  still exists and the kernel is numerically identical.
- The wheel ships no `llc`, so `build.sh` compiles the textual LLVM IR with
  `clang -x ir` instead of `llc -filetype=obj` + `clang -shared`.
- LLVM 24's `.ll` output needs two fixes before an older system clang
  (here Homebrew clang 20) accepts it: `tools/ll_fp_compat.py` (float
  literal syntax; see Chapter 28), and removing the `nocreateundeforpoison`
  attribute that LLVM 24 attaches to the `@llvm.maximum.f32` intrinsic
  declaration (`arith.maximumf` lowers to it). Clang 20 reports the latter
  as `error: unterminated attribute group`; `ll_fp_compat.py` drops it
  too. It is an optimization hint only, so removing it is safe.
- `mlir-translate` emits no `target triple`, so `clang` warns that it is
  overriding the module's (empty) triple with the host's;
  `-Wno-override-module` silences it.
- The driver now uses the shared `tools/np_memref.py` instead of the source's
  per-chapter `common/np_memref.py`, takes the library path as an argument,
  and seeds its RNG so the printed output above is reproducible.

## Where to go next

- **[Chapter 26: Transformers and a Compiled Softmax](../ch26-transformers-and-softmax/README.md)**
  wires dense layers, layer norm and attention into a GPT-2 forward pass in
  NumPy, and compiles attention's defining op — softmax — the same way.
- Notice what Part B cost: one hand-written kernel, one hand-written driver,
  for *one* layer. A real network is dozens of layers, and a real model's
  weights come from a framework, not from `np.random`. Writing every layer by
  hand does not scale — which is why
  [Chapter 27](../ch27-torch-mlir-import/README.md) stops writing IR and
  **imports** it from PyTorch: the importer emits exactly the
  `linalg.matmul` + `linalg.generic` patterns you just wrote, for the whole
  model at once.
