# Part 5: Structured and Parallel Compute

Four independent capstones on the machinery of Parts 2–4: designing an op
*interface* and pointing a tile DSL at a GPU; driving upstream's
linalg/transform stack instead of building your own; a whole-program
optimizer that calls an ILP solver; and equality saturation, which replaces
greedy rewriting with an e-graph that keeps every rewrite at once. After this part you have seen every
major extension mechanism MLIR offers.

| Chapter | One line |
|---|---|
| [ch21: Op Interfaces and a GPU Tile DSL](ch21-op-interfaces-gpu-tiles/README.md) | `tiny_tile`: layouts, a lowering interface each op implements, SIMT codegen |
| [ch22: Linalg and the Transform Dialect](ch22-linalg-transform-dialect/README.md) | Structured ops, and scheduling (tile/fuse/vectorize) as an IR you write |
| [ch23: Dataflow Analysis and Global Optimization](ch23-dataflow-analysis/README.md) | The `noisy` dialect: an upstream analysis hijacked as a noise meter, and or-tools placing ops optimally |
| [ch24: Equality Saturation and E-graphs](ch24-equality-saturation-egraphs/README.md) | Build an e-graph in Python, saturate it with rewrites, extract the cheapest program, emit MLIR and run it |

**Prerequisites:** Part 4. ch23's optimizer pass needs the optional solver
build (`-DENABLE_SOLVER=ON`, see [ch23](ch23-dataflow-analysis/README.md)
or the repo [README](../README.md)); everything else runs on the default
build. ch24 is plain Python plus the wheel's tools and can be read right
after Part 3's pattern chapters (ch16–17). **Next:** [Part 6](../part6-end-to-end-ml/README.md), where the
input stops being hand-written IR and starts being PyTorch.
