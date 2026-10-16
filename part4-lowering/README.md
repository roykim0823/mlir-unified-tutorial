# Part 4: Lowering

Getting *out* of your dialect: type-changing conversion, the descent to
the LLVM dialect, and — the payoff of the whole poly arc so far — a native
binary computing polynomial arithmetic, called from C. Plus the `tiny`
dialect, a second, smaller lowering built around region ops.

| Chapter | One line |
|---|---|
| [ch18: Dialect Conversion](ch18-dialect-conversion/README.md) | Type converters, legality, conversion patterns: `!poly.poly<10>` becomes `tensor<10xi32>` |
| [ch19: Lowering to LLVM and a Native Binary](ch19-to-llvm-native-binary/README.md) | The full ladder to the llvm dialect, `mlir-translate`, `clang -x ir`, and a running executable |
| [ch20: Region Ops and Lowering to SCF](ch20-region-ops-to-scf/README.md) | The `tiny`/`tiny_loop` dialects: ops that carry regions, lowered to structured control flow |

**Prerequisites:** Part 3 for ch18–19 (they lower `poly`); ch20 stands
mostly alone after Part 2. A Python venv is needed for the `tiny` DSL
frontend (`python/`). **Next:**
[Part 5](../part5-structured-and-parallel/README.md) — what upstream
already provides for structured compute, so you stop hand-writing it.
