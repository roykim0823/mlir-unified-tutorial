# Chapter 9: Testing with lit and FileCheck

> Reworked from the mlir-tutorial fork's chapter 2 for the unified repo.

In Chapters 1–2 we ran lowerings by hand and eyeballed the output of
`mlir-opt`. This chapter turns that into automated tests, using the same
two tools every LLVM and MLIR project uses — and the same harness that
validates every exercise in Parts 2–4 of this tutorial.

## What you will learn

- How this repo's `test/` tree, CMake targets (`check-all`,
  `check-part1..4`), and pip-provided `lit`/`FileCheck` fit together.
- The FileCheck directive family — what `CHECK:` really matches, and how
  `CHECK-NOT`, `CHECK-LABEL`, `CHECK-SAME`, `CHECK-NEXT`, `CHECK-DAG`,
  and capture variables differ from it.
- How to run a single test file with lit, how to replay its `RUN:`
  command by hand when you need to deviate from it, and how to read
  lit's and FileCheck's failure reports.
- How FileCheck (the test-time verifier) and `generate-test-checks.py`
  (the authoring-time draft generator) divide the work of exhaustive tests.
- Common pitfalls: checks that skip too far, dead check prefixes, the
  `CHECK-NOT` window, and blindly regenerated assertions.

## Prerequisites

- [Chapter 1](../ch01-why-mlir-and-reading-ir/README.md) and
  [Chapter 2](../ch02-lowering-pipelines/README.md).
- A configured build in `build/` per the top-level
  [README](../../README.md) — in particular, the virtualenv `.venv`
  exists and CMake was configured with `-DLLVM_EXTERNAL_LIT=$(which lit)`.

Run every command in this chapter from the repo root, with the
virtualenv active: `source .venv/bin/activate`.

---

## 1. Concepts: lit and FileCheck

A compiler is an awkward thing to unit-test: its inputs and outputs are
large strings in domain-specific languages, and what you usually want to
assert is "when I run *this tool* on *this program*, the output has
*this shape*". LLVM and MLIR grew a pair of tools for exactly that:
**[`lit`](https://llvm.org/docs/CommandGuide/lit.html)** (LLVM Integrated
Tester) discovers test files and runs the shell commands embedded in
them, and **[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html)**
makes assertions about the output of those commands.

Both live *inside* the test file, as specially formatted comments: `RUN:`
comments tell lit what commands to execute, and `CHECK` comments tell
FileCheck what those commands' output must (and must not) contain. The
test *is* the input program, annotated.

In this repo the two tools come from two different places, both installed
by the root README's setup: `lit` is a **Python package** (see
[`requirements.txt`](../../requirements.txt)) — activating `.venv` puts
the `lit` command on your `PATH` — while `FileCheck`, `mlir-opt`, `not`,
and the other LLVM binaries ship inside the **MLIR pip wheel**, under
`$(python -m mlir_wheel --root-dir)/bin/`.

## 2. Step: run this repo's tests

The lit tests live under [`test/`](../../test/), one directory per part:

```
test/
├── CMakeLists.txt        defines check-all and check-part1..check-part4
├── lit.site.cfg.py.in    template; CMake fills in concrete paths
├── part1/
│   ├── ctlz.mlir
│   └── ctlz_simple.mlir
└── part2..part4/ ...
```

Run all of Part 1's tests through CMake (this also rebuilds
`tutorial-opt` first if needed):

```bash
cmake --build build --target check-part1
```

```
[100%] Running part1 tests
-- Testing: 2 of 33 tests, 2 workers --
PASS: MLIRUnifiedTutorial :: part1/ctlz_simple.mlir (1 of 2)
PASS: MLIRUnifiedTutorial :: part1/ctlz.mlir (2 of 2)

Total Discovered Tests: 33
  Excluded: 31 (93.94%)
  Passed  :  2 (6.06%)
```

`cmake --build build --target check-all` runs the whole suite. You can
also skip CMake and invoke `lit` directly on the *configured* test
directory in the build tree — this is what the targets do under the hood,
but it rebuilds nothing, so build first if you changed code:

```bash
lit build/test --filter=part1 -v      # everything under part1/
lit build/test --filter=ctlz -v       # every test whose name matches "ctlz"
lit build/test/part1/ctlz_simple.mlir -v   # exactly one test
```

`--filter` takes a regex matched against the test's full name (e.g.
`MLIRUnifiedTutorial :: part1/ctlz.mlir`). Note that you point lit at
`build/test`, not `test/` — section 7 explains why.

> **Which `lit` runs?** All `check-*` targets use the lit named by
> `-DLLVM_EXTERNAL_LIT` at configure time (the venv's `.venv/bin/lit`),
> so they work whether or not the venv is active. Only when you invoke
> `lit` by hand, as above, does your `PATH` — and therefore the venv —
> matter.

Green lines alone tell you nothing about *how* this works, though. The
sections below open the box. The rule of thumb they follow: **passing
tests run through lit** (single-file invocations like the ones above);
**deviations from a test's `RUN:` line are replayed by hand**, running
`mlir-opt` and `FileCheck` exactly as lit would. Inside the harness, lit
puts the wheel's `bin/` on `PATH` for you; for the hand-runs, do the
same in your shell:

```bash
export PATH=$(python -m mlir_wheel --root-dir)/bin:$PATH
which mlir-opt   # .../site-packages/mlir_wheel/bin/mlir-opt
```

## 3. A FileCheck primer

FileCheck's job is simple to state: it reads a program's output on
**stdin**, reads *check directives* from a file you name, and exits 0 if
every directive is satisfied, 1 otherwise. It is **silent on success** —
the exit code is the entire interface, which is what makes it composable
into test harnesses. All the subtlety is in the directives. Every
directive is a prefix plus an optional suffix — `CHECK:`, `CHECK-NOT:`,
`CHECK-LABEL:`, `CHECK-SAME:` — and the suffix changes **where the
pattern is allowed to match**.

The precise rules for a plain `CHECK:`:

- The pattern only needs to match a **substring of one line, anywhere on
  the line** — `CHECK: arith.addi` matches
  `%0 = arith.addi %arg0, %arg0 : i32`.
- The pattern is (mostly) **literal text** — but runs of whitespace match
  any whitespace, and you can embed a real regex between `{{` and `}}`
  (e.g. `{{[0-9]+}}`), or a *capture* between `[[` and `]]` (explained
  below).
- Directives must match **in order**: each one starts scanning **where
  the previous match ended** — but it may **skip any number of lines** on
  its way to a match.

That last rule is a loophole: a check will happily skip past the *end of
the thing you meant to test* and find its match somewhere else entirely.
`CHECK: func.func @square` followed by `CHECK: arith.addi` passes even
when `@square` contains no `arith.addi` — FileCheck just skips ahead
until some *later* function provides one, and a regression in `@square`
goes unnoticed. The suffixed directives exist to pin matches down:

- **`CHECK-LABEL:`** closes exactly that loophole. FileCheck first finds
  the line matching each label, uses those lines to **split the input
  into blocks**, and then processes every other directive **only within
  its own block** — a check between two labels can never match text
  beyond the next label. Labels also localize failures: without them, one
  bad check throws all the *following* checks out of alignment, and the
  error lands far from the real problem. Two rules when choosing a label:
  it should match **exactly one line** of the output — function
  signatures are the classic choice — and a label pattern **may not
  define or use capture variables** (a rule that shapes how real tests
  check signatures, as the next directive shows).
- **`CHECK-SAME:`** requires its pattern to match **on the same line as
  the previous match**, picking up where that match ended; if FileCheck
  would have to cross a newline, the check fails. Its two practical jobs:
  splitting one long assertion into readable pieces (a signature check
  becomes a label plus several `-SAME` lines), and attaching captures to
  a label (a label can't contain captures; the `-SAME` directly under it
  can).
- **`CHECK-NOT:`** inverts the assertion: its pattern must appear
  **nowhere** between the previous match and the next positive match —
  the directive for "the lowered op is *gone*".
- **`CHECK-NEXT:`** requires its pattern to match on the line
  **immediately after** the previous match — no skipping at all. It pins
  down an exact sequence: a label on a function's signature followed by
  one `CHECK-NEXT:` per body line asserts the body is *exactly* those
  lines, nothing more.
- **`CHECK-DAG:`** relaxes ordering instead: a run of consecutive
  `CHECK-DAG` lines may match **in any order**, within the window bounded
  by the surrounding plain checks. Many passes emit groups of operations —
  hoisted constants are the classic case — in an order they make no
  promises about. If a pass emits `arith.constant 4` before
  `arith.constant 2`, the plain checks `CHECK: arith.constant 2` then
  `CHECK: arith.constant 4` fail — after matching `constant 2`, the scan
  for `constant 4` starts *below* it — while the same two assertions as
  `CHECK-DAG:` pass regardless of emission order.
- **Capture variables:** `[[V:regex]]` **captures** whatever the regex
  matched into a variable named `V`, and a later `[[V]]` (no colon)
  asserts that the *same* text appears again. This is essential for SSA
  values: you can't hard-code `%0` or `%arg0` in a test, because the
  printer is free to rename values — what matters is that the value
  *defined here* is the one *used there*.

Two more mechanics before we read a real test. First, *where* directives
physically sit in the check file is irrelevant to FileCheck — it just
collects them into an ordered list, so a test can interleave its checks
with the code they describe (the repo's tests do). Second, the
`--check-prefix=NAME` flag makes FileCheck ignore `CHECK` and instead
look for directives spelled `NAME:`, `NAME-LABEL:`, `NAME-SAME:`, and so
on — that is how one file bundles several independent groups of checks
(exercise 3 uses it).

The family at a glance:

| Directive | The pattern must match... | Typical use |
|---|---|---|
| `CHECK:` | some line at/after the previous match; substring of the line; may skip lines | "this op appears (eventually)" |
| `CHECK-NOT:` | **nowhere** between the previous and the next positive match | "the lowered op is gone" |
| `CHECK-SAME:` | on the **same line** as the previous match | wrapped signatures; captures after a label |
| `CHECK-NEXT:` | on the line **immediately after** the previous match | airtight sequences |
| `CHECK-DAG:` | like `CHECK:`, but consecutive `-DAG`s may match in any order | hoisted constants; unordered op groups |
| `CHECK-LABEL:` | exactly one line; splits input into independent blocks; no captures | function boundaries |

(There are more — `CHECK-COUNT-<n>`, `CHECK-EMPTY:`, ... — see the
[FileCheck docs](https://llvm.org/docs/CommandGuide/FileCheck.html).
The next sections put the table to work on this repo's real tests.)

## 4. Step: read a real test, and run it with lit

Now the primer pays off. A real test file is section 1's idea in the
flesh — the input program, annotated. Here is the simplest real test in
this repo, [`test/part1/ctlz_simple.mlir`](../../test/part1/ctlz_simple.mlir):

```mlir
// RUN: mlir-opt %s --convert-math-to-funcs=convert-ctlz | FileCheck %s

func.func @main(%arg0: i32) -> i32 {
  // CHECK-NOT:  math.ctlz
  // CHECK:  call
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

The `CHECK` lines you can already read: some output line must contain
`call`, and `math.ctlz` must not appear before it — `CHECK-NOT` is the
absence directive from the table, evaluated over the window that ends at
the next positive match. Together: "after the lowering, the ctlz op is
gone and a function call appeared." (Deliberately the loose style —
plain `CHECK`s that may skip lines, nothing pinned down by labels; how
tight to make a test is a judgment call we return to in section 6.) The
genuinely new line is the first one:

- `// RUN: <command>` — lit executes this shell command; the test passes
  if it exits 0. `%s` is substituted with the path of the test file
  itself, so this pipes the file through the lowering and hands the
  result to FileCheck. `%s` is one of a family of *substitutions* — `%t`
  names a per-test temp file, the
  [lit docs](https://llvm.org/docs/CommandGuide/lit.html#substitutions)
  have the full table, and a suite can define its own (this repo's
  [`lit.cfg.py`](../../lit.cfg.py) adds `%project_source_dir`).
- `FileCheck %s` — the pass output arrives on stdin, and the *check file*
  is the test file itself (`%s` again): input program and check
  directives interleaved, each `CHECK` next to the code it describes.

Run just this test through lit, as in section 2:

```bash
lit build/test/part1/ctlz_simple.mlir -v
```

```
-- Testing: 1 tests, 1 workers --
PASS: MLIRUnifiedTutorial :: part1/ctlz_simple.mlir (1 of 1)
```

lit discovered the file as a test (per the config, every `.mlir` file
under `test/` is one), executed its `RUN:` line, saw exit code 0, and
declared PASS. But lit is not magic: all it did was substitute `%s` and
run the result as a shell command with the tools on `PATH` — a job you
can do yourself whenever you need to run something *other* than the
`RUN:` line as written. lit can't help with that — it only ever runs the
`RUN:` line exactly as it appears in the file; there is no flag for "run
this test with a different command" — so deviations are always replayed
by hand.

To see the test **fail**, we want exactly such a deviation: the same
pipeline but *without* the lowering pass, so `math.ctlz` survives and
the `CHECK` assertions trip. Do lit's job yourself — substitute `%s`
with the file path — and drop the pass:

```bash
mlir-opt test/part1/ctlz_simple.mlir | FileCheck test/part1/ctlz_simple.mlir
```

FileCheck reports the first assertion it cannot satisfy — and, perhaps
surprisingly, that is `CHECK: call`, not the `CHECK-NOT`. A `CHECK-NOT`
is only evaluated over the window ending at the next positive match;
since `call` never matches, that window never closes, and FileCheck gives
up on `call` first:

```
test/part1/ctlz_simple.mlir:5:12: error: CHECK: expected string not found in input
 // CHECK: call
           ^
<stdin>:1:1: note: scanning from here
module {
^
<stdin>:3:12: note: possible intended match here
 %0 = math.ctlz %arg0 : i32
           ^
```

This is FileCheck's standard error report; its anatomy is worth learning
once. The `error:` names the directive that could not be satisfied (file
and line in the *check* file); `scanning from here` points into the
*input* at where the search began; `possible intended match here` is
FileCheck's guess at what you meant — here it correctly fingers the
un-lowered `math.ctlz`. Below that (abbreviated away here) it prints the
whole input, annotated with where each directive matched or gave up.

That's the entire mechanism. Everything lit added on top — discovering
the file as a test, setting `PATH`, substituting `%s` (later `%t`),
reporting results — and everything CMake adds on top of lit in section 7
(configured paths, rebuilding tools before testing) is automation around
this one pipeline.

## 5. Step: watch a test fail through lit

What does a failure look like when *lit* runs the test, as `check-part1`
would? Rather than break a checked-in test, copy one outside the repo and
break the copy. lit needs only one thing to treat a directory as a test
suite: a `lit.cfg.py`. A minimal one is four lines — the tiny, no-CMake
version of the machinery section 7 dissects:

```bash
mkdir -p /tmp/lit-demo
cp test/part1/ctlz_simple.mlir /tmp/lit-demo/
cat > /tmp/lit-demo/lit.cfg.py <<'EOF'
import lit.formats
config.name = "lit-demo"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
EOF
```

Now sabotage the copy — change `// CHECK:  call` to
`// CHECK:  call_nonexistent` in `/tmp/lit-demo/ctlz_simple.mlir` — and
run lit on the directory (your shell's `PATH` still has the wheel's tools
from section 2; the minimal config inherits it):

```bash
lit /tmp/lit-demo -v
```

```
-- Testing: 1 tests, 1 workers --
FAIL: lit-demo :: ctlz_simple.mlir (1 of 1)
******************** TEST 'lit-demo :: ctlz_simple.mlir' FAILED ********************
Exit Code: 1

Command Output (stdout):
--
# RUN: at line 1
mlir-opt /private/tmp/lit-demo/ctlz_simple.mlir --convert-math-to-funcs=convert-ctlz | FileCheck /private/tmp/lit-demo/ctlz_simple.mlir
# executed command: mlir-opt /private/tmp/lit-demo/ctlz_simple.mlir --convert-math-to-funcs=convert-ctlz
# executed command: FileCheck /private/tmp/lit-demo/ctlz_simple.mlir
# .---command stderr------------
# | /private/tmp/lit-demo/ctlz_simple.mlir:5:12: error: CHECK: expected string not found in input
# |  // CHECK: call_nonexistent
# |            ^
# | <stdin>:3:7: note: possible intended match here
# |  %0 = call @__mlir_math_ctlz_i32(%arg0) : (i32) -> i32
...
Failed Tests (1):
  lit-demo :: ctlz_simple.mlir
```

Read it top to bottom: lit shows the `RUN:` line *after substitution*
(`%s` became the absolute path), then each executed command, then that
command's stderr — exactly the FileCheck error format from section 4,
and the `possible intended match` line even shows the fix. When a real
test breaks, this is the report you will read. Clean up with
`rm -rf /tmp/lit-demo`.

## 6. Step: exhaustive assertions, and `generate-test-checks.py`

`ctlz_simple.mlir` locks in only the barest facts. Its sibling,
[`test/part1/ctlz.mlir`](../../test/part1/ctlz.mlir), pins down the
*entire* generated function. Its check block begins:

```mlir
// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32
// CHECK-SAME:                       ) {
// CHECK:           %[[VAL_1:.*]] = call @__mlir_math_ctlz_i32(%[[VAL_0]]) : (i32) -> i32
```

After section 3 you can read this fluently: a label on each function
signature, `-SAME` continuations carrying the argument capture (a label
itself can't capture — the primer's rule in the wild), and `%[[VAL_0]]`
asserting the function's argument is what gets passed to the generated
call. The dense middle of the file is captures all the way down — open it
and note how `%[[ARG]]` and `%[[N]]`, captured many lines earlier, are
asserted to be the `scf.for` loop's `iter_args`. It's a passing test, so
per section 2's rule of thumb, run it with lit:

```bash
lit build/test/part1/ctlz.mlir -v
```

```
PASS: MLIRUnifiedTutorial :: part1/ctlz.mlir (1 of 1)
```

### Where do such checks come from?

Nobody writes 35 lines of captures by hand. LLVM ships a helper script,
`generate-test-checks.py` — easy to conflate with FileCheck, but the two
could not be more different:

- **FileCheck is a verifier that runs every time the test runs.** A
  compiled binary (here: in the wheel). Input: the pass's actual output +
  the `CHECK` directives. Output: an exit code. It generates nothing.
- **`generate-test-checks.py` is an authoring aid that runs once, at
  test-writing time.** A Python script in LLVM's *source tree* — not
  shipped in the wheel. Input: one concrete output of your pass. Output:
  a **draft of the `CHECK` lines** to paste into the test file. It
  verifies nothing and is never executed while tests run.

The division of labor: the *script* writes a first draft of the
assertions once; *FileCheck* enforces them forever after. Since this repo
has no LLVM source checkout, fetch the single self-contained file once
and feed it the lowering's output:

```bash
curl -sLo /tmp/generate-test-checks.py \
  https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/utils/generate-test-checks.py

mlir-opt test/part1/ctlz.mlir --convert-math-to-funcs=convert-ctlz \
  | python3 /tmp/generate-test-checks.py
```

The draft it prints (abbreviated) starts with a disclaimer about itself:

```
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. ...

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[ARG0:.*]]: i32) {
// CHECK:           %[[VAL_0:.*]] = call @__mlir_math_ctlz_i32(%[[ARG0]]) : (i32) -> i32
...
```

Compare this with the checked-in `test/part1/ctlz.mlir`: the **shape is
identical** — `CHECK-LABEL` per signature, `CHECK-SAME` for the rest of
it, captures for every SSA value — and the machine-chosen capture names
were **renamed by hand** to `ARG`, `C_32`, `ARGCMP`, `FOR_RET`, ... so a
human can read the assertions. Generated checks are a starting point, not
a finished test.

The trade-off between the loose style (`ctlz_simple.mlir`) and the
exhaustive style is a judgment call: exhaustive checks catch more
regressions, but break on harmless changes to the pass's output — see the
pitfalls section for how that goes wrong.

## 7. Step: how the suite is wired into CMake

Three small files turn `test/` into the `check-*` targets — the exact
pattern every MLIR project uses.

**(a) [`lit.cfg.py`](../../lit.cfg.py)** (repo root) is the suite's main
configuration — the industrial version of section 5's four-liner. The
whole file:

```python
import os
import lit.formats
from lit.llvm import llvm_config

config.name = "MLIRUnifiedTutorial"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.join(os.path.dirname(__file__), "test")
config.test_exec_root = os.path.join(config.tutorial_obj_root, "test")

config.substitutions.append(("%project_source_dir", config.tutorial_src_root))

# Add tools directories to PATH
llvm_config.with_environment("PATH", config.tutorial_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
```

Reading it top to bottom: the `config` object is nowhere imported — lit
inserts an instance into the module's scope when executing this file.
`config.suffixes` is the test-discovery rule: every `.mlir` file under
`test/` is a test, automatically — adding a test is *just adding a file*.
The `test_source_root` / `test_exec_root` split means tests are *found*
in the source tree but *run* under `build/test/` (temp files like `%t`
land there) — which is why you invoke `lit build/test`, not `lit test`.
And the two `with_environment` calls put this repo's own tool directory
(`build/tools`, home of `tutorial-opt` from Part 2 on) and the wheel's
LLVM tools directory on `PATH` — *that* is why a `RUN:` line can say just
`mlir-opt`.

But where do `config.tutorial_obj_root` and `config.llvm_tools_dir` come
from? From **(b)
[`test/lit.site.cfg.py.in`](../../test/lit.site.cfg.py.in)**, which runs
first. It solves the one genuinely awkward problem: the config needs
concrete paths, and those differ per machine. It is a *template* — CMake
fills in the `@...@` placeholders at configure time, producing
`build/test/lit.site.cfg.py`:

```python
@LIT_SITE_CFG_IN_HEADER@

config.llvm_tools_dir = "@LLVM_TOOLS_BINARY_DIR@"
config.tutorial_obj_root = "@CMAKE_BINARY_DIR@"
config.tutorial_src_root = "@CMAKE_SOURCE_DIR@"
config.tutorial_tools_dir = "@CMAKE_BINARY_DIR@/tools"

import lit.llvm
lit.llvm.initialize(lit_config, config)

lit_config.load_config(config, "@CMAKE_SOURCE_DIR@/lit.cfg.py")
```

So the chain when you run `lit build/test`: lit finds the generated
`lit.site.cfg.py` there, which sets the machine-specific paths
(`LLVM_TOOLS_BINARY_DIR` points into the pip wheel, because that's the
LLVM that `CMAKE_PREFIX_PATH` found at configure time) and then loads the
generic `lit.cfg.py` from the source tree.

**(c) [`test/CMakeLists.txt`](../../test/CMakeLists.txt)** ties it into
the build:

```cmake
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_SOURCE_DIR}/lit.cfg.py
)

set(UNIFIED_TEST_DEPENDS
  tutorial-opt
  FileCheck
)

add_lit_testsuite(check-all "Running all unified tutorial tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS -v
  DEPENDS ${UNIFIED_TEST_DEPENDS}
)

foreach(part part1 part2 part3 part4)
  add_custom_target(check-${part}
    COMMAND ${LLVM_EXTERNAL_LIT} ${CMAKE_CURRENT_BINARY_DIR} --filter=${part} -v
    DEPENDS ${UNIFIED_TEST_DEPENDS}
    COMMENT "Running ${part} tests"
  )
endforeach()
```

`configure_lit_site_cfg` performs the template substitution from (b);
`add_lit_testsuite` (an LLVM CMake helper) defines `check-all` using the
lit named by `-DLLVM_EXTERNAL_LIT`; the `foreach` stamps out
`check-part1..4` as plain targets calling that same lit with
`--filter=partN`. The
`DEPENDS` list is the "tools must exist" rule: every program a `RUN:`
line invokes must be built (or shipped) before lit runs — `FileCheck` and
`mlir-opt` come prebuilt in the wheel; `tutorial-opt`, this repo's own
pass driver, is rebuilt on demand, which is why `check-part1` compiled
things before testing in section 2.

## 8. The workflow for the rest of this tutorial

This chapter is load-bearing for everything that follows. From Part 2 on,
**every exercise is validated by lit tests** under `test/part2/`,
`test/part3/`, `test/part4/` — the loop is always:

1. Write or modify a pass / dialect / lowering under `lib/` (registered
   in `tools/tutorial-opt.cpp`).
2. Write a `.mlir` test: input IR, a `RUN:` line invoking `tutorial-opt`
   (or `mlir-opt`) with your pass, and `CHECK` lines.
3. Drop the file under `test/partN/` — discovery is automatic — and run
   `cmake --build build --target check-partN`.
4. When it fails, replay the `RUN:` line by hand (section 4) and read the
   lit/FileCheck report (section 5).

One honest limitation to carry forward: everything FileCheck can assert
is about what the output *looks like*, not what it *computes*. The
classic MLIR answer is `mlir-runner`, a JIT that executes fully-lowered
IR — but the pip wheel this repo builds against does not ship it. Part 4
closes the gap differently: it lowers all the way to LLVM IR, compiles it
with `clang -x ir`, and runs the resulting *native binary* inside a lit
test, FileCheck-ing the program's runtime output.

## Common pitfalls

Collected from the sections above — these account for most confusing test
experiences:

- **Checks that skip too far.** A plain `CHECK` may skip any number of
  lines, including past the function you meant to test (section 3's
  loophole). Habit: `CHECK-LABEL` on every function.
- **A dead check prefix never fails — it silently does nothing.**
  FileCheck only looks for the prefixes it is told about; a typo like
  `CHEK:`, or a prefix no `RUN` line mentions, just never runs. This repo
  contains a live specimen: the last line of `test/part1/ctlz.mlir` is
  `// NOCVT-NOT: __mlir_math_ctlz_i32`, and no `RUN` line passes
  `--check-prefix=NOCVT` — that assertion has never once been checked.
  Exercise 3 brings it to life.
- **`CHECK-NOT` needs an anchor.** It is evaluated over the window ending
  at the next positive match, so a file containing *only* `CHECK-NOT`s
  can pass vacuously, and its failure often surfaces as the *next*
  `CHECK` failing instead (section 4). Pair every `CHECK-NOT` with a
  positive check.
- **Deliberately-failing checks need `not`.** The wheel's `not` utility
  inverts a command's exit code, so a `RUN` line can assert that a
  command *fails*: `RUN: mlir-opt %s | not FileCheck %s` reads "these
  checks must NOT match". Nothing in `test/` currently uses it, but it
  is the standard tool whenever a test wants to lock in a failure.
- **Regenerating checks without reading the diff.** If a pass change
  breaks an exhaustive test, rerunning `generate-test-checks.py` always
  "fixes" it — including when the new output is wrong.
- **Forgetting the venv when running `lit` by hand.** The `check-*`
  targets have the venv's lit path baked in at configure time, but a
  bare `lit build/test ...` resolves from your `PATH` and dies with
  "lit: command not found" unless the venv is active.

## Exercises

1. Copy `test/part1/ctlz_simple.mlir` to `/tmp` and tighten its checks
   into the strict style: a `CHECK-LABEL:` on `@main`'s signature, a
   capture `%[[V:.*]]` on the generated `call`'s result, and a
   `CHECK-NEXT: return %[[V]]` asserting the call's result is returned
   on the very next line. Confirm it passes — drop it into a
   section-5-style demo directory and run lit on it, or replay the `RUN`
   command by hand (section 4) — then break it once on purpose (rename
   the second `%[[V]]`) to prove the checks really run.
2. Break `test/part1/ctlz.mlir`'s checks in your editor (change one use
   of `%[[ARG]]` to `%[[N]]`), run
   `lit build/test --filter=part1/ctlz.mlir -v`, and read the failure
   report end to end. Revert with `git checkout -- test/part1/ctlz.mlir`.
3. Resurrect the dead `NOCVT` assertion in `test/part1/ctlz.mlir`: add a
   second `RUN` line that runs `mlir-opt` **without** the option
   (`--convert-math-to-funcs` alone) and pipes into
   `FileCheck %s --check-prefix=NOCVT` — asserting that without
   `convert-ctlz`, no ctlz helper function is generated. Verify with
   `check-part1`, then convince yourself it guards something by pointing
   the NOCVT FileCheck at the *converted* output — you should see
   `error: NOCVT-NOT: excluded string found in input`.
4. Write a brand-new test `test/part1/exp_lowering.mlir`: a function
   containing `%0 = math.exp %a : f32`, a `RUN` line applying
   `--convert-math-to-libm` (a lowering strategy that emits calls to the
   C math library instead of generating the implementation as IR), and
   checks that `math.exp` is gone (`CHECK-NOT`) and a `call @expf`
   appeared. Confirm lit discovers it with no build-system changes, then
   delete it (or keep it — it's a fine test).

## Where to go next

You have now seen the full test workflow this tutorial relies on: write
MLIR by hand, run passes on it, and lock in behavior with lit + FileCheck
tests under `test/`. Time to stop only *using* other people's passes:
[Part 2, Chapter 10: Writing Your First Pass in C++](../../part2-writing-passes/ch10-first-pass-cpp/README.md)
builds a pass from scratch — and `tutorial-opt`, this repo's own
`mlir-opt`, finally enters the `RUN:` lines.
