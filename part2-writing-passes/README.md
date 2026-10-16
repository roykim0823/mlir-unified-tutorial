# Part 2: Writing Passes

First C++: transformations packaged as passes on the `tutorial-opt`
driver. Chapter 10 does everything by hand so you see all the moving parts;
Chapter 11 lets tablegen generate the boilerplate — the shape every later
pass in this repo uses.

| Chapter | One line |
|---|---|
| [ch10: Writing Your First Pass in C++](ch10-first-pass-cpp/README.md) | `PassWrapper`, `walk`, rewrite patterns, and a hand-registered driver (preserved verbatim in [`code/`](ch10-first-pass-cpp/code/)) |
| [ch11: Tablegen for Passes](ch11-tablegen-passes/README.md) | The same passes declared in tablegen: generated base classes, options, docs |

**Prerequisites:** Part 1, plus the repo build
(`cmake` steps in the [README](../README.md)) — from here on you are
compiling `tutorial-opt` yourself. **Next:**
[Part 3](../part3-building-a-dialect/README.md) builds the dialect these
passes will operate on.
