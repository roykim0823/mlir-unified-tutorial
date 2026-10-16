// Chapter 24: the pure-Python demos. A destructive (greedy) rewriter gets
// stuck depending on rule order; the e-graph does not. Equality saturation
// must extract the cheapest equivalent form of each expression, and the
// inspection script must show the saturated e-graph (nothing deleted; a and
// (a*2)/2 share an e-class).
//
// RUN: python3 %project_source_dir/part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/greedy_demo.py | FileCheck %s --check-prefix=GREEDY
// RUN: python3 %project_source_dir/part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/optimize_demo.py | FileCheck %s --check-prefix=DEMO
// RUN: python3 %project_source_dir/part5-structured-and-parallel/ch24-equality-saturation-egraphs/examples/inspect_egraph.py | FileCheck %s --check-prefix=INSPECT

// GREEDY: cancel; then strength-reduce   ->  a
// GREEDY-NEXT: strength-reduce; then cancel   ->  ((a shl 1) / 2)   (stuck)
// GREEDY: visiting parents first         ->  a
// GREEDY-NEXT: visiting operands first        ->  ((a shl 1) / 2)   (stuck)

// DEMO: (a*2)/2           ->  a
// DEMO-NEXT: e^x * e^x * e^x   ->  exp(((x + x) + x))
// DEMO-NEXT: (A^T)^T           ->  A
// DEMO-NEXT: All e-graph demos succeeded.

// INSPECT-LABEL: after saturate: root = c3
// INSPECT: c2: { *(c3, c1), shl(c3, c4) }
// INSPECT-NEXT: c3: { /(c2, c1), a }
// INSPECT: extract(root) = a
