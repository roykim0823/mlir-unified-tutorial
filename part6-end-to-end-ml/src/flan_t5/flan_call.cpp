// pybind11 extension module exposing the compiled flan-t5-small to Python.
//
//   import flan_call
//   tokens = flan_call.generate_tokens(input_ids, attention_mask)
//
// Python tokenizes and detokenizes; the greedy loop (flan_generate.h) and
// every model call run natively. pybind11/stl.h converts Python lists to
// std::vector<int64_t> and back.
//
// Build: sh run_mlir_pipeline.sh && sh build_flan_call.sh (Chapter 30).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flan_generate.h"

PYBIND11_MODULE(flan_call, m) {
  m.doc() = "flan-t5-small compiled with MLIR: greedy generation";
  m.def("generate_tokens", &generate_tokens,
        "Greedy-decode from encoder token ids; returns the decoder tokens "
        "(starting with the <pad> start token, ending with </s>).",
        pybind11::arg("input_ids"), pybind11::arg("attention_mask"),
        pybind11::arg("verbose") = true);
}
