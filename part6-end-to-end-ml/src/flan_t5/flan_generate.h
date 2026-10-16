// Greedy autoregressive generation over the compiled flan-t5-small.
//
// Shared by flan_call.cpp (the pybind11 module) and flan_call_benchmark.cpp.
// The compiled function is
//   flan_decoder(input_ids: [1, enc_len] i64, attention_mask: [1, enc_len] i64,
//                decoder_input_ids: [1, dec_len] i64) -> logits [1, dec_len, vocab]
// with enc_len <= 128 and dec_len <= 32 dynamic (lower_flan_t5_model.py).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

const int64_t VOCAB_SIZE = 32128;
const int64_t PAD_TOKEN = 0; // also T5's decoder start token
const int64_t EOS_TOKEN = 1;
const int64_t MAX_DECODER_LEN = 32; // = Dim("decoder_seq_len", max=32)
const int MAX_STEPS = 20;

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
// Exactly the wrapper's three inputs, in order. (The source tutorial
// declared a fourth, decoder_attention_mask, that the exported function
// does not have; the extra pointer was silently ignored.)
void _mlir_ciface_flan_decoder(MemRefDescriptor<float, 3> *output,
                               MemRefDescriptor<int64_t, 2> *input_ids,
                               MemRefDescriptor<int64_t, 2> *attention_mask,
                               MemRefDescriptor<int64_t, 2> *decoder_input_ids);
}

inline std::vector<int64_t> generate_tokens(std::vector<int64_t> input_ids,
                                            std::vector<int64_t> attention_mask,
                                            bool verbose = true) {
  int64_t input_size = static_cast<int64_t>(input_ids.size());
  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      input_ids.data(), input_ids.data(), 0, {1, input_size}, {input_size, 1}};
  MemRefDescriptor<int64_t, 2> attention_mask_MemRef = {
      attention_mask.data(), attention_mask.data(), 0, {1, input_size},
      {input_size, 1}};

  // The decoder input is always passed padded to MAX_DECODER_LEN. That is
  // safe because T5's decoder self-attention is causal: the logits at
  // position t depend only on tokens 0..t, so trailing pads cannot change
  // the row we read. (It also wastes work — see the Chapter 30 exercises.)
  int64_t decoder_input_ids[MAX_DECODER_LEN];

  std::vector<int64_t> generated_tokens = {PAD_TOKEN};

  for (int step = 0; step < MAX_STEPS; step++) {
    int64_t current_len = static_cast<int64_t>(generated_tokens.size());
    if (current_len > MAX_DECODER_LEN) {
      std::cerr << "error: sequence exceeds max decoder length ("
                << MAX_DECODER_LEN << ")\n";
      break;
    }
    for (int64_t i = 0; i < MAX_DECODER_LEN; i++)
      decoder_input_ids[i] = i < current_len ? generated_tokens[i] : PAD_TOKEN;

    MemRefDescriptor<int64_t, 2> decoder_input_ids_MemRef = {
        decoder_input_ids, decoder_input_ids, 0, {1, MAX_DECODER_LEN},
        {MAX_DECODER_LEN, 1}};

    // Out-slot: overwritten with the model-allocated [1, 32, 32128] result
    // (Chapter 28, section 3) — freed at the end of the step.
    MemRefDescriptor<float, 3> outputMemRef = {};

    _mlir_ciface_flan_decoder(&outputMemRef, &input_ids_MemRef,
                              &attention_mask_MemRef,
                              &decoder_input_ids_MemRef);

    // Logits of the LAST real position: [0, current_len - 1, :].
    const float *row = outputMemRef.aligned + outputMemRef.offset +
                       (current_len - 1) * outputMemRef.strides[1];
    int64_t vstride = outputMemRef.strides[2];

    std::vector<std::pair<float, int64_t>> scores;
    scores.reserve(VOCAB_SIZE);
    for (int64_t j = 0; j < outputMemRef.sizes[2]; j++) {
      float logit = row[j * vstride];
      if (!std::isnan(logit) && !std::isinf(logit)) // filter invalid values
        scores.push_back({logit, j});
    }
    free(outputMemRef.allocated);
    if (scores.empty()) {
      std::cerr << "error: no valid logits at step " << step << "\n";
      break;
    }

    int top = std::min<int>(5, static_cast<int>(scores.size()));
    std::partial_sort(scores.begin(), scores.begin() + top, scores.end(),
                      [](const auto &a, const auto &b) {
                        return a.first > b.first;
                      });
    int64_t next_token = scores[0].second; // greedy

    if (verbose) {
      std::cout << "step " << step << ": decoder [";
      for (size_t j = 0; j < generated_tokens.size(); ++j)
        std::cout << (j ? ", " : "") << generated_tokens[j];
      std::cout << "] -> top 5:";
      for (int j = 0; j < top; j++)
        std::cout << ' ' << scores[j].second << " (" << std::fixed
                  << std::setprecision(4) << scores[j].first << ")";
      std::cout << "\n";
    }

    generated_tokens.push_back(next_token);
    if (next_token == EOS_TOKEN)
      break;
  }
  return generated_tokens;
}
