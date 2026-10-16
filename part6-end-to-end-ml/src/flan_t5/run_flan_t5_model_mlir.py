"""Translate one sentence with the MLIR-compiled flan-t5-small.

Python does the tokenizing and detokenizing; the generation loop and every
model call run in native code (flan_call.cpp + the compiled flan_decoder),
reached through the pybind11 extension module `flan_call`.

Build the extension first (from this directory):
  sh run_mlir_pipeline.sh   # flan_t5_model.o (position-independent)
  sh build_flan_call.sh     # flan_call<ext-suffix>.so
Needs only `transformers` (for the tokenizer) on the host — no torch.
"""

from transformers import AutoTokenizer

import flan_call

INPUT_TEXT = "translate English to German: What is your name?"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    encoded = tokenizer(INPUT_TEXT)
    print(f'Input text: "{INPUT_TEXT}"')
    print("Input IDs:     ", encoded["input_ids"])
    print("Attention mask:", encoded["attention_mask"])

    tokens = flan_call.generate_tokens(encoded["input_ids"],
                                       encoded["attention_mask"])
    print("Generated tokens:", tokens)
    print("Decoded output:  ", tokenizer.decode(tokens))
