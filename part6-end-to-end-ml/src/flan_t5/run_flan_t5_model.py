"""Translate one sentence with flan-t5-small in PyTorch (greedy generate).

The token ids printed here are the reference for the MLIR-compiled model
driven by run_flan_t5_model_mlir.py.
"""

from transformers import AutoTokenizer, T5ForConditionalGeneration

INPUT_TEXT = "translate English to German: What is your name?"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.eval()

    encoded = tokenizer(INPUT_TEXT, return_tensors="pt")
    print(f'Input text: "{INPUT_TEXT}"')
    print("Input IDs:     ", encoded.input_ids.tolist())
    print("Attention mask:", encoded.attention_mask.tolist())

    outputs = model.generate(encoded.input_ids)  # greedy by default
    print("Output IDs:    ", outputs.tolist())
    print("Decoded output:", tokenizer.decode(outputs[0]))
