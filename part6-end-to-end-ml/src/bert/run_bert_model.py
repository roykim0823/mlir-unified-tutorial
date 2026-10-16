"""Run bert-base-uncased in PyTorch on the export's example sentence.

Prints exactly what bert_call.cpp prints — the first 10 values of each of
the 12 last_hidden_state vectors and the first 10 pooler_output values —
so the two can be compared line by line (or with compare_outputs.py).
"""

import torch
from transformers import BertModel, BertTokenizer

TEXT = "This is a sample input for BERT model export."

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    encoded = tokenizer(TEXT, return_tensors="pt")
    print("input_ids:", encoded["input_ids"].tolist())

    with torch.no_grad():
        output = model(**encoded)
    hidden = output.last_hidden_state  # [1, 12, 768]
    pooled = output.pooler_output      # [1, 768]
    print("shapes:", tuple(hidden.shape), tuple(pooled.shape))

    print("Last hidden state (first 10 values of each of the 12 vectors):")
    for row in hidden[0]:
        print(" ".join(f"{v:.5f}" for v in row[:10].tolist()))
    print("\nPooler output:")
    print(" ".join(f"{v:.5f}" for v in pooled[0, :10].tolist()))
