"""Greedy autoregressive generation with HuggingFace GPT-2, in PyTorch.

Prints the next-token predictions the compiled artifact should match.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(41)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    model.config.use_cache = False

    text = "The capital of France is"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    for step in range(10):
        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
        next_token = logits[0, -1].argmax().reshape(1, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype)],
            dim=1,
        )
        if next_token.item() == tokenizer.eos_token_id:
            break

    print(tokenizer.decode(input_ids[0]))
