"""Export HuggingFace GPT-2 to MLIR with a dynamic sequence length.

GPT-2 is a causal (decoder-only) language model: it takes input_ids and an
attention_mask and returns logits for every position. For autoregressive
generation you append the chosen token to input_ids and call it again, so
the sequence length must be a *dynamic* dimension in the exported program.

Two-step lowering: fx.export_and_import to the TORCH backend contract, then
the torch-backend-to-linalg-on-tensors-backend-pipeline. (The direct
LINALG_ON_TENSORS export path fails for GPT-2 — see the chapter text.)

Writes:
  gpt2_torch.mlir             torch dialect, full (large: weights embedded)
  gpt2_linalg.mlir            linalg-on-tensors, full
  gpt2_linalg_stripped.mlir   linalg-on-tensors without the weight blobs
                              (small; for reading and lit tests only)
"""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(41)


class GPT2Wrapper(torch.nn.Module):
    """Expose a clean forward(input_ids, attention_mask) -> logits.

    The raw HuggingFace forward() takes many optional keyword arguments and
    returns a dictionary-like object with caches and hidden states —
    torch.export cannot digest that. The wrapper pins the signature down to
    the two tensors we need and returns only the logits tensor.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def export_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    input_text = "The capital of France is"
    encoded = tokenizer(input_text, return_tensors="pt")
    print(f"input: '{input_text}' -> input_ids shape {encoded['input_ids'].shape}")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    # Disable the KV cache: cache tensors are side state that torch.export
    # cannot represent as a pure function of the inputs.
    model.config.use_cache = False

    wrapped = GPT2Wrapper(model)

    # Mark dimension 1 (sequence length) of both inputs as dynamic, shared,
    # and bounded: the compiled artifact will accept 2..128 tokens.
    seq_len = torch.export.Dim("seq_len", max=128)
    ep = torch.export.export(
        wrapped,
        (encoded["input_ids"], encoded["attention_mask"]),
        dynamic_shapes={
            "input_ids": {1: seq_len},
            "attention_mask": {1: seq_len},
        },
    )
    # Decompose composite ATen ops into primitives torch-mlir can import.
    ep = ep.run_decompositions()

    m = fx.export_and_import(ep, output_type=OutputType.TORCH,
                             func_name="gpt2_model")
    with open("gpt2_torch.mlir", "w") as f:
        f.write(str(m))
    print("wrote gpt2_torch.mlir")

    # Step 2: the same lowering LINALG_ON_TENSORS would have run internally.
    run_pipeline_with_repro_report(
        m,
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "lowering torch backend contract to linalg-on-tensors",
    )
    mlir_str = str(m)
    with open("gpt2_linalg.mlir", "w") as f:
        f.write(mlir_str)
    print(f"wrote gpt2_linalg.mlir ({len(mlir_str):,} bytes)")

    # A readable snapshot: drop the dialect_resources blob holding the
    # weights. NOT runnable — for inspection and lit tests only.
    stripped = mlir_str.split("{-#")[0].rstrip() + "\n"
    with open("gpt2_linalg_stripped.mlir", "w") as f:
        f.write(stripped)
    print(f"wrote gpt2_linalg_stripped.mlir ({len(stripped):,} bytes)")


if __name__ == "__main__":
    export_gpt2()
