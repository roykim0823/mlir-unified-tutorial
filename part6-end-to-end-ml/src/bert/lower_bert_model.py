"""Export HuggingFace bert-base-uncased to MLIR.

Three things are new relative to GPT-2 (Chapter 27, section 5):

* two outputs — the wrapper returns (last_hidden_state, pooler_output), and
  the compiled function returns both (one struct on the C side);
* two lifted buffers — BertEmbeddings keeps `position_ids` and
  `token_type_ids` as buffers, which torch-mlir leaves as function
  arguments (Chapter 27, section 7). This port keeps them that way on
  purpose: get_buffers.py dumps them to tensor.csv and bert_call.cpp
  feeds them in at run time, so you see the problem *and* a runtime fix.
  (`token_type_ids` is a dead argument: we pass explicit token_type_ids.)
* aten.any.dim — part of BERT's attention-mask handling; the source
  tutorial needed a patched torch-mlir to lower it (Chapter 27, section 8).

Two-step lowering, as for GPT-2: import to TORCH, then run the backend
pipeline.

Writes:
  bert_torch.mlir             torch dialect (large: weights embedded)
  bert_linalg.mlir            linalg-on-tensors (large)
  bert_linalg_stripped.mlir   linalg-on-tensors without the weight blobs
"""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)
from transformers import BertModel, BertTokenizer

torch.manual_seed(41)

TEXT = "This is a sample input for BERT model export."


class BertWrapper(torch.nn.Module):
    """forward(input_ids, token_type_ids, attention_mask)
    -> (last_hidden_state, pooler_output)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state, outputs.pooler_output


def export_bert() -> None:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    encoded = tokenizer(TEXT, return_tensors="pt")
    print(f"input: '{TEXT}' -> input_ids {encoded['input_ids'].tolist()}")

    ep = torch.export.export(
        BertWrapper(model),
        (encoded["input_ids"], encoded["token_type_ids"],
         encoded["attention_mask"]),
    )
    ep = ep.run_decompositions()

    # Step 1: torch dialect.
    m = fx.export_and_import(ep, output_type=OutputType.TORCH,
                             func_name="bert_model")
    with open("bert_torch.mlir", "w") as f:
        f.write(str(m))
    print("wrote bert_torch.mlir")

    # Step 2: what OutputType.LINALG_ON_TENSORS would run internally. The CLI
    # equivalent is:
    #   torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline \
    #       bert_torch.mlir -o bert_linalg.mlir
    run_pipeline_with_repro_report(
        m,
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "lowering torch backend contract to linalg-on-tensors",
    )
    mlir_str = str(m)
    with open("bert_linalg.mlir", "w") as f:
        f.write(mlir_str)
    print(f"wrote bert_linalg.mlir ({len(mlir_str):,} bytes)")

    stripped = mlir_str.split("{-#")[0].rstrip() + "\n"
    with open("bert_linalg_stripped.mlir", "w") as f:
        f.write(stripped)
    print(f"wrote bert_linalg_stripped.mlir ({len(stripped):,} bytes)")


if __name__ == "__main__":
    export_bert()
