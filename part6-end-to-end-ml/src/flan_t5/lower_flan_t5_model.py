"""Export google/flan-t5-small (encoder-decoder) with dynamic sequence lengths.

T5 is an encoder-decoder: the encoder reads the prompt once, the decoder
produces the answer one token at a time, attending to the encoder output.
Generation therefore calls the model repeatedly with a growing
decoder_input_ids — so the decoder length must be dynamic, and so (to accept
prompts of any length) must the encoder length.

There is no KV cache (use_cache=False): every step recomputes the whole
decoder prefix. Slower, but a pure function torch.export can capture.

Writes:
  flan_t5_linalg.mlir            linalg-on-tensors, func @flan_decoder
                                 (large: weights embedded)
  flan_t5_linalg_stripped.mlir   without the weight blobs (for reading)
"""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoTokenizer, T5ForConditionalGeneration

torch.manual_seed(41)


class T5Wrapper(torch.nn.Module):
    """forward(input_ids, attention_mask, decoder_input_ids) -> logits.

    T5's own forward() takes a dozen optional keyword arguments (labels,
    decoder_attention_mask, output_hidden_states, return_dict, ...) and
    returns a Seq2SeqLMOutput carrying encoder states, cross-attentions and
    caches. torch.export copes badly with optional kwargs, dictionary
    outputs and side state, so the wrapper pins the interface to three
    tensors in and the decoder logits out.

    return_all_logits=False would return only the last position's logits
    (batch, 1, vocab) — all a greedy step needs. We export all positions
    (batch, decoder_len, vocab) so the C++ side can pad the decoder input to
    a fixed length and read the row it wants.
    """

    def __init__(self, model, return_all_logits: bool = True):
        super().__init__()
        self.model = model
        self.return_all_logits = return_all_logits

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        if self.return_all_logits:
            return outputs.logits
        return outputs.logits[:, -1:, :]


def export_flan_t5() -> None:
    # 1. Tokenizer + example input: the sample torch.export traces with.
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    encoded = tokenizer("translate English to German: How are you?",
                        return_tensors="pt")
    print(f"example input_ids: {encoded['input_ids'].tolist()}")

    # 2. Model, KV cache off, wrapped.
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.eval()
    model.config.use_cache = False
    wrapped = T5Wrapper(model)

    # 3. Decoder start: TWO <pad> tokens (id 0, T5's decoder start token).
    # torch.export specializes dimensions whose example size is 0 or 1, so a
    # dynamic dimension needs a non-trivial example length.
    decoder_start = torch.tensor([[tokenizer.pad_token_id] * 2])

    # 4. Dynamic shapes: batch fixed to 1; encoder length shared by
    # input_ids and attention_mask (<= 128); decoder length <= 32.
    encoder_seq_len = torch.export.Dim("encoder_seq_len", max=128)
    decoder_seq_len = torch.export.Dim("decoder_seq_len", max=32)
    ep = torch.export.export(
        wrapped,
        (encoded["input_ids"], encoded["attention_mask"], decoder_start),
        dynamic_shapes={
            "input_ids": {1: encoder_seq_len},
            "attention_mask": {1: encoder_seq_len},
            "decoder_input_ids": {1: decoder_seq_len},
        },
    )

    # 5. Decompose composite ATen ops into primitives the importer knows.
    ep = ep.run_decompositions()

    # 6. Import straight to linalg-on-tensors (export_and_import would call
    # torch.export itself if handed the module; we did it explicitly to
    # pass dynamic_shapes and inspect the program).
    m = fx.export_and_import(ep, output_type=OutputType.LINALG_ON_TENSORS,
                             func_name="flan_decoder")
    mlir_str = str(m)
    with open("flan_t5_linalg.mlir", "w") as f:
        f.write(mlir_str)
    print(f"wrote flan_t5_linalg.mlir ({len(mlir_str):,} bytes)")

    stripped = mlir_str.split("{-#")[0].rstrip() + "\n"
    with open("flan_t5_linalg_stripped.mlir", "w") as f:
        f.write(stripped)
    print(f"wrote flan_t5_linalg_stripped.mlir ({len(stripped):,} bytes)")


if __name__ == "__main__":
    export_flan_t5()
