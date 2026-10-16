"""Write bert-base-uncased's buffers to tensor.csv for bert_call.cpp.

The exported bert_model function takes the model's two buffers
(embeddings.position_ids and embeddings.token_type_ids, both int64 [1, 512])
as its first two arguments — torch-mlir does not freeze buffers (Chapter
27, section 7). bert_call.cpp loads them from this file at run time: one
CSV row per buffer, in named_buffers() order, which is also the argument
order.
"""

import numpy as np
from transformers import BertModel

if __name__ == "__main__":
    model = BertModel.from_pretrained("bert-base-uncased")
    buffers = dict(model.named_buffers(remove_duplicate=False))

    with open("tensor.csv", "w") as f:
        for name, value in buffers.items():
            value = value.numpy()
            print(f"{name}: shape {value.shape}, dtype {value.dtype}")
            np.savetxt(f, [value.flatten()], delimiter=",", fmt="%d")
    print(f"wrote tensor.csv ({len(buffers)} rows)")
