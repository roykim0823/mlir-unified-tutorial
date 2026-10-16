"""Dump ResNet-18's buffers as MLIR resource blobs (the source tutorial's
manual alternative to freeze_buffers() in lower_resnet18_model.py).

Writes resnet18_params.txt with two blocks:
  1. resource entries to paste into the file's `dialect_resources: {builtin:
     {...}}` section:
       torch_tensor_custom_<i>.float32: "0x04000000<hex floats>",
  2. one literal per buffer argument, to paste at the top of the function
     body (after deleting %arg<i> from the signature):
       %arg<i> = torch.vtensor.literal(dense_resource<...> : tensor<Nxf32>)
                 : !torch.vtensor<[N],f32>

A resource blob is a 4-byte little-endian alignment header (0x04000000 =
alignment 4) followed by the raw little-endian element bytes.

<i> is the buffer's position in named_buffers(), which is also its argument
number in resnet18_model_torch_lifted.mlir: torch.export places lifted
state before the user input, in module order. The scalar
num_batches_tracked buffers are skipped — they are the dead arguments; just
delete them from the signature.
"""

import struct

import torch.utils._pytree as pytree
from transformers import AutoModelForImageClassification

resnet18 = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
resnet18.eval()


def tensor_to_hex(tensor) -> str:
    """Convert a float32 tensor into an MLIR resource hex literal."""
    values = tensor.float().detach().cpu().flatten().numpy()
    return "0x04000000" + "".join(
        struct.pack("<f", v).hex().upper() for v in values)


def get_params() -> None:
    params = dict(resnet18.named_buffers(remove_duplicate=False))
    params_flat, _ = pytree.tree_flatten(params)

    with open("resnet18_params.txt", "w") as f:
        tensor_sizes = []
        for i, tensor in enumerate(params_flat):
            if tensor.dim() == 0:  # num_batches_tracked: unused, skip
                continue
            hex_blob = tensor_to_hex(tensor)
            f.write(f'torch_tensor_custom_{i}.float32: "{hex_blob}",\n')
            tensor_sizes.append((i, tensor.numel()))

        for i, n in tensor_sizes:
            f.write(
                f"%arg{i} = torch.vtensor.literal("
                f"dense_resource<torch_tensor_custom_{i}.float32> "
                f": tensor<{n}xf32>) : !torch.vtensor<[{n}],f32>\n")
    print(f"wrote resnet18_params.txt ({len(tensor_sizes)} buffers)")


if __name__ == "__main__":
    get_params()
