"""Export HuggingFace ResNet-18 (microsoft/resnet-18) to MLIR.

ResNet-18 is the first model in this part with *buffers*: every BatchNorm
layer carries running_mean, running_var and num_batches_tracked, which are
module state but not parameters. torch.export lifts all state into graph
inputs, and torch-mlir's import_frozen_program inlines the *parameters* as
dense_resource literals — but not the buffers (see Chapter 27, section 7).
Exported as-is, the model function takes 61 arguments: the image plus
60 BatchNorm buffers, 20 of which (num_batches_tracked) are never used.

This script writes both versions:
  resnet18_model_torch_lifted.mlir  torch dialect, buffers lifted to
                                    arguments (the problem, for reading)
  resnet18_model_linalg.mlir        linalg-on-tensors after freeze_buffers():
                                    one argument, all state inlined
  resnet18_model_linalg_stripped.mlir   the same without the weight blobs
                                        (small; for reading only)

The source tutorial instead pasted the buffers into the torch-dialect file by
hand (get_buffers_in_mlir_format.py); freeze_buffers() automates that step.
"""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType
from transformers import AutoModelForImageClassification

torch.manual_seed(41)


def load_model():
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    model.eval()
    return model


def freeze_buffers(model: torch.nn.Module) -> torch.nn.Module:
    """Turn every buffer into a frozen parameter, so the importer inlines it.

    torch-mlir's import_frozen_program freezes inputs_to_parameters (and
    lifted tensor constants) into literals but leaves inputs_to_buffers as
    function arguments. Re-registering each buffer as a requires_grad=False
    Parameter moves it into the first category. num_batches_tracked is only
    read in training mode, so it is dropped instead.
    """
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            del module._buffers[name]
            if name == "num_batches_tracked":
                module._buffers[name] = None
            else:
                module.register_parameter(
                    name, torch.nn.Parameter(buf, requires_grad=False))
    return model


def write(m, filename: str) -> str:
    mlir_str = str(m)
    with open(filename, "w") as f:
        f.write(mlir_str)
    print(f"wrote {filename} ({len(mlir_str):,} bytes)")
    return mlir_str


def export_resnet18() -> None:
    example = torch.randn(1, 3, 224, 224)

    # 1. As-is: the BatchNorm buffers become function arguments.
    m = fx.export_and_import(load_model(), example,
                             output_type=OutputType.TORCH, func_name="resnet18")
    write(m, "resnet18_model_torch_lifted.mlir")

    # 2. Buffers frozen: a clean one-argument function.
    m = fx.export_and_import(freeze_buffers(load_model()), example,
                             output_type=OutputType.LINALG_ON_TENSORS,
                             func_name="resnet18")
    mlir_str = write(m, "resnet18_model_linalg.mlir")

    stripped = mlir_str.split("{-#")[0].rstrip() + "\n"
    with open("resnet18_model_linalg_stripped.mlir", "w") as f:
        f.write(stripped)
    print(f"wrote resnet18_model_linalg_stripped.mlir ({len(stripped):,} bytes)")


if __name__ == "__main__":
    export_resnet18()
