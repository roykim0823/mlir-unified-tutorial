"""Export the small CNN to MLIR via the torch-mlir FX importer.

Same recipe as sample/mnist; the batch size is 4 (static).

Writes:
  cnn_model_torch.mlir    the torch dialect
  cnn_model_linalg.mlir   linalg-on-tensors (weights embedded, ~0.5 MB)
"""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from model import ConvolutionalNetwork

torch.manual_seed(41)


def export(output_type: OutputType, filename: str) -> None:
    # Re-seed before every construction so each export — and the reference
    # run in run_cnn_model.py — sees identical random weights.
    torch.manual_seed(41)
    m = fx.export_and_import(
        ConvolutionalNetwork(),
        torch.randn(4, 1, 28, 28),
        output_type=output_type,
        func_name="cnn_model",
    )
    with open(filename, "w") as f:
        f.write(str(m))
    print(f"wrote {filename}")


if __name__ == "__main__":
    export(OutputType.TORCH, "cnn_model_torch.mlir")
    export(OutputType.LINALG_ON_TENSORS, "cnn_model_linalg.mlir")
