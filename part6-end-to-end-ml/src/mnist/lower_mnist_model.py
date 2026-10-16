"""Export the MNIST MLP to MLIR via the torch-mlir FX importer."""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from model import MnistNetwork

torch.manual_seed(41)


def export(output_type: OutputType, filename: str) -> None:
    # Re-seed before every construction so each export — and the reference
    # run in run_mnist_model.py — sees identical random weights.
    torch.manual_seed(41)
    m = fx.export_and_import(
        MnistNetwork(),
        torch.randn(1, 1, 28, 28),
        output_type=output_type,
        func_name="mnist_model",
    )
    with open(filename, "w") as f:
        f.write(str(m))
    print(f"wrote {filename}")


if __name__ == "__main__":
    export(OutputType.TORCH, "mnist_model_torch.mlir")
    export(OutputType.LINALG_ON_TENSORS, "mnist_model_linalg.mlir")
