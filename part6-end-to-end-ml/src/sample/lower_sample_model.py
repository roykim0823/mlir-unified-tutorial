"""Export the sample model to MLIR via the torch-mlir FX importer.

Writes three snapshots of the same model, one per abstraction level:
  sample_model_torch.mlir    the torch dialect (backend-contract form)
  sample_model_linalg.mlir   linalg-on-tensors (no torch ops left)
"""

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from model import Sample

torch.manual_seed(41)


def export(output_type: OutputType, filename: str) -> None:
    # Re-seed before every construction so each export — and the reference
    # run in run_sample_model.py — sees identical random weights.
    torch.manual_seed(41)
    m = fx.export_and_import(
        Sample(),
        torch.randn(3, 4),
        output_type=output_type,
        func_name="sample_model",
    )
    with open(filename, "w") as f:
        f.write(str(m))
    print(f"wrote {filename}")


if __name__ == "__main__":
    export(OutputType.TORCH, "sample_model_torch.mlir")
    export(OutputType.LINALG_ON_TENSORS, "sample_model_linalg.mlir")
