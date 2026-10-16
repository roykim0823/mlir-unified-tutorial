"""Run HuggingFace ResNet-18 in PyTorch on an all-ones 224x224 image.

The printed logits are the reference the compiled C++ binary
(resnet18_call.cpp) must reproduce.
"""

import os

# Limit to a single thread for reproducibility.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from transformers import AutoModelForImageClassification

torch.manual_seed(41)

if __name__ == "__main__":
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    model.eval()
    x = torch.ones((1, 3, 224, 224))
    with torch.no_grad():
        logits = model(x).logits
    torch.set_printoptions(precision=5)
    print("Input shape:        ", x.shape)
    print("Output logits shape:", logits.shape)
    print("First 10 logits:    ", logits[0, :10])
    print("Top class index:    ", logits.argmax(dim=1).item())
