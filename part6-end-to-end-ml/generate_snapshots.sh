#!/bin/sh
# Regenerate the checked-in MLIR snapshots for Part 6.
#
# torch-mlir wheels are Linux-only, so on macOS this runs in a Linux
# container (works on Apple Silicon: the wheels ship for aarch64 too).
# On Linux you can instead create a venv, `pip install -r requirements.txt`,
# and run the lower_*.py scripts directly.
#
# Usage (from part6-end-to-end-ml/):
#   ./generate_snapshots.sh                 # sample mnist gpt2
#   ./generate_snapshots.sh sample mnist    # a subset
#   ./generate_snapshots.sh cnn resnet18 bert flan_t5   # the Chapter 30 models
#
# For each model directory src/<model>/ this runs, in order:
#   lower_<model>_model.py   the export (required)
#   get_buffers.py           if present: dumps lifted buffers (bert's tensor.csv)
#   run_<model>_model.py     if present: prints the PyTorch reference output
# The HuggingFace models (gpt2, resnet18, bert, flan_t5) are downloaded on
# first use; bert's and flan_t5's full exports are several hundred MB each.
set -eu

MODELS="${*:-sample mnist gpt2}"

for model in $MODELS; do
  if [ ! -f "src/$model/lower_${model}_model.py" ]; then
    echo "error: src/$model/lower_${model}_model.py not found" >&2
    exit 1
  fi
done

docker run --rm -v "$PWD":/work -w /work python:3.12-slim-bookworm sh -c "
  set -eu
  pip install --quiet --no-cache-dir -r requirements.txt
  for model in $MODELS; do
    echo \"=== \$model\"
    cd /work/src/\$model
    python lower_\${model}_model.py
    if [ -f get_buffers.py ]; then python get_buffers.py; fi
    if [ -f run_\${model}_model.py ]; then python run_\${model}_model.py; fi
  done
"
