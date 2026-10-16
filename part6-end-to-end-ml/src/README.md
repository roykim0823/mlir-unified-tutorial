# Part 6 model sources

One directory per model: `lower_<model>_model.py` exports it to MLIR (run
these through `../generate_snapshots.sh`), `run_<model>_model.py` prints
PyTorch's reference output, and the C++ callers and `run_mlir_pipeline.sh`
scripts compile and run the MLIR side.

| Model | Chapters |
|---|---|
| [`sample/`](sample/) | ch27–29, ch31 (walkthrough model) |
| [`mnist/`](mnist/) | ch27–29, ch31 |
| [`gpt2/`](gpt2/) | ch27–28 (dynamic shapes; lowered to an object file only) |
| [`cnn/`](cnn/), [`resnet18/`](resnet18/), [`bert/`](bert/), [`flan_t5/`](flan_t5/) | ch30, and ch31 for cnn/resnet18 (snapshots not generated yet) |

`compare_outputs.py` diffs a compiled binary's output against PyTorch's.
