import torch
import torchao

print("Torch version:", torch.__version__)
print("TorchAO imported successfully")

from torchao.quantization import int4_weight_only
print("INT4 weight-only quantization available")
