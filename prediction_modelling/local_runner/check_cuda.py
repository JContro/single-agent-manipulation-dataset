import torch
import sys
import os
print(torch.__file__)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Try to create a CUDA tensor
try:
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Successfully created CUDA tensor: {x.device}")
except Exception as e:
    print(f"Error creating CUDA tensor: {str(e)}")