"""GPU / CUDA driver sanity check — verifies PyTorch CUDA build and nvidia-smi."""
import torch
import subprocess
import sys

# 1. Check PyTorch CUDA build
cuda_version = torch.version.cuda
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch CUDA Build: {cuda_version}")
if cuda_version is None:
    print("PROBLEM: CPU-only PyTorch installed!")

# 2. Check NVIDIA Driver
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Driver: OK (nvidia-smi works)")
        print(result.stdout.split('\n')[3]) # Shows driver version
    else:
        print("PROBLEM: nvidia-smi failed")
except FileNotFoundError:
    print("PROBLEM: nvidia-smi not found on PATH")
