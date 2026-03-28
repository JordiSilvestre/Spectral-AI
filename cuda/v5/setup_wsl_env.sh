#!/bin/bash
# Setup WSL2 venv for LiquidBit with PyTorch CUDA support
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

PROJECT="/mnt/j/Proyectos/LiquidBit Zero-Matrix"
VENV="$PROJECT/.venv_wsl"

echo "=== Creating WSL2 venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "=== Installing PyTorch with CUDA 12.8 ==="
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing other deps ==="
pip install numpy tqdm

echo "=== Verifying ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
from torch.utils.cpp_extension import CUDAExtension
print('CUDAExtension: OK')
"

echo ""
echo "=== Venv location: $VENV ==="
echo "=== Activate with: source $VENV/bin/activate ==="
