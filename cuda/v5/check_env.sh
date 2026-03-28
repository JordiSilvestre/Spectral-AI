#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
export HOME=/root

echo "=== CUDA ==="
nvcc --version 2>&1

echo ""
echo "=== Python ==="
python3 --version

echo ""
echo "=== PyTorch ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1

echo ""
echo "=== torch.utils.cpp_extension ==="
python3 -c "from torch.utils.cpp_extension import CUDAExtension; print('CUDAExtension available')" 2>&1

echo ""
echo "=== pip packages ==="
pip3 list 2>/dev/null | grep -E "torch|numpy|tqdm|tiktoken" 2>&1

echo ""
echo "=== Venv check ==="
ls -la "/mnt/j/Proyectos/LiquidBit Zero-Matrix/.venv/" 2>&1

echo ""
echo "=== libbvh_router.so ==="
ls -la "/mnt/j/Proyectos/LiquidBit Zero-Matrix/cuda/v5/libbvh_router.so" 2>&1
nm -D "/mnt/j/Proyectos/LiquidBit Zero-Matrix/cuda/v5/libbvh_router.so" 2>&1 | grep bvh_router | head -5
