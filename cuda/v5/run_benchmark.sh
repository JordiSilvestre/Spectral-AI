#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

cd "/mnt/j/Proyectos/LiquidBit Zero-Matrix"

echo "=== Running benchmark_cuda_e2e.py ==="
python3 python/benchmark_cuda_e2e.py 2>&1
