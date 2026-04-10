#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Compilando test_router ==="
nvcc -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_120,code=sm_120 \
  --use_fast_math -Xcompiler -fPIC \
  --expt-relaxed-constexpr \
  --maxrregcount=64 -DNDEBUG \
  -o test_router test_router.cu -lcudart 2>&1
echo "Compile exit: $?"

echo ""
echo "=== Ejecutando test_router ==="
./test_router 2>&1
echo "Test exit: $?"
