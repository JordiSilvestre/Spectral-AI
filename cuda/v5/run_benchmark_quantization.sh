#!/bin/bash
# run_benchmark_quantization.sh — FASE 5: Benchmark de Cuantización
# Ejecutar desde WSL2: bash /tmp/spectral/cuda/v5/run_benchmark_quantization.sh

set -e

export HOME=/home/jordi
VENV="$HOME/liquidbit_venv"
PROJECT="/tmp/spectral"

# Activar venv
source "$VENV/bin/activate"

echo "============================================"
echo "FASE 5 — Benchmark de Cuantización"
echo "============================================"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")')"
echo ""

cd "$PROJECT/python"

# Ejecutar benchmark
python benchmark_expert_types.py 2>&1 | tee "$PROJECT/data/fase5_benchmark_output.txt"

echo ""
echo "Output guardado en: $PROJECT/data/fase5_benchmark_output.txt"
echo "JSON guardado en: $PROJECT/data/fase5_quantization_benchmark.json"
