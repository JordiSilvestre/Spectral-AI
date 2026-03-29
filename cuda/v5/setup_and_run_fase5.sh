#!/bin/bash
# setup_and_run_fase5.sh — Setup symlink + run FASE 5 benchmark
set -e

export HOME=/home/jordi
VENV="$HOME/liquidbit_venv"
PROJECT_WIN="/mnt/j/Proyectos/SpectralAI Zero-Matrix"
PROJECT="/tmp/spectral"

# Recrear symlink si no existe
if [ ! -L "$PROJECT" ]; then
    ln -sf "$PROJECT_WIN" "$PROJECT"
    echo "Symlink creado: $PROJECT -> $PROJECT_WIN"
fi

# Verificar venv
if [ ! -d "$VENV" ]; then
    echo "ERROR: venv no encontrado en $VENV"
    echo "Ejecuta primero: python -m venv $VENV && source $VENV/bin/activate && pip install torch tiktoken datasets tqdm numpy"
    exit 1
fi

source "$VENV/bin/activate"

echo "============================================"
echo "FASE 5 — Benchmark de Cuantización"
echo "============================================"
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"
echo ""

cd "$PROJECT/python"

# Ejecutar benchmark
python benchmark_expert_types.py 2>&1 | tee "$PROJECT/data/fase5_benchmark_output.txt"
