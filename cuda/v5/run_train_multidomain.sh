#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
export HOME=/home/jordi

PROJ_LINK="/tmp/spectral"
rm -f "$PROJ_LINK"
ln -s "/mnt/j/Proyectos/SpectralAI Zero-Matrix" "$PROJ_LINK"

source /home/jordi/liquidbit_venv/bin/activate
cd "$PROJ_LINK"

# Deps extra
pip install tiktoken datasets -q 2>&1 | tail -3

echo ""
echo "=== Training Multi-Dominio FASE 4 ==="
echo ""
python3 python/train_multi_domain.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-4 \
    --expert_dim 128 \
    --expert_layers 2 \
    --alpha_router 0.1 \
    --device cuda \
    2>&1
