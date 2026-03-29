#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
export HOME=/home/jordi

# Symlink sin espacios
PROJ_LINK="/tmp/spectral"
rm -f "$PROJ_LINK"
ln -s "/mnt/j/Proyectos/SpectralAI Zero-Matrix" "$PROJ_LINK"

source /home/jordi/liquidbit_venv/bin/activate
cd "$PROJ_LINK"

echo "=== Benchmark Final E2E ==="
echo ""
python3 python/benchmark_e2e_final.py 2>&1
