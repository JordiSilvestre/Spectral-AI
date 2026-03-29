#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
export HOME=/home/jordi

# El linker de ld no soporta paths con espacios.
# Solución: symlink sin espacios al proyecto y a la venv.
PROJ_LINK="/tmp/spectral"
rm -f "$PROJ_LINK"
ln -s "/mnt/j/Proyectos/SpectralAI Zero-Matrix" "$PROJ_LINK"

# Crear venv en path sin espacios si no existe
VENV="/home/jordi/liquidbit_venv"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "=== Creando venv en $VENV ==="
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip -q
    pip install torch --index-url https://download.pytorch.org/whl/cu128 -q
    pip install numpy tqdm ninja -q
else
    source "$VENV/bin/activate"
fi

echo "=== Python: $(python3 --version) ==="
echo "=== nvcc: $(nvcc --version 2>&1 | tail -1) ==="
echo "=== torch: $(python3 -c 'import torch; print(torch.__version__)') ==="
echo ""

cd "$PROJ_LINK"
python3 cuda/v5/build_ext.py 2>&1
