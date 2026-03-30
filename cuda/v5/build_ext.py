"""
build_ext.py -- Compilar la extension PyTorch del BVH Router
SpectralAI v5.0

Uso desde WSL2:
    cd /tmp/spectral
    source .venv_wsl/bin/activate
    python cuda/v5/build_ext.py

La extension se instala como modulo importable 'bvh_router_ext'.
"""

import os
import shutil
import sys
from pathlib import Path

# Asegurar que estamos en el directorio del proyecto
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)

import torch
import torch.utils.cpp_extension as cpp_ext

print(f"PyTorch {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")

# Detectar arquitecturas CUDA soportadas
cuda_arch_flags = []
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    arch = f"{cap[0]}{cap[1]}"
    cuda_arch_flags = [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    print(f"Compilando para sm_{arch}")

# ============================================================================
# Workaround for spaces in paths (ninja linker can't handle them)
# Strategy: symlink torch/lib to a safe path, then monkey-patch torch's
# internal library_paths() so the generated ninja file uses our symlink.
# ============================================================================

build_dir = os.path.expanduser("~/.cache/torch_extensions/bvh_router_ext")
os.makedirs(build_dir, exist_ok=True)

# Copy .cu source to a path without spaces
cu_source = project_root / "cuda" / "v5" / "bvh_torch_ext.cu"
safe_source = Path(build_dir) / "bvh_torch_ext.cu"
shutil.copy2(cu_source, safe_source)

# Symlink entire torch package dir so ALL derived paths are space-free.
# PyTorch's cpp_extension derives -L paths from torch.__file__ in multiple
# places; patching library_paths alone misses the second -L injection.
torch_pkg_dir = str(Path(torch.__file__).parent)
torch_lib_dir = str(Path(torch_pkg_dir) / "lib")
needs_patch = " " in torch_pkg_dir
safe_torch_lib = torch_lib_dir

_original_torch_file = torch.__file__
_original_torch_path = list(torch.__path__) if torch.__path__ else []

if needs_patch:
    safe_torch_pkg = "/tmp/_torch_pkg_bvh"
    if os.path.islink(safe_torch_pkg):
        os.unlink(safe_torch_pkg)
    os.symlink(torch_pkg_dir, safe_torch_pkg)
    print(f"Torch pkg symlinked: {safe_torch_pkg} -> {torch_pkg_dir}")

    # Redirect ALL internal references through the symlink
    torch.__file__ = torch.__file__.replace(torch_pkg_dir, safe_torch_pkg)
    if torch.__path__:
        torch.__path__[0] = safe_torch_pkg

    safe_torch_lib = str(Path(safe_torch_pkg) / "lib")

    # Patch module-level path variables computed at import time
    if hasattr(cpp_ext, '_TORCH_PATH'):
        cpp_ext._TORCH_PATH = safe_torch_pkg
    if hasattr(cpp_ext, 'TORCH_LIB_PATH'):
        cpp_ext.TORCH_LIB_PATH = safe_torch_lib

    _orig_lib_paths = cpp_ext.library_paths
    def _patched_lib_paths(cuda=False):
        paths = _orig_lib_paths(cuda)
        return [p.replace(torch_pkg_dir, safe_torch_pkg) if torch_pkg_dir in p else p for p in paths]
    cpp_ext.library_paths = _patched_lib_paths

print("\n=== Compilando extension PyTorch bvh_router_ext ===")
ext = cpp_ext.load(
    name="bvh_router_ext",
    sources=[str(safe_source)],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--maxrregcount=64",
        "-DNDEBUG",
    ] + cuda_arch_flags,
    extra_ldflags=[f"-L{safe_torch_lib}"] if needs_patch else [],
    build_directory=build_dir,
    verbose=True,
)

# Restore torch paths
if needs_patch:
    torch.__file__ = _original_torch_file
    if _original_torch_path:
        torch.__path__[:] = _original_torch_path

print("\n=== Extension compilada exitosamente ===")
print(f"Modulo: {ext}")
print(f"Funciones: upload_tree, route, route_sync")

# Test rapido
print("\n=== Test rapido ===")
import numpy as np

BVH_NODES = 85
SPEC_DIM = 64
BATCH = 32

# Crear arbol aleatorio (CPU tensors)
centers = torch.randn(BVH_NODES, 3, dtype=torch.float32)
radii = torch.ones(BVH_NODES, dtype=torch.float32) * 0.5
portals = torch.zeros(BVH_NODES, 12, dtype=torch.float32)
for i in range(BVH_NODES):
    portals[i, 0] = 1.0  # Identidad 3x4
    portals[i, 5] = 1.0
    portals[i, 10] = 1.0
snell_w = torch.randn(BVH_NODES, SPEC_DIM, dtype=torch.float32) * 0.1
snell_b = torch.randn(BVH_NODES, dtype=torch.float32) * 0.1

ext.upload_tree(centers, radii, portals, snell_w, snell_b)
print("upload_tree: OK")

# Crear rayos (CUDA tensors -- zero copy!)
origins = torch.randn(BATCH, 3, dtype=torch.float32, device="cuda")
directions = torch.randn(BATCH, 3, dtype=torch.float32, device="cuda")
spectral = torch.randn(BATCH, SPEC_DIM, dtype=torch.float32, device="cuda")

expert_ids, scores, confidence = ext.route_sync(origins, directions, spectral)
print(f"route_sync: OK")
print(f"  expert_ids shape: {expert_ids.shape}, device: {expert_ids.device}")
print(f"  scores shape: {scores.shape}, device: {scores.device}")
print(f"  confidence shape: {confidence.shape}, device: {confidence.device}")
print(f"  expert_ids[:5]: {expert_ids[:5].tolist()}")
print(f"  confidence[:5]: {confidence[:5].tolist()}")

# Benchmark rapido
import time
torch.cuda.synchronize()
N_ITER = 1000
start = time.perf_counter()
for _ in range(N_ITER):
    ext.route(origins, directions, spectral)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"\n=== Benchmark: {N_ITER} iteraciones, batch={BATCH} ===")
print(f"  Total: {elapsed*1000:.1f} ms")
print(f"  Por iteracion: {elapsed/N_ITER*1e6:.1f} us")
print(f"  Throughput: {BATCH*N_ITER/elapsed:.0f} samples/s")
