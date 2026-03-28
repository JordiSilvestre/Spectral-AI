#!/bin/bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

cd "/mnt/j/Proyectos/LiquidBit Zero-Matrix"

echo "=== Symbols in libbvh_router.so ==="
nm -D cuda/v5/libbvh_router.so | grep -E "bvh_router|ray_batch|router_output|bvh_tree" | head -20

echo ""
echo "=== Test ctypes minimal ==="
python3 - << 'EOF'
import ctypes, sys, os
sys.path.insert(0, "python")

# Ensure CUDA is initialized by PyTorch first
import torch
print(f"PyTorch CUDA: {torch.cuda.is_available()}")
x = torch.zeros(1, device='cuda')  # Force CUDA init
torch.cuda.synchronize()
print(f"PyTorch CUDA initialized, device: {torch.cuda.current_device()}")

# Load library
lib = ctypes.CDLL("cuda/v5/libbvh_router.so")
print("Library loaded OK")

# Set argtypes explicitly
class RayBatchDevice(ctypes.Structure):
    _fields_ = [
        ("origins",    ctypes.c_void_p),
        ("directions", ctypes.c_void_p),
        ("spectral",   ctypes.c_void_p),
        ("batch_size", ctypes.c_int),
    ]

class RouterOutput(ctypes.Structure):
    _fields_ = [
        ("selected_expert", ctypes.c_void_p),
        ("routing_scores",  ctypes.c_void_p),
        ("traversal_path",  ctypes.c_void_p),
        ("confidence",      ctypes.c_void_p),
    ]

lib.ray_batch_alloc.restype = ctypes.c_int
lib.ray_batch_alloc.argtypes = [ctypes.POINTER(RayBatchDevice), ctypes.c_int]

lib.router_output_alloc.restype = ctypes.c_int
lib.router_output_alloc.argtypes = [ctypes.POINTER(RouterOutput), ctypes.c_int]

lib.bvh_router_launch_sync.restype = ctypes.c_int
lib.bvh_router_launch_sync.argtypes = [
    ctypes.POINTER(RayBatchDevice),
    ctypes.POINTER(RouterOutput),
    ctypes.c_int
]

BATCH = 256
rays = RayBatchDevice()
out  = RouterOutput()

err = lib.ray_batch_alloc(ctypes.byref(rays), ctypes.c_int(BATCH))
print(f"ray_batch_alloc: {err}")

err = lib.router_output_alloc(ctypes.byref(out), ctypes.c_int(BATCH))
print(f"router_output_alloc: {err}")

# Upload BVH tree (random data)
import numpy as np

BVH_NODES = 85
SPEC_DIM = 64
centers = np.random.randn(BVH_NODES, 3).astype(np.float32)
radii   = np.ones(BVH_NODES, dtype=np.float32) * 0.5
portals = np.zeros((BVH_NODES, 3, 4), dtype=np.float32)
for i in range(BVH_NODES):
    for j in range(3):
        portals[i, j, j] = 1.0
snell_w = np.random.randn(BVH_NODES, SPEC_DIM).astype(np.float32) * 0.1
snell_b = np.random.randn(BVH_NODES).astype(np.float32) * 0.1

lib.bvh_tree_upload_const.restype = ctypes.c_int
lib.bvh_tree_upload_const.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

err = lib.bvh_tree_upload_const(
    centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    portals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    snell_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    snell_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
)
print(f"bvh_tree_upload_const: {err}")

# Upload rays using cudaMemcpy
cuda_rt = ctypes.CDLL("libcudart.so.13")
cuda_rt.cudaMemcpy.restype = ctypes.c_int
cuda_rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

origins_np  = np.random.randn(BATCH, 3).astype(np.float32)
dirs_np     = np.random.randn(BATCH, 3).astype(np.float32)
spec_np     = np.random.randn(BATCH, SPEC_DIM).astype(np.float32)

err = cuda_rt.cudaMemcpy(rays.origins, origins_np.ctypes.data_as(ctypes.c_void_p), BATCH*3*4, 1)
print(f"cudaMemcpy origins: {err}")
err = cuda_rt.cudaMemcpy(rays.directions, dirs_np.ctypes.data_as(ctypes.c_void_p), BATCH*3*4, 1)
print(f"cudaMemcpy dirs: {err}")
err = cuda_rt.cudaMemcpy(rays.spectral, spec_np.ctypes.data_as(ctypes.c_void_p), BATCH*SPEC_DIM*4, 1)
print(f"cudaMemcpy spectral: {err}")

# Launch kernel
print(f"Launching with batch={BATCH}...")
err = lib.bvh_router_launch_sync(ctypes.byref(rays), ctypes.byref(out), ctypes.c_int(BATCH))
print(f"bvh_router_launch_sync: {err}")
if err == 0:
    print("SUCCESS!")
else:
    print(f"FAILED with code {err}")
EOF
