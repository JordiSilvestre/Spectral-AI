#!/usr/bin/env python3
"""Build CUDA extensions (.so) for WSL/Linux via PyTorch JIT."""
import os
import sys

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_HOME"] = "/usr/local/cuda"

from torch.utils.cpp_extension import load

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda_v5 = os.path.join(base, "cuda", "v5")

print("=" * 60)
print("Building CUDA extensions for WSL/Linux")
print("=" * 60)

# 1. Ternary expert
print("\n[1/2] Compiling ternary_expert_ext...")
ternary = load(
    name="ternary_expert_ext",
    sources=[os.path.join(cuda_v5, "ternary_torch_ext.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)
print(f"  OK: {ternary}")

# 2. BVH router
print("\n[2/2] Compiling bvh_router_ext...")
bvh = load(
    name="bvh_router_ext",
    sources=[os.path.join(cuda_v5, "bvh_torch_ext.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)
print(f"  OK: {bvh}")

print("\n" + "=" * 60)
print("Both extensions compiled successfully!")
print("=" * 60)
