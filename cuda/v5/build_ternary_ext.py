"""
build_ternary_ext.py -- Compile the PyTorch Ternary Expert extension
LiquidBit v5.0

Usage from WSL2:
    cd /tmp/liquidbit
    source .venv_wsl/bin/activate
    python cuda/v5/build_ternary_ext.py

The extension installs as importable module 'ternary_expert_ext'.
"""

import os
import shutil
from pathlib import Path

# Ensure project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)

import torch
import torch.utils.cpp_extension as cpp_ext

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")

# Detect CUDA architecture
cuda_arch_flags = []
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    arch = f"{cap[0]}{cap[1]}"
    cuda_arch_flags = [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    print(f"Compiling for sm_{arch}")

# ============================================================================
# Workaround for spaces in paths (ninja linker can't handle them)
# Strategy: symlink torch/lib to a safe path, then monkey-patch torch's
# internal _get_torch_lib_path so the generated ninja file uses our symlink.
# ============================================================================

build_dir = os.path.expanduser("~/.cache/torch_extensions/ternary_expert_ext")
os.makedirs(build_dir, exist_ok=True)

# Copy .cu source to a path without spaces
cu_source = project_root / "cuda" / "v5" / "ternary_torch_ext.cu"
safe_source = Path(build_dir) / "ternary_torch_ext.cu"
shutil.copy2(cu_source, safe_source)

torch_lib_dir = str(Path(torch.__file__).parent / "lib")
needs_patch = " " in torch_lib_dir

if needs_patch:
    safe_torch_lib = "/tmp/_torch_lib_ternary"
    if os.path.islink(safe_torch_lib):
        os.unlink(safe_torch_lib)
    os.symlink(torch_lib_dir, safe_torch_lib)
    print(f"Torch lib has spaces — symlinked to: {safe_torch_lib}")

    # Monkey-patch torch's internal library path resolver
    _orig_prepare = cpp_ext._prepare_ldflags if hasattr(cpp_ext, '_prepare_ldflags') else None

    # Patch the TORCH_LIB_PATH that gets baked into the ninja file
    torch_lib_escaped = torch_lib_dir.replace(" ", "\\ ")

    # Also patch _join_cuda_home and library_paths
    _orig_lib_paths = cpp_ext.library_paths
    def _patched_lib_paths(cuda=False):
        paths = _orig_lib_paths(cuda)
        return [p.replace(torch_lib_dir, safe_torch_lib) if torch_lib_dir in p else p for p in paths]
    cpp_ext.library_paths = _patched_lib_paths

    # Patch TORCH_LIB_PATH in the module namespace
    if hasattr(cpp_ext, 'TORCH_LIB_PATH'):
        cpp_ext.TORCH_LIB_PATH = safe_torch_lib

print(f"\n=== Compiling ternary_expert_ext ===")
ext = cpp_ext.load(
    name="ternary_expert_ext",
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

print("\n=== Extension compiled successfully ===")
print(f"Module: {ext}")
print(f"Functions: pack_ternary, ternary_linear, ternary_gated_mlp")

# ============================================================================
# Quick test
# ============================================================================
print("\n=== Quick test ===")

# Test pack_ternary
IN_DIM = 1024
INTER_DIM = 2048
OUT_DIM = 1024
BATCH = 4

# Create random ternary weights (PyTorch layout: [out_features, in_features])
import numpy as np
rng = np.random.default_rng(42)

# PyTorch weight shapes
gate_pt = torch.from_numpy(
    rng.choice([-1, 0, 1], size=(INTER_DIM, IN_DIM)).astype(np.int8)
)
up_pt = torch.from_numpy(
    rng.choice([-1, 0, 1], size=(INTER_DIM, IN_DIM)).astype(np.int8)
)
down_pt = torch.from_numpy(
    rng.choice([-1, 0, 1], size=(OUT_DIM, INTER_DIM)).astype(np.int8)
)

# CRITICAL: kernel expects packed layout [ceil(in_features/16), out_features]
# PyTorch weight is [out, in], so we TRANSPOSE before packing
gate_packed = ext.pack_ternary(gate_pt.T.contiguous())  # [IN, INTER] -> packed [ceil(IN/16), INTER]
up_packed = ext.pack_ternary(up_pt.T.contiguous())
down_packed = ext.pack_ternary(down_pt.T.contiguous())  # [INTER, OUT] -> packed [ceil(INTER/16), OUT]
print(f"pack_ternary: gate {gate_pt.shape} -> T -> packed {gate_packed.shape}")
print(f"pack_ternary: up   {up_pt.shape} -> T -> packed {up_packed.shape}")
print(f"pack_ternary: down {down_pt.shape} -> T -> packed {down_packed.shape}")

# Move packed weights to GPU
gate_packed = gate_packed.cuda()
up_packed = up_packed.cuda()
down_packed = down_packed.cuda()

gate_scale = torch.ones(INTER_DIM, device="cuda")
up_scale = torch.ones(INTER_DIM, device="cuda")
down_scale = torch.ones(OUT_DIM, device="cuda")

# Test ternary_linear
x = torch.randn(BATCH, IN_DIM, device="cuda")
out = ext.ternary_linear(x, gate_packed, gate_scale)
print(f"\nternary_linear: ({BATCH}, {IN_DIM}) -> {out.shape}")
print(f"  output[:2, :5]: {out[:2, :5].tolist()}")

# Test ternary_gated_mlp
out_mlp = ext.ternary_gated_mlp(
    x, gate_packed, up_packed, down_packed,
    gate_scale, up_scale, down_scale
)
print(f"\nternary_gated_mlp: ({BATCH}, {IN_DIM}) -> {out_mlp.shape}")
print(f"  output[:2, :5]: {out_mlp[:2, :5].tolist()}")

# Verify correctness against PyTorch F.linear
print("\n=== Correctness check vs PyTorch F.linear ===")
x_test = torch.randn(1, IN_DIM, device="cuda")

# PyTorch reference: F.linear(x, w) = x @ w.T
ref = torch.nn.functional.linear(x_test, gate_pt.float().cuda())  # [1, INTER]

# Our kernel (packed weights already transposed)
our = ext.ternary_linear(x_test, gate_packed, torch.ones(INTER_DIM, device="cuda"))

diff = (ref - our).abs().max().item()
print(f"  Max absolute diff: {diff:.6f}")
print(f"  Match: {'YES' if diff < 0.01 else 'NO'}")

# Benchmark
import time
print(f"\n=== Benchmark: ternary_gated_mlp ({IN_DIM}->{INTER_DIM}->{OUT_DIM}) ===")
x_bench = torch.randn(BATCH, IN_DIM, device="cuda")

N_ITER = 1000
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(N_ITER):
    ext.ternary_gated_mlp(
        x_bench, gate_packed, up_packed, down_packed,
        gate_scale, up_scale, down_scale
    )
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"  {N_ITER} iterations, batch={BATCH}")
print(f"  Total: {elapsed*1000:.1f} ms")
print(f"  Per iteration: {elapsed/N_ITER*1e6:.1f} us")
print(f"  Throughput: {BATCH*N_ITER/elapsed:.0f} samples/s")

# Compare vs F.linear (PyTorch)
print(f"\n=== Benchmark: PyTorch F.linear (same dims) ===")
gate_f = gate_pt.float().cuda()
up_f = up_pt.float().cuda()
down_f = down_pt.float().cuda()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(N_ITER):
    g = torch.nn.functional.linear(x_bench, gate_f)
    u = torch.nn.functional.linear(x_bench, up_f)
    h = torch.nn.functional.silu(g) * u
    _ = torch.nn.functional.linear(h, down_f)
torch.cuda.synchronize()
elapsed_pt = time.perf_counter() - start

print(f"  Per iteration: {elapsed_pt/N_ITER*1e6:.1f} us")
print(f"  Speedup: {elapsed_pt/elapsed:.1f}x")
