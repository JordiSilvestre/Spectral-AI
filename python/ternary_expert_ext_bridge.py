#!/usr/bin/env python3
"""
ternary_expert_ext_bridge.py — Zero-Copy Ternary Expert using CUDA Extension

Drop-in replacement for TernaryExpertModule that uses the ternary_expert_ext
PyTorch extension (POPCOUNT ternary matmul, no FP multiplications).

Usage:
    from ternary_expert_ext_bridge import CUDATernaryExpertModule, HAS_TERNARY_EXT

    if HAS_TERNARY_EXT:
        expert = CUDATernaryExpertModule(ternary_expert_data, output_proj)
    else:
        expert = TernaryExpertModule(ternary_expert_data, output_proj)  # fallback

Copyright (c) 2026 SpectralAI Studio — Apache 2.0
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Try to import the CUDA extension
# JIT-compiled extensions live in ~/.cache/torch_extensions/ (Linux)
# or cuda/v5/ternary_expert_ext.pyd (Windows split build)
HAS_TERNARY_EXT = False

def _setup_dll_dirs():
    """Add CUDA and Torch DLL directories on Windows so .pyd can find them."""
    import os as _os, sys as _sys
    if _sys.platform == "win32" and hasattr(_os, "add_dll_directory"):
        cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
        if _os.path.isdir(cuda_bin):
            try:
                _os.add_dll_directory(cuda_bin)
            except OSError:
                pass
        try:
            import torch as _torch
            torch_lib = str(Path(_torch.__file__).parent / "lib")
            if _os.path.isdir(torch_lib):
                _os.add_dll_directory(torch_lib)
        except Exception:
            pass

_setup_dll_dirs()

_search_dirs = [
    # Windows split build location
    str(Path(__file__).resolve().parent.parent / "cuda" / "v5"),
    # Linux JIT build location
    str(Path("~/.cache/torch_extensions/ternary_expert_ext").expanduser()),
    # Windows JIT build location
    str(Path("~/AppData/Local/torch_extensions/ternary_expert_ext").expanduser()),
]

try:
    import ternary_expert_ext
    HAS_TERNARY_EXT = True
except ImportError:
    import sys as _sys
    for _ext_dir in _search_dirs:
        if Path(_ext_dir).is_dir() and _ext_dir not in _sys.path:
            _sys.path.insert(0, _ext_dir)
            try:
                import ternary_expert_ext
                HAS_TERNARY_EXT = True
                break
            except ImportError:
                _sys.path.remove(_ext_dir)


class CUDATernaryExpertModule(nn.Module):
    """
    Ternary expert using CUDA POPCOUNT extension.

    Replaces F.linear() with ternary_expert_ext.ternary_gated_mlp():
    - Weights packed as 2-bit (16 per uint32) — 16x compression vs FP32
    - Forward uses only add/sub/skip — ZERO multiplications
    - All data stays on GPU — zero-copy between PyTorch and kernel

    Expected input: TernaryExpert dataclass with:
        gate_ternary: np.ndarray [intermediate, hidden] int values {-1,0,+1}
        up_ternary:   np.ndarray [intermediate, hidden]
        down_ternary: np.ndarray [hidden, intermediate]
        gate_scale:   np.ndarray [intermediate] float32
        up_scale:     np.ndarray [intermediate] float32
        down_scale:   np.ndarray [hidden] float32
    """

    def __init__(self, expert, output_proj: Optional[nn.Linear] = None):
        super().__init__()
        if not HAS_TERNARY_EXT:
            raise RuntimeError(
                "ternary_expert_ext not available. "
                "Compile with: python cuda/v5/build_ternary_ext.py"
            )

        self.expert_id = expert.expert_id

        # Pack ternary weights to 2-bit encoding (CPU, then move to GPU)
        # IMPORTANT: kernel expects packed layout [ceil(in_features/16), out_features]
        # PyTorch convention: weight is [out_features, in_features]
        # So we must transpose before packing to align input groups with packed rows
        gate_int8 = torch.from_numpy(expert.gate_ternary.T.copy().astype(np.int8))
        up_int8 = torch.from_numpy(expert.up_ternary.T.copy().astype(np.int8))
        down_int8 = torch.from_numpy(expert.down_ternary.T.copy().astype(np.int8))

        # Pack: [rows, cols] int8 -> [ceil(rows/16), cols] int32
        gate_packed = ternary_expert_ext.pack_ternary(gate_int8)
        up_packed = ternary_expert_ext.pack_ternary(up_int8)
        down_packed = ternary_expert_ext.pack_ternary(down_int8)

        # Register as buffers (move to GPU with .cuda())
        self.register_buffer('gate_packed', gate_packed)
        self.register_buffer('up_packed', up_packed)
        self.register_buffer('down_packed', down_packed)

        # Scales as float32
        self.register_buffer('gate_s', torch.from_numpy(expert.gate_scale.copy()))
        self.register_buffer('up_s', torch.from_numpy(expert.up_scale.copy()))
        self.register_buffer('down_s', torch.from_numpy(expert.down_scale.copy()))

        # Shared output projection
        self.output_proj = output_proj

        # Store dimensions for reporting (original PyTorch layout: [out, in])
        self._in_dim = expert.gate_ternary.shape[1]     # hidden (input to expert)
        self._inter_dim = expert.gate_ternary.shape[0]   # intermediate
        self._out_dim = expert.down_ternary.shape[0]     # hidden (output of expert)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, hidden) — hidden features from backbone
        Returns: (B, S, out_dim) — expert hidden output BEFORE output_proj
        """
        B, S, H = x.shape
        # Flatten batch*seq for the kernel
        x_flat = x.reshape(B * S, H).float().contiguous()

        # CUDA POPCOUNT ternary gated MLP — ZERO multiplications!
        out_flat = ternary_expert_ext.ternary_gated_mlp(
            x_flat,
            self.gate_packed,
            self.up_packed,
            self.down_packed,
            self.gate_s,
            self.up_s,
            self.down_s,
        )

        # Reshape back to (B, S, out_dim)
        return out_flat.reshape(B, S, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, hidden) — hidden features from backbone
        Returns: (B, S, output_dim) via shared output_proj
        """
        out = self.forward_hidden(x)
        if self.output_proj is not None:
            return self.output_proj(out)
        return out

    def memory_bytes(self) -> int:
        """Total memory in bytes (packed weights = 16x smaller than FP32)."""
        packed_bytes = (
            self.gate_packed.numel() + self.up_packed.numel() + self.down_packed.numel()
        ) * 4  # int32 = 4 bytes
        scale_bytes = (
            self.gate_s.numel() + self.up_s.numel() + self.down_s.numel()
        ) * 4  # float32
        return packed_bytes + scale_bytes


def create_expert_module(expert, output_proj=None, prefer_cuda=True,
                         fallback_class=None):
    """
    Factory: create the best available expert module.

    Returns CUDATernaryExpertModule if extension available, else falls back
    to fallback_class (must be passed by caller to avoid circular imports).
    """
    if prefer_cuda and HAS_TERNARY_EXT:
        return CUDATernaryExpertModule(expert, output_proj)

    if fallback_class is not None:
        return fallback_class(expert, output_proj)

    raise RuntimeError(
        "ternary_expert_ext not available and no fallback_class provided. "
        "Either compile the extension (python cuda/v5/build_ternary_ext.py) "
        "or pass fallback_class=TernaryExpertModule."
    )


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CUDATernaryExpertModule — Test")
    print("=" * 60)

    if not HAS_TERNARY_EXT:
        print("ERROR: ternary_expert_ext not found!")
        print("Compile with: python cuda/v5/build_ternary_ext.py")
        exit(1)

    print(f"ternary_expert_ext loaded: {ternary_expert_ext}")

    # Create fake expert data
    from dataclasses import dataclass

    @dataclass
    class FakeExpert:
        expert_id: int
        source_layer: int
        gate_ternary: np.ndarray
        up_ternary: np.ndarray
        down_ternary: np.ndarray
        gate_scale: np.ndarray
        up_scale: np.ndarray
        down_scale: np.ndarray
        sparsity: float
        size_bytes: int

    IN_DIM = 1024
    INTER = 2048
    rng = np.random.default_rng(42)

    expert_data = FakeExpert(
        expert_id=0,
        source_layer=0,
        gate_ternary=rng.choice([-1, 0, 1], size=(INTER, IN_DIM)).astype(np.float32),
        up_ternary=rng.choice([-1, 0, 1], size=(INTER, IN_DIM)).astype(np.float32),
        down_ternary=rng.choice([-1, 0, 1], size=(IN_DIM, INTER)).astype(np.float32),
        gate_scale=np.ones(INTER, dtype=np.float32) * 0.1,
        up_scale=np.ones(INTER, dtype=np.float32) * 0.1,
        down_scale=np.ones(IN_DIM, dtype=np.float32) * 0.01,
        sparsity=0.33,
        size_bytes=0,
    )

    # No output proj for testing
    module = CUDATernaryExpertModule(expert_data, output_proj=None)
    module = module.cuda()

    x = torch.randn(2, 8, IN_DIM, device="cuda")  # (B=2, S=8, H=1024)
    out = module(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Memory: {module.memory_bytes() / 1024:.1f} KB")
    print(f"out[0, 0, :5]: {out[0, 0, :5].tolist()}")

    # Benchmark
    import time
    N = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        module(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1e6
    print(f"\nBenchmark: {dt:.1f} us/forward (B=2, S=8, H={IN_DIM})")
    print(f"           {dt/16:.1f} us/token")
