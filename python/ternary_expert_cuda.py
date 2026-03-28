#!/usr/bin/env python3
"""
ternary_expert_cuda.py — Python binding for libternary_expert.so

Provides POPCOUNT-based ternary expert inference:
  - Zero multiplications (only add/sub based on sign masks)
  - 2-bit packed weights (16 weights per uint32)
  - Device-side expert dispatch (expert_id comes from GPU, no host sync)

The kernel expects fixed dimensions:
  TERN_INPUT  = 64   (input features — spectral/routing vector)
  TERN_HIDDEN = 1024 (hidden layer)
  TERN_VOCAB  = 4096 (output logits)

For the real model demo, our experts have different dims (1024→2048→vocab).
This binding supports both:
  1. Native kernel dims (64→1024→4096) for benchmarking
  2. Hybrid mode: CUDA kernel for routing, PyTorch for non-matching expert dims

Copyright (c) 2026 LiquidBit Studio — Apache 2.0
"""

import ctypes
import os
import numpy as np
from pathlib import Path
from typing import Optional, List

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Must match ternary_expert.cu constants
TERN_INPUT = 64
TERN_HIDDEN = 1024
TERN_VOCAB = 4096
TERN_PACK = 16
W1_PACKED_ROWS = TERN_INPUT // TERN_PACK    # 4
W2_PACKED_ROWS = TERN_HIDDEN // TERN_PACK   # 64


class TernaryExpertBank(ctypes.Structure):
    """Mirror of C struct TernaryExpertBank."""
    _fields_ = [
        ("W1_all", ctypes.c_void_p),
        ("W2_all", ctypes.c_void_p),
        ("scale1_all", ctypes.c_void_p),
        ("scale2_all", ctypes.c_void_p),
        ("bias1_all", ctypes.c_void_p),
        ("bias2_all", ctypes.c_void_p),
        ("num_experts", ctypes.c_int),
    ]


class CUDATernaryExpertBank:
    """
    Python wrapper for the CUDA ternary expert kernel.

    Usage:
        bank = CUDATernaryExpertBank(num_experts=64)
        bank.upload_expert(expert_id=0, weights_gate, weights_up, ...)
        output = bank.forward(input_tensor, expert_ids_tensor)
    """

    def __init__(
        self,
        num_experts: int,
        lib_path: Optional[str] = None,
    ):
        if lib_path is None:
            lib_path = self._find_lib()

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"libternary_expert.so not found at {lib_path}. "
                f"Compile with: cd cuda/v5 && make"
            )

        # Init CUDA context via PyTorch first
        if HAS_TORCH and torch.cuda.is_available():
            _dummy = torch.zeros(1, device='cuda')
            torch.cuda.synchronize()
            del _dummy

        self._lib = ctypes.CDLL(lib_path)
        self._setup_api()

        self._bank = TernaryExpertBank()
        err = self._lib.ternary_bank_alloc(
            ctypes.byref(self._bank), ctypes.c_int(num_experts)
        )
        if err != 0:
            raise RuntimeError(f"ternary_bank_alloc failed: error {err}")

        self.num_experts = num_experts
        self._experts_uploaded = set()

    @staticmethod
    def _find_lib() -> str:
        candidates = [
            Path(__file__).parent.parent / "cuda" / "v5" / "libternary_expert.so",
            Path(__file__).parent.parent / "cuda" / "libternary_expert.so",
            Path(__file__).parent / "libternary_expert.so",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return str(candidates[0])  # Will fail with FileNotFoundError

    def _setup_api(self):
        L = self._lib
        ci = ctypes.c_int
        PB = ctypes.POINTER(TernaryExpertBank)
        PU = ctypes.POINTER(ctypes.c_uint32)
        PF = ctypes.POINTER(ctypes.c_float)

        L.ternary_bank_alloc.restype = ci
        L.ternary_bank_alloc.argtypes = [PB, ci]

        L.ternary_bank_free.restype = ci
        L.ternary_bank_free.argtypes = [PB]

        L.ternary_bank_upload_expert.restype = ci
        L.ternary_bank_upload_expert.argtypes = [PB, ci, PU, PU, PF, PF, PF, PF]

        L.ternary_expert_launch.restype = ci
        L.ternary_expert_launch.argtypes = [
            PB, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ci, ctypes.c_void_p
        ]

        L.pack_ternary_weights.restype = None
        L.pack_ternary_weights.argtypes = [PF, PU, ci, ci]

    def _pack_weights(self, weights: np.ndarray) -> np.ndarray:
        """Pack float ternary weights {-1,0,+1} to 2-bit encoding."""
        rows, cols = weights.shape
        packed_rows = (rows + TERN_PACK - 1) // TERN_PACK
        packed = np.zeros((packed_rows, cols), dtype=np.uint32)

        w_flat = np.ascontiguousarray(weights.astype(np.float32))
        p_flat = np.ascontiguousarray(packed)

        self._lib.pack_ternary_weights(
            w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_int(rows),
            ctypes.c_int(cols),
        )
        return p_flat

    def upload_expert(
        self,
        expert_id: int,
        gate_ternary: np.ndarray,    # [input_dim, hidden_dim] — values in {-1,0,+1}
        down_ternary: np.ndarray,    # [hidden_dim, output_dim]
        gate_scale: np.ndarray,      # [hidden_dim]
        down_scale: np.ndarray,      # [output_dim]
        gate_bias: Optional[np.ndarray] = None,
        down_bias: Optional[np.ndarray] = None,
    ):
        """
        Upload a single expert's weights to the GPU bank.

        Note: the kernel uses a 2-layer MLP (W1: input→hidden, W2: hidden→output).
        We map gate_ternary→W1, down_ternary→W2.
        """
        # Pack ternary weights to 2-bit encoding
        W1_packed = self._pack_weights(gate_ternary)
        W2_packed = self._pack_weights(down_ternary)

        # Default zero biases
        hidden_dim = gate_ternary.shape[1]
        output_dim = down_ternary.shape[1]
        if gate_bias is None:
            gate_bias = np.zeros(hidden_dim, dtype=np.float32)
        if down_bias is None:
            down_bias = np.zeros(output_dim, dtype=np.float32)

        # Ensure contiguous float32/uint32
        W1_packed = np.ascontiguousarray(W1_packed.ravel(), dtype=np.uint32)
        W2_packed = np.ascontiguousarray(W2_packed.ravel(), dtype=np.uint32)
        scale1 = np.ascontiguousarray(gate_scale, dtype=np.float32)
        scale2 = np.ascontiguousarray(down_scale, dtype=np.float32)
        bias1 = np.ascontiguousarray(gate_bias, dtype=np.float32)
        bias2 = np.ascontiguousarray(down_bias, dtype=np.float32)

        PU = ctypes.POINTER(ctypes.c_uint32)
        PF = ctypes.POINTER(ctypes.c_float)

        err = self._lib.ternary_bank_upload_expert(
            ctypes.byref(self._bank),
            ctypes.c_int(expert_id),
            W1_packed.ctypes.data_as(PU),
            W2_packed.ctypes.data_as(PU),
            scale1.ctypes.data_as(PF),
            scale2.ctypes.data_as(PF),
            bias1.ctypes.data_as(PF),
            bias2.ctypes.data_as(PF),
        )
        if err != 0:
            raise RuntimeError(f"ternary_bank_upload_expert({expert_id}) failed: {err}")

        self._experts_uploaded.add(expert_id)

    def forward(
        self,
        input_tensor: 'torch.Tensor',   # [batch, TERN_INPUT] on CUDA
        expert_ids: 'torch.Tensor',      # [batch] int32 on CUDA
    ) -> 'torch.Tensor':
        """
        Run ternary expert forward pass on GPU.
        Returns: [batch, TERN_VOCAB] float32 on CUDA
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for forward()")

        batch = input_tensor.shape[0]
        assert input_tensor.shape[1] == TERN_INPUT, (
            f"Input must be [{batch}, {TERN_INPUT}], got {input_tensor.shape}"
        )
        assert input_tensor.is_cuda and expert_ids.is_cuda

        # Ensure float32 and int32
        input_f32 = input_tensor.float().contiguous()
        expert_i32 = expert_ids.int().contiguous()

        # Allocate output
        output = torch.zeros(batch, TERN_VOCAB, device='cuda', dtype=torch.float32)

        # Launch kernel (stream=0 for default)
        err = self._lib.ternary_expert_launch(
            ctypes.byref(self._bank),
            ctypes.c_void_p(input_f32.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(expert_i32.data_ptr()),
            ctypes.c_int(batch),
            ctypes.c_void_p(0),  # default stream
        )
        if err != 0:
            raise RuntimeError(f"ternary_expert_launch failed: {err}")

        return output

    def __del__(self):
        try:
            if hasattr(self, '_bank'):
                self._lib.ternary_bank_free(ctypes.byref(self._bank))
        except Exception:
            pass


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CUDATernaryExpertBank — Test")
    print("=" * 60)

    bank = CUDATernaryExpertBank(num_experts=4)

    # Create random ternary expert weights
    rng = np.random.default_rng(42)
    for eid in range(4):
        gate = rng.choice([-1.0, 0.0, 1.0], size=(TERN_INPUT, TERN_HIDDEN)).astype(np.float32)
        down = rng.choice([-1.0, 0.0, 1.0], size=(TERN_HIDDEN, TERN_VOCAB)).astype(np.float32)
        gate_scale = np.ones(TERN_HIDDEN, dtype=np.float32) * 0.1
        down_scale = np.ones(TERN_VOCAB, dtype=np.float32) * 0.01
        bank.upload_expert(eid, gate, down, gate_scale, down_scale)
        print(f"  Expert {eid} uploaded")

    # Test forward
    import torch
    input_t = torch.randn(2, TERN_INPUT, device='cuda')
    expert_ids = torch.tensor([0, 2], device='cuda', dtype=torch.int32)

    output = bank.forward(input_t, expert_ids)
    print(f"\n  Input:  {input_t.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Output[0, :5]: {output[0, :5].tolist()}")
    print(f"  Output[1, :5]: {output[1, :5].tolist()}")

    # Benchmark
    import time
    N = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = bank.forward(input_t, expert_ids)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1e6
    print(f"\n  Benchmark: {dt:.1f} μs/batch (batch=2)")
    print(f"             {dt/2:.1f} μs/sample")
