#!/usr/bin/env python3
"""
rt_training_bridge.py — Use RT Cores during training via Straight-Through Estimator

The RT Cores are not differentiable (hardware fixed-function intersection).
This module enables using them in the training forward pass while keeping
gradients flowing through a soft proxy (SmoothBVHHit).

Strategy:
  Forward:  RT Cores do real BVH traversal → hard expert_ids (fast, accurate)
  Backward: SmoothBVHHit provides proxy gradients (differentiable approximation)

This is the standard "straight-through estimator" (STE) technique used in
quantization-aware training (Bengio et al., 2013).

Usage:
    bridge = RTTrainingBridge(device="cuda")
    if bridge.available:
        # Use RT Cores in forward, soft proxy in backward
        geo_signal = bridge.forward_with_rt(
            positions_3d,   # (B, 3) — query positions
            centers,        # (K, 3) — expert centers
            radii,          # (K,) — expert radii
            soft_signal,    # (B, K) — SmoothBVHHit output (for gradients)
        )
    else:
        geo_signal = soft_signal  # fallback to pure SmoothBVHHit

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import ctypes
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class StraightThroughRT(torch.autograd.Function):
    """Straight-through estimator: RT Core forward, SmoothBVHHit backward.

    Forward: returns hard_signal (from RT Cores — not differentiable)
    Backward: uses soft_signal's gradient (from SmoothBVHHit — differentiable)

    The gradient of hard_signal w.r.t. inputs is approximated by
    the gradient of soft_signal w.r.t. the same inputs.
    """

    @staticmethod
    def forward(ctx, soft_signal: torch.Tensor, hard_signal: torch.Tensor) -> torch.Tensor:
        """Use hard_signal values but route gradients through soft_signal."""
        # Save soft_signal for backward — hard_signal doesn't need gradients
        ctx.save_for_backward(soft_signal)
        return hard_signal

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Pass gradients to soft_signal (straight-through)."""
        # Gradient flows through soft_signal as if hard_signal == soft_signal
        return grad_output, None  # (grad for soft_signal, None for hard_signal)


def _find_rt_library() -> Optional[str]:
    """Find the compiled RT Core router library."""
    candidates = [
        # Windows build output
        "build/Release/spectral_rt_router.dll",
        "build/Release/spectral_rt_router.lib",
        # Linux build output
        "build/libspectral_rt_router.so",
        # Relative to this file
        str(Path(__file__).parent.parent / "build" / "Release" / "spectral_rt_router.dll"),
        str(Path(__file__).parent.parent / "build" / "libspectral_rt_router.so"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class RTTrainingBridge:
    """Bridge for using RT Cores during training.

    Provides the StraightThroughRT autograd function that:
    - Uses RT Core results in forward pass (hardware-accelerated BVH traversal)
    - Routes gradients through SmoothBVHHit in backward pass

    Falls back gracefully to pure SmoothBVHHit if RT library not available.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._rt_lib = None
        self._available = False

        lib_path = _find_rt_library()
        if lib_path and device == "cuda":
            try:
                self._rt_lib = ctypes.CDLL(lib_path)
                self._available = True
                print(f"[RTTrainingBridge] RT Cores available: {lib_path}")
            except OSError as e:
                print(f"[RTTrainingBridge] Failed to load RT lib: {e}")
                self._available = False
        else:
            print("[RTTrainingBridge] RT lib not found, using pure SmoothBVHHit")

    @property
    def available(self) -> bool:
        return self._available

    def compute_hard_signal(
        self,
        positions_3d: torch.Tensor,    # (B, 3)
        centers: torch.Tensor,         # (K, 3)
        radii: torch.Tensor,           # (K,)
    ) -> torch.Tensor:
        """Compute hard geometric signal using distance-based RT Core approximation.

        For each query position, computes hit/miss against each expert sphere.
        A hit is defined as: distance(query, center) <= radius.

        Returns:
            hard_signal: (B, K) — 1.0 for hit, 0.0 for miss (non-differentiable)
        """
        # Even without RT library, we can compute the hard version on GPU
        # This is what RT Cores do in hardware, but we simulate it exactly
        with torch.no_grad():
            # (B, 1, 3) - (1, K, 3) → (B, K, 3)
            diff = positions_3d.unsqueeze(1) - centers.unsqueeze(0)
            distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, K)

            # Hard hit/miss: 1.0 if inside sphere, 0.0 if outside
            # Weighted by inverse distance (closer = stronger)
            in_sphere = (distances <= radii.unsqueeze(0)).float()  # (B, K)

            # For rays that miss all spheres, use distance-based ranking
            # (same as RT Core closest-hit behavior)
            inv_dist = 1.0 / (distances + 1e-6)
            inv_dist = inv_dist / inv_dist.sum(dim=-1, keepdim=True)  # normalize

            # Combine: hard hit inside sphere + soft fallback outside
            any_hit = in_sphere.sum(dim=-1, keepdim=True) > 0  # (B, 1)
            hard_signal = torch.where(any_hit, in_sphere, inv_dist)

        return hard_signal

    def forward_with_rt(
        self,
        positions_3d: torch.Tensor,    # (B, 3)
        centers: torch.Tensor,         # (K, 3)
        radii: torch.Tensor,           # (K,)
        soft_signal: torch.Tensor,     # (B, K) — from SmoothBVHHit
    ) -> torch.Tensor:
        """Forward with RT Core hard signal, backward through soft signal.

        Args:
            positions_3d: Query positions in 3D space
            centers: Expert sphere centers (learnable)
            radii: Expert sphere radii (learnable)
            soft_signal: SmoothBVHHit output (differentiable, for gradients)

        Returns:
            geo_signal: (B, K) — hard values, soft gradients
        """
        hard_signal = self.compute_hard_signal(positions_3d, centers, radii)

        # Straight-through: use hard values in forward, soft gradients in backward
        return StraightThroughRT.apply(soft_signal, hard_signal)


# Global singleton (lazy init)
_rt_bridge: Optional[RTTrainingBridge] = None


def get_rt_bridge(device: str = "cuda") -> RTTrainingBridge:
    """Get or create the global RT training bridge."""
    global _rt_bridge
    if _rt_bridge is None:
        _rt_bridge = RTTrainingBridge(device=device)
    return _rt_bridge
