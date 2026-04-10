"""
optix_training_bridge.py — OptiX RT Core Integration for BVH Router Training

Connects the compiled OptiX RT Core pipeline to the PyTorch training loop
using Straight-Through Estimation (STE):

  Forward:  RT Cores (hardware BVH traversal, ~1µs)  → hard expert selection
  Backward: SmoothBVHHit soft proxy                   → differentiable gradients

This hybrid approach uses RT Core hardware for fast routing decisions
while maintaining gradient flow through a soft approximation.

Usage:
    bridge = OptiXTrainingBridge()
    if bridge.available:
        bridge.build_gas(centers, radii)
        expert_probs = bridge.forward_with_ste(
            positions_3d, directions, centers, radii, soft_signal
        )

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("optix_training_bridge")

# ── Try loading the compiled OptiX extension ──────────────────────────

_optix_ext = None
HAS_OPTIX_EXT = False

def _try_load_optix_ext():
    """Attempt to load the optix_training_ext PyTorch extension."""
    global _optix_ext, HAS_OPTIX_EXT

    try:
        import optix_training_ext
        _optix_ext = optix_training_ext
        HAS_OPTIX_EXT = True
        log.info("OptiX training extension loaded successfully")
        return True
    except ImportError:
        pass

    # Try JIT compilation as fallback
    try:
        from torch.utils.cpp_extension import load
        project_root = Path(__file__).parent.parent
        ext_src = project_root / "cuda" / "v5" / "optix_training_ext.cu"
        optix_host = project_root / "cuda" / "optix_router_host.cpp"

        if ext_src.exists() and optix_host.exists():
            _optix_ext = load(
                name="optix_training_ext",
                sources=[str(ext_src), str(optix_host)],
                extra_cflags=["-DOPTIX_TRAINING_STANDALONE"],
                verbose=False,
            )
            HAS_OPTIX_EXT = True
            log.info("OptiX training extension compiled via JIT")
            return True
    except Exception as exc:
        log.debug("OptiX JIT compilation failed: %s", exc)

    log.info("OptiX training extension not available — using PyTorch-only fallback")
    return False


# ── Auto-detect PTX files ─────────────────────────────────────────────

def _find_ptx_paths() -> Tuple[Optional[str], Optional[str]]:
    """Find compiled PTX shader files."""
    project_root = Path(__file__).parent.parent
    search_dirs = [
        project_root / "build" / "ptx",
        project_root / "build" / "Release" / "ptx",
        project_root / "build",
    ]

    raygen_path = None
    hitgroup_path = None

    for d in search_dirs:
        if not d.exists():
            continue
        # Prefer OptiX IR (.optixir) over PTX (.ptx) — faster pipeline creation
        for ext in ["*.optixir", "*.ptx"]:
            for f in d.glob(ext):
                name = f.stem.lower()
                if "raygen" in name and "router" in name and raygen_path is None:
                    raygen_path = str(f)
                elif "hitgroup" in name and "router" in name and hitgroup_path is None:
                    hitgroup_path = str(f)

    return raygen_path, hitgroup_path


# ═══════════════════════════════════════════════════════════════════════
# StraightThroughOptiX: autograd Function for hybrid RT/gradient training
# ═══════════════════════════════════════════════════════════════════════

class StraightThroughOptiX(torch.autograd.Function):
    """
    Straight-Through Estimator with OptiX RT Cores.

    Forward:  Uses RT Core hardware to perform BVH traversal → hard expert IDs.
              Converts hard IDs to one-hot probabilities (non-differentiable).
    Backward: Routes gradients through `soft_signal` (SmoothBVHHit output),
              which IS differentiable w.r.t. positions, centers, and radii.

    This is the core innovation: hardware-accelerated forward + soft backward.
    """

    @staticmethod
    def forward(
        ctx,
        soft_signal: torch.Tensor,     # (B, K) — differentiable soft routing
        hard_expert_ids: torch.Tensor,  # (B,) — from OptiX, int32
        num_experts: int,
    ) -> torch.Tensor:
        """
        Args:
            soft_signal:     Differentiable routing weights from SmoothBVHHit (B, K)
            hard_expert_ids: Expert IDs from OptiX RT Core traversal (B,) int32
            num_experts:     Total number of experts K

        Returns:
            (B, K) one-hot expert probabilities based on RT Core selection.
            Gradients flow through soft_signal in backward.
        """
        ctx.save_for_backward(soft_signal)

        B = hard_expert_ids.shape[0]
        device = hard_expert_ids.device

        # Create one-hot from hard expert IDs (non-differentiable forward)
        hard_onehot = torch.zeros(B, num_experts, device=device, dtype=soft_signal.dtype)

        # Handle miss sentinels (0xFFFFFFFF = -1 in int32)
        valid_mask = hard_expert_ids >= 0
        valid_ids = hard_expert_ids.clamp(0, num_experts - 1).long()
        hard_onehot.scatter_(1, valid_ids.unsqueeze(1), 1.0)

        # Zero out rows where RT Core missed (sentinel)
        hard_onehot = hard_onehot * valid_mask.float().unsqueeze(1)

        return hard_onehot

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Route gradients through the soft signal (STE)."""
        soft_signal, = ctx.saved_tensors

        # Straight-through: grad flows to soft_signal as if it were the forward
        grad_soft = grad_output

        return grad_soft, None, None


# ═══════════════════════════════════════════════════════════════════════
# SmoothBVHHit: differentiable soft proxy for RT Core traversal
# ═══════════════════════════════════════════════════════════════════════

class SmoothBVHHit(nn.Module):
    """
    Differentiable approximation of BVH sphere-ray intersection.

    Given 3D positions and sphere centers/radii, computes soft membership
    scores that approximate what RT Cores do in hardware.

    This provides the gradient signal for StraightThroughOptiX backward pass.
    """

    def __init__(self, sharpness: float = 10.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        positions: torch.Tensor,  # (B, 3) — query positions in 3D space
        centers: torch.Tensor,    # (K, 3) — sphere centers
        radii: torch.Tensor,      # (K,)   — sphere radii
    ) -> torch.Tensor:
        """
        Returns:
            (B, K) soft membership scores ∈ [0, 1].
            High score = query is inside or near sphere k.
        """
        # Distance from each query to each sphere center: (B, K)
        dists = torch.cdist(positions.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)

        # Normalized distance: how far outside the sphere (negative = inside)
        radii_expanded = radii.unsqueeze(0)  # (1, K)
        normalized_dist = (dists - radii_expanded) / radii_expanded.clamp(min=1e-6)

        # Soft hit: sigmoid gives smooth transition at sphere boundary
        soft_hit = torch.sigmoid(-normalized_dist * self.sharpness)

        return soft_hit


# ═══════════════════════════════════════════════════════════════════════
# OptiXTrainingBridge: high-level interface
# ═══════════════════════════════════════════════════════════════════════

class OptiXTrainingBridge:
    """
    High-level bridge between PyTorch BVH Router training and OptiX RT Cores.

    Handles:
      - Loading and initializing the OptiX extension
      - Building GAS from trained sphere parameters
      - Hybrid forward: RT Core hardware + SmoothBVHHit gradients
      - Graceful fallback to pure SmoothBVHHit when OptiX unavailable
    """

    def __init__(self, auto_init: bool = True):
        self._initialized = False
        self._gas_built = False
        self._smooth_hit = SmoothBVHHit(sharpness=10.0)

        if auto_init:
            self._try_init()

    @property
    def available(self) -> bool:
        """True if OptiX RT Cores are ready for use."""
        return self._initialized and self._gas_built

    @property
    def has_extension(self) -> bool:
        """True if the OptiX extension is loaded (even if GAS not built yet)."""
        return self._initialized

    def _try_init(self) -> bool:
        """Try to load extension and initialize OptiX."""
        if not HAS_OPTIX_EXT:
            _try_load_optix_ext()

        if not HAS_OPTIX_EXT or _optix_ext is None:
            return False

        if _optix_ext.is_ready():
            self._initialized = True
            self._gas_built = True
            return True

        raygen_ptx, hitgroup_ptx = _find_ptx_paths()
        if raygen_ptx is None or hitgroup_ptx is None:
            log.warning("PTX files not found. Build with CMake first: cmake --build build")
            return False

        try:
            _optix_ext.initialize(raygen_ptx, hitgroup_ptx)
            self._initialized = True
            log.info("OptiX initialized: raygen=%s, hitgroup=%s", raygen_ptx, hitgroup_ptx)
            return True
        except RuntimeError as exc:
            log.warning("OptiX initialization failed: %s", exc)
            return False

    def build_gas(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        use_triangles: bool = True,
    ) -> bool:
        """
        Build OptiX Geometry Acceleration Structure from sphere parameters.

        Args:
            centers: (K, 3) float32 — sphere centers (CPU or GPU, will be moved to CPU)
            radii:   (K,)   float32 — sphere radii
            use_triangles: Use octahedron GAS (more precise) vs AABB

        Returns:
            True if GAS built successfully
        """
        if not self._initialized:
            log.warning("Cannot build GAS: OptiX not initialized")
            return False

        # OptiX buildGAS expects CPU tensors
        centers_cpu = centers.detach().cpu().float().contiguous()
        radii_cpu = radii.detach().cpu().float().contiguous()

        try:
            _optix_ext.build_gas(centers_cpu, radii_cpu, use_triangles)
            self._gas_built = True
            gas_kb = _optix_ext.gas_size() / 1024
            log.info("GAS built: %d experts, %.1f KB, triangles=%s",
                     _optix_ext.num_experts(), gas_kb, use_triangles)
            return True
        except RuntimeError as exc:
            log.error("GAS build failed: %s", exc)
            return False

    def route_rt(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route using OptiX RT Cores (no gradients).

        Args:
            positions:  (B, 3) float32 CUDA — 3D query positions
            directions: (B, 3) float32 CUDA — ray directions

        Returns:
            (expert_ids: int32 (B,), distances: float32 (B,))
        """
        if not self.available:
            raise RuntimeError("OptiX not ready. Call build_gas() first.")

        return _optix_ext.route(
            positions.float().contiguous(),
            directions.float().contiguous(),
        )

    def route_rt_topk(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        top_k: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route with top-K using OptiX RT Cores.

        Returns:
            (topk_ids: int32 (B, K), topk_dists: float32 (B, K))
        """
        if not self.available:
            raise RuntimeError("OptiX not ready. Call build_gas() first.")

        return _optix_ext.route_topk(
            positions.float().contiguous(),
            directions.float().contiguous(),
            top_k,
        )

    def forward_with_ste(
        self,
        positions_3d: torch.Tensor,  # (B, 3) — from to_3d projection
        centers: torch.Tensor,       # (K, 3) — sphere centers (requires_grad)
        radii: torch.Tensor,         # (K,)   — sphere radii (requires_grad)
        num_experts: int,
    ) -> torch.Tensor:
        """
        Hybrid forward: RT Core hardware + STE backward.

        Forward path:
          1. Compute soft signal via SmoothBVHHit (differentiable)
          2. Compute directions from positions to nearest centers
          3. Route through OptiX RT Cores → hard expert IDs
          4. StraightThroughOptiX: return hard one-hot, grad through soft

        If OptiX unavailable, falls back to pure soft routing (Gumbel-Softmax style).

        Args:
            positions_3d: (B, 3) query positions in 3D semantic space
            centers:      (K, 3) sphere centers (learnable)
            radii:        (K,)   sphere radii (learnable)
            num_experts:  total number of experts

        Returns:
            (B, K) expert probabilities
        """
        # Step 1: Compute soft signal (always, for gradients)
        soft_signal = self._smooth_hit(positions_3d, centers, radii)  # (B, K)

        # Step 2: If OptiX available, use RT Cores for forward
        if self.available and not positions_3d.requires_grad:
            # Inference mode: pure RT Core, no STE needed
            directions = self._compute_directions(positions_3d, centers, soft_signal)
            expert_ids, _ = self.route_rt(positions_3d, directions)
            # Return one-hot (no grad needed)
            B = positions_3d.shape[0]
            onehot = torch.zeros(B, num_experts, device=positions_3d.device)
            valid = expert_ids >= 0
            onehot.scatter_(1, expert_ids.clamp(0).long().unsqueeze(1), 1.0)
            return onehot * valid.float().unsqueeze(1)

        if self.available:
            # Training mode: RT Core forward + STE backward
            directions = self._compute_directions(positions_3d, centers, soft_signal)

            with torch.no_grad():
                expert_ids, _ = self.route_rt(positions_3d, directions)

            return StraightThroughOptiX.apply(soft_signal, expert_ids, num_experts)

        # Fallback: pure soft routing (normalized soft_signal)
        return F.softmax(soft_signal * 10.0, dim=-1)

    def _compute_directions(
        self,
        positions: torch.Tensor,  # (B, 3)
        centers: torch.Tensor,    # (K, 3)
        soft_signal: torch.Tensor,  # (B, K)
    ) -> torch.Tensor:
        """
        Compute ray directions as weighted sum of directions to sphere centers.

        The direction for each query points toward the most likely expert
        (based on soft signal), ensuring the RT Core ray hits the right region.
        """
        # Weighted center: (B, 3)
        weights = F.softmax(soft_signal * 5.0, dim=-1)  # sharpen
        target = torch.mm(weights, centers)  # (B, 3)

        # Direction = target - origin (normalized)
        direction = target - positions
        direction = direction / (direction.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        return direction

    def shutdown(self) -> None:
        """Release OptiX resources."""
        if HAS_OPTIX_EXT and _optix_ext is not None:
            _optix_ext.shutdown()
        self._initialized = False
        self._gas_built = False


# ── Module-level convenience ──────────────────────────────────────────

_global_bridge: Optional[OptiXTrainingBridge] = None


def get_optix_bridge() -> OptiXTrainingBridge:
    """Get or create the global OptiX training bridge."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = OptiXTrainingBridge(auto_init=True)
    return _global_bridge
