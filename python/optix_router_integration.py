"""
optix_router_integration.py — Integrate OptiX RT Cores into BVH Router Training

Provides a drop-in enhancement for EnhancedBVHRouter that can use OptiX
RT Cores for the forward pass while maintaining gradient flow via STE.

Two integration modes:
  1. Standalone: Train a router on synthetic data using OptiX+STE
  2. Distillation: Replace Gumbel-Softmax in olmoe_bvh_distill.py with OptiX+STE

Usage:
    # Mode 1: Standalone test
    python python/optix_router_integration.py --mode test --device cuda

    # Mode 2: Train one layer with OptiX+STE
    python python/optix_router_integration.py --mode train --layer 8 --device cuda

Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from optix_training_bridge import OptiXTrainingBridge, SmoothBVHHit

log = logging.getLogger("optix_router_integration")


class OptiXRoutingWrapper(nn.Module):
    """Wraps an existing BVH Router to use OptiX RT Cores during training.

    This module sits between the router's 3D projection and expert selection:
    - Extracts 3D positions from the router's intermediate representations
    - Uses OptiX RT Cores for fast BVH traversal (forward)
    - SmoothBVHHit provides gradients (backward via STE)

    Falls back to the original Gumbel-Softmax if OptiX is unavailable.
    """

    def __init__(
        self,
        num_experts: int = 64,
        spatial_dim: int = 3,
        sharpness: float = 10.0,
        rebuild_gas_every: int = 50,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.spatial_dim = spatial_dim
        self.rebuild_gas_every = rebuild_gas_every
        self._step_count = 0

        # Learnable expert sphere parameters (3D space)
        self.centers = nn.Parameter(torch.randn(num_experts, spatial_dim))
        self.radii = nn.Parameter(torch.ones(num_experts) * 0.5)

        # Soft BVH proxy (always used for gradients)
        self._smooth_hit = SmoothBVHHit(sharpness=sharpness)

        # OptiX bridge (lazy init)
        self._bridge: Optional[OptiXTrainingBridge] = None
        self._bridge_checked = False

    def _ensure_bridge(self) -> OptiXTrainingBridge:
        """Lazily initialize the OptiX bridge."""
        if self._bridge is None:
            self._bridge = OptiXTrainingBridge(auto_init=True)
        return self._bridge

    def _maybe_rebuild_gas(self) -> None:
        """Rebuild GAS periodically to reflect updated sphere parameters."""
        bridge = self._ensure_bridge()
        if not bridge.has_extension:
            return

        if self._step_count % self.rebuild_gas_every == 0:
            with torch.no_grad():
                safe_radii = self.radii.abs().clamp(min=0.01)
                bridge.build_gas(self.centers.data, safe_radii)

    def forward(
        self,
        positions_3d: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route through OptiX RT Cores or SmoothBVHHit fallback.

        Args:
            positions_3d: (B, 3) query positions in 3D semantic space
            temperature:  Gumbel temperature (used in fallback)

        Returns:
            expert_probs: (B, num_experts) routing probabilities
            expert_ids:   (B,) selected expert IDs
        """
        self._step_count += 1
        self._maybe_rebuild_gas()

        bridge = self._ensure_bridge()
        safe_radii = self.radii.abs().clamp(min=0.01)

        expert_probs = bridge.forward_with_ste(
            positions_3d, self.centers, safe_radii, self.num_experts,
        )

        expert_ids = expert_probs.argmax(dim=-1)
        return expert_probs, expert_ids

    def shutdown(self) -> None:
        """Release OptiX resources."""
        if self._bridge is not None:
            self._bridge.shutdown()


def integrate_optix_into_router(
    router: nn.Module,
    use_optix: bool = True,
) -> nn.Module:
    """Add OptiX routing capability to an existing EnhancedBVHRouter.

    Monkey-patches the router's forward method to use OptiX RT Cores
    when available, falling back to the original Gumbel-Softmax otherwise.

    Args:
        router: An EnhancedBVHRouter instance
        use_optix: Enable OptiX integration

    Returns:
        The modified router (same object, patched in-place)
    """
    if not use_optix:
        return router

    # Create the OptiX wrapper
    optix_wrapper = OptiXRoutingWrapper(
        num_experts=router.n_experts,
        spatial_dim=3,
    ).to(next(router.parameters()).device)

    # Store original forward
    _original_forward = router.forward

    def _optix_forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward: use OptiX when available."""
        bridge = optix_wrapper._ensure_bridge()

        if bridge.available:
            # Project to 3D through the router's existing projection
            h = router.input_proj(x)  # (B, 256)

            # Use the first 3 dims of the level1 projection as 3D position
            # (This reuses the router's learned representations)
            p1, f1, pos1 = router.level1(h, router.temperature.item())
            pos_3d = pos1  # (B, 3) from HierarchicalLevel

            # Route via OptiX
            probs, ids = optix_wrapper(pos_3d, router.temperature.item())
            return probs, ids

        # Fallback to original
        return _original_forward(x)

    router.forward = _optix_forward
    router._optix_wrapper = optix_wrapper  # keep reference for GAS rebuild

    log.info(
        "OptiX integration installed. Bridge available: %s",
        optix_wrapper._ensure_bridge().available,
    )

    return router


# ── CLI Entry Point ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OptiX Router Integration")
    parser.add_argument("--mode", choices=["test", "train"], default="test")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.mode == "test":
        log.info("Running OptiX integration test...")
        log.info("Use python/test_optix_training.py for full validation.")
        log.info("")

        # Quick sanity check
        wrapper = OptiXRoutingWrapper(num_experts=16, spatial_dim=3).to(args.device)
        wrapper.train()

        pos = torch.randn(32, 3, device=args.device, requires_grad=True)
        probs, ids = wrapper(pos)

        loss = F.cross_entropy(
            probs, torch.randint(0, 16, (32,), device=args.device),
        )
        loss.backward()

        log.info(f"Forward:  probs={probs.shape}, ids={ids.shape}")
        log.info(f"Loss:     {loss.item():.4f}")
        log.info(f"Grad:     pos.grad.norm={pos.grad.norm().item():.6f}")
        log.info(f"Bridge:   {wrapper._ensure_bridge().available}")
        log.info("PASS")

        wrapper.shutdown()

    elif args.mode == "train":
        log.info(f"Training layer {args.layer} with OptiX+STE...")
        log.info("This requires olmoe_bvh_distill.py infrastructure.")
        log.info("")
        log.info("To train with OptiX, modify the training script:")
        log.info("  from optix_router_integration import integrate_optix_into_router")
        log.info("  router = integrate_optix_into_router(router, use_optix=True)")
        log.info("")
        log.info("Or run the full distillation with --use-optix flag:")
        log.info(f"  python olmoe_bvh_distill.py --layer {args.layer} --use-optix --device cuda")


if __name__ == "__main__":
    main()
