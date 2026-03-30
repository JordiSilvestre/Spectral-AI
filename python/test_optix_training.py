#!/usr/bin/env python3
"""
test_optix_training.py — Validate OptiX RT Core Training Bridge

Tests both the OptiX hardware path (if available) and the SmoothBVHHit
fallback. Compares training curves: Gumbel-Softmax vs OptiX+STE vs
pure SmoothBVHHit on a synthetic routing task.

Usage:
    python python/test_optix_training.py [--device cuda] [--steps 200]

Copyright (c) 2026 SpectralAI Studio -- Apache 2.0
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Synthetic Routing Dataset ──────────────────────────────────────────

def create_synthetic_routing_data(
    num_samples: int = 2048,
    input_dim: int = 256,
    num_experts: int = 16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data where inputs cluster around expert centers.

    Each sample is generated near one expert center, creating a clear
    routing target. This isolates the router's learning ability.
    """
    # Create expert centers spread in input space
    centers = torch.randn(num_experts, input_dim, device=device) * 2.0

    # Generate samples clustered around random experts
    expert_labels = torch.randint(0, num_experts, (num_samples,), device=device)
    noise = torch.randn(num_samples, input_dim, device=device) * 0.3
    inputs = centers[expert_labels] + noise

    return inputs, expert_labels


# ── Simple Router Module (for testing) ─────────────────────────────────

class SimpleRouter(nn.Module):
    """Minimal router for testing different routing strategies."""

    def __init__(self, input_dim: int, num_experts: int, spatial_dim: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.spatial_dim = spatial_dim

        # Project to 3D space (for BVH routing)
        self.to_3d = nn.Linear(input_dim, spatial_dim)

        # Learnable expert sphere centers and radii
        self.centers = nn.Parameter(torch.randn(num_experts, spatial_dim) * 1.0)
        self.radii = nn.Parameter(torch.ones(num_experts) * 0.5)

        # Direct logit head (for Gumbel-Softmax comparison)
        self.logit_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_experts),
        )

    def forward_gumbel(
        self, x: torch.Tensor, temperature: float = 1.0,
    ) -> torch.Tensor:
        """Standard Gumbel-Softmax routing (baseline)."""
        logits = self.logit_head(x)
        if self.training:
            return F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=False)
        return F.softmax(logits / max(temperature, 0.1), dim=-1)

    def forward_smooth_bvh(self, x: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
        """Pure SmoothBVHHit routing (differentiable BVH proxy)."""
        pos_3d = self.to_3d(x)  # (B, 3)

        # Distance-based soft membership
        dists = torch.cdist(pos_3d.unsqueeze(0), self.centers.unsqueeze(0)).squeeze(0)
        radii_exp = self.radii.abs().unsqueeze(0).clamp(min=1e-6)
        normalized = (dists - radii_exp) / radii_exp
        soft_hit = torch.sigmoid(-normalized * sharpness)

        return F.softmax(soft_hit * 10.0, dim=-1)

    def forward_optix_ste(
        self, x: torch.Tensor, bridge: "OptiXTrainingBridge",
    ) -> torch.Tensor:
        """OptiX RT Core + STE routing (hardware forward, soft backward)."""
        pos_3d = self.to_3d(x)
        return bridge.forward_with_ste(
            pos_3d, self.centers, self.radii, self.num_experts,
        )


# ── Training Loop ──────────────────────────────────────────────────────

def train_router(
    router: SimpleRouter,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    bridge: object = None,
    num_steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> Dict[str, List[float]]:
    """Train a router with the given method and track metrics."""
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"loss": [], "top1_acc": [], "top8_acc": []}

    num_samples = inputs.shape[0]
    temperature = 1.0

    for step in range(num_steps):
        # Random batch
        idx = torch.randint(0, num_samples, (batch_size,), device=inputs.device)
        batch_x = inputs[idx]
        batch_y = labels[idx]

        # Forward based on method
        if method == "gumbel":
            probs = router.forward_gumbel(batch_x, temperature)
        elif method == "smooth_bvh":
            probs = router.forward_smooth_bvh(batch_x)
        elif method == "optix_ste":
            probs = router.forward_optix_ste(batch_x, bridge)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cross-entropy loss
        loss = F.cross_entropy(probs + 1e-8, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            top1 = (probs.argmax(dim=-1) == batch_y).float().mean().item()
            _, topk = probs.topk(min(8, router.num_experts), dim=-1)
            top8 = (topk == batch_y.unsqueeze(1)).any(dim=-1).float().mean().item()

        history["loss"].append(loss.item())
        history["top1_acc"].append(top1)
        history["top8_acc"].append(top8)

        # Anneal temperature
        if step % 20 == 0 and step > 0:
            temperature = max(0.1, temperature * 0.95)

        if step % 50 == 0:
            log.info(
                f"  [{method:12s}] step {step:4d} | "
                f"loss={loss.item():.4f} | "
                f"top1={top1:.3f} | "
                f"top8={top8:.3f} | "
                f"T={temperature:.3f}"
            )

    return history


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test OptiX Training Bridge")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = args.device
    log.info("=" * 70)
    log.info("SpectralAI OptiX RT Core Training Bridge — Validation Test")
    log.info("=" * 70)

    if device == "cuda" and torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Device: {device}")
    log.info(f"Experts: {args.num_experts}")
    log.info(f"Steps: {args.steps}")
    log.info("")

    # Create data
    input_dim = 256
    inputs, labels = create_synthetic_routing_data(
        num_samples=4096, input_dim=input_dim,
        num_experts=args.num_experts, device=device,
    )
    log.info(f"Synthetic data: {inputs.shape[0]} samples, {input_dim}D -> {args.num_experts} experts")
    log.info("")

    results = {}

    # ── Method 1: Gumbel-Softmax (baseline) ──────────────────────
    log.info("=" * 50)
    log.info("Method 1: Gumbel-Softmax (standard MoE baseline)")
    log.info("=" * 50)
    router_gumbel = SimpleRouter(input_dim, args.num_experts).to(device)
    router_gumbel.train()
    t0 = time.time()
    results["gumbel"] = train_router(
        router_gumbel, inputs, labels, "gumbel",
        num_steps=args.steps, lr=args.lr, batch_size=args.batch_size,
    )
    t_gumbel = time.time() - t0
    log.info(f"  Time: {t_gumbel:.1f}s")
    log.info("")

    # ── Method 2: SmoothBVHHit (differentiable BVH proxy) ────────
    log.info("=" * 50)
    log.info("Method 2: SmoothBVHHit (differentiable BVH)")
    log.info("=" * 50)
    router_smooth = SimpleRouter(input_dim, args.num_experts).to(device)
    router_smooth.train()
    t0 = time.time()
    results["smooth_bvh"] = train_router(
        router_smooth, inputs, labels, "smooth_bvh",
        num_steps=args.steps, lr=args.lr, batch_size=args.batch_size,
    )
    t_smooth = time.time() - t0
    log.info(f"  Time: {t_smooth:.1f}s")
    log.info("")

    # ── Method 3: OptiX+STE (if available) ────────────────────────
    try:
        from optix_training_bridge import OptiXTrainingBridge

        bridge = OptiXTrainingBridge(auto_init=True)

        if bridge.has_extension:
            log.info("=" * 50)
            log.info("Method 3: OptiX RT Core + STE (hardware-accelerated)")
            log.info("=" * 50)

            router_optix = SimpleRouter(input_dim, args.num_experts).to(device)
            router_optix.train()

            # Build GAS from initial sphere parameters
            with torch.no_grad():
                bridge.build_gas(router_optix.centers.data, router_optix.radii.data.abs())

            t0 = time.time()
            results["optix_ste"] = train_router(
                router_optix, inputs, labels, "optix_ste",
                bridge=bridge, num_steps=args.steps,
                lr=args.lr, batch_size=args.batch_size,
            )
            t_optix = time.time() - t0
            log.info(f"  Time: {t_optix:.1f}s")

            bridge.shutdown()
        else:
            log.info("[INFO] OptiX extension loaded but not initialized (no PTX?)")
            log.info("       OptiX+STE test skipped. Fallback to SmoothBVHHit inside bridge.")

            # Test the fallback path
            log.info("=" * 50)
            log.info("Method 3: OptiX bridge FALLBACK (pure soft routing)")
            log.info("=" * 50)
            router_fallback = SimpleRouter(input_dim, args.num_experts).to(device)
            router_fallback.train()
            t0 = time.time()
            results["optix_fallback"] = train_router(
                router_fallback, inputs, labels, "optix_ste",
                bridge=bridge, num_steps=args.steps,
                lr=args.lr, batch_size=args.batch_size,
            )
            t_fallback = time.time() - t0
            log.info(f"  Time: {t_fallback:.1f}s")

    except ImportError:
        log.info("[INFO] optix_training_bridge not importable — skipping OptiX test")
        log.info("       Build with: python cuda/v5/build_optix_ext.py")

    # ── Summary ───────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY — Final Metrics (last 20 steps average)")
    log.info("=" * 70)
    log.info(f"{'Method':<20s} {'Loss':>8s} {'Top-1':>8s} {'Top-8':>8s}")
    log.info("-" * 50)

    for method, hist in results.items():
        avg_loss = sum(hist["loss"][-20:]) / 20
        avg_top1 = sum(hist["top1_acc"][-20:]) / 20
        avg_top8 = sum(hist["top8_acc"][-20:]) / 20
        log.info(f"{method:<20s} {avg_loss:>8.4f} {avg_top1:>8.3f} {avg_top8:>8.3f}")

    log.info("")
    log.info("If OptiX+STE matches Gumbel-Softmax accuracy within ~5%,")
    log.info("the RT Core bridge is validated for production training.")
    log.info("=" * 70)

    # Check SmoothBVHHit module independently
    log.info("")
    log.info("=" * 50)
    log.info("Unit Test: SmoothBVHHit gradient flow")
    log.info("=" * 50)

    from optix_training_bridge import SmoothBVHHit

    smooth = SmoothBVHHit(sharpness=10.0)
    pos = torch.randn(4, 3, device=device, requires_grad=True)
    ctrs = torch.randn(8, 3, device=device, requires_grad=True)
    # radii is a leaf tensor — SmoothBVHHit uses it in operations, so we need
    # retain_grad() if we want .grad on intermediate tensors. Using a leaf directly:
    rads = torch.ones(8, device=device) * 0.5
    rads.requires_grad_(True)  # leaf + requires_grad

    out = smooth(pos, ctrs, rads)
    loss_test = out.sum()
    loss_test.backward()

    assert pos.grad is not None, "No gradient on positions!"
    assert ctrs.grad is not None, "No gradient on centers!"
    assert rads.grad is not None, "No gradient on radii!"
    log.info(f"  Output shape: {out.shape}")
    log.info(f"  pos.grad norm:  {pos.grad.norm().item():.6f}")
    log.info(f"  ctrs.grad norm: {ctrs.grad.norm().item():.6f}")
    log.info(f"  rads.grad norm: {rads.grad.norm().item():.6f}")
    log.info("  PASS: All gradients flowing correctly")

    # Check StraightThroughOptiX
    log.info("")
    log.info("=" * 50)
    log.info("Unit Test: StraightThroughOptiX gradient passthrough")
    log.info("=" * 50)

    from optix_training_bridge import StraightThroughOptiX

    soft = torch.randn(4, 8, device=device, requires_grad=True)
    hard_ids = torch.tensor([0, 3, 5, 1], dtype=torch.int32, device=device)

    result = StraightThroughOptiX.apply(soft, hard_ids, 8)
    assert result.shape == (4, 8), f"Wrong shape: {result.shape}"

    # Check it's one-hot
    assert (result.sum(dim=-1) == 1.0).all(), "Not one-hot!"
    assert result[0, 0] == 1.0, f"Wrong expert selected: {result[0]}"
    assert result[1, 3] == 1.0, f"Wrong expert selected: {result[1]}"

    # Check gradient flows through soft
    loss_ste = (result * torch.randn_like(result)).sum()
    loss_ste.backward()
    assert soft.grad is not None, "No gradient through STE!"
    log.info(f"  Output shape: {result.shape}")
    log.info(f"  One-hot: PASS")
    log.info(f"  soft.grad norm: {soft.grad.norm().item():.6f}")
    log.info("  PASS: STE gradient passthrough working")

    log.info("")
    log.info("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
