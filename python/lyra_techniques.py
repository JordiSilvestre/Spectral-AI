"""Lyra-AGI techniques adapted for LiquidBit Zero-Matrix.

Implements 6 techniques from Lyra-AGI for BVH training and optimization:
  1. SmoothTernarySTE — Differentiable BVH via beta annealing
  2. RMSNorm (SubLN) — Mandatory post-routing normalization
  3. LiquidTimeGate — Per-channel LOCAL/GLOBAL temporal receptive field
  4. DualLR param groups — Separate LR for discrete BVH params
  5. MetabolicBVH — Age tracking + reserves + auto-pruning
  6. SmoothBVHHit — Soft BVH hit function (differentiable closest_hit)

Origin: jordisilvestre/Lyra-AGI (lyra/model/lyra_block.py, lyra/core/connectivity.py)
Adapted: 2026-03-29 for LiquidBit BVH training pipeline
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# 1. SmoothTernarySTE — differentiable ternary quantization with annealing
# ---------------------------------------------------------------------------

_ste_beta: float = 1.0


def get_ste_beta() -> float:
    return _ste_beta


def set_ste_beta(beta: float):
    global _ste_beta
    _ste_beta = beta


class SmoothTernarySTE(torch.autograd.Function):
    """Smooth Straight-Through Estimator for ternary/discrete quantization.

    Forward: D_ternary = sign(D_cont) * tanh(beta * (|D_cont| - 0.5))
    Backward: Straight-through with gradient scaling to prevent dead zones.

    Beta annealing schedule:
      - beta=1.0 (start): soft, gradients flow freely, network explores
      - beta=10.0 (end): hard ternary {-1, 0, +1}, equivalent to RT Core discrete

    For LiquidBit BVH: replaces hard hit/miss with soft hit function.
    """

    @staticmethod
    def forward(ctx, D_cont):
        beta = _ste_beta
        if beta >= 10.0:
            D_ternary = torch.where(
                D_cont > 0.5, torch.ones_like(D_cont),
                torch.where(D_cont < -0.5, -torch.ones_like(D_cont),
                             torch.zeros_like(D_cont)))
        else:
            magnitude = torch.tanh(beta * (D_cont.abs() - 0.5)).clamp(0, 1)
            D_ternary = magnitude * D_cont.sign()
        ctx.save_for_backward(D_cont)
        ctx.beta = beta
        return D_ternary

    @staticmethod
    def backward(ctx, grad_output):
        D_cont, = ctx.saved_tensors
        scale = 1.0 - torch.tanh(D_cont.abs() - 2.0).clamp(0, 1)
        return grad_output * scale


ternary_ste = SmoothTernarySTE.apply


# ---------------------------------------------------------------------------
# 2. SmoothBVHHit — soft BVH hit for differentiable attention
# ---------------------------------------------------------------------------

class SmoothBVHHit(nn.Module):
    """Differentiable BVH hit function using SmoothSTE principle.

    Replaces discrete hit/miss in closest_hit.cu with a soft version
    for training. At high beta, converges to hard RT Core behavior.

    Math:
      soft_hit = tanh(beta * (semantic_radius - distance))
      attention = soft_hit * energy * exp(-lambda * distance)

    This is the KEY technique that enables end-to-end BVH training.
    """

    def __init__(self, lambda_decay: float = 0.1):
        super().__init__()
        self.lambda_decay = lambda_decay

    def forward(self, distances: torch.Tensor, radii: torch.Tensor,
                energy: torch.Tensor) -> torch.Tensor:
        """Compute soft attention weights.

        Args:
            distances: (B, N) semantic distances to BVH nodes
            radii: (N,) semantic radii of BVH spheres
            energy: (B,) remaining ray energy

        Returns:
            attention_weights: (B, N) soft attention for each node
        """
        beta = get_ste_beta()
        soft_hit = torch.tanh(beta * (radii.unsqueeze(0) - distances)).clamp(0, 1)
        decay = torch.exp(-self.lambda_decay * distances)
        attention = soft_hit * energy.unsqueeze(1) * decay
        return attention


# ---------------------------------------------------------------------------
# 3. RMSNorm (SubLN) — mandatory post-routing normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    MANDATORY after BVH routing to prevent scale collapse.
    Without SubLN: 100% saturation, all connections collapse.
    With SubLN: stable convergence for 1000+ steps.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# 4. LiquidTimeGate — per-channel temporal receptive field
# ---------------------------------------------------------------------------

class LiquidTimeGate(nn.Module):
    """Per-channel temporal gate with LOCAL/GLOBAL specialization.

    gate(i, pos) = sigmoid(10 * a_i * dist + b_i)

    During training, channels self-specialize:
      a < 0 → LOCAL (attenuates distant tokens)
      a > 0 → GLOBAL (favors broad context)

    For LiquidBit spectral rays:
      LOCAL channels → short-range spectral rays (blue)
      GLOBAL channels → long-range spectral rays (red)

    Validated: -6.4% loss vs no gate (Lyra, TinyStories)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.time_a = nn.Parameter(torch.zeros(d_model))
        self.time_b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal gating. Input: (B, T, D) or (B, D)."""
        if x.dim() == 2:
            return x  # no temporal dimension, passthrough

        B, T, D = x.shape
        device = x.device
        pos = torch.arange(T, device=device, dtype=torch.float32)
        dist = (T - 1 - pos) / max(T - 1, 1)
        dist = dist.unsqueeze(1)  # (T, 1)

        gate_logits = (10.0 * self.time_a * dist + self.time_b).clamp(-50, 50)
        gate = torch.sigmoid(gate_logits)  # (T, D)
        return x * gate.unsqueeze(0)

    def gate_stats(self) -> dict:
        with torch.no_grad():
            a = self.time_a
            return {
                "n_local": int((a < -0.01).sum()),
                "n_global": int((a > 0.01).sum()),
                "n_uniform": int(a.numel()) - int((a < -0.01).sum()) - int((a > 0.01).sum()),
                "a_mean": float(a.mean()),
                "a_min": float(a.min()),
                "a_max": float(a.max()),
            }


# ---------------------------------------------------------------------------
# 5. DualLR — separate learning rates for BVH/discrete vs float params
# ---------------------------------------------------------------------------

def get_dual_lr_param_groups(model: nn.Module, lr: float = 3e-4,
                             bvh_lr_mult: float = 0.1,
                             weight_decay: float = 0.01,
                             bvh_param_keywords: tuple = ('D_cont', 'centroid',
                                                          'radius', 'bvh_weight')
                             ) -> list:
    """Create parameter groups with dual learning rates.

    BVH/discrete params get lower LR to prevent oscillation.
    Float params (FFN, norms, embeddings) get standard LR.

    Without Dual LR: NaN in <10 steps.
    With Dual LR: 100% stability, 0 NaN in 10K+ steps.

    Args:
        model: the model
        lr: base learning rate for float params
        bvh_lr_mult: multiplier for BVH params (0.1 = 10x lower)
        weight_decay: weight decay for float params with dim >= 2
        bvh_param_keywords: parameter name patterns for BVH group
    """
    bvh_params = []
    float_params_decay = []
    float_params_no_decay = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in bvh_param_keywords):
            bvh_params.append(p)
        elif p.dim() >= 2:
            float_params_decay.append(p)
        else:
            float_params_no_decay.append(p)

    groups = []
    if bvh_params:
        groups.append({
            "params": bvh_params, "lr": lr * bvh_lr_mult,
            "weight_decay": 0.0, "name": "bvh_discrete",
        })
    if float_params_decay:
        groups.append({
            "params": float_params_decay, "lr": lr,
            "weight_decay": weight_decay, "name": "float_decay",
        })
    if float_params_no_decay:
        groups.append({
            "params": float_params_no_decay, "lr": lr,
            "weight_decay": 0.0, "name": "float_no_decay",
        })
    return groups


# ---------------------------------------------------------------------------
# 6. MetabolicBVH — age tracking + reserves + auto-pruning
# ---------------------------------------------------------------------------

class MetabolicBVH:
    """Self-pruning BVH with metabolic reserves and age tracking.

    Each BVH node has:
      - age: steps since last ray hit (uint16)
      - reserves: metabolic energy (float, 0-2)
      - hit_count: rays received this step

    Pruning rules:
      - Nodes with age > max_age are removed
      - Nodes with reserves <= 0 collapse (too many children, not enough hits)
      - Pruned nodes free memory and reduce traversal depth

    Results in Lyra: sparsity auto-grows 0.90 → 0.95

    Args:
        n_nodes: number of BVH nodes/spheres
        max_age: steps before pruning unused nodes
        energy_cost: energy cost per child connection per step
        energy_regen: energy regenerated per step (base)
        energy_hit_bonus: extra energy per ray hit
    """

    def __init__(self, n_nodes: int, max_age: int = 1000,
                 energy_cost: float = 0.001, energy_regen: float = 0.01,
                 energy_hit_bonus: float = 0.005):
        self.n_nodes = n_nodes
        self.max_age = max_age
        self.energy_cost = energy_cost
        self.energy_regen = energy_regen
        self.energy_hit_bonus = energy_hit_bonus

        self.age = np.zeros(n_nodes, dtype=np.uint16)
        self.reserves = np.ones(n_nodes, dtype=np.float32)
        self.hit_counts = np.zeros(n_nodes, dtype=np.int32)
        self.active = np.ones(n_nodes, dtype=bool)

        # Stats
        self._last_pruned = 0
        self._last_revived = 0

    def record_hits(self, hit_node_ids: np.ndarray):
        """Record which nodes were hit by rays this step."""
        for nid in hit_node_ids:
            if 0 <= nid < self.n_nodes:
                self.hit_counts[nid] += 1

    def step(self, children_counts: Optional[np.ndarray] = None) -> dict:
        """Advance one metabolic step. Call after each inference batch.

        Args:
            children_counts: (n_nodes,) number of children per node.
                If None, assumes 1 child per active node.

        Returns:
            Stats dict with pruned/revived counts.
        """
        if children_counts is None:
            children_counts = np.ones(self.n_nodes, dtype=np.float32)

        # Age: increment for all active, reset for hit nodes
        was_hit = self.hit_counts > 0
        self.age[self.active & ~was_hit] += 1
        self.age[was_hit] = 0

        # Reserves: cost for children, regen base + hit bonus
        cost = self.energy_cost * children_counts
        regen = self.energy_regen + self.energy_hit_bonus * self.hit_counts
        self.reserves = np.clip(self.reserves - cost + regen, 0.0, 2.0)

        # Pruning: too old or out of energy
        prune_age = self.active & (self.age > self.max_age)
        prune_energy = self.active & (self.reserves <= 0)
        pruned = prune_age | prune_energy

        self._last_pruned = int(pruned.sum())
        self.active[pruned] = False
        self.age[pruned] = 0
        self.reserves[pruned] = 0.0

        # Reset hit counts for next step
        self.hit_counts[:] = 0

        return self.stats()

    def revive(self, node_ids: np.ndarray):
        """Revive pruned nodes (e.g., when new tokens arrive)."""
        for nid in node_ids:
            if 0 <= nid < self.n_nodes and not self.active[nid]:
                self.active[nid] = True
                self.age[nid] = 0
                self.reserves[nid] = 1.0
                self._last_revived += 1

    def stats(self) -> dict:
        n_active = int(self.active.sum())
        return {
            "n_active": n_active,
            "n_pruned": self.n_nodes - n_active,
            "last_pruned": self._last_pruned,
            "sparsity": 1.0 - n_active / max(self.n_nodes, 1),
            "reserves_mean": float(self.reserves[self.active].mean()) if n_active > 0 else 0.0,
            "age_mean": float(self.age[self.active].mean()) if n_active > 0 else 0.0,
            "age_max": int(self.age.max()),
        }


# ---------------------------------------------------------------------------
# 7. BetaScheduler — manages STE beta annealing
# ---------------------------------------------------------------------------

class BetaScheduler:
    """Linear beta annealing for SmoothSTE.

    Schedule: beta = 1.0 during warmup, then linear to max_beta.

    Args:
        max_beta: final beta value (10.0 recommended, >15 causes instability)
        warmup_steps: steps before annealing starts
        total_steps: total training steps
    """

    def __init__(self, max_beta: float = 10.0, warmup_steps: int = 1000,
                 total_steps: int = 10000):
        self.max_beta = min(max_beta, 15.0)  # cap at 15 for safety
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def step(self, current_step: int):
        """Update global STE beta based on current step."""
        if current_step < self.warmup_steps:
            beta = 1.0
        else:
            progress = (current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            beta = 1.0 + (self.max_beta - 1.0) * progress
        set_ste_beta(beta)
        return beta


# ---------------------------------------------------------------------------
# GPU-ready CUDA kernel stubs (for user to compile on RTX 5070 Ti)
# ---------------------------------------------------------------------------

SMOOTH_BVH_HIT_CUDA_STUB = """
// smooth_bvh_hit.cu — Differentiable BVH hit for training
// Compile with: nvcc -arch=sm_89 -arch=sm_120 -c smooth_bvh_hit.cu
//
// Replaces hard hit/miss in closest_hit.cu with soft version:
//   soft_hit = tanhf(beta * (semantic_radius - distance));
//   attention = soft_hit * energy * expf(-lambda * distance);
//
// In closest_hit.cu, replace lines 111-118 with:
//
//   float soft_hit = tanhf(c_beta * (semantic_radius - semantic_distance));
//   soft_hit = fmaxf(soft_hit, 0.0f);  // clamp negative
//   float attention_weight = soft_hit * energy_remaining
//                          * expf(-c_lambda * semantic_distance);
//
// Where c_beta is a __constant__ float that anneals 1.0 → 10.0
// Add to ray_attention.cu constants:
//   __constant__ float c_beta = 1.0f;
//
// Host-side update per training step:
//   cudaMemcpyToSymbol(c_beta, &new_beta, sizeof(float));
"""
