#!/usr/bin/env python3
"""
olmoe_bvh_distill.py -- Train BVH Router to replicate OLMoE gate routing

FASE A v2: Instead of training experts from scratch, use OLMoE's 64 real
pre-specialized SwiGLU experts (frozen). Train ONLY the BVH router to
replicate OLMoE's linear gate routing decisions.

v2.1 FIX: The original BVH router compressed 2048→3 dims per level,
losing all routing information. The EnhancedBVHRouter preserves rich
features through the hierarchy while still using 3D geometry for
the hierarchical structure.

Training objective:
    L_total = L_ce + w_balance * L_balance

Where:
    L_ce:      Cross-entropy on top-1 expert selection (hard target)
    L_balance: Load balancing (even expert usage)

Usage:
    python olmoe_bvh_distill.py [--epochs 30] [--layer 8] [--device cuda]

Copyright (c) 2026 LiquidBit Studio -- Apache 2.0
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from olmoe_extract import load_olmoe_layer, OLMoELayer, OLMoELayerConfig


# ─────────────────────────────────────────────────────────────────
# Sparse Upcycling — initialize BVH Router from OLMoE gate weights
# ─────────────────────────────────────────────────────────────────

def initialize_router_from_gate(
    router: "EnhancedBVHRouter",
    gate_weight: torch.Tensor,
    verbose: bool = True,
) -> None:
    """
    Sparse Upcycling: initialize the BVH router hierarchy from OLMoE's
    linear gate weight matrix [64, 2048].

    Each row of the gate is an "expert prototype" — the direction in
    hidden-state space that activates that expert. We use K-means
    clustering on these prototypes to build the 3-level hierarchy:

        Level 1: 4 clusters of 16 experts each (domains)
        Level 2: 4 sub-clusters within each domain (subdomains)
        Level 3: 4 sub-sub-clusters (concepts → individual experts)

    This gives the router a massive head start: instead of learning
    from scratch which regions of hidden-state space map to which experts,
    it starts with the gate's own knowledge of expert specialization.

    Also initializes:
        - input_proj first layer from gate weight PCA
        - expert_head last layer from gate weight directly
        - level centers from K-means on projected gate rows
    """
    W = gate_weight.detach().float().cpu()  # [64, 2048]
    n_experts, hidden_dim = W.shape

    if verbose:
        print(f"\n  [Sparse Upcycling] Initializing router from gate [{n_experts}, {hidden_dim}]")

    # ── 1. Initialize input_proj first layer from gate's principal components ──
    # The gate's rows span the subspace the gate cares about.
    # Use SVD to find the most important directions.
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)  # U:[64,64], S:[64], Vt:[64,2048]
    # Top-512 directions (input_proj maps 2048→512)
    n_components = min(512, Vt.shape[0])
    top_directions = Vt[:n_components]  # [512, 2048] (or [64, 2048] if <512)

    with torch.no_grad():
        # Initialize input_proj[0] (Linear 2048→512) from gate's top directions
        proj_weight = router.input_proj[0].weight  # [512, 2048]
        n_copy = min(top_directions.shape[0], proj_weight.shape[0])
        # Copy SVD directions into the first n_copy rows
        proj_weight[:n_copy] = top_directions[:n_copy]
        # Scale only the copied rows to match expected activation magnitude
        scale = (S[:n_copy] / S[0]).unsqueeze(1)  # [n_copy, 1]
        proj_weight[:n_copy] *= scale

    if verbose:
        variance_explained = (S[:n_copy] ** 2).sum() / (S ** 2).sum()
        print(f"  input_proj initialized from top-{n_copy} SVD directions "
              f"({variance_explained:.1%} variance)")

    # ── 2. Hierarchical K-means on gate rows for level centers ──
    # Project gate rows to the router's internal representation
    with torch.no_grad():
        # Pass gate rows through input_proj to get 256-dim representations
        W_device = W.to(router.input_proj[0].weight.device)
        h = router.input_proj(W_device)  # [64, 256]
        h = h.cpu()

    # Level 1: cluster 64 experts into 4 groups
    l1_labels, l1_centers = _kmeans(h, router.n_level1, seed=42)

    if verbose:
        for i in range(router.n_level1):
            members = (l1_labels == i).sum().item()
            print(f"  L1 cluster {i}: {members} experts")

    # Set Level 1 centers (project to 3D for geometric routing)
    with torch.no_grad():
        pos1 = router.level1.to_3d(l1_centers.to(router.level1.to_3d.weight.device))
        router.level1.centers.copy_(pos1)

    # Level 2: within each L1 cluster, cluster into 4 sub-groups
    l2_labels_global = torch.zeros(n_experts, dtype=torch.long)
    l2_features = []
    for g1 in range(router.n_level1):
        mask = (l1_labels == g1)
        group_h = h[mask]
        if len(group_h) < router.n_level2:
            # Too few experts — assign sequentially
            sub_labels = torch.arange(len(group_h)) % router.n_level2
        else:
            # Get features from Level 1
            with torch.no_grad():
                _, f1, _ = router.level1(
                    group_h.to(router.level1.to_3d.weight.device)
                )
                f1 = f1.cpu()
            sub_labels, sub_centers = _kmeans(f1, router.n_level2, seed=42 + g1)
            l2_features.append((f1, sub_labels, sub_centers))

        # Map back to global indices
        global_idx = mask.nonzero(as_tuple=True)[0]
        for local_i, global_i in enumerate(global_idx):
            l2_labels_global[global_i] = g1 * router.n_level2 + sub_labels[local_i]

    # Set Level 2 centers from the first group's sub-centers (representative)
    if l2_features:
        _, _, first_sub_centers = l2_features[0]
        with torch.no_grad():
            pos2 = router.level2.to_3d(
                first_sub_centers.to(router.level2.to_3d.weight.device)
            )
            router.level2.centers.copy_(pos2)

    if verbose:
        n_l2_clusters = l2_labels_global.unique().numel()
        print(f"  L2: {n_l2_clusters} sub-clusters created")

    # ── 3. Initialize expert_head last layer from gate weight ──
    # The expert_head maps [feature_dim + n_l1 + n_l2 + n_l3] → 64
    # Initialize its output weights so each expert's output neuron
    # has a reasonable starting point
    with torch.no_grad():
        head_last = router.expert_head[-1]  # Linear(256, 64)
        # Initialize bias: experts that are more "popular" in the gate
        # get a slight positive bias
        gate_norms = W.norm(dim=1)  # [64]
        gate_norms = (gate_norms - gate_norms.mean()) / (gate_norms.std() + 1e-8)
        if head_last.bias is not None:
            head_last.bias.copy_(gate_norms * 0.1)

    if verbose:
        print(f"  expert_head bias initialized from gate weight norms")
        print(f"  [Sparse Upcycling] Done — router pre-initialized from gate knowledge\n")


def _kmeans(
    data: torch.Tensor, k: int, n_iter: int = 50, seed: int = 42
) -> tuple:
    """
    Simple K-means clustering on CPU tensors.

    Returns:
        labels: (N,) cluster assignments
        centers: (K, D) cluster centers
    """
    N, D = data.shape
    rng = torch.Generator().manual_seed(seed)
    # K-means++ initialization
    indices = [torch.randint(N, (1,), generator=rng).item()]
    for _ in range(1, k):
        dists = torch.cdist(data, data[indices]).min(dim=1).values  # (N,)
        probs = dists ** 2
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1, generator=rng).item()
        indices.append(idx)

    centers = data[indices].clone()  # (K, D)

    for _ in range(n_iter):
        # Assign
        dists = torch.cdist(data, centers)  # (N, K)
        labels = dists.argmin(dim=1)  # (N,)

        # Update centers
        new_centers = torch.zeros_like(centers)
        for i in range(k):
            mask = (labels == i)
            if mask.any():
                new_centers[i] = data[mask].mean(dim=0)
            else:
                new_centers[i] = centers[i]

        if (new_centers - centers).abs().max() < 1e-6:
            break
        centers = new_centers

    return labels, centers


# ─────────────────────────────────────────────────────────────────
# Enhanced BVH Router — preserves information through hierarchy
# ─────────────────────────────────────────────────────────────────

class HierarchicalLevel(nn.Module):
    """
    One level of the BVH hierarchy.

    Instead of pure geometric distance in 3D, this uses a learned
    projection to 3D for the geometric structure, but ALSO extracts
    a high-dimensional feature vector that carries forward.

    The 3D geometry determines the routing structure (which branch),
    while the feature vector preserves the full semantic information.
    """

    def __init__(self, input_dim: int, n_children: int, feature_dim: int = 128):
        super().__init__()
        self.n_children = n_children
        self.feature_dim = feature_dim

        # 3D geometric routing (the BVH part)
        self.to_3d = nn.Linear(input_dim, 3)
        self.centers = nn.Parameter(torch.randn(n_children, 3) * 0.3)
        self.log_radii = nn.Parameter(torch.zeros(n_children))

        # Feature extraction (preserves information)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Combine geometry + features for routing logits
        self.route_head = nn.Linear(feature_dim + 3, n_children)

    def forward(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, input_dim)

        Returns:
            probs:    (B, n_children) — soft routing probabilities
            features: (B, feature_dim) — extracted features for next level
            pos_3d:   (B, 3) — geometric position (for visualization)
        """
        # Geometric position
        pos_3d = self.to_3d(x)  # (B, 3)

        # Distance-based geometric signal
        diff = pos_3d.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, K, 3)
        d_sq = (diff ** 2).sum(dim=-1)  # (B, K)
        geo_signal = -d_sq / (2.0 * temperature ** 2 + 1e-8)  # (B, K)

        # Rich features
        features = self.feature_net(x)  # (B, feature_dim)

        # Combine for routing: features inform WHICH child, geometry structures HOW
        combined = torch.cat([features, pos_3d], dim=-1)  # (B, feature_dim + 3)
        logits = self.route_head(combined)  # (B, K)

        # Add geometric bias (the 3D structure matters)
        logits = logits + 0.5 * geo_signal

        if self.training:
            probs = F.gumbel_softmax(logits, tau=max(temperature, 0.1), hard=False)
        else:
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)

        return probs, features, pos_3d


class EnhancedBVHRouter(nn.Module):
    """
    Enhanced BVH Router that preserves information through hierarchy.

    Architecture:
        Level 1: 2048 → 4 domains     (extract 128-dim features)
        Level 2: 128  → 4 subdomains   (extract 128-dim features)
        Level 3: 128  → 4 concepts     (extract 128-dim features)
        Output:  concatenated features → 64-dim expert logits

    The 3D geometry at each level provides the STRUCTURE (which branch
    of the tree), while the feature vectors carry the INFORMATION needed
    to make accurate routing decisions.

    Total: 4 × 4 × 4 = 64 experts (matches OLMoE exactly)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        n_level1: int = 4,
        n_level2: int = 4,
        n_level3: int = 4,
        feature_dim: int = 128,
        temperature_init: float = 1.0,
    ):
        super().__init__()
        self.n_level1 = n_level1
        self.n_level2 = n_level2
        self.n_level3 = n_level3
        self.n_experts = n_level1 * n_level2 * n_level3
        self.feature_dim = feature_dim

        # Input projection (compress from 2048 to manageable size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )

        # Three hierarchical levels
        self.level1 = HierarchicalLevel(256, n_level1, feature_dim)
        self.level2 = HierarchicalLevel(feature_dim, n_level2, feature_dim)
        self.level3 = HierarchicalLevel(feature_dim, n_level3, feature_dim)

        # Final expert logit head: maps hierarchical routing to 64 expert probs
        # Uses: level features + routing decisions at each level
        head_input_dim = feature_dim + n_level1 + n_level2 + n_level3
        self.expert_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.n_experts),
        )

        # Temperature
        self.register_buffer('temperature', torch.tensor(temperature_init))

        # Expert usage tracking
        self.register_buffer('expert_counts', torch.zeros(self.n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 2048) — hidden states

        Returns:
            expert_probs: (B, 64) — probability over 64 experts
            expert_ids:   (B,) — argmax expert selection
        """
        T = self.temperature.item()

        # Input projection
        h = self.input_proj(x)  # (B, 256)

        # Level 1: domains
        p1, f1, pos1 = self.level1(h, T)  # p1: (B, 4), f1: (B, 128)

        # Level 2: subdomains (conditioned on level 1 features)
        p2, f2, pos2 = self.level2(f1, T)  # p2: (B, 4), f2: (B, 128)

        # Level 3: concepts (conditioned on level 2 features)
        p3, f3, pos3 = self.level3(f2, T)  # p3: (B, 4), f3: (B, 128)

        # Combine all routing info for final expert selection
        combined = torch.cat([f3, p1, p2, p3], dim=-1)  # (B, 128+4+4+4)
        logits = self.expert_head(combined)  # (B, 64)
        self._last_logits = logits

        if self.training:
            expert_probs = F.gumbel_softmax(
                logits, tau=max(T, 0.1), hard=False
            )
        else:
            expert_probs = F.softmax(logits / max(T, 0.1), dim=-1)

        expert_ids = expert_probs.argmax(dim=-1)

        # Track usage
        if self.training:
            for eid in expert_ids:
                self.expert_counts[eid] += 1

        return expert_probs, expert_ids

    def anneal_temperature(self, decay: float = 0.95):
        new_temp = max(0.1, self.temperature.item() * decay)
        self.temperature.fill_(new_temp)

    def reset_expert_counts(self):
        self.expert_counts.zero_()

    def load_balancing_loss(self) -> torch.Tensor:
        total = self.expert_counts.sum()
        if total < 1:
            return torch.tensor(0.0, device=self.expert_counts.device)
        usage = self.expert_counts / total
        target = 1.0 / self.n_experts
        return ((usage - target) ** 2).sum()


# ─────────────────────────────────────────────────────────────────
# MLP Baseline Router — sanity check (no BVH hierarchy)
# ─────────────────────────────────────────────────────────────────

class MLPBaselineRouter(nn.Module):
    """
    Simple MLP router as a sanity check / upper bound.

    If this can't match the linear gate with >90% top-8, the problem
    is in the data or training, not the BVH hierarchy. If it CAN,
    then the BVH hierarchy is the bottleneck.

    Architecture: 2048 → 512 → 256 → 64 (same capacity as EnhancedBVHRouter)
    Params: ~1.2M (similar to BVH router's 1.35M)
    """

    def __init__(self, input_dim: int = 2048, n_experts: int = 64,
                 temperature_init: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_level1 = 4  # compatibility stubs for training loop
        self.n_level2 = 4
        self.n_level3 = 4
        self.feature_dim = 256

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, n_experts),
        )

        self.register_buffer('temperature', torch.tensor(temperature_init))
        self.register_buffer('expert_counts', torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        self._last_logits = logits

        T = max(self.temperature.item(), 0.1)
        if self.training:
            expert_probs = F.gumbel_softmax(logits, tau=T, hard=False)
        else:
            expert_probs = F.softmax(logits / T, dim=-1)

        expert_ids = expert_probs.argmax(dim=-1)

        if self.training:
            self.expert_counts.scatter_add_(
                0, expert_ids,
                torch.ones_like(expert_ids, dtype=self.expert_counts.dtype),
            )

        self._last_expert_probs = expert_probs
        return expert_probs, expert_ids

    def get_last_logits(self) -> torch.Tensor:
        return self._last_logits

    def anneal_temperature(self, decay: float = 0.95):
        new_temp = max(0.1, self.temperature.item() * decay)
        self.temperature.fill_(new_temp)

    def reset_expert_counts(self):
        self.expert_counts.zero_()

    def load_balancing_loss(self) -> torch.Tensor:
        if not hasattr(self, '_last_expert_probs'):
            return torch.tensor(0.0, device=self.expert_counts.device)
        probs = self._last_expert_probs
        expert_mask = F.one_hot(probs.argmax(dim=-1), self.n_experts).float()
        f = expert_mask.mean(dim=0)
        P = probs.mean(dim=0)
        return self.n_experts * (f * P).sum()


# ─────────────────────────────────────────────────────────────────
# Dataset: hidden states from OLMoE + gate routing labels
# ─────────────────────────────────────────────────────────────────

class GateDistillationDataset(Dataset):
    """
    Generate hidden state samples and their OLMoE gate routing targets.

    Uses RMS-normalized random vectors to mimic post-LayerNorm distribution.
    """

    def __init__(
        self,
        olmoe_layer: OLMoELayer,
        n_samples: int = 100_000,
        hidden_size: int = 2048,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.n_samples = n_samples

        print(f"  Generating {n_samples:,} training samples...")
        t0 = time.time()

        all_hidden = []
        all_gate_logits = []
        all_topk_ids = []

        batch = 1024
        with torch.no_grad():
            for i in range(0, n_samples, batch):
                bs = min(batch, n_samples - i)
                h = torch.randn(bs, hidden_size, device=device, dtype=dtype)
                h = h / h.norm(dim=-1, keepdim=True) * math.sqrt(hidden_size)

                logits = olmoe_layer.gate_logits(h)
                _, topk = logits.topk(8, dim=-1)

                all_hidden.append(h.cpu().float())
                all_gate_logits.append(logits.cpu().float())
                all_topk_ids.append(topk.cpu())

        self.hidden_states = torch.cat(all_hidden, dim=0)
        self.gate_logits = torch.cat(all_gate_logits, dim=0)
        self.topk_ids = torch.cat(all_topk_ids, dim=0)

        # Pre-compute hard labels (top-1 expert from gate)
        self.top1_labels = self.topk_ids[:, 0]

        elapsed = time.time() - t0
        print(f"  Generated in {elapsed:.1f}s")
        print(f"  Hidden: {self.hidden_states.shape}")
        print(f"  Top-1 expert distribution: {self.top1_labels.unique().numel()} unique experts used")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.hidden_states[idx],
            self.gate_logits[idx],
            self.topk_ids[idx],
            self.top1_labels[idx],
        )


class RealHiddensDataset(Dataset):
    """
    Load pre-extracted real hidden states from extract_real_hiddens.py.

    The .pt file contains:
        hidden_states: (N, 2048) fp16
        gate_logits:   (N, 64) fp16  (actually softmax probabilities)
        topk_ids:      (N, 8) int64
    """

    def __init__(self, path: str, max_samples: int = None):
        super().__init__()
        print(f"  Loading real hidden states from {path}...")
        t0 = time.time()

        data = torch.load(path, map_location="cpu", weights_only=False)
        self.hidden_states = data["hidden_states"].float()
        self.gate_logits = data["gate_logits"].float()
        self.topk_ids = data["topk_ids"]

        if max_samples and len(self.hidden_states) > max_samples:
            idx = torch.randperm(len(self.hidden_states))[:max_samples]
            self.hidden_states = self.hidden_states[idx]
            self.gate_logits = self.gate_logits[idx]
            self.topk_ids = self.topk_ids[idx]

        self.top1_labels = self.topk_ids[:, 0]
        self.n_samples = len(self.hidden_states)

        elapsed = time.time() - t0
        print(f"  Loaded {self.n_samples:,} samples in {elapsed:.1f}s")
        print(f"  Hidden: {self.hidden_states.shape}")
        print(f"  Top-1 expert distribution: {self.top1_labels.unique().numel()} unique experts used")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.hidden_states[idx],
            self.gate_logits[idx],
            self.topk_ids[idx],
            self.top1_labels[idx],
        )


# ─────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha_soft: float = 0.7,
) -> torch.Tensor:
    """
    Standard knowledge distillation loss.

    Combines:
    - Soft targets: KL between teacher and student softmax (at temperature T)
    - Hard targets: CE on top-1 expert label

    NOTE: teacher_logits from extract_real_hiddens.py are actually PROBABILITIES
    (post-softmax), not raw logits. We convert them back to log-space before
    applying temperature scaling to avoid double-softmax.
    """
    # Convert teacher probs back to logits (undo the softmax in extract_real_hiddens.py)
    # log(softmax(x)) ∝ x (up to a constant), so log(probs) recovers the logit scale
    teacher_log = (teacher_logits + 1e-9).log()  # (B, 64) — now in logit space

    # Soft loss (KL on temperature-scaled softmax)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_log / temperature, dim=-1)
    l_soft = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    # Hard loss (CE on top-1)
    l_hard = F.cross_entropy(student_logits, hard_labels)

    return alpha_soft * l_soft + (1.0 - alpha_soft) * l_hard


def entropy_regularization(
    student_logits: torch.Tensor,
    n_experts: int = 64,
) -> torch.Tensor:
    """
    Entropy sweet-spot regularization (from train_dispersion.py PolysemicRouter).

    Penalizes both:
    - Too HIGH entropy (router is indecisive → poor routing)
    - Too LOW entropy (router collapses to single expert → kills top-8 diversity)

    Target entropy = log(n_experts) * 0.5 ≈ 2.08 for 64 experts.
    This is the sweet spot between "spread across many" and "focused on a few".

    This directly addresses the top-8 declining problem: without this,
    the router overfits to top-1 at the expense of top-8 diversity.
    """
    probs = F.softmax(student_logits, dim=-1)  # (B, n_experts)
    entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()  # scalar
    target_entropy = math.log(n_experts) * 0.5
    return (entropy - target_entropy).abs()


def topk_matching_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """
    Direct top-k matching loss: maximize student probability mass on
    teacher's top-k expert indices.

    For each token, we sum the student's softmax probability at the
    teacher's top-k positions. A perfect router would put ~100% of its
    probability mass there. This directly optimizes top-8 overlap.

    Loss = 1.0 - mean(sum of student probs at teacher's top-k)

    This is THE key missing piece: distillation_loss only optimizes
    full-distribution KL (soft) + top-1 CE (hard), but neither
    specifically targets the top-8 set that OLMoE actually uses.
    """
    student_probs = F.softmax(student_logits, dim=-1)  # (B, 64)
    # Gather student probs at teacher's top-k positions
    teacher_topk_probs = student_probs.gather(1, teacher_topk_ids[:, :k])  # (B, k)
    # Sum probability mass on teacher's top-k experts
    topk_mass = teacher_topk_probs.sum(dim=-1)  # (B,)
    # Loss: 1 - average probability mass on teacher's top-k
    return (1.0 - topk_mass).mean()


def compute_topk_accuracy(
    student_probs: torch.Tensor,
    gate_topk_ids: torch.Tensor,
    k: int = 8,
) -> Tuple[float, float]:
    """Compute top-k overlap and top-1 accuracy."""
    B = student_probs.shape[0]
    student_topk = student_probs.topk(k, dim=-1).indices
    student_top1 = student_probs.argmax(dim=-1)
    gate_top1 = gate_topk_ids[:, 0]

    # Top-k overlap (vectorized)
    # For each sample, count intersection of sets
    overlap = 0.0
    for b in range(B):
        s_set = set(student_topk[b].tolist())
        g_set = set(gate_topk_ids[b].tolist())
        overlap += len(s_set & g_set) / k

    topk_acc = overlap / B
    top1_acc = (student_top1 == gate_top1).float().mean().item()

    return topk_acc, top1_acc


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

def train_bvh_distillation(
    olmoe_layer: OLMoELayer,
    router: EnhancedBVHRouter,
    n_train: int = 500_000,
    n_val: int = 10_000,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_balance: float = 0.5,
    distill_temp: float = 4.0,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    real_data_path: str = None,
):
    """
    Train enhanced BVH router to match OLMoE gate via knowledge distillation.

    If real_data_path is provided, loads pre-extracted hidden states instead of
    generating synthetic random data. Real data gives dramatically better results
    (91.7% vs 74.4% top-8 accuracy).
    """
    print("\n" + "=" * 60)
    print("  Enhanced BVH Router Distillation from OLMoE Gate")
    print("=" * 60)

    dtype = torch.float16 if device == "cuda" else torch.float32

    if real_data_path is not None:
        # Use pre-extracted real hidden states
        print("\n[1/4] Loading REAL hidden states (pre-extracted)...")
        full_ds = RealHiddensDataset(real_data_path)
        # Split into train/val
        n_total = len(full_ds)
        n_val_actual = min(n_val, n_total // 10)
        n_train_actual = n_total - n_val_actual
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train_actual, n_val_actual]
        )
        print(f"\n[2/4] Split: {n_train_actual:,} train + {n_val_actual:,} val")
    else:
        # Generate synthetic data from OLMoE gate
        print("\n[1/4] Generating training data (synthetic)...")
        train_ds = GateDistillationDataset(
            olmoe_layer, n_samples=n_train, device=device, dtype=dtype
        )
        print("\n[2/4] Generating validation data (synthetic)...")
        val_ds = GateDistillationDataset(
            olmoe_layer, n_samples=n_val, device=device, dtype=dtype
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Setup
    router = router.to(device).float()
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    n_params = sum(p.numel() for p in router.parameters())
    print(f"\n[3/4] Enhanced BVH Router: {n_params:,} trainable params")
    print(f"  Hierarchy: {router.n_level1}x{router.n_level2}x{router.n_level3} = {router.n_experts} experts")
    print(f"  Feature dim: {router.feature_dim}")
    print(f"  Temperature: {router.temperature.item():.2f}")
    alpha_soft = 0.7
    weight_entropy = 0.01
    weight_topk = 0.0
    print(f"  Distillation temp: {distill_temp}, alpha_soft: {alpha_soft}")
    print(f"  Weights: balance={weight_balance}, entropy={weight_entropy}, topk={weight_topk}")
    print(f"  LR: {lr}, Epochs: {epochs}, Batch: {batch_size}")

    os.makedirs(save_dir, exist_ok=True)
    best_topk_acc = 0.0
    best_top1_acc = 0.0

    # ── Training ──────────────────────────────────────────────────
    print(f"\n[4/4] Training...")
    print(f"{'Ep':>3} {'Loss':>8} {'Soft':>8} {'Hard':>8} {'Bal':>8} {'Ent':>8} {'TkL':>8} "
          f"{'Top8%':>7} {'Top1%':>7} {'Temp':>6} {'LR':>10}")
    print("-" * 99)

    global_step = 0
    for epoch in range(epochs):
        router.train()
        router.reset_expert_counts()
        epoch_loss = 0.0
        epoch_soft = 0.0
        epoch_hard = 0.0
        epoch_bal = 0.0
        epoch_ent = 0.0
        epoch_topk = 0.0
        n_batches = 0

        for hidden, gate_logits_batch, topk_ids, top1_labels in train_loader:
            hidden = hidden.to(device)
            gate_logits_batch = gate_logits_batch.to(device)
            top1_labels = top1_labels.to(device)

            # Router forward
            expert_probs, expert_ids = router(hidden)

            # Get raw logits for distillation (before softmax/gumbel)
            # Re-run to get logits — use the expert_head output directly
            T = router.temperature.item()
            h = router.input_proj(hidden)
            p1, f1, _ = router.level1(h, T)
            p2, f2, _ = router.level2(f1, T)
            p3, f3, _ = router.level3(f2, T)
            combined = torch.cat([f3, p1, p2, p3], dim=-1)
            student_logits = router.expert_head(combined)

            # Distillation loss
            l_distill = distillation_loss(
                student_logits, gate_logits_batch, top1_labels,
                temperature=distill_temp, alpha_soft=0.7,
            )
            l_balance = router.load_balancing_loss()
            l_entropy = entropy_regularization(student_logits, router.n_experts)

            loss = l_distill + weight_balance * l_balance + weight_entropy * l_entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track components of distillation loss for reporting
            with torch.no_grad():
                student_soft = F.log_softmax(student_logits / distill_temp, dim=-1)
                teacher_soft = F.softmax(gate_logits_batch / distill_temp, dim=-1)
                l_soft = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (distill_temp ** 2)
                l_hard = F.cross_entropy(student_logits, top1_labels)

            epoch_loss += loss.item()
            epoch_soft += l_soft.item()
            epoch_hard += l_hard.item()
            epoch_bal += l_balance.item()
            epoch_ent += l_entropy.item()
            epoch_topk += 0.0  # topk loss not used in current config
            n_batches += 1
            global_step += 1

        epoch_loss /= n_batches
        epoch_soft /= n_batches
        epoch_hard /= n_batches
        epoch_bal /= n_batches
        epoch_ent /= n_batches
        epoch_topk /= n_batches

        # Validation
        router.eval()
        val_topk_acc = 0.0
        val_top1_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for hidden, gate_logits_batch, topk_ids, top1_labels in val_loader:
                hidden = hidden.to(device)
                topk_ids = topk_ids.to(device)

                expert_probs, expert_ids = router(hidden)
                tk_acc, t1_acc = compute_topk_accuracy(expert_probs, topk_ids)
                val_topk_acc += tk_acc
                val_top1_acc += t1_acc
                n_val_batches += 1

        val_topk_acc /= n_val_batches
        val_top1_acc /= n_val_batches

        # Temperature annealing (gentle)
        if epoch >= 5:
            router.anneal_temperature(decay=0.97)

        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch+1:>3} {epoch_loss:>8.4f} {epoch_soft:>8.4f} {epoch_hard:>8.4f} "
              f"{epoch_bal:>8.5f} {epoch_ent:>8.4f} {epoch_topk:>8.4f} {val_topk_acc*100:>6.1f}% {val_top1_acc*100:>6.1f}% "
              f"{router.temperature.item():>6.3f} {current_lr:>10.2e}")

        # Save best (prioritize top-8 overlap)
        if val_topk_acc > best_topk_acc:
            best_topk_acc = val_topk_acc
            best_top1_acc = val_top1_acc
            is_mlp = isinstance(router, MLPBaselineRouter)
            ckpt_name = "mlp_router_best.pt" if is_mlp else "bvh_router_best.pt"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            torch.save({
                "epoch": epoch + 1,
                "router_state_dict": router.state_dict(),
                "router_type": "mlp" if is_mlp else "bvh",
                "topk_accuracy": val_topk_acc,
                "top1_accuracy": val_top1_acc,
                "config": {
                    "input_dim": 2048,
                    "n_experts": router.n_experts,
                    "n_level1": router.n_level1,
                    "n_level2": router.n_level2,
                    "n_level3": router.n_level3,
                    "feature_dim": router.feature_dim,
                },
            }, ckpt_path)
            print(f"  -> NEW BEST (top-8: {val_topk_acc*100:.1f}%, top-1: {val_top1_acc*100:.1f}%)")

        # Expert usage
        if (epoch + 1) % 5 == 0 or epoch == 0:
            counts = router.expert_counts
            active = (counts > 0).sum().item()
            print(f"  -> Active experts: {active}/{router.n_experts}, "
                  f"max: {counts.max().item():.0f}, min: {counts.min().item():.0f}")

    # ── Final report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Best top-8 overlap: {best_topk_acc*100:.1f}%")
    print(f"  Best top-1 accuracy: {best_top1_acc*100:.1f}%")
    print(f"  Checkpoint: {save_dir}/bvh_router_best.pt")

    baseline_topk = 8.0 / 64.0 * 100
    baseline_top1 = 1.0 / 64.0 * 100
    print(f"\n  Random baseline: top-8={baseline_topk:.1f}%, top-1={baseline_top1:.1f}%")

    if best_topk_acc >= 0.8:
        print(f"  SUCCESS: BVH router achieves >=80% top-8 overlap!")
    elif best_topk_acc >= 0.5:
        print(f"  GOOD PROGRESS: {best_topk_acc*100:.0f}% top-8 overlap")
    elif best_topk_acc >= 0.25:
        print(f"  LEARNING: {best_topk_acc*100:.0f}% > random ({baseline_topk:.0f}%)")
    else:
        print(f"  NEEDS WORK: {best_topk_acc*100:.0f}% near random")

    return router, best_topk_acc


# ─────────────────────────────────────────────────────────────────
# Benchmark: BVH routing vs linear gate output quality
# ─────────────────────────────────────────────────────────────────

def benchmark_routing(
    olmoe_layer: OLMoELayer,
    router: EnhancedBVHRouter,
    n_samples: int = 5000,
    device: str = "cuda",
):
    """Compare BVH routing vs linear gate on expert output cosine similarity."""
    print("\n" + "=" * 60)
    print("  Routing Quality Benchmark")
    print("=" * 60)

    dtype = torch.float16 if device == "cuda" else torch.float32
    router.eval()

    cos_sims = []
    topk_overlaps = []
    top1_matches = 0

    with torch.no_grad():
        for i in range(0, n_samples, 256):
            bs = min(256, n_samples - i)
            h = torch.randn(bs, 2048, device=device, dtype=dtype)
            h = h / h.norm(dim=-1, keepdim=True) * math.sqrt(2048)

            # Gate routing (ground truth)
            gate_probs, gate_topk_ids = olmoe_layer.gate_topk(h)
            gate_out = olmoe_layer.forward_topk(h, gate_probs, gate_topk_ids)

            # BVH routing
            bvh_probs, bvh_ids = router(h.float())
            bvh_topk = bvh_probs.topk(8, dim=-1)
            bvh_probs_norm = F.softmax(bvh_topk.values, dim=-1).to(dtype)
            bvh_out = olmoe_layer.forward_topk(h, bvh_probs_norm, bvh_topk.indices)

            # Cosine similarity
            cos = F.cosine_similarity(gate_out.float(), bvh_out.float(), dim=-1)
            cos_sims.extend(cos.tolist())

            # Top-k overlap
            for b in range(bs):
                g_set = set(gate_topk_ids[b].tolist())
                b_set = set(bvh_topk.indices[b].tolist())
                topk_overlaps.append(len(g_set & b_set) / 8)
                if gate_topk_ids[b, 0] == bvh_topk.indices[b, 0]:
                    top1_matches += 1

    avg_cos = sum(cos_sims) / len(cos_sims)
    avg_topk = sum(topk_overlaps) / len(topk_overlaps)
    top1_acc = top1_matches / n_samples

    print(f"\n  Samples: {n_samples:,}")
    print(f"  Output cosine similarity: {avg_cos:.4f}")
    print(f"  Top-8 overlap: {avg_topk*100:.1f}%")
    print(f"  Top-1 accuracy: {top1_acc*100:.1f}%")

    if avg_cos > 0.95:
        print(f"\n  EXCELLENT: Expert outputs nearly identical")
    elif avg_cos > 0.85:
        print(f"\n  GOOD: Expert outputs similar")
    elif avg_cos > 0.5:
        print(f"\n  MODERATE: Partial agreement")
    else:
        print(f"\n  DIVERGENT: Expert outputs differ significantly")

    return {"cos_sim": avg_cos, "topk_overlap": avg_topk, "top1_acc": top1_acc}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced BVH Router distillation from OLMoE")
    parser.add_argument("--model-dir", type=str,
                        default="/mnt/j/Proyectos/models/olmoe-1b-7b")
    parser.add_argument("--layer", type=int, default=8,
                        help="OLMoE layer to extract (0-15)")
    parser.add_argument("--real-data", type=str, default=None,
                        help="Path to pre-extracted real hidden states (.pt). "
                             "If provided, skips loading full OLMoE model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=500_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints/olmoe_distill")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run output quality benchmark after training")
    parser.add_argument("--distill-temp", type=float, default=4.0,
                        help="Distillation temperature")
    parser.add_argument("--mlp-baseline", action="store_true",
                        help="Use MLP baseline instead of BVH (sanity check)")
    parser.add_argument("--no-upcycle", action="store_true",
                        help="Skip sparse upcycling initialization")
    args = parser.parse_args()

    print("=" * 60)
    print("  LiquidBit FASE A v2.1 -- Enhanced BVH Distillation")
    print("=" * 60)

    olmoe_layer = None

    # Step 1: Load OLMoE layer (only needed if no --real-data or for upcycling)
    need_model = (args.real_data is None) or (not args.no_upcycle and not args.mlp_baseline)
    if need_model:
        print(f"\n[Step 1] Loading OLMoE layer {args.layer}...")
        olmoe_layer = load_olmoe_layer(
            args.model_dir, args.layer,
            device=args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )
    else:
        print(f"\n[Step 1] Skipping model load (--real-data provided + --no-upcycle)")

    # Step 2: Create router
    if args.mlp_baseline:
        print(f"\n[Step 2] Creating MLP Baseline Router (sanity check)...")
        router = MLPBaselineRouter(input_dim=2048, n_experts=64, temperature_init=1.0)
        n_params = sum(p.numel() for p in router.parameters())
        print(f"  MLP params: {n_params:,}")
    else:
        print(f"\n[Step 2] Creating Enhanced BVH Router (4x4x4 = 64 experts)...")
        router = EnhancedBVHRouter(
            input_dim=2048,
            n_level1=4,
            n_level2=4,
            n_level3=4,
            feature_dim=128,
            temperature_init=1.0,
        )

        # Step 2b: Sparse Upcycling — initialize router from gate weights
        if not args.no_upcycle and olmoe_layer is not None:
            print(f"\n[Step 2b] Sparse Upcycling — initializing from gate weights...")
            gate_weight = olmoe_layer.gate.weight.data.clone()  # [64, 2048]
            initialize_router_from_gate(router, gate_weight, verbose=True)
        else:
            print(f"\n[Step 2b] Skipping Sparse Upcycling")

    # Free OLMoE layer if using real data (no longer needed, saves ~7GB VRAM)
    if args.real_data is not None and olmoe_layer is not None:
        olmoe_layer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n  Freed OLMoE layer from memory (using pre-extracted real data)")

    # Step 3: Train
    router, best_acc = train_bvh_distillation(
        olmoe_layer=None,  # not needed when using real data
        router=router,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        distill_temp=args.distill_temp,
        device=args.device,
        save_dir=args.save_dir,
        real_data_path=args.real_data,
    )

    # Step 4: Benchmark (only if model is loaded)
    if olmoe_layer is not None and (args.benchmark or best_acc >= 0.25):
        benchmark_routing(
            olmoe_layer=olmoe_layer,
            router=router,
            n_samples=5000,
            device=args.device,
        )


if __name__ == "__main__":
    main()
