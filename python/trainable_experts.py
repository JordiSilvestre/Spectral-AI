#!/usr/bin/env python3
"""
trainable_experts.py -- LiquidBit FASE A: Trainable MLP Experts

Lightweight MLP experts that CAN specialize during training.
Unlike the frozen ternary clones from BitNet, these start random
and learn to handle different input patterns via the BVH router.

Architecture per expert:
    hidden_dim → 4*hidden_dim → hidden_dim  (SwiGLU activation)
    ~4M params each @ hidden_dim=512

Total for 16 experts: ~64M params (fits easily in 16 GB VRAM).

Copyright (c) 2026 LiquidBit Studio — Apache 2.0
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainableExpertConfig:
    """Immutable config for the expert pool."""
    n_experts: int = 16
    hidden_dim: int = 512       # Input/output dimension
    intermediate_dim: int = 2048  # 4x expansion (SwiGLU)
    dropout: float = 0.0
    init_std: float = 0.02


# ─────────────────────────────────────────────────────────────────
# Single Expert — SwiGLU MLP
# ─────────────────────────────────────────────────────────────────

class SwiGLUExpert(nn.Module):
    """
    Single expert MLP with SwiGLU activation (LLaMA style).

    SwiGLU: out = (W_gate(x) * silu(W_up(x))) @ W_down
    More expressive than simple ReLU MLP, used in all modern LLMs.

    Params per expert:
        W_gate: hidden_dim × intermediate_dim
        W_up:   hidden_dim × intermediate_dim
        W_down: intermediate_dim × hidden_dim
        Total: 3 × hidden_dim × intermediate_dim
        @512×2048: 3 × 512 × 2048 = 3.1M params
    """

    def __init__(self, config: TrainableExpertConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        nn.init.normal_(self.gate_proj.weight, std=std)
        nn.init.normal_(self.up_proj.weight, std=std)
        # Down proj gets smaller init to start with small expert contribution
        nn.init.normal_(self.down_proj.weight, std=std * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, hidden_dim) → (B, S, hidden_dim)"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ─────────────────────────────────────────────────────────────────
# Expert Pool — All experts + output projection
# ─────────────────────────────────────────────────────────────────

class TrainableExpertPool(nn.Module):
    """
    Pool of trainable MLP experts + shared output projection.

    This is the FASE A replacement for the frozen ternary experts.
    All experts are trainable from random init, allowing specialization
    through the BVH router's gradient signal.

    Pipeline:
        hidden_states → expert_i(hidden) → output_proj → logits
    """

    def __init__(
        self,
        config: TrainableExpertConfig,
        vocab_size: int,
    ):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.hidden_dim = config.hidden_dim

        # Create expert pool
        self.experts = nn.ModuleList([
            SwiGLUExpert(config) for _ in range(config.n_experts)
        ])

        # Shared output projection: expert hidden → vocab logits
        self.output_proj = nn.Linear(config.hidden_dim, vocab_size, bias=False)
        nn.init.normal_(self.output_proj.weight, std=config.init_std)

        # Stats tracking
        self._expert_usage = torch.zeros(config.n_experts, dtype=torch.long)

    def forward_expert(
        self,
        expert_id: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward through a single expert.

        Args:
            expert_id: Which expert to use
            hidden_states: (B, S, hidden_dim) — input hidden states

        Returns:
            (B, S, hidden_dim) — expert output (before output_proj)
        """
        return self.experts[expert_id](hidden_states)

    def forward_topk(
        self,
        hidden_states: torch.Tensor,
        expert_probs: torch.Tensor,
        top_k: int = 2,
    ) -> torch.Tensor:
        """
        Forward through top-k experts, weighted by routing probabilities.

        Args:
            hidden_states: (B, S, hidden_dim)
            expert_probs: (B, n_experts) — routing probabilities
            top_k: number of experts to activate

        Returns:
            (B, S, vocab_size) — blended expert logits
        """
        B, S, H = hidden_states.shape

        # Select top-k experts per sample
        topk_probs, topk_ids = expert_probs.topk(top_k, dim=-1)  # (B, k)
        # Renormalize top-k probs
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Track usage
        for eid in topk_ids.reshape(-1).tolist():
            self._expert_usage[eid] += 1

        # Forward through each selected expert
        combined_hidden = torch.zeros(B, S, H, device=hidden_states.device, dtype=hidden_states.dtype)

        for k_idx in range(top_k):
            for b in range(B):
                eid = topk_ids[b, k_idx].item()
                weight = topk_probs[b, k_idx]
                expert_out = self.experts[eid](hidden_states[b:b+1])  # (1, S, H)
                combined_hidden[b:b+1] += weight * expert_out

        # Shared output projection
        return self.output_proj(combined_hidden)

    def forward_topk_batched(
        self,
        hidden_states: torch.Tensor,
        expert_probs: torch.Tensor,
        top_k: int = 2,
    ) -> torch.Tensor:
        """
        Optimized batched version — groups samples by expert to avoid loops.

        Args:
            hidden_states: (B, S, hidden_dim)
            expert_probs: (B, n_experts)
            top_k: number of experts to activate

        Returns:
            (B, S, vocab_size) — blended expert logits
        """
        B, S, H = hidden_states.shape

        topk_probs, topk_ids = expert_probs.topk(top_k, dim=-1)  # (B, k)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Track usage
        for eid in topk_ids.reshape(-1).tolist():
            self._expert_usage[eid] += 1

        combined_hidden = torch.zeros(B, S, H, device=hidden_states.device, dtype=hidden_states.dtype)

        # Group by expert for efficient batching
        for k_idx in range(top_k):
            eids = topk_ids[:, k_idx]  # (B,)
            weights = topk_probs[:, k_idx]  # (B,)

            unique_eids = eids.unique()
            for eid in unique_eids:
                mask = (eids == eid)  # (B,)
                if not mask.any():
                    continue
                batch_hidden = hidden_states[mask]  # (n, S, H)
                expert_out = self.experts[eid.item()](batch_hidden)  # (n, S, H)
                batch_weights = weights[mask].unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)
                combined_hidden[mask] += batch_weights * expert_out

        return self.output_proj(combined_hidden)

    def get_expert_usage(self) -> Dict[int, int]:
        """Return expert usage counts."""
        return {i: self._expert_usage[i].item() for i in range(self.n_experts)}

    def reset_usage(self):
        """Reset usage counters."""
        self._expert_usage.zero_()

    def param_count(self) -> Dict[str, int]:
        """Return parameter counts by component."""
        expert_params = sum(p.numel() for e in self.experts for p in e.parameters())
        proj_params = sum(p.numel() for p in self.output_proj.parameters())
        return {
            "experts_total": expert_params,
            "per_expert": expert_params // self.n_experts,
            "output_proj": proj_params,
            "total": expert_params + proj_params,
        }


# ─────────────────────────────────────────────────────────────────
# Lightweight LiquidBit MoE — No backbone, pure MoE
# ─────────────────────────────────────────────────────────────────

class LiquidBitMoE(nn.Module):
    """
    Complete lightweight MoE model for FASE A training.

    Architecture:
        token_ids → embedding → LayerNorm → BVH Router → top-k experts → logits

    This model does NOT use a frozen backbone. Everything is trainable:
    - Token embeddings (vocab_size × hidden_dim)
    - BVH Router (3-level hierarchical)
    - Expert pool (n_experts × SwiGLU MLPs)
    - Output projection (hidden_dim → vocab_size, tied with embeddings)

    Total params for default config:
        Embeddings: 128256 × 512 = 65.7M (tied with output)
        Experts: 16 × 3.1M = 49.7M
        Router: ~750K
        Total: ~116M trainable params
    """

    def __init__(
        self,
        vocab_size: int,
        expert_config: TrainableExpertConfig,
        router_embed_dim: int = 256,
        seq_len: int = 512,
        top_k: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = expert_config.hidden_dim
        self.top_k = top_k
        self.seq_len = seq_len

        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, expert_config.hidden_dim)
        self.pos_embeddings = nn.Embedding(seq_len, expert_config.hidden_dim)
        self.ln_in = nn.LayerNorm(expert_config.hidden_dim)
        self.drop = nn.Dropout(expert_config.dropout)

        # Hidden → router space projection
        self.hidden_proj = nn.Linear(expert_config.hidden_dim, router_embed_dim)

        # Expert pool
        self.expert_pool = TrainableExpertPool(expert_config, vocab_size)

        # Tie output weights with embeddings
        self.expert_pool.output_proj.weight = self.embeddings.weight

        # Blend gate: learn when to use experts vs embeddings-only
        self.blend_gate = nn.Linear(expert_config.hidden_dim, 1, bias=True)
        nn.init.constant_(self.blend_gate.bias, -2.0)  # Start mostly embedding-based

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)
        nn.init.normal_(self.hidden_proj.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        router: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Full forward pass.

        Args:
            input_ids: (B, S) token ids
            router: BVH Router module (passed separately for flexibility)

        Returns:
            logits: (B, S, vocab_size)
            expert_probs: (B, n_experts)
            info: dict with alpha, expert_ids, etc.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0)
        hidden = self.drop(self.ln_in(
            self.embeddings(input_ids) + self.pos_embeddings(pos)
        ))  # (B, S, H)

        # Router: sequence-level routing decision
        prompt_emb = hidden.mean(dim=1).float()  # (B, H)
        router_input = self.hidden_proj(prompt_emb)  # (B, router_dim)
        route_result = router(router_input, hard=not self.training)

        # Expert forward (top-k)
        expert_logits = self.expert_pool.forward_topk_batched(
            hidden, route_result.expert_probs, top_k=self.top_k
        )  # (B, S, vocab)

        # Simple embedding-based logits (baseline)
        embed_logits = self.expert_pool.output_proj(hidden)  # (B, S, vocab)

        # Blend
        alpha = torch.sigmoid(self.blend_gate(hidden))  # (B, S, 1)
        logits = (1.0 - alpha) * embed_logits + alpha * expert_logits

        info = {
            "alpha": alpha.mean().item(),
            "expert_probs": route_result.expert_probs,
            "expert_id": route_result.expert_id,
            "confidence": route_result.confidence,
        }

        return logits, route_result.expert_probs, info

    def param_summary(self) -> str:
        """Human-readable parameter summary."""
        emb = self.embeddings.weight.numel()
        pos = self.pos_embeddings.weight.numel()
        expert_info = self.expert_pool.param_count()
        router_proj = sum(p.numel() for p in self.hidden_proj.parameters())
        blend = sum(p.numel() for p in self.blend_gate.parameters())
        total = sum(p.numel() for p in self.parameters())

        lines = [
            f"  Embeddings:   {emb:>12,} ({emb * 4 / 1e6:.1f} MB)",
            f"  Pos embed:    {pos:>12,} ({pos * 4 / 1e6:.1f} MB)",
            f"  Experts:      {expert_info['experts_total']:>12,} "
            f"({expert_info['per_expert']:,}/expert × {self.expert_pool.n_experts})",
            f"  Output proj:  (tied with embeddings)",
            f"  Router proj:  {router_proj:>12,}",
            f"  Blend gate:   {blend:>12,}",
            f"  ─────────────────────────────",
            f"  TOTAL:        {total:>12,} ({total * 4 / 1e6:.1f} MB FP32)",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bvh_router import RouterConfig
    from bvh_router_bridge import HybridBVHRouter

    print("=" * 60)
    print("  LiquidBit FASE A — Trainable Expert Pool Test")
    print("=" * 60)

    config = TrainableExpertConfig(
        n_experts=16,
        hidden_dim=512,
        intermediate_dim=2048,
    )

    model = LiquidBitMoE(
        vocab_size=128256,  # BitNet vocab
        expert_config=config,
        router_embed_dim=256,
    )

    print(f"\nModel Architecture:")
    print(model.param_summary())

    # Create router
    n_l1, n_l2, n_l3 = 2, 2, 4  # 2×2×4 = 16 experts
    router_cfg = RouterConfig(
        embed_dim=256,
        n_level1=n_l1,
        n_level2=n_l2,
        n_level3=n_l3,
        n_experts=16,
    )
    router = HybridBVHRouter(router_cfg)

    # Test forward
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    router = router.to(device)

    x = torch.randint(0, 128256, (2, 64), device=device)
    model.train()
    logits, expert_probs, info = model(x, router.pytorch_router)

    print(f"\nForward pass:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Alpha:  {info['alpha']:.4f}")
    print(f"  Expert: {info['expert_id'].tolist()}")

    # Test backward
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
    loss.backward()
    print(f"  Loss:   {loss.item():.4f}")

    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Grads:  {'✅ all flowing' if grad_ok else '❌ missing grads'}")

    print(f"\n  Expert usage: {model.expert_pool.get_expert_usage()}")
    print("\n✅ FASE A expert pool ready for training")
