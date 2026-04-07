#!/usr/bin/env python3
"""
olmoe_extract.py -- Extract expert FFNs + gate from OLMoE-1B-7B

Loads a single layer's 64 SwiGLU experts and linear gate from the
OLMoE model (safetensors), wraps them as frozen nn.Modules ready
for BVH router distillation.

OLMoE architecture (per layer):
    - 64 experts, each: gate_proj [1024, 2048], up_proj [1024, 2048], down_proj [2048, 1024]
    - Linear gate: [64, 2048]
    - Activation: SiLU (SwiGLU)
    - Top-8 routing

Usage:
    layer = load_olmoe_layer(model_dir, layer_idx=8)
    expert_out = layer.forward_expert(expert_id=5, hidden_states)
    gate_logits = layer.gate(hidden_states)

Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors import safe_open
except ImportError:
    raise ImportError("pip install safetensors")


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OLMoELayerConfig:
    """Immutable config matching OLMoE-1B-7B architecture."""
    hidden_size: int = 2048
    intermediate_size: int = 1024
    n_experts: int = 64
    top_k: int = 8
    vocab_size: int = 50304


# ─────────────────────────────────────────────────────────────────
# Frozen SwiGLU Expert (loaded from OLMoE weights)
# ─────────────────────────────────────────────────────────────────

class FrozenSwiGLUExpert(nn.Module):
    """Single OLMoE expert — frozen, no gradients."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., hidden_size) -> (..., hidden_size)"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────────────────────────
# OLMoE Layer — 64 frozen experts + linear gate
# ─────────────────────────────────────────────────────────────────

class OLMoELayer(nn.Module):
    """
    One layer of OLMoE: 64 frozen SwiGLU experts + linear gate.

    The gate produces logits over 64 experts. Top-8 are selected,
    their outputs weighted by softmax(top-8 logits), and summed.
    """

    def __init__(self, config: OLMoELayerConfig):
        super().__init__()
        self.config = config

        # 64 frozen experts
        self.experts = nn.ModuleList([
            FrozenSwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.n_experts)
        ])

        # Linear gate (frozen — we use this as the distillation target)
        self.gate = nn.Linear(config.hidden_size, config.n_experts, bias=False)
        self.gate.weight.requires_grad = False

    def gate_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (..., hidden_size) -> (..., n_experts)"""
        return self.gate(hidden_states)

    def gate_topk(
        self, hidden_states: torch.Tensor, k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (topk_probs, topk_ids) for the gate's top-k selection.

        topk_probs: (..., k) — renormalized softmax probs
        topk_ids:   (..., k) — expert indices
        """
        k = k or self.config.top_k
        logits = self.gate_logits(hidden_states)
        topk_logits, topk_ids = logits.topk(k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        return topk_probs, topk_ids

    def forward_expert(
        self, expert_id: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward through a single expert."""
        return self.experts[expert_id](hidden_states)

    def forward_topk(
        self,
        hidden_states: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward through top-k experts, weighted by probs.

        hidden_states: (B, hidden_size)
        topk_probs:    (B, k)
        topk_ids:      (B, k)

        Returns: (B, hidden_size)
        """
        B = hidden_states.shape[0]
        k = topk_ids.shape[-1]
        out = torch.zeros_like(hidden_states)

        # Group by expert for efficiency
        for ki in range(k):
            eids = topk_ids[:, ki]  # (B,)
            weights = topk_probs[:, ki]  # (B,)

            for eid in eids.unique():
                mask = (eids == eid)
                if not mask.any():
                    continue
                expert_out = self.experts[eid.item()](hidden_states[mask])
                out[mask] += weights[mask].unsqueeze(-1) * expert_out

        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Full forward: gate -> top-k -> weighted expert outputs.

        hidden_states: (B, hidden_size) -> (B, hidden_size)
        """
        topk_probs, topk_ids = self.gate_topk(hidden_states)
        return self.forward_topk(hidden_states, topk_probs, topk_ids)


# ─────────────────────────────────────────────────────────────────
# Loading from safetensors
# ─────────────────────────────────────────────────────────────────

def _find_safetensors_files(model_dir: str) -> List[str]:
    """Find all safetensors files and their weight map."""
    model_dir = Path(model_dir)

    # Check for index file
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    # Single file
    single = model_dir / "model.safetensors"
    if single.exists():
        return {"*": "model.safetensors"}

    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def load_olmoe_layer(
    model_dir: str,
    layer_idx: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> OLMoELayer:
    """
    Load a single OLMoE layer's experts + gate from safetensors.

    Args:
        model_dir: Path to OLMoE model directory
        layer_idx: Which layer to extract (0-15)
        device: Target device
        dtype: Weight dtype (float16 saves memory)

    Returns:
        OLMoELayer with frozen weights loaded
    """
    model_dir = Path(model_dir)
    config = OLMoELayerConfig()
    layer = OLMoELayer(config)

    weight_map = _find_safetensors_files(model_dir)

    # Collect which files we need
    prefix = f"model.layers.{layer_idx}.mlp."
    needed_keys = {}

    # Gate weight
    gate_key = f"{prefix}gate.weight"
    needed_keys[gate_key] = "gate"

    # Expert weights
    for e in range(config.n_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}experts.{e}.{proj}.weight"
            needed_keys[key] = (e, proj)

    # Group by file
    file_to_keys: Dict[str, List[str]] = {}
    for key in needed_keys:
        if key in weight_map:
            fname = weight_map[key]
        else:
            # Try wildcard
            fname = weight_map.get("*", None)
            if fname is None:
                raise KeyError(f"Weight key {key} not found in weight map")
        file_to_keys.setdefault(fname, []).append(key)

    # Load weights
    loaded = 0
    for fname, keys in file_to_keys.items():
        fpath = model_dir / fname
        print(f"  Loading {len(keys)} tensors from {fname}...")
        with safe_open(str(fpath), framework="pt", device=device) as f:
            for key in keys:
                tensor = f.get_tensor(key).to(dtype=dtype)
                target = needed_keys[key]

                if target == "gate":
                    layer.gate.weight.data.copy_(tensor)
                else:
                    e_idx, proj_name = target
                    expert = layer.experts[e_idx]
                    getattr(expert, proj_name).weight.data.copy_(tensor)
                loaded += 1

    print(f"  Loaded {loaded}/{len(needed_keys)} tensors for layer {layer_idx}")

    # Move to device and freeze
    layer = layer.to(device=device, dtype=dtype)
    layer.eval()
    for p in layer.parameters():
        p.requires_grad = False

    return layer


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OLMOE_MODEL_DIR", "./olmoe-1b-7b")
    layer_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    print("=" * 60)
    print(f"  OLMoE Expert Extraction — Layer {layer_idx}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Model: {model_dir}")
    print(f"Loading layer {layer_idx}...")

    layer = load_olmoe_layer(model_dir, layer_idx, device=device, dtype=torch.float16)

    # Stats
    n_params = sum(p.numel() for p in layer.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in layer.parameters()) / 1e6
    print(f"\nLayer stats:")
    print(f"  Parameters: {n_params:,}")
    print(f"  Memory: {mem_mb:.1f} MB")
    print(f"  Experts: {len(layer.experts)}")
    print(f"  Gate shape: {layer.gate.weight.shape}")

    # Test forward
    x = torch.randn(4, 2048, device=device, dtype=torch.float16)

    # Gate logits
    logits = layer.gate_logits(x)
    print(f"\nGate logits: {logits.shape}")
    topk_probs, topk_ids = layer.gate_topk(x)
    print(f"Top-8 experts: {topk_ids[0].tolist()}")
    print(f"Top-8 probs:   {topk_probs[0].tolist()}")

    # Expert forward
    out = layer.forward(x)
    print(f"\nFull MoE forward: {x.shape} -> {out.shape}")
    print(f"Output norm: {out.norm(dim=-1).mean().item():.4f}")

    # Single expert test
    e_out = layer.forward_expert(0, x)
    print(f"Expert 0 output norm: {e_out.norm(dim=-1).mean().item():.4f}")

    print("\n[OK] OLMoE layer extraction complete")
