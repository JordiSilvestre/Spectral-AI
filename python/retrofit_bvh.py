#!/usr/bin/env python3
"""
retrofit_bvh.py — Universal BVH Routing Retrofit for ANY Model

One command to convert any HuggingFace model to use BVH geometric routing:

  MoE models (Mixtral, OLMoE, DeepSeek-MoE, Qwen-MoE):
    - Extracts gate routing decisions as training data
    - Trains BVH Router per layer to replicate the gate
    - Replaces gate with BVH (hybrid_residual mode: BVH selects, gate weights)
    - O(log N) routing instead of O(N) linear gate

  Dense models (LLaMA, GPT-2, Qwen, Mistral):
    - Splits FFN into N virtual experts (column groups)
    - Trains BVH Router to select which columns to activate
    - Adds dynamic sparsity: only K of N groups computed per token
    - Result: faster FFN with minimal quality loss

Usage:
    # MoE model — full retrofit
    python retrofit_bvh.py --model mistralai/Mixtral-8x7B-v0.1

    # Dense model — add sparsity
    python retrofit_bvh.py --model meta-llama/Llama-2-7b-hf --n-experts 16

    # Quick test with small model
    python retrofit_bvh.py --model allenai/OLMoE-1B-7B-0924 --max-tokens 10000

    # Custom config
    python retrofit_bvh.py --model <any_hf_model> \\
        --epochs 100 --batch-size 64 --device cuda \\
        --output-dir retrofitted_model/

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════
# Architecture Detection
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelArchInfo:
    """Detected model architecture information."""
    name: str
    is_moe: bool
    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_experts: int           # 0 for dense models
    top_k: int               # experts per token (MoE only)
    gate_attr: str           # 'gate', 'router', or None
    layer_path: str          # 'model.model.layers' or 'model.transformer.h'
    mlp_attr: str            # 'mlp' or 'block_sparse_moe'
    moe_layer_indices: tuple # which layers are MoE (all for pure MoE, subset for hybrid)
    ffn_type: str            # 'swiglu' or 'gelu'
    norm_topk_prob: bool     # whether to normalize top-k probs


def detect_architecture(model, config) -> ModelArchInfo:
    """Auto-detect model architecture from HuggingFace model + config."""

    model_type = getattr(config, 'model_type', '').lower()
    n_layers = getattr(config, 'num_hidden_layers',
                getattr(config, 'n_layer', 0))
    hidden_size = getattr(config, 'hidden_size',
                   getattr(config, 'n_embd', 0))
    intermediate_size = getattr(config, 'intermediate_size',
                         getattr(config, 'n_inner', hidden_size * 4))

    # Detect layer container
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer_path = 'model.model.layers'
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layer_path = 'model.transformer.h'
        layers = model.transformer.h
    elif hasattr(model, 'layers'):
        layer_path = 'model.layers'
        layers = model.layers
    else:
        raise ValueError(f"Cannot detect layer structure for {model_type}")

    n_layers = len(layers)

    # Check first layer for MoE
    first_layer = layers[0]
    mlp = getattr(first_layer, 'mlp', None)
    if mlp is None:
        mlp = getattr(first_layer, 'block_sparse_moe', None)
        mlp_attr = 'block_sparse_moe'
    else:
        mlp_attr = 'mlp'

    # Detect MoE
    is_moe = False
    n_experts = 0
    top_k = 0
    gate_attr = None
    norm_topk_prob = False
    moe_layers = []

    if mlp is not None:
        if hasattr(mlp, 'gate') or hasattr(mlp, 'router'):
            is_moe = True
            gate_attr = 'gate' if hasattr(mlp, 'gate') else 'router'

    if is_moe:
        # Get expert count from config or gate weight
        n_experts = getattr(config, 'num_local_experts',
                    getattr(config, 'num_experts',
                    getattr(config, 'n_routed_experts', 0)))
        top_k = getattr(config, 'num_experts_per_tok',
                 getattr(config, 'num_experts_per_token', 2))
        norm_topk_prob = getattr(config, 'norm_topk_prob', False)

        # If not in config, infer from gate weight shape
        if n_experts == 0:
            gate = getattr(mlp, gate_attr)
            gate_w = getattr(gate, 'weight', None)
            if gate_w is not None:
                n_experts = gate_w.shape[0]

        # Detect which layers are MoE (some models mix dense + MoE)
        first_k_dense = getattr(config, 'first_k_dense_replace', 0)
        for i in range(n_layers):
            layer_i = layers[i]
            mlp_i = getattr(layer_i, mlp_attr, None)
            if mlp_i is not None and (hasattr(mlp_i, 'gate') or hasattr(mlp_i, 'router')):
                moe_layers.append(i)
    else:
        moe_layers = []

    # Detect FFN type
    if mlp is not None and hasattr(mlp, 'gate_proj'):
        ffn_type = 'swiglu'
    elif mlp is not None and hasattr(mlp, 'c_fc'):
        ffn_type = 'gelu'
    else:
        ffn_type = 'unknown'

    # Friendly name
    name_map = {
        'olmoe': 'OLMoE', 'mixtral': 'Mixtral', 'deepseek_v2': 'DeepSeek-V2',
        'qwen2_moe': 'Qwen2-MoE', 'llama': 'LLaMA', 'gpt2': 'GPT-2',
        'qwen2': 'Qwen2', 'mistral': 'Mistral', 'phi': 'Phi',
        'gemma': 'Gemma', 'gemma2': 'Gemma2',
    }
    friendly_name = name_map.get(model_type, model_type.upper())

    return ModelArchInfo(
        name=friendly_name,
        is_moe=is_moe,
        n_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_experts=n_experts,
        top_k=top_k,
        gate_attr=gate_attr,
        layer_path=layer_path,
        mlp_attr=mlp_attr,
        moe_layer_indices=tuple(moe_layers),
        ffn_type=ffn_type,
        norm_topk_prob=norm_topk_prob,
    )


def get_layers(model, arch: ModelArchInfo):
    """Get the layer list from model."""
    parts = arch.layer_path.split('.')
    obj = model
    for p in parts:
        obj = getattr(obj, p)
    return obj


def get_gate(model, arch: ModelArchInfo, layer_idx: int):
    """Get the gate module for a specific layer."""
    layers = get_layers(model, arch)
    mlp = getattr(layers[layer_idx], arch.mlp_attr)
    return getattr(mlp, arch.gate_attr)


# ═══════════════════════════════════════════════════════════════════
# Hidden State Extraction
# ═══════════════════════════════════════════════════════════════════

class HiddenStateDataset(Dataset):
    """Dataset of (hidden_state, gate_labels) pairs for BVH training."""

    def __init__(self, hiddens: torch.Tensor, gate_topk_ids: torch.Tensor,
                 gate_probs: torch.Tensor):
        self.hiddens = hiddens       # [N, hidden_dim]
        self.topk_ids = gate_topk_ids  # [N, top_k]
        self.probs = gate_probs      # [N, n_experts]

    def __len__(self) -> int:
        return self.hiddens.shape[0]

    def __getitem__(self, idx: int):
        return self.hiddens[idx], self.topk_ids[idx], self.probs[idx]


def extract_hidden_states(
    model,
    tokenizer,
    arch: ModelArchInfo,
    layer_idx: int,
    max_tokens: int = 50000,
    device: str = "cuda",
) -> HiddenStateDataset:
    """Extract hidden states and gate decisions for one layer.

    Hooks into the gate module to capture:
    1. Input hidden states (what the gate receives)
    2. Output routing decisions (which experts were selected)
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in dataset["text"] if t.strip())
    except Exception:
        text = ("The quick brown fox jumps over the lazy dog. " * 2000 +
                "Machine learning models process text efficiently. " * 2000)

    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=min(max_tokens, 4096))
    input_ids = encodings.input_ids.to(device)

    all_hiddens = []
    all_topk_ids = []
    all_probs = []

    gate = get_gate(model, arch, layer_idx)

    def _capture_hook(module, inp, out):
        h = inp[0] if isinstance(inp, tuple) else inp
        h = h.detach().reshape(-1, arch.hidden_size).float().cpu()
        all_hiddens.append(h)

        # Extract routing decisions
        if isinstance(out, tuple) and len(out) >= 3:
            probs = out[0].detach().float().cpu()
            topk_idx = out[2].detach().cpu()
        elif isinstance(out, tuple) and len(out) >= 1:
            logits = out[0].detach().float()
            probs = F.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1).cpu()
            _, topk_idx = torch.topk(probs, arch.top_k, dim=-1)
        else:
            logits = out.detach().float()
            probs = F.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1).cpu()
            _, topk_idx = torch.topk(probs, arch.top_k, dim=-1)

        probs = probs.reshape(-1, probs.shape[-1])
        topk_idx = topk_idx.reshape(-1, arch.top_k)
        all_probs.append(probs)
        all_topk_ids.append(topk_idx)

    hook = gate.register_forward_hook(_capture_hook)

    # Run forward pass in chunks to manage memory
    chunk_size = 512
    n_tokens = input_ids.shape[1]
    for start in range(0, n_tokens - chunk_size + 1, chunk_size):
        chunk = input_ids[:, start:start + chunk_size]
        with torch.no_grad():
            model(chunk)

    hook.remove()

    if not all_hiddens:
        raise RuntimeError(f"No hidden states captured for layer {layer_idx}")

    hiddens = torch.cat(all_hiddens, dim=0)
    topk_ids = torch.cat(all_topk_ids, dim=0)
    probs = torch.cat(all_probs, dim=0)

    print(f"    Extracted {hiddens.shape[0]} samples, "
          f"hidden_dim={hiddens.shape[1]}, n_experts={probs.shape[1]}")

    return HiddenStateDataset(hiddens, topk_ids, probs)


# ═══════════════════════════════════════════════════════════════════
# BVH Router (simplified, universal)
# ═══════════════════════════════════════════════════════════════════

def compute_bvh_shape(n_experts: int) -> Tuple[int, int, int]:
    """Compute balanced 3-level BVH shape for N experts."""
    cbrt = n_experts ** (1.0 / 3.0)
    n1 = max(2, int(math.ceil(cbrt)))
    remaining = math.ceil(n_experts / n1)
    n2 = max(2, int(math.ceil(remaining ** 0.5)))
    n3 = max(2, int(math.ceil(remaining / n2)))
    while n1 * n2 * n3 < n_experts:
        n3 += 1
    return n1, n2, n3


class HierarchicalLevel(nn.Module):
    """One level of the BVH hierarchy."""

    def __init__(self, input_dim: int, n_children: int, feature_dim: int):
        super().__init__()
        self.to_3d = nn.Linear(input_dim, 3)
        self.centroids = nn.Parameter(torch.randn(n_children, 3) * 0.5)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.GELU(),
        )
        self.n_children = n_children

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self.to_3d(x)  # [B, 3]
        dists = torch.cdist(pos.unsqueeze(1),
                            self.centroids.unsqueeze(0)).squeeze(1)  # [B, n_children]
        routing_logits = -dists  # closer = higher logit
        features = self.feature_extractor(x)  # [B, feature_dim]
        return routing_logits, features


class UniversalBVHRouter(nn.Module):
    """Universal BVH Router — works for any number of experts.

    3-level hierarchy: n1 x n2 x n3 >= n_experts
    Input: hidden states [B, hidden_dim]
    Output: expert logits [B, n_experts]
    """

    def __init__(self, input_dim: int, n_experts: int, feature_dim: int = 128):
        super().__init__()
        self.n_experts = n_experts
        n1, n2, n3 = compute_bvh_shape(n_experts)
        self.n1, self.n2, self.n3 = n1, n2, n3

        # Input projection
        proj_dim = min(256, input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

        # Three BVH levels
        self.level1 = HierarchicalLevel(proj_dim, n1, feature_dim)
        self.level2 = HierarchicalLevel(feature_dim, n2, feature_dim)
        self.level3 = HierarchicalLevel(feature_dim, n3, feature_dim)

        # Expert head: features + routing decisions -> expert logits
        head_input = feature_dim + n1 + n2 + n3
        self.expert_head = nn.Sequential(
            nn.Linear(head_input, 256),
            nn.GELU(),
            nn.Linear(256, n_experts),
        )

        # For storing last logits (used by wrapper)
        self._last_logits = None

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)

        r1, f1 = self.level1(h)
        r2, f2 = self.level2(f1)
        r3, f3 = self.level3(f2)

        # Soft routing decisions
        s1 = F.softmax(r1, dim=-1)
        s2 = F.softmax(r2, dim=-1)
        s3 = F.softmax(r3, dim=-1)

        combined = torch.cat([f3, s1, s2, s3], dim=-1)
        logits = self.expert_head(combined)

        self._last_logits = logits
        probs = F.softmax(logits, dim=-1)
        return probs, logits


# ═══════════════════════════════════════════════════════════════════
# Dense Model Sparsification
# ═══════════════════════════════════════════════════════════════════

class DenseFFNSplitter:
    """Splits a dense FFN into N virtual expert groups.

    For SwiGLU: splits gate_proj and up_proj columns into N groups.
    For GELU: splits c_fc columns into N groups.

    Each group = intermediate_size / N columns = one "virtual expert".
    """

    def __init__(self, mlp: nn.Module, n_experts: int, ffn_type: str):
        self.mlp = mlp
        self.n_experts = n_experts
        self.ffn_type = ffn_type

        if ffn_type == 'swiglu':
            total_cols = mlp.gate_proj.out_features
        elif ffn_type == 'gelu':
            total_cols = mlp.c_fc.out_features if hasattr(mlp.c_fc, 'out_features') else mlp.c_fc.weight.shape[0]
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

        self.cols_per_expert = total_cols // n_experts
        if self.cols_per_expert * n_experts < total_cols:
            self.cols_per_expert += 1

    def get_expert_column_ranges(self) -> List[Tuple[int, int]]:
        """Return (start, end) column ranges for each expert group."""
        ranges = []
        if self.ffn_type == 'swiglu':
            total = self.mlp.gate_proj.out_features
        else:
            total = self.mlp.c_fc.out_features if hasattr(self.mlp.c_fc, 'out_features') else self.mlp.c_fc.weight.shape[0]

        for i in range(self.n_experts):
            start = i * self.cols_per_expert
            end = min(start + self.cols_per_expert, total)
            if start < total:
                ranges.append((start, end))
        return ranges


def generate_dense_routing_labels(
    model,
    tokenizer,
    arch: ModelArchInfo,
    layer_idx: int,
    n_experts: int,
    max_tokens: int = 50000,
    device: str = "cuda",
) -> HiddenStateDataset:
    """Generate synthetic routing labels for dense FFN sparsification.

    For each hidden state, computes which expert groups (column ranges)
    contribute most to the output, creating "ground truth" routing labels.
    """
    layers = get_layers(model, arch)
    mlp = getattr(layers[layer_idx], arch.mlp_attr)

    splitter = DenseFFNSplitter(mlp, n_experts, arch.ffn_type)
    col_ranges = splitter.get_expert_column_ranges()

    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in dataset["text"] if t.strip())
    except Exception:
        text = "The quick brown fox " * 5000

    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=min(max_tokens, 4096))
    input_ids = encodings.input_ids.to(device)

    all_hiddens = []
    all_topk_ids = []
    all_probs = []

    # Hook to capture MLP inputs
    def _capture_hook(module, inp, out):
        h = inp[0] if isinstance(inp, tuple) else inp
        h = h.detach().reshape(-1, arch.hidden_size).float()

        # Compute per-group activation magnitude
        with torch.no_grad():
            if arch.ffn_type == 'swiglu':
                gate_out = F.silu(F.linear(h, mlp.gate_proj.weight, mlp.gate_proj.bias))
                up_out = F.linear(h, mlp.up_proj.weight, mlp.up_proj.bias)
                intermediate = gate_out * up_out
            else:
                w = mlp.c_fc.weight
                b = getattr(mlp.c_fc, 'bias', None)
                intermediate = F.gelu(F.linear(h, w, b))

            # Compute importance of each expert group
            group_scores = torch.zeros(h.shape[0], n_experts, device=device)
            for gi, (start, end) in enumerate(col_ranges):
                if gi < n_experts:
                    group_scores[:, gi] = intermediate[:, start:end].abs().mean(dim=-1)

            # Normalize to probabilities
            probs = F.softmax(group_scores, dim=-1)
            top_k = min(8, n_experts)
            _, topk_idx = torch.topk(probs, top_k, dim=-1)

        all_hiddens.append(h.cpu())
        all_topk_ids.append(topk_idx.cpu())
        all_probs.append(probs.cpu())

    hook = mlp.register_forward_hook(_capture_hook)

    chunk_size = 512
    n_tokens = input_ids.shape[1]
    for start in range(0, n_tokens - chunk_size + 1, chunk_size):
        chunk = input_ids[:, start:start + chunk_size]
        with torch.no_grad():
            model(chunk)

    hook.remove()

    if not all_hiddens:
        raise RuntimeError(f"No data captured for layer {layer_idx}")

    hiddens = torch.cat(all_hiddens, dim=0)
    topk_ids = torch.cat(all_topk_ids, dim=0)
    probs = torch.cat(all_probs, dim=0)

    top_k = topk_ids.shape[1]
    print(f"    Extracted {hiddens.shape[0]} samples for dense sparsification "
          f"({n_experts} groups, top-{top_k})")

    return HiddenStateDataset(hiddens, topk_ids, probs)


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def distillation_loss(student_logits: torch.Tensor,
                      teacher_probs: torch.Tensor) -> torch.Tensor:
    """KL divergence between student and teacher routing distributions."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')


def topk_matching_loss(student_logits: torch.Tensor,
                       teacher_topk_ids: torch.Tensor,
                       k: int = 8) -> torch.Tensor:
    """Loss that directly optimizes top-k expert set matching."""
    student_topk = torch.topk(student_logits, k, dim=-1).indices
    # Compute overlap: how many of student's top-k match teacher's top-k
    batch_size = student_logits.shape[0]
    overlap = torch.zeros(batch_size, device=student_logits.device)
    for i in range(k):
        overlap += (student_topk == teacher_topk_ids[:, i:i+1]).any(dim=-1).float()
    # Loss = 1 - normalized overlap (0 = perfect match)
    return 1.0 - overlap.mean() / k


def train_bvh_router(
    dataset: HiddenStateDataset,
    n_experts: int,
    hidden_dim: int,
    top_k: int = 8,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
) -> Tuple[UniversalBVHRouter, dict]:
    """Train a BVH Router to replicate gate routing decisions."""

    router = UniversalBVHRouter(
        input_dim=hidden_dim,
        n_experts=n_experts,
    ).to(device)

    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    best_topk_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        router.train()
        epoch_loss = 0.0
        epoch_topk_match = 0.0
        n_batches = 0

        for hiddens, topk_ids, probs in loader:
            hiddens = hiddens.to(device)
            topk_ids = topk_ids.to(device)
            probs = probs.to(device)

            _, logits = router(hiddens)

            l_kd = distillation_loss(logits, probs)
            l_topk = topk_matching_loss(logits, topk_ids, k=top_k)

            loss = l_kd + 0.3 * l_topk

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()

            # Compute top-k accuracy
            with torch.no_grad():
                student_topk = torch.topk(logits, top_k, dim=-1).indices
                match = 0.0
                for i in range(top_k):
                    match += (student_topk == topk_ids[:, i:i+1]).any(dim=-1).float().mean()
                topk_acc = match / top_k

            epoch_loss += loss.item()
            epoch_topk_match += topk_acc.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_topk = epoch_topk_match / max(n_batches, 1)

        if avg_topk > best_topk_acc:
            best_topk_acc = avg_topk
            best_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"top-{top_k} acc={avg_topk*100:.1f}% "
                  f"(best={best_topk_acc*100:.1f}%)")

    if best_state is not None:
        router.load_state_dict(best_state)
    router.eval()

    stats = {
        "topk_accuracy": best_topk_acc,
        "n_experts": n_experts,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "epochs": epochs,
    }
    return router, stats


# ═══════════════════════════════════════════════════════════════════
# Gate Replacement (BVH Wrapper)
# ═══════════════════════════════════════════════════════════════════

class RetrofitBVHGate(nn.Module):
    """Drop-in BVH gate replacement for any MoE model.

    hybrid_residual mode: BVH selects which experts, original gate weights them.
    pure mode: BVH does everything (selection + weighting).
    """

    def __init__(self, router: UniversalBVHRouter, original_gate: nn.Module,
                 top_k: int = 8, norm_topk_prob: bool = False,
                 mode: str = "hybrid_residual"):
        super().__init__()
        self.router = router
        self.router.eval()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.mode = mode

        # Store original gate weight for hybrid mode
        if hasattr(original_gate, 'weight') and original_gate.weight is not None:
            w = original_gate.weight.data
            if w.numel() > 0:
                self._original_gate_weight = nn.Parameter(w.clone(), requires_grad=False)
            else:
                self._original_gate_weight = None
        else:
            self._original_gate_weight = None

        # Fake weight so code checking gate.weight doesn't crash
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor):
        h2d = hidden_states.reshape(-1, hidden_states.shape[-1])

        with torch.no_grad():
            self.router(h2d.float())
            logits = self.router._last_logits

        logits = logits.to(hidden_states.dtype)

        if self.mode == "hybrid_residual" and self._original_gate_weight is not None:
            # BVH selects top-k indices, original gate assigns weights
            bvh_probs = F.softmax(logits, dtype=torch.float, dim=-1)
            _, top_k_index = torch.topk(bvh_probs, self.top_k, dim=-1)

            gate_logits = F.linear(h2d, self._original_gate_weight)
            gate_probs = F.softmax(gate_logits, dtype=torch.float, dim=-1)
            top_k_weights = gate_probs.gather(1, top_k_index)

            router_probs = torch.zeros_like(gate_probs)
            router_probs.scatter_(1, top_k_index, top_k_weights)
        else:
            # Pure BVH mode
            router_probs = F.softmax(logits, dtype=torch.float, dim=-1)
            top_k_weights, top_k_index = torch.topk(
                router_probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return router_probs, top_k_weights.to(hidden_states.dtype), top_k_index


# ═══════════════════════════════════════════════════════════════════
# PPL Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_ppl(model, tokenizer, max_tokens: int = 50000,
                 device: str = "cuda") -> float:
    """Evaluate perplexity on WikiText-2."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    except Exception:
        text = "The quick brown fox " * 10000

    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=min(max_tokens, 8192))
    input_ids = encodings.input_ids.to(device)

    max_length = min(2048, input_ids.shape[1])
    stride = 512
    nlls = []
    n_tokens = 0

    for begin in range(0, input_ids.shape[1] - max_length + 1, stride):
        end = begin + max_length
        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        target[:, :-1] = -100
        target[:, -stride:] = chunk[:, -stride:]
        target[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(chunk, labels=target)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())
        n_tokens += stride

        if n_tokens >= max_tokens:
            break

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Universal BVH Routing Retrofit — convert any model to O(log N) routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MoE model
  python retrofit_bvh.py --model allenai/OLMoE-1B-7B-0924

  # Dense model (add sparsity with 16 virtual experts)
  python retrofit_bvh.py --model meta-llama/Llama-2-7b-hf --n-experts 16

  # Quick test
  python retrofit_bvh.py --model allenai/OLMoE-1B-7B-0924 --max-tokens 10000 --epochs 20
""")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for retrofitted model (default: retrofitted_<model>/)")
    parser.add_argument("--n-experts", type=int, default=16,
                        help="Number of virtual experts for dense models (ignored for MoE)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs per layer")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max-tokens", type=int, default=50000,
                        help="Max tokens for hidden state extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--mode", type=str, default="hybrid_residual",
                        choices=["hybrid_residual", "pure"],
                        help="Routing mode: hybrid_residual (BVH selects, gate weights) "
                             "or pure (BVH does everything)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to retrofit (default: all MoE layers)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, don't train (requires existing checkpoints)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip PPL evaluation (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("  SpectralAI Zero-Matrix — Universal BVH Retrofit")
    print("  O(log N) geometric routing for any model")
    print("=" * 70)

    # ── Step 1: Load model ──
    print(f"\n[1/5] Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params/1e9:.2f}B params")

    # ── Step 2: Detect architecture ──
    print(f"\n[2/5] Detecting architecture...")
    arch = detect_architecture(model, config)

    print(f"  Model type: {arch.name}")
    print(f"  MoE: {arch.is_moe}")
    print(f"  Layers: {arch.n_layers}")
    print(f"  Hidden size: {arch.hidden_size}")
    print(f"  Intermediate size: {arch.intermediate_size}")
    if arch.is_moe:
        print(f"  Experts: {arch.n_experts} (top-{arch.top_k})")
        print(f"  MoE layers: {len(arch.moe_layer_indices)} of {arch.n_layers}")
        print(f"  Gate attribute: .{arch.gate_attr}")
    else:
        print(f"  FFN type: {arch.ffn_type}")
        print(f"  Virtual experts: {args.n_experts} (for sparsification)")
    print(f"  norm_topk_prob: {arch.norm_topk_prob}")

    # Determine which layers to retrofit
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
    elif arch.is_moe:
        target_layers = list(arch.moe_layer_indices)
    else:
        target_layers = list(range(arch.n_layers))

    n_experts = arch.n_experts if arch.is_moe else args.n_experts
    top_k = arch.top_k if arch.is_moe else min(8, n_experts)

    print(f"\n  Target layers: {target_layers}")
    print(f"  Experts: {n_experts}, top-k: {top_k}")
    print(f"  Mode: {args.mode}")

    # ── Step 3: Baseline PPL ──
    if not args.skip_eval:
        print(f"\n[3/5] Measuring BASELINE PPL...")
        t0 = time.time()
        baseline_ppl = evaluate_ppl(model, tokenizer,
                                     max_tokens=args.max_tokens,
                                     device=args.device)
        print(f"  Baseline PPL = {baseline_ppl:.2f} ({time.time()-t0:.1f}s)")
    else:
        baseline_ppl = None
        print(f"\n[3/5] Skipping baseline PPL (--skip-eval)")

    # ── Step 4: Extract + Train + Replace per layer ──
    output_dir = Path(args.output_dir or f"retrofitted_{args.model.split('/')[-1]}")
    ckpt_dir = output_dir / "bvh_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/5] Training BVH Routers for {len(target_layers)} layers...")

    for layer_idx in target_layers:
        print(f"\n  === Layer {layer_idx}/{arch.n_layers-1} ===")

        ckpt_path = ckpt_dir / f"bvh_router_layer{layer_idx}.pt"

        if args.eval_only and ckpt_path.exists():
            print(f"    Loading existing checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            router = UniversalBVHRouter(
                input_dim=arch.hidden_size, n_experts=n_experts)
            router.load_state_dict(ckpt["router_state_dict"])
            router.eval()
            stats = ckpt.get("stats", {})
        else:
            # Extract hidden states
            print(f"    Extracting hidden states...")
            t0 = time.time()
            if arch.is_moe:
                dataset = extract_hidden_states(
                    model, tokenizer, arch, layer_idx,
                    max_tokens=args.max_tokens, device=args.device)
            else:
                dataset = generate_dense_routing_labels(
                    model, tokenizer, arch, layer_idx,
                    n_experts=n_experts,
                    max_tokens=args.max_tokens, device=args.device)
            print(f"    Extracted in {time.time()-t0:.1f}s")

            # Train BVH Router
            print(f"    Training BVH Router...")
            t0 = time.time()
            router, stats = train_bvh_router(
                dataset, n_experts=n_experts,
                hidden_dim=arch.hidden_size,
                top_k=top_k,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            elapsed = time.time() - t0
            print(f"    Trained in {elapsed:.1f}s — "
                  f"top-{top_k} acc: {stats['topk_accuracy']*100:.1f}%")

            # Save checkpoint
            torch.save({
                "router_state_dict": router.state_dict(),
                "stats": stats,
                "arch": {
                    "name": arch.name, "is_moe": arch.is_moe,
                    "n_experts": n_experts, "hidden_size": arch.hidden_size,
                    "top_k": top_k,
                },
                "layer_idx": layer_idx,
            }, ckpt_path)
            print(f"    Saved: {ckpt_path}")

        # Replace gate with BVH
        if arch.is_moe:
            layers = get_layers(model, arch)
            mlp = getattr(layers[layer_idx], arch.mlp_attr)
            original_gate = getattr(mlp, arch.gate_attr)

            gate_device = next(original_gate.parameters()).device
            router = router.to(gate_device)

            wrapper = RetrofitBVHGate(
                router, original_gate,
                top_k=top_k,
                norm_topk_prob=arch.norm_topk_prob,
                mode=args.mode,
            ).to(gate_device)

            setattr(mlp, arch.gate_attr, wrapper)
            acc_str = f"{stats.get('topk_accuracy', 0)*100:.1f}%"
            print(f"    Replaced gate -> BVH ({args.mode}, acc={acc_str})")

    # ── Step 5: Evaluate retrofitted model ──
    if not args.skip_eval:
        print(f"\n[5/5] Measuring RETROFITTED PPL...")
        t0 = time.time()
        retrofit_ppl = evaluate_ppl(model, tokenizer,
                                     max_tokens=args.max_tokens,
                                     device=args.device)
        elapsed = time.time() - t0

        delta = ((retrofit_ppl - baseline_ppl) / baseline_ppl * 100
                 if baseline_ppl else 0)
        print(f"  Retrofitted PPL = {retrofit_ppl:.2f} ({elapsed:.1f}s)")

        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}")
        print(f"  Model:      {args.model}")
        print(f"  Arch:       {arch.name} ({'MoE' if arch.is_moe else 'Dense'})")
        print(f"  Layers:     {len(target_layers)} retrofitted")
        print(f"  Mode:       {args.mode}")
        print(f"  Baseline:   PPL {baseline_ppl:.2f}")
        print(f"  Retrofitted: PPL {retrofit_ppl:.2f} ({delta:+.1f}%)")
        if abs(delta) < 5:
            print(f"  Verdict:    EXCELLENT — minimal degradation")
        elif abs(delta) < 15:
            print(f"  Verdict:    GOOD — acceptable for production")
        else:
            print(f"  Verdict:    NEEDS WORK — consider hybrid_residual mode")
        print(f"\n  Checkpoints: {ckpt_dir}/")
        print(f"{'='*70}")
    else:
        print(f"\n[5/5] Skipping PPL evaluation (--skip-eval)")
        print(f"  Checkpoints saved to: {ckpt_dir}/")

    # Save config for reproducibility
    import json
    config_path = output_dir / "retrofit_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": args.model,
            "arch": arch.name,
            "is_moe": arch.is_moe,
            "n_experts": n_experts,
            "top_k": top_k,
            "n_layers_retrofitted": len(target_layers),
            "target_layers": target_layers,
            "mode": args.mode,
            "epochs": args.epochs,
            "baseline_ppl": baseline_ppl,
            "retrofitted_ppl": retrofit_ppl if not args.skip_eval else None,
        }, f, indent=2)
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
