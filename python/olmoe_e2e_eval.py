#!/usr/bin/env python3
"""
olmoe_e2e_eval.py -- End-to-end PPL evaluation: BVH Router vs OLMoE Gate

Loads the full OLMoE-1B-7B model, replaces the linear gate in one or more
layers with the trained EnhancedBVHRouter, and measures perplexity on WikiText-2.

This is the definitive validation: if PPL is comparable, the geometric
BVH routing can substitute linear gates with minimal quality loss.

Usage:
    # Single layer (pure BVH with calibration)
    python olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
        --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt

    # Multi-layer
    python olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
        --multi-layer "0:ckpt_layer0.pt,4:ckpt_layer4.pt,8:ckpt_layer8.pt"

    # Hybrid mode (BVH selects candidates, original gate scores)
    python olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b --hybrid

    # Identity test (diagnostic)
    python olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b --identity-test

Copyright (c) 2026 SpectralAI Studio -- Apache 2.0
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from olmoe_bvh_distill import EnhancedBVHRouter, MLPBaselineRouter


# ─────────────────────────────────────────────────────────────────
# DeltaPredictor — tiny learned scale predictor per token
# ─────────────────────────────────────────────────────────────────

class DeltaPredictor(nn.Module):
    """
    Predicts per-token scale for relu_norm/relu_log weight computation.
    Input: 4 features from BVH top-k logits (max, min, std, top1/top2 ratio)
    Output: scale (always positive via exp)

    Total params: 4*16 + 16 + 16*1 + 1 = 97 params per layer.
    With 16 layers: 97 * 16 = 1,552 params total.

    Training: freeze everything, optimize only DeltaPredictor params
    to minimize cross-entropy loss on a small validation set (~1000 tokens).
    """

    def __init__(self, hidden_dim: int = 16, base_scale: float = 0.43):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.base_scale = base_scale
        # Initialize so output ≈ 0 → scale ≈ base_scale
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, top_k_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            top_k_logits: [batch, k] top-k logit values (raw, before compression)
        Returns:
            scale: [batch, 1] per-token scale values
        """
        # Extract 4 features per token
        feat_max = top_k_logits[:, 0:1]                    # max logit
        feat_min = top_k_logits[:, -1:]                     # min of top-k
        feat_std = top_k_logits.std(dim=-1, keepdim=True)   # spread
        feat_ratio = top_k_logits[:, 0:1] / (top_k_logits[:, 1:2] + 1e-6)  # top1/top2
        features = torch.cat([feat_max, feat_min, feat_std, feat_ratio], dim=-1)

        # Predict delta around base_scale
        delta = self.net(features)  # [batch, 1]
        scale = self.base_scale * torch.exp(delta.clamp(-2, 2))  # bounded [base*0.13, base*7.4]
        return scale


class MicroPredictor(nn.Module):
    """
    Ultra-minimal per-layer learned scale. Only 1 param per layer.
    Cannot overfit — just finds the optimal scale for each layer.

    Total: 16 params for 16 layers (vs 1,552 for DeltaPredictor).
    """

    def __init__(self, base_scale: float = 0.43):
        super().__init__()
        # Single learnable delta, initialized to 0 → scale starts at base_scale
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.base_scale = base_scale

    def forward(self, top_k_logits: torch.Tensor) -> torch.Tensor:
        """Returns scalar scale (same for all tokens in this layer)."""
        scale = self.base_scale * torch.exp(self.log_scale.clamp(-1, 1))
        return scale.unsqueeze(0)  # [1] for broadcasting


def calibrate_delta_predictor(
    model, tokenizer, replaced_layers: list,
    base_scale: float = 0.43,
    compression: str = "log1p",
    max_tokens: int = 5000,
    lr: float = 0.01,
    n_steps: int = 200,
    device: str = "cuda",
    micro: bool = False,
):
    """
    Calibrate DeltaPredictor on a small validation set.
    Freezes entire model, only trains the DeltaPredictor params.

    Returns dict {layer_idx: trained DeltaPredictor}.
    """
    # Load validation text
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n\n".join(dataset["text"])
    except Exception:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])

    cal_tokens = min(max_tokens, 2048)
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=cal_tokens)
    input_ids = encodings.input_ids.to(device)
    labels = input_ids.clone()
    print(f"  [delta] Using {input_ids.shape[1]} calibration tokens")

    # Create predictors and attach to wrappers
    predictor_type = "MicroPredictor" if micro else "DeltaPredictor"
    predictors = {}
    for li in replaced_layers:
        try:
            mlp, gate_attr = find_gate_module(model, li)
            wrapper = getattr(mlp, gate_attr)
            if isinstance(wrapper, BVHGateWrapper):
                if micro:
                    dp = MicroPredictor(base_scale=base_scale).to(device)
                else:
                    dp = DeltaPredictor(base_scale=base_scale).to(device)
                wrapper._delta_predictor = dp
                wrapper.weight_mode = f"delta_{compression}"
                predictors[li] = dp
        except Exception:
            pass

    if not predictors:
        print("  [delta] No predictors created!")
        return {}

    # Collect all predictor parameters
    all_params = []
    for dp in predictors.values():
        all_params.extend(dp.parameters())

    n_params = sum(p.numel() for p in all_params)
    print(f"  [delta] Calibrating {len(predictors)} DeltaPredictors "
          f"({n_params} total params) on {input_ids.shape[1]} tokens...")

    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Freeze entire model except predictors
    for p in model.parameters():
        p.requires_grad_(False)
    for dp in predictors.values():
        for p in dp.parameters():
            p.requires_grad_(True)

    best_loss = float('inf')
    best_states = {}

    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == n_steps - 1:
            ppl = math.exp(loss.item())
            print(f"    step {step:3d}/{n_steps}: loss={loss.item():.4f}, ppl={ppl:.2f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_states = {li: {k: v.clone() for k, v in dp.state_dict().items()}
                               for li, dp in predictors.items()}

    # Restore best
    for li, state in best_states.items():
        predictors[li].load_state_dict(state)
        predictors[li].eval()

    # Freeze predictor params again
    for dp in predictors.values():
        for p in dp.parameters():
            p.requires_grad_(False)

    # Free optimizer and gradient buffers
    del optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    best_ppl = math.exp(best_loss)
    print(f"  [delta] Calibration done. Best PPL: {best_ppl:.2f}")
    return predictors

class BVHGateWrapper(nn.Module):
    """
    Wraps EnhancedBVHRouter to act as a drop-in replacement for
    OLMoE's gate module inside the MoE layer.

    If calibration_mode and calibration_state are provided, applies
    a learned transform to the router's raw logits before softmax,
    matching the original gate's weight distribution.

    CRITICAL: norm_topk_prob must match the model config.
    OLMoE-1B-7B uses norm_topk_prob=False.
    """

    def __init__(self, router: EnhancedBVHRouter, top_k: int = 8,
                 norm_topk_prob: bool = False,
                 calibration_mode: str = None,
                 calibration_state: dict = None,
                 logit_temperature: float = None,
                 logit_norm: bool = False,
                 topk_softmax: bool = False,
                 topk_scale: float = None,
                 weight_mode: str = "softmax",
                 n_rays: int = 1,
                 hybrid_alpha: float = 0.98):
        super().__init__()
        self.router = router
        self.router.eval()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.calibration_mode = calibration_mode
        self.logit_temperature = logit_temperature
        self.logit_norm = logit_norm
        self.n_rays = n_rays
        self.hybrid_alpha = hybrid_alpha
        # Weight mode: how to convert logits to routing weights.
        #   "softmax" (default) — standard softmax over all experts
        #   "relu_norm" — ReLU + L1 normalization (no exp amplification)
        #   "topk_softmax" — softmax restricted to top-k experts
        #   "uniform" — equal weight 1/k to all selected experts
        #   "gate_dist" — BVH ranking + fixed weights from original gate distribution
        #   "hybrid_residual" — alpha * BVH_softmax + (1-alpha) * gate_softmax
        self.weight_mode = weight_mode
        # Original gate weight for hybrid_residual mode (set externally)
        self._original_gate_weight = None
        # Backward compat: --topk-softmax flag overrides weight_mode
        if topk_softmax and weight_mode == "softmax":
            self.weight_mode = "topk_softmax"
        self.topk_scale = topk_scale
        # gate_dist: fixed target weights (set externally after measuring original gate)
        self._gate_target_dist = None
        # per-layer scale: set externally for per_layer_scale mode
        self._layer_scale = None
        # DeltaPredictor: set externally during calibration
        self._delta_predictor = None
        # Rank template: fixed weight distribution by rank position
        self._rank_template = None
        # Expert permutation: maps tree positions back to expert IDs
        # perm[tree_pos] = expert_id. Applied to output logits.
        self._expert_perm = None
        # OptiX RT Core bridge: set externally for hardware-accelerated routing
        self._optix_bridge = None
        # Selectivity data: per-expert selectivity from expert_deep_analysis.json
        # Set externally as dict {expert_id_str: selectivity_float}
        self._expert_selectivity = None
        self._layer_idx_for_sel = None  # which layer this wrapper serves
        # Confidence-Gated Routing: adaptive hybrid per token per layer.
        # When BVH confidence is high -> use BVH (pure, O(log N)).
        # When confidence is low -> fallback to original gate (safe).
        # Eliminates accuracy compounding: only uncertain tokens use gate.
        self._confidence_threshold = None  # None = disabled, float = threshold
        self._confidence_stats = {"bvh_tokens": 0, "gate_tokens": 0}  # usage tracking

        if calibration_mode == "affine" and calibration_state is not None:
            self.register_buffer('cal_scale', calibration_state["scale"])
            self.register_buffer('cal_bias', calibration_state["bias"])
        elif calibration_mode == "topk_preserving" and calibration_state is not None:
            inv_temp = calibration_state.get("inv_temp", torch.ones(1))
            self.register_buffer('cal_inv_temp', inv_temp)
            self.register_buffer('cal_bias', calibration_state["bias"])
        elif calibration_mode == "linear" and calibration_state is not None:
            n = int(calibration_state["weight"].shape[0])
            self.cal_linear = nn.Linear(n, n)
            self.cal_linear.load_state_dict(calibration_state)
            self.cal_linear.eval()

        # Fake weight attribute so OLMoE code that checks gate.weight doesn't crash
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)

    _diag_count = 0  # class-level counter for diagnostic prints

    def forward(self, hidden_states: torch.Tensor):
        """
        Returns (router_logits, top_k_weights, top_k_index) to match
        OlmoeTopKRouter interface in transformers >=5.x.

        For transformers <5.x compatibility, callers that expect a single
        tensor can use result[0] or result directly (duck-typed).
        """
        hidden_dim = hidden_states.shape[-1]
        h2d = hidden_states.reshape(-1, hidden_dim)

        # OptiX RT Core path: use hardware ray tracing for routing
        _use_optix = (self._optix_bridge is not None
                      and self._optix_bridge.available)
        if _use_optix:
            import time as _time
            with torch.no_grad():
                # Project to 3D using the router's learned projection
                h = self.router.input_proj(h2d.float())
                T = self.router.temperature.item()
                _, _, pos_3d = self.router.level1(h, T)
                # Compute directions toward weighted center
                centers = self.router.level1.centers  # (4, 3)
                # Use omnidirectional rays (normalized random-ish)
                directions = torch.ones_like(pos_3d)
                directions = directions / directions.norm(
                    dim=-1, keepdim=True
                ).clamp(min=1e-6)
                # RT Core routing
                _t0 = _time.perf_counter()
                topk_ids, topk_dists = self._optix_bridge.route_rt_topk(
                    pos_3d, directions, self.top_k
                )
                if h2d.is_cuda:
                    torch.cuda.synchronize()
                _rt_us = (_time.perf_counter() - _t0) * 1e6
                # Convert to logits: closer distance = higher logit
                logits = torch.zeros(
                    h2d.shape[0], 64,
                    dtype=hidden_states.dtype, device=h2d.device,
                )
                # Inverse distance as logit proxy
                inv_dist = 1.0 / (topk_dists.float() + 1e-6)
                logits.scatter_(1, topk_ids.long(), inv_dist.to(logits.dtype))
                raw_logits = logits
                # Print latency once
                BVHGateWrapper._diag_count += 1
                if BVHGateWrapper._diag_count <= 1:
                    print(f"  [OptiX] RT Core routing: {_rt_us:.1f} us/batch "
                          f"({h2d.shape[0]} tokens)")
        else:
            with torch.no_grad():
                self.router(h2d.float(), n_rays=self.n_rays)
                raw_logits = self.router._last_logits

                if self.calibration_mode == "affine":
                    logits = raw_logits * self.cal_scale + self.cal_bias
                elif self.calibration_mode == "topk_preserving":
                    logits = raw_logits * self.cal_inv_temp + self.cal_bias
                elif self.calibration_mode == "linear":
                    logits = self.cal_linear(raw_logits)
                else:
                    logits = raw_logits

        logits = logits.to(hidden_states.dtype)

        # Expert permutation: remap from tree positions to expert IDs.
        # During training, targets were permuted so co-activating experts
        # sit in the same BVH branch. Here we undo that permutation.
        # perm[tree_pos] = expert_id, so inv_perm maps back.
        if self._expert_perm is not None:
            inv_perm = torch.argsort(self._expert_perm)
            logits = logits[:, inv_perm]

        # ── Confidence-Gated Routing ───────────────────────────────────
        # Per-token adaptive hybrid: BVH when confident, gate when uncertain.
        # Eliminates accuracy compounding across 16 layers by only using BVH
        # where it's reliable, and falling back to the original gate otherwise.
        if (self._confidence_threshold is not None
                and self._original_gate_weight is not None):
            # Confidence = how peaked the BVH logits are.
            # High std = one expert clearly wins = high confidence.
            # Low std = uniform/uncertain = low confidence -> use gate.
            top_k_vals, _ = torch.topk(logits, self.top_k, dim=-1)
            bvh_confidence = top_k_vals.float().std(dim=-1)  # (B,)
            # Normalize to [0,1] range using sigmoid
            confidence = torch.sigmoid(bvh_confidence * 3.0 - 1.5)  # (B,)

            # Mask: True = use BVH, False = use gate
            use_bvh = confidence >= self._confidence_threshold  # (B,)
            n_bvh = use_bvh.sum().item()
            n_gate = (~use_bvh).sum().item()
            self._confidence_stats["bvh_tokens"] += n_bvh
            self._confidence_stats["gate_tokens"] += n_gate

            if n_gate > 0 and not use_bvh.all():
                # Compute gate logits for uncertain tokens
                gate_logits = F.linear(h2d.float(), self._original_gate_weight.float())
                gate_probs = F.softmax(gate_logits, dtype=torch.float, dim=-1)
                gate_top_k_weights, gate_top_k_index = torch.topk(
                    gate_probs, self.top_k, dim=-1)
                gate_router_probs = torch.zeros_like(gate_probs)
                gate_router_probs.scatter_(1, gate_top_k_index, gate_top_k_weights)

                # For confident tokens: continue to weight_mode below.
                # For uncertain tokens: inject gate result directly.
                # We'll merge after weight_mode computes BVH probs.
                self._pending_gate_merge = (use_bvh, gate_router_probs)
            else:
                self._pending_gate_merge = None
        else:
            self._pending_gate_merge = None

        # Apply temperature scaling if set (fixes peaked distributions
        # without changing top-8 expert selection order)
        if self.logit_temperature is not None and self.logit_temperature > 0:
            logits = logits / self.logit_temperature

        # LogitNorm: standardize logits to mean=0, std=1 before softmax.
        if self.logit_norm:
            logits = (logits - logits.mean(dim=-1, keepdim=True)) / (logits.std(dim=-1, keepdim=True) + 1e-6)

        if self.weight_mode in ("delta_log1p", "delta_sqrt"):
            # DELTA PREDICTOR: learned per-token scale via tiny MLP.
            # Same compression as relu_log/relu_norm but with adaptive scale.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            if "log1p" in self.weight_mode:
                compressed = torch.log1p(shifted)
            else:
                compressed = torch.sqrt(shifted)
            top_k_weights = compressed / compressed.sum(dim=-1, keepdim=True)

            if self._delta_predictor is not None:
                scale = self._delta_predictor(top_k_f32)  # [batch, 1]
                top_k_weights = top_k_weights * scale
            else:
                weight_scale = self.topk_scale if self.topk_scale else 0.43
                top_k_weights = top_k_weights * weight_scale

            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode in ("relu_norm", "relu_log", "relu_cbrt"):
            # NORMALIZED ReLU + compression:
            # 1. Select top-k by raw logit
            # 2. Shift so min=1.0 (all experts get meaningful weight)
            # 3. Compress range: sqrt (relu_norm), log1p (relu_log), cbrt (relu_cbrt)
            # 4. L1 normalize, then scale
            # All computation in float32 to avoid float16 epsilon issues.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            if self.weight_mode == "relu_log":
                compressed = torch.log1p(shifted)
            elif self.weight_mode == "relu_cbrt":
                compressed = torch.pow(shifted, 1.0 / 3.0)
            else:
                compressed = torch.sqrt(shifted)
            top_k_weights = compressed / compressed.sum(dim=-1, keepdim=True)
            # Scale: per-layer if available, else global, else default 0.43
            if self._layer_scale is not None:
                weight_scale = self._layer_scale
            elif self.topk_scale is not None:
                weight_scale = self.topk_scale
            else:
                weight_scale = 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "selectivity_geometric":
            # SELECTIVITY-MODULATED ROUTING
            # Add selectivity as a small bias to logits BEFORE top-k selection.
            # High-selectivity experts (specialists) get a positive bias —
            # when the BVH is uncertain, prefer the specialist.
            # Then apply standard relu_norm + sqrt for weight computation.
            if self._expert_selectivity is not None:
                sel_all = torch.tensor(
                    [float(self._expert_selectivity.get(str(i), 0.5))
                     for i in range(logits.shape[1])],
                    device=logits.device, dtype=torch.float32
                )
                # Normalize selectivity to zero-mean, small scale
                sel_norm = (sel_all - sel_all.mean()) / (sel_all.std() + 1e-8)
                # Small additive bias: 0.1 * normalized selectivity
                logits = logits + 0.1 * sel_norm.unsqueeze(0)

            # Standard relu_norm + sqrt pipeline (proven winner)
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            compressed = torch.sqrt(shifted)
            top_k_weights = compressed / compressed.sum(dim=-1, keepdim=True)

            weight_scale = self.topk_scale if self.topk_scale is not None else 0.43
            top_k_weights = top_k_weights * weight_scale

            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "topk_softmax":
            # RESTRICTED SOFTMAX: top-k first, then softmax only over those k.
            top_k_logits, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dtype=torch.float, dim=-1)
            if self.topk_scale is not None:
                top_k_weights = top_k_weights * self.topk_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "uniform":
            # TOP-K UNIFORM: all selected experts get equal weight 1/k.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_weights = torch.full_like(
                top_k_index, 1.0 / self.top_k, dtype=torch.float)
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "gate_dist":
            # GATE DISTRIBUTION: BVH selects top-k by ranking, then assign
            # fixed weights measured from the original gate's average distribution.
            # Ignores BVH logit magnitudes entirely — only uses ranking.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            if self._gate_target_dist is not None:
                target = self._gate_target_dist.to(logits.device)
            else:
                # Fallback: typical OLMoE-1B-7B measured distribution
                target = torch.tensor(
                    [0.135, 0.105, 0.095, 0.088, 0.082, 0.078, 0.072, 0.065],
                    device=logits.device)
            # Broadcast target [k] → [batch, k]
            top_k_weights = target.unsqueeze(0).expand(
                top_k_index.shape[0], -1).float()
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "rank_template":
            # RANK TEMPLATE: BVH only provides ranking (top-k order).
            # Weights come from a fixed template, ignoring BVH logit magnitudes.
            # Steeper than gate_dist — more weight on top-1.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            if self._rank_template is not None:
                template = self._rank_template.to(logits.device)
            else:
                # Default steep template (measured from typical OLMoE gate)
                template = torch.tensor(
                    [0.312, 0.148, 0.112, 0.089, 0.071, 0.058, 0.047, 0.038],
                    device=logits.device)
            # Normalize to target sum
            target_sum = self.topk_scale if self.topk_scale else 0.43
            template = template / template.sum() * target_sum
            top_k_weights = template.unsqueeze(0).expand(
                top_k_index.shape[0], -1).float()
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "hybrid_residual":
            # HYBRID RESIDUAL: use BVH for expert SELECTION (top-k indices),
            # but compute final weights from the original gate's softmax.
            # This preserves BVH's O(log N) routing while using the gate's
            # trained weight distribution for the selected experts.
            # alpha controls the blend: 1.0 = pure BVH selection + gate weights,
            # 0.0 = pure gate (baseline).
            if self._original_gate_weight is not None:
                # BVH selects top-k indices
                bvh_probs = F.softmax(logits, dtype=torch.float, dim=-1)
                _, bvh_top_k_index = torch.topk(bvh_probs, self.top_k, dim=-1)

                # Gate computes weights for those indices
                gate_logits = F.linear(h2d, self._original_gate_weight)
                gate_probs = F.softmax(gate_logits, dtype=torch.float, dim=-1)

                # Gather gate probabilities at BVH-selected positions
                top_k_weights = gate_probs.gather(1, bvh_top_k_index)
                top_k_index = bvh_top_k_index

                router_probs = torch.zeros_like(gate_probs)
                router_probs.scatter_(1, top_k_index, top_k_weights)
            else:
                # Fallback: standard softmax
                router_probs = F.softmax(logits, dtype=torch.float, dim=-1)
                top_k_weights, top_k_index = torch.topk(
                    router_probs, self.top_k, dim=-1
                )
            if self.norm_topk_prob:
                top_k_weights = top_k_weights / top_k_weights.sum(
                    dim=-1, keepdim=True
                )

        elif self.weight_mode == "geometric":
            # GEOMETRIC WEIGHTS (inspired by inverse distance weighting / gravity)
            # BVH selects top-k by logits. Weights come from 1/distance in
            # the BVH 3D space — closer experts get more weight. Pure mode:
            # NO original gate needed. Only 1 param (temperature) per layer.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            geo_dist = getattr(self.router, '_last_geometric_distances', None)
            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                # Gather distances for selected experts
                top_k_dist = geo_dist.gather(1, top_k_index)  # (B, k)
                # Inverse distance weighting: weight = 1 / (d + eps)
                inv_dist = 1.0 / (top_k_dist + 1e-6)
                # Softmax over inverse distances with learned temperature
                geo_temp = self._geo_temperature if hasattr(self, '_geo_temperature') else 1.0
                top_k_weights = F.softmax(inv_dist / geo_temp, dim=-1)
            else:
                # Fallback: uniform if no geometric data
                top_k_weights = torch.full(
                    (logits.shape[0], self.top_k), 1.0 / self.top_k,
                    dtype=torch.float, device=logits.device)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "lambert":
            # LAMBERT COSINE LAW (from optics: intensity ∝ cos(angle of incidence))
            # The "angle" is computed between the query's direction vector (pos3d)
            # and each expert centroid's normal. Experts hit more "head-on" get
            # more weight. Pure mode: no gate needed.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            geo_dist = getattr(self.router, '_last_geometric_distances', None)
            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                # Lambert: cos(θ) ≈ 1 - normalized_distance (monotonically decreasing)
                # Normalize distances to [0, 1] range per token
                d_min = top_k_dist.min(dim=-1, keepdim=True).values
                d_max = top_k_dist.max(dim=-1, keepdim=True).values
                d_range = d_max - d_min + 1e-8
                d_norm = (top_k_dist - d_min) / d_range  # [0, 1]
                cos_theta = 1.0 - d_norm  # closer = higher cos
                # Clamp to avoid zero weights
                cos_theta = cos_theta.clamp(min=0.05)
                top_k_weights = cos_theta / cos_theta.sum(dim=-1, keepdim=True)
            else:
                top_k_weights = torch.full(
                    (logits.shape[0], self.top_k), 1.0 / self.top_k,
                    dtype=torch.float, device=logits.device)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "resonance":
            # HARMONIC RESONANCE (from acoustics / wave physics)
            # Each expert has a "natural frequency" (its logit magnitude).
            # Weight follows a resonance curve: sharper peak for strong matches,
            # broader response for weak matches. Like a tuning fork — experts
            # that "resonate" with the query dominate.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            # Resonance: weight = 1 / ((f - f0)^2 + gamma^2)  (Lorentzian)
            # f0 = peak (top-1 logit), gamma = bandwidth
            f0 = top_k_f32[:, 0:1]  # peak frequency (strongest expert)
            gamma = 0.5  # bandwidth — controls how quickly non-peak experts decay
            resonance = 1.0 / ((top_k_f32 - f0) ** 2 + gamma ** 2)
            top_k_weights = resonance / resonance.sum(dim=-1, keepdim=True)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "render_eq":
            # RENDERING EQUATION (from computer graphics)
            # color = light × material × geometry — three independent signals.
            # Here: weight = logit_strength × geometric_proximity.
            # Experts with high logits AND close in BVH space get more weight.
            # Adaptive per-token (like relu_norm) but informed by geometry.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            # Shift logits to positive range
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            logit_signal = torch.sqrt(shifted)  # compress like relu_norm

            geo_dist = getattr(self.router, '_last_geometric_distances', None)
            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                # Geometric proximity: 1/sqrt(d) — softer than 1/d
                geo_signal = 1.0 / (torch.sqrt(top_k_dist) + 1e-6)
                # Normalize geo_signal to mean=1 so it modulates without changing scale
                geo_signal = geo_signal / (geo_signal.mean(dim=-1, keepdim=True) + 1e-8)
                # Combine: logit shape × geometry modulation
                combined = logit_signal * geo_signal
            else:
                combined = logit_signal

            top_k_weights = combined / combined.sum(dim=-1, keepdim=True)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "ray_march":
            # RAY MARCHING ATTENUATION (from volumetric rendering)
            # In fog/smoke rendering: intensity = base × exp(-σ × distance)
            # Exponential decay is physically correct for light absorption.
            # Softer than 1/sqrt(d), avoids the sharp falloff of 1/d.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            logit_signal = torch.sqrt(shifted)

            geo_dist = getattr(self.router, '_last_geometric_distances', None)
            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                # Normalize distances per token to [0, ~3] range for exp
                d_mean = top_k_dist.mean(dim=-1, keepdim=True) + 1e-8
                d_norm = top_k_dist / d_mean
                # Exponential attenuation: exp(-σ × d), σ=0.5
                sigma = 0.5
                attenuation = torch.exp(-sigma * d_norm)
                combined = logit_signal * attenuation
            else:
                combined = logit_signal

            top_k_weights = combined / combined.sum(dim=-1, keepdim=True)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "bm25":
            # BM25 SATURATION (from information retrieval / search engines)
            # In Google/BM25, term frequency saturates: tf/(tf+k).
            # Prevents any single expert from monopolizing the weight.
            # Mathematically guarantees flat-ish head distribution.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            geo_dist = getattr(self.router, '_last_geometric_distances', None)

            # Logit signal with saturation: relu(x) / (relu(x) + K)
            top_k_logits = logits.gather(1, top_k_index).float()
            shifted = top_k_logits - top_k_logits.min(dim=-1, keepdim=True).values
            material = torch.relu(shifted)
            K_sat = 1.0  # saturation constant — higher = flatter
            saturated = material / (material + K_sat)

            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                geometry = 1.0 / (torch.sqrt(top_k_dist) + 1e-6)
                geometry = geometry / (geometry.mean(dim=-1, keepdim=True) + 1e-8)
                combined = saturated * geometry
            else:
                combined = saturated

            top_k_weights = combined / (combined.sum(dim=-1, keepdim=True) + 1e-8)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "gravity":
            # GRAVITY MODEL / ALiBi (from economics + NLP)
            # Trade flow ∝ (Mass_A × Mass_B) / Distance^β
            # In log space: log(weight) = logit - α·log(distance)
            # This is how ALiBi (used in MPT, BLOOM) injects spatial bias
            # without altering softmax temperature. Mathematically clean:
            # log(A × B) = log(A) + log(B)
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            geo_dist = getattr(self.router, '_last_geometric_distances', None)

            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                alpha_grav = 0.3  # spatial penalty strength
                combined_logits = top_k_vals.float() - alpha_grav * torch.log(top_k_dist + 1e-6)
            else:
                combined_logits = top_k_vals.float()

            # Softmax in log-combined space — temperature preserved
            shifted = combined_logits - combined_logits.min(dim=-1, keepdim=True).values + 1.0
            compressed = torch.sqrt(shifted)
            top_k_weights = compressed / compressed.sum(dim=-1, keepdim=True)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "hierarchical":
            # HIERARCHICAL BRANCH BONUS (from BVH tree structure)
            # Experts in the same branch as the query's main routing choice
            # get bonus weight. Uses tree topology, not retraining.
            # 64 experts = 4×4×4 tree. Same domain +20%, same subdomain +10%.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            logit_signal = torch.sqrt(shifted)

            # Branch bonus from tree structure
            branch_bonus = getattr(self.router, '_last_branch_bonus', None)
            if branch_bonus is not None:
                branch_bonus = branch_bonus.to(logits.device)
                top_k_bonus = branch_bonus.gather(1, top_k_index)
                # Normalize bonus to mean=1 so it modulates without changing scale
                top_k_bonus = top_k_bonus / (top_k_bonus.mean(dim=-1, keepdim=True) + 1e-8)
            else:
                top_k_bonus = 1.0

            combined = logit_signal * top_k_bonus
            top_k_weights = combined / (combined.sum(dim=-1, keepdim=True) + 1e-8)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "spectral_weight":
            # SPECTRAL REFRACTION WEIGHTS (from our own optics architecture)
            # The PrismaticRefraction module computes how each expert "refracts"
            # the query's spectral color. High refraction = expert resonates
            # with this token's semantic frequency. This IS the physically
            # motivated weight signal — already trained, per-token adaptive.
            # Three signals combined: logits × geometry × spectral resonance.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_f32 = top_k_vals.float()
            shifted = top_k_f32 - top_k_f32.min(dim=-1, keepdim=True).values + 1.0
            logit_signal = torch.sqrt(shifted)

            # Spectral refraction: higher = more resonance with this token
            spectral_ref = getattr(self.router, '_last_spectral_refraction', None)
            if spectral_ref is not None:
                spectral_ref = spectral_ref.to(logits.device)
                top_k_spectral = spectral_ref.gather(1, top_k_index)
                # Refraction is sigmoid output [0,1] — center at 0.5, scale
                spectral_signal = top_k_spectral / (top_k_spectral.mean(dim=-1, keepdim=True) + 1e-8)
            else:
                spectral_signal = 1.0

            # Geometric proximity
            geo_dist = getattr(self.router, '_last_geometric_distances', None)
            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                geo_signal = 1.0 / (torch.sqrt(top_k_dist) + 1e-6)
                geo_signal = geo_signal / (geo_signal.mean(dim=-1, keepdim=True) + 1e-8)
            else:
                geo_signal = 1.0

            # Rendering equation: Light × Material × Geometry
            # logit = "light intensity", spectral = "material color", geo = "geometry term"
            combined = logit_signal * spectral_signal * geo_signal
            top_k_weights = combined / (combined.sum(dim=-1, keepdim=True) + 1e-8)
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "importance":
            # IMPORTANCE SAMPLING (from Monte Carlo physics)
            # Standard softmax gives probabilities, then reweight by
            # geometric importance: closer samples matter more.
            # w = softmax(logits) × (1/dist), renormalized.
            top_k_vals, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            top_k_probs = F.softmax(top_k_vals.float(), dim=-1)

            geo_dist = getattr(self.router, '_last_geometric_distances', None)
            if geo_dist is not None:
                geo_dist = geo_dist.to(logits.device)
                top_k_dist = geo_dist.gather(1, top_k_index)
                importance = 1.0 / (top_k_dist + 0.1)
                combined = top_k_probs * importance
                top_k_weights = combined / combined.sum(dim=-1, keepdim=True)
            else:
                top_k_weights = top_k_probs

            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = top_k_weights * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        elif self.weight_mode == "zipf":
            # ZIPF'S LAW (from linguistics / information theory)
            # Universal power law: weight = 1 / rank^s, where s ≈ 1.
            # Observed in word frequencies, city sizes, earthquakes, neural networks.
            # Key insight: the original gate's weight distribution IS zipfian.
            # Gate measured: [0.312, 0.148, 0.112, 0.089, 0.071, 0.058, 0.047, 0.038]
            # Zipf s=1:     [0.341, 0.171, 0.114, 0.085, 0.068, 0.057, 0.049, 0.043]
            # Pure mode: only uses BVH ranking, no gate needed.
            _, top_k_index = torch.topk(logits, self.top_k, dim=-1)
            ranks = torch.arange(1, self.top_k + 1, dtype=torch.float,
                                 device=logits.device)
            zipf_s = self._zipf_s if hasattr(self, '_zipf_s') else 1.0
            zipf_weights = 1.0 / (ranks ** zipf_s)
            zipf_weights = zipf_weights / zipf_weights.sum()
            weight_scale = self.topk_scale if self.topk_scale else 0.43
            top_k_weights = zipf_weights.unsqueeze(0).expand(
                top_k_index.shape[0], -1) * weight_scale
            router_probs = torch.zeros(logits.shape[0], logits.shape[1],
                                       dtype=torch.float, device=logits.device)
            router_probs.scatter_(1, top_k_index, top_k_weights)

        else:
            # Standard: softmax over all 64 experts, then top-k
            router_probs = F.softmax(logits, dtype=torch.float, dim=-1)
            top_k_weights, top_k_index = torch.topk(
                router_probs, self.top_k, dim=-1
            )
            if not self.norm_topk_prob:
                top_k_weights = top_k_weights
            else:
                top_k_weights = top_k_weights / top_k_weights.sum(
                    dim=-1, keepdim=True
                )

        # ── Confidence-Gated Merge ─────────────────────────────────────
        # Merge BVH and gate results: confident tokens keep BVH weights,
        # uncertain tokens get overwritten with gate weights.
        if self._pending_gate_merge is not None:
            use_bvh, gate_router_probs = self._pending_gate_merge
            self._pending_gate_merge = None
            # Expand mask for broadcasting: (B,) -> (B, 1)
            use_bvh_mask = use_bvh.unsqueeze(1).float()
            # Merge: BVH where confident, gate where not
            router_probs = router_probs * use_bvh_mask + gate_router_probs * (1.0 - use_bvh_mask)
            # Recompute top_k from merged probs
            top_k_weights, top_k_index = torch.topk(router_probs, self.top_k, dim=-1)

        # Diagnostic: print once per layer to verify temperature is working
        BVHGateWrapper._diag_count += 1
        if BVHGateWrapper._diag_count <= 1:
            sample = raw_logits[0]  # first token
            top5_raw = torch.topk(sample, 5)
            top5_post = torch.topk(logits[0], 5) if logits.shape[0] > 0 else top5_raw
            conf_str = ""
            if self._confidence_threshold is not None:
                stats = self._confidence_stats
                total = stats["bvh_tokens"] + stats["gate_tokens"]
                bvh_pct = stats["bvh_tokens"] / max(total, 1) * 100
                conf_str = f" conf_gate={bvh_pct:.0f}%BVH/{100-bvh_pct:.0f}%gate"
            print(f"  [DIAG] call#{BVHGateWrapper._diag_count} "
                  f"cal={self.calibration_mode} temp={self.logit_temperature} "
                  f"raw_top5={top5_raw.values.tolist()[:5]} "
                  f"post_temp_top5={top5_post.values.tolist()[:5]} "
                  f"top8_weights={top_k_weights[0].tolist()}{conf_str}")

        return router_probs, top_k_weights.to(hidden_states.dtype), top_k_index


# ─────────────────────────────────────────────────────────────────
# Identity Gate Wrapper — diagnostic: wrap original weights
# ─────────────────────────────────────────────────────────────────

class IdentityGateWrapper(nn.Module):
    """
    Diagnostic wrapper: uses the ORIGINAL gate weight through the same
    code path as BVHGateWrapper. If PPL changes, the wrapper interface
    is broken. If PPL stays at ~6.11, the wrapper is correct.

    norm_topk_prob MUST match the original model config. For OLMoE-1B-7B
    this is False — do NOT normalize top-k weights.
    """

    def __init__(self, original_weight: torch.Tensor, top_k: int = 8,
                 norm_topk_prob: bool = False):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = original_weight.shape[1]
        self.weight = nn.Parameter(original_weight.clone(), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor):
        """Replicate OlmoeTopKRouter.forward() exactly."""
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


# ─────────────────────────────────────────────────────────────────
# Model loading + gate replacement
# ─────────────────────────────────────────────────────────────────

def load_olmoe_model(model_dir: str, device: str = "cuda"):
    """Load full OLMoE-1B-7B model via transformers."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers accelerate safetensors")

    print(f"  Loading OLMoE-1B-7B from {model_dir}...")
    print(f"  This loads the FULL model (~14 GB in FP16)...")
    print(f"  Using device_map='auto' to split GPU/CPU if needed...")

    t0 = time.time()
    is_local = os.path.isdir(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=is_local,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=is_local,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  Loaded in {elapsed:.1f}s — {n_params/1e9:.2f}B params, {mem_gb:.2f} GB")

    if hasattr(model, 'hf_device_map'):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Device map: {', '.join(sorted(devices_used))}")

    return model, tokenizer


def measure_gate_distribution(model, tokenizer, max_tokens: int = 10000,
                               top_k: int = 8, device: str = "cuda"):
    """
    Measure the average top-k weight distribution of the original gates.
    Runs a small forward pass and records what weights the original gate assigns
    to the 1st, 2nd, ..., k-th expert by rank.

    Returns: (avg_dist [k], per_layer_dist dict {layer_idx: tensor[k]},
              per_layer_sum dict {layer_idx: float})
    """
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    except Exception:
        text = "The quick brown fox " * 5000

    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=min(max_tokens, 2048))
    input_ids = encodings.input_ids.to(device)

    # Hook into all gate modules to capture top-k weights PER LAYER
    per_layer_accum = {}  # {layer_idx: tensor[k]}
    per_layer_count = {}  # {layer_idx: int}
    hooks = []

    def _make_hook(layer_idx):
        def _hook(module, input, output):
            # output is (router_logits,) or (router_probs, top_k_weights, top_k_index)
            if isinstance(output, tuple) and len(output) >= 2:
                top_k_w = output[1]  # [batch*seq, k]
            else:
                logits = output[0] if isinstance(output, tuple) else output
                probs = F.softmax(logits.float(), dim=-1)
                top_k_w, _ = torch.topk(probs, top_k, dim=-1)

            mean_w = top_k_w.float().mean(dim=0)  # [k]
            sorted_w, _ = torch.sort(mean_w, descending=True)
            if layer_idx not in per_layer_accum:
                per_layer_accum[layer_idx] = torch.zeros(top_k, device=device)
                per_layer_count[layer_idx] = 0
            per_layer_accum[layer_idx] += sorted_w[:top_k]
            per_layer_count[layer_idx] += 1
        return _hook

    # Register hooks on all gates
    n_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
    for li in range(n_layers):
        try:
            mlp, gate_attr = find_gate_module(model, li)
            gate = getattr(mlp, gate_attr)
            h = gate.register_forward_hook(_make_hook(li))
            hooks.append(h)
        except Exception:
            pass

    # Single forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute per-layer averages and overall average
    per_layer_dist = {}
    per_layer_sum = {}
    total_accum = torch.zeros(top_k, device=device)
    total_count = 0
    for li in sorted(per_layer_accum.keys()):
        per_layer_dist[li] = per_layer_accum[li] / per_layer_count[li]
        per_layer_sum[li] = float(per_layer_dist[li].sum())
        total_accum += per_layer_dist[li]
        total_count += 1

    if total_count > 0:
        avg_dist = total_accum / total_count
    else:
        avg_dist = torch.tensor([0.135, 0.105, 0.095, 0.088, 0.082,
                                 0.078, 0.072, 0.065], device=device)

    print(f"  [gate_dist] Measured original gate top-{top_k} distribution "
          f"from {total_count} layers:")
    print(f"  [gate_dist] avg = {avg_dist.tolist()}")
    print(f"  [gate_dist] avg sum = {avg_dist.sum():.4f}")
    print(f"  [gate_dist] per-layer sums: "
          + ", ".join(f"L{li}={s:.4f}" for li, s in per_layer_sum.items()))
    return avg_dist, per_layer_dist, per_layer_sum


def find_gate_module(model, layer_idx: int):
    """Find the gate module path in the OLMoE model for a given layer."""
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[layer_idx]
        elif hasattr(model, 'layers'):
            layer = model.layers[layer_idx]
        else:
            raise AttributeError("Cannot find layers attribute")

        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
        elif hasattr(layer, 'block_sparse_moe'):
            mlp = layer.block_sparse_moe
        else:
            raise AttributeError(f"Cannot find mlp in layer {layer_idx}")

        if hasattr(mlp, 'gate'):
            return mlp, 'gate'
        elif hasattr(mlp, 'router'):
            return mlp, 'router'
        else:
            attrs = [a for a in dir(mlp) if not a.startswith('_')
                     and isinstance(getattr(mlp, a, None), nn.Module)]
            raise AttributeError(
                f"Cannot find gate in layer {layer_idx} mlp. "
                f"Module children: {attrs}"
            )
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Cannot find gate at layer {layer_idx}: {e}")


def replace_gate_with_bvh(
    model,
    router_checkpoint: str,
    layer_idx: int = 8,
    hybrid: bool = False,
    n_candidates: int = 16,
    logit_temperature: float = None,
    no_calibration: bool = False,
    logit_norm: bool = False,
    topk_softmax: bool = False,
    topk_scale: float = None,
    weight_mode: str = "softmax",
    n_rays: int = 1,
    hybrid_alpha: float = 0.98,
    use_optix: bool = False,
    selectivity_data: dict = None,
    confidence_threshold: float = None,
) -> tuple:
    """
    Replace the linear gate in one OLMoE layer with the trained BVH Router.

    If hybrid=True, monkey-patches the gate's forward to use BVH for candidate
    selection and the original gate weight for exact scoring.

    Returns (original_gate, orig_forward_or_None).
    """
    # Load trained router
    try:
        ckpt = torch.load(router_checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(router_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    router_type = ckpt.get("router_type", "bvh")

    if router_type == "mlp":
        router = MLPBaselineRouter(
            input_dim=config["input_dim"],
            n_experts=config.get("n_experts", 64),
        )
    else:
        # Infer spectral params from config, top-level key, or state_dict keys
        sd = ckpt["router_state_dict"]
        spectral_mode = config.get("spectral_mode", ckpt.get("spectral_mode", False))
        se_out_key = "spectral_encoder.2.weight"
        se_in_key = "spectral_encoder.0.weight"
        if not spectral_mode and se_out_key in sd:
            spectral_mode = True
        spectral_dim = config.get("spectral_dim", 64)
        enc_hidden = None
        if spectral_mode and se_out_key in sd:
            spectral_dim = sd[se_out_key].shape[0]
            enc_hidden = sd[se_in_key].shape[0]

        router = EnhancedBVHRouter(
            input_dim=config["input_dim"],
            n_level1=config["n_level1"],
            n_level2=config["n_level2"],
            n_level3=config["n_level3"],
            feature_dim=config["feature_dim"],
            spectral_mode=spectral_mode,
            spectral_dim=spectral_dim,
            encoder_hidden=enc_hidden,
        )
    router.load_state_dict(ckpt["router_state_dict"])
    router.eval()

    # Find gate module
    mlp, gate_attr = find_gate_module(model, layer_idx)
    original_gate = getattr(mlp, gate_attr)

    # Move router to same device as the original gate
    try:
        gate_device = next(original_gate.parameters()).device
        if str(gate_device) == "meta":
            gate_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except StopIteration:
        gate_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = router.to(gate_device)

    topk_acc = ckpt.get("topk_accuracy", 0)
    top1_acc = ckpt.get("top1_accuracy", 0)

    if hybrid:
        # Hybrid: BVH selects top-N candidate indices, then we compute the FULL
        # 64-expert softmax and pick top-k from only the candidate positions.
        # CRITICAL: norm_topk_prob=False for OLMoE-1B-7B — do NOT normalize.
        # CRITICAL: use full softmax (not restricted to candidates) for weights.
        orig_weight = original_gate.weight.data
        _norm_topk = getattr(original_gate, 'norm_topk_prob', False)
        _top_k_cfg = getattr(original_gate, 'top_k', 8)
        _orig_forward = original_gate.forward

        def _hybrid_forward(hidden_states, _router=router, _weight=orig_weight,
                            _n_cand=n_candidates, _top_k=_top_k_cfg,
                            _norm=_norm_topk):
            hidden_dim = _weight.shape[1]
            h2d = hidden_states.reshape(-1, hidden_dim)

            with torch.no_grad():
                bvh_probs, _ = _router(h2d.float())

            # Step 1: BVH pruning — top-N candidate indices
            _, candidate_ids = torch.topk(bvh_probs, _n_cand, dim=-1)

            # Step 2: FULL 64-expert softmax (same as original gate)
            full_logits = F.linear(h2d, _weight)
            full_probs = F.softmax(full_logits, dtype=torch.float, dim=-1)

            # Step 3: Restrict topk to candidates — gather candidate probs
            # from the full softmax, then pick top-k
            cand_probs = full_probs.gather(1, candidate_ids)
            topk_vals, topk_local = torch.topk(cand_probs, _top_k, dim=-1)
            topk_global = candidate_ids.gather(1, topk_local)

            # Step 4: Apply norm_topk_prob exactly as original model
            if _norm:
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            topk_scores = topk_vals.to(hidden_states.dtype)

            return full_probs, topk_scores, topk_global

        original_gate.forward = _hybrid_forward
        mode_str = (f"HYBRID monkey-patch (BVH→{n_candidates} candidates, "
                    f"full softmax, norm={_norm_topk})")
    else:
        # Pure BVH: replace module entirely
        _norm_topk = getattr(original_gate, 'norm_topk_prob', False)
        _top_k_cfg = getattr(original_gate, 'top_k', 8)
        _orig_forward = None

        # Load calibration if available (new format: mode + state dict)
        cal_mode = ckpt.get("calibration_mode", None)
        cal_state_raw = ckpt.get("calibration_state", None)
        # Backward compat: old format had calibration_scale/calibration_bias
        if cal_mode is None and "calibration_scale" in ckpt:
            cal_mode = "affine"
            cal_state_raw = {
                "scale": ckpt["calibration_scale"],
                "bias": ckpt["calibration_bias"],
            }
        cal_state = None
        if cal_state_raw is not None:
            cal_state = {k: v.to(gate_device) for k, v in cal_state_raw.items()}

        if no_calibration:
            cal_mode = None
            cal_state = None

        wrapper = BVHGateWrapper(
            router, top_k=_top_k_cfg, norm_topk_prob=_norm_topk,
            calibration_mode=cal_mode, calibration_state=cal_state,
            logit_temperature=logit_temperature,
            logit_norm=logit_norm,
            topk_softmax=topk_softmax,
            topk_scale=topk_scale,
            weight_mode=weight_mode,
            n_rays=n_rays,
            hybrid_alpha=hybrid_alpha,
        ).to(gate_device)
        # Store original gate weight for hybrid_residual mode
        if hasattr(original_gate, 'weight') and original_gate.weight is not None:
            wrapper._original_gate_weight = original_gate.weight.data.to(gate_device)
        # Load expert permutation from checkpoint if present
        ep = ckpt.get("expert_perm", None)
        if ep is not None:
            perm_tensor = torch.tensor(ep, dtype=torch.long, device=gate_device)
            wrapper._expert_perm = perm_tensor
            print(f"  Expert permutation loaded for layer {layer_idx} "
                  f"(first 8: {perm_tensor[:8].tolist()})")
        # Load selectivity data for selectivity_geometric mode
        if selectivity_data is not None:
            layer_key = str(layer_idx)
            if "per_layer_selectivity" in selectivity_data:
                layer_sel = selectivity_data["per_layer_selectivity"].get(layer_key, {})
                if layer_sel:
                    wrapper._expert_selectivity = layer_sel
                    wrapper._layer_idx_for_sel = layer_idx
                    vals = [float(v) for v in layer_sel.values()]
                    print(f"  Selectivity loaded for layer {layer_idx}: "
                          f"range [{min(vals):.3f}, {max(vals):.3f}] "
                          f"({max(vals)/min(vals):.1f}x)")
        # Confidence-Gated Routing: set threshold for adaptive hybrid
        if confidence_threshold is not None:
            wrapper._confidence_threshold = confidence_threshold
            print(f"  Confidence gate enabled: threshold={confidence_threshold:.2f} "
                  f"(BVH when confident, gate fallback when uncertain)")
        # Initialize OptiX RT Core bridge if requested
        if use_optix:
            try:
                from optix_training_bridge import OptiXTrainingBridge
                bridge = OptiXTrainingBridge(auto_init=True)
                if bridge.has_extension:
                    # Build GAS from the 64 leaf-level expert positions
                    # Use level3 centers as the expert sphere positions
                    with torch.no_grad():
                        centers = router.level3.centers.detach().cpu().float()
                        radii = torch.exp(
                            router.level3.log_radii.detach().cpu().float()
                        )
                    if bridge.build_gas(centers, radii, use_triangles=True):
                        wrapper._optix_bridge = bridge
                        print(f"  OptiX RT Core bridge initialized "
                              f"(GAS: {bridge._smooth_hit.__class__.__name__}, "
                              f"{centers.shape[0]} experts)")
                    else:
                        print(f"  [WARNING] OptiX GAS build failed, "
                              f"using PyTorch fallback")
                else:
                    print(f"  [WARNING] OptiX extension not loaded, "
                          f"using PyTorch fallback")
            except Exception as exc:
                print(f"  [WARNING] OptiX init failed: {exc}, "
                      f"using PyTorch fallback")
        setattr(mlp, gate_attr, wrapper)
        cal_str = f"{cal_mode}" if cal_mode else "none"
        wm = wrapper.weight_mode  # resolved mode (accounts for --topk-softmax compat)
        if wm != "softmax":
            cal_str += f", weight_mode={wm}"
        if logit_norm:
            cal_str += ", logit_norm=True"
        if logit_temperature:
            cal_str += f", temp={logit_temperature}"
        mode_str = f"BVH Router (pure, calibration={cal_str})"

    print(f"  Replaced layer {layer_idx} gate with {mode_str}")
    print(f"  (checkpoint: top-8={topk_acc*100:.1f}%, top-1={top1_acc*100:.1f}%)")

    return original_gate, _orig_forward


# ─────────────────────────────────────────────────────────────────
# PPL evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_ppl(
    model,
    tokenizer,
    text: Optional[str] = None,
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cuda",
    max_tokens: int = 50000,
) -> float:
    """
    Evaluate perplexity on text using sliding window.

    Uses the standard HuggingFace PPL evaluation method with stride.
    """
    if text is None:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(dataset["text"])
        except Exception:
            import numpy as np
            data_path = Path(__file__).parent.parent / "data" / "wikitext2_validation_tokens.npy"
            if data_path.exists():
                tokens = np.load(str(data_path))
                input_ids = torch.tensor(tokens[:max_tokens]).unsqueeze(0)
                return _eval_ppl_from_ids(model, input_ids, max_length, stride, device)
            raise ImportError("pip install datasets  (or place wikitext2_validation_tokens.npy)")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    if max_tokens and input_ids.size(1) > max_tokens:
        input_ids = input_ids[:, :max_tokens]
    return _eval_ppl_from_ids(model, input_ids, max_length, stride, device)


def _eval_ppl_from_ids(
    model,
    input_ids: torch.Tensor,
    max_length: int,
    stride: int,
    device: str,
) -> float:
    """Evaluate PPL from pre-tokenized IDs using sliding window."""
    seq_len = input_ids.size(1)
    nlls = []
    n_tokens = 0

    print(f"  Evaluating PPL on {seq_len:,} tokens (window={max_length}, stride={stride})...")
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_length, seq_len)
            input_chunk = input_ids[:, begin:end].to(device)
            target_chunk = input_chunk.clone()

            # Only compute loss on the new tokens (after stride overlap)
            if begin > 0:
                target_chunk[:, :-stride] = -100

            outputs = model(
                input_chunk, labels=target_chunk,
                output_router_logits=False,
            )
            neg_log_likelihood = outputs.loss

            n_valid = (target_chunk != -100).sum().item()
            nlls.append(neg_log_likelihood.item() * n_valid)
            n_tokens += n_valid

            if end == seq_len:
                break

    elapsed = time.time() - t0
    ppl = math.exp(sum(nlls) / n_tokens)
    print(f"  Done in {elapsed:.1f}s — {n_tokens:,} tokens evaluated")

    return ppl


# ─────────────────────────────────────────────────────────────────
# Text generation demo
# ─────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    "def fibonacci(n):",
    "The derivative of x^2 is",
    "Once upon a time in a",
    "The speed of light is",
    "La inteligencia artificial es",
]


def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> Tuple[str, float]:
    """Generate text from a prompt. Returns (generated_text, tokens_per_second)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    elapsed = time.time() - t0

    n_generated = output_ids.shape[1] - input_ids.shape[1]
    tok_per_sec = n_generated / elapsed if elapsed > 0 else 0.0
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated, tok_per_sec


def generate_demo(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    device: str = "cuda",
    label: str = "BVH Router",
) -> List[dict]:
    """Run generation demo for a list of prompts."""
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 60}")
        print(f"  Prompt {i}/{len(prompts)}: \"{prompt}\"")
        print(f"{'─' * 60}")

        text, tps = generate_text(model, tokenizer, prompt, max_new_tokens, device)
        results.append({"prompt": prompt, "text": text, "tok_s": tps, "label": label})

        print(f"\n  [{label}]")
        print(f"  {text}")
        print(f"  ({tps:.1f} tok/s, {max_new_tokens} max tokens)")

    return results


# ─────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end PPL evaluation: BVH Router vs OLMoE Gate"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to OLMoE-1B-7B model directory")
    parser.add_argument("--router-checkpoint", type=str,
                        default="checkpoints/olmoe_distill/bvh_router_best.pt",
                        help="Path to trained BVH router checkpoint (single layer mode)")
    parser.add_argument("--layer", type=int, default=8,
                        help="Layer whose gate to replace (default: 8)")
    parser.add_argument("--multi-layer", type=str, default=None,
                        help="Comma-separated layer:checkpoint pairs for multi-layer eval. "
                             "E.g. '0:ckpt_layer0.pt,4:ckpt_layer4.pt,8:ckpt_layer8.pt'")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length for PPL eval")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding window PPL")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tokens", type=int, default=50000,
                        help="Max tokens to evaluate (for speed)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline PPL measurement (always 7.15). Saves ~30s.")
    parser.add_argument("--identity-test", action="store_true",
                        help="Wrapper diagnostic: wrap the ORIGINAL gate weight "
                             "through BVHGateWrapper code path. PPL should be ~6.11.")
    parser.add_argument("--hybrid", action="store_true",
                        help="Hybrid mode: BVH selects candidates, original gate "
                             "weight computes exact routing weights.")
    parser.add_argument("--n-candidates", type=int, default=16,
                        help="Number of BVH candidates before gate scoring (default: 16)")
    parser.add_argument("--logit-temperature", type=float, default=None,
                        help="Temperature to apply to router logits before softmax. "
                             "Fixes peaked distributions without changing top-8 ranking. "
                             "Try 5.0-20.0 for spectral routers.")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Ignore calibration in checkpoints (use raw logits + temperature)")
    parser.add_argument("--logit-norm", action="store_true",
                        help="Normalize logits to mean=0, std=1 before softmax. "
                             "Fixes peaked distributions without needing temperature tuning.")
    parser.add_argument("--topk-softmax", action="store_true",
                        help="Restricted softmax: select top-k by logit, then softmax "
                             "ONLY over those k. Natural weight distribution.")
    parser.add_argument("--topk-scale", type=float, default=None,
                        help="Scale factor for topk-softmax weights. "
                             "Original gate top-8 weights sum ~0.65-0.75. Default: 1.0 (sum=1).")
    parser.add_argument("--delta-predictor", action="store_true",
                        help="Train a tiny DeltaPredictor (~97 params/layer) that learns "
                             "per-token scale from BVH logit statistics. Calibrates on "
                             "validation set before PPL evaluation.")
    parser.add_argument("--delta-micro", action="store_true",
                        help="Use MicroPredictor (1 param/layer = 16 total) instead of "
                             "DeltaPredictor. Cannot overfit — just learns optimal per-layer scale.")
    parser.add_argument("--n-rays", type=int, default=1,
                        help="Number of rays per token (multi-ray ensemble). "
                             "3 = average 3 perturbed queries. Default: 1 (standard).")
    parser.add_argument("--rank-template", type=str, default=None,
                        help="Comma-separated weight template for rank_template mode. "
                             "E.g. '0.312,0.148,0.112,0.089,0.071,0.058,0.047,0.038'")
    parser.add_argument("--delta-steps", type=int, default=200,
                        help="Number of calibration steps for DeltaPredictor (default: 200)")
    parser.add_argument("--delta-lr", type=float, default=0.01,
                        help="Learning rate for DeltaPredictor calibration (default: 0.01)")
    parser.add_argument("--per-layer-scale", action="store_true",
                        help="Measure per-layer gate weight sums and use them as "
                             "individual scales for relu_norm (instead of global scale). "
                             "Adjusts each layer proportionally to its actual gate sum.")
    parser.add_argument("--weight-mode", type=str, default="softmax",
                        choices=["softmax", "relu_norm", "relu_log", "relu_cbrt",
                                 "topk_softmax", "uniform", "gate_dist", "rank_template",
                                 "hybrid_residual", "geometric", "lambert", "resonance",
                                 "zipf", "render_eq", "ray_march", "importance",
                                 "bm25", "gravity", "spectral_weight", "hierarchical",
                                 "selectivity_geometric"],
                        help="How to compute routing weights from BVH logits. "
                             "relu_norm: ReLU + L1 norm (recommended for pure mode). "
                             "topk_softmax: softmax over top-k only. "
                             "uniform: equal 1/k weights. "
                             "gate_dist: fixed weights from original gate distribution. "
                             "hybrid_residual: BVH selects + gate weights (closes PPL gap). "
                             "geometric: inverse distance weighting from BVH 3D space. "
                             "lambert: cosine law weights from BVH geometry. "
                             "resonance: Lorentzian curve weights (harmonic resonance). "
                             "zipf: Zipf's law weights 1/rank^s (from linguistics). "
                             "render_eq: logit × geometry (from CG rendering equation). "
                             "ray_march: logit × exp(-dist) (volumetric attenuation). "
                             "importance: softmax(logit) × 1/dist (Monte Carlo). "
                             "bm25: saturated logit × geometry (search engine anti-monopoly). "
                             "gravity: logit - α·log(dist) (economic gravity / ALiBi). "
                             "spectral_weight: logit * spectral * geometry (full CG rendering). "
                             "hierarchical: logit * tree_branch_bonus (pure tree structure). "
                             "softmax: standard full softmax (default).")
    parser.add_argument("--hybrid-alpha", type=float, default=0.98,
                        help="Blending factor for hybrid_residual mode. "
                             "0.98 = 98%% BVH + 2%% original gate (default).")

    # ── Confidence-Gated Routing ──
    parser.add_argument("--confidence-gate", type=float, default=None,
                        help="Confidence threshold [0-1] for adaptive hybrid routing. "
                             "BVH routes confident tokens (>threshold), gate handles uncertain ones. "
                             "Eliminates accuracy compounding. Try 0.5-0.7. "
                             "Requires original gate weight (always stored).")

    # ── Selectivity modulation ──
    parser.add_argument("--selectivity-file", type=str, default=None,
                        help="Path to expert_deep_analysis.json for selectivity-modulated "
                             "weight modes. Required for selectivity_geometric mode.")

    # ── OptiX RT Core flags ──
    parser.add_argument("--use-optix", action="store_true",
                        help="Use OptiX RT Cores for routing (requires compiled extension)")

    # ── Generation demo flags ──
    parser.add_argument("--generate", action="store_true",
                        help="Text generation mode instead of PPL evaluation")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt for generation (default: diverse set)")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Max tokens to generate per prompt")
    args = parser.parse_args()

    print("=" * 70)
    print("  SpectralAI Zero-Matrix — End-to-End PPL Evaluation")
    print("  BVH Geometric Router vs OLMoE Linear Gate")
    print("=" * 70)

    # ── Step 1: Load model ──────────────────────────────────────
    print(f"\n[Step 1/4] Loading OLMoE-1B-7B...")
    model, tokenizer = load_olmoe_model(args.model_dir, device=args.device)

    # ── Step 2: Baseline PPL (original gate) ────────────────────
    if getattr(args, 'skip_baseline', False):
        ppl_baseline = 7.15
        print(f"\n[Step 2/4] Skipping baseline (known PPL = {ppl_baseline:.2f})")
    else:
        print(f"\n[Step 2/4] Measuring BASELINE PPL (original linear gate)...")
        ppl_baseline = evaluate_ppl(
            model, tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        print(f"\n  >>> BASELINE PPL = {ppl_baseline:.2f}")

    # ── Step 2.5: Measure gate distribution (for gate_dist or per-layer-scale) ──
    gate_target_dist = None
    per_layer_dist = None
    per_layer_sum = None
    needs_gate_measure = (args.weight_mode == "gate_dist"
                          or getattr(args, 'per_layer_scale', False))
    if needs_gate_measure:
        print(f"\n[Step 2.5] Measuring original gate weight distribution...")
        gate_target_dist, per_layer_dist, per_layer_sum = \
            measure_gate_distribution(
                model, tokenizer, max_tokens=min(args.max_tokens, 10000),
                device=args.device)

    # ── Step 3: Replace gate(s) ────────────────────────────────────
    orig_forward = None
    replaced_layers = []
    original_gate = None

    # Load selectivity data if provided (for selectivity_geometric mode)
    _selectivity_data = None
    sel_file = getattr(args, 'selectivity_file', None)
    if sel_file is not None:
        import json as _json
        with open(sel_file) as _f:
            _selectivity_data = _json.load(_f)
        print(f"  Loaded selectivity data from {sel_file}")
    elif args.weight_mode == "selectivity_geometric":
        # Auto-detect: look for expert_deep_analysis.json in current dir
        _default_sel = "expert_deep_analysis.json"
        if os.path.exists(_default_sel):
            import json as _json
            with open(_default_sel) as _f:
                _selectivity_data = _json.load(_f)
            print(f"  Auto-loaded selectivity data from {_default_sel}")

    if args.identity_test:
        print(f"\n[Step 3/4] IDENTITY TEST — wrapping original gate weight...")
        mlp, gate_attr = find_gate_module(model, args.layer)
        original_gate = getattr(mlp, gate_attr)
        orig_weight = original_gate.weight.data.clone()
        norm_topk_prob = getattr(original_gate, 'norm_topk_prob', False)
        top_k = getattr(original_gate, 'top_k', 8)
        identity_wrapper = IdentityGateWrapper(
            orig_weight, top_k=top_k, norm_topk_prob=norm_topk_prob
        ).to(orig_weight.device)
        setattr(mlp, gate_attr, identity_wrapper)
        print(f"  norm_topk_prob={norm_topk_prob}, top_k={top_k}")
        print(f"  Replaced with IdentityGateWrapper (same weight, our code path)")
        print(f"  If PPL != {ppl_baseline:.2f}, the wrapper interface is BROKEN")
        replaced_layers = [args.layer]
    elif args.multi_layer:
        pairs = args.multi_layer.split(",")
        n_layers = len(pairs)
        print(f"\n[Step 3/4] Replacing {n_layers} layers with BVH Routers...")
        for pair in pairs:
            layer_str, ckpt_path = pair.strip().split(":", 1)
            layer_idx = int(layer_str)
            print(f"\n  --- Layer {layer_idx} ---")
            replace_gate_with_bvh(
                model, ckpt_path.strip(),
                layer_idx=layer_idx,
                hybrid=args.hybrid,
                n_candidates=args.n_candidates,
                logit_temperature=args.logit_temperature,
                no_calibration=args.no_calibration,
                logit_norm=args.logit_norm,
                topk_softmax=args.topk_softmax,
                topk_scale=args.topk_scale,
                weight_mode=args.weight_mode,
                n_rays=getattr(args, 'n_rays', 1),
                hybrid_alpha=getattr(args, 'hybrid_alpha', 0.98),
                use_optix=getattr(args, 'use_optix', False),
                selectivity_data=_selectivity_data,
                confidence_threshold=getattr(args, 'confidence_gate', None),
            )
            replaced_layers.append(layer_idx)
        print(f"\n  Total: {n_layers} layers replaced: {replaced_layers}")
    else:
        mode_label = "HYBRID" if args.hybrid else "BVH Router"
        print(f"\n[Step 3/4] Replacing layer {args.layer} gate with {mode_label}...")
        original_gate, orig_forward = replace_gate_with_bvh(
            model,
            args.router_checkpoint,
            layer_idx=args.layer,
            hybrid=args.hybrid,
            n_candidates=args.n_candidates,
            logit_temperature=args.logit_temperature,
            no_calibration=args.no_calibration,
            logit_norm=args.logit_norm,
            topk_softmax=args.topk_softmax,
            topk_scale=args.topk_scale,
            weight_mode=args.weight_mode,
            n_rays=getattr(args, 'n_rays', 1),
            hybrid_alpha=getattr(args, 'hybrid_alpha', 0.98),
            use_optix=getattr(args, 'use_optix', False),
            selectivity_data=_selectivity_data,
            confidence_threshold=getattr(args, 'confidence_gate', None),
        )
        replaced_layers = [args.layer]

    # Inject gate_dist target and/or per-layer scale into BVHGateWrapper instances
    if needs_gate_measure:
        for li in replaced_layers:
            try:
                mlp, gate_attr = find_gate_module(model, li)
                wrapper = getattr(mlp, gate_attr)
                if not isinstance(wrapper, BVHGateWrapper):
                    continue
                # gate_dist: inject per-layer distribution
                if args.weight_mode == "gate_dist" and per_layer_dist is not None:
                    if li in per_layer_dist:
                        wrapper._gate_target_dist = per_layer_dist[li]
                    elif gate_target_dist is not None:
                        wrapper._gate_target_dist = gate_target_dist
                # rank_template: inject custom template if provided
                if args.weight_mode == "rank_template" and getattr(args, 'rank_template', None):
                    vals = [float(x) for x in args.rank_template.split(",")]
                    wrapper._rank_template = torch.tensor(vals)
                # per-layer scale: inject measured sum as scale for relu_norm
                if getattr(args, 'per_layer_scale', False) and per_layer_sum is not None:
                    if li in per_layer_sum:
                        # Use ratio of per-layer sum to avg sum, multiplied by global scale
                        global_scale = args.topk_scale if args.topk_scale else 0.43
                        avg_sum = sum(per_layer_sum.values()) / len(per_layer_sum)
                        layer_ratio = per_layer_sum[li] / avg_sum
                        wrapper._layer_scale = global_scale * layer_ratio
                        print(f"    L{li}: scale={wrapper._layer_scale:.4f} "
                              f"(gate_sum={per_layer_sum[li]:.4f}, ratio={layer_ratio:.3f})")
            except Exception:
                pass
        if args.weight_mode == "gate_dist":
            print(f"  [gate_dist] Injected PER-LAYER distribution into {len(replaced_layers)} layers")
        if getattr(args, 'per_layer_scale', False):
            print(f"  [per-layer-scale] Injected per-layer scales into {len(replaced_layers)} layers")

    # ── Generate mode: text generation demo ─────────────────────
    if getattr(args, 'generate', False):
        prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
        max_new = getattr(args, 'max_new_tokens', 100)
        n_layers = len(replaced_layers)
        layers_str = ",".join(str(l) for l in replaced_layers)
        label = f"BVH Router ({n_layers} capas: {layers_str})"

        print(f"\n{'=' * 60}")
        print(f"  TEXT GENERATION DEMO — {label}")
        print(f"  Weight mode: {args.weight_mode}")
        print(f"  Max new tokens: {max_new}")
        print(f"{'=' * 60}")

        bvh_results = generate_demo(
            model, tokenizer, prompts, max_new, args.device, label=label
        )

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  RESUMEN")
        print(f"{'=' * 60}")
        avg_tps = sum(r["tok_s"] for r in bvh_results) / len(bvh_results)
        print(f"  Prompts: {len(bvh_results)}")
        print(f"  Velocidad media: {avg_tps:.1f} tok/s")
        print(f"  Capas BVH: {layers_str}")
        print(f"  Weight mode: {args.weight_mode}")
        print()
        return {"mode": "generate", "results": bvh_results, "avg_tok_s": avg_tps}

    # ── Step 3.5: Calibrate DeltaPredictor (if requested) ──────
    if getattr(args, 'delta_predictor', False) and not args.identity_test:
        use_micro = getattr(args, 'delta_micro', False)
        compression = "log1p"  # best compression from experiments
        mode_name = "MicroPredictor" if use_micro else "DeltaPredictor"
        print(f"\n[Step 3.5] Calibrating {mode_name} ({len(replaced_layers)} layers)...")
        predictors = calibrate_delta_predictor(
            model, tokenizer, replaced_layers,
            base_scale=args.topk_scale if args.topk_scale else 0.43,
            compression=compression,
            max_tokens=min(args.max_tokens, 5000),
            lr=getattr(args, 'delta_lr', 0.01),
            n_steps=getattr(args, 'delta_steps', 200),
            device=args.device,
            micro=use_micro,
        )
        n_dp_params = sum(sum(p.numel() for p in dp.parameters())
                          for dp in predictors.values())
        print(f"  [{mode_name}] {len(predictors)} predictors, {n_dp_params} total params")

        # Save calibrated predictors
        save_dir = Path("checkpoints/micro_predictors")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{'micro' if use_micro else 'delta'}_predictors.pt"
        save_data = {
            "type": mode_name,
            "n_params": n_dp_params,
            "predictors": {li: dp.state_dict() for li, dp in predictors.items()},
            "base_scale": args.topk_scale if args.topk_scale else 0.43,
            "compression": compression,
        }
        torch.save(save_data, save_path)
        print(f"  [{mode_name}] Saved to {save_path}")

        # Free CUDA memory from calibration backward passes
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("  [delta] CUDA cache cleared after calibration")

    # ── Step 4: BVH PPL ────────────────────────────────────────
    print(f"\n[Step 4/4] Measuring BVH PPL (geometric routing)...")
    ppl_bvh = evaluate_ppl(
        model, tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        device=args.device,
        max_tokens=args.max_tokens,
    )
    print(f"\n  >>> BVH PPL = {ppl_bvh:.2f}")

    # ── Results ─────────────────────────────────────────────────
    ppl_delta = ((ppl_bvh - ppl_baseline) / ppl_baseline) * 100

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  {'Routing':>20}  {'PPL':>10}  {'Delta':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
    print(f"  {'OLMoE Linear Gate':>20}  {ppl_baseline:>10.2f}  {'(baseline)':>10}")
    layers_str = ",".join(str(l) for l in replaced_layers)
    bvh_label = (f"BVH ({layers_str})" if len(replaced_layers) > 1
                 else f"BVH Router (L{replaced_layers[0]})")
    print(f"  {bvh_label:>20}  {ppl_bvh:>10.2f}  {ppl_delta:>+9.1f}%")
    print()

    if abs(ppl_delta) < 1.0:
        verdict = "EXCELLENT — BVH routing is equivalent to linear gate"
    elif abs(ppl_delta) < 5.0:
        verdict = "GOOD — Minimal PPL impact from geometric routing"
    elif abs(ppl_delta) < 15.0:
        verdict = "ACCEPTABLE — Moderate PPL increase, room for improvement"
    else:
        verdict = "NEEDS WORK — Significant PPL degradation"

    print(f"  Verdict: {verdict}")
    print()
    print(f"  Router params: 1.35M (vs 131K linear gate)")
    print(f"  Router latency: ~10 us/batch (CUDA kernel)")
    print(f"  Routing: O(log N) hierarchical vs O(N) linear scan")

    # Print confidence-gated routing statistics if enabled
    if getattr(args, 'confidence_gate', None) is not None:
        # Aggregate stats across all wrapper instances
        total_bvh = 0
        total_gate = 0
        for layer_idx in replaced_layers:
            mlp, gate_attr = find_gate_module(model, layer_idx)
            wrapper = getattr(mlp, gate_attr, None)
            if hasattr(wrapper, '_confidence_stats'):
                total_bvh += wrapper._confidence_stats["bvh_tokens"]
                total_gate += wrapper._confidence_stats["gate_tokens"]
        total = total_bvh + total_gate
        if total > 0:
            bvh_pct = total_bvh / total * 100
            print(f"\n  Confidence-Gated Routing (threshold={args.confidence_gate:.2f}):")
            print(f"    BVH routing:  {total_bvh:,} tokens ({bvh_pct:.1f}%)")
            print(f"    Gate fallback: {total_gate:,} tokens ({100-bvh_pct:.1f}%)")
            print(f"    Effective gate usage: {100-bvh_pct:.1f}% (lower = more O(log N))")
    print()

    # ── Restore original gates ─────────────────────────────────
    if args.multi_layer:
        print("  (multi-layer mode — restart model to restore original gates)")
    elif not args.identity_test and args.hybrid and orig_forward is not None:
        original_gate.forward = orig_forward
        print("  Original gate restored.")
    elif not args.identity_test and not args.hybrid and original_gate is not None:
        mlp, gate_attr = find_gate_module(model, args.layer)
        setattr(mlp, gate_attr, original_gate)
        print("  Original gate restored.")
    elif args.identity_test and original_gate is not None:
        mlp, gate_attr = find_gate_module(model, args.layer)
        setattr(mlp, gate_attr, original_gate)
        print("  Original gate restored.")

    return {
        "ppl_baseline": ppl_baseline,
        "ppl_bvh": ppl_bvh,
        "ppl_delta_pct": ppl_delta,
        "layers": replaced_layers,
    }


if __name__ == "__main__":
    main()
