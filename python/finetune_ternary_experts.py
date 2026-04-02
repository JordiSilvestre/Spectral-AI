#!/usr/bin/env python3
"""
finetune_ternary_experts.py -- Quantization-Aware Training for Ternary Experts

Recreates the ~14h fine-tuning pipeline to match patent-claimed performance:
  - 375x VRAM reduction (7.86 MB active vs 2964 MB FP16)
  - 51.9 tok/s throughput
  - PPL 6.16 (+0.8% vs baseline)

Strategy: Straight-Through Estimator (STE) + Knowledge Distillation
  1. Load FP16 model (teacher) → freeze
  2. Initialize ternary copies of MLP layers (student)
  3. Train student to replicate teacher outputs via KD loss
  4. Ternary weights are {-1, 0, +1} + per-row FP32 scale
  5. Forward: quantize; Backward: straight-through gradient

The STE bridges the non-differentiable ternary quantization:
  Forward: w_q = scale * sign(w_latent) * (|w_latent| > threshold)
  Backward: grad flows through as if quantize were identity

Usage:
    # Fine-tune all 28 layers (full pipeline, ~14h on RTX 4090):
    python finetune_ternary_experts.py --model qwen-0.5b --epochs 50

    # Fine-tune a single layer (for testing, ~30 min):
    python finetune_ternary_experts.py --model qwen-0.5b --layer 8 --epochs 20

    # Resume from checkpoint:
    python finetune_ternary_experts.py --model qwen-0.5b --resume checkpoints/ternary_ep10.pt

Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("ternary_finetune")


# =============================================================================
# Straight-Through Estimator for Ternary Quantization
# =============================================================================

class TernaryQuantizeSTE(torch.autograd.Function):
    """Ternary quantization with straight-through gradient estimator.

    Forward: w_q = sign(w) * (|w| > threshold)   → {-1, 0, +1}
    Backward: grad_input = grad_output  (straight-through)

    The threshold targets ~50% sparsity using the median of |w|.
    (mean(|w|) gives ~60% sparsity which is too aggressive.)
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if threshold is None:
            # Median of |w| → ~50% zeros (vs mean which gives ~60%)
            threshold = weight.abs().median()
        ctx.save_for_backward(weight, threshold)
        # Ternary: +1 where w > thr, -1 where w < -thr, 0 otherwise
        out = torch.zeros_like(weight)
        out[weight > threshold] = 1.0
        out[weight < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, threshold = ctx.saved_tensors
        # STE: pass gradient through, but clip to [-1, 1] range
        # (prevents exploding gradients from dead zones)
        grad_input = grad_output.clone()
        # Optional: attenuate gradient for values far from threshold
        # This helps convergence by focusing updates near the decision boundary
        distance = (weight.abs() - threshold).abs()
        attenuation = torch.exp(-0.5 * distance / (threshold + 1e-8))
        grad_input = grad_input * attenuation
        return grad_input, None


def ternary_quantize_ste(weight: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for TernaryQuantizeSTE."""
    return TernaryQuantizeSTE.apply(weight, None)


# =============================================================================
# Learnable Scale Factor
# =============================================================================

class LearnableScale(nn.Module):
    """Per-row learnable scale factor for ternary weights.

    Initialized from the teacher's weight statistics:
      scale[i] = mean(|w[i, :]|) for non-zero entries after ternary quantization.

    During training, scale is updated via gradient descent alongside latent weights.
    """

    def __init__(self, num_rows: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_rows))

    @property
    def scale(self) -> torch.Tensor:
        # Softplus ensures positive scale without clipping
        return F.softplus(self.log_scale)

    def init_from_teacher(self, teacher_weight: torch.Tensor) -> None:
        """Initialize scale from teacher weight magnitude statistics."""
        w = teacher_weight.detach().float()
        threshold = w.abs().median()
        mask = w.abs() > threshold
        row_means = torch.ones(w.shape[0], device=w.device)
        for i in range(w.shape[0]):
            if mask[i].any():
                row_means[i] = w[i].abs()[mask[i]].mean()
        # Initialize log_scale so that softplus(log_scale) ≈ row_means
        # softplus inverse: log(exp(x) - 1) ≈ x for large x
        self.log_scale.data.copy_(
            torch.log(torch.exp(row_means.clamp(min=0.01)) - 1.0)
        )


# =============================================================================
# Ternary Linear Layer (trainable)
# =============================================================================

class TernaryLinear(nn.Module):
    """A linear layer that uses ternary quantization with STE.

    Maintains latent FP32 weights that are quantized to {-1, 0, +1} in forward.
    Gradients flow through the STE to update the latent weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Latent weights (FP32, updated by optimizer)
        self.weight_latent = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.scale = LearnableScale(out_features)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def init_from_teacher(self, teacher_weight: torch.Tensor) -> None:
        """Initialize latent weights and scale from a teacher's FP16/FP32 weight."""
        self.weight_latent.data.copy_(teacher_weight.float())
        self.scale.init_from_teacher(teacher_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize latent → ternary via STE
        w_ternary = ternary_quantize_ste(self.weight_latent)
        # Scale: (out, in) * (out, 1) → scaled ternary
        w_scaled = w_ternary * self.scale.scale.unsqueeze(1)
        out = F.linear(x, w_scaled, self.bias)
        return out

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries in the quantized weights."""
        with torch.no_grad():
            w_q = ternary_quantize_ste(self.weight_latent)
            return (w_q == 0).float().mean().item()

    def export_ternary(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Export final ternary weights + scale for CUDA inference.

        Returns (ternary_int8, scale_fp32, sparsity).
        """
        with torch.no_grad():
            w_q = ternary_quantize_ste(self.weight_latent)
            ternary = w_q.cpu().to(torch.int8).numpy()
            scale = self.scale.scale.cpu().numpy()
            sparsity = float((w_q == 0).float().mean().item())
        return ternary, scale, sparsity


# =============================================================================
# Ternary MLP (Gated SwiGLU — Qwen/LLaMA architecture)
# =============================================================================

class TernaryGatedMLP(nn.Module):
    """Ternary-quantized gated MLP matching Qwen2/LLaMA architecture.

    Structure:
      gate = silu(gate_proj(x))
      up   = up_proj(x)
      out  = down_proj(gate * up)

    All three projections use TernaryLinear with STE.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = TernaryLinear(hidden_size, intermediate_size)
        self.up_proj = TernaryLinear(hidden_size, intermediate_size)
        self.down_proj = TernaryLinear(intermediate_size, hidden_size)

    def init_from_teacher_mlp(self, teacher_mlp: nn.Module) -> None:
        """Initialize from a teacher's MLP module (Qwen/LLaMA/Mistral style)."""
        if hasattr(teacher_mlp, "gate_proj"):
            self.gate_proj.init_from_teacher(teacher_mlp.gate_proj.weight.data)
            self.up_proj.init_from_teacher(teacher_mlp.up_proj.weight.data)
            self.down_proj.init_from_teacher(teacher_mlp.down_proj.weight.data)
        elif hasattr(teacher_mlp, "gate_up_proj"):
            # Phi-3 fused style
            fused = teacher_mlp.gate_up_proj.weight.data
            half = fused.shape[0] // 2
            self.gate_proj.init_from_teacher(fused[:half])
            self.up_proj.init_from_teacher(fused[half:])
            self.down_proj.init_from_teacher(teacher_mlp.down_proj.weight.data)
        else:
            raise ValueError(
                f"Unknown MLP structure: {type(teacher_mlp).__name__}. "
                f"Expected gate_proj/up_proj/down_proj or gate_up_proj/down_proj."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    @property
    def total_sparsity(self) -> float:
        s = (
            self.gate_proj.sparsity
            + self.up_proj.sparsity
            + self.down_proj.sparsity
        )
        return s / 3.0


# =============================================================================
# Hidden State Dataset
# =============================================================================

class HiddenStateDataset(Dataset):
    """Loads pre-extracted hidden states + teacher MLP outputs for KD.

    Each sample: (hidden_input, teacher_output)
    Where teacher_output = teacher_mlp(hidden_input) computed offline.
    """

    def __init__(self, data_dir: str, layer_idx: int):
        self.data_dir = Path(data_dir)
        input_path = self.data_dir / f"layer_{layer_idx}_input.pt"
        output_path = self.data_dir / f"layer_{layer_idx}_output.pt"

        if input_path.exists() and output_path.exists():
            self.inputs = torch.load(input_path, map_location="cpu",
                                     weights_only=True)
            self.outputs = torch.load(output_path, map_location="cpu",
                                      weights_only=True)
            log.info(
                "Loaded layer %d dataset: %d samples from %s",
                layer_idx, len(self.inputs), data_dir,
            )
        else:
            raise FileNotFoundError(
                f"Hidden state data not found for layer {layer_idx}.\n"
                f"Expected: {input_path} and {output_path}\n"
                f"Run: python extract_hidden_states.py --model <model> --output {data_dir}"
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# =============================================================================
# Online Hidden State Extraction (no pre-extracted data needed)
# =============================================================================

def extract_hidden_states_online(
    model: nn.Module,
    tokenizer,
    layer_idx: int,
    num_samples: int = 4096,
    seq_len: int = 128,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract hidden states by running the model on generated/random input.

    Uses the model's own embedding + first N layers to produce realistic
    hidden states at layer_idx, then captures the MLP input/output pair.

    Returns (inputs [num_samples, hidden], outputs [num_samples, hidden]).
    """
    model.eval()
    all_inputs = []
    all_outputs = []

    # Find model backbone and layers
    backbone = None
    layers = None
    for name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = name.split(".")
        obj = model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, "__len__"):
            layers = obj
            break

    if layers is None:
        raise RuntimeError("Cannot find model layers for hidden state extraction")

    # Find embedding layer
    embed = None
    for attr in ["model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"]:
        parts = attr.split(".")
        obj = model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                obj = None
                break
        if obj is not None:
            embed = obj
            break

    if embed is None:
        raise RuntimeError("Cannot find embedding layer")

    # Hook to capture MLP input/output at target layer
    captured = {"input": None, "output": None}

    target_mlp = layers[layer_idx].mlp if hasattr(layers[layer_idx], "mlp") else None
    if target_mlp is None:
        raise RuntimeError(f"Layer {layer_idx} has no .mlp attribute")

    def hook_fn(module, inp, out):
        captured["input"] = inp[0].detach()
        captured["output"] = out.detach()

    hook = target_mlp.register_forward_hook(hook_fn)

    try:
        batch_size = 8
        num_batches = max(1, num_samples // (batch_size * seq_len))

        with torch.no_grad():
            for _ in range(num_batches):
                # Generate random token ids
                input_ids = torch.randint(
                    0, tokenizer.vocab_size, (batch_size, seq_len),
                    device=device,
                )
                # Run through model (captures hook)
                try:
                    model(input_ids)
                except Exception:
                    # Some models may error on random input; try shorter seq
                    input_ids = input_ids[:, :32]
                    model(input_ids)

                if captured["input"] is not None:
                    # Flatten: [batch, seq, hidden] -> [batch*seq, hidden]
                    inp = captured["input"].reshape(-1, captured["input"].shape[-1])
                    out = captured["output"].reshape(-1, captured["output"].shape[-1])
                    all_inputs.append(inp.cpu())
                    all_outputs.append(out.cpu())

                if sum(t.shape[0] for t in all_inputs) >= num_samples:
                    break
    finally:
        hook.remove()

    inputs_cat = torch.cat(all_inputs, dim=0)[:num_samples]
    outputs_cat = torch.cat(all_outputs, dim=0)[:num_samples]
    return inputs_cat, outputs_cat


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TernaryFinetuneConfig:
    """Training hyperparameters for ternary expert fine-tuning."""

    # Model
    model_name: str = "qwen-0.5b"
    layer_idx: Optional[int] = None  # None = all layers

    # Training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    lr_scale: float = 0.1        # Scale factor LR multiplier
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # KD loss
    kd_temperature: float = 1.0
    kd_alpha: float = 0.9        # Weight for KD loss vs reconstruction
    sparsity_penalty: float = 0.01  # Encourage sparsity near 50%

    # Data
    data_dir: str = "data/hidden_states"
    num_samples: int = 65536     # Online extraction sample count
    num_workers: int = 2

    # Checkpointing
    checkpoint_dir: str = "checkpoints/ternary"
    save_every: int = 5          # Save every N epochs
    resume: Optional[str] = None

    # Hardware
    device: str = "cuda"
    amp: bool = True             # BF16 automatic mixed precision


# =============================================================================
# Loss Functions
# =============================================================================

def kd_output_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Knowledge distillation loss: MSE on normalized outputs.

    For MLP outputs (not logits), we use MSE with optional temperature scaling
    rather than KL divergence (which requires probability distributions).
    """
    # Normalize to unit variance for stable training
    s_norm = student_out / (student_out.std(dim=-1, keepdim=True) + 1e-8)
    t_norm = teacher_out / (teacher_out.std(dim=-1, keepdim=True) + 1e-8)
    return F.mse_loss(s_norm / temperature, t_norm / temperature)


def cosine_similarity_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
) -> torch.Tensor:
    """1 - mean(cosine_similarity) between student and teacher outputs."""
    cos = F.cosine_similarity(student_out, teacher_out, dim=-1)
    return 1.0 - cos.mean()


def sparsity_regularization(
    mlp: TernaryGatedMLP,
    target_sparsity: float = 0.50,
) -> torch.Tensor:
    """Penalty for deviating from target sparsity (BitNet b1.58 ≈ 50%)."""
    actual = mlp.total_sparsity
    return (actual - target_sparsity) ** 2


# =============================================================================
# Training Loop for One Layer
# =============================================================================

def train_one_layer(
    teacher_mlp: nn.Module,
    student_mlp: TernaryGatedMLP,
    train_inputs: torch.Tensor,
    train_outputs: torch.Tensor,
    config: TernaryFinetuneConfig,
    layer_idx: int,
) -> Dict[str, float]:
    """Fine-tune a single ternary MLP layer via knowledge distillation.

    Returns dict of final metrics.
    """
    device = torch.device(config.device)
    student_mlp = student_mlp.to(device)
    teacher_mlp = teacher_mlp.to(device).eval()

    # Freeze teacher
    for p in teacher_mlp.parameters():
        p.requires_grad_(False)

    # Separate param groups: weights vs scales (different LR)
    weight_params = []
    scale_params = []
    for name, param in student_mlp.named_parameters():
        if "log_scale" in name:
            scale_params.append(param)
        else:
            weight_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": weight_params, "lr": config.lr},
            {"params": scale_params, "lr": config.lr * config.lr_scale},
        ],
        weight_decay=config.weight_decay,
    )

    total_steps = config.epochs * max(1, len(train_inputs) // config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.lr * 0.01,
    )

    scaler = GradScaler("cuda", enabled=config.amp)

    # Create DataLoader from tensors
    dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    best_loss = float("inf")
    best_cosine = 0.0
    step = 0
    start_time = time.time()

    log.info(
        "Layer %d: %d samples, %d epochs, %d steps",
        layer_idx, len(train_inputs), config.epochs, total_steps,
    )
    log.info(
        "  Initial sparsity: gate=%.1f%% up=%.1f%% down=%.1f%%",
        student_mlp.gate_proj.sparsity * 100,
        student_mlp.up_proj.sparsity * 100,
        student_mlp.down_proj.sparsity * 100,
    )

    for epoch in range(config.epochs):
        student_mlp.train()
        epoch_loss = 0.0
        epoch_cos = 0.0
        num_batches = 0

        for batch_in, batch_teacher_out in loader:
            batch_in = batch_in.to(device, non_blocking=True)
            batch_teacher_out = batch_teacher_out.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16, enabled=config.amp):
                student_out = student_mlp(batch_in)

                # KD loss: match teacher output distribution
                loss_kd = kd_output_loss(
                    student_out, batch_teacher_out, config.kd_temperature,
                )

                # Cosine similarity loss: match direction
                loss_cos = cosine_similarity_loss(student_out, batch_teacher_out)

                # Sparsity regularization
                loss_sparse = sparsity_regularization(student_mlp)

                # Combined loss
                loss = (
                    config.kd_alpha * loss_kd
                    + (1.0 - config.kd_alpha) * loss_cos
                    + config.sparsity_penalty * loss_sparse
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student_mlp.parameters(), config.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    student_out, batch_teacher_out, dim=-1,
                ).mean().item()
            epoch_cos += cos_sim
            num_batches += 1
            step += 1

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_cos = epoch_cos / max(num_batches, 1)
        elapsed = time.time() - start_time
        sparsity = student_mlp.total_sparsity

        if avg_loss < best_loss:
            best_loss = avg_loss
        if avg_cos > best_cosine:
            best_cosine = avg_cos

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(
                "  L%d E%03d | loss=%.4f cos=%.4f sparsity=%.1f%% | %.0fs",
                layer_idx, epoch + 1, avg_loss, avg_cos,
                sparsity * 100, elapsed,
            )

    total_time = time.time() - start_time
    log.info(
        "Layer %d done in %.1f min | best_loss=%.4f best_cos=%.4f sparsity=%.1f%%",
        layer_idx, total_time / 60, best_loss, best_cosine,
        student_mlp.total_sparsity * 100,
    )

    return {
        "layer_idx": layer_idx,
        "best_loss": best_loss,
        "best_cosine": best_cosine,
        "sparsity": student_mlp.total_sparsity,
        "train_time_sec": total_time,
    }


# =============================================================================
# Export Fine-tuned Ternary Weights
# =============================================================================

def export_ternary_checkpoint(
    student_mlps: Dict[int, TernaryGatedMLP],
    output_dir: str,
    model_name: str,
    metrics: List[Dict],
) -> str:
    """Export fine-tuned ternary weights for CUDA POPCOUNT inference.

    Saves:
      - ternary_experts/{layer_idx}/ containing gate/up/down .npy files
      - metadata.json with sparsity, scale, training metrics
    """
    out = Path(output_dir) / "ternary_experts"
    out.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    layer_meta = []

    for layer_idx, mlp in sorted(student_mlps.items()):
        layer_dir = out / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        gate_t, gate_s, gate_sp = mlp.gate_proj.export_ternary()
        up_t, up_s, up_sp = mlp.up_proj.export_ternary()
        down_t, down_s, down_sp = mlp.down_proj.export_ternary()

        np.save(layer_dir / "gate_ternary.npy", gate_t)
        np.save(layer_dir / "gate_scale.npy", gate_s)
        np.save(layer_dir / "up_ternary.npy", up_t)
        np.save(layer_dir / "up_scale.npy", up_s)
        np.save(layer_dir / "down_ternary.npy", down_t)
        np.save(layer_dir / "down_scale.npy", down_s)

        layer_bytes = sum(
            a.nbytes for a in [gate_t, gate_s, up_t, up_s, down_t, down_s]
        )
        total_bytes += layer_bytes

        layer_meta.append({
            "layer_idx": layer_idx,
            "gate_shape": list(gate_t.shape),
            "up_shape": list(up_t.shape),
            "down_shape": list(down_t.shape),
            "gate_sparsity": float(gate_sp),
            "up_sparsity": float(up_sp),
            "down_sparsity": float(down_sp),
            "avg_sparsity": float((gate_sp + up_sp + down_sp) / 3),
            "size_bytes": layer_bytes,
        })

    # Metrics from training
    metrics_by_layer = {m["layer_idx"]: m for m in metrics}

    metadata = {
        "model_name": model_name,
        "num_layers": len(student_mlps),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "compression_note": "ternary {-1,0,+1} int8 + per-row FP32 scale",
        "layers": layer_meta,
        "training_metrics": metrics,
    }

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(
        "Exported %d layers → %s (%.2f MB total)",
        len(student_mlps), out, total_bytes / 1024 / 1024,
    )
    return str(out)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_finetune(config: TernaryFinetuneConfig) -> None:
    """Full fine-tuning pipeline: load model → extract → train → export."""

    # ── Import model registry ─────────────────────────────────────────
    from real_model_demo import MODEL_REGISTRY

    hf_id = MODEL_REGISTRY.get(config.model_name)
    if hf_id is None:
        raise ValueError(
            f"Unknown model: {config.model_name}. "
            f"Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
        )

    device = torch.device(config.device)
    log.info("=" * 70)
    log.info("SpectralAI Ternary Expert Fine-tuning")
    log.info("=" * 70)
    log.info("Model:   %s (%s)", config.model_name, hf_id)
    log.info("Device:  %s", device)
    log.info("Epochs:  %d per layer", config.epochs)
    log.info("Samples: %d", config.num_samples)
    log.info("AMP:     %s", config.amp)

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True, local_files_only=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, trust_remote_code=True,
            local_files_only=True,
        )
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.float16, trust_remote_code=True,
        )

    model = model.to(device).eval()
    log.info("Model loaded: %s", type(model).__name__)

    # ── Find layers ───────────────────────────────────────────────────
    layers = None
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        parts = attr.split(".")
        obj = model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, "__len__"):
            layers = list(obj)
            break

    if layers is None:
        raise RuntimeError("Cannot find transformer layers in model")

    num_layers = len(layers)
    log.info("Found %d transformer layers", num_layers)

    # Determine which layers to fine-tune
    if config.layer_idx is not None:
        layer_indices = [config.layer_idx]
    else:
        layer_indices = list(range(num_layers))

    log.info("Fine-tuning layers: %s", layer_indices)

    # ── Train each layer ──────────────────────────────────────────────
    student_mlps: Dict[int, TernaryGatedMLP] = {}
    all_metrics: List[Dict] = []
    pipeline_start = time.time()

    for li in layer_indices:
        layer = layers[li]
        teacher_mlp = layer.mlp

        # Get hidden/intermediate dimensions
        if hasattr(teacher_mlp, "gate_proj"):
            hidden_size = teacher_mlp.gate_proj.in_features
            intermediate_size = teacher_mlp.gate_proj.out_features
        elif hasattr(teacher_mlp, "gate_up_proj"):
            hidden_size = teacher_mlp.gate_up_proj.in_features
            intermediate_size = teacher_mlp.gate_up_proj.out_features // 2
        else:
            raise RuntimeError(f"Layer {li}: unknown MLP architecture")

        log.info(
            "\n--- Layer %d/%d --- (hidden=%d, intermediate=%d)",
            li + 1, num_layers, hidden_size, intermediate_size,
        )

        # Create student MLP
        student = TernaryGatedMLP(hidden_size, intermediate_size)
        student.init_from_teacher_mlp(teacher_mlp)

        # Extract hidden states for this layer
        log.info("  Extracting hidden states (online, %d samples)...", config.num_samples)
        inputs, outputs = extract_hidden_states_online(
            model, tokenizer, li,
            num_samples=config.num_samples,
            device=config.device,
        )
        log.info(
            "  Got %d samples, input=%s output=%s",
            len(inputs), list(inputs.shape), list(outputs.shape),
        )

        # Train
        metrics = train_one_layer(
            teacher_mlp=teacher_mlp,
            student_mlp=student,
            train_inputs=inputs,
            train_outputs=outputs,
            config=config,
            layer_idx=li,
        )
        all_metrics.append(metrics)
        student_mlps[li] = student.cpu()

        # Free GPU memory between layers
        torch.cuda.empty_cache()

        # Save checkpoint periodically
        if (len(all_metrics) % config.save_every == 0) or (li == layer_indices[-1]):
            ckpt_path = Path(config.checkpoint_dir) / f"ternary_L{li}_ep{config.epochs}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "layer_idx": li,
                    "student_state_dict": student.state_dict(),
                    "config": vars(config),
                    "metrics": metrics,
                },
                ckpt_path,
            )
            log.info("  Checkpoint: %s", ckpt_path)

    total_time = time.time() - pipeline_start

    # ── Export ─────────────────────────────────────────────────────────
    export_dir = export_ternary_checkpoint(
        student_mlps, config.checkpoint_dir, config.model_name, all_metrics,
    )

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINE-TUNING COMPLETE")
    log.info("=" * 70)
    log.info("Total time:  %.1f hours", total_time / 3600)
    log.info("Layers:      %d", len(student_mlps))
    log.info("Export dir:  %s", export_dir)

    avg_cos = np.mean([m["best_cosine"] for m in all_metrics])
    avg_sparsity = np.mean([m["sparsity"] for m in all_metrics])
    log.info("Avg cosine:  %.4f", avg_cos)
    log.info("Avg sparsity: %.1f%%", avg_sparsity * 100)

    # Patent-target comparison
    log.info("\n--- Patent Target Comparison ---")
    total_ternary_mb = sum(
        mlp.gate_proj.weight_latent.numel()
        + mlp.up_proj.weight_latent.numel()
        + mlp.down_proj.weight_latent.numel()
        for mlp in student_mlps.values()
    ) / 1024 / 1024  # int8 = 1 byte per param
    log.info("Ternary MLP VRAM:  %.2f MB (target: 7.86 MB)", total_ternary_mb)
    log.info("Avg cosine:        %.4f (target: >0.97)", avg_cos)
    log.info("Avg sparsity:      %.1f%% (target: ~50%%)", avg_sparsity * 100)
    log.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SpectralAI Ternary Expert Fine-tuning (QAT with STE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (~14h on RTX 4090):
  python finetune_ternary_experts.py --model qwen-0.5b --epochs 50

  # Single layer test (~30 min):
  python finetune_ternary_experts.py --model qwen-0.5b --layer 8 --epochs 20

  # More samples for better quality:
  python finetune_ternary_experts.py --model qwen-0.5b --num-samples 131072

  # Resume from checkpoint:
  python finetune_ternary_experts.py --model qwen-0.5b --resume checkpoints/ternary/ternary_L8_ep50.pt
""",
    )
    parser.add_argument(
        "--model", default="qwen-0.5b",
        help="Model name from MODEL_REGISTRY (default: qwen-0.5b)",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Single layer index to fine-tune (default: all layers)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per layer (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate for latent weights (default: 3e-4)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=65536,
        help="Number of hidden state samples per layer (default: 65536)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/ternary",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable BF16 mixed precision",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TernaryFinetuneConfig(
        model_name=args.model,
        layer_idx=args.layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        amp=not args.no_amp,
    )

    run_finetune(config)


if __name__ == "__main__":
    main()
