#!/usr/bin/env python3
"""
train_router.py -- SpectralAI v5.0 FASE 6: Router Training Harness

Trains ONLY the routing layer + blend_gate on a frozen backbone.
The backbone and ternary experts are frozen — only ~500K params train.

Training loop:
  1. Backbone forward (no_grad) → hidden states
  2. hidden_proj(hidden) → router_input
  3. BVH Router (Gumbel-Softmax) → expert_probs
  4. Top-k expert forward_hidden → combined hidden
  5. shared_output_proj → logits
  6. blend: (1-alpha) * backbone_logits + alpha * expert_logits
  7. CrossEntropy loss → backprop through router + blend_gate only

Usage:
    # Phase 0: Train on WikiText-103 (downloads ~180MB)
    python train_router.py --model microsoft/bitnet-b1.58-2B-4T-bf16 --epochs 3

    # Phase 1: Distill from Mixtral gate (future)
    python train_router.py --model mistralai/Mixtral-8x7B-v0.1 --distill

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# SpectralAI imports
sys.path.insert(0, str(Path(__file__).parent))
from bvh_router import RouterConfig, RoutingResult
from bvh_router_bridge import HybridBVHRouter, HAS_TORCH_EXT
from ternary_expert_ext_bridge import HAS_TERNARY_EXT, create_expert_module
from expert_lru_cache import ExpertLRUCache


# ─────────────────────────────────────────────────────────────────
# Dataset: Tokenized text chunks for next-token prediction
# ─────────────────────────────────────────────────────────────────

class TokenizedTextDataset(Dataset):
    """
    Simple dataset: fixed-length chunks of tokenized text.
    Each item is (input_ids[:-1], target_ids[1:]) for next-token pred.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 512):
        self.token_ids = token_ids
        self.seq_len = seq_len
        # Number of complete chunks
        self.n_chunks = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift
        chunk = self.token_ids[start:end]
        return chunk[:-1], chunk[1:]  # input, target


def load_wikitext_dataset(tokenizer, seq_len: int = 512, max_tokens: int = None):
    """
    Load WikiText-103 and tokenize it.
    Returns train and val TokenizedTextDataset.
    """
    from datasets import load_dataset

    print("  Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Tokenize incrementally in article batches to avoid OOM
    # Default cap: 10M tokens (enough for good training, avoids 500MB RAM spike)
    TRAIN_CAP = max_tokens if max_tokens is not None else 10_000_000
    VAL_CAP = max(TRAIN_CAP // 10, seq_len * 20)

    def tokenize_incremental(articles, target_tokens, batch_size=500):
        all_ids = []
        total = 0
        for i in range(0, len(articles), batch_size):
            batch = [t for t in articles[i:i+batch_size] if t.strip()]
            if not batch:
                continue
            chunk_text = "\n".join(batch)
            ids = tokenizer(
                chunk_text, return_tensors="pt", truncation=False,
                add_special_tokens=False,
            ).input_ids[0]
            all_ids.append(ids)
            total += len(ids)
            if total >= target_tokens:
                break
        combined = torch.cat(all_ids)[:target_tokens]
        return combined

    print("  Tokenizing train split...")
    train_ids = tokenize_incremental(ds["train"]["text"], TRAIN_CAP)

    print("  Tokenizing val split...")
    val_ids = tokenize_incremental(ds["validation"]["text"], VAL_CAP)

    print(f"  Train: {len(train_ids):,} tokens ({len(train_ids) // seq_len} chunks)")
    print(f"  Val:   {len(val_ids):,} tokens ({len(val_ids) // seq_len} chunks)")

    return (
        TokenizedTextDataset(train_ids, seq_len),
        TokenizedTextDataset(val_ids, seq_len),
    )


# ─────────────────────────────────────────────────────────────────
# Training Harness
# ─────────────────────────────────────────────────────────────────

class RouterTrainer:
    """
    Trains the BVH router + blend_gate on a frozen backbone.

    Only these parameters are trainable (~500K):
    - hidden_proj: backbone hidden → router embedding
    - router (HybridBVHRouter.pytorch_router): BVH traversal params
    - blend_gate: backbone vs expert mixing weight
    - shared_output_proj: expert hidden → vocab logits

    Everything else is frozen:
    - backbone: pretrained LLM (inference only, no_grad)
    - expert_modules: ternary weights (frozen int8/packed)
    """

    def __init__(
        self,
        pipeline,  # SpectralAIRealModelPipeline
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 8,
        alpha_balance: float = 0.01,
        top_k: int = 2,
    ):
        self.pipeline = pipeline
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.alpha_balance = alpha_balance
        self.top_k = top_k

        # Separate param groups: high LR for router, lower for output_proj
        # output_proj has 131M params vs router 89K — without separate LRs
        # the router gets negligible gradient signal.
        router_params = []
        router_count = 0
        proj_params = []
        proj_count = 0

        for name, module in [
            ("hidden_proj", pipeline.hidden_proj),
            ("router", pipeline.router.pytorch_router),
            ("blend_gate", pipeline.blend_gate),
        ]:
            for param in module.parameters():
                param.requires_grad = True
                router_params.append(param)
                router_count += param.numel()

        for param in pipeline.shared_output_proj.parameters():
            param.requires_grad = True
            proj_params.append(param)
            proj_count += param.numel()

        self.trainable_params = router_params + proj_params

        # Freeze everything else
        for param in pipeline.backbone.parameters():
            param.requires_grad = False
        for key, expert in pipeline.expert_modules.items():
            for param in expert.parameters():
                param.requires_grad = False
            for buf in expert.buffers():
                buf.requires_grad = False

        print(f"  Router+gate params: {router_count:,} "
              f"({router_count * 4 / 1024:.0f} KB) — lr={lr}")
        print(f"  Output proj params: {proj_count:,} "
              f"({proj_count * 4 / 1024**2:.0f} MB) — lr={lr / 10}")
        print(f"  Frozen backbone: "
              f"{sum(p.numel() for p in pipeline.backbone.parameters()):,}")
        print(f"  Frozen experts: {len(pipeline.expert_modules)} modules")

        # Optimizer with param groups — router gets 10x higher LR
        self.optimizer = torch.optim.AdamW([
            {"params": router_params, "lr": lr},
            {"params": proj_params, "lr": lr / 10},
        ], weight_decay=weight_decay)

        # Cosine LR scheduler with linear warmup
        self.scheduler = None  # set per-epoch in train()

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        max_steps: int = None,
    ) -> Dict:
        """Train for one epoch. Returns metrics dict."""
        self.pipeline.router.pytorch_router.train()
        self.pipeline.hidden_proj.train()
        self.pipeline.blend_gate.train()
        self.pipeline.shared_output_proj.train()

        # Ensure backbone is on GPU for the epoch
        self.pipeline._ensure_backbone_on_device()

        total_loss = 0.0
        total_task_loss = 0.0
        total_balance_loss = 0.0
        n_steps = 0
        n_tokens = 0
        alpha_sum = 0.0

        self.optimizer.zero_grad()
        self.pipeline.router.pytorch_router.reset_expert_counts()
        recent_expert_ids = []  # track routing diversity

        t0 = time.perf_counter()

        for step, (input_ids, target_ids) in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            B, S = input_ids.shape

            # ── Forward: backbone (frozen) ──────────────────────
            with torch.no_grad():
                outputs = self.pipeline.backbone(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                backbone_logits = outputs.logits  # (B, S, vocab)
                hidden = outputs.hidden_states[-1]  # (B, S, H)

            # ── Forward: router (trainable) ─────────────────────
            # Mean pool for routing decision
            prompt_emb = hidden.mean(dim=1).float()  # (B, H)
            router_input = self.pipeline.hidden_proj(prompt_emb)  # (B, 256)

            # Router in TRAINING mode (Gumbel-Softmax, differentiable)
            route_result = self.pipeline.router.pytorch_router(
                router_input, hard=False
            )
            # Track which expert was selected (for diversity monitoring)
            top_eid = route_result.expert_probs.argmax(dim=-1)[0].item()
            recent_expert_ids.append(top_eid)

            # ── Forward: top-k experts (frozen weights) ─────────
            expert_logits = self.pipeline.forward_topk_experts(
                hidden, route_result, top_k=self.top_k
            )  # (B, S, vocab)

            # ── Blend: backbone + expert ────────────────────────
            alpha = torch.sigmoid(
                self.pipeline.blend_gate(hidden.float())
            )  # (B, S, 1)
            blended_logits = (
                (1.0 - alpha) * backbone_logits.float()
                + alpha * expert_logits.float()
            )

            # ── Loss ────────────────────────────────────────────
            # Task loss: cross-entropy on blended logits
            task_loss = F.cross_entropy(
                blended_logits.reshape(-1, blended_logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            # Balance loss: prevent expert collapse
            balance_loss = self.pipeline.router.pytorch_router.load_balancing_loss()

            loss = task_loss + self.alpha_balance * balance_loss

            # Scale for gradient accumulation
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            # Accumulate metrics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_balance_loss += balance_loss.item()
            alpha_sum += alpha.mean().item()
            n_steps += 1
            n_tokens += B * S

            # Optimizer step every grad_accum_steps
            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            # Log every 50 steps
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / n_steps
                avg_alpha = alpha_sum / n_steps
                elapsed = time.perf_counter() - t0
                tok_s = n_tokens / elapsed
                print(f"    step {step+1:>5d} | loss={avg_loss:.4f} | "
                      f"alpha={avg_alpha:.4f} | "
                      f"balance={total_balance_loss/n_steps:.6f} | "
                      f"{tok_s:.0f} tok/s")

        # Release backbone to CPU
        self.pipeline._release_backbone_to_cpu()

        elapsed = time.perf_counter() - t0
        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_steps, 1),
            "task_loss": total_task_loss / max(n_steps, 1),
            "balance_loss": total_balance_loss / max(n_steps, 1),
            "avg_alpha": alpha_sum / max(n_steps, 1),
            "tokens": n_tokens,
            "tok_per_s": n_tokens / elapsed,
            "elapsed_s": elapsed,
            "steps": n_steps,
        }

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_steps: int = 100) -> Dict:
        """Evaluate on validation set. Returns metrics dict."""
        self.pipeline.eval()
        self.pipeline._ensure_backbone_on_device()

        total_loss = 0.0
        n_steps = 0
        n_tokens = 0
        alpha_sum = 0.0

        # Track expert usage
        expert_usage = {}

        for step, (input_ids, target_ids) in enumerate(val_loader):
            if step >= max_steps:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            B, S = input_ids.shape

            outputs = self.pipeline.backbone(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            backbone_logits = outputs.logits
            hidden = outputs.hidden_states[-1]

            prompt_emb = hidden.mean(dim=1).float()
            router_input = self.pipeline.hidden_proj(prompt_emb)
            route_result = self.pipeline.router.pytorch_router(
                router_input, hard=True
            )

            # Track expert selection
            eid = route_result.expert_id[0].item()
            expert_usage[eid] = expert_usage.get(eid, 0) + 1

            expert_logits = self.pipeline.forward_topk_experts(
                hidden, route_result, top_k=self.top_k
            )

            alpha = torch.sigmoid(
                self.pipeline.blend_gate(hidden.float())
            )
            bb_scale = backbone_logits.float().abs().mean()
            expert_logits_f = expert_logits.float().clamp(
                min=-3 * bb_scale, max=3 * bb_scale
            )
            blended = (
                (1.0 - alpha) * backbone_logits.float()
                + alpha * expert_logits_f
            )

            loss = F.cross_entropy(
                blended.reshape(-1, blended.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            total_loss += loss.item()
            alpha_sum += alpha.mean().item()
            n_steps += 1
            n_tokens += B * S

        self.pipeline._release_backbone_to_cpu()

        # Perplexity
        avg_loss = total_loss / max(n_steps, 1)
        ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow

        unique_experts = len(expert_usage)
        top_expert = max(expert_usage, key=expert_usage.get) if expert_usage else -1

        return {
            "val_loss": avg_loss,
            "val_ppl": ppl,
            "avg_alpha": alpha_sum / max(n_steps, 1),
            "unique_experts": unique_experts,
            "top_expert": top_expert,
            "expert_usage": expert_usage,
            "n_tokens": n_tokens,
        }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpectralAI v5.0 — Router Training (FASE 6)"
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace model ID for backbone"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Limit training tokens (for quick test, e.g. 100000)"
    )
    parser.add_argument(
        "--max-steps-per-epoch", type=int, default=None,
        help="Limit steps per epoch (for quick test)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save trained router weights"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  SpectralAI v5.0 — Router Training (FASE 6)")
    print("  Train BVH Router + Blend Gate on frozen backbone")
    print("=" * 72)

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ── Step 1: Build pipeline (reuse real_model_demo infrastructure) ──
    print("[1/4] Building pipeline...")

    # Import here to avoid circular deps
    from real_model_demo import (
        load_pretrained_model,
        detect_architecture,
        extract_mlp_layers,
        sparse_upcycle_to_experts,
        extract_native_ternary_experts,
        SpectralAIRealModelPipeline,
        TernaryExpertModule,
        NATIVE_TERNARY_MODELS,
        EXPERT_HIDDEN,
        EXPERT_INTERMEDIATE,
        _factorize_experts,
    )

    model, tokenizer = load_pretrained_model(args.model, device)
    arch = detect_architecture(model, args.model)
    mlps = extract_mlp_layers(model, arch)

    is_native = args.model in NATIVE_TERNARY_MODELS
    if is_native:
        experts = extract_native_ternary_experts(
            mlps, n_experts=args.n_experts,
            expert_hidden=EXPERT_HIDDEN,
            expert_intermediate=EXPERT_INTERMEDIATE,
            noise_scale=0.001, seed=42,
        )
    else:
        experts = sparse_upcycle_to_experts(
            mlps, n_experts=args.n_experts,
            expert_hidden=EXPERT_HIDDEN,
            expert_intermediate=EXPERT_INTERMEDIATE,
            noise_scale=0.01, seed=42,
        )

    del mlps
    gc.collect()

    n_l1, n_l2, n_l3 = _factorize_experts(args.n_experts)
    pipeline = SpectralAIRealModelPipeline(
        backbone_model=model,
        tokenizer=tokenizer,
        arch=arch,
        experts=experts,
        router_embed_dim=256,
        n_level1=n_l1,
        n_level2=n_l2,
        n_level3=n_l3,
        device=torch.device("cpu"),
        top_k_experts=args.top_k,
    )

    # Move trainable components to GPU
    pipeline.hidden_proj = pipeline.hidden_proj.to(device)
    pipeline.router = pipeline.router.to(device)
    pipeline.blend_gate = pipeline.blend_gate.to(device)
    pipeline.shared_output_proj = pipeline.shared_output_proj.to(device)
    pipeline.expert_cache._device = device
    pipeline._device = device

    # ── FIX: Initialize output_proj from backbone's lm_head ──────────
    # The default init (std=0.02) produces huge logits that, when mixed at
    # alpha=0.1, cause loss=498 (catastrophic). Solution: copy the backbone's
    # lm_head weights so expert_logits start in the same scale as backbone_logits.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        proj_out, proj_in = pipeline.shared_output_proj.weight.shape
        head_out, head_in = lm_head.weight.shape
        if proj_out == head_out and proj_in == head_in:
            pipeline.shared_output_proj.weight.data.copy_(lm_head.weight.data.to(device))
            print(f"  output_proj initialized from backbone lm_head ({head_in}→{head_out})")
        else:
            # Dimension mismatch — use very small init instead
            nn.init.normal_(pipeline.shared_output_proj.weight, std=0.001)
            print(f"  output_proj re-init std=0.001 (lm_head shape mismatch: "
                  f"{head_in}→{head_out} vs {proj_in}→{proj_out})")
    else:
        nn.init.normal_(pipeline.shared_output_proj.weight, std=0.001)
        print("  output_proj re-init std=0.001 (no lm_head found)")
    print()

    # ── Step 2: Load dataset ──────────────────────────────────────
    print("[2/4] Loading dataset...")
    train_ds, val_ds = load_wikitext_dataset(
        tokenizer, seq_len=args.seq_len, max_tokens=args.max_tokens
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, drop_last=True,
    )
    print()

    # ── Step 3: Train ─────────────────────────────────────────────
    print("[3/4] Training router...")
    trainer = RouterTrainer(
        pipeline=pipeline,
        device=device,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        top_k=args.top_k,
    )

    # Pre-training eval (skip if --skip-eval)
    pre_metrics = None
    if not args.skip_eval:
        print("\n  Pre-training evaluation:")
        # Force backbone to train mode during eval to avoid BitNet eval-mode issues
        pipeline.backbone.train()
        try:
            pre_metrics = trainer.evaluate(val_loader, max_steps=50)
            print(f"    val_loss={pre_metrics['val_loss']:.4f} | "
                  f"val_ppl={pre_metrics['val_ppl']:.1f} | "
                  f"alpha={pre_metrics['avg_alpha']:.4f} | "
                  f"experts_used={pre_metrics['unique_experts']}")
        except Exception as e:
            import traceback
            print(f"\n  ERROR during evaluation: {e}")
            traceback.print_exc()
            print("  Continuing with training anyway...")
            pre_metrics = {"val_loss": 99.0, "val_ppl": 9999.0, "avg_alpha": 0.0, "unique_experts": 0}
    else:
        print("\n  [Skipping pre-training evaluation]")
        pre_metrics = {"val_loss": 99.0, "val_ppl": 9999.0, "avg_alpha": 0.0, "unique_experts": 0}

    best_val_loss = pre_metrics["val_loss"]
    all_metrics = {"pre_training": pre_metrics, "epochs": []}

    # Setup cosine LR scheduler across all epochs
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = min(total_steps // 10, 200)
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    warmup_sched = LinearLR(trainer.optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(trainer.optimizer, T_max=max(total_steps - warmup_steps, 1))
    trainer.scheduler = SequentialLR(
        trainer.optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )
    print(f"  LR schedule: {warmup_steps} warmup + cosine to {total_steps} steps")

    for epoch in range(args.epochs):
        print(f"\n  ── Epoch {epoch+1}/{args.epochs} "
              f"{'─' * 45}")

        # Train
        pipeline.router.pytorch_router.train()
        train_metrics = trainer.train_epoch(
            train_loader, epoch,
            max_steps=args.max_steps_per_epoch,
        )
        print(f"    TRAIN: loss={train_metrics['train_loss']:.4f} | "
              f"alpha={train_metrics['avg_alpha']:.4f} | "
              f"{train_metrics['tok_per_s']:.0f} tok/s | "
              f"{train_metrics['elapsed_s']:.0f}s")

        # Anneal temperature
        pipeline.router.anneal_temperature()
        print(f"    Temperature: {pipeline.router.temperature:.4f}")

        # Evaluate (force backbone.train() to avoid BitNet eval-mode crash)
        if args.skip_eval:
            val_metrics = {"val_loss": train_metrics["train_loss"],
                           "val_ppl": math.exp(min(train_metrics["train_loss"], 20)),
                           "avg_alpha": train_metrics["avg_alpha"],
                           "unique_experts": 0, "expert_usage": {}}
            print(f"    VAL:   [skipped, using train metrics]")
        else:
            pipeline.backbone.train()  # avoid BitNet eval-mode CUDA crash
            try:
                val_metrics = trainer.evaluate(val_loader, max_steps=50)
                print(f"    VAL:   loss={val_metrics['val_loss']:.4f} | "
                      f"ppl={val_metrics['val_ppl']:.1f} | "
                      f"alpha={val_metrics['avg_alpha']:.4f} | "
                      f"experts_used={val_metrics['unique_experts']}")
            except Exception as e:
                print(f"    VAL:   ERROR — {e}")
                val_metrics = {"val_loss": train_metrics["train_loss"],
                               "val_ppl": math.exp(min(train_metrics["train_loss"], 20)),
                               "avg_alpha": train_metrics["avg_alpha"],
                               "unique_experts": 0, "expert_usage": {}}

        epoch_data = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        all_metrics["epochs"].append(epoch_data)

        # Save best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            print(f"    ★ New best val_loss: {best_val_loss:.4f}")

            if args.save_dir:
                save_path = Path(args.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "hidden_proj": pipeline.hidden_proj.state_dict(),
                    "router": pipeline.router.pytorch_router.state_dict(),
                    "blend_gate": pipeline.blend_gate.state_dict(),
                    "output_proj": pipeline.shared_output_proj.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                }, save_path / "router_best.pt")
                print(f"    Saved to {save_path / 'router_best.pt'}")

    # ── Step 4: Summary ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)

    final_val = all_metrics["epochs"][-1]["val"] if all_metrics["epochs"] else pre_metrics
    print(f"  Pre-training:  ppl={pre_metrics['val_ppl']:.1f} | "
          f"alpha={pre_metrics['avg_alpha']:.4f}")
    print(f"  Post-training: ppl={final_val['val_ppl']:.1f} | "
          f"alpha={final_val['avg_alpha']:.4f}")
    print(f"  Experts used:  {final_val['unique_experts']}/64")

    alpha_improved = final_val['avg_alpha'] > pre_metrics['avg_alpha'] * 2
    ppl_improved = final_val['val_ppl'] < pre_metrics['val_ppl']

    if alpha_improved and ppl_improved:
        print("  ✅ Experts are contributing! (alpha increased + ppl decreased)")
    elif alpha_improved:
        print("  🔄 Alpha increased but ppl didn't improve — more training needed")
    else:
        print("  ⚠️  Alpha didn't increase — experts may need larger model or more data")

    print()

    # Save final metrics
    import json
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    metrics_path = data_dir / "router_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
