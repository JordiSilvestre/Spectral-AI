#!/usr/bin/env python3
"""
train_moe.py -- SpectralAI FASE A: Train MoE from Scratch

Trains a complete MoE model with BVH routing from random initialization.
Unlike train_router.py (which uses frozen backbone + frozen experts),
this trains EVERYTHING: embeddings, router, experts, output projection.

Architecture:
    tokens → embeddings → BVH Router → top-k SwiGLU experts → logits

Why from scratch? The frozen BitNet experts were clones (identical MLPs).
The router couldn't learn because all experts produced the same output.
Training from scratch lets experts SPECIALIZE naturally.

Usage:
    # Quick test (1 epoch, small dataset)
    python train_moe.py --epochs 1 --max-tokens 500000

    # Full training
    python train_moe.py --epochs 10 --max-tokens 10000000

    # Resume from checkpoint
    python train_moe.py --resume data/moe_checkpoint.pt

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# SpectralAI imports
sys.path.insert(0, str(Path(__file__).parent))
from bvh_router import RouterConfig
from bvh_router_bridge import HybridBVHRouter
from trainable_experts import TrainableExpertConfig, SpectralAIMoE


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class TokenizedTextDataset(Dataset):
    """Simple chunked text dataset."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        n_chunks = len(token_ids) // seq_len
        self.data = token_ids[:n_chunks * seq_len].reshape(n_chunks, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return ids[:-1], ids[1:]  # input, target


def load_wikitext(tokenizer, seq_len: int, max_tokens: int):
    """Load and tokenize WikiText-103."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_split(articles, target_tokens):
        all_ids = []
        total = 0
        for i in range(0, len(articles), 500):
            batch = [t for t in articles[i:i+500] if t.strip()]
            if not batch:
                continue
            ids = tokenizer(
                "\n".join(batch), return_tensors="pt",
                truncation=False, add_special_tokens=False,
            ).input_ids[0]
            all_ids.append(ids)
            total += len(ids)
            if total >= target_tokens:
                break
        return torch.cat(all_ids)[:target_tokens]

    val_cap = max(max_tokens // 10, seq_len * 20)

    print("  Tokenizing train split...")
    train_ids = tokenize_split(ds["train"]["text"], max_tokens)
    print("  Tokenizing val split...")
    val_ids = tokenize_split(ds["validation"]["text"], val_cap)

    print(f"  Train: {len(train_ids):,} tokens ({len(train_ids) // seq_len} chunks)")
    print(f"  Val:   {len(val_ids):,} tokens ({len(val_ids) // seq_len} chunks)")

    return (
        TokenizedTextDataset(train_ids, seq_len),
        TokenizedTextDataset(val_ids, seq_len),
    )


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

class MoETrainer:
    """Trains the full SpectralAI MoE model."""

    def __init__(
        self,
        model: SpectralAIMoE,
        router: HybridBVHRouter,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        alpha_balance: float = 0.1,
        grad_accum_steps: int = 1,
        top_k: int = 2,
    ):
        self.model = model
        self.router = router
        self.device = device
        self.alpha_balance = alpha_balance
        self.grad_accum_steps = grad_accum_steps
        self.top_k = top_k
        self.scheduler = None

        # All params trainable
        all_params = list(model.parameters()) + list(router.pytorch_router.parameters())
        total = sum(p.numel() for p in all_params)
        print(f"  Total trainable params: {total:,} ({total * 4 / 1e6:.1f} MB FP32)")

        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        max_steps: Optional[int] = None,
    ) -> Dict:
        self.model.train()
        self.router.pytorch_router.train()
        self.router.pytorch_router.reset_expert_counts()

        total_loss = 0.0
        total_balance = 0.0
        alpha_sum = 0.0
        n_steps = 0
        n_tokens = 0
        recent_experts = []

        self.optimizer.zero_grad()
        t0 = time.perf_counter()

        for step, (input_ids, target_ids) in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            B, S = input_ids.shape

            # Forward
            logits, expert_probs, info = self.model(
                input_ids, self.router.pytorch_router
            )

            # Task loss
            task_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            # Per-batch load balancing loss (Switch Transformer style)
            # expert_probs: (B, n_experts) — routing probabilities
            n_exp = expert_probs.shape[-1]
            # f_i = fraction of tokens routed to expert i (average prob)
            f = expert_probs.mean(dim=0)  # (n_experts,)
            # P_i = average routing probability for expert i
            P = expert_probs.mean(dim=0)  # same as f for soft routing
            # L_balance = n_experts * Σ(f_i * P_i)
            # This penalizes when f and P are correlated (same expert gets high prob AND high usage)
            balance_loss = n_exp * (f * P).sum()

            # Entropy regularization: maximize routing entropy to prevent collapse
            # H(p) = -Σ p_i log(p_i); maximize → penalize low entropy
            probs_clamped = expert_probs.clamp(min=1e-8)
            entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1).mean()
            max_entropy = math.log(n_exp)
            entropy_loss = (max_entropy - entropy) / max_entropy  # normalized [0, 1]

            loss = task_loss + self.alpha_balance * balance_loss + 0.01 * entropy_loss

            # Backward
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            # Track
            total_loss += loss.item()
            total_balance += balance_loss.item()
            alpha_sum += info["alpha"]
            n_steps += 1
            n_tokens += B * S
            recent_experts.append(info["expert_id"][0].item())

            # Optimizer step
            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.router.pytorch_router.parameters()),
                    1.0,
                )
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
                n_unique = len(set(recent_experts[-50:]))
                lr_now = self.optimizer.param_groups[0]["lr"]
                ppl = math.exp(min(avg_loss, 20))
                print(f"    step {step+1:>5d} | loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                      f"alpha={avg_alpha:.4f} | "
                      f"experts={n_unique}/{self.model.expert_pool.n_experts} | "
                      f"bal={total_balance/n_steps:.4f} | "
                      f"lr={lr_now:.1e} | {tok_s:.0f} tok/s")

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_steps, 1)
        return {
            "train_loss": avg_loss,
            "train_ppl": math.exp(min(avg_loss, 20)),
            "alpha": alpha_sum / max(n_steps, 1),
            "tok_s": n_tokens / elapsed,
            "elapsed": elapsed,
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_steps: int = 200) -> Dict:
        self.model.eval()
        self.router.pytorch_router.eval()
        self.model.expert_pool.reset_usage()

        total_loss = 0.0
        n_steps = 0

        for step, (input_ids, target_ids) in enumerate(val_loader):
            if step >= max_steps:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits, _, _ = self.model(input_ids, self.router.pytorch_router)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        usage = self.model.expert_pool.get_expert_usage()
        used = sum(1 for v in usage.values() if v > 0)

        return {
            "val_loss": avg_loss,
            "val_ppl": math.exp(min(avg_loss, 20)),
            "experts_used": used,
            "expert_usage": usage,
        }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpectralAI FASE A: Train MoE from scratch")
    parser.add_argument("--n-experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--intermediate-dim", type=int, default=2048, help="Expert MLP intermediate dim")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k experts per token")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-tokens", type=int, default=5_000_000, help="Max training tokens")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-dir", type=str, default="data", help="Checkpoint save dir")
    parser.add_argument("--tokenizer", type=str, default="microsoft/bitnet-b1.58-2B-4T-bf16",
                        help="Tokenizer to use")
    args = parser.parse_args()

    print("=" * 70)
    print("  SpectralAI FASE A — Train MoE from Scratch")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({vram:.1f} GB)")
    print()

    # ── Step 1: Load tokenizer ─────────────────────────────────
    print("[1/4] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size:,}")

    # ── Step 2: Build model ────────────────────────────────────
    print("\n[2/4] Building MoE model...")
    expert_config = TrainableExpertConfig(
        n_experts=args.n_experts,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
    )

    model = SpectralAIMoE(
        vocab_size=vocab_size,
        expert_config=expert_config,
        router_embed_dim=256,
        seq_len=args.seq_len,
        top_k=args.top_k,
    ).to(device)

    print(model.param_summary())

    # BVH Router
    # Factorize n_experts into 3 levels
    n = args.n_experts
    if n == 16:
        n_l1, n_l2, n_l3 = 2, 2, 4
    elif n == 32:
        n_l1, n_l2, n_l3 = 2, 4, 4
    elif n == 64:
        n_l1, n_l2, n_l3 = 4, 4, 4
    else:
        # Generic factorization
        n_l3 = min(n, 4)
        n_l2 = min(n // n_l3, 4)
        n_l1 = n // (n_l2 * n_l3)

    router_cfg = RouterConfig(
        embed_dim=256,
        n_level1=n_l1,
        n_level2=n_l2,
        n_level3=n_l3,
    )
    router = HybridBVHRouter(router_cfg, device=device).to(device)
    print(f"  Router: {n_l1}×{n_l2}×{n_l3} = {n_l1*n_l2*n_l3} experts")

    # ── Step 3: Load dataset ───────────────────────────────────
    print(f"\n[3/4] Loading dataset...")
    train_ds, val_ds = load_wikitext(tokenizer, args.seq_len, args.max_tokens)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
    )

    # ── Step 4: Train ──────────────────────────────────────────
    print(f"\n[4/4] Training...")
    trainer = MoETrainer(
        model=model,
        router=router,
        device=device,
        lr=args.lr,
        top_k=args.top_k,
    )

    # LR scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(total_steps // 10, 200)

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup = LinearLR(trainer.optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(trainer.optimizer, T_max=max(total_steps - warmup_steps, 1))
    trainer.scheduler = SequentialLR(
        trainer.optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    print(f"  LR schedule: {warmup_steps} warmup + cosine to {total_steps} steps")

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        router.pytorch_router.load_state_dict(ckpt["router"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at epoch {start_epoch}")

    # Initial eval
    print(f"\n  Pre-training evaluation...")
    pre_eval = trainer.evaluate(val_loader)
    print(f"  VAL: loss={pre_eval['val_loss']:.4f} | ppl={pre_eval['val_ppl']:.1f} | "
          f"experts={pre_eval['experts_used']}/{args.n_experts}")

    best_val_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n  ── Epoch {epoch+1}/{args.epochs} "
              f"{'─' * 45}")

        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"    TRAIN: loss={train_metrics['train_loss']:.4f} | "
              f"ppl={train_metrics['train_ppl']:.1f} | "
              f"alpha={train_metrics['alpha']:.4f} | "
              f"{train_metrics['tok_s']:.0f} tok/s | "
              f"{train_metrics['elapsed']:.0f}s")

        # Eval
        val_metrics = trainer.evaluate(val_loader)
        print(f"    VAL:   loss={val_metrics['val_loss']:.4f} | "
              f"ppl={val_metrics['val_ppl']:.1f} | "
              f"experts={val_metrics['experts_used']}/{args.n_experts}")

        # Expert usage distribution
        usage = val_metrics["expert_usage"]
        top5 = sorted(usage.items(), key=lambda x: -x[1])[:5]
        print(f"    Top-5 experts: {', '.join(f'#{e}({c})' for e, c in top5)}")

        # Save best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt_path = save_dir / "moe_best.pt"
            torch.save({
                "model": model.state_dict(),
                "router": router.pytorch_router.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "val_ppl": val_metrics["val_ppl"],
                "args": vars(args),
            }, ckpt_path)
            print(f"    ★ New best! Saved to {ckpt_path}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Pre-training PPL:  {pre_eval['val_ppl']:.1f}")
    print(f"  Post-training PPL: {val_metrics['val_ppl']:.1f}")
    improvement = pre_eval["val_ppl"] / max(val_metrics["val_ppl"], 1)
    print(f"  Improvement:       {improvement:.1f}x")
    print(f"  Experts used:      {val_metrics['experts_used']}/{args.n_experts}")
    if val_metrics["val_ppl"] < pre_eval["val_ppl"]:
        print(f"  ✅ Model improved! BVH routing + expert specialization WORKS")
    else:
        print(f"  ⚠️  No improvement — may need more training or tuning")


if __name__ == "__main__":
    main()
