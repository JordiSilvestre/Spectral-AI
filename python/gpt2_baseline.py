#!/usr/bin/env python3
"""
gpt2_baseline.py — GPT-2 baseline para comparativa justa con SpectralAI

Mismas condiciones exactas:
  - Vocab:    50,257 tokens (GPT-2 BPE via tiktoken)
  - Embed:    256d
  - Capas:    4
  - Heads:    4
  - Contexto: 256 tokens
  - Dataset:  WikiText-2
  - Optimizer: AdamW, lr=5e-4, cosine decay

Única diferencia vs SpectralAI: atención = scaled dot-product (MatMul O(N²))

Entrenamiento:
    python gpt2_baseline.py --epochs 10 --batch-size 32

Benchmark rápido (solo mide velocidad, sin training):
    python gpt2_baseline.py --benchmark-only
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Reusar WikiTextDataset del training script de SpectralAI
sys.path.insert(0, str(Path(__file__).parent))
from train_spectral_lm import WikiTextDataset, SyntheticTextDataset

# ─────────────────────────────────────────────────────────────────
# GPT-2 estándar (arquitectura idéntica a SpectralAI, atención diferente)
# ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Scaled dot-product attention estándar — O(N²) MatMul."""

    def __init__(self, embed_dim: int, num_heads: int, context_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask fija
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.split(D, dim=2)

        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # O(N²) MatMul — esto es lo que SpectralAI reemplaza
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.bias[:, :, :S, :S] == 0, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(embed_dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class GPT2Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 context_len: int, mlp_hidden: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, context_len)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.mlp  = MLP(embed_dim, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Baseline(nn.Module):
    """
    GPT-2 con mismos hiperparámetros que SpectralAI-small:
      vocab=50257, embed=256, layers=4, heads=4, ctx=256, mlp=1024
    """

    def __init__(
        self,
        vocab_size:  int = 50_257,
        embed_dim:   int = 256,
        num_layers:  int = 4,
        num_heads:   int = 4,
        context_len: int = 256,
        mlp_hidden:  int = 1024,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.context_len = context_len

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(context_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.h   = nn.ModuleList([
            GPT2Block(embed_dim, num_heads, context_len, mlp_hidden)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tied weights (igual que GPT-2 original)
        self.lm_head.weight = self.wte.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        assert S <= self.context_len

        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x   = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits   = self(idx_cond)
            logits   = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GPT-2 Baseline — Comparativa con SpectralAI")
    print("=" * 70)
    print(f"Device:     {device}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Modelo
    model = GPT2Baseline(
        vocab_size=50_257, embed_dim=256, num_layers=4,
        num_heads=4, context_len=256, mlp_hidden=1024,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros: {total_params:,} ({total_params/1e6:.1f}M)")
    print("Atención:   CausalSelfAttention — MatMul O(N²)")

    # Dataset
    if args.use_synthetic:
        train_ds = SyntheticTextDataset(vocab_size=50_257, seq_len=256, num_samples=5_000)
        val_ds   = SyntheticTextDataset(vocab_size=50_257, seq_len=256, num_samples=500)
    else:
        train_ds = WikiTextDataset(split="train",      seq_len=256)
        val_ds   = WikiTextDataset(split="validation", seq_len=256)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    best_val_loss = float('inf')
    log = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            loss   = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                batch[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - t0

        # ── Val ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                logits = model(batch[:, :-1])
                val_loss += F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    batch[:, 1:].reshape(-1),
                ).item()
        val_loss /= len(val_loader)
        val_ppl   = math.exp(min(val_loss, 20))

        print(f"\nEpoch {epoch:>2}/{args.epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"ppl={val_ppl:.1f} | {epoch_time:.1f}s")

        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "ppl": val_ppl,
                    "epoch_time_s": epoch_time})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = Path(__file__).parent.parent / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), str(ckpt_dir / "gpt2_baseline_best.pt"))

    # ── Guardar log ──────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    with open(str(data_dir / "gpt2_baseline_log.json"), "w") as f:
        json.dump({"model": "gpt2_baseline", "params": total_params,
                   "training": log}, f, indent=2)
    print(f"\nMejor val_loss: {best_val_loss:.4f}  (ppl={math.exp(min(best_val_loss,20)):.1f})")
    print("Guardado: gpt2_baseline_best.pt, gpt2_baseline_log.json")

    return best_val_loss


# ─────────────────────────────────────────────────────────────────
# Benchmark rápido (sin training)
# ─────────────────────────────────────────────────────────────────

def benchmark_speed(device_str: str = "cuda"):
    """Mide tokens/sec y latencia de generación."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model  = GPT2Baseline().to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("Benchmark: GPT-2 baseline vs SpectralAI (misma arquitectura)")
    print("=" * 70)

    results = {}

    # Forward throughput
    batch = torch.randint(0, 50_257, (32, 255), device=device)
    for _ in range(3):  # warmup
        with torch.no_grad():
            _ = model(batch)
    torch.cuda.synchronize()

    t0 = time.time()
    N  = 20
    with torch.no_grad():
        for _ in range(N):
            _ = model(batch)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / N

    tokens_per_sec = 32 * 255 / elapsed
    print(f"\nForward (batch=32, seq=255):")
    print(f"  Tiempo:     {elapsed*1000:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    results["forward_ms"] = elapsed * 1000
    results["tokens_per_sec"] = tokens_per_sec

    # Generación autoregresiva (1 token a la vez)
    prompt = torch.randint(0, 50_257, (1, 10), device=device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=100, temperature=1.0)
    torch.cuda.synchronize()
    gen_time = time.time() - t0
    print(f"\nGeneración (100 tokens):")
    print(f"  Tiempo:  {gen_time*1000:.1f} ms")
    print(f"  Velocidad: {100/gen_time:.1f} tokens/sec")
    results["gen_100tok_ms"] = gen_time * 1000
    results["gen_tokens_per_sec"] = 100 / gen_time

    # Memoria GPU
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nMemoria GPU pico: {mem:.1f} MB")
        results["peak_mem_mb"] = mem

    print("\n" + "-" * 70)
    print("Para comparar — lanza SpectralAI con mismos params:")
    print("  python spectral_lm.py --benchmark")

    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-2 baseline — comparativa con SpectralAI")
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=5e-4)
    parser.add_argument("--device",          type=str,   default="cuda")
    parser.add_argument("--use-synthetic",   action="store_true")
    parser.add_argument("--benchmark-only",  action="store_true",
                        help="Solo benchmark de velocidad, sin training")
    args = parser.parse_args()

    if args.benchmark_only:
        benchmark_speed(args.device)
    else:
        train(args)


if __name__ == "__main__":
    main()
