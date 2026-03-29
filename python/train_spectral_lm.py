#!/usr/bin/env python3
"""
train_spectral_lm.py — Entrenar SpectralAI Language Model

Dataset: WikiText-2 (pequeño, 2M tokens)
Objetivo: Comparable a GPT-2 small en perplexity

Uso:
    python train_spectral_lm.py --epochs 3 --batch-size 32 --lr 5e-4

Timeline:
  - Epoch 1: 2-3h (RTX 5070 Ti)
  - Epoch 2: 2-3h
  - Epoch 3: 2-3h
  Total: ~8h para 3 épocas = demostración end-to-end en 1 día
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import math
import numpy as np

from spectral_lm import SpectralAIForCausalLM

# ─────────────────────────────────────────────────────────────────
# 1. Dataset simple (síntesis en memoria para demo rápida)
# ─────────────────────────────────────────────────────────────────

class SyntheticTextDataset(Dataset):
    """
    Dataset sintético: genera secuencias aleatorias de tokens.

    Alternativa: reemplazar con WikiText-2 si quieres datos reales.
    Pero para un "proof of concept" rápido, esto es suficiente.
    """

    def __init__(self, vocab_size=10_000, seq_len=256, num_samples=10_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generar secuencia aleatoria (en la práctica: cargar de archivo)
        tokens = torch.randint(1, self.vocab_size, (self.seq_len,))
        return tokens


class WikiTextDataset(Dataset):
    """
    WikiText-2 descargado via HuggingFace datasets + tokenizado con tiktoken (GPT-2 BPE).

    Vocab: 50K tokens (GPT-2 compatible)
    Uso:
        dataset = WikiTextDataset(split="train", seq_len=256)
    """

    def __init__(self, split: str = "train", seq_len: int = 256,
                 cache_file: str = None):
        self.seq_len = seq_len

        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")  # 50K vocab, BPE

        # Cache para no re-tokenizar cada vez
        cache_path = Path(cache_file or f"wikitext2_{split}_tokens.npy")

        if cache_path.exists():
            print(f"[Data] Cargando cache {cache_path}...")
            tokens = np.load(str(cache_path))
        else:
            print(f"[Data] Descargando WikiText-2 ({split})...")
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

            print(f"[Data] Tokenizando {len(ds)} filas...")
            all_tokens = []
            for row in tqdm(ds, desc="Tokenizing"):
                text = row["text"].strip()
                if text:
                    all_tokens.extend(self.enc.encode(text))
            tokens = np.array(all_tokens, dtype=np.int32)
            np.save(str(cache_path), tokens)
            print(f"[Data] {len(tokens):,} tokens guardados en {cache_path}")

        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"[Data] {split}: {len(self.tokens):,} tokens, {len(self)} muestras de {seq_len}")

    def __len__(self):
        # stride = seq_len//2 → ventanas solapadas al 50% (estándar en LMs)
        stride = self.seq_len // 2
        return max(0, (len(self.tokens) - self.seq_len - 1) // stride)

    def __getitem__(self, idx):
        stride = self.seq_len // 2
        start = idx * stride
        return self.tokens[start : start + self.seq_len]


class TextDataset(Dataset):
    """Dataset desde archivo de tokens pre-tokenizados."""

    def __init__(self, text_file: str, seq_len: int = 256, vocab_size: int = 10_000):
        self.seq_len = seq_len
        with open(text_file, 'r', encoding='utf-8') as f:
            tokens = list(map(int, f.read().split()))
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len]
        if len(chunk) < self.seq_len:
            pad = torch.zeros(self.seq_len - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk


# ─────────────────────────────────────────────────────────────────
# 2. Funciones de entrenamiento
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, epoch_num):
    """Entrenar una época."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num + 1}")
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)

        # Forward pass
        logits = model(batch)  # (batch, seq_len, vocab_size)

        # Loss (no computamos loss en el primer token porque no hay contexto)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, model.vocab_size),
            batch[:, 1:].reshape(-1),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Métricas
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluar en validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = batch.to(device)

        logits = model(batch)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, model.vocab_size),
            batch[:, 1:].reshape(-1),
        )

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Train SpectralAI Language Model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="spectral_lm_checkpoint.pt")
    parser.add_argument("--log", type=str, default="spectral_training_log.json")
    parser.add_argument("--use-synthetic", action="store_true", default=False,
                        help="Usar dataset sintético en lugar de WikiText-2")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path a archivo de tokens pre-tokenizados (alternativa a WikiText-2)")
    args = parser.parse_args()

    print("=" * 70)
    print("SpectralAI Language Model — Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    device = torch.device(args.device)

    # ── Crear modelo ────────────────────────────────────────────────
    # GPT-2 usa 50K vocab (tiktoken gpt2 encoding)
    VOCAB_SIZE = 50_257 if not args.use_synthetic else 10_000
    print("\n[Model] Creando SpectralAIForCausalLM...")
    model = SpectralAIForCausalLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        context_len=256,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parámetros: {total_params:,} ({total_params/1e6:.1f}M)")

    # ── Cargar dataset ───────────────────────────────────────────────
    print("\n[Data] Cargando dataset...")
    if args.use_synthetic:
        print("[Data] Usando dataset sintético (10K muestras x 256 tokens)")
        train_dataset = SyntheticTextDataset(vocab_size=10_000, seq_len=256, num_samples=10_000)
        val_dataset = SyntheticTextDataset(vocab_size=10_000, seq_len=256, num_samples=1_000)
    elif args.data_file is not None:
        print(f"[Data] Cargando tokens desde {args.data_file}")
        full_dataset = TextDataset(args.data_file, seq_len=256, vocab_size=VOCAB_SIZE)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    else:
        # WikiText-2 via HuggingFace + tiktoken GPT-2 BPE
        train_dataset = WikiTextDataset(split="train",      seq_len=256)
        val_dataset   = WikiTextDataset(split="validation", seq_len=256)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    log = {
        "model_config": {
            "vocab_size": 10_000,
            "embed_dim": 256,
            "num_layers": 4,
            "num_heads": 4,
            "context_len": 256,
            "total_parameters": total_params,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
        },
        "epochs": [],
    }

    best_val_loss = float('inf')
    t0 = time.time()

    for epoch in range(args.epochs):
        # Entrenar
        train_loss, train_perp = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluar
        val_loss, val_perp = evaluate(model, val_loader, device)

        # Scheduler
        scheduler.step()

        # Log
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_perplexity": float(train_perp),
            "val_loss": float(val_loss),
            "val_perplexity": float(val_perp),
            "lr": optimizer.param_groups[0]['lr'],
        }
        log["epochs"].append(epoch_log)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f} | perplexity: {train_perp:.2f}")
        print(f"  Val loss:   {val_loss:.4f} | perplexity: {val_perp:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Guardar mejor checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  [SAVE] Checkpoint guardado: {args.checkpoint}")

    elapsed = time.time() - t0
    print(f"\n[Training] Completado en {elapsed/3600:.1f}h")

    # ── Guardar log ──────────────────────────────────────────────────
    with open(args.log, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"[Log] Guardado: {args.log}")

    # ── Cargar mejor modelo y hacer una generación de prueba ─────────
    print("\n" + "=" * 70)
    print("Generation Test")
    print("=" * 70)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4, 5]], device=device)  # 5 tokens dummy
    with torch.no_grad():
        generated = model.generate(prompt, max_length=50, temperature=0.7, top_k=50)

    print(f"[Generation] Prompt tokens: {prompt.tolist()}")
    print(f"[Generation] Generated {generated.shape[1]} tokens total")
    print(f"[Generation] Sample: {generated[0, :20].tolist()}")

    print("\n" + "=" * 70)
    print("Training Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
