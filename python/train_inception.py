#!/usr/bin/env python3
"""
train_inception.py — Training del sistema SpectralAI v4.0 COMPLETO

Entrena SpectralAIInceptionLM con:
  - WikiText-2 (GPT-2 BPE, 50K vocab)
  - L_total = L_task + alpha * L_spatial
  - Temperature annealing (soft → hard sphere assignment)
  - Gradient clipping
  - Cosine LR decay

Comparativa justa con GPT-2 baseline (mismas condiciones exactas).

Ejecutar:
    python train_inception.py --epochs 10 --batch-size 32 --device cuda
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

sys.path.insert(0, str(Path(__file__).parent))
from inception_attention import SpectralAIInceptionLM, InceptionConfig
from train_spectral_lm import WikiTextDataset


# ─────────────────────────────────────────────────────────────────
# Spatial Loss (simplificada para integración en training loop)
# ─────────────────────────────────────────────────────────────────

def compute_spatial_loss(model: SpectralAIInceptionLM) -> torch.Tensor:
    """
    L_spatial = L_proximity + L_coverage + L_regularization

    Calcula la pérdida espacial directamente desde los parámetros del modelo.
    Optimizado: usa operaciones vectorizadas sin loops internos.
    """
    parts = []

    for block in model.h:
        traversal = block.attn.traversal

        for level in [traversal.level1, traversal.level2, traversal.level3]:
            centers = level.centers   # (K, 3)
            radii   = level.radii     # (K,)

            # L_proximity: penalizar centros demasiado cercanos
            if centers.shape[0] > 1:
                # Distancias cuadráticas sin cdist (más rápido para K pequeño)
                diff = centers.unsqueeze(0) - centers.unsqueeze(1)  # (K, K, 3)
                d_sq = (diff ** 2).sum(-1)  # (K, K)
                # Solo triángulo superior
                mask = torch.triu(torch.ones_like(d_sq, dtype=torch.bool), diagonal=1)
                parts.append(torch.exp(-d_sq[mask]).mean())

            # L_coverage + L_reg: vectorizado
            parts.append(F.relu(0.3 - radii).mean())
            parts.append((radii ** 2).mean() * 0.01)

        # Regularizar portales (no desviarse de identidad)
        for portal in [traversal.portal1, traversal.portal2, traversal.portal3]:
            # Solo penalizar off-diagonal y traslación
            T = portal.transform  # (K, 3, 4)
            eye = torch.zeros_like(T)
            for i in range(3):
                eye[:, i, i] = 1.0
            parts.append(((T - eye) ** 2).mean() * 0.2)

    return torch.stack(parts).sum()


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SpectralAI v4.0 Inception Engine — Training COMPLETO")
    print("=" * 70)
    print(f"Device:      {device}")
    if device.type == "cuda":
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Alpha spatial: {args.alpha_spatial}")

    # ── Modelo ────────────────────────────────────────────────────
    model = SpectralAIInceptionLM(
        vocab_size=50_257,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        context_len=256,
        mlp_hidden=1024,
        n_domains=4,
        n_subdomains=4,
        n_concepts=4,
        spectral_dim=64,
        num_fourier_modes=8,
        temperature_init=1.0,
        temperature_min=0.1,
        temperature_decay=0.99,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParametros: {total_params:,} ({total_params/1e6:.1f}M)")
    print("Atencion:   InceptionAttention — 4 niveles BVH + Fourier + Spectral")

    # Guardar referencia al modelo sin compilar (para acceder a attrs)
    raw_model = model

    # ── torch.compile: fusión de operaciones automática ───────────
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            # "reduce-overhead" requiere Triton (no disponible en Windows)
            # Usamos backend "inductor" con mode="default" que funciona sin Triton
            import importlib
            _has_triton = importlib.util.find_spec("triton") is not None
            _compile_mode = "reduce-overhead" if _has_triton else "default"
            _compile_backend = "inductor" if _has_triton else "eager"
            model = torch.compile(model, mode=_compile_mode, backend=_compile_backend)
            print(f"[OPT] torch.compile activado (mode={_compile_mode}, backend={_compile_backend})")
        except Exception as e:
            print(f"[OPT] torch.compile no disponible: {e}")

    # ── AMP: mixed precision FP16 ────────────────────────────────
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[OPT] AMP FP16 activado")

    # ── Dataset ───────────────────────────────────────────────────
    train_ds = WikiTextDataset(split="train",      seq_len=256)
    val_ds   = WikiTextDataset(split="validation", seq_len=256)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.1, betas=(0.9, 0.95),
    )

    # Cosine schedule with linear warmup (stabilizes BVH initialization)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)  # 10% of training or 500 steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[OPT] LR warmup: {warmup_steps} steps, then cosine decay over {total_steps} steps")

    best_val_loss = float('inf')
    log = []

    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        total_task_loss = 0.0
        total_spatial_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        cached_spat = 0.0
        for step, batch in enumerate(pbar):
            batch = batch.to(device)

            # Forward con AMP (FP16 automático donde sea seguro)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(batch[:, :-1])
                task_loss = F.cross_entropy(
                    logits.reshape(-1, raw_model.vocab_size),
                    batch[:, 1:].reshape(-1),
                )

                # L_spatial: every step for consistent spatial enforcement
                spatial_loss = compute_spatial_loss(raw_model)
                cached_spat = spatial_loss.item()
                total_loss = task_loss + args.alpha_spatial * spatial_loss

            # Backward con GradScaler (FP16 → FP32 gradientes)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            total_task_loss    += task_loss.item()
            total_spatial_loss += cached_spat
            pbar.set_postfix(
                loss=f"{task_loss.item():.4f}",
                spat=f"{cached_spat:.3f}",
                T=f"{raw_model.h[0].attn.temperature.item():.3f}",
            )

        train_task  = total_task_loss / len(train_loader)
        train_spat  = total_spatial_loss / len(train_loader)
        epoch_time  = time.time() - t0

        # ── Temperature annealing ────────────────────────────────
        raw_model.anneal_temperature()
        curr_temp = raw_model.h[0].attn.temperature.item()

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch[:, :-1])
                val_loss += F.cross_entropy(
                    logits.reshape(-1, raw_model.vocab_size),
                    batch[:, 1:].reshape(-1),
                ).item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(min(val_loss, 20))

        # ── Log ──────────────────────────────────────────────────
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch:>2}/{args.epochs} | "
              f"task={train_task:.4f} | spat={train_spat:.4f} | "
              f"val={val_loss:.4f} | ppl={val_ppl:.1f} | "
              f"T={curr_temp:.3f} | {epoch_time:.1f}s")

        log.append({
            "epoch":       epoch,
            "train_loss":  train_task,
            "spatial_loss": train_spat,
            "val_loss":    val_loss,
            "ppl":         val_ppl,
            "temperature": curr_temp,
            "lr":          lr,
            "epoch_time_s": epoch_time,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = Path(__file__).parent.parent / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(raw_model.state_dict(), str(ckpt_dir / "inception_best.pt"))
            print("  [SAVE] checkpoints/inception_best.pt")

    # ── Guardar log ──────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    output = {
        "model":  "SpectralAIInceptionLM",
        "params": total_params,
        "config": {
            "n_domains": 4, "n_subdomains": 4, "n_concepts": 4,
            "spectral_dim": 64, "num_fourier_modes": 8,
            "embed_dim": 256, "num_layers": 4, "num_heads": 4,
        },
        "training": log,
    }
    with open(str(data_dir / "inception_training_log.json"), "w") as f:
        json.dump(output, f, indent=2)

    best_ppl = math.exp(min(best_val_loss, 20))
    print(f"\nMejor val_loss: {best_val_loss:.4f} (ppl={best_ppl:.1f})")
    print("Guardado: inception_best.pt, inception_training_log.json")

    # ── Generación demo ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Generation Test")
    print(f"{'='*70}")

    model.eval()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    prompts = [
        "The history of",
        "In the year 2025",
        "Scientists discovered that",
    ]

    for prompt_text in prompts:
        tokens = enc.encode(prompt_text)
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=30,
                                    temperature=0.8, top_k=40)
        generated = enc.decode(output[0].tolist())
        print(f"\nPrompt:    {prompt_text}")
        print(f"Generated: {generated}")

    return best_val_loss


# ─────────────────────────────────────────────────────────────────
# Comparativa automática
# ─────────────────────────────────────────────────────────────────

def compare_results():
    """Lee los logs de los 3 modelos y genera tabla comparativa."""
    models = {}

    data_dir = Path(__file__).parent.parent / "data"
    for name, path in [
        ("GPT-2 (MatMul)",     data_dir / "gpt2_baseline_log.json"),
        ("SpectralAI Inception", data_dir / "inception_training_log.json"),
    ]:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            models[name] = data

    if len(models) < 2:
        print("[compare] Necesitas entrenar ambos modelos primero.")
        return

    print("\n" + "=" * 72)
    print("COMPARATIVA FINAL: SpectralAI Inception vs GPT-2")
    print("=" * 72)
    print(f"{'Epoch':>5}", end="")
    for name in models:
        print(f" | {name:>22}", end="")
    print()
    print("-" * 72)

    max_epochs = max(len(m["training"]) for m in models.values())
    for i in range(max_epochs):
        print(f"{i+1:>5}", end="")
        for name, data in models.items():
            epochs = data["training"]
            if i < len(epochs):
                ppl = epochs[i]["ppl"]
                print(f" | {ppl:>22.1f}", end="")
            else:
                print(f" | {'--':>22}", end="")
        print()

    print("=" * 72)
    for name, data in models.items():
        best = min(e["ppl"] for e in data["training"])
        params = data.get("params", "?")
        print(f"{name}: best ppl={best:.1f}, params={params:,}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpectralAI Inception Engine — Training completo"
    )
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--alpha-spatial", type=float, default=0.15,
                        help="Peso de L_spatial en L_total")
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--compare-only",  action="store_true",
                        help="Solo mostrar comparativa de logs existentes")
    args = parser.parse_args()

    if args.compare_only:
        compare_results()
    else:
        train(args)
        compare_results()


if __name__ == "__main__":
    main()
