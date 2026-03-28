#!/usr/bin/env python3
"""
train_multi_domain.py — Training Multi-Dominio para LiquidBit v5.0 FASE 4

Entrena el Orchestrator con 4 dominios supervisados:
  0 = General (WikiText-2)
  1 = Codigo Python
  2 = Ciencia
  3 = Legal

Mide routing accuracy: cuando llega codigo, va al experto de codigo?

Ejecutar:
  python train_multi_domain.py --epochs 10 --batch_size 32
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from orchestrator import LiquidBitOrchestrator, OrchestratorConfig
from multi_domain_dataset import (
    create_multi_domain_dataset, collate_with_domain,
    DOMAIN_NAMES, N_DOMAINS,
)

SAVE_DIR = Path(__file__).parent.parent / "checkpoints"


def evaluate(model, val_dl, device):
    """Evaluar en validacion: ppl + routing accuracy por dominio."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_expert_ids = []
    all_domain_ids = []

    with torch.no_grad():
        for batch_tokens, batch_domains in val_dl:
            batch_tokens = batch_tokens.to(device)
            batch_domains = batch_domains.to(device)
            inp = batch_tokens[:, :-1]
            tgt = batch_tokens[:, 1:]

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits, info = model(inp, tgt, domain_ids=batch_domains)

            total_loss += info['task_loss'].item() * inp.shape[0]
            total_tokens += inp.shape[0]
            all_expert_ids.append(info['expert_id'].cpu())
            all_domain_ids.append(batch_domains.cpu())

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = np.exp(min(avg_loss, 20))

    # Routing accuracy
    expert_ids = torch.cat(all_expert_ids)
    domain_ids = torch.cat(all_domain_ids)
    acc_info = model.routing_accuracy(expert_ids, domain_ids, n_domains=N_DOMAINS)

    return avg_loss, ppl, acc_info


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("LiquidBit v5.0 — Training Multi-Dominio FASE 4")
    print("=" * 70)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config
    cfg = OrchestratorConfig(
        vocab_size=50_257,
        router_embed_dim=256,
        expert_embed_dim=args.expert_dim,
        n_level1=4, n_level2=4, n_level3=4,  # 64 expertos
        expert_layers=args.expert_layers,
        expert_heads=4,
        context_len=256,
        spectral_dim=64,
        alpha_balance=0.01,
        alpha_router=args.alpha_router,
    )

    model = LiquidBitOrchestrator(cfg, device).to(device)
    pc = model.param_count()

    print(f"Expertos:        {cfg.n_experts} (4x4x4)")
    print(f"Expert dim:      {cfg.expert_embed_dim}")
    print(f"Expert layers:   {cfg.expert_layers}")
    print(f"Params total:    {pc['total']:,} ({pc['total']/1e6:.1f}M)")
    print(f"Alpha router:    {cfg.alpha_router}")
    print(f"Dominios:        {N_DOMAINS} ({', '.join(DOMAIN_NAMES)})")

    # Datasets multi-dominio
    print("\n--- Datasets ---")
    train_ds = create_multi_domain_dataset("train", seq_len=256)
    val_ds = create_multi_domain_dataset("validation", seq_len=256,
                                          max_tokens_per_domain=500_000)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_with_domain,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_with_domain,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_dl), eta_min=1e-5,
    )

    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    log = []

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.router.reset_expert_counts()
        epoch_loss = 0.0
        epoch_routing_loss = 0.0
        epoch_balance_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Routing accuracy acumulada durante training
        train_expert_ids = []
        train_domain_ids = []

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_tokens, batch_domains in pbar:
            batch_tokens = batch_tokens.to(device)
            batch_domains = batch_domains.to(device)
            inp = batch_tokens[:, :-1]
            tgt = batch_tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, info = model(inp, tgt, domain_ids=batch_domains)
                    loss = info['total_loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, info = model(inp, tgt, domain_ids=batch_domains)
                loss = info['total_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += info['task_loss'].item()
            epoch_balance_loss += info['balance_loss'].item()
            if 'routing_loss' in info:
                epoch_routing_loss += info['routing_loss'].item()
            n_batches += 1

            train_expert_ids.append(info['expert_id'].detach().cpu())
            train_domain_ids.append(batch_domains.detach().cpu())

            pbar.set_postfix({
                'loss': f"{info['task_loss'].item():.3f}",
                'r_loss': f"{info.get('routing_loss', torch.tensor(0)).item():.3f}",
                'T': f"{model.router.temperature.item():.3f}",
            })

        # Anneal
        model.anneal_temperature()
        dt = time.time() - t0

        # Training routing accuracy
        train_expert_ids_cat = torch.cat(train_expert_ids)
        train_domain_ids_cat = torch.cat(train_domain_ids)
        train_acc = model.routing_accuracy(
            train_expert_ids_cat, train_domain_ids_cat, n_domains=N_DOMAINS)

        # Validation
        val_loss, val_ppl, val_acc = evaluate(model, val_dl, device)

        # Log
        entry = {
            'epoch': epoch,
            'train_loss': epoch_loss / n_batches,
            'train_routing_loss': epoch_routing_loss / n_batches,
            'train_balance_loss': epoch_balance_loss / n_batches,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'train_routing_acc': train_acc['accuracy'],
            'val_routing_acc': val_acc['accuracy'],
            'val_acc_per_domain': val_acc['per_domain'],
            'temperature': model.router.temperature.item(),
            'time_s': dt,
        }
        log.append(entry)

        # Print
        print(f"\nEpoch {epoch}/{args.epochs} ({dt:.1f}s)")
        print(f"  Train loss: {entry['train_loss']:.4f} | "
              f"Routing loss: {entry['train_routing_loss']:.4f} | "
              f"Balance: {entry['train_balance_loss']:.4f}")
        print(f"  Val loss:   {val_loss:.4f} | Val ppl: {val_ppl:.1f}")
        print(f"  Routing accuracy (train): {train_acc['accuracy']*100:.1f}%")
        print(f"  Routing accuracy (val):   {val_acc['accuracy']*100:.1f}%")
        for d in range(N_DOMAINS):
            d_acc = val_acc['per_domain'].get(d, 0.0)
            print(f"    {DOMAIN_NAMES[d]:>10}: {d_acc*100:.1f}%")

        # Expert distribution
        counts_per_domain = {}
        for d in range(N_DOMAINS):
            mask = train_domain_ids_cat == d
            if mask.any():
                experts_used = train_expert_ids_cat[mask].unique()
                counts_per_domain[DOMAIN_NAMES[d]] = len(experts_used)
        print(f"  Expertos unicos por dominio: {counts_per_domain}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       SAVE_DIR / "orchestrator_multidomain_best.pt")
            print(f"  -> Guardado mejor modelo (val_loss={val_loss:.4f})")

    # Save log
    log_path = Path(__file__).parent.parent / "data" / "multidomain_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\nLog guardado: {log_path}")

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val ppl:  {np.exp(min(best_val_loss, 20)):.1f}")
    best_epoch = min(log, key=lambda x: x['val_loss'])
    print(f"Best routing accuracy: {best_epoch['val_routing_acc']*100:.1f}%")
    print(f"Per-domain accuracy:")
    for d in range(N_DOMAINS):
        print(f"  {DOMAIN_NAMES[d]:>10}: {best_epoch['val_acc_per_domain'].get(str(d), best_epoch['val_acc_per_domain'].get(d, 0))*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expert_dim", type=int, default=128)
    parser.add_argument("--expert_layers", type=int, default=2)
    parser.add_argument("--alpha_router", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
