#!/usr/bin/env python3
"""
benchmark_expert_types.py — FASE 5: Benchmark de Cuantización
SpectralAI v5.0 "Orchestrator"

Compara 4 variantes de backbone:
  1. FP32 (baseline)
  2. FP16 (half precision)
  3. INT8 (PyTorch dynamic quantization — CPU only)
  4. Ternario BitNet {-1, 0, +1}

Para cada variante mide:
  - VRAM / RAM (MB)
  - Throughput (tok/s)
  - Perplexity en validación multi-dominio
  - Tamaño en disco (MB)
  - Routing accuracy (debe mantenerse 100%)

Requisitos:
  pip install torch tiktoken datasets tqdm

Copyright (c) 2026 SpectralAI Studio — Apache 2.0
"""

import os
import sys
import time
import json
import copy
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import OrchestratorConfig, SpectralAIOrchestrator
from micro_expert import (
    MiniTransformerLM, TernaryLinear, quantize_ternary,
    quantize_model_ternary,
)
from multi_domain_dataset import (
    create_multi_domain_dataset, collate_with_domain,
)


# ── Constants ────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOMAINS = 4
DOMAIN_NAMES = {0: "general", 1: "code", 2: "science", 3: "legal"}
BATCH_SIZE = 16
SEQ_LEN = 256
WARMUP_BATCHES = 5
BENCHMARK_BATCHES = 50
TRAINING_EPOCHS = 5
LR = 3e-4
MAX_TOKENS_PER_DOMAIN = 500_000  # Menor para benchmark rápido


# ── Helpers ──────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    """Total parameters (trainable + buffers)."""
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return params + buffers


def model_memory_mb(model: nn.Module) -> float:
    """Memory footprint in MB (params + buffers)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


def model_disk_size_mb(model: nn.Module) -> float:
    """Size on disk when saved with torch.save (MB)."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / (1024 * 1024)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def gpu_memory_allocated_mb() -> float:
    """Current GPU memory allocated (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


# ── Backbone Variants ────────────────────────────────────────────

def create_fp32_backbone(cfg: OrchestratorConfig) -> nn.Module:
    """FP32 baseline backbone."""
    return MiniTransformerLM(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.expert_embed_dim,
        num_layers=cfg.expert_layers,
        num_heads=cfg.expert_heads,
        context_len=cfg.context_len,
        mlp_hidden=cfg.expert_embed_dim * 4,
    )


def create_fp16_backbone(cfg: OrchestratorConfig) -> nn.Module:
    """FP16 backbone (half precision)."""
    model = create_fp32_backbone(cfg)
    return model.half()


def create_int8_backbone(cfg: OrchestratorConfig) -> nn.Module:
    """INT8 dynamic quantization (CPU only in PyTorch)."""
    model = create_fp32_backbone(cfg)
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def create_ternary_backbone(cfg: OrchestratorConfig) -> nn.Module:
    """Ternary BitNet {-1, 0, +1} backbone."""
    model = create_fp32_backbone(cfg)
    return quantize_model_ternary(model)


# ── Training Loop (mini) ────────────────────────────────────────

def train_orchestrator(
    model: SpectralAIOrchestrator,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    label: str,
) -> Dict:
    """Train orchestrator for a few epochs, return metrics."""
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_routing_loss = 0.0
        n_batches = 0
        correct_routing = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"[{label}] Epoch {epoch+1}/{epochs}",
                          leave=False):
            tokens, domain_ids = batch
            tokens = tokens.to(device)
            domain_ids = domain_ids.to(device)

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits, info = model(inputs, targets=targets, domain_ids=domain_ids)

            loss = info['total_loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += info['task_loss'].item()
            if 'routing_loss' in info:
                total_routing_loss += info['routing_loss'].item()
            n_batches += 1

            # Routing accuracy
            acc_info = model.routing_accuracy(info['expert_id'], domain_ids)
            correct_routing += acc_info['accuracy'] * tokens.shape[0]
            total_samples += tokens.shape[0]

        avg_loss = total_loss / max(n_batches, 1)
        avg_routing = total_routing_loss / max(n_batches, 1)
        routing_acc = correct_routing / max(total_samples, 1)

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'routing_loss': avg_routing,
            'routing_accuracy': routing_acc,
        })

    return {
        'final_train_loss': history[-1]['train_loss'] if history else float('inf'),
        'final_routing_acc': history[-1]['routing_accuracy'] if history else 0.0,
        'history': history,
    }


# ── Evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate_orchestrator(
    model: SpectralAIOrchestrator,
    val_loader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate perplexity, routing accuracy, per-domain metrics."""
    model.eval()
    model.to(device)

    total_loss = 0.0
    n_tokens = 0
    correct_routing = 0
    total_samples = 0
    domain_correct = {d: 0 for d in range(N_DOMAINS)}
    domain_total = {d: 0 for d in range(N_DOMAINS)}
    expert_usage = {d: set() for d in range(N_DOMAINS)}

    for batch in val_loader:
        tokens, domain_ids = batch
        tokens = tokens.to(device)
        domain_ids = domain_ids.to(device)

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, info = model(inputs, targets=targets, domain_ids=domain_ids)

        # Per-token loss
        loss = F.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size),
            targets.reshape(-1),
            reduction='sum',
        )
        total_loss += loss.item()
        n_tokens += targets.numel()

        # Routing accuracy
        acc_info = model.routing_accuracy(info['expert_id'], domain_ids)
        correct_routing += acc_info['accuracy'] * tokens.shape[0]
        total_samples += tokens.shape[0]

        # Per-domain tracking
        domain_ids_cpu = domain_ids.cpu()
        expert_ids_cpu = info['expert_id'].cpu()
        for i in range(tokens.shape[0]):
            d = domain_ids_cpu[i].item()
            eid = expert_ids_cpu[i].item()
            experts_per_domain = model.cfg.n_experts // N_DOMAINS
            pred_domain = eid // experts_per_domain
            domain_total[d] += 1
            if pred_domain == d:
                domain_correct[d] += 1
            expert_usage[d].add(eid)

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = min(float(np.exp(avg_loss)), 1e6)  # Cap at 1M
    routing_acc = correct_routing / max(total_samples, 1)

    per_domain_acc = {}
    for d in range(N_DOMAINS):
        if domain_total[d] > 0:
            per_domain_acc[DOMAIN_NAMES[d]] = domain_correct[d] / domain_total[d]
        else:
            per_domain_acc[DOMAIN_NAMES[d]] = 0.0

    unique_experts = {DOMAIN_NAMES[d]: len(expert_usage[d]) for d in range(N_DOMAINS)}

    return {
        'val_loss': avg_loss,
        'val_ppl': ppl,
        'routing_accuracy': routing_acc,
        'per_domain_accuracy': per_domain_acc,
        'unique_experts_per_domain': unique_experts,
    }


# ── Throughput Benchmark ────────────────────────────────────────

@torch.no_grad()
def benchmark_throughput(
    model: SpectralAIOrchestrator,
    device: str,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    warmup: int = WARMUP_BATCHES,
    n_batches: int = BENCHMARK_BATCHES,
) -> Dict:
    """Measure inference throughput (tok/s) and latency."""
    model.eval()
    model.to(device)

    # Generate random input
    tokens = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), device=device)
    domain_ids = torch.randint(0, N_DOMAINS, (batch_size,), device=device)

    # Use autocast to handle mixed dtypes (FP16 backbone + FP32 router)
    use_autocast = device == "cuda" and any(
        p.dtype == torch.float16 for p in model.parameters()
    )

    # Warmup
    for _ in range(warmup):
        with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
            _ = model(tokens[:, :-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_batches):
        with torch.amp.autocast(device_type="cuda", enabled=use_autocast):
            _ = model(tokens[:, :-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    total_tokens = batch_size * (seq_len - 1) * n_batches
    throughput = total_tokens / elapsed
    latency_ms = (elapsed / n_batches) * 1000

    return {
        'throughput_tok_s': throughput,
        'latency_ms_per_batch': latency_ms,
        'total_tokens': total_tokens,
        'elapsed_s': elapsed,
    }


# ── Quantize Orchestrator Backbone ──────────────────────────────

def create_orchestrator_variant(
    variant: str,
    cfg: OrchestratorConfig,
    device: str,
) -> SpectralAIOrchestrator:
    """
    Create orchestrator with specific backbone type.
    Returns a new model (immutable — never modifies originals).
    """
    model = SpectralAIOrchestrator(cfg, device=torch.device(device))

    if variant == "fp32":
        # Default — already FP32
        pass

    elif variant == "fp16":
        # FP16 created as FP32 here — post-training .half() applied in main loop
        # (training in FP32 is more stable, quantize after)
        pass

    elif variant == "ternary":
        # Quantize backbone linear layers to ternary
        model.expert_backbone = quantize_model_ternary(model.expert_backbone)

    elif variant == "int8_cpu":
        # INT8: start as FP32, quantize post-training (handled in main loop)
        pass

    return model


# ── Main Benchmark ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FASE 5 — Benchmark de Cuantización")
    print("SpectralAI v5.0 Orchestrator")
    print("=" * 70)

    device = DEVICE
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Config ───────────────────────────────────────────────────
    cfg = OrchestratorConfig(
        vocab_size=50_257,
        router_embed_dim=256,
        expert_embed_dim=128,
        n_level1=4, n_level2=4, n_level3=4,  # 64 expertos
        expert_layers=2,
        expert_heads=4,
        context_len=SEQ_LEN,
        spectral_dim=64,
        alpha_balance=0.01,
        alpha_router=0.1,
    )
    print(f"\nConfig: {cfg.n_experts} expertos, embed={cfg.expert_embed_dim}, "
          f"layers={cfg.expert_layers}, context={cfg.context_len}")

    # ── Datasets ─────────────────────────────────────────────────
    print("\n--- Cargando datasets multi-dominio ---")
    train_ds = create_multi_domain_dataset(
        split="train", seq_len=SEQ_LEN,
        max_tokens_per_domain=MAX_TOKENS_PER_DOMAIN,
    )
    val_ds = create_multi_domain_dataset(
        split="val", seq_len=SEQ_LEN,
        max_tokens_per_domain=MAX_TOKENS_PER_DOMAIN // 5,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_with_domain, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_with_domain, num_workers=0, drop_last=True,
    )

    print(f"Train: {len(train_ds)} muestras | Val: {len(val_ds)} muestras")

    # ── Variants to benchmark ────────────────────────────────────
    # INT8 solo en CPU (PyTorch limitation), los demás en GPU
    gpu_variants = ["fp32", "fp16", "ternary"]
    cpu_variants = ["int8_cpu"]

    # Determinar qué variantes correr
    if device == "cuda":
        variants = gpu_variants + cpu_variants
    else:
        variants = ["fp32", "ternary", "int8_cpu"]

    results = {}

    for variant in variants:
        print(f"\n{'='*70}")
        print(f"  VARIANTE: {variant.upper()}")
        print(f"{'='*70}")

        # Device para esta variante
        var_device = "cpu" if variant == "int8_cpu" else device

        # ── Crear modelo ─────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            vram_before = gpu_memory_allocated_mb()

        model = create_orchestrator_variant(variant, cfg, var_device)
        model = model.to(var_device)

        # ── Métricas estáticas ───────────────────────────────────
        n_params = count_params(model)
        mem_mb = model_memory_mb(model)
        disk_mb = model_disk_size_mb(model)

        if torch.cuda.is_available() and var_device == "cuda":
            vram_after = gpu_memory_allocated_mb()
            vram_model = vram_after - vram_before
        else:
            vram_model = 0.0

        print(f"\n  Params: {n_params:,}")
        print(f"  Memory (param+buffer): {mem_mb:.2f} MB")
        print(f"  Disk size: {disk_mb:.2f} MB")
        if vram_model > 0:
            print(f"  VRAM allocated: {vram_model:.2f} MB")

        # ── Backbone-only stats ──────────────────────────────────
        backbone_mem = model_memory_mb(model.expert_backbone)
        backbone_params = sum(p.numel() for p in model.expert_backbone.parameters())
        backbone_params += sum(b.numel() for b in model.expert_backbone.buffers())
        print(f"  Backbone: {backbone_params:,} params, {backbone_mem:.2f} MB")

        # ── Train ────────────────────────────────────────────────
        print(f"\n  Entrenando {TRAINING_EPOCHS} epochs...")
        t_start = time.perf_counter()

        # FP16 y Ternario: entrenar en FP32, cuantizar después (post-training quant)
        # Esto es necesario porque:
        #   - FP16 training sin GradScaler es inestable
        #   - TernaryLinear usa int8 buffers, no nn.Parameter → optimizer no los actualiza
        # Todas las variantes cuantizadas entrenan en FP32 y cuantizan después
        needs_post_quant = variant in ("fp16", "ternary", "int8_cpu")
        if needs_post_quant:
            model_train = create_orchestrator_variant("fp32", cfg, var_device)
            model_train = model_train.to(var_device)
        else:
            model_train = copy.deepcopy(model)  # Never alias — always independent copy

        train_metrics = train_orchestrator(
            model_train, train_loader, TRAINING_EPOCHS, LR, var_device,
            label=variant,
        )
        train_time = time.perf_counter() - t_start

        # Post-training quantization: apply quantization to trained weights
        if variant == "fp16":
            model_eval = copy.deepcopy(model_train)
            # Convert ONLY the expert backbone to FP16 — router must stay FP32
            # (FP16 router crashes on einsum in portal.apply_all)
            model_eval.expert_backbone = model_eval.expert_backbone.half()
        elif variant == "ternary":
            model_eval = copy.deepcopy(model_train)
            model_eval.expert_backbone = quantize_model_ternary(
                model_eval.expert_backbone
            )
        elif variant == "int8_cpu":
            model_eval = copy.deepcopy(model_train)
            model_eval.expert_backbone = torch.ao.quantization.quantize_dynamic(
                model_eval.expert_backbone, {nn.Linear}, dtype=torch.qint8
            )
        else:
            model_eval = copy.deepcopy(model_train)  # FP32: independent copy

        print(f"  Train time: {train_time:.1f}s")
        print(f"  Final train loss: {train_metrics['final_train_loss']:.4f}")
        print(f"  Final routing acc: {train_metrics['final_routing_acc']*100:.1f}%")

        # ── Evaluate ─────────────────────────────────────────────
        print(f"\n  Evaluando en validación...")

        # FP16 model is fully .half() — no need for autocast wrapper
        eval_metrics = evaluate_orchestrator(model_eval, val_loader, var_device)

        print(f"  Val loss: {eval_metrics['val_loss']:.4f}")
        print(f"  Val ppl:  {eval_metrics['val_ppl']:.1f}")
        print(f"  Routing accuracy: {eval_metrics['routing_accuracy']*100:.1f}%")
        for d_name, acc in eval_metrics['per_domain_accuracy'].items():
            print(f"    {d_name:>10s}: {acc*100:.1f}%")
        print(f"  Expertos únicos: {eval_metrics['unique_experts_per_domain']}")

        # ── Throughput ───────────────────────────────────────────
        print(f"\n  Benchmarking throughput...")
        tp_metrics = benchmark_throughput(
            model_eval, var_device,
            batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        )
        print(f"  Throughput: {tp_metrics['throughput_tok_s']:,.0f} tok/s")
        print(f"  Latency: {tp_metrics['latency_ms_per_batch']:.2f} ms/batch")

        # ── Post-quant stats + peak VRAM ─────────────────────────
        post_mem = model_memory_mb(model_eval)
        post_disk = model_disk_size_mb(model_eval)

        if torch.cuda.is_available() and var_device == "cuda":
            vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            vram_peak = 0.0

        # ── Collect results ──────────────────────────────────────
        results[variant] = {
            'device': var_device,
            'params': n_params,
            'memory_mb': post_mem,
            'disk_mb': post_disk,
            'vram_mb': vram_model,
            'vram_peak_mb': vram_peak,
            'backbone_params': backbone_params,
            'backbone_mb': backbone_mem,
            'train_time_s': train_time,
            'train_loss': train_metrics['final_train_loss'],
            'routing_accuracy': train_metrics['final_routing_acc'],
            'val_loss': eval_metrics['val_loss'],
            'val_ppl': eval_metrics['val_ppl'],
            'val_routing_accuracy': eval_metrics['routing_accuracy'],
            'per_domain_accuracy': eval_metrics['per_domain_accuracy'],
            'unique_experts': eval_metrics['unique_experts_per_domain'],
            'throughput_tok_s': tp_metrics['throughput_tok_s'],
            'latency_ms': tp_metrics['latency_ms_per_batch'],
        }

        # Cleanup
        del model, model_train, model_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN FINAL — FASE 5: Benchmark de Cuantización")
    print("=" * 70)

    # Header
    print(f"\n{'Variante':<12s} | {'Device':<6s} | {'Mem MB':>7s} | {'Disk MB':>7s} | "
          f"{'Val PPL':>10s} | {'Route%':>7s} | {'tok/s':>10s} | {'Lat ms':>7s}")
    print("-" * 90)

    # Rows
    baseline_throughput = results.get("fp32", {}).get('throughput_tok_s', 1.0)
    for v, r in results.items():
        speedup = r['throughput_tok_s'] / baseline_throughput if baseline_throughput > 0 else 0
        print(f"{v:<12s} | {r['device']:<6s} | {r['memory_mb']:>7.2f} | {r['disk_mb']:>7.2f} | "
              f"{r['val_ppl']:>10.1f} | {r['val_routing_accuracy']*100:>6.1f}% | "
              f"{r['throughput_tok_s']:>10,.0f} | {r['latency_ms']:>7.2f}")

    print(f"\n  NOTA: INT8 corre en CPU — throughput NO es comparable con variantes GPU.")

    # Compression ratios
    if "fp32" in results:
        fp32_mem = results["fp32"]["memory_mb"]
        fp32_disk = results["fp32"]["disk_mb"]
        print(f"\n--- Ratios de compresión vs FP32 ---")
        for v, r in results.items():
            if v == "fp32":
                continue
            mem_ratio = fp32_mem / max(r['memory_mb'], 0.01)
            disk_ratio = fp32_disk / max(r['disk_mb'], 0.01)
            speed_ratio = r['throughput_tok_s'] / baseline_throughput
            print(f"  {v:<12s}: Mem {mem_ratio:.2f}x | Disk {disk_ratio:.2f}x | "
                  f"Speed {speed_ratio:.2f}x | PPL delta {r['val_ppl'] - results['fp32']['val_ppl']:+.1f}")

    # Per-domain routing accuracy
    print(f"\n--- Routing Accuracy por Dominio ---")
    print(f"{'Variante':<12s} | {'general':>8s} | {'code':>8s} | {'science':>8s} | {'legal':>8s}")
    print("-" * 55)
    for v, r in results.items():
        pda = r['per_domain_accuracy']
        print(f"{v:<12s} | {pda.get('general',0)*100:>7.1f}% | {pda.get('code',0)*100:>7.1f}% | "
              f"{pda.get('science',0)*100:>7.1f}% | {pda.get('legal',0)*100:>7.1f}%")

    # Expert specialization
    print(f"\n--- Expertos Únicos por Dominio ---")
    print(f"{'Variante':<12s} | {'general':>8s} | {'code':>8s} | {'science':>8s} | {'legal':>8s}")
    print("-" * 55)
    for v, r in results.items():
        ue = r['unique_experts']
        print(f"{v:<12s} | {ue.get('general',0):>8d} | {ue.get('code',0):>8d} | "
              f"{ue.get('science',0):>8d} | {ue.get('legal',0):>8d}")

    # ── Save results ─────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fase5_quantization_benchmark.json"

    # Convert sets to lists for JSON serialization
    json_results = {}
    for v, r in results.items():
        json_results[v] = {k: (list(val) if isinstance(val, set) else val)
                           for k, val in r.items()}

    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResultados guardados: {out_path}")

    print("\n" + "=" * 70)
    print("FASE 5 COMPLETADA")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
