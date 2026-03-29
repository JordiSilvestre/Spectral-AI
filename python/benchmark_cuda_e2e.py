#!/usr/bin/env python3
"""
benchmark_cuda_e2e.py — SpectralAI v5.0: CUDA Kernel conectado al Orchestrator

Mide el pipeline completo con el kernel CUDA integrado:

  1. Routing solo:
     PyTorch BVHRouter (batch=256) vs CUDA kernel (batch=256)
  2. Orchestrator completo:
     Router PyTorch + Backbone vs Router CUDA + Backbone
  3. GPT-2 baseline

Para resultados reales con CUDA kernel: ejecutar desde WSL2.
En Windows: muestra solo PyTorch + GPT-2 (fallback automatico).

Ejecutar:
    # WSL2 (numeros reales con CUDA kernel):
    python3 python/benchmark_cuda_e2e.py

    # Windows (solo PyTorch):
    .venv/Scripts/python.exe python/benchmark_cuda_e2e.py
"""

import sys
import time
import json
import platform
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bvh_router import BVHRouter, RouterConfig
from bvh_router_hybrid import HybridBVHRouter, BVH_LEAVES
from orchestrator import SpectralAIOrchestrator, OrchestratorConfig
from gpt2_baseline import GPT2Baseline


# ─────────────────────────────────────────────────────────────────
# Configuracion: 4x4x4 = 64 expertos (coincide con CUDA kernel)
# ─────────────────────────────────────────────────────────────────

def make_orch_config_64() -> OrchestratorConfig:
    """Configuracion con 64 expertos (necesario para el CUDA kernel)."""
    return OrchestratorConfig(
        n_level1=4, n_level2=4, n_level3=4,
        expert_embed_dim=128, expert_layers=2,
    )


# ─────────────────────────────────────────────────────────────────
# Benchmark 1: Routing solo
# ─────────────────────────────────────────────────────────────────

def benchmark_routing(device: torch.device, n_iters: int = 500) -> dict:
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Routing — PyTorch vs CUDA Kernel")
    print("=" * 70)

    cfg = RouterConfig(embed_dim=256, n_level1=4, n_level2=4, n_level3=4)
    pytorch_router = BVHRouter(cfg).to(device).eval()
    hybrid_router  = HybridBVHRouter(pytorch_router, device=str(device))

    results = {}

    for batch_size in [1, 32, 128, 256]:
        emb = torch.randn(batch_size, 256, device=device)

        # --- PyTorch ---
        for _ in range(50):
            pytorch_router(emb, hard=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            pytorch_router(emb, hard=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        pt_us   = (t1 - t0) / n_iters * 1e6
        pt_toks = batch_size * n_iters / (t1 - t0)

        # --- CUDA Hybrid ---
        if hybrid_router.cuda_available:
            for _ in range(50):
                hybrid_router.route(emb)

            t0 = time.perf_counter()
            for _ in range(n_iters):
                hybrid_router.route(emb)
            t1 = time.perf_counter()

            cu_us   = (t1 - t0) / n_iters * 1e6
            cu_toks = batch_size * n_iters / (t1 - t0)
            speedup = pt_us / cu_us
            cu_str  = f"{cu_us:8.1f} us | {speedup:5.1f}x speedup"
        else:
            cu_us   = None
            cu_str  = "N/A (ejecutar desde WSL2)"

        print(f"  batch={batch_size:4d} | PyTorch: {pt_us:8.1f} us | CUDA: {cu_str}")
        results[batch_size] = {"pytorch_us": pt_us, "cuda_us": cu_us}

    return results


# ─────────────────────────────────────────────────────────────────
# Benchmark 2: Orchestrator completo
# ─────────────────────────────────────────────────────────────────

def benchmark_orchestrator_pytorch(device: torch.device,
                                   n_iters: int = 200) -> dict:
    """Orchestrator con router PyTorch puro."""
    cfg   = make_orch_config_64()
    model = SpectralAIOrchestrator(cfg, device).to(device).eval()
    pc    = model.param_count()

    batch_size = 32
    seq_len    = 128
    tokens     = torch.randint(0, 50_257, (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(20):
            model(tokens)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            logits, _ = model(tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms   = (t1 - t0) / n_iters * 1000
    toks = batch_size * seq_len * n_iters / (t1 - t0)

    print(f"\n  [PyTorch Router] {ms:.2f} ms/batch | {toks:,.0f} tok/s | {pc['total']:,} params")
    return {"ms": ms, "toks": toks, "params": pc["total"]}


def benchmark_orchestrator_cuda(device: torch.device,
                                n_iters: int = 200) -> dict:
    """
    Orchestrator con CUDA kernel para routing.
    Sustituye router.forward() por hybrid_router.route() en cada forward pass.
    """
    cfg   = make_orch_config_64()
    model = SpectralAIOrchestrator(cfg, device).to(device).eval()

    # Crear hybrid router usando el router interno del modelo
    hybrid = HybridBVHRouter(model.router, device=str(device))
    if not hybrid.cuda_available:
        return {"ms": None, "toks": None, "note": "CUDA kernel no disponible"}

    # Parchear el modelo para usar CUDA routing via context manager
    from contextlib import contextmanager

    @contextmanager
    def patched_router(mdl, new_forward):
        original_fwd = mdl.router.forward
        mdl.router.forward = new_forward
        try:
            yield
        finally:
            mdl.router.forward = original_fwd

    batch_size = 32
    seq_len    = 128
    tokens     = torch.randint(0, 50_257, (batch_size, seq_len), device=device)

    with patched_router(model, hybrid.route):
        with torch.no_grad():
            for _ in range(20):
                model(tokens)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                logits, _ = model(tokens)
        t1 = time.perf_counter()

    ms   = (t1 - t0) / n_iters * 1000
    toks = batch_size * seq_len * n_iters / (t1 - t0)

    print(f"  [CUDA Router]    {ms:.2f} ms/batch | {toks:,.0f} tok/s")
    return {"ms": ms, "toks": toks}


# ─────────────────────────────────────────────────────────────────
# Benchmark 3: GPT-2 baseline
# ─────────────────────────────────────────────────────────────────

def benchmark_gpt2(device: torch.device, n_iters: int = 200) -> dict:
    model = GPT2Baseline(
        vocab_size=50_257, embed_dim=256, num_layers=4,
        num_heads=4, context_len=256, mlp_hidden=1024,
    ).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    batch_size = 32
    seq_len    = 128
    tokens     = torch.randint(0, 50_257, (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(20):
            model(tokens)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms   = (t1 - t0) / n_iters * 1000
    toks = batch_size * seq_len * n_iters / (t1 - t0)

    print(f"  [GPT-2 baseline] {ms:.2f} ms/batch | {toks:,.0f} tok/s | {total_params:,} params")
    return {"ms": ms, "toks": toks, "params": total_params}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--n-iters",   type=int, default=200)
    parser.add_argument("--routing-iters", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("SpectralAI v5.0 — CUDA Kernel Conectado al Orchestrator")
    print("=" * 70)
    print(f"Sistema:    {platform.system()} {platform.release()}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
        print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Routing
    routing_results = benchmark_routing(device, args.routing_iters)

    # 2. Orchestrator
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Orchestrator Completo (batch=32, seq=128)")
    print("=" * 70)
    orch_pt  = benchmark_orchestrator_pytorch(device, args.n_iters)
    orch_cu  = benchmark_orchestrator_cuda(device, args.n_iters)

    # 3. GPT-2
    print("\n" + "=" * 70)
    print("BENCHMARK 3: GPT-2 Baseline")
    print("=" * 70)
    gpt2_r = benchmark_gpt2(device, args.n_iters)

    # ── Tabla final ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO FINAL")
    print("=" * 70)
    print(f"{'Sistema':<32} | {'Latencia':>10} | {'Throughput':>14} | {'Params':>12}")
    print("-" * 75)
    print(f"{'GPT-2 (MatMul O(N2))':<32} | {gpt2_r['ms']:>8.2f} ms | "
          f"{gpt2_r['toks']:>12,.0f} /s | {gpt2_r['params']:>12,}")
    print(f"{'Orchestrator (PyTorch router)':<32} | {orch_pt['ms']:>8.2f} ms | "
          f"{orch_pt['toks']:>12,.0f} /s | {orch_pt['params']:>12,}")

    if orch_cu.get("ms"):
        print(f"{'Orchestrator (CUDA router)':<32} | {orch_cu['ms']:>8.2f} ms | "
              f"{orch_cu['toks']:>12,.0f} /s | {orch_pt['params']:>12,}")
    else:
        print(f"{'Orchestrator (CUDA router)':<32} | {'WSL2':>10} | {'requerido':>14} |")

    print("-" * 75)

    # Kernel routing speedup (batch=256)
    r256 = routing_results.get(256, {})
    pt256_us = r256.get("pytorch_us", 0)
    cu256_us = r256.get("cuda_us")
    if cu256_us:
        kern_speedup = pt256_us / cu256_us
        print(f"\n  Kernel CUDA vs PyTorch (routing batch=256): {kern_speedup:.0f}x mas rapido")
        print(f"    PyTorch: {pt256_us:.1f} us | CUDA: {cu256_us:.1f} us")

    sp_pt = gpt2_r["ms"] / orch_pt["ms"] if orch_pt["ms"] > 0 else 0
    print(f"\n  Orchestrator (PyTorch) vs GPT-2: {sp_pt:.2f}x {'mas rapido' if sp_pt > 1 else 'mas lento'}")

    if orch_cu.get("ms"):
        sp_cu = gpt2_r["ms"] / orch_cu["ms"] if orch_cu["ms"] > 0 else 0
        print(f"  Orchestrator (CUDA)    vs GPT-2: {sp_cu:.2f}x {'mas rapido' if sp_cu > 1 else 'mas lento'}")

    # Guardar
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    results = {
        "routing":             {str(k): v for k, v in routing_results.items()},
        "orchestrator_pytorch": orch_pt,
        "orchestrator_cuda":   orch_cu,
        "gpt2":                gpt2_r,
    }
    out_path = data_dir / "benchmark_cuda_e2e.json"
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en data/benchmark_cuda_e2e.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
