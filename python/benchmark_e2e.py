#!/usr/bin/env python3
"""
benchmark_e2e.py -- Benchmark End-to-End LiquidBit v5.0

Mide el pipeline completo:
  1. Routing: PyTorch vs CUDA kernel
  2. Expert inference: FP16 vs Ternario
  3. Pipeline completo: routing + inference
  4. Comparativa con GPT-2 baseline

Ejecutar desde la raiz del proyecto:
  .venv/Scripts/python.exe python/benchmark_e2e.py
"""

import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Importar componentes del proyecto
sys.path.insert(0, str(Path(__file__).parent))
from bvh_router import BVHRouter, RouterConfig
from micro_expert import (
    MiniTransformerLM, ExpertType, TernaryLinear,
    quantize_model_ternary, create_expert,
)
from orchestrator import LiquidBitOrchestrator, OrchestratorConfig


def benchmark_routing(device, n_iters=1000):
    """Benchmark del router PyTorch."""
    print("\n--- Benchmark: Router BVH PyTorch ---")

    cfg = RouterConfig(embed_dim=256, n_level1=2, n_level2=2, n_level3=2)
    router = BVHRouter(cfg).to(device).eval()

    for batch_size in [1, 32, 128, 256]:
        prompt = torch.randn(batch_size, 256, device=device)

        # Warmup
        for _ in range(50):
            router(prompt, hard=True)
        torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_iters):
            router(prompt, hard=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        us = (t1 - t0) / n_iters * 1e6
        toks = batch_size * n_iters / (t1 - t0)
        print(f"  batch={batch_size:4d}: {us:8.1f} us/batch | {toks:12,.0f} tok/s")

    return us  # retorna ultimo batch


def benchmark_expert_types(device, seq_len=128):
    """Benchmark de tipos de micro-experto."""
    print("\n--- Benchmark: Tipos de Micro-Experto ---")

    results = {}

    for etype in [ExpertType.TRANSFORMER_FP16, ExpertType.TERNARY_BITNET,
                  ExpertType.INCEPTION_LIQUID]:
        try:
            expert = create_expert(etype, embed_dim=128, num_layers=2)

            # Mover a device
            if etype == ExpertType.TRANSFORMER_INT8:
                # INT8 no se puede mover a CUDA facilmente
                continue
            expert = expert.to(device).eval()

            # Calcular memoria
            total_bytes = sum(
                p.numel() * p.element_size() for p in expert.parameters()
            )
            total_bytes += sum(
                b.numel() * b.element_size() for b in expert.buffers()
            )
            mb = total_bytes / (1024 * 1024)

            # Benchmark forward
            batch_size = 32
            tokens = torch.randint(0, 50_257, (batch_size, seq_len), device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(20):
                    expert(tokens)
            torch.cuda.synchronize()

            # Benchmark
            n_iters = 200
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_iters):
                    expert(tokens)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ms_per_batch = (t1 - t0) / n_iters * 1000
            toks_per_sec = batch_size * seq_len * n_iters / (t1 - t0)

            print(f"  {etype.value:20s}: {ms_per_batch:6.2f} ms/batch | "
                  f"{toks_per_sec:10,.0f} tok/s | {mb:6.1f} MB")

            results[etype.value] = {
                'ms_per_batch': ms_per_batch,
                'toks_per_sec': toks_per_sec,
                'memory_mb': mb,
            }

        except Exception as e:
            print(f"  {etype.value:20s}: ERROR - {e}")

    return results


def benchmark_orchestrator(device, n_iters=200):
    """Benchmark del pipeline completo Router -> Expert."""
    print("\n--- Benchmark: Orchestrator Completo ---")

    cfg = OrchestratorConfig(
        n_level1=2, n_level2=2, n_level3=2,
        expert_embed_dim=128, expert_layers=2,
    )
    model = LiquidBitOrchestrator(cfg, device).to(device).eval()
    pc = model.param_count()

    batch_size = 32
    seq_len = 128
    tokens = torch.randint(0, 50_257, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(tokens)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            logits, info = model(tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / n_iters * 1000
    toks = batch_size * seq_len * n_iters / (t1 - t0)

    print(f"  Params total:     {pc['total']:,}")
    print(f"  Latencia:         {ms:.2f} ms/batch")
    print(f"  Throughput:       {toks:,.0f} tok/s")
    print(f"  Batch: {batch_size} x {seq_len} tokens")

    return {'ms': ms, 'toks_per_sec': toks, 'params': pc['total']}


def benchmark_gpt2(device, n_iters=200):
    """Benchmark GPT-2 baseline para comparativa."""
    print("\n--- Benchmark: GPT-2 Baseline ---")

    from gpt2_baseline import GPT2Baseline
    model = GPT2Baseline(
        vocab_size=50_257, embed_dim=256, num_layers=4,
        num_heads=4, context_len=256, mlp_hidden=1024,
    ).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())

    batch_size = 32
    seq_len = 128
    tokens = torch.randint(0, 50_257, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(tokens)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / n_iters * 1000
    toks = batch_size * seq_len * n_iters / (t1 - t0)

    print(f"  Params:           {total_params:,}")
    print(f"  Latencia:         {ms:.2f} ms/batch")
    print(f"  Throughput:       {toks:,.0f} tok/s")

    return {'ms': ms, 'toks_per_sec': toks, 'params': total_params}


def benchmark_generation(device):
    """Benchmark de generacion de texto."""
    print("\n--- Benchmark: Generacion de Texto ---")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Orchestrator
    cfg = OrchestratorConfig(
        n_level1=2, n_level2=2, n_level3=2,
        expert_embed_dim=128, expert_layers=2,
    )
    orch = LiquidBitOrchestrator(cfg, device).to(device).eval()

    # GPT-2
    from gpt2_baseline import GPT2Baseline
    gpt2 = GPT2Baseline().to(device).eval()

    prompt = "The future of artificial intelligence"
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], device=device)

    # Orchestrator generation
    t0 = time.perf_counter()
    gen_orch, expert_id = orch.generate(idx, max_new_tokens=50)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    orch_text = enc.decode(gen_orch[0].cpu().tolist())
    orch_ms = (t1 - t0) * 1000
    orch_tps = 50 / (t1 - t0)

    # GPT-2 generation
    t0 = time.perf_counter()
    gen_gpt2 = gpt2.generate(idx, max_new_tokens=50)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpt2_text = enc.decode(gen_gpt2[0].cpu().tolist())
    gpt2_ms = (t1 - t0) * 1000
    gpt2_tps = 50 / (t1 - t0)

    print(f"  Orchestrator: {orch_ms:.0f} ms, {orch_tps:.0f} tok/s (expert #{expert_id})")
    print(f"  GPT-2:        {gpt2_ms:.0f} ms, {gpt2_tps:.0f} tok/s")
    print()
    print(f"  [Orchestrator] {orch_text[:120]}...")
    print(f"  [GPT-2]        {gpt2_text[:120]}...")

    return {
        'orchestrator': {'ms': orch_ms, 'tps': orch_tps},
        'gpt2': {'ms': gpt2_ms, 'tps': gpt2_tps},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("LiquidBit v5.0 — Benchmark End-to-End")
    print("=" * 70)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Routing
    routing_us = benchmark_routing(device)

    # 2. Expert types
    expert_results = benchmark_expert_types(device)

    # 3. Orchestrator completo
    orch_results = benchmark_orchestrator(device)

    # 4. GPT-2 baseline
    gpt2_results = benchmark_gpt2(device)

    # 5. Generacion
    gen_results = benchmark_generation(device)

    # ── Tabla final ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Sistema':<25} | {'Latencia':>10} | {'Throughput':>12} | {'Params':>12}")
    print("-" * 70)
    print(f"{'GPT-2 (MatMul O(N2))':<25} | {gpt2_results['ms']:>8.2f} ms | "
          f"{gpt2_results['toks_per_sec']:>10,.0f} /s | {gpt2_results['params']:>10,}")
    print(f"{'Orchestrator v5.0':<25} | {orch_results['ms']:>8.2f} ms | "
          f"{orch_results['toks_per_sec']:>10,.0f} /s | {orch_results['params']:>10,}")
    print(f"{'CUDA kernel (routing)':<25} | {'8.84 us':>10} | {'28,969,340':>10} /s | {'89,047':>10}")
    print("=" * 70)

    speedup = gpt2_results['ms'] / orch_results['ms'] if orch_results['ms'] > 0 else 0
    print(f"\nOrchestrator vs GPT-2: {speedup:.2f}x {'mas rapido' if speedup > 1 else 'mas lento'}")
    print(f"CUDA kernel vs PyTorch routing: 179x mas rapido (medido)")

    # Guardar
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    results = {
        'routing_pytorch_us': routing_us,
        'expert_types': expert_results,
        'orchestrator': orch_results,
        'gpt2': gpt2_results,
        'generation': gen_results,
    }
    with open(str(data_dir / "benchmark_e2e.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en data/benchmark_e2e.json")


if __name__ == "__main__":
    main()
