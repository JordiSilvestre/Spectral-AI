#!/usr/bin/env python3
"""
benchmark_cuda_pipeline.py — Full CUDA Pipeline Benchmark

Benchmarks all SpectralAI CUDA components individually and end-to-end:
  1. BVH Router CUDA Kernel (bvh_router_ext) — 105x speedup target
  2. Ternary Expert POPCOUNT (ternary_expert_ext) — Zero-multiplication MLP
  3. Full pipeline: BVH routing -> expert selection -> ternary forward

Results are formatted for paper/patent documentation.

Usage:
    python benchmark_cuda_pipeline.py [--warmup 50] [--iters 500]

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

BVH_NODES = 85
BVH_LEAVES = 64
SPEC_DIM = 64
PORTAL_SIZE = 12

BATCH_SIZES = [1, 4, 16, 64, 256, 1024]
DEFAULT_WARMUP = 50
DEFAULT_ITERS = 500

TERNARY_CKPT_DIR = Path("checkpoints/ternary/ternary_experts")


# ─────────────────────────────────────────────────────────────────
# Extension loading
# ─────────────────────────────────────────────────────────────────

def load_bvh_ext():
    """Try to import bvh_router_ext from cuda/v5/."""
    ext_path = str(Path(__file__).parent.parent / "cuda" / "v5")
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    try:
        import bvh_router_ext
        return bvh_router_ext
    except ImportError as e:
        print(f"  [SKIP] bvh_router_ext not available: {e}")
        return None


def load_ternary_ext():
    """Try to import ternary_expert_ext from cuda/v5/."""
    ext_path = str(Path(__file__).parent.parent / "cuda" / "v5")
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    try:
        import ternary_expert_ext
        return ternary_expert_ext
    except ImportError as e:
        print(f"  [SKIP] ternary_expert_ext not available: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# BVH Router Benchmark
# ─────────────────────────────────────────────────────────────────

def create_synthetic_tree() -> Tuple[torch.Tensor, ...]:
    """Create a synthetic BVH tree with 85 nodes for benchmarking."""
    torch.manual_seed(42)
    centers = torch.randn(BVH_NODES, 3, dtype=torch.float32)
    radii = torch.ones(BVH_NODES, dtype=torch.float32) * 0.5
    portals = torch.zeros(BVH_NODES, PORTAL_SIZE, dtype=torch.float32)
    # Identity portals (3x4 affine = [I | 0])
    for i in range(BVH_NODES):
        portals[i, 0] = 1.0  # (0,0)
        portals[i, 4] = 1.0  # (1,1)
        portals[i, 8] = 1.0  # (2,2)
    snell_w = torch.randn(BVH_NODES, SPEC_DIM, dtype=torch.float32) * 0.1
    snell_b = torch.zeros(BVH_NODES, dtype=torch.float32)
    return centers, radii, portals, snell_w, snell_b


def benchmark_bvh_cuda(
    bvh_ext, warmup: int, iters: int
) -> Dict[int, dict]:
    """Benchmark BVH CUDA kernel across batch sizes."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 1: BVH Router CUDA Kernel")
    print("=" * 60)

    # Upload tree
    centers, radii, portals, snell_w, snell_b = create_synthetic_tree()
    bvh_ext.upload_tree(centers, radii, portals, snell_w, snell_b)
    print("  Tree uploaded (85 nodes, 64 leaves)")

    results = {}
    device = torch.device("cuda")

    for bs in BATCH_SIZES:
        # Prepare inputs OUTSIDE the timing loop (same methodology as certified benchmark)
        origins = torch.randn(bs, 3, device=device, dtype=torch.float32).contiguous()
        directions = F.normalize(
            torch.randn(bs, 3, device=device, dtype=torch.float32), dim=-1
        ).contiguous()
        spectral = torch.randn(bs, SPEC_DIM, device=device, dtype=torch.float32).contiguous()

        # Use async route (kernel only), sync once at end
        def kernel_call():
            bvh_ext.route(origins, directions, spectral)

        # Warmup
        for _ in range(warmup):
            kernel_call()
        torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(iters):
            kernel_call()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        avg_us = (elapsed / iters) * 1e6
        throughput = bs * iters / elapsed

        # Sample output (outside timing)
        expert_ids, scores, confidence, path = bvh_ext.route(
            origins, directions, spectral
        )
        results[bs] = {
            "avg_us": avg_us,
            "throughput_tok_s": throughput,
            "expert_ids_sample": expert_ids[:4].cpu().tolist(),
        }
        print(f"  batch={bs:>5d}  |  {avg_us:>8.2f} us  |  {throughput:>12,.0f} tok/s")

    return results


def benchmark_bvh_pytorch(warmup: int, iters: int) -> Dict[int, dict]:
    """Benchmark PyTorch BVHRouter as baseline."""
    print("\n  PyTorch baseline (BVHRouter forward)...")
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from bvh_router import BVHRouter, RouterConfig
    except ImportError:
        print("  [SKIP] bvh_router.py not importable")
        return {}

    cfg = RouterConfig(
        embed_dim=256,
        n_level1=4,
        n_level2=4,
        n_level3=4,
        spectral_dim=SPEC_DIM,
    )
    router = BVHRouter(cfg).cuda().eval()

    results = {}
    for bs in BATCH_SIZES:
        if bs > 256:
            continue  # PyTorch too slow for large batches
        x = torch.randn(bs, 256, device="cuda", dtype=torch.float32)

        # Warmup
        with torch.no_grad():
            for _ in range(min(warmup, 10)):
                router(x, hard=True)

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(min(iters, 100)):
                router(x, hard=True)
        torch.cuda.synchronize()
        actual_iters = min(iters, 100)
        elapsed = time.perf_counter() - t0

        avg_us = (elapsed / actual_iters) * 1e6
        results[bs] = {"avg_us": avg_us}
        print(f"  batch={bs:>5d}  |  {avg_us:>8.2f} µs  (PyTorch baseline)")

    return results


# ─────────────────────────────────────────────────────────────────
# Ternary Expert Benchmark
# ─────────────────────────────────────────────────────────────────

def load_ternary_weights(layer_idx: int = 0) -> dict:
    """Load real ternary expert weights from checkpoints."""
    layer_dir = TERNARY_CKPT_DIR / f"layer_{layer_idx}"
    if not layer_dir.exists():
        return {}

    weights = {}
    for name in ["gate", "up", "down"]:
        ternary_path = layer_dir / f"{name}_ternary.npy"
        scale_path = layer_dir / f"{name}_scale.npy"
        if ternary_path.exists() and scale_path.exists():
            weights[f"{name}_ternary"] = np.load(str(ternary_path))
            weights[f"{name}_scale"] = np.load(str(scale_path))

    return weights


def benchmark_ternary(
    tern_ext, warmup: int, iters: int
) -> Dict[str, dict]:
    """Benchmark ternary expert POPCOUNT vs FP16 linear."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 2: Ternary Expert POPCOUNT")
    print("=" * 60)

    # Load real weights
    weights = load_ternary_weights(layer_idx=0)
    if not weights:
        print("  [SKIP] No ternary weights found in checkpoints/ternary/ternary_experts/")
        return {}

    # Weights are stored as [out_features, in_features] (PyTorch convention)
    # Kernel expects [in_features, out_features] — transpose needed
    gate_tern = weights["gate_ternary"].T.copy()   # [896, 4864]
    gate_scale_np = weights["gate_scale"]           # [4864]
    up_tern = weights["up_ternary"].T.copy()        # [896, 4864]
    up_scale_np = weights["up_scale"]               # [4864]
    down_tern = weights["down_ternary"].T.copy()    # [4864, 896]
    down_scale_np = weights["down_scale"]           # [896]

    in_dim = gate_tern.shape[0]       # 896
    inter_dim = gate_tern.shape[1]    # 4864
    out_dim = down_tern.shape[1]      # 896

    print(f"  Expert dimensions: {in_dim} -> {inter_dim} (gate/up) -> {out_dim}")
    print(f"  Ternary compression: {gate_tern.nbytes / 1024:.1f} KB "
          f"(vs {in_dim * inter_dim * 2 / 1024:.1f} KB FP16)")

    # Pack ternary weights
    gate_packed = tern_ext.pack_ternary(
        torch.from_numpy(gate_tern.astype(np.int8))
    )
    up_packed = tern_ext.pack_ternary(
        torch.from_numpy(up_tern.astype(np.int8))
    )
    down_packed = tern_ext.pack_ternary(
        torch.from_numpy(down_tern.astype(np.int8))
    )

    # Move to GPU
    device = torch.device("cuda")
    gate_packed_gpu = gate_packed.to(device)
    up_packed_gpu = up_packed.to(device)
    down_packed_gpu = down_packed.to(device)
    gate_scale = torch.from_numpy(gate_scale_np).float().to(device)
    up_scale = torch.from_numpy(up_scale_np).float().to(device)
    down_scale = torch.from_numpy(down_scale_np).float().to(device)

    results = {}

    for bs in [1, 4, 16, 64, 256]:
        x = torch.randn(bs, in_dim, device=device, dtype=torch.float32)

        # ── Ternary POPCOUNT ──
        for _ in range(warmup):
            tern_ext.ternary_gated_mlp(
                x, gate_packed_gpu, up_packed_gpu, down_packed_gpu,
                gate_scale, up_scale, down_scale,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out_tern = tern_ext.ternary_gated_mlp(
                x, gate_packed_gpu, up_packed_gpu, down_packed_gpu,
                gate_scale, up_scale, down_scale,
            )
        torch.cuda.synchronize()
        tern_us = (time.perf_counter() - t0) / iters * 1e6

        # ── FP16 baseline ──
        gate_fp = nn.Linear(in_dim, inter_dim, bias=False).half().to(device)
        up_fp = nn.Linear(in_dim, inter_dim, bias=False).half().to(device)
        down_fp = nn.Linear(inter_dim, out_dim, bias=False).half().to(device)
        x_half = x.half()

        for _ in range(warmup):
            with torch.no_grad():
                h = F.silu(gate_fp(x_half)) * up_fp(x_half)
                down_fp(h)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                h = F.silu(gate_fp(x_half)) * up_fp(x_half)
                out_fp = down_fp(h)
        torch.cuda.synchronize()
        fp_us = (time.perf_counter() - t0) / iters * 1e6

        speedup = fp_us / tern_us if tern_us > 0 else 0
        results[bs] = {
            "ternary_us": tern_us,
            "fp16_us": fp_us,
            "speedup": speedup,
        }
        print(f"  batch={bs:>4d}  |  Ternary: {tern_us:>8.1f} µs  |  "
              f"FP16: {fp_us:>8.1f} µs  |  Speedup: {speedup:.1f}x")

    # Compression ratio
    fp16_size = (in_dim * inter_dim * 2 + in_dim * inter_dim * 2
                 + inter_dim * out_dim * 2)  # bytes
    tern_size = (gate_packed.numel() * 4 + up_packed.numel() * 4
                 + down_packed.numel() * 4
                 + (inter_dim + inter_dim + out_dim) * 4)  # scales
    ratio = fp16_size / tern_size if tern_size > 0 else 0
    print(f"\n  Storage: FP16={fp16_size/1024:.1f} KB, "
          f"Ternary={tern_size/1024:.1f} KB, "
          f"Compression: {ratio:.1f}x")
    results["compression"] = {"fp16_bytes": fp16_size, "ternary_bytes": tern_size,
                              "ratio": ratio}
    return results


# ─────────────────────────────────────────────────────────────────
# Full Pipeline Benchmark
# ─────────────────────────────────────────────────────────────────

def benchmark_full_pipeline(
    bvh_ext, tern_ext, warmup: int, iters: int
) -> Dict[str, float]:
    """Benchmark full pipeline: embedding -> BVH routing -> ternary expert."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 3: Full Pipeline (BVH + Ternary)")
    print("=" * 60)

    if bvh_ext is None or tern_ext is None:
        print("  [SKIP] Requires both bvh_router_ext and ternary_expert_ext")
        return {}

    # Load ternary weights
    weights = load_ternary_weights(layer_idx=0)
    if not weights:
        print("  [SKIP] No ternary weights")
        return {}

    device = torch.device("cuda")

    # Upload BVH tree
    tree = create_synthetic_tree()
    bvh_ext.upload_tree(*tree)

    # Prepare ternary (transpose: [out, in] -> [in, out] for kernel)
    gate_packed = tern_ext.pack_ternary(
        torch.from_numpy(weights["gate_ternary"].T.copy().astype(np.int8))
    ).to(device)
    up_packed = tern_ext.pack_ternary(
        torch.from_numpy(weights["up_ternary"].T.copy().astype(np.int8))
    ).to(device)
    down_packed = tern_ext.pack_ternary(
        torch.from_numpy(weights["down_ternary"].T.copy().astype(np.int8))
    ).to(device)
    gate_scale = torch.from_numpy(weights["gate_scale"]).float().to(device)
    up_scale = torch.from_numpy(weights["up_scale"]).float().to(device)
    down_scale = torch.from_numpy(weights["down_scale"]).float().to(device)

    in_dim = weights["gate_ternary"].shape[1]  # 896 (in_features)

    # Simulate input embedding projection
    embed_proj = nn.Linear(256, 3).to(device)
    spectral_proj = nn.Linear(256, SPEC_DIM).to(device)

    results = {}
    for bs in [1, 16, 64, 256]:
        embedding = torch.randn(bs, 256, device=device, dtype=torch.float32)

        def full_pipeline():
            with torch.no_grad():
                # Step 1: Project to 3D + spectral
                origins = embed_proj(embedding)
                directions = F.normalize(origins, dim=-1)
                spectral = spectral_proj(embedding)

                # Step 2: BVH routing (CUDA kernel)
                expert_ids, scores, confidence, path = bvh_ext.route(
                    origins, directions, spectral
                )

                # Step 3: Prepare expert input
                # (in real pipeline, this selects from token hidden states)
                expert_input = torch.randn(
                    bs, in_dim, device=device, dtype=torch.float32
                )

                # Step 4: Ternary expert forward (POPCOUNT)
                output = tern_ext.ternary_gated_mlp(
                    expert_input, gate_packed, up_packed, down_packed,
                    gate_scale, up_scale, down_scale,
                )
            return output

        # Warmup
        for _ in range(warmup):
            full_pipeline()

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            full_pipeline()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        avg_us = (elapsed / iters) * 1e6
        throughput = bs * iters / elapsed
        results[bs] = {"avg_us": avg_us, "throughput": throughput}
        print(f"  batch={bs:>4d}  |  {avg_us:>8.1f} µs  |  "
              f"{throughput:>10,.0f} tok/s  (BVH + Ternary)")

    return results


# ─────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────

def print_summary(
    bvh_results: dict,
    pytorch_results: dict,
    ternary_results: dict,
    pipeline_results: dict,
) -> None:
    """Print formatted summary table for paper."""
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE — SpectralAI CUDA Pipeline Benchmarks")
    print("=" * 70)

    print(f"\n  {'Component':<25} {'Method':<15} {'Latency':<12} "
          f"{'Speedup':<10} {'Notes'}")
    print(f"  {'-'*25} {'-'*15} {'-'*12} {'-'*10} {'-'*20}")

    # BVH Router rows
    if 256 in bvh_results:
        cuda_us = bvh_results[256]["avg_us"]
        pytorch_us = pytorch_results.get(256, {}).get("avg_us", 0)
        speedup = pytorch_us / cuda_us if cuda_us > 0 and pytorch_us > 0 else 0
        print(f"  {'BVH Router (B=256)':<25} {'CUDA Kernel':<15} "
              f"{cuda_us:>8.2f} µs  {speedup:>7.0f}x    RT-inspired")
        if pytorch_us > 0:
            print(f"  {'BVH Router (B=256)':<25} {'PyTorch':<15} "
                  f"{pytorch_us:>8.2f} µs  {'1x':>7}    Baseline")

    if 1 in bvh_results:
        cuda_us = bvh_results[1]["avg_us"]
        print(f"  {'BVH Router (B=1)':<25} {'CUDA Kernel':<15} "
              f"{cuda_us:>8.2f} µs  {'—':>7}    Single token")

    # Ternary rows
    if 64 in ternary_results:
        t = ternary_results[64]
        print(f"  {'Expert MLP (B=64)':<25} {'Ternary':<15} "
              f"{t['ternary_us']:>8.1f} µs  {t['speedup']:>7.1f}x    POPCOUNT")
        print(f"  {'Expert MLP (B=64)':<25} {'FP16':<15} "
              f"{t['fp16_us']:>8.1f} µs  {'1x':>7}    Baseline")

    if "compression" in ternary_results:
        c = ternary_results["compression"]
        print(f"  {'Storage (1 expert)':<25} {'Ternary':<15} "
              f"{c['ternary_bytes']/1024:>6.1f} KB   {c['ratio']:>7.1f}x    2-bit encoding")

    # Pipeline rows
    if 256 in pipeline_results:
        p = pipeline_results[256]
        print(f"  {'Full Pipeline (B=256)':<25} {'CUDA+Tern':<15} "
              f"{p['avg_us']:>8.1f} µs  {'—':>7}    End-to-end")

    # Quality rows (from FASE E)
    print(f"\n  {'PPL (3 capas BVH)':<25} {'EnhancedBVH':<15} "
          f"{'7.42':>8}      {'+3.9%':>7}    vs 7.15 baseline")
    print(f"  {'PPL (16 capas BVH)':<25} {'EnhancedBVH':<15} "
          f"{'8.42':>8}      {'+17.8%':>7}    vs 7.15 baseline")
    print(f"  {'Generation (3 capas)':<25} {'PyTorch':<15} "
          f"{'15 tok/s':>8}    {'—':>7}    Coherent text")

    print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SpectralAI CUDA Pipeline Benchmark")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    args = parser.parse_args()

    print("=" * 60)
    print("  SpectralAI Zero-Matrix — CUDA Pipeline Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [ERROR] CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")

    # Load extensions
    bvh_ext = load_bvh_ext()
    tern_ext = load_ternary_ext()

    bvh_results = {}
    pytorch_results = {}
    ternary_results = {}
    pipeline_results = {}

    # Benchmark 1: BVH Router
    if bvh_ext is not None:
        bvh_results = benchmark_bvh_cuda(bvh_ext, args.warmup, args.iters)
        pytorch_results = benchmark_bvh_pytorch(args.warmup, args.iters)
    else:
        print("\n  [SKIP] BVH Router benchmark — extension not loaded")

    # Benchmark 2: Ternary Expert
    if tern_ext is not None:
        ternary_results = benchmark_ternary(tern_ext, args.warmup, args.iters)
    else:
        print("\n  [SKIP] Ternary Expert benchmark — extension not loaded")

    # Benchmark 3: Full Pipeline
    if bvh_ext is not None and tern_ext is not None:
        pipeline_results = benchmark_full_pipeline(
            bvh_ext, tern_ext, args.warmup, args.iters
        )

    # Summary
    print_summary(bvh_results, pytorch_results, ternary_results, pipeline_results)


if __name__ == "__main__":
    main()
