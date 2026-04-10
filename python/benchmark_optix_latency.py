#!/usr/bin/env python3
"""
benchmark_optix_latency.py -- OptiX RT Core Router latency comparison

Compares 4 routing modes on RT Cores:
  1. AABB + sync     (original, ~94 us)
  2. AABB + async    (stream-based, ~30-50 us expected)
  3. Triangle + sync (native RT Core, ~40-60 us expected)
  4. Triangle + async(best: ~5-15 us expected)

Usage:
    # From project root, inside build_win.bat shell:
    python python/benchmark_optix_latency.py

    # Or with custom params:
    python python/benchmark_optix_latency.py --experts 64 --batch 256 --iters 200

Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optix_bench")

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR = PROJECT_ROOT / "cuda"
PTX_DIR = PROJECT_ROOT / "build" / "ptx"

# Ensure DLL directories are available (Windows)
if sys.platform == "win32":
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
    cuda_bin = Path(os.environ.get("CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")) / "bin"
    if cuda_bin.exists():
        os.add_dll_directory(str(cuda_bin))


def load_optix_ext():
    """Load the OptiX training extension (JIT or prebuilt)."""
    # Try direct import first (already built)
    try:
        import optix_training_ext
        return optix_training_ext
    except ImportError:
        pass

    # Build from source
    build_script = CUDA_DIR / "v5" / "build_optix_ext.py"
    if not build_script.exists():
        raise FileNotFoundError(f"Build script not found: {build_script}")

    sys.path.insert(0, str(CUDA_DIR / "v5"))
    import build_optix_ext  # noqa: E402
    if not build_optix_ext.build_optix_training_ext():
        raise RuntimeError("Failed to build OptiX extension")

    import optix_training_ext
    return optix_training_ext


def find_ptx_files() -> tuple[str, str]:
    """Locate raygen and hitgroup PTX files."""
    ptx_raygen = PTX_DIR / "optix_router_raygen.ptx"
    ptx_hitgroup = PTX_DIR / "optix_router_hitgroup.ptx"

    if not ptx_raygen.exists() or not ptx_hitgroup.exists():
        raise FileNotFoundError(
            f"PTX files not found in {PTX_DIR}. "
            "Run build_ptx_win.bat first."
        )
    return str(ptx_raygen), str(ptx_hitgroup)


def generate_expert_positions(
    num_experts: int = 64,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate expert centers on a sphere (deterministic)."""
    import math

    centers = torch.zeros(num_experts, 3)
    radii = torch.full((num_experts,), 0.5)

    for i in range(num_experts):
        theta = 2.0 * math.pi * i / num_experts
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / num_experts)
        centers[i, 0] = 10.0 * math.sin(phi) * math.cos(theta)
        centers[i, 1] = 10.0 * math.sin(phi) * math.sin(theta)
        centers[i, 2] = 10.0 * math.cos(phi)

    return centers, radii


def generate_queries(
    centers: torch.Tensor,
    batch_size: int = 256,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate query positions + directions aimed at specific experts."""
    import math

    num_experts = centers.shape[0]
    positions = torch.zeros(batch_size, 3, device=device)
    directions = torch.zeros(batch_size, 3, device=device)

    for i in range(batch_size):
        t = i / batch_size
        positions[i] = torch.tensor([
            math.sin(t * 2 * math.pi) * 2.0,
            math.cos(t * 2 * math.pi) * 2.0,
            math.sin(t * math.pi) * 2.0,
        ], device=device)

        target = i % num_experts
        direction = centers[target].to(device) - positions[i]
        length = direction.norm()
        if length > 1e-6:
            direction = direction / length
        directions[i] = direction

    return positions, directions


def benchmark_python_route(
    ext,
    positions: torch.Tensor,
    directions: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
    label: str = "route",
    use_async: bool = False,
) -> dict:
    """Benchmark route() or route_async() from Python side."""
    route_fn = ext.route_async if use_async else ext.route

    # Warmup
    for _ in range(num_warmup):
        route_fn(positions, directions)
    if use_async:
        ext.sync()
    torch.cuda.synchronize()

    # Timed loop
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(num_iters):
        expert_ids, distances = route_fn(positions, directions)
    if use_async:
        ext.sync()
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = start_evt.elapsed_time(end_evt)
    us_per_iter = (elapsed_ms * 1000.0) / num_iters

    # Verify correctness
    if use_async:
        ext.sync()
    expert_ids_cpu = expert_ids.cpu()
    num_experts = ext.num_experts()
    batch_size = positions.shape[0]
    correct = sum(
        1 for i in range(batch_size)
        if expert_ids_cpu[i].item() == (i % num_experts)
    )
    accuracy = 100.0 * correct / batch_size

    return {
        "label": label,
        "us_per_batch": us_per_iter,
        "accuracy": accuracy,
        "throughput_mqps": batch_size / (us_per_iter * 1e-6) / 1e6,
    }


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full latency benchmark."""
    log.info("=== OptiX RT Core Router — Latency Benchmark ===")
    log.info(f"Experts: {args.experts}, Batch: {args.batch}, Iters: {args.iters}")

    # Load extension
    log.info("Loading OptiX extension...")
    ext = load_optix_ext()

    # Find PTX
    ptx_raygen, ptx_hitgroup = find_ptx_files()
    log.info(f"PTX raygen:   {ptx_raygen}")
    log.info(f"PTX hitgroup: {ptx_hitgroup}")

    # Ensure clean state
    try:
        ext.shutdown()
    except Exception:
        pass

    # Touch GPU to ensure PyTorch CUDA context is established BEFORE OptiX
    # OptiX needs to share the CUDA context created by PyTorch
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Initialize OptiX (will use PyTorch's existing CUDA context)
    ext.initialize(ptx_raygen, ptx_hitgroup)
    log.info("OptiX initialized")

    # Generate data
    centers, radii = generate_expert_positions(args.experts)
    positions, directions = generate_queries(centers, args.batch)

    results = []

    # ── Test 1: AABB + sync ──────────────────────────────────
    log.info("\n--- Test 1: AABB + sync (original) ---")
    ext.build_gas(centers, radii, use_triangles=False)
    r = benchmark_python_route(
        ext, positions, directions,
        num_warmup=args.warmup, num_iters=args.iters,
        label="AABB sync", use_async=False,
    )
    results.append(r)
    log.info(f"  {r['us_per_batch']:.1f} us/batch, {r['accuracy']:.0f}% accuracy, "
             f"{r['throughput_mqps']:.2f} M queries/s")

    # ── Test 2: AABB + async ─────────────────────────────────
    log.info("\n--- Test 2: AABB + async ---")
    r = benchmark_python_route(
        ext, positions, directions,
        num_warmup=args.warmup, num_iters=args.iters,
        label="AABB async", use_async=True,
    )
    results.append(r)
    log.info(f"  {r['us_per_batch']:.1f} us/batch, {r['accuracy']:.0f}% accuracy, "
             f"{r['throughput_mqps']:.2f} M queries/s")

    # ── Test 3: Triangle + sync ──────────────────────────────
    log.info("\n--- Test 3: Triangle + sync (native RT Core) ---")
    ext.build_gas(centers, radii, use_triangles=True)
    r = benchmark_python_route(
        ext, positions, directions,
        num_warmup=args.warmup, num_iters=args.iters,
        label="Triangle sync", use_async=False,
    )
    results.append(r)
    log.info(f"  {r['us_per_batch']:.1f} us/batch, {r['accuracy']:.0f}% accuracy, "
             f"{r['throughput_mqps']:.2f} M queries/s")

    # ── Test 4: Triangle + async ─────────────────────────────
    log.info("\n--- Test 4: Triangle + async (FASTEST) ---")
    r = benchmark_python_route(
        ext, positions, directions,
        num_warmup=args.warmup, num_iters=args.iters,
        label="Triangle async", use_async=True,
    )
    results.append(r)
    log.info(f"  {r['us_per_batch']:.1f} us/batch, {r['accuracy']:.0f}% accuracy, "
             f"{r['throughput_mqps']:.2f} M queries/s")

    # ── Also run C++ benchmark for pure-metal numbers ────────
    if hasattr(ext, 'benchmark'):
        log.info("\n--- C++ native benchmark (no Python overhead) ---")
        try:
            ext.benchmark(
                num_experts=args.experts,
                batch_size=args.batch,
                num_warmup=args.warmup,
                num_iters=args.iters,
                ptx_raygen=ptx_raygen,
                ptx_hitgroup=ptx_hitgroup,
            )
        except Exception as e:
            log.warning(f"C++ benchmark failed: {e}")

    # ── Summary ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY (Python-side measurements)")
    log.info("=" * 60)
    log.info(f"{'Mode':<20} {'Latency':>10} {'Accuracy':>10} {'Throughput':>14}")
    log.info("-" * 60)
    for r in results:
        log.info(
            f"{r['label']:<20} {r['us_per_batch']:>8.1f} us {r['accuracy']:>8.0f}% "
            f"{r['throughput_mqps']:>10.2f} Mq/s"
        )

    best = min(results, key=lambda x: x["us_per_batch"])
    worst = max(results, key=lambda x: x["us_per_batch"])
    log.info(f"\nBest:  {best['label']} = {best['us_per_batch']:.1f} us")
    log.info(f"Worst: {worst['label']} = {worst['us_per_batch']:.1f} us")
    log.info(f"Speedup: {worst['us_per_batch'] / best['us_per_batch']:.1f}x")

    # Cleanup
    ext.shutdown()
    log.info("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OptiX RT Core latency benchmark")
    parser.add_argument("--experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
