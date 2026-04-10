#!/usr/bin/env python3
"""
benchmark_rt_crossover.py — Find the crossover point where RT Cores beat CUDA kernel.

Runs the compiled rt_router_benchmark executable with varying num_experts (N)
and batch sizes to identify when O(log N) RT Core traversal outperforms
O(N) CUDA kernel linear scan.

Usage:
    python scripts/benchmark_rt_crossover.py [--build-dir build/Release]

Requires: rt_router_benchmark.exe compiled (cmake --build build --config Release)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_BUILD_DIR = PROJECT_DIR / "build" / "Release"


def run_benchmark(
    exe_path: Path,
    ptx_dir: Path,
    batch_size: int,
    num_iters: int,
    num_experts: int,
) -> dict | None:
    """Run rt_router_benchmark and parse output."""
    cmd = [
        str(exe_path),
        str(ptx_dir),
        str(batch_size),
        str(num_iters),
        str(num_experts),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: N={num_experts}, batch={batch_size}")
        return None

    output = result.stdout + result.stderr

    # Parse latency and throughput
    latency_match = re.search(r"Latency:\s+([\d.]+)\s+us/batch", output)
    throughput_match = re.search(r"Throughput:\s+([\d.]+)\s+M queries/s", output)
    gas_match = re.search(r"GAS size:\s+(\d+)\s+bytes", output)
    accuracy_match = re.search(r"Accuracy:\s+([\d.]+)%", output)

    if not latency_match:
        print(f"  FAILED to parse output for N={num_experts}, batch={batch_size}")
        if result.returncode != 0:
            print(f"  Exit code: {result.returncode}")
            # Show last few lines of output for debugging
            lines = output.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")
        return None

    return {
        "num_experts": num_experts,
        "batch_size": batch_size,
        "latency_us": float(latency_match.group(1)),
        "throughput_mqs": float(throughput_match.group(1)) if throughput_match else 0,
        "gas_bytes": int(gas_match.group(1)) if gas_match else 0,
        "accuracy": float(accuracy_match.group(1)) if accuracy_match else 0,
    }


def cuda_kernel_estimate(num_experts: int, batch_size: int) -> float:
    """Estimate CUDA kernel latency in microseconds.

    The CUDA kernel does O(N) comparisons per query.
    Measured: ~8.84 us at N=64, batch=256.
    Scaling: ~linear in N (each query scans all experts).
    """
    # Base measurement
    base_us = 8.84
    base_n = 64
    base_batch = 256

    # Scale with N (linear scan) and batch (parallel queries)
    # At small batch, kernel launch overhead dominates
    # At large batch, compute dominates
    overhead_us = 3.0  # kernel launch overhead
    compute_per_query = (base_us - overhead_us) / base_batch * (num_experts / base_n)
    return overhead_us + compute_per_query * batch_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=str, default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--batch-sizes", type=str, default="256,4096,16384",
                        help="Comma-separated batch sizes")
    parser.add_argument("--expert-counts", type=str, default="64,128,256,512,1024,2048,4096",
                        help="Comma-separated expert counts")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    exe = build_dir / "rt_router_benchmark.exe"
    ptx_dir = build_dir

    if not exe.exists():
        print(f"ERROR: {exe} not found. Build first with:")
        print(f"  cmake --build build --config Release")
        return 1

    # Check PTX files exist
    ptx_raygen = ptx_dir / "ptx" / "optix_router_raygen.ptx"
    if not ptx_raygen.exists():
        print(f"ERROR: {ptx_raygen} not found")
        return 1

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    expert_counts = [int(x) for x in args.expert_counts.split(",")]

    print("=" * 70)
    print("  RT Core vs CUDA Kernel — Crossover Analysis")
    print("=" * 70)
    print()

    results = []

    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")
        print(f"{'N experts':>10} | {'RT Core (us)':>12} | {'CUDA est (us)':>13} | "
              f"{'RT/CUDA ratio':>13} | {'Winner':>8} | {'GAS (KB)':>9}")
        print("-" * 80)

        for n in expert_counts:
            result = run_benchmark(exe, ptx_dir, batch_size, args.iters, n)
            if result is None:
                continue

            cuda_us = cuda_kernel_estimate(n, batch_size)
            ratio = result["latency_us"] / cuda_us if cuda_us > 0 else float("inf")
            winner = "RT Core" if ratio < 1.0 else "CUDA"
            gas_kb = result["gas_bytes"] / 1024

            print(f"{n:>10} | {result['latency_us']:>12.1f} | {cuda_us:>13.1f} | "
                  f"{ratio:>13.2f}x | {winner:>8} | {gas_kb:>8.1f}")

            results.append({
                **result,
                "cuda_estimate_us": cuda_us,
                "ratio": ratio,
                "winner": winner,
            })

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    rt_wins = [r for r in results if r["winner"] == "RT Core"]
    if rt_wins:
        min_n = min(r["num_experts"] for r in rt_wins)
        print(f"\nRT Core crossover point: N >= {min_n} experts")
        print(f"RT Core wins in {len(rt_wins)}/{len(results)} configurations")
    else:
        print("\nRT Core does NOT beat CUDA kernel in any tested configuration.")
        print("The crossover likely requires N > max tested value.")
        if results:
            best = min(results, key=lambda r: r["ratio"])
            print(f"Closest: N={best['num_experts']}, ratio={best['ratio']:.2f}x "
                  f"(batch={best['batch_size']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
