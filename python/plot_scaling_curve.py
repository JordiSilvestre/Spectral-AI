#!/usr/bin/env python3
"""
plot_scaling_curve.py — Generate BVH O(log N) vs Linear Gate O(N) scaling figure.

Produces a publication-ready PNG showing the theoretical and measured scaling
advantage of BVH routing over linear gating as expert count grows.

Can run in two modes:
  1. Analytical only (no GPU needed): generates theoretical curves
  2. With benchmark data: overlays measured latencies from benchmark_scaling.py

Usage:
    # Analytical-only (CPU, no deps beyond matplotlib/numpy)
    python plot_scaling_curve.py --output figures/scaling_curve.png

    # With measured data from a JSON results file
    python plot_scaling_curve.py --data scaling_results.json --output figures/scaling_curve.png

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG generation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

EXPERT_COUNTS_ANALYTICAL = [
    32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
]

# RT Core benchmark data from RTX 5070 Ti (measured 2026-04-01)
RT_CORE_MEASURED = {
    64: {"bvh_us": 19.1, "linear_gate_us": 45.2},  # Triangle async mode
}

# CUDA kernel projection: base latency at N=64
CUDA_BASE_US = 10.0  # Measured fused 3-level kernel


# ─────────────────────────────────────────────────────────────────
# Analytical curves
# ─────────────────────────────────────────────────────────────────

def compute_analytical_curves(
    expert_counts: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute O(N) and O(log N) normalized curves.

    Returns (ns, linear_cost, log_cost) normalized to N=64 baseline.
    """
    ns = np.array(expert_counts, dtype=np.float64)
    base_n = 64.0
    linear_cost = ns / base_n
    log_cost = np.log2(ns) / np.log2(base_n)
    return ns, linear_cost, log_cost


def compute_cuda_projection(
    expert_counts: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project CUDA BVH kernel latency for each N.

    Based on measured 10µs at N=64 with O(log N) scaling.
    """
    ns = np.array(expert_counts, dtype=np.float64)
    log_factor = np.log2(ns) / np.log2(64)
    latencies = CUDA_BASE_US * log_factor
    return ns, latencies


def compute_linear_gate_projection(
    expert_counts: List[int],
    base_latency_us: float = 45.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project linear gate latency: O(N) scaling from measured base."""
    ns = np.array(expert_counts, dtype=np.float64)
    latencies = base_latency_us * (ns / 64.0)
    return ns, latencies


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def create_scaling_figure(
    output_path: str,
    measured_data: Optional[List[dict]] = None,
    dpi: int = 200,
) -> str:
    """Generate the dual-panel scaling curve figure.

    Panel 1: Latency vs N (log-log scale)
    Panel 2: Speedup ratio vs N (log-x scale)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "BVH Routing O(log N) vs Linear Gate O(N) — Expert Scaling",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ns_analytical = EXPERT_COUNTS_ANALYTICAL

    # ── Panel 1: Latency curves ──
    # Theoretical O(N) linear gate
    ns_lin, lat_lin = compute_linear_gate_projection(ns_analytical)
    ax1.loglog(ns_lin, lat_lin, "r--", linewidth=2, alpha=0.7,
               label="Linear Gate O(N) — projected")

    # Theoretical O(log N) BVH CUDA kernel
    ns_bvh, lat_bvh = compute_cuda_projection(ns_analytical)
    ax1.loglog(ns_bvh, lat_bvh, "b-", linewidth=2, alpha=0.7,
               label="BVH Router O(log N) — projected")

    # Measured data points (if available)
    if measured_data:
        gate_ns = [r["n_experts"] for r in measured_data]
        gate_us = [r["gate_us"] for r in measured_data]
        bvh_us = [r["bvh_us"] for r in measured_data]
        ax1.scatter(gate_ns, gate_us, color="red", s=80, zorder=5,
                    edgecolors="darkred", linewidths=1.5,
                    label="Linear Gate — measured")
        ax1.scatter(gate_ns, bvh_us, color="blue", s=80, zorder=5,
                    edgecolors="darkblue", linewidths=1.5,
                    label="BVH Router — measured")

    # RT Core measured point
    for n, data in RT_CORE_MEASURED.items():
        ax1.scatter([n], [data["bvh_us"]], color="cyan", s=120, zorder=6,
                    edgecolors="darkblue", linewidths=2, marker="*",
                    label=f"RT Core (RTX 5070 Ti) — {data['bvh_us']}µs")
        ax1.scatter([n], [data["linear_gate_us"]], color="salmon", s=80,
                    zorder=5, edgecolors="darkred", linewidths=1.5,
                    marker="D", label=f"Linear Gate baseline — {data['linear_gate_us']}µs")

    # Crossover annotation
    ax1.axvline(x=256, color="gray", linestyle=":", alpha=0.5)
    ax1.annotate("Crossover ~256", xy=(256, 80), fontsize=9,
                 color="gray", ha="center")

    ax1.set_xlabel("Number of Experts (N)", fontsize=12)
    ax1.set_ylabel("Latency (µs)", fontsize=12)
    ax1.set_title("Routing Latency", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(24, 100000)

    # ── Panel 2: Speedup ratio ──
    ns_a, linear_cost, log_cost = compute_analytical_curves(ns_analytical)
    speedup_theoretical = linear_cost / log_cost

    ax2.semilogx(ns_a, speedup_theoretical, "g-", linewidth=2.5,
                 label="Theoretical O(N)/O(log N)")

    # CUDA kernel projected speedup
    cuda_speedup = lat_lin / lat_bvh
    ax2.semilogx(ns_bvh, cuda_speedup, "b--", linewidth=2, alpha=0.7,
                 label="CUDA kernel projected")

    # Measured speedup points
    if measured_data:
        meas_ns = [r["n_experts"] for r in measured_data]
        meas_speedup = [r["speedup"] for r in measured_data]
        ax2.scatter(meas_ns, meas_speedup, color="blue", s=80, zorder=5,
                    edgecolors="darkblue", linewidths=1.5,
                    label="Measured (PyTorch BVH)")

    # RT Core speedup
    for n, data in RT_CORE_MEASURED.items():
        rt_speedup = data["linear_gate_us"] / data["bvh_us"]
        ax2.scatter([n], [rt_speedup], color="cyan", s=120, zorder=6,
                    edgecolors="darkblue", linewidths=2, marker="*",
                    label=f"RT Core — {rt_speedup:.1f}x")

    # Key annotations
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.4,
                label="Break-even (1x)")
    ax2.annotate(
        f"N=65K: {speedup_theoretical[-1]:.0f}x theoretical advantage",
        xy=(65536, speedup_theoretical[-1]),
        xytext=(8000, speedup_theoretical[-1] * 0.7),
        fontsize=9, color="green",
        arrowprops={"arrowstyle": "->", "color": "green", "alpha": 0.6},
    )

    ax2.set_xlabel("Number of Experts (N)", fontsize=12)
    ax2.set_ylabel("Speedup (Gate/BVH)", fontsize=12)
    ax2.set_title("BVH Routing Advantage", fontsize=12)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(24, 100000)

    # Footer
    fig.text(
        0.5, 0.01,
        "SpectralAI Zero-Matrix — RTX 5070 Ti (sm_120) · "
        "CUDA 13.2 · OptiX 9.1 · April 2026",
        fontsize=8, ha="center", color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    return output_path


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BVH vs Linear Gate scaling curve PNG"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="JSON file with measured benchmark data (from benchmark_scaling.py)",
    )
    parser.add_argument(
        "--output", type=str, default="figures/scaling_curve.png",
        help="Output PNG path (default: figures/scaling_curve.png)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    measured_data = None
    if args.data and os.path.exists(args.data):
        with open(args.data, "r") as f:
            measured_data = json.load(f)
        print(f"Loaded {len(measured_data)} measured data points from {args.data}")

    output = create_scaling_figure(
        output_path=args.output,
        measured_data=measured_data,
        dpi=args.dpi,
    )
    print(f"Scaling curve saved to: {output}")
    file_size = os.path.getsize(output)
    print(f"File size: {file_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
