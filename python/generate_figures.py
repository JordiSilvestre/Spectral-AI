#!/usr/bin/env python3
"""
generate_figures.py — Generate ALL publication figures for 3 Zenodo papers.

Paper 1 (Expert Specialization):
  figures/selectivity_u_shape.png   — U-shaped selectivity across 3 models
  figures/topic_specialization.png  — Topic specialization weakness
  figures/cluster_stability.png     — Cluster stability heatmap

Paper 2 (SpectralAI / BVH Routing):
  figures/per_layer_accuracy.png    — Per-layer routing accuracy (16 layers)
  figures/ppl_comparison.png        — PPL across configurations
  figures/rt_core_speedup.png       — RT Core speedup vs PyTorch
  figures/prefilter_sweep.png       — Pre-filter candidate sweep

Paper 3 (Spectral Routing):
  figures/polysemy_ablation.png     — Polysemy resolution ablation
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Shared Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "OLMoE": "#2196F3",
    "Qwen": "#FF9800",
    "DeepSeek": "#4CAF50",
}

RESULT_DIRS = {
    "OLMoE": "results/olmoe",
    "Qwen": "results/qwen_moe",
    "DeepSeek": "results/deepseek_moe",
}

MODEL_LABELS = {
    "OLMoE": "OLMoE-1B-7B (16L, top-8)",
    "Qwen": "Qwen1.5-MoE-A2.7B (24L, top-4)",
    "DeepSeek": "DeepSeek-MoE-16B (12/27L, top-6)",
}

FIGDIR = Path("figures")


def _save(fig: plt.Figure, name: str) -> None:
    out = FIGDIR / name
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def load_deep(model_key: str) -> dict:
    path = Path(RESULT_DIRS[model_key]) / "expert_deep_analysis.json"
    with open(path) as f:
        return json.load(f)


def load_catalog(model_key: str) -> dict:
    path = Path(RESULT_DIRS[model_key]) / "expert_catalog.json"
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# PAPER 1: Expert Specialization
# ═══════════════════════════════════════════════════════════════════

def fig_selectivity_u_shape() -> None:
    """U-shaped selectivity curve across layers for 3 MoE models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_key in ["OLMoE", "Qwen", "DeepSeek"]:
        deep = load_deep(model_key)
        moe_layers = sorted(deep["moe_layer_indices"])
        selectivity = deep["per_layer_selectivity"]

        xs, ys = [], []
        for layer_idx in moe_layers:
            sels = selectivity.get(str(layer_idx), {})
            if sels:
                mean_val = float(np.mean(list(sels.values())))
                if mean_val > 0.01:
                    xs.append(layer_idx / max(moe_layers))
                    ys.append(mean_val)

        color = COLORS[model_key]
        label = MODEL_LABELS[model_key]
        ax.plot(xs, ys, "o-", color=color, label=label,
                markersize=5, linewidth=2, alpha=0.9)

        if len(ys) >= 6:
            ax.fill_between(xs, ys, min(ys) * 0.95,
                            alpha=0.08, color=color)

    ax.set_xlabel("Relative Layer Position (0 = first MoE layer, 1 = last)")
    ax.set_ylabel("Mean Expert Selectivity (std of activation rates)")
    ax.set_title("U-Shaped Expert Selectivity Across Layers — 3 MoE Models")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhspan(0, 0.5, alpha=0.04, color="gray", label=None)

    ax.annotate("Middle-layer\ntrough",
                xy=(0.5, 0.38), xytext=(0.62, 0.25),
                fontsize=9, color="gray", style="italic",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    _save(fig, "selectivity_u_shape.png")


def fig_topic_specialization() -> None:
    """Top-15 most specialized experts per model — shows weakness of topic specialization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for i, model_key in enumerate(["OLMoE", "Qwen", "DeepSeek"]):
        ax = axes[i]
        cat = load_catalog(model_key)
        catalog = cat["catalog"]
        n_categories = len(cat["categories"])
        uniform = 100.0 / n_categories

        experts = sorted(catalog.items(),
                         key=lambda x: -x[1]["primary_pct"])[:15]

        names = [f"E{eid}" for eid, _ in experts]
        pcts = [info["primary_pct"] for _, info in experts]
        topics = [info["primary"][:8] for _, info in experts]

        ax.barh(range(len(names)), pcts,
                color=COLORS[model_key], alpha=0.8, edgecolor="white")
        ax.axvline(x=uniform, color="red", linestyle="--", linewidth=1.5,
                   label=f"Uniform ({uniform:.1f}%)")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n} ({t})" for n, t in zip(names, topics)],
                           fontsize=8)
        ax.set_xlabel("Activation %")
        ax.set_title(model_key, fontweight="bold", color=COLORS[model_key])
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 10)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    fig.suptitle("Topic Specialization Is Weak: Best Experts Barely Exceed Uniform Baseline",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "topic_specialization.png")


def fig_cluster_stability() -> None:
    """Co-activation cluster stability between adjacent layers."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, model_key in enumerate(["OLMoE", "Qwen", "DeepSeek"]):
        ax = axes[i]
        deep = load_deep(model_key)
        stability = deep.get("cluster_stability", {})

        pairs = []
        for pair_str, pct in stability.items():
            parts = pair_str.replace("L", "").split("-")
            if len(parts) == 2:
                la, lb = int(parts[0]), int(parts[1])
                if abs(lb - la) <= 2 and pct > 0.5:
                    pairs.append((la, lb, pct))

        if not pairs:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(model_key)
            continue

        pairs.sort()
        labels = [f"L{a}-L{b}" for a, b, _ in pairs]
        values = [p for _, _, p in pairs]

        colors = [plt.cm.RdYlGn(v / 100.0) for v in values]
        ax.bar(range(len(labels)), values, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, fontsize=6, ha="right")
        ax.set_ylabel("Stability %" if i == 0 else "")
        ax.set_title(model_key, fontweight="bold", color=COLORS[model_key])
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Co-Activation Cluster Stability Between Adjacent Layers",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "cluster_stability.png")


# ═══════════════════════════════════════════════════════════════════
# PAPER 2: SpectralAI (BVH Routing)
# ═══════════════════════════════════════════════════════════════════

def fig_per_layer_accuracy() -> None:
    """Per-layer BVH routing accuracy for OLMoE-1B-7B (16 layers)."""
    layers = list(range(16))
    accs = [95.40, 93.36, 96.11, 96.17, 95.15, 96.14, 96.40, 96.62,
            89.27, 96.81, 97.20, 97.19, 97.42, 96.97, 97.47, 97.58]
    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#e53935" if a < 93 else "#2196F3" if a < 96 else "#4CAF50" for a in accs]
    bars = ax.bar(layers, accs, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)

    ax.axhline(y=mean_acc, color="#FF9800", linestyle="--", linewidth=2,
               label=f"Mean: {mean_acc:.2f}%")
    ax.axhline(y=93, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label="93% threshold")

    # Annotate min and max
    min_idx = int(np.argmin(accs))
    max_idx = int(np.argmax(accs))
    ax.annotate(f"{accs[min_idx]:.1f}%", xy=(min_idx, accs[min_idx]),
                xytext=(min_idx + 1.2, accs[min_idx] - 2),
                fontsize=9, color="#e53935", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e53935", lw=1.2))
    ax.annotate(f"{accs[max_idx]:.1f}%", xy=(max_idx, accs[max_idx]),
                xytext=(max_idx - 2.5, accs[max_idx] + 0.5),
                fontsize=9, color="#4CAF50", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.2))

    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Top-8 Routing Accuracy (%)")
    ax.set_title("BVH Router Per-Layer Accuracy — OLMoE-1B-7B (16 Layers)")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.set_ylim(86, 100)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    _save(fig, "per_layer_accuracy.png")


def fig_ppl_comparison() -> None:
    """PPL across different routing configurations."""
    configs = [
        ("Baseline\n(linear gate)", 6.69, "#757575"),
        ("Pre-filter\n48 cand.", 6.79, "#4CAF50"),
        ("Pre-filter\n32 cand.", 7.36, "#8BC34A"),
        ("Hybrid\n3-layer", 7.17, "#2196F3"),
        ("Hybrid\n16-layer", 7.30, "#1565C0"),
        ("Pure\n3-layer", 7.33, "#FF9800"),
        ("Pure\n16-layer", 9.11, "#e53935"),
    ]

    labels = [c[0] for c in configs]
    ppls = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), ppls, color=colors, edgecolor="white",
                  linewidth=0.8, alpha=0.9)

    # Annotate values
    for i, (bar, ppl) in enumerate(zip(bars, ppls)):
        delta = ((ppl / 6.69) - 1) * 100 if i > 0 else 0
        label = f"{ppl:.2f}"
        if i > 0 and i <= 2:
            label += f"\n(+{delta:.1f}%)"
        elif i > 2:
            # Different baseline for 50K
            delta50 = ((ppl / 7.15) - 1) * 100
            label += f"\n(+{delta50:.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=6.69, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title("Perplexity Across Routing Configurations — OLMoE-1B-7B")
    ax.set_ylim(0, 10.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    _save(fig, "ppl_comparison.png")


def fig_rt_core_speedup() -> None:
    """RT Core speedup vs PyTorch routing at different batch sizes."""
    batch_sizes = [1, 256, 1024]
    pytorch_us = [1260, 1412, 2371]
    cuda_us = [11, 10, 10.9]
    speedups = [p / c for p, c in zip(pytorch_us, cuda_us)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: latency comparison (log scale)
    x = np.arange(len(batch_sizes))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, pytorch_us, width, label="PyTorch Gate",
                    color="#e53935", alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + width / 2, cuda_us, width, label="CUDA RT Core",
                    color="#4CAF50", alpha=0.85, edgecolor="white")

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Batch {b}" for b in batch_sizes])
    ax1.set_ylabel("Latency (μs) — log scale")
    ax1.set_title("Routing Latency: PyTorch vs RT Core")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars1, pytorch_us):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                 f"{val:,}μs", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, cuda_us):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                 f"{val}μs", ha="center", va="bottom", fontsize=8)

    # Right: speedup line
    ax2.plot(batch_sizes, speedups, "s-", color="#FF9800", linewidth=2.5,
             markersize=10, markerfacecolor="#FF9800", markeredgecolor="white",
             markeredgewidth=2)

    for bs, sp in zip(batch_sizes, speedups):
        ax2.annotate(f"{sp:.0f}×", xy=(bs, sp), xytext=(0, 12),
                     textcoords="offset points", ha="center",
                     fontsize=11, fontweight="bold", color="#E65100")

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("RT Core Speedup Over PyTorch")
    ax2.set_ylim(0, 260)
    ax2.set_xscale("log")
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels([str(b) for b in batch_sizes])
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.fill_between(batch_sizes, speedups, alpha=0.1, color="#FF9800")

    fig.tight_layout()
    _save(fig, "rt_core_speedup.png")


def fig_prefilter_sweep() -> None:
    """Pre-filter candidate sweep: PPL vs search reduction."""
    candidates = [16, 24, 32, 48, 64]
    ppls = [15.56, 8.49, 7.36, 6.79, 6.69]
    deltas = [132.5, 26.8, 10.0, 1.5, 0.0]
    reductions = [4.0, 2.7, 2.0, 1.3, 1.0]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    color1 = "#2196F3"
    color2 = "#FF9800"

    line1 = ax1.plot(candidates, ppls, "o-", color=color1, linewidth=2.5,
                     markersize=8, label="PPL", zorder=5)
    ax1.fill_between(candidates, ppls, min(ppls), alpha=0.1, color=color1)

    line2 = ax2.plot(candidates, reductions, "s--", color=color2, linewidth=2,
                     markersize=8, label="Search Reduction", zorder=4)

    # Annotate the sweet spot
    ax1.annotate("Sweet spot\n(+1.5% PPL, 1.3× reduction)",
                 xy=(48, 6.79), xytext=(30, 10),
                 fontsize=9, color="#1565C0", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    ax1.set_xlabel("Pre-filter Candidates (out of 64)")
    ax1.set_ylabel("Perplexity (PPL)", color=color1)
    ax2.set_ylabel("Search Reduction (×)", color=color2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Pre-filter Sweep: PPL vs Search Reduction — 16 Layers, 20K Tokens")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xticks(candidates)

    _save(fig, "prefilter_sweep.png")


# ═══════════════════════════════════════════════════════════════════
# PAPER 3: Spectral Routing
# ═══════════════════════════════════════════════════════════════════

def fig_polysemy_ablation() -> None:
    """Ablation study: each optical mechanism adds polysemy resolution."""
    methods = [
        "Linear gate\n(baseline)",
        "Snell\nrefraction",
        "+ Chromatic\naberration",
        "+ Total internal\nreflection",
        "+ Phase-coherent\ninterference",
    ]
    accs = [72.3, 80.1, 93.7, 96.8, 98.4]
    deltas = [0, 7.8, 21.4, 24.5, 26.1]

    colors_bar = ["#757575", "#64B5F6", "#42A5F5", "#2196F3", "#1565C0"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(range(len(methods)), accs, color=colors_bar,
                  edgecolor="white", linewidth=1, alpha=0.9)

    # Add delta annotations
    for i, (bar, acc, delta) in enumerate(zip(bars, accs, deltas)):
        # Accuracy label
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.8,
                f"{acc}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        # Delta label
        if delta > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, acc / 2,
                    f"+{delta} pp", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    # Draw incremental arrows between consecutive bars
    for i in range(1, len(accs)):
        mid_x = i - 0.5
        ax.annotate("", xy=(i, accs[i] - 1), xytext=(i - 1, accs[i - 1] + 1),
                     arrowprops=dict(arrowstyle="->", color="#FF9800",
                                     lw=1.5, connectionstyle="arc3,rad=0.2"))

    ax.axhline(y=72.3, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label="Baseline (72.3%)")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Polysemy Resolution Accuracy (%)")
    ax.set_title("Spectral Routing: Cumulative Ablation — 80 Words, 442 Context Pairs")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    _save(fig, "polysemy_ablation.png")


def fig_spectral_overhead() -> None:
    """Computational overhead of spectral routing components."""
    components = ["Single-band\nrefraction", "Chromatic\n(B=4 bands)"]
    overheads = [0.04, 0.12]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(range(len(components)), overheads,
                   color=["#64B5F6", "#1565C0"], alpha=0.9, edgecolor="white",
                   height=0.5)

    for bar, val in zip(bars, overheads):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Overhead (% of base BVH traversal)")
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=10)
    ax.set_title("Spectral Routing Computational Overhead")
    ax.set_xlim(0, 0.25)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    _save(fig, "spectral_overhead.png")


# ═══════════════════════════════════════════════════════════════════
# CROSS-PAPER: Scaling / Summary
# ═══════════════════════════════════════════════════════════════════

def fig_vram_comparison() -> None:
    """VRAM comparison: full model MLPs vs BVH router."""
    labels = ["Full MLPs\n(dense, 1.5B)", "BVH Router\n+ 1 expert"]
    sizes_mb = [2944, 4.03]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(range(len(labels)), sizes_mb,
                  color=["#e53935", "#4CAF50"], edgecolor="white",
                  linewidth=1, alpha=0.9, width=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("VRAM (MB) — log scale")
    ax.set_title("VRAM Usage: Dense MLPs vs BVH Router — 731× Reduction",
                 pad=15)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(1, 8000)

    # Labels INSIDE the bars to avoid title overlap
    ax.text(bars[0].get_x() + bars[0].get_width() / 2, 1500,
            "2,944 MB", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(bars[1].get_x() + bars[1].get_width() / 2, 8,
            "4.03 MB", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#2E7D32")

    # Draw the 731x arrow
    ax.annotate("731×\nreduction", xy=(1, 4.03), xytext=(0.5, 80),
                fontsize=12, fontweight="bold", color="#1565C0",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    _save(fig, "vram_comparison.png")


# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("\nGenerating ALL publication figures...")

    print("\n  [Paper 1: Expert Specialization]")
    fig_selectivity_u_shape()
    fig_topic_specialization()
    fig_cluster_stability()

    print("\n  [Paper 2: SpectralAI / BVH Routing]")
    fig_per_layer_accuracy()
    fig_ppl_comparison()
    fig_rt_core_speedup()
    fig_prefilter_sweep()
    fig_vram_comparison()

    print("\n  [Paper 3: Spectral Routing]")
    fig_polysemy_ablation()
    fig_spectral_overhead()

    print(f"\nDone. {len(list(FIGDIR.glob('*.png')))} figures in {FIGDIR}/\n")


if __name__ == "__main__":
    main()
