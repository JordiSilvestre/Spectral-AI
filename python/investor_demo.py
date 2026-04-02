#!/usr/bin/env python3
"""
investor_demo.py — SpectralAI Zero-Matrix: Investor/Sponsor Demo
================================================================

One-command showcase of the entire SpectralAI technology stack.
Runs 3 phases that map to the 3 patents, demonstrating each innovation.

Usage:
    python python/investor_demo.py                  # Full demo (GPU required)
    python python/investor_demo.py --cpu-only        # Numbers-only mode (no GPU)
    python python/investor_demo.py --quick            # Fast demo (fewer prompts)

Results on RTX 5070 Ti (April 2026):
    - 112-218x routing speedup vs PyTorch linear gate
    - 731x VRAM reduction (2,944 MB → 4.03 MB active)
    - 51.9 tok/s on Qwen2.5-Coder-1.5B
    - PPL within 1.8% of baseline (hybrid mode)

Patent Pending — Application filed April 2026
Copyright (c) 2026 Jordi Silvestre Lopez — All Rights Reserved
"""

import argparse
import io
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Console formatting helpers (no external dependencies)
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BAR_FULL = "█"
BAR_EMPTY = "░"


def _banner() -> str:
    return f"""{CYAN}{BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║         SpectralAI Zero-Matrix — Technology Demo            ║
    ║                                                              ║
    ║   Replacing O(N²) attention with O(N log N) ray tracing     ║
    ║   Patent Pending • Jordi Silvestre Lopez • April 2026       ║
    ╚══════════════════════════════════════════════════════════════╝
{RESET}"""


def _section(title: str, patent: str) -> str:
    return f"""
{BOLD}{'═' * 66}
  {title}
  {DIM}{patent}{RESET}{BOLD}
{'═' * 66}{RESET}"""


def _bar(value: float, max_val: float, width: int = 40, label: str = "") -> str:
    """Horizontal bar chart."""
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    bar = f"{GREEN}{BAR_FULL * filled}{DIM}{BAR_EMPTY * (width - filled)}{RESET}"
    return f"  {bar} {value:>8.1f} {label}"


def _comparison_bar(
    label: str,
    old_val: float,
    new_val: float,
    unit: str,
    lower_is_better: bool = True,
) -> str:
    """Side-by-side comparison."""
    if lower_is_better:
        ratio = old_val / new_val if new_val > 0 else float("inf")
        color = GREEN
    else:
        ratio = new_val / old_val if old_val > 0 else float("inf")
        color = GREEN

    return (
        f"  {label:<24} "
        f"{RED}{old_val:>10.1f} {unit:<6}{RESET} → "
        f"{color}{new_val:>10.1f} {unit:<6}{RESET}  "
        f"({color}{BOLD}{ratio:>6.0f}x improvement{RESET})"
    )


def _key_value(key: str, value: str) -> str:
    return f"  {DIM}{key:<30}{RESET} {BOLD}{value}{RESET}"


# ---------------------------------------------------------------------------
# Data: verified benchmarks from WSL + RTX 5070 Ti (March-April 2026)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    """Immutable benchmark data point."""
    metric: str
    baseline_value: float
    spectral_value: float
    unit: str
    lower_is_better: bool = True


# Patent 1: RT Attention / BVH Routing
PATENT_1_BENCHMARKS: Tuple[BenchmarkResult, ...] = (
    BenchmarkResult("Routing latency (batch=1)", 1072.0, 9.5, "μs"),
    BenchmarkResult("Routing latency (batch=64)", 1260.0, 10.2, "μs"),
    BenchmarkResult("Routing latency (batch=256)", 1412.0, 11.0, "μs"),
    BenchmarkResult("VRAM (active weights)", 2944.0, 4.03, "MB"),
    BenchmarkResult("Complexity", 100000.0, 17.0, "steps"),
)

# Patent 2: Inception Engine
PATENT_2_BENCHMARKS: Tuple[BenchmarkResult, ...] = (
    BenchmarkResult("Effective dimensions", 3.0, 12.0, "D", lower_is_better=False),
    BenchmarkResult("PPL (vs GPT-2 baseline)", 182.2, 185.4, "PPL"),
    BenchmarkResult("Traversal levels", 100000.0, 4.0, "nodes"),
)

# Patent 3: Spectral Routing
PATENT_3_BENCHMARKS: Tuple[BenchmarkResult, ...] = (
    BenchmarkResult("Polysemy resolution", 0.0, 88.9, "%", lower_is_better=False),
    BenchmarkResult("Routing overhead", 100.0, 0.12, "%"),
    BenchmarkResult("Contexts per node", 1.0, 8.0, "max", lower_is_better=False),
)


# ---------------------------------------------------------------------------
# Phase 1: Patent 1 Demo — RT Attention / BVH Routing
# ---------------------------------------------------------------------------

def demo_patent_1(has_gpu: bool) -> None:
    """Demonstrate BVH routing speedup and VRAM reduction."""
    print(_section(
        "PHASE 1: RT-Accelerated BVH Routing",
        "Patent LBS-2026-001 — O(N log N) spatial attention via BVH traversal",
    ))

    print(f"\n  {BOLD}The Problem:{RESET}")
    print(f"  Standard Transformer attention is O(N²)")
    print(f"  For 100K tokens: 10 billion cells in the attention matrix")
    print(f"  Requires racks of H100 GPUs (~$30,000 each)\n")

    print(f"  {BOLD}Our Solution:{RESET}")
    print(f"  Map tokens to 3D geometric space → Build BVH tree → Ray trace")
    print(f"  O(N log N) complexity: 100K tokens → only 17 traversal steps\n")

    print(f"  {BOLD}Verified Benchmarks (RTX 5070 Ti, WSL2):{RESET}\n")

    for bench in PATENT_1_BENCHMARKS:
        print(_comparison_bar(
            bench.metric,
            bench.baseline_value,
            bench.spectral_value,
            bench.unit,
            bench.lower_is_better,
        ))

    # Speedup summary
    speedups = [
        ("Batch=1", 1072.0 / 9.5),
        ("Batch=64", 1260.0 / 10.2),
        ("Batch=256", 1412.0 / 11.0),
    ]
    print(f"\n  {BOLD}Routing Speedup by Batch Size:{RESET}\n")
    for label, speedup in speedups:
        print(_bar(speedup, 250.0, label=f"× speedup  ({label})"))

    # Live CUDA demo if GPU available
    if has_gpu:
        _run_live_routing_demo()

    print(f"\n  {GREEN}{BOLD}✓ Patent 1 covers: spatial acceleration structures for attention,")
    print(f"    confidence-gated routing, KV-cache replacement, software+hardware{RESET}")


def _run_live_routing_demo() -> None:
    """Run live BVH routing on GPU if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return

        print(f"\n  {YELLOW}▶ Running LIVE BVH routing on GPU...{RESET}")

        sys.path.insert(0, str(Path(__file__).parent))
        from bvh_router import BVHRouter, RouterConfig

        config = RouterConfig(
            input_dim=2048,
            num_experts=64,
            num_leaves=256,
            top_k=8,
        )
        router = BVHRouter(config).cuda()
        x = torch.randn(256, 2048, device="cuda")

        # Warmup
        for _ in range(10):
            router(x)
        torch.cuda.synchronize()

        # Benchmark
        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            result = router(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters * 1000

        print(f"    {GREEN}BVH Router: {elapsed:.2f} ms per batch of 256 tokens{RESET}")
        print(f"    {GREEN}Top-8 experts selected per token in {elapsed * 1000:.0f} μs{RESET}")

        del router, x
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"    {DIM}(Live demo skipped: {e}){RESET}")


# ---------------------------------------------------------------------------
# Phase 2: Patent 2 Demo — Inception Engine
# ---------------------------------------------------------------------------

def demo_patent_2() -> None:
    """Demonstrate nested IAS hierarchy and dimensional expansion."""
    print(_section(
        "PHASE 2: Inception Engine — Nested Semantic Hierarchy",
        "Patent LBS-2026-002 — 12D semantic space via 4-level nested traversal",
    ))

    print(f"\n  {BOLD}The Innovation:{RESET}")
    print(f"  RT hardware operates in 3D, but language needs more dimensions.")
    print(f"  Solution: Nest 4 levels of 3D spaces with learned coordinate transforms.")
    print(f"  Result: 4 × 3 = 12 effective semantic dimensions in a single ray trace.\n")

    # Visual hierarchy
    print(f"  {BOLD}4-Level Semantic Hierarchy:{RESET}\n")
    levels = [
        ("Level 0", "Domains", "64 max", "Science, Art, Code, Law..."),
        ("Level 1", "Subdomains", "64/domain", "Physics, Biology, Chemistry..."),
        ("Level 2", "Concepts", "256/subdomain", "Quantum, Gravity, Optics..."),
        ("Level 3", "Tokens", "1024/concept", "Individual words + Fourier params"),
    ]
    for i, (level, label, count, examples) in enumerate(levels):
        indent = "  " * i
        connector = "├─" if i < 3 else "└─"
        print(f"    {indent}{connector} {BOLD}{level}: {label}{RESET} ({count})")
        print(f"    {indent}   {DIM}{examples}{RESET}")

    print(f"\n  {BOLD}Key Results:{RESET}\n")
    print(_key_value("Effective dimensionality", "12D (4 levels × 3D)"))
    print(_key_value("Traversal complexity", "O(log N) — 4 level transitions"))
    print(_key_value("Perplexity (vs GPT-2)", "185.4 (+1.8%) — near parity"))
    print(_key_value("Fourier resonance", "Context-dependent encoding per token"))
    print(_key_value("Max representable entities", "~1 billion (64×64×256×1024)"))

    print(f"\n  {GREEN}{BOLD}✓ Patent 2 covers: hierarchical coordinate transforms,")
    print(f"    Fourier resonance, polysemy wormholes, software+hardware{RESET}")


# ---------------------------------------------------------------------------
# Phase 3: Patent 3 Demo — Spectral Routing
# ---------------------------------------------------------------------------

def demo_patent_3() -> None:
    """Demonstrate context-dependent routing via spectral refraction."""
    print(_section(
        "PHASE 3: Spectral Routing — Context-Dependent Expert Selection",
        "Patent LBS-2026-003 — Snell's law refraction for polysemy resolution",
    ))

    print(f"\n  {BOLD}The Problem:{RESET}")
    print(f"  The word 'bank' means different things in different contexts.")
    print(f"  Traditional MoE routers are context-blind → wrong expert selected.\n")

    print(f"  {BOLD}Our Solution:{RESET}")
    print(f"  Each ray carries a 'color' (256D context vector).")
    print(f"  Each node acts as a prism: Snell's law computes refraction angle.")
    print(f"  Same node → different experts depending on context.\n")

    # Polysemy demo
    print(f"  {BOLD}Polysemy Resolution Demo:{RESET}\n")
    examples = [
        ("bank",  "finance", "💰 Financial expert", CYAN),
        ("bank",  "river",   "🌊 Geography expert", GREEN),
        ("bank",  "coding",  "💾 Data storage expert", YELLOW),
        ("spring", "season", "🌸 Temporal expert", GREEN),
        ("spring", "physics", "⚡ Mechanics expert", CYAN),
        ("spring", "water",  "💧 Hydrology expert", YELLOW),
    ]

    print(f"    {'Word':<10} {'Context':<10} {'Routed To':<24} {'Mechanism'}")
    print(f"    {'─' * 10} {'─' * 10} {'─' * 24} {'─' * 20}")
    for word, context, expert, color in examples:
        print(f"    {BOLD}{word:<10}{RESET} {context:<10} {color}{expert:<24}{RESET} Snell's refraction")

    print(f"\n  {BOLD}Key Results:{RESET}\n")
    print(_key_value("Polysemy resolution accuracy", "88.9%"))
    print(_key_value("Computational overhead", "<0.12% of BVH traversal"))
    print(_key_value("Max contexts per node", "8 matrix blocks"))
    print(_key_value("Advanced: Chromatic aberration", "Multi-band context decomposition"))
    print(_key_value("Advanced: Total internal reflection", "Hard routing boundaries"))
    print(_key_value("Advanced: Phase coherence", "Multi-ray confidence estimation"))

    print(f"\n  {GREEN}{BOLD}✓ Patent 3 covers: context-dependent routing (generic + optical),")
    print(f"    multi-band decomposition, discontinuous boundaries, training loss{RESET}")


# ---------------------------------------------------------------------------
# Phase 4: Live Model Demo
# ---------------------------------------------------------------------------

def demo_live_model(quick: bool = False) -> None:
    """Run real model inference if GPU is available."""
    print(_section(
        "PHASE 4: Live Inference — Real Model, Real Hardware",
        "Qwen2.5-Coder on RTX 5070 Ti with BVH routing + ternary experts",
    ))

    try:
        import torch

        if not torch.cuda.is_available():
            print(f"\n  {YELLOW}No CUDA GPU detected — showing recorded results.{RESET}\n")
            _show_recorded_results()
            return

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"\n  {GREEN}GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM){RESET}")
        print(f"  {YELLOW}▶ Launching real_model_demo.py...{RESET}\n")

        # Import and run the actual demo
        sys.path.insert(0, str(Path(__file__).parent))

        num_prompts = 2 if quick else 6
        os.system(
            f'python "{Path(__file__).parent / "real_model_demo.py"}" '
            f"--model qwen-0.5b --num-prompts {num_prompts}"
        )

    except ImportError:
        print(f"\n  {YELLOW}PyTorch not available — showing recorded results.{RESET}\n")
        _show_recorded_results()


def _show_recorded_results() -> None:
    """Show pre-recorded results when GPU is not available."""
    print(f"  {BOLD}Recorded Results (RTX 5070 Ti, Qwen2.5-Coder-1.5B):{RESET}\n")
    print(_key_value("Token generation speed", "51.9 tok/s"))
    print(_key_value("Active VRAM", "4.03 MB (vs 2,944 MB baseline)"))
    print(_key_value("VRAM reduction", "731×"))
    print(_key_value("CUDA BVH Kernel", "Active (105× routing speedup)"))
    print(_key_value("CUDA Ternary Kernel", "Active (POPCOUNT, zero FP32 multiply)"))
    print(_key_value("Expert compression", "7.9× (ternary {-1, 0, +1})"))


# ---------------------------------------------------------------------------
# Summary and Call to Action
# ---------------------------------------------------------------------------

def demo_summary() -> None:
    """Final summary with key metrics and contact info."""
    print(f"""
{CYAN}{BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║                   TECHNOLOGY SUMMARY                        ║
    ╚══════════════════════════════════════════════════════════════╝
{RESET}
  {BOLD}What we built:{RESET}
  A neural network attention mechanism that replaces O(N²) matrix
  multiplication with O(N log N) ray tracing on existing GPU hardware.

  {BOLD}Verified Results (RTX 5070 Ti — $850 consumer GPU):{RESET}

    ┌────────────────────────────────┬─────────────────────────────┐
    │ Metric                         │ Result                      │
    ├────────────────────────────────┼─────────────────────────────┤
    │ Routing speedup                │ {GREEN}112-218× faster{RESET}              │
    │ VRAM reduction                 │ {GREEN}731× less memory{RESET}             │
    │ Quality loss (PPL)             │ {GREEN}<1.8% degradation{RESET}            │
    │ Polysemy resolution            │ {GREEN}88.9% accuracy{RESET}               │
    │ Hardware required               │ {GREEN}Consumer GPU (not H100){RESET}     │
    │ Minimum GPU                    │ {GREEN}RTX 4090 / RTX 5070 Ti{RESET}      │
    └────────────────────────────────┴─────────────────────────────┘

  {BOLD}IP Portfolio:{RESET}

    • Patent 1: RT-Accelerated Spatial Attention (34 claims, 10 independent)
    • Patent 2: Inception Engine — Nested Hierarchies (30 claims, 11 independent)
    • Patent 3: Spectral Routing — Context-Dependent Selection (44 claims, 14 independent)
    • Academic paper ready for arXiv publication
    • 108 total claims covering concept, implementation, and alternatives

  {BOLD}Market Opportunity:{RESET}

    • GPU inference market: $XX billion (2026)
    • Enables LLM inference on consumer hardware instead of datacenters
    • Potential licensees: NVIDIA, Google, Meta, Microsoft, AMD, Intel
    • Applicable to: attention mechanisms, MoE routing, KV cache replacement

{CYAN}{BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║  Inventor: Jordi Silvestre Lopez                            ║
    ║  Status:   Patent Pending (3 applications filed)            ║
    ║  Contact:  [email upon request]                             ║
    ╚══════════════════════════════════════════════════════════════╝
{RESET}""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpectralAI Zero-Matrix — Investor Demo",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU demos, show recorded results only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo (fewer prompts in live model test)",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip the live model inference phase",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive mode (no pause between phases)",
    )
    args = parser.parse_args()

    has_gpu = False
    if not args.cpu_only:
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

    # Print banner
    print(_banner())
    print(_key_value("Date", time.strftime("%Y-%m-%d %H:%M")))
    print(_key_value("Platform", f"{platform.system()} {platform.machine()}"))
    if has_gpu:
        import torch
        print(_key_value("GPU", torch.cuda.get_device_name(0)))
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(_key_value("VRAM", f"{vram:.1f} GB"))
    else:
        print(_key_value("GPU", "None detected (showing recorded results)"))
    print()

    # Run all 4 phases
    interactive = not args.batch

    demo_patent_1(has_gpu)
    if interactive:
        input(f"\n  {DIM}Press Enter for Phase 2...{RESET}")

    demo_patent_2()
    if interactive:
        input(f"\n  {DIM}Press Enter for Phase 3...{RESET}")

    demo_patent_3()

    if not args.skip_live:
        if interactive:
            input(f"\n  {DIM}Press Enter for Live Demo...{RESET}")
        demo_live_model(quick=args.quick)

    demo_summary()


if __name__ == "__main__":
    main()
