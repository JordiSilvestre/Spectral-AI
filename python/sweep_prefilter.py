#!/usr/bin/env python3
"""
sweep_prefilter.py -- Pre-filter candidate sweep for BVH Router.

Tests how many candidates the BVH Router needs to pre-select before
the original gate computes exact routing weights. Measures PPL for
each candidate count to find the minimum without degradation.

Sweep: 16, 18, 20, 22, 24, 28, 32, 48, 64 candidates.

Usage:
    python sweep_prefilter.py --model-dir /path/to/olmoe-1b-7b \
        --router-dir checkpoints/olmoe_distill --max-tokens 50000

    # Quick test
    python sweep_prefilter.py --model-dir /path/to/olmoe-1b-7b \
        --max-tokens 10000 --candidates 16 24 32

Copyright (c) 2026 SpectralAI Studio
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure Python path includes our modules
sys.path.insert(0, os.path.dirname(__file__))


DEFAULT_CANDIDATES = [16, 18, 20, 22, 24, 28, 32, 48, 64]
DEFAULT_LAYERS = list(range(16))  # All 16 layers


def load_model(model_dir: str):
    """Load OLMoE model and tokenizer."""
    print(f"Loading model from {model_dir}...")
    is_local = os.path.isdir(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=is_local,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def evaluate_ppl(model, tokenizer, max_length: int = 2048,
                 stride: int = 512, max_tokens: int = 50000,
                 device: str = "cuda") -> float:
    """Sliding window perplexity on WikiText-2 validation."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="validation")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    if input_ids.size(1) > max_tokens:
        input_ids = input_ids[:, :max_tokens]

    seq_len = input_ids.size(1)
    nlls = []
    n_tokens = 0

    with torch.no_grad():
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_length, seq_len)
            chunk = input_ids[:, begin:end].to(device)
            target = chunk.clone()

            if begin > 0:
                target[:, :-stride] = -100

            outputs = model(chunk, labels=target, output_router_logits=False)
            n_valid = (target != -100).sum().item()
            nlls.append(outputs.loss.item() * n_valid)
            n_tokens += n_valid

    return math.exp(sum(nlls) / n_tokens)


def install_prefilter_hooks(model, router_dir: str, layers: List[int],
                            num_candidates: int) -> List[int]:
    """Install BVH pre-filter hooks on specified layers.

    The hook replaces the gate's forward to:
    1. Run BVH Router to get top-num_candidates experts
    2. Run original gate on full input
    3. Zero out experts not in the BVH top-num_candidates
    4. Re-normalize remaining weights

    This simulates the hybrid mode with pre-filtering.
    """
    from bvh_router import BVHRouter, RouterConfig

    installed = []

    for layer_idx in layers:
        # Find checkpoint
        ckpt_path = os.path.join(router_dir, f"layer{layer_idx}",
                                 "bvh_router_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(router_dir, "bvh_router_best.pt")
        if not os.path.exists(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", RouterConfig(embed_dim=1024))

        router = BVHRouter(cfg)
        router.load_state_dict(ckpt["router_state_dict"])
        router.eval()

        moe_layer = model.model.layers[layer_idx].mlp
        gate = moe_layer.gate
        router = router.to(gate.weight.device)

        # Store originals
        original_forward = gate.forward

        def make_hook(bvh_router, orig_fwd, n_cand):
            def hooked_forward(x):
                # 1. BVH routing to get candidate set
                with torch.no_grad():
                    bvh_result = bvh_router(x)
                    bvh_logits = bvh_result.expert_logits

                # Get top-n_cand indices from BVH
                _, bvh_top = torch.topk(bvh_logits, min(n_cand, 64), dim=-1)

                # 2. Original gate forward
                gate_logits = orig_fwd(x)

                # 3. Mask out experts not in BVH candidates
                mask = torch.zeros_like(gate_logits, dtype=torch.bool)
                mask.scatter_(1, bvh_top, True)
                gate_logits = gate_logits.masked_fill(~mask, float("-inf"))

                return gate_logits
            return hooked_forward

        gate.forward = make_hook(router, original_forward, num_candidates)
        installed.append(layer_idx)

    return installed


def remove_hooks(model, layers: List[int]):
    """Remove pre-filter hooks (restore original gate forward)."""
    for layer_idx in layers:
        moe_layer = model.model.layers[layer_idx].mlp
        gate = moe_layer.gate
        if hasattr(gate, "_original_forward"):
            gate.forward = gate._original_forward


def run_sweep(model, tokenizer, router_dir: str,
              candidate_counts: List[int], layers: List[int],
              max_tokens: int, device: str) -> Dict:
    """Run the full pre-filter sweep."""
    results = {}

    # 1. Baseline (no pre-filtering)
    print("\n--- Baseline (no pre-filter, full 64 experts) ---")
    t0 = time.time()
    baseline_ppl = evaluate_ppl(model, tokenizer, max_tokens=max_tokens,
                                device=device)
    print(f"  PPL: {baseline_ppl:.4f} ({time.time()-t0:.1f}s)")
    results["baseline"] = {"candidates": 64, "ppl": baseline_ppl, "delta": 0.0}

    # 2. Sweep each candidate count
    for n_cand in candidate_counts:
        print(f"\n--- Candidates: {n_cand}/{64} ---")

        installed = install_prefilter_hooks(
            model, router_dir, layers, n_cand
        )
        print(f"  Installed hooks on {len(installed)} layers")

        t0 = time.time()
        ppl = evaluate_ppl(model, tokenizer, max_tokens=max_tokens,
                           device=device)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        elapsed = time.time() - t0

        print(f"  PPL: {ppl:.4f} (delta: {delta:+.2f}%) ({elapsed:.1f}s)")

        results[str(n_cand)] = {
            "candidates": n_cand,
            "ppl": ppl,
            "delta_pct": delta,
            "layers_hooked": len(installed),
            "search_reduction": f"{64/n_cand:.1f}x",
        }

        # Remove hooks for next iteration
        remove_hooks(model, installed)

    return results


def print_summary(results: Dict):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("PRE-FILTER SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Candidates':>12} | {'PPL':>8} | {'Delta':>8} | {'Search Reduction':>16}")
    print("-" * 70)

    baseline = results.get("baseline", {}).get("ppl", 0)

    for key in sorted(results.keys(), key=lambda k: results[k]["candidates"]):
        r = results[key]
        cand = r["candidates"]
        ppl = r["ppl"]
        delta = r.get("delta_pct", 0.0)
        reduction = r.get("search_reduction", "1.0x")
        marker = " <-- BASELINE" if cand == 64 else ""
        marker = " <-- BEST" if abs(delta) < 0.5 and cand < 64 and cand == min(
            [results[k]["candidates"] for k in results
             if abs(results[k].get("delta_pct", 999)) < 0.5
             and results[k]["candidates"] < 64],
            default=64
        ) else marker
        print(f"{cand:>12} | {ppl:>8.4f} | {delta:>+7.2f}% | {reduction:>16}{marker}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter candidate sweep for BVH Router"
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--router-dir", type=str,
                        default="checkpoints/olmoe_distill")
    parser.add_argument("--candidates", type=int, nargs="*",
                        default=DEFAULT_CANDIDATES,
                        help="Candidate counts to test")
    parser.add_argument("--layers", type=int, nargs="*",
                        default=DEFAULT_LAYERS,
                        help="Layers to apply pre-filter")
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="sweep_prefilter.json")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)

    results = run_sweep(
        model, tokenizer, args.router_dir,
        args.candidates, args.layers, args.max_tokens, args.device
    )

    print_summary(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
