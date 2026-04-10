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

Copyright (c) 2026 Jordi Silvestre Lopez
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


def _load_enhanced_router(ckpt_path: str, device: str = "cpu"):
    """Load an EnhancedBVHRouter from a checkpoint file.

    Handles config extraction and spectral mode detection,
    matching the logic in olmoe_e2e_eval.py.
    """
    from olmoe_bvh_distill import EnhancedBVHRouter

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    if not isinstance(config, dict):
        config = {"input_dim": 1024, "n_level1": 4, "n_level2": 4,
                  "n_level3": 4, "feature_dim": 128}

    sd = ckpt["router_state_dict"]
    spectral_mode = config.get("spectral_mode",
                               ckpt.get("spectral_mode", False))
    se_out_key = "spectral_encoder.2.weight"
    se_in_key = "spectral_encoder.0.weight"
    if not spectral_mode and se_out_key in sd:
        spectral_mode = True
    spectral_dim = config.get("spectral_dim", 64)
    enc_hidden = None
    if spectral_mode and se_out_key in sd:
        spectral_dim = sd[se_out_key].shape[0]
        enc_hidden = sd[se_in_key].shape[0]

    router = EnhancedBVHRouter(
        input_dim=config.get("input_dim", 1024),
        n_level1=config.get("n_level1", 4),
        n_level2=config.get("n_level2", 4),
        n_level3=config.get("n_level3", 4),
        feature_dim=config.get("feature_dim", 128),
        spectral_mode=spectral_mode,
        spectral_dim=spectral_dim,
        encoder_hidden=enc_hidden,
    )
    router.load_state_dict(sd)
    router.eval()
    router = router.to(device)
    return router


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
    installed = []

    # Checkpoint naming: checkpoints/olmoe_distill_layer{N}/bvh_router_best.pt
    base_dir = os.path.dirname(router_dir)  # e.g. checkpoints/
    base_name = os.path.basename(router_dir)  # e.g. olmoe_distill

    for layer_idx in layers:
        # Try per-layer checkpoint first (olmoe_distill_layer8/)
        ckpt_path = os.path.join(
            base_dir, f"{base_name}_layer{layer_idx}",
            "bvh_router_best.pt"
        )
        if not os.path.exists(ckpt_path):
            # Fallback: subdirectory format
            ckpt_path = os.path.join(
                router_dir, f"layer{layer_idx}", "bvh_router_best.pt"
            )
        if not os.path.exists(ckpt_path):
            # Last fallback: global checkpoint
            ckpt_path = os.path.join(router_dir, "bvh_router_best.pt")
        if not os.path.exists(ckpt_path):
            continue

        moe_layer = model.model.layers[layer_idx].mlp
        gate = moe_layer.gate

        # Determine device from gate weights
        try:
            gate_device = next(gate.parameters()).device
            if str(gate_device) == "meta":
                gate_device = torch.device("cuda")
        except StopIteration:
            gate_device = torch.device("cuda")

        router = _load_enhanced_router(ckpt_path, device=str(gate_device))

        def make_hook(bvh_router, gate_module, n_cand):
            """Hook that pre-filters via BVH then runs original gate logic.

            OlmoeTopKRouter.forward returns (router_logits, scores, indices).
            We intercept: run BVH for candidate set, then run original gate
            but mask out non-candidate experts before topk selection.
            """
            import torch.nn.functional as F

            # Capture gate attributes before hooking
            gate_weight = gate_module.weight
            gate_hidden_dim = gate_module.hidden_dim
            gate_top_k = gate_module.top_k
            gate_norm = gate_module.norm_topk_prob

            def hooked_forward(x):
                # 1. BVH routing to get candidate set
                with torch.no_grad():
                    bvh_probs, _ = bvh_router(x.float())  # (B, 64)

                # Get top-n_cand indices from BVH
                _, bvh_top = torch.topk(
                    bvh_probs, min(n_cand, 64), dim=-1
                )

                # 2. Original gate linear + softmax
                x_flat = x.reshape(-1, gate_hidden_dim)
                logits = F.linear(x_flat, gate_weight)  # (B, 64)

                # 3. Mask out experts not in BVH candidates
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, bvh_top, True)
                logits = logits.masked_fill(~mask, float("-inf"))

                # 4. Softmax + topk (same as original gate)
                probs = F.softmax(logits, dtype=torch.float, dim=-1)
                top_val, top_idx = torch.topk(
                    probs, gate_top_k, dim=-1
                )
                if gate_norm:
                    top_val = top_val / top_val.sum(dim=-1, keepdim=True)
                top_val = top_val.to(probs.dtype)

                return probs, top_val, top_idx
            return hooked_forward

        gate.forward = make_hook(router, gate, num_candidates)
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
