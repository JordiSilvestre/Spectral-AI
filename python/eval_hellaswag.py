#!/usr/bin/env python3
"""
HellaSwag evaluation for SpectralAI BVH Router.

Measures downstream task accuracy (commonsense reasoning) with and without
BVH Router replacement. Uses the lm-eval-harness format internally but
implemented standalone for minimal dependencies.

Usage:
    # Baseline (no BVH router)
    python eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b

    # With BVH router on specific layers
    python eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b \
        --router-layers 3 8 15 \
        --router-dir checkpoints/olmoe_distill

    # Quick test (fewer samples)
    python eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b --max-samples 100
"""

import argparse
import json
import math
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_dir: str, device: str = "cuda"):
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


def replace_router_layers(model, router_layers: List[int], router_dir: str,
                          mode: str = "hybrid"):
    """Replace gate in specified layers with BVH Router.

    Uses olmoe_e2e_eval's replace_router_in_layer for correct loading
    of EnhancedBVHRouter checkpoints.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from olmoe_e2e_eval import replace_gate_with_bvh

    replaced = []
    for layer_idx in router_layers:
        ckpt_path = os.path.join(router_dir, f"layer{layer_idx}",
                                 "bvh_router_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(router_dir, "bvh_router_best.pt")

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: No checkpoint for layer {layer_idx}, skipping")
            continue

        is_hybrid = mode == "hybrid"
        try:
            replace_gate_with_bvh(
                model, ckpt_path, layer_idx=layer_idx,
                hybrid=is_hybrid, weight_mode="relu_norm",
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            acc = ckpt.get("topk_accuracy", 0)
            print(f"  Layer {layer_idx}: BVH Router loaded (top-8: {acc:.1%})")
            replaced.append(layer_idx)
        except Exception as e:
            print(f"  WARNING: Layer {layer_idx} failed: {e}")

    print(f"  Replaced {len(replaced)}/{len(router_layers)} layers")
    return replaced


def score_completions(model, tokenizer, context: str,
                      completions: List[str], device: str = "cuda") -> List[float]:
    """Score each completion given the context. Returns log-likelihoods."""
    scores = []

    for completion in completions:
        full_text = context + completion
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        input_ids = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Only score the completion tokens (after context)
        ctx_len = len(ctx_ids)
        if ctx_len >= len(full_ids):
            scores.append(float("-inf"))
            continue

        completion_logits = logits[0, ctx_len - 1:-1, :]  # shifted by 1
        completion_targets = input_ids[0, ctx_len:]

        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(
            1, completion_targets.unsqueeze(1)
        ).squeeze(1)

        # Length-normalized log-likelihood
        score = token_log_probs.sum().item() / max(len(completion_targets), 1)
        scores.append(score)

    return scores


def evaluate_hellaswag(model, tokenizer, max_samples: Optional[int] = None,
                       device: str = "cuda") -> dict:
    """Run HellaSwag evaluation. Returns accuracy and per-sample results."""
    print("Loading HellaSwag dataset...")
    dataset = load_dataset("Rowan/hellaswag", split="validation",
                           trust_remote_code=True)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)
    correct = 0
    results = []

    print(f"Evaluating {total} samples...")
    t0 = time.time()

    for i, sample in enumerate(dataset):
        context = sample["ctx"]
        endings = sample["endings"]
        label = int(sample["label"])

        scores = score_completions(model, tokenizer, context, endings, device)
        prediction = max(range(len(scores)), key=lambda j: scores[j])

        is_correct = prediction == label
        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "correct": is_correct,
            "predicted": prediction,
            "label": label,
            "scores": scores,
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc = correct / (i + 1)
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] Accuracy: {acc:.1%} "
                  f"({correct}/{i+1}) | {rate:.1f} samples/s | "
                  f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    accuracy = correct / total

    summary = {
        "task": "hellaswag",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_seconds": elapsed,
        "samples_per_second": total / elapsed,
    }

    print(f"\n{'='*60}")
    print(f"HellaSwag Results:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s ({total/elapsed:.1f} samples/s)")
    print(f"{'='*60}")

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="HellaSwag evaluation")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to OLMoE model")
    parser.add_argument("--router-layers", type=int, nargs="*", default=None,
                        help="Layers to replace with BVH Router (e.g. 3 8 15)")
    parser.add_argument("--router-dir", type=str,
                        default="checkpoints/olmoe_distill",
                        help="Directory with router checkpoints")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["hybrid", "pure"],
                        help="Router mode: hybrid (BVH select, gate weight) "
                             "or pure (BVH only)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_dir, args.device)

    # Replace layers if specified
    config_str = "baseline"
    if args.router_layers:
        replaced = replace_router_layers(
            model, args.router_layers, args.router_dir, args.mode
        )
        config_str = f"bvh_{args.mode}_L{'_'.join(map(str, replaced))}"

    # Run evaluation
    summary, results = evaluate_hellaswag(
        model, tokenizer, args.max_samples, args.device
    )
    summary["config"] = config_str

    # Save results
    if args.output is None:
        args.output = f"hellaswag_{config_str}.json"

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
