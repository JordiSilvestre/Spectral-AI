#!/usr/bin/env python3
"""
eval_all_16_layers.py — Run 16/16 PPL evaluation once all checkpoints are ready.

Generates the --multi-layer argument for olmoe_e2e_eval.py and runs it.
Pre-validates all 16 checkpoints exist and are calibrated.

Usage:
    python scripts/eval_all_16_layers.py [--model-dir /path/to/olmoe] [--device cuda]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).parent.parent
CHECKPOINT_BASE = PROJECT_DIR / "checkpoints"
NUM_LAYERS = 16


def get_checkpoint_path(layer: int) -> Path:
    if layer == 8:
        return CHECKPOINT_BASE / "olmoe_distill" / "bvh_router_best.pt"
    return CHECKPOINT_BASE / f"olmoe_distill_layer{layer}" / "bvh_router_best.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to OLMoE-1B-7B model directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    # Validate all 16 checkpoints
    pairs = []
    missing = []
    uncalibrated = []

    for layer in range(NUM_LAYERS):
        path = get_checkpoint_path(layer)
        if not path.exists():
            missing.append(layer)
            continue

        if not args.skip_validation:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if ckpt.get("calibration_mode") is None:
                uncalibrated.append(layer)

        pairs.append(f"{layer}:{path}")

    if missing:
        print(f"FAIL: Missing checkpoints for layers: {missing}")
        print("Run: python scripts/train_missing_layers.py")
        return 1

    if uncalibrated:
        print(f"WARN: Uncalibrated layers: {uncalibrated}")
        print("Run: python scripts/calibrate_all_layers.py")
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != "y":
            return 1

    multi_layer_arg = ",".join(pairs)
    print(f"Running 16/16 evaluation...")
    print(f"Multi-layer arg: {multi_layer_arg[:100]}...")

    eval_script = PROJECT_DIR / "python" / "olmoe_e2e_eval.py"
    cmd = [
        sys.executable, str(eval_script),
        "--model-dir", args.model_dir,
        "--multi-layer", multi_layer_arg,
        "--device", args.device,
        "--max-tokens", str(args.max_tokens),
    ]

    return subprocess.run(cmd, cwd=str(PROJECT_DIR)).returncode


if __name__ == "__main__":
    sys.exit(main())
