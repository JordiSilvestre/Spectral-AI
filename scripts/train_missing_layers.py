#!/usr/bin/env python3
"""
train_missing_layers.py — Train BVH routers for layers that don't have checkpoints yet.

Usage:
    python scripts/train_missing_layers.py [--device cuda] [--epochs 30]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).parent.parent
CHECKPOINT_BASE = PROJECT_DIR / "checkpoints"
DATA_DIR = PROJECT_DIR / "data"
NUM_LAYERS = 16


def get_checkpoint_path(layer: int) -> Path:
    if layer == 8:
        return CHECKPOINT_BASE / "olmoe_distill" / "bvh_router_best.pt"
    return CHECKPOINT_BASE / f"olmoe_distill_layer{layer}" / "bvh_router_best.pt"


def get_save_dir(layer: int) -> Path:
    if layer == 8:
        return CHECKPOINT_BASE / "olmoe_distill"
    return CHECKPOINT_BASE / f"olmoe_distill_layer{layer}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model-dir", type=str,
                        default=str(PROJECT_DIR.parent / "models" / "olmoe-1b-7b"),
                        help="Path to OLMoE model for sparse upcycling init")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = []
    for layer in range(NUM_LAYERS):
        ckpt = get_checkpoint_path(layer)
        data = DATA_DIR / f"real_hiddens_layer{layer}.pt"
        if not ckpt.exists() and data.exists():
            missing.append(layer)

    if not missing:
        print("All layer checkpoints exist!")
        return 0

    print(f"Missing layers: {missing}")
    if args.dry_run:
        return 0

    train_script = PROJECT_DIR / "python" / "olmoe_bvh_distill.py"
    calibrate_script = PROJECT_DIR / "python" / "calibrate_router.py"
    failed = []

    for layer in missing:
        print(f"\n{'='*60}")
        print(f"  Training Layer {layer}")
        print(f"{'='*60}")

        save_dir = get_save_dir(layer)
        data_path = DATA_DIR / f"real_hiddens_layer{layer}.pt"

        # Train
        cmd = [
            sys.executable, str(train_script),
            "--layer", str(layer),
            "--real-data", str(data_path),
            "--save-dir", str(save_dir),
            "--model-dir", args.model_dir,
            "--epochs", str(args.epochs),
            "--device", args.device,
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
        if result.returncode != 0:
            print(f"  FAIL: Layer {layer} training failed")
            failed.append(layer)
            continue

        # Calibrate immediately after training
        ckpt_path = get_checkpoint_path(layer)
        if ckpt_path.exists():
            print(f"\n  Calibrating Layer {layer}...")
            cal_cmd = [
                sys.executable, str(calibrate_script),
                "--router-checkpoint", str(ckpt_path),
                "--real-data", str(data_path),
                "--mode", "linear",
                "--epochs", "100",
                "--device", args.device,
            ]
            cal_result = subprocess.run(cal_cmd, cwd=str(PROJECT_DIR))
            if cal_result.returncode != 0:
                print(f"  WARN: Layer {layer} calibration failed")

    print(f"\n{'='*60}")
    print(f"  Training complete: {len(missing) - len(failed)}/{len(missing)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'='*60}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
