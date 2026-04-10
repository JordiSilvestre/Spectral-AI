#!/usr/bin/env python3
"""
calibrate_all_layers.py — Batch calibrate all uncalibrated BVH router checkpoints.

Finds layers with existing checkpoints but no calibration data,
and runs linear calibration on each.

Usage:
    python scripts/calibrate_all_layers.py [--device cpu] [--epochs 100]
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
CHECKPOINT_BASE = PROJECT_DIR / "checkpoints"
DATA_DIR = PROJECT_DIR / "data"
NUM_LAYERS = 16

import torch


def get_checkpoint_path(layer: int) -> Path:
    if layer == 8:
        return CHECKPOINT_BASE / "olmoe_distill" / "bvh_router_best.pt"
    return CHECKPOINT_BASE / f"olmoe_distill_layer{layer}" / "bvh_router_best.pt"


def needs_calibration(path: Path) -> bool:
    """Check if checkpoint exists but lacks calibration."""
    if not path.exists():
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("calibration_mode") is None
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device for calibration")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--mode", default="linear", choices=["affine", "linear"])
    parser.add_argument("--dry-run", action="store_true", help="Show what would be calibrated")
    args = parser.parse_args()

    uncalibrated = []
    for layer in range(NUM_LAYERS):
        path = get_checkpoint_path(layer)
        if needs_calibration(path):
            data_path = DATA_DIR / f"real_hiddens_layer{layer}.pt"
            if data_path.exists():
                uncalibrated.append((layer, path, data_path))
            else:
                print(f"  SKIP L{layer}: checkpoint exists but no data at {data_path}")

    if not uncalibrated:
        print("All existing checkpoints are already calibrated!")
        return 0

    print(f"Found {len(uncalibrated)} layers needing calibration: "
          f"{[l for l, _, _ in uncalibrated]}")

    if args.dry_run:
        return 0

    calibrate_script = PROJECT_DIR / "python" / "calibrate_router.py"
    failed = []

    for layer, ckpt_path, data_path in uncalibrated:
        print(f"\n{'='*60}")
        print(f"  Calibrating Layer {layer}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(calibrate_script),
            "--router-checkpoint", str(ckpt_path),
            "--real-data", str(data_path),
            "--mode", args.mode,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--device", args.device,
        ]

        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
        if result.returncode != 0:
            print(f"  FAIL: Layer {layer} calibration failed (exit {result.returncode})")
            failed.append(layer)
        else:
            print(f"  OK: Layer {layer} calibrated")

    print(f"\n{'='*60}")
    print(f"  Calibration complete: {len(uncalibrated) - len(failed)}/{len(uncalibrated)} succeeded")
    if failed:
        print(f"  Failed layers: {failed}")
    print(f"{'='*60}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
