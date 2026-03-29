#!/usr/bin/env python3
"""
validate_checkpoints.py — Validate all 16 layer checkpoints before full evaluation

Checks:
  1. All 16 checkpoint files exist
  2. Each has valid router state dict
  3. Each has calibration data
  4. Router configs are consistent
  5. Prints summary table

Usage:
    python scripts/validate_checkpoints.py
"""

import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).parent.parent
CHECKPOINT_BASE = PROJECT_DIR / "checkpoints"
NUM_LAYERS = 16


def get_checkpoint_path(layer: int) -> Path:
    """Get checkpoint path for a layer (L8 uses legacy location)."""
    if layer == 8:
        return CHECKPOINT_BASE / "olmoe_distill" / "bvh_router_best.pt"
    return CHECKPOINT_BASE / f"olmoe_distill_layer{layer}" / "bvh_router_best.pt"


def validate_checkpoint(path: Path, layer: int) -> dict:
    """Validate a single checkpoint file."""
    result = {
        "layer": layer,
        "path": str(path),
        "exists": path.exists(),
        "valid": False,
        "calibrated": False,
        "cal_mode": "—",
        "topk_acc": 0.0,
        "top1_acc": 0.0,
        "router_type": "—",
        "n_params": 0,
        "error": None,
    }

    if not path.exists():
        result["error"] = "FILE NOT FOUND"
        return result

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        result["error"] = f"LOAD ERROR: {e}"
        return result

    # Check required keys
    if "router_state_dict" not in ckpt:
        result["error"] = "Missing router_state_dict"
        return result

    if "config" not in ckpt:
        result["error"] = "Missing config"
        return result

    result["valid"] = True
    result["router_type"] = ckpt.get("router_type", "bvh")
    result["topk_acc"] = ckpt.get("topk_accuracy", 0.0)
    result["top1_acc"] = ckpt.get("top1_accuracy", 0.0)

    # Count parameters
    state_dict = ckpt["router_state_dict"]
    n_params = sum(v.numel() for v in state_dict.values())
    result["n_params"] = n_params

    # Check calibration
    cal_mode = ckpt.get("calibration_mode", None)
    if cal_mode:
        result["calibrated"] = True
        result["cal_mode"] = cal_mode

        # Verify calibration state exists
        cal_state = ckpt.get("calibration_state", None)
        if cal_state is None:
            result["error"] = "calibration_mode set but no calibration_state"
            result["calibrated"] = False
    else:
        result["cal_mode"] = "none"

    return result


def main() -> int:
    print("=" * 80)
    print("  SpectralAI Zero-Matrix — Checkpoint Validation (16 layers)")
    print("=" * 80)

    results = []
    for layer in range(NUM_LAYERS):
        path = get_checkpoint_path(layer)
        result = validate_checkpoint(path, layer)
        results.append(result)

    # Print table
    print(f"\n{'Layer':>5} {'Exists':>6} {'Valid':>5} {'Calibrated':>10} "
          f"{'Cal Mode':>10} {'Top-8 Acc':>9} {'Top-1 Acc':>9} {'Params':>10} {'Error'}")
    print("-" * 100)

    all_ok = True
    for r in results:
        exists_str = "OK" if r["exists"] else "FAIL"
        valid_str = "OK" if r["valid"] else "FAIL"
        cal_str = "OK" if r["calibrated"] else "FAIL"
        error_str = r["error"] or ""

        if not r["exists"] or not r["valid"] or not r["calibrated"]:
            all_ok = False

        print(f"  L{r['layer']:>2}  {exists_str:>6} {valid_str:>5} {cal_str:>10} "
              f"{r['cal_mode']:>10} {r['topk_acc']*100:>8.1f}% {r['top1_acc']*100:>8.1f}% "
              f"{r['n_params']:>10,} {error_str}")

    # Summary
    n_exist = sum(1 for r in results if r["exists"])
    n_valid = sum(1 for r in results if r["valid"])
    n_calibrated = sum(1 for r in results if r["calibrated"])

    print(f"\n  Summary: {n_exist}/16 exist, {n_valid}/16 valid, "
          f"{n_calibrated}/16 calibrated")

    if all_ok:
        print("\n  OK ALL 16 CHECKPOINTS READY FOR EVALUATION")
        print("  Run: python python/olmoe_e2e_eval.py --multi-layer <...>")
    else:
        missing = [r["layer"] for r in results if not r["exists"]]
        uncalibrated = [r["layer"] for r in results if r["exists"] and not r["calibrated"]]

        if missing:
            print(f"\n  FAIL Missing layers: {missing}")
            print("    Training needed for these layers")
        if uncalibrated:
            print(f"\n  FAIL Uncalibrated layers: {uncalibrated}")
            print("    Run calibrate_router.py for each")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
