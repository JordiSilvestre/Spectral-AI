#!/usr/bin/env python3
"""
calibrate_router.py — Post-hoc weight calibration for pure BVH routing.

The BVH router selects the right experts (91.7% top-8) but assigns wrong
weights because its internal softmax distribution differs from the original
gate's. This script learns a calibration layer on the router's raw logits
so that the resulting softmax matches the gate's.

Two modes:
  --mode affine   : per-expert scale + bias (128 params) — fast, decent
  --mode linear   : Linear(64,64) mixing layer (4160 params) — better, captures
                    inter-expert interactions. Still 30x smaller than gate (131K).

Usage:
    python calibrate_router.py \
        --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt \
        --real-data data/real_hiddens_layer8.pt \
        --mode linear

Copyright (c) 2026 LiquidBit Studio — Apache 2.0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from olmoe_bvh_distill import EnhancedBVHRouter


def calibrate(
    router: EnhancedBVHRouter,
    hidden_states: torch.Tensor,
    gate_probs: torch.Tensor,
    mode: str = "affine",
    n_experts: int = 64,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 1024,
    device: str = "cuda",
) -> dict:
    """
    Learn a calibration transform on BVH logits to match the gate's distribution.

    Mode 'affine': calibrated = logits * scale + bias  (128 params)
    Mode 'linear': calibrated = Linear(logits)         (4160 params)

    Returns dict with calibration state.
    """
    router.eval()
    router = router.to(device)

    if mode == "affine":
        scale = nn.Parameter(torch.ones(n_experts, device=device))
        bias = nn.Parameter(torch.zeros(n_experts, device=device))
        params = [scale, bias]
        n_params = 2 * n_experts

        def apply_cal(logits):
            return logits * scale + bias

    elif mode == "linear":
        cal_layer = nn.Linear(n_experts, n_experts).to(device)
        # Init as identity + zero bias (start from no-op)
        nn.init.eye_(cal_layer.weight)
        nn.init.zeros_(cal_layer.bias)
        params = list(cal_layer.parameters())
        n_params = sum(p.numel() for p in params)

        def apply_cal(logits):
            return cal_layer(logits)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(hidden_states, gate_probs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    print(f"\n  Mode: {mode} ({n_params:,} params)")
    print(f"  Data: {len(hidden_states):,} samples, {epochs} epochs")

    t0 = time.time()
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for h_batch, gate_probs_batch in loader:
            h_batch = h_batch.to(device)
            gate_probs_batch = gate_probs_batch.to(device)

            with torch.no_grad():
                router(h_batch.float())
                bvh_logits = router._last_logits

            cal_logits = apply_cal(bvh_logits)

            cal_log_probs = F.log_softmax(cal_logits, dim=-1)
            # gate_probs are already probabilities (from extract_real_hiddens)
            loss = F.kl_div(cal_log_probs, gate_probs_batch, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            if mode == "affine":
                best_state = {"scale": scale.data.clone(), "bias": bias.data.clone()}
            else:
                best_state = {k: v.clone() for k, v in cal_layer.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s — best KL loss: {best_loss:.6f}")

    return {"mode": mode, "n_params": n_params, "state": best_state}


def apply_calibration(bvh_logits: torch.Tensor, cal_data: dict,
                      device: str = "cuda") -> torch.Tensor:
    """Apply saved calibration to BVH logits."""
    state = cal_data["state"]
    if cal_data["mode"] == "affine":
        return bvh_logits * state["scale"].to(device) + state["bias"].to(device)
    else:
        layer = nn.Linear(bvh_logits.shape[-1], bvh_logits.shape[-1])
        layer.load_state_dict({k: v.to(device) for k, v in state.items()})
        layer = layer.to(device)
        layer.eval()
        with torch.no_grad():
            return layer(bvh_logits)


def evaluate_calibration(
    router: EnhancedBVHRouter,
    cal_data: dict,
    hidden_states: torch.Tensor,
    gate_probs: torch.Tensor,
    device: str = "cuda",
):
    """Show how calibration improves weight distribution matching."""
    router.eval()
    router = router.to(device)

    n_samples = min(10000, len(hidden_states))
    h = hidden_states[:n_samples].to(device)
    gate_p = gate_probs[:n_samples].to(device)

    with torch.no_grad():
        # Uncalibrated
        probs_raw, _ = router(h.float())
        raw_top8_vals, raw_top8_ids = torch.topk(probs_raw, 8, dim=-1)

        # Calibrated
        router(h.float())
        bvh_logits = router._last_logits
        cal_logits = apply_calibration(bvh_logits, cal_data, device)
        probs_cal = F.softmax(cal_logits, dim=-1)
        cal_top8_vals, cal_top8_ids = torch.topk(probs_cal, 8, dim=-1)

        # Gate reference
        gate_top8_vals, gate_top8_ids = torch.topk(gate_p, 8, dim=-1)

    # Weight distribution comparison
    print(f"\n  Weight Distribution (top-8 mean values, {n_samples:,} samples):")
    print(f"  {'Position':>10}  {'Gate':>10}  {'BVH raw':>10}  {'BVH cal':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k in range(8):
        g = gate_top8_vals[:, k].mean().item()
        r = raw_top8_vals[:, k].mean().item()
        c = cal_top8_vals[:, k].mean().item()
        print(f"  {'top-'+str(k+1):>10}  {g:>10.4f}  {r:>10.4f}  {c:>10.4f}")

    # Cosine similarity
    cos_raw = F.cosine_similarity(
        gate_p.float().flatten().unsqueeze(0),
        probs_raw.float().flatten().unsqueeze(0),
    ).item()
    cos_cal = F.cosine_similarity(
        gate_p.float().flatten().unsqueeze(0),
        probs_cal.float().flatten().unsqueeze(0),
    ).item()
    print(f"\n  Full distribution cosine: raw={cos_raw:.4f} -> calibrated={cos_cal:.4f}")

    # Top-8 overlap
    overlap_raw = 0
    overlap_cal = 0
    for i in range(n_samples):
        g_set = set(gate_top8_ids[i].tolist())
        overlap_raw += len(g_set & set(raw_top8_ids[i].tolist())) / 8
        overlap_cal += len(g_set & set(cal_top8_ids[i].tolist())) / 8
    print(f"  Top-8 overlap: raw={overlap_raw/n_samples*100:.1f}% -> calibrated={overlap_cal/n_samples*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc weight calibration for BVH router")
    parser.add_argument("--router-checkpoint", type=str,
                        default="checkpoints/olmoe_distill/bvh_router_best.pt")
    parser.add_argument("--real-data", type=str,
                        default="data/real_hiddens_layer8.pt")
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["affine", "linear"],
                        help="affine: scale+bias (128 params), linear: 64x64 (4160 params)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Max samples to use (20K is plenty for 4K params)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LiquidBit — BVH Router Weight Calibration")
    print("=" * 60)

    # Load router
    print("\n[1/4] Loading router...")
    try:
        ckpt = torch.load(args.router_checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(args.router_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    router = EnhancedBVHRouter(
        input_dim=config["input_dim"],
        n_level1=config["n_level1"],
        n_level2=config["n_level2"],
        n_level3=config["n_level3"],
        feature_dim=config["feature_dim"],
    )
    router.load_state_dict(ckpt["router_state_dict"])
    print(f"  top-8={ckpt.get('topk_accuracy', 0)*100:.1f}%, "
          f"top-1={ckpt.get('top1_accuracy', 0)*100:.1f}%")

    # Load real data (subsample to save memory — 20K is plenty for 4K params)
    print("\n[2/4] Loading real hidden states...")
    try:
        data = torch.load(args.real_data, map_location="cpu", weights_only=True)
    except Exception:
        data = torch.load(args.real_data, map_location="cpu", weights_only=False)
    hidden_states = data["hidden_states"]
    gate_probs = data["gate_logits"]
    del data  # free the dict shell
    if args.max_samples and len(hidden_states) > args.max_samples:
        idx = torch.randperm(len(hidden_states))[:args.max_samples]
        hidden_states = hidden_states[idx].clone()
        gate_probs = gate_probs[idx].clone()
        del idx
    print(f"  {len(hidden_states):,} samples (from file, capped at {args.max_samples})")

    # Calibrate
    print("\n[3/4] Calibrating...")
    cal_data = calibrate(
        router, hidden_states, gate_probs,
        mode=args.mode, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=args.device,
    )

    # Save to checkpoint (before evaluation to avoid losing data on print errors)
    ckpt["calibration_mode"] = cal_data["mode"]
    ckpt["calibration_state"] = {k: v.cpu() for k, v in cal_data["state"].items()}
    # Keep backward compat: also save as scale/bias if affine
    if cal_data["mode"] == "affine":
        ckpt["calibration_scale"] = cal_data["state"]["scale"].cpu()
        ckpt["calibration_bias"] = cal_data["state"]["bias"].cpu()
    else:
        # Remove old affine keys if present
        ckpt.pop("calibration_scale", None)
        ckpt.pop("calibration_bias", None)

    save_path = args.router_checkpoint
    torch.save(ckpt, save_path)
    print(f"\n  Saved {cal_data['mode']} calibration ({cal_data['n_params']:,} params) to {save_path}")

    # Evaluate (optional — print quality metrics)
    print("\n[4/4] Evaluating calibration quality...")
    evaluate_calibration(router, cal_data, hidden_states, gate_probs, args.device)


if __name__ == "__main__":
    main()
