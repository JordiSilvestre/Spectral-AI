#!/usr/bin/env python3
"""
export_calibration.py — Export calibration weights for OptiX 9.0 cooperative vectors.

Converts trained calibration weights from calibrate_router.py (PyTorch FP32)
to the binary format consumed by the OptiX closest-hit shader via
optixCoopVecMatMul (FP16, INFERENCING_OPTIMAL layout).

Output formats:
  1. Binary blob (.bin) — loaded by bvh_router_bridge.py at runtime
  2. C header (.h)      — embeds weights as constexpr arrays for constant memory

Usage:
    python export_calibration.py \\
        --checkpoint checkpoints/olmoe_distill/bvh_router_best.pt \\
        --output cuda/v5/calibration_weights \\
        --mode affine

Copyright (c) 2026 SpectralAI Studio — Apache 2.0
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Number of experts must match SPECTRAL_NUM_EXPERTS in optical_attention.h
NUM_EXPERTS = 64

# CalibrationMode enum values (must match optical_attention.h)
CALIBRATION_MODE_NONE = 0
CALIBRATION_MODE_AFFINE = 1
CALIBRATION_MODE_LINEAR = 2


def load_calibration_state(checkpoint_path: str) -> dict:
    """Load calibration state from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # The checkpoint may contain calibration data directly or nested
    if "calibration" in ckpt:
        return ckpt["calibration"]
    if "state" in ckpt and "mode" in ckpt:
        return ckpt
    # Try to find calibration keys directly
    if "scale" in ckpt or "weight" in ckpt:
        return ckpt

    raise ValueError(
        f"Cannot find calibration weights in {checkpoint_path}. "
        "Expected keys: 'calibration', or 'state'+'mode', or 'scale'/'weight'."
    )


def export_affine(state: dict, output_dir: Path) -> dict:
    """
    Export affine calibration weights (scale + bias) to FP16 binary.

    Binary format (CalibrationWeights struct, affine portion):
      [0:4]     uint32  mode = 1 (CALIBRATION_MODE_AFFINE)
      [4:16]    uint32  _pad[3]
      [16:144]  half[64] affine_scale
      [144:272] half[64] affine_bias

    Total: 272 bytes
    """
    scale = state["scale"].float().cpu()
    bias = state["bias"].float().cpu()

    assert scale.shape == (NUM_EXPERTS,), f"Expected scale[{NUM_EXPERTS}], got {scale.shape}"
    assert bias.shape == (NUM_EXPERTS,), f"Expected bias[{NUM_EXPERTS}], got {bias.shape}"

    scale_fp16 = scale.half().numpy()
    bias_fp16 = bias.half().numpy()

    # Build binary blob matching CalibrationWeights struct layout
    blob = bytearray()
    blob += struct.pack("<I", CALIBRATION_MODE_AFFINE)  # mode
    blob += struct.pack("<III", 0, 0, 0)                # _pad[3]
    blob += scale_fp16.tobytes()                        # affine_scale[64]
    blob += bias_fp16.tobytes()                         # affine_bias[64]

    bin_path = output_dir / "calibration_affine.bin"
    bin_path.write_bytes(bytes(blob))

    info = {
        "mode": "affine",
        "mode_id": CALIBRATION_MODE_AFFINE,
        "n_params": 2 * NUM_EXPERTS,
        "size_bytes": len(blob),
        "bin_path": str(bin_path),
        "scale_range": (float(scale.min()), float(scale.max())),
        "bias_range": (float(bias.min()), float(bias.max())),
    }
    print(f"  Affine weights exported: {len(blob)} bytes")
    print(f"    scale range: [{info['scale_range'][0]:.4f}, {info['scale_range'][1]:.4f}]")
    print(f"    bias range:  [{info['bias_range'][0]:.4f}, {info['bias_range'][1]:.4f}]")

    return info


def export_linear(state: dict, output_dir: Path) -> dict:
    """
    Export linear calibration weights (W[64x64] + bias[64]) to FP16 binary.

    The matrix is exported in ROW_MAJOR FP16. The host-side code
    (bvh_router_bridge.py) converts to INFERENCING_OPTIMAL layout at load
    time using optixCoopVecMatrixConvert().

    Binary format:
      [0:4]       uint32  mode = 2 (CALIBRATION_MODE_LINEAR)
      [4:16]      uint32  _pad[3]
      [16:8208]   half[64*64]  linear_weight (row-major)
      [8208:8336] half[64]     linear_bias

    Total: 8336 bytes
    """
    weight = state["weight"].float().cpu()
    bias = state["bias"].float().cpu()

    assert weight.shape == (NUM_EXPERTS, NUM_EXPERTS), (
        f"Expected weight[{NUM_EXPERTS}x{NUM_EXPERTS}], got {weight.shape}"
    )
    assert bias.shape == (NUM_EXPERTS,), f"Expected bias[{NUM_EXPERTS}], got {bias.shape}"

    weight_fp16 = weight.half().numpy()
    bias_fp16 = bias.half().numpy()

    # Build binary blob
    blob = bytearray()
    blob += struct.pack("<I", CALIBRATION_MODE_LINEAR)  # mode
    blob += struct.pack("<III", 0, 0, 0)                # _pad[3]
    blob += weight_fp16.tobytes()                       # linear_weight[64x64] row-major
    blob += bias_fp16.tobytes()                         # linear_bias[64]

    bin_path = output_dir / "calibration_linear.bin"
    bin_path.write_bytes(bytes(blob))

    info = {
        "mode": "linear",
        "mode_id": CALIBRATION_MODE_LINEAR,
        "n_params": NUM_EXPERTS * NUM_EXPERTS + NUM_EXPERTS,
        "size_bytes": len(blob),
        "bin_path": str(bin_path),
        "weight_norm": float(weight.norm()),
        "bias_range": (float(bias.min()), float(bias.max())),
    }
    print(f"  Linear weights exported: {len(blob)} bytes")
    print(f"    weight Frobenius norm: {info['weight_norm']:.4f}")
    print(f"    bias range: [{info['bias_range'][0]:.4f}, {info['bias_range'][1]:.4f}]")

    return info


def export_c_header(state: dict, mode: str, output_dir: Path) -> None:
    """
    Generate a C header with calibration weights as constexpr arrays.
    Useful for embedding weights in constant memory without file I/O.
    """
    header_path = output_dir / "calibration_weights.h"

    lines = [
        "// Auto-generated by export_calibration.py — DO NOT EDIT",
        "// Calibration weights for OptiX 9.0 cooperative vector in-shader MLP",
        "#pragma once",
        "#include <cuda_fp16.h>",
        "",
        f"#define CALIBRATION_MODE {CALIBRATION_MODE_AFFINE if mode == 'affine' else CALIBRATION_MODE_LINEAR}",
        f"#define CALIBRATION_NUM_EXPERTS {NUM_EXPERTS}",
        "",
    ]

    if mode == "affine":
        scale = state["scale"].float().cpu().half().numpy()
        bias = state["bias"].float().cpu().half().numpy()

        lines.append("// Affine calibration: calibrated[i] = logits[i] * scale[i] + bias[i]")
        lines.append(f"static const unsigned short k_cal_scale[{NUM_EXPERTS}] = {{")
        scale_u16 = scale.view(np.uint16)
        for i in range(0, NUM_EXPERTS, 8):
            chunk = ", ".join(f"0x{v:04x}" for v in scale_u16[i:i+8])
            lines.append(f"    {chunk},")
        lines.append("};")
        lines.append("")
        lines.append(f"static const unsigned short k_cal_bias[{NUM_EXPERTS}] = {{")
        bias_u16 = bias.view(np.uint16)
        for i in range(0, NUM_EXPERTS, 8):
            chunk = ", ".join(f"0x{v:04x}" for v in bias_u16[i:i+8])
            lines.append(f"    {chunk},")
        lines.append("};")

    elif mode == "linear":
        weight = state["weight"].float().cpu().half().numpy()
        bias = state["bias"].float().cpu().half().numpy()

        lines.append("// Linear calibration: calibrated = W @ logits + bias")
        lines.append(f"static const unsigned short k_cal_weight[{NUM_EXPERTS * NUM_EXPERTS}] = {{")
        weight_u16 = weight.flatten().view(np.uint16)
        for i in range(0, len(weight_u16), 8):
            chunk = ", ".join(f"0x{v:04x}" for v in weight_u16[i:i+8])
            lines.append(f"    {chunk},")
        lines.append("};")
        lines.append("")
        lines.append(f"static const unsigned short k_cal_bias[{NUM_EXPERTS}] = {{")
        bias_u16 = bias.view(np.uint16)
        for i in range(0, NUM_EXPERTS, 8):
            chunk = ", ".join(f"0x{v:04x}" for v in bias_u16[i:i+8])
            lines.append(f"    {chunk},")
        lines.append("};")

    lines.append("")

    header_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  C header exported: {header_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export calibration weights for OptiX 9.0 cooperative vectors"
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to calibration checkpoint (.pt file from calibrate_router.py)"
    )
    parser.add_argument(
        "--output", "-o", default="cuda/v5/calibration_weights",
        help="Output directory for binary and header files"
    )
    parser.add_argument(
        "--mode", "-m", choices=["affine", "linear", "auto"], default="auto",
        help="Calibration mode to export (auto-detect from checkpoint)"
    )
    parser.add_argument(
        "--c-header", action="store_true",
        help="Also generate a C header with constexpr weight arrays"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    cal_data = load_calibration_state(args.checkpoint)

    # Auto-detect mode
    mode = args.mode
    if mode == "auto":
        if "mode" in cal_data:
            mode = cal_data["mode"]
        elif "state" in cal_data:
            state = cal_data["state"]
            if "weight" in state:
                mode = "linear"
            elif "scale" in state:
                mode = "affine"
            else:
                raise ValueError("Cannot auto-detect calibration mode from checkpoint keys")
        else:
            raise ValueError("Cannot auto-detect calibration mode")
    print(f"Mode: {mode}")

    # Extract state dict
    state = cal_data.get("state", cal_data)

    if mode == "affine":
        info = export_affine(state, output_dir)
    elif mode == "linear":
        info = export_linear(state, output_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if args.c_header:
        export_c_header(state, mode, output_dir)

    print(f"\nExport complete. Files in: {output_dir}")
    print(f"  Binary: {info['bin_path']} ({info['size_bytes']} bytes)")
    print(f"  Params: {info['n_params']:,}")


if __name__ == "__main__":
    main()
