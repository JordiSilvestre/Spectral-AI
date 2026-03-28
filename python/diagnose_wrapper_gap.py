#!/usr/bin/env python3
"""
diagnose_wrapper_gap.py — Pinpoint the 7.67 vs 6.11 PPL gap

Runs three diagnostics:
1. Check hooks on the original gate module (accelerate, forward_hooks, etc.)
2. Compare original gate vs IdentityGateWrapper output numerically
3. Measure PPL twice without replacement (stability check)

Usage:
    python diagnose_wrapper_gap.py --model-dir /path/to/olmoe-1b-7b
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from olmoe_e2e_eval import (
    IdentityGateWrapper,
    evaluate_ppl,
    find_gate_module,
    load_olmoe_model,
)


def diagnose_hooks(model, layer_idx: int):
    """Print ALL hooks registered on the gate module."""
    print("\n" + "=" * 70)
    print(f"  DIAGNOSTIC 1: Hooks on layer {layer_idx} gate")
    print("=" * 70)

    mlp, gate_attr = find_gate_module(model, layer_idx)
    gate = getattr(mlp, gate_attr)

    print(f"  Gate class: {type(gate).__name__}")
    print(f"  Gate module path: {gate_attr}")

    # Standard PyTorch hooks
    print(f"\n  _forward_hooks:     {dict(gate._forward_hooks)}")
    print(f"  _forward_pre_hooks: {dict(gate._forward_pre_hooks)}")
    print(f"  _backward_hooks:    {dict(gate._backward_hooks)}")

    # Accelerate hooks
    if hasattr(gate, '_hf_hook'):
        hook = gate._hf_hook
        print(f"\n  ACCELERATE _hf_hook found: {type(hook).__name__}")
        print(f"    execution_device:  {getattr(hook, 'execution_device', 'N/A')}")
        print(f"    offload:           {getattr(hook, 'offload', 'N/A')}")
        print(f"    io_same_device:    {getattr(hook, 'io_same_device', 'N/A')}")
        print(f"    place_submodules:  {getattr(hook, 'place_submodules', 'N/A')}")
        if hasattr(hook, 'pre_forward'):
            print(f"    pre_forward:       {hook.pre_forward}")
        if hasattr(hook, 'post_forward'):
            print(f"    post_forward:      {hook.post_forward}")
    else:
        print(f"\n  No accelerate _hf_hook found")

    # Check parent MLP hooks too
    print(f"\n  Parent MLP class: {type(mlp).__name__}")
    print(f"  MLP _forward_hooks:     {dict(mlp._forward_hooks)}")
    print(f"  MLP _forward_pre_hooks: {dict(mlp._forward_pre_hooks)}")
    if hasattr(mlp, '_hf_hook'):
        hook = mlp._hf_hook
        print(f"  MLP ACCELERATE _hf_hook: {type(hook).__name__}")
        print(f"    execution_device: {getattr(hook, 'execution_device', 'N/A')}")
        print(f"    io_same_device:   {getattr(hook, 'io_same_device', 'N/A')}")
    else:
        print(f"  MLP: No accelerate _hf_hook")

    # Check the decoder layer too
    layer = model.model.layers[layer_idx]
    print(f"\n  Decoder layer class: {type(layer).__name__}")
    if hasattr(layer, '_hf_hook'):
        hook = layer._hf_hook
        print(f"  Layer ACCELERATE _hf_hook: {type(hook).__name__}")
        print(f"    execution_device: {getattr(hook, 'execution_device', 'N/A')}")
        print(f"    io_same_device:   {getattr(hook, 'io_same_device', 'N/A')}")
    else:
        print(f"  Layer: No accelerate _hf_hook")

    return gate


def diagnose_numerical(model, layer_idx: int, tokenizer, device: str):
    """Compare original gate vs IdentityGateWrapper output."""
    print("\n" + "=" * 70)
    print(f"  DIAGNOSTIC 2: Numerical comparison (gate vs IdentityGateWrapper)")
    print("=" * 70)

    mlp, gate_attr = find_gate_module(model, layer_idx)
    original_gate = getattr(mlp, gate_attr)

    # Capture original gate output for one batch
    captured_original = {}

    def capture_hook(module, args, output):
        captured_original['input'] = args[0].detach().clone()
        captured_original['output'] = tuple(o.detach().clone() for o in output)

    handle = original_gate.register_forward_hook(capture_hook)

    # Run one forward pass
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs, output_router_logits=False)

    handle.remove()

    if not captured_original:
        print("  ERROR: Hook did not fire! Gate forward was not called.")
        return

    print(f"  Captured original gate input:  shape={captured_original['input'].shape}, "
          f"dtype={captured_original['input'].dtype}, "
          f"device={captured_original['input'].device}")
    print(f"  Captured original gate output[0] (logits): shape={captured_original['output'][0].shape}")
    print(f"  Captured original gate output[1] (scores): shape={captured_original['output'][1].shape}")
    print(f"  Captured original gate output[2] (indices): shape={captured_original['output'][2].shape}")

    # Now replace with IdentityGateWrapper and capture its output
    orig_weight = original_gate.weight.data.clone()
    identity_wrapper = IdentityGateWrapper(orig_weight).to(orig_weight.device)

    # Run the SAME input through the wrapper manually
    with torch.no_grad():
        wrapper_output = identity_wrapper(captured_original['input'])

    print(f"\n  Comparing outputs:")
    names = ['router_logits', 'router_scores', 'router_indices']
    all_match = True
    for i, name in enumerate(names):
        orig = captured_original['output'][i]
        wrap = wrapper_output[i]
        if orig.dtype != wrap.dtype:
            print(f"  {name}: DTYPE MISMATCH — orig={orig.dtype}, wrapper={wrap.dtype}")
        if orig.shape != wrap.shape:
            print(f"  {name}: SHAPE MISMATCH — orig={orig.shape}, wrapper={wrap.shape}")
            all_match = False
            continue

        if orig.dtype in (torch.int32, torch.int64, torch.long):
            match = torch.equal(orig, wrap)
            n_diff = (orig != wrap).sum().item()
            print(f"  {name}: {'EXACT MATCH' if match else f'MISMATCH ({n_diff} elements differ)'}")
            if not match:
                all_match = False
        else:
            max_diff = (orig.float() - wrap.float()).abs().max().item()
            mean_diff = (orig.float() - wrap.float()).abs().mean().item()
            cosine = F.cosine_similarity(orig.float().flatten(), wrap.float().flatten(), dim=0).item()
            print(f"  {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, cosine={cosine:.8f}")
            if max_diff > 1e-5:
                all_match = False
                print(f"          *** SIGNIFICANT DIFFERENCE ***")

    if all_match:
        print(f"\n  ✓ Gate outputs are NUMERICALLY IDENTICAL")
        print(f"    If PPL still differs, the problem is NOT in the gate computation")
        print(f"    → Must be accelerate hooks, dispatch, or dtype handling")
    else:
        print(f"\n  ✗ Gate outputs DIFFER — this explains the PPL gap")

    # Now test: replace the gate in the model and run the SAME input
    print(f"\n  Testing in-model replacement...")
    setattr(mlp, gate_attr, identity_wrapper)

    captured_replaced = {}

    def capture_hook2(module, args, output):
        captured_replaced['input'] = args[0].detach().clone()
        captured_replaced['output'] = tuple(o.detach().clone() for o in output)

    handle2 = identity_wrapper.register_forward_hook(capture_hook2)

    with torch.no_grad():
        _ = model(**inputs, output_router_logits=False)

    handle2.remove()

    # Restore original
    setattr(mlp, gate_attr, original_gate)

    if captured_replaced:
        print(f"  Wrapper received input:  shape={captured_replaced['input'].shape}, "
              f"dtype={captured_replaced['input'].dtype}")

        # Compare input received by original vs wrapper
        input_diff = (captured_original['input'].float() - captured_replaced['input'].float()).abs()
        print(f"  Input to gate — same? max_diff={input_diff.max().item():.2e}")
        if input_diff.max().item() > 1e-5:
            print(f"  *** GATE RECEIVES DIFFERENT INPUT after replacement! ***")
            print(f"  This means accelerate hooks or device dispatch changed the input")

        # Compare output
        for i, name in enumerate(names):
            orig = captured_original['output'][i]
            repl = captured_replaced['output'][i]
            if orig.dtype in (torch.int32, torch.int64, torch.long):
                n_diff = (orig != repl).sum().item()
                print(f"  In-model {name}: {n_diff} elements differ")
            else:
                max_diff = (orig.float() - repl.float()).abs().max().item()
                print(f"  In-model {name}: max_diff={max_diff:.2e}")
    else:
        print(f"  ERROR: Wrapper hook did not fire!")


def diagnose_stability(model, tokenizer, device: str, max_tokens: int):
    """Run PPL twice without any modification to check stability."""
    print("\n" + "=" * 70)
    print(f"  DIAGNOSTIC 3: PPL stability (two runs, no modification)")
    print("=" * 70)

    ppl1 = evaluate_ppl(model, tokenizer, max_length=2048, stride=512,
                        device=device, max_tokens=max_tokens)
    print(f"  Run 1 PPL = {ppl1:.4f}")

    ppl2 = evaluate_ppl(model, tokenizer, max_length=2048, stride=512,
                        device=device, max_tokens=max_tokens)
    print(f"  Run 2 PPL = {ppl2:.4f}")

    delta = abs(ppl2 - ppl1)
    print(f"  Delta = {delta:.4f} ({'STABLE' if delta < 0.01 else 'UNSTABLE'})")

    return ppl1, ppl2


def diagnose_hook_preservation(model, layer_idx: int, tokenizer, device: str, max_tokens: int):
    """
    The KEY test: replace gate, copy ALL hooks from original, measure PPL.
    """
    print("\n" + "=" * 70)
    print(f"  DIAGNOSTIC 4: IdentityGateWrapper WITH hook preservation")
    print("=" * 70)

    mlp, gate_attr = find_gate_module(model, layer_idx)
    original_gate = getattr(mlp, gate_attr)

    # Copy the original weight
    orig_weight = original_gate.weight.data.clone()
    identity_wrapper = IdentityGateWrapper(orig_weight).to(orig_weight.device)

    # Copy ALL hooks from original gate to wrapper
    print(f"  Copying hooks from original gate to wrapper...")

    # Copy _forward_hooks
    for key, hook in original_gate._forward_hooks.items():
        identity_wrapper.register_forward_hook(hook)
        print(f"    Copied forward_hook: {key}")

    # Copy _forward_pre_hooks
    for key, hook in original_gate._forward_pre_hooks.items():
        identity_wrapper.register_forward_pre_hook(hook)
        print(f"    Copied forward_pre_hook: {key}")

    # Copy accelerate hook if present
    if hasattr(original_gate, '_hf_hook'):
        hf_hook = original_gate._hf_hook
        print(f"    Found accelerate hook: {type(hf_hook).__name__}")
        # Use accelerate's add_hook_to_module
        try:
            from accelerate.hooks import add_hook_to_module
            add_hook_to_module(identity_wrapper, hf_hook)
            print(f"    Installed accelerate hook on wrapper")
        except ImportError:
            print(f"    WARNING: accelerate not available, cannot copy hook")

    # Replace gate
    setattr(mlp, gate_attr, identity_wrapper)

    # Measure PPL
    ppl = evaluate_ppl(model, tokenizer, max_length=2048, stride=512,
                       device=device, max_tokens=max_tokens)
    print(f"\n  PPL with hook-preserved IdentityGateWrapper = {ppl:.4f}")

    # Restore
    setattr(mlp, gate_attr, original_gate)
    print(f"  Original gate restored")

    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tokens", type=int, default=20000,
                        help="Fewer tokens for faster diagnostics")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip the slow PPL stability test")
    args = parser.parse_args()

    print("=" * 70)
    print("  LiquidBit — Wrapper Gap Diagnostic")
    print("  Investigating 7.67 vs 6.11 PPL gap")
    print("=" * 70)

    model, tokenizer = load_olmoe_model(args.model_dir, args.device)

    # Diagnostic 1: hooks
    gate = diagnose_hooks(model, args.layer)

    # Diagnostic 2: numerical comparison
    diagnose_numerical(model, args.layer, tokenizer, args.device)

    if not args.skip_stability:
        # Diagnostic 3: PPL stability
        ppl1, ppl2 = diagnose_stability(model, tokenizer, args.device, args.max_tokens)

        # Diagnostic 4: hook-preserved wrapper
        ppl_hooked = diagnose_hook_preservation(
            model, args.layer, tokenizer, args.device, args.max_tokens
        )

        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Baseline PPL (run 1):          {ppl1:.4f}")
        print(f"  Baseline PPL (run 2):          {ppl2:.4f}")
        print(f"  Identity + hooks PPL:          {ppl_hooked:.4f}")
        delta = abs(ppl_hooked - ppl1)
        print(f"  Gap (hooks vs baseline):       {delta:.4f}")
        if delta < 0.05:
            print(f"\n  ✓ Hook preservation FIXES the gap!")
        else:
            print(f"\n  ✗ Hook preservation does NOT fix the gap")
            print(f"    The issue is deeper — check dtype, device dispatch, or model state")


if __name__ == "__main__":
    main()
