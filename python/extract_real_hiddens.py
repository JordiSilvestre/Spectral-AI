#!/usr/bin/env python3
"""
extract_real_hiddens.py -- Extract real hidden states from OLMoE for BVH router training.

Runs WikiText-2 through the full OLMoE-1B-7B model and captures the hidden
states at a specific layer BEFORE the MoE block (post-LayerNorm). These are
the actual inputs the gate sees during inference.

This replaces the synthetic Gaussian vectors in GateDistillationDataset,
which don't match the real hidden state distribution and cause PPL blowup.

Usage:
    python extract_real_hiddens.py --model-dir /path/to/olmoe-1b-7b --layer 8

Output:
    data/real_hiddens_layer{N}.pt  (dict with 'hidden_states', 'gate_logits', 'topk_ids')

Copyright (c) 2026 LiquidBit Studio -- Apache 2.0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
# Hook to capture hidden states before MoE block
# ─────────────────────────────────────────────────────────────────

class HiddenStateCapture:
    """Register a forward hook on a decoder layer to capture post-LayerNorm hidden states."""

    def __init__(self):
        self.captured = []
        self._hook = None

    def hook_fn(self, module, args, output):
        """
        OlmoeDecoderLayer.forward():
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(...)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)  <-- we want THIS
            hidden_states = self.mlp(hidden_states)                       <-- gate sees THIS
            ...

        We hook the post_attention_layernorm to capture what the MoE gate actually receives.
        """
        # Output of post_attention_layernorm is a tensor
        if isinstance(output, torch.Tensor):
            self.captured.append(output.detach().cpu().half())

    def register(self, layer_norm_module):
        self._hook = layer_norm_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def extract_hidden_states(
    model_dir: str,
    layer_idx: int = 8,
    max_tokens: int = 200_000,
    batch_seq_len: int = 512,
    device: str = "cuda",
    output_dir: str = "data",
):
    """
    Extract real hidden states from OLMoE model running on WikiText-2.

    1. Loads the full OLMoE model
    2. Hooks post_attention_layernorm at layer_idx
    3. Runs WikiText-2 tokens through the model
    4. Captures hidden states + gate routing decisions
    5. Saves to disk
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print(f"  Extracting Real Hidden States — Layer {layer_idx}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    print(f"\n[1/4] Loading OLMoE-1B-7B from {model_dir}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")

    # ── Load WikiText-2 ──────────────────────────────────────
    print(f"\n[2/4] Loading WikiText-2...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(dataset["text"])
    except ImportError:
        # Fallback: use local tokenized data
        import numpy as np
        npy_path = Path(__file__).parent / "wikitext2_train_tokens.npy"
        if not npy_path.exists():
            npy_path = Path(__file__).parent.parent / "data" / "wikitext2_train_tokens.npy"
        if npy_path.exists():
            tokens = np.load(str(npy_path))[:max_tokens]
            all_token_ids = torch.tensor(tokens, dtype=torch.long)
            print(f"  Loaded {len(all_token_ids):,} tokens from .npy")
        else:
            raise ImportError("pip install datasets  (or place wikitext2_train_tokens.npy)")
        text = None

    if text is not None:
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        all_token_ids = encodings.input_ids.squeeze(0)[:max_tokens]
        print(f"  Tokenized: {len(all_token_ids):,} tokens")

    # ── Find the target layer and hook ──────────────────────
    print(f"\n[3/4] Hooking layer {layer_idx} post_attention_layernorm...")

    # Navigate model structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[layer_idx]
    elif hasattr(model, 'layers'):
        target_layer = model.layers[layer_idx]
    else:
        raise AttributeError("Cannot find layers in model")

    # Hook the post_attention_layernorm (input to MoE block)
    ln = target_layer.post_attention_layernorm
    capture = HiddenStateCapture()
    capture.register(ln)

    # Also get the gate for routing labels
    if hasattr(target_layer.mlp, 'gate'):
        gate = target_layer.mlp.gate
    else:
        raise AttributeError(f"Cannot find gate in layer {layer_idx}")

    # ── Run inference and capture ────────────────────────────
    print(f"\n[4/4] Running inference to capture hidden states...")
    t0 = time.time()

    all_hidden = []
    all_gate_logits = []
    all_topk_ids = []
    n_captured = 0

    with torch.no_grad():
        for start in range(0, len(all_token_ids) - batch_seq_len, batch_seq_len):
            chunk = all_token_ids[start:start + batch_seq_len].unsqueeze(0)
            chunk = chunk.to(device)

            # Clear previous captures
            capture.captured.clear()

            # Forward pass — hook captures post-LN hidden states
            _ = model(chunk)

            if len(capture.captured) == 0:
                print(f"  WARNING: No hidden states captured at offset {start}")
                continue

            # Get the captured hidden states (B=1, S, H)
            h = capture.captured[0]  # (1, seq_len, hidden_size) in fp16 on CPU
            h_flat = h.squeeze(0)  # (seq_len, hidden_size)

            # Get gate routing decisions for these exact hidden states
            h_gate = h_flat.to(device=device, dtype=torch.float16)
            # Run through the gate's linear layer only (not full forward which does topk)
            gate_logits = F.linear(h_gate, gate.weight.to(h_gate.device))
            gate_logits = F.softmax(gate_logits.float(), dim=-1)
            _, topk_ids = gate_logits.topk(8, dim=-1)

            all_hidden.append(h_flat)
            all_gate_logits.append(gate_logits.cpu().half())
            all_topk_ids.append(topk_ids.cpu())

            n_captured += h_flat.shape[0]
            if n_captured % 10000 < batch_seq_len:
                print(f"  Captured {n_captured:,} hidden states...")

    capture.remove()

    # Concatenate all
    hidden_states = torch.cat(all_hidden, dim=0)  # (N, 2048)
    gate_logits = torch.cat(all_gate_logits, dim=0)  # (N, 64)
    topk_ids = torch.cat(all_topk_ids, dim=0)  # (N, 8)

    elapsed = time.time() - t0
    print(f"\n  Captured {hidden_states.shape[0]:,} real hidden states in {elapsed:.1f}s")
    print(f"  Shape: {hidden_states.shape}")
    print(f"  Unique experts used (top-1): {topk_ids[:, 0].unique().numel()}/64")

    # ── Save ──────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"real_hiddens_layer{layer_idx}.pt"
    torch.save({
        'hidden_states': hidden_states,  # (N, 2048) fp16
        'gate_logits': gate_logits,  # (N, 64) fp16
        'topk_ids': topk_ids,  # (N, 8) int64
        'layer_idx': layer_idx,
        'n_samples': hidden_states.shape[0],
    }, str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved to {out_path} ({size_mb:.1f} MB)")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Extract real hidden states from OLMoE")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=200_000,
                        help="Max tokens from WikiText-2 to process")
    parser.add_argument("--batch-seq-len", type=int, default=512,
                        help="Sequence length per batch")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default=None,
                        help="Full output path (overrides --output-dir)")
    args = parser.parse_args()

    # If --output is given, use its directory as output_dir
    if args.output is not None:
        out_path = Path(args.output)
        args.output_dir = str(out_path.parent)
        # Validate the filename matches expected pattern
        expected_name = f"real_hiddens_layer{args.layer}.pt"
        if out_path.name != expected_name:
            print(f"  WARNING: --output filename '{out_path.name}' != expected '{expected_name}'")

    extract_hidden_states(
        model_dir=args.model_dir,
        layer_idx=args.layer,
        max_tokens=args.max_tokens,
        batch_seq_len=args.batch_seq_len,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
