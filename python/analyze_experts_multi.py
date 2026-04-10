#!/usr/bin/env python3
"""
analyze_experts_multi.py — Multi-model expert specialization analysis.

Generalizes analyze_experts.py to work with any HuggingFace MoE model.
Auto-detects architecture (gate path, expert count, top-k) and runs
identical 30-category analysis + deep token-level analysis.

Validated models:
  - allenai/OLMoE-1B-7B-0924  (64 experts, top-8, 16 MoE layers) -- CONFIRMED
  - Qwen/Qwen1.5-MoE-A2.7B   (60 experts, top-4, 24 MoE layers) -- CONFIRMED
  - deepseek-ai/deepseek-moe-16b-base (64 experts, top-6, 27 MoE layers)
    NOTE: DeepSeek requires auto-patching of deprecated transformers APIs
    and returns pre-computed (topk_idx, topk_weight) from its gate.
    Compatibility patches are applied automatically at load time.

Usage:
  python analyze_experts_multi.py \\
      --model-id allenai/OLMoE-1B-7B-0924 \\
      --output results/olmoe/ \\
      --dtype auto --device cuda

  python analyze_experts_multi.py \\
      --model-id Qwen/Qwen1.5-MoE-A2.7B \\
      --output results/qwen_moe/ \\
      --dtype auto

  python analyze_experts_multi.py \\
      --model-id deepseek-ai/deepseek-moe-16b-base \\
      --output results/deepseek_moe/ \\
      --dtype auto
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from expert_analysis_common import CATEGORIES, FUNCTION_WORDS, classify_token


# ═══════════════════════════════════════════════════════════════════
# MoE Architecture Detection
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MoEArchInfo:
    """Detected MoE model architecture."""
    model_id: str
    friendly_name: str
    model_type: str
    n_layers: int
    n_routed_experts: int
    n_shared_experts: int
    top_k: int
    hidden_size: int
    moe_layer_indices: tuple
    gate_attr: str          # 'gate' or 'router'
    mlp_attr: str           # 'mlp' or 'block_sparse_moe'
    gate_output_is_probs: bool  # True if gate returns softmaxed probs


def _get_layers(model):
    """Get the layer list from model, trying common structures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    if hasattr(model, 'layers'):
        return model.layers
    raise ValueError("Cannot detect layer structure")


def _find_gate(mlp_module):
    """Find gate/router submodule within an MLP/MoE block."""
    for attr in ('gate', 'router', 'gate_network'):
        if hasattr(mlp_module, attr):
            return attr
    return None


def _find_mlp(layer):
    """Find the MLP/MoE block within a transformer layer."""
    for attr in ('mlp', 'block_sparse_moe', 'moe'):
        if hasattr(layer, attr):
            return attr
    return None


NAME_MAP = {
    'olmoe': 'OLMoE', 'mixtral': 'Mixtral', 'deepseek_v2': 'DeepSeek-V2',
    'deepseek': 'DeepSeek', 'qwen2_moe': 'Qwen2-MoE', 'qwen_moe': 'Qwen-MoE',
    'llama': 'LLaMA', 'mistral': 'Mistral',
}


def detect_moe_arch(model, config, model_id: str) -> MoEArchInfo:
    """Auto-detect MoE architecture from a HuggingFace model."""
    model_type = getattr(config, 'model_type', '').lower()
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 0))

    layers = _get_layers(model)
    n_layers = len(layers)

    # Find MLP and gate structure from first layer that has one
    mlp_attr = None
    gate_attr = None
    for layer in layers:
        mlp_attr = _find_mlp(layer)
        if mlp_attr is not None:
            mlp_module = getattr(layer, mlp_attr)
            gate_attr = _find_gate(mlp_module)
            if gate_attr is not None:
                break

    if gate_attr is None:
        raise ValueError(
            f"No MoE gate found in model {model_id}. "
            f"Checked attributes: mlp/block_sparse_moe + gate/router"
        )

    # Expert counts from config
    n_routed_experts = getattr(
        config, 'num_local_experts',
        getattr(config, 'num_experts',
        getattr(config, 'n_routed_experts', 0))
    )
    # Fallback: infer from gate weight shape
    if n_routed_experts == 0:
        gate_mod = getattr(getattr(layers[0], mlp_attr), gate_attr)
        gate_w = getattr(gate_mod, 'weight', None)
        if gate_w is not None:
            n_routed_experts = gate_w.shape[0]

    top_k = getattr(
        config, 'num_experts_per_tok',
        getattr(config, 'num_experts_per_token',
        getattr(config, 'top_k', 2))
    )
    n_shared = getattr(config, 'n_shared_experts', 0)

    # Detect which layers are MoE (some models mix dense + MoE)
    moe_indices = []
    for i in range(n_layers):
        mlp_i = getattr(layers[i], mlp_attr, None)
        if mlp_i is not None and _find_gate(mlp_i) is not None:
            moe_indices.append(i)

    friendly_name = NAME_MAP.get(model_type, model_type.upper() or model_id.split('/')[-1])

    return MoEArchInfo(
        model_id=model_id,
        friendly_name=friendly_name,
        model_type=model_type,
        n_layers=n_layers,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_layer_indices=tuple(moe_indices),
        gate_attr=gate_attr,
        mlp_attr=mlp_attr,
        gate_output_is_probs=False,  # detected later
    )


# ═══════════════════════════════════════════════════════════════════
# Hook Registration
# ═══════════════════════════════════════════════════════════════════

def register_gate_hooks(model, arch: MoEArchInfo, gate_outputs: dict,
                        capture_logits: bool = False):
    """Register forward hooks on all MoE gate modules.

    Args:
        gate_outputs: mutable dict to store results, keyed by layer index.
        capture_logits: if True, store (top_k_indices, top_k_values).

    Returns:
        list of hook handles (call .remove() when done).
    """
    layers = _get_layers(model)
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input_tensor, output):
            # --- Detect gate output format ---
            # Case A: Gate returns pre-computed (topk_idx, topk_weight, ...)
            #   e.g. DeepSeek MoEGate → (topk_idx [int], topk_weight [float], aux_loss)
            # Case B: Gate returns raw logits (n_experts) or (logits, ...)
            #   e.g. OLMoE/Qwen → logits tensor of shape (tokens, n_experts)

            if isinstance(output, tuple) and len(output) >= 2:
                first, second = output[0], output[1]
                # Heuristic: if first tensor is integer-like and second is float,
                # this is (indices, weights, ...) format
                if (first.dtype in (torch.int32, torch.int64, torch.long)
                        or (first.is_floating_point()
                            and first.dim() == 2
                            and first.shape[-1] == arch.top_k
                            and (first - first.round()).abs().max() < 0.01)):
                    # Pre-computed top-k format (DeepSeek style)
                    top_idx = first.long().detach().cpu()
                    top_vals = second.detach().cpu().float()
                    if top_idx.dim() == 3:
                        top_idx = top_idx.reshape(-1, top_idx.shape[-1])
                        top_vals = top_vals.reshape(-1, top_vals.shape[-1])
                    if capture_logits:
                        gate_outputs[layer_idx] = (top_idx, top_vals)
                    else:
                        gate_outputs[layer_idx] = top_idx
                    return
                # Otherwise, first element is logits
                logits = first
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Standard logits path (OLMoE / Qwen style)
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            if logits.dim() != 2:
                return
            top_vals, top_idx = torch.topk(logits, arch.top_k, dim=-1)
            if capture_logits:
                gate_outputs[layer_idx] = (
                    top_idx.detach().cpu(),
                    top_vals.detach().cpu().float(),
                )
            else:
                gate_outputs[layer_idx] = top_idx.detach().cpu()
        return hook_fn

    for layer_idx in arch.moe_layer_indices:
        mlp_module = getattr(layers[layer_idx], arch.mlp_attr)
        gate_module = getattr(mlp_module, arch.gate_attr)
        h = gate_module.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    return hooks


def detect_gate_output_type(gate_outputs: dict) -> bool:
    """Check if gate outputs are probabilities (softmaxed) rather than logits.

    Returns True if values look like probabilities (all in [0,1], sum ~1).
    """
    for layer_idx, data in gate_outputs.items():
        if isinstance(data, tuple):
            vals = data[1]  # top_k_values
        else:
            continue
        if vals.numel() == 0:
            continue
        # Check a few rows
        sample = vals[:min(10, vals.shape[0])]
        if (sample >= -0.01).all() and (sample <= 1.01).all():
            row_sums = sample.sum(dim=-1)
            if (row_sums > 0.5).all() and (row_sums < 1.5).all():
                return True
        return False
    return False


# ═══════════════════════════════════════════════════════════════════
# DeepSeek Compatibility Patches
# ═══════════════════════════════════════════════════════════════════

def _patch_deepseek_rope_scaling(model_id: str) -> None:
    """Patch cached DeepSeek modeling file for rope_scaling compatibility.

    Newer transformers may convert rope_scaling=null to a dict without
    the 'type' key, causing KeyError in _init_rope(). This patches both
    the source model dir and the HF modules cache.
    """
    import glob
    import os
    import pathlib

    search_paths = [
        os.path.join(model_id, "modeling_deepseek.py"),
        *glob.glob(os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/*deepseek*/modeling_deepseek.py"
        )),
    ]

    old_pattern = 'scaling_type = self.config.rope_scaling["type"]'
    new_block = 'scaling_type = _rope_cfg["type"]'
    fx_import = 'from transformers.utils import (\n'
    fx_shim = 'is_torch_fx_available = lambda: True\n'

    for fpath in search_paths:
        if not os.path.isfile(fpath):
            continue
        text = pathlib.Path(fpath).read_text(encoding="utf-8")
        patched = False
        # Patch 1: is_torch_fx_available shim
        if 'is_torch_fx_available' in text and fx_shim not in text:
            if 'from transformers.utils import' in text:
                # Insert shim after the utils import block
                idx = text.find('from transformers.utils import')
                end = text.find('\n)', idx)
                if end > 0:
                    text = text[:end+2] + fx_shim + text[end+2:]
                    patched = True
        # Patch 2: rope_scaling compatibility
        if old_pattern in text and '_rope_cfg = self.config.rope_scaling' not in text:
            text = text.replace(
                '        if self.config.rope_scaling is None:\n',
                '        _rope_cfg = self.config.rope_scaling\n'
                '        if _rope_cfg is None or "type" not in _rope_cfg:\n',
            )
            text = text.replace(old_pattern, new_block)
            text = text.replace(
                'scaling_factor = self.config.rope_scaling["factor"]',
                'scaling_factor = _rope_cfg["factor"]',
            )
            patched = True
        # Patch 3: DynamicCache.from_legacy_cache removed in newer transformers
        legacy_from = 'past_key_values = DynamicCache.from_legacy_cache(past_key_values)'
        if legacy_from in text and 'hasattr(DynamicCache, "from_legacy_cache")' not in text:
            text = text.replace(
                legacy_from,
                'if hasattr(DynamicCache, "from_legacy_cache"):\n'
                '                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)\n'
                '                else:\n'
                '                    past_key_values = DynamicCache()',
            )
            patched = True
        legacy_to = 'next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache'
        if legacy_to in text and 'hasattr(next_decoder_cache, "to_legacy_cache")' not in text:
            text = text.replace(
                legacy_to,
                '(next_decoder_cache.to_legacy_cache() if hasattr(next_decoder_cache, "to_legacy_cache") else next_decoder_cache) if use_legacy_cache else next_decoder_cache',
            )
            patched = True
        # Patch 4: get_usable_length(seq_len, layer_idx) → get_seq_length(layer_idx)
        import re as _re
        if '.get_usable_length(' in text:
            # In attention: get_usable_length(kv_seq_len, self.layer_idx) → get_seq_length(self.layer_idx)
            text = _re.sub(
                r'\.get_usable_length\(\w+,\s*(self\.layer_idx)\)',
                r'.get_seq_length(\1)',
                text,
            )
            # In model forward: get_usable_length(seq_length) → get_seq_length()
            text = _re.sub(
                r'\.get_usable_length\(seq_length\)',
                '.get_seq_length()',
                text,
            )
            patched = True
        if patched:
            pathlib.Path(fpath).write_text(text, encoding="utf-8")
            print(f"  [PATCH] Fixed DeepSeek compat in {fpath}")


# ═══════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════

def load_model(model_id: str, dtype: str, device: str):
    """Load a HuggingFace model with appropriate quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Determine torch_dtype and quantization
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if dtype == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif dtype == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif dtype == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif dtype == "auto":
        # Heuristic based on model name to choose quantization
        model_name_lower = model_id.lower()
        if any(tag in model_name_lower for tag in ['16b', '22b', '34b', '46b', '70b']):
            print(f"  Auto-selecting INT4 quantization for large model")
            from transformers import BitsAndBytesConfig
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
        elif any(tag in model_name_lower for tag in ['7b', '8b', '13b', '14b']):
            print(f"  Auto-selecting INT8 quantization for medium model")
            from transformers import BitsAndBytesConfig
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            print(f"  Auto-selecting FP16 for small model")
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float16

    # Patch DeepSeek cached modeling file for rope_scaling compat
    if "deepseek" in model_id.lower():
        _patch_deepseek_rope_scaling(model_id)

    print(f"  Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Category Activation Catalog
# ═══════════════════════════════════════════════════════════════════

def analyze_category_activations(model, tokenizer, arch: MoEArchInfo,
                                 device: str, max_tokens: int = 128) -> dict:
    """Run all 30 categories through the model and record expert activations."""
    n_experts = arch.n_routed_experts
    moe_layers = arch.moe_layer_indices

    activations = {
        layer: {exp: defaultdict(int) for exp in range(n_experts)}
        for layer in moe_layers
    }
    category_tokens = defaultdict(int)
    gate_outputs = {}

    hooks = register_gate_hooks(model, arch, gate_outputs, capture_logits=False)

    print(f"\n  [Phase 1] Category activation analysis ({len(CATEGORIES)} categories)...")

    for cat_name, texts in CATEGORIES.items():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                max_length=max_tokens, truncation=True,
            ).to(device)

            n_tok = inputs["input_ids"].shape[1]
            category_tokens[cat_name] += n_tok

            with torch.no_grad():
                gate_outputs.clear()
                model(**inputs)

                for layer_idx, indices in gate_outputs.items():
                    for expert_id in indices.reshape(-1).tolist():
                        activations[layer_idx][expert_id][cat_name] += 1

    for h in hooks:
        h.remove()

    # Build catalog
    global_activations = {exp: defaultdict(float) for exp in range(n_experts)}
    for layer_idx in moe_layers:
        for exp_id in range(n_experts):
            for cat, count in activations[layer_idx][exp_id].items():
                if category_tokens[cat] > 0:
                    global_activations[exp_id][cat] += count / category_tokens[cat]

    catalog = {}
    for exp_id in range(n_experts):
        cats = global_activations[exp_id]
        total = sum(cats.values())
        if total == 0:
            catalog[exp_id] = {"primary": "unknown", "primary_pct": 0, "distribution": {}}
            continue
        dist = {cat: val / total * 100 for cat, val in cats.items()}
        sorted_cats = sorted(dist.items(), key=lambda x: -x[1])
        catalog[exp_id] = {
            "primary": sorted_cats[0][0],
            "primary_pct": round(sorted_cats[0][1], 2),
            "distribution": {cat: round(pct, 1) for cat, pct in sorted_cats},
        }

    return {
        "catalog": catalog,
        "category_tokens": dict(category_tokens),
        "activations": activations,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Deep Token-Level Analysis
# ═══════════════════════════════════════════════════════════════════

def deep_token_analysis(model, tokenizer, arch: MoEArchInfo,
                        device: str, max_tokens: int = 128) -> dict:
    """Deep per-token analysis: token types, co-activation, selectivity, clusters."""
    n_experts = arch.n_routed_experts
    moe_layers = arch.moe_layer_indices
    top_k = arch.top_k

    # Storage
    token_type_counts = {
        layer: {exp: defaultdict(int) for exp in range(n_experts)}
        for layer in moe_layers
    }
    co_activation = {
        layer: np.zeros((n_experts, n_experts), dtype=np.int32)
        for layer in moe_layers
    }
    expert_logits = {
        layer: {exp: [] for exp in range(n_experts)}
        for layer in moe_layers
    }

    gate_data = {}
    hooks = register_gate_hooks(model, arch, gate_data, capture_logits=True)

    print(f"\n  [Phase 2] Deep token-level analysis ({len(moe_layers)} MoE layers)...")

    all_texts = [(cat, text) for cat, texts in CATEGORIES.items() for text in texts]

    # Check gate output type on first pass
    gate_is_probs = False

    for text_idx, (cat_name, text) in enumerate(all_texts):
        if text_idx % 60 == 0:
            print(f"    Processing text {text_idx + 1}/{len(all_texts)}...")

        inputs = tokenizer(
            text, return_tensors="pt",
            max_length=max_tokens, truncation=True,
        ).to(device)

        token_ids = inputs["input_ids"][0].cpu().tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
        token_types = [classify_token(ts) for ts in token_strs]

        with torch.no_grad():
            gate_data.clear()
            model(**inputs)

            # Detect gate output type once
            if text_idx == 0:
                gate_is_probs = detect_gate_output_type(gate_data)
                if gate_is_probs:
                    print("    NOTE: Gate outputs are probabilities, applying log() for selectivity")

            for layer_idx in moe_layers:
                if layer_idx not in gate_data:
                    continue
                indices, top_vals = gate_data[layer_idx]

                # If gate returns probs, convert to log-space for selectivity
                logit_vals = torch.log(top_vals + 1e-8) if gate_is_probs else top_vals

                for tok_pos in range(len(token_ids)):
                    tok_type = token_types[tok_pos]
                    experts_active = indices[tok_pos].tolist()

                    for rank in range(top_k):
                        exp_id = experts_active[rank]
                        token_type_counts[layer_idx][exp_id][tok_type] += 1
                        expert_logits[layer_idx][exp_id].append(
                            logit_vals[tok_pos, rank].item()
                        )

                    # Co-activation pairs
                    for ii in range(len(experts_active)):
                        for jj in range(ii + 1, len(experts_active)):
                            e1, e2 = experts_active[ii], experts_active[jj]
                            co_activation[layer_idx][e1][e2] += 1
                            co_activation[layer_idx][e2][e1] += 1

    for h in hooks:
        h.remove()

    # ── Clustering ──
    print(f"    Computing co-activation clusters...")
    all_layer_clusters = {}
    for layer_idx in moe_layers:
        coact = co_activation[layer_idx].astype(float)
        row_sums = coact.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        coact_norm = coact / row_sums

        used = set()
        clusters = []
        for seed in range(n_experts):
            if seed in used:
                continue
            cluster = [seed]
            used.add(seed)
            affinities = coact_norm[seed].copy()
            for _ in range(n_experts - 1):
                affinities[list(used)] = -1
                best = np.argmax(affinities)
                if affinities[best] <= 0:
                    break
                cluster.append(int(best))
                used.add(int(best))
            clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        all_layer_clusters[layer_idx] = clusters

    # ── Selectivity ──
    print(f"    Computing selectivity per layer...")
    all_selectivities = {}
    for layer_idx in moe_layers:
        sels = []
        for exp_id in range(n_experts):
            logits_list = expert_logits[layer_idx][exp_id]
            if not logits_list:
                sels.append(0.0)
                continue
            arr = np.array(logits_list)
            mean_l = arr.mean()
            std_l = arr.std()
            sel = std_l / abs(mean_l) if abs(mean_l) > 1e-6 else 0
            sels.append(float(sel))
        all_selectivities[layer_idx] = sels

    # ── Cluster stability ──
    print(f"    Computing cluster stability...")

    def clusters_to_membership(clusters_list):
        membership = {}
        for cidx, cluster in enumerate(clusters_list):
            for exp_id in cluster:
                membership[exp_id] = cidx
        return membership

    # Compare adjacent pairs and distant pairs
    stability_pairs = []
    sorted_moe = sorted(moe_layers)
    for i in range(len(sorted_moe) - 1):
        stability_pairs.append((sorted_moe[i], sorted_moe[i + 1]))
    # Also add some distant pairs (quarter, half, full span)
    if len(sorted_moe) >= 4:
        q = len(sorted_moe) // 4
        stability_pairs.append((sorted_moe[0], sorted_moe[q]))
        stability_pairs.append((sorted_moe[0], sorted_moe[len(sorted_moe) // 2]))
        stability_pairs.append((sorted_moe[0], sorted_moe[-1]))

    cluster_stability = {}
    for la, lb in stability_pairs:
        if la not in all_layer_clusters or lb not in all_layer_clusters:
            continue
        mem_a = clusters_to_membership(all_layer_clusters[la])
        mem_b = clusters_to_membership(all_layer_clusters[lb])
        same = 0
        total_pairs = 0
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                same_a = (mem_a.get(i, -1) == mem_a.get(j, -2))
                same_b = (mem_b.get(i, -1) == mem_b.get(j, -2))
                if same_a and same_b:
                    same += 1
                total_pairs += 1
        pct = same * 100 / total_pairs if total_pairs > 0 else 0
        cluster_stability[f"L{la}-L{lb}"] = round(pct, 2)

    # ── Assemble results ──
    deep_results = {
        "n_moe_layers": len(moe_layers),
        "n_routed_experts": n_experts,
        "moe_layer_indices": list(moe_layers),
        "gate_output_is_probs": gate_is_probs,
        "per_layer_clusters": {
            str(layer): [
                {"id": cidx, "experts": cluster, "size": len(cluster)}
                for cidx, cluster in enumerate(clusters)
            ]
            for layer, clusters in all_layer_clusters.items()
        },
        "per_layer_selectivity": {
            str(layer): {str(exp): round(s, 4) for exp, s in enumerate(sels)}
            for layer, sels in all_selectivities.items()
        },
        "per_layer_token_types": {
            str(layer): {
                str(exp): dict(token_type_counts[layer][exp])
                for exp in range(n_experts)
            }
            for layer in moe_layers
        },
        "cluster_stability": cluster_stability,
    }

    return deep_results


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_analysis(model_id: str, output_dir: str, dtype: str = "auto",
                 device: str = "cuda", max_tokens: int = 128) -> None:
    """Run full expert specialization analysis on a single model."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  EXPERT SPECIALIZATION ANALYSIS")
    print(f"  Model: {model_id}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    # Load model
    model, tokenizer = load_model(model_id, dtype, device)
    arch = detect_moe_arch(model, model.config, model_id)

    print(f"\n  Architecture detected:")
    print(f"    Name:           {arch.friendly_name}")
    print(f"    Type:           {arch.model_type}")
    print(f"    Total layers:   {arch.n_layers}")
    print(f"    MoE layers:     {len(arch.moe_layer_indices)}")
    print(f"    Routed experts: {arch.n_routed_experts}")
    print(f"    Shared experts: {arch.n_shared_experts}")
    print(f"    Top-k:          {arch.top_k}")
    print(f"    Hidden size:    {arch.hidden_size}")
    print(f"    Gate attr:      {arch.mlp_attr}.{arch.gate_attr}")

    # Save model info
    model_info = {
        "model_id": model_id,
        "friendly_name": arch.friendly_name,
        "model_type": arch.model_type,
        "n_total_layers": arch.n_layers,
        "n_moe_layers": len(arch.moe_layer_indices),
        "moe_layer_indices": list(arch.moe_layer_indices),
        "n_routed_experts": arch.n_routed_experts,
        "n_shared_experts": arch.n_shared_experts,
        "top_k": arch.top_k,
        "hidden_size": arch.hidden_size,
        "dtype": dtype,
    }
    with open(out / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # Phase 1: Category catalog
    cat_result = analyze_category_activations(
        model, tokenizer, arch, device, max_tokens
    )
    catalog_out = {
        "n_experts": arch.n_routed_experts,
        "n_moe_layers": len(arch.moe_layer_indices),
        "categories": list(CATEGORIES.keys()),
        "category_tokens": cat_result["category_tokens"],
        "catalog": {str(k): v for k, v in cat_result["catalog"].items()},
    }
    with open(out / "expert_catalog.json", "w") as f:
        json.dump(catalog_out, f, indent=2)
    print(f"\n  Catalog saved to {out / 'expert_catalog.json'}")

    # Print top-10 most specialized
    catalog = cat_result["catalog"]
    sorted_experts = sorted(
        catalog.items(),
        key=lambda x: x[1].get("primary_pct", 0),
        reverse=True,
    )
    print(f"\n  Top-10 most topic-specialized experts:")
    print(f"  {'Expert':>7} {'Primary':>15} {'Pct':>6}")
    print(f"  {'-'*7} {'-'*15} {'-'*6}")
    for exp_id, info in sorted_experts[:10]:
        print(f"  E{exp_id:>4d}  {info['primary']:>15s} {info['primary_pct']:>5.1f}%")

    # Phase 2: Deep analysis
    deep = deep_token_analysis(model, tokenizer, arch, device, max_tokens)
    out.mkdir(parents=True, exist_ok=True)  # ensure dir still exists (WSL/NTFS)
    with open(out / "expert_deep_analysis.json", "w") as f:
        json.dump(deep, f, indent=2)
    print(f"\n  Deep analysis saved to {out / 'expert_deep_analysis.json'}")

    # Print selectivity summary (U-shape check)
    moe_layers = sorted(arch.moe_layer_indices)
    print(f"\n  Selectivity per MoE layer (U-shape indicator):")
    print(f"  {'Layer':>6} {'Mean Sel':>9}")
    print(f"  {'-'*6} {'-'*9}")
    for layer_idx in moe_layers:
        sels = deep["per_layer_selectivity"].get(str(layer_idx), {})
        if sels:
            mean_sel = np.mean(list(sels.values()))
            print(f"  L{layer_idx:>3d}  {mean_sel:>8.3f}")

    # Print cluster summary
    print(f"\n  Clusters per MoE layer:")
    print(f"  {'Layer':>6} {'N clusters':>10} {'Largest':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*8}")
    for layer_idx in moe_layers:
        clusters = deep["per_layer_clusters"].get(str(layer_idx), [])
        n_cl = len(clusters)
        largest = clusters[0]["size"] if clusters else 0
        print(f"  L{layer_idx:>3d}  {n_cl:>10d} {largest:>8d}")

    # Print stability
    print(f"\n  Cluster stability:")
    for pair, pct in deep.get("cluster_stability", {}).items():
        print(f"    {pair}: {pct:.1f}%")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE: {model_id}")
    print(f"  Results in: {output_dir}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model expert specialization analysis"
    )
    parser.add_argument(
        "--model-id", type=str, required=True,
        help="HuggingFace model ID (e.g., allenai/OLMoE-1B-7B-0924)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for results (e.g., results/olmoe/)"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "fp16", "bf16", "int8", "int4"],
        help="Quantization strategy (default: auto)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max tokens per text sample (default: 128)"
    )
    args = parser.parse_args()

    run_analysis(
        model_id=args.model_id,
        output_dir=args.output,
        dtype=args.dtype,
        device=args.device,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
