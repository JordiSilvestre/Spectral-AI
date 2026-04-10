# Reproduction Guide: Expert Specialization in MoE Language Models

This document provides step-by-step instructions to reproduce all results in the paper
"Expert Specialization in Mixture-of-Experts Language Models: Syntactic Roles Dominate Semantic Topics."

---

## Prerequisites

### Hardware

- **GPU:** NVIDIA GPU with 16+ GB VRAM (tested on RTX 5070 Ti 16 GB)
- **RAM:** 32+ GB system memory (models are split across GPU+CPU via `accelerate`)
- **Disk:** ~50 GB free for model downloads

### Software

- Python 3.12+ (tested on 3.12.3)
- CUDA 12.x+ (tested on CUDA 13.2)
- Linux or WSL2 recommended

### Setup

```bash
cd /path/to/spectral-ai
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` contents:
```
torch>=2.1
numpy
transformers>=4.40
datasets
accelerate>=0.28
safetensors
scikit-learn
tqdm
matplotlib
bitsandbytes
```

---

## Step 1: Download Models

Models can be stored anywhere. We use `models/` as convention.

```bash
# OLMoE-1B-7B (~14 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('allenai/OLMoE-1B-7B-0924', local_dir='models/olmoe-1b-7b')
"

# Qwen1.5-MoE-A2.7B (~28 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen1.5-MoE-A2.7B', local_dir='models/qwen-moe')
"
```

---

## Step 2: Run Per-Model Analysis

Each run executes two phases:
- **Phase 1:** Category activation catalog (30 categories, ~4000 tokens)
- **Phase 2:** Deep analysis (selectivity, clustering, stability)

```bash
# OLMoE (~10 min on RTX 5070 Ti)
python python/analyze_experts_multi.py \
    --model-id models/olmoe-1b-7b \
    --output results/olmoe/ \
    --dtype fp16

# Qwen1.5-MoE (~25 min)
python python/analyze_experts_multi.py \
    --model-id models/qwen-moe \
    --output results/qwen_moe/ \
    --dtype fp16
```

### Output files per model

| File | Content |
|---|---|
| `model_info.json` | Architecture metadata (experts, layers, top-k, etc.) |
| `expert_catalog.json` | Phase 1: per-expert activation counts across 30 categories |
| `expert_deep_analysis.json` | Phase 2: selectivity, clusters, stability, token types |

### dtype options

| Flag | Memory | Speed | When to use |
|---|---|---|---|
| `fp16` | ~2x model size | Fast | Default, works with CPU offloading |
| `bf16` | ~2x model size | Fast | If model was trained in BF16 |
| `int8` | ~1x model size | Slower | Reduce VRAM (needs bitsandbytes) |
| `int4` | ~0.5x model size | Slowest | Very large models (needs bitsandbytes) |
| `auto` | Varies | Varies | Auto-selects based on model size |

---

## Step 3: Cross-Model Comparison

CPU-only script that reads pre-computed JSON results:

```bash
python python/compare_expert_findings.py \
    --model-dirs results/olmoe/ results/qwen_moe/ \
    --output results/comparison.json
```

This checks 5 findings across all models:
1. Syntactic > semantic specialization
2. Topic specialization is weak (max < 3x uniform)
3. U-shaped selectivity curve
4. Co-activation cluster count in [3,6] range
5. Cluster stability decreases with layer distance

Output: `results/comparison.json` with per-model results and generalization summary.

---

## Step 4: Verify Specific Findings

### Finding 1: Topic specialization (Table 1 in paper)

From `results/<model>/expert_catalog.json`:
```python
import json
with open("results/olmoe/expert_catalog.json") as f:
    cat = json.load(f)

# Most specialized expert
for eid, info in sorted(cat["catalog"].items(), key=lambda x: -x[1]["primary_pct"])[:10]:
    print(f"E{eid:>3}: {info['primary']:>15}  {info['primary_pct']:.1f}%")
```

Expected: max ~6.8% for OLMoE, ~5.4% for Qwen (vs. 3.3% uniform baseline).

### Finding 2: U-shaped selectivity (Table 3 in paper)

From `results/<model>/expert_deep_analysis.json`:
```python
import json, numpy as np
with open("results/olmoe/expert_deep_analysis.json") as f:
    deep = json.load(f)

for layer_str, sels in sorted(deep["per_layer_selectivity"].items(), key=lambda x: int(x[0])):
    mean = np.mean(list(sels.values()))
    print(f"L{layer_str:>2}: {mean:.3f}")
```

Expected: U-shaped pattern (high -> low -> high across layers).

### Finding 3: Cluster stability (Table 4 in paper)

```python
for pair, pct in deep["cluster_stability"].items():
    print(f"  {pair}: {pct:.1f}%")
```

Expected: stability trough in middle layers, higher at edges.

---

## Extending to New Models

```bash
# Any HuggingFace MoE model
python python/analyze_experts_multi.py \
    --model-id <model-id-or-path> \
    --output results/<name>/ \
    --dtype fp16

# Then add to comparison
python python/compare_expert_findings.py \
    --model-dirs results/olmoe/ results/qwen_moe/ results/<name>/ \
    --output results/comparison_extended.json
```

The script auto-detects MoE architecture for models using standard patterns:
- OLMoE / Mixtral: `block_sparse_moe.gate`
- Qwen-MoE: `mlp.gate`
- DBRX: similar pattern
- Custom models with `trust_remote_code=True`

### Known limitations

- **DeepSeek-MoE:** Custom `modeling_deepseek.py` has compatibility issues with transformers >= 4.40 (removed `is_torch_fx_available`, `DynamicCache.from_legacy_cache`, `get_usable_length`). The script includes auto-patching for these issues. Additionally, DeepSeek's gate returns pre-computed `(topk_idx, topk_weight, aux_loss)` instead of raw logits, which requires the specialized hook detection logic included in the script.

- **Very large models (70B+):** May require INT4 quantization and/or multiple GPUs.

- **Non-English models:** Token classification is English-centric (function word list). Results for non-English-dominant models should be interpreted with this caveat.

---

## File Inventory

### Scripts

| File | Lines | Purpose |
|---|---|---|
| `python/analyze_experts_multi.py` | ~830 | Main multi-model analysis (Phases 1+2) |
| `python/expert_analysis_common.py` | ~280 | Shared: 30 categories, 240 prompts, token classifier |
| `python/compare_expert_findings.py` | ~390 | Cross-model comparison (CPU-only) |
| `python/analyze_experts.py` | ~600 | Original single-model OLMoE analysis (reference) |

### Pre-computed Results

| Directory | Model | Files |
|---|---|---|
| `results/olmoe/` | OLMoE-1B-7B | model_info.json, expert_catalog.json, expert_deep_analysis.json |
| `results/qwen_moe/` | Qwen1.5-MoE-A2.7B | model_info.json, expert_catalog.json, expert_deep_analysis.json |
| `results/` | OLMoE (original) | expert_catalog_exhaustive.json, expert_deep_analysis.json |

### Paper

| File | Purpose |
|---|---|
| `zenodo/paper_expert_specialization/expert_specialization.md` | Full paper (Markdown) |
| `zenodo/paper_expert_specialization/REPRODUCTION.md` | This file |

---

## Environment Used for Published Results

```
OS:             Ubuntu 22.04 (WSL2 on Windows 11)
GPU:            NVIDIA RTX 5070 Ti (16 GB VRAM)
RAM:            32 GB
Python:         3.12.3
torch:          2.7.0+cu128
transformers:   4.51.3
accelerate:     1.6.0
scikit-learn:   1.6.1
numpy:          2.2.4
CUDA:           13.2
```
