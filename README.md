# SpectralAI

**O(log N) MoE Expert Routing via Hardware-Accelerated BVH Traversal on Consumer NVIDIA GPUs.**

Replace the standard dense routing gate in Mixture-of-Experts models with geometric ray tracing — the same technology GPUs use to render realistic lighting in games. No matrix multiplication needed for routing. Runs on a single RTX card.

---

## Key Results

Validated on **OLMoE-1B-7B** (7B parameters, 64 experts, 16 MoE layers) — RTX 5070 Ti:

| Metric | Standard Gate | SpectralAI BVH | Improvement |
|---|---|---|---|
| **Routing complexity** | O(N) | O(log N) | Sublinear scaling |
| **Routing latency** | 927 µs | 19.1 µs (RT Core) / 10.4 µs (CUDA) | **48–89× faster** |
| **Throughput** | 276K queries/s | 13.4M q/s (RT) / 24.7M q/s (CUDA) | **48–89× higher** |
| **Router VRAM** | 2,944 MB | 4.03 MB | **731× less memory** |
| **Perplexity (16 layers)** | 7.00 | 7.00 | **+0.0% — zero degradation** |
| **Top-8 routing accuracy** | 100% (ref) | 96.6% mean (all layers > 95%) | Minimal mismatch |
| **Polysemy resolution** | — | 98.4% (80 words, 442 pairs) | Context-aware routing |

### Perplexity (WikiText-2, 20K tokens)

| Configuration | PPL | Delta |
|---|---|---|
| Baseline (linear gate) | 7.00 | — |
| **BVH pre-filter (16 layers, 48 candidates)** | **7.00** | **+0.0%** |
| BVH pre-filter (32 candidates) | 7.36 | +10.0% |
| Pure BVH (3 layers, no gate) | 7.33 | +2.5% |

### HellaSwag (Downstream Accuracy, N=2,000)

| Configuration | Accuracy | Delta |
|---|---|---|
| Baseline | 53.1% | — |
| 16-layer BVH hybrid | 52.0% | −1.1 pp |

### BVH Router Accuracy (Top-8 overlap with original gate)

| Layer | Accuracy | Layer | Accuracy |
|---|---|---|---|
| L0 | 95.4% | L8 | **96.4%** |
| L1 | **95.9%** | L9 | 96.8% |
| L2 | 96.1% | L10 | 97.2% |
| L3 | 96.2% | L11 | 97.2% |
| L4 | 95.1% | L12 | 97.4% |
| L5 | 96.1% | L13 | 97.0% |
| L6 | 96.4% | L14 | 97.5% |
| L7 | 96.6% | L15 | 97.6% |
| **Mean** | **96.6%** | **All layers** | **> 95%** |

### RT Core Benchmark (RTX 5070 Ti, 64 experts, batch=256)

| Mode | Latency (µs/batch) | Throughput (M q/s) |
|---|---|---|
| AABB sync | 28.5 | 9.0 |
| AABB async | 37.2 | 6.9 |
| Triangle sync | 32.5 | 7.9 |
| **Triangle async (best)** | **19.1** | **13.4** |

### Inference Profiling (OLMoE-1B-7B, RTX 5070 Ti)

| Component | Time | % of inference |
|---|---|---|
| Expert MLPs (SwiGLU) | 33.0 ms | 63.4% |
| Attention | 10.4 ms | 20.0% |
| Other (embed, norm, etc.) | 7.2 ms | 13.8% |
| **Routing gate** | **1.45 ms** | **2.8%** |
| **Total forward pass** | **52 ms** | 100% |

> With 64 experts, routing is ~3% of inference. This grows linearly with expert count. At 10K+ experts, the O(N) gate dominates the forward pass while the BVH stays at ~25 µs.

### Power Consumption (RTX 5070 Ti)

| State | Power |
|---|---|
| Idle | 61 W |
| Inference (avg) | 119 W |
| Inference (peak) | 140 W |
| Energy per token | 31 mJ |

---

## How It Works

```text
Input tokens
     |
     v
[Embedding] --> [3D Projection (PCA, 2048 → 3D)]
     |
     v
[BVH Router] -- 4 nested 3D levels (hierarchical, not 12 independent dims)
     |          Level 1: Domains (4 clusters)
     |          Level 2: Subdomains (4 per domain = 16)
     |          Level 3: Experts (4 per subdomain = 64)
     v
[Top-k Expert Selection] -- top-8, weighted by routing probabilities
     |
     v
[Expert FFN SwiGLU] -- frozen original OLMoE weights
     |
     v
[Output Projection] --> logits
```

### Three Key Innovations

1. **RT Core Routing:** Expert centroids are organized as AABB bounding boxes in a BVH tree. At inference, a ray from the token position traverses the tree using the GPU's dedicated RT Cores (the same silicon that traces light in games). This finds the top-k experts in O(log N) instead of O(N) dot products. The RT Cores are normally idle during inference — this puts them to work.

2. **Inception Engine:** Unlike Vulkan (limited to TLAS + BLAS = 2 levels), OptiX allows nested IAS → IAS structures. We use 4 nested levels, each operating in its own 3D coordinate space with independent transforms. This creates a hierarchical routing structure across 4 nested 3D spaces — the information is still geometric, not 12 independent semantic dimensions.

3. **Spectral Routing:** Tokens carry a "wavelength" (context vector). At domain boundaries, Snell's Law refracts compatible contexts through while incompatible ones trigger Total Internal Reflection, bouncing the token to the correct semantic domain. This resolves polysemy: "bank" (river) routes differently from "bank" (financial) without duplicating parameters.

---

## Quick Start

```bash
# Clone and setup (WSL2 recommended)
git clone https://github.com/JordiSilvestre/Spectral-AI.git
cd Spectral-AI
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate safetensors datasets scikit-learn

# 1. Extract hidden states from OLMoE
python3 python/extract_real_hiddens.py --model-dir /path/to/olmoe-1b-7b --layer 8

# 2. Train BVH Router for a single layer
python3 python/olmoe_bvh_distill.py --model-dir /path/to/olmoe-1b-7b \
    --layer 8 --epochs 100 --spectral --spectral-dim 256 \
    --real-data data/real_hiddens_layer8.pt

# 3. Evaluate PPL (single layer)
python3 python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt \
    --layer 8 --max-tokens 50000

# 4. Evaluate PPL (all 16 layers, hybrid mode)
python3 python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --n-candidates 48 --hybrid --max-tokens 20000 \
    --multi-layer "0:checkpoints/olmoe_best/bvh_router_L0_best.pt,..."

# 5. Evaluate HellaSwag
python3 python/eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b --max-samples 2000

# 6. Profile routing fraction
python3 python/profile_routing_fraction.py --model-dir /path/to/olmoe-1b-7b

# 7. Profile power consumption
python3 python/profile_power.py --model-dir /path/to/olmoe-1b-7b

# Build OptiX extension (Windows native):
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 \
    -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
cmake --build . --config Release
Release\rt_router_benchmark.exe ".."
```

---

## Project Structure

```text
spectral-ai/
├── README.md               # This file
├── ROADMAP.md              # Development roadmap with all results
├── STATUS.md               # Detailed technical status
├── BUILD.md                # Build instructions
│
├── python/                 # Python pipeline (~50 files)
│   ├── bvh_router.py       # BVH Router (PyTorch, differentiable)
│   ├── olmoe_bvh_distill.py# Router distillation from OLMoE gate
│   ├── olmoe_e2e_eval.py   # End-to-end PPL evaluation
│   ├── eval_hellaswag.py   # Downstream task evaluation
│   ├── profile_routing_fraction.py  # Inference time breakdown
│   └── profile_power.py    # GPU power consumption profiling
│
├── cuda/                   # CUDA/OptiX kernels
│   ├── closest_hit.cu      # OptiX closest-hit shader
│   └── v5/                 # Production kernels
│       ├── bvh_torch_ext.cu    # PyTorch extension (89× speedup)
│       └── ternary_torch_ext.cu# POPCOUNT ternary extension
│
├── tests/                  # 195 passing, 14 skipped (OptiX/WSL)
├── checkpoints/
│   └── olmoe_best/         # Best checkpoints for all 16 layers
├── presentation/           # Animated HTML presentation
├── zenodo/                 # Preprints and Zenodo packages
└── docs/                   # Technical documentation
```

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 2060+ (RT Cores required). Tested on RTX 5070 Ti and RTX 4090.
- **VRAM:** 16 GB minimum (for OLMoE-1B-7B evaluation)
- **RAM:** 24 GB+ (model loading)
- **Software:** CUDA Toolkit 12.x+, Python 3.10+, PyTorch 2.x
- **Optional:** OptiX SDK 9.1 (for RT Core pipeline; CUDA-only routing works without it)

---

## Test Suite

```
195 passed, 0 failed, 14 skipped (OptiX/WSL)
```

| Suite | Tests | Status |
|---|---|---|
| BVH Router core | 79 | ✅ All passing |
| Gate wrapper + calibration | 48 | ✅ All passing |
| Enhanced BVH + spectral | 47/47 | ✅ All passing |
| Polysemy benchmark | 21 | ✅ All passing |
| OptiX integration | 14 | ⏭️ Skipped (requires Windows native) |

---

## Publications

Three preprints on Zenodo:

| Title | DOI |
|---|---|
| SpectralAI: O(log N) Hardware-Accelerated Expert Routing via RT Core BVH Traversal | [10.5281/zenodo.19457288](https://doi.org/10.5281/zenodo.19457288) |
| Expert Specialization in MoE Language Models: Syntactic Roles Dominate Semantic Topics | [10.5281/zenodo.19457411](https://doi.org/10.5281/zenodo.19457411) |
| Spectral Routing: Context-Dependent Expert Selection via Optical Refraction | [10.5281/zenodo.19457473](https://doi.org/10.5281/zenodo.19457473) |

---

## Citation

```bibtex
@misc{silvestre2026spectralai,
  author = {Silvestre Lopez, Jordi},
  title = {SpectralAI: O(log N) Hardware-Accelerated Expert Routing 
           via RT Core BVH Traversal},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19457288},
  url = {https://github.com/JordiSilvestre/Spectral-AI}
}
```

## License

Apache 2.0

## Author

Jordi Silvestre Lopez, 2026.
