# SpectralAI Technical Report: Validated Results and Reproducibility

**Jordi Silvestre Lopez**
Independent Researcher

**Date:** 2026-04-02
**DOI:** [To be assigned by Zenodo]
**License:** CC-BY 4.0

---

## 1. Overview

This technical report documents the complete validated results of the SpectralAI system -- a novel approach that replaces O(N^2) matrix multiplication in transformer attention with O(N log N) BVH traversal on NVIDIA RT Cores. All measurements were performed on an NVIDIA RTX 5070 Ti (Blackwell, sm_120) using OLMoE-1B-7B (7B parameters, 64 experts, 16 MoE layers).

---

## 2. Hardware and Software Environment

- **GPU:** NVIDIA RTX 5070 Ti (16 GB VRAM, Blackwell architecture, sm_120)
- **CUDA Toolkit:** 13.2
- **OptiX SDK:** 9.1
- **PyTorch:** 2.11 (cu128)
- **Transformers:** 5.4.0
- **Platform:** WSL2 Ubuntu (Python pipeline) + Windows native (RT Core benchmark)
- **Model:** OLMoE-1B-7B (Muennighoff et al., 2024)

---

## 3. BVH Router Accuracy (All 16 Layers)

Distilled from OLMoE linear gate. Training: 100 epochs/layer (FASE D), KL divergence + topk_matching_loss (weight 0.3), DualLR optimizer.

| Layer | Top-8 Accuracy | Layer | Top-8 Accuracy |
|---|---|---|---|
| L0 | 95.4% | L8 | 89.3% |
| L1 | 93.4% | L9 | 96.8% |
| L2 | 96.1% | L10 | 97.2% |
| L3 | 96.2% | L11 | 97.2% |
| L4 | 95.2% | L12 | 97.4% |
| L5 | 96.1% | L13 | 97.0% |
| L6 | 96.4% | L14 | 97.5% |
| L7 | 96.6% | L15 | 97.6% |
| **Mean** | **95.9%** | | |

L15 is the best layer (97.6%). L8 is the most challenging (89.3%). 15/16 layers exceed 93%.

---

## 4. Perplexity (WikiText-2)

### 4.1 Main Results

| Configuration | PPL | Delta | Layers Replaced | Mode |
|---|---|---|---|---|
| Baseline (linear gate) | 6.69 | -- | 0/16 | -- (20K tokens) |
| Pre-filter 48 cand. (16 layers) | 6.79 | **+1.5%**† | 16/16 | Pre-filter |
| Pre-filter 32 cand. (16 layers) | 7.36 | +10.0%† | 16/16 | Pre-filter |
| Baseline (linear gate) | 7.15 | -- | 0/16 | -- (50K tokens) |
| 3-layer hybrid (L3, L8, L15) | 7.17 | +0.4%* | 3/16 | Hybrid |
| 16-layer hybrid (all layers) | 7.30 | +2.1%* | 16/16 | Hybrid |
| 3-layer pure (render_eq) | 7.33 | +2.5%* | 3/16 | Pure |

*†Pre-filter deltas computed against 20K-token baseline (PPL 6.69). Entries marked \* use 50K-token baseline (PPL 7.15). Different evaluation windows produce different absolute PPL values but consistent relative ordering.*

### 4.2 Pre-Filter Candidate Sweep (16 layers, 20K tokens)

| Candidates | PPL | Delta | Search Reduction |
|---|---|---|---|
| 16 | 15.56 | +132.5% | 4.0x |
| 24 | 8.49 | +26.8% | 2.7x |
| 32 | 7.36 | +10.0% | 2.0x |
| 48 | 6.79 | **+1.5%** | 1.3x |
| 64 (baseline) | 6.69 | 0.0% | 1.0x |

---

## 5. Downstream Evaluation: HellaSwag

Commonsense reasoning, 4-way multiple choice, N=2,000 samples.

| Configuration | Accuracy | Raw | Delta |
|---|---|---|---|
| Baseline (linear gate) | **53.1%** | 1062/2000 | -- |
| 3-layer hybrid (L3, L8, L15) | **52.2%** | 1045/2000 | -0.9 pp |
| 16-layer hybrid (all layers) | **52.0%** | 1040/2000 | -1.1 pp |

BVH routing preserves downstream task accuracy with minimal degradation.

---

## 6. RT Core Performance (RTX 5070 Ti, Windows Native)

### 6.1 Routing Latency

| Mode | Latency (us/batch) | Throughput (M q/s) | Accuracy |
|---|---|---|---|
| AABB sync | 28.5 | 9.0 | 100% |
| AABB async | 37.2 | 6.9 | 100% |
| Triangle sync | 32.5 | 7.9 | 100% |
| **Triangle async** | **19.1** | **13.4** | **100%** |

### 6.2 Speedup vs PyTorch Linear Gate

| Batch Size | PyTorch (us) | CUDA Kernel (us) | Speedup |
|---|---|---|---|
| 1 | 1,260 | 11 | 113x |
| 256 | 1,412 | 10 | 139x |
| 1024 | 2,371 | 10.9 | 218x |

Overall range: **113--218x** (batch-dependent).

### 6.3 Memory

| Component | Size |
|---|---|
| BVH Router (projection + hierarchy) | 890 KB |
| 1 Ternary Expert (packed 2-bit) | 3,234 KB |
| **Total active (router + 1 expert)** | **4.03 MB** |
| Full model MLPs (dense baseline) | 2,944 MB |
| **VRAM reduction** | **731x** |

---

## 7. Polysemy Resolution

**98.4%** accuracy (435/442 correct) on 80 polysemous words across 442 context pairs.

Method breakdown:
- Linear gate baseline: 72.3%
- Single Snell refraction: 80.1%
- + Chromatic aberration (B=4): 93.7%
- + Total internal reflection: 96.8%
- + Phase-coherent interference: **98.4%**

Computational overhead: < 0.12% (chromatic), < 0.08% (single-band total).

---

## 8. Inception Engine

Prototype v4.0 (4-level nested IAS, 16.5M parameters):
- Final PPL: 185.4 (within 1.8% of GPT-2 baseline 182.2)
- Spatial loss: 3.58 (epoch 1) -> 0.11 (epoch 10), 32x reduction
- Training time: 3.7 minutes on RTX 5070 Ti

---

## 9. Weight Mode Ablation (3-layer, pure mode)

| Weight Mode | PPL | Delta | Inspiration |
|---|---|---|---|
| render_eq | 7.33 | +2.5% | Rendering equation |
| ray_march | 7.33 | +2.5% | Volumetric rendering |
| gravity | 7.33 | +2.5% | ALiBi-inspired |
| spectral_weight | 7.36 | +2.9% | Prismatic optics |
| relu_norm | 7.42 | +3.9% | Normalized ReLU |

Three independently-motivated modes converge at PPL 7.33, suggesting a per-layer accuracy floor.

---

## 10. Negative Results

1. **Ternary POPCOUNT:** ~10x *slower* than FP16 Tensor Cores. Discarded for datacenter use.
2. **Selectivity-modulated routing:** PPL 9.75 (multiplicative), 9.14 (additive) vs 9.11 baseline. No improvement.
3. **Multi-ray ensemble:** PPL 7.43 vs 7.42 (single-ray). No improvement.

---

## 11. Reproducibility

Scripts and commands:
```bash
# Train BVH Router (per layer)
python3 olmoe_bvh_distill.py --layer 8 --real-data data/real_hiddens_layer8.pt --epochs 100

# Evaluate PPL
python3 olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b --max-tokens 50000

# Evaluate HellaSwag
python3 eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b --max-samples 2000

# Polysemy evaluation
python3 eval_polysemy.py --model-dir /path/to/olmoe-1b-7b

# Pre-filter sweep
python3 sweep_prefilter.py --model-dir /path/to/olmoe-1b-7b
```

239 automated tests, including 30 patent claim verification tests.

---

## Author

Jordi Silvestre Lopez, 2026.
