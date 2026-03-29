# SpectralAI Zero-Matrix — Training Components

> Last updated: 2026-03-27

## Training Approaches (in chronological order)

### 1. Inception Attention Training (v4.0) -- COMPLETED

Direct replacement of MatMul attention with 4-level BVH traversal.

- **Script:** `train_inception.py`
- **Result:** PPL=191.3 (vs GPT-2 baseline 187.4) -- only 2.1% worse
- **Params:** 16.5M
- **Checkpoint:** `checkpoints/inception_best.pt`

### 2. Orchestrator Multi-Domain (v5.0) -- COMPLETED

Router BVH + backbone conditioned on expert selection, 4 synthetic domains.

- **Script:** `train_multi_domain.py`
- **Result:** 100% routing accuracy across 4 domains
- **Params:** 19.9M (router ~750K, backbone ~19M)
- **Checkpoint:** `checkpoints/orchestrator_multidomain_best.pt`

### 3. MoE From Scratch -- FAILED (ceiling)

16 trainable SwiGLU experts trained from random init on WikiText-2.

- **Script:** `train_moe.py`
- **Expert code:** `trainable_experts.py`
- **Result:** PPL=186, alpha decay 0.35 -> 0.11 (experts never specialized)
- **Root cause:** 116M params with only 5M tokens -- not enough data for specialization
- **Checkpoint:** `data/moe_best.pt` (1.4 GB)

### 4. OLMoE Expert Distillation (v2.1) -- IN PROGRESS

Use OLMoE-1B-7B's 64 pre-specialized experts (frozen). Train only the BVH Router
to replicate OLMoE's linear gate routing decisions.

- **Extraction:** `olmoe_extract.py` -- loads 1 layer (64 experts + gate)
- **Training:** `olmoe_bvh_distill.py` -- knowledge distillation (KL + CE)
- **v1 result:** FAILED -- 13% top-8 overlap (= random). 2048->3D bottleneck.
- **v2.1:** EnhancedBVHRouter with 128-dim features + distillation loss. Training in progress.

---

## Training Components Reference

### DuplScore Optimizer (`dupl_score_optimizer.py`)

Decides whether to duplicate a polysemous concept across multiple spheres or use O(1) wormhole pointers.

```
DuplScore(C) = (Sum_s f(C,s) * R(C,s)) * exp(-gamma * D(Sc)) - delta * (|Sc|-1) * size(C)
If DuplScore > tau: DUPLICATE. Else: WORMHOLE.
```

### Fuzzy BSH (`fuzzy_bsh.py`)

Differentiable BSH tree with soft membership for end-to-end training.

```
P(token in sphere_k) = softmax(-||token - center_k||^2 / (2*T^2))
```

Temperature annealing: T starts at 1.0, decays by 0.9 every N epochs.
Final accuracy: 91.7% cluster assignment.

### Spatial Loss (`spatial_loss.py`)

Three-component loss for geometric training:
- `L_prox`: Similar tokens must be close in 3D space
- `L_cover`: Each sphere must cover its assigned tokens
- `L_inter`: Polysemous tokens must be at sphere intersections

### Semantic Initializer (`semantic_initializer.py`)

K-means hierarchical initialization for BVH centers from embedding statistics.
Pending integration with real (non-synthetic) datasets.

---

## Available Checkpoints

| File | Size | Description |
|---|---|---|
| `data/moe_best.pt` | 1.4 GB | MoE from scratch (16 experts, PPL=186) |
| `checkpoints/orchestrator_multidomain_best.pt` | 77 MB | Multi-domain (100% routing) |
| `checkpoints/inception_best.pt` | 64 MB | Inception attention (PPL=191.3) |
| `checkpoints/gpt2_baseline_best.pt` | 63 MB | GPT-2 baseline (PPL=187.4) |
| `checkpoints/olmoe_distill/bvh_router_best.pt` | 5.2 MB | Router distillation v1 (failed) |

---

## Datasets

| File | Size | Description |
|---|---|---|
| `data/wikitext2_*_tokens.npy` | 10 MB | WikiText-2 tokenized (train/val/test) |
| `python/{code,legal,science}_*_tokens.npy` | 48 MB | Synthetic domain datasets |
| `data/gate_labels_500k.npz` | 26 MB | OLMoE gate routing labels (layer 8) |
| `python/embeddings_full.npy` | 58 MB | GloVe embeddings |
| `python/embeddings_3d.npy` | 588 KB | PCA-projected 3D embeddings |
