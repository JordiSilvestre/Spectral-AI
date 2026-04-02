# SpectralAI Zero-Matrix — System Architecture

> Last updated: 2026-03-27. For full mathematical details see ARCHITECTURE.md.

## Overview

SpectralAI replaces the O(N^2) attention mechanism in Transformers with O(N log N) ray tracing.

```
Traditional Transformer:
    Q, K, V = Linear(hidden)
    Attention = softmax(Q @ K^T / sqrt(d)) @ V    <-- O(N^2) MatMul

SpectralAI:
    pos_3d = PCA_project(hidden)                   <-- O(N)
    expert_id = BVH_traverse(pos_3d, spectral)     <-- O(log N) per token
    output = Expert_FFN[expert_id](hidden)          <-- O(k^2) local MatMul
```

## Core Components

### 1. BVH Router (`python/bvh_router.py`)

Hierarchical 3-level router using 3D geometry:

```
Level 1 (Dims 1-3):  4 domains       --> Science, Code, Humanities, General
Level 2 (Dims 4-6):  4 subdomains    --> per domain (16 total)
Level 3 (Dims 7-9):  4 concepts      --> per subdomain (64 total = n_experts)
```

- Training: Gumbel-Softmax (differentiable, soft probs)
- Inference: argmax (hard routing, deterministic)
- Spectral encoding: context vector modulates routing via Snell's law refraction
- Parameters: ~750K (vs 131K for OLMoE's linear gate)

### 2. Expert Pool

Two modes:

**A. Trainable SwiGLU experts** (`python/trainable_experts.py`)
- 16 experts, 512-dim hidden, 2048-dim intermediate
- 3.1M params per expert, ~50M total
- Status: ceiling at PPL=186 from scratch

**B. Frozen OLMoE experts** (`python/olmoe_extract.py`)
- 64 experts from OLMoE-1B-7B (layer 8)
- 2048-dim hidden, 1024-dim intermediate
- 6.3M params per expert, 402M per layer
- Already specialized after pre-training

### 3. CUDA Kernels (`cuda/v5/`)

| Kernel | Function | Latency | Status |
|---|---|---|---|
| `bvh_router_kernel.cu` | Fused 3-level router, constant mem | 8.83 us | Validated |
| `bvh_torch_ext.cu` | PyTorch zero-copy extension | 10 us | Integrated |
| `ternary_torch_ext.cu` | POPCOUNT ternary expert | - | Integrated |
| `bvh_router_deep.cu` | Scalable 3-8 levels (65K experts) | ~15 us | Compiled |
| `optix_bvh_router.cu` | RT Core routing via OptiX | - | Needs SDK |

### 4. Inception Engine (`include/inception_engine.h`)

4-level nested IAS for 12-dimensional semantic search using 3D hardware:

```
IAS_root (Level 1, dims 1-3)
  +-- IAS_science (Level 2, dims 4-6)     <-- AffinePortal transform
        +-- IAS_quantum (Level 3, dims 7-9)
              +-- GAS leaf (Level 4, dims 10-12)  <-- SemanticString
```

Each `AffinePortal` is a 4x4 transform matrix that "resets" coordinates.
OptiX resolves this natively via `optixGetInstanceTransformFromHandle()`.

### 5. Spectral Routing (`include/spectral_ray.h`)

Rays carry a "color" (context vector f in R^64):
- Same node routes differently based on context
- Refraction index: `n(sphere, f) = sigmoid(W_dispersion @ f)`
- Snell's law determines routing direction
- Resolves polysemy without duplicating parameters

## Data Flow (Inference)

```
1. Input tokens  --> Embedding lookup
2. Embeddings    --> PCA projection to 3D space
3. 3D positions  --> BVH traversal (3 levels)
                     + Spectral encoding (context color)
                     + Prismatic refraction (Snell's law)
4. Expert IDs    --> Top-k selection (k=2 default)
5. Expert FFNs   --> SwiGLU forward (frozen or trainable)
6. Expert output  --> Blend gate (alpha weighting)
7. Logits        --> Vocabulary projection
```

## File Organization

| Directory | Content | Lines |
|---|---|---|
| `python/` | 49 Python files — router, experts, training, demos | ~24K |
| `cuda/v4/` | Inception Engine kernels (OptiX shaders) | ~4.1K |
| `cuda/v5/` | Orchestrator kernels (router, expert, pipeline) | ~4.8K |
| `include/` | 7 C++ headers (source of truth for structs) | ~3.6K |
| `src/` | 3 C++ implementations | ~1.5K |
| `tests/` | 7 C++ test/benchmark files | ~3K |

## Key Metrics

| Metric | Value | Source |
|---|---|---|
| Routing latency (CUDA ext) | 10 us / batch=256 | `bvh_torch_ext.cu` |
| Routing speedup vs PyTorch | 105x | Benchmark |
| E2E speedup (Orchestrator) | 1.89x | `benchmark_cuda_e2e.py` |
| Demo throughput (Qwen 1.5B) | 51.9 tok/s | `real_model_demo.py` |
| VRAM savings | 375x (Qwen), 519x (BitNet) | Demo results |
| Multi-domain routing | 100% accuracy (4 domains) | `train_multi_domain.py` |
| Inception attention PPL | 191.3 (vs GPT-2 187.4) | Only 2.1% worse |

## Known Limitations

1. **3D bottleneck**: Projecting 2048-dim hidden states to 3D loses information.
   The EnhancedBVHRouter (v2.1) mitigates this with 128-dim feature preservation.

2. **No differentiable RT Cores**: RT Cores are not differentiable. Training uses
   CUDA Gumbel-Softmax as a surrogate; only inference runs on RT Cores.

3. **Expert loading**: With 64+ experts, only top-k fit in VRAM simultaneously.
   LRU cache handles expert eviction. NVMe-backed cache planned for 65K experts.

4. **MoE training from scratch**: Failed at PPL=186 ceiling with alpha decay.
   Pivoted to using pre-trained experts from OLMoE-1B-7B.
