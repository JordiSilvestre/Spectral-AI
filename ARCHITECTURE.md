# SpectralAI — Architecture Reference

> Architecture guide for contributors working on this project.

## Project Vision

**SpectralAI** replaces the O(N^2) Transformer attention mechanism with O(N log N) ray tracing operations using NVIDIA RT Cores. Instead of computing a dense attention matrix (Query x Key), tokens are projected into a 3D geometric space organized as a BVH (Bounding Volume Hierarchy). A ray from the query token traverses the tree, finding semantically relevant tokens in O(log N) steps.

### The Problem

Traditional Transformers (GPT-4, Gemini) have O(N^2) attention complexity:
- 100K tokens = 10 billion cells in the attention matrix
- ~80 trillion FLOPs for the attention layer alone
- Requires racks of H100 GPUs (~$30K/unit)
- KV Cache with 96 layers consumes ~307 GB VRAM

### The Solution: O(N log N) Optical Attention

Tokens are mapped as polygons in a 3D space structured as a BVH tree:
- Complexity: **O(N log N)** — for 100K tokens, only log2(100,000) ~ 17 traversal steps
- Operations: ~6.9 billion ray intersections (vs 80 trillion FLOPs)
- Target hardware: **NVIDIA RTX 5070 Ti** (or RTX 4090) with dedicated RT Cores
- VRAM: A 100K-polygon BVH weighs ~10-50 MB (vs 307 GB KV Cache)

---

## Project Structure

```
spectral-ai/
├── README.md               # Public-facing project description
├── ARCHITECTURE.md         # This file (architecture reference)
├── LEARNINGS.md            # Decision log, failures, discoveries
├── STATUS.md               # Detailed status with file inventory
├── ROADMAP.md              # Development roadmap
├── BUILD.md                # Build instructions
├── CMakeLists.txt          # C++/CUDA build system
│
├── python/                 # ~50 files, ~25K lines
│   ├── bvh_router.py              # BVH Router (PyTorch, differentiable)
│   ├── orchestrator.py            # Full pipeline: Router -> Expert -> Output
│   ├── olmoe_bvh_distill.py       # BVH Router distillation from OLMoE gate
│   ├── olmoe_e2e_eval.py          # End-to-end PPL evaluation (multi-layer)
│   ├── eval_hellaswag.py          # HellaSwag downstream evaluation
│   ├── sweep_prefilter.py         # Pre-filter candidate sweep
│   ├── calibrate_router.py        # Post-hoc weight calibration (affine/linear)
│   ├── export_calibration.py      # Export calibration to FP16 binary + C header
│   ├── spectral_techniques.py     # 6 techniques: SmoothBVHHit, RMSNorm, DualLR...
│   ├── bvh_router_bridge.py       # Hybrid: PyTorch <-> CUDA kernel auto-select
│   ├── spectral_bridge.py         # Python <-> C++ bridge (tokenize -> 3D -> binary)
│   └── benchmark_scaling.py       # O(log N) vs O(N) scaling curve
│
├── cuda/
│   ├── README.md                  # CUDA/OptiX kernel documentation
│   ├── ray_generation.cu          # OptiX ray generation shader
│   ├── closest_hit.cu             # OptiX closest-hit + CoopVec calibration
│   ├── spectral_kernels.cu        # OptiX raygen 4-level IAS
│   └── v5/                        # Production kernels
│       ├── bvh_torch_ext.cu           # PyTorch extension zero-copy (105x speedup)
│       ├── ternary_torch_ext.cu       # POPCOUNT ternary extension
│       └── calibration_weights/       # Exported FP16 weights for in-shader use
│
├── include/                # C++ public headers
│   ├── token_geometry.h        # TokenNode: token -> BVH geometric object
│   ├── semantic_bvh.h          # Semantic BVH tree management
│   ├── optical_attention.h     # Optical attention mechanism interface
│   ├── spectral_ray.h          # Spectral encoding: PrismaticRay, Snell's law
│   └── alpha_bsh.h             # Semantic spheres + MatrixBlocks
│
├── src/                    # C++ implementations
│   ├── token_geometry.cpp      # Embedding -> 3D space projection
│   ├── semantic_bvh.cpp        # BVH construction and updates
│   └── alpha_bsh.cpp           # AlphaBSH orchestration Phase A/B
│
├── tests/                  # 223 automated tests
│   └── figures/                # 17 technical design figures + FIGURE_SPECS.md
├── paper/                  # Academic paper (arXiv submission)
├── figures/                # Publication figures
├── results/                # Evaluation result JSONs
├── scripts/                # Automation scripts
├── docs/
│   └── internal/           # Internal design notes and technical memos
└── checkpoints/            # Trained BVH Router weights (16 layers)
    └── olmoe_distill_layer{N}/bvh_router_best.pt
```

---

## Key Mathematical Concepts

### 1. Token -> 3D Geometry Projection

Each token has a D-dimensional embedding (e.g., D=2048 for OLMoE-1B-7B). We project it to 3D space via spherical PCA:

```
embedding in R^D  ->  position in R^3  (polygon centroid)
```

The 3D position preserves cosine similarity from the embedding space:
- Semantically similar tokens -> nearby AABBs in 3D space
- Geometric clustering reflects semantic clusters

### 2. TokenNode Structure (BVH Object)

```cpp
struct TokenNode {
    uint32_t token_id;           // Vocabulary token ID
    uint32_t position_in_seq;    // Sequence position (0..N-1)

    // Geometry (for RT Cores)
    float3   centroid;           // 3D position in semantic space
    float3   aabb_min;           // Semantic bounding box min
    float3   aabb_max;           // Semantic bounding box max
    float    semantic_radius;    // Semantic radius (context diversity)

    // Compressed embedding
    half     embedding[256];     // Reduced FP16 embedding (D -> 256 projection)

    // Accumulated attention
    float    attention_weight;   // Attention weight computed by the ray
    float    energy_remaining;   // Remaining ray energy after collision
};
```

### 3. Optical Attention Mechanism

**Ray Generation (Query -> Rays):**
- The query token emits `num_rays` rays from its semantic position
- Each ray represents a "thought dimension" (analogous to query heads)
- Initial direction: computed from the query token embedding

**Closest Hit (Semantic Collision):**
- When a ray hits a TokenNode, it computes relevance
- "Energy Loss" acts as Attention Decay

**Attention Decay Formula:**
```
attention_weight = E_0 * exp(-lambda * d_semantic)
```
Where:
- `E_0` = initial ray energy (1.0)
- `lambda` = semantic absorption coefficient (hyperparameter, ~0.1)
- `d_semantic` = semantic distance in 3D space (proxy for irrelevance)

**Resulting complexity:** O(N log N) — the BVH discards half the space at each level

### 4. Computational Advantage

| Metric | GPT-4 (MatMul) | SpectralAI (Ray Tracing) |
|---|---|---|
| Complexity | O(N^2) | O(N log N) |
| Operations (N=100K) | ~80T FLOPs | ~6.9B intersections |
| Hardware | Tensor Cores saturated | Dedicated RT Cores (idle silicon) |
| VRAM (KV Cache) | ~307 GB (96 layers) | ~10-50 MB (BVH) |
| Minimum hardware | Rack of H100s | Single RTX 5070 Ti |

---

## Three Key Innovations

### 1. RT Core Attention

BVH traversal replaces dense MatMul. O(log N) instead of O(N^2). OptiX 9.0 Cooperative Vectors enable in-shader calibration via Tensor Cores.

### 2. Inception Engine

4 nested IAS (Instance Acceleration Structure) levels encode 12 semantic dimensions using only 3D hardware. Each level is a "dimensional portal" that resets coordinates:
- Level 1: Domains (Science, Code, Humanities, General)
- Level 2: Subdomains (4 per domain = 16)
- Level 3: Concepts (4 per subdomain = 64 experts)

### 3. Spectral Routing

Rays carry a "color" (context vector `f in R^64`). Nodes act as prisms via Snell's law — the same node routes differently based on context, resolving polysemy without duplicating parameters.

```
n(sphere, f) = sigma(W_dispersion * f)       # Learned in training
sin(theta_out) = sin(theta_in) / n(sphere, f)
```

**Polysemy example:**
- BLUE ray (context=Code) hits "Loop" sphere -> refracts 45 deg -> programming experts
- RED ray (context=Music) hits the SAME sphere -> refracts 90 deg -> rhythm experts
- One point in space. Two different routings. Overhead: 0.03% of total compute.

---

## Current Results (2026-04-02)

Validated on **OLMoE-1B-7B** (7B parameters, 64 experts, 16 MoE layers).

### BVH Router Accuracy (Top-8, per layer)

| Layer | Accuracy | Layer | Accuracy |
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

### Perplexity (WikiText-2, 50K tokens)

| Configuration | PPL | Delta |
|---|---|---|
| Baseline (linear gate) | 6.69 | -- |
| Pre-filter 48 candidates (16 layers) | 6.79 | +1.5% |
| Hybrid 3 layers (L3, L8, L15) | 7.17 | +0.4% |
| Hybrid 16 layers | 7.30 | +2.1% |

### HellaSwag (N=2,000)

| Configuration | Accuracy | Delta |
|---|---|---|
| Baseline | 53.1% (1062/2000) | -- |
| 3-layer hybrid | 52.2% (1045/2000) | -0.9 pp |
| 16-layer hybrid | 52.0% (1040/2000) | -1.1 pp |

### Polysemy Resolution

**98.4%** accuracy (80 polysemous words, 442 context pairs).

### RT Core Benchmark (RTX 5070 Ti)

Triangle async mode: **19.1 us/batch**, **13.4 M queries/sec** — **48x speedup** vs PyTorch linear gate.

---

## Technology Stack

| Component | Technology |
|---|---|
| Core languages | C++17 + CUDA 13.2 |
| Ray tracing API | NVIDIA OptiX 9.1 |
| Build system | CMake 3.28+ |
| Python pipeline | PyTorch 2.x, Transformers, scikit-learn |
| Target hardware | NVIDIA RTX 4090 / RTX 5070 Ti (RT Cores required) |
| CUDA Compute | sm_89 (Ada) / sm_120 (Blackwell) |

---

## Design Decisions

1. **D -> 3D Projection:** PCA with cosine metric preservation. Information loss is acceptable because the BVH only needs relative topology, not exact semantics. The 256-float compressed embedding preserves 95%+ variance.

2. **Differentiability:** RT Cores are not differentiable by default. Solution: train with a differentiable Soft BVH (Gumbel-Softmax), then deploy to hardware RT Cores at inference time.

3. **Operation equivalence:** One ray-triangle intersection ~ 20-30 elementary FLOPs. This reduces the real advantage to ~380x (vs theoretical 11,500x), but remains substantial.

4. **BVH construction:** Built once per sequence, reused across all layers. Construction cost is O(N log N) amortized.

---

## Instructions for Contributors

When working on this project:

1. **Read LEARNINGS.md first** before implementing anything — it contains decisions already made and mistakes already made.
2. **Update LEARNINGS.md** when you discover a failure, make an important decision, or find a better alternative.
3. **Headers in `include/` are the source of truth** — if there's a conflict between a .cpp and a .h, the .h wins.
4. **Do not use std::vector in CUDA hot paths** — use flat arrays or thrust::device_vector.
5. **All GPU memory must have a corresponding free** — there is no GC in CUDA.
6. **The prototype does not need to be production-ready** — it needs to be correct and demonstrate O(N log N) viability.
7. **Per-layer checkpoints** are stored at `checkpoints/olmoe_distill_layer{N}/bvh_router_best.pt` — each layer has its own directory.
8. **WSL2 for Python pipeline** — all Python evaluation scripts run under WSL2, not Windows native.

---

## Author

Jordi Silvestre Lopez, 2026.
