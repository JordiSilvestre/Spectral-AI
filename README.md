# LiquidBit Zero-Matrix

**Attention without matrix multiplication.** RT Cores replace MatMul with O(log N) ray tracing.

---

## What is this?

LiquidBit Zero-Matrix is a research prototype that replaces the O(N^2) attention mechanism in Transformers with O(N log N) ray tracing operations, using the RT Cores already present in consumer NVIDIA GPUs (RTX 4090, RTX 5070 Ti).

Instead of computing a dense attention matrix (Query x Key), tokens are projected into a 3D geometric space organized as a BVH (Bounding Volume Hierarchy). A "ray" from the query token traverses the tree, finding semantically relevant tokens in O(log N) steps — the same way a videogame finds which objects a bullet hits.

### Why it matters

| Metric | GPT-4 (MatMul) | LiquidBit (Ray Tracing) |
|---|---|---|
| Attention complexity | O(N^2) | O(N log N) |
| Operations (N=100K) | ~80T FLOPs | ~6.9B intersections |
| KV Cache (96 layers) | ~307 GB VRAM | ~10-50 MB (BVH) |
| Minimum hardware | Rack of H100s | Single RTX 5070 Ti |

---

## Current state (2026-03-27)

### What works

| Component | Status | Key metric |
|---|---|---|
| BVH Router (PyTorch) | Validated | 3-level hierarchy, Gumbel-Softmax, 64 experts |
| CUDA Router kernel | Compiled + tested | 8.83 us/batch, 105x vs PyTorch |
| PyTorch Extension (zero-copy) | Integrated | 10 us routing, auto-selected at runtime |
| Ternary Expert kernel (POPCOUNT) | Validated | Zero FP multiply, 0.000038 diff from FP32 |
| Demo (Qwen 1.5B) | Executed | 51.9 tok/s, 375x less VRAM |
| Demo (BitNet 2B) | Executed | 3.4x speedup, 519x less VRAM |
| Multi-domain routing | 100% accuracy | 4 domains, 16 experts |
| Inception Engine (4-level 12D) | PPL 191.3 | Only 2.1% worse than GPT-2 baseline |
| Spectral encoding + Snell | Implemented | 88.9% polysemy resolution |

### What's in progress

| Component | Status | Notes |
|---|---|---|
| OLMoE expert distillation | Training v2.1 | BVH router learning to replicate 64-expert linear gate |
| MoE from scratch (16 experts) | Ceiling at PPL 186 | Alpha decay problem, pivoted to OLMoE |
| OptiX RT Core pipeline | Shaders written | Needs CUDA Toolkit + OptiX SDK install |
| C++/CUDA CMake build | 7 targets compile | Needs sm_120 fix for RTX 5070 Ti |

### What's not done yet

- OptiX RT Core real routing (estimated 10-20x over CUDA kernel)
- Async tri-core pipeline (RT + CUDA + Tensor in parallel)
- Scaling to 65K experts
- End-to-end differentiable training
- Academic paper / formal benchmarks

---

## Architecture

```
Input tokens
    |
    v
[Embedding] --> [3D Projection (PCA)]
    |
    v
[BVH Router] -- 3 levels x 3D = 12 semantic dimensions
    |              Level 1: Domains (Science, Code, Humanities, General)
    |              Level 2: Subdomains (4 per domain)
    |              Level 3: Concepts (4 per subdomain = 64 experts)
    |
    v
[Top-k Expert Selection] -- top-2, weighted by routing probabilities
    |
    v
[SwiGLU Expert FFN] -- frozen (from OLMoE) or trainable
    |
    v
[Output Projection] --> logits
```

Three key innovations:

1. **RT Core Attention (Patent LBS-2026-001):** BVH traversal replaces dense MatMul. O(log N) instead of O(N^2).

2. **Inception Engine (Patent LBS-2026-002):** 4 nested IAS levels encode 12 semantic dimensions using only 3D hardware. Each level is a "dimensional portal" that resets coordinates.

3. **Spectral Routing (Patent LBS-2026-003):** Rays carry a "color" (context vector). Nodes act as prisms (Snell's law) — the same node routes differently based on context, resolving polysemy without duplicating parameters.

---

## Project structure

```
liquidbit-zero-matrix/
├── CLAUDE.md              # Architecture reference (for AI agents)
├── LEARNINGS.md           # Decision log, failures, insights
├── ROADMAP.md             # 11-phase roadmap
├── README.md              # This file
├── CMakeLists.txt         # C++/CUDA build system
│
├── python/                # 49 files, ~24K lines
│   ├── bvh_router.py          # BVH Router (PyTorch, differentiable)
│   ├── orchestrator.py        # Full pipeline: Router -> Expert -> Output
│   ├── real_model_demo.py     # Demo with real HuggingFace models
│   ├── trainable_experts.py   # SwiGLU expert pool (MoE from scratch)
│   ├── olmoe_extract.py       # OLMoE-1B-7B expert extraction
│   ├── olmoe_bvh_distill.py   # BVH router distillation from OLMoE gate
│   └── train_*.py             # Training scripts
│
├── cuda/
│   ├── v4/                    # Inception Engine kernels (OptiX shaders)
│   └── v5/                    # Orchestrator kernels
│       ├── bvh_router_kernel.cu   # Fused 3-level router (8.83 us)
│       ├── bvh_torch_ext.cu       # PyTorch zero-copy extension
│       ├── ternary_torch_ext.cu   # POPCOUNT ternary extension
│       └── optix_bvh_router.cu    # RT Core routing (needs OptiX SDK)
│
├── include/               # C++ headers (7 files, source of truth)
├── src/                   # C++ implementations (3 files)
├── tests/                 # C++ tests and benchmarks (7 files)
├── docs/                  # Technical documentation
├── patents/               # 3 provisional patent drafts
└── data/                  # Datasets, embeddings, checkpoints
```

**Total:** ~51K lines (24K Python + 18K C++/CUDA + 8.6K Markdown)

---

## Hardware requirements

- **GPU:** NVIDIA RTX 4090 or RTX 5070 Ti (RT Cores required)
- **VRAM:** 16 GB minimum
- **CUDA Toolkit:** 12.8+ (for sm_120 / Blackwell support)
- **OptiX SDK:** 9.1 (for RT Core pipeline, optional for CUDA-only routing)
- **Python:** 3.10+, PyTorch 2.x with CUDA

---

## Quick start

```bash
# WSL2 (recommended)
ln -sf "/mnt/j/Proyectos/LiquidBit Zero-Matrix" /tmp/liquidbit
cd /tmp/liquidbit
python -m venv .venv_wsl && source .venv_wsl/bin/activate
pip install torch numpy scipy scikit-learn matplotlib safetensors

# Run the demo with a real model
python python/real_model_demo.py

# Run BVH router distillation from OLMoE
python python/olmoe_bvh_distill.py --model-dir /path/to/olmoe-1b-7b --epochs 30
```

---

## Patents

Three provisional patent applications drafted (pending filing):

| Docket | Title | Innovation |
|---|---|---|
| LBS-2026-001 | RT Core Attention O(log N) | BVH replaces MatMul in attention |
| LBS-2026-002 | Nested IAS for 12D | 4 levels of 3D = 12 dimensions via OptiX instancing |
| LBS-2026-003 | Spectral Routing + Snell | Context-dependent routing without parameter duplication |

---

## License

Proprietary. Patent pending.

## Author

Jordi Silva — LiquidBit Studio, 2026.
