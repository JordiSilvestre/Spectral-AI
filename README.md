# SpectralAI

**Attention without matrix multiplication.** RT Cores replace MatMul with O(N log N) ray tracing.

---

## What is this?

SpectralAI is a research prototype that replaces the O(N^2) Transformer attention mechanism with O(N log N) ray tracing operations, using the RT Cores already present in consumer NVIDIA GPUs (RTX 4090, RTX 5070 Ti).

Instead of computing a dense attention matrix (Query x Key), tokens are projected into a 3D geometric space organized as a BVH (Bounding Volume Hierarchy). A "ray" from the query token traverses the tree, finding semantically relevant tokens in O(log N) steps -- the same way a video game finds which objects a bullet hits.

### Why it matters

| Metric | GPT-4 (MatMul) | SpectralAI (Ray Tracing) |
|---|---|---|
| Attention complexity | O(N^2) | O(N log N) |
| Operations (N=100K) | ~80T FLOPs | ~6.9B intersections |
| KV Cache (96 layers) | ~307 GB VRAM | ~10-50 MB (BVH) |
| Minimum hardware | Rack of H100s | Single RTX 5070 Ti |

---

## Current Results (2026-04-02)

Validated on **OLMoE-1B-7B** (7B parameters, 64 experts, 16 MoE layers):

### Perplexity (WikiText-2)

| Configuration | PPL | Delta | Mode |
|---|---|---|---|
| Baseline (linear gate) | 6.69 | -- | Reference |
| Pre-filter 48 candidates (16 layers) | 6.79 | **+1.5%** | Pre-filter |
| Hybrid 3 layers (L3, L8, L15) | 7.17 | +0.4% | Hybrid |
| Hybrid 16 layers | 7.30 | +2.1% | Hybrid |
| Pure 3 layers (render_eq) | 7.33 | +2.5% | Pure |
| Pre-filter 32 candidates (16 layers) | 7.36 | +10.0% | Pre-filter |

### HellaSwag (Downstream Accuracy, N=2,000)

| Configuration | Accuracy | Delta |
|---|---|---|
| Baseline | 53.1% (1062/2000) | -- |
| 3-layer hybrid | 52.2% (1045/2000) | -0.9 pp |
| 16-layer hybrid | 52.0% (1040/2000) | **-1.1 pp** |

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

### RT Core Benchmark (RTX 5070 Ti)

| Mode | Latency (us/batch) | Throughput (M q/s) | Accuracy |
|---|---|---|---|
| AABB sync | 28.5 | 9.0 | 100% |
| AABB async | 37.2 | 6.9 | 100% |
| Triangle sync | 32.5 | 7.9 | 100% |
| **Triangle async** | **19.1** | **13.4** | **100%** |

**~48x speedup** vs PyTorch linear gate (~927 us). CUDA kernel: **85-170x** speedup.

### Polysemy Resolution

**98.4%** accuracy (80 polysemous words, 442 context pairs) -- the MoE gate routes the same word to different expert subsets depending on context, and our BVH router preserves this behavior.

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
[Top-k Expert Selection] -- top-8, weighted by routing probabilities
    |
    v
[Expert FFN SwiGLU] -- frozen (from OLMoE) or trainable
    |
    v
[Output Projection] --> logits
```

Three key innovations:

1. **RT Core Attention:** BVH traversal replaces dense MatMul. O(log N) instead of O(N^2). OptiX 9.0 Cooperative Vectors enable in-shader calibration via Tensor Cores.

2. **Inception Engine:** 4 nested IAS levels encode 12 semantic dimensions using only 3D hardware. Each level is a "dimensional portal" that resets coordinates.

3. **Spectral Routing:** Rays carry a "color" (context vector). Nodes act as prisms (Snell's law) -- the same node routes differently based on context, resolving polysemy without duplicating parameters.

---

## Project Structure

```
spectral-ai/
├── README.md              # This file
├── ARCHITECTURE.md        # Architecture reference for contributors
├── LEARNINGS.md           # Decision log, failures, discoveries
├── STATUS.md              # Detailed status with file inventory
├── ROADMAP.md             # Development roadmap
├── BUILD.md               # Build instructions
├── CMakeLists.txt         # C++/CUDA build system
│
├── python/                # ~50 files, ~25K lines
│   ├── bvh_router.py          # BVH Router (PyTorch, differentiable)
│   ├── orchestrator.py        # Full pipeline: Router -> Expert -> Output
│   ├── olmoe_bvh_distill.py   # BVH Router distillation from OLMoE gate
│   ├── olmoe_e2e_eval.py      # End-to-end PPL evaluation (multi-layer)
│   ├── eval_hellaswag.py      # HellaSwag downstream evaluation
│   ├── sweep_prefilter.py     # Pre-filter candidate sweep
│   ├── calibrate_router.py    # Post-hoc weight calibration (affine/linear)
│   ├── export_calibration.py  # Export calibration to FP16 binary + C header
│   └── benchmark_scaling.py   # O(log N) vs O(N) scaling curve
│
├── cuda/
│   ├── closest_hit.cu         # OptiX closest-hit shader + CoopVec calibration
│   ├── ray_generation.cu      # OptiX ray generation shader
│   └── v5/                    # Production kernels
│       ├── bvh_torch_ext.cu       # PyTorch extension zero-copy (105x speedup)
│       ├── ternary_torch_ext.cu   # POPCOUNT ternary extension
│       └── calibration_weights/   # Exported FP16 weights for in-shader use
│
├── include/               # C++ public headers
├── src/                   # C++ implementations
├── tests/                 # 223 automated tests
├── patents/               # 3 technical design documents + 17 figures
├── paper/                 # Academic paper (arXiv submission)
├── figures/               # Publication figures
├── scripts/               # Automation scripts
├── docs/                  # Technical documentation
│   └── internal/          # Internal design notes
└── checkpoints/           # Trained BVH Router weights (16 layers)
```

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 4090 or RTX 5070 Ti (RT Cores required)
- **VRAM:** 16 GB minimum
- **RAM:** 24 GB+ (for loading OLMoE-1B-7B during evaluation)
- **CUDA Toolkit:** 13.2+ (for sm_120 / Blackwell support)
- **OptiX SDK:** 9.1 (for RT Core pipeline; optional for CUDA-only routing)
- **Python:** 3.10+, PyTorch 2.x with CUDA

---

## Quick Start

```bash
# WSL2 (recommended for Python pipeline)
cd /mnt/j/Proyectos/SPECTRAL\ AI
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate safetensors datasets scikit-learn

# Step-by-step:

# 1. Extract hidden states from OLMoE
python python/extract_real_hiddens.py --model-dir /path/to/olmoe-1b-7b --layer 8

# 2. Train BVH Router
python python/olmoe_bvh_distill.py --layer 8 --real-data data/real_hiddens_layer8.pt --epochs 50

# 3. Calibrate weights
python python/calibrate_router.py --mode linear --epochs 100 \
    --real-data data/real_hiddens_layer8.pt --device cpu

# 4. Evaluate PPL
python python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt --max-tokens 50000

# 5. Evaluate HellaSwag (downstream task)
python python/eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b --max-samples 2000

# Build C++/CUDA (Windows native with OptiX):
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 \
    -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
cmake --build . --config Release

# Run RT Core benchmark:
Release\rt_router_benchmark.exe ".."
```

---

## Publications

Three preprints available on Zenodo:

| Title | Scope |
|---|---|
| SpectralAI: O(N log N) Hardware-Accelerated Expert Routing | RT Core attention + Inception Engine |
| Spectral Routing: Context-Dependent Expert Selection | Snell's law, polysemy, TIR |
| Expert Specialization in MoE Language Models | Syntactic roles, U-shaped selectivity |

---

## License

CC-BY 4.0.

## Author

Jordi Silvestre Lopez, 2026.
