# SpectralAI Zero-Matrix

**Attention without matrix multiplication.** RT Cores replace MatMul with O(N log N) ray tracing.

---

## What is this?

SpectralAI Zero-Matrix is a research prototype that replaces the O(N^2) Transformer attention mechanism with O(N log N) ray tracing operations, using the RT Cores already present in consumer NVIDIA GPUs (RTX 4090, RTX 5070 Ti).

Instead of computing a dense attention matrix (Query x Key), tokens are projected into a 3D geometric space organized as a BVH (Bounding Volume Hierarchy). A "ray" from the query token traverses the tree, finding semantically relevant tokens in O(log N) steps -- the same way a video game finds which objects a bullet hits.

### Why it matters

| Metric | GPT-4 (MatMul) | SpectralAI (Ray Tracing) |
|---|---|---|
| Attention complexity | O(N^2) | O(N log N) |
| Operations (N=100K) | ~80T FLOPs | ~6.9B intersections |
| KV Cache (96 layers) | ~307 GB VRAM | ~10-50 MB (BVH) |
| Minimum hardware | Rack of H100s | Single RTX 5070 Ti |

---

## Current status (2026-04-02)

### What works

| Component | Status | Key metric |
|---|---|---|
| BVH Router (PyTorch) | Validated | 3-level hierarchy, Gumbel-Softmax, 64 experts |
| CUDA router kernel | Compiled + tested | 10.4 us/batch, 85-170x vs PyTorch |
| OptiX RT Core pipeline | Validated (Windows) | **19.1 us/batch, 13.4M q/s, 100% accuracy** |
| OptiX 9.0 CoopVec | Integrated | In-shader calibration via Tensor Cores |
| PyTorch extension (zero-copy) | Integrated | 10 us routing, auto kernel selection |
| Ternary expert kernel (POPCOUNT) | Validated | Zero FP multiplications, 0.000038 diff vs FP32 |
| Demo (Qwen 1.5B) | Executed | 51.9 tok/s, 375x less VRAM |
| Inception Engine (4-level 12D) | PPL 185.4 | Only 1.8% worse than GPT-2 baseline |
| Spectral encoding + Snell | Implemented | 88.9% polysemy resolution |
| Calibration export | Working | PyTorch -> FP16 binary (272B affine) + C header |

### OLMoE Distillation -- Main result

We replaced the linear gate of OLMoE-1B-7B (a real 7B-parameter MoE model with 64 experts) with our geometric BVH Router and measured perplexity impact:

| Configuration | PPL | Delta vs baseline | Status |
|---|---|---|---|
| Baseline (OLMoE linear gate) | 7.15 | -- | Reference |
| BVH Router hybrid 3 layers | 7.17 | **+0.4%** | Validated |
| BVH Router hybrid 16 layers | 7.30 | **+2.1%** | Validated |
| BVH Router pure 3 layers (render_eq) | 7.33 | **+2.5%** | Validated |
| Confidence-gated 16 layers (T=0.90) | 8.37 | +17.1% | 69% BVH routing |

### RT Core Benchmark (RTX 5070 Ti, Windows native)

| Mode | Latency (us/batch) | Throughput (M q/s) | Accuracy |
|---|---|---|---|
| AABB sync | 28.5 | 9.0 | 100% |
| AABB async | 37.2 | 6.9 | 100% |
| Triangle sync | 32.5 | 7.9 | 100% |
| **Triangle async** | **19.1** | **13.4** | **100%** |

**~48x speedup** vs PyTorch linear gate (~927 us). GAS memory: 11 KB for 64 experts.

### Routing Speed Comparison

```
RT Core (OptiX triangle async): 19.1 us/batch -> 13.4M q/s
BVH Router (CUDA kernel):       10.4 us/batch -> 24.7M tok/s
Linear gate (PyTorch):         ~927  us/batch -> 48-94x slower
```

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

1. **RT Core Attention (Patent LBS-2026-001):** BVH traversal replaces dense MatMul. O(log N) instead of O(N^2). OptiX 9.0 Cooperative Vectors enable in-shader calibration via Tensor Cores.

2. **Inception Engine (Patent LBS-2026-002):** 4 nested IAS levels encode 12 semantic dimensions using only 3D hardware. Each level is a "dimensional portal" that resets coordinates.

3. **Spectral Routing (Patent LBS-2026-003):** Rays carry a "color" (context vector). Nodes act as prisms (Snell's law) -- the same node routes differently based on context, resolving polysemy without duplicating parameters.

---

## Project Structure

```
spectral-ai/
├── CLAUDE.md              # Architecture reference (for AI agents)
├── LEARNINGS.md           # Decision log, failures, discoveries
├── STATUS.md              # Detailed status with file inventory
├── README.md              # This file
├── CMakeLists.txt         # C++/CUDA build system
│
├── python/                # ~50 files, ~25K lines
│   ├── bvh_router.py          # BVH Router (PyTorch, differentiable)
│   ├── orchestrator.py        # Full pipeline: Router -> Expert -> Output
│   ├── olmoe_bvh_distill.py   # BVH Router distillation from OLMoE gate
│   ├── olmoe_e2e_eval.py      # End-to-end PPL evaluation (multi-layer)
│   ├── eval_hellaswag.py      # HellaSwag downstream evaluation
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
├── include/               # C++ public headers (7 files)
├── src/                   # C++ implementations (3 files)
├── tests/                 # C++ tests and benchmarks (7 files)
├── patents/               # 3 provisional patent drafts
├── paper/                 # Academic paper draft
├── scripts/               # Automation scripts
├── data/                  # Datasets, embeddings (generated, not in git)
└── checkpoints/           # Trained models (generated, not in git)
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

# 4. Evaluate PPL (should give ~7.17 hybrid, +0.4% vs baseline 7.15)
python python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt --max-tokens 50000

# 5. Evaluate HellaSwag (downstream task)
python python/eval_hellaswag.py --model-dir /path/to/olmoe-1b-7b --max-samples 200

# Build C++/CUDA (Windows native with OptiX):
cd build_win
cmake .. -G "Visual Studio 17 2022" -A x64 \
    -DOptiX_INSTALL_DIR="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
cmake --build . --config Release

# Run RT Core benchmark:
Release\rt_router_benchmark.exe ".."
```

---

## Patents

Three provisional patent applications (filed 2026):

| Docket | Title | Innovation |
|---|---|---|
| LBS-2026-001 | RT Core O(log N) Attention | BVH replaces MatMul in attention + in-shader calibration via Cooperative Vectors |
| LBS-2026-002 | Nested IAS for 12D | 4 levels of 3D = 12 dimensions via OptiX instancing |
| LBS-2026-003 | Spectral Routing + Snell | Context-dependent routing without parameter duplication |

---

## License

Proprietary. Patent pending.

## Author

Jordi Silvestre Lopez -- SpectralAI Studio, 2026.
