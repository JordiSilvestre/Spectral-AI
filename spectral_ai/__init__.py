"""
spectral_ai — O(log N) MoE expert routing via BVH traversal on NVIDIA RT Cores.

Install
-------
    pip install -e .                  # development
    pip install spectral-ai           # release (PyPI, coming soon)

Quick start
-----------
    from spectral_ai import BVHRouter, RouterConfig, HybridBVHRouter, ExpertLRUCache

    # 1. Configure the router
    cfg = RouterConfig(
        embed_dim=2048,   # must match your model's hidden size
        n_level1=4,       # top-level domains
        n_level2=4,       # sub-domains per domain
        n_level3=4,       # experts per sub-domain  →  4×4×4 = 64 experts total
    )

    # 2. Pure PyTorch router (training + fallback inference)
    router = BVHRouter(cfg)
    result = router(hidden_state)          # RoutingResult(expert_id, expert_probs, ...)

    # 3. Hybrid router (auto-selects CUDA kernel > PyTorch extension > pure PyTorch)
    router = HybridBVHRouter(cfg, device="cuda")
    router.eval()
    router.sync_to_cuda()                  # export trained weights to CUDA kernel
    result = router(hidden_state)          # uses fastest available backend

    # 4. Expert LRU cache (keep only top-k experts in VRAM)
    cache = ExpertLRUCache(expert_modules, max_gpu_slots=8, device="cuda")
    expert = cache.get(expert_id)

Public API
----------
    BVHRouter        — PyTorch BVH router (differentiable, Gumbel-Softmax training)
    RouterConfig     — Hyperparameter container for BVHRouter
    RoutingResult    — NamedTuple returned by BVHRouter.forward()
    HybridBVHRouter  — Auto-selects fastest available backend at inference
    ExpertLRUCache   — LRU cache for expert modules (minimises VRAM footprint)

Author
------
    Jordi Silvestre Lopez, 2026
    Apache 2.0 — https://github.com/JordiSilvestre/Spectral-AI
"""

import sys
from pathlib import Path

# Add python/ to path so package imports work without moving files
_python_dir = Path(__file__).parent.parent / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from bvh_router import BVHRouter, RouterConfig, RoutingResult
from bvh_router_bridge import HybridBVHRouter
from expert_lru_cache import ExpertLRUCache

__all__ = [
    "BVHRouter",
    "RouterConfig",
    "RoutingResult",
    "HybridBVHRouter",
    "ExpertLRUCache",
]

__version__ = "0.1.0"
__author__  = "Jordi Silvestre Lopez"
