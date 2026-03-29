#!/usr/bin/env python3
"""
bvh_router_bridge.py — Bridge entre BVHRouter (PyTorch) y CUDABVHRouter (kernel)

Permite:
1. Entrenar el router con PyTorch (backprop, Gumbel-Softmax)
2. Exportar los parámetros al formato del kernel CUDA
3. Rutear en inferencia con el kernel real (105x más rápido)

Uso:
    from bvh_router_bridge import HybridBVHRouter

    router = HybridBVHRouter(cfg, device)
    # Training: usa PyTorch internamente
    router.train()
    result = router(prompt_embedding)

    # Inference: exporta a CUDA kernel y usa eso
    router.eval()
    router.sync_to_cuda()  # exporta parámetros entrenados al kernel
    result = router(prompt_embedding)  # ahora usa el kernel CUDA real

Copyright (c) 2026 LiquidBit Studio — Apache 2.0
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from bvh_router import BVHRouter, RouterConfig, RoutingResult

# Constantes del kernel CUDA (deben coincidir con bvh_router_kernel.cu)
BVH_BF = 4
BVH_LEVELS = 3
BVH_LEAVES = 64
BVH_NODES = 85  # 1 + 4 + 16 + 64
BVH_LEAF_OFFSET = 21  # 1 + 4 + 16
SPEC_DIM = 64

# Try to import CUDA router (ctypes bridge — fallback)
try:
    from bvh_router_cuda import CUDABVHRouter, generate_random_bvh
    HAS_CUDA_ROUTER = True
except (ImportError, FileNotFoundError):
    HAS_CUDA_ROUTER = False

# Try to import PyTorch zero-copy extension (preferred over ctypes)
# JIT-compiled extensions live in ~/.cache/torch_extensions/
HAS_TORCH_EXT = False
try:
    import bvh_router_ext
    HAS_TORCH_EXT = True
except ImportError:
    try:
        import sys as _sys
        _ext_dir = os.path.expanduser("~/.cache/torch_extensions/bvh_router_ext")
        if os.path.isdir(_ext_dir) and _ext_dir not in _sys.path:
            _sys.path.insert(0, _ext_dir)
        import bvh_router_ext
        HAS_TORCH_EXT = True
    except ImportError:
        pass


def _find_lib_path() -> Optional[str]:
    """Find libbvh_router.so in the project."""
    candidates = [
        Path(__file__).parent.parent / "cuda" / "libbvh_router.so",
        Path(__file__).parent.parent / "cuda" / "v5" / "libbvh_router.so",
        Path(__file__).parent / "libbvh_router.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


class HybridBVHRouter(nn.Module):
    """
    Router híbrido: PyTorch para training, kernel CUDA para inference.

    En training:
      - Usa BVHRouter (nn.Module) con Gumbel-Softmax diferenciable
      - Parámetros se actualizan via backprop

    En inference (después de sync_to_cuda()):
      - Exporta parámetros del BVHRouter al formato del kernel
      - Usa CUDABVHRouter con constant memory + CUDA Graphs
      - ~105x más rápido que PyTorch

    Los parámetros entrenables son SIEMPRE del BVHRouter PyTorch.
    El kernel CUDA es solo la ruta de inferencia optimizada.
    """

    def __init__(
        self,
        cfg: RouterConfig,
        device: torch.device = torch.device("cpu"),
        lib_path: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self._device = device

        # PyTorch router (siempre presente — fuente de verdad de parámetros)
        self.pytorch_router = BVHRouter(cfg)

        # CUDA router (lazy init on first sync_to_cuda)
        self._cuda_router = None
        self._cuda_synced = False
        self._torch_ext_synced = False  # PyTorch zero-copy extension
        self._lib_path = lib_path or _find_lib_path()

    def _init_cuda_router(self, batch_size: int = 1):
        """Initialize the CUDA kernel router."""
        if not HAS_CUDA_ROUTER:
            raise RuntimeError(
                "CUDABVHRouter not available. "
                "Compile with: cd cuda/v5 && make"
            )
        if self._lib_path is None:
            raise FileNotFoundError(
                "libbvh_router.so not found. "
                "Compile with: cd cuda/v5 && make"
            )

        self._cuda_router = CUDABVHRouter(
            lib_path=self._lib_path,
            batch_size=max(batch_size, 16),
            use_graph=True,
        )

    def _export_tree_from_pytorch(self) -> dict:
        """
        Extract BVH tree parameters from the PyTorch router and convert
        to the flat format expected by the CUDA kernel.

        Returns dict with: centers, radii, portals, snell_weights, snell_bias
        All arrays are (BVH_NODES, ...) shaped.
        """
        pr = self.pytorch_router

        # Allocate flat arrays for all 85 nodes
        centers = np.zeros((BVH_NODES, 3), dtype=np.float32)
        radii = np.ones(BVH_NODES, dtype=np.float32)
        portals = np.zeros((BVH_NODES, 3, 4), dtype=np.float32)
        snell_w = np.zeros((BVH_NODES, SPEC_DIM), dtype=np.float32)
        snell_b = np.zeros(BVH_NODES, dtype=np.float32)

        # Initialize portals as identity transforms
        for i in range(BVH_NODES):
            portals[i, 0, 0] = 1.0
            portals[i, 1, 1] = 1.0
            portals[i, 2, 2] = 1.0

        # Root node (index 0): identity
        # Level 1 nodes: indices 1..4 (children of root)
        # Level 2 nodes: indices 5..20 (children of L1)
        # Level 3 nodes: indices 21..84 (leaves = experts)

        with torch.no_grad():
            # Level 1: 4 spheres (indices 1-4)
            l1_centers = pr.level1.centers.cpu().numpy()  # (4, 3)
            l1_radii = pr.level1.radii.cpu().numpy()  # (4,)
            l1_portals = pr.portal1.transform.cpu().numpy()  # (4, 3, 4)
            l1_snell_w = pr.refract1.W_dispersion.weight.cpu().numpy()  # (4, spectral_dim)
            l1_snell_b = pr.refract1.W_dispersion.bias.cpu().numpy()  # (4,)

            for k in range(BVH_BF):
                idx = 1 + k  # nodes 1-4
                centers[idx] = l1_centers[k]
                radii[idx] = l1_radii[k]
                portals[idx] = l1_portals[k]
                # Pad or truncate snell weights to SPEC_DIM
                sw = l1_snell_w[k]
                dim = min(len(sw), SPEC_DIM)
                if len(sw) > SPEC_DIM:
                    warnings.warn(
                        f"Spectral dim truncated: model has {len(sw)} but CUDA kernel expects {SPEC_DIM}. "
                        f"Recompile kernel with SPEC_DIM>={len(sw)} for full fidelity."
                    )
                snell_w[idx, :dim] = sw[:dim]
                snell_b[idx] = l1_snell_b[k]

            # Level 2: 16 spheres (indices 5-20)
            l2_centers = pr.level2.centers.cpu().numpy()  # (16, 3)
            l2_radii = pr.level2.radii.cpu().numpy()  # (16,)
            l2_portals = pr.portal2.transform.cpu().numpy()  # (16, 3, 4)
            l2_snell_w = pr.refract2.W_dispersion.weight.cpu().numpy()  # (16, spectral_dim)
            l2_snell_b = pr.refract2.W_dispersion.bias.cpu().numpy()  # (16,)

            for k in range(BVH_BF * BVH_BF):
                idx = 5 + k  # nodes 5-20
                centers[idx] = l2_centers[k]
                radii[idx] = l2_radii[k]
                portals[idx] = l2_portals[k]
                sw = l2_snell_w[k]
                dim = min(len(sw), SPEC_DIM)
                snell_w[idx, :dim] = sw[:dim]
                snell_b[idx] = l2_snell_b[k]

            # Level 3: 64 spheres (indices 21-84 = leaves)
            l3_centers = pr.level3.centers.cpu().numpy()  # (64, 3)
            l3_radii = pr.level3.radii.cpu().numpy()  # (64,)
            # Level 3 has refraction but no portals (leaves)
            l3_snell_w = pr.refract3.W_dispersion.weight.cpu().numpy()  # (64, spectral_dim)
            l3_snell_b = pr.refract3.W_dispersion.bias.cpu().numpy()  # (64,)

            for k in range(BVH_LEAVES):
                idx = BVH_LEAF_OFFSET + k  # nodes 21-84
                centers[idx] = l3_centers[k]
                radii[idx] = l3_radii[k]
                sw = l3_snell_w[k]
                dim = min(len(sw), SPEC_DIM)
                snell_w[idx, :dim] = sw[:dim]
                snell_b[idx] = l3_snell_b[k]

        return {
            "centers": centers,
            "radii": radii,
            "portals": portals,
            "snell_weights": snell_w,
            "snell_bias": snell_b,
        }

    def sync_to_cuda(self, batch_size: int = 1):
        """
        Export trained PyTorch parameters to the CUDA kernel.
        Call this after training, before inference.
        """
        if self._cuda_router is None:
            self._init_cuda_router(batch_size)

        tree = self._export_tree_from_pytorch()
        self._cuda_router.upload_tree(
            tree["centers"],
            tree["radii"],
            tree["portals"],
            tree["snell_weights"],
            tree["snell_bias"],
        )
        self._cuda_synced = True

    def sync_to_torch_ext(self):
        """
        Export trained PyTorch parameters to the bvh_router_ext zero-copy extension.
        This is FASTER than sync_to_cuda (no numpy conversions at route time).
        """
        if not HAS_TORCH_EXT:
            raise RuntimeError(
                "bvh_router_ext not available. "
                "Compile with: python cuda/v5/build_ext.py"
            )

        tree = self._export_tree_from_pytorch()

        # Upload as CPU float32 tensors (extension copies to constant memory)
        bvh_router_ext.upload_tree(
            torch.from_numpy(tree["centers"]).contiguous(),
            torch.from_numpy(tree["radii"]).contiguous(),
            torch.from_numpy(tree["portals"].reshape(BVH_NODES, -1)).contiguous(),
            torch.from_numpy(tree["snell_weights"]).contiguous(),
            torch.from_numpy(tree["snell_bias"]).contiguous(),
        )
        self._torch_ext_synced = True

    def _torch_ext_route(
        self, prompt_embedding: torch.Tensor
    ) -> RoutingResult:
        """
        Route using bvh_router_ext zero-copy extension.
        All data stays on GPU — no numpy conversions.
        """
        B = prompt_embedding.shape[0]
        device = prompt_embedding.device

        with torch.no_grad():
            pos_3d = self.pytorch_router.to_3d(prompt_embedding)       # (B, 3)
            spectral = self.pytorch_router.spectral(prompt_embedding)   # (B, spec_dim)

        # Directions: normalized position (pointing toward center)
        directions = pos_3d.clone()
        norms = torch.clamp(directions.norm(dim=-1, keepdim=True), min=1e-8)
        directions = directions / norms

        # Pad/truncate spectral to SPEC_DIM on GPU (no CPU transfer!)
        spec_dim = spectral.shape[1]
        if spec_dim < SPEC_DIM:
            pad = torch.zeros(B, SPEC_DIM - spec_dim, device=device, dtype=torch.float32)
            spectral_padded = torch.cat([spectral.float(), pad], dim=1)
        elif spec_dim > SPEC_DIM:
            spectral_padded = spectral[:, :SPEC_DIM].float()
        else:
            spectral_padded = spectral.float()

        # Zero-copy route! All tensors stay on CUDA
        expert_ids, scores, confidence, _path = bvh_router_ext.route(
            pos_3d.float().contiguous(),
            directions.float().contiguous(),
            spectral_padded.contiguous(),
        )

        # Already PyTorch tensors on CUDA — no conversion needed
        return RoutingResult(
            expert_id=expert_ids.long(),
            expert_probs=scores,
            route_path=torch.zeros(B, BVH_LEVELS, dtype=torch.long, device=device),
            confidence=confidence,
        )

    def _cuda_route(
        self, prompt_embedding: torch.Tensor
    ) -> RoutingResult:
        """
        Route using the CUDA kernel.
        Handles: prompt → 3D + spectral (PyTorch) → kernel route → RoutingResult
        """
        B = prompt_embedding.shape[0]
        device = prompt_embedding.device

        with torch.no_grad():
            # Use PyTorch router's projection layers
            pos_3d = self.pytorch_router.to_3d(prompt_embedding)  # (B, 3)
            spectral = self.pytorch_router.spectral(prompt_embedding)  # (B, spectral_dim)

        # Convert to numpy for CUDA kernel
        origins = pos_3d.cpu().float().numpy()
        # Initial direction: normalized position (pointing toward center)
        directions = origins.copy()
        norms = np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8)
        directions = directions / norms

        spectral_np = spectral.cpu().float().numpy()
        # Pad/truncate spectral to SPEC_DIM
        if spectral_np.shape[1] < SPEC_DIM:
            pad = np.zeros((B, SPEC_DIM - spectral_np.shape[1]), dtype=np.float32)
            spectral_np = np.concatenate([spectral_np, pad], axis=1)
        elif spectral_np.shape[1] > SPEC_DIM:
            spectral_np = spectral_np[:, :SPEC_DIM]

        # Ensure batch size is compatible
        if B > self._cuda_router._batch_size:
            self._init_cuda_router(B)
            self.sync_to_cuda(B)

        # Route via CUDA kernel!
        expert_ids, scores, path, confidence = self._cuda_router.route(
            origins, directions, spectral_np
        )

        # Convert back to PyTorch tensors
        expert_id = torch.from_numpy(expert_ids).long().to(device)
        expert_probs = torch.from_numpy(scores).float().to(device)
        route_path = torch.from_numpy(path).long().to(device)
        conf = torch.from_numpy(confidence).float().to(device)

        return RoutingResult(
            expert_id=expert_id,
            expert_probs=expert_probs,
            route_path=route_path,
            confidence=conf,
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        hard: bool = False,
    ) -> RoutingResult:
        """
        Route a prompt to an expert.

        In training: uses PyTorch BVHRouter (differentiable Gumbel-Softmax)
        In eval + synced: uses CUDA kernel (105x faster)
        In eval + not synced: falls back to PyTorch router
        """
        if self.training:
            # Training: always use PyTorch (needs gradients)
            return self.pytorch_router(prompt_embedding, hard=hard)

        # Prefer zero-copy PyTorch extension (fastest — no numpy at all)
        if self._torch_ext_synced and HAS_TORCH_EXT:
            return self._torch_ext_route(prompt_embedding)

        # Fallback to ctypes CUDA kernel (numpy conversion overhead)
        if self._cuda_synced and self._cuda_router is not None:
            return self._cuda_route(prompt_embedding)

        # Last resort: PyTorch inference
        return self.pytorch_router(prompt_embedding, hard=True)

    # ── Proxy methods ────────────────────────────────────────────

    def anneal_temperature(self):
        self.pytorch_router.anneal_temperature()

    def reset_expert_counts(self):
        self.pytorch_router.reset_expert_counts()

    def load_balancing_loss(self) -> torch.Tensor:
        return self.pytorch_router.load_balancing_loss()

    @property
    def temperature(self):
        return self.pytorch_router.temperature

    # Access sub-modules for compatibility
    @property
    def to_3d(self):
        return self.pytorch_router.to_3d

    @property
    def spectral(self):
        return self.pytorch_router.spectral

    def is_cuda_active(self) -> bool:
        """Check if any CUDA kernel is being used for routing."""
        if self.training:
            return False
        return (
            self._torch_ext_synced
            or (self._cuda_synced and self._cuda_router is not None)
        )

    def status(self) -> str:
        """Return human-readable status of the router."""
        if self.training:
            return "TRAINING (PyTorch Gumbel-Softmax)"
        if self._torch_ext_synced and HAS_TORCH_EXT:
            return "INFERENCE (PyTorch zero-copy ext — fastest, no numpy)"
        if self._cuda_synced and self._cuda_router is not None:
            return "INFERENCE (CUDA kernel ctypes — 105x optimized)"
        if HAS_TORCH_EXT:
            return "INFERENCE (PyTorch fallback — call sync_to_torch_ext())"
        if HAS_CUDA_ROUTER and self._lib_path:
            return "INFERENCE (PyTorch fallback — call sync_to_cuda())"
        return "INFERENCE (PyTorch only — no CUDA extensions found)"


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HybridBVHRouter — Bridge Test")
    print("=" * 60)

    cfg = RouterConfig(embed_dim=256, n_level1=4, n_level2=4, n_level3=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    router = HybridBVHRouter(cfg, device=device)
    router = router.to(device)

    prompt = torch.randn(4, 256, device=device)

    # Test training mode
    router.train()
    print(f"\nStatus: {router.status()}")
    result = router(prompt)
    print(f"[TRAIN] Experts: {result.expert_id.tolist()}")
    print(f"[TRAIN] Confidence: {[f'{c:.3f}' for c in result.confidence.tolist()]}")

    # Test eval mode (PyTorch fallback)
    router.eval()
    print(f"\nStatus: {router.status()}")
    result = router(prompt)
    print(f"[EVAL/PT] Experts: {result.expert_id.tolist()}")

    # Test CUDA kernel
    lib_path = _find_lib_path()
    if lib_path:
        print(f"\nFound CUDA kernel: {lib_path}")
        try:
            router.sync_to_cuda(batch_size=4)
            print(f"Status: {router.status()}")
            result = router(prompt)
            print(f"[EVAL/CUDA] Experts: {result.expert_id.tolist()}")
            print(f"[EVAL/CUDA] Path: {result.route_path.tolist()}")
            print(f"[EVAL/CUDA] Confidence: {[f'{c:.3f}' for c in result.confidence.tolist()]}")

            # Quick benchmark
            import time
            N = 100
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                _ = router(prompt)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / N * 1e6
            print(f"\n[BENCHMARK] {dt:.1f} μs/batch (batch=4)")
            print(f"            {dt/4:.1f} μs/prompt")
        except Exception as e:
            print(f"[ERROR] CUDA kernel failed: {e}")
    else:
        print("\nlibbvh_router.so not found — skipping ctypes CUDA test")
        print("Compile with: cd cuda/v5 && make")

    # Test PyTorch zero-copy extension (if available)
    if HAS_TORCH_EXT:
        print(f"\n--- Testing bvh_router_ext (zero-copy) ---")
        router2 = HybridBVHRouter(cfg, device=device).to(device).eval()
        try:
            router2.sync_to_torch_ext()
            print(f"Status: {router2.status()}")
            result = router2(prompt)
            print(f"[EVAL/EXT] Experts: {result.expert_id.tolist()}")
            print(f"[EVAL/EXT] Confidence: {[f'{c:.3f}' for c in result.confidence.tolist()]}")

            import time
            N = 1000
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                _ = router2(prompt)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / N * 1e6
            print(f"\n[BENCHMARK EXT] {dt:.1f} μs/batch (batch=4)")
            print(f"                {dt/4:.1f} μs/prompt")
        except Exception as e:
            print(f"[ERROR] torch ext failed: {e}")
    else:
        print("\nbvh_router_ext not found — skipping zero-copy test")
        print("Compile with: python cuda/v5/build_ext.py")
