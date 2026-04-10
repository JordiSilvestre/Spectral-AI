#!/usr/bin/env python3
"""
bvh_router_hybrid.py — Router Hibrido: PyTorch projection + CUDA kernel traversal

Combina lo mejor de ambos mundos:
  - PyTorch: proyeccion embedding -> 3D (diferenciable, cross-platform)
  - CUDA kernel: traversal BVH ultra-rapido (8.84 us vs 1580 us PyTorch)

Requiere n_experts=64 (n_l1=4, n_l2=4, n_l3=4) para coincidir con el kernel.

Uso:
    from bvh_router_hybrid import HybridBVHRouter
    router = HybridBVHRouter(pytorch_router)
    result = router.route(prompt_embedding)  # RoutingResult compatible

Ejecutar en WSL2 para acceder al kernel CUDA. En Windows, usa PyTorch fallback.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from bvh_router import BVHRouter, RoutingResult

# Constantes del kernel (deben coincidir con bvh_router_kernel.cu)
BVH_BF         = 4
BVH_LEVELS     = 3
BVH_LEAVES     = 64
BVH_NODES      = 85   # 1(root) + 4(L1) + 16(L2) + 64(L3)
BVH_LEAF_OFFSET = 21  # 1 + 4 + 16
SPEC_DIM       = 64


def _find_lib(custom_path: Optional[str] = None) -> Optional[str]:
    """Busca libbvh_router.so en rutas conocidas del proyecto."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    root = Path(__file__).parent.parent
    for candidate in [
        root / "cuda" / "v5" / "libbvh_router.so",
        root / "cuda" / "libbvh_router.so",
        Path("cuda/v5/libbvh_router.so"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


class HybridBVHRouter:
    """
    Router BVH hibrido: PyTorch projection + CUDA kernel traversal.

    Pipeline en CUDA mode:
      prompt_emb (B,256) -> to_3d -> pos_3d (B,3)   [PyTorch en GPU]
                         -> spectral -> spec (B,64)  [PyTorch en GPU]
      pos_3d + spec -> CUDA kernel -> expert_id (B,) [8.84 us/batch]

    Pipeline fallback (Windows/sin kernel):
      prompt_emb -> BVHRouter.forward() [PyTorch puro, 1580 us]

    Requiere que el PyTorch router tenga n_experts=64 (4x4x4).
    """

    REQUIRED_EXPERTS = BVH_LEAVES  # 64

    def __init__(self, pytorch_router: BVHRouter,
                 lib_path: Optional[str] = None,
                 device: str = "cuda"):
        self.router = pytorch_router
        self.device = torch.device(device)
        self._cuda_ok = False

        cfg = pytorch_router.cfg
        if cfg.n_experts != self.REQUIRED_EXPERTS:
            raise ValueError(
                f"HybridBVHRouter necesita n_experts={self.REQUIRED_EXPERTS} "
                f"(4x4x4). Actual: {cfg.n_experts}. "
                f"Configura OrchestratorConfig(n_level1=4, n_level2=4, n_level3=4)."
            )

        found = _find_lib(lib_path)
        if found is None:
            print("[Hybrid] libbvh_router.so no encontrado — fallback PyTorch")
            return

        try:
            from bvh_router_cuda import CUDABVHRouter
            self._cuda_router = CUDABVHRouter(
                lib_path=found,
                batch_size=256,
                use_graph=False,
            )
            self._pack_and_upload_tree()
            self._cuda_ok = True
            print(f"[Hybrid] Kernel CUDA cargado: {found}")
            print(f"[Hybrid] Latencia kernel: ~8.84 us/batch (179x vs PyTorch)")
        except Exception as e:
            print(f"[Hybrid] Kernel no disponible ({e}) — fallback PyTorch")

    def _pack_and_upload_tree(self):
        """
        Empaqueta parametros del router PyTorch en array plano de 85 nodos
        y los sube a constant memory del kernel CUDA.

        Layout:
          node[0]     = root (dummy, sin datos reales)
          node[1:5]   = L1 (4 dominios)
          node[5:21]  = L2 (16 subdominios)
          node[21:85] = L3 (64 hojas = expertos)
        """
        centers = np.zeros((BVH_NODES, 3),         dtype=np.float32)
        radii   = np.ones(BVH_NODES,               dtype=np.float32) * 0.5
        portals = np.zeros((BVH_NODES, 3, 4),      dtype=np.float32)
        snell_w = np.zeros((BVH_NODES, SPEC_DIM),  dtype=np.float32)
        snell_b = np.zeros(BVH_NODES,              dtype=np.float32)

        # Portales inicializados como identidad
        for i in range(BVH_NODES):
            for j in range(3):
                portals[i, j, j] = 1.0

        with torch.no_grad():
            r = self.router

            # --- L1: nodos 1:5 ---
            centers[1:5] = r.level1.centers.cpu().numpy()        # (4, 3)
            radii[1:5]   = r.level1.radii.cpu().numpy()          # (4,)
            portals[1:5] = r.portal1.transform.cpu().numpy()     # (4, 3, 4)
            snell_w[1:5] = r.refract1.W_dispersion.weight.cpu().numpy()  # (4, 64)
            snell_b[1:5] = r.refract1.W_dispersion.bias.cpu().numpy()    # (4,)

            # --- L2: nodos 5:21 ---
            centers[5:21] = r.level2.centers.cpu().numpy()       # (16, 3)
            radii[5:21]   = r.level2.radii.cpu().numpy()         # (16,)
            portals[5:21] = r.portal2.transform.cpu().numpy()    # (16, 3, 4)
            snell_w[5:21] = r.refract2.W_dispersion.weight.cpu().numpy() # (16, 64)
            snell_b[5:21] = r.refract2.W_dispersion.bias.cpu().numpy()   # (16,)

            # --- L3: nodos 21:85 (hojas, sin portal) ---
            centers[21:85] = r.level3.centers.cpu().numpy()      # (64, 3)
            radii[21:85]   = r.level3.radii.cpu().numpy()        # (64,)
            snell_w[21:85] = r.refract3.W_dispersion.weight.cpu().numpy() # (64, 64)
            snell_b[21:85] = r.refract3.W_dispersion.bias.cpu().numpy()   # (64,)

        self._cuda_router.upload_tree(centers, radii, portals, snell_w, snell_b)

    @torch.no_grad()
    def _route_cuda(self, prompt_embedding: torch.Tensor) -> RoutingResult:
        """Routing con CUDA kernel. Requiere Linux/WSL2."""
        r = self.router

        # Proyeccion PyTorch: embedding -> 3D + espectral
        pos_3d   = r.to_3d(prompt_embedding).cpu().numpy()   # (B, 3)
        spectral = r.spectral(prompt_embedding).cpu().numpy() # (B, 64)

        # Direccion del rayo: normalizar pos_3d
        norms      = np.linalg.norm(pos_3d, axis=-1, keepdims=True).clip(min=1e-8)
        directions = pos_3d / norms

        # CUDA kernel: 8.84 us
        expert_ids, scores, path_arr, confidence = self._cuda_router.route(
            pos_3d, directions, spectral
        )

        dev = prompt_embedding.device
        return RoutingResult(
            expert_id   = torch.from_numpy(expert_ids).long().to(dev),
            expert_probs= torch.from_numpy(scores).float().to(dev),
            route_path  = torch.from_numpy(path_arr).long().to(dev),
            confidence  = torch.from_numpy(confidence).float().to(dev),
        )

    def route(self, prompt_embedding: torch.Tensor,
              hard: bool = True) -> RoutingResult:
        """
        Routing hibrido: CUDA si disponible, PyTorch si no.
        API compatible con BVHRouter.forward().
        """
        if self._cuda_ok:
            return self._route_cuda(prompt_embedding)
        return self.router(prompt_embedding, hard=hard)

    @property
    def cuda_available(self) -> bool:
        return self._cuda_ok

    def __call__(self, prompt_embedding: torch.Tensor,
                 hard: bool = True) -> RoutingResult:
        return self.route(prompt_embedding, hard=hard)
