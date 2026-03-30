#!/usr/bin/env python3
"""
benchmark_e2e_final.py — Benchmark definitivo: PyTorch vs CUDA Extension
SpectralAI v5.0 "Orchestrator"

Compara:
  1. PyTorch routing puro (BVHRouter.forward)
  2. CUDA Extension zero-copy (bvh_router_ext)
  3. Orchestrator completo con cada routing

Ejecutar desde WSL2:
  cd /tmp/spectral
  source /home/jordi/liquidbit_venv/bin/activate
  python python/benchmark_e2e_final.py
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Importar módulos del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bvh_router import BVHRouter, RouterConfig, RoutingResult
from orchestrator import SpectralAIOrchestrator, OrchestratorConfig

# Intentar importar extensión CUDA
try:
    import bvh_router_ext
    HAS_EXT = True
    print("bvh_router_ext: CARGADO (zero-copy)")
except ImportError:
    # Intentar desde el build cache de torch
    ext_dir = os.path.expanduser("~/.cache/torch_extensions/bvh_router_ext")
    if os.path.isdir(ext_dir):
        sys.path.insert(0, ext_dir)
        try:
            import bvh_router_ext
            HAS_EXT = True
            print(f"bvh_router_ext: CARGADO desde {ext_dir}")
        except ImportError:
            HAS_EXT = False
            print("bvh_router_ext: NO DISPONIBLE (build cache existe pero import falla)")
    else:
        HAS_EXT = False
        print("bvh_router_ext: NO DISPONIBLE")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Utilidades
# ============================================================================

BVH_NODES = 85
BVH_LEAVES = 64
SPEC_DIM = 64
WARMUP = 50
N_ITER = 500


def timer_ms(fn, warmup=WARMUP, iters=N_ITER):
    """Medir latencia media en ms con warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000  # ms


def pack_bvh_tree(router: BVHRouter):
    """Extraer parámetros del router PyTorch y formatear para la extensión CUDA."""
    with torch.no_grad():
        centers = torch.zeros(BVH_NODES, 3, dtype=torch.float32)
        radii = torch.ones(BVH_NODES, dtype=torch.float32) * 0.5
        portals = torch.zeros(BVH_NODES, 12, dtype=torch.float32)
        snell_w = torch.zeros(BVH_NODES, SPEC_DIM, dtype=torch.float32)
        snell_b = torch.zeros(BVH_NODES, dtype=torch.float32)

        # Nodo raíz: dummy
        # Nivel 1: nodos 1-4
        l1_centers = router.level1.centers.data.cpu()  # (4, 3)
        l1_radii = router.level1.log_radii.data.exp().cpu()  # (4,)
        for i in range(4):
            centers[1 + i] = l1_centers[i]
            radii[1 + i] = l1_radii[i]

        # Nivel 2: nodos 5-20
        l2_centers = router.level2.centers.data.cpu()  # (16, 3)
        l2_radii = router.level2.log_radii.data.exp().cpu()  # (16,)
        for i in range(16):
            centers[5 + i] = l2_centers[i]
            radii[5 + i] = l2_radii[i]

        # Nivel 3: nodos 21-84
        l3_centers = router.level3.centers.data.cpu()  # (64, 3)
        l3_radii = router.level3.log_radii.data.exp().cpu()  # (64,)
        for i in range(64):
            centers[21 + i] = l3_centers[i]
            radii[21 + i] = l3_radii[i]

        # Portales: identidad por defecto
        for i in range(BVH_NODES):
            portals[i, 0] = 1.0
            portals[i, 5] = 1.0
            portals[i, 10] = 1.0

        # Extraer portales reales de level1/level2
        if hasattr(router, 'portal1') and hasattr(router.portal1, 'transforms'):
            p1 = router.portal1.transforms.data.cpu()  # (4, 3, 4)
            for i in range(min(4, p1.shape[0])):
                portals[1 + i] = p1[i].reshape(12)

        if hasattr(router, 'portal2') and hasattr(router.portal2, 'transforms'):
            p2 = router.portal2.transforms.data.cpu()  # (16, 3, 4)
            for i in range(min(16, p2.shape[0])):
                portals[5 + i] = p2[i].reshape(12)

        # Snell weights de refraction layers
        for level_idx, (refract, offset, count) in enumerate([
            (router.refract1, 1, 4),
            (router.refract2, 5, 16),
            (router.refract3, 21, 64),
        ]):
            if hasattr(refract, 'dispersion') and hasattr(refract.dispersion, 'weight'):
                w = refract.dispersion.weight.data.cpu()  # (count, SPEC_DIM)
                b = refract.dispersion.bias.data.cpu() if refract.dispersion.bias is not None else torch.zeros(count)
                for i in range(min(count, w.shape[0])):
                    snell_w[offset + i, :w.shape[1]] = w[i]
                    snell_b[offset + i] = b[i]

        return centers, radii, portals, snell_w, snell_b


# ============================================================================
# Benchmark 1: Routing puro
# ============================================================================

def benchmark_routing():
    print("\n" + "=" * 70)
    print("BENCHMARK 1: ROUTING PURO (sin backbone)")
    print("=" * 70)

    cfg = RouterConfig(
        embed_dim=256,
        spectral_dim=64,
        n_level1=4, n_level2=4, n_level3=4,
    )
    router = BVHRouter(cfg).to(DEVICE).eval()

    # Preparar extensión CUDA
    if HAS_EXT:
        centers, radii, portals, snell_w, snell_b = pack_bvh_tree(router)
        bvh_router_ext.upload_tree(centers, radii, portals, snell_w, snell_b)

    batch_sizes = [1, 32, 128, 256]
    print(f"\n{'Batch':>6} | {'PyTorch (ms)':>14} | {'CUDA Ext (ms)':>14} | {'Speedup':>8}")
    print("-" * 60)

    for B in batch_sizes:
        x = torch.randn(B, 256, device=DEVICE)

        # PyTorch routing
        def pytorch_route():
            with torch.no_grad():
                router(x, hard=True)
        pt_ms = timer_ms(pytorch_route)

        # CUDA Extension routing
        cuda_ms = float('nan')
        speedup = '-'
        if HAS_EXT:
            with torch.no_grad():
                pos_3d = router.to_3d(x)
                spectral = router.spectral(x)

            origins = pos_3d.contiguous()
            # Dirección = normalización de la posición 3D
            directions = torch.nn.functional.normalize(pos_3d, dim=-1).contiguous()
            spec = spectral.contiguous()

            def cuda_route():
                bvh_router_ext.route(origins, directions, spec)
            cuda_ms = timer_ms(cuda_route)
            speedup = f"{pt_ms / cuda_ms:.1f}x"

        print(f"{B:>6} | {pt_ms:>14.3f} | {cuda_ms:>14.3f} | {speedup:>8}")

    return router


# ============================================================================
# Benchmark 2: Orchestrator completo
# ============================================================================

def benchmark_orchestrator(router):
    print("\n" + "=" * 70)
    print("BENCHMARK 2: ORCHESTRATOR COMPLETO (routing + backbone)")
    print("=" * 70)

    orch_cfg = OrchestratorConfig(
        vocab_size=5000,
        router_embed_dim=256,
        expert_embed_dim=128,
        n_level1=4, n_level2=4, n_level3=4,
        expert_layers=2,
        expert_heads=4,
        context_len=128,
        spectral_dim=64,
    )
    dev = torch.device(DEVICE)
    model = SpectralAIOrchestrator(orch_cfg, device=dev).to(dev).eval()

    # Preparar extensión CUDA con pesos del modelo
    if HAS_EXT:
        centers, radii, portals, snell_w, snell_b = pack_bvh_tree(model.router)
        bvh_router_ext.upload_tree(centers, radii, portals, snell_w, snell_b)

    batch_sizes = [1, 32]
    seq_len = 64

    print(f"\n{'Batch':>6} | {'PyTorch (ms)':>14} | {'CUDA Ext (ms)':>14} | {'Speedup':>8}")
    print("-" * 60)

    for B in batch_sizes:
        tokens = torch.randint(0, 5000, (B, seq_len), device=DEVICE)

        # PyTorch puro
        def pytorch_orch():
            with torch.no_grad():
                model(tokens)
        pt_ms = timer_ms(pytorch_orch, warmup=20, iters=200)

        # Con CUDA Extension para el routing
        cuda_ms = float('nan')
        speedup = '-'
        if HAS_EXT:
            # Parchear el routing para usar la extensión
            original_forward = model.router.forward

            def cuda_router_forward(prompt_embedding, hard=False):
                with torch.no_grad():
                    pos_3d = model.router.to_3d(prompt_embedding)
                    spectral = model.router.spectral(prompt_embedding)

                origins = pos_3d.contiguous()
                directions = torch.nn.functional.normalize(pos_3d, dim=-1).contiguous()
                spec = spectral.contiguous()

                route_out = bvh_router_ext.route(
                    origins, directions, spec)
                if len(route_out) == 4:
                    expert_ids, scores, confidence, _path = route_out
                else:
                    expert_ids, scores, confidence = route_out

                # Convertir a RoutingResult compatible
                B_local = prompt_embedding.shape[0]
                expert_probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
                route_path = torch.zeros(B_local, 3, dtype=torch.long, device=DEVICE)

                return RoutingResult(
                    expert_id=expert_ids,
                    expert_probs=expert_probs,
                    route_path=route_path,
                    confidence=confidence,
                )

            model.router.forward = cuda_router_forward

            def cuda_orch():
                with torch.no_grad():
                    model(tokens)
            cuda_ms = timer_ms(cuda_orch, warmup=20, iters=200)
            speedup = f"{pt_ms / cuda_ms:.2f}x"

            # Restaurar
            model.router.forward = original_forward

        print(f"{B:>6} | {pt_ms:>14.3f} | {cuda_ms:>14.3f} | {speedup:>8}")


# ============================================================================
# Benchmark 3: Latencia micro del kernel
# ============================================================================

def benchmark_kernel_micro():
    print("\n" + "=" * 70)
    print("BENCHMARK 3: LATENCIA MICRO DEL KERNEL CUDA")
    print("=" * 70)

    if not HAS_EXT:
        print("  (extensión no disponible, saltando)")
        return

    # Crear árbol aleatorio
    centers = torch.randn(BVH_NODES, 3, dtype=torch.float32)
    radii = torch.ones(BVH_NODES, dtype=torch.float32) * 0.5
    portals = torch.zeros(BVH_NODES, 12, dtype=torch.float32)
    for i in range(BVH_NODES):
        portals[i, 0] = 1.0; portals[i, 5] = 1.0; portals[i, 10] = 1.0
    snell_w = torch.randn(BVH_NODES, SPEC_DIM, dtype=torch.float32) * 0.1
    snell_b = torch.randn(BVH_NODES, dtype=torch.float32) * 0.1
    bvh_router_ext.upload_tree(centers, radii, portals, snell_w, snell_b)

    batch_sizes = [1, 32, 128, 256, 512, 1024]
    print(f"\n{'Batch':>6} | {'Latencia (us)':>14} | {'Throughput (M/s)':>16}")
    print("-" * 45)

    for B in batch_sizes:
        origins = torch.randn(B, 3, device="cuda")
        directions = torch.randn(B, 3, device="cuda")
        spectral = torch.randn(B, SPEC_DIM, device="cuda")

        def fn():
            bvh_router_ext.route(origins, directions, spectral)

        ms = timer_ms(fn, warmup=100, iters=2000)
        us = ms * 1000
        throughput = B / (ms / 1000) / 1e6
        print(f"{B:>6} | {us:>14.1f} | {throughput:>16.2f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"\nPyTorch {torch.__version__}")
    if DEVICE == "cuda":
        print(f"CUDA Compute: {torch.cuda.get_device_capability(0)}")

    router = benchmark_routing()
    benchmark_orchestrator(router)
    benchmark_kernel_micro()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETADO")
    print("=" * 70)
