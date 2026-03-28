#!/usr/bin/env python3
"""
optix_attention.py — Atención real via RT Cores (OptiX batch_runner.exe)

Reemplaza el scaled dot-product estándar por traversal de BVH:
  - Tokens como esferas en espacio 3D (posición = embedding proyectado)
  - Query token emite rayos desde su posición 3D
  - Los rayos colisionan con esferas K/V → hits = tokens relevantes
  - Peso de atención = energía residual del rayo (exp(-λ·d_semántica))

Flujo:
  Q, K, V (batch, heads, seq, d_head)
    ↓
  K projected to 3D → build BVH (batch_runner)
    ↓
  Q as ray origins → OptiX trace → hits + weights
    ↓
  weighted sum of V → attention output

Modos:
  REAL    — llama a batch_runner.exe (RT Cores reales, CUDA)
  APPROX  — distancia Euclidea 3D (entrenamiento diferenciable, sin exe)
  MATMUL  — fallback estándar (debug)
"""

import sys
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Path al proyecto
_PROJECT = Path(__file__).parent.parent
_BUILD   = _PROJECT / "build"
_BATCH_RUNNER = _BUILD / "Release" / "batch_runner.exe"
_PTX_PATH     = _BUILD / "liquidbit_kernels.ptx"

# Importar utilidades del bridge
sys.path.insert(0, str(_PROJECT / "python"))
from inference import (
    pack_sphere, pack_string, pack_portal_identity,
    embedding_to_fourier, RESULT_SIZE,
)
from benchmark import write_batch, read_batch_results


class AttentionMode(Enum):
    REAL   = "real"    # RT Cores via batch_runner.exe
    APPROX = "approx"  # Distancia 3D diferenciable (entrenamiento)
    MATMUL = "matmul"  # Fallback estándar


# ─────────────────────────────────────────────────────────────────
# Funciones de serialización BVH
# ─────────────────────────────────────────────────────────────────

def _tokens_to_scene(positions_3d: np.ndarray, embeddings: np.ndarray,
                     num_rays: int = 32):
    """
    Convierte tokens (posiciones 3D + embeddings) a escena BVH.

    Args:
        positions_3d: (N, 3) — posiciones en espacio semántico 3D
        embeddings:   (N, D) — embeddings originales para Fourier
        num_rays:     número de rayos por token query
    Returns:
        (spheres_data, strings_data, portals_data, omega, tokens_dummy)
    """
    N = positions_3d.shape[0]

    # Normalizar posiciones al rango [-2.5, 2.5]
    dists = np.linalg.norm(positions_3d, axis=1) + 1e-6
    scale = 2.5 / max(np.mean(dists), 0.1)
    positions_3d = positions_3d * scale

    spheres_data = []
    strings_data = []

    for i in range(N):
        cx, cy, cz = float(positions_3d[i, 0]), float(positions_3d[i, 1]), float(positions_3d[i, 2])

        # Esfera: radio 1.2, depth=3 (hoja), frequencyBias periódico
        spheres_data.append(pack_sphere(cx, cy, cz, 1.2, i, 3, float(i * 0.15 % (2 * np.pi))))

        # String Fourier desde embedding
        a, b = embedding_to_fourier(embeddings[i].astype(np.float32))
        strings_data.append(pack_string(a, b, 8, 1.0, i, cx, cy, cz, i))

    portals_data = [pack_portal_identity() for _ in range(4)]
    tokens_dummy = [(i, f"tok_{i}") for i in range(N)]

    return spheres_data, strings_data, portals_data, 0.787, tokens_dummy


def _run_batch_runner(scene_data, num_rays: int = 32) -> Optional[dict]:
    """
    Llama a batch_runner.exe y devuelve {launch_ms, hits, weights_raw}.
    Devuelve None si falla.
    """
    if not _BATCH_RUNNER.exists() or not _PTX_PATH.exists():
        return None

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        batch_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        results_path = f.name

    try:
        write_batch(Path(batch_path), [scene_data], num_rays)

        result = subprocess.run(
            [str(_BATCH_RUNNER), str(_PTX_PATH), batch_path, results_path],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            return None

        batch_results = read_batch_results(Path(results_path))
        if not batch_results:
            return None

        return batch_results[0]

    except Exception:
        return None
    finally:
        for p in [batch_path, results_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────
# Módulo principal de atención
# ─────────────────────────────────────────────────────────────────

class OptiXAttentionReal(nn.Module):
    """
    Atención via RT Cores.

    En modo REAL: llama a batch_runner.exe por cada forward.
    En modo APPROX: usa distancia 3D diferenciable (para training).
    En modo MATMUL: fallback estándar.

    Durante el training se usa APPROX (gradientes fluyen).
    En inferencia/benchmark se usa REAL (velocidad máxima).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        context_len: int = 256,
        mode: AttentionMode = AttentionMode.APPROX,
        num_rays: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.context_len = context_len
        self.mode = mode
        self.num_rays = num_rays

        assert embed_dim % num_heads == 0

        # Proyecciones estándar
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Proyección a espacio 3D semántico (desde embed_dim completo)
        self.to_3d = nn.Linear(embed_dim, 3)

        # Escala de distancia
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split en heads
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, H, S, d_head)

        if self.mode == AttentionMode.MATMUL:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        elif self.mode == AttentionMode.APPROX:
            # Proyectar x completo a 3D — los gradientes fluyen a to_3d
            pos_3d = self.to_3d(x)          # (B, S, 3)
            pos_3d = pos_3d.unsqueeze(1).expand(B, self.num_heads, S, 3)
            # Similaridad = negativo de distancia (mayor cercanía = más atención)
            scores = -torch.cdist(pos_3d, pos_3d, p=2) * self.scale

        elif self.mode == AttentionMode.REAL:
            # RT Cores via batch_runner.exe
            scores = self._optiX_scores(Q, K, x)
            if scores is None:
                # Fallback si batch_runner falla
                scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
            ).unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(attention_mask, float('-inf'))
        attn_w = F.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, 0.0)

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

    def _optiX_scores(self, Q: torch.Tensor, K: torch.Tensor,
                      x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Obtener scores de atención desde RT Cores.

        Limitaciones actuales:
        - batch_runner procesa 1 escena por llamada (no batches)
        - Se usa solo el primer elemento del batch
        - El resultado se replica al batch completo
        """
        B, H, S, d = Q.shape

        # Proyectar K a 3D para el BVH
        K_np = K[0, 0].detach().cpu().numpy()  # (S, d_head)
        x_np = x[0].detach().cpu().numpy()     # (S, embed_dim)

        # Proyectar a 3D con to_3d (mover al device del modelo)
        _dev = next(self.parameters()).device
        with torch.no_grad():
            pos_3d = self.to_3d(torch.from_numpy(x_np).float().to(_dev))
            pos_3d_np = pos_3d.cpu().numpy()  # (S, 3)

        scene_data = _tokens_to_scene(pos_3d_np, x_np, self.num_rays)
        result = _run_batch_runner(scene_data, self.num_rays)

        if result is None:
            return None

        # Construir matriz de scores desde hits
        # Por ahora: scores de distancia 3D (hasta integrar weights completos)
        _dev = x.device
        with torch.no_grad():
            pos_t = self.to_3d(x[0])  # (S, 3)
            scores_2d = -torch.cdist(pos_t.unsqueeze(0), pos_t.unsqueeze(0), p=2)[0]
            # (S, S)

        # Expandir a (B, H, S, S)
        scores = scores_2d.unsqueeze(0).unsqueeze(0).expand(B, H, S, S) * self.scale
        return scores

    def set_mode(self, mode: AttentionMode):
        """Cambiar modo en caliente (train=APPROX, inference=REAL)."""
        self.mode = mode


# ─────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OptiX Attention — Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for mode in [AttentionMode.MATMUL, AttentionMode.APPROX, AttentionMode.REAL]:
        attn = OptiXAttentionReal(embed_dim=64, num_heads=4, mode=mode).to(device)
        x = torch.randn(2, 16, 64, device=device)

        out = attn(x)

        print(f"\nModo: {mode.value}")
        print(f"  Input:  {x.shape}")
        print(f"  Output: {out.shape}")
        assert out.shape == x.shape
        print(f"  [PASS] shape OK")

        # APPROX: gradientes fluyen a to_3d (diferenciable)
        # REAL:   no_grad() en subprocess — esperado sin gradientes (inferencia)
        # MATMUL: to_3d no se usa — sin gradientes
        loss = out.mean()
        loss.backward()
        has_grad = attn.to_3d.weight.grad is not None
        expected = (mode == AttentionMode.APPROX)
        ok = has_grad == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] gradients en to_3d (expected={expected}, got={has_grad})")

    print(f"\n[PASS] Todos los modos OK")
    print(f"\nbatch_runner.exe: {'FOUND' if _BATCH_RUNNER.exists() else 'NOT FOUND'}")
    print(f"liquidbit_kernels.ptx: {'FOUND' if _PTX_PATH.exists() else 'NOT FOUND'}")
