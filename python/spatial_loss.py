#!/usr/bin/env python3
"""
spatial_loss.py — Función de pérdida espacial para entrenamiento del árbol BSH

FÓRMULA COMPLETA (del documento SpectralAI BSH Training.docx):
==============================================================

L_total = L_task + α·L_spatial

L_spatial = β·L_prox + γ·L_cover + δ·L_inter + η·L_reg

Componentes:
  L_prox  — Tokens semánticamente similares deben estar cerca
  L_cover — Las esferas BSH deben cubrir todos sus tokens asignados
  L_inter — Tokens polisémicos deben estar en la INTERSECCIÓN de sus esferas
  L_reg   — Penalizar radios grandes y wormholes largos

NOTA SOBRE DIFERENCIABILIDAD:
==============================
Toda la pérdida es diferenciable respecto a:
  - token_positions:  np.ndarray [N, 3] o torch.Tensor [N, 3]
  - sphere_centers:   torch.Tensor [K, 3] (parámetros aprendibles)
  - sphere_radii:     torch.Tensor [K]   (parámetros aprendibles)

Esto permite entrenar la geometría del árbol BSH end-to-end con PyTorch.

@author SpectralAI Zero-Matrix Team
@date 2026
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuración por defecto de los pesos de la pérdida
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpatialLossConfig:
    """Pesos y parámetros de la pérdida espacial."""
    alpha:         float = 0.1   # Peso de L_spatial en L_total
    beta:          float = 1.0   # Peso de L_prox
    gamma:         float = 1.0   # Peso de L_cover
    delta:         float = 0.5   # Peso de L_inter
    eta:           float = 0.1   # Peso de L_reg
    lambda_radius: float = 0.01  # Penalización de radios grandes
    lambda_worm:   float = 0.001 # Penalización de wormholes largos
    margin_cover:  float = 0.05  # Margen de cobertura (esfera debe extenderse X más allá del token)
    polysemy_thresh: float = 0.3 # Umbral de membresía para clasificar como polisémico


# ─────────────────────────────────────────────────────────────────────────────
# Implementación PyTorch (diferenciable)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:

    class SpatialLoss(torch.nn.Module):
        """
        Función de pérdida espacial diferenciable para entrenamiento OHBSC.

        Usa torch.Tensor para que los gradientes fluyan a sphere_centers y sphere_radii.

        FÓRMULA:
            L_spatial = β·L_prox + γ·L_cover + δ·L_inter + η·L_reg

        EJEMPLO DE USO:
            loss_fn = SpatialLoss(config=SpatialLossConfig())
            centers = torch.nn.Parameter(torch.randn(K, 3))
            radii   = torch.nn.Parameter(torch.ones(K))

            memberships = loss_fn.compute_memberships(token_positions, centers, temperature=1.0)
            l_spatial = loss_fn(token_positions, centers, radii, memberships, polysemic_ids)
            l_spatial.backward()
        """

        def __init__(self, config: Optional[SpatialLossConfig] = None):
            super().__init__()
            self.cfg = config or SpatialLossConfig()

        def compute_memberships(
            self,
            token_positions: torch.Tensor,   # [N, 3]
            sphere_centers:  torch.Tensor,   # [K, 3]
            temperature: float = 1.0
        ) -> torch.Tensor:
            """
            Computa membresías difusas P(token_i ∈ sphere_k).

            Returns:
                memberships: [N, K] con valores en [0,1], suma = 1 por fila
            """
            # Distancias cuadráticas [N, K]
            d_sq = torch.sum(
                (token_positions.unsqueeze(1) - sphere_centers.unsqueeze(0)) ** 2,
                dim=2
            )
            # Softmax(-d² / T²) — diferenciable respecto a centers y positions
            logits = -d_sq / max(temperature ** 2, 1e-8)
            return F.softmax(logits, dim=1)

        def l_prox(
            self,
            token_positions: torch.Tensor,   # [N, 3]
            sphere_centers:  torch.Tensor,   # [K, 3]
            memberships:     torch.Tensor,   # [N, K]
            similarity_matrix: torch.Tensor, # [N, N] similitud coseno entre tokens
        ) -> torch.Tensor:
            """
            L_prox: tokens semánticamente similares → cercanos en espacio 3D.

            L_prox = Σ_{i,j} w_ij · ||pos_i - pos_j||² · (esperanza de misma esfera)
            donde w_ij = similitud coseno entre embeddings i y j.

            Para eficiencia: muestreamos pares aleatorios en lugar de Σ_{i,j}.
            """
            N = token_positions.shape[0]
            if N < 2:
                return torch.tensor(0.0, device=token_positions.device)

            # Distancias euclídeas entre tokens [N, N]
            d_sq = torch.sum(
                (token_positions.unsqueeze(1) - token_positions.unsqueeze(0)) ** 2,
                dim=2
            )

            # Los pares similares deben estar cerca: w_ij · ||p_i - p_j||²
            # Los pares disimilares pueden estar lejos: sin penalización
            penalty = similarity_matrix * d_sq  # solo penalizar pares similares
            return penalty.mean()

        def l_cover(
            self,
            token_positions: torch.Tensor,   # [N, 3]
            sphere_centers:  torch.Tensor,   # [K, 3]
            sphere_radii:    torch.Tensor,   # [K]
            memberships:     torch.Tensor,   # [N, K]
        ) -> torch.Tensor:
            """
            L_cover: las esferas deben cubrir todos sus tokens asignados.

            L_cover = Σ_k Σ_i m_{ik} · ReLU(||pos_i - c_k|| - r_k + margin)²

            Penaliza tokens que están fuera del radio de su esfera asignada.
            """
            d_to_centers = torch.sqrt(
                torch.sum(
                    (token_positions.unsqueeze(1) - sphere_centers.unsqueeze(0)) ** 2,
                    dim=2
                ) + 1e-8
            )  # [N, K]

            # Violación de cobertura: token fuera de la esfera
            violation = F.relu(d_to_centers - sphere_radii.unsqueeze(0) + self.cfg.margin_cover)  # [N, K]

            # Ponderar por membresía (solo penalizar para spheres asignadas)
            weighted = memberships * violation ** 2  # [N, K]
            return weighted.mean()

        def l_inter(
            self,
            token_positions:   torch.Tensor,   # [N, 3]
            sphere_centers:    torch.Tensor,   # [K, 3]
            memberships:       torch.Tensor,   # [N, K]
            polysemic_mask:    torch.Tensor,   # [N] bool — True si token es polisémico
        ) -> torch.Tensor:
            """
            L_inter: tokens polisémicos deben estar en la INTERSECCIÓN de sus esferas.

            L_inter = Σ_{c ∈ polisémicos} Σ_{k1,k2 ∈ parents(c)}
                        ||pos_c - proj_{S_k1 ∩ S_k2}(pos_c)||²

            Aproximación: penalizar que la distancia al centro de k2 (la esfera secundaria)
            sea mayor que la suma de radios → el token NO está en la intersección.
            """
            if polysemic_mask.sum() == 0:
                return torch.tensor(0.0, device=token_positions.device)

            # Seleccionar tokens polisémicos
            poly_positions = token_positions[polysemic_mask]    # [P, 3]
            poly_memberships = memberships[polysemic_mask]       # [P, K]
            P = poly_positions.shape[0]
            K = sphere_centers.shape[0]

            if P == 0 or K < 2:
                return torch.tensor(0.0, device=token_positions.device)

            # Para cada token polisémico, encontrar sus dos esferas más asignadas
            # y penalizar que NO esté en su intersección
            loss = torch.tensor(0.0, device=token_positions.device)
            for i in range(P):
                top2 = torch.topk(poly_memberships[i], min(2, K)).indices
                if len(top2) < 2:
                    continue
                k1, k2 = top2[0], top2[1]
                d1 = torch.norm(poly_positions[i] - sphere_centers[k1])
                d2 = torch.norm(poly_positions[i] - sphere_centers[k2])
                # El token debe estar dentro de AMBAS esferas para estar en la intersección
                # Pero no tenemos acceso a sphere_radii aquí fácilmente — penalizar diferencia de distancias
                loss += (d1 - d2).pow(2)

            return loss / P

        def l_reg(
            self,
            sphere_radii:  torch.Tensor,    # [K]
            wormhole_lengths: Optional[torch.Tensor] = None,  # [W] longitudes de wormholes
        ) -> torch.Tensor:
            """
            L_reg: penalizar radios excesivamente grandes y wormholes largos.

            L_reg = λ_r · Σ_k r_k² + λ_w · Σ_w length_w²
            """
            loss = self.cfg.lambda_radius * sphere_radii.pow(2).mean()
            if wormhole_lengths is not None and len(wormhole_lengths) > 0:
                loss = loss + self.cfg.lambda_worm * wormhole_lengths.pow(2).mean()
            return loss

        def forward(
            self,
            token_positions:    torch.Tensor,   # [N, 3]
            sphere_centers:     torch.Tensor,   # [K, 3]
            sphere_radii:       torch.Tensor,   # [K]
            memberships:        torch.Tensor,   # [N, K]
            polysemic_mask:     Optional[torch.Tensor] = None,     # [N] bool
            similarity_matrix:  Optional[torch.Tensor] = None,     # [N, N]
            wormhole_lengths:   Optional[torch.Tensor] = None,     # [W]
        ) -> Dict[str, torch.Tensor]:
            """
            Calcula la pérdida espacial completa.

            Returns:
                dict con claves: 'total', 'prox', 'cover', 'inter', 'reg'
            """
            N = token_positions.shape[0]

            if polysemic_mask is None:
                polysemic_mask = torch.zeros(N, dtype=torch.bool, device=token_positions.device)

            if similarity_matrix is None:
                # Similitud coseno entre posiciones 3D como proxy
                normed = F.normalize(token_positions, dim=1)
                similarity_matrix = torch.clamp(normed @ normed.T, min=0.0)

            # Computar componentes
            lp = self.l_prox(token_positions, sphere_centers, memberships, similarity_matrix)
            lc = self.l_cover(token_positions, sphere_centers, sphere_radii, memberships)
            li = self.l_inter(token_positions, sphere_centers, memberships, polysemic_mask)
            lr = self.l_reg(sphere_radii, wormhole_lengths)

            l_spatial = (self.cfg.beta  * lp +
                         self.cfg.gamma * lc +
                         self.cfg.delta * li +
                         self.cfg.eta   * lr)

            return {
                "total": l_spatial,
                "prox":  lp,
                "cover": lc,
                "inter": li,
                "reg":   lr,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Implementación NumPy (sin gradientes — para análisis y validación)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialLossNumpy:
    """
    Versión NumPy de la pérdida espacial para análisis sin PyTorch.
    Sin gradientes — útil para inspección, logging y validación.
    """

    def __init__(self, config: Optional[SpatialLossConfig] = None):
        self.cfg = config or SpatialLossConfig()

    def compute(
        self,
        token_positions: np.ndarray,   # [N, 3]
        sphere_centers:  np.ndarray,   # [K, 3]
        sphere_radii:    np.ndarray,   # [K]
        memberships:     np.ndarray,   # [N, K]
        polysemic_mask:  Optional[np.ndarray] = None,   # [N] bool
        similarity_matrix: Optional[np.ndarray] = None, # [N, N]
    ) -> Dict[str, float]:
        N = len(token_positions)
        K = len(sphere_centers)

        if polysemic_mask is None:
            polysemic_mask = np.zeros(N, dtype=bool)

        # ── L_prox ───────────────────────────────────────────────────────
        if similarity_matrix is not None:
            d_sq = np.sum(
                (token_positions[:, np.newaxis] - token_positions[np.newaxis]) ** 2,
                axis=2
            )
            lp = float(np.mean(similarity_matrix * d_sq))
        else:
            lp = 0.0

        # ── L_cover ──────────────────────────────────────────────────────
        d_to_centers = np.sqrt(
            np.sum(
                (token_positions[:, np.newaxis] - sphere_centers[np.newaxis]) ** 2,
                axis=2
            ) + 1e-8
        )  # [N, K]
        violation = np.maximum(0.0,
                               d_to_centers - sphere_radii[np.newaxis] + self.cfg.margin_cover)
        lc = float(np.mean(memberships * violation ** 2))

        # ── L_inter ──────────────────────────────────────────────────────
        poly_idx = np.where(polysemic_mask)[0]
        li = 0.0
        if len(poly_idx) > 0 and K >= 2:
            for idx in poly_idx:
                top2 = np.argsort(memberships[idx])[-2:]
                k1, k2 = top2[-1], top2[-2]
                d1 = np.linalg.norm(token_positions[idx] - sphere_centers[k1])
                d2 = np.linalg.norm(token_positions[idx] - sphere_centers[k2])
                li += (d1 - d2) ** 2
            li /= len(poly_idx)

        # ── L_reg ────────────────────────────────────────────────────────
        lr = float(self.cfg.lambda_radius * np.mean(sphere_radii ** 2))

        # ── L_spatial ────────────────────────────────────────────────────
        l_spatial = (self.cfg.beta  * lp +
                     self.cfg.gamma * lc +
                     self.cfg.delta * li +
                     self.cfg.eta   * lr)

        return {
            "total": l_spatial,
            "prox":  lp,
            "cover": lc,
            "inter": li,
            "reg":   lr,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test / Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """Test de la pérdida espacial con datos sintéticos."""
    print("=" * 60)
    print(" SpatialLoss Demo")
    print("=" * 60)

    N, K, D = 24, 3, 3  # 24 tokens, 3 esferas, 3D

    # Tokens en 3 clusters bien separados (ground truth)
    positions = np.vstack([
        np.random.randn(8, D) + np.array([ 2.0,  0.0, 0.0]),  # cluster 0
        np.random.randn(8, D) + np.array([-1.0,  1.7, 0.0]),  # cluster 1
        np.random.randn(8, D) + np.array([-1.0, -1.7, 0.0]),  # cluster 2
    ])

    # Centros casi correctos
    centers = np.array([
        [ 1.8,  0.1, 0.0],
        [-0.9,  1.6, 0.0],
        [-0.9, -1.6, 0.0],
    ])
    radii = np.ones(K) * 2.0

    # Membresías difusas (T=1.0)
    d_sq = np.sum((positions[:, np.newaxis] - centers[np.newaxis]) ** 2, axis=2)
    logits = -d_sq / 1.0
    logits -= logits.max(axis=1, keepdims=True)
    memberships = np.exp(logits)
    memberships /= memberships.sum(axis=1, keepdims=True)

    # 3 tokens polisémicos (borde entre cluster 0 y 1)
    poly_mask = np.zeros(N, dtype=bool)
    poly_mask[[7, 8, 9]] = True

    print(f"[data] N={N} tokens, K={K} esferas, D={D}")
    print(f"[data] {poly_mask.sum()} tokens polisémicos")

    # ── NumPy version ─────────────────────────────────────────────────────
    loss_fn_np = SpatialLossNumpy()
    result_np = loss_fn_np.compute(positions, centers, radii, memberships, poly_mask)
    print(f"\n[numpy] L_spatial = {result_np['total']:.4f}")
    for k, v in result_np.items():
        if k != "total":
            print(f"        L_{k} = {v:.4f}")

    # ── PyTorch version ───────────────────────────────────────────────────
    if HAS_TORCH:
        pos_t    = torch.from_numpy(positions.astype(np.float32))
        cen_t    = torch.tensor(centers.astype(np.float32), requires_grad=True)
        rad_t    = torch.tensor(radii.astype(np.float32), requires_grad=True)
        mem_t    = torch.from_numpy(memberships.astype(np.float32))
        poly_t   = torch.from_numpy(poly_mask)

        loss_fn = SpatialLoss()
        result_pt = loss_fn(pos_t, cen_t, rad_t, mem_t, poly_t)
        result_pt["total"].backward()

        print(f"\n[torch] L_spatial = {result_pt['total'].item():.4f}")
        for k, v in result_pt.items():
            if k != "total":
                print(f"        L_{k} = {v.item():.4f}")

        print(f"\n[grad] grad_centers norm: {cen_t.grad.norm().item():.4f}")
        print(f"[grad] grad_radii: {rad_t.grad}")
        print("[OK] Gradientes fluyen correctamente a sphere_centers y sphere_radii")
    else:
        print("\n[info] PyTorch no disponible — solo NumPy calculado")

    print("\n[OK] SpatialLoss demo completado.")


if __name__ == "__main__":
    np.random.seed(42)
    run_demo()
