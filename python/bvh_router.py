#!/usr/bin/env python3
"""
bvh_router.py — Router BVH Jerárquico para SpectralAI v5.0 "Orchestrator"

El RT Core NO calcula atención — RUTEA.
4 niveles × 3D = 12 dimensiones semánticas → selección de micro-experto.

Reutiliza componentes validados de inception_attention.py v4.0:
  - SpectralEncoder (color del rayo)
  - AffinePortal (portales dimensionales)
  - PrismaticRefraction (Snell semántico)

Diferencia clave vs v4.0: hard routing (argmax) en inferencia,
                          Gumbel-Softmax en training (diferenciable).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple


# ─────────────────────────────────────────────────────────────────
# Configuración del Router
# ─────────────────────────────────────────────────────────────────

class RouterConfig:
    """Hiperparámetros del Router BVH v5.0."""
    __slots__ = (
        'embed_dim', 'spectral_dim',
        'n_level1', 'n_level2', 'n_level3',
        'total_l1', 'total_l2', 'total_l3',
        'n_experts',
        'temperature_init', 'temperature_min', 'temperature_decay',
        'gumbel_hard', 'dropout',
    )

    def __init__(
        self,
        embed_dim:         int   = 256,
        spectral_dim:      int   = 64,
        n_level1:          int   = 4,   # Dominios (Ciencia, Código, Humanidades, General)
        n_level2:          int   = 4,   # Subdominios por dominio
        n_level3:          int   = 4,   # Conceptos por subdominio
        temperature_init:  float = 1.0,
        temperature_min:   float = 0.1,
        temperature_decay: float = 0.95,  # Más agresivo que v4.0
        gumbel_hard:       bool  = False, # True en inferencia
        dropout:           float = 0.1,
    ):
        self.embed_dim         = embed_dim
        self.spectral_dim      = spectral_dim
        self.n_level1          = n_level1
        self.n_level2          = n_level2
        self.n_level3          = n_level3
        self.total_l1          = n_level1
        self.total_l2          = n_level1 * n_level2
        self.total_l3          = n_level1 * n_level2 * n_level3
        self.n_experts         = self.total_l3  # 1 experto por hoja
        self.temperature_init  = temperature_init
        self.temperature_min   = temperature_min
        self.temperature_decay = temperature_decay
        self.gumbel_hard       = gumbel_hard
        self.dropout           = dropout


# ─────────────────────────────────────────────────────────────────
# Resultado del routing
# ─────────────────────────────────────────────────────────────────

class RoutingResult(NamedTuple):
    """Resultado del router BVH."""
    expert_id: torch.Tensor       # (B,) — ID del experto seleccionado
    expert_probs: torch.Tensor    # (B, n_experts) — probabilidades soft
    route_path: torch.Tensor      # (B, 3) — camino [nivel1, nivel2, nivel3]
    confidence: torch.Tensor      # (B,) — confianza de la selección (max prob)


# ─────────────────────────────────────────────────────────────────
# Componentes reutilizados de v4.0 (adaptados para routing puro)
# ─────────────────────────────────────────────────────────────────

class SpectralEncoder(nn.Module):
    """Contexto → color del rayo (reutilizado de v4.0)."""

    def __init__(self, embed_dim: int, spectral_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, spectral_dim),
            nn.Tanh(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class AffinePortal(nn.Module):
    """Portal dimensional: transforma coordenadas 3D entre niveles."""

    def __init__(self, n_portals: int):
        super().__init__()
        eye = torch.zeros(n_portals, 3, 4)
        for i in range(3):
            eye[:, i, i] = 1.0
        self.transform = nn.Parameter(eye)

    def apply_all(self, pos_3d: torch.Tensor) -> torch.Tensor:
        """(B, 3) → (B, K, 3) aplicando todos los K portales."""
        ones = torch.ones(*pos_3d.shape[:-1], 1, device=pos_3d.device)
        pos_h = torch.cat([pos_3d, ones], dim=-1)
        return torch.einsum('kij,bj->bki', self.transform, pos_h)


class PrismaticRefraction(nn.Module):
    """Snell semántico: el color del rayo modifica las probabilidades."""

    def __init__(self, n_spheres: int, spectral_dim: int):
        super().__init__()
        self.W_dispersion = nn.Linear(spectral_dim, n_spheres)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.W_dispersion(spectral))


# ─────────────────────────────────────────────────────────────────
# RouterLevel — un nivel de routing con Gumbel-Softmax
# ─────────────────────────────────────────────────────────────────

class RouterLevel(nn.Module):
    """
    Un nivel del árbol BVH para routing.

    Diferencia vs SphereLevel de v4.0:
    - Gumbel-Softmax para routing discreto diferenciable
    - argmax en inferencia (hard routing real)
    - Sin energía — solo devuelve la decisión
    """

    def __init__(self, n_spheres: int, parent_spheres: int = 1):
        super().__init__()
        self.n_spheres = n_spheres
        self.parent_spheres = parent_spheres
        total = parent_spheres * n_spheres

        # Centros en coordenadas locales
        self.centers = nn.Parameter(torch.randn(total, 3) * 0.5)
        self.log_radii = nn.Parameter(torch.zeros(total))

    @property
    def radii(self) -> torch.Tensor:
        return F.softplus(self.log_radii)

    def forward(
        self,
        pos_3d: torch.Tensor,
        temperature: torch.Tensor,
        parent_choice: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pos_3d:        (B, 3) — posición del query en espacio del padre
        temperature:   scalar tensor
        parent_choice: (B, parent_spheres) — soft/hard assignment del padre
        hard:          si True, usar argmax (inferencia)

        returns:
            probs:  (B, total) — probabilidades de cada esfera
            choice: (B,) — índice de la esfera elegida
        """
        B = pos_3d.shape[0]
        total = self.parent_spheres * self.n_spheres

        # Distancias a centros: (B, total)
        diff = pos_3d.unsqueeze(1) - self.centers.unsqueeze(0)
        d_sq = (diff ** 2).sum(dim=-1)

        # Logits con temperatura
        logits = -d_sq / (2.0 * temperature ** 2 + 1e-8)

        # Enmascarar por padre si aplica
        if parent_choice is not None:
            parent_expanded = parent_choice.repeat_interleave(
                self.n_spheres, dim=1
            )
            logits = logits + torch.log(parent_expanded + 1e-10)

        if hard or not self.training:
            # Inferencia: argmax puro
            choice = logits.argmax(dim=-1)
            probs = F.one_hot(choice, total).float()
        else:
            # Training: Gumbel-Softmax (diferenciable)
            # hard=False → soft probs in forward AND backward.
            # With hard=True, one-hot forward kills gradient flow through
            # hierarchical levels (parent_choice masking becomes binary).
            # Soft probs allow gradient to explore ALL branches of the BVH.
            probs = F.gumbel_softmax(logits, tau=temperature.clamp(min=0.01),
                                     hard=False)
            choice = probs.argmax(dim=-1)

        return probs, choice


# ─────────────────────────────────────────────────────────────────
# BVHRouter — Router completo de 4 niveles
# ─────────────────────────────────────────────────────────────────

class BVHRouter(nn.Module):
    """
    Router BVH jerárquico de 4 niveles (12 dimensiones semánticas).

    Nivel 1 (Dim 1-3):  Universo Temático    → 4 dominios
    Nivel 2 (Dim 4-6):  Galaxia de Conceptos → 16 subdominios
    Nivel 3 (Dim 7-9):  Sistema de Contextos → 64 conceptos
    Nivel 4:            Selección de experto  → expert_id

    Flujo:
    1. Prompt → embedding medio → posición 3D
    2. Contexto → color espectral
    3. Rayo atraviesa 3 niveles con portales + refracción
    4. Resultado: expert_id (qué micro-modelo cargar)
    """

    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg

        # Proyección prompt → 3D
        self.to_3d = nn.Linear(cfg.embed_dim, 3)

        # Codificación espectral
        self.spectral = SpectralEncoder(cfg.embed_dim, cfg.spectral_dim)

        # 3 niveles de routing
        self.level1 = RouterLevel(cfg.n_level1, parent_spheres=1)
        self.portal1 = AffinePortal(cfg.total_l1)
        self.refract1 = PrismaticRefraction(cfg.total_l1, cfg.spectral_dim)

        self.level2 = RouterLevel(cfg.n_level2, parent_spheres=cfg.total_l1)
        self.portal2 = AffinePortal(cfg.total_l2)
        self.refract2 = PrismaticRefraction(cfg.total_l2, cfg.spectral_dim)

        self.level3 = RouterLevel(cfg.n_level3, parent_spheres=cfg.total_l2)
        self.refract3 = PrismaticRefraction(cfg.total_l3, cfg.spectral_dim)

        # Temperatura (annealing)
        self.register_buffer('temperature',
                             torch.tensor(cfg.temperature_init))

        # Load balancing: contador de uso por experto
        self.register_buffer('expert_counts',
                             torch.zeros(cfg.n_experts))

    def anneal_temperature(self):
        """Reduce temperatura (llamar por epoch)."""
        new_temp = max(
            self.cfg.temperature_min,
            self.temperature.item() * self.cfg.temperature_decay
        )
        self.temperature.fill_(new_temp)

    def reset_expert_counts(self):
        """Reset contadores de balanceo."""
        self.expert_counts.zero_()

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        hard: bool = False,
    ) -> RoutingResult:
        """
        prompt_embedding: (B, D) — embedding medio del prompt
        hard:             forzar hard routing (inferencia)

        returns: RoutingResult con expert_id, probs, path, confidence
        """
        B = prompt_embedding.shape[0]
        T = self.temperature
        use_hard = hard or not self.training

        # ── Prompt → 3D + espectral ─────────────────────────────
        pos_3d = self.to_3d(prompt_embedding)       # (B, 3)
        spectral = self.spectral(prompt_embedding)   # (B, spectral_dim)

        # ── Nivel 1: Dominios ────────────────────────────────────
        n1 = self.refract1(spectral)  # (B, K1) — modulación por color
        p1, c1 = self.level1(pos_3d, T, hard=use_hard)  # p1: (B, K1)
        p1 = p1 * n1
        p1 = p1 / (p1.sum(dim=-1, keepdim=True) + 1e-8)

        # Portal: transformar al espacio local del dominio elegido
        all_locals1 = self.portal1.apply_all(pos_3d)  # (B, K1, 3)
        pos_l1 = (all_locals1 * p1.unsqueeze(-1)).sum(dim=1)  # (B, 3)

        # ── Nivel 2: Subdominios ─────────────────────────────────
        n2 = self.refract2(spectral)
        p2, c2 = self.level2(pos_l1, T, parent_choice=p1, hard=use_hard)
        p2 = p2 * n2
        p2 = p2 / (p2.sum(dim=-1, keepdim=True) + 1e-8)

        all_locals2 = self.portal2.apply_all(pos_l1)  # (B, K2_total, 3)
        pos_l2 = (all_locals2 * p2.unsqueeze(-1)).sum(dim=1)  # (B, 3)

        # ── Nivel 3: Conceptos (= expertos) ──────────────────────
        n3 = self.refract3(spectral)
        p3, c3 = self.level3(pos_l2, T, parent_choice=p2, hard=use_hard)
        p3 = p3 * n3
        p3 = p3 / (p3.sum(dim=-1, keepdim=True) + 1e-8)

        expert_id = p3.argmax(dim=-1)  # (B,) — ID del experto
        confidence = p3.max(dim=-1).values  # (B,)

        # Actualizar contadores de balanceo
        if self.training:
            for eid in expert_id:
                self.expert_counts[eid] += 1

        # Path: [dominio, subdominio, concepto]
        route_path = torch.stack([c1, c2, c3], dim=-1)  # (B, 3)

        return RoutingResult(
            expert_id=expert_id,
            expert_probs=p3,
            route_path=route_path,
            confidence=confidence,
        )

    def load_balancing_loss(self) -> torch.Tensor:
        """
        Pérdida de balanceo: penaliza uso desigual de expertos.
        L_balance = Σ (usage_i - 1/K)²
        """
        total = self.expert_counts.sum()
        if total < 1:
            return torch.tensor(0.0, device=self.expert_counts.device)
        usage = self.expert_counts / total
        target = 1.0 / self.cfg.n_experts
        return ((usage - target) ** 2).sum()


# ─────────────────────────────────────────────────────────────────
# Test rápido
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BVHRouter v5.0 — Test")
    print("=" * 60)

    cfg = RouterConfig(embed_dim=256, n_level1=4, n_level2=4, n_level3=4)
    router = BVHRouter(cfg)

    total_params = sum(p.numel() for p in router.parameters())
    print(f"Params router: {total_params:,}")
    print(f"Esferas: L1={cfg.total_l1}, L2={cfg.total_l2}, L3={cfg.total_l3}")
    print(f"Total expertos: {cfg.n_experts}")

    # Test con batch de prompts
    B = 8
    prompt = torch.randn(B, 256)

    # Training mode (Gumbel-Softmax)
    router.train()
    result = router(prompt)
    print(f"\n[TRAIN] Expert IDs: {result.expert_id.tolist()}")
    print(f"[TRAIN] Confidence: {result.confidence.tolist()}")
    print(f"[TRAIN] Routes:     {result.route_path.tolist()}")
    print(f"[TRAIN] Balance loss: {router.load_balancing_loss().item():.4f}")

    # Inference mode (hard routing)
    router.eval()
    result2 = router(prompt, hard=True)
    print(f"\n[EVAL]  Expert IDs: {result2.expert_id.tolist()}")
    print(f"[EVAL]  Confidence: {result2.confidence.tolist()}")
    print(f"[EVAL]  Routes:     {result2.route_path.tolist()}")

    # Determinismo en eval
    result3 = router(prompt, hard=True)
    assert torch.equal(result2.expert_id, result3.expert_id), "ERROR: eval no determinista"
    print("\n[OK] Eval determinista")

    # Gradientes fluyen
    router.train()
    prompt_grad = torch.randn(4, 256, requires_grad=True)
    res = router(prompt_grad)
    loss = res.expert_probs.sum()
    loss.backward()
    assert prompt_grad.grad is not None, "ERROR: gradientes no fluyen"
    print("[OK] Gradientes fluyen")

    print("\n[OK] BVHRouter v5.0 -- Todos los tests pasados")
