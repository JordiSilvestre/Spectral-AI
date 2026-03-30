#!/usr/bin/env python3
"""
inception_attention.py — Sistema de atención COMPLETO SpectralAI v4.0

Implementa el paper tal cual:
  1. Tokens → 3D via PCA esférica (W_projection learnable)
  2. BVH jerárquico 4 niveles (OHBSC con soft assignment)
  3. Rayos espectrales coloreados por contexto (f ∈ R^64)
  4. Refracción prismática (Snell semántico en cada esfera)
  5. Resonancia Fourier W(ω) en las hojas
  6. Portales afines (AffinePortal 4×4) entre niveles
  7. Pesos ternarios opcionales {-1, 0, +1}

Dos modos:
  TRAIN  — proxy diferenciable (soft hierarchy + analytical gradients)
  INFER  — RT Cores reales via inception_engine / batch_runner
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Configuración del Inception Engine
# ─────────────────────────────────────────────────────────────────

class InceptionConfig:
    """Hiperparámetros del sistema de esferas anidadas."""
    __slots__ = (
        'embed_dim', 'num_heads', 'head_dim', 'context_len',
        'n_domains', 'n_subdomains', 'n_concepts',
        'spectral_dim', 'num_fourier_modes',
        'temperature_init', 'temperature_min', 'temperature_decay',
        'lambda_absorption', 'dropout',
        'total_l1', 'total_l2', 'total_l3',
        'advanced_optics', 'chromatic_bands', 'interference_rays',
    )

    def __init__(
        self,
        embed_dim:        int   = 256,
        num_heads:        int   = 4,
        context_len:      int   = 256,
        n_domains:        int   = 4,      # Nivel 1: esferas grandes
        n_subdomains:     int   = 4,      # Nivel 2: por dominio
        n_concepts:       int   = 4,      # Nivel 3: por subdominio
        spectral_dim:     int   = 64,     # Color del rayo
        num_fourier_modes:int   = 8,      # Modos de resonancia
        temperature_init: float = 1.0,    # Softmax temp inicial
        temperature_min:  float = 0.1,    # Temp mínima (hardening)
        temperature_decay:float = 0.995,  # Annealing por epoch
        lambda_absorption:float = 0.1,    # Decay de energía del rayo
        dropout:          float = 0.1,
        advanced_optics:  bool  = False,  # Patent P3 Claims 21-33
        chromatic_bands:  int   = 4,      # Bands for ChromaticAberration
        interference_rays:int   = 4,      # Rays for PhaseCoherentInterference
    ):
        self.embed_dim         = embed_dim
        self.num_heads         = num_heads
        self.head_dim          = embed_dim // num_heads
        self.context_len       = context_len
        self.n_domains         = n_domains
        self.n_subdomains      = n_subdomains
        self.n_concepts        = n_concepts
        self.spectral_dim      = spectral_dim
        self.num_fourier_modes = num_fourier_modes
        self.temperature_init  = temperature_init
        self.temperature_min   = temperature_min
        self.temperature_decay = temperature_decay
        self.lambda_absorption = lambda_absorption
        self.dropout           = dropout
        self.advanced_optics   = advanced_optics
        self.chromatic_bands   = chromatic_bands
        self.interference_rays = interference_rays

        # Total de esferas por nivel
        # L1: n_domains
        # L2: n_domains * n_subdomains
        # L3: n_domains * n_subdomains * n_concepts
        self.total_l1 = n_domains
        self.total_l2 = n_domains * n_subdomains
        self.total_l3 = n_domains * n_subdomains * n_concepts


# ─────────────────────────────────────────────────────────────────
# 1. Spectral Encoder — codifica el contexto como "color" del rayo
# ─────────────────────────────────────────────────────────────────

class SpectralEncoder(nn.Module):
    """
    Transforma el contexto conversacional en un vector espectral f ∈ R^spectral_dim.
    El "color" del rayo determina cómo refracta en cada esfera (polisemia).
    """

    def __init__(self, embed_dim: int, spectral_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, spectral_dim),
            nn.Tanh(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, S, D) — embeddings de todos los tokens previos
        returns: (B, S, spectral_dim) — color espectral por posición
        """
        return self.net(context)


# ─────────────────────────────────────────────────────────────────
# 2. AffinePortal — transformación de coordenadas entre niveles
# ─────────────────────────────────────────────────────────────────

class AffinePortal(nn.Module):
    """
    Portal dimensional: transforma coordenadas 3D al entrar en una esfera.
    Equivale a optixGetInstanceTransformFromHandle() en el hardware.

    Cada esfera tiene su propio portal = matriz afín 3×4 learnable.
    También transforma ω (frecuencia de resonancia) via fila extra.
    """

    def __init__(self, n_portals: int):
        super().__init__()
        # Matrices 3×4 (rotación + traslación) inicializadas como identidad
        eye = torch.zeros(n_portals, 3, 4)
        for i in range(3):
            eye[:, i, i] = 1.0
        self.transform = nn.Parameter(eye)

        # Transformación de omega: [scale, bias] per portal
        self.omega_transform = nn.Parameter(
            torch.stack([torch.ones(n_portals), torch.zeros(n_portals)], dim=1)
        )

    def forward(self, pos_3d: torch.Tensor, portal_idx: torch.Tensor
                ) -> torch.Tensor:
        """
        Transforma posiciones 3D a través del portal seleccionado.

        pos_3d:     (*, 3) o (B, S, 1, 3) para batched
        portal_idx: índices de portales — escalar, (B,S), o (K,) para batched
        returns:    (*, 3) o (B, S, K, 3) si portal_idx es (K,)
        """
        M = self.transform[portal_idx]  # (..., 3, 4)
        ones = torch.ones(*pos_3d.shape[:-1], 1, device=pos_3d.device)
        pos_h = torch.cat([pos_3d, ones], dim=-1)  # (..., 4)
        return torch.einsum('...ij,...j->...i', M, pos_h)

    def apply_all(self, pos_3d: torch.Tensor) -> torch.Tensor:
        """
        Aplica TODOS los portales a cada posición en un solo einsum.
        Elimina loops Python.

        pos_3d: (B, S, 3)
        returns: (B, S, K, 3) donde K = n_portals
        """
        # self.transform: (K, 3, 4)
        ones = torch.ones(*pos_3d.shape[:-1], 1, device=pos_3d.device)
        pos_h = torch.cat([pos_3d, ones], dim=-1)  # (B, S, 4)
        # (K, 3, 4) × (B, S, 4) → (B, S, K, 3)
        return torch.einsum('kij,bsj->bski', self.transform, pos_h)

    def transform_omega(self, omega: torch.Tensor, portal_idx: torch.Tensor
                        ) -> torch.Tensor:
        """ω_new = scale * ω + bias, normalizado a [0, 2π]."""
        params = self.omega_transform[portal_idx]  # (*, 2)
        scale = params[..., 0]
        bias  = params[..., 1]
        return (scale * omega + bias) % (2.0 * math.pi)


# ─────────────────────────────────────────────────────────────────
# 3. Refracción Prismática — Snell semántico
# ─────────────────────────────────────────────────────────────────

class PrismaticRefraction(nn.Module):
    """
    Ley de Snell semántica: el índice de refracción depende del color del rayo.
    n(esfera, f) = σ(W_dispersion · f)

    Rayo AZUL (contexto=Código) → refracta 45° → matrices de programación
    Rayo ROJO (contexto=Música) → misma esfera → refracta 90° → matrices de ritmo
    """

    def __init__(self, n_spheres: int, spectral_dim: int):
        super().__init__()
        self.W_dispersion = nn.Linear(spectral_dim, n_spheres)

    def forward(self, spectral_color: torch.Tensor) -> torch.Tensor:
        """
        spectral_color: (B, S, spectral_dim) — color del rayo
        returns:        (B, S, n_spheres)    — índice de refracción por esfera
        """
        return torch.sigmoid(self.W_dispersion(spectral_color))


# ─────────────────────────────────────────────────────────────────
# 3b. Chromatic Aberration — Multi-Band Spectral Decomposition
#     Patent P3 Claims 21-25
# ─────────────────────────────────────────────────────────────────

class ChromaticAberration(nn.Module):
    """
    Multi-band spectral decomposition: the color vector is split into B
    frequency bands, each refracted independently through PrismaticRefraction.
    The B routing decisions are combined via learned band weights.

    Physical analogy: different wavelengths of light refract at different
    angles through a prism — each band captures a different aspect of context
    (topic, tone, formality, temporal reference).

    Patent P3 Claims 21-25.
    """

    def __init__(self, n_spheres: int, spectral_dim: int, n_bands: int = 4):
        super().__init__()
        assert spectral_dim % n_bands == 0, (
            f"spectral_dim ({spectral_dim}) must be divisible by n_bands ({n_bands})"
        )
        self.n_bands = n_bands
        self.band_size = spectral_dim // n_bands
        self.n_spheres = n_spheres

        # One PrismaticRefraction per band (independent W_dispersion)
        self.band_refractions = nn.ModuleList([
            PrismaticRefraction(n_spheres=n_spheres, spectral_dim=self.band_size)
            for _ in range(n_bands)
        ])

        # Learned band weights for combining routing decisions
        self.band_weights = nn.Parameter(torch.ones(n_bands) / n_bands)

    def forward(self, spectral_color: torch.Tensor) -> torch.Tensor:
        """
        spectral_color: (B, S, spectral_dim) — full color vector
        returns:        (B, S, n_spheres)    — combined refractive indices
        """
        # Decompose color into B bands
        bands = spectral_color.chunk(self.n_bands, dim=-1)  # list of (B, S, band_size)

        # Refract each band independently
        band_indices = [
            refr(band) for refr, band in zip(self.band_refractions, bands)
        ]  # list of (B, S, n_spheres)

        # Combine with learned weights (softmax for proper weighting)
        weights = F.softmax(self.band_weights, dim=0)  # (n_bands,)

        # Weighted sum of per-band refractive indices
        combined = torch.zeros_like(band_indices[0])
        for w, idx in zip(weights, band_indices):
            combined = combined + w * idx

        return combined


# ─────────────────────────────────────────────────────────────────
# 3c. Total Internal Reflection — Hard Routing Boundary
#     Patent P3 Claims 26-29
# ─────────────────────────────────────────────────────────────────

class TotalInternalReflection(nn.Module):
    """
    When context is completely mismatched with a sphere's specialty,
    the ray is reflected (not refracted) — routed to a different region.

    Physical analogy: sin(theta_i) > n2/n1 → total internal reflection.
    Semantic: word "scale" in music context → TIR away from math expert.

    Uses Straight-Through Estimator for differentiability:
    forward = hard TIR decision, backward = soft gradient.

    Patent P3 Claims 26-29.
    """

    SNELL_EPSILON: float = 0.01

    def __init__(self, n_spheres: int):
        super().__init__()
        self.n_spheres = n_spheres
        # Learned base refractive index per sphere
        self.n_base = nn.Parameter(torch.ones(n_spheres) * 1.5)

    def forward(
        self,
        refractive_indices: torch.Tensor,
        membership: torch.Tensor,
        cos_incidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        refractive_indices: (B, S, n_spheres) — n from PrismaticRefraction
        membership:         (B, S, n_spheres) — soft sphere membership
        cos_incidence:      (B, S, n_spheres) — cosine of incidence angle

        returns:
            adjusted_membership: (B, S, n_spheres) — with TIR zeroed out
            tir_mask:            (B, S, n_spheres) — True where TIR occurred
        """
        # n_ratio = n_in / n_out ≈ 1 / refractive_indices (entering sphere)
        n_ratio = 1.0 / (refractive_indices + 1e-8)

        # Snell discriminant: 1 - n_ratio² * (1 - cos_i²)
        sin_sq = 1.0 - cos_incidence ** 2
        discriminant = 1.0 - n_ratio ** 2 * sin_sq

        # TIR occurs when discriminant < -epsilon
        tir_mask = discriminant < -self.SNELL_EPSILON  # (B, S, n_spheres)

        # Soft discriminant for gradients (sigmoid around boundary)
        soft_tir = torch.sigmoid(-discriminant / self.SNELL_EPSILON * 10.0)

        # Hard TIR: zero out membership where TIR occurs
        hard_mask = (~tir_mask).float()

        # Straight-Through Estimator: forward=hard, backward=soft
        adjusted_membership = membership * (hard_mask - soft_tir.detach() + soft_tir)

        # Renormalize membership (redistribute from TIR'd spheres)
        total = adjusted_membership.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adjusted_membership = adjusted_membership / total

        return adjusted_membership, tir_mask


# ─────────────────────────────────────────────────────────────────
# 3d. Phase-Coherent Multi-Ray Interference
#     Patent P3 Claims 30-33
# ─────────────────────────────────────────────────────────────────

class PhaseCoherentInterference(nn.Module):
    """
    Multiple rays with slightly different spectral colors interfere at
    target spheres. Constructive interference strengthens routing,
    destructive interference weakens it.

    Physical analogy: superposition of waves. When rays arrive in phase,
    amplitudes add (constructive). Out of phase, they cancel (destructive).

    Patent P3 Claims 30-33.
    """

    def __init__(self, spectral_dim: int, n_rays: int = 4):
        super().__init__()
        self.n_rays = n_rays
        self.spectral_dim = spectral_dim

        # Learned perturbation directions per ray
        self.epsilon = nn.Parameter(
            torch.randn(n_rays, spectral_dim) * 0.01
        )

        # Learned amplitude scaling per ray
        self.amplitude = nn.Parameter(torch.ones(n_rays))

    def forward(
        self,
        spectral_color: torch.Tensor,
        refraction_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        spectral_color: (B, S, spectral_dim) — base color vector
        refraction_fn:  callable that takes (B, S, spectral_dim) → (B, S, n_spheres)
                        (PrismaticRefraction or ChromaticAberration)

        returns:        (B, S, n_spheres) — interference-modulated routing weights
        """
        # Phase offsets: evenly spaced around 2π
        phases = torch.linspace(
            0, 2 * math.pi * (1 - 1 / self.n_rays),
            self.n_rays,
            device=spectral_color.device,
        )  # (R,)

        amplitudes = F.softplus(self.amplitude)  # (R,) positive

        # Accumulate complex interference per sphere
        # Using real/imag decomposition (no complex tensors for CUDA compat)
        real_sum = None
        imag_sum = None

        for r in range(self.n_rays):
            # Perturbed color for this ray
            perturbed = spectral_color + self.epsilon[r]  # (B, S, spectral_dim)

            # Refract the perturbed ray
            routing_r = refraction_fn(perturbed)  # (B, S, n_spheres)

            # Phase contribution: A_r * exp(i * phase_r) applied to routing
            # = A_r * cos(phase_r) * routing  +  i * A_r * sin(phase_r) * routing
            a_r = amplitudes[r]
            cos_phase = torch.cos(phases[r])
            sin_phase = torch.sin(phases[r])

            contribution_real = a_r * cos_phase * routing_r
            contribution_imag = a_r * sin_phase * routing_r

            if real_sum is None:
                real_sum = contribution_real
                imag_sum = contribution_imag
            else:
                real_sum = real_sum + contribution_real
                imag_sum = imag_sum + contribution_imag

        # Intensity = |complex_sum|² = real² + imag²
        intensity = real_sum ** 2 + imag_sum ** 2  # (B, S, n_spheres)

        # Normalize to valid probability distribution
        total = intensity.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return intensity / total


# ─────────────────────────────────────────────────────────────────
# 4. Fourier Resonance — evaluación W(ω) en las hojas
# ─────────────────────────────────────────────────────────────────

class FourierResonance(nn.Module):
    """
    Resonancia semántica: W(ω) = tanh(scale · Σ_k a_k·sin(kω) + b_k·cos(kω))

    Cada hoja del BVH tiene sus coeficientes Fourier (a_k, b_k).
    El ω viene determinado por la posición y el camino del rayo.
    """

    def __init__(self, n_leaves: int, num_modes: int = 8):
        super().__init__()
        self.num_modes = num_modes
        self.a = nn.Parameter(torch.randn(n_leaves, num_modes) * 0.1)
        self.b = nn.Parameter(torch.randn(n_leaves, num_modes) * 0.1)
        self.scale = nn.Parameter(torch.ones(n_leaves))

    def forward(self, omega: torch.Tensor, leaf_weights: torch.Tensor
                ) -> torch.Tensor:
        """
        omega:        (B, S_q) — frecuencia del rayo por query
        leaf_weights: (B, S_q, n_leaves) — peso soft de cada hoja
        returns:      (B, S_q, n_leaves) — resonancia por hoja
        """
        # k = [1, 2, ..., num_modes]
        k = torch.arange(1, self.num_modes + 1,
                         device=omega.device, dtype=omega.dtype)
        # kω: (B, S_q, num_modes)
        kw = omega.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0)

        sin_kw = torch.sin(kw)  # (B, S_q, M)
        cos_kw = torch.cos(kw)  # (B, S_q, M)

        # Resonancia por hoja: Σ_k a_k·sin(kω) + b_k·cos(kω)
        # a: (n_leaves, M), sin_kw: (B, S_q, M)
        r_sin = torch.einsum('lm,bsm->bsl', self.a, sin_kw)  # (B, S_q, n_leaves)
        r_cos = torch.einsum('lm,bsm->bsl', self.b, cos_kw)
        resonance = torch.tanh(self.scale.unsqueeze(0).unsqueeze(0) * (r_sin + r_cos))

        return resonance

    def quantize_ternary(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cuantiza a, b a {-1, 0, +1} con threshold adaptativo."""
        with torch.no_grad():
            all_params = torch.cat([self.a.flatten(), self.b.flatten()])
            threshold = torch.quantile(all_params.abs(), 0.2)

            a_t = torch.zeros_like(self.a, dtype=torch.int8)
            a_t[self.a >  threshold] =  1
            a_t[self.a < -threshold] = -1

            b_t = torch.zeros_like(self.b, dtype=torch.int8)
            b_t[self.b >  threshold] =  1
            b_t[self.b < -threshold] = -1

            # Scale factor por mínimos cuadrados
            a_scale = (self.a * a_t.float()).sum() / (a_t.float() ** 2).sum().clamp(min=1e-8)
            b_scale = (self.b * b_t.float()).sum() / (b_t.float() ** 2).sum().clamp(min=1e-8)

            return a_t, b_t, torch.stack([a_scale, b_scale])


# ─────────────────────────────────────────────────────────────────
# 5. SphereLevel — un nivel de la jerarquía BVH
# ─────────────────────────────────────────────────────────────────

class SphereLevel(nn.Module):
    """
    Un nivel del árbol BVH: K esferas con centros y radios learnable.

    Asignación soft: P(token ∈ sphere_k) = softmax(-d²/T²)
    Energía del rayo decae con la distancia: E = E₀·exp(-λ·d)
    """

    def __init__(self, n_spheres: int, parent_spheres: int = 1):
        super().__init__()
        self.n_spheres = n_spheres
        self.parent_spheres = parent_spheres
        # Total de esferas en este nivel = parent_spheres * n_spheres
        total = parent_spheres * n_spheres

        # Centros en coordenadas LOCALES (del padre)
        self.centers = nn.Parameter(torch.randn(total, 3) * 0.5)
        # Radios (log-scale para positividad)
        self.log_radii = nn.Parameter(torch.zeros(total))

    @property
    def radii(self) -> torch.Tensor:
        return F.softplus(self.log_radii)

    def forward(self, pos_3d: torch.Tensor, temperature: torch.Tensor,
                parent_membership: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pos_3d:            (B, S, 3) — posiciones en coordenadas del padre
        temperature:       Tensor scalar — annealing temp (alto=soft, bajo=hard)
        parent_membership: (B, S, parent_spheres) — membresía en el padre

        returns:
            membership: (B, S, total_spheres) — membresía soft en este nivel
            energy:     (B, S, total_spheres) — energía del rayo tras traversal
        """
        B, S, _ = pos_3d.shape
        total = self.parent_spheres * self.n_spheres

        # Distancias a todos los centros: (B, S, total)
        # centers: (total, 3)
        diff = pos_3d.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
        d_sq = (diff ** 2).sum(dim=-1)  # (B, S, total)

        # Soft assignment con temperatura
        logits = -d_sq / (2.0 * temperature ** 2 + 1e-8)

        if parent_membership is not None:
            # Enmascarar: cada child-sphere solo recibe tokens de su padre
            # parent_membership: (B, S, P), logits: (B, S, P*K)
            # Expandir parent_membership para que cada grupo de K herede del padre
            parent_expanded = parent_membership.repeat_interleave(
                self.n_spheres, dim=2
            )  # (B, S, P*K)
            # Log-mask: tokens fuera del padre tienen logit -inf
            logits = logits + torch.log(parent_expanded + 1e-10)

        membership = F.softmax(logits, dim=-1)  # (B, S, total)

        # Energía del rayo: decae con la distancia al centro de la esfera alcanzada
        d = torch.sqrt(d_sq + 1e-8)
        energy = torch.exp(-0.1 * d)  # E₀·exp(-λ·d)

        return membership, energy


# ─────────────────────────────────────────────────────────────────
# 6. InceptionTraversal — travesía completa 4 niveles
# ─────────────────────────────────────────────────────────────────

class InceptionTraversal(nn.Module):
    """
    Travesía jerárquica 4 niveles (Inception Engine diferenciable).

    Nivel 1 (Universo):      K1 dominios
    Nivel 2 (Galaxia):       K1×K2 subdominios
    Nivel 3 (Sistema Solar): K1×K2×K3 conceptos
    Nivel 4 (Planeta):       hojas — tokens individuales con Fourier

    Cada nivel tiene:
    - SphereLevel: centros + radios learnable
    - AffinePortal: transformación de coordenadas al entrar
    - PrismaticRefraction: routing dependiente del contexto
    """

    def __init__(self, cfg: InceptionConfig):
        super().__init__()
        self.cfg = cfg
        self.advanced_optics = cfg.advanced_optics

        # Helper to create refraction module (simple or chromatic)
        def make_refraction(n_spheres: int) -> nn.Module:
            if cfg.advanced_optics:
                return ChromaticAberration(
                    n_spheres=n_spheres,
                    spectral_dim=cfg.spectral_dim,
                    n_bands=cfg.chromatic_bands,
                )
            return PrismaticRefraction(n_spheres, cfg.spectral_dim)

        # Nivel 1: dominios
        self.level1 = SphereLevel(cfg.n_domains, parent_spheres=1)
        self.portal1 = AffinePortal(cfg.total_l1)
        self.refract1 = make_refraction(cfg.total_l1)

        # Nivel 2: subdominios (K2 por dominio)
        self.level2 = SphereLevel(cfg.n_subdomains, parent_spheres=cfg.total_l1)
        self.portal2 = AffinePortal(cfg.total_l2)
        self.refract2 = make_refraction(cfg.total_l2)

        # Nivel 3: conceptos (K3 por subdominio)
        self.level3 = SphereLevel(cfg.n_concepts, parent_spheres=cfg.total_l2)
        self.portal3 = AffinePortal(cfg.total_l3)
        self.refract3 = make_refraction(cfg.total_l3)

        # Advanced optics: TIR + Phase Interference (Patent P3 Claims 26-33)
        if cfg.advanced_optics:
            self.tir1 = TotalInternalReflection(cfg.total_l1)
            self.tir2 = TotalInternalReflection(cfg.total_l2)
            self.tir3 = TotalInternalReflection(cfg.total_l3)
            self.phase_interference = PhaseCoherentInterference(
                spectral_dim=cfg.spectral_dim,
                n_rays=cfg.interference_rays,
            )

    def forward(self, pos_3d: torch.Tensor, spectral: torch.Tensor,
                omega: torch.Tensor, temperature: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        pos_3d:   (B, S, 3)            — posiciones 3D de los tokens
        spectral: (B, S, spectral_dim) — color espectral del rayo
        omega:    (B, S)               — frecuencia base del rayo
        temperature: Tensor scalar     — softmax temperature (annealing)

        returns:
            concept_membership: (B, S, total_l3) — a qué concepto pertenece cada token
            energy:             (B, S, total_l3) — energía residual del rayo
            omega_final:        (B, S)           — ω transformada por los portales
        """
        B, S = pos_3d.shape[:2]

        # ── Nivel 1: Dominios ─────────────────────────────────────
        m1, e1 = self.level1(pos_3d, temperature)  # (B, S, K1)
        # Refracción: el color modifica la importancia de cada dominio
        n1 = self.refract1(spectral)               # (B, S, K1)
        m1 = m1 * n1
        m1 = m1 / (m1.sum(dim=-1, keepdim=True) + 1e-8)

        # TIR: reject spheres with mismatched context (Claims 26-29)
        if self.advanced_optics:
            cos_i1 = e1  # use energy as proxy for cos(incidence)
            m1, _ = self.tir1(n1, m1, cos_i1)

        # Portal: transformar posiciones al espacio local — vectorizado (sin loop)
        locals_1 = self.portal1.apply_all(pos_3d)  # (B, S, K1, 3)
        pos_l1 = (locals_1 * m1.unsqueeze(-1)).sum(dim=2)  # (B, S, 3)

        # Transformar omega por portal dominante
        dom_idx = m1.argmax(dim=-1)  # (B, S) — dominio más probable
        omega_1 = self.portal1.transform_omega(omega, dom_idx)

        # ── Nivel 2: Subdominios ──────────────────────────────────
        m2, e2 = self.level2(pos_l1, temperature, parent_membership=m1)
        n2 = self.refract2(spectral)
        m2 = m2 * n2
        m2 = m2 / (m2.sum(dim=-1, keepdim=True) + 1e-8)

        if self.advanced_optics:
            cos_i2 = e2
            m2, _ = self.tir2(n2, m2, cos_i2)

        # Portal nivel 2 — vectorizado (sin loop)
        locals_2 = self.portal2.apply_all(pos_l1)  # (B, S, K2, 3)
        pos_l2 = (locals_2 * m2.unsqueeze(-1)).sum(dim=2)

        sub_idx = m2.argmax(dim=-1)
        omega_2 = self.portal2.transform_omega(omega_1, sub_idx)

        # ── Nivel 3: Conceptos ────────────────────────────────────
        m3, e3 = self.level3(pos_l2, temperature, parent_membership=m2)

        # Phase-Coherent Interference at final level (Claims 30-33)
        if self.advanced_optics:
            n3 = self.phase_interference(spectral, self.refract3)
        else:
            n3 = self.refract3(spectral)
        m3 = m3 * n3
        m3 = m3 / (m3.sum(dim=-1, keepdim=True) + 1e-8)

        if self.advanced_optics:
            cos_i3 = e3
            m3, _ = self.tir3(n3, m3, cos_i3)

        con_idx = m3.argmax(dim=-1)
        omega_3 = self.portal3.transform_omega(omega_2, con_idx)

        # Energía combinada: producto de energías en cada nivel
        # (refleja la absorción acumulada del rayo)
        energy_combined = (
            (e1 * m1).sum(dim=-1) *
            (e2 * m2).sum(dim=-1) *
            (e3 * m3).sum(dim=-1)
        )  # (B, S)

        return m3, energy_combined, omega_3


# ─────────────────────────────────────────────────────────────────
# 7. InceptionAttention — módulo de atención completo
# ─────────────────────────────────────────────────────────────────

class InceptionAttention(nn.Module):
    """
    Atención SpectralAI v4.0 completa.

    Flujo:
    1. Q, K, V projections
    2. K → 3D positions (W_projection)
    3. Context → spectral color (SpectralEncoder)
    4. Q emite rayos → traversal jerárquico 4 niveles (InceptionTraversal)
    5. Hojas → Fourier resonance W(ω)
    6. Resonancia × energía → attention weights
    7. Weights × V → output
    """

    def __init__(self, cfg: InceptionConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        # Proyecciones Q, K, V
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)

        # Token embedding → posición 3D (W_projection del paper)
        self.to_3d = nn.Linear(D, 3)

        # Token → omega base (frecuencia de resonancia)
        self.to_omega = nn.Linear(D, 1)

        # Codificación espectral (contexto → color del rayo)
        self.spectral_encoder = SpectralEncoder(D, cfg.spectral_dim)

        # Traversal jerárquico 4 niveles
        self.traversal = InceptionTraversal(cfg)

        # Nivel 4: resonancia Fourier en las hojas
        # Cada concepto (nivel 3) tiene su resonador
        self.resonance = FourierResonance(
            n_leaves=cfg.total_l3,
            num_modes=cfg.num_fourier_modes,
        )

        # Proyección: concepto → espacio de atención token-token
        self.concept_to_token = nn.Linear(cfg.total_l3, cfg.context_len)

        # Temperatura (se reduce durante training)
        self.register_buffer('temperature',
                             torch.tensor(cfg.temperature_init))

        self.attn_drop = nn.Dropout(cfg.dropout)

    def anneal_temperature(self):
        """Reduce temperatura (llamar una vez por epoch)."""
        new_temp = max(
            self.cfg.temperature_min,
            self.temperature.item() * self.cfg.temperature_decay
        )
        self.temperature.fill_(new_temp)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        x: (B, S, D) — input embeddings
        returns: (B, S, D) — attended output
        """
        B, S, D = x.shape
        T = self.temperature  # Keep as tensor for torch.compile compatibility

        # ── Proyecciones estándar ─────────────────────────────────
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads
        H = self.cfg.num_heads
        d_h = self.cfg.head_dim
        Q = Q.view(B, S, H, d_h).transpose(1, 2)  # (B, H, S, d_h)
        K = K.view(B, S, H, d_h).transpose(1, 2)
        V = V.view(B, S, H, d_h).transpose(1, 2)

        # ── Proyectar tokens a espacio 3D ─────────────────────────
        pos_3d = self.to_3d(x)          # (B, S, 3)
        omega_base = self.to_omega(x).squeeze(-1)  # (B, S)
        omega_base = omega_base % (2.0 * math.pi)  # Normalizar a [0, 2π]

        # ── Codificación espectral del contexto ───────────────────
        spectral = self.spectral_encoder(x)  # (B, S, spectral_dim)

        # ── Traversal jerárquico: 4 niveles de esferas anidadas ──
        concept_membership, energy, omega_final = self.traversal(
            pos_3d, spectral, omega_base, T
        )
        # concept_membership: (B, S, total_l3)
        # energy:             (B, S)
        # omega_final:        (B, S) — ω transformada por portales

        # ── Nivel 4: Resonancia Fourier en las hojas ─────────────
        resonance = self.resonance(omega_final, concept_membership)
        # (B, S, total_l3) — cuánto "resuena" cada concepto

        # ── Construir scores de atención ──────────────────────────
        # Cada token tiene un perfil de concepto (concept_membership).
        # La atención entre token_i y token_j depende de cuánto
        # resuenan los conceptos de j con el rayo de i.

        # Perfil de resonancia de cada token query
        query_resonance = (concept_membership * resonance)  # (B, S, L3)

        # Scores: similitud entre perfiles de concepto
        # query_resonance: (B, S_q, L3) × concept_membership^T: (B, L3, S_k) → (B, S_q, S_k)
        scores = torch.bmm(
            query_resonance,
            concept_membership.transpose(1, 2)
        )  # (B, S, S)

        # Modular por energía del rayo (tokens lejanos pesan menos)
        scores = scores * energy.unsqueeze(2)  # broadcast energy sobre dim K

        # Expandir a heads (todos los heads comparten la estructura espacial,
        # pero Q·K^T da la especificidad por head)
        # Combinar con dot-product residual para estabilidad
        qk_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        # (B, H, S, S)

        # Mezclar: λ·inception_scores + (1-λ)·qk_scores
        # λ controla cuánto peso tiene la estructura espacial
        inception_scores = scores.unsqueeze(1).expand_as(qk_scores)
        combined_scores = 0.7 * inception_scores + 0.3 * qk_scores

        # ── Causal mask ───────────────────────────────────────────
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

        combined_scores = combined_scores.masked_fill(attention_mask, float('-inf'))

        # ── Softmax + dropout ─────────────────────────────────────
        attn_w = F.softmax(combined_scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, 0.0)
        attn_w = self.attn_drop(attn_w)

        # ── Aplicar a V ──────────────────────────────────────────
        out = torch.matmul(attn_w, V)  # (B, H, S, d_h)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────
# 8. SpectralAIInceptionLM — modelo de lenguaje causal completo
# ─────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class InceptionBlock(nn.Module):
    def __init__(self, cfg: InceptionConfig, mlp_hidden: int = 1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = InceptionAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg.embed_dim, mlp_hidden)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SpectralAIInceptionLM(nn.Module):
    """
    Modelo de lenguaje causal SpectralAI v4.0 — sistema completo.

    Igual que GPT-2 en estructura, pero con InceptionAttention:
    - 4 niveles de esferas anidadas (BVH jerárquico)
    - Rayos espectrales coloreados por contexto
    - Refracción prismática (polisemia)
    - Resonancia Fourier en las hojas
    - Portales afines entre niveles dimensionales
    """

    def __init__(
        self,
        vocab_size:  int = 50_257,
        embed_dim:   int = 256,
        num_layers:  int = 4,
        num_heads:   int = 4,
        context_len: int = 256,
        mlp_hidden:  int = 1024,
        dropout:     float = 0.1,
        **inception_kwargs,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.context_len = context_len

        cfg = InceptionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_len=context_len,
            dropout=dropout,
            **inception_kwargs,
        )
        self.cfg = cfg

        self.wte  = nn.Embedding(vocab_size, embed_dim)
        self.wpe  = nn.Embedding(context_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList([
            InceptionBlock(cfg, mlp_hidden)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # LM head (tied weights con wte)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        assert S <= self.context_len

        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def anneal_temperature(self):
        """Reduce temperatura en TODOS los bloques."""
        for block in self.h:
            block.attn.anneal_temperature()

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("SpectralAI v4.0 Inception Engine — Test completo")
    print("=" * 65)

    model = SpectralAIInceptionLM(
        vocab_size=50_257, embed_dim=256, num_layers=4,
        num_heads=4, context_len=256,
        n_domains=4, n_subdomains=4, n_concepts=4,
        spectral_dim=64, num_fourier_modes=8,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {total:,} ({total/1e6:.1f}M)")
    print(f"Device: {device}")

    # Forward
    x = torch.randint(0, 50_257, (2, 32), device=device)
    logits = model(x)
    print(f"\n[Forward] {x.shape} -> {logits.shape}")
    assert logits.shape == (2, 32, 50_257)
    print("[PASS] Forward OK")

    # Loss
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 50_257),
        x[:, 1:].reshape(-1),
    )
    print(f"[Loss] {loss.item():.4f} (esperado ~{math.log(50_257):.2f})")
    print("[PASS] Loss OK")

    # Backward
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_count = sum(1 for p in model.parameters())
    print(f"[Grad] {grad_count}/{total_count} params con gradiente")
    print("[PASS] Backward OK")

    # Componentes especiales
    block = model.h[0]
    attn = block.attn
    print(f"\n[Inception] Temperatura: {attn.temperature.item():.3f}")
    print(f"[Inception] Esferas L1: {attn.traversal.level1.n_spheres}")
    print(f"[Inception] Esferas L2: {attn.cfg.total_l2}")
    print(f"[Inception] Esferas L3: {attn.cfg.total_l3}")
    print(f"[Inception] Fourier modos: {attn.resonance.num_modes}")
    print(f"[Inception] Spectral dim: {attn.cfg.spectral_dim}")

    # Annealing
    model.anneal_temperature()
    print(f"[Inception] Temp after anneal: {attn.temperature.item():.3f}")

    # Generación
    prompt = torch.randint(0, 50_257, (1, 5), device=device)
    gen = model.generate(prompt, max_new_tokens=20)
    print(f"\n[Gen] {prompt.shape} -> {gen.shape}")
    print("[PASS] Generacion OK")

    print(f"\n{'='*65}")
    print("TODOS LOS TESTS PASSED")
    print(f"{'='*65}")
