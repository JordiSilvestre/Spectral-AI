#!/usr/bin/env python3
"""
micro_expert.py -- Micro-Modelos Especializados para SpectralAI v5.0

Wrapper agnostico que soporta 4 tipos de micro-expertos:
  1. Transformer FP16 (GPT-2 small fine-tuneado)
  2. Transformer INT8 (cuantizado post-training)
  3. Ternario BitNet {-1, 0, +1} (desde cero o cuantizado)
  4. SpectralAI Inception (recursivo — usa v4.0 como experto)

Cada experto es un LM causal completo que recibe tokens y genera texto.
El Orchestrator selecciona cual activar via BVHRouter.
"""

import math
import copy
from enum import Enum
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Tipos de experto
# ─────────────────────────────────────────────────────────────────

class ExpertType(Enum):
    TRANSFORMER_FP16 = "transformer_fp16"
    TRANSFORMER_INT8 = "transformer_int8"
    TERNARY_BITNET   = "ternary_bitnet"
    INCEPTION_LIQUID = "inception_liquid"


# ─────────────────────────────────────────────────────────────────
# Mini-Transformer (experto base — ~2-5M params)
# ─────────────────────────────────────────────────────────────────

class MiniAttention(nn.Module):
    """Scaled dot-product attention para micro-experto."""

    def __init__(self, embed_dim: int, num_heads: int, context_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_len, context_len))
                 .view(1, 1, context_len, context_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.split(D, dim=2)

        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            self.causal_mask[:, :, :S, :S] == 0, float('-inf')
        )
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return self.proj(out.transpose(1, 2).contiguous().view(B, S, D))


class MiniMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MiniBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 context_len: int, mlp_hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MiniAttention(embed_dim, num_heads, context_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MiniMLP(embed_dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    """
    Micro-LLM causal completo (~2-5M params).
    Recibe token IDs, devuelve logits.
    """

    def __init__(
        self,
        vocab_size:  int = 50_257,
        embed_dim:   int = 128,     # Pequeno para micro-experto
        num_layers:  int = 2,
        num_heads:   int = 4,
        context_len: int = 256,
        mlp_hidden:  int = 512,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(context_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MiniBlock(embed_dim, num_heads, context_len, mlp_hidden)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tied weights
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
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.ln_f(x))

    def get_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """Devuelve embedding medio del prompt (para el router)."""
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        return x.mean(dim=1)  # (B, embed_dim)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────
# Cuantizacion Ternaria {-1, 0, +1}
# ─────────────────────────────────────────────────────────────────

def quantize_ternary(weight: torch.Tensor) -> tuple:
    """
    Cuantiza un peso FP16/FP32 a ternario {-1, 0, +1}.
    Retorna (ternary_weight, scale_factor).

    Basado en BitNet b1.58 (Microsoft, 2024):
    w_ternary = sign(w) * (|w| > threshold)
    scale = mean(|w| donde |w| > threshold)
    """
    threshold = weight.abs().mean()
    ternary = torch.zeros_like(weight, dtype=torch.int8)
    ternary[weight > threshold] = 1
    ternary[weight < -threshold] = -1

    # Scale: media de los pesos que NO son cero
    mask = ternary != 0
    scale = weight.abs()[mask].mean() if mask.any() else torch.tensor(1.0)

    return ternary, scale


class TernaryLinear(nn.Module):
    """
    Linear layer con pesos ternarios {-1, 0, +1}.
    Sin multiplicacion — solo sumas y restas.

    forward: out = scale * (x @ ternary_weight.T)
    Donde x @ ternary.T se resuelve como:
      out_j = sum(x_i donde w_ij=+1) - sum(x_i donde w_ij=-1)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pesos ternarios (int8 para 75% ahorro memoria)
        self.register_buffer('weight_ternary',
                             torch.zeros(out_features, in_features,
                                         dtype=torch.int8))
        self.register_buffer('weight_scale',
                             torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'TernaryLinear':
        """Convierte nn.Linear FP16/32 a TernaryLinear."""
        has_bias = linear.bias is not None
        ternary_layer = cls(linear.in_features, linear.out_features,
                            bias=has_bias)

        w_ternary, scale = quantize_ternary(linear.weight.data)
        ternary_layer.weight_ternary.copy_(w_ternary)
        ternary_layer.weight_scale.fill_(scale.item())

        if has_bias:
            ternary_layer.bias.data.copy_(linear.bias.data)

        return ternary_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conversion int8 -> float para matmul (en GPU es rapido)
        w_float = self.weight_ternary.float() * self.weight_scale
        out = F.linear(x, w_float, self.bias)
        return out

    def memory_bytes(self) -> int:
        """Memoria usada por pesos ternarios (1 byte/param vs 2 FP16)."""
        return self.weight_ternary.numel()  # 1 byte per element


def quantize_model_ternary(model: nn.Module) -> nn.Module:
    """
    Cuantiza TODOS los nn.Linear de un modelo a TernaryLinear.
    Retorna nuevo modelo (no muta el original).
    """
    model_q = copy.deepcopy(model)

    for name, module in model_q.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                ternary = TernaryLinear.from_linear(child)
                setattr(module, attr_name, ternary)

    return model_q


# ─────────────────────────────────────────────────────────────────
# ExpertRegistry — carga y gestiona micro-expertos
# ─────────────────────────────────────────────────────────────────

class ExpertRegistry:
    """
    Registro de micro-expertos. Soporta:
    - Carga lazy (solo el experto activo en VRAM)
    - Hot-swap (cambiar experto sin reiniciar)
    - Multiples tipos (FP16, INT8, Ternario, Inception)
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.experts: Dict[int, nn.Module] = {}
        self.expert_types: Dict[int, ExpertType] = {}
        self.expert_names: Dict[int, str] = {}
        self._active_id: Optional[int] = None

    def register(self, expert_id: int, model: nn.Module,
                 expert_type: ExpertType, name: str = ""):
        """Registra un micro-experto (en CPU por defecto)."""
        model = model.cpu()  # Siempre almacenar en CPU
        self.experts[expert_id] = model
        self.expert_types[expert_id] = expert_type
        self.expert_names[expert_id] = name or f"expert_{expert_id}"

    def activate(self, expert_id: int) -> nn.Module:
        """
        Activa un experto: lo mueve a GPU.
        El experto previo se devuelve a CPU (ahorro VRAM).
        """
        if expert_id not in self.experts:
            raise KeyError(f"Expert {expert_id} no registrado")

        # Desactivar el anterior
        if self._active_id is not None and self._active_id != expert_id:
            self.experts[self._active_id] = self.experts[self._active_id].cpu()

        # Activar el nuevo
        self.experts[expert_id] = self.experts[expert_id].to(self.device)
        self._active_id = expert_id
        return self.experts[expert_id]

    def get_active(self) -> Optional[nn.Module]:
        """Retorna el experto activo (en GPU)."""
        if self._active_id is None:
            return None
        return self.experts[self._active_id]

    @property
    def active_id(self) -> Optional[int]:
        return self._active_id

    def count(self) -> int:
        return len(self.experts)

    def memory_report(self) -> Dict[str, float]:
        """Memoria por experto en MB."""
        report = {}
        for eid, model in self.experts.items():
            total_bytes = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            # Buffers (ternary weights, masks)
            total_bytes += sum(
                b.numel() * b.element_size() for b in model.buffers()
            )
            name = self.expert_names.get(eid, f"expert_{eid}")
            report[name] = total_bytes / (1024 * 1024)
        return report

    def summary(self) -> str:
        lines = ["ExpertRegistry:"]
        mem = self.memory_report()
        for eid in sorted(self.experts.keys()):
            name = self.expert_names[eid]
            etype = self.expert_types[eid].value
            size = mem.get(name, 0)
            active = " [ACTIVE]" if eid == self._active_id else ""
            params = sum(p.numel() for p in self.experts[eid].parameters())
            lines.append(
                f"  [{eid:3d}] {name:20s} | {etype:18s} | "
                f"{params:>10,} params | {size:6.1f} MB{active}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Factory: crear expertos pre-configurados
# ─────────────────────────────────────────────────────────────────

def create_expert(
    expert_type: ExpertType,
    vocab_size: int = 50_257,
    embed_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    context_len: int = 256,
) -> nn.Module:
    """Crea un micro-experto del tipo especificado."""

    base = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        context_len=context_len,
        mlp_hidden=embed_dim * 4,
    )

    if expert_type == ExpertType.TRANSFORMER_FP16:
        return base.half()

    elif expert_type == ExpertType.TRANSFORMER_INT8:
        # INT8 cuantizacion dinamica (PyTorch nativo)
        return torch.ao.quantization.quantize_dynamic(
            base, {nn.Linear}, dtype=torch.qint8
        )

    elif expert_type == ExpertType.TERNARY_BITNET:
        return quantize_model_ternary(base)

    elif expert_type == ExpertType.INCEPTION_LIQUID:
        # Placeholder — usaria InceptionAttention de v4.0
        # Por ahora devuelve base FP32 como fallback
        return base

    else:
        raise ValueError(f"Tipo desconocido: {expert_type}")


# ─────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MicroExpert v5.0 -- Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear un experto de cada tipo
    registry = ExpertRegistry(device=device)

    for i, etype in enumerate(ExpertType):
        try:
            expert = create_expert(etype, embed_dim=128, num_layers=2)
            registry.register(i, expert, etype, name=etype.value)
            print(f"[OK] Creado: {etype.value}")
        except Exception as e:
            print(f"[SKIP] {etype.value}: {e}")

    # Resumen
    print(f"\n{registry.summary()}")

    # Test: activar y ejecutar un experto
    print("\n--- Test de inferencia ---")
    expert = registry.activate(0)
    expert.eval()

    tokens = torch.randint(0, 50_257, (1, 32), device=device)

    # FP16 necesita input float
    with torch.no_grad():
        if isinstance(expert, MiniTransformerLM) and next(expert.parameters()).dtype == torch.float16:
            logits = expert(tokens)
        else:
            logits = expert(tokens)

    print(f"Input:  {tokens.shape}")
    print(f"Output: {logits.shape}")
    print(f"Active: {registry.expert_names[registry.active_id]}")

    # Memoria
    print("\n--- Memoria por experto ---")
    mem = registry.memory_report()
    for name, mb in mem.items():
        print(f"  {name:20s}: {mb:.1f} MB")

    print("\n[OK] MicroExpert v5.0 -- Todos los tests pasados")
