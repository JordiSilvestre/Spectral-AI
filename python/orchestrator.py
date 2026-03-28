#!/usr/bin/env python3
"""
orchestrator.py -- LiquidBit v5.0 "Orchestrator" Pipeline Completo

Flujo:
  1. Prompt -> embedding medio
  2. BVHRouter selecciona expert_id (O(log N) via 4 niveles BVH)
  3. ExpertRegistry activa micro-modelo (lazy loading GPU)
  4. Micro-experto genera respuesta
  5. Resultado final al usuario

Entrenamiento end-to-end:
  - Router aprende con Gumbel-Softmax + load balancing loss
  - Expertos aprenden con cross-entropy (cada uno su dominio)
  - Embeddings compartidos entre router y expertos
"""

import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import numpy as np

from bvh_router import BVHRouter, RouterConfig, RoutingResult
from micro_expert import (
    MiniTransformerLM, ExpertRegistry, ExpertType,
    create_expert, quantize_model_ternary, TernaryLinear,
)


# ─────────────────────────────────────────────────────────────────
# Orchestrator Config
# ─────────────────────────────────────────────────────────────────

class OrchestratorConfig:
    """Configuracion del sistema Router + Expertos."""
    __slots__ = (
        'vocab_size', 'router_embed_dim', 'expert_embed_dim',
        'n_experts', 'n_level1', 'n_level2', 'n_level3',
        'expert_layers', 'expert_heads', 'context_len',
        'spectral_dim', 'alpha_balance', 'alpha_router',
    )

    def __init__(
        self,
        vocab_size:       int   = 50_257,
        router_embed_dim: int   = 256,
        expert_embed_dim: int   = 128,
        n_level1:         int   = 4,
        n_level2:         int   = 4,
        n_level3:         int   = 4,
        expert_layers:    int   = 2,
        expert_heads:     int   = 4,
        context_len:      int   = 256,
        spectral_dim:     int   = 64,
        alpha_balance:    float = 0.01,  # Peso del load balancing loss
        alpha_router:     float = 0.1,   # Peso del router loss
    ):
        self.vocab_size       = vocab_size
        self.router_embed_dim = router_embed_dim
        self.expert_embed_dim = expert_embed_dim
        self.n_level1         = n_level1
        self.n_level2         = n_level2
        self.n_level3         = n_level3
        self.n_experts        = n_level1 * n_level2 * n_level3
        self.expert_layers    = expert_layers
        self.expert_heads     = expert_heads
        self.context_len      = context_len
        self.spectral_dim     = spectral_dim
        self.alpha_balance    = alpha_balance
        self.alpha_router     = alpha_router


# ─────────────────────────────────────────────────────────────────
# LiquidBit Orchestrator -- Pipeline completo
# ─────────────────────────────────────────────────────────────────

class LiquidBitOrchestrator(nn.Module):
    """
    Pipeline completo v5.0:
      Tokens -> Router BVH -> Micro-Experto -> Logits

    En training: todos los expertos entrenan en paralelo.
    En inferencia: solo el experto seleccionado se ejecuta.
    """

    def __init__(self, cfg: OrchestratorConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self._device = device

        # Embedding compartido para el router
        self.shared_embedding = nn.Embedding(cfg.vocab_size, cfg.router_embed_dim)
        self.shared_pos_embedding = nn.Embedding(cfg.context_len, cfg.router_embed_dim)

        # Router BVH
        router_cfg = RouterConfig(
            embed_dim=cfg.router_embed_dim,
            spectral_dim=cfg.spectral_dim,
            n_level1=cfg.n_level1,
            n_level2=cfg.n_level2,
            n_level3=cfg.n_level3,
        )
        self.router = BVHRouter(router_cfg)

        # Expert backbone compartido (1 solo modelo, condicionado por domain embedding)
        self.expert_backbone = MiniTransformerLM(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.expert_embed_dim,
            num_layers=cfg.expert_layers,
            num_heads=cfg.expert_heads,
            context_len=cfg.context_len,
            mlp_hidden=cfg.expert_embed_dim * 4,
        )

        # Domain embeddings: cada experto tiene un vector que condiciona el backbone
        self.domain_embeddings = nn.Embedding(cfg.n_experts, cfg.expert_embed_dim)

        # Proyeccion router_dim -> expert_dim (para inyectar routing info)
        self.router_to_expert = nn.Linear(cfg.router_embed_dim, cfg.expert_embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.shared_embedding.weight, std=0.02)
        nn.init.normal_(self.shared_pos_embedding.weight, std=0.02)

    def get_prompt_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """Tokens -> embedding medio para el router."""
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        emb = self.shared_embedding(idx) + self.shared_pos_embedding(pos)
        return emb.mean(dim=1)  # (B, D_router)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        idx:        (B, S) token IDs
        targets:    (B, S) target token IDs (para loss)
        domain_ids: (B,) domain labels [0..N_DOMAINS-1] (para routing supervision)

        returns:
            logits: (B, S, V) logits del experto seleccionado
            info:   dict con losses y metadata del routing
        """
        B, S = idx.shape

        # 1. Embedding del prompt para routing
        prompt_emb = self.get_prompt_embedding(idx)  # (B, D_router)

        # 2. Router BVH: seleccionar experto
        route = self.router(prompt_emb)  # RoutingResult

        # 3. Ejecutar backbone condicionado por dominio (1 SOLO forward pass)
        # El domain embedding se suma a cada token embedding dentro del backbone
        expert_ids = route.expert_id  # (B,)
        domain_emb = self.domain_embeddings(expert_ids)  # (B, expert_dim)
        routing_emb = self.router_to_expert(prompt_emb)  # (B, expert_dim)

        # Condicionar: domain + routing se suman como bias al backbone
        conditioning = domain_emb + routing_emb  # (B, expert_dim)

        # Forward del backbone con conditioning inyectado
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.expert_backbone.drop(
            self.expert_backbone.wte(idx) + self.expert_backbone.wpe(pos)
        )
        # Inyectar conditioning en cada posicion (cast to match backbone dtype)
        x = x + conditioning.unsqueeze(1).to(x.dtype)  # (B, S, expert_dim)

        for block in self.expert_backbone.blocks:
            x = block(x)

        logits = self.expert_backbone.lm_head(self.expert_backbone.ln_f(x))

        # 4. Calcular losses
        info = {
            'expert_id': route.expert_id,
            'confidence': route.confidence,
            'route_path': route.route_path,
        }

        if targets is not None:
            # Task loss: cross-entropy
            task_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
            # Balance loss: uso uniforme de expertos
            balance_loss = self.router.load_balancing_loss()

            info['task_loss'] = task_loss
            info['balance_loss'] = balance_loss

            total_loss = task_loss + self.cfg.alpha_balance * balance_loss

            # Routing supervision: si tenemos domain_ids, supervisar routing
            if domain_ids is not None:
                routing_loss = self._routing_supervision_loss(
                    route.expert_probs, domain_ids)
                info['routing_loss'] = routing_loss
                total_loss = total_loss + self.cfg.alpha_router * routing_loss

            info['total_loss'] = total_loss

        return logits, info

    def _routing_supervision_loss(
        self,
        expert_probs: torch.Tensor,  # (B, n_experts)
        domain_ids: torch.Tensor,    # (B,) in [0, N_DOMAINS)
    ) -> torch.Tensor:
        """
        Supervision de routing: los expertos del dominio correcto
        deben recibir mas probabilidad.

        Con 64 expertos y 4 dominios, dominio d -> expertos [d*16..(d+1)*16)
        Loss = -log(sum(probs del grupo correcto)) — queremos que la masa
        de probabilidad caiga en el grupo correcto.
        """
        n_domains = 4
        experts_per_domain = self.cfg.n_experts // n_domains  # 16

        # Crear mascara: para cada sample, 1s en los expertos de su dominio
        B = domain_ids.shape[0]
        # domain_start[b] = domain_ids[b] * experts_per_domain
        domain_start = domain_ids * experts_per_domain  # (B,)

        # Sumar probabilidad de los expertos del dominio correcto
        domain_prob = torch.zeros(B, device=expert_probs.device)
        for b in range(B):
            start = domain_start[b].item()
            domain_prob[b] = expert_probs[b, start:start + experts_per_domain].sum()

        # Loss: -log(prob del grupo correcto)
        routing_loss = -torch.log(domain_prob + 1e-8).mean()
        return routing_loss

    @torch.no_grad()
    def routing_accuracy(
        self,
        expert_ids: torch.Tensor,   # (B,)
        domain_ids: torch.Tensor,   # (B,)
        n_domains: int = 4,
    ) -> dict:
        """Medir si el routing envía cada dominio al grupo correcto."""
        experts_per_domain = self.cfg.n_experts // n_domains
        # predicted_domain = expert_id // experts_per_domain
        pred_domain = expert_ids // experts_per_domain
        correct = (pred_domain == domain_ids).float()

        # Accuracy global
        acc_global = correct.mean().item()

        # Accuracy por dominio
        acc_per_domain = {}
        for d in range(n_domains):
            mask = domain_ids == d
            if mask.any():
                acc_per_domain[d] = correct[mask].mean().item()
            else:
                acc_per_domain[d] = 0.0

        return {
            'accuracy': acc_global,
            'per_domain': acc_per_domain,
        }

    def anneal_temperature(self):
        self.router.anneal_temperature()

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 50) -> Tuple[torch.Tensor, int]:
        """Genera texto: ruta una vez, genera con backbone condicionado."""
        self.eval()

        # Routing: una sola vez para todo el prompt
        prompt_emb = self.get_prompt_embedding(idx)
        route = self.router(prompt_emb, hard=True)
        expert_id = route.expert_id[0].item()

        # Domain conditioning (fijo para toda la generacion)
        domain_emb = self.domain_embeddings(route.expert_id)  # (1, expert_dim)
        routing_emb = self.router_to_expert(prompt_emb)
        conditioning = domain_emb + routing_emb  # (1, expert_dim)

        ctx = self.expert_backbone.context_len

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -ctx:]
            S = idx_cond.shape[1]
            pos = torch.arange(S, device=idx.device).unsqueeze(0)
            x = self.expert_backbone.drop(
                self.expert_backbone.wte(idx_cond) + self.expert_backbone.wpe(pos)
            )
            x = x + conditioning.unsqueeze(1).to(x.dtype)
            for block in self.expert_backbone.blocks:
                x = block(x)
            logit = self.expert_backbone.lm_head(self.expert_backbone.ln_f(x))[:, -1, :]
            logit = logit / temperature
            if top_k > 0:
                v, _ = torch.topk(logit, top_k)
                logit[logit < v[:, -1:]] = float('-inf')
            probs = F.softmax(logit, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx, expert_id

    def param_count(self) -> Dict[str, int]:
        """Desglose de parametros."""
        router_p = sum(p.numel() for p in self.router.parameters())
        shared_p = sum(p.numel() for p in self.shared_embedding.parameters())
        shared_p += sum(p.numel() for p in self.shared_pos_embedding.parameters())
        backbone_p = sum(p.numel() for p in self.expert_backbone.parameters())
        domain_p = sum(p.numel() for p in self.domain_embeddings.parameters())
        proj_p = sum(p.numel() for p in self.router_to_expert.parameters())
        total = router_p + shared_p + backbone_p + domain_p + proj_p
        return {
            'router': router_p,
            'shared_embeddings': shared_p,
            'backbone': backbone_p,
            'domain_embeddings': domain_p,
            'router_to_expert': proj_p,
            'total': total,
            'active_inference': total,  # Todo activo (backbone compartido)
        }


# ─────────────────────────────────────────────────────────────────
# Dataset (reutilizado)
# ─────────────────────────────────────────────────────────────────

class WikiTextDataset(Dataset):
    """WikiText-2 tokenizado con tiktoken GPT-2 BPE."""

    def __init__(self, split: str = "train", seq_len: int = 256):
        self.seq_len = seq_len
        cache_path = Path(__file__).parent / f"wikitext2_{split}_tokens.npy"

        if cache_path.exists():
            print(f"[Data] Cargando cache {cache_path.name}...")
            self.tokens = np.load(str(cache_path))
        else:
            print(f"[Data] Tokenizando WikiText-2 {split}...")
            import tiktoken
            from datasets import load_dataset
            enc = tiktoken.get_encoding("gpt2")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            all_tokens = []
            for row in ds:
                text = row["text"].strip()
                if text:
                    all_tokens.extend(enc.encode(text))
            self.tokens = np.array(all_tokens, dtype=np.int32)
            np.save(str(cache_path), self.tokens)
            print(f"[Data] Guardado cache: {len(self.tokens):,} tokens")

        stride = seq_len // 2
        self.n_samples = max(1, (len(self.tokens) - seq_len - 1) // stride)
        self.stride = stride
        print(f"[Data] {split}: {len(self.tokens):,} tokens, "
              f"{self.n_samples} muestras (stride={stride})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        start = i * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end].astype(np.int64)
        return torch.from_numpy(chunk)


# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("LiquidBit v5.0 Orchestrator -- Training")
    print("=" * 70)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config -- empezamos con pocos expertos para validar
    cfg = OrchestratorConfig(
        vocab_size=50_257,
        router_embed_dim=256,
        expert_embed_dim=args.expert_dim,
        n_level1=args.n_l1,
        n_level2=args.n_l2,
        n_level3=args.n_l3,
        expert_layers=args.expert_layers,
        expert_heads=4,
        context_len=256,
    )

    model = LiquidBitOrchestrator(cfg, device).to(device)
    pc = model.param_count()

    print(f"Dominios:        {cfg.n_experts} ({cfg.n_level1}x{cfg.n_level2}x{cfg.n_level3})")
    print(f"Expert dim:      {cfg.expert_embed_dim}")
    print(f"Expert layers:   {cfg.expert_layers}")
    print(f"Params total:    {pc['total']:,} ({pc['total']/1e6:.1f}M)")
    print(f"  Router:        {pc['router']:,}")
    print(f"  Backbone:      {pc['backbone']:,}")
    print(f"  Domain embs:   {pc['domain_embeddings']:,}")
    print(f"Arquitectura:    Backbone compartido + {cfg.n_experts} domain embeddings")

    # Dataset
    train_ds = WikiTextDataset("train", seq_len=256)
    val_ds = WikiTextDataset("validation", seq_len=256)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_dl), eta_min=1e-5
    )

    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    log = []

    print("=" * 70)
    print("Training")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.router.reset_expert_counts()
        epoch_loss = 0.0
        epoch_balance = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, info = model(inp, tgt)
                    loss = info['total_loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, info = model(inp, tgt)
                loss = info['total_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += info['task_loss'].item()
            epoch_balance += info['balance_loss'].item()
            n_batches += 1

            T_val = model.router.temperature.item()
            pbar.set_postfix({
                'loss': f"{info['task_loss'].item():.4f}",
                'bal': f"{info['balance_loss'].item():.3f}",
                'T': f"{T_val:.3f}",
            })

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits, info = model(inp, tgt)
                else:
                    logits, info = model(inp, tgt)
                val_loss += info['task_loss'].item()
                val_n += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_balance = epoch_balance / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)
        ppl = math.exp(min(avg_val, 20))
        dt = time.time() - t0

        # Annealing
        model.anneal_temperature()

        # Expert usage
        counts = model.router.expert_counts
        active_experts = (counts > 0).sum().item()

        print(f"Epoch {epoch}/{args.epochs} | train={avg_train:.4f} | "
              f"bal={avg_balance:.4f} | val={avg_val:.4f} | ppl={ppl:.1f} | "
              f"experts_used={active_experts}/{cfg.n_experts} | "
              f"T={model.router.temperature.item():.3f} | {dt:.1f}s")

        entry = {
            'epoch': epoch,
            'train_loss': avg_train,
            'balance_loss': avg_balance,
            'val_loss': avg_val,
            'val_ppl': ppl,
            'active_experts': active_experts,
            'temperature': model.router.temperature.item(),
            'time_s': dt,
        }
        log.append(entry)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_dir = Path(__file__).parent.parent / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "orchestrator_best.pt")

    # Guardar log
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    log_path = data_dir / "orchestrator_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nGuardado: orchestrator_best.pt, {log_path.name}")

    # ── Generation test ──────────────────────────────────────────
    print("=" * 70)
    print("Generation Test")
    print("=" * 70)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    prompts = [
        "The history of",
        "In the year 2025",
        "Scientists discovered that",
    ]

    model.eval()
    for p in prompts:
        tokens = enc.encode(p)
        idx = torch.tensor([tokens], device=device)
        generated, expert_id = model.generate(idx, max_new_tokens=30)
        text = enc.decode(generated[0].cpu().tolist())
        path = model.router(model.get_prompt_embedding(idx), hard=True)
        print(f"Prompt:  {p}")
        print(f"Expert:  #{expert_id} (route: {path.route_path[0].tolist()})")
        print(f"Output:  {text}")
        print()

    # ── Comparativa con GPT-2 ────────────────────────────────────
    gpt2_log_path = Path(__file__).parent.parent / "data" / "gpt2_baseline_log.json"
    if gpt2_log_path.exists():
        with open(gpt2_log_path) as f:
            gpt2_log = json.load(f)
        # Soporta formato lista [{epoch,val_ppl,...}] o dict {training:[...]}
        gpt2_epochs = (
            gpt2_log if isinstance(gpt2_log, list)
            else gpt2_log.get('training', [])
        )
        print("=" * 70)
        print("COMPARATIVA: Orchestrator v5.0 vs GPT-2 Baseline")
        print("=" * 70)
        print(f"{'Epoch':>5} | {'GPT-2':>15} | {'Orchestrator':>15}")
        print("-" * 45)
        for i, entry in enumerate(log):
            if i < len(gpt2_epochs):
                g = gpt2_epochs[i]
                gpt2_ppl = g.get('val_ppl', g.get('ppl', 'N/A'))
            else:
                gpt2_ppl = "N/A"
            ppl_str = f"{gpt2_ppl:.1f}" if isinstance(gpt2_ppl, float) else str(gpt2_ppl)
            print(f"{entry['epoch']:5d} | {ppl_str:>15} | {entry['val_ppl']:>15.1f}")
        print("=" * 70)
        best_ppl = min(e['val_ppl'] for e in log)
        print(f"Orchestrator best ppl: {best_ppl:.1f}")
        print(f"Orchestrator params:   {pc['total']:,}")
        print(f"GPT-2 params:          16,090,880")

    return model, log


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiquidBit v5.0 Orchestrator")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--expert-dim", type=int, default=128,
                        help="Dimension de cada micro-experto")
    parser.add_argument("--expert-layers", type=int, default=2,
                        help="Capas por micro-experto")
    parser.add_argument("--n-l1", type=int, default=2,
                        help="Esferas nivel 1 (dominios)")
    parser.add_argument("--n-l2", type=int, default=2,
                        help="Esferas nivel 2 (subdominios)")
    parser.add_argument("--n-l3", type=int, default=2,
                        help="Esferas nivel 3 (conceptos)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
