#!/usr/bin/env python3
"""
train_dispersion.py — Entrenamiento de W_dispersion para routing polisémico
============================================================================

Resuelve el problema crítico: W_dispersion sin entrenar da 11% accuracy
(routing aleatorio). Este script lo lleva a >70%.

ARQUITECTURA:
  - PolysemicRouter: aprende qué "sentido" tiene cada token según su contexto
  - W_dispersion: matriz [num_senses, embed_dim] — el corazón del routing
  - auxiliary_loss: load balancing + entropía — evita colapso de representación

DATOS:
  - Usa GloVe-300d (embeddings_full.npy) como representación base
  - Genera ejemplos sintéticos de polisemia con palabras reales del vocab
  - Evalúa: dado el embedding de un token, ¿ruteamos al sentido correcto?

USO:
    python train_dispersion.py
    python train_dispersion.py --epochs 200 --num-senses 8 --lr 1e-3
    python train_dispersion.py --eval-only --checkpoint w_dispersion.pt

SALIDA:
    w_dispersion.pt      — pesos entrenados (torch state_dict)
    w_dispersion.npy     — W_dispersion como numpy para C++
    train_results.json   — curva de training
"""

import sys
import os
import json
import argparse
import time
import math
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    print("ERROR: PyTorch requerido. Instalar con:")
    print("  pip install torch")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent

# ============================================================================
# 1. Carga de embeddings
# ============================================================================

def load_embeddings(script_dir: Path):
    vocab_path = script_dir / "vocab.txt"
    emb_path   = script_dir / "embeddings_full.npy"

    if not vocab_path.exists() or not emb_path.exists():
        print("ERROR: vocab.txt o embeddings_full.npy no encontrado.")
        print("  Ejecutar primero: python download_embeddings_v2.py")
        sys.exit(1)

    vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    emb   = np.load(str(emb_path)).astype(np.float32)
    word2id = {w: i for i, w in enumerate(vocab)}
    print(f"[data] {len(vocab)} palabras x {emb.shape[1]}d cargadas")
    return vocab, emb, word2id


# ============================================================================
# 2. Dataset sintético de polisemia
# ============================================================================

# Grupos de palabras que son "el mismo concepto en distintos sentidos"
# Cada grupo = un sentido. El router debe aprender a distinguirlos.
POLYSEMY_GROUPS = {
    # sentido 0 — programacion
    0: ["python", "code", "function", "class", "loop", "array", "compiler",
        "algorithm", "variable", "module", "script", "debug", "syntax", "binary"],
    # sentido 1 — musica
    1: ["music", "rhythm", "beat", "melody", "chord", "guitar", "piano",
        "drum", "harmony", "tempo", "song", "bass", "note", "pitch"],
    # sentido 2 — fisica
    2: ["physics", "force", "energy", "mass", "gravity", "wave", "field",
        "quantum", "photon", "electron", "particle", "orbit", "velocity", "momentum"],
    # sentido 3 — finanzas
    3: ["bank", "money", "market", "stock", "trade", "capital", "loan",
        "credit", "invest", "fund", "price", "bond", "rate", "profit"],
    # sentido 4 — naturaleza
    4: ["tree", "river", "mountain", "ocean", "forest", "animal", "plant",
        "sky", "earth", "sun", "rain", "wind", "lake", "stone"],
    # sentido 5 — medicina
    5: ["doctor", "hospital", "disease", "treatment", "blood", "brain",
        "heart", "surgery", "patient", "drug", "cell", "gene", "virus", "cure"],
    # sentido 6 — politica
    6: ["government", "president", "election", "democracy", "law", "policy",
        "parliament", "vote", "nation", "constitution", "party", "minister", "rights", "state"],
    # sentido 7 — comida
    7: ["food", "bread", "meat", "fruit", "vegetable", "cook", "recipe",
        "taste", "sweet", "salt", "oil", "drink", "meal", "kitchen"],
}


def build_dataset(vocab, emb, word2id, num_senses=8, noise_std=0.05, seed=42):
    """
    Construye dataset (X, y) donde:
      X[i] = embedding de una palabra + ruido gaussiano ligero
      y[i] = etiqueta del sentido (0..num_senses-1)

    El ruido simula que el mismo token puede tener leve variación de contexto.
    """
    np.random.seed(seed)
    X_list, y_list, words_list = [], [], []
    dim = emb.shape[1]

    for sense_id in range(num_senses):
        if sense_id not in POLYSEMY_GROUPS:
            continue
        group = POLYSEMY_GROUPS[sense_id]
        for word in group:
            if word not in word2id:
                continue
            idx = word2id[word]
            vec = emb[idx].copy()
            # Cada palabra genera N_AUG variantes con ruido
            for _ in range(20):
                noisy = vec + np.random.randn(dim).astype(np.float32) * noise_std
                # Normalizar
                noisy /= (np.linalg.norm(noisy) + 1e-8)
                X_list.append(noisy)
                y_list.append(sense_id)
                words_list.append(word)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/val split 80/20
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[data] Dataset: {len(X_train)} train / {len(X_val)} val | {num_senses} sentidos")
    return X_train, y_train, X_val, y_val


# ============================================================================
# 3. Modelo: PolysemicRouter
# ============================================================================

class PolysemicRouter(nn.Module):
    """
    Router polisémico que aprende a qué sentido pertenece un embedding.

    Arquitectura:
      - W_dispersion: [num_senses, embed_dim] — proyección lineal principal
      - sense_bias:   [num_senses]            — sesgo por sentido
      - context_proj: pequeña MLP [embed_dim -> 64 -> num_senses]

    El routing usa soft-assignment (softmax) durante training para que los
    gradientes fluyan. En inferencia se puede usar argmax (hard routing).
    """

    def __init__(self, embed_dim: int, num_senses: int):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_senses = num_senses

        # W_dispersion: el corazon del sistema
        # Inicializar con valores pequeños para evitar saturación
        self.W_dispersion = nn.Parameter(
            torch.randn(num_senses, embed_dim) * 0.01,
            requires_grad=True
        )

        # MLP de contexto: captura interacciones no-lineales
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_senses),
        )

        # Combinar W_dispersion lineal + MLP
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, embed_dim]
        returns: logits [batch, num_senses]
        """
        # Rama 1: proyección lineal via W_dispersion
        # [batch, num_senses]
        linear_logits = x @ self.W_dispersion.t()

        # Rama 2: MLP de contexto
        mlp_logits = self.context_mlp(x)

        # Combinar con alpha aprendible
        alpha = torch.sigmoid(self.alpha)
        logits = alpha * linear_logits + (1 - alpha) * mlp_logits
        return logits

    def route_soft(self, x: torch.Tensor):
        """Soft routing (para training): pesos suaves via softmax."""
        return F.softmax(self.forward(x), dim=-1)

    def route_hard(self, x: torch.Tensor):
        """Hard routing (para inferencia): sentido más probable."""
        return torch.argmax(self.forward(x), dim=-1)


# ============================================================================
# 4. Loss functions
# ============================================================================

def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy estándar."""
    return F.cross_entropy(logits, targets)


def auxiliary_loss(logits: torch.Tensor, lambda_balance: float = 0.01,
                   lambda_entropy: float = 0.001) -> torch.Tensor:
    """
    Loss de regularización para evitar colapso de routing.

    Dos componentes:
    1. Load balancing: todos los sentidos deben usarse por igual.
       Penaliza cuando un sentido monopoliza el batch.
    2. Entropía: queremos decisiones "claras" pero no extremas.
       Penaliza entropía muy alta (indeciso) o muy baja (colapso).

    Fuente: documentos de diseño (1.docx, 2.docx) — standard MoE technique.
    """
    probs = F.softmax(logits, dim=-1)  # [batch, num_senses]

    # 1. Load balancing: uso promedio debe ser uniforme
    avg_usage   = probs.mean(dim=0)  # [num_senses]
    target_usage = torch.ones_like(avg_usage) / probs.shape[1]
    balance_loss = F.kl_div(
        (avg_usage + 1e-8).log(),
        target_usage,
        reduction="batchmean"
    )

    # 2. Entropía: sweet spot alrededor de log(num_senses)/2
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    target_entropy = math.log(probs.shape[1]) * 0.5
    entropy_penalty = (entropy - target_entropy).abs()

    return lambda_balance * balance_loss + lambda_entropy * entropy_penalty


def total_loss(logits, targets, lambda_balance=0.01, lambda_entropy=0.001):
    cls  = classification_loss(logits, targets)
    aux  = auxiliary_loss(logits, lambda_balance, lambda_entropy)
    return cls + aux, cls.item(), aux.item()


# ============================================================================
# 5. Training loop
# ============================================================================

def train(model, X_train, y_train, X_val, y_val,
          epochs=150, lr=1e-3, batch_size=64,
          lambda_balance=0.01, lambda_entropy=0.001,
          device="cpu", verbose=True):
    """
    Entrena el PolysemicRouter con early stopping.
    Retorna historial de training.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t   = torch.from_numpy(X_val).to(device)
    y_val_t   = torch.from_numpy(y_val).to(device)

    n_train   = len(X_train)
    history   = []
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_cls  = 0.0
        epoch_aux  = 0.0
        n_batches  = 0

        # Mini-batches
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx   = perm[start:start + batch_size]
            xb    = X_train_t[idx]
            yb    = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss, cls_v, aux_v = total_loss(logits, yb, lambda_balance, lambda_entropy)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls  += cls_v
            epoch_aux  += aux_v
            n_batches  += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds  = val_logits.argmax(dim=-1)
            val_acc    = (val_preds == y_val_t).float().mean().item() * 100

            train_logits = model(X_train_t)
            train_preds  = train_logits.argmax(dim=-1)
            train_acc    = (train_preds == y_train_t).float().mean().item() * 100

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_cls  = epoch_cls  / max(n_batches, 1)
        avg_aux  = epoch_aux  / max(n_batches, 1)

        history.append({
            "epoch": epoch, "loss": avg_loss, "cls_loss": avg_cls,
            "aux_loss": avg_aux, "train_acc": train_acc, "val_acc": val_acc,
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs} | loss={avg_loss:.4f} "
                  f"(cls={avg_cls:.4f} aux={avg_aux:.4f}) | "
                  f"train={train_acc:.1f}% val={val_acc:.1f}%")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_acc


# ============================================================================
# 6. Evaluación detallada
# ============================================================================

def evaluate(model, X_val, y_val, device="cpu", num_senses=8):
    """Imprime matriz de confusion y métricas por sentido."""
    model = model.cpu()
    model.eval()
    X_t = torch.from_numpy(X_val)
    y_t = torch.from_numpy(y_val)

    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=-1).numpy()

    y_np = y_val

    print(f"\n  Accuracy global: {(preds == y_np).mean() * 100:.1f}%")
    print(f"\n  Por sentido:")

    sense_names = {
        0: "programacion", 1: "musica", 2: "fisica",
        3: "finanzas", 4: "naturaleza", 5: "medicina",
        6: "politica", 7: "comida"
    }

    for s in range(num_senses):
        mask = y_np == s
        if mask.sum() == 0:
            continue
        acc_s = (preds[mask] == s).mean() * 100
        name  = sense_names.get(s, f"sentido_{s}")
        bar   = "#" * int(acc_s / 5)
        print(f"    sentido {s} ({name:<14}): {acc_s:5.1f}%  {bar}")

    # Uso de sentidos (load balancing)
    usage = np.bincount(preds, minlength=num_senses) / len(preds) * 100
    print(f"\n  Uso de sentidos (debe ser ~{100/num_senses:.0f}% cada uno):")
    for s in range(num_senses):
        bar  = "#" * int(usage[s] / 2)
        name = sense_names.get(s, f"sentido_{s}")
        print(f"    sentido {s} ({name:<14}): {usage[s]:5.1f}%  {bar}")


# ============================================================================
# 7. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpectralAI W_dispersion trainer")
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--num-senses",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lambda-balance", type=float, default=0.01)
    parser.add_argument("--lambda-entropy", type=float, default=0.001)
    parser.add_argument("--noise-std",    type=float, default=0.05)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--checkpoint",   type=str,   default="w_dispersion.pt")
    parser.add_argument("--output-npy",   type=str,   default="w_dispersion.npy")
    parser.add_argument("--results-json", type=str,   default="train_results.json")
    parser.add_argument("--eval-only",    action="store_true")
    parser.add_argument("--no-cuda",      action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"
    if not args.no_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[train] CPU mode")

    # Cargar datos
    vocab, emb, word2id = load_embeddings(SCRIPT_DIR)
    embed_dim = emb.shape[1]

    # Dataset
    X_train, y_train, X_val, y_val = build_dataset(
        vocab, emb, word2id,
        num_senses=args.num_senses,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    # Modelo
    model = PolysemicRouter(embed_dim=embed_dim, num_senses=args.num_senses)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] PolysemicRouter: {n_params:,} params | embed_dim={embed_dim} | senses={args.num_senses}")

    # Baseline antes de training
    print("\n[eval] Accuracy ANTES del training:")
    model.eval()
    with torch.no_grad():
        X_v = torch.from_numpy(X_val)
        y_v = torch.from_numpy(y_val)
        preds_before = model(X_v).argmax(dim=-1)
        acc_before = (preds_before == y_v).float().mean().item() * 100
    print(f"  Val accuracy (random init): {acc_before:.1f}%  (esperado ~{100/args.num_senses:.0f}%)")

    if args.eval_only:
        checkpoint = SCRIPT_DIR / args.checkpoint
        if checkpoint.exists():
            model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
            print(f"[eval] Cargado checkpoint: {checkpoint}")
        evaluate(model, X_val, y_val, device="cpu", num_senses=args.num_senses)
        return

    # Training
    print(f"\n[train] Iniciando training: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    print(f"[train] lambda_balance={args.lambda_balance}, lambda_entropy={args.lambda_entropy}")
    print()

    t0 = time.perf_counter()
    history, best_val_acc = train(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_balance=args.lambda_balance,
        lambda_entropy=args.lambda_entropy,
        device=device,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n[train] Completado en {elapsed:.1f}s | Best val accuracy: {best_val_acc:.1f}%")

    # Evaluación final detallada
    print("\n[eval] Evaluacion final detallada:")
    evaluate(model, X_val, y_val, device="cpu", num_senses=args.num_senses)

    # Verificar objetivo
    if best_val_acc >= 70.0:
        print(f"\n  OBJETIVO CUMPLIDO: {best_val_acc:.1f}% >= 70%")
    else:
        print(f"\n  ADVERTENCIA: {best_val_acc:.1f}% < 70%. Considerar mas epochs o ajustar lr.")

    # Guardar checkpoint
    ckpt_path = SCRIPT_DIR / args.checkpoint
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"\n[save] Checkpoint: {ckpt_path}")

    # Exportar W_dispersion como numpy (para C++/OptiX)
    W_np = model.W_dispersion.detach().cpu().numpy()
    npy_path = SCRIPT_DIR / args.output_npy
    np.save(str(npy_path), W_np)
    print(f"[save] W_dispersion.npy: {npy_path} — shape={W_np.shape}")

    # Guardar historial
    results = {
        "config": vars(args),
        "acc_before": acc_before,
        "best_val_acc": best_val_acc,
        "embed_dim": embed_dim,
        "n_params": n_params,
        "training_time_s": elapsed,
        "history": history,
    }
    json_path = SCRIPT_DIR / args.results_json
    with open(str(json_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Resultados: {json_path}")

    # Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN W_DISPERSION TRAINING")
    print(f"{'='*60}")
    print(f"  Antes:  {acc_before:.1f}% accuracy (init aleatorio)")
    print(f"  Despues: {best_val_acc:.1f}% accuracy (entrenado)")
    print(f"  Mejora:  +{best_val_acc - acc_before:.1f} puntos porcentuales")
    print(f"  W_dispersion shape: {W_np.shape}  (exportado a .npy para C++)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
