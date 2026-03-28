#!/usr/bin/env python3
"""
train_liquidbit.py — Pipeline de entrenamiento end-to-end de LiquidBit v4.0

PIPELINE COMPLETO:
==================
1. Cargar embeddings GloVe-300d y proyectar a 3D (PCA)
2. Construir árbol OHBSC inicial (clustering difuso con solapamiento)
3. Entrenar conjuntamente:
   - sphere_centers y sphere_radii (geometría del árbol BSH)
   - fourier_coeffs a[], b[] (SemanticStrings en los nodos hoja)
   - W_projection (matriz de proyección D→3D aprendible)
4. Re-clustering OHBSC cada K epochs (actualiza estructura del árbol)
5. Exportar modelo entrenado: ohbsc_tree.json + fourier_coeffs.npy

FUNCIÓN DE PÉRDIDA TOTAL:
==========================
L_total = L_task + α·L_spatial

L_task    = CrossEntropy(predicciones, etiquetas) — tarea de clasificación semántica
L_spatial = β·L_prox + γ·L_cover + δ·L_inter + η·L_reg

USO:
====
  python train_liquidbit.py --epochs 200 --lr 1e-3 --batch-size 64
  python train_liquidbit.py --eval-only  # solo evaluar con modelo guardado

SALIDAS:
========
  python/trained_model.json    — configuración del modelo entrenado
  python/sphere_params.npy     — centros y radios de esferas [K, 4]
  python/fourier_coeffs.npy    — coeficientes Fourier de leaves [L, 2*M]
  python/w_projection.npy      — matriz de proyección D→3D [3, D]
  python/training_curve.json   — métricas por epoch

@author LiquidBit Zero-Matrix Team
@date 2026
"""

import sys
import os
import json
import time
import argparse
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[ERROR] PyTorch no disponible. Instalar: pip install torch")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Imports locales
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ohbsc import OHBSCBuilder, OHBSCNode, NodeType
    from spatial_loss import SpatialLoss, SpatialLossConfig
    from fuzzy_bsh_autograd import FuzzyBSHLayer
except ImportError as e:
    print(f"[ERROR] Import fallido: {e}")
    print("        Asegúrate de estar en el directorio python/ con todos los archivos")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Modelo: LiquidBitModel
# ─────────────────────────────────────────────────────────────────────────────

class LiquidBitModel(nn.Module):
    """
    Modelo LiquidBit v4.0 completo para entrenamiento.

    Parámetros entrenables:
        W_projection:    [3, embed_dim]   — proyección learnable D→3D
        sphere_centers:  [K, 3]           — centros de esferas (vía FuzzyBSHLayer)
        sphere_radii:    [K]              — radios de esferas
        fourier_a:       [K, num_modes]   — coeficientes seno de SemanticStrings
        fourier_b:       [K, num_modes]   — coeficientes coseno de SemanticStrings
        output_scales:   [K]              — escalas de salida por nodo hoja
    """

    def __init__(
        self,
        embed_dim:  int = 300,
        num_modes:  int = 8,
        num_spheres: int = 3,
        num_classes: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_modes   = num_modes
        self.num_spheres = num_spheres

        # ── Proyección D→3D learnable ────────────────────────────────────
        # Inicializar desde PCA (si disponible) o aleatorio
        self.W_projection = nn.Parameter(
            torch.randn(3, embed_dim) * 0.01
        )

        # ── Geometría BSH (via FuzzyBSHLayer) ───────────────────────────
        self.fuzzy_bsh = FuzzyBSHLayer(
            n_spheres = num_spheres,
            embed_dim = 3           # trabaja en espacio 3D proyectado
        )
        # Temperatura de membresía difusa (annealing durante training)
        self.register_buffer("temperature", torch.tensor(temperature))

        # ── Radios de esferas ────────────────────────────────────────────
        self.sphere_radii = nn.Parameter(
            torch.ones(num_spheres)
        )

        # ── Coeficientes Fourier de SemanticStrings ──────────────────────
        # a_k, b_k para k=1..num_modes, por nodo esfera
        self.fourier_a = nn.Parameter(
            torch.randn(num_spheres, num_modes) * 0.1
        )
        self.fourier_b = nn.Parameter(
            torch.randn(num_spheres, num_modes) * 0.1
        )
        self.output_scales = nn.Parameter(
            torch.ones(num_spheres)
        )

        # ── Clasificador de tarea ────────────────────────────────────────
        # Mapea la resonancia acumulada → predicciones de clase
        self.task_head = nn.Sequential(
            nn.Linear(num_spheres, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def project_to_3d(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Proyecta embeddings de D→3D con la matriz learnable."""
        # [N, D] × [D, 3] → [N, 3]
        return F.normalize(embeddings @ self.W_projection.T, dim=1)

    def fourier_resonance(
        self,
        memberships: torch.Tensor,   # [N, K]
        omega:       torch.Tensor,   # [N] frecuencias de contexto
    ) -> torch.Tensor:
        """
        Calcula resonancia Fourier ponderada por membresía.

        W(omega) = outputScale · tanh(Σ a_k·sin(kω) + b_k·cos(kω))
        Output = Σ_k membership_{i,k} · W_k(omega_i)

        Returns:
            resonance: [N, K] valores de resonancia por nodo
        """
        N = omega.shape[0]
        K = self.num_spheres

        # Expandir omega para todos los modos: [N, K, M]
        omega_expanded = omega.unsqueeze(1).unsqueeze(2).expand(N, K, self.num_modes)
        modes = torch.arange(1, self.num_modes + 1,
                             device=omega.device, dtype=torch.float32)
        modes = modes.unsqueeze(0).unsqueeze(0)  # [1, 1, M]

        kw = modes * omega_expanded  # [N, K, M]

        # Resonancia Fourier: [N, K]
        a_k = self.fourier_a.unsqueeze(0).expand(N, K, self.num_modes)  # [N, K, M]
        b_k = self.fourier_b.unsqueeze(0).expand(N, K, self.num_modes)  # [N, K, M]

        sum_fourier = (a_k * torch.sin(kw) + b_k * torch.cos(kw)).sum(dim=2)  # [N, K]
        resonance   = self.output_scales.unsqueeze(0) * torch.tanh(sum_fourier)  # [N, K]

        return resonance * memberships  # [N, K] — ponderado por membresía

    def forward(
        self,
        embeddings: torch.Tensor,    # [N, D]
        omega:      torch.Tensor,    # [N] frecuencias de contexto
    ) -> Dict_:
        # Proyectar a 3D
        pos_3d = self.project_to_3d(embeddings)  # [N, 3]

        # Membresías difusas del BSH
        memberships = self.fuzzy_bsh.compute_memberships(pos_3d, self.temperature.item())  # [N, K]

        # Resonancia Fourier ponderada
        resonance = self.fourier_resonance(memberships, omega)  # [N, K]

        # Predicciones de tarea
        logits = self.task_head(resonance)  # [N, num_classes]

        return {
            "logits":      logits,
            "pos_3d":      pos_3d,
            "memberships": memberships,
            "resonance":   resonance,
        }

    def anneal_temperature(self, factor: float = 0.995):
        """Annealing gradual de la temperatura del BSH."""
        with torch.no_grad():
            self.temperature.mul_(factor)
            # No dejar que la temperatura caiga demasiado (evitar colapso)
            self.temperature.clamp_(min=0.01)

    def get_sphere_centers(self) -> torch.Tensor:
        """Acceso a los centros de esferas del FuzzyBSHLayer."""
        return self.fuzzy_bsh.sphere_centers

    def sphere_params_numpy(self) -> np.ndarray:
        """Exporta [K, 4] = [cx, cy, cz, r] para C++/OptiX."""
        centers = self.get_sphere_centers().detach().cpu().numpy()  # [K, 3]
        radii   = self.sphere_radii.detach().cpu().numpy()          # [K]
        return np.hstack([centers, radii[:, np.newaxis]])           # [K, 4]

    def fourier_params_numpy(self) -> np.ndarray:
        """Exporta [K, 2*M] = [a_1..a_M, b_1..b_M] para C++/OptiX."""
        a = self.fourier_a.detach().cpu().numpy()  # [K, M]
        b = self.fourier_b.detach().cpu().numpy()  # [K, M]
        return np.hstack([a, b])                    # [K, 2M]


# Alias para la type annotation dentro de la clase
Dict_ = dict


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: pares (embedding, omega, etiqueta_semantica)
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_GROUPS = {
    "programming": ["python", "function", "loop", "variable", "class", "array",
                    "code", "algorithm", "debug", "compiler", "syntax", "object"],
    "music":       ["rhythm", "melody", "chord", "tempo", "beat", "bass",
                    "treble", "harmony", "pitch", "scale", "tone", "note"],
    "physics":     ["energy", "mass", "force", "velocity", "gravity", "orbit",
                    "field", "vector", "quantum", "particle", "wave", "frequency"],
    "finance":     ["market", "stock", "bond", "interest", "rate", "dividend",
                    "portfolio", "equity", "asset", "trade", "profit", "loss"],
    "nature":      ["forest", "river", "mountain", "ocean", "climate", "weather",
                    "species", "plant", "animal", "ecosystem", "soil", "wind"],
    "medicine":    ["cell", "protein", "tissue", "drug", "disease", "treatment",
                    "symptom", "diagnosis", "therapy", "vaccine", "gene", "organ"],
    "politics":    ["election", "government", "policy", "democracy", "law", "vote",
                    "constitution", "parliament", "leader", "party", "reform", "state"],
    "food":        ["bread", "fruit", "vegetable", "spice", "recipe", "cook",
                    "flavor", "salt", "sweet", "sauce", "dish", "ingredient"],
}

def build_dataset(
    embeddings: np.ndarray,
    vocab: list,
    n_augments: int = 20,
    noise_std: float = 0.05
) -> tuple:
    """
    Construye dataset de entrenamiento a partir de grupos semánticos y GloVe.

    Returns:
        (X, labels, group_names) donde X es [N, embed_dim] y labels es [N]
    """
    vocab_lower = [w.lower() for w in vocab]
    groups = list(SEMANTIC_GROUPS.keys())
    X_list, y_list = [], []

    for label_id, group_name in enumerate(groups):
        words = SEMANTIC_GROUPS[group_name]
        for word in words:
            if word in vocab_lower:
                idx = vocab_lower.index(word)
                emb = embeddings[idx]
                # Augmentación con ruido gaussiano
                for _ in range(n_augments):
                    noisy = emb + np.random.randn(*emb.shape) * noise_std * np.linalg.norm(emb)
                    X_list.append(noisy.astype(np.float32))
                    y_list.append(label_id)

    if len(X_list) == 0:
        print("[WARN] No se encontraron palabras del vocabulario en los grupos semánticos")
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32), np.zeros(0, dtype=int), groups

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, groups


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 65)
    print(" LiquidBit v4.0 — Training Pipeline")
    print("=" * 65)

    # ── 1. Cargar embeddings ─────────────────────────────────────────────
    emb_path   = os.path.join(SCRIPT_DIR, "embeddings_full.npy")
    vocab_path = os.path.join(SCRIPT_DIR, "vocab.txt")

    if not os.path.exists(emb_path):
        print(f"[ERROR] {emb_path} no encontrado.")
        print("        Ejecutar: python download_embeddings_v2.py --dim 300")
        sys.exit(1)

    print(f"[load] Cargando embeddings desde {emb_path}...")
    embeddings = np.load(emb_path)
    vocab = []
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [l.strip() for l in f.readlines()]

    embed_dim = embeddings.shape[1]
    print(f"[load] {len(embeddings)} embeddings × {embed_dim}d, vocab={len(vocab)}")

    # ── 2. Construir dataset ─────────────────────────────────────────────
    print(f"\n[data] Construyendo dataset (n_augments={args.n_augments})...")
    X, y, group_names = build_dataset(embeddings, vocab, n_augments=args.n_augments)
    num_classes = len(group_names)
    print(f"[data] {len(X)} muestras, {num_classes} clases")

    # Split train/val
    split = int(0.8 * len(X))
    idx = np.random.permutation(len(X))
    X_train, y_train = X[idx[:split]],  y[idx[:split]]
    X_val,   y_val   = X[idx[split:]],  y[idx[split:]]
    print(f"[data] Train: {len(X_train)}, Val: {len(X_val)}")

    # ── 3. Inicializar modelo ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[model] Dispositivo: {device}")

    model = LiquidBitModel(
        embed_dim   = embed_dim,
        num_modes   = args.num_modes,
        num_spheres = len(group_names),
        num_classes = num_classes,
        temperature = 1.0,
    ).to(device)

    # Inicializar W_projection desde PCA
    emb_3d_path = os.path.join(SCRIPT_DIR, "embeddings_3d.npy")
    if os.path.exists(emb_3d_path):
        from sklearn.decomposition import PCA
        print("[model] Inicializando W_projection desde PCA...")
        pca = PCA(n_components=3)
        pca.fit(embeddings[:5000])
        W_pca = pca.components_.astype(np.float32)  # [3, D]
        with torch.no_grad():
            model.W_projection.copy_(torch.from_numpy(W_pca).to(device))

    # Inicializar sphere_centers desde centroides de grupos semánticos
    print("[model] Inicializando sphere_centers desde datos...")
    group_centroids = []
    for label_id in range(num_classes):
        mask = y_train == label_id
        if mask.sum() > 0:
            centroid = X_train[mask].mean(axis=0)
            # Proyectar a 3D con W_projection inicial
            with torch.no_grad():
                c_t = torch.from_numpy(centroid).to(device)
                pos3d = F.normalize(c_t @ model.W_projection.T, dim=0).cpu().numpy()
            group_centroids.append(pos3d)
        else:
            group_centroids.append(np.random.randn(3))
    group_centroids = np.array(group_centroids, dtype=np.float32)
    with torch.no_grad():
        model.fuzzy_bsh.sphere_centers.data.copy_(
            torch.from_numpy(group_centroids).to(device)
        )
    print(f"[model] Centros inicializados desde {num_classes} grupos semánticos")

    # ── 4. Optimizer y losses ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    spatial_loss_fn = SpatialLoss(SpatialLossConfig(alpha=args.alpha_spatial))
    task_loss_fn    = nn.CrossEntropyLoss()

    # ── 5. DataLoaders ───────────────────────────────────────────────────
    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train).to(device)
    X_vl_t = torch.from_numpy(X_val).to(device)
    y_vl_t = torch.from_numpy(y_val).to(device)
    # Omegas: uno por muestra, derivado de la etiqueta (distribución uniforme en [0, 2π])
    omega_train = torch.FloatTensor(len(X_train)).uniform_(0, 2 * np.pi).to(device)
    omega_val   = torch.FloatTensor(len(X_val)).uniform_(0, 2 * np.pi).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t, omega_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # ── 6. Training loop ─────────────────────────────────────────────────
    print(f"\n[train] epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
    print(f"[train] num_modes={args.num_modes}, alpha_spatial={args.alpha_spatial}\n")

    curve = []
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_l_task, total_l_spatial = 0.0, 0.0, 0.0
        n_batches = 0

        for batch_emb, batch_y, batch_omega in train_dl:
            optimizer.zero_grad()

            out = model(batch_emb, batch_omega)
            logits      = out["logits"]
            pos_3d      = out["pos_3d"]
            memberships = out["memberships"]

            # L_task
            l_task = task_loss_fn(logits, batch_y)

            # L_spatial
            poly_mask = (memberships.max(dim=1).values < 0.7)  # alta incertidumbre = polisémico
            sp_out    = spatial_loss_fn(
                pos_3d,
                model.get_sphere_centers(),
                model.sphere_radii,
                memberships,
                poly_mask,
            )
            l_sp = sp_out["total"]

            # L_total
            loss = l_task + args.alpha_spatial * l_sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss     += loss.item()
            total_l_task   += l_task.item()
            total_l_spatial += l_sp.item()
            n_batches += 1

        scheduler.step()
        model.anneal_temperature(factor=0.995)

        # ── Validación ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_out  = model(X_vl_t, omega_val)
            val_preds = val_out["logits"].argmax(dim=1)
            val_acc   = (val_preds == y_vl_t).float().mean().item()

        avg_loss = total_loss / max(n_batches, 1)
        avg_task = total_l_task / max(n_batches, 1)
        avg_sp   = total_l_spatial / max(n_batches, 1)
        temp     = model.temperature.item()

        curve.append({
            "epoch": epoch,
            "loss": avg_loss,
            "l_task": avg_task,
            "l_spatial": avg_sp,
            "val_acc": val_acc,
            "temperature": temp,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d} | loss={avg_loss:.4f} | l_task={avg_task:.4f} "
                  f"| l_sp={avg_sp:.4f} | val_acc={val_acc:.1%} | T={temp:.4f}")

        # ── Re-clustering OHBSC cada K epochs ────────────────────────────
        if args.recluster_every > 0 and epoch > 0 and epoch % args.recluster_every == 0:
            print(f"[recluster] Epoch {epoch}: re-ejecutando OHBSC...")
            with torch.no_grad():
                all_pos = model.project_to_3d(X_tr_t[:2000]).cpu().numpy()
            builder = OHBSCBuilder(branching=len(group_names), max_depth=2, min_size=5)
            root = builder.build(all_pos)
            # Actualizar centros desde árbol re-clusterizado
            new_centers = np.array([c.center for c in root.children[:len(group_names)]])
            if len(new_centers) == len(group_names):
                with torch.no_grad():
                    model.fuzzy_bsh.sphere_centers.data.copy_(
                        torch.from_numpy(new_centers.astype(np.float32)).to(device)
                    )
                print(f"[recluster] Centros actualizados: {new_centers.shape}")

    print(f"\n[done] Mejor val_acc: {best_val_acc:.1%}")

    # ── 7. Exportar modelo ───────────────────────────────────────────────
    print("\n[export] Guardando modelo...")

    sphere_params = model.sphere_params_numpy()
    fourier_params = model.fourier_params_numpy()
    w_proj = model.W_projection.detach().cpu().numpy()

    np.save(os.path.join(SCRIPT_DIR, "sphere_params.npy"), sphere_params)
    np.save(os.path.join(SCRIPT_DIR, "fourier_coeffs.npy"), fourier_params)
    np.save(os.path.join(SCRIPT_DIR, "w_projection.npy"), w_proj)

    with open(os.path.join(SCRIPT_DIR, "training_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)

    model_info = {
        "embed_dim":      embed_dim,
        "num_modes":      args.num_modes,
        "num_spheres":    len(group_names),
        "num_classes":    num_classes,
        "group_names":    group_names,
        "best_val_acc":   best_val_acc,
        "final_temp":     model.temperature.item(),
        "epochs_trained": args.epochs,
        "sphere_params":  "sphere_params.npy",
        "fourier_coeffs": "fourier_coeffs.npy",
        "w_projection":   "w_projection.npy",
    }
    with open(os.path.join(SCRIPT_DIR, "trained_model.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"[export] sphere_params.npy:  {sphere_params.shape}  (K×4: [cx,cy,cz,r])")
    print(f"[export] fourier_coeffs.npy: {fourier_params.shape} (K×2M: [a_k, b_k])")
    print(f"[export] w_projection.npy:   {w_proj.shape}         (3×D)")
    print(f"[export] trained_model.json: {num_spheres=} {num_classes=}")
    print(f"\n[OK] Training completado. Val accuracy: {best_val_acc:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LiquidBit v4.0 Training Pipeline")
    p.add_argument("--epochs",          type=int,   default=100,
                   help="Número de epochs de entrenamiento")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate inicial (AdamW)")
    p.add_argument("--batch-size",      type=int,   default=64,
                   help="Batch size")
    p.add_argument("--num-modes",       type=int,   default=8,
                   help="Número de modos Fourier por SemanticString")
    p.add_argument("--alpha-spatial",   type=float, default=0.1,
                   help="Peso de L_spatial en L_total")
    p.add_argument("--n-augments",      type=int,   default=20,
                   help="Aumentaciones por palabra")
    p.add_argument("--recluster-every", type=int,   default=50,
                   help="Re-ejecutar OHBSC cada N epochs (0=desactivado)")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Alias para la clase (evita NameError en forward)
    num_spheres = 8  # 8 grupos semánticos

    train(args)
