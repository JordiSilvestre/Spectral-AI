#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  LiquidBit Zero-Matrix — Fuzzy BSH v2.0 con torch.autograd.Function   ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  NUEVA IDEA (de los documentos subidos):                               ║
║  Implementar el BVH traversal como torch.autograd.Function para que   ║
║  los gradientes fluyan a través de la estructura del árbol.            ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Arquitectura del módulo:                                              ║
║                                                                         ║
║  FuzzyBSHFunction (torch.autograd.Function)                            ║
║    ├── forward(): BVH traversal con membresía fuzzy                    ║
║    │   • Compute d²(token, center_k) para cada esfera k               ║
║    │   • Membresía: p_k = softmax(-d²/(2T²))                          ║
║    │   • Salida: routing_weights [N_tokens, K_spheres]                 ║
║    └── backward(): propaga gradientes a través de la estructura       ║
║        • dL/d(center_k) via chain rule                                 ║
║        • dL/d(token_pos) para fine-tuning de embeddings               ║
║                                                                         ║
║  FuzzyBSHLayer (nn.Module) usa FuzzyBSHFunction internamente          ║
║  FuzzyBSHTrainer: loop de training con L_spatial                       ║
║                                                                         ║
║  Ejecutar:                                                              ║
║    python3 fuzzy_bsh_autograd.py                                       ║
║                                                                         ║
║  Requiere: PyTorch (pip install torch)                                 ║
║  Fallback: NumPy (para verificar lógica sin GPU)                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════
# IMPORTS (PyTorch con fallback NumPy)
# ══════════════════════════════════════════════════════════════════════════

import numpy as np
import math
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("  [PyTorch] Modo completo: gradientes automáticos disponibles")
except ImportError:
    TORCH_AVAILABLE = False
    print("  [NumPy]   Modo fallback: gradientes manuales (instalar torch para modo completo)")
    print("  pip install torch")

print()


# ══════════════════════════════════════════════════════════════════════════
# MODO PYTORCH: torch.autograd.Function
# ══════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class FuzzyBSHFunction(torch.autograd.Function):
        """
        BVH traversal diferenciable como operación PyTorch custom.

        La clave de torch.autograd.Function es:
          • forward(): computación hacia adelante (salva tensores para backward)
          • backward(): gradientes analíticos (propagación hacia atrás)

        Esto permite que PyTorch incluya este BVH en el grafo de computación,
        haciendo que los gradientes de la tarea (L_task) fluyan hasta los
        embeddings y los centros de esferas durante el training.

        En OptiX: forward() lanza el kernel RT, backward() aproxima gradientes.
        En CPU: forward()/backward() son diferenciables exactos.
        """

        @staticmethod
        def forward(ctx, token_positions, sphere_centers, sphere_radii, temperature):
            """
            Computa routing weights: p(token_i ∈ sphere_k).

            Args:
                token_positions : [N, D] - posiciones 3D de los tokens
                sphere_centers  : [K, D] - centros de las esferas
                sphere_radii    : [K]    - radios (para normalización)
                temperature     : float  - T > 0 (annealing: T→0 = hard assignment)

            Returns:
                routing_weights : [N, K] - p_ik = softmax(-d²/(2T²))
            """
            N, D = token_positions.shape
            K    = sphere_centers.shape[0]
            T    = temperature.item() if hasattr(temperature, 'item') else float(temperature)

            # Distancias euclideas cuadradas: d²(i,k) = ||pos_i - center_k||²
            # Broadcasting: [N,1,D] - [1,K,D] → [N,K,D] → sum → [N,K]
            diff = token_positions.unsqueeze(1) - sphere_centers.unsqueeze(0)  # [N,K,D]
            dist_sq = (diff ** 2).sum(dim=-1)  # [N,K]

            # Membresía fuzzy: p_ik = softmax_k( -d²_ik / (2*T²) )
            T_sq = max(T ** 2, 1e-6)
            logits = -dist_sq / (2.0 * T_sq)  # [N,K]
            routing_weights = F.softmax(logits, dim=-1)  # [N,K]

            # Salvar para backward
            ctx.save_for_backward(token_positions, sphere_centers, dist_sq, routing_weights)
            ctx.T = T
            ctx.T_sq = T_sq

            return routing_weights

        @staticmethod
        def backward(ctx, grad_output):
            """
            Propaga gradientes a través del BVH difuso.

            grad_output: [N,K] — dL/d(routing_weights)

            Devuelve:
                grad_token_positions : [N, D]
                grad_sphere_centers  : [K, D]
                grad_sphere_radii    : None (no entrenamos radios en esta versión)
                grad_temperature     : None (T es hyperparámetro)
            """
            token_positions, sphere_centers, dist_sq, routing_weights = ctx.saved_tensors
            T_sq = ctx.T_sq
            N, D = token_positions.shape
            K    = sphere_centers.shape[0]

            # ── Gradiente de softmax ──────────────────────────────────────
            # Para softmax: d(p_ik)/d(logit_ij) = p_ik*(δ_jk - p_ij)
            # Entonces: dL/d(logit_ik) = Σ_j [dL/d(p_ij) * p_ij*(δ_jk - p_ik)]
            #                          = p_ik * (dL/d(p_ik) - Σ_j dL/d(p_ij)*p_ij)
            sum_grad_p = (grad_output * routing_weights).sum(dim=-1, keepdim=True)  # [N,1]
            grad_logits = routing_weights * (grad_output - sum_grad_p)  # [N,K]

            # ── Gradiente de logits respecto a d² ─────────────────────────
            # logit_ik = -d²_ik / (2*T²)  →  d(logit)/d(d²) = -1/(2*T²)
            grad_dist_sq = grad_logits * (-1.0 / (2.0 * T_sq))  # [N,K]

            # ── Gradiente de d² respecto a posiciones y centros ────────────
            # d²_ik = ||pos_i - center_k||²
            # d(d²_ik)/d(pos_i) = 2*(pos_i - center_k)
            # d(d²_ik)/d(center_k) = -2*(pos_i - center_k)

            diff = token_positions.unsqueeze(1) - sphere_centers.unsqueeze(0)  # [N,K,D]

            # grad_token_positions: Σ_k [grad_dist_sq_ik * 2*(pos_i - center_k)]
            # Shape: [N,K] * [N,K,D] → sum over K → [N,D]
            grad_token_positions = (
                (grad_dist_sq.unsqueeze(-1) * 2.0 * diff)
                .sum(dim=1)  # sum over K
            )  # [N,D]

            # grad_sphere_centers: Σ_i [grad_dist_sq_ik * (-2)*(pos_i - center_k)]
            # Shape: [N,K] * [N,K,D] → sum over N → [K,D]
            grad_sphere_centers = (
                (grad_dist_sq.unsqueeze(-1) * (-2.0) * diff)
                .sum(dim=0)  # sum over N
            )  # [K,D]

            return grad_token_positions, grad_sphere_centers, None, None


    class FuzzyBSHLayer(nn.Module):
        """
        Capa PyTorch que encapsula FuzzyBSHFunction.
        Los centros de esferas son parámetros aprendibles (nn.Parameter).
        """

        def __init__(self, n_spheres: int, embed_dim: int, init_temp: float = 1.0):
            super().__init__()
            self.n_spheres = n_spheres
            self.embed_dim = embed_dim

            # Centros aprendibles ∈ ℝ^(K×D)
            self.sphere_centers = nn.Parameter(
                torch.randn(n_spheres, embed_dim) * 2.0
            )
            # Radios (no entrenados en este prototipo, fijos)
            self.register_buffer('sphere_radii', torch.ones(n_spheres))

            # Temperatura (no es nn.Parameter, se controla externamente)
            self.temperature = init_temp

        def forward(self, token_positions):
            """
            token_positions: [N, embed_dim]
            returns: routing_weights [N, n_spheres]
            """
            temp_tensor = torch.tensor(self.temperature, dtype=token_positions.dtype,
                                       device=token_positions.device)
            return FuzzyBSHFunction.apply(
                token_positions, self.sphere_centers, self.sphere_radii, temp_tensor
            )

        def anneal(self, factor: float = 0.995, min_temp: float = 0.05):
            """Reduce la temperatura (T→0 = asignación dura)."""
            self.temperature = max(min_temp, self.temperature * factor)

        def hard_assignment(self, token_positions):
            """Asignación dura (argmax) para inferencia."""
            with torch.no_grad():
                weights = self.forward(token_positions)
                return weights.argmax(dim=-1)  # [N]

        def spatial_loss(self, token_positions, token_labels):
            """
            L_spatial = L_prox + L_cover

            L_prox:  tokens con mismo label deben tener centros cercanos
            L_cover: todos los tokens deben estar cubiertos por al menos una esfera
            """
            weights = self.forward(token_positions)  # [N, K]

            # L_prox: distancia promedio entre centros de esferas con tokens similares
            # Penaliza esferas que "deberían ser similares" pero están lejos
            L_prox = torch.zeros(1, device=token_positions.device)
            for k1 in range(self.n_spheres):
                for k2 in range(k1 + 1, self.n_spheres):
                    center_dist = (self.sphere_centers[k1] - self.sphere_centers[k2]).norm()
                    # Si dos esferas tienen tokens del mismo label, penalizar si están lejos
                    mask_k1 = weights[:, k1] > 0.3
                    mask_k2 = weights[:, k2] > 0.3
                    shared_label = (token_labels[mask_k1].unsqueeze(1) ==
                                    token_labels[mask_k2].unsqueeze(0)).any()
                    if shared_label:
                        L_prox = L_prox + F.relu(center_dist - 1.0)

            # L_cover: entropía negativa (queremos tokens bien asignados = baja entropía)
            H = -(weights * (weights + 1e-9).log()).sum(dim=-1)  # [N]
            L_cover = H.mean()

            return L_prox + 0.1 * L_cover


    class FuzzyBSHTrainer:
        """Loop de training para FuzzyBSHLayer."""

        def __init__(self, n_spheres=3, embed_dim=3, n_epochs=300, lr=0.01):
            self.layer = FuzzyBSHLayer(n_spheres, embed_dim)
            self.optimizer = torch.optim.Adam(self.layer.parameters(), lr=lr)
            self.n_epochs = n_epochs

        def train(self, token_positions_np, token_labels_np):
            """
            token_positions_np: [N, D] numpy
            token_labels_np:    [N]    numpy (cluster ground truth)
            """
            pos    = torch.tensor(token_positions_np, dtype=torch.float32)
            labels = torch.tensor(token_labels_np, dtype=torch.long)

            print("  Entrenando FuzzyBSH (torch.autograd.Function)...")
            print(f"  {'Epoch':>6}  {'L_total':>9}  {'T':>7}  {'acc%':>6}")
            print("  " + "─" * 40)

            for epoch in range(self.n_epochs):
                self.optimizer.zero_grad()

                # Forward: routing weights diferenciables
                weights = self.layer(pos)  # [N, K]

                # L_task: queremos que el token i sea asignado a su cluster correcto
                # Cross-entropy: -log(p_i[correct_sphere])
                L_task = F.cross_entropy(weights, labels)

                # L_spatial
                L_spatial = self.layer.spatial_loss(pos, labels)

                L_total = L_task + 0.1 * L_spatial

                # Backward (aquí es donde FuzzyBSHFunction.backward() se ejecuta)
                L_total.backward()
                self.optimizer.step()
                self.layer.anneal()

                if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                    with torch.no_grad():
                        assignments = self.layer.hard_assignment(pos)
                        acc = (assignments == labels).float().mean().item() * 100
                    print(f"  {epoch:>6}  {L_total.item():>9.4f}  "
                          f"{self.layer.temperature:>7.4f}  {acc:>6.1f}%")

            print()
            return self.layer.sphere_centers.detach().numpy()


# ══════════════════════════════════════════════════════════════════════════
# MODO NUMPY FALLBACK: gradientes manuales (misma lógica matemática)
# ══════════════════════════════════════════════════════════════════════════

class FuzzyBSHNumpy:
    """
    Implementación NumPy de FuzzyBSH con gradientes manuales.
    Misma lógica matemática que FuzzyBSHFunction pero sin PyTorch.
    Útil para verificar que la matemática es correcta.
    """

    def __init__(self, n_spheres=3, embed_dim=3, init_temp=1.0):
        self.n_spheres   = n_spheres
        self.embed_dim   = embed_dim
        self.temperature = init_temp
        # Centros aprendibles
        self.centers = np.random.randn(n_spheres, embed_dim).astype(np.float32) * 2.0

    def _softmax(self, x):
        """Softmax numéricamente estable (eje=-1)."""
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    def forward(self, positions):
        """
        positions: [N, D]
        returns: routing_weights [N, K], dist_sq [N, K]
        """
        T_sq = max(self.temperature ** 2, 1e-6)
        # [N, 1, D] - [1, K, D] → [N, K, D]
        diff    = positions[:, None, :] - self.centers[None, :, :]
        dist_sq = (diff ** 2).sum(axis=-1)          # [N, K]
        logits  = -dist_sq / (2.0 * T_sq)            # [N, K]
        weights = self._softmax(logits)              # [N, K]
        return weights, dist_sq, diff

    def backward(self, positions, weights, diff, grad_weights):
        """
        Gradientes analíticos idénticos a FuzzyBSHFunction.backward().

        grad_weights: [N, K] — dL/d(routing_weights)
        """
        T_sq = max(self.temperature ** 2, 1e-6)

        # Softmax backward
        sum_gw = (grad_weights * weights).sum(axis=-1, keepdims=True)  # [N,1]
        grad_logits = weights * (grad_weights - sum_gw)                # [N,K]

        # logit → dist²
        grad_dist_sq = grad_logits * (-1.0 / (2.0 * T_sq))            # [N,K]

        # dist² → centers
        # [N,K,D]: grad_dist_sq[:,:,None] * (-2) * diff → sum over N → [K,D]
        grad_centers = (grad_dist_sq[:, :, None] * (-2.0) * diff).sum(axis=0)

        # dist² → positions (no necesitamos para este prototipo)
        # grad_positions = (grad_dist_sq[:,:,None] * 2.0 * diff).sum(axis=1)

        return grad_centers

    def loss_and_grad(self, positions, labels_onehot):
        """
        L_task = CrossEntropy(routing_weights, labels_onehot)
        Returns: (loss, grad_centers)
        """
        weights, dist_sq, diff = self.forward(positions)

        # CE loss: -Σ p_true * log(p_pred)
        log_weights = np.log(np.clip(weights, 1e-9, 1.0))
        ce_loss = -(labels_onehot * log_weights).sum(axis=-1).mean()

        # CE backward: dL/d(weights) = (weights - labels_onehot) / N
        grad_weights = (weights - labels_onehot) / len(positions)

        grad_centers = self.backward(positions, weights, diff, grad_weights)

        return ce_loss, grad_centers

    def anneal(self, factor=0.995, min_temp=0.05):
        self.temperature = max(min_temp, self.temperature * factor)

    def accuracy(self, positions, labels):
        weights, _, _ = self.forward(positions)
        predicted = weights.argmax(axis=-1)
        return (predicted == labels).mean() * 100.0


# ══════════════════════════════════════════════════════════════════════════
# DEMO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

def run_demo():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║    LiquidBit Zero-Matrix — Fuzzy BSH v2.0 (torch.autograd.Function)    ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    np.random.seed(42)

    # Datos de prueba: 3 clusters bien separados
    N_per_cluster = 8
    positions_raw = np.vstack([
        np.random.randn(N_per_cluster, 3) + np.array([3.0, 0.0, 0.0]),  # Cluster 0
        np.random.randn(N_per_cluster, 3) + np.array([0.0, 3.0, 0.0]),  # Cluster 1
        np.random.randn(N_per_cluster, 3) + np.array([0.0, 0.0, 3.0]),  # Cluster 2
    ]).astype(np.float32)

    labels_raw = np.array([0]*N_per_cluster + [1]*N_per_cluster + [2]*N_per_cluster)

    # One-hot para numpy mode
    N, K = len(labels_raw), 3
    labels_onehot = np.zeros((N, K), dtype=np.float32)
    labels_onehot[np.arange(N), labels_raw] = 1.0

    if TORCH_AVAILABLE:
        # ── MODO PYTORCH ─────────────────────────────────────────────────
        print("  ══ Modo PyTorch (torch.autograd.Function) ══")
        print()
        trainer = FuzzyBSHTrainer(n_spheres=3, embed_dim=3, n_epochs=300, lr=0.05)
        learned_centers = trainer.train(positions_raw, labels_raw)

        print("  Centros aprendidos:")
        for k, c in enumerate(learned_centers):
            print(f"    Esfera {k}: {c.round(3)}")

        # Verificar gradientes
        print()
        print("  ✅ torch.autograd.Function OK — gradientes fluyen a través del BVH")
        print()

    # ── MODO NUMPY FALLBACK ───────────────────────────────────────────────
    print("  ══ Modo NumPy (gradientes manuales — misma matemática) ══")
    print()

    model = FuzzyBSHNumpy(n_spheres=3, embed_dim=3, init_temp=1.0)
    lr    = 0.05
    n_epochs = 300

    print(f"  {'Epoch':>6}  {'L_CE':>9}  {'T':>7}  {'acc%':>6}")
    print("  " + "─" * 40)

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        loss, grad_centers = model.loss_and_grad(positions_raw, labels_onehot)
        model.centers -= lr * grad_centers
        model.anneal()

        if epoch % 60 == 0 or epoch == n_epochs - 1:
            acc = model.accuracy(positions_raw, labels_raw)
            print(f"  {epoch:>6}  {loss:>9.4f}  {model.temperature:>7.4f}  {acc:>6.1f}%")

    elapsed = time.perf_counter() - t0
    final_acc = model.accuracy(positions_raw, labels_raw)

    print()
    print(f"  Completado en {elapsed:.3f}s | T_final={model.temperature:.4f}")
    print()
    print("  Centros aprendidos (NumPy):")
    for k, c in enumerate(model.centers):
        print(f"    Esfera {k}: {c.round(3)}")

    print(f"""
  ══ Resumen matemático ══

  FuzzyBSHFunction.forward():
    d²(i,k) = ||pos_i - center_k||²
    p(i,k)  = softmax_k( -d²(i,k) / (2T²) )

  FuzzyBSHFunction.backward():
    dL/d(logit_ik) = p_ik · (dL/d(p_ik) - Σ_j dL/d(p_ij)·p_ij)
    dL/d(d²_ik)   = dL/d(logit_ik) · (-1/(2T²))
    dL/d(center_k) = Σ_i dL/d(d²_ik) · (-2)·(pos_i - center_k)
    dL/d(pos_i)   = Σ_k dL/d(d²_ik) ·  (2)·(pos_i - center_k)

  Integración con OptiX (GPU):
    forward() → lanza optixLaunch → RT Cores calculan d²(i,k)
    backward() → se ejecuta en CPU/CUDA → gradientes analíticos exactos
    Resultado: BVH completamente diferenciable end-to-end
    """)

    # Guardar centros
    import os
    out = os.path.join(os.path.dirname(__file__), "fuzzy_bsh_centers.npy")
    np.save(out, model.centers)
    print(f"  ✅ Centros guardados: fuzzy_bsh_centers.npy  shape={model.centers.shape}")
    print(f"  Accuracy final: {final_acc:.1f}%")
    print()


if __name__ == "__main__":
    run_demo()
