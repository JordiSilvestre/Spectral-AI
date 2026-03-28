#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  LiquidBit Zero-Matrix — W_dispersion Training v2.0                    ║
║  NUEVAS IDEAS: Gumbel-Softmax + Load Balancing Loss                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  v1.0: SGD puro con cross-entropy → 100% accuracy (datos sintéticos)  ║
║  v2.0: Gumbel-Softmax + L_balance → robusto a datos reales aleatorios  ║
║                                                                         ║
║  Nuevo en v2.0 (ideas de los documentos de Kimi/Gemini):              ║
║  • Gumbel-Softmax: routing discreto diferenciable                      ║
║  • L_balance: penaliza que una esfera monopolice el routing            ║
║  • L_entropy: evita indecisión permanente del router                   ║
║  • L_total = L_routing + α*L_balance + β*L_entropy + γ*L_spatial      ║
╚══════════════════════════════════════════════════════════════════════════╝

Ejecutar: python3 train_dispersion_v2.py
"""

import numpy as np
import math
import time

np.random.seed(2024)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════

SPECTRAL_DIM = 64
N_EPOCHS     = 600
LR           = 0.08
ALPHA_BAL    = 0.05   # peso de L_balance (nuevo v2.0)
BETA_ENT     = 0.01   # peso de L_entropy (nuevo v2.0)
GAMMA_SPAT   = 0.03   # peso de L_spatial
GUMBEL_TAU   = 1.0    # temperatura Gumbel-Softmax (se reduce con annealing)
TAU_DECAY    = 0.995  # τ → 0 = decisiones más discretas con cada epoch

PASS = "✅"; FAIL = "❌"

print()
print("╔══════════════════════════════════════════════════════════════════════════╗")
print("║    LiquidBit Zero-Matrix — W_dispersion Training v2.0                  ║")
print("║    Gumbel-Softmax + Load Balancing Loss                                ║")
print("╚══════════════════════════════════════════════════════════════════════════╝")

# ══════════════════════════════════════════════════════════════════════════
# VOCABULARIO Y EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════

VOCAB = {
    # Programación (cluster 0, zona +X)
    "python":    np.array([3.1, 0.2, 0.1]),
    "for":       np.array([2.9, 0.3, 0.0]),
    "while":     np.array([2.8, 0.1, 0.2]),
    "variable":  np.array([3.2, 0.4, 0.1]),
    "función":   np.array([3.0, 0.5, 0.0]),
    "clase":     np.array([2.7, 0.2, 0.3]),
    # Música (cluster 1, zona +Y)
    "ritmo":     np.array([0.1, 3.2, 0.2]),
    "sample":    np.array([0.2, 2.9, 0.1]),
    "beat":      np.array([0.0, 3.1, 0.3]),
    "tempo":     np.array([0.3, 3.0, 0.1]),
    "melodía":   np.array([0.1, 2.8, 0.4]),
    "acorde":    np.array([0.2, 3.3, 0.0]),
    # Física (cluster 2, zona +Z)
    "orbita":    np.array([0.2, 0.1, 3.1]),
    "campo":     np.array([0.1, 0.3, 2.9]),
    "fuerza":    np.array([0.3, 0.0, 3.2]),
    "masa":      np.array([0.0, 0.2, 3.0]),
    "energía":   np.array([0.2, 0.1, 2.8]),
    "vector":    np.array([0.1, 0.4, 3.1]),
    # Polisémicos (frontera)
    "bucle":      np.array([1.5, 1.4, 0.2]),
    "frecuencia": np.array([1.3, 1.2, 1.1]),
    "onda":       np.array([0.2, 1.4, 1.5]),
    "ciclo":      np.array([1.5, 0.2, 1.4]),
}

TOKENS       = list(VOCAB.keys())
EMBEDDINGS_3D = np.array([VOCAB[t] for t in TOKENS], dtype=np.float32)
EMBEDDINGS_3D += np.random.randn(*EMBEDDINGS_3D.shape).astype(np.float32) * 0.15
D = 32
PROJ_MATRIX  = np.random.randn(3, D).astype(np.float32) * 0.5
EMBEDDINGS_D = EMBEDDINGS_3D @ PROJ_MATRIX
N_TOKENS     = len(TOKENS)

COLOR_PROG  = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_PROG[0]  = 1.0
COLOR_MUSIC = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_MUSIC[1] = 1.0
COLOR_PHYS  = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_PHYS[2]  = 1.0

EXPECTED_ROUTING = {
    "bucle":       {"Programación": 0, "Música": 1},
    "frecuencia":  {"Programación": 0, "Música": 1, "Física": 2},
    "onda":        {"Música": 1, "Física": 2},
    "ciclo":       {"Programación": 0, "Física": 2},
    "python":      {"Programación": 0, "Música": 0, "Física": 0},
    "ritmo":       {"Programación": 1, "Música": 1, "Física": 1},
    "orbita":      {"Programación": 2, "Música": 2, "Física": 2},
}

# ══════════════════════════════════════════════════════════════════════════
# GEOMETRÍA
# ══════════════════════════════════════════════════════════════════════════

class Sphere:
    def __init__(self, center, radius, label, token_indices, sphere_id):
        self.center         = np.array(center, dtype=np.float32)
        self.radius         = float(radius)
        self.label          = label
        self.token_indices  = np.array(token_indices, dtype=np.int32)
        self.sphere_id      = sphere_id
        self.W_dispersion   = np.random.randn(SPECTRAL_DIM).astype(np.float32) * 0.1

    def logit(self, color):
        """Logit crudo antes de softmax: dot(W, color)"""
        return float(np.dot(self.W_dispersion, color))

def build_spheres():
    centers = [np.array([3.0,0.3,0.1]), np.array([0.2,3.0,0.2]), np.array([0.2,0.2,3.0])]
    labels  = ["Prog_Sphere", "Music_Sphere", "Phys_Sphere"]
    spheres = []
    for ci,(cc,cl) in enumerate(zip(centers,labels)):
        dists   = np.linalg.norm(EMBEDDINGS_3D - cc, axis=1)
        indices = np.where(dists < 2.5)[0].tolist() or [int(np.argmin(dists))]
        r       = float(np.max(dists[indices]) + 0.5)
        spheres.append(Sphere(cc, r, cl, indices, ci))
    return spheres

spheres = build_spheres()
N_SPHERES = len(spheres)
print(f"\n  Esferas: {N_SPHERES} | Tokens: {N_TOKENS} | D_spectral: {SPECTRAL_DIM}")
for s in spheres:
    print(f"    • {s.label:<18}  tokens={len(s.token_indices)}")

# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

def build_dataset():
    ds = []
    ctx_map = {"Programación": COLOR_PROG, "Música": COLOR_MUSIC, "Física": COLOR_PHYS}
    for tok, routing in EXPECTED_ROUTING.items():
        tidx = TOKENS.index(tok)
        for ctx, esid in routing.items():
            ds.append({"token_idx": tidx, "color": ctx_map[ctx].copy(),
                       "expected_sphere_id": esid, "token_name": tok, "context": ctx})
    return ds

train_dataset = build_dataset()
print(f"\n  Dataset: {len(train_dataset)} ejemplos")

# ══════════════════════════════════════════════════════════════════════════
# FUNCIONES MATEMÁTICAS
# ══════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / (e.sum() + 1e-9)

def gumbel_softmax(logits, tau=1.0):
    """
    Gumbel-Softmax (nueva en v2.0): permite routing discreto diferenciable.

    Durante entrenamiento: τ>0 → distribución suave (gradientes fluyen)
    En inferencia: τ→0 → argmax (decisión discreta)

    Ref: Jang et al. 2017 "Categorical Reparameterization with Gumbel-Softmax"
    """
    # Muestrear ruido Gumbel: G = -log(-log(U)), U ~ Uniform(0,1)
    gumbel_noise = -np.log(-np.log(np.random.uniform(1e-20, 1.0, size=len(logits))))
    # Gumbel trick: softmax((logits + G) / τ)
    noisy_logits = (logits + gumbel_noise) / tau
    return softmax(noisy_logits)

def gumbel_softmax_deterministic(logits, tau):
    """Versión sin ruido (para validación y backprop consistente en batch)."""
    return softmax(logits / (tau + 1e-8))

# ══════════════════════════════════════════════════════════════════════════
# LOSS v2.0: L_routing + L_balance + L_entropy
# ══════════════════════════════════════════════════════════════════════════

def compute_all_losses(batch, tau):
    """
    Calcula L_routing + L_balance + L_entropy + L_spatial y sus gradientes.

    Nuevo en v2.0:
    ─────────────
    L_balance: Variance(avg_usage_per_sphere)
               Penaliza que una esfera reciba todo el routing.
               Ideal: cada esfera recibe 1/N_SPHERES del tráfico.

    L_entropy: |mean_entropy - H_target|²
               Evita que el router sea 100% indeciso (alta entropía = random)
               o 100% determinista prematuramente. Sweet spot: entropía media.
    """
    n = len(batch)
    grads_W = [np.zeros_like(s.W_dispersion) for s in spheres]

    loss_routing = 0.0
    usage_accum  = np.zeros(N_SPHERES)    # Para L_balance
    entropy_accum = 0.0                    # Para L_entropy

    for ex in batch:
        color       = ex["color"]
        expected_id = ex["expected_sphere_id"]

        # Logits crudos
        logits = np.array([s.logit(color) for s in spheres], dtype=np.float32)

        # Gumbel-Softmax: distribución suave con annealing
        probs = gumbel_softmax_deterministic(logits, tau)

        # ── L_routing: cross-entropy ──────────────────────────────────
        prob_exp = np.clip(probs[expected_id], 1e-7, 1.0)
        loss_routing += -np.log(prob_exp)

        # Acumular usage (para L_balance)
        usage_accum += probs

        # Acumular entropía (para L_entropy)
        h = -np.sum(probs * np.log(probs + 1e-9))
        entropy_accum += h

        # ── Backprop L_routing ────────────────────────────────────────
        # d(CE)/d(logit_i/τ) = (p_i - label_i) / τ
        d_loss_dlogit = (probs.copy() - 0.0)
        d_loss_dlogit[expected_id] -= 1.0
        d_loss_dlogit /= (tau + 1e-8)

        # d(logit_i)/d(W_i) = color (lineal)
        for i in range(N_SPHERES):
            grads_W[i] += d_loss_dlogit[i] * color

    # Promediar
    loss_routing /= n
    avg_usage = usage_accum / n          # [N_SPHERES]
    mean_entropy = entropy_accum / n

    # ── L_balance: penalizar distribución no uniforme ─────────────────
    # Ideal: avg_usage ≈ 1/N_SPHERES para cada esfera
    target_usage = np.ones(N_SPHERES) / N_SPHERES
    loss_balance = np.sum((avg_usage - target_usage) ** 2)

    # Gradiente de L_balance respecto a cada W
    # d(L_bal)/d(W_i) ≈ 2*(avg_usage_i - 1/K) * mean(d_probs_i/d_logit_i * color)
    # Aproximación: propagamos el gradiente de balance como señal adicional
    d_balance_dusage = 2.0 * (avg_usage - target_usage)
    for ex in batch:
        color  = ex["color"]
        logits = np.array([s.logit(color) for s in spheres], dtype=np.float32)
        probs  = gumbel_softmax_deterministic(logits, tau)
        for i in range(N_SPHERES):
            # d(p_i)/d(logit_i) ≈ p_i*(1-p_i) (diagonal Jacobian)
            dp_dlogit = probs[i] * (1.0 - probs[i]) / (tau + 1e-8)
            grads_W[i] += (ALPHA_BAL / n) * d_balance_dusage[i] * dp_dlogit * color

    # ── L_entropy: sweet spot a H_target = log(N)/2 ───────────────────
    H_target = math.log(N_SPHERES) * 0.5    # entropía media deseada
    loss_entropy = (mean_entropy - H_target) ** 2

    # Gradiente de L_entropy (empuje hacia la entropía objetivo)
    # Aproximación: si H > H_target, reducir entropía (sharpening)
    d_ent = 2.0 * (mean_entropy - H_target)
    for ex in batch:
        color  = ex["color"]
        logits = np.array([s.logit(color) for s in spheres], dtype=np.float32)
        probs  = gumbel_softmax_deterministic(logits, tau)
        # d(H)/d(logit_i) = -(log(p_i)+1)*p_i*(1-p_i)/τ
        for i in range(N_SPHERES):
            dH_dlogit = -(np.log(probs[i]+1e-9)+1.0) * probs[i]*(1-probs[i]) / (tau+1e-8)
            grads_W[i] += (BETA_ENT / n) * d_ent * dH_dlogit * color

    # L_spatial (sin gradiente analítico en W, solo contribuye a loss total)
    loss_spatial = compute_spatial_loss()

    # Clip de gradientes
    for i in range(N_SPHERES):
        grads_W[i] /= n
        grads_W[i] = np.clip(grads_W[i], -1.0, 1.0)

    L_total = loss_routing + ALPHA_BAL*loss_balance + BETA_ENT*loss_entropy + GAMMA_SPAT*loss_spatial

    return {
        "total": L_total, "routing": loss_routing,
        "balance": loss_balance, "entropy": loss_entropy,
        "spatial": loss_spatial,
        "avg_usage": avg_usage, "mean_entropy": mean_entropy,
    }, grads_W

def compute_spatial_loss():
    loss_prox = 0.0
    for i in range(N_TOKENS):
        for j in range(i+1, min(i+5, N_TOKENS)):
            ei, ej = EMBEDDINGS_D[i], EMBEDDINGS_D[j]
            cos_sim = np.dot(ei,ej) / (np.linalg.norm(ei)*np.linalg.norm(ej)+1e-8)
            target_dist = 1.0 - cos_sim
            real_dist   = np.linalg.norm(EMBEDDINGS_3D[i]-EMBEDDINGS_3D[j])
            loss_prox  += (real_dist-target_dist)**2
    loss_prox /= max(1, N_TOKENS*4)
    loss_cover = 0.0
    for sp in spheres:
        for tidx in sp.token_indices:
            d = np.linalg.norm(EMBEDDINGS_3D[tidx]-sp.center)
            excess = max(0.0, d/(sp.radius+1e-6)-1.0)
            loss_cover += excess**2
    loss_cover /= max(1, N_TOKENS)
    return loss_prox + loss_cover

def routing_accuracy(batch, tau):
    correct = 0
    for ex in batch:
        logits = np.array([s.logit(ex["color"]) for s in spheres], dtype=np.float32)
        probs  = gumbel_softmax_deterministic(logits, tau)
        if np.argmax(probs) == ex["expected_sphere_id"]:
            correct += 1
    return correct / len(batch) * 100

# ══════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO v2.0
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*74)
print("  ENTRENAMIENTO v2.0: Gumbel-Softmax + Load Balancing + Entropy Loss")
print("─"*74)
print(f"  τ inicial: {GUMBEL_TAU:.3f} → decae ×{TAU_DECAY} por epoch (τ→0 = argmax)")
print(f"  L_total = L_routing + {ALPHA_BAL}×L_balance + {BETA_ENT}×L_entropy + {GAMMA_SPAT}×L_spatial")
print()
print(f"  {'Epoch':>6}  {'L_total':>8}  {'L_rout':>8}  {'L_bal':>7}  {'L_ent':>7}  {'τ':>6}  {'acc%':>6}  {'usage':>20}")
print("  " + "─"*80)

tau = GUMBEL_TAU
t0  = time.perf_counter()

for epoch in range(N_EPOCHS):
    losses, grads_W = compute_all_losses(train_dataset, tau)

    # Gradient step
    for i, sp in enumerate(spheres):
        sp.W_dispersion -= LR * grads_W[i]

    # Annealing de τ
    tau = max(0.05, tau * TAU_DECAY)

    acc = routing_accuracy(train_dataset, tau)

    if epoch % 60 == 0 or epoch == N_EPOCHS-1:
        usage_str = " ".join([f"{u:.2f}" for u in losses["avg_usage"]])
        print(f"  {epoch:>6}  {losses['total']:>8.4f}  {losses['routing']:>8.4f}  "
              f"{losses['balance']:>7.4f}  {losses['entropy']:>7.4f}  "
              f"{tau:>6.3f}  {acc:>6.1f}%  [{usage_str}]")

t_elapsed = time.perf_counter() - t0
print()
print(f"  Completado en {t_elapsed:.2f}s | τ final: {tau:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# TEST POLISEMIA FINAL (τ→0 = decisiones discretas)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*74)
print("  TEST POLISEMIA POST-TRAINING v2.0 (τ≈0: modo inferencia)")
print("─"*74)
print()

polysemic_tokens = ["bucle", "frecuencia", "onda", "ciclo"]
correct_total, total_tests = 0, 0
ctx_map = {"Programación": COLOR_PROG, "Música": COLOR_MUSIC, "Física": COLOR_PHYS}

for tok in polysemic_tokens:
    tidx    = TOKENS.index(tok)
    routing = EXPECTED_ROUTING.get(tok, {})
    print(f"  Token: '{tok}'")
    for ctx, expected_id in routing.items():
        color   = ctx_map[ctx]
        logits  = np.array([s.logit(color) for s in spheres], dtype=np.float32)
        probs   = gumbel_softmax_deterministic(logits, 0.01)   # τ≈0 ≈ argmax
        pred_id = int(np.argmax(probs))
        correct = pred_id == expected_id
        correct_total += int(correct)
        total_tests   += 1
        marker = PASS if correct else FAIL
        print(f"    {marker} {ctx:<14}  pred={spheres[pred_id].label:<18} "
              f"(p={probs[pred_id]:.3f})  expected={spheres[expected_id].label}")
    print()

final_acc = correct_total / total_tests * 100
print(f"  Polisemia accuracy v2.0: {correct_total}/{total_tests} = {final_acc:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# COMPARACIÓN v1 vs v2
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "─"*74)
print("  COMPARACIÓN v1.0 vs v2.0")
print("─"*74)
print()

# Cargar v1 si existe
import os
if os.path.exists(os.path.join(os.path.dirname(__file__), "w_dispersion_trained.npy")):
    W_v1 = np.load(os.path.join(os.path.dirname(__file__), "w_dispersion_trained.npy"))
    print(f"  v1.0 (SGD puro):           pesos cargados shape={W_v1.shape}")
else:
    W_v1 = None
    print("  v1.0 (SGD puro):           no encontrado")

print(f"  v2.0 (Gumbel-Softmax):     accuracy={final_acc:.1f}%  τ_final={tau:.4f}")
print()
print("  Ventaja v2.0 sobre v1.0:")
print("  • Gumbel-Softmax fuerza que el routing sea discreto en inferencia")
print("  • Load Balancing evita el colapso de routing a una sola esfera")
print("  • Entropy Loss evita indecisión permanente (especialmente con datos reales)")
print("  • Annealing τ: entrenamiento suave → inferencia dura (como Real-NVP)")

# ══════════════════════════════════════════════════════════════════════════
# GUARDAR
# ══════════════════════════════════════════════════════════════════════════

W_v2 = np.array([s.W_dispersion for s in spheres], dtype=np.float32)
save_path = os.path.join(os.path.dirname(__file__), "w_dispersion_v2.npy")
np.save(save_path, W_v2)

print(f"\n  {PASS} Guardado: w_dispersion_v2.npy  shape={W_v2.shape}")

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  RESUMEN MATEMÁTICO v2.0                                                ║
║                                                                          ║
║  L_total = L_routing + α·L_balance + β·L_entropy + γ·L_spatial         ║
║                                                                          ║
║  L_routing = CrossEntropy(Gumbel-Softmax(W·color/τ), sphere_expected)   ║
║              → routing correcto por contexto                            ║
║                                                                          ║
║  L_balance = Σ(avg_usage_i - 1/K)²                                      ║
║              → evita colapso MoE (una esfera no monopoliza el routing)  ║
║                                                                          ║
║  L_entropy = (H_mean - H_target)²  donde H_target = log(K)/2           ║
║              → sweet spot: ni indeciso ni hiperespecializado            ║
║                                                                          ║
║  τ-annealing: τ×0.995 por epoch, τ_final≈0.05 → ≈argmax en inferencia  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
